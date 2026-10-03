import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,traceback,hashlib,platform
from pathlib import Path
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['check','validate'],default='validate');ARGS=parser.parse_args()
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/processed/validation';OUT.mkdir(parents=True,exist_ok=True)
REF=ROOT;EVIDENCE=ROOT/'experiments/dpr/evidence/results'
GOLD=json.loads((ROOT/'experiments/dpr/evidence/validation_reference.json').read_text(encoding='utf-8'))
sys.path[:0]=[str(ROOT.parent),str(ROOT/'data'),str(ROOT/'experiments/dpr')]
import numpy as np,pandas as pd,sklearn
from sklearn.metrics import roc_auc_score,average_precision_score
import run_sop_unified_ci_leakfree as lf
import run_sop_robustness_battery as bat
import run_sop_deepening as dep
original_factory=bat._make_model
def serial_factory(*a,**k):
 m=original_factory(*a,**k)
 if 'n_jobs' in m.get_params():m.set_params(n_jobs=1)
 return m
bat._make_model=serial_factory
assert Path(lf.__file__).resolve().is_relative_to(ROOT)
assert sys.prefix != sys.base_prefix, 'Use a separate venv for delivery verification.'
assert Path(bat.__file__).resolve().is_relative_to(ROOT)
df=lf.load_homeacf_contacts();y=df.tst_pos10.to_numpy(dtype=int);g=df.record_id.to_numpy();bmi=bat.load_raw_bmi(df)
assert (len(y),int(y.sum()),len(np.unique(g)))==(2725,359,877)
def data_for(seed):
 pf=lf.prior_features(df,lf.assign_screening_order(df,seed=seed,mode='random'))
 data=df.copy();data['bmi_h']=bmi;data=pd.concat([data.reset_index(drop=True),pf],axis=1)
 data['prior_rate']=np.where(pf.prior_n>0,pf.prior_pos/np.maximum(pf.prior_n,1),np.nan)
 folds=lf._group_cv_indices(g,y,n_splits=5,seed=seed)
 for tr,te in folds:assert not set(g[tr])&set(g[te])
 return data,pf,folds
def close(actual,expected,label,tol=1e-10):
 error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
 if error>tol:raise AssertionError(f'{label}: max_error={error}, tolerance={tol}')
 return error
def primary():
 rows=[]
 for seed in ([0] if ARGS.scope=='check' else [0,19]):
  data,pf,folds=data_for(seed);cols=lf.inf_feature_columns('exposure')
  scores=np.asarray([lf.rf_oof_lf(cs,folds,y,data,seed) for cs in [cols,cols+bat.COLS_PRIOR]])
  np.savez_compressed(OUT/f'fresh_primary_seed_{seed}.npz',scores=scores)
  expected=GOLD['primary'][str(seed)]
  errors=[close([roc_auc_score(y,p) for p in scores],expected['auroc'],'primary AUROC'),close([average_precision_score(y,p) for p in scores],expected['auprc'],'primary AUPRC'),close(scores.sum(axis=1),expected['score_sum'],'score sums',1e-7),close((scores*scores).sum(axis=1),expected['score_sumsq'],'score squared sums',1e-7)]
  error=max(errors)
  rows.append({'seed':seed,'max_aggregate_error':error,'auroc':[roc_auc_score(y,p) for p in scores],'auprc':[average_precision_score(y,p) for p in scores]})
  print('primary seed',seed,'reproduced',flush=True)
 return rows
def same_information():
 import run_same_information_baseline_v1 as si
 data,pf,folds=data_for(0);cols=lf.inf_feature_columns('exposure');pn=pf.prior_n.to_numpy(float);pp=pf.prior_pos.to_numpy(float)
 scores=np.zeros((7,len(y)));scores[:2]=np.load(OUT/'fresh_primary_seed_0.npz')['scores']
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 for f,(tr,te) in enumerate(folds):
  pi,k=dep.eb_shrinkage_params(y[tr],g[tr]);d=data.copy();d['prior_rate_eb']=(pp+k*pi)/(pn+k)
  for a,kind,cs in [(2,'rf',cols+['prior_rate_eb']),(3,'rf',cols+['prior_rate_eb','prior_n','prior_screened']),(4,'lr',cols),(5,'lr',cols+bat.COLS_PRIOR),(6,'lr',cols+['prior_rate_eb'])]:
   X=lf._foldclean_matrix(cs,[(tr,te)],y,d)[0][0]
   m=serial_factory('random_forest',0) if kind=='rf' else make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
   m.fit(X[tr],y[tr]);scores[a,te]=m.predict_proba(X[te])[:,1]
 reference=GOLD['same_information']
 errors=[close([roc_auc_score(y,p) for p in scores],reference['auroc'],'same-information AUROC'),close([average_precision_score(y,p) for p in scores],reference['auprc'],'same-information AUPRC'),close(scores.sum(axis=1),reference['score_sum'],'same-information score sums',1e-7),close((scores*scores).sum(axis=1),reference['score_sumsq'],'same-information score squared sums',1e-7)]
 return {'seed':0,'arms':si.NAMES,'max_aggregate_error':max(errors)}
def nested():
 from sklearn.linear_model import LogisticRegression
 from scipy.special import logit
 data,pf,outer=data_for(0);raw=np.zeros((2,len(y)));cal=np.zeros_like(raw)
 for f,(tr,te) in enumerate(outer):
  inner=lf._group_cv_indices(g[tr],y[tr],n_splits=4,seed=100000+f)
  for itr,ite in inner:assert not set(g[tr][itr])&set(g[tr][ite])
  for arm in range(2):
   cols=lf.inf_feature_columns('exposure')+(bat.COLS_PRIOR if arm else [])
   ip=lf.rf_oof_lf(cols,inner,y[tr],data.iloc[tr],0)
   calibrator=LogisticRegression(C=1e6,max_iter=2000).fit(logit(np.clip(ip,1e-6,1-1e-6))[:,None],y[tr])
   X=lf._foldclean_matrix(cols,[(tr,te)],y,data)[0][0];m=serial_factory('random_forest',0);m.fit(X[tr],y[tr]);p=m.predict_proba(X[te])[:,1]
   raw[arm,te]=p;cal[arm,te]=calibrator.predict_proba(logit(np.clip(p,1e-6,1-1e-6))[:,None])[:,1]
  print('nested calibration outer',f+1,'/5',flush=True)
 reference=GOLD['nested']
 errors=[close(np.mean((cal-y)**2,axis=1),reference['brier'],'nested Brier'),close(cal.sum(axis=1),reference['score_sum'],'nested score sums',1e-7),close((cal*cal).sum(axis=1),reference['score_sumsq'],'nested score squared sums',1e-7)]
 return {'seed':0,'outer_folds':5,'inner_folds':4,'max_aggregate_error':max(errors),'brier':np.mean((cal-y)**2,axis=1).tolist()}
def budgets():
 import run_budget_audit_v2 as budget
 rows=[];_,gid=np.unique(g,return_inverse=True)
 for seed in [0,19]:
  scores=np.load(OUT/f'fresh_primary_seed_{seed}.npz')['scores'][0];pi=np.zeros(len(y));folds=lf._group_cv_indices(g,y,n_splits=5,seed=seed)
  for tr,te in folds:pi[te]=dep.eb_shrinkage_params(y[tr],g[tr])[0]
  for K in [.1,.2,.3]:
   actual=budget.metrics(scores,y,gid,pi,K);expected=next(r['values'] for r in GOLD['budgets'] if r['seed']==seed and r['budget']==K)
   rows.append({'seed':seed,'budget':K,'max_error':close(actual,expected,'charged budget'),'cascade_capture':float(actual[0]),'static_capture':float(actual[1])})
 return rows
def fullrefit():
 import run_sop_fullrefit_bootstrap as a1
 base=df.copy();base['_raw_bmi']=bmi;base['_cv_group']=g
 value=a1.once(a1.boot(base,np.random.default_rng(91001)))[0]
 ref=json.loads((EVIDENCE/'homeacf_dpr_a1_fullrefit_20261001.json').read_text())['delta_auroc']['distribution'][0]
 import run_homeacf_e1_fullrefit as e1
 raw=e1.e.raw_data();raw['_cv_group']=raw.record_id;ids=raw.record_id.unique();blocks={h:raw[raw.record_id==h].copy() for h in ids};parts=[]
 for j,h in enumerate(np.random.default_rng(91203).choice(ids,len(ids),replace=True)):
  part=blocks[h].copy();part['record_id']=f'copy_{j:04d}';parts.append(part)
 ev=e1.once(pd.concat(parts,ignore_index=True));er=json.loads((EVIDENCE/'homeacf_dpr_e1_fullrefit_v2.json').read_text())['replicates'][0]
 return {'unweighted_replicate':0,'unweighted_delta':float(value),'unweighted_error':close(value,ref,'unweighted full refit'),'ipw_replicate':0,'ipw_actual':ev,'ipw_reference':er,'ipw_delta_error':close(ev['delta'],er['delta'],'IPW delta',1e-9),'ipw_ess_error':close(ev['ess'],er['ess'],'IPW effective sample size',1e-6),'ipw_n_error':close(ev['n'],er['n'],'IPW sample size',0),'tolerance_note':'AUROC absolute tolerance 1e-9; ESS absolute tolerance 1e-6. ESS is a descriptive sample-size statistic, not AUROC; all actual differences are retained.'}
def simulation():
 import run_d1_strengthened_v1 as sim
 refs={}
 for line in (EVIDENCE/'dpr_d1_strengthened_v1/datasets.jsonl').read_text().splitlines():
  r=json.loads(line)
  if r['kind']=='lr' and r['rep']==0 and r['config']['config_id'] in [0,2,6,8]:refs[r['config']['config_id']]=r
 rows=[]
 for i in [0,2,6,8]:
  a=sim.run_one((sim.configs()[i],0,'lr',False));b=refs[i];errors=[]
  for k in sim.CONTRASTS:errors.append(close([a['contrasts'][k]['mean']]+a['contrasts'][k]['conditional_ci95'],[b['contrasts'][k]['mean']]+b['contrasts'][k]['conditional_ci95'],f'D1 cell {i} {k}'))
  for k in sim.ORDER_NAMES:errors.append(close(a['equivalence_ci96_667'][k],b['equivalence_ci96_667'][k],f'D1 equivalence {i} {k}'))
  assert a['diagnostic_output']==b['diagnostic_output'] and a['margin_sensitivity']==b['margin_sensitivity']
  rows.append({'config_id':i,'rep':0,'output':a['diagnostic_output'],'max_error':max(errors)})
  print('simulation cell',i,'reproduced',flush=True)
 return rows
if __name__=='__main__':
 start=time.time();results=[]
 def save(state):
  (OUT/'progress.json').write_text(json.dumps({'status':state,'completed':len(results),'target':1 if ARGS.scope=='check' else 6,'elapsed_seconds':time.time()-start,'checks':results},indent=2),encoding='utf-8')
 save('running')
 checks=[('primary',primary)] if ARGS.scope=='check' else [('primary',primary),('same_information',same_information),('nested_calibration',nested),('charged_budget',budgets),('full_refit',fullrefit),('known_truth_simulation',simulation)]
 for name,func in checks:
  t=time.time();print('START',name,flush=True)
  try:result=func();results.append({'check':name,'status':'passed','results':result,'seconds':time.time()-t})
  except Exception as ex:results.append({'check':name,'status':'failed','error':repr(ex),'traceback':traceback.format_exc(),'seconds':time.time()-t});print(traceback.format_exc(),flush=True)
  save('running');print('FINISH',name,results[-1]['status'],flush=True)
 status='passed' if all(r['status']=='passed' for r in results) else 'issues_found'
 save(status)
 (OUT/'result.json').write_text(json.dumps({'status':status,'checks':results,'python':platform.python_version(),'sklearn':sklearn.__version__,'elapsed_seconds':time.time()-start,'source_imports':{'lf':lf.__file__,'bat':bat.__file__},'limits':['Offline copied dependencies, not a fresh network installation or independent external lab.','Selected seeds and replicates only, not a rerun of 20 seeds/300 bootstraps/3700 datasets.','Only aggregate reference metrics are shipped; no archived participant predictions used for fitting.','Same household and unavailable timestamp/causal interpretation limitations remain.']},indent=2),encoding='utf-8')
 print('DONE',status,flush=True)
 if status!='passed':sys.exit(1)
