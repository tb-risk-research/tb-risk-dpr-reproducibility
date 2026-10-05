"""Bounded post-hoc reviewer sensitivity; original results are read-only."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,csv,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(r'D:\tb_risk_DPR')
OUT=ROOT/'data/processed/dpr_major_review_v011_20261003'
sys.path.insert(0,str(ROOT/'data'));sys.path.insert(0,str(ROOT/'experiments/dpr'))
import run_same_information_baseline_v1 as si
import run_sop_unified_ci_leakfree as lf
import run_sop_robustness_battery as bat
import run_sop_deepening as dep
SCENARIOS=['age_ascending','age_descending','baseline_score_descending_nested']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');t.replace(p)
def orders(df,values,descending):
 order=np.zeros(len(df),int)
 for _,ix in df.groupby('record_id',sort=False).indices.items():
  ix=np.asarray(ix);ordered=ix[np.argsort(-values[ix] if descending else values[ix],kind='stable')]
  order[ordered]=np.arange(len(ix))
 return pd.Series(order,index=df.index)
def one(scenario,seed):
 start=time.time();df,y,g,bmi,raw=si.load();z=np.load(si.OUT/f'seed_{seed:02d}.npz',allow_pickle=True)
 assert np.array_equal(z['y'],y) and np.array_equal(z['g'],g)
 # Frozen OOF baseline scores are outcome-blind with respect to each held-out household.
 base=z['scores'][0];lrbase=z['scores'][4]
 values=df.contact_age.to_numpy(float) if scenario.startswith('age_') else base
 assert np.isfinite(values).all()
 pf=lf.prior_features(df,orders(df,values,scenario!='age_ascending'))
 pn=pf.prior_n.to_numpy(float);pp=pf.prior_pos.to_numpy(float)
 assert (pp<=pn).all()
 d=df.copy();d['bmi_h']=bmi;d=pd.concat([d.reset_index(drop=True),pf],axis=1)
 d['prior_rate']=np.where(pn>0,pp/np.maximum(pn,1),np.nan)
 cols=lf.inf_feature_columns('exposure');folds=lf._group_cv_indices(g,y,n_splits=5,seed=seed)
 scores=np.zeros((5,len(y)));scores[0]=base;scores[3]=lrbase
 params=[]
 for f,(tr,te) in enumerate(folds):
  assert not set(g[tr])&set(g[te])
  if scenario=='baseline_score_descending_nested':
   # Global cross-fitted scores for training rows can depend on outer-test labels.
   innerdata=df.iloc[tr].reset_index(drop=True).copy();innerdata['bmi_h']=bmi[tr]
   innerfolds=lf._group_cv_indices(g[tr],y[tr],n_splits=4,seed=seed)
   innerp=np.zeros(len(tr))
   for Xi,itr,ite in lf._foldclean_matrix(cols,innerfolds,y[tr],innerdata):
    assert not set(g[tr][itr])&set(g[tr][ite])
    mi=bat._make_model('random_forest',seed);mi.set_params(n_jobs=1)
    mi.fit(Xi[itr],y[tr][itr]);innerp[ite]=mi.predict_proba(Xi[ite])[:,1]
   foldvalues=base.copy();foldvalues[tr]=innerp
   pf=lf.prior_features(df,orders(df,foldvalues,True))
   pn=pf.prior_n.to_numpy(float);pp=pf.prior_pos.to_numpy(float)
   d=df.copy();d['bmi_h']=bmi;d=pd.concat([d.reset_index(drop=True),pf],axis=1)
   d['prior_rate']=np.where(pn>0,pp/np.maximum(pn,1),np.nan)
  pi,k=dep.eb_shrinkage_params(y[tr],g[tr]);params.append({'fold':f,'pi':pi,'k':k})
  fd=d.copy();fd['prior_rate_eb']=(pp+k*pi)/(pn+k)
  for a,kind,cs in [(1,'rf',cols+bat.COLS_PRIOR),(2,'lr',cols+['prior_rate_eb']),(4,'lr',cols+bat.COLS_PRIOR)]:
   X,_,_=lf._foldclean_matrix(cs,[(tr,te)],y,fd)[0]
   if kind=='rf':m=bat._make_model('random_forest',seed);m.set_params(n_jobs=1)
   else:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
   m.fit(X[tr],y[tr]);scores[a,te]=m.predict_proba(X[te])[:,1]
 assert np.isfinite(scores).all()
 np.savez_compressed(OUT/'analysis'/f'{scenario}_{seed:02d}.npz',scores=scores,y=y,g=g)
 atomic(OUT/'analysis'/f'{scenario}_{seed:02d}.json',{'scenario':scenario,'seed':seed,'runtime_seconds':time.time()-start,'eb_parameters':params})
 return scenario,seed
def metric(p,y):
 ix=np.argsort(-p,kind='stable')[:int(np.ceil(.1*len(y)))];hits=y[ix].sum()
 return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
         'free_capture_10':float(hits/y.sum()),'descriptive_nns_10':float(len(ix)/hits)}
def finish_orders():
 from run_d1_strengthened_v1 import bootstrap_auc
 _,y,g,_,_=si.load();_,gid=np.unique(g,return_inverse=True);h=gid.max()+1
 draws=np.random.default_rng(91303).multinomial(h,np.full(h,1/h),size=1000)
 names=['rf_base','rf_four','lr_eb','lr_base','lr_four'];result={};records=[]
 for sc in ['random']+SCENARIOS:
  allscores=[];vals=[];boot=[]
  for seed in range(20):
   if sc=='random':
    z=np.load(si.OUT/f'seed_{seed:02d}.npz',allow_pickle=True);scores=z['scores'][[0,1,6,4,5]]
   else:scores=np.load(OUT/'analysis'/f'{sc}_{seed:02d}.npz',allow_pickle=True)['scores']
   allscores.append(scores);vals.append([metric(p,y) for p in scores]);boot.append(bootstrap_auc(scores.T,y,gid,draws))
  b=np.array(boot).mean(axis=0)
  for a,name in enumerate(names):
   v={key:float(np.mean([x[a][key] for x in vals])) for key in vals[0][a]}
   base=0 if name.startswith('rf') else 3
   delta=np.array([x[a]['auroc']-x[base]['auroc'] for x in vals])
   v.update({'scenario':sc,'model':name,'delta_auroc':float(delta.mean()),'conditional_ci95':np.quantile(b[:,a]-b[:,base],[.025,.975]).tolist(),
             'delta_seed_range':[float(delta.min()),float(delta.max())],
             'capture_seed_range':[float(min(x[a]['free_capture_10'] for x in vals)),float(max(x[a]['free_capture_10'] for x in vals))],
             'delta_free_capture_pp':float(100*np.mean([x[a]['free_capture_10']-x[base]['free_capture_10'] for x in vals]))})
   records.append(v)
  result[sc]=records[-5:]
 atomic(OUT/'analysis/order_sensitivity.json',{'status':'complete','results':result,'n':len(y),'events':int(y.sum()),'households':int(h),'cv_seeds':list(range(20)),
  'interval':'1000 paired household resamples of fixed OOF predictions; same draws across orders/seeds/arms',
  'limits':['Post-hoc sensitivity, specified before this new run, not preregistered.','No timestamps or real return process.','Baseline score order varies with the frozen cross-validation seed.','No between-scenario hypothesis tests or selection of best order.','Free capture excludes acquisition of earlier results.'],
  'script_sha256':sha(__file__),'raw_sha256':sha(ROOT/'data/raw/homeacf_tstsa.rda')})
 with (OUT/'analysis/order_sensitivity.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def charged(p,y,g,pi,total,q,k,w):
 n=len(y);m=int(np.ceil(q*n));nt=int(np.ceil(total*n));order=np.argsort(-p,kind='stable');first=order[:m]
 count=np.bincount(g[first],minlength=int(g.max())+1);pos=np.bincount(g[first],weights=y[first],minlength=len(count))
 update=dep.dep_scores(p,count[g],pos[g],pi,k,w)
 remaining=np.ones(n,bool);remaining[first]=False;rem=np.flatnonzero(remaining)
 second=rem[np.argsort(-update[rem],kind='stable')[:nt-m]]
 assert len(second)+len(first)==nt and not np.intersect1d(first,second).size
 hits=y[first].sum()+y[second].sum();static=y[order[:nt]].sum()
 return float(hits/y.sum()),float(static/y.sum())
def budget():
 z=np.load(ROOT/'data/processed/homeacf_dpr_budget_audit_v2.npz');old=json.loads((ROOT/'data/processed/homeacf_dpr_budget_audit_v2.json').read_text())
 y,g=z['y'],z['g'];assert len(y)==2725 and y.sum()==359 and len(np.unique(g))==877
 rows=[];waves=[]
 for model in ['RF','LR']:
  ps=z[model.lower()];pis=z['pi']
  for total in [.1,.2,.3]:
   gate=np.array([charged(p,y,g,pi,total,.05,2,7) for p,pi in zip(ps,pis)])
   assert np.allclose(gate.mean(axis=0),[old['results'][f'{model}_{total}']['cascade_capture'],old['results'][f'{model}_{total}']['static_capture']],atol=1e-12)
   configs=[('w_q_surface',q,2,w) for q in [.01,.025,.05,.075] for w in [1,3,5,7,9,12]]
   configs += [('k_sensitivity',.05,k,7) for k in [.5,1,2,4,8]]
   for family,q,k,w in configs:
    vs=np.array([charged(p,y,g,pi,total,q,k,w) for p,pi in zip(ps,pis)])
    rows.append({'model':model,'total_budget':total,'family':family,'q':q,'k':k,'w':w,'first_tests':int(np.ceil(q*len(y))),
     'second_tests':int(np.ceil(total*len(y)))-int(np.ceil(q*len(y))),'cascade_capture':float(vs[:,0].mean()),'static_capture':float(vs[:,1].mean()),
     'delta_pp':float(100*(vs[:,0]-vs[:,1]).mean()),'seed_min_pp':float(100*(vs[:,0]-vs[:,1]).min()),'seed_max_pp':float(100*(vs[:,0]-vs[:,1]).max())})
   waves.append({'model':model,'total_budget':total,'first':int(np.ceil(.05*len(y))),'second':int(np.ceil(total*len(y)))-int(np.ceil(.05*len(y)))})
 oldarchive=ROOT/'data/processed/temporal_deployment_20260825.json';a=json.loads(oldarchive.read_text(encoding='utf8'))
 atomic(OUT/'analysis/charged_parameter_sensitivity.json',{'status':'complete','records':rows,'waves':waves,'cache_gate':'all RF/LR original budget points reproduced to 1e-12',
  'parameter_selection_evidence':{'archive':str(oldarchive),'sha256':sha(oldarchive),'calibration_seed':a['calibration']['seed'],'best_w':a['calibration']['best_w'],'best_k':a['calibration']['best_k'],
   'same_cohort_selection':True,'q_origin':'5% was a retrospective scenario; independent capacity rationale not documented'},
  'limits':['No new fitting or optimization; all specified settings reported.','Grid does not remove optimism from historical same-cohort selection.','Descriptive sensitivity of fixed models; no new confidence intervals or policy recommendation.','k and w jointly tuned historically; current k sensitivity holds w/q fixed.','Unknown result-return timing; unit test costs assumed equal.'],
  'script_sha256':sha(__file__),'cache_sha256':sha(ROOT/'data/processed/homeacf_dpr_budget_audit_v2.npz')})
 with (OUT/'analysis/charged_parameter_sensitivity.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis').mkdir(exist_ok=True);start=time.time()
 protocol={'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'status':'frozen before new sensitivity outputs','post_hoc':True,
 'question':'Sensitivity to assumed ordering and retrospectively selected charged update parameters; no causal attribution/policy selection',
 'orders':SCENARIOS,'tie_break':'source row position, outcome-independent','models':'frozen RF four features, LR single fold-specific EB rate and LR four features; identical group folds; cached baselines',
 'seed_count':20,'conditional_bootstrap':1000,'budget_w':[1,3,5,7,9,12],'q':[.01,.025,.05,.075],'k':[.5,1,2,4,8],
 'existing_data_only':True,'no_best_configuration_selection':True,'script_sha256':sha(__file__)}
 if not (OUT/'analysis/protocol.json').exists():atomic(OUT/'analysis/protocol.json',protocol)
 budget();atomic(OUT/'analysis/progress.json',{'status':'running','completed':0,'target':60,'pid':os.getpid()})
 missing=[(sc,s) for sc in SCENARIOS for s in range(20) if not (OUT/'analysis'/f'{sc}_{s:02d}.json').exists()]
 done=60-len(missing)
 with ProcessPoolExecutor(max_workers=4) as pool:
  jobs=[pool.submit(one,*args) for args in missing]
  for fut in as_completed(jobs):
   sc,s=fut.result();done+=1
   atomic(OUT/'analysis/progress.json',{'status':'running','completed':done,'target':60,'elapsed_seconds':time.time()-start,'pid':os.getpid()})
   print(sc,s,done,'/60',flush=True)
 atomic(OUT/'analysis/progress.json',{'status':'summarizing','completed':60,'target':60})
 finish_orders();atomic(OUT/'analysis/progress.json',{'status':'complete','completed':60,'target':60,'elapsed_seconds':time.time()-start})
 print('Reviewer sensitivities complete.',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:atomic(OUT/'analysis/progress.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc()});raise
