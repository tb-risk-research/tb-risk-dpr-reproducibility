import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,hashlib,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'data'))
import run_sop_unified_ci_leakfree as lf
import run_sop_robustness_battery as bat
import run_homeacf_tst_missingness_ipw as e1
OUT=ROOT/'data/processed/dpr_feature_audit_sensitivity_v1';OUT.mkdir(exist_ok=True)
DROP=['idx_dead','hiv_pos_h','hiv_unknown_h','idx_smear_pos','idx_smear_known','idx_xpert_pos','idx_hiv_pos']
EXTRA=['diabetes_missing','airspace_missing','timespent_missing','relationship_missing','index_hiv_unknown','index_xpert_not_done']
def atomic(p,obj):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');t.replace(p)
def one(task):
 scenario,seed=task;start=time.time();df=lf.load_homeacf_contacts();raw=e1.raw_data();raw=raw.loc[raw.tst_observed.eq(1)].reset_index(drop=True)
 assert np.array_equal(raw.contact_id.to_numpy(),df.contact_id.to_numpy())
 y=df.tst_pos10.to_numpy(int);g=df.record_id.to_numpy();cols=lf.inf_feature_columns('exposure')
 pf=lf.prior_features(df,lf.assign_screening_order(df,seed=seed,mode='random'))
 data=df.copy();data['bmi_h']=bat.load_raw_bmi(df);data=pd.concat([data.reset_index(drop=True),pf],axis=1)
 data['prior_rate']=np.where(pf.prior_n>0,pf.prior_pos/np.maximum(pf.prior_n,1),np.nan)
 if scenario=='conservative':cols=[c for c in cols if c not in DROP]
 elif scenario=='coding':
  data['rel_parent']=raw.relationship_h.eq('Parent/Parent-in-law').astype(float)
  data['hiv_unknown_h']=(raw.hivfinal_h.eq('c) HIV unknown')|raw.hivfinal_h.isna()).astype(float)
  for new,source in zip(EXTRA[:4],['diabetes_h','airspace_h','timespent_h','relationship_h']):data[new]=raw[source].isna().astype(float)
  data['index_hiv_unknown']=raw.hiv_i.eq('c) HIV-unknown').astype(float)
  data['index_xpert_not_done']=raw.xpert_i.eq('c) Xpert not done').astype(float)
  cols=cols+EXTRA
 else:raise ValueError(scenario)
 folds=lf._group_cv_indices(g,y,n_splits=5,seed=seed)
 for tr,te in folds:assert not set(g[tr])&set(g[te])
 scores=np.array([lf.rf_oof_lf(cs,folds,y,data,seed) for cs in [cols,cols+bat.COLS_PRIOR]])
 np.savez_compressed(OUT/f'{scenario}_{seed:02d}.npz',scores=scores,y=y,g=g)
 metrics=[{'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p))} for p in scores]
 atomic(OUT/f'{scenario}_{seed:02d}.json',{'scenario':scenario,'seed':seed,'columns_base':cols,'metrics':metrics,'seconds':time.time()-start})
 return task
def finish():
 from run_d1_strengthened_v1 import bootstrap_auc
 allrows={};results={}
 for scenario in ['conservative','coding']:
  rows=[json.loads((OUT/f'{scenario}_{s:02d}.json').read_text()) for s in range(20)]
  zs=[np.load(OUT/f'{scenario}_{s:02d}.npz',allow_pickle=True) for s in range(20)]
  y=zs[0]['y'];g=zs[0]['g'];_,gid=np.unique(g,return_inverse=True);h=gid.max()+1
  for z in zs:assert np.array_equal(y,z['y']) and np.array_equal(g,z['g'])
  draws=np.random.default_rng(91208).multinomial(h,np.full(h,1/h),size=2000)
  boots=np.array([bootstrap_auc(z['scores'].T,y,gid,draws) for z in zs]).mean(axis=0)
  delta=[r['metrics'][1]['auroc']-r['metrics'][0]['auroc'] for r in rows]
  results[scenario]={'base_auroc':float(np.mean([r['metrics'][0]['auroc'] for r in rows])),
   'sop_auroc':float(np.mean([r['metrics'][1]['auroc'] for r in rows])),
   'base_auprc':float(np.mean([r['metrics'][0]['auprc'] for r in rows])),
   'sop_auprc':float(np.mean([r['metrics'][1]['auprc'] for r in rows])),
   'delta_auroc':float(np.mean(delta)),'conditional_ci95':np.quantile(boots[:,1]-boots[:,0],[.025,.975]).tolist(),
   'positive_seeds':int(np.sum(np.array(delta)>0)),'seed_range':[float(min(delta)),float(max(delta))],
   'columns_base':rows[0]['columns_base']}
  allrows[scenario]=rows
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 atomic(OUT/'result.json',{'status':'complete','n':len(y),'events':int(y.sum()),'households':int(h),
  'seeds':20,'bootstrap':2000,'bootstrap_seed':91208,'results':results,'records':allrows,
  'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'docs/dpr_feature_audit_protocol_v1.md'),
  'raw_sha256':sha(ROOT/'data/raw/homeacf_tstsa.rda'),
  'limits':['Post-hoc sensitivity, not replacement primary analysis.','No real information-availability timestamps.','Conditional fixed-prediction intervals, not full-refit.','Coding and conservative scenarios are separate; not jointly corrected.']})
 print(json.dumps(results),flush=True)
def main():
 start=time.time();tasks=[(c,s) for c in ['conservative','coding'] for s in range(20)]
 tasks=[t for t in tasks if not (OUT/f'{t[0]}_{t[1]:02d}.json').exists()];done=40-len(tasks)
 atomic(OUT/'progress.json',{'status':'running','completed':done,'target':40,'pid':os.getpid()})
 try:
  with ProcessPoolExecutor(max_workers=4) as pool:
   for f in as_completed([pool.submit(one,t) for t in tasks]):
    t=f.result();done+=1;atomic(OUT/'progress.json',{'status':'running','completed':done,'target':40,'seconds':time.time()-start,'pid':os.getpid()});print('completed',t,done,'/40',flush=True)
  atomic(OUT/'progress.json',{'status':'summarizing','completed':40,'target':40});finish()
  atomic(OUT/'progress.json',{'status':'complete','completed':40,'target':40,'seconds':time.time()-start,'error':None})
 except Exception as exc:
  atomic(OUT/'progress.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
