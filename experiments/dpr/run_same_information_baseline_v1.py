import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,hashlib,platform,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'data'))
import run_sop_unified_ci_leakfree as lf
import run_sop_robustness_battery as bat
import run_sop_deepening as dep
OUT=ROOT/'data/processed/dpr_same_information_baseline_v1'; OUT.mkdir(exist_ok=True)
NAMES=['rf_base','rf_sop','rf_eb','rf_eb_counts','lr_base','lr_sop','lr_eb']
def atomic(p,obj):
 t=p.with_suffix('.tmp'); t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8'); t.replace(p)
def load():
 df=lf.load_homeacf_contacts();y=df.tst_pos10.to_numpy(dtype=int);g=df.record_id.to_numpy();bmi=bat.load_raw_bmi(df)
 z=np.load(ROOT/'data/processed/homeacf_dpr_nested_calibration_v1.npz',allow_pickle=True)
 assert np.array_equal(y,z['y']) and np.array_equal(g,z['g']) and z['raw'].shape==(20,2,len(y))
 return df,y,g,bmi,z['raw']
def one(seed):
 start=time.time(); df,y,g,bmi,raw=load(); cols=lf.inf_feature_columns('exposure')
 pf=lf.prior_features(df,lf.assign_screening_order(df,seed=seed,mode='random'))
 data=df.copy();data['bmi_h']=bmi;data=pd.concat([data.reset_index(drop=True),pf],axis=1)
 pn=pf.prior_n.to_numpy(float);pp=pf.prior_pos.to_numpy(float)
 data['prior_rate']=np.where(pn>0,pp/np.maximum(pn,1),np.nan)
 folds=lf._group_cv_indices(g,y,n_splits=5,seed=seed)
 scores=np.zeros((7,len(y)));scores[:2]=raw[seed];params=[]
 if seed==0:
  for j,cs in enumerate([cols,cols+bat.COLS_PRIOR]):
   check=lf.rf_oof_lf(cs,folds,y,data,seed)
   assert np.max(np.abs(check-raw[seed,j]))<1e-12
 for f,(tr,te) in enumerate(folds):
  assert not set(g[tr])&set(g[te])
  pi,k=dep.eb_shrinkage_params(y[tr],g[tr]);params.append({'fold':f,'pi':pi,'k':k})
  fold_data=data.copy();fold_data['prior_rate_eb']=(pp+k*pi)/(pn+k)
  specs=[(2,'rf',cols+['prior_rate_eb']),(3,'rf',cols+['prior_rate_eb','prior_n','prior_screened']),
         (4,'lr',cols),(5,'lr',cols+bat.COLS_PRIOR),(6,'lr',cols+['prior_rate_eb'])]
  for a,kind,cs in specs:
   X,_,_=lf._foldclean_matrix(cs,[(tr,te)],y,fold_data)[0]
   if kind=='rf':m=bat._make_model('random_forest',seed);m.set_params(n_jobs=1)
   else:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
   m.fit(X[tr],y[tr]); scores[a,te]=m.predict_proba(X[te])[:,1]
 assert np.isfinite(scores).all()
 metrics={}
 for a,name in enumerate(NAMES):
  ix=np.argsort(-scores[a],kind='stable')[:int(np.ceil(.1*len(y)))]
  metrics[name]={'auroc':float(roc_auc_score(y,scores[a])),'auprc':float(average_precision_score(y,scores[a])),
                 'capture_10_free':float(y[ix].sum()/y.sum())}
 np.savez_compressed(OUT/f'seed_{seed:02d}.npz',scores=scores,y=y,g=g)
 atomic(OUT/f'seed_{seed:02d}.json',{'seed':seed,'metrics':metrics,'eb_params':params,'runtime':time.time()-start,'cache_reproduction_gate':seed==0})
 return seed
def finish():
 _,y,g,_,_=load(); records=[json.loads((OUT/f'seed_{s:02d}.json').read_text()) for s in range(20)]
 scores=np.array([np.load(OUT/f'seed_{s:02d}.npz',allow_pickle=True)['scores'] for s in range(20)])
 metrics={n:{m:float(np.mean([r['metrics'][n][m] for r in records])) for m in ['auroc','auprc','capture_10_free']} for n in NAMES}
 pairs=[('rf_sop_vs_eb_counts',1,3),('rf_sop_vs_eb',1,2),('lr_sop_vs_eb',5,6),('rf_sop_vs_base',1,0),('rf_eb_vs_base',2,0),('rf_eb_counts_vs_base',3,0),('lr_sop_vs_base',5,4),('lr_eb_vs_base',6,4)]
 _,gid=np.unique(g,return_inverse=True);h=gid.max()+1
 from run_d1_strengthened_v1 import bootstrap_auc
 rng=np.random.default_rng(91207);boot=[]
 for s in range(20):
  # The same resampled households are used for all seeds and arms.
  draw=np.random.default_rng(91207).multinomial(h,np.full(h,1/h),size=2000)
  boot.append(bootstrap_auc(scores[s].T,y,gid,draw))
 boot=np.array(boot).mean(axis=0);comps={}
 for name,a,b in pairs:
  delta=np.array([r['metrics'][NAMES[a]]['auroc']-r['metrics'][NAMES[b]]['auroc'] for r in records])
  comps[name]={'delta_auroc':float(delta.mean()),'conditional_ci95':np.quantile(boot[:,a]-boot[:,b],[.025,.975]).tolist(),
   'positive_seeds':int((delta>0).sum()),'seed_range':[float(delta.min()),float(delta.max())]}
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 result={'status':'complete','n':len(y),'events':int(y.sum()),'households':int(h),'seeds':20,'bootstrap':2000,
  'metrics':metrics,'comparisons':comps,'records':records,'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__},
  'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'docs/dpr_same_information_baseline_protocol_v1.md'),
  'raw_sha256':sha(ROOT/'data/raw/homeacf_tstsa.rda'),
  'limits':['Conditional fixed-prediction intervals; no full-refit EB comparison.','Capture excludes prior-result acquisition costs.','No equivalence claim from a non-significant difference.','Beta-binomial shrinkage proxy is not a complete mixed-effects model comparison.']}
 atomic(OUT/'result.json',result)
 print(json.dumps({'metrics':metrics,'comparisons':comps}),flush=True)
def main():
 start=time.time();atomic(OUT/'progress.json',{'status':'running','completed':0,'target':20,'pid':os.getpid()})
 try:
  missing=[s for s in range(20) if not (OUT/f'seed_{s:02d}.json').exists()]
  # Gate first, then independent seeds in four workers.
  if 0 in missing:one(0);missing.remove(0)
  done=20-len(missing)
  with ProcessPoolExecutor(max_workers=4) as pool:
   jobs={pool.submit(one,s):s for s in missing}
   for future in as_completed(jobs):
    seed=future.result();done+=1
    atomic(OUT/'progress.json',{'status':'running','completed':done,'target':20,'elapsed_seconds':time.time()-start,'pid':os.getpid()})
    print('completed seed',seed,'total',done,'/20',flush=True)
  atomic(OUT/'progress.json',{'status':'summarizing','completed':20,'target':20,'pid':os.getpid()})
  finish();atomic(OUT/'progress.json',{'status':'complete','completed':20,'target':20,'elapsed_seconds':time.time()-start,'error':None})
 except Exception as e:
  atomic(OUT/'progress.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
