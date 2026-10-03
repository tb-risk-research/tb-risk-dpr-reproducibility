import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'data'))
import run_sop_unified_ci_leakfree as lf
import run_sop_robustness_battery as bat
OUT=ROOT/'data/processed/homeacf_dpr_availability_sensitivity_v1.json'
def main():
 t=time.time();df=lf.load_homeacf_contacts();y=df.tst_pos10.to_numpy(dtype=int);g=df.record_id.to_numpy()
 bmi=bat.load_raw_bmi(df); scores=[];rows=[]
 for s in range(20):
  folds=lf._group_cv_indices(g,y,5,s);pf=lf.prior_features(df,lf.assign_screening_order(df,s,'random'))
  data=df.copy();data['bmi_h']=bmi;data=pd.concat([data.reset_index(drop=True),pf],axis=1)
  data['prior_rate']=np.where(pf.prior_n>0,pf.prior_pos/np.maximum(pf.prior_n,1),np.nan)
  cols=[c for c in lf.inf_feature_columns('exposure') if c!='idx_dead']
  assert len(cols)==len(lf.inf_feature_columns('exposure'))-1
  p=lf.rf_oof_lf(cols,folds,y,data,s);q=lf.rf_oof_lf(cols+bat.COLS_PRIOR,folds,y,data,s)
  scores.append((p,q));a=roc_auc_score(y,p);b=roc_auc_score(y,q)
  rows.append({'seed':s,'baseline_auc':a,'sop_auc':b,'delta_auc':b-a}); print('no-death seed',s+1,'/20',flush=True)
 np.savez_compressed(OUT.with_suffix('.npz'),scores=np.asarray(scores),y=y,g=g)
 blocks=[np.flatnonzero(g==h) for h in np.unique(g)];rng=np.random.default_rng(91205);boot=[]
 for b in range(2000):
  ix=np.concatenate([blocks[k] for k in rng.integers(0,len(blocks),len(blocks))])
  boot.append(float(np.mean([roc_auc_score(y[ix],q[ix])-roc_auc_score(y[ix],p[ix]) for p,q in scores])))
  if (b+1)%200==0:print('no-death bootstrap',b+1,'/2000',flush=True)
 result={'status':'complete','n':len(y),'events':int(y.sum()),'households':len(blocks),'rows':rows,
  'summary':{'baseline_mean_auc':float(np.mean([r['baseline_auc'] for r in rows])),
   'sop_mean_auc':float(np.mean([r['sop_auc'] for r in rows])),
   'delta_mean':float(np.mean([r['delta_auc'] for r in rows])),
   'conditional_ci95':np.quantile(boot,[.025,.975]).tolist()},
  'design':'Exclude idx_dead in BOTH arms; frozen RF, original grouped folds and household order, seeds 0-19; no retuning.',
  'bootstrap':{'replicates':2000,'seed':91205,'type':'conditional fixed predictions'},
  'limits':['No full-refit uncertainty.','Does not establish timing of remaining baseline covariates.',
    'Post-hoc availability sensitivity, not a replacement chosen for better performance.'],
  'runtime_seconds':time.time()-t,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 OUT.write_text(json.dumps(result,indent=2),encoding='utf-8');print('COMPLETE',result['summary'],flush=True)
if __name__=='__main__':main()
