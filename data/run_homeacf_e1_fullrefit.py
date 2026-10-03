"""E1 v2: resample all raw households and refit observation and prediction models."""
import os, sys, json, time, hashlib, platform
import numpy as np
import pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import run_homeacf_tst_missingness_ipw as e
from tb_risk.validation.real_data_infection import _derive_features, feature_columns
from tb_risk.validation.household_temporal import assign_screening_order, prior_features
from tb_risk.validation.real_data_pi import _group_cv_indices
SMOKE='smoke' in sys.argv; N=3 if SMOKE else 300
OUT=os.path.join(HERE,'processed','homeacf_dpr_e1_fullrefit_v2%s.json'%('_smoke' if SMOKE else ''))
def once(raw):
 p=e.observation_oof(raw); keep=raw.tst_observed.eq(1).to_numpy(); pk=p[keep]
 assert np.isfinite(pk).all() and (pk>0).all()
 w=1/pk; lo,hi=np.quantile(w,[.01,.99]); w=np.clip(w,lo,hi); w/=w.mean()
 df=_derive_features(raw); y=df.tst_pos10.astype(int).to_numpy()
 assert np.array_equal(df.contact_id.to_numpy(),raw.loc[keep,'contact_id'].to_numpy())
 g=df._cv_group.to_numpy(); folds=_group_cv_indices(g,y,n_splits=5,seed=0)
 pf=prior_features(df,assign_screening_order(df,seed=0,mode='random'))
 matrix=pd.concat([df.reset_index(drop=True),pf],axis=1)
 matrix['bmi_h']=pd.to_numeric(raw.loc[keep,'bmi_h'],errors='coerce').to_numpy(dtype=float,na_value=np.nan)
 matrix['prior_rate']=np.where(pf.prior_n.to_numpy()>0,pf.prior_pos.to_numpy()/np.maximum(pf.prior_n.to_numpy(),1),np.nan)
 cols=feature_columns('exposure'); a=e.lf.rf_oof_lf(cols,folds,y,matrix,0); b=e.lf.rf_oof_lf(cols+e.bat.COLS_PRIOR,folds,y,matrix,0)
 return {'delta':e.weighted_auc(y,b,w)-e.weighted_auc(y,a,w),'ess':float(w.sum()**2/(w*w).sum()),'n':len(y)}
def main():
 raw=e.raw_data(); raw['_cv_group']=raw.record_id
 ids=raw.record_id.unique(); blocks={h:raw[raw.record_id==h].copy() for h in ids}; t=time.time()
 out={'status':'running','n_requested':N,'smoke':SMOKE,'seed_fixed':0,'bootstrap_seed':91203,'replicates':[],'raw_sha256':hashlib.sha256(open(e.RDA,'rb').read()).hexdigest(),'script_sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest(),'python':platform.python_version(),'design':'All raw households resampled; each replicate refits cross-fitted unbalanced observation model and both fold-clean RF arms. Duplicate copies of the same original household stay in the same CV fold; SOP constructed within each sampled copy. Seed 0 fixed.','limitations':['MAR and correct observation modelling assumed; weights truncated at 1st/99th percentiles.','Missing household outcomes cannot be reconstructed; this weights complete-case predictions, not full-population screening trajectories.']}
 def save():
  out['runtime_sec']=time.time()-t
  with open(OUT+'.tmp','w',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
  os.replace(OUT+'.tmp',OUT)
 try:
  out['point_seed0']=once(raw); save()
  for i in range(N):
   rng=np.random.default_rng(91203+i); parts=[]
   for j,h in enumerate(rng.choice(ids,len(ids),replace=True)):
    part=blocks[h].copy(); part['record_id']='copy_%04d'%j; parts.append(part)
   out['replicates'].append(once(pd.concat(parts,ignore_index=True))); save()
   print('%d/%d, %.0fs'%(i+1,N,time.time()-t),flush=True)
  vals=[r['delta'] for r in out['replicates']]
  out['summary']={'mean':float(np.mean(vals)),'ci95_percentile':np.percentile(vals,[2.5,97.5]).tolist(),'n_effective':len(vals)}; out['status']='complete'; save(); print(out['summary'],flush=True)
 except Exception as ex:
  out['status']='failed'; out['error']=repr(ex); save(); raise
if __name__=='__main__': main()
