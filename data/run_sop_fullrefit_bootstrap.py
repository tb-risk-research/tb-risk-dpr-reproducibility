#!/usr/bin/env python3
"""A1: household bootstrap with full fold-clean refitting; seed 0 fixed."""
import json, os, sys, time
import numpy as np
import pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import run_sop_robustness_battery as bat
import run_sop_unified_ci_leakfree as lf
import run_sop_unified_ci as u
from tb_risk.validation.household_temporal import assign_screening_order, prior_features
from tb_risk.validation.real_data_infection import feature_columns, load_homeacf_contacts
from tb_risk.validation.real_data_pi import _group_cv_indices

SMOKE=len(sys.argv)>1 and sys.argv[1]=='smoke'; N=5 if SMOKE else 300; SEED=0
OUT=os.path.join(HERE,'processed','homeacf_dpr_a1_fullrefit_%s.json'%time.strftime('%Y%m%d'))
def boot(base,rng):
    ids=base.record_id.drop_duplicates().to_numpy(); out=[]
    for j,h in enumerate(rng.choice(ids,len(ids),replace=True)):
        x=base[base.record_id==h].copy(); x['_cv_group']=h; x['record_id']='boot_%04d'%j; out.append(x)
    return pd.concat(out,ignore_index=True)
def once(df):
    y=df.tst_pos10.astype(int).to_numpy(); g=df._cv_group.to_numpy(); bmi=df._raw_bmi.to_numpy(float)
    d=df.drop(columns=['_raw_bmi','_cv_group']); folds=_group_cv_indices(g,y,n_splits=5,seed=SEED)
    pf=prior_features(d,assign_screening_order(d,seed=SEED,mode='random')); raw=pd.concat([d.reset_index(drop=True),pf],axis=1)
    raw['bmi_h']=bmi; raw['prior_rate']=np.where(pf.prior_n.to_numpy()>0,pf.prior_pos.to_numpy()/np.maximum(pf.prior_n.to_numpy(),1),np.nan)
    cols=feature_columns('exposure'); a=lf.rf_oof_lf(cols,folds,y,raw,SEED); b=lf.rf_oof_lf(cols+bat.COLS_PRIOR,folds,y,raw,SEED)
    return float(u._auroc(y,b)-u._auroc(y,a)),int(len(y)),int(y.sum()),int(len(np.unique(g)))
def main():
    df=load_homeacf_contacts(); df['_raw_bmi']=bat.load_raw_bmi(df); df['_cv_group']=df.record_id; rng=np.random.default_rng(91001); vals=[]; sizes=[]; fails=[]; t=time.time()
    for i in range(N):
        try: vals.append(once(boot(df,rng))); 
        except Exception as e: fails.append({'replicate':i,'error':repr(e)})
        if (i+1)%10==0: print('%d/%d %.0fs'%(i+1,N,time.time()-t),flush=True)
    d=np.array([x[0] for x in vals]); out={'name':'homeacf_dpr_a1_fullrefit','smoke':SMOKE,'seed_fixed':SEED,'n_requested':N,'n_effective':len(d),'failures':fails,'design':'household bootstrap; each replicate rebuilds SOP features, fold-clean preprocessing, grouped 5-fold CV, and both RF arms','delta_auroc':{'mean':float(d.mean()),'ci95':[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))],'distribution':d.tolist()},'sample_summary':{'n_mean':float(np.mean([x[1] for x in vals])),'events_mean':float(np.mean([x[2] for x in vals])),'households_mean':float(np.mean([x[3] for x in vals]))}}
    with open(OUT,'w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=1)
    print(OUT, out['delta_auroc'],flush=True)
if __name__=='__main__': main()
