#!/usr/bin/env python3
"""F1 operating metrics from frozen leak-free SOP pipeline."""
import json, os, sys, time
import numpy as np
from sklearn.metrics import average_precision_score
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import run_sop_unified_ci_leakfree as lf
import run_sop_robustness_battery as bat
from tb_risk.validation.real_data_infection import load_homeacf_contacts
SMOKE=len(sys.argv)>1 and sys.argv[1]=='smoke'; N=2 if SMOKE else 20
OUT=os.path.join(HERE,'processed','homeacf_dpr_f1_operating_metrics_%s.json'%time.strftime('%Y%m%d'))
BUDGETS=(.05,.10,.20,.30)
def metrics(y,s,q):
    n=max(1,int(np.ceil(len(y)*q))); ix=np.argsort(-s)[:n]; captured=int(y[ix].sum()); total=int(y.sum())
    return {'capture_rate':captured/total,'captured_events':captured,'screened_n':n,'nns':n/captured if captured else None}
def main():
 t=time.time(); df=load_homeacf_contacts(); y=df.tst_pos10.astype(int).to_numpy(); g=df.record_id.to_numpy(); bmi=bat.load_raw_bmi(df); rows=[]
 for seed in range(N):
  r=lf.seed_full_lf(seed,df,y,g,bmi)
  for arm in ('exposure','exposure_prior'):
   s=r['oof'][arm]; rows.append({'seed':seed,'arm':arm,'auprc':float(average_precision_score(y,s)),'budgets':{str(q):metrics(y,s,q) for q in BUDGETS}})
  print('seed %d/%d'%(seed+1,N),flush=True)
 out={'name':'homeacf_dpr_f1_operating_metrics','smoke':SMOKE,'design':'Frozen leak-free RF OOF; 20 seeds; household-grouped five-fold CV; no retuning.','sample':{'n':len(y),'events':int(y.sum()),'households':int(len(np.unique(g)))},'rows':rows,'runtime_sec':time.time()-t}
 with open(OUT,'w',encoding='utf8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
 print(OUT)
if __name__=='__main__': main()
