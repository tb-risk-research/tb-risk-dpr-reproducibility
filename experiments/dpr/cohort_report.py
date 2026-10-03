"""Regenerate cohort flow and raw-variable summaries; no participant rows are exported."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'data'))
import run_homeacf_tst_missingness_ipw as e
import run_sop_unified_ci_leakfree as lf
raw=e.raw_data();df=lf.load_homeacf_contacts();keep=raw.tst_observed.eq(1)
flow={'raw_contacts':len(raw),'raw_households':int(raw.record_id.nunique()),'missing_tst':int((~keep).sum()),
 'complete_case_contacts':len(df),'complete_case_households':int(df.record_id.nunique()),'positive10':int(df.tst_pos10.sum()),
 'households_no_observed_tst':len(set(raw.record_id)-set(df.record_id))}
assert list(flow.values())==[2985,924,260,2725,877,359,47]
groups={'all':np.ones(len(raw),bool),'recorded':keep,'missing':~keep}
numerics={};categoricals={}
for c in ['ageyears_h','bmi_h','num_contacts','ageyears_i','coughdays_i']:
    numerics[c]={}
    for name,mask in groups.items():
        v=pd.to_numeric(raw.loc[mask,c],errors='coerce')
        numerics[c][name]={'missing':int(v.isna().sum()),'median':float(v.median()),'q1':float(v.quantile(.25)),'q3':float(v.quantile(.75))}
for c in e.CAT:
    categoricals[c]={n:{str(k):int(v) for k,v in raw.loc[m,c].astype('string').fillna('<missing>').value_counts().items()} for n,m in groups.items()}
out={'flow':flow,'numeric':numerics,'categorical':categoricals,'raw_missing_counts':{c:int(raw[c].isna().sum()) for c in raw},
 'columns':list(raw.columns),'raw_sha256':hashlib.sha256(Path(e.RDA).read_bytes()).hexdigest(),
 'limit':'Unknown, not done and true missing are distinct; no real screening or result-return timestamps.'}
p=ROOT/'data/processed/cohort_rebuilt.json';p.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print(json.dumps(flow))
