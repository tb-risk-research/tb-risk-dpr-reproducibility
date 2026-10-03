#!/usr/bin/env python3
"""E1: audit missing HomeACF TST outcomes and IPW-reweight the SOP comparison.

The observation model uses only variables available before a TST result.  It
never uses tstdiam_h, tstread_h, tstdone_h or any SOP-derived variable.
Prediction scores are regenerated with the frozen leak-free grouped-CV
pipeline; IPW changes only the performance estimand among complete cases.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import rdata
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_sop_unified_ci_leakfree as lf
import run_sop_robustness_battery as bat
import run_sop_unified_ci as uni
from tb_risk.validation.real_data_infection import warnings_catch, load_homeacf_contacts
from tb_risk.validation.real_data_pi import _group_cv_indices

SMOKE = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
N_SEEDS = 2 if SMOKE else 20
N_BOOT = 200 if SMOKE else 2000
OUT = os.path.join(HERE, 'processed', 'homeacf_dpr_e1_tst_missingness_ipw_%s_v2%s.json' % (time.strftime('%Y%m%d'), '_smoke' if SMOKE else ''))
RDA = os.path.join(HERE, 'raw', 'homeacf_tstsa.rda')
NUM = ['ageyears_h','bmi_h','num_contacts','coughdays_i','ageyears_i','totalrooms_h','windows_h','numsmokers_h']
CAT = ['sex_h','relationship_h','employment_h','airspace_h','timespent_h','smoke_h','alcohol_h','diabetes_h','hivfinal_h','site','dead_i','sex_i','xpert_i','culture_i','smear_i','hiv_i','housetype_h','pipedwater_h','toilet_h']

def raw_data():
    with warnings_catch():
        d = rdata.conversion.convert(rdata.parser.parse_file(RDA))['tstsa'].copy()
    d.columns = [str(x) for x in d.columns]
    d['tst_observed'] = pd.to_numeric(d['tstdiam_h'], errors='coerce').notna().astype(int)
    return d.reset_index(drop=True)

def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    den = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2)
    return float((np.nanmean(a) - np.nanmean(b)) / den) if den else 0.0

def audit(d):
    rows=[]; obs=d.tst_observed.eq(1)
    for c in NUM:
        x=pd.to_numeric(d[c], errors='coerce')
        rows.append({'variable':c,'type':'numeric','observed_mean':float(x[obs].mean()),'missing_mean':float(x[~obs].mean()),'smd':smd(x[obs],x[~obs]),'n_missing_value':int(x.isna().sum())})
    for c in CAT:
        x=d[c].astype('string').fillna('<missing>')
        for level in sorted(x.unique().tolist()):
            z=x.eq(level).astype(float)
            rows.append({'variable':c+'='+str(level),'type':'category','observed_mean':float(z[obs].mean()),'missing_mean':float(z[~obs].mean()),'smd':smd(z[obs],z[~obs]),'n_missing_value':int(x.eq('<missing>').sum())})
    return rows

def observation_oof(d):
    x=d[NUM+CAT].copy(); y=d.tst_observed.to_numpy(); g=d['_cv_group'].to_numpy() if '_cv_group' in d else d.record_id.to_numpy()
    for c in NUM: x[c]=pd.to_numeric(x[c],errors='coerce').astype(float).replace([np.inf,-np.inf],np.nan)
    for c in CAT: x[c]=x[c].astype('string').fillna('<missing>').astype(str)
    prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),CAT)])
    p=np.zeros(len(d))
    for tr,te in _group_cv_indices(g,y,n_splits=5,seed=91201):
        m=Pipeline([('prep',prep),('lr',LogisticRegression(C=1.0,max_iter=2000))])
        m.fit(x.iloc[tr],y[tr]); p[te]=m.predict_proba(x.iloc[te])[:,1]
    return p

def weighted_auc(y, score, weight):
    return float(roc_auc_score(y,score,sample_weight=weight))
    # Weighted Mann-Whitney AUC, tie-safe.
    order=np.argsort(score,kind='mergesort'); yy=y[order]; ww=weight[order]; ss=score[order]
    pos=float(ww[yy==1].sum()); neg=float(ww[yy==0].sum()); acc=0.; neg_before=0.; i=0
    while i<len(y):
        j=i+1
        while j<len(y) and ss[j]==ss[i]: j+=1
        wp=float(ww[i:j][yy[i:j]==1].sum()); wn=float(ww[i:j][yy[i:j]==0].sum())
        acc += wp*(neg_before+0.5*wn); neg_before += wn; i=j
    return acc/(pos*neg)

def main():
    t=time.time(); raw=raw_data(); audit_rows=audit(raw); p=observation_oof(raw)
    keep=raw.tst_observed.eq(1).to_numpy(); pk=p[keep]
    # Complete-case rows preserve raw order after _derive_features drops missing tstdiam.
    lo,hi=np.quantile(1/pk,[.01,.99]); w=np.clip(1/pk,lo,hi); w=w/w.mean()
    df=load_homeacf_contacts(); y=df.tst_pos10.astype(int).to_numpy(); g=df.record_id.to_numpy(); bmi=bat.load_raw_bmi(df)
    assert len(df)==int(keep.sum())==len(w)
    assert np.array_equal(df.contact_id.to_numpy(),raw.loc[keep,'contact_id'].to_numpy())
    deltas=[]; unweighted=[]; details=[]
    scores=[]
    for seed in range(N_SEEDS):
        r=lf.seed_full_lf(seed,df,y,g,bmi); a=r['oof']['exposure']; b=r['oof']['exposure_prior']
        ua=uni._auroc(y,a); ub=uni._auroc(y,b); wa=weighted_auc(y,a,w); wb=weighted_auc(y,b,w)
        deltas.append(wb-wa); unweighted.append(ub-ua); details.append({'seed':seed,'weighted_exposure_auroc':wa,'weighted_sop_auroc':wb,'weighted_delta':wb-wa,'unweighted_delta':ub-ua})
        scores.append((a,b))
        print('seed %d/%d' % (seed+1,N_SEEDS),flush=True)
    # Conditional household-cluster bootstrap of the mean paired OOF difference.
    # It captures sampling variability of the weighted comparison; fitted score
    # and observation models remain fixed, so it is not a full-refit interval.
    rng=np.random.default_rng(91202); ids=np.unique(g); rows=[np.flatnonzero(g==h) for h in ids]; boot=[]
    for j in range(N_BOOT):
        ix=np.concatenate([rows[k] for k in rng.integers(0,len(ids),len(ids))])
        boot.append(float(np.mean([weighted_auc(y[ix],b[ix],w[ix])-weighted_auc(y[ix],a[ix],w[ix]) for a,b in scores])))
    out={'name':'homeacf_dpr_e1_tst_missingness_ipw','smoke':SMOKE,'design':'Raw 2985-row TST observation audit; 5-fold household-grouped cross-fitted logistic observation model with pre-result covariates only; complete-case SOP OOF scores reweighted by stabilized inverse observation probability, truncated at 1st/99th percentiles.','raw_data':{'n_total':int(len(raw)),'n_tst_observed':int(keep.sum()),'n_tst_missing':int((~keep).sum()),'n_households':int(raw.record_id.nunique()),'raw_rda_sha256':None},'audit':audit_rows,'observation_model':{'oof_probability_summary':{'min':float(p.min()),'max':float(p.max()),'mean':float(p.mean())},'complete_case_probability_summary':{'min':float(pk.min()),'max':float(pk.max()),'mean':float(pk.mean())},'weights':{'raw_min':float((1/pk).min()),'raw_max':float((1/pk).max()),'truncation':[float(lo),float(hi)],'effective_sample_size':float((w.sum()**2)/(w*w).sum())}},'ipw_per_seed':details,'summary':{'weighted_delta_mean':float(np.mean(deltas)),'weighted_delta_range':[float(min(deltas)),float(max(deltas))],'weighted_all_positive':bool(np.all(np.array(deltas)>0)),'weighted_delta_conditional_household_bootstrap_ci95':[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],'n_bootstrap':N_BOOT,'unweighted_delta_mean':float(np.mean(unweighted)),'difference_ipw_minus_unweighted':float(np.mean(deltas)-np.mean(unweighted))},'runtime_sec':time.time()-t}
    with open(OUT,'w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
    print(OUT); print(out['summary'])
if __name__=='__main__': main()
