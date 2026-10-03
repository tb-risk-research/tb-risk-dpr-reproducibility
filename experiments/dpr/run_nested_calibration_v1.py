"""Outer household holdout, inner grouped OOF Platt; no retuning."""
import sys,json,time,hashlib,platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss,average_precision_score
from scipy.optimize import brentq
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'data'))
import run_sop_unified_ci_leakfree as lf
import run_sop_robustness_battery as bat
OUT=ROOT/'data/processed/homeacf_dpr_nested_calibration_v1.json'
CACHE=OUT.with_suffix('.npz')
N=20; BOOT=2000; THRESHOLDS=[.05,.1,.2,.3]

def cal_stats(y,p):
    x=logit(np.clip(p,1e-6,1-1e-6))
    lr=LogisticRegression(C=1e6,max_iter=2000).fit(x[:,None],y)
    cil=brentq(lambda a: np.mean(expit(x+a))-np.mean(y),-25,25)
    binid=np.minimum((p*10).astype(int),9)
    bins=[]; ece=0.
    for b in range(10):
        m=binid==b
        if m.any():
            pred=float(p[m].mean()); obs=float(y[m].mean())
            ece+=m.mean()*abs(pred-obs)
            bins.append({'bin':b,'n':int(m.sum()),'mean_predicted':pred,'observed':obs})
    return {'brier':float(brier_score_loss(y,p)),'auroc':float(roc_auc_score(y,p)),
      'auprc':float(average_precision_score(y,p)), 'mean_prediction':float(p.mean()),
      'calibration_in_large_offset':float(cil), 'calibration_intercept':float(lr.intercept_[0]),
      'calibration_slope':float(lr.coef_[0,0]),'ece_10_equal_width':float(ece),'bins':bins}

def main():
    start=time.time(); df=lf.load_homeacf_contacts(); y=df.tst_pos10.to_numpy(dtype=int)
    g=df.record_id.to_numpy(); bmi=bat.load_raw_bmi(df)
    names=['exposure','exposure_prior']; raw_scores=[]; calibrated=[]; rows=[]
    if OUT.exists():
        state=json.loads(OUT.read_text()); rows=state.get('rows',[])
        if rows:
            z=np.load(CACHE); raw_scores=list(z['raw']); calibrated=list(z['calibrated'])
        assert len(rows)==len(raw_scores)==len(calibrated)
    for seed in range(len(rows),N):
        order=lf.assign_screening_order(df,seed=seed,mode='random')
        pf=lf.prior_features(df,order)
        data=df.copy(); data['bmi_h']=bmi
        data=pd.concat([data.reset_index(drop=True),pf],axis=1)
        data['prior_rate']=np.where(pf.prior_n>0,pf.prior_pos/np.maximum(pf.prior_n,1),np.nan)
        outer=lf._group_cv_indices(g,y,n_splits=5,seed=seed)
        rp=np.zeros((2,len(y))); cp=np.zeros_like(rp)
        for f,(tr,te) in enumerate(outer):
            assert not set(g[tr]) & set(g[te])
            inner_seed=100000+seed*5+f
            inner=lf._group_cv_indices(g[tr],y[tr],n_splits=4,seed=inner_seed)
            for itr,ite in inner: assert not set(g[tr][itr]) & set(g[tr][ite])
            for arm,name in enumerate(names):
                cols=lf.inf_feature_columns('exposure')+(bat.COLS_PRIOR if arm else [])
                inner_oof=lf.rf_oof_lf(cols,inner,y[tr],data.iloc[tr],seed)
                calibrator=LogisticRegression(C=1e6,max_iter=2000)
                calibrator.fit(logit(np.clip(inner_oof,1e-6,1-1e-6))[:,None],y[tr])
                X,_,_=lf._foldclean_matrix(cols,[(tr,te)],y,data)[0]
                model=bat._make_model('random_forest',seed)
                model.fit(X[tr],y[tr]); p=model.predict_proba(X[te])[:,1]
                rp[arm,te]=p
                cp[arm,te]=calibrator.predict_proba(logit(np.clip(p,1e-6,1-1e-6))[:,None])[:,1]
            print('seed',seed,'outer',f+1,'/5',flush=True)
        raw_scores.append(rp); calibrated.append(cp)
        rows.append({'seed':seed,'arms':{name:{'raw':cal_stats(y,rp[i]),'nested_calibrated':cal_stats(y,cp[i])} for i,name in enumerate(names)}})
        np.savez_compressed(CACHE,raw=np.asarray(raw_scores),calibrated=np.asarray(calibrated),y=y,g=g)
        checkpoint={'status':'running','completed_seeds':len(rows),'target_seeds':N,'rows':rows}
        OUT.with_suffix('.tmp').write_text(json.dumps(checkpoint,indent=2)); OUT.with_suffix('.tmp').replace(OUT)
        print('saved',len(rows),'/',N,flush=True)
    rp=np.asarray(raw_scores); cp=np.asarray(calibrated)
    # Gate: outer uncalibrated fit reproduces original frozen primary comparison.
    reference=json.loads((ROOT/'data/processed/sop_unified_ci_leakfree_homeacf_20261001.json').read_text())
    raw_auc=np.array([[r['arms'][name]['raw']['auroc'] for name in names] for r in rows])
    assert abs(raw_auc[:,0].mean()-.6440512883302684)<1e-9
    assert abs(raw_auc[:,1].mean()-.7141509122974732)<1e-9
    losses=(cp-y)**2; delta_loss=(losses[:,1,:]-losses[:,0,:]).mean(axis=0)
    nb={}
    for pt in THRESHOLDS:
        selected=cp>=pt
        utility=selected*(y-(1-y)*pt/(1-pt))
        nb[str(pt)]={'arms_mean':utility.mean(axis=(0,2)).tolist(),
          'delta_by_contact':(utility[:,1,:]-utility[:,0,:]).mean(axis=0)}
    ids=np.unique(g); blocks=[np.flatnonzero(g==h) for h in ids]
    rng=np.random.default_rng(91204); brier_boot=[]; nb_boot={str(t):[] for t in THRESHOLDS}
    for b in range(BOOT):
        ix=np.concatenate([blocks[k] for k in rng.integers(0,len(blocks),len(blocks))])
        brier_boot.append(float(delta_loss[ix].mean()))
        for pt in THRESHOLDS: nb_boot[str(pt)].append(float(nb[str(pt)]['delta_by_contact'][ix].mean()))
    for pt in THRESHOLDS:
        v=nb[str(pt)]; v['delta_mean']=float(v.pop('delta_by_contact').mean())
        v['conditional_ci95']=np.quantile(nb_boot[str(pt)],[.025,.975]).tolist()
        v['treat_all_nb']=float(y.mean()-(1-y.mean())*pt/(1-pt)); v['treat_none_nb']=0.
    stats={}
    for name in names:
        stats[name]={}
        for kind in ['raw','nested_calibrated']:
            stats[name][kind]={key:{'mean':float(np.mean([r['arms'][name][kind][key] for r in rows])),
                                   'seed_min':float(min(r['arms'][name][kind][key] for r in rows)),
                                   'seed_max':float(max(r['arms'][name][kind][key] for r in rows))}
                for key in ['brier','auroc','auprc','mean_prediction','calibration_in_large_offset',
                            'calibration_intercept','calibration_slope','ece_10_equal_width']}
    result={'status':'complete','completed_seeds':N,'target_seeds':N,'n':len(y),'events':int(y.sum()),
      'households':len(ids),'rows':rows,'summary':stats,'dca':nb,
      'brier_delta':{'mean':float(delta_loss.mean()),'conditional_ci95':np.quantile(brier_boot,[.025,.975]).tolist()},
      'design':{'outer':'5 household-grouped folds seeds 0-19','inner':'4 household-grouped folds, seed 100000+5*outerseed+fold',
        'platt':'LR C=1e6 max_iter=2000 on logit(inner OOF); no balancing',
        'base_models':'frozen RF; refit outer training after calibration; no tuning',
        'bootstrap_replicates':BOOT,'bootstrap_seed':91204,'thresholds':THRESHOLDS},
      'limits':['Conditional intervals fix all fitted models and calibrators.',
         'Inner-trained score calibration is transferred to a larger outer-training fit; calibration is empirical, not guaranteed.',
         'DCA thresholds are illustrative, not established clinical thresholds; acquisition costs omitted.',
         'Calibration intercept/slope and ECE are descriptive OOF summaries; seed ranges are not confidence intervals.',
         'No observed screening order or real implementation outcomes.'],
      'raw_primary_auc_reproduction_passed':True,'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__},
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'data_sha256':hashlib.sha256(Path(bat._RDA_PATH).read_bytes()).hexdigest(),
      'runtime_current_session_seconds':time.time()-start}
    OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps(result['brier_delta']),flush=True)
if __name__=='__main__': main()
