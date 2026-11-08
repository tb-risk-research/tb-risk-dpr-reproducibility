"""Local, frozen source-trained external assessment; see protocol.md.

Source project: https://github.com/tiande888/tb_risk
Base commit: 3cc3c82f42dbf1e2bb8c3c4209e078a99db7c5c7.
EB implementation is loaded from that project's local estimator, not rewritten.
No publication or manuscript mutation is performed by this script.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import ast, hashlib, json, pickle, platform, time, traceback, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import rdata, sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.special import expit, logit
from scipy.optimize import brentq, minimize

ROOT = Path(os.environ.get('DPR_ROOT', 'D:/tb_risk_DPR'))
OUT = ROOT/'data/processed/dpr_external_validation_20261005'
ARMS = ['lr_base','lr_eb','lr_four','rf_base','rf_eb','rf_four']
METRICS = ['auroc','auprc','brier','log_loss','capture10','capture20','capture30']
START = time.time()

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p, obj):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(p)
def progress(stage, **kw):
    obj = dict(stage=stage, status='running', pid=os.getpid(), elapsed_seconds=round(time.time()-START,1), **kw)
    atomic(OUT/'progress.json', obj)
    print(json.dumps(obj, ensure_ascii=False), flush=True)

def load_eb():
    path = ROOT/'data/run_sop_deepening.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eb_shrinkage_params')
    ns = {'np':np,'pd':pd}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    return ns['eb_shrinkage_params']
EB = load_eb()

def load_data():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        raw = rdata.conversion.convert(rdata.parser.parse_file(ROOT/'data/raw/homeacf_tstsa.rda'))['tstsa']
    atomic(OUT/'audit/rdata_warnings.json', sorted(set(str(w.message) for w in caught)))
    a = raw.loc[raw.tstdiam_h.notna()].copy().reset_index(drop=True)
    sx = {'Male':1.,'Female':0.}
    Xs = np.column_stack([a.ageyears_h, a.sex_h.map(sx), a.ageyears_i, a.sex_i.map(sx)]).astype(float)
    ys = (a.tstdiam_h.to_numpy() >= 10).astype(int)
    gs = a.record_id.astype(str).to_numpy()
    b = pd.read_csv(OUT/'raw/brazil2018_hhc_derived.csv')
    assert b.contactid.is_unique and b.indexid.notna().all() and b.ppd1.notna().all()
    assert set(b.HHCgender.dropna().unique()) <= {1,2}
    assert set(b.genderindex.dropna().unique()) <= {1,2}
    Xt = np.column_stack([b.HHCage, b.HHCgender.map({1:1.,2:0.}), b.ageindex, b.genderindex.map({1:1.,2:0.})]).astype(float)
    yt = b.ppd1.ge(10).to_numpy(dtype=int)
    gt = b.indexid.astype(str).to_numpy()
    assert len(ys)==2725 and ys.sum()==359 and len(np.unique(gs))==877
    assert len(yt)==230 and yt.sum()==154 and len(np.unique(gt))==48
    assert np.all(yt == b.outcome.eq(2).to_numpy())
    assert np.nanmin(Xs[:,[0,2]])>=0 and np.nanmax(Xs[:,[0,2]])<120
    assert np.nanmin(Xt[:,[0,2]])>=0 and np.nanmax(Xt[:,[0,2]])<120
    return Xs,ys,gs,Xt,yt,gt

def ranks(g, seed=0, key=None, descending=False):
    # RandomState ranks reproduce the existing household_temporal order convention.
    rng=np.random.RandomState(seed); r=np.empty(len(g), int)
    for h in np.unique(g):
        ix=np.flatnonzero(g==h)
        if key is None: r[ix]=rng.permutation(len(ix))
        else:
            order=ix[np.argsort((-1 if descending else 1)*key[ix], kind='stable')]
            r[order]=np.arange(len(ix))
    return r

def prior(y,g,r):
    n=np.zeros(len(y)); p=np.zeros(len(y))
    for h in np.unique(g):
        ix=np.flatnonzero(g==h); ix=ix[np.argsort(r[ix],kind='stable')]
        n[ix]=np.arange(len(ix)); p[ix]=np.r_[0,np.cumsum(y[ix])[:-1]]
    return n,p

def design(X,y,g,r,pi,k,repr_):
    if repr_=='base': return X
    n,p=prior(y,g,r)
    if repr_=='eb': return np.c_[X,(p+k*pi)/(n+k)]
    return np.c_[X,n,p,np.where(n>0,p/np.maximum(n,1),pi),(n>0).astype(float)]

def model(arm, seed=0):
    if arm.startswith('lr'):
        return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                             LogisticRegression(C=1,max_iter=2000,random_state=seed))
    return make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(
        n_estimators=300,max_depth=7,min_samples_split=5,min_samples_leaf=2,
        class_weight='balanced',random_state=seed,n_jobs=1))

def weighted_metrics(y,p,W):
    # W: independent household-bootstrap multiplicities, B x participants.
    ix=np.argsort(-p,kind='stable'); pp=p[ix]; yy=y[ix]
    start=np.r_[0,np.flatnonzero(np.diff(pp)!=0)+1]
    wp=np.add.reduceat(W[:,ix]*yy,start,axis=1)
    wn=np.add.reduceat(W[:,ix]*(1-yy),start,axis=1)
    pos=wp.sum(axis=1); neg=wn.sum(axis=1); N=pos+neg
    cp=wp.cumsum(axis=1); cn=wn.cumsum(axis=1); total=cp+cn
    good=(pos>0)&(neg>0)
    auc=(wp*(neg[:,None]-cn+.5*wn)).sum(axis=1)/(pos*neg)
    ap=(wp/pos[:,None]*cp/np.maximum(total,1e-15)).sum(axis=1)
    q=np.clip(p,1e-12,1-1e-12)
    bri=(W*(y-p)**2).sum(axis=1)/N
    ll=(W*(-y*np.log(q)-(1-y)*np.log(1-q))).sum(axis=1)/N
    caps=[]
    wt=wp+wn; before=total-wt
    for f in [.1,.2,.3]:
        tests=np.ceil(f*N)
        allocate=np.clip((tests[:,None]-before)/np.maximum(wt,1e-15),0,1)
        caps.append((wp*allocate).sum(axis=1)/pos)
    out=np.column_stack([auc,ap,bri,ll,*caps]);out[~good,:]=np.nan
    return out

def diagnostics(y,p):
    q=np.clip(p,1e-8,1-1e-8); l=logit(q)
    offset=brentq(lambda a: np.mean(expit(a+l))-np.mean(y),-40,40)
    def fun(z):
        eta=z[0]+z[1]*l
        return np.mean(np.logaddexp(0,eta)-y*eta)
    opt=minimize(fun,[0.,1.],method='BFGS')
    bins=np.minimum((p*10).astype(int),9); ece=0.
    for b in range(10):
        m=bins==b
        if m.any(): ece+=m.mean()*abs(y[m].mean()-p[m].mean())
    return dict(mean_predicted=float(p.mean()),ece10=float(ece),
                calibration_intercept_offset=float(offset),
                calibration_intercept_joint=float(opt.x[0]),
                calibration_slope=float(opt.x[1]),calibration_optimizer_success=bool(opt.success))

def local_cv(X,y,g,label):
    cache=OUT/'results'/f'{label}_internal_lr_predictions.npz'
    if cache.exists(): return np.load(cache)['predictions']
    pred=np.empty((20,3,len(y)))
    for seed in range(20):
        r=ranks(g,seed)
        folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(X,y,g))
        for train,test in folds:
            assert not set(g[train]) & set(g[test])
            pi,k=EB(y[train],g[train])
            for j,arm in enumerate(ARMS[:3]):
                D=design(X,y,g,r,pi,k,arm.split('_')[1]); m=model(arm,seed)
                m.fit(D[train],y[train]);pred[seed,j,test]=m.predict_proba(D[test])[:,1]
        if seed%5==0: progress('local_internal_cv',cohort=label,seed_completed=seed+1,total_seeds=20)
    np.savez_compressed(cache,predictions=pred)
    return pred

def summary(pred,y,g,label):
    _,gid=np.unique(g,return_inverse=True); H=gid.max()+1
    counts=np.random.default_rng(1052026).multinomial(H,np.full(H,1/H),size=2000)
    W=counts[:,gid]
    point=np.empty((pred.shape[0],pred.shape[1],len(METRICS)))
    boot=np.zeros((pred.shape[1],2000,len(METRICS)))
    for s in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            point[s,j]=weighted_metrics(y,pred[s,j],np.ones((1,len(y))))[0]
            v=weighted_metrics(y,pred[s,j],W)
            boot[j]+=v/pred.shape[0]
        if s%5==0: progress('household_bootstrap',cohort=label,order_completed=s+1,total_orders=pred.shape[0])
    valid=np.isfinite(boot).all(axis=(0,2))
    out={'n':len(y),'events':int(y.sum()),'households':int(H),'prevalence':float(y.mean()),
         'bootstrap_replicates':2000,'valid_bootstrap_replicates':int(valid.sum()),'models':{},'comparisons':{}}
    for j,arm in enumerate(ARMS[:pred.shape[1]]):
        out['models'][arm]={name:{'mean':float(point[:,j,k].mean()),
            'conditional_ci95':np.quantile(boot[j,valid,k],[.025,.975]).tolist(),
            'order_range':[float(point[:,j,k].min()),float(point[:,j,k].max())]}
            for k,name in enumerate(METRICS)}
    pairs=[('lr_eb_minus_base',1,0),('lr_four_minus_base',2,0),('lr_four_minus_eb',2,1)]
    if pred.shape[1]>3: pairs += [('rf_eb_minus_base',4,3),('rf_four_minus_base',5,3),('rf_four_minus_eb',5,4)]
    for name,a,b in pairs:
        out['comparisons'][name]={key:{'mean_difference':float((point[:,a,k]-point[:,b,k]).mean()),
            'conditional_ci95':np.quantile((boot[a]-boot[b])[valid,k],[.025,.975]).tolist()}
            for k,key in enumerate(METRICS)}
    np.savez_compressed(OUT/'results'/f'{label}_bootstrap.npz',bootstrap=boot,points=point,valid=valid)
    return out

def main():
    for sub in ['raw','audit','code','results','figures']: (OUT/sub).mkdir(parents=True,exist_ok=True)
    protocol=OUT/'protocol.md'; assert protocol.exists()
    progress('data_checks')
    Xs,ys,gs,Xt,yt,gt=load_data()
    r=ranks(gt,0);n,p=prior(yt,gt,r)
    for i in range(len(yt)):
        earlier=(gt==gt[i])&(r<r[i])
        assert n[i]==earlier.sum() and p[i]==yt[earlier].sum()
        changed=yt.copy();changed[i]=1-yt[i]
        assert prior(changed,gt,r)[1][i]==p[i]
    audit={'source_n':len(ys),'source_households':len(np.unique(gs)),'source_events':int(ys.sum()),
           'target_n':len(yt),'target_households':len(np.unique(gt)),'target_events':int(yt.sum()),
           'target_complete_baseline_n':int(np.isfinite(Xt).all(axis=1).sum()),
           'source_missing_baseline':np.isnan(Xs).sum(axis=0).tolist(),
           'target_missing_baseline':np.isnan(Xt).sum(axis=0).tolist(),
           'no_own_or_future_label_in_prior':True,
           'protocol_sha256_before_fitting':sha(protocol),'script_sha256':sha(__file__),
           'fitting_started_utc':datetime.now(timezone.utc).isoformat(),
           'source_sha256':sha(ROOT/'data/raw/homeacf_tstsa.rda'),
           'target_xlsx_sha256':sha(OUT/'raw/brazil2018_aerosol_s002.xlsx'),
           'target_csv_sha256':sha(OUT/'raw/brazil2018_hhc_derived.csv'),
           'eb_source_sha256':sha(ROOT/'data/run_sop_deepening.py')}
    atomic(OUT/'audit/pre_fit_audit.json',audit)
    pi,k=EB(ys,gs);rs=ranks(gs,0)
    freeze=OUT/'results/frozen_source_models.pkl'
    if freeze.exists():
        with freeze.open('rb') as f: models=pickle.load(f)
        assert models['audit']==audit or models['protocol_sha256']==sha(protocol)
        models=models['models']
    else:
        models={}
        for arm in ARMS:
            progress('source_training',arm=arm,external_training=False)
            m=model(arm);m.fit(design(Xs,ys,gs,rs,pi,k,arm.split('_')[1]),ys);models[arm]=m
        with freeze.open('wb') as f: pickle.dump({'models':models,'audit':audit,'protocol_sha256':sha(protocol),'pi':pi,'k':k},f)
    # Pure predict on Brazil: no target fit and no target calibration.
    pred=np.empty((20,len(ARMS),len(yt)))
    for seed in range(20):
        rt=ranks(gt,seed)
        for j,arm in enumerate(ARMS):
            pred[seed,j]=models[arm].predict_proba(design(Xt,yt,gt,rt,pi,k,arm.split('_')[1]))[:,1]
    assert np.isfinite(pred).all()
    np.savez_compressed(OUT/'results/external_predictions_local.npz',predictions=pred,y=yt,groups=gt)
    progress('external_predictions_complete',source_prevalence=pi,target_prevalence=float(yt.mean()))
    external=summary(pred,yt,gt,'external_frozen')
    atomic(OUT/'results/external_frozen_summary.json',external)
    rows=[]
    for label,key,descending in [('age_ascending',np.nan_to_num(Xt[:,0],nan=float(np.nanmedian(Xs[:,0]))),False),
                                  ('age_descending',np.nan_to_num(Xt[:,0],nan=float(np.nanmedian(Xs[:,0]))),True),
                                  ('baseline_risk_descending',pred[0,0],True)]:
        rt=ranks(gt,key=key,descending=descending)
        for arm in ARMS:
            pp=models[arm].predict_proba(design(Xt,yt,gt,rt,pi,k,arm.split('_')[1]))[:,1]
            values=weighted_metrics(yt,pp,np.ones((1,len(yt))))[0]
            rows.append({'ordering':label,'arm':arm,**dict(zip(METRICS,values))})
    pd.DataFrame(rows).to_csv(OUT/'results/structured_ordering_sensitivity.csv',index=False)
    complete=np.isfinite(Xt).all(axis=1);rows=[]
    for j,arm in enumerate(ARMS):
        for s in range(20):
            values=weighted_metrics(yt[complete],pred[s,j,complete],np.ones((1,complete.sum())))[0]
            rows.append({'seed':s,'arm':arm,**dict(zip(METRICS,values))})
    pd.DataFrame(rows).groupby('arm')[METRICS].mean().to_csv(OUT/'results/complete_baseline_sensitivity.csv')
    cal={}
    for j,arm in enumerate(ARMS):
        per=[diagnostics(yt,pred[s,j]) for s in range(20)]
        cal[arm]={'mean_across_orders':{key:float(np.mean([v[key] for v in per])) for key in per[0] if key!='calibration_optimizer_success'},
                  'successful_joint_calibration_fits':sum(v['calibration_optimizer_success'] for v in per)}
    atomic(OUT/'results/calibration_diagnostics.json',cal)
    src=local_cv(Xs,ys,gs,'source');tar=local_cv(Xt,yt,gt,'target')
    # Local internal validations are descriptive comparison, not external intervals.
    internal={}
    for label,yy,pr in [('source',ys,src),('target',yt,tar)]:
        internal[label]={arm:{name:float(np.mean([weighted_metrics(yy,pr[s,j],np.ones((1,len(yy))))[0,k]
                         for s in range(20)])) for k,name in enumerate(METRICS)} for j,arm in enumerate(ARMS[:3])}
    atomic(OUT/'results/local_internal_lr_summary.json',internal)
    result={'status':'complete','study_role':'exploratory external transport assessment of reduced common-feature extension',
        'primary_comparison':'lr_eb_minus_base','external_frozen':external,'internal_separate':internal,
        'source_parameters':{'pi':pi,'k':k,'source_order_seed':0,'rf_fit_seed':0},'audit':audit,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
        'limits':['Not external validation of the original full 29-predictor model.',
                 'Independent cohort but no verified screening/return timestamps; order simulated.',
                 'Only 48 external households; wide uncertainty and substantial case-mix/prevalence shift.',
                 'Conditional frozen-model household intervals do not cover source refitting.',
                 'Free-evidence budgets exclude prior result acquisition costs.',
                 'Upstream excluded missing-TST participants unavailable; observed-case cohort.',
                 'Local Brazil CV is internal method replication, not frozen external validation.',
                 'No formal equivalence or clinical deployment claim; no target recalibration in primary.']}
    atomic(OUT/'results/result.json',result)
    flat=[]
    for arm,met in external['models'].items():
        for key,val in met.items(): flat.append({'arm':arm,'metric':key,'estimate':val['mean'],'ci_low':val['conditional_ci95'][0],'ci_high':val['conditional_ci95'][1]})
    pd.DataFrame(flat).to_csv(OUT/'results/external_metrics.csv',index=False)
    make_figures(external,pred,yt)
    atomic(OUT/'progress.json',{'status':'complete','stage':'complete','elapsed_seconds':round(time.time()-START,1),'error':None})
    print(json.dumps({'primary':external['comparisons']['lr_eb_minus_base'],'internal':internal},indent=2),flush=True)

def make_figures(external,pred,y):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for j,arm in enumerate(ARMS):
        v=external['models'][arm]['auroc'];lo,hi=v['conditional_ci95']
        axs[0].errorbar(v['mean'],j,xerr=[[v['mean']-lo],[hi-v['mean']]],fmt='o',color='#165a91' if j<3 else '#bc641a')
    axs[0].set_yticks(range(6),ARMS);axs[0].axvline(.5,ls='--',color='gray');axs[0].set_xlabel('AUROC (frozen models, household 95% CI)');axs[0].invert_yaxis()
    for j,arm in enumerate(ARMS[:3]):
        p=pred[0,j];bins=np.minimum((p*5).astype(int),4);xx=[];yy=[]
        for b in range(5):
            m=bins==b
            if m.sum()>=5:xx.append(p[m].mean());yy.append(y[m].mean())
        axs[1].plot(xx,yy,'o-',label=arm)
    axs[1].plot([0,1],[0,1],'--',color='gray');axs[1].set(xlim=(0,1),ylim=(0,1),xlabel='Predicted probability',ylabel='Observed TST positivity')
    axs[1].legend();axs[1].set_title('Calibration, target order seed 0')
    fig.suptitle('Brazil 2018: 230 contacts, 48 households; baseline TST >=10 mm')
    fig.savefig(OUT/'figures/external_transport.png',dpi=180);fig.savefig(OUT/'figures/external_transport.pdf');plt.close(fig)

if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic(OUT/'progress.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc()})
        raise
