"""Conditional budget audit: frozen models, isolated bootstrap household copies."""
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'data'))
import run_sop_unified_ci_leakfree as lf
import run_sop_robustness_battery as bat
import run_sop_scenario_ci as scn
import run_sop_deepening as rsd

OUT = ROOT / 'data/processed/homeacf_dpr_budget_audit_v2.json'
CACHE = OUT.with_suffix('.npz')
Q = .05
KS = [.1, .2, .3]
B = 2000

def metrics(p, y, g, pi, K):
    n = len(y); m = int(np.ceil(Q*n)); total = int(np.ceil(K*n))
    order = np.argsort(-p, kind='mergesort')
    first = order[:m]
    count = np.bincount(g[first], minlength=int(g.max())+1)
    pos = np.bincount(g[first], weights=y[first], minlength=len(count))
    scores = rsd.dep_scores(p, count[g], pos[g], pi, bat.DEP_K, bat.DEP_W)
    remaining = np.ones(n, dtype=bool); remaining[first] = False
    rem = np.flatnonzero(remaining)
    second = rem[np.argsort(-scores[rem], kind='mergesort')[:total-m]]
    assert len(second) + len(first) == total
    assert not np.intersect1d(first, second).size
    hits = int(y[first].sum()+y[second].sum())
    static_hits = int(y[order[:total]].sum())
    return np.array([hits/y.sum(), static_hits/y.sum(), total/hits,
                     total/static_hits, hits-static_hits], dtype=float)

def main():
    t = time.time()
    df = lf.load_homeacf_contacts()
    y = df.tst_pos10.to_numpy(dtype=int)
    original = df.record_id.to_numpy()
    _, g = np.unique(original, return_inverse=True)
    bmi = bat.load_raw_bmi(df)
    raw = df.copy(); raw['bmi_h'] = bmi
    cols = lf.inf_feature_columns('exposure')
    if CACHE.exists():
        z = np.load(CACHE); rf, lr, pis = z['rf'], z['lr'], z['pi']
        assert np.array_equal(y, z['y']) and np.array_equal(g, z['g'])
    else:
        rf, lr, pis = [], [], []
        for s in range(20):
            folds = lf._group_cv_indices(original, y, n_splits=5, seed=s)
            pi = np.zeros(len(y))
            for tr, te in folds: pi[te] = rsd.eb_shrinkage_params(y[tr], original[tr])[0]
            rf.append(lf.rf_oof_lf(cols, folds, y, raw, s))
            lr.append(lf.lr_oof_lf(cols, folds, y, raw, s)); pis.append(pi)
            print('fitted seed', s, flush=True)
        rf, lr, pis = map(np.asarray, (rf, lr, pis))
        np.savez_compressed(CACHE, rf=rf, lr=lr, pi=pis, y=y, g=g)
    point = {}
    for name, ps in [('RF', rf), ('LR', lr)]:
        for K in KS:
            vals = np.array([metrics(p,y,g,pi,K) for p,pi in zip(ps,pis)])
            # Independent comparison against the existing pandas implementation.
            for s in [0,19]:
                fn = bat.cascade_once if name == 'RF' else scn.cascade_once_lr
                old = fn(ps[s],y,original,pis[s],Q,K,'exposure_top',np.random.default_rng(s))
                assert np.allclose(vals[s,[0,2]],old[:2], atol=1e-12)
            point[f'{name}_{K}'] = {'cascade_capture':float(vals[:,0].mean()),
                'static_capture':float(vals[:,1].mean()), 'cascade_nns':float(vals[:,2].mean()),
                'static_nns':float(vals[:,3].mean()), 'delta':float((vals[:,0]-vals[:,1]).mean()),
                'per_seed_delta':(vals[:,0]-vals[:,1]).tolist(),
                'mean_additional_positives':float(vals[:,4].mean()),
                'total_tests':int(np.ceil(K*len(y))), 'first_wave_tests':int(np.ceil(Q*len(y)))}
    boot = {k:[] for k in point}
    blocks = [np.flatnonzero(g==h) for h in range(g.max()+1)]
    for b in range(B):
        draw = np.random.default_rng(70000+b).integers(0,len(blocks),len(blocks))
        selected = [blocks[h] for h in draw]
        idx = np.concatenate(selected)
        clone_g = np.repeat(np.arange(len(draw)), [len(x) for x in selected])
        yb = y[idx]
        for name, ps in [('RF',rf),('LR',lr)]:
            for K in KS:
                values = [metrics(p[idx],yb,clone_g,pi[idx],K) for p,pi in zip(ps,pis)]
                boot[f'{name}_{K}'].append(float(np.mean([v[0]-v[1] for v in values])))
        if (b+1)%100 == 0: print('bootstrap',b+1,'/',B, flush=True)
    for k in point: point[k]['conditional_ci95'] = np.quantile(boot[k],[.025,.975]).tolist()
    result = {'status':'complete', 'n':len(y),'events':int(y.sum()),'households':len(blocks),
        'seeds':list(range(20)), 'first_wave_fraction':Q, 'primary_total_budget':.1,
        'secondary_budgets':[.2,.3], 'bootstrap_replicates':B,'bootstrap_seed':70000,
        'estimator':'mean paired capture difference across 20 seeds',
        'bootstrap':'fixed OOF predictions; each sampled household copy receives distinct aggregation ID',
        'rule':{'k':bat.DEP_K,'w':bat.DEP_W,'selection':'top baseline first, then frozen updated score'},
        'limits':['Conditional intervals omit model refitting uncertainty.',
            'All first-wave results assumed returned before second wave; no observed timing.',
            'Unit costs equal per contact; no delay, attendance, monetary cost or treatment outcomes.',
            'Frozen rule was developed retrospectively; no new parameter optimization in this audit.',
            'Cascade rule differs from feature-augmented RF; do not attribute contrasts only to costs.'],
        'results':point,'runtime_seconds':time.time()-t,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data_sha256':hashlib.sha256(Path(bat._RDA_PATH).read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(point,indent=2),flush=True)

if __name__ == '__main__': main()
