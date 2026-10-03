#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2（第七轮）：时序先证「组粒度-效应量」定量曲线（2026-08-26）。

问题：三国双终点证据链已确认机制存在（HomeACF 户级 +0.058 →
SINAN 市级 +0.002），但「组粒度→效应量」关系未量化。疾控落地时
该按户/楼/村/哪一级分组，机制才值钱？本实验把这条关系画成曲线。

设计（在组粒度轴上采样效应量）：
  1. SINAN 内部按市规模分层（全量分析集每市行数：小市 <1000 /
     中市 1000-10000 / 大市 ≥10000）：各层独立跑同协议
     （ind vs ind+先证块，5 种子市级分组 CV，LR+LGBM，seed0
     市级簇 bootstrap 500）——同一数据集内的粒度-规模变体，
     控制国家/终点/登记口径混杂；
  2. HomeACF 户级（南非，感染终点 TST）：Δ 直接引用归档
     household_temporal_screening_20260825.json（+0.0585），
     ICC 本实验补算（tst_pos10 within record_id）；
  3. PACTS 户级（马拉维，症状终点）：Δ 引用归档
     pacts_temporal_household_20260825.json——终点天花板
     （ind 0.993），inconclusive 灰色参考点，不进主结论；
  4. SINAN 全体（市级）：Δ 引用 v2 归档（全因死亡终点）。

「组内共享暴露比例」代理 = ICC(1)（单因素随机效应 ANOVA 组内
相关系数）：终点在组内的聚集强度，即组内多大比例的结局变异
由组级共享暴露承载。

臂结构（相对 v2 精简——块分解 v2 已完成，此处只回答效应量
随粒度的变化）：ind / ind_all（先证块 = 结局块 + 密度块）。

终点：全因死亡（y = SITUA_ENCE∈{3,4}，字典修正后的稳健终点，
sinan_label_dictionary_20260826.json）。

输出：
  data/processed/granularity_effect_curve_20260826.json
  data/processed/granularity_effect_curve_20260826.png
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 插入 repo 父目录（Desktop）而非 repo 本身——杂散 tb_risk/tb_risk
# 命名空间目录会在后者情况下遮蔽真包（项目已知陷阱）
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from scipy import sparse  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

BASE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
PROC = os.path.join(BASE, 'tb_risk', 'data', 'processed')
OUT_JSON = os.path.join(PROC, 'granularity_effect_curve_20260826.json')
OUT_PNG = os.path.join(PROC, 'granularity_effect_curve_20260826.png')

# ---------------------------------------------------------------------------
# 常量（与 run_sinan_temporal_prior.py v2 同口径）
# ---------------------------------------------------------------------------
BIN_AGRAV = ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
             'AGRAVDROGA', 'AGRAVTABAC', 'POP_LIBER', 'POP_RUA',
             'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV']
CAT_FEATS = ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'FORMA', 'EXTRAPU1_N',
             'RAIOX_TORA', 'TESTE_TUBE', 'CULTURA_ES', 'HISTOPATOL',
             'TEST_MOLEC', 'BACILOSC_E', 'CS_GESTANT']
OUTCOME_BLOCK = ['prior_n', 'prior_death', 'prior_rate']
DENSITY_BLOCK = ['notif_1y', 'log_notif_1y']
K_SHRINK = 2.0
MISSING_ENCERRA_SENTINEL = 10 ** 9

N_SEEDS = 5
N_FOLDS = 5
N_BOOT = 500
MUNIP_CAP = 60          # 层内每市建模采样上限（先证特征仍全量构造）
LAYER_MAX_ROWS = 150000  # 层内建模行数上限（超出则随机抽市）

# 市规模分层阈值（全量分析集每市行数）
SMALL_MAX = 1000
MID_MAX = 10000


# ---------------------------------------------------------------------------
# 复用函数（自 run_sinan_temporal_prior.py 复制——该模块顶层按
# sys.argv 决定终点，import 有副作用，故复制而非引用）
# ---------------------------------------------------------------------------

def _day_number(date_str):
    s = pd.Series(date_str).astype(str).str.strip()
    ok = s.str.fullmatch(r'\d{8}', na=False)
    dt = pd.to_datetime(s.where(ok), format='%Y%m%d', errors='coerce')
    days = dt.values.astype('datetime64[D]').astype(np.int64)
    return days, ok.values


def build_prior_features(mun, notif_day, enc_day, yv, base_rate):
    """市级时序先证特征（部署合法，同 v2：严格正滞后 + 结案<=通知）。"""
    n = len(yv)
    prior_n = np.zeros(n, dtype=np.int64)
    prior_death = np.zeros(n, dtype=np.int64)
    notif_1y = np.zeros(n, dtype=np.int64)

    um, gidx = np.unique(mun, return_inverse=True)
    order = np.argsort(gidx, kind='stable')
    bounds = np.searchsorted(gidx[order], np.arange(len(um) + 1))

    for g in range(len(um)):
        rows = order[bounds[g]:bounds[g + 1]]
        t = notif_day[rows]
        o = np.argsort(t, kind='stable')
        rows, t = rows[o], t[o]
        hi = np.searchsorted(t, t, side='right')
        lo = np.searchsorted(t, t - 365, side='left')
        notif_1y[rows] = hi - lo - 1
        e = enc_day[rows]
        yd = yv[rows]
        v = (e < MISSING_ENCERRA_SENTINEL // 2) & (e > t)
        ev, ydv = e[v], yd[v]
        if len(ev):
            oe = np.argsort(ev, kind='stable')
            ev, ydv = ev[oe], ydv[oe]
            pos = np.searchsorted(ev, t, side='right')
            prior_n[rows] = pos
            cd = np.concatenate(([0], np.cumsum(ydv)))
            prior_death[rows] = cd[pos]

    out = pd.DataFrame({'prior_n': prior_n, 'prior_death': prior_death,
                        'notif_1y': notif_1y})
    out['prior_rate'] = ((out['prior_death'] + K_SHRINK * base_rate)
                         / (out['prior_n'] + K_SHRINK))
    out['log_notif_1y'] = np.log1p(out['notif_1y'].values)
    return out


def build_ind_matrix(d):
    num = np.column_stack([
        np.clip(pd.to_numeric(d['AGE_YEARS'], errors='coerce')
                .fillna(d['AGE_YEARS'].median()).values, 0, 100),
        np.clip(pd.to_numeric(d['NU_CONTATO'], errors='coerce')
                .fillna(0).values, 0, 10)]).astype(np.float32)
    cat_df = pd.DataFrame({
        c: d[c].astype(str).str.strip().replace('', 'UNK')
        for c in BIN_AGRAV + CAT_FEATS})
    enc = OneHotEncoder(handle_unknown='ignore')
    Z = enc.fit_transform(cat_df).astype(np.float32)
    return sparse.hstack([sparse.csr_matrix(num), Z], format='csr')


def stratified_munip_subsample(mun, y, cap, rng):
    um, gidx = np.unique(mun, return_inverse=True)
    keep = np.zeros(len(y), dtype=bool)
    for g in range(len(um)):
        rows = np.where(gidx == g)[0]
        if len(rows) <= cap:
            keep[rows] = True
            continue
        pos = rows[y[rows] == 1]
        neg = rows[y[rows] == 0]
        n_pos = int(round(cap * len(pos) / len(rows)))
        n_pos = min(n_pos, len(pos))
        n_neg = cap - n_pos
        if n_neg > len(neg):
            n_neg = len(neg)
            n_pos = cap - n_neg
        keep[rng.choice(pos, size=n_pos, replace=False)] = True
        keep[rng.choice(neg, size=n_neg, replace=False)] = True
    return np.where(keep)[0]


def make_models(seed):
    models = {'LR': LogisticRegression(max_iter=2000, C=1.0,
                                       solver='lbfgs', random_state=seed)}
    if HAS_LGBM:
        models['LGBM'] = LGBMClassifier(
            n_estimators=400, learning_rate=0.06, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=seed, verbose=-1)
    return models


def icc_anova(y, groups):
    """单因素随机效应 ICC(1)：二元终点在组内的聚集强度。"""
    df = pd.DataFrame({'y': y, 'g': groups})
    k = df['g'].nunique()
    n = len(df)
    if k < 2 or n <= k:
        return float('nan')
    grand = df['y'].mean()
    grp = df.groupby('g')['y'].agg(['mean', 'count'])
    ssb = float((grp['count'] * (grp['mean'] - grand) ** 2).sum())
    ssw = float(((df['y'] - df['g'].map(grp['mean'])) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    n0 = (n - float((grp['count'] ** 2).sum()) / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    if denom <= 0:
        return float('nan')
    return float((msb - msw) / denom)


# ---------------------------------------------------------------------------
# Part 1：SINAN 市规模分层实验
# ---------------------------------------------------------------------------

def run_sinan_strata():
    d = pd.read_parquet(
        os.path.join(PROC, 'sinan', 'tubebra_2001_2019.parquet'),
        columns=BIN_AGRAV + CAT_FEATS + ['AGE_YEARS', 'NU_CONTATO',
                                         'ID_MUNICIP', 'DT_NOTIFIC',
                                         'DT_ENCERRA', 'SITUA_ENCE'])
    s = d['SITUA_ENCE'].astype(str)
    d = d[s.isin(['1', '3', '4'])].copy()
    y = d['SITUA_ENCE'].astype(str).isin(['3', '4']).astype(int).values
    base_rate = float(y.mean())
    print('analysis set %d, all-cause death %.2f%%' % (len(y),
                                                       100 * base_rate))

    mun = d['ID_MUNICIP'].astype(str).str.strip().values
    notif_day, _ = _day_number(d['DT_NOTIFIC'].values)
    enc_day_raw, enc_ok = _day_number(d['DT_ENCERRA'].values)
    enc_day = np.where(enc_ok, enc_day_raw,
                       MISSING_ENCERRA_SENTINEL).astype(np.int64)

    prior = build_prior_features(mun, notif_day, enc_day, y, base_rate)
    X_ind = build_ind_matrix(d)
    del d

    # 市规模分层（全量行数定层——聚集强度是数据属性，不受建模采样影响）
    mun_counts = pd.Series(mun).value_counts()
    stratum_of_mun = pd.Series(index=mun_counts.index, dtype=object)
    stratum_of_mun[mun_counts < SMALL_MAX] = 'small'
    stratum_of_mun[(mun_counts >= SMALL_MAX)
                   & (mun_counts < MID_MAX)] = 'mid'
    stratum_of_mun[mun_counts >= MID_MAX] = 'large'
    stratum = stratum_of_mun.reindex(pd.Index(mun)).values

    res = {}
    for layer in ['small', 'mid', 'large']:
        rows_l = np.where(stratum == layer)[0]
        n_mun_l = int(len(np.unique(mun[rows_l])))
        icc_l = icc_anova(y[rows_l], mun[rows_l])
        print('\n=== layer %s: %d rows, %d munips, ICC %.4f ==='
              % (layer, len(rows_l), n_mun_l, icc_l))

        # 层内建模采样
        rng = np.random.default_rng(0)
        mun_l = mun[rows_l]
        y_l = y[rows_l]
        sub_l = stratified_munip_subsample(mun_l, y_l, MUNIP_CAP, rng)
        if len(sub_l) > LAYER_MAX_ROWS:
            um_l = np.unique(mun_l[sub_l])
            keep_m = rng.choice(um_l, size=LAYER_MAX_ROWS // MUNIP_CAP,
                                replace=False)
            sub_l = sub_l[np.isin(mun_l[sub_l], keep_m)]
        y_s, mun_s = y_l[sub_l], mun_l[sub_l]
        prior_s = prior.iloc[rows_l[sub_l]].reset_index(drop=True)
        X_ind_s = X_ind[rows_l[sub_l]]
        print('model rows %d (%.1f%% events), %d munips'
              % (len(sub_l), 100 * y_s.mean(),
                 len(np.unique(mun_s))))

        prior_cols = OUTCOME_BLOCK + DENSITY_BLOCK
        Pz = ((prior_s[prior_cols] - prior_s[prior_cols].mean())
              / prior_s[prior_cols].std().replace(0, 1.0))
        P = sparse.csr_matrix(Pz.values.astype(np.float32))
        Xs = {'ind': X_ind_s,
              'ind_all': sparse.hstack([X_ind_s, P], format='csr')}

        auroc = {m: {a: [] for a in Xs} for m in make_models(0)}
        oof_store = {}
        for seed in range(N_SEEDS):
            cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                      random_state=seed)
            preds = {m: {a: np.zeros(len(y_s)) for a in Xs}
                     for m in make_models(seed)}
            for tr, te in cv.split(Xs['ind'], y_s, groups=mun_s):
                for mname, m in make_models(seed).items():
                    for arm, X in Xs.items():
                        m.fit(X[tr], y_s[tr])
                        preds[mname][arm][te] = m.predict_proba(X[te])[:, 1]
            for mname in preds:
                for arm in Xs:
                    auroc[mname][arm].append(
                        float(roc_auc_score(y_s, preds[mname][arm])))
            if seed == 0:
                oof_store = preds
            print('seed %d: LR %.4f/%.4f | LGBM %.4f/%.4f'
                  % (seed, auroc['LR']['ind'][-1],
                     auroc['LR']['ind_all'][-1],
                     auroc['LGBM']['ind'][-1] if HAS_LGBM else -1,
                     auroc['LGBM']['ind_all'][-1] if HAS_LGBM else -1))

        # seed0 市级簇 bootstrap ΔCI
        um_s, gidx_s = np.unique(mun_s, return_inverse=True)
        group_rows = [np.where(gidx_s == g)[0] for g in range(len(um_s))]
        boot_rng = np.random.default_rng(1)
        deltas = {m: [] for m in oof_store}
        for b in range(N_BOOT):
            gs = boot_rng.integers(0, len(um_s), size=len(um_s))
            rows_b = np.concatenate([group_rows[g] for g in gs])
            yy = y_s[rows_b]
            if len(np.unique(yy)) < 2:
                continue
            for mname, arm_preds in oof_store.items():
                deltas[mname].append(
                    roc_auc_score(yy, arm_preds['ind_all'][rows_b])
                    - roc_auc_score(yy, arm_preds['ind'][rows_b]))
            if (b + 1) % 250 == 0:
                print('  bootstrap %d/%d' % (b + 1, N_BOOT))

        res[layer] = {
            'n_rows_full': int(len(rows_l)),
            'n_munips': int(n_mun_l),
            'mean_group_size_full': round(float(len(rows_l)) / n_mun_l, 1),
            'icc': round(icc_l, 5),
            'prior_n_median': float(prior_s['prior_n'].median()),
            'n_model_rows': int(len(sub_l)),
            'event_rate': round(100 * float(y_s.mean()), 2),
            'auroc_ind': {m: round(float(np.mean(v['ind'])), 4)
                          for m, v in auroc.items()},
            'auroc_ind_all': {m: round(float(np.mean(v['ind_all'])), 4)
                              for m, v in auroc.items()},
            'delta_per_seed': {
                m: [round(v['ind_all'][s] - v['ind'][s], 4)
                    for s in range(N_SEEDS)] for m, v in auroc.items()},
            'delta_bootstrap': {},
        }
        for mname in deltas:
            arr = np.array(deltas[mname])
            lo, hi = np.percentile(arr, [2.5, 97.5])
            res[layer]['delta_bootstrap'][mname] = {
                'mean': round(float(arr.mean()), 5),
                'ci95': [round(float(lo), 5), round(float(hi), 5)],
                'significant': bool(lo > 0),
            }
        print('layer %s Δ: LR %s | LGBM %s'
              % (layer,
                 res[layer]['delta_bootstrap']['LR'],
                 res[layer]['delta_bootstrap'].get('LGBM')))

    res['ALL'] = {'icc': round(icc_anova(y, mun), 5),
                  'n_rows_full': int(len(y)),
                  'mean_group_size_full': round(
                      float(len(y)) / len(np.unique(mun)), 1)}
    return res


# ---------------------------------------------------------------------------
# Part 2：HomeACF / PACTS 户级 ICC（Δ 引用归档）
# ---------------------------------------------------------------------------

def household_points():
    pts = {}
    # HomeACF（南非，感染终点）
    try:
        from tb_risk.validation.real_data_infection import (
            load_homeacf_contacts)
        h = load_homeacf_contacts()
        pts['homeacf_household'] = {
            'country': '南非', 'granularity': '户（record_id）',
            'endpoint': 'tst_pos10（TST≥10mm 感染）',
            'icc': round(icc_anova(
                h['tst_pos10'].values, h['record_id'].values), 5),
            'mean_group_size': round(
                float(len(h)) / h['record_id'].nunique(), 1),
            'delta_from': 'household_temporal_screening_20260825.json',
        }
        with open(os.path.join(
                PROC, 'household_temporal_screening_20260825.json'),
                encoding='utf-8') as f:
            acc = json.load(f)['design']['acceptance']['vs_ind_baseline']
        pts['homeacf_household']['delta'] = round(acc['mean'], 5)
        pts['homeacf_household']['delta_ci95'] = [
            round(x, 5) for x in acc['ci']]
        pts['homeacf_household']['ceiling_inconclusive'] = False
    except Exception as exc:  # 可选数据源缺失 → 显式降级（非静默）
        pts['homeacf_household'] = {'error': repr(exc)}
    # PACTS（马拉维，症状终点，天花板）
    try:
        from tb_risk.validation.real_data_pi import load_pacts_contacts
        p = load_pacts_contacts()
        pts['pacts_household'] = {
            'country': '马拉维', 'granularity': '户（id）',
            'endpoint': 'sx_3m（3 月症状，终点天花板 ind 0.993）',
            'icc': round(icc_anova(
                pd.to_numeric(p['sx_3m'], errors='coerce').values,
                p['id'].values), 5),
            'mean_group_size': round(
                float(len(p)) / p['id'].nunique(), 1),
            'delta_from': 'pacts_temporal_household_20260825.json',
        }
        with open(os.path.join(
                PROC, 'pacts_temporal_household_20260825.json'),
                encoding='utf-8') as f:
            dl = json.load(f)['deltas_cluster_bootstrap'][
                'exposure_prior:RF - ind:RF']
        pts['pacts_household']['delta'] = round(dl['mean'], 5)
        pts['pacts_household']['delta_ci95'] = [
            round(x, 5) for x in dl['bootstrap_ci']]
        pts['pacts_household']['ceiling_inconclusive'] = True
    except Exception as exc:
        pts['pacts_household'] = {'error': repr(exc)}
    return pts


# ---------------------------------------------------------------------------
# Part 3：散点图
# ---------------------------------------------------------------------------

def plot_curve(points, strata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fam in ['Microsoft YaHei', 'SimHei']:
        try:
            font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams['font.family'] = fam
            break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for name, p in points.items():
        if 'error' in p or 'delta' not in p:
            continue
        icc, dlt = p['icc'], p['delta'] * 1000
        ci = [c * 1000 for c in p.get('delta_ci95', [dlt, dlt])]
        grey = p.get('ceiling_inconclusive', False)
        ax.errorbar(icc, dlt, yerr=[[dlt - ci[0]], [ci[1] - dlt]],
                    fmt='x' if grey else 'o', markersize=11,
                    color='#888888' if grey else '#1f77b4',
                    ecolor='#888888' if grey else '#1f77b4',
                    capsize=4, linewidth=1.6,
                    label=name + ('（终点天花板）' if grey else ''))
    ax.set_xlabel('组内相关 ICC(1)（组内共享暴露比例代理：'
                  '终点在组内的聚集强度）')
    ax.set_ylabel('时序先证 ΔAUROC（×10⁻³，ind+先证 − ind）')
    ax.set_title('组粒度 → 时序先证效应量：户级 vs 市级（三国证据链）')
    ax.axhline(0, color='#cccccc', lw=0.8, ls='--')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print('saved', OUT_PNG)


def main():
    t0 = time.time()
    strata = run_sinan_strata()
    hh = household_points()

    # SINAN 全体 Δ：优先 v2（全因死亡）归档，缺省回退 code4 并注明
    sinan_all = {'source': None}
    for cand, note in [
            ('sinan_temporal_prior_v2_allcause_20260826.json', 'v2 全因死亡'),
            ('sinan_temporal_prior_20260826.json', 'code4 非 TB 死亡')]:
        fp = os.path.join(PROC, cand)
        if os.path.exists(fp):
            with open(fp, encoding='utf-8') as f:
                dv = json.load(f)['delta_vs_ind']
            sinan_all = {
                'source': cand,
                'delta_LR': dv['LR']['ind_all'],
                'delta_LGBM': dv['LGBM']['ind_all'],
                'icc': strata['ALL']['icc'],
            }
            break

    points = {}
    if sinan_all['source']:
        for layer in ['small', 'mid', 'large']:
            r = strata[layer]
            points['sinan_munip_' + layer] = {
                'country': '巴西', 'granularity': '市（ID_MUNICIP，%s 市）'
                % layer,
                'endpoint': '全因死亡（SITUA_ENCE∈{3,4}）',
                'icc': r['icc'],
                'mean_group_size': r['mean_group_size_full'],
                'delta': r['delta_bootstrap']['LGBM']['mean']
                if HAS_LGBM else r['delta_bootstrap']['LR']['mean'],
                'delta_ci95': r['delta_bootstrap']['LGBM']['ci95']
                if HAS_LGBM else r['delta_bootstrap']['LR']['ci95'],
                'delta_LR': r['delta_bootstrap']['LR'],
                'ceiling_inconclusive': False,
            }
    points.update(hh)

    # 单调性判定（主曲线点：排除天花板 inconclusive）
    eff = [(p['icc'], p['delta'], k) for k, p in points.items()
           if 'icc' in p and 'delta' in p
           and not p.get('ceiling_inconclusive')]
    eff.sort()
    if len(eff) >= 3:
        x = [e[0] for e in eff]
        yv = [e[1] for e in eff]
        rho = float(pd.Series(x).corr(pd.Series(yv), method='spearman'))
        monotonic = all(yv[i] < yv[i + 1] for i in range(len(yv) - 1))
    else:
        rho, monotonic = None, None

    res = {
        'date': '2026-08-26',
        'experiment': 'granularity_effect_curve_v1',
        'question': '组粒度（ICC 组内共享暴露比例）→ 时序先证效应量的'
                    '定量关系；疾控部署按哪一级分组机制才值钱',
        'strata_sinan': strata,
        'sinan_all_reference': sinan_all,
        'points': points,
        'monotonicity': {
            'spearman_icc_vs_delta': rho,
            'strictly_monotonic': monotonic,
            'note': '主曲线点 = 户级（HomeACF）+ 市级三层（SINAN）；'
                    'PACTS 因终点天花板（ind 0.993）列为灰色参考，'
                    '不参与单调性判定',
        },
        'protocol': {
            'sinan_endpoint': 'y = SITUA_ENCE∈{3,4}（全因死亡，'
                              'sinan_label_dictionary_20260826.json）',
            'strata': 'small <%d / mid %d-%d / large ≥%d 全量分析集每市行数'
                      % (SMALL_MAX, SMALL_MAX, MID_MAX, MID_MAX),
            'arms': 'ind vs ind_all（先证块 = 结局块 prior_n/prior_death/'
                    'prior_rate + 密度块 notif_1y/log_notif_1y；块分解'
                    '见 v2 归档）',
            'cv': '5 种子 × 5 折 StratifiedGroupKFold 市级分组',
            'ci': 'seed0 市级簇 bootstrap ×%d 配对差' % N_BOOT,
            'munip_cap': MUNIP_CAP, 'layer_max_rows': LAYER_MAX_ROWS,
            'icc': 'ICC(1) 单因素随机效应 ANOVA（全量分层计算，'
                   '不受建模采样影响）',
        },
    }
    res['total_seconds'] = round(time.time() - t0, 1)

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('saved', OUT_JSON)
    print('points:', json.dumps(
        {k: {kk: p.get(kk) for kk in ('icc', 'delta', 'mean_group_size')}
         for k, p in points.items() if 'icc' in p},
        ensure_ascii=False, indent=1))
    print('monotonicity:', res['monotonicity'])

    try:
        plot_curve(points, strata)
    except Exception as exc:
        print('plot failed:', repr(exc))
    return res


if __name__ == '__main__':
    main()
