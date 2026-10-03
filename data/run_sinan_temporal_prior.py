#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0（第六轮）：SINAN 市级时序先证机制迁移实验（决定性实验，2026-08-26）。

问题：HomeACF（南非 877 户，感染终点）验证的「时序先证」机制——
同户已筛查成员结果作为先证信号——能否迁移到巴西 SINAN 百万级、
死亡终点、市级粒度？

机制迁移的同构映射：
  HomeACF（时序家庭筛查）          SINAN（市级时序通报史）
  -------------------------       -----------------------------
  同户成员                         同市通报个案（5,424 个市）
  筛查顺序（随机）                 通报顺序（时间固定，不可重排）
  已筛查成员 TST 结果              已结案（DT_ENCERRA ≤ 当前 DT_NOTIFIC）
                                   个案的结局（治愈/TB 死亡）
  prior_n/prior_pos/prior_rate     prior_n/prior_death/prior_rate
  （位置累积）                     （结案时间 searchsorted 累积）

部署合法性（与 HomeACF 同一法理）：
  先证特征只使用「当前个案通知时点之前已结案」的同市结局
  （DT_ENCERRA_j <= DT_NOTIFIC_i，结案滞后中位 190 天）；
  分组 CV 按市分组（StratifiedGroupKFold，同市同折），
  测试折先证值引用的本市结局从未进入训练。
  通知时点立即可知的「同期同市已确诊者密度」（notif_1y，
  追溯 365 天通报计数）为用户指定的密度先证信号。

锚（科学上限，不可部署）：munip_rate_loo = 同市留一死亡率，
  判断市级聚集信号本身是否存在（区别「无聚集信号」与
  「有聚集但历史先证捕获不到」两种阴性机制）。

对照臂：ind（159 维基线，同 run_sinan_scale_training）/
  ind+outcome_block / ind+density_block / ind+all / prior_only。
模型：LR + LGBM（RF 在 125 万行成本过高，如实记录）。
统计：5 种子 × 5 折市级分组 CV 池化 OOF；种子 0 市级簇
  bootstrap（500 次重抽样）配对差 CI。
验收（同 ERASE-TB 移交标准）：ind+all − ind 的簇 bootstrap CI
  排除 0 且为正 → 机制复制（三国、双终点证据链闭合）。

归档：data/processed/sinan_temporal_prior_20260826.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

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

PROC = os.path.join(BASE, 'tb_risk', 'data', 'processed')

# 终点选项（第七轮字典修正后新增 allcause）：
#   code4    ：y = SITUA_ENCE=='4'（round-5/6 原终点，经字典裁定实为
#              「非 TB 死亡」，2001-05 段含全因死亡成分）——保持
#              向后兼容，输出 sinan_temporal_prior_20260826.json
#   allcause ：y = SITUA_ENCE∈{3,4}（全因死亡，修正后的稳健死亡终点）
#              ——输出 sinan_temporal_prior_v2_allcause_20260826.json
OUTCOME = 'allcause' if (len(sys.argv) > 1
                          and sys.argv[1] == '--allcause') else 'code4'
if OUTCOME == 'allcause':
    OUT = os.path.join(
        PROC, 'sinan_temporal_prior_v2_allcause_20260826.json')
else:
    OUT = os.path.join(PROC, 'sinan_temporal_prior_20260826.json')

# 与 run_sinan_scale_training.py 完全一致的特征定义
BIN_AGRAV = ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
             'AGRAVDROGA', 'AGRAVTABAC', 'POP_LIBER', 'POP_RUA',
             'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV']
CAT_FEATS = ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'FORMA', 'EXTRAPU1_N',
             'RAIOX_TORA', 'TESTE_TUBE', 'CULTURA_ES', 'HISTOPATOL',
             'TEST_MOLEC', 'BACILOSC_E', 'CS_GESTANT']

# 先证特征块
OUTCOME_BLOCK = ['prior_n', 'prior_death', 'prior_rate']
DENSITY_BLOCK = ['notif_1y', 'log_notif_1y']
K_SHRINK = 2.0            # 贝叶斯收缩（同 temporal_household DEFAULT k=2.0）
MUNIP_CAP = 150            # 每市采样上限（成本控制；先证构造仍用全量）
N_SEEDS = 5
N_FOLDS = 5
N_BOOT = 500
MISSING_ENCERRA_SENTINEL = 10 ** 9  # 无结案日期 → 永不进入任何先证池


# ----------------------------------------------------------------------------
# 先证特征构造（全量、向量化、部署合法）
# ----------------------------------------------------------------------------

def _day_number(date_str):
    """YYYYMMDD 字符串 → 天数整数（datetime64[D] 基准）；无效 → None 掩码。"""
    s = pd.Series(date_str).astype(str).str.strip()
    ok = s.str.fullmatch(r'\d{8}', na=False)
    dt = pd.to_datetime(s.where(ok), format='%Y%m%d', errors='coerce')
    days = dt.values.astype('datetime64[D]').astype(np.int64)
    return days, ok.values


def build_prior_features(mun, notif_day, enc_day, yv, base_rate):
    """市级时序先证特征（部署合法版本）+ LOO 锚。

    参数均为与个案对齐的数组；返回 DataFrame。
    合法性核心（结构性证明）：池成员资格 = 同市 且
    严格正滞后（notif_j < enc_j）且 enc_j <= notif_i，
    故 notif_j < notif_i——晚通知个案的结局不可能进入
    早通知个案的先证池（SINAN 负滞后录入错误已排除）。
    """
    n = len(yv)
    prior_n = np.zeros(n, dtype=np.int64)
    prior_death = np.zeros(n, dtype=np.int64)
    notif_1y = np.zeros(n, dtype=np.int64)
    loo_rate = np.full(n, base_rate, dtype=np.float64)

    um, gidx = np.unique(mun, return_inverse=True)
    order = np.argsort(gidx, kind='stable')
    bounds = np.searchsorted(gidx[order], np.arange(len(um) + 1))

    for g in range(len(um)):
        rows = order[bounds[g]:bounds[g + 1]]
        t = notif_day[rows]
        o = np.argsort(t, kind='stable')
        rows, t = rows[o], t[o]

        # 密度块：追溯 365 天同市通报计数（含当日，减去自身）
        hi = np.searchsorted(t, t, side='right')
        lo = np.searchsorted(t, t - 365, side='left')
        notif_1y[rows] = hi - lo - 1

        # 结局块：已结案（enc_day <= notif_i）同市个案的结局累积。
        # 池成员资格 = 严格正滞后（enc > notif）：SINAN 有 1.2% 负滞后
        # （录入错误，中位 -26,051 天）与 1.4% 零滞后记录，必须排除
        # 才能保持结构性合法证明 notif_j < encerra_j <= notif_i
        # => notif_j < notif_i（晚通知个案结局不可能进入早通知先证池）。
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

        # LOO 锚（科学上限，非部署特征）
        D, N = int(yd.sum()), len(yd)
        if N > 1:
            loo_rate[rows] = (D - yd) / (N - 1)

    out = pd.DataFrame({
        'prior_n': prior_n,
        'prior_death': prior_death,
        'notif_1y': notif_1y,
    })
    out['prior_rate'] = ((out['prior_death'] + K_SHRINK * base_rate)
                         / (out['prior_n'] + K_SHRINK))
    out['log_notif_1y'] = np.log1p(out['notif_1y'].values)
    out['munip_rate_loo'] = loo_rate
    return out


def legality_check(mun, notif_day, enc_day, yv, base_rate, rng, n_groups=20):
    """无未来泄漏实证检验：翻转晚通知个案的结局，
    早期个案的先证特征必须不变（household_temporal 同款检验）。"""
    um, gidx = np.unique(mun, return_inverse=True)
    sizes = np.bincount(gidx)
    cand = np.argsort(sizes)[::-1][:n_groups]
    checked = 0
    for g in cand:
        rows = np.where(gidx == g)[0]
        if len(rows) < 20:
            continue
        rows = rows[np.argsort(notif_day[rows])]  # 按通知时间排序
        late = rows[len(rows) // 2:]              # 后半段（晚通知）
        early = rows[:len(rows) // 2]              # 前半段（早通知）
        if len(early) == 0 or len(late) == 0:
            continue
        base = build_prior_features(mun, notif_day, enc_day, yv,
                                    base_rate).iloc[early]
        y2 = yv.copy()
        y2[late] = 1 - y2[late]                    # 翻转晚通知结局
        pert = build_prior_features(mun, notif_day, enc_day, y2,
                                     base_rate).iloc[early]
        cols = OUTCOME_BLOCK + DENSITY_BLOCK
        if not np.allclose(base[cols].values, pert[cols].values):
            return {'ok': False, 'n_checked': checked}
        checked += len(early)
    return {'ok': True, 'n_checked': checked,
            'note': '晚通知个案结局翻转不影响早通知个案先证特征（结构性合法）'}


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def load_analysis():
    """全量分析集（同 run_sinan_scale_training.build_matrix 口径）。

    终点按 OUTCOME 全局选项：code4 = 旧终点（{1,4} 内 y=4，
    字典修正后实为非 TB 死亡）；allcause = 全因死亡（{1,3,4} 内
    y∈{3,4}，修正后的稳健死亡终点）。
    """
    d = pd.read_parquet(
        os.path.join(PROC, 'sinan', 'tubebra_2001_2019.parquet'),
        columns=BIN_AGRAV + CAT_FEATS + ['AGE_YEARS', 'NU_CONTATO',
                                         'ID_MUNICIP', 'DT_NOTIFIC',
                                         'DT_ENCERRA', 'SITUA_ENCE'])
    keep = ['1', '3', '4'] if OUTCOME == 'allcause' else ['1', '4']
    d = d[d['SITUA_ENCE'].astype(str).isin(keep)].copy()
    s = d['SITUA_ENCE'].astype(str)
    y = (s.isin(['3', '4']) if OUTCOME == 'allcause'
         else (s == '4')).astype(int).values
    return d, y


def build_ind_matrix(d):
    """个体基线特征（159 维，同规模训练实验口径）。"""
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
    X = sparse.hstack([sparse.csr_matrix(num), Z], format='csr')
    return X


def stratified_munip_subsample(mun, y, cap, rng):
    """每市按结局分层采样上限 cap（保留全部小市；先证特征已按全量构造）。"""
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


ARMS = {
    'ind': [],
    'ind_outcome': OUTCOME_BLOCK,
    'ind_density': DENSITY_BLOCK,
    'ind_all': OUTCOME_BLOCK + DENSITY_BLOCK,
    'prior_only': '__prior_only__',
}


def main():
    t0 = time.time()
    res = {'date': '2026-08-26',
           'experiment': ('sinan_temporal_prior_v2_allcause'
                          if OUTCOME == 'allcause'
                          else 'sinan_temporal_prior_v1'),
           'question': ('HomeACF 时序先证机制能否迁移到 SINAN 百万级'
                        + ('全因死亡终点（字典修正版）' if OUTCOME == 'allcause'
                           else '死亡终点')),
           'outcome_definition': ('y = SITUA_ENCE∈{3,4}（全因死亡，'
                                  'sinan_label_dictionary_20260826.json '
                                  '裁定后的稳健终点）'
                                  if OUTCOME == 'allcause' else
                                  'y = SITUA_ENCE=="4"（round-5 原终点；'
                                  '字典裁定实为非 TB 死亡，2001-05 段'
                                  '含全因死亡成分）'),
           'models_available': {'lgbm': HAS_LGBM}}

    d, y = load_analysis()
    base_rate = float(y.mean())
    print('analysis set %d, death %.2f%%' % (len(y), 100 * base_rate))

    mun = d['ID_MUNICIP'].astype(str).str.strip().values
    notif_day, _ = _day_number(d['DT_NOTIFIC'].values)
    enc_day_raw, enc_ok = _day_number(d['DT_ENCERRA'].values)
    pool_eligible_pct = 100 * float(
        (enc_ok & (enc_day_raw > notif_day)).mean())
    enc_day = np.where(enc_ok, enc_day_raw,
                       MISSING_ENCERRA_SENTINEL).astype(np.int64)

    prior = build_prior_features(mun, notif_day, enc_day, y, base_rate)
    print('prior built: prior_n median %.0f p90 %.0f, zero-pool %.1f%%, '
          'notif_1y median %.0f'
          % (prior['prior_n'].median(), prior['prior_n'].quantile(0.9),
             100 * (prior['prior_n'] == 0).mean(),
             prior['notif_1y'].median()))

    res['legality_check'] = legality_check(
        mun, notif_day, enc_day, y, base_rate, np.random.default_rng(0))
    print('legality check:', res['legality_check'])

    X_ind = build_ind_matrix(d)
    del d
    n = len(y)

    # 采样（先证构造已用全量；采样仅为建模成本）
    rng = np.random.default_rng(0)
    sub = stratified_munip_subsample(mun, y, MUNIP_CAP, rng)
    y_s = y[sub]
    mun_s = mun[sub]
    prior_s = prior.iloc[sub].reset_index(drop=True)
    X_ind_s = X_ind[sub]
    print('subsample %d rows (%.1f%% events), %d munips'
          % (len(sub), 100 * y_s.mean(), len(np.unique(mun_s))))

    # 先证列标准化（z-score，单调线性变换，树模型不受影响）
    prior_cols = OUTCOME_BLOCK + DENSITY_BLOCK
    Pz = ((prior_s[prior_cols] - prior_s[prior_cols].mean())
          / prior_s[prior_cols].std().replace(0, 1.0))
    P = sparse.csr_matrix(Pz.values.astype(np.float32))
    X_prior_only = sparse.csr_matrix(Pz.values.astype(np.float32))

    Xs = {'ind': X_ind_s,
          'ind_outcome': sparse.hstack([X_ind_s, P[:, :3]], format='csr'),
          'ind_density': sparse.hstack([X_ind_s, P[:, 3:]], format='csr'),
          'ind_all': sparse.hstack([X_ind_s, P], format='csr'),
          'prior_only': X_prior_only}

    # LOO 锚（科学上限单变量）
    res['anchor'] = {
        'munip_rate_loo': {
            'auroc': float(roc_auc_score(y_s,
                                         prior_s['munip_rate_loo'].values)),
            'note': '同市留一死亡率（科学上限锚，非部署特征）：'
                    '判断市级聚集信号是否存在'},
    }
    print('anchor munip_rate_loo AUROC %.4f'
          % res['anchor']['munip_rate_loo']['auroc'])

    # 5 种子 × 5 折市级分组 CV 池化 OOF
    seeds = list(range(N_SEEDS))
    auroc = {m: {a: [] for a in ARMS} for m in ['LR', 'LGBM']}
    oof_store = {}   # seed 0 的池化 OOF（bootstrap 用）
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=seed)
        preds = {m: {a: np.zeros(len(y_s)) for a in ARMS}
                 for m in make_models(seed)}
        for tr, te in cv.split(Xs['ind'], y_s, groups=mun_s):
            for mname, m in make_models(seed).items():
                for arm, X in Xs.items():
                    m.fit(X[tr], y_s[tr])
                    preds[mname][arm][te] = m.predict_proba(X[te])[:, 1]
        for mname in preds:
            for arm in ARMS:
                auroc[mname][arm].append(
                    float(roc_auc_score(y_s, preds[mname][arm])))
        if seed == 0:
            oof_store = preds
        print('seed %d done: LR ind %.4f all %.4f | LGBM ind %.4f all %.4f'
              % (seed, auroc['LR']['ind'][-1], auroc['LR']['ind_all'][-1],
                 auroc['LGBM']['ind'][-1] if HAS_LGBM else -1,
                 auroc['LGBM']['ind_all'][-1] if HAS_LGBM else -1))

    res['auroc_by_seed'] = {
        m: {a: [round(v, 4) for v in vs] for a, vs in arms.items()}
        for m, arms in auroc.items()}

    # 种子 0 市级簇 bootstrap（配对差 CI）
    um_s, gidx_s = np.unique(mun_s, return_inverse=True)
    group_rows = [np.where(gidx_s == g)[0] for g in range(len(um_s))]
    boot_rng = np.random.default_rng(1)
    deltas = {m: {a: [] for a in ARMS if a != 'ind'}
              for m in oof_store}
    for b in range(N_BOOT):
        gs = boot_rng.integers(0, len(um_s), size=len(um_s))
        rows = np.concatenate([group_rows[g] for g in gs])
        yy = y_s[rows]
        if len(np.unique(yy)) < 2:
            continue
        for mname, arm_preds in oof_store.items():
            p_ind = arm_preds['ind'][rows]
            for arm in ARMS:
                if arm == 'ind':
                    continue
                deltas[mname][arm].append(
                    roc_auc_score(yy, arm_preds[arm][rows])
                    - roc_auc_score(yy, p_ind))
        if (b + 1) % 100 == 0:
            print('bootstrap %d/%d' % (b + 1, N_BOOT))

    res['delta_vs_ind'] = {}
    for mname in deltas:
        res['delta_vs_ind'][mname] = {}
        for arm in deltas[mname]:
            arr = np.array(deltas[mname][arm])
            per_seed = [round(auroc[mname][arm][s] - auroc[mname]['ind'][s], 4)
                        for s in range(N_SEEDS)]
            lo, hi = np.percentile(arr, [2.5, 97.5])
            res['delta_vs_ind'][mname][arm] = {
                'mean': round(float(arr.mean()), 4),
                'ci95': [round(float(lo), 4), round(float(hi), 4)],
                'per_seed': per_seed,
                'seeds_positive': int(sum(p > 0 for p in per_seed)),
                'significant': bool(lo > 0),
            }

    # 判定
    key_arm = 'ind_all'
    sig = [res['delta_vs_ind'][m][key_arm]['significant']
           for m in res['delta_vs_ind']]
    anchor_auroc = res['anchor']['munip_rate_loo']['auroc']
    all_pos = all(res['delta_vs_ind'][m][key_arm]['mean'] > 0
                  for m in res['delta_vs_ind'])
    if all(sig):
        verdict = 'REPLICATED'
        endp = ('全因死亡' if OUTCOME == 'allcause' else '非 TB 死亡')
        note = ('时序先证机制在百万级、%s终点、拉美人群上复制：'
                '三国（南非/巴西+巴基斯坦方向对照）、双终点'
                '（感染+死亡）证据链闭合。' % endp)
    elif anchor_auroc < 0.55:
        verdict = 'NO_CLUSTERING_SIGNAL'
        note = ('市级 LOO 锚接近 0.5：该数据几乎不存在市级结局聚集信号，'
                '机制在此粒度不可检验（非机制否证）。')
    elif all_pos:
        verdict = 'DIRECTION_POSITIVE_NOT_SIGNIFICANT'
        note = ('方向一致为正但簇 bootstrap CI 含 0：机制方向迁移成功，'
                '幅度未达显著（死亡终点聚集强度弱于感染终点）。')
    else:
        verdict = 'NOT_REPLICATED'
        note = ('机制未迁移到死亡终点市级粒度：先证增量不显著或为负。'
                '机制边界 = 时序先证可能仅在感染终点/家庭粒度成立。')
    res['verdict'] = {'type': verdict, 'note': note,
                      'anchor_auroc': round(anchor_auroc, 4)}

    res['protocol'] = {
        'groups': 'ID_MUNICIP 市级分组（同市同折，测试折先证值引用的'
                  '本市结局从未进入训练）',
        'prior_legality': '池成员 = 同市 且 notif_j < DT_ENCERRA_j <= '
                          'DT_NOTIFIC_i（严格正滞后排除 1.2% 负滞后与'
                          '1.4% 零滞后录入错误记录的池资格；'
                          '结案滞后中位 190 天）',
        'k_shrinkage': K_SHRINK, 'base_rate': round(base_rate, 4),
        'munip_cap': MUNIP_CAP, 'n_seeds': N_SEEDS, 'n_folds': N_FOLDS,
        'n_bootstrap': N_BOOT,
        'models': 'LR + LGBM（RF 于 125 万行成本过高，如实记录）',
        'isomorphism': 'HomeACF 同户已筛查成员结果 → SINAN 同市已结案'
                       '个案结局；筛查顺序随机 → 通报顺序时间固定'
                       '（顺序包络不适用，如实记录差异）',
    }
    res['data'] = {
        'n_analysis_full': int(n),
        'events_full': int(y.sum()),
        'n_subsample': int(len(sub)),
        'events_subsample': int(y_s.sum()),
        'prevalence_pct': round(100 * float(y_s.mean()), 2),
        'n_munip': int(len(um_s)),
        'encerra_valid_pct': round(100 * float(enc_ok.mean()), 2),
        'pool_eligible_pct': round(pool_eligible_pct, 2),
        'prior_n_zero_pct': round(
            100 * float((prior_s['prior_n'] == 0).mean()), 2),
        'prior_n_median': float(prior_s['prior_n'].median()),
        'notif_1y_median': float(prior_s['notif_1y'].median()),
    }
    res['total_seconds'] = round(time.time() - t0, 1)

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('saved', OUT)
    print('VERDICT:', verdict)
    return res


if __name__ == '__main__':
    main()
