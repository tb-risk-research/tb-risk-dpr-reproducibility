#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""已知真相模拟研究（2026-09-29，JCE v1(1) 外审第 3 点）。

外审要求：论文声称 battery 可推广到病房/学校/诊所等序贯场景，但实证
核心是单一无真实筛查顺序的队列；要求补"已知真相"模拟——分别生成
"只有家庭聚集""只有时间/结果返回效应""两者都有""两者都没有"四种
情景，改变家庭大小、阳性率与检查顺序，报告 battery 判定正确率与误判
模式；否则收窄定位。本脚本执行该模拟（全部结果为 simulated scenario）。

DGP（风格化家庭接触者队列，与主稿管线同构）：
  户结构：G 户（N≈2000），户大小 small=clip(1+Poi(1.5),1,6) /
    large=clip(2+Poi(2.5),2,10)；户内筛查顺序 random（均匀置换）或
    age_sorted（年长先筛）。
  协变量（exposure 块，8 列）：x1-x3 ~ N(0,1) 个体；xh1-xh2 ~ N(0,1)
    户级共享；age ~ U(5,85)；sex ~ Bern(0.5)；hiv ~ Bern(0.08)。
  结局（沿筛查顺序逐户顺序生成）：
    logit P(y_t=1) = beta0 + beta'x_t + sigma_u·z_h + gamma·1{户内已有先筛阳性}
  z_h ~ N(0,1) 户随机效应（聚集机制）；触发项（时序机制：一次已返回
  阳性揭示/抬升后续成员风险，单向——未来结局不进入先筛成员的生成）。
  beta0 按配置二分校准至边际 pi（公共随机数，单调二分）。

四情景（sigma_u, gamma）：
  S1 clustering-only (sigma*, 0)   S2 temporal-only (0, gamma*)
  S3 both (sigma*, gamma*)         S4 neither (0, 0)
  sigma*/gamma* 在参考配置（pi=0.13, large, random）校准：主增益
  ep-exp ≈ 0.07（对齐真实数据 +0.0701）；校准先于全网格（calibrate
  模式），常量冻结后运行，结果披露于归档。

臂（9，全部折内无泄漏口径——rate 回退 NaN 由训练折阳性率填补）：
  exp | ep(先证块) | fut(未来块) | past1/fut1/past2/fut2（k=1,2 配平
  窗口，与 run_samecohort_controls.windowed_block 同一口径）|
  permG_ep / permW_ep（全局/户内置换结局上重建的先证块）。

估计量（与主稿统一口径同构）：Δ̄ = 逐种子配对差均值（5 种子 CV 重播）；
CI = 户级 cluster bootstrap ×300（条件于折外预测，同一重抽套用全部
种子，百分位 2.5/97.5；快通道向量化秩和 AUROC，与 sklearn 逐数据集
 Gate |Δ|<0.002 校验，失败回退慢通道）。

判定规则（预固定，先于全网格运行）：
  D1 增益存在：CI(ep-exp) 下界 > 0
  D2 顺序特异成分：CI(past1-fut1)、CI(past2-fut2)、CI(ep-fut) 任一
     下界 > 0（past 更 informative；S1 下三对照假阳性 ~7% 为 battery
     固有多重性，如实报告）
  verdict：!D1 → none；D1&!D2 → clustering_consistent；D1&D2 →
     order_specific
  真相映射：S4→none；S1→clustering_consistent；S2/S3→order_specific
  正确率 = 判定与真相一致的 replicate 比例；误判分类学：
    S1 false_order（D2 误发）；S2/S3 underpowered（!D1）/
    misattributed_clustering（D1&!D2）；S4 false_gain（D1 误发）。

网格：4 情景 × pi{0.05,0.13,0.30} × 户大小{small,large} × 顺序
{random,age_sorted} × 10 replicates（LR 主评分器）+ 参考配置 RF 复核
（4 情景 × 3 replicates）。

诚实边界：
  - 模拟检验的是 battery 判定逻辑在已知 DGP 下的判定正确率；不证明
    真实 HomeACF 的 DGP 与任一情景匹配；
  - 触发式时序与方差成分聚集是风格化机制；真实机制（剂量反应/共享
    脆弱性/传播链）可能不同，方向性结论限于本 DGP 族；
  - 主评分器 LR（折内标准化 L2，与 battery B④ 同款）；RF 仅参考配置
    复核——判定逻辑与评分器解耦，但效应幅度可能随评分器变化；
  - beta0 二分命中边际 pi（±20% 相对 Gate）；sigma*/gamma* 仅在参考
    配置校准，其余网格点不重校准（真实场景增益幅度天然变化）；
  - 本模拟为外审后补设计（post-hoc design, pre-fixed verdict rules），
    非预注册分析；校准在判定规则冻结后、全网格前运行并完整披露；
  - bootstrap 条件于折外预测（与主稿同一条件性口径），不覆盖重拟合
    不确定性；replicate 间差异覆盖数据集级抽样变异。

用法：
    python data/run_known_truth_simulation.py calibrate  # σ/γ 网格校准
    python data/run_known_truth_simulation.py smoke      # 2 情景 × 2 replicates
    python data/run_known_truth_simulation.py            # 全网格 + RF 复核
输出：
    data/processed/sop_knowntruth_sim_YYYYMMDD.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))
sys.path.insert(0, HERE)

import run_samecohort_controls as scc   # noqa: E402  (windowed_block/permute_outcome)
import run_sop_future_placebo as futp   # noqa: E402  (future_features)

from tb_risk.validation.household_temporal import prior_features  # noqa: E402
from tb_risk.validation.real_data_pi import _group_cv_indices, _make_model  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else 'full'
assert MODE in ('calibrate', 'smoke', 'full'), MODE

# ------------------------------------------------------------------ 常量 --
N_TARGET = 2725            # 每数据集目标行数（对齐真实 HomeACF n）
G_BISECT = 4000            # beta0 二分试点户数
N_SPLITS = 5
N_SEEDS_SIM = 10           # 每数据集 CV 重播种子数
N_BOOTSTRAP = 500
R_REPLICATES = 10          # 每配置 replicate 数据集数（smoke=2）
RF_REPS = 3                # RF 复核 replicates
PI_GRID = (0.05, 0.13, 0.30)
HH_MODES = ('small', 'large')
ORDER_MODES = ('random', 'age_sorted')
SCENARIOS = ('S1', 'S2', 'S3', 'S4')
REF = {'pi': 0.13, 'hh': 'large', 'order': 'random'}   # 参考配置

# σ/γ 由 calibrate 模式确定后冻结（编辑此两行）
# calibrate 2026-09-29：σ=1.5 → S1 gain 0.0765（past1−fut1≈0）；
# γ=1.6 → S2 gain 0.0697（past1−fut1 +0.0068）；S4 exp AUROC 0.6579
SIGMA_STAR = 1.65          # 2026-10-01 fine calibration: S1 gain +0.0715
GAMMA_STAR = 2.0           # 2026-10-01 calibration: S2 gain +0.0786
SIGMA_GRID = (1.55, 1.60, 1.65, 1.70, 1.75)
GAMMA_GRID = (0.8, 1.2, 1.6, 2.0, 2.5)
TARGET_GAIN = 0.07         # 校准目标（对齐真实 ep-exp 0.0701）

BETA = {'x1': 0.35, 'x2': 0.25, 'x3': 0.20, 'xh1': 0.30, 'xh2': 0.20,
        'age_c': 0.012, 'sex': 0.15, 'hiv': 0.40}
COV_COLS = ['x1', 'x2', 'x3', 'xh1', 'xh2', 'age', 'sex', 'hiv']

DATA_BASE, PERM_BASE, BOOT_BASE, BISECT_BASE = 30000, 61000, 62000, 70100

ARMS = ['exp', 'ep', 'fut', 'past1', 'fut1', 'past2', 'fut2',
        'permG_ep', 'permW_ep']
ARM_IDX = {a: i for i, a in enumerate(ARMS)}
CONTRASTS = [
    ('gain', 'ep', 'exp'),
    ('fut_gain', 'fut', 'exp'),
    ('order_k1', 'past1', 'fut1'),
    ('order_k2', 'past2', 'fut2'),
    ('order_full', 'ep', 'fut'),
    ('perm_global', 'permG_ep', 'exp'),
    ('perm_within', 'permW_ep', 'exp'),
]
BLOCK_COLS = {'ep': 'prior', 'fut': 'future'}
PREFIX = {'past1': 'p1', 'fut1': 'f1', 'past2': 'p2', 'fut2': 'f2',
          'permG_ep': 'pg', 'permW_ep': 'pw'}

OUT = os.path.join(HERE, 'processed',
                   'sop_knowntruth_sim_%s.json' % time.strftime('%Y%m%d'))

_EPS = 1e-9


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(np.asarray(y, dtype=int),
                                   np.asarray(s, dtype=float)))
    except ValueError:
        return float('nan')


# ------------------------------------------------------------------ DGP --
def _draw_households(rng, hh_mode, n_target):
    if hh_mode == 'small':
        lam, lo, hi = 1.5, 1, 6
    else:
        lam, lo, hi = 2.5, 2, 10
    e_size = float(np.mean(np.clip(lo + rng.poisson(lam, 200000), lo, hi)))
    e_size = min(max(e_size, 1.5), 8.0)
    g = max(50, int(round(n_target / e_size)))
    sizes = np.clip(lo + rng.poisson(lam, g), lo, hi).astype(int)
    hh = np.repeat(np.arange(g), sizes)
    return hh, sizes


def _draw_base(rng, hh_mode, n_target, order_mode):
    """CRN 基础抽取（beta0/sigma/gamma 之外的全体随机量）。"""
    hh, sizes = _draw_households(rng, hh_mode, n_target)
    n = len(hh)
    g = len(sizes)
    x1, x2, x3 = rng.randn(n), rng.randn(n), rng.randn(n)
    xh1_all, xh2_all = rng.randn(g), rng.randn(g)
    xh1, xh2 = xh1_all[hh], xh2_all[hh]
    age = rng.uniform(5, 85, n)
    sex = (rng.rand(n) < 0.5).astype(float)
    hiv = (rng.rand(n) < 0.08).astype(float)
    z_h = rng.randn(g)
    unif = rng.rand(n)
    eta_rest = (BETA['x1'] * x1 + BETA['x2'] * x2 + BETA['x3'] * x3
                + BETA['xh1'] * xh1 + BETA['xh2'] * xh2
                + BETA['age_c'] * (age - 45.0) + BETA['sex'] * sex
                + BETA['hiv'] * hiv)
    # 户内筛查顺序（random=均匀置换；age_sorted=年长先筛）
    order_pos = np.zeros(n, dtype=float)
    for h in range(g):
        idx = np.where(hh == h)[0]
        if order_mode == 'random':
            pos = rng.permutation(len(idx))
        else:
            pos = np.argsort(-age[idx], kind='stable')
        order_pos[idx[pos]] = np.arange(len(idx))
    # 户内按筛查顺序的行索引序列（顺序生成用）
    seq = []
    for h in range(g):
        idx = np.where(hh == h)[0]
        seq.append(idx[np.argsort(order_pos[idx], kind='stable')])
    cov = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'xh1': xh1, 'xh2': xh2,
                        'age': age, 'sex': sex, 'hiv': hiv})
    return {'hh': hh, 'sizes': sizes, 'cov': cov, 'z_h': z_h, 'unif': unif,
            'eta_rest': eta_rest, 'order_pos': order_pos, 'seq': seq}


def _gen_y(base, beta0, sigma, gamma):
    """顺序结局生成（触发式时序 + 户随机效应）。"""
    eta = beta0 + base['eta_rest'] + sigma * base['z_h'][base['hh']]
    unif = base['unif']
    y = np.zeros(len(eta), dtype=int)
    for seq in base['seq']:
        trig = 0
        for i in seq:
            if unif[i] < _sigmoid(eta[i] + gamma * trig):
                y[i] = 1
                trig = 1
    return y


_BETA0_CACHE = {}


def solve_beta0(pi, hh_mode, order_mode, sigma, gamma):
    """beta0 二分（CRN，单调）命中边际 pi。"""
    key = (pi, hh_mode, order_mode, sigma, gamma)
    if key in _BETA0_CACHE:
        return _BETA0_CACHE[key]
    rng = np.random.RandomState(
        BISECT_BASE + (len(_BETA0_CACHE) * 7919) % 100000)
    base = _draw_base(rng, hh_mode, G_BISECT, order_mode)
    lo, hi = -9.0, 5.0
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if _gen_y(base, mid, sigma, gamma).mean() > pi:
            hi = mid
        else:
            lo = mid
    beta0 = 0.5 * (lo + hi)
    _BETA0_CACHE[key] = beta0
    return beta0


def scen_params(scenario):
    if SIGMA_STAR is None or GAMMA_STAR is None:
        raise RuntimeError('SIGMA_STAR/GAMMA_STAR 未冻结（先运行 calibrate）')
    return {'S1': (SIGMA_STAR, 0.0), 'S2': (0.0, GAMMA_STAR),
            'S3': (SIGMA_STAR, GAMMA_STAR), 'S4': (0.0, 0.0)}[scenario]


# ------------------------------------------------------------ 特征块 --
def _nan_fallback(pf):
    """回退率置 NaN（折内填补交给评分器），与 scc._prior_from 同款。"""
    n_ = pf['prior_n'].to_numpy(dtype=float)
    p_ = pf['prior_pos'].to_numpy(dtype=float)
    pf = pf.copy()
    pf['prior_rate'] = np.where(n_ > 0, p_ / np.maximum(n_, 1.0), np.nan)
    return pf


def _nan_fallback_fut(ff):
    n_ = ff['future_n'].to_numpy(dtype=float)
    p_ = ff['future_pos'].to_numpy(dtype=float)
    ff = ff.copy()
    ff['future_rate'] = np.where(n_ > 0, p_ / np.maximum(n_, 1.0), np.nan)
    return ff


def build_data(cov, y, hh, order_pos, rng_perm):
    """全部 9 臂特征矩阵（唯一列名）。"""
    df_f = pd.DataFrame({'record_id': hh, 'tst_pos10': y})
    order_s = pd.Series(order_pos)
    pf = _nan_fallback(prior_features(df_f, order_s))
    ff = _nan_fallback_fut(futp.future_features(df_f, order_s))
    blocks = [pf, ff]
    for k in (1, 2):
        for d, tag in (('past', 'p%d' % k), ('future', 'f%d' % k)):
            wb = scc.windowed_block(y, hh, order_pos, k, d)
            wb.columns = [tag + c.split('_', 1)[1] for c in wb.columns]
            blocks.append(wb)
    y_pg = scc.permute_outcome(y, hh, rng_perm, within_household=False)
    y_pw = scc.permute_outcome(y, hh, rng_perm, within_household=True)
    for y_p, tag in ((y_pg, 'pg'), (y_pw, 'pw')):
        df_p = pd.DataFrame({'record_id': hh, 'tst_pos10': y_p})
        pb = _nan_fallback(prior_features(df_p, order_s))
        pb.columns = [tag + c.split('_', 1)[1] for c in pb.columns]
        blocks.append(pb)
    data = pd.concat([cov.reset_index(drop=True)] + blocks, axis=1)
    return data


ARM_COLS = {
    'exp': COV_COLS,
    'ep': COV_COLS + ['prior_n', 'prior_pos', 'prior_rate', 'prior_screened'],
    'fut': COV_COLS + ['future_n', 'future_pos', 'future_rate',
                       'future_screened'],
    'past1': COV_COLS + ['p1n', 'p1pos', 'p1rate', 'p1screened'],
    'fut1': COV_COLS + ['f1n', 'f1pos', 'f1rate', 'f1screened'],
    'past2': COV_COLS + ['p2n', 'p2pos', 'p2rate', 'p2screened'],
    'fut2': COV_COLS + ['f2n', 'f2pos', 'f2rate', 'f2screened'],
    'permG_ep': COV_COLS + ['pgn', 'pgpos', 'pgrate', 'pgscreened'],
    'permW_ep': COV_COLS + ['pwn', 'pwpos', 'pwrate', 'pwscreened'],
}


# -------------------------------------------------------------- 评分器 --
def _fit_oof(cols, folds, y, data, seed, kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X = data[cols].to_numpy(dtype=float)
    nanmask = np.isnan(X)
    oof = np.zeros(len(y))
    for tr, te in folds:
        Xf = np.where(nanmask, float(y[tr].mean()), X) if nanmask.any() else X
        if kind == 'lr':
            m = Pipeline([('sc', StandardScaler()),
                          ('lr', LogisticRegression(C=1.0, max_iter=2000))])
        else:
            m = _make_model('random_forest', seed)
        m.fit(Xf[tr], y[tr])
        oof[te] = m.predict_proba(Xf[te])[:, 1]
    return oof


def fast_auroc_cols(M, yb):
    """逐列秩和 AUROC（无并列校正；数据集级 Gate 校验后使用）。"""
    n = M.shape[0]
    order = np.argsort(M, axis=0, kind='mergesort')
    ranks = np.empty_like(M)
    ranks[order, np.arange(M.shape[1])] = np.arange(n)[:, None]
    pos = yb == 1
    npos = int(pos.sum())
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return np.full(M.shape[1], np.nan)
    s = ranks[pos].sum(axis=0)
    return (s - npos * (npos + 1.0) / 2.0) / (npos * nneg)


# ------------------------------------------------------------ 单数据集 --
def run_dataset(scenario, pi, hh_mode, order_mode, rep_index, kind='lr',
                n_boot=None):
    """一个 replicate 数据集：生成 → 9 臂 → 统一口径 → 判定。"""
    sigma, gamma = scen_params(scenario) \
        if scenario in SCENARIOS else (None, None)
    if sigma is None:   # calibrate 模式外部直接传参
        raise RuntimeError('calibrate 模式不应调用 run_dataset')
    rng = np.random.RandomState(DATA_BASE + rep_index)
    rng_perm = np.random.RandomState(PERM_BASE + rep_index)
    beta0 = solve_beta0(pi, hh_mode, order_mode, sigma, gamma)
    base = _draw_base(rng, hh_mode, N_TARGET, order_mode)
    y = _gen_y(base, beta0, sigma, gamma)
    hh, cov, order_pos = base['hh'], base['cov'], base['order_pos']
    data = build_data(cov, y, hh, order_pos, rng_perm)

    # NaN 仅允许出现在 rate 列（折内填补语义）
    rate_cols = [c for cols in ARM_COLS.values() for c in cols
                 if c.endswith('rate')]
    for c in data.columns:
        if c not in rate_cols:
            assert not data[c].isna().any(), '非 rate 列出现 NaN: %s' % c

    folds_list = [_group_cv_indices(hh, y, n_splits=N_SPLITS, seed=s)
                  for s in range(N_SEEDS_SIM)]
    scores = {a: [] for a in ARMS}
    for s in range(N_SEEDS_SIM):
        folds = folds_list[s]
        for a in ARMS:
            scores[a].append(_fit_oof(ARM_COLS[a], folds, y, data, s, kind))

    # 点估计（sklearn 金标准）
    au = np.zeros((len(ARMS), N_SEEDS_SIM))
    for i, a in enumerate(ARMS):
        for s in range(N_SEEDS_SIM):
            au[i, s] = _auroc(y, scores[a][s])
    score_mat = np.column_stack([scores[a][s] for a in ARMS
                                 for s in range(N_SEEDS_SIM)])

    # 快通道校验（原始行上 vs sklearn）
    fast_full = fast_auroc_cols(score_mat, y)
    gate_fast = float(np.nanmax(np.abs(
        fast_full.reshape(len(ARMS), N_SEEDS_SIM) - au)))
    use_fast = gate_fast < 0.002

    nb = n_boot if n_boot else N_BOOTSTRAP
    hh_ids = np.unique(hh)
    hh_rows = [np.where(hh == g)[0] for g in hh_ids]
    g_n = len(hh_ids)
    rng_b = np.random.RandomState(BOOT_BASE + rep_index)
    boot = {name: [] for name, _, _ in CONTRASTS}
    for _ in range(nb):
        pick = rng_b.randint(0, g_n, g_n)
        rows = np.concatenate([hh_rows[j] for j in pick])
        if use_fast:
            aub = fast_auroc_cols(score_mat[rows], y[rows])
        else:
            aub = np.array([_auroc(y[rows], score_mat[rows][:, j])
                            for j in range(score_mat.shape[1])])
        if np.isnan(aub).any():
            continue
        aub = aub.reshape(len(ARMS), N_SEEDS_SIM)
        for name, a1, a0 in CONTRASTS:
            boot[name].append(float(np.mean(
                aub[ARM_IDX[a1]] - aub[ARM_IDX[a0]])))

    contrasts = {}
    for name, a1, a0 in CONTRASTS:
        d_bar = float(np.mean(au[ARM_IDX[a1]] - au[ARM_IDX[a0]]))
        arr = np.asarray(boot[name])
        lo, hi = (float(np.percentile(arr, 2.5)),
                  float(np.percentile(arr, 97.5))) if len(arr) else \
            (float('nan'), float('nan'))
        contrasts[name] = {'mean': round(d_bar, 4),
                           'ci95': [round(lo, 4), round(hi, 4)],
                           'ci_excludes_zero': bool(lo > 0 or hi < 0)}

    gain = contrasts['gain']['mean']
    d1 = bool(contrasts['gain']['ci95'][0] > 0)
    d2_which = [n for n in ('order_k1', 'order_k2', 'order_full')
                if contrasts[n]['ci95'][0] > 0]
    d2 = bool(d2_which)
    verdict = 'none' if not d1 else (
        'order_specific' if d2 else 'clustering_consistent')
    truth = {'S4': 'none', 'S1': 'clustering_consistent',
             'S2': 'order_specific', 'S3': 'order_specific'}[scenario]
    r_fut = (contrasts['fut_gain']['mean'] / gain) if gain > 1e-6 else None
    ret_g = (contrasts['perm_global']['mean'] / gain) if gain > 1e-6 else None
    ret_w = (contrasts['perm_within']['mean'] / gain) if gain > 1e-6 else None
    return {
        'scenario': scenario, 'pi': pi, 'hh': hh_mode, 'order': order_mode,
        'rep': rep_index, 'kind': kind, 'n': int(len(y)),
        'households': int(g_n), 'achieved_pi': round(float(y.mean()), 4),
        'arm_auroc': {a: round(float(np.mean(au[ARM_IDX[a]])), 4)
                      for a in ARMS},
        'contrasts': contrasts, 'R_fut': None if r_fut is None else round(
            r_fut, 3),
        'retention_perm_global': None if ret_g is None else round(ret_g, 3),
        'retention_perm_within': None if ret_w is None else round(ret_w, 3),
        'd1': d1, 'd2': d2, 'd2_fired': d2_which,
        'verdict': verdict, 'truth': truth, 'correct': verdict == truth,
        'fastboot_gate': round(gate_fast, 5), 'fastboot_used': bool(use_fast),
    }


# ------------------------------------------------------------ calibrate --
def calibrate():
    t0 = time.time()
    out = {'sigma': {}, 'gamma': {}, 's4_exp_auroc': None}
    arms_c = ['exp', 'ep', 'fut', 'past1', 'fut1']
    for sigma in SIGMA_GRID:
        gains, ok1s = [], []
        for r in range(3):
            beta0 = solve_beta0(REF['pi'], REF['hh'], REF['order'], sigma, 0.0)
            rng = np.random.RandomState(DATA_BASE + 5000 + r)
            base = _draw_base(rng, REF['hh'], N_TARGET, REF['order'])
            y = _gen_y(base, beta0, sigma, 0.0)
            rng_perm = np.random.RandomState(PERM_BASE + 5000 + r)
            data = build_data(base['cov'], y, base['hh'],
                              base['order_pos'], rng_perm)
            aurocs = {a: [] for a in arms_c}
            for s in range(N_SEEDS_SIM):
                folds = _group_cv_indices(base['hh'], y, N_SPLITS, seed=s)
                for a in arms_c:
                    aurocs[a].append(
                        _auroc(y, _fit_oof(ARM_COLS[a], folds, y, data,
                                           s, 'lr')))
            gains.append(float(np.mean(aurocs['ep'])
                               - np.mean(aurocs['exp'])))
            ok1s.append(float(np.mean(aurocs['past1'])
                              - np.mean(aurocs['fut1'])))
        out['sigma'][sigma] = {'gain': round(float(np.mean(gains)), 4),
                               'past1_minus_fut1': round(
                                   float(np.mean(ok1s)), 4)}
        print('sigma=%.1f  gain(ep-exp)=%.4f  past1-fut1=%+.4f  (%ds)'
              % (sigma, out['sigma'][sigma]['gain'],
                 out['sigma'][sigma]['past1_minus_fut1'], time.time() - t0))
    for gamma in GAMMA_GRID:
        gains, ok1s = [], []
        for r in range(3):
            beta0 = solve_beta0(REF['pi'], REF['hh'], REF['order'], 0.0, gamma)
            rng = np.random.RandomState(DATA_BASE + 6000 + r)
            base = _draw_base(rng, REF['hh'], N_TARGET, REF['order'])
            y = _gen_y(base, beta0, 0.0, gamma)
            rng_perm = np.random.RandomState(PERM_BASE + 6000 + r)
            data = build_data(base['cov'], y, base['hh'],
                              base['order_pos'], rng_perm)
            aurocs = {a: [] for a in arms_c}
            for s in range(N_SEEDS_SIM):
                folds = _group_cv_indices(base['hh'], y, N_SPLITS, seed=s)
                for a in arms_c:
                    aurocs[a].append(
                        _auroc(y, _fit_oof(ARM_COLS[a], folds, y, data,
                                           s, 'lr')))
            gains.append(float(np.mean(aurocs['ep'])
                               - np.mean(aurocs['exp'])))
            ok1s.append(float(np.mean(aurocs['past1'])
                              - np.mean(aurocs['fut1'])))
        out['gamma'][gamma] = {'gain': round(float(np.mean(gains)), 4),
                               'past1_minus_fut1': round(
                                   float(np.mean(ok1s)), 4)}
        print('gamma=%.1f  gain(ep-exp)=%.4f  past1-fut1=%+.4f  (%ds)'
              % (gamma, out['gamma'][gamma]['gain'],
                 out['gamma'][gamma]['past1_minus_fut1'], time.time() - t0))
    # S4 参照：exp 臂判别力
    beta0 = solve_beta0(REF['pi'], REF['hh'], REF['order'], 0.0, 0.0)
    rng = np.random.RandomState(DATA_BASE + 7000)
    base = _draw_base(rng, REF['hh'], N_TARGET, REF['order'])
    y = _gen_y(base, beta0, 0.0, 0.0)
    exp_a = np.mean([_auroc(y, _fit_oof(ARM_COLS['exp'],
                                        _group_cv_indices(base['hh'], y,
                                                          N_SPLITS, seed=s),
                                        y, pd.concat(
                                            [base['cov']], axis=1), s, 'lr'))
                     for s in range(N_SEEDS_SIM)])
    out['s4_exp_auroc'] = round(float(exp_a), 4)
    print('S4 reference exp-arm AUROC = %.4f' % exp_a)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(OUT.replace('.json', '_calibrate.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'mode': 'calibrate', 'target_gain': TARGET_GAIN,
                   'reference': REF, 'grid': out}, f, ensure_ascii=False,
                  indent=1)
    print('校准归档: %s' % OUT.replace('.json', '_calibrate.json'))


# ------------------------------------------------------------- 汇总 --
def _rate(vals):
    return round(float(np.mean(vals)), 3) if len(vals) else None


def _num(vals):
    v = [np.nan if x is None else x for x in vals]
    return round(float(np.nanmean(v)), 3) if v else None


def summarize(results):
    by_key = {}
    for r in results:
        by_key.setdefault((r['scenario'], r['pi'], r['hh'], r['order']),
                          []).append(r)
    cfg_summary = {}
    for key, rs in sorted(by_key.items()):
        n = len(rs)
        cfg_summary['|'.join(map(str, key))] = {
            'n_replicates': n,
            'correct_rate': _rate([x['correct'] for x in rs]),
            'd1_rate': _rate([x['d1'] for x in rs]),
            'd2_rate': _rate([x['d2'] for x in rs]),
            'mean_gain': round(float(np.mean(
                [x['contrasts']['gain']['mean'] for x in rs])), 4),
            'mean_order_k1': round(float(np.mean(
                [x['contrasts']['order_k1']['mean'] for x in rs])), 4),
            'mean_R_fut': _num([x['R_fut'] for x in rs]),
            'mean_achieved_pi': round(float(np.mean(
                [x['achieved_pi'] for x in rs])), 4),
        }
    tax = {
        'S1_false_order': _rate([r['d2'] for r in results
                                 if r['scenario'] == 'S1']),
        'S4_false_gain': _rate([r['d1'] for r in results
                                if r['scenario'] == 'S4']),
        'S2S3_underpowered': _rate(
            [not r['d1'] for r in results
             if r['scenario'] in ('S2', 'S3')]),
        'S2S3_misattributed_clustering': _rate(
            [r['d1'] and not r['d2'] for r in results
             if r['scenario'] in ('S2', 'S3')]),
    }
    by_scen = {s: _rate([r['correct'] for r in results
                         if r['scenario'] == s])
               for s in sorted(set(r['scenario'] for r in results))}
    return cfg_summary, tax, by_scen


def main():
    t0 = time.time()
    assert SIGMA_STAR is not None and GAMMA_STAR is not None, \
        '先运行 calibrate 并冻结 SIGMA_STAR/GAMMA_STAR'
    n_reps = 2 if MODE == 'smoke' else R_REPLICATES
    scen_list = ('S1', 'S2') if MODE == 'smoke' else SCENARIOS
    counter = 0
    results = []
    for scen in scen_list:
        for pi in ((0.13,) if MODE == 'smoke' else PI_GRID):
            for hh_m in (('large',) if MODE == 'smoke' else HH_MODES):
                for order_m in (('random',) if MODE == 'smoke'
                                else ORDER_MODES):
                    for _ in range(n_reps):
                        results.append(run_dataset(
                            scen, pi, hh_m, order_m, counter, 'lr',
                            n_boot=100 if MODE == 'smoke' else None))
                        counter += 1
                    last = results[-1]
                    print('%s pi=%.2f %s %s: correct %d/%d (%ds)' % (
                        scen, pi, hh_m, order_m,
                        sum(x['correct'] for x in results[-n_reps:]),
                        n_reps, time.time() - t0))
    rf_check = []
    if MODE == 'full':
        for scen in SCENARIOS:
            for _ in range(RF_REPS):
                rf_check.append(run_dataset(
                    scen, REF['pi'], REF['hh'], REF['order'], counter, 'rf'))
                counter += 1
            print('RF check %s done (%ds)' % (scen, time.time() - t0))

    cfg_summary, tax, by_scen = summarize(results)
    def _med(vals):
        return float(np.median(vals)) if len(vals) else float('nan')

    lr_ref = [r for r in results if r['pi'] == REF['pi']
              and r['hh'] == REF['hh'] and r['order'] == REF['order']]
    gates = {
        'cal_gain_s1': abs(float(np.mean([r['contrasts']['gain']['mean']
                                          for r in lr_ref
                                          if r['scenario'] == 'S1']))
                           - TARGET_GAIN) <= 0.02,
        'cal_gain_s2': abs(float(np.mean([r['contrasts']['gain']['mean']
                                          for r in lr_ref
                                          if r['scenario'] == 'S2']))
                           - TARGET_GAIN) <= 0.02,
        'exp_auroc_range': all(0.58 <= r['arm_auroc']['exp'] <= 0.72
                               for r in lr_ref),
        's1_fut_retained': _med([r['R_fut'] for r in results
                                 if r['scenario'] == 'S1'
                                 and r['R_fut'] is not None]) >= 0.80,
        'permg_destroys': _med(
            [r['retention_perm_global'] for r in results
             if r['scenario'] in ('S1', 'S2', 'S3')
             and r['retention_perm_global'] is not None]) <= 0.20,
        'achieved_pi': _med(
            [abs(r['achieved_pi'] - r['pi']) / r['pi'] for r in results]
        ) <= 0.20,
        'fastboot_all': all(r['fastboot_used'] for r in results + rf_check),
    }
    archive = {
        'date': time.strftime('%Y-%m-%d %H:%M:%S'), 'mode': MODE,
        'name': 'sop_knowntruth_sim',
        'status': '已知真相模拟（simulated scenario）：4 情景 × pi × 户大小 '
                  '× 顺序 × %d replicates，LR 主评分器%s；判定规则预固定'
                  % (n_reps, ' + RF 参考复核' if MODE == 'full' else ''),
        'design': {
            'dgp': 'logit P(y_t) = beta0 + beta·x + sigma·z_h + '
                   'gamma·1{户内先筛阳性}；触发式单向时序 + 户随机效应聚集',
            'sigma_star': SIGMA_STAR, 'gamma_star': GAMMA_STAR,
            'beta': BETA, 'n_target': N_TARGET,
            'grid': {'scenarios': scen_list, 'pi': PI_GRID,
                     'hh': HH_MODES, 'order': ORDER_MODES},
            'estimator': 'Δ̄ = %d 种子配对差均值；CI = 户级 cluster '
                         'bootstrap ×%d（条件于 OOF，同重抽套全部种子）'
                         % (N_SEEDS_SIM, N_BOOTSTRAP),
            'verdict_rules_pre_fixed': {
                'D1': 'CI(ep-exp) 下界>0', 'D2': 'CI(past1-fut1)/CI('
                'past2-fut2)/CI(ep-fut) 任一下界>0',
                'mapping': 'S4→none; S1→clustering_consistent; '
                           'S2/S3→order_specific'},
            'calibration': 'sigma*/gamma* 于参考配置校准至 gain≈%.2f '
                           '（calibrate 归档随附）' % TARGET_GAIN,
        },
        'gates': gates,
        'results_lr': results,
        'results_rf_check': rf_check,
        'config_summary': cfg_summary,
        'misjudgment_taxonomy': tax,
        'correct_rate_by_scenario': by_scen,
        'boundaries': [
            '模拟检验 battery 判定逻辑的判定正确率，不证明真实队列 DGP '
            '与任一情景匹配',
            '触发式时序与方差成分聚集为风格化机制，方向性结论限于本 DGP 族',
            'LR 主评分器；RF 仅参考配置复核；效应幅度可能随评分器变化',
            '外审后补设计（post-hoc design, pre-fixed verdict rules），'
            '非预注册分析',
            '全部读数为 simulated scenario',
        ],
        'runtime_sec': round(time.time() - t0, 1),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(archive, f, ensure_ascii=False, indent=1)
    print('gates:', json.dumps(gates, ensure_ascii=False))
    print('correct_rate_by_scenario:', by_scen)
    print('taxonomy:', {k: (round(v, 3) if v is not None else None)
                        for k, v in tax.items()})
    print('归档: %s  (runtime %.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    if MODE == 'calibrate':
        calibrate()
    else:
        main()
