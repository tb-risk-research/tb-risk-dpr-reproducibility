#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOP 时序先证深化（S1，2026-09-07）：EB 自适应 k + π 失配曲线 + 交互结构检验。

地位：SOP 是本线程唯一已判决的真实数据增益（household_temporal_
screening_v1：exposure_prior 0.7142 vs ind 0.6428，+0.0585 CI
[+0.031,+0.087]）；部署近似（logit 位移 σ(logit(p)+w(r̂−π))，网格
校准 w=7/k=2）已兑现 RF 参考锚的 100.6%（temporal_deployment 归档）。
本脚本回答两个深化问题：

Q1（k/π 校准，C5 条款）：
  固定 k=2 来自 seed-0 网格 {0.5,1,2,4,8} 选优；π 用全队列真值。
  C5 条款指出 π/k 失配会稀释增益但从未定量。
  - H-K1（EB 自适应）：Beta-binomial MoM 的 k̂=(1−ICC)/ICC（ICC=户内
    组内相关，训练折内估计）+ π̂（训练折率）替代固定 (k=2, π_true)，
    同 w=7 下 AUROC 不降（Δ≥0）；小户亚组（prior_n≤2）改善更大——
    EB 收缩的理论收益集中于小证据量户；
  - H-K2（π 失配稀释曲线）：π_assumed ∈ {0.05,...,0.50} 偏离真值
    0.132 越远，时序增益单调稀释——C5 的定量兑现，给 ERASE-TB 的
    π̂ 估计精度要求提供依据（协议已定 π̂=训练折估计）。

Q2（交互结构）：
  prior_only 0.6443 ≈ ind 0.6428 而组合 0.7142 ≫ 两者——"增益来自
  交互"是措辞不精确的：两个边际 AUROC 相等的信号可经**加性互补**
  组合超越各自（非交互）；交互的严格定义=组合超越边际的**最优加性
  组合**。三个预声明判据：
  - H-I1（加性充分性）：NB 式加性栈 score=logit(p_exp)+logit(p_prior)
    −logit(π̂_fold) vs RF(exposure+prior)——差值 CI 含零 → 增量为
    加性互补，部署统一位移结构充分；不含零 → 存在真交互，部署公式
    需交互项；
  - H-I2（显式交互不增益）：预声明交互块（prior×age/hiv/sex，部署
    配方 v3 stack 变量对齐）加入 RF 后 Δ≈0——RF 隐式交互充分；
  - H-I3（户内通道结构，**冒烟测试后修正版预声明**）：原预声明
    "统一位移构造上保持户内序 → within(dep_fixed)≡within(exposure)
    恒等"被 seed-0 冒烟测试**证伪**（within: exposure 0.596 /
    exposure_prior 0.360 / dep_fixed 0.242）——固定顺序口径下位移
    是逐行位置依赖（各行用自己的先筛证据），非户级常量；且先证块
    存在**自排除反相关**（prior_pos 聚合他人结果：单阳性户中阳性
    成员 prior_pos 恒 0，排其后的阴性反而上调；多成员户内
    E[prior_pos|y=1]<E[prior_pos|y=0] 由留一恒等式给出）。修正后
    预声明：(a) 户内通道为结构性负贡献而非可忽略；(b) 反相关强度
    随户阳性数 M 分层——M=1 户（混合户主体）远强于 M≥2 户；(c)
    池化增益不受损（户内对占比 <1%，户间通道主导）。该发现为
    SOP 机制的结构性质（§8.4-P1 精化），非性能缺陷；序贯部署
    语义（剩余成员共享当前证据）不受影响。

分层同质性（H-I4，预声明）：时序增益在 age<15/≥15、HIV、sex、
prior_n≤2/>2 分层间报告（部署公式是否需要分层权重 w_s 的证据基础）。

协议（与 v1 逐位同构）：HomeACF tst_pos10，StratifiedGroupKFold(5)
by record_id × 20 种子（0-19），random 筛查顺序，冻结注册表 RF，
逐行平均池化 OOF；末种子户级 cluster bootstrap ×2000；逐种子 Δ
四栏（R5a 纪律）。锚臂（ind/exposure/prior_only/exposure_prior）
必须复现 v1 归档（sanity gate：|Δ复现|<0.002）。

诚实边界：
  - EB k̂/π̂ 与 w=7 均在同一队列评估（无独立外样本；与 v1 网格校准
    同款边界）——结论口径为"机制与结构"，跨人群迁移归 ERASE-TB；
  - π 失配曲线是回顾性模拟（真值已知场景下人为失配），非前瞻校准
    误差的实际分布；
  - 单队列 LTBI 终点；within-household AUROC 仅多成员户（户内需
    同时有阴阳）；
  - 交互块为预声明小集合（5 项），非穷尽搜索。

用法：
    python data/run_sop_deepening.py
输出：
    data/processed/sop_deepening_homeacf_YYYYMMDD.json
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

from tb_risk.validation.household_temporal import (  # noqa: E402
    assign_screening_order,
    prior_features,
)
from tb_risk.validation.real_data_infection import (  # noqa: E402
    feature_columns as inf_feature_columns,
    load_homeacf_contacts,
)
from tb_risk.validation.real_data_pi import (  # noqa: E402
    _cluster_bootstrap_delta,
    _group_cv_indices,
    _make_model,
)

OUT = os.path.join(HERE, 'processed',
                   'sop_deepening_homeacf_%s.json' % time.strftime('%Y%m%d'))

N_SEEDS = 20
N_SPLITS = 5
N_BOOTSTRAP = 2000
SEED_START = 0
DEP_W = 7.0            # v1 网格校准权重（temporal_deployment best_w）
DEP_K = 2.0            # v1 网格校准收缩
PI_MISMATCH_GRID = (0.05, 0.08, 0.10, 0.132, 0.20, 0.30, 0.50)
EB_W_GRID = (3.0, 5.0, 7.0, 9.0, 11.0, 13.0)   # 末种子包络（次要）
_PROB_EPS = 1e-6

COLS_PRIOR = ['prior_n', 'prior_pos', 'prior_rate', 'prior_screened']


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _PROB_EPS, 1.0 - _PROB_EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _auroc(y, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y, dtype=int),
                               np.asarray(scores, dtype=float)))


def eb_shrinkage_params(y_tr, g_tr):
    """训练折内 EB 收缩参数（Beta-binomial MoM ↔ 单因素 ANOVA ICC）。

    户率 r_h ~ Beta(kπ, k(1−π)) ⟺ 户内相关 ρ = 1/(k+1)
    ⟹ k̂ = (1−ICC)/ICC（ICC(1) 不等距单因素 MoM）。
    退化保护：MS_B≤MS_W 或多成员户<5 → k̂=100（近完全收缩）。

    Returns:
        (pi_hat, k_hat)
    """
    y_tr = np.asarray(y_tr, dtype=float)
    pi_hat = float(y_tr.mean())
    dfh = pd.DataFrame({'y': y_tr, 'g': np.asarray(g_tr)})
    agg = dfh.groupby('g')['y'].agg(['size', 'sum'])
    agg = agg[agg['size'] >= 2]
    h = len(agg)
    if h < 5:
        return pi_hat, 100.0
    n = agg['size'].to_numpy(dtype=float)
    pos = agg['sum'].to_numpy(dtype=float)
    r = pos / n
    n_tot = float(n.sum())
    grand = float(pos.sum()) / n_tot
    ms_b = float((n * (r - grand) ** 2).sum() / (h - 1))
    # 折内 SS 的闭式：Σ_h pos(n−pos)/n
    within_ss = float((pos * (n - pos) / n).sum())
    ms_w = within_ss / (n_tot - h) if n_tot > h else 0.0
    if ms_w <= 0 or ms_b <= ms_w:
        return pi_hat, 100.0
    n0 = (n_tot - float((n ** 2).sum()) / n_tot) / (h - 1)
    icc = (ms_b - ms_w) / (ms_b + (n0 - 1.0) * ms_w)
    icc = float(np.clip(icc, 1e-4, 0.99))
    k_hat = float(np.clip((1.0 - icc) / icc, 0.5, 100.0))
    return pi_hat, k_hat


def rf_oof(cols, folds, y, data, seed):
    """冻结注册表 RF 的 OOF 概率（与 household_temporal 同口径）。"""
    X = np.nan_to_num(data[cols].to_numpy(dtype=float), nan=0.0)
    oof = np.zeros(len(y))
    for tr, te in folds:
        m = _make_model('random_forest', seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def dep_scores(p_base, prior_n, prior_pos, pi, k, w):
    """部署公式：σ(logit(p_base) + w·(r̂−π))，r̂=(pos+kπ)/(n+k)。

    pi/k 可为标量或逐行向量（EB 口径）。
    """
    prior_n = np.asarray(prior_n, dtype=float)
    prior_pos = np.asarray(prior_pos, dtype=float)
    pi = np.broadcast_to(np.asarray(pi, dtype=float), prior_n.shape)
    k = np.broadcast_to(np.asarray(k, dtype=float), prior_n.shape)
    r_hat = (prior_pos + k * pi) / (prior_n + k)
    return np.clip(_sigmoid(_logit(p_base) + w * (r_hat - pi)),
                   _PROB_EPS, 1.0 - _PROB_EPS)


def within_hh_contribs(scores, y, groups):
    """逐户 (num_g, den_g)：户内 Mann-Whitney 对贡献（可加 → bootstrap 快）。

    仅计入多成员且户内阴阳皆有的户。
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        sg = scores[idx]
        yg = y[idx]
        pos = sg[yg == 1]
        neg = sg[yg == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diffs = pos[:, None] - neg[None, :]
        num = float((diffs > 0).sum()) + 0.5 * float((diffs == 0).sum())
        out[g] = (num, float(diffs.size))
    return out


def within_hh_auroc(contribs):
    num = sum(v[0] for v in contribs.values())
    den = sum(v[1] for v in contribs.values())
    return num / den if den > 0 else float('nan')


def within_hh_by_m(scores, y, groups):
    """按户阳性数 M 分层的户内 AUROC（H-I3b 机制验证）。

    M=1 户预测最强反相关（阳性者 prior_pos 恒 0——自排除）；
    M≥2 户反相关减弱（其他阳性可进入先证窗口）。
    """
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    hh_pos = pd.Series(y).groupby(groups).sum()
    contribs = within_hh_contribs(scores, y, groups)
    out = {}
    for label, sel in (('M_eq_1', lambda m: m == 1),
                       ('M_ge_2', lambda m: m >= 2)):
        sub = {g: v for g, v in contribs.items() if sel(hh_pos[g])}
        out[label] = {
            'within_auroc': round(within_hh_auroc(sub), 4)
            if sub else float('nan'),
            'n_households': len(sub),
        }
    return out


def within_pairs_share(y, groups):
    """户内 pos-neg 对占全部 pos-neg 对的比例（H-I3c：户间主导）。"""
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    dfh = pd.DataFrame({'y': y, 'g': groups})
    agg = dfh.groupby('g')['y'].agg(['size', 'sum'])
    pos = agg['sum'].to_numpy(dtype=float)
    neg = agg['size'].to_numpy(dtype=float) - pos
    within = float((pos * neg).sum())
    total = float(pos.sum() * neg.sum())
    return within / total if total > 0 else float('nan')


def bootstrap_within_delta(scores_a, scores_b, y, groups, n_boot, seed):
    """户内 AUROC Δ 的 cluster bootstrap（配对同一户重采样）。"""
    ca = within_hh_contribs(scores_a, y, groups)
    cb = within_hh_contribs(scores_b, y, groups)
    keys = sorted(set(ca) & set(cb))
    if not keys:
        return {'mean': float('nan'), 'bootstrap_ci': [float('nan')] * 2,
                'p_positive': float('nan'), 'ci_excludes_zero': False,
                'n_households': 0}
    rng = np.random.RandomState(seed)
    arr = np.array(keys)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(arr, size=len(arr), replace=True)
        na = sum(ca[g][0] for g in pick)
        da = sum(ca[g][1] for g in pick)
        nb = sum(cb[g][0] for g in pick)
        db = sum(cb[g][1] for g in pick)
        if da == 0 or db == 0:
            continue
        deltas.append(na / da - nb / db)
    deltas = np.asarray(deltas)
    return {
        'mean': float(np.mean(deltas)),
        'bootstrap_ci': [float(np.percentile(deltas, 2.5)),
                         float(np.percentile(deltas, 97.5))],
        'p_positive': float((deltas > 0).mean()),
        'ci_excludes_zero': bool(np.percentile(deltas, 2.5) > 0
                                 or np.percentile(deltas, 97.5) < 0),
        'n_households': len(keys),
    }


def stratum_cluster_bootstrap(oof_a, oof_b, y, groups, mask, n_boot, seed):
    """分层限定的 ΔAUROC cluster bootstrap（重采样户后限 mask 行）。"""
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    mask = np.asarray(mask, dtype=bool)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in pick])
        rows = rows[mask[rows]]
        if len(rows) == 0:
            continue
        yr = y[rows]
        if yr.sum() == 0 or yr.sum() == len(yr):
            continue
        try:
            deltas.append(_auroc(yr, oof_a[rows]) - _auroc(yr, oof_b[rows]))
        except ValueError:
            continue
    if not deltas:
        return {'mean': float('nan'), 'bootstrap_ci': [float('nan')] * 2,
                'p_positive': float('nan'), 'ci_excludes_zero': False}
    deltas = np.asarray(deltas)
    return {
        'mean': float(np.mean(deltas)),
        'bootstrap_ci': [float(np.percentile(deltas, 2.5)),
                         float(np.percentile(deltas, 97.5))],
        'p_positive': float((deltas > 0).mean()),
        'ci_excludes_zero': bool(np.percentile(deltas, 2.5) > 0
                                 or np.percentile(deltas, 97.5) < 0),
    }


def run_seed(seed, df, y, groups):
    """单种子全臂 OOF（随机顺序口径，与 v1 同构）。"""
    folds = _group_cv_indices(groups, y, n_splits=N_SPLITS, seed=seed)
    order = assign_screening_order(df, seed=seed, mode='random')
    pf = prior_features(df, order)
    data = pd.concat([df.reset_index(drop=True), pf], axis=1)

    # 逐折 EB 参数 + 逐行折归属
    fold_of = np.zeros(len(y), dtype=int)
    eb_params = []
    for f, (tr, te) in enumerate(folds):
        fold_of[te] = f
        eb_params.append(eb_shrinkage_params(y[tr], groups[tr]))

    prior_n = pf['prior_n'].to_numpy(dtype=float)
    prior_pos = pf['prior_pos'].to_numpy(dtype=float)

    # EB prior_rate（逐行用其所属测试折的训练折参数——部署者视角）
    pi_row = np.array([eb_params[f][0] for f in fold_of])
    k_row = np.array([eb_params[f][1] for f in fold_of])
    prior_rate_eb = (prior_pos + k_row * pi_row) / (prior_n + k_row)

    data['prior_rate_eb'] = prior_rate_eb

    # 预声明交互块（部署配方 v3 stack 变量对齐：age/hiv/sex）
    age10 = data['contact_age'].to_numpy(dtype=float) / 10.0
    data['it_rate_age'] = data['prior_rate'] * age10
    data['it_pos_age'] = data['prior_pos'] * age10
    data['it_rate_hiv'] = data['prior_rate'] * data['hiv_pos_h']
    data['it_pos_hiv'] = data['prior_pos'] * data['hiv_pos_h']
    data['it_rate_sex'] = data['prior_rate'] * data['contact_sex_m']
    cols_x = ['it_rate_age', 'it_pos_age', 'it_rate_hiv',
              'it_pos_hiv', 'it_rate_sex']

    cols_ind = inf_feature_columns('ind')
    cols_exp = inf_feature_columns('exposure')
    cols_eb = cols_exp + ['prior_n', 'prior_pos', 'prior_rate_eb',
                          'prior_screened']
    cols_x_full = cols_exp + COLS_PRIOR + cols_x

    oof = {
        'ind': rf_oof(cols_ind, folds, y, data, seed),
        'exposure': rf_oof(cols_exp, folds, y, data, seed),
        'prior_only': rf_oof(COLS_PRIOR, folds, y, data, seed),
        'exposure_prior': rf_oof(cols_exp + COLS_PRIOR, folds, y, data, seed),
        'exposure_prior_eb': rf_oof(cols_eb, folds, y, data, seed),
        'exposure_prior_x': rf_oof(cols_x_full, folds, y, data, seed),
    }

    # 部署公式臂（p_base = exposure OOF）
    pi_true = float(np.mean(y))
    oof['dep_static'] = oof['exposure']
    oof['dep_fixed'] = dep_scores(oof['exposure'], prior_n, prior_pos,
                                  pi_true, DEP_K, DEP_W)
    oof['dep_eb'] = dep_scores(oof['exposure'], prior_n, prior_pos,
                               pi_row, k_row, DEP_W)
    for pi_a in PI_MISMATCH_GRID:
        oof['dep_pi_%.3f' % pi_a] = dep_scores(
            oof['exposure'], prior_n, prior_pos, pi_a, DEP_K, DEP_W)

    # NB 加性栈（逐行训练折 π̂）
    logit_prior = _logit(oof['prior_only'])
    logit_exp = _logit(oof['exposure'])
    oof['stack_add'] = np.clip(
        _sigmoid(logit_exp + logit_prior - _logit(pi_row)),
        _PROB_EPS, 1.0 - _PROB_EPS)

    return {
        'seed': seed,
        'oof': oof,
        'pi_row': pi_row,
        'k_row': k_row,
        'prior_n': prior_n,
        'prior_pos': prior_pos,
        'eb_params': [{'fold': f, 'pi_hat': round(p, 4),
                       'k_hat': round(k, 2)}
                      for f, (p, k) in enumerate(eb_params)],
        'pi_true': pi_true,
    }


def per_seed_delta(seeds_results, arm_a, arm_b):
    """R5a 逐种子 Δ 四栏（池化 AUROC 差）。"""
    deltas = []
    for r in seeds_results:
        deltas.append(_auroc(r['y'], r['oof'][arm_a])
                      - _auroc(r['y'], r['oof'][arm_b]))
    deltas = np.asarray(deltas)
    return {
        'mean': float(np.mean(deltas)),
        'sd': float(np.std(deltas)),
        'min': float(np.min(deltas)),
        'max': float(np.max(deltas)),
        'share_positive': float((deltas > 0).mean()),
    }


def main():
    t0 = time.time()
    df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    print('HomeACF: n=%d events=%d households=%d' % (
        len(y), int(y.sum()), len(np.unique(groups))))

    seeds_results = []
    for s in range(SEED_START, SEED_START + N_SEEDS):
        r = run_seed(s, df, y, groups)
        r['y'] = y
        r['groups'] = groups
        seeds_results.append(r)
        aucs = {k: _auroc(y, v) for k, v in r['oof'].items()
                if not k.startswith('dep_pi_')}
        print('seed %2d: exp_prior=%.4f exp_prior_eb=%.4f dep_eb=%.4f '
              '(%.0fs)' % (
                  s, aucs['exposure_prior'], aucs['exposure_prior_eb'],
                  aucs['dep_eb'], time.time() - t0))

    last = seeds_results[-1]
    y_l, g_l = last['y'], last['groups']
    oof_l = last['oof']

    # ---- 锚臂复现 gate ----
    anchor_expect = {'ind': 0.6428, 'exposure': 0.6441,
                     'prior_only': 0.6443, 'exposure_prior': 0.7142}
    arm_mean = {a: float(np.mean([_auroc(r['y'], r['oof'][a])
                                  for r in seeds_results]))
                for a in anchor_expect}
    anchor_gate = {a: {'this_run': round(v, 4),
                       'v1_archive': anchor_expect[a],
                       'reproduced': bool(abs(v - anchor_expect[a]) < 0.002)}
                   for a, v in arm_mean.items()}
    print('\n=== 锚臂复现 gate ===')
    for a, g in anchor_gate.items():
        print('  %-16s %.4f vs %.4f  %s' % (
            a, g['this_run'], g['v1_archive'],
            'OK' if g['reproduced'] else 'FAIL'))

    # ---- 主对比（末种子 cluster bootstrap + 逐种子四栏）----
    contrast_pairs = [
        ('exposure_prior', 'ind', 'H0 anchor：v1 主增益复现'),
        ('dep_fixed', 'dep_static', '部署公式增益复现（v1 校准口径）'),
        ('dep_eb', 'dep_fixed', 'H-K1 主：EB(k̂,π̂) vs 固定(k=2,π_true)'),
        ('exposure_prior_eb', 'exposure_prior', 'H-K1 RF 臂：EB rate vs raw rate'),
        ('exposure_prior', 'stack_add', 'H-I1：RF vs NB 加性栈（交互必要性）'),
        ('exposure_prior_x', 'exposure_prior', 'H-I2：显式交互块增量'),
    ]
    ladder = {}
    for a, b, note in contrast_pairs:
        cb = _cluster_bootstrap_delta(oof_l[a], oof_l[b], y_l, g_l,
                                      n_bootstrap=N_BOOTSTRAP,
                                      seed=SEED_START)
        ladder['%s_minus_%s' % (a, b)] = {
            'mean': cb['mean'], 'bootstrap_ci': cb['bootstrap_ci'],
            'p_positive': cb['p_positive'],
            'ci_excludes_zero': cb['ci_excludes_zero'],
            'per_seed_delta': per_seed_delta(seeds_results, a, b),
            'note': note,
        }
        print('%-38s %+0.4f CI [%+0.4f,%+0.4f] per-seed share+=%.2f'
              % ('%s−%s' % (a, b), cb['mean'], *cb['bootstrap_ci'],
                 ladder['%s_minus_%s' % (a, b)]['per_seed_delta']
                 ['share_positive']))

    # ---- H-I3：户内通道（含 M 分层机制验证 + 户内对占比）----
    within = {}
    for arm in ('exposure', 'exposure_prior', 'exposure_prior_x',
                'dep_fixed', 'stack_add'):
        contribs = within_hh_contribs(oof_l[arm], y_l, g_l)
        within[arm] = {
            'within_auroc': round(within_hh_auroc(contribs), 4),
            'n_households': len(contribs),
            'mean': round(float(np.mean([
                within_hh_auroc(within_hh_contribs(r['oof'][arm],
                                                   r['y'], r['groups']))
                for r in seeds_results])), 4),
            'by_m': within_hh_by_m(oof_l[arm], y_l, g_l),
        }
    within_contrasts = {
        'exposure_prior_minus_exposure': bootstrap_within_delta(
            oof_l['exposure_prior'], oof_l['exposure'], y_l, g_l,
            N_BOOTSTRAP, SEED_START),
        'dep_fixed_minus_exposure': bootstrap_within_delta(
            oof_l['dep_fixed'], oof_l['exposure'], y_l, g_l,
            N_BOOTSTRAP, SEED_START),
    }
    pairs_share = {
        'within_pairs_share': round(within_pairs_share(y_l, g_l), 5),
        'note': '户内 pos-neg 对占全部对比例——池化 AUROC 由户间对主导',
    }
    print('\n=== H-I3 户内通道（多成员户，户内对占比 %.4f）===' %
          pairs_share['within_pairs_share'])
    for arm, v in within.items():
        print('  %-16s within=%.4f (20seed %.4f | M=1: %.4f n=%d | '
              'M≥2: %.4f n=%d)' % (
                  arm, v['within_auroc'], v['mean'],
                  v['by_m']['M_eq_1']['within_auroc'],
                  v['by_m']['M_eq_1']['n_households'],
                  v['by_m']['M_ge_2']['within_auroc'],
                  v['by_m']['M_ge_2']['n_households']))
    for k, v in within_contrasts.items():
        print('  %-38s %+0.4f CI [%+0.4f,%+0.4f]' % (
            k, v['mean'], *v['bootstrap_ci']))

    # ---- H-K2：π 失配曲线 ----
    pi_curve = {}
    for pi_a in PI_MISMATCH_GRID:
        key = 'dep_pi_%.3f' % pi_a
        aucs = [_auroc(r['y'], r['oof'][key]) for r in seeds_results]
        base = [_auroc(r['y'], r['oof']['dep_static'])
                for r in seeds_results]
        deltas = np.asarray(aucs) - np.asarray(base)
        pi_curve['%.3f' % pi_a] = {
            'auroc_mean': round(float(np.mean(aucs)), 4),
            'auroc_sd': round(float(np.std(aucs)), 4),
            'gain_vs_static_mean': round(float(np.mean(deltas)), 4),
            'gain_sd': round(float(np.std(deltas)), 4),
            'per_seed_share_positive': round(float((deltas > 0).mean()), 3),
        }
    print('\n=== H-K2 π 失配曲线（w=7, k=2, true π=%.3f）===' % last['pi_true'])
    for k, v in pi_curve.items():
        print('  π=%s  AUROC=%.4f  gain=%+.4f  share+=%.2f' % (
            k, v['auroc_mean'], v['gain_vs_static_mean'],
            v['per_seed_share_positive']))

    # ---- H-K1 附：小户/大户分层 ----
    prior_n_l = last['prior_n']
    strata_defs = {
        'age_lt15': (df['contact_age'].to_numpy() < 15),
        'age_ge15': (df['contact_age'].to_numpy() >= 15),
        'hiv_pos': (df['hiv_pos_h'].to_numpy() == 1),
        'hiv_neg': (df['hiv_pos_h'].to_numpy() == 0),
        'sex_m': (df['contact_sex_m'].to_numpy() == 1),
        'sex_f': (df['contact_sex_m'].to_numpy() == 0),
        'prior_n_le2': (prior_n_l <= 2),
        'prior_n_gt2': (prior_n_l > 2),
    }
    strata = {}
    for name, mask in strata_defs.items():
        mask = np.asarray(mask, dtype=bool)
        entry = {'n': int(mask.sum()),
                 'n_events': int(y[mask].sum())}
        for arm in ('exposure', 'exposure_prior', 'dep_fixed', 'dep_eb'):
            vals = [_auroc(r['y'][mask], r['oof'][arm][mask])
                    for r in seeds_results
                    if r['y'][mask].sum() > 0
                    and r['y'][mask].sum() < mask.sum()]
            entry[arm] = round(float(np.mean(vals)), 4) if vals else None
        entry['delta_exposure_prior_minus_exposure'] = \
            stratum_cluster_bootstrap(
                oof_l['exposure_prior'], oof_l['exposure'], y_l, g_l,
                mask, N_BOOTSTRAP, SEED_START)
        entry['delta_dep_eb_minus_dep_fixed'] = stratum_cluster_bootstrap(
            oof_l['dep_eb'], oof_l['dep_fixed'], y_l, g_l,
            mask, N_BOOTSTRAP, SEED_START)
        strata[name] = entry
    print('\n=== H-I4 分层（20seed 均值 AUROC）===')
    for name, e in strata.items():
        print('  %-12s n=%5d ev=%4d  exp=%.4f exp_prior=%.4f '
              'dep_fixed=%.4f dep_eb=%.4f' % (
                  name, e['n'], e['n_events'], e['exposure'],
                  e['exposure_prior'], e['dep_fixed'], e['dep_eb']))

    # ---- EB 包络（末种子 w 网格，次要）----
    eb_w_env = {}
    for w in EB_W_GRID:
        s_dep = dep_scores(oof_l['exposure'], last['prior_n'],
                           last['prior_pos'], last['pi_row'],
                           last['k_row'], w)
        eb_w_env['w_%.0f' % w] = round(_auroc(y_l, s_dep), 4)
    print('\n=== EB w 包络（末种子，in-sample 网格披露）===')
    print(' ', eb_w_env)

    # ---- EB 参数稳定性 ----
    eb_flat = [p for r in seeds_results for p in r['eb_params']]
    k_hats = [p['k_hat'] for p in eb_flat]
    pi_hats = [p['pi_hat'] for p in eb_flat]
    eb_summary = {
        'k_hat_median': float(np.median(k_hats)),
        'k_hat_iqr': [float(np.percentile(k_hats, 25)),
                      float(np.percentile(k_hats, 75))],
        'k_hat_min': float(np.min(k_hats)), 'k_hat_max': float(np.max(k_hats)),
        'pi_hat_median': float(np.median(pi_hats)),
        'note': '20 种子 × 5 折训练折内 MoM 估计；k=100 为退化保护'
                '（MS_B≤MS_W → 完全收缩）',
    }
    print('\n=== EB 参数 ===  k̂ median=%.2f IQR[%.2f,%.2f]  π̂ median=%.4f'
          % (eb_summary['k_hat_median'], *eb_summary['k_hat_iqr'],
             eb_summary['pi_hat_median']))

    # ---- 汇总归档 ----
    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'sop_deepening_homeacf_v1',
        'status': '探索性深化（预声明 H-K1/H-K2/H-I1/H-I2/H-I3/H-I4）；'
                  '结论口径=机制与结构，跨人群迁移归 ERASE-TB',
        'design': {
            'source': 'HomeACF（github petermacp/tstsa）',
            'endpoint': 'tst_pos10（TST≥10mm LTBI）',
            'n': int(len(y_l)), 'n_events': int(y_l.sum()),
            'n_households': int(len(np.unique(g_l))),
            'n_seeds': N_SEEDS, 'seed_start': SEED_START,
            'cv': 'StratifiedGroupKFold(%d) by household' % N_SPLITS,
            'ci': '户级 cluster bootstrap ×%d（末种子 OOF）' % N_BOOTSTRAP,
            'order_mode': 'random（部署期望口径）',
            'dep_formula': 'σ(logit(p_exp)+w·(r̂−π)), r̂=(pos+kπ)/(n+k), '
                           'w=%.1f（v1 网格校准）' % DEP_W,
            'eb': 'k̂=(1−ICC)/ICC（训练折 MoM），π̂=训练折率',
            'pi_true': round(last['pi_true'], 4),
            'anchor_gate': anchor_gate,
        },
        'hypotheses': {
            'H-K1': 'EB(k̂,π̂) ≥ 固定(k=2,π_true)，小户亚组改善更大',
            'H-K2': 'π 失配单调稀释时序增益（C5 定量）',
            'H-I1': 'RF − NB 加性栈 ≈ 0 → 加性互补（部署结构充分）',
            'H-I2': '显式交互块 Δ≈0 → RF 隐式交互充分',
            'H-I3': '（冒烟后修正）户内通道=自排除反相关负贡献，'
                    'M=1 户最强；池化增益由户间对主导（户内对占比<1%）',
            'H-I4': '时序增益分层同质性（部署 w 是否需分层）',
        },
        'ladder': ladder,
        'within_household': {'arms': within,
                             'contrasts': within_contrasts,
                             'pairs_share': pairs_share},
        'pi_mismatch_curve': pi_curve,
        'strata': strata,
        'eb_w_envelope_last_seed': eb_w_env,
        'eb_summary': eb_summary,
        'eb_params_per_seed_fold': [r['eb_params']
                                    for r in seeds_results],
        'per_seed_arm_auroc': [
            {'seed': r['seed'],
             **{a: round(_auroc(r['y'], v), 4)
                for a, v in r['oof'].items()
                if not a.startswith('dep_pi_')}}
            for r in seeds_results],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
