#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H-I3 泛化：自排除恒等式（Self-Exclusion Identity, SEI）形式化 + 概念验证。

地位：§8.11 H-I3 在 HomeACF（TB/TST）上发现的"先证块户内结构性反相关"
不是 TB 特有现象，而是**序贯揭示标签的簇内 screening 通用机制**。本脚本
把它升格为跨标签通用命题（类似 SOP 的 C1-C5 形式化路线），并在真实户
结构 × 非 TB 标签上做概念验证——泛化的主张是"机制源于构造而非疾病"，
故正确 PoC = 同一簇结构上换标签（HIV/糖尿病/吸烟/BCG 疤痕/性别），
而非寻找新疾病队列。

形式化（记号：簇 c 有 M 名成员、m 例阳性；随机揭示序；个体 i 的先证块
prior_pos_i = 先于 i 揭示的同户阳性数）：

- SEI-1（within 恒等式，代数）：给定簇组成 (M,m)，
  E[prior_pos | y=1, c] = (m−1)/2，E[prior_pos | y=0, c] = m/2，
  差 = **−1/2 恒定**——与疾病、终点、簇大小、基率、标签聚集度均无关。
  证明：随机序下每个其他阳性以 1/2 概率先于 i（期望线性）。
- SEI-2（m=1 路径恒等）：m=1 簇中阳性成员 prior_pos ≡ 0（逐路径，
  非仅期望）——唯一阳性就是自己，自排除。
- SEI-3（池化分解）：E[prior_pos|y=1]−E[prior_pos|y=0] = δ/2，其中
  δ = E[m_c|y=1]−E[m_c|y=0]−1 ≥ 0 为超出自身标签的残余簇内聚集；
  δ=0（i.i.d. 标签）⟹ 池化移位=0；δ>0（ICC>0）⟹ 池化为正。
- SEI-4（通道排他性推论）：先证块的**within 通道恒为负（SEI-1），
  其全部池化判别价值来自 between 通道（SEI-3）**——index-linked
  序贯先验的设计原理：within 不可修复（构造性），between 是唯一
  价值来源（C1-C5 的 ICC 条件即 δ>0）。

命题（预声明，先于运行）：
- G1（within 标签无关性，主命题）：真实户结构 × 全部标签
  （TB 参照 + HIV + 糖尿病 + 吸烟 + BCG 疤痕 + 性别）上，簇内
  平均移位 ≈ −0.5（20 种子均值 |偏差| < 0.05）。
- G2（SEI-3 分解兑现）：池化移位 ≈ δ/2 逐标签成立（回归斜率
  ~1、截距 ~0）；i.i.d. 型标签（δ≈0）池化 AUROC(prior_pos) ≈ 0.5，
  聚集标签（δ>0）池化 > 0.5。
- G3（within 反转普适）：within-AUROC(prior_pos) < 0.5 全标签
  （末种子 cluster bootstrap CI 上界 < 0.5）；m=1 层强于 m≥2 层
  （更远离 0.5）。
- G4（模拟精确性）：构造标签（i.i.d. 与 Beta-binomial 两型）上
  SEI-1 精确成立（|shift+0.5| < 0.01）、m=1 路径零比例 = 1.0。

诚实边界：
- 泛化主张 = "机制标签无关"（构造层），非"在新疾病队列上复现增益"——
  后者归各疾病 index-linked 程序自己的外验；本 PoC 的标签（HIV/糖尿病/
  吸烟/BCG/性别）非 TB，但簇结构复用接触者队列（真实家庭结构）；
- within-AUROC 为 prior_pos 单变量口径（§8.11 的 dep_fixed within
  塌陷由 prior 分量主导——本处直接验证该分量）；
- 糖尿病/BCG/性别等低 ICC 标签的 δ 估计噪声大，G2 回归以 TB/HIV/
  吸烟等聚集标签为主锚；
- 模拟腿为构造性验证（合成标签），只证恒等式不证性能——合成上界
  纪律沿用。

用法：
    python data/run_hi3_generalization.py
输出：
    data/processed/hi3_generalization_sei_YYYYMMDD.json
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

import run_sop_deepening as rsd  # noqa: E402
import run_uga_dual_exposure_gradient as uga  # noqa: E402

from tb_risk.validation.real_data_infection import (  # noqa: E402
    load_homeacf_contacts,
)
from tb_risk.validation.household_temporal import (  # noqa: E402
    assign_screening_order,
    prior_features,
)

OUT = os.path.join(HERE, 'processed',
                   'hi3_generalization_sei_%s.json' % time.strftime('%Y%m%d'))

N_SEEDS = 20
N_BOOTSTRAP = 2000
SHIFT_TOL = 0.05


def _auroc(y, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y, dtype=int),
                               np.asarray(scores, dtype=float)))


def bootstrap_within_auroc(contribs, n_boot, seed):
    """单分数 within-AUROC 的户级 cluster bootstrap CI。"""
    keys = np.array(sorted(contribs))
    if len(keys) == 0:
        return {'mean': float('nan'), 'ci': [float('nan')] * 2,
                'n_households': 0}
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        num = sum(contribs[g][0] for g in pick)
        den = sum(contribs[g][1] for g in pick)
        if den > 0:
            vals.append(num / den)
    vals = np.asarray(vals)
    return {
        'mean': float(np.mean(vals)),
        'ci': [float(np.percentile(vals, 2.5)),
               float(np.percentile(vals, 97.5))],
        'n_households': int(len(keys)),
    }


def within_shift(prior_pos, y, groups):
    """簇内平均移位：mean(prior_pos|y=1,c)−mean(prior_pos|y=0,c) 的簇均值。

    仅计同时含阴阳的多成员簇。返回 (shift_mean, n_clusters_used)。
    """
    prior_pos = np.asarray(prior_pos, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    shifts = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        sg = prior_pos[idx]
        yg = y[idx]
        pos = sg[yg == 1]
        neg = sg[yg == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        shifts.append(float(pos.mean() - neg.mean()))
    if not shifts:
        return float('nan'), 0
    return float(np.mean(shifts)), len(shifts)


def m1_pathwise_zero(prior_pos, y, groups):
    """m=1 簇（恰 1 阳性）中阳性成员 prior_pos==0 的比例（SEI-2，应=1.0）。"""
    prior_pos = np.asarray(prior_pos, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    hh_pos = pd.Series(y).groupby(groups).sum()
    frac, n = [], 0
    for g in np.unique(groups):
        if hh_pos[g] != 1:
            continue
        idx = np.where(groups == g)[0]
        p = idx[y[idx] == 1]
        n += len(p)
        frac.extend((prior_pos[p] == 0).tolist())
    if n == 0:
        return float('nan'), 0
    return float(np.mean(frac)), n


def pooled_delta(prior_pos, y, groups):
    """池化移位与 δ：shift=E[prior_pos|1]−E[prior_pos|0]，δ=E[m_c|1]−E[m_c|0]−1。"""
    prior_pos = np.asarray(prior_pos, dtype=float)
    y = np.asarray(y, dtype=int)
    dfh = pd.DataFrame({'y': y, 'g': np.asarray(groups)})
    m_c = dfh.groupby('g')['y'].transform('sum').to_numpy(dtype=float)
    shift = float(prior_pos[y == 1].mean() - prior_pos[y == 0].mean())
    delta = float(m_c[y == 1].mean() - m_c[y == 0].mean() - 1.0)
    return shift, delta


def analyze_label(y, groups, label_name, cohort_name):
    """单标签全量分析：20 种子随机序 → SEI-1/2/3/4 读数。"""
    y = np.asarray(y, dtype=int)
    n_pos = int(y.sum())
    result = {
        'cohort': cohort_name, 'label': label_name,
        'n': int(len(y)), 'n_events': n_pos,
        'n_clusters': int(len(np.unique(groups))),
    }
    if n_pos == 0 or n_pos == len(y):
        result['note'] = '无方差标签，跳过'
        return result

    df = pd.DataFrame({'record_id': np.asarray(groups), 'tst_pos10': y})
    shift_seeds, pooled_shift_seeds, delta_seeds = [], [], []
    within_seeds, pooled_auroc_seeds = [], []
    last = None
    for s in range(N_SEEDS):
        order = assign_screening_order(df, seed=s, mode='random')
        pf = prior_features(df, order)
        prior_pos = pf['prior_pos'].to_numpy(dtype=float)
        sh, n_sh = within_shift(prior_pos, y, groups)
        psh, dl = pooled_delta(prior_pos, y, groups)
        w_auc = rsd.within_hh_auroc(
            rsd.within_hh_contribs(prior_pos, y, groups))
        p_auc = _auroc(y, prior_pos)
        shift_seeds.append(sh)
        pooled_shift_seeds.append(psh)
        delta_seeds.append(dl)
        within_seeds.append(w_auc)
        pooled_auroc_seeds.append(p_auc)
        last = (prior_pos, w_auc)
    prior_pos_l, _ = last

    # 末种子 bootstrap + m 分层
    contribs = rsd.within_hh_contribs(prior_pos_l, y, groups)
    boot = bootstrap_within_auroc(contribs, N_BOOTSTRAP, 0)
    by_m = rsd.within_hh_by_m(prior_pos_l, y, groups)
    mz, n_m1 = m1_pathwise_zero(prior_pos_l, y, groups)

    # ICC（标签簇内聚集度，MoM）
    _, k_hat = rsd.eb_shrinkage_params(y, groups)
    icc = 1.0 / (1.0 + k_hat) if k_hat < 99.0 else 0.0

    result.update({
        'sei1_within_shift': {
            'mean': round(float(np.mean(shift_seeds)), 4),
            'min': round(float(np.min(shift_seeds)), 4),
            'max': round(float(np.max(shift_seeds)), 4),
            'n_clusters_used': n_sh,
            'abs_dev_from_minus_half': round(
                abs(float(np.mean(shift_seeds)) + 0.5), 4),
            'within_tol_0.05': bool(
                abs(float(np.mean(shift_seeds)) + 0.5) < SHIFT_TOL),
        },
        'sei3_pooled': {
            'pooled_shift_mean': round(
                float(np.mean(pooled_shift_seeds)), 4),
            'delta_mean': round(float(np.mean(delta_seeds)), 4),
            'delta_half': round(float(np.mean(delta_seeds)) / 2.0, 4),
            'pooled_auroc_mean': round(
                float(np.mean(pooled_auroc_seeds)), 4),
            'pooled_auroc_min': round(float(np.min(pooled_auroc_seeds)), 4),
            'pooled_auroc_max': round(float(np.max(pooled_auroc_seeds)), 4),
            'pooled_auroc_ci_upper_exceeds_half': bool(
                float(np.min(pooled_auroc_seeds)) > 0.5),
        },
        'sei3_within': {
            'within_auroc_mean': round(float(np.mean(within_seeds)), 4),
            'last_seed_bootstrap': boot,
            'ci_upper_below_half': bool(boot['ci'][1] < 0.5),
            'by_m': by_m,
        },
        'sei2_m1_pathwise_zero_frac': round(mz, 4),
        'sei2_m1_n_members': n_m1,
        'icc_label': round(icc, 4),
    })
    return result


def simulate_sei():
    """构造标签的精确性验证：i.i.d. 与 Beta-binomial 两型（G4）。"""
    rng = np.random.RandomState(42)
    n_clusters = 4000
    M = rng.randint(2, 9, size=n_clusters)
    out = {}
    for scheme, note in (('iid', '标签簇内独立（δ=0 极端）'),
                         ('beta_binom', '户率 r_c~Beta(2π,2(1−π))，ICC=1/3')):
        shifts, pooled_shifts, deltas = [], [], []
        m1_zero = []
        for s in range(10):
            groups = np.repeat(np.arange(n_clusters), 0).tolist()
            ys, gs = [], []
            for c, m in enumerate(M):
                if scheme == 'iid':
                    y_c = rng.rand(m) < 0.2
                else:
                    r_c = rng.beta(2 * 0.2, 2 * 0.8)
                    y_c = rng.rand(m) < r_c
                ys.extend(y_c.astype(int).tolist())
                gs.extend([c] * m)
            y = np.asarray(ys)
            g = np.asarray(gs)
            df = pd.DataFrame({'record_id': g, 'tst_pos10': y})
            order = assign_screening_order(df, seed=s, mode='random')
            pf = prior_features(df, order)
            prior_pos = pf['prior_pos'].to_numpy(dtype=float)
            sh, _ = within_shift(prior_pos, y, g)
            psh, dl = pooled_delta(prior_pos, y, g)
            shifts.append(sh)
            pooled_shifts.append(psh)
            deltas.append(dl)
            mz, _ = m1_pathwise_zero(prior_pos, y, g)
            m1_zero.append(mz)
        out[scheme] = {
            'note': note,
            'sei1_within_shift_mean': round(float(np.mean(shifts)), 4),
            'sei1_abs_dev_from_minus_half': round(
                abs(float(np.mean(shifts)) + 0.5), 4),
            'sei3_pooled_shift_mean': round(
                float(np.mean(pooled_shifts)), 4),
            'sei3_delta_mean': round(float(np.mean(deltas)), 4),
            'sei3_delta_half': round(float(np.mean(deltas)) / 2.0, 4),
            'sei2_m1_pathwise_zero_frac': round(float(np.mean(m1_zero)), 4),
        }
    return out


def main():
    t0 = time.time()
    print('=== Part 0：构造标签精确性（G4，SEI 代数验证）===')
    sim = simulate_sei()
    for k, v in sim.items():
        print('  %-12s within_shift=%+.4f (dev %.4f) pooled=%+.4f '
              'δ/2=%+.4f m1_zero=%.3f' % (
                  k, v['sei1_within_shift_mean'],
                  v['sei1_abs_dev_from_minus_half'],
                  v['sei3_pooled_shift_mean'], v['sei3_delta_half'],
                  v['sei2_m1_pathwise_zero_frac']))

    print('\n=== Part 1：HomeACF 真实户结构 × 标签阶梯 ===')
    df = load_homeacf_contacts()
    groups_h = df['record_id'].to_numpy()
    labels_h = [
        ('tst_pos10', 'TB 参照（TST≥10）', df['tst_pos10'].to_numpy(dtype=int)),
        ('hiv_pos_h', 'HIV（index-linked 筛查类比）',
         df['hiv_pos_h'].to_numpy(dtype=float)),
        ('smoke_ever_h', '吸烟（行为，中度聚集）',
         df['smoke_ever_h'].to_numpy(dtype=float)),
        ('diabetes_h_f', '糖尿病（低聚集极端）',
         df['diabetes_h_f'].to_numpy(dtype=float)),
    ]
    results_h = []
    for col, name, y_raw in labels_h:
        mask = ~np.isnan(np.asarray(y_raw, dtype=float))
        y = np.asarray(y_raw, dtype=float)[mask].astype(int)
        g = groups_h[mask]
        if len(np.unique(y)) < 2:
            print('  %-14s 无方差，跳过' % col)
            continue
        r = analyze_label(y, g, name, 'HomeACF')
        results_h.append(r)
        print('  %-14s n=%4d ev=%3d ICC=%.3f | within_shift=%+.4f '
              '(dev %.4f) | pooled AUROC=%.4f δ/2=%+.4f | '
              'within-AUROC=%.4f CI[%.3f,%.3f]' % (
                  col, r['n'], r['n_events'], r['icc_label'],
                  r['sei1_within_shift']['mean'],
                  r['sei1_within_shift']['abs_dev_from_minus_half'],
                  r['sei3_pooled']['pooled_auroc_mean'],
                  r['sei3_pooled']['delta_half'],
                  r['sei3_within']['within_auroc_mean'],
                  r['sei3_within']['last_seed_bootstrap']['ci'][0],
                  r['sei3_within']['last_seed_bootstrap']['ci'][1]))

    print('\n=== Part 2：Muchuro 真实户结构 × 非 TB 标签 ===')
    d = uga.load_muchuro()
    d = d.dropna(subset=['y']).reset_index(drop=True)
    g_m = d['group'].to_numpy()
    mu_raw = pd.read_excel(os.path.join(
        HERE, 'raw', 'figshare', 'uganda_household_igra_s001.xlsx'))
    labels_m = [
        ('igra_pos', 'TB 参照（IGRA）', d['y'].to_numpy(dtype=int)),
        ('bcg_no_scar', 'BCG 无疤（疫苗衍生，低聚集）',
         (mu_raw['bcgscar'] == 'No').astype(int).to_numpy()),
        ('male', '男性（人口学）',
         (mu_raw['gender'] == 'Male').astype(int).to_numpy()),
    ]
    results_m = []
    for col, name, y in labels_m:
        r = analyze_label(np.asarray(y, dtype=int), g_m, name, 'Muchuro')
        results_m.append(r)
        if 'sei1_within_shift' in r:
            print('  %-12s n=%4d ev=%3d ICC=%.3f | within_shift=%+.4f '
                  '(dev %.4f) | pooled AUROC=%.4f δ/2=%+.4f | '
                  'within-AUROC=%.4f' % (
                      col, r['n'], r['n_events'], r['icc_label'],
                      r['sei1_within_shift']['mean'],
                      r['sei1_within_shift']['abs_dev_from_minus_half'],
                      r['sei3_pooled']['pooled_auroc_mean'],
                      r['sei3_pooled']['delta_half'],
                      r['sei3_within']['within_auroc_mean']))

    # ---- 判决汇总 ----
    all_results = results_h + results_m
    valid = [r for r in all_results if 'sei1_within_shift' in r]
    g1 = all(r['sei1_within_shift']['within_tol_0.05'] for r in valid)
    # G2：池化移位 vs δ/2 回归（去掉 δ 估计噪声过大的点看总体）
    deltas = np.array([r['sei3_pooled']['delta_mean'] for r in valid])
    pshift = np.array([r['sei3_pooled']['pooled_shift_mean'] for r in valid])
    slope, intercept = np.polyfit(deltas / 2.0, pshift, 1)
    g2 = bool(abs(slope - 1.0) < 0.5 and abs(intercept) < 0.1)
    g3 = all(r['sei3_within']['ci_upper_below_half'] for r in valid
             if not np.isnan(r['sei3_within']['last_seed_bootstrap']['ci'][1]))
    g3_ci_detail = {r['label']: r['sei3_within'][
        'last_seed_bootstrap']['ci'] for r in valid}
    g4 = all(v['sei1_abs_dev_from_minus_half'] < 0.01
             and v['sei2_m1_pathwise_zero_frac'] == 1.0
             for v in sim.values())
    verdicts = {
        'G1_within_shift_minus_half_label_agnostic': bool(g1),
        'G2_pooled_shift_equals_delta_half': {
            'slope': round(float(slope), 3),
            'intercept': round(float(intercept), 4),
            'n_labels': len(valid),
            'pass': g2},
        'G3_within_auroc_below_half_all_labels': bool(g3),
        'G3_ci_detail': {k: [round(a, 3), round(b, 3)]
                         for k, (a, b) in g3_ci_detail.items()},
        'G4_simulation_exactness': bool(g4),
    }
    print('\n=== 预声明命题判决 ===')
    for k, v in verdicts.items():
        print('  %s: %s' % (k, v))

    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'hi3_generalization_sei_v1',
        'status': 'H-I3 升格为自排除恒等式（SEI）通用命题：形式化 + '
                  '真实户结构×非 TB 标签概念验证（预声明 G1-G4）',
        'formalization': {
            'name': '自排除恒等式（Self-Exclusion Identity, SEI）',
            'SEI-1': 'within 恒等式：E[prior_pos|y=1,c]=(m−1)/2, '
                     'E[prior_pos|y=0,c]=m/2，差=−1/2 恒定（代数，'
                     '标签/疾病/簇大小/基率无关）',
            'SEI-2': 'm=1 路径恒等：单阳性簇中阳性成员 prior_pos≡0（逐路径）',
            'SEI-3': '池化分解：池化移位=δ/2，δ=E[m_c|y=1]−E[m_c|y=0]−1≥0'
                     '（超出自身标签的残余簇内聚集；δ=0⟹池化 null）',
            'SEI-4': '通道排他性：先证块 within 通道恒负、全部池化价值'
                     '来自 between 通道——index-linked 序贯先验的设计原理',
            'relation_to_C1-C5': 'C 条件=先证块何时有增益（between 通道'
                                 '存在性，δ>0 即 ICC>0）；SEI=构造性代价'
                                 '（within 不可修复）。二者合成完整设计'
                                 '理论。',
        },
        'hypotheses': {
            'G1': 'within 移位 ≈ −0.5 全标签（|dev|<0.05，20 种子均值）',
            'G2': '池化移位 ≈ δ/2 回归斜率~1 截距~0；δ≈0 标签池化 '
                  'AUROC≈0.5',
            'G3': 'within-AUROC<0.5 全标签（bootstrap CI 上界<0.5）；'
                  'm=1 层更远离 0.5',
            'G4': '构造标签上 SEI-1 精确（|dev|<0.01）、m=1 路径零=1.0',
        },
        'verdicts': verdicts,
        'simulation': sim,
        'homeacf_labels': results_h,
        'muchuro_labels': results_m,
        'honest_boundaries': [
            '泛化主张="机制标签无关"（构造层），非"新疾病队列上复现增益"'
            '——后者归各疾病 index-linked 程序外验；标签非 TB 但簇结构'
            '复用接触者队列（真实家庭结构）',
            'within-AUROC 为 prior_pos 单变量口径（§8.11 dep_fixed within '
            '塌陷由 prior 分量主导——本处直接验证该分量）',
            '低 ICC 标签（糖尿病/BCG/性别）的 δ 估计噪声大，G2 回归以'
            '聚集标签为主锚；Muchuro n=352 小样本 CI 宽',
            '模拟腿为构造性验证（合成标签）只证恒等式不证性能——合成'
            '上界纪律沿用',
        ],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
