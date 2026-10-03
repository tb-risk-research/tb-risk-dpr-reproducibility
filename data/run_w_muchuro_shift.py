#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Muchuro 平移维度补充验证（W-M 系列，2026-09-07）：门槛判决的
最后一个缺口。

背景（两阶段 HomeACF 子抽样的结论链）：
- 同总体下部署公式在**所有样本量**上胜 RF（结构先验省样本：
  h=50 时 pathA 0.5718 vs dep 0.6473）；
- w 误选（→12，网格极值）在同总体内几乎无代价（AUROC-w 曲线
  峰平坦：全量 w=7:0.7210 vs w=12:0.7188）；
- ⇒ 样本量维度**不构成**门槛的机制——真正的风险维度是人群平移。
- Muchuro 天然实验（§8.13 M 系列）只测过 w=7 迁移失败
  （1.90× < RF 2.27×），**未测本地重校准**。

本脚本闭合该缺口：Muchuro（h=59，门槛下 + 平移）上跑 §12 路径 B
完整本地选择协议（w 网格 {1,3,5,7,9,12}，OOF AUROC 中位数选 w，
并列取小），对比路径 A（RF full+prior）。

命题（预声明，先于运行）：
- W-M1（选择偏置平移复现）：本地选择 w* 落在网格高端（≥9，
  预期 12）——小样本选择偏置在平移队列上复现。
- W-M2（曲线形态平移反转）：Muchuro AUROC-w 曲线峰在低端
  （w ∈ {1,3,5}），且 AUROC(w=12) − AUROC(w_peak) ≤ −0.010
  （非平坦——与 HomeACF 平坦峰形成对照：平移下先证块信号弱
  （M3 prior 增量 +0.0169 vs HomeACF +0.0585），最优 w 应更低，
  高 w 过度加权伤害更大）。
- W-M3（主判决：门槛下平移 + 本地重校准仍失败）：
  mean AUROC(dep w*) − mean AUROC(RF full+prior) < 0——路径 B
  即使本地重校准也不敌路径 A ⇒ §12"门槛下默认路径 A"的机制
  在平移维度闭合（选择偏置 + 非平坦曲线的复合伤害）。

gate：host/exp/full 锚（0.5887/0.6585/0.7247，20 种子 pooled）
+ M3 增量锚（full_prior − full 逐种子均值 +0.0169 ±0.002）。

诚实边界：dep 基座 = full RF OOF（§8.13 M 系列存档口径，非
HomeACF 的 exposure 基座——两队列基座差异如实披露）；π=队列率
（已知 π 简化，H-K2 容忍带）；W-M3 的 dep(w*) 用同 OOF 选择又
评估（乐观偏置偏袒 dep——若仍 < full_prior 则结论保守强）；
单队列 n=1 平移实验；h=59 单点不构成剂量响应。

用法：
    python data/run_w_muchuro_shift.py
输出：
    data/processed/w_muchuro_shift_YYYYMMDD.json
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
from run_muchuro_capture import fit_arm_oof  # noqa: E402

from tb_risk.validation.household_temporal import (  # noqa: E402
    assign_screening_order,
    prior_features,
)

OUT = os.path.join(HERE, 'processed',
                   'w_muchuro_shift_%s.json' % time.strftime('%Y%m%d'))

W_GRID = (1.0, 3.0, 5.0, 7.0, 9.0, 12.0)
DEP_K = 2.0
N_SEEDS = 20
ANCHOR = {'host': 0.5887, 'exp': 0.6585, 'full': 0.7247}
ANCHOR_M3_DELTA = 0.0169
GATE_TOL = 0.002


def main():
    t0 = time.time()
    d = uga.load_muchuro()
    d = d.dropna(subset=['y']).reset_index(drop=True)
    y = d['y'].astype(int).to_numpy()
    groups = d['group'].to_numpy()
    pi_cohort = float(y.mean())
    print('Muchuro: n=%d events=%d households=%d π=%.4f' % (
        len(y), int(y.sum()), len(np.unique(groups)), pi_cohort))

    hh = pd.DataFrame({'record_id': groups, 'tst_pos10': y})
    feat_cols = []
    seen = set()
    for c in uga.MU_ARMS['full']:
        if c not in seen:
            seen.add(c)
            feat_cols.append(c)
    X_all = d[feat_cols].to_numpy(dtype=float)
    col_ix = {c: i for i, c in enumerate(feat_cols)}
    X_full = X_all[:, [col_ix[c] for c in uga.MU_ARMS['full']]]

    seeds_data = []
    for s in range(N_SEEDS):
        order = assign_screening_order(hh, seed=s, mode='random')
        pf = prior_features(hh, order)
        prior_n = pf['prior_n'].to_numpy(dtype=float)
        prior_pos = pf['prior_pos'].to_numpy(dtype=float)
        X_full_prior = np.column_stack([
            X_full, prior_n, prior_pos,
            pf['prior_rate'].to_numpy(dtype=float),
            pf['prior_screened'].to_numpy(dtype=float)])
        oof_full = fit_arm_oof(X_full, y, groups, s)
        oof_full_prior = fit_arm_oof(X_full_prior, y, groups, s)
        oof_host = fit_arm_oof(
            X_all[:, [col_ix[c] for c in uga.MU_ARMS['host']]],
            y, groups, s)
        oof_exp = fit_arm_oof(
            X_all[:, [col_ix[c] for c in uga.MU_ARMS['exp']]],
            y, groups, s)
        dep_by_w = {w: rsd.dep_scores(oof_full, prior_n, prior_pos,
                                      pi_cohort, DEP_K, w)
                    for w in W_GRID}
        seeds_data.append({
            'seed': s,
            'oof': {'host': oof_host, 'exp': oof_exp, 'full': oof_full,
                    'full_prior': oof_full_prior, 'dep': dep_by_w},
            'prior_n': prior_n, 'prior_pos': prior_pos,
        })

    # ---- gate：锚复现 ----
    pooled = {a: np.mean([r['oof'][a] for r in seeds_data], axis=0)
              for a in ('host', 'exp', 'full', 'full_prior')}
    gates = {a: abs(rsd._auroc(y, pooled[a]) - v) < GATE_TOL
             for a, v in ANCHOR.items()}
    m3_deltas = [rsd._auroc(y, r['oof']['full_prior'])
                 - rsd._auroc(y, r['oof']['full'])
                 for r in seeds_data]
    m3_mean = float(np.mean(m3_deltas))
    gates['M3_delta'] = abs(m3_mean - ANCHOR_M3_DELTA) < GATE_TOL
    print('\ngates: %s | M3 Δ=%.4f (锚 %.4f)' % (
        gates, m3_mean, ANCHOR_M3_DELTA))

    # ---- w 曲线 + 选择 ----
    curve_median = {}
    for w in W_GRID:
        per_seed = [rsd._auroc(y, r['oof']['dep'][w])
                    for r in seeds_data]
        curve_median['%g' % w] = round(float(np.median(per_seed)), 4)
    w_star, best = None, -np.inf
    for w in sorted(W_GRID):
        if curve_median['%g' % w] > best:
            best, w_star = curve_median['%g' % w], w
    w_peak = max(curve_median, key=curve_median.get)
    flat_gap = curve_median['12'] - curve_median[w_peak]
    print('\nAUROC-w 曲线（中位数）: %s' % curve_median)
    print('w*=%g | 峰 w=%s | AUROC(12)−AUROC(峰)=%.4f' % (
        w_star, w_peak, flat_gap))

    # ---- W-M3：dep(w*) vs full_prior（逐种子）----
    dep_star = [rsd._auroc(y, r['oof']['dep'][w_star])
                for r in seeds_data]
    fp = [rsd._auroc(y, r['oof']['full_prior']) for r in seeds_data]
    deltas = np.asarray(dep_star) - np.asarray(fp)
    wm3 = {
        'mean_delta_dep_star_minus_full_prior': round(
            float(deltas.mean()), 5),
        'share_positive': round(float((deltas > 0).mean()), 3),
        'pooled_dep_star': round(rsd._auroc(
            y, np.mean([r['oof']['dep'][w_star]
                        for r in seeds_data], axis=0)), 4),
        'pooled_full_prior': round(rsd._auroc(y, pooled[
            'full_prior']), 4),
    }
    print('W-M3: dep(w*=%g) − full_prior: mean=%+.4f share+=%.2f | '
          'pooled %.4f vs %.4f' % (
              w_star, deltas.mean(), (deltas > 0).mean(),
              wm3['pooled_dep_star'], wm3['pooled_full_prior']))

    verdicts = {
        'gate': bool(all(gates.values())),
        'W-M1_bias_replicates': w_star >= 9.0,
        'W-M2_curve_peaks_low_and_steep': bool(
            float(w_peak) <= 5.0 and flat_gap <= -0.010),
        'W-M3_local_recal_fails_under_shift': bool(
            deltas.mean() < 0),
    }
    print('\n=== 预声明判决 ===')
    for k, v in verdicts.items():
        print('  %s: %s' % (k, v))

    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'w_muchuro_shift_v1',
        'status': '门槛判决平移维度闭合：门槛下+平移+本地重校准的'
                  '路径 B vs 路径 A（§12 前置条件机制的最终读数）',
        'hypotheses': {
            'W-M1': '选择偏置平移复现（w*≥9）',
            'W-M2': '曲线峰低端且非平坦（峰 w≤5，w=12 比峰低 ≥0.010）',
            'W-M3': 'dep(w*) < full_prior（本地重校准仍败）',
        },
        'verdicts': verdicts,
        'gate_detail': {
            'anchors': {a: round(rsd._auroc(y, pooled[a]), 4)
                        for a in ANCHOR},
            'm3_delta_mean': round(m3_mean, 4),
            'gates': gates,
        },
        'w_curve_median': curve_median,
        'w_star': w_star,
        'w_peak': w_peak,
        'flat_gap_w12_minus_peak': round(flat_gap, 4),
        'wm3': wm3,
        'honest_boundaries': [
            'dep 基座 = full RF OOF（§8.13 M 系列存档口径，非 '
            'HomeACF 的 exposure 基座——两队列基座差异披露）',
            'W-M3 的 dep(w*) 用同 OOF 选择又评估（乐观偏置偏袒 '
            'dep——若仍 < full_prior 则结论保守强）',
            'π=队列率（已知 π 简化，H-K2 容忍带）',
            '单队列 n=1 平移实验；h=59 单点不构成剂量响应',
        ],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
