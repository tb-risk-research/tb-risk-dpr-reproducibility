#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W-M4：陡曲线 × 事件稀缺象限——门槛最后一项主张的直接检验。

背景（门槛验证证据矩阵，2026-09-07 三判决链）：
- 同总体×低事件（HomeACF h=50-100 holdout V5）：平曲线 → w 误选
  无害，路径 B 反而 +0.02 胜路径 A（结构先验省样本）；
- 平移×足事件（Muchuro 全量 W-M，h=59/事件 115，仅事件项达标）：
  陡曲线（峰 w=3，w=12 −0.0594）→ 本地选择准确（w*=3），本地
  重校准 ≈ 路径 A parity（+0.0078 选择乐观偏置内）；
- ⇒ 两已测象限均不支持"门槛下路径 B 危险"。
- **未测象限 = 陡曲线 × 事件稀缺**——门槛（事件≥80）真正声称
  守护的场景：新站点（平移）+ 影子数据不足。Muchuro 户级子抽样
  到事件稀缺区是唯一可构造的直接检验。

命题（预声明，先于运行）：
- W-M4a（选择退化方向）：w* 相对全量曲线真峰的成本随事件数
  下降单调变差（mean_cost(h=30) < mean_cost(h=50)）；
- W-M4b（门槛锚定）：若 h=30（事件≈58，门槛下）share_costly
  （成本 ≤−0.010）比 h=50（事件≈97）高 ≥0.20 → 门槛事件项
  获直接经验支持；若两水平 share_costly 均 ≤0.20 → 门槛最后
  一项主张亦证伪，门槛整体降级为"w* 点估计可引用性"提示。

成本口径：cost = ref_curve[w*] − ref_curve[ref_peak]，ref =
全量 59 户曲线（20 种子 OOF 中位数，W-M 存档同流复算）。
选择协议 = §12 路径 B 原文：SGKF(5)×10 种子 OOF 中位数选 w
（并列取小），π = 子样本队列率（部署者 π̂）；每种子独立随机
筛查顺序（CV 种子驱动，W-M 存档同口径——顺序方差平均）。

gate：全量曲线复现 W-M 存档（峰 w=3，中位数逐点 |Δ|<0.001）。

诚实边界：59 户母体子抽样 → 各 rep 高度重叠（有限总体），
R=20 的份额读数为方向级；ref 曲线本身是估计量（陡峭结构
稳定但逐点值有噪声）；单队列 n=1；h=30 时 n≈180（远低于
门槛 n≥500——本检验只针对事件稀缺通道，非全门槛剂量响应）。

用法：
    python data/run_w_muchuro_scarcity.py
输出：
    data/processed/w_muchuro_scarcity_YYYYMMDD.json
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
                   'w_muchuro_scarcity_%s.json' % time.strftime('%Y%m%d'))

W_GRID = (1.0, 3.0, 5.0, 7.0, 9.0, 12.0)
DEP_K = 2.0
N_SEEDS_REF = 20   # 全量参考曲线（W-M 同流）
N_SEEDS_SEL = 10   # §12 协议选择种子数
H_LEVELS = (30, 40, 50)
N_REPS = 20
COSTLY_THR = -0.010
GATE_TOL = 0.001

# W-M 存档全量曲线（gate 对照）
WM_REF_CURVE = {'1': 0.7351, '3': 0.7458, '5': 0.7361,
                '7': 0.7184, '9': 0.7038, '12': 0.6864}


def w_curve_median(X, y, groups, pi, n_seeds):
    """OOF AUROC 对 w 网格的中位数曲线。

    每种子独立随机筛查顺序（CV 种子同时驱动顺序）——与 W-M
    存档同口径，先证块对每种子重抽（顺序方差平均）。
    """
    hh = pd.DataFrame({'record_id': groups, 'tst_pos10': y})
    per_w = {w: [] for w in W_GRID}
    for s in range(n_seeds):
        order = assign_screening_order(hh, seed=s, mode='random')
        pf = prior_features(hh, order)
        pn = pf['prior_n'].to_numpy(dtype=float)
        pp = pf['prior_pos'].to_numpy(dtype=float)
        oof = fit_arm_oof(X, y, groups, s)
        for w in W_GRID:
            per_w[w].append(rsd._auroc(
                y, rsd.dep_scores(oof, pn, pp, pi, DEP_K, w)))
    return {'%g' % w: float(np.median(v)) for w, v in per_w.items()}


def select_w(curve):
    """§12 协议：OOF 中位数选 w，并列取小。"""
    w_star, best = None, -np.inf
    for w in sorted(W_GRID):
        if curve['%g' % w] > best:
            best, w_star = curve['%g' % w], w
    return w_star


def main():
    t0 = time.time()
    d = uga.load_muchuro()
    d = d.dropna(subset=['y']).reset_index(drop=True)
    y = d['y'].astype(int).to_numpy()
    groups = d['group'].to_numpy()
    pi_full = float(y.mean())
    households = np.unique(groups)

    feat_cols = []
    seen = set()
    for c in uga.MU_ARMS['full']:
        if c not in seen:
            seen.add(c)
            feat_cols.append(c)
    X_full = d[feat_cols].to_numpy(dtype=float)

    # ---- 全量参考曲线（20 种子，W-M 同流复算）----
    ref_curve = w_curve_median(X_full, y, groups, pi_full, N_SEEDS_REF)
    ref_peak = max(ref_curve, key=ref_curve.get)
    gate_curve = all(abs(ref_curve[k] - v) < GATE_TOL
                     for k, v in WM_REF_CURVE.items())
    gate_peak = ref_peak == '3'
    print('全量参考曲线: %s' % ref_curve)
    print('gate 曲线复现: %s | 峰=%s' % (gate_curve, ref_peak))
    if not (gate_curve and gate_peak):
        print('!! gate 未过，结果仅供参考')
        gate = False
    else:
        gate = True

    # ---- 子抽样扫描 ----
    scan = {}
    for h in H_LEVELS:
        rng = np.random.RandomState(7000 + h)
        rows = []
        for rep in range(N_REPS):
            chosen = rng.choice(households, size=h, replace=False)
            mask = np.isin(groups, chosen)
            d_s = d[mask].reset_index(drop=True)
            y_s = d_s['y'].astype(int).to_numpy()
            g_s = d_s['group'].to_numpy()
            if y_s.sum() < 10 or len(np.unique(y_s)) < 2:
                continue
            X_s = d_s[feat_cols].to_numpy(dtype=float)
            pi_s = float(y_s.mean())
            curve = w_curve_median(X_s, y_s, g_s, pi_s, N_SEEDS_SEL)
            w_star = select_w(curve)
            cost = ref_curve['%g' % w_star] - ref_curve[ref_peak]
            rows.append({
                'rep': rep, 'n': int(len(y_s)), 'events': int(y_s.sum()),
                'w_star': w_star, 'cost': round(float(cost), 5),
                'sub_curve': {k: round(v, 4) for k, v in curve.items()},
            })
        costs = np.array([r['cost'] for r in rows])
        w_dist = {}
        for r in rows:
            w_dist['%g' % r['w_star']] = w_dist.get('%g' % r['w_star'], 0) + 1
        scan['%d' % h] = {
            'h_households': h, 'n_reps': len(rows),
            'mean_n': float(np.mean([r['n'] for r in rows])),
            'mean_events': round(float(
                np.mean([r['events'] for r in rows])), 1),
            'w_star_distribution': w_dist,
            'share_w_peak': round(w_dist.get(ref_peak, 0)
                                  / max(len(rows), 1), 3),
            'mean_cost': round(float(costs.mean()), 5),
            'sd_cost': round(float(costs.std(ddof=1)), 5) if len(rows) > 1 else None,
            'share_costly': round(float((costs <= COSTLY_THR).mean()), 3),
            'max_cost': round(float(costs.min()), 5),
        }
        print('h=%d: events≈%.0f | w* 分布=%s | mean_cost=%+.4f '
              'share_costly=%.2f max_cost=%+.4f' % (
                  h, scan['%d' % h]['mean_events'], w_dist,
                  costs.mean(), (costs <= COSTLY_THR).mean(),
                  costs.min()))

    # ---- 预声明判决 ----
    c30 = scan['30']['mean_cost']
    c50 = scan['50']['mean_cost']
    s30 = scan['30']['share_costly']
    s50 = scan['50']['share_costly']
    verdicts = {
        'gate': bool(gate),
        'W-M4a_cost_worsens_with_scarcity': bool(c30 < c50),
        'W-M4b_threshold_supported': bool(s30 - s50 >= 0.20),
        'W-M4b_last_claim_refuted': bool(s30 <= 0.20 and s50 <= 0.20),
    }
    print('\n=== 预声明判决 ===')
    for k, v in verdicts.items():
        print('  %s: %s' % (k, v))

    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'w_muchuro_scarcity_v1',
        'status': '陡曲线×事件稀缺象限直接检验：门槛事件项（≥80）的'
                  '最后主张——选择协议同 §12 路径 B 原文',
        'hypotheses': {
            'W-M4a': 'mean_cost 随事件稀缺单调变差（h=30 < h=50）',
            'W-M4b': 'h=30 share_costly 比 h=50 高 ≥0.20 → 门槛支持；'
                     '两水平均 ≤0.20 → 最后主张证伪',
        },
        'verdicts': verdicts,
        'ref_curve': ref_curve,
        'ref_peak': ref_peak,
        'costly_threshold': COSTLY_THR,
        'scan': scan,
        'honest_boundaries': [
            '59 户母体子抽样 → 各 rep 高度重叠（有限总体），份额读数'
            '为方向级',
            'ref 曲线本身是估计量（陡峭结构稳定、逐点值有噪声）',
            '单队列 n=1；h=30 时 n≈180——只测事件稀缺通道，非全门槛'
            '剂量响应',
            '选择与成本口径分离：w* 由子样本曲线选，成本对全量曲线计'
            '（部署问题：真曲线未知、有限观测选 w、代价付在真值上）',
            '每种子独立随机筛查顺序（与 W-M 存档同口径）——部署单'
            '顺序观测的方差未在此分离',
            'π=子样本队列率（部署 π̂，H-K2 容忍带）',
        ],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
