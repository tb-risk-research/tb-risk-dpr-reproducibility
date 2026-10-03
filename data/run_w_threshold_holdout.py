#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""w 门槛验证第二阶段（V5/V6，holdout 修正协议，2026-09-07）。

背景：第一阶段（w_threshold_validation_homeacf_20260907.json）的
V2 读数被**选择乐观性**污染——B_recal 的 w* 用同一 OOF 选择又评估
（事件≈20 时 6 点 max 选择的乐观偏置 +0.02-0.07，随 h 单调收敛，
形态与量级一致）；且 V1 的失败模型预期错误（预期"不稳定"，实际
"稳定地选错"：h≤100 层 modal w=12 网格极值，份额 0.70/0.90）。

修正协议（本阶段）：外层 70/30 户级分割——内层（70%）跑 §12
路径 B 完整协议选 w*，外层（30%）干净评估路径 A / 路径 B(w*) /
路径 B(w=7 迁移)。选择与评估分离，乐观性归零。

部署者视角实现：每 rep 固定一个筛查顺序实现（order_seed=rep，
内层 CV 与外层评估共享——部署现实=单一观测顺序）；外层 π=训练
折率（部署者估计）；RF 基座 10 种子均值预测。

命题（预声明，先于运行）：
- V5（选择伤害，门槛下）：h≤100：holdout Δ(A−B_recal) 均值 > 0
  （误选 w 伤害部署性能，A 优）；且 Δ(B_w7−B_recal) 均值 > 0
  （w=7 本可更好——选择不仅不同而且有害的直接证据）。
- V6（门槛上）：h=400：|holdout Δ(A−B_recal)| ≤ 0.010（B 不劣）；
  选择准确率 share(w*=7) ≥ 0.5；h=200 为过渡带如实报告。
- 综合（门槛判决）：若 V5+V6 成立——门槛（事件≥80 ⟺ h≈200）
  在 holdout 口径下把"路径 B 可安全启用"与"不可"分开，第一阶段
  的乐观性污染与"稳定选错"模式构成门槛必要性的双重证据。

诚实边界：holdout 在 h=50 层事件均值 ≈6——单 rep 噪声大，
以 20 rep 均值为主读数（无偏性保持）；第一阶段子抽样完全复用
（同 RNG 流，可比性）；单一源人群；V4（Muchuro 平移维度）引用
不变。

用法：
    python data/run_w_threshold_holdout.py
输出：
    data/processed/w_threshold_holdout_homeacf_YYYYMMDD.json
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

from tb_risk.validation.household_temporal import (  # noqa: E402
    assign_screening_order,
    prior_features,
)
from tb_risk.validation.real_data_infection import (  # noqa: E402
    feature_columns as inf_feature_columns,
    load_homeacf_contacts,
)
from tb_risk.validation.real_data_pi import (  # noqa: E402
    _group_cv_indices,
    _make_model,
)

OUT = os.path.join(HERE, 'processed',
                   'w_threshold_holdout_homeacf_%s.json'
                   % time.strftime('%Y%m%d'))
PHASE1_ARCHIVE = os.path.join(HERE, 'processed',
                               'w_threshold_validation_homeacf_'
                               '20260907.json')

W_GRID = (1.0, 3.0, 5.0, 7.0, 9.0, 12.0)
DEP_K = 2.0
N_SEEDS = 10
H_LEVELS = (50, 100, 200, 400)
R_SUB = {50: 20, 100: 20, 200: 20, 400: 10}
TRAIN_FRAC = 0.70

COLS_PRIOR = rsd.COLS_PRIOR


def rf_mean_predict(X_tr, y_tr, X_te, seeds):
    """10 种子均值预测（部署者多种子平均口径）。"""
    preds = []
    for s in seeds:
        m = _make_model('random_forest', s)
        m.fit(X_tr, y_tr)
        preds.append(m.predict_proba(X_te)[:, 1])
    return np.mean(preds, axis=0)


def main():
    t0 = time.time()
    df = load_homeacf_contacts()
    y_full = df['tst_pos10'].astype(int).to_numpy()
    g_full = df['record_id'].to_numpy()
    households = np.unique(g_full)
    cols_exp = inf_feature_columns('exposure')
    cols_A = cols_exp + COLS_PRIOR

    results_by_h = {}
    for h in H_LEVELS:
        reps = R_SUB[h]
        sub_rng = np.random.RandomState(1000 + h)   # 与第一阶段同流
        split_rng = np.random.RandomState(3000 + h)
        rows = []
        print('=== h=%d 户（R=%d，70/30 户级 holdout）===' % (h, reps))
        for rep in range(reps):
            chosen = sub_rng.choice(households, size=h, replace=False)
            mask = np.isin(g_full, chosen)
            df_sub = df[mask].reset_index(drop=True)
            y_s = df_sub['tst_pos10'].astype(int).to_numpy()
            g_s = df_sub['record_id'].to_numpy()
            if y_s.sum() < 5 or len(np.unique(y_s)) < 2:
                continue
            # 固定筛查顺序实现（部署者单顺序观测）
            order = assign_screening_order(df_sub, seed=rep, mode='random')
            pf = prior_features(df_sub, order)
            data = pd.concat([df_sub.reset_index(drop=True), pf], axis=1)
            pn = pf['prior_n'].to_numpy(dtype=float)
            pp = pf['prior_pos'].to_numpy(dtype=float)

            # 外层 70/30 户级分割
            hh = np.unique(g_s)
            n_tr_hh = max(5, int(round(TRAIN_FRAC * len(hh))))
            tr_hh = set(split_rng.choice(hh, size=n_tr_hh,
                                         replace=False).tolist())
            tr_m = np.isin(g_s, list(tr_hh))
            te_m = ~tr_m
            if y_s[te_m].sum() < 1 or y_s[tr_m].sum() < 3:
                continue

            # ---- 内层：§12 路径 B 协议选 w*（仅训练 70%）----
            a_dep = {w: [] for w in W_GRID}
            for seed in range(N_SEEDS):
                folds = _group_cv_indices(g_s[tr_m], y_s[tr_m],
                                          n_splits=5, seed=seed)
                X_tr_exp = np.nan_to_num(
                    data[cols_exp].to_numpy(dtype=float)[tr_m], nan=0.0)
                oof_exp = np.zeros(int(tr_m.sum()))
                for tr_i, te_i in folds:
                    m = _make_model('random_forest', seed)
                    m.fit(X_tr_exp[tr_i], y_s[tr_m][tr_i])
                    oof_exp[te_i] = m.predict_proba(
                        X_tr_exp[te_i])[:, 1]
                pi_in = float(y_s[tr_m].mean())
                for w in W_GRID:
                    sc = rsd.dep_scores(oof_exp, pn[tr_m], pp[tr_m],
                                        pi_in, DEP_K, w)
                    a_dep[w].append(rsd._auroc(y_s[tr_m], sc))
            med = {w: float(np.median(v)) for w, v in a_dep.items()}
            w_star = None
            best = -np.inf
            for w in sorted(med):
                if med[w] > best:
                    best, w_star = med[w], w

            # ---- 外层：干净评估（holdout 未见 w 选择）----
            X_all_exp = np.nan_to_num(
                data[cols_exp].to_numpy(dtype=float), nan=0.0)
            X_all_A = np.nan_to_num(
                data[cols_A].to_numpy(dtype=float), nan=0.0)
            seeds = tuple(range(N_SEEDS))
            p_base_te = rf_mean_predict(
                X_all_exp[tr_m], y_s[tr_m], X_all_exp[te_m], seeds)
            p_A_te = rf_mean_predict(
                X_all_A[tr_m], y_s[tr_m], X_all_A[te_m], seeds)
            pi_dep = float(y_s[tr_m].mean())   # 部署者 π̂（训练率）
            b_recal_te = rsd.dep_scores(
                p_base_te, pn[te_m], pp[te_m], pi_dep, DEP_K, w_star)
            b_w7_te = rsd.dep_scores(
                p_base_te, pn[te_m], pp[te_m], pi_dep, DEP_K, 7.0)
            y_te = y_s[te_m]
            rows.append({
                'n': int(len(y_s)), 'events': int(y_s.sum()),
                'n_holdout': int(te_m.sum()),
                'events_holdout': int(y_te.sum()),
                'w_star': w_star,
                'auroc_A': rsd._auroc(y_te, p_A_te),
                'auroc_B_recal': rsd._auroc(y_te, b_recal_te),
                'auroc_B_w7': rsd._auroc(y_te, b_w7_te),
                'delta_A_minus_Brecal': rsd._auroc(y_te, p_A_te)
                - rsd._auroc(y_te, b_recal_te),
                'delta_Bw7_minus_Brecal': rsd._auroc(y_te, b_w7_te)
                - rsd._auroc(y_te, b_recal_te),
            })
        valid = rows
        w_counts = {}
        for x in valid:
            k = '%g' % x['w_star']
            w_counts[k] = w_counts.get(k, 0) + 1

        def _sum(key):
            d = np.asarray([x[key] for x in valid], dtype=float)
            return {'mean': round(float(d.mean()), 5),
                    'sd': round(float(d.std(ddof=1)), 5),
                    'min': round(float(d.min()), 5),
                    'max': round(float(d.max()), 5),
                    'share_positive': round(float((d > 0).mean()), 3),
                    'n': int(len(d))}

        agg = {
            'h_households': h,
            'n_subsamples': len(valid),
            'mean_events': round(float(np.mean(
                [x['events'] for x in valid])), 1),
            'mean_events_holdout': round(float(np.mean(
                [x['events_holdout'] for x in valid])), 1),
            'w_star_distribution': w_counts,
            'share_w7': round(w_counts.get('7', 0) / len(valid), 3),
            'share_w12': round(w_counts.get('12', 0) / len(valid), 3),
            'auroc_A': _sum('auroc_A'),
            'auroc_B_recal': _sum('auroc_B_recal'),
            'auroc_B_w7': _sum('auroc_B_w7'),
            'delta_A_minus_Brecal': _sum('delta_A_minus_Brecal'),
            'delta_Bw7_minus_Brecal': _sum('delta_Bw7_minus_Brecal'),
        }
        results_by_h['%d' % h] = agg
        print('  events=%.0f (holdout %.0f) | w*: 7=%2d%% 12=%2d%% | '
              'Δ(A−B*) mean=%+.4f share+=%.2f | '
              'Δ(B7−B*) mean=%+.4f share+=%.2f' % (
                  agg['mean_events'], agg['mean_events_holdout'],
                  100 * agg['share_w7'], 100 * agg['share_w12'],
                  agg['delta_A_minus_Brecal']['mean'],
                  agg['delta_A_minus_Brecal']['share_positive'],
                  agg['delta_Bw7_minus_Brecal']['mean'],
                  agg['delta_Bw7_minus_Brecal']['share_positive']))

    r50, r100, r200, r400 = (results_by_h['%d' % h] for h in H_LEVELS)
    v5 = {
        'h50_A_beats_miscal': r50['delta_A_minus_Brecal']['mean'] > 0,
        'h100_A_beats_miscal': r100['delta_A_minus_Brecal']['mean'] > 0,
        'h50_w7_better_than_selected':
            r50['delta_Bw7_minus_Brecal']['mean'] > 0,
        'h100_w7_better_than_selected':
            r100['delta_Bw7_minus_Brecal']['mean'] > 0,
    }
    v6 = {
        'h400_B_not_worse': abs(
            r400['delta_A_minus_Brecal']['mean']) <= 0.010,
        'h400_selection_accuracy': r400['share_w7'] >= 0.5,
    }
    print('\n=== 预声明判决 ===')
    print('  V5 门槛下选择伤害: %s' % v5)
    print('  V6 门槛上安全: %s' % v6)

    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'w_threshold_holdout_homeacf_v1',
        'status': '第一阶段 V2 乐观性污染修正：外层 70/30 户级 '
                  'holdout 分离选择与评估——门槛判决的干净读数',
        'protocol': {
            'outer': '70/30 户级分割（split_rng 独立于子抽样流，'
                     '子抽样与第一阶段同流复用）',
            'inner': '§12 路径 B 原文协议：StratifiedGroupKFold(5)×10 '
                     '种子，OOF AUROC 中位数选 w（并列取小）',
            'order': '每 rep 固定单一筛查顺序实现（order_seed=rep，'
                     '部署者单顺序观测——与第一阶段逐种子重抽的差异'
                     '已披露）',
            'pi': '训练折率（部署者 π̂）',
            'rf': '外层 RF 基座与路径 A 均为 10 种子均值预测',
        },
        'hypotheses': {
            'V5': 'h≤100：Δ(A−B*)>0 且 Δ(B_w7−B*)>0（误选有害）',
            'V6': 'h=400：|Δ(A−B*)|≤0.010 且 share(w*=7)≥0.5',
        },
        'verdicts': {'V5': v5, 'V6': v6},
        'subsample_scan': results_by_h,
        'honest_boundaries': [
            'h=50 层 holdout 事件均值≈6——单 rep 噪声大，20 rep 均值'
            '为主读数（无偏性保持）',
            '第一阶段 V1/V2 预声明判据未兑现的两点原因（选择乐观性'
            '污染 + 失败模型预期错误）已在本阶段修正——两阶段均归档，'
            '不作事后改写',
            '单一源人群（HomeACF）；人群平移维度仍由 V4 Muchuro '
            '天然实验（n=1）支撑',
            '固定单一筛查顺序实现与第一阶段逐种子重抽的口径差异'
            '（部署现实 vs 方差平均）',
        ],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
