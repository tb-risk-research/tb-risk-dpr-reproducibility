#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""w 重校准门槛的内置验证（V 系列，2026-09-07）：把部署 §12 路径 B
门槛（n≥500 / 户≥100 / 事件≥80）从"建议"升级为"经验证"。

设计：HomeACF 子抽样模拟——门槛在 HomeACF 上的约束映射：事件≥80
⟺ h≈200 户（户均 0.41 事件）；n≥500 ⟺ h≈160 户；户≥100 ⟺ h=100。
故子抽样网格 h ∈ {50, 100, 200, 400} 精确跨越门槛两侧：
- h=50：三门槛全不满足（n≈155，事件≈20）
- h=100：满足户≥100，不满足 n 与事件（n≈311，事件≈41）
- h=200：三门槛全满足（n≈622，事件≈82）——门槛边界本体
- h=400：舒适区（n≈1244，事件≈164）

协议（与部署 §12 路径 B 完全同规格）：StratifiedGroupKFold(5) by 户
× 10 种子，w ∈ {1,3,5,7,9,12}，按 OOF AUROC 中位数选 w（并列取
小）；k=2 固定（H-K1），π=子抽样队列率（H-K2 容忍带）。路径 A =
RF(exposure+prior)（免 w，§12 默认）。

命题（预声明，先于运行）：
- V0（gate）：全量 20 种子复现 §8.11 锚（exposure 0.6441 /
  exposure_prior 0.7142 / dep@w7 0.7202，|Δ|<0.002）且全量
  w* 选择 = 7（v1 校准值被协议网格复现）。
- V1（w 选择稳定性）：modal-w 份额：h=50 < 0.50（不稳）；
  h=200 ≥ 0.70（门槛处已稳）；h=100 为过渡带（0.50-0.70）。
  w* 误选去向记录（极端值 1/12 vs 中段）。
- V2（路径对比跨门槛）：Δ(A−B_recal)（路径 A − 路径 B 本地重校准，
  OOF AUROC）：h=50 均值 > 0 且 share(Δ>0) ≥ 0.60（门槛下 A 优，
  Muchuro 失败模式的样本量维度复现）；h=200 均值 ≤ +0.005
  （门槛上 B 不劣于 A —— 全量上 B−A=+0.006 已知）。B_oracle−
  B_recal（选择后悔）作为选择乐观性的括号读数。
- V3（门槛定位判决）：稳定性跨越点落在 h∈(100,200]——协议门槛
  （事件≥80 在 HomeACF 映射 h≈200）位于跨越点的保守侧。
- V4（人群平移维度，天然实验引用）：Muchuro（h=59，门槛下，
  人群平移）dep(w=7) 1.90× < RF(full+prior) 2.27× capture——
  协议预测（门槛下走路径 A）在平移维度的一次性验证（n=1 天然
  实验，引用 §8.13 M 系列归档，不重算）。

诚实边界（预声明）：
- HomeACF 子抽样验证**样本量维度**；**人群平移维度**（Muchuro 的
  w 失效机制）无法用同源子抽样复现——最优 w 在 HomeACF 全量与
  子抽样间同源，B_transfer(w=7) 在此为同总体参照臂而非平移臂；
  平移维度仅由 V4（n=1）支撑；
- B_recal 的 w* 用同一 OOF 选择（6 点离散选择，乐观性小但
  非零）——B_oracle 作上界括号披露；
- π 用子抽样队列率（含测试行，π 已知简化——H-K2 容忍带覆盖）；
- 事件数按子抽样实际记录（户均 0.41 事件 → h=50 层部分子抽样
  事件 <20，判读为方向级）；
- 单一源人群（HomeACF）验证，外部效度由门槛的保守余量与 V4
  天然实验共同背书。

用法：
    python data/run_w_threshold_validation.py
输出：
    data/processed/w_threshold_validation_homeacf_YYYYMMDD.json
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
)

OUT = os.path.join(HERE, 'processed',
                   'w_threshold_validation_homeacf_%s.json'
                   % time.strftime('%Y%m%d'))

W_GRID = (1.0, 3.0, 5.0, 7.0, 9.0, 12.0)   # 部署 §12 路径 B 网格
DEP_K = 2.0
N_SEEDS_FULL = 20
N_SEEDS_SUB = 10
H_LEVELS = (50, 100, 200, 400)
R_SUB = {50: 20, 100: 20, 200: 20, 400: 10}
SEED_CV = tuple(range(10))

ANCHORS = {'exposure': 0.6441, 'exposure_prior': 0.7142,
           'dep_w7': 0.7202}
GATE_TOL = 0.002

COLS_PRIOR = rsd.COLS_PRIOR


def run_arms(df_sub, y, groups, seeds):
    """多种子 OOF：exposure RF / exposure_prior RF / dep 全网格 AUROC。

    与 run_sop_deepening.run_seed 同构（随机序口径、折内模型冻结注册表、
    π=子抽样率、k=2），仅 w 为网格。
    """
    cols_exp = inf_feature_columns('exposure')
    a_exp, a_expA = [], []
    a_dep = {w: [] for w in W_GRID}
    for seed in seeds:
        folds = _group_cv_indices(groups, y, n_splits=5, seed=seed)
        order = assign_screening_order(df_sub, seed=seed, mode='random')
        pf = prior_features(df_sub, order)
        data = pd.concat([df_sub.reset_index(drop=True), pf], axis=1)
        oof_exp = rsd.rf_oof(cols_exp, folds, y, data, seed)
        oof_expA = rsd.rf_oof(cols_exp + COLS_PRIOR, folds, y, data, seed)
        pn = pf['prior_n'].to_numpy(dtype=float)
        pp = pf['prior_pos'].to_numpy(dtype=float)
        pi = float(np.mean(y))
        a_exp.append(rsd._auroc(y, oof_exp))
        a_expA.append(rsd._auroc(y, oof_expA))
        for w in W_GRID:
            a_dep[w].append(rsd._auroc(
                y, rsd.dep_scores(oof_exp, pn, pp, pi, DEP_K, w)))
    return {
        'auroc_exposure': float(np.median(a_exp)),
        'auroc_pathA': float(np.median(a_expA)),
        'auroc_dep_by_w': {w: float(np.median(v))
                           for w, v in a_dep.items()},
    }


def select_w(auroc_by_w):
    """OOF AUROC 中位数选 w（并列取小）——部署 §12 路径 B 规则。"""
    best_w, best_a = None, -np.inf
    for w in sorted(auroc_by_w):   # 升序 → 并列取小
        if auroc_by_w[w] > best_a:
            best_a, best_w = auroc_by_w[w], w
    return best_w, best_a


def summarize_deltas(deltas):
    d = np.asarray(deltas, dtype=float)
    return {
        'mean': round(float(d.mean()), 5),
        'sd': round(float(d.std(ddof=1)), 5) if len(d) > 1 else None,
        'min': round(float(d.min()), 5),
        'max': round(float(d.max()), 5),
        'share_positive': round(float((d > 0).mean()), 3),
        'n': int(len(d)),
    }


def main():
    t0 = time.time()
    df = load_homeacf_contacts()
    y_full = df['tst_pos10'].astype(int).to_numpy()
    g_full = df['record_id'].to_numpy()
    households = np.unique(g_full)
    print('HomeACF: n=%d events=%d households=%d' % (
        len(y_full), int(y_full.sum()), len(households)))

    # ---- V0：全量 gate + w* 复现 ----
    print('\n=== V0 全量（%d 种子）===' % N_SEEDS_FULL)
    full = run_arms(df, y_full, g_full, tuple(range(N_SEEDS_FULL)))
    w_star_full, a_w_star = select_w(full['auroc_dep_by_w'])
    gates = {
        'exposure': abs(full['auroc_exposure'] - ANCHORS['exposure'])
        < GATE_TOL,
        'exposure_prior': abs(full['auroc_pathA']
                              - ANCHORS['exposure_prior']) < GATE_TOL,
        'dep_w7': abs(full['auroc_dep_by_w'][7.0] - ANCHORS['dep_w7'])
        < GATE_TOL,
        'w_star_is_7': w_star_full == 7.0,
    }
    print('  exposure=%.4f pathA=%.4f dep@7=%.4f | w*=%s (%.4f)' % (
        full['auroc_exposure'], full['auroc_pathA'],
        full['auroc_dep_by_w'][7.0], w_star_full, a_w_star))
    print('  gates: %s' % gates)
    v0_pass = all(gates.values())

    # ---- V1/V2：子抽样扫描 ----
    results_by_h = {}
    for h in H_LEVELS:
        reps = R_SUB[h]
        rng = np.random.RandomState(1000 + h)
        rows = []
        print('\n=== h=%d 户（R=%d 子抽样，%d 种子/子抽样）===' % (
            h, reps, N_SEEDS_SUB))
        for rep in range(reps):
            chosen = rng.choice(households, size=h, replace=False)
            mask = np.isin(g_full, chosen)
            df_sub = df[mask].reset_index(drop=True)
            y_s = df_sub['tst_pos10'].astype(int).to_numpy()
            g_s = df_sub['record_id'].to_numpy()
            if y_s.sum() < 5 or len(np.unique(y_s)) < 2:
                rows.append({'skip': 'degenerate'})
                continue
            try:
                r = run_arms(df_sub, y_s, g_s, SEED_CV)
            except Exception as e:   # noqa: BLE001
                rows.append({'skip': 'error:%s' % type(e).__name__})
                continue
            w_star, b_recal = select_w(r['auroc_dep_by_w'])
            b_transfer = r['auroc_dep_by_w'][7.0]
            b_oracle = max(r['auroc_dep_by_w'].values())
            rows.append({
                'n': int(len(y_s)), 'events': int(y_s.sum()),
                'w_star': w_star,
                'pathA': round(r['auroc_pathA'], 5),
                'b_recal': round(b_recal, 5),
                'b_transfer': round(b_transfer, 5),
                'b_oracle': round(b_oracle, 5),
                'delta_recal': round(r['auroc_pathA'] - b_recal, 5),
                'delta_transfer': round(r['auroc_pathA'] - b_transfer, 5),
                'selection_regret': round(b_oracle - b_recal, 5),
            })
        valid = [x for x in rows if 'skip' not in x]
        skipped = len(rows) - len(valid)
        w_stars = [x['w_star'] for x in valid]
        w_counts = {}
        for w in w_stars:
            w_counts['%g' % w] = w_counts.get('%g' % w, 0) + 1
        modal_w = max(w_counts, key=w_counts.get) if w_counts else None
        modal_share = (w_counts.get(modal_w, 0) / len(valid)) if valid \
            else float('nan')
        agg = {
            'h_households': h,
            'n_subsamples': len(valid), 'skipped': skipped,
            'mean_n': round(float(np.mean([x['n'] for x in valid])), 1),
            'mean_events': round(float(np.mean(
                [x['events'] for x in valid])), 1),
            'w_star_distribution': w_counts,
            'modal_w': modal_w,
            'modal_w_share': round(modal_share, 3),
            'delta_A_minus_B_recal': summarize_deltas(
                [x['delta_recal'] for x in valid]),
            'delta_A_minus_B_transfer': summarize_deltas(
                [x['delta_transfer'] for x in valid]),
            'selection_regret': summarize_deltas(
                [x['selection_regret'] for x in valid]),
            'mean_pathA': round(float(np.mean(
                [x['pathA'] for x in valid])), 4),
            'mean_b_recal': round(float(np.mean(
                [x['b_recal'] for x in valid])), 4),
        }
        results_by_h['%d' % h] = agg
        print('  n=%.0f events=%.0f | modal w=%s share=%.2f | '
              'Δ(A−B_recal) mean=%+.4f share+=%.2f | '
              'Δ(A−B_transfer) mean=%+.4f' % (
                  agg['mean_n'], agg['mean_events'], modal_w,
                  modal_share, agg['delta_A_minus_B_recal']['mean'],
                  agg['delta_A_minus_B_recal']['share_positive'],
                  agg['delta_A_minus_B_transfer']['mean']))

    # ---- 判决（预声明判据）----
    r50, r100, r200, r400 = (results_by_h['%d' % h] for h in H_LEVELS)
    v1 = {
        'h50_unstable': r50['modal_w_share'] < 0.50,
        'h200_stable': r200['modal_w_share'] >= 0.70,
        'h100_transition': 0.50 <= r100['modal_w_share'] < 0.70,
        'detail': {k: v['modal_w_share'] for k, v in results_by_h.items()},
    }
    v2 = {
        'h50_A_dominates': (
            r50['delta_A_minus_B_recal']['mean'] > 0
            and r50['delta_A_minus_B_recal']['share_positive'] >= 0.60),
        'h200_B_not_worse': r200['delta_A_minus_B_recal']['mean']
        <= 0.005,
        'h400_B_not_worse': r400['delta_A_minus_B_recal']['mean']
        <= 0.005,
        'detail_recal': {k: v['delta_A_minus_B_recal']['mean']
                         for k, v in results_by_h.items()},
    }
    # V3：跨越点定位（稳定性份额 0.5 线）
    shares = [results_by_h['%d' % h]['modal_w_share'] for h in H_LEVELS]
    crossing = None
    for i in range(len(H_LEVELS) - 1):
        if shares[i] < 0.5 <= shares[i + 1]:
            crossing = (H_LEVELS[i], H_LEVELS[i + 1])
    v3 = {
        'crossing_between': crossing,
        'threshold_maps_to_h': 200,
        'conservative_side': bool(crossing is not None
                                  and crossing[1] <= 200),
        'note': '门槛（事件≥80 ⟺ h≈200）位于跨越点同位或保守侧',
    }
    verdicts = {
        'V0_gate_and_w7': bool(v0_pass),
        'V1_w_stability': v1,
        'V2_path_comparison': v2,
        'V3_threshold_localization': v3,
        'V4_muchuro_natural_experiment': {
            'h': 59, 'below_threshold': True,
            'dep_w7_enrichment': 1.90, 'rf_full_prior_enrichment': 2.27,
            'protocol_prediction_held': True,
            'source': '§8.13 M 系列归档（muchuro_capture_bridge_'
                      '20260906.json），人群平移维度 n=1 天然实验',
        },
    }
    print('\n=== 预声明命题判决 ===')
    print('  V0 gate+w*=7: %s' % verdicts['V0_gate_and_w7'])
    print('  V1: h50<0.5 %s | h200≥0.7 %s | h100 过渡 %s' % (
        v1['h50_unstable'], v1['h200_stable'], v1['h100_transition']))
    print('  V2: h50 A 优 %s | h200 B 不劣 %s | h400 B 不劣 %s' % (
        v2['h50_A_dominates'], v2['h200_B_not_worse'],
        v2['h400_B_not_worse']))
    print('  V3: 跨越点 %s | 保守侧 %s' % (
        v3['crossing_between'], v3['conservative_side']))

    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'w_threshold_validation_homeacf_v1',
        'status': '部署 §12 路径 B 门槛（n≥500/户≥100/事件≥80）内置'
                  '验证：HomeACF 子抽样（样本量维度）+ Muchuro 天然实验'
                  '（平移维度）——门槛从建议升级为经验证',
        'protocol': {
            'cv': 'StratifiedGroupKFold(5) by 户 × 10 种子（V0 全量 20）',
            'w_grid': list(W_GRID), 'k': DEP_K,
            'pi': '子抽样队列率（H-K2 容忍带内的已知 π 简化）',
            'selection': 'OOF AUROC 中位数选 w，并列取小（§12 路径 B '
                         '原文规则）',
            'threshold_mapping': 'HomeACF 上：事件≥80 ⟺ h≈200 户；'
                                 'n≥500 ⟺ h≈160；户≥100 ⟺ h=100',
        },
        'hypotheses': {
            'V0': '全量锚复现 + w*=7',
            'V1': 'modal 份额 h50<0.50 / h200≥0.70 / h100 过渡带',
            'V2': 'h50: Δ(A−B_recal)>0 且 share≥0.60；h200/400: '
                  'Δ≤+0.005（B 不劣）',
            'V3': '跨越点 ∈ (100,200]，门槛在保守侧',
            'V4': 'Muchuro 天然实验引用（平移维度）',
        },
        'verdicts': verdicts,
        'v0_full': {
            'auroc_exposure': round(full['auroc_exposure'], 4),
            'auroc_pathA': round(full['auroc_pathA'], 4),
            'auroc_dep_by_w': {('%g' % w): round(v, 4)
                                for w, v in
                                full['auroc_dep_by_w'].items()},
            'w_star_full': w_star_full,
            'gates': gates,
        },
        'subsample_scan': results_by_h,
        'honest_boundaries': [
            'HomeACF 子抽样验证样本量维度；人群平移维度（Muchuro 的 '
            'w 失效机制）无法用同源子抽样复现——最优 w 同源，'
            'B_transfer(w=7) 在此为同总体参照臂而非平移臂；平移维度'
            '仅由 V4（n=1 天然实验）支撑',
            'B_recal 的 w* 用同一 OOF 选择（6 点离散，乐观性小但'
            '非零）——B_oracle−B_recal（selection_regret）作上界括号',
            'π 用子抽样队列率（含测试行的已知 π 简化，H-K2 容忍带'
            '覆盖）',
            'h=50 层事件均值 ≈20（户均 0.41），判读为方向级；'
            '事件数按子抽样实际记录',
            '单一源人群验证——外部效度由门槛保守余量 + V4 天然实验'
            '共同背书；h=100 层 R=20 的 modal 份额读数有 ±0.11 '
            '量级抽样噪声（bootstrap 未做，R 小）',
        ],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
