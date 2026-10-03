#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 特征集在 HomeACF（南非家庭接触者队列）上的复验（2026-08-25
第四轮 P3）。

v3 = v2 个体 15 列 + 11 网络特征化列（typed 5 / window 5 /
physics_lambda_calibrated）。HomeACF 谱系的网络列可构造性：

  type_exposure_*（5）   真实版 = 关系类型哑元（rel_child / rel_spouse /
                        rel_sibling / rel_parent + lives_in 分级）——
                        已在 exposure 臂内承载，不重复计数；
  window_exposure_*（5） 无时间窗分解源字段 → 不可构造；
  physics_lambda_calibrated（1）
                        可构造：λ 精简子集（乘子审计移除 3 个方向
                        翻转项 age_lt5 / index_hiv / sleep_same_bed，
                        SIMPLIFIED_COMPONENTS 5 分量）的 OOF 校准
                        logit 作为 1 列特征进 RF（第三轮 P2 的
                        exposure_calsimpl_pi 臂，本复验以 v3 名义定版）。

臂结构（主模型 = 冻结注册表 RF，户分组 CV，全部 OOF）：
  ind:RF          个体基线（11 列）
  exposure:RF     + 指示病例 + 暴露分级（29 列；typed 真实版所在）
  v3_homeacf:RF   exposure + 精简校准 λ（30 列；v3 在本谱系的
                  可构造最大集）

验收（谱系降级语义 + λ 弱信号结论双确认）：
  v3_homeacf − exposure ≈ 0（|Δ| ≤ 0.01，第三轮结论"作为特征无害
  无益 -0.0016"的定版复现——物理 λ 在 LTBI 终点是弱信号）。
  v3_homeacf − ind 参考口径（exposure 家族在 HomeACF 对 ind 本就
  不显著，既往结论 ~+0.001；本复验不以其显著性为验收点——显著
  增益属于时序先验层，见 household_temporal 归档）。

产物（只追加不覆盖）：
- data/processed/feature_set_v3_homeacf_20260825.json
"""

import json
import os
import sys
import time

# 注意：插入 repo 的父目录（Desktop）而非 repo 本身——杂散
# tb_risk/tb_risk 命名空间目录会在后者情况下遮蔽真包。
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tb_risk.validation.lambda_calibration import (  # noqa: E402
    SIMPLIFIED_COMPONENTS,
    _oof_calibrated_lambda,
    _rf_oof_df,
)
from tb_risk.validation.real_data_infection import (  # noqa: E402
    feature_columns as inf_feature_columns,
    load_homeacf_contacts,
)
from tb_risk.validation.real_data_pi import (  # noqa: E402
    _cluster_bootstrap_delta,
    _group_cv_indices,
)
from tb_risk.validation.threshold_spec import compute_auc  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'processed')
OUT_JSON = os.path.join(OUT_DIR, 'feature_set_v3_homeacf_20260825.json')

ARMS = ('ind:RF', 'exposure:RF', 'v3_homeacf:RF')


def run_homeacf_v3_once(seed, n_splits, df):
    """单种子：三臂 OOF（ind / exposure / exposure+精简校准 λ）。"""
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    folds = _group_cv_indices(groups, y, n_splits=n_splits, seed=seed)

    # v3 网络列可构造部分：精简 λ 的 OOF 校准 logit（1 列）
    cal_logit_s, cal_coefs_s, cal_cols_s = _oof_calibrated_lambda(
        df, y, folds, seed, components=SIMPLIFIED_COMPONENTS)

    df_v3 = pd.concat(
        [df.reset_index(drop=True),
         pd.DataFrame({'cal_lambda_simplified': cal_logit_s})], axis=1)

    oof = {
        'ind:RF': _rf_oof_df(df, inf_feature_columns('ind'), y, folds, seed),
        'exposure:RF': _rf_oof_df(
            df, inf_feature_columns('exposure'), y, folds, seed),
        'v3_homeacf:RF': _rf_oof_df(
            df_v3, list(inf_feature_columns('exposure'))
            + ['cal_lambda_simplified'], y, folds, seed),
    }
    return {
        'seed': seed,
        'y': y,
        'groups': groups,
        'oof': oof,
        'auroc': {k: float(compute_auc(v, y.tolist()))
                  for k, v in oof.items()},
        'cal_coefs_simplified': cal_coefs_s,
        'cal_cols_simplified': cal_cols_s,
    }


def main(n_seeds=20, seed_start=0, n_splits=5, n_bootstrap=2000):
    print('=' * 70)
    print('HomeACF 复验：v3 特征集（可构造子集 = exposure + 精简校准 λ）')
    print('=' * 70)
    df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    print(f'数据: {len(df)} 接触者, {y.sum()} TST 阳性 ({y.mean():.3f}), '
          f'{pd.Series(groups).nunique()} 户')

    per_seed = []
    t0 = time.time()
    for i in range(n_seeds):
        rep = run_homeacf_v3_once(seed_start + i, n_splits, df)
        per_seed.append(rep)
        print(f'  seed {rep["seed"]:2d}: '
              + ' '.join(f'{k}={v:.4f}' for k, v in rep['auroc'].items()))
    print(f'{n_seeds} 种子完成 ({time.time() - t0:.1f}s)')

    # ---- 汇总：均值 + 种子级 sd ----
    summary = {}
    for arm in ARMS:
        vals = [rep['auroc'][arm] for rep in per_seed]
        summary[arm] = {
            'mean_auroc': round(float(np.mean(vals)), 4),
            'sd_across_seeds': round(float(np.std(vals)), 4),
        }

    # ---- 末种子 pooled OOF 的户级 cluster bootstrap（同 P1/P2 口径） ----
    last = per_seed[-1]
    deltas = {}
    for a, b in (('v3_homeacf:RF', 'exposure:RF'),
                 ('v3_homeacf:RF', 'ind:RF'),
                 ('exposure:RF', 'ind:RF')):
        deltas[f'{a} - {b}'] = _cluster_bootstrap_delta(
            last['oof'][a], last['oof'][b], last['y'], last['groups'],
            n_bootstrap=n_bootstrap, seed=last['seed'])

    v3_vs_exp = np.mean(
        [rep['auroc']['v3_homeacf:RF'] - rep['auroc']['exposure:RF']
         for rep in per_seed])
    v3_vs_ind = np.mean(
        [rep['auroc']['v3_homeacf:RF'] - rep['auroc']['ind:RF']
         for rep in per_seed])

    out = {
        'design': {
            'dataset': 'HomeACF（南非家庭接触者队列，LTBI 终点 tst_pos10）',
            'n_seeds': n_seeds, 'seed_start': seed_start,
            'n_splits': n_splits, 'n_bootstrap': n_bootstrap,
            'model': '冻结注册表 RF（scoring/ml/training.py '
                     'MODEL_REGISTRY，只注入种子）',
            'cv': 'StratifiedGroupKFold 户分组（record_id）',
            'v3_definition': (
                'SELECTED_FEATURES_V3 = v2 15 个体 + 11 网络特征化；'
                'HomeACF 谱系可构造子集 = exposure（29 列，含 typed '
                '真实版 rel_*/ts_*）+ 精简校准 λ（1 列）= 30 列'),
            'lineage_note': (
                'window 5 列无时间窗源字段不可构造；typed 5 列的真实版'
                '（关系/接触强度哑元）已含于 exposure 臂；λ 以乘子审计'
                '后精简子集（SIMPLIFIED_COMPONENTS）的 OOF 校准构造'),
            'simplified_components': list(SIMPLIFIED_COMPONENTS),
            'cal_cols': per_seed[0]['cal_cols_simplified'],
        },
        'arms_summary': summary,
        'deltas_cluster_bootstrap': deltas,
        'per_seed': [{'seed': r['seed'], 'auroc': r['auroc']}
                     for r in per_seed],
        'verdict': {
            'v3_minus_exposure_mean': round(float(v3_vs_exp), 4),
            'v3_minus_ind_mean': round(float(v3_vs_ind), 4),
            'lambda_weak_signal_confirmed': bool(abs(v3_vs_exp) <= 0.01),
            'exposure_family_gain_significant': bool(
                deltas['v3_homeacf:RF - ind:RF']['bootstrap_ci'][0] > 0),
            'interpretation': (
                'v3（HomeACF 可构造子集）与 exposure 基线持平'
                if abs(v3_vs_exp) <= 0.01 else
                'v3（HomeACF 可构造子集）与 exposure 基线偏离超噪声，'
                '需复核'),
        },
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print('\n'.join(f'[{k}] mean={v["mean_auroc"]:.4f}'
                    for k, v in summary.items()))
    for k, v in deltas.items():
        ci = v['bootstrap_ci']
        print(f'{k}: {v["mean"]:+.4f} CI [{ci[0]:+.4f}, {ci[1]:+.4f}] '
              f'p_positive={v["p_positive"]:.3f}')
    print(f"判定: {out['verdict']['interpretation']}"
          f"（v3 - exposure = {v3_vs_exp:+.4f}）")
    print(f'已保存: {OUT_JSON}')
    return out


if __name__ == '__main__':
    main()
