#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题3/5 验证实验（2026-08-24）：v1/v2 特征集对比 + 消融全量运行。

实验 A（特征精简的证据）：同一场景训练 CSV（mechanistic 标签）上，
v1（22 维）vs v2（15 维精简集）各训练全部 5 个基模型（同种子、同 CV5），
逐模型对比 AUROC——精简标准是"独立临床含义且无法互相推导"，
判别力不应显著下降（容差 0.01，与项目"单种子增益 <0.01 是噪声"惯例一致）。

实验 B（消融统计效力升级的全量运行）：run_multi_seed_ablation 默认
500 接触者 × 20 种子（用户规格），产出效应量 + bootstrap 95% CI +
逐种子 DeLong p + 先验功效分析（Hanley-McNeil）。

产物（只追加不覆盖）：
- data/processed/feature_set_v1_v2_comparison_20260824.json
- data/processed/layer_ablation_multiseed_20260824.json
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import _train_all_registered_models  # noqa: E402
from tb_risk.validation.layer_ablation import run_multi_seed_ablation  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'processed')
CSV = os.path.join(OUT_DIR, 'ml_training_scenario_aligned.csv')


class _ComparisonPredictor(MLRiskPredictor):
    """复用真实特征名/描述，仅替换训练入口侧不需要的行为。"""


def experiment_a():
    print('=' * 70)
    print('实验 A：v1（22 维）vs v2（15 维）特征集对比（mech 场景 CSV）')
    print('=' * 70)
    df = pd.read_csv(CSV)
    y = df['tb_outcome'].values.astype(int)
    X = df[MLRiskPredictor.ALL_FEATURE_NAMES].values.astype(float)
    print(f'数据: {df.shape[0]} 样本, 阳性率 {y.mean():.4f}')

    results = {}
    for feature_set in ('v1', 'v2'):
        t0 = time.time()
        pred = _ComparisonPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42, data_source_label='synthetic',
            enable_calibration=False, feature_set=feature_set)
        assert ok, f'{feature_set} 训练失败'
        aurocs = {key: entry['AUROC']
                  for key, entry in pred.model_performance.items()}
        results[feature_set] = aurocs
        print(f'[{feature_set}] {time.time() - t0:.1f}s '
              + ' '.join(f'{k}={v:.4f}' for k, v in aurocs.items()))

    comparison = {'model': {key: {
        'auroc_v1': round(results['v1'][key], 4),
        'auroc_v2': round(results['v2'][key], 4),
        'delta_v2_minus_v1': round(results['v2'][key] - results['v1'][key], 4),
    } for key in results['v1']}}
    worst = min(comparison['model'].values(),
                key=lambda d: d['delta_v2_minus_v1'])
    comparison['summary'] = {
        'csv': CSV,
        'n_samples': int(df.shape[0]),
        'random_state': 42,
        'cv_folds': 5,
        'worst_delta': worst['delta_v2_minus_v1'],
        'verdict': (
            'v2 判别力保持（最差单模型 Δ ≥ -0.01，噪声量级内）'
            if worst['delta_v2_minus_v1'] >= -0.01 else
            'v2 存在超噪声量级的判别力损失，需复核移除清单'),
    }
    out = os.path.join(OUT_DIR, 'feature_set_v1_v2_comparison_20260824.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(comparison, fh, ensure_ascii=False, indent=2)
    print(f"结论: {comparison['summary']['verdict']}"
          f"（最差 Δ={worst['delta_v2_minus_v1']:+.4f}）")
    print(f'已保存: {out}')
    return comparison


def experiment_b():
    print()
    print('=' * 70)
    print('实验 B：网络层消融全量运行（500 接触者 × 20 种子，用户规格）')
    print('=' * 70)
    t0 = time.time()
    report = run_multi_seed_ablation(n_contacts=500, n_seeds=20,
                                     random_state=42)
    print(f'耗时 {time.time() - t0:.1f}s')
    es = report['effect_size']
    ci = report['bootstrap_ci']
    dl = report['delong']
    pd_ = report['power_design']
    print(f"效应量 ΔAUROC = {es['delta_auroc_mean']:.4f} "
          f"± {es['delta_auroc_std']:.4f}（范围 {es['delta_auroc_min']:.4f}"
          f"~{es['delta_auroc_max']:.4f}）")
    print(f"bootstrap 95% CI = [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
          f"（种子级重采样 n={ci['n_bootstrap']}）")
    print(f"DeLong 配对检验: 中位 p={dl['p_median']:.2e}, "
          f"显著种子 {dl['n_significant']}/{report['n_seeds']}")
    print(f"基线 AUROC（均值）= {es['auroc_baseline_mean']:.4f}, "
          f"完整 = {es['auroc_full_mean']:.4f}")
    print(f"功效设计（Δ=0.05, power=0.80）: "
          f"实验阳性率口径需 n="
          f"{pd_['required_n_experiment_prevalence']['n_total']}, "
          f"密接实际阳性率（2.85%）口径需 n="
          f"{pd_['required_n_close_contact_prevalence']['n_total']}")
    print(f"network_contributes = {report['network_contributes']}")
    print(f"结论: {report['conclusion']}")
    out = os.path.join(OUT_DIR, 'layer_ablation_multiseed_20260824.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f'已保存: {out}')
    return report


if __name__ == '__main__':
    experiment_a()
    experiment_b()
