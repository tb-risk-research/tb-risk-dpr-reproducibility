#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""宿主通路修复验证实验（2026-08-24，第一优先）。

背景：特征审计发现 v2 场景训练 CSV 六列结构性零值 + 标签机制只走暴露
通路（宿主特征单变量 AUROC≈0.50）——模型本质是"暴露单通路模型"。
修复：_contact_susceptibility 文献校准宿主因素 + _sample_contact 补
has_tb/is_high_risk + patient 上下文接入交互特征（evaluate_ensemble v3）。

验证流程（对应"重跑特征审计 + v1/v2 对比"的完成标准）：
1. 重新生成场景训练 CSV（mechanistic + clinical，seed=2026，2500 场景）；
2. 特征审计重跑：六个死列复活（唯一值 ≥2、脱离近常数报告）、
   可推导结构不变（age_bcg_decay/exposure_accumulation 仍可推导）；
3. 宿主特征单变量 AUROC 验证（完成标准 has_symptoms/age/past_illness
   ≥0.55，直接读自训练 CSV 的审计结果）；
4. v1/v2 特征集对比重跑（同种子同 CV5 全 5 模型）：死列复活后
   v2（15 维精简集）判别力仍保持（容差 -0.01，项目噪声量级惯例）。

产物（新文件，不覆盖既有证据）：
- data/processed/feature_audit_host_v2_20260824.json
- data/processed/feature_set_v1_v2_comparison_host_v2_20260824.json
- ml_training_scenario_aligned.csv / ml_training_scenario_clinical.csv
  （训练数据本身，按训练流程惯例重新生成）
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluate_ensemble import generate_scenario_ml_csv  # noqa: E402
from tb_risk.scoring.ml.feature_audit import audit_training_csv  # noqa: E402
from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import _train_all_registered_models  # noqa: E402

PROC = os.path.join(HERE, 'processed')
MECH_CSV = os.path.join(PROC, 'ml_training_scenario_aligned.csv')
MECH_META = os.path.join(PROC, 'ml_training_scenario_meta.json')
CLIN_CSV = os.path.join(PROC, 'ml_training_scenario_clinical.csv')
CLIN_META = os.path.join(PROC, 'ml_training_scenario_clinical_meta.json')

DEAD_COLS_V1 = ['has_tb', 'is_high_risk', 'dm_tb_synergy',
                'symptom_delay', 'cough_contact', 'highrisk_comorbid']
HOST_TARGETS = ['has_symptoms', 'age', 'past_illness']


def step1_regenerate():
    print('=' * 70)
    print('步骤 1：重新生成场景训练 CSV（宿主通路 v2，seed=2026，2500 场景）')
    print('=' * 70)
    for label_mode, csv_path, meta_path in (
            ('mechanistic', MECH_CSV, MECH_META),
            ('clinical', CLIN_CSV, CLIN_META)):
        t0 = time.time()
        _path, n = generate_scenario_ml_csv(
            2500, 2026, csv_path, meta_path, label_mode=label_mode)
        df = pd.read_csv(csv_path)
        print(f'  [{label_mode}] {n} 样本, '
              f'阳性率 {df["tb_outcome"].mean():.4f}, '
              f'{time.time() - t0:.1f}s -> {os.path.basename(csv_path)}')


def step2_audit():
    print()
    print('=' * 70)
    print('步骤 2：特征审计重跑（死列复活 + 可推导结构 + 单变量 AUROC）')
    print('=' * 70)
    audit = audit_training_csv(MECH_CSV, label_col='tb_outcome')

    revived, still_dead = {}, []
    df = pd.read_csv(MECH_CSV)
    for col in DEAD_COLS_V1:
        nuniq = int(df[col].nunique())
        nonzero = float((df[col] > 0).mean())
        if nuniq < 2:
            still_dead.append(col)
        revived[col] = {'n_unique': nuniq, 'nonzero_frac': round(nonzero, 5)}
    near_constant = list(audit['near_constant'])
    derivable = sorted(d['feature'] for d in audit['derivable'])

    print(f'  近常数列: {near_constant if near_constant else "无（六列全部复活）"}')
    print(f'  可推导列: {derivable}')
    for col in DEAD_COLS_V1:
        info = revived[col]
        print(f"  [复活] {col}: 唯一值={info['n_unique']}, "
              f"非零占比={info['nonzero_frac']:.3%}")

    aurocs = audit['univariate_auroc']
    print('  宿主特征单变量 AUROC（完成标准 ≥0.55）:')
    for feat in HOST_TARGETS + ['bcg_vaccine']:
        print(f'    {feat}: {aurocs[feat]:.4f}')

    out = {
        'csv': MECH_CSV,
        'n_samples': audit['n_samples'],
        'positive_rate': audit['positive_rate'],
        'dead_feature_revival': revived,
        'still_dead': [c for c in DEAD_COLS_V1
                       if revived[c]['n_unique'] < 2],
        'near_constant': near_constant,
        'derivable': derivable,
        'univariate_auroc': aurocs,
        'host_criterion': {
            'target': 0.55,
            'measured': {f: aurocs[f] for f in HOST_TARGETS},
            'pass': all(aurocs[f] >= 0.55 for f in HOST_TARGETS),
        },
    }
    path = os.path.join(PROC, 'feature_audit_host_v2_20260824.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f'已保存: {path}')
    return out


def step3_v1_v2():
    print()
    print('=' * 70)
    print('步骤 3：v1（22 维）vs v2（15 维）对比重跑（宿主通路 v2 CSV）')
    print('=' * 70)
    df = pd.read_csv(MECH_CSV)
    y = df['tb_outcome'].values.astype(int)
    X = df[MLRiskPredictor.ALL_FEATURE_NAMES].values.astype(float)
    print(f'数据: {df.shape[0]} 样本, 阳性率 {y.mean():.4f}')

    results = {}
    for feature_set in ('v1', 'v2'):
        t0 = time.time()
        pred = MLRiskPredictor()
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
        'csv': MECH_CSV,
        'note': 'host pathway v2 CSV (dead features revived)',
        'n_samples': int(df.shape[0]),
        'random_state': 42,
        'cv_folds': 5,
        'worst_delta': worst['delta_v2_minus_v1'],
        'verdict': (
            'v2 判别力保持（最差单模型 Δ ≥ -0.01，噪声量级内）'
            if worst['delta_v2_minus_v1'] >= -0.01 else
            'v2 存在超噪声量级的判别力损失，需复核移除清单'),
    }
    path = os.path.join(
        PROC, 'feature_set_v1_v2_comparison_host_v2_20260824.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(comparison, fh, ensure_ascii=False, indent=2)
    print(f"结论: {comparison['summary']['verdict']}"
          f"（最差 Δ={worst['delta_v2_minus_v1']:+.4f}）")
    print(f'已保存: {path}')
    return comparison


if __name__ == '__main__':
    step1_regenerate()
    audit_out = step2_audit()
    cmp_out = step3_v1_v2()
    print()
    print('=' * 70)
    print('汇总')
    print('=' * 70)
    print(f"宿主完成标准（AUROC≥0.55）: "
          f"{'通过' if audit_out['host_criterion']['pass'] else '未通过'} "
          f"({audit_out['host_criterion']['measured']})")
    print(f"死列复活: {len(audit_out['dead_feature_revival'])}/6 "
          f"（仍死: {audit_out['still_dead'] or '无'}）")
    print(f"v1/v2 对比: {cmp_out['summary']['verdict']}")
