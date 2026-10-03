#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2/v3 特征集在公开数据（Kenya）上的复验（2026-08-25 第四轮 P3）。

背景：v3（26 维 = v2 个体 15 + 网络特征化 11）按组合机制消融结论
定版（合成 +0.06~0.09 次可加，不写三机制相加）。typed/window/
λ 的源字段（图邻接的边类型/时间窗分解）仅合成验证层存在——
Kenya 2016 全国 TB 患病率调查（63050 样本，细菌学确诊标签，
阳性率 0.53%）是**部署谱系**：v3 的 11 个网络列在本谱系全部
补零死列。本复验检验的就是谱系降级语义：**v3（补零）≈ v2，
死列不引入劣化**。

协议（与 v1/v2 Kenya 复验严格一致）：
- 同一 CSV、同种子 42、同 CV5，v2/v3 各训练全部 5 个基模型；
- v3 输入 = 22 列 v1 全名矩阵（numpy）→ 训练内核按位置选 v2 15
  列 + 网络列补零（谱系降级路径，dead 列审计随 entry 归档）；
- 判别力保持标准：最差单模型 Δ ≥ -0.01（项目"单种子增益 <0.01
  是噪声"惯例）；
- 死列审计：v3 的 11 网络列在 Kenya 谱系恒 0（源字段不存在，
  非数据缺失）。

产物（只追加不覆盖）：
- data/processed/feature_set_v2_v3_comparison_kenya_20260825.json
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

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import _train_all_registered_models  # noqa: E402
from tb_risk.scoring.ml.feature_audit import (  # noqa: E402
    NETWORK_FEATURE_NAMES_V3,
    SELECTED_FEATURES_V2,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'processed')
CSV = os.path.join(OUT_DIR, 'kenya_ml_training.csv')
OUT_JSON = os.path.join(OUT_DIR, 'feature_set_v2_v3_comparison_kenya_20260825.json')


def _add_interaction_features(df):
    """补齐 9 个交互特征列（公式与 train_from_real_data 一致，修改须同步）。"""
    if 'past_illness_type' not in df.columns:
        df['past_illness_type'] = 'none'
        df.loc[df['past_illness'] == 1, 'past_illness_type'] = 'other'
    hiv = (df['past_illness_type'].isin(['hiv', 'HIV'])).astype(float)
    diabetes = (df['past_illness_type'].isin(['diabetes', '糖尿病'])).astype(float)
    immunosupp = (df['past_illness_type'].isin(['immunosuppressants', '免疫抑制'])).astype(float)
    immuno_score = (hiv * 2.0 + diabetes * 1.0 + immunosupp * 1.5
                    + df['past_illness'].astype(float) * 0.5)
    df['age_immuno'] = df['age'].astype(float) * immuno_score
    df['dm_tb_synergy'] = diabetes * df['has_tb'].astype(float)
    df['age_bcg_decay'] = df['age'].astype(float) * (1 - df['bcg_vaccine'].astype(float))
    delay = df.get('delay_days', df.get('ftd', 0))
    delay = delay.astype(float) if hasattr(delay, 'astype') else float(delay)
    df['symptom_delay'] = df['has_symptoms'].astype(float) * np.minimum(delay / 30, 1.0)
    cough = df.get('cough_frequency', df.get('cough_freq', 0))
    cough = cough.astype(float) if hasattr(cough, 'astype') else float(cough)
    contacts = df.get('contact_count', 5)
    contacts = contacts.astype(float) if hasattr(contacts, 'astype') else float(contacts)
    df['cough_contact'] = np.minimum(cough / 20, 1.0) * (contacts / 10)
    df['highrisk_comorbid'] = df['is_high_risk'].astype(float) * df['past_illness'].astype(float)
    df['immune_bcg'] = (1 - df['bcg_vaccine'].astype(float)) * immuno_score
    ce = df.get('cumulative_exposure', 0)
    ce = ce.astype(float) if hasattr(ce, 'astype') else float(ce)
    ts = df.get('time_span', 4)
    ts = ts.astype(float) if hasattr(ts, 'astype') else float(ts)
    df['exposure_accumulation'] = (ce / 80) * (ts / 10)
    df['age_diabetes'] = df['age'].astype(float) * diabetes
    return df


def main():
    print('=' * 70)
    print('Kenya 公开数据复验：v2（15 维）vs v3（26 维谱系降级）特征集')
    print('=' * 70)
    df = pd.read_csv(CSV)
    df = _add_interaction_features(df)
    y = df['tb_outcome'].values.astype(int)
    X = df[MLRiskPredictor.ALL_FEATURE_NAMES].values.astype(float)
    print(f'数据: {df.shape[0]} 样本, 阳性率 {y.mean():.4f} '
          f'（{int(y.sum())} 阳性）')
    print(f'v3 输入形态: {X.shape[1]} 列 numpy（v1 全名矩阵，'
          f'训练内核按位置选 v2 15 列 + 11 网络列补零）')

    results = {}
    entries = {}
    for feature_set in ('v2', 'v3'):
        t0 = time.time()
        pred = MLRiskPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42, data_source_label='real',
            enable_calibration=False, feature_set=feature_set)
        assert ok, f'{feature_set} 训练失败'
        aurocs = {key: entry['AUROC']
                  for key, entry in pred.model_performance.items()}
        results[feature_set] = aurocs
        if feature_set == 'v3':
            entries = {key: {
                'n_features': entry.get('n_features'),
                'dead_network_columns': entry.get(
                    'dead_network_columns', []),
            } for key, entry in pred.model_performance.items()}
        print(f'[{feature_set}] {time.time() - t0:.1f}s '
              + ' '.join(f'{k}={v:.4f}' for k, v in aurocs.items()))

    dead_all = entries and all(
        e['dead_network_columns'] == list(NETWORK_FEATURE_NAMES_V3)
        for e in entries.values())
    comparison = {'model': {key: {
        'auroc_v2': round(results['v2'][key], 4),
        'auroc_v3': round(results['v3'][key], 4),
        'delta_v3_minus_v2': round(results['v3'][key] - results['v2'][key], 4),
    } for key in results['v2']}}
    worst = min(comparison['model'].values(),
                key=lambda d: d['delta_v3_minus_v2'])
    comparison['summary'] = {
        'csv': CSV,
        'dataset': 'Kenya National TB Prevalence Survey 2016 (real data)',
        'n_samples': int(df.shape[0]),
        'positive_rate': round(float(y.mean()), 6),
        'random_state': 42,
        'cv_folds': 5,
        'v3_definition': (
            'SELECTED_FEATURES_V3 = v2 15 个体列 + 11 网络特征化列'
            '（type_exposure_*5 / window_exposure_*5 / '
            'physics_lambda_calibrated）'),
        'v3_lineage': (
            'Kenya 为部署谱系：网络列源字段（图邻接边类型/时间窗分解）'
            '不存在 → 11 列全部补零死列，v3 判别力退化为 v2 口径'),
        'dead_network_columns_confirmed': bool(dead_all),
        'n_features_v3_confirmed': bool(entries) and all(
            e['n_features'] == 26 for e in entries.values()),
        'worst_delta': worst['delta_v3_minus_v2'],
        'verdict': (
            'v3 在 Kenya 部署谱系上判别力保持（谱系降级语义验证通过：'
            '死列补零 ≈ v2，最差单模型 Δ ≥ -0.01）'
            if worst['delta_v3_minus_v2'] >= -0.01 else
            'v3 在 Kenya 部署谱系上存在超噪声量级的判别力损失，'
            '谱系降级语义需复核'),
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(comparison, fh, ensure_ascii=False, indent=2)
    print(f"死列审计: {'11/11 网络列确认补零' if dead_all else '异常——见 JSON'}")
    print(f"结论: {comparison['summary']['verdict']}"
          f"（最差 Δ={worst['delta_v3_minus_v2']:+.4f}）")
    print(f'已保存: {OUT_JSON}')
    return comparison


if __name__ == '__main__':
    main()
