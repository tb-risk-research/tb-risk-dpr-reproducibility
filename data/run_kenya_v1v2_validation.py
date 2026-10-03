#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v1/v2 特征集在公开数据（Kenya）上的复验（2026-08-24，遗留事项收尾）。

背景：v2（15 维精简集）vs v1（22 维）此前只在场景 CSV（机制标签，
seed 2026）上对比过（feature_set_v1_v2_comparison_20260824.json，最差
Δ=-0.0064）；用户要求在公开数据上复验。Kenya 2016 全国 TB 患病率
调查（63050 样本，细菌学确诊标签，阳性率 0.53%）是项目唯一公开的
真实个体级 ML 训练集（PLOS ONE doi:10.1371/journal.pone.0209098）。

协议（与场景 CSV 对比严格一致）：
- 同一 CSV、同种子 42、同 CV5，v1/v2 各训练全部 5 个基模型；
- 判别力保持标准：最差单模型 Δ ≥ -0.01（项目"单种子增益 <0.01
  是噪声"惯例）；
- 附 Kenya 谱系特有的特征变异审计：symptom_delay/cough_contact
  （源字段 delay_days/cough_frequency 不存在）与
  dm_tb_synergy/age_diabetes（源字段 past_illness_type 不存在，
  无糖尿病分型）在本谱系恒 0——v2 移除 cough_contact 在此是
  纯增益（少一个死列），其余死列两代特征集共有。

产物（只追加不覆盖）：
- data/processed/feature_set_v1_v2_comparison_kenya_20260824.json
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
from tb_risk.scoring.ml.feature_audit import SELECTED_FEATURES_V2  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'processed')
CSV = os.path.join(OUT_DIR, 'kenya_ml_training.csv')
OUT_JSON = os.path.join(OUT_DIR, 'feature_set_v1_v2_comparison_kenya_20260824.json')


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


def _dead_feature_audit(df):
    """两代特征集在 Kenya 谱系上的死列（唯一值数 ≤ 1）清单。"""
    audit = {}
    for tag, feats in (('v1', list(MLRiskPredictor.ALL_FEATURE_NAMES)),
                       ('v2', list(SELECTED_FEATURES_V2))):
        audit[tag] = {
            'dead': [f for f in feats if df[f].nunique(dropna=False) <= 1],
            'n_features': len(feats),
        }
    return audit


def main():
    print('=' * 70)
    print('Kenya 公开数据复验：v1（22 维）vs v2（15 维）特征集对比')
    print('=' * 70)
    df = pd.read_csv(CSV)
    df = _add_interaction_features(df)
    y = df['tb_outcome'].values.astype(int)
    X = df[MLRiskPredictor.ALL_FEATURE_NAMES].values.astype(float)
    print(f'数据: {df.shape[0]} 样本, 阳性率 {y.mean():.4f} '
          f'（{int(y.sum())} 阳性）')

    dead = _dead_feature_audit(df)
    for tag, info in dead.items():
        print(f"[{tag}] 死列 {len(info['dead'])} 个: {info['dead']}")

    results = {}
    for feature_set in ('v1', 'v2'):
        t0 = time.time()
        pred = MLRiskPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42, data_source_label='real',
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
        'dataset': 'Kenya National TB Prevalence Survey 2016 (real data)',
        'n_samples': int(df.shape[0]),
        'positive_rate': round(float(y.mean()), 6),
        'random_state': 42,
        'cv_folds': 5,
        'worst_delta': worst['delta_v2_minus_v1'],
        'dead_features_kenya_spectrum': dead,
        'verdict': (
            'v2 在 Kenya 真实数据上判别力保持（最差单模型 Δ ≥ -0.01）'
            if worst['delta_v2_minus_v1'] >= -0.01 else
            'v2 在 Kenya 真实数据上存在超噪声量级的判别力损失，需复核'),
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(comparison, fh, ensure_ascii=False, indent=2)
    print(f"结论: {comparison['summary']['verdict']}"
          f"（最差 Δ={worst['delta_v2_minus_v1']:+.4f}）")
    print(f'已保存: {OUT_JSON}')
    return comparison


if __name__ == '__main__':
    main()
