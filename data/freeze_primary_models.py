#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主模型冻结（2026-08-24，遗留事项：五模型选主冻结）。

背景：训练管线产出 5 个基模型（RF/GBDT/XGB/LGBM/CatBoost）并列记录，
从未执行"冻结主模型"决策；且 2026-08-24 宿主通路修复（evaluate_ensemble
v3 标签：易感性项加入文献校准的年龄 U 型/症状/共病/BCG 衰减）改变了
mechanistic 标签分布——新旧标签下的 AUROC 不可直接比较，best.json 的
"仅更高时覆盖"机制因此不会保存新代 checkpoint（旧标签 mech 0.8395 是
不同任务的分数）。

本脚本执行显式冻结：
1. 用宿主通路修复后的场景 CSV（mech/clinical，seed 2026 生成、
   train_from_real_data seed 42 训练——与归档 run 20260824_225829/
   clinical run 完全同参数，确定性复现）重训 5 个基模型；
2. 逐标签族选主模型（per_model CV AUROC 最高者）；
3. 显式保存 checkpoint（best_ml_<stamp>_ns12509_cv5.joblib）——
   不依赖 best.json 的跨代比较；
4. 冻结决策归档 data/processed/primary_model_freeze_20260824.json
   （含选主证据链与被替代的旧 checkpoint）。

注意：本脚本不做 TrainingLogger 归档（正式训练记录已由
train_with_public_data.py 落档）；本脚本只做模型选择 + 冻结保存。
"""

import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(HERE, 'training_archive', 'models')
PROC = os.path.join(HERE, 'processed')
OUT_JSON = os.path.join(PROC, 'primary_model_freeze_20260824.json')

# 标签族 → (CSV, 被替代的旧 checkpoint（2026-08-22 冻结，旧标签 v2 代）)
LABELS = {
    'mechanistic': {
        'csv': os.path.join(PROC, 'ml_training_scenario_aligned.csv'),
        'superseded': 'best_ml_20260822_185119_ns12476_cv5.joblib',
        'label_generation': 'v3-host-pathway',
    },
    'clinical': {
        'csv': os.path.join(PROC, 'ml_training_scenario_clinical.csv'),
        'superseded': 'best_ml_20260822_185411_ns12476_cv5.joblib',
        'label_generation': 'clinical-rule-v1',
    },
}


def _freeze_one(label_mode, spec):
    from tb_risk.scoring.predictor import MLRiskPredictor

    csv_path = spec['csv']
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    predictor = MLRiskPredictor()
    result = predictor.train_from_real_data(
        csv_path, target_column='tb_outcome',
        label_generation=spec['label_generation'])
    if not result.get('success'):
        raise RuntimeError(f'{label_mode} 训练失败: {result.get("diagnostics")}')

    per_model = {
        key: {'name': info['name'], 'auroc': round(info['AUROC'], 4),
              'auprc': round(info['AUPRC'], 4)}
        for key, info in predictor.model_performance.items()
    }
    primary_key = max(per_model, key=lambda k: per_model[k]['auroc'])

    ckpt_name = (f'best_ml_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
                 f'_ns{result["n_samples"]}_cv5.joblib')
    ckpt_path = os.path.join(MODELS_DIR, ckpt_name)
    if not predictor.save_model(ckpt_path):
        raise RuntimeError(f'{label_mode} checkpoint 保存失败: {ckpt_path}')

    print(f"[{label_mode}] 主模型 = {per_model[primary_key]['name']}"
          f"（{primary_key}, AUROC={per_model[primary_key]['auroc']}）")
    for key, info in per_model.items():
        marker = ' ←主' if key == primary_key else ''
        print(f"    {info['name']:<8} AUROC={info['auroc']:.4f}"
              f" AUPRC={info['auprc']:.4f}{marker}")
    print(f"    checkpoint: {ckpt_name}")

    return {
        'primary_model_key': primary_key,
        'primary_model_name': per_model[primary_key]['name'],
        'primary_auroc': per_model[primary_key]['auroc'],
        'per_model': per_model,
        'checkpoint': ckpt_name,
        'label_generation': spec['label_generation'],
        'n_samples': result['n_samples'],
        'positive_rate': round(result['positive_rate'], 4),
        'superseded_checkpoint': spec['superseded'],
        'random_state': 42,
        'cv_folds': 5,
    }


def main():
    print('=' * 70)
    print('主模型冻结（宿主通路修复 v3 标签代）')
    print('=' * 70)

    frozen = {mode: _freeze_one(mode, spec)
              for mode, spec in LABELS.items()}

    decision = {
        'frozen_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'rationale': (
            '宿主通路修复（2026-08-24，evaluate_ensemble v3 标签：'
            '易感性加入年龄U型/症状/共病RR/BCG衰减）改变了 mechanistic '
            '标签分布，新旧代 AUROC 跨任务不可比；best.json 的跨代比较'
            '不适用于代际切换，故显式冻结本代五模型中 CV AUROC 最优者'
            '作为部署/集成评估的主模型'),
        'label_mechanisms': {
            'mechanistic': 'v3（暴露+宿主双通路）',
            'clinical': '临床规则（未变）',
        },
        'frozen': frozen,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(decision, fh, ensure_ascii=False, indent=2)
    print(f'冻结决策已归档: {OUT_JSON}')


if __name__ == '__main__':
    main()
