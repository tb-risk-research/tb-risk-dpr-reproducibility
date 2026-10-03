#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1：真实患病率口径的部署评估（2026-08-24，完成标准：NNS 量化答案）。

背景（用户指令）：场景训练数据阳性率 20.7%，Kenya 真实 0.53%——患病率
断层 40 倍，阈值与概率不可直接迁移；此前验证只报 AUROC，缺部署决策
指标（AUPRC / PPV@敏感度80% / NNS）。

协议：
- Part A（Kenya 真实患病率）：冻结 mech 主模型（宿主通路 v3 代标签，
  20.7% 患病率训练）在 Kenya 63050 真实样本上打分——不重训练（保持
  外部验证集纯净），概率用 logit 先验校正（Saerens 2002）从 20.7%
  平移到 Kenya 实际患病率，报告 AUPRC / PPV@80% / NNS / 校准前后 Brier；
- Part B（场景患病率档）：生成 0.5% / 1% / 3% 三档场景数据（mech v3
  标签，seed 2027 与训练 seed 2026 不重叠），同一冻结模型跨档评估
  （阈值迁移检验：20.7% 训练 → 低患病率部署），并与各档本地重训
  对照（判别力上限参照）；
- 完成标准输出：「在 0.5% 人群、敏感度 80% 约束下，每筛查 N 人发现
  1 例」——Kenya 真实 + 场景 0.5% 档两个口径。

产物（只追加不覆盖）：
- data/processed/prevalence_deployment_validation_20260824.json
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import _train_all_registered_models  # noqa: E402
from tb_risk.scoring.ml.evaluation import _ensemble_predict  # noqa: E402
from tb_risk.scoring.ml.deployment_metrics import screening_metrics  # noqa: E402
from run_kenya_v1v2_validation import _add_interaction_features  # noqa: E402

MODELS_DIR = os.path.join(HERE, 'training_archive', 'models')
PROC = os.path.join(HERE, 'processed')
FREEZE_JSON = os.path.join(PROC, 'primary_model_freeze_20260824.json')
OUT_JSON = os.path.join(PROC, 'prevalence_deployment_validation_20260824.json')

SCEN_SEED = 2027          # 评估档 seed（训练 2026，不重叠）
N_SCENARIOS = 5000        # 每档场景数（~25000 接触者；0.5% ≈ 125 阳性）
PREVALENCE_BINS = (0.005, 0.01, 0.03)
SENSITIVITY_TARGET = 0.80


def _load_frozen_mech():
    """从冻结决策档案加载 mech 主模型（含代际校验）。"""
    with open(FREEZE_JSON, encoding='utf-8') as fh:
        freeze = json.load(fh)
    ckpt = freeze['frozen']['mechanistic']['checkpoint']
    path = os.path.join(MODELS_DIR, ckpt)
    pred = MLRiskPredictor()
    pred.expected_label_generation = 'v3-host-pathway'
    assert pred.load_model(path), f'冻结 mech checkpoint 加载失败: {path}'
    print(f"[加载] 冻结 mech 主模型: {ckpt}"
          f"（label_generation={pred.label_generation}）")
    return pred


def _train_prevalence(csv_path):
    """某患病率档的本地重训对照（seed 42 CV5 全 5 模型，同冻结路径）。

    半区协议：训练半区训练、测试半区报告——与 evaluate_ensemble 的
    train/test 半区口径一致。全量训练+全量评估会被 RF 记忆化成
    AUROC=1.0 的完美分数（2026-08-25 实测），无参考价值。
    """
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(csv_path)
    y = df['tb_outcome'].values.astype(int)
    X = df[MLRiskPredictor.ALL_FEATURE_NAMES].values.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y)
    pred = MLRiskPredictor()
    ok = _train_all_registered_models(
        pred, X_tr, y_tr, random_state=42, data_source_label='real',
        enable_calibration=False,
        label_generation='v3-host-pathway')
    assert ok
    scores = _ensemble_predict(pred.models, X_te)
    return screening_metrics(y_te, scores, sensitivity_target=SENSITIVITY_TARGET)


def part_a_kenya(pred_mech):
    """Part A：Kenya 真实患病率（0.53%）部署指标。"""
    print('\n' + '=' * 70)
    print('Part A：Kenya 真实患病率部署评估（冻结 mech 模型，无重训练）')
    print('=' * 70)
    df = pd.read_csv(os.path.join(PROC, 'kenya_ml_training.csv'))
    df = _add_interaction_features(df)
    y = df['tb_outcome'].values.astype(int)
    X = df[MLRiskPredictor.ALL_FEATURE_NAMES].values.astype(float)
    kenya_rate = float(y.mean())

    # 训练侧患病率：冻结模型训练 CSV 的实际阳性率
    with open(os.path.join(PROC, 'ml_training_scenario_meta.json'),
              encoding='utf-8') as fh:
        train_rate = float(json.load(fh)['positive_rate'])

    scores = _ensemble_predict(pred_mech.models, X)
    res = screening_metrics(
        y, scores, source_prevalence=train_rate,
        target_prevalence=kenya_rate,
        sensitivity_target=SENSITIVITY_TARGET)
    res['training_prevalence'] = train_rate
    res['model'] = 'frozen mech (v3-host-pathway), no retraining'
    res['caliber_note'] = (
        'AUROC/AUPRC/PPV/NNS 直接在 Kenya 真实标签上计算；'
        'logit shift 仅作用于概率口径（Brier/均值），不影响排序')
    w = res['ppv_at_sensitivity']
    print(f"  Kenya 阳性率 {kenya_rate:.4%}（{res['n_pos']}/{res['n']}）")
    print(f"  AUROC={res['auroc']:.4f}  AUPRC={res['auprc']:.4f}")
    print(f"  敏感度 {SENSITIVITY_TARGET:.0%} 工作点：筛查 {w['n_flagged']} 人"
          f"（{w['flagged_rate']:.2%}），TP={w['tp']} FP={w['fp']}")
    print(f"  PPV={w['ppv']:.4f}  →  NNS = 每筛查 {res['nns']:.1f} 人发现 1 例")
    print(f"  Brier: 原始 {res['brier_raw']:.4f} → "
          f"先验校正后 {res['brier_shifted']:.4f}")
    return res


def part_b_prevalence_bins(pred_mech):
    """Part B：场景患病率档阈值迁移（0.5%/1%/3%）。"""
    print('\n' + '=' * 70)
    print(f'Part B：场景患病率档阈值迁移（seed {SCEN_SEED}，'
          f'冻结模型 vs 本地重训）')
    print('=' * 70)
    with open(os.path.join(PROC, 'ml_training_scenario_meta.json'),
              encoding='utf-8') as fh:
        train_rate = float(json.load(fh)['positive_rate'])

    results = {}
    for target in PREVALENCE_BINS:
        tag = f'{target:.1%}'
        csv_path = os.path.join(
            PROC, f'ml_training_scenario_prev{int(target * 1000)}.csv')
        meta_path = csv_path.replace('.csv', '_meta.json')
        from evaluate_ensemble import generate_scenario_ml_csv
        generate_scenario_ml_csv(
            N_SCENARIOS, SCEN_SEED, csv_path, meta_path,
            label_mode='mechanistic', prevalence_target=target)

        df = pd.read_csv(csv_path)
        y = df['tb_outcome'].values.astype(int)
        X = df[MLRiskPredictor.ALL_FEATURE_NAMES].values.astype(float)
        bin_rate = float(y.mean())

        # 冻结模型跨档评估（阈值迁移：20.7% 训练 → 低患病率部署）
        scores = _ensemble_predict(pred_mech.models, X)
        frozen = screening_metrics(
            y, scores, source_prevalence=train_rate,
            target_prevalence=bin_rate,
            sensitivity_target=SENSITIVITY_TARGET)

        # 本地重训对照（同谱系上限参照）
        t0 = time.time()
        local = _train_prevalence(csv_path)

        results[tag] = {
            'prevalence_target': target,
            'observed_rate': bin_rate,
            'n': int(len(y)),
            'n_pos': int(y.sum()),
            'frozen_model': frozen,
            'local_retrain': local,
        }
        wf, wl = frozen['ppv_at_sensitivity'], local['ppv_at_sensitivity']
        print(f"\n  [{tag} 档] 实测阳性率 {bin_rate:.4%}"
              f"（{int(y.sum())}/{len(y)}）")
        print(f"    冻结模型: AUROC={frozen['auroc']:.4f} "
              f"AUPRC={frozen['auprc']:.4f} | PPV={wf['ppv']:.4f} "
              f"NNS={frozen['nns']:.1f}")
        print(f"    本地重训: AUROC={local['auroc']:.4f} "
              f"AUPRC={local['auprc']:.4f} | PPV={wl['ppv']:.4f} "
              f"NNS={local['nns']:.1f}（{time.time() - t0:.0f}s）")
    return results


def main():
    pred_mech = _load_frozen_mech()
    kenya = part_a_kenya(pred_mech)
    bins = part_b_prevalence_bins(pred_mech)

    # 完成标准的量化答案（用户原话：在 0.5% 人群、敏感度 80% 约束下，
    # 每筛查 N 人发现 1 例）
    lo = bins['0.5%']
    answers = {
        'kenya_real_0_53pct': {
            'population': 'Kenya 2016 全国患病率调查（真实数据）',
            'prevalence': kenya['base_rate'],
            'nns': kenya['nns'],
            'ppv_at_sens80': kenya['ppv_at_sensitivity']['ppv'],
            'auroc': kenya['auroc'],
            'auprc': kenya['auprc'],
        },
        'scenario_0_5pct': {
            'population': f'场景 0.5% 患病率档（mech v3 标签，seed {SCEN_SEED}）',
            'prevalence': lo['observed_rate'],
            'nns_frozen': lo['frozen_model']['nns'],
            'nns_local_retrain': lo['local_retrain']['nns'],
            'auroc_frozen': lo['frozen_model']['auroc'],
            'auroc_local': lo['local_retrain']['auroc'],
        },
    }
    report = {
        'date': '2026-08-24',
        'purpose': ('真实患病率口径的部署决策指标（AUPRC/PPV@敏感度80%/NNS/'
                    'logit 先验校正）——患病率断层 40 倍下的阈值迁移检验'),
        'sensitivity_target': SENSITIVITY_TARGET,
        'part_a_kenya': kenya,
        'part_b_prevalence_bins': bins,
        'completion_standard_answers': answers,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print('\n' + '=' * 70)
    print('完成标准答案（敏感度 80% 约束下的 NNS）')
    print('=' * 70)
    for key, ans in answers.items():
        if 'nns' in ans:
            print(f"  [{ans['population']}] 患病率 {ans['prevalence']:.3%}: "
                  f"每筛查 {ans['nns']:.1f} 人发现 1 例（PPV="
                  f"{ans['ppv_at_sens80']:.4f}）")
        else:
            print(f"  [{ans['population']}] 患病率 {ans['prevalence']:.3%}: "
                  f"冻结模型 NNS={ans['nns_frozen']:.1f} vs 本地重训 "
                  f"NNS={ans['nns_local_retrain']:.1f}")
    print(f'\n已保存: {OUT_JSON}')


if __name__ == '__main__':
    main()
