#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题4（训练流程接入）：统一训练内核的不平衡记录 / 真实数据校准 / 决策参考

覆盖 _train_all_registered_models 的三个新增行为：
1. 每模型记录 imbalance_handling 元信息（训练侧类不平衡处理的单一档案）；
2. 训练完成自动接入真实数据校准（calibrate_models_on_data，
   Platt/isotonic 自动选择，Brier/AUROC 前后对照）；
3. 基于校准评估半区概率构建 decision_reference（排序+截断点+双口径 PPV）。

文献：
- He H, Garcia EA. Learning from Imbalanced Data. IEEE TKDE 2009
  （类加权 vs 重采样）
- Platt J. (1999)；Niculescu-Mizil & Caruana (2005)：校准方法选择
- Bayes PPV 换算：Fletcher & Fletcher, Clinical Epidemiology
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.scoring.ml.training import _train_all_registered_models  # noqa: E402
from tb_risk import constants as C  # noqa: E402


class _PipelineStubPredictor:
    """满足统一训练内核协议的最小 predictor 桩。"""

    ALL_FEATURE_NAMES = [f'f{i}' for i in range(6)]

    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.is_trained = False
        self.calibrated_models = None
        self.calibration_eval = None
        self._is_calibrated = False
        self._calibration_method = None
        self._stop_training = False


def _make_imbalanced_data(n=600, pos_rate=0.15, seed=42):
    """合成可分数据（阳性率 ~16% ≈ 96 阳性 < 200 → auto 选 sigmoid 校准；
    2026-09-05 地基审计：阈值由 500 对齐记忆约束 200，截距 -2.0→-3.7
    使实际阳性数与注释声明一致——原数据 377 阳性落在旧灰区）。"""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 6)
    logit = 3.0 * X[:, 0] + 2.0 * X[:, 1] - 3.7 + rng.normal(0, 0.5, n)
    y = (logit > 0).astype(int)
    if y.sum() < 20:
        y[:20] = 1
    return X, y


MODEL_KEYS = ['random_forest', 'gradient_boosting']


@pytest.fixture(scope='module')
def trained_stub():
    """模块级共享：训练一次（RF + GB，小数据集，避免测试过慢）。"""
    X, y = _make_imbalanced_data()
    pred = _PipelineStubPredictor()
    ok = _train_all_registered_models(
        pred, X, y, random_state=42,
        data_source_label='real', model_keys=MODEL_KEYS)
    assert ok is True
    return pred


# ============================================================================
# 1. imbalance_handling 元信息
# ============================================================================

class TestImbalanceHandling:

    def test_every_model_records_imbalance_handling(self, trained_stub):
        for key in MODEL_KEYS:
            entry = trained_stub.model_performance[key]
            info = entry.get('imbalance_handling')
            assert isinstance(info, dict), f'{key} 缺 imbalance_handling'
            assert info.get('method'), f'{key} 的 method 为空'
            assert 'scale_pos_weight' in info

    def test_rf_uses_class_weight_balanced(self, trained_stub):
        info = trained_stub.model_performance['random_forest']['imbalance_handling']
        assert 'class_weight' in info['method']
        assert 'balanced' in info['method']

    def test_gb_uses_sample_weight(self, trained_stub):
        info = trained_stub.model_performance['gradient_boosting']['imbalance_handling']
        assert 'sample_weight' in info['method']

    def test_scale_pos_weight_matches_data(self, trained_stub):
        X, y = _make_imbalanced_data()
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        expected = max(n_neg / max(n_pos, 1.0), 1.0)
        for key in MODEL_KEYS:
            info = trained_stub.model_performance[key]['imbalance_handling']
            assert info['scale_pos_weight'] == pytest.approx(expected)


# ============================================================================
# 2. 训练流程自动接入真实数据校准
# ============================================================================

class TestTrainingPipelineCalibration:

    def test_pipeline_calibrates_on_real_data(self, trained_stub):
        assert trained_stub._is_calibrated is True
        assert trained_stub.calibrated_models is not None
        assert set(trained_stub.calibrated_models.keys()) == set(MODEL_KEYS)

    def test_calibration_metrics_recorded(self, trained_stub):
        for key in MODEL_KEYS:
            cal = trained_stub.model_performance[key]['calibration']
            for field in ('method', 'n_positive', 'brier_before', 'brier_after',
                          'auroc_before', 'auroc_after'):
                assert field in cal, f'{key} 缺 {field}'

    def test_auto_selects_sigmoid_for_few_positives(self, trained_stub):
        # 600 样本 × ~16% ≈ 96 阳性 < 200 → Platt(sigmoid)
        cal = trained_stub.model_performance['random_forest']['calibration']
        assert cal['method'] == 'sigmoid'
        assert cal['n_positive'] < 200


# ============================================================================
# 3. decision_reference（排序+截断点+双口径 PPV）
# ============================================================================

class TestTrainingPipelineDecisionReference:

    def test_decision_reference_recorded(self, trained_stub):
        for key in MODEL_KEYS:
            ref = trained_stub.model_performance[key].get('decision_reference')
            assert isinstance(ref, dict), f'{key} 缺 decision_reference'
            assert ref['decision_form'] == 'ranking_cutoff'
            # 三口径（2026-09-17 缺陷1）：固定双口径 + cohort_local
            # （训练样本阳性率——与训练队列同人群的站点读表口径）
            assert set(ref['ppv'].keys()) == {
                'close_contact', 'general_population', 'cohort_local'}
            assert '不可混用' in ref['note']

    def test_ppv_uses_prevalence_calibers(self, trained_stub):
        ref = trained_stub.model_performance['random_forest']['decision_reference']
        for caliber_key, caliber in C.PREVALENCE_CALIBERS.items():
            block = ref['ppv'][caliber_key]
            assert block['prevalence'] == pytest.approx(caliber['value'])
            assert 0.0 <= block['ppv'] <= 1.0

    def test_operating_point_present(self, trained_stub):
        ref = trained_stub.model_performance['random_forest']['decision_reference']
        op = ref['operating_point']
        for field in ('sensitivity', 'specificity', 'threshold', 'n_flagged'):
            assert field in op


# ============================================================================
# 4. 校准可关闭（失败/性能场景不阻断训练）
# ============================================================================

class TestCalibrationToggle:

    def test_enable_calibration_false_skips_calibration(self):
        X, y = _make_imbalanced_data()
        pred = _PipelineStubPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42,
            data_source_label='real', model_keys=MODEL_KEYS,
            enable_calibration=False)
        assert ok is True
        assert pred._is_calibrated is False
        # 不平衡元信息与决策参考表仍应记录
        info = pred.model_performance['random_forest']['imbalance_handling']
        assert info['method']
