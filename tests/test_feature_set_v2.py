#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题3：v2 精简特征集（22→15）接入统一训练内核与预测路径。

覆盖：
1. feature_set='v2' 训练：X 按列名筛选到 SELECTED_FEATURES_V2，模型
   feature_names_in_ 记录 15 个真实特征名，训练档案记录 feature_set /
   n_features；
2. 性能可比性：信号位于保留特征上时 v2 AUROC ≥ v1 - 0.05（精简标准是
   "每个特征都有独立临床含义且无法互相推导"，不以牺牲判别力为代价）；
3. 预测端列子集对齐（_model_input_frame）：15 维模型直接消费 22 维特征
   向量（按名选列），v1 全名模型与旧 numpy 档行为不变；
4. 默认 feature_set='v1' 向后兼容（不传参数行为与旧调用完全一致）。

文献：
- Riley RD et al. BMJ 2020;368:m443（TRIPOD：预测因子应有独立临床含义）
- 项目教训（2026-08-24 特征审计）：exposure_accumulation/age_bcg_decay
  可由其他列精确重构（R²≈1），属信息重复计数。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import _train_all_registered_models  # noqa: E402
from tb_risk.scoring.ml.feature_audit import SELECTED_FEATURES_V2  # noqa: E402

ALL_FEATURE_NAMES = list(MLRiskPredictor.ALL_FEATURE_NAMES)
_COL_IDX = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}
MODEL_KEYS = ['random_forest', 'gradient_boosting']


class _V2StubPredictor:
    """满足统一训练内核协议的最小 predictor 桩（22 维真实特征名）。"""

    ALL_FEATURE_NAMES = ALL_FEATURE_NAMES

    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.is_trained = False
        self.calibrated_models = None
        self.calibration_eval = None
        self._is_calibrated = False
        self._calibration_method = None
        self._stop_training = False


def _make_data(n=800, seed=42):
    """22 维合成数据，信号仅位于 v2 保留特征（age / cumulative_exposure /
    has_symptoms）——精简不应损失判别力。"""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 22) * 10.0
    X[:, _COL_IDX['age']] = rng.randint(15, 85, n)
    X[:, _COL_IDX['cumulative_exposure']] = rng.uniform(0, 800, n)
    X[:, _COL_IDX['has_symptoms']] = rng.randint(0, 2, n)
    logit = (0.035 * (X[:, _COL_IDX['age']] - 45)
             + 0.004 * X[:, _COL_IDX['cumulative_exposure']]
             + 0.9 * X[:, _COL_IDX['has_symptoms']]
             - 1.2 + rng.normal(0, 0.5, n))
    y = (logit > 0).astype(int)
    if y.sum() < 20:
        y[:20] = 1
    return X, y


@pytest.fixture(scope='module')
def trained_v1():
    X, y = _make_data()
    pred = _V2StubPredictor()
    ok = _train_all_registered_models(
        pred, X, y, random_state=42,
        data_source_label='real', model_keys=MODEL_KEYS,
        enable_calibration=False)
    assert ok is True
    return pred


@pytest.fixture(scope='module')
def trained_v2():
    X, y = _make_data()
    pred = _V2StubPredictor()
    ok = _train_all_registered_models(
        pred, X, y, random_state=42,
        data_source_label='real', model_keys=MODEL_KEYS,
        enable_calibration=False, feature_set='v2')
    assert ok is True
    return pred


# ============================================================================
# 1. v2 训练：列筛选 + 档案记录
# ============================================================================

class TestV2Training:

    def test_models_fitted_on_15_columns(self, trained_v2):
        for key in MODEL_KEYS:
            names = list(getattr(trained_v2.models[key], 'feature_names_in_', []))
            assert names == list(SELECTED_FEATURES_V2), (
                f'{key} 应记录 15 个 v2 特征名，实际 {len(names)} 个')

    def test_entry_records_feature_set(self, trained_v2):
        for key in MODEL_KEYS:
            entry = trained_v2.model_performance[key]
            assert entry.get('feature_set') == 'v2'
            assert entry.get('n_features') == 15

    def test_v1_default_records_22(self, trained_v1):
        for key in MODEL_KEYS:
            entry = trained_v1.model_performance[key]
            assert entry.get('feature_set') == 'v1'
            assert entry.get('n_features') == 22

    def test_v2_performance_comparable_to_v1(self, trained_v1, trained_v2):
        for key in MODEL_KEYS:
            a1 = trained_v1.model_performance[key]['AUROC']
            a2 = trained_v2.model_performance[key]['AUROC']
            assert a2 >= a1 - 0.05, (
                f'{key}: v2 AUROC {a2:.4f} 比 v1 {a1:.4f} 下降超过 0.05')

    def test_unknown_feature_set_rejected(self):
        X, y = _make_data(n=200)
        pred = _V2StubPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42, model_keys=MODEL_KEYS,
            enable_calibration=False, feature_set='v9')
        assert ok is False
        assert pred.models == {}


# ============================================================================
# 2. 预测端列子集对齐（_model_input_frame）
# ============================================================================

class TestModelInputFrame:

    def _fit_v2_model(self):
        from sklearn.ensemble import RandomForestClassifier
        X, y = _make_data(n=300)
        pos = [_COL_IDX[c] for c in SELECTED_FEATURES_V2]
        frame = __import__('pandas').DataFrame(
            X[:, pos], columns=list(SELECTED_FEATURES_V2))
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(frame, y)
        return model, X

    def test_subset_selection_for_v2_model(self):
        from tb_risk.scoring.ml.evaluation import _model_input_frame
        model, X = self._fit_v2_model()
        out = _model_input_frame(model, X[:5], ALL_FEATURE_NAMES)
        assert list(out.columns) == list(model.feature_names_in_)
        # 按名选列的数值正确性：age 列取自 22 维矩阵的 age 位置
        age_pos = list(out.columns).index('age')
        assert np.allclose(
            out.values[:, age_pos], X[:5, _COL_IDX['age']])

    def test_full_name_model_unchanged(self):
        from sklearn.ensemble import RandomForestClassifier
        from tb_risk.scoring.ml.evaluation import _model_input_frame
        import pandas as pd
        X, y = _make_data(n=300)
        frame = pd.DataFrame(X, columns=ALL_FEATURE_NAMES)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(frame, y)
        out = _model_input_frame(model, X[:5], ALL_FEATURE_NAMES)
        assert list(out.columns) == ALL_FEATURE_NAMES
        assert np.allclose(out.values, X[:5])

    def test_numpy_checkpoint_passthrough(self):
        from sklearn.ensemble import RandomForestClassifier
        from tb_risk.scoring.ml.evaluation import _model_input_frame
        X, y = _make_data(n=300)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)  # numpy 拟合：无 feature_names_in_
        out = _model_input_frame(model, X[:5], ALL_FEATURE_NAMES)
        assert isinstance(out, np.ndarray)


# ============================================================================
# 3. 端到端：v2 训练的 predictor 仍可 predict_risk（22 维输入）
# ============================================================================

class TestPredictRiskWithV2:

    def test_predict_risk_works_with_v2_models(self):
        X, y = _make_data(n=500)
        predictor = MLRiskPredictor()
        ok = _train_all_registered_models(
            predictor, X, y, random_state=42,
            data_source_label='real', model_keys=MODEL_KEYS,
            enable_calibration=False, feature_set='v2')
        assert ok is True

        contact = {
            'age': 68, 'cumulative_exposure': 720, 'has_symptoms': 1,
            'bcg_vaccine': 1, 'past_illness': 1, 'is_high_risk': 1,
            'single_duration': 120, 'freq_density': 12, 'time_span': 9,
        }
        from tb_risk.scoring.ml.evaluation import predict_risk
        result = predict_risk(predictor, contact, contact_type='family')
        assert result is not None
        assert 'ensemble' in result
        # 高危接触者：集成概率应为正（列对齐失败会走零值回退）
        assert result['ensemble']['risk_probability'] > 0.0
        for key in MODEL_KEYS:
            assert result[key]['risk_probability'] > 0.0, (
                f'{key} 走了零值回退——v2 列子集对齐失败')
