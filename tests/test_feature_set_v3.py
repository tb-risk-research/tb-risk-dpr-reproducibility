#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四轮 P3：v3 特征集（v2 个体 15 + 网络特征化 11 = 26 维）接入
统一训练内核与预测路径。

覆盖：
1. v3 全量训练：26 列 DataFrame（网络列带信号）→ 模型记录 26 个
   真实特征名、entry 记录 feature_set='v3' / n_features=26 /
   dead_network_columns=[]；
2. 网络列信号增益：标签信号位于网络列时 v3 AUROC 显著高于 v2
   （组合消融结论"特征化拿大头"的部署层复现，合成口径
   +0.06~0.09 次可加——此处只验证方向性与幅度下限）；
3. 谱系降级（核心部署语义）：22 列 numpy（Kenya/部署 GUI 口径，
   无网络源字段）→ 网络列补零、训练成功、entry 记录 11 个 dead
   网络列、判别力 ≈ v2（死列不引入劣化）；
4. 预测端谱系感知补零（_model_input_frame）：26 名模型消费 22 维
   部署向量 → 个体列按名选、网络列全零；
5. 端到端 predict_risk：v3 训练的 predictor 消费 22 维特征向量
   正常出分；
6. 跨谱系列名一致性：NETWORK_FEATURE_NAMES_V3 与
   validation/combined_network.NET_FEATURE_NAMES 逐字一致（防漂移）；
7. 向后兼容：v1/v2 默认行为不变、不可识别输入显式失败。

文献：
- Riley RD et al. BMJ 2020;368:m443（TRIPOD：预测因子谱系边界
  应显式声明）
- 项目结论（2026-08-25 组合机制消融）：特征化 all−ind +0.087（RF，
  20 种子），协同项 −0.027 CI 不含零 → 次可加，对外口径
  "+0.06~0.09（合成）"，不写三机制相加。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import _train_all_registered_models  # noqa: E402
from tb_risk.scoring.ml.feature_audit import (  # noqa: E402
    NETWORK_FEATURE_NAMES_V3,
    SELECTED_FEATURES_V2,
    SELECTED_FEATURES_V3,
)

ALL_FEATURE_NAMES = list(MLRiskPredictor.ALL_FEATURE_NAMES)
_COL_IDX = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}
MODEL_KEYS = ['random_forest', 'gradient_boosting']


class _V3StubPredictor:
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


def _make_base_data(n=800, seed=42):
    """22 维合成数据（个体信号弱 + 网络列独立强信号的基底）。

    个体信号：age / cumulative_exposure / has_symptoms 弱 logit；
    网络信号由 _add_network_columns 注入（typed/window/λ 三机制
    冗余刻画——同一隐变量 + 独立噪声，模拟次可加结构）。
    """
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 22) * 10.0
    X[:, _COL_IDX['age']] = rng.randint(15, 85, n)
    X[:, _COL_IDX['cumulative_exposure']] = rng.uniform(0, 800, n)
    X[:, _COL_IDX['has_symptoms']] = rng.randint(0, 2, n)
    logit = (0.01 * (X[:, _COL_IDX['age']] - 45)
             + 0.001 * X[:, _COL_IDX['cumulative_exposure']]
             + 0.4 * X[:, _COL_IDX['has_symptoms']]
             - 1.2 + rng.normal(0, 0.8, n))
    y = (logit > 0).astype(int)
    if y.sum() < 20:
        y[:20] = 1
    return X, y


def _add_network_columns(X, y, seed=42, strength=2.0):
    """生成 11 网络特征列（次可加结构：共享隐变量 + 独立噪声）。

    typed（5 列）与 window（5 列）由同一隐变量 z（真实网络暴露）
    加独立噪声生成——特征间冗余（次可加的成因）；λ 列 = z 的第三
    个噪声副本。返回 (net_cols dict, y)。
    """
    rng = np.random.RandomState(seed + 1)
    n = X.shape[0]
    z = (0.5 * (y - y.mean()) / max(y.std(), 1e-9)
         + rng.normal(0, 1.0, n))
    cols = {}
    for name in NETWORK_FEATURE_NAMES_V3[:5]:
        cols[name] = np.clip(z + rng.normal(0, 1.2, n), 0, None) * strength
    for name in NETWORK_FEATURE_NAMES_V3[5:10]:
        cols[name] = np.clip(0.8 * z + rng.normal(0, 1.2, n), 0, None) \
            * strength
    cols[NETWORK_FEATURE_NAMES_V3[10]] = \
        np.clip(z + rng.normal(0, 1.5, n), 0, None) * strength
    return cols, y


def _make_v3_full_frame(X, net_cols):
    """v3 全量输入形态：22 个体列 + 11 网络列的 33 列 DataFrame
    （训练内核按名选 26 列）。"""
    import pandas as pd
    frame = pd.DataFrame(X, columns=ALL_FEATURE_NAMES)
    for c in NETWORK_FEATURE_NAMES_V3:
        frame[c] = net_cols[c]
    return frame


def _make_26_frame(X, net_cols):
    """26 列 v3 顺序矩阵（v2 15 列按位置 + 11 网络列）→ 直接喂
    sklearn 拟合的形态。"""
    import pandas as pd
    pos = [_COL_IDX[c] for c in SELECTED_FEATURES_V2]
    data = {c: X[:, _COL_IDX[c]] for c in SELECTED_FEATURES_V2}
    data.update(net_cols)
    return pd.DataFrame(data)[list(SELECTED_FEATURES_V3)]


def _make_15_frame(X):
    """22 列 numpy → v2 15 列 DataFrame（谱系降级输入形态）。"""
    import pandas as pd
    pos = [_COL_IDX[c] for c in SELECTED_FEATURES_V2]
    return pd.DataFrame(X[:, pos], columns=list(SELECTED_FEATURES_V2))


@pytest.fixture(scope='module')
def trained_v3_full():
    """v3 全量训练（网络列带信号）。"""
    X, y = _make_base_data()
    net_cols, y = _add_network_columns(X, y)
    pred = _V3StubPredictor()
    ok = _train_all_registered_models(
        pred, _make_v3_full_frame(X, net_cols), y, random_state=42,
        data_source_label='real', model_keys=MODEL_KEYS,
        enable_calibration=False, feature_set='v3')
    assert ok is True
    return pred


@pytest.fixture(scope='module')
def trained_v3_degraded():
    """v3 谱系降级训练（22 列 numpy，网络列补零）。"""
    X, y = _make_base_data()
    pred = _V3StubPredictor()
    ok = _train_all_registered_models(
        pred, X, y, random_state=42,
        data_source_label='real', model_keys=MODEL_KEYS,
        enable_calibration=False, feature_set='v3')
    assert ok is True
    return pred


@pytest.fixture(scope='module')
def trained_v2():
    X, y = _make_base_data()
    pred = _V3StubPredictor()
    ok = _train_all_registered_models(
        pred, X, y, random_state=42,
        data_source_label='real', model_keys=MODEL_KEYS,
        enable_calibration=False, feature_set='v2')
    assert ok is True
    return pred


# ============================================================================
# 1. v3 全量训练：列数 / 档案记录 / 跨谱系一致性
# ============================================================================

class TestV3Training:

    def test_models_fitted_on_26_columns(self, trained_v3_full):
        for key in MODEL_KEYS:
            names = list(getattr(
                trained_v3_full.models[key], 'feature_names_in_', []))
            assert names == list(SELECTED_FEATURES_V3), (
                f'{key} 应记录 26 个 v3 特征名，实际 {len(names)} 个')

    def test_entry_records_v3(self, trained_v3_full):
        for key in MODEL_KEYS:
            entry = trained_v3_full.model_performance[key]
            assert entry.get('feature_set') == 'v3'
            assert entry.get('n_features') == 26
            assert entry.get('dead_network_columns') == []

    def test_network_names_match_validation_layer(self):
        """部署副本与 validation 单一真值源逐字一致（防漂移）。"""
        from tb_risk.validation.combined_network import NET_FEATURE_NAMES
        assert list(NETWORK_FEATURE_NAMES_V3) == list(NET_FEATURE_NAMES), (
            '部署谱系网络列名与验证层漂移：'
            f'{set(NETWORK_FEATURE_NAMES_V3) ^ set(NET_FEATURE_NAMES)}')

    def test_v3_structure(self):
        assert len(SELECTED_FEATURES_V3) == 26
        assert SELECTED_FEATURES_V3[:15] == list(SELECTED_FEATURES_V2)
        assert SELECTED_FEATURES_V3[15:] == list(NETWORK_FEATURE_NAMES_V3)


# ============================================================================
# 2. 网络列信号增益（特征化路线的方向性验证）
# ============================================================================

class TestV3NetworkGain:

    def test_v3_beats_v2_when_signal_in_network_cols(
            self, trained_v3_full, trained_v2):
        """标签信号位于网络列时 v3 > v2（幅度下限 0.05——组合消融
        合成口径 +0.06~0.09 的部署层方向性复现）。"""
        for key in MODEL_KEYS:
            a2 = trained_v2.model_performance[key]['AUROC']
            a3 = trained_v3_full.model_performance[key]['AUROC']
            assert a3 >= a2 + 0.05, (
                f'{key}: v3 AUROC {a3:.4f} 未显著超过 v2 {a2:.4f}'
                '（+0.05 下限）——网络列信号未兑现')


# ============================================================================
# 3. 谱系降级：22 列 numpy（部署谱系无网络源字段）
# ============================================================================

class TestV3LineageDegradation:

    def test_degraded_training_succeeds(self, trained_v3_degraded):
        for key in MODEL_KEYS:
            entry = trained_v3_degraded.model_performance[key]
            assert entry.get('feature_set') == 'v3'
            assert entry.get('n_features') == 26
            # 11 个网络列全部记为 dead（Kenya/部署谱系）
            assert entry.get('dead_network_columns') == \
                list(NETWORK_FEATURE_NAMES_V3)

    def test_degraded_models_have_26_names(self, trained_v3_degraded):
        for key in MODEL_KEYS:
            names = list(getattr(
                trained_v3_degraded.models[key], 'feature_names_in_', []))
            assert names == list(SELECTED_FEATURES_V3)

    def test_degraded_matches_v2_performance(
            self, trained_v3_degraded, trained_v2):
        """死列不引入劣化：补零 v3 ≈ v2（RF/LGBM 忽略零方差列）。"""
        for key in MODEL_KEYS:
            a2 = trained_v2.model_performance[key]['AUROC']
            a3 = trained_v3_degraded.model_performance[key]['AUROC']
            assert abs(a3 - a2) <= 0.01, (
                f'{key}: v3 降级 AUROC {a3:.4f} 与 v2 {a2:.4f} 偏离'
                '超噪声（死列引入劣化）')

    def test_dataframe_lineage_degradation(self):
        """v2 15 列 DataFrame 输入（HomeACF 谱系形态）同样降级成功。"""
        X, y = _make_base_data(n=300)
        pred = _V3StubPredictor()
        ok = _train_all_registered_models(
            pred, _make_15_frame(X), y, random_state=42,
            model_keys=['random_forest'],
            enable_calibration=False, feature_set='v3')
        assert ok is True
        entry = pred.model_performance['random_forest']
        assert entry.get('dead_network_columns') == \
            list(NETWORK_FEATURE_NAMES_V3)


# ============================================================================
# 4. 预测端谱系感知补零（_model_input_frame）
# ============================================================================

class TestModelInputFrameV3:

    def _fit_v3_model(self):
        from sklearn.ensemble import RandomForestClassifier
        X, y = _make_base_data(n=300)
        net_cols, y = _add_network_columns(X, y)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(_make_26_frame(X, net_cols), y)
        return model, X

    def test_v3_model_consumes_22_dim_vector(self):
        from tb_risk.scoring.ml.evaluation import _model_input_frame
        model, X = self._fit_v3_model()
        out = _model_input_frame(model, X[:5], ALL_FEATURE_NAMES)
        assert list(out.columns) == list(model.feature_names_in_)
        # 网络列全零（谱系降级）
        for col in NETWORK_FEATURE_NAMES_V3:
            assert np.allclose(out[col].values, 0.0), (
                f'网络列 {col} 应补零')
        # 个体列按名选的数值正确性
        age_pos = list(out.columns).index('age')
        assert np.allclose(
            out.values[:, age_pos], X[:5, _COL_IDX['age']])

    def test_unrelated_model_names_passthrough(self):
        """完全不相干的模型名（无交集）不走补零路径——原样返回。"""
        from sklearn.ensemble import RandomForestClassifier
        from tb_risk.scoring.ml.evaluation import _model_input_frame
        rng = np.random.RandomState(0)
        X = rng.rand(60, 8)
        y = rng.randint(0, 2, 60)
        model = RandomForestClassifier(n_estimators=5, random_state=0)
        model.fit(X, y)  # feature_names_in_ = Column_0..7
        out = _model_input_frame(model, rng.rand(5, 22), ALL_FEATURE_NAMES)
        assert out.shape == (5, 22)  # 原样返回，不误配


# ============================================================================
# 5. 端到端：v3 训练的 predictor 仍可 predict_risk（22 维输入）
# ============================================================================

class TestPredictRiskWithV3:

    def test_predict_risk_works_with_v3_models(self):
        X, y = _make_base_data(n=500)
        net_cols, y = _add_network_columns(X, y)
        predictor = MLRiskPredictor()
        ok = _train_all_registered_models(
            predictor, _make_v3_full_frame(X, net_cols), y, random_state=42,
            data_source_label='real', model_keys=MODEL_KEYS,
            enable_calibration=False, feature_set='v3')
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
        assert result['ensemble']['risk_probability'] > 0.0
        for key in MODEL_KEYS:
            assert result[key]['risk_probability'] > 0.0, (
                f'{key} 出分失败——v3 谱系感知补零路径不通')


# ============================================================================
# 6. 向后兼容与显式失败
# ============================================================================

class TestBackwardCompatibility:

    def test_v1_default_unchanged(self):
        X, y = _make_base_data(n=300)
        pred = _V3StubPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42, model_keys=['random_forest'],
            enable_calibration=False)
        assert ok is True
        entry = pred.model_performance['random_forest']
        assert entry.get('feature_set') == 'v1'
        assert entry.get('n_features') == 22
        assert entry.get('dead_network_columns') == []

    def test_v2_unchanged(self, trained_v2):
        for key in MODEL_KEYS:
            entry = trained_v2.model_performance[key]
            assert entry.get('feature_set') == 'v2'
            assert entry.get('n_features') == 15

    def test_unrecognizable_input_fails(self):
        rng = np.random.RandomState(0)
        X = rng.rand(200, 33)  # 既非 22 也非 26
        y = rng.randint(0, 2, 200)
        pred = _V3StubPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42, model_keys=['random_forest'],
            enable_calibration=False, feature_set='v3')
        assert ok is False
        assert pred.models == {}
