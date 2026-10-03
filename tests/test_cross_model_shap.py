#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题3（解释口径统一）：compute_cross_model_shap 跨模型 SHAP 共识。

用户问题：各模型 feature_importances_ 的 gain/split 口径不可跨模型比较，
造成"随机森林说暴露最重要、XGBoost 说症状最重要"的表象分歧。

解决方案（Lundberg & Lee, NeurIPS 2017：SHAP 是唯一同时满足局部精确性
与一致性的加性归因方法）：全部树模型在同一特征矩阵上计算 mean|SHAP|，
按模型内归一化为份额（消除输出量纲差异），聚合出跨模型共识排序；
残余分歧以 share_std / 排名区间显式量化——若分歧在 SHAP 口径下仍在，
是真实模型分歧而非口径假象。

覆盖：
1. compute_cross_model_shap：全部模型参与、份额归一化、共识排序、
   信号特征进 Top、排名信息齐全、确定性、max_samples 采样；
2. v2 特征集（15 维）模型与 v1（22 维）模型在同一入口下支持；
3. format_cross_model_shap_table：Markdown 共识表（含分歧与排名区间）；
4. 训练报告 §3 接入跨模型共识段。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import _train_all_registered_models  # noqa: E402
from tb_risk.scoring.ml.feature_audit import SELECTED_FEATURES_V2  # noqa: E402

try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SHAP_AVAILABLE, reason='shap 未安装，跨模型 SHAP 测试跳过')

ALL_FEATURE_NAMES = list(MLRiskPredictor.ALL_FEATURE_NAMES)
_COL_IDX = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}
MODEL_KEYS = ['random_forest', 'gradient_boosting']
SIGNAL_FEATURES = ['age', 'cumulative_exposure', 'has_symptoms']


class _ShapStubPredictor:
    """满足 compute_cross_model_shap 协议的最小 predictor 桩。"""

    ALL_FEATURE_NAMES = ALL_FEATURE_NAMES

    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.is_trained = False
        self._stop_training = False


def _make_data(n=600, seed=42):
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
def trained_predictor():
    """RF + GB 训练于同一 22 维数据（信号在 age/累积暴露/症状上）。"""
    X, y = _make_data()
    pred = _ShapStubPredictor()
    ok = _train_all_registered_models(
        pred, X, y, random_state=42,
        data_source_label='real', model_keys=MODEL_KEYS,
        enable_calibration=False)
    assert ok is True
    return pred


# ============================================================================
# 1. compute_cross_model_shap 核心行为
# ============================================================================

class TestComputeCrossModelShap:

    def test_all_models_and_features_covered(self, trained_predictor):
        from tb_risk.scoring.ml.shap_analysis import compute_cross_model_shap
        X, _ = _make_data(n=200)
        result = compute_cross_model_shap(trained_predictor, X)
        assert result is not None
        assert result['n_models'] == 2
        assert set(result['per_model'].keys()) == set(MODEL_KEYS)
        assert set(result['feature_names']) == set(ALL_FEATURE_NAMES)

    def test_shares_normalized_per_model(self, trained_predictor):
        from tb_risk.scoring.ml.shap_analysis import compute_cross_model_shap
        X, _ = _make_data(n=200)
        result = compute_cross_model_shap(trained_predictor, X)
        for key, info in result['per_model'].items():
            total = sum(info['shares'].values())
            assert abs(total - 1.0) < 1e-6, f'{key} 份额未归一化: {total}'

    def test_signal_features_rank_top5(self, trained_predictor):
        from tb_risk.scoring.ml.shap_analysis import compute_cross_model_shap
        X, _ = _make_data(n=300)
        result = compute_cross_model_shap(trained_predictor, X)
        top5 = {entry['feature'] for entry in result['consensus'][:5]}
        for feat in SIGNAL_FEATURES:
            assert feat in top5, f'信号特征 {feat} 未进共识 Top5'

    def test_consensus_sorted_desc(self, trained_predictor):
        from tb_risk.scoring.ml.shap_analysis import compute_cross_model_shap
        X, _ = _make_data(n=200)
        result = compute_cross_model_shap(trained_predictor, X)
        shares = [e['consensus_share'] for e in result['consensus']]
        assert shares == sorted(shares, reverse=True)

    def test_rank_info_complete(self, trained_predictor):
        from tb_risk.scoring.ml.shap_analysis import compute_cross_model_shap
        X, _ = _make_data(n=200)
        result = compute_cross_model_shap(trained_predictor, X)
        for entry in result['consensus']:
            assert set(entry['ranks'].keys()) == set(MODEL_KEYS)
            assert entry['rank_min'] <= entry['rank_max']
            assert entry['rank_max'] - entry['rank_min'] >= 0
        # 排名区间量化模型间分歧：至少存在一个特征两模型排名差 >= 1
        spreads = [e['rank_max'] - e['rank_min'] for e in result['consensus']]
        assert max(spreads) >= 1

    def test_deterministic(self, trained_predictor):
        from tb_risk.scoring.ml.shap_analysis import compute_cross_model_shap
        X, _ = _make_data(n=200)
        r1 = compute_cross_model_shap(trained_predictor, X)
        r2 = compute_cross_model_shap(trained_predictor, X)
        assert [e['feature'] for e in r1['consensus']] == \
               [e['feature'] for e in r2['consensus']]
        assert np.allclose(
            [e['consensus_share'] for e in r1['consensus']],
            [e['consensus_share'] for e in r2['consensus']])

    def test_max_samples_respected(self, trained_predictor):
        from tb_risk.scoring.ml.shap_analysis import compute_cross_model_shap
        X, _ = _make_data(n=600)
        result = compute_cross_model_shap(trained_predictor, X, max_samples=50)
        assert result['n_samples'] <= 50

    def test_single_model_returns_none(self):
        """共识需要 >= 2 个模型——单模型无可比性。"""
        from tb_risk.scoring.ml.shap_analysis import compute_cross_model_shap
        from sklearn.ensemble import RandomForestClassifier
        X, y = _make_data(n=200)
        pred = _ShapStubPredictor()
        pred.models['random_forest'] = RandomForestClassifier(
            n_estimators=10, random_state=42).fit(X, y)
        assert compute_cross_model_shap(pred, X[:50]) is None


# ============================================================================
# 2. v2 特征集（15 维）模型支持
# ============================================================================

class TestV2ModelsSupported:

    def test_v2_models_consensus_over_15_features(self):
        from tb_risk.scoring.ml.shap_analysis import compute_cross_model_shap
        X, y = _make_data(n=300)
        pred = _ShapStubPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42,
            data_source_label='real', model_keys=MODEL_KEYS,
            enable_calibration=False, feature_set='v2')
        assert ok is True
        # 22 维矩阵输入：按各模型 feature_names_in_ 选列
        result = compute_cross_model_shap(pred, X[:100])
        assert result is not None
        assert set(result['feature_names']) == set(SELECTED_FEATURES_V2)
        top5 = {e['feature'] for e in result['consensus'][:5]}
        assert 'age' in top5
        assert 'cumulative_exposure' in top5


# ============================================================================
# 3. Markdown 共识表
# ============================================================================

class TestFormatTable:

    def _result(self, trained_predictor):
        from tb_risk.scoring.ml.shap_analysis import compute_cross_model_shap
        X, _ = _make_data(n=150)
        return compute_cross_model_shap(trained_predictor, X)

    def test_table_contains_key_columns(self, trained_predictor):
        from tb_risk.scoring.ml.shap_analysis import (
            compute_cross_model_shap, format_cross_model_shap_table)
        X, _ = _make_data(n=150)
        result = compute_cross_model_shap(trained_predictor, X)
        md = format_cross_model_shap_table(result)
        assert '共识份额' in md
        assert '排名区间' in md
        # 信号特征应出现在表中
        assert 'age' in md
        assert 'cumulative_exposure' in md

    def test_table_top_n_respected(self, trained_predictor):
        from tb_risk.scoring.ml.shap_analysis import (
            compute_cross_model_shap, format_cross_model_shap_table)
        X, _ = _make_data(n=150)
        result = compute_cross_model_shap(trained_predictor, X)
        md = format_cross_model_shap_table(result, top_n=5)
        data_rows = [ln for ln in md.strip().splitlines()
                     if ln.startswith('|') and '---' not in ln
                     and '共识份额' not in ln]
        assert len(data_rows) == 5


# ============================================================================
# 4. 训练报告 §3 接入
# ============================================================================

class TestReportIntegration:

    def _make_report_predictor(self):
        """带真实拟合模型 + last_feature_matrix 的报告桩。"""
        from sklearn.ensemble import (RandomForestClassifier,
                                      GradientBoostingClassifier)
        import pandas as pd

        class _Predictor:
            FEATURE_NAMES = ['age', 'cumulative_exposure', 'has_symptoms']
            INTERACTION_FEATURE_NAMES = []
            ALL_FEATURE_NAMES = FEATURE_NAMES
            FEATURE_DESCRIPTIONS = {'age': '年龄'}
            INTERACTION_FEATURE_DESCRIPTIONS = {}

            def __init__(self):
                self.is_trained = True
                self.training_sample_count = 200
                self.use_real_data = True
                self.shap_explainer = None
                self.last_shap_values = None
                self.last_feature_matrix = self._X
                self.models = self._models
                self.model_performance = {
                    'random_forest': {'name': '随机森林', 'AUROC': 0.85,
                                      'AUPRC': 0.4},
                    'gradient_boosting': {'name': '梯度提升', 'AUROC': 0.84,
                                          'AUPRC': 0.4},
                }

            def compute_bootstrap_ci(self, n_bootstrap=50):
                return {}

            def compute_calibration(self, n_bins=10, random_state=42):
                return {}

        rng = np.random.RandomState(42)
        n = 200
        X = np.column_stack([
            rng.randint(15, 85, n),
            rng.uniform(0, 800, n),
            rng.randint(0, 2, n),
        ])
        logit = 0.035 * (X[:, 0] - 45) + 0.004 * X[:, 1] + 0.9 * X[:, 2] - 1.2
        y = (logit + rng.normal(0, 0.5, n) > 0).astype(int)
        frame = pd.DataFrame(X, columns=_Predictor.ALL_FEATURE_NAMES)
        _Predictor._X = X
        _Predictor._models = {
            'random_forest': RandomForestClassifier(
                n_estimators=30, random_state=42).fit(frame, y),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=42).fit(frame, y),
        }
        return _Predictor()

    def test_report_contains_cross_model_section(self):
        from tb_risk.export_utils.report_export import _generate_markdown_report
        md = _generate_markdown_report(self._make_report_predictor())
        assert '跨模型 SHAP 共识' in md
        assert '共识份额' in md

    def test_report_without_models_has_no_section(self):
        """旧桩（无 models / last_feature_matrix）不出现该段——向后兼容。"""
        from tb_risk.export_utils.report_export import _generate_markdown_report

        class _Bare:
            FEATURE_NAMES = ['age']
            INTERACTION_FEATURE_NAMES = []
            ALL_FEATURE_NAMES = ['age']
            FEATURE_DESCRIPTIONS = {}
            INTERACTION_FEATURE_DESCRIPTIONS = {}
            is_trained = True
            training_sample_count = 10
            use_real_data = False
            shap_explainer = None
            last_shap_values = None
            model_performance = {'random_forest': {
                'name': '随机森林', 'AUROC': 0.8, 'AUPRC': 0.3}}

            def compute_bootstrap_ci(self, n_bootstrap=50):
                return {}

            def compute_calibration(self, n_bins=10, random_state=42):
                return {}

        md = _generate_markdown_report(_Bare())
        assert '跨模型 SHAP 共识' not in md
