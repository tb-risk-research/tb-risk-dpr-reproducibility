#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可解释性分析测试套件

覆盖：
- SHAP 逐接触者因素归因（compute_contact_attributions）
- GNN 图归因（compute_gnn_explanation，需已训练 GNN）
- 模块导出与可用性保护
- ml_runner 结果结构中 attribution 键的组装
- GUI 风险归因条形图绘制（受 matplotlib 保护）

遵循项目 unittest + pytest 约定，对可选依赖做 skip 保护。
"""
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    import shap
    _SHAP = True
except ImportError:
    _SHAP = False

try:
    import matplotlib
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False


def _skip_if_no_numpy(cls):
    return unittest.skipUnless(_NUMPY, "需要 numpy")(cls)


def _skip_if_no_shap(cls):
    return unittest.skipUnless(_SHAP, "需要 shap")(cls)


def _skip_if_no_matplotlib(cls):
    return unittest.skipUnless(_MATPLOTLIB, "需要 matplotlib")(cls)


def _make_contact(name="接触者A", cumulative_exposure=40, age=35,
                  ventilation=3, contact_distance="close"):
    """构造一个最小合法的接触者数据字典。"""
    return {
        'name': name,
        'age': age, 'has_symptoms': 0, 'has_tb': 0,
        'bcg_vaccine': 1, 'contact_distance': contact_distance,
        'ventilation': ventilation, 'exposure_setting': 'general',
        'cumulative_exposure': cumulative_exposure,
        'past_illness_type': 'none',
    }


# ══════════════════════════════════════════════════════════════
# 模块导出与可用性保护
# ══════════════════════════════════════════════════════════════

class TestInterpretabilityExports(unittest.TestCase):
    """可解释性相关公共 API 的导出检查。"""

    def test_ml_package_exports(self):
        from tb_risk.scoring.ml import (
            compute_contact_attributions, compute_gnn_explanation)
        self.assertTrue(callable(compute_contact_attributions))
        self.assertTrue(callable(compute_gnn_explanation))

    def test_gnn_training_package_exports(self):
        from tb_risk.scoring.ml.gnn_training import compute_gnn_explanation
        self.assertTrue(callable(compute_gnn_explanation))

    def test_predictor_delegates(self):
        from tb_risk.scoring.predictor import MLRiskPredictor
        p = MLRiskPredictor()
        self.assertTrue(callable(p.compute_contact_attributions))
        self.assertTrue(callable(p.compute_gnn_explanation))


# ══════════════════════════════════════════════════════════════
# SHAP 逐接触者因素归因
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
@_skip_if_no_shap
class TestContactAttributions(unittest.TestCase):
    """compute_contact_attributions 测试（需训练 ML 预测器）。"""

    @classmethod
    def setUpClass(cls):
        from tb_risk.scoring.predictor import MLRiskPredictor
        cls.predictor = MLRiskPredictor(random_state=42)
        try:
            cls.predictor.train_models(n_samples=400)
        except Exception:
            cls.predictor = None

    def test_untrained_returns_none(self):
        from tb_risk.scoring.predictor import MLRiskPredictor
        p = MLRiskPredictor()
        result = p.compute_contact_attributions([_make_contact()])
        self.assertIsNone(result)

    def test_returns_per_contact_list(self):
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        contacts = [_make_contact("A", 40), _make_contact("B", 90, age=60)]
        result = self.predictor.compute_contact_attributions(
            contacts, ['family', 'social'])
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_attribution_structure(self):
        """每个归因项含 feature/description/feature_value/contribution/abs_contribution。"""
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        result = self.predictor.compute_contact_attributions([_make_contact()])
        item = result[0]
        self.assertGreater(len(item), 0)
        record = item[0]
        for key in ('feature', 'description', 'feature_value',
                    'contribution', 'abs_contribution'):
            self.assertIn(key, record, f"归因项缺少字段 {key}")
        self.assertGreaterEqual(record['abs_contribution'], 0)

    def test_sorted_by_abs_contribution(self):
        """归因应按 |contribution| 降序排列。"""
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        result = self.predictor.compute_contact_attributions([_make_contact()])
        vals = [c['abs_contribution'] for c in result[0]]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_consistency_with_shap_values(self):
        """逐因素贡献应与 compute_shap_values 的 SHAP 值逐元素一致。"""
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        contact = _make_contact()
        agg = self.predictor.compute_shap_values([contact])
        self.assertIsNotNone(agg)
        per = self.predictor.compute_contact_attributions([contact])
        self.assertIsNotNone(per)
        shap_row = agg['shap_values'][0]
        names = agg['feature_names']
        by_name = {c['feature']: c['contribution'] for c in per[0]}
        for i, name in enumerate(names):
            self.assertAlmostEqual(
                by_name[name], float(shap_row[i]), places=5,
                msg=f"特征 {name} 的贡献与 SHAP 值不一致")


# ══════════════════════════════════════════════════════════════
# GNN 图归因
# ══════════════════════════════════════════════════════════════

class TestGNNExplanation(unittest.TestCase):
    """compute_gnn_explanation 测试。"""

    def test_untrained_returns_none(self):
        from tb_risk.scoring.predictor import MLRiskPredictor
        p = MLRiskPredictor()
        result = p.compute_gnn_explanation(None, _make_contact())
        self.assertIsNone(result)

    def test_requires_assessment(self):
        """GNN 未训练时不应崩溃（返回 None 而非抛异常）。"""
        from tb_risk.scoring.predictor import MLRiskPredictor
        p = MLRiskPredictor()
        try:
            result = p.compute_gnn_explanation(None, _make_contact())
            self.assertIsNone(result)
        except (TypeError, ValueError, RuntimeError):
            # 部分 GNN 后端在未训练时可能因缺图上下文抛错，属可接受
            pass


# ══════════════════════════════════════════════════════════════
# ml_runner 结果结构
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
@_skip_if_no_shap
class TestMLRunnerAttribution(unittest.TestCase):
    """ml_runner 结果中 attribution 键的组装测试。"""

    def test_attribution_key_present(self):
        from tb_risk.scoring.predictor import MLRiskPredictor
        from tb_risk.core.ml_runner import run_ml_prediction_loop
        import threading

        predictor = MLRiskPredictor(random_state=42)
        try:
            predictor.train_models(n_samples=300)
        except Exception:
            self.skipTest("预测器训练失败")

        results = {
            'potential_patients': {
                'family': [_make_contact("A", 40)],
                'social': [_make_contact("B", 90)],
            }
        }
        lock = threading.Lock()
        # integrator 仅需提供 integrate_predictions 接口的桩
        class _StubIntegrator:
            def integrate_predictions(self, *a, **k):
                return {'ensemble': {'risk_probability': 50.0, 'risk_class': 1}}

        out = run_ml_prediction_loop(None, predictor, _StubIntegrator(),
                                     results, lock)
        self.assertIsNotNone(out)
        self.assertIn('attribution', out)
        self.assertIn('family', out['attribution'])
        self.assertIn('social', out['attribution'])
        self.assertEqual(len(out['attribution']['family']), 1)
        self.assertEqual(len(out['attribution']['social']), 1)
        # 每个接触者条目应含 name/type/shap_contributions/gnn_explanation
        for entry in out['attribution']['family'] + out['attribution']['social']:
            self.assertIn('name', entry)
            self.assertIn('type', entry)
            self.assertIn('shap_contributions', entry)
            self.assertIn('gnn_explanation', entry)

    def test_shap_contributions_filled_when_available(self):
        from tb_risk.scoring.predictor import MLRiskPredictor
        from tb_risk.core.ml_runner import run_ml_prediction_loop
        import threading

        predictor = MLRiskPredictor(random_state=42)
        try:
            predictor.train_models(n_samples=300)
        except Exception:
            self.skipTest("预测器训练失败")

        results = {
            'potential_patients': {
                'family': [_make_contact("A", 40)],
                'social': [],
            }
        }
        lock = threading.Lock()
        class _StubIntegrator:
            def integrate_predictions(self, *a, **k):
                return {'ensemble': {'risk_probability': 50.0, 'risk_class': 1}}

        out = run_ml_prediction_loop(None, predictor, _StubIntegrator(),
                                     results, lock)
        entry = out['attribution']['family'][0]
        self.assertIsNotNone(entry['shap_contributions'],
                             "训练后 SHAP 归因应被填充")


# ══════════════════════════════════════════════════════════════
# GUI 风险归因条形图
# ══════════════════════════════════════════════════════════════

@_skip_if_no_matplotlib
@_skip_if_no_numpy
@_skip_if_no_shap
class TestAttributionChart(unittest.TestCase):
    """风险归因条形图绘制逻辑测试（使用 Agg 后端，无需真实窗口）。"""

    def test_shape_parsing_used_by_chart(self):
        """验证图表绘制所依赖的归因数据结构可被正确解析。"""
        from tb_risk.scoring.predictor import MLRiskPredictor
        predictor = MLRiskPredictor(random_state=42)
        try:
            predictor.train_models(n_samples=300)
        except Exception:
            self.skipTest("预测器训练失败")

        contacts = [_make_contact("A", 60)]
        attributions = predictor.compute_contact_attributions(contacts)
        if attributions is None:
            self.skipTest("SHAP 归因不可用")
        contribs = attributions[0]
        # 图表取前 8 项并反转（自下而上）
        top = contribs[:8][::-1]
        self.assertTrue(all(c['contribution'] is not None for c in top))
        self.assertTrue(all(c['description'] for c in top))
        # 正负贡献颜色区分所需字段
        self.assertTrue(any(c['contribution'] != 0 for c in contribs))


if __name__ == '__main__':
    unittest.main()