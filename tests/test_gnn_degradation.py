#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN 依赖降级测试套件（问题三）

覆盖：
- GNN 环境分级检测（_env：tier 0-3 + 状态 + 安装指引）
- 纯 NumPy 轻量图卷积回退（_numpy_gnn：特征提取 / PageRank 传播 / 风险输出）
- 推理回退接入（inference.predict_gnn_risk：无 torch/PyG 或未训练时回退 NumPy）
- 三方向集成器（integrator：GNN 分支始终有可用输出并参与集成）

遵循项目 unittest + pytest 约定，对可选依赖做 skip 保护。
"""
import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

from tb_risk.ml.gnn import (  # noqa: E402
    NUMPY_AVAILABLE,
    gnn_degradation_level,
    gnn_degradation_name,
    gnn_tier_description,
    gnn_install_guidance,
    gnn_env_status,
    gnn_env_status_text,
    NumPyGraphRiskPredictor,
    predict_risk_numpy,
)


def _skip_if_no_numpy(cls):
    return unittest.skipUnless(_NUMPY, "需要 numpy")(cls)


def _make_contact(name="接触者A", cumulative_exposure=40, age=35,
                  ventilation=3, has_symptoms=0, record_id=None, _id=None):
    """构造一个最小合法的接触者数据字典。"""
    return {
        'name': name,
        'record_id': record_id,
        '_id': _id,
        'age': age, 'has_symptoms': has_symptoms, 'has_tb': 0,
        'bcg_vaccine': 1, 'contact_distance': 'close',
        'ventilation': ventilation, 'exposure_setting': 'general',
        'cumulative_exposure': cumulative_exposure,
        'single_duration': 60, 'freq_density': 10, 'time_span': 8,
    }


def _make_assessment(family_entries, social_entries=None, patient_info=None):
    """构造最小合法的评估对象（供 inference/integrator 消费）。"""
    class _Assessment:
        pass
    a = _Assessment()
    a.family_entries = family_entries or []
    a.social_entries = social_entries or []
    a.patient_info = patient_info or {'basic_info': {'sputum_smear': 1, 'has_cavity': 1}}
    a.results = {
        'base_infection_probability': 30,
        'potential_patients': {
            'family': a.family_entries,
            'social': a.social_entries,
        },
    }
    return a


# ---------------------------------------------------------------------------
# 1. GNN 环境分级检测
# ---------------------------------------------------------------------------

class TestGNNEnvDetection(unittest.TestCase):
    """_env：环境分级、状态汇总与安装指引。"""

    def test_level_in_range(self):
        self.assertIn(gnn_degradation_level(), (0, 1, 2, 3))

    def test_level0_requires_both(self):
        # 只有 torch+pyg 同时在位才算 tier 0
        from tb_risk.ml.gnn import _env
        # 仅 torch → tier 1（手写 GATv2 层）
        with mock.patch.object(_env, 'PYTORCH_AVAILABLE', True), \
                mock.patch.object(_env, 'PYG_AVAILABLE', False):
            self.assertEqual(_env.gnn_degradation_level(), 1)
        # 仅 pyg（无 torch）：torch_geometric 无法脱离 torch 使用，
        # 属于退化组合，按能力落到 tier 2（仅 NumPy 回退）
        with mock.patch.object(_env, 'PYTORCH_AVAILABLE', False), \
                mock.patch.object(_env, 'PYG_AVAILABLE', True):
            self.assertEqual(_env.gnn_degradation_level(), 2)

    def test_level2_numpy_only(self):
        from tb_risk.ml.gnn import _env
        with mock.patch.object(_env, 'PYTORCH_AVAILABLE', False), \
                mock.patch.object(_env, 'PYG_AVAILABLE', False), \
                mock.patch.object(_env, 'NUMPY_AVAILABLE', True):
            self.assertEqual(_env.gnn_degradation_level(), 2)

    def test_level3_nothing(self):
        from tb_risk.ml.gnn import _env
        with mock.patch.object(_env, 'PYTORCH_AVAILABLE', False), \
                mock.patch.object(_env, 'PYG_AVAILABLE', False), \
                mock.patch.object(_env, 'NUMPY_AVAILABLE', False):
            self.assertEqual(_env.gnn_degradation_level(), 3)

    def test_env_status_keys(self):
        st = gnn_env_status()
        for key in ('level', 'name', 'description', 'torch_available',
                    'pyg_available', 'numpy_available', 'install_command',
                    'full_gnn_ready', 'branch_available'):
            self.assertIn(key, st)
        # branch_available 与 full_gnn_ready 的关系
        self.assertEqual(st['branch_available'], st['level'] <= 2)
        self.assertEqual(st['full_gnn_ready'], st['level'] == 0)

    def test_env_status_full_gnn_ready_only_level0(self):
        st = gnn_env_status()
        self.assertEqual(st['full_gnn_ready'],
                         bool(st['torch_available'] and st['pyg_available']))

    def test_install_guidance_consistency(self):
        # 各级别名称/说明/指引一致
        for level in (0, 1, 2, 3):
            name = gnn_degradation_name(level)
            desc = gnn_tier_description(level)
            cmd = gnn_install_guidance(level)
            self.assertIsInstance(name, str)
            self.assertIsInstance(desc, str)
            self.assertTrue(name)
            self.assertTrue(desc)
            if level == 0:
                self.assertIsNone(cmd)
            else:
                self.assertIsInstance(cmd, str)
                self.assertTrue(cmd)

    def test_install_guidance_upgrade_path(self):
        # tier 1 → 一键 [gnn]；tier 2 → 安装 torch；tier 3 → 安装 numpy
        self.assertIn('gnn', gnn_install_guidance(1))
        self.assertIn('torch', gnn_install_guidance(2))
        self.assertIn('numpy', gnn_install_guidance(3))

    def test_status_text(self):
        text = gnn_env_status_text()
        self.assertIsInstance(text, str)
        self.assertIn('GNN 环境', text)

    def test_exported_from_packages(self):
        # 从 ml/gnn 与 ml/framework 均可导入（_env 已导出）
        from tb_risk.ml.gnn import gnn_env_status as g1
        from tb_risk.ml.framework import gnn_env_status as g2
        self.assertIs(g1, g2)


# ---------------------------------------------------------------------------
# 2. NumPy 轻量图卷积回退
# ---------------------------------------------------------------------------

@_skip_if_no_numpy
class TestNumPyGraphRiskPredictor(unittest.TestCase):
    """_numpy_gnn：特征提取、图传播与风险输出。"""

    def setUp(self):
        self.predictor = NumPyGraphRiskPredictor(random_state=42)

    def test_feature_vector_dim(self):
        from tb_risk.ml.gnn._numpy_gnn import feature_vector
        f = feature_vector(_make_contact(), 'family')
        self.assertEqual(f.shape, (12,))
        # 特征应归一化到 [0, 1]
        self.assertTrue(float(f.min()) >= 0.0)
        self.assertTrue(float(f.max()) <= 1.0)

    def test_feature_vector_family_social_differ(self):
        from tb_risk.ml.gnn._numpy_gnn import feature_vector
        f_family = feature_vector(_make_contact(), 'family')
        f_social = feature_vector(_make_contact(), 'social')
        # 家庭接触基线（最后一位）不同
        self.assertNotEqual(float(f_family[-1]), float(f_social[-1]))

    def test_feature_vector_deterministic(self):
        from tb_risk.ml.gnn._numpy_gnn import feature_vector
        c = _make_contact()
        self.assertTrue(np.array_equal(feature_vector(c, 'family'),
                                       feature_vector(c, 'family')))

    def test_predict_contact_risk_structure(self):
        r = self.predictor.predict_contact_risk(_make_contact(), 'family')
        self.assertIn('risk_probability', r)
        self.assertIn('risk_class', r)
        self.assertEqual(r['backend'], 'numpy')
        self.assertTrue(r['gnn_used'])
        self.assertFalse(r['network_aware'])
        self.assertTrue(0.0 <= r['risk_probability'] <= 100.0)
        self.assertIn(r['risk_class'], (0, 1))

    def test_risk_monotonic_exposure(self):
        low = self.predictor.predict_contact_risk(
            _make_contact(cumulative_exposure=5), 'family')
        high = self.predictor.predict_contact_risk(
            _make_contact(cumulative_exposure=500), 'family')
        self.assertGreater(high['risk_probability'], low['risk_probability'])

    def test_risk_monotonic_symptoms(self):
        low = self.predictor.predict_contact_risk(
            _make_contact(has_symptoms=0), 'family')
        high = self.predictor.predict_contact_risk(
            _make_contact(has_symptoms=1), 'family')
        self.assertGreaterEqual(high['risk_probability'], low['risk_probability'])

    def test_predict_network_risk_network_aware(self):
        family = [
            _make_contact(name='家1', cumulative_exposure=80, record_id='f0'),
            _make_contact(name='家2', cumulative_exposure=20, record_id='f1'),
        ]
        social = [_make_contact(name='社1', cumulative_exposure=10, record_id='s0')]
        target = family[0]
        r = self.predictor.predict_network_risk(
            target, 'family', {'basic_info': {'sputum_smear': 2, 'has_cavity': 2}},
            family, social)
        self.assertEqual(r['backend'], 'numpy')
        self.assertTrue(r['network_aware'])
        self.assertTrue(r['gnn_used'])
        self.assertIn('node_idx', r)
        self.assertIsInstance(r['node_idx'], int)
        self.assertTrue(0 <= r['node_idx'])
        self.assertTrue(0.0 <= r['risk_probability'] <= 100.0)

    def test_predict_network_risk_target_by_record_id(self):
        family = [
            _make_contact(name='家1', cumulative_exposure=50, record_id='t1'),
            _make_contact(name='家2', cumulative_exposure=20, record_id='t2'),
        ]
        # 目标按 record_id 匹配，命中的是 t1（暴露更高）
        r = self.predictor.predict_network_risk(
            _make_contact(name='任意', cumulative_exposure=1, record_id='t1'),
            'family', None, family, [])
        self.assertTrue(r['network_aware'])
        self.assertEqual(r['node_idx'], 1)  # 节点0为患者，family 从 1 起

    def test_predict_network_risk_deterministic(self):
        family = [_make_contact(cumulative_exposure=50, record_id='f0')]
        a = self.predictor.predict_network_risk(
            family[0], 'family', None, family, [])
        b = self.predictor.predict_network_risk(
            family[0], 'family', None, family, [])
        self.assertEqual(a['risk_probability'], b['risk_probability'])

    def test_predict_unified_entry(self):
        # 有网络 → 网络感知；无网络 → 单节点估计
        r_net = self.predictor.predict(
            _make_contact(record_id='x'), 'family', None,
            [_make_contact(record_id='x')], [])
        self.assertTrue(r_net['network_aware'])
        r_single = self.predictor.predict(_make_contact(), 'family')
        self.assertFalse(r_single['network_aware'])

    def test_predict_risk_numpy_function(self):
        r = predict_risk_numpy(_make_contact(), 'family')
        self.assertEqual(r['backend'], 'numpy')
        self.assertTrue(0.0 <= r['risk_probability'] <= 100.0)


# ---------------------------------------------------------------------------
# 3. 推理回退接入（inference.predict_gnn_risk）
# ---------------------------------------------------------------------------

class _FakePredictor:
    """最小预测器：仅具备 predict_gnn_risk 所需属性（gnn_is_trained）。"""
    gnn_is_trained = False


@_skip_if_no_numpy
class TestInferenceFallback(unittest.TestCase):
    """inference：predict_gnn_risk 在无 torch/PyG 或未训练时回退 NumPy。"""

    def _call(self):
        from tb_risk.scoring.ml.gnn_training.inference import predict_gnn_risk
        contact = _make_contact(cumulative_exposure=60, record_id='c0')
        assessment = _make_assessment(
            family_entries=[contact],
            social_entries=[_make_contact(cumulative_exposure=5, record_id='s0')],
        )
        return predict_gnn_risk(_FakePredictor(), assessment, contact, 'family')

    def test_not_trained_returns_numpy_fallback(self):
        r = self._call()
        self.assertIsNotNone(r)
        self.assertEqual(r['backend'], 'numpy')
        self.assertEqual(r['degradation'], 'not_trained')
        self.assertTrue(r['gnn_used'])
        self.assertTrue(0.0 <= r['risk_probability'] <= 100.0)

    def test_torch_missing_returns_numpy_fallback(self):
        import tb_risk.scoring.ml.gnn_training.inference as inf
        with mock.patch.object(inf, 'PYTORCH_AVAILABLE', False), \
                mock.patch.object(inf, 'PYG_AVAILABLE', True):
            r = self._call()
        self.assertIsNotNone(r)
        self.assertEqual(r['degradation'], 'torch_missing')

    def test_pyg_missing_returns_numpy_fallback(self):
        import tb_risk.scoring.ml.gnn_training.inference as inf
        with mock.patch.object(inf, 'PYTORCH_AVAILABLE', True), \
                mock.patch.object(inf, 'PYG_AVAILABLE', False):
            r = self._call()
        self.assertIsNotNone(r)
        self.assertEqual(r['degradation'], 'pyg_missing')

    def test_fallback_uses_network_when_available(self):
        # 有家庭成员/社会接触者时，回退结果应是网络感知的
        from tb_risk.scoring.ml.gnn_training.inference import predict_gnn_risk
        contact = _make_contact(cumulative_exposure=60, record_id='c0')
        assessment = _make_assessment(
            family_entries=[contact, _make_contact(record_id='c1')],
            social_entries=[],
        )
        r = predict_gnn_risk(_FakePredictor(), assessment, contact, 'family')
        self.assertIsNotNone(r)
        self.assertTrue(r['network_aware'])


# ---------------------------------------------------------------------------
# 4. 三方向集成器（GNN 分支始终有输出）
# ---------------------------------------------------------------------------

@_skip_if_no_numpy
class TestIntegratorGNNBranch(unittest.TestCase):
    """integrator：GNN 不可用/未训练时，NumPy 回退结果仍参与集成。"""

    def test_integrate_has_gnn_branch_and_contribution(self):
        from tb_risk.core.integrator import ThreeDirectionIntegrator
        from tb_risk.scoring.predictor import MLRiskPredictor

        family = [_make_contact(cumulative_exposure=60, record_id='c0')]
        assessment = _make_assessment(family_entries=family)
        contact = family[0]

        integrator = ThreeDirectionIntegrator()
        predictor = MLRiskPredictor(random_state=42)  # 未训练 → GNN 回退
        results = integrator.integrate_predictions(
            predictor, assessment, contact, 'family')

        # GNN 分支始终有可用输出（NumPy 回退）
        self.assertIn('gnn', results)
        self.assertEqual(results['gnn'].get('backend'), 'numpy')
        # 集成结果存在且 GNN 贡献被标记
        self.assertIn('ensemble', results)
        self.assertTrue(results['ensemble'].get('has_gnn_contribution'))
        self.assertIn('ml', results)
        self.assertIn('seir', results)


if __name__ == '__main__':
    unittest.main()
