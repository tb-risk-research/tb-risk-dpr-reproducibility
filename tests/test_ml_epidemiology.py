#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - ML 流行病学计算模块单元测试

针对 scoring/ml/epidemiology.py 做行为级测试：
- calculate_treatment_infectivity_factor（治疗阶段传染性衰减）
- calculate_multilayer_network_r0（三层网络基本再生数，下一代矩阵法）

这些函数无状态，仅依赖 predictor 的 treatment_infectivity_decay 与
constants 中的传播系数，可用轻量桩测试，无需训练模型。
"""
import os
import sys
import unittest

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)


class _PredictorStub:
    """最小 predictor 桩，仅提供治疗传染性衰减表。"""

    def __init__(self):
        self.treatment_infectivity_decay = {
            'week_0': 0.85, 'week_2': 0.60,
            'week_8': 0.30, 'week_24': 0.10,
        }


class TestTreatmentInfectivityFactor(unittest.TestCase):
    """治疗阶段传染性衰减因子测试。"""

    def setUp(self):
        from tb_risk.scoring.ml.epidemiology import calculate_treatment_infectivity_factor
        self._fn = calculate_treatment_infectivity_factor
        self.predictor = _PredictorStub()

    def test_below_2_weeks(self):
        self.assertEqual(self._fn(self.predictor, 0), 0.85)
        self.assertEqual(self._fn(self.predictor, 1), 0.85)

    def test_2_to_8_weeks(self):
        self.assertEqual(self._fn(self.predictor, 2), 0.60)
        self.assertEqual(self._fn(self.predictor, 7), 0.60)

    def test_8_to_24_weeks(self):
        self.assertEqual(self._fn(self.predictor, 8), 0.30)
        self.assertEqual(self._fn(self.predictor, 23), 0.30)

    def test_24_weeks_and_beyond(self):
        self.assertEqual(self._fn(self.predictor, 24), 0.10)
        self.assertEqual(self._fn(self.predictor, 52), 0.10)

    def test_monotonic_decreasing(self):
        """治疗越久传染性越低（各阶段因子应单调递减）。"""
        decay = self.predictor.treatment_infectivity_decay
        stages = ['week_0', 'week_2', 'week_8', 'week_24']
        values = [decay[s] for s in stages]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_returns_value_in_range(self):
        factor = self._fn(self.predictor, 3)
        self.assertGreaterEqual(factor, 0.0)
        self.assertLessEqual(factor, 1.0)


class TestMultilayerNetworkR0(unittest.TestCase):
    """三层网络基本再生数 R₀ 测试。"""

    def setUp(self):
        from tb_risk.scoring.ml.epidemiology import (
            calculate_multilayer_network_r0,
        )
        self._fn = calculate_multilayer_network_r0
        self.predictor = _PredictorStub()

    def _contact(self, exposure_hours):
        return {'cumulative_exposure': exposure_hours}

    def test_empty_contacts_returns_structure(self):
        result = self._fn(self.predictor, [], [])
        self.assertIn('R0_family', result)
        self.assertIn('R0_social', result)
        self.assertIn('R0_community', result)
        self.assertIn('R0_total', result)
        self.assertIn('contact_matrix', result)
        self.assertIn('method', result)
        self.assertEqual(result['R0_family'], 0.0)
        self.assertEqual(result['R0_social'], 0.0)
        # 空接触时仅社区层对角线贡献社区基线
        K = np.array(result['contact_matrix'])
        self.assertEqual(K.shape, (3, 3))
        self.assertEqual(K[0, 0], 0.0)
        self.assertEqual(K[1, 1], 0.0)
        self.assertAlmostEqual(K[2, 2], 0.05, places=6)

    def test_community_baseline_contribution(self):
        result = self._fn(self.predictor, [], [])
        self.assertAlmostEqual(result['R0_community'], 0.05, places=6)

    def test_family_r0_proportional_to_exposure(self):
        from tb_risk.constants import BETA_FAMILY_ANNUAL, HOURS_PER_YEAR
        contact = self._contact(8760)  # 一年暴露
        result = self._fn(self.predictor, [contact], [])
        self.assertAlmostEqual(result['R0_family'], BETA_FAMILY_ANNUAL, places=6)

    def test_social_r0_proportional_to_exposure(self):
        from tb_risk.constants import BETA_SOCIAL_ANNUAL, HOURS_PER_YEAR
        contact = self._contact(8760)
        result = self._fn(self.predictor, [], [contact])
        self.assertAlmostEqual(result['R0_social'], BETA_SOCIAL_ANNUAL, places=6)

    def test_r0_total_dominates_diagonals(self):
        """谱半径 ≥ 每个对角元素（非负矩阵性质）。"""
        family = [self._contact(8760) for _ in range(2)]
        social = [self._contact(17520)]
        result = self._fn(self.predictor, family, social)
        K = np.array(result['contact_matrix'])
        self.assertEqual(K.shape, (3, 3))
        diag = np.max(np.diag(K))
        self.assertGreaterEqual(result['R0_total'], diag)

    def test_high_exposure_above_threshold(self):
        family = [self._contact(8760 * 10) for _ in range(3)]
        result = self._fn(self.predictor, family, [])
        self.assertEqual(result['is_above_threshold'], result['R0_total'] > 1.0)

    def test_low_exposure_below_threshold(self):
        result = self._fn(self.predictor, [], [])
        self.assertFalse(result['is_above_threshold'])

    def test_eigenvalues_length(self):
        result = self._fn(self.predictor, [self._contact(8760)], [])
        self.assertEqual(len(result['eigenvalues']), 3)

    def test_eigenvalues_are_real(self):
        """谱半径方法中特征值应可转为实数值。"""
        result = self._fn(self.predictor, [self._contact(8760)], [])
        for v in result['eigenvalues']:
            self.assertIsInstance(v, float)
            self.assertEqual(v, v)  # 非 NaN


if __name__ == '__main__':
    unittest.main()
