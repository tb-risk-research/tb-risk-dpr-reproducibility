#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""要点C: v4.0 后验预测分布与下一代矩阵 R0 测试

测试 ``SEIRParameterUncertainty.generate_posterior_predictive_v4`` 与
``_compute_R0_v4_ngm`` 的正确性。

文献:
  Diekmann O, Heesterbeek JA, Metz JA. (1990) J Math Biol — 下一代矩阵
  van den Driessche P, Watmough J. (2008) Math Biosci Eng — R0 谱半径
"""

import os
import sys
import unittest

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.seir.uncertainty import SEIRParameterUncertainty
from tb_risk.seir.stochastic import StochasticSEIRModel
from tb_risk.seir._stochastic_common import (
    N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3, STATE_SIZE_V4,
)


def _make_v4_initial_state(population=1000, i0=10):
    """构造 270D v4 初始状态: 年龄=2(15-49), HIV=0(HIV-), DR=0(DS)"""
    state = np.zeros(STATE_SIZE_V4)
    a0, h0, d0 = 2, 0, 0  # 15-49, HIV-, DS
    base_idx = ((a0 * N_HIV_V4 + h0) * N_DR_V4 + d0) * N_COMPARTMENTS_V3
    state[base_idx + 0] = float(population - i0)  # S
    state[base_idx + 5] = float(i0)               # I_sp
    return state


class TestComputeR0V4NGM(unittest.TestCase):
    """要点C: v4 下一代矩阵 R0 单元测试"""

    def setUp(self):
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.inf = SEIRParameterUncertainty(
            stochastic_seir_model=self.model, population=1000)

    def test_returns_positive_float(self):
        """R0 为正有限浮点数"""
        r0 = self.inf._compute_R0_v4_ngm(beta=0.3)
        self.assertTrue(np.isfinite(r0), f"R0={r0} 非有限")
        self.assertGreater(r0, 0.0, f"R0={r0} 非正")

    def test_r0_scales_with_beta(self):
        """R0 与 β 成正比（线性关系）"""
        r0_low = self.inf._compute_R0_v4_ngm(beta=0.1)
        r0_high = self.inf._compute_R0_v4_ngm(beta=0.5)
        # 比值应接近 5.0 (0.5/0.1)
        ratio = r0_high / max(r0_low, 1e-10)
        self.assertAlmostEqual(ratio, 5.0, places=2,
                               msg=f"R0 比值 {ratio} 不接近 5.0")

    def test_r0_decreases_with_art_reduction(self):
        """ART 降低应减小 R0"""
        r0_no_art = self.inf._compute_R0_v4_ngm(
            beta=0.3, art_reduction=0.0)
        r0_with_art = self.inf._compute_R0_v4_ngm(
            beta=0.3, art_reduction=0.9)
        self.assertLess(r0_with_art, r0_no_art,
                        "ART 降低应减小 R0")

    def test_r0_decreases_with_dr_fitness_cost(self):
        """DR 适合度代价应减小 R0"""
        r0_no_cost = self.inf._compute_R0_v4_ngm(
            beta=0.3, dr_fitness_cost=0.0)
        r0_with_cost = self.inf._compute_R0_v4_ngm(
            beta=0.3, dr_fitness_cost=0.8)
        self.assertLess(r0_with_cost, r0_no_cost,
                        "DR 适合度代价应减小 R0")

    def test_r0_with_custom_waifw(self):
        """自定义 WAIFW 矩阵可用"""
        W = np.eye(5) * 10.0  # 完全同年龄混合
        r0 = self.inf._compute_R0_v4_ngm(beta=0.3, waifw=W)
        self.assertTrue(np.isfinite(r0))
        self.assertGreater(r0, 0.0)


class TestGeneratePosteriorPredictiveV4(unittest.TestCase):
    """要点C: generate_posterior_predictive_v4 集成测试"""

    def setUp(self):
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.inf = SEIRParameterUncertainty(
            stochastic_seir_model=self.model, population=1000)
        self.init_state = _make_v4_initial_state(1000, 10)

        # 构造合成后验样本（围绕默认值的小扰动）
        rng = np.random.RandomState(42)
        n = 30
        self.inf.posterior_samples = {
            'beta': 0.3 + rng.normal(0, 0.02, n),
            'rho_fast': 0.1 / 365.0 + rng.normal(0, 1e-5, n),
            'rho_react': 0.004 / 365.0 + rng.normal(0, 1e-6, n),
            'sigma_clear': 1.0 / 365.0 + rng.normal(0, 1e-5, n),
            'gamma': 0.05 + rng.normal(0, 1e-4, n),
            'omega_reg_m': 1.0 / 365.0 + rng.normal(0, 1e-5, n),
            'omega_reg_sub': 1.0 / 365.0 + rng.normal(0, 1e-5, n),
            'beta_exo': 0.33 + rng.normal(0, 1e-3, n),
            'eta_sn': 0.35 + rng.normal(0, 1e-3, n),
            'rr_hiv': 20.0 + rng.normal(0, 0.5, n),
            'art_reduction': 0.65 + rng.normal(0, 1e-3, n),
            'hiv_infection_rate': 0.001 + rng.normal(0, 1e-5, n),
            'art_initiation_rate': 0.1 / 365.0 + rng.normal(0, 1e-6, n),
            'dr_fitness_cost': 0.15 + rng.normal(0, 1e-3, n),
            'p_acq': 0.03 + rng.normal(0, 1e-4, n),
        }

    def test_returns_required_keys(self):
        """返回必需键"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3, use_sde=False)
        for key in ['trajectories', 'active_I', 'ci_bands',
                    'by_age', 'by_hiv', 'by_dr',
                    'r0_samples', 'r0_mean', 'r0_ci', 'p_r0_gt_1',
                    'peak_I_samples', 'n_samples', 'method']:
            self.assertIn(key, result, f"缺少键: {key}")

    def test_no_posterior_returns_error(self):
        """无后验样本返回 error"""
        inf = SEIRParameterUncertainty(population=1000)
        result = inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3)
        self.assertIn('error', result)

    def test_empty_posterior_returns_error(self):
        """空后验样本返回 error"""
        inf = SEIRParameterUncertainty(population=1000)
        inf.posterior_samples = {'beta': np.array([])}
        result = inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3)
        self.assertIn('error', result)

    def test_missing_beta_returns_error(self):
        """后验样本缺少 beta 返回 error"""
        inf = SEIRParameterUncertainty(population=1000)
        inf.posterior_samples = {'rho_fast': np.array([0.1])}
        result = inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3)
        self.assertIn('error', result)

    def test_trajectories_shape(self):
        """轨迹形状为 (n_samples, n_steps, 270)"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3, use_sde=False)
        trajs = result['trajectories']
        self.assertEqual(trajs.shape[0], 3)
        self.assertEqual(trajs.shape[2], STATE_SIZE_V4)
        self.assertGreater(trajs.shape[1], 1)

    def test_ci_bands_structure(self):
        """ci_bands 包含 lower/upper/median/time"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3, use_sde=False)
        ci = result['ci_bands']
        for key in ['lower', 'upper', 'median', 'time']:
            self.assertIn(key, ci)
        self.assertEqual(len(ci['lower']), len(ci['upper']))
        self.assertEqual(len(ci['lower']), len(ci['median']))

    def test_by_age_has_5_groups(self):
        """by_age 包含 5 个年龄组"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3, use_sde=False)
        self.assertEqual(len(result['by_age']['median']), N_AGE_V4)

    def test_by_hiv_has_3_layers(self):
        """by_hiv 包含 3 个 HIV 层"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3, use_sde=False)
        self.assertEqual(len(result['by_hiv']['median']), N_HIV_V4)

    def test_by_dr_has_2_layers(self):
        """by_dr 包含 2 个 DR 层"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3, use_sde=False)
        self.assertEqual(len(result['by_dr']['median']), N_DR_V4)

    def test_r0_samples_length_matches_n_samples(self):
        """R0 样本数等于 n_samples"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=5, use_sde=False)
        self.assertEqual(len(result['r0_samples']), result['n_samples'])

    def test_r0_mean_finite(self):
        """R0 均值为有限值"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3, use_sde=False)
        self.assertTrue(np.isfinite(result['r0_mean']))

    def test_p_r0_gt_1_in_range(self):
        """P(R0 > 1) 在 [0, 1] 范围内"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3, use_sde=False)
        self.assertGreaterEqual(result['p_r0_gt_1'], 0.0)
        self.assertLessEqual(result['p_r0_gt_1'], 1.0)

    def test_method_ode_when_sde_disabled(self):
        """use_sde=False 时 method 为 ODE"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3, use_sde=False)
        self.assertEqual(result['method'], 'ODE')

    def test_n_subsample_limits_pool(self):
        """n_subsample 限制后验样本池"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=5,
            use_sde=False, n_subsample=10)
        self.assertLessEqual(result['n_samples'], 5)

    def test_peak_I_nonneg(self):
        """峰值活动性 TB 非负"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3, use_sde=False)
        for p in result['peak_I_samples']:
            self.assertGreaterEqual(p, -1e-6)

    def test_n_age_hiv_dr_layers_reported(self):
        """返回中报告年龄/HIV/DR 层数"""
        result = self.inf.generate_posterior_predictive_v4(
            self.init_state, (0, 10), 2.0, n_samples=3, use_sde=False)
        self.assertEqual(result['n_age_groups'], N_AGE_V4)
        self.assertEqual(result['n_hiv_layers'], N_HIV_V4)
        self.assertEqual(result['n_dr_layers'], N_DR_V4)


if __name__ == '__main__':
    unittest.main()
