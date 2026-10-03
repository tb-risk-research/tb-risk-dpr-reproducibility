#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — v3.0 9-房室 SEIR 模型

覆盖改进 4-6：
  4. 疾病状态波动（M 房室 + ω_reg_m/ω_reg_sub 回退流）
  5. 涂阳/涂阴分层（I_sp/I_sn 不同 μ/r/η）
  6. 内源性再激活 vs 外源性再感染分离（β_exo）

房室结构：[S, L_fast, L_slow, M, I_sub, I_sp, I_sn, R, C]

文献：
  Horton et al. (2023) PNAS 120(47):e2221186120 — 疾病状态波动
  Ragonnet et al. (2021) Clin Infect Dis 73(1):e88-e96 — 涂阳/涂阴自然史
  Vynnycky & Fine (1997) Epidemiol Rev 19(2):183-201 — 外源性再感染
"""

import os
import sys
import unittest

import numpy as np
import pytest

# 确保包路径在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.seir.stochastic import (
    StochasticSEIRModel,
    N_COMPARTMENTS_V3,
    IDX_S_V3, IDX_LF_V3, IDX_LS_V3, IDX_M_V3,
    IDX_ISUB_V3, IDX_ISP_V3, IDX_ISN_V3, IDX_R_V3, IDX_C_V3,
    DEFAULT_RHO_FAST, DEFAULT_RHO_CONV, DEFAULT_RHO_REACT,
    DEFAULT_SIGMA_CLEAR, DEFAULT_RHO_PROG, DEFAULT_ETA_SUB,
    DEFAULT_P_CLIN, DEFAULT_GAMMA, DEFAULT_BETA_REINF,
    DEFAULT_RHO_MIN, DEFAULT_P_M,
    DEFAULT_OMEGA_REG_M, DEFAULT_OMEGA_REG_SUB,
    DEFAULT_P_SP, DEFAULT_MU_SP, DEFAULT_R_SP,
    DEFAULT_MU_SN, DEFAULT_R_SN, DEFAULT_P_SN2SP,
    DEFAULT_ETA_SN, DEFAULT_BETA_EXO,
)
from tb_risk.seir.bayesian import BayesianSEIRInference


def _make_initial_state(population=1000, i0=10):
    """构造 9-房室初始状态 [S, Lf, Ls, M, I_sub, I_sp, I_sn, R, C]"""
    e0 = min(i0 * 30, population - i0)
    lf0 = int(e0 * 0.3)
    ls0 = e0 - lf0
    s0 = max(0, population - i0 - e0)
    # 活动性 TB 在 I_sub/I_sp/I_sn 之间分配
    isub0 = int(i0 * 0.5)
    isp0 = int(i0 * 0.3)
    isn0 = i0 - isub0 - isp0
    return np.array([float(s0), float(lf0), float(ls0), 0.0,
                     float(isub0), float(isp0), float(isn0), 0.0, 0.0])


class TestDeterministicRHSv3(unittest.TestCase):
    """deterministic_rhs_v3 单元测试"""

    def setUp(self):
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.state = _make_initial_state(1000)

    def test_returns_9d_array(self):
        """返回值维度为 9"""
        rhs = self.model.deterministic_rhs_v3(self.state, 0.0, beta=0.3)
        self.assertEqual(len(rhs), N_COMPARTMENTS_V3)
        self.assertIsInstance(rhs, np.ndarray)

    def test_mass_conservation(self):
        """所有房室导数之和应为 0（封闭群体，无出生死亡）"""
        rhs = self.model.deterministic_rhs_v3(self.state, 0.0, beta=0.3)
        # 涂阳/涂阴死亡率 μ_sp, μ_sn 会从 R 中扣除（流出到"死亡"），但本模型
        # 的 R 流出包含 (r_sp+mu_sp)*Isp，将死亡也合并到 R 中。检查 dS+dLf+...+dC ≈ 0
        total = float(np.sum(rhs))
        self.assertAlmostEqual(total, 0.0, places=6,
                               msg=f"质量守恒违反: sum(rhs)={total}")

    def test_no_flow_when_no_infectious(self):
        """无可传播个体时，S 不被感染"""
        state_no_inf = self.state.copy()
        state_no_inf[IDX_ISUB_V3] = 0.0
        state_no_inf[IDX_ISP_V3] = 0.0
        state_no_inf[IDX_ISN_V3] = 0.0
        rhs = self.model.deterministic_rhs_v3(state_no_inf, 0.0, beta=0.3)
        self.assertAlmostEqual(rhs[IDX_S_V3], 0.0, places=10)

    def test_undulation_flows_present(self):
        """改进4：M→L_slow (ω_reg_m) 与 I_sub→M (ω_reg_sub) 回退流存在

        当 M > 0 时，dLs 应包含 +ω_reg_m*M 的贡献；
        当 I_sub > 0 时，dM 应包含 +ω_reg_sub*I_sub 的贡献。
        """
        # 给 M 注入个体，关闭其他所有进展路径以隔离回退流
        state = np.zeros(9)
        state[IDX_M_V3] = 100.0
        # 用极端参数：关闭所有进展，仅保留回退
        rhs = self.model.deterministic_rhs_v3(
            state, 0.0, beta=0.0,
            rho_fast=0.0, rho_conv=0.0, rho_react=0.0,
            sigma_clear=0.0, rho_prog=0.0, rho_min=0.0,
            eta_sub=0.0, p_clin=0.0, gamma=0.0, beta_reinf=0.0,
            p_m=0.5,
            omega_reg_m=0.5,  # M→Ls 回退率
            omega_reg_sub=0.0,
        )
        # dLs 应为 +omega_reg_m * M = 0.5 * 100 = 50
        self.assertAlmostEqual(rhs[IDX_LS_V3], 50.0, places=6)
        # dM 应为 -50
        self.assertAlmostEqual(rhs[IDX_M_V3], -50.0, places=6)

        # 反向：I_sub→M 回退
        state2 = np.zeros(9)
        state2[IDX_ISUB_V3] = 100.0
        rhs2 = self.model.deterministic_rhs_v3(
            state2, 0.0, beta=0.0,
            rho_fast=0.0, rho_conv=0.0, rho_react=0.0,
            sigma_clear=0.0, rho_prog=0.0, rho_min=0.0,
            eta_sub=0.0, p_clin=0.0, gamma=0.0, beta_reinf=0.0,
            omega_reg_m=0.0,
            omega_reg_sub=0.3,  # I_sub→M 回退率
        )
        # dM = +omega_reg_sub * I_sub = 0.3 * 100 = 30
        self.assertAlmostEqual(rhs2[IDX_M_V3], 30.0, places=6)
        # dI_sub = -30
        self.assertAlmostEqual(rhs2[IDX_ISUB_V3], -30.0, places=6)

    def test_smear_stratified_outflow(self):
        """改进5：I_sp 与 I_sn 具有不同的流出率 (μ_sp+r_sp vs μ_sn+r_sn)"""
        # 涂阳流出率 = (0.389 + 0.231)/365 ≈ 0.0017/day
        # 涂阴流出率 = (0.025 + 0.130)/365 ≈ 0.00042/day
        state_sp = np.zeros(9)
        state_sp[IDX_ISP_V3] = 100.0
        rhs_sp = self.model.deterministic_rhs_v3(
            state_sp, 0.0, beta=0.0,
            rho_fast=0.0, rho_conv=0.0, rho_react=0.0,
            sigma_clear=0.0, rho_prog=0.0, rho_min=0.0,
            eta_sub=0.0, p_clin=0.0, gamma=0.0, beta_reinf=0.0,
            p_sn2sp=0.0,
        )
        expected_sp_out = (DEFAULT_MU_SP + DEFAULT_R_SP) * 100.0
        self.assertAlmostEqual(rhs_sp[IDX_ISP_V3], -expected_sp_out, places=8)
        # 流出到 R
        self.assertAlmostEqual(rhs_sp[IDX_R_V3], +expected_sp_out, places=8)

        state_sn = np.zeros(9)
        state_sn[IDX_ISN_V3] = 100.0
        rhs_sn = self.model.deterministic_rhs_v3(
            state_sn, 0.0, beta=0.0,
            rho_fast=0.0, rho_conv=0.0, rho_react=0.0,
            sigma_clear=0.0, rho_prog=0.0, rho_min=0.0,
            eta_sub=0.0, p_clin=0.0, gamma=0.0, beta_reinf=0.0,
            p_sn2sp=0.0,
        )
        expected_sn_out = (DEFAULT_MU_SN + DEFAULT_R_SN) * 100.0
        self.assertAlmostEqual(rhs_sn[IDX_ISN_V3], -expected_sn_out, places=8)
        # 涂阳流出率应高于涂阴
        self.assertGreater(expected_sp_out, expected_sn_out)

    def test_exogenous_reinfection_flow(self):
        """改进6：β_exo·λ·L_slow 外源性再感染流（L_slow→L_fast）"""
        # 设置 Ls 和感染源 I_sp，关闭其他路径
        state = np.zeros(9)
        state[IDX_LS_V3] = 100.0
        state[IDX_ISP_V3] = 10.0  # 提供 λ
        rhs = self.model.deterministic_rhs_v3(
            state, 0.0, beta=0.3,
            rho_fast=0.0, rho_conv=0.0, rho_react=0.0,
            sigma_clear=0.0, rho_prog=0.0, rho_min=0.0,
            eta_sub=0.0, p_clin=0.0, gamma=0.0, beta_reinf=0.0,
            p_sn2sp=0.0,
            eta_sn=0.0,
            beta_exo=0.5,
        )
        # λ = 0.3 * (0*0 + 10 + 0*0) / 1000 = 0.003
        lam = 0.3 * 10.0 / 1000.0
        expected_exo = 0.5 * lam * 100.0  # β_exo·λ·Ls
        # dLf 应包含 +exo_reinfection
        self.assertAlmostEqual(rhs[IDX_LF_V3], +expected_exo, places=8)
        # dLs 应包含 -exo_reinfection
        self.assertAlmostEqual(rhs[IDX_LS_V3], -expected_exo, places=8)


class TestSimulateSDEv3(unittest.TestCase):
    """simulate_sde_v3 单元测试"""

    def setUp(self):
        self.model = StochasticSEIRModel(population=1000, noise_scale=0.05, seed=42)
        self.state = _make_initial_state(1000)

    def test_trajectory_shape(self):
        """轨迹形状为 (n_steps, 9)"""
        times, traj, _ = self.model.simulate_sde_v3(
            self.state, (0, 30), 0.1, beta=0.3)
        self.assertEqual(traj.shape[1], N_COMPARTMENTS_V3)
        self.assertEqual(len(times), traj.shape[0])
        self.assertGreater(traj.shape[0], 1)

    def test_non_negative(self):
        """所有房室值非负"""
        _, traj, _ = self.model.simulate_sde_v3(
            self.state, (0, 30), 0.1, beta=0.3)
        self.assertTrue(np.all(traj >= -1e-9),
                        f"出现负值: min={traj.min()}")

    def test_stochastic_forcing_shape(self):
        """随机强迫项形状为 (n_steps, 21) — 要点A+F: 16→21 流"""
        _, _, sto = self.model.simulate_sde_v3(
            self.state, (0, 10), 0.1, beta=0.3)
        self.assertEqual(sto.shape[1], 21)
        self.assertEqual(sto.shape[0], 10 * 10 + 1)  # n_steps = 10/0.1 + 1

    def test_undulation_can_regress(self):
        """改进4：M 房室可被回退流消耗（M→L_slow）"""
        # 设置初始 M=50，关闭所有进展，仅保留回退
        state = np.zeros(9)
        state[IDX_M_V3] = 50.0
        # 把人口基数设大，避免 N=0
        state[IDX_S_V3] = 950.0
        _, traj, _ = self.model.simulate_sde_v3(
            state, (0, 100), 1.0, beta=0.0,
            rho_fast=0.0, rho_conv=0.0, rho_react=0.0,
            sigma_clear=0.0, rho_prog=0.0, rho_min=0.0,
            eta_sub=0.0, p_clin=0.0, gamma=0.0, beta_reinf=0.0,
            omega_reg_m=0.1,  # 显著回退率
            omega_reg_sub=0.0,
        )
        # M 应单调减少（无进展补偿）
        self.assertLess(traj[-1, IDX_M_V3], traj[0, IDX_M_V3])
        # Ls 应增加
        self.assertGreater(traj[-1, IDX_LS_V3], traj[0, IDX_LS_V3])


class TestSimulateCTMCv3(unittest.TestCase):
    """simulate_ctmc_v3 单元测试"""

    def setUp(self):
        self.model = StochasticSEIRModel(population=1000, seed=42)
        # CTMC 需要整数初始状态
        self.state = _make_initial_state(1000)
        self.state = np.array([int(round(s)) for s in self.state])

    def test_returns_integer_states(self):
        """CTMC 状态为整数"""
        times, states = self.model.simulate_ctmc_v3(
            self.state, max_time=10.0, beta=0.3)
        self.assertEqual(states.shape[1], N_COMPARTMENTS_V3)
        # 所有状态应为非负整数
        self.assertTrue(np.all(states >= 0))
        # 整数检查（CTMC 离散事件）
        np.testing.assert_array_equal(states, np.round(states))

    def test_total_population_conserved(self):
        """总群体数守恒（无出生死亡）"""
        initial_total = int(np.sum(self.state))
        times, states = self.model.simulate_ctmc_v3(
            self.state, max_time=10.0, beta=0.3)
        totals = states.sum(axis=1)
        # 所有时间点的总数应等于初始总数
        np.testing.assert_array_equal(totals, initial_total)

    def test_event_count_at_least_one(self):
        """至少触发一个事件（初始有感染源）"""
        times, states = self.model.simulate_ctmc_v3(
            self.state, max_time=50.0, beta=0.5)
        self.assertGreater(len(times), 1)


class TestSimulateMultiplev3(unittest.TestCase):
    """simulate_multiple_v3 单元测试"""

    def test_returns_multiple_trajectories(self):
        """返回多条活动性 TB (I_sp+I_sn) 轨迹

        返回形状: (n_trajectories, n_steps) — 每条轨迹是活动性临床 TB 计数
        """
        model = StochasticSEIRModel(population=1000, seed=42)
        state = _make_initial_state(1000)
        trajs = model.simulate_multiple_v3(
            state, (0, 30), 1.0, beta=0.3, n_trajectories=5)
        # 形状: (n_trajectories, n_steps)
        self.assertEqual(trajs.shape[0], 5)
        self.assertGreater(trajs.shape[1], 1)
        # 活动性 TB 应非负
        self.assertTrue(np.all(trajs >= -1e-9))


class TestBayesianV3Priors(unittest.TestCase):
    """log_prior_v3 单元测试"""

    def setUp(self):
        self.inf = BayesianSEIRInference(seed=42)

    def test_valid_params_finite(self):
        """有效参数下先验为有限值"""
        lp = self.inf.log_prior_v3(
            beta=0.2,
            rho_fast=0.1 / 365.0,
            rho_react=0.004 / 365.0,
            sigma_clear=2.0 / 365.0,
            gamma=0.05,
            omega_reg_m=1.0 / 365.0,
            omega_reg_sub=1.0 / 365.0,
            beta_exo=0.33,
        )
        self.assertTrue(np.isfinite(lp), f"先验应为有限值: lp={lp}")
        self.assertGreater(lp, -np.inf)

    def test_invalid_beta_returns_neg_inf(self):
        """β 超出支撑域返回 -inf"""
        lp = self.inf.log_prior_v3(
            beta=2.0,  # 超过上界 0.95
            rho_fast=0.1 / 365.0, rho_react=0.004 / 365.0,
            sigma_clear=2.0 / 365.0, gamma=0.05,
        )
        self.assertEqual(lp, -np.inf)

    def test_invalid_omega_reg_returns_neg_inf(self):
        """ω_reg_m 超出支撑域返回 -inf"""
        lp = self.inf.log_prior_v3(
            beta=0.2,
            rho_fast=0.1 / 365.0, rho_react=0.004 / 365.0,
            sigma_clear=2.0 / 365.0, gamma=0.05,
            omega_reg_m=10.0,  # 超过上界 0.05
        )
        self.assertEqual(lp, -np.inf)

    def test_invalid_beta_exo_returns_neg_inf(self):
        """β_exo 超出支撑域返回 -inf"""
        lp = self.inf.log_prior_v3(
            beta=0.2,
            rho_fast=0.1 / 365.0, rho_react=0.004 / 365.0,
            sigma_clear=2.0 / 365.0, gamma=0.05,
            beta_exo=0.0,  # 低于下界 0.01
        )
        self.assertEqual(lp, -np.inf)

    def test_eta_sn_optional(self):
        """eta_sn 可选：传入有效值时先验增加一项"""
        lp_no_eta = self.inf.log_prior_v3(
            beta=0.2, rho_fast=0.1/365.0, rho_react=0.004/365.0,
            sigma_clear=2.0/365.0, gamma=0.05,
        )
        lp_with_eta = self.inf.log_prior_v3(
            beta=0.2, rho_fast=0.1/365.0, rho_react=0.004/365.0,
            sigma_clear=2.0/365.0, gamma=0.05,
            eta_sn=0.35,
        )
        # 添加 eta_sn 项后先验值应不同（增加一个 Beta 项）
        self.assertNotAlmostEqual(lp_no_eta, lp_with_eta, places=6)
        self.assertTrue(np.isfinite(lp_with_eta))


class TestBayesianV3Likelihood(unittest.TestCase):
    """log_likelihood_v3 与 log_posterior_v3 单元测试"""

    def setUp(self):
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.inf = BayesianSEIRInference(seir_model=self.model, seed=42)
        self.init_state = _make_initial_state(1000)
        self.observed_data = (
            np.array([5, 8, 12, 15, 20, 25]),  # obs_I
            np.array([5, 10, 15, 20, 25, 30]),  # obs_times
        )

    def test_likelihood_finite(self):
        """似然值为有限浮点数"""
        params = (0.3, 0.1/365.0, 0.004/365.0, 2.0/365.0, 0.05,
                  1.0/365.0, 1.0/365.0, 0.33)
        ll = self.inf.log_likelihood_v3(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        self.assertTrue(np.isfinite(ll), f"似然应为有限值: ll={ll}")

    def test_likelihood_deterministic(self):
        """同一参数集两次调用应返回相同似然值（detailed balance 保护）"""
        params = (0.3, 0.1/365.0, 0.004/365.0, 2.0/365.0, 0.05,
                  1.0/365.0, 1.0/365.0, 0.33)
        ll1 = self.inf.log_likelihood_v3(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        ll2 = self.inf.log_likelihood_v3(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        self.assertEqual(ll1, ll2)

    def test_posterior_finite(self):
        """后验值为有限浮点数"""
        params = (0.3, 0.1/365.0, 0.004/365.0, 2.0/365.0, 0.05,
                  1.0/365.0, 1.0/365.0, 0.33)
        lp = self.inf.log_posterior_v3(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        self.assertTrue(np.isfinite(lp), f"后验应为有限值: lp={lp}")

    def test_posterior_with_eta_sn(self):
        """包含 eta_sn 的后验路径"""
        params = (0.3, 0.1/365.0, 0.004/365.0, 2.0/365.0, 0.05,
                  1.0/365.0, 1.0/365.0, 0.33, 0.35)
        lp = self.inf.log_posterior_v3(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0, include_eta_sn=True)
        self.assertTrue(np.isfinite(lp), f"含 eta_sn 后验应为有限值: lp={lp}")

    def test_invalid_params_posterior_neg_inf(self):
        """无效参数下后验为 -inf"""
        params = (2.0,  # β 超界
                  0.1/365.0, 0.004/365.0, 2.0/365.0, 0.05,
                  1.0/365.0, 1.0/365.0, 0.33)
        lp = self.inf.log_posterior_v3(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        self.assertEqual(lp, -np.inf)


class TestPosteriorNestedFactors(unittest.TestCase):
    """posterior.py v3.0 嵌套结构（涂阳/涂阴分层）测试"""

    def test_factors_nested_structure(self):
        """factors 返回 {smear_positive, smear_negative} 嵌套字典"""
        from tb_risk.seir.posterior import PosteriorDrivenInfectivity
        pdi = PosteriorDrivenInfectivity(seir_inference=None)
        factors = pdi.factors
        self.assertIn('smear_positive', factors)
        self.assertIn('smear_negative', factors)
        for key in ('smear_positive', 'smear_negative'):
            sub = factors[key]
            for stage in ('pre_treatment', 'early_treatment', 'mid_treatment',
                          'late_treatment', 'completed_treatment'):
                self.assertIn(stage, sub)
                self.assertGreaterEqual(sub[stage], 0.0)
                self.assertLessEqual(sub[stage], 1.0)

    def test_smear_positive_anchor(self):
        """涂阳 pre_treatment 恒为 1.0 (η_sp 锚点)"""
        from tb_risk.seir.posterior import PosteriorDrivenInfectivity
        pdi = PosteriorDrivenInfectivity(seir_inference=None)
        self.assertEqual(pdi.factors['smear_positive']['pre_treatment'], 1.0)

    def test_smear_negative_baseline(self):
        """涂阴 pre_treatment = η_sn ≈ 0.35（基线较低）"""
        from tb_risk.seir.posterior import PosteriorDrivenInfectivity
        pdi = PosteriorDrivenInfectivity(seir_inference=None)
        sn = pdi.factors['smear_negative']
        self.assertAlmostEqual(sn['pre_treatment'], 0.35, places=2)
        # 涂阴基线应低于涂阳
        sp = pdi.factors['smear_positive']
        self.assertLess(sn['pre_treatment'], sp['pre_treatment'])

    def test_smear_positive_early_decays_faster(self):
        """涂阳 early_treatment 衰减比涂阳更激进

        文献：涂阳治疗 2-3 周内涂片转阴 → early 阶段衰减更快
        涂阳 early=0.55 < 涂阴 early=0.85（相对各自的 pre_treatment）
        """
        from tb_risk.seir.posterior import PosteriorDrivenInfectivity
        pdi = PosteriorDrivenInfectivity(seir_inference=None)
        sp = pdi.factors['smear_positive']
        sn = pdi.factors['smear_negative']
        # 相对衰减比例：early/pre
        sp_decay = sp['early_treatment'] / sp['pre_treatment']
        sn_decay = sn['early_treatment'] / sn['pre_treatment']
        self.assertLess(sp_decay, sn_decay,
                        "涂阳 early 阶段相对衰减应比涂阴更激进")

    def test_posterior_eta_sn_overrides_default(self):
        """后验 eta_sn 覆盖默认 0.35"""
        from tb_risk.seir.posterior import PosteriorDrivenInfectivity

        class MockInference:
            pass

        mock = MockInference()
        mock.posterior_samples = {
            'beta': [0.3],
            'gamma': [0.05],
            'eta_sn': [0.5],  # 显著高于默认 0.35
        }
        pdi = PosteriorDrivenInfectivity(seir_inference=mock)
        sn = pdi.factors['smear_negative']
        # pre_treatment 应取后验均值 0.5
        self.assertAlmostEqual(sn['pre_treatment'], 0.5, places=4)

    def test_update_from_mcmc_resets_cache(self):
        """update_from_mcmc 重置缓存"""
        from tb_risk.seir.posterior import PosteriorDrivenInfectivity
        pdi = PosteriorDrivenInfectivity(seir_inference=None)
        _ = pdi.factors
        self.assertIsNone(pdi._posterior_factors)
        pdi.update_from_mcmc(seir_inference=None)
        self.assertIsNone(pdi._posterior_factors)


class TestNegBinomLikelihoodAndEtaSub(unittest.TestCase):
    """改进5 (负二项似然) + 改进9 (eta_sub 推断) 单元测试"""

    def setUp(self):
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.inf = BayesianSEIRInference(seir_model=self.model, seed=42)
        self.init_state = _make_initial_state(1000)
        self.observed_data = (
            np.array([5, 8, 12, 15, 20, 25]),
            np.array([5, 10, 15, 20, 25, 30]),
        )
        self.base_params = (
            0.3, 0.1/365.0, 0.004/365.0, 2.0/365.0, 0.05,
            1.0/365.0, 1.0/365.0, 0.33,
        )

    # ---------- 改进5: 负二项似然 ----------

    def test_neg_binom_likelihood_finite(self):
        """负二项似然（默认 use_neg_binom=True）为有限值"""
        ll = self.inf.log_likelihood_v3(
            self.base_params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        self.assertTrue(np.isfinite(ll), f"负二项似然应为有限值: ll={ll}")

    def test_neg_binom_likelihood_deterministic(self):
        """负二项似然满足 detailed balance（确定性）"""
        ll1 = self.inf.log_likelihood_v3(
            self.base_params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        ll2 = self.inf.log_likelihood_v3(
            self.base_params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        self.assertEqual(ll1, ll2)

    def test_neg_binom_differs_from_gaussian(self):
        """负二项似然与高斯似然值不同"""
        ll_nb = self.inf.log_likelihood_v3(
            self.base_params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0, use_neg_binom=True)
        ll_gauss = self.inf.log_likelihood_v3(
            self.base_params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0, use_neg_binom=False)
        self.assertTrue(np.isfinite(ll_nb))
        self.assertTrue(np.isfinite(ll_gauss))
        self.assertNotAlmostEqual(ll_nb, ll_gauss, places=3)

    def test_neg_binom_k_parameter_affects_value(self):
        """过度离散参数 k 影响似然值"""
        ll_k_small = self.inf.log_likelihood_v3(
            self.base_params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0, neg_binom_k=0.1)
        ll_k_large = self.inf.log_likelihood_v3(
            self.base_params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0, neg_binom_k=10.0)
        self.assertNotAlmostEqual(ll_k_small, ll_k_large, places=3)

    def test_neg_binom_helper_zero_sim_protection(self):
        """sim=0 时数值保护（加 1e-6）不产生 NaN/inf"""
        ll = self.inf._neg_binom_loglik(
            np.array([0, 5, 10]), np.array([0.0, 0.0, 5.0]), k=0.2)
        self.assertTrue(np.isfinite(ll))

    # ---------- 改进9: eta_sub 推断 ----------

    def test_eta_sub_prior_in_range_finite(self):
        """eta_sub 在 [0.1, 3.0] 内先验有限"""
        lp = self.inf.log_prior_v2(
            0.2, 0.1/365.0, 0.004/365.0, 2.0/365.0, 0.05,
            eta_sub=1.5)
        self.assertTrue(np.isfinite(lp))

    def test_eta_sub_prior_out_of_range_neg_inf(self):
        """eta_sub 超出 [0.1, 3.0] 返回 -inf"""
        lp_low = self.inf.log_prior_v2(
            0.2, 0.1/365.0, 0.004/365.0, 2.0/365.0, 0.05,
            eta_sub=0.05)  # 低于下界 0.1
        self.assertEqual(lp_low, -np.inf)
        lp_high = self.inf.log_prior_v2(
            0.2, 0.1/365.0, 0.004/365.0, 2.0/365.0, 0.05,
            eta_sub=3.5)  # 高于上界 3.0
        self.assertEqual(lp_high, -np.inf)

    def test_eta_sub_posterior_finite(self):
        """include_eta_sub=True 时 9 维后验有限"""
        params = self.base_params + (1.5,)  # eta_sub=1.5
        lp = self.inf.log_posterior_v3(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0, include_eta_sub=True)
        self.assertTrue(np.isfinite(lp), f"含 eta_sub 后验应为有限值: lp={lp}")

    def test_eta_sub_likelihood_finite(self):
        """include_eta_sub=True 时似然有限"""
        params = self.base_params + (1.5,)
        ll = self.inf.log_likelihood_v3(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0, include_eta_sub=True)
        self.assertTrue(np.isfinite(ll))

    def test_eta_sub_and_eta_sn_combined_finite(self):
        """include_eta_sub + include_eta_sn 同时启用 (10 维) 后验有限"""
        params = self.base_params + (1.5, 0.35)  # eta_sub, eta_sn
        lp = self.inf.log_posterior_v3(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0,
            include_eta_sub=True, include_eta_sn=True)
        self.assertTrue(np.isfinite(lp))

    def test_v3_9comp_param_config(self):
        """V3_9COMP 参数配置维度正确"""
        self.assertEqual(
            len(self.inf.V3_9COMP_PARAM_NAMES_WITH_ETA_SUB), 9)
        self.assertEqual(
            len(self.inf.V3_9COMP_INIT_PARAMS_WITH_ETA_SUB), 9)
        self.assertEqual(
            len(self.inf.V3_9COMP_LOWER_BOUNDS_WITH_ETA_SUB), 9)
        self.assertEqual(
            len(self.inf.V3_9COMP_UPPER_BOUNDS_WITH_ETA_SUB), 9)
        # eta_sub 下界/上界为 0.1 / 3.0
        self.assertAlmostEqual(
            self.inf.V3_9COMP_LOWER_BOUNDS_WITH_ETA_SUB[-1], 0.1)
        self.assertAlmostEqual(
            self.inf.V3_9COMP_UPPER_BOUNDS_WITH_ETA_SUB[-1], 3.0)


class TestRunMCMCv3(unittest.TestCase):
    """改进4: run_mcmc_v3 自适应 Metropolis-Hastings 采样器测试"""

    def setUp(self):
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.inf = BayesianSEIRInference(seir_model=self.model, seed=42)
        self.init_state = _make_initial_state(1000)
        self.observed_data = (
            np.array([5, 8, 12, 15, 20, 25]),
            np.array([5, 10, 15, 20, 25, 30]),
        )

    def test_run_mcmc_v3_base_8dim(self):
        """8 维基础参数空间 MCMC 可运行"""
        samples = self.inf.run_mcmc_v3(
            self.observed_data, self.init_state,
            t_span=(0, 30), dt=2.0,
            n_iterations=30, burn_in=10, n_chains=2)
        # 应返回 8 个参数的后验样本
        self.assertEqual(len(self.inf.V3_9COMP_PARAM_NAMES), 8)
        for name in self.inf.V3_9COMP_PARAM_NAMES:
            self.assertIn(name, samples)
            self.assertEqual(samples[name].shape[0], 2 * 20)  # 2链 × 20后验
        # 诊断信息
        diag = self.inf.convergence_diagnostics
        self.assertEqual(diag['n_params'], 8)
        self.assertIn('acceptance_rate', diag)
        self.assertIn('r_hat', diag)

    def test_run_mcmc_v3_with_eta_sub_9dim(self):
        """9 维 (含 eta_sub) 参数空间 MCMC 可运行"""
        samples = self.inf.run_mcmc_v3(
            self.observed_data, self.init_state,
            t_span=(0, 30), dt=2.0,
            n_iterations=30, burn_in=10, n_chains=2,
            include_eta_sub=True)
        self.assertIn('eta_sub', samples)
        self.assertEqual(samples['eta_sub'].shape[0], 2 * 20)
        diag = self.inf.convergence_diagnostics
        self.assertEqual(diag['n_params'], 9)
        self.assertTrue(diag['include_eta_sub'])

    def test_run_mcmc_v3_acceptance_rate_nonneg(self):
        """接受率非负"""
        self.inf.run_mcmc_v3(
            self.observed_data, self.init_state,
            t_span=(0, 30), dt=2.0,
            n_iterations=30, burn_in=10, n_chains=2)
        self.assertGreaterEqual(
            self.inf.convergence_diagnostics['acceptance_rate'], 0.0)

    def test_run_mcmc_v3_gaussian_likelihood(self):
        """use_neg_binom=False 时使用高斯似然"""
        self.inf.run_mcmc_v3(
            self.observed_data, self.init_state,
            t_span=(0, 30), dt=2.0,
            n_iterations=30, burn_in=10, n_chains=2,
            use_neg_binom=False)
        self.assertEqual(
            self.inf.convergence_diagnostics['likelihood'], 'gaussian')


class TestComputeR0v3(unittest.TestCase):
    """改进6: compute_R0_v3 基本再生数计算测试"""

    def setUp(self):
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.inf = BayesianSEIRInference(seir_model=self.model, seed=42)

    def test_r0_positive(self):
        """R0 为正值"""
        r0 = self.inf.compute_R0_v3(beta=0.01)
        self.assertGreater(r0, 0.0)

    def test_r0_scales_with_beta(self):
        """R0 与 β 成正比"""
        r0_a = self.inf.compute_R0_v3(beta=0.01)
        r0_b = self.inf.compute_R0_v3(beta=0.02)
        self.assertAlmostEqual(r0_b / r0_a, 2.0, places=5)

    def test_r0_eta_sub_increases_r0(self):
        """eta_sub 增大 → R0 增大"""
        r0_low = self.inf.compute_R0_v3(beta=0.01, eta_sub=0.5)
        r0_high = self.inf.compute_R0_v3(beta=0.01, eta_sub=2.0)
        self.assertGreater(r0_high, r0_low)

    def test_r0_formula_known_values(self):
        """已知参数下 R0 与公式手算值一致"""
        # r_sp+mu_sp = (0.231+0.389)/365 = 0.62/365
        # r_sn+mu_sn+p_sn2sp = (0.130+0.025+0.1)/365 = 0.255/365
        # rho_prog = 0.05
        beta = 0.01
        eta_sub, eta_sn = 1.0, 0.35
        expected = beta * (eta_sub / 0.05
                           + 1.0 / (0.62 / 365.0)
                           + eta_sn / (0.255 / 365.0))
        r0 = self.inf.compute_R0_v3(beta=beta, eta_sub=eta_sub, eta_sn=eta_sn)
        self.assertAlmostEqual(r0, expected, places=5)

    def test_r0_eta_sp_anchor(self):
        """eta_sp 默认为 1.0（涂阳锚点）"""
        r0_default = self.inf.compute_R0_v3(beta=0.01)
        r0_sp1 = self.inf.compute_R0_v3(beta=0.01, eta_sp=1.0)
        self.assertAlmostEqual(r0_default, r0_sp1, places=8)

    def test_r0_zero_beta(self):
        """β=0 时 R0=0"""
        r0 = self.inf.compute_R0_v3(beta=0.0)
        self.assertEqual(r0, 0.0)

    def test_r0_division_by_zero_protection(self):
        """流出率为 0 时不除零"""
        r0 = self.inf.compute_R0_v3(
            beta=0.01, rho_prog=0.0, r_sp=0.0, mu_sp=0.0,
            r_sn=0.0, mu_sn=0.0, p_sn2sp=0.0)
        self.assertTrue(np.isfinite(r0))
        self.assertGreater(r0, 0.0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestPosteriorPredictiveV3(unittest.TestCase):
    """改进6: generate_posterior_predictive_v3 后验预测分布测试

    要点6 (第十四轮): 暂未迁移到 BayesianInferenceV3, 因为
    generate_posterior_predictive_v3 方法当前只存在于
    seir/uncertainty/_posterior_v3.py, seir/inference/v3_inference.py
    未提供等价方法。完整迁移需先在 BayesianInferenceV3 中实现
    generate_posterior_predictive_v3 (或调用 uncertainty/ 的 Mixin),
    超出本轮范围。此处用 @pytest.mark.filterwarnings 抑制弃用警告,
    作为技术债务保留, 待后续轮次统一处理 uncertainty/ → inference/
    的方法迁移后再批量迁移。
    """

    def setUp(self):
        from tb_risk.seir.uncertainty import SEIRParameterUncertainty
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.inf = BayesianSEIRInference(seir_model=self.model, seed=42)
        self.unc = SEIRParameterUncertainty(
            stochastic_seir_model=self.model, population=1000, random_seed=42)
        self.init_state = _make_initial_state(1000)
        self.observed_data = (
            np.array([5, 8, 12, 15, 20, 25]),
            np.array([5, 10, 15, 20, 25, 30]),
        )

    def _run_mcmc_and_predict(self, use_sde=False, n_samples=5,
                              include_eta_sub=True):
        """辅助：运行 MCMC 后生成后验预测"""
        self.inf.run_mcmc_v3(
            self.observed_data, self.init_state,
            t_span=(0, 20), dt=2.0,
            n_iterations=30, burn_in=10, n_chains=2,
            include_eta_sub=include_eta_sub)
        self.unc.posterior_samples = self.inf.posterior_samples
        return self.unc.generate_posterior_predictive_v3(
            self.init_state, t_span=(0, 20), dt=2.0,
            n_samples=n_samples, use_sde=use_sde)

    def test_predict_error_no_posterior(self):
        """无后验样本时返回 error"""
        result = self.unc.generate_posterior_predictive_v3(
            self.init_state, t_span=(0, 20), dt=2.0)
        self.assertIn('error', result)

    def test_predict_ode_mode(self):
        """ODE 模式后验预测可运行"""
        result = self._run_mcmc_and_predict(use_sde=False)
        self.assertNotIn('error', result, msg=str(result))
        self.assertIn('trajectories', result)
        self.assertIn('ci_bands', result)
        self.assertIn('r0_samples', result)
        self.assertEqual(result['method'], 'ODE')

    def test_predict_sde_mode(self):
        """SDE 模式后验预测可运行"""
        result = self._run_mcmc_and_predict(use_sde=True)
        self.assertNotIn('error', result, msg=str(result))
        self.assertEqual(result['method'], 'SDE')

    def test_predict_ci_bands_structure(self):
        """置信带包含 lower/upper/median/time"""
        result = self._run_mcmc_and_predict(use_sde=False)
        ci = result['ci_bands']
        for key in ('lower', 'upper', 'median', 'time'):
            self.assertIn(key, ci)
            self.assertGreater(len(ci[key]), 0)
        # lower <= median <= upper
        for lo, med, hi in zip(ci['lower'], ci['median'], ci['upper']):
            self.assertLessEqual(lo, med + 1e-6)
            self.assertLessEqual(med, hi + 1e-6)

    def test_predict_r0_samples_nonempty(self):
        """r0_samples 非空且为正"""
        result = self._run_mcmc_and_predict(use_sde=False)
        self.assertGreater(len(result['r0_samples']), 0)
        for r0 in result['r0_samples']:
            self.assertGreater(r0, 0.0)
        self.assertGreater(result['r0_mean'], 0.0)

    def test_predict_r0_ci_ordered(self):
        """R0 置信区间下界 <= 均值 <= 上界"""
        result = self._run_mcmc_and_predict(use_sde=False)
        lo, hi = result['r0_ci']
        self.assertLessEqual(lo, result['r0_mean'] + 1e-6)
        self.assertGreaterEqual(hi, result['r0_mean'] - 1e-6)

    def test_predict_p_r0_gt_1_in_range(self):
        """P(R0>1) 在 [0, 1]"""
        result = self._run_mcmc_and_predict(use_sde=False)
        p = result['p_r0_gt_1']
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_predict_peak_I_nonempty(self):
        """peak_I_samples 非空且非负"""
        result = self._run_mcmc_and_predict(use_sde=False)
        self.assertGreater(len(result['peak_I_samples']), 0)
        for peak in result['peak_I_samples']:
            self.assertGreaterEqual(peak, 0.0)

    def test_predict_trajectories_shape(self):
        """轨迹形状为 (n_samples, n_steps, 9)"""
        result = self._run_mcmc_and_predict(use_sde=False, n_samples=5)
        traj = result['trajectories']
        self.assertEqual(traj.ndim, 3)
        self.assertEqual(traj.shape[0], result['n_samples'])
        self.assertEqual(traj.shape[2], 9)

    def test_predict_eta_sn_default(self):
        """后验无 eta_sn 时使用默认值 0.35"""
        result = self._run_mcmc_and_predict(
            use_sde=False, include_eta_sub=False)
        self.assertNotIn('error', result, msg=str(result))


if __name__ == '__main__':
    unittest.main()
