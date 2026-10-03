#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — v4.0 年龄×HIV×耐药分层 SEIR 模型

覆盖改进 7-9：
  7. 年龄分层（5 组: 0-4, 5-14, 15-49, 50-64, 65+; WAIFW 混合矩阵）
  8. HIV 共感染分层（3 层: HIV-, HIV+未治疗, HIV+ART; RR_HIV≈20）
  9. 耐药 TB 分层（2 层: DS, DR; 获得性耐药 p_acq; 适合度代价 f_cost）

状态向量: (5 age, 3 HIV, 2 DR, 9 comp) = 270D

文献：
  Marais et al. (2011) IJTLD 15(11):1478-1485 — 儿童TB进展风险
  Davies et al. (2006) JRSM 99(6):298-303 — 老年再激活风险
  Mossong et al. (2008) PLoS Med 5(3):e74 — POLYMOD 社会接触矩阵
  Pawlowski et al. (2012) PLoS Pathog 8(2):e1002464 — HIV/TB 共感染
  Houben et al. (2016) BMC Med 14:56 — TIME Impact HIV/ART 分层
  Karmakar et al. (2022) BMC Infect Dis 22(1):82 — 耐药TB扩增
  Houben & Dodd (2016) PLoS Med — LTBI 耐药株比例
"""

import os
import sys
import unittest

import numpy as np

# 确保包路径在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.seir.stochastic import (
    StochasticSEIRModel,
    N_AGE_V4, N_HIV_V4, N_DR_V4, STATE_SIZE_V4, N_COMPARTMENTS_V3,
    IDX_S_V3, IDX_LF_V3, IDX_LS_V3, IDX_M_V3,
    IDX_ISUB_V3, IDX_ISP_V3, IDX_ISN_V3, IDX_R_V3, IDX_C_V3,
    DEFAULT_RHO_FAST, DEFAULT_RHO_CONV, DEFAULT_RHO_REACT,
    DEFAULT_SIGMA_CLEAR, DEFAULT_RHO_PROG, DEFAULT_ETA_SUB,
    DEFAULT_P_CLIN, DEFAULT_GAMMA, DEFAULT_BETA_REINF,
    DEFAULT_OMEGA_REG_M, DEFAULT_OMEGA_REG_SUB, DEFAULT_ETA_SN,
    DEFAULT_BETA_EXO, DEFAULT_RHO_MIN, DEFAULT_P_M,
    DEFAULT_WAIFW, DEFAULT_AGE_PROGRESSION,
    DEFAULT_RR_HIV, DEFAULT_ART_REDUCTION,
    DEFAULT_HIV_INFECTION_RATE, DEFAULT_ART_INITIATION_RATE,
    DEFAULT_DR_FITNESS_COST, DEFAULT_P_ACQ,
)
from tb_risk.seir.bayesian import BayesianSEIRInference


def _make_v4_initial_state(population=1000, i0=10):
    """构造 270D 初始状态: (5 age, 3 HIV, 2 DR, 9 comp)

    群体主要分布在 15-49 岁（age=2）与 HIV-/DS 层，
    少量初始感染放在 I_sub/I_sp/I_sn 以提供传播源。
    """
    state = np.zeros((N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3))
    # 年龄分布（粗略，参考中国人口结构）
    age_dist = np.array([0.06, 0.12, 0.55, 0.18, 0.09])
    # 初始感染分配到 15-49 岁 HIV- DS 层
    n_inf = i0
    isub0 = int(n_inf * 0.5)
    isp0 = int(n_inf * 0.3)
    isn0 = max(0, n_inf - isub0 - isp0)
    # 潜伏感染（近期）分配
    lf_total = min(i0 * 30, population - i0)
    for a in range(N_AGE_V4):
        n_a = int(population * age_dist[a])
        # 95% 易感者，5% 近期潜伏
        s_a = int(n_a * 0.95)
        lf_a = n_a - s_a
        # 全部归 HIV-/DS
        state[a, 0, 0, IDX_S_V3] = float(s_a)
        state[a, 0, 0, IDX_LF_V3] = float(lf_a)
    # 在 15-49 岁 HIV-/DS 层注入活动性 TB
    state[2, 0, 0, IDX_ISUB_V3] = float(isub0)
    state[2, 0, 0, IDX_ISP_V3] = float(isp0)
    state[2, 0, 0, IDX_ISN_V3] = float(isn0)
    # 从 S 中扣除感染个体
    state[2, 0, 0, IDX_S_V3] = max(0.0, state[2, 0, 0, IDX_S_V3] - n_inf)
    return state.flatten()


def _make_v4_observed_data():
    """构造观测数据 (obs_I, obs_times)"""
    obs_I = np.array([5, 8, 12, 15, 20, 25])
    obs_times = np.array([5, 10, 15, 20, 25, 30])
    return (obs_I, obs_times)


class TestDeterministicRHSv4(unittest.TestCase):
    """deterministic_rhs_v4 单元测试"""

    def setUp(self):
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.state = _make_v4_initial_state(1000)

    def test_returns_270d_array(self):
        """返回值维度为 270"""
        rhs = self.model.deterministic_rhs_v4(self.state, 0.0, beta=0.3)
        self.assertEqual(len(rhs), STATE_SIZE_V4)
        self.assertIsInstance(rhs, np.ndarray)

    def test_mass_conservation(self):
        """所有房室导数之和应接近 0（封闭群体）

        注：HIV 进展（HIV- → HIV+）在 HIV 维度内部转移，总质量守恒；
        涂阳/涂阴死亡率 μ 流出到 R（合并死亡与恢复），仍守恒。
        """
        rhs = self.model.deterministic_rhs_v4(self.state, 0.0, beta=0.3)
        total = float(np.sum(rhs))
        # 相对于初始群体 1000，相对误差应 < 1e-6
        self.assertAlmostEqual(total, 0.0, places=3,
                               msg=f"质量守恒违反: sum(rhs)={total}")

    def test_no_flow_when_no_infectious(self):
        """无可传播个体时，S 不被感染（排除 HIV 进展流）"""
        state_no_inf = self.state.copy().reshape(
            N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
        state_no_inf[:, :, :, IDX_ISUB_V3] = 0.0
        state_no_inf[:, :, :, IDX_ISP_V3] = 0.0
        state_no_inf[:, :, :, IDX_ISN_V3] = 0.0
        rhs = self.model.deterministic_rhs_v4(
            state_no_inf.flatten(), 0.0, beta=0.3,
            hiv_infection_rate=0.0,  # 关闭 HIV 进展以隔离感染流
        )
        rhs_4d = rhs.reshape(N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
        # 所有年龄/HIV/DR 的 S 导数应 ≈ 0
        s_deriv = rhs_4d[:, :, :, IDX_S_V3]
        self.assertTrue(np.allclose(s_deriv, 0.0, atol=1e-10),
                        f"S 导数非零: max|dS|={np.max(np.abs(s_deriv))}")

    def test_age_specific_force_of_infection(self):
        """改进7：WAIFW 矩阵使不同年龄组感染力不同

        当所有感染源集中在 15-49 岁（age=2）时，
        15-49 岁易感者（对角元素 12.0）的感染力应高于 0-4 岁（元素 1.5）。
        """
        state = np.zeros((N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3))
        # 所有年龄组都有 100 易感者（HIV-/DS）
        for a in range(N_AGE_V4):
            state[a, 0, 0, IDX_S_V3] = 100.0
        # 感染源集中在 15-49 岁 HIV-/DS 层
        state[2, 0, 0, IDX_ISP_V3] = 10.0
        rhs = self.model.deterministic_rhs_v4(
            state.flatten(), 0.0, beta=0.3,
            rho_fast=0.0, rho_conv=0.0, rho_react=0.0,
            sigma_clear=0.0, rho_prog=0.0, rho_min=0.0,
            eta_sub=0.0, p_clin=0.0, gamma=0.0, beta_reinf=0.0,
            p_sn2sp=0.0, eta_sn=0.0, beta_exo=0.0,
        )
        rhs_4d = rhs.reshape(N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
        # age=2 (15-49) S 的感染率应最大（WAIFW[2,2]=12.0）
        # age=0 (0-4) 的感染率最小（WAIFW[0,2]=1.5）
        dS_adult = -rhs_4d[2, 0, 0, IDX_S_V3]  # 正值=感染率
        dS_child = -rhs_4d[0, 0, 0, IDX_S_V3]
        self.assertGreater(dS_adult, 0.0, "成人应被感染")
        self.assertGreater(dS_child, 0.0, "儿童应被感染（跨年龄接触）")
        self.assertGreater(dS_adult, dS_child,
                          "成人感染率应高于儿童（WAIFW 对角占优）")

    def test_hiv_modified_reactivation(self):
        """改进8：HIV+ 个体的 L_slow→I_sp 再激活率高于 HIV-

        RR_HIV=20 → HIV+ 未治疗（hiv=1）的再激活速率为 HIV- 的 20 倍。
        ART（hiv=2）速率 = RR × (1 - ART_reduction) = 20 × 0.35 = 7 倍。
        关闭 HIV 进展流（hiv_infection_rate=0）以隔离再激活效应。
        """
        state = np.zeros((N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3))
        # 三个 HIV 层各有 100 L_slow 个体（15-49 岁, DS）
        state[2, 0, 0, IDX_LS_V3] = 100.0  # HIV-
        state[2, 1, 0, IDX_LS_V3] = 100.0  # HIV+ 未治疗
        state[2, 2, 0, IDX_LS_V3] = 100.0  # HIV+ ART
        rhs = self.model.deterministic_rhs_v4(
            state.flatten(), 0.0, beta=0.0,
            rho_fast=0.0, rho_conv=0.0,
            sigma_clear=0.0, rho_prog=0.0, rho_min=0.0,
            eta_sub=0.0, p_clin=1.0, gamma=0.0, beta_reinf=0.0,
            p_sn2sp=0.0, eta_sn=0.0, beta_exo=0.0,
            hiv_infection_rate=0.0,  # 关闭 HIV 进展以隔离再激活
            art_initiation_rate=0.0,  # 关闭 ART 启动以隔离再激活
        )
        rhs_4d = rhs.reshape(N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
        # L_slow 流出率（再激活）: HIV+ 未治疗 > HIV+ ART > HIV-
        rate_neg = -rhs_4d[2, 0, 0, IDX_LS_V3]
        rate_pos = -rhs_4d[2, 1, 0, IDX_LS_V3]
        rate_art = -rhs_4d[2, 2, 0, IDX_LS_V3]
        self.assertGreater(rate_pos, rate_neg,
                          "HIV+ 未治疗再激活率应高于 HIV-")
        self.assertGreater(rate_art, rate_neg,
                          "HIV+ ART 再激活率仍应高于 HIV-")
        self.assertGreater(rate_pos, rate_art,
                          "HIV+ 未治疗再激活率应高于 HIV+ ART")

    def test_dr_fitness_cost(self):
        """改进9：DR 株传染性低于 DS 株 (β_DR = β_DS × (1 - f_cost))"""
        state = np.zeros((N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3))
        # DS 和 DR 各有 100 易感者与 10 涂阳感染者
        state[2, 0, 0, IDX_S_V3] = 100.0   # DS 易感
        state[2, 0, 0, IDX_ISP_V3] = 10.0  # DS 感染源
        state[2, 0, 1, IDX_S_V3] = 100.0   # DR 易感
        state[2, 0, 1, IDX_ISP_V3] = 10.0  # DR 感染源
        rhs = self.model.deterministic_rhs_v4(
            state.flatten(), 0.0, beta=0.3,
            rho_fast=0.0, rho_conv=0.0, rho_react=0.0,
            sigma_clear=0.0, rho_prog=0.0, rho_min=0.0,
            eta_sub=0.0, p_clin=0.0, gamma=0.0, beta_reinf=0.0,
            p_sn2sp=0.0, eta_sn=0.0, beta_exo=0.0,
            dr_fitness_cost=0.15,
        )
        rhs_4d = rhs.reshape(N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
        # DS 感染率（S 流出）应高于 DR（因 DR 有 15% 适合度代价）
        ds_infection = -rhs_4d[2, 0, 0, IDX_S_V3]
        dr_infection = -rhs_4d[2, 0, 1, IDX_S_V3]
        self.assertGreater(ds_infection, 0.0)
        self.assertGreater(dr_infection, 0.0)
        self.assertGreater(ds_infection, dr_infection,
                          "DS 感染率应高于 DR（适合度代价）")

    def test_acquired_resistance_flow(self):
        """改进9：DS→DR 获得性耐药流（p_acq × 治疗中 DS 患者）

        在 deterministic_rhs_v4 中，DS 的 I_sub/I_sp/I_sn 的一部分
        通过 p_acq 流向对应的 DR 房室。
        """
        state = np.zeros((N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3))
        # 仅 DS I_sp 有个体（15-49, HIV-）
        state[2, 0, 0, IDX_ISP_V3] = 100.0
        rhs = self.model.deterministic_rhs_v4(
            state.flatten(), 0.0, beta=0.0,
            rho_fast=0.0, rho_conv=0.0, rho_react=0.0,
            sigma_clear=0.0, rho_prog=0.0, rho_min=0.0,
            eta_sub=0.0, p_clin=0.0, gamma=0.0, beta_reinf=0.0,
            p_sn2sp=0.0, eta_sn=0.0, beta_exo=0.0,
            p_acq=0.05,
        )
        rhs_4d = rhs.reshape(N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
        # DR I_sp 应获得正流量（DS→DR）
        dr_isp_inflow = rhs_4d[2, 0, 1, IDX_ISP_V3]
        self.assertGreater(dr_isp_inflow, 0.0,
                          "DR I_sp 应从 DS 获得耐药流")
        # DS I_sp 应有相应流出
        ds_isp_outflow = -rhs_4d[2, 0, 0, IDX_ISP_V3]
        self.assertGreater(ds_isp_outflow, 0.0,
                          "DS I_sp 应有获得性耐药流出")


class TestSimulateSDEv4(unittest.TestCase):
    """simulate_sde_v4 单元测试"""

    def setUp(self):
        self.model = StochasticSEIRModel(
            population=1000, noise_scale=0.05, seed=42)
        self.state = _make_v4_initial_state(1000)

    def test_trajectory_shape(self):
        """轨迹形状为 (n_steps, 270)"""
        times, traj, _ = self.model.simulate_sde_v4(
            self.state, (0, 10), 1.0, beta=0.3)
        self.assertEqual(traj.shape[1], STATE_SIZE_V4)
        self.assertEqual(len(times), traj.shape[0])
        self.assertGreater(traj.shape[0], 1)

    def test_non_negative(self):
        """所有房室值非负"""
        _, traj, _ = self.model.simulate_sde_v4(
            self.state, (0, 10), 1.0, beta=0.3)
        self.assertTrue(np.all(traj >= -1e-9),
                        f"出现负值: min={traj.min()}")

    def test_third_return_none(self):
        """第三个返回值为 None（v4 不返回 stochastic_forcing）"""
        _, _, sto = self.model.simulate_sde_v4(
            self.state, (0, 5), 1.0, beta=0.3)
        self.assertIsNone(sto)

    def test_custom_waifw(self):
        """自定义 WAIFW 矩阵被正确使用（不报错）"""
        custom_waifw = np.ones((N_AGE_V4, N_AGE_V4)) * 2.0
        np.fill_diagonal(custom_waifw, 10.0)
        _, traj, _ = self.model.simulate_sde_v4(
            self.state, (0, 5), 1.0, beta=0.3, waifw=custom_waifw)
        self.assertEqual(traj.shape[1], STATE_SIZE_V4)


class TestSimulateCTMCv4(unittest.TestCase):
    """simulate_ctmc_v4 单元测试（135D 聚合版本）"""

    def setUp(self):
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.state = _make_v4_initial_state(1000)

    def test_returns_array(self):
        """返回 (times, states)，states 形状为 (n_events, 5, 3, 9)"""
        times, states = self.model.simulate_ctmc_v4(
            self.state, max_time=5.0, beta=0.3)
        self.assertIsInstance(times, np.ndarray)
        self.assertIsInstance(states, np.ndarray)
        # 聚合后维度: (5, 3, 9) = 135
        self.assertEqual(states.shape[1], N_AGE_V4)
        self.assertEqual(states.shape[2], N_HIV_V4)
        self.assertEqual(states.shape[3], N_COMPARTMENTS_V3)
        # 至少有初始状态
        self.assertGreaterEqual(len(times), 1)

    def test_non_negative_states(self):
        """所有状态非负"""
        _, states = self.model.simulate_ctmc_v4(
            self.state, max_time=5.0, beta=0.3)
        self.assertTrue(np.all(states >= 0),
                        f"CTMC 出现负值: min={states.min()}")

    def test_at_least_one_event(self):
        """初始有感染源时应触发至少一个事件"""
        times, _ = self.model.simulate_ctmc_v4(
            self.state, max_time=20.0, beta=0.5)
        self.assertGreater(len(times), 1,
                           "应触发至少一个 CTMC 事件")


class TestSimulateMultipleV4(unittest.TestCase):
    """simulate_multiple_v4 单元测试"""

    def test_returns_multiple_trajectories(self):
        """返回多条活动性 TB (I_sp+I_sn) 轨迹

        返回形状: (n_trajectories, n_steps, n_age)
        """
        model = StochasticSEIRModel(population=1000, seed=42)
        state = _make_v4_initial_state(1000)
        trajs = model.simulate_multiple_v4(
            state, (0, 10), 1.0, beta=0.3, n_trajectories=3)
        # 形状: (n_trajectories, n_steps, n_age)
        self.assertEqual(trajs.shape[0], 3)
        self.assertEqual(trajs.shape[2], N_AGE_V4)
        self.assertGreater(trajs.shape[1], 1)
        # 活动 TB 应非负
        self.assertTrue(np.all(trajs >= -1e-9))


class TestBayesianV4Priors(unittest.TestCase):
    """log_prior_v4 单元测试"""

    def setUp(self):
        self.inf = BayesianSEIRInference(seed=42)

    def test_valid_params_finite(self):
        """有效参数下先验为有限值"""
        lp = self.inf.log_prior_v4(
            beta=0.2,
            rho_fast=0.1 / 365.0,
            rho_react=0.004 / 365.0,
            sigma_clear=2.0 / 365.0,
            gamma=0.05,
            omega_reg_m=1.0 / 365.0,
            omega_reg_sub=1.0 / 365.0,
            beta_exo=0.33,
            eta_sn=0.35,
            rr_hiv=20.0,
            art_reduction=0.65,
            hiv_infection_rate=0.001,
            art_initiation_rate=0.1 / 365.0,
            dr_fitness_cost=0.15,
            p_acq=0.03,
        )
        self.assertTrue(np.isfinite(lp), f"先验应为有限值: lp={lp}")
        self.assertGreater(lp, -np.inf)

    def test_invalid_beta_returns_neg_inf(self):
        """β 超出支撑域返回 -inf"""
        lp = self.inf.log_prior_v4(
            beta=2.0,  # 超过上界 0.95
            rho_fast=0.1 / 365.0, rho_react=0.004 / 365.0,
            sigma_clear=2.0 / 365.0, gamma=0.05,
        )
        self.assertEqual(lp, -np.inf)

    def test_invalid_rr_hiv_returns_neg_inf(self):
        """RR_HIV 超出支撑域 [5, 50] 返回 -inf"""
        lp = self.inf.log_prior_v4(
            beta=0.2,
            rho_fast=0.1 / 365.0, rho_react=0.004 / 365.0,
            sigma_clear=2.0 / 365.0, gamma=0.05,
            rr_hiv=100.0,  # 超过上界 50
        )
        self.assertEqual(lp, -np.inf)

    def test_invalid_art_reduction_returns_neg_inf(self):
        """ART_reduction 超出支撑域 [0.20, 0.90] 返回 -inf"""
        lp = self.inf.log_prior_v4(
            beta=0.2,
            rho_fast=0.1 / 365.0, rho_react=0.004 / 365.0,
            sigma_clear=2.0 / 365.0, gamma=0.05,
            art_reduction=0.95,  # 超过上界 0.90
        )
        self.assertEqual(lp, -np.inf)

    def test_invalid_dr_fitness_cost_returns_neg_inf(self):
        """DR_fitness_cost 超出支撑域 [0.02, 0.40] 返回 -inf"""
        lp = self.inf.log_prior_v4(
            beta=0.2,
            rho_fast=0.1 / 365.0, rho_react=0.004 / 365.0,
            sigma_clear=2.0 / 365.0, gamma=0.05,
            dr_fitness_cost=0.50,  # 超过上界 0.40
        )
        self.assertEqual(lp, -np.inf)

    def test_invalid_p_acq_returns_neg_inf(self):
        """p_acq 超出支撑域 [0.005, 0.10] 返回 -inf"""
        lp = self.inf.log_prior_v4(
            beta=0.2,
            rho_fast=0.1 / 365.0, rho_react=0.004 / 365.0,
            sigma_clear=2.0 / 365.0, gamma=0.05,
            p_acq=0.20,  # 超过上界 0.10
        )
        self.assertEqual(lp, -np.inf)

    def test_invalid_hiv_infection_rate_returns_neg_inf(self):
        """hiv_infection_rate 超出支撑域返回 -inf"""
        lp = self.inf.log_prior_v4(
            beta=0.2,
            rho_fast=0.1 / 365.0, rho_react=0.004 / 365.0,
            sigma_clear=2.0 / 365.0, gamma=0.05,
            hiv_infection_rate=0.1,  # 超过上界 0.01
        )
        self.assertEqual(lp, -np.inf)

    def test_v4_param_names_length(self):
        """V4_PARAM_NAMES 长度为 15"""
        self.assertEqual(len(self.inf.V4_PARAM_NAMES), 15)
        self.assertEqual(self.inf.V4_N_PARAMS, 15)

    def test_v4_init_params_shape(self):
        """V4_INIT_PARAMS 形状为 (15,)"""
        self.assertEqual(self.inf.V4_INIT_PARAMS.shape, (15,))
        # 所有初始值应在支撑域内
        for val, lo, hi in zip(self.inf.V4_INIT_PARAMS,
                               self.inf.V4_LOWER_BOUNDS,
                               self.inf.V4_UPPER_BOUNDS):
            self.assertGreater(val, lo, f"初始值 {val} 低于下界 {lo}")
            self.assertLess(val, hi, f"初始值 {val} 高于上界 {hi}")


class TestBayesianV4Likelihood(unittest.TestCase):
    """log_likelihood_v4 与 log_posterior_v4 单元测试"""

    def setUp(self):
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.inf = BayesianSEIRInference(seir_model=self.model, seed=42)
        self.init_state = _make_v4_initial_state(1000)
        self.observed_data = _make_v4_observed_data()

    def _make_params(self):
        """构造默认参数元组（15 元组）"""
        return (
            0.3,                      # beta
            0.1 / 365.0,              # rho_fast
            0.004 / 365.0,            # rho_react
            2.0 / 365.0,              # sigma_clear
            0.05,                     # gamma
            1.0 / 365.0,              # omega_reg_m
            1.0 / 365.0,              # omega_reg_sub
            0.33,                     # beta_exo
            0.35,                     # eta_sn
            20.0,                     # rr_hiv
            0.65,                     # art_reduction
            0.001,                    # hiv_infection_rate
            0.1 / 365.0,              # art_initiation_rate
            0.15,                     # dr_fitness_cost
            0.03,                     # p_acq
        )

    def test_likelihood_finite(self):
        """似然值为有限浮点数"""
        params = self._make_params()
        ll = self.inf.log_likelihood_v4(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        self.assertTrue(np.isfinite(ll), f"似然应为有限值: ll={ll}")

    def test_likelihood_deterministic(self):
        """同一参数集两次调用应返回相同似然值（detailed balance 保护）"""
        params = self._make_params()
        ll1 = self.inf.log_likelihood_v4(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        ll2 = self.inf.log_likelihood_v4(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        self.assertEqual(ll1, ll2)

    def test_posterior_finite(self):
        """后验值为有限浮点数"""
        params = self._make_params()
        lp = self.inf.log_posterior_v4(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        self.assertTrue(np.isfinite(lp), f"后验应为有限值: lp={lp}")

    def test_invalid_params_posterior_neg_inf(self):
        """无效参数下后验为 -inf"""
        params = list(self._make_params())
        params[0] = 2.0  # β 超界
        lp = self.inf.log_posterior_v4(
            tuple(params), self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        self.assertEqual(lp, -np.inf)

    def test_invalid_rr_hiv_posterior_neg_inf(self):
        """RR_HIV 超界时后验为 -inf"""
        params = list(self._make_params())
        params[9] = 100.0  # rr_hiv 超过上界 50
        lp = self.inf.log_posterior_v4(
            tuple(params), self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0)
        self.assertEqual(lp, -np.inf)

    def test_custom_waifw_in_likelihood(self):
        """似然函数接受自定义 WAIFW 矩阵"""
        params = self._make_params()
        custom_waifw = np.ones((N_AGE_V4, N_AGE_V4)) * 2.0
        np.fill_diagonal(custom_waifw, 10.0)
        ll = self.inf.log_likelihood_v4(
            params, self.observed_data, self.init_state,
            t_span=(0, 30), dt=1.0, waifw=custom_waifw)
        self.assertTrue(np.isfinite(ll), f"自定义 WAIFW 似然应为有限值: ll={ll}")


class TestUncertaintyV4Priors(unittest.TestCase):
    """uncertainty.py v4 先验字典与支撑域测试"""

    def test_v4_priors_in_dict(self):
        """v4 新参数出现在 SEIR_PARAM_PRIORS 字典中"""
        from tb_risk.seir.uncertainty import SEIRParameterUncertainty
        priors = SEIRParameterUncertainty.SEIR_PARAM_PRIORS
        for name in ('rr_hiv', 'art_reduction', 'hiv_infection_rate',
                     'art_initiation_rate', 'dr_fitness_cost', 'p_acq'):
            self.assertIn(name, priors, f"v4 参数 {name} 未在先验字典中")

    def test_v4_prior_distributions(self):
        """v4 参数分布类型正确"""
        from tb_risk.seir.uncertainty import SEIRParameterUncertainty
        priors = SEIRParameterUncertainty.SEIR_PARAM_PRIORS
        # HIV 参数: rr_hiv (Gamma), art_reduction (Beta),
        # hiv_infection_rate (Gamma), art_initiation_rate (Gamma)
        self.assertEqual(priors['rr_hiv']['dist'], 'gamma')
        self.assertEqual(priors['art_reduction']['dist'], 'beta')
        self.assertEqual(priors['hiv_infection_rate']['dist'], 'gamma')
        self.assertEqual(priors['art_initiation_rate']['dist'], 'gamma')
        # 耐药参数: dr_fitness_cost (Beta), p_acq (Beta)
        self.assertEqual(priors['dr_fitness_cost']['dist'], 'beta')
        self.assertEqual(priors['p_acq']['dist'], 'beta')

    def test_v4_prior_means(self):
        """v4 参数先验均值接近文献值"""
        from tb_risk.seir.uncertainty import SEIRParameterUncertainty
        priors = SEIRParameterUncertainty.SEIR_PARAM_PRIORS
        # RR_HIV: Gamma(2, scale=10) → 均值=20
        self.assertAlmostEqual(priors['rr_hiv']['shape'] *
                               priors['rr_hiv']['scale'], 20.0, places=6)
        # ART_reduction: Beta(13, 7) → 均值=13/20=0.65
        self.assertAlmostEqual(priors['art_reduction']['a'] /
                               (priors['art_reduction']['a'] +
                                priors['art_reduction']['b']), 0.65, places=6)
        # DR_fitness_cost: Beta(6, 30) → 均值=6/36≈0.167
        self.assertAlmostEqual(
            priors['dr_fitness_cost']['a'] /
            (priors['dr_fitness_cost']['a'] + priors['dr_fitness_cost']['b']),
            0.167, places=2)
        # p_acq: Beta(3, 90) → 均值=3/93≈0.032
        self.assertAlmostEqual(
            priors['p_acq']['a'] /
            (priors['p_acq']['a'] + priors['p_acq']['b']),
            0.032, places=3)

    def test_v4_support_domain_rejects_invalid(self):
        """v4 支撑域检查拒绝无效参数"""
        # 要点6 (第十四轮): 从 SEIRParameterUncertainty 迁移到
        # BayesianInferenceV4 (seir.inference.get_inference_engine('v4'))
        # 消除弃用警告。原接口 _log_prior(dict) → 新接口 log_prior_v4(**kwargs)
        from tb_risk.seir.inference import get_inference_engine
        BayesianInferenceV4 = get_inference_engine('v4')
        inf = BayesianInferenceV4(seed=42)
        # log_prior_v4 必填位置参数: beta, rho_fast, rho_react, sigma_clear, gamma
        # 其余 10 个参数有默认值
        valid = {
            'beta': 0.2,
            'rho_fast': 0.1 / 365.0,
            'rho_react': 0.004 / 365.0,
            'sigma_clear': 2.0 / 365.0,
            'gamma': 0.05,
            'rr_hiv': 20.0, 'art_reduction': 0.65,
            'dr_fitness_cost': 0.15, 'p_acq': 0.03,
            'hiv_infection_rate': 0.001, 'art_initiation_rate': 0.1 / 365.0,
        }
        self.assertTrue(np.isfinite(inf.log_prior_v4(**valid)))
        # RR_HIV 超界
        invalid = dict(valid, rr_hiv=100.0)
        self.assertEqual(inf.log_prior_v4(**invalid), -np.inf)
        # p_acq 超界
        invalid = dict(valid, p_acq=0.5)
        self.assertEqual(inf.log_prior_v4(**invalid), -np.inf)
        # dr_fitness_cost 超界
        invalid = dict(valid, dr_fitness_cost=0.5)
        self.assertEqual(inf.log_prior_v4(**invalid), -np.inf)


if __name__ == '__main__':
    unittest.main()
