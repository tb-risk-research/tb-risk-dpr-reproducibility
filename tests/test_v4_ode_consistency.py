#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — NumPy/PyTorch v4 ODE 数值一致性验证（改进11）

验证 _stochastic_v4.py 的 NumPy ODE (deterministic_rhs_v4, 使用原地切片赋值)
与 autodiff.py 的 PyTorch ODE (torch_v4_rhs, 使用函数式 torch.stack 构造)
在相同参数和初始状态下产生数值一致的结果（容差 1e-10）。

关键差异点（需验证一致）：
  1. 获得性耐药修正: NumPy 用 dIsub[:,:,0] -= dr_acq (原地); PyTorch 用 torch.stack([-acq, acq])
  2. HIV 进展修正: NumPy 用 dS[:,0,:] -= hiv_01 (原地); PyTorch 用 torch.stack([-hiv_01, hiv_01-art_12, art_12])
  3. DR 适合度代价: NumPy 用 infectious[:,1] *= (1-f_cost) (原地); PyTorch 用 dr_mult 张量相乘

文献：
  Hoffman MD, Gelman A. (2014) JMLR 15(1):1593-1623 — HMC 需要可微 ODE
  Paszke A et al. (2019) NeurIPS — PyTorch autograd 计算图
"""

import os
import sys
import unittest

import numpy as np

# 确保包路径在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.seir.stochastic import (
    StochasticSEIRModel,
    N_AGE_V4, N_HIV_V4, N_DR_V4, STATE_SIZE_V4, N_COMPARTMENTS_V3,
    IDX_S_V3, IDX_LF_V3, IDX_LS_V3, IDX_M_V3,
    IDX_ISUB_V3, IDX_ISP_V3, IDX_ISN_V3, IDX_R_V3, IDX_C_V3,
    DEFAULT_WAIFW, DEFAULT_AGE_PROGRESSION,
    DEFAULT_RHO_CONV, DEFAULT_RHO_PROG, DEFAULT_ETA_SUB,
    DEFAULT_P_CLIN, DEFAULT_BETA_REINF, DEFAULT_RHO_MIN, DEFAULT_P_M,
    DEFAULT_P_SP, DEFAULT_MU_SP, DEFAULT_R_SP,
    DEFAULT_MU_SN, DEFAULT_R_SN, DEFAULT_P_SN2SP,
)

# 可选依赖：torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def _make_consistency_state(population=1000):
    """构造 270D 测试状态: (5 age, 3 HIV, 2 DR, 9 comp)

    状态设计：覆盖所有房室与所有分层维度，确保 HIV 进展流、
    获得性耐药流、DR 适合度代价等修正项均被激活。
    """
    state = np.zeros((N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3))
    # 年龄分布
    age_dist = np.array([0.06, 0.12, 0.55, 0.18, 0.09])
    for a in range(N_AGE_V4):
        n_a = int(population * age_dist[a])
        # 分布到 HIV-/DS 的 S
        state[a, 0, 0, IDX_S_V3] = float(n_a) * 0.90
        state[a, 0, 0, IDX_LF_V3] = float(n_a) * 0.05
        state[a, 0, 0, IDX_LS_V3] = float(n_a) * 0.03
        state[a, 0, 0, IDX_C_V3] = float(n_a) * 0.02
        # 在 HIV+ 未治疗层放少量 R（激活 HIV 进展修正）
        state[a, 1, 0, IDX_R_V3] = float(n_a) * 0.01
        # 在 15-49 岁层注入活动性 TB (DS 和 DR)
        if a == 2:
            state[a, 0, 0, IDX_ISUB_V3] = 5.0
            state[a, 0, 0, IDX_ISP_V3] = 3.0
            state[a, 0, 0, IDX_ISN_V3] = 2.0
            # DR 层也有少量感染（激活 DR 适合度代价）
            state[a, 0, 1, IDX_ISUB_V3] = 1.0
            state[a, 0, 1, IDX_ISP_V3] = 0.5
            # HIV+ 层有少量 M 房室（激活 HIV 进展的全房室覆盖）
            state[a, 1, 0, IDX_M_V3] = 1.0
            state[a, 2, 0, IDX_LF_V3] = 2.0
    return state.flatten()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch 不可用")
class TestNumpyTorchV4ODEConsistency(unittest.TestCase):
    """NumPy (deterministic_rhs_v4) vs PyTorch (torch_v4_rhs) 数值一致性测试"""

    def setUp(self):
        from tb_risk.seir.autodiff import (constrain_params, N_INFER_PARAMS,
                                            PARAM_BOUNDS, PARAM_NAMES)
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.state_np = _make_consistency_state(1000)
        self.PARAM_BOUNDS = PARAM_BOUNDS
        self.PARAM_NAMES = PARAM_NAMES
        self.N_INFER_PARAMS = N_INFER_PARAMS
        self.constrain_params = constrain_params

        # 用 y=0 (sigmoid 中点) 获取一组一致的推断参数
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64)
        params_torch, _ = constrain_params(y)
        self.params_torch = params_torch
        # 提取标量值用于 NumPy 调用
        self.params_scalar = {name: float(params_torch[name])
                              for name in PARAM_NAMES}

    def _call_numpy_rhs(self, state):
        """调用 NumPy deterministic_rhs_v4"""
        p = self.params_scalar
        return self.model.deterministic_rhs_v4(
            state, 0.0,
            beta=p['beta'],
            rho_fast=p['rho_fast'],
            rho_react=p['rho_react'],
            sigma_clear=p['sigma_clear'],
            omega_reg_m=p['omega_reg_m'],
            omega_reg_sub=p['omega_reg_sub'],
            beta_exo=p['beta_exo'],
            eta_sn=p['eta_sn'],
            rr_hiv=p['rr_hiv'],
            art_reduction=p['art_reduction'],
            hiv_infection_rate=p['hiv_infection_rate'],
            art_initiation_rate=p['art_initiation_rate'],
            dr_fitness_cost=p['dr_fitness_cost'],
            p_acq=p['p_acq'],
            # 固定参数使用默认值（与 PyTorch fixed_params=None 一致）
            waifw=DEFAULT_WAIFW,
            age_progression=DEFAULT_AGE_PROGRESSION,
        )

    def _call_torch_rhs(self, state_np):
        """调用 PyTorch torch_v4_rhs"""
        from tb_risk.seir.autodiff import torch_v4_rhs
        waifw_t = torch.tensor(DEFAULT_WAIFW, dtype=torch.float64)
        age_prog_t = torch.tensor(DEFAULT_AGE_PROGRESSION, dtype=torch.float64)
        state_t = torch.tensor(state_np, dtype=torch.float64)
        rhs = torch_v4_rhs(state_t, self.params_torch, waifw_t, age_prog_t)
        return rhs.detach().numpy()

    def test_rhs_shapes_match(self):
        """两实现输出形状应一致 (270,)"""
        rhs_np = self._call_numpy_rhs(self.state_np)
        rhs_torch = self._call_torch_rhs(self.state_np)
        self.assertEqual(rhs_np.shape, rhs_torch.shape)
        self.assertEqual(rhs_np.shape[0], STATE_SIZE_V4)

    def test_rhs_finite_both(self):
        """两实现输出均应为有限值"""
        rhs_np = self._call_numpy_rhs(self.state_np)
        rhs_torch = self._call_torch_rhs(self.state_np)
        self.assertTrue(np.all(np.isfinite(rhs_np)),
                        f"NumPy rhs 含 NaN/Inf: {rhs_np}")
        self.assertTrue(np.all(np.isfinite(rhs_torch)),
                        f"PyTorch rhs 含 NaN/Inf: {rhs_torch}")

    def test_rhs_numerical_consistency(self):
        """核心测试: 两实现数值一致 (容差 1e-10)

        这是改进11的关键验证：NumPy 原地切片赋值与 PyTorch 函数式
        torch.stack 构造应产生相同的导数结果。
        """
        rhs_np = self._call_numpy_rhs(self.state_np)
        rhs_torch = self._call_torch_rhs(self.state_np)
        max_diff = float(np.max(np.abs(rhs_np - rhs_torch)))
        self.assertLess(max_diff, 1e-10,
                        f"NumPy/PyTorch v4 ODE 不一致: max_diff={max_diff}")

    def test_mass_conservation_both(self):
        """两实现均应满足质量守恒 (sum ≈ 0)"""
        rhs_np = self._call_numpy_rhs(self.state_np)
        rhs_torch = self._call_torch_rhs(self.state_np)
        # 相对于群体 1000，相对误差应 < 1e-3
        self.assertAlmostEqual(float(np.sum(rhs_np)), 0.0, places=3)
        self.assertAlmostEqual(float(np.sum(rhs_torch)), 0.0, places=3)

    def test_hiv_progression_consistency(self):
        """HIV 进展修正一致性 (hiv=0→1, hiv=1→2)

        NumPy: dS[:,0,:] -= hiv_01; dS[:,1,:] += hiv_01 - art_12; dS[:,2,:] += art_12
        PyTorch: hiv_adj = torch.stack([-hiv_01, hiv_01-art_12, art_12], dim=2)
        """
        # 构造仅 HIV 进展流激活的状态（无感染源）
        state = np.zeros((N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3))
        # 在 HIV- 层放 S, Lf, Ls, M, R, C（所有房室）
        for comp in [IDX_S_V3, IDX_LF_V3, IDX_LS_V3, IDX_M_V3,
                     IDX_R_V3, IDX_C_V3]:
            state[2, 0, 0, comp] = 10.0
        # 在 HIV+ 未治疗层放少量个体（激活 art_12 流）
        state[2, 1, 0, IDX_S_V3] = 5.0
        state_flat = state.flatten()

        rhs_np = self._call_numpy_rhs(state_flat)
        rhs_torch = self._call_torch_rhs(state_flat)
        max_diff = float(np.max(np.abs(rhs_np - rhs_torch)))
        self.assertLess(max_diff, 1e-10,
                        f"HIV 进展修正不一致: max_diff={max_diff}")

    def test_dr_acquisition_consistency(self):
        """获得性耐药修正一致性 (DS → DR)

        NumPy: dIsub[:,:,0] -= dr_acq; dIsub[:,:,1] += dr_acq (原地切片)
        PyTorch: dIsub + torch.stack([-acq, acq], dim=-1) (函数式)
        """
        # 构造仅 DR 获得性耐药流激活的状态
        state = np.zeros((N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3))
        # DS 层的 Isub/Isp/Isn（激活 dr_acq）
        state[2, 0, 0, IDX_ISUB_V3] = 10.0
        state[2, 0, 0, IDX_ISP_V3] = 5.0
        state[2, 0, 0, IDX_ISN_V3] = 3.0
        state_flat = state.flatten()

        rhs_np = self._call_numpy_rhs(state_flat)
        rhs_torch = self._call_torch_rhs(state_flat)
        max_diff = float(np.max(np.abs(rhs_np - rhs_torch)))
        self.assertLess(max_diff, 1e-10,
                        f"DR 获得性耐药修正不一致: max_diff={max_diff}")

    def test_dr_fitness_cost_consistency(self):
        """DR 适合度代价一致性

        NumPy: infectious[:,1] *= (1 - dr_fitness_cost) (原地)
        PyTorch: infectious * dr_mult (torch.stack 构造 dr_mult)
        """
        # 构造 DR 层有感染源的状态
        state = np.zeros((N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3))
        state[2, 0, 0, IDX_S_V3] = 900.0
        # DR 层的 Isp（激活 DR 适合度代价）
        state[2, 0, 1, IDX_ISP_V3] = 10.0
        state_flat = state.flatten()

        rhs_np = self._call_numpy_rhs(state_flat)
        rhs_torch = self._call_torch_rhs(state_flat)
        max_diff = float(np.max(np.abs(rhs_np - rhs_torch)))
        self.assertLess(max_diff, 1e-10,
                        f"DR 适合度代价不一致: max_diff={max_diff}")

    def test_euler_integration_consistency(self):
        """单步 Euler 积分后两实现仍一致 (容差 1e-8)"""
        dt = 0.1  # 单步
        rhs_np = self._call_numpy_rhs(self.state_np)
        rhs_torch = self._call_torch_rhs(self.state_np)
        state_next_np = self.state_np + dt * rhs_np
        state_next_torch = self.state_np + dt * rhs_torch
        max_diff = float(np.max(np.abs(state_next_np - state_next_torch)))
        self.assertLess(max_diff, 1e-9,
                        f"Euler 单步积分不一致: max_diff={max_diff}")

    def test_multi_step_consistency(self):
        """多步 Euler 积分一致性 (5 步, 容差 1e-7)

        累积误差应仍在可接受范围内。
        """
        dt = 0.05
        n_steps = 5
        state_np = self.state_np.copy()
        state_torch = self.state_np.copy()
        for _ in range(n_steps):
            rhs_np = self._call_numpy_rhs(state_np)
            rhs_torch = self._call_torch_rhs(state_torch)
            state_np = state_np + dt * rhs_np
            state_torch = state_torch + dt * rhs_torch
        max_diff = float(np.max(np.abs(state_np - state_torch)))
        self.assertLess(max_diff, 1e-7,
                        f"多步 Euler 积分不一致: max_diff={max_diff}")

    def test_different_params_consistency(self):
        """不同参数下两实现仍一致 (y=1.0)"""
        y = torch.ones(self.N_INFER_PARAMS, dtype=torch.float64)
        params_torch, _ = self.constrain_params(y)
        params_scalar = {name: float(params_torch[name])
                         for name in self.PARAM_NAMES}

        rhs_np = self.model.deterministic_rhs_v4(
            self.state_np, 0.0,
            beta=params_scalar['beta'],
            rho_fast=params_scalar['rho_fast'],
            rho_react=params_scalar['rho_react'],
            sigma_clear=params_scalar['sigma_clear'],
            omega_reg_m=params_scalar['omega_reg_m'],
            omega_reg_sub=params_scalar['omega_reg_sub'],
            beta_exo=params_scalar['beta_exo'],
            eta_sn=params_scalar['eta_sn'],
            rr_hiv=params_scalar['rr_hiv'],
            art_reduction=params_scalar['art_reduction'],
            hiv_infection_rate=params_scalar['hiv_infection_rate'],
            art_initiation_rate=params_scalar['art_initiation_rate'],
            dr_fitness_cost=params_scalar['dr_fitness_cost'],
            p_acq=params_scalar['p_acq'],
        )

        from tb_risk.seir.autodiff import torch_v4_rhs
        waifw_t = torch.tensor(DEFAULT_WAIFW, dtype=torch.float64)
        age_prog_t = torch.tensor(DEFAULT_AGE_PROGRESSION, dtype=torch.float64)
        state_t = torch.tensor(self.state_np, dtype=torch.float64)
        rhs_torch = torch_v4_rhs(state_t, params_torch, waifw_t, age_prog_t)
        rhs_torch_np = rhs_torch.detach().numpy()

        max_diff = float(np.max(np.abs(rhs_np - rhs_torch_np)))
        self.assertLess(max_diff, 1e-10,
                        f"不同参数下不一致: max_diff={max_diff}")


if __name__ == '__main__':
    unittest.main()
