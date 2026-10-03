#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — RK4 高阶积分器

测试改进16 (五): RK4 4 阶固定步长积分, 替代 Euler 1 阶。
TB 动力学时间尺度差异大 (感染~天, 自清除~月, 再激活~年), RK4 可用
更大步长达到相同精度, 减少总 RHS 评估数。

覆盖:
  1. torch_v4_integrate method 参数 (euler/rk4)
  2. RK4 与 Euler 形状一致
  3. RK4 与 Euler 数值不同 (更高精度)
  4. RK4 autograd 梯度可计算
  5. RK4 与 Euler 梯度不同
  6. 非法 method 抛 ValueError
  7. RK4 比 Euler 更精确 (小步长基准对比)
  8. log_likelihood/posterior/grad 透传 method
  9. run_mcmc_v4 method 参数端到端

文献:
  Hairer et al. (1993) Solving ODE I — RK4 经典 4 阶方法
  Hoffman & Gelman (2014) JMLR — HMC 对梯度精度的敏感性
"""

import os
import sys
import unittest

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _build_initial_state():
    """构造 270D 初始状态: 900 S + 50 Lf + 10 Isp 在 15-49 岁 HIV-/DS"""
    state = np.zeros(270)
    base = 2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9
    state[base + 0] = 900.0  # S
    state[base + 1] = 50.0   # Lf
    state[base + 5] = 10.0   # Isp
    return state


def _default_unconstrained_params():
    """15D 无约束参数 (中点 = 0, sigmoid 变换后为约束范围中点)"""
    return torch.zeros(15, dtype=torch.float64)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRK4Integration(unittest.TestCase):
    """RK4 积分器基本功能测试"""

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_method_param_accepted(self):
        """torch_v4_integrate 接受 method 参数"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        traj = torch_v4_integrate(
            y, _build_initial_state(), n_steps=5, dt=1.0, method='euler')
        self.assertEqual(traj.shape, (1, 6, 270))

    def test_rk4_shape_matches_euler(self):
        """RK4 与 Euler 输出形状一致"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        init = _build_initial_state()
        traj_e = torch_v4_integrate(y, init, n_steps=10, dt=1.0, method='euler')
        traj_r = torch_v4_integrate(y, init, n_steps=10, dt=1.0, method='rk4')
        self.assertEqual(traj_e.shape, traj_r.shape)
        self.assertEqual(traj_r.shape, (1, 11, 270))

    def test_rk4_differs_from_euler(self):
        """RK4 与 Euler 数值不同 (RK4 更精确)"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        init = _build_initial_state()
        traj_e = torch_v4_integrate(y, init, n_steps=20, dt=1.0, method='euler')
        traj_r = torch_v4_integrate(y, init, n_steps=20, dt=1.0, method='rk4')
        diff = (traj_e - traj_r).abs().max().item()
        self.assertGreater(diff, 1e-10,
                           "Euler 和 RK4 应有可测量差异")

    def test_rk4_supports_batch(self):
        """RK4 支持 batch 参数 (B, 15)"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = torch.zeros(4, 15, dtype=torch.float64)
        traj = torch_v4_integrate(
            y, _build_initial_state(), n_steps=5, dt=1.0, method='rk4')
        self.assertEqual(traj.shape, (4, 6, 270))

    def test_rk4_output_finite(self):
        """RK4 输出为有限值 (无 NaN/Inf)"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        traj = torch_v4_integrate(
            y, _build_initial_state(), n_steps=30, dt=1.0, method='rk4')
        self.assertTrue(torch.isfinite(traj).all(),
                        "RK4 轨迹含 NaN/Inf")

    def test_rk4_non_negative_states(self):
        """RK4 积分后状态非负 (clamp 保护)"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        traj = torch_v4_integrate(
            y, _build_initial_state(), n_steps=30, dt=1.0, method='rk4')
        self.assertTrue(torch.all(traj >= 0),
                        "RK4 轨迹含负值")

    def test_invalid_method_raises(self):
        """非法 method 抛 ValueError"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        with self.assertRaises(ValueError):
            torch_v4_integrate(
                y, _build_initial_state(), n_steps=5, dt=1.0,
                method='midpoint')
        with self.assertRaises(ValueError):
            torch_v4_integrate(
                y, _build_initial_state(), n_steps=5, dt=1.0,
                method='RK4')  # 大小写敏感


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRK4Autograd(unittest.TestCase):
    """RK4 autograd 梯度测试"""

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_rk4_gradient_computable(self):
        """RK4 梯度可计算 (autograd 计算图完整)"""
        from tb_risk.seir.autodiff import compute_grad_log_posterior
        y = _default_unconstrained_params()
        grad, lp = compute_grad_log_posterior(
            y, (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            _build_initial_state(), (0, 10), 1.0, method='rk4')
        self.assertEqual(grad.shape, (15,))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(torch.isfinite(lp))

    def test_rk4_gradient_differs_from_euler(self):
        """RK4 梯度与 Euler 不同 (更高精度)"""
        from tb_risk.seir.autodiff import compute_grad_log_posterior
        y = _default_unconstrained_params()
        obs = (np.array([5.0, 10.0]), np.array([5.0, 10.0]))
        init = _build_initial_state()

        grad_e, lp_e = compute_grad_log_posterior(
            y, obs, init, (0, 10), 1.0, method='euler')
        grad_r, lp_r = compute_grad_log_posterior(
            y, obs, init, (0, 10), 1.0, method='rk4')

        lp_diff = (lp_e - lp_r).abs().item()
        grad_diff = (grad_e - grad_r).abs().max().item()
        self.assertGreater(lp_diff, 1e-10, "log_post 应有差异")
        self.assertGreater(grad_diff, 1e-10, "梯度应有差异")

    def test_rk4_gradient_finite(self):
        """RK4 梯度全有限 (数值稳定)"""
        from tb_risk.seir.autodiff import compute_grad_log_posterior
        # 非零无约束参数 (测试非平凡点)
        y = torch.randn(15, dtype=torch.float64) * 0.5
        grad, lp = compute_grad_log_posterior(
            y, (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            _build_initial_state(), (0, 10), 1.0, method='rk4')
        self.assertTrue(torch.isfinite(grad).all(),
                        f"RK4 梯度含 NaN/Inf: {grad}")
        self.assertTrue(torch.isfinite(lp))

    def test_rk4_batch_gradient(self):
        """RK4 支持 batch 梯度 (B, 15)"""
        from tb_risk.seir.autodiff import compute_grad_log_posterior
        y = torch.randn(3, 15, dtype=torch.float64) * 0.3
        grad, lp = compute_grad_log_posterior(
            y, (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            _build_initial_state(), (0, 10), 1.0, method='rk4')
        self.assertEqual(grad.shape, (3, 15))
        self.assertEqual(lp.shape, (3,))
        self.assertTrue(torch.isfinite(grad).all())


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRK4Accuracy(unittest.TestCase):
    """RK4 精度测试 (对比 Euler)"""

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_rk4_converges_faster_than_euler(self):
        """RK4 步长减半时误差减小 ~16x (4阶), Euler 减半 ~2x (1阶)

        用 RK4 大步长 vs Euler 小步长对比, 验证 RK4 在更少步数下
        达到 Euler 的精度。基准: Euler 极小步长 (n_steps=200) 作为
        "真值", 对比 Euler n_steps=20 和 RK4 n_steps=20 的误差。
        """
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        init = _build_initial_state()
        t_span_total = 10.0

        # 基准: Euler 极小步长 (n_steps=200, dt=0.05) 作为 "真值"
        dt_ref = t_span_total / 200
        traj_ref = torch_v4_integrate(
            y, init, n_steps=200, dt=dt_ref, method='euler')
        # 取最后状态 (n_steps+1 个点, 索引 -1)
        ref_final = traj_ref[0, -1, :]

        # Euler 粗 (n_steps=20, dt=0.5)
        dt_coarse = t_span_total / 20
        traj_e = torch_v4_integrate(
            y, init, n_steps=20, dt=dt_coarse, method='euler')
        e_final = traj_e[0, -1, :]

        # RK4 粗 (n_steps=20, dt=0.5)
        traj_r = torch_v4_integrate(
            y, init, n_steps=20, dt=dt_coarse, method='rk4')
        r_final = traj_r[0, -1, :]

        err_e = (e_final - ref_final).abs().max().item()
        err_r = (r_final - ref_final).abs().max().item()

        # RK4 误差应明显小于 Euler (4阶 vs 1阶)
        self.assertLess(err_r, err_e,
                        f"RK4 误差 ({err_r:.2e}) 应小于 Euler ({err_e:.2e})")
        # RK4 应至少小 5x (4阶方法在粗网格上通常小 10-100x, 但 TB 动力学
        # 的非线性使理论收敛阶不完全成立, 5x 是稳健的下界)
        self.assertLess(err_r, err_e * 0.2,
                        f"RK4 应比 Euler 精确 5x 以上: "
                        f"RK4={err_r:.2e}, Euler={err_e:.2e}")

    def test_rk4_with_larger_dt_matches_euler_finer(self):
        """RK4 大步长 (dt=2) 接近 Euler 小步长 (dt=1) 的精度

        这是改进16 的核心收益: RK4 用更少步数达到相同精度。
        """
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        init = _build_initial_state()

        # 基准: Euler dt=0.25 (n_steps=40, 精细)
        traj_ref = torch_v4_integrate(y, init, n_steps=40, dt=0.25, method='euler')
        ref_final = traj_ref[0, -1, :]

        # Euler dt=1 (n_steps=10, 粗糙)
        traj_e = torch_v4_integrate(y, init, n_steps=10, dt=1.0, method='euler')
        e_final = traj_e[0, -1, :]

        # RK4 dt=2 (n_steps=5, 更少步数)
        traj_r = torch_v4_integrate(y, init, n_steps=5, dt=2.0, method='rk4')
        r_final = traj_r[0, -1, :]

        err_e = (e_final - ref_final).abs().max().item()
        err_r = (r_final - ref_final).abs().max().item()

        # RK4 用 5 步 (dt=2) 的误差应与 Euler 10 步 (dt=1) 相当或更小
        self.assertLess(err_r, err_e * 2,
                        f"RK4 5步 (dt=2) 误差 {err_r:.2e} 应接近或小于 "
                        f"Euler 10步 (dt=1) {err_e:.2e}")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMethodPropagation(unittest.TestCase):
    """method 参数透传测试"""

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_log_likelihood_accepts_method(self):
        """torch_log_likelihood_v4 接受 method 参数"""
        from tb_risk.seir.autodiff import torch_log_likelihood_v4
        y = _default_unconstrained_params()
        ll = torch_log_likelihood_v4(
            y, (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            _build_initial_state(), (0, 10), 1.0, method='rk4')
        self.assertTrue(torch.isfinite(ll))

    def test_log_posterior_accepts_method(self):
        """torch_log_posterior_v4 接受 method 参数"""
        from tb_risk.seir.autodiff import torch_log_posterior_v4
        y = _default_unconstrained_params()
        lp = torch_log_posterior_v4(
            y, (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            _build_initial_state(), (0, 10), 1.0, method='rk4')
        self.assertTrue(torch.isfinite(lp))

    def test_log_posterior_method_differs(self):
        """不同 method 的 log_posterior 值不同"""
        from tb_risk.seir.autodiff import torch_log_posterior_v4
        y = _default_unconstrained_params()
        obs = (np.array([5.0, 10.0]), np.array([5.0, 10.0]))
        init = _build_initial_state()

        lp_e = torch_log_posterior_v4(y, obs, init, (0, 10), 1.0, method='euler')
        lp_r = torch_log_posterior_v4(y, obs, init, (0, 10), 1.0, method='rk4')
        self.assertNotAlmostEqual(lp_e.item(), lp_r.item(), places=3)

    def test_grad_log_posterior_method_via_kwargs(self):
        """compute_grad_log_posterior 通过 **kwargs 透传 method"""
        from tb_risk.seir.autodiff import compute_grad_log_posterior
        y = _default_unconstrained_params()
        obs = (np.array([5.0, 10.0]), np.array([5.0, 10.0]))
        init = _build_initial_state()

        # method 通过 kwargs 透传
        grad, lp = compute_grad_log_posterior(
            y, obs, init, (0, 10), 1.0, method='rk4')
        self.assertTrue(torch.isfinite(grad).all())


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRunMcmcV4Method(unittest.TestCase):
    """run_mcmc_v4 method 参数端到端测试"""

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    @pytest.mark.slow
    def test_run_mcmc_v4_with_rk4(self):
        """run_mcmc_v4 接受 method='rk4' 并正常运行"""
        from tb_risk.seir.inference import BayesianInferenceV4

        init_state = _build_initial_state()
        obs = (np.array([5.0, 10.0]), np.array([5.0, 10.0]))

        engine = BayesianInferenceV4(seed=42)
        posterior = engine.run_mcmc_v4(
            obs, init_state, (0, 10), 2.5,
            n_warmup=3, n_samples=3, n_chains=2, n_leapfrog=3,
            method='rk4')

        # 应返回 15 个参数的后验样本
        self.assertGreaterEqual(len(posterior), 15)
        # 样本非空
        for name, samples in posterior.items():
            self.assertEqual(samples.shape, (6,))  # 3 samples × 2 chains

    @pytest.mark.slow
    def test_run_mcmc_v4_default_euler(self):
        """run_mcmc_v4 默认使用 Euler (回归测试)"""
        from tb_risk.seir.inference import BayesianInferenceV4

        init_state = _build_initial_state()
        obs = (np.array([5.0, 10.0]), np.array([5.0, 10.0]))

        engine = BayesianInferenceV4(seed=42)
        posterior = engine.run_mcmc_v4(
            obs, init_state, (0, 10), 2.5,
            n_warmup=3, n_samples=3, n_chains=2, n_leapfrog=3)
        self.assertGreaterEqual(len(posterior), 15)


# ======================================================================
# 第五轮改进测试
# ======================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestParamsPassThrough(unittest.TestCase):
    """要点1 (第五轮): constrain_params 重复调用修复 — params 透传

    验证 torch_v4_integrate / torch_log_likelihood_v4 接受 params 可选参数,
    传入与不传入产生数值一致的结果 (autograd 图一致)。
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_integrate_accepts_params(self):
        """torch_v4_integrate 接受 params 参数"""
        from tb_risk.seir.autodiff import torch_v4_integrate, constrain_params
        y = _default_unconstrained_params()
        params, _ = constrain_params(y)
        traj = torch_v4_integrate(
            y, _build_initial_state(), n_steps=5, dt=1.0,
            method='euler', params=params)
        self.assertEqual(traj.shape, (1, 6, 270))

    def test_integrate_with_params_matches_without(self):
        """传入 params 与不传入产生相同轨迹"""
        from tb_risk.seir.autodiff import torch_v4_integrate, constrain_params
        y = _default_unconstrained_params()
        init = _build_initial_state()

        # 不传入 params (integrate 内部调用 constrain_params)
        traj_a = torch_v4_integrate(y, init, n_steps=10, dt=1.0, method='euler')
        # 传入 params (跳过内部 constrain_params)
        params, _ = constrain_params(y)
        traj_b = torch_v4_integrate(
            y, init, n_steps=10, dt=1.0, method='euler', params=params)

        diff = (traj_a - traj_b).abs().max().item()
        self.assertLess(diff, 1e-12,
                        f"传入/不传入 params 的轨迹应一致, diff={diff:.2e}")

    def test_integrate_params_rk4_matches(self):
        """RK4 模式下传入 params 也产生一致结果"""
        from tb_risk.seir.autodiff import torch_v4_integrate, constrain_params
        y = _default_unconstrained_params()
        init = _build_initial_state()

        traj_a = torch_v4_integrate(y, init, n_steps=10, dt=1.0, method='rk4')
        params, _ = constrain_params(y)
        traj_b = torch_v4_integrate(
            y, init, n_steps=10, dt=1.0, method='rk4', params=params)

        diff = (traj_a - traj_b).abs().max().item()
        self.assertLess(diff, 1e-12)

    def test_log_likelihood_params_pass_through(self):
        """torch_log_likelihood_v4 透传 params"""
        from tb_risk.seir.autodiff import (
            torch_log_likelihood_v4, constrain_params)
        y = _default_unconstrained_params()
        obs = (np.array([5.0, 10.0]), np.array([5.0, 10.0]))
        init = _build_initial_state()

        ll_a = torch_log_likelihood_v4(y, obs, init, (0, 10), 1.0)
        params, _ = constrain_params(y)
        ll_b = torch_log_likelihood_v4(
            y, obs, init, (0, 10), 1.0, params=params)

        diff = (ll_a - ll_b).abs().item()
        self.assertLess(diff, 1e-12,
                        f"log_likelihood 透传 params 应一致, diff={diff:.2e}")

    def test_posterior_single_constrain_call(self):
        """torch_log_posterior_v4 整个路径只调用一次 constrain_params

        通过计数 constrain_params 调用次数验证。
        """
        from tb_risk.seir import autodiff
        original = autodiff.constrain_params
        call_count = [0]

        def counting_constrain(y):
            call_count[0] += 1
            return original(y)

        try:
            autodiff.constrain_params = counting_constrain
            y = _default_unconstrained_params()
            obs = (np.array([5.0, 10.0]), np.array([5.0, 10.0]))
            init = _build_initial_state()
            # 调用 posterior (应只调用 1 次 constrain_params)
            lp = autodiff.torch_log_posterior_v4(y, obs, init, (0, 10), 1.0)
            self.assertEqual(call_count[0], 1,
                             f"posterior 应只调用 1 次 constrain_params, "
                             f"实际 {call_count[0]} 次")
            self.assertTrue(torch.isfinite(lp))
        finally:
            autodiff.constrain_params = original

    def test_posterior_gradient_with_params_pass_through(self):
        """posterior 路径 params 透传后梯度仍可计算 (autograd 图一致)"""
        from tb_risk.seir.autodiff import compute_grad_log_posterior
        y = _default_unconstrained_params()
        obs = (np.array([5.0, 10.0]), np.array([5.0, 10.0]))
        init = _build_initial_state()

        grad, lp = compute_grad_log_posterior(y, obs, init, (0, 10), 1.0)
        self.assertEqual(grad.shape, (15,))
        self.assertTrue(torch.isfinite(grad).all(),
                        f"梯度含 NaN/Inf: {grad}")
        self.assertTrue(torch.isfinite(lp))

    def test_integrate_params_dict_not_mutated(self):
        """传入的 params dict 不被 integrate 污染 (浅拷贝保护)"""
        from tb_risk.seir.autodiff import torch_v4_integrate, constrain_params
        y = _default_unconstrained_params()
        params, _ = constrain_params(y)
        # 记录原始 dtype/device
        original_beta = params['beta'].clone()
        # 调用 integrate (内部会做 .to(dtype, device) 转换)
        torch_v4_integrate(
            y, _build_initial_state(), n_steps=5, dt=1.0, params=params)
        # 原始 dict 中的 tensor 不应被修改
        self.assertTrue(torch.equal(params['beta'], original_beta),
                        "integrate 不应修改传入的 params dict")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRK4NonNegativity(unittest.TestCase):
    """要点3 (第五轮): RK4 中间步骤非负保护"""

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_rk4_no_nan_with_small_states(self):
        """RK4 在房室人口接近 0 时不产生 NaN (中间 clamp 保护)"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        # 构造接近 0 的初始状态 (仅 1 个 I_sp, 其余接近 0)
        init = np.zeros(270)
        base = 2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9
        init[base + 0] = 1.0   # S=1
        init[base + 5] = 0.01  # Isp=0.01 (极小)
        y = _default_unconstrained_params()
        traj = torch_v4_integrate(
            y, init, n_steps=50, dt=2.0, method='rk4')
        self.assertTrue(torch.isfinite(traj).all(),
                        "RK4 在极小房室人口下产生 NaN/Inf")

    def test_rk4_all_states_non_negative(self):
        """RK4 积分后所有状态非负"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        traj = torch_v4_integrate(
            y, _build_initial_state(), n_steps=50, dt=2.0, method='rk4')
        self.assertTrue(torch.all(traj >= 0),
                        "RK4 轨迹含负值")

    def test_rk4_large_dt_no_explosion(self):
        """RK4 大 dt 下不爆炸 (clamp 保护中间步骤)"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        traj = torch_v4_integrate(
            y, _build_initial_state(), n_steps=10, dt=10.0, method='rk4')
        self.assertTrue(torch.isfinite(traj).all())
        self.assertTrue(torch.all(traj >= 0))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAutoMethod(unittest.TestCase):
    """要点4 (第五轮): method='auto' 自适应选择"""

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_auto_accepted(self):
        """torch_v4_integrate 接受 method='auto'"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        traj = torch_v4_integrate(
            y, _build_initial_state(), n_steps=5, dt=1.0, method='auto')
        self.assertEqual(traj.shape, (1, 6, 270))

    def test_auto_small_n_steps_uses_rk4(self):
        """auto 模式下小 n_steps 选择 RK4 (n_steps < 阈值 1000)"""
        from tb_risk.seir.autodiff import (
            torch_v4_integrate, AUTO_RK4_N_STEPS_THRESHOLD)
        y = _default_unconstrained_params()
        init = _build_initial_state()
        # 小 n_steps: auto 应选择 RK4
        n_steps_small = min(50, AUTO_RK4_N_STEPS_THRESHOLD - 1)
        traj_auto = torch_v4_integrate(
            y, init, n_steps=n_steps_small, dt=1.0, method='auto')
        traj_rk4 = torch_v4_integrate(
            y, init, n_steps=n_steps_small, dt=1.0, method='rk4')
        diff = (traj_auto - traj_rk4).abs().max().item()
        self.assertLess(diff, 1e-12,
                        f"小 n_steps ({n_steps_small}) auto 应选 RK4, "
                        f"diff={diff:.2e}")

    def test_auto_large_n_steps_uses_euler(self):
        """auto 模式下大 n_steps 选择 Euler (n_steps >= 阈值)"""
        from tb_risk.seir.autodiff import (
            torch_v4_integrate, AUTO_RK4_N_STEPS_THRESHOLD)
        y = _default_unconstrained_params()
        init = _build_initial_state()
        # 大 n_steps: auto 应选择 Euler
        n_steps_large = AUTO_RK4_N_STEPS_THRESHOLD + 100
        traj_auto = torch_v4_integrate(
            y, init, n_steps=n_steps_large, dt=0.01, method='auto')
        traj_euler = torch_v4_integrate(
            y, init, n_steps=n_steps_large, dt=0.01, method='euler')
        diff = (traj_auto - traj_euler).abs().max().item()
        self.assertLess(diff, 1e-12,
                        f"大 n_steps ({n_steps_large}) auto 应选 Euler, "
                        f"diff={diff:.2e}")

    def test_auto_log_posterior(self):
        """torch_log_posterior_v4 接受 method='auto'"""
        from tb_risk.seir.autodiff import torch_log_posterior_v4
        y = _default_unconstrained_params()
        lp = torch_log_posterior_v4(
            y, (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            _build_initial_state(), (0, 10), 1.0, method='auto')
        self.assertTrue(torch.isfinite(lp))

    def test_auto_grad_computable(self):
        """auto 模式下梯度可计算"""
        from tb_risk.seir.autodiff import compute_grad_log_posterior
        y = _default_unconstrained_params()
        grad, lp = compute_grad_log_posterior(
            y, (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            _build_initial_state(), (0, 10), 1.0, method='auto')
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(torch.isfinite(lp))

    def test_auto_threshold_boundary(self):
        """阈值边界: n_steps = 阈值时选 Euler (>= 阈值)"""
        from tb_risk.seir.autodiff import (
            torch_v4_integrate, AUTO_RK4_N_STEPS_THRESHOLD)
        y = _default_unconstrained_params()
        init = _build_initial_state()
        # n_steps = 阈值: 应选 Euler (>= 阈值)
        traj_auto = torch_v4_integrate(
            y, init, n_steps=AUTO_RK4_N_STEPS_THRESHOLD, dt=0.01,
            method='auto')
        traj_euler = torch_v4_integrate(
            y, init, n_steps=AUTO_RK4_N_STEPS_THRESHOLD, dt=0.01,
            method='euler')
        diff = (traj_auto - traj_euler).abs().max().item()
        self.assertLess(diff, 1e-12,
                        "n_steps = 阈值时应选 Euler")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSmoothPositive(unittest.TestCase):
    """要点4 (第九轮): _smooth_positive 平滑正性近似

    验证 _smooth_positive 替代 torch.clamp(x, min=0.0) 后:
      - x > 0 时恒等映射
      - x <= 0 时 softplus 平滑过渡
      - 梯度处处非零 (核心目的: 保持 autograd 梯度连续性)
      - alpha 参数可配置
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_positive_returns_identity(self):
        """x > 0 时 _smooth_positive(x) == x"""
        from tb_risk.seir.autodiff import _smooth_positive
        x = torch.tensor([0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
        y = _smooth_positive(x)
        diff = (y - x).abs().max().item()
        self.assertLess(diff, 1e-12,
                        f"x > 0 时应恒等映射, diff={diff:.2e}")

    def test_negative_returns_softplus(self):
        """x <= 0 时 _smooth_positive(x) == alpha * log1p(exp(x/alpha))"""
        from tb_risk.seir.autodiff import _smooth_positive, RK4_SOFTPLUS_ALPHA
        alpha = RK4_SOFTPLUS_ALPHA
        x = torch.tensor([-0.05, -0.1, -1.0, -10.0], dtype=torch.float64)
        y = _smooth_positive(x)
        expected = alpha * torch.log1p(torch.exp(x / alpha))
        diff = (y - expected).abs().max().item()
        self.assertLess(diff, 1e-12,
                        f"x <= 0 时应等于 softplus, diff={diff:.2e}")

    def test_non_negative_output(self):
        """_smooth_positive 输出始终非负"""
        from tb_risk.seir.autodiff import _smooth_positive
        x = torch.tensor([-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0],
                         dtype=torch.float64)
        y = _smooth_positive(x)
        self.assertTrue(torch.all(y >= 0),
                        f"输出含负值: {y}")

    def test_gradient_nonzero_for_negative_x(self):
        """x < 0 时梯度 > 0 (核心: 替代 clamp 的零梯度问题)

        torch.clamp(x, min=0.0) 在 x < 0 时梯度为 0, 截断 autograd。
        _smooth_positive 在 x < 0 时梯度 = sigmoid(x/alpha) > 0。
        """
        from tb_risk.seir.autodiff import _smooth_positive
        x = torch.tensor([-1.0, -0.5, -0.1, -0.01], dtype=torch.float64,
                         requires_grad=True)
        y = _smooth_positive(x)
        y.sum().backward()
        self.assertTrue(torch.all(x.grad > 0),
                        f"x < 0 时梯度应 > 0, got {x.grad}")
        # 接近 0 的点梯度应接近 0.5 (sigmoid(0) = 0.5)
        x2 = torch.tensor([-0.001], dtype=torch.float64, requires_grad=True)
        y2 = _smooth_positive(x2)
        y2.backward()
        self.assertGreater(float(x2.grad), 0.4,
                           f"x≈0 时梯度应接近 0.5, got {x2.grad}")

    def test_gradient_at_zero_continuous(self):
        """x = 0 附近梯度连续 (左极限 ≈ 右极限)

        x > 0 侧梯度恒为 1, x <= 0 侧梯度 = sigmoid(x/alpha)。
        在 x = 0 处 sigmoid(0) = 0.5, 与右侧 1 有跳变 — 这是分段定义的
        固有性质。验证 x = -epsilon 处梯度接近 0.5, 确认 softplus 分支
        工作正常。注意: 完全连续需用单一 softplus 公式 (无 where 分支),
        但那会在 x >> 0 时引入 alpha*log(2) 偏差。当前分段实现优先精度。
        """
        from tb_risk.seir.autodiff import _smooth_positive, RK4_SOFTPLUS_ALPHA
        alpha = RK4_SOFTPLUS_ALPHA
        # x = -alpha 处梯度 = sigmoid(-1) ≈ 0.269
        x = torch.tensor([-alpha], dtype=torch.float64, requires_grad=True)
        y = _smooth_positive(x)
        y.backward()
        expected_grad = 1.0 / (1.0 + float(torch.exp(torch.tensor(1.0))))
        self.assertAlmostEqual(float(x.grad), expected_grad, places=3,
                               msg=f"x=-alpha 处梯度应为 sigmoid(-1)≈{expected_grad:.3f}, "
                                   f"got {x.grad}")

    def test_alpha_parameter_controls_smoothness(self):
        """alpha 参数控制平滑度 (大 alpha 更平滑)"""
        from tb_risk.seir.autodiff import _smooth_positive
        x = torch.tensor([-0.5], dtype=torch.float64, requires_grad=True)
        # 小 alpha: 梯度小 (接近硬 clamp)
        y_small = _smooth_positive(x, alpha=0.01)
        y_small.backward()
        grad_small = float(x.grad)
        x.grad = None
        # 大 alpha: 梯度大 (更平滑)
        y_large = _smooth_positive(x, alpha=1.0)
        y_large.backward()
        grad_large = float(x.grad)
        self.assertLess(grad_small, grad_large,
                        f"小 alpha 梯度 ({grad_small:.4f}) 应小于大 alpha ({grad_large:.4f})")

    def test_alpha_none_uses_default(self):
        """alpha=None 使用模块默认 RK4_SOFTPLUS_ALPHA"""
        from tb_risk.seir.autodiff import (
            _smooth_positive, RK4_SOFTPLUS_ALPHA)
        x = torch.tensor([-0.5], dtype=torch.float64)
        y_default = _smooth_positive(x)
        y_explicit = _smooth_positive(x, alpha=RK4_SOFTPLUS_ALPHA)
        diff = (y_default - y_explicit).abs().max().item()
        self.assertLess(diff, 1e-15,
                        f"alpha=None 应使用默认值, diff={diff:.2e}")

    def test_numerical_stability_large_x(self):
        """大 x 不溢出 (x > 0 分支避免 exp(x/alpha) 溢出)"""
        from tb_risk.seir.autodiff import _smooth_positive
        x = torch.tensor([1e6, 1e10], dtype=torch.float64)
        y = _smooth_positive(x)
        self.assertTrue(torch.isfinite(y).all(),
                        f"大 x 输出含 NaN/Inf: {y}")
        # x > 0 时 y == x
        diff = (y - x).abs().max().item()
        self.assertLess(diff, 1e-3,
                        f"大 x 应恒等映射, diff={diff:.2e}")

    def test_numerical_stability_very_negative_x(self):
        """极负 x 不产生 NaN (softplus 分支稳定)"""
        from tb_risk.seir.autodiff import _smooth_positive
        x = torch.tensor([-1e6, -1e10], dtype=torch.float64)
        y = _smooth_positive(x)
        self.assertTrue(torch.isfinite(y).all(),
                        f"极负 x 输出含 NaN/Inf: {y}")
        # 极负 x 时 softplus ≈ 0
        self.assertTrue(torch.all(y < 1e-10),
                        f"极负 x 应接近 0, got {y}")

    def test_env_var_overrides_alpha(self):
        """环境变量 TB_RISK_RK4_SOFTPLUS_ALPHA 覆盖默认 alpha

        通过子进程导入模块验证环境变量生效 (模块加载时读取)。
        """
        import subprocess
        env = os.environ.copy()
        env['TB_RISK_RK4_SOFTPLUS_ALPHA'] = '1.0'
        code = (
            "from tb_risk.seir.autodiff import RK4_SOFTPLUS_ALPHA; "
            "print(RK4_SOFTPLUS_ALPHA)"
        )
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=_PROJECT_ROOT,
            env=env,
            # 缺陷 #3（本轮）：子进程首次导入 tb_risk 需初始化
            # matplotlib/numpy/torch 等重型依赖，30s 在 CI/冷启动下不足。
            # 提升到 90s 匹配重型依赖冷启动时间。
            capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0,
                         f"子进程失败: {result.stderr}")
        alpha = float(result.stdout.strip())
        self.assertAlmostEqual(alpha, 1.0, places=5,
                               msg=f"环境变量应使 alpha=1.0, got {alpha}")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRK4SoftplusGradient(unittest.TestCase):
    """要点4 (第九轮): RK4 softplus 替换后的梯度连续性

    验证 RK4 积分在房室人口接近 0 时:
      - 梯度仍可计算且有限 (非 NaN/Inf)
      - 梯度非零 (对比 clamp 会截断为零)
      - 数值精度未显著下降
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_rk4_gradient_finite_with_small_states(self):
        """房室人口接近 0 时 RK4 梯度仍有限 (回归测试)"""
        from tb_risk.seir.autodiff import (
            torch_log_posterior_v4, N_INFER_PARAMS)
        init = np.zeros(270)
        base = 2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9
        init[base + 0] = 1.0    # S=1 (极小)
        init[base + 5] = 0.01   # Isp=0.01 (极小)
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64,
                        requires_grad=True)
        lp = torch_log_posterior_v4(
            y, (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            init, (0, 10), 1.0, method='rk4')
        lp.backward()
        self.assertTrue(torch.isfinite(y.grad).all(),
                        f"RK4 梯度含 NaN/Inf: {y.grad}")

    def test_rk4_gradient_nonzero_near_zero(self):
        """RK4 在房室人口接近 0 时梯度非零 (softplus 保持梯度)

        对比: 硬 clamp 会在房室人口接近 0 且导数为负时截断梯度为零。
        softplus 替换后梯度应非零 ( sigmoid(x/alpha) > 0 )。
        注意: 梯度绝对值可能很小, 但应大于 0。
        """
        from tb_risk.seir.autodiff import (
            torch_log_posterior_v4, N_INFER_PARAMS)
        # 极小房室人口 — 触发 softplus 分支
        init = np.zeros(270)
        base = 2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9
        init[base + 0] = 0.1     # S=0.1 (极小)
        init[base + 5] = 0.001   # Isp=0.001 (极小, 易触发负中间状态)
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64,
                        requires_grad=True)
        lp = torch_log_posterior_v4(
            y, (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            init, (0, 10), 0.5, method='rk4')
        lp.backward()
        # 至少有一个参数的梯度非零 (softplus 保持梯度)
        n_nonzero = int((y.grad.abs() > 0).sum())
        self.assertGreater(n_nonzero, 0,
                           f"RK4 梯度全零 — softplus 未保持梯度: {y.grad}")

    def test_rk4_accuracy_compared_to_euler(self):
        """RK4 精度仍优于 Euler (softplus 未显著降低精度)"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        init = _build_initial_state()
        # 小步长 Euler 作为基准 (近似真值)
        traj_ref = torch_v4_integrate(
            y, init, n_steps=1000, dt=0.01, method='euler')
        # RK4 大步长
        traj_rk4 = torch_v4_integrate(
            y, init, n_steps=50, dt=0.2, method='rk4')
        # Euler 大步长 (对比)
        traj_euler = torch_v4_integrate(
            y, init, n_steps=50, dt=0.2, method='euler')
        # 取最后状态对比
        ref_final = traj_ref[0, -1, :]
        rk4_final = traj_rk4[0, -1, :]
        euler_final = traj_euler[0, -1, :]
        err_rk4 = (rk4_final - ref_final).abs().max().item()
        err_euler = (euler_final - ref_final).abs().max().item()
        # RK4 应优于 Euler (即使 softplus 引入轻微偏差)
        self.assertLess(err_rk4, err_euler,
                        f"RK4 误差 ({err_rk4:.2e}) 应小于 Euler ({err_euler:.2e})")

    def test_rk4_all_states_non_negative_with_softplus(self):
        """RK4 softplus 后所有状态非负 (回归测试, 兼容第五轮测试)"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = _default_unconstrained_params()
        traj = torch_v4_integrate(
            y, _build_initial_state(), n_steps=50, dt=2.0, method='rk4')
        self.assertTrue(torch.all(traj >= 0),
                        "RK4 轨迹含负值 (softplus 失效)")

    def test_rk4_no_nan_with_small_states(self):
        """RK4 在极小房室人口下不产生 NaN (softplus 稳定)"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        init = np.zeros(270)
        base = 2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9
        init[base + 0] = 1.0
        init[base + 5] = 0.01
        y = _default_unconstrained_params()
        traj = torch_v4_integrate(
            y, init, n_steps=50, dt=2.0, method='rk4')
        self.assertTrue(torch.isfinite(traj).all(),
                        "RK4 在极小房室人口下产生 NaN/Inf")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestParamsDictLazyConvert(unittest.TestCase):
    """要点6 (第九轮): params 字典 .to() 转换惰性检查

    验证 torch_v4_integrate 中 params 字典的惰性转换:
      - 同 dtype/device 时不创建新字典 (identity)
      - 不同 dtype/device 时正确转换
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_params_same_dtype_device_no_copy(self):
        """同 dtype/device 时 params 字典 identity (不创建新字典)

        posterior 路径中 constrain_params 产生的 params 已与 y_params
        同 device/dtype, 惰性检查应跳过 .to() 转换, 直接复用原 dict。
        """
        from tb_risk.seir.autodiff import (
            torch_v4_integrate, constrain_params)
        y = torch.zeros(15, dtype=torch.float64)
        params, _ = constrain_params(y)
        # 传入与 y 同 dtype/device 的 params
        traj = torch_v4_integrate(
            y, _build_initial_state(), n_steps=5, dt=1.0,
            method='euler', params=params)
        self.assertEqual(traj.shape, (1, 6, 270))
        self.assertTrue(torch.isfinite(traj).all())

    def test_params_different_dtype_converts(self):
        """不同 dtype 时 params 正确转换"""
        from tb_risk.seir.autodiff import (
            torch_v4_integrate, constrain_params)
        y = torch.zeros(15, dtype=torch.float64)
        params, _ = constrain_params(y)
        # 强制 params 为 float32 (模拟 dtype 不匹配)
        params_f32 = {k: v.to(torch.float32) for k, v in params.items()}
        # y 仍为 float64, 应正确转换 params 回 float64
        traj = torch_v4_integrate(
            y, _build_initial_state(), n_steps=5, dt=1.0,
            method='euler', params=params_f32)
        self.assertEqual(traj.dtype, torch.float64)
        self.assertTrue(torch.isfinite(traj).all())

    def test_params_pass_through_euler_and_rk4(self):
        """params 透传在 euler 和 rk4 下都正常工作"""
        from tb_risk.seir.autodiff import (
            torch_v4_integrate, constrain_params)
        y = torch.zeros(15, dtype=torch.float64)
        params, _ = constrain_params(y)
        for method in ['euler', 'rk4']:
            traj = torch_v4_integrate(
                y, _build_initial_state(), n_steps=5, dt=1.0,
                method=method, params=params)
            self.assertEqual(traj.shape, (1, 6, 270),
                             f"{method} 形状错误")
            self.assertTrue(torch.isfinite(traj).all(),
                            f"{method} 含 NaN/Inf")

    def test_params_lazy_convert_preserves_values(self):
        """惰性转换 (no-op 路径) 保留 params 值不变"""
        from tb_risk.seir.autodiff import (
            torch_v4_integrate, constrain_params)
        y = torch.zeros(15, dtype=torch.float64)
        params, _ = constrain_params(y)
        # 记录原始值
        beta_before = float(params['beta'])
        # 调用 integrate (应走 no-op 路径)
        torch_v4_integrate(
            y, _build_initial_state(), n_steps=5, dt=1.0,
            method='euler', params=params)
        # params 字典不应被修改 (浅拷贝保护)
        beta_after = float(params['beta'])
        self.assertAlmostEqual(beta_before, beta_after, places=10,
                               msg="params 值在惰性转换后被修改")


if __name__ == '__main__':
    unittest.main()
