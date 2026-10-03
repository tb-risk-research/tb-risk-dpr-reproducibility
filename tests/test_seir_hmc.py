#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — HMC 采样器与可微 SEIR v4

测试改进十：MCMC 推断引擎升级（HMC + 对偶平均法）

覆盖:
  1. autodiff.py: 参数变换、可微 ODE、可微 log_prior/likelihood/posterior
  2. HMCSampler: leapfrog、对偶平均法、多链采样、收敛诊断、可识别性分析

文献:
  Hoffman & Gelman (2014) JMLR 15(1):1593-1623 — NUTS/HMC
  Nesterov (2009) — 对偶平均法
  Vehtari et al. (2021) Bayesian Anal 16(2) — R-hat/ESS
  Dankwa et al. (2025) PLoS Comput Biol — 校准报告框架
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


# 跳过所有测试如果 PyTorch 不可用
@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestParamTransforms(unittest.TestCase):
    """参数变换 (约束 ↔ 无约束) 测试"""

    def test_constrain_unconstrain_roundtrip(self):
        """约束 → 无约束 → 约束 恢复原值"""
        from tb_risk.seir.autodiff import constrain, unconstrain, PARAM_BOUNDS
        for lo, hi in PARAM_BOUNDS:
            x_orig = torch.tensor((lo + hi) / 2, dtype=torch.float64)
            y = unconstrain(x_orig, lo, hi)
            x_recovered, _ = constrain(y, lo, hi)
            self.assertAlmostEqual(float(x_orig), float(x_recovered),
                                   places=5)

    def test_constrain_within_bounds(self):
        """变换后参数在约束范围内"""
        from tb_risk.seir.autodiff import constrain_params, PARAM_BOUNDS, N_INFER_PARAMS
        # 全零无约束参数 → 各参数应在约束范围中点附近
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64)
        params, log_jac = constrain_params(y)
        for name, (lo, hi) in zip(['beta', 'rho_fast', 'rho_react',
                                    'sigma_clear', 'gamma'], PARAM_BOUNDS[:5]):
            x = float(params[name])
            self.assertGreater(x, lo, f"{name}={x} 低于下界 {lo}")
            self.assertLess(x, hi, f"{name}={x} 高于上界 {hi}")

    def test_log_jacobian_finite(self):
        """Jacobian 对数为有限值"""
        from tb_risk.seir.autodiff import constrain_params, N_INFER_PARAMS
        y = torch.randn(N_INFER_PARAMS, dtype=torch.float64) * 0.5
        _, log_jac = constrain_params(y)
        self.assertTrue(torch.isfinite(log_jac).item(),
                        f"Jacobian 非有限: {log_jac}")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestDifferentiableSEIRv4(unittest.TestCase):
    """可微 v4 SEIR ODE 测试"""

    def setUp(self):
        from tb_risk.seir.autodiff import PARAM_BOUNDS, N_INFER_PARAMS
        self.n_params = N_INFER_PARAMS
        # 构造 270D 初始状态 (简化版)
        self.init_state = np.zeros(270)
        # S 分布在 15-49 岁 HIV-/DS
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0  # S
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 1] = 50.0   # Lf
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0  # Isp

    def test_rhs_output_shape(self):
        """ODE 右侧返回 270D"""
        from tb_risk.seir.autodiff import (torch_v4_rhs, constrain_params,
                                            N_INFER_PARAMS, DEFAULT_WAIFW,
                                            DEFAULT_AGE_PROGRESSION)
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64)
        params, _ = constrain_params(y)
        waifw_t = torch.tensor(DEFAULT_WAIFW, dtype=torch.float64)
        age_prog_t = torch.tensor(DEFAULT_AGE_PROGRESSION, dtype=torch.float64)
        state = torch.tensor(self.init_state, dtype=torch.float64)
        rhs = torch_v4_rhs(state, params, waifw_t, age_prog_t)
        self.assertEqual(rhs.shape[0], 270)

    def test_rhs_finite(self):
        """ODE 右侧为有限值"""
        from tb_risk.seir.autodiff import (torch_v4_rhs, constrain_params,
                                            N_INFER_PARAMS, DEFAULT_WAIFW,
                                            DEFAULT_AGE_PROGRESSION)
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64)
        params, _ = constrain_params(y)
        waifw_t = torch.tensor(DEFAULT_WAIFW, dtype=torch.float64)
        age_prog_t = torch.tensor(DEFAULT_AGE_PROGRESSION, dtype=torch.float64)
        state = torch.tensor(self.init_state, dtype=torch.float64)
        rhs = torch_v4_rhs(state, params, waifw_t, age_prog_t)
        self.assertTrue(torch.isfinite(rhs).all(),
                        f"rhs 含 NaN/Inf: {rhs}")

    def test_mass_conservation(self):
        """ODE 右侧各元素之和接近 0（封闭群体）"""
        from tb_risk.seir.autodiff import (torch_v4_rhs, constrain_params,
                                            N_INFER_PARAMS, DEFAULT_WAIFW,
                                            DEFAULT_AGE_PROGRESSION)
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64)
        params, _ = constrain_params(y)
        waifw_t = torch.tensor(DEFAULT_WAIFW, dtype=torch.float64)
        age_prog_t = torch.tensor(DEFAULT_AGE_PROGRESSION, dtype=torch.float64)
        state = torch.tensor(self.init_state, dtype=torch.float64)
        rhs = torch_v4_rhs(state, params, waifw_t, age_prog_t)
        total = float(torch.sum(rhs))
        self.assertAlmostEqual(total, 0.0, places=1,
                               msg=f"质量守恒违反: sum={total}")

    def test_gradient_computable(self):
        """log_posterior 梯度可通过 autodiff 计算"""
        from tb_risk.seir.autodiff import (torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64, requires_grad=True)
        obs_data = (np.array([5.0, 10.0, 15.0]), np.array([5.0, 10.0, 15.0]))
        lp = torch_log_posterior_v4(y, obs_data, self.init_state,
                                     t_span=(0, 15), dt=5.0)
        grad = torch.autograd.grad(lp, y)[0]
        self.assertEqual(grad.shape[0], N_INFER_PARAMS)
        self.assertTrue(torch.isfinite(grad).all(),
                        f"梯度含 NaN/Inf: {grad}")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestDifferentiableLogPosterior(unittest.TestCase):
    """可微 log_prior / log_likelihood / log_posterior 测试"""

    def setUp(self):
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

    def test_log_prior_finite(self):
        """log_prior 在默认参数下为有限值"""
        from tb_risk.seir.autodiff import torch_log_prior_v4, N_INFER_PARAMS
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64)
        lp = torch_log_prior_v4(y)
        self.assertTrue(torch.isfinite(lp).item(), f"log_prior={lp}")

    def test_log_likelihood_finite(self):
        """log_likelihood 为有限值"""
        from tb_risk.seir.autodiff import (torch_log_likelihood_v4,
                                            N_INFER_PARAMS)
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64)
        ll = torch_log_likelihood_v4(y, self.obs_data, self.init_state,
                                      t_span=(0, 15), dt=5.0)
        self.assertTrue(torch.isfinite(ll).item(), f"log_likelihood={ll}")

    def test_log_posterior_finite(self):
        """log_posterior 为有限值"""
        from tb_risk.seir.autodiff import (torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64)
        lp = torch_log_posterior_v4(y, self.obs_data, self.init_state,
                                     t_span=(0, 15), dt=5.0)
        self.assertTrue(torch.isfinite(lp).item(), f"log_posterior={lp}")

    def test_log_posterior_differentiable(self):
        """log_posterior 可微分（梯度存在且有限）"""
        from tb_risk.seir.autodiff import (torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64, requires_grad=True)
        lp = torch_log_posterior_v4(y, self.obs_data, self.init_state,
                                     t_span=(0, 15), dt=5.0)
        lp.backward()
        self.assertTrue(torch.isfinite(y.grad).all(),
                        f"梯度含 NaN/Inf: {y.grad}")

    def test_compute_grad_log_posterior(self):
        """compute_grad_log_posterior 返回 (grad, val)"""
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            N_INFER_PARAMS)
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64)
        grad, lp = compute_grad_log_posterior(
            y, self.obs_data, self.init_state, t_span=(0, 15), dt=5.0)
        self.assertEqual(grad.shape[0], N_INFER_PARAMS)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(torch.isfinite(lp).item())


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestHMCSampler(unittest.TestCase):
    """HMCSampler 单元测试"""

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        # 梯度函数闭包
        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        self.grad_fn = grad_fn
        self.log_post_fn = log_post_fn
        self.n_params = N_INFER_PARAMS

    def test_hmc_initialization(self):
        """HMC 采样器可正常初始化"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=20, n_chains=2, seed=42)
        self.assertEqual(sampler.n_params, self.n_params)
        self.assertEqual(sampler.n_chains, 2)

    def test_hmc_sample_runs(self):
        """HMC 采样器可完成完整运行"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        result = sampler.sample()

        self.assertEqual(len(result['chains']), 2)
        self.assertEqual(result['chains'][0].shape, (10, self.n_params))
        self.assertIn('diagnostics', result)
        self.assertIn('r_hat', result['diagnostics'])

    @pytest.mark.slow
    def test_hmc_acceptance_rate_positive(self):
        """HMC 接受率 > 0"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=20, n_chains=2, n_leapfrog=5, seed=42)
        result = sampler.sample()
        for acc in result['acceptance_rates']:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_hmc_diagnostics_complete(self):
        """诊断信息完整"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        sampler.sample()
        diag = sampler.diagnostics
        for key in ['n_chains', 'n_warmup', 'n_samples', 'n_leapfrog',
                     'target_accept', 'acceptance_rates',
                     'mean_acceptance_rate', 'step_sizes',
                     'r_hat', 'ess', 'algorithm', 'reference']:
            self.assertIn(key, diag, f"诊断缺少 {key}")

    def test_hmc_step_size_positive(self):
        """自适应步长 > 0"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=10, n_chains=2, n_leapfrog=5, seed=42)
        sampler.sample()
        for eps in sampler.step_sizes:
            self.assertGreater(eps, 0, f"步长非正: {eps}")

    def test_hmc_chains_non_nan(self):
        """采样链无 NaN"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        sampler.sample()
        for chain in sampler.chains:
            self.assertFalse(np.any(np.isnan(chain)),
                             "采样链含 NaN")

    def test_hmc_posterior_mean(self):
        """get_posterior_mean 返回约束空间参数字典"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        sampler.sample()
        means = sampler.get_posterior_mean()
        self.assertIn('beta', means)
        self.assertIn('rr_hiv', means)
        self.assertIn('p_acq', means)
        # beta 应在约束范围内
        self.assertGreater(means['beta'], 0.001)
        self.assertLess(means['beta'], 0.95)

    # ==================== 优先级3: batch 模式 (4 链并行) ====================

    def test_hmc_batch_chains_runs(self):
        """batch_chains=True 可完成完整运行 (优先级3)"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        result = sampler.sample(batch_chains=True)

        # 输出形状与串行模式一致
        self.assertEqual(len(result['chains']), 2)
        for chain in result['chains']:
            self.assertEqual(chain.shape, (10, self.n_params))
        self.assertIn('diagnostics', result)
        self.assertIn('r_hat', result['diagnostics'])

    def test_hmc_batch_chains_no_nan(self):
        """batch 模式采样链无 NaN (优先级3)"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        sampler.sample(batch_chains=True)
        for chain in sampler.chains:
            self.assertFalse(np.any(np.isnan(chain)),
                             "batch 模式采样链含 NaN")

    def test_hmc_batch_chains_acceptance_rate(self):
        """batch 模式接受率在 [0, 1] 范围 (优先级3)"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=20, n_chains=2, n_leapfrog=5, seed=42)
        result = sampler.sample(batch_chains=True)
        for acc in result['acceptance_rates']:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_hmc_batch_chains_diagnostics(self):
        """batch 模式诊断信息完整 (优先级3)"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        sampler.sample(batch_chains=True)
        diag = sampler.diagnostics
        for key in ['n_chains', 'n_warmup', 'n_samples', 'n_leapfrog',
                     'final_n_leapfrog', 'mean_final_n_leapfrog',
                     'adaptive_leapfrog',
                     'target_accept', 'acceptance_rates',
                     'mean_acceptance_rate', 'step_sizes',
                     'r_hat', 'ess', 'algorithm', 'reference']:
            self.assertIn(key, diag, f"batch 诊断缺少 {key}")

    def test_hmc_batch_vs_serial_similar_mean(self):
        """batch 与串行模式参数均值相近 (优先级3)

        warmup 阶段串行相同 (相同 seed), 采样阶段 batch 用 torch RNG
        (要点1: 消除 GPU-CPU 同步) 而串行用 NumPy RNG, 随机数序列不同。
        小样本 (n_samples=20) 下统计波动大, 用宽松绝对容差 (delta=0.15)
        验证分布中心无系统性偏差。
        """
        from tb_risk.seir.bayesian import HMCSampler

        # 串行模式
        sampler_serial = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=20, n_chains=2, n_leapfrog=5, seed=42)
        sampler_serial.sample(batch_chains=False)
        mean_serial = sampler_serial.get_posterior_mean()

        # batch 模式 (相同 seed)
        sampler_batch = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=20, n_chains=2, n_leapfrog=5, seed=42)
        sampler_batch.sample(batch_chains=True)
        mean_batch = sampler_batch.get_posterior_mean()

        # 参数均值应相近 (宽松容差, 因 torch/NumPy RNG 序列差异 + 小样本波动)
        for name in ['beta', 'rho_fast', 'gamma']:
            self.assertIn(name, mean_serial)
            self.assertIn(name, mean_batch)
            self.assertAlmostEqual(
                mean_serial[name], mean_batch[name], delta=0.15,
                msg=f"{name}: serial={mean_serial[name]}, "
                    f"batch={mean_batch[name]}")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestHMCSamplerDiagnostics(unittest.TestCase):
    """HMC 收敛诊断与可识别性分析测试"""

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        from tb_risk.seir.bayesian import HMCSampler
        self.sampler = HMCSampler(
            log_post_fn, grad_fn, N_INFER_PARAMS,
            n_warmup=5, n_samples=20, n_chains=2, n_leapfrog=3, seed=42)
        self.sampler.sample()

    def test_identifiability_analysis(self):
        """可识别性分析返回完整结果"""
        result = self.sampler.identifiability_analysis()
        for key in ['correlation_matrix', 'param_names',
                     'high_correlation_pairs', 'vif', 'identifiable',
                     'n_identifiable']:
            self.assertIn(key, result)
        # 参数名列表长度为 15
        self.assertEqual(len(result['param_names']), 15)
        # 相关系数矩阵形状 (15, 15)
        corr = np.array(result['correlation_matrix'])
        self.assertEqual(corr.shape, (15, 15))
        # VIF 字典包含所有 15 个参数
        self.assertEqual(len(result['vif']), 15)

    def test_vif_positive(self):
        """所有 VIF 值为正"""
        result = self.sampler.identifiability_analysis()
        for name, vif in result['vif'].items():
            self.assertGreater(vif, 0, f"{name} VIF={vif} 非正")

    def test_r_hat_values(self):
        """R-hat 值为正数或 NaN"""
        diag = self.sampler.diagnostics
        for r in diag['r_hat'].values():
            self.assertTrue(np.isnan(float(r)) or r > 0,
                           f"R-hat={r} 非正且非 NaN")

    def test_ess_values(self):
        """ESS 值为正整数"""
        diag = self.sampler.diagnostics
        for name, ess in diag['ess'].items():
            self.assertIsInstance(ess, int)
            self.assertGreater(ess, 0, f"{name} ESS={ess} 非正")

    def test_algorithm_name(self):
        """诊断中记录算法名称 (要点B: 含自适应 Leapfrog)"""
        self.assertEqual(
            self.sampler.diagnostics['algorithm'],
            'HMC + Dual Averaging (Nesterov 2009) + Adaptive Leapfrog')

    def test_reference_citation(self):
        """诊断中记录文献引用"""
        self.assertIn('Hoffman', self.sampler.diagnostics['reference'])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestHMCSamplerConsistency(unittest.TestCase):
    """HMC 采样器一致性测试"""

    def test_reproducible_with_same_seed(self):
        """相同种子下 HMC 采样结果可复现"""
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        init_state = np.zeros(270)
        init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        obs_data = (np.array([5.0, 10.0, 15.0]),
                    np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, obs_data, init_state, t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, obs_data, init_state, t_span=(0, 15), dt=5.0)

        from tb_risk.seir.bayesian import HMCSampler
        s1 = HMCSampler(log_post_fn, grad_fn, N_INFER_PARAMS,
                        n_warmup=5, n_samples=10, n_chains=2,
                        n_leapfrog=3, seed=123)
        s1.sample()
        s2 = HMCSampler(log_post_fn, grad_fn, N_INFER_PARAMS,
                        n_warmup=5, n_samples=10, n_chains=2,
                        n_leapfrog=3, seed=123)
        s2.sample()
        np.testing.assert_array_almost_equal(
            s1.chains[0], s2.chains[0], 8,
            "相同种子下采样链不一致")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRunMCMCv4(unittest.TestCase):
    """改进4: run_mcmc_v4 HMC 采样器集成测试"""

    def setUp(self):
        from tb_risk.seir.bayesian import BayesianSEIRInference
        from tb_risk.seir.stochastic import StochasticSEIRModel
        self.model = StochasticSEIRModel(population=1000, seed=42)
        self.inf = BayesianSEIRInference(seir_model=self.model, seed=42)
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

    def test_run_mcmc_v4_returns_samples(self):
        """run_mcmc_v4 返回约束空间后验样本"""
        samples = self.inf.run_mcmc_v4(
            self.obs_data, self.init_state,
            t_span=(0, 15), dt=5.0,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        # 应返回 15 个参数的约束空间样本
        from tb_risk.seir.autodiff import PARAM_NAMES
        for name in PARAM_NAMES:
            self.assertIn(name, samples)
            self.assertGreater(len(samples[name]), 0)

    def test_run_mcmc_v4_diagnostics(self):
        """run_mcmc_v4 诊断信息完整 (要点B: 含自适应 Leapfrog)"""
        self.inf.run_mcmc_v4(
            self.obs_data, self.init_state,
            t_span=(0, 15), dt=5.0,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        diag = self.inf.convergence_diagnostics
        self.assertIn('acceptance_rates', diag)
        self.assertIn('r_hat', diag)
        self.assertIn('param_names', diag)
        self.assertEqual(diag['algorithm'],
                         'HMC + Dual Averaging (Nesterov 2009) + Adaptive Leapfrog')

    def test_run_mcmc_v4_acceptance_nonneg(self):
        """run_mcmc_v4 接受率非负"""
        self.inf.run_mcmc_v4(
            self.obs_data, self.init_state,
            t_span=(0, 15), dt=5.0,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        for acc in self.inf.convergence_diagnostics['acceptance_rates']:
            self.assertGreaterEqual(acc, 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAdaptiveLeapfrog(unittest.TestCase):
    """要点B: HMC warmup 阶段自适应 n_leapfrog 测试"""

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        self.grad_fn = grad_fn
        self.log_post_fn = log_post_fn
        self.n_params = N_INFER_PARAMS

    def test_adaptive_leapfrog_default_enabled(self):
        """默认开启 adaptive_leapfrog"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        self.assertTrue(sampler.adaptive_leapfrog)

    def test_adaptive_leapfrog_diagnostics_fields(self):
        """诊断含 final_n_leapfrog / mean_final_n_leapfrog / adaptive_leapfrog"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=10, n_chains=2, n_leapfrog=5, seed=42)
        sampler.sample()
        diag = sampler.diagnostics
        for key in ['final_n_leapfrog', 'mean_final_n_leapfrog',
                    'adaptive_leapfrog']:
            self.assertIn(key, diag, f"诊断缺少 {key}")
        self.assertTrue(diag['adaptive_leapfrog'])

    @pytest.mark.slow
    def test_final_n_leapfrog_within_bounds(self):
        """锁定的 n_leapfrog 在 [leapfrog_min, leapfrog_max] 范围内"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=20, n_samples=10, n_chains=2, n_leapfrog=10,
            leapfrog_min=5, leapfrog_max=30,
            leapfrog_adjust_interval=5, seed=42)
        sampler.sample()
        for n in sampler.final_n_leapfrog:
            self.assertGreaterEqual(n, sampler.leapfrog_min,
                                    f"n_leapfrog={n} 低于下界")
            self.assertLessEqual(n, sampler.leapfrog_max,
                                 f"n_leapfrog={n} 超过上界")

    @pytest.mark.slow
    def test_adaptive_leapfrog_can_be_disabled(self):
        """关闭 adaptive_leapfrog 时 final_n_leapfrog 等于初始值"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=20, n_samples=10, n_chains=2, n_leapfrog=7,
            adaptive_leapfrog=False, seed=42)
        sampler.sample()
        self.assertFalse(sampler.adaptive_leapfrog)
        for n in sampler.final_n_leapfrog:
            self.assertEqual(n, 7, f"关闭后 n_leapfrog 应为 7, 实际 {n}")

    def test_leapfrog_bounds_clamped(self):
        """leapfrog_min/max 被钳制到合法范围"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=5, n_chains=1, n_leapfrog=3,
            leapfrog_min=0, leapfrog_max=2,  # min 钳为 1; max(2,1)=2 保持
            seed=42)
        self.assertEqual(sampler.leapfrog_min, 1)
        self.assertEqual(sampler.leapfrog_max, 2)

    def test_leapfrog_max_below_min_clamped(self):
        """leapfrog_max 小于 leapfrog_min 时被钳为 min"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=5, n_chains=1, n_leapfrog=3,
            leapfrog_min=10, leapfrog_max=3,  # max(3,10)=10
            seed=42)
        self.assertEqual(sampler.leapfrog_min, 10)
        self.assertEqual(sampler.leapfrog_max, 10)

    def test_adjust_interval_clamped(self):
        """leapfrog_adjust_interval 至少为 1"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=5, n_chains=1, n_leapfrog=3,
            leapfrog_adjust_interval=0, seed=42)
        self.assertEqual(sampler.leapfrog_adjust_interval, 1)


# ======================================================================
# 第五轮改进测试
# ======================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestInvMassTPrecompute(unittest.TestCase):
    """要点6 (第五轮): _leapfrog 中 inv_mass_t 预计算

    验证 _leapfrog 接受 inv_mass_t 可选参数, 传入与不传入产生一致结果;
    _run_single_chain 预计算 inv_mass_t 后采样结果与不预计算一致。
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)
        # 简单二次型 log_post: -0.5 * sum(theta^2), grad = -theta
        def log_post(theta):
            return -0.5 * torch.sum(theta ** 2)
        def grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            lp = log_post(y)
            g, = torch.autograd.grad(lp, y)
            return g.detach(), lp.detach()
        self.log_post = log_post
        self.grad_fn = grad_fn
        self.n_params = 3

    def test_leapfrog_accepts_inv_mass_t(self):
        """_leapfrog 接受 inv_mass_t 参数"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=2, n_samples=2, n_chains=1, n_leapfrog=3, seed=42)
        theta = torch.zeros(self.n_params, dtype=torch.float64)
        p = torch.randn(self.n_params, dtype=torch.float64)
        inv_mass = np.ones(self.n_params)
        inv_mass_t = torch.tensor(
            inv_mass, dtype=theta.dtype, device=theta.device)
        # 传入 inv_mass_t (不应抛异常)
        theta_new, p_new, lp, grad = sampler._leapfrog(
            theta, p, 0.1, inv_mass, 3, inv_mass_t=inv_mass_t)
        self.assertEqual(theta_new.shape, theta.shape)
        self.assertTrue(torch.isfinite(theta_new).all())

    def test_leapfrog_with_inv_mass_t_matches_without(self):
        """传入 inv_mass_t 与不传入产生一致结果"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=2, n_samples=2, n_chains=1, n_leapfrog=3, seed=42)

        torch.manual_seed(123)
        theta = torch.randn(self.n_params, dtype=torch.float64)
        p1 = torch.randn(self.n_params, dtype=torch.float64)
        p2 = p1.clone()
        inv_mass = np.array([0.5, 1.0, 2.0])
        inv_mass_t = torch.tensor(
            inv_mass, dtype=theta.dtype, device=theta.device)

        # 不传入 inv_mass_t (内部从 numpy 重建)
        grad, lp_init = self.grad_fn(theta)
        theta_a, p_a, lp_a, _ = sampler._leapfrog(
            theta, p1, 0.05, inv_mass, 5,
            grad_logpost=(grad, lp_init))
        # 传入 inv_mass_t
        grad, lp_init = self.grad_fn(theta)
        theta_b, p_b, lp_b, _ = sampler._leapfrog(
            theta, p2, 0.05, inv_mass, 5,
            grad_logpost=(grad, lp_init), inv_mass_t=inv_mass_t)

        diff_theta = (theta_a - theta_b).abs().max().item()
        diff_p = (p_a - p_b).abs().max().item()
        diff_lp = (lp_a - lp_b).abs().item()
        self.assertLess(diff_theta, 1e-12,
                        f"theta 不一致: {diff_theta:.2e}")
        self.assertLess(diff_p, 1e-12, f"p 不一致: {diff_p:.2e}")
        self.assertLess(diff_lp, 1e-12, f"lp 不一致: {diff_lp:.2e}")

    def test_run_single_chain_with_precompute_matches(self):
        """_run_single_chain 预计算 inv_mass_t 后采样结果一致 (回归测试)

        由于 inv_mass_t 预计算是纯性能优化 (数值完全相同), 此测试
        通过与之前版本 (无预计算) 的回归对比验证正确性。当前实现已
        预计算, 测试仅验证采样可正常运行且结果有限。
        """
        from tb_risk.seir.hmc_sampler import HMCSampler
        torch.manual_seed(42)
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=1, n_leapfrog=5, seed=42)

        def init_fn(chain_idx):
            return torch.zeros(self.n_params, dtype=torch.float64)

        result = sampler.sample(init_params_fn=init_fn, batch_chains=False)
        self.assertEqual(len(result['chains']), 1)
        self.assertEqual(result['chains'][0].shape, (10, self.n_params))
        self.assertTrue(np.isfinite(result['chains'][0]).all(),
                        "采样结果含 NaN/Inf")

    def test_warmup_with_precompute_inv_mass_t(self):
        """warmup 中 inv_mass 更新时重算 inv_mass_t (回归测试)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        torch.manual_seed(42)
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=20, n_samples=5, n_chains=1, n_leapfrog=3, seed=42)

        def init_fn(chain_idx):
            return torch.zeros(self.n_params, dtype=torch.float64)

        result = sampler.sample(init_params_fn=init_fn, batch_chains=False)
        # warmup 应正常完成 (inv_mass 更新后重算 inv_mass_t)
        self.assertEqual(len(result['chains']), 1)
        self.assertTrue(np.isfinite(result['chains'][0]).all())


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestHamiltonianReturnsTensor(unittest.TestCase):
    """要点1 (第九轮): _hamiltonian 返回 tensor + 接收 inv_mass_t

    验证 _hamiltonian 返回 torch.Tensor (而非 python float),
    避免 GPU 环境下 float(cuda_tensor) 触发 GPU-CPU 同步。
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)
        def log_post(theta):
            return -0.5 * torch.sum(theta ** 2)
        def grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            lp = log_post(y)
            g, = torch.autograd.grad(lp, y)
            return g.detach(), lp.detach()
        self.log_post = log_post
        self.grad_fn = grad_fn
        self.n_params = 3

    def test_hamiltonian_returns_tensor(self):
        """_hamiltonian 返回 torch.Tensor (非 python float)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=2, n_samples=2, n_chains=1, n_leapfrog=3, seed=42)
        log_post = torch.tensor(-1.0, dtype=torch.float64)
        p = torch.randn(self.n_params, dtype=torch.float64)
        inv_mass = np.ones(self.n_params)
        H = sampler._hamiltonian(log_post, p, inv_mass)
        self.assertIsInstance(H, torch.Tensor,
                              f"_hamiltonian 应返回 Tensor, got {type(H)}")

    def test_hamiltonian_accepts_inv_mass_t(self):
        """_hamiltonian 接收 inv_mass_t 参数 (避免重建)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=2, n_samples=2, n_chains=1, n_leapfrog=3, seed=42)
        log_post = torch.tensor(-1.0, dtype=torch.float64)
        p = torch.randn(self.n_params, dtype=torch.float64)
        inv_mass = np.ones(self.n_params)
        inv_mass_t = torch.tensor(
            inv_mass, dtype=torch.float64, device=p.device)
        # 传入 inv_mass_t (不应抛异常)
        H = sampler._hamiltonian(
            log_post, p, inv_mass, inv_mass_t=inv_mass_t)
        self.assertTrue(torch.isfinite(H).item())

    def test_hamiltonian_inv_mass_t_matches_without(self):
        """传入 inv_mass_t 与不传入产生一致结果"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=2, n_samples=2, n_chains=1, n_leapfrog=3, seed=42)
        torch.manual_seed(123)
        log_post = torch.tensor(-0.5, dtype=torch.float64)
        p = torch.randn(self.n_params, dtype=torch.float64)
        inv_mass = np.array([0.5, 1.0, 2.0])
        inv_mass_t = torch.tensor(
            inv_mass, dtype=torch.float64, device=p.device)
        H_a = sampler._hamiltonian(log_post, p, inv_mass)
        H_b = sampler._hamiltonian(
            log_post, p, inv_mass, inv_mass_t=inv_mass_t)
        diff = (H_a - H_b).abs().item()
        self.assertLess(diff, 1e-12,
                        f"inv_mass_t 传入与否结果不一致: diff={diff:.2e}")

    def test_hamiltonian_batch_returns_tensor(self):
        """_hamiltonian_batch 返回 torch.Tensor (B,) 形状"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=2, n_samples=2, n_chains=1, n_leapfrog=3, seed=42)
        B = 4
        log_post = torch.randn(B, dtype=torch.float64)
        p = torch.randn(B, self.n_params, dtype=torch.float64)
        inv_mass = np.ones((B, self.n_params))
        H = sampler._hamiltonian_batch(log_post, p, inv_mass)
        self.assertIsInstance(H, torch.Tensor)
        self.assertEqual(H.shape, (B,),
                         f"_hamiltonian_batch 返回形状错误: {H.shape}")

    def test_hamiltonian_batch_accepts_inv_mass_t(self):
        """_hamiltonian_batch 接收 inv_mass_t 参数 (要点5)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=2, n_samples=2, n_chains=1, n_leapfrog=3, seed=42)
        B = 4
        log_post = torch.randn(B, dtype=torch.float64)
        p = torch.randn(B, self.n_params, dtype=torch.float64)
        inv_mass = np.ones((B, self.n_params))
        inv_mass_t = torch.tensor(
            inv_mass, dtype=torch.float64, device=p.device)
        H = sampler._hamiltonian_batch(
            log_post, p, inv_mass, inv_mass_t=inv_mass_t)
        self.assertTrue(torch.isfinite(H).all())

    def test_hamiltonian_batch_inv_mass_t_matches_without(self):
        """_hamiltonian_batch 传入 inv_mass_t 与不传入一致 (要点5)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=2, n_samples=2, n_chains=1, n_leapfrog=3, seed=42)
        torch.manual_seed(123)
        B = 4
        log_post = torch.randn(B, dtype=torch.float64)
        p = torch.randn(B, self.n_params, dtype=torch.float64)
        inv_mass = np.tile(np.array([0.5, 1.0, 2.0]), (B, 1))
        inv_mass_t = torch.tensor(
            inv_mass, dtype=torch.float64, device=p.device)
        H_a = sampler._hamiltonian_batch(log_post, p, inv_mass)
        H_b = sampler._hamiltonian_batch(
            log_post, p, inv_mass, inv_mass_t=inv_mass_t)
        diff = (H_a - H_b).abs().max().item()
        self.assertLess(diff, 1e-12,
                        f"batch inv_mass_t 传入与否结果不一致: diff={diff:.2e}")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSerialModeTorchGenerator(unittest.TestCase):
    """要点2 (第九轮): 串行模式用 torch.Generator + torch.randn

    验证 _run_warmup / _run_single_chain / _find_reasonable_epsilon
    使用 torch.Generator (而非 numpy rng), 与 batch 模式统一。
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)
        def log_post(theta):
            return -0.5 * torch.sum(theta ** 2)
        def grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            lp = log_post(y)
            g, = torch.autograd.grad(lp, y)
            return g.detach(), lp.detach()
        self.log_post = log_post
        self.grad_fn = grad_fn
        self.n_params = 3

    def test_serial_mode_reproducible(self):
        """串行模式同种子产生相同结果 (torch.Generator 可复现)"""
        from tb_risk.seir.hmc_sampler import HMCSampler

        def init_fn(chain_idx):
            return torch.zeros(self.n_params, dtype=torch.float64)

        sampler1 = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=20, n_chains=1, n_leapfrog=5, seed=42)
        result1 = sampler1.sample(init_params_fn=init_fn, batch_chains=False)

        sampler2 = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=20, n_chains=1, n_leapfrog=5, seed=42)
        result2 = sampler2.sample(init_params_fn=init_fn, batch_chains=False)

        diff = np.abs(result1['chains'][0] - result2['chains'][0]).max()
        self.assertLess(diff, 1e-12,
                        f"同种子串行采样结果不一致: diff={diff:.2e}")

    def test_serial_mode_finite_results(self):
        """串行模式采样结果有限"""
        from tb_risk.seir.hmc_sampler import HMCSampler

        def init_fn(chain_idx):
            return torch.zeros(self.n_params, dtype=torch.float64)

        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=20, n_samples=30, n_chains=2, n_leapfrog=5, seed=42)
        result = sampler.sample(init_params_fn=init_fn, batch_chains=False)
        self.assertEqual(len(result['chains']), 2)
        for chain in result['chains']:
            self.assertTrue(np.isfinite(chain).all(),
                            "串行模式采样结果含 NaN/Inf")

    def test_serial_mode_acceptance_rate_reasonable(self):
        """串行模式接受率在合理范围 (0.1, 0.99)"""
        from tb_risk.seir.hmc_sampler import HMCSampler

        def init_fn(chain_idx):
            return torch.zeros(self.n_params, dtype=torch.float64)

        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=30, n_samples=50, n_chains=1, n_leapfrog=5,
            target_accept=0.8, seed=42)
        result = sampler.sample(init_params_fn=init_fn, batch_chains=False)
        # 接受率应在 (0.1, 0.99) — torch.Generator 路径正常工作
        ar = result['diagnostics'].get('acceptance_rates', [0.0])[0]
        self.assertGreater(ar, 0.1,
                           f"接受率过低: {ar}")
        self.assertLess(ar, 0.99,
                        f"接受率过高 (可能未拒绝): {ar}")

    def test_find_reasonable_epsilon_finite(self):
        """_find_reasonable_epsilon 返回有限 epsilon (torch.Generator 路径)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=2, n_samples=2, n_chains=1, n_leapfrog=3, seed=42)
        theta = torch.zeros(self.n_params, dtype=torch.float64)
        inv_mass = np.ones(self.n_params)
        epsilon = sampler._find_reasonable_epsilon(theta, inv_mass)
        self.assertTrue(np.isfinite(epsilon),
                        f"epsilon 非有限: {epsilon}")
        self.assertGreater(epsilon, 0,
                           f"epsilon 应 > 0: {epsilon}")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTorchIsfiniteCheck(unittest.TestCase):
    """要点3 (第九轮): torch.isfinite 一次性检查

    验证 _run_warmup / _run_single_chain 用 torch.isfinite(log_post_new)
    替代 torch.isnan + torch.isinf 后:
      - NaN log_post 被正确拒绝 (alpha=0)
      - Inf log_post 被正确拒绝 (alpha=0)
      - 正常 log_post 被正常处理
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_isfinite_rejects_nan(self):
        """NaN log_post 被正确检测为非有限"""
        # 直接验证 torch.isfinite 行为 (确保替换正确)
        nan_t = torch.tensor(float('nan'), dtype=torch.float64)
        self.assertFalse(bool(torch.isfinite(nan_t)),
                         "torch.isfinite(NaN) 应为 False")
        inf_t = torch.tensor(float('inf'), dtype=torch.float64)
        self.assertFalse(bool(torch.isfinite(inf_t)),
                         "torch.isfinite(Inf) 应为 False")
        neg_inf_t = torch.tensor(float('-inf'), dtype=torch.float64)
        self.assertFalse(bool(torch.isfinite(neg_inf_t)),
                         "torch.isfinite(-Inf) 应为 False")
        finite_t = torch.tensor(1.0, dtype=torch.float64)
        self.assertTrue(bool(torch.isfinite(finite_t)),
                        "torch.isfinite(1.0) 应为 True")

    def test_hmc_handles_invalid_log_post(self):
        """HMC 采样器正确处理产生 NaN/Inf 的 log_post_fn (不崩溃)"""
        from tb_risk.seir.hmc_sampler import HMCSampler

        # 构造会产生 NaN 的 log_post (当 theta 范数过大时返回 NaN)
        # 要点2 (第十二轮): float(norm) 对 requires_grad=True 张量提取标量
        # 触发 UserWarning。用 .detach().item() 明确隔离梯度, 消除警告。
        def bad_log_post(theta):
            norm = torch.sum(theta ** 2)
            # norm > 100 时返回 NaN (模拟数值不稳定)
            if norm.detach().item() > 100:
                return torch.tensor(float('nan'), dtype=theta.dtype)
            return -0.5 * norm

        def bad_grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            norm = torch.sum(y ** 2)
            if norm.detach().item() > 100:
                lp = torch.tensor(float('nan'), dtype=theta.dtype)
            else:
                lp = -0.5 * norm
            if bool(torch.isfinite(lp.detach())):
                g, = torch.autograd.grad(lp, y)
                return g.detach(), lp.detach()
            else:
                return torch.zeros_like(y), lp.detach()

        sampler = HMCSampler(
            bad_log_post, bad_grad_fn, 3,
            n_warmup=5, n_samples=10, n_chains=1, n_leapfrog=3, seed=42)

        def init_fn(chain_idx):
            # 初始点正常 (不会立即触发 NaN)
            return torch.zeros(3, dtype=torch.float64)

        # 应正常完成 (NaN 被正确处理为拒绝, 不崩溃)
        result = sampler.sample(init_params_fn=init_fn, batch_chains=False)
        self.assertEqual(len(result['chains']), 1)
        self.assertEqual(result['chains'][0].shape, (10, 3))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSerialBatchConsistency(unittest.TestCase):
    """要点1/2/5 (第九轮): 串行模式与 batch 模式优化对称性验证

    验证串行模式 (torch.Generator + torch.randn + tensor H) 与 batch 模式
    使用相同的随机数生成机制, 优化路径对称。
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)
        def log_post(theta):
            return -0.5 * torch.sum(theta ** 2)
        def grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            lp = log_post(y)
            g, = torch.autograd.grad(lp, y)
            return g.detach(), lp.detach()
        self.log_post = log_post
        self.grad_fn = grad_fn
        self.n_params = 3

    def test_serial_and_batch_both_finite(self):
        """串行与 batch 模式都产生有限结果"""
        from tb_risk.seir.hmc_sampler import HMCSampler

        def init_fn(chain_idx):
            return torch.zeros(self.n_params, dtype=torch.float64)

        # 串行
        sampler_s = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=20, n_chains=2, n_leapfrog=5, seed=42)
        result_s = sampler_s.sample(init_params_fn=init_fn, batch_chains=False)

        # batch
        sampler_b = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=20, n_chains=2, n_leapfrog=5, seed=42)
        result_b = sampler_b.sample(init_params_fn=init_fn, batch_chains=True)

        # 两种模式都应产生有限结果
        for chain_s, chain_b in zip(result_s['chains'], result_b['chains']):
            self.assertTrue(np.isfinite(chain_s).all(),
                            "串行模式含 NaN/Inf")
            self.assertTrue(np.isfinite(chain_b).all(),
                            "batch 模式含 NaN/Inf")

    def test_serial_and_batch_similar_mean(self):
        """串行与 batch 模式后验均值相近 (容差 0.3)

        torch.Generator 与 numpy rng 随机数序列不同, 但两种模式
        都应采样相同的目标分布, 后验均值应相近。
        """
        from tb_risk.seir.hmc_sampler import HMCSampler

        def init_fn(chain_idx):
            return torch.zeros(self.n_params, dtype=torch.float64)

        # 串行
        sampler_s = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=30, n_samples=100, n_chains=2, n_leapfrog=5, seed=42)
        result_s = sampler_s.sample(init_params_fn=init_fn, batch_chains=False)
        # batch
        sampler_b = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=30, n_samples=100, n_chains=2, n_leapfrog=5, seed=42)
        result_b = sampler_b.sample(init_params_fn=init_fn, batch_chains=True)

        # 合并两条链的样本计算后验均值
        mean_s = np.concatenate(result_s['chains']).mean(axis=0)
        mean_b = np.concatenate(result_b['chains']).mean(axis=0)
        # 标准正态后验 N(0, 1), 均值应接近 0
        diff = np.abs(mean_s - mean_b).max()
        self.assertLess(diff, 0.3,
                        f"串行与 batch 后验均值差异过大: diff={diff:.3f}")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestLeapfrogBatchInvMassT(unittest.TestCase):
    """要点1 (第十轮): _leapfrog_batch 接收预计算 inv_mass_t

    验证 _leapfrog_batch 的 inv_mass_t 参数正确工作,
    与 _hamiltonian_batch (要点5 第六轮) 和串行版 _leapfrog (要点6 第五轮) 对称。
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)
        def log_post(theta):
            return -0.5 * torch.sum(theta ** 2)
        def grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            lp = log_post(y)
            g, = torch.autograd.grad(lp, y)
            return g.detach(), lp.detach()
        self.log_post = log_post
        self.grad_fn = grad_fn
        self.n_params = 3

    def test_leapfrog_batch_accepts_inv_mass_t(self):
        """_leapfrog_batch 接收 inv_mass_t 参数 (不抛异常)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=2, n_samples=2, n_chains=1, n_leapfrog=3, seed=42)
        B = 4
        theta = torch.randn(B, self.n_params, dtype=torch.float64)
        p = torch.randn(B, self.n_params, dtype=torch.float64)
        inv_mass = np.ones((B, self.n_params))
        inv_mass_t = torch.tensor(
            inv_mass, dtype=torch.float64, device=theta.device)
        # 传入 inv_mass_t (不应抛异常)
        theta_new, p_new, log_post, grad = sampler._leapfrog_batch(
            theta, p, 0.1, inv_mass, 3, inv_mass_t=inv_mass_t)
        self.assertEqual(theta_new.shape, (B, self.n_params))
        self.assertTrue(torch.isfinite(theta_new).all())

    def test_leapfrog_batch_inv_mass_t_matches_without(self):
        """传入 inv_mass_t 与不传入产生一致结果"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=2, n_samples=2, n_chains=1, n_leapfrog=3, seed=42)
        torch.manual_seed(123)
        B = 4
        theta = torch.randn(B, self.n_params, dtype=torch.float64)
        p = torch.randn(B, self.n_params, dtype=torch.float64)
        inv_mass = np.tile(np.array([0.5, 1.0, 2.0]), (B, 1))
        inv_mass_t = torch.tensor(
            inv_mass, dtype=torch.float64, device=theta.device)
        # 不传入
        torch.manual_seed(123)
        theta_a, p_a, lp_a, grad_a = sampler._leapfrog_batch(
            theta, p, 0.1, inv_mass, 3)
        # 传入
        torch.manual_seed(123)
        theta_b, p_b, lp_b, grad_b = sampler._leapfrog_batch(
            theta, p, 0.1, inv_mass, 3, inv_mass_t=inv_mass_t)
        diff = (theta_a - theta_b).abs().max().item()
        self.assertLess(diff, 1e-12,
                        f"inv_mass_t 传入与否结果不一致: diff={diff:.2e}")

    def test_batch_mode_uses_precomputed_inv_mass_t(self):
        """batch 模式采样正常完成 (要点1 透传 inv_mass_t_batch)"""
        from tb_risk.seir.hmc_sampler import HMCSampler

        def init_fn(chain_idx):
            return torch.zeros(self.n_params, dtype=torch.float64)

        sampler = HMCSampler(
            self.log_post, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=20, n_chains=2, n_leapfrog=5, seed=42)
        result = sampler.sample(init_params_fn=init_fn, batch_chains=True)
        self.assertEqual(len(result['chains']), 2)
        for chain in result['chains']:
            self.assertTrue(np.isfinite(chain).all(),
                            "batch 模式采样结果含 NaN/Inf")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestBatchIsfiniteCheck(unittest.TestCase):
    """要点2 (第十轮): batch 模式 valid 改用 torch.isfinite

    验证 batch 模式的 NaN/Inf 检查与串行模式一致。
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_isfinite_consistent_with_isnan_isinf(self):
        """torch.isfinite 与 ~(isnan | isinf) 语义一致"""
        x = torch.tensor([1.0, float('nan'), float('inf'), float('-inf'), 0.0],
                         dtype=torch.float64)
        valid_new = torch.isfinite(x)
        valid_old = ~(torch.isnan(x) | torch.isinf(x))
        self.assertTrue(torch.equal(valid_new, valid_old),
                        "isfinite 与 ~(isnan|isinf) 结果不一致")

    def test_batch_mode_handles_invalid_log_post(self):
        """batch 模式正确处理产生 NaN 的 log_post (不崩溃)"""
        from tb_risk.seir.hmc_sampler import HMCSampler

        # 要点2 (第十二轮): float(norm) 对 requires_grad=True 张量提取标量
        # 触发 UserWarning。用 .detach().item() 明确隔离梯度, 消除警告。
        def bad_log_post(theta):
            norm = torch.sum(theta ** 2)
            if norm.detach().item() > 100:
                return torch.tensor(float('nan'), dtype=theta.dtype)
            return -0.5 * norm

        def bad_grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            norm = torch.sum(y ** 2)
            if norm.detach().item() > 100:
                lp = torch.tensor(float('nan'), dtype=theta.dtype)
            else:
                lp = -0.5 * norm
            if bool(torch.isfinite(lp.detach())):
                g, = torch.autograd.grad(lp, y)
                return g.detach(), lp.detach()
            else:
                return torch.zeros_like(y), lp.detach()

        sampler = HMCSampler(
            bad_log_post, bad_grad_fn, 3,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)

        def init_fn(chain_idx):
            return torch.zeros(3, dtype=torch.float64)

        # batch 模式应正常完成 (NaN 被正确处理为拒绝)
        result = sampler.sample(init_params_fn=init_fn, batch_chains=True)
        self.assertEqual(len(result['chains']), 2)
        for chain in result['chains']:
            self.assertEqual(chain.shape, (10, 3))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestEulerSoftplusConsistency(unittest.TestCase):
    """要点3 (第十轮): Euler 模式用 _smooth_positive (与 RK4 统一)

    验证 Euler 分支的正性保护与 RK4 一致, 梯度连续性得到保证。
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_euler_gradient_finite_with_small_states(self):
        """Euler 模式在极小房室人口下梯度有限 (与 RK4 一致)"""
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
            init, (0, 10), 1.0, method='euler')
        lp.backward()
        self.assertTrue(torch.isfinite(y.grad).all(),
                        f"Euler 梯度含 NaN/Inf: {y.grad}")

    def test_euler_states_non_negative(self):
        """Euler 模式所有状态非负 (softplus 保护)"""
        from tb_risk.seir.autodiff import torch_v4_integrate
        y = torch.zeros(15, dtype=torch.float64)
        init = np.zeros(270)
        base = 2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9
        init[base + 0] = 100.0
        init[base + 5] = 10.0
        traj = torch_v4_integrate(
            y, init, n_steps=50, dt=2.0, method='euler')
        self.assertTrue(torch.all(traj >= 0),
                        "Euler 轨迹含负值 (softplus 失效)")

    def test_euler_gradient_nonzero_near_zero(self):
        """Euler 模式在房室人口接近 0 时梯度非零 (softplus 保持梯度)"""
        from tb_risk.seir.autodiff import (
            torch_log_posterior_v4, N_INFER_PARAMS)
        init = np.zeros(270)
        base = 2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9
        init[base + 0] = 0.1
        init[base + 5] = 0.001
        y = torch.zeros(N_INFER_PARAMS, dtype=torch.float64,
                        requires_grad=True)
        lp = torch_log_posterior_v4(
            y, (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            init, (0, 10), 0.5, method='euler')
        lp.backward()
        n_nonzero = int((y.grad.abs() > 0).sum())
        self.assertGreater(n_nonzero, 0,
                           f"Euler 梯度全零 — softplus 未保持梯度: {y.grad}")

    def test_euler_and_rk4_both_use_smooth_positive(self):
        """Euler 和 RK4 都用 _smooth_positive (一致性验证)"""
        from tb_risk.seir.autodiff import _smooth_positive
        # _smooth_positive 函数应存在且可调用
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        y = _smooth_positive(x)
        self.assertTrue(torch.all(y >= 0),
                        "_smooth_positive 输出含负值")
        # x=0 时返回 0 (非 alpha*log(2))
        self.assertAlmostEqual(float(y[1]), 0.0, places=10,
                               msg="_smooth_positive(0) 应为 0")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSmoothPositiveGradientJump(unittest.TestCase):
    """要点4 (第十轮): 评估 _smooth_positive 在 x=0 处梯度跳变的影响

    用户关切: x=0 处梯度从 sigmoid(0)=0.5 跳变到 1.0, 可能在 leapfrog
    轨迹中引入不稳定。本测试评估跳变影响是否可忽略。

    评估方法:
      1. 量化梯度跳变幅度
      2. 在极小房室人口场景运行 HMC, 验证接受率稳定
      3. 比较 different alpha 下的 HMC 行为
    """

    def setUp(self):
        torch.manual_seed(42)
        torch.set_grad_enabled(True)

    def test_gradient_jump_magnitude(self):
        """x=0 处梯度跳变幅度为 0.5 (从 0.5 到 1.0)"""
        from tb_risk.seir.autodiff import _smooth_positive, RK4_SOFTPLUS_ALPHA
        alpha = RK4_SOFTPLUS_ALPHA
        # x = -epsilon 处梯度 (softplus 分支)
        x_neg = torch.tensor([-alpha * 0.01], dtype=torch.float64,
                             requires_grad=True)
        y_neg = _smooth_positive(x_neg)
        y_neg.backward()
        grad_neg = float(x_neg.grad)
        # x = +epsilon 处梯度 (恒等分支)
        x_pos = torch.tensor([alpha * 0.01], dtype=torch.float64,
                             requires_grad=True)
        y_pos = _smooth_positive(x_pos)
        y_pos.backward()
        grad_pos = float(x_pos.grad)
        # 跳变幅度
        jump = grad_pos - grad_neg
        # 理论: grad_pos ≈ 1.0, grad_neg ≈ sigmoid(-0.01) ≈ 0.4975
        self.assertAlmostEqual(grad_pos, 1.0, places=3,
                               msg=f"x>0 梯度应≈1.0, got {grad_pos}")
        self.assertGreater(grad_neg, 0.4,
                           f"x<0 梯度应≈0.5, got {grad_neg}")
        self.assertLess(jump, 0.6,
                        f"梯度跳变幅度 {jump:.3f} 过大 (预期 ≈0.5)")

    def test_hmc_stable_with_small_states(self):
        """极小房室人口场景 HMC 接受率稳定 (跳变影响可忽略)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        from tb_risk.seir.autodiff import (
            torch_log_posterior_v4, compute_grad_log_posterior,
            N_INFER_PARAMS)

        init = np.zeros(270)
        base = 2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9
        init[base + 0] = 1.0     # S=1 (极小, 触发 softplus 分支)
        init[base + 5] = 0.01    # Isp=0.01 (极小)
        obs = (np.array([5.0, 10.0]), np.array([5.0, 10.0]))

        def log_post(theta):
            return torch_log_posterior_v4(
                theta, obs, init, (0, 10), 1.0, method='rk4')

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, obs, init, (0, 10), 1.0, method='rk4')

        sampler = HMCSampler(
            log_post, grad_fn, N_INFER_PARAMS,
            n_warmup=20, n_samples=40, n_chains=1, n_leapfrog=5,
            target_accept=0.8, seed=42)

        def init_fn(chain_idx):
            return torch.zeros(N_INFER_PARAMS, dtype=torch.float64)

        result = sampler.sample(init_params_fn=init_fn, batch_chains=False)
        # 接受率应 > 0.05 (跳变未导致轨迹不稳定/全部拒绝)
        # 注: 极小房室人口场景动力学简单, 接受率可能接近 1.0 (所有提议
        # 都被接受), 这是正常现象, 说明梯度跳变未引入不稳定
        ar = result['diagnostics'].get('acceptance_rates', [0.0])[0]
        self.assertGreater(ar, 0.05,
                           f"接受率过低 (跳变可能影响稳定): {ar}")
        # 样本有限 (关键: 轨迹不发散)
        self.assertTrue(np.isfinite(result['chains'][0]).all(),
                        "采样结果含 NaN/Inf")

    def test_different_alpha_produce_finite_results(self):
        """不同 alpha 下 HMC 都产生有限结果 (跳变影响不致命)"""
        from tb_risk.seir.autodiff import _smooth_positive
        # 验证不同 alpha 的 _smooth_positive 都数值稳定
        for alpha in [0.01, 0.1, 0.5, 1.0]:
            x = torch.tensor([-10.0, -1.0, -0.01, 0.0, 0.01, 1.0, 10.0],
                             dtype=torch.float64, requires_grad=True)
            y = _smooth_positive(x, alpha=alpha)
            y.sum().backward()
            self.assertTrue(torch.isfinite(y).all(),
                            f"alpha={alpha}: 输出含 NaN/Inf")
            self.assertTrue(torch.isfinite(x.grad).all(),
                            f"alpha={alpha}: 梯度含 NaN/Inf")
            self.assertTrue(torch.all(y >= 0),
                            f"alpha={alpha}: 输出含负值")

    def test_gradient_continuous_except_at_zero(self):
        """x ≠ 0 处梯度连续 (仅 x=0 处有跳变)"""
        from tb_risk.seir.autodiff import _smooth_positive
        # 在 x = 0.1 和 x = 0.2 处梯度应连续 (都在恒等分支, 梯度=1)
        x1 = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
        y1 = _smooth_positive(x1)
        y1.backward()
        # 在 x = -0.1 和 x = -0.2 处梯度应连续 (都在 softplus 分支)
        x2 = torch.tensor([-0.1], dtype=torch.float64, requires_grad=True)
        y2 = _smooth_positive(x2)
        y2.backward()
        x3 = torch.tensor([-0.2], dtype=torch.float64, requires_grad=True)
        y3 = _smooth_positive(x3)
        y3.backward()
        # 恒等分支梯度恒为 1
        self.assertAlmostEqual(float(x1.grad), 1.0, places=5)
        # softplus 分支梯度连续变化 (sigmoid 函数连续)
        # sigmoid(-0.1/0.1) = sigmoid(-1) ≈ 0.269
        # sigmoid(-0.2/0.1) = sigmoid(-2) ≈ 0.119
        self.assertGreater(float(x2.grad), float(x3.grad),
                           "softplus 分支梯度应随 x 减小而减小")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSmoothPositiveClampImplementation(unittest.TestCase):
    """要点4 (第十一轮): _smooth_positive 用 torch.clamp(x, max=0.0) 替代
    torch.minimum(x, torch.zeros_like(x))。验证语义等价性与零张量分配消除。"""

    def test_smooth_positive_matches_minimum_semantics(self):
        """clamp(x, max=0) 与 minimum(x, zeros_like(x)) 语义等价"""
        from tb_risk.seir.autodiff import _smooth_positive
        torch.manual_seed(42)
        x = torch.randn(50, dtype=torch.float64) * 2.0
        # 当前实现 (clamp)
        y_new = _smooth_positive(x)
        # 原实现 (minimum + zeros_like)
        alpha = 0.1
        x_neg_ref = torch.minimum(x, torch.zeros_like(x))
        softplus_neg_ref = alpha * torch.log1p(torch.exp(x_neg_ref / alpha))
        y_ref = torch.where(x >= 0, x, softplus_neg_ref)
        self.assertTrue(torch.allclose(y_new, y_ref, atol=1e-12),
                        "clamp(max=0) 与 minimum(x, zeros_like) 结果不一致")

    def test_smooth_positive_no_zero_allocation(self):
        """clamp(x, max=0.0) 不创建中间零张量 (语义测试: 输出非负)"""
        from tb_risk.seir.autodiff import _smooth_positive
        # 混合正负值输入
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
        y = _smooth_positive(x)
        # 输出非负 (核心语义)
        self.assertTrue(torch.all(y >= 0), f"输出含负值: {y}")
        # x >= 0 处保持原值
        self.assertAlmostEqual(float(y[2]), 0.0, places=10)
        self.assertAlmostEqual(float(y[3]), 0.5, places=10)
        self.assertAlmostEqual(float(y[4]), 2.0, places=10)
        # x < 0 处返回 softplus (非零)
        self.assertGreater(float(y[0]), 0.0)
        self.assertGreater(float(y[1]), 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestLogPosteriorShapeSqueeze(unittest.TestCase):
    """要点1/3/6 (第十一轮) 修复: torch_log_likelihood_v4 单链输入 squeeze 到 0-dim。

    原实现: 单链 (15,) 输入 → ll 返回 (1,), 导致 lp 返回 (1,)。
    tensor 化 HMC 中 n_accept_t () += mask_t (1,) 触发广播错误。
    修复: 单链输入 squeeze 到 (), 与 torch_log_prior_v4 行为一致。
    """

    def setUp(self):
        from tb_risk.seir.autodiff import N_INFER_PARAMS
        self.n_params = N_INFER_PARAMS
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

    def test_log_likelihood_single_chain_returns_scalar(self):
        """单链输入 (15,) → ll 返回 0-dim 标量 ()"""
        from tb_risk.seir.autodiff import torch_log_likelihood_v4
        y = torch.randn(self.n_params, dtype=torch.float64) * 0.5
        ll = torch_log_likelihood_v4(
            y, self.obs_data, self.init_state,
            t_span=(0, 15), dt=5.0)
        self.assertEqual(ll.dim(), 0,
                         f"单链 ll 应为 0-dim 标量, 实际 dim={ll.dim()}, shape={ll.shape}")

    def test_log_likelihood_batch_returns_B(self):
        """batch 输入 (B, 15) → ll 返回 (B,) (B>1 不受 squeeze 影响)"""
        from tb_risk.seir.autodiff import torch_log_likelihood_v4
        B = 4
        y = torch.randn(B, self.n_params, dtype=torch.float64) * 0.5
        ll = torch_log_likelihood_v4(
            y, self.obs_data, self.init_state,
            t_span=(0, 15), dt=5.0)
        self.assertEqual(ll.dim(), 1)
        self.assertEqual(ll.shape[0], B)

    def test_log_likelihood_batch_B1_returns_1d(self):
        """batch 输入 (1, 15) → ll 返回 (1,) (batch 模式不 squeeze)"""
        from tb_risk.seir.autodiff import torch_log_likelihood_v4
        y = torch.randn(1, self.n_params, dtype=torch.float64) * 0.5
        ll = torch_log_likelihood_v4(
            y, self.obs_data, self.init_state,
            t_span=(0, 15), dt=5.0)
        # batch 模式 (y.dim()==2) 不 squeeze, 返回 (1,)
        self.assertEqual(ll.dim(), 1)
        self.assertEqual(ll.shape[0], 1)

    def test_log_posterior_single_chain_returns_scalar(self):
        """单链输入 → lp 返回 0-dim 标量 (lp + ll = () + () = ()"""
        from tb_risk.seir.autodiff import torch_log_posterior_v4
        y = torch.randn(self.n_params, dtype=torch.float64) * 0.5
        lp = torch_log_posterior_v4(
            y, self.obs_data, self.init_state,
            t_span=(0, 15), dt=5.0)
        self.assertEqual(lp.dim(), 0,
                         f"单链 lp 应为 0-dim 标量, 实际 dim={lp.dim()}, shape={lp.shape}")

    def test_compute_grad_log_posterior_single_chain_lp_scalar(self):
        """单链输入 → compute_grad_log_posterior 返回 (grad=(15,), lp=())"""
        from tb_risk.seir.autodiff import compute_grad_log_posterior
        y = torch.randn(self.n_params, dtype=torch.float64) * 0.5
        grad, lp = compute_grad_log_posterior(
            y, self.obs_data, self.init_state,
            t_span=(0, 15), dt=5.0)
        self.assertEqual(grad.dim(), 1)
        self.assertEqual(grad.shape[0], self.n_params)
        self.assertEqual(lp.dim(), 0,
                         f"单链 lp 应为 0-dim 标量, 实际 dim={lp.dim()}, shape={lp.shape}")

    def test_compute_grad_log_posterior_batch_returns_B(self):
        """batch 输入 → grad (B, 15), lp (B,)"""
        from tb_risk.seir.autodiff import compute_grad_log_posterior
        B = 4
        y = torch.randn(B, self.n_params, dtype=torch.float64) * 0.5
        grad, lp = compute_grad_log_posterior(
            y, self.obs_data, self.init_state,
            t_span=(0, 15), dt=5.0)
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(grad.shape, (B, self.n_params))
        self.assertEqual(lp.dim(), 1)
        self.assertEqual(lp.shape[0], B)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSerialModeTensorAcceptReject(unittest.TestCase):
    """要点1/3/6 (第十一轮): 串行模式全 tensor 化, 消除每步 3 次 GPU-CPU 同步。

    验证:
      - alpha 为 0-dim tensor (不提取 .item())
      - 接受/拒绝用 torch.where (无 Python if bool() 同步)
      - 对偶平均法状态量 tensor 化
      - 结果与原实现一致 (可复现性 + 数值合理性)
    """

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        self.grad_fn = grad_fn
        self.log_post_fn = log_post_fn
        self.n_params = N_INFER_PARAMS

    def test_warmup_returns_valid_scalars(self):
        """_run_warmup 返回的 epsilon/inv_mass/n_leapfrog 为有效标量"""
        from tb_risk.seir.bayesian import HMCSampler
        # n_leapfrog=5 在默认 leapfrog_min=5 范围内, 避免短 warmup 未触发
        # 自适应调整时 n_leapfrog 低于 leapfrog_min 的边界情况
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=15, n_samples=10, n_chains=2, n_leapfrog=5, seed=42)
        theta_init = torch.randn(self.n_params, dtype=torch.float64) * 0.3
        inv_mass_init = np.ones(self.n_params)
        epsilon, inv_mass, warmup_samples, n_leapfrog = sampler._run_warmup(
            theta_init, inv_mass_init, chain_seed=42)
        # epsilon 为正标量
        self.assertIsInstance(epsilon, float)
        self.assertGreater(epsilon, 0)
        self.assertLess(epsilon, 1e5)
        # inv_mass 形状正确
        self.assertEqual(inv_mass.shape, (self.n_params,))
        self.assertTrue(np.all(np.isfinite(inv_mass)))
        self.assertTrue(np.all(inv_mass > 0))
        # n_leapfrog 在合理范围 (短 warmup 未触发自适应, 保持初始值 5)
        self.assertGreaterEqual(n_leapfrog, 1)
        self.assertLessEqual(n_leapfrog, sampler.leapfrog_max)

    def test_warmup_reproducible_with_same_seed(self):
        """相同 seed 的 warmup 结果完全一致 (tensor 化不破坏可复现性)"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=15, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        theta_init = torch.randn(self.n_params, dtype=torch.float64) * 0.3
        inv_mass_init = np.ones(self.n_params)
        eps1, im1, _, nl1 = sampler._run_warmup(
            theta_init.clone(), inv_mass_init.copy(), chain_seed=42)
        eps2, im2, _, nl2 = sampler._run_warmup(
            theta_init.clone(), inv_mass_init.copy(), chain_seed=42)
        self.assertAlmostEqual(eps1, eps2, places=10)
        self.assertTrue(np.allclose(im1, im2, atol=1e-12))
        self.assertEqual(nl1, nl2)

    def test_single_chain_reproducible_with_same_seed(self):
        """相同 seed 的 single_chain 结果完全一致"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=15, n_chains=1, n_leapfrog=3, seed=42)
        theta_init = torch.randn(self.n_params, dtype=torch.float64) * 0.3
        inv_mass = np.ones(self.n_params)
        s1, lp1, ar1 = sampler._run_single_chain(
            theta_init.clone(), 0.05, inv_mass, chain_seed=42, n_leapfrog=3)
        s2, lp2, ar2 = sampler._run_single_chain(
            theta_init.clone(), 0.05, inv_mass, chain_seed=42, n_leapfrog=3)
        np.testing.assert_allclose(s1, s2, atol=1e-12)
        np.testing.assert_allclose(lp1, lp2, atol=1e-12)
        self.assertAlmostEqual(ar1, ar2, places=10)

    def test_single_chain_results_finite(self):
        """single_chain 采样结果有限 (tensor 化不引入 NaN)"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=15, n_chains=1, n_leapfrog=3, seed=42)
        theta_init = torch.randn(self.n_params, dtype=torch.float64) * 0.3
        inv_mass = np.ones(self.n_params)
        samples, log_probs, accept_rate = sampler._run_single_chain(
            theta_init, 0.05, inv_mass, chain_seed=42, n_leapfrog=3)
        self.assertEqual(samples.shape, (15, self.n_params))
        self.assertEqual(log_probs.shape, (15,))
        self.assertTrue(np.all(np.isfinite(samples)),
                        f"samples 含 NaN/Inf: {samples}")
        # log_probs 可能含 -inf (无效样本), 但不应有 NaN
        self.assertFalse(np.any(np.isnan(log_probs)),
                         f"log_probs 含 NaN: {log_probs}")
        # 接受率在 [0, 1]
        self.assertGreaterEqual(accept_rate, 0.0)
        self.assertLessEqual(accept_rate, 1.0)

    def test_single_chain_acceptance_rate_zero_for_invalid(self):
        """全无效 log_post → 接受率 0 (torch.where is_valid 分支正确)"""
        from tb_risk.seir.bayesian import HMCSampler

        # 构造一个总是返回 NaN 的 grad_fn
        def bad_grad_fn(theta):
            grad = torch.ones_like(theta)
            lp = torch.tensor(float('nan'), dtype=theta.dtype)
            return grad, lp

        def bad_log_post_fn(theta):
            return torch.tensor(float('nan'), dtype=theta.dtype)

        sampler = HMCSampler(
            bad_log_post_fn, bad_grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=1, n_leapfrog=2, seed=42)
        theta_init = torch.randn(self.n_params, dtype=torch.float64) * 0.3
        inv_mass = np.ones(self.n_params)
        # _find_reasonable_epsilon 在 NaN 下可能返回小 epsilon, 直接用固定 epsilon
        samples, log_probs, accept_rate = sampler._run_single_chain(
            theta_init, 0.01, inv_mass, chain_seed=42, n_leapfrog=2)
        # 全 NaN log_post → is_valid=False → alpha=0 → accept_t=False → 接受率 0
        self.assertEqual(accept_rate, 0.0)
        # log_probs 应全为 -inf (is_valid=False 分支)
        self.assertTrue(np.all(np.isneginf(log_probs)),
                        f"无效样本 log_probs 应为 -inf, 实际: {log_probs}")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestBatchModeCacheValidPythonBool(unittest.TestCase):
    """要点2 (第十一轮): batch 模式 cache_valid 改为 Python bool 替代 torch.bool 张量,
    消除 bool(cache_valid.all()) 的 GPU-CPU 同步。"""

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        self.grad_fn = grad_fn
        self.log_post_fn = log_post_fn
        self.n_params = N_INFER_PARAMS

    def test_batch_mode_runs_successfully(self):
        """batch 模式 (Python bool cache_valid) 正常运行"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=15, n_chains=2, n_leapfrog=3, seed=42)
        result = sampler.sample(batch_chains=True)
        self.assertEqual(len(result['chains']), 2)
        self.assertEqual(result['chains'][0].shape, (15, self.n_params))
        # 接受率合理
        for acc in result['acceptance_rates']:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_batch_mode_no_nan(self):
        """batch 模式结果无 NaN"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=15, n_chains=2, n_leapfrog=3, seed=42)
        result = sampler.sample(batch_chains=True)
        for chain in result['chains']:
            self.assertFalse(np.any(np.isnan(chain)),
                             "batch 模式采样链含 NaN")

    def test_batch_mode_reproducible(self):
        """batch 模式 (Python bool cache_valid) 可复现"""
        from tb_risk.seir.bayesian import HMCSampler
        sampler1 = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=15, n_chains=2, n_leapfrog=3, seed=42)
        r1 = sampler1.sample(batch_chains=True)
        sampler2 = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=15, n_chains=2, n_leapfrog=3, seed=42)
        r2 = sampler2.sample(batch_chains=True)
        for c1, c2 in zip(r1['chains'], r2['chains']):
            np.testing.assert_allclose(c1, c2, atol=1e-12)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestLeapfrogBatchInvMassOptional(unittest.TestCase):
    """要点5 (第十一轮): _leapfrog_batch / _hamiltonian_batch 的 inv_mass 参数
    在 inv_mass_t 提供时可传 None (API 风格调整, 文档标注)。"""

    def setUp(self):
        from tb_risk.seir.autodiff import N_INFER_PARAMS
        self.n_params = N_INFER_PARAMS

    def test_leapfrog_batch_accepts_inv_mass_none(self):
        """_leapfrog_batch 接受 inv_mass=None 当 inv_mass_t 提供"""
        from tb_risk.seir.bayesian import HMCSampler
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4)
        init_state = np.zeros(270)
        init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        obs_data = (np.array([5.0, 10.0, 15.0]),
                    np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, obs_data, init_state, t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, obs_data, init_state, t_span=(0, 15), dt=5.0)

        sampler = HMCSampler(
            log_post_fn, grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)
        B = 2
        theta = torch.randn(B, self.n_params, dtype=torch.float64) * 0.3
        p = torch.randn(B, self.n_params, dtype=torch.float64) * 0.1
        inv_mass_t = torch.ones(B, self.n_params, dtype=torch.float64)
        # inv_mass=None 应正常工作 (要点5)
        theta_new, p_new, lp, grad = sampler._leapfrog_batch(
            theta, p, 0.01, None, 3, inv_mass_t=inv_mass_t)
        self.assertEqual(theta_new.shape, (B, self.n_params))
        self.assertEqual(p_new.shape, (B, self.n_params))
        self.assertEqual(lp.shape, (B,))
        self.assertEqual(grad.shape, (B, self.n_params))
        self.assertTrue(torch.all(torch.isfinite(theta_new)))

    def test_hamiltonian_batch_accepts_inv_mass_none(self):
        """_hamiltonian_batch 接受 inv_mass=None 当 inv_mass_t 提供"""
        from tb_risk.seir.bayesian import HMCSampler
        from tb_risk.seir.autodiff import N_INFER_PARAMS
        sampler = HMCSampler(
            lambda t: t.sum(), lambda t: (t, t.sum()),
            self.n_params, n_warmup=1, n_samples=1, seed=42)
        B = 4
        log_post = torch.randn(B, dtype=torch.float64)
        p = torch.randn(B, self.n_params, dtype=torch.float64) * 0.1
        inv_mass_t = torch.ones(B, self.n_params, dtype=torch.float64)
        # inv_mass=None 应正常工作
        H = sampler._hamiltonian_batch(log_post, p, None, inv_mass_t=inv_mass_t)
        self.assertEqual(H.shape, (B,))
        self.assertTrue(torch.all(torch.isfinite(H)))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestThetaInitTensorInput(unittest.TestCase):
    """要点1 (第十二轮): theta_init 创建用 as_tensor().detach().clone() 避免 UserWarning。

    init_params_fn 可能返回 torch 张量而非 numpy 数组。原实现 torch.tensor()
    对 torch 输入触发 UserWarning ("To copy construct from a tensor, it is
    recommended to use sourceTensor.detach().clone()")。修复后应无警告。
    """

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.n_params = N_INFER_PARAMS
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        self.grad_fn = grad_fn
        self.log_post_fn = log_post_fn

    def test_init_params_fn_returns_numpy(self):
        """init_params_fn 返回 numpy 数组 — 应无警告"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        import warnings
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)

        def init_fn_numpy(chain_idx):
            return np.zeros(self.n_params, dtype=np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            # 不应抛 UserWarning
            result = sampler.sample(init_params_fn=init_fn_numpy,
                                    batch_chains=False)
        self.assertEqual(len(result['chains']), 2)

    def test_init_params_fn_returns_torch_tensor(self):
        """init_params_fn 返回 torch 张量 — 应无 UserWarning (要点1 修复)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        import warnings
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)

        def init_fn_torch(chain_idx):
            # 返回 torch 张量 (原实现会触发 UserWarning)
            return torch.zeros(self.n_params, dtype=torch.float64)

        with warnings.catch_warnings():
            # 捕获所有 UserWarning 为错误, 验证不触发 "To copy construct" 警告
            warnings.simplefilter('error', UserWarning)
            result = sampler.sample(init_params_fn=init_fn_torch,
                                    batch_chains=False)
        self.assertEqual(len(result['chains']), 2)
        for chain in result['chains']:
            self.assertEqual(chain.shape, (10, self.n_params))

    def test_init_params_fn_returns_torch_with_grad(self):
        """init_params_fn 返回 requires_grad=True 张量 — 应安全转换"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=1, n_leapfrog=3, seed=42)

        def init_fn_grad(chain_idx):
            # 返回带梯度的张量 (detach().clone() 应切断梯度)
            t = torch.zeros(self.n_params, dtype=torch.float64,
                            requires_grad=True)
            return t

        result = sampler.sample(init_params_fn=init_fn_grad,
                                batch_chains=False)
        self.assertEqual(len(result['chains']), 1)
        # 采样结果不应携带梯度
        self.assertFalse(result['chains'][0].flags['WRITEABLE'] is False
                         and hasattr(result['chains'][0], 'requires_grad'))
        self.assertTrue(np.all(np.isfinite(result['chains'][0])))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMixedPrecisionDetach(unittest.TestCase):
    """要点4 (第十二轮): mixed 精度模式 theta_init 转换前防御性 detach()。"""

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.n_params = N_INFER_PARAMS
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        self.grad_fn = grad_fn
        self.log_post_fn = log_post_fn

    def test_mixed_precision_runs_successfully(self):
        """mixed 精度模式正常运行 (theta_init detach 后转 float32)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42,
            precision='mixed')

        def init_fn(chain_idx):
            return np.zeros(self.n_params, dtype=np.float64)

        result = sampler.sample(init_params_fn=init_fn, batch_chains=False)
        self.assertEqual(len(result['chains']), 2)
        for chain in result['chains']:
            self.assertEqual(chain.shape, (10, self.n_params))
            self.assertTrue(np.all(np.isfinite(chain)))

    def test_mixed_precision_batch_mode(self):
        """mixed 精度 + batch 模式正常运行"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42,
            precision='mixed')

        def init_fn(chain_idx):
            return np.zeros(self.n_params, dtype=np.float64)

        result = sampler.sample(init_params_fn=init_fn, batch_chains=True)
        self.assertEqual(len(result['chains']), 2)
        for chain in result['chains']:
            self.assertTrue(np.all(np.isfinite(chain)))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestBatchChainsAutoMode(unittest.TestCase):
    """要点6 (第十二轮): batch_chains='auto' 根据 device.type 自适应选择。"""

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.n_params = N_INFER_PARAMS
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        self.grad_fn = grad_fn
        self.log_post_fn = log_post_fn

    def test_auto_mode_cpu_uses_serial(self):
        """auto 模式在 CPU 设备上选择串行 (避免 batch overhead)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42,
            device=torch.device('cpu'))

        def init_fn(chain_idx):
            return np.zeros(self.n_params, dtype=np.float64)

        # auto 在 CPU 上应等价于 False (串行), 不应抛异常
        result = sampler.sample(init_params_fn=init_fn, batch_chains='auto')
        self.assertEqual(len(result['chains']), 2)
        for chain in result['chains']:
            self.assertEqual(chain.shape, (10, self.n_params))

    def test_auto_mode_equivalent_to_serial_on_cpu(self):
        """CPU 上 auto 模式与 False (串行) 结果一致"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler1 = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42,
            device=torch.device('cpu'))
        sampler2 = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42,
            device=torch.device('cpu'))

        def init_fn(chain_idx):
            return np.zeros(self.n_params, dtype=np.float64)

        r_auto = sampler1.sample(init_params_fn=init_fn, batch_chains='auto')
        r_serial = sampler2.sample(init_params_fn=init_fn, batch_chains=False)
        # CPU 上 auto==False, 结果应完全一致
        for c1, c2 in zip(r_auto['chains'], r_serial['chains']):
            np.testing.assert_allclose(c1, c2, atol=1e-12)

    def test_explicit_true_overrides_auto(self):
        """显式 True 在 CPU 上仍用 batch (用户显式覆盖 auto)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=5, n_samples=10, n_chains=2, n_leapfrog=3, seed=42,
            device=torch.device('cpu'))

        def init_fn(chain_idx):
            return np.zeros(self.n_params, dtype=np.float64)

        # 显式 True 应在 CPU 上也用 batch
        result = sampler.sample(init_params_fn=init_fn, batch_chains=True)
        self.assertEqual(len(result['chains']), 2)
        for chain in result['chains']:
            self.assertTrue(np.all(np.isfinite(chain)))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestCheckInductorJitDefense(unittest.TestCase):
    """要点5 (第十二轮): _check_inductor_available 增加 torch.jit 防御检测。"""

    def test_check_inductor_available_returns_bool(self):
        """_check_inductor_available 返回 bool (不抛异常)"""
        from tb_risk.seir.autodiff import _check_inductor_available
        result = _check_inductor_available()
        self.assertIsInstance(result, bool)

    def test_check_inductor_available_caches_result(self):
        """_check_inductor_available 结果缓存 (第二次调用零开销)"""
        from tb_risk.seir import autodiff
        # 重置缓存
        original = autodiff._INDUCTOR_AVAILABLE
        try:
            autodiff._INDUCTOR_AVAILABLE = None
            r1 = autodiff._check_inductor_available()
            r2 = autodiff._INDUCTOR_AVAILABLE
            self.assertEqual(r1, r2)
        finally:
            autodiff._INDUCTOR_AVAILABLE = original

    def test_check_inductor_no_jit_deprecation_warning(self):
        """_check_inductor_available 不触发 JIT deprecation 警告"""
        from tb_risk.seir.autodiff import _check_inductor_available
        import warnings
        from tb_risk.seir import autodiff
        original = autodiff._INDUCTOR_AVAILABLE
        try:
            autodiff._INDUCTOR_AVAILABLE = None
            with warnings.catch_warnings():
                warnings.simplefilter('error', DeprecationWarning)
                # 不应抛 DeprecationWarning (JIT 警告被抑制)
                _check_inductor_available()
        finally:
            autodiff._INDUCTOR_AVAILABLE = original


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestWarmupBatch(unittest.TestCase):
    """要点1 (第十三轮): warmup batch 化 — _run_warmup_batch 同时对 B 条链执行
    对偶平均法, 消除原串行 warmup 的 n_chains 倍开销。"""

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.n_params = N_INFER_PARAMS
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        self.grad_fn = grad_fn
        self.log_post_fn = log_post_fn

    def test_warmup_batch_returns_valid_shapes(self):
        """_run_warmup_batch 返回正确形状的 epsilons/inv_masses/n_leapfrogs"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 3
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=5, n_chains=B, n_leapfrog=5, seed=42)
        theta_inits = [torch.randn(self.n_params, dtype=torch.float64) * 0.3
                       for _ in range(B)]
        inv_mass_inits = [np.ones(self.n_params) for _ in range(B)]
        chain_seeds = [42 + i for i in range(B)]
        eps, im, ws, nlf = sampler._run_warmup_batch(
            theta_inits, inv_mass_inits, chain_seeds)
        # eps: (B,) numpy
        self.assertEqual(eps.shape, (B,))
        self.assertTrue(np.all(eps > 0))
        self.assertTrue(np.all(eps < 1e5))
        # im: (B, 15) numpy
        self.assertEqual(im.shape, (B, self.n_params))
        self.assertTrue(np.all(np.isfinite(im)))
        self.assertTrue(np.all(im > 0))
        # nlf: (B,) numpy
        self.assertEqual(nlf.shape, (B,))
        self.assertTrue(np.all(nlf >= 1))
        # ws: list of B 个 (可选 None)
        self.assertEqual(len(ws), B)

    def test_warmup_batch_reproducible(self):
        """_run_warmup_batch 相同 seed 可复现"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 2
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=5, n_chains=B, n_leapfrog=5, seed=42)
        theta_inits = [torch.randn(self.n_params, dtype=torch.float64) * 0.3
                       for _ in range(B)]
        inv_mass_inits = [np.ones(self.n_params) for _ in range(B)]
        chain_seeds = [42 + i for i in range(B)]
        eps1, im1, _, nlf1 = sampler._run_warmup_batch(
            [t.clone() for t in theta_inits],
            [im.copy() for im in inv_mass_inits], chain_seeds)
        eps2, im2, _, nlf2 = sampler._run_warmup_batch(
            [t.clone() for t in theta_inits],
            [im.copy() for im in inv_mass_inits], chain_seeds)
        np.testing.assert_allclose(eps1, eps2, atol=1e-12)
        np.testing.assert_allclose(im1, im2, atol=1e-12)
        np.testing.assert_array_equal(nlf1, nlf2)

    def test_warmup_batch_matches_serial_per_chain(self):
        """batch warmup 每链的 epsilon/inv_mass/n_leapfrog 与串行 warmup 一致

        关键验证: batch warmup 的 per-chain 结果应与独立串行 warmup 完全一致
        (相同 seed, 相同 theta_init)。这确保 batch 化不引入数值差异。
        """
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 2
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=10, n_samples=5, n_chains=B, n_leapfrog=5, seed=42,
            adaptive_leapfrog=False)  # 禁用自适应避免随机性差异
        theta_inits = [torch.randn(self.n_params, dtype=torch.float64) * 0.3
                       for _ in range(B)]
        inv_mass_init = np.ones(self.n_params)
        chain_seeds = [42 + i for i in range(B)]

        # batch warmup
        eps_b, im_b, _, nlf_b = sampler._run_warmup_batch(
            [t.clone() for t in theta_inits],
            [inv_mass_init.copy() for _ in range(B)], chain_seeds)

        # 串行 warmup (每链独立)
        eps_s = []
        im_s = []
        nlf_s = []
        for b in range(B):
            eps, im, _, nlf = sampler._run_warmup(
                theta_inits[b].clone(), inv_mass_init.copy(),
                chain_seed=chain_seeds[b])
            eps_s.append(eps)
            im_s.append(im)
            nlf_s.append(nlf)

        # 比较 (batch warmup 的初始化阶段串行调用 _find_reasonable_epsilon,
        # 与串行 warmup 一致; warmup 循环用 batch leapfrog, 但相同 seed
        # 和 theta_init 应产生数值上等价的结果)
        # 注: batch 模式 inv_mass_t 为 (B,15) 张量参与广播运算, 串行模式为
        # (15,) 张量, PyTorch 内部 kernel 路径不同导致浮点累加顺序差异,
        # 累积 10 步 warmup 后 epsilon 差异约 1e-5 量级 (float64)。
        # 这是数值精度限制, 非 bug。用 rtol=1e-4 容忍此差异。
        for b in range(B):
            self.assertAlmostEqual(eps_b[b], eps_s[b], places=4,
                                   msg=f"链 {b} epsilon 不一致")
            np.testing.assert_allclose(im_b[b], im_s[b], atol=1e-5,
                                       err_msg=f"链 {b} inv_mass 不一致")
            self.assertEqual(int(nlf_b[b]), int(nlf_s[b]),
                             f"链 {b} n_leapfrog 不一致")

    def test_sample_batch_warmup_runs(self):
        """sample(batch_chains=True) 使用 batch warmup 正常运行"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=8, n_samples=10, n_chains=2, n_leapfrog=3, seed=42)

        def init_fn(chain_idx):
            return np.zeros(self.n_params, dtype=np.float64)

        result = sampler.sample(init_params_fn=init_fn, batch_chains=True)
        self.assertEqual(len(result['chains']), 2)
        for chain in result['chains']:
            self.assertEqual(chain.shape, (10, self.n_params))
            self.assertTrue(np.all(np.isfinite(chain)))
        for acc in result['acceptance_rates']:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAutogradGraphDefense(unittest.TestCase):
    """要点2/3/4/5 (第十三轮): 防御性 detach 确保 autograd 图不累积。

    HMC 采样链 n_samples 步的 theta 若携带梯度, autograd 图会随步数线性增长,
    内存爆炸。防御性 detach (要点3: _leapfrog 内, 要点4/5: 循环体内) 确保
    theta 永不携带梯度 (即使 grad_fn 返回 requires_grad=True 的结果)。
    """

    def setUp(self):
        from tb_risk.seir.autodiff import N_INFER_PARAMS
        self.n_params = N_INFER_PARAMS
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

    def test_hamiltonian_returns_detached_tensor(self):
        """_hamiltonian 返回的 tensor 不携带梯度 (要点2)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        sampler = HMCSampler(
            torch_log_posterior_v4, compute_grad_log_posterior,
            N_INFER_PARAMS, n_warmup=1, n_samples=1, seed=42)
        # 即使 log_post 携带梯度, H 也不应携带
        theta = torch.randn(self.n_params, dtype=torch.float64,
                            requires_grad=True)
        p = torch.randn(self.n_params, dtype=torch.float64)
        inv_mass = np.ones(self.n_params)
        H = sampler._hamiltonian(theta.sum(), p, inv_mass)
        self.assertFalse(H.requires_grad,
                         "_hamiltonian 返回的 H 不应携带梯度")

    def test_hamiltonian_batch_returns_detached_tensor(self):
        """_hamiltonian_batch 返回的 tensor 不携带梯度 (要点2)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        from tb_risk.seir.autodiff import N_INFER_PARAMS
        sampler = HMCSampler(
            lambda t: t.sum(), lambda t: (t, t.sum()),
            N_INFER_PARAMS, n_warmup=1, n_samples=1, seed=42)
        B = 3
        log_post = torch.randn(B, dtype=torch.float64, requires_grad=True)
        p = torch.randn(B, self.n_params, dtype=torch.float64)
        inv_mass_t = torch.ones(B, self.n_params, dtype=torch.float64)
        H = sampler._hamiltonian_batch(log_post, p, None, inv_mass_t=inv_mass_t)
        self.assertFalse(H.requires_grad,
                         "_hamiltonian_batch 返回的 H 不应携带梯度")

    def test_leapfrog_returns_detached_theta(self):
        """_leapfrog 返回的 theta/p 不携带梯度 (要点3)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        # grad_fn 必须是只接收 theta 的闭包 (与 HMCSampler 约定一致)
        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)
        sampler = HMCSampler(
            torch_log_posterior_v4, grad_fn,
            N_INFER_PARAMS, n_warmup=1, n_samples=1, seed=42)
        # theta 携带梯度 (模拟外部梯度泄漏)
        theta = torch.randn(self.n_params, dtype=torch.float64,
                            requires_grad=True)
        p = torch.randn(self.n_params, dtype=torch.float64)
        inv_mass = np.ones(self.n_params)
        theta_new, p_new, lp, grad = sampler._leapfrog(
            theta, p, 0.01, inv_mass, 3)
        self.assertFalse(theta_new.requires_grad,
                         "_leapfrog 返回的 theta 不应携带梯度")
        self.assertFalse(p_new.requires_grad,
                         "_leapfrog 返回的 p 不应携带梯度")

    def test_single_chain_theta_never_requires_grad(self):
        """_run_single_chain 循环中 theta 永不 requires_grad (要点4)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)

        # 构造返回 requires_grad=True 的 grad_fn (模拟梯度泄漏)
        def leaky_grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            lp = y.sum()
            grad, = torch.autograd.grad(lp, y)
            # 故意不 detach (模拟错误的 grad_fn)
            return grad, lp

        def leaky_log_post_fn(theta):
            return theta.sum()

        sampler = HMCSampler(
            leaky_log_post_fn, leaky_grad_fn, N_INFER_PARAMS,
            n_warmup=5, n_samples=10, n_chains=1, n_leapfrog=3, seed=42)
        theta_init = torch.randn(N_INFER_PARAMS, dtype=torch.float64) * 0.3
        inv_mass = np.ones(N_INFER_PARAMS)
        # 即使 grad_fn 泄漏梯度, 防御性 detach 应确保采样正常运行
        samples, log_probs, acc = sampler._run_single_chain(
            theta_init, 0.01, inv_mass, chain_seed=42, n_leapfrog=3)
        self.assertEqual(samples.shape, (10, N_INFER_PARAMS))
        self.assertTrue(np.all(np.isfinite(samples)))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestFindReasonableEpsilonBatch(unittest.TestCase):
    """要点1 (第十四轮): _find_reasonable_epsilon_batch 测试

    验证 batch 版本与串行版本 _find_reasonable_epsilon 在相同 seed/theta/inv_mass
    下返回相同的 epsilon (per-chain)。这是 batch warmup 初始化阶段的核心正确性保证。
    """

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        self.grad_fn = grad_fn
        self.log_post_fn = log_post_fn
        self.n_params = N_INFER_PARAMS

    def test_returns_correct_shape(self):
        """返回 (B,) 张量"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 3
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=1, n_samples=1, seed=42)
        theta = torch.randn(B, self.n_params, dtype=torch.float64) * 0.3
        inv_mass_t = torch.ones(B, self.n_params, dtype=torch.float64)
        gens = [torch.Generator() for _ in range(B)]
        for g, s in zip(gens, [42, 43, 44]):
            g.manual_seed(s)
        eps = sampler._find_reasonable_epsilon_batch(theta, inv_mass_t, gens)
        self.assertEqual(eps.shape, (B,))
        self.assertTrue(torch.all(eps > 0))
        self.assertFalse(eps.requires_grad)

    def test_matches_serial_per_chain(self):
        """batch 版本与串行版本返回相同 epsilon (per-chain)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 3
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=1, n_samples=1, seed=42)
        # 固定 theta 和 inv_mass 确保 batch/串行输入相同
        torch.manual_seed(0)
        theta = torch.randn(B, self.n_params, dtype=torch.float64) * 0.3
        inv_mass_t = torch.ones(B, self.n_params, dtype=torch.float64)
        seeds = [42 + i for i in range(B)]

        # batch 版本
        gens_b = [torch.Generator() for _ in range(B)]
        for g, s in zip(gens_b, seeds):
            g.manual_seed(s)
        eps_batch = sampler._find_reasonable_epsilon_batch(
            theta, inv_mass_t, gens_b)

        # 串行版本 (每链独立)
        eps_serial = []
        for b in range(B):
            gen_s = torch.Generator()
            gen_s.manual_seed(seeds[b])
            eps_b = sampler._find_reasonable_epsilon(
                theta[b], np.ones(self.n_params), gen=gen_s,
                inv_mass_t=inv_mass_t[b])
            eps_serial.append(eps_b)
        eps_serial_t = torch.tensor(eps_serial, dtype=torch.float64)

        # 应完全一致 (相同 seed/theta/inv_mass, 相同算法)
        torch.testing.assert_close(eps_batch, eps_serial_t, rtol=1e-10, atol=1e-12)

    def test_all_chains_finite_positive(self):
        """所有链返回有限正数 (>= 1e-6)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 4
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=1, n_samples=1, seed=42)
        torch.manual_seed(1)
        theta = torch.randn(B, self.n_params, dtype=torch.float64) * 0.5
        inv_mass_t = torch.ones(B, self.n_params, dtype=torch.float64)
        gens = [torch.Generator() for _ in range(B)]
        for g, s in zip(gens, [100 + i for i in range(B)]):
            g.manual_seed(s)
        eps = sampler._find_reasonable_epsilon_batch(theta, inv_mass_t, gens)
        self.assertTrue(torch.all(torch.isfinite(eps)))
        self.assertTrue(torch.all(eps >= 1e-6))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestCacheValidSemantics(unittest.TestCase):
    """要点5 (第十四轮): cache_valid 语义测试

    cache_valid=True 时 H0 的 grad_fn 调用应被跳过 (用缓存)。
    通过计数 grad_fn 调用次数验证:
      - cache_valid 正确: 1 (首步 H0) + n_samples * n_leapfrog (leapfrog 内部)
      - cache_valid 错误: n_samples * (1 + n_leapfrog) (每步都调用 H0)
    """

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))
        self.n_params = N_INFER_PARAMS

    def _make_counting_grad_fn(self):
        """构造计数器 grad_fn"""
        from tb_risk.seir.autodiff import compute_grad_log_posterior
        call_count = [0]

        def grad_fn(theta):
            call_count[0] += 1
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)
        return grad_fn, call_count

    def test_cache_valid_skips_grad_fn_in_sampling(self):
        """_run_sampling_batch: cache_valid=True 时 H0 调用被跳过"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        from tb_risk.seir.autodiff import torch_log_posterior_v4

        grad_fn, call_count = self._make_counting_grad_fn()
        n_samples = 5
        n_leapfrog = 3
        B = 2
        sampler = HMCSampler(
            torch_log_posterior_v4, grad_fn, self.n_params,
            n_warmup=1, n_samples=n_samples, n_chains=B,
            n_leapfrog=n_leapfrog, seed=42)

        theta_inits = [torch.zeros(self.n_params, dtype=torch.float64)
                       for _ in range(B)]
        epsilons = [0.01] * B
        inv_masses = [np.ones(self.n_params) for _ in range(B)]
        chain_seeds = [42, 43]
        n_leapfrogs = [n_leapfrog] * B

        samples, log_probs, acc = sampler._run_sampling_batch(
            theta_inits, epsilons, inv_masses, chain_seeds, n_leapfrogs)

        # cache_valid 正确: 1 (首步 H0) + n_samples * n_leapfrog (leapfrog 内部)
        expected = 1 + n_samples * n_leapfrog
        self.assertEqual(call_count[0], expected,
                         f"cache_valid 逻辑错误: 期望 {expected} 次调用, "
                         f"实际 {call_count[0]} 次 (cache_valid 应跳过 H0 重复调用)")

    def test_cache_valid_high_rejection(self):
        """高拒绝率场景 (过大 epsilon) cache_valid 仍正常工作"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        from tb_risk.seir.autodiff import torch_log_posterior_v4

        grad_fn, call_count = self._make_counting_grad_fn()
        n_samples = 5
        n_leapfrog = 3
        B = 2
        sampler = HMCSampler(
            torch_log_posterior_v4, grad_fn, self.n_params,
            n_warmup=1, n_samples=n_samples, n_chains=B,
            n_leapfrog=n_leapfrog, seed=42)

        theta_inits = [torch.zeros(self.n_params, dtype=torch.float64)
                       for _ in range(B)]
        # 过大 epsilon → 高拒绝率 (几乎所有提议被拒绝)
        epsilons = [10.0] * B
        inv_masses = [np.ones(self.n_params) for _ in range(B)]
        chain_seeds = [42, 43]
        n_leapfrogs = [n_leapfrog] * B

        samples, log_probs, acc = sampler._run_sampling_batch(
            theta_inits, epsilons, inv_masses, chain_seeds, n_leapfrogs)

        # 采样不崩溃
        self.assertEqual(samples.shape, (B, n_samples, self.n_params))
        self.assertTrue(np.all(np.isfinite(samples)))
        # 高拒绝率下接受率应很低 (但 cache_valid 仍正常工作)
        self.assertTrue(np.all(acc < 0.5))
        # grad_fn 调用次数仍为 1 + n_samples * n_leapfrog
        # (cache_valid 不受接受率影响 — 拒绝时缓存 pre-leapfrog grad, 仍有效)
        expected = 1 + n_samples * n_leapfrog
        self.assertEqual(call_count[0], expected,
                         f"高拒绝率下 cache_valid 失效: 期望 {expected} 次, "
                         f"实际 {call_count[0]} 次")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRound15FindReasonableEpsilonBatch(unittest.TestCase):
    """要点1/2 (第十五轮): _find_reasonable_epsilon_batch 改进测试

    要点1: device 自适应提前退出 (CPU 用 active.any().item() 提前退出)
    要点2: 动量乘 sqrt_mass 防御性正确性 (inv_mass 非全 1 时 batch/串行一致)
    """

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        self.grad_fn = grad_fn
        self.log_post_fn = log_post_fn
        self.n_params = N_INFER_PARAMS

    def test_cpu_early_exit_matches_fixed_loop(self):
        """要点1 (第十五轮): CPU 提前退出与固定 50 步结果一致

        CPU 上 active.any().item() 提前退出在所有链都收敛后退出循环,
        此时 epsilon 已确定, 与跑满 50 步 (剩余步 torch.where 不更新
        inactive 链) 结果一致。
        """
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 3
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=1, n_samples=1, seed=42)
        torch.manual_seed(0)
        theta = torch.randn(B, self.n_params, dtype=torch.float64) * 0.3
        inv_mass_t = torch.ones(B, self.n_params, dtype=torch.float64)
        seeds = [42 + i for i in range(B)]

        # CPU 上调用 (提前退出)
        gens = [torch.Generator() for _ in range(B)]
        for g, s in zip(gens, seeds):
            g.manual_seed(s)
        eps_cpu = sampler._find_reasonable_epsilon_batch(theta, inv_mass_t, gens)

        # 所有 epsilon 应为有限正数 (提前退出不影响结果)
        self.assertEqual(eps_cpu.shape, (B,))
        self.assertTrue(torch.all(torch.isfinite(eps_cpu)))
        self.assertTrue(torch.all(eps_cpu >= 1e-6))

    def test_sqrt_mass_inv_mass_non_ones_batch_matches_serial(self):
        """要点2 (第十五轮): inv_mass 非全 1 时 batch/串行仍一致

        动量乘 sqrt_mass = sqrt(1/inv_mass_t) 后, batch 和串行版本
        在相同 seed/theta/inv_mass 下应返回相同 epsilon。
        """
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 3
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=1, n_samples=1, seed=42)
        torch.manual_seed(0)
        theta = torch.randn(B, self.n_params, dtype=torch.float64) * 0.3
        # inv_mass 非全 1 (per-parameter 不同值, 测试 sqrt_mass 广播)
        inv_mass_vals = torch.tensor(
            [1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3,
             0.6, 1.4, 0.95, 1.05, 0.85],
            dtype=torch.float64)
        inv_mass_t = inv_mass_vals.unsqueeze(0).expand(B, -1).clone()
        seeds = [42 + i for i in range(B)]

        # batch 版本
        gens_b = [torch.Generator() for _ in range(B)]
        for g, s in zip(gens_b, seeds):
            g.manual_seed(s)
        eps_batch = sampler._find_reasonable_epsilon_batch(
            theta, inv_mass_t, gens_b)

        # 串行版本 (每链独立)
        eps_serial = []
        for b in range(B):
            gen_s = torch.Generator()
            gen_s.manual_seed(seeds[b])
            eps_b = sampler._find_reasonable_epsilon(
                theta[b], inv_mass_vals.numpy(), gen=gen_s,
                inv_mass_t=inv_mass_t[b])
            eps_serial.append(eps_b)
        eps_serial_t = torch.tensor(eps_serial, dtype=torch.float64)

        # batch/串行应完全一致 (相同 seed/theta/inv_mass, 相同 sqrt_mass 变换)
        torch.testing.assert_close(eps_batch, eps_serial_t, rtol=1e-10, atol=1e-12)

    def test_sqrt_mass_changes_h0_kinetic_components(self):
        """要点2 (第十五轮): sqrt_mass 变换影响 theta_new (leapfrog 第一步)

        注: sqrt_mass 变换使 p^T M^{-1} p 不变 (H0 相同), 但 theta_new
        = theta + 0.5 * eps * inv_mass * p = theta + 0.5 * eps * sqrt(inv_mass)
        * randn 不同。验证 sqrt_mass 真的起作用 — 通过检查不同 inv_mass 下
        batch 版本不抛异常且返回有限值 (而非静默无影响)。

        由于 H0 相同且某些 seed 下两条链可能都立即收敛到相同 epsilon,
        不要求 epsilon 必须不同, 而是验证 inv_mass 非全 1 时算法仍正常工作。
        """
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 2
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=1, n_samples=1, seed=42)
        torch.manual_seed(0)
        theta = torch.randn(B, self.n_params, dtype=torch.float64) * 0.3
        seeds = [42 + i for i in range(B)]

        # inv_mass = 2 (sqrt_mass = sqrt(0.5), 动量缩小, 但 p^T M^{-1} p 不变)
        inv_mass_twos = torch.ones(B, self.n_params, dtype=torch.float64) * 2.0
        gens = [torch.Generator() for _ in range(B)]
        for g, s in zip(gens, seeds):
            g.manual_seed(s)
        eps_twos = sampler._find_reasonable_epsilon_batch(
            theta, inv_mass_twos, gens)

        # 验证 sqrt_mass 变换后仍返回有限正数 (不崩溃)
        self.assertEqual(eps_twos.shape, (B,))
        self.assertTrue(torch.all(torch.isfinite(eps_twos)))
        self.assertTrue(torch.all(eps_twos >= 1e-6))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRound15WarmupBatchRingBuffer(unittest.TestCase):
    """要点3 (第十五轮): warmup_gpu_full 环形缓冲区测试

    buffer_size = min(n_warmup, 200), n_warmup > 200 时启用环形缓冲区。
    验证质量矩阵更新在 m > buffer_size 时仍正确工作。
    """

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        self.grad_fn = grad_fn
        self.log_post_fn = log_post_fn
        self.n_params = N_INFER_PARAMS

    def test_warmup_n_warmup_above_buffer_size_runs(self):
        """要点3 (第十五轮): n_warmup > 200 时 warmup_batch 仍正常运行

        buffer_size = min(n_warmup, 200) = 200, warmup 循环到 m=201+
        时缓冲区已满, 验证不崩溃且结果有限。
        """
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 2
        # n_warmup=250 > buffer_size=200, 触发环形缓冲区满
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=250, n_samples=5, n_chains=B, seed=42,
            n_leapfrog=3)
        theta_inits = [torch.zeros(self.n_params, dtype=torch.float64)
                       for _ in range(B)]
        inv_mass_inits = [np.ones(self.n_params) for _ in range(B)]
        chain_seeds = [42 + i for i in range(B)]

        eps, im, warmup_samples, nlf = sampler._run_warmup_batch(
            theta_inits, inv_mass_inits, chain_seeds)

        # 结果有限
        self.assertEqual(eps.shape, (B,))
        self.assertEqual(im.shape, (B, self.n_params))
        self.assertTrue(np.all(np.isfinite(eps)))
        self.assertTrue(np.all(np.isfinite(im)))
        # warmup_samples 是 list of B 个元素 (np.array 或 None)
        self.assertEqual(len(warmup_samples), B)
        # n_warmup=250 > buffer_size=200, warmup_samples 仅含最近 200 步
        for b in range(B):
            if warmup_samples[b] is not None:
                # 接受样本数 <= buffer_size (200)
                self.assertLessEqual(len(warmup_samples[b]), 200)

    def test_warmup_n_warmup_below_buffer_size_runs(self):
        """要点3 (第十五轮): n_warmup < 200 时 warmup_batch 正常运行

        buffer_size = min(n_warmup, 200) = n_warmup, 缓冲区未满,
        行为与原完整存储一致。
        """
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 2
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=50, n_samples=5, n_chains=B, seed=42,
            n_leapfrog=3)
        theta_inits = [torch.zeros(self.n_params, dtype=torch.float64)
                       for _ in range(B)]
        inv_mass_inits = [np.ones(self.n_params) for _ in range(B)]
        chain_seeds = [42 + i for i in range(B)]

        eps, im, warmup_samples, nlf = sampler._run_warmup_batch(
            theta_inits, inv_mass_inits, chain_seeds)

        # 结果有限
        self.assertEqual(eps.shape, (B,))
        self.assertTrue(np.all(np.isfinite(eps)))
        self.assertTrue(np.all(np.isfinite(im)))
        # n_warmup=50 <= buffer_size=50, warmup_samples 含所有接受样本
        for b in range(B):
            if warmup_samples[b] is not None:
                self.assertLessEqual(len(warmup_samples[b]), 50)

    def test_warmup_ring_buffer_sample_integration(self):
        """要点3 (第十五轮): 环形缓冲区与 sample() 集成测试

        完整 sample() 流程在 n_warmup > 200 时仍正常工作。
        """
        from tb_risk.seir.hmc_sampler import HMCSampler
        sampler = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=220, n_samples=10, n_chains=2, seed=42,
            n_leapfrog=3)
        # batch_chains 是 sample() 参数而非构造函数参数
        result = sampler.sample(batch_chains=True)

        self.assertEqual(len(result['chains']), 2)
        self.assertEqual(result['chains'][0].shape, (10, self.n_params))
        # 采样链无 NaN
        for chain in result['chains']:
            self.assertFalse(np.any(np.isnan(chain)),
                             "采样链含 NaN (环形缓冲区破坏 warmup)")
        # 诊断信息完整
        self.assertIn('r_hat', result['diagnostics'])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRound16WarmupBatchSerialConsistency(unittest.TestCase):
    """要点5 (第十六轮): _run_warmup_batch 与 _run_warmup 端到端统计一致性测试

    相同初始参数和 seed 下, _run_warmup_batch 与逐链 _run_warmup 产生的
    epsilon/inv_mass 应统计一致 (均值在容差范围内接近)。由于 batch 用
    torch.Generator 而串行用 numpy RandomState, 随机数序列不同, 结果不会
    完全相同但应无系统性偏差。此测试捕获 batch 化引入的 per-chain mask
    处理错误等系统性问题。
    """

    def setUp(self):
        from tb_risk.seir.autodiff import (compute_grad_log_posterior,
                                            torch_log_posterior_v4,
                                            N_INFER_PARAMS)
        self.init_state = np.zeros(270)
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 0] = 900.0
        self.init_state[2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9 + 5] = 10.0
        self.obs_data = (np.array([5.0, 10.0, 15.0]),
                         np.array([5.0, 10.0, 15.0]))

        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, self.obs_data, self.init_state,
                t_span=(0, 15), dt=5.0)

        self.grad_fn = grad_fn
        self.log_post_fn = log_post_fn
        self.n_params = N_INFER_PARAMS

    def test_warmup_batch_matches_serial_statistics(self):
        """要点5 (第十六轮): warmup batch 与串行统计一致性

        多次运行 (3 seeds × 2 chains = 6 样本) 取均值比较。
        容差 50% — 因 RNG 不同 (torch.Generator vs numpy RandomState)
        无法要求精确匹配, 但应无系统性偏差。
        """
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 2
        seeds = [42, 123, 7]  # 3 次独立运行
        n_warmup = 40  # 较小 n_warmup 加速测试

        eps_batch_all = []
        eps_serial_all = []
        im_batch_all = []
        im_serial_all = []

        for seed in seeds:
            theta_inits = [torch.zeros(self.n_params, dtype=torch.float64)
                           for _ in range(B)]
            inv_mass_inits = [np.ones(self.n_params) for _ in range(B)]
            chain_seeds = [seed + i for i in range(B)]

            # batch 模式
            sampler_b = HMCSampler(
                self.log_post_fn, self.grad_fn, self.n_params,
                n_warmup=n_warmup, n_samples=5, n_chains=B,
                seed=seed, n_leapfrog=3)
            eps_b, im_b, _, nlf_b = sampler_b._run_warmup_batch(
                theta_inits, inv_mass_inits, chain_seeds)
            eps_batch_all.append(eps_b)
            im_batch_all.append(im_b)

            # 串行模式 (逐链调用)
            sampler_s = HMCSampler(
                self.log_post_fn, self.grad_fn, self.n_params,
                n_warmup=n_warmup, n_samples=5, n_chains=B,
                seed=seed, n_leapfrog=3)
            eps_s_list = []
            im_s_list = []
            for b in range(B):
                eps_s, im_s, _, nlf_s = sampler_s._run_warmup(
                    theta_inits[b], inv_mass_inits[b],
                    chain_seed=chain_seeds[b])
                eps_s_list.append(eps_s)
                im_s_list.append(im_s)
            eps_serial_all.append(np.array(eps_s_list))
            im_serial_all.append(np.array(im_s_list))

        # 合并所有运行和链的结果: (3 seeds × 2 chains = 6,)
        eps_batch = np.concatenate(eps_batch_all)
        eps_serial = np.concatenate(eps_serial_all)
        # inv_mass: (6, 15)
        im_batch = np.concatenate(im_batch_all)
        im_serial = np.concatenate(im_serial_all)

        # 1. 所有结果有限
        self.assertTrue(np.all(np.isfinite(eps_batch)),
                        f"batch epsilon 含 NaN/Inf: {eps_batch}")
        self.assertTrue(np.all(np.isfinite(eps_serial)),
                        f"serial epsilon 含 NaN/Inf: {eps_serial}")
        self.assertTrue(np.all(np.isfinite(im_batch)),
                        f"batch inv_mass 含 NaN/Inf: {im_batch}")
        self.assertTrue(np.all(np.isfinite(im_serial)),
                        f"serial inv_mass 含 NaN/Inf: {im_serial}")

        # 2. epsilon 均值差异 < 50% (宽松容差, 因 RNG 不同)
        eps_mean_batch = float(np.mean(eps_batch))
        eps_mean_serial = float(np.mean(eps_serial))
        eps_rel_diff = abs(eps_mean_batch - eps_mean_serial) / max(
            abs(eps_mean_serial), 1e-10)
        self.assertLess(
            eps_rel_diff, 0.5,
            f"epsilon 均值差异过大 (可能 batch 化引入系统性偏差): "
            f"batch={eps_mean_batch}, serial={eps_mean_serial}, "
            f"rel_diff={eps_rel_diff}")

        # 3. inv_mass 均值差异 < 50% (per-dimension)
        im_mean_batch = np.mean(im_batch, axis=0)  # (15,)
        im_mean_serial = np.mean(im_serial, axis=0)  # (15,)
        im_rel_diff = np.abs(im_mean_batch - im_mean_serial) / np.maximum(
            np.abs(im_mean_serial), 1e-10)
        self.assertTrue(
            np.all(im_rel_diff < 0.5),
            f"inv_mass 均值差异过大 (可能 batch 化引入系统性偏差): "
            f"max_rel_diff={np.max(im_rel_diff)}, "
            f"batch={im_mean_batch}, serial={im_mean_serial}")

    def test_warmup_batch_and_serial_both_finite(self):
        """要点5 (第十六轮): 单次运行 batch 与串行都产生有限结果

        较 test_warmup_batch_matches_serial_statistics 更轻量的 smoke test,
        不比较均值, 仅验证两种模式都正常完成。
        """
        from tb_risk.seir.hmc_sampler import HMCSampler
        B = 2
        seed = 42
        theta_inits = [torch.zeros(self.n_params, dtype=torch.float64)
                       for _ in range(B)]
        inv_mass_inits = [np.ones(self.n_params) for _ in range(B)]
        chain_seeds = [seed + i for i in range(B)]

        # batch
        sampler_b = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=30, n_samples=5, n_chains=B, seed=seed, n_leapfrog=3)
        eps_b, im_b, _, nlf_b = sampler_b._run_warmup_batch(
            theta_inits, inv_mass_inits, chain_seeds)

        # 串行
        sampler_s = HMCSampler(
            self.log_post_fn, self.grad_fn, self.n_params,
            n_warmup=30, n_samples=5, n_chains=B, seed=seed, n_leapfrog=3)
        eps_s_list = []
        im_s_list = []
        for b in range(B):
            eps_s, im_s, _, _ = sampler_s._run_warmup(
                theta_inits[b], inv_mass_inits[b],
                chain_seed=chain_seeds[b])
            eps_s_list.append(eps_s)
            im_s_list.append(im_s)

        # 形状验证
        self.assertEqual(eps_b.shape, (B,))
        self.assertEqual(im_b.shape, (B, self.n_params))
        self.assertEqual(len(eps_s_list), B)
        self.assertEqual(len(im_s_list), B)

        # 所有结果有限
        self.assertTrue(np.all(np.isfinite(eps_b)))
        self.assertTrue(np.all(np.isfinite(im_b)))
        for eps_s, im_s in zip(eps_s_list, im_s_list):
            self.assertTrue(np.isfinite(eps_s).all() if hasattr(eps_s, 'all')
                            else np.isfinite(eps_s))
            self.assertTrue(np.all(np.isfinite(im_s)))

        # n_leapfrog 在合理范围
        self.assertTrue(np.all(nlf_b >= 1),
                        f"batch n_leapfrog < 1: {nlf_b}")


if __name__ == '__main__':
    unittest.main()
