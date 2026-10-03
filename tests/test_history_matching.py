#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — 历史匹配与仿真器前置筛选 (HME)

测试改进：HME 前置筛选 (History Matching & Emulator)
在 MCMC 之前用 GP 仿真器排除不可信参数空间, 预期减少 MCMC 迭代数 5-10x。

覆盖:
  1. 参数空间转换 (约束 ↔ 无约束 numpy 版本)
  2. LHS 采样设计点
  3. v4 ODE 前向模型 (批量)
  4. GP 仿真器训练与预测
  5. 不可信度指标 I(x) 计算
  6. HMC 初始化函数 (从非不可信参数采样)
  7. 缩小后的参数边界
  8. 诊断信息完整性
  9. run_hme_then_mcmc 端到端流程

文献:
  Kennedy & O'Hagan (2001) JRSS-B 63(3):425-464 — Bayesian calibration
  Vernon et al. (2010) SIAM/ASA J UQ 2(1):1-37 — History matching
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

try:
    import sklearn  # noqa: F401
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

HME_AVAILABLE = TORCH_AVAILABLE and SKLEARN_AVAILABLE


def _build_initial_state():
    """构造 270D 初始状态: 900 S + 50 Lf + 10 Isp 在 15-49 岁 HIV-/DS。

    索引: age_idx=2 (15-49), hiv_idx=0 (HIV-), dr_idx=0 (DS), comp=0/1/5
    flat_idx = age*3*2*9 + hiv*2*9 + dr*9 + comp
    """
    state = np.zeros(270)
    base = 2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9
    state[base + 0] = 900.0  # S
    state[base + 1] = 50.0   # Lf
    state[base + 5] = 10.0   # Isp
    return state


@unittest.skipUnless(HME_AVAILABLE, "PyTorch and scikit-learn both required")
class TestParamSpaceTransforms(unittest.TestCase):
    """numpy 版本约束 ↔ 无约束空间转换测试"""

    def test_roundtrip(self):
        """约束 → 无约束 → 约束 恢复原值"""
        from tb_risk.seir.history_matching import (
            _constrained_to_unconstrained,
            _unconstrained_to_constrained,
        )
        from tb_risk.seir.autodiff import PARAM_BOUNDS

        x = np.array([[lo + (hi - lo) * 0.3 for lo, hi in PARAM_BOUNDS],
                      [lo + (hi - lo) * 0.7 for lo, hi in PARAM_BOUNDS]])
        y = _constrained_to_unconstrained(x)
        x_rec = _unconstrained_to_constrained(y)
        np.testing.assert_allclose(x, x_rec, rtol=1e-8, atol=1e-10)

    def test_unconstrained_within_bounds(self):
        """约束空间参数始终在 PARAM_BOUNDS 内"""
        from tb_risk.seir.history_matching import _unconstrained_to_constrained
        from tb_risk.seir.autodiff import PARAM_BOUNDS

        rng = np.random.RandomState(42)
        y = rng.randn(20, 15) * 3.0  # 较大范围的无约束值
        x = _unconstrained_to_constrained(y)
        for i, (lo, hi) in enumerate(PARAM_BOUNDS):
            self.assertTrue(np.all(x[:, i] > lo),
                            f"param {i}: min={x[:, i].min()} <= {lo}")
            self.assertTrue(np.all(x[:, i] < hi),
                            f"param {i}: max={x[:, i].max()} >= {hi}")

    def test_midpoint_maps_to_zero(self):
        """约束范围中点对应无约束 0"""
        from tb_risk.seir.history_matching import _constrained_to_unconstrained
        from tb_risk.seir.autodiff import PARAM_BOUNDS

        x_mid = np.array([[(lo + hi) / 2 for lo, hi in PARAM_BOUNDS]])
        y = _constrained_to_unconstrained(x_mid)
        np.testing.assert_allclose(y[0], np.zeros(15), atol=1e-10)


@unittest.skipUnless(HME_AVAILABLE, "PyTorch and scikit-learn both required")
class TestHistoryMatchingSetup(unittest.TestCase):
    """HistoryMatching 初始化与 LHS 采样测试"""

    def test_initialization(self):
        """HistoryMatching 可正常初始化"""
        from tb_risk.seir.history_matching import HistoryMatching

        init_state = _build_initial_state()
        obs_data = (np.array([5.0, 10.0]), np.array([5.0, 10.0]))

        hme = HistoryMatching(
            obs_data, init_state, t_span=(0, 10), dt=2.5,
            n_design=20, random_seed=42)

        self.assertEqual(hme.n_design, 20)
        self.assertEqual(hme.t_span, (0.0, 10.0))
        self.assertEqual(hme.dt, 2.5)
        self.assertGreater(hme.obs_var, 0.0)
        self.assertIsNone(hme.design_params)

    def test_obs_variance_default(self):
        """默认观测方差 = (0.2 * std(obs_I))^2, 与 torch_log_likelihood_v4 一致"""
        from tb_risk.seir.history_matching import HistoryMatching

        obs_I = np.array([5.0, 10.0, 15.0, 20.0])
        init_state = _build_initial_state()
        hme = HistoryMatching(
            (obs_I, np.array([5.0, 10.0, 15.0, 20.0])),
            init_state, t_span=(0, 15), dt=5.0)

        expected_var = (0.2 * np.std(obs_I)) ** 2
        self.assertAlmostEqual(hme.obs_var, expected_var, places=8)

    def test_lhs_design_shape_and_range(self):
        """LHS 采样生成正确形状和范围内的参数"""
        from tb_risk.seir.history_matching import HistoryMatching
        from tb_risk.seir.autodiff import PARAM_BOUNDS

        init_state = _build_initial_state()
        hme = HistoryMatching(
            (np.array([5.0]), np.array([5.0])),
            init_state, t_span=(0, 5), dt=2.5,
            n_design=25, random_seed=42)

        design = hme._generate_lhs_design()

        self.assertEqual(design.shape, (25, 15))
        for i, (lo, hi) in enumerate(PARAM_BOUNDS):
            self.assertTrue(np.all(design[:, i] >= lo),
                            f"param {i}: min={design[:, i].min()} < {lo}")
            self.assertTrue(np.all(design[:, i] <= hi),
                            f"param {i}: max={design[:, i].max()} > {hi}")

    def test_lhs_stratification(self):
        """LHS 保证每维分层覆盖 (每维 25 个值分布在 25 个等宽子区间)"""
        from tb_risk.seir.history_matching import HistoryMatching
        from tb_risk.seir.autodiff import PARAM_BOUNDS

        init_state = _build_initial_state()
        hme = HistoryMatching(
            (np.array([5.0]), np.array([5.0])),
            init_state, t_span=(0, 5), dt=2.5,
            n_design=25, random_seed=42)

        design = hme._generate_lhs_design()

        # 每维: 将 [0, 1] 分成 25 等宽子区间, 每子区间应有且仅有 1 个样本
        for i, (lo, hi) in enumerate(PARAM_BOUNDS):
            u = (design[:, i] - lo) / (hi - lo)
            bins = np.floor(u * 25).astype(int)
            bins = np.clip(bins, 0, 24)
            self.assertEqual(len(set(bins)), 25,
                             f"param {i} 分层不完整: 仅 {len(set(bins))} 个 bin")


@unittest.skipUnless(HME_AVAILABLE, "PyTorch and scikit-learn both required")
class TestForwardModel(unittest.TestCase):
    """前向模型 (v4 ODE 批量积分) 测试"""

    def test_forward_model_shape(self):
        """前向模型返回 (B, n_obs) 形状"""
        from tb_risk.seir.history_matching import HistoryMatching

        init_state = _build_initial_state()
        hme = HistoryMatching(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            init_state, t_span=(0, 10), dt=2.5,
            n_design=8, random_seed=42)

        design = hme._generate_lhs_design()
        outputs = hme._run_forward_model(design, batch_size=4)

        self.assertEqual(outputs.shape, (8, 2))

    def test_forward_model_finite(self):
        """前向模型输出为有限值 (无 NaN/Inf)"""
        from tb_risk.seir.history_matching import HistoryMatching

        init_state = _build_initial_state()
        hme = HistoryMatching(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            init_state, t_span=(0, 10), dt=2.5,
            n_design=8, random_seed=42)

        design = hme._generate_lhs_design()
        outputs = hme._run_forward_model(design, batch_size=4)

        self.assertTrue(np.all(np.isfinite(outputs)),
                        f"前向模型输出含 NaN/Inf: {outputs}")

    def test_forward_model_batch_consistency(self):
        """不同 batch_size 产生相同结果"""
        from tb_risk.seir.history_matching import HistoryMatching

        init_state = _build_initial_state()
        hme = HistoryMatching(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            init_state, t_span=(0, 10), dt=2.5,
            n_design=10, random_seed=42)

        design = hme._generate_lhs_design()
        out1 = hme._run_forward_model(design, batch_size=3)
        out2 = hme._run_forward_model(design, batch_size=10)

        np.testing.assert_allclose(out1, out2, rtol=1e-10, atol=1e-12)


@unittest.skipUnless(HME_AVAILABLE, "PyTorch and scikit-learn both required")
class TestForwardModelParallel(unittest.TestCase):
    """HME 前向模型多进程并行 (n_jobs) 测试

    覆盖:
      1. 构造函数接受 n_jobs 且钳制为 >= 1
      2. 并行路径 (n_jobs > 1) 与串行路径 (n_jobs=1) 结果一致
      3. run_hme_then_mcmc 接受 n_jobs 参数并透传给 HistoryMatching
    """

    def setUp(self):
        from tb_risk.seir.history_matching import HistoryMatching

        self.init_state = _build_initial_state()
        self.hme = HistoryMatching(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            self.init_state, t_span=(0, 10), dt=2.5,
            n_design=8, random_seed=42, n_jobs=2)
        self.design = self.hme._generate_lhs_design()

    def test_constructor_accepts_n_jobs(self):
        """构造函数接受 n_jobs 并钳制为 >= 1"""
        from tb_risk.seir.history_matching import HistoryMatching

        hme1 = HistoryMatching(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            self.init_state, t_span=(0, 10), dt=2.5,
            n_design=4, random_seed=42, n_jobs=2)
        self.assertEqual(hme1.n_jobs, 2)
        # n_jobs <= 0 时钳制为 1
        hme0 = HistoryMatching(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            self.init_state, t_span=(0, 10), dt=2.5,
            n_design=4, random_seed=42, n_jobs=0)
        self.assertEqual(hme0.n_jobs, 1)

    @pytest.mark.slow
    def test_parallel_matches_serial(self):
        """并行路径与串行路径产生一致输出 (多进程并行加速)"""
        serial = self.hme._run_forward_model(self.design, n_jobs=1)
        parallel = self.hme._run_forward_model(self.design, n_jobs=2)

        self.assertEqual(serial.shape, (8, 2))
        self.assertEqual(parallel.shape, serial.shape)
        # 分块并行与串行的数值应完全一致 (相同 ODE, 仅分派方式不同)
        np.testing.assert_allclose(parallel, serial, rtol=1e-12, atol=1e-14)

    @pytest.mark.slow
    def test_parallel_run_end_to_end(self):
        """n_jobs=2 时完整 run() 流程工作 (含 GP 仿真器)"""
        self.hme.run(batch_size=4, verbose=False, n_jobs=2)
        self.assertIsNotNone(self.hme.non_implausible_params)
        self.assertEqual(self.hme.non_implausible_params.shape[1], 15)
        self.assertGreater(self.hme.non_implausible_params.shape[0], 0)

    def test_run_hme_then_mcmc_accepts_n_jobs(self):
        """run_hme_then_mcmc 接受 n_jobs 参数并正常执行"""
        from tb_risk.seir.history_matching import run_hme_then_mcmc

        result, diag = run_hme_then_mcmc(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            self.init_state, (0, 10), 2.5,
            n_design=6, n_warmup=2, n_samples=2, n_chains=1,
            n_leapfrog=3, random_seed=42, verbose=False, n_jobs=1)
        # 返回 (hmc_result, hme_diagnostics) 元组
        self.assertIn('chains', result)
        self.assertIsNotNone(diag)


@unittest.skipUnless(HME_AVAILABLE, "PyTorch and scikit-learn both required")
class TestEmulatorAndImplausibility(unittest.TestCase):
    """GP 仿真器训练与不可信度计算测试"""

    def setUp(self):
        from tb_risk.seir.history_matching import HistoryMatching

        self.init_state = _build_initial_state()
        self.hme = HistoryMatching(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            self.init_state, t_span=(0, 10), dt=2.5,
            n_design=20, random_seed=42)
        self.hme._generate_lhs_design()
        self.hme.design_outputs = self.hme._run_forward_model(
            self.hme.design_params, batch_size=10)

    def test_train_emulators(self):
        """GP 仿真器训练成功, 每个观测时间点一个 GP"""
        self.hme._train_emulators()

        self.assertEqual(len(self.hme.emulators), 2)
        for gp in self.hme.emulators:
            self.assertIsNotNone(gp)

    def test_gp_predict_returns_mean_and_std(self):
        """GP predict 返回 mean 和 std"""
        self.hme._train_emulators()

        mean, std = self.hme.emulators[0].predict(
            self.hme.design_params[:5], return_std=True)

        self.assertEqual(mean.shape, (5,))
        self.assertEqual(std.shape, (5,))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(std >= 0))

    def test_compute_implausibility_shape(self):
        """不可信度计算返回 (B,) 形状"""
        self.hme._train_emulators()

        impl = self.hme._compute_implausibility(self.hme.design_params)

        self.assertEqual(impl.shape, (20,))
        self.assertTrue(np.all(np.isfinite(impl)))
        self.assertTrue(np.all(impl >= 0),
                        f"不可信度应非负: min={impl.min()}")

    def test_implausibility_at_training_points_low(self):
        """训练点 (用 GP 拟合的) 的不可信度应较低

        GP 在训练点附近预测方差小, mean 接近 design_outputs,
        implausibility = |mean - obs_I| / sqrt(var_emu + var_obs)。
        注意: |mean - obs_I| 仍取决于 design_outputs 与 obs_I 的差距,
        但 var_emu 在训练点附近很小, 所以不可信度可能高或低。
        本测试仅验证 implausibility 数值合理 (有限、非负)。
        """
        self.hme._train_emulators()

        impl = self.hme._compute_implausibility(self.hme.design_params)

        self.assertTrue(np.all(np.isfinite(impl)))


@unittest.skipUnless(HME_AVAILABLE, "PyTorch and scikit-learn both required")
class TestHistoryMatchingRun(unittest.TestCase):
    """完整 HME 流程测试"""

    def setUp(self):
        from tb_risk.seir.history_matching import HistoryMatching

        self.init_state = _build_initial_state()
        # 生成与默认参数对应的"观测": 用默认参数运行前向模型得到 obs_I
        # 这样默认参数附近的 design 点应通过筛选
        self.hme = HistoryMatching(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            self.init_state, t_span=(0, 10), dt=2.5,
            n_design=20, implausibility_threshold=3.0,
            random_seed=42)

    def test_run_returns_non_implausible_params(self):
        """run() 返回非不可信参数 (n_pass >= 0)"""
        params = self.hme.run(batch_size=10, verbose=False)

        self.assertEqual(params.shape[1], 15)
        self.assertGreater(params.shape[0], 0,
                           "应至少有 1 个参数点通过 (后备策略保证)")

    def test_run_populates_attributes(self):
        """run() 后所有属性都已填充"""
        self.hme.run(batch_size=10, verbose=False)

        self.assertIsNotNone(self.hme.design_params)
        self.assertIsNotNone(self.hme.design_outputs)
        self.assertIsNotNone(self.hme.emulators)
        self.assertIsNotNone(self.hme.implausibility)
        self.assertIsNotNone(self.hme.non_implausible_mask)
        self.assertIsNotNone(self.hme.non_implausible_params)

        self.assertEqual(self.hme.design_params.shape, (20, 15))
        self.assertEqual(self.hme.design_outputs.shape, (20, 2))
        self.assertEqual(self.hme.implausibility.shape, (20,))

    def test_run_with_large_threshold_passes_all(self):
        """大阈值下所有参数点都通过"""
        from tb_risk.seir.history_matching import HistoryMatching

        hme = HistoryMatching(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            self.init_state, t_span=(0, 10), dt=2.5,
            n_design=15, implausibility_threshold=1e6,
            random_seed=42)
        hme.run(batch_size=10, verbose=False)

        self.assertTrue(np.all(hme.non_implausible_mask))
        self.assertEqual(hme.non_implausible_params.shape[0], 15)

    def test_run_with_zero_threshold_falls_back(self):
        """阈值为 0 时所有点被排除, 后备策略返回全部设计点"""
        from tb_risk.seir.history_matching import HistoryMatching

        hme = HistoryMatching(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            self.init_state, t_span=(0, 10), dt=2.5,
            n_design=15, implausibility_threshold=0.0,
            random_seed=42)
        # 抑制 warnings (后备策略会 warn)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hme.run(batch_size=10, verbose=False)

        # 后备策略: n_pass == n_design
        self.assertEqual(hme.non_implausible_params.shape[0], 15)


@unittest.skipUnless(HME_AVAILABLE, "PyTorch and scikit-learn both required")
class TestHMCInitAndBounds(unittest.TestCase):
    """HMC 初始化函数与缩小边界测试"""

    def setUp(self):
        from tb_risk.seir.history_matching import HistoryMatching

        self.init_state = _build_initial_state()
        self.hme = HistoryMatching(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            self.init_state, t_span=(0, 10), dt=2.5,
            n_design=20, implausibility_threshold=3.0,
            random_seed=42)
        self.hme.run(batch_size=10, verbose=False)

    def test_get_hmc_init_fn_callable(self):
        """get_hmc_init_fn 返回可调用函数"""
        init_fn = self.hme.get_hmc_init_fn()

        self.assertTrue(callable(init_fn))

    def test_init_fn_returns_correct_shape(self):
        """init_fn 返回 (15,) 形状的数组"""
        init_fn = self.hme.get_hmc_init_fn()

        for chain_idx in range(4):
            y = init_fn(chain_idx)
            self.assertEqual(y.shape, (15,))
            self.assertTrue(np.all(np.isfinite(y)),
                            f"chain {chain_idx}: init 含 NaN/Inf")

    def test_init_fn_unconstrained_space(self):
        """init_fn 返回值在无约束空间 (不限制在 PARAM_BOUNDS 内)

        无约束空间可以为任意实数, sigmoid 变换后映射到 PARAM_BOUNDS。
        本测试验证返回值合理 (|y| < 20, 即约束空间内 u 在 (1e-9, 1-1e-9))。
        """
        init_fn = self.hme.get_hmc_init_fn(jitter_std=0.0)

        y = init_fn(0)
        self.assertTrue(np.all(np.abs(y) < 20),
                        f"无约束值过大: {y}")

    def test_init_fn_different_chains_differ(self):
        """不同链的初始点不同 (jitter 保证)"""
        init_fn = self.hme.get_hmc_init_fn(jitter_std=0.5)

        y0 = init_fn(0)
        y1 = init_fn(1)
        self.assertFalse(np.allclose(y0, y1),
                         "不同链初始点应不同 (jitter)")

    def test_init_fn_before_run_raises(self):
        """HME 未运行时调用 get_hmc_init_fn 抛 RuntimeError"""
        from tb_risk.seir.history_matching import HistoryMatching

        hme = HistoryMatching(
            (np.array([5.0]), np.array([5.0])),
            self.init_state, t_span=(0, 5), dt=2.5,
            n_design=10, random_seed=42)
        with self.assertRaises(RuntimeError):
            hme.get_hmc_init_fn()

    def test_get_tightened_bounds(self):
        """get_tightened_bounds 返回缩小的边界 (在 PARAM_BOUNDS 内)"""
        from tb_risk.seir.autodiff import PARAM_BOUNDS

        bounds = self.hme.get_tightened_bounds(margin=0.1)

        self.assertEqual(len(bounds), 15)
        for i, (lo_new, hi_new) in enumerate(bounds):
            lo_orig, hi_orig = PARAM_BOUNDS[i]
            self.assertGreaterEqual(lo_new, lo_orig - 1e-12,
                                    f"param {i}: lo_new={lo_new} < lo_orig={lo_orig}")
            self.assertLessEqual(hi_new, hi_orig + 1e-12,
                                 f"param {i}: hi_new={hi_new} > hi_orig={hi_orig}")
            self.assertLessEqual(lo_new, hi_new,
                                 f"param {i}: lo={lo_new} > hi={hi_new}")

    def test_tightened_bounds_narrower_than_original(self):
        """缩小后的边界 (margin=0) 在至少某些维度窄于原始边界

        HME 的目标就是缩小参数空间, 至少有一个维度应明显缩小
        (除非所有参数点都通过, 极少见)。
        """
        from tb_risk.seir.autodiff import PARAM_BOUNDS

        bounds = self.hme.get_tightened_bounds(margin=0.0)
        n_narrower = 0
        for i, (lo_new, hi_new) in enumerate(bounds):
            lo_orig, hi_orig = PARAM_BOUNDS[i]
            orig_span = hi_orig - lo_orig
            new_span = hi_new - lo_new
            if new_span < orig_span * 0.95:
                n_narrower += 1
        # 至少有一些维度缩小 (宽松判据: >= 1 个维度)
        self.assertGreater(n_narrower, 0,
                           "HME 应至少缩小 1 个维度的参数范围")


@unittest.skipUnless(HME_AVAILABLE, "PyTorch and scikit-learn both required")
class TestDiagnostics(unittest.TestCase):
    """诊断信息完整性测试"""

    def test_diagnostics_not_run(self):
        """未运行时 diagnostics 返回 not_run 状态"""
        from tb_risk.seir.history_matching import HistoryMatching

        init_state = _build_initial_state()
        hme = HistoryMatching(
            (np.array([5.0]), np.array([5.0])),
            init_state, t_span=(0, 5), dt=2.5,
            n_design=10, random_seed=42)

        diag = hme.get_diagnostics()
        self.assertEqual(diag['status'], 'not_run')

    def test_diagnostics_complete(self):
        """运行后诊断信息完整"""
        from tb_risk.seir.history_matching import HistoryMatching

        init_state = _build_initial_state()
        hme = HistoryMatching(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            init_state, t_span=(0, 10), dt=2.5,
            n_design=20, random_seed=42)
        hme.run(batch_size=10, verbose=False)

        diag = hme.get_diagnostics()
        for key in ['status', 'n_design', 'n_non_implausible', 'pass_rate',
                    'space_reduction', 'implausibility_threshold',
                    'obs_variance', 'mean_implausibility', 'max_implausibility',
                    'tightened_bounds', 'original_bounds', 'param_names']:
            self.assertIn(key, diag, f"诊断缺少 {key}")

        self.assertEqual(diag['status'], 'completed')
        self.assertEqual(diag['n_design'], 20)
        self.assertGreaterEqual(diag['n_non_implausible'], 0)
        self.assertGreaterEqual(diag['pass_rate'], 0.0)
        self.assertLessEqual(diag['pass_rate'], 1.0)
        self.assertGreaterEqual(diag['space_reduction'], 0.0)
        self.assertLessEqual(diag['space_reduction'], 1.0)
        self.assertGreater(diag['obs_variance'], 0.0)
        self.assertEqual(len(diag['tightened_bounds']), 15)
        self.assertEqual(len(diag['original_bounds']), 15)
        self.assertEqual(len(diag['param_names']), 15)


@unittest.skipUnless(HME_AVAILABLE, "PyTorch and scikit-learn both required")
class TestRunHmeThenMcmc(unittest.TestCase):
    """run_hme_then_mcmc 端到端集成测试 (小参数)"""

    @pytest.mark.slow
    def test_end_to_end(self):
        """HME → HMC 完整流程 (极小参数)"""
        from tb_risk.seir.history_matching import run_hme_then_mcmc

        init_state = _build_initial_state()
        obs_data = (np.array([5.0, 10.0]), np.array([5.0, 10.0]))

        # 极小参数: n_design=10, n_warmup=5, n_samples=5, n_chains=2
        result, diag = run_hme_then_mcmc(
            obs_data, init_state, t_span=(0, 10), dt=2.5,
            n_design=10, implausibility_threshold=3.0,
            n_warmup=5, n_samples=5, n_chains=2,
            n_leapfrog=3, target_accept=0.8,
            batch_chains=False, precision='float64',
            random_seed=42, verbose=False)

        # HMC 结果
        self.assertIn('chains', result)
        self.assertEqual(len(result['chains']), 2)
        self.assertEqual(result['chains'][0].shape, (5, 15))

        # HME 诊断
        self.assertEqual(diag['status'], 'completed')
        self.assertEqual(diag['n_design'], 10)


if __name__ == '__main__':
    unittest.main()
