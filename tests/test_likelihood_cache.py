#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — 似然缓存 (LikelihoodCache)

测试改进15 (六): MD5 hash 似然缓存 + LRU 淘汰。
在 HMC/v3 MH 中, 当同一参数集被重复查询时, 缓存可跳过昂贵的
ODE/SDE 积分。

覆盖:
  1. hash_theta: numpy/torch 一致性
  2. LikelihoodCache 基本 put/get
  3. LRU 淘汰策略
  4. 统计信息 (hits/misses/hit_rate)
  5. 线程安全 (threading.Lock)
  6. 边界条件 (maxsize=1, 空缓存)
  7. HMC sampler use_hash_cache 集成
  8. v3 MH use_hash_cache 集成

文献:
  Andrieu & Doucet (2010) JRSS-B 72(3):269-342 — detailed balance
"""

import os
import sys
import threading
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
    torch = None  # 确保torch符号始终有定义，避免内层装饰器引用时NameError


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestHashTheta(unittest.TestCase):
    """hash_theta 函数测试"""

    def test_numpy_array(self):
        """numpy 数组可哈希"""
        from tb_risk.seir.likelihood_cache import hash_theta
        h = hash_theta(np.array([1.0, 2.0, 3.0]))
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 32)  # MD5 hex

    def test_torch_tensor(self):
        """torch.Tensor 可哈希"""
        from tb_risk.seir.likelihood_cache import hash_theta
        h = hash_theta(torch.tensor([1.0, 2.0, 3.0]))
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 32)

    def test_numpy_torch_consistency(self):
        """同值 numpy 和 torch 产生相同哈希"""
        from tb_risk.seir.likelihood_cache import hash_theta
        h_n = hash_theta(np.array([1.0, 2.0, 3.0]))
        h_t = hash_theta(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        self.assertEqual(h_n, h_t)

    def test_different_values_different_hash(self):
        """不同值产生不同哈希"""
        from tb_risk.seir.likelihood_cache import hash_theta
        h1 = hash_theta(np.array([1.0, 2.0, 3.0]))
        h2 = hash_theta(np.array([1.0, 2.0, 3.1]))
        self.assertNotEqual(h1, h2)

    def test_same_value_same_hash(self):
        """同值 (不同对象) 产生相同哈希"""
        from tb_risk.seir.likelihood_cache import hash_theta
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])  # 不同对象, 同值
        self.assertEqual(hash_theta(a), hash_theta(b))

    def test_dtype_invariance(self):
        """float32 和 float64 同值产生相同哈希 (内部统一转 float64)"""
        from tb_risk.seir.likelihood_cache import hash_theta
        h32 = hash_theta(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        h64 = hash_theta(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        self.assertEqual(h32, h64)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestLikelihoodCacheBasic(unittest.TestCase):
    """LikelihoodCache 基本功能测试"""

    def test_put_get(self):
        """put 后 get 命中"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=10)
        theta = np.array([1.0, 2.0])
        cache.put(theta, -10.5, np.array([0.1, 0.2]))
        result = cache.get(theta)
        self.assertIsNotNone(result)
        log_post, grad = result
        self.assertEqual(log_post, -10.5)
        np.testing.assert_array_equal(grad, [0.1, 0.2])

    def test_miss_returns_none(self):
        """未存入的 theta 返回 None"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=10)
        self.assertIsNone(cache.get(np.array([1.0, 2.0])))

    def test_value_based_hit(self):
        """值匹配即命中 (不依赖对象身份)"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=10)
        theta1 = np.array([1.0, 2.0, 3.0])
        theta2 = np.array([1.0, 2.0, 3.0])  # 不同对象, 同值
        cache.put(theta1, -5.0, np.array([0.0, 0.0, 0.0]))
        self.assertIsNotNone(cache.get(theta2))

    def test_contains(self):
        """__contains__ 支持 `in` 语法"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=10)
        theta = np.array([1.0])
        cache.put(theta, 1.0, np.array([0.0]))
        self.assertIn(theta, cache)
        self.assertNotIn(np.array([2.0]), cache)

    def test_len(self):
        """__len__ 返回当前条目数"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=10)
        self.assertEqual(len(cache), 0)
        cache.put(np.array([1.0]), 1.0, np.array([0.0]))
        self.assertEqual(len(cache), 1)
        cache.put(np.array([2.0]), 2.0, np.array([0.0]))
        self.assertEqual(len(cache), 2)

    def test_clear(self):
        """clear 清空缓存和统计"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=10)
        cache.put(np.array([1.0]), 1.0, np.array([0.0]))
        cache.get(np.array([1.0]))  # hit
        cache.get(np.array([2.0]))  # miss
        cache.clear()
        self.assertEqual(len(cache), 0)
        stats = cache.stats()
        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestLRUEviction(unittest.TestCase):
    """LRU 淘汰策略测试"""

    def test_evict_oldest(self):
        """超出 maxsize 时淘汰最久未访问"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=2)
        cache.put(np.array([1.0]), 1.0, np.array([0.0]))
        cache.put(np.array([2.0]), 2.0, np.array([0.0]))
        cache.put(np.array([3.0]), 3.0, np.array([0.0]))  # 淘汰 [1.0]
        self.assertIsNone(cache.get(np.array([1.0])))
        self.assertIsNotNone(cache.get(np.array([2.0])))
        self.assertIsNotNone(cache.get(np.array([3.0])))

    def test_lru_updates_on_get(self):
        """get 访问更新 LRU 顺序, 避免被淘汰"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=2)
        cache.put(np.array([1.0]), 1.0, np.array([0.0]))
        cache.put(np.array([2.0]), 2.0, np.array([0.0]))
        # 访问 [1.0], 使其成为最近访问
        cache.get(np.array([1.0]))
        cache.put(np.array([3.0]), 3.0, np.array([0.0]))  # 淘汰 [2.0]
        self.assertIsNotNone(cache.get(np.array([1.0])))
        self.assertIsNone(cache.get(np.array([2.0])))

    def test_lru_updates_on_put_existing(self):
        """put 已存在的键更新 LRU 顺序"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=2)
        cache.put(np.array([1.0]), 1.0, np.array([0.0]))
        cache.put(np.array([2.0]), 2.0, np.array([0.0]))
        # 重新 put [1.0], 更新值和 LRU 顺序
        cache.put(np.array([1.0]), 1.5, np.array([0.5]))
        cache.put(np.array([3.0]), 3.0, np.array([0.0]))  # 淘汰 [2.0]
        self.assertIsNotNone(cache.get(np.array([1.0])))
        result = cache.get(np.array([1.0]))
        self.assertEqual(result[0], 1.5)  # 更新后的值
        self.assertIsNone(cache.get(np.array([2.0])))

    def test_maxsize_one(self):
        """maxsize=1 时只保留最近一个"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=1)
        cache.put(np.array([1.0]), 1.0, np.array([0.0]))
        cache.put(np.array([2.0]), 2.0, np.array([0.0]))
        self.assertIsNone(cache.get(np.array([1.0])))
        self.assertIsNotNone(cache.get(np.array([2.0])))

    def test_invalid_maxsize_raises(self):
        """maxsize < 1 抛 ValueError"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        with self.assertRaises(ValueError):
            LikelihoodCache(maxsize=0)
        with self.assertRaises(ValueError):
            LikelihoodCache(maxsize=-1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestCacheStats(unittest.TestCase):
    """缓存统计测试"""

    def test_initial_stats(self):
        """初始统计全为 0"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=10)
        stats = cache.stats()
        self.assertEqual(stats['size'], 0)
        self.assertEqual(stats['maxsize'], 10)
        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 0)
        self.assertEqual(stats['hit_rate'], 0.0)

    def test_hit_miss_counts(self):
        """hit/miss 计数正确"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=10)
        cache.put(np.array([1.0]), 1.0, np.array([0.0]))
        cache.get(np.array([1.0]))  # hit
        cache.get(np.array([1.0]))  # hit
        cache.get(np.array([2.0]))  # miss
        stats = cache.stats()
        self.assertEqual(stats['hits'], 2)
        self.assertEqual(stats['misses'], 1)
        self.assertAlmostEqual(stats['hit_rate'], 2.0 / 3.0)

    def test_stats_after_clear(self):
        """clear 重置统计"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=10)
        cache.put(np.array([1.0]), 1.0, np.array([0.0]))
        cache.get(np.array([1.0]))
        cache.clear()
        stats = cache.stats()
        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 0)
        self.assertEqual(stats['hit_rate'], 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestThreadSafety(unittest.TestCase):
    """线程安全测试"""

    def test_concurrent_access(self):
        """多线程并发 put/get 不出错"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=100)
        errors = []

        def worker(seed):
            try:
                rng = np.random.RandomState(seed)
                for _ in range(50):
                    theta = rng.randn(5)
                    cache.put(theta, float(rng.randn()), rng.randn(5))
                    cache.get(theta)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0, f"线程错误: {errors}")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestHMCIntegration(unittest.TestCase):
    """HMC sampler use_hash_cache 集成测试"""

    def test_hmc_accepts_use_hash_cache_param(self):
        """HMCSampler 接受 use_hash_cache 参数"""
        from tb_risk.seir.hmc_sampler import HMCSampler

        def log_post(theta):
            return -0.5 * torch.sum(theta ** 2)

        def grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            lp = log_post(y)
            g, = torch.autograd.grad(lp, y)
            return g.detach(), lp.detach()

        sampler = HMCSampler(
            log_post, grad_fn, n_params=3,
            n_warmup=2, n_samples=2, n_chains=1,
            use_hash_cache=True, cache_maxsize=64)
        self.assertTrue(sampler.use_hash_cache)
        self.assertIsNotNone(sampler._hash_cache)

    def test_hmc_default_no_cache(self):
        """默认不启用 hash cache"""
        from tb_risk.seir.hmc_sampler import HMCSampler

        def log_post(theta):
            return -0.5 * torch.sum(theta ** 2)

        def grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            lp = log_post(y)
            g, = torch.autograd.grad(lp, y)
            return g.detach(), lp.detach()

        sampler = HMCSampler(
            log_post, grad_fn, n_params=3,
            n_warmup=2, n_samples=2, n_chains=1)
        self.assertFalse(sampler.use_hash_cache)
        self.assertIsNone(sampler._hash_cache)

    def test_hmc_with_cache_runs(self):
        """启用 cache 后 HMC 可正常运行"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        torch.manual_seed(42)

        def log_post(theta):
            return -0.5 * torch.sum(theta ** 2)

        def grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            lp = log_post(y)
            g, = torch.autograd.grad(lp, y)
            return g.detach(), lp.detach()

        sampler = HMCSampler(
            log_post, grad_fn, n_params=3,
            n_warmup=5, n_samples=5, n_chains=2,
            n_leapfrog=3, use_hash_cache=True, cache_maxsize=32)

        def init_fn(chain_idx):
            return torch.zeros(3, dtype=torch.float64)

        result = sampler.sample(init_params_fn=init_fn, batch_chains=False)
        self.assertEqual(len(result['chains']), 2)
        # cache stats 应出现在诊断中
        self.assertIn('hash_cache_stats', result['diagnostics'])
        stats = result['diagnostics']['hash_cache_stats']
        self.assertIn('hits', stats)
        self.assertIn('misses', stats)

    def test_hmc_with_and_without_cache_similar(self):
        """启用/不启用 cache 的 HMC 采样结果应相似 (同一随机种子)"""
        from tb_risk.seir.hmc_sampler import HMCSampler
        torch.manual_seed(42)

        def log_post(theta):
            return -0.5 * torch.sum(theta ** 2)

        def grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            lp = log_post(y)
            g, = torch.autograd.grad(lp, y)
            return g.detach(), lp.detach()

        def init_fn(chain_idx):
            return torch.zeros(3, dtype=torch.float64)

        # 不启用 cache
        torch.manual_seed(42)
        sampler1 = HMCSampler(
            log_post, grad_fn, n_params=3,
            n_warmup=5, n_samples=10, n_chains=1,
            n_leapfrog=3, use_hash_cache=False)
        result1 = sampler1.sample(init_params_fn=init_fn, batch_chains=False)

        # 启用 cache (同种子)
        torch.manual_seed(42)
        sampler2 = HMCSampler(
            log_post, grad_fn, n_params=3,
            n_warmup=5, n_samples=10, n_chains=1,
            n_leapfrog=3, use_hash_cache=True, cache_maxsize=64)
        result2 = sampler2.sample(init_params_fn=init_fn, batch_chains=False)

        # 样本应完全一致 (cache 不影响数值结果, 仅跳过重复计算)
        np.testing.assert_array_almost_equal(
            result1['chains'][0], result2['chains'][0], decimal=10)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestV3MHIntegration(unittest.TestCase):
    """v3 MH use_hash_cache 集成测试"""

    def test_v3_accepts_use_hash_cache_param(self):
        """run_mcmc_v3 接受 use_hash_cache 参数 (不抛异常)"""
        from tb_risk.seir.inference import BayesianInferenceV3

        # v3 9-compartment state (S, Lf, Ls, M, Isub, Isp, Isn, R, C)
        init_state = np.zeros(9)
        init_state[0] = 900.0  # S
        init_state[5] = 10.0   # Isp

        engine = BayesianInferenceV3(seed=42)
        # 极小参数, 仅验证参数被接受
        posterior = engine.run_mcmc_v3(
            (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
            init_state, (0, 10), 2.5,
            n_iterations=10, burn_in=2, n_chains=1,
            use_hash_cache=True, cache_maxsize=32)

        # 应返回后验样本 (非空)
        self.assertGreater(len(posterior), 0)
        # 诊断应含 cache stats
        diag = engine.convergence_diagnostics
        self.assertIn('hash_cache_stats', diag)
        self.assertEqual(len(diag['hash_cache_stats']), 1)  # 1 链


# ======================================================================
# 第五轮改进测试
# ======================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestHashThetaContiguity(unittest.TestCase):
    """要点5 (第五轮): hash_theta 跳过 ascontiguousarray 当已 C 连续"""

    def test_contiguous_array_same_hash(self):
        """连续与非连续数组同值产生相同哈希"""
        from tb_risk.seir.likelihood_cache import hash_theta
        # 连续数组
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        # 非连续视图 (切片步长 2)
        b_noncontig = np.array([1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0],
                                dtype=np.float64)[::2]
        self.assertFalse(b_noncontig.flags['C_CONTIGUOUS'],
                         "测试前提: b 应为非连续")
        # 同值应产生相同哈希
        self.assertEqual(hash_theta(a), hash_theta(b_noncontig))

    def test_contiguous_no_copy(self):
        """已连续的 float64 数组直接使用 (跳过 ascontiguousarray)"""
        from tb_risk.seir.likelihood_cache import hash_theta
        # 连续 float64 数组 (常见情况)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        self.assertTrue(a.flags['C_CONTIGUOUS'])
        # 哈希应正常计算
        h = hash_theta(a)
        self.assertEqual(len(h), 32)

    def test_torch_contiguous_same_as_numpy(self):
        """torch 张量 (CPU) 与 numpy 同值产生相同哈希"""
        from tb_risk.seir.likelihood_cache import hash_theta
        arr_np = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        arr_torch = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)
        self.assertEqual(hash_theta(arr_np), hash_theta(arr_torch))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestGpuAutoDisable(unittest.TestCase):
    """要点2 (第五轮): GPU 环境自动禁用 hash cache"""

    def test_is_gpu_tensor_cpu_returns_false(self):
        """is_gpu_tensor 对 CPU 张量返回 False"""
        from tb_risk.seir.likelihood_cache import is_gpu_tensor
        cpu_tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
        self.assertFalse(is_gpu_tensor(cpu_tensor))
        self.assertFalse(is_gpu_tensor(np.array([1.0, 2.0])))
        self.assertFalse(is_gpu_tensor("not a tensor"))

    @unittest.skipUnless(TORCH_AVAILABLE and torch is not None and torch.cuda.is_available(), "CUDA not available")
    def test_is_gpu_tensor_cuda_returns_true(self):
        """is_gpu_tensor 对 CUDA 张量返回 True (需 GPU)"""
        from tb_risk.seir.likelihood_cache import is_gpu_tensor
        cuda_tensor = torch.tensor([1.0, 2.0], device='cuda')
        self.assertTrue(is_gpu_tensor(cuda_tensor))

    def test_cache_works_with_cpu_tensor(self):
        """LikelihoodCache 对 CPU 张量正常工作 (不禁用)"""
        from tb_risk.seir.likelihood_cache import LikelihoodCache
        cache = LikelihoodCache(maxsize=10, auto_disable_on_gpu=True)
        theta = torch.tensor([1.0, 2.0], dtype=torch.float64)
        cache.put(theta, -5.0, torch.tensor([0.1, 0.2]))
        result = cache.get(theta)
        self.assertIsNotNone(result)
        self.assertFalse(cache.disabled)

    def test_cache_auto_disable_on_gpu_tensor(self):
        """LikelihoodCache 检测到 GPU 张量时自动禁用 (mock is_gpu_tensor)"""
        from tb_risk.seir import likelihood_cache
        from tb_risk.seir.likelihood_cache import LikelihoodCache

        original = likelihood_cache.is_gpu_tensor
        try:
            # mock: 所有张量视为 GPU
            likelihood_cache.is_gpu_tensor = lambda theta: True

            cache = LikelihoodCache(maxsize=10, auto_disable_on_gpu=True)
            theta = torch.tensor([1.0, 2.0], dtype=torch.float64)
            # 首次 put 应触发警告 (检测到 GPU 张量)
            import warnings as _warnings
            with _warnings.catch_warnings(record=True) as w:
                _warnings.simplefilter("always")
                cache.put(theta, -5.0, torch.tensor([0.1, 0.2]))
            self.assertTrue(any('GPU' in str(wi.message) for wi in w),
                            f"应触发 GPU 警告, got: {[str(wi.message) for wi in w]}")
            # get 应返回 None (已禁用)
            self.assertIsNone(cache.get(theta))
            self.assertTrue(cache.disabled)
            # stats 应反映禁用状态
            stats = cache.stats()
            self.assertTrue(stats['disabled'])
        finally:
            likelihood_cache.is_gpu_tensor = original

    def test_cache_no_auto_disable_when_flag_off(self):
        """auto_disable_on_gpu=False 时不禁用"""
        from tb_risk.seir import likelihood_cache
        from tb_risk.seir.likelihood_cache import LikelihoodCache

        original = likelihood_cache.is_gpu_tensor
        try:
            likelihood_cache.is_gpu_tensor = lambda theta: True
            cache = LikelihoodCache(
                maxsize=10, auto_disable_on_gpu=False)
            theta = torch.tensor([1.0, 2.0], dtype=torch.float64)
            cache.put(theta, -5.0, torch.tensor([0.1, 0.2]))
            result = cache.get(theta)
            self.assertIsNotNone(result)
            self.assertFalse(cache.disabled)
        finally:
            likelihood_cache.is_gpu_tensor = original

    def test_hmc_sampler_force_disables_hash_cache_on_cuda(self):
        """HMCSampler 在 device='cuda' 时强制 use_hash_cache=False"""
        from tb_risk.seir.hmc_sampler import HMCSampler

        def log_post(theta):
            return -0.5 * torch.sum(theta ** 2)

        def grad_fn(theta):
            y = theta.clone().detach().requires_grad_(True)
            lp = log_post(y)
            g, = torch.autograd.grad(lp, y)
            return g.detach(), lp.detach()

        # 用 mock device 测试: 直接构造 sampler 时 device='cuda'
        # 但实际不执行 GPU 操作 (仅测试 __init__ 逻辑)
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            sampler = HMCSampler(
                log_post, grad_fn, n_params=3,
                n_warmup=5, n_samples=10, n_chains=1,
                device='cuda',  # 触发强制禁用
                use_hash_cache=True,  # 用户请求启用
            )
            # 应有警告
            self.assertTrue(any('use_hash_cache' in str(wi.message)
                                for wi in w))
            # 应被强制禁用
            self.assertFalse(sampler.use_hash_cache)
            self.assertIsNone(sampler._hash_cache)


if __name__ == '__main__':
    unittest.main()
