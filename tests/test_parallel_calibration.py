#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — 多区域并行校准 (Multi-Region Parallel Calibration)

测试改进: 多区域并行校准 (ProcessPoolExecutor + threadpoolctl)
各省份/区域的 MCMC 链完全独立, 可并行执行接近线性加速。

覆盖:
  1. 模块导入与 worker 函数可 pickle (ProcessPoolExecutor 必需)
  2. 单区域串行校准 (worker 函数直接调用)
  3. ParallelCalibrator 串行模式 (calibrate_regions_serial)
  4. ParallelCalibrator 并行模式 (calibrate_regions, ProcessPoolExecutor)
  5. 错误隔离 (一个区域失败不影响其他)
  6. 便捷函数 calibrate_regions_parallel
  7. 线程限制 (threadpoolctl)
  8. 版本选择 (v3 / v4)

注意: Windows 下 ProcessPoolExecutor 使用 spawn, worker 函数必须可 pickle
(顶层函数)。pytest 测试函数本身不在主模块顶层, 不受 spawn 限制。

文献:
  Python 官方文档 concurrent.futures — ProcessPoolExecutor
  Mathieu et al. (2021) JOSS 6(64):3242 — threadpoolctl
"""

import os
import pickle
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
    import threadpoolctl
    THREADPOOLCTL_AVAILABLE = True
except ImportError:
    THREADPOOLCTL_AVAILABLE = False


def _build_initial_state():
    """构造 270D 初始状态: 900 S + 10 Isp 在 15-49 岁 HIV-/DS"""
    state = np.zeros(270)
    base = 2 * 3 * 2 * 9 + 0 * 2 * 9 + 0 * 9
    state[base + 0] = 900.0  # S
    state[base + 5] = 10.0   # Isp
    return state


def _make_region(region_id, seed, n_warmup=5, n_samples=5, n_chains=2):
    """构造一个小参数的区域配置 (用于快速测试)"""
    return {
        'region_id': region_id,
        'observed_data': (np.array([5.0, 10.0]), np.array([5.0, 10.0])),
        'initial_state': _build_initial_state(),
        't_span': (0, 10), 'dt': 2.5,
        'n_warmup': n_warmup, 'n_samples': n_samples,
        'n_chains': n_chains, 'n_leapfrog': 3,
        'seed': seed,
    }


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestModuleImport(unittest.TestCase):
    """模块导入与 worker 函数可 pickle 测试"""

    def test_module_import(self):
        """parallel_calibration 模块可正常导入"""
        from tb_risk.seir import parallel_calibration
        self.assertTrue(hasattr(parallel_calibration, 'ParallelCalibrator'))
        self.assertTrue(hasattr(parallel_calibration, 'calibrate_region_v4'))
        self.assertTrue(hasattr(parallel_calibration, 'calibrate_region_v3'))
        self.assertTrue(hasattr(parallel_calibration, 'calibrate_regions_parallel'))

    def test_worker_v4_pickleable(self):
        """calibrate_region_v4 可 pickle (ProcessPoolExecutor 必需)"""
        from tb_risk.seir.parallel_calibration import calibrate_region_v4
        pickled = pickle.dumps(calibrate_region_v4)
        unpickled = pickle.loads(pickled)
        self.assertTrue(callable(unpickled))

    def test_worker_v3_pickleable(self):
        """calibrate_region_v3 可 pickle"""
        from tb_risk.seir.parallel_calibration import calibrate_region_v3
        pickled = pickle.dumps(calibrate_region_v3)
        unpickled = pickle.loads(pickled)
        self.assertTrue(callable(unpickled))

    def test_threadpoolctl_available(self):
        """threadpoolctl 已安装 (worker 线程限制依赖)"""
        self.assertTrue(THREADPOOLCTL_AVAILABLE,
                        "threadpoolctl 应已安装 (scipy 依赖通常自带)")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestParallelCalibratorSetup(unittest.TestCase):
    """ParallelCalibrator 初始化测试"""

    def test_default_init(self):
        """默认初始化 (n_workers=os.cpu_count, version=v4)"""
        from tb_risk.seir.parallel_calibration import ParallelCalibrator
        cal = ParallelCalibrator()
        self.assertEqual(cal.version, 'v4')
        self.assertGreater(cal.n_workers, 0)
        self.assertTrue(cal.thread_limit)

    def test_custom_init(self):
        """自定义参数初始化"""
        from tb_risk.seir.parallel_calibration import ParallelCalibrator
        cal = ParallelCalibrator(n_workers=4, version='v3',
                                  thread_limit=False, use_hme=True)
        self.assertEqual(cal.n_workers, 4)
        self.assertEqual(cal.version, 'v3')
        self.assertFalse(cal.thread_limit)
        self.assertTrue(cal.use_hme)

    def test_invalid_version_raises(self):
        """非法 version 抛出 ValueError"""
        from tb_risk.seir.parallel_calibration import ParallelCalibrator
        with self.assertRaises(ValueError):
            ParallelCalibrator(version='v5')

    def test_worker_fn_selection(self):
        """version 决定 worker_fn"""
        from tb_risk.seir.parallel_calibration import (
            ParallelCalibrator, calibrate_region_v4, calibrate_region_v3)
        cal_v4 = ParallelCalibrator(version='v4')
        cal_v3 = ParallelCalibrator(version='v3')
        self.assertIs(cal_v4._worker_fn, calibrate_region_v4)
        self.assertIs(cal_v3._worker_fn, calibrate_region_v3)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestWorkerDirectCall(unittest.TestCase):
    """worker 函数直接调用测试 (不经过进程池)"""

    def test_v4_worker_success(self):
        """calibrate_region_v4 直接调用成功"""
        from tb_risk.seir.parallel_calibration import calibrate_region_v4
        region = _make_region('test_direct', seed=42)
        result = calibrate_region_v4(region)

        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['region_id'], 'test_direct')
        self.assertIn('posterior_samples', result)
        self.assertIn('traces', result)
        self.assertIn('diagnostics', result)
        # v4 后验应包含 15 个参数
        self.assertGreaterEqual(len(result['posterior_samples']), 15)

    def test_v4_worker_missing_key(self):
        """缺少必需键时返回 error (不抛异常)"""
        from tb_risk.seir.parallel_calibration import calibrate_region_v4
        bad_region = {'region_id': 'bad'}  # 缺 observed_data 等
        result = calibrate_region_v4(bad_region)

        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['region_id'], 'bad')
        self.assertIn('error', result)
        self.assertIn('KeyError', result['error'])

    def test_v4_worker_default_region_id(self):
        """region_config 无 region_id 时使用 'unknown'"""
        from tb_risk.seir.parallel_calibration import calibrate_region_v4
        region = _make_region('test_default', seed=42)
        del region['region_id']
        result = calibrate_region_v4(region)

        # 即使 region_id 缺失, worker 不抛异常 (用 'unknown')
        self.assertEqual(result['region_id'], 'unknown')


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSerialMode(unittest.TestCase):
    """串行模式测试 (calibrate_regions_serial)"""

    def test_serial_single_region(self):
        """单区域串行校准"""
        from tb_risk.seir.parallel_calibration import ParallelCalibrator
        cal = ParallelCalibrator(n_workers=2, version='v4')
        regions = [_make_region('A', seed=42)]
        results = cal.calibrate_regions_serial(regions, verbose=False)

        self.assertEqual(len(results), 1)
        self.assertEqual(results['A']['status'], 'success')

    def test_serial_multiple_regions(self):
        """多区域串行校准 (按顺序执行)"""
        from tb_risk.seir.parallel_calibration import ParallelCalibrator
        cal = ParallelCalibrator(n_workers=2, version='v4')
        regions = [_make_region('A', seed=42),
                   _make_region('B', seed=123)]
        results = cal.calibrate_regions_serial(regions, verbose=False)

        self.assertEqual(len(results), 2)
        self.assertEqual(results['A']['status'], 'success')
        self.assertEqual(results['B']['status'], 'success')

    def test_serial_empty_regions(self):
        """空区域列表返回空字典"""
        from tb_risk.seir.parallel_calibration import ParallelCalibrator
        cal = ParallelCalibrator()
        results = cal.calibrate_regions_serial([], verbose=False)
        self.assertEqual(results, {})

    def test_serial_error_isolation(self):
        """串行模式下一个区域失败不影响其他"""
        from tb_risk.seir.parallel_calibration import ParallelCalibrator
        cal = ParallelCalibrator(n_workers=2, version='v4')
        # 第一个区域配置错误 (缺 observed_data)
        bad_region = {'region_id': 'bad'}
        good_region = _make_region('good', seed=42)
        results = cal.calibrate_regions_serial(
            [bad_region, good_region], verbose=False)

        self.assertEqual(len(results), 2)
        self.assertEqual(results['bad']['status'], 'error')
        self.assertEqual(results['good']['status'], 'success')


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestParallelMode(unittest.TestCase):
    """并行模式测试 (ProcessPoolExecutor)

    注意: 这些测试会 spawn 子进程, 较慢。标记为 @slow。
    Windows 下 spawn 需要主模块有 ``if __name__ == '__main__':`` 保护,
    但 pytest 测试函数不在主模块顶层, 不受此限制。
    """

    @pytest.mark.slow
    def test_parallel_single_region(self):
        """单区域并行校准 (1 worker, 1 region)"""
        from tb_risk.seir.parallel_calibration import ParallelCalibrator
        cal = ParallelCalibrator(n_workers=1, version='v4')
        regions = [_make_region('solo', seed=42)]
        results = cal.calibrate_regions(regions, verbose=False)

        self.assertEqual(len(results), 1)
        self.assertEqual(results['solo']['status'], 'success')

    @pytest.mark.slow
    def test_parallel_two_regions(self):
        """2 区域 2 进程并行校准"""
        from tb_risk.seir.parallel_calibration import ParallelCalibrator
        cal = ParallelCalibrator(n_workers=2, version='v4')
        regions = [_make_region('A', seed=42),
                   _make_region('B', seed=123)]
        results = cal.calibrate_regions(regions, verbose=False)

        self.assertEqual(len(results), 2)
        self.assertEqual(results['A']['status'], 'success')
        self.assertEqual(results['B']['status'], 'success')

        # 两区域后验样本形状一致 (相同 n_samples × n_chains)
        a_beta = results['A']['posterior_samples']['beta']
        b_beta = results['B']['posterior_samples']['beta']
        self.assertEqual(a_beta.shape, b_beta.shape)

    @pytest.mark.slow
    def test_parallel_error_isolation(self):
        """并行模式下一个区域失败不影响其他"""
        from tb_risk.seir.parallel_calibration import ParallelCalibrator
        cal = ParallelCalibrator(n_workers=2, version='v4')
        bad_region = {'region_id': 'bad'}  # 缺必需键
        good_region = _make_region('good', seed=42)
        results = cal.calibrate_regions(
            [bad_region, good_region], verbose=False)

        self.assertEqual(len(results), 2)
        self.assertEqual(results['bad']['status'], 'error')
        self.assertEqual(results['good']['status'], 'success')

    @pytest.mark.slow
    def test_parallel_more_regions_than_workers(self):
        """区域数 > worker 数时排队执行"""
        from tb_risk.seir.parallel_calibration import ParallelCalibrator
        cal = ParallelCalibrator(n_workers=1, version='v4')
        regions = [_make_region(f'r{i}', seed=42 + i)
                   for i in range(3)]
        results = cal.calibrate_regions(regions, verbose=False)

        self.assertEqual(len(results), 3)
        for i in range(3):
            self.assertEqual(results[f'r{i}']['status'], 'success')


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestConvenienceFunction(unittest.TestCase):
    """便捷函数 calibrate_regions_parallel 测试"""

    @pytest.mark.slow
    def test_convenience_parallel(self):
        """calibrate_regions_parallel 便捷函数工作"""
        from tb_risk.seir.parallel_calibration import (
            calibrate_regions_parallel)
        regions = [_make_region('A', seed=42),
                   _make_region('B', seed=123)]
        results = calibrate_regions_parallel(
            regions, n_workers=2, version='v4', verbose=False)

        self.assertEqual(len(results), 2)
        self.assertEqual(results['A']['status'], 'success')
        self.assertEqual(results['B']['status'], 'success')


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestThreadLimiting(unittest.TestCase):
    """BLAS 线程限制测试"""

    def test_set_thread_limits(self):
        """_set_thread_limits 返回控制器对象"""
        if not THREADPOOLCTL_AVAILABLE:
            self.skipTest("threadpoolctl not available")
        from tb_risk.seir.parallel_calibration import _set_thread_limits
        controller = _set_thread_limits()
        self.assertIsNotNone(controller)

    def test_thread_limit_applied(self):
        """worker 调用后 BLAS 线程数限制为 1

        注意: threadpoolctl 限制仅在调用上下文有效, 此测试验证
        _set_thread_limits 不抛异常且返回有效控制器。
        """
        if not THREADPOOLCTL_AVAILABLE:
            self.skipTest("threadpoolctl not available")
        from tb_risk.seir.parallel_calibration import _set_thread_limits

        # 调用前可能有多线程 BLAS
        controller = _set_thread_limits()
        try:
            # 检查当前所有 BLAS 库的线程数
            infos = threadpoolctl.threadpool_info()
            for info in infos:
                self.assertLessEqual(info['num_threads'], 1,
                                     f"{info['prefix']} 线程数 > 1: "
                                     f"{info['num_threads']}")
        finally:
            if controller is not None:
                controller.unregister()


def _param_scan_square(params, offset=0.0):
    """顶层可 pickle 的参数扫描测试函数 (平方 + 偏移)"""
    return float(params * params) + offset


def _param_scan_may_fail(params, fail_at=0.0):
    """顶层可 pickle 的参数扫描测试函数 (指定点抛错, 验证错误隔离)"""
    if float(params) == float(fail_at):
        raise ValueError(f"fail on {params}")
    return float(params) * 2.0


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestParamScan(unittest.TestCase):
    """通用参数扫描 parallel_param_scan 测试"""

    def test_module_export(self):
        """parallel_param_scan 已导出且可调用"""
        from tb_risk.seir.parallel_calibration import (
            parallel_param_scan, _param_scan_worker)
        self.assertTrue(callable(parallel_param_scan))
        self.assertTrue(callable(_param_scan_worker))

    def test_worker_pickleable(self):
        """_param_scan_worker 可 pickle (ProcessPoolExecutor spawn 必需)"""
        from tb_risk.seir.parallel_calibration import _param_scan_worker
        pickled = pickle.dumps(_param_scan_worker)
        self.assertTrue(callable(pickle.loads(pickled)))

    def test_empty_input(self):
        """空参数列表返回空列表 (不启动进程池)"""
        from tb_risk.seir.parallel_calibration import parallel_param_scan
        self.assertEqual(
            parallel_param_scan(_param_scan_square, [], verbose=False), [])

    @pytest.mark.slow
    def test_basic_parallel(self):
        """多进程并行扫描返回与输入顺序一致的逐点结果"""
        from tb_risk.seir.parallel_calibration import parallel_param_scan
        param_list = [1.0, 2.0, 3.0, 4.0]
        results = parallel_param_scan(
            _param_scan_square, param_list, n_workers=2,
            fn_kwargs={'offset': 10.0}, verbose=False)
        self.assertEqual(len(results), 4)
        expected = {0: 11.0, 1: 14.0, 2: 19.0, 3: 26.0}
        for r in results:
            self.assertEqual(r['status'], 'success')
            self.assertAlmostEqual(
                r['result'], expected[r['index']], places=9)
        # 结果按输入索引排序返回
        self.assertEqual([r['index'] for r in results], [0, 1, 2, 3])

    @pytest.mark.slow
    def test_single_point(self):
        """单点扫描 (1 worker) 正常工作"""
        from tb_risk.seir.parallel_calibration import parallel_param_scan
        results = parallel_param_scan(
            _param_scan_square, [3.0], n_workers=1, verbose=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['status'], 'success')
        self.assertAlmostEqual(results[0]['result'], 9.0, places=9)

    @pytest.mark.slow
    def test_error_isolation(self):
        """单个参数点失败不影响其他点 (返回 status='error')"""
        from tb_risk.seir.parallel_calibration import parallel_param_scan
        results = parallel_param_scan(
            _param_scan_may_fail, [1.0, 0.0, 3.0],
            n_workers=2, fn_kwargs={'fail_at': 0.0}, verbose=False)
        self.assertEqual(len(results), 3)
        statuses = {r['index']: r['status'] for r in results}
        self.assertEqual(statuses[0], 'success')
        self.assertEqual(statuses[1], 'error')
        self.assertEqual(statuses[2], 'success')
        self.assertIn('ValueError', results[1]['error'])


if __name__ == '__main__':
    unittest.main()
