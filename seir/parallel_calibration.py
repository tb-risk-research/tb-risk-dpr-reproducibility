#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多区域并行校准 (Multi-Region Parallel Calibration)

各省份/区域的 MCMC 链完全独立, 可用 ``concurrent.futures.ProcessPoolExecutor``
按 CPU 核心数线性加速。32 核服务器可同时处理 32 个省份的校准任务, 总时间
从 31×单省时间降到 1×单省时间。

设计要点:
  1. 顶层 worker 函数 (非方法) — ProcessPoolExecutor 在 Windows (spawn) 下
     需要 pickle 序列化, 闭包/lambda/类方法无法 pickle。
  2. 每个进程限制 numpy/scipy 线程数为 1 (OMP/MKL/OpenBLAS), 否则每个进程
     都会尝试使用所有核心导致争抢。使用 ``threadpoolctl.threadpool_limits``
     在 worker 内部限制 (比 env var 更可靠, 因 numpy 可能已 import)。
  3. 每区域独立 try/except, 一个失败不影响其他。
  4. 可选 HME 前置筛选 (region_config['hme_config']) — 减少该区域 MCMC 迭代数。
  5. 可选 device 指定 (CPU/GPU) — 多 GPU 时为每个进程分配独立 GPU。

文献:
  ProcessPoolExecutor: Python 官方文档 ``concurrent.futures``
  threadpoolctl: Mathieu et al. (2021) JOSS 6(64):3242 — 多线程 BLAS 限流
  多区域贝叶斯分层模型: Ragonnet et al. (2021) Clin Infect Dis
"""

import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

LOGGER = logging.getLogger("tb_risk.seir.parallel_calibration")

try:
    import threadpoolctl
    THREADPOOLCTL_AVAILABLE = True
except ImportError:
    THREADPOOLCTL_AVAILABLE = False


__all__ = [
    'ParallelCalibrator',
    'calibrate_region_v4',
    'calibrate_region_v3',
    'calibrate_regions_parallel',
    'calibrate_chain_v4',
    'calibrate_region_chains_parallel',
    'calibrate_region_chains_serial',
    'parallel_param_scan',
]


# =====================================================================
# 顶层 worker 函数 (必须可 pickle, 用于 ProcessPoolExecutor)
# =====================================================================

def _set_thread_limits():
    """限制当前进程的 BLAS/OpenMP 线程数为 1。

    在多进程场景下, 若每个进程都让 MKL/OpenBLAS 用满所有核心, 会发生
    线程争抢反而减速。限制为 1 后, 并行度完全由进程数决定, 接近线性加速。

    使用 ``threadpoolctl.threadpool_limits`` (上下文管理器也可, 但 worker
    整个生命周期都需要限制, 故用一次性调用)。如果 threadpoolctl 不可用,
    回退到设置环境变量 (但 numpy 已 import 时无效, 仅作尽力而为)。

    文献: threadpoolctl JOSS 6(64):3242
    """
    if THREADPOOLCTL_AVAILABLE:
        # 返回一个可调用的控制器, 永久应用限制
        return threadpoolctl.threadpool_limits(limits=1)
    return None


def calibrate_region_v4(region_config):
    """单区域 v4 HMC 校准 (顶层函数, 可 pickle)

    在 ProcessPoolExecutor 的 worker 进程中执行。流程:
      1. 限制 BLAS 线程数为 1
      2. (可选) 运行 HME 前置筛选, 生成 init_params_fn
      3. 实例化 BayesianInferenceV4, 调用 run_mcmc_v4
      4. 返回 dict (含 region_id / posterior_samples / diagnostics / error)

    Args:
        region_config: dict, 必须包含以下键:
            - region_id: 区域标识符 (str 或 int)
            - observed_data: (obs_I, obs_times) 元组
            - initial_state: 270D 初始状态
            - t_span: (t_start, t_end)
            - dt: 积分步长
          可选键:
            - n_warmup, n_samples, n_chains, n_leapfrog, target_accept
            - seed (默认 42)
            - init_params: 15D 约束空间初始参数 (None 用 V4_INIT_PARAMS)
            - waifw, age_progression: 固定结构参数
            - hme_config: dict, 若存在则启用 HME 前置筛选
              {n_design, implausibility_threshold, ...}
            - device: 'cpu' / 'cuda:0' 等 (目前仅 'cpu' 完全支持)

    Returns:
        dict: {
            'region_id': ...,
            'status': 'success' / 'error',
            'posterior_samples': dict (success 时),
            'traces': dict (success 时),
            'diagnostics': dict (success 时),
            'error': str (error 时),
        }
    """
    # 限制 BLAS 线程 (worker 进程整个生命周期)
    _set_thread_limits()

    region_id = region_config.get('region_id', 'unknown')
    try:
        # 延迟导入, 避免主进程 import 时也触发 torch 加载
        from .inference import BayesianInferenceV4
        from .autodiff import PARAM_NAMES, N_INFER_PARAMS

        observed_data = region_config['observed_data']
        initial_state = region_config['initial_state']
        t_span = region_config['t_span']
        dt = region_config['dt']

        n_warmup = region_config.get('n_warmup', 1000)
        n_samples = region_config.get('n_samples', 2000)
        n_chains = region_config.get('n_chains', 4)
        n_leapfrog = region_config.get('n_leapfrog', 20)
        target_accept = region_config.get('target_accept', 0.8)
        seed = region_config.get('seed', 42)
        init_params = region_config.get('init_params', None)
        waifw = region_config.get('waifw', None)
        age_progression = region_config.get('age_progression', None)

        # 创建推断引擎实例
        engine = BayesianInferenceV4(seed=seed)

        # (可选) HME 前置筛选: 生成 init_params (约束空间)
        hme_diag = None
        if 'hme_config' in region_config and region_config['hme_config']:
            from .history_matching import HistoryMatching
            hme_cfg = region_config['hme_config']
            hme = HistoryMatching(
                observed_data, initial_state, t_span, dt,
                fixed_params=region_config.get('fixed_params', None),
                waifw=waifw, age_progression=age_progression,
                n_design=hme_cfg.get('n_design', 150),
                implausibility_threshold=hme_cfg.get(
                    'implausibility_threshold', 3.0),
                random_seed=hme_cfg.get('random_seed', seed))
            hme.run(verbose=False)
            hme_diag = hme.get_diagnostics()
            # 用 HME 缩小后的范围中点作为 init_params (约束空间)
            # 而非默认 V4_INIT_PARAMS, 加速 HMC 收敛
            tightened = hme.get_tightened_bounds(margin=0.0)
            init_params = np.array([(lo + hi) / 2
                                    for lo, hi in tightened])

        # 运行 MCMC
        posterior = engine.run_mcmc_v4(
            observed_data, initial_state, t_span, dt,
            n_warmup=n_warmup, n_samples=n_samples,
            n_chains=n_chains, n_leapfrog=n_leapfrog,
            target_accept=target_accept,
            init_params=init_params,
            waifw=waifw, age_progression=age_progression,
            seed=seed)

        return {
            'region_id': region_id,
            'status': 'success',
            'posterior_samples': posterior,
            'traces': engine.traces,
            'diagnostics': engine.convergence_diagnostics,
            'hme_diagnostics': hme_diag,
        }

    except Exception as e:
        return {
            'region_id': region_id,
            'status': 'error',
            'error': f"{type(e).__name__}: {e}",
        }


def calibrate_region_v3(region_config):
    """单区域 v3 MH 校准 (顶层函数, 可 pickle)

    与 :func:`calibrate_region_v4` 对称, 但调用 ``run_mcmc_v3`` (Metropolis-
    Hastings + Haario 2001 自适应协方差)。v3 在 8-9 维参数空间, 比 v4 便宜,
    适合快速校准或作为 v4 的 warm start。

    Args:
        region_config: 同 :func:`calibrate_region_v4`, 但额外可选:
            - n_iterations, burn_in: 替代 n_warmup/n_samples
            - include_eta_sub, include_eta_sn: 是否推断额外参数
            - use_neg_binom: 是否使用负二项似然
            - proposal_cov: 初始提案协方差

    Returns:
        dict: 同 :func:`calibrate_region_v4`
    """
    _set_thread_limits()

    region_id = region_config.get('region_id', 'unknown')
    try:
        from .inference import BayesianInferenceV3

        observed_data = region_config['observed_data']
        initial_state = region_config['initial_state']
        t_span = region_config['t_span']
        dt = region_config['dt']

        n_iterations = region_config.get('n_iterations', 10000)
        burn_in = region_config.get('burn_in', 2000)
        n_chains = region_config.get('n_chains', 4)
        proposal_cov = region_config.get('proposal_cov', None)
        init_params = region_config.get('init_params', None)
        adapt = region_config.get('adapt', True)
        adapt_interval = region_config.get('adapt_interval', 100)
        adapt_start = region_config.get('adapt_start', 50)
        include_eta_sub = region_config.get('include_eta_sub', False)
        include_eta_sn = region_config.get('include_eta_sn', False)
        use_neg_binom = region_config.get('use_neg_binom', True)
        neg_binom_k = region_config.get('neg_binom_k', None)
        seed = region_config.get('seed', 42)

        engine = BayesianInferenceV3(seed=seed)

        posterior = engine.run_mcmc_v3(
            observed_data, initial_state, t_span, dt,
            n_iterations=n_iterations, burn_in=burn_in,
            n_chains=n_chains, proposal_cov=proposal_cov,
            init_params=init_params, adapt=adapt,
            adapt_interval=adapt_interval, adapt_start=adapt_start,
            include_eta_sub=include_eta_sub,
            include_eta_sn=include_eta_sn,
            use_neg_binom=use_neg_binom,
            neg_binom_k=neg_binom_k)

        return {
            'region_id': region_id,
            'status': 'success',
            'posterior_samples': posterior,
            'traces': engine.traces,
            'diagnostics': engine.convergence_diagnostics,
        }

    except Exception as e:
        return {
            'region_id': region_id,
            'status': 'error',
            'error': f"{type(e).__name__}: {e}",
        }


# =====================================================================
# 链级并行 (Chain-Level Parallelism) — 单区域多链拆分到多进程
#
# 与上面的"区域级并行"互补：
#   - 区域级并行 (calibrate_regions): 多个区域 → 多进程
#   - 链级并行 (calibrate_region_chains_parallel): 单个区域的 n_chains
#     条 MCMC 链 → 多进程（单机多核场景，如单省校准）
#
# 设计要点 (与区域级并行一致):
#   1. 顶层 worker 函数 (非方法) — Windows spawn 下可 pickle。
#   2. 每个 worker 限制 BLAS 线程数为 1, 避免多进程争抢。
#   3. 链种子由 base_seed 确定性派生 (与 BaseBayesianInference 一致),
#      保证并行结果与串行逐链执行逐位一致。
#   4. 一条链失败不影响其他链 (单独 try/except)。
# =====================================================================

def calibrate_chain_v4(chain_config):
    """单链 v4 HMC 校准 (顶层函数, 可 pickle, 供链级多进程并行)

    Args:
        chain_config: dict, 区域配置的超集, 额外含:
            - chain_index: 链索引 (用于确定性派生该链的随机种子)
          区域配置字段同 :func:`calibrate_region_v4` (n_warmup/n_samples/
          n_leapfrog/target_accept/init_params/waifw/age_progression/
          method 等), 内部强制 ``n_chains=1``。

    Returns:
        dict: {
            'chain_index': ...,
            'status': 'success' / 'error',
            'posterior_samples': dict (success 时, 单链约束空间样本),
            'traces': dict (success 时),
            'diagnostics': dict (success 时),
            'error': str (error 时),
        }
    """
    # 限制 BLAS 线程 (worker 进程整个生命周期)
    _set_thread_limits()

    chain_index = chain_config.get('chain_index', 0)
    try:
        from .inference import BayesianInferenceV4

        # 单链配置: 剔除链级专属键, 强制单链
        region = {k: v for k, v in chain_config.items()
                  if k != 'chain_index'}
        region['n_chains'] = 1

        # 确定性派生该链种子 (与 BaseBayesianInference._derive_chain_seed 一致)
        base_seed = region.get('seed', 42)
        from .inference._base import BaseBayesianInference
        chain_seed = (
            base_seed
            + chain_index * BaseBayesianInference._CHAIN_SEED_PRIME
        ) % BaseBayesianInference._CHAIN_SEED_MOD
        region['seed'] = chain_seed

        observed_data = region['observed_data']
        initial_state = region['initial_state']
        t_span = region['t_span']
        dt = region['dt']

        n_warmup = region.get('n_warmup', 1000)
        n_samples = region.get('n_samples', 2000)
        n_leapfrog = region.get('n_leapfrog', 20)
        target_accept = region.get('target_accept', 0.8)
        init_params = region.get('init_params', None)
        waifw = region.get('waifw', None)
        age_progression = region.get('age_progression', None)
        method = region.get('method', 'euler')

        engine = BayesianInferenceV4(seed=chain_seed)
        posterior = engine.run_mcmc_v4(
            observed_data, initial_state, t_span, dt,
            n_warmup=n_warmup, n_samples=n_samples,
            n_chains=1, n_leapfrog=n_leapfrog,
            target_accept=target_accept,
            init_params=init_params,
            waifw=waifw, age_progression=age_progression,
            seed=chain_seed, method=method)

        return {
            'chain_index': chain_index,
            'status': 'success',
            'posterior_samples': posterior,
            'traces': engine.traces,
            'diagnostics': engine.convergence_diagnostics,
        }

    except Exception as e:
        return {
            'chain_index': chain_index,
            'status': 'error',
            'error': f"{type(e).__name__}: {e}",
        }


def _merge_chain_results(chain_results, region_config):
    """合并多条单链结果 → 与单次 ``run_mcmc_v4`` 等价的结构

    后验样本: 逐参数拼接各链样本 (n_chains × n_samples)。
    轨迹: 拼接各链 chains/log_probs/acceptance_rates。
    诊断: 用合并后的轨迹重算 R-hat/ESS (``seir.diagnostics``),
    并补齐 v4_inference 输出的 algorithm/reference/param_mean 字段。

    Args:
        chain_results: list[dict], 每条链的返回 (见 :func:`calibrate_chain_v4`)
        region_config: dict, 原始区域配置 (用于取 n_warmup/n_samples 等)

    Returns:
        dict: {
            'status': 'success' / 'error',
            'posterior_samples': dict,
            'traces': dict,
            'diagnostics': dict,
            'error': str (所有链均失败时),
        }
    """
    from .autodiff import PARAM_NAMES
    from .diagnostics import compute_r_hat, compute_ess

    success = [r for r in chain_results if r.get('status') == 'success']
    failed = [r for r in chain_results if r.get('status') == 'error']

    if not success:
        return {
            'status': 'error',
            'error': failed[0].get('error', 'all chains failed'),
        }

    # 合并约束空间后验样本 (逐参数拼接)
    posterior = {}
    for name in PARAM_NAMES:
        posterior[name] = np.concatenate(
            [r['posterior_samples'][name] for r in success], axis=0)

    # 合并无约束空间轨迹
    chains = []
    log_probs = []
    accept_rates = []
    for r in success:
        tr = r['traces']
        chains.extend(tr['chains'])
        log_probs.extend(tr['log_probs'])
        accept_rates.extend(tr['acceptance_rates'])

    merged = np.concatenate(chains, axis=0)
    diag = {
        'n_chains': len(chain_results),
        'n_warmup': region_config.get('n_warmup', 1000),
        'n_samples': region_config.get('n_samples', 2000),
        'n_leapfrog': region_config.get('n_leapfrog', 20),
        'acceptance_rates': list(accept_rates),
        'mean_acceptance_rate': float(np.mean(accept_rates)),
        'r_hat': compute_r_hat(chains, 0, list(PARAM_NAMES)),
        'ess': compute_ess(merged, list(PARAM_NAMES)),
        'param_names': list(PARAM_NAMES),
        'algorithm': ('HMC + Dual Averaging (Nesterov 2009) '
                      '+ Adaptive Leapfrog'),
        'reference': 'Hoffman & Gelman (2014) JMLR 15(1):1593-1623',
    }
    for name in PARAM_NAMES:
        arr = posterior[name]
        if len(arr) > 0:
            diag[f'{name}_mean'] = float(np.mean(arr))

    return {
        'status': 'success',
        'posterior_samples': posterior,
        'traces': {
            'chains': chains,
            'log_probs': log_probs,
            'acceptance_rates': accept_rates,
        },
        'diagnostics': diag,
    }


def _prepare_chain_configs(region_config, n_chains):
    """把区域配置展开为 n_chains 个链配置 (list[dict], 含 chain_index)"""
    n_chains = int(n_chains or region_config.get('n_chains', 4))
    return [dict(region_config, chain_index=i) for i in range(n_chains)], n_chains


def calibrate_region_chains_serial(region_config, n_chains=None, verbose=True):
    """单区域多链串行校准 (链级, 调试/单核环境回退)

    与 :func:`calibrate_region_chains_parallel` 接口一致, 但在主进程逐链
    执行。用于验证并行结果与串行逐位一致、以及单核环境。
    """
    chain_configs, n_chains = _prepare_chain_configs(region_config, n_chains)
    results = []
    for i, cfg in enumerate(chain_configs):
        if verbose:
            LOGGER.info("[Chain] [%d/%d] 校准链 %d (串行)...",
                        i + 1, n_chains, cfg['chain_index'])
        results.append(calibrate_chain_v4(cfg))
    merged = _merge_chain_results(results, region_config)
    merged['n_chains_requested'] = n_chains
    return merged


def calibrate_region_chains_parallel(region_config, n_workers=None,
                                     n_chains=None, verbose=True):
    """单区域多链并行校准 (链级进程池, 单机多核)

    将单个区域的 ``n_chains`` 条 MCMC 链拆分到 ``n_workers`` 个进程独立
    执行 (每条链 ``run_mcmc_v4(n_chains=1)``), 再合并后验/轨迹并重算
    收敛诊断。适用于单区域多核场景 (如单省校准), 与区域级并行互补。

    注意:
        - 应只在主进程调用, 不要在 :func:`calibrate_region_v4` 的 worker
          进程内部嵌套 (避免 spawn 下进程池嵌套)。
        - Windows 下 ProcessPoolExecutor 使用 spawn, worker 函数
          (:func:`calibrate_chain_v4`) 为顶层函数, 满足 pickle 要求。
        - 每条链从 base_seed 确定性派生种子, 并行与串行逐位一致。

    Args:
        region_config: dict, 区域配置 (见 :func:`calibrate_region_v4`)
        n_workers: 进程数 (None = min(CPU 核数, n_chains))
        n_chains: 链数 (None = region_config['n_chains'], 默认 4)
        verbose: 是否打印进度

    Returns:
        dict: {
            'status': 'success' / 'error',
            'posterior_samples': dict (成功时, 合并 n_chains 条链),
            'traces': dict (成功时),
            'diagnostics': dict (成功时, 基于合并轨迹重算 R-hat/ESS),
            'error': str (所有链均失败时),
        }
    """
    chain_configs, n_chains = _prepare_chain_configs(region_config, n_chains)
    n_workers = n_workers or min(os.cpu_count() or 1, n_chains)

    if verbose:
        LOGGER.info("[Chain] 启动 %d 进程, 校准 %d 条链 (version=v4)",
                    n_workers, n_chains)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(calibrate_chain_v4, cfg): i
            for i, cfg in enumerate(chain_configs)
        }
        results = [None] * n_chains
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:  # worker 崩溃兜底
                results[idx] = {
                    'chain_index': idx,
                    'status': 'error',
                    'error': f"worker crashed: {type(e).__name__}: {e}",
                }

    if verbose:
        n_success = sum(1 for r in results if r.get('status') == 'success')
        LOGGER.info("[Chain] 完成: %d/%d 链成功", n_success, n_chains)

    merged = _merge_chain_results(results, region_config)
    merged['n_chains_requested'] = n_chains
    return merged


# =====================================================================
# ParallelCalibrator 类 (封装 ProcessPoolExecutor 生命周期)
# =====================================================================

class ParallelCalibrator:
    """多区域并行校准编排器

    用 :class:`concurrent.futures.ProcessPoolExecutor` 并行校准多个区域
    (省份/国家)。每个区域作为独立任务分配到不同 CPU 核心, 接近线性加速。

    用法:
        ::

            calibrator = ParallelCalibrator(n_workers=4, version='v4')
            regions = [
                {'region_id': 'beijing', 'observed_data': (...), ...},
                {'region_id': 'shanghai', 'observed_data': (...), ...},
                ...
            ]
            results = calibrator.calibrate_regions(regions)
            # results: {region_id: result_dict}

    注意:
        - Windows 下 ProcessPoolExecutor 使用 spawn, worker 函数必须可 pickle
          (本模块的 :func:`calibrate_region_v4` / :func:`calibrate_region_v3`
          是顶层函数, 满足要求)
        - 每个 worker 进程限制 BLAS 线程数为 1, 避免争抢
        - 一个区域失败不影响其他, 失败区域在结果中 ``status='error'``

    Args:
        n_workers: 进程数 (None = os.cpu_count(), 建议不超过 CPU 核心数)
        version: 'v4' (HMC, 默认) 或 'v3' (MH)
        thread_limit: 是否在 worker 中限制 BLAS 线程 (默认 True)
        use_hme: 是否启用 HME 前置筛选 (默认 False, 由 region_config
            的 hme_config 键单独控制)
    """

    def __init__(self, n_workers=None, version='v4',
                 thread_limit=True, use_hme=False):
        self.n_workers = n_workers or os.cpu_count() or 1
        self.version = version.lower()
        if self.version not in ('v3', 'v4'):
            raise ValueError(f"version 必须为 'v3' 或 'v4', got {version!r}")
        self.thread_limit = bool(thread_limit)
        self.use_hme = bool(use_hme)

        # 选择 worker 函数
        self._worker_fn = (calibrate_region_v4 if self.version == 'v4'
                           else calibrate_region_v3)

    def calibrate_regions(self, regions, verbose=True):
        """并行校准多个区域

        Args:
            regions: list of dict, 每个 dict 是一个区域的配置
                (见 :func:`calibrate_region_v4` 的 region_config)
            verbose: 是否打印进度

        Returns:
            dict: {region_id: result_dict}, result_dict 结构同 worker 返回值
        """
        if not regions:
            return {}

        # 主进程也限制 BLAS 线程 (避免主进程与 worker 争抢)
        if self.thread_limit and THREADPOOLCTL_AVAILABLE:
            with threadpoolctl.threadpool_limits(limits=1):
                return self._run_parallel(regions, verbose)
        else:
            return self._run_parallel(regions, verbose)

    def calibrate_regions_serial(self, regions, verbose=True):
        """串行校准 (调试/单核环境回退)

        与 :meth:`calibrate_regions` 接口一致, 但在主进程中逐区域执行。
        用于:
          - 调试 (错误更易追踪, 不需要重新 spawn 进程)
          - 单核环境
          - 验证并行结果与串行一致
        """
        results = {}
        for i, region_config in enumerate(regions):
            region_id = region_config.get('region_id', f'region_{i}')
            if verbose:
                LOGGER.info("[Parallel] [%d/%d] 校准 %s (串行)...",
                            i + 1, len(regions), region_id)
            result = self._worker_fn(region_config)
            results[region_id] = result
            if verbose:
                status = result.get('status', 'unknown')
                if status == 'success':
                    LOGGER.info("[Parallel]   %s 完成", region_id)
                else:
                    LOGGER.info("[Parallel]   %s 失败: %s",
                                region_id, result.get('error', '?'))
        return results

    def _run_parallel(self, regions, verbose):
        """实际并行执行 (已限制 BLAS 线程)"""
        results = {}
        n_regions = len(regions)

        if verbose:
            LOGGER.info("[Parallel] 启动 %d 进程, 校准 %d 个区域 (version=%s)",
                        self.n_workers, n_regions, self.version)

        # 使用 spawn 兼容的 ProcessPoolExecutor
        # Windows 默认 spawn, Linux/macOS 可用 fork (更快, 但本实现两者兼容)
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # 提交所有任务
            future_to_region = {}
            for i, region_config in enumerate(regions):
                region_id = region_config.get('region_id', f'region_{i}')
                future = executor.submit(self._worker_fn, region_config)
                future_to_region[future] = region_id

            # 按完成顺序收集结果
            for i, future in enumerate(as_completed(future_to_region)):
                region_id = future_to_region[future]
                try:
                    result = future.result()
                except Exception as e:
                    # worker 抛出未捕获异常 (不应发生, worker 内有 try/except)
                    result = {
                        'region_id': region_id,
                        'status': 'error',
                        'error': f"worker crashed: {type(e).__name__}: {e}",
                    }
                results[region_id] = result

                if verbose:
                    status = result.get('status', 'unknown')
                    if status == 'success':
                        LOGGER.info("[Parallel] [%d/%d] %s 完成",
                                    i + 1, n_regions, region_id)
                    else:
                        LOGGER.info("[Parallel] [%d/%d] %s 失败: %s",
                                    i + 1, n_regions, region_id,
                                    result.get('error', '?'))

        if verbose:
            n_success = sum(1 for r in results.values()
                            if r.get('status') == 'success')
            n_error = n_regions - n_success
            LOGGER.info("[Parallel] 全部完成: %d/%d 成功, %d 失败",
                        n_success, n_regions, n_error)

        return results


def calibrate_regions_parallel(regions, n_workers=None, version='v4',
                                verbose=True):
    """便捷函数: 多区域并行校准 (一次性调用)

    等价于::

        calibrator = ParallelCalibrator(n_workers=n_workers, version=version)
        return calibrator.calibrate_regions(regions, verbose=verbose)

    Args:
        regions: list of dict (见 :func:`calibrate_region_v4`)
        n_workers: 进程数 (None = os.cpu_count())
        version: 'v3' 或 'v4'
        verbose: 是否打印进度

    Returns:
        dict: {region_id: result_dict}
    """
    calibrator = ParallelCalibrator(
        n_workers=n_workers, version=version, thread_limit=True)
    return calibrator.calibrate_regions(regions, verbose=verbose)


# =====================================================================
# 通用参数扫描并行 (Generic Parameter Scan)
#
# 与区域级/链级并行不同, 这是完全通用的并行执行器: 用户提供任意顶层
# 可 pickle 的评估函数 eval_fn 和一组参数点, 并行地对每个点调用 eval_fn。
# 适用于 MCMC 前的参数扫描 / 网格搜索 / 敏感性分析等"逐点独立评估"场景
# (单机计算优化: SEIR 参数扫描从逐条 Python 循环升级为多进程并行)。
#
# 设计要点:
#   1. eval_fn 与公共关键字参数 fn_kwargs 通过进程池 initializer 注入
#      (每个 worker 只 pickle 一次), 避免逐任务重复 pickle 大数据
#      (如 observed_data / initial_state / WAIFW 矩阵)。
#   2. 每个 worker 限制 BLAS 线程数为 1 (见 _set_thread_limits),
#      避免多进程争抢, 并行度完全由进程数决定。
#   3. 单个参数点失败不影响其他点 (返回 status='error')。
# =====================================================================

_WORKER_FN = None
_WORKER_FN_KWARGS = None


def _init_scan_worker(fn, fn_kwargs):
    """进程池初始化函数: 注入 eval_fn 与公共 kwargs (每个 worker 执行一次)

    Args:
        fn: 顶层可 pickle 的评估函数 ``f(params, **fn_kwargs)``
        fn_kwargs: dict, 传给 fn 的公共关键字参数
    """
    global _WORKER_FN, _WORKER_FN_KWARGS
    # 限制 BLAS 线程 (worker 进程整个生命周期)
    _set_thread_limits()
    _WORKER_FN = fn
    _WORKER_FN_KWARGS = fn_kwargs


def _param_scan_worker(params, index):
    """单参数点评估 (顶层函数, 可 pickle)

    Args:
        params: 参数点 (任意可 pickle 对象, 由 parallel_param_scan 传入)
        index: 参数点索引

    Returns:
        dict: {
            'index': index,
            'status': 'success' / 'error',
            'result': ... (success 时),
            'error': str (error 时),
        }
    """
    try:
        result = _WORKER_FN(params, **_WORKER_FN_KWARGS)
        return {'index': index, 'status': 'success', 'result': result}
    except Exception as e:
        return {'index': index, 'status': 'error',
                'error': f"{type(e).__name__}: {e}"}


def parallel_param_scan(eval_fn, param_list, n_workers=None,
                        fn_kwargs=None, verbose=True):
    """通用参数扫描 (多进程并行)

    将 ``param_list`` 中的参数点分派到进程池, 并行调用 ``eval_fn``。
    适用于参数扫描 / 网格搜索 / 敏感性分析等"逐点独立评估"场景, 将
    原来的逐条 Python 循环升级为多核并行, 接近线性加速。

    约束:
        - ``eval_fn`` 必须是模块顶层可 pickle 的函数 (Windows spawn 下
          无法 pickle 闭包 / lambda / 类方法)。
        - 除参数点本身外的只读数据 (观测数据、初始状态、结构矩阵等)
          通过 ``fn_kwargs`` 传入, 每个 worker 只 pickle 一次。
        - 每个 worker 自动限制 BLAS 线程数为 1, 避免进程间争抢。
        - 单个参数点失败不影响其他点。

    Args:
        eval_fn: 可调用对象 ``f(params, **fn_kwargs) -> result``,
            返回结果必须可 pickle。
        param_list: list, 每个元素为一个参数点 (任意可 pickle 对象)。
        n_workers: 进程数 (None = min(CPU 核数, 点数))。
        fn_kwargs: dict | None, 传给 eval_fn 的公共关键字参数。
        verbose: 是否打印进度日志。

    Returns:
        list[dict]: 与 ``param_list`` 顺序一致的逐点结果:
            {'index': i, 'status': 'success' | 'error',
             'result': ... (success 时), 'error': str (error 时)}
    """
    n_tasks = len(param_list)
    if n_tasks == 0:
        return []
    n_workers = n_workers or min(os.cpu_count() or 1, n_tasks)
    fn_kwargs = dict(fn_kwargs or {})

    if verbose:
        LOGGER.info("[Scan] 启动 %d 进程, 扫描 %d 个参数点...",
                    n_workers, n_tasks)

    with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_scan_worker,
            initargs=(eval_fn, fn_kwargs)) as executor:
        future_to_idx = {
            executor.submit(_param_scan_worker, params, i): i
            for i, params in enumerate(param_list)
        }
        results = [None] * n_tasks
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:  # worker 崩溃兜底
                results[idx] = {'index': idx, 'status': 'error',
                                'error': f"worker crashed: "
                                         f"{type(e).__name__}: {e}"}

    if verbose:
        n_success = sum(1 for r in results if r.get('status') == 'success')
        LOGGER.info("[Scan] 完成: %d/%d 点成功", n_success, n_tasks)

    return results
