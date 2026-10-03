#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史匹配与仿真器前置筛选 (History Matching & Emulator, HME)

在 MCMC 之前用 GP 仿真器排除不可信参数空间, 减少 MCMC 迭代数 5-10x。

流程:
  1. LHS 采样 100-200 个参数点 (约束空间)
  2. 对每个参数点运行 v4 ODE, 提取观测时间点的活动性 TB 计数
  3. 训练 GP 仿真器 (每个观测时间点一个独立 GP)
  4. 计算不可信度指标 I(x) = |f(x) - z| / sqrt(Var[emulator] + Var[obs])
  5. 排除 I(x) > threshold 的参数点 (通常 threshold=3)
  6. 用非不可信参数点初始化 HMC (缩小初始范围)

文献:
  Kennedy & O'Hagan (2001) JRSS-B 63(3):425-464 — Bayesian calibration
  Vernon et al. (2010) SIAM/ASA J UQ 2(1):1-37 — History matching
  Craig et al. (1997) in Bayesian Statistics 5 — Bayesian history matching
"""

import logging
import warnings

import numpy as np
from scipy.stats import qmc

LOGGER = logging.getLogger("tb_risk.seir.hme")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF, WhiteKernel, ConstantKernel as C,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .autodiff import (
    PARAM_BOUNDS, PARAM_NAMES, N_INFER_PARAMS,
    torch_v4_integrate,
)
from .stochastic import (
    N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3,
    IDX_ISUB_V3, IDX_ISP_V3, IDX_ISN_V3,
)

__all__ = ['HistoryMatching', 'run_hme_then_mcmc']


def _constrained_to_unconstrained(x_constrained):
    """约束空间 → 无约束空间 (numpy, logit 变换)

    x_constrained: (..., 15) 约束空间参数
    返回: (..., 15) 无约束空间参数
    """
    y = np.zeros_like(x_constrained, dtype=np.float64)
    for i, (lo, hi) in enumerate(PARAM_BOUNDS):
        u = np.clip(
            (x_constrained[..., i] - lo) / (hi - lo),
            1e-10, 1 - 1e-10)
        y[..., i] = np.log(u / (1.0 - u))
    return y


def _unconstrained_to_constrained(y_unconstrained):
    """无约束空间 → 约束空间 (numpy, sigmoid 变换)

    y_unconstrained: (..., 15) 无约束空间参数
    返回: (..., 15) 约束空间参数
    """
    x = np.zeros_like(y_unconstrained, dtype=np.float64)
    for i, (lo, hi) in enumerate(PARAM_BOUNDS):
        s = 1.0 / (1.0 + np.exp(-y_unconstrained[..., i]))
        x[..., i] = lo + (hi - lo) * s
    return x


def _forward_chunk_worker(params_constrained, obs_times, t_span, dt,
                          waifw, age_progression, fixed_params,
                          initial_state):
    """单进程内对一批参数运行 v4 ODE 并提取观测点活动性 TB (顶层函数)

    供 :meth:`HistoryMatching._run_forward_model` 的多进程并行路径
    (n_jobs > 1) 使用: 每个 worker 处理一个参数分块, 各分块由
    ``parallel_param_scan`` 分派到不同进程, 从而在多核上并行加速
    前向模型评估。worker 内限制 BLAS 线程为 1, 避免进程间争抢。

    Args:
        params_constrained: (B, 15) 约束空间参数 (单个分块)
        obs_times: (n_obs,) 观测时间点
        t_span: (t_start, t_end) 时间范围
        dt: 积分步长
        waifw: 5×5 WAIFW 矩阵 (None 用默认)
        age_progression: 5 维年龄进展率 (None 用默认)
        fixed_params: 固定参数字典 (与 torch_v4_integrate 一致)
        initial_state: (270,) 初始状态

    Returns:
        outputs: (B, n_obs) 各观测时间点的活动性 TB 计数;
            若 ODE 积分失败返回 (B, n_obs) 全 NaN (标记为不可信)。
    """
    from .parallel_calibration import _set_thread_limits
    _set_thread_limits()

    params_constrained = np.asarray(params_constrained, dtype=np.float64)
    B = params_constrained.shape[0]
    n_obs = len(obs_times)
    n_steps = int((t_span[1] - t_span[0]) / dt)

    # 转换为无约束空间 (torch_v4_integrate 期望无约束参数)
    params_unconstrained = _constrained_to_unconstrained(params_constrained)

    waifw_t = (torch.tensor(waifw, dtype=torch.float64)
               if waifw is not None else None)
    age_prog_t = (torch.tensor(age_progression, dtype=torch.float64)
                  if age_progression is not None else None)
    initial_state_t = torch.tensor(initial_state, dtype=torch.float64)

    # 时间轴 (numpy, 用于插值)
    sim_times_np = np.linspace(t_span[0], t_span[1], n_steps + 1)

    try:
        y_params = torch.tensor(params_unconstrained, dtype=torch.float64)
        with torch.no_grad():
            traj = torch_v4_integrate(
                y_params, initial_state_t, n_steps, dt,
                waifw_t, age_prog_t, fixed_params,
                check_conservation=True)
    except Exception as e:
        warnings.warn(
            f"v4 ODE 积分失败: {e}, 这些参数点将被标记为不可信")
        return np.full((B, n_obs), np.nan, dtype=np.float64)

    # 提取活动性 TB: I_sub + I_sp + I_sn (sum over age, HIV, DR)
    traj_5d = traj.view(
        B, n_steps + 1, N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
    sim_I = (traj_5d[:, :, :, :, :, IDX_ISUB_V3].sum(dim=(2, 3, 4)) +
             traj_5d[:, :, :, :, :, IDX_ISP_V3].sum(dim=(2, 3, 4)) +
             traj_5d[:, :, :, :, :, IDX_ISN_V3].sum(dim=(2, 3, 4)))
    # sim_I: (B, n_steps+1)
    sim_I_np = sim_I.detach().cpu().numpy()

    # 插值到观测时间点
    outputs = np.zeros((B, n_obs), dtype=np.float64)
    for j, t_obs in enumerate(obs_times):
        idx = np.searchsorted(sim_times_np, t_obs)
        idx = int(np.clip(idx, 1, len(sim_times_np) - 1))
        t0 = sim_times_np[idx - 1]
        t1 = sim_times_np[idx]
        alpha = (t_obs - t0) / (t1 - t0 + 1e-10)
        v0 = sim_I_np[:, idx - 1]
        v1 = sim_I_np[:, idx]
        outputs[:, j] = v0 + alpha * (v1 - v0)

    return outputs


class HistoryMatching:
    """历史匹配与仿真器前置筛选 (HME)

    在 MCMC 之前用 GP 仿真器排除不可信参数空间。

    用法:
        hme = HistoryMatching(observed_data, initial_state, t_span, dt)
        hme.run()
        init_fn = hme.get_hmc_init_fn()
        result = hmc.sample(init_params_fn=init_fn, batch_chains=True)

    文献:
        Kennedy & O'Hagan (2001) JRSS-B 63(3):425-464
        Vernon et al. (2010) SIAM/ASA J UQ 2(1):1-37
    """

    def __init__(self, observed_data, initial_state, t_span, dt,
                 fixed_params=None, waifw=None, age_progression=None,
                 n_design=150, implausibility_threshold=3.0,
                 obs_variance=None, random_seed=42, n_jobs=1):
        """初始化 HME。

        Args:
            observed_data: (obs_I, obs_times) 元组, 与 torch_log_likelihood_v4 一致
            initial_state: (270,) 初始状态
            t_span: (t_start, t_end) 时间范围
            dt: 积分步长
            fixed_params: 固定参数字典 (与 torch_v4_integrate 一致)
            waifw: 5×5 WAIFW 矩阵 (None 用默认)
            age_progression: 5维年龄进展率 (None 用默认)
            n_design: LHS 采样点数 (100-200 推荐)
            implausibility_threshold: 不可信度阈值 (通常 3.0, 3-sigma 规则)
            obs_variance: 观测方差标量 (None 则用 (0.2*std(obs_I))**2,
                与 torch_log_likelihood_v4 的 obs_std 一致)
            random_seed: 随机种子
            n_jobs: 前向模型并行进程数 (1 = 串行; >1 时按设计点分块用
                ProcessPoolExecutor 多进程并行评估 v4 ODE, 单机多核优化)。
                可在 :meth:`run` 中覆盖。
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "HME 需要 scikit-learn。请安装: pip install scikit-learn")
        if not TORCH_AVAILABLE:
            raise ImportError("HME 需要 PyTorch。请安装: pip install torch")

        self.obs_I = np.asarray(observed_data[0], dtype=np.float64)
        self.obs_times = np.asarray(observed_data[1], dtype=np.float64)
        self.initial_state = np.asarray(initial_state, dtype=np.float64)
        self.t_span = (float(t_span[0]), float(t_span[1]))
        self.dt = float(dt)
        self.fixed_params = fixed_params
        self.waifw = waifw
        self.age_progression = age_progression
        self.n_design = int(n_design)
        self.implausibility_threshold = float(implausibility_threshold)
        self.random_seed = int(random_seed)
        self.n_jobs = max(1, int(n_jobs))

        # 观测方差 (与 torch_log_likelihood_v4 的 obs_std 一致)
        if obs_variance is not None:
            self.obs_var = float(obs_variance)
        else:
            obs_std = max(0.2 * np.std(self.obs_I), 0.5) if len(self.obs_I) > 0 else 0.5
            self.obs_var = obs_std ** 2

        # 结果存储
        self.design_params = None          # (n_design, 15) 约束空间
        self.design_outputs = None         # (n_design, n_obs) 前向模型输出
        self.emulators = None              # list of n_obs 个 GaussianProcessRegressor
        self.implausibility = None         # (n_design,) 最大不可信度
        self.non_implausible_mask = None   # (n_design,) bool
        self.non_implausible_params = None  # (n_pass, 15) 约束空间

    def _generate_lhs_design(self):
        """LHS 采样生成参数设计点 (约束空间)

        使用 scipy.stats.qmc.LatinHypercube 在 [0,1]^15 中采样,
        然后缩放到 PARAM_BOUNDS。LHS 保证每个参数维度的分层覆盖。
        """
        sampler = qmc.LatinHypercube(d=N_INFER_PARAMS, seed=self.random_seed)
        unit_samples = sampler.random(n=self.n_design)  # (n_design, 15) in [0, 1]
        bounds = np.array(PARAM_BOUNDS)
        lo = bounds[:, 0]
        hi = bounds[:, 1]
        self.design_params = lo + (hi - lo) * unit_samples
        return self.design_params

    def _run_forward_model(self, params_constrained, batch_size=20, n_jobs=None):
        """对一批参数运行 v4 ODE, 提取观测时间点的活动性 TB 计数

        Args:
            params_constrained: (B, 15) 约束空间参数
            batch_size: 批量大小 (控制内存, ODE 轨迹 (B, n_steps+1, 270) 较大;
                仅在串行路径生效)
            n_jobs: 并行进程数 (None = 构造时的 self.n_jobs; 1 走串行路径;
                >1 时按设计点分块, 用 ``parallel_param_scan`` 多进程并行评估)
        Returns:
            outputs: (B, n_obs) 各观测时间点的活动性 TB 计数
        """
        B = params_constrained.shape[0]
        n_obs = len(self.obs_times)

        # 多进程并行路径 (n_jobs > 1): 按点分块, 每块一个 worker
        if n_jobs is None:
            n_jobs = self.n_jobs
        if int(n_jobs) > 1 and B > 1:
            return self._run_forward_model_parallel(
                params_constrained, n_jobs=int(n_jobs))

        n_steps = int((self.t_span[1] - self.t_span[0]) / self.dt)

        # 转换为无约束空间 (torch_v4_integrate 期望无约束参数)
        params_unconstrained = _constrained_to_unconstrained(params_constrained)

        all_outputs = np.zeros((B, n_obs), dtype=np.float64)

        # 准备固定张量
        waifw_t = None
        age_prog_t = None
        if self.waifw is not None:
            waifw_t = torch.tensor(self.waifw, dtype=torch.float64)
        if self.age_progression is not None:
            age_prog_t = torch.tensor(self.age_progression, dtype=torch.float64)
        initial_state_t = torch.tensor(self.initial_state, dtype=torch.float64)

        # 时间轴 (numpy, 用于插值)
        sim_times_np = np.linspace(
            self.t_span[0], self.t_span[1], n_steps + 1)

        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            batch = params_unconstrained[start:end]
            y_params = torch.tensor(batch, dtype=torch.float64)

            try:
                with torch.no_grad():
                    traj = torch_v4_integrate(
                        y_params, initial_state_t, n_steps, self.dt,
                        waifw_t, age_prog_t, self.fixed_params,
                        check_conservation=True)
            except Exception as e:
                warnings.warn(
                    f"v4 ODE 积分失败 (batch {start}:{end}): {e}, "
                    f"这些参数点将被标记为不可信")
                all_outputs[start:end, :] = np.nan
                continue

            # 提取活动性 TB: I_sub + I_sp + I_sn (sum over age, HIV, DR)
            traj_5d = traj.view(
                traj.shape[0], n_steps + 1,
                N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
            sim_I = (traj_5d[:, :, :, :, :, IDX_ISUB_V3].sum(dim=(2, 3, 4)) +
                     traj_5d[:, :, :, :, :, IDX_ISP_V3].sum(dim=(2, 3, 4)) +
                     traj_5d[:, :, :, :, :, IDX_ISN_V3].sum(dim=(2, 3, 4)))
            # sim_I: (batch, n_steps+1)
            sim_I_np = sim_I.detach().cpu().numpy()

            # 插值到观测时间点
            for j, t_obs in enumerate(self.obs_times):
                idx = np.searchsorted(sim_times_np, t_obs)
                idx = int(np.clip(idx, 1, len(sim_times_np) - 1))
                t0 = sim_times_np[idx - 1]
                t1 = sim_times_np[idx]
                alpha = (t_obs - t0) / (t1 - t0 + 1e-10)
                v0 = sim_I_np[:, idx - 1]
                v1 = sim_I_np[:, idx]
                all_outputs[start:end, j] = v0 + alpha * (v1 - v0)

        return all_outputs

    def _run_forward_model_parallel(self, params_constrained, n_jobs):
        """多进程并行前向模型: 按设计点分块, 每块一个 worker 进程

        将参数点均分为 ``n_jobs`` 个分块, 用 ``parallel_param_scan`` 把各
        分块分派到不同进程并行评估 v4 ODE (每个 worker 限制 BLAS 线程为 1)。
        失败分块整体标记为 NaN (不可信), 与串行路径行为一致。

        Args:
            params_constrained: (B, 15) 约束空间参数
            n_jobs: 并行进程数 (已钳制到 [2, B])

        Returns:
            outputs: (B, n_obs) 各观测时间点的活动性 TB 计数
        """
        from .parallel_calibration import parallel_param_scan

        n_total = params_constrained.shape[0]
        n_obs = len(self.obs_times)
        n_jobs = min(n_jobs, n_total)

        # 均分分块 (np.linspace 边界保证块大小尽量均衡)
        bounds = np.linspace(0, n_total, n_jobs + 1).astype(int)
        chunks = [params_constrained[bounds[i]:bounds[i + 1]]
                  for i in range(n_jobs)]

        fn_kwargs = {
            'obs_times': self.obs_times,
            't_span': self.t_span,
            'dt': self.dt,
            'waifw': self.waifw,
            'age_progression': self.age_progression,
            'fixed_params': self.fixed_params,
            'initial_state': self.initial_state,
        }

        results = parallel_param_scan(
            _forward_chunk_worker, chunks, n_workers=n_jobs,
            fn_kwargs=fn_kwargs, verbose=False)

        outputs = np.zeros((n_total, n_obs), dtype=np.float64)
        offset = 0
        for r in results:
            chunk_len = len(chunks[r['index']])
            if r['status'] == 'success':
                outputs[offset:offset + chunk_len] = r['result']
            else:
                warnings.warn(
                    f"v4 ODE 积分失败 (chunk {r['index']}): "
                    f"{r.get('error', '?')}, 这些参数点将被标记为不可信")
                outputs[offset:offset + chunk_len] = np.nan
            offset += chunk_len
        return outputs

    def _train_emulators(self):
        """为每个观测时间点训练独立 GP 仿真器

        每个观测时间点对应一个 GP, 输入为 15D 约束空间参数, 输出为该
        时间点的活动性 TB 计数。独立 GP 允许各时间点有不同的长度尺度
        和噪声水平。RBF 核 + WhiteKernel 提供平滑拟合和噪声估计。
        """
        n_obs = len(self.obs_times)
        self.emulators = []

        for j in range(n_obs):
            # 过滤 NaN (前向模型失败的点)
            valid = ~np.isnan(self.design_outputs[:, j])
            if valid.sum() < 10:
                warnings.warn(
                    f"观测时间点 {j} (t={self.obs_times[j]}) 仅有 "
                    f"{valid.sum()} 个有效设计点, GP 拟合可能不可靠")
                if valid.sum() < 3:
                    self.emulators.append(None)
                    continue

            X = self.design_params[valid]
            y = self.design_outputs[valid, j]

            # 核函数: 常数 × RBF (各维独立长度尺度) + 白噪声
            kernel = C(1.0, (1e-3, 1e3)) * \
                RBF(length_scale=np.ones(N_INFER_PARAMS),
                    length_scale_bounds=(1e-2, 1e2)) + \
                WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))

            gp = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-10,
                n_restarts_optimizer=2,
                random_state=self.random_seed + j,
                normalize_y=True)
            gp.fit(X, y)
            self.emulators.append(gp)

    def _compute_implausibility(self, params):
        """计算不可信度指标 I(x) = |f(x) - z| / sqrt(Var[emu] + Var[obs])

        对每个观测时间点计算不可信度, 取最大值 (最保守, 任一时间点
        不匹配即排除)。Var[emu] 来自 GP 预测方差 (含 WhiteKernel 噪声),
        Var[obs] 为观测方差。

        Args:
            params: (B, 15) 约束空间参数
        Returns:
            implausibility: (B,) 最大不可信度 (跨观测时间点)
        """
        B = params.shape[0]
        n_obs = len(self.obs_times)
        all_implausibility = np.full((B, n_obs), np.inf)

        for j, gp in enumerate(self.emulators):
            if gp is None:
                continue
            mean, std = gp.predict(params, return_std=True)
            variance = std ** 2
            all_implausibility[:, j] = np.abs(mean - self.obs_I[j]) / \
                np.sqrt(variance + self.obs_var)

        return all_implausibility.max(axis=1)

    def run(self, batch_size=20, verbose=True, n_jobs=None):
        """运行完整 HME 流程

        1. LHS 采样
        2. 前向模型 (v4 ODE)
        3. 训练 GP 仿真器
        4. 计算不可信度
        5. 筛选非不可信参数

        Args:
            batch_size: 前向模型批量大小
            verbose: 是否打印进度
            n_jobs: 前向模型并行进程数 (None = 构造时的 self.n_jobs;
                1 = 串行, >1 = 多进程并行, 单机多核优化)
        Returns:
            non_implausible_params: (n_pass, 15) 约束空间非不可信参数
        """
        if verbose:
            LOGGER.info("[HME] 步骤 1/5: LHS 采样...")
        self._generate_lhs_design()

        if verbose:
            LOGGER.info("[HME] 步骤 2/5: 运行 v4 ODE (%d 点, batch=%d, "
                        "n_jobs=%s)...",
                        self.n_design, batch_size,
                        (self.n_jobs if n_jobs is None else n_jobs))
        self.design_outputs = self._run_forward_model(
            self.design_params, batch_size=batch_size, n_jobs=n_jobs)

        if verbose:
            n_nan = np.isnan(self.design_outputs).any(axis=1).sum()
            LOGGER.info("[HME]   ODE 完成, %d 点失败 (NaN)", n_nan)
            LOGGER.info("[HME] 步骤 3/5: 训练 GP 仿真器 (%d 个独立 GP)...",
                       len(self.obs_times))
        self._train_emulators()

        if verbose:
            LOGGER.info("[HME] 步骤 4/5: 计算不可信度...")
        self.implausibility = self._compute_implausibility(self.design_params)

        if verbose:
            LOGGER.info("[HME] 步骤 5/5: 筛选非不可信参数...")
        self.non_implausible_mask = self.implausibility <= self.implausibility_threshold
        self.non_implausible_params = self.design_params[self.non_implausible_mask]

        n_pass = self.non_implausible_params.shape[0]
        n_total = self.n_design
        if verbose:
            LOGGER.info("[HME] 完成: %d/%d (%.1f%%) 参数点通过筛选",
                       n_pass, n_total, 100 * n_pass / n_total)
            if n_pass > 0:
                LOGGER.info("[HME] 参数范围缩小:")
                for i, name in enumerate(PARAM_NAMES):
                    lo_orig, hi_orig = PARAM_BOUNDS[i]
                    lo_new = self.non_implausible_params[:, i].min()
                    hi_new = self.non_implausible_params[:, i].max()
                    LOGGER.info("  %s: [%.6f, %.6f] → [%.6f, %.6f]",
                               name, lo_orig, hi_orig, lo_new, hi_new)

        if n_pass == 0:
            warnings.warn(
                "所有参数点都被排除 (n_pass=0)。建议增大 n_design 或 "
                "放宽 implausibility_threshold。将返回全部设计点作为后备。")
            self.non_implausible_params = self.design_params.copy()

        return self.non_implausible_params

    def get_hmc_init_fn(self, jitter_std=0.1):
        """生成 HMC 初始化函数 (无约束空间)

        从非不可信参数点中随机采样作为各链的起始点, 加高斯扰动。
        所有参数转换到无约束空间 (HMCSampler 期望无约束空间)。

        Args:
            jitter_std: 扰动标准差 (无约束空间, 默认 0.1)
        Returns:
            init_params_fn: (chain_idx) -> np.ndarray (15,) 无约束空间
        """
        if self.non_implausible_params is None:
            raise RuntimeError("HME 尚未运行, 请先调用 run()")

        # 转换为无约束空间
        params_unc = _constrained_to_unconstrained(
            self.non_implausible_params)

        rng = np.random.RandomState(self.random_seed + 1000)
        n_pass = params_unc.shape[0]

        def init_params_fn(chain_idx):
            idx = rng.randint(n_pass)
            return params_unc[idx] + rng.normal(
                0, jitter_std, N_INFER_PARAMS)

        return init_params_fn

    def get_tightened_bounds(self, margin=0.1):
        """获取缩小的参数边界 (约束空间)

        基于非不可信参数点的 min/max, 向外扩展 margin 比例,
        不超过原始 PARAM_BOUNDS。

        Args:
            margin: 边界扩展比例 (0.1 = 向外扩展 10%)
        Returns:
            bounds: list of (lo, hi) 缩小后的边界
        """
        if self.non_implausible_params is None:
            raise RuntimeError("HME 尚未运行, 请先调用 run()")

        bounds = []
        for i in range(N_INFER_PARAMS):
            lo_orig, hi_orig = PARAM_BOUNDS[i]
            lo_new = float(self.non_implausible_params[:, i].min())
            hi_new = float(self.non_implausible_params[:, i].max())
            span = hi_new - lo_new
            if span > 0:
                lo_new = max(lo_orig, lo_new - margin * span)
                hi_new = min(hi_orig, hi_new + margin * span)
            bounds.append((lo_new, hi_new))
        return bounds

    def get_diagnostics(self):
        """返回诊断信息"""
        if self.non_implausible_params is None:
            return {'status': 'not_run'}

        n_pass = self.non_implausible_params.shape[0]
        n_total = self.n_design

        # 原始空间 vs 缩小空间的体积比 (衡量空间缩减程度)
        orig_volume = 1.0
        new_volume = 1.0
        tightened_bounds = self.get_tightened_bounds()
        for i in range(N_INFER_PARAMS):
            lo_orig, hi_orig = PARAM_BOUNDS[i]
            lo_new, hi_new = tightened_bounds[i]
            orig_volume *= (hi_orig - lo_orig)
            new_volume *= (hi_new - lo_new)

        return {
            'status': 'completed',
            'n_design': n_total,
            'n_non_implausible': n_pass,
            'pass_rate': n_pass / n_total,
            'space_reduction': 1.0 - new_volume / orig_volume,
            'implausibility_threshold': self.implausibility_threshold,
            'obs_variance': self.obs_var,
            'mean_implausibility': float(np.mean(self.implausibility)),
            'max_implausibility': float(np.max(self.implausibility)),
            'tightened_bounds': tightened_bounds,
            'original_bounds': list(PARAM_BOUNDS),
            'param_names': list(PARAM_NAMES),
        }


def run_hme_then_mcmc(observed_data, initial_state, t_span, dt,
                      n_design=150, implausibility_threshold=3.0,
                      n_warmup=500, n_samples=1000, n_chains=4,
                      n_leapfrog=20, target_accept=0.8,
                      fixed_params=None, waifw=None, age_progression=None,
                      batch_chains=True, precision='float64',
                      random_seed=42, verbose=True, n_jobs=1):
    """HME 前置筛选 + HMC 采样的便捷封装

    先运行 HME 排除不可信参数空间, 再用非不可信参数点初始化 HMC。
    MCMC 迭代数可从 5000 降到 1000-2000 (因采样集中在高后验密度区域)。

    Args:
        (同 HistoryMatching + HMCSampler 的主要参数, 另加 n_jobs:
         HME 前向模型并行进程数, 1 = 串行, >1 = 多进程并行)
    Returns:
        (hmc_result, hme_diagnostics) 元组
    """
    from .hmc_sampler import HMCSampler
    from .autodiff import (
        torch_log_posterior_v4, compute_grad_log_posterior,
    )

    # Step 1: HME 前置筛选
    hme = HistoryMatching(
        observed_data, initial_state, t_span, dt,
        fixed_params=fixed_params, waifw=waifw,
        age_progression=age_progression,
        n_design=n_design,
        implausibility_threshold=implausibility_threshold,
        random_seed=random_seed,
        n_jobs=n_jobs)
    hme.run(verbose=verbose)

    # Step 2: 用 HME 结果初始化 HMC
    if verbose:
        LOGGER.info("[HME+MCMC] 启动 HMC 采样...")
    torch.manual_seed(random_seed)
    torch.set_grad_enabled(True)

    grad_fn = lambda y: compute_grad_log_posterior(
        y, observed_data, initial_state, t_span, dt,
        waifw_t=torch.tensor(waifw, dtype=torch.float64) if waifw is not None else None,
        age_prog_t=torch.tensor(age_progression, dtype=torch.float64) if age_progression is not None else None,
        fixed_params=fixed_params)

    init_fn = hme.get_hmc_init_fn()

    hmc = HMCSampler(
        torch_log_posterior_v4, grad_fn, N_INFER_PARAMS,
        target_accept=target_accept,
        n_leapfrog=n_leapfrog,
        n_warmup=n_warmup, n_samples=n_samples,
        n_chains=n_chains,
        seed=random_seed,
        precision=precision)

    result = hmc.sample(init_params_fn=init_fn, batch_chains=batch_chains)
    return result, hme.get_diagnostics()
