"""v4 推断引擎：年龄×HIV×耐药分层 270D 模型 HMC/NUTS 采样。

包含方法：
  log_prior_v4 / log_likelihood_v4 / log_posterior_v4
  run_mcmc_v4
"""

import hashlib

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..stochastic import (
    DEFAULT_OMEGA_REG_M, DEFAULT_OMEGA_REG_SUB, DEFAULT_BETA_EXO,
    DEFAULT_ETA_SN,
    DEFAULT_RR_HIV, DEFAULT_ART_REDUCTION,
    DEFAULT_HIV_INFECTION_RATE, DEFAULT_ART_INITIATION_RATE,
    DEFAULT_DR_FITNESS_COST, DEFAULT_P_ACQ,
    DEFAULT_AGE_PROGRESSION, DEFAULT_WAIFW,
    N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3,
    IDX_ISUB_V3, IDX_ISP_V3, IDX_ISN_V3,
)
from ..hmc_sampler import HMCSampler
from .v3_inference import BayesianInferenceV3


class BayesianInferenceV4(BayesianInferenceV3):
    """v4 年龄×HIV×耐药分层 270D 模型贝叶斯推断（集成 HMC）。

    15 维标量参数空间（V4_PARAM_NAMES），WAIFW 与 age_progression
    作为固定结构参数。利用 PyTorch 自动微分在无约束空间运行 HMC。
    """

    # ==================== v4.0 年龄×HIV×耐药分层 ====================

    # v4 推断参数名（标量参数；WAIFW 与 age_progression 视为固定结构参数）
    V4_PARAM_NAMES = [
        'beta', 'rho_fast', 'rho_react', 'sigma_clear', 'gamma',
        'omega_reg_m', 'omega_reg_sub', 'beta_exo', 'eta_sn',
        'rr_hiv', 'art_reduction', 'hiv_infection_rate',
        'art_initiation_rate', 'dr_fitness_cost', 'p_acq',
    ]
    V4_N_PARAMS = 15

    # 初始值（按 V4_PARAM_NAMES 顺序）
    V4_INIT_PARAMS = np.array([
        0.2,                      # beta
        0.1 / 365.0,              # rho_fast
        0.004 / 365.0,            # rho_react
        2.0 / 365.0,              # sigma_clear
        0.05,                     # gamma
        DEFAULT_OMEGA_REG_M,      # omega_reg_m
        DEFAULT_OMEGA_REG_SUB,    # omega_reg_sub
        DEFAULT_BETA_EXO,         # beta_exo
        DEFAULT_ETA_SN,           # eta_sn
        DEFAULT_RR_HIV,           # rr_hiv
        DEFAULT_ART_REDUCTION,    # art_reduction
        DEFAULT_HIV_INFECTION_RATE,        # hiv_infection_rate
        DEFAULT_ART_INITIATION_RATE,       # art_initiation_rate
        DEFAULT_DR_FITNESS_COST,           # dr_fitness_cost
        DEFAULT_P_ACQ,                     # p_acq
    ])

    # 提案标准差（粗略，TB 时间尺度）
    V4_PROPOSAL_SD = np.array([
        0.02, 5e-5, 2e-6, 1e-3, 5e-3,
        1e-3, 1e-3, 0.05, 0.05,
        2.0, 0.05, 2e-4, 5e-4, 0.03, 0.01,
    ])

    # 支撑域下界 / 上界
    V4_LOWER_BOUNDS = np.array([
        0.001, 1e-5, 1e-7, 1e-4, 5e-3,
        1e-4, 1e-4, 0.01, 0.01,
        5.0, 0.20, 1e-6, 1e-5, 0.02, 0.005,
    ])
    V4_UPPER_BOUNDS = np.array([
        0.95, 1e-2, 1e-4, 5e-2, 0.2,
        5e-2, 5e-2, 0.95, 1.0,
        50.0, 0.90, 0.01, 1e-2, 0.40, 0.10,
    ])

    def log_prior_v4(self, beta, rho_fast, rho_react, sigma_clear, gamma,
                     omega_reg_m=DEFAULT_OMEGA_REG_M,
                     omega_reg_sub=DEFAULT_OMEGA_REG_SUB,
                     beta_exo=DEFAULT_BETA_EXO,
                     eta_sn=None,
                     rr_hiv=DEFAULT_RR_HIV,
                     art_reduction=DEFAULT_ART_REDUCTION,
                     hiv_infection_rate=DEFAULT_HIV_INFECTION_RATE,
                     art_initiation_rate=DEFAULT_ART_INITIATION_RATE,
                     dr_fitness_cost=DEFAULT_DR_FITNESS_COST,
                     p_acq=DEFAULT_P_ACQ):
        """v4.0 年龄×HIV×耐药分层模型先验分布

        在 v3 先验基础上增加 v4 新参数：
          RR_HIV:              Gamma(2, 10)  均值≈20, 范围 [5, 50] (Pawlowski 2012)
          ART_reduction:       Beta(13, 7)   均值≈0.65 (ART 降低 ~65% TB 风险)
          HIV_infection_rate:  Gamma(2, 5e-4) 均值≈0.001/day (外生 HIV 感染率)
          ART_initiation_rate: Gamma(2, 1.4e-4) 均值≈2.7e-4/day ≈ 0.1/年
          DR_fitness_cost:     Beta(6, 30)   均值≈0.15 (Karmakar 2022)
          p_acq:               Beta(3, 90)   均值≈0.032 (获得性耐药概率, Karmakar 2022)

        WAIFW 与 age_progression 为固定结构参数（来自 POLYMOD 调查，Mossong 2008），
        不进入推断参数空间，避免维度爆炸。
        """
        np = self._np
        # 复用 v3 先验的核心参数（含波动/外源再感染/涂阴）
        lp = self.log_prior_v3(
            beta, rho_fast, rho_react, sigma_clear, gamma,
            omega_reg_m=omega_reg_m, omega_reg_sub=omega_reg_sub,
            beta_exo=beta_exo, eta_sn=eta_sn,
        )
        if not np.isfinite(lp):
            return -np.inf

        try:
            from scipy import stats
            _has = True
        except ImportError:
            _has = False

        # ---------- HIV 共感染参数 (Pawlowski 2012, Houben 2016) ----------
        # RR_HIV: Gamma(2, scale=10), 均值≈20, 范围 [5, 50]
        if not (5.0 < rr_hiv < 50.0):
            return -np.inf
        if _has:
            lp += stats.gamma.logpdf(rr_hiv, 2, scale=10.0)
        else:
            lp += (2 - 1) * np.log(rr_hiv + 1e-10) - rr_hiv / 10.0

        # ART_reduction: Beta(13, 7), 均值≈0.65
        if not (0.20 < art_reduction < 0.90):
            return -np.inf
        if _has:
            lp += stats.beta.logpdf(art_reduction, 13, 7)
        else:
            lp += (13 - 1) * np.log(art_reduction + 1e-10) + \
                  (7 - 1) * np.log(1 - art_reduction + 1e-10)

        # HIV 感染率: Gamma(2, scale=5e-4), 均值≈0.001/day
        if not (1e-6 < hiv_infection_rate < 0.01):
            return -np.inf
        if _has:
            lp += stats.gamma.logpdf(hiv_infection_rate, 2, scale=5e-4)
        else:
            lp += (2 - 1) * np.log(hiv_infection_rate + 1e-10) - \
                  hiv_infection_rate / 5e-4

        # ART 启动率: Gamma(2, scale=1.4e-4), 均值≈2.7e-4/day ≈ 0.1/年
        if not (1e-5 < art_initiation_rate < 1e-2):
            return -np.inf
        if _has:
            lp += stats.gamma.logpdf(art_initiation_rate, 2, scale=1.4e-4)
        else:
            lp += (2 - 1) * np.log(art_initiation_rate + 1e-10) - \
                  art_initiation_rate / 1.4e-4

        # ---------- 耐药 TB 参数 (Karmakar 2022, Houben & Dodd 2016) ----------
        # DR 适合度代价: Beta(6, 30), 均值≈0.15, 范围 [0.02, 0.40]
        if not (0.02 < dr_fitness_cost < 0.40):
            return -np.inf
        if _has:
            lp += stats.beta.logpdf(dr_fitness_cost, 6, 30)
        else:
            lp += (6 - 1) * np.log(dr_fitness_cost + 1e-10) + \
                  (30 - 1) * np.log(1 - dr_fitness_cost + 1e-10)

        # 获得性耐药概率: Beta(3, 90), 均值≈0.032, 范围 [0.005, 0.10]
        if not (0.005 < p_acq < 0.10):
            return -np.inf
        if _has:
            lp += stats.beta.logpdf(p_acq, 3, 90)
        else:
            lp += (3 - 1) * np.log(p_acq + 1e-10) + \
                  (90 - 1) * np.log(1 - p_acq + 1e-10)

        return lp

    def log_likelihood_v4(self, params, observed_data, initial_state,
                           t_span, dt, waifw=None, age_progression=None,
                           use_neg_binom=True, neg_binom_k=None):
        """v4.0 270D 模型对数似然

        使用 simulate_sde_v4 模拟，从聚合的活动性 TB（I_sub+I_sp+I_sn，所有分层求和）
        轨迹与观测数据计算似然。

        改进5: 默认使用负二项分布似然（Lloyd-Smith et al. 2005, Nature），
               捕获 TB 报告数据的过度离散与聚集性；可通过 use_neg_binom=False
               回退到高斯似然。

        params: 15 元组，顺序同 V4_PARAM_NAMES
          (beta, rho_fast, rho_react, sigma_clear, gamma,
           omega_reg_m, omega_reg_sub, beta_exo, eta_sn,
           rr_hiv, art_reduction, hiv_infection_rate,
           art_initiation_rate, dr_fitness_cost, p_acq)

        initial_state: 270D 向量或 (5, 3, 2, 9) 数组
        """
        np = self._np
        (beta, rho_fast, rho_react, sigma_clear, gamma,
         omega_reg_m, omega_reg_sub, beta_exo, eta_sn,
         rr_hiv, art_reduction, hiv_infection_rate,
         art_initiation_rate, dr_fitness_cost, p_acq) = params

        if waifw is None:
            waifw = DEFAULT_WAIFW
        if age_progression is None:
            age_progression = DEFAULT_AGE_PROGRESSION

        # 固定随机种子（deterministic likelihood, detailed balance 保护）
        param_str = (f"{beta:.17g},{rho_fast:.17g},{rho_react:.17g},"
                     f"{sigma_clear:.17g},{gamma:.17g},"
                     f"{omega_reg_m:.17g},{omega_reg_sub:.17g},"
                     f"{beta_exo:.17g},{eta_sn:.17g},"
                     f"{rr_hiv:.17g},{art_reduction:.17g},"
                     f"{hiv_infection_rate:.17g},{art_initiation_rate:.17g},"
                     f"{dr_fitness_cost:.17g},{p_acq:.17g}")
        param_hash = int(hashlib.md5(param_str.encode()).hexdigest(), 16)
        old_state = self.seir_model.rng.get_state()
        try:
            self.seir_model.rng.seed(abs(param_hash) % (2**31 - 1))
            _, traj, _ = self.seir_model.simulate_sde_v4(
                initial_state, t_span, dt, beta,
                rho_fast=rho_fast, rho_react=rho_react,
                sigma_clear=sigma_clear, gamma=gamma,
                omega_reg_m=omega_reg_m, omega_reg_sub=omega_reg_sub,
                beta_exo=beta_exo, eta_sn=eta_sn,
                waifw=waifw, age_progression=age_progression,
                rr_hiv=rr_hiv, art_reduction=art_reduction,
                hiv_infection_rate=hiv_infection_rate,
                art_initiation_rate=art_initiation_rate,
                dr_fitness_cost=dr_fitness_cost, p_acq=p_acq,
            )
        finally:
            self.seir_model.rng.seed(0)
            self.seir_model.rng.set_state(old_state)

        # 聚合活动性 TB: 对所有 age×HIV×DR 的 I_sub+I_sp+I_sn 求和
        # traj shape: (n_steps, 270) → reshape (n_steps, 5, 3, 2, 9)
        traj_4d = traj.reshape(-1, N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
        sim_I = (traj_4d[:, :, :, :, IDX_ISUB_V3].sum(axis=(1, 2, 3)) +
                 traj_4d[:, :, :, :, IDX_ISP_V3].sum(axis=(1, 2, 3)) +
                 traj_4d[:, :, :, :, IDX_ISN_V3].sum(axis=(1, 2, 3)))

        obs_I, obs_times = observed_data
        interpolated = np.interp(
            obs_times, np.linspace(t_span[0], t_span[1], len(sim_I)), sim_I)

        if use_neg_binom:
            # 改进5: 负二项似然 (捕获过度离散与聚集性)
            ll = self._neg_binom_loglik(obs_I, interpolated, neg_binom_k)
        else:
            # 高斯似然（向后兼容）
            std_obs = np.std(obs_I) if len(obs_I) > 0 else 0.0
            if not np.isfinite(std_obs):
                std_obs = 0.0
            obs_noise = max(std_obs * 0.2, 0.5)
            residuals = obs_I - interpolated
            ll = -0.5 * np.sum((residuals / obs_noise) ** 2) - \
                 len(obs_I) * np.log(obs_noise * np.sqrt(2 * np.pi))
        if np.isnan(ll) or np.isinf(ll):
            return -np.inf
        return float(ll)

    def log_posterior_v4(self, params, observed_data, initial_state,
                          t_span, dt, waifw=None, age_progression=None,
                          use_neg_binom=True, neg_binom_k=None):
        """v4.0 270D 模型对数后验"""
        np = self._np
        (beta, rho_fast, rho_react, sigma_clear, gamma,
         omega_reg_m, omega_reg_sub, beta_exo, eta_sn,
         rr_hiv, art_reduction, hiv_infection_rate,
         art_initiation_rate, dr_fitness_cost, p_acq) = params
        lp = self.log_prior_v4(
            beta, rho_fast, rho_react, sigma_clear, gamma,
            omega_reg_m=omega_reg_m, omega_reg_sub=omega_reg_sub,
            beta_exo=beta_exo, eta_sn=eta_sn,
            rr_hiv=rr_hiv, art_reduction=art_reduction,
            hiv_infection_rate=hiv_infection_rate,
            art_initiation_rate=art_initiation_rate,
            dr_fitness_cost=dr_fitness_cost, p_acq=p_acq,
        )
        if np.isneginf(lp):
            return -np.inf
        return lp + self.log_likelihood_v4(
            params, observed_data, initial_state, t_span, dt,
            waifw=waifw, age_progression=age_progression,
            use_neg_binom=use_neg_binom, neg_binom_k=neg_binom_k,
        )

    # ==================================================================
    # 改进4: v4.0 270D 模型 HMC/NUTS 采样
    # ==================================================================

    def run_mcmc_v4(self, observed_data, initial_state, t_span, dt,
                    n_warmup=1000, n_samples=2000, n_chains=4,
                    n_leapfrog=20, target_accept=0.8,
                    init_params=None, waifw=None, age_progression=None,
                    seed=42, use_hash_cache=False, cache_maxsize=1024,
                    method='auto'):
        """v4.0 270D 模型 HMC 采样（改进4）

        利用 autodiff.py 提供的 PyTorch 梯度，在 15 维无约束参数空间中
        运行 HMC（leapfrog + 对偶平均法步长自适应 + 对角质量矩阵）。

        参数:
          observed_data: (obs_I, obs_times)
          initial_state: 270D 向量或 (5,3,2,9)
          t_span, dt: 时间跨度与步长
          n_warmup: 预热迭代（对偶平均法 + 质量矩阵估计）
          n_samples: 每链采样数
          n_chains: 并行链数
          n_leapfrog: leapfrog 步数
          target_accept: 目标接受率（默认 0.8）
          init_params: 15 维约束空间初始参数（None 则用 V4_INIT_PARAMS）
          seed: 随机种子
          use_hash_cache: 改进15 (六) 是否启用 MD5 hash 似然缓存 (默认 False)
          cache_maxsize: hash 缓存最大条目数 (默认 1024)
          method: 改进16 (五) ODE 积分方法, 'auto' (默认, 短轨迹 RK4 /
            长轨迹 Euler — M 级审计修复: 270 维模型对 dt 敏感)、
            'euler' 或 'rk4'

        返回:
          dict: 约束空间后验样本 + 诊断信息

        参考文献:
          Hoffman & Gelman (2014) JMLR 15(1):1593-1623 — NUTS/HMC
          Nesterov (2009) — 对偶平均法
        """
        np = self._np
        if not TORCH_AVAILABLE:
            raise ImportError("run_mcmc_v4 需要 PyTorch。请安装: pip install torch")
        from ..autodiff import (compute_grad_log_posterior,
                                torch_log_posterior_v4,
                                unconstrain,
                                PARAM_BOUNDS, PARAM_NAMES, N_INFER_PARAMS)

        # 初始参数 → 无约束空间
        if init_params is None:
            init_params = self.V4_INIT_PARAMS.copy()
        init_arr = np.asarray(init_params, dtype=float)
        y_init = np.zeros(N_INFER_PARAMS)
        for i in range(N_INFER_PARAMS):
            lo, hi = PARAM_BOUNDS[i]
            y_init[i] = float(unconstrain(
                torch.tensor(init_arr[i]), lo, hi))

        # 固定结构参数
        waifw_t = None
        age_prog_t = None
        if waifw is not None or age_progression is not None:
            import torch as _torch
            if waifw is not None:
                waifw_t = _torch.tensor(np.asarray(waifw, dtype=float),
                                         dtype=_torch.float64)
            if age_progression is not None:
                age_prog_t = _torch.tensor(
                    np.asarray(age_progression, dtype=float),
                    dtype=_torch.float64)

        # 梯度函数闭包（匹配 HMCSampler.grad_fn 签名）
        # 改进16 (五): 透传 method 参数支持 RK4
        def grad_fn(theta):
            return compute_grad_log_posterior(
                theta, observed_data, initial_state,
                t_span, dt, waifw_t=waifw_t, age_prog_t=age_prog_t,
                method=method)

        def log_post_fn(theta):
            return torch_log_posterior_v4(
                theta, observed_data, initial_state,
                t_span, dt, waifw_t=waifw_t, age_prog_t=age_prog_t,
                method=method)

        # 运行 HMC 采样
        sampler = HMCSampler(
            log_post_fn, grad_fn, N_INFER_PARAMS,
            target_accept=target_accept, n_leapfrog=n_leapfrog,
            n_warmup=n_warmup, n_samples=n_samples, n_chains=n_chains,
            seed=seed,
            use_hash_cache=use_hash_cache, cache_maxsize=cache_maxsize)

        def init_params_fn(chain_idx):
            # 各链从初始点加扰动开始
            jitter = self.rng.normal(0, 0.1, N_INFER_PARAMS)
            return y_init + jitter

        result = sampler.sample(init_params_fn=init_params_fn)

        # 提取约束空间后验样本
        constrained = sampler.get_constrained_samples()
        self.posterior_samples = constrained
        self.traces = {
            'chains': result['chains'],
            'log_probs': result['log_probs'],
            'acceptance_rates': result['acceptance_rates'],
        }

        # 组装诊断
        diag = dict(result['diagnostics'])
        diag['param_names'] = list(PARAM_NAMES)
        diag['algorithm'] = 'HMC + Dual Averaging (Nesterov 2009) + Adaptive Leapfrog'
        diag['reference'] = ('Hoffman & Gelman (2014) JMLR 15(1):1593-1623')
        for name in PARAM_NAMES:
            samples = constrained[name]
            if len(samples) > 0:
                diag[f'{name}_mean'] = float(np.mean(samples))

        # 人口守恒显式检查 (M 级审计修复): 采样结束后用后验均值参数
        # (无约束空间链均值为代表性参数点) 重积分一次 (no_grad),
        # 记录 N=S+...+C 最大相对漂移。封闭 v4 模型解析解严格守恒,
        # 漂移来自 Euler/softplus 数值路径 — 该指标超阈值说明当前
        # dt/method 配置不可信, 提示减小 dt 或改用 rk4。
        # 开销: 一次前向积分, 相对采样总成本可忽略。
        try:
            import warnings as _warnings
            import torch as _torch
            from ..autodiff import (
                torch_v4_integrate, population_conservation_error,
                POP_CONSERVATION_RTOL)
            y_mean = _torch.tensor(
                np.concatenate(result['chains'], axis=0).mean(axis=0),
                dtype=_torch.float64)
            n_steps_check = int((t_span[1] - t_span[0]) / dt)
            with _torch.no_grad():
                traj_check = torch_v4_integrate(
                    y_mean, initial_state, n_steps_check, dt,
                    waifw_t, age_prog_t, method=method)
            pop_err = population_conservation_error(traj_check)
            diag['pop_conservation_max_rel_err'] = pop_err
            diag['pop_conservation_rtol'] = POP_CONSERVATION_RTOL
            if pop_err > POP_CONSERVATION_RTOL:
                _warnings.warn(
                    f"v4 HMC 后验轨迹人口守恒违反: "
                    f"max|N(t)-N(0)|/N(0) = {pop_err:.3e} > "
                    f"{POP_CONSERVATION_RTOL:.1e} (method={method}, "
                    f"dt={dt})。后验动力学推断的可信度受限, 建议减小 "
                    f"dt 或 method='rk4' 重跑。")
        except Exception as _e:  # 守恒检查失败不影响采样结果返回
            diag['pop_conservation_max_rel_err'] = float('nan')
            diag['pop_conservation_check_error'] = str(_e)

        self.convergence_diagnostics = diag
        return self.posterior_samples
