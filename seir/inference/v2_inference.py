"""v2 推断引擎：7-房室 SEIR 模型自适应 Metropolis-Hastings MCMC。

包含方法：
  log_prior_v2 / log_likelihood_v2 / log_posterior_v2
  metropolis_hastings_v2 / _run_single_chain_v2

7-房室参数配置（V3_PARAM_NAMES 系列）定义于本类。
"""

import math
import hashlib
import warnings

import numpy as np

from ..diagnostics import compute_r_hat, compute_ess
from ..stochastic import DEFAULT_ETA_SUB, IDX_ISUB, IDX_ICLIN
from .v1_inference import BayesianInferenceV1


class BayesianInferenceV2(BayesianInferenceV1):
    """v2 7-房室 SEIR 模型贝叶斯推断（v3.0 快/慢分层 + 自清除 + 亚临床）。

    .. deprecated::
        v2 已弃用，建议使用 BayesianInferenceV4（HMC + 270 维年龄×HIV×耐药分层）。
        v2 仅保留用于向后兼容和单元测试参照。

    参数空间（per day）：
      β, ρ_fast, ρ_react, σ_clear, γ [, η_sub]
    """

    def __init__(self, seir_model=None, seed=42, **kwargs):
        warnings.warn(
            "BayesianInferenceV2 已弃用，建议使用 BayesianInferenceV4。"
            "v2 仅保留用于向后兼容。",
            DeprecationWarning,
            stacklevel=2,
        )
        # 直接调用 BaseBayesianInference.__init__ 避免 v1 的重复 DeprecationWarning
        from ._base import BaseBayesianInference
        BaseBayesianInference.__init__(self, seir_model=seir_model, seed=seed)

    # ========== v3.0 7-房室模型参数配置 ==========
    # 核心推断参数名列表（按顺序）
    V3_PARAM_NAMES = ['beta', 'rho_fast', 'rho_react', 'sigma_clear', 'gamma']
    V3_PARAM_NAMES_EXTENDED = ['beta', 'rho_fast', 'rho_react', 'sigma_clear', 'gamma', 'eta_sub']
    V3_N_PARAMS = 5
    V3_N_PARAMS_EXTENDED = 6

    # 参数初始值（per day）
    V3_INIT_PARAMS = np.array([
        0.2,               # beta: 传播率
        0.1 / 365.0,       # rho_fast: ~0.000274/day (Andrews 2012)
        0.004 / 365.0,     # rho_react: ~1.1e-5/day (Vynnycky 1997)
        2.0 / 365.0,       # sigma_clear: ~0.00548/day (Horton 2023)
        0.05,              # gamma: 恢复率 /day
    ])
    V3_INIT_PARAMS_EXTENDED = np.array([
        0.2, 0.1 / 365.0, 0.004 / 365.0, 2.0 / 365.0, 0.05, 1.0
    ])

    # 提案协方差对角元（每个参数的标准差）
    V3_PROPOSAL_SD = np.array([0.02, 0.00005, 2e-6, 0.001, 0.005])
    V3_PROPOSAL_SD_EXTENDED = np.array([0.02, 0.00005, 2e-6, 0.001, 0.005, 0.1])

    # 参数支撑域下界（per day）
    V3_LOWER_BOUNDS = np.array([
        0.001,             # beta: 0.001 < β < 0.95
        0.00001,           # rho_fast: ~0.004/year ~ 0.00001/day
        1e-7,              # rho_react: ~0.00004/year ~ 1e-7/day
        0.0001,            # sigma_clear: ~0.036/year ~ 0.0001/day
        0.005,             # gamma: 5-200天传染期
    ])
    V3_LOWER_BOUNDS_EXTENDED = np.array([
        0.001, 0.00001, 1e-7, 0.0001, 0.005, 0.01
    ])

    # 参数支撑域上界（per day）
    V3_UPPER_BOUNDS = np.array([
        0.95,              # beta
        0.01,              # rho_fast: ~3.65/year
        0.0001,            # rho_react: ~0.036/year
        0.05,              # sigma_clear: ~18/year
        0.2,               # gamma: 5天
    ])
    V3_UPPER_BOUNDS_EXTENDED = np.array([
        0.95, 0.01, 0.0001, 0.05, 0.2, 5.0
    ])

    def log_prior_v2(self, beta, rho_fast, rho_react, sigma_clear, gamma,
                     eta_sub=None):
        """7-房室模型先验分布（v3.0）

        参数先验（per day，文献校准）：
          β:        Beta(2, 8)          均值≈0.2
          ρ_fast:   Gamma(2, 0.05/365)  均值≈0.000274/day ≈ 0.1/年 (Andrews 2012)
          ρ_react:  Gamma(2, 0.002/365) 均值≈1.1e-5/day ≈ 0.004/年 (Vynnycky 1997)
          σ_clear:  Gamma(2, 1/365)     均值≈0.00548/day ≈ 2.0/年 (Horton 2023)
          γ:        Gamma(2, 0.025)     均值≈0.05/day

        可选参数：
          η_sub:    Beta(2, 5) 或 Uniform(0.3, 1.0) — 亚临床相对传染性
                    (Emery et al. 2023, eLife)
        """
        np = self._np
        try:
            from scipy import stats
            _has_scipy_stats = True
        except ImportError:
            _has_scipy_stats = False

        # 参数支撑域检查（per day）
        if not (0.001 < beta < 0.95):
            return -np.inf
        if not (0.00001 < rho_fast < 0.01):      # ~0.004~3.65/年
            return -np.inf
        if not (1e-7 < rho_react < 0.0001):       # ~0.00004~0.036/年
            return -np.inf
        if not (0.0001 < sigma_clear < 0.05):     # ~0.036~18/年
            return -np.inf
        if not (0.005 < gamma < 0.2):             # 5~200天传染期
            return -np.inf

        if _has_scipy_stats:
            lp = stats.beta.logpdf(beta, 2, 8)
            lp += stats.gamma.logpdf(rho_fast, 2, scale=0.05 / 365.0)
            lp += stats.gamma.logpdf(rho_react, 2, scale=0.002 / 365.0)
            lp += stats.gamma.logpdf(sigma_clear, 2, scale=1.0 / 365.0)
            lp += stats.gamma.logpdf(gamma, 2, scale=0.025)
        else:
            lp = (2 - 1) * np.log(beta + 1e-10) + (8 - 1) * np.log(1 - beta + 1e-10)
            lp += (2 - 1) * np.log(rho_fast + 1e-10) - rho_fast / (0.05 / 365.0)
            lp += (2 - 1) * np.log(rho_react + 1e-10) - rho_react / (0.002 / 365.0)
            lp += (2 - 1) * np.log(sigma_clear + 1e-10) - sigma_clear / (1.0 / 365.0)
            lp += (2 - 1) * np.log(gamma + 1e-10) - gamma / 0.025

        # 可选：η_sub 先验 (改进9: Beta(2,2) 缩放到 [0.1, 3.0], Emery 2023 eLife)
        # Emery et al. 估计 η_sub 中位数 1.93 (95% PI: 0.62-6.18)，此处取保守区间
        if eta_sub is not None:
            if not (0.1 < eta_sub < 3.0):
                return -np.inf
            # eta_sub = 0.1 + x * 2.9,  x ~ Beta(2, 2); 含 Jacobian 项 -log(2.9)
            x = (eta_sub - 0.1) / (3.0 - 0.1)
            if _has_scipy_stats:
                lp += stats.beta.logpdf(x, 2, 2) - np.log(3.0 - 0.1)
            else:
                if 0 < x < 1:
                    lp += ((2 - 1) * np.log(x + 1e-10)
                           + (2 - 1) * np.log(1 - x + 1e-10)
                           - np.log(3.0 - 0.1))

        return lp

    def log_likelihood_v2(self, params, observed_data, initial_state, t_span, dt,
                          include_eta_sub=False):
        """7-房室模型对数似然函数（v3.0）

        使用 simulate_sde_v2 进行7-房室SDE模拟，
        从模拟的 I_sub + I_clin 轨迹与观测数据计算残差。

        参数：
            params: 参数元组
                (beta, rho_fast, rho_react, sigma_clear, gamma) 或
                (beta, rho_fast, rho_react, sigma_clear, gamma, eta_sub)
            observed_data: (obs_I, obs_times) 观测序列
            initial_state: 7-房室初始状态 [S, Lf, Ls, Isub, Iclin, R, C]
            t_span, dt: 时间跨度与步长
            include_eta_sub: 是否包含 eta_sub 参数
        """
        np = self._np
        if include_eta_sub:
            beta, rho_fast, rho_react, sigma_clear, gamma, eta_sub = params
        else:
            beta, rho_fast, rho_react, sigma_clear, gamma = params
            eta_sub = DEFAULT_ETA_SUB

        # 固定随机种子（deterministic likelihood）
        param_str = (f"{beta:.17g},{rho_fast:.17g},{rho_react:.17g},"
                     f"{sigma_clear:.17g},{gamma:.17g},{eta_sub:.17g}")
        param_hash = int(hashlib.md5(param_str.encode()).hexdigest(), 16)
        old_state = self.seir_model.rng.get_state()
        try:
            self.seir_model.rng.seed(abs(param_hash) % (2**31 - 1))
            _, traj, _ = self.seir_model.simulate_sde_v2(
                initial_state, t_span, dt, beta,
                rho_fast=rho_fast, rho_react=rho_react,
                sigma_clear=sigma_clear, gamma=gamma,
                eta_sub=eta_sub,
            )
        finally:
            self.seir_model.rng.seed(0)
            self.seir_model.rng.set_state(old_state)

        # 活动性TB = I_sub + I_clin
        sim_I = traj[:, IDX_ISUB] + traj[:, IDX_ICLIN]
        obs_I, obs_times = observed_data
        interpolated = np.interp(
            obs_times, np.linspace(t_span[0], t_span[1], len(sim_I)), sim_I)
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

    def log_posterior_v2(self, params, observed_data, initial_state, t_span, dt,
                         include_eta_sub=False):
        """7-房室模型对数后验（v3.0）"""
        if include_eta_sub:
            beta, rho_fast, rho_react, sigma_clear, gamma, eta_sub = params
            lp = self.log_prior_v2(beta, rho_fast, rho_react, sigma_clear, gamma,
                                   eta_sub=eta_sub)
        else:
            beta, rho_fast, rho_react, sigma_clear, gamma = params
            lp = self.log_prior_v2(beta, rho_fast, rho_react, sigma_clear, gamma)
        if np.isneginf(lp):
            return -np.inf
        return lp + self.log_likelihood_v2(
            params, observed_data, initial_state, t_span, dt,
            include_eta_sub=include_eta_sub)

    # ==================== v3.0 7-房室 MCMC 方法 ====================

    def metropolis_hastings_v2(self, observed_data, initial_state, t_span, dt,
                               n_iterations=None, burn_in=None,
                               n_chains=None, proposal_cov=None, init_params=None,
                               adapt=True, adapt_interval=100, adapt_start=50,
                               include_eta_sub=False):
        """7-房室模型自适应 Metropolis-Hastings MCMC（v3.0）

        参数：
            observed_data: (obs_I, obs_times) 观测序列
            initial_state: 7-房室初始状态 [S, Lf, Ls, Isub, Iclin, R, C]
            t_span, dt: 时间跨度与步长
            n_iterations: 总迭代数（默认 10000）
            burn_in: 预热期（默认 2000）
            n_chains: 并行链数（默认 4）
            proposal_cov: 初始提案协方差
            init_params: 初始参数
                (beta, rho_fast, rho_react, sigma_clear, gamma) 或
                (beta, rho_fast, rho_react, sigma_clear, gamma, eta_sub)
            adapt: 是否启用 Haario 2001 自适应提案
            adapt_interval: 自适应更新间隔
            adapt_start: 自适应启动所需最小样本数
            include_eta_sub: 是否包含 eta_sub 参数（6维模式）

        返回：
            dict: 后验采样结果

        参考文献：
            Haario H, Saksman E, Tamminen J. (2001) An adaptive Metropolis algorithm.
            Bernoulli 7(2):223-242.
            Vehtari A et al. (2021) Rank-normalization, folding, and localization.
            Bayesian Anal 16(2).
        """
        n_iterations = n_iterations or self.DEFAULT_N_ITERATIONS
        burn_in = burn_in or self.DEFAULT_N_BURNIN
        n_chains = n_chains or self.DEFAULT_N_CHAINS

        # 配置参数维度
        if include_eta_sub:
            param_names = list(self.V3_PARAM_NAMES_EXTENDED)
            n_params = self.V3_N_PARAMS_EXTENDED
            default_init = self.V3_INIT_PARAMS_EXTENDED.copy()
            default_sd = self.V3_PROPOSAL_SD_EXTENDED
            lower_bounds = self.V3_LOWER_BOUNDS_EXTENDED
        else:
            param_names = list(self.V3_PARAM_NAMES)
            n_params = self.V3_N_PARAMS
            default_init = self.V3_INIT_PARAMS.copy()
            default_sd = self.V3_PROPOSAL_SD
            lower_bounds = self.V3_LOWER_BOUNDS

        if init_params is None:
            init_params = default_init
        if proposal_cov is None:
            proposal_cov = np.diag(default_sd ** 2)

        # 运行多条独立链（使用确定性种子派生，不依赖numpy内部状态结构）
        all_chains = []
        all_acceptance_rates = []
        for chain_idx in range(n_chains):
            chain_seed = self._derive_chain_seed(chain_idx)
            chain_rng = np.random.RandomState(chain_seed)

            chain, acc_rate = self._run_single_chain_v2(
                observed_data, initial_state, t_span, dt,
                n_iterations, burn_in, proposal_cov, init_params,
                adapt, adapt_interval, adapt_start, chain_rng,
                n_params, param_names, lower_bounds, include_eta_sub)
            all_chains.append(chain)
            all_acceptance_rates.append(acc_rate)

        # 合并后验样本（burn-in 后）
        merged_chain = np.concatenate(
            [c[burn_in:] for c in all_chains], axis=0)

        self.posterior_samples = {}
        for i, name in enumerate(param_names):
            self.posterior_samples[name] = merged_chain[:, i]
        self.traces = {
            'chains': all_chains,
            'all_acceptance_rates': all_acceptance_rates,
        }

        # 收敛诊断
        r_hat = compute_r_hat(all_chains, burn_in, param_names)
        ess = compute_ess(merged_chain, param_names)

        diag = {
            'n_chains': n_chains,
            'n_iterations': n_iterations,
            'burn_in': burn_in,
            'n_params': n_params,
            'param_names': param_names,
            'acceptance_rate': float(np.mean(all_acceptance_rates)),
            'acceptance_rates_per_chain': all_acceptance_rates,
            'r_hat': r_hat,
            'ess': ess,
            'converged': all(r < 1.1 for r in r_hat.values()),
            'adaptive_mcmc': {
                'enabled': bool(adapt),
                'adapt_interval': adapt_interval,
                'adapt_start': adapt_start,
                'reference': 'Haario et al., 2001, Bernoulli 7(2):223-242',
            },
        }
        for i, name in enumerate(param_names):
            diag[f'{name}_mean'] = float(np.mean(merged_chain[:, i]))
        self.convergence_diagnostics = diag
        return self.posterior_samples

    def _run_single_chain_v2(self, observed_data, initial_state, t_span, dt,
                             n_iterations, burn_in, proposal_cov, init_params,
                             adapt, adapt_interval, adapt_start, chain_rng,
                             n_params, param_names, lower_bounds,
                             include_eta_sub):
        """运行单条 v3.0 MCMC 链（内部方法）。"""
        params = np.array(init_params, dtype=float)
        current_lp = self.log_posterior_v2(
            params, observed_data, initial_state, t_span, dt,
            include_eta_sub=include_eta_sub)

        try:
            L_chol = np.linalg.cholesky(proposal_cov)
        except np.linalg.LinAlgError:
            L_chol = np.sqrt(np.diag(proposal_cov)) * np.eye(n_params)

        chain = np.zeros((n_iterations, n_params))
        accepted = 0
        d = n_params
        s_d = (2.4 ** 2) / max(d, 1)
        eps_reg = 1e-8

        for i in range(n_iterations):
            proposal = params + L_chol @ chain_rng.normal(0, 1, n_params)

            # 下界检查
            if np.any(proposal < lower_bounds):
                proposal_lp = -np.inf
            else:
                proposal_lp = self.log_posterior_v2(
                    proposal, observed_data, initial_state, t_span, dt,
                    include_eta_sub=include_eta_sub)

            if np.isfinite(proposal_lp):
                log_alpha = proposal_lp - current_lp
                u = chain_rng.uniform()
                if u > 0 and math.log(u) < min(log_alpha, 0):
                    params, current_lp = proposal, proposal_lp
                    accepted += 1

            chain[i] = params

            # 自适应提案协方差更新（仅 burn-in 期内）
            if (adapt and i < burn_in
                    and i >= adapt_start
                    and (i + 1) % adapt_interval == 0):
                hist = chain[:i + 1]
                emp_cov = np.cov(hist, rowvar=False)
                if emp_cov.ndim == 0:
                    emp_cov = emp_cov.reshape(1, 1)
                new_cov = s_d * (emp_cov + eps_reg * np.eye(d))
                try:
                    L_chol = np.linalg.cholesky(new_cov)
                    proposal_cov = new_cov
                except np.linalg.LinAlgError:
                    pass

        return chain, accepted / max(n_iterations, 1)
