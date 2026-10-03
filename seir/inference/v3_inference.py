"""v3 推断引擎：9-房室 SEIR 模型自适应 Metropolis-Hastings MCMC。

包含方法：
  log_prior_v3 / log_likelihood_v3 / log_posterior_v3
  run_mcmc_v3 / _run_single_chain_v3
  compute_R0_v3
"""

import math
import hashlib

import numpy as np

from ..diagnostics import compute_r_hat, compute_ess
from ..stochastic import (
    DEFAULT_ETA_SUB, DEFAULT_ETA_SN,
    DEFAULT_OMEGA_REG_M, DEFAULT_OMEGA_REG_SUB, DEFAULT_BETA_EXO,
    IDX_ISUB_V3, IDX_ISP_V3, IDX_ISN_V3,
    DEFAULT_RHO_PROG, DEFAULT_R_SP, DEFAULT_MU_SP,
    DEFAULT_R_SN, DEFAULT_MU_SN, DEFAULT_P_SN2SP,
)
from .v2_inference import BayesianInferenceV2


class BayesianInferenceV3(BayesianInferenceV2):
    """v3 9-房室 SEIR 模型贝叶斯推断（v3.0 改进4/5/6/9）。

    参数空间（per day）：
      基础 8 维: (β, ρ_fast, ρ_react, σ_clear, γ, ω_reg_m, ω_reg_sub, β_exo)
      +η_sub (改进9), +η_sn (末位)
    似然默认负二项 (改进5)。
    """

    # ========== v3.0 9-房室模型参数配置 (改进9: 含 eta_sub 推断) ==========
    # 9-房室基础 8 维推断参数（不含 eta_sub / eta_sn）
    V3_9COMP_PARAM_NAMES = [
        'beta', 'rho_fast', 'rho_react', 'sigma_clear', 'gamma',
        'omega_reg_m', 'omega_reg_sub', 'beta_exo',
    ]
    V3_9COMP_N_PARAMS = 8
    # 改进9: 增加 eta_sub → 9 维（Emery 2023 eLife, 高度不确定参数）
    V3_9COMP_PARAM_NAMES_WITH_ETA_SUB = (
        V3_9COMP_PARAM_NAMES + ['eta_sub'])
    V3_9COMP_N_PARAMS_WITH_ETA_SUB = 9

    V3_9COMP_INIT_PARAMS = np.array([
        0.2,                   # beta
        0.1 / 365.0,           # rho_fast
        0.004 / 365.0,         # rho_react
        2.0 / 365.0,           # sigma_clear
        0.05,                  # gamma
        DEFAULT_OMEGA_REG_M,   # omega_reg_m
        DEFAULT_OMEGA_REG_SUB, # omega_reg_sub
        DEFAULT_BETA_EXO,      # beta_exo
    ])
    V3_9COMP_INIT_PARAMS_WITH_ETA_SUB = np.append(
        V3_9COMP_INIT_PARAMS, 1.5)   # eta_sub 初始值 (Emery 中位数 1.93)

    V3_9COMP_PROPOSAL_SD = np.array([
        0.02, 5e-5, 2e-6, 1e-3, 5e-3, 1e-3, 1e-3, 0.05,
    ])
    V3_9COMP_PROPOSAL_SD_WITH_ETA_SUB = np.append(
        V3_9COMP_PROPOSAL_SD, 0.3)   # eta_sub 提案标准差

    V3_9COMP_LOWER_BOUNDS = np.array([
        0.001, 1e-5, 1e-7, 1e-4, 5e-3, 1e-4, 1e-4, 0.01,
    ])
    V3_9COMP_LOWER_BOUNDS_WITH_ETA_SUB = np.append(
        V3_9COMP_LOWER_BOUNDS, 0.1)   # eta_sub 下界 (改进9: Beta(2,2)→[0.1,3.0])

    V3_9COMP_UPPER_BOUNDS = np.array([
        0.95, 1e-2, 1e-4, 5e-2, 0.2, 5e-2, 5e-2, 0.95,
    ])
    V3_9COMP_UPPER_BOUNDS_WITH_ETA_SUB = np.append(
        V3_9COMP_UPPER_BOUNDS, 3.0)   # eta_sub 上界

    def log_prior_v3(self, beta, rho_fast, rho_react, sigma_clear, gamma,
                     omega_reg_m=DEFAULT_OMEGA_REG_M,
                     omega_reg_sub=DEFAULT_OMEGA_REG_SUB,
                     beta_exo=DEFAULT_BETA_EXO,
                     eta_sub=None, eta_sn=None):
        """9-房室模型先验分布（v3.0）

        在 v2 先验基础上增加：
          ω_reg_m:    Gamma(2, 0.5/365)   均值≈1.0/年 (Horton 2023 波动)
          ω_reg_sub:  Gamma(2, 0.5/365)   均值≈1.0/年
          β_exo:      Beta(2, 4)           均值≈0.33 (外源再感染易感性)
          η_sn:       Beta(2, 5) 缩放[0,1] 均值≈0.22 (涂阴相对传染性, 保守)
        """
        np = self._np
        # 复用 v2 先验的核心参数
        lp = self.log_prior_v2(beta, rho_fast, rho_react, sigma_clear, gamma,
                               eta_sub=eta_sub)
        if not np.isfinite(lp):
            return -np.inf

        try:
            from scipy import stats
            _has = True
        except ImportError:
            _has = False

        # 波动率先验
        if not (0.0001 < omega_reg_m < 0.05):
            return -np.inf
        if not (0.0001 < omega_reg_sub < 0.05):
            return -np.inf
        if _has:
            lp += stats.gamma.logpdf(omega_reg_m, 2, scale=0.5 / 365.0)
            lp += stats.gamma.logpdf(omega_reg_sub, 2, scale=0.5 / 365.0)
        else:
            lp += (2-1) * np.log(omega_reg_m + 1e-10) - omega_reg_m / (0.5/365.0)
            lp += (2-1) * np.log(omega_reg_sub + 1e-10) - omega_reg_sub / (0.5/365.0)

        # β_exo 先验
        if not (0.01 < beta_exo < 0.95):
            return -np.inf
        if _has:
            lp += stats.beta.logpdf(beta_exo, 2, 4)
        else:
            lp += (2-1) * np.log(beta_exo + 1e-10) + (4-1) * np.log(1 - beta_exo + 1e-10)

        # η_sn 先验 (可选)
        if eta_sn is not None:
            if not (0.01 < eta_sn < 1.0):
                return -np.inf
            if _has:
                lp += stats.beta.logpdf(eta_sn, 2, 5)
            else:
                lp += (2-1) * np.log(eta_sn + 1e-10) + (5-1) * np.log(1 - eta_sn + 1e-10)

        return lp

    def log_likelihood_v3(self, params, observed_data, initial_state, t_span, dt,
                          include_eta_sn=False, include_eta_sub=False,
                          use_neg_binom=True, neg_binom_k=None):
        """9-房室模型对数似然（v3.0）

        使用 simulate_sde_v3 模拟，从 I_sub+I_sp+I_sn 轨迹与观测计算似然。

        改进5: 默认使用负二项分布似然（Lloyd-Smith et al. 2005, Nature），
               捕获 TB 报告数据的过度离散与聚集性；可通过 use_neg_binom=False
               回退到高斯似然。
        改进9: include_eta_sub=True 时将 eta_sub 纳入推断参数空间。

        params 顺序:
          基础 8 维: (beta, rho_fast, rho_react, sigma_clear, gamma,
                      omega_reg_m, omega_reg_sub, beta_exo)
          +eta_sub (include_eta_sub): 第 9 位
          +eta_sn   (include_eta_sn):   末位（在 eta_sub 之后）
        initial_state: 9-房室 [S, Lf, Ls, M, Isub, Isp, Isn, R, C]
        """
        np = self._np
        # ---- 按 flags 解包参数 ----
        if include_eta_sub and include_eta_sn:
            (beta, rho_fast, rho_react, sigma_clear, gamma,
             omega_reg_m, omega_reg_sub, beta_exo,
             eta_sub, eta_sn) = params
        elif include_eta_sub:
            (beta, rho_fast, rho_react, sigma_clear, gamma,
             omega_reg_m, omega_reg_sub, beta_exo, eta_sub) = params
            eta_sn = DEFAULT_ETA_SN
        elif include_eta_sn:
            (beta, rho_fast, rho_react, sigma_clear, gamma,
             omega_reg_m, omega_reg_sub, beta_exo, eta_sn) = params
            eta_sub = DEFAULT_ETA_SUB
        else:
            (beta, rho_fast, rho_react, sigma_clear, gamma,
             omega_reg_m, omega_reg_sub, beta_exo) = params
            eta_sub = DEFAULT_ETA_SUB
            eta_sn = DEFAULT_ETA_SN

        param_str = (f"{beta:.17g},{rho_fast:.17g},{rho_react:.17g},"
                     f"{sigma_clear:.17g},{gamma:.17g},"
                     f"{omega_reg_m:.17g},{omega_reg_sub:.17g},"
                     f"{beta_exo:.17g},{eta_sub:.17g},{eta_sn:.17g}")
        param_hash = int(hashlib.md5(param_str.encode()).hexdigest(), 16)
        old_state = self.seir_model.rng.get_state()
        try:
            self.seir_model.rng.seed(abs(param_hash) % (2**31 - 1))
            _, traj, _ = self.seir_model.simulate_sde_v3(
                initial_state, t_span, dt, beta,
                rho_fast=rho_fast, rho_react=rho_react,
                sigma_clear=sigma_clear, gamma=gamma,
                eta_sub=eta_sub,
                omega_reg_m=omega_reg_m, omega_reg_sub=omega_reg_sub,
                beta_exo=beta_exo, eta_sn=eta_sn,
            )
        finally:
            self.seir_model.rng.seed(0)
            self.seir_model.rng.set_state(old_state)

        # 活动性TB = I_sub + I_sp + I_sn
        sim_I = (traj[:, IDX_ISUB_V3] + traj[:, IDX_ISP_V3] + traj[:, IDX_ISN_V3])
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

    def log_posterior_v3(self, params, observed_data, initial_state, t_span, dt,
                         include_eta_sn=False, include_eta_sub=False,
                         use_neg_binom=True, neg_binom_k=None):
        """9-房室模型对数后验（v3.0）

        改进9: include_eta_sub=True 时 eta_sub 进入先验与推断空间。
        """
        np = self._np
        # ---- 按 flags 解包参数 ----
        if include_eta_sub and include_eta_sn:
            (beta, rho_fast, rho_react, sigma_clear, gamma,
             omega_reg_m, omega_reg_sub, beta_exo,
             eta_sub, eta_sn) = params
        elif include_eta_sub:
            (beta, rho_fast, rho_react, sigma_clear, gamma,
             omega_reg_m, omega_reg_sub, beta_exo, eta_sub) = params
            eta_sn = None
        elif include_eta_sn:
            (beta, rho_fast, rho_react, sigma_clear, gamma,
             omega_reg_m, omega_reg_sub, beta_exo, eta_sn) = params
            eta_sub = None
        else:
            (beta, rho_fast, rho_react, sigma_clear, gamma,
             omega_reg_m, omega_reg_sub, beta_exo) = params
            eta_sub = None
            eta_sn = None

        lp = self.log_prior_v3(beta, rho_fast, rho_react, sigma_clear, gamma,
                                omega_reg_m=omega_reg_m,
                                omega_reg_sub=omega_reg_sub,
                                beta_exo=beta_exo,
                                eta_sub=eta_sub, eta_sn=eta_sn)
        if np.isneginf(lp):
            return -np.inf
        return lp + self.log_likelihood_v3(
            params, observed_data, initial_state, t_span, dt,
            include_eta_sn=include_eta_sn, include_eta_sub=include_eta_sub,
            use_neg_binom=use_neg_binom, neg_binom_k=neg_binom_k)

    # ==================================================================
    # 改进4: v3.0 9-房室模型自适应 Metropolis-Hastings MCMC
    # ==================================================================

    def run_mcmc_v3(self, observed_data, initial_state, t_span, dt,
                    n_iterations=None, burn_in=None,
                    n_chains=None, proposal_cov=None, init_params=None,
                    adapt=True, adapt_interval=100, adapt_start=50,
                    include_eta_sub=False, include_eta_sn=False,
                    use_neg_binom=True, neg_binom_k=None,
                    use_hash_cache=False, cache_maxsize=1024):
        """9-房室模型自适应 Metropolis-Hastings MCMC（改进4）

        参数空间:
          - 基础 8 维: (beta, rho_fast, rho_react, sigma_clear, gamma,
                        omega_reg_m, omega_reg_sub, beta_exo)
          - include_eta_sub=True → 9 维 (增加 eta_sub, 改进9)
          - include_eta_sn=True → 末位增加 eta_sn

        复用 v1/v2 框架: Haario 2001 自适应协方差 + rank-normalized
        R-hat (Vehtari 2021) + ESS。似然默认负二项 (改进5)。

        改进15 (六): ``use_hash_cache=True`` 启用 MD5 hash 似然缓存。
        v3 MH 中每次 ``log_posterior_v3`` 调用运行完整随机 SDE 仿真
        (昂贵, ~ms), 缓存键为 MD5(params.tobytes())。连续空间中同一
        参数被提议两次的概率低, 但 v3 已用 param_hash 固定 RNG 种子
        (确保 detailed balance), 缓存与之一致 — 同参数必同轨迹必同
        似然, 缓存零风险。命中率 1-5% 时仍有正收益 (每次命中省 ~ms)。

        参考文献:
          Haario et al. (2001) Bernoulli 7(2):223-242
          Vehtari et al. (2021) Bayesian Anal 16(2)
        """
        np = self._np
        n_iterations = n_iterations or self.DEFAULT_N_ITERATIONS
        burn_in = burn_in or self.DEFAULT_N_BURNIN
        n_chains = n_chains or self.DEFAULT_N_CHAINS

        # 配置参数维度
        if include_eta_sub:
            param_names = list(self.V3_9COMP_PARAM_NAMES_WITH_ETA_SUB)
            n_params = self.V3_9COMP_N_PARAMS_WITH_ETA_SUB
            default_init = self.V3_9COMP_INIT_PARAMS_WITH_ETA_SUB.copy()
            default_sd = self.V3_9COMP_PROPOSAL_SD_WITH_ETA_SUB
            lower_bounds = self.V3_9COMP_LOWER_BOUNDS_WITH_ETA_SUB
        else:
            param_names = list(self.V3_9COMP_PARAM_NAMES)
            n_params = self.V3_9COMP_N_PARAMS
            default_init = self.V3_9COMP_INIT_PARAMS.copy()
            default_sd = self.V3_9COMP_PROPOSAL_SD
            lower_bounds = self.V3_9COMP_LOWER_BOUNDS

        if include_eta_sn:
            param_names = param_names + ['eta_sn']
            n_params += 1
            default_init = np.append(default_init, DEFAULT_ETA_SN)
            default_sd = np.append(default_sd, 0.05)
            lower_bounds = np.append(lower_bounds, 0.01)

        if init_params is None:
            init_params = default_init
        if proposal_cov is None:
            proposal_cov = np.diag(default_sd ** 2)

        # 运行多条独立链（使用确定性种子派生，不依赖numpy内部状态结构）
        all_chains = []
        all_acceptance_rates = []
        all_cache_stats = []  # 改进15 (六): 各链缓存统计
        for chain_idx in range(n_chains):
            chain_seed = self._derive_chain_seed(chain_idx)
            chain_rng = np.random.RandomState(chain_seed)

            chain, acc_rate, cache_stats = self._run_single_chain_v3(
                observed_data, initial_state, t_span, dt,
                n_iterations, burn_in, proposal_cov, init_params,
                adapt, adapt_interval, adapt_start, chain_rng,
                n_params, lower_bounds, include_eta_sub, include_eta_sn,
                use_neg_binom, neg_binom_k,
                use_hash_cache=use_hash_cache, cache_maxsize=cache_maxsize)
            all_chains.append(chain)
            all_acceptance_rates.append(acc_rate)
            all_cache_stats.append(cache_stats)

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
            'include_eta_sub': include_eta_sub,
            'include_eta_sn': include_eta_sn,
            'likelihood': 'negative_binomial' if use_neg_binom else 'gaussian',
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
        # 改进15 (六): hash cache 统计 (仅 use_hash_cache=True 时有意义)
        if use_hash_cache:
            diag['hash_cache_stats'] = all_cache_stats
        self.convergence_diagnostics = diag
        return self.posterior_samples

    def _run_single_chain_v3(self, observed_data, initial_state, t_span, dt,
                             n_iterations, burn_in, proposal_cov, init_params,
                             adapt, adapt_interval, adapt_start, chain_rng,
                             n_params, lower_bounds, include_eta_sub,
                             include_eta_sn, use_neg_binom, neg_binom_k,
                             use_hash_cache=False, cache_maxsize=1024):
        """运行单条 v3.0 9-房室 MCMC 链（内部方法）。

        改进15 (六): ``use_hash_cache=True`` 时启用 MD5 hash 似然缓存。
        每次 ``log_posterior_v3`` 调用前先查缓存, 命中则跳过 SDE 仿真。
        返回值增加 cache_stats (dict 或 None)。
        """
        # 改进15 (六): 初始化 hash 缓存
        cache = None
        if use_hash_cache:
            from ..likelihood_cache import LikelihoodCache
            cache = LikelihoodCache(maxsize=cache_maxsize)

        def _cached_log_posterior(p):
            """带缓存的 log_posterior_v3 调用"""
            if cache is not None:
                cached = cache.get(p)
                if cached is not None:
                    return cached
            lp = self.log_posterior_v3(
                p, observed_data, initial_state, t_span, dt,
                include_eta_sub=include_eta_sub,
                include_eta_sn=include_eta_sn,
                use_neg_binom=use_neg_binom, neg_binom_k=neg_binom_k)
            if cache is not None and np.isfinite(lp):
                cache.put(p, lp)
            return lp

        params = np.array(init_params, dtype=float)
        current_lp = _cached_log_posterior(params)

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
                proposal_lp = _cached_log_posterior(proposal)

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

        cache_stats = cache.stats() if cache is not None else None
        return chain, accepted / max(n_iterations, 1), cache_stats

    # ==================================================================
    # 改进6: v3.0 9-房室模型 R0 计算
    # ==================================================================

    def compute_R0_v3(self, beta, eta_sub=DEFAULT_ETA_SUB, eta_sp=1.0,
                      eta_sn=DEFAULT_ETA_SN, rho_prog=DEFAULT_RHO_PROG,
                      r_sp=DEFAULT_R_SP, mu_sp=DEFAULT_MU_SP,
                      r_sn=DEFAULT_R_SN, mu_sn=DEFAULT_MU_SN,
                      p_sn2sp=DEFAULT_P_SN2SP):
        """9-房室 v3 模型基本再生数 R0（改进6）

        R0 = β·(η_sub/ρ_prog + η_sp/(r_sp+μ_sp) + η_sn/(r_sn+μ_sn+p_sn2sp))

        各项为三个传染性房室的「相对传染性 × 预期传染持续时间」:
          - I_sub: 持续 1/ρ_prog (亚临床→临床进展), 相对传染性 η_sub
          - I_sp:  持续 1/(r_sp+μ_sp) (涂阳恢复/死亡), 相对传染性 η_sp=1.0 (锚点)
          - I_sn:  持续 1/(r_sn+μ_sn+p_sn2sp) (涂阴恢复/死亡/转阳), 相对传染性 η_sn

        参考: Ragonnet et al. (2021) PLoS Comput Biol (涂阳未治疗持续时间 1.57 年)
        """
        rho_prog_safe = max(float(rho_prog), 1e-10)
        sp_out = max(float(r_sp) + float(mu_sp), 1e-10)
        sn_out = max(float(r_sn) + float(mu_sn) + float(p_sn2sp), 1e-10)
        r0 = float(beta) * (
            float(eta_sub) / rho_prog_safe
            + float(eta_sp) / sp_out
            + float(eta_sn) / sn_out
        )
        return r0
