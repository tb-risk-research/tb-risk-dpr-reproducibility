"""v1 推断引擎：4-房室 SEIR 模型自适应 Metropolis-Hastings MCMC。

包含方法：
  log_prior / log_likelihood / log_posterior
  metropolis_hastings / _run_single_chain
  _init_mcmc_state / _finalize_mcmc_results
"""

import math
import hashlib
import warnings

import numpy as np

from ..diagnostics import compute_r_hat, compute_ess
from ._base import BaseBayesianInference


class BayesianInferenceV1(BaseBayesianInference):
    """v1 4-房室 SEIR 模型贝叶斯推断（Particle-MCMC + 自适应 Metropolis-Hastings）。

    .. deprecated::
        v1 已弃用，建议使用 BayesianInferenceV4（HMC + 270 维年龄×HIV×耐药分层）。
        v1 仅保留用于向后兼容和单元测试参照。

    参数空间：β (传播率), σ (潜伏→发病率), γ (恢复率)。
    """

    def __init__(self, seir_model=None, seed=42, **kwargs):
        warnings.warn(
            "BayesianInferenceV1 已弃用，建议使用 BayesianInferenceV4。"
            "v1 仅保留用于向后兼容。",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(seir_model=seir_model, seed=seed)

    def log_prior(self, beta, sigma, gamma):
        """结核病专属先验分布（v2.0 旧版4-房室，向后兼容）

        TB 生物学时间尺度（Vynnycky & Fine, 1997; Ragonnet et al., 2021）：
          σ: 潜伏→发病率 10⁻⁴~10⁻³/day → Gamma(2, 0.001) 均值≈0.002/day
          γ: 恢复率 0.01-0.1/day → Gamma(2, 0.025) 均值≈0.05/day
          β: 传播率 Beta(2, 8) 均值≈0.2 → R0=β/γ≈4 (TB 典型范围 1-10)
        """
        np = self._np
        try:
            from scipy import stats
            _has_scipy_stats = True
        except ImportError:
            _has_scipy_stats = False

        # TB 专属参数范围
        if not (0.001 < beta < 0.95): return -np.inf
        if not (0.0001 < sigma < 0.01): return -np.inf        # 100天~27年潜伏期
        if not (0.005 < gamma < 0.2): return -np.inf           # 5天~200天传染期

        if _has_scipy_stats:
            lp = stats.beta.logpdf(beta, 2, 8)                 # 均值≈0.2
            lp += stats.gamma.logpdf(sigma, 2, scale=0.001)     # 均值≈0.002/day
            lp += stats.gamma.logpdf(gamma, 2, scale=0.025)     # 均值≈0.05/day
        else:
            lp = (2 - 1) * np.log(beta + 1e-10) + (8 - 1) * np.log(1 - beta + 1e-10)
            lp += (2 - 1) * np.log(sigma + 1e-10) - sigma / 0.001
            lp += (2 - 1) * np.log(gamma + 1e-10) - gamma / 0.025
        return lp

    def log_likelihood(self, params, observed_data, initial_state, t_span, dt):
        """对数似然函数（确定化版本，v2.0 旧版4-房室）

        **detailed balance 保护**：
        固定随机种子，确保同一参数集的似然值为确定函数。
        在 MCMC 框架下，目标分布必须是参数的确定函数，否则违反 detailed balance。
        采用 seed-hashing 方法：将参数哈希为种子，使似然值对参数确定。

        参考文献：
          Andrieu C, Doucet A, Holenstein R. (2010) Particle Markov chain Monte Carlo methods.
          JRSS-B 72(3):269-342.
        """
        np = self._np
        beta, sigma, gamma = params
        # 固定随机种子：确保同一参数集的似然值为确定函数
        param_str = f"{beta:.17g},{sigma:.17g},{gamma:.17g}"
        param_hash = int(hashlib.md5(param_str.encode()).hexdigest(), 16)
        old_state = self.seir_model.rng.get_state()
        try:
            self.seir_model.rng.seed(abs(param_hash) % (2**31 - 1))
            _, traj, _ = self.seir_model.simulate_sde(
                initial_state, t_span, dt, beta, sigma, gamma)
        finally:
            self.seir_model.rng.seed(0)  # 重置为确定状态
            self.seir_model.rng.set_state(old_state)  # 恢复原始状态

        sim_I = traj[:, 2]
        obs_I, obs_times = observed_data
        interpolated = np.interp(obs_times, np.linspace(t_span[0], t_span[1], len(sim_I)), sim_I)
        std_obs = np.std(obs_I) if len(obs_I) > 0 else 0.0
        if not np.isfinite(std_obs):
            std_obs = 0.0
        obs_noise = max(std_obs * 0.2, 0.5)
        residuals = obs_I - interpolated
        ll = -0.5 * np.sum((residuals / obs_noise) ** 2) - \
             len(obs_I) * np.log(obs_noise * np.sqrt(2 * np.pi))
        if np.isnan(ll) or np.isinf(ll):
            return -np.inf  # 使用 -inf 确保严格拒绝无效提议
        return float(ll)

    def log_posterior(self, params, observed_data, initial_state, t_span, dt):
        lp = self.log_prior(*params)
        if np.isneginf(lp): return -np.inf
        return lp + self.log_likelihood(params, observed_data, initial_state, t_span, dt)

    def _init_mcmc_state(self, observed_data, initial_state, t_span, dt,
                         init_params, proposal_cov):
        """初始化 MCMC 状态：参数、提案协方差、Cholesky 分解"""
        if init_params is None:
            init_params = np.array([0.2, 0.002, 0.05])        # TB时间尺度: β, σ, γ
        if proposal_cov is None:
            proposal_cov = np.diag([0.02, 0.0002, 0.005]) ** 2  # TB尺度提案方差
        params = np.array(init_params, dtype=float)
        current_lp = self.log_posterior(params, observed_data, initial_state, t_span, dt)

        try:
            L_chol = np.linalg.cholesky(proposal_cov)
        except np.linalg.LinAlgError:
            L_chol = np.sqrt(np.diag(proposal_cov)) * np.eye(3)

        return params, current_lp, L_chol, proposal_cov

    def _finalize_mcmc_results(self, chain, log_probs, accepted, n_iterations,
                                 burn_in, adapt, adapt_interval, adapt_start,
                                 n_adapt_updates, proposal_cov):
        """组装 MCMC 后验样本、轨迹和收敛诊断"""
        acceptance_rate = accepted / n_iterations
        self.posterior_samples = {
            'beta': chain[burn_in:, 0],
            'sigma': chain[burn_in:, 1],
            'gamma': chain[burn_in:, 2],
        }
        self.traces = {'chain': chain, 'log_probs': log_probs}
        self.convergence_diagnostics = {
            'acceptance_rate': acceptance_rate,
            'burn_in': burn_in, 'n_iterations': n_iterations,
            'beta_mean': float(np.mean(chain[burn_in:, 0])),
            'sigma_mean': float(np.mean(chain[burn_in:, 1])),
            'gamma_mean': float(np.mean(chain[burn_in:, 2])),
            'adaptive_mcmc': {
                'enabled': bool(adapt),
                'adapt_interval': adapt_interval,
                'adapt_start': adapt_start,
                'n_updates': n_adapt_updates,
                'final_proposal_cov': proposal_cov.tolist() if hasattr(proposal_cov, 'tolist') else proposal_cov,
                'reference': 'Haario et al., 2001, Bernoulli 7(2):223-242',
            },
        }
        return self.posterior_samples

    def metropolis_hastings(self, observed_data, initial_state, t_span, dt,
                            n_iterations=None, burn_in=None,
                            n_chains=None, proposal_cov=None, init_params=None,
                            adapt=True, adapt_interval=100, adapt_start=50):
        """自适应 Metropolis-Hastings MCMC（v2.0：多链 + 收敛诊断）

        参数：
            observed_data: (obs_I, obs_times) 观测序列
            initial_state: SEIR 初始状态
            t_span, dt: 时间跨度与步长
            n_iterations: 总迭代数（默认 10000）
            burn_in: 预热期（默认 2000）
            n_chains: 并行链数（默认 4，≥2 时启用 R-hat）
            proposal_cov: 初始提案协方差
            init_params: 初始参数 [beta, sigma, gamma]
            adapt: 是否启用 Haario 2001 自适应提案
            adapt_interval: 自适应更新间隔
            adapt_start: 自适应启动所需最小样本数

        参考文献：
            Haario H, Saksman E, Tamminen J. (2001) An adaptive Metropolis algorithm.
            Bernoulli 7(2):223-242.
            Vehtari A et al. (2021) Rank-normalization, folding, and localization:
            An improved R-hat for assessing convergence of MCMC. Bayesian Anal 16(2).
        """
        # 使用默认值（与 uncertainty.py 一致）
        n_iterations = n_iterations or self.DEFAULT_N_ITERATIONS
        burn_in = burn_in or self.DEFAULT_N_BURNIN
        n_chains = n_chains or self.DEFAULT_N_CHAINS

        # 运行多条独立链（使用确定性种子派生，不依赖numpy内部状态结构）
        all_chains = []
        all_acceptance_rates = []
        for chain_idx in range(n_chains):
            chain_seed = self._derive_chain_seed(chain_idx)
            chain_rng = np.random.RandomState(chain_seed)

            chain, acc_rate = self._run_single_chain(
                observed_data, initial_state, t_span, dt,
                n_iterations, burn_in, proposal_cov, init_params,
                adapt, adapt_interval, adapt_start, chain_rng, chain_idx)
            all_chains.append(chain)
            all_acceptance_rates.append(acc_rate)

        # 合并后验样本（burn-in 后）
        merged_chain = np.concatenate(
            [c[burn_in:] for c in all_chains], axis=0)

        self.posterior_samples = {
            'beta': merged_chain[:, 0],
            'sigma': merged_chain[:, 1],
            'gamma': merged_chain[:, 2],
        }
        self.traces = {
            'chains': all_chains,
            'all_acceptance_rates': all_acceptance_rates,
        }

        # 收敛诊断
        r_hat = compute_r_hat(all_chains, burn_in)
        ess = compute_ess(merged_chain)

        self.convergence_diagnostics = {
            'n_chains': n_chains,
            'n_iterations': n_iterations,
            'burn_in': burn_in,
            'acceptance_rate': float(np.mean(all_acceptance_rates)),
            'acceptance_rates_per_chain': all_acceptance_rates,
            'beta_mean': float(np.mean(merged_chain[:, 0])),
            'sigma_mean': float(np.mean(merged_chain[:, 1])),
            'gamma_mean': float(np.mean(merged_chain[:, 2])),
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
        return self.posterior_samples

    def _run_single_chain(self, observed_data, initial_state, t_span, dt,
                          n_iterations, burn_in, proposal_cov, init_params,
                          adapt, adapt_interval, adapt_start, chain_rng, chain_idx):
        """运行单条 MCMC 链（内部方法）。"""
        params, current_lp, L_chol, proposal_cov = self._init_mcmc_state(
            observed_data, initial_state, t_span, dt, init_params, proposal_cov)
        chain = np.zeros((n_iterations, 3))
        log_probs = np.zeros(n_iterations)
        accepted = 0
        d = 3
        s_d = (2.4 ** 2) / d
        eps_reg = 1e-8

        for i in range(n_iterations):
            proposal = params + L_chol @ chain_rng.normal(0, 1, 3)
            proposal_lp = self.log_posterior(proposal, observed_data,
                                              initial_state, t_span, dt)
            if np.isfinite(proposal_lp):
                log_alpha = proposal_lp - current_lp
                u = chain_rng.uniform()
                if u > 0 and math.log(u) < min(log_alpha, 0):
                    params, current_lp = proposal, proposal_lp
                    accepted += 1
            chain[i] = params
            log_probs[i] = current_lp

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
