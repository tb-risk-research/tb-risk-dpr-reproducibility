"""SEIR 参数不确定性量化 — MCMC 采样模块

对数似然（v3.0 9-房室 ODE + 负二项）与 Metropolis-Hastings 自适应 MCMC 采样。
拆分自原 seir/uncertainty.py（业务逻辑不变）。
"""

import numpy as np

try:
    from scipy.integrate import odeint
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import scipy.stats as sps
    SCIPY_STATS_AVAILABLE = True
except ImportError:
    SCIPY_STATS_AVAILABLE = False


class _MCMCMixin:
    """MCMC 采样: 对数似然 + 自适应 Metropolis-Hastings"""

    def _log_likelihood(self, params, observed_data):
        """计算观测数据负二项对数似然（v3.0 9-房室 ODE）

        负二项分布是流行病学计数数据的标准似然函数，
        其过度离散参数 k 自然捕获报告延迟和聚集性。
        参考：Lloyd-Smith et al. (2005) "Superspreading and the
        effect of individual variation on disease emergence", Nature 438: 355-359.

        ODE 积分失败时返回 -inf，拒绝该提议，保护 detailed balance。
        """
        beta = params['beta']
        rho_fast = params.get('rho_fast', 0.1 / 365.0)
        rho_react = params.get('rho_react', 0.004 / 365.0)
        sigma_clear = params.get('sigma_clear', 2.0 / 365.0)
        gamma = params['gamma']
        eta_sub = params.get('eta_sub', 1.0)
        omega_reg_m = params.get('omega_reg_m', 1.0 / 365.0)
        omega_reg_sub = params.get('omega_reg_sub', 1.0 / 365.0)
        beta_exo = params.get('beta_exo', 0.33)
        eta_sn = params.get('eta_sn', 0.35)
        k = params.get('k', 0.2)
        s_factor = params.get('social_factor', 1.0)
        h_factor = params.get('household_factor', 1.0)

        observed_infections = observed_data.get('new_infections', [])
        observed_prevalence = observed_data.get('active_cases', [])
        n_obs = len(observed_infections)
        if n_obs == 0:
            return 0.0
        if len(observed_prevalence) < n_obs:
            observed_prevalence = list(observed_prevalence) + [0] * (n_obs - len(observed_prevalence))

        if not SCIPY_AVAILABLE:
            import logging
            logging.getLogger('tb_risk').warning(
                "scipy 不可用，_log_likelihood 回退为 Poisson 近似")
            ll = 0.0
            N = self.population
            for obs_inf in observed_infections:
                noise = max(0.01, obs_inf * 0.15)
                ll += -0.5 * np.log(2 * np.pi * noise**2) - \
                      (obs_inf - beta * N * 0.01)**2 / (2 * noise**2)
            return float(ll)

        ll = 0.0

        try:
            # 9-房室 ODE（与 stochastic.py deterministic_rhs_v3 一致）
            # 状态: [S, L_fast, L_slow, M, I_sub, I_sp, I_sn, R, C]
            def seir_ode_v3(y, t, b, rf, rr, sc, gam, eta_s, eta_n,
                            w_m, w_sub, b_exo, N):
                S, Lf, Ls, M, Isub, Isp, Isn, R, C = y
                # 固定参数
                rho_conv = 1.0 / 365.0
                p_clin = 0.332
                rho_prog = 0.05
                p_m = 0.5
                rho_min = 0.5 / 365.0
                p_sp = 0.55
                r_sp = 0.231 / 365.0; mu_sp = 0.389 / 365.0
                r_sn = 0.130 / 365.0; mu_sn = 0.025 / 365.0
                p_sn2sp = 0.1 / 365.0
                beta_reinf = 0.4

                lam = b * (eta_s * Isub + Isp + eta_n * Isn) / max(N, 1e-10)

                infection = lam * S
                reinfection_C = beta_reinf * lam * C
                reinfection_R = beta_reinf * lam * R
                exo_reinfection = b_exo * lam * Ls

                fast_to_min = rf * p_m * Lf
                fast_to_sub = rf * (1 - p_m) * (1 - p_clin) * Lf
                fast_to_sp = rf * (1 - p_m) * p_clin * Lf
                fast_to_slow = rho_conv * Lf
                clear_fast = sc * Lf

                slow_to_min = rr * p_m * Ls
                slow_to_sub = rr * (1 - p_m) * (1 - p_clin) * Ls
                slow_to_sp = rr * (1 - p_m) * p_clin * Ls
                clear_slow = sc * Ls

                min_to_sub = rho_min * M
                reg_m = w_m * M
                reg_sub = w_sub * Isub

                sub_to_sp = rho_prog * p_sp * Isub
                sub_to_sn = rho_prog * (1 - p_sp) * Isub
                sn_to_sp = p_sn2sp * Isn
                sp_out = (r_sp + mu_sp) * Isp
                sn_out = (r_sn + mu_sn) * Isn

                dS = -infection
                dLf = (infection + reinfection_C + reinfection_R + exo_reinfection
                       - fast_to_min - fast_to_sub - fast_to_sp - fast_to_slow - clear_fast)
                dLs = (fast_to_slow + reg_m
                       - slow_to_min - slow_to_sub - slow_to_sp - clear_slow - exo_reinfection)
                dM = (fast_to_min + slow_to_min + reg_sub - min_to_sub - reg_m)
                dIsub = (fast_to_sub + slow_to_sub + min_to_sub
                         - sub_to_sp - sub_to_sn - reg_sub)
                dIsp = (fast_to_sp + slow_to_sp + sub_to_sp + sn_to_sp - sp_out)
                dIsn = (sub_to_sn - sn_to_sp - sn_out)
                dR = (sp_out + sn_out - reinfection_R)
                dC = (clear_fast + clear_slow - reinfection_C)
                return [dS, dLf, dLs, dM, dIsub, dIsp, dIsn, dR, dC]

            N = self.population
            I0 = max(1, min(observed_data.get('initial_infected', int(N * 0.001)), N))
            E0 = min(max(1, int(I0 * 30)), N - I0)
            Lf0 = int(E0 * 0.3)
            Ls0 = E0 - Lf0
            S0 = max(0.0, N - I0 - E0)
            # 9-房室初始状态: [S, Lf, Ls, M, I_sub, I_sp, I_sn, R, C]
            initial = [float(S0), float(Lf0), float(Ls0), 0.0,
                       float(I0) * 0.5, float(I0) * 0.3, float(I0) * 0.2, 0.0, 0.0]
            t = np.arange(n_obs)

            w_s, w_h = 0.33, 0.67
            effective_beta = beta * (w_s * s_factor + w_h * h_factor)
            sol = odeint(seir_ode_v3, initial, t,
                        args=(effective_beta, rho_fast, rho_react,
                              sigma_clear, gamma, eta_sub, eta_sn,
                              omega_reg_m, omega_reg_sub, beta_exo, N))

            if np.any(np.isnan(sol)):
                return -np.inf

            if not SCIPY_STATS_AVAILABLE:
                return -np.inf

            # 活动性TB = I_sub + I_sp + I_sn (列 4, 5, 6)
            active_I = sol[:, 4] + sol[:, 5] + sol[:, 6]

            for i, obs_inf in enumerate(observed_infections):
                obs_prev = observed_prevalence[i]
                obs_inf = max(0, int(round(obs_inf)))
                obs_prev = max(0, int(round(obs_prev)))
                pred_inf = max(0.0, active_I[i] - active_I[max(0, i-1)] +
                               sol[i, 7] - sol[max(0, i-1), 7])
                pred_prev = max(0.0, active_I[i])

                lambda_inf = max(pred_inf, 1e-6)
                if lambda_inf > 0 and k > 0:
                    p_inf = k / (k + lambda_inf)
                    ll += sps.nbinom.logpmf(int(obs_inf), n=k, p=p_inf)
                else:
                    ll += sps.poisson.logpmf(int(obs_inf), lambda_inf)

                lambda_prev = max(pred_prev, 1e-6)
                if lambda_prev > 0 and k > 0:
                    p_prev = k / (k + lambda_prev)
                    ll += sps.nbinom.logpmf(int(obs_prev), n=k, p=p_prev)
                else:
                    ll += sps.poisson.logpmf(int(obs_prev), lambda_prev)

        except (ValueError, RuntimeError, ZeroDivisionError,
                NameError, OverflowError, IndexError, TypeError):
            return -np.inf

        return float(ll)

    def run_mcmc(self, observed_data, n_iterations=10000, n_burnin=2000,
                 n_chains=4, adapt_step=True, verbose=False):
        """运行MCMC采样（Metropolis-Hastings with adaptive covariance）

        参数：
            observed_data: 观测数据 dict
            n_iterations: 迭代次数
            n_burnin: 预热期
            n_chains: 链数
            adapt_step: 自适应步长
            verbose: 是否打印信息

        返回：
            dict: 后验采样结果
        """
        param_names = list(self.SEIR_PARAM_PRIORS.keys())
        n_params = len(param_names)

        init_values = {
            'beta': 0.2,
            'rho_fast': 0.1 / 365.0,
            'rho_react': 0.004 / 365.0,
            'sigma_clear': 2.0 / 365.0,
            'gamma': 0.05,
            'eta_sub': 1.0,
            'omega_reg_m': 1.0 / 365.0,
            'omega_reg_sub': 1.0 / 365.0,
            'beta_exo': 0.33,
            'eta_sn': 0.35,
            'social_factor': 1.5, 'household_factor': 2.0,
            'k': 0.2,
        }
        init_vec = np.array([init_values.get(p, 1.0) for p in param_names])

        proposal_cov = np.eye(n_params) * 0.001
        # TB 时间尺度下界（per day）
        lower_bounds = np.array([
            0.001, 0.00001, 1e-7, 0.0001, 0.005, 0.01,
            0.0001, 0.0001, 0.01, 0.01,
            0.01, 0.01, 1e-6
        ][:n_params])

        all_chains = []
        all_accept_rates = []
        global_best_lp = -np.inf

        for chain_i in range(n_chains):
            # 每链使用派生独立种子，避免链间共享 RNG 状态
            chain_seed = self.random_seed + chain_i * 10000
            chain_rng = np.random.RandomState(chain_seed)
            current = init_vec + chain_rng.randn(n_params) * 0.01
            current = np.maximum(current, lower_bounds)
            current_dict = {n: current[i] for i, n in enumerate(param_names)}
            current_lp = self._log_prior(current_dict) + \
                         self._log_likelihood(current_dict, observed_data)

            chain_samples = np.zeros((n_iterations, n_params))
            n_accept = 0

            adapt_start = 500
            adapt_interval = 200
            for it in range(n_iterations):
                if adapt_step and it > adapt_start:
                    # Haario 自适应 Metropolis 算法（Haario et al., Bernoulli 2001）
                    # 使用历史样本的经验协方差构建提案分布
                    # s_d = 2.38^2 / d 是 d 维最优缩放因子（Gelman et al. 1996）
                    if it > adapt_start + adapt_interval:
                        hist = chain_samples[max(0, it - adapt_interval):it]
                        emp_cov = np.cov(hist, rowvar=False)
                        if emp_cov.ndim == 0:
                            emp_cov = emp_cov.reshape(1, 1)
                        s_d = 2.38 ** 2 / n_params
                        proposal_cov = s_d * (emp_cov + 1e-8 * np.eye(n_params))
                    proposal = chain_rng.multivariate_normal(
                        current, proposal_cov)
                else:
                    proposal = chain_rng.multivariate_normal(
                        current, proposal_cov * 0.1)

                if np.any(proposal < lower_bounds):
                    proposal_lp = -np.inf
                else:
                    proposal_dict = {n: proposal[i]
                                     for i, n in enumerate(param_names)}
                    proposal_lp = self._log_prior(proposal_dict) + \
                                  self._log_likelihood(proposal_dict, observed_data)

                if np.isfinite(proposal_lp):
                    log_accept_ratio = proposal_lp - current_lp
                    u = chain_rng.random()
                    if u > 0 and np.log(u) < log_accept_ratio:
                        current = proposal
                        current_lp = proposal_lp
                        n_accept += 1

                chain_samples[it] = current

            accept_rate = n_accept / n_iterations
            all_accept_rates.append(accept_rate)
            all_chains.append(chain_samples)

            if current_lp > global_best_lp:
                global_best_lp = current_lp

            if verbose:
                pass

        posterior = {}
        for i, name in enumerate(param_names):
            all_chain_vals = np.concatenate(
                [c[n_burnin:, i] for c in all_chains])
            posterior[name] = all_chain_vals

        self.posterior_samples = posterior

        summary = {}
        for name, samples in posterior.items():
            summary[name] = {
                'mean': float(np.mean(samples)),
                'std': float(np.std(samples)),
                'median': float(np.median(samples)),
                'ci_2_5': float(np.percentile(samples, 2.5)),
                'ci_97_5': float(np.percentile(samples, 97.5)),
                'prior_type': self.SEIR_PARAM_PRIORS[name]['dist'],
                'unit': self.SEIR_PARAM_PRIORS[name]['unit'],
            }
        self.posterior_summary = summary

        r0_samples = posterior['beta'] / (posterior['gamma'] + 1e-10)
        self.r0_posterior = {
            'mean': float(np.mean(r0_samples)),
            'median': float(np.median(r0_samples)),
            'ci_2_5': float(np.percentile(r0_samples, 2.5)),
            'ci_97_5': float(np.percentile(r0_samples, 97.5)),
            'p_r0_gt_1': float(np.mean(r0_samples > 1.0)),
            'p_r0_gt_1_5': float(np.mean(r0_samples > 1.5)),
            'samples': r0_samples,
        }

        self.mcmc_diagnostics = {
            'n_iterations': n_iterations,
            'n_burnin': n_burnin,
            'n_chains': n_chains,
            'accept_rates': [float(r) for r in all_accept_rates],
            'mean_accept_rate': float(np.mean(all_accept_rates)),
            'r_hat': self._compute_r_hat(all_chains, n_burnin),
            'ess': self._compute_ess(all_chains, n_burnin),
        }

        self.inference_completed = True

        return {
            'summary': self.posterior_summary,
            'r0_posterior': self.r0_posterior,
            'diagnostics': self.mcmc_diagnostics,
        }
