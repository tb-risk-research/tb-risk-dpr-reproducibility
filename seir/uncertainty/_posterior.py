"""SEIR 参数不确定性量化 — v1 后验预测模块

从后验参数采样生成 SEIR 后验预测轨迹（v3.0 9-房室 ODE）。
拆分自原 seir/uncertainty.py（业务逻辑不变）。
"""

import numpy as np

try:
    from scipy.integrate import odeint
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class _PosteriorPredictiveMixin:
    """后验预测分布 v1: 9-房室 ODE 轨迹与置信带"""

    def generate_posterior_predictive(self, n_samples=100, horizon=365):
        """从后验参数采样生成SEIR后验预测轨迹（v3.0 9-房室）

        返回：
            dict: {'trajectories': [n_samples, n_steps, 9],
                   'ci_bands': {lower: [...], upper: [...]},
                   'r0_samples': [...],
                   'peak_I_samples': [...]}
        """
        if not self.inference_completed or not SCIPY_AVAILABLE:
            return {'error': 'MCMC尚未完成或scipy不可用'}

        try:
            # 9-房室 ODE（与 _log_likelihood 中的 seir_ode_v3 一致）
            def seir_ode_v3(y, t, b, rf, rr, sc, gam, eta_s, eta_n,
                            w_m, w_sub, b_exo, N):
                S, Lf, Ls, M, Isub, Isp, Isn, R, C = y
                rho_conv = 1.0 / 365.0; p_clin = 0.332; rho_prog = 0.05
                p_m = 0.5; rho_min = 0.5 / 365.0; p_sp = 0.55
                r_sp = 0.231 / 365.0; mu_sp = 0.389 / 365.0
                r_sn = 0.130 / 365.0; mu_sn = 0.025 / 365.0
                p_sn2sp = 0.1 / 365.0; beta_reinf = 0.4

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
                reg_m = w_m * M; reg_sub = w_sub * Isub
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
            I0 = max(1, min(int(N * 0.001), N))
            E0 = min(max(1, int(I0 * 30)), N - I0)
            Lf0 = int(E0 * 0.3)
            Ls0 = E0 - Lf0
            S0 = max(0.0, N - I0 - E0)
            initial = [float(S0), float(Lf0), float(Ls0), 0.0,
                       float(I0) * 0.5, float(I0) * 0.3, float(I0) * 0.2, 0.0, 0.0]
            t = np.linspace(0, horizon, horizon)

            trajectories = []
            r0_samples = []
            peak_I_samples = []

            n_posterior = len(self.posterior_samples['beta'])
            indices = self.rng.choice(n_posterior, size=n_samples)
            beta_vals = self.posterior_samples['beta'][indices]
            rho_fast_vals = self.posterior_samples['rho_fast'][indices]
            rho_react_vals = self.posterior_samples['rho_react'][indices]
            sigma_clear_vals = self.posterior_samples['sigma_clear'][indices]
            gamma_vals = self.posterior_samples['gamma'][indices]
            eta_sub_vals = self.posterior_samples['eta_sub'][indices]
            s_factor_vals = self.posterior_samples['social_factor'][indices]
            h_factor_vals = self.posterior_samples['household_factor'][indices]
            # v3 新参数（如果后验中存在）
            omega_m_vals = self.posterior_samples.get('omega_reg_m',
                np.full(n_posterior, 1.0/365.0))[indices]
            omega_sub_vals = self.posterior_samples.get('omega_reg_sub',
                np.full(n_posterior, 1.0/365.0))[indices]
            beta_exo_vals = self.posterior_samples.get('beta_exo',
                np.full(n_posterior, 0.33))[indices]
            eta_sn_vals = self.posterior_samples.get('eta_sn',
                np.full(n_posterior, 0.35))[indices]

            for i in range(n_samples):
                eff_beta = beta_vals[i] * (0.33 * s_factor_vals[i] + 0.67 * h_factor_vals[i])
                sol = odeint(seir_ode_v3, initial, t,
                            args=(eff_beta, rho_fast_vals[i], rho_react_vals[i],
                                  sigma_clear_vals[i], gamma_vals[i],
                                  eta_sub_vals[i], eta_sn_vals[i],
                                  omega_m_vals[i], omega_sub_vals[i],
                                  beta_exo_vals[i], N))
                trajectories.append(sol)
                r0_samples.append(float(eff_beta / gamma_vals[i]))
                # 活动性TB = I_sub + I_sp + I_sn (列 4, 5, 6)
                peak_I_samples.append(float((sol[:, 4] + sol[:, 5] + sol[:, 6]).max()))

            trajectories = np.array(trajectories)

            # 活动性TB = I_sub + I_sp + I_sn
            active_I = trajectories[:, :, 4] + trajectories[:, :, 5] + trajectories[:, :, 6]
            ci_lower_i = np.percentile(active_I, 2.5, axis=0)
            ci_upper_i = np.percentile(active_I, 97.5, axis=0)

            return {
                'trajectories': trajectories,
                'ci_bands': {
                    'lower': ci_lower_i.tolist(),
                    'upper': ci_upper_i.tolist(),
                    'time': t.tolist(),
                },
                'r0_samples': r0_samples,
                'peak_I_samples': peak_I_samples,
                'r0_mean': float(np.mean(r0_samples)),
                'r0_ci': [float(np.percentile(r0_samples, 2.5)),
                          float(np.percentile(r0_samples, 97.5))],
                'peak_I_mean': float(np.mean(peak_I_samples)),
                'peak_I_ci': [float(np.percentile(peak_I_samples, 2.5)),
                              float(np.percentile(peak_I_samples, 97.5))],
            }
        except Exception as e:
            return {'error': str(e)}
