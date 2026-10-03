"""SEIR 参数不确定性量化 — v3 后验预测模块

v3.0 9-房室后验预测分布、R0 计算（inline）与 ODE 轨迹回退。
拆分自原 seir/uncertainty.py（业务逻辑不变）。
"""

import logging
import numpy as np

LOGGER = logging.getLogger("tb_risk.seir.uncertainty")

try:
    from scipy.integrate import odeint
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class _PosteriorV3Mixin:
    """后验预测分布 v3: 9-房室 SDE/ODE 轨迹、R0、置信带"""

    def _compute_R0_v3_inline(self, beta, eta_sub, eta_sn):
        """v3 9-房室 R0 (改进6): β·(η_sub/ρ_prog + 1/(r_sp+μ_sp) + η_sn/(r_sn+μ_sn+p_sn2sp))"""
        f = self._V3_FIXED
        rho_prog = max(f['rho_prog'], 1e-10)
        sp_out = max(f['r_sp'] + f['mu_sp'], 1e-10)
        sn_out = max(f['r_sn'] + f['mu_sn'] + f['p_sn2sp'], 1e-10)
        return float(beta) * (
            float(eta_sub) / rho_prog
            + 1.0 / sp_out
            + float(eta_sn) / sn_out
        )

    def generate_posterior_predictive_v3(self, initial_state, t_span, dt,
                                         n_samples=100, use_sde=True):
        """v3.0 9-房室后验预测分布（改进6）

        从 run_mcmc_v3 产生的后验样本中采样参数，运行 v3 SDE（或 ODE 回退）
        生成轨迹集合，输出活动性 TB (I_sub+I_sp+I_sn) 的 95% 置信带与 R0 分布。

        参数:
          initial_state: 9 维初始状态 [S, Lf, Ls, M, Isub, Isp, Isn, R, C]
          t_span: (t_start, t_end)
          dt: 积分步长（天）
          n_samples: 后验采样数
          use_sde: True 用 simulate_sde_v3（CLE 随机），False 用 ODE（确定性）

        返回:
          dict: {'trajectories', 'ci_bands', 'r0_samples', 'peak_I_samples', ...}
        """
        if not self.posterior_samples:
            return {'error': '后验样本不存在，请先运行 run_mcmc_v3'}
        ps = self.posterior_samples
        if 'beta' not in ps:
            return {'error': '后验样本缺少 beta 键'}

        n_posterior = len(ps['beta'])
        if n_posterior == 0:
            return {'error': '后验样本为空'}
        indices = self.rng.choice(n_posterior, size=min(n_samples, n_posterior))

        # 采样后验参数（缺失键用默认值填充）
        def _sample(key, default):
            if key in ps:
                return ps[key][indices]
            return np.full(len(indices), default)

        beta_vals = _sample('beta', 0.2)
        rho_fast_vals = _sample('rho_fast', 0.1 / 365.0)
        rho_react_vals = _sample('rho_react', 0.004 / 365.0)
        sigma_clear_vals = _sample('sigma_clear', 2.0 / 365.0)
        gamma_vals = _sample('gamma', 0.05)
        omega_m_vals = _sample('omega_reg_m', 1.0 / 365.0)
        omega_sub_vals = _sample('omega_reg_sub', 1.0 / 365.0)
        beta_exo_vals = _sample('beta_exo', 0.33)
        eta_sub_vals = _sample('eta_sub', self._V3_DEFAULT_ETA_SUB)
        eta_sn_vals = _sample('eta_sn', self._V3_DEFAULT_ETA_SN)

        N_total = max(float(np.sum(initial_state)), 1e-10)

        # SDE 可用性检查
        sde_available = (use_sde and self.stochastic_seir is not None
                         and hasattr(self.stochastic_seir, 'simulate_sde_v3'))

        trajectories = []
        r0_samples = []
        peak_I_samples = []
        n_actual = len(indices)

        for i in range(n_actual):
            r0_samples.append(self._compute_R0_v3_inline(
                beta_vals[i], eta_sub_vals[i], eta_sn_vals[i]))

            if sde_available:
                try:
                    _times, traj, _sto = self.stochastic_seir.simulate_sde_v3(
                        initial_state, t_span, dt,
                        beta=float(beta_vals[i]),
                        rho_fast=float(rho_fast_vals[i]),
                        rho_react=float(rho_react_vals[i]),
                        sigma_clear=float(sigma_clear_vals[i]),
                        gamma=float(gamma_vals[i]),
                        eta_sub=float(eta_sub_vals[i]),
                        eta_sn=float(eta_sn_vals[i]),
                        omega_reg_m=float(omega_m_vals[i]),
                        omega_reg_sub=float(omega_sub_vals[i]),
                        beta_exo=float(beta_exo_vals[i]),
                    )
                    traj = np.asarray(traj)
                except Exception as e:
                    LOGGER.debug("v3 SDE轨迹生成失败，回退到ODE: %s", e)
                    traj = self._v3_ode_trajectory(
                        initial_state, t_span, dt,
                        beta_vals[i], rho_fast_vals[i], rho_react_vals[i],
                        sigma_clear_vals[i], gamma_vals[i],
                        eta_sub_vals[i], eta_sn_vals[i],
                        omega_m_vals[i], omega_sub_vals[i],
                        beta_exo_vals[i], N_total)
            else:
                traj = self._v3_ode_trajectory(
                    initial_state, t_span, dt,
                    beta_vals[i], rho_fast_vals[i], rho_react_vals[i],
                    sigma_clear_vals[i], gamma_vals[i],
                    eta_sub_vals[i], eta_sn_vals[i],
                    omega_m_vals[i], omega_sub_vals[i],
                    beta_exo_vals[i], N_total)

            trajectories.append(traj)
            # 活动性 TB = I_sub + I_sp + I_sn (列 4, 5, 6)
            active_I = traj[:, 4] + traj[:, 5] + traj[:, 6]
            peak_I_samples.append(float(np.max(active_I)))

        trajectories = np.array(trajectories)
        active_I_all = trajectories[:, :, 4] + trajectories[:, :, 5] + trajectories[:, :, 6]
        ci_lower = np.percentile(active_I_all, 2.5, axis=0)
        ci_upper = np.percentile(active_I_all, 97.5, axis=0)
        median_I = np.percentile(active_I_all, 50, axis=0)

        n_steps = trajectories.shape[1]
        times = np.linspace(t_span[0], t_span[1], n_steps)

        return {
            'trajectories': trajectories,
            'active_I': active_I_all,
            'ci_bands': {
                'lower': ci_lower.tolist(),
                'upper': ci_upper.tolist(),
                'median': median_I.tolist(),
                'time': times.tolist(),
            },
            'r0_samples': r0_samples,
            'r0_mean': float(np.mean(r0_samples)),
            'r0_ci': [float(np.percentile(r0_samples, 2.5)),
                      float(np.percentile(r0_samples, 97.5))],
            'r0_median': float(np.percentile(r0_samples, 50)),
            'p_r0_gt_1': float(np.mean(np.array(r0_samples) > 1.0)),
            'peak_I_samples': peak_I_samples,
            'peak_I_mean': float(np.mean(peak_I_samples)),
            'peak_I_ci': [float(np.percentile(peak_I_samples, 2.5)),
                          float(np.percentile(peak_I_samples, 97.5))],
            'n_samples': n_actual,
            'method': 'SDE' if sde_available else 'ODE',
        }

    def _v3_ode_trajectory(self, initial_state, t_span, dt,
                           beta, rho_fast, rho_react, sigma_clear, gamma,
                           eta_sub, eta_sn, omega_m, omega_sub, beta_exo, N):
        """v3 9-房室 ODE 轨迹（SDE 不可用时的回退）"""
        f = self._V3_FIXED
        n_steps = min(int((t_span[1] - t_span[0]) / dt) + 1, 100000)
        t = np.linspace(t_span[0], t_span[1], n_steps)

        def seir_ode_v3(y, t):
            S, Lf, Ls, M, Isub, Isp, Isn, R, C = y
            lam = beta * (eta_sub * Isub + Isp + eta_sn * Isn) / max(N, 1e-10)
            infection = lam * S
            reinfection_C = f['beta_reinf'] * lam * C
            reinfection_R = f['beta_reinf'] * lam * R
            exo_reinfection = beta_exo * lam * Ls
            fast_to_min = rho_fast * f['p_m'] * Lf
            fast_to_sub = rho_fast * (1 - f['p_m']) * (1 - f['p_clin']) * Lf
            fast_to_sp = rho_fast * (1 - f['p_m']) * f['p_clin'] * Lf
            fast_to_slow = f['rho_conv'] * Lf
            clear_fast = sigma_clear * Lf
            slow_to_min = rho_react * f['p_m'] * Ls
            slow_to_sub = rho_react * (1 - f['p_m']) * (1 - f['p_clin']) * Ls
            slow_to_sp = rho_react * (1 - f['p_m']) * f['p_clin'] * Ls
            clear_slow = sigma_clear * Ls
            min_to_sub = f['rho_min'] * M
            reg_m = omega_m * M
            reg_sub = omega_sub * Isub
            sub_to_sp = f['rho_prog'] * f['p_sp'] * Isub
            sub_to_sn = f['rho_prog'] * (1 - f['p_sp']) * Isub
            sn_to_sp = f['p_sn2sp'] * Isn
            sp_out = (f['r_sp'] + f['mu_sp']) * Isp
            sn_out = (f['r_sn'] + f['mu_sn']) * Isn
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

        if SCIPY_AVAILABLE:
            sol = odeint(seir_ode_v3, initial_state, t)
            return sol
        # Euler 回退
        y = np.array(initial_state, dtype=float)
        traj = np.zeros((n_steps, 9))
        traj[0] = y
        for i in range(1, n_steps):
            dy = np.array(seir_ode_v3(y, t[i]))
            y = y + dy * dt
            traj[i] = y
        return traj
