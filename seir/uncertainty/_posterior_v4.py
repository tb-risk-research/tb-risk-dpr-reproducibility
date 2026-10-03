"""SEIR 参数不确定性量化 — v4 后验预测模块

v4.0 270D 年龄×HIV×耐药分层后验预测分布、R0（NGM 下一代矩阵）与 ODE 轨迹回退。
拆分自原 seir/uncertainty.py（业务逻辑不变）。
"""

import logging
import numpy as np

LOGGER = logging.getLogger("tb_risk.seir.uncertainty")


class _PosteriorV4Mixin:
    """后验预测分布 v4: 270D 分层 SDE/ODE、NGM R0、分层置信带"""

    def _compute_R0_v4_ngm(self, beta, rr_hiv=20.0, art_reduction=0.65,
                            dr_fitness_cost=0.15, eta_sn=0.35,
                            waifw=None, age_prog=None):
        """要点C: v4.0 下一代矩阵 R0

        v4 模型有 5 年龄 × 3 HIV × 2 DR = 30 个"传染性产生"子群
        （每个子群包含 I_sub, I_sp, I_sn 三个传染性房室）。
        R0 = 下一代矩阵 K 的谱半径 (最大特征值)。

        K[i,j] = (子群 j 中一个感染者在其传染期内，在子群 i 中产生的
                 新感染数)。

        简化假设（保持解析可处理）：
          - 传染性持续时间 d = 加权平均 (I_sub: 1/ρ_prog, I_sp: 1/(r_sp+μ_sp),
            I_sn: 1/(r_sn+μ_sn+p_sn2sp))，权重取 p_sp, (1-p_sp) 等
          - HIV+ 未治疗相对传染性 = 1（与 HIV- 相同，但再激活风险高），
            HIV+ART 传染性 = 1 - art_reduction（减少）
          - DR 株传染性 = 1 - dr_fitness_cost
          - 年龄混合通过 WAIFW 矩阵
          - 年龄进展通过 age_prog 调整感染性产生

        文献:
          Diekmann O, Heesterbeek JA, Metz JA. (1990) J Math Biol
          van den Driessche P, Watmough J. (2008) Math Biosci Eng
        """
        from .._stochastic_common import (N_AGE_V4, N_HIV_V4, N_DR_V4,
                                           DEFAULT_WAIFW, DEFAULT_AGE_PROGRESSION)
        W = np.asarray(waifw if waifw is not None else DEFAULT_WAIFW,
                       dtype=float)
        ap = np.asarray(age_prog if age_prog is not None
                        else DEFAULT_AGE_PROGRESSION, dtype=float)
        f = self._V4_FIXED

        # 传染性持续时间（天）: 加权平均
        d_sp = 1.0 / max(f['r_sp'] + f['mu_sp'], 1e-10)
        d_sn = 1.0 / max(f['r_sn'] + f['mu_sn'] + f['p_sn2sp'], 1e-10)
        p_sp = f['p_sp']
        p_sn = 1.0 - p_sp
        # I_sub 占比小（很快进展），这里简化为传染性主要来自 I_sp/I_sn
        d_infectious = p_sp * d_sp + p_sn * d_sn

        # HIV 层传染性乘子
        hiv_trans = np.array([1.0, 1.0, 1.0 - art_reduction])
        # DR 层传染性乘子
        dr_trans = np.array([1.0, 1.0 - dr_fitness_cost])
        # 年龄层再激活风险乘子（影响传染性产生，非感染力）
        age_mult = ap / ap.mean()

        # 构建 30×30 下一代矩阵（仅传染性产生部分）
        # 简化: 感染力 β * W[a_i, a_j] / N_age * trans_factors
        # K[(a,h,d), (a',h',d')] = beta * W[a, a'] * d_infectious
        #   * hiv_trans[h'] * dr_trans[d'] * age_mult[a']
        # 注: N_age 归一化在 W 中已隐含（W 是接触矩阵）
        n_groups = N_AGE_V4 * N_HIV_V4 * N_DR_V4
        K = np.zeros((n_groups, n_groups))

        for a_i in range(N_AGE_V4):
            for h_i in range(N_HIV_V4):
                for d_i in range(N_DR_V4):
                    i = (a_i * N_HIV_V4 + h_i) * N_DR_V4 + d_i
                    for a_j in range(N_AGE_V4):
                        for h_j in range(N_HIV_V4):
                            for d_j in range(N_DR_V4):
                                j = ((a_j * N_HIV_V4 + h_j) * N_DR_V4
                                     + d_j)
                                K[i, j] = (
                                    float(beta) * W[a_i, a_j]
                                    * d_infectious
                                    * hiv_trans[h_j]
                                    * dr_trans[d_j]
                                    * age_mult[a_j]
                                )

        # R0 = 谱半径（最大特征值绝对值）
        try:
            eigenvalues = np.linalg.eigvals(K)
            r0 = float(np.max(np.abs(eigenvalues)))
        except np.linalg.LinAlgError:
            r0 = float('nan')
        return r0

    def _v4_ode_trajectory(self, initial_state, t_span, dt, params_dict,
                            N_total):
        """v4.0 270D ODE 轨迹（SDE 不可用时的回退）

        使用 StochasticSEIRModel.deterministic_rhs_v4 + Euler 积分。
        """
        if self.stochastic_seir is None or not hasattr(
                self.stochastic_seir, 'deterministic_rhs_v4'):
            # 进一步回退：返回零轨迹
            n_steps = min(int((t_span[1] - t_span[0]) / dt) + 1, 50000)
            return np.zeros((n_steps, 270))

        n_steps = min(int((t_span[1] - t_span[0]) / dt) + 1, 50000)
        traj = np.zeros((n_steps, len(initial_state)))
        y = np.array(initial_state, dtype=float).flatten()
        traj[0] = y

        for i in range(1, n_steps):
            rhs = self.stochastic_seir.deterministic_rhs_v4(
                y, 0.0, beta=params_dict['beta'],
                rho_fast=params_dict.get('rho_fast', 0.1 / 365.0),
                rho_react=params_dict.get('rho_react', 0.004 / 365.0),
                sigma_clear=params_dict.get('sigma_clear', 1.0 / 365.0),
                gamma=params_dict.get('gamma', 0.05),
                omega_reg_m=params_dict.get('omega_reg_m', 1.0 / 365.0),
                omega_reg_sub=params_dict.get('omega_reg_sub', 1.0 / 365.0),
                beta_exo=params_dict.get('beta_exo', 0.33),
                eta_sn=params_dict.get('eta_sn', 0.35),
                rr_hiv=params_dict.get('rr_hiv', 20.0),
                art_reduction=params_dict.get('art_reduction', 0.65),
                hiv_infection_rate=params_dict.get(
                    'hiv_infection_rate', 0.001),
                art_initiation_rate=params_dict.get(
                    'art_initiation_rate', 0.1 / 365.0),
                dr_fitness_cost=params_dict.get('dr_fitness_cost', 0.15),
                p_acq=params_dict.get('p_acq', 0.03),
            )
            y = y + np.asarray(rhs) * dt
            # 非负约束
            y = np.maximum(y, 0)
            traj[i] = y

        # 人口守恒显式检查 (M 级审计修复): 封闭 v4 模型 N=S+...+C 应严格
        # 守恒, Euler 截断 + np.maximum 非负截断会引入质量漂移。超阈值
        # 仅告警 (轨迹仍返回, 由调用方决定是否采信)。
        try:
            from ..autodiff import (population_conservation_error,
                                    POP_CONSERVATION_RTOL)
            pop_err = population_conservation_error(traj)
            if pop_err > POP_CONSERVATION_RTOL:
                LOGGER.warning(
                    "v4 ODE 回退轨迹人口守恒违反: "
                    "max|N(t)-N(0)|/N(0) = %.3e > %.1e (dt=%s)。"
                    "提示: 减小 dt 或改用 SDE 路径 (use_sde=True)。",
                    pop_err, POP_CONSERVATION_RTOL, dt)
        except Exception:  # 检查失败不影响回退轨迹返回
            pass

        return traj

    def generate_posterior_predictive_v4(self, initial_state, t_span, dt,
                                          n_samples=100, use_sde=True,
                                          n_subsample=None):
        """要点C: v4.0 270D 后验预测分布

        从 ``run_mcmc_v4`` 产生的后验样本中采样参数，运行 v4 SDE（或 ODE 回退）
        生成轨迹集合，输出按年龄×HIV×DR 分层活动性 TB 的置信带与 R0 分布。

        输出指标:
          - 各年龄组活动性 TB 发病率（I_sub+I_sp+I_sn, 按年龄聚合）
          - 各 HIV 层 TB 占比
          - DS/DR TB 比例
          - R0 分布（下一代矩阵谱半径）

        参数:
          initial_state: 270D 初始状态
          t_span: (t_start, t_end)
          dt: 积分步长（天）
          n_samples: 后验采样数
          use_sde: True 用 simulate_sde_v4, False 用 ODE
          n_subsample: 后验样本池大小（None = 全部）

        返回:
          dict: {
            'trajectories', 'active_I_total', 'ci_bands',
            'by_age', 'by_hiv', 'by_dr',
            'r0_samples', 'r0_mean', 'r0_ci', 'p_r0_gt_1',
            'peak_I_samples', 'n_samples', 'method'
          }
        """
        from .._stochastic_common import (N_AGE_V4, N_HIV_V4, N_DR_V4,
                                           N_COMPARTMENTS_V3)

        if not self.posterior_samples:
            return {'error': '后验样本不存在，请先运行 run_mcmc_v4'}
        ps = self.posterior_samples
        if 'beta' not in ps:
            return {'error': '后验样本缺少 beta 键'}

        n_posterior = len(ps['beta'])
        if n_posterior == 0:
            return {'error': '后验样本为空'}

        if n_subsample is not None:
            pool = min(n_subsample, n_posterior)
        else:
            pool = n_posterior
        indices = self.rng.choice(pool, size=min(n_samples, pool))

        # 采样后验参数（v4 PARAM_NAMES）
        def _sample(key, default):
            if key in ps:
                arr = np.asarray(ps[key])
                if len(arr) >= pool:
                    return arr[indices]
            return np.full(len(indices), default)

        beta_vals = _sample('beta', 0.2)
        rho_fast_vals = _sample('rho_fast', 0.1 / 365.0)
        rho_react_vals = _sample('rho_react', 0.004 / 365.0)
        sigma_clear_vals = _sample('sigma_clear', 1.0 / 365.0)
        gamma_vals = _sample('gamma', 0.05)
        omega_m_vals = _sample('omega_reg_m', 1.0 / 365.0)
        omega_sub_vals = _sample('omega_reg_sub', 1.0 / 365.0)
        beta_exo_vals = _sample('beta_exo', 0.33)
        eta_sn_vals = _sample('eta_sn', 0.35)
        rr_hiv_vals = _sample('rr_hiv', 20.0)
        art_red_vals = _sample('art_reduction', 0.65)
        hiv_rate_vals = _sample('hiv_infection_rate', 0.001)
        art_rate_vals = _sample('art_initiation_rate', 0.1 / 365.0)
        dr_cost_vals = _sample('dr_fitness_cost', 0.15)
        p_acq_vals = _sample('p_acq', 0.03)

        N_total = max(float(np.sum(initial_state)), 1e-10)
        n_actual = len(indices)

        # SDE 可用性检查
        sde_available = (use_sde and self.stochastic_seir is not None
                         and hasattr(self.stochastic_seir, 'simulate_sde_v4'))

        trajectories = []
        r0_samples = []
        peak_I_samples = []

        for i in range(n_actual):
            # R0（下一代矩阵）
            r0_samples.append(self._compute_R0_v4_ngm(
                float(beta_vals[i]),
                rr_hiv=float(rr_hiv_vals[i]),
                art_reduction=float(art_red_vals[i]),
                dr_fitness_cost=float(dr_cost_vals[i]),
                eta_sn=float(eta_sn_vals[i]),
            ))

            params_dict = {
                'beta': float(beta_vals[i]),
                'rho_fast': float(rho_fast_vals[i]),
                'rho_react': float(rho_react_vals[i]),
                'sigma_clear': float(sigma_clear_vals[i]),
                'gamma': float(gamma_vals[i]),
                'omega_reg_m': float(omega_m_vals[i]),
                'omega_reg_sub': float(omega_sub_vals[i]),
                'beta_exo': float(beta_exo_vals[i]),
                'eta_sn': float(eta_sn_vals[i]),
                'rr_hiv': float(rr_hiv_vals[i]),
                'art_reduction': float(art_red_vals[i]),
                'hiv_infection_rate': float(hiv_rate_vals[i]),
                'art_initiation_rate': float(art_rate_vals[i]),
                'dr_fitness_cost': float(dr_cost_vals[i]),
                'p_acq': float(p_acq_vals[i]),
            }

            if sde_available:
                try:
                    _times, traj, _sto = self.stochastic_seir.simulate_sde_v4(
                        initial_state, t_span, dt,
                        beta=params_dict['beta'],
                        rho_fast=params_dict['rho_fast'],
                        rho_react=params_dict['rho_react'],
                        sigma_clear=params_dict['sigma_clear'],
                        gamma=params_dict['gamma'],
                        omega_reg_m=params_dict['omega_reg_m'],
                        omega_reg_sub=params_dict['omega_reg_sub'],
                        beta_exo=params_dict['beta_exo'],
                        eta_sn=params_dict['eta_sn'],
                        rr_hiv=params_dict['rr_hiv'],
                        art_reduction=params_dict['art_reduction'],
                        hiv_infection_rate=params_dict['hiv_infection_rate'],
                        art_initiation_rate=params_dict['art_initiation_rate'],
                        dr_fitness_cost=params_dict['dr_fitness_cost'],
                        p_acq=params_dict['p_acq'],
                    )
                    traj = np.asarray(traj)
                except Exception as e:
                    LOGGER.debug("v4 SDE轨迹生成失败，回退到ODE: %s", e)
                    traj = self._v4_ode_trajectory(
                        initial_state, t_span, dt, params_dict, N_total)
            else:
                traj = self._v4_ode_trajectory(
                    initial_state, t_span, dt, params_dict, N_total)

            trajectories.append(traj)
            # 活动性 TB = I_sub + I_sp + I_sn (房室 4, 5, 6)
            active_I = (traj[:, 4::9] + traj[:, 5::9] + traj[:, 6::9]).sum(axis=1)
            peak_I_samples.append(float(np.max(active_I)))

        trajectories = np.array(trajectories)  # (n_samples, n_steps, 270)
        # 活动性 TB 总量（所有分层求和）
        active_I_all = (trajectories[:, :, 4::9]
                        + trajectories[:, :, 5::9]
                        + trajectories[:, :, 6::9]).sum(axis=2)

        ci_lower = np.percentile(active_I_all, 2.5, axis=0)
        ci_upper = np.percentile(active_I_all, 97.5, axis=0)
        median_I = np.percentile(active_I_all, 50, axis=0)

        n_steps = trajectories.shape[1]
        times = np.linspace(t_span[0], t_span[1], n_steps)

        # 按年龄/HIV/DR 分层（使用最后一个时间步的快照）
        final_traj = trajectories[:, -1, :].reshape(
            n_actual, N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)

        # by_age: 各年龄组活动性 TB (5 组)
        active_by_age = (final_traj[:, :, :, :, 4]
                         + final_traj[:, :, :, :, 5]
                         + final_traj[:, :, :, :, 6]).sum(axis=(2, 3))
        # by_hiv: 各 HIV 层活动性 TB (3 层)
        active_by_hiv = (final_traj[:, :, :, :, 4]
                         + final_traj[:, :, :, :, 5]
                         + final_traj[:, :, :, :, 6]).sum(axis=(1, 3))
        # by_dr: DS/DR 活动性 TB (2 组)
        active_by_dr = (final_traj[:, :, :, :, 4]
                        + final_traj[:, :, :, :, 5]
                        + final_traj[:, :, :, :, 6]).sum(axis=(1, 2))

        def _pctl_arr(arr, axis=0):
            lo = np.percentile(arr, 2.5, axis=axis)
            md = np.percentile(arr, 50.0, axis=axis)
            hi = np.percentile(arr, 97.5, axis=axis)
            return {'lower': lo.tolist(), 'median': md.tolist(),
                    'upper': hi.tolist()}

        return {
            'trajectories': trajectories,
            'active_I': active_I_all,
            'ci_bands': {
                'lower': ci_lower.tolist(),
                'upper': ci_upper.tolist(),
                'median': median_I.tolist(),
                'time': times.tolist(),
            },
            'by_age': _pctl_arr(active_by_age),
            'by_hiv': _pctl_arr(active_by_hiv),
            'by_dr': _pctl_arr(active_by_dr),
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
            'n_age_groups': N_AGE_V4,
            'n_hiv_layers': N_HIV_V4,
            'n_dr_layers': N_DR_V4,
        }
