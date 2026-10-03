"""随机SEIR模型 v3.0 9-房室方法（Mixin）。

由原 seir/stochastic.py 拆分而来，包含 9-房室扩展版的确定性 ODE、
CLE SDE、Gillespie CTMC 与多轨迹模拟方法。

业务逻辑与拆分前完全一致，仅做结构性重组。
"""

import math

import numpy as np

from ._stochastic_common import *


class StochasticSEIRMixinV3:
    """9-房室随机SEIR模型方法（v3.0）。

    v3.0 新增改进：
      4. 疾病状态波动：M 房室 + 回退流 (ω_reg_M, ω_reg_sub)
      5. 涂阳/涂阴分层：I_clin → I_sp + I_sn (不同死亡率/自愈率/传染性)
      6. 内源性/外源性再感染分离：β_exo 参数

    房室结构（9维）：[S, L_fast, L_slow, M, I_sub, I_sp, I_sn, R, C]
    """

    # ==================== v3.0 9-房室模型方法 ====================

    def deterministic_rhs_v3(self, state, t, beta,
                             rho_fast=DEFAULT_RHO_FAST,
                             rho_conv=DEFAULT_RHO_CONV,
                             rho_react=DEFAULT_RHO_REACT,
                             sigma_clear=DEFAULT_SIGMA_CLEAR,
                             rho_prog=DEFAULT_RHO_PROG,
                             eta_sub=DEFAULT_ETA_SUB,
                             p_clin=DEFAULT_P_CLIN,
                             gamma=DEFAULT_GAMMA,
                             beta_reinf=DEFAULT_BETA_REINF,
                             # v3 新参数
                             rho_min=DEFAULT_RHO_MIN,
                             p_m=DEFAULT_P_M,
                             omega_reg_m=DEFAULT_OMEGA_REG_M,
                             omega_reg_sub=DEFAULT_OMEGA_REG_SUB,
                             p_sp=DEFAULT_P_SP,
                             mu_sp=DEFAULT_MU_SP, r_sp=DEFAULT_R_SP,
                             mu_sn=DEFAULT_MU_SN, r_sn=DEFAULT_R_SN,
                             p_sn2sp=DEFAULT_P_SN2SP,
                             eta_sn=DEFAULT_ETA_SN,
                             beta_exo=DEFAULT_BETA_EXO):
        """9-房室确定性ODE右侧

        state: [S, L_fast, L_slow, M, I_sub, I_sp, I_sn, R, C]

        v3.0 新增流：
          - 疾病波动: M→L_slow (ω_reg_m), I_sub→M (ω_reg_sub)
          - 涂阳/涂阴: I_sp/I_sn 不同流出率
          - 外源性再感染: L_slow→L_fast (β_exo·λ), R→L_fast (β_reinf·λ)
        """
        S, Lf, Ls, M, Isub, Isp, Isn, R, C = state
        N = max(self.population, 1e-10)

        # 感染力：η_sp=1.0 锚点，η_sn, η_sub 为相对传染性
        lambda_force = beta * (eta_sub * Isub + Isp + eta_sn * Isn) / N

        # --- 感染流 ---
        infection_S = lambda_force * S
        reinfection_C = beta_reinf * lambda_force * C
        reinfection_R = beta_reinf * lambda_force * R
        exo_reinfection_Ls = beta_exo * lambda_force * Ls

        # --- L_fast 流出 ---
        fast_to_min = rho_fast * p_m * Lf
        fast_to_sub = rho_fast * (1 - p_m) * (1 - p_clin) * Lf
        fast_to_sp = rho_fast * (1 - p_m) * p_clin * Lf
        fast_to_slow = rho_conv * Lf
        clear_fast = sigma_clear * Lf

        # --- L_slow 流出 ---
        slow_to_min = rho_react * p_m * Ls
        slow_to_sub = rho_react * (1 - p_m) * (1 - p_clin) * Ls
        slow_to_sp = rho_react * (1 - p_m) * p_clin * Ls
        clear_slow = sigma_clear * Ls

        # --- M 流 ---
        min_to_sub = rho_min * M
        reg_m_to_ls = omega_reg_m * M

        # --- I_sub 流 ---
        reg_sub_to_m = omega_reg_sub * Isub
        sub_to_sp = rho_prog * p_sp * Isub
        sub_to_sn = rho_prog * (1 - p_sp) * Isub

        # --- I_sn 流 ---
        sn_to_sp = p_sn2sp * Isn

        # --- I_sp/I_sn 流出 ---
        sp_out = (r_sp + mu_sp) * Isp
        sn_out = (r_sn + mu_sn) * Isn

        # --- 导数 ---
        dS = -infection_S
        dLf = (infection_S + reinfection_C + reinfection_R + exo_reinfection_Ls
               - fast_to_min - fast_to_sub - fast_to_sp - fast_to_slow - clear_fast)
        dLs = (fast_to_slow + reg_m_to_ls
               - slow_to_min - slow_to_sub - slow_to_sp - clear_slow - exo_reinfection_Ls)
        dM = (fast_to_min + slow_to_min + reg_sub_to_m - min_to_sub - reg_m_to_ls)
        dIsub = (fast_to_sub + slow_to_sub + min_to_sub
                 - sub_to_sp - sub_to_sn - reg_sub_to_m)
        dIsp = (fast_to_sp + slow_to_sp + sub_to_sp + sn_to_sp - sp_out)
        dIsn = (sub_to_sn - sn_to_sp - sn_out)
        dR = (sp_out + sn_out - reinfection_R)
        dC = (clear_fast + clear_slow - reinfection_C)

        return np.array([dS, dLf, dLs, dM, dIsub, dIsp, dIsn, dR, dC])

    def simulate_sde_v3(self, initial_state, t_span, dt, beta,
                        rho_fast=DEFAULT_RHO_FAST,
                        rho_conv=DEFAULT_RHO_CONV,
                        rho_react=DEFAULT_RHO_REACT,
                        sigma_clear=DEFAULT_SIGMA_CLEAR,
                        rho_prog=DEFAULT_RHO_PROG,
                        eta_sub=DEFAULT_ETA_SUB,
                        p_clin=DEFAULT_P_CLIN,
                        gamma=DEFAULT_GAMMA,
                        beta_reinf=DEFAULT_BETA_REINF,
                        rho_min=DEFAULT_RHO_MIN,
                        p_m=DEFAULT_P_M,
                        omega_reg_m=DEFAULT_OMEGA_REG_M,
                        omega_reg_sub=DEFAULT_OMEGA_REG_SUB,
                        p_sp=DEFAULT_P_SP,
                        mu_sp=DEFAULT_MU_SP, r_sp=DEFAULT_R_SP,
                        mu_sn=DEFAULT_MU_SN, r_sn=DEFAULT_R_SN,
                        p_sn2sp=DEFAULT_P_SN2SP,
                        eta_sn=DEFAULT_ETA_SN,
                        beta_exo=DEFAULT_BETA_EXO):
        """9-房室CLE SDE模拟

        要点A+F: 共21个独立随机流（CLE 标度律）。
          - 要点A: 新增4个流的自清除/恢复扩散项 (clear_fast, clear_slow, sp_out, sn_out)
          - 要点F: 流14拆分为独立的 I_sub→I_sp 和 I_sub→I_sn (独立 dW, 消除人工相关性)
        每个流遵循 sqrt(rate)·dW 标度律，施加 [0, 源房室余量] 逐流约束。

        流编号 (0-20):
          0:  S→Lf      1:  C→Lf      2:  R→Lf      3:  Ls→Lf
          4:  Lf→M      5:  Lf→Isub   6:  Lf→Isp    7:  Lf→Ls
          8:  Ls→M      9:  Ls→Isub   10: Ls→Isp
          11: M→Isub    12: M→Ls      13: Isub→M
          14: Isub→Isp  15: Isub→Isn  (要点F: 独立 dW)
          16: Isn→Isp
          17: Lf→C (clear_fast)  18: Ls→C (clear_slow)  (要点A: 新增扩散)
          19: Isp→R (sp_out)     20: Isn→R (sn_out)     (要点A: 新增扩散)
        """
        state = np.array(initial_state, dtype=float)
        N_total = max(sum(state), 1e-10)
        n_steps = min(int((t_span[1] - t_span[0]) / dt) + 1, 100000)
        times = np.linspace(t_span[0], t_span[1], n_steps)
        trajectory = np.zeros((n_steps, N_COMPARTMENTS_V3))
        n_flows = 21
        stochastic_forcing = np.zeros((n_steps, n_flows))
        trajectory[0] = state
        sqrt_dt = math.sqrt(max(dt, 0))
        eps = self.noise_scale

        for i in range(1, n_steps):
            S, Lf, Ls, M, Isub, Isp, Isn, R, C = state
            N = max(S + Lf + Ls + M + Isub + Isp + Isn + R + C, 1e-10)
            lam = beta * (eta_sub * Isub + Isp + eta_sn * Isn) / N

            # 21个随机流速率 (要点A+F)
            rates = np.array([
                max(lam * S, 0),                              # 0: S→Lf
                max(beta_reinf * lam * C, 0),                 # 1: C→Lf
                max(beta_reinf * lam * R, 0),                 # 2: R→Lf
                max(beta_exo * lam * Ls, 0),                  # 3: Ls→Lf
                max(rho_fast * p_m * Lf, 0),                  # 4: Lf→M
                max(rho_fast * (1-p_m) * (1-p_clin) * Lf, 0), # 5: Lf→Isub
                max(rho_fast * (1-p_m) * p_clin * Lf, 0),     # 6: Lf→Isp
                max(rho_conv * Lf, 0),                        # 7: Lf→Ls
                max(rho_react * p_m * Ls, 0),                 # 8: Ls→M
                max(rho_react * (1-p_m) * (1-p_clin) * Ls, 0),# 9: Ls→Isub
                max(rho_react * (1-p_m) * p_clin * Ls, 0),    # 10: Ls→Isp
                max(rho_min * M, 0),                          # 11: M→Isub
                max(omega_reg_m * M, 0),                      # 12: M→Ls
                max(omega_reg_sub * Isub, 0),                 # 13: Isub→M
                max(rho_prog * p_sp * Isub, 0),               # 14: Isub→Isp (要点F: 独立)
                max(rho_prog * (1-p_sp) * Isub, 0),           # 15: Isub→Isn (要点F: 独立)
                max(p_sn2sp * Isn, 0),                        # 16: Isn→Isp
                max(sigma_clear * Lf, 0),                     # 17: Lf→C (要点A: 扩散)
                max(sigma_clear * Ls, 0),                     # 18: Ls→C (要点A: 扩散)
                max((r_sp + mu_sp) * Isp, 0),                 # 19: Isp→R (要点A: 扩散)
                max((r_sn + mu_sn) * Isn, 0),                 # 20: Isn→R (要点A: 扩散)
            ])

            dW = self.rng.normal(0, sqrt_dt, n_flows)
            sto = eps * np.sqrt(rates) * dW

            # ---- 逐流约束 [0, 源房室余量] ----
            f0 = min(max(rates[0]*dt + sto[0], 0), S)
            f1 = min(max(rates[1]*dt + sto[1], 0), C)
            f2 = min(max(rates[2]*dt + sto[2], 0), R)
            f3 = min(max(rates[3]*dt + sto[3], 0), Ls)
            # Lf 流出: f4, f5, f6, f7, f17 (clear_fast)
            f4 = min(max(rates[4]*dt + sto[4], 0), Lf)
            f5 = min(max(rates[5]*dt + sto[5], 0), Lf - f4)
            f6 = min(max(rates[6]*dt + sto[6], 0), Lf - f4 - f5)
            rem_Lf = max(Lf - f4 - f5 - f6, 0)
            f7 = min(max(rates[7]*dt + sto[7], 0), rem_Lf)
            f17 = min(max(rates[17]*dt + sto[17], 0),  # 要点A: clear_fast 扩散
                      max(Lf - f4 - f5 - f6 - f7, 0))
            # Ls 流出: f3, f8, f9, f10, f18 (clear_slow)
            rem_Ls = max(Ls - f3, 0)
            f8 = min(max(rates[8]*dt + sto[8], 0), rem_Ls)
            f9 = min(max(rates[9]*dt + sto[9], 0), rem_Ls - f8)
            f10 = min(max(rates[10]*dt + sto[10], 0), rem_Ls - f8 - f9)
            f18 = min(max(rates[18]*dt + sto[18], 0),  # 要点A: clear_slow 扩散
                      max(Ls - f3 - f8 - f9 - f10, 0))
            # M 流出: f11, f12
            f11 = min(max(rates[11]*dt + sto[11], 0), M)
            rem_M = max(M - f11, 0)
            f12 = min(max(rates[12]*dt + sto[12], 0), rem_M)
            # Isub 流出: f13, f14, f15 (要点F: f14/f15 独立 dW)
            f13 = min(max(rates[13]*dt + sto[13], 0), Isub)
            rem_Isub = max(Isub - f13, 0)
            f14 = min(max(rates[14]*dt + sto[14], 0), rem_Isub)      # Isub→Isp
            f15 = min(max(rates[15]*dt + sto[15], 0),                # Isub→Isn
                      max(rem_Isub - f14, 0))
            # Isn 流出: f16, f20 (sn_out)
            f16 = min(max(rates[16]*dt + sto[16], 0), Isn)
            f20 = min(max(rates[20]*dt + sto[20], 0),  # 要点A: sn_out 扩散
                      max(Isn - f16, 0))
            # Isp 流出: f19 (sp_out)
            f19 = min(max(rates[19]*dt + sto[19], 0), Isp)  # 要点A: sp_out 扩散

            # 更新状态
            S_new = S - f0
            Lf_new = (Lf + f0 + f1 + f2 + f3
                      - f4 - f5 - f6 - f7 - f17)
            Ls_new = (Ls + f7 + f12
                      - f3 - f8 - f9 - f10 - f18)
            M_new = M + f4 + f8 + f13 - f11 - f12
            Isub_new = (Isub + f5 + f9 + f11
                        - f13 - f14 - f15)
            Isp_new = (Isp + f6 + f10 + f14 + f16 - f19)
            Isn_new = (Isn + f15 - f16 - f20)
            R_new = R + f19 + f20 - f2
            C_new = C + f17 + f18 - f1

            state = np.maximum(np.array([
                S_new, Lf_new, Ls_new, M_new, Isub_new,
                Isp_new, Isn_new, R_new, C_new]), 0)

            # 总人口守恒投影
            total = state.sum()
            if total > 0 and abs(total - N_total) > 1e-6:
                deficit = N_total - total
                nz = state > 0
                if nz.any():
                    state[nz] += deficit * state[nz] / state[nz].sum()

            trajectory[i] = state
            stochastic_forcing[i] = sto

        return times, trajectory, stochastic_forcing

    def simulate_ctmc_v3(self, initial_state, max_time, beta,
                         rho_fast=DEFAULT_RHO_FAST,
                         rho_conv=DEFAULT_RHO_CONV,
                         rho_react=DEFAULT_RHO_REACT,
                         sigma_clear=DEFAULT_SIGMA_CLEAR,
                         rho_prog=DEFAULT_RHO_PROG,
                         eta_sub=DEFAULT_ETA_SUB,
                         p_clin=DEFAULT_P_CLIN,
                         gamma=DEFAULT_GAMMA,
                         beta_reinf=DEFAULT_BETA_REINF,
                         rho_min=DEFAULT_RHO_MIN,
                         p_m=DEFAULT_P_M,
                         omega_reg_m=DEFAULT_OMEGA_REG_M,
                         omega_reg_sub=DEFAULT_OMEGA_REG_SUB,
                         p_sp=DEFAULT_P_SP,
                         mu_sp=DEFAULT_MU_SP, r_sp=DEFAULT_R_SP,
                         mu_sn=DEFAULT_MU_SN, r_sn=DEFAULT_R_SN,
                         p_sn2sp=DEFAULT_P_SN2SP,
                         eta_sn=DEFAULT_ETA_SN,
                         beta_exo=DEFAULT_BETA_EXO):
        """9-房室Gillespie算法CTMC模拟

        21个事件以 (源, 目标, 速率函数) 元组列表形式组织。
        """
        state = np.array(initial_state, dtype=float)
        S, Lf, Ls, M, Isub, Isp, Isn, R, C = [int(round(s)) for s in state]

        times = [0.0]
        states = [[S, Lf, Ls, M, Isub, Isp, Isn, R, C]]
        t = 0.0
        max_iter = 100000

        # 事件列表: (src_idx, dst_idx, rate_fn)
        # rate_fn 签名: (st, lam) 其中 st=[S,Lf,Ls,M,Isub,Isp,Isn,R,C]
        events = [
            (IDX_S_V3, IDX_LF_V3, lambda st, lam: max(lam * st[0], 0)),
            (IDX_C_V3, IDX_LF_V3, lambda st, lam: max(beta_reinf * lam * st[8], 0)),
            (IDX_R_V3, IDX_LF_V3, lambda st, lam: max(beta_reinf * lam * st[7], 0)),
            (IDX_LS_V3, IDX_LF_V3, lambda st, lam: max(beta_exo * lam * st[2], 0)),
            (IDX_LF_V3, IDX_M_V3, lambda st, lam: max(rho_fast * p_m * st[1], 0)),
            (IDX_LF_V3, IDX_ISUB_V3, lambda st, lam: max(rho_fast * (1-p_m) * (1-p_clin) * st[1], 0)),
            (IDX_LF_V3, IDX_ISP_V3, lambda st, lam: max(rho_fast * (1-p_m) * p_clin * st[1], 0)),
            (IDX_LF_V3, IDX_LS_V3, lambda st, lam: max(rho_conv * st[1], 0)),
            (IDX_LS_V3, IDX_M_V3, lambda st, lam: max(rho_react * p_m * st[2], 0)),
            (IDX_LS_V3, IDX_ISUB_V3, lambda st, lam: max(rho_react * (1-p_m) * (1-p_clin) * st[2], 0)),
            (IDX_LS_V3, IDX_ISP_V3, lambda st, lam: max(rho_react * (1-p_m) * p_clin * st[2], 0)),
            (IDX_M_V3, IDX_ISUB_V3, lambda st, lam: max(rho_min * st[3], 0)),
            (IDX_M_V3, IDX_LS_V3, lambda st, lam: max(omega_reg_m * st[3], 0)),
            (IDX_ISUB_V3, IDX_M_V3, lambda st, lam: max(omega_reg_sub * st[4], 0)),
            (IDX_ISUB_V3, IDX_ISP_V3, lambda st, lam: max(rho_prog * p_sp * st[4], 0)),
            (IDX_ISUB_V3, IDX_ISN_V3, lambda st, lam: max(rho_prog * (1-p_sp) * st[4], 0)),
            (IDX_ISN_V3, IDX_ISP_V3, lambda st, lam: max(p_sn2sp * st[6], 0)),
            (IDX_ISP_V3, IDX_R_V3, lambda st, lam: max((r_sp + mu_sp) * st[5], 0)),
            (IDX_ISN_V3, IDX_R_V3, lambda st, lam: max((r_sn + mu_sn) * st[6], 0)),
            (IDX_LF_V3, IDX_C_V3, lambda st, lam: max(sigma_clear * st[1], 0)),
            (IDX_LS_V3, IDX_C_V3, lambda st, lam: max(sigma_clear * st[2], 0)),
        ]

        st = [S, Lf, Ls, M, Isub, Isp, Isn, R, C]
        iteration = 0
        while t < max_time and (Lf > 0 or Ls > 0 or M > 0 or
                                Isub > 0 or Isp > 0 or Isn > 0) and iteration < max_iter:
            iteration += 1
            N = max(sum(st), 1e-10)
            lam = beta * (eta_sub * st[4] + st[5] + eta_sn * st[6]) / N

            event_rates = [fn(st, lam) for _, _, fn in events]
            cum_rates = np.cumsum(event_rates)
            a_total = cum_rates[-1]

            if a_total <= 1e-12:
                break

            tau = self.rng.exponential(1.0 / a_total)
            t += tau
            u = self.rng.uniform(0, a_total)
            event_idx = int(np.searchsorted(cum_rates, u))

            src, dst, _ = events[event_idx]
            if st[src] > 0:
                st[src] -= 1
                st[dst] += 1

            st = [max(int(s), 0) for s in st]
            times.append(t)
            states.append(list(st))

        return np.array(times), np.array(states)

    def simulate_multiple_v3(self, initial_state, t_span, dt, beta,
                             n_trajectories=100, **kwargs):
        """9-房室多轨迹SDE模拟，返回活动性TB (I_sp+I_sn) 轨迹"""
        trajectories = []
        for _ in range(n_trajectories):
            _, traj, _ = self.simulate_sde_v3(initial_state, t_span, dt, beta, **kwargs)
            trajectories.append(traj)
        # 活动性临床TB = I_sp + I_sn
        active_clin = np.stack(trajectories, axis=0)
        return active_clin[:, :, IDX_ISP_V3] + active_clin[:, :, IDX_ISN_V3]
