"""随机SEIR模型 v4.0 年龄×HIV×耐药分层方法（Mixin）。

由原 seir/stochastic.py 拆分而来，包含 270 维分层模型（5 年龄 × 3 HIV ×
2 耐药 × 9 房室）的确定性 ODE、CLE SDE、简化 CTMC 与多轨迹模拟方法。

业务逻辑与拆分前完全一致，仅做结构性重组。
"""

import math

import numpy as np

from ._stochastic_common import *


class StochasticSEIRMixinV4:
    """v4.0 年龄×HIV×耐药分层随机SEIR模型方法（270D）。

    改进7: 年龄分层 — WAIFW 混合矩阵 + 年龄特异性进展率
    改进8: HIV共感染 — ρ_react 受 HIV 状态调节
    改进9: 耐药TB — DR 株适合度代价 + 获得性耐药
    """

    # ==================================================================
    # v4.0: 年龄 × HIV × 耐药分层模型（270D）
    # ==================================================================

    def deterministic_rhs_v4(self, state_flat, t, beta,
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
                             beta_exo=DEFAULT_BETA_EXO,
                             # v4 新参数
                             waifw=None,
                             age_progression=None,
                             rr_hiv=DEFAULT_RR_HIV,
                             art_reduction=DEFAULT_ART_REDUCTION,
                             hiv_infection_rate=DEFAULT_HIV_INFECTION_RATE,
                             art_initiation_rate=DEFAULT_ART_INITIATION_RATE,
                             dr_fitness_cost=DEFAULT_DR_FITNESS_COST,
                             p_acq=DEFAULT_P_ACQ):
        """v4.0 年龄×HIV×耐药分层确定性ODE

        状态: 4D 数组 (5 age, 3 HIV, 2 DR, 9 compartment) = 270D
          age:  0=0-4, 1=5-14, 2=15-49, 3=50-64, 4=65+
          hiv:  0=HIV-, 1=HIV+未治疗, 2=HIV+ART
          dr:   0=DS, 1=DR
          comp: 0=S,1=Lf,2=Ls,3=M,4=Isub,5=Isp,6=Isn,7=R,8=C

        改进7: 年龄分层 — WAIFW 混合矩阵 + 年龄特异性进展率
        改进8: HIV共感染 — ρ_react 受 HIV 状态调节
        改进9: 耐药TB — DR 株适合度代价 + 获得性耐药
        """
        state = np.asarray(state_flat, dtype=float).reshape(
            N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)

        # 参数准备
        if waifw is None:
            waifw = DEFAULT_WAIFW
        if age_progression is None:
            age_progression = DEFAULT_AGE_PROGRESSION
        waifw = np.asarray(waifw, dtype=float)
        age_prog = np.asarray(age_progression, dtype=float)

        # HIV 再激活倍数: [1.0, RR_HIV, RR_HIV*(1-ART_reduction)]
        hiv_react_mult = np.array([
            1.0, rr_hiv, rr_hiv * (1.0 - art_reduction)])

        # 提取房室 (5, 3, 2)
        S = state[:, :, :, 0]
        Lf = state[:, :, :, 1]
        Ls = state[:, :, :, 2]
        M = state[:, :, :, 3]
        Isub = state[:, :, :, 4]
        Isp = state[:, :, :, 5]
        Isn = state[:, :, :, 6]
        R = state[:, :, :, 7]
        C = state[:, :, :, 8]

        # 各年龄组总人口
        N_age = state.sum(axis=(1, 2, 3))  # (5,)
        N_age = np.maximum(N_age, 1e-10)

        # --- 感染力（年龄特异性 + WAIFW + DR 适合度代价） ---
        # 按年龄聚合感染源（区分 DS/DR）
        def _age_inf(arr):
            """arr: (5,3,2) → 按 age 求和, 分 DR 返回 (5,2)"""
            return arr.sum(axis=1)  # (5, 2)

        Isub_age = _age_inf(Isub)   # (5, 2): [age, dr]
        Isp_age = _age_inf(Isp)
        Isn_age = _age_inf(Isn)

        infectious_per_age_dr = (
            eta_sub * Isub_age + Isp_age + eta_sn * Isn_age)  # (5, 2)
        # DR 适合度代价
        infectious_per_age_dr[:, 1] *= (1.0 - dr_fitness_cost)

        # λ[age, dr] = Σ_j waifw[age, j] * beta * infectious[j, dr] / N[j]
        inf_norm_ds = beta * infectious_per_age_dr[:, 0] / N_age  # (5,)
        inf_norm_dr = beta * infectious_per_age_dr[:, 1] / N_age
        lam_age_dr = np.zeros((N_AGE_V4, N_DR_V4))
        lam_age_dr[:, 0] = waifw @ inf_norm_ds
        lam_age_dr[:, 1] = waifw @ inf_norm_dr
        # 扩展到 (5, 3, 2): HIV 不影响感染力
        lam = lam_age_dr[:, None, :]  # (5, 1, 2) → broadcast

        # --- 感染流 ---
        infection = lam * S               # S → Lf
        reinfection_C = beta_reinf * lam * C
        reinfection_R = beta_reinf * lam * R
        exo_reinfection = beta_exo * lam * Ls  # Ls → Lf

        # --- 年龄特异性进展率 ---
        rho_fast_age = rho_fast * age_prog          # (5,)
        rho_react_age = rho_react * age_prog        # (5,)
        rho_fast_arr = rho_fast_age[:, None, None]   # (5,1,1)
        # HIV 调节再激活: (5,3,1)
        rho_react_arr = (rho_react_age[:, None, None]
                         * hiv_react_mult[None, :, None])

        # --- L_fast 流出 ---
        fast_to_min = rho_fast_arr * p_m * Lf
        fast_to_sub = rho_fast_arr * (1 - p_m) * (1 - p_clin) * Lf
        fast_to_sp = rho_fast_arr * (1 - p_m) * p_clin * Lf
        fast_to_slow = rho_conv * Lf
        clear_fast = sigma_clear * Lf

        # --- L_slow 流出 ---
        slow_to_min = rho_react_arr * p_m * Ls
        slow_to_sub = rho_react_arr * (1 - p_m) * (1 - p_clin) * Ls
        slow_to_sp = rho_react_arr * (1 - p_m) * p_clin * Ls
        clear_slow = sigma_clear * Ls

        # --- M / I_sub 流 ---
        min_to_sub = rho_min * M
        reg_m = omega_reg_m * M
        reg_sub = omega_reg_sub * Isub
        sub_to_sp = rho_prog * p_sp * Isub
        sub_to_sn = rho_prog * (1 - p_sp) * Isub
        sn_to_sp = p_sn2sp * Isn
        sp_out = (r_sp + mu_sp) * Isp
        sn_out = (r_sn + mu_sn) * Isn

        # --- 获得性耐药: DS → DR (在 I_sub, I_sp, I_sn 中) ---
        dr_acq_sub = p_acq * Isub[:, :, 0]    # (5, 3) DS→DR
        dr_acq_sp = p_acq * Isp[:, :, 0]
        dr_acq_sn = p_acq * Isn[:, :, 0]

        # --- HIV 进展 (所有房室: hiv=0→1, hiv=1→2) ---
        hiv_01 = hiv_infection_rate * state[:, 0, :, :]   # (5, 2, 9)
        art_12 = art_initiation_rate * state[:, 1, :, :]  # (5, 2, 9)

        # --- 导数计算 ---
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

        # 获得性耐药修正 (DS 减少, DR 增加)
        dIsub[:, :, 0] -= dr_acq_sub
        dIsub[:, :, 1] += dr_acq_sub
        dIsp[:, :, 0] -= dr_acq_sp
        dIsp[:, :, 1] += dr_acq_sp
        dIsn[:, :, 0] -= dr_acq_sn
        dIsn[:, :, 1] += dr_acq_sn

        # HIV 进展修正 (hiv=0 减少到 hiv=1, hiv=1 减少到 hiv=2)
        dS[:, 0, :] -= hiv_01[:, :, 0]
        dS[:, 1, :] += hiv_01[:, :, 0]
        dS[:, 1, :] -= art_12[:, :, 0]
        dS[:, 2, :] += art_12[:, :, 0]

        dLf[:, 0, :] -= hiv_01[:, :, 1]
        dLf[:, 1, :] += hiv_01[:, :, 1]
        dLf[:, 1, :] -= art_12[:, :, 1]
        dLf[:, 2, :] += art_12[:, :, 1]

        dLs[:, 0, :] -= hiv_01[:, :, 2]
        dLs[:, 1, :] += hiv_01[:, :, 2]
        dLs[:, 1, :] -= art_12[:, :, 2]
        dLs[:, 2, :] += art_12[:, :, 2]

        dM[:, 0, :] -= hiv_01[:, :, 3]
        dM[:, 1, :] += hiv_01[:, :, 3]
        dM[:, 1, :] -= art_12[:, :, 3]
        dM[:, 2, :] += art_12[:, :, 3]

        dIsub[:, 0, :] -= hiv_01[:, :, 4]
        dIsub[:, 1, :] += hiv_01[:, :, 4]
        dIsub[:, 1, :] -= art_12[:, :, 4]
        dIsub[:, 2, :] += art_12[:, :, 4]

        dIsp[:, 0, :] -= hiv_01[:, :, 5]
        dIsp[:, 1, :] += hiv_01[:, :, 5]
        dIsp[:, 1, :] -= art_12[:, :, 5]
        dIsp[:, 2, :] += art_12[:, :, 5]

        dIsn[:, 0, :] -= hiv_01[:, :, 6]
        dIsn[:, 1, :] += hiv_01[:, :, 6]
        dIsn[:, 1, :] -= art_12[:, :, 6]
        dIsn[:, 2, :] += art_12[:, :, 6]

        dR[:, 0, :] -= hiv_01[:, :, 7]
        dR[:, 1, :] += hiv_01[:, :, 7]
        dR[:, 1, :] -= art_12[:, :, 7]
        dR[:, 2, :] += art_12[:, :, 7]

        dC[:, 0, :] -= hiv_01[:, :, 8]
        dC[:, 1, :] += hiv_01[:, :, 8]
        dC[:, 1, :] -= art_12[:, :, 8]
        dC[:, 2, :] += art_12[:, :, 8]

        # 组装并展平
        rhs = np.stack([dS, dLf, dLs, dM, dIsub, dIsp, dIsn, dR, dC], axis=-1)
        return rhs.flatten()

    def simulate_sde_v4(self, initial_state, t_span, dt, beta,
                        **kwargs):
        """v4.0 年龄×HIV×耐药分层 SDE（CLE, Euler-Maruyama, 逐流扩散）

        状态: (5, 3, 2, 9) = 270D

        要点A+F: 共21个独立随机流（CLE 标度律）。
          - 要点A: 新增4个流的自清除/恢复扩散项 (clear_fast, clear_slow, sp_out, sn_out)
          - 要点F: 流14拆分为独立的 I_sub→I_sp 和 I_sub→I_sn (独立 dW, 消除人工相关性)
        每个流遵循 sqrt(rate)·dW 标度律，施加 [0, 源房室余量] 逐流约束。

        流编号 (0-20, 与 v3 一致):
          0:  S→Lf      1:  C→Lf      2:  R→Lf      3:  Ls→Lf
          4:  Lf→M      5:  Lf→Isub   6:  Lf→Isp    7:  Lf→Ls
          8:  Ls→M      9:  Ls→Isub   10: Ls→Isp
          11: M→Isub    12: M→Ls      13: Isub→M
          14: Isub→Isp  15: Isub→Isn  (要点F: 独立 dW)
          16: Isn→Isp
          17: Lf→C (clear_fast)  18: Ls→C (clear_slow)  (要点A: 新增扩散)
          19: Isp→R (sp_out)     20: Isn→R (sn_out)     (要点A: 新增扩散)

        文献:
          Gillespie (2000) J Phys Chem B 104:25-33 (CLE 标度律)
          Allen (2007) An Introduction to Stochastic Epidemic Models
        """
        state = np.array(initial_state, dtype=float).flatten()
        N_total = max(state.sum(), 1e-10)
        n_steps = min(int((t_span[1] - t_span[0]) / dt) + 1, 50000)
        times = np.linspace(t_span[0], t_span[1], n_steps)
        trajectory = np.zeros((n_steps, STATE_SIZE_V4))
        trajectory[0] = state
        sqrt_dt = math.sqrt(max(dt, 0))
        eps = self.noise_scale

        # ---- 提取参数 ----
        rho_fast = kwargs.get('rho_fast', DEFAULT_RHO_FAST)
        rho_conv = kwargs.get('rho_conv', DEFAULT_RHO_CONV)
        rho_react = kwargs.get('rho_react', DEFAULT_RHO_REACT)
        sigma_clear = kwargs.get('sigma_clear', DEFAULT_SIGMA_CLEAR)
        rho_prog = kwargs.get('rho_prog', DEFAULT_RHO_PROG)
        eta_sub = kwargs.get('eta_sub', DEFAULT_ETA_SUB)
        p_clin = kwargs.get('p_clin', DEFAULT_P_CLIN)
        beta_reinf = kwargs.get('beta_reinf', DEFAULT_BETA_REINF)
        rho_min = kwargs.get('rho_min', DEFAULT_RHO_MIN)
        p_m = kwargs.get('p_m', DEFAULT_P_M)
        omega_reg_m = kwargs.get('omega_reg_m', DEFAULT_OMEGA_REG_M)
        omega_reg_sub = kwargs.get('omega_reg_sub', DEFAULT_OMEGA_REG_SUB)
        p_sp = kwargs.get('p_sp', DEFAULT_P_SP)
        mu_sp = kwargs.get('mu_sp', DEFAULT_MU_SP)
        r_sp = kwargs.get('r_sp', DEFAULT_R_SP)
        mu_sn = kwargs.get('mu_sn', DEFAULT_MU_SN)
        r_sn = kwargs.get('r_sn', DEFAULT_R_SN)
        p_sn2sp = kwargs.get('p_sn2sp', DEFAULT_P_SN2SP)
        eta_sn = kwargs.get('eta_sn', DEFAULT_ETA_SN)
        beta_exo = kwargs.get('beta_exo', DEFAULT_BETA_EXO)
        waifw = np.asarray(kwargs.get('waifw', DEFAULT_WAIFW), dtype=float)
        age_prog = np.asarray(kwargs.get('age_progression', DEFAULT_AGE_PROGRESSION), dtype=float)
        rr_hiv = kwargs.get('rr_hiv', DEFAULT_RR_HIV)
        art_reduction = kwargs.get('art_reduction', DEFAULT_ART_REDUCTION)
        hiv_rate = kwargs.get('hiv_infection_rate', DEFAULT_HIV_INFECTION_RATE)
        art_rate = kwargs.get('art_initiation_rate', DEFAULT_ART_INITIATION_RATE)
        dr_fitness_cost = kwargs.get('dr_fitness_cost', DEFAULT_DR_FITNESS_COST)
        p_acq = kwargs.get('p_acq', DEFAULT_P_ACQ)

        hiv_react_mult = np.array([1.0, rr_hiv, rr_hiv * (1.0 - art_reduction)])
        n_flows = 21

        for i in range(1, n_steps):
            st = state.reshape(N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
            S, Lf, Ls, M, Isub, Isp, Isn, R, C = [
                st[:, :, :, c] for c in range(9)]

            # 感染力
            N_age = np.maximum(st.sum(axis=(1, 2, 3)), 1e-10)
            Isub_age = Isub.sum(axis=1)  # (5, 2)
            Isp_age = Isp.sum(axis=1)
            Isn_age = Isn.sum(axis=1)
            inf_dr = eta_sub * Isub_age + Isp_age + eta_sn * Isn_age
            inf_dr[:, 1] *= (1.0 - dr_fitness_cost)
            lam_dr = np.zeros((N_AGE_V4, N_DR_V4))
            lam_dr[:, 0] = waifw @ (beta * inf_dr[:, 0] / N_age)
            lam_dr[:, 1] = waifw @ (beta * inf_dr[:, 1] / N_age)
            lam = lam_dr[:, None, :]  # (5,1,2) → broadcast

            # 年龄/HIV 特异进展率
            rho_fa = (rho_fast * age_prog)[:, None, None]      # (5,1,1)
            rho_ra = (rho_react * age_prog)[:, None, None] * \
                     hiv_react_mult[None, :, None]             # (5,3,1)

            # ---- 21 个随机流速率 (要点A+F, 每个 5×3×2) ----
            rates = [
                np.maximum(lam * S, 0),                                    # 0: S→Lf
                np.maximum(beta_reinf * lam * C, 0),                       # 1: C→Lf
                np.maximum(beta_reinf * lam * R, 0),                       # 2: R→Lf
                np.maximum(beta_exo * lam * Ls, 0),                        # 3: Ls→Lf
                np.maximum(rho_fa * p_m * Lf, 0),                          # 4: Lf→M
                np.maximum(rho_fa * (1-p_m) * (1-p_clin) * Lf, 0),         # 5: Lf→Isub
                np.maximum(rho_fa * (1-p_m) * p_clin * Lf, 0),             # 6: Lf→Isp
                np.maximum(rho_conv * Lf, 0),                              # 7: Lf→Ls
                np.maximum(rho_ra * p_m * Ls, 0),                          # 8: Ls→M
                np.maximum(rho_ra * (1-p_m) * (1-p_clin) * Ls, 0),         # 9: Ls→Isub
                np.maximum(rho_ra * (1-p_m) * p_clin * Ls, 0),             # 10: Ls→Isp
                np.maximum(rho_min * M, 0),                                # 11: M→Isub
                np.maximum(omega_reg_m * M, 0),                            # 12: M→Ls
                np.maximum(omega_reg_sub * Isub, 0),                       # 13: Isub→M
                np.maximum(rho_prog * p_sp * Isub, 0),                     # 14: Isub→Isp (要点F)
                np.maximum(rho_prog * (1-p_sp) * Isub, 0),                 # 15: Isub→Isn (要点F)
                np.maximum(p_sn2sp * Isn, 0),                              # 16: Isn→Isp
                np.maximum(sigma_clear * Lf, 0),                           # 17: Lf→C (要点A)
                np.maximum(sigma_clear * Ls, 0),                           # 18: Ls→C (要点A)
                np.maximum((r_sp + mu_sp) * Isp, 0),                       # 19: Isp→R (要点A)
                np.maximum((r_sn + mu_sn) * Isn, 0),                       # 20: Isn→R (要点A)
            ]

            # 逐流独立维纳增量 & 扩散项
            dW = self.rng.normal(0, sqrt_dt, (n_flows, N_AGE_V4, N_HIV_V4, N_DR_V4))
            sto = eps * np.sqrt(rates) * dW  # 逐流 sqrt(rate)·dW

            # ---- 逐流约束 [0, 源房室余量] (要点A+F) ----
            f0 = np.minimum(np.maximum(rates[0]*dt + sto[0], 0), S)
            f1 = np.minimum(np.maximum(rates[1]*dt + sto[1], 0), C)
            f2 = np.minimum(np.maximum(rates[2]*dt + sto[2], 0), R)
            f3 = np.minimum(np.maximum(rates[3]*dt + sto[3], 0), Ls)
            # Lf 流出: f4, f5, f6, f7, f17 (clear_fast)
            f4 = np.minimum(np.maximum(rates[4]*dt + sto[4], 0), Lf)
            f5 = np.minimum(np.maximum(rates[5]*dt + sto[5], 0), np.maximum(Lf - f4, 0))
            f6 = np.minimum(np.maximum(rates[6]*dt + sto[6], 0), np.maximum(Lf - f4 - f5, 0))
            rem_Lf = np.maximum(Lf - f4 - f5 - f6, 0)
            f7 = np.minimum(np.maximum(rates[7]*dt + sto[7], 0), rem_Lf)
            f17 = np.minimum(np.maximum(rates[17]*dt + sto[17], 0),  # 要点A
                             np.maximum(Lf - f4 - f5 - f6 - f7, 0))
            # Ls 流出: f3, f8, f9, f10, f18 (clear_slow)
            rem_Ls = np.maximum(Ls - f3, 0)
            f8 = np.minimum(np.maximum(rates[8]*dt + sto[8], 0), rem_Ls)
            f9 = np.minimum(np.maximum(rates[9]*dt + sto[9], 0), np.maximum(rem_Ls - f8, 0))
            f10 = np.minimum(np.maximum(rates[10]*dt + sto[10], 0), np.maximum(rem_Ls - f8 - f9, 0))
            f18 = np.minimum(np.maximum(rates[18]*dt + sto[18], 0),  # 要点A
                             np.maximum(Ls - f3 - f8 - f9 - f10, 0))
            # M 流出: f11, f12
            f11 = np.minimum(np.maximum(rates[11]*dt + sto[11], 0), M)
            rem_M = np.maximum(M - f11, 0)
            f12 = np.minimum(np.maximum(rates[12]*dt + sto[12], 0), rem_M)
            # Isub 流出: f13, f14, f15 (要点F: f14/f15 独立 dW)
            f13 = np.minimum(np.maximum(rates[13]*dt + sto[13], 0), Isub)
            rem_Isub = np.maximum(Isub - f13, 0)
            f14 = np.minimum(np.maximum(rates[14]*dt + sto[14], 0), rem_Isub)       # Isub→Isp
            f15 = np.minimum(np.maximum(rates[15]*dt + sto[15], 0),                 # Isub→Isn
                             np.maximum(rem_Isub - f14, 0))
            # Isn 流出: f16, f20 (sn_out)
            f16 = np.minimum(np.maximum(rates[16]*dt + sto[16], 0), Isn)
            f20 = np.minimum(np.maximum(rates[20]*dt + sto[20], 0),  # 要点A
                             np.maximum(Isn - f16, 0))
            # Isp 流出: f19 (sp_out)
            f19 = np.minimum(np.maximum(rates[19]*dt + sto[19], 0), Isp)  # 要点A

            # ---- 更新状态 ----
            S_new = S - f0
            Lf_new = Lf + f0 + f1 + f2 + f3 - f4 - f5 - f6 - f7 - f17
            Ls_new = Ls + f7 + f12 - f3 - f8 - f9 - f10 - f18
            M_new = M + f4 + f8 + f13 - f11 - f12
            Isub_new = Isub + f5 + f9 + f11 - f13 - f14 - f15
            Isp_new = Isp + f6 + f10 + f14 + f16 - f19
            Isn_new = Isn + f15 - f16 - f20
            R_new = R + f19 + f20 - f2
            C_new = C + f17 + f18 - f1

            state_new = np.stack(
                [S_new, Lf_new, Ls_new, M_new, Isub_new,
                 Isp_new, Isn_new, R_new, C_new], axis=-1)
            state_new = np.maximum(state_new, 0)

            # ---- 获得性耐药 (确定性) ----
            for comp_idx, dr_acq_arr in [(4, Isub_new), (5, Isp_new), (6, Isn_new)]:
                acq = np.minimum(p_acq * dr_acq_arr[:, :, 0] * dt,
                                 dr_acq_arr[:, :, 0])
                state_new[:, :, 0, comp_idx] -= acq
                state_new[:, :, 1, comp_idx] += acq

            # ---- HIV 进展 (确定性, 逐房室) ----
            for comp in range(9):
                h01 = np.minimum(hiv_rate * state_new[:, 0, :, comp] * dt,
                                 state_new[:, 0, :, comp])
                a12 = np.minimum(art_rate * state_new[:, 1, :, comp] * dt,
                                 state_new[:, 1, :, comp])
                state_new[:, 0, :, comp] -= h01
                state_new[:, 1, :, comp] += h01 - a12
                state_new[:, 2, :, comp] += a12

            state_new = np.maximum(state_new, 0)

            # ---- 总人口守恒投影 (与 v3 一致) ----
            total = state_new.sum()
            if total > 0 and abs(total - N_total) > 1e-6:
                deficit = N_total - total
                flat = state_new.flatten()
                nz = flat > 0
                if nz.any():
                    flat[nz] += deficit * flat[nz] / flat[nz].sum()
                state_new = flat.reshape(
                    N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)

            state = state_new.flatten()
            trajectory[i] = state

        return times, trajectory, None

    def simulate_ctmc_v4(self, initial_state, max_time, beta,
                         **kwargs):
        """v4.0 简化 CTMC（年龄×HIV, 135D, 无 DR）

        在 270D 下 Gillespie 算法过慢，此处使用年龄×HIV 聚合版本。
        DR 效果通过参数调节近似。

        v3.0 改进 (改进3): Lf/Ls 进展事件拆分为按 p_m/p_clin 概率分配的子事件，
        正确反映 Minimal/Subclinical/Clinical 三条进展路径。
        """
        state_4d = np.array(initial_state, dtype=float).reshape(
            N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
        # 聚合 DR 维度: (5, 3, 9)
        state = state_4d.sum(axis=2)
        st = state.copy()
        times = [0.0]
        states = [st.copy().tolist()]
        t = 0.0
        max_iter = 50000

        waifw = kwargs.get('waifw', DEFAULT_WAIFW)
        age_prog = kwargs.get('age_progression', DEFAULT_AGE_PROGRESSION)
        rr_hiv = kwargs.get('rr_hiv', DEFAULT_RR_HIV)
        art_red = kwargs.get('art_reduction', DEFAULT_ART_REDUCTION)
        hiv_rate = kwargs.get('hiv_infection_rate', DEFAULT_HIV_INFECTION_RATE)
        art_rate = kwargs.get('art_initiation_rate', DEFAULT_ART_INITIATION_RATE)
        p_m = kwargs.get('p_m', DEFAULT_P_M)
        p_clin = kwargs.get('p_clin', DEFAULT_P_CLIN)

        hiv_mult = np.array([1.0, rr_hiv, rr_hiv * (1 - art_red)])

        for _ in range(max_iter):
            if t >= max_time:
                break
            S, Lf, Ls, M, Isub, Isp, Isn, R, C = [
                st[:, :, c] for c in range(9)]
            N_age = st.sum(axis=(1, 2))
            N_age = np.maximum(N_age, 1e-10)

            # 感染力
            inf_age = (kwargs.get('eta_sub', DEFAULT_ETA_SUB) * Isub.sum(axis=1)
                       + Isp.sum(axis=1)
                       + kwargs.get('eta_sn', DEFAULT_ETA_SN) * Isn.sum(axis=1))
            lam_age = waifw @ (beta * inf_age / N_age)
            lam = lam_age[:, None]  # (5, 1) → broadcast to (5, 3)

            # ---- v3.0 改进3: 事件 4/5 拆分为 6 个子事件 ----
            # 构建 (22, 5, 3, 9) 一致形状数组:
            #   事件 0-3:  感染类 (5,3) 速率放置在源房室位置
            #   事件 4-6:  Lf 进展 → M / Isub / Isp (按 p_m, p_clin 分配)
            #   事件 7-9:  Ls 再激活 → M / Isub / Isp (按 p_m, p_clin 分配)
            #   事件 10-19: 其他房室转换
            #   事件 20-21: HIV/ART 进展
            n_events = 22
            rates_4d = np.zeros((n_events, N_AGE_V4, N_HIV_V4, N_COMPARTMENTS_V3))
            rates_4d[0, :, :, IDX_S_V3] = lam * S
            rates_4d[1, :, :, IDX_C_V3] = \
                kwargs.get('beta_reinf', DEFAULT_BETA_REINF) * lam * C
            rates_4d[2, :, :, IDX_R_V3] = \
                kwargs.get('beta_reinf', DEFAULT_BETA_REINF) * lam * R
            rates_4d[3, :, :, IDX_LS_V3] = \
                kwargs.get('beta_exo', DEFAULT_BETA_EXO) * lam * Ls
            # Lf 进展拆分: M / Isub / Isp
            rho_fast_k = kwargs.get('rho_fast', DEFAULT_RHO_FAST)
            rates_4d[4, :, :, IDX_LF_V3] = \
                rho_fast_k * p_m * age_prog[:, None] * Lf
            rates_4d[5, :, :, IDX_LF_V3] = \
                rho_fast_k * (1 - p_m) * (1 - p_clin) * age_prog[:, None] * Lf
            rates_4d[6, :, :, IDX_LF_V3] = \
                rho_fast_k * (1 - p_m) * p_clin * age_prog[:, None] * Lf
            # Ls 再激活拆分: M / Isub / Isp
            rho_react_k = kwargs.get('rho_react', DEFAULT_RHO_REACT)
            rates_4d[7, :, :, IDX_LS_V3] = \
                rho_react_k * p_m * age_prog[:, None] * hiv_mult[None, :] * Ls
            rates_4d[8, :, :, IDX_LS_V3] = \
                rho_react_k * (1 - p_m) * (1 - p_clin) * \
                age_prog[:, None] * hiv_mult[None, :] * Ls
            rates_4d[9, :, :, IDX_LS_V3] = \
                rho_react_k * (1 - p_m) * p_clin * \
                age_prog[:, None] * hiv_mult[None, :] * Ls
            # 其他房室转换
            rates_4d[10, :, :, IDX_LF_V3] = \
                kwargs.get('rho_conv', DEFAULT_RHO_CONV) * Lf
            rates_4d[11, :, :, IDX_LF_V3] = \
                kwargs.get('sigma_clear', DEFAULT_SIGMA_CLEAR) * Lf
            rates_4d[12, :, :, IDX_LS_V3] = \
                kwargs.get('sigma_clear', DEFAULT_SIGMA_CLEAR) * Ls
            rates_4d[13, :, :, IDX_M_V3] = \
                kwargs.get('rho_min', DEFAULT_RHO_MIN) * M
            rates_4d[14, :, :, IDX_M_V3] = \
                kwargs.get('omega_reg_m', DEFAULT_OMEGA_REG_M) * M
            rates_4d[15, :, :, IDX_ISUB_V3] = \
                kwargs.get('omega_reg_sub', DEFAULT_OMEGA_REG_SUB) * Isub
            rates_4d[16, :, :, IDX_ISUB_V3] = \
                kwargs.get('rho_prog', DEFAULT_RHO_PROG) * Isub
            rates_4d[17, :, :, IDX_ISN_V3] = \
                kwargs.get('p_sn2sp', DEFAULT_P_SN2SP) * Isn
            rates_4d[18, :, :, IDX_ISP_V3] = \
                (kwargs.get('r_sp', DEFAULT_R_SP)
                 + kwargs.get('mu_sp', DEFAULT_MU_SP)) * Isp
            rates_4d[19, :, :, IDX_ISN_V3] = \
                (kwargs.get('r_sn', DEFAULT_R_SN)
                 + kwargs.get('mu_sn', DEFAULT_MU_SN)) * Isn
            rates_4d[20, :, 0, :] = hiv_rate * st[:, 0, :]   # HIV 0→1
            rates_4d[21, :, 1, :] = art_rate * st[:, 1, :]   # ART 1→2

            total_rate = float(rates_4d.sum())
            if total_rate <= 1e-12:
                break

            tau = self.rng.exponential(1.0 / total_rate)
            t += tau
            if t >= max_time:
                break

            # 随机选择事件类型和位置
            flat_rates = rates_4d.flatten()
            cum = np.cumsum(flat_rates)
            u = self.rng.uniform(0, total_rate)
            idx = int(np.searchsorted(cum, u))

            # 解码索引: (event_type, age, hiv, comp)
            n_age, n_hiv, n_comp = 5, 3, 9
            event_type = idx // (n_age * n_hiv * n_comp)
            remainder = idx % (n_age * n_hiv * n_comp)
            age_idx = remainder // (n_hiv * n_comp)
            remainder = remainder % (n_hiv * n_comp)
            hiv_idx = remainder // n_comp
            comp_idx = remainder % n_comp

            # 执行事件（移动一个个体）
            if event_type < 4:
                # 感染类: src→Lf
                src_map = {0: 0, 1: 8, 2: 7, 3: 2}  # S, C, R, Ls
                src = src_map[event_type]
                if st[age_idx, hiv_idx, src] > 0:
                    st[age_idx, hiv_idx, src] -= 1
                    st[age_idx, hiv_idx, 1] += 1  # → Lf
            elif event_type == 4:  # Lf→M
                if st[age_idx, hiv_idx, 1] > 0:
                    st[age_idx, hiv_idx, 1] -= 1
                    st[age_idx, hiv_idx, 3] += 1
            elif event_type == 5:  # Lf→Isub
                if st[age_idx, hiv_idx, 1] > 0:
                    st[age_idx, hiv_idx, 1] -= 1
                    st[age_idx, hiv_idx, 4] += 1
            elif event_type == 6:  # Lf→Isp
                if st[age_idx, hiv_idx, 1] > 0:
                    st[age_idx, hiv_idx, 1] -= 1
                    st[age_idx, hiv_idx, 5] += 1
            elif event_type == 7:  # Ls→M
                if st[age_idx, hiv_idx, 2] > 0:
                    st[age_idx, hiv_idx, 2] -= 1
                    st[age_idx, hiv_idx, 3] += 1
            elif event_type == 8:  # Ls→Isub
                if st[age_idx, hiv_idx, 2] > 0:
                    st[age_idx, hiv_idx, 2] -= 1
                    st[age_idx, hiv_idx, 4] += 1
            elif event_type == 9:  # Ls→Isp
                if st[age_idx, hiv_idx, 2] > 0:
                    st[age_idx, hiv_idx, 2] -= 1
                    st[age_idx, hiv_idx, 5] += 1
            elif event_type == 10:  # Lf→Ls
                if st[age_idx, hiv_idx, 1] > 0:
                    st[age_idx, hiv_idx, 1] -= 1
                    st[age_idx, hiv_idx, 2] += 1
            elif event_type == 11:  # Lf→C
                if st[age_idx, hiv_idx, 1] > 0:
                    st[age_idx, hiv_idx, 1] -= 1
                    st[age_idx, hiv_idx, 8] += 1
            elif event_type == 12:  # Ls→C
                if st[age_idx, hiv_idx, 2] > 0:
                    st[age_idx, hiv_idx, 2] -= 1
                    st[age_idx, hiv_idx, 8] += 1
            elif event_type == 13:  # M→Isub
                if st[age_idx, hiv_idx, 3] > 0:
                    st[age_idx, hiv_idx, 3] -= 1
                    st[age_idx, hiv_idx, 4] += 1
            elif event_type == 14:  # M→Ls
                if st[age_idx, hiv_idx, 3] > 0:
                    st[age_idx, hiv_idx, 3] -= 1
                    st[age_idx, hiv_idx, 2] += 1
            elif event_type == 15:  # Isub→M
                if st[age_idx, hiv_idx, 4] > 0:
                    st[age_idx, hiv_idx, 4] -= 1
                    st[age_idx, hiv_idx, 3] += 1
            elif event_type == 16:  # Isub→Isp/Isn
                if st[age_idx, hiv_idx, 4] > 0:
                    st[age_idx, hiv_idx, 4] -= 1
                    if self.rng.random() < kwargs.get('p_sp', DEFAULT_P_SP):
                        st[age_idx, hiv_idx, 5] += 1
                    else:
                        st[age_idx, hiv_idx, 6] += 1
            elif event_type == 17:  # Isn→Isp
                if st[age_idx, hiv_idx, 6] > 0:
                    st[age_idx, hiv_idx, 6] -= 1
                    st[age_idx, hiv_idx, 5] += 1
            elif event_type == 18:  # Isp→R
                if st[age_idx, hiv_idx, 5] > 0:
                    st[age_idx, hiv_idx, 5] -= 1
                    st[age_idx, hiv_idx, 7] += 1
            elif event_type == 19:  # Isn→R
                if st[age_idx, hiv_idx, 6] > 0:
                    st[age_idx, hiv_idx, 6] -= 1
                    st[age_idx, hiv_idx, 7] += 1
            elif event_type == 20:  # HIV 0→1
                if st[age_idx, 0, comp_idx] > 0:
                    st[age_idx, 0, comp_idx] -= 1
                    st[age_idx, 1, comp_idx] += 1
            elif event_type == 21:  # ART 1→2
                if st[age_idx, 1, comp_idx] > 0:
                    st[age_idx, 1, comp_idx] -= 1
                    st[age_idx, 2, comp_idx] += 1

            st = np.maximum(st, 0)
            times.append(t)
            states.append(st.copy().tolist())

        return np.array(times), np.array(states)

    def simulate_multiple_v4(self, initial_state, t_span, dt, beta,
                             n_trajectories=50, **kwargs):
        """v4.0 多轨迹SDE模拟，返回活动性TB (I_sp+I_sn) 轨迹

        返回形状: (n_trajectories, n_steps, n_age)
        """
        trajectories = []
        for _ in range(n_trajectories):
            _, traj, _ = self.simulate_sde_v4(
                initial_state, t_span, dt, beta, **kwargs)
            # 聚合: 活动 TB = I_sp + I_sn, 按年龄求和 (over HIV, DR)
            st = traj.reshape(-1, N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
            active = (st[:, :, :, :, IDX_ISP_V3]
                      + st[:, :, :, :, IDX_ISN_V3]).sum(axis=(2, 3))
            trajectories.append(active)
        return np.stack(trajectories, axis=0)
