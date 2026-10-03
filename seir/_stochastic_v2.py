"""随机SEIR模型 v2 7-房室方法（Mixin）。

由原 seir/stochastic.py 拆分而来，包含：
  - 旧版4-房室向后兼容 API（deterministic_rhs / simulate_sde /
    simulate_ctmc / simulate_multiple）
  - 7-房室 v2 实现（*_v2 后缀）

业务逻辑与拆分前完全一致，仅做结构性重组。
"""

import math

import numpy as np

from ._stochastic_common import *


class StochasticSEIRMixinV2:
    """7-房室随机SEIR模型方法（v2.0）。

    房室结构（7维）：[S, L_fast, L_slow, I_sub, I_clin, R, C]
    """

    # ==================== 确定性 ODE ====================

    def deterministic_rhs(self, state, t, beta, sigma, gamma):
        """旧版4-房室确定性ODE（向后兼容，委托到 v2）"""
        S, E, I, R = state
        # 将旧参数映射到新模型：E→L_fast, I→I_clin
        state_v2 = np.array([S, E, 0.0, 0.0, I, R, 0.0])
        rhs = self.deterministic_rhs_v2(
            state_v2, t, beta,
            rho_fast=sigma,        # 旧 sigma ≈ 新 rho_fast
            rho_conv=0.0,          # 无快→慢转换
            rho_react=0.0,         # 无慢再激活
            sigma_clear=0.0,       # 无自清除
            rho_prog=0.0,          # 无亚临床进展
            eta_sub=0.0,           # 无亚临床传染
            p_clin=1.0,            # 全部直接临床
            gamma=gamma,
            beta_reinf=0.0,        # 无再感染
        )
        # 映射回4维: dS, dE=dL_fast+dL_slow, dI=dI_sub+dI_clin, dR
        return np.array([
            rhs[IDX_S],
            rhs[IDX_LF] + rhs[IDX_LS],
            rhs[IDX_ISUB] + rhs[IDX_ICLIN],
            rhs[IDX_R],
        ])

    def deterministic_rhs_v2(self, state, t, beta,
                             rho_fast=DEFAULT_RHO_FAST,
                             rho_conv=DEFAULT_RHO_CONV,
                             rho_react=DEFAULT_RHO_REACT,
                             sigma_clear=DEFAULT_SIGMA_CLEAR,
                             rho_prog=DEFAULT_RHO_PROG,
                             eta_sub=DEFAULT_ETA_SUB,
                             p_clin=DEFAULT_P_CLIN,
                             gamma=DEFAULT_GAMMA,
                             beta_reinf=DEFAULT_BETA_REINF):
        """7-房室确定性ODE右侧

        state: [S, L_fast, L_slow, I_sub, I_clin, R, C]
        """
        S, Lf, Ls, Isub, Iclin, R, C = state
        N = self.population

        # 感染力（含亚临床传染性加权）
        lambda_force = beta * (eta_sub * Isub + Iclin) / max(N, 1e-10)

        # 各流速率
        infection = lambda_force * S
        reinfection = beta_reinf * lambda_force * C
        fast_prog_sub = rho_fast * (1 - p_clin) * Lf   # L_fast → I_sub
        fast_prog_clin = rho_fast * p_clin * Lf         # L_fast → I_clin
        fast_to_slow = rho_conv * Lf                     # L_fast → L_slow
        clear_fast = sigma_clear * Lf                    # L_fast → C
        slow_prog_sub = rho_react * (1 - p_clin) * Ls    # L_slow → I_sub
        slow_prog_clin = rho_react * p_clin * Ls          # L_slow → I_clin
        clear_slow = sigma_clear * Ls                     # L_slow → C
        sub_to_clin = rho_prog * Isub                     # I_sub → I_clin
        recovery = gamma * Iclin                          # I_clin → R

        # 导数
        dS = -infection + recovery  # 注：R→S 此处简化，实际为 death+birth 平衡
        # 实际: dS = μ*N - infection - μ*S + ...  简化处理
        dLf = infection - (fast_prog_sub + fast_prog_clin + fast_to_slow + clear_fast) + reinfection
        dLs = fast_to_slow - (slow_prog_sub + slow_prog_clin + clear_slow)
        dIsub = fast_prog_sub + slow_prog_sub - sub_to_clin
        dIclin = fast_prog_clin + slow_prog_clin + sub_to_clin - recovery
        dR = recovery  # 简化：含 S→R 的其他路径
        dC = clear_fast + clear_slow - reinfection

        return np.array([dS, dLf, dLs, dIsub, dIclin, dR, dC])

    # ==================== 随机微分方程 (SDE) ====================

    def simulate_sde(self, initial_state, t_span, dt, beta, sigma, gamma):
        """旧版4-房室SDE（向后兼容，委托到 v2）"""
        S0, E0, I0, R0 = initial_state
        init_v2 = np.array([S0, E0, 0.0, 0.0, I0, R0, 0.0])
        times, traj_v2, _ = self.simulate_sde_v2(
            init_v2, t_span, dt, beta,
            rho_fast=sigma, rho_conv=0.0, rho_react=0.0,
            sigma_clear=0.0, rho_prog=0.0, eta_sub=0.0,
            p_clin=1.0, gamma=gamma, beta_reinf=0.0,
        )
        traj = np.column_stack([
            traj_v2[:, IDX_S],
            traj_v2[:, IDX_LF] + traj_v2[:, IDX_LS],
            traj_v2[:, IDX_ISUB] + traj_v2[:, IDX_ICLIN],
            traj_v2[:, IDX_R],
        ])
        return times, traj, np.zeros((len(times), 3))

    def simulate_sde_v2(self, initial_state, t_span, dt, beta,
                        rho_fast=DEFAULT_RHO_FAST,
                        rho_conv=DEFAULT_RHO_CONV,
                        rho_react=DEFAULT_RHO_REACT,
                        sigma_clear=DEFAULT_SIGMA_CLEAR,
                        rho_prog=DEFAULT_RHO_PROG,
                        eta_sub=DEFAULT_ETA_SUB,
                        p_clin=DEFAULT_P_CLIN,
                        gamma=DEFAULT_GAMMA,
                        beta_reinf=DEFAULT_BETA_REINF):
        """7-房室化学朗之万方程(CLE) SDE模拟

        Euler-Maruyama步进，扩散项遵循Gillespie(2000)标度律 sqrt(rate)·dW。
        共8个扩散项（对应8个独立流），每个流约束在[0, 源房室]范围内。

        返回: (times, trajectory, stochastic_forcing)
        """
        state = np.array(initial_state, dtype=float)
        N_total = max(sum(state), 1e-10)
        n_steps = int((t_span[1] - t_span[0]) / dt) + 1
        n_steps = min(n_steps, 100000)
        times = np.linspace(t_span[0], t_span[1], n_steps)
        trajectory = np.zeros((n_steps, N_COMPARTMENTS))
        n_flows = 8  # 感染、快进展临床、快进展亚临床、快→慢、慢进展临床、慢进展亚临床、亚临床→临床、恢复
        stochastic_forcing = np.zeros((n_steps, n_flows))
        trajectory[0] = state
        sqrt_dt = math.sqrt(max(dt, 0))
        eps = self.noise_scale

        for i in range(1, n_steps):
            S, Lf, Ls, Isub, Iclin, R, C = state
            N = max(S + Lf + Ls + Isub + Iclin + R + C, 1e-10)
            lambda_force = beta * (eta_sub * Isub + Iclin) / N

            # 8个流的确定性速率
            rate_infection = max(lambda_force * S + beta_reinf * lambda_force * C, 0)
            rate_fast_clin = max(rho_fast * p_clin * Lf, 0)
            rate_fast_sub = max(rho_fast * (1 - p_clin) * Lf, 0)
            rate_fast_to_slow = max(rho_conv * Lf, 0)
            rate_slow_clin = max(rho_react * p_clin * Ls, 0)
            rate_slow_sub = max(rho_react * (1 - p_clin) * Ls, 0)
            rate_sub_to_clin = max(rho_prog * Isub, 0)
            rate_recovery = max(gamma * Iclin, 0)

            rates = np.array([
                rate_infection, rate_fast_clin, rate_fast_sub,
                rate_fast_to_slow, rate_slow_clin, rate_slow_sub,
                rate_sub_to_clin, rate_recovery,
            ])

            # 独立维纳过程增量
            dW = self.rng.normal(0, sqrt_dt, n_flows)
            sto = eps * np.sqrt(np.maximum(rates, 0)) * dW

            # Euler-Maruyama步进，每个流施加不可逆约束
            # 流1: 感染 (S + C → L_fast)
            flow_infection = rate_infection * dt + sto[0]
            flow_infection = min(max(flow_infection, 0.0), S + C)
            # 从S和C按比例扣除
            total_src = max(S + C, 1e-10)
            from_S = flow_infection * S / total_src
            from_C = flow_infection * C / total_src

            # 流2: L_fast → I_clin
            flow_fast_clin = rate_fast_clin * dt + sto[1]
            flow_fast_clin = min(max(flow_fast_clin, 0.0), Lf)

            # 流3: L_fast → I_sub
            flow_fast_sub = rate_fast_sub * dt + sto[2]
            flow_fast_sub = min(max(flow_fast_sub, 0.0), Lf - flow_fast_clin)

            # 流4: L_fast → L_slow
            remaining_Lf = max(Lf - flow_fast_clin - flow_fast_sub, 0)
            flow_fast_to_slow = rate_fast_to_slow * dt + sto[3]
            flow_fast_to_slow = min(max(flow_fast_to_slow, 0.0), remaining_Lf)

            # 流5: L_slow → I_clin
            flow_slow_clin = rate_slow_clin * dt + sto[4]
            flow_slow_clin = min(max(flow_slow_clin, 0.0), Ls)

            # 流6: L_slow → I_sub
            flow_slow_sub = rate_slow_sub * dt + sto[5]
            flow_slow_sub = min(max(flow_slow_sub, 0.0), Ls - flow_slow_clin)

            # 流7: I_sub → I_clin
            flow_sub_to_clin = rate_sub_to_clin * dt + sto[6]
            flow_sub_to_clin = min(max(flow_sub_to_clin, 0.0), Isub)

            # 流8: I_clin → R
            flow_recovery = rate_recovery * dt + sto[7]
            flow_recovery = min(max(flow_recovery, 0.0), Iclin)

            # 自清除流（确定性部分，无独立扩散项）
            clear_fast = min(max(sigma_clear * Lf * dt, 0.0),
                             max(Lf - flow_fast_clin - flow_fast_sub - flow_fast_to_slow, 0))
            clear_slow = min(max(sigma_clear * Ls * dt, 0.0),
                             max(Ls - flow_slow_clin - flow_slow_sub, 0))

            # 更新状态
            S_new = S - from_S
            Lf_new = (Lf + flow_infection
                      - flow_fast_clin - flow_fast_sub
                      - flow_fast_to_slow - clear_fast)
            Ls_new = (Ls + flow_fast_to_slow
                      - flow_slow_clin - flow_slow_sub - clear_slow)
            Isub_new = (Isub + flow_fast_sub + flow_slow_sub
                        - flow_sub_to_clin)
            Iclin_new = (Iclin + flow_fast_clin + flow_slow_clin
                         + flow_sub_to_clin - flow_recovery)
            R_new = R + flow_recovery
            C_new = C + clear_fast + clear_slow - from_C

            state = np.array([S_new, Lf_new, Ls_new, Isub_new, Iclin_new, R_new, C_new])
            state = np.maximum(state, 0)

            # 投影：保持总人口守恒
            total = state.sum()
            if total > 0:
                deficit = N_total - total
                if deficit != 0:
                    non_zero_mask = state > 0
                    if non_zero_mask.any():
                        state[non_zero_mask] += deficit * state[non_zero_mask] / state[non_zero_mask].sum()

            trajectory[i] = state
            stochastic_forcing[i] = sto

        return times, trajectory, stochastic_forcing

    # ==================== 连续时间马尔可夫链 (CTMC) ====================

    def simulate_ctmc(self, initial_state, max_time, beta, sigma, gamma):
        """旧版4-房室CTMC（向后兼容，委托到 v2）"""
        S0, E0, I0, R0 = initial_state
        init_v2 = np.array([S0, E0, 0.0, 0.0, I0, R0, 0.0])
        times, states_v2 = self.simulate_ctmc_v2(
            init_v2, max_time, beta,
            rho_fast=sigma, rho_conv=0.0, rho_react=0.0,
            sigma_clear=0.0, rho_prog=0.0, eta_sub=0.0,
            p_clin=1.0, gamma=gamma, beta_reinf=0.0,
        )
        states = np.column_stack([
            states_v2[:, IDX_S],
            states_v2[:, IDX_LF] + states_v2[:, IDX_LS],
            states_v2[:, IDX_ISUB] + states_v2[:, IDX_ICLIN],
            states_v2[:, IDX_R],
        ])
        return times, states

    def simulate_ctmc_v2(self, initial_state, max_time, beta,
                         rho_fast=DEFAULT_RHO_FAST,
                         rho_conv=DEFAULT_RHO_CONV,
                         rho_react=DEFAULT_RHO_REACT,
                         sigma_clear=DEFAULT_SIGMA_CLEAR,
                         rho_prog=DEFAULT_RHO_PROG,
                         eta_sub=DEFAULT_ETA_SUB,
                         p_clin=DEFAULT_P_CLIN,
                         gamma=DEFAULT_GAMMA,
                         beta_reinf=DEFAULT_BETA_REINF):
        """7-房室Gillespie算法CTMC模拟

        事件列表以 (源房室索引, 目标房室索引, 速率函数) 元组列表形式组织，
        使Gillespie算法通过遍历列表自动适应任意房室数。

        8个事件：
          0: S → L_fast (感染)
          1: C → L_fast (再感染)
          2: L_fast → I_clin (快进展临床)
          3: L_fast → I_sub (快进展亚临床)
          4: L_fast → L_slow (快→慢转换)
          5: L_slow → I_clin (慢再激活临床)
          6: L_slow → I_sub (慢再激活亚临床)
          7: I_sub → I_clin (亚临床进展)
          8: I_clin → R (恢复)
          9: L_fast → C (自清除)
          10: L_slow → C (自清除)
        """
        state = np.array(initial_state, dtype=float)
        # CTMC状态必须为整数
        S, Lf, Ls, Isub, Iclin, R, C = [int(round(s)) for s in state]

        times = [0.0]
        states = [[S, Lf, Ls, Isub, Iclin, R, C]]
        t = 0.0
        max_iterations = 100000

        # 事件列表：(源房室, 目标房室, 速率函数)
        # 速率函数签名为 (S, Lf, Ls, Isub, Iclin, R, C, N, lambda_force)
        events = [
            # (src, dst, rate_fn)
            (IDX_S, IDX_LF, lambda s, lf, ls, iu, ic, r, c, N, lam: max(lam * s, 0)),
            (IDX_C, IDX_LF, lambda s, lf, ls, iu, ic, r, c, N, lam: max(beta_reinf * lam * c, 0)),
            (IDX_LF, IDX_ICLIN, lambda s, lf, ls, iu, ic, r, c, N, lam: max(rho_fast * p_clin * lf, 0)),
            (IDX_LF, IDX_ISUB, lambda s, lf, ls, iu, ic, r, c, N, lam: max(rho_fast * (1 - p_clin) * lf, 0)),
            (IDX_LF, IDX_LS, lambda s, lf, ls, iu, ic, r, c, N, lam: max(rho_conv * lf, 0)),
            (IDX_LS, IDX_ICLIN, lambda s, lf, ls, iu, ic, r, c, N, lam: max(rho_react * p_clin * ls, 0)),
            (IDX_LS, IDX_ISUB, lambda s, lf, ls, iu, ic, r, c, N, lam: max(rho_react * (1 - p_clin) * ls, 0)),
            (IDX_ISUB, IDX_ICLIN, lambda s, lf, ls, iu, ic, r, c, N, lam: max(rho_prog * iu, 0)),
            (IDX_ICLIN, IDX_R, lambda s, lf, ls, iu, ic, r, c, N, lam: max(gamma * ic, 0)),
            (IDX_LF, IDX_C, lambda s, lf, ls, iu, ic, r, c, N, lam: max(sigma_clear * lf, 0)),
            (IDX_LS, IDX_C, lambda s, lf, ls, iu, ic, r, c, N, lam: max(sigma_clear * ls, 0)),
        ]

        iteration = 0
        # 有潜伏者或活动性TB即继续
        while t < max_time and (Lf > 0 or Ls > 0 or Isub > 0 or Iclin > 0) and iteration < max_iterations:
            iteration += 1
            N = max(S + Lf + Ls + Isub + Iclin + R + C, 1e-10)
            lambda_force = beta * (eta_sub * Isub + Iclin) / N

            # 计算所有事件速率
            event_rates = []
            for src, dst, rate_fn in events:
                rate = rate_fn(S, Lf, Ls, Isub, Iclin, R, C, N, lambda_force)
                event_rates.append(rate)

            cum_rates = np.cumsum(event_rates)
            a_total = cum_rates[-1]

            if a_total <= 1e-12:
                break

            tau = self.rng.exponential(1.0 / a_total)
            t += tau

            # 选择事件
            u = self.rng.uniform(0, a_total)
            event_idx = int(np.searchsorted(cum_rates, u))

            # 执行事件
            src, dst, _ = events[event_idx]
            state_arr = [S, Lf, Ls, Isub, Iclin, R, C]
            if state_arr[src] > 0:
                state_arr[src] -= 1
                state_arr[dst] += 1
            S, Lf, Ls, Isub, Iclin, R, C = state_arr

            S, Lf, Ls, Isub, Iclin, R, C = (
                max(S, 0), max(Lf, 0), max(Ls, 0),
                max(Isub, 0), max(Iclin, 0), max(R, 0), max(C, 0))

            times.append(t)
            states.append([S, Lf, Ls, Isub, Iclin, R, C])

        return np.array(times), np.array(states)

    def simulate_multiple(self, initial_state, t_span, dt, beta, sigma, gamma,
                          n_trajectories=100):
        """旧版多轨迹模拟（向后兼容）"""
        trajectories = []
        for _ in range(n_trajectories):
            _, traj, _ = self.simulate_sde(initial_state, t_span, dt, beta, sigma, gamma)
            trajectories.append(traj)
        return np.stack(trajectories, axis=0)[:, :, 2]  # 返回I室轨迹

    def simulate_multiple_v2(self, initial_state, t_span, dt, beta,
                             rho_fast=DEFAULT_RHO_FAST,
                             rho_conv=DEFAULT_RHO_CONV,
                             rho_react=DEFAULT_RHO_REACT,
                             sigma_clear=DEFAULT_SIGMA_CLEAR,
                             rho_prog=DEFAULT_RHO_PROG,
                             eta_sub=DEFAULT_ETA_SUB,
                             p_clin=DEFAULT_P_CLIN,
                             gamma=DEFAULT_GAMMA,
                             beta_reinf=DEFAULT_BETA_REINF,
                             n_trajectories=100):
        """7-房室多轨迹SDE模拟，返回所有轨迹的I_clin房室"""
        trajectories = []
        for _ in range(n_trajectories):
            _, traj, _ = self.simulate_sde_v2(
                initial_state, t_span, dt, beta,
                rho_fast=rho_fast, rho_conv=rho_conv,
                rho_react=rho_react, sigma_clear=sigma_clear,
                rho_prog=rho_prog, eta_sub=eta_sub,
                p_clin=p_clin, gamma=gamma, beta_reinf=beta_reinf,
            )
            trajectories.append(traj)
        return np.stack(trajectories, axis=0)[:, :, IDX_ICLIN]
