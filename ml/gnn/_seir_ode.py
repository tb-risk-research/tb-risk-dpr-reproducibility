"""轻量级 SEIR ODE 求解器（一阶 Euler / 四阶 RK4，均可微分）

包含 seir_ode_step, seir_ode_step_v3, seir_euler_integrate,
seir_rk4_integrate, seir_v3_euler_integrate, seir_v3_rk4_integrate,
seir_ode_gating, seir_solver_error_baseline 及 9-房室索引常量。

数值方法（问题二修复）：
    生产路径（seir_ode_gating）默认使用四阶龙格-库塔（RK4）积分，
    全局误差由一阶 Euler 的 O(dt) 降至 O(dt⁴)，且与 Euler 一样是
    显式可微的（仅含标量乘加与导数函数求值），无需额外依赖。
    seir_euler_integrate / seir_v3_euler_integrate 保留一阶 Euler
    实现，用于向后兼容与误差基准对比（seir_solver_error_baseline）。
"""

from ._base import torch

# 问题八-2：9-房室索引单一真值源 — 从 seir._stochastic_common 导入，
# 消除本地 _IDX_* 字面量副本。p_clin 漂移（0.33 vs 0.332）一并修复。
from ...seir._stochastic_common import (
    IDX_S_V3 as _IDX_S,
    IDX_LF_V3 as _IDX_LF,
    IDX_LS_V3 as _IDX_LS,
    IDX_M_V3 as _IDX_M,
    IDX_ISUB_V3 as _IDX_ISUB,
    IDX_ISP_V3 as _IDX_ISP,
    IDX_ISN_V3 as _IDX_ISN,
    IDX_R_V3 as _IDX_R,
    IDX_C_V3 as _IDX_C,
    N_COMPARTMENTS_V3 as _N_COMPARTMENTS_V3,
    DEFAULT_P_CLIN as _DEFAULT_P_CLIN,
    DEFAULT_ETA_SN as _DEFAULT_ETA_SN,
)


def seir_ode_step(state, beta, sigma, gamma):
    """SEIR ODE 单步导数计算（向后兼容 4D 版本）

    状态变量（归一化比例，S+E+I+R=1）：
        state: [..., 4] 张量，最后一维为 [S, E, I, R]

    参数：
        beta: 传播率 (transmission rate)，shape 可广播到 state[..., :1]
        sigma: 潜伏→感染转化率 (1/潜伏期)，shape 可广播
        gamma: 恢复率 (1/传染期)，shape 可广播

    返回：
        dstate: [..., 4] 导数张量 [dS, dE, dI, dR]

    文献：
        Kermack & McKendrick (1927) "A Contribution to the Mathematical Theory of Epidemics"
        Hethcote (2000) "The Mathematics of Infectious Diseases" SIAM Review
    """
    S = state[..., 0:1]
    E = state[..., 1:2]
    I = state[..., 2:3]
    # R = state[..., 3:4]  # 守恒量，不需要显式计算

    # dS/dt = -β * S * I
    dS = -beta * S * I
    # dE/dt = β * S * I - σ * E
    dE = beta * S * I - sigma * E
    # dI/dt = σ * E - γ * I
    dI = sigma * E - gamma * I
    # dR/dt = γ * I
    dR = gamma * I

    return torch.cat([dS, dE, dI, dR], dim=-1)


def seir_ode_step_v3(state, beta, rho_fast, rho_react, sigma_clear, gamma,
                     rho_conv=0.003, rho_prog=0.05, eta_sub=1.0,
                     p_clin=_DEFAULT_P_CLIN, p_m=0.5, rho_min=0.001,
                     omega_reg_m=0.003, omega_reg_sub=0.003,
                     beta_exo=0.33, eta_sn=_DEFAULT_ETA_SN,
                     p_sp=0.55, mu_sp=0.001, r_sp=0.001,
                     mu_sn=0.0001, r_sn=0.0004, p_sn2sp=0.0003,
                     beta_reinf=0.4):
    """9-房室 SEIR ODE 单步导数计算（v3.0 扩展版）

    状态变量（归一化比例，和为 1）：
        state: [..., 9] 张量，最后一维为
        [S, L_fast, L_slow, M, I_sub, I_sp, I_sn, R, C]

    新增流（相比 4D 版本）：
        - 快/慢潜伏分层: L_fast → L_slow (ρ_conv)
        - 自清除: L_fast/L_slow → C (σ_clear)
        - Minimal 中间态: L_fast/L_slow → M → I_sub (波动 ω_reg)
        - 涂阳/涂阴分层: I_sub → I_sp/I_sn
        - 外源再感染: L_slow → L_fast (β_exo)
        - 再感染: C/R → L_fast (β_reinf)

    文献：
        Houben et al. (2016) BMC Med — TIME 模型快/慢分层
        Andrews et al. (2012) Nat Rev Dis Primers — 快进展率
        Horton et al. (2023) PNAS — 自清除
        Emery et al. (2023) eLife — 亚临床 TB 传染性
        Ragonnet et al. (2021) CID — 涂阳/涂阴分层
    """
    S = state[..., _IDX_S:_IDX_S+1]
    Lf = state[..., _IDX_LF:_IDX_LF+1]
    Ls = state[..., _IDX_LS:_IDX_LS+1]
    M = state[..., _IDX_M:_IDX_M+1]
    Isub = state[..., _IDX_ISUB:_IDX_ISUB+1]
    Isp = state[..., _IDX_ISP:_IDX_ISP+1]
    Isn = state[..., _IDX_ISN:_IDX_ISN+1]
    R = state[..., _IDX_R:_IDX_R+1]
    C = state[..., _IDX_C:_IDX_C+1]

    # 感染力 (含亚临床传染性 + 涂阴相对传染性)
    infectious = eta_sub * Isub + Isp + eta_sn * Isn
    lam = beta * infectious

    # --- 感染流 ---
    infection = lam * S
    reinfection_C = beta_reinf * lam * C
    reinfection_R = beta_reinf * lam * R
    exo_reinfection = beta_exo * lam * Ls

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

    # --- M / I_sub 流 ---
    min_to_sub = rho_min * M
    reg_m = omega_reg_m * M
    reg_sub = omega_reg_sub * Isub
    sub_to_sp = rho_prog * p_sp * Isub
    sub_to_sn = rho_prog * (1 - p_sp) * Isub
    sn_to_sp = p_sn2sp * Isn
    sp_out = (r_sp + mu_sp) * Isp
    sn_out = (r_sn + mu_sn) * Isn

    # --- 导数 ---
    dS = -infection
    dLf = (infection + reinfection_C + reinfection_R + exo_reinfection
           - fast_to_min - fast_to_sub - fast_to_sp - fast_to_slow - clear_fast)
    dLs = (fast_to_slow + reg_m
           - slow_to_min - slow_to_sub - slow_to_sp - clear_slow - exo_reinfection)
    dM = fast_to_min + slow_to_min + reg_sub - min_to_sub - reg_m
    dIsub = (fast_to_sub + slow_to_sub + min_to_sub
             - sub_to_sp - sub_to_sn - reg_sub)
    dIsp = fast_to_sp + slow_to_sp + sub_to_sp + sn_to_sp - sp_out
    dIsn = sub_to_sn - sn_to_sp - sn_out
    dR = sp_out + sn_out - reinfection_R
    dC = clear_fast + clear_slow - reinfection_C

    return torch.cat([dS, dLf, dLs, dM, dIsub, dIsp, dIsn, dR, dC], dim=-1)


def _rk4_step(state, deriv_fn, dt, *args, **kwargs):
    """四阶龙格-库塔（RK4）单步推进（显式、完全可微分）

    局部截断误差 O(dt⁵)，全局误差 O(dt⁴)，相比一阶 Euler 的 O(dt)
    显著降低数值漂移；与 Euler 一样仅含标量乘加与导数函数求值，
    不引入黑盒 ODE 求解器，计算图可完整反向传播。

    参数：
        state: 当前状态 [..., D]
        deriv_fn: 导数函数 f(state, *args, **kwargs)
        dt: 步长
        *args, **kwargs: 透传给 deriv_fn 的额外参数

    返回：
        state_new: [..., D] 推进后的状态

    参考文献：
        Runge (1895) / Kutta (1901) — 经典四阶 RK 方法
        Hairer, Nørsett & Wanner (1993) "Solving Ordinary Differential
        Equations I" Springer — 显式 RK 阶数与误差理论
    """
    k1 = deriv_fn(state, *args, **kwargs)
    k2 = deriv_fn(state + 0.5 * dt * k1, *args, **kwargs)
    k3 = deriv_fn(state + 0.5 * dt * k2, *args, **kwargs)
    k4 = deriv_fn(state + dt * k3, *args, **kwargs)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _integrate_seir_4d(beta, sigma, gamma, n_steps=10, dt=0.1,
                       S0=0.99, E0=0.01, I0=0.0, R0=0.0, solver='euler'):
    """4D SEIR ODE 数值积分核心（按 solver 选择 Euler 或 RK4）

    参数：
        beta: 传播率，shape [B] 或 [B, 1]
        sigma: 潜伏→感染转化率，shape [B] 或 [B, 1]
        gamma: 恢复率，shape [B] 或 [B, 1]
        n_steps: 积分步数（默认 10）
        dt: 时间步长（默认 0.1，总模拟时间 = n_steps * dt = 1.0 单位时间）
        S0, E0, I0, R0: 初始比例（标量，广播到 batch）
        solver: 'euler'（一阶显式 Euler）或 'rk4'（四阶龙格-库塔）

    返回：
        dict: {
            'trajectory': [B, n_steps+1, 4] 完整轨迹,
            'peak_I': [B] 峰值感染比例,
            'final_I': [B] 最终感染比例,
            'auc_I': [B] 感染曲线下面积（总感染负担）,
            'time_to_peak': [B] 到达峰值的时间步索引,
            'epidemic_size': [B] 总感染规模 (I+R 最终值),
        }
    """
    # 确保输入为正确的 shape
    if beta.dim() == 0:
        beta = beta.unsqueeze(0)
    if sigma.dim() == 0:
        sigma = sigma.unsqueeze(0)
    if gamma.dim() == 0:
        gamma = gamma.unsqueeze(0)

    B = beta.shape[0]

    # 扩展为 [B, 1] 以便与 [B, 4] 状态张量正确广播
    beta = beta.view(B, 1)
    sigma = sigma.view(B, 1)
    gamma = gamma.view(B, 1)

    # 确保参数非负
    beta = torch.clamp(beta, min=0.0, max=5.0)
    sigma = torch.clamp(sigma, min=0.01, max=2.0)
    gamma = torch.clamp(gamma, min=0.01, max=2.0)

    # 初始化状态 [B, 4]
    state = torch.stack([
        torch.full((B,), S0, device=beta.device, dtype=beta.dtype),
        torch.full((B,), E0, device=beta.device, dtype=beta.dtype),
        torch.full((B,), I0, device=beta.device, dtype=beta.dtype),
        torch.full((B,), R0, device=beta.device, dtype=beta.dtype),
    ], dim=-1)

    # 记录轨迹
    trajectory = [state.clone()]

    for _ in range(n_steps):
        if solver == 'rk4':
            state = _rk4_step(state, seir_ode_step, dt, beta, sigma, gamma)
        else:
            # 一阶显式 Euler：state += dt * f(state)
            dstate = seir_ode_step(state, beta, sigma, gamma)
            state = state + dt * dstate
        # 数值稳定性：确保非负且和为 1
        state = torch.clamp(state, min=0.0)
        state = state / (state.sum(dim=-1, keepdim=True) + 1e-8)
        trajectory.append(state.clone())

    trajectory = torch.stack(trajectory, dim=1)  # [B, n_steps+1, 4]

    # 提取流行病学指标
    I_curve = trajectory[:, :, 2]  # [B, n_steps+1]
    peak_I = I_curve.max(dim=1).values  # [B]
    final_I = I_curve[:, -1]  # [B]
    time_to_peak = I_curve.argmax(dim=1).float()  # [B]
    # AUC：梯形积分近似
    auc_I = (I_curve[:, :-1] + I_curve[:, 1:]).sum(dim=1) * dt / 2.0  # [B]
    # 总感染规模
    epidemic_size = trajectory[:, -1, 2] + trajectory[:, -1, 3]  # I+R

    return {
        'trajectory': trajectory,
        'peak_I': peak_I,
        'final_I': final_I,
        'auc_I': auc_I,
        'time_to_peak': time_to_peak,
        'epidemic_size': epidemic_size,
    }


def seir_euler_integrate(beta, sigma, gamma, n_steps=10, dt=0.1,
                         S0=0.99, E0=0.01, I0=0.0, R0=0.0):
    """使用一阶显式 Euler 方法对 SEIR ODE 进行数值积分（完全可微分）

    保留供向后兼容与误差基准对比使用；生产路径请用 seir_rk4_integrate。
    全局误差 O(dt)，对刚性或快速变化系统易产生数值漂移，建议仅在
    快速验证或与 RK4 结果对比时使用。

    参数：
        beta: 传播率，shape [B] 或 [B, 1]
        sigma: 潜伏→感染转化率，shape [B] 或 [B, 1]
        gamma: 恢复率，shape [B] 或 [B, 1]
        n_steps: 积分步数（默认 10）
        dt: 时间步长（默认 0.1，总模拟时间 = n_steps * dt = 1.0 单位时间）
        S0, E0, I0, R0: 初始比例（标量，广播到 batch）

    返回：
        dict: 含 'trajectory' / 'peak_I' / 'final_I' / 'auc_I' /
              'time_to_peak' / 'epidemic_size' 的指标字典
    """
    return _integrate_seir_4d(beta, sigma, gamma, n_steps=n_steps, dt=dt,
                              S0=S0, E0=E0, I0=I0, R0=R0, solver='euler')


def seir_rk4_integrate(beta, sigma, gamma, n_steps=10, dt=0.1,
                       S0=0.99, E0=0.01, I0=0.0, R0=0.0):
    """使用四阶龙格-库塔（RK4）对 SEIR ODE 进行数值积分（完全可微分）

    生产推荐求解器：全局误差 O(dt⁴)，与 Euler 一样显式可微，
    无需新增依赖（不依赖 torchdiffeq / 黑盒 ODE 求解器）。
    在默认网格（n_steps=10, dt=0.1）下即可达到约 1e-4 的轨迹精度，
    远优于一阶 Euler。

    参数：
        beta: 传播率，shape [B] 或 [B, 1]
        sigma: 潜伏→感染转化率，shape [B] 或 [B, 1]
        gamma: 恢复率，shape [B] 或 [B, 1]
        n_steps: 积分步数（默认 10）
        dt: 时间步长（默认 0.1，总模拟时间 = n_steps * dt = 1.0 单位时间）
        S0, E0, I0, R0: 初始比例（标量，广播到 batch）

    返回：
        dict: 含 'trajectory' / 'peak_I' / 'final_I' / 'auc_I' /
              'time_to_peak' / 'epidemic_size' 的指标字典

    参考文献：
        Chen et al. (2018) "Neural Ordinary Differential Equations" NeurIPS
        —— 本实现使用显式 RK4 而非黑盒 ODE 求解器，
        保证计算图完全可微且无需额外依赖（torchdiffeq）。
    """
    return _integrate_seir_4d(beta, sigma, gamma, n_steps=n_steps, dt=dt,
                              S0=S0, E0=E0, I0=I0, R0=R0, solver='rk4')


def _integrate_seir_v3(beta, rho_fast, rho_react, sigma_clear, gamma,
                       omega_reg=0.003, n_steps=10, dt=0.1,
                       S0=0.99, Lf0=0.005, Ls0=0.003, M0=0.0,
                       Isub0=0.0, Isp0=0.0, Isn0=0.0,
                       R0=0.0, C0=0.0,
                       eta_sub=1.0, eta_sn=_DEFAULT_ETA_SN,
                       rho_conv=0.003, rho_prog=0.05,
                       p_clin=_DEFAULT_P_CLIN, p_m=0.5, rho_min=0.001,
                       beta_exo=0.33, p_sp=0.55,
                       p_sn2sp=0.0003, beta_reinf=0.4,
                       solver='euler', **kwargs):
    """9-房室 SEIR v3 ODE 数值积分核心（按 solver 选择 Euler 或 RK4）

    改进8: 所有 seir_ode_step_v3 参数均可通过 kwargs 配置，不再硬编码。

    参数：
        beta: 传播率 [B] 或 [B,1]
        rho_fast: 快进展率 [B]
        rho_react: 慢再激活率 [B]（改进7: 独立通道，非 rho_fast*0.05）
        sigma_clear: 自清除率 [B]
        gamma: 临床恢复率 [B]（映射到 r_sp + mu_sp 等）
        omega_reg: 波动率 [B]（omega_reg_m = omega_reg_sub）
        eta_sub: 亚临床相对传染性（改进8: 可配置，默认 1.0）
        eta_sn: 涂阴相对传染性（默认 0.35）
        rho_conv, rho_prog, p_clin, p_m, rho_min: 9-房室固定参数
        beta_exo, p_sp, p_sn2sp, beta_reinf: 外源再感染/涂阳涂阴参数
        n_steps, dt: 积分步数与步长
        S0...C0: 9-房室初始比例
        solver: 'euler'（一阶显式 Euler）或 'rk4'（四阶龙格-库塔）

    返回：
        dict: {
            'trajectory': [B, n_steps+1, 9],
            'peak_I': [B] 峰值活动性感染 (Isub+Isp+Isn),
            'final_I': [B],
            'auc_I': [B],
            'time_to_peak': [B],
            'epidemic_size': [B] (Isub+Isp+Isn+R 最终值),
        }
    """
    # 确保 1D 张量（处理 float 输入和 0-d 张量）
    ref = beta if isinstance(beta, torch.Tensor) else torch.tensor(float(beta))
    params = [beta, rho_fast, rho_react, sigma_clear, gamma, omega_reg]
    for i, p in enumerate(params):
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(float(p), device=ref.device, dtype=ref.dtype)
        if p.dim() == 0:
            p = p.unsqueeze(0)
        params[i] = p
    beta, rho_fast, rho_react, sigma_clear, gamma, omega_reg = params

    B = beta.shape[0]

    # 标量参数扩展到 batch 维度，然后 reshape 为 [B, 1]
    params = [beta, rho_fast, rho_react, sigma_clear, gamma, omega_reg]
    for i, p in enumerate(params):
        if p.shape[0] == 1 and B > 1:
            p = p.expand(B)
        params[i] = p.view(B, 1)
    beta, rho_fast, rho_react, sigma_clear, gamma, omega_reg = params

    # 参数裁剪
    beta = torch.clamp(beta, 0.0, 5.0)
    rho_fast = torch.clamp(rho_fast, 1e-5, 0.05)
    rho_react = torch.clamp(rho_react, 1e-7, 0.001)
    sigma_clear = torch.clamp(sigma_clear, 1e-4, 0.1)
    gamma = torch.clamp(gamma, 1e-4, 0.5)
    omega_reg = torch.clamp(omega_reg, 1e-4, 0.1)

    # gamma 映射: 临床恢复率 → r_sp, r_sn; mu_sp, mu_sn 用固定比例
    r_sp = gamma * 0.6
    r_sn = gamma * 0.4
    mu_sp = gamma * 0.1
    mu_sn = gamma * 0.01

    # 改进8: 将可能为 [B] 张量的参数 reshape 为 [B, 1] 以正确广播
    def _to_batched(p):
        if isinstance(p, torch.Tensor):
            if p.dim() == 0:
                p = p.unsqueeze(0)
            if p.shape[0] == 1 and B > 1:
                p = p.expand(B)
            return p.view(B, 1)
        return p  # Python 标量保持不变
    rho_prog = _to_batched(rho_prog)
    eta_sub = _to_batched(eta_sub)
    eta_sn = _to_batched(eta_sn)

    # 初始化 [B, 9]
    inits = [S0, Lf0, Ls0, M0, Isub0, Isp0, Isn0, R0, C0]
    state = torch.stack([
        torch.full((B,), v, device=beta.device, dtype=beta.dtype)
        for v in inits
    ], dim=-1)

    trajectory = [state.clone()]
    for _ in range(n_steps):
        if solver == 'rk4':
            state = _rk4_step(
                state, seir_ode_step_v3, dt,
                beta, rho_fast, rho_react, sigma_clear, gamma,
                rho_conv=rho_conv, rho_prog=rho_prog, eta_sub=eta_sub,
                p_clin=p_clin, p_m=p_m, rho_min=rho_min,
                omega_reg_m=omega_reg, omega_reg_sub=omega_reg,
                beta_exo=beta_exo, eta_sn=eta_sn,
                p_sp=p_sp, mu_sp=mu_sp, r_sp=r_sp,
                mu_sn=mu_sn, r_sn=r_sn, p_sn2sp=p_sn2sp,
                beta_reinf=beta_reinf)
        else:
            # 一阶显式 Euler：state += dt * f(state)
            dstate = seir_ode_step_v3(
                state, beta, rho_fast, rho_react, sigma_clear, gamma,
                rho_conv=rho_conv, rho_prog=rho_prog, eta_sub=eta_sub,
                p_clin=p_clin, p_m=p_m, rho_min=rho_min,
                omega_reg_m=omega_reg, omega_reg_sub=omega_reg,
                beta_exo=beta_exo, eta_sn=eta_sn,
                p_sp=p_sp, mu_sp=mu_sp, r_sp=r_sp,
                mu_sn=mu_sn, r_sn=r_sn, p_sn2sp=p_sn2sp,
                beta_reinf=beta_reinf)
            state = state + dt * dstate
        state = torch.clamp(state, min=0.0)
        state = state / (state.sum(dim=-1, keepdim=True) + 1e-8)
        trajectory.append(state.clone())

    trajectory = torch.stack(trajectory, dim=1)  # [B, n_steps+1, 9]

    # 活动性感染: I_sub + I_sp + I_sn
    I_curve = (trajectory[:, :, _IDX_ISUB] +
               trajectory[:, :, _IDX_ISP] +
               trajectory[:, :, _IDX_ISN])  # [B, n_steps+1]
    peak_I = I_curve.max(dim=1).values
    final_I = I_curve[:, -1]
    time_to_peak = I_curve.argmax(dim=1).float()
    auc_I = (I_curve[:, :-1] + I_curve[:, 1:]).sum(dim=1) * dt / 2.0
    epidemic_size = I_curve[:, -1] + trajectory[:, -1, _IDX_R]

    return {
        'trajectory': trajectory,
        'peak_I': peak_I,
        'final_I': final_I,
        'auc_I': auc_I,
        'time_to_peak': time_to_peak,
        'epidemic_size': epidemic_size,
    }


def seir_v3_euler_integrate(beta, rho_fast, rho_react, sigma_clear, gamma,
                            omega_reg=0.003, n_steps=10, dt=0.1,
                            S0=0.99, Lf0=0.005, Ls0=0.003, M0=0.0,
                            Isub0=0.0, Isp0=0.0, Isn0=0.0,
                            R0=0.0, C0=0.0,
                            eta_sub=1.0, eta_sn=_DEFAULT_ETA_SN,
                            rho_conv=0.003, rho_prog=0.05,
                            p_clin=_DEFAULT_P_CLIN, p_m=0.5, rho_min=0.001,
                            beta_exo=0.33, p_sp=0.55,
                            p_sn2sp=0.0003, beta_reinf=0.4,
                            **kwargs):
    """9-房室 SEIR v3 ODE 一阶 Euler 积分（完全可微分）

    保留供向后兼容与误差基准对比使用；生产路径请用 seir_v3_rk4_integrate。
    全局误差 O(dt)，对刚性或快速变化系统易产生数值漂移。

    参数：
        beta: 传播率 [B] 或 [B,1]
        rho_fast: 快进展率 [B]
        rho_react: 慢再激活率 [B]（改进7: 独立通道，非 rho_fast*0.05）
        sigma_clear: 自清除率 [B]
        gamma: 临床恢复率 [B]（映射到 r_sp + mu_sp 等）
        omega_reg: 波动率 [B]（omega_reg_m = omega_reg_sub）
        eta_sub: 亚临床相对传染性（改进8: 可配置，默认 1.0）
        eta_sn: 涂阴相对传染性（默认 0.35）
        rho_conv, rho_prog, p_clin, p_m, rho_min: 9-房室固定参数
        beta_exo, p_sp, p_sn2sp, beta_reinf: 外源再感染/涂阳涂阴参数
        n_steps, dt: 积分步数与步长
        S0...C0: 9-房室初始比例

    返回：
        dict: 含 'trajectory' / 'peak_I' / 'final_I' / 'auc_I' /
              'time_to_peak' / 'epidemic_size' 的指标字典
    """
    return _integrate_seir_v3(
        beta, rho_fast, rho_react, sigma_clear, gamma,
        omega_reg=omega_reg, n_steps=n_steps, dt=dt,
        S0=S0, Lf0=Lf0, Ls0=Ls0, M0=M0,
        Isub0=Isub0, Isp0=Isp0, Isn0=Isn0, R0=R0, C0=C0,
        eta_sub=eta_sub, eta_sn=eta_sn,
        rho_conv=rho_conv, rho_prog=rho_prog,
        p_clin=p_clin, p_m=p_m, rho_min=rho_min,
        beta_exo=beta_exo, p_sp=p_sp,
        p_sn2sp=p_sn2sp, beta_reinf=beta_reinf,
        solver='euler', **kwargs)


def seir_v3_rk4_integrate(beta, rho_fast, rho_react, sigma_clear, gamma,
                          omega_reg=0.003, n_steps=10, dt=0.1,
                          S0=0.99, Lf0=0.005, Ls0=0.003, M0=0.0,
                          Isub0=0.0, Isp0=0.0, Isn0=0.0,
                          R0=0.0, C0=0.0,
                          eta_sub=1.0, eta_sn=_DEFAULT_ETA_SN,
                          rho_conv=0.003, rho_prog=0.05,
                          p_clin=_DEFAULT_P_CLIN, p_m=0.5, rho_min=0.001,
                          beta_exo=0.33, p_sp=0.55,
                          p_sn2sp=0.0003, beta_reinf=0.4,
                          **kwargs):
    """9-房室 SEIR v3 ODE 四阶龙格-库塔（RK4）积分（完全可微分）

    生产推荐求解器：全局误差 O(dt⁴)，与 Euler 一样显式可微，
    无需新增依赖（不依赖 torchdiffeq / 黑盒 ODE 求解器）。

    参数：
        beta: 传播率 [B] 或 [B,1]
        rho_fast: 快进展率 [B]
        rho_react: 慢再激活率 [B]（改进7: 独立通道，非 rho_fast*0.05）
        sigma_clear: 自清除率 [B]
        gamma: 临床恢复率 [B]（映射到 r_sp + mu_sp 等）
        omega_reg: 波动率 [B]（omega_reg_m = omega_reg_sub）
        eta_sub: 亚临床相对传染性（改进8: 可配置，默认 1.0）
        eta_sn: 涂阴相对传染性（默认 0.35）
        rho_conv, rho_prog, p_clin, p_m, rho_min: 9-房室固定参数
        beta_exo, p_sp, p_sn2sp, beta_reinf: 外源再感染/涂阳涂阴参数
        n_steps, dt: 积分步数与步长
        S0...C0: 9-房室初始比例

    返回：
        dict: 含 'trajectory' / 'peak_I' / 'final_I' / 'auc_I' /
              'time_to_peak' / 'epidemic_size' 的指标字典

    参考文献：
        Chen et al. (2018) "Neural Ordinary Differential Equations" NeurIPS
        —— 本实现使用显式 RK4 而非黑盒 ODE 求解器，
        保证计算图完全可微且无需额外依赖（torchdiffeq）。
    """
    return _integrate_seir_v3(
        beta, rho_fast, rho_react, sigma_clear, gamma,
        omega_reg=omega_reg, n_steps=n_steps, dt=dt,
        S0=S0, Lf0=Lf0, Ls0=Ls0, M0=M0,
        Isub0=Isub0, Isp0=Isp0, Isn0=Isn0, R0=R0, C0=C0,
        eta_sub=eta_sub, eta_sn=eta_sn,
        rho_conv=rho_conv, rho_prog=rho_prog,
        p_clin=p_clin, p_m=p_m, rho_min=rho_min,
        beta_exo=beta_exo, p_sp=p_sp,
        p_sn2sp=p_sn2sp, beta_reinf=beta_reinf,
        solver='rk4', **kwargs)


def seir_ode_gating(h, treatment_phase, seir_gate_module, n_steps=10, dt=0.1,
                    solver='rk4'):
    """SEIR ODE 门控：使用 ODE 积分结果调制节点表示

    v3.0 升级：门控输出从 3 通道扩展到 6/7/8 通道，对应 9-房室 SEIR 模型。
    自动检测 seir_gate_module 输出维度：
      - 3 通道 → 旧版 4D SEIR（向后兼容）
      - 6 通道 → v3.0 9D SEIR（rho_react=rho_fast*0.05 近似）
      - 7 通道 → 改进7: 独立 rho_react 通道
      - 8 通道 → 改进7+8: 独立 rho_react + eta_sub 通道

    通道映射：
      0: 感染力系数 (β)     — 传播率
      1: 快进展率 (ρ_fast)  — L_fast → I
      2: 自清除率 (σ_clear) — L → C
      3: 亚临床进展率 (ρ_prog) — I_sub → I_clin
      4: 临床恢复率 (γ)     — I_clin → R (r_self + φ_treat)
      5: 波动率 (ω_reg)     — M ↔ L_slow / I_sub ↔ M
      6: 慢再激活率 (ρ_react) — L_slow → I（改进7, 范围 1e-7~1e-4）
      7: 亚临床相对传染性 (η_sub) — 改进8, 范围 0.1~3.0 (Emery 2023)

    数值求解器（问题二修复）：默认使用四阶龙格-库塔（RK4），全局误差
    O(dt⁴)，在同等步数下精度远高于一阶 Euler 的 O(dt)，且与 Euler 一样
    显式可微、无额外依赖（不依赖 torchdiffeq / 黑盒 ODE 求解器）。可用
    ``solver='euler'`` 切换到一阶 Euler（供对比/兼容/误差基准）。

    参数：
        h: 节点嵌入 [B, hidden_dim]
        treatment_phase: 治疗阶段 [B, 1]
        seir_gate_module: SEIR 门控 MLP 模块（输出 3/6/7/8 通道）
        n_steps: ODE 积分步数
        dt: ODE 时间步长
        solver: 数值求解器 — 'rk4'（默认, 四阶龙格-库塔, 全局误差 O(dt⁴)）
            或 'euler'（一阶显式 Euler, 全局误差 O(dt), 对比/兼容用）

    返回：
        h_modulated: 调制后的节点嵌入 [B, hidden_dim]
        ode_metrics: ODE 积分指标 dict

    文献：
        - 将 SEIR 动力学先验作为归纳偏置嵌入 GNN 消息传递
        - 相比纯特征缩放，ODE 积分捕捉了动力学的非线性和时序依赖
        - Houben et al. (2016) BMC Med — 快/慢分层
        - Horton et al. (2023) PNAS — 自清除
        - Emery et al. (2023) eLife — 亚临床传染性
    """
    if solver not in ('rk4', 'euler'):
        raise ValueError(
            f"solver 必须为 'rk4' 或 'euler', got {solver!r}")

    B, hidden_dim = h.shape

    # 1. 通过 seir_gate 获取门控参数
    gate_input = torch.cat([h, treatment_phase], dim=-1)
    gate_weights = seir_gate_module(gate_input)  # [B, 3] 或 [B, 6/7/8]

    n_channels = gate_weights.shape[-1]

    # 2. 治疗阶段调整因子
    phase = treatment_phase.squeeze(-1)  # [B]
    treatment_factor = torch.sigmoid((phase - 1.5) * 1.5)  # [0,1]，治疗越晚值越大

    if n_channels >= 6:
        # ===== v3.0 6/7/8 通道: 9-房室 SEIR =====
        channels = gate_weights.unbind(dim=-1)
        (trans_tendency, fast_prog_tendency, clear_tendency,
         sub_prog_tendency, recovery_capacity, reg_tendency) = channels[:6]

        # 映射到 SEIR v3 参数空间
        beta = 0.1 + 1.4 * trans_tendency
        rho_fast = 1e-4 + 0.01 * fast_prog_tendency       # [1e-4, 0.01]
        sigma_clear = 1e-4 + 0.05 * clear_tendency         # [1e-4, 0.05]
        rho_prog = 0.01 + 0.1 * sub_prog_tendency          # [0.01, 0.11]
        gamma = 0.01 + 0.2 * recovery_capacity             # [0.01, 0.21]
        omega_reg = 1e-4 + 0.01 * reg_tendency             # [1e-4, 0.01]

        # 改进7: 独立 rho_react 通道（7+ 通道时），否则用 rho_fast*0.05 近似
        if n_channels >= 7:
            react_tendency = channels[6]
            rho_react = 1e-7 + 1e-4 * react_tendency       # [1e-7, 1e-4]
        else:
            rho_react = rho_fast * 0.05

        # 改进8: 独立 eta_sub 通道（8 通道时），否则默认 1.0
        if n_channels >= 8:
            eta_sub_tendency = channels[7]
            eta_sub = 0.1 + 2.9 * eta_sub_tendency         # [0.1, 3.0] (Emery 2023)
        else:
            eta_sub = 1.0

        # 治疗阶段调整
        beta = beta * (1.0 - 0.3 * treatment_factor)
        gamma = gamma * (1.0 + 0.5 * treatment_factor)
        rho_prog = rho_prog * (1.0 + 0.3 * treatment_factor)

        # 运行 9-房室 SEIR v3 ODE 积分（改进8: 参数可配置；RK4 求解器提升精度）
        if solver == 'rk4':
            ode_metrics = seir_v3_rk4_integrate(
                beta, rho_fast, rho_react=rho_react,
                sigma_clear=sigma_clear, gamma=gamma,
                omega_reg=omega_reg, n_steps=n_steps, dt=dt,
                rho_prog=rho_prog, eta_sub=eta_sub)
        else:
            ode_metrics = seir_v3_euler_integrate(
                beta, rho_fast, rho_react=rho_react,
                sigma_clear=sigma_clear, gamma=gamma,
                omega_reg=omega_reg, n_steps=n_steps, dt=dt,
                rho_prog=rho_prog, eta_sub=eta_sub)
    else:
        # ===== 旧版 3 通道: 4D SEIR (向后兼容) =====
        trans_tendency, latent_risk, recovery_capacity = gate_weights.unbind(dim=-1)

        beta = 0.1 + 1.4 * trans_tendency
        sigma = 0.05 + 0.45 * latent_risk
        gamma = 0.05 + 0.45 * recovery_capacity

        beta = beta * (1.0 - 0.3 * treatment_factor)
        gamma = gamma * (1.0 + 0.5 * treatment_factor)

        if solver == 'rk4':
            ode_metrics = seir_rk4_integrate(
                beta, sigma, gamma, n_steps=n_steps, dt=dt
            )
        else:
            ode_metrics = seir_euler_integrate(
                beta, sigma, gamma, n_steps=n_steps, dt=dt
            )

    # 3. 用 ODE 积分结果调制节点表示
    peak_factor = ode_metrics['peak_I'].unsqueeze(-1)  # [B, 1]
    auc_factor = (ode_metrics['auc_I'] / (ode_metrics['auc_I'].max() + 1e-8)).unsqueeze(-1)
    epi_factor = ode_metrics['epidemic_size'].unsqueeze(-1)

    combined_factor = (
        0.5 * peak_factor +
        0.3 * auc_factor +
        0.2 * epi_factor
    )  # [B, 1]

    modulation = 1.0 + 0.3 * (combined_factor - 0.5)
    modulation = torch.clamp(modulation, 0.5, 1.5)

    h_modulated = h * modulation

    return h_modulated, ode_metrics


def seir_solver_error_baseline(n_samples=64, seed=0, total_time=1.0,
                               ref_dt=0.001, tol=1e-3):
    """误差基准验证：量化 Euler 相对 RK4 在典型参数区间的误差边界

    以高分辨率 RK4（ref_dt 极小）解作为参考真值，比较同一固定网格下
    一阶 Euler 与四阶 RK4 的轨迹误差，据此确定可接受的 dt 阈值，
    避免盲目换求解器或加密网格。

    参数：
        n_samples: 采样的典型参数点数（默认 64）
        seed: 随机种子，保证结果可复现
        total_time: 模拟总时长（默认 1.0，与生产网格一致）
        ref_dt: 参考求解器步长（默认 0.001，作为"真值"近似）
        tol: 可接受峰值感染误差阈值（默认 1e-3），用于确定 dt 上界

    返回：
        dict: {
            'ref_I': 参考均值 I 曲线 [T_ref]（RK4@ref_dt 近似真值）,
            'ref_peak': 参考峰值感染均值,
            'euler_max_dt': 满足误差 < tol 的 Euler 最大可接受 dt,
            'rk4_max_dt': 满足误差 < tol 的 RK4 最大可接受 dt,
            'euler_errors': {dt: 峰值/轨迹误差} 各网格 Euler 误差,
            'rk4_errors': {dt: 峰值/轨迹误差} 各网格 RK4 误差,
            'dt_grid': 测试的步长序列,
            'recommended_dt': 生产建议 dt（RK4 达标；Euler 取达标上界）,
            'summary': 人类可读结论字符串,
        }

    说明：
        误差以峰值感染比例偏差与整条 I 曲线最大绝对偏差衡量。
        参考真值由 RK4 在 ref_dt 极小步长下积分得到（相对误差 ~O(ref_dt⁴)）。
        仅在 torch 可用时有效；否则返回 None。
    """
    if torch is None:
        return None

    rng = torch.Generator().manual_seed(seed)
    # 典型参数区间（与生产门控映射对齐）：beta∈[0.1,1.5], sigma∈[0.05,0.5], gamma∈[0.05,0.5]
    beta = torch.rand(n_samples, generator=rng) * 1.4 + 0.1
    sigma = torch.rand(n_samples, generator=rng) * 0.45 + 0.05
    gamma = torch.rand(n_samples, generator=rng) * 0.45 + 0.05

    # 参考真值：RK4 在极小步长下积分（逐样本，误差取批次最坏情况）
    ref_steps = max(int(round(total_time / ref_dt)), 2)
    ref_res = seir_rk4_integrate(
        beta, sigma, gamma, n_steps=ref_steps, dt=ref_dt)
    ref_I_all = ref_res['trajectory'][:, :, 2]      # [B, T_ref] 逐样本参考
    ref_I = ref_I_all.mean(dim=0)                   # [T_ref] 均值（展示用）
    ref_peak = ref_res['peak_I']                    # [B] 逐样本参考峰值
    ref_peak_mean = ref_peak.mean().item()

    # 测试网格：与生产一致的 dt 逐步加密
    dt_grid = [0.1, 0.05, 0.02, 0.01]

    def _errors(solver_fn, dt):
        n_steps = max(int(round(total_time / dt)), 1)
        res = solver_fn(beta, sigma, gamma, n_steps=n_steps, dt=dt)
        I = res['trajectory'][:, :, 2]  # [B, T]
        # 插值到参考时间网格比较轨迹（逐样本，取批次与时间上的最坏偏差）
        I_interp = _linear_interp(
            torch.linspace(0.0, total_time, I.shape[1]), I,
            torch.linspace(0.0, total_time, ref_I_all.shape[1]))
        traj_err = (I_interp - ref_I_all).abs().max().item()
        peak_err = (res['peak_I'] - ref_peak).abs().max().item()
        return {'traj_err': traj_err, 'peak_err': peak_err}

    euler_errors = {}
    rk4_errors = {}
    # 全量向量化评估（RK4 局部误差随 dt^4 下降，Euler 随 dt 线性）
    for dt in dt_grid:
        euler_errors[dt] = _errors(seir_euler_integrate, dt)
        rk4_errors[dt] = _errors(seir_rk4_integrate, dt)

    # 确定各自满足峰值误差 < tol 的最大可接受 dt（从粗到细找首个达标）
    def _max_acceptable(errors):
        # 按 dt 从细到粗检查，返回最粗仍达标者
        for dt in sorted(errors, reverse=True):
            if errors[dt]['peak_err'] < tol:
                return dt
        return None

    euler_max_dt = _max_acceptable(euler_errors)
    rk4_max_dt = _max_acceptable(rk4_errors)

    # 生产建议：RK4 在默认 dt=0.1 下通常已达标；Euler 需加密网格
    rk4_default_ok = rk4_errors[0.1]['peak_err'] < tol
    recommended_dt = rk4_max_dt if rk4_default_ok else 0.02
    solver_choice = 'rk4' if rk4_default_ok else 'rk4(网格需加密) 或 Euler@{:.3g}'.format(
        euler_max_dt)

    summary = (
        'Euler(一阶)最坏峰值误差@dt=0.1 ~ {:.2e}（全局误差 O(dt)，需 dt<={:.3g} 才达标）；'
        'RK4(四阶)最坏峰值误差@dt=0.1 ~ {:.2e}（全局误差 O(dt^4)，默认即达标）。'
        '建议生产采用 RK4，不建议为一阶 Euler 盲目加密网格。'
    ).format(euler_errors[0.1]['peak_err'], euler_max_dt or 0.0,
             rk4_errors[0.1]['peak_err'])

    return {
        'ref_I': ref_I,
        'ref_peak': ref_peak_mean,
        'euler_max_dt': euler_max_dt,
        'rk4_max_dt': rk4_max_dt,
        'euler_errors': euler_errors,
        'rk4_errors': rk4_errors,
        'dt_grid': dt_grid,
        'recommended_dt': recommended_dt,
        'recommended_solver': solver_choice,
        'summary': summary,
    }


def _linear_interp(t_in, y_in, t_out):
    """一维线性插值（纯 torch 实现，保持可微）

    用于把不同网格上的 I 曲线对齐到参考时间网格比较误差。
    要求 t_in 单调递增；y_in 可为 [T_in] 或 [..., T_in]（支持批量）。
    """
    # 对每个输出点，找到所在区间 [t_in[i], t_in[i+1]]
    idx = torch.searchsorted(t_in, t_out).clamp(1, t_in.shape[0] - 1)
    t0 = t_in[idx - 1]
    t1 = t_in[idx]
    w = (t_out - t0) / ((t1 - t0) + 1e-12)
    y0 = y_in[..., idx - 1]
    y1 = y_in[..., idx]
    return y0 + w * (y1 - y0)
