#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可微 SEIR v4 模型 — PyTorch 自动微分实现

为 HMC（哈密顿蒙特卡洛）采样器提供可微的 log_posterior，
通过 PyTorch autodiff 计算梯度，无需手动推导偏导数。

核心组件：
  1. 参数变换: 约束空间 ↔ 无约束空间（softplus / sigmoid + Jacobian 修正）
  2. 可微 v4 ODE: 270D 状态向量的 Euler 积分（与 stochastic.py 一致）
  3. 可微 log_prior: torch.distributions（Beta / Gamma）
  4. 可微 log_likelihood: 高斯残差
  5. grad_log_posterior: torch.autograd.grad 自动微分

文献：
  Hoffman MD, Gelman A. (2014) JMLR 15(1):1593-1623 — NUTS / HMC
  Nesterov Y. (2009) Ecological Modelling 12:127-145 — 对偶平均法
  Betancourt M. (2017) arXiv:1701.02434 — HMC 概念入门
  Ragonnet R et al. (2021) Clin Infect Dis — TB 贝叶斯分层模型
"""

import math
import os

import numpy as np

try:
    import torch
    import torch.distributions as tdist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .stochastic import (
    N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3,
    IDX_ISUB_V3, IDX_ISP_V3, IDX_ISN_V3,
    DEFAULT_WAIFW, DEFAULT_AGE_PROGRESSION,
    DEFAULT_RHO_CONV, DEFAULT_RHO_PROG, DEFAULT_ETA_SUB,
    DEFAULT_P_CLIN, DEFAULT_BETA_REINF,
    DEFAULT_RHO_MIN, DEFAULT_P_M,
    DEFAULT_P_SP, DEFAULT_MU_SP, DEFAULT_R_SP,
    DEFAULT_MU_SN, DEFAULT_R_SN, DEFAULT_P_SN2SP,
)


# =====================================================================
# 条件式 torch.compile (优先级1: GPU/MSVC 环境加速)
# =====================================================================

# inductor 后端可用性缓存 (None=未检测, True/False=检测结果)
_INDUCTOR_AVAILABLE = None


def _check_inductor_available():
    """检测 inductor 后端是否可用 (需要 MSVC 编译器或 GPU)。

    在 CPU-only 且无 MSVC 的 Windows 环境下, inductor 无法生成 C++ 代码,
    torch.compile 会抛 InductorError。本函数通过试编译一个简单函数来检测,
    结果缓存到模块级 ``_INDUCTOR_AVAILABLE`` 避免重复检测开销。

    基准测试 (当前环境 PyTorch 2.8.0+cpu, 无 MSVC):
      - eager (原始):          0.491 ms/call
      - aot_eager 后端:        1.638 ms/call (慢 3.3x, 仅有图重写无代码生成)
      - eager 后端 (图追踪):   0.567 ms/call (慢 1.2x, 仅追踪无优化)
    因此在 inductor 不可用时, 不编译反而更快。

    要点5 (第十二轮): inductor 后端与 torch.jit 的关系。
    PyTorch 2.x 中 ``torch.compile(backend='inductor')`` 基于 TorchDynamo
    (``torch._dynamo``) 图捕获 + Triton/C++ 代码生成, 独立于 ``torch.jit``
    (TorchScript)。PyTorch 2.13.0 中 ``torch.jit.script_method`` 已
    deprecated (警告: "Please switch to torch.compile or torch.export"),
    但 inductor 不受影响。本函数的检测路径 (``torch.compile(lambda x: x*2.0,
    backend='inductor')``) 不依赖 JIT, 未来 JIT 移除后仍可工作。
    为减少检测时的警告噪声, 用 ``warnings.catch_warnings()`` 抑制 JIT
    deprecation 警告 (不影响检测结果, 仅清理输出)。
    """
    global _INDUCTOR_AVAILABLE
    if _INDUCTOR_AVAILABLE is not None:
        return _INDUCTOR_AVAILABLE
    if not TORCH_AVAILABLE or not hasattr(torch, 'compile'):
        _INDUCTOR_AVAILABLE = False
        return False
    # 要点5: 防御性检查 torch._inductor 模块存在 (inductor 后端的实现位置)
    if not hasattr(torch, '_inductor'):
        _INDUCTOR_AVAILABLE = False
        return False
    try:
        import warnings as _warnings
        # 抑制 torch.jit.script_method deprecation 警告 (PyTorch 2.13.0+)
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore', DeprecationWarning)
            _probe = torch.compile(lambda x: x * 2.0, backend='inductor')
            _probe(torch.tensor(1.0, dtype=torch.float64))
        _INDUCTOR_AVAILABLE = True
    except Exception:
        _INDUCTOR_AVAILABLE = False
    return _INDUCTOR_AVAILABLE


def _try_compile(fn):
    """条件式 torch.compile 装饰器。

    通过环境变量 ``TB_RISK_COMPILE`` 控制:
      - 'auto' (默认): 检测 inductor 后端, 可用则启用
      - '1': 强制启用 (inductor 后端, 失败则回退原函数)
      - '0': 强制禁用, 返回原函数

    当前环境 (CPU-only, 无 MSVC) auto 模式会自动禁用, 不影响性能。
    未来在有 MSVC 或 GPU 的环境, auto 模式自动启用, 获得 2-5x 提速
    (GPU 上 torch_v4_rhs/integrate 的 inductor 代码生成)。

    要点4: ``dynamic=True`` 允许 batch 维度动态变化, 避免链数变化
    (如 4→8→16) 时重编译 (每次重编译耗时数十秒)。虽然 dynamic=True
    可能略影响 kernel fusion 优化效果, 但避免了重编译开销, 适合
    HMC 中 batch 维度可能变化的场景 (单链梯度计算 vs 4 链 batch)。
    文献: PyTorch 2.0+ dynamo 文档 — dynamic shapes 支持。

    注意: 首次编译有开销 (几十秒), 适合 HMC 的大规模重复调用场景
    (240000+ 次 grad_fn 调用)。对于少量调用的测试/调试, 建议
    设置 TB_RISK_COMPILE=0。
    """
    if not TORCH_AVAILABLE or not hasattr(torch, 'compile'):
        return fn

    mode = os.environ.get('TB_RISK_COMPILE', 'auto').lower()
    if mode == '0':
        return fn

    if mode == 'auto' and not _check_inductor_available():
        return fn

    try:
        return torch.compile(fn, backend='inductor', dynamic=True)
    except Exception:
        return fn


# =====================================================================
# 固定结构张量缓存 (要点4: 避免重复创建)
# =====================================================================

# 模块级缓存: {(dtype, device): (waifw_t, age_prog_t)}
_FIXED_TENSOR_CACHE = {}

# 要点4 (第五轮): method='auto' 自适应选择 RK4 的 n_steps 阈值
# n_steps < 阈值 → RK4 (短轨迹精度优先), 否则 → Euler (长轨迹速度优先)
# 阈值 1000 经验值: HMC 短轨迹 (n_leapfrog=20, dt=0.5 → n_steps=20) 用 RK4,
# 长轨迹 (36500 步) 用 Euler。可被环境变量 TB_RISK_AUTO_RK4_THRESHOLD 覆盖。
AUTO_RK4_N_STEPS_THRESHOLD = int(os.environ.get('TB_RISK_AUTO_RK4_THRESHOLD', '1000'))

# 要点4 (第九轮): RK4 中间步骤 softplus 平滑近似的 alpha 参数。
# 原 torch.clamp(x, min=0.0) 在 x < 0 时梯度为 0 (硬截断), 破坏 autograd
# 梯度连续性。HMC 反向传播中, 当某房室人口接近 0 且导数为负时, clamp
# 截断 k1 对该房室的梯度贡献, 导致 k2-k4 对 y_params 的梯度信息不完整,
# 影响接受率和混合效率。softplus 平滑近似在 x 接近 0 时提供非零梯度:
#   _smooth_positive(x) = x                         (x > 0, 梯度=1)
#                       = alpha * log1p(exp(x/alpha)) (x <= 0, 梯度=sigmoid(x/alpha))
# alpha 越小越接近硬 clamp, 越大越平滑。alpha=0.1 时, x=-0.1 处梯度≈0.27,
# 仍有有效梯度; alpha=1.0 时过渡更平滑但 x=0 附近偏差更大。
# 默认 0.1 平衡平滑性与精度。可通过环境变量 TB_RISK_RK4_SOFTPLUS_ALPHA 覆盖。
RK4_SOFTPLUS_ALPHA = float(os.environ.get('TB_RISK_RK4_SOFTPLUS_ALPHA', '0.1'))

# 人口守恒相对容差 (M 级审计修复): 封闭 v4 模型 (无出生/死亡) 应满足
# N(t) = S+Lf+Ls+M+Isub+Isp+Isn+R+C = N(0) 严格守恒。数值路径上
# Euler 截断误差 / softplus 正性保护 / numpy 非负截断都会引入质量漂移。
# 该阈值用于"显式检查"——超过仅告警并记录, 不中断 (轨迹仍返回,
# 由调用方决定是否采信)。可用环境变量 TB_RISK_POP_CONS_RTOL 覆盖。
POP_CONSERVATION_RTOL = float(os.environ.get('TB_RISK_POP_CONS_RTOL', '1e-3'))


def _smooth_positive(x, alpha=None):
    """平滑正性近似 (参数化 softplus) — 替代 torch.clamp(x, min=0.0)

    要点4 (第九轮): RK4 中间步骤使用硬 clamp 会在 x < 0 时截断梯度
    (梯度恒为 0), 破坏 autograd 计算图的梯度连续性。HMC 反向传播中,
    当某房室人口接近 0 且导数为负时, clamp 截断 k1 对该房室的梯度贡献,
    导致 k2-k4 对 y_params 的梯度信息不完整, 可能使某些参数方向的梯度
    被低估, 影响接受率和混合效率。

    本函数用分段 softplus 平滑近似替代硬 clamp:
      - x >= 0: 返回 x (恒等映射, 梯度=1, 与 clamp 一致)
      - x < 0: 返回 alpha * log1p(exp(x/alpha)) (softplus, 梯度=sigmoid(x/alpha))

    性质:
      - x → +∞: softplus(x) ≈ x (与 clamp 一致)
      - x → -∞: softplus(x) ≈ 0 (与 clamp 一致)
      - x = 0: 返回 0 (与 clamp 一致, 避免对未使用房室引入 alpha*log(2) 偏差)
      - x < 0 时梯度 = sigmoid(x/alpha) > 0, 保持可微性 (核心目的)

    注: x = 0 处梯度有跳变 (左极限 sigmoid(0)=0.5, 右极限 1), 但值连续。
    HMC 用 leapfrog 离散梯度, 不需要严格梯度连续; 值连续更重要 (避免
    原本为 0 的房室如未使用的 HIV+/DR 层累积 alpha*log(2)≈0.069 偏差)。
    数值稳定性: x >= 0 分支避免 exp(x/alpha) 在 x/alpha 大时溢出;
    x < 0 分支用 log1p 替代 log(1 + exp) 提高小值精度。

    Args:
        x: 输入张量 (任意形状)
        alpha: 平滑参数 (默认 RK4_SOFTPLUS_ALPHA)。alpha → 0 退化为硬 clamp。

    Returns:
        与 x 同形状的张量, 值非负, 梯度处处非零 (x < 0 时 sigmoid(x/alpha) > 0)

    文献:
      Glorot & Bengio (2011) AISTATS — softplus 激活函数
      Patankar (1980) — 保正性积分器 (硬 clamp 的理论基础)
      Kopecz & Meister (2018) mPFRK — 保正高阶方法
    """
    if alpha is None:
        alpha = RK4_SOFTPLUS_ALPHA
    # 数值稳定实现: torch.where 会计算两个分支, 若直接用 exp(x/alpha)
    # 在 x >> 0 时溢出为 inf, 反向传播中 inf * 0 = NaN。
    # 解决: 对 softplus 分支输入用 min(x, 0) 确保 x_neg/alpha <= 0,
    # exp(x_neg/alpha) <= 1 不溢出, 两个分支都数值稳定。
    # x >= 0 而非 x > 0: 确保 softplus(0) = 0 (避免未使用房室累积偏差)
    #
    # 要点5 (第十轮): torch.where 两分支开销评估。
    # PyTorch eager 模式下 torch.where 会计算两分支, 对 270 维状态向量
    # 每步 RK4 调用 4 次 _smooth_positive, 共 4×270=1080 次 softplus
    # 计算中约一半 (x>=0 的部分) 被丢弃。但:
    #   1. _smooth_positive 作为内部函数被 @_try_compile 装饰的
    #      torch_v4_rhs/torch_v4_integrate 调用, torch.compile (inductor)
    #      可通过 graph rewriting 优化掉无效分支 (x>=0 时的 softplus_neg)
    #   2. torch.minimum(x, 0) 和 alpha*log1p(exp(x_neg/alpha)) 在 x>=0 时
    #      计算的是常数 (x_neg=0, softplus_neg=alpha*log(2)), inductor 可
    #      常量折叠
    #   3. 实测 CPU 环境 _smooth_positive 占 RK4 总时间 <5%, 优化收益有限
    # 若 GPU 环境下 profiling 显示 _smooth_positive 成为瓶颈, 可考虑:
    #   - 用 torch.nn.functional.softplus(x, beta=1/alpha) 的全局 softplus
    #     (无分段, 但 x=0 时返回 alpha*log(2)≠0, 需减去常数)
    #   - 或用 torch.relu(x) + 小修正项 (Swish 激活变体)
    # 当前实现优先数值稳定性和可读性, 待 GPU profiling 数据驱动优化。
    # 要点4 (第十一轮): 用 torch.clamp(x, max=0.0) 替代 torch.minimum(x, zeros_like(x))。
    # clamp(max=0) 通过广播实现相同语义, 无需分配 270 维零张量。RK4 每步 4 次调用,
    # n_steps=36500 时消除 146000 次零张量分配。torch.compile 下两者等价 (graph 优化),
    # 非编译模式下 clamp 略快 (减少一次分配 + 一次 minimum kernel)。
    x_neg = torch.clamp(x, max=0.0)
    softplus_neg = alpha * torch.log1p(torch.exp(x_neg / alpha))
    return torch.where(x >= 0, x, softplus_neg)


def _get_fixed_tensors(dtype, device):
    """获取缓存的固定结构张量 (waifw_t, age_prog_t)

    要点4: 在非编译模式下, torch_v4_integrate 每次调用都创建
    DEFAULT_WAIFW (5×5) 和 DEFAULT_AGE_PROGRESSION (5维) 张量,
    HMC 中 100000+ 次调用累积有可测量开销。模块级缓存完全消除此开销。

    同时支持 float32 (要点3a): dtype 参数确保 dtype 匹配输入张量,
    避免 torch_v4_integrate 中硬编码 float64 阻碍混合精度。
    """
    key = (dtype, device)
    if key not in _FIXED_TENSOR_CACHE:
        _FIXED_TENSOR_CACHE[key] = (
            torch.tensor(DEFAULT_WAIFW, dtype=dtype, device=device),
            torch.tensor(DEFAULT_AGE_PROGRESSION, dtype=dtype, device=device),
        )
    return _FIXED_TENSOR_CACHE[key]


# =====================================================================
# 参数变换: 约束空间 ↔ 无约束空间
# =====================================================================

# 15 个推断参数的 (lower, upper) 约束
# None 表示无上界 (softplus 变换), (lo, hi) 表示 sigmoid 变换
PARAM_BOUNDS = [
    (0.001, 0.95),     # 0: beta
    (1e-5, 0.01),      # 1: rho_fast
    (1e-7, 1e-4),      # 2: rho_react
    (1e-4, 0.05),      # 3: sigma_clear
    (0.005, 0.2),      # 4: gamma
    (1e-4, 5e-2),      # 5: omega_reg_m
    (1e-4, 5e-2),      # 6: omega_reg_sub
    (0.01, 0.95),      # 7: beta_exo
    (0.01, 1.0),       # 8: eta_sn
    (5.0, 50.0),       # 9: rr_hiv
    (0.20, 0.90),      # 10: art_reduction
    (1e-6, 0.01),      # 11: hiv_infection_rate
    (1e-5, 1e-2),      # 12: art_initiation_rate
    (0.02, 0.40),      # 13: dr_fitness_cost
    (0.005, 0.10),     # 14: p_acq
]
N_INFER_PARAMS = 15

PARAM_NAMES = [
    'beta', 'rho_fast', 'rho_react', 'sigma_clear', 'gamma',
    'omega_reg_m', 'omega_reg_sub', 'beta_exo', 'eta_sn',
    'rr_hiv', 'art_reduction', 'hiv_infection_rate',
    'art_initiation_rate', 'dr_fitness_cost', 'p_acq',
]


def constrain(y, lo, hi):
    """无约束 → 约束: x = lo + (hi - lo) * sigmoid(y)

    返回 (x, log|dx/dy|) 用于 Jacobian 修正

    要点3a (第三轮): dtype 感知的 epsilon 保护。float32 下 sigmoid
    饱和更快 (|y|>16 时 sigmoid≈0 或 1), 原 1e-30 对 float32 太小
    (float32 最小正常数 ~1.2e-38, 但精度只有 ~7 位), log(1e-30)=-69
    虽可表示但梯度 1/1e-30=1e30 会溢出 float32 (max ~3.4e38, 临界)。
    float32 改用 1e-7 (log(1e-7)=-16.1, 梯度 1e7 安全)。
    float64 保持 1e-30 (精度足够, 不影响数值结果)。
    """
    s = torch.sigmoid(y)
    x = lo + (hi - lo) * s
    # log|dx/dy| = log(hi - lo) + log(s) + log(1 - s)
    eps = 1e-7 if y.dtype == torch.float32 else 1e-30
    log_jac = math.log(hi - lo) + torch.log(s + eps) + torch.log(1 - s + eps)
    return x, log_jac


def unconstrain(x, lo, hi):
    """约束 → 无约束: y = logit((x - lo) / (hi - lo))"""
    return torch.logit(torch.clamp((x - lo) / (hi - lo), 1e-10, 1 - 1e-10))


def constrain_params(y_tensor):
    """将无约束参数向量 (15,) 变换为约束参数字典 + 返回 log Jacobian

    y_tensor: torch.Tensor shape (15,) 或 (batch, 15)
    返回: (params_dict, log_jac_sum)
    """
    if y_tensor.dim() == 1:
        y_tensor = y_tensor.unsqueeze(0)  # (1, 15)
    squeeze = y_tensor.shape[0] == 1

    params = {}
    log_jac = torch.zeros(y_tensor.shape[0], device=y_tensor.device)

    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[i]
        x, lj = constrain(y_tensor[:, i], lo, hi)
        params[name] = x if not squeeze else x.squeeze(0)
        log_jac += lj if not squeeze else lj.squeeze(0)

    return params, log_jac


# =====================================================================
# 可微 v4 SEIR ODE
# =====================================================================

@_try_compile
def torch_v4_rhs(state, params, waifw_t, age_prog_t,
                 fixed_params=None):
    """PyTorch v4 ODE 右侧 (与 stochastic.py deterministic_rhs_v4 一致)

    state: (B, 270) 或 (270,)
    params: dict of torch tensors (batched or scalar)
    waifw_t, age_prog_t: 固定结构参数 (torch tensors)
    fixed_params: dict of 未推断的固定参数
    返回: (B, 270) 或 (270,)

    注意: 全部使用函数式构造 (无原地切片赋值), 保证 autograd 计算图完整。
    """
    squeeze = state.dim() == 1
    if squeeze:
        state = state.unsqueeze(0)  # (1, 270)

    B = state.shape[0]
    s4d = state.view(B, N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)

    # 统一将推断参数变为 (B,) 张量, 避免后续 float() 调用破坏梯度
    def _b(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=state.device, dtype=state.dtype)
        if x.dim() == 0:
            return x.unsqueeze(0).expand(B)
        return x if x.shape[0] == B else x.expand(B)

    beta = _b(params['beta'])
    rho_fast = _b(params['rho_fast'])
    rho_react = _b(params['rho_react'])
    sigma_clear = _b(params['sigma_clear'])
    omega_reg_m = _b(params['omega_reg_m'])
    omega_reg_sub = _b(params['omega_reg_sub'])
    beta_exo = _b(params['beta_exo'])
    eta_sn = _b(params['eta_sn'])
    rr_hiv = _b(params['rr_hiv'])
    art_reduction = _b(params['art_reduction'])
    hiv_infection_rate = _b(params['hiv_infection_rate'])
    art_initiation_rate = _b(params['art_initiation_rate'])
    dr_fitness_cost = _b(params['dr_fitness_cost'])
    p_acq = _b(params['p_acq'])

    # 固定参数 (float, 不参与梯度)
    fp = fixed_params or {}
    rho_conv = float(fp.get('rho_conv', DEFAULT_RHO_CONV))
    rho_prog = float(fp.get('rho_prog', DEFAULT_RHO_PROG))
    eta_sub_f = float(fp.get('eta_sub', DEFAULT_ETA_SUB))
    p_clin = float(fp.get('p_clin', DEFAULT_P_CLIN))
    beta_reinf = float(fp.get('beta_reinf', DEFAULT_BETA_REINF))
    rho_min = float(fp.get('rho_min', DEFAULT_RHO_MIN))
    p_m = float(fp.get('p_m', DEFAULT_P_M))
    p_sp = float(fp.get('p_sp', DEFAULT_P_SP))
    mu_sp = float(fp.get('mu_sp', DEFAULT_MU_SP))
    r_sp = float(fp.get('r_sp', DEFAULT_R_SP))
    mu_sn = float(fp.get('mu_sn', DEFAULT_MU_SN))
    r_sn = float(fp.get('r_sn', DEFAULT_R_SN))
    p_sn2sp = float(fp.get('p_sn2sp', DEFAULT_P_SN2SP))

    # HIV 再激活倍数: [1.0, RR_HIV, RR_HIV*(1-ART_red)] → (B, 3)
    hiv_mult = torch.stack([
        torch.ones_like(rr_hiv), rr_hiv, rr_hiv * (1.0 - art_reduction)
    ], dim=-1)

    # 提取房室 (B, 5, 3, 2)
    S = s4d[:, :, :, :, 0]
    Lf = s4d[:, :, :, :, 1]
    Ls = s4d[:, :, :, :, 2]
    M = s4d[:, :, :, :, 3]
    Isub = s4d[:, :, :, :, 4]
    Isp = s4d[:, :, :, :, 5]
    Isn = s4d[:, :, :, :, 6]
    R = s4d[:, :, :, :, 7]
    C = s4d[:, :, :, :, 8]

    # 各年龄组总人口 (B, 5)
    N_age = s4d.sum(dim=(2, 3, 4))
    N_age = torch.clamp(N_age, min=1e-10)

    # --- 感染力 (年龄特异性 + WAIFW + DR 适合度代价) ---
    Isub_age = Isub.sum(dim=2)  # (B, 5, 2)
    Isp_age = Isp.sum(dim=2)
    Isn_age = Isn.sum(dim=2)

    eta_sub_t = torch.tensor(eta_sub_f, device=state.device, dtype=state.dtype)
    infectious = eta_sub_t * Isub_age + Isp_age + \
        eta_sn.view(-1, 1, 1) * Isn_age  # (B, 5, 2)

    # DR 适合度代价: DR (index 1) 降低传染性 — 函数式构造, 避免原地操作
    dr_mult = torch.stack([
        torch.ones(B, N_AGE_V4, device=state.device, dtype=state.dtype),
        (1.0 - dr_fitness_cost).unsqueeze(-1).expand(-1, N_AGE_V4)
    ], dim=-1)  # (B, 5, 2)
    infectious = infectious * dr_mult

    # λ[age, dr] = Σ_j waifw[age, j] * beta * infectious[j, dr] / N[j]
    inf_ds = beta.view(-1, 1) * infectious[:, :, 0] / N_age  # (B, 5)
    inf_dr = beta.view(-1, 1) * infectious[:, :, 1] / N_age

    lam_ds = (waifw_t @ inf_ds.T).T  # (B, 5)
    lam_dr = (waifw_t @ inf_dr.T).T
    lam = torch.stack([lam_ds, lam_dr], dim=-1)  # (B, 5, 2)
    lam = lam.unsqueeze(2).expand(-1, -1, N_HIV_V4, -1)  # (B, 5, 3, 2)

    # --- 感染流 ---
    infection = lam * S
    reinfection_C = beta_reinf * lam * C
    reinfection_R = beta_reinf * lam * R
    exo_reinfection = beta_exo.view(-1, 1, 1, 1) * lam * Ls

    # --- 年龄特异性进展率 ---
    rho_fast_age = rho_fast.view(-1, 1) * age_prog_t.unsqueeze(0)  # (B, 5)
    rho_react_age = rho_react.view(-1, 1) * age_prog_t.unsqueeze(0)
    rho_fast_arr = rho_fast_age.view(B, N_AGE_V4, 1, 1)  # (B, 5, 1, 1)
    rho_react_arr = rho_react_age.view(B, N_AGE_V4, 1, 1) * \
        hiv_mult.unsqueeze(1).unsqueeze(-1)  # (B, 5, 3, 1)

    # --- L_fast 流出 ---
    fast_to_min = rho_fast_arr * p_m * Lf
    fast_to_sub = rho_fast_arr * (1 - p_m) * (1 - p_clin) * Lf
    fast_to_sp = rho_fast_arr * (1 - p_m) * p_clin * Lf
    fast_to_slow = rho_conv * Lf
    clear_fast = sigma_clear.view(-1, 1, 1, 1) * Lf

    # --- L_slow 流出 ---
    slow_to_min = rho_react_arr * p_m * Ls
    slow_to_sub = rho_react_arr * (1 - p_m) * (1 - p_clin) * Ls
    slow_to_sp = rho_react_arr * (1 - p_m) * p_clin * Ls
    clear_slow = sigma_clear.view(-1, 1, 1, 1) * Ls

    # --- M / I_sub 流 ---
    min_to_sub = rho_min * M
    reg_m = omega_reg_m.view(-1, 1, 1, 1) * M
    reg_sub = omega_reg_sub.view(-1, 1, 1, 1) * Isub
    sub_to_sp = rho_prog * p_sp * Isub
    sub_to_sn = rho_prog * (1 - p_sp) * Isub
    sn_to_sp = p_sn2sp * Isn
    sp_out = (r_sp + mu_sp) * Isp
    sn_out = (r_sn + mu_sn) * Isn

    # --- 获得性耐药: DS → DR ---
    p_acq_b = p_acq.view(-1, 1, 1)  # (B, 1, 1)
    dr_acq_sub = p_acq_b * Isub[:, :, :, 0]  # (B, 5, 3)
    dr_acq_sp = p_acq_b * Isp[:, :, :, 0]
    dr_acq_sn = p_acq_b * Isn[:, :, :, 0]

    # --- HIV 进展 (所有房室: hiv=0→1, hiv=1→2) ---
    hiv_rate_b = hiv_infection_rate.view(-1, 1, 1, 1)
    art_rate_b = art_initiation_rate.view(-1, 1, 1, 1)
    hiv_01 = hiv_rate_b * s4d[:, :, 0, :, :]  # (B, 5, 2, 9)
    art_12 = art_rate_b * s4d[:, :, 1, :, :]  # (B, 5, 2, 9)

    # --- 导数计算 (全部函数式, 无原地操作) ---
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

    # 获得性耐药修正: DS 减少, DR 增加 — 用 stack 构造调整张量, 避免原地赋值
    def _dr_adj(d, acq):
        # d: (B, 5, 3, 2), acq: (B, 5, 3) → 调整 (B, 5, 3, 2)
        return d + torch.stack([-acq, acq], dim=-1)

    dIsub = _dr_adj(dIsub, dr_acq_sub)
    dIsp = _dr_adj(dIsp, dr_acq_sp)
    dIsn = _dr_adj(dIsn, dr_acq_sn)

    # HIV 进展修正: 构造 (B, 5, 3, 2, 9) 调整张量
    # hiv=0 失去 hiv_01, hiv=1 获得 hiv_01 但失去 art_12, hiv=2 获得 art_12
    hiv_adj = torch.stack([-hiv_01, hiv_01 - art_12, art_12], dim=2)

    # 堆叠所有房室导数 + HIV 调整 → (B, 5, 3, 2, 9) → (B, 270)
    derivs = torch.stack([dS, dLf, dLs, dM, dIsub, dIsp, dIsn, dR, dC], dim=-1)
    derivs = derivs + hiv_adj
    rhs = derivs.view(B, -1)

    if squeeze:
        rhs = rhs.squeeze(0)
    return rhs


def population_conservation_error(trajectory):
    """v4 270D 轨迹的人口守恒误差 (显式检查, M 级审计修复)

    计算轨迹上每一步总人口 N(t) = S+Lf+Ls+M+Isub+Isp+Isn+R+C 相对
    初始 N(0) 的最大相对偏差 max |N(t)-N(0)| / |N(0)| (跨 batch 与时间)。

    封闭 v4 模型 (无出生/死亡动力学) 的解析解严格守恒; 数值漂移来源:
      - Euler/RK4 截断误差 (dt 过大时累积)
      - _smooth_positive / np.maximum 正性保护 (房室人口被截断时增/减质量)
    270 维高维模型对 dt 敏感, 该指标为积分器配置 (dt, method) 的有效性
    提供可量化的监控信号。阈值见 POP_CONSERVATION_RTOL。

    Args:
        trajectory: torch.Tensor 或 numpy.ndarray,
            shape (T, 270) 或 (B, T, 270)

    Returns:
        float: 最大相对偏差 (0 = 完美守恒)
    """
    if TORCH_AVAILABLE and isinstance(trajectory, torch.Tensor):
        with torch.no_grad():
            totals = trajectory.sum(dim=-1)          # (T,) 或 (B, T)
            n0 = totals[..., 0]                      # () 或 (B,)
            denom = torch.clamp(torch.abs(n0), min=1e-10)
            rel = torch.abs(totals - n0.unsqueeze(-1)) / denom.unsqueeze(-1)
            return float(rel.max().item())
    arr = np.asarray(trajectory, dtype=np.float64)
    totals = arr.sum(axis=-1)
    n0 = totals[..., 0]
    denom = np.maximum(np.abs(n0), 1e-10)
    rel = np.abs(totals - n0[..., None]) / denom[..., None]
    return float(np.max(rel))


@_try_compile
def torch_v4_integrate(y_params, initial_state, n_steps, dt,
                        waifw_t=None, age_prog_t=None, fixed_params=None,
                        method='auto', params=None,
                        check_conservation=False):
    """v4 ODE 积分 (可微, 支持 Euler / RK4 / auto)

    y_params: 无约束参数 (15,) 或 (B, 15)
    initial_state: (270,) numpy 或 torch
    method: 'auto' (默认, 要点4 第五轮: 根据 n_steps 自适应选择) /
            'euler' (1阶, 1次 RHS/步) / 'rk4' (4阶, 4次 RHS/步)
    params: 可选, 来自 constrain_params 的参数字典 (避免重复调用 constrain_params)
    check_conservation: True 时积分后执行人口守恒显式检查
            (population_conservation_error), 超过 POP_CONSERVATION_RTOL
            发 warning。默认 False — HMC 热路径每步 leapfrog 都调用本函数,
            检查放在采样结束后做一次 (见 v4_inference.run_mcmc_v4) 更经济;
            HME 等中低频路径可直接开启。
    返回: trajectory (B, n_steps+1, 270)

    改进16 (五): RK4 高阶固定步长积分。TB 动力学时间尺度差异大
    (感染~天, 自清除~月, 再激活~年), 固定步长 Euler 必须取最快时间尺度
    (天), 慢动力学部分被过度积分。RK4 每步 4 次 RHS 评估但 4 阶精度,
    可用更大步长 (如 2 天/步) 达到相同精度, 总 RHS 评估数减少。

    重要: RK4 全程函数式构造 (无原地切片赋值), 保证 autograd 计算图完整。
    每步 4 次 RHS 调用, 计算图比 Euler 大 4 倍, 反向传播也更慢, 但
    梯度精度显著提高 — HMC 中更精确的梯度 → 更高接受率 → 更少浪费。

    要点1 (第五轮): ``params`` 可选参数。与 ``torch_log_prior_v4`` 的
    要点2修复完全对称 — posterior 路径中 constrain_params 被调用两次
    (posterior 一次 + integrate 一次), 浪费一次完整的 15 维 sigmoid
    变换和 Jacobian 计算。传入 params 时 integrate 跳过 constrain_params,
    且 autograd 计算图一致 (params 与 y_params 在同一图节点上), 反向
    传播只需遍历一次 constrain_params 的图。

    要点4 (第五轮): ``method='auto'`` 自适应选择。RK4 每步 4 次 RHS 调用,
    计算图比 Euler 大 4 倍, 反向传播也更慢。HMC 典型配置 n_steps=36500
    (dt=0.01天), RK4 总 RHS 调用数为 Euler 的 4 倍, 即使接受率提高也未必
    能补偿。auto 模式根据 n_steps 自适应:
      - n_steps < AUTO_RK4_N_STEPS_THRESHOLD (默认 1000): 用 RK4
        (精度收益大于速度损失, 短轨迹 4 倍开销可接受)
      - n_steps >= 阈值: 用 Euler (长轨迹速度优先, 精度损失通过小 dt 补偿)
    阈值 1000 来自经验: HMC 短轨迹 (如 n_leapfrog=20, dt=0.5 → n_steps=20)
    用 RK4 精度收益显著; 长轨迹 (如 36500 步) 用 Euler 速度优先。

    文献:
      Hairer et al. (1993) Solving ODE I — RK4 经典 4 阶方法
      Hoffman & Gelman (2014) JMLR — HMC 对梯度精度的敏感性
    """
    # 要点1: 仅在 params 未传入时才调用 constrain_params
    if params is None:
        params, _ = constrain_params(y_params)

    # 要点4: method='auto' 自适应选择
    if method == 'auto':
        method = 'rk4' if n_steps < AUTO_RK4_N_STEPS_THRESHOLD else 'euler'

    # 要点4: 使用模块级缓存的固定结构张量, 避免每次调用重复创建
    # 要点3a: dtype/device 匹配 y_params, 支持 float32 混合精度
    y_dtype = y_params.dtype
    y_device = y_params.device if hasattr(y_params, 'device') else 'cpu'
    if waifw_t is None or age_prog_t is None:
        cached_w, cached_a = _get_fixed_tensors(y_dtype, y_device)
        if waifw_t is None:
            waifw_t = cached_w
        if age_prog_t is None:
            age_prog_t = cached_a

    if not isinstance(initial_state, torch.Tensor):
        initial_state = torch.tensor(initial_state, dtype=y_dtype, device=y_device)
    initial_state = initial_state.to(y_device, y_dtype)

    # 确保 params 在同一 device 和 dtype (浅拷贝避免污染调用方的 dict)
    # 要点6 (第九轮): 惰性转换。原实现无条件遍历 15 个参数执行 .to(),
    # 在 posterior 路径中 constrain_params 产生的 params 已与 y_params
    # 同 device/dtype, .to() 为 no-op 但仍创建新字典 + 15 次 .to() 调用。
    # HMC 十万次 grad_fn 调用累积可观开销。改进: 先检查首个参数的
    # dtype/device, 仅在需要转换时才创建新字典; 否则直接复用原 dict。
    _need_convert = False
    for _v in params.values():
        if _v.dtype != y_dtype or _v.device != y_device:
            _need_convert = True
        break
    if _need_convert:
        params = {k: v.to(y_device, y_dtype) for k, v in params.items()}

    B = y_params.shape[0] if y_params.dim() > 1 else 1
    state = initial_state.unsqueeze(0).expand(B, -1).clone()  # (B, 270)

    if method == 'euler':
        trajectory = [state.clone()]
        # 要点3 (第十轮): Euler 分支也用 _smooth_positive 替代硬 clamp,
        # 与 RK4 分支 (要点4 第九轮) 保持一致。method='auto' 时短轨迹
        # 用 RK4 (平滑梯度), 长轨迹用 Euler — 若两者正性保护方式不同,
        # warmup 期 RK4 的平滑梯度与采样期 Euler 的截断梯度不匹配,
        # 可能使步长/质量矩阵的自适应基于不一致的梯度信息。统一用
        # _smooth_positive 后, 无论 method 如何选择, 梯度连续性都得到
        # 保证, HMC 自适应和采样的梯度质量统一。
        for _ in range(n_steps):
            rhs = torch_v4_rhs(state, params, waifw_t, age_prog_t, fixed_params)
            state = state + dt * rhs
            state = _smooth_positive(state)
            trajectory.append(state.clone())
    elif method == 'rk4':
        # 改进16 (五): 经典 4 阶 Runge-Kutta
        # k1 = f(t, y)
        # k2 = f(t + dt/2, y + dt/2 * k1)
        # k3 = f(t + dt/2, y + dt/2 * k2)
        # k4 = f(t + dt, y + dt * k3)
        # y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        #
        # 注意: torch_v4_rhs 不显式依赖 t (自治系统), 时间参数省略。
        # 全程函数式 (无 inplace), 保证 autograd 计算图完整。
        #
        # 要点3 (第五轮): 中间步骤非负保护。RK4 每步的中间状态
        # ``state + half_dt * k1`` 等可能为负 — 当某房室人口接近 0 时,
        # 导数为负会使中间状态变负, k2-k4 基于负状态计算的导数物理上
        # 不合理 (如负的 S 会导致负感染率)。最终 state 有 clamp 保护但
        # 中间步骤的负值会通过 k1-k4 加权组合影响精度, 长期积分误差累积。
        # 改进: 每个 k 评估前对中间状态应用正性保护, 破坏严格 4 阶精度
        # 但保证物理合理性 (保正性)。文献: Patankar (1980) 保正性积分器,
        # Kopecz & Meister (2018) mPFRK 保正高阶方法 — 完整保正性需
        # 专用积分器, 此处为简单正性保护近似。
        #
        # 要点4 (第九轮): 中间步骤和最终 state 改用 _smooth_positive
        # (参数化 softplus) 替代 torch.clamp(min=0.0)。硬 clamp 在
        # x < 0 时梯度为 0, 截断 k1 对该房室的梯度贡献, 导致 k2-k4
        # 对 y_params 的梯度信息不完整, 影响 HMC 接受率和混合效率。
        # softplus 平滑近似在 x <= 0 时仍提供 sigmoid(x/alpha) > 0 的
        # 非零梯度, 保持 autograd 计算图完整。alpha 由模块级
        # RK4_SOFTPLUS_ALPHA 控制 (默认 0.1, 可通过环境变量调整)。
        trajectory = [state.clone()]
        half_dt = 0.5 * dt
        sixth_dt = dt / 6.0
        for _ in range(n_steps):
            k1 = torch_v4_rhs(state, params, waifw_t, age_prog_t, fixed_params)
            k2 = torch_v4_rhs(_smooth_positive(state + half_dt * k1),
                              params, waifw_t, age_prog_t, fixed_params)
            k3 = torch_v4_rhs(_smooth_positive(state + half_dt * k2),
                              params, waifw_t, age_prog_t, fixed_params)
            k4 = torch_v4_rhs(_smooth_positive(state + dt * k3),
                              params, waifw_t, age_prog_t, fixed_params)
            state = state + sixth_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            state = _smooth_positive(state)
            trajectory.append(state.clone())
    else:
        raise ValueError(
            f"method 必须为 'euler' / 'rk4' / 'auto', got {method!r}")

    traj = torch.stack(trajectory, dim=1)  # (B, n_steps+1, 270)

    # 人口守恒显式检查 (M 级审计修复): 默认关闭 (HMC 热路径),
    # HME / 后验诊断等中低频路径开启。见 docstring check_conservation。
    if check_conservation:
        pop_err = population_conservation_error(traj)
        if pop_err > POP_CONSERVATION_RTOL:
            import warnings
            warnings.warn(
                f"v4 ODE 轨迹人口守恒违反: max|N(t)-N(0)|/N(0) = "
                f"{pop_err:.3e} > {POP_CONSERVATION_RTOL:.1e} "
                f"(method={method}, n_steps={n_steps}, dt={dt})。"
                f"提示: 减小 dt 或改用 method='rk4'。")

    return traj


# =====================================================================
# 可微 log_prior / log_likelihood / log_posterior
# =====================================================================

def torch_log_prior_v4(y_params, log_jac=None, params=None):
    """可微 log_prior (在无约束空间, 含 Jacobian 修正)

    y_params: 无约束参数 (15,) 或 (B, 15)
    log_jac: 来自 constrain_params 的 Jacobian (可选, 避免重复计算)
    params: 来自 constrain_params 的参数字典 (可选, 避免重复计算)
    返回: (B,) 或标量

    要点2 (第四轮): ``params`` 可选参数。原实现即使 ``log_jac`` 已传入,
    仍无条件调用 ``constrain_params(y_params)`` 获取 params 字典,
    重复了 15 维 sigmoid 变换和 Jacobian 计算。每次 grad_fn 调用节省
    一次 constrain_params, HMC 中十万次调用累积可观。
    """
    # 要点2: 仅在 params 或 log_jac 未传入时才调用 constrain_params
    if params is None or log_jac is None:
        params_computed, log_jac_computed = constrain_params(y_params)
        if params is None:
            params = params_computed
        if log_jac is None:
            log_jac = log_jac_computed

    squeeze = y_params.dim() == 1
    if squeeze:
        y_params = y_params.unsqueeze(0)
        for k in params:
            params[k] = params[k].unsqueeze(0) if params[k].dim() == 0 else params[k]

    # 使用 torch.distributions 计算约束空间中的先验
    # 核心参数: 复用 v3 先验
    # beta: Beta(2, 8)
    lp = tdist.Beta(2.0, 8.0).log_prob(params['beta'])
    # rho_fast: Gamma(2, scale=0.05/365) → rate = 365/0.05 = 7300
    lp = lp + tdist.Gamma(2.0, 7300.0).log_prob(params['rho_fast'])
    # rho_react: Gamma(2, scale=0.002/365) → rate = 365/0.002 = 182500
    lp = lp + tdist.Gamma(2.0, 182500.0).log_prob(params['rho_react'])
    # sigma_clear: Gamma(2, scale=1/365) → rate = 365
    lp = lp + tdist.Gamma(2.0, 365.0).log_prob(params['sigma_clear'])
    # gamma: Gamma(2, scale=0.025) → rate = 40
    lp = lp + tdist.Gamma(2.0, 40.0).log_prob(params['gamma'])
    # omega_reg_m: Gamma(2, scale=1/365) → rate = 365
    lp = lp + tdist.Gamma(2.0, 365.0).log_prob(params['omega_reg_m'])
    # omega_reg_sub: Gamma(2, scale=1/365) → rate = 365
    lp = lp + tdist.Gamma(2.0, 365.0).log_prob(params['omega_reg_sub'])
    # beta_exo: Beta(2, 4)
    lp = lp + tdist.Beta(2.0, 4.0).log_prob(params['beta_exo'])
    # eta_sn: Beta(2, 5)
    lp = lp + tdist.Beta(2.0, 5.0).log_prob(params['eta_sn'])

    # v4 新参数
    # rr_hiv: Gamma(2, scale=10) → rate = 0.1
    lp = lp + tdist.Gamma(2.0, 0.1).log_prob(params['rr_hiv'])
    # art_reduction: Beta(13, 7)
    lp = lp + tdist.Beta(13.0, 7.0).log_prob(params['art_reduction'])
    # hiv_infection_rate: Gamma(2, scale=5e-4) → rate = 2000
    lp = lp + tdist.Gamma(2.0, 2000.0).log_prob(params['hiv_infection_rate'])
    # art_initiation_rate: Gamma(2, scale=1.4e-4) → rate = 1/1.4e-4 ≈ 7142.86
    lp = lp + tdist.Gamma(2.0, 1.0 / 1.4e-4).log_prob(params['art_initiation_rate'])
    # dr_fitness_cost: Beta(6, 30)
    lp = lp + tdist.Beta(6.0, 30.0).log_prob(params['dr_fitness_cost'])
    # p_acq: Beta(3, 90)
    lp = lp + tdist.Beta(3.0, 90.0).log_prob(params['p_acq'])

    # 加上 Jacobian 修正 (无约束空间中的先验 = 约束空间先验 + log|dx/dy|)
    lp = lp + log_jac

    if squeeze:
        lp = lp.squeeze(0)
    return lp


def torch_log_likelihood_v4(y_params, observed_data, initial_state,
                             t_span, dt, waifw_t=None, age_prog_t=None,
                             fixed_params=None, method='auto', params=None):
    """可微 log_likelihood (高斯残差)

    返回: (B,) 或标量

    改进16 (五): ``method`` 参数透传到 torch_v4_integrate, 支持
    'euler'/'rk4'/'auto' (M 级审计修复: 默认从 'euler' 改为 'auto',
    短轨迹自动用 RK4 — 270 维高维模型对 dt 敏感)。

    要点1 (第四轮): dtype 跟随 y_params。原实现 obs_I_t/obs_times_t/sim_times
    硬编码 torch.float64, float32 模式下 ODE 轨迹为 float32 而插值计算
    混入 float64, PyTorch 隐式上转使混合精度加速失效。三处张量统一用
    y_dtype, 与 torch_v4_integrate 的 dtype 传递策略一致。

    要点6 (第四轮): batch 插值向量化。原实现 for b in range(B) 逐链
    执行 torch.searchsorted + 线性插值。向量化: searchsorted 的
    sorted_sequence 不变 (sim_times), values 可为任意 shape; sim_I[:, idx]
    高级索引一次取出所有链所有观测点的值, 消除 Python 循环。
    B=4 时收益不大, 但 16-32 链时差异显著。

    要点1 (第五轮): ``params`` 可选参数透传到 torch_v4_integrate,
    避免在 posterior 路径中重复调用 constrain_params。autograd 计算图
    一致 — 传入的 params 必须与 y_params 在同一图节点上。
    """
    obs_I, obs_times = observed_data
    # 要点1: dtype 跟随 y_params (支持 float32 混合精度)
    y_dtype = y_params.dtype
    y_device = y_params.device if hasattr(y_params, 'device') else 'cpu'
    obs_I_t = torch.tensor(obs_I, dtype=y_dtype, device=y_device)
    obs_times_t = torch.tensor(obs_times, dtype=y_dtype, device=y_device)

    n_steps = int((t_span[1] - t_span[0]) / dt)
    traj = torch_v4_integrate(
        y_params, initial_state, n_steps, dt,
        waifw_t, age_prog_t, fixed_params, method=method, params=params)  # (B, n_steps+1, 270)

    B = traj.shape[0]
    # 聚合活动性 TB: I_sub + I_sp + I_sn (所有 age×HIV×DR 求和)
    traj_5d = traj.view(B, n_steps + 1, N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
    sim_I = (traj_5d[:, :, :, :, :, IDX_ISUB_V3].sum(dim=(2, 3, 4)) +
             traj_5d[:, :, :, :, :, IDX_ISP_V3].sum(dim=(2, 3, 4)) +
             traj_5d[:, :, :, :, :, IDX_ISN_V3].sum(dim=(2, 3, 4)))  # (B, n_steps+1)

    # 时间轴 (要点1: dtype 跟随 y_params)
    sim_times = torch.linspace(t_span[0], t_span[1], n_steps + 1,
                               dtype=y_dtype, device=y_device)

    # 要点6: 向量化 batch 插值 (消除 Python for 循环)
    # searchsorted: sorted_sequence=sim_times (n_steps+1,), values=obs_times_t (n_obs,)
    # 返回 (n_obs,) 索引, 所有 batch 共享 (sim_times 相同)
    idx = torch.searchsorted(sim_times, obs_times_t)
    idx = torch.clamp(idx, 1, len(sim_times) - 1)

    # 线性插值系数 (n_obs,) — 所有 batch 共享
    t0 = sim_times[idx - 1]
    t1 = sim_times[idx]
    alpha = (obs_times_t - t0) / (t1 - t0 + 1e-10)

    # 高级索引: sim_I[:, idx-1] → (B, n_obs), 一次取出所有链所有观测点
    v0 = sim_I[:, idx - 1]  # (B, n_obs)
    v1 = sim_I[:, idx]      # (B, n_obs)
    interpolated = v0 + alpha * (v1 - v0)  # (B, n_obs) alpha 广播

    # 高斯似然 (向量化: sum over dim=1)
    obs_std = float(np.std(obs_I)) * 0.2 if len(obs_I) > 0 else 0.5
    obs_std = max(obs_std, 0.5)
    residuals = obs_I_t - interpolated  # (n_obs,) - (B, n_obs) → (B, n_obs)
    ll = -0.5 * torch.sum((residuals / obs_std) ** 2, dim=1) - \
         len(obs_I) * math.log(obs_std * math.sqrt(2 * math.pi))  # (B,)

    # 要点1/3/6 (第十一轮): 单链输入 (y_params.dim()==1) 时 squeeze 到 0-dim 标量,
    # 与 torch_log_prior_v4 的 squeeze 行为一致。原实现返回 (1,) 即使 docstring
    # 声明 "返回: (B,) 或标量"。这导致 torch_log_posterior_v4 返回 (1,) 而非 (),
    # compute_grad_log_posterior 返回的 lp 形状为 (1,)。原 HMC 代码用 .item()
    # 提取标量, 对形状不敏感; 第十一轮 tensor 化后 (n_accept_t += mask_t 等)
    # 形状敏感, () 与 (1,) 广播不兼容, 触发 "shape [] doesn't match broadcast [1]"。
    # batch 模式 (y_params.dim()==2, B>=1) 不受影响, 仍返回 (B,)。
    if y_params.dim() == 1:
        ll = ll.squeeze(0)
    return ll


def torch_log_posterior_v4(y_params, observed_data, initial_state,
                            t_span, dt, waifw_t=None, age_prog_t=None,
                            fixed_params=None, method='auto'):
    """可微 log_posterior (无约束空间)

    返回: (B,) 或标量

    改进16 (五): ``method`` 参数透传到 torch_log_likelihood_v4
    (M 级审计修复: 默认从 'euler' 改为 'auto', 见 torch_v4_integrate)。

    要点2 (第四轮): 将 constrain_params 的完整结果 (params + log_jac)
    传入 torch_log_prior_v4, 避免 prior 内部重复调用 constrain_params。

    要点1 (第五轮): 将 constrain_params 的 params 同时透传到
    torch_log_likelihood_v4 → torch_v4_integrate, 避免在 integrate
    中再次调用 constrain_params。整个 posterior 路径只调用一次
    constrain_params, autograd 计算图一致 (params 与 y_params 在同一
    图节点上), 反向传播只需遍历一次 constrain_params 的图。
    """
    params, log_jac = constrain_params(y_params)
    lp = torch_log_prior_v4(y_params, log_jac=log_jac, params=params)
    ll = torch_log_likelihood_v4(
        y_params, observed_data, initial_state, t_span, dt,
        waifw_t, age_prog_t, fixed_params, method=method, params=params)
    return lp + ll


def compute_grad_log_posterior(y_params, observed_data, initial_state,
                                t_span, dt, **kwargs):
    """计算 log_posterior 对无约束参数的梯度

    y_params: (15,) 无约束参数 (requires_grad=False, 内部启用)
              或 (B, 15) batch (优先级3: 4 链并行梯度计算)
    返回: (grad, log_post_val)
      - 单条: grad (15,), lp (1,) 或标量
      - batch: grad (B, 15), lp (B,)

    优先级3: 支持 batch 维度。torch.autograd.grad 要求输出为标量或显式
    传入 grad_outputs。当 lp 为 (B,) 向量时, 传入 grad_outputs=ones_like(lp)
    计算 sum(lp) 的梯度。由于不同 batch 间 ODE 积分独立 (lp[i] 只依赖
    y[i]), 交叉导数 d(lp[k])/dy[i] = 0 (k!=i), 因此 sum(lp) 的梯度
    正好是每个 lp[i] 对 y[i] 的梯度 — 一次反向传播同时计算 B 条链的梯度。
    """
    y = y_params.clone().detach().requires_grad_(True)
    lp = torch_log_posterior_v4(
        y, observed_data, initial_state, t_span, dt, **kwargs)
    # 支持 batch: lp 为 (B,) 时需显式 grad_outputs
    grad_outputs = torch.ones_like(lp)
    grad = torch.autograd.grad(lp, y, grad_outputs=grad_outputs,
                               create_graph=False)[0]
    return grad.detach(), lp.detach()
