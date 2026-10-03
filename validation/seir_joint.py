#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4a：SEIR 参数联合学习（可微分 RK4 反传，用户方案 2）+ 窗形状消融。

背景（用户 2026-08-25 "下一步优化" P4）：
  "进行 SEIR 参数联合学习（可微分方案 2/3）"——组合 DGP（combined_
  network）已刻意预埋**可学习的物理失配**：
    - 参数失配：真值 β/I0 抖动 ±20%，物理基线用文献默认值；
    - 形状失配：真值窗口化 Λ(t_w)（SEIR 轨迹 5 个窗中点时点力），
      物理基线平坦 Λ_ref（8 周单值，"均匀暴露假设"）。

学习器（方案 2 = 特征级联合）：
    可学习 (δβ, δI0)（exp 残差参数化，默认值起点）
      → 可微 RK4 积分 80 步（与 pi_network._seir_rk4_numpy 逐步镜像：
        每步 clamp 非负 + 重归一化，均可微）
      → 窗口中点感染力 Λ̂(t_w) = β̂·I(t_w) [5]
      → λ̂_i = host_i · corr_i · Σ_w Λ̂(t_w)·decay(w)·E_{i,w}
        （E = window_exposure，P1 特征列——部署可观测；
          decay = 8 周半衰期文献锚点，沿用 temporal_network）
      → z = k·λ̂,  p = 1−exp(−z),  L = BCE(logit, y_train)
    k 不进梯度：每 50 epoch 用训练半区二分重校准（交替优化）——
    消除 β↔k 尺度不可辨识（Λ 整体缩放 c 与 k/c 完全等价，联合
    梯度会在等价流形上漂移，参数恢复诊断失真）。

对照臂 free_windows（结构先验价值的裁决）：
    5 个自由 log Λ_w 直接学习（无 SEIR 动力学约束），同起点
    （默认参数窗值）、同 k 交替校准。seir 臂 2 参数重构整条
    Λ(t) 轨迹（5 窗点 + 参考点全部由 (β,I0) 决定）——物理结构
    约束的样本效率假设：n=400 下 2 参数应优于 5 自由参数。

五臂消融：
    seir_default   k̂·λ_default（默认参数平坦 Λ_ref，部署现状）
    seir_joint     学习 (β̂,Î0) → 窗口化 λ̂（方案 2 主体）
    free_windows   学习 5 自由 Λ_w（无结构对照）
    oracle_window  真值 β/I0 的窗口化 λ（参数完美已知上界——
                   仍不含类型 β_t 与源 s_c，与 oracle_nu 的差
                   = 类型 + 源异质性剩余间隙）
    oracle_nu      真值 ν（三机制全知上界）

阶梯：
    joint − default          联合学习回收的失配；
    free − default           无结构学习回收的失配；
    free − joint             SEIR 结构先验的价值（>0 = 约束赢）；
    oracle_window − default  窗口化失配总量（可回收上界）；
    oracle_window − joint    学习后残余 gap；
    recovery = joint_gain / oracle_window_gain
                             回收比例（参数学习效率）；
    oracle_nu − oracle_window  类型 + 源异质性剩余。

参数恢复诊断（逐种子）：
    β̂/β_truth、Î0/I0_truth 相对误差（尺度不变量：形状误差
    ||Λ̂/mean(Λ̂) − Λ*/mean(Λ*)||₂ 另报，不受 β↔k 缩放影响）。

协议沿用三/四层消融：分层 50/50 半区、训练半区拟合、测试半区
报告、逐种子 DeLong 配对 + 种子级 bootstrap CI。

诚实边界：失配幅度（±20% 抖动 + 平坦 vs 窗口化）为构造性设定；
结论口径 = "本失配量级下可微联合学习能回收多少 + 结构约束是否
值得"，非真实世界承诺（真实数据验证 = P4b）。

文献：
  Raissi, Perdikaris & Karniadakis (2019) J. Comput. Phys. —— PINN
      （物理参数作为可学习量、方程约束进损失/结构的理论起点）；
  Chen et al. (2018) Neural ODE NeurIPS —— 可微 ODE 求解器
      （RK4 显式积分计算图完整反传，无需黑盒求解器）；
  Runge (1895) / Kutta (1901) —— RK4 数值方法。
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover —— 环境守卫
    TORCH_AVAILABLE = False
    torch = None

from .layer_ablation import delong_paired_test
from .threshold_spec import compute_auc
from .temporal_network import DECAY_WEIGHTS
from .pi_network import calibrate_physics_k
from .combined_network import (
    COMBINED_SEIR_SPEC, _WINDOW_MID_STEPS, build_combined_network,
)


# ==============================================================================
# 可微 RK4 内核（numpy 镜像）
# ==============================================================================

def _seir_rk4_torch(beta, sigma, gamma, n_steps, dt, S0, E0, I0, R0):
    """4D SEIR RK4 积分（torch float64，逐步镜像 _seir_rk4_numpy）。

    每步 RK4 推进后 clamp 非负 + 重归一化——与 numpy 版逐位一致
    （交叉验证测试），且全部操作可微（exp 残差参数化保证参数正，
    状态远离 0 时 clamp 不触发）。
    """
    state = torch.stack([
        torch.as_tensor(S0, dtype=torch.float64),
        torch.as_tensor(E0, dtype=torch.float64),
        torch.as_tensor(I0, dtype=torch.float64),
        torch.as_tensor(R0, dtype=torch.float64),
    ])

    def _deriv(s):
        S, E, I = s[0], s[1], s[2]
        infection = beta * S * I
        return torch.stack([-infection,
                            infection - sigma * E,
                            sigma * E - gamma * I,
                            gamma * I])

    trajectory = [state.clone()]
    for _ in range(n_steps):
        k1 = _deriv(state)
        k2 = _deriv(state + 0.5 * dt * k1)
        k3 = _deriv(state + 0.5 * dt * k2)
        k4 = _deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state = torch.clamp(state, min=0.0)
        state = state / (state.sum() + 1e-8)
        trajectory.append(state.clone())
    return torch.stack(trajectory)          # [n_steps+1, 4]


def _window_forces_from_params(log_d_beta, log_d_i0):
    """从可学习参数重构窗口感染力 Λ̂(t_w) = β̂·I(t_w) [5]。"""
    spec = COMBINED_SEIR_SPEC
    beta = spec['beta'] * torch.exp(log_d_beta)
    i0 = spec['I0'] * torch.exp(log_d_i0)
    traj = _seir_rk4_torch(beta, spec['sigma'], spec['gamma'],
                           spec['n_steps'], spec['dt'],
                           spec['S0'], spec['E0'], i0, spec['R0'])
    idx = torch.as_tensor(_WINDOW_MID_STEPS, dtype=torch.long)
    return beta * traj[idx, 2]              # [5]


# ==============================================================================
# 窗口化物理感染力（numpy，oracle/评估共用）
# ==============================================================================

def windowed_lambda(net, lam_w):
    """窗口化物理感染力：λ = host·corr·Σ_w Λ_w·decay(w)·E_{i,w}。

    与真值核 mech = Σ_c Λ_wc·β_tc·decay(wc)·intens·s_c 的差别仅在
    类型 β_t（E 按窗聚合已含 intens·s）——oracle_window 臂 = 除类型
    不对称外全部已知的上界。

    Args:
        net: build_combined_network 输出
        lam_w: 5 窗感染力（真值 / 默认 / 学习重构均可）
    """
    M = net['M']
    host = np.array([n['host_multiplier'] for n in net['nodes'][M:]],
                    dtype=float)
    E = np.asarray(net['exposure_by_window'], dtype=float)
    w = np.asarray(lam_w, dtype=float) * np.asarray(DECAY_WEIGHTS,
                                                    dtype=float)
    return host * np.asarray(net['corr'], dtype=float) * (E @ w)


def _calibrate_k_generic(lam, labels, train_idx):
    """k̂ 二分校准（逻辑同 pi_network.calibrate_physics_k，任意 λ 输入）。

    k̂ 使 mean(1−exp(−k̂·λ_train)) = 训练半区阳性率（部署校准口径，
    各物理臂统一使用——排序与 k 无关，Brier/概率口径需校准）。
    """
    lam = np.asarray(lam, dtype=float)
    labels = np.asarray(labels, dtype=float)
    tr = np.asarray(train_idx, dtype=np.int64)
    lam_tr = lam[tr]
    if lam_tr.max() <= 0:
        return 1.0
    target = float(labels[tr].mean())
    lo, hi = 1e-9, 1e9
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if (1.0 - np.exp(-mid * lam_tr)).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ==============================================================================
# 学习器（方案 2 主体）
# ==============================================================================

def learn_windowed_seir(net, labels, train_idx, mode='seir', epochs=1200,
                        lr=0.03, seed=0, k_refit_every=50):
    """可微分 SEIR 参数联合学习（确定性，单线程）。

    Args:
        net: build_combined_network 输出
        labels: 接触者标签
        train_idx: 训练半区接触者下标
        mode: 'seir'——可学习 (δβ, δI0)（RK4 计算图反传，2 参数重构
              整条 Λ(t) 轨迹）；'free'——5 个自由 log Λ_w（无 SEIR
              约束对照，同起点默认参数窗值）
        epochs / lr: Adam 全批
        k_refit_every: k 交替重校准间隔（epoch；不进梯度——消除
              β↔k 尺度不可辨识，见模块 docstring）

    Returns:
        dict: p（=1−exp(−k̂·λ̂)）、score（=k̂·λ̂，AUROC 用）、
        k_hat、mode、lam_w_hat [5]、beta_hat / I0_hat（seir 模式）、
        loss_first / loss_last（收敛诊断）
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError('PyTorch 不可用，无法联合学习')
    if mode not in ('seir', 'free'):
        raise ValueError("mode 必须是 'seir' 或 'free'")
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    M = net['M']
    spec = COMBINED_SEIR_SPEC
    E = torch.tensor(np.asarray(net['exposure_by_window']),
                     dtype=torch.float64)
    decay = torch.tensor(np.asarray(DECAY_WEIGHTS), dtype=torch.float64)
    corr = torch.tensor(np.asarray(net['corr']), dtype=torch.float64)
    host = torch.tensor(
        [n['host_multiplier'] for n in net['nodes'][M:]],
        dtype=torch.float64)
    y = torch.tensor(np.asarray(labels, dtype=float), dtype=torch.float64)
    tr = torch.as_tensor(np.asarray(train_idx, dtype=np.int64))

    if mode == 'seir':
        params = [torch.zeros((), dtype=torch.float64, requires_grad=True),
                  torch.zeros((), dtype=torch.float64, requires_grad=True)]
        beta_hat = i0_hat = None

        def _lam_w():
            return _window_forces_from_params(params[0], params[1])
    else:
        lam_w_default = np.asarray(
            net['seir_info']['lam_w_default'], dtype=float)
        params = [torch.tensor(np.log(lam_w_default),
                               dtype=torch.float64, requires_grad=True)]

        def _lam_w():
            return torch.exp(params[0])

    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(params, lr=lr)

    def _forward():
        lam_w = _lam_w()
        lam = host * corr * (E @ (lam_w * decay))
        z = torch.clamp(_forward.k * lam, min=1e-9)
        logit = z + torch.log1p(-torch.exp(-z))
        return lam_w, lam, logit

    # 初始 k：默认参数 λ_joint 上的校准（与 seir_default 臂同口径）
    with torch.no_grad():
        lam_w0 = _lam_w().numpy()
    lam0 = windowed_lambda(net, lam_w0)
    k = _calibrate_k_generic(lam0, labels, train_idx)
    _forward.k = torch.tensor(float(k), dtype=torch.float64)

    loss_first = loss_last = None
    for epoch in range(epochs):
        opt.zero_grad()
        lam_w, lam, logit = _forward()
        loss = bce(logit[tr], y[tr])
        if loss_first is None:
            loss_first = float(loss.detach())
        loss.backward()
        opt.step()
        loss_last = float(loss.detach())
        if (epoch + 1) % k_refit_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                lam_np = windowed_lambda(net, _lam_w().numpy())
            _forward.k = torch.tensor(
                float(_calibrate_k_generic(lam_np, labels, train_idx)),
                dtype=torch.float64)

    with torch.no_grad():
        lam_w_hat = _lam_w().numpy().copy()
        lam_hat = windowed_lambda(net, lam_w_hat)
        k_hat = float(_calibrate_k_generic(lam_hat, labels, train_idx))
        score = k_hat * lam_hat
        if mode == 'seir':
            beta_hat = float(spec['beta'] * np.exp(params[0].item()))
            i0_hat = float(spec['I0'] * np.exp(params[1].item()))

    out = {
        'p': 1.0 - np.exp(-score),
        'score': score,
        'k_hat': k_hat,
        'mode': mode,
        'lam_w_hat': lam_w_hat.tolist(),
        'loss_first': loss_first,
        'loss_last': loss_last,
    }
    if mode == 'seir':
        out['beta_hat'] = beta_hat
        out['I0_hat'] = i0_hat
    return out


# ==============================================================================
# 五臂消融协议
# ==============================================================================

ARMS = ('seir_default', 'seir_joint', 'free_windows', 'oracle_window',
        'oracle_nu')


def _recall_at_budget(scores, y, budget=0.25):
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    n_top = max(1, int(np.ceil(budget * len(y))))
    top = np.argsort(-scores)[:n_top]
    return float(y[top].sum() / max(y.sum(), 1))


def _stratified_split(labels, rng):
    """分层 50/50（与 combined_ablation 同实现）。"""
    labels = np.asarray(labels, dtype=int)
    train_idx, test_idx = [], []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        train_idx.extend(idx[:half].tolist())
        test_idx.extend(idx[half:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def run_seir_joint_ablation_once(n_contacts=400, seed=42, target_rate=0.25,
                                 budget=0.25, joint_epochs=1200, lr=0.03):
    """单次五臂消融（一个种子 = 一个网络 + 三次拟合/校准）。"""
    net = build_combined_network(n_contacts=n_contacts,
                                 target_rate=target_rate,
                                 random_state=seed)
    labels = np.asarray(net['labels'], dtype=int)
    rng = np.random.RandomState(seed)
    train_idx, test_idx = _stratified_split(labels, rng)

    # ---- 臂分数 ----
    k_default = calibrate_physics_k(net, labels, train_idx)
    lam_default = np.asarray(net['lam'], dtype=float)

    joint = learn_windowed_seir(net, labels, train_idx, mode='seir',
                                epochs=joint_epochs, lr=lr, seed=seed)
    free = learn_windowed_seir(net, labels, train_idx, mode='free',
                               epochs=joint_epochs, lr=lr, seed=seed)

    lam_w_truth = np.asarray(net['seir_info']['lam_w_truth'], dtype=float)
    lam_ow = windowed_lambda(net, lam_w_truth)
    k_ow = _calibrate_k_generic(lam_ow, labels, train_idx)

    arms = {
        'seir_default': {'scores': k_default * lam_default,
                         'p': 1.0 - np.exp(-k_default * lam_default)},
        'seir_joint': {'scores': joint['score'], 'p': joint['p']},
        'free_windows': {'scores': free['score'], 'p': free['p']},
        'oracle_window': {'scores': k_ow * lam_ow,
                          'p': 1.0 - np.exp(-k_ow * lam_ow)},
        'oracle_nu': {'scores': np.asarray(net['nu'], dtype=float),
                      'p': 1.0 - np.exp(
                          -net['k_calibration']
                          * np.asarray(net['nu'], dtype=float))},
    }

    # ---- 统一评估（测试半区）----
    y_test = labels[test_idx]
    for name, arm in arms.items():
        s = arm['scores'][test_idx]
        arm['test_scores'] = s.tolist()
        arm['auroc'] = compute_auc(arm['test_scores'], y_test.tolist())
        arm['recall_at_budget'] = _recall_at_budget(s, y_test, budget)
        arm['brier'] = float(np.mean(
            (np.asarray(arm['p'], dtype=float)[test_idx] - y_test) ** 2))
        del arm['scores'], arm['p']

    # ---- 阶梯 ----
    a = {k: v['auroc'] for k, v in arms.items()}
    ow_gain = a['oracle_window'] - a['seir_default']
    j_gain = a['seir_joint'] - a['seir_default']
    ladder = {
        'joint_minus_default': j_gain,
        'free_minus_default': a['free_windows'] - a['seir_default'],
        'free_minus_joint': a['free_windows'] - a['seir_joint'],
        'oracle_window_minus_default': ow_gain,
        'oracle_window_minus_joint': a['oracle_window'] - a['seir_joint'],
        'recovery_ratio': (float(j_gain / ow_gain)
                           if abs(ow_gain) > 1e-9 else float('nan')),
        'oracle_nu_minus_oracle_window': a['oracle_nu'] - a['oracle_window'],
    }

    # ---- 参数恢复诊断（尺度不变量：形状误差不受 β↔k 缩放影响）----
    lam_w_hat = np.asarray(joint['lam_w_hat'], dtype=float)
    shape_hat = lam_w_hat / lam_w_hat.mean()
    shape_true = lam_w_truth / lam_w_truth.mean()
    param_recovery = {
        'beta_rel_err': abs(joint['beta_hat']
                            / net['seir_info']['beta_truth'] - 1.0),
        'i0_rel_err': abs(joint['I0_hat']
                          / net['seir_info']['I0_truth'] - 1.0),
        'lam_w_shape_err': float(np.sqrt(
            np.mean((shape_hat - shape_true) ** 2))),
        'free_shape_err': float(np.sqrt(np.mean(
            (np.asarray(free['lam_w_hat']) / np.asarray(
                free['lam_w_hat']).mean() - shape_true) ** 2))),
        # 注：k 每 50 epoch 外生重校准会重置损失景观，BCE 非单调
        # （n=400 实测 drop 可为微负）；判别效果看阶梯不看此字段
        'joint_loss_drop': joint['loss_first'] - joint['loss_last'],
        'free_loss_drop': free['loss_first'] - free['loss_last'],
    }

    def _pair(x, ykey):
        return delong_paired_test(y_test.tolist(),
                                  arms[x]['test_scores'],
                                  arms[ykey]['test_scores'])

    delong = {
        'joint_vs_default': _pair('seir_joint', 'seir_default'),
        'free_vs_default': _pair('free_windows', 'seir_default'),
        'free_vs_joint': _pair('free_windows', 'seir_joint'),
        'oracle_window_vs_default': _pair('oracle_window', 'seir_default'),
    }

    return {
        'seed': seed,
        'n_contacts': n_contacts,
        'target_rate': target_rate,
        'budget': budget,
        'n_test': int(len(test_idx)),
        'test_labels': y_test.tolist(),
        'arms': arms,
        'ladder': ladder,
        'param_recovery': param_recovery,
        'delong': delong,
    }


_DELONG_MAP = {
    'joint_minus_default': 'joint_vs_default',
    'free_minus_default': 'free_vs_default',
    'free_minus_joint': 'free_vs_joint',
    'oracle_window_minus_default': 'oracle_window_vs_default',
}


def run_multi_seed_seir_joint_ablation(n_contacts=400, n_seeds=20,
                                       seed_start=101, target_rate=0.25,
                                       budget=0.25, joint_epochs=1200,
                                       lr=0.03, n_bootstrap=2000):
    """多种子五臂消融：效应量 + 种子级 bootstrap CI + DeLong 汇总。"""
    seeds_reports = []
    for i in range(n_seeds):
        rep = run_seir_joint_ablation_once(
            n_contacts=n_contacts, seed=seed_start + i,
            target_rate=target_rate, budget=budget,
            joint_epochs=joint_epochs, lr=lr)
        seeds_reports.append({
            'seed': rep['seed'],
            'auroc': {k: rep['arms'][k]['auroc'] for k in ARMS},
            'recall_at_budget': {k: rep['arms'][k]['recall_at_budget']
                                 for k in ARMS},
            'brier': {k: rep['arms'][k]['brier'] for k in ARMS},
            'ladder': rep['ladder'],
            'param_recovery': rep['param_recovery'],
            'delong_p': {k: v['p_value'] for k, v in rep['delong'].items()},
            'positive_rate': float(np.mean(rep['test_labels'])),
        })

    rng = np.random.RandomState(seed_start)

    def _boot_ci(values):
        values = np.asarray(values, dtype=float)
        means = [values[rng.randint(0, len(values), len(values))].mean()
                 for _ in range(n_bootstrap)]
        return [float(np.percentile(means, 2.5)),
                float(np.percentile(means, 97.5))]

    arm_summary = {}
    for k in ARMS:
        entry = {}
        for metric in ('auroc', 'recall_at_budget', 'brier'):
            vals = [s[metric][k] for s in seeds_reports]
            entry['mean_' + metric] = float(np.mean(vals))
            entry['bootstrap_ci_' + metric] = _boot_ci(vals)
        arm_summary[k] = entry

    ladder_summary = {}
    for k, rep_lad in seeds_reports[0]['ladder'].items():
        vals = [s['ladder'][k] for s in seeds_reports]
        ci = _boot_ci(vals)
        entry = {
            'mean': float(np.nanmean(vals)),
            'bootstrap_ci': ci,
            'positive_seeds': int(sum(v > 0 for v in vals
                                      if not np.isnan(v))),
            'ci_excludes_zero': bool((ci[0] > 0.0) or (ci[1] < 0.0)),
        }
        pair = _DELONG_MAP.get(k)
        if pair:
            p_vals = [s['delong_p'][pair] for s in seeds_reports]
            entry['delong_significant_frac'] = float(np.mean(
                [p < 0.05 for p in p_vals]))
        ladder_summary[k] = entry

    recovery_summary = {}
    for k in seeds_reports[0]['param_recovery']:
        vals = [s['param_recovery'][k] for s in seeds_reports]
        recovery_summary[k] = {
            'mean': float(np.mean(vals)),
            'bootstrap_ci': _boot_ci(vals),
        }

    # ---- P4a 结论：联合学习是否回收失配 + 结构先验是否值得 ----
    j = ladder_summary['joint_minus_default']
    fj = ladder_summary['free_minus_joint']
    joint_helps = bool(j['mean'] > 0.0 and j['bootstrap_ci'][0] > 0.0)
    if fj['bootstrap_ci'][0] > 0.0:
        prior_verdict = 'seir_prior_wins（结构约束占优）'
    elif fj['bootstrap_ci'][1] < 0.0:
        prior_verdict = 'free_wins（自由参数占优——结构约束过强）'
    else:
        prior_verdict = 'equivalent_within_noise（两者相当）'
    conclusion = {
        'joint_learning_helps': joint_helps,
        'structure_prior_verdict': prior_verdict,
        'recovery_ratio': ladder_summary['recovery_ratio'],
    }

    return {
        'design': {
            'name': 'seir_joint_ablation_v1',
            'n_contacts': n_contacts, 'n_seeds': n_seeds,
            'seed_start': seed_start, 'target_rate': target_rate,
            'budget': budget, 'joint_epochs': joint_epochs, 'lr': lr,
            'learner': "方案 2（特征级可微联合）：(δβ,δI0) exp 残差 + RK4"
                       " 反传 + k 交替校准（不进梯度）；free = 5 自由"
                       'log Λ_w 对照',
            'arms': list(ARMS),
            'dgp': 'combined_mechanism_network_v1（物理失配预埋：'
                   '±20% 参数抖动 + 平坦 Λ_ref vs 窗口化 Λ(t_w)）',
            'split': 'stratified 50/50 half-split（接触者空间）',
            'metrics': 'AUROC + recall@budget + Brier',
        },
        'seeds': seeds_reports,
        'arm_summary': arm_summary,
        'ladder_summary': ladder_summary,
        'param_recovery_summary': recovery_summary,
        'conclusion': conclusion,
    }
