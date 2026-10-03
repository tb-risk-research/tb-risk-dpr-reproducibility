#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机制×图×时序端到端联合 GNN：SEIR ODE 嵌入消息传递（P5，2026-09-04）。

背景（用户 2026-09-04 "下一步创新"）：
  "PI-GNN / STGNN / temporal GNN 现为并列成员，集成只追平、不显著
   超越。创新落点：把 SEIR ODE 嵌入 GNN 消息传递 + 时序先验作为
   可学习特征，做单一模型内的物理信息联合，而非事后加权。"

与三成员/P4a 的结构差异（对照 combined DGP 真值核）：

    真值   ν_i = host_i · corr_i · Σ_c Λ(t_{w_c})·β_{t_c}·decay(w_c)
                                          ·intens_{c,i}·s_c

    本模型 ν̂_i = host_i · corr_i · Σ_c Λ̂(w_c)·β̂_{t_c}·d̂(w_c)·ŝ_c

  逐成员边消息 = ODE 轨迹感染力 Λ̂(t_w)（可微 RK4，可学习 δβ/δI0）
  × 可学习类型权重 β̂_t（文献初值）× 可学习窗衰减 d̂_w（时序先验
  作为可学习特征，文献 8 周半衰期初值）× 源传染性 ŝ_c（图节点
  特征 log_infectivity，PI-GNN v11 验证的合法读取通道）。
  消息沿 membership 边（接触者↔指示病例）聚合——**消息传递即
  物理机制**，而非"GNN 学表征 + 物理特征拼接"。

  - vs PI-GNN（成员）：物理 λ 冻结 + 残差修正，类型/窗口不进
    消息传递——本模型把类型/窗口/ODE 全部嵌入消息；
  - vs seir_joint（P4a 特征级联合）：λ̂ = host·corr·E@(Λ̂·decay)
    用**边际**窗口暴露和（E 含 intens·s，无类型分解）——本模型
    逐边 (类型×窗口×源) 乘积结构，能表达真值核（intens 除外），
    P4a 与特征化树（只有边际和）均不能；
  - vs 三成员集成：集成只能事后组合各成员的边际读数，无法组合
    逐边乘积结构。

可辨识性（P4a 教训沿用）：β̂·β̂_t·d̂_w·ŝ 的全局尺度与 k 等价
（Λ 整体缩放 c ↔ k/c），k 不进梯度、每 50 epoch 训练半区二分
重校准（交替优化）；形状参数（SEIR 2 + 类型 5 + 衰减 5 = 12 个）
可辨识。

参数预算（小样本政权，v7-v11 教训：万参数 GAT 在 n=400 记忆化，
18 参数机制读出兑现 91% 间隙）：物理头恰好 12 个机制参数 + k；
residual 变体追加有界乘性残差头（池化源特征线性读出 → tanh，
z = k·ν̂·exp(a·t)，z/物理 ∈ [e^−a, e^a]——PI 哲学：物理覆盖
机制，残差只学物理没覆盖的部分，本 DGP 中为 intens 缺失）。

先验收缩（部署语义"数据弱时回落物理"）：对 12 个 log 参数加
prior_lambda·‖θ−θ_lit‖² 收缩到文献初值。诚实声明：本 DGP 中
类型/衰减真值=文献值（收缩有利）、SEIR 真值抖动 ±20%/±50%
（收缩不利）——两向并存非单向作弊；真实部署中文献值同样是
先验最优猜测。

文献：
  Raissi et al. (2019) J. Comput. Phys. —— PINN（方程嵌入结构）；
  Chen et al. (2018) NeurIPS —— Neural ODE（可微积分器反传）；
  Zong et al. (2020) NeurIPS —— 机制消息传递（Lagrangian GNN，
      消息由物理约束参数化的思想）；
  P4a（本仓库 seir_joint.py）—— 可微 RK4 + k 交替校准协议出处。
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:  # pragma: no cover —— 环境守卫
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None

if PYTORCH_AVAILABLE:
    # 单源复用：可微 RK4 内核 + 窗口感染力重构 + k 二分校准
    # （validation 层无 ml.gnn 模块级导入，无循环依赖——
    #   combined_ablation→pi_gnn 已验证该方向导入安全）
    from ...validation.seir_joint import (
        _calibrate_k_generic,
        _seir_rk4_torch,
        _window_forces_from_params,
    )
    from ...validation.combined_network import _WINDOW_MID_STEPS


class JointMechGNN(nn.Module if PYTORCH_AVAILABLE else object):
    """机制×图×时序联合消息传递网络（物理头 [+ 有界残差头]）。

    Args:
        beta_t_init: [5] 各类型权重初值（文献 BETA_BY_TYPE）
        decay_init: [5] 各窗衰减初值（时序先验文献值 0.5^(mid/8w)）
        residual: True → 追加有界乘性残差头（池化源特征线性读出）
        residual_scale: 残差尺度 a（z/物理 ∈ [e^−a, e^a]）
        pooled_dim: 残差头输入维度（池化源节点特征维度，默认 14）
        seir_mode: 'learn'（可学 δβ/δI0，默认）/ 'frozen'（文献默认
            Λ，P0.2 归因消融——隔离"学 ODE 参数"的贡献）
        msg_mode: 'mech'（机制乘积 Λ·β·d，默认）/ 'free'（自由
            type×window 消息表，初值=机制乘积——P0.2 归因消融，
            隔离"物理因子分解形式"的贡献；无 SEIR/类型/衰减结构）
    """

    def __init__(self, beta_t_init, decay_init, residual=False,
                 residual_scale=1.0, pooled_dim=14, seir_mode='learn',
                 msg_mode='mech'):
        super().__init__()
        if not PYTORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError('PyTorch 不可用')
        if seir_mode not in ('learn', 'frozen'):
            raise ValueError("seir_mode ∈ {'learn','frozen'}")
        if msg_mode not in ('mech', 'free'):
            raise ValueError("msg_mode ∈ {'mech','free'}")
        self.seir_mode = seir_mode
        self.msg_mode = msg_mode
        beta_t_init = np.asarray(beta_t_init, dtype=np.float64)
        decay_init = np.asarray(decay_init, dtype=np.float64)
        # SEIR 残差参数（exp 参数化：0 = 文献默认值；frozen 为 buffer）
        if seir_mode == 'learn':
            self.log_d_beta = nn.Parameter(
                torch.zeros((), dtype=torch.float64))
            self.log_d_i0 = nn.Parameter(
                torch.zeros((), dtype=torch.float64))
        else:
            self.register_buffer(
                'log_d_beta', torch.zeros((), dtype=torch.float64))
            self.register_buffer(
                'log_d_i0', torch.zeros((), dtype=torch.float64))
        # 类型权重 / 窗衰减（log 参数化，文献初值）
        self.log_beta_t = nn.Parameter(
            torch.log(torch.as_tensor(beta_t_init,
                                      dtype=torch.float64)).clone())
        self.log_decay = nn.Parameter(
            torch.log(torch.as_tensor(decay_init,
                                      dtype=torch.float64)).clone())
        if msg_mode == 'free':
            # 自由消息表 [T, W]：初值 = 机制乘积 Λ⁰_w·β⁰_t·d⁰_w
            # （与 mech 模式同起点——归因消融的控制变量口径）
            _fill_defaults()
            lam0 = _window_forces_from_params(
                torch.zeros((), dtype=torch.float64),
                torch.zeros((), dtype=torch.float64)).detach()
            msg0 = (lam0[None, :] * torch.as_tensor(
                beta_t_init, dtype=torch.float64)[:, None]
                * torch.as_tensor(decay_init, dtype=torch.float64)[None, :])
            self.log_msg_wt = nn.Parameter(torch.log(msg0))
        self.residual = bool(residual)
        if self.residual:
            self.residual_scale = float(residual_scale)
            self.delta_head = nn.Linear(pooled_dim, 1).double()

    # ---- 机制参数读取（诊断/恢复检查）----
    def mechanism_params(self):
        """当前机制参数（numpy，脱离计算图）。"""
        _fill_defaults()
        with torch.no_grad():
            return {
                'beta_hat': float(_BETA_DEFAULT
                                  * torch.exp(self.log_d_beta)),
                'I0_hat': float(_I0_DEFAULT * torch.exp(self.log_d_i0)),
                'beta_t_hat': torch.exp(self.log_beta_t).numpy().tolist(),
                'decay_hat': torch.exp(self.log_decay).numpy().tolist(),
            }

    def _lam_w(self):
        """窗口感染力 Λ̂(t_w) = β̂·I(t_w)（可微 RK4，5 窗中点）。"""
        return _window_forces_from_params(self.log_d_beta, self.log_d_i0)

    def forward(self, mem_contact, mem_src, mem_type, mem_win,
                log_infectivity, host, corr, x_nodes=None, k=1.0):
        """前向：返回 (logit [n_contacts], nu_hat [n_contacts], t [n])。

        消息传递（物理机制本体）：
            msg_c = Λ̂(w_c)·β̂_{t_c}·d̂_{w_c}·exp(log_infectivity_{src_c})
            ν̂_i  = host_i·corr_i·Σ_{c∈memberships(i)} msg_c
        概率头（部署口径，P4a 同式）：
            z = clamp(k·ν̂, min=1e-9)；logit = z + log1p(−exp(−z))
            （孤立接触者 ν̂=0 → p≈0——物理正确：无暴露无感染，
             平坦基线 λ=host·Λ_ref·corr 对孤立者的错误在此被修复）

        Args:
            mem_contact/mem_src/mem_type/mem_win: LongTensor [E_m]
                （membership 边的接触者下标 / 源节点 id / 类型 idx /
                 窗 idx；E_m=0 时全部允许为空张量）
            log_infectivity: [N] 全图节点 log 传染性（接触者段为 0）
            host/corr: [n] 接触者宿主乘子 / 暴露修正（物理已知）
            x_nodes: [N, pooled_dim] 节点特征（residual=True 时必需）
            k: 校准参数（不进梯度——交替优化协议）
        """
        lam_w = self._lam_w()                                   # [5]
        beta_t = torch.exp(self.log_beta_t)                     # [5]
        decay = torch.exp(self.log_decay)                       # [5]
        n = host.shape[0]
        if mem_contact.numel() == 0:
            mech = torch.zeros(n, dtype=torch.float64)
        else:
            s_src = torch.exp(log_infectivity[mem_src])         # [E_m]
            if self.msg_mode == 'free':
                msg_wt = torch.exp(self.log_msg_wt)             # [T, W]
                msg = msg_wt[mem_type, mem_win] * s_src         # [E_m]
            else:
                msg = (lam_w[mem_win] * beta_t[mem_type]
                       * decay[mem_win] * s_src)                # [E_m]
            mech = torch.zeros(n, dtype=torch.float64)
            mech.index_add_(0, mem_contact, msg)
        nu_hat = host * corr * mech

        t = torch.zeros(n, dtype=torch.float64)
        if self.residual:
            pooled = torch.zeros(n, x_nodes.shape[1],
                                 dtype=torch.float64)
            cnt = torch.zeros(n, 1, dtype=torch.float64)
            ones = torch.ones(mem_contact.shape[0], 1,
                              dtype=torch.float64)
            pooled.index_add_(0, mem_contact, x_nodes[mem_src])
            cnt.index_add_(0, mem_contact, ones)
            pooled = pooled / cnt.clamp(min=1.0)
            t = torch.tanh(self.delta_head(pooled).squeeze(-1))
            z = torch.clamp(k * nu_hat
                            * torch.exp(self.residual_scale * t),
                            min=1e-9)
        else:
            z = torch.clamp(k * nu_hat, min=1e-9)
        logit = z + torch.log1p(-torch.exp(-z))
        return logit, nu_hat, t


# 文献默认值（延迟填充：import 时从 combined_network 单源读取）
_BETA_DEFAULT = None
_I0_DEFAULT = None


def _fill_defaults():
    """SEIR 文献默认值单源填充（避免模块级 import 循环风险）。"""
    global _BETA_DEFAULT, _I0_DEFAULT
    if _BETA_DEFAULT is None:
        from ...validation.combined_network import COMBINED_SEIR_SPEC
        _BETA_DEFAULT = float(COMBINED_SEIR_SPEC['beta'])
        _I0_DEFAULT = float(COMBINED_SEIR_SPEC['I0'])
    return _BETA_DEFAULT, _I0_DEFAULT


def membership_tensors(net):
    """combined 网络的 membership 边张量（图输入，torch float64/long）。

    Returns:
        dict: mem_contact [E_m]（接触者 0-based）/ mem_src [E_m]
        （源节点 id 0..M-1）/ mem_type [E_m]（EDGE_TYPE_IDS idx）/
        mem_win [E_m]（窗 idx）/ log_infectivity [M+n] /
        host [n] / corr [n] / x_nodes [M+n, 14]（pi 基础特征——
        残差头输入；log_infectivity 取自同一特征的第 1 维）
    """
    import torch as _torch

    from ...validation.pi_network import node_features
    from ...validation.typed_network import EDGE_TYPE_IDS

    _fill_defaults()
    M = int(net['M'])
    n = len(net['nodes']) - M
    t_idx = {t: k for k, t in enumerate(EDGE_TYPE_IDS)}
    c_list, s_list, t_list, w_list = [], [], [], []
    for i, mems in enumerate(net['memberships']):
        for (src, t, w, _inten, _s) in mems:
            c_list.append(i)
            s_list.append(int(src))
            t_list.append(t_idx[t])
            w_list.append(int(w))
    X = node_features(net)
    host = np.asarray([nd['host_multiplier']
                       for nd in net['nodes'][M:]], dtype=np.float64)
    corr = np.asarray(net['corr'], dtype=np.float64)
    return {
        'mem_contact': _torch.tensor(np.asarray(c_list, dtype=np.int64),
                                     dtype=_torch.long),
        'mem_src': _torch.tensor(np.asarray(s_list, dtype=np.int64),
                                 dtype=_torch.long),
        'mem_type': _torch.tensor(np.asarray(t_list, dtype=np.int64),
                                  dtype=_torch.long),
        'mem_win': _torch.tensor(np.asarray(w_list, dtype=np.int64),
                                 dtype=_torch.long),
        'log_infectivity': _torch.tensor(X[:, 1], dtype=_torch.float64),
        'host': _torch.tensor(host, dtype=_torch.float64),
        'corr': _torch.tensor(corr, dtype=_torch.float64),
        'x_nodes': _torch.tensor(X, dtype=_torch.float64),
        'n_memberships': len(c_list),
    }


def train_joint_mech_gnn(net, labels, train_contacts, residual=False,
                         epochs=1200, lr=0.03, k_refit_every=50,
                         prior_lambda=1e-3, seed=0, residual_scale=1.0,
                         seir_mode='learn', msg_mode='mech'):
    """在 combined 网络上端到端训练 JointMechGNN（确定性，单线程）。

    协议（P4a 沿用）：Adam 全批 BCE（仅训练半区接触者）+ k 交替
    校准（每 k_refit_every epoch，不进梯度）+ 先验收缩（机制 log
    参数 → 文献初值；残差头参数走 weight_decay=3e-3，
    PI-GNN v11 验证值）。

    Args:
        net: build_combined_network 输出
        labels: 接触者标签
        train_contacts: 训练半区接触者下标（接触者空间）
        residual: True → joint_res 臂（+ 有界乘性残差头）
        epochs / lr: 训练预算（P4a 同值）
        k_refit_every: k 交替重校准间隔（epoch）
        prior_lambda: 机制参数先验收缩系数（0 = 关闭）
        seed: torch 种子
        seir_mode: 'learn' / 'frozen'（P0.2 归因消融）
        msg_mode: 'mech' / 'free'（P0.2 归因消融）

    Returns:
        dict: score（= k̂·ν̂·[exp(a·t)]，AUROC 用）/ p（=1−exp(−z)）/
        k_hat / params（机制参数估计）/ lam_w_hat [5] /
        loss_first / loss_last / residual / n_memberships
    """
    if not PYTORCH_AVAILABLE:
        raise RuntimeError('PyTorch 不可用，无法联合训练')
    from ...validation.combined_network import COMBINED_SEIR_SPEC
    from ...validation.temporal_network import DECAY_WEIGHTS
    from ...validation.typed_network import BETA_BY_TYPE, EDGE_TYPE_IDS

    _fill_defaults()
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    g = membership_tensors(net)
    labels = np.asarray(labels, dtype=float)
    tr = np.asarray(train_contacts, dtype=np.int64)
    tr_t = torch.as_tensor(tr, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float64)

    beta_t_init = np.asarray([BETA_BY_TYPE[t] for t in EDGE_TYPE_IDS],
                             dtype=np.float64)
    decay_init = np.asarray(DECAY_WEIGHTS, dtype=np.float64)
    model = JointMechGNN(beta_t_init, decay_init, residual=residual,
                         residual_scale=residual_scale,
                         pooled_dim=g['x_nodes'].shape[1],
                         seir_mode=seir_mode, msg_mode=msg_mode)

    # 先验锚点（log 空间文献初值；free 消息表锚定机制乘积起点）
    anchors = {
        'log_d_beta': torch.zeros((), dtype=torch.float64),
        'log_d_i0': torch.zeros((), dtype=torch.float64),
        'log_beta_t': torch.log(torch.as_tensor(
            beta_t_init, dtype=torch.float64)).clone(),
        'log_decay': torch.log(torch.as_tensor(
            decay_init, dtype=torch.float64)).clone(),
    }
    prior_pairs = []
    if msg_mode == 'mech':
        physics_params = [model.log_beta_t, model.log_decay]
        prior_pairs = [(model.log_beta_t, anchors['log_beta_t']),
                       (model.log_decay, anchors['log_decay'])]
        if seir_mode == 'learn':
            physics_params = ([model.log_d_beta, model.log_d_i0]
                              + physics_params)
            prior_pairs = [(model.log_d_beta, anchors['log_d_beta']),
                           (model.log_d_i0, anchors['log_d_i0'])] \
                + prior_pairs
    else:
        physics_params = [model.log_msg_wt]
        prior_pairs = [(model.log_msg_wt, model.log_msg_wt.detach().clone())]
    if residual:
        opt = torch.optim.Adam(
            [{'params': physics_params},
             {'params': model.delta_head.parameters(),
              'weight_decay': 3e-3}], lr=lr)
    else:
        opt = torch.optim.Adam(physics_params, lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()

    def _forward(k):
        return model(g['mem_contact'], g['mem_src'], g['mem_type'],
                     g['mem_win'], g['log_infectivity'], g['host'],
                     g['corr'], x_nodes=g['x_nodes'], k=k)

    def _prior():
        if prior_lambda <= 0 or not prior_pairs:
            return torch.zeros((), dtype=torch.float64)
        return prior_lambda * sum(
            (p - a).pow(2).sum() for p, a in prior_pairs)

    # 初始 k：文献默认参数 ν̂ 上的训练半区校准
    with torch.no_grad():
        _, nu0, _ = _forward(1.0)
    k = _calibrate_k_generic(nu0.numpy(), labels, tr)
    k_t = torch.tensor(float(k), dtype=torch.float64)

    loss_first = loss_last = None
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        logit, _, _ = _forward(k_t)
        loss = bce(logit[tr_t], y[tr_t]) + _prior()
        if loss_first is None:
            loss_first = float(loss.detach())
        loss.backward()
        opt.step()
        loss_last = float(loss.detach())
        if (epoch + 1) % k_refit_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                _, nu_cur, _ = _forward(1.0)
            k = _calibrate_k_generic(nu_cur.numpy(), labels, tr)
            k_t = torch.tensor(float(k), dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        logit, nu_hat, _ = _forward(k_t)
        k_hat = float(_calibrate_k_generic(nu_hat.numpy(), labels, tr))
        # 概率口径用末端重校准 k（部署校准语义，P4a 同口径）
        z = torch.clamp(k_hat * nu_hat, min=1e-9)
        if residual:
            # 残差因子单独重放（k̂ 校准后的最终概率）
            pooled = torch.zeros(g['host'].shape[0],
                                 g['x_nodes'].shape[1],
                                 dtype=torch.float64)
            cnt = torch.zeros(g['host'].shape[0], 1, dtype=torch.float64)
            ones = torch.ones(g['mem_contact'].shape[0], 1,
                              dtype=torch.float64)
            pooled.index_add_(0, g['mem_contact'],
                              g['x_nodes'][g['mem_src']])
            cnt.index_add_(0, g['mem_contact'], ones)
            pooled = pooled / cnt.clamp(min=1.0)
            t_fin = torch.tanh(model.delta_head(pooled).squeeze(-1))
            z = torch.clamp(k_hat * nu_hat
                            * torch.exp(model.residual_scale * t_fin),
                            min=1e-9)
        p = 1.0 - torch.exp(-z)
        score = (k_hat * nu_hat).numpy().copy()
        if residual:
            score = z.numpy().copy()          # 排序含残差因子的 z
        params = model.mechanism_params()
        with torch.no_grad():
            lam_w_hat = model._lam_w().numpy().tolist()

    return {
        'score': score,
        'p': p.numpy(),
        'k_hat': k_hat,
        'params': params,
        'lam_w_hat': lam_w_hat,
        'loss_first': loss_first,
        'loss_last': loss_last,
        'residual': bool(residual),
        'n_memberships': int(g['n_memberships']),
    }
