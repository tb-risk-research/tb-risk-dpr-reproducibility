#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PI-GNN：SEIR 物理锚定 + 有界网络残差融合（第三层）。

背景（用户"PI-GNN：把 SEIR 先验注入网络层"，2026-08-25）：
  "已知用方程，未知用学习"——最终风险 = SEIR 理论值（物理已知）
  + GNN 残差（数据驱动，只学物理模型没覆盖的部分）。

结构（用户"残差融合层"规格：p_i = 1 − exp(−(λ_i + δ_i))，δ 有界）：

    msg_i = mean_{j ∈ 源(i)} x_j          # 类型化单层聚合（无参数：
    t_i   = tanh(delta_head(msg_i))       #   只池化指示病例邻居——
                                          #   接触调查天然二部结构）
    z_i   = λ̃_i · exp(a · t_i)            # 感染力 z > 0（a = 残差尺度）
    p_i   = 1 − exp(−z_i)                 # 用户公式，δ = λ̃·(exp(a·t)−1)

  δ 的乘性结构与 DGP 残差 r = λ·(s̄−1) 同构：零暴露节点残差为零
  （无暴露则无感染修正）、物理方向保持（z/λ̃ ∈ [e^−a, e^a]，a=1
  时 ∈ [0.37, 2.72]，覆盖 DGP s̄ 的 q10~q90）。

参数化定稿依据（v7-v11 诊断扫描，2026-08-25，8 种子 n=400）：
  - 2 层 GAT（万参数）端到端：Δ(test AUROC − SEIR) = −0.016，
    0/8 为正——train AUROC 0.97 vs test 0.80，容量全部用于
    记忆 200 个训练节点的 Bernoulli 标签噪声；
  - 每源嵌入 v_j（80 参数，t_i = mean v_j）：无收缩 Δ=−0.15
    （簇级过拟合，每源仅 ~3.5 个训练标签）、强收缩 Δ≈0——
    统计功率不足的死路；
  - **源专属池化 + 线性读出（本结构，~18 参数）：Δ=+0.0235，
    8/8 为正，corr(t, log s̄)=0.96——兑现 oracle 间隙（+0.026）
    的 91%**。机制：学的不是每源标量（功率不足），而是"读邻居
    源 log 传染性"的机制函数（个位数参数，功率充足）——PI
    哲学的递归应用：机制先验进结构，只学尺度。
  - 浅 MLP 读出与线性等价（+0.0251 vs +0.0235）——信号本质
    线性；wd=3e-3 必要（wd=0 时 2 参数亦过拟合，3/8）。
  - 源专属（分母只数指示病例边）vs 全邻居池化：corr 0.96 vs
    0.48——伙伴数不同的非均匀稀释（λ·inf/(1+deg)）干扰排序。

  logit = log(p/(1−p)) = z + log1p(−exp(−z))（数值稳定实现；
  z = clamp(z, min=1e-6) 保护 λ̃=0 孤立节点的梯度链）。

损失（用户"损失级注入"规格，mode='feature_loss'）：
    L = L_label + λ_b·L_bounded + λ_c·L_consistency
  - L_bounded = mean(t²)：软有界——残差越小越好，"修正而非接管"；
  - L_consistency = relu(ρ_min − ρ(z, λ̃))：排序一致性——预测感染
    力与 SEIR 理论值的 Pearson 相关不得低于 ρ_min（Spearman 的
    可微代理；物理给方向，残差微调排序但不许颠覆）。
  - L_physics（SEIR 方程残差）在"SEIR 冻结"方案（用户方案 1）下
    无待约束参数，不适用；L_smooth（图平滑）默认不启用——第一/
    二层实证过度平滑拉平排序恰是网络层无增量的原因（用户规格
    亦注明"这个正则需要弱权重"）。

四臂消融中的角色（validation/pi_ablation.py）：
  - 纯 SEIR 臂：k̂·λ 排序（无学习，k̂ 训练半区校准）；
  - 纯 GNN 臂：train_pure_gnn（14 维基础特征端到端，无物理——
    万参数 GAT 在 n=400 下的记忆化对照，即用户"纯 GNN 小样本
    容易过拟合"主张的实证）；
  - PI 特征级：train_pi_gnn(mode='feature')——物理特征拼接 +
    融合结构，仅 L_label；
  - PI 特征+损失级：train_pi_gnn(mode='feature_loss')——追加
    物理一致性损失。

训练协议沿用第一/二层：单线程 + 固定 seed 确定性、Adam 全批、
损失只作用于训练半区接触者（指示病例无标签、参与消息传递）。

文献：
  Raissi, Perdikaris & Karniadakis (2019) J. Comput. Phys. —— PINN
      （物理残差作为损失约束，PI-GNN 理论起点）；
  HeatGNN (Zheng et al., arXiv 2024) —— 机制损失对齐 SIR 行为
      （L_consistency 的"物理给方向"思想来源）；
  CSTGNN (Han et al., arXiv 2025) —— 物理量作因果先验特征注入
      （物理特征拼接路线）。
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
    from .typed_gat import TypedGATNet


# ==============================================================================
# P3：PI 注入路径样本量自适应（2026-08-25 产品化）
# ==============================================================================

PI_INJECTION_SWITCH_THRESHOLD = 150   # 接触者数阈值（用户 P3 规格）

PI_INJECTION_EVIDENCE = {
    'n400_feature_minus_loss': '+0.0038 CI[+0.0017,+0.0060] 20/20（特征级占优）',
    'n200_feature_minus_loss': '-0.0029 CI 含 0（两者相当）',
    'n100_feature_minus_loss': '-0.0083 CI[-0.0152,-0.0030]（损失级反超——'
                               '物理一致性约束在小样本的正则价值）',
    'artifact': 'data/processed/pi_ablation_multiseed_20260825.json + '
                'pi_ablation_smalln_20260825.json',
}


def select_pi_injection_mode(n_contacts, threshold=PI_INJECTION_SWITCH_THRESHOLD):
    """按接触者数量自适应选择 PI 注入路径（用户 P3 规格）。

    规则：n_contacts < 150 → 'feature_loss'（损失级注入——物理一致性
    约束 L_bounded/L_consistency 提供正则，小样本下优于特征级）；
    ≥ 150 → 'feature'（特征级注入——常规样本下略优且更简单）。

    证据（正式消融 2026-08-25，分层 50/50 + DeLong + 种子 bootstrap）：
      - n=400（20 种子）：特征级 − 损失级 = +0.0038，CI 不含 0；
      - n=200（12 种子）：−0.0029，CI 含 0（相当）；
      - n=100（12 种子）：−0.0083，CI 不含 0——损失级反超。
    交叉点在 n=100~200 之间，取 150 为切换阈值。

    注意：n=100 时 oracle 间隙本身已消失（oracle−seir 仅 +0.0054，
    CI 含 0）——模式选择的收益量级有限，本函数是稳健性配置而非
    性能开关。

    Args:
        n_contacts: 接触者数量（队列口径，训练+推断合计）
        threshold: 切换阈值（默认 150）

    Returns:
        str: 'feature' 或 'feature_loss'（train_pi_gnn 的 mode 取值）
    """
    if n_contacts < 0:
        raise ValueError('n_contacts 不能为负')
    return 'feature_loss' if n_contacts < threshold else 'feature'


class PIGNNNet(nn.Module if PYTORCH_AVAILABLE else object):
    """物理锚定 + 源池化残差融合网络。

    结构：类型化单层均值聚合（源专属，无参数）→ 浅读出（默认
    线性，~18 参数）→ tanh → 乘性融合。参数化定稿依据见模块
    docstring（v7-v11 扫描）。

    Args:
        node_dim: 池化消息维度（= 节点特征维度，含物理特征 = 17）
        residual_scale: 残差尺度 a——z/λ̃ ∈ [e^−a, e^a]
            （默认 1.0；有界残差的"界"）
        readout: 'linear'（默认，验证值）或 'mlp'（浅两层，
            两者实测等价）
        hidden_dim: readout='mlp' 时的隐藏维度
    """

    def __init__(self, node_dim, residual_scale=1.0, readout='linear',
                 hidden_dim=32):
        super().__init__()
        if not PYTORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError('PyTorch 不可用')
        self.residual_scale = residual_scale
        if readout == 'linear':
            self.delta_head = nn.Linear(node_dim, 1)
        elif readout == 'mlp':
            self.delta_head = nn.Sequential(
                nn.Linear(node_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1))
        else:
            raise ValueError("readout 必须是 'linear' 或 'mlp'")

    @staticmethod
    def source_pool(x, edge_index, num_index_cases):
        """源专属均值池化：接触者聚合其指示病例邻居的特征。

        只池化"指示病例 → 接触者"方向的边（节点 id < M 的邻居），
        不含同簇互连边——接触调查的二部结构先验。全邻居池化会因
        伙伴数不同的非均匀稀释干扰排序（v11 扫描：corr 0.48 vs
        0.96）。无参数、可微（梯度经 x 反传）。

        Args:
            x: [N, D] 节点特征
            edge_index: [2, E] 双向边
            num_index_cases: M（前 M 个节点为指示病例）
        """
        src, dst = edge_index
        m = (src < num_index_cases) & (dst >= num_index_cases)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst[m], x[src[m]])
        cnt = torch.zeros(x.shape[0], 1, dtype=x.dtype)
        cnt.index_add_(0, dst[m],
                       torch.ones(int(m.sum()), 1, dtype=x.dtype))
        return agg / cnt.clamp(min=1)

    def forward(self, x, edge_index, lam_tilde, num_index_cases):
        """前向：返回 (logit [N], z [N], t [N])。

        logit = z + log1p(−exp(−z)) 即 log-odds(1 − exp(−z))，
        BCEWithLogitsLoss 直接可用；z = λ̃·exp(a·t) 为融合后感染力。

        Args:
            x: [N, node_dim]（N = 指示病例数 + 接触者数）
            edge_index: LongTensor [2, E]（扁平同质图，双向边）
            lam_tilde: [N] 校准物理感染力（k̂·λ；指示病例为 0）
            num_index_cases: M（指示病例数，池化掩码用）
        """
        msg = self.source_pool(x, edge_index, num_index_cases)
        t = torch.tanh(self.delta_head(msg).squeeze(-1))       # (−1, 1)
        z = torch.clamp(lam_tilde * torch.exp(self.residual_scale * t),
                        min=1e-6)
        logit = z + torch.log1p(-torch.exp(-z))
        return logit, z, t


def physics_consistency_loss(z, t, lam_tilde, rho_min=0.5):
    """物理一致性损失（用户损失级注入：L_bounded + L_consistency）。

    Args:
        z: 预测感染力 [T]（训练接触者节点）
        t: tanh 残差激活 [T]
        lam_tilde: 校准物理感染力 [T]
        rho_min: 排序一致性下界（Pearson 代理 Spearman）

    Returns:
        dict: {'bounded', 'consistency'} 两个标量张量
    """
    l_bounded = t.pow(2).mean()
    lz = z - z.mean()
    ll = lam_tilde - lam_tilde.mean()
    denom = torch.sqrt(lz.pow(2).mean() * ll.pow(2).mean())
    if float(denom.detach()) < 1e-12:        # λ̃ 退化（全零）时跳过
        l_consistency = z.new_zeros(())
    else:
        rho = (lz * ll).mean() / denom
        l_consistency = torch.relu(
            torch.as_tensor(float(rho_min), dtype=z.dtype) - rho)
    return {'bounded': l_bounded, 'consistency': l_consistency}


def train_pi_gnn(net, labels, train_contacts, mode='auto', epochs=800,
                 lr=0.05, weight_decay=3e-3, hidden_dim=32, seed=0,
                 residual_scale=1.0, readout='linear', lam_bounded=0.1,
                 lam_consist=0.1, rho_min=0.5):
    """在 PI 网络上训练 PIGNNNet（确定性，单线程）。

    物理校准 k̂ 只用训练半区标签（SEIR 冻结 + 单参数校准 = 用户
    方案 1 的最简起步）；训练损失只作用于训练半区接触者。

    默认超参为 v11 扫描验证值：线性读出、epochs=800、lr=0.05、
    wd=3e-3（wd 必要——2 参数亦会过拟合）。

    Args:
        net: build_pi_network 输出
        labels: 接触者标签（长度 = 接触者数）
        train_contacts: 训练半区接触者下标（0-based，相对接触者数组）
        mode: 'auto'（默认，P3 产品化：<150 接触者自动用损失级，
              ≥150 用特征级——select_pi_injection_mode）、
              'feature'（特征级注入，仅 L_label）或
              'feature_loss'（+ 物理一致性损失）
        readout: 'linear'（默认）或 'mlp'
        lam_bounded / lam_consist: 损失权重（feature_loss 模式）
        rho_min: 排序一致性下界

    Returns:
        dict: score（接触者 logit，AUROC 用）、p（=1−exp(−z)，校准
        用）、delta_t（tanh 残差激活）、k_hat、mode（auto 解析后的
        实际注入路径）
    """
    if not PYTORCH_AVAILABLE:
        raise RuntimeError('PyTorch 不可用，无法训练 PI-GNN')
    from ...validation.pi_network import to_pi_graph, calibrate_physics_k

    if mode == 'auto':
        mode = select_pi_injection_mode(len(labels))
    elif mode not in ('feature', 'feature_loss'):
        raise ValueError("mode 必须是 'auto'、'feature' 或 'feature_loss'")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    M = net['M']
    labels = np.asarray(labels, dtype=np.float64)
    tr_contacts = np.asarray(train_contacts, dtype=np.int64)

    k_hat = calibrate_physics_k(net, labels, tr_contacts)
    g = to_pi_graph(net, include_physics=True, k_hat=k_hat)
    x, ei, lam_t = g['x'], g['edge_index'], g['lam_tilde']
    y_full = torch.zeros(x.shape[0], dtype=torch.float32)
    y_full[M:] = torch.tensor(labels, dtype=torch.float32)
    tr_nodes = torch.tensor(M + tr_contacts, dtype=torch.long)

    model = PIGNNNet(x.shape[1], residual_scale=residual_scale,
                     readout=readout, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits, z, t = model(x, ei, lam_t, M)
        loss = bce(logits[tr_nodes], y_full[tr_nodes])
        if mode == 'feature_loss':
            pc = physics_consistency_loss(z[tr_nodes], t[tr_nodes],
                                          lam_t[tr_nodes], rho_min)
            loss = loss + lam_bounded * pc['bounded'] \
                + lam_consist * pc['consistency']
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits, z, t = model(x, ei, lam_t, M)
    p = torch.sigmoid(logits)                 # = 1 − exp(−z)
    return {
        'score': logits[M:].numpy().astype(np.float64),
        'p': p[M:].numpy().astype(np.float64),
        'delta_t': t[M:].numpy().astype(np.float64),
        'k_hat': float(k_hat),
        'mode': mode,
    }


def train_pure_gnn(net, labels, train_contacts, epochs=500, lr=0.02,
                   weight_decay=3e-3, hidden_dim=32, seed=0):
    """纯 GNN 对照臂：14 维基础特征端到端（无物理注入）。

    复用 TypedGATNet（num_edge_types=1，同质单通道 + 边特征注意力）
    ——与 PI 臂同 GAT 组件、同训练预算，唯一差异是无物理特征与
    融合结构（公平消融口径）。

    Returns:
        dict: score（接触者 logit）、p（sigmoid 概率）
    """
    if not PYTORCH_AVAILABLE:
        raise RuntimeError('PyTorch 不可用，无法训练纯 GNN')
    from ...validation.pi_network import to_pi_graph

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    M = net['M']
    labels = np.asarray(labels, dtype=np.float64)
    tr_contacts = np.asarray(train_contacts, dtype=np.int64)

    g = to_pi_graph(net)                      # 14 维基础特征（无物理）
    x, ei, ea = g['x'], g['edge_index'], g['edge_attr']
    y_full = torch.zeros(x.shape[0], dtype=torch.float32)
    y_full[M:] = torch.tensor(labels, dtype=torch.float32)
    tr_nodes = torch.tensor(M + tr_contacts, dtype=torch.long)

    model = TypedGATNet(x.shape[1], hidden_dim=hidden_dim, num_edge_types=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x, [ei], [ea])
        loss = bce(logits[tr_nodes], y_full[tr_nodes])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(x, [ei], [ea])
    p = torch.sigmoid(logits)
    return {
        'score': logits[M:].numpy().astype(np.float64),
        'p': p[M:].numpy().astype(np.float64),
    }
