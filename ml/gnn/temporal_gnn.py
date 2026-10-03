#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量时序 GNN：时间切片子图聚合 + 窗间 GRU 状态传递。

背景（用户"第二层：时序 GNN（接触时间维度）"，2026-08-25）：
  接触网络不是静态的——把网络按时间切片（每窗一张快照），窗内做
  GAT 聚合、窗间用 RNN 传递时序状态，建模风险随接触时间的演化。
  本模块给出最小可训练组合，接入 validation/temporal_network.py
  （v1：时间切片 + 衰减权重的信息不对称构造）的时序接触网络：

  - **TemporalGNN**（snapshot 序列形态，用户规格"子图内聚合 +
    子图间状态传递"的字面实现）：节点嵌入 → 每窗 HeteroGATLayer
    聚合（同一静态基表示 + 该窗子图，参数跨窗共享）+ 残差门控 →
    窗序列（最旧 → 最新）进 GRU 传递时序状态 → 每步 readout 输出
    **时序风险曲线**（最新窗 readout = 当前风险）；
  - **train_temporal_gnn**：BCE + Adam，单线程 + 固定 seed 确定性
    训练；**损失只作用于接触者节点、只监督最后一窗**（标签是
    "当前感染状态"，中间窗无标签——不是深度监督）；
  - **时序盲对照**（windows_merged=True）：全部窗的边并入单一
    静态子图（窗标注抹除 + 跨窗重复边去重），同一架构跑单步——
    隔离"时间切片 + 窗间状态传递"贡献的严格消融对照（同结构、
    同预算、同训练协议）。

架构选型依据（文献调研，2026-08-25）：
  - snapshot 序列 + 窗间 RNN 是时序 GNN 的标准形态（TGAT/TGN 的
    离散化特例；UTG 统一框架显示 snapshot 模型推理快一个数量级）；
  - SE-HTGNN（NeurIPS 2025）消融：GRU 与 LSTM 性能相当且更高效，
    Transformer/Mamba 变体反而欠佳——故窗间递归选 GRU。

超参起点（沿用 typed_gat 2026-08-25 验证值，如欠拟合再调）：
  hidden=32、dropout=0.4、gate_init=0.0、heads=4、epochs=500、
  lr=0.02、weight_decay=3e-3。

与 scoring/ml/gnn_training/training_stgnn.py 的关系：该骨架用
SEIRTimeAwareGNN + 合成图数据、未接入主流程；本模块回答第二层
问题——时序 GNN 能否把"接触时间分布"信息兑现为判别增量，接入
validation/ 消融体系。

文献：
  Li Y et al. (2018) ICLR —— DCRNN：扩散卷积 + 递归的时空图网络
      （GNN 输入 = RNN 隐状态的递归结构）；
  Rossi E et al. (2020) ICLR —— TGN：时间图网络（记忆模块）；
  Xu D et al. (2024) —— UTG 统一时序图框架（snapshot vs 事件流）；
  SE-HTGNN (NeurIPS 2025) —— GRU 优于 Transformer 的时序门控实证。
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
    from ._layers import HeteroGATLayer


class TemporalGNN(nn.Module if PYTORCH_AVAILABLE else object):
    """轻量时序 GNN：每窗 GAT 聚合 + 窗间 GRU 状态传递。

    前向（snapshot 序列形态，用户规格字面实现；窗序列按最旧 →
    最新排序，最后一步隐状态 = 当前风险）：

        h = relu(embed(x))                         # [N, D] 静态基表示
        for 每窗 w（最旧 → 最新）:
            z_w = h + sigmoid(gate)·GAT(h, 边_w)   # 窗内聚合（残差门控）
        seq = [z_最旧, ..., z_最新]                 # [N, W, D]
        out = GRU(seq)                             # 窗间状态传递
        curve[w] = risk_head(out[:, w])            # 该窗风险 readout
        return curve                               # [W, N] 时序风险曲线

    Args:
        node_dim: 节点特征维度（temporal_network.node_features = 12）
        hidden_dim: 隐藏层维度（验证值 32）
        num_windows: 时间窗数（默认 5；时序盲对照用 1）
        dropout: dropout 比例（验证值 0.4）
        gate_init: 残差门初始 logit（sigmoid(gate) 为初始步长；
            验证值 0.0 → 初始步长 0.5）
    """

    def __init__(self, node_dim, hidden_dim=32, num_windows=5,
                 dropout=0.4, gate_init=0.0):
        super().__init__()
        self.num_windows = num_windows
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        # 每窗聚合层：参数跨窗共享（STGNN 标准做法，窗"深度"来自
        # GRU 递归而非堆叠层）
        self.window_gat = HeteroGATLayer(
            hidden_dim, hidden_dim, num_edge_types=1, heads=4,
            dropout=dropout)
        self.residual_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x, windows):
        """前向：返回时序风险曲线 logits（[W, N]，最后一行 = 最新窗）。

        用户规格的字面实现（snapshot 序列形态）：
        每窗从同一静态基表示独立聚合（窗内 GAT + 残差），窗序列
        （最旧 → 最新）进 GRU 传递时序状态，每步 readout 输出该窗
        风险——最新窗 readout = 当前风险。

        （注：曾试 DCRNN 式递归（GAT 输入 = 上一步时序状态），但
        状态注入使计算链加深、小样本下记忆化训练集（train loss →
        0.02 而测试 AUROC 反降）；snapshot 序列形态更贴用户规格
        且更正则化，2026-08-25 实测定稿。）

        Args:
            x: [N, node_dim]（N = 指示病例数 + 接触者数）
            windows: 窗序列（**最旧 → 最新**），每项
                {'edge_index': LongTensor [2, E_w],
                 'edge_attr': FloatTensor [E_w, 6]}
        """
        h = torch.relu(self.node_embedding(x))       # [N, D] 静态基表示
        gate = torch.sigmoid(self.residual_gate)
        seq = []
        for w in windows:
            # 窗内聚合：同一基表示 + 该窗子图（残差门控）
            g = self.window_gat(h, [w['edge_index']], [w['edge_attr']])
            seq.append(h + gate * g)
        seq = torch.stack(seq, dim=1)                # [N, W, D]
        out, _ = self.gru(seq)                       # [N, W, D] 窗间状态
        curve = self.risk_head(out).squeeze(-1)      # [W, N] 每窗 readout
        return curve.transpose(0, 1)                 # 最旧 → 最新


def train_temporal_gnn(net, labels, train_contacts, epochs=500, lr=0.02,
                       weight_decay=3e-3, hidden_dim=32, seed=0,
                       windows_merged=False):
    """在时序接触网络上训练 TemporalGNN（确定性，单线程）。

    训练损失只作用于接触者节点、只监督最后一窗（最新窗 readout =
    当前风险；中间窗无标签，不深度监督）。指示病例作为传染源节点
    参与每窗消息传递。

    Args:
        net: build_temporal_network 输出（v1）
        labels: 接触者标签（长度 = 接触者数）
        train_contacts: 训练接触者下标（**接触者空间** 0..n-1；
            测试接触者不进损失——外部验证语义）
        epochs / lr / weight_decay / hidden_dim: 训练超参（默认为
            2026-08-25 验证值起点）
        seed: torch 种子（确定性训练）
        windows_merged: True → 时序盲对照（全部窗的边并入单一静态
            子图，同架构单步），隔离"时间切片 + 窗间状态传递"贡献

    Returns:
        dict: scores（接触者当前风险 0-1，= 最新窗 readout，长度 =
        接触者数）、risk_curve（[W, n_contacts] 时序风险曲线，
        最旧 → 最新）、initial_loss / final_loss、epochs、
        windows_merged
    """
    if not PYTORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError('PyTorch 不可用，无法训练 TemporalGNN')

    from ...validation.temporal_network import to_stgnn_graph

    torch.set_num_threads(1)      # scatter 确定性（单线程）
    torch.manual_seed(seed)
    g = to_stgnn_graph(net, windows_merged=windows_merged)
    x, windows = g['x'], g['windows']

    model = TemporalGNN(node_dim=x.shape[1], hidden_dim=hidden_dim,
                        num_windows=len(windows))
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    M = int(net['M'])
    y = torch.tensor(np.asarray(labels, dtype=np.float32))
    tr_contacts = torch.tensor(np.asarray(train_contacts, dtype=np.int64))
    tr_nodes = tr_contacts + M          # 图节点空间（接触者段从 M 起）

    def _loss():
        curve = model(x, windows)              # [W, N]
        logits = curve[-1]                     # 最新窗 = 当前风险
        return nn.functional.binary_cross_entropy_with_logits(
            logits[tr_nodes], y[tr_contacts])

    model.train()
    initial_loss = float(_loss().item())
    final_loss = initial_loss
    for _ in range(epochs):
        opt.zero_grad()
        loss = _loss()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
    model.eval()
    with torch.no_grad():
        curve = model(x, windows)              # [W, N]
        probs = torch.sigmoid(curve)
        scores = probs[-1][M:]                 # 当前风险（最新窗）
        risk_curve = probs[:, M:]              # 时序风险曲线（最旧→最新）
    return {
        'scores': scores.numpy().tolist(),
        'risk_curve': risk_curve.numpy().tolist(),
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'epochs': epochs,
        'windows_merged': bool(windows_merged),
    }
