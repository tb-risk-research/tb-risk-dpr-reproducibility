#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量类型感知 GAT：HeteroGATLayer 骨架的首个训练接入（2 层 + 残差）。

背景（用户"第一层：异质边 + 注意力"，2026-08-25）：
  HeteroTimeVaryingGNN / HeteroGATLayer 骨架长期停留在"有 forward、
  没在真实场景训练验证"。本模块给出最小可训练组合，接入
  validation/typed_network.py（v3：传染源节点 + 同分布边特征）的
  类型化接触网络：

  - **TypedGATNet**：节点嵌入 → 2 层 HeteroGATLayer（残差门控连接，
    用户规格："2 层就够，太多层会过平滑"）→ 风险 readout；
  - **train_typed_gat**：BCE + Adam，单线程 + 固定 seed 确定性训练；
    **损失只作用于接触者节点**（指示病例无标签），输出接触者分数
    （半区训练/半区评估由调用方控制——不读取测试标签）；
  - **同质对照**（edge_types_merged=True）：把 5 类边全部并入单一类型
    通道，注意力参数无类型区分——隔离"类型感知"贡献的严格消融对照
    （同结构、同预算、同训练协议）。

超参（2026-08-25 调优验证，10 种子）：hidden=32、dropout=0.4、
gate_init=0.0（残差门 sigmoid(0)=0.5，放开 GAT 步长）、heads=4、
epochs=500、lr=0.02、weight_decay=3e-3。

与 HeteroTimeVaryingGNN 的关系：本模块不启用 SEIR ODE 门控/时序注意力/
因果约束（第二~四层机制），只回答第一层问题——异质边注意力能否把
"接触类型构成"信息兑现为判别增量。

文献：
  Veličković P et al. (2018) ICLR —— GAT 图注意力网络
  Kipf & Welling (2017) ICLR —— GCN 与过平滑
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


class TypedGATNet(nn.Module if PYTORCH_AVAILABLE else object):
    """轻量类型感知 GAT：2 层 HeteroGATLayer + 残差门控 + 风险 readout。

    Args:
        node_dim: 节点特征维度（typed_network.node_features 输出 = 12）
        hidden_dim: 隐藏层维度（验证值 32）
        num_edge_types: 边类型通道数（默认 5；同质对照用 1）
        num_layers: GAT 层数（默认 2，用户规格）
        dropout: dropout 比例（验证值 0.4）
        gate_init: 残差门初始 logit（sigmoid(gate) 为初始步长；
            验证值 0.0 → 初始步长 0.5）
    """

    def __init__(self, node_dim, hidden_dim=32, num_edge_types=5,
                 num_layers=2, dropout=0.4, gate_init=0.0):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.num_layers = num_layers
        self.residual = True
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([
            HeteroGATLayer(hidden_dim, hidden_dim,
                           num_edge_types=num_edge_types, heads=4,
                           dropout=dropout)
            for _ in range(num_layers)
        ])
        # 残差门（每层一个可学习标量：h ← h + sigmoid(gate)·GAT(h)）
        self.residual_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(float(gate_init)))
            for _ in range(num_layers)
        ])
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x, edge_indices, edge_attrs):
        """前向：返回全节点风险 logit（BCEWithLogitsLoss 直接可用）。

        Args:
            x: [N, node_dim]（N = 指示病例数 + 接触者数）
            edge_indices: num_edge_types 个 LongTensor [2, E_k]
            edge_attrs: num_edge_types 个 FloatTensor [E_k, 6]
        """
        h = torch.relu(self.node_embedding(x))
        for layer, gate in zip(self.gat_layers, self.residual_gates):
            h = h + torch.sigmoid(gate) * layer(h, edge_indices, edge_attrs)
            h = torch.relu(h)
        return self.risk_head(h).squeeze(-1)       # [N] logit


def train_typed_gat(net, labels, train_contacts, epochs=500, lr=0.02,
                    weight_decay=3e-3, hidden_dim=32, seed=0,
                    edge_types_merged=False):
    """在类型化网络上训练 TypedGATNet（确定性，单线程）。

    训练损失只作用于接触者节点（指示病例无标签）；指示病例仍作为
    传染源节点参与消息传递（其 infectivity 经图传播影响接触者表征）。

    Args:
        net: build_typed_network 输出（v3）
        labels: 接触者标签（长度 = 接触者数）
        train_contacts: 训练接触者下标（**接触者空间** 0..n-1；
            测试接触者不进损失——外部验证语义）
        epochs / lr / weight_decay / hidden_dim: 训练超参（默认为
            2026-08-25 验证值）
        seed: torch 种子（确定性训练）
        edge_types_merged: True → 同质对照（5 类边并入单通道，
            注意力无类型区分），隔离"类型感知"贡献

    Returns:
        dict: scores（接触者 0-1 风险，长度 = 接触者数）、
        initial_loss / final_loss、epochs、edge_types_merged
    """
    if not PYTORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError('PyTorch 不可用，无法训练 TypedGATNet')

    from ...validation.typed_network import to_gat_graph

    torch.set_num_threads(1)      # scatter 确定性（单线程）
    torch.manual_seed(seed)
    g = to_gat_graph(net, edge_types_merged=edge_types_merged)
    x, edge_indices, edge_attrs = g['x'], g['edge_indices'], g['edge_attrs']
    n_types = len(edge_indices)

    model = TypedGATNet(node_dim=x.shape[1], hidden_dim=hidden_dim,
                        num_edge_types=n_types)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    M = int(net['M'])
    y = torch.tensor(np.asarray(labels, dtype=np.float32))
    tr_contacts = torch.tensor(np.asarray(train_contacts, dtype=np.int64))
    tr_nodes = tr_contacts + M          # 图节点空间（接触者段从 M 起）

    def _loss():
        logits = model(x, edge_indices, edge_attrs)
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
        logits = model(x, edge_indices, edge_attrs)
        scores = torch.sigmoid(logits).numpy()[M:]   # 仅接触者
    return {
        'scores': scores.tolist(),
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'epochs': epochs,
        'edge_types_merged': bool(edge_types_merged),
    }
