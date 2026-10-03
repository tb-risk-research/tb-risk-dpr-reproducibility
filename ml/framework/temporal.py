#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序编码模块与时间感知 GNN（方案三）。"""

from ._common import (
    _GNNFrameworkPlaceholder, PYTORCH_AVAILABLE, PYG_AVAILABLE,
    nn, F, torch, global_mean_pool, GATv2Conv,
    seir_ode_gating, _ODE_GATING_AVAILABLE,
)

TemporalEncoder = _GNNFrameworkPlaceholder
SEIRTimeAwareGNN = _GNNFrameworkPlaceholder

if PYTORCH_AVAILABLE and PYG_AVAILABLE:
    class TemporalEncoder(nn.Module):
        """
        时序编码模块（基于 GRU）

        用途：将节点的历史特征序列编码为当前时刻的隐状态

        文献支撑：
        - DCRNN (Diffusion Convolutional Recurrent Neural Network, Li et al. 2018)
        - 使用 GRU 建模时间序列依赖关系

        参数：
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: GRU 层数
        """

        def __init__(self, input_dim, hidden_dim, num_layers=2):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.gru = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )

            self.layer_norm = nn.LayerNorm(hidden_dim)

        def forward(self, x_seq):
            """
            参数：
                x_seq: 时序特征 [batch, seq_len, input_dim] 或 [num_nodes, seq_len, input_dim]

            返回：
                h_n: 最后一个时间步的隐状态 [batch, hidden_dim] 或 [num_nodes, hidden_dim]
            """
            # GRU 前向传播
            _, h_n = self.gru(x_seq)

            # 取最后一层的隐状态
            h_last = h_n[-1]

            # 层归一化
            h_last = self.layer_norm(h_last)

            return h_last


    class SEIRTimeAwareGNN(nn.Module):
        """
        时间感知的图神经网络（方案三：最贴合业务）

        v2.1: 门控升级为 SEIR ODE 积分器。
        使用 Euler 方法对 SEIR 动力学方程进行数值积分，
        用积分结果（峰值感染率、AUC、疫情规模）调制节点表示。
        保留 use_ode_gating=False 回退到纯特征缩放模式。

        核心机制：
        1. 将治疗阶段衰减作为时间注意力权重
        2. 结合时序编码的历史信息
        3. SEIR ODE 门控：Euler 积分替代纯特征缩放（v2.1）
        4. 保持与现有 SEIRInformedGNN 架构的兼容性

        文献支撑：
        - WHO 2024 指南中的"接触者追踪时间窗"概念
        - Kermack & McKendrick (1927) SEIR 模型
        - Chen et al. (2018) Neural ODE, NeurIPS

        参数：
            node_feature_dim: 节点特征维度（默认 30，v3.0）
            edge_feature_dim: 边特征维度（默认 6）
            hidden_dim: 隐藏层维度
            num_layers: GNN 层数
            dropout: Dropout 概率
        """

        def __init__(self, node_feature_dim=30, edge_feature_dim=6, hidden_dim=64, num_layers=1, dropout=0.3):
            super().__init__()

            self.node_feature_dim = node_feature_dim
            self.edge_feature_dim = edge_feature_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            # 节点嵌入投影
            self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)

            # 边特征嵌入投影
            self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim)

            # 时间注意力模块（方案三核心：输入治疗周数）
            self.time_attention = nn.Linear(1, hidden_dim)

            # 时序编码模块（用于处理历史序列）
            self.temporal_encoder = TemporalEncoder(node_feature_dim, hidden_dim, num_layers=2)

            # GATv2Conv 图注意力层（带边特征，文献：Brody et al. 2022 ICLR）
            self.gat_layers = nn.ModuleList()
            for _ in range(num_layers):
                self.gat_layers.append(
                    GATv2Conv(hidden_dim, hidden_dim, heads=4, dropout=dropout,
                              concat=False, edge_dim=hidden_dim)
                )

            # 治疗阶段感知特征调制（替代 SEIR 动力学门控，不执行 ODE 积分）
            self.seir_gate = nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim),  # +1用于治疗阶段特征
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),  # 输出 3 路门控：传播倾向、潜伏风险、恢复能力
                nn.Sigmoid()
            )

            # 风险预测头
            self.risk_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index, treatment_week=None, edge_attr=None, x_seq=None, batch=None,
                use_ode_gating=True):
            """
            参数：
                x: 当前时刻节点特征 [num_nodes, node_feature_dim]
                edge_index: 图边索引 [2, num_edges]
                treatment_week: 每个节点的已治疗周数 [num_nodes, 1]（可选）
                edge_attr: 边特征 [num_edges, edge_feature_dim]（可选）
                x_seq: 历史时序特征 [num_nodes, seq_len, node_feature_dim]（可选）
                batch: 批量训练时的图索引 [num_nodes]，指示每个节点属于哪个图
                use_ode_gating: 是否使用 ODE 门控（v2.1），False 回退到特征缩放

            返回：
                risk_scores: 每个节点的风险评分 [num_nodes, 1]
                graph_emb: 图级嵌入 [1, hidden_dim]
            """
            # 初始节点嵌入
            h = self.node_embedding(x)
            h = F.relu(h)
            h = self.dropout(h)

            # 使用时序编码增强（如果提供 x_seq）
            if x_seq is not None:
                temporal_emb = self.temporal_encoder(x_seq)
                h = h + temporal_emb  # 残差融合
                h = F.relu(h)

            # 时间注意力加权（方案三核心）
            if treatment_week is not None:
                # time_weight = sigmoid(W*treatment_week + b)
                time_weight = torch.sigmoid(self.time_attention(treatment_week))
                h = h * time_weight  # 时间加权特征

            # 初始边特征嵌入（如果有edge_attr）
            edge_emb = None
            if edge_attr is not None and edge_attr.shape[1] > 0:
                edge_emb = self.edge_embedding(edge_attr)
                edge_emb = F.relu(edge_emb)

            # 图注意力传播（支持edge_attr）
            for i, gat_layer in enumerate(self.gat_layers):
                if edge_emb is not None:
                    h_new = gat_layer(h, edge_index, edge_attr=edge_emb)
                else:
                    h_new = gat_layer(h, edge_index)

                h = h + h_new
                h = F.relu(h)
                h = self.dropout(h)

            # SEIR 门控：v2.1 ODE 积分器 或 回退特征缩放
            # treatment_week 作为 treatment_phase 的代理（周数映射到阶段）
            if treatment_week is not None:
                if use_ode_gating and _ODE_GATING_AVAILABLE:
                    # v2.1: SEIR ODE 门控 — Euler 积分调制节点表示
                    h, _ode_metrics = seir_ode_gating(h, treatment_week, self.seir_gate)
                else:
                    # 回退：纯特征缩放门控（保持向后兼容）
                    gate_input = torch.cat([h, treatment_week], dim=-1)
                    gate_weights = self.seir_gate(gate_input)
                    trans_tendency, latent_risk, recovery_capacity = gate_weights.unbind(dim=-1)
                    week = treatment_week.squeeze(-1)
                    w_trans = torch.sigmoid(-(week - 8.0) / 4.0)
                    w_latent = torch.sigmoid(-(week - 12.0) / 6.0)
                    w_recovery = torch.sigmoid((week - 8.0) / 4.0)
                    w_sum = w_trans + w_latent + w_recovery + 1e-8
                    w_trans = w_trans / w_sum
                    w_latent = w_latent / w_sum
                    w_recovery = w_recovery / w_sum
                    gated_modulation = (
                        w_trans.unsqueeze(-1) * trans_tendency.unsqueeze(-1) +
                        w_latent.unsqueeze(-1) * latent_risk.unsqueeze(-1) +
                        w_recovery.unsqueeze(-1) * recovery_capacity.unsqueeze(-1)
                    )
                    h = h * gated_modulation

            # 图级池化
            if batch is None:
                # 单图情况：构造全零索引
                batch_idx = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            else:
                batch_idx = batch
            graph_emb = global_mean_pool(h, batch_idx)

            # 节点级风险预测
            risk_scores = self.risk_predictor(h)

            return risk_scores, graph_emb
