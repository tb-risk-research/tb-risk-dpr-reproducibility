#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEIR-informed 异质图神经网络（方向二核心模型）。"""

from ._common import (
    _GNNFrameworkPlaceholder, PYTORCH_AVAILABLE, PYG_AVAILABLE,
    nn, F, torch, global_mean_pool, GATv2Conv,
    seir_ode_gating, _ODE_GATING_AVAILABLE,
)

SEIRInformedGNN = _GNNFrameworkPlaceholder

if PYTORCH_AVAILABLE and PYG_AVAILABLE:
    class SEIRInformedGNN(nn.Module):
        """
        SEIR-informed异质图神经网络（方向二核心）

        v2.1: 门控升级为 SEIR ODE 积分器。
        用积分结果（峰值感染率、AUC、疫情规模）调制节点表示。
        保留 use_ode_gating=False 回退到纯特征缩放模式。
        v3.0 起：默认改用四阶 RK4 求解器（全局误差 O(dt⁴)），
        精度优于一阶 Euler 的 O(dt)，可减少冗余步数。

        v3.0: 门控输出从 3 通道扩展为 6 通道，对应 9-房室 SEIR 模型；
              节点特征维度从 22 扩展为 30（新增潜伏/疾病状态 one-hot）。
              seir_ode_gating 自动检测通道数并分发到 4D/9D 积分。

        文献支撑：
        - HHAN (Zhang et al., 2025 AAAI): 异质超图神经网络用于传染病溯源
        - GAST (PLOS ONE, 2025): 图注意力时空网络用于流行病建模
        - MPUGAT: 多补片图注意力网络用于时变传播率估计
        - Kermack & McKendrick (1927) SEIR 模型
        - Chen et al. (2018) Neural ODE, NeurIPS
        - Houben et al. (2016) BMC Med — 快/慢分层
        - Horton et al. (2023) PNAS — 自清除
        - Emery et al. (2023) eLife — 亚临床传染性

        核心创新：
        1. SEIR-informed消息传递：将流行病学先验作为归纳偏置嵌入GNN
        2. 三层异质图：患者-家庭-社会-社区层次结构
        3. 时变注意力：根据治疗阶段动态调整传播率权重
        4. 边特征增强：融合接触频率、时长、场景等边特征
        5. SEIR ODE 门控：RK4 积分替代纯特征缩放（v2.1 引入, v3.0 起默认 RK4）
        6. v3.0 6 通道门控：对应 9-房室 SEIR 关键流参数
        """

        def __init__(self, node_feature_dim=30, edge_feature_dim=6, hidden_dim=64, num_layers=1, dropout=0.3):
            super().__init__()

            self.node_feature_dim = node_feature_dim
            self.edge_feature_dim = edge_feature_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            # 节点嵌入投影
            self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)

            # 边特征嵌入投影（新增！）
            self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim)

            # SEIR-informed图注意力层（GATv2Conv with edge_dim，边特征参与注意力计算）
            # 文献：Brody et al. (2022) "How Attentive are Graph Attention Networks?" ICLR
            self.gat_layers = nn.ModuleList()
            for _ in range(num_layers):
                self.gat_layers.append(
                    GATv2Conv(hidden_dim, hidden_dim, heads=4, dropout=dropout,
                              concat=False, edge_dim=hidden_dim)
                )

            # 治疗阶段感知特征调制（SEIR ODE 门控模块，改进7+8: 8 通道对应 9-房室 SEIR）
            self.seir_gate = nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim),  # +1用于治疗阶段特征
                nn.ReLU(),
                nn.Linear(hidden_dim, 8),  # 改进7+8: 8 路门控 (β, ρ_fast, σ_clear, ρ_prog, γ, ω_reg, ρ_react, η_sub)
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

        def forward(self, x, edge_index, edge_attr=None, treatment_phase=None, batch=None,
                use_ode_gating=True):
            """
            参数：
                x: 节点特征矩阵 [num_nodes, node_feature_dim]
                edge_index: 图边索引 [2, num_edges]
                edge_attr: 边特征矩阵 [num_edges, edge_feature_dim] (可选)
                treatment_phase: 治疗阶段特征 [num_nodes, 1] (可选，用于SEIR ODE 门控)
                batch: 批量训练时的图索引 [num_nodes]，指示每个节点属于哪个图
                use_ode_gating: 是否使用 ODE 门控（v2.1），False 回退到特征缩放

            文献支撑：
                - GATv2Conv支持edge_attr: Zhu et al., GAST 2024 (PLOS ONE)
                - 边特征重要性：Yang et al., HGAT-AMR 2021
                - SEIR ODE 门控: Kermack & McKendrick (1927); Chen et al. (2018) Neural ODE
            """
            # 初始节点嵌入
            h = self.node_embedding(x)
            h = F.relu(h)
            h = self.dropout(h)

            # 初始边特征嵌入（如果有edge_attr）
            edge_emb = None
            if edge_attr is not None and edge_attr.shape[1] > 0:
                edge_emb = self.edge_embedding(edge_attr)
                edge_emb = F.relu(edge_emb)

            # SEIR-informed图注意力传播（支持edge_attr）
            for i, gat_layer in enumerate(self.gat_layers):
                # GAT卷积（根据是否有edge_attr选择调用方式）
                if edge_emb is not None:
                    h_new = gat_layer(h, edge_index, edge_attr=edge_emb)
                else:
                    h_new = gat_layer(h, edge_index)

                # 残差连接
                h = h + h_new
                h = F.relu(h)
                h = self.dropout(h)

            # SEIR 门控：v2.1 ODE 积分器 或 回退特征缩放
            if treatment_phase is not None:
                if use_ode_gating and _ODE_GATING_AVAILABLE:
                    # v2.1: SEIR ODE 门控 — Euler 积分调制节点表示
                    h, _ode_metrics = seir_ode_gating(h, treatment_phase, self.seir_gate)
                else:
                    # 回退：纯特征缩放门控（改进7+8: 8 通道）
                    gate_input = torch.cat([h, treatment_phase], dim=-1)
                    gate_weights = self.seir_gate(gate_input)  # [B, 8]
                    phase = treatment_phase.squeeze(-1)
                    # 8 通道：β, ρ_fast, σ_clear, ρ_prog, γ, ω_reg, ρ_react, η_sub
                    w_trans = torch.sigmoid(-(phase - 1.5) * 2.0)
                    w_fast = torch.sigmoid(-(phase - 2.0) * 2.0) * \
                             torch.sigmoid((phase - 0.5) * 3.0)
                    w_clear = torch.sigmoid((phase - 1.0) * 1.5)
                    w_prog = torch.sigmoid((phase - 1.5) * 2.0)
                    w_recov = torch.sigmoid((phase - 1.5) * 2.0)
                    w_reg = torch.ones_like(phase) * 0.5
                    w_react = torch.sigmoid(-(phase - 3.0) * 1.5)
                    w_eta = torch.ones_like(phase) * 0.5
                    w_sum = (w_trans + w_fast + w_clear + w_prog + w_recov
                             + w_reg + w_react + w_eta + 1e-8)
                    ch = gate_weights.unbind(dim=-1)
                    w_list = [w_trans, w_fast, w_clear, w_prog, w_recov,
                              w_reg, w_react, w_eta]
                    gated_modulation = sum(
                        (w / w_sum).unsqueeze(-1) * c.unsqueeze(-1)
                        for w, c in zip(w_list, ch)
                    )
                    h = h * gated_modulation

            # 图级池化（全局平均）
            if batch is None:
                # 单图情况：构造全零索引
                batch_idx = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            else:
                batch_idx = batch
            graph_emb = global_mean_pool(h, batch_idx)

            # 节点级风险预测
            risk_scores = self.risk_predictor(h)

            return risk_scores, graph_emb
