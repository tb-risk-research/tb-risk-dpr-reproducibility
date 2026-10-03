#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时空结核病 GNN（ST-GCN 架构）。"""

from ._common import (
    _GNNFrameworkPlaceholder, PYTORCH_AVAILABLE, PYG_AVAILABLE,
    nn, torch,
)
from .temporal import SEIRTimeAwareGNN

SpatioTemporalTBGNN = _GNNFrameworkPlaceholder

if PYTORCH_AVAILABLE and PYG_AVAILABLE:
    class SpatioTemporalTBGNN(nn.Module):
        """
        完整的时空结核病 GNN（方案二：更强大的 ST-GCN 架构）

        结合时间卷积 + 图卷积处理时空依赖

        文献支撑：
        - ST-GCN (Spatial Temporal Graph Convolutional Network, Yan et al. 2018)
        - ASTGNN (Attention Spatial-Temporal Graph Neural Network, Guo et al. 2021)

        参数：
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度
            hidden_dim: 隐藏层维度
            time_steps: 时间步数（默认 52 周）
            num_layers: GNN 层数
        """

        def __init__(self, node_feature_dim=30, edge_feature_dim=6, hidden_dim=64, time_steps=52, num_layers=1):
            super().__init__()

            self.time_steps = time_steps

            # 时间卷积层（1D 卷积处理时间维度）
            self.temporal_conv = nn.Conv1d(
                in_channels=node_feature_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
            )

            # 时间感知的 GNN
            self.seir_time_gnn = SEIRTimeAwareGNN(
                node_feature_dim=hidden_dim,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )

            # 风险预测头（输出时序风险曲线）
            self.temporal_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, time_steps),
                nn.Sigmoid()
            )

        def forward(self, x_seq, edge_index, treatment_week_seq=None, edge_attr=None):
            """
            参数：
                x_seq: 时序特征 [batch, time_steps, num_nodes, node_feature_dim]
                edge_index: 图边索引 [2, num_edges]（静态图，可扩展为动态边）
                treatment_week_seq: 时序治疗周数 [batch, time_steps, num_nodes, 1]（可选）
                edge_attr: 边特征 [num_edges, edge_feature_dim]（可选）

            返回：
                risk_curve: 每个节点在未来各时间点的风险 [batch, time_steps, num_nodes]
            """
            batch_size, time_steps, num_nodes, feat_dim = x_seq.shape

            # 对每个节点的时间序列应用 1D 时间卷积
            x_reshaped = x_seq.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, feat_dim, time_steps)
            temporal_conv_out = self.temporal_conv(x_reshaped)
            temporal_conv_out = temporal_conv_out.permute(0, 2, 1).reshape(batch_size, num_nodes, time_steps, -1)

            # 对每个时间步分别应用 GNN（简化处理）
            risk_curve = []
            for t in range(time_steps):
                x_t = temporal_conv_out[:, :, t, :].reshape(batch_size * num_nodes, -1)

                treatment_week_t = None
                if treatment_week_seq is not None:
                    treatment_week_t = treatment_week_seq[:, t, :, :].reshape(batch_size * num_nodes, -1)

                risk_t, _ = self.seir_time_gnn(x_t, edge_index, treatment_week_t, edge_attr)
                risk_t = risk_t.reshape(batch_size, num_nodes)
                risk_curve.append(risk_t)

            risk_curve = torch.stack(risk_curve, dim=1)
            return risk_curve
