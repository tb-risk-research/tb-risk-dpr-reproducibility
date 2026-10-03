#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN 可解释性分析 — 节点/边重要性（GNNExplainer + 梯度边显著性）。

本模块回答"GNN 判定该接触者风险时，哪些图节点/边影响最大"。

实现说明：
- 节点重要性：使用 torch_geometric.explain.Explainer + GNNExplainer 算法，
  对目标接触者节点学习一个节点掩码（node_mask_type='object'），
  掩码值越大表示该节点对判定越关键。
- 边重要性：当前 GNN 使用 GATv2Conv（边特征参与注意力，edge_attr 通路），
  GNNExplainer 的边掩码机制依赖 message-passing 层的 edge_weight，与 GAT 的
  edge_attr 通路不兼容（edge_mask 梯度为 None）。因此边重要性改用梯度显著性
  （target 风险对每条边特征梯度的 L2 范数），作为 GNNExplainer 节点掩码的
  互补信号。

文献支撑：Ying et al. (2019) "GNNExplainer: Generating Explanations for
Graph Neural Networks", NeurIPS。
"""

import logging

from ._common import LOGGER, PYTORCH_AVAILABLE, PYG_AVAILABLE


def _build_node_labels(homo_data, tb_assessment):
    """为同构图每个节点构建可读标签（与 _convert_hetero_to_homo 的节点顺序一致）。

    节点顺序：患者(0..num_patient) → 家庭 → 社会 → 社区。
    """
    labels = []
    num_patient = int(getattr(homo_data, 'num_patient', 0))
    num_family = int(getattr(homo_data, 'num_family', 0))
    num_social = int(getattr(homo_data, 'num_social', 0))
    num_community = int(getattr(homo_data, 'num_community', 0))
    num_region = int(getattr(homo_data, 'num_region', 0))

    family_entries = getattr(tb_assessment, 'family_entries', []) or []
    social_entries = getattr(tb_assessment, 'social_entries', []) or []

    for i in range(num_patient):
        labels.append('患者(传染源)')
    for i in range(num_family):
        entry = family_entries[i] if i < len(family_entries) else {}
        labels.append(entry.get('name') or f'家庭成员{i + 1}')
    for i in range(num_social):
        entry = social_entries[i] if i < len(social_entries) else {}
        labels.append(entry.get('name') or f'社会接触者{i + 1}')
    for i in range(num_community):
        labels.append('社区')
    for i in range(num_region):
        labels.append('区域')
    return labels


def _compute_edge_importance(model, homo_data, target_idx):
    """基于梯度的边重要性：目标节点风险对每条边特征的梯度 L2 范数。

    返回：
        np.ndarray | None: 形状 [E] 的边重要性得分；无边特征或梯度不可用时返回 None。
    """
    if not (hasattr(homo_data, 'edge_attr') and homo_data.edge_attr is not None):
        return None

    import torch

    try:
        model.eval()
        edge_attr = homo_data.edge_attr.detach().clone().requires_grad_(True)
        risk_scores, _ = model(
            homo_data.x, homo_data.edge_index, edge_attr=edge_attr)
        target_score = risk_scores[target_idx, 0]
        model.zero_grad()
        target_score.backward()
        grad = edge_attr.grad
        if grad is None:
            return None
        importance = torch.norm(grad, dim=1).detach().cpu().numpy().flatten()
        return importance
    except Exception as e:
        LOGGER.warning("GNN边重要性计算失败: %s", e, exc_info=True)
        return None


def _normalize(scores):
    """将一维得分归一化到 [0, 1]（相对重要性）。"""
    import numpy as np
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def compute_gnn_explanation(predictor, tb_assessment, contact_data, contact_type='family'):
    """计算目标接触者的 GNN 图归因（节点重要性 + 边重要性）。

    参数：
        predictor: MLRiskPredictor（需已加载并训练 GNN）
        tb_assessment: 结核病风险评估实例（图构建上下文）
        contact_data (dict): 目标接触者数据
        contact_type (str): 'family' | 'social'

    返回：
        dict | None::

            {
                'node_idx': int,                # 目标接触者节点索引
                'node_importance': [            # 按重要性降序，取 Top-N
                    {'node_idx': int, 'label': str, 'importance': float},
                    ...
                ],
                'edge_importance': [            # 按重要性降序，取 Top-N
                    {'source': int, 'target': int, 'source_label': str,
                     'target_label': str, 'importance': float},
                    ...
                ],
                'model_name': str,
            }

        GNN 未训练 / PyG 不可用 / 计算失败时返回 None。
    """
    if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
        return None
    if not predictor.gnn_is_trained or predictor.gnn_model is None:
        return None

    try:
        import torch
        from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

        # 步骤1: 构建同构图（与 predict_gnn_risk 一致）
        hetero_data = predictor.gnn_network_builder.build_from_assessment(tb_assessment)
        homo_data = predictor._convert_hetero_to_homo(hetero_data)

        # 步骤2: 定位目标接触者节点
        target_idx = predictor._find_target_node_idx(
            contact_data, homo_data, contact_type, tb_assessment)
        if target_idx is None:
            LOGGER.error("GNN归因失败：未找到目标接触者节点")
            return None

        device = next(predictor.gnn_model.parameters()).device
        homo_data = homo_data.to(device)
        model = predictor.gnn_model.eval()
        edge_attr = homo_data.edge_attr if hasattr(homo_data, 'edge_attr') else None

        # 步骤3: 包装模型，仅返回节点级风险得分（GNNExplainer 需要单张量输出）
        class _RiskScoreWrapper(torch.nn.Module):
            def __init__(self, gnn):
                super().__init__()
                self.gnn = gnn

            def forward(self, x, edge_index, **kwargs):
                scores, _ = self.gnn(x, edge_index, **kwargs)
                return scores

        kwargs = {}
        if edge_attr is not None:
            kwargs['edge_attr'] = edge_attr

        explainer = Explainer(
            model=_RiskScoreWrapper(model),
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type=None,  # GAT edge_attr 通路与边掩码不兼容，边重要性走梯度
            model_config=ModelConfig(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        explanation = explainer(
            homo_data.x, homo_data.edge_index, index=target_idx, **kwargs)

        # 节点重要性（GNNExplainer 学到的节点掩码）
        node_mask = explanation.node_mask.detach().cpu().numpy().flatten() \
            if explanation.node_mask is not None else None

        # 边重要性（梯度显著性）
        edge_importance = _compute_edge_importance(model, homo_data, target_idx)

        # 步骤4: 组装结果
        node_labels = _build_node_labels(homo_data, tb_assessment)

        node_items = []
        if node_mask is not None:
            norm_node = _normalize(node_mask)
            for idx, score in enumerate(norm_node):
                node_items.append({
                    'node_idx': int(idx),
                    'label': node_labels[idx] if idx < len(node_labels) else f'节点{idx}',
                    'importance': float(score),
                })
            node_items.sort(key=lambda x: x['importance'], reverse=True)

        edge_items = []
        if edge_importance is not None:
            norm_edge = _normalize(edge_importance)
            edge_index = homo_data.edge_index.detach().cpu().numpy()
            for e, score in enumerate(norm_edge):
                src, dst = int(edge_index[0, e]), int(edge_index[1, e])
                edge_items.append({
                    'source': src,
                    'target': dst,
                    'source_label': node_labels[src] if src < len(node_labels) else f'节点{src}',
                    'target_label': node_labels[dst] if dst < len(node_labels) else f'节点{dst}',
                    'importance': float(score),
                })
            edge_items.sort(key=lambda x: x['importance'], reverse=True)

        return {
            'node_idx': int(target_idx),
            'node_importance': node_items[:10],
            'edge_importance': edge_items[:10],
            'model_name': 'GNN图神经网络(GNNExplainer)',
        }

    except Exception as e:
        LOGGER.warning("GNN归因分析失败: %s", e, exc_info=True)
        return None