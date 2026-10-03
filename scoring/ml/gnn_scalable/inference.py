#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""子图推理（inference）— 对目标患者仅在其 k-hop 邻域子图上做前向。

大规模图下，对单个目标节点的推理无需在全图上运行 GNN。抽取目标节点的
k-hop 邻域子图（局部图），在其上进行消息传递，得到目标节点的风险概率。
子图规模远小于全图，显存/内存占用与延迟都大幅下降。

文献：
- Hamilton et al. GraphSAGE. NeurIPS 2017. — 归纳式/局部邻域推理。
- Chiang et al. Cluster-GCN. KDD 2019. — 分块子图计算。
"""
import logging
from typing import Optional

from ._common import (
    LOGGER, PYTORCH_AVAILABLE, PYG_AVAILABLE, torch,
)
from .sampling import sample_k_hop_subgraph

__all__ = ["predict_target_k_hop", "predict_subgraph"]


def predict_subgraph(model, sub_data, target_mapping=None, device=None,
                     forward_fn=None):
    """对已抽取的子图 ``sub_data`` 执行前向，返回各节点风险概率。

    参数：
        model: GNN 模块（forward 返回 (risk_scores, graph_emb)）
        sub_data: k-hop 子图 ``Data``
        target_mapping: 目标节点在子图内的索引；None 返回所有节点
        device: 计算设备
        forward_fn: 自定义前向；None 使用 ``model(x, edge_index, edge_attr=...)``

    返回：
        dict: {risk_probability, risk_class, node_idx, subgraph_size, n_edges}
        依赖缺失返回 None。
    """
    if not (PYTORCH_AVAILABLE and PYG_AVAILABLE) or sub_data is None:
        return None
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _default_forward(model, x, edge_index, edge_attr):
        if edge_attr is not None:
            risk, _ = model(x, edge_index, edge_attr=edge_attr)
        else:
            risk, _ = model(x, edge_index)
        return risk

    if forward_fn is None:
        forward_fn = _default_forward

    model.eval()
    model = model.to(device)
    sub_data = sub_data.to(device)
    edge_attr = sub_data.edge_attr if (hasattr(sub_data, 'edge_attr') and sub_data.edge_attr is not None) else None
    with torch.no_grad():
        risk = forward_fn(model, sub_data.x, sub_data.edge_index, edge_attr)

    probs = risk.squeeze().detach().cpu().numpy()
    if probs.ndim == 0:
        probs = probs.reshape(1)

    def _clip(v):
        return float(max(0.0, min(100.0, float(v) * 100.0)))

    if target_mapping is None:
        return {
            "risk_probabilities": [float(p) for p in probs],
            "subgraph_size": int(sub_data.num_nodes),
            "n_edges": int(sub_data.edge_index.shape[1]),
        }

    target_mapping = int(target_mapping)
    if target_mapping >= len(probs):
        return None
    p = probs[target_mapping]
    gnn_risk = _clip(p)
    return {
        "risk_probability": gnn_risk,
        "risk_class": 1 if gnn_risk > 50 else 0,
        "node_idx": target_mapping,
        "subgraph_size": int(sub_data.num_nodes),
        "n_edges": int(sub_data.edge_index.shape[1]),
    }


def predict_target_k_hop(model, large_data, target_idx, num_hops=2, device=None,
                         forward_fn=None):
    """对大型图中目标节点 ``target_idx`` 做 k-hop 子图推理（不跑全图）。

    参数：
        model: GNN 模块
        large_data: 大图 ``Data``
        target_idx: 目标节点在原图中的索引
        num_hops: 邻域跳数（消息传递深度）
        device: 计算设备
        forward_fn: 自定义前向

    返回：
        dict: predict_subgraph 的结果（含 risk_probability / subgraph_size）
        依赖缺失或抽取失败返回 None。
    """
    if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
        return None
    sub, mapping = sample_k_hop_subgraph(large_data, target_idx, num_hops)
    if sub is None:
        return None
    return predict_subgraph(model, sub, mapping, device=device, forward_fn=forward_fn)