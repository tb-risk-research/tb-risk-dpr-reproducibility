#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增量推理（incremental）— 接触网络新增节点时仅局部重算，而非全图重算。

现实场景：接触网持续更新（今日新增一名接触者）。若每次新增都重建整图并全图
前向，代价随图规模线性增长。本模块：
1. 将新节点及其边增量追加到大图上（不重建整图）；
2. 仅对新节点的 k-hop 邻域子图做前向推理，得到其风险概率；
3. 返回子图规模 vs 全图规模的对比，量化"局部重算"的收益。

文献：
- Hamilton et al. GraphSAGE. NeurIPS 2017. — 归纳式推理支持新节点。
- Xu et al. How Powerful are GNNs? ICLR 2019. — k 层 GNN 依赖 k-hop 邻域。
"""
import logging
from typing import List, Optional

from ._common import (
    LOGGER, PYTORCH_AVAILABLE, PYG_AVAILABLE, torch, Data,
)
from .sampling import sample_k_hop_subgraph
from .inference import predict_subgraph

__all__ = ["add_new_node", "incremental_predict_new_node"]


def add_new_node(large_data, new_feature, neighbor_indices: List[int],
                 edge_attr: Optional[List] = None):
    """向大图增量追加一个新节点及其边（返回新 Data，原图不变）。

    参数：
        large_data: 原图 ``Data``
        new_feature: 新节点特征（长度 = 特征维）
        neighbor_indices: 新节点要连接的既有节点索引列表
        edge_attr: 新节点入边特征（形状 (len(neighbors), edge_dim) 或 None）

    返回：
        Data: 追加新节点后的图，原图不被修改。
    """
    if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
        return None
    new_idx = large_data.num_nodes

    # 复制节点特征
    x_new = torch.cat([large_data.x, torch.as_tensor(new_feature, dtype=large_data.x.dtype).reshape(1, -1)], dim=0)

    # 双向边（新节点 <-> 每个邻居）
    n_nb = len(neighbor_indices)
    src = [new_idx] * n_nb + list(neighbor_indices)
    dst = list(neighbor_indices) + [new_idx] * n_nb
    edge_add = torch.tensor([src, dst], dtype=torch.long)

    edge_index = torch.cat([large_data.edge_index, edge_add], dim=1)

    # 边特征：若无新边特征则补零；双向复制
    if large_data.edge_attr is not None:
        if edge_attr is None:
            edge_dim = large_data.edge_attr.shape[1]
            feat = torch.zeros((n_nb, edge_dim), dtype=large_data.edge_attr.dtype)
        else:
            feat = torch.as_tensor(edge_attr, dtype=large_data.edge_attr.dtype)
        edge_attr_all = torch.cat([feat, feat], dim=0)
        edge_attr_new = torch.cat([large_data.edge_attr, edge_attr_all], dim=0)
    else:
        edge_attr_new = None

    new_data = Data(x=x_new, edge_index=edge_index)
    if edge_attr_new is not None:
        new_data.edge_attr = edge_attr_new
    if large_data.y is not None:
        # 新节点标签未知，追加 NaN
        y_new = torch.cat([large_data.y, torch.full((1,), float('nan'), dtype=large_data.y.dtype)])
        new_data.y = y_new
    return new_data


def incremental_predict_new_node(model, large_data, new_feature,
                                 neighbor_indices: List[int], num_hops=2,
                                 edge_attr: Optional[List] = None, device=None,
                                 forward_fn=None):
    """对新增接触者做增量推理：追加节点 + k-hop 子图局部前向。

    返回：
        dict: {risk_probability, risk_class, new_node_idx, subgraph_size,
               full_graph_size, reduction_ratio, note}
        依赖缺失或失败返回 None。
    """
    if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
        return None
    new_data = add_new_node(large_data, new_feature, neighbor_indices, edge_attr)
    if new_data is None:
        return None
    new_idx = new_data.num_nodes - 1

    sub, mapping = sample_k_hop_subgraph(new_data, new_idx, num_hops)
    if sub is None or mapping is None:
        return None
    res = predict_subgraph(model, sub, mapping, device=device, forward_fn=forward_fn)
    if res is None:
        return None

    full_size = int(new_data.num_nodes)
    sub_size = int(res.get("subgraph_size", full_size))
    reduction = (1.0 - (sub_size / max(full_size, 1))) * 100.0
    res["new_node_idx"] = new_idx
    res["full_graph_size"] = full_size
    res["reduction_ratio"] = reduction
    res["note"] = (
        f"新增节点 #{new_idx} 推理仅计算 {sub_size} 节点（全图 {full_size}），"
        f"局部子图占用减少 {reduction:.1f}%。"
    )
    return res