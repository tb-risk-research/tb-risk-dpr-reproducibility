#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图采样（sampling）— k-hop 邻域子图抽取与节点级 mini-batch 采样。

大规模图网络（如 10 万级节点的接触网络）无法整体加载进显存，因此：
1. 训练时按种子节点抽取 k-hop 邻域子图作为 mini-batch（节点级采样）；
2. 推理时对目标患者仅抽取其 k-hop 邻域子图进行前向，而非在全图上运行。

本模块优先使用 PyG 的 ``k_hop_subgraph``（含边特征保留），并提供不开 PyG
就可用的手动 BFS 回退实现，确保渐进降级。

文献：
- Hamilton, Ying, Leskovec. Inductive Representation Learning on Large Graphs
  (GraphSAGE). NeurIPS 2017. — 邻居采样/邻域聚合。
- Zeng et al. GraphSAINT. ICLR 2020. — 图采样训练。
- Chen et al. FastGCN. ICLR 2018. — 节点采样。
"""
import logging
from typing import List, Optional, Tuple

import numpy as np

from ._common import (
    LOGGER, PYTORCH_AVAILABLE, PYG_AVAILABLE,
    torch, Data, k_hop_subgraph,
)

__all__ = [
    "sample_k_hop_subgraph",
    "manual_k_hop_subgraph",
    "build_neighbor_sampler",
    "NeighborBatch",
]


def sample_k_hop_subgraph(data, node_idx, num_hops=2,
                          relabel_nodes=True):
    """抽取以 ``node_idx`` 为中心的 k-hop 邻域子图（重索引）。

    返回 ``(sub_data, mapping)``：
    - ``sub_data``：PyG ``Data``，含 ``x/edge_index`` 及（若存在）``edge_attr``、
      ``y``，节点已重索引为 0..n-1。
    - ``mapping``：原始节点索引 -> 子图内索引（int，用于定位目标节点）。

    采用 PyG ``k_hop_subgraph``；不可用时回退到手动 BFS 实现。
    """
    if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
        return None, None
    if not isinstance(node_idx, (list, tuple)):
        node_idx = [node_idx]
    node_idx = [int(i) for i in node_idx]

    if k_hop_subgraph is not None:
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, num_hops, data.edge_index,
            relabel_nodes=relabel_nodes, num_nodes=data.num_nodes)
        sub = Data(x=data.x[subset], edge_index=edge_index)
        if data.edge_attr is not None:
            sub.edge_attr = data.edge_attr[edge_mask]
        if data.y is not None:
            sub.y = data.y[subset]
        if hasattr(data, 'mask') and data.mask is not None:
            sub.mask = data.mask[subset]
        # 目标节点在子图内的索引（mapping 与 node_idx 顺序一致）
        target_mapping = int(mapping[0]) if len(mapping) else None
        return sub, target_mapping

    # 回退：手动 BFS
    return manual_k_hop_subgraph(data, node_idx, num_hops, relabel_nodes)


def manual_k_hop_subgraph(data, node_idx, num_hops=2,
                          relabel_nodes=True):
    """手动 BFS k-hop 子图抽取（无 PyG 时的回退实现）。

    返回 ``(sub_data, target_mapping)``。逻辑与 ``k_hop_subgraph`` 对齐：
    取 k-hop 邻域顶点集，再在该集合上取诱导子图（保留内部全部边与边特征）。
    """
    if not PYTORCH_AVAILABLE:
        return None, None
    node_idx = [int(i) for i in (node_idx if isinstance(node_idx, (list, tuple)) else [node_idx])]
    edge_index = data.edge_index
    n = data.num_nodes

    # 邻接表（同时记录边索引）
    adj = [[] for _ in range(n)]
    for e in range(edge_index.shape[1]):
        u = int(edge_index[0, e]); v = int(edge_index[1, e])
        adj[u].append((v, e))

    visited = set(node_idx)
    frontier = list(node_idx)
    for _ in range(num_hops):
        nxt = []
        for u in frontier:
            for v, _e in adj[u]:
                if v not in visited:
                    visited.add(v)
                    nxt.append(v)
        frontier = nxt

    node_list = sorted(visited)
    old2new = {old: i for i, old in enumerate(node_list)}
    sub_edges, sub_edge_idx = [], []
    for u in visited:
        for v, e in adj[u]:
            if v in visited:
                sub_edges.append([old2new[u], old2new[v]])
                sub_edge_idx.append(e)

    sub = Data(x=data.x[torch.tensor(node_list, dtype=torch.long)])
    sub.edge_index = torch.tensor(sub_edges, dtype=torch.long).t().contiguous()
    if data.edge_attr is not None:
        sub.edge_attr = data.edge_attr[torch.tensor(sub_edge_idx, dtype=torch.long)]
    if data.y is not None:
        sub.y = data.y[torch.tensor(node_list, dtype=torch.long)]
    if hasattr(data, 'mask') and data.mask is not None:
        sub.mask = data.mask[torch.tensor(node_list, dtype=torch.long)]

    target_mapping = old2new.get(node_idx[0], None) if relabel_nodes else node_idx[0]
    return sub, target_mapping


class NeighborBatch:
    """节点级 mini-batch：一个抽样子图及其映射回原图的 n_id。

    字段与 PyG ``NeighborLoader`` 产出的 batch 形态对齐，便于统一训练循环：
      - x / edge_index / edge_attr / y：子图局部数据
      - n_id：子图节点 -> 原图节点索引（用于标签/损失对齐）
    """
    __slots__ = ("x", "edge_index", "edge_attr", "y", "n_id")

    def __init__(self, x, edge_index, edge_attr, y, n_id):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.n_id = n_id

    @property
    def num_nodes(self):
        return int(self.x.shape[0]) if self.x is not None else 0


def build_neighbor_sampler(data, num_neighbors: Optional[List[int]] = None,
                           batch_size: int = 256, seed: Optional[int] = 42):
    """构造节点级邻域采样器（生成器）。

    每次产生一个 ``NeighborBatch``（随机种子节点 + k-hop 子图，k = len(num_neighbors)）。
    若 PyG 的邻域子图可用则复用 ``sample_k_hop_subgraph``，否则手动 BFS——
    无需把整图加载进显存，仅保留子图张量。

    参数：
        data: 大图 ``Data``（x, edge_index, 可选 edge_attr/y）
        num_neighbors: 每层采样邻居数（长度即跳数 k），默认 [10, 5]
        batch_size: 每个 batch 的种子节点数
        seed: 随机种子（None 使用全局状态）

    返回：
        generator: 产出 ``NeighborBatch`` 的惰性生成器
    """
    if not PYTORCH_AVAILABLE:
        return
    if num_neighbors is None:
        num_neighbors = [10, 5]
    num_hops = len(num_neighbors)
    rng = np.random.RandomState(seed)
    total = data.num_nodes

    def _generator():
        while True:
            seeds = rng.choice(total, size=min(batch_size, total), replace=False)
            for s in seeds:
                sub, _m = sample_k_hop_subgraph(data, int(s), num_hops)
                if sub is None:
                    continue
                n_id = torch.tensor([int(s)], dtype=torch.long)
                yield NeighborBatch(
                    x=sub.x, edge_index=sub.edge_index,
                    edge_attr=getattr(sub, 'edge_attr', None),
                    y=getattr(sub, 'y', None), n_id=n_id,
                )
    return _generator()