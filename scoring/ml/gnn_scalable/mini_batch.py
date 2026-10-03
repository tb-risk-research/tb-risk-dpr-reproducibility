#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mini-batch GNN 训练 — 面向十万级节点接触网络的节点级采样训练。

问题：整图全量前向在显存/内存上不可行（10 万节点 × 全邻域 GAT 传播）。
本模块将训练改为节点级 mini-batch：
- 优先使用 PyG ``NeighborLoader``（分层邻居采样，GraphSAGE 范式）；
- 不可用时回退到 ``sampling.build_neighbor_sampler`` 的 k-hop 子图采样。

两者都只把"抽样子图"送入模型，整图常驻 CPU 内存/磁盘，不整图进显存。

文献：
- Hamilton, Ying, Leskovec. Inductive Representation Learning on Large Graphs
  (GraphSAGE). NeurIPS 2017.
- Zeng et al. GraphSAINT. ICLR 2020.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ._common import (
    LOGGER, PYTORCH_AVAILABLE, PYG_AVAILABLE, NEIGHBOR_LOADER_AVAILABLE,
    torch, Data, NeighborLoader,
)
from .sampling import build_neighbor_sampler

__all__ = ["MinibatchConfig", "train_gnn_minibatch", "NEIGHBOR_LOADER_AVAILABLE"]


@dataclass
class MinibatchConfig:
    """mini-batch GNN 训练配置。"""
    num_neighbors: List[int] = field(default_factory=lambda: [10, 5])
    batch_size: int = 256
    num_epochs: int = 20
    learning_rate: float = 0.001
    shuffle: bool = True
    seed: Optional[int] = 42


def _default_loss(pred, target):
    import torch.nn.functional as F
    return F.binary_cross_entropy(pred.squeeze(), target.float())


def train_gnn_minibatch(model, large_data, config: Optional[MinibatchConfig] = None,
                        optimizer=None, loss_fn: Optional[Callable] = None,
                        forward_fn: Optional[Callable] = None,
                        device=None, progress_callback: Optional[Callable] = None):
    """在大型图上以节点级 mini-batch 方式训练 GNN。

    参数：
        model: PyTorch GNN 模块（forward 返回 ``(risk_scores, graph_emb)``）
        large_data: 大图 ``Data``（x, edge_index, 可选 edge_attr/y）
        config: ``MinibatchConfig``；None 使用默认
        optimizer: 自定义优化器；None 自动创建 Adam
        loss_fn: ``loss(pred, target)``；None 使用二元交叉熵
        forward_fn: ``fn(model, x, edge_index, edge_attr) -> risk_scores``；
                    None 使用默认 ``model(x, edge_index, edge_attr=...)``
        device: 计算设备；None 自动选择 cuda/cpu
        progress_callback: ``(epoch, total_epochs)``

    返回：
        dict: {success, backend, losses, num_epochs, device}
        依赖缺失时返回 {success: False, backend: 'unavailable'}。
    """
    if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
        return {"success": False, "backend": "unavailable"}
    if config is None:
        config = MinibatchConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if loss_fn is None:
        loss_fn = _default_loss

    model = model.to(device)

    # 选择后端：优先 PyG NeighborLoader，不可实例化时回退手动 k-hop 采样
    backend = "neighbor_loader" if NEIGHBOR_LOADER_AVAILABLE else "manual_sampler"
    # 手动采样器每次产生单个种子节点的子图 batch；每批种子数 = budget，
    # 故每轮的 batch 数按"覆盖整图节点"折算，避免无限生成器导致死循环。
    batches_per_epoch = max(1, math.ceil(large_data.num_nodes / config.batch_size))

    def _default_forward(model, x, edge_index, edge_attr):
        if edge_attr is not None:
            risk, _ = model(x, edge_index, edge_attr=edge_attr)
        else:
            risk, _ = model(x, edge_index)
        return risk

    if forward_fn is None:
        forward_fn = _default_forward

    losses: List[float] = []
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_nodes = 0

        if NEIGHBOR_LOADER_AVAILABLE:
            # NeighborLoader 依赖 pyg-lib/torch-sparse，缺失时实例化会抛错；
            # 此处用 try 捕获并在当轮回退到手动采样器。
            try:
                loader = NeighborLoader(
                    large_data,
                    num_neighbors=config.num_neighbors,
                    batch_size=config.batch_size,
                    shuffle=config.shuffle,
                )
                for batch in loader:
                    batch = batch.to(device)
                    edge_attr = batch.edge_attr if (hasattr(batch, 'edge_attr') and batch.edge_attr is not None) else None
                    optimizer.zero_grad()
                    risk = forward_fn(model, batch.x, batch.edge_index, edge_attr)
                    loss = loss_fn(risk, batch.y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.item()) * batch.num_nodes
                    n_nodes += batch.num_nodes
                continue
            except (ImportError, RuntimeError, TypeError) as e:
                LOGGER.debug("NeighborLoader 不可用，回退手动采样器: %s", e)
                backend = "manual_sampler"

        sampler = build_neighbor_sampler(
            large_data, config.num_neighbors, config.batch_size, config.seed)
        for _ in range(batches_per_epoch):
            batch = next(sampler, None)
            if batch is None or batch.y is None:
                continue
            optimizer.zero_grad()
            risk = forward_fn(model, batch.x, batch.edge_index, batch.edge_attr)
            loss = loss_fn(risk, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch.num_nodes
            n_nodes += batch.num_nodes

        losses.append(epoch_loss / max(n_nodes, 1))
        if progress_callback:
            try:
                progress_callback(epoch + 1, config.num_epochs)
            except Exception:
                pass

    return {
        "success": True,
        "backend": backend,
        "losses": losses,
        "num_epochs": config.num_epochs,
        "device": str(device),
    }