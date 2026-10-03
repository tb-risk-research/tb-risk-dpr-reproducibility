#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大规模图网络处理子包 — 共享可用性标志与软依赖探测。

同项目惯例：所有可选依赖（torch / torch_geometric）均在做功前探测，
缺失时优雅降级（返回 None / 空结果 / False），不引发 ImportError。
"""
import logging

LOGGER = logging.getLogger("tb_risk.ml.gnn_scalable")

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    PYTORCH_AVAILABLE = False

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import k_hop_subgraph
    from torch_geometric.loader import DataLoader
    PYG_AVAILABLE = True
except ImportError:
    Data = None
    k_hop_subgraph = None
    DataLoader = None
    PYG_AVAILABLE = False

try:
    from torch_geometric.loader import NeighborLoader
    NEIGHBOR_LOADER_AVAILABLE = True
except ImportError:
    NeighborLoader = None
    NEIGHBOR_LOADER_AVAILABLE = False

try:
    from torch.quantization import quantize_dynamic
    DYNAMIC_QUANT_AVAILABLE = True
except ImportError:
    quantize_dynamic = None
    DYNAMIC_QUANT_AVAILABLE = False

__all__ = [
    "LOGGER",
    "PYTORCH_AVAILABLE", "PYG_AVAILABLE",
    "NEIGHBOR_LOADER_AVAILABLE", "DYNAMIC_QUANT_AVAILABLE",
    "torch", "Data", "k_hop_subgraph", "NeighborLoader", "quantize_dynamic",
]