#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN框架共享模块 - 依赖检测与占位符定义。

所有子模块统一从此处引用 numpy / torch / torch_geometric 及
``seir_ode_gating``，避免重复定义和注册状态不一致。
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

try:
    from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv, global_mean_pool
    from torch_geometric.data import HeteroData, Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    GATConv = None
    GATv2Conv = None
    GCNConv = None
    SAGEConv = None
    global_mean_pool = None
    HeteroData = None
    Data = None

# SEIR ODE 门控（从 gnn.py 导入，避免重复定义）
try:
    from ..gnn import seir_ode_gating
    _ODE_GATING_AVAILABLE = True
except ImportError:
    _ODE_GATING_AVAILABLE = False
    seir_ode_gating = None


class _GNNFrameworkPlaceholder:
    """GNN框架占位符类（当PyTorch不可用时）"""
    pass
