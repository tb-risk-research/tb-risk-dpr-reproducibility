"""GNN 共享基础设施：导入守卫与基类定义

将原 ml/gnn.py 顶部的模块级导入守卫与 _GNN_BASE_CLASS 兼容层集中于此，
供子包内各模块复用，避免重复定义。
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    nn = None
    F = None
    torch = None

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data, HeteroData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    GATConv = None
    GCNConv = None
    SAGEConv = None
    global_mean_pool = None
    GATv2Conv = None
    Data = None
    HeteroData = None


# =============================================================================
# _GNN_BASE_CLASS 兼容层
# =============================================================================

try:
    _GNN_BASE_CLASS = nn.Module
except (NameError, AttributeError):
    class _GNNPlaceholder:
        def __init__(self, *args, **kwargs): pass
        def train(self, mode=True): return self
        def eval(self): return self
        def parameters(self): return iter([])
        def forward(self, *args, **kwargs): return None
    _GNN_BASE_CLASS = _GNNPlaceholder
