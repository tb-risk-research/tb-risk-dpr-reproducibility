"""异质图注意力层"""

from ._base import torch, nn, _GNN_BASE_CLASS


# =============================================================================
# 异质图注意力层
# =============================================================================

class HeteroGATLayer(_GNN_BASE_CLASS):
    """异质图注意力层：不同关系类型独立注意力参数"""

    # 边特征维度常量（接触频率、单次时长、持续周数、通风条件、接触距离、暴露场景）
    EDGE_FEATURE_DIM = 6

    def __init__(self, in_dim, out_dim, num_edge_types=5, heads=4, dropout=0.3):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.num_edge_types = num_edge_types
        self.node_lin = nn.Linear(in_dim, out_dim * heads)
        self.att_src = nn.ParameterList([
            nn.Parameter(torch.randn(heads, out_dim)) for _ in range(num_edge_types)
        ])
        self.att_dst = nn.ParameterList([
            nn.Parameter(torch.randn(heads, out_dim)) for _ in range(num_edge_types)
        ])
        self.edge_transform = nn.ModuleList([
            nn.Linear(self.EDGE_FEATURE_DIM, heads) for _ in range(num_edge_types)
        ])
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def _segment_softmax(self, src, index, num_segments=None):
        """分段softmax实现，避免对torch_geometric的硬依赖"""
        if num_segments is None:
            num_segments = int(index.max().item()) + 1 if index.numel() > 0 else 1
        max_vals = torch.full((num_segments, src.shape[1]), -1e9, device=src.device)
        max_vals = max_vals.scatter_reduce(0, index.unsqueeze(-1).expand_as(src),
                                           src, reduce='amax', include_self=False)
        src_exp = torch.exp(src - max_vals[index])
        sum_vals = torch.zeros(num_segments, src.shape[1], device=src.device)
        sum_vals = sum_vals.scatter_add(0, index.unsqueeze(-1).expand_as(src_exp), src_exp)
        return src_exp / (sum_vals[index] + 1e-10)

    def forward(self, x, edge_indices, edge_attrs):
        try:
            from torch_geometric.utils import softmax as pyg_softmax
            _has_pyg_softmax = True
        except ImportError:
            _has_pyg_softmax = False

        h = self.node_lin(x).view(-1, self.heads, self.out_dim)
        out_h = torch.zeros_like(h)
        for k in range(self.num_edge_types):
            e_idx_k = edge_indices[k] if k < len(edge_indices) else None
            if e_idx_k is None or e_idx_k.shape[1] == 0: continue
            e_attr_k = edge_attrs[k] if edge_attrs and k < len(edge_attrs) \
                                        and edge_attrs[k] is not None else None
            src, dst = e_idx_k[0], e_idx_k[1]
            h_src, h_dst = h[src], h[dst]
            att = (h_src * self.att_src[k].unsqueeze(0)).sum(dim=-1) + \
                  (h_dst * self.att_dst[k].unsqueeze(0)).sum(dim=-1)
            if e_attr_k is not None:
                num_edges = e_idx_k.shape[1]
                n_attr = e_attr_k.shape[0]
                if n_attr >= num_edges:
                    # 边特征行数 >= 边数：截断到匹配
                    e_attr_k_aligned = e_attr_k[:num_edges]
                else:
                    # 边特征行数 < 边数：零填充到匹配
                    # （防御数据不一致：正常构建的图不应触发此分支，
                    #   但避免 e_attr_k[:num_edges] 静默返回较少行后
                    #   在 att + edge_transform(...) 处引发广播错误）
                    padding = torch.zeros(
                        num_edges - n_attr, e_attr_k.shape[1],
                        device=e_attr_k.device, dtype=e_attr_k.dtype)
                    e_attr_k_aligned = torch.cat([e_attr_k, padding], dim=0)
                att = att + self.edge_transform[k](e_attr_k_aligned)
            att = self.leaky_relu(att)
            if _has_pyg_softmax:
                att = pyg_softmax(att, dst)
            else:
                att = self._segment_softmax(att, dst)
            att = self.dropout(att)
            msg = h_src * att.unsqueeze(-1)
            # scatter_add 需要 index 与 msg 形状一致 [num_edges, heads, out_dim]
            # dst 为 [num_edges]，需两次 unsqueeze 得到 [num_edges, 1, 1] 再 expand
            # 原代码仅一次 unsqueeze（[num_edges, 1]），expand 到 3D 时维度不匹配
            out_h = out_h.scatter_add(
                0, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.out_dim), msg)
        return out_h.mean(dim=1)
