"""多模态特征融合 GNN"""

from ._base import torch, nn, PYG_AVAILABLE, GATv2Conv, _GNN_BASE_CLASS
from ..encoder import EnvironmentEncoder, BehaviorPatternEncoder, TemporalNodeMemory


# =============================================================================
# 多模态特征融合 GNN
# =============================================================================

class MultimodalGNN(_GNN_BASE_CLASS):
    """多模态特征融合 GNN

    支持三种融合模式：
      1. 早期融合：环境特征直接拼入节点特征
      2. 晚期融合：三个独立编码分支 + 注意力加权
      3. 时间聚合：TGN节点记忆 + 时空Transformer

    继承 SEIRInformedGNN 的核心结构，增加多模态和时间模块。

    文献：
      Kipf T et al. (ICLR 2019) GCN
      Velickovic P et al. (ICLR 2018) GAT
      Rossi E et al. (2020) TGN
      Vaswani A et al. (2017) Transformer
    """

    FUSION_MODES = ('early', 'late', 'temporal_tgn', 'temporal_transformer')

    def __init__(self, node_feature_dim=30, edge_feature_dim=6, hidden_dim=64,
                 output_dim=1, n_nodes=20, fusion_mode='late',
                 num_gnn_layers=3, heads=4, dropout=0.3,
                 env_feature_dim=4, behavior_feature_dim=6,
                 memory_dim=64, time_window=7, **kwargs):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_nodes = n_nodes
        self.fusion_mode = fusion_mode
        self.time_window = time_window

        self._init_common_layers(edge_feature_dim, hidden_dim, num_gnn_layers,
                                  heads, dropout)
        self._init_fusion_module(fusion_mode, hidden_dim, output_dim, dropout,
                                  env_feature_dim, behavior_feature_dim,
                                  n_nodes, memory_dim)

    def _init_common_layers(self, edge_feature_dim, hidden_dim,
                             num_gnn_layers, heads, dropout):
        """初始化通用层：节点编码器、边编码器、GNN 层"""
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gnn_layers = nn.ModuleList()
        if PYG_AVAILABLE:
            for _ in range(num_gnn_layers):
                self.gnn_layers.append(
                    GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                              edge_dim=hidden_dim, dropout=dropout))

    def _init_fusion_module(self, fusion_mode, hidden_dim, output_dim, dropout,
                             env_feature_dim, behavior_feature_dim,
                             n_nodes, memory_dim):
        """根据融合模式初始化对应的融合模块和风险预测器"""
        if fusion_mode == 'early':
            self.risk_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        elif fusion_mode == 'late':
            self.env_encoder = EnvironmentEncoder(
                env_feature_dim=env_feature_dim, hidden_dim=hidden_dim,
                output_dim=hidden_dim)
            self.behavior_encoder = BehaviorPatternEncoder(
                behavior_feature_dim=behavior_feature_dim,
                hidden_dim=hidden_dim, output_dim=hidden_dim)
            self.fusion_attention = nn.MultiheadAttention(
                hidden_dim, num_heads=4, batch_first=True, dropout=dropout)
            self.fusion_layer_norm = nn.LayerNorm(hidden_dim)
            self.risk_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        elif fusion_mode in ('temporal_tgn', 'temporal_transformer'):
            # input_dim=hidden_dim：node_encoder 输出 [n_nodes, hidden_dim]，
            # 记忆维度为 memory_dim。显式传入 input_dim 使二者解耦，
            # 避免当 hidden_dim != memory_dim 时 GRUCell 输入维度不匹配。
            self.node_memory = TemporalNodeMemory(n_nodes, memory_dim, input_dim=hidden_dim)
            self.env_encoder = EnvironmentEncoder(
                env_feature_dim=env_feature_dim, hidden_dim=hidden_dim,
                output_dim=hidden_dim)
            if fusion_mode == 'temporal_transformer':
                self.temporal_transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
                        dropout=dropout, batch_first=True),
                    num_layers=3)
            self.temporal_proj = nn.Linear(hidden_dim * 3, hidden_dim)
            self.risk_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )

    def forward(self, snapshot_sequence, env_sequence=None,
                behavior_sequence=None, return_intermediate=False):
        """多模态GNN前向传播

        参数：
            snapshot_sequence: list[dict] 图快照序列
            env_sequence: [B, T, 4] 环境特征序列
            behavior_sequence: [B, T, 6] 行为特征序列
            return_intermediate: 是否返回中间嵌入

        返回：
            risk: [B, 1] 感染风险预测
            (intermediates): 可选中间嵌入
        """
        if self.fusion_mode == 'early':
            return self._forward_early(snapshot_sequence, return_intermediate)
        elif self.fusion_mode == 'late':
            return self._forward_late(snapshot_sequence, env_sequence,
                                       behavior_sequence, return_intermediate)
        elif self.fusion_mode in ('temporal_tgn', 'temporal_transformer'):
            return self._forward_temporal(snapshot_sequence, env_sequence,
                                           return_intermediate)
        else:
            return self._forward_early(snapshot_sequence, return_intermediate)

    def _forward_early(self, snapshot_sequence, return_intermediate=False):
        snap = snapshot_sequence[-1] if isinstance(snapshot_sequence, list) \
               else snapshot_sequence

        x = snap['x'] if isinstance(snap, dict) else snap.x
        edge_index = snap['edge_index'] if isinstance(snap, dict) else snap.edge_index
        edge_attr = snap.get('edge_attr', None) if isinstance(snap, dict) else \
                    getattr(snap, 'edge_attr', None)

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=next(self.node_encoder.parameters()).device)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=next(self.node_encoder.parameters()).device)
        if edge_attr is not None and not isinstance(edge_attr, torch.Tensor):
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=next(self.node_encoder.parameters()).device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr) if edge_attr is not None else None

        if len(self.gnn_layers) == 0:
            import warnings
            warnings.warn(
                "MultimodalGNN.gnn_layers 为空（PyG 不可用或未初始化），"
                "模型退化为纯 MLP，图结构信息未被利用。"
                "请安装 torch-geometric 以启用 GNN 功能。",
                RuntimeWarning, stacklevel=2)
        for layer in self.gnn_layers:
            h_new = layer(h, edge_index, edge_attr=e)
            h = h + h_new

        out = h.mean(dim=0, keepdim=True)
        risk = torch.sigmoid(self.risk_predictor(out))

        if return_intermediate:
            return risk, {'gnn_embedding': h, 'pooled': out}
        return risk

    def _forward_late(self, snapshot_sequence, env_sequence=None,
                       behavior_sequence=None, return_intermediate=False):
        snap = snapshot_sequence[-1] if isinstance(snapshot_sequence, list) \
               else snapshot_sequence

        x = snap['x'] if isinstance(snap, dict) else snap.x
        edge_index = snap['edge_index'] if isinstance(snap, dict) else snap.edge_index
        edge_attr = snap.get('edge_attr', None) if isinstance(snap, dict) else \
                    getattr(snap, 'edge_attr', None)

        if not isinstance(x, torch.Tensor):
            device = next(self.parameters()).device
            x = torch.tensor(x, dtype=torch.float32, device=device)
        if not isinstance(edge_index, torch.Tensor):
            device = next(self.parameters()).device
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
        if edge_attr is not None and not isinstance(edge_attr, torch.Tensor):
            device = next(self.parameters()).device
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr) if edge_attr is not None else None

        if len(self.gnn_layers) == 0:
            import warnings
            warnings.warn(
                "MultimodalGNN.gnn_layers 为空（PyG 不可用或未初始化），"
                "模型退化为纯 MLP，图结构信息未被利用。"
                "请安装 torch-geometric 以启用 GNN 功能。",
                RuntimeWarning, stacklevel=2)
        for layer in self.gnn_layers:
            h_new = layer(h, edge_index, edge_attr=e)
            h = h + h_new

        social_emb = h.mean(dim=0, keepdim=True)

        if env_sequence is not None:
            if not isinstance(env_sequence, torch.Tensor):
                env_sequence = torch.tensor(env_sequence, dtype=torch.float32,
                                            device=social_emb.device)
            if env_sequence.dim() == 2:
                env_sequence = env_sequence.unsqueeze(0)
            env_emb = self.env_encoder(env_sequence)
        else:
            env_emb = torch.zeros(1, self.hidden_dim, device=social_emb.device)

        if behavior_sequence is not None:
            if not isinstance(behavior_sequence, torch.Tensor):
                behavior_sequence = torch.tensor(behavior_sequence, dtype=torch.float32,
                                                 device=social_emb.device)
            if behavior_sequence.dim() == 2:
                behavior_sequence = behavior_sequence.unsqueeze(0)
            behav_emb = self.behavior_encoder(behavior_sequence)
        else:
            behav_emb = torch.zeros(1, self.hidden_dim, device=social_emb.device)

        # Fuse via multi-head attention: social_emb as query, [env_emb, behav_emb] as key/value
        kv = torch.stack([env_emb, behav_emb], dim=1)  # [1, 2, hidden_dim]
        attn_out, _ = self.fusion_attention(social_emb.unsqueeze(1), kv, kv)
        fused = self.fusion_layer_norm(social_emb + attn_out.squeeze(1))
        risk = torch.sigmoid(self.risk_predictor(torch.cat([fused, env_emb, behav_emb], dim=-1)))

        if return_intermediate:
            return risk, {
                'social_embedding': social_emb,
                'environment_embedding': env_emb,
                'behavior_embedding': behav_emb,
            }
        return risk

    def _forward_temporal(self, snapshot_sequence, env_sequence=None,
                           return_intermediate=False):
        if isinstance(snapshot_sequence, list):
            snap = snapshot_sequence[-1]
        else:
            snap = snapshot_sequence

        x = snap['x'] if isinstance(snap, dict) else snap.x
        edge_index = snap['edge_index'] if isinstance(snap, dict) else snap.edge_index
        edge_attr = snap.get('edge_attr', None) if isinstance(snap, dict) else \
                    getattr(snap, 'edge_attr', None)

        if not isinstance(x, torch.Tensor):
            device = next(self.parameters()).device
            x = torch.tensor(x, dtype=torch.float32, device=device)
        if not isinstance(edge_index, torch.Tensor):
            device = next(self.parameters()).device
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
        if edge_attr is not None and not isinstance(edge_attr, torch.Tensor):
            device = next(self.parameters()).device
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr) if edge_attr is not None else None

        if len(self.gnn_layers) == 0:
            import warnings
            warnings.warn(
                "MultimodalGNN.gnn_layers 为空（PyG 不可用或未初始化），"
                "模型退化为纯 MLP，图结构信息未被利用。"
                "请安装 torch-geometric 以启用 GNN 功能。",
                RuntimeWarning, stacklevel=2)
        for layer in self.gnn_layers:
            h_new = layer(h, edge_index, edge_attr=e)
            h = h + h_new

        memory = self.node_memory(h)

        if env_sequence is not None:
            if not isinstance(env_sequence, torch.Tensor):
                env_sequence = torch.tensor(env_sequence, dtype=torch.float32,
                                            device=x.device)
            if env_sequence.dim() == 2:
                env_sequence = env_sequence.unsqueeze(0)
            # env_encoder 已返回 [B, hidden_dim]（B=1），无需额外 unsqueeze
            # 之前此处多余的 .unsqueeze(0) 会产生 [1,1,hidden_dim]，
            # 与 else 分支的 [1,hidden_dim] 维度不一致，导致下方 torch.cat 崩溃
            env_emb = self.env_encoder(env_sequence)
        else:
            env_emb = torch.zeros(1, self.hidden_dim, device=x.device)

        if self.fusion_mode == 'temporal_transformer' and isinstance(
            snapshot_sequence, list) and len(snapshot_sequence) > 1:
            temporal_embs = []
            for s in snapshot_sequence[-min(len(snapshot_sequence), self.time_window):]:
                sx = s['x'] if isinstance(s, dict) else s.x
                if not isinstance(sx, torch.Tensor):
                    sx = torch.tensor(sx, dtype=torch.float32, device=x.device)
                else:
                    sx = sx.to(x.device)
                if sx.dim() == 1:
                    sx = sx.unsqueeze(0)
                sh = self.node_encoder(sx)
                temporal_embs.append(sh.mean(dim=0))
            temporal_stack = torch.stack(temporal_embs, dim=0).unsqueeze(0)
            temporal_emb = self.temporal_transformer(temporal_stack).mean(dim=1)
        else:
            temporal_emb = memory.mean(dim=0, keepdim=True)

        combined = torch.cat([memory.mean(dim=0, keepdim=True), env_emb, temporal_emb],
                              dim=-1)
        fused = self.temporal_proj(combined)
        risk = torch.sigmoid(self.risk_predictor(fused))

        if return_intermediate:
            return risk, {'memory': memory, 'env_embedding': env_emb}
        return risk

    def predict_risk_curve(self, snapshot_sequence, env_sequence=None,
                            behavior_sequence=None):
        """预测风险变化曲线

        对序列中每个时间步预测风险，输出风险轨迹。

        返回：
            list[float]: 各时间步的风险值
        """
        self.eval()
        risks = []
        with torch.no_grad():
            for t in range(len(snapshot_sequence)):
                snap_seq = snapshot_sequence[:t+1]
                env_seq = env_sequence[:t+1] if env_sequence is not None else None
                behav_seq = behavior_sequence[:t+1] if behavior_sequence is not None else None
                risk = self.forward(snap_seq, env_seq, behav_seq)
                risks.append(float(risk.item()))
        return risks
