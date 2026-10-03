"""异质时变图神经网络 + SEIR ODE 门控 + 因果约束"""

from ._base import torch, nn, F, PYTORCH_AVAILABLE, _GNN_BASE_CLASS
from ._layers import HeteroGATLayer
from ._seir_ode import seir_ode_gating


# =============================================================================
# 异质时变图神经网络
# =============================================================================

class HeteroTimeVaryingGNN(_GNN_BASE_CLASS):
    """异质时变图神经网络 + SEIR ODE 门控 + 因果约束

    v2.1: 门控升级为 SEIR ODE 积分器，替代纯特征缩放。
    用积分结果（峰值感染率、AUC、疫情规模）调制节点表示。
    v3.0 起：默认改用四阶 RK4 求解器（全局误差 O(dt⁴)），
    精度优于一阶 Euler 的 O(dt)，可减少冗余步数。

    v3.0: 门控输出从 3 通道扩展为 6 通道，对应 9-房室 SEIR 模型；
          节点特征维度从 22 扩展为 30（新增潜伏/疾病状态 one-hot）。
          seir_ode_gating 自动检测通道数并分发到 4D/9D 积分。

    文献：
        Kermack & McKendrick (1927) SEIR 模型
        Chen et al. (2018) Neural ODE, NeurIPS
        Houben et al. (2016) BMC Med — 快/慢分层
        Horton et al. (2023) PNAS — 自清除
        Emery et al. (2023) eLife — 亚临床传染性
    """

    def __init__(self, node_feature_dim=30, edge_feature_dim=6, hidden_dim=64,
                 num_edge_types=5, num_layers=1, num_time_steps=4,
                 causal_graph=None, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.num_time_steps = num_time_steps
        self.causal_graph = causal_graph
        self.causal_mask = None
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim)
        self.hetero_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hetero_layers.append(
                HeteroGATLayer(hidden_dim, hidden_dim, num_edge_types, heads=4, dropout=dropout))
        self._build_causal_mask()
        self.time_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.seir_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 8), nn.Sigmoid())  # 改进7+8: 8 通道门控 (β, ρ_fast, σ_clear, ρ_prog, γ, ω_reg, ρ_react, η_sub)
        self.risk_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.causal_loss_weight = 0.1
        self.causal_adj_matrix = None

    def _build_causal_mask(self):
        if self.causal_graph is None: return
        # v3.0 扩展：在原 14 变量基础上新增 5 个房室结构相关变量
        # 新增：自清除、亚临床传播、外源再感染、内源性再激活、波动
        variables = ['age', 'BCG', 'HIV', 'diabetes', 'past_tb', 'ventilation',
                     'contact_distance', 'symptoms', 'exposure', 'immunosuppression',
                     'screening', 'treatment_delay', 'env_intervention', 'infection',
                     # v3.0 新增（对应 9-房室 SEIR 关键流参数）
                     'self_clearance',          # σ_clear: L → C
                     'subclinical_transmission', # η_sub: I_sub 传染性
                     'exogenous_reinfection',   # β_exo: L_slow → L_fast
                     'endogenous_reactivation', # ρ_react: L_slow → I
                     'undulation']              # ω_reg: M ↔ L_slow / I_sub ↔ M
        n_vars = len(variables)
        var_to_idx = {v: i for i, v in enumerate(variables)}
        # 与模型参数同设备，并注册为 buffer 以随 .to(device) 迁移
        device = next(self.parameters()).device if any(self.parameters()) else torch.device('cpu')
        causal_matrix = torch.zeros(n_vars, n_vars, device=device)
        for src, targets in self.causal_graph.causal_graph.items():
            if src in var_to_idx:
                for tgt in targets:
                    if tgt in var_to_idx:
                        causal_matrix[var_to_idx[src], var_to_idx[tgt]] = 1.0
        for src, targets in self.causal_graph.causal_graph.items():
            if src in var_to_idx:
                for mid in targets:
                    if mid in var_to_idx:
                        for tgt in self.causal_graph.causal_graph.get(mid, []):
                            if tgt in var_to_idx and mid != tgt:
                                causal_matrix[var_to_idx[src], var_to_idx[tgt]] = 1.0
        self.register_buffer('causal_adj_matrix', causal_matrix.float())
        self.register_buffer('causal_mask', (causal_matrix > 0).float())

    def compute_causal_consistency_loss(self, attention_weights):
        if self.causal_adj_matrix is None or attention_weights is None:
            # 使用与模型一致的设备，避免 GPU/CPU 不匹配
            device = attention_weights.device if attention_weights is not None else (
                next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu')
            return torch.tensor(0.0, device=device)
        attn = attention_weights
        adj = self.causal_adj_matrix.to(attn.device)
        min_dim = min(attn.shape[0], adj.shape[0])
        return F.mse_loss(attn[:min_dim, :min_dim], adj[:min_dim, :min_dim])

    def forward(self, node_features_list, edge_indices_list, edge_attrs_list,
                edge_types_list=None, treatment_phase=None, use_ode_gating=True):
        T = len(node_features_list)
        time_embeddings = []
        ode_metrics_list = []  # 收集各时间步的 ODE 指标
        for t in range(T):
            x_t = node_features_list[t]
            e_idx_t = edge_indices_list[t] if t < len(edge_indices_list) else []
            e_attr_t = edge_attrs_list[t] if t < len(edge_attrs_list) else []
            h = self.node_embedding(x_t); h = F.relu(h); h = self.dropout(h)
            for layer in self.hetero_layers:
                h_new = layer(h, e_idx_t, e_attr_t)
                h = h + h_new; h = F.relu(h); h = self.dropout(h)
            if treatment_phase is not None:
                n_nodes = h.shape[0]
                if treatment_phase.shape[0] > n_nodes:
                    tp_t = treatment_phase[:n_nodes]
                elif treatment_phase.shape[0] < n_nodes:
                    padding = treatment_phase[-1:].expand(n_nodes - treatment_phase.shape[0], -1)
                    tp_t = torch.cat([treatment_phase, padding], dim=0)
                else:
                    tp_t = treatment_phase

                if use_ode_gating and PYTORCH_AVAILABLE:
                    # v2.1: SEIR ODE 门控 — 使用 Euler 积分结果调制节点表示
                    h, ode_metrics = seir_ode_gating(h, tp_t, self.seir_gate)
                    ode_metrics_list.append(ode_metrics)
                else:
                    # 回退：纯特征缩放门控（改进7+8: 8 通道）
                    gate_input = torch.cat([h, tp_t], dim=-1)
                    seir_w = self.seir_gate(gate_input)  # [B, 8]
                    phase = tp_t.squeeze(-1)
                    # 8 通道：β, ρ_fast, σ_clear, ρ_prog, γ, ω_reg, ρ_react, η_sub
                    # 治疗阶段对各通道的不同调制权重
                    w_trans = torch.sigmoid(-(phase - 1.5) * 2.0)      # 传播：早期高
                    w_fast = torch.sigmoid(-(phase - 2.0) * 2.0) * \
                             torch.sigmoid((phase - 0.5) * 3.0)        # 快进展：中期
                    w_clear = torch.sigmoid((phase - 1.0) * 1.5)       # 自清除：后期增强
                    w_prog = torch.sigmoid((phase - 1.5) * 2.0)        # 进展：后期
                    w_recov = torch.sigmoid((phase - 1.5) * 2.0)       # 恢复：后期
                    w_reg = torch.ones_like(phase) * 0.5               # 波动：恒定
                    w_react = torch.sigmoid(-(phase - 3.0) * 1.5)      # 慢再激活：早期高
                    w_eta = torch.ones_like(phase) * 0.5               # 亚临床传染性：恒定
                    w_sum = (w_trans + w_fast + w_clear + w_prog + w_recov
                             + w_reg + w_react + w_eta + 1e-8)
                    ch = seir_w.unbind(dim=-1)
                    w_list = [w_trans, w_fast, w_clear, w_prog, w_recov,
                              w_reg, w_react, w_eta]
                    gated_modulation = sum(
                        (w / w_sum).unsqueeze(-1) * c.unsqueeze(-1)
                        for w, c in zip(w_list, ch)
                    )
                    h = h * gated_modulation
            time_embeddings.append(h.mean(dim=0))
        time_stack = torch.stack(time_embeddings, dim=0)
        if T > 1:
            time_seq = time_stack.unsqueeze(0)
            attn_output, attn_weights = self.time_attention(time_seq, time_seq, time_seq)
            causal_loss = self.compute_causal_consistency_loss(
                attn_weights.squeeze(0) if attn_weights is not None else None)
            final_emb = self.time_proj(attn_output.squeeze(0)).mean(dim=0)
        else:
            causal_loss = torch.tensor(0.0, device=time_stack.device)
            final_emb = time_stack[0]
        risk = self.risk_predictor(final_emb.unsqueeze(0)).squeeze()
        return risk, final_emb, causal_loss, time_embeddings

    def forward_single(self, node_features, edge_indices, edge_attrs,
                       edge_types_list=None, treatment_phase=None):
        ei_list = [edge_indices] if not isinstance(edge_indices, list) else edge_indices
        ea_list = [edge_attrs] if not isinstance(edge_attrs, list) else edge_attrs
        et_list = [edge_types_list] if edge_types_list is not None and not isinstance(edge_types_list, list) else edge_types_list
        return self.forward([node_features], ei_list, ea_list,
                           et_list, treatment_phase)[0]
