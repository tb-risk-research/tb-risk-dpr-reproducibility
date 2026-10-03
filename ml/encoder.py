"""
多模态编码器模块

包含环境编码器、行为模式编码器和时序节点记忆模块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 基础类：torch 已无条件导入，nn.Module 始终可用
_GNN_BASE_CLASS = nn.Module


class EnvironmentEncoder(_GNN_BASE_CLASS):
    """环境暴露分支编码器

    使用 1D CNN 或 MLP 处理时变环境特征序列，
    捕捉环境暴露的时间模式。

    文献：Li Y et al. (2018) Adaptive GCN
    """

    def __init__(self, env_feature_dim=4, hidden_dim=64, output_dim=32,
                 window_size=7, use_cnn=True):
        super().__init__()
        self.env_feature_dim = env_feature_dim
        self.window_size = window_size
        self.use_cnn = use_cnn

        if use_cnn:
            self.conv1 = nn.Conv1d(env_feature_dim, hidden_dim // 2, kernel_size=3,
                                    padding=1)
            self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3,
                                    padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(env_feature_dim * window_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, env_sequence):
        if self.use_cnn:
            x = env_sequence.transpose(1, 2)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
        else:
            x = env_sequence.reshape(env_sequence.shape[0], -1)
            x = self.mlp(x)
        return self.output_layer(x)


class BehaviorPatternEncoder(_GNN_BASE_CLASS):
    """行为模式分支编码器

    使用 LSTM/GRU 捕捉时序行为模式，
    从轮班数据和移动轨迹中学习行为嵌入。

    文献：Hochreiter S et al. (1997) LSTM
    """

    def __init__(self, behavior_feature_dim=6, hidden_dim=64, output_dim=32,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.behavior_feature_dim = behavior_feature_dim

        self.gru = nn.GRU(behavior_feature_dim, hidden_dim,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0,
                          bidirectional=True)

        self.output_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, behavior_sequence):
        _, h_n = self.gru(behavior_sequence)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        combined = torch.cat([h_forward, h_backward], dim=-1)
        return self.output_layer(combined)


class TemporalNodeMemory(_GNN_BASE_CLASS):
    """节点记忆模块 (TGN风格)

    用 GRU 维护每个节点的历史状态，
    每个时间步 GNN 处理后更新记忆。

    文献：Rossi E et al. (2020) Temporal Graph Networks
    """

    def __init__(self, n_nodes, memory_dim=64, input_dim=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.memory_dim = memory_dim
        # TGN 设计：记忆维度与输入（消息）维度可不同。
        # GRUCell(input_size, hidden_size)：输入为 node_embeddings[*, input_dim]，
        # 隐状态为 memory[*, memory_dim]。当上游 GNN 输出维度（hidden_dim）
        # 与 memory_dim 不一致时，旧实现用 GRUCell(memory_dim, memory_dim) 会在
        # forward 处抛 RuntimeError: input.size(-1) must be equal to input_size。
        # 此处显式分离 input_dim 与 memory_dim，修复该潜在崩溃。
        # 向后兼容：未指定 input_dim 时退化为 memory_dim（原行为）。
        self.input_dim = input_dim if input_dim is not None else memory_dim
        self.gru = nn.GRUCell(self.input_dim, memory_dim)
        # 使用 register_buffer 而非 nn.Parameter，因为 memory 不参与梯度计算
        self.register_buffer('_memory', torch.zeros(n_nodes, memory_dim))

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, value):
        self._memory.copy_(value)

    def forward(self, node_embeddings):
        n_actual = node_embeddings.shape[0]
        if n_actual != self._memory.shape[0]:
            # 实际节点数与初始化时不一致：动态 resize memory。
            # 这是为了修复 MultimodalGNN 默认 n_nodes=20 但实际图节点数不同导致的
            # RuntimeError: input.size(-1) must be equal to input_size。
            # 保持已学习 memory 的前 min(n_actual, n_nodes) 行，其余补零。
            new_memory = torch.zeros(
                n_actual, self.memory_dim,
                device=self._memory.device, dtype=self._memory.dtype)
            n_keep = min(n_actual, self._memory.shape[0])
            if n_keep > 0:
                new_memory[:n_keep] = self._memory[:n_keep]
            self._memory = new_memory
        self.memory = self.gru(node_embeddings, self.memory).detach()
        return self.memory

    def reset_memory(self):
        self.memory.zero_()