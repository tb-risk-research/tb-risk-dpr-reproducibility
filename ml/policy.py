"""
偏好条件多目标策略网络

将成本权重作为额外输入，使单一策略能根据决策者偏好
输出不同强度（成本敏感 vs 效果敏感）的干预序列。

文献：Hayes B et al. (2022) Multi-objective RL with preference conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容原始的 _GNN_BASE_CLASS 模式
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


class PreferenceConditionedPolicy(_GNN_BASE_CLASS):
    """偏好条件多目标策略网络

    将成本权重 w_c 作为额外输入，使单一策略能根据决策者偏好
    输出不同强度（成本敏感 vs 效果敏感）的干预序列。

    文献：Hayes B et al. (2022) Multi-objective RL with preference conditioning
    """

    def __init__(self, state_dim=31, n_actions=12, preference_dim=2,
                 hidden_dim=128, dropout=0.2):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.preference_dim = preference_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.pref_encoder = nn.Sequential(
            nn.Linear(preference_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.causal_graph_constraint = None

    def set_causal_constraint(self, causal_constraint):
        self.causal_graph_constraint = causal_constraint

    def forward(self, state, preference):
        s_emb = self.state_encoder(state)
        p_emb = self.pref_encoder(preference)
        fused = self.fusion(torch.cat([s_emb, p_emb], dim=-1))
        action_logits = self.actor(fused)
        state_value = self.critic(fused)
        return action_logits, state_value

    def get_action_probs(self, state, preference):
        logits, value = self.forward(state, preference)
        return F.softmax(logits, dim=-1), value

    def select_action(self, state, preference, deterministic=False):
        with torch.no_grad():
            device = next(self.parameters()).device
            probs, _ = self.get_action_probs(
                torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                torch.tensor(preference, dtype=torch.float32, device=device).unsqueeze(0))
            if deterministic:
                action = int(probs.argmax(dim=-1).item())
            else:
                action = int(torch.multinomial(probs, 1).item())
            return action

    def compute_causal_loss(self, state, preference):
        if self.causal_graph_constraint is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        probs, _ = self.get_action_probs(state, preference)
        return self.causal_graph_constraint.causal_regularization_loss(
            probs, state)

    def train_step(self, state, action, reward, next_state, done,
                   preference, optimizer, gamma=0.95):
        device = next(self.parameters()).device
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        pref_t = torch.tensor(preference, dtype=torch.float32, device=device).unsqueeze(0)

        action_logits, value = self.forward(state_t, pref_t)
        _, next_value = self.forward(next_state_t, pref_t)

        target = reward + gamma * next_value.detach().squeeze() * (1 - float(done))
        critic_loss = F.mse_loss(value.squeeze(), target)

        probs = F.softmax(action_logits, dim=-1)
        log_prob = torch.log(probs.squeeze()[action] + 1e-10)
        advantage = target - value.detach().squeeze()
        actor_loss = -log_prob * advantage
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        causal_loss = self.compute_causal_loss(state_t, pref_t)

        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy + 0.1 * causal_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            'total_loss': float(total_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'critic_loss': float(critic_loss.item()),
            'entropy': float(entropy.item()),
            'causal_loss': float(causal_loss.item()) if isinstance(causal_loss, torch.Tensor) else 0.0,
        }