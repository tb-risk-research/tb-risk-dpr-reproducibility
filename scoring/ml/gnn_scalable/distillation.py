#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知识蒸馏（distillation）— 用大 GNN 蒸馏出轻量 MLP 学生模型。

实时临床决策需要毫秒级单样本推理。GNN 需在图/子图上做消息传递，延迟较高；
蒸馏出的轻量 MLP 学生模型只依赖节点自身特征（无需图结构），推理仅为若干
线性层，延迟可降到毫秒以下，适合部署到边缘/实时链路。

本模块实现：
- ``GNNStudentMLP``：轻量多层感知机学生（输入 = 节点特征）。
- ``distill_gnn_to_mlp``：用 GNN 教师在大图节点上的软输出（风险概率）训练学生，
  并支持 ``distill_with_subgraph`` 仅用目标节点邻域子图采样训练（避免全图前向）。

文献：
- Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network. 2015.
- Gou et al. Knowledge Distillation: A Survey. IJCV 2021.
"""
import copy
import logging
from typing import Dict, List, Optional

from ._common import (
    LOGGER, PYTORCH_AVAILABLE, PYG_AVAILABLE, torch,
)
from .sampling import sample_k_hop_subgraph

__all__ = ["GNNStudentMLP", "distill_gnn_to_mlp", "predict_with_mlp"]


if PYTORCH_AVAILABLE:
    import torch.nn as nn
    import torch.nn.functional as F

    class GNNStudentMLP(nn.Module):
        """轻量 MLP 学生：仅用节点特征预测风险概率（无需图结构）。"""

        def __init__(self, input_dim=30, hidden_dims=(64, 32), dropout=0.1):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, 1))
            layers.append(nn.Sigmoid())
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

        @property
        def n_params(self):
            return sum(int(p.numel()) for p in self.parameters())
else:
    GNNStudentMLP = None


def _teacher_forward(model, x, edge_index, edge_attr):
    if edge_attr is not None:
        risk, _ = model(x, edge_index, edge_attr=edge_attr)
    else:
        risk, _ = model(x, edge_index)
    return risk.squeeze()


def distill_gnn_to_mlp(teacher, large_data, student=None, input_dim=None,
                       num_hops=2, n_epochs=50, learning_rate=0.001,
                       batch_size=256, device=None, seed=42,
                       progress_callback=None):
    """用 GNN 教师在大图上的软输出蒸馏出轻量 MLP 学生。

    训练样本通过子图采样生成（避免整图前向）：
    1. 随机采样种子节点；
    2. 对每个种子节点抽取 k-hop 子图，教师对该种子节点输出软概率作为学生标签；
    3. 用 (节点特征, 教师软概率) 训练 MLP 学生（MSE 到教师软输出）。

    参数：
        teacher: 已训练 GNN 教师
        large_data: 大图 ``Data``
        student: 可选学生实例；None 自动创建 ``GNNStudentMLP``
        input_dim: 学生输入维度（默认取 large_data.x 特征维）
        num_hops: 教师推理邻域跳数
        n_epochs: 学生训练轮数
        batch_size: 每轮使用的种子节点数
        progress_callback: ``(epoch, total)``

    返回：
        dict: {success, student, loss_history, n_params_teacher, n_params_student}
        依赖缺失返回 {success: False}。
    """
    if not (PYTORCH_AVAILABLE and PYG_AVAILABLE) or GNNStudentMLP is None:
        return {"success": False}
    import torch.nn as nn
    import torch.nn.functional as F

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if input_dim is None:
        input_dim = int(large_data.x.shape[1])
    if student is None:
        student = GNNStudentMLP(input_dim=input_dim)
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    loss_history: List[float] = []
    rng = __import__('numpy').random.RandomState(seed)
    total = large_data.num_nodes

    for epoch in range(n_epochs):
        student.train()
        epoch_loss = 0.0
        n = 0
        seeds = rng.choice(total, size=min(batch_size, total), replace=False)
        for s in seeds:
            sub, mapping = sample_k_hop_subgraph(large_data, int(s), num_hops)
            if sub is None or mapping is None:
                continue
            sub = sub.to(device)
            with torch.no_grad():
                edge_attr = sub.edge_attr if (hasattr(sub, 'edge_attr') and sub.edge_attr is not None) else None
                soft = _teacher_forward(teacher, sub.x, sub.edge_index, edge_attr)
                if soft.dim() == 0:
                    soft = soft.reshape(1)
                target = soft[mapping].clamp(0.0, 1.0).reshape(1)
            x_node = sub.x[mapping].reshape(1, -1)
            optimizer.zero_grad()
            pred = student(x_node).reshape(1)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n += 1
        loss_history.append(epoch_loss / max(n, 1))
        if progress_callback:
            try:
                progress_callback(epoch + 1, n_epochs)
            except Exception:
                pass

    return {
        "success": True,
        "student": student,
        "loss_history": loss_history,
        "n_epochs": n_epochs,
        "n_params_teacher": int(sum(p.numel() for p in teacher.parameters())),
        "n_params_student": int(student.n_params),
    }


def predict_with_mlp(student, node_features, device=None):
    """用 MLP 学生仅基于节点特征预测风险概率（无图结构，毫秒级）。

    返回：
        dict: {risk_probability, risk_class}
        依赖缺失或输入非法返回 None。
    """
    if not PYTORCH_AVAILABLE or student is None:
        return None
    if device is None:
        device = next(student.parameters()).device
    student.eval()
    x = torch.as_tensor(node_features, dtype=torch.float32, device=device)
    if x.dim() == 1:
        x = x.reshape(1, -1)
    with torch.no_grad():
        prob = student(x).squeeze().detach().cpu().item()
    gnn_risk = float(max(0.0, min(100.0, prob * 100.0)))
    return {
        "risk_probability": gnn_risk,
        "risk_class": 1 if gnn_risk > 50 else 0,
    }