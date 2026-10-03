#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 训练面板模块

TrainingMixin 组合了 GNNMixin（GNN 训练与干预优化）和
MLTrainingMixin（ML 训练、不确定性量化、生存分析、RL 规划）。

子模块：
- _shared.py:   共享导入和常量
- gnn_tab.py:   GNN 训练与干预优化（GNNMixin）
- ml_tab.py:    ML 训练与高级分析（MLTrainingMixin）
"""

from ._shared import *  # noqa: F401,F403 — 提供所有子模块共享的导入和常量
from .gnn_tab import GNNMixin
from .ml_tab import MLTrainingMixin


class TrainingMixin(GNNMixin, MLTrainingMixin):
    """GUI 训练面板方法（组合 GNN + ML 训练功能）"""
    pass