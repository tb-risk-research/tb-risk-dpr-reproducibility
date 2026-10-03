#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 训练面板 - 共享导入和常量（供 gnn_tab / ml_tab 子模块引用）

标准库、科学计算库及项目内部依赖统一维护在 gui/_imports.py 单一真相源中；
本文件仅保留训练面板特有的导入。
"""

from .._imports import *  # noqa: F401,F403 — 标准库、科学计算库与项目内部依赖共享导入

# 训练面板特有：torch 用于 GNN/ML 模型训练与推理
# （torch 已在 _imports.py 中作为可选依赖导入，此处显式引用以确保面板内可用）
try:
    import torch  # noqa: F811 — 显式重引用，面板内频繁使用
except ImportError:
    torch = None  # type: ignore
