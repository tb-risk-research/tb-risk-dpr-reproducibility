#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN框架模块 - 异质图神经网络

仅导出公共 API 类；内部依赖（numpy/torch/PyG 组件等）请从 _common 显式导入。

公共导入路径：
    - ``from tb_risk.ml.framework import HeterogeneousTBNetwork``
    - ``from .framework import SEIRInformedGNN``        （ml 包内）
    - ``from ..ml.framework import SEIRTimeAwareGNN``    （scoring 包内）
"""

from ._common import PYTORCH_AVAILABLE, PYG_AVAILABLE
from .seir_informed import SEIRInformedGNN
from .heterogeneous_network import HeterogeneousTBNetwork
from .temporal import TemporalEncoder, SEIRTimeAwareGNN
from .spatio_temporal import SpatioTemporalTBGNN

# GNN 环境分级检测（问题三：无 torch/PyG 时的降级指引）
from ..gnn._env import (
    NUMPY_AVAILABLE,
    gnn_degradation_level,
    gnn_degradation_name,
    gnn_tier_description,
    gnn_install_guidance,
    gnn_env_status,
    gnn_env_status_text,
)

__all__ = [
    # 公共GNN模型类
    'SEIRInformedGNN',
    'HeterogeneousTBNetwork',
    'TemporalEncoder',
    'SEIRTimeAwareGNN',
    'SpatioTemporalTBGNN',
    # 依赖可用性标志
    'PYTORCH_AVAILABLE',
    'PYG_AVAILABLE',
    # GNN 环境分级（问题三）
    'NUMPY_AVAILABLE',
    'gnn_degradation_level',
    'gnn_degradation_name',
    'gnn_tier_description',
    'gnn_install_guidance',
    'gnn_env_status',
    'gnn_env_status_text',
]
