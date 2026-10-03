#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN训练包：SEIRInformedGNN、ST-GNN、PI-GNN训练与评估

从原 scoring/ml/gnn_training.py（单文件）拆分为多个子模块的包。
所有函数无状态，通过参数传递数据。

本 ``__init__.py`` 通过 re-export 保留所有原有公共 API（类、函数、常量），
使现有导入路径 ``from tb_risk.scoring.ml.gnn_training import XXX`` 继续有效。

子模块结构：
    - _common: 共享常量、可用性标志、框架类引用
    - data_preparation: 合成图数据集生成（数据准备）
    - training_utils: Focal Loss 与单次训练辅助函数
    - training_gnn: SEIRInformedGNN 完整训练循环
    - training_pignn: Physics-Informed GNN 训练（占位壳）
    - training_stgnn: 时空图神经网络训练循环
    - evaluation: GNN 模型评估
    - inference: GNN 风险预测
"""
from ._common import (
    LOGGER,
    SKLEARN_AVAILABLE,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    CATBOOST_AVAILABLE,
    SHAP_AVAILABLE,
    JOBLIB_AVAILABLE,
    PANDAS_AVAILABLE,
    PYTORCH_AVAILABLE,
    PYG_AVAILABLE,
    NUMPY_AVAILABLE,
    PIGNN_AVAILABLE,
    HeterogeneousTBNetwork,
    SEIRInformedGNN,
    SEIRTimeAwareGNN,
    PIGNN,
)
from .data_preparation import _generate_synthetic_graphs
from .training_utils import _focal_loss, _train_gnn_once
from .training_gnn import train_gnn
from .training_pignn import train_pignn
from .training_stgnn import train_stgnn
from .evaluation import _evaluate_gnn
from .inference import predict_gnn_risk, predict_time_aware_gnn_risk, predict_gnn_increment
from .explain import compute_gnn_explanation

__all__ = [
    # 常量与可用性标志
    'LOGGER',
    'SKLEARN_AVAILABLE',
    'XGBOOST_AVAILABLE',
    'LIGHTGBM_AVAILABLE',
    'CATBOOST_AVAILABLE',
    'SHAP_AVAILABLE',
    'JOBLIB_AVAILABLE',
    'PANDAS_AVAILABLE',
    'PYTORCH_AVAILABLE',
    'PYG_AVAILABLE',
    'NUMPY_AVAILABLE',
    'PIGNN_AVAILABLE',
    # 框架类引用
    'HeterogeneousTBNetwork',
    'SEIRInformedGNN',
    'SEIRTimeAwareGNN',
    'PIGNN',
    # 公共函数
    'train_gnn',
    'train_pignn',
    'train_stgnn',
    'predict_gnn_risk',
    'predict_time_aware_gnn_risk',
    'predict_gnn_increment',
    'compute_gnn_explanation',
    # 内部函数（被 predictor 绑定为方法，需保持可导入）
    '_generate_synthetic_graphs',
    '_focal_loss',
    '_train_gnn_once',
    '_evaluate_gnn',
]
