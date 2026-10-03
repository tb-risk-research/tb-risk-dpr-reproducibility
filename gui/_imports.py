#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 子模块共享导入和常量

供 gui/base.py、gui/charts/_shared.py、gui/training_panel/_shared.py 等
gui 子模块引用，避免在每个文件中重复相同的导入块。

使用方式：
    from .._imports import *  # noqa: F401,F403
"""

# ==================== 标准库 ====================
import copy
import csv
import datetime
import json
import logging
import math
import os
import queue
import random
import re
import sys
import threading
import time

# tkinter 是 GUI 依赖，在无图形环境（服务器/容器）中可能不可用
from ..utils import TKINTER_AVAILABLE  # tkinter 可用性标志（单一真相源）
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:
    tk = None
    ttk = None
    messagebox = None
    filedialog = None

# ==================== 科学计算库（可选依赖） ====================
# numpy/matplotlib 从 ..utils 统一导入（单一真相源，避免重复设置后端）
from ..utils import (
    np, NUMPY_AVAILABLE,
    Figure, plt, FigureCanvasTkAgg, NavigationToolbar2Tk, MATPLOTLIB_AVAILABLE,
)

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

try:
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy = None

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    sklearn = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None

try:
    import torch_geometric
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    torch_geometric = None

# ==================== 可用性标志（未就绪模块） ====================
# 因果强化学习（ml/policy.py 模块存在但 RLInterventionEngine 依赖未就绪，暂硬编码为 False）
# TODO: 当 RL 模块就绪后，改为 from ..integrator import CAUSAL_RL_AVAILABLE, RLInterventionEngine
CAUSAL_RL_AVAILABLE = False
TBInterventionEnv = None
RLInterventionEngine = None

# Physics-Informed GNN（ml/framework.py 中未定义 PIGNN 类，与 scoring/predictor.py 保持一致）
PIGNN_AVAILABLE = False

# 生存分析 & 多结局竞争风险（ml/survival.py 模块不存在，与 scoring/predictor.py 保持一致）
# TODO: 当 survival 模块就绪后，改为 from ..scoring.predictor import SURVIVAL_BAYESIAN_AVAILABLE, MULTISTATE_AVAILABLE
SURVIVAL_BAYESIAN_AVAILABLE = False
MULTISTATE_AVAILABLE = False

# ==================== 项目内部依赖（可选，统一 try/except 守卫） ====================
# 这些导入原先散落在 base.py 和各面板 _shared.py 中，现集中到单一真相源，
# 子模块通过 `from .._imports import *` 即可获得全部符号。

# 接触者编辑对话框（仅依赖 tkinter，与 mixins 同生命周期加载）
from .dialog import ContactEditDialog

# 克拉玛依本土化适配器（可选依赖）
try:
    from ..karamay.localizer import KaramayLocalizer
    KARAMAY_LOCALIZER_AVAILABLE = True
except ImportError:
    KaramayLocalizer = None
    KARAMAY_LOCALIZER_AVAILABLE = False

try:
    from ..karamay.validator import KaramayValidator
    KARAMAY_VALIDATOR_AVAILABLE = True
except ImportError:
    KaramayValidator = None
    KARAMAY_VALIDATOR_AVAILABLE = False

# 工具函数与日志器（始终可用，utils.py 仅依赖标准库）
from ..utils import LOGGER, _is_yes

# core 业务层函数（供子模块直接调用，消除 assessment.py 委托方法）
try:
    from ..core import calculate_cumulative_exposure
except ImportError:
    calculate_cumulative_exposure = None

# data_io 统一常量（Phase 0 集成）
try:
    from ..data_io.conf import CLINICAL_TEXT_MAP
except ImportError:
    CLINICAL_TEXT_MAP = {}

# scipy odeint（可选，SCIPY_AVAILABLE 已在上方检测）
try:
    from scipy.integrate import odeint
except ImportError:
    odeint = None

# PyG 可用性标志与多模态图构建（与 ml/graph.py 一致）
try:
    from ..ml.graph import (_TORCH_GEOMETRIC_PIGNN_OK, MultiModalDataAccess,
                             EnhancedTemporalGraphBuilder)
except ImportError:
    _TORCH_GEOMETRIC_PIGNN_OK = False
    MultiModalDataAccess = None
    EnhancedTemporalGraphBuilder = None

# GNN 模型类（可选，依赖 torch/pyg）
try:
    from ..ml.gnn import HeteroTimeVaryingGNN, MultimodalGNN
except ImportError:
    HeteroTimeVaryingGNN = None
    MultimodalGNN = None

# 训练器（可选）
try:
    from ..ml.trainer import ThreePhaseTrainer, TemporalTrainer
except ImportError:
    ThreePhaseTrainer = None
    TemporalTrainer = None

# 干预规划与验证（可选）
try:
    from ..ml.planner import DynamicInterventionPlanner, InterventionValidator
except ImportError:
    DynamicInterventionPlanner = None
    InterventionValidator = None

# 反事实优化器（可选）
try:
    from ..ml.optimizer import DifferentiableCounterfactualOptimizer
except ImportError:
    DifferentiableCounterfactualOptimizer = None

# 偏好条件策略网络（可选）
try:
    from ..ml.policy import PreferenceConditionedPolicy
except ImportError:
    PreferenceConditionedPolicy = None

# SEIR 参数不确定性（可选）
try:
    from ..seir.uncertainty import SEIRParameterUncertainty
except ImportError:
    SEIRParameterUncertainty = None

# 不确定性量化集成（可选）
try:
    from ..ml.uncertainty import (DeepEnsemblePredictor, ConformalPredictor,
                                   MCDropoutGNNUncertainty, SWAGEstimator,
                                   UncertaintyFusionEngine, UncertaintyVisualizer,
                                   QuantileRegressionForest)
except ImportError:
    DeepEnsemblePredictor = None
    ConformalPredictor = None
    MCDropoutGNNUncertainty = None
    SWAGEstimator = None
    UncertaintyFusionEngine = None
    UncertaintyVisualizer = None
    QuantileRegressionForest = None


# ==================== 受控导出列表 ====================
# 仅导出明确声明的公共名称，避免内部标志位泄漏
# 所有可用性标志位统一使用 PYTORCH_AVAILABLE / PYG_AVAILABLE 等公开命名
__all__ = [
    # 标准库
    'copy', 'csv', 'datetime', 'json', 'logging', 'math', 'os', 'queue',
    'random', 're', 'threading', 'time', 'tk', 'ttk', 'messagebox', 'filedialog',
    'TKINTER_AVAILABLE',
    # 科学计算库（模块引用 + 可用性标志）
    'np', 'NUMPY_AVAILABLE',
    'MATPLOTLIB_AVAILABLE', 'FigureCanvasTkAgg', 'NavigationToolbar2Tk',
    'Figure', 'plt',
    'sns', 'SEABORN_AVAILABLE',
    'scipy', 'SCIPY_AVAILABLE',
    'sklearn', 'SKLEARN_AVAILABLE',
    'xgb', 'XGBOOST_AVAILABLE',
    'lgb', 'LIGHTGBM_AVAILABLE',
    'cb', 'CATBOOST_AVAILABLE',
    'shap', 'SHAP_AVAILABLE',
    'joblib', 'JOBLIB_AVAILABLE',
    'pd', 'PANDAS_AVAILABLE',
    'torch', 'PYTORCH_AVAILABLE',
    'torch_geometric', 'PYG_AVAILABLE',
    # 可用性标志
    'CAUSAL_RL_AVAILABLE', 'PIGNN_AVAILABLE',
    'SURVIVAL_BAYESIAN_AVAILABLE', 'MULTISTATE_AVAILABLE',
    # 项目内部依赖（对话框 / 克拉玛依 / 工具 / core / data_io / scipy odeint）
    'ContactEditDialog',
    'KARAMAY_LOCALIZER_AVAILABLE', 'KaramayLocalizer',
    'KARAMAY_VALIDATOR_AVAILABLE', 'KaramayValidator',
    'LOGGER', '_is_yes',
    'calculate_cumulative_exposure',
    'CLINICAL_TEXT_MAP',
    'odeint',
    # PyG & 图构建
    '_TORCH_GEOMETRIC_PIGNN_OK', 'MultiModalDataAccess', 'EnhancedTemporalGraphBuilder',
    # GNN 模型
    'HeteroTimeVaryingGNN', 'MultimodalGNN',
    # 训练器
    'ThreePhaseTrainer', 'TemporalTrainer',
    # 干预规划
    'DynamicInterventionPlanner', 'InterventionValidator',
    # 反事实优化
    'DifferentiableCounterfactualOptimizer',
    # 策略网络
    'PreferenceConditionedPolicy',
    # SEIR
    'SEIRParameterUncertainty',
    # 不确定性量化
    'DeepEnsemblePredictor', 'ConformalPredictor', 'MCDropoutGNNUncertainty',
    'SWAGEstimator', 'UncertaintyFusionEngine', 'UncertaintyVisualizer',
    'QuantileRegressionForest',
]