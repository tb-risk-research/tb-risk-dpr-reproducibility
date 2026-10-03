#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN训练子包共享模块 — 常量、可用性标志、框架类引用。

从原 scoring/ml/gnn_training.py 拆分而来，供 gnn_training 包内各子模块共享，
避免循环导入。所有子模块通过 ``from ._common import XXX`` 获取共享名称。
"""
import copy
import datetime
import logging
import math
import os
import random

import numpy as np

LOGGER = logging.getLogger("tb_risk.ml")

# 条件导入（与主类保持一致）
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 accuracy_score, precision_score, recall_score,
                                 f1_score, brier_score_loss)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# GNN子包不直接使用XGBoost/LightGBM/CatBoost，保留标志位用于API兼容性
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

NUMPY_AVAILABLE = True



try:
    from ....ml.framework import HeterogeneousTBNetwork, SEIRInformedGNN, SEIRTimeAwareGNN
except ImportError:
    HeterogeneousTBNetwork = None
    SEIRInformedGNN = None
    SEIRTimeAwareGNN = None

PIGNN = None
PIGNN_AVAILABLE = False
