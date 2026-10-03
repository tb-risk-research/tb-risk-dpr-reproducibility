#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生存分析：多状态生存模型训练

从 MLRiskPredictor 拆分出的独立函数模块。
所有函数无状态，通过参数传递数据。
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



# ============================================================
# 以下函数从 MLRiskPredictor 的实例方法拆分而来
# 每个函数接收 predictor 实例作为第一个参数
# ============================================================

def train_survival_multistate(predictor, n_samples=500, epochs=50, batch_size=64,
                                   learning_rate=0.001, log_callback=None,
                                   random_state=42):
        """训练竞争风险多结局生存分析模型（⚠️ 模块未实现，调用会失败）

        MultiStateSurvivalPredictor 尚未实现，此方法为占位壳。
        调用将返回错误信息而非崩溃。
        """
        if not predictor._multistate_ready or predictor.multi_state_predictor is None:
            if log_callback:
                log_callback("竞争风险模块不可用（缺少PyTorch）")
            return {'status': 'not_available'}

        msp = predictor.multi_state_predictor

        if log_callback:
            log_callback("生成多结局合成数据...")

        X, times, events, causes = msp.generate_multistate_data(
            n_samples=n_samples, random_state=random_state)

        n_events = (events > 0).sum()
        if log_callback:
            log_callback(f"生成{n_samples}样本 | 事件: {n_events} | "
                        f"截尾: {n_samples - n_events}")

        if log_callback:
            log_callback(f"训练竞争风险模型: {epochs}轮...")

        result = msp.fit(
            X, times, events, causes,
            epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, log_callback=log_callback)

        if log_callback:
            log_callback(f"训练完成! loss={result.get('final_loss', 'N/A')}")

        if log_callback:
            log_callback("评估模型性能...")

        eval_metrics = msp.evaluate(X, times, events, causes)

        if log_callback:
            log_callback("=== 原因别性能指标 ===")
            for outcome_label, metrics in eval_metrics.items():
                c_idx = metrics.get('c_index', 0)
                auc_52 = metrics.get('auc_52w', 0)
                log_callback(f"  {outcome_label}: C-index={c_idx:.4f}, "
                            f"AUC(52w)={auc_52:.4f}")

        return {
            'status': 'trained',
            'final_loss': result.get('final_loss'),
            'n_samples': n_samples,
            'n_events': n_events,
            'epochs': epochs,
            'eval_metrics': eval_metrics,
        }
    

