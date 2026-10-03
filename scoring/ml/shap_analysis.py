#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SHAP可解释性分析

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

def compute_shap_values(predictor, contact_data_list, contact_types=None):
        """计算SHAP值用于可解释性分析
        
        由于树模型训练时不使用标准化，SHAP值直接解释原始特征，
        便于临床解读（如"年龄每增加1岁，风险增加X%"）。
        
        参数：
            contact_data_list (list): 接触者数据字典列表
            contact_types (list): 对应的接触者类型列表，默认均为'family'
        
        返回：
            dict: SHAP分析结果，包含shap_values(numpy数组)、
                  feature_importance(排序后的特征重要性列表)、
                  feature_names(特征名列表)、feature_descriptions(特征中文描述)
        """
        if not SHAP_AVAILABLE or predictor.shap_explainer is None or not predictor.is_trained:
            return None
        
        if contact_types is None:
            contact_types = ['family'] * len(contact_data_list)
        
        try:
            all_features = []
            for contact_data, contact_type in zip(contact_data_list, contact_types):
                features = predictor.extract_features_with_interactions(contact_data, contact_type)
                all_features.append(features[0])
            
            feature_matrix = np.array(all_features)
            
            shap_values = predictor.shap_explainer.shap_values(feature_matrix)
            
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]
            elif isinstance(shap_values, list) and len(shap_values) == 1:
                shap_values = shap_values[0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
            
            predictor.last_shap_values = shap_values
            predictor.last_feature_matrix = feature_matrix
            
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            feature_importance = sorted(
                zip(predictor.ALL_FEATURE_NAMES, mean_abs_shap),
                key=lambda x: x[1], reverse=True
            )
            
            return {
                'shap_values': shap_values,
                'feature_matrix': feature_matrix,
                'feature_importance': feature_importance,
                'feature_names': predictor.ALL_FEATURE_NAMES,
                'feature_descriptions': predictor.ALL_FEATURE_DESCRIPTIONS
            }

        except Exception as e:
            LOGGER.warning("SHAP计算失败: %s", e, exc_info=True)
            return None


def compute_contact_attributions(predictor, contact_data_list, contact_types=None):
    """为每个接触者计算逐因素 SHAP 风险归因（可解释性分析）。

    与 ``compute_shap_values`` 的差异：后者返回全体样本聚合的特征重要性，
    本函数返回与输入一一对应的"逐接触者"归因列表，便于在评估结果下方
    展示"该接触者风险分是如何算出来的"。

    每个接触者的归因项形如：:

        {
            'feature': 'single_duration',
            'description': '单次接触时长(分钟)',
            'feature_value': 120.0,          # 该接触者的原始特征值
            'contribution': 12.3,            # SHAP 贡献（>0 推高风险，<0 降低风险）
            'abs_contribution': 12.3,        # 用于排序的绝对值
        }

    返回：
        list | None: 与 ``contact_data_list`` 等长对齐的归因列表，
        每项是按 |contribution| 降序排列的因素贡献 dict 列表；
        SHAP 不可用/未训练/计算失败时返回 None。
    """
    if not SHAP_AVAILABLE or predictor.shap_explainer is None or not predictor.is_trained:
        return None

    if contact_types is None:
        contact_types = ['family'] * len(contact_data_list)

    try:
        all_features = []
        for contact_data, contact_type in zip(contact_data_list, contact_types):
            features = predictor.extract_features_with_interactions(contact_data, contact_type)
            all_features.append(features[0])

        feature_matrix = np.array(all_features)

        shap_values = predictor.shap_explainer.shap_values(feature_matrix)

        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        elif isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        feature_names = predictor.ALL_FEATURE_NAMES
        feature_descriptions = predictor.ALL_FEATURE_DESCRIPTIONS

        attributions = []
        for feature_row, shap_row in zip(feature_matrix, shap_values):
            contributions = []
            for i, name in enumerate(feature_names):
                contribution = float(shap_row[i])
                contributions.append({
                    'feature': name,
                    'description': feature_descriptions.get(name, name),
                    'feature_value': float(feature_row[i]),
                    'contribution': contribution,
                    'abs_contribution': abs(contribution),
                })
            contributions.sort(key=lambda c: c['abs_contribution'], reverse=True)
            attributions.append(contributions)

        return attributions

    except Exception as e:
        LOGGER.warning("SHAP逐因素归因计算失败: %s", e, exc_info=True)
        return None


# ============================================================
# 跨模型 SHAP 共识（问题3：统一解释口径）
# ============================================================

def _positive_class_shap(shap_values):
    """把各类 SHAP 输出形态归一为 (n_samples, n_features) 正类贡献矩阵。

    兼容：list[class][n,f]（旧 shap）、ndarray[n,f,classes]（新 shap）、
    shap.Explanation（.values）。
    """
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_values = shap_values[1]
        elif shap_values:
            shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = (shap_values[:, :, 1] if shap_values.shape[2] == 2
                       else shap_values[:, :, -1])
    return shap_values


def compute_cross_model_shap(predictor, feature_matrix=None,
                             contact_data_list=None, contact_types=None,
                             max_samples=200, random_state=42):
    """跨模型 SHAP 统一解释口径（问题3）。

    各模型的 feature_importances_（gain/split 口径）不可跨模型比较，
    造成"随机森林说暴露最重要、XGBoost 说症状最重要"的表象分歧。
    本函数对全部已训练树模型分别建 TreeExplainer，在**同一特征矩阵**上
    计算 mean|SHAP|，按模型内归一化为份额（消除各模型输出量纲差异），
    再聚合跨模型共识：

    - ``consensus_share``：各模型份额均值（共识重要性）；
    - ``share_std``：份额标准差（模型间分歧幅度）；
    - ``ranks`` / ``rank_min`` / ``rank_max``：各模型内排名与排名区间
      ——若分歧在 SHAP 口径下仍存在，是真实模型分歧而非口径假象。

    模型输入按各自 ``feature_names_in_`` 选列（v1 22 维 / v2 15 维模型
    可同批参与；旧 numpy 档按位置映射）。

    Args:
        predictor: 已训练的 predictor（models 至少 2 个树模型）
        feature_matrix: (n, 22) 特征矩阵；None 且提供 contact_data_list
            时由 extract_features_with_interactions 构建
        contact_data_list: 接触者数据字典列表（feature_matrix 为 None 时用）
        contact_types: 对应接触类型列表
        max_samples: SHAP 计算的确定性子采样上限（默认 200，树 SHAP 足够）
        random_state: 子采样种子（同输入同结果）

    Returns:
        dict | None: 成功模型 < 2 或依赖缺失时返回 None
    """
    if not SHAP_AVAILABLE:
        return None

    # 1. 特征矩阵构建
    if feature_matrix is None:
        if not contact_data_list:
            return None
        if contact_types is None:
            contact_types = ['family'] * len(contact_data_list)
        try:
            rows = []
            for contact_data, contact_type in zip(contact_data_list,
                                                  contact_types):
                features = predictor.extract_features_with_interactions(
                    contact_data, contact_type)
                rows.append(features[0])
            feature_matrix = np.array(rows)
        except Exception as e:
            LOGGER.warning("跨模型 SHAP 特征构建失败: %s", e)
            return None

    arr = np.asarray(feature_matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    all_names = list(getattr(predictor, 'ALL_FEATURE_NAMES', None) or [])
    if len(all_names) != arr.shape[1]:
        LOGGER.warning(
            "跨模型 SHAP 需要与 ALL_FEATURE_NAMES 同维的特征矩阵"
            "（%d 列），实际 %d 列", len(all_names), arr.shape[1])
        return None

    # 2. 确定性子采样（同输入同结果）
    if arr.shape[0] > max_samples:
        rng = np.random.RandomState(random_state)
        idx = np.sort(rng.choice(arr.shape[0], size=max_samples,
                                 replace=False))
        arr = arr[idx]

    # 3. 逐模型 SHAP 份额
    from .evaluation import _model_input_frame
    per_model = {}
    model_display = {}
    for key, model in (getattr(predictor, 'models', None) or {}).items():
        try:
            names_in = getattr(model, 'feature_names_in_', None)
            if names_in is not None and set(names_in) <= set(all_names):
                feat_names = list(names_in)          # v1 全名 / v2 子集
            else:
                feat_names = list(all_names)         # 旧 numpy/Column_N 档按位置
            X_model = _model_input_frame(model, arr, all_names)
            explainer = shap.TreeExplainer(model)
            sv = _positive_class_shap(explainer.shap_values(X_model))
            if sv.shape != (arr.shape[0], len(feat_names)):
                LOGGER.warning(
                    "模型 %s SHAP 形状 %s 与特征名 %d 不符，跳过",
                    key, sv.shape, len(feat_names))
                continue
            mean_abs = np.mean(np.abs(sv), axis=0)
            total = float(mean_abs.sum())
            if total <= 0:
                continue
            per_model[key] = {
                'shares': {feat_names[j]: float(mean_abs[j]) / total
                           for j in range(len(feat_names))},
            }
            model_display[key] = (
                getattr(predictor, 'model_performance', {})
                .get(key, {}).get('name', key))
        except Exception as e:
            LOGGER.warning("模型 %s 跨模型 SHAP 计算失败，跳过: %s", key, e)

    if len(per_model) < 2:
        LOGGER.warning("跨模型 SHAP 共识需要 >= 2 个可用模型，实际 %d",
                       len(per_model))
        return None

    # 4. 各模型内排名（份额降序，1 = 最重要）
    for key in per_model:
        ordered = sorted(per_model[key]['shares'].items(),
                         key=lambda kv: kv[1], reverse=True)
        per_model[key]['ranks'] = {feat: rank for rank, (feat, _)
                                   in enumerate(ordered, 1)}

    # 5. 共识聚合（特征顺序按 all_names 确定性给出，覆盖出现过的特征）
    feature_names = [f for f in all_names
                     if any(f in per_model[k]['shares'] for k in per_model)]
    consensus = []
    for feat in feature_names:
        shares_list = [per_model[k]['shares'][feat] for k in per_model
                       if feat in per_model[k]['shares']]
        ranks = {k: per_model[k]['ranks'][feat] for k in per_model
                 if feat in per_model[k]['shares']}
        consensus.append({
            'feature': feat,
            'consensus_share': float(np.mean(shares_list)),
            'share_std': float(np.std(shares_list)),
            'ranks': ranks,
            'rank_min': int(min(ranks.values())),
            'rank_max': int(max(ranks.values())),
        })
    consensus.sort(key=lambda e: e['consensus_share'], reverse=True)

    return {
        'method': 'cross_model_shap',
        'n_models': len(per_model),
        'model_names': model_display,
        'n_samples': int(arr.shape[0]),
        'feature_names': feature_names,
        'per_model': per_model,
        'consensus': consensus,
        'top_features': [(e['feature'], e['consensus_share'])
                         for e in consensus[:10]],
    }


def format_cross_model_shap_table(result, top_n=10):
    """跨模型 SHAP 共识表（Markdown）。

    列：排名 | 特征 | 共识份额 | 分歧(标准差) | 排名区间 | 各模型排名。
    排名区间 = 各模型内最好-最差排名——区间越宽，该特征重要性
    在模型间分歧越大（真实分歧，非 gain/split 口径假象）。
    """
    if not result or not result.get('consensus'):
        return ''
    model_keys = (list(result.get('model_names', {}).keys())
                  or list(result['per_model'].keys()))
    lines = [
        '| 排名 | 特征 | 共识份额 | 分歧(标准差) | 排名区间 | 各模型排名 |',
        '|------|------|----------|--------------|----------|------------|',
    ]
    for i, entry in enumerate(result['consensus'][:top_n], 1):
        per = '/'.join(str(entry['ranks'].get(k, '-')) for k in model_keys)
        lines.append(
            f"| {i} | {entry['feature']} | {entry['consensus_share']:.4f} "
            f"| {entry['share_std']:.4f} "
            f"| {entry['rank_min']}-{entry['rank_max']} | {per} |")
    return '\n'.join(lines)


