#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML模型训练：随机森林、梯度提升、XGBoost、LightGBM、CatBoost统一训练框架

重构说明：
    - 所有模型配置（构造函数、默认参数、超参数网格）统一在 MODEL_REGISTRY 中管理
    - 抽离 _train_single_model、_evaluate_model_cv、_train_all_registered_models 三个核心函数
    - train_models / train_from_arrays / train_from_real_data 三个入口函数共享相同的训练内核
    - 消除了之前三处（_train_models_from_arrays / train_from_arrays / train_from_real_data）
      完整重复的模型训练逻辑，后续修改模型参数只需改一处。
"""
import copy
import datetime
import logging
import math
import os
import random
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("tb_risk.ml")

# 条件导入（与主类保持一致）
try:
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (cross_val_score, cross_val_predict,
                                         StratifiedKFold, StratifiedGroupKFold)
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 accuracy_score, precision_score, recall_score,
                                 f1_score, brier_score_loss)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    clone = None
    RandomForestClassifier = None
    GradientBoostingClassifier = None
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None

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


def _log_model_availability():
    """在日志中输出模型可用性信息"""
    available = []
    unavailable = []
    model_checks = [
        ("随机森林 (RandomForest)", SKLEARN_AVAILABLE, "scikit-learn"),
        ("梯度提升 (GradientBoosting)", SKLEARN_AVAILABLE, "scikit-learn"),
        ("XGBoost", XGBOOST_AVAILABLE, "xgboost"),
        ("LightGBM", LIGHTGBM_AVAILABLE, "lightgbm"),
        ("CatBoost", CATBOOST_AVAILABLE, "catboost"),
        ("逻辑回归 (Logistic, 标准化管道)", SKLEARN_AVAILABLE, "scikit-learn"),
    ]
    for name, ok, pkg in model_checks:
        if ok:
            available.append(name)
        else:
            unavailable.append(f"{name} (pip install {pkg})")
    if available:
        LOGGER.info("可用ML模型: %s", ", ".join(available))
    if unavailable:
        LOGGER.info("以下ML模型因依赖缺失不可用（不影响其他模型）: %s",
                     "; ".join(unavailable))


# 注意：日志提示统一由 scoring/predictor.py 主入口触发，
# 避免通过 scoring.ml 直接导入时产生重复日志。
# 如需手动触发，可显式调用 _log_model_availability()。

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

# 族感知 v4 特征空间（P1，2026-09-16）：V4_SPECS/泄漏规则/哨兵清洗/
# 缺失指示器的单一真值源（pandas 缺失时置不可用，train_from_real_data
# 的 v4 请求自动回退 v1）
try:
    from .cohort_features import apply_v4_feature_space, detect_cohort
    COHORT_FEATURES_AVAILABLE = True
except ImportError:
    COHORT_FEATURES_AVAILABLE = False

# Cox PH 生存路径（P5，2026-09-16）：peru_mdr 转正——incident TB 天然是
# time-to-event 终点，follow_up_days 在场时生存路径接管（statsmodels
# 缺失时置不可用，train_from_real_data 的生存分支跳过并披露）
try:
    from .survival_cox import (SURVIVAL_TIME_COLUMNS,
                               STATSMODELS_SURVIVAL_AVAILABLE,
                               train_survival_model)
    SURVIVAL_COX_IMPORTABLE = True
except ImportError:
    SURVIVAL_COX_IMPORTABLE = False

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


# ============================================================================
# 统一模型注册表：所有模型配置集中管理，避免重复定义
# ============================================================================

@dataclass
class ModelSpec:
    """单个模型的规格说明"""
    key: str                          # 模型在 predictor.models 中的键名
    name_cn: str                      # 中文显示名
    name_en: str                      # 英文显示名
    model_cls: Callable               # 模型构造函数/类
    default_params: Dict[str, Any]    # 默认训练参数
    param_grid: Dict[str, List[Any]]  # 超参数搜索网格
    random_state_param: str = 'random_state'  # 随机种子参数名（CatBoost用random_seed）
    available: bool = True            # 依赖是否可用
    catch_exceptions: bool = False    # 是否捕获训练异常（LightGBM/CatBoost需要）


def _make_logistic_pipeline(**kwargs):
    """逻辑回归管道工厂（StandardScaler → LogisticRegression）。

    kwargs 全部透传给 LogisticRegression（嵌套参数 lr__C 等由
    Pipeline.set_params 处理，见 _interruptible_grid_search 的
    clone 路径）。22 维特征含 age / 暴露天数等异尺度列，LR 对
    尺度敏感必须标准化；树成员不受影响。
    """
    if not SKLEARN_AVAILABLE:  # pragma: no cover —— 环境守卫
        raise RuntimeError('scikit-learn 不可用')
    lr_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith('lr__')}
    # random_state_param（默认 'random_state'）无前缀，显式透传保种子可复现
    if 'random_state' in kwargs:
        lr_kwargs['random_state'] = kwargs['random_state']
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(**lr_kwargs)),
    ])


def _build_model_registry() -> Dict[str, ModelSpec]:
    """构建模型注册表（所有模型的参数统一在这里定义）"""
    registry = {}

    # ---- 随机森林 ----
    registry['random_forest'] = ModelSpec(
        key='random_forest',
        name_cn='随机森林',
        name_en='Random Forest',
        model_cls=RandomForestClassifier,
        default_params={
            'n_estimators': 300,
            'max_depth': 7,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
        },
        param_grid={
            'n_estimators': [100, 200],
            'max_depth': [6, 8],
            'min_samples_leaf': [2, 5],
        },
        available=SKLEARN_AVAILABLE,
    )

    # ---- 梯度提升 ----
    registry['gradient_boosting'] = ModelSpec(
        key='gradient_boosting',
        name_cn='梯度提升',
        name_en='Gradient Boosting',
        model_cls=GradientBoostingClassifier,
        default_params={
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
        },
        param_grid={
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
        },
        available=SKLEARN_AVAILABLE,
    )

    # ---- XGBoost ----
    registry['xgboost'] = ModelSpec(
        key='xgboost',
        name_cn='XGBoost',
        name_en='XGBoost',
        model_cls=xgb.XGBClassifier if xgb else None,
        default_params={
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'verbosity': 0,
        },
        param_grid={
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'min_child_weight': [1, 3],
        },
        available=XGBOOST_AVAILABLE,
    )

    # ---- LightGBM ----
    registry['lightgbm'] = ModelSpec(
        key='lightgbm',
        name_cn='LightGBM',
        name_en='LightGBM',
        model_cls=lgb.LGBMClassifier if lgb else None,
        default_params={
            'n_estimators': 300,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': -1,
            'class_weight': 'balanced',
        },
        param_grid={
            'n_estimators': [100, 200],
            'num_leaves': [15, 31],
            'learning_rate': [0.05, 0.1],
            'min_child_samples': [10, 30],
        },
        available=LIGHTGBM_AVAILABLE,
        catch_exceptions=True,
    )

    # ---- CatBoost ----
    registry['catboost'] = ModelSpec(
        key='catboost',
        name_cn='CatBoost',
        name_en='CatBoost',
        model_cls=cb.CatBoostClassifier if cb else None,
        default_params={
            'iterations': 300,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3.0,
            'verbose': 0,
            'auto_class_weights': 'Balanced',
        },
        param_grid={
            'iterations': [100, 200],
            'depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'l2_leaf_reg': [1.0, 5.0],
        },
        random_state_param='random_seed',
        available=CATBOOST_AVAILABLE,
        catch_exceptions=True,
    )

    # ---- 逻辑回归（加性线性，注册表唯一非树成员）----
    # 多样性依据（2026-09-05 地基审计）：注册表原 5 成员全为树家族、
    # 高度相关；真实队列复验（real_data_joint_homeacf/pacts_20260904.json，
    # 20 种子户级 CV）显示同信息集下加性 logistic 是真实数据最优操作点
    # （HomeACF logit_lin 0.6544 > best_existing 0.6486 > joint 0.6357，
    # 逐种子 DeLong 75% 显著）——树集成在合成地基上学到的"更优"排序
    # 在真实数据上反转。加 LR 为树间相关性提供正交成员；特征尺度
    # 敏感（22 维含 age/暴露天数等异尺度列）故必须 Pipeline 标准化，
    # 树成员不受影响仍直接吃原始特征（SHAP 可解释原始列）。
    registry['logistic'] = ModelSpec(
        key='logistic',
        name_cn='逻辑回归',
        name_en='Logistic Regression',
        model_cls=_make_logistic_pipeline,
        default_params={
            'lr__C': 1.0,
            'lr__max_iter': 2000,
            'lr__class_weight': 'balanced',
        },
        param_grid={
            'lr__C': [0.1, 1.0, 10.0],
            'lr__class_weight': [None, 'balanced'],
        },
        available=SKLEARN_AVAILABLE,
    )

    return registry


# 全局模型注册表
MODEL_REGISTRY = _build_model_registry()

# 默认训练顺序（logistic 置末：新成员不改变既有顺序语义，
# 树成员间相关性由其正交加性结构打破）
DEFAULT_MODEL_ORDER = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost', 'logistic']


# ============================================================================
# 核心训练函数：统一模型训练与评估流程
# ============================================================================

def _init_shap_explainer_with_fallback(predictor):
    """初始化SHAP解释器，带次优模型回退机制

    按AUROC从高到低尝试每个树模型创建TreeExplainer，
    直到成功或所有模型都试过。确保CatBoost等模型创建失败时
    能自动回退到XGBoost/LightGBM等兼容模型。
    """
    if not SHAP_AVAILABLE or len(predictor.models) == 0:
        predictor.shap_explainer = None
        return

    # 按AUROC从高到低排序模型
    sorted_models = sorted(
        predictor.model_performance.items(),
        key=lambda x: x[1].get('AUROC', 0),
        reverse=True
    )

    for model_name, perf in sorted_models:
        if model_name not in predictor.models:
            continue
        model = predictor.models[model_name]
        try:
            predictor.shap_explainer = shap.TreeExplainer(model)
            LOGGER.info("SHAP解释器初始化成功，使用模型: %s (AUROC=%.4f)",
                       perf.get('name', model_name), perf.get('AUROC', 0))
            return
        except Exception as e:
            LOGGER.warning("SHAP解释器使用 %s 创建失败，尝试下一个模型: %s",
                          perf.get('name', model_name), e)
            continue

    LOGGER.warning("所有模型均无法创建SHAP解释器，特征重要性分析将不可用")
    predictor.shap_explainer = None


def _interruptible_grid_search(predictor, model, param_grid, X, y, cv=5,
                               scoring='roc_auc', sample_weight=None,
                               groups=None):
    """可中断的网格搜索，支持中途停止

    参数：
        model: 基础模型
        param_grid: 参数网格
        X: 特征矩阵
        y: 标签
        cv: 交叉验证折数或 splitter 对象（组感知路径传入
            StratifiedGroupKFold 实例）
        scoring: 评分指标
        sample_weight: 类不平衡加权（GB 路径）：CV 评分与最终拟合
                       均使用同一权重（sklearn 自动按训练折切片），
                       保证超参搜索条件与实际训练条件一致
        groups (array-like|None): 样本所属组（household/community id）。
            提供时 cross_val_score 走组感知折——同组样本不跨训练/验证
            折，防止户级泄漏使超参选择偏向乐观（对齐实验脚本协议）。

    返回：
        最优模型（或默认模型如果被中断）
    """
    from itertools import product
    import numpy as np

    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    best_score = -np.inf
    best_params = None

    # 逐个尝试参数组合，每步检查停止标志
    for params_tuple in param_combinations:
        if predictor._stop_training:
            break

        # 将参数元组转为字典
        params = dict(zip(param_names, params_tuple))

        try:
            # sklearn 惯例 clone + set_params：与 model.__class__(**get_params(),
            # **params) 对普通估计器语义等价，且唯一支持 Pipeline 成员
            # （logistic = StandardScaler + LR 管道；Pipeline(**get_params())
            # 会因 steps/scaler 等嵌套键 TypeError）
            current_model = clone(model).set_params(**params)

            # 交叉验证评分：sample_weight 与最终拟合同权重
            # （此前的评分-拟合条件不一致使选出的超参偏向全局 AUROC；
            #  sklearn cross_val_score 自动按训练折索引切权重数组）
            fit_params = {}
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
            scores = cross_val_score(current_model, X, y, cv=cv,
                                     scoring=scoring, n_jobs=-1,
                                     params=fit_params or None,
                                     groups=groups)
            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        except (ValueError, TypeError, AttributeError):
            continue

    # 使用最佳参数训练最终模型（clone + set_params：Pipeline 兼容，
    # 见上方同理由；sample_weight=None 不传，Pipeline.fit 参数路由）
    if best_params is not None:
        best_model = clone(model).set_params(**best_params)
        if sample_weight is not None:
            best_model.fit(X, y, sample_weight=sample_weight)
        else:
            best_model.fit(X, y)
        return best_model
    else:
        # 如果被中断或没有找到最佳参数，使用默认模型
        if sample_weight is not None:
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)
        return model


def _class_imbalance_fit_kwargs(spec_key: str, y) -> tuple:
    """类不平衡统一加权（公共函数，所有训练路径复用）

    RF(class_weight='balanced')、LightGBM(class_weight='balanced')、
    CatBoost(auto_class_weights='Balanced') 已在注册表默认参数中；
    此处补齐剩余两个模型：
      - XGBoost：scale_pos_weight = n_neg/n_pos（构造参数）
      - sklearn GB：不支持 class_weight，用 balanced sample_weight

    返回：
        (extra_kwargs, sample_weight)：构造参数增补字典与 fit 样本权重
        （无加权时分别为 {} / None）
    """
    n_pos = float(np.sum(np.asarray(y) == 1))
    n_neg = float(np.sum(np.asarray(y) == 0))
    scale_pos_weight = max(n_neg / max(n_pos, 1.0), 1.0)
    extra_kwargs = {}
    if spec_key == 'xgboost':
        extra_kwargs['scale_pos_weight'] = scale_pos_weight
    sample_weight = None
    if spec_key == 'gradient_boosting':
        sample_weight = np.where(np.asarray(y) == 1, scale_pos_weight, 1.0)
    return extra_kwargs, sample_weight


def _train_single_model(spec: ModelSpec, X, y, *, enable_hyperopt: bool,
                         random_state: int, cv_folds: int = 3,
                         stop_check: Optional[Callable[[], bool]] = None,
                         groups=None):
    """训练单个模型（核心函数，所有入口共用）

    参数：
        spec: ModelSpec 模型规格
        X: 特征矩阵
        y: 标签
        enable_hyperopt: 是否启用超参数搜索
        random_state: 随机种子
        cv_folds: 超参数搜索的交叉验证折数
        stop_check: 停止检查回调函数，返回True表示中断
        groups (array-like|None): 样本所属组（household/community id）。
            提供时超参搜索 CV 切换为 StratifiedGroupKFold（同组样本
            不跨训练/验证折），防止户级泄漏使超参选择失真。

    返回：
        训练好的模型实例，失败返回None
    """
    if not spec.available or spec.model_cls is None:
        return None

    if stop_check and stop_check():
        return None

    # 准备基础参数字典（设置随机种子）
    base_kwargs = {spec.random_state_param: random_state}
    default_kwargs = {**spec.default_params, spec.random_state_param: random_state}

    # ---- 类不平衡统一加权（阳性率 ~3.6% 时必要）----
    # 公共加权逻辑抽至 _class_imbalance_fit_kwargs，与
    # _train_models_quick（评估/对比路径）共用同一实现，
    # 保证 Wilcoxon 比较与主训练模型的类不平衡处理一致。
    imbalance_kwargs, sample_weight = _class_imbalance_fit_kwargs(spec.key, y)
    base_kwargs.update(imbalance_kwargs)
    default_kwargs.update(imbalance_kwargs)

    # 组感知超参搜索 splitter（cv_folds 钳制到组数，防 n_splits>n_groups）
    cv_search = cv_folds
    if groups is not None:
        n_groups = len(np.unique(np.asarray(groups)))
        n_splits = max(2, min(int(cv_folds), int(n_groups)))
        if n_splits < int(cv_folds):
            LOGGER.warning('%s 组感知搜索折数由 %d 降至 %d（组数不足）',
                           spec.name_cn, cv_folds, n_splits)
        cv_search = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                         random_state=random_state)

    try:
        if enable_hyperopt:
            try:
                base_model = spec.model_cls(**base_kwargs)
                best_model = _interruptible_grid_search(
                    _StubPredictor(stop_check), base_model, spec.param_grid,
                    X, y, cv=cv_search, sample_weight=sample_weight,
                    groups=groups
                )
                if stop_check and stop_check():
                    return None
                return best_model
            except (ValueError, RuntimeError, OSError, TypeError) as e:
                LOGGER.warning('%s超参数搜索失败，回退到默认参数: %s', spec.name_cn, e)

        # 使用默认参数训练（sample_weight 仅在非 None 时传——
        # 新版 sklearn Pipeline.fit 的参数路由不接受 sample_weight=None
        # 关键字；树模型 fit(X,y) 与 fit(X,y,sample_weight=None) 等价）
        model = spec.model_cls(**default_kwargs)
        if sample_weight is not None:
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)
        return model

    except Exception as e:
        if spec.catch_exceptions:
            LOGGER.warning('%s训练失败，跳过: %s', spec.name_cn, e)
            return None
        raise


class _StubPredictor:
    """网格搜索用的轻量stub，只实现_stop_training检查"""
    def __init__(self, stop_check):
        self._stop_fn = stop_check or (lambda: False)

    @property
    def _stop_training(self):
        return self._stop_fn()


def _evaluate_model_cv(model, X, y, cv, groups=None, n_bootstrap=500,
                       random_state=0, X_cv_raw=None) -> Dict[str, Any]:
    """使用交叉验证评估模型性能

    无组路径（groups=None）：逐折 AUROC/AUPRC 均值——存量语义原样
    保留（存量 checkpoint / 训练档案可比性）。
    组路径（groups 提供时）：单次池化 OOF（cross_val_predict）→
    全量单次 AUROC/AUPRC 点估计 + 户级 cluster bootstrap 95% CI
    （对齐实验脚本判决性协议：validation/real_data_pi.py）。
    池化 OOF 同时规避个别折单类导致 roc_auc 抛 ValueError 被
    吞成 0.0 的退化路径——只要全量 y 双类（上游已保证）即可计算。

    X_cv_raw（缺陷1口径统一，2026-09-16）：v4 extras 未填补矩阵
    （NaN 保留）。提供时 CV 指标改在 X_cv_raw 上评估
    Pipeline(SimpleImputer(median, keep_empty_features) → model)——
    填补器在训练折拟合（折内中位），与 F2 评估脚本 _fold_fill 判决
    口径一致（全 NaN 列填 0 同语义）；最终模型仍在填补后的 X 上
    拟合，部署工件与 fill_medians 落盘语义不变。None 时存量语义。

    返回：
        无组路径：{'AUROC': float, 'AUPRC': float}
        组路径：额外含 'AUROC_ci95'/'AUPRC_ci95'（[lo,hi] 或 None）
        与 'n_bootstrap_effective'（非退化重采样次数）
    """
    eval_model, eval_X = model, X
    if X_cv_raw is not None:
        try:
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            imputer = SimpleImputer(
                strategy='median', keep_empty_features=True)
            # P1-a（特征名告警清理，2026-09-20）：imputer 默认输出 ndarray，
            # LGBM sklearn 包装器在 ndarray 拟合时写入自动特征名（Column_N），
            # 此后每折 numpy 打分触发一条 "X does not have valid feature
            # names" UserWarning（实测 5 折 × 6 模型 × 7 队列刷屏数百条，
            # 曾只能靠 run 脚本 filterwarnings 压制）。set_output(transform=
            # 'pandas') 让 imputer 按拟合时记录的列名输出 DataFrame → 模型
            # 拟合与预测均带名，告警消除；cross_val_score/cross_val_predict
            # 内部 clone 保留该配置（实测 sklearn 1.6.1），数值逐位不变。
            # pandas 不可用或旧版 sklearn 无 set_output 时原样回退（告警
            # 仍在但结果不受影响）。
            if PANDAS_AVAILABLE:
                try:
                    imputer = imputer.set_output(transform='pandas')
                except (AttributeError, TypeError):
                    pass
            eval_model = Pipeline([
                ('impute', imputer),
                ('model', model)])
            eval_X = X_cv_raw
        except ImportError:  # pragma: no cover - sklearn 缺失时不可达
            LOGGER.warning('SimpleImputer 不可用 → CV 指标回退全数据填补口径')
            eval_model, eval_X = model, X
    if groups is not None:
        try:
            oof = cross_val_predict(eval_model, eval_X, y, groups=groups,
                                    cv=cv, method='predict_proba',
                                    n_jobs=-1)[:, 1]
            from .validation import cluster_bootstrap_metric_ci
            # 两次调用同种子 → 同一重采样序列，AUROC/AUPRC CI 配对
            auroc_b = cluster_bootstrap_metric_ci(
                oof, y, groups, metric='auroc',
                n_bootstrap=n_bootstrap, seed=random_state)
            auprc_b = cluster_bootstrap_metric_ci(
                oof, y, groups, metric='auprc',
                n_bootstrap=n_bootstrap, seed=random_state)
            return {
                'AUROC': (float(auroc_b['point'])
                          if auroc_b['point'] is not None else 0.0),
                'AUPRC': (float(auprc_b['point'])
                          if auprc_b['point'] is not None else 0.0),
                'AUROC_ci95': auroc_b['ci95'],
                'AUPRC_ci95': auprc_b['ci95'],
                'n_bootstrap_effective': int(auroc_b['n_effective']),
            }
        except Exception as e:
            LOGGER.warning('组感知交叉验证评估失败: %s', e)
            return {'AUROC': 0.0, 'AUPRC': 0.0}
    try:
        auroc = cross_val_score(eval_model, eval_X, y, cv=cv, scoring='roc_auc').mean()
        auprc = cross_val_score(eval_model, eval_X, y, cv=cv, scoring='average_precision').mean()
        return {'AUROC': float(auroc), 'AUPRC': float(auprc)}
    except Exception as e:
        LOGGER.warning('交叉验证评估失败: %s', e)
        return {'AUROC': 0.0, 'AUPRC': 0.0}


def _get_git_hash() -> str:
    """获取当前代码版本的 git commit hash，无 git 环境返回 'unknown'"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return 'unknown'


def _summarize_hyperparams(model) -> Dict[str, Any]:
    """提取模型关键超参数摘要（用于 model_performance 元信息）"""
    summary: Dict[str, Any] = {}
    try:
        params = model.get_params() if hasattr(model, 'get_params') else {}
        # 提取常见关键超参数，避免保存全部参数（包含对象引用会导致序列化失败）
        for key in ('n_estimators', 'max_depth', 'learning_rate', 'num_leaves',
                    'n_iterations', 'iterations', 'C', 'lr__C', 'max_features',
                    'min_samples_split', 'min_samples_leaf', 'subsample',
                    'colsample_bytree', 'reg_alpha', 'reg_lambda',
                    'random_state'):
            if key in params:
                val = params[key]
                # 只保存基本类型（避免对象引用）
                if isinstance(val, (int, float, str, bool, type(None))):
                    summary[key] = val
    except Exception as e:
        LOGGER.debug("提取模型超参数摘要失败: %s", e)
    return summary


# 类不平衡处理方式档案（问题4训练侧）：各模型的加权机制单一真值源，
# 与 _train_single_model 中的注入逻辑一一对应。
# 文献：He H, Garcia EA. Learning from Imbalanced Data. IEEE TKDE 2009
# 21(9):1263-1284（代价敏感学习/类加权 vs 重采样）
IMBALANCE_HANDLING_DESC = {
    'random_forest': "class_weight='balanced'（注册表默认参数）",
    'gradient_boosting': 'balanced sample_weight（sklearn GB 不支持 class_weight，'
                          '按 scale_pos_weight 加权正样本）',
    'xgboost': 'scale_pos_weight=n_neg/n_pos（构造参数注入，网格搜索路径同样生效）',
    'lightgbm': "class_weight='balanced'（注册表默认参数）",
    'catboost': "auto_class_weights='Balanced'（注册表默认参数）",
    'logistic': "lr__class_weight='balanced'（注册表默认参数，标准化管道内）",
}


def _attach_calibration_and_decision_reference(predictor, X, y, random_state):
    """训练完成后接入真实数据校准 + 决策参考表（问题4校准侧）

    1. calibrate_models_on_data：分层对半切分的泄漏无关校准，
       method='auto'（阳性 < 200 → Platt sigmoid，≥ 200 → isotonic），
       每模型记录 Brier/AUROC 前后对照；
    2. 用校准评估半区概率构建 decision_reference（排序+截断点+
       双口径 PPV），写入 model_performance[key]['decision_reference']。

    任何失败仅记录警告，不阻断训练主流程（训练结果已可用）。
    """
    try:
        from .calibration import calibrate_models_on_data
        from ...core.ppv import build_decision_reference
    except ImportError as e:  # pragma: no cover - 依赖缺失防御
        LOGGER.warning('校准模块不可用，跳过真实数据校准: %s', e)
        return

    try:
        cal_ok = calibrate_models_on_data(
            predictor, X, y, method='auto', random_state=random_state)
    except Exception as e:
        LOGGER.warning('真实数据校准异常（不影响训练结果）: %s', e)
        return

    if not cal_ok:
        LOGGER.warning('真实数据校准未完成（不影响训练结果）')
        return

    # 基于泄漏无关评估半区概率构建决策参考表（覆盖风险最高前 10%）。
    # 第三口径（缺陷1，2026-09-17）：训练队列实际阳性率——固定参照
    # 口径（密接 2.85%）在低患病率队列上高估 PPV 4-5 倍（kenya 实际
    # 0.53%），与训练队列同人群的站点读表以 cohort_local 为准。
    eval_data = getattr(predictor, 'calibration_eval', None) or {}
    y_arr = np.asarray(y, dtype=float)
    local_prev = (float(y_arr.mean())
                  if y_arr.size and 0.0 < y_arr.mean() < 1.0 else None)
    v4_cohort = getattr(predictor, 'v4_cohort', None)
    local_label = (f'{v4_cohort} 训练样本阳性率' if v4_cohort
                   else '训练样本阳性率')
    for model_name, ev in eval_data.items():
        try:
            ref = build_decision_reference(
                ev['p_after'], ev['y_test'], top_fraction=0.1,
                local_prevalence=local_prev, local_label=local_label)
            perf = predictor.model_performance.setdefault(model_name, {})
            perf['decision_reference'] = ref
        except (ValueError, KeyError, TypeError) as e:
            LOGGER.warning('模型 %s 决策参考表构建失败: %s', model_name, e)


def _select_feature_columns(predictor, X, wanted):
    """按列名把特征矩阵筛选到 wanted 列（问题3：v2 精简特征集）。

    支持三类输入：DataFrame（按列名选）、22 列 numpy（按 ALL_FEATURE_NAMES
    位置选）、已按 wanted 顺序的 numpy（直接包装）。不可识别时返回 None
    （调用方应显式失败——静默降级会让 v2 档案贴错特征空间标签）。
    """
    if PANDAS_AVAILABLE and isinstance(X, pd.DataFrame):
        if all(c in X.columns for c in wanted):
            return X[list(wanted)]
        return None
    arr = np.asarray(X)
    all_names = list(getattr(predictor, 'ALL_FEATURE_NAMES', []))
    if arr.ndim == 2 and all_names and arr.shape[1] == len(all_names):
        try:
            pos = [all_names.index(c) for c in wanted]
        except ValueError:
            return None
        if PANDAS_AVAILABLE:
            return pd.DataFrame(arr[:, pos], columns=list(wanted))
        return arr[:, pos]
    if arr.ndim == 2 and arr.shape[1] == len(wanted):
        if PANDAS_AVAILABLE:
            return pd.DataFrame(arr, columns=list(wanted))
        return arr
    return None


def _select_feature_columns_v3(predictor, X, wanted, network_cols):
    """v3 谱系感知选列（15 个体 + 11 网络特征化，第四轮 P3）。

    与 v2 的防错档语义（不可识别 → None 显式失败）不同，v3 增加
    **谱系降级**路径：网络列的源字段（图邻接的边类型/时间窗分解）
    仅合成验证层存在，部署谱系（Kenya/HomeACF/部署 GUI）缺列时
    补零死列而非失败——这是数据谱系的已知属性，不是训练输入错误
    （降级后判别力退化为 v2，Kenya 复验检验的正是这一点）。

    可识别输入（与 _select_feature_columns 对齐 + 网络列）：
      1. DataFrame 含全部 26 列 → 全量选；
      2. DataFrame 含 v2 15 列（网络列缺失）→ 15 列按名选 + 网络列补零；
      3. 22 列 numpy（v1 全名矩阵）→ v2 列按位置选 + 网络列补零；
      4. 26 列 numpy（v3 顺序）→ 直接按名包装。
    返回 (X_selected [26 列带名], dead_network_cols)；个体列不可识别
    时返回 (None, [])——调用方显式失败（v2 同语义：防贴错标签）。
    """
    if PANDAS_AVAILABLE and isinstance(X, pd.DataFrame):
        ind_cols = [c for c in wanted if c not in network_cols]
        if all(c in X.columns for c in ind_cols):
            frame = X[ind_cols].copy()
            dead = [c for c in network_cols if c not in X.columns]
            for c in network_cols:
                frame[c] = 0.0 if c in dead else X[c].values
            return frame[list(wanted)], dead
        return None, []
    arr = np.asarray(X)
    all_names = list(getattr(predictor, 'ALL_FEATURE_NAMES', []))
    if arr.ndim == 2 and all_names and arr.shape[1] == len(all_names):
        try:
            pos = [all_names.index(c) for c in wanted
                   if c not in network_cols]
        except ValueError:
            return None, []
        if PANDAS_AVAILABLE:
            frame = pd.DataFrame(arr[:, pos],
                                 columns=[c for c in wanted
                                          if c not in network_cols])
            for c in network_cols:
                frame[c] = 0.0
            return frame[list(wanted)], list(network_cols)
        return np.concatenate(
            [arr[:, pos], np.zeros((arr.shape[0], len(network_cols)))],
            axis=1), list(network_cols)
    if arr.ndim == 2 and arr.shape[1] == len(wanted):
        if PANDAS_AVAILABLE:
            return pd.DataFrame(arr, columns=list(wanted)), []
        return arr, []
    return None, []


def _train_all_registered_models(predictor, X, y, *, random_state: int = 42,
                                   enable_hyperopt: bool = False,
                                   data_source_label: str = 'synthetic',
                                   model_keys: Optional[List[str]] = None,
                                   cv_folds: int = 5,
                                   n_real_samples: Optional[int] = None,
                                   n_synth_samples: Optional[int] = None,
                                   enable_calibration: bool = True,
                                   feature_set: str = 'v1',
                                   label_generation: Optional[str] = None,
                                   groups=None,
                                   group_column_name: Optional[str] = None,
                                   X_cv_raw=None) -> bool:
    """训练注册表中所有可用模型（统一训练内核）

    参数：
        predictor: MLRiskPredictor 实例（或兼容对象）
        X: 特征矩阵 (n_samples, n_features)
        y: 标签数组 (n_samples,)
        random_state: 随机种子
        enable_hyperopt: 是否启用超参数调优
        data_source_label: 数据来源标签（'synthetic'/'real'）
        model_keys: 指定要训练的模型列表，None表示按DEFAULT_MODEL_ORDER训练全部可用模型
        cv_folds: 评估用交叉验证折数
        groups (array-like|None): 样本所属组（household/community id）。
            提供时全链路（超参搜索 + 性能评估）切换组感知协议：
            StratifiedGroupKFold（折间组不相交）+ 池化 OOF 点估计 +
            户级 cluster bootstrap 95% CI——对齐实验脚本判决性协议
            （validation/real_data_pi.py）。TREATS group_id / Peru
            family_id 的户成员不再同时进训练与验证折。
        group_column_name (str|None): groups 的来源列名（档案披露用）
        X_cv_raw (array-like|None): CV 指标专用未填补矩阵（缺陷1口径
            统一，v4 extras NaN 保留版）。提供时 _evaluate_model_cv 用
            Pipeline(SimpleImputer(median) → model) 在其上评估——折内
            中位填补（训练折拟合填补器），与 F2 评估脚本判决口径一致；
            最终模型仍在填补后的 X 上拟合（部署工件不变）。
            None（v1/v2/v3 或合成路径）时存量语义原样。
        n_synth_samples: 合成样本数量（None时根据data_source_label推断）
        enable_calibration: 训练完成后是否自动接入真实数据校准
                            （Platt/isotonic 自动选择）+ 决策参考表
                            （问题4校准侧；关闭时仅跳过校准，不影响训练）
        feature_set: 特征集版本（问题3特征审计）。'v1'（默认，22 维）/
            'v2'（15 维精简集，SELECTED_FEATURES_V2——审计结论：
            移除可精确重构列与暴露时序冗余）/ 'v3'（26 维 = v2 个体
            15 列 + 11 网络特征化列，SELECTED_FEATURES_V3——第四轮
            P3 定版：组合机制消融结论特征化落地，合成 +0.06~0.09
            次可加；网络列缺源字段的部署谱系自动补零降级，判别力
            退化为 v2）。预测端由 evaluation._model_input_frame 按模型
            自身特征名选列，三代特征集可共存于同一部署环境。
        label_generation: 标签机制代际（P4，2026-08-24）。None（默认，
            旧调用路径语义）表示未声明；场景 mech 代传
            'v3-host-pathway'、clinical 代传 'clinical-rule-v1'。
            写入每模型 entry 与 predictor（save_model 随 checkpoint
            落盘，load_model 按 expected_label_generation 校验，
            防止标签换代后静默加载错代模型——best.json 跨代比较
            失效的根因即此）。

    返回：
        bool: 是否至少成功训练了一个模型
    """
    if not SKLEARN_AVAILABLE:
        return False

    if feature_set not in ('v1', 'v2', 'v3', 'v4'):
        LOGGER.error("未知 feature_set: %r（支持 'v1'/'v2'/'v3'/'v4'）",
                     feature_set)
        return False

    dead_network_cols = []
    try:
        # 特征集筛选（问题3）：v2 = 审计后的 15 维精简集。
        # 不可识别的输入矩阵 → 显式失败（防止 v2 档案贴错特征空间标签）。
        if feature_set == 'v2':
            from .feature_audit import SELECTED_FEATURES_V2
            X_selected = _select_feature_columns(
                predictor, X, SELECTED_FEATURES_V2)
            if X_selected is None:
                LOGGER.error(
                    "feature_set='v2' 需要可按列名识别的特征矩阵"
                    "（22 列 numpy / 含全部 22 特征名的 DataFrame / "
                    "15 列 v2 顺序矩阵），当前输入无法识别")
                return False
            X = X_selected
        elif feature_set == 'v3':
            # 第四轮 P3：v3 = v2 个体 + 11 网络特征化（合成 +0.06~0.09
            # 次可加）。谱系感知——网络列缺源字段时补零降级（dead 列
            # 审计随 entry 归档），个体列不可识别仍显式失败。
            from .feature_audit import (NETWORK_FEATURE_NAMES_V3,
                                        SELECTED_FEATURES_V3)
            X_selected, dead_network_cols = _select_feature_columns_v3(
                predictor, X, SELECTED_FEATURES_V3,
                list(NETWORK_FEATURE_NAMES_V3))
            if X_selected is None:
                LOGGER.error(
                    "feature_set='v3' 需要可按列名识别的特征矩阵"
                    "（22 列 numpy / 含 v2 15 特征名的 DataFrame / "
                    "26 列 v3 顺序矩阵），当前输入无法识别")
                return False
            X = X_selected
            if dead_network_cols:
                LOGGER.warning(
                    "feature_set='v3' 谱系降级：%d/%d 网络列缺源字段"
                    "已补零（%s...），判别力退化为 v2 口径",
                    len(dead_network_cols), len(NETWORK_FEATURE_NAMES_V3),
                    dead_network_cols[0])
        elif feature_set == 'v4':
            # v4（族感知，P1 2026-09-16）：X 已由 train_from_real_data
            # 构造为 22 维 + per-cohort extras（哨兵清洗 + 缺失指示器 +
            # 全数据中位数填补，spec 单一真值源见 ml/cohort_features.py），
            # 不做子集选择。特征名清单挂 predictor.v4_feature_names——
            # 特征名对齐段与 persistence.feature_contract（随 checkpoint
            # 落盘）据此记录真实列契约。
            _v4_names = list(getattr(predictor, 'v4_feature_names', []) or [])
            if not _v4_names:
                LOGGER.error(
                    "feature_set='v4' 需要 predictor.v4_feature_names"
                    "（由 train_from_real_data 的 v4 构造层写入）；"
                    "直接调用统一内核的编排层应自行构造并挂载")
                return False
        predictor.feature_set = feature_set
        # P4（标签代际）：挂到 predictor，save_model 随 checkpoint 落盘
        predictor.label_generation = label_generation
        # 特征名对齐（2026-08-24）：LightGBM sklearn 包装器在 numpy 拟合时
        # 也会写入自动特征名（Column_N），cross_val_score 内部的 numpy 打分
        # 因此每折触发一条 "X does not have valid feature names" UserWarning
        # （实测 5 折 × 2 指标 = 每个模型 10 条）。统一把 X 转成带
        # ALL_FEATURE_NAMES 列名的 DataFrame：fit 与 CV 特征名一致 → 警告
        # 消除，新 checkpoint 记录真实特征名（利于解释），数值结果不变；
        # 维度不匹配或 pandas 不可用时保持 numpy 原样（预测端由
        # evaluation._aligned_feature_frame 按模型自身记录的名字兜底）。
        if PANDAS_AVAILABLE and not isinstance(X, pd.DataFrame) \
                and hasattr(predictor, 'ALL_FEATURE_NAMES'):
            _names = list(predictor.ALL_FEATURE_NAMES)
            if feature_set == 'v4':
                _names = list(getattr(predictor, 'v4_feature_names', [])
                              or _names)
            _X_arr = np.asarray(X)
            if _X_arr.ndim == 2 and _X_arr.shape[1] == len(_names):
                X = pd.DataFrame(_X_arr, columns=_names)
            # X_cv_raw 同名列包装（折内填补 Pipeline 特征名一致；numpy
            # 也能跑但会触发 LightGBM 特征名警告）
            if X_cv_raw is not None and not isinstance(X_cv_raw,
                                                       pd.DataFrame):
                _raw_arr = np.asarray(X_cv_raw)
                if _raw_arr.ndim == 2 and _raw_arr.shape[1] == len(_names):
                    X_cv_raw = pd.DataFrame(_raw_arr, columns=_names)

        # 组感知 CV（任务1，2026-09-14，对齐实验脚本协议）：groups 提供
        # 时折间组不相交（TREATS 社区 / Peru 家庭不跨折），评估走池化
        # OOF + 户级 cluster bootstrap（见 _evaluate_model_cv）；组数不足
        # 时钳制折数（StratifiedGroupKFold 要求 n_splits ≤ n_groups）。
        # groups=None 时维持原 StratifiedKFold（存量语义完全不变）。
        if groups is not None:
            n_groups = len(np.unique(np.asarray(groups)))
            n_splits = max(2, min(int(cv_folds), int(n_groups)))
            if n_splits < int(cv_folds):
                LOGGER.warning('组感知 CV 折数由 %d 降至 %d（组数 %d 不足）',
                               cv_folds, n_splits, n_groups)
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=random_state)
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                 random_state=random_state)

        is_synthetic = data_source_label == 'synthetic'
        ds_warning = ('基于合成数据训练，AUC反映模型复现评分引擎的能力，'
                     '非真实流行病学泛化能力') if is_synthetic else None

        keys = model_keys or DEFAULT_MODEL_ORDER
        trained_count = 0

        # 训练元信息（所有模型共享）
        training_start = datetime.datetime.now().isoformat(timespec='seconds')
        git_hash = _get_git_hash()
        n_total = int(len(y))
        n_positive = int(np.sum(y == 1)) if y is not None else 0
        n_negative = n_total - n_positive
        # 真实/合成样本计数：优先用调用方传入的精确计数，否则根据 data_source_label 推断
        if n_real_samples is not None:
            n_real = int(n_real_samples)
        else:
            n_real = n_total if not is_synthetic else 0
        if n_synth_samples is not None:
            n_synthetic = int(n_synth_samples)
        else:
            n_synthetic = n_total if is_synthetic else 0

        # 清空已有模型
        predictor.models = getattr(predictor, 'models', {})
        predictor.model_performance = getattr(predictor, 'model_performance', {})
        predictor.models.clear()
        predictor.model_performance.clear()

        # 统一的停止检查（在循环外定义，避免闭包问题）
        def _stop_check():
            return getattr(predictor, '_stop_training', False)

        for key in keys:
            spec = MODEL_REGISTRY.get(key)
            if spec is None:
                continue
            if not spec.available:
                continue

            model = _train_single_model(
                spec, X, y,
                enable_hyperopt=enable_hyperopt,
                random_state=random_state,
                cv_folds=3,
                stop_check=_stop_check,
                groups=groups,
            )

            if _stop_check():
                LOGGER.info('训练被中断，已完成 %d 个模型', trained_count)
                return trained_count > 0

            if model is None:
                continue

            metrics = _evaluate_model_cv(model, X, y, cv, groups=groups,
                                         n_bootstrap=500,
                                         random_state=random_state,
                                         X_cv_raw=X_cv_raw)
            predictor.models[key] = model
            entry = {
                'AUROC': metrics['AUROC'],
                'AUPRC': metrics['AUPRC'],
                # 组感知协议档案（任务1）：CI 键仅组路径存在（只增不改，
                # AUROC/AUPRC 点估计键语义不变，存量消费方全部兼容）
                'AUROC_ci95': metrics.get('AUROC_ci95'),
                'AUPRC_ci95': metrics.get('AUPRC_ci95'),
                'n_bootstrap_effective': metrics.get('n_bootstrap_effective'),
                'name': spec.name_cn,
                'name_en': spec.name_en,
                'data_source': data_source_label,
                'model_version': '1.0.0',
                'training_started_at': training_start,
                'training_completed_at': datetime.datetime.now().isoformat(timespec='seconds'),
                'git_commit': git_hash,
                'n_total_samples': n_total,
                'n_positive': n_positive,
                'n_negative': n_negative,
                'n_real_samples': n_real,
                'n_synthetic_samples': n_synthetic,
                'cv_folds': cv_folds,
                # 组感知协议披露（任务1）：协议类型 / 组列名 / 组数——
                # 随 checkpoint 落盘，档案可审计"这份 AUROC 是否组感知"
                'cv_protocol': ('stratified_group_kfold' if groups is not None
                                else 'stratified_kfold'),
                'group_column': group_column_name,
                'n_groups': (int(len(np.unique(np.asarray(groups))))
                             if groups is not None else None),
                # 问题3（特征审计）：特征集版本与实际维度（v1=22 /
                # v2=15 / v3=26 / v4=22+per-cohort extras，spec 见
                # ml/cohort_features.py；v3 谱系降级时记录 dead 网络列审计）
                'feature_set': feature_set,
                'dead_network_columns': list(dead_network_cols),
                # P4（标签代际）：mech 'v3-host-pathway' / clinical
                # 'clinical-rule-v1' / None 未声明（旧路径）
                'label_generation': label_generation,
                'n_features': int(np.asarray(X).shape[1]) if np.asarray(X).ndim == 2 else None,
                'hyperparameters': _summarize_hyperparams(model),
                # 问题4（训练侧）：类不平衡处理档案（与 _train_single_model
                # 的注入逻辑一一对应，见 IMBALANCE_HANDLING_DESC）
                'imbalance_handling': {
                    'method': IMBALANCE_HANDLING_DESC.get(
                        key, '未记录（未知模型键）'),
                    'scale_pos_weight': round(max(
                        n_negative / max(n_positive, 1), 1.0), 6),
                },
            }
            if ds_warning:
                entry['warning'] = ds_warning
            # v4（P1）：队列名随 entry 落盘——checkpoint 加载端可审计
            # extras 属于哪个队列（列集 per-cohort，跨队列不可比）
            if feature_set == 'v4':
                entry['v4_cohort'] = getattr(predictor, 'v4_cohort', None)
                # 缺陷1口径披露：CV 指标折内中位填补（与 F2 判决口径
                # 一致）或回退口径（raw 不可用时）
                entry['cv_imputation'] = ('fold_internal_median'
                                          if X_cv_raw is not None
                                          else 'median_full')
            predictor.model_performance[key] = entry
            trained_count += 1
            LOGGER.info('%s 训练完成: AUROC=%.4f, AUPRC=%.4f',
                       spec.name_cn, metrics['AUROC'], metrics['AUPRC'])

        if trained_count == 0:
            LOGGER.warning('没有可用模型被训练')
            return False

        _init_shap_explainer_with_fallback(predictor)
        predictor.is_trained = True

        # 问题4（校准侧）：训练完成自动接入真实数据校准（Platt/isotonic
        # 自动选择）+ 决策参考表（排序+截断点+双口径 PPV）；失败不阻断
        if enable_calibration:
            _attach_calibration_and_decision_reference(
                predictor, X, y, random_state)

        return True

    except Exception as e:
        LOGGER.warning("ML模型训练失败: %s", e, exc_info=True)
        return False


def _train_models_quick(model_keys: List[str], X_train, y_train, random_state: int) -> Dict[str, Any]:
    """快速训练模型（不进行交叉验证和超参数搜索，用于特殊模式）

    类不平衡加权与主训练路径（_train_single_model）共用
    _class_imbalance_fit_kwargs，保证评估/对比路径训出的模型
    与主模型在类不平衡处理上可比。
    """
    models = {}
    for key in model_keys:
        spec = MODEL_REGISTRY.get(key)
        if spec is None or not spec.available:
            continue
        try:
            kwargs = {**spec.default_params, spec.random_state_param: random_state}
            # 与主训练路径同一加权实现（XGBoost scale_pos_weight /
            # GB balanced sample_weight；RF/LGBM/CatBoost 走注册表默认）
            imbalance_kwargs, sample_weight = _class_imbalance_fit_kwargs(key, y_train)
            kwargs.update(imbalance_kwargs)
            model = spec.model_cls(**kwargs)
            if sample_weight is not None:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)
            models[key] = model
        except Exception as e:
            LOGGER.warning("快速训练 %s 失败: %s", key, e)
    return models


# ============================================================================
# 公开入口函数（保留原有API，内部全部委托给统一内核）
# ============================================================================

def _train_with_fallback(predictor, model_cls, default_params, param_grid, X, y,
                          enable_hyperopt, random_state, cv=3, random_state_param='random_state'):
    """训练单个模型（超参数搜索或默认参数）——保留兼容包装

    注意：新代码请直接使用 _train_single_model(spec, ...)
    """
    # 动态构造临时ModelSpec
    spec = ModelSpec(
        key='_tmp',
        name_cn='_tmp',
        name_en='_tmp',
        model_cls=model_cls,
        default_params=default_params,
        param_grid=param_grid,
        random_state_param=random_state_param,
        available=True,
    )
    return _train_single_model(
        spec, X, y,
        enable_hyperopt=enable_hyperopt,
        random_state=random_state,
        cv_folds=cv,
        stop_check=lambda: getattr(predictor, '_stop_training', False),
    )


def train_models(predictor, n_samples=2000, random_state=42, enable_hyperopt=False, data_source=None,
                 feature_set='v1'):
    """训练所有机器学习模型

    注意：树模型（随机森林、梯度提升、XGBoost）对特征尺度不敏感，
    因此不使用StandardScaler标准化。这样SHAP值可直接解释原始特征，
    便于临床解读。

    参数：
        n_samples (int): 合成训练数据样本量，默认2000。
            当 data_source 提供时，若真实数据不足 n_samples，自动用合成数据增强。
        random_state (int): 随机种子，确保可复现性
        enable_hyperopt (bool): 是否启用超参数调优，默认False
        data_source (str|None): 真实数据文件路径。支持 .csv/.json/.jsonl/.xlsx/.xls/.parquet。
            当传入时，优先通过 simulator.load_or_generate 加载真实数据；
            当为 None 时，使用合成数据（默认行为）。
        feature_set (str): 'v1'（默认，22 维）/ 'v2'（15 维精简集，
            问题3特征审计结论；预测端自动按模型特征名选列）

    返回：
        bool: 训练是否成功
    """
    if not SKLEARN_AVAILABLE:
        return False

    try:
        # 重置停止标志
        predictor.reset_stop_flag()

        if data_source and os.path.exists(data_source):
            # 真实数据路径：通过 simulator.load_or_generate 加载
            LOGGER.info("从真实数据源训练: %s", data_source)
            try:
                from .simulator import ScreeningDataSimulator
                simulator = ScreeningDataSimulator(random_state=random_state)
                records, meta = simulator.load_or_generate(
                    filepath=data_source, n_samples=n_samples, include_labels=True
                )

                # 标记数据来源
                n_real = sum(1 for r in records if r.get('_data_source') == 'real')
                n_synthetic = len(records) - n_real
                if n_real > 0:
                    predictor.use_real_data = True
                    LOGGER.info(
                        "真实数据加载完成: %d 条真实 + %d 条合成增强",
                        n_real, n_synthetic
                    )
                else:
                    predictor.use_real_data = False
                    LOGGER.info("未找到真实数据，使用合成数据")

                # 提取特征
                X, y = predictor._extract_features_from_records(records)
                predictor.training_sample_count = len(records)

                data_source_label = 'real' if n_real > 0 else 'synthetic'
                return _train_all_registered_models(
                    predictor, X, y,
                    random_state=random_state,
                    enable_hyperopt=enable_hyperopt,
                    data_source_label=data_source_label,
                    n_real_samples=n_real,
                    n_synth_samples=n_synthetic,
                    feature_set=feature_set,
                )

            except ImportError as e:
                LOGGER.warning(
                    "simulator 不可用 (%s)，回退到合成数据训练", e
                )
            except Exception as e:
                LOGGER.warning(
                    "真实数据加载失败 (%s)，回退到合成数据训练", e
                )

        # 默认路径：合成数据（地基审计 2026-09-05——两重已知局限，
        # 每次合成训练显式告警，防"合成 AUC 0.84"被误读为真实能力）：
        #   1. 循环标签：合成 y 的 log-odds 系数与 ScoringEngine 评分
        #      逻辑同源（preprocessing._generate_synthetic_training_data
        #      docstring 警告），AUC≈0.84 只测复现评分引擎的能力；
        #   2. 相对排序不外推：PACTS/HomeACF 真实队列复验
        #      （real_data_joint_{pacts,homeacf}_20260904.json，20 种子
        #      户级 CV）显示合成世界最优结构（joint/GNN 机制模型）
        #      在真实数据上排序反转（PACTS joint−best = −0.046
        #      CI[−0.090,−0.013]）。合成模型输出禁止作为真实人群
        #      性能证据引用。
        LOGGER.warning(
            "合成数据训练（data_source 未提供）：标签与评分引擎同源"
            "（循环论证，AUC≈0.84 只测复现能力）+ 真实队列排序反转实证"
            "——本模型输出不可作为真实人群性能证据；真实部署前需本地"
            "真实数据重训（train_models(data_source=...)）"
        )
        X, y = predictor._generate_synthetic_training_data(n_samples, random_state)
        predictor.training_sample_count = n_samples
        predictor.use_real_data = False

        return _train_all_registered_models(
            predictor, X, y,
            random_state=random_state,
            enable_hyperopt=enable_hyperopt,
            data_source_label='synthetic',
            feature_set=feature_set,
        )

    except Exception as e:
        LOGGER.warning("ML模型训练失败: %s", e, exc_info=True)
        return False


def _train_models_from_arrays(predictor, X, y, random_state=42, enable_hyperopt=False,
                               data_source_label='synthetic'):
    """从特征矩阵训练所有模型（train_models 和 train_from_arrays 的共享内核）"""
    return _train_all_registered_models(
        predictor, X, y,
        random_state=random_state,
        enable_hyperopt=enable_hyperopt,
        data_source_label=data_source_label,
    )


def train_from_arrays(predictor, X, y, random_state=42, enable_hyperopt=False,
                      feature_set='v1'):
    """从特征矩阵和标签数组训练ML模型

    Args:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签数组 (n_samples,)
        random_state: 随机种子
        enable_hyperopt: 是否启用超参数调优
        feature_set: 'v1'（22 维，默认）/ 'v2'（15 维精简集，问题3）

    Returns:
        bool: 训练是否成功
    """
    if not SKLEARN_AVAILABLE:
        return False

    try:
        predictor.reset_stop_flag()

        if len(X) < 10:
            LOGGER.warning("样本量过少，至少需要10个样本")
            return False

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            LOGGER.warning("标签只有一种类型，无法训练")
            return False

        predictor.training_sample_count = len(X)
        predictor.use_real_data = True

        result = _train_all_registered_models(
            predictor, X, y,
            random_state=random_state,
            enable_hyperopt=enable_hyperopt,
            data_source_label='real',
            feature_set=feature_set,
        )

        if result:
            predictor._cached_training_data = None
            predictor._cached_training_params = None
        return result

    except Exception as e:
        LOGGER.warning("ML模型训练失败: %s", e, exc_info=True)
        return False


def _train_models_for_mode(predictor, X_train, y_train, model_keys, random_state):
    """快速训练模式专用模型（不持久化到 predictor.models，不做交叉验证）"""
    return _train_models_quick(model_keys, X_train, y_train, random_state)


def ensure_interaction_features(df, interaction_feature_names):
    """逐列缺失才补的交互特征计算（公共助手，2026-09-14 抽出）。

    从 train_from_real_data 内联段抽出的单一真值源：CSV 只存 13 基础
    列时自动补齐 9 维交互特征。data/train_with_public_data.py 与
    data/run_peru_mdr_sop_temporal.py 尚存同逻辑副本（本次不动，
    范围控制），新代码（多队列池化 / 迁移脚本）一律引用本函数。

    参数：
        df: DataFrame（就地修改）
        interaction_feature_names: 需要保证存在的交互列名列表

    返回：
        (df, missing_before)：补列前的缺失列表（供调用方记录日志）
    """
    missing = [f for f in interaction_feature_names if f not in df.columns]
    if not missing:
        return df, []

    if 'past_illness_type' not in df.columns:
        df['past_illness_type'] = 'none'
        df.loc[df['past_illness'] == 1, 'past_illness_type'] = 'other'

    hiv = (df['past_illness_type'].isin(['hiv', 'HIV'])).astype(float)
    diabetes = (df['past_illness_type'].isin(['diabetes', '糖尿病'])).astype(float)
    immunosupp = (df['past_illness_type'].isin(['immunosuppressants', '免疫抑制'])).astype(float)
    immunosuppression_score = hiv * 2.0 + diabetes * 1.0 + immunosupp * 1.5 + df['past_illness'].astype(float) * 0.5

    if 'age_immuno' not in df.columns:
        df['age_immuno'] = df['age'].astype(float) * immunosuppression_score
    if 'dm_tb_synergy' not in df.columns:
        df['dm_tb_synergy'] = diabetes * df['has_tb'].astype(float)
    if 'age_bcg_decay' not in df.columns:
        df['age_bcg_decay'] = df['age'].astype(float) * (1 - df['bcg_vaccine'].astype(float))
    if 'symptom_delay' not in df.columns:
        delay_val = df.get('delay_days', df.get('ftd', 0))
        delay = delay_val.astype(float) if hasattr(delay_val, 'astype') else float(delay_val)
        df['symptom_delay'] = df['has_symptoms'].astype(float) * np.minimum(delay / 30, 1.0)
    if 'cough_contact' not in df.columns:
        cough_val = df.get('cough_frequency', df.get('cough_freq', 0))
        cough = cough_val.astype(float) if hasattr(cough_val, 'astype') else float(cough_val)
        contacts_val = df.get('contact_count', 5)
        contacts = contacts_val.astype(float) if hasattr(contacts_val, 'astype') else float(contacts_val)
        df['cough_contact'] = np.minimum(cough / 20, 1.0) * (contacts / 10)
    if 'highrisk_comorbid' not in df.columns:
        df['highrisk_comorbid'] = df['is_high_risk'].astype(float) * df['past_illness'].astype(float)
    if 'immune_bcg' not in df.columns:
        df['immune_bcg'] = (1 - df['bcg_vaccine'].astype(float)) * immunosuppression_score
    if 'exposure_accumulation' not in df.columns:
        ce_val = df.get('cumulative_exposure', 0)
        ce = ce_val.astype(float) if hasattr(ce_val, 'astype') else float(ce_val)
        ts_val = df.get('time_span', 4)
        ts = ts_val.astype(float) if hasattr(ts_val, 'astype') else float(ts_val)
        df['exposure_accumulation'] = (ce / 80) * (ts / 10)
    if 'age_diabetes' not in df.columns:
        df['age_diabetes'] = df['age'].astype(float) * diabetes

    return df, missing


# 组感知 CV 的候选组列（任务1）：按顺序探测第一个命中列。
# Brazil 的 group_raw（TST 分组，泄漏排除）刻意不在候选内。
GROUP_COLUMN_CANDIDATES = ('family_id', 'group_id', 'household_id',
                           'hh_id', 'cluster_id')


# ---- P2（小队列部署模型路由）：LR 优先触发条件 ----
# 证据（cohort_v4_features_20260915 / family_pooling_20260915）：
# v4 宽特征空间下 LR 配对增益普遍大于树集成（taiwan +0.279 vs LGBM
# +0.20；crp +0.186 vs +0.11）；小队列（brazil 202 / treats / peru_mdr）
# LGBM Δ 不显著甚至为负（brazil -0.025）——小样本 + 宽空间下树模型
# 过拟合，正则化线性模型占优。大样本（nhanes 11.8k / kenya 63k）树
# 集成无损。规则：n<1000 或特征空间宽于基础 22 维（v4 extras 在场）
# → LR 优先部署；全模型 CV 表随路由披露，树模型显著更优时可人工
# 复核覆盖（lr_vs_best_gap 落后幅度可见）。
LR_FIRST_MIN_SAMPLES = 1000
LR_FIRST_BASE_N_FEATURES = 22


def select_deployment_model(n_samples, feature_set, n_features,
                            model_performance):
    """P2：部署模型路由——小样本/宽特征空间时 LR 优先。

    参数：
        n_samples: 训练样本量（None 时跳过该维度判定）
        feature_set: 实际生效的特征集版本（'v1'/'v2'/'v3'/'v4'）
        n_features: 实际特征维度（None 时跳过该维度判定）
        model_performance: 各模型 CV 档案（取 'AUROC' 做 CV 表与
            非触发时的最优路由）

    返回 dict（挂 predictor.deployment_model 并随 checkpoint 落盘）：
        preferred_key: 路由选中的模型键（触发 → 'logistic'；未触发
            → CV AUROC 最优，与 predict_risk 的 best_model 同口径；
            logistic 未训练时同样回退 CV 最优并披露）
        rule / triggered / reasons: 规则名 / 是否触发 / 逐条原因
        cv_auroc: 全模型 CV AUROC 表（透明度披露）
        lr_vs_best_gap: LR 与 CV 最优的 AUROC 差（≥0 = LR 落后幅度，
            供人工复核判断是否覆盖路由）
    """
    cv_auroc = {k: float(v['AUROC']) for k, v in model_performance.items()
                if isinstance(v, dict) and v.get('AUROC') is not None}
    reasons = []
    if n_samples is not None and n_samples < LR_FIRST_MIN_SAMPLES:
        reasons.append(f'n={n_samples} < {LR_FIRST_MIN_SAMPLES}'
                       '（小样本，树集成过拟合风险）')
    if (feature_set == 'v4' and n_features is not None
            and n_features > LR_FIRST_BASE_N_FEATURES):
        reasons.append(f'特征空间 {n_features} 维 > 基础 '
                       f'{LR_FIRST_BASE_N_FEATURES} 维（v4 extras 在场，'
                       '正则化线性占优）')
    lr_available = 'logistic' in cv_auroc
    triggered = bool(reasons) and lr_available
    if not cv_auroc:
        return {'preferred_key': None, 'rule': 'no_cv_metrics',
                'triggered': False, 'reasons': reasons,
                'cv_auroc': {}, 'lr_vs_best_gap': None,
                'note': '无 CV 指标，路由不可用'}
    best_key = max(cv_auroc, key=cv_auroc.get)
    lr_gap = (cv_auroc[best_key] - cv_auroc['logistic']
              if lr_available else None)
    if triggered:
        preferred, rule = 'logistic', 'lr_first_small_or_wide'
        note = ('LR 优先部署（小样本/宽特征空间路由）；树模型 CV 更优时'
                '可人工复核覆盖' if lr_gap and lr_gap > 0 else
                'LR 优先部署（小样本/宽特征空间路由），且 LR 即 CV 最优')
    else:
        preferred, rule = best_key, 'best_cv_auroc'
        note = ('未触发 LR 优先（大样本且特征空间未宽于基础维度）'
                if not reasons else
                '触发条件命中但 logistic 未训练，回退 CV 最优')
    return {'preferred_key': preferred,
            'rule': rule,
            'triggered': triggered,
            'reasons': reasons,
            'cv_auroc': cv_auroc,
            'lr_vs_best_gap': lr_gap,
            'note': note}


def train_from_real_data(predictor, data_filepath, target_column='tb_outcome',
                         enable_hyperopt=False, label_generation=None,
                         group_column=None, feature_set='v1'):
    """从真实数据文件训练模型（支持 CSV/JSON/Excel/Parquet，自动编码检测）

    文件应包含与 ALL_FEATURE_NAMES（22维）对应的列，以及一个目标列（0/1）。
    如果仅包含13维基础特征，将自动计算9维交互特征。
    支持中文字段名自动映射到标准字段名。

    参数：
        data_filepath (str): 数据文件路径（支持 .csv/.json/.jsonl/.xlsx/.xls/.parquet）
        target_column (str): 目标变量列名，默认'tb_outcome'
        enable_hyperopt (bool): 对 5 个基模型启用网格超参搜索
        label_generation (str|None): 标签机制代际（P4，2026-08-24）——
            'v3-host-pathway'（场景 mechanistic 代标签）/
            'clinical-rule-v1'（场景 clinical 代标签）/ None 未声明。
            随 checkpoint 落盘供 load_model 代际校验。
        group_column (str|None): 组感知 CV 的组列（任务1）。None（默认）
            按 GROUP_COLUMN_CANDIDATES 自动探测（TREATS group_id /
            Peru family_id 等）；命中则全链路（超参搜索 + 评估）切换
            StratifiedGroupKFold + 池化 OOF + 户级 cluster bootstrap CI；
            未命中维持随机 StratifiedKFold（无户结构数据集的正确行为）。
            显式指定但列缺失 → error_type='missing_group_column' 失败。
        feature_set (str): 'v1'（默认，22 维）/ 'v4'（族感知，P1
            2026-09-16）。v4 按列签名自动探测队列
            （cohort_features.COHORT_SIGNATURES，不依赖文件名），命中则
            使用 22 维基础 + 队列专属 extras（哨兵清洗 + 缺失指示器 +
            全数据中位数填补，泄漏排除规则见 spec）；特征名清单挂
            predictor.v4_feature_names，随 checkpoint 进 feature_contract。
            未识别队列（合成/未知数据集）自动回退 v1（22 维）并在
            result['feature_space'] 披露——不冒认。spec 单一真值源：
            ml/cohort_features.py（评估脚本 data/run_cohort_v4_features.py
            与本管线共享列集，填补口径差异在 audit 披露）。kenya 自 P6
            （2026-09-15 原始数据再审）起有 spec——v4 含 16 extras。

    返回：
        dict: {
            'success': bool,
            'error_type': str|None,
            'n_samples': int,
            'n_positive': int,
            'n_negative': int,
            'positive_rate': float,
            'model_performance': dict,
            'suggestions': list[str],
            'diagnostics': str,
            'model_selection'（P2，恒在）: 部署模型路由——n<1000 或
                v4 extras 在场 → preferred_key='logistic'（LR 优先，
                小样本/宽空间下树集成过拟合的证据见
                cohort_v4_features_20260915），否则 CV AUROC 最优；
                含 triggered/reasons/cv_auroc/lr_vs_best_gap 披露，
                随 checkpoint 落盘（predict_risk 报 preferred_model）。
            'survival'（P5，时间-事件数据恒在）: 生存路径（peru_mdr
                转正）——follow_up_days 在场时 Cox PH 接管为该数据的
                canonical 训练路径：OOF Harrell C + 固定窗 AUROC
                （180d/365d）+ HR 表 + Schoenfeld PH 检验；部署工件
                predictor.survival_model 随 checkpoint 落盘
                （predict_risk 报 survival_cox 风险分）。
        }
    """
    def _make_result(success, error_type=None, n_samples=0, n_positive=0,
                     n_negative=0, positive_rate=0.0, model_performance=None,
                     suggestions=None, diagnostics=''):
        return {
            'success': success,
            'error_type': error_type,
            'n_samples': n_samples,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'positive_rate': positive_rate,
            'model_performance': model_performance or {},
            'suggestions': suggestions or [],
            'diagnostics': diagnostics,
        }

    if not SKLEARN_AVAILABLE:
        return _make_result(
            False, error_type='sklearn_unavailable',
            diagnostics='scikit-learn 未安装，无法训练模型',
            suggestions=['安装 scikit-learn: pip install scikit-learn'])

    try:
        ext = os.path.splitext(data_filepath)[1].lower()
        supported = ('.csv', '.json', '.jsonl', '.xlsx', '.xls', '.parquet')

        if ext not in supported:
            LOGGER.warning("不支持的文件格式: %s。支持: %s", ext, supported)
            return _make_result(
                False, error_type='unsupported_format',
                diagnostics=f'不支持的文件格式: {ext}。支持: {supported}',
                suggestions=[f'支持的格式: {", ".join(supported)}'])

        if not os.path.exists(data_filepath):
            LOGGER.warning("数据文件不存在: %s", data_filepath)
            return _make_result(
                False, error_type='file_not_found',
                diagnostics=f'数据文件不存在: {data_filepath}',
                suggestions=['请检查文件路径是否正确'])

        # 自动编码检测（CSV/JSON 格式）
        encoding = 'utf-8-sig'
        if ext == '.csv':
            try:
                from ...io_utils import detect_file_encoding
                encoding = detect_file_encoding(data_filepath)
                LOGGER.info("检测到文件编码: %s", encoding)
            except ImportError:
                pass

        # 加载数据
        if ext == '.csv':
            df = pd.read_csv(data_filepath, encoding=encoding)
        elif ext in ('.json', '.jsonl'):
            df = pd.read_json(data_filepath, lines=ext == '.jsonl')
        elif ext in ('.xlsx', '.xls'):
            df = pd.read_excel(data_filepath)
        elif ext == '.parquet':
            df = pd.read_parquet(data_filepath)
        else:
            return _make_result(
                False, error_type='unsupported_format',
                diagnostics=f'不支持的文件格式: {ext}',
                suggestions=[f'支持的格式: {", ".join(supported)}'])

        # 字段名自动映射（中英文同义词）
        from ...io_utils import map_columns
        col_mapping = map_columns(list(df.columns), verbose=True)
        # 保护已是 ML 规范特征名的列：模糊匹配可能把
        # contact_distance_score 等规范列名错误改写为原始筛查字段
        # （如 contact_distance），导致后续 FEATURE_NAMES 校验失败。
        _ml_canonical = set(predictor.ALL_FEATURE_NAMES) | {target_column}
        # v4（P1，2026-09-16）：extras 源列/签名列同受保护——map_columns
        # 会把 ethnic→ethnicity、race_eth→ethnicity、index_smear_grade→
        # sputum_smear、tb_history_raw→has_tb 等，轻则 v4 签名探测失败
        # 静默回退 v1，重则与基础列改名碰撞产生重复列。这些列名是
        # cohort spec 的契约（ml/cohort_features.py）。
        if COHORT_FEATURES_AVAILABLE:
            from .cohort_features import (V4_SOURCE_COLUMNS,
                                          LEAK_REFERENCE_COLUMNS)
            _ml_canonical |= set(V4_SOURCE_COLUMNS) | set(
                LEAK_REFERENCE_COLUMNS)
        col_mapping = {
            orig: std for orig, std in col_mapping.items()
            if orig not in _ml_canonical or std == orig
        }
        if col_mapping:
            df = df.rename(columns=col_mapping)
            # 模糊映射碰撞防御（2026-09-05，CRP 数据集实测）：
            # map_columns 可能把多个原始列改写为同一目标名
            # （symptom_cough / symptom_cough_weeks → cough_freq；
            #  age_group → age 与原 age 碰撞；tb_history_raw → has_tb），
            # 重复列使 df['age'] 返回 DataFrame 而非 Series，
            # 后续算术运算以晦涩的 "cannot reindex on an axis with
            # duplicate labels" 失败。处置：保留首次出现的列（原始语义
            # 优先，因规范名列通常排在附加列之前），丢弃后续副本。
            if df.columns.duplicated().any():
                dup_names = df.columns[df.columns.duplicated()].unique().tolist()
                LOGGER.warning("字段映射产生重复列: %s，保留首个出现", dup_names)
                df = df.loc[:, ~df.columns.duplicated()]

        # 目标列映射
        if target_column not in df.columns:
            from ...io_utils import dynamic_column_match
            original_target = target_column
            matched = dynamic_column_match(target_column)
            if matched and matched in df.columns:
                target_column = matched
                LOGGER.info("目标列自动映射: '%s' → '%s'", original_target, matched)
            else:
                LOGGER.warning("CSV缺少目标列: %s。可用列: %s", target_column, list(df.columns))
                return _make_result(
                    False, error_type='missing_target_column',
                    diagnostics=f'数据缺少目标列: {target_column}',
                    suggestions=['请指定正确的目标列名', '确保数据包含标签列(0/1)'])

        # 数据完整性预验证
        try:
            from ...io_utils import validate_data_integrity
            records = df.to_dict('records')
            integrity = validate_data_integrity(records, verbose=True)
            if not integrity['valid']:
                LOGGER.warning("数据完整性预验证未通过，但尝试继续训练...")
        except ImportError:
            pass

        base_missing = [f for f in predictor.FEATURE_NAMES if f not in df.columns]
        if base_missing:
            LOGGER.warning("CSV缺少以下基础特征列: %s", base_missing)
            return _make_result(
                False, error_type='missing_base_features',
                diagnostics=f'缺少基础特征列: {base_missing}',
                suggestions=['确保数据包含所有必需的基础特征列',
                             f'必需列: {predictor.FEATURE_NAMES}'])

        interaction_missing = [f for f in predictor.INTERACTION_FEATURE_NAMES if f not in df.columns]
        if interaction_missing:
            LOGGER.info("CSV缺少交互特征列%s，将自动计算...", interaction_missing)
            df, _filled = ensure_interaction_features(
                df, predictor.INTERACTION_FEATURE_NAMES)

        # ---- 组感知 CV（任务1，2026-09-14）----
        # group_column=None → 按 GROUP_COLUMN_CANDIDATES 自动探测（TREATS
        # group_id=社区 / Peru family_id=户）；显式指定但列缺失 → 响亮失败
        # （静默回退随机 CV 会让"以为组感知"的调用方拿到泄漏偏乐观的
        # AUROC，比不启用更危险）。组列不在 ALL_FEATURE_NAMES 内，
        # 天然不进特征矩阵。
        groups = None
        detected_group_column = None
        if group_column is None:
            for cand in GROUP_COLUMN_CANDIDATES:
                if cand in df.columns:
                    detected_group_column = cand
                    break
        elif group_column in df.columns:
            detected_group_column = group_column
        else:
            LOGGER.warning("指定的组列不存在: %s。可用列: %s",
                           group_column, list(df.columns))
            return _make_result(
                False, error_type='missing_group_column',
                diagnostics=f'指定的组列不存在: {group_column}',
                suggestions=['检查 group_column 拼写',
                             f'候选组列: {list(GROUP_COLUMN_CANDIDATES)}'])
        if detected_group_column is not None:
            groups = df[detected_group_column].to_numpy()
            LOGGER.info("组感知 CV 启用: 组列=%s, 组数=%d",
                        detected_group_column,
                        int(len(np.unique(groups))))

        X = df[predictor.ALL_FEATURE_NAMES].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        y = pd.to_numeric(df[target_column], errors='coerce').fillna(0).values.astype(int)

        # ---- 族感知 v4 特征空间（P1，2026-09-16）----
        # 列签名探测队列 → 22 维基础（fillna(0)，与存量 v1 语义一致）+
        # per-cohort extras（哨兵清洗 + 缺失指示器 + 全数据中位数填补）。
        # 未识别队列（合成/未知）回退 v1 并在 result 披露——不冒认。
        # X_cv_raw（缺陷1口径统一）：extras NaN 保留版，供 CV 指标折内
        # 中位填补（与 F2 判决口径一致）；部署工件仍用 X（全数据中位）。
        v4_audit = None
        X_cv_raw = None
        feature_set_effective = feature_set
        if feature_set == 'v4':
            if not COHORT_FEATURES_AVAILABLE:
                LOGGER.warning("cohort_features 模块不可用（pandas 缺失？）→ v4 回退 v1")
                feature_set_effective = 'v1'
            else:
                X_v4, X_v4_raw, names_v4, v4_audit = apply_v4_feature_space(
                    df, list(predictor.ALL_FEATURE_NAMES), return_raw=True)
                if X_v4 is None:
                    LOGGER.warning(
                        "v4 队列未识别（cohort=%s）→ 回退 v1（22 维）",
                        v4_audit.get('cohort'))
                    feature_set_effective = 'v1'
                else:
                    X = X_v4
                    X_cv_raw = X_v4_raw
                    predictor.v4_feature_names = list(names_v4)
                    predictor.v4_cohort = v4_audit.get('cohort')
                    # 训练中位数随 predictor 落盘（部署端单样本构造
                    # 必须复用，绝不在部署数据上重算）
                    predictor.v4_fill_medians = (v4_audit.get('fill_medians')
                                                 or {})
                    feature_set_effective = 'v4'
                    LOGGER.info(
                        "v4 族感知特征空间: cohort=%s，%d 基础 + %d extras = %d 列",
                        v4_audit.get('cohort'), v4_audit.get('n_base'),
                        v4_audit.get('n_extra_cols'), v4_audit.get('n_total'))
                    # P1-c（treats 降级机制护栏，2026-09-20）：队列命中
                    # RESEARCH_ONLY_COHORTS → predictor 标记 research_only
                    # （随 checkpoint 落盘，predict_risk 部署入口拦截）。
                    # 降级判决此前仅有文档约束，任何站点仍可静默加载
                    # treats checkpoint 做筛查决策。
                    from ...constants import RESEARCH_ONLY_COHORTS
                    _downgrade = RESEARCH_ONLY_COHORTS.get(
                        predictor.v4_cohort)
                    if _downgrade is not None:
                        predictor.deployment_status = 'research_only'
                        predictor.research_only_reason = _downgrade['reason']
                        LOGGER.warning(
                            '队列 %s 已降级为研究资产（禁止部署）：'
                            '%s；checkpoint 将带 research_only 标记，'
                            'predict_risk 部署入口将拦截（研究用途显式 '
                            'allow_research=True）',
                            predictor.v4_cohort, _downgrade['reason'])
        if feature_set_effective != 'v4':
            # 回退/非 v4 请求：清除陈旧挂载，避免重复训练时特征契约误读
            for _attr in ('v4_feature_names', 'v4_cohort',
                          'v4_fill_medians'):
                if hasattr(predictor, _attr):
                    delattr(predictor, _attr)
            # 非本次 v4 降级队列训练时，清除陈旧降级标记（同一 predictor
            # 重复用于多队列训练时不残留）
            if getattr(predictor, 'deployment_status', None) == 'research_only':
                delattr(predictor, 'deployment_status')
                if hasattr(predictor, 'research_only_reason'):
                    delattr(predictor, 'research_only_reason')

        if len(np.unique(y)) < 2:
            unique_values, counts = np.unique(y, return_counts=True)
            pos_count = int(np.sum(y == 1))
            neg_count = int(np.sum(y == 0))
            total = len(y)
            LOGGER.warning(
                "目标变量只有一个类别，无法训练分类模型。\n"
                "  诊断信息：\n"
                "    总样本数: %d\n"
                "    正样本数(确诊): %d (%.2f%%)\n"
                "    负样本数(未确诊): %d (%.2f%%)\n"
                "    唯一值: %s\n"
                "    建议：\n"
                "      - 确保数据包含至少两类标签（0/1）\n"
                "      - 所需最小正样本量建议 ≥ 10\n"
                "      - 如数据极度不均衡，可考虑过采样（SMOTE）或调整类别权重\n"
                "      - 目标列当前值分布: %s",
                total, pos_count, pos_count/max(total,1)*100,
                neg_count, neg_count/max(total,1)*100,
                list(unique_values),
                dict(zip(unique_values.astype(int), counts))
            )
            return _make_result(
                False, error_type='single_class_label',
                n_samples=total, n_positive=pos_count, n_negative=neg_count,
                positive_rate=pos_count / max(total, 1),
                diagnostics=f'目标变量仅含单一类别 (值: {list(unique_values)})，无法训练分类模型',
                suggestions=[
                    '确保数据包含至少两类标签（0/1）',
                    '所需最小正样本量建议 ≥ 10',
                    '如数据极度不均衡，可考虑过采样（SMOTE）或调整类别权重',
                ])

        predictor.training_sample_count = len(y)
        predictor.use_real_data = True
        predictor.reset_stop_flag()

        # 委托统一训练内核（groups 透传 → 组感知 CV 全链路；
        # feature_set_effective 透传 → v4 时内核不选列，特征契约挂
        # predictor.v4_feature_names；X_cv_raw 透传 → v4 CV 指标折内
        # 中位填补口径）
        success = _train_all_registered_models(
            predictor, X, y,
            random_state=42,
            enable_hyperopt=enable_hyperopt,
            data_source_label='real',
            label_generation=label_generation,
            groups=groups,
            group_column_name=detected_group_column,
            feature_set=feature_set_effective,
            X_cv_raw=X_cv_raw,
        )

        if not success:
            return _make_result(
                False, error_type='general_error',
                diagnostics='模型训练失败，请查看日志',
                suggestions=['检查日志获取详细错误信息'])

        result = _make_result(
            True,
            n_samples=len(y),
            n_positive=int(np.sum(y == 1)),
            n_negative=int(np.sum(y == 0)),
            positive_rate=float(np.mean(y)) if len(y) > 0 else 0.0,
            model_performance=predictor.model_performance,
            diagnostics='模型训练成功')
        # 组感知协议信息（任务1）：编排层（train_with_public_data）直接
        # 读取入档，无需翻 model_performance
        result['group_column'] = detected_group_column
        result['cv_protocol'] = ('stratified_group_kfold'
                                 if groups is not None else 'stratified_kfold')
        if groups is not None:
            result['n_groups'] = int(len(np.unique(groups)))
        # v4 特征空间披露（P1）：队列/extras 规模/哨兵与指示器审计/回退说明
        if feature_set == 'v4':
            result['feature_space'] = {
                'requested': 'v4',
                'effective': feature_set_effective,
                'cohort': (v4_audit or {}).get('cohort'),
                'n_base': (v4_audit or {}).get('n_base'),
                'n_extra_cols': (v4_audit or {}).get('n_extra_cols'),
                'n_total': (v4_audit or {}).get('n_total'),
                'sentinels_applied': (v4_audit or {}).get('sentinels_applied'),
                'indicators_added': (v4_audit or {}).get('indicators_added'),
                'note': ((v4_audit or {}).get('note')
                         if feature_set_effective != 'v4' else None),
            }
        # P1-c（treats 降级机制护栏）：降级状态随 result 披露（编排层
        # run JSON 直接可读，不翻 checkpoint）
        if getattr(predictor, 'deployment_status', None) == 'research_only':
            result['deployment_status'] = {
                'status': 'research_only',
                'cohort': predictor.v4_cohort,
                'reason': getattr(predictor, 'research_only_reason', ''),
                'note': 'predict_risk 部署入口已拦截；研究用途显式 '
                        'allow_research=True',
            }
        # P2（小队列部署模型路由）：n<1000 或 v4 extras 在场 → LR 优先；
        # 全模型 CV 表 + LR 落后幅度随路由披露（挂 predictor + result，
        # checkpoint 落盘后部署端 predict_risk 报 preferred_model）
        selector = select_deployment_model(
            len(y), feature_set_effective,
            int(np.asarray(X).shape[1]) if np.asarray(X).ndim == 2 else None,
            predictor.model_performance)
        predictor.deployment_model = selector
        result['model_selection'] = selector
        LOGGER.info('部署模型路由（P2）: preferred=%s rule=%s triggered=%s',
                    selector['preferred_key'], selector['rule'],
                    selector['triggered'])

        # ---- P5 生存路径（peru_mdr 转正，2026-09-16）----
        # follow_up_days 在场 = 时间-事件结构（incident TB 天然是
        # time-to-event 终点；二分类口径扔掉事件/删失 275 天中位信息差，
        # F4 判决 B 版 Harrell C 0.678 > A 版 0.629）→ Cox PH 接管为
        # 该数据的正式训练路径：特征空间沿用本管线 effective 集（v4 =
        # B 版 22+peru extras / v1 = A 版 22 维），OOF Harrell C + 固定窗
        # AUROC + HR 表 + Schoenfeld PH 检验随 result 披露，部署工件
        # predictor.survival_model 随 checkpoint 落盘。分类六模型仍训练
        # （作对照基线），但时间-事件数据的 canonical 工件是生存模型。
        time_col = next(
            (c for c in SURVIVAL_TIME_COLUMNS if c in df.columns), None)
        if time_col is None:
            # 无时间轴列：清除陈旧挂载，避免重复训练时误读（镜像 v4
            # 回退清理模式）
            if hasattr(predictor, 'survival_model'):
                delattr(predictor, 'survival_model')
        elif not SURVIVAL_COX_IMPORTABLE or not STATSMODELS_SURVIVAL_AVAILABLE:
            LOGGER.warning(
                '检测到时间-事件列 %s 但 statsmodels 不可用 → 生存路径'
                '跳过（分类基线仍训练）', time_col)
        else:
            try:
                t_s = pd.to_numeric(df[time_col], errors='coerce')
                ok = t_s.notna() & (t_s > 0)
                surv_names = (list(predictor.v4_feature_names)
                              if feature_set_effective == 'v4'
                              else list(predictor.ALL_FEATURE_NAMES))
                surv_result = train_survival_model(
                    t_s[ok].to_numpy(), y[ok], np.asarray(X)[ok],
                    surv_names, groups=(groups[ok] if groups is not None
                                        else None),
                    time_column=time_col,
                    cohort=(v4_audit or {}).get('cohort'))
                if surv_result.get('success'):
                    predictor.survival_model = surv_result.pop('model')
                    result['survival'] = surv_result
                    LOGGER.info(
                        '生存路径（P5）: cohort=%s n=%d events=%d '
                        'HarrellC=%.4f',
                        (v4_audit or {}).get('cohort'),
                        surv_result['n_samples'], surv_result['n_events'],
                        surv_result['harrell_c_oof'])
                else:
                    result['survival'] = surv_result
                    LOGGER.warning('生存路径跳过: %s (%s)',
                                   surv_result.get('error_type'),
                                   surv_result.get('note'))
            except Exception as surv_err:
                LOGGER.warning('生存路径异常（分类结果不受影响）: %s',
                               surv_err, exc_info=True)
                result['survival'] = {'success': False,
                                      'error_type': 'survival_error',
                                      'note': str(surv_err)}
        return result

    except Exception as e:
        LOGGER.warning("从真实数据训练失败: %s", e, exc_info=True)
        return _make_result(
            False, error_type='general_error',
            diagnostics=f'训练失败: {e}',
            suggestions=['请检查数据格式', '查看日志获取详细错误信息'])
