#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高级功能初始化器 — 初始化 GNN / 干预 / 多模态 / 不确定性模块

从 assessment.py 的 _init_advanced_features 方法提取，供 GUI 和 CLI 共用。

解耦设计：返回 AdvancedFeaturesInitResult 数据对象而非 mutate 外部 assessment
实例，调用方接收后自行赋值，core 不再依赖 GUI 控制器属性契约。
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class AdvancedFeaturesInitResult:
    """高级功能初始化产物（纯数据契约，core 自定义）

    所有字段默认为 None / False / []，与原 assessment 属性名一一对应，
    调用方可通过 vars(result) 批量赋值到 self。
    """
    # GNN
    hetero_gnn: Optional[Any] = None
    time_varying_builder: Optional[Any] = None
    three_phase_trainer: Optional[Any] = None
    causal_gnn_graph: Optional[Any] = None
    # 干预优化
    counterfactual_optimizer: Optional[Any] = None
    preference_policy: Optional[Any] = None
    dynamic_planner: Optional[Any] = None
    intervention_validator: Optional[Any] = None
    _intervention_optimizer_ready: bool = False
    # 多模态
    multimodal_data_access: Optional[Any] = None
    temporal_graph_builder: Optional[Any] = None
    multimodal_gnn: Optional[Any] = None
    temporal_trainer: Optional[Any] = None
    gnn_snapshot_cache: List = field(default_factory=list)
    _multimodal_ready: bool = False
    # 不确定性量化
    deep_ensemble: Optional[Any] = None
    conformal_predictor: Optional[Any] = None
    mc_dropout_gnn: Optional[Any] = None
    swag_estimator: Optional[Any] = None
    uncertainty_fusion: Optional[Any] = None
    uncertainty_visualizer: Optional[Any] = None
    seir_posterior_predictive: Optional[Any] = None
    _uncertainty_ready: bool = False


def init_advanced_features() -> AdvancedFeaturesInitResult:
    """初始化高级功能模块，返回 AdvancedFeaturesInitResult。

    所有字段初始化为 None / False / []（占位），后续由各子模块按需填充。
    调用方接收后自行赋值到 self，core 不再 mutate 外部实例。

    Returns:
        AdvancedFeaturesInitResult: GNN / 干预 / 多模态 / 不确定性 占位字段。
    """
    return AdvancedFeaturesInitResult()
