#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 核心业务层

纯业务逻辑，零依赖 tkinter 或任何 GUI 库。
可在 CLI/Docker/Web 环境中直接使用，无需加载 GUI。

模块：
- assessment_service: 风险评估服务（编排层，含所有业务逻辑方法）
- probability: 概率计算（Sigmoid、暴露-感染转换、SEIR概率）
- data_conversion: 数据转换（中文映射、类型转换、字段标准化）
- weight_optimizer: 集成权重自动优化器
- ppv: PPV 贝叶斯换算与"排序+截断点"决策参考（阳性率双口径）
"""

from .assessment_service import RiskAssessmentService
from .patient_scorer import PatientScorer
from .contact_risk_calculator import ContactRiskCalculator
from .seir_integration import SEIRIntegration, fuse_probabilities
# 业务常量单一真值源来自 constants.py，此处仅做聚合再导出
from ..constants import (
    TREATMENT_INFECTIVITY_FACTORS,
    BCG_PROTECTION_PARAMS,
    PATIENT_TYPE_ADJUSTMENTS,
    SYMPTOMS_PRESENT_BONUS,
    AGE_PROGRESSION,
    DEFAULT_DISTANCE_FACTOR,
    DEFAULT_VENTILATION_FACTOR,
    DEFAULT_SETTING_FACTOR,
    RISK_THRESHOLDS,
)
# 向后兼容别名（与 AGE_PROGRESSION 同一真值源）
AGE_PROGRESSION_FACTORS = AGE_PROGRESSION
# LATENT_TO_ACTIVE_BASELINE 是 RiskAssessmentService 的类属性，不是模块级常量
_LATENT_BASELINE = RiskAssessmentService.LATENT_TO_ACTIVE_BASELINE
from .probability import (
    sigmoid_base_infection,
    sigmoid_total_score,
    calculate_cumulative_exposure,
    safe_float,
    safe_int,
    exposure_to_infection_probability,
    get_exposure_risk_score,
)
from .data_conversion import (
    convert_chinese_to_value,
    standardize_contact_fields,
    apply_scenario_defaults,
    get_priority_and_recommendation,
    DISTANCE_TEXT_MAP,
    SETTING_TEXT_MAP,
)
from .weight_optimizer import EnsembleWeightOptimizer
from .ml_runner import run_ml_prediction_loop
from .seir_initializer import init_seir_models, SEIRInitResult
from .features_initializer import init_advanced_features, AdvancedFeaturesInitResult
from .integrator import ThreeDirectionIntegrator
from . import ppv

__all__ = [
    'RiskAssessmentService', 'EnsembleWeightOptimizer',
    'PatientScorer', 'ContactRiskCalculator', 'SEIRIntegration',
    'fuse_probabilities',
    'run_ml_prediction_loop', 'init_seir_models', 'init_advanced_features',
    'SEIRInitResult', 'AdvancedFeaturesInitResult',
    'ThreeDirectionIntegrator', 'ppv',
    'TREATMENT_INFECTIVITY_FACTORS', 'BCG_PROTECTION_PARAMS',
    'PATIENT_TYPE_ADJUSTMENTS', 'SYMPTOMS_PRESENT_BONUS',
    'AGE_PROGRESSION_FACTORS',
    'DEFAULT_DISTANCE_FACTOR', 'DEFAULT_VENTILATION_FACTOR',
    'DEFAULT_SETTING_FACTOR', 'RISK_THRESHOLDS',
    'sigmoid_base_infection', 'sigmoid_total_score',
    'calculate_cumulative_exposure', 'safe_float', 'safe_int',
    'exposure_to_infection_probability', 'get_exposure_risk_score',
    'convert_chinese_to_value', 'standardize_contact_fields',
    'apply_scenario_defaults', 'get_priority_and_recommendation',
    'DISTANCE_TEXT_MAP', 'SETTING_TEXT_MAP',
]