#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEIR 模型初始化器 — 初始化随机 SEIR + 贝叶斯推断模块

从 assessment.py 的 _init_seir_models 方法提取，供 GUI 和 CLI 共用。

解耦设计：返回 SEIRInitResult 数据对象而非 mutate 外部 assessment 实例，
调用方接收后自行赋值，core 不再依赖 GUI 控制器属性契约。

问题三（统一贝叶斯框架）：原先同时实例化 BayesianSEIRInference（V4/HMC）
与 SEIRParameterUncertainty（MH-MCMC Mixin），两套框架各自维护后验采样，
分别喂给不同消费者，导致同一份后验推断被做两遍。现统一为单一 V4 实例：
  - SEIRIntegration 通过 seir_param_uncertainty（V4 别名）读取
    inference_completed / posterior_summary / mcmc_diagnostics
    （后者由 BaseBayesianInference 适配属性派生计算）
  - PosteriorDrivenInfectivity 通过 seir_inference 读取 posterior_samples
    （创建时注入，修复原 wiring 缺失 bug）
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from ..constants import SEIR_RANDOM_SEED

LOGGER = logging.getLogger(__name__)


@dataclass
class SEIRInitResult:
    """SEIR 初始化产物（纯数据契约，core 自定义）

    注：seir_param_uncertainty 与 seir_inference 指向同一 V4 实例（别名），
    保留双字段是为了向后兼容 SEIRIntegration 的 set_param_uncertainty 接口。
    """
    stochastic_seir: Optional[Any] = None
    seir_inference: Optional[Any] = None
    posterior_infectivity: Optional[Any] = None
    seir_param_uncertainty: Optional[Any] = None


def init_seir_models(seir_population: int = 10000,
                     random_state: Optional[int] = None) -> SEIRInitResult:
    """初始化随机 SEIR + 贝叶斯推断模块，返回 SEIRInitResult。

    Args:
        seir_population: SEIR 种群规模（由调用方从 karamay_localizer 解析后传入）
        random_state: 随机种子（透传给 V4 推断引擎）

    Returns:
        SEIRInitResult: 含 stochastic_seir / seir_inference /
        posterior_infectivity / seir_param_uncertainty；SEIR 不可用时字段为 None。
        seir_param_uncertainty 与 seir_inference 指向同一 V4 实例。
    """
    try:
        from ..seir.stochastic import StochasticSEIRModel
        from ..seir.bayesian import BayesianSEIRInference
        from ..seir.posterior import PosteriorDrivenInfectivity
        SEIR_AVAILABLE = True
    except ImportError:
        StochasticSEIRModel = None
        BayesianSEIRInference = None
        PosteriorDrivenInfectivity = None
        SEIR_AVAILABLE = False

    if not SEIR_AVAILABLE:
        return SEIRInitResult()

    result = SEIRInitResult()
    try:
        seir_seed = random_state if random_state is not None else SEIR_RANDOM_SEED
        result.stochastic_seir = StochasticSEIRModel(
            population=seir_population, noise_scale=0.05, seed=seir_seed)
        # 单一 V4 实例：BayesianSEIRInference 是 BayesianInferenceV4 的别名
        result.seir_inference = BayesianSEIRInference(
            result.stochastic_seir, seed=seir_seed)
        # 修复 wiring bug：创建 PosteriorDrivenInfectivity 时注入 seir_inference，
        # 否则 factors 属性始终走 _default_factors()，V4 后验从未被消费
        result.posterior_infectivity = PosteriorDrivenInfectivity(
            seir_inference=result.seir_inference)
        # 统一框架：seir_param_uncertainty 作为 V4 实例的别名，
        # 供 SEIRIntegration.set_param_uncertainty 接口消费
        result.seir_param_uncertainty = result.seir_inference
    except (ValueError, TypeError) as e:
        LOGGER.warning(
            "SEIR 核心模块初始化失败: %s", e, exc_info=True)
        return result

    return result
