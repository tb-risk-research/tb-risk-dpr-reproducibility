#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯概率计算函数

包含 Sigmoid 感染概率、累积暴露计算、SEIR 概率等核心数学函数。
所有函数零依赖 tkinter 和 GUI，仅依赖 math 和 ScoringEngine 常量。
"""

import math

from ..scoring.engine import ScoringEngine


def sigmoid_base_infection(risk_score, coeff=None, offset=None, cap=None):
    """统一的 sigmoid 感染概率计算

    公式: cap / (1 + exp(-coeff * (risk_score - offset)))，上限 cap
    用于将风险评分映射到感染概率(0-95%)。

    参数从 ScoringEngine 类属性读取（单一真值源），与 scoring/engine.py 保持一致。

    参数：
        risk_score: float — 风险评分
        coeff: float | None — sigmoid 系数（默认使用 ScoringEngine.SIGMOID_COEFF）
        offset: float | None — sigmoid 偏移（默认使用 ScoringEngine.SIGMOID_OFFSET）
        cap: float | None — sigmoid 上限（默认使用 ScoringEngine.SIGMOID_CAP）

    返回：
        float: 感染概率 (0-cap)
    """
    if coeff is None:
        coeff = ScoringEngine.SIGMOID_COEFF
    if offset is None:
        offset = ScoringEngine.SIGMOID_OFFSET
    if cap is None:
        cap = ScoringEngine.SIGMOID_CAP
    z = -coeff * (risk_score - offset)
    z_clipped = max(-ScoringEngine.SIGMOID_CLIP_THRESHOLD,
                    min(ScoringEngine.SIGMOID_CLIP_THRESHOLD, z))
    return min(cap / (1 + math.exp(z_clipped)), cap)


def sigmoid_total_score(x):
    """患者整体风险 sigmoid 评分映射 (0-95%)

    用于将总评分映射到基础感染概率。
    参数与 ScoringEngine 保持一致：coeff=0.3, offset=10, cap=95。

    参数：
        x: float — 总风险评分

    返回：
        float: 基础感染概率
    """
    z = -ScoringEngine.SIGMOID_COEFF * (x - ScoringEngine.SIGMOID_OFFSET)
    z_clipped = max(-ScoringEngine.SIGMOID_CLIP_THRESHOLD,
                    min(ScoringEngine.SIGMOID_CLIP_THRESHOLD, z))
    return ScoringEngine.SIGMOID_CAP / (1.0 + math.exp(z_clipped))


def calculate_cumulative_exposure(single_duration, freq_density, time_span,
                                   workplace_type='非油田'):
    """计算累积暴露时长（小时）

    参数：
        single_duration: float | None — 单次有效接触时长（分钟）
        freq_density: float | None — 每周接触频次（次数）
        time_span: float | None — 接触持续周期（周数）
        workplace_type: str — 工作场景（'油田'/'非油田'），油田轮班制修正系数0.70

    返回：
        float: 累积暴露时长（小时）
    """
    if single_duration is None:
        single_duration = 0
    if freq_density is None:
        freq_density = 0
    if time_span is None:
        time_span = 0
    single_duration_hours = max(float(single_duration), 0) / 60.0
    cumulative_exposure = single_duration_hours * float(freq_density) * float(time_span)
    if workplace_type == '油田':
        cumulative_exposure *= 0.70
    return cumulative_exposure


def safe_float(value, default=None):
    """安全转换为 float，失败返回 default

    参数：
        value: Any — 待转换的值
        default: Any — 默认值

    返回：
        float | default_type
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=None):
    """安全转换为 int，失败返回 default

    参数：
        value: Any — 待转换的值
        default: Any — 默认值

    返回：
        int | default_type
    """
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def exposure_to_infection_probability(cumulative_hours, max_prob=1.0):
    """暴露时长 → 感染概率转换

    使用指数衰减模型：P = 1 - exp(-λ * hours)，确保暴露 → ∞ 时 P → 1.0。

    注：指数暴露—感染关系即 Wells-Riley 方程形式（P = 1 - exp(-Iqpt/Q)），
    其中 80 小时这一等效接触阈值（λ=1/80）为自设假设，无直接文献给出该
    具体数值；量级可对照下述文献。

    参数：
        cumulative_hours: float — 累积暴露时长（小时）
        max_prob: float — 最大感染概率（默认 1.0）

    返回：
        float: 感染概率 (0 - max_prob)
    """
    # λ 参数：80 小时暴露对应约 63% 概率（1 - 1/e）
    # 文献：Wells WF. Airborne Contagion and Air Hygiene. 1955；
    #       Riley RL et al. Am Rev Respir Dis 1978;117(5):893-908（Wells-Riley
    #       指数暴露—感染关系）；
    #       Escombe AR et al. PLoS Med 2009;6(8):e1000152（通风对结核传播的
    #       定量影响，可支撑暴露时长系数量级）
    lam = 1.0 / 80.0
    prob = max_prob * (1.0 - math.exp(-lam * max(cumulative_hours, 0)))
    return min(prob, max_prob)


def get_exposure_risk_score(cumulative_exposure) -> float:
    """根据累积暴露时长获取风险评分（0~4）

    共享辅助函数：对累积暴露时长做类型安全转换后，
    委托到 ScoringEngine._exposure_to_risk 的阈值分级逻辑。
    被 ContactRiskCalculator 和 PatientScorer 共同使用。

    参数：
        cumulative_exposure: float | None — 累积暴露时长（小时）

    返回：
        float: 暴露风险评分（0/1/2/3/4）
    """
    try:
        cumulative_exposure = (
            float(cumulative_exposure)
            if cumulative_exposure is not None else 0.0
        )
    except (ValueError, TypeError):
        cumulative_exposure = 0.0
    return ScoringEngine._exposure_to_risk(cumulative_exposure)