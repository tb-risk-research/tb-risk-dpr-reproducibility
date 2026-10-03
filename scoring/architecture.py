#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三层递进架构编排器（个体基础层 → 网络增强层 → 社区干预层）。

对评审核心批评"加权平均集成"的正面回应：用"堆叠 + 门控"替代概率层面的
线性平均。三层各司其职、分别训练、分别评估，仅在**决策层**做整合。

第 1 层（个体基础层）——ML 个体风险模型
  输出每个个体的基线患病概率 ``P_base(x)``，作为后续所有层的输入。
  该层可用公开个体数据（如 WHO/国家监测 + 文献效应量合成）即可训练。

第 2 层（网络增强层）——两条路径，时序先证为部署真实实现
  路径 A（2026-08-25 第四轮 P1，优先）：**时序家庭先验**
  （scoring.temporal_household）——评分流程支持"户内已筛查
  结果"作为输入：同户有人筛查阳性时其余成员风险上调并自动
  重排（户间），首筛查者回退个体基线。验证依据：HomeACF 20
  种子 exposure+prior:RF 0.7142 vs ind:RF 0.6428（+0.0585 CI
  显著）——网络层构造首次在真实数据上取得显著正增益。
  路径 B（无筛查状态时的原有路径）：GNN 学习"增量"而非"并列估计"
  - GNN 节点特征不再只用原始人口学/临床特征，而是把第 1 层输出的
    ``P_base`` 作为节点特征之一（堆叠）；
  - GNN 学习目标改为**残差** ``residual = 真实结局 − 个体基线预测``，
    只负责解释"接触结构带来的额外风险"。
  输出可解释为"网络增量风险"：``个体风险 + 网络增量``，语义清晰，
  消融实验（有/无 GNN 层 ΔAUROC / ΔC-index）直接量化网络贡献，
  并避免了加权平均中"网络层与个体层重复计分"的问题。

第 3 层（社区干预层）——SEIR 只做"干预反事实模拟"
  给定第 1、2 层识别出的高风险个体集合，模拟不同干预策略（预防性治疗、
  缩短筛查间隔等）下社区 6-24 个月的病例数变化，输出**人群层面**干预
  效果曲线（病例数 / 二代病例 / 可避免病例），不再输出个体概率，
  与个体/网络层在语义上彻底解耦。

三层之间的连接方式（堆叠 + 门控）：
  - 第 2 层以第 1 层输出为特征（堆叠 / stacking）；
  - 第 3 层以第 1、2 层输出的高风险集合为输入；
  - 各层分别训练、分别评估；
  - 最后只在"决策层"做整合：``个体概率 + 网络增量 + 干预收益``
    联合决策（门控 / gating），而非概率层面线性平均。

文献：
  Wolpert (1992) Stacked Generalization — 堆叠泛化
  Jacobs et al. (1991) Adaptive Mixtures of Local Experts — 门控机制
  Breiman (1996) Stacked Regressions
  Caruana et al. (2004) Ensemble Selection — 层级集成
"""

import logging
import math

import numpy as np

from ..constants import DISEASE_PROB_VERY_HIGH, DEFAULT_RISK_CLASS_THRESHOLD
from .temporal_household import (
    DEFAULT_TEMPORAL_SHRINKAGE,
    DEFAULT_TEMPORAL_WEIGHT,
    household_rate_estimate,
    temporal_increment,
    temporal_prior_features,
)

LOGGER = logging.getLogger("tb_risk.scoring.architecture")

# ==============================================================================
# 常量与默认参数
# ==============================================================================

# 高风险干预集合判定阈值（M 级修复：单一真值源 + 语义对齐）
# 引用 constants.DISEASE_PROB_VERY_HIGH（= threshold_spec.DEFAULT_VERY_HIGH），
# 即四档分级中的【极高风险】切点（≥15）——注意 threshold_spec 的"高风险"
# 档切点是 8（DISEASE_PROB_HIGH）。select_high_risk_set 采用极高风险切点作为
# 进入第 3 层干预模拟的保守下限（高风险档 8 仅触发分级告警，不进干预模拟）。
DEFAULT_HIGH_RISK_THRESHOLD = DISEASE_PROB_VERY_HIGH   # 0-100 尺度，≥15 → 极高风险

# 门控默认参数
DEFAULT_GATE_MIN = 0.0                # 无网络证据时网络增量不生效
DEFAULT_GATE_MAX = 1.0                # 强网络证据时网络增量完全生效

# 决策层默认权重（个体 / 干预；网络增量已通过门控并入个体概率，不重复计分，
# 故不含 network 键。注意：这是手调常数归一（0.6/0.1，和为 0.7 由
# gated_integration 内部归一），并非 softmax 输出——softmax 加权仅用于
# integrator.optimize_weights_from_validation 的方向级权重学习）
# ⚠ 2026-09-05 地基审计：以下权重为手调直觉常数，未经扫描/交叉验证——
# 真实数据校准前不构成可引用证据（对照：constants.SCENARIO_WEIGHTS 中
# 三/四方向权重已经过扫描验证，two_* 双方向组尚未验证）
DEFAULT_DECISION_WEIGHTS = {
    'individual': 0.6,
    'intervention': 0.1,
}

# 干预收益可避免病例→个体决策的增益系数（每避免 1 病例·天折算决策加分）
# ⚠ 同上：手调常数，未验证
INTERVENTION_BENEFIT_GAIN = 0.05


# ==============================================================================
# 第 1 层：个体基础层（ML 基线概率 P_base）
# ==============================================================================

def compute_individual_base(ml_predictor, contact_data, contact_type='family'):
    """第 1 层：计算个体基线患病概率 ``P_base(x)``。

    该层只依赖个体级特征，不依赖接触网络结构，可用公开个体数据训练。

    参数：
        ml_predictor: 已训练的 MLRiskPredictor（predict_risk 输出 0-100 集成概率）
        contact_data (dict): 单个接触者数据
        contact_type (str): 'family' / 'social'

    返回：
        dict:
            - 'p_base': 0-1 基线患病概率
            - 'p_base_percent': 0-100 基线患病概率（评分尺度）
            - 'risk_class': 二分类（1 = 高风险）
            - 'is_trained': ML 是否已训练
            - 'model_performance': 模型性能摘要（若有）
    """
    p_base_percent = 30.0  # 与 integrator 缺省一致的保守默认
    is_trained = False
    model_performance = None

    if ml_predictor is not None:
        try:
            pred = ml_predictor.predict_risk(contact_data, contact_type)
            if pred and 'ensemble' in pred:
                p_base_percent = float(
                    pred['ensemble'].get('risk_probability', 30.0))
                is_trained = bool(getattr(ml_predictor, 'is_trained', False))
                model_performance = getattr(ml_predictor, 'model_performance', None)
        except Exception as e:  # 单层失败不阻断整条流水线
            LOGGER.debug("第 1 层个体基线预测失败，使用默认值: %s", e)

    p_base_percent = float(np.clip(p_base_percent, 0.0, 100.0))
    return {
        'p_base': p_base_percent / 100.0,
        'p_base_percent': p_base_percent,
        'risk_class': 1 if p_base_percent > 50 else 0,
        'is_trained': is_trained,
        'model_performance': model_performance,
        'layer': 1,
        'layer_name': '个体基础层 (ML)',
    }


# ==============================================================================
# 第 2 层路径 A：时序家庭先验（部署真实实现，2026-08-25 第四轮 P1）
# ==============================================================================

def compute_temporal_household_increment(p_base_percent, screened_results,
                                         base_rate,
                                         weight=DEFAULT_TEMPORAL_WEIGHT,
                                         shrinkage=DEFAULT_TEMPORAL_SHRINKAGE):
    """第 2 层路径 A：时序家庭先验网络增量。

    部署语义：同户已筛查成员结果（screened_results）→ 贝叶斯收缩
    户级率 r̂ → logit 空间上调其余成员风险。首筛查者（n=0）增量为 0
    （回退个体基线）。验证依据与诚实边界见
    ``scoring.temporal_household`` 模块 docstring。

    返回 dict 与 GNN 路径同构（network_risk / network_increment /
    network_aware / backend / residual_learning / n_contacts），
    额外携带 prior_features（四列先证特征）与
    household_rate_estimate（收缩户级率）。
    """
    increment = temporal_increment(
        p_base_percent, screened_results, base_rate,
        weight=weight, shrinkage=shrinkage)
    prior = temporal_prior_features(screened_results, base_rate)
    n_screened = int(prior['prior_n'])
    updated = float(np.clip(p_base_percent + increment, 0.0, 100.0))
    return {
        'baseline_probability': float(p_base_percent),
        'network_risk': updated,
        'network_increment': float(increment),
        'network_aware': n_screened > 0,
        'backend': 'temporal_household',
        'residual_learning': True,
        'n_contacts': n_screened,
        'prior_features': prior,
        'household_rate_estimate': household_rate_estimate(
            screened_results, base_rate, shrinkage=shrinkage),
        'temporal_weight': float(weight),
        'temporal_shrinkage': float(shrinkage),
        'gating_mode': 'evidence',
    }


# ==============================================================================
# 第 2 层路径 B：网络增强层（GNN 残差增量）
# ==============================================================================

def _numpy_network_increment(contact_data, contact_type, p_base_percent,
                             tb_assessment=None):
    """纯 NumPy 网络增量兜底：用轻量图卷积网络风险 − 个体基线 = 网络增量。

    无 torch/PyG 或预测器不可用时，通过 ``NumPyGraphRiskPredictor`` 构建
    接触网络并计算网络感知风险，再求与个体基线的残差，保证第 2 层在
    任何环境下都有确定性的输出。

    返回 dict（与完整 GNN 路径同构）。
    """
    try:
        from ..ml.gnn import NumPyGraphRiskPredictor
        predictor_np = NumPyGraphRiskPredictor()
        family_entries = getattr(tb_assessment, 'family_entries', None) or []
        social_entries = getattr(tb_assessment, 'social_entries', None) or []
        patient_info = getattr(tb_assessment, 'patient_info', None)

        if family_entries or social_entries:
            result = predictor_np.predict_network_risk(
                contact_data, contact_type, patient_info,
                family_entries, social_entries, p_base=p_base_percent)
        else:
            result = predictor_np.predict_contact_risk(
                contact_data, contact_type, p_base=p_base_percent)
    except Exception as e:
        LOGGER.debug("NumPy 网络增量兜底失败: %s", e)
        # 无网络时网络增量为 0（个体风险即全部风险）
        return {
            'network_risk': p_base_percent,
            'network_increment': 0.0,
            'baseline_probability': p_base_percent,
            'network_aware': False,
            'backend': 'fallback',
            'residual_learning': True,
            'degradation': 'numpy_failed',
        }

    if result is None:
        return {
            'network_risk': p_base_percent,
            'network_increment': 0.0,
            'baseline_probability': p_base_percent,
            'network_aware': False,
            'backend': 'fallback',
            'residual_learning': True,
        }
    network_risk = float(result.get('risk_probability', p_base_percent))
    increment = network_risk - p_base_percent
    return {
        'network_risk': network_risk,
        'network_increment': increment,
        'baseline_probability': p_base_percent,
        'network_aware': bool(result.get('network_aware', False)),
        'backend': result.get('backend', 'numpy'),
        'residual_learning': True,
    }


def compute_network_increment(ml_predictor, tb_assessment, contact_data,
                              contact_type='family', p_base=None,
                              use_gnn=True, screened_results=None,
                              household_base_rate=None,
                              temporal_weight=DEFAULT_TEMPORAL_WEIGHT,
                              temporal_shrinkage=DEFAULT_TEMPORAL_SHRINKAGE):
    """第 2 层：计算网络增量风险。

    路径 A（screened_results 非 None，优先）：时序家庭先验——
    户内已筛查结果 → 贝叶斯收缩户级率 → logit 上调。部署真实实现
    （HomeACF 验证 +0.0585 显著）；此时 ``household_base_rate`` 必填
    （人群基线率，部署口径由调用方提供）。

    路径 B（screened_results 为 None）：GNN 残差（原路径）。
    残差语义：``网络增量 = 网络感知风险 − 个体基线概率``。
    无网络结构时增量自然趋于 0（隔离个体不因网络加分），
    避免"网络层与个体层重复计分"。

    参数：
        ml_predictor: MLRiskPredictor（提供 predict_gnn_risk）
        tb_assessment: 结核病风险评估实例（含 family/social 接触者列表）
        contact_data (dict): 单个接触者数据
        contact_type (str): 'family' / 'social'
        p_base (float|None): 0-100 个体基线概率；None 时用第 1 层计算
        use_gnn (bool): 路径 B 是否尝试完整 GNN；False 时直接用 NumPy 兜底
        screened_results (sequence[int]|None): 同户已筛查成员结果（1=阳性）
        household_base_rate (float|None): 队列/人群基线率（路径 A 必填）
        temporal_weight (float): 路径 A logit 上调权重 w
        temporal_shrinkage (float): 路径 A 收缩伪计数 k

    返回：
        dict:
            - 'baseline_probability': 0-100 个体基线
            - 'network_risk': 0-100 网络感知风险（有网络时）
            - 'network_increment': 网络增量（可为负，表示网络未增强）
            - 'network_aware': 是否使用了网络结构
            - 'residual_learning': 是否残差语义
            - 'backend': 'temporal_household' / 'pytorch' / 'numpy' / 'fallback'
            - 'n_contacts': 网络证据规模（路径 A = 已筛查人数）
    """
    if p_base is None:
        base_result = compute_individual_base(ml_predictor, contact_data, contact_type)
        p_base = base_result['p_base_percent']

    # ---- 路径 A：时序家庭先验（部署真实实现，优先） ----
    if screened_results is not None:
        if household_base_rate is None:
            raise ValueError(
                'screened_results 提供时必须同时提供 household_base_rate'
                '（人群基线率——部署口径的先证回退值）')
        return compute_temporal_household_increment(
            p_base, screened_results, household_base_rate,
            weight=temporal_weight, shrinkage=temporal_shrinkage)

    # ---- 路径 B：GNN 残差（无筛查状态时的原路径） ----
    # 尝试完整 GNN（predict_gnn_risk 内部会处理 PyG 路径与 NumPy 回退）
    if use_gnn and ml_predictor is not None:
        gnn_fn = getattr(ml_predictor, 'predict_gnn_risk', None)
        if gnn_fn is not None:
            try:
                gnn_result = gnn_fn(tb_assessment, contact_data, contact_type)
                if gnn_result is not None and gnn_result.get('risk_probability') is not None:
                    network_risk = float(gnn_result['risk_probability'])
                    return {
                        'baseline_probability': p_base,
                        'network_risk': network_risk,
                        'network_increment': network_risk - p_base,
                        'network_aware': bool(gnn_result.get('network_aware', False)),
                        'backend': gnn_result.get('backend',
                                                  'pytorch' if gnn_result.get('gnn_used') else 'numpy'),
                        'residual_learning': True,
                        'n_contacts': _count_contacts(tb_assessment),
                        'degradation': gnn_result.get('degradation'),
                    }
            except Exception as e:
                LOGGER.debug("第 2 层 GNN 预测异常，回退 NumPy: %s", e)

    # 兜底：NumPy 轻量图卷积（确定性）
    result = _numpy_network_increment(
        contact_data, contact_type, p_base, tb_assessment)
    result['n_contacts'] = _count_contacts(tb_assessment)
    return result


def _count_contacts(tb_assessment):
    """统计接触网络规模（家庭 + 社会接触者数量）。"""
    if tb_assessment is None:
        return 0
    n_family = len(getattr(tb_assessment, 'family_entries', None) or [])
    n_social = len(getattr(tb_assessment, 'social_entries', None) or [])
    return n_family + n_social


# ==============================================================================
# 高风险集合选择（第 1、2 层输出 → 第 3 层输入）
# ==============================================================================

def select_high_risk_set(individual_results, threshold=DEFAULT_HIGH_RISK_THRESHOLD):
    """从个体级评估结果中筛选高风险个体集合（供第 3 层干预模拟使用）。

    判定依据：决策风险 = 个体概率 + 网络增量（含负增量时用个体概率保底）。

    阈值语义（M 级修复澄清）：默认 ``DEFAULT_HIGH_RISK_THRESHOLD`` 引用
    ``constants.DISEASE_PROB_VERY_HIGH``（= ``threshold_spec.DEFAULT_VERY_HIGH``），
    对应四档分级中的【极高风险】切点（≥15）——threshold_spec 的"高风险"
    档切点为 8（``DISEASE_PROB_HIGH``）。本函数以极高风险切点作为进入
    第 3 层干预模拟的保守下限；如需纳入高风险档（≥8），显式传
    ``threshold=8.0`` 或引用 ``constants.DISEASE_PROB_HIGH``。

    参数：
        individual_results: 个体评估结果列表；每项含
            'record_id'/'p_base_percent'/'network_increment'/'decision_risk'（可选）
        threshold (float): 干预集合阈值（0-100 尺度，默认 15 = 极高风险切点）

    返回：
        list[dict]: 高风险个体子集（含决策风险与来源标注）
    """
    high_risk = []
    for rec in individual_results or []:
        if not isinstance(rec, dict):
            continue
        p_base = float(rec.get('p_base_percent', 0.0))
        incr = float(rec.get('network_increment', 0.0))
        decision = float(rec.get('decision_risk',
                                 max(p_base, p_base + incr)))
        if decision >= threshold:
            high_risk.append({
                'record_id': rec.get('record_id', rec.get('_id', 'unknown')),
                'p_base_percent': p_base,
                'network_increment': incr,
                'decision_risk': decision,
                'source': rec.get('layer_source', 'individual+network'),
            })
    return high_risk


# ==============================================================================
# 第 3 层：社区干预层（SEIR 反事实模拟）
# ==============================================================================

def simulate_community_intervention(high_risk_contacts=None, population=10000,
                                    t_horizon_days=730, strategies=None,
                                    random_state=42, **kwargs):
    """第 3 层：SEIR 社区干预反事实模拟（人群层，不输出个体概率）。

    给定第 1、2 层识别的高风险个体集合，用 ``seir.intervention`` 的
    ``InterventionCounterfactualSimulator`` 模拟基线 vs 各干预策略在
    6-24 个月水平内的病例数变化，输出人群层干预效果曲线。

    参数：
        high_risk_contacts: 高风险个体集合（list/dict），用于推算覆盖率
        population (int): 社区总人口
        t_horizon_days (int): 模拟水平（默认 730 天 = 24 个月）
        strategies (list|None): 干预策略列表（None = 全部三种）
        random_state (int): 随机种子
        **kwargs: 透传（beta/coverage/dt/initial_state 等）

    返回：
        dict: 人群层干预效果报告（见 seir.intervention.simulate_intervention_effects）
    """
    try:
        from ..seir.intervention import simulate_intervention_effects
        report = simulate_intervention_effects(
            high_risk_contacts=high_risk_contacts,
            population=population,
            t_horizon_days=t_horizon_days,
            strategies=strategies,
            random_state=random_state,
            **kwargs,
        )
        report['layer'] = '社区干预层 (SEIR 反事实模拟)'
        report['layer_name'] = '社区干预层 (SEIR)'
        report['level'] = 3
        return report
    except Exception as e:
        LOGGER.debug("第 3 层干预模拟失败: %s", e)
        return {
            'level': 3,
            'layer': '社区干预层 (SEIR 反事实模拟)',
            'layer_name': '社区干预层 (SEIR)',
            'status': 'error',
            'error': str(e),
            'n_high_risk': len(high_risk_contacts) if high_risk_contacts else 0,
        }


# ==============================================================================
# 决策层：堆叠 + 门控整合（替代加权平均）
# ==============================================================================

def _gating_signal(n_contacts, network_aware, residual_learning=True,
                   gating_mode='contact_count'):
    """门控信号：网络证据强度 → 网络增量的生效程度 g ∈ [0, 1]。

    规则（gating_mode='contact_count'，GNN 残差路径默认）：
      - 无网络结构（n_contacts == 0 或非 network_aware）→ g = 0，
        网络增量不生效（避免无网络时凭空加分）；
      - 有网络 → 随网络规模单调上升并饱和到 DEFAULT_GATE_MAX。

    规则（gating_mode='evidence'，时序家庭先验路径）：
      - 增量的不确定性已由贝叶斯收缩 r̂ = (pos + k·π)/(n + k)
        内建（证据越少越接近基线 → 增量自然趋 0），不再按接触数
        二次衰减 → 有证据时 g = 1，无证据时 g = 0。

    参数：
        n_contacts (int): 接触网络规模
        network_aware (bool): 本次预测是否实际使用了网络结构
        gating_mode (str): 'contact_count' / 'evidence'

    返回：
        float: 门控系数 g
    """
    if not network_aware or n_contacts <= 0:
        return DEFAULT_GATE_MIN
    if gating_mode == 'evidence':
        return DEFAULT_GATE_MAX
    # 网络规模 1-20 线性抬升，之后饱和
    # ⚠ 2026-09-05 地基审计：饱和点 20 为手调常数，未经真实数据验证
    g = DEFAULT_GATE_MIN + (DEFAULT_GATE_MAX - DEFAULT_GATE_MIN) * min(
        n_contacts / 20.0, 1.0)
    return float(np.clip(g, DEFAULT_GATE_MIN, DEFAULT_GATE_MAX))


def _normalize_weights(weights):
    """归一化权重 dict（总和 = 1），非法时回退均匀。"""
    total = sum(float(v) for v in weights.values() if v is not None)
    if total <= 0:
        n = max(len(weights), 1)
        return {k: 1.0 / n for k in weights}
    return {k: (float(v) / total if v is not None else 0.0)
            for k, v in weights.items()}


def gated_integration(p_base_percent, network_increment, n_contacts=0,
                      network_aware=False, intervention=None,
                      decision_weights=None, gating_mode='contact_count'):
    """决策层：堆叠 + 门控联合决策。

    替代旧"概率层面线性平均"：
      1. **门控网络增量**：``gated_network = g × network_increment``，
         其中门控系数 g 由网络证据强度（规模/awareness）决定；
      2. **个体概率 + 网络增量**（语义清晰的"个体风险 + 网络增量"）：
         ``combined = clip(p_base + gated_network, 0, 100)``；
      3. **干预收益修正**（第 3 层输出，仅作联合决策微调）：
         高风险个体纳入干预的"可避免病例收益"以门控方式叠加。

    参数：
        p_base_percent (float): 0-100 个体基线概率（第 1 层）
        network_increment (float): 网络增量（第 2 层）
        n_contacts (int): 接触网络规模
        network_aware (bool): 是否实际使用了网络结构
        intervention (dict|None): 第 3 层干预报告（含 best_avoidable_case_days 等）
        decision_weights (dict|None): 决策权重（默认见 DEFAULT_DECISION_WEIGHTS）
        gating_mode (str): 'contact_count'（GNN 路径）/ 'evidence'
            （时序先验路径——增量已内建收缩，不按接触数二次衰减）

    返回：
        dict:
            - 'gating': 门控系数 g
            - 'gated_network_increment': 门控后的网络增量
            - 'combined_probability': 个体概率 + 网络增量（0-100）
            - 'intervention_benefit': 干预收益决策修正（0-100 加分）
            - 'decision_risk': 最终决策风险（0-100）
            - 'decision_class': 二分类
            - 'integration_method': 'stacking_gating'
    """
    g = _gating_signal(n_contacts, network_aware,
                       gating_mode=gating_mode)
    gated_network = float(network_increment) * g

    combined = float(np.clip(p_base_percent + gated_network, 0.0, 100.0))

    # 干预收益（第 3 层）→ 决策修正：仅当该个体属于高风险集合时才加分
    intervention_benefit = 0.0
    if intervention is not None:
        avoidable = intervention.get('best_avoidable_case_days') or \
            intervention.get('summary', {}).get('best_avoidable_case_days')
        if avoidable:
            intervention_benefit = min(
                INTERVENTION_BENEFIT_GAIN * float(avoidable), 5.0)

    # 决策层联合（非概率层面线性平均）：
    #   - 网络增量已通过门控 g 叠加进 combined（个体概率 + 网络增量），
    #     不再作为独立概率项重复计分 → 消除"重复计分"批评；
    #   - 干预收益（第 3 层人群层结论）以决策修正项参与联合决策。
    # 权重归一化分两步（有意为之，非冗余）：
    #   1) _normalize_weights 对全部键归一（含调用方可能传入的额外键）；
    #   2) 本层只使用 individual/intervention 两键，按其子集和再归一
    #      ——默认两键时第二步为恒等变换（分母=1），有额外键时保证
    #      使用键比例正确。
    w = _normalize_weights(decision_weights or DEFAULT_DECISION_WEIGHTS)
    indiv_w = w.get('individual', 0.6)
    inter_w = w.get('intervention', 0.1)
    denom = indiv_w + inter_w
    if denom <= 0:
        # 两使用键全零：回退默认比例，避免除零
        indiv_w, inter_w, denom = 0.6, 0.1, 0.7
    decision_risk = (indiv_w * combined + inter_w * intervention_benefit) / denom
    decision_risk = float(np.clip(decision_risk, 0.0, 100.0))

    return {
        'gating': float(g),
        'gated_network_increment': float(gated_network),
        'combined_probability': float(combined),
        'intervention_benefit': float(intervention_benefit),
        'decision_risk': decision_risk,
        'decision_class': 1 if decision_risk > DEFAULT_RISK_CLASS_THRESHOLD else 0,
        'integration_method': 'stacking_gating',
    }


# ==============================================================================
# 三层流水线编排
# ==============================================================================

class ThreeLayerArchitecture:
    """三层递进架构编排器。

    用法：
        arch = ThreeLayerArchitecture()
        result = arch.run_pipeline(ml_predictor, tb_assessment, contact_data,
                                   contact_type='family')

    各层可独立调用（``compute_individual_base`` / ``compute_network_increment`` /
    ``simulate_community_intervention``），亦可在 ``decision_layer`` 单独整合。
    """

    def __init__(self, high_risk_threshold=DEFAULT_HIGH_RISK_THRESHOLD,
                 decision_weights=None):
        self.high_risk_threshold = high_risk_threshold
        self.decision_weights = dict(
            decision_weights or DEFAULT_DECISION_WEIGHTS)

    # ---- 单层入口 ----
    def individual_base(self, ml_predictor, contact_data, contact_type='family'):
        """第 1 层：个体基线概率。"""
        return compute_individual_base(ml_predictor, contact_data, contact_type)

    def network_enhancement(self, ml_predictor, tb_assessment, contact_data,
                            contact_type='family', p_base=None, use_gnn=True,
                            screened_results=None, household_base_rate=None,
                            temporal_weight=DEFAULT_TEMPORAL_WEIGHT,
                            temporal_shrinkage=DEFAULT_TEMPORAL_SHRINKAGE):
        """第 2 层：网络增量（时序先验优先，否则 GNN 残差）。"""
        return compute_network_increment(
            ml_predictor, tb_assessment, contact_data, contact_type,
            p_base=p_base, use_gnn=use_gnn,
            screened_results=screened_results,
            household_base_rate=household_base_rate,
            temporal_weight=temporal_weight,
            temporal_shrinkage=temporal_shrinkage)

    def community_intervention(self, high_risk_contacts, **kwargs):
        """第 3 层：社区干预反事实模拟。"""
        return simulate_community_intervention(
            high_risk_contacts, **kwargs)

    def decision_layer(self, base, network, intervention=None):
        """决策层：堆叠 + 门控整合三层输出。"""
        return gated_integration(
            p_base_percent=base['p_base_percent'],
            network_increment=network['network_increment'],
            n_contacts=network.get('n_contacts', 0),
            network_aware=network.get('network_aware', False),
            intervention=intervention,
            decision_weights=self.decision_weights,
            gating_mode=network.get('gating_mode', 'contact_count'),
        )

    # ---- 全流水线 ----
    def run_pipeline(self, ml_predictor, tb_assessment, contact_data,
                     contact_type='family', run_intervention=True,
                     population=10000, t_horizon_days=730, random_state=42,
                     screened_results=None, household_base_rate=None,
                     temporal_weight=DEFAULT_TEMPORAL_WEIGHT,
                     temporal_shrinkage=DEFAULT_TEMPORAL_SHRINKAGE):
        """端到端运行三层流水线。

        第 1 层 → P_base；第 2 层 → 网络增量（screened_results
        提供时走时序家庭先验路径，否则 GNN 残差）；第 3 层 → 干预
        收益；决策层 → 联合决策风险。

        参数：
            ml_predictor: MLRiskPredictor
            tb_assessment: 风险评估实例
            contact_data (dict): 目标接触者数据
            contact_type (str): 'family' / 'social'
            run_intervention (bool): 是否运行第 3 层（较重，默认 True）
            population / t_horizon_days / random_state: 第 3 层参数
            screened_results (sequence[int]|None): 同户已筛查成员结果
                （1=阳性）——第 2 层时序先证路径入口
            household_base_rate (float|None): 人群基线率（路径 A 必填）
            temporal_weight / temporal_shrinkage: 时序先证参数 w / k

        返回：
            dict: 三层流水线完整报告（layers / decision / summary）
        """
        # 第 1 层
        base = self.individual_base(ml_predictor, contact_data, contact_type)
        # 第 2 层（以第 1 层输出为特征 → 堆叠）
        network = self.network_enhancement(
            ml_predictor, tb_assessment, contact_data, contact_type,
            p_base=base['p_base_percent'],
            screened_results=screened_results,
            household_base_rate=household_base_rate,
            temporal_weight=temporal_weight,
            temporal_shrinkage=temporal_shrinkage)
        # 高风险集合（第 1 + 2 层）
        individual_results = [{
            'record_id': contact_data.get('record_id') if isinstance(contact_data, dict) else None,
            'p_base_percent': base['p_base_percent'],
            'network_increment': network['network_increment'],
        }]
        high_risk = select_high_risk_set(
            individual_results, threshold=self.high_risk_threshold)

        # 第 3 层（以高风险集合为输入）
        intervention = None
        if run_intervention:
            intervention = self.community_intervention(
                high_risk_contacts=high_risk,
                population=population, t_horizon_days=t_horizon_days,
                random_state=random_state)

        # 决策层
        decision = self.decision_layer(base, network, intervention)

        return self._build_report(base, network, intervention, decision,
                                  high_risk, contact_data)

    @staticmethod
    def _build_report(base, network, intervention, decision, high_risk,
                      contact_data):
        """聚合三层报告。"""
        layers = {
            'layer1_individual': base,
            'layer2_network': network,
            'layer3_intervention': intervention,
        }
        summary = {
            'baseline_probability': base['p_base_percent'],
            'network_increment': network['network_increment'],
            'combined_probability': decision['combined_probability'],
            'decision_risk': decision['decision_risk'],
            'decision_class': decision['decision_class'],
            'integration_method': decision['integration_method'],
            'gating': decision['gating'],
            'n_high_risk': len(high_risk),
            'network_aware': network.get('network_aware', False),
        }
        if intervention is not None and intervention.get('summary'):
            summary['best_strategy'] = intervention['summary'].get('best_strategy_label')
            summary['best_averted_percent'] = intervention['summary'].get(
                'best_averted_percent')
        return {
            'architecture': 'three_layer',
            'layers': layers,
            'decision': decision,
            'high_risk_set': high_risk,
            'summary': summary,
            'target': {
                'record_id': contact_data.get('record_id')
                if isinstance(contact_data, dict) else None,
                'contact_type': contact_data.get('contact_type')
                if isinstance(contact_data, dict) else None,
            },
        }


def run_three_layer_pipeline(ml_predictor, tb_assessment, contact_data,
                             contact_type='family', **kwargs):
    """模块级便捷入口（等价 ThreeLayerArchitecture().run_pipeline）。"""
    arch = ThreeLayerArchitecture()
    return arch.run_pipeline(ml_predictor, tb_assessment, contact_data,
                             contact_type=contact_type, **kwargs)


__all__ = [
    'DEFAULT_HIGH_RISK_THRESHOLD',
    'DEFAULT_GATE_MIN',
    'DEFAULT_GATE_MAX',
    'DEFAULT_DECISION_WEIGHTS',
    'INTERVENTION_BENEFIT_GAIN',
    'compute_individual_base',
    'compute_temporal_household_increment',
    'compute_network_increment',
    'select_high_risk_set',
    'simulate_community_intervention',
    'gated_integration',
    'ThreeLayerArchitecture',
    'run_three_layer_pipeline',
]
