#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""结核病（TB）领域默认因果有向无环图

基于 tb_risk 项目 ML 特征体系（scoring/predictor.py 的 FEATURE_NAMES）与
constants.py 中文献来源编码变量间的因果关系结构。

节点命名与 ML 特征名严格对齐，使反事实分析能直接复用 predict_risk。

文献支撑（与 constants.py 同源）：
- WHO TB Technical Guidelines (2020): BCG 保护效力 ~50%
- Colditz et al., JAMA 1994: BCG 保护效力 meta 分析
- Trunz et al., Bull WHO 2021: BCG 衰减动力学
- Dooley & Chaisson, Lancet ID 2009: 糖尿病进展风险 2.5 倍
- Lawn et al., AIDS 2009: HIV 进展风险 8 倍
- Marais et al., IJTLD 2011: <5岁进展风险为成人 4 倍
- Davies et al., JRSM 2006: >65岁进展风险翻倍
- Escombe et al., PLoS Med 2009: 自然通风降低 TB 传播 >70%
- China CDC TB Annual Report (2023): 既往结核史复发风险

因果假设说明：
本 DAG 表达的是变量间的因果生成机制假设，而非统计相关。
DAG 是因果推断的可证伪前提——若假设错误，后续 do-calculus 估计的干预效应
也会有偏。临床使用者应基于领域知识审查并按需修改边结构。
"""
import logging

from .dag import CausalDAG, NodeType

LOGGER = logging.getLogger("tb_risk.validation.causal.tb_dag")

# 默认暴露与结局变量名（与 ML 特征名对齐）
DEFAULT_EXPOSURE = 'cumulative_exposure'
DEFAULT_OUTCOME = 'disease_probability'

# 结局变量名（模型输出，非特征）
OUTCOME_NODE = 'disease_probability'


def build_tb_dag() -> CausalDAG:
    """构建结核病风险评估默认因果 DAG。

    节点角色：
    - 暴露：cumulative_exposure（累积暴露时长，可干预：减少接触）
    - 结局：disease_probability（发病风险概率，模型输出）
    - 混杂：age, bcg_vaccine, has_tb, past_illness, contact_distance_score,
            ventilation_score, exposure_setting_score, is_high_risk
    - 中介：has_symptoms（暴露→症状→发病）
    - 暴露决定因素：single_duration, freq_density, time_span（决定累积暴露量）

    返回：
        CausalDAG: 已编码 TB 领域因果结构的 DAG 实例
    """
    dag = CausalDAG()

    # ==================== 节点定义 ====================
    # 人口学与既往史（混杂因素）
    dag.add_node('age', NodeType.CONFOUNDER, '年龄（影响接种、合并症与进展）')
    dag.add_node('bcg_vaccine', NodeType.CONFOUNDER, '卡介苗接种（保护性因素）')
    dag.add_node('has_tb', NodeType.CONFOUNDER, '既往结核史（复发风险）')
    dag.add_node('past_illness', NodeType.CONFOUNDER, '慢性病史/免疫抑制（糖尿病/HIV）')

    # 暴露决定因素（影响累积暴露量）
    dag.add_node('contact_distance_score', NodeType.CONFOUNDER, '接触距离评分')
    dag.add_node('ventilation_score', NodeType.CONFOUNDER, '通风条件评分')
    dag.add_node('exposure_setting_score', NodeType.CONFOUNDER, '暴露场景评分')
    dag.add_node('single_duration', NodeType.OBSERVED, '单次接触时长（分钟）')
    dag.add_node('freq_density', NodeType.OBSERVED, '每周接触频次')
    dag.add_node('time_span', NodeType.OBSERVED, '持续周期（周）')

    # 高危人群标签（由既往史决定）
    dag.add_node('is_high_risk', NodeType.CONFOUNDER, '高危人群标签')

    # 暴露（干预对象）
    dag.add_node('cumulative_exposure', NodeType.EXPOSURE,
                 '累积暴露时长（小时）— 可干预')

    # 中介：暴露→感染→症状→发病
    dag.add_node('has_symptoms', NodeType.MEDIATOR, '有症状（暴露→发病中介）')

    # 结局
    dag.add_node(OUTCOME_NODE, NodeType.OUTCOME, '发病风险概率（模型输出）')

    # ==================== 因果边 ====================
    # --- 人口学 → 接种/合并症/结局 ---
    # 年龄影响 BCG 接种状态（年长队列接种率低）与 BCG 衰减（Trunz 2021）
    dag.add_edge('age', 'bcg_vaccine',
                 '年长队列 BCG 接种率低；保护效力随年龄衰减')
    # 年龄影响合并症发生（糖尿病风险随年龄上升，Dooley 2009）
    dag.add_edge('age', 'past_illness', '年龄→糖尿病等合并症')
    # 年龄直接影响进展（Marais 2011: <5岁4倍；Davies 2006: >65岁翻倍）
    dag.add_edge('age', OUTCOME_NODE, '年龄进展因子（<5岁/>65岁高风险）')

    # BCG 直接保护（WHO 2020: ~50% 保护效力）
    dag.add_edge('bcg_vaccine', OUTCOME_NODE, 'BCG 保护效力（Colditz 1994）')

    # 既往结核史 → 复发（China CDC 2023）
    dag.add_edge('has_tb', OUTCOME_NODE, '既往结核史复发风险')

    # 合并症 → 高危标签 + 直接进展
    dag.add_edge('past_illness', 'is_high_risk', '合并症定义高危人群')
    dag.add_edge('past_illness', OUTCOME_NODE,
                 '糖尿病2.5倍/HIV 8倍进展风险（Dooley 2009, Lawn 2009）')
    dag.add_edge('is_high_risk', OUTCOME_NODE, '高危人群标签效应')

    # --- 暴露决定因素 → 累积暴露 ---
    # 接触距离、通风、场景决定单次暴露强度（Escombe 2009: 通风降低>70%）
    dag.add_edge('contact_distance_score', 'cumulative_exposure',
                 '距离越近暴露越强')
    dag.add_edge('ventilation_score', 'cumulative_exposure',
                 '通风差暴露累积更快（Escombe 2009）')
    dag.add_edge('exposure_setting_score', 'cumulative_exposure',
                 '场景影响暴露强度')
    # 时长/频次/周期 → 累积暴露（剂量-反应）
    dag.add_edge('single_duration', 'cumulative_exposure', '单次时长累加')
    dag.add_edge('freq_density', 'cumulative_exposure', '频次累加')
    dag.add_edge('time_span', 'cumulative_exposure', '周期累加')

    # 暴露决定因素同时直接影响结局（距离/通风本身降低感染概率）
    dag.add_edge('contact_distance_score', OUTCOME_NODE,
                 '距离直接影响飞沫传播')
    dag.add_edge('ventilation_score', OUTCOME_NODE,
                 '通风直接降低飞沫浓度')

    # --- 暴露 → 中介 → 结局 ---
    # 累积暴露 → 感染 → 症状（暴露剂量-反应）
    dag.add_edge('cumulative_exposure', 'has_symptoms',
                 '暴露→感染→症状出现')
    # 症状 → 发病判定（症状是发病的临床指征）
    dag.add_edge('has_symptoms', OUTCOME_NODE, '症状指示活动性发病')

    # 累积暴露直接→结局（亚临床感染）
    dag.add_edge('cumulative_exposure', OUTCOME_NODE,
                 '累积暴露剂量-反应（直接感染概率）')

    LOGGER.debug("TB 因果 DAG 构建完成: %s", dag.summary())
    return dag


def get_default_adjustment_set(exposure: str = DEFAULT_EXPOSURE,
                               outcome: str = DEFAULT_OUTCOME):
    """获取 TB 领域默认暴露-结局对的充分调整集。

    返回 (adjustment_set, dag) 元组。若自动求解失败，返回 (None, dag)。
    """
    dag = build_tb_dag()
    adj = dag.find_adjustment_set(exposure, outcome)
    if adj is None:
        LOGGER.warning(
            "默认 TB DAG 无法自动求解调整集 (exposure=%s, outcome=%s)；"
            "请人工审查后门路径", exposure, outcome)
    return adj, dag
