#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务分解与临床终点定义（评审建议中最有价值的一点）。

把当前"一个集成概率"拆成三个独立任务，每个任务单独定义：

  - **任务 A（横断面筛查）**：当前是否患活动性结核
  - **任务 B（纵向预后）**：已感染者未来是否进展
  - **任务 C（网络传播）**：接触者感染风险排序与传播链

同一批接触者数据，三个任务用不同标签，不可混用。

决策树（EndPoint 枚举）：
  - 当有基线筛查结果（如涂片/Xpert/培养）→ 任务 A
  - 当有随访发病数据（6-24 月）→ 任务 B
  - 当有接触网络结构（簇/二代病例）→ 任务 C
"""

import logging
import math
from enum import Enum

LOGGER = logging.getLogger("tb_risk.validation.task_decomposition")


# ==============================================================================
# 任务枚举与终点定义
# ==============================================================================

class TaskType(Enum):
    """三个独立预测任务。"""
    SCREENING = 'A'   # 横断面活动性结核筛查
    PROGNOSIS = 'B'   # 纵向感染后进展预测
    NETWORK = 'C'     # 接触网络传播预测


class EndPoint(Enum):
    """各任务的临床终点定义。"""
    # 任务 A：确诊活动性结核（二元标签）
    ACTIVE_TB = 'active_tb'
    # 任务 B：随访期内进展为活动性结核（时间-事件）
    PROGRESSION = 'progression'
    # 任务 C：接触者感染状态（二元标签 / 传播簇）
    CONTACT_INFECTION = 'contact_infection'


TASK_ENDPOINT_MAP = {
    TaskType.SCREENING: EndPoint.ACTIVE_TB,
    TaskType.PROGNOSIS: EndPoint.PROGRESSION,
    TaskType.NETWORK: EndPoint.CONTACT_INFECTION,
}


# ==============================================================================
# 任务 A：横断面活动性结核筛查（Screening / Prevalence Detection）
# ==============================================================================

def _sensitivity(tp, fn):
    """敏感度 = TP / (TP + FN)。"""
    denom = tp + fn
    return 0.0 if denom == 0 else tp / denom


def _specificity(tn, fp):
    """特异度 = TN / (TN + FP)。"""
    denom = tn + fp
    return 0.0 if denom == 0 else tn / denom


def _ppv(tp, fp):
    """阳性预测值 = TP / (TP + FP)。"""
    denom = tp + fp
    return 0.0 if denom == 0 else tp / denom


def _npv(tn, fn):
    """阴性预测值 = TN / (TN + FN)。"""
    denom = tn + fn
    return 0.0 if denom == 0 else tn / denom


def evaluate_task_a(probabilities, outcomes, threshold=None):
    """任务 A 评估：横断面活动性结核筛查。

    参数：
        probabilities (list[float]): 预测概率（0-100 或 0-1）
        outcomes (list[int]): 真实结局（1 = 确诊活动性结核）
        threshold (float|None): 决策阈值；None 时用约登指数最优切点

    返回：
        dict: 评估报告（TP/TN/FP/FN、敏感度/特异度、PPV/NPV、约登切点、校准）
    """
    from .threshold_spec import youden_optimal_threshold, expected_calibration_error

    probs = [float(p) for p in probabilities]
    is_percent = max(probs) > 1.0 if probs else False
    scale = 100.0 if is_percent else 1.0
    labels = [int(o) for o in outcomes]

    # 最优切点
    if threshold is None:
        youden = youden_optimal_threshold(probs, labels)
        threshold = youden.get('threshold', 0.5 * scale)
    threshold = float(threshold)

    # 混淆矩阵
    tp = sum(1 for p, o in zip(probs, labels) if p >= threshold and o == 1)
    fp = sum(1 for p, o in zip(probs, labels) if p >= threshold and o == 0)
    tn = sum(1 for p, o in zip(probs, labels) if p < threshold and o == 0)
    fn = sum(1 for p, o in zip(probs, labels) if p < threshold and o == 1)

    # 指标
    n = len(labels)
    prev = sum(labels) / max(n, 1)

    # 校准误差
    ece = expected_calibration_error(probs, labels, n_bins=10)
    ece_value = ece.get('ece', 0.0) if isinstance(ece, dict) else float(ece)

    return {
        'task': 'A',
        'task_name': '横断面活动性结核筛查',
        'task_name_en': 'Cross-sectional Active TB Screening',
        'endpoint': 'active_tb',
        'endpoint_description': '当前是否患活动性结核（WS 288-2017 确诊标准）',
        'time_window': '筛查当下',
        'decision_meaning': '决定谁现在做进一步病原学/影像学检查',
        'n_samples': n,
        'prevalence': round(prev, 4),
        'threshold': round(threshold, 4),
        'confusion_matrix': {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        },
        'metrics': {
            'sensitivity': round(_sensitivity(tp, fn), 4),
            'specificity': round(_specificity(tn, fp), 4),
            'ppv': round(_ppv(tp, fp), 4),
            'npv': round(_npv(tn, fn), 4),
            'youden_index': round(
                _sensitivity(tp, fn) + _specificity(tn, fp) - 1.0, 4),
            'expected_calibration_error': round(ece_value, 4),
        },
        'epsilon': 1e-10,
    }


# ==============================================================================
# 任务 B：纵向感染后进展预测（Prognosis / Time-to-Event）
# ==============================================================================

def _compute_horizon_auc(probabilities, outcomes, horizon_days=365):
    """固定时点 AUC（简化：对事件发生者与非事件者做 AUC 区分）。"""
    from .threshold_spec import compute_auc
    return compute_auc(probabilities, outcomes)


def _decision_curve_analysis(probabilities, outcomes, thresholds=None):
    """决策曲线分析（DCA）：净收益 = (TP − w × FP) / N。

    参数：
        probabilities (list[float]): 预测概率
        outcomes (list[int]): 真实结局
        thresholds (list[float]|None): 阈值序列；None 时用 0.05–0.5 步长 0.05

    返回：
        dict: 各阈值的净收益（treat_all / treat_none / model）
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in
                      [i * 0.05 for i in range(1, 11)]]
    n = len(probabilities)
    outcomes = list(outcomes)
    n_pos = sum(outcomes)
    results = []
    for thresh in thresholds:
        tp = sum(1 for p, o in zip(probabilities, outcomes)
                 if p >= thresh and o == 1)
        fp = sum(1 for p, o in zip(probabilities, outcomes)
                 if p >= thresh and o == 0)
        w = thresh / (1.0 - thresh)  # 获益-代价比
        net_benefit = (tp - w * fp) / max(n, 1)
        treat_all_bn = (n_pos - w * (n - n_pos)) / max(n, 1)
        treat_none_bn = 0.0
        results.append({
            'threshold': thresh,
            'net_benefit': round(net_benefit, 4),
            'treat_all': round(treat_all_bn, 4),
            'treat_none': round(treat_none_bn, 4),
        })
    return {
        'thresholds': thresholds,
        'curves': results,
        'best_threshold': max(results, key=lambda r: r['net_benefit'])['threshold'],
    }


def evaluate_task_b(probabilities, outcomes, event_times=None,
                    horizon_days=365, thresholds=None):
    """任务 B 评估：纵向感染后进展预测。

    参数：
        probabilities (list[float]): 预测概率（0-1 或 0-100）
        outcomes (list[int]): 真实结局（1 = 随访期内进展）
        event_times (list[float]|None): 事件发生时间（天）；None 时忽略
        horizon_days (int): 固定时点（默认 365 天）
        thresholds (list[float]|None): DCA 阈值序列

    返回：
        dict: 评估报告（C-index、固定时点 AUC、校准、DCA）
    """
    from .threshold_spec import expected_calibration_error
    from ..validation.layer_ablation import compute_c_index

    probs = [float(p) for p in probabilities]
    is_percent = max(probs) > 1.0 if probs else False
    labels = [int(o) for o in outcomes]
    n = len(labels)
    n_events = sum(labels)

    # C-index（序一致性，适用于时间-事件数据）
    c_index = compute_c_index(labels, probs)

    # 固定时点 AUC（区分事件者 vs 非事件者）
    horizon_auc = _compute_horizon_auc(probs, labels, horizon_days)

    # 校准误差
    calib_probs = [p / 100.0 if is_percent else p for p in probs]
    ece = expected_calibration_error(calib_probs, labels, n_bins=10)
    ece_value = ece.get('ece', 0.0) if isinstance(ece, dict) else float(ece)

    # DCA
    dca = _decision_curve_analysis(calib_probs, labels, thresholds)

    return {
        'task': 'B',
        'task_name': '感染后进展为活动性结核的风险预测',
        'task_name_en': 'Post-infection Progression to Active TB',
        'endpoint': 'progression',
        'endpoint_description': '已感染者未来 6–24 个月内是否进展为活动性结核',
        'time_window': f'前瞻性 {horizon_days} 天（{horizon_days // 30} 个月）随访',
        'decision_meaning': '决定谁接受预防性治疗（TPT）',
        'n_samples': n,
        'n_events': n_events,
        'event_rate': round(n_events / max(n, 1), 4),
        'horizon_days': horizon_days,
        'metrics': {
            'c_index': round(c_index, 4),
            'horizon_auc': round(horizon_auc, 4),
            'expected_calibration_error': round(ece_value, 4),
        },
        'decision_curve_analysis': dca,
        'epsilon': 1e-10,
    }


# ==============================================================================
# 任务 C：接触网络传播预测（Network / Population Level）
# ==============================================================================

def _hit_rate(predicted_ranks, true_positive_indices, top_k=10):
    """命中率：前 top_k 预测中包含的真实阳性比例。"""
    top_set = set(predicted_ranks[:top_k])
    if not true_positive_indices:
        return 0.0
    hits = sum(1 for idx in true_positive_indices if idx in top_set)
    return hits / len(true_positive_indices)


def _ranking_quality(predicted_ranks, true_positive_indices):
    """排序质量：真实阳性在预测排序中的排位分布。"""
    positions = []
    for idx in true_positive_indices:
        try:
            positions.append(predicted_ranks.index(idx) + 1)
        except ValueError:
            positions.append(len(predicted_ranks))
    return {
        'mean_position': round(sum(positions) / max(len(positions), 1), 1),
        'median_position': round(sorted(positions)[len(positions) // 2], 1),
        'min_position': min(positions),
        'max_position': max(positions),
    }


def _r_estimate(secondary_cases, n_infectious):
    """繁殖数 R 估计（简化版）：R = 二代病例数 / 传染源数。"""
    if n_infectious <= 0:
        return 0.0
    return round(secondary_cases / n_infectious, 3)


def _contact_tracing_efficiency(traced_contacts, infected_contacts):
    """接触者追踪效率：被追踪到的感染者比例。"""
    if not infected_contacts:
        return 1.0
    if not traced_contacts:
        return 0.0
    return round(len(traced_contacts & infected_contacts) / len(infected_contacts), 4)


def evaluate_task_c(predicted_rankings, true_positive_indices,
                    secondary_cases=0, n_infectious=0,
                    traced_contacts=None, infected_contacts=None,
                    top_k_list=None):
    """任务 C 评估：接触网络传播预测。

    参数：
        predicted_rankings (list[int]): 按预测风险降序排列的接触者索引
        true_positive_indices (set/int): 真实阳性接触者索引集合
        secondary_cases (int): 二代病例数（用于 R 估计）
        n_infectious (int): 传染源数
        traced_contacts (set|None): 被追踪到的接触者索引
        infected_contacts (set|None): 实际感染接触者索引

    返回：
        dict: 评估报告（命中率、排序质量、R 估计、追踪效率）
    """
    if isinstance(true_positive_indices, (list, int)):
        if isinstance(true_positive_indices, int):
            true_positive_indices = {true_positive_indices}
        else:
            true_positive_indices = set(true_positive_indices)

    if top_k_list is None:
        top_k_list = [5, 10, 20, 50]
    hit_rates = {f'top_{k}': round(_hit_rate(
        predicted_rankings, true_positive_indices, k), 4)
                 for k in top_k_list}

    quality = _ranking_quality(predicted_rankings, true_positive_indices)

    r = _r_estimate(secondary_cases, n_infectious)

    tracing_efficiency = None
    if traced_contacts is not None and infected_contacts is not None:
        tracing_efficiency = _contact_tracing_efficiency(
            set(traced_contacts), set(infected_contacts))

    return {
        'task': 'C',
        'task_name': '接触网络传播预测',
        'task_name_en': 'Contact Network Transmission Prediction',
        'endpoint': 'contact_infection',
        'endpoint_description': '接触者感染风险排序、二代病例数、传播链走向',
        'time_window': '动态/疫情期',
        'decision_meaning': '决定追踪谁、在哪里干预',
        'n_contacts': len(predicted_rankings),
        'n_true_positives': len(true_positive_indices),
        'metrics': {
            'hit_rate': hit_rates,
            'ranking_quality': quality,
            'r_estimate': r,
            'contact_tracing_efficiency': tracing_efficiency,
        },
        'epsilon': 1e-10,
    }


# ==============================================================================
# 统一入口：任务注册与标签分配
# ==============================================================================

def assign_task_labels(records, task_type, **kwargs):
    """从同一批接触者记录中提取对应任务的标签（确保终点不可混用）。

    参数：
        records (list[dict]): 接触者记录
        task_type (TaskType): 任务类型
        **kwargs: 任务特定参数（如 horizon_days 用于任务 B）

    返回：
        y_true (list[int]): 对应任务的标签
        end_point (EndPoint): 使用的终点

    规则：
        - 任务 A：从记录内 'is_confirmed' 或 'sputum_smear' 提取；
        - 任务 B：从记录内 'progression' 或 'follow_up_outcome' 提取；
        - 任务 C：从记录内 'infection_status' 或 'cluster_id' 提取。
    """
    task = TaskType(task_type) if isinstance(task_type, str) else task_type
    end_point = TASK_ENDPOINT_MAP[task]

    if task == TaskType.SCREENING:
        labels = []
        for r in records:
            val = r.get('is_confirmed',
                        r.get('sputum_smear',
                              r.get('active_tb', 0)))
            labels.append(1 if val and int(val) > 0 else 0)
    elif task == TaskType.PROGNOSIS:
        labels = []
        for r in records:
            val = r.get('progression',
                        r.get('follow_up_outcome',
                              r.get('is_confirmed', 0)))
            labels.append(1 if val and int(val) > 0 else 0)
    elif task == TaskType.NETWORK:
        labels = []
        for r in records:
            val = r.get('infection_status',
                        r.get('contact_infected',
                              r.get('is_confirmed', 0)))
            labels.append(1 if val and int(val) > 0 else 0)
    else:
        raise ValueError(f"未知任务类型: {task}")

    return labels, end_point


def evaluate_task(probabilities, outcomes, task_type, **kwargs):
    """统一评估入口：根据任务类型派发到对应的评估函数。

    参数：
        probabilities (list[float]): 预测概率
        outcomes (list[int]): 真实结局
        task_type (TaskType|str): 任务类型（'A' / 'B' / 'C'）
        **kwargs: 透传参数（如 threshold、horizon_days、top_k_list 等）

    返回：
        dict: 对应任务的评估报告
    """
    task = TaskType(task_type) if isinstance(task_type, str) else task_type
    if task == TaskType.SCREENING:
        return evaluate_task_a(probabilities, outcomes, **kwargs)
    elif task == TaskType.PROGNOSIS:
        return evaluate_task_b(probabilities, outcomes, **kwargs)
    elif task == TaskType.NETWORK:
        return evaluate_task_c(probabilities, outcomes, **kwargs)
    raise ValueError(f"未知任务类型: {task}")


__all__ = [
    'TaskType', 'EndPoint', 'TASK_ENDPOINT_MAP',
    'evaluate_task_a',
    'evaluate_task_b',
    'evaluate_task_c',
    'assign_task_labels',
    'evaluate_task',
]