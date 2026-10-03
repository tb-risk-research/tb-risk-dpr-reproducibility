#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序家庭先验评分服务（第 2 层网络增强层的部署实现，2026-08-25 第四轮 P1）。

验证依据（validation/household_temporal.py，HomeACF 20 种子归档）：
  - exposure+prior:RF 0.7142 vs ind:RF 0.6428（+0.0585 CI[+0.031,+0.087]
    显著），兑现同户 LOO 率上界（0.7150）的 99%——网络层构造第一次
    在真实数据上取得统计显著正增益；
  - 机制：户内**已筛查成员**的结果携带聚集信号（prior_n / prior_pos /
    prior_rate / prior_screened 四列，与实验完全同构）。

部署语义（用户规格）：评分流程支持"户内已筛查结果"作为输入——
同户有人筛查阳性时，其余成员风险上调并自动重排；首筛查者无先证
信息，回退个体基线。两阶段筛查：第一轮查出阳性户，第二轮在户内
按更新后风险排序。

实现（可部署近似，零深度学习依赖）：
  1. 先证特征四列（temporal_prior_features）——与实验 RF 特征同构；
  2. 户级率贝叶斯收缩估计 r̂ = (pos + k·π)/(n + k)（Laplace 平滑；
     k = 收缩伪计数，证据越多 r̂ 越接近经验率，n=0 时 r̂ = π）；
  3. logit 空间风险上调：p_temporal = σ(logit(p_base) + w·(r̂ − π))
     - n = 0 → r̂ = π → 增量 = 0（首筛查者回退个体基线）；
     - 同户阳性 → r̂ > π → 风险上调；全阴 → 下调（证据缺失也是证据）；
     - w（logit/率单位）与 k 由 HomeACF OOF 网格校准
       （validation/temporal_deployment.py，20 种子归档）。

序贯筛查模拟（HouseholdScreeningState / sequential_screening_trace）：
  - 每步筛查"当前更新后风险最高"的未筛成员（首步按 first_policy）；
  - 结果揭示后重算户内先证特征并重排其余成员；
  - 离线重放供部署口径分析（P2：全筛 vs 序贯+停止阈值的检出数 /
    NNS）与 GUI 动态演示（户内第一人 → 风险更新 → 重排）。

首筛查者策略（用户 P2 建议）：'highest_risk'（个体风险最高者先筛
——既早抓病例又早释放户信号）或 'random'（部署期望口径）。

诚实边界：
  - w/k 在 HomeACF（南非、LTBI/TST≥10mm 终点）校准，跨人群迁移
    未验证（P4 线：PACTS 复验；ERASE-TB 数据到位后首优先实验）；
  - 本服务是验证机制的可部署近似（logit 线性上调），不是实验中的
    RF(exposure+prior) 本身——RF 需要完整特征矩阵，部署评分流程
    只有基线概率输出；
  - logit 上调对同户未筛成员是**统一位移** → 户内排序保持基线序，
    重排发生在**户间**（阳性户成员整体上移越过其他户——这正是
    两阶段筛查的部署语义；RF 学到的户内交互重排为二阶效应，
    可部署近似不包含，校准时以整体 AUROC 兑现度度量）；
  - 户级证据只从"已筛查成员"聚合，绝不使用目标成员自身结果
    （无未来泄漏，构造上与实验协议一致）。

位置说明：本模块放在 ``scoring/`` 顶层而非 ``scoring/ml/``——它是
纯评分服务（仅依赖 numpy），且 ``scoring/architecture.py`` 在模块级
导入它；若置于 ``scoring/ml/`` 会触发 ``scoring/ml/__init__`` 的重
依赖链（persistence → ml.framework → core → integrator → 回头
import 部分初始化的 scoring.architecture），形成循环导入。
"""

import numpy as np

__all__ = [
    'PRIOR_FEATURE_NAMES',
    'DEFAULT_TEMPORAL_WEIGHT',
    'DEFAULT_TEMPORAL_SHRINKAGE',
    'FIRST_SCREENER_POLICIES',
    'temporal_prior_features',
    'household_rate_estimate',
    'temporal_risk_update',
    'temporal_increment',
    'sequential_screening_trace',
    'HouseholdScreeningState',
]

# 与 household_temporal.py 实验一致的先证特征列（部署口径）
PRIOR_FEATURE_NAMES = ('prior_n', 'prior_pos', 'prior_rate', 'prior_screened')

# logit 上调权重（每单位户级率差的 logit 位移）。
# 解析先验：家庭内已现 1 例阳性（r̂≈0.42 vs π≈0.13，k=2）对应
# 风险比约 2.3（logit 位移 ≈ 2.7）→ w ≈ 9.4；正式值由
# temporal_deployment.py 的 HomeACF OOF 网格校准覆写并归档。
DEFAULT_TEMPORAL_WEIGHT = 9.0

# 收缩伪计数 k：r̂ = (pos + k·π)/(n + k)；k 越大越保守
DEFAULT_TEMPORAL_SHRINKAGE = 2.0

FIRST_SCREENER_POLICIES = ('highest_risk', 'random')

_PROB_EPS = 1e-6


def _logit(p):
    p = np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS)
    return float(np.log(p / (1.0 - p)))


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-x)))


def _clean_results(screened_results):
    """已筛查结果序列 → 0/1 列表（防御布尔/整型混用）。"""
    if screened_results is None:
        return []
    return [1 if int(r) else 0 for r in screened_results]


def temporal_prior_features(screened_results, base_rate):
    """已筛查结果序列 → 四列先证特征（与实验 RF 特征同构）。

    参数：
        screened_results (sequence[int]): 已筛查同户成员的结果（1=阳性）；
            只允许"早于目标成员筛查"的结果进入（时序留出由调用方保证）。
        base_rate (float): 队列基线阳性率（部署口径 = 人群患病率）。

    返回：
        dict: prior_n / prior_pos / prior_rate / prior_screened。
    """
    res = _clean_results(screened_results)
    n = len(res)
    pos = float(sum(res))
    rate = pos / n if n > 0 else float(base_rate)
    return {
        'prior_n': float(n),
        'prior_pos': pos,
        'prior_rate': float(rate),
        'prior_screened': 1.0 if n > 0 else 0.0,
    }


def household_rate_estimate(screened_results, base_rate,
                            shrinkage=DEFAULT_TEMPORAL_SHRINKAGE):
    """贝叶斯收缩户级率：r̂ = (pos + k·π)/(n + k)。

    n=0 → r̂ = π（无证据 → 基线）；n→∞ → 经验率。
    """
    k = float(shrinkage)
    if k < 0:
        raise ValueError('shrinkage 必须非负')
    res = _clean_results(screened_results)
    n = len(res)
    pos = float(sum(res))
    return (pos + k * float(base_rate)) / (n + k)


def temporal_risk_update(p_base_percent, screened_results, base_rate,
                          weight=DEFAULT_TEMPORAL_WEIGHT,
                          shrinkage=DEFAULT_TEMPORAL_SHRINKAGE):
    """logit 空间时序先验风险更新（0-100 尺度）。

    p_temporal = σ(logit(p_base) + w·(r̂ − π))；n = 0 → 原样返回
    （首筛查者回退个体基线——用户规格）。

    参数：
        p_base_percent (float): 0-100 个体基线概率（第 1 层输出）。
        screened_results (sequence[int]): 户内已筛查成员结果。
        base_rate (float): 队列基线阳性率。
        weight (float): logit/率单位上调权重 w。
        shrinkage (float): 收缩伪计数 k。

    返回：
        float: 更新后 0-100 风险。
    """
    p0 = float(np.clip(p_base_percent, 0.0, 100.0))
    res = _clean_results(screened_results)
    if not res:
        return p0
    pi = float(base_rate)
    if not (0.0 <= pi <= 1.0):
        raise ValueError('base_rate 必须在 [0, 1]')
    r_hat = household_rate_estimate(res, pi, shrinkage)
    delta = float(weight) * (r_hat - pi)
    p_new = _sigmoid(_logit(p0 / 100.0) + delta)
    return float(np.clip(p_new * 100.0, 0.0, 100.0))


def temporal_increment(p_base_percent, screened_results, base_rate,
                       weight=DEFAULT_TEMPORAL_WEIGHT,
                       shrinkage=DEFAULT_TEMPORAL_SHRINKAGE):
    """时序先验网络增量 = 更新后风险 − 个体基线（0-100 尺度，可负）。"""
    p0 = float(np.clip(p_base_percent, 0.0, 100.0))
    return temporal_risk_update(p0, screened_results, base_rate,
                                 weight=weight, shrinkage=shrinkage) - p0


class HouseholdScreeningState:
    """单户序贯筛查状态（GUI 在线交互 / 模拟共用）。

    用法：
        state = HouseholdScreeningState(base_scores, names, base_rate)
        idx = state.next_to_screen()          # 当前风险最高的未筛成员
        state.record_result(idx, 1)           # 揭示结果后自动重排
        state.current_scores()                 # 更新后风险列表
    """

    def __init__(self, base_scores, names=None, base_rate=0.1,
                 weight=DEFAULT_TEMPORAL_WEIGHT,
                 shrinkage=DEFAULT_TEMPORAL_SHRINKAGE):
        self.base_scores = [float(s) for s in base_scores]
        n = len(self.base_scores)
        if n == 0:
            raise ValueError('家庭至少需要 1 名成员')
        self.names = list(names) if names is not None else \
            [f'成员{i + 1}' for i in range(n)]
        if len(self.names) != n:
            raise ValueError('names 长度必须与 base_scores 一致')
        self.base_rate = float(base_rate)
        self.weight = float(weight)
        self.shrinkage = float(shrinkage)
        self.screened_flags = [False] * n
        self.results = [None] * n
        self.history = []   # 每步记录（审计/可视化）

    # ---- 状态查询 ----
    def screened_results(self):
        """已筛查结果序列（按筛查发生顺序）。"""
        return [self.results[i] for i in
                sorted(range(len(self.results)),
                       key=lambda k: self._screen_order(k))
                if self.screened_flags[i]]

    def _screen_order(self, idx):
        for step, rec in enumerate(self.history):
            if rec['screened_idx'] == idx:
                return step
        return len(self.history)

    def current_scores(self):
        """全部成员当前风险（已筛成员冻结为个体基线——筛查已完成）。"""
        res = [int(self.results[i]) for i in range(len(self.results))
               if self.screened_flags[i]]
        out = []
        for i, s in enumerate(self.base_scores):
            if self.screened_flags[i]:
                out.append(float(s))
            else:
                out.append(temporal_risk_update(
                    s, res, self.base_rate,
                    weight=self.weight, shrinkage=self.shrinkage))
        return out

    def next_to_screen(self):
        """当前更新后风险最高的未筛成员索引（全筛完 → None）。"""
        remaining = [i for i in range(len(self.base_scores))
                     if not self.screened_flags[i]]
        if not remaining:
            return None
        scores = self.current_scores()
        return max(remaining, key=lambda i: (scores[i], -i))

    def reset(self):
        """清空筛查状态（回到首筛查者）。"""
        self.screened_flags = [False] * len(self.base_scores)
        self.results = [None] * len(self.base_scores)
        self.history = []

    # ---- 状态推进 ----
    def record_result(self, idx, result):
        """登记成员 idx 的筛查结果并自动重排其余成员。

        返回：
            dict: 本步记录（screened_idx / result / risk_before /
            prior_before / updated_scores / remaining）。
        """
        if not (0 <= idx < len(self.base_scores)):
            raise ValueError('idx 越界')
        if self.screened_flags[idx]:
            raise ValueError('该成员已筛查')
        scores_before = self.current_scores()
        prior_before = temporal_prior_features(
            self.screened_results(), self.base_rate)
        self.screened_flags[idx] = True
        self.results[idx] = 1 if int(result) else 0
        scores_after = self.current_scores()
        rec = {
            'step': len(self.history) + 1,
            'screened_idx': idx,
            'name': self.names[idx],
            'risk_before': scores_before[idx],
            'result': int(result),
            'prior_before': prior_before,
            'updated_scores': scores_after,
        }
        self.history.append(rec)
        return rec


def sequential_screening_trace(base_scores, results, households=None,
                               base_rate=None,
                               weight=DEFAULT_TEMPORAL_WEIGHT,
                               shrinkage=DEFAULT_TEMPORAL_SHRINKAGE,
                               first_policy='highest_risk', seed=0):
    """多户队列序贯筛查离线重放（部署口径分析用，P2）。

    每步筛查"当前更新后风险最高"的未筛成员（跨户全局比较）；结果
    揭示后其所在户的其余成员风险上调/下调并重排（**户间**重排——
    阳性户成员整体上移越过其他户，即两阶段筛查的部署语义）。

    参数：
        base_scores (sequence[float]): 每成员个体基线风险（0-100）。
        results (sequence[int]): 每成员真实标签（1=阳性；仅用于揭示，
            绝不进入风险计算——已筛成员冻结为基线）。
        households (sequence|None): 每成员的户标识；None → 全部视为
            同一户（单户口径）。
        base_rate (float|None): 基线率；None = 队列经验率（分析口径）。
        first_policy (str): 队列首筛查者策略（用户 P2：建议个体风险
            最高者先筛——既早抓病例又早释放户信号）：
            'highest_risk' / 'random'。
        seed (int): random 首筛者的随机种子。

    返回：
        dict: {'steps', 'n_members', 'n_households', 'n_screens',
        'n_positive_found', 'screen_order', 'detection_curve',
        'updated_scores'}
        detection_curve: 第 k 步累计检出数（序贯效率曲线——
        与静态基线序对比即"时序先证的部署价值"）。
    """
    if first_policy not in FIRST_SCREENER_POLICIES:
        raise ValueError('first_policy 必须是 %s 之一'
                         % (FIRST_SCREENER_POLICIES,))
    base_scores = [float(s) for s in base_scores]
    results = [1 if int(r) else 0 for r in results]
    n = len(results)
    if len(base_scores) != n or n == 0:
        raise ValueError('base_scores 与 results 必须等长且非空')
    if households is None:
        households = ['hh_0'] * n
    elif len(households) != n:
        raise ValueError('households 长度必须与 base_scores 一致')
    if base_rate is None:
        base_rate = float(np.mean(results)) if sum(results) else 0.0

    # 每户已筛查结果（按筛查发生顺序追加）
    evidence = {}
    screened_flags = [False] * n
    rng = np.random.RandomState(seed)
    steps = []
    order = []
    found = 0

    def _updated(i):
        ev = evidence.get(households[i], [])
        return temporal_risk_update(base_scores[i], ev, base_rate,
                                    weight=weight, shrinkage=shrinkage)

    while True:
        remaining = [i for i in range(n) if not screened_flags[i]]
        if not remaining:
            break
        if not steps and first_policy == 'random':
            idx = int(rng.choice(remaining))
        else:
            idx = max(remaining, key=lambda i: (_updated(i), -i))
        scores_before = {i: _updated(i) for i in remaining}
        rec_result = results[idx]
        screened_flags[idx] = True
        evidence.setdefault(households[idx], []).append(rec_result)
        found += rec_result
        order.append(idx)
        steps.append({
            'step': len(steps) + 1,
            'screened_idx': idx,
            'household': households[idx],
            'risk_before': scores_before[idx],
            'rank_of_screened': sorted(
                remaining, key=lambda i: (-scores_before[i], i)
            ).index(idx) + 1,
            'result': rec_result,
            'cumulative_found': found,
            'updated_scores': [_updated(i) if not screened_flags[i]
                               else base_scores[i] for i in range(n)],
        })

    return {
        'n_members': n,
        'n_households': len(set(households)),
        'n_screens': n,
        'n_positive_found': found,
        'base_rate': float(base_rate),
        'weight': float(weight),
        'shrinkage': float(shrinkage),
        'first_policy': first_policy,
        'steps': steps,
        'screen_order': order,
        'updated_scores': [base_scores[i] if screened_flags[i]
                           else _updated(i) for i in range(n)],
        'detection_curve': [s['cumulative_found'] for s in steps],
    }
