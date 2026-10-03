#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序先验部署口径决策分析（P2，2026-08-25 第四轮）：把
+0.058 AUROC / +48% PR-AUC 翻译成筛查系统的操作指标。

背景（P1 归档 household_temporal_screening_v1）：时序家庭先验
exposure+prior:RF 0.7142 vs ind:RF 0.6428（户分组 CV、20 种子、
cluster bootstrap CI 显著），兑现同户 LOO 上界的 99%。本模块回答
部署问题：这个判别力增益在筛查操作上值多少检测、多少检出、多少
NNS。

三个问题（用户 P2 规格）：
  Q1 "一轮全筛" vs "时序风险排序逐人筛 + 停止规则"：检测数 /
     检出数 / NNS；
  Q2 首筛查者策略：'highest_risk'（个体风险最高者先筛——既早抓
     病例又早释放户信号）vs 'random'（部署期望口径），是否值得做成
     策略（oracle-anti 0.024 差距的操作翻译）；
  Q3 (w, k) 校准：可部署 logit 上调近似的权重/收缩由 HomeACF OOF
     网格校准（scoring/temporal_household.py 的
     DEFAULT_TEMPORAL_WEIGHT=9.0 / SHRINKAGE=2.0 为解析先验）。

臂结构（每种子全部 OOF，p_base = ind:RF 户分组 CV OOF 概率，
0-1 尺度）：
  static            一轮全筛：按 p_base 排序前 k 人（k 由工作点定）
  seq_random        序贯 + random 首筛（部署期望主口径）
  seq_highest_risk  序贯 + 个体风险最高首筛（用户建议策略）

序贯重放（sequential_replay_fast）：scoring.temporal_household.
sequential_screening_trace 的向量化同语义实现——每步筛查当前
更新后风险最高的未筛成员（跨户全局比较），结果揭示后其所在户
其余成员风险上调/下调并重排（户间重排，两阶段筛查部署语义）。
并列分数选索引小者（与参考实现一致）。

停止规则双口径：
  A. 目标检出率（回顾性决策分析）：达到敏感度目标（默认 80%
     检出）所需筛查数——static 与 seq 同目标直接可比；
  B. 前瞻阈值（部署可用，无需标签）：当前最高更新后风险跌破
     静态工作点阈值（ppv_at_sensitivity 在 p_base 上的 80% 敏感度
     工作点）即停——序贯更新上调阳性户/下调阴性户，同一阈值下
     筛查数与检出数均可变。random 首筛者由策略指定，不看分数
     → 首步不检查阈值；此后每步按分数选择并检查。

时序留出分数（temporal_holdout_scores）：固定筛查顺序口径，
第 i 位成员分数 = σ(logit(p_base_i) + w·(r̂_i − π))，r̂ 为同户
先证（位置更小者）结果的收缩率——校准（Q3）与顺序敏感性
（oracle/anti 包络，可部署近似口径）共用。

诚实边界：
  - (w, k) 在 HomeACF OOF（校准种子）上网格搜索，评估在同一队列
    （无独立外样本）——跨人群迁移未验证（P4 线：PACTS/ERASE-TB
    复验）；网格结果全量归档供审查；
  - 可部署近似是解析变换（logit 线性上调），非 RF(exposure+prior)
    本身——校准目标为时序留出分数 AUROC 兑现度（vs RF 参考锚
    0.714），兑现度入归档；
  - 口径 B 的静态阈值依赖 p_base 校准质量（RF OOF 概率未做
    Platt/isotonic 再校准），阈值仅用于序贯/静态同锚对比；
  - 随机首筛与 random 顺序协议吸收排序随机性（多种子平均），
    真实筛查顺序未记录；
  - 单队列（南非 HomeACF）LTBI 终点：检出 = TST 阳性（LTBI），
    非活动性 TB——NNS 是"每检出 1 例 LTBI 的筛查数"口径；
  - 多种子均值±sd 汇总筛查数/NNS；AUROC 差用末种子户级
    cluster bootstrap（口径同 household_temporal）。
"""

import json
import os
import time

import numpy as np
import pandas as pd

from .household_temporal import (
    assign_screening_order,
    prior_features,
    run_temporal_household_once,
)
from .real_data_infection import (
    feature_columns as inf_feature_columns,
    load_homeacf_contacts,
)
from .real_data_pi import (
    _cluster_bootstrap_delta,
    _group_cv_indices,
    _make_model,
)
from ..scoring.temporal_household import (
    DEFAULT_TEMPORAL_SHRINKAGE,
    DEFAULT_TEMPORAL_WEIGHT,
    FIRST_SCREENER_POLICIES,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (w, k) 校准网格：覆盖解析先验（w≈9.4 / k=2）的邻域
WEIGHT_GRID = (0.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 16.0, 20.0)
SHRINKAGE_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)

_PROB_EPS = 1e-6


def _auroc(y, scores):
    """AUROC（sklearn 优先，缺失时纯 Python fallback）。"""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, scores))
    except ImportError:  # pragma: no cover
        from .threshold_spec import compute_auc
        return float(compute_auc(list(scores), list(y)))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _PROB_EPS, 1.0 - _PROB_EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# 时序留出分数（固定顺序口径——校准与顺序敏感性）
# ---------------------------------------------------------------------------

def temporal_holdout_scores(base_scores, results, households, order_pos,
                            base_rate, weight, shrinkage):
    """固定筛查顺序的时序留出分数（可部署近似，0-1 尺度）。

    第 i 位成员分数 = σ(logit(p_base_i) + w·(r̂_i − π))，其中
    r̂_i = (prior_pos_i + k·π)/(prior_n_i + k)，先证特征来自同户
    **位置更小**成员的结果（temporal 留出 = 部署时序合法；与
    household_temporal.prior_features 同构）。prior_n = 0 →
    r̂ = π → 增量 0（首筛查者回退个体基线，与
    scoring.temporal_household.temporal_risk_update 一致）。

    参数：
        base_scores (sequence[float]): 个体基线概率（0-1）。
        results (sequence[int]): 真实标签（1=阳性）。
        households (sequence): 户标识。
        order_pos (sequence[float]): 户内筛查位置（0 最先）。
        base_rate (float): 队列基线阳性率 π。
        weight (float): logit 上调权重 w。
        shrinkage (float): 收缩伪计数 k。

    返回：
        np.ndarray: 时序留出分数（(eps, 1-eps)）。
    """
    df = pd.DataFrame({
        'tst_pos10': [1 if int(r) else 0 for r in results],
        'record_id': list(households),
    })
    order = pd.Series(np.asarray(order_pos, dtype=float))
    pf = prior_features(df, order)
    prior_n = pf['prior_n'].to_numpy()
    prior_pos = pf['prior_pos'].to_numpy()
    pi = float(base_rate)
    r_hat = (prior_pos + float(shrinkage) * pi) / (prior_n + float(shrinkage))
    delta = float(weight) * (r_hat - pi)
    return np.clip(_sigmoid(_logit(base_scores) + delta),
                   _PROB_EPS, 1.0 - _PROB_EPS)


# ---------------------------------------------------------------------------
# 向量化序贯重放（与 sequential_screening_trace 同语义）
# ---------------------------------------------------------------------------

def sequential_replay_fast(base_scores, results, households, base_rate=None,
                           weight=DEFAULT_TEMPORAL_WEIGHT,
                           shrinkage=DEFAULT_TEMPORAL_SHRINKAGE,
                           first_policy='highest_risk', seed=0,
                           stop_threshold=None):
    """多户队列序贯筛查重放（向量化；n=2725 约 0.1-0.3 秒）。

    与 scoring.temporal_household.sequential_screening_trace 同语义：
    每步筛查当前更新后风险最高的未筛成员（并列选索引小者），
    结果揭示后其户内其余成员风险更新并重排。本实现把"同户未筛
    成员共享户级证据"的不变量提出来：每步只算每户一个 delta，
    分数 = σ(logit(base_i) + delta_{hh(i)})。

    参数：
        base_scores (sequence[float]): 个体基线概率（0-1）。
        results (sequence[int]): 真实标签（仅用于揭示）。
        households (sequence|None): 户标识；None → 全部同一户。
        base_rate (float|None): 基线率；None = 队列经验率。
        weight / shrinkage: logit 上调权重 w 与收缩伪计数 k。
        first_policy (str): 'highest_risk' / 'random'（首步策略；
            此后每步自适应选当前最高更新后风险）。
        seed (int): random 首筛者随机种子。
        stop_threshold (float|None): 前瞻停止阈值（口径 B）——
            按分数选择时当前最高更新后风险 < 阈值即停；
            None → 全量重放。random 首步不看分数，不检查阈值。

    返回：
        dict: n_members / n_households / base_rate / weight /
        shrinkage / first_policy / seed / stop_threshold /
        n_screens / n_positive_found / stopped_by_threshold /
        first_screen_positive / screen_order / detection_curve /
        risk_at_screen（每步被筛者的更新前风险，0-1）。
    """
    if first_policy not in FIRST_SCREENER_POLICIES:
        raise ValueError('first_policy 必须是 %s 之一'
                         % (FIRST_SCREENER_POLICIES,))
    res = np.asarray([1 if int(r) else 0 for r in results], dtype=int)
    n = len(res)
    base = np.asarray(base_scores, dtype=float)
    if len(base) != n or n == 0:
        raise ValueError('base_scores 与 results 必须等长且非空')
    if households is None:
        hh_idx = np.zeros(n, dtype=int)
    else:
        if len(households) != n:
            raise ValueError('households 长度必须与 base_scores 一致')
        _, hh_idx = np.unique(np.asarray(households), return_inverse=True)
    if base_rate is None:
        base_rate = float(res.mean()) if res.sum() else 0.0
    pi = float(base_rate)
    w = float(weight)
    k = float(shrinkage)

    logits = _logit(base)
    H = int(hh_idx.max()) + 1
    hh_n = np.zeros(H, dtype=float)
    hh_pos = np.zeros(H, dtype=float)
    screened = np.zeros(n, dtype=bool)
    rng = np.random.RandomState(seed)

    screen_order = []
    risk_at_screen = []
    detection = []
    found = 0
    stopped = False

    while True:
        remaining = np.flatnonzero(~screened)
        if len(remaining) == 0:
            break
        # 户级收缩率：n_h = 0 时 (0 + kπ)/k = π（数学恒等）
        r_hat = (hh_pos + k * pi) / (hh_n + k)
        delta_h = w * (r_hat - pi)
        scores = _sigmoid(logits + delta_h[hh_idx])
        if not screen_order and first_policy == 'random':
            idx = int(rng.choice(remaining))
        else:
            pick = int(remaining[np.argmax(scores[remaining])])
            if stop_threshold is not None and \
                    scores[pick] < float(stop_threshold):
                stopped = True
                break
            idx = pick
        screen_order.append(idx)
        risk_at_screen.append(float(scores[idx]))
        found += int(res[idx])
        detection.append(found)
        screened[idx] = True
        hh_n[hh_idx[idx]] += 1.0
        hh_pos[hh_idx[idx]] += float(res[idx])

    return {
        'n_members': n,
        'n_households': H,
        'base_rate': pi,
        'weight': w,
        'shrinkage': k,
        'first_policy': first_policy,
        'seed': seed,
        'stop_threshold': None if stop_threshold is None else float(stop_threshold),
        'n_screens': len(screen_order),
        'n_positive_found': found,
        'stopped_by_threshold': stopped,
        'first_screen_positive': int(res[screen_order[0]])
        if screen_order else None,
        'screen_order': screen_order,
        'detection_curve': detection,
        'risk_at_screen': risk_at_screen,
    }


# ---------------------------------------------------------------------------
# 工作点指标
# ---------------------------------------------------------------------------

def screening_workpoint(detection_curve, n_pos, sensitivity_target=0.80):
    """达到目标检出率所需筛查数（停止口径 A，回顾性）。

    detection_curve[k-1] = 前 k 步累计检出数；返回首次达到
    ceil(target·n_pos) 的步数（不可达 → 全量筛查 + reached=False）。
    """
    n_pos = int(n_pos)
    if n_pos <= 0:
        raise ValueError('无阳性样本，工作点无定义')
    target_tp = int(np.ceil(float(sensitivity_target) * n_pos))
    curve = list(detection_curve)
    for step, found in enumerate(curve, start=1):
        if found >= target_tp:
            return {
                'n_screens': step,
                'tp': int(found),
                'sensitivity': float(found) / n_pos,
                'nns': float(step) / found if found else float('inf'),
                'reached': True,
            }
    found = curve[-1] if curve else 0
    return {
        'n_screens': len(curve),
        'tp': int(found),
        'sensitivity': float(found) / n_pos,
        'nns': float(len(curve)) / found if found else float('inf'),
        'reached': False,
    }


def _oof_ind_scores(df, y, folds, seed):
    """ind:RF 户分组 CV OOF 概率（与 household_temporal 同口径）。"""
    X = np.nan_to_num(
        df[inf_feature_columns('ind')].to_numpy(dtype=float), nan=0.0)
    p = np.zeros(len(y))
    for tr, te in folds:
        m = _make_model('random_forest', seed)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


# ---------------------------------------------------------------------------
# Q3：(w, k) 网格校准
# ---------------------------------------------------------------------------

def calibrate_weight_shrinkage(df=None, seed=0, n_splits=5,
                               w_grid=WEIGHT_GRID, k_grid=SHRINKAGE_GRID,
                               rf_reference=True):
    """(w, k) 网格校准：时序留出分数 AUROC 兑现度最大化。

    目标：random 顺序下 temporal_holdout_scores 的 AUROC（部署期望
    口径，与 household_temporal 主口径一致）。rf_reference=True 时
    通过 run_temporal_household_once(seed) 提供 p_base（ind:RF OOF）
    与 RF(exposure+prior) 参考锚（兑现度分母）；False 时轻量自算
    p_base（无参考锚，供测试）。

    返回：
        dict: best_w / best_k / best_auroc / base_auroc（w=0 基线）/
        rf_reference_auroc / realization（兑现度） / grid / seed。
    """
    if df is None:
        df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    households = list(groups)
    base_rate = float(y.mean())

    rf_auroc = None
    if rf_reference:
        rep = run_temporal_household_once(seed=seed, n_splits=n_splits,
                                           df=df)
        p_base = np.asarray(rep['oof']['ind:RF'], dtype=float)
        rf_auroc = float(rep['arms']['exposure_prior:RF']['auroc'])
    else:
        folds = _group_cv_indices(groups, y, n_splits=n_splits, seed=seed)
        p_base = _oof_ind_scores(df, y, folds, seed)

    order = assign_screening_order(df, seed=seed, mode='random')
    order_pos = order.to_numpy()

    grid = []
    best = None
    for w in w_grid:
        for k in k_grid:
            s = temporal_holdout_scores(p_base, y, households, order_pos,
                                       base_rate, w, k)
            a = _auroc(y, s)
            grid.append({'w': float(w), 'k': float(k), 'auroc': a})
            if best is None or a > best['auroc']:
                best = {'w': float(w), 'k': float(k), 'auroc': a}

    return {
        'seed': int(seed),
        'best_w': best['w'],
        'best_k': best['k'],
        'best_auroc': best['auroc'],
        'base_auroc': _auroc(y, p_base),
        'rf_reference_auroc': rf_auroc,
        'realization': (best['auroc'] / rf_auroc) if rf_auroc else None,
        'grid': grid,
        'objective': 'temporal_holdout AUROC（random 顺序，部署期望口径）',
    }


# ---------------------------------------------------------------------------
# Q1/Q2：单种子部署决策分析
# ---------------------------------------------------------------------------

def run_deployment_analysis_once(seed=0, df=None, weight=None,
                                 shrinkage=None, n_splits=5,
                                 sensitivity_target=0.80):
    """单种子：全筛 vs 序贯（双首筛策略）× 双停止口径，全部 OOF。

    返回：
        dict: design / auroc / static / sequential（random +
        highest_risk 两策略：target 口径 A + prospect 口径 B）/
        detection（完整曲线，供多种子汇总采样）/ oof / y / groups。
    """
    if df is None:
        df = load_homeacf_contacts()
    if weight is None:
        weight = DEFAULT_TEMPORAL_WEIGHT
    if shrinkage is None:
        shrinkage = DEFAULT_TEMPORAL_SHRINKAGE
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    households = list(groups)
    n_pos = int(y.sum())
    base_rate = float(y.mean())

    folds = _group_cv_indices(groups, y, n_splits=n_splits, seed=seed)
    p_base = _oof_ind_scores(df, y, folds, seed)

    # ---- 静态臂（一轮全筛）----
    from ..scoring.ml.deployment_metrics import ppv_at_sensitivity
    static_wp = ppv_at_sensitivity(y, p_base,
                                   sensitivity_target=sensitivity_target)
    static_order = np.argsort(-p_base, kind='mergesort')
    static_curve = np.cumsum(y[static_order]).tolist()
    static_target = screening_workpoint(static_curve, n_pos,
                                        sensitivity_target)

    # ---- 序贯臂 × 首筛策略（同阈值锚定口径 B）----
    threshold = float(static_wp['threshold'])
    sequential = {}
    curves = {'static': static_curve}
    for policy in FIRST_SCREENER_POLICIES:
        tr = sequential_replay_fast(
            p_base, y, households, base_rate=base_rate,
            weight=weight, shrinkage=shrinkage,
            first_policy=policy, seed=seed, stop_threshold=threshold)
        target = screening_workpoint(tr['detection_curve'], n_pos,
                                     sensitivity_target)
        prospect = {
            'n_screens': int(tr['n_screens']),
            'tp': int(tr['n_positive_found']),
            'sensitivity': float(tr['n_positive_found']) / n_pos,
            'nns': (float(tr['n_screens']) / tr['n_positive_found']
                    if tr['n_positive_found'] else float('inf')),
            'stopped_by_threshold': bool(tr['stopped_by_threshold']),
        }
        sequential[policy] = {
            'target': target,
            'prospect': prospect,
            'first_screen_positive': tr['first_screen_positive'],
            'n_screens_total': int(tr['n_screens']),
        }
        curves['seq_%s' % policy] = tr['detection_curve']

    # ---- 时序留出分数 + 顺序敏感性（可部署近似口径）----
    order = assign_screening_order(df, seed=seed, mode='random')
    ts_random = temporal_holdout_scores(
        p_base, y, households, order.to_numpy(), base_rate,
        weight, shrinkage)
    auroc = {
        'static': _auroc(y, p_base),
        'temporal_random': _auroc(y, ts_random),
    }
    envelope = {}
    for mode in ('oracle', 'anti'):
        om = assign_screening_order(df, seed=seed, mode=mode)
        es = temporal_holdout_scores(p_base, y, households, om.to_numpy(),
                                     base_rate, weight, shrinkage)
        envelope[mode] = _auroc(y, es)

    return {
        'seed': int(seed),
        'weight': float(weight),
        'shrinkage': float(shrinkage),
        'sensitivity_target': float(sensitivity_target),
        'design': {
            'n': int(len(y)), 'n_events': n_pos,
            'n_households': int(pd.Series(groups).nunique()),
            'n_splits': int(n_splits),
            'endpoint': 'tst_pos10（TST≥10mm LTBI）',
        },
        'auroc': auroc,
        'order_envelope': envelope,
        'static': {
            'wp': static_wp,
            'target': static_target,
        },
        'sequential': sequential,
        'detection': curves,
        'oof': {'p_base': p_base, 'temporal_random': ts_random},
        'y': y,
        'groups': groups,
    }


# ---------------------------------------------------------------------------
# 多种子汇总 + 归档
# ---------------------------------------------------------------------------

def _mean_sd(vals):
    arr = np.asarray(vals, dtype=float)
    return {'mean': float(arr.mean()), 'sd': float(arr.std())}


_BUDGETS = np.round(np.linspace(0.0, 1.0, 21), 4)  # 每 5% 预算采样


def _budget_recall(curves, y):
    """检出曲线 → 各预算档的平均召回（归档采样）。"""
    n = len(y)
    n_pos = max(int(y.sum()), 1)
    out = []
    for b in _BUDGETS:
        k = int(np.ceil(b * n))
        if k == 0:
            out.append(0.0)
            continue
        recall = curves[min(k, len(curves)) - 1] / n_pos
        out.append(float(recall))
    return out


def run_multi_seed_deployment(n_seeds=20, seed_start=0, n_splits=5,
                              df=None, weight=None, shrinkage=None,
                              sensitivity_target=0.80, calibrate=True,
                              n_bootstrap=2000,
                              w_grid=WEIGHT_GRID, k_grid=SHRINKAGE_GRID,
                              rf_reference=True):
    """校准（Q3）→ 多种子主分析（Q1/Q2）→ 汇总（含 cluster bootstrap）。

    校准规则：weight/shrinkage 显式给出 → 直接用；否则 calibrate=True
    时在 seed_start 上网格校准（RF 参考锚口径），仍缺失回落解析先验
    （w=9.0 / k=2.0）。
    """
    if df is None:
        df = load_homeacf_contacts()

    calibration = None
    if weight is None or shrinkage is None:
        if calibrate:
            calibration = calibrate_weight_shrinkage(
                df=df, seed=seed_start, n_splits=n_splits,
                w_grid=w_grid, k_grid=k_grid,
                rf_reference=rf_reference)
            if weight is None:
                weight = calibration['best_w']
            if shrinkage is None:
                shrinkage = calibration['best_k']
        if weight is None:
            weight = DEFAULT_TEMPORAL_WEIGHT
        if shrinkage is None:
            shrinkage = DEFAULT_TEMPORAL_SHRINKAGE

    per_seed = []
    last = None
    for s in range(seed_start, seed_start + n_seeds):
        rep = run_deployment_analysis_once(
            seed=s, df=df, weight=weight, shrinkage=shrinkage,
            n_splits=n_splits, sensitivity_target=sensitivity_target)
        per_seed.append({
            'seed': rep['seed'],
            'auroc': rep['auroc'],
            'order_envelope': rep['order_envelope'],
            'static': {
                'target': rep['static']['target'],
                'prospect': {
                    'n_screens': rep['static']['wp']['n_flagged'],
                    'tp': rep['static']['wp']['tp'],
                    'sensitivity': rep['static']['wp']['sensitivity'],
                    'nns': rep['static']['wp']['nns'],
                },
            },
            'sequential': {
                p: {
                    'target': rep['sequential'][p]['target'],
                    'prospect': rep['sequential'][p]['prospect'],
                    'first_screen_positive':
                        rep['sequential'][p]['first_screen_positive'],
                } for p in FIRST_SCREENER_POLICIES
            },
        })
        last = rep

    y, groups = last['y'], last['groups']
    n_pos = int(y.sum())
    n = int(len(y))

    # AUROC 差：末种子户级 cluster bootstrap（口径同 household_temporal）
    bootstrap = _cluster_bootstrap_delta(
        last['oof']['temporal_random'], last['oof']['p_base'],
        y, groups, n_bootstrap=n_bootstrap, seed=seed_start)

    def _target_node(r, name):
        if name == 'static':
            return r['static']['target']
        return r['sequential'][name.split('seq_')[1]]['target']

    def _prospect_node(r, name):
        if name == 'static':
            return r['static']['prospect']
        return r['sequential'][name.split('seq_')[1]]['prospect']

    target_summary = {}
    prospect_summary = {}
    for name in ('static', 'seq_random', 'seq_highest_risk'):
        target_summary[name] = {
            'n_screens': _mean_sd([_target_node(r, name)['n_screens']
                                   for r in per_seed]),
            'nns': _mean_sd([_target_node(r, name)['nns']
                             for r in per_seed]),
        }
        prospect_summary[name] = {
            'n_screens': _mean_sd([_prospect_node(r, name)['n_screens']
                                   for r in per_seed]),
            'tp': _mean_sd([_prospect_node(r, name)['tp']
                            for r in per_seed]),
            'sensitivity': _mean_sd([_prospect_node(r, name)['sensitivity']
                                     for r in per_seed]),
            'nns': _mean_sd([_prospect_node(r, name)['nns']
                             for r in per_seed]),
        }

    first_hit = {
        'highest_risk': float(np.mean([
            r['sequential']['highest_risk']['first_screen_positive']
            for r in per_seed])),
        'random': float(np.mean([
            r['sequential']['random']['first_screen_positive']
            for r in per_seed])),
        'base_rate': float(y.mean()),
    }

    # 预算-召回曲线（末种子检出曲线采样——与 bootstrap 同末种子口径）
    budget_curves = {
        'static': _budget_recall(last['detection']['static'], y),
        'seq_random': _budget_recall(last['detection']['seq_random'], y),
        'seq_highest_risk': _budget_recall(
            last['detection']['seq_highest_risk'], y),
    }

    savings = {
        'screens_to_target_random_vs_static': float(
            1.0 - np.mean([r['sequential']['random']['target']['n_screens']
                           for r in per_seed])
            / np.mean([r['static']['target']['n_screens']
                       for r in per_seed])),
        'screens_to_target_highest_vs_static': float(
            1.0 - np.mean([
                r['sequential']['highest_risk']['target']['n_screens']
                for r in per_seed])
            / np.mean([r['static']['target']['n_screens']
                       for r in per_seed])),
        'prospect_nns_random_vs_static': float(
            np.mean([r['sequential']['random']['prospect']['nns']
                     for r in per_seed])
            - np.mean([r['static']['prospect']['nns']
                       for r in per_seed])),
        'seed_win_rate_random': float(np.mean([
            r['sequential']['random']['target']['n_screens']
            < r['static']['target']['n_screens'] for r in per_seed])),
    }

    return {
        'design': {
            'name': 'temporal_deployment_decision_v1',
            'source': 'HomeACF（github petermacp/tstsa）',
            'endpoint': 'tst_pos10（TST≥10mm LTBI）',
            'n': n, 'n_events': n_pos,
            'n_households': int(pd.Series(groups).nunique()),
            'n_seeds': n_seeds, 'seed_start': seed_start,
            'cv': f'StratifiedGroupKFold({n_splits}) by household',
            'sensitivity_target': float(sensitivity_target),
            'weight': float(weight), 'shrinkage': float(shrinkage),
            'calibrated': calibration is not None,
            'stop_rules': {
                'A_target_recall': '达到 %d%% 检出所需筛查数（回顾性）'
                                   % round(sensitivity_target * 100),
                'B_prospective_threshold':
                    '当前最高更新后风险 < 静态工作点阈值即停'
                    '（部署可用，无需标签）',
            },
            'leakage_argument':
                'p_base = ind:RF 户分组 CV OOF（同户同折）；时序分数'
                '只用同户先证（位置更小者）结果 = 部署时序合法',
            'acceptance': {
                'temporal_vs_static_auroc': {
                    'mean': bootstrap['mean'],
                    'ci': bootstrap['bootstrap_ci'],
                    'significant_positive': bool(
                        bootstrap['ci_excludes_zero']
                        and bootstrap['mean'] > 0)},
                'seq_screens_below_static_win_rate':
                    savings['seed_win_rate_random'],
                'first_hit_highest_above_base':
                    bool(first_hit['highest_risk']
                         > first_hit['base_rate']),
            },
        },
        'calibration': calibration,
        'target_summary': target_summary,
        'prospect_summary': prospect_summary,
        'savings': savings,
        'first_screen_hit': first_hit,
        'auroc_summary': {
            'static': _mean_sd([r['auroc']['static'] for r in per_seed]),
            'temporal_random': _mean_sd(
                [r['auroc']['temporal_random'] for r in per_seed]),
            'temporal_oracle': _mean_sd(
                [r['order_envelope']['oracle'] for r in per_seed]),
            'temporal_anti': _mean_sd(
                [r['order_envelope']['anti'] for r in per_seed]),
            'temporal_vs_static_bootstrap': bootstrap,
        },
        'budget_recall_curve': {
            'budgets': [float(b) for b in _BUDGETS],
            'note': '末种子检出曲线的预算-召回采样（每 5% 预算）',
            'static': budget_curves['static'],
            'seq_random': budget_curves['seq_random'],
            'seq_highest_risk': budget_curves['seq_highest_risk'],
        },
        'seeds': per_seed,
    }


def save_result(out, path=None):
    """归档 JSON。"""
    if path is None:
        stamp = time.strftime('%Y%m%d')
        path = os.path.join(_REPO_ROOT, 'data', 'processed',
                            f'temporal_deployment_{stamp}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return path


def main(n_seeds=20):
    """正式运行：校准 + 多种子 + 归档。"""
    t0 = time.time()
    out = run_multi_seed_deployment(n_seeds=n_seeds)
    path = save_result(out)
    print('归档：%s（%.1f 秒）' % (path, time.time() - t0))
    print('w/k = %.1f / %.1f' % (out['design']['weight'],
                                 out['design']['shrinkage']))
    print('AUROC：static %.4f → temporal %.4f（oracle %.4f / anti %.4f）'
          % (out['auroc_summary']['static']['mean'],
             out['auroc_summary']['temporal_random']['mean'],
             out['auroc_summary']['temporal_oracle']['mean'],
             out['auroc_summary']['temporal_anti']['mean']))
    print('达到 80%% 检出筛查数：static %.0f vs seq_random %.0f vs '
          'seq_highest %.0f'
          % (out['target_summary']['static']['n_screens']['mean'],
             out['target_summary']['seq_random']['n_screens']['mean'],
             out['target_summary']['seq_highest_risk']['n_screens']['mean']))
    print('口径 B（前瞻阈值）：NNS static %.2f vs seq_random %.2f'
          % (out['prospect_summary']['static']['nns']['mean'],
             out['prospect_summary']['seq_random']['nns']['mean']))
    return out


if __name__ == '__main__':
    main()
