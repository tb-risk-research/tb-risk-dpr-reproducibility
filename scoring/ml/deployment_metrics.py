#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1：真实患病率口径的部署决策指标（2026-08-24）。

背景：场景训练数据阳性率 20.7%，Kenya 真实患病率 0.53%——患病率断层
40 倍。模型在 20% 患病率上训练校准，阈值与概率都不可直接迁移到
0.5% 人群；而 AUROC 对患病率不敏感，掩盖了这一断层。筛查部署的
真正决策指标（本模块提供）：

1. ``logit_shift``：先验校正——按目标患病率平移 logit（Saerens 2002），
   单调变换，排序（AUROC）不变，概率口径迁移到目标人群；
2. ``ppv_at_sensitivity``：敏感度约束下的最优筛查工作点——满足
   敏感度目标（默认 80%）的最高阈值（等价于该约束下 PPV 最大/
   筛查人数最少）；
3. ``screening_metrics``：组合指标（AUROC / AUPRC / PPV@敏感度 /
   NNS / Brier），一次计算完整部署画像；
4. ``population_stability_index``：PSI 分数分布漂移监控（前瞻
   部署：批次分数 vs 验收窗口参考分布）；
5. ``calibration_error``：ECE + Brier 校准漂移监控（批次级
   「预期/实际」一致性，Naeini 2015 等宽分箱口径）。

NNS（Number Needed to Screen）= 1/PPV：在敏感度 80% 约束的工作点，
每筛查 N 人发现 1 例活动性结核。

文献：
- Saerens M, Latinne P, Decaestecker C. Adjusting the Outputs of a
  Classifier to New a Priori Probabilities: A Simple Procedure.
  Neural Computation 2002;14:21-41（logit 先验平移，模型输出向新
  先验的无参数再校准）
- Trevethan R. Sensitivity, Specificity, and Predictive Values:
  Foundations, Pliabilities, and Pitfalls in Research and Practice.
  Educ Health 2017;30:36-38（PPV 随患病率变化的筛查决策含义）
- Saito T, Rehmsmeier M. The Precision-Recall Plot Is More Informative
  than the ROC Plot When Evaluating Binary Classifiers on Imbalanced
  Datasets. PLoS One 2015;10:e0118432（低患病率下 AUPRC 优于 AUROC）
"""

import logging

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False

LOGGER = logging.getLogger('tb_risk.ml.deployment_metrics')

# logit 数值边界（概率裁剪，防 0/1 端点溢出）
_PROB_EPS = 1e-6

# 监控告警线（§11.5 护栏分级，2026-09-05 修订）——单一来源：
# PSI 0.10/0.25 只对 **ind 参照臂**（严格点时基线，无先证块）成立；
# ind_all 的先证块机械漂移放大读数 ~1.7-2×，其 PSI 为诊断通道
# （解读规则见 docs/deployment_ops.md §11.5）；ECE 盯部署工件。
PSI_WARN = 0.10
PSI_DRIFT = 0.25
ECE_WARN = 0.05


def _logit(p):
    p = np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit_shift(p, source_prevalence, target_prevalence):
    """先验校正：把模型概率从源患病率口径平移到目标患病率口径。

    p_adj = sigmoid(logit(p) + logit(π_target) - logit(π_source))

    性质（Saerens 2002）：严格单调 → 排序/AUROC 不变；当模型概率
    在源人群校准良好时，校正后均值收敛到目标患病率。

    参数：
        p (array-like): 模型概率（0-1）
        source_prevalence (float): 训练/校准人群患病率
        target_prevalence (float): 部署人群患病率

    返回：
        np.ndarray: 校正后概率（(eps, 1-eps)）
    """
    p = np.asarray(p, dtype=float)
    offset = (_logit(np.asarray(float(target_prevalence)))
              - _logit(np.asarray(float(source_prevalence))))
    return np.clip(_sigmoid(_logit(p) + float(offset)),
                   _PROB_EPS, 1.0 - _PROB_EPS)


def ppv_at_sensitivity(y_true, scores, sensitivity_target=0.80):
    """敏感度约束下的最优筛查工作点。

    工作点定义：满足 sensitivity >= sensitivity_target 的**最高**阈值
    （并列分数按组进入）——即敏感度约束下筛查人数最少、PPV 最大、
    NNS 最小的决策点。敏感度目标不可达时退化为全量筛查并标记
    ``reached=False``。

    参数：
        y_true (array-like): 二元标签
        scores (array-like): 风险分数（越高越危险）
        sensitivity_target (float): 敏感度约束（默认 0.80）

    返回：
        dict: threshold / sensitivity / specificity / ppv / nns /
              tp / fp / n_flagged / flagged_rate / n_pos / reached

    异常：
        ValueError: 无阳性样本（工作点无定义）
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError('y_true 与 scores 长度不一致')
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError('无阳性样本，敏感度约束工作点无定义')

    # 按分数降序的切断点（并列分数按组进入：每组一个候选阈值）
    order = np.argsort(-s, kind='mergesort')
    y_sorted = y[order]
    s_sorted = s[order]
    # 组边界：分数变化处
    change = np.flatnonzero(np.diff(s_sorted) != 0) + 1
    boundaries = np.concatenate([[0], change, [len(s_sorted)]])

    # 逐组累积，找满足敏感度目标的最后一组（最高阈值）
    cum_tp = np.cumsum(y_sorted)
    best_k = None
    for k in boundaries[:-1]:
        tp = int(cum_tp[k - 1]) if k > 0 else 0
        if tp / n_pos >= sensitivity_target:
            best_k = k
            break
    reached = best_k is not None
    if not reached:
        best_k = len(s_sorted)  # 不可达 → 全量筛查
        LOGGER.warning('敏感度目标 %.2f 不可达（最高可达 %.3f），'
                       '退化为全量筛查', sensitivity_target, 1.0)

    tp = int(cum_tp[best_k - 1]) if best_k > 0 else 0
    fp = int(best_k - tp)
    n_flagged = int(best_k)
    n = len(y)
    sens = tp / n_pos
    ppv = tp / n_flagged if n_flagged else 0.0
    threshold = float(s_sorted[best_k - 1]) if best_k > 0 else float('inf')
    return {
        'threshold': threshold,
        'sensitivity': sens,
        'specificity': (n - n_pos - fp) / (n - n_pos) if n > n_pos else 0.0,
        'ppv': ppv,
        'nns': 1.0 / ppv if ppv > 0 else float('inf'),
        'tp': tp,
        'fp': fp,
        'n_flagged': n_flagged,
        'flagged_rate': n_flagged / n,
        'n_pos': n_pos,
        'n_total': n,
        'reached': reached,
    }


def screening_metrics(y_true, scores, source_prevalence=None,
                      target_prevalence=None, sensitivity_target=0.80):
    """筛查部署组合指标：AUROC / AUPRC / PPV@敏感度 / NNS / Brier。

    参数：
        y_true / scores: 标签与分数（敏感度工作点直接在部署人群的
            真实标签上计算——PPV/NNS 本身就是真实患病率口径）
        source_prevalence (float|None): 训练人群患病率；与
            target_prevalence 同时给出时执行 logit 先验校正并报告
            校正前后 Brier（阈值迁移的概率口径检验）
        target_prevalence (float|None): 部署人群患病率
        sensitivity_target (float): 敏感度约束

    返回：
        dict: base_rate / auroc / auprc / ppv_at_sensitivity / nns /
              brier_raw [ / brier_shifted / mean_prob_shifted /
              calibration_shift_applied ]
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError('y_true 与 scores 长度不一致')

    working = ppv_at_sensitivity(y, s, sensitivity_target=sensitivity_target)
    res = {
        'base_rate': float(y.mean()),
        'n': int(len(y)),
        'n_pos': int(y.sum()),
        'ppv_at_sensitivity': working,
        'nns': working['nns'],
        'sensitivity_target': sensitivity_target,
        'brier_raw': float(np.mean((np.clip(s, 0.0, 1.0) - y) ** 2)),
    }
    if SKLEARN_AVAILABLE and y.min() != y.max():
        res['auroc'] = float(roc_auc_score(y, s))
        res['auprc'] = float(average_precision_score(y, s))
    else:
        res['auroc'] = None
        res['auprc'] = None
        LOGGER.warning('标签退化（全阴/全阳）或 sklearn 不可用，AUROC/AUPRC 置 None')

    if source_prevalence is not None and target_prevalence is not None:
        shifted = logit_shift(np.clip(s, 0.0, 1.0),
                              source_prevalence, target_prevalence)
        res['brier_shifted'] = float(np.mean((shifted - y) ** 2))
        res['mean_prob_shifted'] = float(shifted.mean())
        res['calibration_shift_applied'] = {
            'method': 'logit_shift (Saerens 2002)',
            'source_prevalence': float(source_prevalence),
            'target_prevalence': float(target_prevalence),
        }
    return res


def population_stability_index(ref_scores, scores, n_bins=10):
    """ PSI（Population Stability Index）：分数分布漂移监控。

    分箱：在 ref_scores 分位数上切 n_bins 箱（ref 每箱等占比），
    PSI = Σ (a_i − r_i)·ln(a_i/r_i)，a/r 为新样本/参考样本的箱占比。
    惯例阈值（风控行业标准，PSI_WARN/PSI_DRIFT）：<0.10 稳定；
    0.10-0.25 警告；>0.25 漂移。**通道约定（§11.5）**：该阈值只对
    ind 参照臂（严格点时基线特征）成立——含时序先证块的 ind_all
    读数被机械漂移放大 ~1.7-2×，只作诊断解读。

    参数：
        ref_scores (array-like): 参考分布（模型验收窗口的分数）
        scores (array-like): 待监控批次分数
        n_bins (int): 分位数分箱数

    返回：
        float: PSI（≥0；同分布 → ≈0）

    异常：
        ValueError: 输入为空或长度不足
    """
    r = np.asarray(ref_scores, dtype=float)
    a = np.asarray(scores, dtype=float)
    if len(r) == 0 or len(a) == 0:
        raise ValueError('ref_scores 与 scores 均不能为空')
    if n_bins < 2:
        raise ValueError('n_bins 必须 >= 2')
    # 分位数切点（去重防并列分数产生空箱；端点 -inf/+inf 兜底）
    qs = np.quantile(r, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    edges = np.unique(np.concatenate(([-np.inf], qs, [np.inf])))
    r_idx = np.clip(np.searchsorted(edges, r, side='right') - 1,
                    0, len(edges) - 2)
    a_idx = np.clip(np.searchsorted(edges, a, side='right') - 1,
                    0, len(edges) - 2)
    eps = 1e-6  # 占比下限，防 ln(0)
    r_frac = np.bincount(r_idx, minlength=len(edges) - 1) / len(r) + eps
    a_frac = np.bincount(a_idx, minlength=len(edges) - 1) / len(a) + eps
    r_frac /= r_frac.sum()
    a_frac /= a_frac.sum()
    return float(np.sum((a_frac - r_frac) * np.log(a_frac / r_frac)))


def calibration_error(y_true, scores, n_bins=10):
    """ ECE（等宽分箱 Expected Calibration Error）+ Brier。

    ECE = Σ (n_b/N)·|mean_score_b − pos_rate_b|——校准漂移监控
    的标准量（Naeini 2015）。同时返回 Brier 与分箱明细供诊断。

    参数：
        y_true (array-like): 二元标签
        scores (array-like): 概率分数（0-1）
        n_bins (int): 等宽分箱数（默认 10）

    返回：
        dict: ece / brier / n / n_pos / bin_centers / bin_counts /
              bin_pos_rates（空箱占比 None）

    异常：
        ValueError: 输入为空或长度不一致
    """
    y = np.asarray(y_true, dtype=int)
    s = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    if len(y) == 0:
        raise ValueError('输入不能为空')
    if y.shape != s.shape:
        raise ValueError('y_true 与 scores 长度不一致')
    idx = np.minimum((s * n_bins).astype(int), n_bins - 1)
    centers, counts, pos_rates = [], [], []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        n_b = int(m.sum())
        if n_b == 0:
            centers.append((b + 0.5) / n_bins)
            counts.append(0)
            pos_rates.append(None)
            continue
        ms, my = float(s[m].mean()), float(y[m].mean())
        ece += (n_b / len(y)) * abs(ms - my)
        centers.append(ms)
        counts.append(n_b)
        pos_rates.append(my)
    return {
        'ece': float(ece),
        'brier': float(np.mean((s - y) ** 2)),
        'n': int(len(y)),
        'n_pos': int(y.sum()),
        'bin_centers': centers,
        'bin_counts': counts,
        'bin_pos_rates': pos_rates,
    }
