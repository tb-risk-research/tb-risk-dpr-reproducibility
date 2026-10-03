#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPV 贝叶斯换算与"排序+截断点"决策参考（问题4：概率口径）

设计原则（用户决策口径约定）：
- 界面与报告以"排序 + 截断点"为主要决策形式，绝对概率仅作参考；
- PPV 按目标人群实际阳性率（先验 π）经贝叶斯换算得出；
- 两套阳性率口径（密接人群 2.85% / 全人群 100/10万）来自
  constants.PREVALENCE_CALIBERS 单一真值源，输出必须双口径并列展示，
  严禁混用（见 PREVALENCE_CALIBER_MIXING_WARNING）。

文献：
- 贝叶斯换算 PPV = sens·π / (sens·π + (1-spec)·(1-π))：
  Fletcher RH & Fletcher SW. Clinical Epidemiology: The Essentials,
  5th ed. (筛查试验阳预测值与患病率的关系)
- Wong HB. Singapore Med J 2005;46(3):132-7: 低患病率人群 PPV 塌缩
- WHO Global TB Report 2023: 全人群发病率量级 100/10万
- Fox GJ et al. Lancet Infect Dis 2013;13(5):369-381: 密接检出率 3.1%
"""

from ..constants import PREVALENCE_CALIBERS, PREVALENCE_CALIBER_MIXING_WARNING

__all__ = [
    'ppv_at_cutoff',
    'operating_point',
    'build_decision_reference',
    'percentile_ranks',
    'format_decision_reference_summary',
]


def percentile_ranks(scores):
    """分数 → 群内分位（0-100，最高分 = 100%）。

    排序决策形式的数据基础：界面"分位"列 = 该接触者在本次评估队列中
    的风险分位，而非绝对概率——不依赖概率口径的绝对正确性。

    并列分数取平均秩（与 scipy.stats.rankdata 一致）。

    Args:
        scores: 分数序列（任意量纲，仅用于排序）

    Returns:
        list[float]: 与输入等长的分位列表（0 < p ≤ 100）

    Raises:
        ValueError: 空输入
    """
    n = len(scores)
    if n == 0:
        raise ValueError('scores 为空')
    # 升序排序的索引；并列时用平均秩
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # 秩从 1 开始：order[i..j] 的样本并列，占秩 i+1..j+1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = 100.0 * avg_rank / n
        i = j + 1
    return ranks


def ppv_at_cutoff(sensitivity, specificity, prevalence):
    """贝叶斯换算：给定敏感度/特异度与人群阳性率，计算阳预测值 PPV。

    PPV = sens·π / (sens·π + (1-spec)·(1-π))

    Args:
        sensitivity: 敏感度，[0, 1]
        specificity: 特异度，[0, 1]
        prevalence: 目标人群阳性率（先验），开区间 (0, 1)

    Returns:
        float: PPV（阳性预测值），[0, 1]

    Raises:
        ValueError: 任一参数越界
    """
    sensitivity = float(sensitivity)
    specificity = float(specificity)
    prevalence = float(prevalence)
    if not (0.0 <= sensitivity <= 1.0):
        raise ValueError(f'敏感度须在 [0, 1]，收到 {sensitivity}')
    if not (0.0 <= specificity <= 1.0):
        raise ValueError(f'特异度须在 [0, 1]，收到 {specificity}')
    if not (0.0 < prevalence < 1.0):
        raise ValueError(f'阳性率须在开区间 (0, 1)，收到 {prevalence}')
    numerator = sensitivity * prevalence
    denominator = numerator + (1.0 - specificity) * (1.0 - prevalence)
    if denominator <= 0.0:
        raise ValueError('PPV 分母非正：sens·π + (1-spec)·(1-π) ≤ 0')
    return numerator / denominator


def operating_point(scores, labels, top_fraction=0.1):
    """排序+截断点：按分数降序取前 top_fraction 比例为"阳性判定"。

    这是问题4输出侧的核心决策形式：不依赖绝对概率的绝对正确性，
    只依赖排序质量——筛查资源有限时按分数从高到低覆盖前 k% 人群。

    Args:
        scores: 风险分数序列（任意量纲，仅用于排序）
        labels: 真实标签序列（0/1），与 scores 等长
        top_fraction: 截断比例，(0, 1]，如 0.1 = 覆盖风险最高的 10%

    Returns:
        dict: {
            'sensitivity': 截断点敏感度（TP/P；P=0 时为 0.0），
            'specificity': 截断点特异度（TN/N；N=0 时为 0.0），
            'threshold':   决策阈值 = 最高未入选分数
                           （"严格大于该值者入选"；全部入选时取
                           min(scores)-1e-12），
            'n_flagged':   入选人数 k,
            'n_total':     总人数,
            'top_fraction': 截断比例,
        }

    Raises:
        ValueError: 长度不一致 / 空输入 / top_fraction 越界
    """
    if len(scores) != len(labels):
        raise ValueError(f'scores 与 labels 长度不一致: {len(scores)} vs {len(labels)}')
    n = len(scores)
    if n == 0:
        raise ValueError('scores 为空')
    top_fraction = float(top_fraction)
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError(f'top_fraction 须在 (0, 1]，收到 {top_fraction}')

    k = max(1, int(round(top_fraction * n)))
    k = min(k, n)
    # 稳定降序：同分按原始顺序先到先入选
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    flagged = set(order[:k])

    tp = sum(1 for i in flagged if int(labels[i]) == 1)
    fp = k - tp
    p = sum(1 for v in labels if int(v) == 1)
    n_neg = n - p

    sensitivity = (tp / p) if p > 0 else 0.0
    specificity = ((n_neg - fp) / n_neg) if n_neg > 0 else 0.0

    unflagged_scores = [scores[i] for i in range(n) if i not in flagged]
    if unflagged_scores:
        threshold = max(unflagged_scores)
    else:
        # 全部入选：阈值压到最低分之下
        threshold = min(scores) - 1e-12

    return {
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'threshold': float(threshold),
        'n_flagged': int(k),
        'n_total': int(n),
        'top_fraction': top_fraction,
    }


def build_decision_reference(scores, labels, top_fraction=0.1, calibers=None,
                             local_prevalence=None, local_label=None):
    """构建决策参考表：截断点操作特性 + 双口径 PPV + 口径警示。

    Args:
        scores: 风险分数序列
        labels: 真实标签序列（0/1）
        top_fraction: 截断比例（默认 0.1 = 覆盖前 10%）
        calibers: 阳性率口径字典（默认 constants.PREVALENCE_CALIBERS，
                  测试可注入）
        local_prevalence: 训练队列实际阳性率（0-1 开区间）。提供时输出
            第三口径 cohort_local——部署人群与训练队列同源时的首要
            读数。动机（缺陷1，2026-09-17）：固定参照口径（密接
            2.85%）在低患病率队列上高估 PPV 4-5 倍（kenya 实际
            0.53%，密接口径 PPV 21% vs 本地口径约 4.6%）。
        local_label: 队列来源标注（如 'kenya 训练样本'）

    Returns:
        dict: {
            'decision_form': 'ranking_cutoff'（排序+截断点决策形式）,
            'top_fraction': 截断比例,
            'n_flagged': 入选人数,
            'operating_point': operating_point() 的完整输出,
            'ppv': {口径键: {'prevalence', 'ppv', 'display', 'label',
                    （cohort_local 另含 'source'/'note'）}},
            'note': 口径不可混用警示（含 cohort_local 指引，若提供）,
        }
    """
    calibers = PREVALENCE_CALIBERS if calibers is None else calibers
    op = operating_point(scores, labels, top_fraction)

    ppv_blocks = {}
    for key, caliber in calibers.items():
        ppv_blocks[key] = {
            'prevalence': float(caliber['value']),
            'ppv': ppv_at_cutoff(op['sensitivity'], op['specificity'],
                                 caliber['value']),
            'display': caliber['display'],
            'label': caliber['label'],
        }

    note = PREVALENCE_CALIBER_MIXING_WARNING
    if local_prevalence is not None:
        lp = float(local_prevalence)
        if not (0.0 < lp < 1.0):
            raise ValueError(
                f'队列实际阳性率须在开区间 (0, 1)，收到 {local_prevalence}')
        ppv_blocks['cohort_local'] = {
            'prevalence': lp,
            'ppv': ppv_at_cutoff(op['sensitivity'], op['specificity'], lp),
            'display': f'{lp:.2%}',
            'label': local_label or '训练队列实际阳性率',
            'source': '训练数据终点阳性率（y 均值）',
            'note': ('部署人群与训练队列同源时以此口径为首要读数；'
                     '病例富集设计（case-control）的分析样本阳性率'
                     '不等于部署人群先验，该场景下仅作换算参考'),
        }
        note = (note + ' 第三口径 cohort_local 为训练队列实际阳性率：'
                '与训练队列同人群的站点读表以该口径为准（固定参照口径'
                '在低患病率队列上会高估 PPV 数倍）。')

    return {
        'decision_form': 'ranking_cutoff',
        'top_fraction': op['top_fraction'],
        'n_flagged': op['n_flagged'],
        'operating_point': op,
        'ppv': ppv_blocks,
        'note': note,
    }


def format_decision_reference_summary(ref):
    """把 decision_reference 压缩成一行中文摘要（界面/报告共用）。

    含决策形式、截断覆盖、操作特性、双口径 PPV 与"不可混用"警示，
    供 GUI 状态栏与训练报告直接渲染。

    Args:
        ref: build_decision_reference() 的返回值

    Returns:
        str: 单行摘要
    """
    op = ref.get('operating_point', {})
    ppv = ref.get('ppv', {})
    cc = ppv.get('close_contact', {})
    gp = ppv.get('general_population', {})
    lc = ppv.get('cohort_local', {})
    local_seg = ''
    if lc:
        local_seg = (f"；队列实际({lc.get('display', '')}) "
                     f"{lc.get('ppv', 0):.1%}")
    return (
        f"决策形式：排序+截断点｜覆盖风险最高 {ref.get('top_fraction', 0):.0%}"
        f"（{op.get('n_flagged', 0)}/{op.get('n_total', 0)} 人）｜"
        f"敏感度 {op.get('sensitivity', 0):.0%}、"
        f"特异度 {op.get('specificity', 0):.0%}｜"
        f"PPV 换算：密接人群({cc.get('display', '')}) "
        f"{cc.get('ppv', 0):.1%}；全人群({gp.get('display', '')}) "
        f"{gp.get('ppv', 0):.1%}{local_seg}｜"
        f"两套口径不可混用，绝对概率仅作参考"
    )
