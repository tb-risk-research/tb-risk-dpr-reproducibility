#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开数据验证策略：明确"公开数据能验证什么"（改进三-第三步）。

结论（写入方案）：
- **任务 A/B**：个体模型可用公开数据做**外部验证**（TB Portals、SMH-TB、
  ERASE-TB、中国疾控周报 2024 学校队列）。
- **任务 C**：网络/传播层公开数据不足——公开的"患者-接触者关联 + 纵向随访"
  数据极稀缺，方案中**显式说明**，并设计
  **"合成网络 + 文献参数校准 + 敏感性分析"**作为过渡验证方案；
  把真实网络数据列为**合作数据需求的第一优先级**。

本模块提供：
1. ``EXTERNAL_VALIDATION_PLAN`` —— 逐任务的验证策略（能验证什么 / 怎么验证 /
   指标 / 局限）；
2. ``transition_validation_steps()`` —— 任务 C 过渡验证的实施步骤；
3. ``cooperation_data_needs()`` —— 合作数据需求（按优先级排序）；
4. ``build_validation_strategy()`` —— 汇总为结构化策略；
5. ``render_strategy_document()`` —— 渲染为可发布的 Markdown 文档。

不产生新的模型/数据；仅把验证设计文档化并可与既有模块衔接：
- 公开数据集目录：``validation.public_datasets``
- 任务评估函数：``validation.task_decomposition.evaluate_task_*``
- 敏感性分析：``validation.sensitivity.MorrisSensitivityAnalyzer``
"""

import logging

from .public_datasets import (
    LAYER_INDIVIDUAL,
    LAYER_CONTACT,
    LAYER_TRANSMISSION,
    list_public_datasets,
    get_public_dataset,
)

LOGGER = logging.getLogger("tb_risk.validation.public_data_strategy")


# ==============================================================================
# 逐任务验证策略
# ==============================================================================

EXTERNAL_VALIDATION_PLAN = {
    'A': {
        'task_name': '横断面活动性结核筛查',
        'can_external_validate': True,
        'datasets': ['tb_portals', 'smh_tb'],
        'what': ('个体筛查模型在外部人群上的判别与校准外部验证'
                 '（敏感度/特异度/PPV/NPV、约登切点、校准曲线）'),
        'metrics': ['sensitivity', 'specificity', 'ppv', 'npv', 'ece'],
        'design': ('在外部数据集上重跑任务 A 评估（evaluate_task_a），'
                   '报告外部敏感度/特异度/PPV/NPV 与 ECE；'
                   '对比内部 vs 外部指标差'),
        'caveat': ('外部人群与克拉玛依存在域偏移（分布/患病率不同），'
                   '应报告外部-内部指标差并据此判断是否需要重新校准阈值'),
    },
    'B': {
        'task_name': '感染后进展为活动性结核的风险预测',
        'can_external_validate': True,
        'datasets': ['tb_portals', 'smh_tb', 'erase_tb',
                     'china_ccdc_school_2024'],
        'what': ('接触者感染/进展风险预测的外部验证'
                 '（C-index、固定时点 AUC、校准、DCA）'),
        'metrics': ['c_index', 'horizon_auc', 'ece', 'dca'],
        'design': ('优先用 ERASE-TB 家庭接触者队列（1,905 人 IGRA 结局）'
                   '与国内学校队列（6,893 人）对标"接触者早期风险识别"；'
                   '在外部数据上重跑任务 B 评估（evaluate_task_b）'),
        'caveat': ('任务 B 需要随访时间-事件标签；公开队列的随访窗口与'
                   '6–24 月时间窗可能不完全一致，需对齐定义'),
    },
    'C': {
        'task_name': '接触网络传播预测',
        'can_external_validate': False,
        'datasets': ['bc_wgs_clusters'],
        'reason': ('公开的"患者-接触者关联 + 纵向随访"数据极稀缺；'
                   'BC-WGS 仅提供方法学参考（WGS 预测传播簇），'
                   '数据本身仍需合作获取'),
        'transition': {
            'approach': '合成网络 + 文献参数校准 + 敏感性分析',
            'steps': [
                '合成网络：复用传播链 DGP（种子→接触者）与网络增强层 '
                    'build_synthetic_network 生成接触网络（度分布/簇结构）',
                '文献参数校准：R_eff(0.8–1.2)、家庭接触者感染率(20–50%)、'
                    '最大代际(≤3) 等参数校准网络生成（LITERATURE_REFERENCES）',
                '敏感性分析：对 R_eff / 接触者数 / 感染率等参数做敏感性分析'
                    '（复用 validation.sensitivity.MorrisSensitivityAnalyzer）',
                '报告网络层指标：命中率 / 排序质量 / R 估计 / 接触者追踪效率'
                    '（evaluate_task_c）',
                '真实网络数据到位后：切换为外部验证 + 参数再校准',
            ],
        },
        'note': '此层以"公开数据校准参数 + 合成网络"为主；'
                '结论仅供内部评估，不作为临床决策依据，直至真实网络数据验证。',
    },
}


def _task_plan(task: str) -> dict:
    """读取某任务的验证计划（未知任务抛 KeyError）。"""
    if task not in EXTERNAL_VALIDATION_PLAN:
        raise KeyError(f"未知任务: {task}，可用: {list(EXTERNAL_VALIDATION_PLAN)}")
    return EXTERNAL_VALIDATION_PLAN[task]


def transition_validation_steps() -> list:
    """任务 C 过渡验证实施步骤（合成网络 + 文献校准 + 敏感性分析）。"""
    return list(EXTERNAL_VALIDATION_PLAN['C']['transition']['steps'])


# ==============================================================================
# 合作数据需求（按优先级）
# ==============================================================================

COOPERATION_DATA_NEEDS = [
    {
        'priority': 1,
        'data': '真实"患者-接触者关联 + 纵向随访"网络数据',
        'layer': LAYER_TRANSMISSION,
        'tasks': ['C'],
        'why': '任务 C 外部验证的唯一可靠来源；替换合成网络的过渡验证，'
               '提供真实度分布/传播簇/二代病例标签',
        'example': '地区结核病报告 + 接触者追踪 + 随访结局的联动数据'
                   '（含 IGRA/TST 与发病时间）',
    },
    {
        'priority': 2,
        'data': '本地（克拉玛依）真实筛查个体数据',
        'layer': LAYER_INDIVIDUAL,
        'tasks': ['A', 'B'],
        'why': '替换/校准合成数据，消除任务 A/B 的域偏移，'
               '支持阈值与基线再标定（SyntheticDataCalibrator 对接）',
        'example': '重点人群筛查电子记录（人口学/接触/临床/环境 + 确诊标签）',
    },
    {
        'priority': 3,
        'data': '国内学校/社区密切接触者随访队列',
        'layer': LAYER_CONTACT,
        'tasks': ['B'],
        'why': '任务 B 的国内分布校准与外部验证（对齐 6–24 月随访时间窗）',
        'example': '密切接触者 IGRA 基线 + 6–24 月随访发病数据',
    },
]


def cooperation_data_needs() -> list:
    """合作数据需求清单（按优先级升序返回）。"""
    return [dict(n) for n in COOPERATION_DATA_NEEDS]


def top_cooperation_data_need() -> dict:
    """返回第一优先级的合作数据需求（真实网络数据）。"""
    return dict(COOPERATION_DATA_NEEDS[0])


# ==============================================================================
# 策略汇总与文档渲染
# ==============================================================================

def build_validation_strategy() -> dict:
    """汇总公开数据验证策略（结构化 dict）。"""
    return {
        'title': '公开数据验证策略：明确"公开数据能验证什么"',
        'plan': {task: dict(_task_plan(task)) for task in ('A', 'B', 'C')},
        'cooperation_data_needs': cooperation_data_needs(),
        'summary': {
            'A': '可外部验证（个体层：TB Portals / SMH-TB）',
            'B': '可外部验证（接触者层：ERASE-TB / 学校队列为主）',
            'C': '公开数据不足 → 合成网络 + 文献校准 + 敏感性分析过渡，'
                 '真实网络数据为第一优先级合作需求',
        },
    }


def render_strategy_document(strategy=None) -> str:
    """把验证策略渲染为可发布的 Markdown 文档。"""
    if strategy is None:
        strategy = build_validation_strategy()

    from .public_datasets import PRIORITY_LABELS

    lines = []
    lines.append(f"# {strategy['title']}")
    lines.append('')
    lines.append('> 结论：任务 A/B 的个体模型可用公开数据做**外部验证**；'
                 '任务 C 的网络/传播层公开数据不足，需显式说明并采用'
                 '**过渡验证方案**，同时把真实网络数据列为'
                 '**合作数据需求的第一优先级**。')
    lines.append('')

    for task in ('A', 'B', 'C'):
        plan = strategy['plan'][task]
        lines.append(f"## 任务 {task}：{plan['task_name']}")
        lines.append('')
        if plan['can_external_validate']:
            lines.append(f"- **可外部验证**：是。")
            lines.append(f"- **能验证什么**：{plan['what']}。")
            lines.append(f"- **指标**：{', '.join(plan['metrics'])}。")
            lines.append(f"- **验证设计**：{plan['design']}。")
            lines.append(f"- **局限**：{plan['caveat']}。")
            lines.append(f"- **匹配数据集**：")
            for ds_id in plan['datasets']:
                try:
                    ds = get_public_dataset(ds_id)
                    lines.append(f"  - `{ds_id}` {ds['name']}（{ds['size']}）")
                except KeyError:
                    lines.append(f"  - `{ds_id}`")
        else:
            lines.append(f"- **可外部验证**：**否**。")
            lines.append(f"- **原因**：{plan['reason']}。")
            lines.append(f"- **过渡验证方案**：{plan['transition']['approach']}。")
            for i, step in enumerate(plan['transition']['steps'], 1):
                lines.append(f"  {i}. {step}")
            lines.append(f"- **注意**：{plan['note']}。")
        lines.append('')

    lines.append('## 合作数据需求（按优先级）')
    lines.append('')
    lines.append('| 优先级 | 数据 | 层级 | 任务 | 理由 |')
    lines.append('| --- | --- | --- | --- | --- |')
    for need in strategy['cooperation_data_needs']:
        from .public_datasets import LAYER_LABELS
        lines.append(
            f"| {PRIORITY_LABELS.get(need['priority'], need['priority'])} "
            f"| {need['data']} | {LAYER_LABELS.get(need['layer'], need['layer'])} "
            f"| {'/'.join(need['tasks'])} | {need['why']} |")
    lines.append('')

    lines.append('## 摘要')
    lines.append('')
    for task in ('A', 'B', 'C'):
        lines.append(f"- 任务 {task}：{strategy['summary'][task]}。")
    lines.append('')

    return '\n'.join(lines)


__all__ = [
    'EXTERNAL_VALIDATION_PLAN',
    'COOPERATION_DATA_NEEDS',
    'transition_validation_steps',
    'cooperation_data_needs',
    'top_cooperation_data_need',
    'build_validation_strategy',
    'render_strategy_document',
]
