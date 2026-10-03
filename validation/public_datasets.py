#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开数据集目录：按任务（A/B/C）与层级（个体/接触者/传播）匹配。

改进三-第二步：把可用的公开数据资源组织为结构化目录，按
"任务 A/B 优先、任务 C 靠后"的顺序接入，用于现实校准与外部验证。

层级划分：
- ``individual``   个体层（任务 A/B）：个体风险模型的特征与标签设计
- ``contact``      接触者层（任务 B 最相关）：接触者早期风险识别
- ``transmission`` 传播层（任务 C）：网络/传播簇预测

覆盖的公开资源（来源为用户核查 + WHO/CDC 官方渠道）：
- TB Portals：开放获取多域数据（临床/影像/基因组，已匿名化，2000+ 样本）
- SMH-TB：St. Michael's Hospital 结核 EHR 回顾性队列（伦理审批可申请）
- ERASE-TB：坦桑尼亚/莫桑比克/津巴布韦 1905 名家庭接触者 IGRA 队列
- 中国疾控周报 2024 河南/广州学校密切接触者（6,893 人）潜伏感染 ML 建模
- BC-WGS：不列颠哥伦比亚省 WGS 传播簇预测方法学（数据需合作获取）

重要边界：本目录仅做**元数据记录**（不下载、不缓存数据）。实际接入前须核对
各数据集的现行数据使用协议/许可与获取条件。

URL 说明：仅收录我方可确认稳定的官方入口；其余标注"需查看数据可用性声明"。
"""

import logging

LOGGER = logging.getLogger("tb_risk.validation.public_datasets")

# 层级常量
LAYER_INDIVIDUAL = 'individual'
LAYER_CONTACT = 'contact'
LAYER_TRANSMISSION = 'transmission'

LAYER_LABELS = {
    LAYER_INDIVIDUAL: '个体层',
    LAYER_CONTACT: '接触者层',
    LAYER_TRANSMISSION: '传播层',
}

# 接入优先级（任务 A/B 优先、任务 C 靠后）
PRIORITY_LABELS = {1: '优先（任务 A/B）', 2: '次优先（任务 B）', 3: '靠后（任务 C）'}

# 任务覆盖
TASKS = ('A', 'B', 'C')


PUBLIC_DATASETS = [
    # ------------------------------------------------------------------------
    # 个体层（任务 A/B）— 优先接入
    # ------------------------------------------------------------------------
    {
        'id': 'tb_portals',
        'name': 'TB Portals（开放获取多域数据库）',
        'layer': LAYER_INDIVIDUAL,
        'tasks': ['A', 'B'],
        'access': 'open',                      # 开放获取
        'size': '2000+ 样本（多国）',
        'data': '临床 / 影像 / 基因组，已匿名化',
        'use': '个体风险模型的特征与标签设计；任务 A 外部验证'
               '（活动性 TB 表型判别）',
        'priority': 1,
        'source': 'TB Portals（NIAID 支持的开源多国结核数据库）',
        'url': 'https://tbportals.niaid.nih.gov/',
        'note': '开放获取；数据使用须遵守其许可条款（推荐引用其数据声明）',
    },
    {
        'id': 'smh_tb',
        'name': 'St. Michael\'s Hospital TB 数据库（SMH-TB）',
        'layer': LAYER_INDIVIDUAL,
        'tasks': ['A', 'B'],
        'access': 'application',               # 经伦理审批可申请
        'size': '回顾性 EHR 队列',
        'data': '电子健康档案（EHR）个体级记录',
        'use': '任务 A/B 个体模型的 EHR 外部验证（真实临床特征分布）',
        'priority': 1,
        'source': 'St. Michael\'s Hospital / University of Toronto',
        'url': None,
        'note': '回顾性 EHR 队列，经伦理审批 + 数据共享协议可申请研究使用',
    },
    # ------------------------------------------------------------------------
    # 接触者层（任务 B 最相关）— 次优先
    # ------------------------------------------------------------------------
    {
        'id': 'erase_tb',
        'name': 'ERASE-TB 家庭接触者队列',
        'layer': LAYER_CONTACT,
        'tasks': ['B'],
        'access': 'open_availability',         # 有数据可用性声明
        'size': '1,905 名家庭接触者',
        'data': 'IGRA 结局 + 索引病例 / 家庭 / 接触者因素',
        'use': '任务 B 直接对标：接触者感染预测模型外部验证'
               '（C-index / DCA / 校准）',
        'priority': 2,
        'source': 'ERASE-TB 研究（坦桑尼亚 / 莫桑比克 / 津巴布韦）',
        'url': None,
        'note': '开发了接触者感染预测模型，是"接触者早期风险识别"的'
                '公开对标队列；需按其数据可用性声明申请',
    },
    {
        'id': 'china_ccdc_school_2024',
        'name': '中国疾控周报 2024 学校密切接触者研究',
        'layer': LAYER_CONTACT,
        'tasks': ['B'],
        'access': 'author_contact',            # 联系作者 / 复现方法
        'size': '6,893 名学校密切接触者（河南/广州）',
        'data': '密切接触者潜伏感染风险 ML 建模（因素 + 结局）',
        'use': '任务 B 方法学复现与国内分布校准；可联系作者获取',
        'priority': 2,
        'source': 'China CDC Weekly（中国疾控周报）2024',
        'url': None,
        'note': '国内学校密切接触者潜伏感染风险 ML 建模；可联系作者'
                '获取数据或复现其方法',
    },
    # ------------------------------------------------------------------------
    # 传播层（任务 C）— 靠后
    # ------------------------------------------------------------------------
    {
        'id': 'bc_wgs_clusters',
        'name': '不列颠哥伦比亚省 WGS 传播簇方法学',
        'layer': LAYER_TRANSMISSION,
        'tasks': ['C'],
        'access': 'methodology_open',          # 方法学公开，数据需合作
        'size': '—（方法学）',
        'data': 'WGS 聚类 + 患者-接触者关联（数据需合作获取）',
        'use': '任务 C 方法学参考：WGS 预测传播簇、传播链分析框架',
        'priority': 3,
        'source': 'British Columbia（不列颠哥伦比亚省）结核传播研究',
        'url': None,
        'note': '公开的"患者-接触者关联 + 纵向随访"数据极稀缺；'
                '此层以"公开数据校准参数 + 合成网络"为主，'
                '真实网络数据列为合作数据需求第一优先级',
    },
]


def list_public_datasets(task=None, layer=None) -> list:
    """按任务 / 层级过滤公开数据集目录（按优先级升序返回）。

    参数：
        task (str|None): 'A' / 'B' / 'C'
        layer (str|None): LAYER_INDIVIDUAL / LAYER_CONTACT / LAYER_TRANSMISSION

    返回：
        list[dict]: 匹配的数据集（深拷贝，避免调用方误改目录）
    """
    import copy

    results = []
    for ds in PUBLIC_DATASETS:
        if task is not None and task not in ds['tasks']:
            continue
        if layer is not None and ds['layer'] != layer:
            continue
        results.append(copy.deepcopy(ds))
    results.sort(key=lambda d: d['priority'])
    return results


def get_public_dataset(dataset_id: str) -> dict:
    """按 id 获取单个数据集条目；不存在则抛 KeyError。"""
    for ds in PUBLIC_DATASETS:
        if ds['id'] == dataset_id:
            import copy
            return copy.deepcopy(ds)
    raise KeyError(f"未知数据集 id: {dataset_id}，可用: "
                   f"{[d['id'] for d in PUBLIC_DATASETS]}")


def _dataset_external_validatable(ds: dict) -> bool:
    """判断数据集是否可提供真实数据用于外部验证。

    仅"方法学公开"（methodology_open）视为不可验证——数据本身仍需合作获取；
    其余获取方式（开放 / 申请 / 数据可用性声明 / 联系作者）均视为可验证。
    """
    return ds.get('access') != 'methodology_open'


def dataset_coverage_by_task() -> dict:
    """每个任务的外部验证数据覆盖情况。

    返回：
        dict: {task: {'datasets': [...], 'n': int,
                      'external_validatable': bool, 'methodology_only': bool}}
    """
    coverage = {}
    for task in TASKS:
        datasets = list_public_datasets(task=task)
        validatable = [d for d in datasets if _dataset_external_validatable(d)]
        coverage[task] = {
            'datasets': datasets,
            'n': len(datasets),
            'external_validatable': len(validatable) > 0,
            'methodology_only': datasets and len(validatable) == 0,
        }
    return coverage


def recommended_access_order() -> list:
    """推荐接入顺序：任务 A/B 优先、任务 C 靠后（按优先级分组）。"""
    order = []
    for priority in (1, 2, 3):
        group = [d['id'] for d in sorted(
            (x for x in PUBLIC_DATASETS if x['priority'] == priority),
            key=lambda x: x['id'])]
        if group:
            order.append({'priority': priority,
                          'label': PRIORITY_LABELS[priority],
                          'datasets': group})
    return order


def summarize_public_datasets() -> dict:
    """目录摘要（供 UI / 报告展示）。"""
    return {
        'total': len(PUBLIC_DATASETS),
        'by_layer': {
            LAYER_LABELS[LAYER_INDIVIDUAL]:
                len(list_public_datasets(layer=LAYER_INDIVIDUAL)),
            LAYER_LABELS[LAYER_CONTACT]:
                len(list_public_datasets(layer=LAYER_CONTACT)),
            LAYER_LABELS[LAYER_TRANSMISSION]:
                len(list_public_datasets(layer=LAYER_TRANSMISSION)),
        },
        'coverage_by_task': dataset_coverage_by_task(),
        'access_order': recommended_access_order(),
    }


__all__ = [
    'LAYER_INDIVIDUAL',
    'LAYER_CONTACT',
    'LAYER_TRANSMISSION',
    'LAYER_LABELS',
    'PRIORITY_LABELS',
    'TASKS',
    'PUBLIC_DATASETS',
    'list_public_datasets',
    'get_public_dataset',
    'dataset_coverage_by_task',
    'recommended_access_order',
    'summarize_public_datasets',
]
