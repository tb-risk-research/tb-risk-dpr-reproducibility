#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 提示词模板集合。

Layer 3-5: 各业务场景的提示词模板。

模块组织：
- EXTRACTION_SYSTEM_PROMPT:        文本抽取系统提示
- EXTRACTION_USER_TEMPLATE:        文本抽取用户提示模板（{text} 占位符）
- REPORT_SYSTEM_PROMPT:            风险报告生成系统提示
- REPORT_USER_TEMPLATE:            风险报告生成用户提示模板（{context} 占位符）
- QA_SYSTEM_PROMPT:                智能问答系统提示
- QA_USER_TEMPLATE:                智能问答用户提示模板（{context}/{question} 占位符）
- DATA_QUALITY_SYSTEM_PROMPT:      数据质量诊断系统提示
- DATA_QUALITY_USER_TEMPLATE:      数据质量诊断用户提示模板（{record} 占位符）

缺陷 #15：支持外置提示词目录。get_prompt(name, prompts_dir) 优先读取
prompts_dir/<name>.txt，失败回退到模块级默认常量。
不同医院/科室可通过 ~/.tb_risk/prompts/ 自定义提示词，无需修改源码。
"""

from __future__ import annotations

import os
from typing import Dict, Optional

# ==============================================================================
# 文本抽取提示词
# ==============================================================================

EXTRACTION_SYSTEM_PROMPT = """你是结核病临床文本结构化抽取助手。任务：从输入的临床叙述文本中抽取结构化字段，输出 JSON。

抽取字段（仅输出以下字段，未知字段不要输出）：
- age (int): 年龄，如 35
- gender (str): 性别，"男" 或 "女"
- sputum_smear (int): 痰涂片，1=阳性/涂阳，0=阴性/涂阴
- has_cavity (int): 是否有空洞，1=有，0=无
- bcg_vaccine (int): 卡介苗接种，1=已接种，0=未接种
- treatment (int): 治疗状态，1=已治疗/正在治疗，0=未治疗
- treatment_duration (int): 治疗时长（月），如 6
- ventilation (str): 通风条件，"极差"/"差"/"一般"/"好"/"极好"
- contact_distance (str): 接触距离，"极近"/"近"/"中等"/"远"/"极远"
- exposure_setting (str): 暴露场所，"拥挤"/"密闭"/"一般"/"户外"/"油田"/"营地"
- has_tb (int): 既往结核史，1=有，0=无
- cough_freq (int): 咳嗽频率（次/天），如 5
- delay_days (int): 延迟就诊天数，如 14
- family_living_conditions (str): 居住条件，"极差"/"差"/"一般"/"好"/"极好"
- bmi (float): BMI，如 22.5
- weight (float): 体重（kg），如 70.5
- ethnicity (str): 民族，如 "汉族"/"维吾尔族"/"哈萨克族"

输出格式（严格遵守）：
- 多个患者时返回数组：{"records": [{...}, {...}]}
- 单个患者时可返回 {"records": [{...}]} 或直接 {...}
- 字段未在文本中出现时不要输出该字段（不要输出 null）
- 仅输出 JSON，不要有任何解释文字、不要 markdown 代码块"""

EXTRACTION_USER_TEMPLATE = """请从以下结核病临床叙述文本中抽取结构化字段：

---
{text}
---

请按系统提示的 JSON 格式输出抽取结果。"""


# ==============================================================================
# 风险报告生成提示词
# ==============================================================================

REPORT_SYSTEM_PROMPT = """你是结核病风险评估报告撰写助手。任务：根据输入的风险评估数据，生成符合中国结核病防治规范的临床风险评估报告。

报告要求：
1. 使用中文，符合医院临床文档规范
2. 结构：评估摘要 / 关键风险因素 / 接触者风险分层 / 干预建议 / 不确定性说明
3. 数据驱动：报告中引用具体数值（概率、风险评分、SEIR 参数）
4. 区分规则风险与 SEIR 动力学风险
5. 给出明确的干预优先级排序
6. 说明 ML 预测的不确定性（如适用 SHAP 特征重要性）
7. 报告长度 500-1000 字

输出：纯文本报告（非 JSON、非 markdown）。"""

REPORT_USER_TEMPLATE = """请基于以下风险评估数据生成中文临床风险评估报告：

---
{context}
---

请按系统提示的结构生成报告。"""


# ==============================================================================
# 智能问答提示词
# ==============================================================================

QA_SYSTEM_PROMPT = """你是结核病风险咨询助手。任务：基于当前评估结果上下文，回答用户关于结核病风险的问题。

回答要求：
1. 使用中文
2. 回答必须基于提供的上下文数据，不要编造
3. 如果上下文不足以回答，明确说明需要哪些额外信息
4. 涉及医学判断时提醒用户咨询专业医生
5. 可以解释风险评分的计算逻辑、SEIR 参数含义、干预策略
6. 回答简洁明了，避免冗长

上下文包含：当前评估的患者信息、接触者风险、SEIR 参数、ML 预测等。

可选参考知识（如提供）：
- karamay/ 本土化参数（油田营地特殊暴露、民族分布、气候因素）
- 结核病防治指南片段"""

QA_USER_TEMPLATE = """【当前评估上下文】
{context}

【参考知识】
{knowledge}

【用户问题】
{question}

请基于上下文回答用户问题。"""


# ==============================================================================
# 数据质量诊断提示词
# ==============================================================================

DATA_QUALITY_SYSTEM_PROMPT = """你是结核病数据质量诊断助手。任务：分析质量评分较低的记录，给出具体的字段填充或修正建议。

输出 JSON 格式：
{
  "issues": [
    {
      "field": "字段名",
      "problem": "问题描述",
      "suggestion": "具体建议（含建议值）",
      "confidence": 0.0-1.0
    }
  ],
  "overall_quality": "低/中/高",
  "summary": "整体诊断摘要"
}

诊断维度：
- 字段缺失（必填字段为空）
- 字段异常（如年龄 > 120、BMI > 60）
- 字段冲突（如性别与既往史不符）
- 业务规则（如油田工人应标注 oilfield_camp 暴露）

仅输出 JSON，不要解释文字。"""

DATA_QUALITY_USER_TEMPLATE = """请诊断以下患者/接触者记录的数据质量问题：

---
{record}
---

按系统提示的 JSON 格式输出诊断结果。"""


__all__ = [
    'EXTRACTION_SYSTEM_PROMPT',
    'EXTRACTION_USER_TEMPLATE',
    'REPORT_SYSTEM_PROMPT',
    'REPORT_USER_TEMPLATE',
    'QA_SYSTEM_PROMPT',
    'QA_USER_TEMPLATE',
    'DATA_QUALITY_SYSTEM_PROMPT',
    'DATA_QUALITY_USER_TEMPLATE',
    # 缺陷 #15：按需加载函数
    'get_prompt',
    'load_all_prompts',
    'DEFAULT_PROMPT_NAMES',
]


# ==============================================================================
# 缺陷 #15：外置提示词加载
# ==============================================================================

# 所有已知提示词名（用于 load_all_prompts 遍历）
DEFAULT_PROMPT_NAMES = (
    'EXTRACTION_SYSTEM_PROMPT',
    'EXTRACTION_USER_TEMPLATE',
    'REPORT_SYSTEM_PROMPT',
    'REPORT_USER_TEMPLATE',
    'QA_SYSTEM_PROMPT',
    'QA_USER_TEMPLATE',
    'DATA_QUALITY_SYSTEM_PROMPT',
    'DATA_QUALITY_USER_TEMPLATE',
)


def _get_default_prompt(name: str) -> Optional[str]:
    """从模块全局命名空间获取默认提示词常量

    Args:
        name: 提示词名（如 'EXTRACTION_SYSTEM_PROMPT'）

    Returns:
        str or None: 默认提示词内容；未知名称返回 None
    """
    import sys
    module = sys.modules.get(__name__)
    if module is None:
        return None
    return getattr(module, name, None)


def get_prompt(name: str, prompts_dir: Optional[str] = None) -> Optional[str]:
    """按需获取提示词内容（缺陷 #15）

    优先级：
    1. prompts_dir/<name>.txt 文件（若 prompts_dir 非空且文件存在）
    2. 模块级默认常量（EXTRACTION_SYSTEM_PROMPT 等）

    Args:
        name: 提示词名（如 'EXTRACTION_SYSTEM_PROMPT'）
        prompts_dir: 外置提示词目录路径，None 时不查文件

    Returns:
        str or None: 提示词内容；未知名称返回 None
    """
    # 1. 尝试从外置目录读取
    if prompts_dir:
        file_path = os.path.join(prompts_dir, f'{name}.txt')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 剥离文件末尾的换行符（编辑器常自动追加），避免格式干扰
                return f.read().rstrip('\n')
        except (FileNotFoundError, OSError):
            pass  # 文件不存在或不可读，回退到默认

    # 2. 回退到模块级默认常量
    return _get_default_prompt(name)


def load_all_prompts(prompts_dir: Optional[str] = None) -> Dict[str, str]:
    """批量加载所有已知提示词（缺陷 #15）

    Args:
        prompts_dir: 外置提示词目录路径，None 时仅返回所有默认提示词

    Returns:
        dict: {提示词名: 内容}，包含所有 DEFAULT_PROMPT_NAMES 中的提示词。
        prompts_dir 中存在的文件会覆盖对应默认值。
    """
    result: Dict[str, str] = {}
    for name in DEFAULT_PROMPT_NAMES:
        content = get_prompt(name, prompts_dir)
        if content is not None:
            result[name] = content
    return result
