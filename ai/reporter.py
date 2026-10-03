#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 风险报告生成模块。

Layer 5a: 把 RiskAssessmentService.assess() 返回的 result dict 喂给 LLM，
要求输出符合中国结核病防治规范的中文临床风险评估报告。

流程：
1. _build_context(result) → 把 result 关键字段组装成 LLM 可读的纯文本上下文
2. 构造 system + user 提示词
3. 调用 LLMClient.chat() 获取报告
4. 失败时返回 None（调用方降级到模板化报告）

支持字段（assess() 返回 dict）：
- patient_score (float): 患者风险评分
- potential_patients: {'family': [...], 'social': [...]}
- summary: {'overall_risk', 'total_contacts', 'high_risk_contacts', ...}
- ml_results: {'probability', 'top_features': [...], ...}
- seir_results: {'R0', 'peak_time', ...}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .client import make_client
from .config import AIConfig
from .prompts import get_prompt

LOGGER = logging.getLogger("tb_risk.ai.reporter")


# ==============================================================================
# 上下文构造
# ==============================================================================

def _build_context(result: Optional[Dict[str, Any]],
                   skip_redact: bool = False) -> str:
    """把 assess() 返回的 result dict 组装成 LLM 可读的纯文本上下文。

    体积约束（避免上下文膨胀触发模型窗口限制 / 显著增加成本）：
    - 接触者列表：仅展示前 20 条（按 family → social 顺序）
    - ML top_features：仅保留前 5 个
    - SEIR 结果：仅保留关键标量（R0/peak_time/peak_size 等），
      排除 weekly_incidence / weekly_prevalence 等时间序列

    Args:
        result: assess() 返回的 dict
        skip_redact: 是否跳过脱敏（本地模式数据不出本机时传 True）

    Returns:
        str: 组装后的上下文文本。空 result 返回空字符串。
    """
    if not isinstance(result, dict) or not result:
        return ''

    lines: List[str] = []

    # --- 1. 患者主评分 ---
    patient_score = result.get('patient_score')
    if patient_score is None:
        # 没有 patient_score 视为无效 result
        return ''
    lines.append(f"【患者风险评分】{patient_score}")

    # --- 2. 汇总信息 ---
    summary = result.get('summary') or {}
    if summary:
        lines.append("【评估汇总】")
        for k, v in summary.items():
            lines.append(f"  - {k}: {v}")

    # --- 3. 接触者风险分层（仅展示前 20 条） ---
    from .redact import redact_record
    potential = result.get('potential_patients') or {}
    if potential:
        lines.append("【接触者风险分层】")
        # 按顺序遍历 family → social，最多展示 20 条
        MAX_CONTACTS = 20
        shown = 0
        for group in ('family', 'social'):
            contacts = potential.get(group) or []
            if not contacts:
                continue
            total_in_group = len(contacts)
            # 当前组应展示的数量
            n_show = min(len(contacts), MAX_CONTACTS - shown)
            if n_show <= 0:
                # 已展示 20 条，仅输出该组数量统计
                lines.append(f"  {group} 组 ({total_in_group} 人): [已超出展示上限，省略]")
                continue
            shown_in_group = 0
            lines.append(f"  {group} 组 ({total_in_group} 人):")
            for c in contacts:
                if shown >= MAX_CONTACTS:
                    break
                # 脱敏单条接触者记录（云端模式避免姓名泄露给 LLM）
                # 本地模式（skip_redact=True）数据不出本机，跳过脱敏保留原始信息
                safe_c = redact_record(c, skip=skip_redact)
                name = safe_c.get('name', '未知')
                risk = safe_c.get('risk_score', 0.0)
                priority = safe_c.get('priority', '未知')
                inf_prob = safe_c.get('infection_probability', 0.0)
                lines.append(
                    f"    - {name}: 风险评分={risk}, 优先级={priority}, "
                    f"感染概率={inf_prob}%"
                )
                shown += 1
                shown_in_group += 1
            # 该组剩余被省略的数量
            omitted = total_in_group - shown_in_group
            if omitted > 0:
                lines.append(f"    ... (省略 {omitted} 条)")
            if shown >= MAX_CONTACTS:
                break
        # 总接触者统计
        total = sum(len(potential.get(g) or []) for g in ('family', 'social'))
        high_n = sum(
            1 for g in ('family', 'social')
            for c in (potential.get(g) or [])
            if c.get('priority') == '高'
        )
        lines.append(f"  总计: {total} 名接触者, 其中高风险 {high_n} 名"
                     f" (上下文展示前 {shown} 条)")

    # --- 4. ML 预测结果（top_features 仅保留前 5 个） ---
    ml = result.get('ml_results')
    if ml:
        lines.append("【ML 机器学习预测】")
        for k, v in ml.items():
            if k == 'top_features':
                feats = v or []
                # 仅保留前 5 个
                top5 = [f for f in feats if isinstance(f, dict)][:5]
                feat_str = ', '.join(
                    f"{f.get('feature', '?')}={f.get('importance', 0):.3f}"
                    for f in top5
                )
                omitted = len(feats) - len(top5)
                suffix = f" (共 {len(feats)} 个，省略 {omitted} 个)" if omitted > 0 else ""
                lines.append(f"  - top_features: {feat_str}{suffix}")
            else:
                lines.append(f"  - {k}: {v}")

    # --- 5. SEIR 动力学结果（仅保留关键标量，排除时间序列/嵌套结构） ---
    seir = result.get('seir_results')
    if seir:
        lines.append("【SEIR 动力学模型】")
        for k, v in seir.items():
            # 跳过时间序列（list）与嵌套 dict（避免上下文膨胀）
            if isinstance(v, (list, dict)):
                continue
            # 跳过过长的字符串值（> 200 字符）
            if isinstance(v, str) and len(v) > 200:
                continue
            lines.append(f"  - {k}: {v}")

    return '\n'.join(lines)


# ==============================================================================
# 主入口
# ==============================================================================

def generate_report(
        result: Dict[str, Any],
        client: Optional[Any] = None,
        config: Optional[AIConfig] = None,
        ) -> Optional[str]:
    """生成中文临床风险评估报告。

    Args:
        result: RiskAssessmentService.assess() 返回的 dict
        client: 可选 LLMClient/LocalLLMClient（测试注入用），None 时按 config 创建
        config: 可选 AIConfig，None 时从环境变量加载

    Returns:
        str: LLM 生成的报告文本。
        未配置/失败/result 无效时返回 None。
    """
    # 空 result 短路
    if not isinstance(result, dict) or not result:
        return None
    if result.get('patient_score') is None:
        return None

    # 获取 client
    if client is None:
        client = make_client(config)
    if client is None:
        # 未配置 → 调用方降级
        return None

    # 判断是否跳过脱敏（本地模式数据不出本机，可跳过脱敏保留原始信息）
    # 优先用显式传入的 config；config 为 None 时从 client.config 获取（make_client 创建的客户端自带 config）
    effective_config = config if config is not None else getattr(client, 'config', None)
    skip_redact = bool(effective_config and effective_config.is_local)

    # 缺陷 #15：优先使用 config.prompts_dir 中的外置提示词，回退到默认常量
    prompts_dir = getattr(effective_config, 'prompts_dir', None) if effective_config else None
    report_sys = get_prompt('REPORT_SYSTEM_PROMPT', prompts_dir)
    report_user_tpl = get_prompt('REPORT_USER_TEMPLATE', prompts_dir)

    # 构造上下文
    context = _build_context(result, skip_redact=skip_redact)
    if not context:
        return None

    # 构造消息
    messages = [
        {"role": "system", "content": report_sys},
        {"role": "user", "content": report_user_tpl.format(context=context)},
    ]

    # 调用 LLM
    # 显式传 max_tokens=2048：风险报告需 500-1000 字（约 700-1400 token），
    # LocalLLMClient 默认 max_new_tokens=2048，但云端 LLMClient 默认不传 max_tokens
    # 可能被服务商截断到 512/1024。统一传 2048 保证生成长度。
    try:
        response = client.chat(messages, max_tokens=2048)
    except Exception as e:
        LOGGER.warning("LLM 报告生成失败: %s", e)
        return None

    if not response or not isinstance(response, str):
        return None

    return response


__all__ = [
    'generate_report',
    '_build_context',
]
