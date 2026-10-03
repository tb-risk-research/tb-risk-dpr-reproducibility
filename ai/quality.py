#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 数据质量诊断模块。

Layer 5c: 对 data_io/scoring.py 标记为低质量的记录，调用 LLM 给出修正建议。

输出结构：
    {
        'issues': [
            {'field': str, 'problem': str, 'suggestion': str, 'confidence': float}
        ],
        'overall_quality': str,  # '低' / '中' / '高' / '未知'
        'summary': str,
    }

调用流程：
1. diagnose_record(record) → 单条记录诊断
2. batch_diagnose(records) → 批量诊断，返回 list 与输入对齐
3. 失败时返回 None（单条）或在列表中对应位置返回 None（批量）
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .client import _validate_schema, make_client
from .config import AIConfig
from .prompts import get_prompt

LOGGER = logging.getLogger("tb_risk.ai.quality")


# ==============================================================================
# 默认值
# ==============================================================================

_DEFAULT_QUALITY = '未知'
_DEFAULT_SUMMARY = ''


# ==============================================================================
# 缺陷 #2（本轮）：嵌套 schema 校验
# ==============================================================================

# 严格 schema：用于 _validate_schema 递归校验 LLM 返回结构
# - issues: list[dict]，每个 issue 必须含 field/problem/suggestion/confidence
# - overall_quality: str
# - summary: str
# confidence 接受 int 或 float（LLM 可能返回 1 而非 1.0）
EXPECTED_SCHEMA: Dict[str, Any] = {
    'issues': [{
        'field': str,
        'problem': str,
        'suggestion': str,
        'confidence': (int, float),
    }],
    'overall_quality': str,
    'summary': str,
}


# ==============================================================================
# 单条诊断
# ==============================================================================

def _normalize_response(response: Any) -> Optional[Dict[str, Any]]:
    """把 LLM 返回 normalize 为标准诊断结果结构。

    Returns:
        dict 含 issues/overall_quality/summary；非 dict 输入返回 None。
    """
    if not isinstance(response, dict):
        return None

    result: Dict[str, Any] = {
        'issues': [],
        'overall_quality': _DEFAULT_QUALITY,
        'summary': _DEFAULT_SUMMARY,
    }

    issues = response.get('issues')
    if isinstance(issues, list):
        # 仅保留 dict 形态的 issue
        result['issues'] = [i for i in issues if isinstance(i, dict)]

    overall = response.get('overall_quality')
    if isinstance(overall, str) and overall:
        result['overall_quality'] = overall

    summary = response.get('summary')
    if isinstance(summary, str):
        result['summary'] = summary

    return result


def _format_record(record: Dict[str, Any]) -> str:
    """把记录 dict 序列化为 LLM 可读的纯文本（JSON 形式）。"""
    try:
        return json.dumps(record, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(record)


def diagnose_record(
        record: Dict[str, Any],
        client: Optional[Any] = None,
        config: Optional[AIConfig] = None,
        ) -> Optional[Dict[str, Any]]:
    """对单条记录进行数据质量诊断。

    Args:
        record: 患者/接触者记录 dict
        client: 可选 LLMClient/LocalLLMClient（测试注入用），None 时按 config 创建
        config: 可选 AIConfig，None 时从环境变量加载

    Returns:
        dict: 含 issues/overall_quality/summary 的诊断结果。
        未配置/失败/空 record 时返回 None。
    """
    # 空 record 短路
    if not isinstance(record, dict) or not record:
        return None

    # 获取 client
    if client is None:
        client = make_client(config)
    if client is None:
        # 未配置 → 调用方降级
        return None

    # 判断是否跳过脱敏（本地模式数据不出本机，可跳过脱敏保留原始信息）
    # 优先用显式传入的 config；config 为 None 时从 client.config 获取
    effective_config = config if config is not None else getattr(client, 'config', None)
    skip_redact = bool(effective_config and effective_config.is_local)

    # 缺陷 #15：优先使用 config.prompts_dir 中的外置提示词，回退到默认常量
    prompts_dir = getattr(effective_config, 'prompts_dir', None) if effective_config else None
    dq_sys = get_prompt('DATA_QUALITY_SYSTEM_PROMPT', prompts_dir)
    dq_user_tpl = get_prompt('DATA_QUALITY_USER_TEMPLATE', prompts_dir)

    # 构造消息（脱敏后送给 LLM；本地模式 skip_redact=True 跳过脱敏）
    from .redact import redact_record
    safe_record = redact_record(record, skip=skip_redact)
    record_str = _format_record(safe_record)
    messages = [
        {"role": "system", "content": dq_sys},
        {"role": "user", "content": dq_user_tpl.format(record=record_str)},
    ]

    # 调用 LLM
    try:
        response = client.chat_json(messages, schema=None)
    except Exception as e:
        LOGGER.warning("LLM 数据质量诊断失败: %s", e)
        return None

    if response is None:
        return None

    # 缺陷 #2（本轮）：用嵌套 schema 严格校验 LLM 返回结构
    # 校验失败时降级到 _normalize_response 浅过滤（仍返回结果，不丢失数据）
    if not _validate_schema(response, EXPECTED_SCHEMA):
        LOGGER.warning(
            "LLM 数据质量诊断返回未通过嵌套 schema 校验，降级到浅过滤: keys=%s",
            list(response.keys()) if isinstance(response, dict) else type(response).__name__,
        )

    return _normalize_response(response)


# ==============================================================================
# 批量诊断
# ==============================================================================

def batch_diagnose(
        records: List[Dict[str, Any]],
        client: Optional[Any] = None,
        config: Optional[AIConfig] = None,
        ) -> List[Optional[Dict[str, Any]]]:
    """批量诊断多条记录。

    Args:
        records: 记录列表
        client: 可选 LLMClient/LocalLLMClient
        config: 可选 AIConfig

    Returns:
        list: 与 records 对齐的诊断结果列表，空记录或失败位置为 None。
    """
    if not records:
        return []

    results: List[Optional[Dict[str, Any]]] = []
    for record in records:
        if not isinstance(record, dict) or not record:
            results.append(None)
            continue
        results.append(diagnose_record(record, client=client, config=config))
    return results


__all__ = [
    'diagnose_record',
    'batch_diagnose',
    '_normalize_response',
    '_format_record',
    'EXPECTED_SCHEMA',
]
