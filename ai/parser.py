#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 文本抽取模块：LLM 文本→结构化数据抽取。

Layer 4: 填充 data_io.text_parser.extract_fields_with_llm 占位接口。

流程：
1. 构造 system + user 提示词（要求 LLM 输出 {"records": [...]} JSON）
2. 调用 LLMClient.chat_json() 获取结构化结果
3. Normalize 每条记录：
   - 仅保留 KNOWN_FIELDS 中的字段
   - 类型转换（int/float/str）
   - 添加 _confidence + _source 元数据
4. 失败时返回 None（调用方降级到正则抽取）

KNOWN_FIELDS 与 data_io/regex_templates.json 字段对齐，
确保 LLM 抽取结果与正则抽取结果可通过 _merge_field_values 融合。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from .client import make_client
from .config import AIConfig
from .prompts import get_prompt

LOGGER = logging.getLogger("tb_risk.ai.parser")

# LLM 抽取置信度（与 _merge_field_values 中 llm_conf 默认值一致）
_LLM_CONFIDENCE = 0.65


# ==============================================================================
# 已知字段表（与 data_io/regex_templates.json 对齐）
# ==============================================================================

KNOWN_FIELDS: Dict[str, type] = {
    # 临床字段
    'age': int,
    'gender': str,
    'sputum_smear': int,        # 1=阳性, 0=阴性
    'has_cavity': int,          # 1=有, 0=无
    'bcg_vaccine': int,         # 1=已接种, 0=未接种
    'treatment': int,           # 1=已治疗, 0=未治疗
    'treatment_duration': int,  # 月
    'has_tb': int,              # 1=既往结核, 0=无
    'cough_freq': int,          # 次/天
    'delay_days': int,          # 延迟就诊天数
    'bmi': float,
    'weight': float,            # kg
    'ethnicity': str,
    # 暴露/环境字段
    'ventilation': str,
    'contact_distance': str,
    'exposure_setting': str,
    'family_living_conditions': str,
}


# ==============================================================================
# 工具函数
# ==============================================================================

def _coerce_value(value: Any, expected_type: type) -> Optional[Any]:
    """类型转换，失败返回 None"""
    if value is None:
        return None
    try:
        if expected_type is int:
            # 处理 "35.0" 这种 float 字符串
            if isinstance(value, str) and '.' in value:
                return int(float(value))
            return int(value)
        if expected_type is float:
            return float(value)
        if expected_type is str:
            return str(value)
    except (ValueError, TypeError):
        return None
    return None


def _normalize_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """将 LLM 返回的单条记录 normalize 为标准形状

    - 仅保留 KNOWN_FIELDS 中的字段
    - 类型转换
    - 添加 _confidence + _source 元数据

    Returns:
        normalize 后的 dict 或 None（无已知字段时）
    """
    if not isinstance(raw, dict):
        return None

    normalized: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}

    for field, expected_type in KNOWN_FIELDS.items():
        if field not in raw:
            continue
        value = raw[field]
        if value is None:
            continue
        coerced = _coerce_value(value, expected_type)
        if coerced is None:
            LOGGER.debug("字段 %s 类型转换失败 (value=%r, expected=%s)",
                         field, value, expected_type.__name__)
            continue
        normalized[field] = coerced
        confidence[field] = _LLM_CONFIDENCE

    if not normalized:
        return None

    normalized['_confidence'] = confidence
    normalized['_source'] = {'source': 'llm'}
    return normalized


def _extract_records_list(response: Any) -> List[Dict[str, Any]]:
    """从 LLM 响应中提取 records 列表

    支持三种格式：
    - {"records": [{...}, ...]}  标准格式
    - [{...}, ...]               直接返回数组
    - {...}                      单 dict，包装为 [dict]
    """
    if response is None:
        return []
    if isinstance(response, list):
        return [r for r in response if isinstance(r, dict)]
    if isinstance(response, dict):
        records = response.get('records')
        if isinstance(records, list):
            return [r for r in records if isinstance(r, dict)]
        # 单 dict 包装
        return [response]
    return []


# ==============================================================================
# 主入口
# ==============================================================================

def extract_with_llm(
        text: str,
        client: Optional[Any] = None,
        config: Optional[AIConfig] = None,
        ) -> Optional[List[Dict[str, Any]]]:
    """LLM 文本抽取

    Args:
        text: 输入文本（临床叙述）
        client: 可选 LLMClient/LocalLLMClient（测试注入用），None 时按 config 创建
        config: 可选 AIConfig，None 时从环境变量加载

    Returns:
        list[dict]: 抽取的记录列表，每条含字段值 + _confidence + _source。
        未配置/失败/无有效记录时返回 None（调用方降级到正则）。
    """
    # 空文本短路
    if not text or not isinstance(text, str):
        return None

    # 获取 client
    if client is None:
        client = make_client(config)
    if client is None:
        # 未配置 → 调用方降级到正则
        return None

    # 缺陷 #15：优先使用 config.prompts_dir 中的外置提示词，回退到默认常量
    effective_config = config if config is not None else getattr(client, 'config', None)
    prompts_dir = getattr(effective_config, 'prompts_dir', None) if effective_config else None
    extraction_sys = get_prompt('EXTRACTION_SYSTEM_PROMPT', prompts_dir)
    extraction_user_tpl = get_prompt('EXTRACTION_USER_TEMPLATE', prompts_dir)

    # 缺陷 #1（本轮）：本地模式数据不出本机时跳过脱敏，保留姓名等关键字段
    # 与 reporter.generate_report / qa.ask 保持一致的 skip_redact 模式
    skip_redact = bool(effective_config and effective_config.is_local)
    # 脱敏后送给 LLM（云端模式避免泄露姓名/身份证/手机号/邮箱）
    from .redact import redact_text
    safe_text = redact_text(text, skip=skip_redact)

    # 构造消息
    messages = [
        {"role": "system", "content": extraction_sys},
        {"role": "user", "content": extraction_user_tpl.format(text=safe_text)},
    ]

    # 调用 LLM
    try:
        response = client.chat_json(messages, schema=None)
    except Exception as e:
        LOGGER.warning("LLM 抽取调用失败: %s", e)
        return None

    if response is None:
        return None

    # 提取 records 列表
    raw_records = _extract_records_list(response)
    if not raw_records:
        LOGGER.debug("LLM 返回无有效 records: %r", response)
        return None

    # Normalize 每条记录
    normalized: List[Dict[str, Any]] = []
    for raw in raw_records:
        rec = _normalize_record(raw)
        if rec is not None:
            normalized.append(rec)

    if not normalized:
        return None

    return normalized


__all__ = [
    'extract_with_llm',
    'KNOWN_FIELDS',
    '_normalize_record',
    '_extract_records_list',
    '_coerce_value',
]
