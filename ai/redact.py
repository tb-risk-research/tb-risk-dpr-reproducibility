#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 脱敏模块：发送给云端 LLM 前对文本/记录做脱敏。

Layer 7: 避免泄露患者身份信息（姓名 / 身份证 / 手机号 / 邮箱）。

脱敏标签：
- [REDACTED_NAME]   替换姓名
- [REDACTED_ID]     替换身份证号
- [REDACTED_PHONE]  替换手机号
- [REDACTED_EMAIL]  替换邮箱

本地部署模式（LocalLLMClient）数据不出本机，可跳过脱敏（skip=True）；
云端模式默认执行脱敏以保持最小权限原则。

正则边界：
- 身份证/手机号用零宽断言 (?<!\\d)...(?!\\d) 替代 \\b，
  在中文上下文（无空格分隔）下也能正确匹配
  （Python re 模块的 \\b 对中文字符的边界判定不一致）
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# ==============================================================================
# PII 正则
# ==============================================================================

# 18 位身份证号（粗匹配：前 17 位数字 + 末位数字或 X）
# 用零宽断言 (?<!\d)...(?!\d) 替代 \b，确保前后非数字
# 避免 19 位银行卡号被部分匹配，也避免中文紧贴时漏匹配
_ID_CARD_PATTERN = re.compile(r'(?<!\d)\d{17}[\dXx](?!\d)')

# 11 位手机号（粗匹配：1 开头 + 10 位数字）
# 同样用零宽断言，避免 10/12 位数字误匹配或漏匹配
_PHONE_PATTERN = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')

# 邮箱（前后用零宽断言替代 \b，避免中文紧贴时边界失效）
# 邮箱本地部分允许 A-Za-z0-9._%+- ，域名允许 A-Za-z0-9.- ，TLD 至少 2 字母
_EMAIL_PATTERN = re.compile(
    r'(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z])'
)

# 中文姓名：'姓名：张三' / '姓名:张三' / '患者张三' / '患者：张三'
# 仅脱敏 2-4 字中文姓名，避免误伤
_NAME_AFTER_LABEL_PATTERN = re.compile(
    r'(姓名|患者)\s*[:：]?\s*([\u4e00-\u9fff]{2,4})'
)


def _redact_name(match: re.Match) -> str:
    """保留标签，仅替换姓名部分"""
    return f"{match.group(1)}[REDACTED_NAME]"


# ==============================================================================
# 文本脱敏
# ==============================================================================

def redact_text(text: Optional[str], skip: bool = False) -> str:
    """对文本做脱敏处理。

    Args:
        text: 输入文本
        skip: 是否跳过脱敏（本地模式数据不出本机时可传 True）

    Returns:
        str: 脱敏后的文本。空输入返回空字符串。skip=True 时原样返回。
    """
    if not text or not isinstance(text, str):
        return ''
    if skip:
        return text

    # 顺序：先脱敏身份证（最长），再手机号，再邮箱，最后姓名
    # 避免 shorter patterns 截断 longer patterns
    result = _ID_CARD_PATTERN.sub('[REDACTED_ID]', text)
    result = _PHONE_PATTERN.sub('[REDACTED_PHONE]', result)
    result = _EMAIL_PATTERN.sub('[REDACTED_EMAIL]', result)
    result = _NAME_AFTER_LABEL_PATTERN.sub(_redact_name, result)

    return result


# ==============================================================================
# 记录脱敏
# ==============================================================================

# 字段名 → 脱敏标签映射
# 完全相等的字段名匹配 → 整体替换为标签
_PII_FIELD_EXACT_MAP = {
    'name': '[REDACTED_NAME]',
    'patient_name': '[REDACTED_NAME]',
    'id_card': '[REDACTED_ID]',
    'id_number': '[REDACTED_ID]',
    'identity': '[REDACTED_ID]',
    'phone': '[REDACTED_PHONE]',
    'mobile': '[REDACTED_PHONE]',
    'tel': '[REDACTED_PHONE]',
    'telephone': '[REDACTED_PHONE]',
    'email': '[REDACTED_EMAIL]',
    'mail': '[REDACTED_EMAIL]',
}

# 允许下划线前缀匹配的 PII 关键词（即 field 形如 `xxx_keyword`）
# mail/tel/name 等歧义词不放入此列表（仅完全相等匹配），
# 避免 mailbox / tel_index / contact_mail 误脱敏
_PII_FIELD_SUFFIX_KEYWORDS = {
    'name': '[REDACTED_NAME]',
    'patient_name': '[REDACTED_NAME]',
    'id_card': '[REDACTED_ID]',
    'id_number': '[REDACTED_ID]',
    'identity': '[REDACTED_ID]',
    'phone': '[REDACTED_PHONE]',
    'mobile': '[REDACTED_PHONE]',
    'telephone': '[REDACTED_PHONE]',
    'email': '[REDACTED_EMAIL]',
    # 注意：mail/tel 不在 suffix 列表（歧义词仅完全相等匹配）
}


def _redact_field_value(field: str, value: Any, skip: bool = False) -> Any:
    """对单字段值做脱敏

    - PII 字段（name/phone/id_card 等）整体替换为标签
    - 字符串值中嵌入的 PII 用 redact_text 处理
    - 非字符串/非 PII 字段保持原样
    - skip=True 时原样返回（本地模式）
    """
    if value is None:
        return None
    if skip:
        return value

    # 字段名匹配 PII 关键词 → 整体替换
    field_lower = field.lower()
    # 1. 完全相等匹配
    if field_lower in _PII_FIELD_EXACT_MAP:
        return _PII_FIELD_EXACT_MAP[field_lower]
    # 2. 下划线前缀匹配（field 形如 xxx_keyword，keyword 在末尾）
    #    仅对非歧义词允许（mail/tel 不在 suffix 列表）
    for kw, tag in _PII_FIELD_SUFFIX_KEYWORDS.items():
        if field_lower.endswith('_' + kw):
            return tag

    # 字符串值中嵌入的 PII → redact_text
    if isinstance(value, str):
        return redact_text(value, skip=skip)

    # 其他类型（int/float/bool/list/dict）保持原样
    return value


def redact_record(record: Optional[Dict[str, Any]],
                  skip: bool = False) -> Optional[Dict[str, Any]]:
    """对记录 dict 做脱敏处理（返回新 dict，不修改输入）。

    Args:
        record: 患者/接触者记录 dict
        skip: 是否跳过脱敏（本地模式数据不出本机时可传 True）

    Returns:
        dict: 脱敏后的新 dict。None 输入返回 None。skip=True 时返回浅拷贝。
    """
    if record is None:
        return None
    if not isinstance(record, dict):
        return record
    if skip:
        return dict(record)  # 浅拷贝，保持返回新 dict 的契约

    return {k: _redact_field_value(k, v, skip=skip) for k, v in record.items()}


__all__ = [
    'redact_text',
    'redact_record',
]
