#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk.ai — AI/LLM 接入子包

Layer 1: 子包结构入口。

子模块：
- config:    配置（环境变量、提供商预设、用户配置文件）
- client:    LLM 客户端封装（OpenAI 兼容协议 + 本地 transformers 模型）
- prompts:   提示词模板
- parser:    文本→结构化数据抽取（填充 data_io.text_parser.extract_fields_with_llm）
- reporter:  风险报告生成
- qa:        智能问答（RAG）
- quality:   数据质量诊断
- redact:    PII 脱敏

设计原则：
1. 不引入 pydantic（与 schemas.py 一致，使用 stdlib dataclasses）。
2. 所有云端调用走 OpenAI 兼容协议（DeepSeek/通义/智谱/Kimi 均兼容）。
3. API Key 不硬编码、不写入版本控制，从环境变量或 ~/.tb_risk/ai_config.json 读取。
4. 失败降级到原正则抽取逻辑，保证核心功能可用。

便捷导出：
    from tb_risk.ai import AIConfig, make_client
    from tb_risk.ai import extract_with_llm, generate_report, ask, diagnose_record
"""

from .config import AIConfig, PROVIDERS, DEFAULT_AI_CONFIG_FILE
from .client import LLMClient, LocalLLMClient, make_client
from .parser import extract_with_llm
from .reporter import generate_report
from .qa import ask
from .quality import diagnose_record, batch_diagnose

__all__ = [
    # 配置
    'AIConfig',
    'PROVIDERS',
    'DEFAULT_AI_CONFIG_FILE',
    # 客户端
    'LLMClient',
    'LocalLLMClient',
    'make_client',
    # 业务入口
    'extract_with_llm',
    'generate_report',
    'ask',
    'diagnose_record',
    'batch_diagnose',
]
