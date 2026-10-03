#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 智能问答模块（RAG 模式）。

Layer 5b: 把当前评估结果作为上下文 + 可选知识库片段，
回答用户关于结核病风险的问题。

RAG 策略（缺陷 #16 改进）：
- knowledge_base: list[{'content': str, 'keywords': list[str]}]
- retrieve_knowledge(query, kb):
  * jieba 中文分词（可选依赖，不可用时回退到字符切分）
  * BM25 加权排序（IDF 归一化 + 文档长度归一化）
  * keywords 字段作为额外加分（保留向后兼容）
  * 适合 kb 条目数 < 100 的场景
  * 未来可扩展为向量检索（如 sentence-transformers + faiss）

调用流程：
1. ask(question, context, knowledge=None) → LLM 回答
2. context 由 _build_context_for_qa 把 result dict 转为 LLM 可读文本
3. knowledge 可直接传入字符串，或通过 retrieve_knowledge 从 kb 检索
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from .client import make_client
from .config import AIConfig
from .prompts import get_prompt

LOGGER = logging.getLogger("tb_risk.ai.qa")

# 可选依赖：jieba 中文分词
try:
    import jieba
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False
    jieba = None  # type: ignore


# ==============================================================================
# 分词工具
# ==============================================================================

# 中文字符正则（用于无 jieba 时的字符切分回退）
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')
# 英文/数字 token 正则（连续的 ASCII 字母数字）
_LATIN_RE = re.compile(r'[A-Za-z0-9]+')
# 标点符号（剥离）— 用 Unicode 转义避免中文引号嵌入 Python 字符串字面量
_PUNCT_RE = re.compile(
    r'[\u3000-\u303f\uff00-\uffef\s,\.!?;:()<>]'  # CJK 符号 + 全角 ASCII + 空白 + 半角标点
    r'|[\u2018\u2019\u201c\u201d]'  # 中文单/双引号
)


def _tokenize(text: str) -> List[str]:
    """中文分词（jieba 优先，回退到字符切分）

    - jieba 可用：jieba.lcut(text, cut_all=False) 精确模式
    - jieba 不可用：中文字符逐字 + 英文/数字连续串
    - 标点符号被剥离

    Args:
        text: 待分词文本

    Returns:
        list[str]: token 列表（已去重空白与标点）
    """
    if not text:
        return []
    # 剥离标点（替换为空格，避免中文字符粘连）
    cleaned = _PUNCT_RE.sub(' ', text)
    if _JIEBA_AVAILABLE:
        try:
            tokens = jieba.lcut(cleaned, cut_all=False)
        except Exception:
            # jieba 异常时回退
            tokens = _fallback_tokenize(cleaned)
    else:
        tokens = _fallback_tokenize(cleaned)
    # 过滤空白 token
    return [t.strip() for t in tokens if t and t.strip()]


def _fallback_tokenize(text: str) -> List[str]:
    """无 jieba 时的回退分词：中文字符逐字 + 英文数字连续串

    Args:
        text: 已剥离标点的文本

    Returns:
        list[str]: token 列表
    """
    tokens: List[str] = []
    # 提取英文/数字连续串
    tokens.extend(_LATIN_RE.findall(text))
    # 提取中文字符（逐字）
    tokens.extend(_CJK_RE.findall(text))
    return tokens


# ==============================================================================
# BM25 检索
# ==============================================================================

# BM25 参数
_BM25_K1 = 1.5  # 词频饱和参数
_BM25_B = 0.75  # 文档长度归一化参数
# keywords 命中加分权重（每个 keyword 命中加这么多分）
_KEYWORD_BOOST = 5.0


def _compute_bm25(
        query_tokens: List[str],
        doc_tokens_list: List[List[str]],
        ) -> List[float]:
    """计算 query 与每个 doc 的 BM25 分数

    BM25 公式：
        score(q, d) = sum_{t in q} IDF(t) * (f(t,d) * (k1 + 1)) /
                                       (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
        IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)

    Args:
        query_tokens: 查询的 token 列表
        doc_tokens_list: 每个文档的 token 列表

    Returns:
        list[float]: 每个文档的 BM25 分数（非负）
    """
    n_docs = len(doc_tokens_list)
    if n_docs == 0:
        return []

    # 文档长度与平均长度
    doc_lens = [len(dt) for dt in doc_tokens_list]
    avgdl = sum(doc_lens) / n_docs if n_docs > 0 else 0.0
    if avgdl == 0:
        avgdl = 1.0  # 避免除零

    # 计算每个 token 的 IDF
    # n(t) = 包含 token t 的文档数
    token_doc_count: Dict[str, int] = {}
    for dt in doc_tokens_list:
        # 该文档中出现的唯一 token 集合
        unique_tokens = set(dt)
        for t in unique_tokens:
            token_doc_count[t] = token_doc_count.get(t, 0) + 1

    idf: Dict[str, float] = {}
    for t, n_t in token_doc_count.items():
        idf[t] = math.log((n_docs - n_t + 0.5) / (n_t + 0.5) + 1.0)

    # 计算每个文档的 BM25 分数
    scores: List[float] = []
    for dt, dl in zip(doc_tokens_list, doc_lens):
        # 计算 doc 中每个 token 的词频
        tf: Dict[str, int] = {}
        for t in dt:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        # query_tokens 去重，避免同一 token 重复加权
        for t in set(query_tokens):
            f = tf.get(t, 0)
            idf_t = idf.get(t, 0.0)
            if f == 0:
                # 子串匹配回退：处理中文复合词。
                # jieba 精确模式会把 "结核病" 等合并为单个 token，导致 query
                # "结核" 无法精确匹配。当 query 词是某文档词的子串时，按该
                # 文档词匹配计数，并用其 idf 作为近似，保证检索召回相关文档。
                for doc_tok, freq in tf.items():
                    if t in doc_tok:
                        f += freq
                        idf_t = max(idf_t, idf.get(doc_tok, 0.0))
                if f == 0:
                    continue
            # BM25 TF 饱和项
            numerator = f * (_BM25_K1 + 1)
            denominator = f + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl)
            score += idf_t * (numerator / denominator if denominator > 0 else 0)
        scores.append(max(0.0, score))  # 确保非负
    return scores


def retrieve_knowledge(
        query: str,
        knowledge_base: Optional[List[Dict[str, Any]]],
        top_k: int = 3,
        ) -> str:
    """从知识库中检索与 query 相关的片段（BM25 + keywords 加权）

    打分策略：
    1. BM25 分数：基于 query 与 entry.content 的 token 匹配，
       含 IDF 归一化与文档长度归一化
    2. keywords 加分：entry.keywords 中每个在 query 中出现的 keyword
       加 _KEYWORD_BOOST 分
    总分 = BM25 + keywords 加分；分数 > 0 的条目按降序排序，取 Top-K

    Args:
        query: 用户查询文本
        knowledge_base: list[{'content': str, 'keywords': list[str]}]
        top_k: 返回最多 K 条相关片段

    Returns:
        str: Top-K 条 content 拼接（按总分降序），无匹配时返回空字符串
    """
    if not query or not knowledge_base:
        return ''

    # 1. 预处理：分词 + 跳过无 content 的条目
    query_tokens = _tokenize(query)
    entries: List[Tuple[str, List[str], List[str]]] = []  # (content, doc_tokens, keywords)
    for entry in knowledge_base:
        content = entry.get('content', '')
        if not content:
            continue
        doc_tokens = _tokenize(content)
        keywords = entry.get('keywords') or []
        entries.append((content, doc_tokens, keywords))

    if not entries:
        return ''

    # 2. 计算 BM25 分数
    doc_tokens_list = [dt for _, dt, _ in entries]
    bm25_scores = _compute_bm25(query_tokens, doc_tokens_list)

    # 3. 计算 keywords 加分 + 总分
    # 把 query 转为字符串集合便于检查 keyword 是否出现
    query_lower = query.lower()
    scored: List[Tuple[float, str]] = []
    for (content, _, keywords), bm25_score in zip(entries, bm25_scores):
        keyword_boost = 0.0
        for kw in keywords:
            if not kw:
                continue
            # keyword 匹配：query 包含 keyword 子串
            # 大小写不敏感
            if kw.lower() in query_lower:
                keyword_boost += _KEYWORD_BOOST
            # keyword 在 query_tokens 中（精确 token 匹配）
            elif kw in query_tokens:
                keyword_boost += _KEYWORD_BOOST
        total = bm25_score + keyword_boost
        if total > 0:
            scored.append((total, content))

    if not scored:
        return ''

    # 4. 按总分降序，取 Top-K
    scored.sort(key=lambda x: x[0], reverse=True)
    top_contents = [c for _, c in scored[:top_k]]
    return '\n'.join(top_contents)


# ==============================================================================
# 上下文构造
# ==============================================================================

def _build_context_for_qa(context: Optional[Dict[str, Any]],
                         skip_redact: bool = False) -> str:
    """把评估结果 dict 组装成 LLM 可读的纯文本上下文（QA 场景）。

    Args:
        context: 评估结果 dict
        skip_redact: 是否跳过脱敏（本地模式数据不出本机时传 True）

    Returns:
        str: 组装后的上下文文本。空 context 返回空字符串。
    """
    if not isinstance(context, dict) or not context:
        return ''

    from .redact import redact_record

    lines: List[str] = []

    patient_score = context.get('patient_score')
    if patient_score is not None:
        lines.append(f"【患者风险评分】{patient_score}")

    summary = context.get('summary') or {}
    if summary:
        lines.append("【评估汇总】")
        for k, v in summary.items():
            lines.append(f"  - {k}: {v}")

    potential = context.get('potential_patients') or {}
    if potential:
        lines.append("【接触者风险分层】")
        for group in ('family', 'social'):
            contacts = potential.get(group) or []
            if not contacts:
                continue
            lines.append(f"  {group} 组 ({len(contacts)} 人):")
            for c in contacts:
                # 脱敏后送给 LLM（本地模式 skip_redact=True 跳过脱敏）
                safe_c = redact_record(c, skip=skip_redact)
                name = safe_c.get('name', '未知')
                risk = safe_c.get('risk_score', 0.0)
                priority = safe_c.get('priority', '未知')
                lines.append(f"    - {name}: 风险评分={risk}, 优先级={priority}")

    ml = context.get('ml_results')
    if ml:
        lines.append("【ML 预测】")
        for k, v in ml.items():
            lines.append(f"  - {k}: {v}")

    seir = context.get('seir_results')
    if seir:
        lines.append("【SEIR 动力学】")
        for k, v in seir.items():
            lines.append(f"  - {k}: {v}")

    return '\n'.join(lines)


# ==============================================================================
# 主入口
# ==============================================================================

def ask(
        question: str,
        context: Optional[Dict[str, Any]] = None,
        knowledge: Optional[str] = None,
        client: Optional[Any] = None,
        config: Optional[AIConfig] = None,
        ) -> Optional[str]:
    """AI 智能问答主入口。

    Args:
        question: 用户问题
        context: 当前评估结果 dict（可选）
        knowledge: 检索到的知识片段（可选，可直接传入字符串）
        client: 可选 LLMClient/LocalLLMClient（测试注入用），None 时按 config 创建
        config: 可选 AIConfig，None 时从环境变量加载

    Returns:
        str: LLM 回答文本。未配置/失败/空 question 时返回 None。
    """
    # 空 question 短路
    if not question or not isinstance(question, str):
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
    qa_sys = get_prompt('QA_SYSTEM_PROMPT', prompts_dir)
    qa_user_tpl = get_prompt('QA_USER_TEMPLATE', prompts_dir)

    # 构造上下文
    context_str = _build_context_for_qa(context, skip_redact=skip_redact) or "（无上下文）"
    knowledge_str = knowledge if knowledge else "（无参考知识）"

    # 构造消息
    messages = [
        {"role": "system", "content": qa_sys},
        {"role": "user", "content": qa_user_tpl.format(
            context=context_str,
            knowledge=knowledge_str,
            question=question,
        )},
    ]

    # 调用 LLM
    try:
        response = client.chat(messages)
    except Exception as e:
        LOGGER.warning("LLM 问答失败: %s", e)
        return None

    if not response or not isinstance(response, str):
        return None

    return response


__all__ = [
    'ask',
    'retrieve_knowledge',
    '_build_context_for_qa',
]
