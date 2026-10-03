#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 客户端封装：OpenAI 兼容协议 + 本地 transformers 模型。

Layer 3b: 客户端基础设施。

两类客户端：
- LLMClient:       云端 OpenAI 兼容协议（DeepSeek/通义/智谱/Kimi/OpenAI）
- LocalLLMClient:  本地 transformers + torch 模型（敏感场景，数据不出本机）

对外接口（两者一致）：
- chat(messages, **kwargs) -> str:           返回模型回复文本
- chat_json(messages, schema=None) -> dict:  返回 JSON dict，schema 验证失败返回 None
- close() / __enter__ / __exit__:            生命周期管理

工厂函数：
- make_client(config=None): 按 config.is_local 选择 LocalLLMClient 或 LLMClient
                            未配置时返回 None（调用方降级）

重试策略（仅 LLMClient）：
- 429 (Rate Limit):    指数退避重试
- 5xx (Server Error):  指数退避重试
- Timeout/Network:     指数退避重试
- 4xx (非 429):        立即抛出（认证/请求错误不可重试）
- 最大尝试次数 = config.max_retries（默认 3）
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import AIConfig

LOGGER = logging.getLogger("tb_risk.ai.client")

# 可选依赖：httpx + tenacity
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore

try:
    from tenacity import (Retrying, retry_if_exception,
                          stop_after_attempt, wait_exponential)
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False


# ==============================================================================
# Schema 验证（轻量级，不引入 pydantic）
# ==============================================================================

def _validate_schema(data: Any, schema: Dict[str, Any]) -> bool:
    """验证 data 是否符合 schema（dict 字段名→类型/嵌套结构）

    支持的 schema 形式：
    - {field: type}:                          平面字段（向后兼容）
    - {field: (type1, type2)}:                多类型（含 Optional，如 (int, type(None))）
    - {field: {sub_field: type}}:             嵌套 dict（递归验证子结构）
    - {field: [type]}:                        list[type]（每个元素是该类型）
    - {field: [{sub_field: type}]}:           list[dict]（每个元素是符合子 schema 的 dict）
    - {field: (sub_schema, type(None))}:      Optional 嵌套（dict 或 None）

    规则：
    - schema 中每个字段都是 required（必须出现在 data 中）
    - data 必须是 dict

    Args:
        data: 待验证数据
        schema: {field_name: type_or_tuple_or_nested_schema_or_list}

    Returns:
        bool: 验证通过 True，否则 False
    """
    if not isinstance(data, dict):
        return False
    if not isinstance(schema, dict):
        return False
    for field, type_spec in schema.items():
        if field not in data:
            return False
        val = data[field]
        if not _validate_value(val, type_spec):
            return False
    return True


def _validate_value(val: Any, spec: Any) -> bool:
    """对单个值按 spec 验证（递归处理嵌套 dict / list / tuple）"""
    # 1. tuple → 多类型（含 None）或 Optional 嵌套
    #    例如 (int, type(None)) 或 ({'id': int}, type(None))
    if isinstance(spec, tuple):
        for s in spec:
            if _validate_value(val, s):
                return True
        return False

    # 2. dict → 嵌套 schema
    if isinstance(spec, dict):
        if not isinstance(val, dict):
            return False
        return _validate_schema(val, spec)

    # 3. list → 元素 schema
    if isinstance(spec, list):
        if not isinstance(val, list):
            return False
        if not spec:
            # spec 为空 list 表示任意 list（不约束元素）
            return True
        elem_spec = spec[0]
        for elem in val:
            if not _validate_value(elem, elem_spec):
                return False
        return True

    # 4. 单个 type（str/int/float/bool/list/dict/...）
    if isinstance(spec, type):
        return isinstance(val, spec)

    # 5. 其他未知 spec 形式 → 不通过
    return False


# ==============================================================================
# JSON 解析工具
# ==============================================================================

_CODE_FENCE_RE = re.compile(r'^```(?:\w+)?\s*\n(.*?)\n```\s*$', re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """剥离 ```json ... ``` 或 ``` ... ``` 代码块"""
    if not text:
        return text
    text = text.strip()
    m = _CODE_FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


def _parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """解析 LLM 返回的 JSON（剥离代码块后解析）"""
    if not content:
        return None
    stripped = _strip_code_fences(content)
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


# ==============================================================================
# LLMClient（云端 OpenAI 兼容）
# ==============================================================================

class LLMClient:
    """OpenAI 兼容协议的 LLM 客户端

    所有云端提供商（DeepSeek/通义/智谱/Kimi/OpenAI）走同一协议，
    仅 base_url 与 api_key 不同。
    """

    def __init__(self, config: AIConfig,
                 http_client: Optional[Any] = None) -> None:
        """
        Args:
            config: AIConfig 实例
            http_client: 可选 httpx.Client（测试注入 MockTransport 用），
                         None 时懒构造
        """
        self.config = config
        self._http = http_client  # None 时懒构造
        self._closed = False
        # 缺陷 #18：token 用量累计（按客户端实例隔离）
        self._token_usage: Dict[str, int] = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'calls': 0,
        }

    def _get_http(self):
        """懒构造 httpx.Client"""
        if self._http is None:
            if not HTTPX_AVAILABLE:
                raise ImportError("httpx 未安装，请运行: pip install httpx")
            self._http = httpx.Client(timeout=self.config.timeout)
        return self._http

    def _is_retryable(self, exc: BaseException) -> bool:
        """判断异常是否可重试"""
        if not HTTPX_AVAILABLE:
            return False
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            # 429 (Rate Limit) 或 5xx (Server Error) 可重试
            return status == 429 or status >= 500
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
            return True
        return False

    def _make_retryer(self):
        """构造 tenacity Retrying 实例（基于 config.max_retries）

        wait_exponential max=10s：原 0.5s 对云端 API 的 429 限速恢复过短，
        重试 3 次后仍会失败。10s 上限匹配主流 LLM API 的 Retry-After 建议。
        """
        if not TENACITY_AVAILABLE:
            raise ImportError("tenacity 未安装，请运行: pip install tenacity")
        return Retrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=10.0),
            retry=retry_if_exception(self._is_retryable),
            reraise=True,
        )

    def _call_api(self, messages: List[Dict[str, Any]],
                  **kwargs) -> Dict[str, Any]:
        """单次 HTTP 调用（不重试，由 chat() 包装重试）"""
        if not self.config.api_key:
            raise ValueError(
                "LLMClient 未配置 api_key，无法调用云端 API。"
                "请设置 TB_AI_API_KEY 环境变量或在 ~/.tb_risk/ai_config.json 中配置。")

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            'Authorization': f"Bearer {self.config.api_key}",
            'Content-Type': 'application/json',
        }
        body: Dict[str, Any] = {
            'model': self.config.model,
            'messages': messages,
            'temperature': self.config.temperature,
        }
        body.update(kwargs)

        http = self._get_http()
        response = http.post(url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()

    def chat(self, messages: List[Dict[str, Any]],
             **kwargs) -> str:
        """发送聊天请求，返回助手回复文本

        Args:
            messages: OpenAI 格式消息列表
                [{"role": "system"/"user"/"assistant", "content": "..."}]
            **kwargs: 额外请求体参数（temperature/max_tokens/top_p 等）

        Returns:
            str: 助手回复文本

        Raises:
            ValueError: 未配置 api_key
            httpx.HTTPStatusError: 4xx 非 429（不可重试）或重试耗尽后的 5xx/429
            httpx.TimeoutException: 重试耗尽后的超时
        """
        # 缺陷 #8：结构化调用审计日志（JSON 格式），成功与失败均记录
        start_time = time.monotonic()
        status = "error"
        error_info: Optional[Tuple[str, str]] = None
        try:
            if not TENACITY_AVAILABLE:
                # 无 tenacity 时直接调用一次（不重试）
                data = self._call_api(messages, **kwargs)
            else:
                retryer = self._make_retryer()
                data = retryer(self._call_api, messages, **kwargs)
            # 缺陷 #18：累加 token 用量
            self._accumulate_usage(data)
            status = "success"
            return self._extract_content(data)
        except Exception as e:
            error_info = (type(e).__name__, str(e))
            raise
        finally:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            self._log_call_structured(latency_ms, status, error_info)

    def _log_call_structured(self, latency_ms: int, status: str,
                             error_info: Optional[Tuple[str, str]] = None) -> None:
        """记录结构化调用日志（JSON 格式，缺陷 #8）

        每次调用（成功或失败）均记录一条 JSON 结构化日志，包含：
        - event:        固定为 'llm_call'
        - timestamp:    ISO 8601 时间戳
        - provider:     提供商名（deepseek/openai/...）
        - model:        模型名
        - calls:        当前累计调用次数（成功才累加）
        - input_tokens/output_tokens/total_tokens: 累计 token 数
        - latency_ms:   本次调用耗时（毫秒）
        - status:       'success' 或 'error'
        - error_type/error_message: 失败时附加（截断 200 字符）

        便于事后审计与成本核算（本地模式同样记录）。

        Args:
            latency_ms: 本次调用耗时（毫秒）
            status: 'success' 或 'error'
            error_info: 失败时为 (error_type, error_message)，成功时为 None
        """
        log_entry = {
            'event': 'llm_call',
            'timestamp': datetime.datetime.now().isoformat(),
            'provider': self.config.provider,
            'model': self.config.model,
            'calls': self._token_usage['calls'],
            'input_tokens': self._token_usage['prompt_tokens'],
            'output_tokens': self._token_usage['completion_tokens'],
            'total_tokens': self._token_usage['total_tokens'],
            'latency_ms': latency_ms,
            'status': status,
        }
        if error_info is not None:
            log_entry['error_type'] = error_info[0]
            # 截断超长错误信息，避免日志爆炸
            log_entry['error_message'] = error_info[1][:200]
        LOGGER.info("LLM 调用审计: %s",
                    json.dumps(log_entry, ensure_ascii=False))

    def _accumulate_usage(self, data: Dict[str, Any]) -> None:
        """从 OpenAI 响应中提取 usage 字段并累加到 _token_usage（缺陷 #18）

        - 响应缺失 usage 字段时不抛异常，仅累加 calls 计数
        - usage 字段只含部分子字段时，缺失部分按 0 累加

        Args:
            data: OpenAI 兼容响应 JSON dict
        """
        self._token_usage['calls'] += 1
        usage = data.get('usage') if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            return
        prompt = int(usage.get('prompt_tokens', 0) or 0)
        completion = int(usage.get('completion_tokens', 0) or 0)
        total = int(usage.get('total_tokens', 0) or 0)
        self._token_usage['prompt_tokens'] += prompt
        self._token_usage['completion_tokens'] += completion
        self._token_usage['total_tokens'] += total

    def get_usage(self) -> Dict[str, int]:
        """查询当前客户端累计的 token 用量（缺陷 #18）

        Returns:
            dict: 包含以下字段：
                - prompt_tokens (int): 累计输入 token 数
                - completion_tokens (int): 累计输出 token 数
                - total_tokens (int): 累计总 token 数
                - calls (int): 累计成功调用次数
        """
        return dict(self._token_usage)

    def reset_usage(self) -> None:
        """清零 token 用量统计（缺陷 #18）

        适用于长会话中按周期审计成本、或测试隔离。
        """
        self._token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'calls': 0,
        }

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        """从 OpenAI 响应 JSON 提取 content"""
        try:
            return data['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"LLM 响应格式异常，无法提取 content: {e}") from e

    def chat_json(self, messages: List[Dict[str, Any]],
                  schema: Optional[Dict[str, Any]] = None,
                  **kwargs) -> Optional[Dict[str, Any]]:
        """发送聊天请求，解析 JSON 返回 dict

        Args:
            messages: OpenAI 格式消息列表
            schema: 可选 schema dict {field: type}，验证失败返回 None
            **kwargs: 额外请求体参数

        Returns:
            dict: 解析后的 JSON dict，解析失败或 schema 验证失败返回 None
        """
        try:
            content = self.chat(messages, **kwargs)
        except Exception as e:
            LOGGER.warning("chat_json 调用失败: %s", e)
            return None

        parsed = _parse_json_response(content)
        if parsed is None:
            LOGGER.warning("LLM 返回内容无法解析为 JSON: %.200s", content)
            return None

        if schema is not None:
            if not _validate_schema(parsed, schema):
                LOGGER.warning("LLM 返回 JSON 未通过 schema 验证: %s",
                               list(parsed.keys()))
                return None
        return parsed

    def close(self) -> None:
        """关闭底层 HTTP 客户端"""
        if self._http is not None and not self._closed:
            try:
                self._http.close()
            except Exception as e:
                LOGGER.warning("关闭 HTTP 客户端失败: %s", e)
            self._closed = True

    def __enter__(self) -> 'LLMClient':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ==============================================================================
# LocalLLMClient（本地 transformers + torch）
# ==============================================================================

class LocalLLMClient:
    """本地 LLM 客户端（transformers + torch，敏感场景）

    模型懒加载：构造时不加载，首次调用 chat() 时才加载。
    适用于 Qwen2.5-7B-Instruct 等本地部署模型。
    """

    def __init__(self, config: AIConfig) -> None:
        """
        Args:
            config: AIConfig 实例（local_model_path 必须设置）
        """
        if not config.local_model_path:
            raise ValueError("LocalLLMClient 需要 config.local_model_path")
        self.config = config
        self._model = None
        self._tokenizer = None
        # 模型加载锁：避免多线程并发触发 chat() 时重复加载 7B 模型
        self._load_lock = threading.Lock()
        # 缺陷 #18：token 用量累计（按客户端实例隔离）
        self._token_usage: Dict[str, int] = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'calls': 0,
        }

    def _load_model(self) -> None:
        """懒加载模型与 tokenizer（线程安全）

        多线程并发调用时，仅第一个线程真正加载模型，其他线程在锁上等待；
        待锁释放后看到 _model 已就绪直接返回。
        """
        # 快速路径：模型已加载直接返回（无锁）
        if self._model is not None and self._tokenizer is not None:
            return
        # 慢路径：加锁后再次检查（double-checked locking）
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "transformers/torch 未安装，无法使用本地 LLM。"
                    "请运行: pip install torch transformers") from e

            LOGGER.info("加载本地 LLM 模型: %s (device=%s)",
                        self.config.local_model_path, self.config.local_device)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.local_model_path, trust_remote_code=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.local_model_path,
                    torch_dtype=torch.float16 if 'cuda' in self.config.local_device
                    else torch.float32,
                    device_map=self.config.local_device
                    if 'cuda' in self.config.local_device else None,
                    trust_remote_code=True,
                )
                if 'cuda' not in self.config.local_device:
                    self._model = self._model.to(self.config.local_device)
                self._model.eval()
            except Exception as e:
                # 重置状态以便下次重试
                self._model = None
                self._tokenizer = None
                raise RuntimeError(
                    f"加载本地 LLM 模型失败 ({self.config.local_model_path}): {e}") from e

    def chat(self, messages: List[Dict[str, Any]],
             **kwargs) -> str:
        """本地生成回复

        Args:
            messages: OpenAI 格式消息列表
            **kwargs: 额外生成参数（max_new_tokens/temperature/top_p 等）

        Returns:
            str: 模型生成的文本
        """
        self._load_model()
        import torch

        # 缺陷 #8：结构化调用审计日志（与 LLMClient 保持一致）
        start_time = time.monotonic()
        status = "error"
        error_info: Optional[Tuple[str, str]] = None
        try:
            # 用 tokenizer 的 chat template 拼接提示
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                # 无 chat template 的模型回退到简单拼接
                prompt = "\n".join(
                    f"[{m['role']}] {m['content']}" for m in messages)

            inputs = self._tokenizer(prompt, return_tensors="pt")
            if 'cuda' in self.config.local_device:
                inputs = {k: v.to(self.config.local_device) for k, v in inputs.items()}

            gen_kwargs = {
                # 默认 2048：风险报告需 500-1000 字（约 700-1400 token），512 会截断
                'max_new_tokens': kwargs.pop('max_tokens', 2048),
                'temperature': kwargs.pop('temperature', self.config.temperature),
                'top_p': kwargs.pop('top_p', 0.9),
                'do_sample': True,
                'pad_token_id': self._tokenizer.eos_token_id,
            }
            gen_kwargs.update(kwargs)

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)

            # 仅取新生成的 token
            input_len = inputs['input_ids'].shape[1]
            new_tokens = outputs[0][input_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

            # 缺陷 #18：本地模型通过 tokenizer 计算 token 数（无 HTTP usage 字段）
            # prompt_tokens = 输入 token 数；completion_tokens = 输出 token 数
            prompt_count = int(input_len)
            completion_count = int(new_tokens.shape[0]) if hasattr(
                new_tokens, 'shape') else len(new_tokens)
            self._accumulate_local_usage(prompt_count, completion_count)

            status = "success"
            return text
        except Exception as e:
            error_info = (type(e).__name__, str(e))
            raise
        finally:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            self._log_call_structured(latency_ms, status, error_info)

    def _log_call_structured(self, latency_ms: int, status: str,
                             error_info: Optional[Tuple[str, str]] = None) -> None:
        """记录结构化调用日志（JSON 格式，缺陷 #8）

        与 LLMClient._log_call_structured 接口一致，便于统一审计。
        """
        log_entry = {
            'event': 'llm_call',
            'timestamp': datetime.datetime.now().isoformat(),
            'provider': 'local',
            'model': self.config.local_model_path or 'unknown',
            'calls': self._token_usage['calls'],
            'input_tokens': self._token_usage['prompt_tokens'],
            'output_tokens': self._token_usage['completion_tokens'],
            'total_tokens': self._token_usage['total_tokens'],
            'latency_ms': latency_ms,
            'status': status,
        }
        if error_info is not None:
            log_entry['error_type'] = error_info[0]
            log_entry['error_message'] = error_info[1][:200]
        LOGGER.info("本地 LLM 调用审计: %s",
                    json.dumps(log_entry, ensure_ascii=False))

    def _accumulate_local_usage(self, prompt_tokens: int,
                                completion_tokens: int) -> None:
        """累加本地模型的 token 用量（缺陷 #18）

        本地模型无 HTTP 响应中的 usage 字段，通过 tokenizer token 数计算。
        total_tokens = prompt_tokens + completion_tokens。

        Args:
            prompt_tokens: 输入 token 数
            completion_tokens: 输出 token 数
        """
        self._token_usage['calls'] += 1
        total = prompt_tokens + completion_tokens
        self._token_usage['prompt_tokens'] += prompt_tokens
        self._token_usage['completion_tokens'] += completion_tokens
        self._token_usage['total_tokens'] += total

    def get_usage(self) -> Dict[str, int]:
        """查询当前客户端累计的 token 用量（缺陷 #18）

        Returns:
            dict: 包含 prompt_tokens / completion_tokens / total_tokens / calls
        """
        return dict(self._token_usage)

    def reset_usage(self) -> None:
        """清零 token 用量统计（缺陷 #18）"""
        self._token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'calls': 0,
        }

    def chat_json(self, messages: List[Dict[str, Any]],
                  schema: Optional[Dict[str, Any]] = None,
                  **kwargs) -> Optional[Dict[str, Any]]:
        """本地生成 + JSON 解析（接口与 LLMClient 一致）"""
        try:
            content = self.chat(messages, **kwargs)
        except Exception as e:
            LOGGER.warning("LocalLLMClient.chat_json 调用失败: %s", e)
            return None

        parsed = _parse_json_response(content)
        if parsed is None:
            LOGGER.warning("本地 LLM 返回内容无法解析为 JSON: %.200s", content)
            return None

        if schema is not None:
            if not _validate_schema(parsed, schema):
                LOGGER.warning("本地 LLM 返回 JSON 未通过 schema 验证")
                return None
        return parsed

    def close(self) -> None:
        """释放模型资源"""
        if self._model is not None:
            try:
                del self._model
                del self._tokenizer
            except Exception as e:
                LOGGER.debug("清理AI模型资源失败: %s", e)
            self._model = None
            self._tokenizer = None

    def __enter__(self) -> 'LocalLLMClient':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ==============================================================================
# 工厂函数
# ==============================================================================

# 会话级客户端缓存：按 config 的稳定字段哈希键
# 避免每次 AI 调用重新建立 HTTP 连接或重新加载本地 7B 模型
_CLIENT_CACHE: Dict[str, Any] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


def _config_cache_key(config: AIConfig) -> str:
    """根据 config 的稳定字段生成缓存键

    仅取影响客户端行为的核心字段（不含 timeout/temperature 等运行时参数）。
    """
    parts = (
        config.provider,
        config.api_key or '',
        config.base_url,
        config.model,
        config.local_model_path or '',
        config.local_device,
    )
    return '|'.join(parts)


def _clear_client_cache() -> None:
    """清空客户端缓存（测试用 / 配置切换时）"""
    with _CLIENT_CACHE_LOCK:
        # 关闭已缓存客户端（best-effort）
        for c in _CLIENT_CACHE.values():
            try:
                if hasattr(c, 'close'):
                    c.close()
            except Exception as e:
                LOGGER.debug("关闭缓存客户端失败: %s", e)
        _CLIENT_CACHE.clear()


def make_client(config: Optional[AIConfig] = None
                ) -> Optional[Union[LLMClient, LocalLLMClient]]:
    """根据配置创建或复用 LLM 客户端

    优先级：
    1. config.is_local (local_model_path 设置) → LocalLLMClient
    2. config.api_key 设置 → LLMClient（云端）
    3. 都未设置 → None（调用方降级到正则/规则）

    缓存行为：
    - 相同 config（核心字段一致）返回同一客户端实例，复用 HTTP 连接 / 已加载模型
    - 未配置时返回 None，且不缓存（便于下次配置变更后重试）
    - 配置变更（api_key/model/local_model_path 切换）会创建新实例

    Args:
        config: AIConfig 实例，None 时从环境变量加载

    Returns:
        LLMClient / LocalLLMClient / None
    """
    if config is None:
        config = AIConfig.from_env()

    if config.is_local:
        key = _config_cache_key(config)
        with _CLIENT_CACHE_LOCK:
            cached = _CLIENT_CACHE.get(key)
            if cached is not None:
                return cached
            client = LocalLLMClient(config)
            _CLIENT_CACHE[key] = client
            return client
    if config.api_key:
        key = _config_cache_key(config)
        with _CLIENT_CACHE_LOCK:
            cached = _CLIENT_CACHE.get(key)
            if cached is not None:
                return cached
            client = LLMClient(config)
            _CLIENT_CACHE[key] = client
            return client
    return None


__all__ = [
    'LLMClient',
    'LocalLLMClient',
    'make_client',
    '_validate_schema',
    '_strip_code_fences',
    '_parse_json_response',
]
