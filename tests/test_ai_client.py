#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — AI 客户端模块 (tb_risk.ai.client)

测试 Layer 3b: LLMClient（OpenAI 兼容协议）+ LocalLLMClient（transformers+torch）+
make_client 工厂。

设计：
- HTTP 用 httpx.MockTransport 注入（无需真实 API Key）
- 重试行为：429/5xx 重试，4xx 不重试，超时重试
- chat_json: 解析 JSON、剥离 markdown 代码块、schema 验证
- LocalLLMClient 测试在无 transformers/torch 时自动 skip
"""

import json
import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import tenacity
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

AI_DEPS_AVAILABLE = HTTPX_AVAILABLE and TENACITY_AVAILABLE


def _make_mock_transport(handler):
    """构造 httpx.MockTransport"""
    return httpx.MockTransport(handler)


def _ok_response(content: str):
    """构造 OpenAI 兼容的成功响应 JSON"""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLLMClientConstruction(unittest.TestCase):
    """LLMClient 构造测试"""

    def test_constructs_with_config(self):
        """LLMClient 接受 AIConfig"""
        from tb_risk.ai.client import LLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test')
        client = LLMClient(cfg)
        self.assertIs(client.config, cfg)

    def test_lazy_http_client(self):
        """HTTP 客户端懒构造（构造时不创建）"""
        from tb_risk.ai.client import LLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test')
        client = LLMClient(cfg)
        # 未调用 chat 前不应创建 http 客户端
        self.assertIsNone(client._http)

    def test_accepts_injected_http_client(self):
        """可注入 httpx.Client（测试用）"""
        from tb_risk.ai.client import LLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test')
        injected = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
        client = LLMClient(cfg, http_client=injected)
        self.assertIs(client._http, injected)


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLLMClientChat(unittest.TestCase):
    """LLMClient.chat() 测试"""

    def _make_client(self, handler, **cfg_kwargs):
        """构造带 MockTransport 的 LLMClient"""
        from tb_risk.ai.client import LLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test', **cfg_kwargs)
        transport = httpx.MockTransport(handler)
        http = httpx.Client(transport=transport, timeout=cfg.timeout)
        return LLMClient(cfg, http_client=http)

    def test_chat_returns_content_string(self):
        """chat() 返回 content 字符串"""
        from tb_risk.ai.client import LLMClient

        def handler(request: httpx.Request):
            return httpx.Response(200, json=_ok_response("你好"))

        client = self._make_client(handler)
        result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "你好")

    def test_chat_sends_openai_format_request(self):
        """chat() 发送 OpenAI 格式 POST 请求"""
        from tb_risk.ai.client import LLMClient
        captured = {}

        def handler(request: httpx.Request):
            captured['url'] = str(request.url)
            captured['headers'] = dict(request.headers)
            captured['body'] = json.loads(request.content)
            return httpx.Response(200, json=_ok_response("ok"))

        client = self._make_client(handler, model='deepseek-chat')
        client.chat([{"role": "user", "content": "hi"}])

        # URL 是 base_url + /chat/completions
        self.assertTrue(captured['url'].endswith('/chat/completions'))
        # Authorization header
        self.assertEqual(captured['headers']['authorization'], 'Bearer sk-test')
        # 请求体含 model/messages/temperature
        self.assertEqual(captured['body']['model'], 'deepseek-chat')
        self.assertEqual(captured['body']['messages'],
                         [{"role": "user", "content": "hi"}])
        self.assertIn('temperature', captured['body'])

    def test_chat_overrides_temperature_via_kwargs(self):
        """chat(temperature=0.5) 覆盖默认温度"""
        captured = {}

        def handler(request: httpx.Request):
            captured['body'] = json.loads(request.content)
            return httpx.Response(200, json=_ok_response("ok"))

        client = self._make_client(handler)
        client.chat([{"role": "user", "content": "hi"}], temperature=0.5)
        self.assertEqual(captured['body']['temperature'], 0.5)

    def test_chat_retries_on_429_then_succeeds(self):
        """429 重试后成功"""
        call_count = {'n': 0}

        def handler(request: httpx.Request):
            call_count['n'] += 1
            if call_count['n'] < 3:
                return httpx.Response(429, json={"error": "rate limit"})
            return httpx.Response(200, json=_ok_response("ok"))

        client = self._make_client(handler, max_retries=5)
        result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "ok")
        self.assertEqual(call_count['n'], 3)

    def test_chat_retries_on_5xx_then_succeeds(self):
        """5xx 重试后成功"""
        call_count = {'n': 0}

        def handler(request: httpx.Request):
            call_count['n'] += 1
            if call_count['n'] < 2:
                return httpx.Response(503, json={"error": "service unavailable"})
            return httpx.Response(200, json=_ok_response("ok"))

        client = self._make_client(handler, max_retries=5)
        result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "ok")
        self.assertGreaterEqual(call_count['n'], 2)

    def test_chat_raises_after_max_retries_on_persistent_5xx(self):
        """持续 5xx 超过 max_retries 后抛出"""
        call_count = {'n': 0}

        def handler(request: httpx.Request):
            call_count['n'] += 1
            return httpx.Response(500, json={"error": "internal"})

        client = self._make_client(handler, max_retries=3)
        with self.assertRaises(Exception):  # HTTPStatusError 或 RetryError
            client.chat([{"role": "user", "content": "hi"}])
        # 最多 3 次尝试
        self.assertLessEqual(call_count['n'], 3)

    def test_chat_does_not_retry_on_4xx_except_429(self):
        """4xx（非 429）不重试，立即抛出"""
        call_count = {'n': 0}

        def handler(request: httpx.Request):
            call_count['n'] += 1
            return httpx.Response(401, json={"error": "unauthorized"})

        client = self._make_client(handler, max_retries=5)
        with self.assertRaises(Exception):
            client.chat([{"role": "user", "content": "hi"}])
        # 仅 1 次尝试（不重试）
        self.assertEqual(call_count['n'], 1)

    def test_chat_retries_on_timeout(self):
        """超时重试"""
        call_count = {'n': 0}

        def handler(request: httpx.Request):
            call_count['n'] += 1
            if call_count['n'] < 2:
                raise httpx.TimeoutException("simulated timeout")
            return httpx.Response(200, json=_ok_response("ok"))

        client = self._make_client(handler, max_retries=5)
        result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "ok")
        self.assertGreaterEqual(call_count['n'], 2)

    def test_chat_raises_on_unconfigured_client(self):
        """未配置 api_key 时调用 chat 抛出 ValueError"""
        from tb_risk.ai.client import LLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig()  # 无 api_key
        client = LLMClient(cfg, http_client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        ))
        with self.assertRaises(ValueError):
            client.chat([{"role": "user", "content": "hi"}])


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLLMClientChatJson(unittest.TestCase):
    """LLMClient.chat_json() 测试"""

    def _make_client(self, handler, **cfg_kwargs):
        from tb_risk.ai.client import LLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test', **cfg_kwargs)
        transport = httpx.MockTransport(handler)
        http = httpx.Client(transport=transport, timeout=cfg.timeout)
        return LLMClient(cfg, http_client=http)

    def test_chat_json_parses_valid_json(self):
        """有效 JSON 返回 dict"""
        payload = '{"name": "张三", "age": 35}'
        client = self._make_client(
            lambda r: httpx.Response(200, json=_ok_response(payload)))
        result = client.chat_json([{"role": "user", "content": "extract"}])
        self.assertEqual(result, {"name": "张三", "age": 35})

    def test_chat_json_strips_markdown_code_fences(self):
        """剥离 ```json ... ``` 代码块"""
        payload = '```json\n{"name": "李四", "age": 40}\n```'
        client = self._make_client(
            lambda r: httpx.Response(200, json=_ok_response(payload)))
        result = client.chat_json([{"role": "user", "content": "extract"}])
        self.assertEqual(result, {"name": "李四", "age": 40})

    def test_chat_json_strips_plain_code_fences(self):
        """剥离 ``` ... ``` 代码块"""
        payload = '```\n{"name": "王五"}\n```'
        client = self._make_client(
            lambda r: httpx.Response(200, json=_ok_response(payload)))
        result = client.chat_json([{"role": "user", "content": "extract"}])
        self.assertEqual(result, {"name": "王五"})

    def test_chat_json_returns_none_on_invalid_json(self):
        """非法 JSON 返回 None（不抛出）"""
        client = self._make_client(
            lambda r: httpx.Response(200, json=_ok_response("not json at all")))
        result = client.chat_json([{"role": "user", "content": "extract"}])
        self.assertIsNone(result)

    def test_chat_json_validates_against_schema(self):
        """schema 验证：缺字段返回 None"""
        schema = {'name': str, 'age': int}
        payload = '{"name": "张三"}'  # 缺 age
        client = self._make_client(
            lambda r: httpx.Response(200, json=_ok_response(payload)))
        result = client.chat_json([{"role": "user", "content": "extract"}],
                                  schema=schema)
        self.assertIsNone(result)

    def test_chat_json_validates_type_mismatch(self):
        """schema 验证：类型不匹配返回 None"""
        schema = {'name': str, 'age': int}
        payload = '{"name": "张三", "age": "thirty-five"}'  # age 是 str 非 int
        client = self._make_client(
            lambda r: httpx.Response(200, json=_ok_response(payload)))
        result = client.chat_json([{"role": "user", "content": "extract"}],
                                  schema=schema)
        self.assertIsNone(result)

    def test_chat_json_passes_valid_schema(self):
        """schema 验证：全部字段正确通过"""
        schema = {'name': str, 'age': int, 'symptoms': list}
        payload = '{"name": "张三", "age": 35, "symptoms": ["咳嗽", "发热"]}'
        client = self._make_client(
            lambda r: httpx.Response(200, json=_ok_response(payload)))
        result = client.chat_json([{"role": "user", "content": "extract"}],
                                  schema=schema)
        self.assertIsNotNone(result)
        self.assertEqual(result['name'], '张三')

    def test_chat_json_accepts_optional_fields(self):
        """schema 支持 Optional 字段（None 也接受）"""
        from tb_risk.ai.client import _validate_schema
        schema = {'name': str, 'age': (int, type(None))}
        # 有 age
        self.assertTrue(_validate_schema({'name': 'x', 'age': 35}, schema))
        # age=None 也接受
        self.assertTrue(_validate_schema({'name': 'x', 'age': None}, schema))
        # 缺 age 不接受
        self.assertFalse(_validate_schema({'name': 'x'}, schema))


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestMakeClientFactory(unittest.TestCase):
    """make_client() 工厂测试"""

    def test_returns_local_when_local_config(self):
        """local_model_path 设置 → 返回 LocalLLMClient"""
        from tb_risk.ai.client import make_client, LocalLLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(local_model_path='/path/to/model')
        client = make_client(cfg)
        self.assertIsInstance(client, LocalLLMClient)

    def test_returns_cloud_when_api_key_only(self):
        """仅 api_key → 返回 LLMClient（云端）"""
        from tb_risk.ai.client import make_client, LLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test')
        client = make_client(cfg)
        self.assertIsInstance(client, LLMClient)

    def test_returns_none_when_unconfigured(self):
        """未配置 → 返回 None（调用方降级）"""
        from tb_risk.ai.client import make_client
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig()  # 无 api_key 无 local
        client = make_client(cfg)
        self.assertIsNone(client)

    def test_local_takes_priority_over_cloud(self):
        """同时设置时本地优先（敏感场景）"""
        from tb_risk.ai.client import make_client, LocalLLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test', local_model_path='/path/to/model')
        client = make_client(cfg)
        self.assertIsInstance(client, LocalLLMClient)

    def test_defaults_to_env_config(self):
        """无参数时从环境变量加载"""
        from tb_risk.ai.client import make_client
        # 清空环境
        saved = {k: os.environ.pop(k, None) for k in [
            'TB_AI_PROVIDER', 'TB_AI_API_KEY', 'TB_AI_LOCAL_MODEL']}
        try:
            client = make_client()
            self.assertIsNone(client)  # 无配置 → None
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLLMClientLifecycle(unittest.TestCase):
    """LLMClient 生命周期测试"""

    def test_close_releases_http_client(self):
        """close() 关闭 httpx 客户端"""
        from tb_risk.ai.client import LLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test')
        http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
        client = LLMClient(cfg, http_client=http)
        client.close()
        self.assertTrue(http.is_closed)

    def test_context_manager_support(self):
        """支持 with 语句"""
        from tb_risk.ai.client import LLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test')
        http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
        with LLMClient(cfg, http_client=http) as client:
            self.assertIsNotNone(client)
        self.assertTrue(http.is_closed)


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLocalLLMClient(unittest.TestCase):
    """LocalLLMClient 测试（无 transformers/torch 时部分 skip）"""

    def test_constructs_with_local_config(self):
        """LocalLLMClient 接受本地配置"""
        from tb_risk.ai.client import LocalLLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(local_model_path='/path/to/Qwen2.5-7B')
        client = LocalLLMClient(cfg)
        self.assertIs(client.config, cfg)

    def test_lazy_model_loading(self):
        """模型懒加载（构造时不加载）"""
        from tb_risk.ai.client import LocalLLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(local_model_path='/path/to/Qwen2.5-7B')
        client = LocalLLMClient(cfg)
        # 构造后模型未加载
        self.assertIsNone(client._model)
        self.assertIsNone(client._tokenizer)

    def test_chat_without_transformers_raises_or_falls_back(self):
        """无 transformers 时 chat() 抛出 RuntimeError 或 ImportError"""
        from tb_risk.ai.client import LocalLLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(local_model_path='/path/to/Qwen2.5-7B')
        client = LocalLLMClient(cfg)
        # 无论 transformers 是否安装，路径不存在必然抛错
        with self.assertRaises((RuntimeError, ImportError, OSError, ValueError)):
            client.chat([{"role": "user", "content": "hi"}])


if __name__ == '__main__':
    unittest.main()
