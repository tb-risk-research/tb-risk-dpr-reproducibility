#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 测试套件 — LLMClient token usage 统计

缺陷 #18：LLMClient.chat() 解析响应中的 usage 字段并累加到 self._token_usage，
提供 get_usage() 方法查询累计用量，日志中记录每次调用的 input/output token 数。

测试覆盖：
- LLMClient 初始化时 _token_usage 为零值
- chat() 成功后 get_usage() 返回累计的 prompt/completion/total tokens
- 多次调用累加（不重置）
- 响应缺失 usage 字段时不抛异常，token_usage 保持 0
- chat_json() 成功也会累加 token usage
- reset_usage() 清零
- 调用日志包含 input/output token 数
- LocalLLMClient 通过 tokenizer token 数计算 usage（依赖 transformers/torch）
"""

import json
import logging
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


def _ok_response(content: str, usage=None):
    """构造 OpenAI 兼容的成功响应 JSON

    Args:
        content: 助手回复内容
        usage: 可选 usage dict；None 表示不带 usage 字段
    """
    resp = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}
        ],
    }
    if usage is not None:
        resp["usage"] = usage
    return resp


def _make_cloud_client(handler, **cfg_kwargs):
    """构造带 MockTransport 的 LLMClient"""
    from tb_risk.ai.client import LLMClient
    from tb_risk.ai.config import AIConfig
    cfg = AIConfig(api_key='sk-test', **cfg_kwargs)
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, timeout=cfg.timeout)
    return LLMClient(cfg, http_client=http)


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLLMClientTokenUsageInit(unittest.TestCase):
    """LLMClient 初始化时 token usage 应为零值"""

    def test_init_has_zero_token_usage(self):
        """构造后 _token_usage 应为零值"""
        from tb_risk.ai.client import LLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test')
        client = LLMClient(cfg)
        usage = client.get_usage()
        self.assertEqual(usage['prompt_tokens'], 0)
        self.assertEqual(usage['completion_tokens'], 0)
        self.assertEqual(usage['total_tokens'], 0)
        self.assertEqual(usage['calls'], 0)

    def test_get_usage_returns_dict(self):
        """get_usage() 返回 dict，包含必要字段"""
        from tb_risk.ai.client import LLMClient
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test')
        client = LLMClient(cfg)
        usage = client.get_usage()
        self.assertIsInstance(usage, dict)
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'calls'):
            self.assertIn(key, usage, f"缺少字段 {key}")


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLLMClientTokenUsageAccumulation(unittest.TestCase):
    """LLMClient.chat() 后 token usage 应累加"""

    def test_chat_accumulates_token_usage(self):
        """单次 chat() 后 get_usage() 返回该次响应的 token 数"""
        usage_payload = {"prompt_tokens": 10, "completion_tokens": 5,
                         "total_tokens": 15}
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok", usage=usage_payload)))
        result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "ok")

        usage = client.get_usage()
        self.assertEqual(usage['prompt_tokens'], 10)
        self.assertEqual(usage['completion_tokens'], 5)
        self.assertEqual(usage['total_tokens'], 15)
        self.assertEqual(usage['calls'], 1)

    def test_chat_accumulates_across_multiple_calls(self):
        """多次调用 chat() 应累加 token 数与 calls 计数"""
        # 第一次：10 + 5 = 15
        # 第二次：20 + 10 = 30
        # 累计：30 + 15 = 45 prompt/completion
        responses = iter([
            httpx.Response(200, json=_ok_response(
                "a", usage={"prompt_tokens": 10, "completion_tokens": 5,
                            "total_tokens": 15})),
            httpx.Response(200, json=_ok_response(
                "b", usage={"prompt_tokens": 20, "completion_tokens": 10,
                            "total_tokens": 30})),
        ])
        client = _make_cloud_client(lambda r: next(responses))

        client.chat([{"role": "user", "content": "first"}])
        client.chat([{"role": "user", "content": "second"}])

        usage = client.get_usage()
        self.assertEqual(usage['prompt_tokens'], 30)   # 10 + 20
        self.assertEqual(usage['completion_tokens'], 15)  # 5 + 10
        self.assertEqual(usage['total_tokens'], 45)    # 15 + 30
        self.assertEqual(usage['calls'], 2)

    def test_chat_without_usage_field_does_not_break(self):
        """响应缺失 usage 字段时不抛异常，token_usage 保持 0"""
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok", usage=None)))
        result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "ok")

        usage = client.get_usage()
        self.assertEqual(usage['prompt_tokens'], 0)
        self.assertEqual(usage['completion_tokens'], 0)
        self.assertEqual(usage['total_tokens'], 0)
        # calls 仍应计数（调用确实发生了，只是响应没 usage）
        self.assertEqual(usage['calls'], 1)

    def test_chat_with_partial_usage_field(self):
        """usage 字段只含部分子字段时仍累加存在的部分"""
        # 仅 prompt_tokens，无 completion_tokens/total_tokens
        usage_payload = {"prompt_tokens": 7}
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok", usage=usage_payload)))
        client.chat([{"role": "user", "content": "hi"}])

        usage = client.get_usage()
        self.assertEqual(usage['prompt_tokens'], 7)
        # 缺失字段按 0 累加
        self.assertEqual(usage['completion_tokens'], 0)
        self.assertEqual(usage['total_tokens'], 0)
        self.assertEqual(usage['calls'], 1)


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLLMClientChatJsonTokenUsage(unittest.TestCase):
    """LLMClient.chat_json() 成功时应累加 token usage"""

    def test_chat_json_accumulates_usage_on_success(self):
        """chat_json() 成功调用后 token usage 应累加"""
        usage_payload = {"prompt_tokens": 12, "completion_tokens": 8,
                         "total_tokens": 20}
        payload = '{"name": "张三", "age": 35}'
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response(payload, usage=usage_payload)))
        result = client.chat_json([{"role": "user", "content": "extract"}])
        self.assertEqual(result, {"name": "张三", "age": 35})

        usage = client.get_usage()
        self.assertEqual(usage['prompt_tokens'], 12)
        self.assertEqual(usage['completion_tokens'], 8)
        self.assertEqual(usage['total_tokens'], 20)
        self.assertEqual(usage['calls'], 1)

    def test_chat_json_accumulates_usage_even_when_schema_fails(self):
        """chat_json() schema 验证失败时 token usage 仍应累加

        因为 API 调用已经发生、token 已经消耗，schema 验证是客户端校验。
        """
        usage_payload = {"prompt_tokens": 15, "completion_tokens": 10,
                         "total_tokens": 25}
        # JSON 解析成功但缺 age 字段
        payload = '{"name": "张三"}'
        schema = {'name': str, 'age': int}
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response(payload, usage=usage_payload)))
        result = client.chat_json([{"role": "user", "content": "extract"}],
                                  schema=schema)
        self.assertIsNone(result)  # schema 验证失败返回 None

        # 但 token usage 仍应累加（API 已调用）
        usage = client.get_usage()
        self.assertEqual(usage['prompt_tokens'], 15)
        self.assertEqual(usage['completion_tokens'], 10)
        self.assertEqual(usage['calls'], 1)


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLLMClientResetUsage(unittest.TestCase):
    """reset_usage() 清零 token 统计"""

    def test_reset_usage_zeroes_counters(self):
        """reset_usage() 后所有字段归零"""
        usage_payload = {"prompt_tokens": 10, "completion_tokens": 5,
                         "total_tokens": 15}
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok", usage=usage_payload)))
        client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(client.get_usage()['calls'], 1)

        client.reset_usage()
        usage = client.get_usage()
        self.assertEqual(usage['prompt_tokens'], 0)
        self.assertEqual(usage['completion_tokens'], 0)
        self.assertEqual(usage['total_tokens'], 0)
        self.assertEqual(usage['calls'], 0)


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLLMClientTokenUsageLogging(unittest.TestCase):
    """chat() 成功后应记录 token 用量日志"""

    def test_logs_token_usage_after_chat(self):
        """chat() 成功后日志应包含 input/output token 数"""
        usage_payload = {"prompt_tokens": 42, "completion_tokens": 13,
                         "total_tokens": 55}
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok", usage=usage_payload)))

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            client.chat([{"role": "user", "content": "hi"}])

        # 至少一条日志包含 token 数信息
        joined = '\n'.join(cm.output)
        self.assertIn('42', joined)  # prompt_tokens
        self.assertIn('13', joined)  # completion_tokens
        # 日志应包含 "token" 关键词
        self.assertTrue(any('token' in msg.lower() for msg in cm.output),
                        f"日志应包含 token 关键词: {cm.output}")


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLLMClientTokenUsageRetries(unittest.TestCase):
    """重试场景下 token usage 累计策略"""

    def test_failed_calls_not_counted_in_usage(self):
        """失败的 HTTP 调用（429/5xx）不应累加 token usage（无 usage 字段）

        即使重试后最终成功，仅成功响应中的 usage 应累加。
        """
        call_count = {'n': 0}

        def handler(request):
            call_count['n'] += 1
            if call_count['n'] < 3:
                return httpx.Response(429, json={"error": "rate limit"})
            return httpx.Response(200, json=_ok_response(
                "ok", usage={"prompt_tokens": 10, "completion_tokens": 5,
                             "total_tokens": 15}))

        client = _make_cloud_client(handler, max_retries=5)
        result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "ok")
        self.assertEqual(call_count['n'], 3)  # 重试 2 次 + 成功 1 次

        usage = client.get_usage()
        # 仅成功那次的 usage 被累加
        self.assertEqual(usage['prompt_tokens'], 10)
        self.assertEqual(usage['completion_tokens'], 5)
        self.assertEqual(usage['calls'], 1)


class TestLocalLLMClientTokenUsage(unittest.TestCase):
    """LocalLLMClient token usage 测试（通过 tokenizer 计算）"""

    def setUp(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("torch/transformers not installed")

    def _make_mock_local_client(self, prompt_token_count=10,
                                output_token_count=5):
        """构造一个 LocalLLMClient，tokenizer 与 model 都被 mock

        Args:
            prompt_token_count: mock tokenizer 返回的 prompt token 数
            output_token_count: mock model.generate 返回的输出 token 数
        """
        from tb_risk.ai.client import LocalLLMClient
        from tb_risk.ai.config import AIConfig

        cfg = AIConfig(local_model_path='/fake/model')

        # Mock tokenizer
        mock_tokenizer = mock.MagicMock()
        # apply_chat_template 返回字符串
        mock_tokenizer.apply_chat_template.return_value = "fake prompt"
        # tokenizer 调用返回 dict {'input_ids': tensor[N, prompt_count]}
        import torch
        mock_tokenizer.return_value = {
            'input_ids': torch.zeros((1, prompt_token_count), dtype=torch.long)
        }
        mock_tokenizer.eos_token_id = 0
        # decode 返回字符串
        mock_tokenizer.decode.return_value = "AI 回复"

        # Mock model: generate 返回形状 [1, prompt + output] 的 tensor
        mock_model = mock.MagicMock()
        total = prompt_token_count + output_token_count
        mock_model.generate.return_value = torch.zeros(
            (1, total), dtype=torch.long)

        client = LocalLLMClient(cfg)
        client._tokenizer = mock_tokenizer
        client._model = mock_model
        return client, mock_tokenizer, mock_model

    def test_local_chat_accumulates_token_usage(self):
        """LocalLLMClient.chat() 应通过 tokenizer 计算 token 数并累加"""
        client, _, _ = self._make_mock_local_client(
            prompt_token_count=15, output_token_count=8)
        result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "AI 回复")

        usage = client.get_usage()
        self.assertEqual(usage['prompt_tokens'], 15)
        self.assertEqual(usage['completion_tokens'], 8)
        self.assertEqual(usage['total_tokens'], 23)  # 15 + 8
        self.assertEqual(usage['calls'], 1)

    def test_local_chat_accumulates_across_multiple_calls(self):
        """LocalLLMClient 多次调用 chat() 累加 token 数"""
        client, _, _ = self._make_mock_local_client(
            prompt_token_count=10, output_token_count=5)
        client.chat([{"role": "user", "content": "first"}])
        client.chat([{"role": "user", "content": "second"}])

        usage = client.get_usage()
        self.assertEqual(usage['prompt_tokens'], 20)  # 10 * 2
        self.assertEqual(usage['completion_tokens'], 10)  # 5 * 2
        self.assertEqual(usage['total_tokens'], 30)
        self.assertEqual(usage['calls'], 2)

    def test_local_reset_usage_zeroes_counters(self):
        """LocalLLMClient.reset_usage() 清零"""
        client, _, _ = self._make_mock_local_client(
            prompt_token_count=10, output_token_count=5)
        client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(client.get_usage()['calls'], 1)

        client.reset_usage()
        usage = client.get_usage()
        self.assertEqual(usage['prompt_tokens'], 0)
        self.assertEqual(usage['completion_tokens'], 0)
        self.assertEqual(usage['total_tokens'], 0)
        self.assertEqual(usage['calls'], 0)


if __name__ == '__main__':
    unittest.main()
