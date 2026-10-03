#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 测试套件 — LLM 调用结构化审计日志

缺陷 #8：LLMClient.chat 与 LocalLLMClient.chat 在每次调用后写一条
结构化日志（JSON 格式），包含 timestamp/provider/model/calls/
input_tokens/output_tokens/latency_ms/status，便于事后审计与成本核算。

测试覆盖：
- 成功调用后日志为可解析 JSON，包含所有必要字段
- status 字段在成功时为 "success"，失败时为 "error"
- latency_ms 为非负整数
- timestamp 为 ISO 8601 格式
- calls 字段递增
- provider/model 与 config 一致
- 失败时包含 error_type 与 error_message
- LocalLLMClient 同样记录结构化日志（mock torch/transformers）
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
    """构造 OpenAI 兼容的成功响应 JSON"""
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


def _extract_structured_log_entries(cm_output):
    """从 assertLogs 捕获的日志中提取可解析为 JSON 的结构化条目

    Returns:
        list[dict]: 成功解析为 dict 的日志条目
    """
    entries = []
    for line in cm_output:
        # 日志格式 "LEVEL:logger:message"，取最后一个冒号后的内容
        # 尝试找到 JSON 起始位置（第一个 '{'）
        brace_idx = line.find('{')
        if brace_idx == -1:
            continue
        json_str = line[brace_idx:]
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                entries.append(parsed)
        except json.JSONDecodeError:
            continue
    return entries


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLLMClientStructuredLogging(unittest.TestCase):
    """缺陷 #8：LLMClient.chat 结构化调用日志（JSON 格式）"""

    def test_chat_emits_structured_json_log(self):
        """chat() 成功后日志中应包含可解析为 JSON 的结构化条目"""
        usage = {"prompt_tokens": 42, "completion_tokens": 13,
                 "total_tokens": 55}
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok", usage=usage)))

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            client.chat([{"role": "user", "content": "hi"}])

        entries = _extract_structured_log_entries(cm.output)
        self.assertTrue(entries, f"未找到 JSON 结构化日志: {cm.output}")

    def test_structured_log_contains_required_fields(self):
        """结构化日志必须包含 event/timestamp/provider/model/calls/
        input_tokens/output_tokens/latency_ms/status 字段"""
        usage = {"prompt_tokens": 10, "completion_tokens": 5,
                 "total_tokens": 15}
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok", usage=usage)))

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            client.chat([{"role": "user", "content": "hi"}])

        entries = _extract_structured_log_entries(cm.output)
        self.assertTrue(entries)
        entry = entries[-1]  # 取最后一条（本次调用的审计日志）
        required_fields = {
            'event', 'timestamp', 'provider', 'model', 'calls',
            'input_tokens', 'output_tokens', 'latency_ms', 'status',
        }
        for field in required_fields:
            self.assertIn(field, entry, f"结构化日志缺少字段 {field}: {entry}")

    def test_structured_log_event_is_llm_call(self):
        """event 字段应为 'llm_call'"""
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok")))

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            client.chat([{"role": "user", "content": "hi"}])

        entries = _extract_structured_log_entries(cm.output)
        self.assertTrue(entries)
        self.assertEqual(entries[-1]['event'], 'llm_call')

    def test_structured_log_status_success_on_success(self):
        """成功调用的 status 应为 'success'"""
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok")))

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            client.chat([{"role": "user", "content": "hi"}])

        entries = _extract_structured_log_entries(cm.output)
        self.assertTrue(entries)
        self.assertEqual(entries[-1]['status'], 'success')

    def test_structured_log_status_error_on_failure(self):
        """失败调用（重试耗尽 5xx）的 status 应为 'error'，
        且包含 error_type 字段"""
        def handler(req):
            return httpx.Response(500, text="Internal Server Error")
        # max_retries=1 避免测试等待过久
        client = _make_cloud_client(handler, max_retries=1)

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            with self.assertRaises(Exception):
                client.chat([{"role": "user", "content": "hi"}])

        entries = _extract_structured_log_entries(cm.output)
        # 失败时也应记录结构化日志
        error_entries = [e for e in entries if e.get('status') == 'error']
        self.assertTrue(error_entries,
                        f"失败调用未记录 status=error 的结构化日志: {entries}")
        entry = error_entries[-1]
        self.assertIn('error_type', entry)
        self.assertTrue(entry['error_type'])

    def test_structured_log_latency_ms_is_non_negative_int(self):
        """latency_ms 应为非负整数"""
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok")))

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            client.chat([{"role": "user", "content": "hi"}])

        entries = _extract_structured_log_entries(cm.output)
        self.assertTrue(entries)
        latency = entries[-1]['latency_ms']
        self.assertIsInstance(latency, int)
        self.assertGreaterEqual(latency, 0)

    def test_structured_log_calls_counter_increments(self):
        """多次调用后 calls 字段应递增"""
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok")))

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            client.chat([{"role": "user", "content": "hi"}])
            client.chat([{"role": "user", "content": "hi"}])
            client.chat([{"role": "user", "content": "hi"}])

        entries = _extract_structured_log_entries(cm.output)
        # 取后三条（每次调用一条审计日志）
        call_entries = [e for e in entries if e.get('event') == 'llm_call']
        self.assertGreaterEqual(len(call_entries), 3)
        calls_sequence = [e['calls'] for e in call_entries[-3:]]
        self.assertEqual(calls_sequence, [1, 2, 3],
                         f"calls 应递增 1→2→3: {calls_sequence}")

    def test_structured_log_includes_provider_and_model(self):
        """日志包含 provider 与 model 字段，与 config 一致"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test', provider='deepseek',
                       model='deepseek-chat')
        transport = httpx.MockTransport(
            lambda r: httpx.Response(200, json=_ok_response("ok")))
        http = httpx.Client(transport=transport, timeout=cfg.timeout)
        from tb_risk.ai.client import LLMClient
        client = LLMClient(cfg, http_client=http)

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            client.chat([{"role": "user", "content": "hi"}])

        entries = _extract_structured_log_entries(cm.output)
        self.assertTrue(entries)
        entry = entries[-1]
        self.assertEqual(entry['provider'], 'deepseek')
        self.assertEqual(entry['model'], 'deepseek-chat')

    def test_structured_log_timestamp_is_iso_format(self):
        """timestamp 应为 ISO 8601 格式（可被 datetime.fromisoformat 解析）"""
        import datetime
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok")))

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            client.chat([{"role": "user", "content": "hi"}])

        entries = _extract_structured_log_entries(cm.output)
        self.assertTrue(entries)
        ts = entries[-1]['timestamp']
        # 应可被 fromisoformat 解析（Python 3.7+ 支持）
        parsed = datetime.datetime.fromisoformat(ts)
        self.assertIsInstance(parsed, datetime.datetime)

    def test_structured_log_total_tokens_equals_input_plus_output(self):
        """total_tokens 应等于 input_tokens + output_tokens"""
        usage = {"prompt_tokens": 100, "completion_tokens": 30,
                 "total_tokens": 130}
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response("ok", usage=usage)))

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            client.chat([{"role": "user", "content": "hi"}])

        entries = _extract_structured_log_entries(cm.output)
        self.assertTrue(entries)
        entry = entries[-1]
        # 累计值（首次调用 = 本次值）
        self.assertEqual(entry['input_tokens'], 100)
        self.assertEqual(entry['output_tokens'], 30)
        self.assertEqual(entry['total_tokens'], 130)

    def test_structured_log_includes_cumulative_tokens(self):
        """多次调用后累计 token 数应正确反映"""
        client = _make_cloud_client(
            lambda r: httpx.Response(200, json=_ok_response(
                "ok", usage={"prompt_tokens": 10, "completion_tokens": 5,
                             "total_tokens": 15})))

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            client.chat([{"role": "user", "content": "hi"}])
            client.chat([{"role": "user", "content": "hi"}])

        entries = [e for e in _extract_structured_log_entries(cm.output)
                   if e.get('event') == 'llm_call']
        self.assertGreaterEqual(len(entries), 2)
        last_entry = entries[-1]
        # 第二次调用后累计：prompt=20, completion=10, total=30
        self.assertEqual(last_entry['input_tokens'], 20)
        self.assertEqual(last_entry['output_tokens'], 10)
        self.assertEqual(last_entry['total_tokens'], 30)
        self.assertEqual(last_entry['calls'], 2)


@unittest.skipUnless(AI_DEPS_AVAILABLE, "httpx/tenacity not installed")
class TestLocalLLMClientStructuredLogging(unittest.TestCase):
    """缺陷 #8：LocalLLMClient.chat 结构化调用日志"""

    def _make_local_client_with_mock_model(self):
        """构造 LocalLLMClient，mock 掉 torch/transformers 依赖"""
        from tb_risk.ai.client import LocalLLMClient
        from tb_risk.ai.config import AIConfig

        cfg = AIConfig(local_model_path='/fake/model', local_device='cpu')
        client = LocalLLMClient(cfg)

        # mock tokenizer
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "fake prompt"
        mock_tokenizer.return_value = {'input_ids': mock.MagicMock(
            shape=mock.MagicMock(__getitem__=lambda self, i: 5))}

        # mock 模型
        mock_model = mock.MagicMock()
        # generate 返回一个 tensor-like 对象
        fake_output = mock.MagicMock()
        fake_output.__getitem__ = lambda self, idx: [10, 20, 30]
        mock_model.generate.return_value = fake_output
        mock_tokenizer.decode.return_value = "fake response"
        mock_tokenizer.eos_token_id = 0

        # 替换 _load_model 中的依赖
        client._model = mock_model
        client._tokenizer = mock_tokenizer

        # mock tokenizer 调用返回的 inputs
        class _FakeInputs:
            def __getitem__(self, key):
                if key == 'input_ids':
                    class _FakeTensor:
                        shape = [1, 5]
                    return _FakeTensor()
                return None
            def items(self):
                return [('input_ids', self['input_ids'])]

        mock_tokenizer.side_effect = lambda *a, **kw: _FakeInputs()
        # new_tokens.shape[0] 用于本地 usage 计算
        # outputs[0][input_len:] 返回 [10, 20, 30]，shape[0] = 3
        return client

    def test_local_chat_emits_structured_json_log(self):
        """LocalLLMClient.chat 成功后应记录结构化 JSON 日志"""
        client = self._make_local_client_with_mock_model()

        with self.assertLogs('tb_risk.ai.client', level='INFO') as cm:
            try:
                client.chat([{"role": "user", "content": "hi"}])
            except Exception:
                pass  # mock 不完整可能抛错，但日志应已记录

        entries = _extract_structured_log_entries(cm.output)
        # 本地客户端也应记录结构化日志
        llm_call_entries = [e for e in entries if e.get('event') == 'llm_call']
        self.assertTrue(llm_call_entries,
                        f"LocalLLMClient 未记录 llm_call 结构化日志: {entries}")


if __name__ == '__main__':
    unittest.main()
