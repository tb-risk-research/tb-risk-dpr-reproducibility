#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — AI 文本抽取模块 (tb_risk.ai.parser) + data_io.text_parser 集成

测试 Layer 4: LLM 文本→结构化数据抽取，填充 data_io.text_parser.extract_fields_with_llm 占位接口。

设计：
- 注入 mock client（无需真实 API 调用）
- 验证 LLM 返回的 records 列表经 normalize 后：
  - 字段过滤（仅保留 KNOWN_FIELDS）
  - 类型转换（int/float/str）
  - 元数据 _confidence + _source={'source':'llm'}
- 集成测试：data_io.text_parser.extract_fields_with_llm 现在真正调用 ai.parser
"""

import json
import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class _FakeClient:
    """伪 LLM 客户端，按预设返回值响应"""

    def __init__(self, chat_json_return=None, chat_json_side_effect=None,
                 chat_return=None, chat_side_effect=None):
        self._chat_json_return = chat_json_return
        self._chat_json_side_effect = chat_json_side_effect
        self._chat_return = chat_return
        self._chat_side_effect = chat_side_effect
        self.chat_json_calls = []
        self.chat_calls = []

    def chat(self, messages, **kwargs):
        self.chat_calls.append({'messages': messages, 'kwargs': kwargs})
        if self._chat_side_effect is not None:
            raise self._chat_side_effect
        return self._chat_return

    def chat_json(self, messages, schema=None, **kwargs):
        self.chat_json_calls.append({
            'messages': messages, 'schema': schema, 'kwargs': kwargs
        })
        if self._chat_json_side_effect is not None:
            raise self._chat_json_side_effect
        return self._chat_json_return

    def close(self):
        pass


class TestExtractWithLlm(unittest.TestCase):
    """ai.parser.extract_with_llm() 测试"""

    def test_returns_none_when_no_client_and_unconfigured(self):
        """未注入 client 且环境未配置 → None（调用方降级到正则）"""
        from tb_risk.ai.parser import extract_with_llm
        # 清空环境
        saved = {k: os.environ.pop(k, None) for k in [
            'TB_AI_API_KEY', 'TB_AI_LOCAL_MODEL']}
        try:
            result = extract_with_llm("患者男，35岁，痰涂片阳性")
            self.assertIsNone(result)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_returns_none_on_empty_text(self):
        """空文本 → None"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return={"records": []})
        result = extract_with_llm("", client=client)
        self.assertIsNone(result)
        # 不应调用 LLM
        self.assertEqual(len(client.chat_json_calls), 0)

    def test_returns_records_on_success(self):
        """成功抽取 → list[dict]"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return={
            "records": [
                {"age": 35, "sputum_smear": 1, "has_cavity": 0},
            ]
        })
        result = extract_with_llm("患者男，35岁，痰涂片阳性", client=client)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['age'], 35)
        self.assertEqual(result[0]['sputum_smear'], 1)

    def test_records_have_confidence_and_source_metadata(self):
        """每条记录含 _confidence (dict) + _source={'source':'llm'}"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return={
            "records": [{"age": 35}]
        })
        result = extract_with_llm("患者35岁", client=client)
        self.assertEqual(result[0]['_source'], {'source': 'llm'})
        self.assertIn('age', result[0]['_confidence'])
        self.assertIsInstance(result[0]['_confidence']['age'], float)

    def test_filters_unknown_fields(self):
        """未知字段被过滤掉（仅保留 KNOWN_FIELDS）"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return={
            "records": [
                {"age": 35, "random_unknown_field": "should be dropped"}
            ]
        })
        result = extract_with_llm("患者35岁", client=client)
        self.assertIn('age', result[0])
        self.assertNotIn('random_unknown_field', result[0])

    def test_type_coercion_int_field(self):
        """int 字段：字符串 '35' → 35"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return={
            "records": [{"age": "35", "cough_freq": "5"}]
        })
        result = extract_with_llm("患者35岁", client=client)
        self.assertEqual(result[0]['age'], 35)
        self.assertIsInstance(result[0]['age'], int)
        self.assertEqual(result[0]['cough_freq'], 5)

    def test_type_coercion_float_field(self):
        """float 字段：字符串 '70.5' → 70.5"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return={
            "records": [{"bmi": "22.5", "weight": "70.5"}]
        })
        result = extract_with_llm("BMI 22.5", client=client)
        self.assertAlmostEqual(result[0]['bmi'], 22.5)
        self.assertIsInstance(result[0]['bmi'], float)
        self.assertAlmostEqual(result[0]['weight'], 70.5)

    def test_type_coercion_failure_drops_field(self):
        """类型转换失败的字段被丢弃（不抛出）"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return={
            "records": [{"age": "not-a-number", "gender": "男"}]
        })
        result = extract_with_llm("患者男", client=client)
        # age 转换失败被丢弃
        self.assertNotIn('age', result[0])
        # gender 正常保留
        self.assertEqual(result[0]['gender'], '男')

    def test_returns_none_when_client_returns_none(self):
        """client 返回 None → None"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return=None)
        result = extract_with_llm("患者35岁", client=client)
        self.assertIsNone(result)

    def test_returns_none_when_client_raises(self):
        """client 抛异常 → None（不向上传播）"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_side_effect=RuntimeError("network error"))
        result = extract_with_llm("患者35岁", client=client)
        self.assertIsNone(result)

    def test_multiple_records(self):
        """多患者文本 → 多条记录"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return={
            "records": [
                {"age": 35, "gender": "男"},
                {"age": 42, "gender": "女"},
            ]
        })
        result = extract_with_llm("两个患者", client=client)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['age'], 35)
        self.assertEqual(result[1]['age'], 42)

    def test_single_dict_wrapped_to_list(self):
        """LLM 返回单 dict（非 records 包装）→ 包装为 [dict]"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return={
            "age": 35, "gender": "男"  # 直接返回 dict，无 records 包装
        })
        result = extract_with_llm("患者35岁", client=client)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['age'], 35)

    def test_returns_none_when_no_valid_records(self):
        """所有记录都无已知字段 → None"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return={
            "records": [
                {"unknown_field_1": "x"},
                {"unknown_field_2": "y"},
            ]
        })
        result = extract_with_llm("无意义文本", client=client)
        self.assertIsNone(result)

    def test_sends_system_prompt_to_client(self):
        """调用 LLM 时发送 system 提示词"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return={"records": [{"age": 35}]})
        extract_with_llm("患者35岁", client=client)
        self.assertEqual(len(client.chat_json_calls), 1)
        messages = client.chat_json_calls[0]['messages']
        # 应有 system + user 至少 2 条消息
        self.assertGreaterEqual(len(messages), 2)
        roles = [m['role'] for m in messages]
        self.assertIn('system', roles)
        self.assertIn('user', roles)

    def test_user_message_contains_input_text(self):
        """user 消息包含输入文本"""
        from tb_risk.ai.parser import extract_with_llm
        client = _FakeClient(chat_json_return={"records": [{"age": 35}]})
        extract_with_llm("患者男，年龄35岁，痰涂片阳性", client=client)
        user_msgs = [m for m in client.chat_json_calls[0]['messages']
                     if m['role'] == 'user']
        self.assertTrue(any("35岁" in m['content'] for m in user_msgs))


class TestExtractWithLlmLocalModeSkipRedact(unittest.TestCase):
    """缺陷 #1（本轮）：本地模式跳过脱敏一致性测试

    parser.extract_with_llm 应与 reporter.generate_report / qa.ask 保持一致：
    根据 effective_config.is_local 计算 skip_redact，传给 redact_text。
    """

    def test_cloud_mode_redacts_pii_in_text(self):
        """云端模式（is_local=False）→ 文本中的 PII 应被脱敏"""
        from tb_risk.ai.parser import extract_with_llm
        from tb_risk.ai.config import AIConfig
        # 云端配置
        cfg = AIConfig(api_key='sk-test')
        self.assertFalse(cfg.is_local)
        # 含手机号的文本
        text_with_pii = "患者男，35岁，联系电话 13800138000，痰涂片阳性"
        client = _FakeClient(chat_json_return={"records": [{"age": 35}]})
        extract_with_llm(text_with_pii, client=client, config=cfg)
        user_msgs = [m for m in client.chat_json_calls[0]['messages']
                     if m['role'] == 'user']
        # 手机号应被脱敏，不在 user 消息中
        self.assertTrue(any("13800138000" not in m['content'] for m in user_msgs),
                        "云端模式下手机号应被脱敏")

    def test_local_mode_preserves_pii_in_text(self):
        """本地模式（is_local=True）→ 文本原文保留（不脱敏）"""
        from tb_risk.ai.parser import extract_with_llm
        from tb_risk.ai.config import AIConfig
        # 本地配置
        cfg = AIConfig(local_model_path='/fake/local/model')
        self.assertTrue(cfg.is_local)
        # 含手机号的文本
        text_with_pii = "患者男，35岁，联系电话 13800138000，痰涂片阳性"
        client = _FakeClient(chat_json_return={"records": [{"age": 35}]})
        extract_with_llm(text_with_pii, client=client, config=cfg)
        user_msgs = [m for m in client.chat_json_calls[0]['messages']
                     if m['role'] == 'user']
        # 本地模式应保留原文，手机号仍在
        self.assertTrue(any("13800138000" in m['content'] for m in user_msgs),
                        "本地模式下手机号应保留原文")

    def test_local_mode_skip_redact_via_client_config(self):
        """config=None 时从 client.config 获取 is_local"""
        from tb_risk.ai.parser import extract_with_llm
        from tb_risk.ai.config import AIConfig
        # 构造一个本地模式的 client（不传 config）
        cfg = AIConfig(local_model_path='/fake/local/model')
        client = _FakeClient(chat_json_return={"records": [{"age": 35}]})
        client.config = cfg
        text_with_pii = "身份证号 110101199001011234，痰涂片阳性"
        extract_with_llm(text_with_pii, client=client)  # 不传 config
        user_msgs = [m for m in client.chat_json_calls[0]['messages']
                     if m['role'] == 'user']
        # 本地模式应保留身份证号
        self.assertTrue(any("110101199001011234" in m['content'] for m in user_msgs),
                        "从 client.config 获取 is_local=True 时应保留原文")

    def test_cloud_mode_redacts_id_card(self):
        """云端模式：身份证号应被脱敏"""
        from tb_risk.ai.parser import extract_with_llm
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test')
        text_with_pii = "身份证号 110101199001011234，痰涂片阳性"
        client = _FakeClient(chat_json_return={"records": [{"age": 35}]})
        extract_with_llm(text_with_pii, client=client, config=cfg)
        user_msgs = [m for m in client.chat_json_calls[0]['messages']
                     if m['role'] == 'user']
        # 身份证号应被脱敏
        self.assertTrue(all("110101199001011234" not in m['content']
                            for m in user_msgs),
                        "云端模式下身份证号应被脱敏")


class TestKnownFields(unittest.TestCase):
    """KNOWN_FIELDS 字段表测试"""

    def test_includes_core_clinical_fields(self):
        """核心临床字段都在表中"""
        from tb_risk.ai.parser import KNOWN_FIELDS
        for field in ('age', 'sputum_smear', 'has_cavity', 'gender',
                      'treatment', 'ventilation'):
            self.assertIn(field, KNOWN_FIELDS, f"缺失字段: {field}")

    def test_field_types_are_correct(self):
        """字段类型映射正确"""
        from tb_risk.ai.parser import KNOWN_FIELDS
        self.assertEqual(KNOWN_FIELDS['age'], int)
        self.assertEqual(KNOWN_FIELDS['sputum_smear'], int)
        self.assertEqual(KNOWN_FIELDS['bmi'], float)
        self.assertEqual(KNOWN_FIELDS['gender'], str)
        self.assertEqual(KNOWN_FIELDS['ventilation'], str)


class TestTextParserIntegration(unittest.TestCase):
    """data_io.text_parser.extract_fields_with_llm 集成测试"""

    def test_returns_none_when_ai_unconfigured(self):
        """AI 未配置时返回 None（保留原占位行为，但无 UserWarning）"""
        from tb_risk.data_io.text_parser import extract_fields_with_llm
        saved = {k: os.environ.pop(k, None) for k in [
            'TB_AI_API_KEY', 'TB_AI_LOCAL_MODEL']}
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # 任何警告转为异常
                result = extract_fields_with_llm("患者35岁")
            self.assertIsNone(result)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_delegates_to_ai_parser_when_configured(self):
        """AI 配置后真正调用 ai.parser.extract_with_llm"""
        from tb_risk.data_io import text_parser
        # mock ai.parser.extract_with_llm
        with mock.patch('tb_risk.ai.parser.extract_with_llm') as mock_extract:
            mock_extract.return_value = [{'age': 35, '_source': {'source': 'llm'}}]
            result = text_parser.extract_fields_with_llm("患者35岁")
            self.assertIsNotNone(result)
            self.assertEqual(result[0]['age'], 35)
            mock_extract.assert_called_once()

    def test_model_path_param_triggers_local_mode(self):
        """model_path 参数触发本地模式"""
        from tb_risk.data_io import text_parser
        with mock.patch('tb_risk.ai.parser.extract_with_llm') as mock_extract:
            mock_extract.return_value = None
            text_parser.extract_fields_with_llm("患者35岁", model_path='/path/to/model')
            # 应调用 ai.parser.extract_with_llm
            mock_extract.assert_called_once()
            # 检查调用参数：应传入 config 且 config.is_local=True
            args, kwargs = mock_extract.call_args
            # extract_with_llm 签名: (text, client=None, config=None)
            # text_parser 应构造 config 并传入
            config = kwargs.get('config') or (args[2] if len(args) > 2 else None)
            self.assertIsNotNone(config)
            self.assertTrue(config.is_local)
            self.assertEqual(config.local_model_path, '/path/to/model')


class TestParseTextInputIntegration(unittest.TestCase):
    """parse_text_input(use_llm=True) 集成测试"""

    def test_use_llm_falls_back_to_regex_when_unconfigured(self):
        """use_llm=True 但 AI 未配置 → 回退正则，records 仍有"""
        from tb_risk.data_io.text_parser import parse_text_input
        saved = {k: os.environ.pop(k, None) for k in [
            'TB_AI_API_KEY', 'TB_AI_LOCAL_MODEL']}
        try:
            text = "患者男，年龄35岁，痰涂片阳性，有空洞，已治疗"
            records, meta = parse_text_input(text, use_llm=True)
            # 应该有正则抽取的记录
            self.assertIsInstance(records, list)
            self.assertGreater(len(records), 0)
            # llm_used 应为 False
            self.assertFalse(meta['llm_used'])
            self.assertEqual(meta['llm_records'], 0)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_use_llm_uses_llm_when_configured(self):
        """use_llm=True 且 AI 配置 → 真正使用 LLM"""
        from tb_risk.data_io.text_parser import parse_text_input
        with mock.patch('tb_risk.ai.parser.extract_with_llm') as mock_extract:
            mock_extract.return_value = [
                {'age': 35, 'sputum_smear': 1, '_source': {'source': 'llm'}},
            ]
            # 注入 fake api_key 到环境
            saved_key = os.environ.pop('TB_AI_API_KEY', None)
            os.environ['TB_AI_API_KEY'] = 'sk-test'
            try:
                text = "患者男，年龄35岁，痰涂片阳性"
                records, meta = parse_text_input(text, use_llm=True)
                self.assertTrue(meta['llm_used'])
                self.assertEqual(meta['llm_records'], 1)
                mock_extract.assert_called_once()
            finally:
                os.environ.pop('TB_AI_API_KEY', None)
                if saved_key is not None:
                    os.environ['TB_AI_API_KEY'] = saved_key


if __name__ == '__main__':
    unittest.main()
