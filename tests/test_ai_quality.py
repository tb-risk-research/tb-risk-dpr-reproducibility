#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — AI 数据质量诊断 (tb_risk.ai.quality)

测试 Layer 5c: 对 data_io/scoring.py 标记为低质量的记录，调用 LLM 给出修正建议。

设计：
- 注入 mock client（无需真实 API 调用）
- 验证 LLM 返回非法 JSON 时的回退路径
- 验证返回结构包含 issues/overall_quality/summary 字段
- 验证未配置/失败时降级返回 None
"""

import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class _FakeClient:
    """伪 LLM 客户端"""

    def __init__(self, chat_json_return=None, chat_json_side_effect=None,
                 chat_return=None):
        self._chat_json_return = chat_json_return
        self._chat_json_side_effect = chat_json_side_effect
        self._chat_return = chat_return
        self.chat_json_calls = []
        self.chat_calls = []

    def chat(self, messages, **kwargs):
        self.chat_calls.append({'messages': messages, 'kwargs': kwargs})
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


class TestDiagnose(unittest.TestCase):
    """ai.quality.diagnose_record() 测试"""

    def test_returns_none_when_no_client_and_unconfigured(self):
        """未注入 client 且环境未配置 → None"""
        from tb_risk.ai.quality import diagnose_record
        saved = {k: os.environ.pop(k, None) for k in [
            'TB_AI_API_KEY', 'TB_AI_LOCAL_MODEL']}
        try:
            result = diagnose_record({'age': 35, 'gender': '男'})
            self.assertIsNone(result)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_returns_none_on_empty_record(self):
        """空 record → None"""
        from tb_risk.ai.quality import diagnose_record
        client = _FakeClient(chat_json_return={'issues': []})
        self.assertIsNone(diagnose_record({}, client=client))
        self.assertIsNone(diagnose_record(None, client=client))
        self.assertEqual(len(client.chat_json_calls), 0)

    def test_returns_dict_on_success(self):
        """成功 → 返回 dict（含 issues/overall_quality/summary）"""
        from tb_risk.ai.quality import diagnose_record
        client = _FakeClient(chat_json_return={
            'issues': [
                {'field': 'age', 'problem': '缺失',
                 'suggestion': '建议填充中位数 38 岁', 'confidence': 0.8}
            ],
            'overall_quality': '中',
            'summary': '该记录年龄缺失但职业为油田工人',
        })
        result = diagnose_record({'age': None, 'occupation': '油田工人'},
                                 client=client)
        self.assertIsNotNone(result)
        self.assertIn('issues', result)
        self.assertIn('overall_quality', result)
        self.assertIn('summary', result)
        self.assertEqual(len(result['issues']), 1)
        self.assertEqual(result['issues'][0]['field'], 'age')

    def test_returns_none_when_client_raises(self):
        """client 抛异常 → None"""
        from tb_risk.ai.quality import diagnose_record
        client = _FakeClient(chat_json_side_effect=RuntimeError("network error"))
        result = diagnose_record({'age': 35}, client=client)
        self.assertIsNone(result)

    def test_returns_none_when_client_returns_none(self):
        """client 返回 None → None"""
        from tb_risk.ai.quality import diagnose_record
        client = _FakeClient(chat_json_return=None)
        result = diagnose_record({'age': 35}, client=client)
        self.assertIsNone(result)

    def test_sends_system_prompt(self):
        """调用 LLM 时发送 system + user 提示词"""
        from tb_risk.ai.quality import diagnose_record
        client = _FakeClient(chat_json_return={'issues': []})
        diagnose_record({'age': 35}, client=client)
        messages = client.chat_json_calls[0]['messages']
        roles = [m['role'] for m in messages]
        self.assertIn('system', roles)
        self.assertIn('user', roles)

    def test_user_prompt_contains_record_fields(self):
        """user 提示词包含记录的字段"""
        from tb_risk.ai.quality import diagnose_record
        client = _FakeClient(chat_json_return={'issues': []})
        diagnose_record({'age': 35, 'gender': '男'}, client=client)
        user_content = ''.join(
            m['content'] for m in client.chat_json_calls[0]['messages']
            if m['role'] == 'user'
        )
        self.assertIn('35', user_content)
        self.assertIn('男', user_content)

    def test_handles_invalid_json_response(self):
        """LLM 返回非 dict → None"""
        from tb_risk.ai.quality import diagnose_record
        client = _FakeClient(chat_json_return=["not", "a", "dict"])
        result = diagnose_record({'age': 35}, client=client)
        self.assertIsNone(result)

    def test_normalizes_response_missing_fields(self):
        """LLM 返回 dict 但缺字段 → 补默认值"""
        from tb_risk.ai.quality import diagnose_record
        client = _FakeClient(chat_json_return={'issues': []})
        result = diagnose_record({'age': 35}, client=client)
        # 缺 overall_quality/summary → 补默认值
        self.assertIn('overall_quality', result)
        self.assertIn('summary', result)
        self.assertEqual(result['issues'], [])
        # 默认 overall_quality 应为 '未知' 或类似
        self.assertIsInstance(result['overall_quality'], str)


class TestDiagnoseSchemaValidation(unittest.TestCase):
    """缺陷 #2（本轮）：diagnose_record 应使用 EXPECTED_SCHEMA 嵌套校验

    ai/client.py:_validate_schema 已支持嵌套 schema，但 quality.diagnose_record
    原先传 schema=None，导致 LLM 返回的 issues 列表内部畸形字段也能通过。
    本测试验证：完整合法响应通过严格 schema 校验；缺字段/类型错误降级到浅过滤。
    """

    def test_expected_schema_constant_exists(self):
        """quality 模块应定义 EXPECTED_SCHEMA 常量"""
        from tb_risk import ai
        from tb_risk.ai.quality import EXPECTED_SCHEMA
        self.assertIsInstance(EXPECTED_SCHEMA, dict)
        # 应包含 issues 嵌套 schema
        self.assertIn('issues', EXPECTED_SCHEMA)
        self.assertIn('overall_quality', EXPECTED_SCHEMA)
        self.assertIn('summary', EXPECTED_SCHEMA)

    def test_expected_schema_issues_is_nested_list(self):
        """EXPECTED_SCHEMA['issues'] 应为 list[dict] 形式（嵌套 schema）"""
        from tb_risk.ai.quality import EXPECTED_SCHEMA
        issues_spec = EXPECTED_SCHEMA['issues']
        self.assertIsInstance(issues_spec, list)
        self.assertGreaterEqual(len(issues_spec), 1)
        # 元素 spec 应是 dict（嵌套 schema）
        elem_spec = issues_spec[0]
        self.assertIsInstance(elem_spec, dict)
        # 嵌套 schema 应包含 field/problem/suggestion/confidence
        for field in ('field', 'problem', 'suggestion', 'confidence'):
            self.assertIn(field, elem_spec,
                          f"嵌套 schema 缺少字段: {field}")

    def test_well_formed_response_passes_schema(self):
        """完整合法响应通过 schema 校验 → 返回正常结果"""
        from tb_risk.ai.quality import diagnose_record
        client = _FakeClient(chat_json_return={
            'issues': [
                {'field': 'age', 'problem': '年龄过大',
                 'suggestion': '请核实', 'confidence': 0.9},
            ],
            'overall_quality': '低',
            'summary': '存在问题',
        })
        result = diagnose_record({'age': 150}, client=client)
        self.assertIsNotNone(result)
        self.assertEqual(len(result['issues']), 1)
        self.assertEqual(result['issues'][0]['field'], 'age')
        self.assertEqual(result['overall_quality'], '低')

    def test_issue_missing_field_falls_back_to_shallow_filter(self):
        """issue 缺字段 → schema 校验失败 → 降级到浅过滤并记录日志"""
        from tb_risk.ai.quality import diagnose_record
        # issue 缺 suggestion 字段
        client = _FakeClient(chat_json_return={
            'issues': [
                {'field': 'age', 'problem': '年龄过大',
                 'confidence': 0.9},  # 缺 suggestion
            ],
            'overall_quality': '低',
            'summary': '存在问题',
        })
        # 应记录警告日志（schema 校验失败）
        with self.assertLogs('tb_risk.ai.quality', level='WARNING') as cm:
            result = diagnose_record({'age': 150}, client=client)
        # 降级到浅过滤，仍返回结果（issue 仍是 dict）
        self.assertIsNotNone(result)
        self.assertEqual(len(result['issues']), 1)
        # 日志应提及 schema 校验失败
        joined = '\n'.join(cm.output)
        self.assertTrue(any('schema' in msg.lower() for msg in cm.output),
                        f"应记录 schema 校验失败警告: {cm.output}")

    def test_issue_wrong_type_falls_back_to_shallow_filter(self):
        """issue 字段类型错误 → schema 校验失败 → 降级到浅过滤"""
        from tb_risk.ai.quality import diagnose_record
        # confidence 应为数值，这里给字符串
        client = _FakeClient(chat_json_return={
            'issues': [
                {'field': 'age', 'problem': '年龄过大',
                 'suggestion': '请核实', 'confidence': '高'},  # 类型错误
            ],
            'overall_quality': '低',
            'summary': '存在问题',
        })
        with self.assertLogs('tb_risk.ai.quality', level='WARNING'):
            result = diagnose_record({'age': 150}, client=client)
        # 降级到浅过滤，仍返回结果
        self.assertIsNotNone(result)

    def test_empty_issues_list_passes_schema(self):
        """空 issues 列表通过 schema 校验"""
        from tb_risk.ai.quality import diagnose_record
        client = _FakeClient(chat_json_return={
            'issues': [],
            'overall_quality': '高',
            'summary': '无问题',
        })
        result = diagnose_record({'age': 35}, client=client)
        self.assertIsNotNone(result)
        self.assertEqual(result['issues'], [])
        self.assertEqual(result['overall_quality'], '高')

    def test_confidence_accepts_int_or_float(self):
        """confidence 字段接受 int 或 float"""
        from tb_risk.ai.quality import diagnose_record
        # confidence 是 int（LLM 可能返回 1 而非 1.0）
        client = _FakeClient(chat_json_return={
            'issues': [
                {'field': 'age', 'problem': '问题',
                 'suggestion': '建议', 'confidence': 1},  # int
            ],
            'overall_quality': '中',
            'summary': '',
        })
        result = diagnose_record({'age': 35}, client=client)
        self.assertIsNotNone(result)
        self.assertEqual(len(result['issues']), 1)


class TestBatchDiagnose(unittest.TestCase):
    """ai.quality.batch_diagnose() 批量诊断测试"""

    def test_returns_empty_list_on_empty_records(self):
        """空 records → 空列表"""
        from tb_risk.ai.quality import batch_diagnose
        client = _FakeClient(chat_json_return={'issues': []})
        result = batch_diagnose([], client=client)
        self.assertEqual(result, [])

    def test_returns_list_aligned_with_input(self):
        """批量诊断 → 返回列表与输入对齐（None 表示该记录无诊断）"""
        from tb_risk.ai.quality import batch_diagnose
        client = _FakeClient(chat_json_return={
            'issues': [{'field': 'age', 'problem': '缺失',
                        'suggestion': '建议填充', 'confidence': 0.8}],
            'overall_quality': '中',
            'summary': '年龄缺失',
        })
        records = [
            {'age': None, 'occupation': '油田工人'},
            {'age': 35, 'gender': '男'},
        ]
        results = batch_diagnose(records, client=client)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    def test_skips_empty_records_in_batch(self):
        """批量诊断时空记录被跳过（不调用 LLM）"""
        from tb_risk.ai.quality import batch_diagnose
        client = _FakeClient(chat_json_return={'issues': []})
        records = [{}, {'age': 35}]
        results = batch_diagnose(records, client=client)
        self.assertEqual(len(results), 2)
        # 第一个空记录应返回 None
        self.assertIsNone(results[0])
        # 只对第二个记录调用 LLM
        self.assertEqual(len(client.chat_json_calls), 1)


if __name__ == '__main__':
    unittest.main()
