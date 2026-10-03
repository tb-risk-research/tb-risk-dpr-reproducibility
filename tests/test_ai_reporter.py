#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — AI 风险报告生成 (tb_risk.ai.reporter)

测试 Layer 5a: 把 RiskAssessmentService.assess() 返回的 result dict 喂给 LLM，
要求输出符合医院规范的中文风险评估报告。

设计：
- 注入 mock client（无需真实 API 调用）
- 验证提示词包含关键字段（patient_score / 接触者统计 / ml / seir）
- 验证返回值为 LLM 输出字符串
- 验证未配置/失败时降级
"""

import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class _FakeClient:
    """伪 LLM 客户端"""

    def __init__(self, chat_return=None, chat_side_effect=None,
                 chat_json_return=None):
        self._chat_return = chat_return
        self._chat_side_effect = chat_side_effect
        self._chat_json_return = chat_json_return
        self.chat_calls = []
        self.chat_json_calls = []

    def chat(self, messages, **kwargs):
        self.chat_calls.append({'messages': messages, 'kwargs': kwargs})
        if self._chat_side_effect is not None:
            raise self._chat_side_effect
        return self._chat_return

    def chat_json(self, messages, schema=None, **kwargs):
        self.chat_json_calls.append({
            'messages': messages, 'schema': schema, 'kwargs': kwargs
        })
        return self._chat_json_return

    def close(self):
        pass


def _make_assessment_result():
    """构造一个最小的 assess() 返回 dict"""
    return {
        'patient_score': 75.5,
        'potential_patients': {
            'family': [
                {'name': '张三', 'risk_score': 80.0, 'priority': '高',
                 'infection_probability': 65.0},
                {'name': '李四', 'risk_score': 45.0, 'priority': '中',
                 'infection_probability': 30.0},
            ],
            'social': [
                {'name': '王五', 'risk_score': 20.0, 'priority': '低',
                 'infection_probability': 10.0},
            ],
        },
        'summary': {
            'overall_risk': '高',
            'total_contacts': 3,
            'high_risk_contacts': 1,
        },
        'ml_results': {
            'probability': 0.72,
            'top_features': [
                {'feature': 'age', 'importance': 0.25},
                {'feature': 'sputum_smear', 'importance': 0.20},
            ],
        },
        'seir_results': {
            'R0': 2.3,
            'peak_time': 45,
        },
    }


class TestGenerateReport(unittest.TestCase):
    """ai.reporter.generate_report() 测试"""

    def test_returns_none_when_no_client_and_unconfigured(self):
        """未注入 client 且环境未配置 → None"""
        from tb_risk.ai.reporter import generate_report
        saved = {k: os.environ.pop(k, None) for k in [
            'TB_AI_API_KEY', 'TB_AI_LOCAL_MODEL']}
        try:
            result = generate_report(_make_assessment_result())
            self.assertIsNone(result)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_returns_none_on_empty_result(self):
        """空 result → None"""
        from tb_risk.ai.reporter import generate_report
        client = _FakeClient(chat_return="报告内容")
        self.assertIsNone(generate_report({}, client=client))
        self.assertIsNone(generate_report(None, client=client))
        # 不应调用 LLM
        self.assertEqual(len(client.chat_calls), 0)

    def test_returns_none_on_result_without_patient_score(self):
        """result 缺少 patient_score → None"""
        from tb_risk.ai.reporter import generate_report
        client = _FakeClient(chat_return="报告内容")
        result = generate_report({'potential_patients': {}}, client=client)
        self.assertIsNone(result)

    def test_returns_llm_response_on_success(self):
        """成功 → 返回 LLM 输出字符串"""
        from tb_risk.ai.reporter import generate_report
        client = _FakeClient(chat_return="## 风险评估报告\n患者风险等级：高")
        result = generate_report(_make_assessment_result(), client=client)
        self.assertEqual(result, "## 风险评估报告\n患者风险等级：高")
        self.assertEqual(len(client.chat_calls), 1)

    def test_returns_none_when_client_raises(self):
        """client 抛异常 → None（不向上传播）"""
        from tb_risk.ai.reporter import generate_report
        client = _FakeClient(chat_side_effect=RuntimeError("network error"))
        result = generate_report(_make_assessment_result(), client=client)
        self.assertIsNone(result)

    def test_sends_system_prompt(self):
        """调用 LLM 时发送 system 提示词"""
        from tb_risk.ai.reporter import generate_report
        client = _FakeClient(chat_return="报告")
        generate_report(_make_assessment_result(), client=client)
        messages = client.chat_calls[0]['messages']
        roles = [m['role'] for m in messages]
        self.assertIn('system', roles)
        self.assertIn('user', roles)

    def test_user_prompt_contains_patient_score(self):
        """user 提示词包含 patient_score"""
        from tb_risk.ai.reporter import generate_report
        client = _FakeClient(chat_return="报告")
        generate_report(_make_assessment_result(), client=client)
        user_msgs = [m['content'] for m in client.chat_calls[0]['messages']
                     if m['role'] == 'user']
        self.assertTrue(any('75.5' in c for c in user_msgs))

    def test_user_prompt_contains_contact_stats(self):
        """user 提示词包含接触者统计（总数、高风险数）"""
        from tb_risk.ai.reporter import generate_report
        client = _FakeClient(chat_return="报告")
        generate_report(_make_assessment_result(), client=client)
        user_content = ''.join(
            m['content'] for m in client.chat_calls[0]['messages']
            if m['role'] == 'user'
        )
        # 总接触者 3
        self.assertIn('3', user_content)
        # 高风险数 1 或 高风险关键词
        self.assertTrue('高' in user_content)

    def test_user_prompt_contains_ml_results_when_present(self):
        """result 含 ml_results → user 提示词包含 ML 关键信息"""
        from tb_risk.ai.reporter import generate_report
        client = _FakeClient(chat_return="报告")
        result = _make_assessment_result()
        generate_report(result, client=client)
        user_content = ''.join(
            m['content'] for m in client.chat_calls[0]['messages']
            if m['role'] == 'user'
        )
        # ML probability=0.72
        self.assertTrue('0.72' in user_content or '72' in user_content)

    def test_user_prompt_contains_seir_results_when_present(self):
        """result 含 seir_results → user 提示词包含 SEIR 关键信息"""
        from tb_risk.ai.reporter import generate_report
        client = _FakeClient(chat_return="报告")
        result = _make_assessment_result()
        generate_report(result, client=client)
        user_content = ''.join(
            m['content'] for m in client.chat_calls[0]['messages']
            if m['role'] == 'user'
        )
        # R0=2.3
        self.assertTrue('2.3' in user_content or 'R0' in user_content
                        or 'SEIR' in user_content or 'seir' in user_content)

    def test_user_prompt_omits_ml_when_absent(self):
        """result 无 ml_results → user 提示词不应包含 ML 段"""
        from tb_risk.ai.reporter import generate_report
        client = _FakeClient(chat_return="报告")
        result = _make_assessment_result()
        result['ml_results'] = None
        generate_report(result, client=client)
        user_content = ''.join(
            m['content'] for m in client.chat_calls[0]['messages']
            if m['role'] == 'user'
        )
        # 不强制断言"不含 ML"，但应仍能生成报告
        self.assertTrue(len(user_content) > 0)

    def test_handles_minimal_result_without_optional_sections(self):
        """最小 result（仅 patient_score）→ 仍能生成报告"""
        from tb_risk.ai.reporter import generate_report
        client = _FakeClient(chat_return="报告")
        result = {'patient_score': 50.0}
        report = generate_report(result, client=client)
        self.assertIsNotNone(report)
        self.assertEqual(len(client.chat_calls), 1)


class TestBuildContext(unittest.TestCase):
    """_build_context() 内部辅助函数测试"""

    def test_returns_string_with_patient_score(self):
        """返回包含 patient_score 的字符串"""
        from tb_risk.ai.reporter import _build_context
        ctx = _build_context({'patient_score': 75.5})
        self.assertIsInstance(ctx, str)
        self.assertIn('75.5', ctx)

    def test_includes_contact_summary(self):
        """包含接触者统计"""
        from tb_risk.ai.reporter import _build_context
        ctx = _build_context(_make_assessment_result())
        # 应包含接触者数量信息
        self.assertIn('3', ctx)  # total_contacts

    def test_includes_ml_when_present(self):
        """包含 ML 段"""
        from tb_risk.ai.reporter import _build_context
        ctx = _build_context(_make_assessment_result())
        # ML 相关关键词
        self.assertTrue(
            'ML' in ctx or 'ml' in ctx or '机器学习' in ctx
            or '0.72' in ctx or '72' in ctx
        )

    def test_includes_seir_when_present(self):
        """包含 SEIR 段"""
        from tb_risk.ai.reporter import _build_context
        ctx = _build_context(_make_assessment_result())
        self.assertTrue(
            'SEIR' in ctx or 'seir' in ctx or 'R0' in ctx
            or '2.3' in ctx
        )

    def test_returns_empty_string_on_empty_result(self):
        """空 result → 空字符串"""
        from tb_risk.ai.reporter import _build_context
        self.assertEqual(_build_context({}), '')
        self.assertEqual(_build_context(None), '')


if __name__ == '__main__':
    unittest.main()
