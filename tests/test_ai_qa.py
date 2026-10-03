#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — AI 智能问答 (tb_risk.ai.qa)

测试 Layer 5b: RAG 模式智能问答。
- 输入：用户问题 + 当前评估结果上下文 + 可选知识库片段
- 输出：基于上下文和知识的 LLM 回答

设计：
- 注入 mock client（无需真实 API 调用）
- 验证提示词包含 question / context / knowledge
- 验证简单 RAG: retrieve_knowledge() 按关键词相关度排序
- 验证降级路径
"""

import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class _FakeClient:
    """伪 LLM 客户端"""

    def __init__(self, chat_return=None, chat_side_effect=None):
        self._chat_return = chat_return
        self._chat_side_effect = chat_side_effect
        self.chat_calls = []

    def chat(self, messages, **kwargs):
        self.chat_calls.append({'messages': messages, 'kwargs': kwargs})
        if self._chat_side_effect is not None:
            raise self._chat_side_effect
        return self._chat_return

    def chat_json(self, messages, schema=None, **kwargs):
        return None

    def close(self):
        pass


class TestAsk(unittest.TestCase):
    """ai.qa.ask() 测试"""

    def test_returns_none_when_no_client_and_unconfigured(self):
        """未注入 client 且环境未配置 → None"""
        from tb_risk.ai.qa import ask
        saved = {k: os.environ.pop(k, None) for k in [
            'TB_AI_API_KEY', 'TB_AI_LOCAL_MODEL']}
        try:
            result = ask("为什么该接触者风险高?",
                         context={'patient_score': 75.0})
            self.assertIsNone(result)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_returns_none_on_empty_question(self):
        """空 question → None"""
        from tb_risk.ai.qa import ask
        client = _FakeClient(chat_return="回答")
        self.assertIsNone(ask("", context={'patient_score': 75.0}, client=client))
        self.assertIsNone(ask(None, context={'patient_score': 75.0}, client=client))
        self.assertEqual(len(client.chat_calls), 0)

    def test_returns_llm_response_on_success(self):
        """成功 → 返回 LLM 输出字符串"""
        from tb_risk.ai.qa import ask
        client = _FakeClient(chat_return="该接触者风险高是因为...")
        result = ask("为什么风险高?", context={'patient_score': 75.0}, client=client)
        self.assertEqual(result, "该接触者风险高是因为...")
        self.assertEqual(len(client.chat_calls), 1)

    def test_returns_none_when_client_raises(self):
        """client 抛异常 → None"""
        from tb_risk.ai.qa import ask
        client = _FakeClient(chat_side_effect=RuntimeError("network error"))
        result = ask("为什么?", context={'patient_score': 75.0}, client=client)
        self.assertIsNone(result)

    def test_sends_system_prompt(self):
        """调用 LLM 时发送 system 提示词"""
        from tb_risk.ai.qa import ask
        client = _FakeClient(chat_return="回答")
        ask("问题?", context={'patient_score': 75.0}, client=client)
        messages = client.chat_calls[0]['messages']
        roles = [m['role'] for m in messages]
        self.assertIn('system', roles)
        self.assertIn('user', roles)

    def test_user_prompt_contains_question(self):
        """user 提示词包含 question"""
        from tb_risk.ai.qa import ask
        client = _FakeClient(chat_return="回答")
        ask("为什么该接触者风险高?", context={'patient_score': 75.0}, client=client)
        user_content = ''.join(
            m['content'] for m in client.chat_calls[0]['messages']
            if m['role'] == 'user'
        )
        self.assertIn("为什么该接触者风险高?", user_content)

    def test_user_prompt_contains_context(self):
        """user 提示词包含 context 关键字段"""
        from tb_risk.ai.qa import ask
        client = _FakeClient(chat_return="回答")
        context = {'patient_score': 75.5, 'summary': {'overall_risk': '高'}}
        ask("问题?", context=context, client=client)
        user_content = ''.join(
            m['content'] for m in client.chat_calls[0]['messages']
            if m['role'] == 'user'
        )
        self.assertIn('75.5', user_content)

    def test_user_prompt_contains_knowledge_when_provided(self):
        """knowledge 提供时 → user 提示词包含 knowledge"""
        from tb_risk.ai.qa import ask
        client = _FakeClient(chat_return="回答")
        knowledge = "油田营地暴露因子为 0.85，高于一般场所。"
        ask("问题?", context={'patient_score': 75.0},
            knowledge=knowledge, client=client)
        user_content = ''.join(
            m['content'] for m in client.chat_calls[0]['messages']
            if m['role'] == 'user'
        )
        self.assertIn("油田营地", user_content)
        self.assertIn("0.85", user_content)

    def test_works_without_knowledge(self):
        """knowledge=None → 仍能正常工作"""
        from tb_risk.ai.qa import ask
        client = _FakeClient(chat_return="回答")
        result = ask("问题?", context={'patient_score': 75.0},
                     knowledge=None, client=client)
        self.assertIsNotNone(result)
        self.assertEqual(len(client.chat_calls), 1)


class TestRetrieveKnowledge(unittest.TestCase):
    """retrieve_knowledge() 简单 RAG 检索测试"""

    def test_returns_empty_string_on_empty_query(self):
        """空 query → 空字符串"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [{'content': '油田营地暴露因子 0.85', 'keywords': ['油田', '营地']}]
        self.assertEqual(retrieve_knowledge("", kb), "")
        self.assertEqual(retrieve_knowledge(None, kb), "")

    def test_returns_empty_string_on_empty_kb(self):
        """空知识库 → 空字符串"""
        from tb_risk.ai.qa import retrieve_knowledge
        self.assertEqual(retrieve_knowledge("油田", []), "")
        self.assertEqual(retrieve_knowledge("油田", None), "")

    def test_returns_matching_entry_content(self):
        """匹配的条目 → 返回其 content"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': '油田营地暴露因子 0.85', 'keywords': ['油田', '营地']},
            {'content': '家庭接触感染概率高', 'keywords': ['家庭', '接触']},
        ]
        result = retrieve_knowledge("油田营地暴露", kb)
        self.assertIn("油田营地", result)

    def test_returns_empty_string_when_no_match(self):
        """无匹配条目 → 空字符串"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': '油田营地暴露因子 0.85', 'keywords': ['油田', '营地']},
        ]
        result = retrieve_knowledge("完全无关的查询xyz123", kb)
        self.assertEqual(result, "")

    def test_ranks_by_keyword_overlap(self):
        """多条匹配时按关键词重叠数排序"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': '低相关条目', 'keywords': ['油田']},
            {'content': '高相关条目: 油田营地暴露', 'keywords': ['油田', '营地', '暴露']},
            {'content': '中相关条目', 'keywords': ['油田', '营地']},
        ]
        # 查询同时包含"油田"和"营地"，高相关条目（overlap=3）应排在中相关（overlap=2）和低相关（overlap=1）之前
        result = retrieve_knowledge("油田营地暴露", kb)
        # 三个条目都应出现在结果中
        self.assertIn("高相关", result)
        self.assertIn("中相关", result)
        self.assertIn("低相关", result)
        # 高相关条目应排在最前（位置最小）
        self.assertLess(result.find("高相关"), result.find("中相关"))
        self.assertLess(result.find("中相关"), result.find("低相关"))


class TestBuildContextForQa(unittest.TestCase):
    """_build_context_for_qa() 内部辅助函数测试"""

    def test_returns_string_with_patient_score(self):
        """返回包含 patient_score 的字符串"""
        from tb_risk.ai.qa import _build_context_for_qa
        ctx = _build_context_for_qa({'patient_score': 75.5})
        self.assertIsInstance(ctx, str)
        self.assertIn('75.5', ctx)

    def test_returns_empty_string_on_empty_context(self):
        """空 context → 空字符串"""
        from tb_risk.ai.qa import _build_context_for_qa
        self.assertEqual(_build_context_for_qa({}), '')
        self.assertEqual(_build_context_for_qa(None), '')

    def test_includes_summary_when_present(self):
        """包含 summary 段"""
        from tb_risk.ai.qa import _build_context_for_qa
        ctx = _build_context_for_qa({
            'patient_score': 75.0,
            'summary': {'overall_risk': '高', 'total_contacts': 3},
        })
        self.assertIn('高', ctx)
        self.assertIn('3', ctx)


if __name__ == '__main__':
    unittest.main()
