#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 测试套件 — retrieve_knowledge BM25 + jieba 分词改进

缺陷 #16：retrieve_knowledge 原实现为纯关键词重叠计数，不支持同义词、
词形变化、TF-IDF 加权。改为：
- jieba 中文分词（可选依赖，不可用时回退到字符切分）
- BM25 加权排序（自带 IDF 归一化与文档长度归一化）
- keywords 字段作为额外加分（保留向后兼容）
- 中英文混合 query 支持

测试覆盖：
- 向后兼容：空 query/空 KB/无匹配仍正确返回空字符串
- BM25 排序：高相关 > 中相关 > 低相关
- keywords 加权：keywords 命中数影响排序
- top_k 限制返回数量
- jieba 可用/不可用两种模式都能工作
- 中英文混合 query
- 长文档归一化（避免长文档只因词频高就排前）
"""

import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class TestRetrieveKnowledgeBackwardCompat(unittest.TestCase):
    """retrieve_knowledge 向后兼容测试"""

    def test_empty_query_returns_empty_string(self):
        """空 query → 空字符串"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [{'content': '油田营地暴露因子 0.85', 'keywords': ['油田', '营地']}]
        self.assertEqual(retrieve_knowledge("", kb), "")
        self.assertEqual(retrieve_knowledge(None, kb), "")

    def test_empty_kb_returns_empty_string(self):
        """空知识库 → 空字符串"""
        from tb_risk.ai.qa import retrieve_knowledge
        self.assertEqual(retrieve_knowledge("油田", []), "")
        self.assertEqual(retrieve_knowledge("油田", None), "")

    def test_no_match_returns_empty_string(self):
        """无匹配条目 → 空字符串"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': '油田营地暴露因子 0.85', 'keywords': ['油田', '营地']},
        ]
        result = retrieve_knowledge("完全无关的查询xyz123", kb)
        self.assertEqual(result, "")

    def test_matched_entry_content_returned(self):
        """匹配的条目 → 返回其 content"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': '油田营地暴露因子 0.85', 'keywords': ['油田', '营地']},
            {'content': '家庭接触感染概率高', 'keywords': ['家庭', '接触']},
        ]
        result = retrieve_knowledge("油田营地暴露", kb)
        self.assertIn("油田营地", result)


class TestRetrieveKnowledgeBM25Ranking(unittest.TestCase):
    """BM25 排序测试"""

    def test_ranks_by_relevance(self):
        """高相关 > 中相关 > 低相关"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': '低相关条目', 'keywords': ['油田']},
            {'content': '高相关条目: 油田营地暴露', 'keywords': ['油田', '营地', '暴露']},
            {'content': '中相关条目', 'keywords': ['油田', '营地']},
        ]
        # 查询同时包含"油田"和"营地"
        result = retrieve_knowledge("油田营地暴露", kb)
        # 三个条目都应出现在结果中
        self.assertIn("高相关", result)
        self.assertIn("中相关", result)
        self.assertIn("低相关", result)
        # 高相关条目应排在最前（位置最小）
        self.assertLess(result.find("高相关"), result.find("中相关"))
        self.assertLess(result.find("中相关"), result.find("低相关"))

    def test_top_k_limits_returned_count(self):
        """top_k 限制返回的条目数量"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': '条目1: 油田营地', 'keywords': ['油田', '营地']},
            {'content': '条目2: 油田营地暴露', 'keywords': ['油田', '营地', '暴露']},
            {'content': '条目3: 油田', 'keywords': ['油田']},
        ]
        # top_k=1 只返回 1 条
        result = retrieve_knowledge("油田营地暴露", kb, top_k=1)
        # 只返回最相关的 1 条
        self.assertEqual(result.count("条目"), 1)
        self.assertIn("条目2", result)  # 最相关的是条目2

    def test_top_k_returns_all_when_k_exceeds_matches(self):
        """top_k 大于匹配条目数时返回所有匹配条目"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': '条目1: 油田', 'keywords': ['油田']},
            {'content': '条目2: 完全无关', 'keywords': ['无关']},
        ]
        result = retrieve_knowledge("油田", kb, top_k=10)
        # 只有 1 条匹配
        self.assertIn("条目1", result)
        self.assertNotIn("条目2", result)


class TestRetrieveKnowledgeKeywordBoost(unittest.TestCase):
    """keywords 字段加权测试"""

    def test_keywords_boost_entry_without_content_match(self):
        """keywords 命中能加分，即使 content 不含查询词"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            # 条目A: content 不含"结核"，但 keywords 含"结核"
            {'content': 'A 文档：传染病预防', 'keywords': ['结核', '预防']},
            # 条目B: content 含"结核"，但 keywords 不含
            {'content': 'B 文档：结核病诊断标准', 'keywords': ['诊断']},
        ]
        # 查询"结核"
        result = retrieve_knowledge("结核", kb, top_k=2)
        # 两个条目都应该返回（都有匹配信号）
        self.assertIn("A 文档", result)
        self.assertIn("B 文档", result)

    def test_more_keyword_matches_ranks_higher(self):
        """keywords 命中数多的条目排序更高"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            # 低 keyword 命中
            {'content': 'X 文档：基础介绍', 'keywords': ['结核']},
            # 高 keyword 命中
            {'content': 'Y 文档：扩展介绍', 'keywords': ['结核', '预防', '治疗']},
        ]
        # 查询同时包含"结核"、"预防"、"治疗"
        result = retrieve_knowledge("结核预防治疗", kb, top_k=2)
        # 两个 content 都不含完整查询词，但 Y 的 keywords 命中更多
        # Y 应该排在 X 之前
        self.assertLess(result.find("Y 文档"), result.find("X 文档"))


class TestRetrieveKnowledgeTokenizerFallback(unittest.TestCase):
    """分词器回退测试（jieba 不可用时也能工作）"""

    def test_works_without_jieba(self):
        """无 jieba 时退回到字符切分，仍能返回结果"""
        from tb_risk.ai import qa
        # 模拟 jieba 不可用
        with mock.patch.object(qa, '_JIEBA_AVAILABLE', False):
            kb = [
                {'content': '油田营地暴露', 'keywords': ['油田', '营地']},
            ]
            result = qa.retrieve_knowledge("油田", kb)
            self.assertIn("油田", result)

    def test_works_with_jieba(self):
        """jieba 可用时使用 jieba 分词"""
        try:
            import jieba  # noqa: F401
        except ImportError:
            self.skipTest("jieba not installed")
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': '结核病是由结核分枝杆菌引起的传染病', 'keywords': ['结核']},
            {'content': '糖尿病是慢性代谢性疾病', 'keywords': ['糖尿病']},
        ]
        # 用 jieba 分词，"结核病" 应能精确匹配第一条
        result = retrieve_knowledge("结核病的治疗", kb)
        self.assertIn("结核分枝杆菌", result)
        self.assertNotIn("糖尿病", result)


class TestRetrieveKnowledgeMixedLanguage(unittest.TestCase):
    """中英文混合 query 测试"""

    def test_mixed_chinese_english_query(self):
        """中英文混合查询应能匹配中英文 content"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': 'TB 风险评估模型 R0=2.5', 'keywords': ['TB', 'R0']},
            {'content': '高血压随访策略', 'keywords': ['高血压']},
        ]
        result = retrieve_knowledge("TB R0", kb)
        self.assertIn("TB", result)
        self.assertIn("R0", result)
        self.assertNotIn("高血压", result)

    def test_english_only_query(self):
        """纯英文查询也能工作"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': 'SEIR model parameters: beta, gamma', 'keywords': ['SEIR']},
            {'content': '家庭接触者风险分层', 'keywords': ['家庭']},
        ]
        result = retrieve_knowledge("SEIR model beta", kb)
        self.assertIn("SEIR", result)
        self.assertNotIn("家庭", result)


class TestRetrieveKnowledgeBM25Normalization(unittest.TestCase):
    """BM25 文档长度归一化测试"""

    def test_short_doc_with_same_tf_ranks_higher(self):
        """相同词频时短文档应排得更高（BM25 b 参数归一化）"""
        from tb_risk.ai.qa import retrieve_knowledge
        # 两个文档都包含"结核"一次
        # A 是长文档（很多其他词）
        # B 是短文档（仅几个词）
        kb = [
            {'content': 'A: 结核病是由结核分枝杆菌引起的慢性传染病，'
                        '可累及全身多个器官，以肺部感染最常见，'
                        '传播途径主要为呼吸道飞沫传播。', 'keywords': []},
            {'content': 'B: 结核简介', 'keywords': []},
        ]
        # 查询"结核"
        result = retrieve_knowledge("结核", kb, top_k=2)
        # 两个文档都应返回
        self.assertIn("A:", result)
        self.assertIn("B:", result)
        # 短文档 B 因 BM25 长度归一化应排在前面
        # （"结核"在 B 中占比更高）
        self.assertLess(result.find("B:"), result.find("A:"),
                        "短文档 B 应排在长文档 A 之前")


class TestRetrieveKnowledgeEdgeCases(unittest.TestCase):
    """边界用例测试"""

    def test_entry_without_keywords_field(self):
        """KB 条目没有 keywords 字段时仍能工作（仅靠 BM25）"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': '油田营地暴露因子 0.85'},  # 无 keywords
        ]
        result = retrieve_knowledge("油田营地", kb)
        self.assertIn("油田营地", result)

    def test_entry_with_empty_keywords(self):
        """keywords 为空列表时仍能工作"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': '油田营地暴露因子 0.85', 'keywords': []},
            {'content': '家庭接触感染', 'keywords': []},
        ]
        result = retrieve_knowledge("油田", kb)
        self.assertIn("油田", result)
        self.assertNotIn("家庭", result)

    def test_entry_without_content_field(self):
        """缺 content 字段的条目应被跳过（不抛异常）"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'keywords': ['油田']},  # 无 content
            {'content': '油田营地暴露', 'keywords': ['油田']},
        ]
        result = retrieve_knowledge("油田", kb)
        # 不应抛异常，且能返回有效条目
        self.assertIn("油田", result)

    def test_punctuation_in_query_does_not_break(self):
        """查询含标点符号应不抛异常"""
        from tb_risk.ai.qa import retrieve_knowledge
        kb = [
            {'content': '油田营地暴露', 'keywords': ['油田']},
        ]
        # 含标点的查询
        result = retrieve_knowledge("油田, 营地! 暴露?", kb)
        # 标点应被忽略或剥离，仍能匹配
        self.assertIn("油田", result)


if __name__ == '__main__':
    unittest.main()
