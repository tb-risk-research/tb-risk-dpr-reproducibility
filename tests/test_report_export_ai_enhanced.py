#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 测试套件 — report_export AI 增强报告测试

缺陷 #14 修复：在 export_utils/report_export.py 的 generate_assessment_report 中
新增 ai_enhanced 参数。当 ai_enhanced=True 且 AI 已配置时，先调用
ai.reporter.generate_report() 生成自然语言摘要，再拼接到模板报告头部。
AI 未配置或失败时降级到模板报告（不抛异常）。

测试设计：
- mock ai.reporter.generate_report 与 ai.client.make_client
- 验证 ai_enhanced=False（默认）行为不变
- 验证 ai_enhanced=True 时 AI 摘要被拼接到头部
- 验证 AI 未配置/失败时优雅降级
- 覆盖 html / text / json 三种 format
"""

import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def _sample_results():
    """构造测试用评估结果 dict"""
    return {
        'family_members': [{'name': '张三', 'age': 35}],
        'social_contacts': [],
        'patient_info': {'sputum_smear': 1},
        'results': {
            'overall_risk': '高',
            'base_infection_probability': 75.5,
            'patient_score': 75.0,
            'summary': {'total_contacts': 1, 'high_risk_contacts': 1},
        },
    }


class TestGenerateAssessmentReportAiEnhancedDisabled(unittest.TestCase):
    """ai_enhanced=False（默认）行为应与原来完全一致"""

    def test_default_param_is_false(self):
        """未传 ai_enhanced 时默认 False，不调用 AI"""
        from tb_risk.export_utils.report_export import generate_assessment_report
        results = _sample_results()
        with mock.patch('tb_risk.ai.reporter.generate_report') as m:
            report = generate_assessment_report(results, format='text')
        self.assertIsNotNone(report)
        # AI 不应被调用
        m.assert_not_called()

    def test_explicit_false_does_not_call_ai(self):
        """显式 ai_enhanced=False → 不调用 AI"""
        from tb_risk.export_utils.report_export import generate_assessment_report
        results = _sample_results()
        with mock.patch('tb_risk.ai.reporter.generate_report') as m:
            report = generate_assessment_report(results, format='text',
                                                ai_enhanced=False)
        self.assertIsNotNone(report)
        m.assert_not_called()

    def test_text_format_unchanged_when_disabled(self):
        """text 格式在 ai_enhanced=False 时内容与原来一致"""
        from tb_risk.export_utils.report_export import generate_assessment_report
        results = _sample_results()
        report = generate_assessment_report(results, format='text',
                                            ai_enhanced=False)
        # 应包含模板报告关键字
        self.assertIn('TB Risk Assessment Report', report)
        self.assertIn('Risk Level: 高', report)


class TestGenerateAssessmentReportAiEnhancedEnabled(unittest.TestCase):
    """ai_enhanced=True 时应集成 AI 摘要"""

    def test_ai_summary_prepended_to_text_report(self):
        """ai_enhanced=True + AI 成功 → 摘要拼接到报告头部"""
        from tb_risk.export_utils.report_export import generate_assessment_report
        results = _sample_results()
        ai_summary = "## AI 风险摘要\n患者属于高风险人群，建议立即隔离。"
        with mock.patch('tb_risk.ai.reporter.generate_report',
                        return_value=ai_summary) as m:
            report = generate_assessment_report(results, format='text',
                                                ai_enhanced=True)
        # AI 应被调用，传入评估结果 dict
        m.assert_called_once()
        call_args = m.call_args
        result_arg = call_args[0][0] if call_args[0] else call_args[1].get('result')
        # 应传入包含 patient_score 的 result（即 results['results']）
        self.assertEqual(result_arg.get('patient_score'), 75.0)
        # 报告头部应包含 AI 摘要
        self.assertIn('AI 风险摘要', report)
        self.assertIn('高风险人群', report)
        # 报告尾部应保留模板内容
        self.assertIn('TB Risk Assessment Report', report)

    def test_ai_summary_prepended_to_html_report(self):
        """ai_enhanced=True + AI 成功 → HTML 报告头部包含 AI 摘要"""
        from tb_risk.export_utils.report_export import generate_assessment_report
        results = _sample_results()
        ai_summary = "## AI 风险摘要\n高风险，建议隔离。"
        with mock.patch('tb_risk.ai.reporter.generate_report',
                        return_value=ai_summary):
            report = generate_assessment_report(results, format='html',
                                                ai_enhanced=True)
        self.assertIn('AI 风险摘要', report)
        self.assertIn('TB Risk Assessment Report', report)

    def test_ai_failure_falls_back_to_template_only(self):
        """ai_enhanced=True + AI 返回 None → 降级到模板报告，不抛异常"""
        from tb_risk.export_utils.report_export import generate_assessment_report
        results = _sample_results()
        with mock.patch('tb_risk.ai.reporter.generate_report',
                        return_value=None):
            # 不应抛异常
            report = generate_assessment_report(results, format='text',
                                                ai_enhanced=True)
        # 仍应有模板内容
        self.assertIn('TB Risk Assessment Report', report)
        # 不应包含 AI 摘要相关字段
        self.assertNotIn('AI 风险摘要', report)

    def test_ai_exception_falls_back_to_template_only(self):
        """ai_enhanced=True + AI 抛异常 → 降级到模板报告，不抛异常"""
        from tb_risk.export_utils.report_export import generate_assessment_report
        results = _sample_results()
        with mock.patch('tb_risk.ai.reporter.generate_report',
                        side_effect=Exception("网络错误")):
            # 不应抛异常
            report = generate_assessment_report(results, format='text',
                                                ai_enhanced=True)
        # 仍应有模板内容
        self.assertIn('TB Risk Assessment Report', report)

    def test_ai_summary_section_has_clear_delimiter(self):
        """AI 摘要区与模板报告之间应有清晰分隔（标题/分隔线）"""
        from tb_risk.export_utils.report_export import generate_assessment_report
        results = _sample_results()
        ai_summary = "AI 摘要内容"
        with mock.patch('tb_risk.ai.reporter.generate_report',
                        return_value=ai_summary):
            report = generate_assessment_report(results, format='text',
                                                ai_enhanced=True)
        # 应有某种分隔标记（"AI 摘要" / "===" / "---" 之一）
        self.assertTrue(
            'AI' in report and '摘要' in report,
            "AI 摘要区应有清晰标题"
        )

    def test_json_format_with_ai_enhanced_includes_summary_field(self):
        """JSON 格式 + ai_enhanced=True → JSON 中包含 ai_summary 字段"""
        import json as _json
        from tb_risk.export_utils.report_export import generate_assessment_report
        results = _sample_results()
        ai_summary = "AI 风险摘要内容"
        with mock.patch('tb_risk.ai.reporter.generate_report',
                        return_value=ai_summary):
            report = generate_assessment_report(results, format='json',
                                                ai_enhanced=True)
        data = _json.loads(report)
        self.assertIn('ai_summary', data)
        self.assertEqual(data['ai_summary'], ai_summary)

    def test_json_format_without_ai_enhanced_has_no_summary_field(self):
        """JSON 格式 + ai_enhanced=False → JSON 中不包含 ai_summary 字段"""
        import json as _json
        from tb_risk.export_utils.report_export import generate_assessment_report
        results = _sample_results()
        report = generate_assessment_report(results, format='json',
                                            ai_enhanced=False)
        data = _json.loads(report)
        self.assertNotIn('ai_summary', data)

    def test_json_format_ai_failure_omits_summary_field(self):
        """JSON 格式 + AI 失败 → JSON 中不包含 ai_summary 字段"""
        import json as _json
        from tb_risk.export_utils.report_export import generate_assessment_report
        results = _sample_results()
        with mock.patch('tb_risk.ai.reporter.generate_report',
                        return_value=None):
            report = generate_assessment_report(results, format='json',
                                                ai_enhanced=True)
        data = _json.loads(report)
        self.assertNotIn('ai_summary', data)


class TestGenerateAssessmentReportAiConfig(unittest.TestCase):
    """ai_config 参数测试 — 允许调用方显式传入 AIConfig"""

    def test_ai_config_passed_to_generate_report(self):
        """ai_config 参数应传给 ai.reporter.generate_report"""
        from tb_risk.export_utils.report_export import generate_assessment_report
        from tb_risk.ai.config import AIConfig
        results = _sample_results()
        cfg = AIConfig(provider='deepseek', api_key='sk-test')
        with mock.patch('tb_risk.ai.reporter.generate_report',
                        return_value="AI 摘要") as m:
            generate_assessment_report(results, format='text',
                                       ai_enhanced=True, ai_config=cfg)
        # 验证 config 参数被传递
        call_kwargs = m.call_args.kwargs
        self.assertIs(call_kwargs.get('config'), cfg)


if __name__ == '__main__':
    unittest.main()
