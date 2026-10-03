#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 测试套件 — AI 提示词外置加载测试

缺陷 #15 修复：把提示词模板从硬编码常量改为可外置到 ~/.tb_risk/prompts/ 目录
的文本文件，AIConfig 增加 prompts_dir 字段，prompts.py 提供按需加载函数。

设计：
- 模块级常量保留为默认值（向后兼容）
- get_prompt(name, prompts_dir=None) 优先读 prompts_dir/<name>.txt，失败回退默认
- load_all_prompts(prompts_dir=None) 批量加载所有提示词
- AIConfig 新增 prompts_dir 字段（默认 None → 使用 ~/.tb_risk/prompts/）
- 业务模块（parser/reporter/qa/quality）通过 config.prompts_dir 加载自定义提示词
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


class TestGetPrompt(unittest.TestCase):
    """get_prompt() 函数测试"""

    def test_returns_default_when_no_prompts_dir(self):
        """prompts_dir=None → 返回模块级默认常量"""
        from tb_risk.ai.prompts import (
            get_prompt, EXTRACTION_SYSTEM_PROMPT,
        )
        result = get_prompt('EXTRACTION_SYSTEM_PROMPT')
        self.assertEqual(result, EXTRACTION_SYSTEM_PROMPT)

    def test_returns_default_when_dir_does_not_exist(self):
        """prompts_dir 指向不存在的目录 → 返回默认"""
        from tb_risk.ai.prompts import (
            get_prompt, REPORT_SYSTEM_PROMPT,
        )
        result = get_prompt('REPORT_SYSTEM_PROMPT',
                            prompts_dir='/nonexistent/path')
        self.assertEqual(result, REPORT_SYSTEM_PROMPT)

    def test_returns_file_content_when_override_exists(self):
        """prompts_dir/<name>.txt 存在 → 返回文件内容"""
        from tb_risk.ai.prompts import get_prompt
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_content = "自定义抽取系统提示词\n带特殊指令"
            file_path = os.path.join(tmpdir, 'EXTRACTION_SYSTEM_PROMPT.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(custom_content)
            result = get_prompt('EXTRACTION_SYSTEM_PROMPT',
                                prompts_dir=tmpdir)
        self.assertEqual(result, custom_content)

    def test_returns_default_when_file_not_in_dir(self):
        """prompts_dir 存在但未提供该提示词文件 → 返回默认"""
        from tb_risk.ai.prompts import (
            get_prompt, QA_SYSTEM_PROMPT,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            # 目录为空，未提供 QA_SYSTEM_PROMPT.txt
            result = get_prompt('QA_SYSTEM_PROMPT', prompts_dir=tmpdir)
        self.assertEqual(result, QA_SYSTEM_PROMPT)

    def test_unknown_prompt_name_returns_none(self):
        """未知提示词名 → 返回 None（不抛异常）"""
        from tb_risk.ai.prompts import get_prompt
        result = get_prompt('UNKNOWN_PROMPT_NAME')
        self.assertIsNone(result)

    def test_unknown_prompt_name_with_dir_returns_none(self):
        """未知提示词名 + prompts_dir 存在 → 返回 None"""
        from tb_risk.ai.prompts import get_prompt
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_prompt('UNKNOWN_PROMPT_NAME', prompts_dir=tmpdir)
        self.assertIsNone(result)

    def test_strips_trailing_newline_from_file(self):
        """文件末尾的换行符应被剥离（避免格式干扰）"""
        from tb_risk.ai.prompts import get_prompt
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, 'REPORT_SYSTEM_PROMPT.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("自定义内容\n\n")
            result = get_prompt('REPORT_SYSTEM_PROMPT',
                                prompts_dir=tmpdir)
        self.assertEqual(result, "自定义内容")


class TestLoadAllPrompts(unittest.TestCase):
    """load_all_prompts() 函数测试"""

    def test_returns_all_default_prompts_when_no_dir(self):
        """prompts_dir=None → 返回所有默认提示词"""
        from tb_risk.ai.prompts import (
            load_all_prompts,
            EXTRACTION_SYSTEM_PROMPT,
            REPORT_SYSTEM_PROMPT,
            QA_SYSTEM_PROMPT,
            DATA_QUALITY_SYSTEM_PROMPT,
        )
        result = load_all_prompts()
        self.assertIn('EXTRACTION_SYSTEM_PROMPT', result)
        self.assertIn('EXTRACTION_USER_TEMPLATE', result)
        self.assertIn('REPORT_SYSTEM_PROMPT', result)
        self.assertIn('REPORT_USER_TEMPLATE', result)
        self.assertIn('QA_SYSTEM_PROMPT', result)
        self.assertIn('QA_USER_TEMPLATE', result)
        self.assertIn('DATA_QUALITY_SYSTEM_PROMPT', result)
        self.assertIn('DATA_QUALITY_USER_TEMPLATE', result)
        self.assertEqual(result['EXTRACTION_SYSTEM_PROMPT'],
                         EXTRACTION_SYSTEM_PROMPT)
        self.assertEqual(result['REPORT_SYSTEM_PROMPT'],
                         REPORT_SYSTEM_PROMPT)

    def test_overrides_only_provided_prompts(self):
        """只覆盖提供的提示词，其他保持默认"""
        from tb_risk.ai.prompts import (
            load_all_prompts,
            QA_SYSTEM_PROMPT as default_qa,
            REPORT_SYSTEM_PROMPT as default_report,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            # 只覆盖 QA_SYSTEM_PROMPT
            with open(os.path.join(tmpdir, 'QA_SYSTEM_PROMPT.txt'),
                      'w', encoding='utf-8') as f:
                f.write("自定义 QA 系统提示")
            result = load_all_prompts(prompts_dir=tmpdir)
        # 被覆盖的应使用自定义内容
        self.assertEqual(result['QA_SYSTEM_PROMPT'], "自定义 QA 系统提示")
        # 未覆盖的应保持默认
        self.assertEqual(result['REPORT_SYSTEM_PROMPT'], default_report)
        # 但 QA_SYSTEM_PROMPT 不等于默认了
        self.assertNotEqual(result['QA_SYSTEM_PROMPT'], default_qa)


class TestAiConfigPromptsDir(unittest.TestCase):
    """AIConfig.prompts_dir 字段测试"""

    def test_config_has_prompts_dir_field(self):
        """AIConfig 应有 prompts_dir 字段，默认为 None"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig()
        self.assertIsNone(cfg.prompts_dir)

    def test_config_prompts_dir_can_be_set(self):
        """prompts_dir 可通过构造函数设置"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(prompts_dir='/custom/prompts')
        self.assertEqual(cfg.prompts_dir, '/custom/prompts')

    def test_config_from_env_reads_tb_ai_prompts_dir(self):
        """环境变量 TB_AI_PROMPTS_DIR 应被读取"""
        from tb_risk.ai.config import AIConfig
        env = {'TB_AI_PROMPTS_DIR': '/env/prompts/dir'}
        cfg = AIConfig.from_env(env=env)
        self.assertEqual(cfg.prompts_dir, '/env/prompts/dir')

    def test_config_to_dict_includes_prompts_dir(self):
        """to_dict 应包含 prompts_dir 字段"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(prompts_dir='/some/path')
        d = cfg.to_dict()
        self.assertIn('prompts_dir', d)
        self.assertEqual(d['prompts_dir'], '/some/path')

    def test_config_from_user_config_file_reads_prompts_dir(self):
        """用户配置文件中的 prompts_dir 字段应被读取"""
        import json
        from tb_risk.ai.config import AIConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, 'ai_config.json')
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump({'prompts_dir': '/from/config/file'}, f)
            cfg = AIConfig.from_user_config_file(cfg_path)
        self.assertEqual(cfg.prompts_dir, '/from/config/file')


class TestBusinessModulesUseCustomPrompts(unittest.TestCase):
    """业务模块应通过 config.prompts_dir 加载自定义提示词"""

    def test_reporter_uses_custom_prompts_from_config(self):
        """reporter.generate_report(config=cfg) → 使用 cfg.prompts_dir 中的提示词"""
        from tb_risk.ai import reporter
        from tb_risk.ai.config import AIConfig
        from tb_risk.ai.client import LLMClient

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建自定义 REPORT_SYSTEM_PROMPT 文件
            custom_sys = "自定义报告系统提示（医院 X 专用）"
            with open(os.path.join(tmpdir, 'REPORT_SYSTEM_PROMPT.txt'),
                      'w', encoding='utf-8') as f:
                f.write(custom_sys)
            cfg = AIConfig(api_key='sk-test', prompts_dir=tmpdir)

            # Mock client 捕获 messages
            captured_messages = []

            class _MockClient:
                config = cfg

                def chat(self, messages, **kwargs):
                    captured_messages.extend(messages)
                    return "AI 报告内容"

            result = {
                'patient_score': 75.0,
                'summary': {'overall_risk': '高'},
            }
            reporter.generate_report(result, client=_MockClient(),
                                     config=cfg)
        # 验证 system 消息使用了自定义提示词
        self.assertTrue(any(
            m['role'] == 'system' and m['content'] == custom_sys
            for m in captured_messages
        ), "应使用 config.prompts_dir 中的自定义系统提示词")

    def test_qa_uses_custom_prompts_from_config(self):
        """qa.ask(config=cfg) → 使用 cfg.prompts_dir 中的提示词"""
        from tb_risk.ai import qa
        from tb_risk.ai.config import AIConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_sys = "自定义问答系统提示"
            with open(os.path.join(tmpdir, 'QA_SYSTEM_PROMPT.txt'),
                      'w', encoding='utf-8') as f:
                f.write(custom_sys)
            cfg = AIConfig(api_key='sk-test', prompts_dir=tmpdir)

            captured_messages = []

            class _MockClient:
                config = cfg

                def chat(self, messages, **kwargs):
                    captured_messages.extend(messages)
                    return "AI 回答"

            qa.ask("为什么?", context={'patient_score': 75.0},
                   client=_MockClient(), config=cfg)
        self.assertTrue(any(
            m['role'] == 'system' and m['content'] == custom_sys
            for m in captured_messages
        ))

    def test_parser_uses_custom_prompts_from_config(self):
        """parser.extract_with_llm(config=cfg) → 使用 cfg.prompts_dir 中的提示词"""
        from tb_risk.ai import parser
        from tb_risk.ai.config import AIConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_sys = "自定义抽取系统提示"
            with open(os.path.join(tmpdir, 'EXTRACTION_SYSTEM_PROMPT.txt'),
                      'w', encoding='utf-8') as f:
                f.write(custom_sys)
            cfg = AIConfig(api_key='sk-test', prompts_dir=tmpdir)

            captured_messages = []

            class _MockClient:
                config = cfg

                def chat_json(self, messages, schema=None):
                    captured_messages.extend(messages)
                    return {'records': [{'age': 35}]}

            parser.extract_with_llm("患者 35 岁",
                                    client=_MockClient(), config=cfg)
        self.assertTrue(any(
            m['role'] == 'system' and m['content'] == custom_sys
            for m in captured_messages
        ))

    def test_quality_uses_custom_prompts_from_config(self):
        """quality.diagnose_record(config=cfg) → 使用 cfg.prompts_dir 中的提示词"""
        from tb_risk.ai import quality
        from tb_risk.ai.config import AIConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_sys = "自定义质量诊断系统提示"
            with open(os.path.join(tmpdir, 'DATA_QUALITY_SYSTEM_PROMPT.txt'),
                      'w', encoding='utf-8') as f:
                f.write(custom_sys)
            cfg = AIConfig(api_key='sk-test', prompts_dir=tmpdir)

            captured_messages = []

            class _MockClient:
                config = cfg

                def chat_json(self, messages, schema=None):
                    captured_messages.extend(messages)
                    return {'issues': [], 'overall_quality': '高',
                            'summary': ''}

            quality.diagnose_record({'name': '张三', 'age': 35},
                                    client=_MockClient(), config=cfg)
        self.assertTrue(any(
            m['role'] == 'system' and m['content'] == custom_sys
            for m in captured_messages
        ))


class TestBackwardCompat(unittest.TestCase):
    """向后兼容测试 — 现有代码不传 prompts_dir 应保持原行为"""

    def test_business_modules_work_without_prompts_dir(self):
        """未设置 prompts_dir 时业务模块应使用默认提示词正常工作"""
        from tb_risk.ai import reporter
        from tb_risk.ai.config import AIConfig
        from tb_risk.ai.prompts import REPORT_SYSTEM_PROMPT

        cfg = AIConfig(api_key='sk-test')  # prompts_dir=None
        captured_messages = []

        class _MockClient:
            config = cfg

            def chat(self, messages, **kwargs):
                captured_messages.extend(messages)
                return "AI 报告"

        reporter.generate_report(
            {'patient_score': 75.0, 'summary': {'overall_risk': '高'}},
            client=_MockClient(), config=cfg)
        # 应使用默认 REPORT_SYSTEM_PROMPT
        self.assertTrue(any(
            m['role'] == 'system' and m['content'] == REPORT_SYSTEM_PROMPT
            for m in captured_messages
        ))


if __name__ == '__main__':
    unittest.main()
