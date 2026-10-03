#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 测试套件 — CLI ai 子命令组测试

缺陷 #13 修复：新增 tb-risk ai extract / report / diagnose 三个子命令，
复用 tb_risk.ai 子包能力，让命令行用户也能使用 AI 功能。

测试设计：
- 使用 click.testing.CliRunner 驱动命令行
- mock tb_risk.ai.extract_with_llm / generate_report / diagnose_record
  避免依赖真实 LLM API
- 验证：参数解析、文件读取、配置加载、未配置降级、输出格式
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

try:
    import click
    from click.testing import CliRunner
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False


@unittest.skipUnless(HAS_CLICK, "click 未安装，跳过 CLI 测试")
class TestCliAiGroupHelp(unittest.TestCase):
    """ai 子命令组帮助信息测试"""

    def setUp(self):
        self.runner = CliRunner()

    def test_ai_group_help_lists_subcommands(self):
        """tb-risk ai --help 应列出 extract / report / diagnose 子命令"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['ai', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('extract', result.output)
        self.assertIn('report', result.output)
        self.assertIn('diagnose', result.output)

    def test_ai_extract_help(self):
        """tb-risk ai extract --help 应显示用法"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['ai', 'extract', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('TEXT_FILE', result.output)

    def test_ai_report_help(self):
        """tb-risk ai report --help 应显示用法"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['ai', 'report', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('RESULT_FILE', result.output)

    def test_ai_diagnose_help(self):
        """tb-risk ai diagnose --help 应显示用法"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['ai', 'diagnose', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('RECORD_FILE', result.output)


@unittest.skipUnless(HAS_CLICK, "click 未安装，跳过 CLI 测试")
class TestCliAiExtract(unittest.TestCase):
    """tb-risk ai extract 子命令测试"""

    def setUp(self):
        self.runner = CliRunner()
        self._tmpdir = tempfile.TemporaryDirectory()
        # 创建测试文本文件
        self.text_file = os.path.join(self._tmpdir.name, 'patient.txt')
        with open(self.text_file, 'w', encoding='utf-8') as f:
            f.write('患者男 35 岁，咳嗽 8 次/天，痰涂片阳性，有空洞')

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_extract_success_outputs_json(self):
        """成功抽取 → 输出 JSON 格式的记录列表"""
        from tb_risk.cli.commands import cli
        sample_records = [
            {'age': 35, 'gender': '男', 'sputum_smear': 1,
             'has_cavity': 1, 'cough_freq': 8, '_source': {'source': 'llm'}},
        ]
        with mock.patch('tb_risk.ai.extract_with_llm',
                        return_value=sample_records) as m:
            result = self.runner.invoke(cli, ['ai', 'extract', self.text_file])
        self.assertEqual(result.exit_code, 0)
        m.assert_called_once()
        # 输出应包含 JSON 内容
        self.assertIn('age', result.output)
        self.assertIn('35', result.output)
        # 调用时应传入文件文本内容
        call_args = m.call_args
        text_arg = call_args[0][0] if call_args[0] else call_args[1].get('text')
        self.assertIn('咳嗽', text_arg)

    def test_extract_writes_to_output_file(self):
        """--output 指定输出文件 → 结果写入文件"""
        from tb_risk.cli.commands import cli
        output_path = os.path.join(self._tmpdir.name, 'out.json')
        sample_records = [{'age': 35}]
        with mock.patch('tb_risk.ai.extract_with_llm',
                        return_value=sample_records):
            result = self.runner.invoke(
                cli, ['ai', 'extract', self.text_file, '--output', output_path])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        self.assertEqual(saved, sample_records)

    def test_extract_file_not_found(self):
        """文件不存在 → 非零退出码"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['ai', 'extract', 'nonexistent.txt'])
        # click.Path(exists=True) 会让 click 自己报错
        self.assertNotEqual(result.exit_code, 0)

    def test_extract_ai_not_configured_returns_nonzero(self):
        """AI 未配置（extract_with_llm 返回 None）→ 错误信息 + 非零退出码"""
        from tb_risk.cli.commands import cli
        with mock.patch('tb_risk.ai.extract_with_llm', return_value=None):
            result = self.runner.invoke(cli, ['ai', 'extract', self.text_file])
        self.assertNotEqual(result.exit_code, 0)
        # 错误信息应提示用户配置 AI
        self.assertIn('AI', result.output)

    def test_extract_passes_config_file_to_load(self):
        """--config 选项 → 加载指定配置文件"""
        from tb_risk.cli.commands import cli
        cfg_path = os.path.join(self._tmpdir.name, 'ai_cfg.json')
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump({'provider': 'deepseek', 'api_key': 'sk-test'}, f)
        with mock.patch('tb_risk.ai.extract_with_llm', return_value=[{'age': 35}]):
            with mock.patch('tb_risk.ai.config.AIConfig.from_user_config_file') as m:
                from tb_risk.ai.config import AIConfig
                m.return_value = AIConfig(provider='deepseek', api_key='sk-test')
                result = self.runner.invoke(
                    cli, ['ai', 'extract', self.text_file,
                          '--config', cfg_path])
        self.assertEqual(result.exit_code, 0)
        m.assert_called_once_with(cfg_path)

    def test_extract_local_flag_overrides_local_model_path(self):
        """--local 选项 → 覆盖 config.local_model_path"""
        from tb_risk.cli.commands import cli
        local_model_path = '/path/to/local/model'
        with mock.patch('tb_risk.ai.extract_with_llm', return_value=[{'age': 35}]) as m:
            result = self.runner.invoke(
                cli, ['ai', 'extract', self.text_file,
                      '--local', local_model_path])
        self.assertEqual(result.exit_code, 0)
        # 验证传给 extract_with_llm 的 config.local_model_path 被设置
        call_kwargs = m.call_args.kwargs
        cfg = call_kwargs.get('config')
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.local_model_path, local_model_path)


@unittest.skipUnless(HAS_CLICK, "click 未安装，跳过 CLI 测试")
class TestCliAiReport(unittest.TestCase):
    """tb-risk ai report 子命令测试"""

    def setUp(self):
        self.runner = CliRunner()
        self._tmpdir = tempfile.TemporaryDirectory()
        # 创建测试 result JSON
        self.result_file = os.path.join(self._tmpdir.name, 'result.json')
        sample_result = {
            'patient_score': 75.0,
            'summary': {'overall_risk': '高', 'total_contacts': 5},
            'potential_patients': {'family': [], 'social': []},
        }
        with open(self.result_file, 'w', encoding='utf-8') as f:
            json.dump(sample_result, f, ensure_ascii=False)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_report_success_outputs_text(self):
        """成功生成报告 → 输出文本"""
        from tb_risk.cli.commands import cli
        sample_report = "## 风险评估报告\n患者风险评分 75 分，属于高风险。"
        with mock.patch('tb_risk.ai.generate_report',
                        return_value=sample_report) as m:
            result = self.runner.invoke(cli, ['ai', 'report', self.result_file])
        self.assertEqual(result.exit_code, 0)
        m.assert_called_once()
        self.assertIn('风险评估报告', result.output)
        # 验证传入的 result 是从 JSON 文件加载的
        call_args = m.call_args
        result_arg = call_args[0][0] if call_args[0] else call_args[1].get('result')
        self.assertEqual(result_arg['patient_score'], 75.0)

    def test_report_writes_to_output_file(self):
        """--output 写入文件"""
        from tb_risk.cli.commands import cli
        output_path = os.path.join(self._tmpdir.name, 'report.md')
        sample_report = "## 报告内容"
        with mock.patch('tb_risk.ai.generate_report',
                        return_value=sample_report):
            result = self.runner.invoke(
                cli, ['ai', 'report', self.result_file,
                      '--output', output_path])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('报告内容', content)

    def test_report_file_not_found(self):
        """文件不存在 → 非零退出码"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['ai', 'report', 'nonexistent.json'])
        self.assertNotEqual(result.exit_code, 0)

    def test_report_invalid_json(self):
        """JSON 格式无效 → 非零退出码"""
        from tb_risk.cli.commands import cli
        bad_path = os.path.join(self._tmpdir.name, 'bad.json')
        with open(bad_path, 'w', encoding='utf-8') as f:
            f.write('{invalid json')
        result = self.runner.invoke(cli, ['ai', 'report', bad_path])
        self.assertNotEqual(result.exit_code, 0)

    def test_report_ai_not_configured_returns_nonzero(self):
        """AI 未配置 → 非零退出码"""
        from tb_risk.cli.commands import cli
        with mock.patch('tb_risk.ai.generate_report', return_value=None):
            result = self.runner.invoke(cli, ['ai', 'report', self.result_file])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('AI', result.output)

    def test_report_local_flag_overrides_local_model_path(self):
        """--local 选项 → 启用本地模式"""
        from tb_risk.cli.commands import cli
        local_model_path = '/path/to/local/model'
        with mock.patch('tb_risk.ai.generate_report',
                        return_value="## 报告") as m:
            result = self.runner.invoke(
                cli, ['ai', 'report', self.result_file,
                      '--local', local_model_path])
        self.assertEqual(result.exit_code, 0)
        call_kwargs = m.call_args.kwargs
        cfg = call_kwargs.get('config')
        self.assertEqual(cfg.local_model_path, local_model_path)


@unittest.skipUnless(HAS_CLICK, "click 未安装，跳过 CLI 测试")
class TestCliAiDiagnose(unittest.TestCase):
    """tb-risk ai diagnose 子命令测试"""

    def setUp(self):
        self.runner = CliRunner()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.record_file = os.path.join(self._tmpdir.name, 'record.json')
        sample_record = {
            'name': '张三',
            'age': 35,
            'sputum_smear': 1,
            'cough_freq': 8,
        }
        with open(self.record_file, 'w', encoding='utf-8') as f:
            json.dump(sample_record, f, ensure_ascii=False)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_diagnose_success_outputs_json(self):
        """成功诊断 → 输出 JSON 格式结果"""
        from tb_risk.cli.commands import cli
        sample_diag = {
            'issues': [
                {'field': 'cough_freq', 'problem': '偏高',
                 'suggestion': '复查', 'confidence': 0.9}
            ],
            'overall_quality': '中',
            'summary': '咳嗽频率偏高，建议复查',
        }
        with mock.patch('tb_risk.ai.diagnose_record',
                        return_value=sample_diag) as m:
            result = self.runner.invoke(cli, ['ai', 'diagnose', self.record_file])
        self.assertEqual(result.exit_code, 0)
        m.assert_called_once()
        self.assertIn('issues', result.output)
        self.assertIn('咳嗽频率偏高', result.output)
        # 验证传入的 record 是从 JSON 文件加载的
        call_args = m.call_args
        record_arg = call_args[0][0] if call_args[0] else call_args[1].get('record')
        self.assertEqual(record_arg['name'], '张三')

    def test_diagnose_writes_to_output_file(self):
        """--output 写入文件"""
        from tb_risk.cli.commands import cli
        output_path = os.path.join(self._tmpdir.name, 'diag.json')
        sample_diag = {'issues': [], 'overall_quality': '高', 'summary': '正常'}
        with mock.patch('tb_risk.ai.diagnose_record',
                        return_value=sample_diag):
            result = self.runner.invoke(
                cli, ['ai', 'diagnose', self.record_file,
                      '--output', output_path])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        self.assertEqual(saved, sample_diag)

    def test_diagnose_file_not_found(self):
        """文件不存在 → 非零退出码"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['ai', 'diagnose', 'nonexistent.json'])
        self.assertNotEqual(result.exit_code, 0)

    def test_diagnose_ai_not_configured_returns_nonzero(self):
        """AI 未配置 → 非零退出码"""
        from tb_risk.cli.commands import cli
        with mock.patch('tb_risk.ai.diagnose_record', return_value=None):
            result = self.runner.invoke(cli, ['ai', 'diagnose', self.record_file])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('AI', result.output)

    def test_diagnose_local_flag_overrides_local_model_path(self):
        """--local 选项 → 启用本地模式（跳过脱敏）"""
        from tb_risk.cli.commands import cli
        local_model_path = '/path/to/local/model'
        with mock.patch('tb_risk.ai.diagnose_record',
                        return_value={'issues': [], 'overall_quality': '高',
                                      'summary': ''}) as m:
            result = self.runner.invoke(
                cli, ['ai', 'diagnose', self.record_file,
                      '--local', local_model_path])
        self.assertEqual(result.exit_code, 0)
        call_kwargs = m.call_args.kwargs
        cfg = call_kwargs.get('config')
        self.assertEqual(cfg.local_model_path, local_model_path)


if __name__ == '__main__':
    unittest.main()
