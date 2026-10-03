#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - CLI 命令行接口测试
"""

import json
import os
import sys
import tempfile
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

try:
    import click
    from click.testing import CliRunner
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False


@unittest.skipUnless(HAS_CLICK, "click 未安装，跳过 CLI 测试")
class TestCLIAssess(unittest.TestCase):
    """CLI assess 子命令测试"""

    def setUp(self):
        self.runner = CliRunner()
        # 创建临时配置文件
        self._tmpdir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self._tmpdir.name, 'test_config.json')
        config = {
            'patient_info': {
                'basic_info': {
                    'sputum_smear': 1,
                    'has_cavity': 1,
                    'cough_freq': 8,
                    'symptoms': 2,
                    'delay_days': 10,
                    'family_living_conditions': 3,
                    'flp_percentage': 10,
                    'hrsp_percentage': 15,
                    'treatment_duration': 2,
                }
            },
            'family_members': [
                {
                    'name': '测试成员',
                    'relationship': '配偶',
                    'age': 35,
                    'contact_distance': 'close',
                    'ventilation': 3,
                    'exposure_setting': 'general',
                    'freq_density': 14,
                    'single_duration': 120,
                    'time_span': 4,
                }
            ],
            'social_contacts': [],
            'use_ml': False,
            'use_seir': False,
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_cli_group_help(self):
        """测试 CLI 主帮助信息"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('assess', result.output)
        self.assertIn('simulate', result.output)

    def test_assess_help(self):
        """测试 assess 子命令帮助"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['assess', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('CONFIG_FILE', result.output)

    def test_assess_basic(self):
        """测试基本评估命令"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['assess', self.config_path])
        self.assertEqual(result.exit_code, 0)

    def test_assess_json_output(self):
        """测试 JSON 格式输出"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(
            cli, ['assess', self.config_path, '--format', 'json'])
        self.assertEqual(result.exit_code, 0)

    def test_assess_table_output(self):
        """测试表格格式输出"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(
            cli, ['assess', self.config_path, '--format', 'table'])
        self.assertEqual(result.exit_code, 0)

    def test_assess_with_ml(self):
        """测试启用 ML 预测的评估"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(
            cli, ['assess', self.config_path, '--ml'])
        self.assertEqual(result.exit_code, 0)

    def test_assess_with_seir(self):
        """测试启用 SEIR 模拟的评估"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(
            cli, ['assess', self.config_path, '--seir'])
        self.assertEqual(result.exit_code, 0)

    def test_assess_missing_file(self):
        """测试配置文件不存在时的错误处理"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['assess', 'nonexistent.json'])
        self.assertNotEqual(result.exit_code, 0)

    def test_assess_invalid_json(self):
        """测试无效 JSON 配置文件时的错误处理"""
        from tb_risk.cli.commands import cli
        invalid_path = os.path.join(self._tmpdir.name, 'invalid.json')
        with open(invalid_path, 'w', encoding='utf-8') as f:
            f.write('{invalid json')
        result = self.runner.invoke(cli, ['assess', invalid_path])
        self.assertNotEqual(result.exit_code, 0)

    def test_assess_csv_output(self):
        """测试 CSV 格式输出"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(
            cli, ['assess', self.config_path, '--format', 'csv'])
        self.assertEqual(result.exit_code, 0)


@unittest.skipUnless(HAS_CLICK, "click 未安装，跳过 CLI 测试")
class TestCLIConfig(unittest.TestCase):
    """CLI config 子命令测试"""

    def setUp(self):
        self.runner = CliRunner()

    def test_config_show(self):
        """测试 config show 命令"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['config', 'show'])
        self.assertEqual(result.exit_code, 0)


@unittest.skipUnless(HAS_CLICK, "click 未安装，跳过 CLI 测试")
class TestCLICheck(unittest.TestCase):
    """CLI check 子命令测试"""

    def setUp(self):
        self.runner = CliRunner()

    def test_check_deps(self):
        """测试依赖检查命令"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['check'])
        self.assertEqual(result.exit_code, 0)


@unittest.skipUnless(HAS_CLICK, "click 未安装，跳过 CLI 测试")
class TestCLISimulate(unittest.TestCase):
    """CLI simulate 子命令测试"""

    def setUp(self):
        self.runner = CliRunner()

    def test_simulate_help(self):
        """测试 simulate 子命令帮助"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['simulate', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('simulate', result.output)


@unittest.skipUnless(HAS_CLICK, "click 未安装，跳过 CLI 测试")
class TestCLIVersion(unittest.TestCase):
    """CLI 版本信息测试"""

    def setUp(self):
        self.runner = CliRunner()

    def test_version(self):
        """测试版本号输出"""
        from tb_risk.cli.commands import cli
        result = self.runner.invoke(cli, ['--version'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('5.0.0', result.output)


if __name__ == '__main__':
    unittest.main()