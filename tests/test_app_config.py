#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 测试套件 — AppConfig 应用配置持久化 (tb_risk.gui.app_config)

测试 Layer 2: AppConfig dataclass 字段扩展、序列化/反序列化往返、
向后兼容（旧 settings.json 无新字段时正常加载）、ai_is_local 派生属性。

设计原则：
- 不依赖真实 ~/.tb_risk/settings.json（用 monkeypatch 重定向 CONFIG_FILE）
- 验证方案 A：ai_api_key 不进入 AppConfig 序列化（仅存 ai_config.json）
- 验证向后兼容：旧配置文件缺失新字段时使用默认值
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestAppConfigDefaults(unittest.TestCase):
    """AppConfig 默认值测试（含 L2 新增 AI 字段）"""

    def test_ai_enabled_default_false(self):
        """ai_enabled 默认 False（用户须显式开启）"""
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig()
        self.assertFalse(cfg.ai_enabled)

    def test_ai_provider_default_deepseek(self):
        """ai_provider 默认 deepseek"""
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig()
        self.assertEqual(cfg.ai_provider, 'deepseek')

    def test_ai_base_url_default_empty(self):
        """ai_base_url 默认空字符串（留空使用服务商预设）"""
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig()
        self.assertEqual(cfg.ai_base_url, '')

    def test_ai_model_default_empty(self):
        """ai_model 默认空字符串（留空使用服务商预设）"""
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig()
        self.assertEqual(cfg.ai_model, '')

    def test_ai_local_model_path_default_empty(self):
        """ai_local_model_path 默认空字符串"""
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig()
        self.assertEqual(cfg.ai_local_model_path, '')

    def test_ai_timeout_default_60(self):
        """ai_timeout 默认 60 秒"""
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig()
        self.assertEqual(cfg.ai_timeout, 60)

    def test_ai_max_retries_default_3(self):
        """ai_max_retries 默认 3"""
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig()
        self.assertEqual(cfg.ai_max_retries, 3)

    def test_ai_temperature_default_0_2(self):
        """ai_temperature 默认 0.2"""
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig()
        self.assertEqual(cfg.ai_temperature, 0.2)

    def test_ai_prompts_dir_default_empty(self):
        """ai_prompts_dir 默认空字符串（不使用外置提示词）"""
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig()
        self.assertEqual(cfg.ai_prompts_dir, '')

    def test_no_ai_api_key_field(self):
        """方案 A：AppConfig 不包含 ai_api_key 字段

        ai_api_key 仅存 ~/.tb_risk/ai_config.json，避免随项目配置分享泄露。
        """
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig()
        # AppConfig 不应有 ai_api_key 字段
        self.assertNotIn('ai_api_key', AppConfig.__dataclass_fields__)
        # 也不应作为属性存在
        self.assertFalse(hasattr(cfg, 'ai_api_key'))


class TestAppConfigAiIsLocal(unittest.TestCase):
    """ai_is_local 只读派生属性测试"""

    def test_ai_is_local_false_when_no_local_model_path(self):
        """local_model_path 为空 → ai_is_local=False"""
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig(ai_local_model_path='')
        self.assertFalse(cfg.ai_is_local)

    def test_ai_is_local_true_when_local_model_path_set(self):
        """local_model_path 非空 → ai_is_local=True"""
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig(ai_local_model_path='/path/to/local/model')
        self.assertTrue(cfg.ai_is_local)

    def test_ai_is_local_is_readonly_property(self):
        """ai_is_local 是只读 property，赋值抛 AttributeError"""
        from tb_risk.gui.app_config import AppConfig
        cfg = AppConfig()
        with self.assertRaises(AttributeError):
            cfg.ai_is_local = True


class TestAppConfigSerialization(unittest.TestCase):
    """AppConfig 序列化/反序列化测试（含 L2 新增 AI 字段）"""

    def test_save_load_round_trip_preserves_ai_fields(self):
        """save → load 往返保持所有 AI 字段"""
        from tb_risk.gui.app_config import AppConfig
        with tempfile.TemporaryDirectory() as tmp:
            fake_config_file = os.path.join(tmp, 'settings.json')
            with mock.patch('tb_risk.gui.app_config.CONFIG_FILE',
                            fake_config_file), \
                 mock.patch('tb_risk.gui.app_config.CONFIG_DIR', tmp):
                original = AppConfig(
                    ai_enabled=True,
                    ai_provider='openai',
                    ai_base_url='https://custom.example.com/v1',
                    ai_model='gpt-4o',
                    ai_local_model_path='/path/to/model',
                    ai_timeout=120,
                    ai_max_retries=5,
                    ai_temperature=0.7,
                    ai_prompts_dir='/path/to/prompts',
                )
                original.save()
                loaded = AppConfig.load()
                self.assertEqual(loaded.ai_enabled, True)
                self.assertEqual(loaded.ai_provider, 'openai')
                self.assertEqual(loaded.ai_base_url,
                                 'https://custom.example.com/v1')
                self.assertEqual(loaded.ai_model, 'gpt-4o')
                self.assertEqual(loaded.ai_local_model_path, '/path/to/model')
                self.assertEqual(loaded.ai_timeout, 120)
                self.assertEqual(loaded.ai_max_retries, 5)
                self.assertEqual(loaded.ai_temperature, 0.7)
                self.assertEqual(loaded.ai_prompts_dir, '/path/to/prompts')

    def test_save_does_not_include_ai_api_key(self):
        """方案 A：save() 写入的 JSON 不包含 ai_api_key 字段"""
        from tb_risk.gui.app_config import AppConfig
        with tempfile.TemporaryDirectory() as tmp:
            fake_config_file = os.path.join(tmp, 'settings.json')
            with mock.patch('tb_risk.gui.app_config.CONFIG_FILE',
                            fake_config_file), \
                 mock.patch('tb_risk.gui.app_config.CONFIG_DIR', tmp):
                cfg = AppConfig(ai_enabled=True, ai_provider='deepseek')
                cfg.save()
                with open(fake_config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.assertNotIn('ai_api_key', data,
                                 "ai_api_key 不应出现在 settings.json 中")

    def test_save_includes_all_non_sensitive_ai_fields(self):
        """save() 写入的 JSON 包含所有非敏感 AI 字段"""
        from tb_risk.gui.app_config import AppConfig
        with tempfile.TemporaryDirectory() as tmp:
            fake_config_file = os.path.join(tmp, 'settings.json')
            with mock.patch('tb_risk.gui.app_config.CONFIG_FILE',
                            fake_config_file), \
                 mock.patch('tb_risk.gui.app_config.CONFIG_DIR', tmp):
                cfg = AppConfig()
                cfg.save()
                with open(fake_config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for field in ('ai_enabled', 'ai_provider', 'ai_base_url',
                              'ai_model', 'ai_local_model_path',
                              'ai_timeout', 'ai_max_retries',
                              'ai_temperature', 'ai_prompts_dir'):
                    self.assertIn(field, data,
                                  f"字段 {field} 应在 settings.json 中")

    def test_load_backward_compatible_with_old_config(self):
        """旧 settings.json（无新 AI 字段）→ load() 用默认值填充"""
        from tb_risk.gui.app_config import AppConfig
        with tempfile.TemporaryDirectory() as tmp:
            fake_config_file = os.path.join(tmp, 'settings.json')
            with mock.patch('tb_risk.gui.app_config.CONFIG_FILE',
                            fake_config_file), \
                 mock.patch('tb_risk.gui.app_config.CONFIG_DIR', tmp):
                # 模拟旧配置文件：仅有 ai_enabled/ai_provider/ai_local_model_path
                old_data = {
                    'window_geometry': '1200x800',
                    'ai_enabled': True,
                    'ai_provider': 'zhipu',
                    'ai_local_model_path': '/old/model',
                }
                with open(fake_config_file, 'w', encoding='utf-8') as f:
                    json.dump(old_data, f)
                cfg = AppConfig.load()
                # 旧字段保持
                self.assertTrue(cfg.ai_enabled)
                self.assertEqual(cfg.ai_provider, 'zhipu')
                self.assertEqual(cfg.ai_local_model_path, '/old/model')
                # 新字段使用默认值
                self.assertEqual(cfg.ai_base_url, '')
                self.assertEqual(cfg.ai_model, '')
                self.assertEqual(cfg.ai_timeout, 60)
                self.assertEqual(cfg.ai_max_retries, 3)
                self.assertEqual(cfg.ai_temperature, 0.2)
                self.assertEqual(cfg.ai_prompts_dir, '')

    def test_load_ignores_unknown_ai_fields(self):
        """load() 忽略未定义的字段（如误写的 ai_api_key）"""
        from tb_risk.gui.app_config import AppConfig
        with tempfile.TemporaryDirectory() as tmp:
            fake_config_file = os.path.join(tmp, 'settings.json')
            with mock.patch('tb_risk.gui.app_config.CONFIG_FILE',
                            fake_config_file), \
                 mock.patch('tb_risk.gui.app_config.CONFIG_DIR', tmp):
                # 模拟配置文件误写 ai_api_key
                bad_data = {
                    'ai_enabled': True,
                    'ai_api_key': 'sk-leaked-should-be-ignored',
                }
                with open(fake_config_file, 'w', encoding='utf-8') as f:
                    json.dump(bad_data, f)
                cfg = AppConfig.load()
                # ai_api_key 被忽略，不会成为 AppConfig 属性
                self.assertNotIn('ai_api_key',
                                 AppConfig.__dataclass_fields__)
                # ai_enabled 正常加载
                self.assertTrue(cfg.ai_enabled)

    def test_load_corrupted_file_returns_defaults(self):
        """损坏的 settings.json → load() 返回默认 AppConfig"""
        from tb_risk.gui.app_config import AppConfig
        with tempfile.TemporaryDirectory() as tmp:
            fake_config_file = os.path.join(tmp, 'settings.json')
            with mock.patch('tb_risk.gui.app_config.CONFIG_FILE',
                            fake_config_file), \
                 mock.patch('tb_risk.gui.app_config.CONFIG_DIR', tmp):
                with open(fake_config_file, 'w', encoding='utf-8') as f:
                    f.write('{ invalid json }')
                cfg = AppConfig.load()
                # 全部使用默认值
                self.assertFalse(cfg.ai_enabled)
                self.assertEqual(cfg.ai_provider, 'deepseek')

    def test_ai_fields_type_precision_no_loss(self):
        """ai_* 字段往返后类型与精度无丢失（#7）

        特别验证 ai_temperature（浮点）和 ai_timeout（整数）的精度。
        """
        from tb_risk.gui.app_config import AppConfig
        with tempfile.TemporaryDirectory() as tmp:
            fake_config_file = os.path.join(tmp, 'settings.json')
            with mock.patch('tb_risk.gui.app_config.CONFIG_FILE',
                            fake_config_file), \
                 mock.patch('tb_risk.gui.app_config.CONFIG_DIR', tmp):
                original = AppConfig(
                    ai_temperature=1.37,
                    ai_timeout=299,
                    ai_max_retries=7,
                    ai_enabled=True,
                    ai_provider='anthropic',
                    ai_base_url='https://api.example.com',
                    ai_model='claude-sonnet',
                    ai_local_model_path='/models/llama',
                    ai_prompts_dir='/prompts/custom',
                )
                original.save()
                loaded = AppConfig.load()
                # 类型验证
                self.assertIsInstance(loaded.ai_temperature, float)
                self.assertIsInstance(loaded.ai_timeout, int)
                self.assertIsInstance(loaded.ai_max_retries, int)
                self.assertIsInstance(loaded.ai_enabled, bool)
                # 精度验证
                self.assertEqual(loaded.ai_temperature, 1.37)
                self.assertEqual(loaded.ai_timeout, 299)
                self.assertEqual(loaded.ai_max_retries, 7)


class TestAppConfigAtomicWrite(unittest.TestCase):
    """AppConfig.save() 原子写入测试"""

    def test_save_no_tmp_file_residual(self):
        """save() 完成后不残留 .tmp 文件"""
        from tb_risk.gui.app_config import AppConfig
        with tempfile.TemporaryDirectory() as tmp:
            fake_config_file = os.path.join(tmp, 'settings.json')
            with mock.patch('tb_risk.gui.app_config.CONFIG_FILE',
                            fake_config_file), \
                 mock.patch('tb_risk.gui.app_config.CONFIG_DIR', tmp):
                AppConfig().save()
                files = os.listdir(tmp)
                # 只应有 settings.json，不应有 .tmp 文件
                self.assertEqual(files, ['settings.json'])


if __name__ == '__main__':
    unittest.main()
