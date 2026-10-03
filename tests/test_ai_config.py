#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — AI 配置模块 (tb_risk.ai.config)

测试 Layer 2-3: 环境变量读取、提供商预设、DeepSeek 默认、
本地模型路径、超时/重试参数、用户配置文件加载、API Key 安全。

设计原则：
- 所有测试均不依赖真实 API Key（用 monkeypatch 设置环境变量）
- 测试覆盖 6 个 OpenAI 兼容提供商：openai/deepseek/dashscope/zhipu/moonshot/anthropic
- 验证 API Key 不会被 repr()/str() 泄露
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class TestProviders(unittest.TestCase):
    """PROVIDERS 预设表测试"""

    def test_providers_contains_deepseek(self):
        """DeepSeek 必须在预设表中"""
        from tb_risk.ai.config import PROVIDERS
        self.assertIn('deepseek', PROVIDERS)

    def test_providers_deepseek_default_model(self):
        """DeepSeek 默认模型为 deepseek-chat"""
        from tb_risk.ai.config import PROVIDERS
        self.assertEqual(PROVIDERS['deepseek']['model'], 'deepseek-chat')

    def test_providers_deepseek_base_url(self):
        """DeepSeek base_url 符合 OpenAI 兼容协议"""
        from tb_risk.ai.config import PROVIDERS
        self.assertEqual(PROVIDERS['deepseek']['base_url'],
                         'https://api.deepseek.com/v1')

    def test_providers_contains_all_compatible(self):
        """6 个 OpenAI 兼容提供商均存在"""
        from tb_risk.ai.config import PROVIDERS
        for name in ('openai', 'deepseek', 'dashscope', 'zhipu', 'moonshot'):
            self.assertIn(name, PROVIDERS, f"缺失提供商: {name}")

    def test_providers_all_have_base_url_and_model(self):
        """每个提供商都有 base_url 和 model"""
        from tb_risk.ai.config import PROVIDERS
        for name, cfg in PROVIDERS.items():
            self.assertIn('base_url', cfg, f"{name} 缺 base_url")
            self.assertIn('model', cfg, f"{name} 缺 model")
            self.assertTrue(cfg['base_url'].startswith('https://'),
                            f"{name} base_url 必须 HTTPS")


class TestAIConfigDefaults(unittest.TestCase):
    """AIConfig 默认值测试"""

    def test_default_provider_is_deepseek(self):
        """未设置任何环境变量时，默认 provider 为 deepseek"""
        from tb_risk.ai.config import AIConfig
        # 清空所有 AI 相关环境变量
        env_keys = ['TB_AI_PROVIDER', 'TB_AI_API_KEY', 'TB_AI_BASE_URL',
                    'TB_AI_MODEL', 'TB_AI_TIMEOUT', 'TB_AI_MAX_RETRIES',
                    'TB_AI_TEMPERATURE', 'TB_AI_LOCAL_MODEL',
                    'TB_AI_LOCAL_DEVICE']
        saved = {k: os.environ.pop(k, None) for k in env_keys}
        try:
            cfg = AIConfig.from_env()
            self.assertEqual(cfg.provider, 'deepseek')
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_default_timeout_is_60(self):
        """默认超时 60 秒（避免 GUI 卡死）"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig()
        self.assertEqual(cfg.timeout, 60)

    def test_default_max_retries_is_3(self):
        """默认最大重试 3 次"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig()
        self.assertEqual(cfg.max_retries, 3)

    def test_default_temperature_is_0_2(self):
        """默认温度 0.2（抽取任务偏向确定性）"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig()
        self.assertAlmostEqual(cfg.temperature, 0.2)

    def test_default_api_key_is_none(self):
        """默认 api_key 为 None（未配置）"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig()
        self.assertIsNone(cfg.api_key)

    def test_default_local_model_is_none(self):
        """默认不启用本地模型"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig()
        self.assertIsNone(cfg.local_model_path)


class TestAIConfigFromEnv(unittest.TestCase):
    """AIConfig.from_env() 环境变量读取测试"""

    def setUp(self):
        """保存当前环境变量，测试后还原"""
        self._saved = dict(os.environ)

    def tearDown(self):
        """还原环境变量"""
        os.environ.clear()
        os.environ.update(self._saved)

    def test_reads_provider_from_env(self):
        """TB_AI_PROVIDER 决定 provider"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_PROVIDER'] = 'zhipu'
        cfg = AIConfig.from_env()
        self.assertEqual(cfg.provider, 'zhipu')

    def test_reads_api_key_from_env(self):
        """TB_AI_API_KEY 决定 api_key"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_API_KEY'] = 'sk-test-12345'
        cfg = AIConfig.from_env()
        self.assertEqual(cfg.api_key, 'sk-test-12345')

    def test_reads_base_url_override_from_env(self):
        """TB_AI_BASE_URL 覆盖预设 base_url（私有部署兼容）"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_PROVIDER'] = 'deepseek'
        os.environ['TB_AI_BASE_URL'] = 'https://internal-proxy.local/v1'
        cfg = AIConfig.from_env()
        self.assertEqual(cfg.base_url, 'https://internal-proxy.local/v1')

    def test_reads_model_override_from_env(self):
        """TB_AI_MODEL 覆盖预设 model"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_PROVIDER'] = 'deepseek'
        os.environ['TB_AI_MODEL'] = 'deepseek-reasoner'
        cfg = AIConfig.from_env()
        self.assertEqual(cfg.model, 'deepseek-reasoner')

    def test_reads_timeout_from_env(self):
        """TB_AI_TIMEOUT 解析为 int"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_TIMEOUT'] = '120'
        cfg = AIConfig.from_env()
        self.assertEqual(cfg.timeout, 120)

    def test_reads_max_retries_from_env(self):
        """TB_AI_MAX_RETRIES 解析为 int"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_MAX_RETRIES'] = '5'
        cfg = AIConfig.from_env()
        self.assertEqual(cfg.max_retries, 5)

    def test_reads_temperature_from_env(self):
        """TB_AI_TEMPERATURE 解析为 float"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_TEMPERATURE'] = '0.7'
        cfg = AIConfig.from_env()
        self.assertAlmostEqual(cfg.temperature, 0.7)

    def test_reads_local_model_path_from_env(self):
        """TB_AI_LOCAL_MODEL 启用本地模型路径"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_LOCAL_MODEL'] = '/path/to/Qwen2.5-7B-Instruct'
        cfg = AIConfig.from_env()
        self.assertEqual(cfg.local_model_path, '/path/to/Qwen2.5-7B-Instruct')

    def test_reads_local_device_from_env(self):
        """TB_AI_LOCAL_DEVICE 指定推理设备"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_LOCAL_DEVICE'] = 'cuda:0'
        cfg = AIConfig.from_env()
        self.assertEqual(cfg.local_device, 'cuda:0')

    def test_invalid_timeout_falls_back_to_default(self):
        """非法 timeout 值回退到默认 60"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_TIMEOUT'] = 'not-a-number'
        cfg = AIConfig.from_env()
        self.assertEqual(cfg.timeout, 60)

    def test_invalid_provider_falls_back_to_deepseek(self):
        """未知 provider 回退到 deepseek"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_PROVIDER'] = 'unknown-vendor'
        cfg = AIConfig.from_env()
        self.assertEqual(cfg.provider, 'deepseek')

    def test_uses_provider_preset_base_url_when_no_override(self):
        """无 base_url 覆盖时使用预设值"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_PROVIDER'] = 'dashscope'
        cfg = AIConfig.from_env()
        self.assertEqual(cfg.base_url,
                         'https://dashscope.aliyuncs.com/compatible-mode/v1')

    def test_uses_provider_preset_model_when_no_override(self):
        """无 model 覆盖时使用预设值"""
        from tb_risk.ai.config import AIConfig
        os.environ['TB_AI_PROVIDER'] = 'zhipu'
        cfg = AIConfig.from_env()
        self.assertEqual(cfg.model, 'glm-4-flash')


class TestAIConfigIsConfigured(unittest.TestCase):
    """AIConfig.is_configured() 判断测试"""

    def test_not_configured_when_no_api_key_and_no_local(self):
        """无 api_key 且无本地模型 → 未配置"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig()  # 全默认
        self.assertFalse(cfg.is_configured())

    def test_configured_when_api_key_present(self):
        """有 api_key → 已配置（云端模式）"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-test')
        self.assertTrue(cfg.is_configured())

    def test_configured_when_local_model_present(self):
        """有 local_model_path → 已配置（本地模式）"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(local_model_path='/path/to/model')
        self.assertTrue(cfg.is_configured())

    def test_is_local_true_when_local_model_set(self):
        """is_local 属性：仅 local_model_path 设置时为 True"""
        from tb_risk.ai.config import AIConfig
        cfg_cloud = AIConfig(api_key='sk-test')
        cfg_local = AIConfig(local_model_path='/path/to/model')
        cfg_both = AIConfig(api_key='sk-test', local_model_path='/path/to/model')
        self.assertFalse(cfg_cloud.is_local)
        self.assertTrue(cfg_local.is_local)
        # 同时设置时优先本地（敏感场景）
        self.assertTrue(cfg_both.is_local)


class TestAIConfigSecurity(unittest.TestCase):
    """API Key 安全测试（不泄露到日志/repr）"""

    def test_repr_does_not_leak_api_key(self):
        """repr() 不能包含 api_key 实际值"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-SUPER-SECRET-KEY-12345')
        rep = repr(cfg)
        self.assertNotIn('sk-SUPER-SECRET-KEY-12345', rep)

    def test_str_does_not_leak_api_key(self):
        """str() 不能包含 api_key 实际值"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-SUPER-SECRET-KEY-12345')
        s = str(cfg)
        self.assertNotIn('sk-SUPER-SECRET-KEY-12345', s)

    def test_to_dict_masks_api_key(self):
        """to_dict() 中 api_key 被掩码"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(api_key='sk-SUPER-SECRET-KEY-12345')
        d = cfg.to_dict()
        self.assertIn('***', str(d['api_key']))
        self.assertNotIn('sk-SUPER-SECRET-KEY-12345', str(d['api_key']))


class TestAIConfigUserFile(unittest.TestCase):
    """AIConfig.from_user_config_file() 用户配置文件测试"""

    def test_loads_from_user_config_file(self):
        """从 ~/.tb_risk/ai_config.json 加载"""
        from tb_risk.ai.config import AIConfig
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, 'ai_config.json')
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'provider': 'moonshot',
                    'api_key': 'sk-from-file',
                    'model': 'moonshot-v1-8k',
                }, f)
            cfg = AIConfig.from_user_config_file(cfg_path)
            self.assertEqual(cfg.provider, 'moonshot')
            self.assertEqual(cfg.api_key, 'sk-from-file')
            self.assertEqual(cfg.model, 'moonshot-v1-8k')

    def test_returns_default_when_file_missing(self):
        """文件不存在时返回默认配置"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig.from_user_config_file('/nonexistent/path/ai_config.json')
        self.assertEqual(cfg.provider, 'deepseek')
        self.assertIsNone(cfg.api_key)

    def test_returns_default_when_file_corrupt(self):
        """文件损坏（非法 JSON）时返回默认配置"""
        from tb_risk.ai.config import AIConfig
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, 'ai_config.json')
            with open(cfg_path, 'w', encoding='utf-8') as f:
                f.write('{"provider": "moonshot", "api_key": broken')
            cfg = AIConfig.from_user_config_file(cfg_path)
            self.assertEqual(cfg.provider, 'deepseek')

    def test_env_overrides_file(self):
        """环境变量优先级高于配置文件"""
        from tb_risk.ai.config import AIConfig
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, 'ai_config.json')
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump({'provider': 'moonshot', 'api_key': 'sk-file'}, f)
            # 环境变量覆盖
            old_key = os.environ.pop('TB_AI_API_KEY', None)
            old_provider = os.environ.pop('TB_AI_PROVIDER', None)
            os.environ['TB_AI_API_KEY'] = 'sk-from-env'
            try:
                cfg = AIConfig.from_user_config_file(cfg_path)
                self.assertEqual(cfg.api_key, 'sk-from-env')
                # provider 未在 env 设置，保留文件值
                self.assertEqual(cfg.provider, 'moonshot')
            finally:
                os.environ.pop('TB_AI_API_KEY', None)
                if old_key is not None:
                    os.environ['TB_AI_API_KEY'] = old_key
                if old_provider is not None:
                    os.environ['TB_AI_PROVIDER'] = old_provider


class TestAIConfigSchemaValidation(unittest.TestCase):
    """缺陷 #9（本轮）：AIConfig.from_user_config_file 字段类型校验

    用户手写配置文件字段类型错误（如 max_retries: "3" 字符串）应在加载时
    抛出明确错误，而不是被 _safe_int 静默回退到默认值。
    """

    def _write_and_load(self, data):
        """辅助：写入临时配置文件并加载"""
        from tb_risk.ai.config import AIConfig
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, 'ai_config.json')
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            return AIConfig.from_user_config_file(cfg_path)

    def test_valid_config_loads_without_error(self):
        """合法配置（字段类型正确）正常加载"""
        cfg = self._write_and_load({
            'provider': 'deepseek',
            'api_key': 'sk-test',
            'timeout': 60,
            'max_retries': 3,
            'temperature': 0.2,
        })
        self.assertEqual(cfg.provider, 'deepseek')
        self.assertEqual(cfg.api_key, 'sk-test')
        self.assertEqual(cfg.timeout, 60)

    def test_partial_config_loads_without_error(self):
        """部分配置（仅部分字段）正常加载"""
        cfg = self._write_and_load({'api_key': 'sk-test'})
        self.assertEqual(cfg.api_key, 'sk-test')
        # 缺字段使用默认值
        self.assertEqual(cfg.provider, 'deepseek')

    def test_timeout_string_raises_type_error(self):
        """timeout 为字符串 → 抛出 TypeError"""
        with self.assertRaises(TypeError) as cm:
            self._write_and_load({'timeout': '60'})
        self.assertIn('timeout', str(cm.exception))
        self.assertIn('int', str(cm.exception))

    def test_max_retries_string_raises_type_error(self):
        """max_retries 为字符串 → 抛出 TypeError"""
        with self.assertRaises(TypeError) as cm:
            self._write_and_load({'max_retries': '3'})
        self.assertIn('max_retries', str(cm.exception))
        self.assertIn('int', str(cm.exception))

    def test_temperature_string_raises_type_error(self):
        """temperature 为字符串 → 抛出 TypeError"""
        with self.assertRaises(TypeError) as cm:
            self._write_and_load({'temperature': '0.2'})
        self.assertIn('temperature', str(cm.exception))

    def test_provider_non_string_raises_type_error(self):
        """provider 为非字符串 → 抛出 TypeError"""
        with self.assertRaises(TypeError) as cm:
            self._write_and_load({'provider': 123})
        self.assertIn('provider', str(cm.exception))

    def test_api_key_non_string_raises_type_error(self):
        """api_key 为非字符串/None → 抛出 TypeError"""
        with self.assertRaises(TypeError) as cm:
            self._write_and_load({'api_key': 12345})
        self.assertIn('api_key', str(cm.exception))

    def test_base_url_non_string_raises_type_error(self):
        """base_url 为非字符串 → 抛出 TypeError"""
        with self.assertRaises(TypeError):
            self._write_and_load({'base_url': ['http://example.com']})

    def test_model_non_string_raises_type_error(self):
        """model 为非字符串 → 抛出 TypeError"""
        with self.assertRaises(TypeError):
            self._write_and_load({'model': 3.14})

    def test_local_device_non_string_raises_type_error(self):
        """local_device 为非字符串 → 抛出 TypeError"""
        with self.assertRaises(TypeError):
            self._write_and_load({'local_device': 0})

    def test_prompts_dir_non_string_raises_type_error(self):
        """prompts_dir 为非字符串/None → 抛出 TypeError"""
        with self.assertRaises(TypeError):
            self._write_and_load({'prompts_dir': ['/prompts']})

    def test_bool_rejected_for_int_field(self):
        """bool 不应被接受为 int 字段（避免 True 被误当作 1）"""
        # bool 是 int 的子类，但语义上不应作为 timeout 使用
        with self.assertRaises(TypeError):
            self._write_and_load({'timeout': True})

    def test_error_message_is_informative(self):
        """错误信息应包含字段名、期望类型与实际类型"""
        with self.assertRaises(TypeError) as cm:
            self._write_and_load({'max_retries': '3'})
        msg = str(cm.exception)
        # 应包含字段名
        self.assertIn('max_retries', msg)
        # 应包含期望类型
        self.assertIn('int', msg.lower())


class TestDefaultConfigPath(unittest.TestCase):
    """默认配置文件路径常量测试"""

    def test_default_config_path_under_tb_risk_dir(self):
        """默认配置文件路径位于 ~/.tb_risk/ai_config.json"""
        from tb_risk.ai.config import DEFAULT_AI_CONFIG_FILE
        self.assertTrue(DEFAULT_AI_CONFIG_FILE.endswith('ai_config.json'))
        self.assertIn('.tb_risk', DEFAULT_AI_CONFIG_FILE)


class TestPromptsDirSecurity(unittest.TestCase):
    """缺陷 #10（本轮）：prompts_dir 路径安全校验

    用户可在配置文件或环境变量中设置 prompts_dir 指向外置提示词目录。
    若不加校验，恶意或误配置的路径（如 /etc、C:\\Windows、~/.ssh）会导致
    get_prompt 读取任意文件。需校验路径必须在用户主目录或项目目录下，
    拒绝系统目录、路径遍历与敏感目录。
    """

    def setUp(self):
        """保存环境变量"""
        self._saved_env = dict(os.environ)
        os.environ.pop('TB_AI_PROMPTS_DIR', None)

    def tearDown(self):
        """还原环境变量"""
        os.environ.clear()
        os.environ.update(self._saved_env)

    # ---- _validate_prompts_dir 函数级测试 ----

    def test_none_prompts_dir_passes(self):
        """None 不抛错（视为未设置）"""
        from tb_risk.ai.config import _validate_prompts_dir
        result = _validate_prompts_dir(None)
        self.assertIsNone(result)

    def test_empty_string_prompts_dir_passes(self):
        """空字符串不抛错（视为未设置）"""
        from tb_risk.ai.config import _validate_prompts_dir
        result = _validate_prompts_dir('')
        self.assertIsNone(result)

    def test_relative_path_passes(self):
        """相对路径（如 prompts/）通过校验"""
        from tb_risk.ai.config import _validate_prompts_dir
        # 相对路径解析在 cwd 下，应通过
        result = _validate_prompts_dir('prompts')
        self.assertEqual(result, 'prompts')

    def test_user_home_subdir_passes(self):
        """用户主目录下的子目录通过校验"""
        from tb_risk.ai.config import _validate_prompts_dir
        home = os.path.expanduser('~')
        safe_path = os.path.join(home, '.tb_risk', 'prompts')
        result = _validate_prompts_dir(safe_path)
        self.assertEqual(result, safe_path)

    def test_path_traversal_rejected(self):
        """路径遍历（包含 ..）被拒绝"""
        from tb_risk.ai.config import _validate_prompts_dir
        with self.assertRaises(ValueError) as cm:
            _validate_prompts_dir('../../../etc/passwd')
        self.assertIn('prompts_dir', str(cm.exception))

    def test_system_dir_rejected(self):
        """系统目录被拒绝（跨平台）"""
        from tb_risk.ai.config import _validate_prompts_dir
        if os.name == 'nt':
            # Windows: C:\Windows\System32
            forbidden = r'C:\Windows\System32'
        else:
            # Unix: /etc
            forbidden = '/etc'
        with self.assertRaises(ValueError) as cm:
            _validate_prompts_dir(forbidden)
        self.assertIn('prompts_dir', str(cm.exception))

    def test_windows_program_files_rejected(self):
        """Windows: C:\\Program Files 被拒绝"""
        if os.name != 'nt':
            self.skipTest("Windows-only test")
        from tb_risk.ai.config import _validate_prompts_dir
        with self.assertRaises(ValueError):
            _validate_prompts_dir(r'C:\Program Files\prompts')

    def test_unix_var_rejected(self):
        """Unix: /var 被拒绝"""
        if os.name == 'nt':
            self.skipTest("Unix-only test")
        from tb_risk.ai.config import _validate_prompts_dir
        with self.assertRaises(ValueError):
            _validate_prompts_dir('/var/prompts')

    def test_ssh_dir_rejected(self):
        """敏感目录 .ssh 被拒绝（即使位于用户主目录下）"""
        from tb_risk.ai.config import _validate_prompts_dir
        home = os.path.expanduser('~')
        ssh_path = os.path.join(home, '.ssh', 'prompts')
        with self.assertRaises(ValueError) as cm:
            _validate_prompts_dir(ssh_path)
        self.assertIn('prompts_dir', str(cm.exception))

    def test_returns_input_path_when_valid(self):
        """合法路径原样返回（不强制绝对化）"""
        from tb_risk.ai.config import _validate_prompts_dir
        result = _validate_prompts_dir('my_prompts')
        self.assertEqual(result, 'my_prompts')

    # ---- 集成测试：from_user_config_file ----

    def test_from_user_config_file_rejects_system_dir(self):
        """from_user_config_file 中 prompts_dir 指向系统目录 → ValueError"""
        from tb_risk.ai.config import AIConfig
        if os.name == 'nt':
            forbidden = r'C:\Windows\System32'
        else:
            forbidden = '/etc'
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, 'ai_config.json')
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump({'prompts_dir': forbidden}, f)
            with self.assertRaises(ValueError) as cm:
                AIConfig.from_user_config_file(cfg_path)
            self.assertIn('prompts_dir', str(cm.exception))

    def test_from_user_config_file_accepts_safe_path(self):
        """from_user_config_file 中 prompts_dir 指向安全路径 → 正常加载"""
        from tb_risk.ai.config import AIConfig
        with tempfile.TemporaryDirectory() as tmp:
            safe_path = os.path.join(tmp, 'prompts')
            cfg_path = os.path.join(tmp, 'ai_config.json')
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump({'prompts_dir': safe_path}, f)
            cfg = AIConfig.from_user_config_file(cfg_path)
            self.assertEqual(cfg.prompts_dir, safe_path)

    def test_from_user_config_file_accepts_none_prompts_dir(self):
        """from_user_config_file 不设置 prompts_dir → 默认 None"""
        from tb_risk.ai.config import AIConfig
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, 'ai_config.json')
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump({'provider': 'deepseek'}, f)
            cfg = AIConfig.from_user_config_file(cfg_path)
            self.assertIsNone(cfg.prompts_dir)

    # ---- 集成测试：from_env ----

    def test_from_env_rejects_system_dir(self):
        """from_env 中 TB_AI_PROMPTS_DIR 指向系统目录 → ValueError"""
        from tb_risk.ai.config import AIConfig
        if os.name == 'nt':
            forbidden = r'C:\Windows\System32'
        else:
            forbidden = '/etc'
        os.environ['TB_AI_PROMPTS_DIR'] = forbidden
        with self.assertRaises(ValueError) as cm:
            AIConfig.from_env()
        self.assertIn('prompts_dir', str(cm.exception))

    def test_from_env_accepts_safe_path(self):
        """from_env 中 TB_AI_PROMPTS_DIR 指向安全路径 → 正常加载"""
        from tb_risk.ai.config import AIConfig
        with tempfile.TemporaryDirectory() as tmp:
            safe_path = os.path.join(tmp, 'prompts')
            os.environ['TB_AI_PROMPTS_DIR'] = safe_path
            cfg = AIConfig.from_env()
            self.assertEqual(cfg.prompts_dir, safe_path)


class TestAIConfigSaveToFile(unittest.TestCase):
    """AIConfig.save_to_file 测试（GUI AI 设置对话框回写配置）

    Layer 1: GUI AISettingsDialog 确定后调用 AIConfig.save_to_file 写入
    ~/.tb_risk/ai_config.json，要求：
    - 原子写入（先临时文件再 rename，防止崩溃损坏）
    - api_key 明文保存（不掩码），供 AI 客户端读取
    - 保存前校验 prompts_dir 路径安全
    - 校验失败时不写入文件
    - 自动创建父目录（如 ~/.tb_risk/ 不存在）
    """

    def setUp(self):
        """保存环境变量"""
        self._saved_env = dict(os.environ)
        # 清理 AI 相关环境变量，避免污染测试
        for key in ('TB_AI_PROVIDER', 'TB_AI_API_KEY', 'TB_AI_BASE_URL',
                    'TB_AI_MODEL', 'TB_AI_TIMEOUT', 'TB_AI_MAX_RETRIES',
                    'TB_AI_TEMPERATURE', 'TB_AI_LOCAL_MODEL',
                    'TB_AI_LOCAL_DEVICE', 'TB_AI_PROMPTS_DIR'):
            os.environ.pop(key, None)

    def tearDown(self):
        """还原环境变量"""
        os.environ.clear()
        os.environ.update(self._saved_env)

    def test_save_to_file_writes_valid_json(self):
        """save_to_file 写入有效 JSON 文件"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(provider='deepseek', api_key='sk-test123')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ai_config.json')
            cfg.save_to_file(path)
            self.assertTrue(os.path.exists(path))
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)
            self.assertEqual(data['provider'], 'deepseek')

    def test_save_to_file_preserves_api_key_plaintext(self):
        """save_to_file 明文保存 api_key（不掩码）

        to_dict() 会掩码 api_key（仅首尾2字符），但 save_to_file 必须明文保存，
        因为该文件供 AI 客户端读取使用。
        """
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(provider='deepseek', api_key='sk-abcdef123456')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ai_config.json')
            cfg.save_to_file(path)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 明文保存，不是掩码
            self.assertEqual(data['api_key'], 'sk-abcdef123456')
            self.assertNotIn('***', data['api_key'])

    def test_save_to_file_round_trip_with_from_user_config_file(self):
        """save_to_file → from_user_config_file 往返一致"""
        from tb_risk.ai.config import AIConfig
        original = AIConfig(
            provider='openai',
            api_key='sk-roundtrip-key',
            base_url='https://custom.example.com/v1',
            model='gpt-4o',
            timeout=120,
            max_retries=5,
            temperature=0.5,
            local_device='cuda',
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ai_config.json')
            original.save_to_file(path)
            loaded = AIConfig.from_user_config_file(path)
            self.assertEqual(loaded.provider, 'openai')
            self.assertEqual(loaded.api_key, 'sk-roundtrip-key')
            self.assertEqual(loaded.base_url, 'https://custom.example.com/v1')
            self.assertEqual(loaded.model, 'gpt-4o')
            self.assertEqual(loaded.timeout, 120)
            self.assertEqual(loaded.max_retries, 5)
            self.assertEqual(loaded.temperature, 0.5)
            self.assertEqual(loaded.local_device, 'cuda')

    def test_save_to_file_creates_parent_directory(self):
        """save_to_file 自动创建父目录（如 ~/.tb_risk/ 不存在）"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(provider='deepseek', api_key='sk-test')
        with tempfile.TemporaryDirectory() as tmp:
            # 父目录尚不存在
            nested_dir = os.path.join(tmp, 'sub1', 'sub2')
            path = os.path.join(nested_dir, 'ai_config.json')
            self.assertFalse(os.path.exists(nested_dir))
            cfg.save_to_file(path)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(nested_dir))

    def test_save_to_file_atomic_write(self):
        """save_to_file 原子写入（先临时文件再 rename）

        写入完成后，目录中不应残留 .tmp 文件。
        """
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(provider='deepseek', api_key='sk-test')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ai_config.json')
            cfg.save_to_file(path)
            # 目标文件存在
            self.assertTrue(os.path.exists(path))
            # 不残留临时文件
            files = os.listdir(tmp)
            self.assertEqual(files, ['ai_config.json'])

    def test_save_to_file_atomic_replace_existing(self):
        """save_to_file 原子替换已存在文件"""
        from tb_risk.ai.config import AIConfig
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ai_config.json')
            # 先写入旧内容
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'old': 'data'}, f)
            # 再覆盖写入
            cfg = AIConfig(provider='deepseek', api_key='sk-new')
            cfg.save_to_file(path)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertEqual(data['api_key'], 'sk-new')
            self.assertNotIn('old', data)

    def test_save_to_file_rejects_system_prompts_dir(self):
        """save_to_file 中 prompts_dir 指向系统目录 → ValueError"""
        from tb_risk.ai.config import AIConfig
        if os.name == 'nt':
            forbidden = r'C:\Windows\System32'
        else:
            forbidden = '/etc'
        cfg = AIConfig(provider='deepseek', api_key='sk-test',
                       prompts_dir=forbidden)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ai_config.json')
            with self.assertRaises(ValueError) as cm:
                cfg.save_to_file(path)
            self.assertIn('prompts_dir', str(cm.exception))

    def test_save_to_file_does_not_write_on_validation_error(self):
        """校验失败时不写入文件（保持原文件不变）"""
        from tb_risk.ai.config import AIConfig
        if os.name == 'nt':
            forbidden = r'C:\Windows\System32'
        else:
            forbidden = '/etc'
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ai_config.json')
            # 先写入旧内容
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'old': 'data'}, f)
            # 尝试保存一个 prompts_dir 非法的配置
            cfg = AIConfig(provider='deepseek', api_key='sk-test',
                           prompts_dir=forbidden)
            with self.assertRaises(ValueError):
                cfg.save_to_file(path)
            # 原文件未被修改
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertEqual(data, {'old': 'data'})

    def test_save_to_file_uses_default_path_when_none(self):
        """path=None 时使用 DEFAULT_AI_CONFIG_FILE

        使用 monkeypatch 修改 DEFAULT_AI_CONFIG_FILE 到临时目录，避免污染真实配置。
        """
        from tb_risk.ai.config import AIConfig, DEFAULT_AI_CONFIG_FILE
        with tempfile.TemporaryDirectory() as tmp:
            fake_default = os.path.join(tmp, 'ai_config.json')
            with mock.patch('tb_risk.ai.config.DEFAULT_AI_CONFIG_FILE',
                            fake_default):
                cfg = AIConfig(provider='deepseek', api_key='sk-default')
                returned_path = cfg.save_to_file(None)
                self.assertEqual(returned_path, fake_default)
                self.assertTrue(os.path.exists(fake_default))

    def test_save_to_file_returns_path(self):
        """save_to_file 返回实际写入的文件路径"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(provider='deepseek', api_key='sk-test')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'sub', 'ai_config.json')
            returned = cfg.save_to_file(path)
            self.assertEqual(returned, path)

    def test_save_to_file_preserves_all_fields(self):
        """所有字段都被保存（含 local_model_path、local_device、prompts_dir）"""
        from tb_risk.ai.config import AIConfig
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = os.path.join(tmp, 'my_prompts')
            os.makedirs(prompts_dir, exist_ok=True)
            cfg = AIConfig(
                provider='zhipu',
                api_key='sk-all-fields',
                base_url='https://open.bigmodel.cn/api/paas/v4',
                model='glm-4-flash',
                timeout=90,
                max_retries=7,
                temperature=0.7,
                local_model_path='/path/to/local/model',
                local_device='cuda',
                prompts_dir=prompts_dir,
            )
            path = os.path.join(tmp, 'ai_config.json')
            cfg.save_to_file(path)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertEqual(data['provider'], 'zhipu')
            self.assertEqual(data['api_key'], 'sk-all-fields')
            self.assertEqual(data['base_url'],
                             'https://open.bigmodel.cn/api/paas/v4')
            self.assertEqual(data['model'], 'glm-4-flash')
            self.assertEqual(data['timeout'], 90)
            self.assertEqual(data['max_retries'], 7)
            self.assertEqual(data['temperature'], 0.7)
            self.assertEqual(data['local_model_path'], '/path/to/local/model')
            self.assertEqual(data['local_device'], 'cuda')
            self.assertEqual(data['prompts_dir'], prompts_dir)

    def test_save_to_file_preserves_none_api_key(self):
        """api_key=None 时正确保存为 null（本地模式场景）"""
        from tb_risk.ai.config import AIConfig
        cfg = AIConfig(provider='deepseek', api_key=None,
                       local_model_path='/path/to/model')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ai_config.json')
            cfg.save_to_file(path)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertIsNone(data['api_key'])
            self.assertEqual(data['local_model_path'], '/path/to/model')


if __name__ == '__main__':
    unittest.main()
