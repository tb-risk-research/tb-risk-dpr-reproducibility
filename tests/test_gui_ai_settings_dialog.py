#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 测试套件 — AISettingsDialog AI 设置对话框 (tb_risk.gui.ai_settings_dialog)

Layer 3 + Layer 7: 测试 AISettingsDialog 提供 GUI 内 AI API 配置入口。

测试策略：
- TestAISettingsDialogStructure：不依赖 Tk，仅验证类结构（方法/属性存在）
- TestAISettingsDialogInteraction：需要真实 Tk root，无 display 时跳过
  * 对话框打开时正确加载当前 AppConfig 字段
  * 确定按钮保存 AppConfig（非敏感字段）到 settings.json
  * 确定按钮保存 AIConfig（含 api_key）到 ai_config.json
  * 取消按钮不修改任何配置
  * prompts_dir 指向系统目录时保存被拒绝并显示错误
- TestAISettingsDialogCallbacks：用 mock 验证回调逻辑（不依赖 Tk 渲染）

设计原则：
- 所有测试不依赖真实 API Key
- 不写入真实 ~/.tb_risk/（用 mock.patch 重定向）
- 控件交互用 update_idletasks + event_generate 模拟
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


# ==============================================================================
# 模块加载工具
# ==============================================================================

def _load_ai_settings_dialog_module():
    """直接加载 gui/ai_settings_dialog.py 模块

    绕过 gui/__init__.py 的 Tkinter 依赖链，仅加载 AISettingsDialog 类定义。
    """
    import importlib.util
    module_path = os.path.join(_PROJECT_ROOT, 'gui', 'ai_settings_dialog.py')
    spec = importlib.util.spec_from_file_location(
        'gui.ai_settings_dialog', module_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ==============================================================================
# 类结构测试（不依赖 Tk）
# ==============================================================================

class TestAISettingsDialogStructure(unittest.TestCase):
    """AISettingsDialog 类结构验证（确保方法/属性存在）"""

    def test_module_imports_successfully(self):
        """ai_settings_dialog 模块可被加载"""
        m = _load_ai_settings_dialog_module()
        self.assertTrue(hasattr(m, 'AISettingsDialog'))

    def test_class_inherits_toplevel(self):
        """AISettingsDialog 继承 tk.Toplevel"""
        import tkinter as tk
        m = _load_ai_settings_dialog_module()
        self.assertTrue(issubclass(m.AISettingsDialog, tk.Toplevel))

    def test_class_has_required_methods(self):
        """AISettingsDialog 定义了所有必需方法"""
        m = _load_ai_settings_dialog_module()
        for method_name in ['__init__', '_on_ok', '_on_cancel',
                            '_save', '_validate']:
            self.assertTrue(hasattr(m.AISettingsDialog, method_name),
                            f"缺失方法: {method_name}")

    def test_class_has_connection_test_methods(self):
        """AISettingsDialog 定义了连接测试相关方法（Layer 6）"""
        m = _load_ai_settings_dialog_module()
        for method_name in ['_on_test_connection',
                            '_test_connection_worker',
                            '_update_test_status']:
            self.assertTrue(hasattr(m.AISettingsDialog, method_name),
                            f"缺失方法: {method_name}")

    def test_class_has_import_export_methods(self):
        """AISettingsDialog 定义了导入/导出相关方法（Layer 10）"""
        m = _load_ai_settings_dialog_module()
        for method_name in ['_on_export_config', '_on_import_config',
                            '_on_provider_changed', '_update_provider_ui']:
            self.assertTrue(hasattr(m.AISettingsDialog, method_name),
                            f"缺失方法: {method_name}")

    def test_class_has_browse_methods(self):
        """AISettingsDialog 定义了浏览按钮回调方法"""
        m = _load_ai_settings_dialog_module()
        for method_name in ['_browse_local_model_path',
                            '_browse_prompts_dir']:
            self.assertTrue(hasattr(m.AISettingsDialog, method_name),
                            f"缺失方法: {method_name}")


# ==============================================================================
# 交互测试（需要真实 Tk root，无 display 时跳过）
# ==============================================================================

class TestAISettingsDialogInteraction(unittest.TestCase):
    """AISettingsDialog 交互测试

    需要真实 Tk root 来创建 Toplevel 对话框。
    无 display 环境（如 CI）跳过整个测试类。
    """

    @classmethod
    def setUpClass(cls):
        """尝试创建 Tk root，失败则跳过整个测试类"""
        try:
            import tkinter as tk
            cls._tk = tk
            cls._root = tk.Tk()
            cls._root.withdraw()  # 不显示窗口
        except Exception:
            cls._tk = None
            cls._root = None

    @classmethod
    def tearDownClass(cls):
        if cls._root is not None:
            try:
                cls._root.destroy()
            except Exception:
                pass

    def setUp(self):
        if self._tk is None or self._root is None:
            self.skipTest("Tkinter display not available")
        # 保存环境变量
        self._saved_env = dict(os.environ)
        for key in ('TB_AI_PROVIDER', 'TB_AI_API_KEY', 'TB_AI_BASE_URL',
                    'TB_AI_MODEL', 'TB_AI_TIMEOUT', 'TB_AI_MAX_RETRIES',
                    'TB_AI_TEMPERATURE', 'TB_AI_LOCAL_MODEL',
                    'TB_AI_LOCAL_DEVICE', 'TB_AI_PROMPTS_DIR'):
            os.environ.pop(key, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)

    def _make_app_config(self, **kwargs):
        """构造测试用 AppConfig"""
        from tb_risk.gui.app_config import AppConfig
        defaults = {
            'ai_enabled': True,
            'ai_provider': 'deepseek',
            'ai_base_url': '',
            'ai_model': '',
            'ai_local_model_path': '',
            'ai_timeout': 60,
            'ai_max_retries': 3,
            'ai_temperature': 0.2,
            'ai_prompts_dir': '',
        }
        defaults.update(kwargs)
        return AppConfig(**defaults)

    def _create_dialog(self, app_config, on_save_callback=None):
        """创建 AISettingsDialog 实例"""
        m = _load_ai_settings_dialog_module()
        dialog = m.AISettingsDialog(
            self._root, app_config, on_save_callback=on_save_callback)
        dialog.withdraw()  # 不显示
        self._root.update_idletasks()
        return dialog

    # ------------------------------------------------------------------
    # 字段加载测试
    # ------------------------------------------------------------------

    def test_dialog_loads_app_config_fields(self):
        """对话框打开时正确加载当前 AppConfig 字段"""
        cfg = self._make_app_config(
            ai_enabled=True,
            ai_provider='openai',
            ai_base_url='https://custom.example.com/v1',
            ai_model='gpt-4o',
            ai_timeout=120,
            ai_max_retries=5,
            ai_temperature=0.7,
        )
        dialog = self._create_dialog(cfg)
        # 检查控件值与 AppConfig 一致
        self.assertTrue(dialog.enabled_var.get())
        self.assertEqual(dialog.provider_var.get(), 'openai')
        self.assertEqual(dialog.base_url_var.get(),
                         'https://custom.example.com/v1')
        self.assertEqual(dialog.model_var.get(), 'gpt-4o')
        self.assertEqual(dialog.timeout_var.get(), 120)
        self.assertEqual(dialog.max_retries_var.get(), 5)
        self.assertAlmostEqual(float(dialog.temperature_var.get()), 0.7)
        dialog.destroy()

    def test_dialog_loads_api_key_from_ai_config_file(self):
        """对话框打开时从 ai_config.json 加载 api_key

        方案 A：api_key 不在 AppConfig 中，需从 ~/.tb_risk/ai_config.json 读取。
        """
        from tb_risk.ai.config import AIConfig
        cfg = self._make_app_config()
        with tempfile.TemporaryDirectory() as tmp:
            fake_ai_config = os.path.join(tmp, 'ai_config.json')
            with open(fake_ai_config, 'w', encoding='utf-8') as f:
                json.dump({'api_key': 'sk-from-file-12345'}, f)
            with mock.patch(
                    'tb_risk.ai.config.DEFAULT_AI_CONFIG_FILE',
                    fake_ai_config):
                dialog = self._create_dialog(cfg)
                self.assertEqual(dialog.api_key_var.get(),
                                 'sk-from-file-12345')
                dialog.destroy()

    def test_dialog_api_key_masked_by_default(self):
        """API Key 输入框默认掩码显示（show='*'）"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        # Entry 的 show 属性应为 '*'
        self.assertEqual(dialog.api_key_entry.cget('show'), '*')
        dialog.destroy()

    def test_provider_combo_has_five_options(self):
        """服务商下拉有 5 个选项（deepseek/openai/dashscope/zhipu/moonshot）"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        values = dialog.provider_combo.cget('values')
        self.assertIn('deepseek', values)
        self.assertIn('openai', values)
        self.assertIn('dashscope', values)
        self.assertIn('zhipu', values)
        self.assertIn('moonshot', values)
        dialog.destroy()

    # ------------------------------------------------------------------
    # 保存测试（用 _save 代替 _on_ok 避免 destroy 引发的事件循环挂起）
    # ------------------------------------------------------------------

    def test_ok_button_saves_app_config_non_sensitive_fields(self):
        """确定按钮保存 AppConfig（非敏感字段）到 settings.json"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        # 修改控件值
        dialog.api_key_var.set('sk-test')
        dialog.provider_var.set('zhipu')
        dialog.model_var.set('glm-4-flash')
        dialog.timeout_var.set(90)
        # Mock AppConfig.save 与 AIConfig.save_to_file 避免写入真实文件
        with mock.patch.object(cfg, 'save') as mock_save, \
             mock.patch('tb_risk.ai.config.AIConfig.save_to_file'):
            dialog._save()
            mock_save.assert_called_once()
        # 验证 AppConfig 字段被更新
        self.assertEqual(cfg.ai_provider, 'zhipu')
        self.assertEqual(cfg.ai_model, 'glm-4-flash')
        self.assertEqual(cfg.ai_timeout, 90)
        dialog.destroy()

    def test_ok_button_saves_ai_config_with_api_key_to_ai_config_file(self):
        """确定按钮保存 AIConfig（含 api_key）到 ai_config.json"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        dialog.api_key_var.set('sk-test-key-67890')
        dialog.provider_var.set('deepseek')
        # Mock AppConfig.save 与 AIConfig.save_to_file 避免写入真实文件
        with mock.patch.object(cfg, 'save'), \
             mock.patch('tb_risk.ai.config.AIConfig.save_to_file') \
                as mock_ai_save:
            result = dialog._save()
            mock_ai_save.assert_called_once()
            self.assertTrue(result)
        dialog.destroy()

    def test_ok_button_invokes_on_save_callback(self):
        """确定按钮保存成功后调用 on_save_callback"""
        cfg = self._make_app_config()
        callback_called = [False]

        def on_save():
            callback_called[0] = True

        dialog = self._create_dialog(cfg, on_save_callback=on_save)
        dialog.api_key_var.set('sk-test')
        with mock.patch.object(cfg, 'save'), \
             mock.patch('tb_risk.ai.config.AIConfig.save_to_file'):
            dialog._save()
        self.assertTrue(callback_called[0])
        dialog.destroy()

    def test_ok_button_sets_result_true(self):
        """确定按钮设置 self.result = True"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        dialog.api_key_var.set('sk-test')
        self.assertFalse(dialog.result)
        with mock.patch.object(cfg, 'save'), \
             mock.patch('tb_risk.ai.config.AIConfig.save_to_file'):
            dialog._save()
        self.assertTrue(dialog.result)
        dialog.destroy()

    # ------------------------------------------------------------------
    # 取消测试（_on_cancel 调用 destroy，但取消不涉及保存逻辑，
    #          destroy 后访问 result 属性仍安全）
    # ------------------------------------------------------------------

    def test_cancel_button_sets_result_false(self):
        """取消按钮设置 self.result = False"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        # mock destroy 避免 Tk 事件循环问题
        with mock.patch.object(dialog, 'destroy'), \
             mock.patch.object(dialog, 'grab_release'):
            dialog._on_cancel()
        self.assertFalse(dialog.result)
        dialog.destroy()

    def test_cancel_button_does_not_save(self):
        """取消按钮不保存任何配置

        用 mock 验证 _on_cancel 不调用 save 方法（避免 destroy 复杂性）。
        """
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        # mock destroy 避免 Tk 事件循环问题
        with mock.patch.object(dialog, 'destroy'), \
             mock.patch.object(dialog, 'grab_release'), \
             mock.patch.object(cfg, 'save') as mock_app_save, \
             mock.patch('tb_risk.ai.config.AIConfig.save_to_file') \
                as mock_ai_save:
            dialog._on_cancel()
            mock_app_save.assert_not_called()
            mock_ai_save.assert_not_called()
        self.assertFalse(dialog.result)
        dialog.destroy()

    def test_cancel_button_does_not_invoke_callback(self):
        """取消按钮不调用 on_save_callback"""
        cfg = self._make_app_config()
        callback_called = [False]

        def on_save():
            callback_called[0] = True

        dialog = self._create_dialog(cfg, on_save_callback=on_save)
        with mock.patch.object(dialog, 'destroy'), \
             mock.patch.object(dialog, 'grab_release'):
            dialog._on_cancel()
        self.assertFalse(callback_called[0])
        dialog.destroy()

    # ------------------------------------------------------------------
    # 校验测试（用 _save 代替 _on_ok，避免 destroy）
    # ------------------------------------------------------------------

    def test_invalid_prompts_dir_shows_error_and_does_not_save(self):
        """prompts_dir 指向系统目录时显示错误且不保存"""
        if os.name == 'nt':
            forbidden = r'C:\Windows\System32'
        else:
            forbidden = '/etc'
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        dialog.prompts_dir_var.set(forbidden)
        with mock.patch.object(cfg, 'save') as mock_app_save, \
             mock.patch('tb_risk.ai.config.AIConfig.save_to_file') \
                as mock_ai_save:
            result = dialog._save()
            mock_app_save.assert_not_called()
            mock_ai_save.assert_not_called()
        self.assertFalse(result)
        # 错误标签应显示错误信息
        self.assertTrue(dialog.error_label.cget('text'))
        self.assertIn('prompts_dir', dialog.error_label.cget('text'))
        # result 仍为 False（未成功保存）
        self.assertFalse(dialog.result)
        dialog.destroy()

    def test_valid_prompts_dir_saves_successfully(self):
        """合法 prompts_dir 保存成功"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        dialog.api_key_var.set('sk-test')
        with tempfile.TemporaryDirectory() as tmp:
            safe_prompts = os.path.join(tmp, 'my_prompts')
            os.makedirs(safe_prompts, exist_ok=True)
            dialog.prompts_dir_var.set(safe_prompts)
            with mock.patch.object(cfg, 'save'), \
                 mock.patch('tb_risk.ai.config.AIConfig.save_to_file'):
                result = dialog._save()
            self.assertTrue(result)
            # 错误标签应为空
            self.assertEqual(dialog.error_label.cget('text'), '')
            self.assertTrue(dialog.result)
        dialog.destroy()

    # ------------------------------------------------------------------
    # 连接测试（Layer 6）
    # ------------------------------------------------------------------

    def test_test_connection_button_exists(self):
        """对话框包含"测试连接"按钮"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        self.assertTrue(hasattr(dialog, 'test_btn'))
        self.assertEqual(dialog.test_btn.cget('text'), '测试连接')
        dialog.destroy()

    def test_test_status_label_exists(self):
        """对话框包含测试状态标签"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        self.assertTrue(hasattr(dialog, 'test_status_label'))
        self.assertEqual(dialog.test_status_label.cget('text'), '')
        dialog.destroy()

    def test_test_connection_warns_when_no_api_key_and_no_local(self):
        """无 api_key 且无本地模型路径时显示警告"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        # 清空 api_key 与本地模型路径
        dialog.api_key_var.set('')
        dialog.local_model_path_var.set('')
        dialog._on_test_connection()
        # 应显示警告（不启动后台线程）
        self.assertIn('API Key', dialog.test_status_label.cget('text'))
        # 测试按钮应仍可用（未启动测试）— 用 str() 兼容 Tk 字符串对象
        self.assertEqual(str(dialog.test_btn.cget('state')), 'normal')
        dialog.destroy()

    def test_test_connection_worker_shows_success(self):
        """_test_connection_worker 成功时显示绿色"连接成功"消息"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        # Mock make_client 返回一个 mock client
        mock_client = mock.MagicMock()
        mock_client.chat.return_value = 'pong'
        # _update_test_status 用 after(0, ...) 调度，mock after 直接执行
        queued_updates = []

        def fake_after(ms, func):
            queued_updates.append(func)

        dialog.after = fake_after
        with mock.patch('tb_risk.ai.client.make_client',
                        return_value=mock_client):
            dialog._test_connection_worker(
                mock.MagicMock(), 'deepseek-chat')
        # 验证 after 被调用，且执行后状态为成功
        self.assertEqual(len(queued_updates), 1)
        queued_updates[0]()
        self.assertIn('连接成功', dialog.test_status_label.cget('text'))
        self.assertIn('deepseek-chat', dialog.test_status_label.cget('text'))
        dialog.destroy()

    def test_test_connection_worker_shows_failure(self):
        """_test_connection_worker 失败时显示红色错误信息"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        # Mock make_client 返回一个会抛异常的 client
        mock_client = mock.MagicMock()
        mock_client.chat.side_effect = RuntimeError('401 Unauthorized')
        queued_updates = []

        def fake_after(ms, func):
            queued_updates.append(func)

        dialog.after = fake_after
        with mock.patch('tb_risk.ai.client.make_client',
                        return_value=mock_client):
            dialog._test_connection_worker(
                mock.MagicMock(), 'gpt-4o')
        self.assertEqual(len(queued_updates), 1)
        queued_updates[0]()
        self.assertIn('连接失败', dialog.test_status_label.cget('text'))
        self.assertIn('401', dialog.test_status_label.cget('text'))
        dialog.destroy()

    def test_test_connection_worker_handles_no_client(self):
        """make_client 返回 None 时显示配置无效提示"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        queued_updates = []

        def fake_after(ms, func):
            queued_updates.append(func)

        dialog.after = fake_after
        with mock.patch('tb_risk.ai.client.make_client',
                        return_value=None):
            dialog._test_connection_worker(
                mock.MagicMock(), 'any-model')
        self.assertEqual(len(queued_updates), 1)
        queued_updates[0]()
        self.assertIn('配置无效', dialog.test_status_label.cget('text'))
        dialog.destroy()

    # ------------------------------------------------------------------
    # 服务商切换交互测试（#5）
    # ------------------------------------------------------------------

    def test_provider_switch_to_local_disables_api_key(self):
        """切换到 local 时禁用 API Key 与 Base URL 输入框"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        # 切换到 local
        dialog.provider_var.set('local')
        dialog._update_provider_ui()
        self.assertEqual(str(dialog.api_key_entry.cget('state')), 'disabled')
        self.assertEqual(str(dialog.api_key_show_btn.cget('state')), 'disabled')
        self.assertEqual(str(dialog.base_url_entry.cget('state')), 'disabled')
        dialog.destroy()

    def test_provider_switch_to_local_enables_local_model_path(self):
        """切换到 local 时启用本地模型路径输入框"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        dialog.provider_var.set('local')
        dialog._update_provider_ui()
        self.assertEqual(str(dialog.local_model_path_entry.cget('state')), 'normal')
        self.assertEqual(str(dialog.local_model_browse_btn.cget('state')), 'normal')
        dialog.destroy()

    def test_provider_switch_to_cloud_restores_api_key(self):
        """切换回云端提供商时恢复 API Key 输入框"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        # 先切换到 local
        dialog.provider_var.set('local')
        dialog._update_provider_ui()
        # 再切换回 deepseek
        dialog.provider_var.set('deepseek')
        dialog._update_provider_ui()
        self.assertEqual(str(dialog.api_key_entry.cget('state')), 'normal')
        self.assertEqual(str(dialog.base_url_entry.cget('state')), 'normal')
        self.assertEqual(str(dialog.local_model_path_entry.cget('state')), 'disabled')
        dialog.destroy()

    def test_validate_rejects_local_without_model_path(self):
        """本地模式下缺少本地模型路径时校验失败"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        dialog.provider_var.set('local')
        dialog.local_model_path_var.set('')
        ok, err = dialog._validate()
        self.assertFalse(ok)
        self.assertIn('本地模型路径', err)
        dialog.destroy()

    def test_validate_rejects_cloud_without_api_key(self):
        """云端模式下缺少 API Key 时校验失败"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        dialog.provider_var.set('deepseek')
        dialog.api_key_var.set('')
        ok, err = dialog._validate()
        self.assertFalse(ok)
        self.assertIn('API Key', err)
        dialog.destroy()

    # ------------------------------------------------------------------
    # ai_enabled=False 保存行为测试（#6）
    # ------------------------------------------------------------------

    def test_save_with_ai_disabled_skips_ai_config(self):
        """ai_enabled=False 时跳过 AIConfig.save_to_file"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        dialog.api_key_var.set('sk-test')
        dialog.enabled_var.set(False)
        with mock.patch.object(cfg, 'save') as mock_app_save, \
             mock.patch('tb_risk.ai.config.AIConfig.save_to_file') \
                as mock_ai_save:
            result = dialog._save()
        self.assertTrue(result)
        mock_app_save.assert_called_once()
        mock_ai_save.assert_not_called()
        self.assertFalse(cfg.ai_enabled)
        dialog.destroy()

    def test_save_with_ai_disabled_still_saves_app_config(self):
        """ai_enabled=False 时仍保存 AppConfig（保留非敏感字段）"""
        cfg = self._make_app_config()
        dialog = self._create_dialog(cfg)
        dialog.api_key_var.set('sk-test')
        dialog.enabled_var.set(False)
        dialog.provider_var.set('openai')
        with mock.patch.object(cfg, 'save') as mock_app_save, \
             mock.patch('tb_risk.ai.config.AIConfig.save_to_file'):
            dialog._save()
        self.assertFalse(cfg.ai_enabled)
        self.assertEqual(cfg.ai_provider, 'openai')
        mock_app_save.assert_called_once()
        dialog.destroy()


# ==============================================================================
# 回调逻辑测试（不依赖 Tk 渲染，用 mock 验证）
# ==============================================================================

class TestAISettingsDialogSaveLogic(unittest.TestCase):
    """AISettingsDialog 保存逻辑测试

    不创建真实 Tk root，直接 mock 控件变量验证 _save 逻辑。
    """

    def test_save_builds_app_config_with_all_fields(self):
        """_save 构造的 AppConfig 包含所有非敏感字段"""
        m = _load_ai_settings_dialog_module()
        from tb_risk.gui.app_config import AppConfig

        # 用 mock 创建对话框实例（绕过 __init__）
        dialog = m.AISettingsDialog.__new__(m.AISettingsDialog)
        dialog.enabled_var = mock.MagicMock()
        dialog.enabled_var.get.return_value = True
        dialog.provider_var = mock.MagicMock()
        dialog.provider_var.get.return_value = 'openai'
        dialog.base_url_var = mock.MagicMock()
        dialog.base_url_var.get.return_value = 'https://custom/v1'
        dialog.model_var = mock.MagicMock()
        dialog.model_var.get.return_value = 'gpt-4o'
        dialog.local_model_path_var = mock.MagicMock()
        dialog.local_model_path_var.get.return_value = ''
        dialog.timeout_var = mock.MagicMock()
        dialog.timeout_var.get.return_value = 90
        dialog.max_retries_var = mock.MagicMock()
        dialog.max_retries_var.get.return_value = 5
        dialog.temperature_var = mock.MagicMock()
        dialog.temperature_var.get.return_value = 0.5
        dialog.prompts_dir_var = mock.MagicMock()
        dialog.prompts_dir_var.get.return_value = ''
        dialog.api_key_var = mock.MagicMock()
        dialog.api_key_var.get.return_value = 'sk-logic-test'
        # error_label mock（_save 中会调用 configure）
        dialog.error_label = mock.MagicMock()

        original_cfg = AppConfig()
        dialog._app_config = original_cfg
        dialog._ai_config_file = None  # 使用默认
        dialog._on_save_callback = None

        with mock.patch.object(original_cfg, 'save'), \
             mock.patch('tb_risk.ai.config.AIConfig.save_to_file'):
            result = dialog._save()

        self.assertTrue(result)
        # 验证 AppConfig 字段被更新
        self.assertTrue(original_cfg.ai_enabled)
        self.assertEqual(original_cfg.ai_provider, 'openai')
        self.assertEqual(original_cfg.ai_base_url, 'https://custom/v1')
        self.assertEqual(original_cfg.ai_model, 'gpt-4o')
        self.assertEqual(original_cfg.ai_timeout, 90)
        self.assertEqual(original_cfg.ai_max_retries, 5)
        self.assertAlmostEqual(original_cfg.ai_temperature, 0.5)

    def test_save_returns_false_on_validation_error(self):
        """_save 校验失败时返回 False"""
        m = _load_ai_settings_dialog_module()

        dialog = m.AISettingsDialog.__new__(m.AISettingsDialog)
        dialog.enabled_var = mock.MagicMock()
        dialog.enabled_var.get.return_value = True
        dialog.provider_var = mock.MagicMock()
        dialog.provider_var.get.return_value = 'deepseek'
        dialog.base_url_var = mock.MagicMock()
        dialog.base_url_var.get.return_value = ''
        dialog.model_var = mock.MagicMock()
        dialog.model_var.get.return_value = ''
        dialog.local_model_path_var = mock.MagicMock()
        dialog.local_model_path_var.get.return_value = ''
        dialog.timeout_var = mock.MagicMock()
        dialog.timeout_var.get.return_value = 60
        dialog.max_retries_var = mock.MagicMock()
        dialog.max_retries_var.get.return_value = 3
        dialog.temperature_var = mock.MagicMock()
        dialog.temperature_var.get.return_value = 0.2
        dialog.prompts_dir_var = mock.MagicMock()
        # 系统目录 → 校验失败
        if os.name == 'nt':
            dialog.prompts_dir_var.get.return_value = r'C:\Windows\System32'
        else:
            dialog.prompts_dir_var.get.return_value = '/etc'
        dialog.api_key_var = mock.MagicMock()
        dialog.api_key_var.get.return_value = 'sk-test'
        # error_label mock（_save 校验失败时会调用 configure 显示错误）
        dialog.error_label = mock.MagicMock()

        from tb_risk.gui.app_config import AppConfig
        original_cfg = AppConfig()
        dialog._app_config = original_cfg
        dialog._ai_config_file = None
        dialog._on_save_callback = None

        with mock.patch.object(original_cfg, 'save') as mock_save, \
             mock.patch('tb_risk.ai.config.AIConfig.save_to_file') \
                as mock_ai_save:
            result = dialog._save()

        self.assertFalse(result)
        mock_save.assert_not_called()
        mock_ai_save.assert_not_called()
        # error_label.configure 被调用显示错误信息
        dialog.error_label.configure.assert_called_once()
        # 错误信息包含 prompts_dir
        call_args = dialog.error_label.configure.call_args
        self.assertIn('prompts_dir', call_args[1].get('text', '') or
                      (call_args[0][0] if call_args[0] else '') or
                      call_args[1].get('text', ''))


if __name__ == '__main__':
    unittest.main()
