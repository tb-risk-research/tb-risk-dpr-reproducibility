#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 设置对话框 — GUI 内 AI API 配置入口

Layer 3: 提供 GUI 内完整的 AI 配置闭环，用户无需手动编辑
~/.tb_risk/ai_config.json 或设置环境变量。

设计要点：
- 仿 ContactEditDialog 风格（tk.Toplevel + transient + grab_set + grid 布局）
- 方案 A：AppConfig 保存非敏感字段（ai_enabled/ai_provider/...）到 settings.json，
  AIConfig 保存 api_key 到 ai_config.json，避免随项目配置分享泄露
- 保存前调用 _validate_prompts_dir 校验路径安全，校验失败时在对话框显示错误
- 确定按钮成功后调用 on_save_callback（供调用方刷新 AI 客户端缓存）

控件：
- 启用 AI 复选框
- 服务商下拉（deepseek/openai/dashscope/zhipu/moonshot）
- API Key 输入（show='*' + 显示/隐藏切换）
- Base URL / 模型名称 / 本地模型路径（带浏览按钮）
- 超时 / 最大重试 / 温度 Spinbox
- 提示词目录（带浏览按钮）
- 错误标签（红色，校验失败时显示）
- 确定 / 取消按钮
"""

import logging
import os
import json
import tkinter as tk
from tkinter import ttk, filedialog

LOGGER = logging.getLogger(__name__)

from tb_risk.gui.app_config import AppConfig
from tb_risk.ai.config import (AIConfig, PROVIDERS, DEFAULT_AI_CONFIG_FILE,
                                _validate_prompts_dir)


class AISettingsDialog(tk.Toplevel):
    """AI 设置对话框

    提供完整的 AI 配置 UI 入口，确定时保存 AppConfig（非敏感字段）与
    AIConfig（含 api_key）到各自的配置文件。

    Args:
        parent: 父窗口（tk.Tk 或 tk.Toplevel）
        app_config: AppConfig 实例（当前配置，将被原地更新）
        on_save_callback: 保存成功后的回调函数（无参数），供调用方
            刷新 AI 客户端缓存等
        ai_config_file: AI 配置文件路径，None 时使用 DEFAULT_AI_CONFIG_FILE
            （测试时可重定向到临时文件）

    Attributes:
        result: bool — True 表示用户点确定并保存成功，False 表示取消或校验失败
    """

    def __init__(self, parent, app_config, on_save_callback=None,
                 ai_config_file=None):
        super().__init__(parent)
        self.result = False
        self._app_config = app_config
        self._on_save_callback = on_save_callback
        self._ai_config_file = ai_config_file or DEFAULT_AI_CONFIG_FILE

        self.transient(parent)
        self.grab_set()
        self.title("AI 设置")

        # 允许调整大小 + 设置最小尺寸（高 DPI/小屏幕友好）
        self.resizable(True, True)
        self.minsize(480, 560)

        # 从 AIConfig 加载 api_key（方案 A：api_key 不在 AppConfig 中）
        self._ai_config_load_error = None
        try:
            self._ai_config = AIConfig.from_user_config_file(
                self._ai_config_file)
        except (FileNotFoundError, json.JSONDecodeError,
                TypeError, ValueError) as e:
            self._ai_config = AIConfig()
            self._ai_config_load_error = (
                f"AI 配置文件加载失败：{type(e).__name__}: {e}")
        except OSError as e:
            self._ai_config = AIConfig()
            self._ai_config_load_error = (
                f"AI 配置文件读取失败：{e}")

        self._build_ui()
        self._load_fields()

        # 若配置文件加载失败，在 error_label 中显示具体错误
        if self._ai_config_load_error:
            self.error_label.configure(
                text=self._ai_config_load_error, foreground='orange')

        # 绑定回车键触发确定，Esc 触发取消
        self.bind('<Return>', lambda e: self._on_ok())
        self.bind('<Escape>', lambda e: self._on_cancel())

    # ==================================================================
    # UI 构建
    # ==================================================================

    def _build_ui(self):
        """构建对话框 UI"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill='both', expand=True)
        main_frame.columnconfigure(1, weight=1)

        row = 0

        # --- 启用 AI 复选框 ---
        self.enabled_var = tk.BooleanVar()
        ttk.Checkbutton(main_frame, text="启用 AI 助手",
                        variable=self.enabled_var).grid(
            row=row, column=0, columnspan=3, sticky='w', padx=5, pady=3)
        row += 1

        # --- 服务商下拉 ---
        ttk.Label(main_frame, text="服务商:").grid(
            row=row, column=0, sticky='e', padx=5, pady=3)
        self.provider_var = tk.StringVar()
        self.provider_combo = ttk.Combobox(
            main_frame, textvariable=self.provider_var,
            values=list(PROVIDERS.keys()), state="readonly", width=15)
        self.provider_combo.grid(row=row, column=1, columnspan=2,
                                 sticky='w', padx=5, pady=3)
        # 绑定切换事件：选择 local 时禁用 API Key/Base URL，启用本地模型路径
        self.provider_combo.bind('<<ComboboxSelected>>',
                                 self._on_provider_changed)
        row += 1

        # --- API Key 输入 + 显示/隐藏按钮 ---
        ttk.Label(main_frame, text="API Key:").grid(
            row=row, column=0, sticky='e', padx=5, pady=3)
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(main_frame, textvariable=self.api_key_var,
                                       show='*', width=30)
        self.api_key_entry.grid(row=row, column=1, sticky='we', padx=5, pady=3)
        self._api_key_visible = False
        self.api_key_show_btn = ttk.Button(
            main_frame, text="显示", width=4,
            command=self._toggle_api_key_visibility)
        self.api_key_show_btn.grid(row=row, column=2, sticky='w', padx=2, pady=3)
        row += 1

        # --- Base URL ---
        ttk.Label(main_frame, text="Base URL:").grid(
            row=row, column=0, sticky='e', padx=5, pady=3)
        self.base_url_var = tk.StringVar()
        self.base_url_entry = ttk.Entry(main_frame, textvariable=self.base_url_var,
                                        width=30)
        self.base_url_entry.grid(row=row, column=1, columnspan=2,
                                 sticky='we', padx=5, pady=3)
        row += 1

        # --- 模型名称 ---
        ttk.Label(main_frame, text="模型名称:").grid(
            row=row, column=0, sticky='e', padx=5, pady=3)
        self.model_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.model_var,
                  width=30).grid(row=row, column=1, columnspan=2,
                                 sticky='we', padx=5, pady=3)
        row += 1

        # --- 本地模型路径 + 浏览按钮 ---
        ttk.Label(main_frame, text="本地模型路径:").grid(
            row=row, column=0, sticky='e', padx=5, pady=3)
        self.local_model_path_var = tk.StringVar()
        self.local_model_path_entry = ttk.Entry(
            main_frame, textvariable=self.local_model_path_var, width=30)
        self.local_model_path_entry.grid(row=row, column=1, sticky='we',
                                         padx=5, pady=3)
        self.local_model_browse_btn = ttk.Button(
            main_frame, text="浏览...",
            command=self._browse_local_model_path)
        self.local_model_browse_btn.grid(
            row=row, column=2, sticky='w', padx=2, pady=3)
        row += 1

        # --- 超时 / 最大重试 / 温度 Spinbox ---
        ttk.Label(main_frame, text="超时(秒):").grid(
            row=row, column=0, sticky='e', padx=5, pady=3)
        self.timeout_var = tk.IntVar()
        ttk.Spinbox(main_frame, from_=10, to=300,
                    textvariable=self.timeout_var, width=6).grid(
            row=row, column=1, sticky='w', padx=5, pady=3)
        row += 1

        ttk.Label(main_frame, text="最大重试次数:").grid(
            row=row, column=0, sticky='e', padx=5, pady=3)
        self.max_retries_var = tk.IntVar()
        ttk.Spinbox(main_frame, from_=0, to=10,
                    textvariable=self.max_retries_var, width=6).grid(
            row=row, column=1, sticky='w', padx=5, pady=3)
        row += 1

        ttk.Label(main_frame, text="温度:").grid(
            row=row, column=0, sticky='e', padx=5, pady=3)
        self.temperature_var = tk.DoubleVar()
        ttk.Spinbox(main_frame, from_=0.0, to=2.0, increment=0.1,
                    textvariable=self.temperature_var, width=6,
                    format="%.1f").grid(
            row=row, column=1, sticky='w', padx=5, pady=3)
        row += 1

        # --- 提示词目录 + 浏览按钮 ---
        ttk.Label(main_frame, text="提示词目录:").grid(
            row=row, column=0, sticky='e', padx=5, pady=3)
        self.prompts_dir_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.prompts_dir_var,
                  width=30).grid(row=row, column=1, sticky='we',
                                 padx=5, pady=3)
        ttk.Button(main_frame, text="浏览...",
                   command=self._browse_prompts_dir).grid(
            row=row, column=2, sticky='w', padx=2, pady=3)
        row += 1

        # --- 错误标签（红色，校验失败时显示） ---
        self.error_label = ttk.Label(main_frame, text="", foreground='red',
                                     font=('Microsoft YaHei', 9))
        self.error_label.grid(row=row, column=0, columnspan=3,
                              sticky='w', padx=5, pady=5)
        row += 1

        # --- 测试连接状态标签（Layer 6：连接测试与状态反馈）---
        # 显示 "正在测试..." / "连接成功，模型：xxx" / 具体错误
        self.test_status_label = ttk.Label(
            main_frame, text="", font=('Microsoft YaHei', 9))
        self.test_status_label.grid(row=row, column=0, columnspan=3,
                                    sticky='w', padx=5, pady=2)
        row += 1

        # --- 测试连接 / 确定 / 取消 按钮 ---
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=row, column=0, columnspan=3,
                       sticky='ew', padx=5, pady=10)
        # 测试连接放左侧（辅助参考，不阻塞对话框关闭）
        self.test_btn = ttk.Button(btn_frame, text="测试连接",
                                   command=self._on_test_connection)
        self.test_btn.pack(side='left', padx=3)
        # 确定/取消放右侧
        ttk.Button(btn_frame, text="确定",
                   command=self._on_ok).pack(side='right', padx=3)
        ttk.Button(btn_frame, text="取消",
                   command=self._on_cancel).pack(side='right', padx=3)
        row += 1

        # --- 导入/导出配置（Layer 10：跨机器迁移）---
        import_export_frame = ttk.Frame(main_frame)
        import_export_frame.grid(row=row, column=0, columnspan=3,
                                 sticky='ew', padx=5, pady=(0, 10))
        ttk.Button(import_export_frame, text="导出配置...",
                   command=self._on_export_config).pack(side='left', padx=3)
        ttk.Button(import_export_frame, text="导入配置...",
                   command=self._on_import_config).pack(side='left', padx=3)

    def _load_fields(self):
        """从 AppConfig 与 AIConfig 加载字段到控件"""
        cfg = self._app_config
        self.enabled_var.set(cfg.ai_enabled)
        self.provider_var.set(cfg.ai_provider)
        self.base_url_var.set(cfg.ai_base_url)
        self.model_var.set(cfg.ai_model)
        self.local_model_path_var.set(cfg.ai_local_model_path)
        self.timeout_var.set(cfg.ai_timeout)
        self.max_retries_var.set(cfg.ai_max_retries)
        self.temperature_var.set(cfg.ai_temperature)
        self.prompts_dir_var.set(cfg.ai_prompts_dir)
        # API Key 从 AIConfig 加载（方案 A）
        self.api_key_var.set(self._ai_config.api_key or '')
        # 根据当前 provider 更新输入框启用/禁用状态
        self._update_provider_ui()

    # ==================================================================
    # 按钮回调
    # ==================================================================

    def _on_ok(self):
        """确定按钮：保存并关闭"""
        if self._save():
            self.grab_release()
            self.destroy()

    def _on_cancel(self):
        """取消按钮：不保存，直接关闭"""
        self.result = False
        self.grab_release()
        self.destroy()

    def _toggle_api_key_visibility(self):
        """切换 API Key 显示/隐藏"""
        self._api_key_visible = not self._api_key_visible
        if self._api_key_visible:
            self.api_key_entry.configure(show='')
            self.api_key_show_btn.configure(text="隐藏")
        else:
            self.api_key_entry.configure(show='*')
            self.api_key_show_btn.configure(text="显示")

    def _on_provider_changed(self, event=None):
        """服务商切换回调：根据 provider 启用/禁用对应输入框

        选择 local 时：禁用 API Key、Base URL、显示/隐藏按钮，启用本地模型路径
        选择云端提供商时：启用 API Key、Base URL，禁用本地模型路径
        """
        self._update_provider_ui()

    def _update_provider_ui(self):
        """根据当前 provider 更新输入框启用/禁用状态"""
        provider = self.provider_var.get()
        is_local = (provider == 'local')
        # 云端模式：启用 API Key 相关控件，禁用本地模型路径
        api_state = 'disabled' if is_local else 'normal'
        local_state = 'normal' if is_local else 'disabled'
        self.api_key_entry.configure(state=api_state)
        self.api_key_show_btn.configure(state=api_state)
        self.base_url_entry.configure(state=api_state)
        self.local_model_path_entry.configure(state=local_state)
        self.local_model_browse_btn.configure(state=local_state)

    def _browse_local_model_path(self):
        """浏览本地模型路径"""
        path = filedialog.askdirectory(title="选择本地模型目录",
                                       parent=self)
        if path:
            self.local_model_path_var.set(path)

    def _browse_prompts_dir(self):
        """浏览提示词目录"""
        path = filedialog.askdirectory(title="选择提示词目录",
                                       parent=self)
        if path:
            self.prompts_dir_var.set(path)

    # ==================================================================
    # 连接测试（Layer 6）
    # ==================================================================

    def _on_test_connection(self):
        """测试连接按钮回调：在后台线程发送最小消息验证配置

        流程：
        1. 禁用测试按钮，显示 "正在测试..."
        2. 启动后台线程构造临时 AIConfig + make_client + chat(ping)
        3. 成功 → "连接成功，模型：xxx"
        4. 失败 → 显示具体错误（401 鉴权失败/超时/网络错误等）
        5. 主线程通过 after(0, ...) 更新 UI（线程安全）

        测试结果不阻塞对话框关闭，仅作辅助参考。
        """
        # 构造临时 AIConfig（不保存，仅用于测试）
        provider = self.provider_var.get()
        api_key = self.api_key_var.get().strip() or None
        base_url = (self.base_url_var.get().strip()
                    or PROVIDERS.get(provider, {}).get('base_url', ''))
        model = (self.model_var.get().strip()
                 or PROVIDERS.get(provider, {}).get('model', ''))
        local_model_path = self.local_model_path_var.get().strip() or None

        # 本地模式不需要 api_key，但需要 local_model_path
        if not local_model_path and not api_key:
            self.test_status_label.configure(
                text="⚠ 请先填写 API Key 或本地模型路径", foreground='orange')
            return

        try:
            timeout = int(self.timeout_var.get())
        except (ValueError, TypeError):
            timeout = 60
        # 测试连接最多等 30 秒，避免用户等待过长
        timeout = min(timeout, 30)

        test_cfg = AIConfig(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=1,  # 测试时减少重试，快速失败
            temperature=0.0,
            local_model_path=local_model_path,
        )

        # 禁用按钮，显示进度
        self.test_btn.configure(state='disabled')
        self.test_status_label.configure(
            text="正在测试连接...", foreground='blue')

        # 后台线程执行测试
        import threading
        thread = threading.Thread(
            target=self._test_connection_worker,
            args=(test_cfg, model),
            daemon=True,
            name='ai-test-connection',
        )
        thread.start()

    def _test_connection_worker(self, test_cfg, model_name):
        """连接测试 worker（后台线程）

        在后台线程中调用 make_client + chat，避免阻塞 Tk 主线程。
        结果通过 self.after(0, ...) 回传主线程更新 UI。

        Args:
            test_cfg: 临时 AIConfig（从当前对话框字段构造）
            model_name: 模型名（用于成功提示）
        """
        try:
            from tb_risk.ai.client import make_client
            client = make_client(test_cfg)
            if client is None:
                self._update_test_status(
                    "⚠ 配置无效：未启用本地模式且无 API Key",
                    foreground='orange')
                return
            # 发送最小消息（max_tokens=1 降低成本）
            reply = client.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            # 成功
            self._update_test_status(
                f"✓ 连接成功，模型：{model_name}",
                foreground='green')
        except Exception as e:
            # 失败：显示具体错误类型与信息
            err_type = type(e).__name__
            err_msg = str(e)[:120]
            self._update_test_status(
                f"✗ 连接失败：{err_type}: {err_msg}",
                foreground='red')

    def _update_test_status(self, text, foreground='black'):
        """从后台线程安全更新测试状态标签

        Args:
            text: 状态文本
            foreground: 文本颜色（'green'/'red'/'orange'/'blue'）
        """
        def _do_update():
            try:
                self.test_status_label.configure(text=text, foreground=foreground)
                self.test_btn.configure(state='normal')
            except Exception as e:
                LOGGER.debug("更新连接测试状态标签失败（对话框可能已关闭）: %s", e)
        try:
            self.after(0, _do_update)
        except Exception as e:
            LOGGER.debug("调度 after(0) 更新测试状态失败: %s", e)

    # ==================================================================
    # 导入/导出配置（Layer 10）
    # ==================================================================

    def _on_export_config(self):
        """导出配置到 JSON 文件

        合并 AppConfig 的 ai_* 字段与 AIConfig（含 api_key），
        api_key 默认掩码导出（仅显示前 4 位 + 后 4 位），
        用户可选择是否导出完整 api_key。
        """
        path = filedialog.asksaveasfilename(
            parent=self,
            title="导出 AI 配置",
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
            initialfile="tb_risk_ai_config.json",
        )
        if not path:
            return

        # 是否导出完整 api_key
        api_key = self.api_key_var.get().strip()
        if api_key and len(api_key) > 8:
            from tkinter import messagebox
            export_full = messagebox.askyesno(
                "导出 API Key",
                "是否导出完整的 API Key？\n\n"
                "选择「否」将仅导出掩码版本（前 4 位 + 后 4 位），"
                "导入时需重新填写。",
                parent=self,
            )
            if not export_full:
                api_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]

        # 合并 AppConfig 与 AIConfig 字段
        export_data = {
            'ai_enabled': self.enabled_var.get(),
            'ai_provider': self.provider_var.get(),
            'api_key': api_key or None,
            'base_url': self.base_url_var.get().strip() or None,
            'model': self.model_var.get().strip() or None,
            'local_model_path': self.local_model_path_var.get().strip() or None,
            'timeout': self.timeout_var.get(),
            'max_retries': self.max_retries_var.get(),
            'temperature': self.temperature_var.get(),
            'prompts_dir': self.prompts_dir_var.get().strip() or None,
            '_export_version': 1,
        }

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            self.test_status_label.configure(
                text=f"✓ 配置已导出到 {os.path.basename(path)}",
                foreground='green')
        except OSError as e:
            self.test_status_label.configure(
                text=f"✗ 导出失败: {e}", foreground='red')

    def _on_import_config(self):
        """从 JSON 文件导入配置

        读取导出的 JSON 文件，将 ai_* 字段写入 AppConfig 控件，
        api_key 写入 AIConfig 控件。导入后不自动保存，需用户点确定。
        """
        path = filedialog.askopenfilename(
            parent=self,
            title="导入 AI 配置",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                self.test_status_label.configure(
                    text="✗ 导入失败: 文件格式无效（非 JSON 对象）",
                    foreground='red')
                return
        except (OSError, json.JSONDecodeError) as e:
            self.test_status_label.configure(
                text=f"✗ 导入失败: {e}", foreground='red')
            return

        # 写入控件（不保存，等用户点确定）
        self.enabled_var.set(data.get('ai_enabled', False))
        self.provider_var.set(data.get('ai_provider', 'deepseek'))
        self.base_url_var.set(data.get('base_url', ''))
        self.model_var.set(data.get('model', ''))
        self.local_model_path_var.set(data.get('local_model_path', ''))
        self.timeout_var.set(data.get('timeout', 60))
        self.max_retries_var.set(data.get('max_retries', 3))
        self.temperature_var.set(data.get('temperature', 0.2))
        self.prompts_dir_var.set(data.get('prompts_dir', ''))

        # API Key：仅当非掩码版本时导入，避免导入掩码覆盖真实 key
        imported_key = data.get('api_key', '')
        if imported_key and '*' not in imported_key:
            self.api_key_var.set(imported_key)

        # 更新 provider UI 状态
        self._update_provider_ui()

        self.test_status_label.configure(
            text=f"✓ 配置已从 {os.path.basename(path)} 加载，请确认后点确定保存",
            foreground='green')

    # ==================================================================
    # 保存逻辑
    # ==================================================================

    def _validate(self):
        """校验输入

        Returns:
            (bool, str): (是否通过, 错误信息)，通过时错误信息为空
        """
        # 1. prompts_dir 路径安全校验
        prompts_dir = self.prompts_dir_var.get().strip()
        try:
            _validate_prompts_dir(prompts_dir if prompts_dir else None)
        except ValueError as e:
            return False, str(e)

        # 2. provider 特定校验
        provider = self.provider_var.get()
        if provider == 'local':
            if not self.local_model_path_var.get().strip():
                return False, "本地模式下必须指定本地模型路径"
        else:
            api_key = self.api_key_var.get().strip()
            if not api_key:
                return False, "云端模式下必须填写 API Key"

        return True, ''

    def _save(self):
        """保存配置

        保存流程：
        1. 校验 prompts_dir 路径安全 + provider 特定字段
        2. 校验失败 → 显示错误，返回 False
        3. 校验成功 → 更新 AppConfig 字段并 save()
        4. 若 ai_enabled=True → 构造 AIConfig（含 api_key）并 save_to_file()
        5. 若 ai_enabled=False → 跳过 AIConfig 保存（用户已明确关闭 AI，
           不写入 api_key 到用户目录，但仍保留 AppConfig 中的非敏感字段
           以便下次启用时无需重新配置 provider/model 等）
        6. 调用 on_save_callback()
        7. 设置 self.result = True，返回 True

        Returns:
            bool: True 表示保存成功，False 表示校验失败未保存
        """
        # 1. 校验
        ok, err = self._validate()
        if not ok:
            self.error_label.configure(text=err)
            return False
        self.error_label.configure(text='')

        # 2. 更新 AppConfig 字段（非敏感字段）
        cfg = self._app_config
        cfg.ai_enabled = self.enabled_var.get()
        cfg.ai_provider = self.provider_var.get()
        cfg.ai_base_url = self.base_url_var.get().strip()
        cfg.ai_model = self.model_var.get().strip()
        cfg.ai_local_model_path = self.local_model_path_var.get().strip()
        cfg.ai_timeout = self.timeout_var.get()
        cfg.ai_max_retries = self.max_retries_var.get()
        cfg.ai_temperature = self.temperature_var.get()
        cfg.ai_prompts_dir = self.prompts_dir_var.get().strip()

        # 3. 保存 AppConfig
        cfg.save()

        # 4. 若 ai_enabled=True → 构造并保存 AIConfig（含 api_key 明文写入 ai_config.json）
        #    若 ai_enabled=False → 跳过 AIConfig 保存（用户已明确关闭 AI，不写入 api_key）
        if cfg.ai_enabled:
            ai_cfg = AIConfig(
                provider=cfg.ai_provider,
                api_key=self.api_key_var.get().strip() or None,
                base_url=cfg.ai_base_url or PROVIDERS.get(cfg.ai_provider, {}).get(
                    'base_url', ''),
                model=cfg.ai_model or PROVIDERS.get(cfg.ai_provider, {}).get(
                    'model', ''),
                timeout=cfg.ai_timeout,
                max_retries=cfg.ai_max_retries,
                temperature=cfg.ai_temperature,
                local_model_path=cfg.ai_local_model_path or None,
                prompts_dir=cfg.ai_prompts_dir or None,
            )
            ai_cfg.save_to_file(self._ai_config_file)

        # 5. 调用回调
        if self._on_save_callback:
            self._on_save_callback()

        # 6. 标记成功
        self.result = True
        return True
