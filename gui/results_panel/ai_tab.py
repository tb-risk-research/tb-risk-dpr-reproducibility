#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 结果面板 - AI 助手子标签页 Mixin

缺陷 #3 修复：在结果面板新增 "AI 助手" 子标签页，提供 UI 入口控件：
- 问题输入 Text + "提问" 按钮 → 调用 AiAssistantMixin.ask_ai()
- "生成 AI 报告" 按钮 → 调用 AiAssistantMixin.generate_ai_report()
- "清空输出" 按钮 → 清空 AI 输出区
- AI 输出 Text 控件 → 显示问答/报告/错误消息
- AI 状态标签 → 实时显示 "就绪/正在思考.../正在生成报告.../错误"

回调注册：_init_ai_tab 末尾覆盖 _init_ai_assistant 中设置的默认 lambda 回调，
将 answer/report/error 消息写入 AI 输出 Text 控件。

依赖：
- AiAssistantMixin（提供 ask_ai / generate_ai_report / is_ai_available）
- _init_ai_assistant 已初始化 _callbacks / last_assessment_result
- ttk.Notebook（由 _init_result_tab 创建并传入）
"""

# 直接导入 tkinter（不通过 _shared.py），使模块可独立加载（便于测试）
import tkinter as tk
from tkinter import ttk


class AiTabMixin:
    """AI 助手子标签页 Mixin — 提供 UI 入口控件与回调绑定

    在结果面板的 result_notebook 中新增 "AI 助手" 子标签页，
    绑定 ask_ai / generate_ai_report 到按钮，并把消息回调写入 Text 控件。
    """

    def _init_ai_tab(self, result_notebook):
        """在结果面板的 result_notebook 中新增 "AI 助手" 子标签页

        Args:
            result_notebook: ttk.Notebook 结果面板的子标签页容器
        """
        ai_tab = ttk.Frame(result_notebook)
        result_notebook.add(ai_tab, text='AI 助手')

        # --- 0. AI 设置入口（Layer 5）---
        # 用户看到"未启用"提示时能一键跳转到设置对话框，形成闭环
        settings_frame = ttk.Frame(ai_tab)
        settings_frame.pack(fill='x', padx=10, pady=(5, 0))
        ttk.Button(settings_frame, text="⚙ AI 设置",
                   command=self._on_ai_settings_clicked).pack(side='right')

        # --- 1. AI 问答输入区 ---
        input_frame = ttk.LabelFrame(ai_tab, text="AI 问答", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(input_frame, text="请输入问题：").pack(anchor='w', padx=5, pady=(0, 3))
        self.ai_question_text = tk.Text(input_frame, wrap='word', height=3,
                                        font=('Microsoft YaHei', 10))
        self.ai_question_text.pack(fill='x', padx=5, pady=2)

        # 按钮栏
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill='x', padx=5, pady=5)
        ttk.Button(btn_frame, text="提问",
                   command=self._on_ask_ai_clicked).pack(side='left', padx=3)
        ttk.Button(btn_frame, text="生成 AI 报告",
                   command=self._on_generate_ai_report_clicked).pack(side='left', padx=3)
        ttk.Button(btn_frame, text="清空输出",
                   command=self._clear_ai_output).pack(side='left', padx=3)
        # 缺陷 #7：取消按钮 — 让用户能中止长时间运行的 AI 任务
        # 通过 AiAssistantMixin.cancel_ai_tasks() 设置 threading.Event，
        # 后台 worker 在调用 AI 前检查并主动退出
        ttk.Button(btn_frame, text="取消 AI 任务",
                   command=self._on_cancel_ai_clicked).pack(side='left', padx=3)

        # AI 状态标签
        self.ai_status_var = tk.StringVar(value="AI 状态：就绪")
        ttk.Label(input_frame, textvariable=self.ai_status_var,
                  font=('Microsoft YaHei', 9)).pack(anchor='w', padx=5, pady=(3, 0))

        # --- 2. AI 输出区 ---
        output_frame = ttk.LabelFrame(ai_tab, text="AI 输出", padding=10)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.ai_output_text = tk.Text(output_frame, wrap='word',
                                      font=('Microsoft YaHei', 10))
        ai_scrollbar = ttk.Scrollbar(output_frame, orient='vertical',
                                     command=self.ai_output_text.yview)
        self.ai_output_text.configure(yscrollcommand=ai_scrollbar.set)
        self.ai_output_text.pack(side='left', fill='both', expand=True)
        ai_scrollbar.pack(side='right', fill='y')

        # --- 3. 覆盖默认回调（_init_ai_assistant 中设置的 lambda） ---
        # 把 answer/report/error/cancelled 消息写入 AI 输出 Text 控件
        # 缺陷 #7：新增 'cancelled' 回调，处理被用户取消的任务
        if hasattr(self, '_callbacks') and isinstance(self._callbacks, dict):
            self._callbacks['answer'] = self._on_ai_answer
            self._callbacks['report'] = self._on_ai_report
            self._callbacks['error'] = self._on_ai_error
            self._callbacks['cancelled'] = self._on_ai_cancelled

    # ======================================================================
    # 按钮回调
    # ======================================================================

    def _on_ask_ai_clicked(self):
        """提问按钮回调：从 Text 控件读取问题，调用 ask_ai()"""
        question = self.ai_question_text.get('1.0', tk.END).strip()
        if not question:
            self._append_ai_output("⚠ 请输入问题后再点击提问。\n")
            return
        # 检查 AI 是否可用
        if not self.is_ai_available():
            self._append_ai_output(
                "⚠ AI 助手未启用。请在配置中开启 AI 功能并设置 API Key。\n")
            return
        # 启动后台问答
        started = self.ask_ai(question)
        if started:
            self.ai_status_var.set("AI 状态：正在思考...")
            self._append_ai_output(f"🙋 问题：{question}\n")
        else:
            self._append_ai_output("⚠ 无法启动 AI 问答（请检查配置）。\n")

    def _on_generate_ai_report_clicked(self):
        """生成 AI 报告按钮回调：调用 generate_ai_report()"""
        if not self.is_ai_available():
            self._append_ai_output(
                "⚠ AI 助手未启用。请在配置中开启 AI 功能并设置 API Key。\n")
            return
        # 检查是否有评估结果
        result = getattr(self, 'last_assessment_result', None) or {}
        if not result:
            self._append_ai_output("⚠ 请先执行风险评估，再生成 AI 报告。\n")
            return
        # 启动后台报告生成
        started = self.generate_ai_report()
        if started:
            self.ai_status_var.set("AI 状态：正在生成报告...")
            self._append_ai_output("📋 正在生成 AI 风险评估报告...\n")
        else:
            self._append_ai_output("⚠ 无法启动 AI 报告生成（请检查配置）。\n")

    def _clear_ai_output(self):
        """清空 AI 输出区"""
        if hasattr(self, 'ai_output_text'):
            self.ai_output_text.delete('1.0', tk.END)
        if hasattr(self, 'ai_status_var'):
            self.ai_status_var.set("AI 状态：就绪")

    def _on_cancel_ai_clicked(self):
        """取消按钮回调：取消所有活动 AI 任务（缺陷 #7）

        通过 AiAssistantMixin.cancel_ai_tasks() 设置所有活动 job 的
        threading.Event，后台 worker 在调用 AI 前检查并主动退出。
        注意：取消仅在 worker 进入 AI 调用前的检查点生效，已在进行中的
        HTTP 请求需等待完成（httpx 不支持中断）。
        """
        if not hasattr(self, 'cancel_ai_tasks'):
            self._append_ai_output("⚠ 取消功能未启用。\n")
            return
        cancelled = self.cancel_ai_tasks()
        if cancelled > 0:
            self.ai_status_var.set(f"AI 状态：正在取消 {cancelled} 个任务...")
            self._append_ai_output(f"🚫 已请求取消 {cancelled} 个 AI 任务。\n")
        else:
            self._append_ai_output("ℹ️ 当前没有正在运行的 AI 任务。\n")

    def _on_ai_settings_clicked(self):
        """AI 设置按钮回调（Layer 5）：跳转到 AI 设置对话框

        让用户在看到"未启用"提示时能一键打开设置对话框，完成 API Key
        等配置闭环。委托给 BaseMixin._open_ai_settings（若可用）。
        """
        opener = getattr(self, '_open_ai_settings', None)
        if opener is None:
            self._append_ai_output("⚠ AI 设置入口未启用。\n")
            return
        opener()

    # ======================================================================
    # AI 消息回调（由 _poll_ai_queue 分发）
    # ======================================================================

    def _on_ai_answer(self, msg):
        """AI 问答结果回调：把回答写入输出区"""
        content = msg.get('content', '')
        self._append_ai_output(f"🤖 回答：\n{content}\n\n")
        if hasattr(self, 'ai_status_var'):
            self.ai_status_var.set("AI 状态：就绪")

    def _on_ai_report(self, msg):
        """AI 报告结果回调：把报告写入输出区"""
        content = msg.get('content', '')
        self._append_ai_output(f"📋 AI 报告：\n{content}\n\n")
        if hasattr(self, 'ai_status_var'):
            self.ai_status_var.set("AI 状态：就绪")

    def _on_ai_error(self, msg):
        """AI 错误回调：把错误信息写入输出区"""
        content = msg.get('content', '')
        self._append_ai_output(f"❌ 错误：{content}\n\n")
        if hasattr(self, 'ai_status_var'):
            self.ai_status_var.set("AI 状态：错误")

    def _on_ai_cancelled(self, msg):
        """AI 任务取消回调：把取消信息写入输出区（缺陷 #7）"""
        content = msg.get('content', 'AI 任务已取消')
        self._append_ai_output(f"🚫 已取消：{content}\n\n")
        if hasattr(self, 'ai_status_var'):
            self.ai_status_var.set("AI 状态：已取消")

    # ======================================================================
    # 辅助方法
    # ======================================================================

    def _append_ai_output(self, text):
        """向 AI 输出区追加文本"""
        if not hasattr(self, 'ai_output_text'):
            return
        self.ai_output_text.insert(tk.END, text)
        self.ai_output_text.see(tk.END)  # 滚动到最新内容
