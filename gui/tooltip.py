#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolTip — 轻量级 tkinter 鼠标悬停提示组件

本模块是 GUI 包的最底层之一，仅依赖 tkinter（可选），不导入任何其他
tb_risk.gui 子模块，避免循环导入。dialog.py、_gui_common.py、assessment.py
等均可安全地从本模块导入 ToolTip。
"""
import logging

_LOGGER = logging.getLogger("tb_risk.gui.tooltip")

# tkinter 可用性标志统一从 utils.py 导入（单一真相源）
from ..utils import TKINTER_AVAILABLE

# tkinter 模块引用（仅在可用时加载，供 ToolTip 类内部使用）
try:
    import tkinter as tk
except ImportError:
    tk = None

# Tooltip 默认配色（深色医疗风格，与主色协调）
_TIP_BG = '#0f172a'        # 深墨蓝底
_TIP_FG = '#f1f5f9'        # 白字
_TIP_BORDER = '#334155'    # 边框色
_TIP_PAD_X = 10
_TIP_PAD_Y = 6


class ToolTip:
    """为 tkinter 控件添加鼠标悬停提示（GUI 专用组件）。

    仅在 tkinter 可用时工作。本类定义在独立的 gui/tooltip.py 模块中，
    不依赖 gui/_imports.py 或其他重量级 GUI 模块，因此可被 dialog.py
    等底层 GUI 模块安全导入而不会引发循环依赖。

    非 GUI 模块（core、scoring、persistence 等）不应导入本类。
    """

    def __init__(self, widget, text):
        if not TKINTER_AVAILABLE:
            raise RuntimeError(
                "ToolTip 需要 tkinter 支持，但当前环境未安装 tkinter。"
                "在无 GUI 环境（Docker/CI/服务器）中不应创建 ToolTip 实例。"
            )
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind('<Enter>', self.show_tip)
        widget.bind('<Leave>', self.hide_tip)
        widget.bind('<ButtonPress>', self.hide_tip)  # 点击时立即隐藏

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_pointerx() + 14
        y = self.widget.winfo_pointery() + 18
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f'+{x}+{y}')
        # 置于顶层
        try:
            tw.attributes('-topmost', True)
        except tk.TclError:
            pass
        # 带边框的深色提示框（relief='solid' 模拟 1px 边框，圆角在原生 tk 无法实现）
        label = tk.Label(
            tw, text=self.text, justify='left',
            background=_TIP_BG, foreground=_TIP_FG,
            relief='solid', borderwidth=1,
            highlightthickness=0,
            font=('Microsoft YaHei UI', 9, 'normal'),
            padx=_TIP_PAD_X, pady=_TIP_PAD_Y,
        )
        label.pack()

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None