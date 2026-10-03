#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 结果面板模块

ResultsMixin 组合了 InitUIMixin（界面初始化）、DisplayMixin（结果显示）、
OrchestratorMixin（评估编排）、ExportMixin（结果导出）和 AiTabMixin（AI 助手标签页）。

子模块：
- _shared.py:       共享导入和常量
- init_ui.py:       界面初始化（_init_result_tab / _clear_form）
- display.py:       结果显示（Classic / Wizard）
- orchestrator.py:  评估编排（_run_assessment / _update_gui_after_assessment）
- export.py:        结果导出（CSV / TXT / 图表 / Excel / JSON / ML / PDF）
- ai_tab.py:        AI 助手子标签页（缺陷 #3 — UI 入口控件）
"""

from ._shared import *  # noqa: F401,F403 — 提供所有子模块共享的导入和常量
from .init_ui import InitUIMixin
from .display import DisplayMixin
from .orchestrator import OrchestratorMixin
from .export import ExportMixin
from .ai_tab import AiTabMixin
from .model_evidence import ModelEvidenceMixin
from ..charts.advanced_charts import AdvancedChartMixin
# Section IX: 统一导出管理器（独立类，不依赖 Mixin）
from .export_manager import ExportManager


class ResultsMixin(InitUIMixin, DisplayMixin, OrchestratorMixin, ExportMixin,
                   AiTabMixin, ModelEvidenceMixin, AdvancedChartMixin):
    """GUI 结果面板 — 组合 InitUI / Display / Orchestrator / Export / AiTab / 模型证据 / AdvancedChart Mixin"""