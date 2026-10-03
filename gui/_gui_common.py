#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 共享导入与常量定义 — 薄包装层

从 gui/_imports.py 统一导入标准库、科学计算库及可用性标志，
并追加 GUI 子模块所需的项目内部依赖（ToolTip 组件）。

子模块（tabs/*、wizard/*）通过 ``from .._gui_common import tk, ttk, ...``
引用，避免各自重复 200 行导入样板。标准库与科学计算库的导入维护在
gui/_imports.py 单一真相源中。
"""

# ==================== 标准库 + 科学计算库（单一真相源：_imports.py） ====================
from ._imports import *  # noqa: F401,F403 — 受控导出，见 _imports.__all__

# ==================== GUI 专用组件（ToolTip 独立模块避免循环依赖） ====================
# ToolTip 定义在独立的 gui/tooltip.py 模块中（避免循环依赖，因为 dialog.py
# 也需要导入 ToolTip，而 dialog 是 _imports 的上游依赖）。
from .tooltip import ToolTip  # noqa: F401 — 重导出给子模块使用
