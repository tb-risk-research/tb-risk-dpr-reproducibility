#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 导入面板 - 共享导入和常量（供 csv_import / excel_import / preview_panel 子模块引用）

标准库、科学计算库及项目内部依赖统一维护在 gui/_imports.py 单一真相源中；
本文件仅保留导入面板特有的导入。
"""

from .._imports import *  # noqa: F401,F403 — 标准库、科学计算库与项目内部依赖共享导入

# 导入面板特有：pandas 用于 DataFrame 预览与列类型推断
# （pandas 已在 _imports.py 中作为可选依赖导入，此处显式引用以确保面板内可用）
try:
    import pandas as pd  # noqa: F811 — 显式重引用，面板内频繁使用
except ImportError:
    pd = None  # type: ignore
