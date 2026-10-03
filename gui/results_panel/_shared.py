#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 结果面板 - 共享导入和常量（供 display / export 子模块引用）

标准库、科学计算库及项目内部依赖统一维护在 gui/_imports.py 单一真相源中；
本文件仅保留结果面板特有的导入（当前无特有导入，完全依赖 _imports.py）。
"""

from .._imports import *  # noqa: F401,F403 — 标准库、科学计算库与项目内部依赖共享导入
