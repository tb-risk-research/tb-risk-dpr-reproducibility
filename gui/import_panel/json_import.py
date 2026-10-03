#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 导入面板 - JSON 导入 Mixin（_load_from_json）"""

from ._shared import *


class JsonImportMixin:
    """JSON 导入方法"""

    def _load_from_json(self):
        """从 JSON 文件加载数据（Section VIII: 走统一导入管线）"""
        from .pipeline import JSONImportPipeline
        pipeline = JSONImportPipeline(self)
        pipeline.run()