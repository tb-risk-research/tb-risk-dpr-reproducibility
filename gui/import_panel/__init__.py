#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 导入面板模块

DataImportMixin 组合了 CSVImportMixin（CSV 导入）、ExcelImportMixin（Excel 导入）、
PreviewMixin（预览与进度）、DatabaseMixin（数据库导入/导出）、ApiImportMixin（API 导入）、
JsonImportMixin（JSON 导入）、DataHelpersMixin（数据助手）和 HealthInteropMixin（医疗标准接口）。

子模块：
- _shared.py:       共享导入和常量
- csv_import.py:    CSV 导入（CSVImportMixin）
- excel_import.py:  Excel 导入（ExcelImportMixin）
- preview_panel.py: 预览与进度显示（PreviewMixin）
- database.py:      数据库导入/导出（DatabaseMixin）
- api_import.py:    API 导入（ApiImportMixin）
- json_import.py:   JSON 导入（JsonImportMixin）
- data_helpers.py:  数据助手（DataHelpersMixin）
- health_interop_import.py: 医疗标准接口接入（HL7 FHIR/HL7 v2/DB/LIS-PACS/CDC/REST）
- pipeline.py:      Section VIII 统一导入管线（模板方法模式）
"""

from ._shared import *  # noqa: F401,F403 — 提供所有子模块共享的导入和常量
from .csv_import import CSVImportMixin
from .excel_import import ExcelImportMixin
from .preview_panel import PreviewMixin
from .database import DatabaseMixin
from .api_import import ApiImportMixin
from .json_import import JsonImportMixin
from .data_helpers import DataHelpersMixin
from .pipeline import (ImportPipeline, CSVImportPipeline, JSONImportPipeline,
                        ExcelImportPipeline, DatabaseImportPipeline, APIImportPipeline)
from .health_interop_import import HealthInteropMixin, HealthInteropImportPipeline


class DataImportMixin(CSVImportMixin, ExcelImportMixin, PreviewMixin,
                      DatabaseMixin, ApiImportMixin, JsonImportMixin,
                      DataHelpersMixin, HealthInteropMixin):
    """GUI 导入面板 — 组合 CSV / Excel / Preview / Database / API / JSON / Helpers / HealthInterop Mixin"""