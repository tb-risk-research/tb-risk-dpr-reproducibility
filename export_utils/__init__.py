#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 导出工具模块

提供结构化报告导出（Excel/JSON/HTML/PDF）、模型训练报告生成、
数据库连接池管理和审计追踪功能。

本包由原 ``export_utils.py`` 按导出格式拆分而来，子模块分组：
    - ``_common``：共享的 LOGGER 与 pandas/numpy 条件导入
    - ``excel_export``：结构化 Excel 导出（多工作表）
    - ``json_export``：结构化 JSON 导出
    - ``pdf_export``：PDF 报告导出（纯函数）
    - ``report_export``：模型训练报告（HTML/Markdown）与对比报告（HTML）
    - ``database``：数据库连接池管理（DatabaseManager）与审计追踪（AuditLogger）

所有原本可从 ``export_utils`` 导入的名字仍在此包顶层可用。

用法：
    from tb_risk.export_utils import (
        export_to_excel, export_to_json, export_model_training_report,
        DatabaseManager, AuditLogger
    )
"""

from ._common import LOGGER, NUMPY_AVAILABLE, PANDAS_AVAILABLE, np, pd
from .excel_export import (
    _write_family_sheet,
    _write_ml_performance_sheet,
    _write_patient_sheet,
    _write_sdoh_sheet,
    _write_seir_sheet,
    _write_social_sheet,
    _write_summary_sheet,
    export_to_excel,
)
from .json_export import _serialize_for_json, export_to_json
from .pdf_export import _export_pdf_report
from .report_export import (
    _generate_comparison_html,
    _generate_html_report,
    _generate_markdown_report,
    export_comparison_report,
    export_model_training_report,
    generate_assessment_report,
)
from .database import AuditLogger, BusinessTableExporter, DatabaseManager

__all__ = [
    'LOGGER', 'PANDAS_AVAILABLE', 'NUMPY_AVAILABLE', 'pd', 'np',
    'export_to_excel', 'export_to_json',
    'export_model_training_report', 'export_comparison_report',
    'generate_assessment_report',
    'DatabaseManager', 'AuditLogger', 'BusinessTableExporter',
]
