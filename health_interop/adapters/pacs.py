#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PACS（影像归档与通信系统）标准数据源适配器

覆盖影像检查与报告（胸片/CT、疑似结核、空洞等征象），来自 PACS。
可复用底层协议适配器（LISPACSAdapter / DICOM / RadiologyNLP），
或通过注入 fetcher 接入任意厂商 PACS 接口。

统一接口继承自 BaseDataSourceAdapter（见 his.py 说明）。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..base import AdapterResult
from ..audit import AuditLogger
from .base import BaseDataSourceAdapter, DataSourceConfig


class PACSAdapter(BaseDataSourceAdapter):
    """PACS 数据源标准适配器

    主要字段域：影像报告、影像所见（NLP 提取疑似结核/空洞/进展）。
    """

    ADAPTER_TYPE: str = "pacs"
    SOURCE_TYPE: str = "PACS"

    def __init__(self, source_config: Optional[DataSourceConfig] = None,
                 auditor: Optional[AuditLogger] = None,
                 fetcher: Optional[Callable] = None,
                 imaging_source: Any = None):
        """初始化 PACS 适配器。

        参数：
            source_config: 数据源配置
            auditor: 审计日志器
            fetcher: 抓取函数
            imaging_source: 可选，底层已有适配器（如 LISPACSAdapter），优先委托
        """
        super().__init__(source_config, auditor, fetcher)
        self._imaging_source = imaging_source

    def _default_fetch(self, patient_id: str, since=None) -> AdapterResult:
        if self._imaging_source is not None:
            if since is None:
                res = self._imaging_source.fetch_patient(patient_id)
                return res if isinstance(res, AdapterResult) else self._coerce(res)
            res = self._imaging_source.fetch_incremental(since)
            if isinstance(res, list):
                return res[0] if res else AdapterResult(success=False,
                                                        source_system=self.SOURCE_TYPE)
            return res if isinstance(res, AdapterResult) else self._coerce(res)
        return super()._default_fetch(patient_id, since)
