#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LIS（检验信息系统）标准数据源适配器

覆盖检验结果（痰涂片/培养/Xpert/ESR/CRP/血糖等）数据，来自 LIS。
可复用底层协议适配器（LISPACSAdapter / FHIRClient / HL7），
或通过注入 fetcher 接入任意厂商 LIS 接口。

统一接口继承自 BaseDataSourceAdapter（见 his.py 说明）。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..base import AdapterResult
from ..audit import AuditLogger
from .base import BaseDataSourceAdapter, DataSourceConfig


class LISAdapter(BaseDataSourceAdapter):
    """LIS 数据源标准适配器

    主要字段域：检验结果、危急值、涂片/分子/血清学/常规检验面板。
    """

    ADAPTER_TYPE: str = "lis"
    SOURCE_TYPE: str = "LIS"

    def __init__(self, source_config: Optional[DataSourceConfig] = None,
                 auditor: Optional[AuditLogger] = None,
                 fetcher: Optional[Callable] = None,
                 lab_source: Any = None):
        """初始化 LIS 适配器。

        参数：
            source_config: 数据源配置
            auditor: 审计日志器
            fetcher: 抓取函数
            lab_source: 可选，底层已有适配器（如 LISPACSAdapter），优先委托
        """
        super().__init__(source_config, auditor, fetcher)
        self._lab_source = lab_source

    def _default_fetch(self, patient_id: str, since=None) -> AdapterResult:
        if self._lab_source is not None:
            if since is None:
                res = self._lab_source.fetch_patient(patient_id)
                return res if isinstance(res, AdapterResult) else self._coerce(res)
            res = self._lab_source.fetch_incremental(since)
            if isinstance(res, list):
                return res[0] if res else AdapterResult(success=False,
                                                        source_system=self.SOURCE_TYPE)
            return res if isinstance(res, AdapterResult) else self._coerce(res)
        return super()._default_fetch(patient_id, since)
