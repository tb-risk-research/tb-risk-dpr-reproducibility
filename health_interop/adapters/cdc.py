#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CDC（疾控中心）标准数据源适配器

覆盖疾控回传数据（传染病报告卡、治疗管理、密切接触者、流调/区域流行病数据）。
可复用底层协议适配器（CDC 回传接收 / 区域平台 WebService / FHIR），
或通过注入 fetcher 接入任意区域平台接口。

统一接口继承自 BaseDataSourceAdapter（见 his.py 说明）。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..base import AdapterResult
from ..audit import AuditLogger
from .base import BaseDataSourceAdapter, DataSourceConfig


class CDCDataSourceAdapter(BaseDataSourceAdapter):
    """CDC 数据源标准适配器

    主要字段域：报告卡回传、密切接触者、治疗管理、流行病学/流调数据。
    """

    ADAPTER_TYPE: str = "cdc"
    SOURCE_TYPE: str = "CDC"

    def __init__(self, source_config: Optional[DataSourceConfig] = None,
                 auditor: Optional[AuditLogger] = None,
                 fetcher: Optional[Callable] = None,
                 cdc_source: Any = None):
        """初始化 CDC 适配器。

        参数：
            source_config: 数据源配置
            auditor: 审计日志器
            fetcher: 抓取函数
            cdc_source: 可选，底层已有适配器/接收器，优先委托
        """
        super().__init__(source_config, auditor, fetcher)
        self._cdc_source = cdc_source

    def _default_fetch(self, patient_id: str, since=None) -> AdapterResult:
        if self._cdc_source is not None:
            if since is None:
                res = self._cdc_source.fetch_patient(patient_id)
                return res if isinstance(res, AdapterResult) else self._coerce(res)
            res = self._cdc_source.fetch_incremental(since)
            if isinstance(res, list):
                return res[0] if res else AdapterResult(success=False,
                                                        source_system=self.SOURCE_TYPE)
            return res if isinstance(res, AdapterResult) else self._coerce(res)
        return super()._default_fetch(patient_id, since)
