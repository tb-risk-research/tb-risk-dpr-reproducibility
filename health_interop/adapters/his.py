#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HIS（医院信息系统）标准数据源适配器

覆盖患者人口学信息、就诊/入院（ADT）、诊断等来自 HIS 的数据。
可复用底层协议适配器（如 DatabaseDirectAdapter / FHIRClient），
或通过注入 fetcher 接入任意厂商 HIS 接口。

统一接口继承自 BaseDataSourceAdapter：
    connect / disconnect / ensure_connected
    fetch_patient(patient_id) / fetch_incremental(since)
    run_sync()（含重试 + 断线重连 + 审计）
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..base import AdapterResult
from ..audit import AuditLogger
from .base import BaseDataSourceAdapter, DataSourceConfig


class HISAdapter(BaseDataSourceAdapter):
    """HIS 数据源标准适配器

    主要字段域：患者基本信息、就诊记录、诊断、用药。

    用法（无需改代码，配置驱动）：
        cfg = DataSourceConfig(source_type="HIS", name="xx医院HIS",
                               protocol="db")
        his = HISAdapter(cfg, fetcher=my_fetch_fn)
        record = his.run_sync()
    """

    ADAPTER_TYPE: str = "his"
    SOURCE_TYPE: str = "HIS"

    def __init__(self, source_config: Optional[DataSourceConfig] = None,
                 auditor: Optional[AuditLogger] = None,
                 fetcher: Optional[Callable] = None,
                 demographic_source: Any = None):
        """初始化 HIS 适配器。

        参数：
            source_config: 数据源配置（缺省按 SOURCE_TYPE 生成）
            auditor: 审计日志器（同步审计）
            fetcher: 抓取函数 fetcher(source_config, patient_id, since) -> AdapterResult
            demographic_source: 可选，底层已有适配器（如 DatabaseDirectAdapter），
                优先委托其 fetch_patient/fetch_incremental
        """
        super().__init__(source_config, auditor, fetcher)
        self._demographic_source = demographic_source

    # -- 优先委托底层适配器 --
    def _default_fetch(self, patient_id: str, since=None) -> AdapterResult:
        if self._demographic_source is not None:
            if since is None:
                res = self._demographic_source.fetch_patient(patient_id)
                return res if isinstance(res, AdapterResult) else self._coerce(res)
            res = self._demographic_source.fetch_incremental(since)
            if isinstance(res, list):
                return res[0] if res else AdapterResult(success=False,
                                                        source_system=self.SOURCE_TYPE)
            return res if isinstance(res, AdapterResult) else self._coerce(res)
        return super()._default_fetch(patient_id, since)
