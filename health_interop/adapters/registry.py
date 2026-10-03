#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""适配器注册表——配置驱动的数据源工厂与管理器

把 HIS/LIS/PACS/CDC 标准适配器按"配置"统一装配，实现：
- register：注册某 source_type 对应的适配器类（可扩展第三方适配器）。
- create / create_from_dict：按 DataSourceConfig（或 dict）创建适配器实例。
- create_all：批量按配置列表创建。
- get / list / remove：运行时管理已创建的数据源。
- apply_mapping：将数据源配置中的字段映射引擎应用到原始记录。

设计目标：实施人员只需提供配置（含字段映射），无需修改代码即可接入新数据源。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Type

from ..base import AdapterResult
from ..audit import AuditLogger
from .base import BaseDataSourceAdapter, DataSourceConfig
from .his import HISAdapter
from .lis import LISAdapter
from .pacs import PACSAdapter
from .cdc import CDCDataSourceAdapter

LOGGER = logging.getLogger("tb_risk.health_interop.adapters.registry")


class AdapterRegistry:
    """标准数据源适配器注册表（工厂 + 运行时管理）"""

    def __init__(self, auditor: Optional[AuditLogger] = None):
        self.auditor = auditor
        # 默认注册四大标准适配器
        self._registry: Dict[str, Type[BaseDataSourceAdapter]] = {
            "HIS": HISAdapter,
            "LIS": LISAdapter,
            "PACS": PACSAdapter,
            "CDC": CDCDataSourceAdapter,
        }
        # 已创建的实例：name -> adapter
        self._adapters: Dict[str, BaseDataSourceAdapter] = {}
        # 已创建的配置：name -> DataSourceConfig
        self._configs: Dict[str, DataSourceConfig] = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(self, source_type: str, adapter_cls: Type[BaseDataSourceAdapter]):
        """注册（或覆盖）某 source_type 的适配器类。"""
        self._registry[source_type.upper()] = adapter_cls

    def unregister(self, source_type: str):
        self._registry.pop(source_type.upper(), None)

    def get_adapter_class(self, source_type: str) -> Optional[Type[BaseDataSourceAdapter]]:
        return self._registry.get(source_type.upper())

    def supported_types(self) -> List[str]:
        return list(self._registry.keys())

    # ------------------------------------------------------------------
    # 创建
    # ------------------------------------------------------------------

    def create(self, source_config: DataSourceConfig,
               fetcher: Optional[Callable] = None,
               underlying: Any = None) -> BaseDataSourceAdapter:
        """按 DataSourceConfig 创建适配器实例。

        参数：
            source_config: 数据源配置（含 source_type/name/schedule/mapping）
            fetcher: 可选注入的抓取函数（便于测试/无网络）
            underlying: 可选底层已有适配器，委托其抓取

        返回：
            BaseDataSourceAdapter 实例

        抛出：
            ValueError: 未注册的 source_type
        """
        st = source_config.source_type.upper()
        cls = self._registry.get(st)
        if cls is None:
            raise ValueError(f"未注册的数据源类型: {source_config.source_type}，"
                             f"可用: {self.supported_types()}")
        adapter = cls(source_config=source_config,
                      auditor=self.auditor,
                      fetcher=fetcher,
                      **self._underlying_kwargs(cls, underlying))
        self._adapters[source_config.name] = adapter
        self._configs[source_config.name] = source_config
        return adapter

    @staticmethod
    def _underlying_kwargs(cls, underlying: Any) -> Dict[str, Any]:
        """将底层适配器按具体子类的参数名注入。"""
        if underlying is None:
            return {}
        param_names = {
            "HISAdapter": "demographic_source",
            "LISAdapter": "lab_source",
            "PACSAdapter": "imaging_source",
            "CDCDataSourceAdapter": "cdc_source",
        }
        key = cls.__name__
        if key in param_names:
            return {param_names[key]: underlying}
        LOGGER.debug("适配器 %s 不接收底层委托，忽略 underlying", key)
        return {}

    def create_from_dict(self, data: Dict[str, Any],
                         fetcher: Optional[Callable] = None,
                         underlying: Any = None) -> BaseDataSourceAdapter:
        """从 dict 配置创建适配器（便于从 JSON/配置文件加载）。"""
        sc = DataSourceConfig.from_dict(data)
        return self.create(sc, fetcher=fetcher, underlying=underlying)

    def create_all(self, configs: List[DataSourceConfig],
                   fetcher: Optional[Callable] = None) -> Dict[str, BaseDataSourceAdapter]:
        """批量创建。返回 name -> adapter。"""
        out = {}
        for sc in configs:
            out[sc.name] = self.create(sc, fetcher=fetcher)
        return out

    # ------------------------------------------------------------------
    # 运行时管理
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[BaseDataSourceAdapter]:
        return self._adapters.get(name)

    def get_config(self, name: str) -> Optional[DataSourceConfig]:
        return self._configs.get(name)

    def list(self) -> List[BaseDataSourceAdapter]:
        return list(self._adapters.values())

    def remove(self, name: str) -> bool:
        adapter = self._adapters.pop(name, None)
        if adapter is not None:
            try:
                adapter.disconnect()
            except Exception as e:  # noqa: BLE001
                LOGGER.debug("断开 %s 失败: %s", name, e)
            self._configs.pop(name, None)
            return True
        return False

    def clear(self):
        for name in list(self._adapters.keys()):
            self.remove(name)

    def snapshot(self) -> List[Dict[str, Any]]:
        """返回所有数据源的状态快照（用于界面展示/监控）。"""
        out = []
        for name, adapter in self._adapters.items():
            out.append({
                "name": name,
                "source_type": adapter.source_config.source_type,
                "protocol": adapter.source_config.protocol,
                "enabled": adapter.source_config.enabled,
                "schedule_enabled": adapter.source_config.schedule.enabled,
                "interval_seconds": adapter.source_config.schedule.interval_seconds,
                "connected": adapter.is_connected(),
                "status": adapter.status.value,
                "last_sync": adapter.last_run.to_dict()
                if adapter.last_run is not None else None,
            })
        return out

    # ------------------------------------------------------------------
    # 字段映射应用
    # ------------------------------------------------------------------

    def apply_mapping(self, source_name: str,
                      records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将某数据源配置中的字段映射引擎应用到原始记录。

        参数：
            source_name: 数据源名称
            records: 医院系统原始字段记录列表

        返回：
            list[dict]: 映射为 tb_risk 标准字段后的记录列表
        """
        sc = self._configs.get(source_name)
        if sc is None or sc.mapping is None:
            return list(records)
        try:
            return sc.mapping.apply_batch(records)
        except Exception as e:  # noqa: BLE001
            LOGGER.error("[%s] 应用字段映射失败: %s", source_name, e)
            return list(records)

    def map_adapter_result(self, source_name: str,
                           result: AdapterResult) -> AdapterResult:
        """对 AdapterResult 中的各记录应用字段映射，并回填到 result。"""
        sc = self._configs.get(source_name)
        if sc is None or sc.mapping is None:
            return result
        # patient_info 为字典（单条），其余为列表（多条）
        if isinstance(result.patient_info, dict) and result.patient_info:
            try:
                result.patient_info = sc.mapping.apply(result.patient_info)
            except Exception as e:  # noqa: BLE001
                LOGGER.error("[%s] 映射 patient_info 失败: %s", source_name, e)
        sections = ["family_contacts", "social_contacts",
                    "lab_results", "imaging_reports", "diagnoses",
                    "medications", "clinical_notes"]
        for section in sections:
            items = getattr(result, section, None)
            if isinstance(items, list) and items:
                try:
                    setattr(result, section, sc.mapping.apply_batch(items))
                except Exception as e:  # noqa: BLE001
                    LOGGER.error("[%s] 映射 %s 失败: %s", source_name, section, e)
        return result
