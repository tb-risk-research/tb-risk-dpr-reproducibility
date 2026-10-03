#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""业务数据源标准适配器包（适配器 + 配置 落地形态）

将 health_interop 升级为"适配器 + 配置"：
- 每个数据源（HIS/LIS/PACS/CDC）都是标准适配器，统一接口。
- 字段映射通过配置（FieldMappingEngine）注入，无需改代码。
- 统一同步编排：定时拉取、失败重试、断线重连、每轮同步审计。
- AdapterRegistry 提供配置驱动的工厂与运行时管理。

对外导出：
    DataSourceConfig / SyncSchedule / SyncStatus / SyncRunRecord
    BaseDataSourceAdapter
    HISAdapter / LISAdapter / PACSAdapter / CDCDataSourceAdapter
    AdapterRegistry
"""

from .base import (
    DATA_SOURCE_TYPES,
    SyncStatus,
    SyncSchedule,
    DataSourceConfig,
    SyncRunRecord,
    BaseDataSourceAdapter,
)
from .his import HISAdapter
from .lis import LISAdapter
from .pacs import PACSAdapter
from .cdc import CDCDataSourceAdapter
from .registry import AdapterRegistry

__all__ = [
    'DATA_SOURCE_TYPES',
    'SyncStatus',
    'SyncSchedule',
    'DataSourceConfig',
    'SyncRunRecord',
    'BaseDataSourceAdapter',
    'HISAdapter',
    'LISAdapter',
    'PACSAdapter',
    'CDCDataSourceAdapter',
    'AdapterRegistry',
]
