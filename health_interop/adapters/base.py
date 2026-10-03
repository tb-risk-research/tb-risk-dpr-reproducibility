#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""业务数据源适配器——统一接口与同步编排基类

将 health_interop 从"协议层"升级为"适配器 + 配置"的落地形态：

- 每个业务数据源（HIS/LIS/PACS/CDC）都是标准适配器，实现统一接口。
- 统一接口提供：连接/断开、按需查询、增量拉取、**同步编排**
  （retry + 断线重连 + 每轮同步审计日志）。
- 数据源与协议解耦：协议层（FHIR/HL7/DB/REST）可被任何标准适配器复用。

本模块定义：
- DataSourceConfig：数据源配置（连接 + 同步 + 字段映射 + 调度）
- SyncStatus / SyncRunRecord：同步状态与单轮同步记录
- BaseDataSourceAdapter：所有业务适配器的统一基类（含同步编排）
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base import HealthcareAdapter, AdapterConfig, AdapterResult
from ..audit import AuditLogger, AuditOperationType

LOGGER = logging.getLogger("tb_risk.health_interop.adapters")

# 支持的业务数据源类型
DATA_SOURCE_TYPES = ("HIS", "LIS", "PACS", "CDC")


class SyncStatus(Enum):
    """同步状态"""
    IDLE = "idle"                # 空闲
    RUNNING = "running"          # 同步中
    SUCCESS = "success"          # 成功
    FAILED = "failed"            # 失败（已达最大重试）
    RETRYING = "retrying"        # 重试中
    RECONNECTING = "reconnecting"  # 断线重连中


@dataclass
class SyncSchedule:
    """数据源同步调度配置"""
    enabled: bool = False              # 是否启用定时拉取
    interval_seconds: int = 300        # 拉取间隔（秒）
    initial_delay_seconds: int = 0     # 首次延时（秒）
    retry_max: int = 3                 # 失败重试次数
    retry_backoff: float = 1.5         # 指数退避基数（秒）
    reconnect_attempts: int = 5        # 断线重连尝试次数
    audit: bool = True                 # 每轮同步是否写审计日志


@dataclass
class DataSourceConfig:
    """业务数据源配置（连接 + 同步 + 字段映射）"""
    source_type: str = "HIS"           # HIS / LIS / PACS / CDC
    name: str = ""                     # 数据源名称（显示用）
    protocol: str = "rest"             # rest / db / fhir / hl7 / ws / file
    config: AdapterConfig = field(default_factory=AdapterConfig)
    schedule: SyncSchedule = field(default_factory=SyncSchedule)
    mapping: Any = None                # Optional[FieldMappingEngine]
    enabled: bool = True
    vendor_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可持久化的 dict"""
        return {
            "source_type": self.source_type,
            "name": self.name,
            "protocol": self.protocol,
            "config": {
                "base_url": self.config.base_url,
                "port": self.config.port,
                "username": self.config.username,
                "auth_type": self.config.auth_type,
                "connect_timeout": self.config.connect_timeout,
                "read_timeout": self.config.read_timeout,
                "max_retries": self.config.max_retries,
                "poll_interval": self.config.poll_interval,
                "history_years": self.config.history_years,
                "incremental": self.config.incremental,
                "ssl_verify": self.config.ssl_verify,
                "vendor_params": self.config.vendor_params,
            },
            "schedule": {
                "enabled": self.schedule.enabled,
                "interval_seconds": self.schedule.interval_seconds,
                "initial_delay_seconds": self.schedule.initial_delay_seconds,
                "retry_max": self.schedule.retry_max,
                "retry_backoff": self.schedule.retry_backoff,
                "reconnect_attempts": self.schedule.reconnect_attempts,
                "audit": self.schedule.audit,
            },
            "enabled": self.enabled,
            "vendor_params": self.vendor_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSourceConfig":
        """从 dict 反序列化（密码等敏感字段不持久化，需外部回填）"""
        ad = data.get("config", {})
        cfg = AdapterConfig(
            base_url=ad.get("base_url", ""),
            port=ad.get("port", 0),
            username=ad.get("username", ""),
            auth_type=ad.get("auth_type", "none"),
            connect_timeout=ad.get("connect_timeout", 5),
            read_timeout=ad.get("read_timeout", 30),
            max_retries=ad.get("max_retries", 3),
            poll_interval=ad.get("poll_interval", 300),
            history_years=ad.get("history_years", 5),
            incremental=ad.get("incremental", True),
            ssl_verify=ad.get("ssl_verify", True),
            vendor_params=ad.get("vendor_params", {}),
        )
        sd = data.get("schedule", {})
        sched = SyncSchedule(
            enabled=sd.get("enabled", False),
            interval_seconds=sd.get("interval_seconds", 300),
            initial_delay_seconds=sd.get("initial_delay_seconds", 0),
            retry_max=sd.get("retry_max", 3),
            retry_backoff=sd.get("retry_backoff", 1.5),
            reconnect_attempts=sd.get("reconnect_attempts", 5),
            audit=sd.get("audit", True),
        )
        return cls(
            source_type=data.get("source_type", "HIS"),
            name=data.get("name", ""),
            protocol=data.get("protocol", "rest"),
            config=cfg,
            schedule=sched,
            enabled=data.get("enabled", True),
            vendor_params=data.get("vendor_params", {}),
        )


@dataclass
class SyncRunRecord:
    """单轮同步记录"""
    source_name: str = ""
    source_type: str = ""
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None
    status: str = SyncStatus.IDLE.value      # success / failed / retrying
    records_processed: int = 0
    attempts: int = 0
    reconnects: int = 0
    error: str = ""
    audit_logged: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


class BaseDataSourceAdapter(HealthcareAdapter):
    """业务数据源适配器统一基类

    在 HealthcareAdapter 之上补充：
    1. 标准同步接口 run_sync()：封装增量拉取 + 重试 + 断线重连 + 审计。
    2. 连接生命周期管理（ensure_connected / reconnect）。
    3. 状态跟踪（SyncStatus）与单轮同步记录（SyncRunRecord）。

    子类职责：实现 _fetch_incremental_impl() 与 connect()/fetch_patient()。
    为便于测试与无网络环境，也可注入 fetcher 可调用对象。
    """

    ADAPTER_TYPE: str = "datasource"
    SOURCE_TYPE: str = "HIS"

    def __init__(self, source_config: Optional[DataSourceConfig] = None,
                 auditor: Optional[AuditLogger] = None,
                 fetcher: Optional[Callable] = None):
        # 保持父类约定：self.config 为 AdapterConfig
        sc = source_config or DataSourceConfig(source_type=self.SOURCE_TYPE)
        super().__init__(sc.config)
        self.source_config = sc
        self.name = sc.name or f"{sc.source_type}-{sc.protocol}"
        self.auditor = auditor
        # 可注入的抓取函数：fetcher(source_config, patient_id, since) -> AdapterResult
        self._fetcher = fetcher
        self._connected = False
        self._status = SyncStatus.IDLE
        self._last_run: Optional[SyncRunRecord] = None

    # ------------------------------------------------------------------
    # 连接生命周期
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """建立连接。子类可覆盖；默认置为已连接。"""
        self._connected = True
        self._status = SyncStatus.IDLE
        return True

    def disconnect(self):
        self._connected = False
        self._status = SyncStatus.IDLE

    def is_connected(self) -> bool:
        return self._connected

    def ensure_connected(self) -> bool:
        """确保连接可用；若断开则重连。"""
        if self._connected:
            return True
        self._status = SyncStatus.RECONNECTING
        for attempt in range(1, self.source_config.schedule.reconnect_attempts + 1):
            try:
                if self.connect():
                    LOGGER.info("[%s] 重连成功 (第 %d 次)", self.name, attempt)
                    self._status = SyncStatus.IDLE
                    return True
            except Exception as e:  # noqa: BLE001
                LOGGER.warning("[%s] 重连失败 (第 %d 次): %s", self.name, attempt, e)
            time.sleep(min(2 ** attempt, 10))
        LOGGER.error("[%s] 重连失败（已达最大尝试次数）", self.name)
        self._status = SyncStatus.FAILED
        return False

    def health_check(self) -> Tuple[bool, str]:
        return (self._connected, "connected" if self._connected else "disconnected")

    # ------------------------------------------------------------------
    # 数据抓取（子类实现或注入 fetcher）
    # ------------------------------------------------------------------

    def _default_fetch(self, patient_id: str, since=None) -> AdapterResult:
        """默认抓取实现：优先使用注入的 fetcher，否则交由子类。"""
        if self._fetcher is not None:
            res = self._fetcher(self.source_config, patient_id, since)
            return res if isinstance(res, AdapterResult) else self._coerce(res)
        raise NotImplementedError(
            f"{type(self).__name__} 未实现 _default_fetch，请注入 fetcher 或实现 _fetch_incremental_impl")

    def _coerce(self, data: Any) -> AdapterResult:
        """将 dict 或 None 转为 AdapterResult。"""
        if data is None:
            return AdapterResult(success=False,
                                 source_system=self.source_config.source_type,
                                 errors=["无返回数据"])
        if isinstance(data, AdapterResult):
            return data
        if isinstance(data, dict):
            return AdapterResult(
                success=data.get("success", False),
                patient_info=data.get("patient_info", {}),
                lab_results=data.get("lab_results", []),
                imaging_reports=data.get("imaging_reports", []),
                diagnoses=data.get("diagnoses", []),
                source_system=data.get("source_system", self.source_config.source_type),
                records_processed=data.get("records_processed", 0),
                errors=data.get("errors", []),
                last_sync_time=datetime.datetime.now(),
            )
        return AdapterResult(success=False,
                             source_system=self.source_config.source_type,
                             errors=[f"无法识别的返回类型: {type(data).__name__}"])

    def fetch_patient(self, patient_id: str) -> AdapterResult:
        """按需查询单个患者（统一接口）。"""
        if not self.ensure_connected():
            return AdapterResult(success=False,
                                 source_system=self.source_config.source_type,
                                 patient_id=patient_id,
                                 errors=["数据源连接不可用"])
        try:
            return self._coerce(self._default_fetch(patient_id, None))
        except NotImplementedError:
            raise
        except Exception as e:  # noqa: BLE001
            LOGGER.exception("[%s] 按需查询失败: %s", self.name, e)
            return AdapterResult(success=False,
                                 source_system=self.source_config.source_type,
                                 patient_id=patient_id,
                                 errors=[str(e)])

    def fetch_incremental(self, since: Optional[datetime.datetime] = None
                          ) -> List[AdapterResult]:
        """增量拉取（统一接口）。返回列表。"""
        if not self.ensure_connected():
            return [AdapterResult(success=False,
                                  source_system=self.source_config.source_type,
                                  errors=["数据源连接不可用"])]
        try:
            result = self._coerce(self._default_fetch("__incremental__", since))
            return [result] if not isinstance(result, list) else result
        except NotImplementedError:
            raise
        except Exception as e:  # noqa: BLE001
            LOGGER.exception("[%s] 增量拉取失败: %s", self.name, e)
            return [AdapterResult(success=False,
                                  source_system=self.source_config.source_type,
                                  errors=[str(e)])]

    # ------------------------------------------------------------------
    # 标准同步编排（retry + 断线重连 + 审计）
    # ------------------------------------------------------------------

    def run_sync(self, since: Optional[datetime.datetime] = None,
                 force_audit: Optional[bool] = None) -> SyncRunRecord:
        """执行一轮完整同步。

        流程：
        1. 记录同步开始，标记 RUNNING。
        2. 确保连接（断开则重连）。
        3. 增量拉取，失败按 retry_max 指数退避重试。
        4. 成功后写审计日志（默认开启，可用 force_audit 覆盖）。
        5. 记录同步结束与统计。

        参数：
            since: 增量起始时间；None 则取上次同步时间。
            force_audit: 覆盖 schedule.audit 是否写审计日志。

        返回：
            SyncRunRecord: 本轮同步记录。
        """
        audit_flag = (self.source_config.schedule.audit
                      if force_audit is None else force_audit)
        record = SyncRunRecord(
            source_name=self.name,
            source_type=self.source_config.source_type,
            started_at=datetime.datetime.now(),
            status=SyncStatus.RUNNING.value,
        )
        self._status = SyncStatus.RUNNING

        try:
            if not self.ensure_connected():
                raise ConnectionError("无法建立/重连数据源连接")

            # 执行增量拉取 + 重试
            results, attempts = self._fetch_with_retry(since)
            record.attempts = attempts

            processed = 0
            ok = True
            errors = []
            for res in results:
                processed += int(getattr(res, "records_processed", 0) or 0)
                if not getattr(res, "success", False):
                    ok = False
                    errors.extend(getattr(res, "errors", []) or ["未知错误"])

            record.records_processed = processed
            record.finished_at = datetime.datetime.now()

            if ok:
                record.status = SyncStatus.SUCCESS.value
                self._status = SyncStatus.SUCCESS
                self._last_sync_time = record.finished_at
            else:
                record.status = SyncStatus.FAILED.value
                record.error = "; ".join(errors) if errors else "同步结果含失败项"
                self._status = SyncStatus.FAILED

        except Exception as e:  # noqa: BLE001
            record.finished_at = datetime.datetime.now()
            record.status = SyncStatus.FAILED.value
            record.error = str(e)
            self._status = SyncStatus.FAILED
            LOGGER.error("[%s] 同步失败: %s", self.name, e)

        # 审计日志
        record.audit_logged = self._audit_sync(record) if audit_flag else False

        self._last_run = record
        return record

    def _fetch_with_retry(self, since) -> Tuple[List[AdapterResult], int]:
        """增量拉取 + 指数退避重试。返回 (results, attempts)。"""
        retry_max = self.source_config.schedule.retry_max
        backoff = self.source_config.schedule.retry_backoff
        attempt = 0
        last_error = None
        while True:
            attempt += 1
            try:
                results = self.fetch_incremental(since)
                # 只要没有 success=False 且无异常，即认为成功
                if all(getattr(r, "success", False) for r in results):
                    return results, attempt
                last_error = "部分同步项失败"
                self._status = SyncStatus.RETRYING
            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                self._status = SyncStatus.RETRYING
                # 连接类异常触发重连
                if self._is_connection_error(e):
                    self._connected = False
                    self.ensure_connected()

            if attempt >= retry_max:
                break
            wait = backoff ** attempt
            LOGGER.warning("[%s] 同步第 %d 次失败(%s)，%.1fs 后重试",
                           self.name, attempt, last_error, wait)
            time.sleep(wait)

        # 返回最后一次失败结果或空
        return [AdapterResult(success=False,
                              source_system=self.source_config.source_type,
                              errors=[last_error or "同步失败"])], attempt

    @staticmethod
    def _is_connection_error(e: Exception) -> bool:
        msg = str(e).lower()
        return any(k in msg for k in ("connect", "connection", "timeout",
                                      "refused", "网络", "连接", "超时"))

    # ------------------------------------------------------------------
    # 同步审计
    # ------------------------------------------------------------------

    def _audit_sync(self, record: SyncRunRecord) -> bool:
        """将单轮同步写入审计日志。"""
        if self.auditor is None:
            return False
        details = {
            "source_type": record.source_type,
            "status": record.status,
            "records_processed": record.records_processed,
            "attempts": record.attempts,
            "reconnects": record.reconnects,
            "duration_seconds": record.duration_seconds(),
        }
        if record.error:
            details["error"] = record.error
        try:
            return bool(self.auditor.log_operation(
                operation_type=AuditOperationType.SYNC,
                user_id="system",
                user_name=f"sync:{self.name}",
                record_id=self.name,
                record_type=f"datasource:{self.source_config.source_type}",
                details=details,
                new_value={"status": record.status},
            ))
        except Exception as e:  # noqa: BLE001
            LOGGER.error("[%s] 写审计日志失败: %s", self.name, e)
            return False

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def status(self) -> SyncStatus:
        return self._status

    @property
    def last_run(self) -> Optional[SyncRunRecord]:
        return self._last_run

    def get_sync_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """返回最近同步历史摘要（当前仅本进程内存，可扩展为持久化）。"""
        if self._last_run is None:
            return []
        r = self._last_run
        return [{
            "source": r.source_name,
            "status": r.status,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "records_processed": r.records_processed,
            "attempts": r.attempts,
            "error": r.error,
            "audit_logged": r.audit_logged,
        }]
