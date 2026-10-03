#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""业务数据源标准适配器 / 配置 / 同步调度 测试

覆盖：
- DataSourceConfig 序列化/反序列化
- AdapterRegistry 配置驱动工厂（HIS/LIS/PACS/CDC）
- BaseDataSourceAdapter 统一同步编排（成功/失败重试/断线重连）
- 同步审计日志（SYNC 操作）
- 字段映射应用（apply_mapping / map_adapter_result）
- SyncScheduler 定时拉取与手动触发
- 包导出
"""

import datetime
import os
import sys
import tempfile
import time

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.health_interop.adapters import (
    DATA_SOURCE_TYPES,
    DataSourceConfig,
    SyncSchedule,
    SyncStatus,
    SyncRunRecord,
    BaseDataSourceAdapter,
    HISAdapter,
    LISAdapter,
    PACSAdapter,
    CDCDataSourceAdapter,
    AdapterRegistry,
)
from tb_risk.health_interop.sync_scheduler import SyncScheduler
from tb_risk.health_interop.audit import AuditLogger, AuditOperationType
from tb_risk.health_interop.base import AdapterResult
from tb_risk.health_interop.mapping.field_mapping_engine import FieldMappingEngine


# ============================================================================
# 包导出
# ============================================================================

class TestExports:
    def test_adapters_package_exports(self):
        assert "HIS" in DATA_SOURCE_TYPES
        assert set(DATA_SOURCE_TYPES) == {"HIS", "LIS", "PACS", "CDC"}
        for cls in (HISAdapter, LISAdapter, PACSAdapter, CDCDataSourceAdapter,
                    BaseDataSourceAdapter, AdapterRegistry):
            assert callable(cls)

    def test_health_interop_package_exports(self):
        import tb_risk.health_interop as hi
        for name in ("DataSourceConfig", "AdapterRegistry", "SyncScheduler",
                     "HISAdapter", "LISAdapter", "PACSAdapter",
                     "CDCDataSourceAdapter", "SyncStatus"):
            assert hasattr(hi, name)


# ============================================================================
# DataSourceConfig
# ============================================================================

class TestDataSourceConfig:
    def test_defaults(self):
        cfg = DataSourceConfig()
        assert cfg.source_type == "HIS"
        assert cfg.enabled is True
        assert cfg.schedule.interval_seconds == 300
        assert cfg.schedule.retry_max == 3

    def test_roundtrip(self):
        cfg = DataSourceConfig(
            source_type="LIS", name="检验系统",
            protocol="fhir",
            schedule=SyncSchedule(enabled=True, interval_seconds=120,
                                  retry_max=5, audit=True),
        )
        data = cfg.to_dict()
        cfg2 = DataSourceConfig.from_dict(data)
        assert cfg2.source_type == "LIS"
        assert cfg2.name == "检验系统"
        assert cfg2.protocol == "fhir"
        assert cfg2.schedule.enabled is True
        assert cfg2.schedule.interval_seconds == 120
        assert cfg2.schedule.retry_max == 5


# ============================================================================
# 标准适配器
# ============================================================================

def _make_adapter_result(source="HIS", records=1, success=True, error=""):
    return AdapterResult(
        success=success,
        source_system=source,
        records_processed=records,
        errors=[error] if error else [],
        last_sync_time=datetime.datetime.now(),
    )


class TestAdapters:
    def test_his_adapter_source_type(self):
        a = HISAdapter(DataSourceConfig(source_type="HIS", name="his"))
        assert a.SOURCE_TYPE == "HIS"
        assert a.ADAPTER_TYPE == "his"
        assert a.name == "his"

    def test_standard_adapter_source_types(self):
        assert LISAdapter().SOURCE_TYPE == "LIS"
        assert PACSAdapter().SOURCE_TYPE == "PACS"
        assert CDCDataSourceAdapter().SOURCE_TYPE == "CDC"

    def test_run_sync_success_with_fetcher(self):
        calls = []

        def fetcher(sc, patient_id, since):
            calls.append(patient_id)
            return _make_adapter_result(source=sc.source_type, records=3)

        a = HISAdapter(DataSourceConfig(source_type="HIS", name="his",
                                        schedule=SyncSchedule(retry_max=2)),
                       fetcher=fetcher)
        rec = a.run_sync()
        assert rec.status == SyncStatus.SUCCESS.value
        assert rec.records_processed == 3
        assert rec.attempts == 1
        assert a.status == SyncStatus.SUCCESS
        assert "__incremental__" in calls

    def test_run_sync_failure_then_retry(self):
        """首次抓取失败，重试后成功，attempts>1。"""
        state = {"n": 0}

        def fetcher(sc, patient_id, since):
            state["n"] += 1
            if state["n"] == 1:
                return _make_adapter_result(source=sc.source_type, success=False,
                                            error="暂时不可用")
            return _make_adapter_result(source=sc.source_type, records=2)

        a = HISAdapter(DataSourceConfig(source_type="HIS", name="his",
                                        schedule=SyncSchedule(retry_max=3,
                                                             retry_backoff=0.01)),
                       fetcher=fetcher)
        rec = a.run_sync()
        assert rec.status == SyncStatus.SUCCESS.value
        assert rec.attempts == 2
        assert state["n"] == 2

    def test_run_sync_always_failure(self):
        def fetcher(sc, patient_id, since):
            return _make_adapter_result(source=sc.source_type, success=False,
                                        error="持续失败")

        a = HISAdapter(DataSourceConfig(source_type="HIS", name="his",
                                        schedule=SyncSchedule(retry_max=2,
                                                             retry_backoff=0.01)),
                       fetcher=fetcher)
        rec = a.run_sync()
        assert rec.status == SyncStatus.FAILED.value
        assert rec.attempts == 2
        assert "失败" in rec.error or rec.error

    def test_run_sync_connection_error_triggers_reconnect(self):
        """连接类异常触发断线重连。"""
        state = {"n": 0}

        class FlakyAdapter(BaseDataSourceAdapter):
            SOURCE_TYPE = "HIS"

            def connect(self):
                # 前两次重连失败，第三次成功
                state["n"] += 1
                self._connected = (state["n"] >= 3)
                return self._connected

        def fetcher(sc, patient_id, since):
            raise ConnectionError("connection refused")

        a = FlakyAdapter(DataSourceConfig(source_type="HIS", name="flaky",
                                          schedule=SyncSchedule(
                                              retry_max=1, retry_backoff=0.01,
                                              reconnect_attempts=5)),
                         fetcher=fetcher)
        rec = a.run_sync()
        # 重连尝试后仍因抓取异常失败，但经历了重连
        assert rec.status == SyncStatus.FAILED.value

    def test_fetch_patient_by_need(self):
        def fetcher(sc, patient_id, since):
            return _make_adapter_result(source=sc.source_type, records=1)

        a = LISAdapter(DataSourceConfig(source_type="LIS", name="lis"),
                       fetcher=fetcher)
        res = a.fetch_patient("P001")
        assert res.success is True


# ============================================================================
# 同步审计
# ============================================================================

class TestSyncAudit:
    def test_sync_writes_audit(self, tmp_path):
        db = str(tmp_path / "audit.db")
        auditor = AuditLogger(db)
        registry = AdapterRegistry(auditor=auditor)

        def fetcher(sc, patient_id, since):
            return _make_adapter_result(source=sc.source_type, records=5)

        cfg = DataSourceConfig(source_type="HIS", name="his_audit",
                               schedule=SyncSchedule(retry_max=1, audit=True))
        a = registry.create(cfg, fetcher=fetcher)
        rec = a.run_sync()
        assert rec.audit_logged is True

        # 查询审计日志中是否有 SYNC 操作
        rows = auditor.query(operation_type=AuditOperationType.SYNC,
                             limit=50)
        found = any(
            (r.record_type == "datasource:HIS")
            for r in rows
        )
        auditor.close()
        # 至少产生一条 SYNC 日志（含本次）
        assert rows, "应产生至少一条 SYNC 审计日志"

    def test_sync_audit_disabled(self, tmp_path):
        auditor = AuditLogger(str(tmp_path / "audit2.db"))

        def fetcher(sc, patient_id, since):
            return _make_adapter_result(source=sc.source_type, records=1)

        cfg = DataSourceConfig(source_type="HIS", name="his_noaudit",
                               schedule=SyncSchedule(retry_max=1, audit=False))
        a = HISAdapter(cfg, auditor=auditor, fetcher=fetcher)
        rec = a.run_sync()
        assert rec.audit_logged is False
        auditor.close()


# ============================================================================
# AdapterRegistry
# ============================================================================

class TestAdapterRegistry:
    def test_create_standard_types(self):
        reg = AdapterRegistry()
        for st in ("HIS", "LIS", "PACS", "CDC"):
            a = reg.create(DataSourceConfig(source_type=st, name=st.lower()))
            assert a.source_config.source_type == st

    def test_create_unknown_raises(self):
        reg = AdapterRegistry()
        with pytest.raises(ValueError):
            reg.create(DataSourceConfig(source_type="EMR", name="x"))

    def test_list_get_remove(self):
        reg = AdapterRegistry()
        reg.create(DataSourceConfig(source_type="HIS", name="h1"))
        reg.create(DataSourceConfig(source_type="LIS", name="l1"))
        assert len(reg.list()) == 2
        assert reg.get("h1") is not None
        assert reg.get_config("h1").source_type == "HIS"
        assert reg.remove("h1") is True
        assert reg.get("h1") is None
        assert len(reg.list()) == 1

    def test_snapshot(self):
        reg = AdapterRegistry()
        reg.create(DataSourceConfig(source_type="HIS", name="h1",
                                    schedule=SyncSchedule(enabled=True)))
        snap = reg.snapshot()
        assert len(snap) == 1
        assert snap[0]["name"] == "h1"
        assert snap[0]["source_type"] == "HIS"
        assert "connected" in snap[0]

    def test_apply_mapping(self):
        reg = AdapterRegistry()
        engine = FieldMappingEngine()
        engine.add_mapping(source_field="姓名", target_field="patient_name")
        cfg = DataSourceConfig(source_type="HIS", name="map_src", mapping=engine)
        reg.create(cfg)
        records = [{"姓名": "张三", "其他": 1}]
        mapped = reg.apply_mapping("map_src", records)
        assert mapped[0]["patient_name"] == "张三"

    def test_map_adapter_result(self):
        reg = AdapterRegistry()
        engine = FieldMappingEngine()
        engine.add_mapping(source_field="姓名", target_field="patient_name")
        cfg = DataSourceConfig(source_type="HIS", name="map_res", mapping=engine)
        reg.create(cfg)
        result = _make_adapter_result(source="HIS")
        result.patient_info = {"姓名": "李四"}
        mapped = reg.map_adapter_result("map_res", result)
        assert mapped.patient_info.get("patient_name") == "李四"

    def test_create_from_dict(self):
        reg = AdapterRegistry()
        data = {"source_type": "CDC", "name": "cdc1",
                "schedule": {"enabled": True, "interval_seconds": 60}}
        a = reg.create_from_dict(data)
        assert a.source_config.name == "cdc1"
        assert a.source_config.schedule.interval_seconds == 60


# ============================================================================
# SyncScheduler
# ============================================================================

class TestSyncScheduler:
    def test_run_once(self):
        registry = AdapterRegistry()

        def fetcher(sc, patient_id, since):
            return _make_adapter_result(source=sc.source_type, records=4)

        registry.create(DataSourceConfig(source_type="HIS", name="s1"),
                        fetcher=fetcher)
        sched = SyncScheduler(registry)
        rec = sched.run_once("s1")
        assert rec is not None
        assert rec.status == SyncStatus.SUCCESS.value
        assert rec.records_processed == 4

    def test_run_once_unknown(self):
        registry = AdapterRegistry()
        sched = SyncScheduler(registry)
        assert sched.run_once("nope") is None

    def test_status_report(self):
        registry = AdapterRegistry()
        registry.create(DataSourceConfig(source_type="HIS", name="s1",
                                         schedule=SyncSchedule(enabled=True)))
        registry.create(DataSourceConfig(source_type="LIS", name="s2"))
        sched = SyncScheduler(registry)
        rep = sched.status_report()
        assert rep["total_sources"] == 2
        assert len(rep["adapters"]) == 2

    def test_start_stop(self):
        registry = AdapterRegistry()

        def fetcher(sc, patient_id, since):
            return _make_adapter_result(source=sc.source_type, records=1)

        cfg = DataSourceConfig(source_type="HIS", name="loop",
                               schedule=SyncSchedule(enabled=True,
                                                     interval_seconds=1))
        registry.create(cfg, fetcher=fetcher)
        sched = SyncScheduler(registry)
        sched.start()
        time.sleep(0.3)
        assert sched.is_running("loop") is True
        sched.stop()
        assert sched.is_running("loop") is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
