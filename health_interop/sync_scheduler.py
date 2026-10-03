#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同步调度器——定时拉取 + 失败重试 + 断线重连 + 审计

为已注册的多个业务数据源提供统一的定时同步编排：
- 后台线程按各数据源的调度配置（interval_seconds）定时触发 run_sync()。
- 单轮同步的重试/断线重连/审计由 BaseDataSourceAdapter.run_sync() 完成。
- 调度器负责：启动/停止、按数据源触发、记录调度状态、可注入"成功回调"。

线程模型：
- 每个数据源一个调度线程（便于独立间隔与独立故障隔离）。
- 使用 threading.Event 实现优雅停止。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .adapters.base import BaseDataSourceAdapter, SyncStatus, SyncRunRecord

LOGGER = logging.getLogger("tb_risk.health_interop.sync_scheduler")


class _SourceRunner(threading.Thread):
    """单个数据源的调度线程。"""

    def __init__(self, adapter: BaseDataSourceAdapter,
                 interval_seconds: int,
                 initial_delay_seconds: int,
                 on_result: Optional[Callable[[SyncRunRecord], None]] = None):
        super().__init__(daemon=True, name=f"sync-{adapter.name}")
        self.adapter = adapter
        self.interval = max(interval_seconds, 1)
        self.initial_delay = max(initial_delay_seconds, 0)
        self.on_result = on_result
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        if self.initial_delay > 0 and not self._stop_event.wait(self.initial_delay):
            # 首次拉取
            self._tick()
        # 定时循环
        while not self._stop_event.wait(self.interval):
            self._tick()

    def _tick(self):
        try:
            record = self.adapter.run_sync()
            if self.on_result is not None:
                try:
                    self.on_result(record)
                except Exception as e:  # noqa: BLE001
                    LOGGER.error("[%s] 同步回调失败: %s", self.adapter.name, e)
        except Exception as e:  # noqa: BLE001
            LOGGER.exception("[%s] 定时同步异常: %s", self.adapter.name, e)


class SyncScheduler:
    """多数据源定时同步调度器

    用法：
        registry = AdapterRegistry(auditor=audit_logger)
        registry.create_all(configs)
        scheduler = SyncScheduler(registry)
        scheduler.start()          # 启动所有启用调度的数据源
        scheduler.run_once("HIS")  # 手动立即同步某数据源
        scheduler.stop()
    """

    def __init__(self, registry,
                 on_sync_result: Optional[Callable[[SyncRunRecord], None]] = None):
        """初始化调度器。

        参数：
            registry: AdapterRegistry（含已创建的数据源）
            on_sync_result: 每轮同步完成后的回调（可写审计/通知/UI刷新）
        """
        self.registry = registry
        self.on_sync_result = on_sync_result
        self._runners: Dict[str, _SourceRunner] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self):
        """为所有已启用且 schedule.enabled 的数据源启动调度线程。"""
        with self._lock:
            for adapter in self.registry.list():
                sc = adapter.source_config
                if not sc.enabled or not sc.schedule.enabled:
                    continue
                if adapter.name in self._runners:
                    continue
                runner = _SourceRunner(
                    adapter,
                    interval_seconds=sc.schedule.interval_seconds,
                    initial_delay_seconds=sc.schedule.initial_delay_seconds,
                    on_result=self.on_sync_result,
                )
                self._runners[adapter.name] = runner
                runner.start()
                LOGGER.info("已启动定时同步: %s (间隔 %ds)",
                            adapter.name, sc.schedule.interval_seconds)

    def stop(self):
        """停止所有调度线程（优雅）。"""
        with self._lock:
            for name, runner in list(self._runners.items()):
                runner.stop()
            for runner in list(self._runners.values()):
                runner.join(timeout=3)
            self._runners.clear()

    def is_running(self, name: Optional[str] = None) -> bool:
        """是否有运行中的调度线程。"""
        if name is not None:
            runner = self._runners.get(name)
            return runner is not None and runner.is_alive()
        return any(r.is_alive() for r in self._runners.values())

    # ------------------------------------------------------------------
    # 手动触发
    # ------------------------------------------------------------------

    def run_once(self, source_name: str) -> Optional[SyncRunRecord]:
        """手动立即同步某数据源（阻塞）。"""
        adapter = self.registry.get(source_name)
        if adapter is None:
            LOGGER.warning("数据源不存在: %s", source_name)
            return None
        record = adapter.run_sync()
        if self.on_sync_result is not None:
            try:
                self.on_sync_result(record)
            except Exception as e:  # noqa: BLE001
                LOGGER.error("同步回调失败: %s", e)
        return record

    def run_all_once(self) -> List[SyncRunRecord]:
        """顺序同步所有数据源（阻塞）。"""
        records = []
        for adapter in self.registry.list():
            records.append(adapter.run_sync())
        return records

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def status_report(self) -> Dict[str, Any]:
        """调度状态报告。"""
        adapters = []
        for adapter in self.registry.list():
            adapters.append({
                "name": adapter.name,
                "source_type": adapter.source_config.source_type,
                "schedule_enabled": adapter.source_config.schedule.enabled,
                "thread_running": self.is_running(adapter.name),
                "connected": adapter.is_connected(),
                "status": adapter.status.value,
            })
        return {
            "running_sources": [n for n, r in self._runners.items() if r.is_alive()],
            "total_sources": len(self.registry.list()),
            "adapters": adapters,
        }
