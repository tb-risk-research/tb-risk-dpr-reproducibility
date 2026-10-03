#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接口监控、熔断与告警模块

提供：
- CircuitBreaker: 熔断器模式（连续失败N次后熔断一段时间）
- MetricsCollector: 接口指标采集（可用率、平均响应时间、P95、数据量）
- InterfaceMonitor: 监控聚合器，整合熔断+指标+告警回调
"""

from __future__ import annotations

import collections
import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("tb_risk.health_interop.monitoring")


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """告警事件"""
    level: AlertLevel
    adapter: str
    endpoint: str
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """熔断器

    状态转换：closed → open（连续失败达到阈值） → half-open（熔断时长过后）
    half-open 状态下放行少量请求试探，成功则回到 closed，失败则重新 open。
    """

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: float = 300.0,
                 half_open_max_calls: int = 2):
        """
        参数：
            failure_threshold: 连续失败次数达到此阈值时熔断
            recovery_timeout: 熔断恢复等待时间（秒），默认5分钟
            half_open_max_calls: half-open 状态下允许试探的请求数
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def _maybe_transition_to_half_open(self):
        """检查是否该从 open 转到 half-open（调用方需持有锁）"""
        if (self._state == self.STATE_OPEN and
                time.time() - self._last_failure_time >= self.recovery_timeout):
            self._state = self.STATE_HALF_OPEN
            self._half_open_calls = 0
            LOGGER.info("熔断器从 OPEN 转为 HALF_OPEN")

    def allow_request(self) -> bool:
        """判断是否允许请求通过"""
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == self.STATE_CLOSED:
                return True
            if self._state == self.STATE_OPEN:
                return False
            # half-open：允许有限次试探
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def record_success(self):
        """记录一次成功调用"""
        with self._lock:
            if self._state == self.STATE_HALF_OPEN:
                LOGGER.info("熔断器 HALF_OPEN 探测成功，恢复为 CLOSED")
            self._state = self.STATE_CLOSED
            self._failure_count = 0
            self._half_open_calls = 0

    def record_failure(self):
        """记录一次失败调用"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == self.STATE_HALF_OPEN:
                # 试探失败，重新熔断
                self._state = self.STATE_OPEN
                self._half_open_calls = 0
                LOGGER.warning("熔断器 HALF_OPEN 探测失败，重新 OPEN")
            elif (self._state == self.STATE_CLOSED and
                  self._failure_count >= self.failure_threshold):
                self._state = self.STATE_OPEN
                LOGGER.warning("熔断器达到失败阈值 %d，切换为 OPEN，"
                               "将在 %.0f 秒后尝试恢复",
                               self.failure_threshold, self.recovery_timeout)

    def reset(self):
        """手动重置熔断器"""
        with self._lock:
            self._state = self.STATE_CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            self._last_failure_time = 0


class MetricsCollector:
    """接口指标采集器（滑动窗口）

    采集的指标：
    - 可用率（成功请求数 / 总请求数）
    - 平均响应时间（ms）
    - P95响应时间（ms）
    - 最近一次请求时间
    - 数据量（每次拉取记录数）
    """

    def __init__(self, window_size: int = 200):
        """
        参数：
            window_size: 滑动窗口大小（保留最近N次调用记录）
        """
        self.window_size = window_size
        self._durations: Dict[str, collections.deque] = {}     # endpoint → deque[ms]
        self._successes: Dict[str, int] = collections.defaultdict(int)
        self._failures: Dict[str, int] = collections.defaultdict(int)
        self._data_counts: Dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    def _ensure_endpoint(self, key: str):
        if key not in self._durations:
            self._durations[key] = collections.deque(maxlen=self.window_size)
            self._data_counts[key] = collections.deque(maxlen=self.window_size)

    def record_success(self, adapter: str, endpoint: str,
                       duration_ms: float, data_count: int = 0):
        """记录成功调用"""
        key = f"{adapter}:{endpoint}"
        with self._lock:
            self._ensure_endpoint(key)
            self._durations[key].append(duration_ms)
            self._data_counts[key].append(data_count)
            self._successes[key] += 1

    def record_failure(self, adapter: str, endpoint: str,
                       duration_ms: float = 0):
        """记录失败调用"""
        key = f"{adapter}:{endpoint}"
        with self._lock:
            self._ensure_endpoint(key)
            self._durations[key].append(duration_ms)
            self._failures[key] += 1

    def get_metrics(self, adapter: Optional[str] = None
                    ) -> Dict[str, Dict[str, Any]]:
        """获取指标快照

        参数：
            adapter: 过滤指定适配器；None返回全部
        """
        with self._lock:
            result = {}
            for key, durations in self._durations.items():
                if adapter and not key.startswith(f"{adapter}:"):
                    continue
                adp, endp = key.split(":", 1)
                succ = self._successes.get(key, 0)
                fail = self._failures.get(key, 0)
                total = succ + fail
                durs = list(durations)
                metrics = {
                    "adapter": adp,
                    "endpoint": endp,
                    "total_requests": total,
                    "successes": succ,
                    "failures": fail,
                    "availability": (succ / total * 100) if total > 0 else 100.0,
                    "avg_duration_ms": statistics.mean(durs) if durs else 0,
                    "p95_duration_ms": (sorted(durs)[int(len(durs) * 0.95)]
                                        if len(durs) >= 5 else max(durs) if durs else 0),
                    "max_duration_ms": max(durs) if durs else 0,
                    "data_count_avg": (statistics.mean(list(self._data_counts[key]))
                                       if self._data_counts[key] else 0),
                    "data_count_total": sum(self._data_counts[key]),
                }
                result[key] = metrics
            return result

    def reset(self):
        """清空所有指标"""
        with self._lock:
            self._durations.clear()
            self._successes.clear()
            self._failures.clear()
            self._data_counts.clear()


class InterfaceMonitor:
    """接口监控聚合器

    整合熔断器、指标采集、告警回调。
    适配器通过 set_monitor(this) 注入后，record_success/record_failure 会自动：
    1. 收集指标
    2. 更新熔断器状态
    3. 检查告警条件并触发回调
    """

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 300.0,
                 availability_threshold: float = 99.0,
                 avg_duration_threshold_ms: float = 3000.0,
                 consecutive_failures_alert: int = 3):
        self.metrics = MetricsCollector()
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._consecutive_failures: Dict[str, int] = collections.defaultdict(int)
        self._lock = threading.Lock()

        # 告警阈值
        self.availability_threshold = availability_threshold
        self.avg_duration_threshold_ms = avg_duration_threshold_ms
        self.consecutive_failures_alert = consecutive_failures_alert
        self._breaker_config = (failure_threshold, recovery_timeout)

    def register_alert_callback(self, cb: Callable[[Alert], None]):
        """注册告警回调（例如写日志、弹窗、发送通知）"""
        self._alert_callbacks.append(cb)

    def _get_breaker(self, key: str) -> CircuitBreaker:
        with self._lock:
            if key not in self._breakers:
                thresh, timeout = self._breaker_config
                self._breakers[key] = CircuitBreaker(thresh, timeout)
            return self._breakers[key]

    def allow_request(self, adapter: str, endpoint: str) -> bool:
        """判断是否允许请求（经过熔断器）"""
        key = f"{adapter}:{endpoint}"
        return self._get_breaker(key).allow_request()

    def record_success(self, adapter: str, endpoint: str,
                       duration_ms: float, data_count: int = 0):
        key = f"{adapter}:{endpoint}"
        self.metrics.record_success(adapter, endpoint, duration_ms, data_count)
        self._get_breaker(key).record_success()
        with self._lock:
            self._consecutive_failures[key] = 0
        self._check_thresholds(adapter, endpoint, duration_ms, success=True)

    def record_failure(self, adapter: str, endpoint: str,
                       error: str = "", duration_ms: float = 0):
        key = f"{adapter}:{endpoint}"
        self.metrics.record_failure(adapter, endpoint, duration_ms)
        self._get_breaker(key).record_failure()
        with self._lock:
            self._consecutive_failures[key] += 1
            cf = self._consecutive_failures[key]

        # 连续失败告警
        if cf >= self.consecutive_failures_alert:
            self._fire_alert(Alert(
                level=AlertLevel.CRITICAL if cf >= self.consecutive_failures_alert * 2
                else AlertLevel.WARNING,
                adapter=adapter, endpoint=endpoint,
                message=f"连续 {cf} 次调用失败，最近错误: {error[:200]}",
                details={"consecutive_failures": cf, "error": error},
            ))

        # 熔断告警
        if self._get_breaker(key).state == CircuitBreaker.STATE_OPEN and cf == 1:
            self._fire_alert(Alert(
                level=AlertLevel.CRITICAL,
                adapter=adapter, endpoint=endpoint,
                message=f"接口已熔断：连续失败达到阈值，将在 "
                        f"{self._breaker_config[1]} 秒后尝试恢复",
                details={"error": error},
            ))

    def _check_thresholds(self, adapter: str, endpoint: str,
                          duration_ms: float, success: bool):
        """检查是否触发可用率/响应时间阈值告警"""
        metrics = self.metrics.get_metrics(adapter)
        key = f"{adapter}:{endpoint}"
        m = metrics.get(key)
        if not m or m["total_requests"] < 5:
            return

        # 可用率告警
        if m["availability"] < self.availability_threshold:
            self._fire_alert(Alert(
                level=AlertLevel.WARNING,
                adapter=adapter, endpoint=endpoint,
                message=f"接口可用率 {m['availability']:.1f}% "
                        f"低于阈值 {self.availability_threshold}%",
                details=dict(m),
            ))

        # 响应时间告警
        if m["avg_duration_ms"] > self.avg_duration_threshold_ms:
            self._fire_alert(Alert(
                level=AlertLevel.WARNING,
                adapter=adapter, endpoint=endpoint,
                message=f"平均响应时间 {m['avg_duration_ms']:.0f}ms "
                        f"超过阈值 {self.avg_duration_threshold_ms}ms",
                details=dict(m),
            ))

    def _fire_alert(self, alert: Alert):
        """触发告警回调"""
        LOGGER.log(
            logging.WARNING if alert.level == AlertLevel.WARNING else logging.ERROR,
            "[%s] %s/%s: %s",
            alert.level.value.upper(), alert.adapter, alert.endpoint,
            alert.message
        )
        for cb in self._alert_callbacks:
            try:
                cb(alert)
            except Exception as e:
                LOGGER.error("告警回调执行失败: %s", e)

    def get_status_report(self) -> Dict[str, Any]:
        """生成监控状态报告（用于GUI显示或健康检查）"""
        metrics = self.metrics.get_metrics()
        breakers = {k: {"state": b.state, "failures": b._failure_count}
                    for k, b in self._breakers.items()}
        return {
            "metrics": metrics,
            "circuit_breakers": breakers,
            "consecutive_failures": dict(self._consecutive_failures),
            "thresholds": {
                "availability": self.availability_threshold,
                "avg_duration_ms": self.avg_duration_threshold_ms,
                "consecutive_failures_alert": self.consecutive_failures_alert,
            }
        }
