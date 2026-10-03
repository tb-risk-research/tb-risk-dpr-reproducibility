#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API 安全：API Key 认证 + 滑动窗口限流。

纯 Python（无 FastAPI 依赖），供 API 应用层（FastAPI 依赖注入）与测试复用。

- ``APIKeyStore``：校验请求头携带的 API Key 是否在允许集合内。
- ``RateLimiter``：滑动窗口限流，按 client（API Key / IP）统计每分钟请求数，
  超过上限返回 True（应拒绝）。线程安全，支持清理过期窗口。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Set

LOGGER = logging.getLogger("tb_risk.api.security")


class APIKeyStore:
    """API Key 校验器。

    用法：
        store = APIKeyStore(keys=["k1", "k2"])
        store.is_valid("X-API-Key", "k1")    # True
        store.is_valid("X-API-Key", "bad")    # False
    """

    def __init__(self, keys: Optional[List[str]] = None):
        self._keys: Set[str] = set(keys or [])
        # 记录认证是否被"启用"过。构造时提供 Key 或调用 add() 都会启用；
        # 一旦启用，即使 Key 被全部移除也不放行（拒绝所有请求）。
        self._configured: bool = bool(self._keys)

    def add(self, key: str) -> None:
        """添加一个允许的 API Key。"""
        if key:
            self._keys.add(key)
            self._configured = True

    def remove(self, key: str) -> None:
        """移除一个 API Key。"""
        self._keys.discard(key)

    def is_valid(self, header_name: str, header_value) -> bool:
        """校验请求头中的 API Key 是否有效。

        参数：
            header_name: 请求头字段名（用于日志，不参与匹配）
            header_value: 请求头值（可能为 None 或列表）

        语义：
        - 认证未启用（从未配置任何 Key）时放行（仅限本地/测试）。
        - 认证已启用时，传入的每个 Key 都必须合法，任一非法即拒绝。
        """
        if not self._configured:
            # 未配置任何 Key：视为关闭认证（仅限本地/测试）
            return True
        if not header_value:
            return False
        # Starlette 的请求头值为 str 或 list[str]
        candidates = header_value if isinstance(header_value, list) else [header_value]
        if all(c in self._keys for c in candidates):
            return True
        LOGGER.info("API Key 校验失败 (header=%s)", header_name)
        return False

    @property
    def enabled(self) -> bool:
        """是否启用了认证（曾配置过 Key）。"""
        return self._configured

    def __len__(self) -> int:
        return len(self._keys)


class RateLimiter:
    """滑动窗口限流器（按 client 维度）。

    记录每个 client 在每个时间窗口内的请求时间戳，超出 ``limit`` 时拒绝。
    窗口宽度固定为 ``window_seconds``，滑动窗口为"最近 window_seconds 内"。

    线程安全；``allow`` 内部惰性清理过期时间戳，避免无界增长。
    """

    def __init__(self, limit: int = 60, window_seconds: int = 60,
                 max_clients: int = 10000):
        """
        参数：
            limit: 窗口内允许的最大请求数（0 表示不限流）
            window_seconds: 窗口宽度（秒）
            max_clients: 可跟踪的最大 client 数（超出后拒绝新 client，防内存膨胀）
        """
        self._limit = limit
        self._window = window_seconds
        self._max_clients = max_clients
        self._lock = threading.Lock()
        self._hits: Dict[str, List[float]] = {}

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def window_seconds(self) -> int:
        return self._window

    def allow(self, client: str) -> bool:
        """判断 client 是否允许本次请求（并记录本次请求）。

        返回 True 表示允许，False 表示应拒绝（超出限流）。
        """
        if self._limit <= 0:
            return True
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            hits = self._hits.get(client)
            if hits is None:
                if len(self._hits) >= self._max_clients:
                    LOGGER.warning("限流器超出最大 client 跟踪数，拒绝 %s", client)
                    return False
                hits = []
                self._hits[client] = hits
            # 清理过期时间戳（保持列表有序，因为按时间追加）
            while hits and hits[0] < cutoff:
                hits.pop(0)
            if len(hits) >= self._limit:
                LOGGER.info("限流触发: client=%s hits=%d limit=%d",
                            client, len(hits), self._limit)
                return False
            hits.append(now)
            return True

    def get_count(self, client: str) -> int:
        """返回 client 在最近窗口内的请求数。"""
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            hits = self._hits.get(client, [])
            return sum(1 for t in hits if t >= cutoff)

    def reset(self, client: Optional[str] = None) -> None:
        """清空指定 client（或全部）的请求记录。"""
        with self._lock:
            if client is None:
                self._hits.clear()
            else:
                self._hits.pop(client, None)

    def prune_expired(self) -> int:
        """清理所有 client 的过期时间戳，返回清理的条目数。"""
        now = time.monotonic()
        cutoff = now - self._window
        removed = 0
        with self._lock:
            for client in list(self._hits.keys()):
                hits = self._hits[client]
                before = len(hits)
                hits[:] = [t for t in hits if t >= cutoff]
                removed += before - len(hits)
                if not hits:
                    del self._hits[client]
        return removed


__all__ = ["APIKeyStore", "RateLimiter"]