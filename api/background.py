#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""异步任务管理器（批量评估等耗时操作的后台执行）。

纯 Python（无 FastAPI 依赖），基于线程池实现任务提交/状态查询/结果获取。
批量评估可能耗时较长，故提供异步接口：提交后立即返回 job_id，
客户端轮询 ``GET /jobs/{job_id}`` 获取状态与结果。

设计：
- 线程池（``ThreadPoolExecutor``），默认 max_workers=4。
- 任务状态机：queued → running → succeeded / failed。
- 结果保存最近 ``max_retention`` 个任务，超限自动清理（防内存膨胀）。
- 线程安全；支持优雅关闭（等待当前任务完成）。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger("tb_risk.api.background")


# 任务状态常量
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"


class JobManager:
    """基于线程池的异步任务管理器。

    用法：
        mgr = JobManager()
        job_id = mgr.submit("batch_assess", fn, arg1, arg2)
        status = mgr.get_status(job_id)   # {'id','status',...}
        result = mgr.get_result(job_id)   # 完成后的返回值或错误信息
    """

    def __init__(self, max_workers: int = 4, max_retention: int = 200):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tb-risk-api-job",
        )
        self._max_retention = max_retention
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 提交
    # ------------------------------------------------------------------

    def submit(self, kind: str, fn: Callable, *args, **kwargs) -> str:
        """提交一个后台任务，立即返回 job_id。

        参数：
            kind: 任务类型（'batch_assess' / 'train' / ...）
            fn: 要执行的函数
            *args/**kwargs: 传给 fn 的参数
        """
        job_id = uuid.uuid4().hex[:16]
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "kind": kind,
                "status": STATUS_QUEUED,
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "progress": 0,
                "message": "已提交",
                "result": None,
                "error": None,
            }
            self._prune_locked()
        future = self._executor.submit(self._run, job_id, fn, args, kwargs)
        # 在后台线程完成时更新状态（不阻塞）
        future.add_done_callback(lambda f: None)
        LOGGER.info("任务已提交: id=%s kind=%s", job_id, kind)
        return job_id

    def _run(self, job_id: str, fn: Callable, args, kwargs) -> None:
        """在线程池中执行任务并更新状态。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["status"] = STATUS_RUNNING
            job["started_at"] = time.time()
        try:
            result = fn(*args, **kwargs)
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = STATUS_SUCCEEDED
                job["finished_at"] = time.time()
                job["progress"] = 100
                job["message"] = "完成"
                job["result"] = result
        except Exception as e:  # noqa: BLE001 - 需捕获全部异常并记录
            LOGGER.warning("任务执行失败: id=%s kind=%s error=%s",
                           job_id, job.get("kind") if job else "?", e)
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = STATUS_FAILED
                job["finished_at"] = time.time()
                job["error"] = str(e)
                job["message"] = "失败"

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态（不含结果/错误详情）。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "id": job["id"],
                "kind": job["kind"],
                "status": job["status"],
                "created_at": job["created_at"],
                "started_at": job["started_at"],
                "finished_at": job["finished_at"],
                "progress": job["progress"],
                "message": job["message"],
            }

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取任务完整信息（含结果/错误）。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            item = dict(job)
            item["result"] = job["result"]
            item["error"] = job["error"]
            return item

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """列出最近任务的状态（按提交时间倒序）。"""
        with self._lock:
            ordered = sorted(self._jobs.values(), key=lambda j: -j["created_at"])
            return [
                {
                    "id": j["id"],
                    "kind": j["kind"],
                    "status": j["status"],
                    "created_at": j["created_at"],
                    "progress": j["progress"],
                    "message": j["message"],
                }
                for j in ordered[:limit]
            ]

    def count(self) -> int:
        """当前跟踪的任务数。"""
        with self._lock:
            return len(self._jobs)

    # ------------------------------------------------------------------
    # 清理与关闭
    # ------------------------------------------------------------------

    def _prune_locked(self) -> None:
        """清理超过保留上限的最旧任务（需持有锁）。"""
        if len(self._jobs) <= self._max_retention:
            return
        # 按创建时间升序，移除最旧的
        ordered = sorted(self._jobs.values(), key=lambda j: j["created_at"])
        for job in ordered[: len(self._jobs) - self._max_retention]:
            self._jobs.pop(job["id"], None)

    def clear_finished(self) -> int:
        """移除所有已结束（成功/失败）的任务，返回移除数量。"""
        removed = 0
        with self._lock:
            for jid in list(self._jobs.keys()):
                if self._jobs[jid]["status"] in (STATUS_SUCCEEDED, STATUS_FAILED):
                    self._jobs.pop(jid, None)
                    removed += 1
        return removed

    def shutdown(self, wait: bool = True) -> None:
        """关闭线程池（等待当前任务完成）。"""
        self._executor.shutdown(wait=wait)


__all__ = [
    "JobManager",
    "STATUS_QUEUED", "STATUS_RUNNING", "STATUS_SUCCEEDED", "STATUS_FAILED",
]