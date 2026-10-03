#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HL7 消息持久化队列

提供：
- 消息持久化到本地SQLite数据库
- 死信队列（处理失败的消息）
- 人工重试机制
- 不丢消息保障：收到消息后先持久化再返回ACK
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger("tb_risk.health_interop.hl7v2.queue")

QUEUE_STATUS_PENDING = "pending"
QUEUE_STATUS_PROCESSING = "processing"
QUEUE_STATUS_DONE = "done"
QUEUE_STATUS_FAILED = "failed"
QUEUE_STATUS_DEAD = "dead"


@dataclass
class QueuedMessage:
    id: int
    message_type: str
    control_id: str
    sending_app: str
    sending_facility: str
    raw_message: str
    received_at: str
    status: str
    retry_count: int
    last_error: str = ""
    processed_at: Optional[str] = None


class HL7MessageQueue:
    """HL7消息持久化队列（SQLite）"""

    def __init__(self, db_path: str = "hl7_messages.db",
                 max_retries: int = 3):
        self.db_path = db_path
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS hl7_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_type TEXT NOT NULL,
                    control_id TEXT NOT NULL,
                    sending_app TEXT DEFAULT '',
                    sending_facility TEXT DEFAULT '',
                    raw_message TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    retry_count INTEGER DEFAULT 0,
                    last_error TEXT DEFAULT '',
                    processed_at TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_status ON hl7_messages(status);
                CREATE INDEX IF NOT EXISTS idx_control_id ON hl7_messages(control_id);
                CREATE INDEX IF NOT EXISTS idx_received ON hl7_messages(received_at);

                CREATE TABLE IF NOT EXISTS hl7_dead_letter (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_id INTEGER,
                    message_type TEXT NOT NULL,
                    control_id TEXT NOT NULL,
                    raw_message TEXT NOT NULL,
                    error TEXT DEFAULT '',
                    retry_count INTEGER DEFAULT 0,
                    moved_to_dlq_at TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolution_note TEXT DEFAULT ''
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def enqueue(self, raw_message: str, parsed_msg=None) -> int:
        """消息入队（持久化），返回消息ID

        必须在返回ACK前调用，确保消息不丢失。
        """
        msg_type = parsed_msg.message_type if parsed_msg else "UNKNOWN"
        control_id = parsed_msg.message_control_id if parsed_msg else ""
        sending_app = parsed_msg.sending_app if parsed_msg else ""
        sending_facility = parsed_msg.sending_facility if parsed_msg else ""
        now = datetime.now().isoformat()

        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO hl7_messages
                   (message_type, control_id, sending_app, sending_facility,
                    raw_message, received_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (msg_type, control_id, sending_app, sending_facility,
                 raw_message, now, QUEUE_STATUS_PENDING)
            )
            return cursor.lastrowid

    def mark_processing(self, msg_id: int):
        """标记为处理中"""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE hl7_messages SET status=? WHERE id=?",
                (QUEUE_STATUS_PROCESSING, msg_id))

    def mark_done(self, msg_id: int):
        """标记处理成功"""
        now = datetime.now().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE hl7_messages SET status=?, processed_at=?, last_error='' WHERE id=?",
                (QUEUE_STATUS_DONE, now, msg_id))

    def mark_failed(self, msg_id: int, error: str):
        """标记处理失败，自动判断是否进入死信队列"""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT retry_count, message_type, control_id, raw_message FROM hl7_messages WHERE id=?",
                (msg_id,)).fetchone()
            if not row:
                return
            new_retry = row["retry_count"] + 1
            if new_retry >= self.max_retries:
                # 移入死信队列
                now = datetime.now().isoformat()
                conn.execute(
                    """INSERT INTO hl7_dead_letter
                       (original_id, message_type, control_id, raw_message,
                        error, retry_count, moved_to_dlq_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (msg_id, row["message_type"], row["control_id"],
                     row["raw_message"], error, new_retry, now)
                )
                conn.execute(
                    "UPDATE hl7_messages SET status=?, last_error=?, retry_count=? WHERE id=?",
                    (QUEUE_STATUS_DEAD, error, new_retry, msg_id))
                LOGGER.error("消息 %d 进入死信队列（重试%d次）: %s",
                            msg_id, new_retry, error[:200])
            else:
                # 待重试
                conn.execute(
                    "UPDATE hl7_messages SET status=?, last_error=?, retry_count=? WHERE id=?",
                    (QUEUE_STATUS_PENDING, error, new_retry, msg_id))
                LOGGER.warning("消息 %d 处理失败(%d/%d)，将重试: %s",
                               msg_id, new_retry, self.max_retries, error[:200])

    def get_pending(self, limit: int = 10) -> List[QueuedMessage]:
        """获取待处理消息（含重试）"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hl7_messages WHERE status=? ORDER BY id ASC LIMIT ?",
                (QUEUE_STATUS_PENDING, limit)
            ).fetchall()
            return [self._row_to_msg(r) for r in rows]

    def get_dead_letter(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取死信队列消息"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hl7_dead_letter WHERE resolved=0 ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def retry_dead_letter(self, dead_id: int) -> bool:
        """人工重试死信消息"""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT original_id, raw_message FROM hl7_dead_letter WHERE id=?",
                (dead_id,)).fetchone()
            if not row:
                return False
            original_id = row["original_id"]
            # 重置原消息状态为pending
            conn.execute(
                "UPDATE hl7_messages SET status=?, retry_count=0, last_error='' WHERE id=?",
                (QUEUE_STATUS_PENDING, original_id))
            # 标记死信为已处理（人工重试）
            conn.execute(
                "UPDATE hl7_dead_letter SET resolved=1, resolution_note=? WHERE id=?",
                (f"人工重试于 {datetime.now().isoformat()}", dead_id))
            LOGGER.info("死信消息 %d（原消息%d）已标记为重试", dead_id, original_id)
            return True

    def get_queue_stats(self) -> Dict[str, int]:
        """获取队列统计"""
        with self._connect() as conn:
            stats = {}
            for status in (QUEUE_STATUS_PENDING, QUEUE_STATUS_PROCESSING,
                          QUEUE_STATUS_DONE, QUEUE_STATUS_FAILED, QUEUE_STATUS_DEAD):
                c = conn.execute(
                    "SELECT COUNT(*) as c FROM hl7_messages WHERE status=?",
                    (status,)).fetchone()
                stats[status] = c["c"] if c else 0
            dlq = conn.execute(
                "SELECT COUNT(*) as c FROM hl7_dead_letter WHERE resolved=0"
            ).fetchone()
            stats["dead_letter_pending"] = dlq["c"] if dlq else 0
            return stats

    def cleanup_old(self, days: int = 30):
        """清理N天前已完成的消息（保留死信）"""
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM hl7_messages WHERE status=? AND processed_at < datetime('now', ?)",
                (QUEUE_STATUS_DONE, f"-{days} days"))
            deleted = conn.total_changes
            LOGGER.info("已清理 %d 条%d天前已完成消息", deleted, days)

    @staticmethod
    def _row_to_msg(row) -> QueuedMessage:
        return QueuedMessage(
            id=row["id"],
            message_type=row["message_type"],
            control_id=row["control_id"],
            sending_app=row["sending_app"],
            sending_facility=row["sending_facility"],
            raw_message=row["raw_message"],
            received_at=row["received_at"],
            status=row["status"],
            retry_count=row["retry_count"],
            last_error=row["last_error"],
            processed_at=row["processed_at"],
        )
