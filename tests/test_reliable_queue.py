#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可靠消息队列模块单元测试"""

import os
import sqlite3
import tempfile
import threading
import time
import pytest

from health_interop.messaging.reliable_queue import (
    ReliableMessageQueue, ReliableMessage, MessageStatus, QueueType
)


class QueueDBManager:
    """测试用SQLite数据库管理器"""
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
    
    def execute(self, sql: str, params=None):
        cursor = self.conn.cursor()
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        self.conn.commit()
        if sql.strip().upper().startswith(("SELECT", "PRAGMA")):
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        return None
    
    def close(self):
        self.conn.close()


class TestReliableMessage:
    """测试可靠消息数据类"""

    def test_message_creation_defaults(self):
        """测试消息创建默认值"""
        msg = ReliableMessage(
            message_id="msg-001",
            message_type="test.event",
            payload={"key": "value"}
        )
        assert msg.message_id == "msg-001"
        assert msg.status == MessageStatus.PENDING
        assert msg.retry_count == 0
        assert msg.max_retries == 3
        assert msg.queue_type == QueueType.MAIN


class TestMemoryQueue:
    """测试内存模式消息队列（db_manager=None）"""

    def _make_queue(self, **kwargs):
        return ReliableMessageQueue(db_manager=None, node_id="test-node", **kwargs)

    def test_publish_message(self):
        """测试发布消息"""
        q = self._make_queue()
        # 此测试仅验证 publish 生成消息ID，不验证消费，因此允许无handler入队
        msg_id = q.publish("test.event", {"data": "hello"}, allow_no_handler=True)
        assert msg_id is not None
        assert len(msg_id) > 0
        q.shutdown()

    def test_publish_and_process(self):
        """测试发布-处理-确认流程"""
        q = self._make_queue()
        processed = []
        
        def handler(message):
            processed.append(message)
            return True
        
        q.register_handler("test.event", handler)
        msg_id = q.publish("test.event", {"value": 42})
        
        q.start_consuming(poll_interval=0.05)
        time.sleep(0.5)
        q.stop_consuming()
        
        assert len(processed) == 1
        assert processed[0].message_id == msg_id
        assert processed[0].payload["value"] == 42
        q.shutdown()

    def test_message_retry_on_failure(self):
        """测试消息失败后重试"""
        q = self._make_queue(max_retries=2, base_retry_delay=0.05)
        call_count = {"n": 0}
        
        def failing_handler(message):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise Exception("Temporary error")
            return True
        
        q.register_handler("retry.test", failing_handler)
        q.publish("retry.test", {"data": "will retry"}, max_retries=2)
        
        q.start_consuming(poll_interval=0.05)
        time.sleep(1.0)
        q.stop_consuming()
        
        # 至少被处理2次（第一次失败后重试成功）
        assert call_count["n"] >= 2
        q.shutdown()

    def test_dead_letter_after_max_retries(self):
        """测试超过最大重试次数后进入死信队列"""
        q = self._make_queue(max_retries=2, base_retry_delay=0.05)
        
        def always_fail(message):
            raise Exception("Permanent failure")
        
        q.register_handler("dlq.test", always_fail)
        q.publish("dlq.test", {"data": "fail"}, max_retries=2)
        
        q.start_consuming(poll_interval=0.05)
        time.sleep(1.5)
        q.stop_consuming()
        
        dlq = q.get_dead_letters()
        assert len(dlq) >= 1
        assert dlq[0].status == MessageStatus.DEAD
        q.shutdown()

    def test_dedup_idempotent(self):
        """测试幂等去重"""
        q = self._make_queue()
        count = {"n": 0}
        
        def handler(message):
            count["n"] += 1
            return True
        
        q.register_handler("dedup.test", handler)
        dedup_key = "unique-key-123"
        
        q.publish("dedup.test", {"v": 1}, dedup_key=dedup_key)
        q.publish("dedup.test", {"v": 2}, dedup_key=dedup_key)
        q.publish("dedup.test", {"v": 3}, dedup_key=dedup_key)
        
        q.start_consuming(poll_interval=0.05)
        time.sleep(0.5)
        q.stop_consuming()
        
        assert count["n"] == 1
        q.shutdown()

    def test_exponential_backoff(self):
        """测试指数退避延迟递增"""
        q = self._make_queue()
        
        delay0 = q._calculate_retry_delay(0)
        delay1 = q._calculate_retry_delay(1)
        delay2 = q._calculate_retry_delay(2)
        
        assert delay0 <= delay1 <= delay2
        assert delay0 > 0
        q.shutdown()

    def test_atomic_fetch_no_duplicate(self):
        """测试原子获取+标记避免重复消费"""
        q = self._make_queue()
        for i in range(10):
            q.publish("fetch.test", {"i": i}, allow_no_handler=True)
        
        # 两个"线程"同时获取
        fetched1 = q._fetch_pending_messages(limit=5)
        fetched2 = q._fetch_pending_messages(limit=5)
        
        ids1 = {m.message_id for m in fetched1}
        ids2 = {m.message_id for m in fetched2}
        # 不应有重叠
        assert len(ids1 & ids2) == 0
        q.shutdown()


class TestPersistentQueue:
    """测试持久化模式消息队列"""

    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = self._tmp.name
        self.db = QueueDBManager(self.db_path)

    def teardown_method(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_persist_and_recover(self):
        """测试消息持久化后重启不丢失"""
        q1 = ReliableMessageQueue(db_manager=self.db, node_id="n1")
        q1.publish("persist.test", {"data": "survives"}, allow_no_handler=True)
        stats1 = q1.get_queue_stats()
        assert stats1['pending_count'] >= 1
        q1.shutdown()
        
        q2 = ReliableMessageQueue(db_manager=self.db, node_id="n2")
        stats2 = q2.get_queue_stats()
        assert stats2['pending_count'] >= 1
        q2.shutdown()

    def test_insert_or_replace_no_error(self):
        """测试INSERT OR REPLACE避免主键冲突"""
        q = ReliableMessageQueue(db_manager=self.db, node_id="n1")
        
        msg = ReliableMessage(
            message_id="same-id",
            message_type="test",
            payload={"version": 1}
        )
        q._save_message(msg)
        
        msg.payload["version"] = 2
        msg.status = MessageStatus.PROCESSING
        q._save_message(msg)  # 不应抛异常
        
        msg2 = q._get_message("same-id")
        assert msg2 is not None
        assert msg2.payload["version"] == 2
        q.shutdown()


class TestQueueConcurrency:
    """测试消息队列线程安全"""

    def test_concurrent_publish(self):
        """测试多线程并发发布不丢失"""
        q = ReliableMessageQueue(db_manager=None, node_id="test")
        num_threads = 5
        msgs_per_thread = 20
        
        def publish_batch():
            for i in range(msgs_per_thread):
                q.publish("concurrent.test", {"i": i}, allow_no_handler=True)
        
        threads = [threading.Thread(target=publish_batch) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        stats = q.get_queue_stats()
        assert stats['pending_count'] == num_threads * msgs_per_thread
        q.shutdown()
