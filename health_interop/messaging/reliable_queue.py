#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可靠消息队列模块

提供企业级消息可靠性保障：
- 消息确认机制（ACK）：至少一次投递保证
- 失败重试与指数退避策略
- 死信队列（DLQ）：多次处理失败的消息存放供人工干预
- 幂等处理：基于消息ID的去重
- 消息顺序保证：按消息入队顺序处理
- 消息持久化：数据库持久化，防止消息丢失

符合：
- 企业集成模式（EIP）
- 医疗数据交换可靠性要求
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import queue
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger("tb_risk.health_interop.messaging")


# ============================================================================
# 消息状态枚举
# ============================================================================

class MessageStatus(Enum):
    """消息状态"""
    PENDING = "pending"           # 待处理
    PROCESSING = "processing"     # 处理中
    ACKED = "acked"               # 已确认（处理成功）
    RETRYING = "retrying"         # 重试中
    DEAD = "dead"                 # 死信（多次处理失败）
    EXPIRED = "expired"           # 已过期


class QueueType(Enum):
    """队列类型"""
    MAIN = "main"                 # 主队列
    RETRY = "retry"               # 重试队列
    DEAD_LETTER = "dead_letter"   # 死信队列


# ============================================================================
# 消息体定义
# ============================================================================

@dataclass
class ReliableMessage:
    """可靠消息"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = ""       # 关联ID（用于请求-响应模式）
    message_type: str = ""         # 消息类型（如HL7_ADT, HL7_ORU, CDC_CALLBACK等）
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # 状态管理
    status: MessageStatus = MessageStatus.PENDING
    queue_type: QueueType = QueueType.MAIN
    
    # 重试配置
    retry_count: int = 0
    max_retries: int = 3
    next_retry_time: Optional[float] = None
    
    # 元数据
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    processed_at: Optional[float] = None
    expires_at: Optional[float] = None
    
    # 处理记录
    last_error: str = ""
    processing_node: str = ""       # 处理节点标识
    dedup_key: str = ""             # 幂等去重键
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'message_id': self.message_id,
            'correlation_id': self.correlation_id,
            'message_type': self.message_type,
            'payload': self.payload,
            'headers': self.headers,
            'status': self.status.value,
            'queue_type': self.queue_type.value,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'next_retry_time': self.next_retry_time,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'processed_at': self.processed_at,
            'expires_at': self.expires_at,
            'last_error': self.last_error,
            'processing_node': self.processing_node,
            'dedup_key': self.dedup_key,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReliableMessage':
        """从字典创建"""
        msg = cls()
        msg.message_id = data.get('message_id', msg.message_id)
        msg.correlation_id = data.get('correlation_id', '')
        msg.message_type = data.get('message_type', '')
        msg.payload = data.get('payload', {})
        msg.headers = data.get('headers', {})
        msg.status = MessageStatus(data.get('status', 'pending'))
        msg.queue_type = QueueType(data.get('queue_type', 'main'))
        msg.retry_count = data.get('retry_count', 0)
        msg.max_retries = data.get('max_retries', 3)
        msg.next_retry_time = data.get('next_retry_time')
        msg.created_at = data.get('created_at', time.time())
        msg.updated_at = data.get('updated_at', time.time())
        msg.processed_at = data.get('processed_at')
        msg.expires_at = data.get('expires_at')
        msg.last_error = data.get('last_error', '')
        msg.processing_node = data.get('processing_node', '')
        msg.dedup_key = data.get('dedup_key', '')
        return msg


# ============================================================================
# 可靠消息队列
# ============================================================================

class ReliableMessageQueue:
    """可靠消息队列
    
    提供至少一次投递保证、死信队列、幂等去重、指数退避重试。
    
    用法：
        queue = ReliableMessageQueue(db_manager=db, node_id='node1')
        
        # 注册消息处理器
        @queue.handler('HL7_ORU')
        def handle_oru(message):
            # 处理检验结果消息
            process_lab_result(message.payload)
        
        # 发布消息
        msg_id = queue.publish('HL7_ORU', {'patient_id': 'P001', 'results': [...]})
        
        # 启动消费
        queue.start_consuming()
    """

    def __init__(self, db_manager=None, node_id: str = "default",
                 data_dir: str = None,
                 max_retries: int = 3,
                 base_retry_delay: float = 1.0,
                 max_retry_delay: float = 300.0,
                 enable_dlq: bool = True,
                 processing_timeout: float = 300.0,
                 message_ttl_days: int = 7,
                 max_file_count_warning: int = 1000,
                 stale_pending_hours: int = 24):
        """初始化可靠消息队列。
        
        参数：
            db_manager: 数据库管理器（优先使用数据库持久化）
            node_id: 节点标识（用于分布式部署时区分处理节点）
            data_dir: 文件存储目录（无数据库时使用）
            max_retries: 默认最大重试次数
            base_retry_delay: 基础重试延迟（秒）
            max_retry_delay: 最大重试延迟（秒）
            enable_dlq: 是否启用死信队列
            processing_timeout: 消息处理超时时间（秒），超时后消息将被重新入队
            message_ttl_days: 已处理消息文件保留天数，超过则清理
            max_file_count_warning: 单目录文件数告警阈值（防inode耗尽）
            stale_pending_hours: PENDING 状态超过此时长（小时）且消费者从未启动则视为
                                 历史残留，启动时发出警告提示清理
        """
        self.db_manager = db_manager
        self.node_id = node_id
        self.data_dir = data_dir or "./msg_queue"
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay
        self.enable_dlq = enable_dlq
        self.processing_timeout = processing_timeout
        self.message_ttl_days = message_ttl_days
        self.max_file_count_warning = max_file_count_warning
        self.stale_pending_hours = stale_pending_hours
        self._file_cleanup_lock = threading.Lock()
        
        # 消息处理器
        self._handlers: Dict[str, Callable] = {}
        
        # 幂等去重缓存（最近处理过的消息ID）
        self._processed_ids: set = set()
        self._processed_ids_max = 10000  # 最多缓存10000个
        
        # 消费线程控制
        self._running = False
        self._consumer_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        
        # 内存队列（无数据库时使用queue.Queue保证线程安全）
        self._memory_main_queue: 'queue.Queue[ReliableMessage]' = queue.Queue()
        self._memory_retry_queue: List[ReliableMessage] = []  # 重试消息按时间排序
        self._memory_dlq: List[ReliableMessage] = []
        self._inflight_messages: Dict[str, ReliableMessage] = {}  # 处理中的消息
        
        # 确保存储
        self._ensure_storage()
        
        # 启动时清理过期文件
        self._cleanup_expired_files()
    
    def _delete_message_file(self, message_id: str):
        """删除消息对应的磁盘文件。"""
        with self._file_cleanup_lock:
            try:
                filename = os.path.join(self.data_dir, f"{message_id}.json")
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                LOGGER.debug("删除消息文件失败: %s, %s", message_id, e)
    
    def _cleanup_expired_files(self):
        """启动时扫描清理消息文件。
        
        执行两层清理：
        1. TTL清理：删除超过保留期限（message_ttl_days）的旧文件
        2. 历史迁移清理：首次启动新清理机制时，扫描所有JSON文件，
           对处于终态（ACKED/DEAD/EXPIRED）的消息文件直接删除，
           解决修复前积累的历史残留文件问题。仅保留PENDING/RETRYING/PROCESSING状态的文件。
        """
        try:
            if not os.path.exists(self.data_dir):
                return
            
            now = time.time()
            ttl_seconds = self.message_ttl_days * 86400
            json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
            file_count = len(json_files)
            
            # 文件数超限告警
            if file_count > self.max_file_count_warning:
                LOGGER.warning(
                    "消息队列目录文件数超过阈值: %d > %d，请关注磁盘空间",
                    file_count, self.max_file_count_warning
                )
            
            # 终态状态集合：这些状态的消息文件可以安全删除
            terminal_statuses = {MessageStatus.ACKED.value, MessageStatus.DEAD.value, MessageStatus.EXPIRED.value}
            
            ttl_cleaned = 0
            legacy_cleaned = 0
            stale_processing_reset = 0
            stale_pending_count = 0
            pending_kept = 0
            pending_total = 0
            stale_pending_threshold = now - self.stale_pending_hours * 3600

            for fname in json_files:
                try:
                    fpath = os.path.join(self.data_dir, fname)
                    mtime = os.path.getmtime(fpath)
                    
                    # 1. 文件修改时间超过TTL则删除（无论状态）
                    if now - mtime > ttl_seconds:
                        os.remove(fpath)
                        ttl_cleaned += 1
                        continue
                    
                    # 2. 历史迁移清理：读取文件状态
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        status = data.get('status', '')
                        if status in terminal_statuses:
                            # 终态消息直接删除
                            os.remove(fpath)
                            legacy_cleaned += 1
                        elif status == MessageStatus.PROCESSING.value:
                            # PROCESSING 状态超过 processing_timeout 视为消费者崩溃，
                            # 重置为 RETRYING 以便重新消费（避免消息永久卡在 PROCESSING）
                            try:
                                updated_at = data.get('updated_at', mtime)
                                if now - updated_at > self.processing_timeout * 2:
                                    data['status'] = MessageStatus.RETRYING.value
                                    data['next_retry_time'] = now
                                    data['retry_count'] = data.get('retry_count', 0) + 1
                                    tmp = fpath + '.tmp'
                                    with open(tmp, 'w', encoding='utf-8') as f:
                                        json.dump(data, f, ensure_ascii=False)
                                    os.replace(tmp, fpath)
                                    stale_processing_reset += 1
                                else:
                                    pending_kept += 1
                            except Exception as _e:
                                LOGGER.debug("恢复PROCESSING消息失败 %s: %s", fname, _e)
                                pending_kept += 1
                        elif status == MessageStatus.PENDING.value:
                            pending_total += 1
                            created_at = data.get('created_at', mtime)
                            if created_at < stale_pending_threshold:
                                stale_pending_count += 1
                            pending_kept += 1
                        elif status == MessageStatus.RETRYING.value:
                            pending_total += 1
                            pending_kept += 1
                        else:
                            pending_kept += 1
                    except (json.JSONDecodeError, IOError, KeyError):
                        # 文件损坏或无法解析，保守保留（不删除）
                        pending_kept += 1
                except Exception as e:
                    LOGGER.debug("清理文件失败: %s, %s", fname, e)
            
            total_cleaned = ttl_cleaned + legacy_cleaned
            if total_cleaned > 0 or stale_processing_reset > 0:
                LOGGER.info(
                    "启动时清理消息文件: TTL过期 %d 个, 历史终态 %d 个, "
                    "重置stale PROCESSING %d 个, 保留待处理 %d 个",
                    ttl_cleaned, legacy_cleaned, stale_processing_reset, pending_kept
                )
            
            # 对长期未消费的 PENDING 消息发出警告
            if stale_pending_count > 0:
                LOGGER.warning(
                    "检测到 %d 条 PENDING 消息超过 %d 小时未被消费。"
                    "请确认是否已注册对应消息处理器并调用 start_consuming() 启动消费。"
                    "若这些是历史残留文件，可手动删除 msg_queue/ 目录清理。",
                    stale_pending_count, self.stale_pending_hours
                )
            # 待处理消息总数过多也警告
            if pending_total > 100:
                LOGGER.warning(
                    "消息队列积压: 当前有 %d 条待处理消息（PENDING+RETRYING），"
                    "请确认消费线程正常运行。",
                    pending_total
                )
        except Exception as e:
            LOGGER.debug("启动清理过期文件异常: %s", e)

    def handler(self, message_type: str):
        """注册消息处理器装饰器。
        
        参数：
            message_type: 消息类型
        """
        def decorator(func: Callable):
            self.register_handler(message_type, func)
            return func
        return decorator

    def register_handler(self, message_type: str, handler: Callable):
        """注册消息处理器。
        
        参数：
            message_type: 消息类型
            handler: 处理函数，接收ReliableMessage参数
        """
        self._handlers[message_type] = handler
        LOGGER.debug("注册消息处理器: %s", message_type)

    # ------------------------------------------------------------------
    # 消息发布
    # ------------------------------------------------------------------

    def publish(self, message_type: str, payload: Dict[str, Any],
                headers: Dict[str, str] = None,
                correlation_id: str = "",
                dedup_key: str = "",
                max_retries: int = None,
                expires_at: float = None,
                allow_no_handler: bool = False) -> str:
        """发布消息到队列。
        
        参数：
            message_type: 消息类型
            payload: 消息内容
            headers: 消息头
            correlation_id: 关联ID
            dedup_key: 幂等去重键（相同键的消息只处理一次）
            max_retries: 该消息的最大重试次数
            expires_at: 过期时间戳
            allow_no_handler: 是否允许发布到无处理器的消息类型（默认 False，
                会抛出 RuntimeError 防止消息黑洞；测试或预注册场景可设为 True）
            
        返回：
            str: 消息ID
            
        异常：
            RuntimeError: 当消息类型未注册处理器且 allow_no_handler=False 时抛出，
                防止消息积压在队列中永远无法被消费
        """
        # 消息黑洞防护：无对应处理器时默认拒绝入队
        if message_type not in self._handlers:
            msg = (
                f"消息类型 '{message_type}' 未注册处理器。消息将积压无法被消费，"
                f"已拒绝入队。请先调用 register_handler('{message_type}', handler_fn) "
                f"注册处理器，或传入 allow_no_handler=True 强制入队（不推荐）。"
            )
            if allow_no_handler:
                LOGGER.warning(msg + "（已通过 allow_no_handler=True 强制入队）")
            else:
                raise RuntimeError(msg)

        # 幂等检查
        if dedup_key and self._is_duplicate(dedup_key):
            LOGGER.info("重复消息（去重键=%s），跳过", dedup_key)
            return ""
        
        message = ReliableMessage(
            message_type=message_type,
            payload=payload,
            headers=headers or {},
            correlation_id=correlation_id,
            dedup_key=dedup_key,
            max_retries=max_retries if max_retries is not None else self.max_retries,
            expires_at=expires_at,
        )
        
        # 先记录到幂等缓存（防止并发场景重复入队）
        if dedup_key:
            self._processed_ids.add(dedup_key)
            if len(self._processed_ids) > self._processed_ids_max:
                self._processed_ids = set(list(self._processed_ids)[-self._processed_ids_max//2:])
        
        # 持久化
        self._save_message(message)
        
        # 添加到内存队列（无数据库时）
        if not self.db_manager:
            self._memory_main_queue.put(message)
        
        LOGGER.debug("消息已入队: %s (类型=%s, ID=%s)", message.message_id, 
                    message_type, message.message_id)
        
        return message.message_id

    def _is_duplicate(self, dedup_key: str) -> bool:
        """检查是否是重复消息（基于幂等键）。"""
        # 先检查内存缓存（包含正在处理和已处理的）
        if dedup_key in self._processed_ids:
            return True
        
        # 检查数据库中是否存在相同dedup_key的消息（不论状态）
        if self.db_manager:
            try:
                result = self.db_manager.execute(
                    "SELECT message_id FROM reliable_messages WHERE dedup_key = ?",
                    [dedup_key]
                )
                if result and len(result) > 0:
                    return True
            except Exception as e:
                LOGGER.debug("检查数据库去重键失败: %s", e)
        
        # 检查内存队列中是否存在
        if not self.db_manager:
            with self._lock:
                # 检查inflight
                for m in self._inflight_messages.values():
                    if m.dedup_key == dedup_key:
                        return True
                for m in self._memory_retry_queue:
                    if m.dedup_key == dedup_key:
                        return True
                # 检查主队列（通过snapshot）
                try:
                    snapshot = list(self._memory_main_queue.queue)
                    for m in snapshot:
                        if m.dedup_key == dedup_key:
                            return True
                except Exception as e:
                    LOGGER.debug("检查内存队列去重键失败: %s", e)
        
        return False

    # ------------------------------------------------------------------
    # 消息消费
    # ------------------------------------------------------------------

    def start_consuming(self, poll_interval: float = 1.0, daemon: bool = True):
        """启动消费线程。
        
        参数：
            poll_interval: 轮询间隔（秒）
            daemon: 是否为守护线程
        """
        if self._running:
            LOGGER.warning("消费线程已在运行")
            return
        
        self._running = True
        self._consumer_thread = threading.Thread(
            target=self._consume_loop,
            args=(poll_interval,),
            daemon=daemon,
            name=f"MsgConsumer-{self.node_id}"
        )
        self._consumer_thread.start()
        LOGGER.info("消息消费线程已启动 (节点=%s)", self.node_id)

    def stop_consuming(self, timeout: float = 5.0):
        """停止消费线程。
        
        参数：
            timeout: 等待线程结束的超时时间
        """
        self._running = False
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=timeout)
        LOGGER.info("消息消费线程已停止")

    def shutdown(self, timeout: float = 30.0, wait_for_inflight: bool = True):
        """优雅关闭队列。
        
        参数：
            timeout: 总关闭超时时间（秒）
            wait_for_inflight: 是否等待正在处理的消息完成
        """
        LOGGER.info("正在关闭消息队列 (节点=%s)...", self.node_id)
        
        # 1. 停止消费循环
        self._running = False
        self._shutdown_event.set()
        
        # 2. 等待消费线程结束
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=min(timeout, 10.0))
        
        # 3. 处理inflight消息（等待完成或超时后重新入队）
        start_time = time.time()
        if wait_for_inflight and self._inflight_messages:
            LOGGER.info("等待 %d 条处理中消息完成...", len(self._inflight_messages))
            while self._inflight_messages and (time.time() - start_time) < timeout:
                time.sleep(0.5)
        
        # 4. 未完成的inflight消息重新标记为PENDING或移入重试
        with self._lock:
            for msg_id, msg in list(self._inflight_messages.items()):
                LOGGER.warning("消息未在关闭前完成: %s, 重新入队", msg_id)
                msg.status = MessageStatus.PENDING
                msg.processing_node = ""
                if self.db_manager:
                    self._update_message(msg)
                else:
                    self._memory_main_queue.put(msg)
            self._inflight_messages.clear()
        
        # 5. 持久化内存队列中的消息到文件（内存模式下）
        if not self.db_manager:
            self._flush_memory_to_disk()
        
        LOGGER.info("消息队列已关闭 (节点=%s)", self.node_id)

    def _consume_loop(self, poll_interval: float):
        """消费循环。"""
        while self._running:
            try:
                self._recover_timed_out_messages()
                self._process_due_messages()
                self._process_retry_queue()
            except Exception as e:
                LOGGER.error("消息消费异常: %s", e, exc_info=True)
            
            time.sleep(poll_interval)

    def _process_due_messages(self):
        """处理到期消息。"""
        messages = self._fetch_pending_messages(limit=10)
        
        for message in messages:
            # 检查过期（在锁外检查，因为消息可能已等待一段时间）
            if message.expires_at and time.time() > message.expires_at:
                self._mark_expired(message)
                continue
            
            try:
                # 处理消息（已在_fetch_pending_messages中标记为PROCESSING）
                self._process_message(message)
                # 处理成功，确认
                self._ack_message(message)
            except Exception as e:
                LOGGER.warning("消息处理失败 (ID=%s, 类型=%s): %s",
                              message.message_id, message.message_type, e)
                # 处理失败，安排重试或移入死信
                self._handle_failure(message, str(e))

    def _recover_timed_out_messages(self):
        """恢复处理超时的消息。
        
        当消费者崩溃或处理卡住时，消息可能永久停留在PROCESSING状态。
        此方法定期扫描超时消息，将其重新入队或移至死信队列。
        """
        now = time.time()
        timeout_threshold = now - self.processing_timeout
        
        if self.db_manager:
            try:
                # 查询超时的PROCESSING消息
                rows = self.db_manager.execute("""
                    SELECT * FROM reliable_messages
                    WHERE status = ? AND updated_at < ?
                """, [MessageStatus.PROCESSING.value, timeout_threshold])
                
                for row in (rows or []):
                    if isinstance(row, dict):
                        msg = self._dict_to_message(row)
                    elif isinstance(row, (tuple, list)):
                        msg = self._row_to_message(row)
                    else:
                        continue
                    
                    LOGGER.warning("检测到处理超时消息: %s (类型=%s, 节点=%s, 已处理%.0f秒)",
                                  msg.message_id, msg.message_type, 
                                  msg.processing_node, now - msg.updated_at)
                    # 超时视为失败，走重试/死信逻辑
                    msg.retry_count += 1
                    self._handle_failure(msg, f"处理超时（超过{self.processing_timeout}秒）")
            except Exception as e:
                LOGGER.debug("超时消息恢复失败: %s", e)
        else:
            # 内存模式：在锁内扫描inflight消息
            with self._lock:
                for msg_id, m in list(self._inflight_messages.items()):
                    if (m.status == MessageStatus.PROCESSING 
                        and m.updated_at < timeout_threshold
                        and m.processing_node == self.node_id):
                        LOGGER.warning("内存队列检测到处理超时消息: %s", m.message_id)
                        del self._inflight_messages[msg_id]
                        m.retry_count += 1
                        # 超时视为失败，走重试/死信逻辑
                        self._handle_failure_nosave(m, f"处理超时（超过{self.processing_timeout}秒）")

    def _process_retry_queue(self):
        """处理重试队列中到期的消息。"""
        now = time.time()
        messages = self._fetch_retry_messages()
        
        for message in messages:
            if message.next_retry_time and message.next_retry_time <= now:
                # 移回主队列重试
                message.queue_type = QueueType.MAIN
                message.status = MessageStatus.PENDING
                message.next_retry_time = None
                self._update_message(message)
                LOGGER.info("消息重试: %s (第%d次)", message.message_id, message.retry_count + 1)
                
                # 内存模式：从重试列表移除并放回主队列
                if not self.db_manager:
                    with self._lock:
                        self._memory_retry_queue = [
                            m for m in self._memory_retry_queue 
                            if m.message_id != message.message_id
                        ]
                    self._memory_main_queue.put(message)

    def _process_message(self, message: ReliableMessage):
        """处理单条消息。"""
        handler = self._handlers.get(message.message_type)
        
        if not handler:
            raise MessageHandlerError(f"未找到消息类型 {message.message_type} 的处理器")
        
        # 记录处理节点
        message.processing_node = self.node_id
        
        # 执行处理器
        handler(message)

    def _ack_message(self, message: ReliableMessage):
        """确认消息处理成功。"""
        message.status = MessageStatus.ACKED
        message.processed_at = time.time()
        message.queue_type = QueueType.MAIN
        message.updated_at = time.time()
        self._update_message(message)
        
        # 从inflight移除
        if not self.db_manager:
            with self._lock:
                self._inflight_messages.pop(message.message_id, None)
        
        # 记录到幂等缓存
        if message.dedup_key:
            self._processed_ids.add(message.dedup_key)
            if len(self._processed_ids) > self._processed_ids_max:
                self._processed_ids = set(list(self._processed_ids)[-self._processed_ids_max//2:])
        
        # 消息确认处理完成后立即删除磁盘文件
        self._delete_message_file(message.message_id)
        
        LOGGER.debug("消息处理成功: %s", message.message_id)

    def _handle_failure(self, message: ReliableMessage, error: str):
        """处理消息失败。"""
        message.retry_count += 1
        message.last_error = error
        message.updated_at = time.time()
        
        if message.retry_count >= message.max_retries:
            if self.enable_dlq:
                self._move_to_dlq(message, error)
            else:
                message.status = MessageStatus.DEAD
                self._update_message(message)
                LOGGER.error("消息处理失败已达最大重试次数: %s, 错误: %s",
                            message.message_id, error)
                # 内存模式：从inflight移除
                if not self.db_manager:
                    with self._lock:
                        self._inflight_messages.pop(message.message_id, None)
        else:
            delay = self._calculate_retry_delay(message.retry_count)
            message.next_retry_time = time.time() + delay
            message.status = MessageStatus.RETRYING
            message.queue_type = QueueType.RETRY
            self._update_message(message)
            LOGGER.info("消息安排重试: %s, 第%d次, 延迟%.1fs",
                       message.message_id, message.retry_count, delay)
            
            # 内存模式：加入重试队列并从inflight移除
            if not self.db_manager:
                with self._lock:
                    self._memory_retry_queue.append(message)
                    self._inflight_messages.pop(message.message_id, None)

    def _handle_failure_nosave(self, message: ReliableMessage, error: str):
        """处理消息失败（内存模式超时恢复用，不依赖_update_message）。"""
        message.last_error = error
        message.updated_at = time.time()
        
        if message.retry_count >= message.max_retries:
            if self.enable_dlq:
                message.status = MessageStatus.DEAD
                message.queue_type = QueueType.DEAD_LETTER
                message.next_retry_time = None
                with self._lock:
                    self._memory_dlq.append(message)
                    self._inflight_messages.pop(message.message_id, None)
                # 死信消息进入终态后删除磁盘文件（已在内存DLQ中保留）
                self._delete_message_file(message.message_id)
                LOGGER.error("消息进入死信队列: %s, 错误: %s, 重试%d次",
                            message.message_id, error, message.retry_count)
            else:
                message.status = MessageStatus.DEAD
                with self._lock:
                    self._inflight_messages.pop(message.message_id, None)
                self._delete_message_file(message.message_id)
        else:
            delay = self._calculate_retry_delay(message.retry_count)
            message.next_retry_time = time.time() + delay
            message.status = MessageStatus.RETRYING
            message.queue_type = QueueType.RETRY
            with self._lock:
                self._memory_retry_queue.append(message)
                self._inflight_messages.pop(message.message_id, None)
            self._save_to_file(message)
            LOGGER.info("消息安排重试: %s, 第%d次, 延迟%.1fs",
                       message.message_id, message.retry_count, delay)

    def _calculate_retry_delay(self, retry_count: int) -> float:
        """计算重试延迟（指数退避 + 随机抖动）。"""
        delay = self.base_retry_delay * (2 ** (retry_count - 1))
        jitter = delay * 0.2 * (random.random() * 2 - 1)
        delay = delay + jitter
        return min(delay, self.max_retry_delay)

    def _move_to_dlq(self, message: ReliableMessage, error: str):
        """将消息移到死信队列。"""
        message.status = MessageStatus.DEAD
        message.queue_type = QueueType.DEAD_LETTER
        self._update_message(message)
        
        if not self.db_manager:
            with self._lock:
                self._memory_dlq.append(message)
                self._inflight_messages.pop(message.message_id, None)
        
        # 死信消息也立即删除磁盘文件（已持久化到数据库或内存DLQ）
        self._delete_message_file(message.message_id)
        
        LOGGER.error("消息进入死信队列: %s, 错误: %s, 重试%d次",
                    message.message_id, error, message.retry_count)

    def _mark_expired(self, message: ReliableMessage):
        """标记消息过期。"""
        message.status = MessageStatus.EXPIRED
        message.updated_at = time.time()
        self._update_message(message)
        
        # 过期消息立即删除磁盘文件
        self._delete_message_file(message.message_id)
        
        LOGGER.info("消息已过期: %s", message.message_id)

    def _mark_processing(self, message: ReliableMessage):
        """标记消息为处理中。"""
        message.status = MessageStatus.PROCESSING
        message.updated_at = time.time()
        self._update_message(message)

    # ------------------------------------------------------------------
    # 死信队列管理
    # ------------------------------------------------------------------

    def get_dead_letters(self, limit: int = 100) -> List[ReliableMessage]:
        """获取死信队列中的消息。"""
        if self.db_manager:
            return self._fetch_dlq_messages(limit)
        
        with self._lock:
            return list(self._memory_dlq[-limit:])

    def replay_dead_letter(self, message_id: str) -> bool:
        """重放死信消息（人工干预后重试）。"""
        message = self._get_message(message_id)
        if not message or message.status != MessageStatus.DEAD:
            return False
        
        message.status = MessageStatus.PENDING
        message.queue_type = QueueType.MAIN
        message.retry_count = 0
        message.next_retry_time = None
        message.last_error = f"重放于 {datetime.datetime.now().isoformat()}"
        self._update_message(message)
        
        # 从内存DLQ移除并加入主队列
        if not self.db_manager:
            with self._lock:
                self._memory_dlq = [m for m in self._memory_dlq if m.message_id != message_id]
            self._memory_main_queue.put(message)
        
        LOGGER.info("死信消息已重放: %s", message_id)
        return True

    def purge_dead_letters(self) -> int:
        """清空死信队列。返回删除的消息数。"""
        count = 0
        if self.db_manager:
            try:
                result = self.db_manager.execute(
                    "DELETE FROM reliable_messages WHERE queue_type = ?",
                    [QueueType.DEAD_LETTER.value]
                )
                count = result.rowcount if hasattr(result, 'rowcount') else 0
            except Exception as e:
                LOGGER.error("清空死信队列失败: %s", e)
        else:
            with self._lock:
                count = len(self._memory_dlq)
                self._memory_dlq.clear()
        
        LOGGER.info("死信队列已清空，删除 %d 条消息", count)
        return count

    # ------------------------------------------------------------------
    # 队列监控
    # ------------------------------------------------------------------

    def get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计信息。"""
        stats = {
            'node_id': self.node_id,
            'registered_handlers': list(self._handlers.keys()),
            'running': self._running,
        }
        
        if self.db_manager:
            try:
                for status in MessageStatus:
                    row = self.db_manager.execute(
                        "SELECT COUNT(*) FROM reliable_messages WHERE status = ?",
                        [status.value]
                    )
                    if row:
                        stats[f'{status.value}_count'] = row[0][0] if isinstance(row[0], (tuple, list)) else list(row[0].values())[0]
                
                for qtype in QueueType:
                    row = self.db_manager.execute(
                        "SELECT COUNT(*) FROM reliable_messages WHERE queue_type = ?",
                        [qtype.value]
                    )
                    if row:
                        stats[f'{qtype.value}_queue_count'] = row[0][0] if isinstance(row[0], (tuple, list)) else list(row[0].values())[0]
            except Exception as e:
                LOGGER.debug("获取队列统计失败: %s", e)
        else:
            with self._lock:
                stats['pending_count'] = self._memory_main_queue.qsize()
                stats['processing_count'] = len(self._inflight_messages)
                stats['retrying_count'] = len(self._memory_retry_queue)
                stats['dead_count'] = len(self._memory_dlq)
                stats['acked_count'] = len(self._processed_ids)
        
        return stats

    # ------------------------------------------------------------------
    # 存储层实现
    # ------------------------------------------------------------------

    def _ensure_storage(self):
        """确保存储结构存在。"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        if self.db_manager:
            try:
                self.db_manager.execute("""
                    CREATE TABLE IF NOT EXISTS reliable_messages (
                        message_id TEXT PRIMARY KEY,
                        correlation_id TEXT,
                        message_type TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        headers TEXT,
                        status TEXT NOT NULL,
                        queue_type TEXT NOT NULL,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        next_retry_time REAL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        processed_at REAL,
                        expires_at REAL,
                        last_error TEXT,
                        processing_node TEXT,
                        dedup_key TEXT
                    )
                """)
                self.db_manager.execute("""
                    CREATE INDEX IF NOT EXISTS idx_msg_status 
                    ON reliable_messages(status, queue_type, next_retry_time)
                """)
                self.db_manager.execute("""
                    CREATE INDEX IF NOT EXISTS idx_msg_type 
                    ON reliable_messages(message_type)
                """)
                self.db_manager.execute("""
                    CREATE INDEX IF NOT EXISTS idx_msg_dedup 
                    ON reliable_messages(dedup_key, status)
                """)
            except Exception as e:
                LOGGER.warning("创建消息队列表失败: %s", e)
                self.db_manager = None  # 回退到文件存储

    def _save_message(self, message: ReliableMessage):
        """保存消息（使用INSERT OR REPLACE确保状态更新正确持久化）。"""
        message.updated_at = time.time()
        
        if self.db_manager:
            try:
                self.db_manager.execute("""
                    INSERT OR REPLACE INTO reliable_messages
                    (message_id, correlation_id, message_type, payload, headers,
                     status, queue_type, retry_count, max_retries, next_retry_time,
                     created_at, updated_at, processed_at, expires_at, last_error,
                     processing_node, dedup_key)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    message.message_id,
                    message.correlation_id,
                    message.message_type,
                    json.dumps(message.payload, ensure_ascii=False, default=str),
                    json.dumps(message.headers, ensure_ascii=False),
                    message.status.value,
                    message.queue_type.value,
                    message.retry_count,
                    message.max_retries,
                    message.next_retry_time,
                    message.created_at,
                    message.updated_at,
                    message.processed_at,
                    message.expires_at,
                    message.last_error,
                    message.processing_node,
                    message.dedup_key,
                ])
                return
            except Exception as e:
                LOGGER.debug("数据库保存消息失败，使用文件: %s", e)
        
        # 文件存储回退
        self._save_to_file(message)

    def _update_message(self, message: ReliableMessage):
        """更新消息。"""
        message.updated_at = time.time()
        
        if self.db_manager:
            try:
                self.db_manager.execute("""
                    UPDATE reliable_messages SET
                        status = ?, queue_type = ?, retry_count = ?,
                        next_retry_time = ?, updated_at = ?, processed_at = ?,
                        last_error = ?, processing_node = ?
                    WHERE message_id = ?
                """, [
                    message.status.value,
                    message.queue_type.value,
                    message.retry_count,
                    message.next_retry_time,
                    message.updated_at,
                    message.processed_at,
                    message.last_error,
                    message.processing_node,
                    message.message_id,
                ])
                return
            except Exception as e:
                LOGGER.debug("数据库更新消息失败，使用文件: %s", e)
        
        # 内存模式：消息队列移动已在专门方法中处理，此处仅持久化到文件
        self._save_to_file(message)

    def _fetch_pending_messages(self, limit: int = 10) -> List[ReliableMessage]:
        """原子性地获取待处理消息并立即标记为PROCESSING状态。
        
        防止多线程并发获取到同一批消息导致重复处理。
        """
        messages = []
        
        if self.db_manager:
            try:
                now = time.time()
                # 数据库模式：使用BEGIN IMMEDIATE事务原子获取并更新
                try:
                    self.db_manager.execute("BEGIN IMMEDIATE")
                except Exception as e:
                    LOGGER.debug("BEGIN IMMEDIATE失败（部分数据库不支持）: %s", e)  # 部分数据库不支持显式事务
                
                rows = self.db_manager.execute("""
                    SELECT * FROM reliable_messages
                    WHERE status = ? AND queue_type = ?
                      AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY created_at ASC LIMIT ?
                """, [MessageStatus.PENDING.value, QueueType.MAIN.value, now, limit])
                
                for row in rows:
                    if isinstance(row, (tuple, list)):
                        msg = self._row_to_message(row)
                    else:
                        msg = self._dict_to_message(dict(row))
                    # 立即标记为处理中
                    msg.status = MessageStatus.PROCESSING
                    msg.processing_node = self.node_id
                    msg.updated_at = time.time()
                    messages.append(msg)
                    # 更新数据库状态
                    self.db_manager.execute("""
                        UPDATE reliable_messages 
                        SET status = ?, processing_node = ?, updated_at = ?
                        WHERE message_id = ?
                    """, [MessageStatus.PROCESSING.value, self.node_id, msg.updated_at, msg.message_id])
                
                try:
                    self.db_manager.execute("COMMIT")
                except Exception as e:
                    LOGGER.debug("COMMIT失败: %s", e)
                    
                return messages
            except Exception as e:
                LOGGER.debug("从数据库获取待处理消息失败: %s", e)
                try:
                    self.db_manager.execute("ROLLBACK")
                except Exception as e2:
                    LOGGER.debug("ROLLBACK失败: %s", e2)
        
        # 内存模式：使用queue.Queue（原子get）+ inflight字典
        now = time.time()
        while len(messages) < limit:
            try:
                m = self._memory_main_queue.get_nowait()
            except queue.Empty:
                break
            
            # 检查过期
            if m.expires_at and now > m.expires_at:
                m.status = MessageStatus.EXPIRED
                self._save_to_file(m)
                continue
            
            # 原子标记为PROCESSING并加入inflight
            m.status = MessageStatus.PROCESSING
            m.processing_node = self.node_id
            m.updated_at = time.time()
            with self._lock:
                self._inflight_messages[m.message_id] = m
            messages.append(m)
        
        return messages

    def _fetch_retry_messages(self) -> List[ReliableMessage]:
        """获取重试队列中的消息。"""
        if self.db_manager:
            try:
                now = time.time()
                rows = self.db_manager.execute("""
                    SELECT * FROM reliable_messages
                    WHERE status = ? AND queue_type = ?
                      AND next_retry_time IS NOT NULL AND next_retry_time <= ?
                    ORDER BY next_retry_time ASC LIMIT 100
                """, [MessageStatus.RETRYING.value, QueueType.RETRY.value, now])
                
                messages = []
                for row in rows:
                    if isinstance(row, (tuple, list)):
                        msg = self._row_to_message(row)
                    else:
                        msg = self._dict_to_message(dict(row))
                    messages.append(msg)
                return messages
            except Exception as e:
                LOGGER.debug("从数据库获取重试消息失败: %s", e)
        
        # 内存队列
        with self._lock:
            now = time.time()
            return [m for m in self._memory_retry_queue 
                   if m.next_retry_time and m.next_retry_time <= now]

    def _fetch_dlq_messages(self, limit: int) -> List[ReliableMessage]:
        """获取死信队列消息。"""
        if self.db_manager:
            try:
                rows = self.db_manager.execute("""
                    SELECT * FROM reliable_messages
                    WHERE queue_type = ? ORDER BY updated_at DESC LIMIT ?
                """, [QueueType.DEAD_LETTER.value, limit])
                
                messages = []
                for row in rows:
                    if isinstance(row, (tuple, list)):
                        msg = self._row_to_message(row)
                    else:
                        msg = self._dict_to_message(dict(row))
                    messages.append(msg)
                return messages
            except Exception as e:
                LOGGER.debug("从数据库获取死信消息失败: %s", e)
        return []

    def _get_message(self, message_id: str) -> Optional[ReliableMessage]:
        """根据ID获取消息。"""
        if self.db_manager:
            try:
                rows = self.db_manager.execute("""
                    SELECT * FROM reliable_messages WHERE message_id = ?
                """, [message_id])
                if rows:
                    row = rows[0]
                    if isinstance(row, (tuple, list)):
                        return self._row_to_message(row)
                    else:
                        return self._dict_to_message(dict(row))
            except Exception as e:
                LOGGER.debug("从数据库获取消息失败: %s", e)
        
        # 从内存查找
        with self._lock:
            # 先查inflight
            if message_id in self._inflight_messages:
                return self._inflight_messages[message_id]
            # 查重试队列和DLQ
            for m in self._memory_retry_queue:
                if m.message_id == message_id:
                    return m
            for m in self._memory_dlq:
                if m.message_id == message_id:
                    return m
            # 查主队列快照
            try:
                for m in list(self._memory_main_queue.queue):
                    if m.message_id == message_id:
                        return m
            except Exception as e:
                LOGGER.debug("查询内存主队列消息失败: %s", e)
        return None

    def _row_to_message(self, row: tuple) -> ReliableMessage:
        """将数据库行（元组）转换为消息对象。"""
        # 按SELECT顺序解析
        columns = ['message_id', 'correlation_id', 'message_type', 'payload', 'headers',
                   'status', 'queue_type', 'retry_count', 'max_retries', 'next_retry_time',
                   'created_at', 'updated_at', 'processed_at', 'expires_at', 'last_error',
                   'processing_node', 'dedup_key']
        data = {}
        for i, col in enumerate(columns):
            if i < len(row):
                data[col] = row[i]
        
        # JSON反序列化
        if data.get('payload'):
            try:
                data['payload'] = json.loads(data['payload'])
            except Exception as e:
                LOGGER.warning("消息payload JSON反序列化失败: %s", e)
        if data.get('headers'):
            try:
                data['headers'] = json.loads(data['headers'])
            except Exception as e:
                LOGGER.debug("消息headers JSON反序列化失败: %s", e)
        
        return ReliableMessage.from_dict(data)

    def _dict_to_message(self, data: Dict[str, Any]) -> ReliableMessage:
        """将字典（数据库行dict）转换为消息对象。"""
        if isinstance(data.get('payload'), str):
            try:
                data['payload'] = json.loads(data['payload'])
            except Exception as e:
                LOGGER.warning("消息payload JSON反序列化失败: %s", e)
        if isinstance(data.get('headers'), str):
            try:
                data['headers'] = json.loads(data['headers'])
            except Exception as e:
                LOGGER.debug("消息headers JSON反序列化失败: %s", e)
        return ReliableMessage.from_dict(data)

    def _save_to_file(self, message: ReliableMessage):
        """保存消息到文件（回退方案）。"""
        try:
            filename = os.path.join(self.data_dir, f"{message.message_id}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(message.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            LOGGER.debug("文件保存消息失败: %s", e)

    def _flush_memory_to_disk(self):
        """将内存中所有消息刷入磁盘（关闭时调用）。"""
        all_messages = []
        with self._lock:
            # inflight消息
            all_messages.extend(self._inflight_messages.values())
            # 重试队列
            all_messages.extend(self._memory_retry_queue)
            # DLQ
            all_messages.extend(self._memory_dlq)
        
        # 主队列
        try:
            while True:
                m = self._memory_main_queue.get_nowait()
                all_messages.append(m)
        except queue.Empty:
            pass
        
        for m in all_messages:
            try:
                self._save_to_file(m)
            except Exception as e:
                LOGGER.debug("持久化消息到文件失败: %s", e)
        
        LOGGER.info("内存队列已持久化到磁盘，共 %d 条消息", len(all_messages))


class MessageHandlerError(Exception):
    """消息处理器异常"""
    pass
