#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MLLP (Minimal Lower Layer Protocol) HL7消息接收服务

作为常驻后台服务运行，监听医院集成平台通过TCP推送的HL7 v2.x消息。
特性：
- 多连接支持（threading）
- MLLP帧解析（\x0b开始，\x1c\x0d结束）
- 三级ACK：AA（接受）/ AR（拒绝）/ AE（错误）
- 消息先持久化到队列再返回ACK，确保不丢消息
- 后台Worker线程消费队列，异步处理消息
- 优雅关闭
"""

from __future__ import annotations

import logging
import socket
import socketserver
import ssl
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from ..base import HealthcareAdapter, AdapterConfig, AdapterResult
from .parser import HL7MessageParser, MLLP_START_BLOCK, MLLP_END_BLOCK
from .ack import HL7AckBuilder
from .queue import HL7MessageQueue, QueuedMessage

LOGGER = logging.getLogger("tb_risk.health_interop.hl7v2.server")


class _MLLPHandler(socketserver.BaseRequestHandler):
    """MLLP TCP连接处理器"""

    def handle(self):
        server: 'MLLPServer' = self._tb_server
        conn: socket.socket = self.request
        conn.settimeout(30)
        buffer = b""
        try:
            while True:
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    break
                if not data:
                    break
                buffer += data
                # 尝试解析完整MLLP帧
                while MLLP_START_BLOCK in buffer and MLLP_END_BLOCK in buffer:
                    start = buffer.find(MLLP_START_BLOCK)
                    end = buffer.find(MLLP_END_BLOCK, start)
                    if end < 0:
                        break
                    frame = buffer[start + 1:end]
                    buffer = buffer[end + len(MLLP_END_BLOCK):]
                    try:
                        raw_msg = frame.decode("utf-8", errors="replace")
                        ack_str = server._process_incoming(raw_msg)
                    except Exception as e:
                        LOGGER.error("消息处理异常: %s", e, exc_info=True)
                        ack_str = HL7AckBuilder.build(
                            raw_msg if 'raw_msg' in dir() else "",
                            HL7AckBuilder.AE,
                            f"处理错误: {str(e)[:100]}"
                        )
                    # 构造并发送ACK（包装MLLP帧）
                    try:
                        ack_frame = MLLP_START_BLOCK + ack_str.encode("utf-8") + MLLP_END_BLOCK
                        conn.sendall(ack_frame)
                    except Exception as e:
                        LOGGER.error("ACK发送失败: %s", e)
        except Exception as e:
            LOGGER.debug("MLLP连接异常: %s", e)


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """多线程TCP服务器，支持多连接"""
    allow_reuse_address = True
    daemon_threads = True


class MLLPServer(HealthcareAdapter):
    """HL7 v2.x MLLP消息接收服务适配器

    使用方法：
        server = MLLPServer(config)
        server.set_message_handler(lambda msg: ...)
        server.start()  # 后台启动
        # ...
        server.stop()
    """

    ADAPTER_TYPE = "hl7v2"

    def __init__(self, config: Optional[AdapterConfig] = None,
                 queue_db_path: str = "hl7_messages.db",
                 ssl_certfile: str = "",
                 ssl_keyfile: str = "",
                 ssl_ca_certs: str = "",
                 require_client_cert: bool = False):
        super().__init__(config or AdapterConfig())
        self._host = "0.0.0.0"
        self._port = self.config.port or 2575  # 默认MLLP端口（TLS使用2576）
        if self.config.base_url:
            # 从base_url解析host:port
            import urllib.parse
            parts = urllib.parse.urlparse(self.config.base_url)
            if parts.hostname:
                self._host = parts.hostname
            if parts.port:
                self._port = parts.port
        
        # TLS/SSL配置（MLLPS支持）
        self._ssl_certfile = ssl_certfile
        self._ssl_keyfile = ssl_keyfile
        self._ssl_ca_certs = ssl_ca_certs
        self._require_client_cert = require_client_cert
        self._tls_enabled = bool(ssl_certfile and ssl_keyfile)
        
        # 如果显式提供了证书但未指定端口，使用标准MLLPS端口2576
        if self._tls_enabled and not self.config.port and not self.config.base_url:
            self._port = 2576
        self._parser = HL7MessageParser()
        self._queue = HL7MessageQueue(queue_db_path)
        self._server: Optional[_ThreadedTCPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._message_handlers: List[Callable[[Any, AdapterResult], None]] = []
        self._supported_types = {
            "ADT^A01", "ADT^A03", "ADT^A04",  # 患者入出院/注册
            "ORM^O01",                          # 检验申请
            "ORU^R01",                          # 检验结果
            "MDM^T02",                          # 病历文档
        }
        self._stop_event = threading.Event()
        self._ssl_context: Optional[ssl.SSLContext] = None
        
        # 初始化SSL上下文（如果配置了证书）
        if self._tls_enabled:
            self._init_ssl_context()
    
    def _init_ssl_context(self):
        """初始化SSL/TLS上下文"""
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=self._ssl_certfile, keyfile=self._ssl_keyfile)
            
            if self._ssl_ca_certs:
                context.load_verify_locations(cafile=self._ssl_ca_certs)
                if self._require_client_cert:
                    context.verify_mode = ssl.CERT_REQUIRED
                else:
                    context.verify_mode = ssl.CERT_OPTIONAL
            
            # 安全配置：禁用不安全的协议版本
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
            
            self._ssl_context = context
            LOGGER.info("MLLP TLS已启用，证书: %s", self._ssl_certfile)
        except Exception as e:
            LOGGER.error("SSL上下文初始化失败: %s", e, exc_info=True)
            self._tls_enabled = False
            self._ssl_context = None

    def set_message_handler(self, handler: Callable[[Any, AdapterResult], None]):
        """设置消息处理回调 (parsed_hl7_message, adapter_result) → None"""
        self._message_handlers.append(handler)

    # ---- HealthcareAdapter 接口 ----

    def connect(self) -> bool:
        """启动MLLP监听服务和Worker线程"""
        return self.start()

    def fetch_patient(self, patient_id: str) -> AdapterResult:
        """HL7 v2是推送模式，不支持主动查询单个患者。
        通过HL7主动查询需使用QBP^Q22等查询消息，此处返回空结果。"""
        result = AdapterResult(source_system="HL7v2", patient_id=patient_id)
        result.warnings.append("HL7 v2 MLLP服务为推送模式，不支持按需查询单个患者")
        return result

    def fetch_incremental(self, since=None) -> List[AdapterResult]:
        """从队列中获取待重试/未处理的消息处理结果"""
        return []

    def disconnect(self):
        self.stop()

    def health_check(self) -> tuple:
        if self._running and self._server:
            proto = "MLLPS(TLS)" if self._tls_enabled else "MLLP"
            return True, f"{proto}服务运行中 port={self._port}"
        return False, "MLLP服务未运行"

    # ---- 服务启停 ----

    def start(self) -> bool:
        """启动MLLP服务（TCP监听+Worker消费）"""
        if self._running:
            return True
        try:
            self._server = _ThreadedTCPServer((self._host, self._port), _MLLPHandler)
            # 将自身绑定到server，handler可访问
            self._server._tb_handler = self  # type: ignore
            
            # 如果启用TLS，包装socket
            if self._tls_enabled and self._ssl_context:
                raw_socket = self._server.socket
                self._server.socket = self._ssl_context.wrap_socket(
                    raw_socket, server_side=True
                )
                LOGGER.info("MLLP socket已包装为TLS")
            
            # 猴子补丁：让handler能访问server实例
            _MLLPHandler_ref = _MLLPHandler

            # 包装handler，注入self
            class _BoundHandler(_MLLPHandler_ref):
                def __init__(h_self, *args, **kwargs):
                    h_self._tb_server = self
                    super().__init__(*args, **kwargs)

            self._server.RequestHandlerClass = _BoundHandler

            self._server_thread = threading.Thread(
                target=self._server.serve_forever, daemon=True,
                name="HL7-MLLP-Server")
            self._server_thread.start()

            self._running = True
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True,
                name="HL7-Worker")
            self._worker_thread.start()

            proto = "MLLPS(TLS)" if self._tls_enabled else "MLLP"
            LOGGER.info("HL7 %s服务已启动: %s:%d", proto, self._host, self._port)
            return True
        except Exception as e:
            LOGGER.error("MLLP服务启动失败: %s", e, exc_info=True)
            self._running = False
            return False

    def stop(self):
        """优雅关闭"""
        if not self._running:
            return
        LOGGER.info("正在停止HL7 MLLP服务...")
        self._running = False
        self._stop_event.set()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._server_thread:
            self._server_thread.join(timeout=5)
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        self._server = None
        self._server_thread = None
        self._worker_thread = None
        LOGGER.info("HL7 MLLP服务已停止")

    # ---- 消息处理 ----

    def _process_incoming(self, raw_msg: str) -> str:
        """处理收到的原始HL7消息：解析→持久化→返回ACK字符串
        
        ACK类型：
        - AA (Application Accept): 消息成功接收并持久化
        - AR (Application Reject): 消息被拒绝（格式错误、不支持的类型、队列满/持久化失败），
                                   发送方应修复问题后重试
        - AE (Application Error): 系统内部错误（解析器异常等），发送方可稍后重试
        """
        try:
            parsed = self._parser.parse(raw_msg)
            if parsed.parse_errors and not parsed.message_type:
                # 消息格式问题导致无法解析 → AR拒绝
                return HL7AckBuilder.reject(parsed, "消息解析失败: " + "; ".join(parsed.parse_errors[:3]))
        except Exception as e:
            # 解析器崩溃属于系统内部错误 → AE
            LOGGER.error("HL7消息解析内部错误: %s", e, exc_info=True)
            return HL7AckBuilder.build(raw_msg, HL7AckBuilder.AE, f"内部错误: {str(e)[:80]}")

        # 消息类型过滤
        msg_type = parsed.message_type
        if msg_type and msg_type not in self._supported_types:
            LOGGER.info("不支持的消息类型: %s，返回AR拒绝", msg_type)
            return HL7AckBuilder.reject(parsed, f"不支持的消息类型: {msg_type}")

        # 持久化到队列（必须在ACK前）
        try:
            self._queue.enqueue(raw_msg, parsed)
        except Exception as e:
            # 队列满/持久化失败 → AR拒绝，让发送方重试
            LOGGER.error("消息持久化失败: %s", e, exc_info=True)
            return HL7AckBuilder.build(raw_msg, HL7AckBuilder.AR, f"队列不可用，持久化失败: {str(e)[:80]}")

        return HL7AckBuilder.accept(parsed)

    def _worker_loop(self):
        """Worker线程：消费队列消息，异步处理"""
        LOGGER.info("HL7 Worker线程启动")
        while not self._stop_event.is_set():
            try:
                pending = self._queue.get_pending(limit=5)
                if not pending:
                    self._stop_event.wait(2.0)
                    continue
                for qmsg in pending:
                    if self._stop_event.is_set():
                        break
                    self._process_queued_message(qmsg)
            except Exception as e:
                LOGGER.error("Worker循环异常: %s", e, exc_info=True)
                self._stop_event.wait(5.0)
        LOGGER.info("HL7 Worker线程退出")

    def _process_queued_message(self, qmsg: QueuedMessage):
        """处理单条队列消息"""
        msg_id = qmsg.id
        self._queue.mark_processing(msg_id)
        try:
            parsed = self._parser.parse(qmsg.raw_message)
            result = self._convert_to_result(parsed)

            # 调用所有注册的handler
            for handler in self._message_handlers:
                try:
                    handler(parsed, result)
                except Exception as e:
                    LOGGER.error("消息handler执行失败: %s", e, exc_info=True)

            self._queue.mark_done(msg_id)
            LOGGER.debug("消息 %d 处理成功: type=%s control=%s",
                        msg_id, qmsg.message_type, qmsg.control_id)
        except Exception as e:
            LOGGER.error("消息 %d 处理失败: %s", msg_id, e, exc_info=True)
            self._queue.mark_failed(msg_id, str(e))

    def _convert_to_result(self, parsed) -> AdapterResult:
        """将解析后的HL7Message转换为AdapterResult"""
        result = AdapterResult(
            source_system="HL7v2",
            patient_id=parsed.patient_info.get("patient_id",
                        parsed.patient_info.get("medical_record_no", "")),
            last_sync_time=parsed.message_time,
            records_processed=len(parsed.lab_results) + len(parsed.clinical_notes),
        )
        result.patient_info = dict(parsed.patient_info)
        result.lab_results = list(parsed.lab_results)
        result.imaging_reports = [
            {"conclusion": n["content"], "title": n.get("title", ""),
             "effective_time": n.get("effective_time"), "nlp_pending": True}
            for n in parsed.clinical_notes if n.get("note_type") == "radiology"
        ]
        result.success = True
        if parsed.parse_errors:
            result.warnings.extend(parsed.parse_errors)
        return result

    @property
    def queue(self) -> HL7MessageQueue:
        return self._queue
