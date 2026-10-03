#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FHIR Subscription 订阅通知处理器（Webhook模式）

医院FHIR服务器在患者有新的检验结果/诊断/影像报告时，主动推送通知到tb_risk的
HTTP Webhook端点，自动触发该患者的风险重算。

注意：这是一个轻量实现。生产部署建议使用独立Web服务器（Flask/FastAPI）或反向代理，
这里提供核心处理逻辑供集成。
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional

from ..base import timestamp_to_datetime

LOGGER = logging.getLogger("tb_risk.health_interop.fhir.subscription")


class FHIRSubscriptionHandler:
    """FHIR Subscription Webhook 处理器

    工作流程：
    1. 启动HTTP服务监听Webhook回调
    2. 收到FHIR Subscription通知后，解析患者ID
    3. 调用回调函数（通常触发 fetch_patient + 风险重算）
    4. 支持订阅/取消订阅话题
    """

    def __init__(self, fhir_client,
                 on_patient_update: Optional[Callable[[str, str], None]] = None,
                 host: str = "127.0.0.1", port: int = 8890):
        """
        参数：
            fhir_client: FHIRClient 实例，用于接收通知后拉取完整数据
            on_patient_update: 回调函数 (patient_id: str, event_type: str) → None
            host: Webhook监听地址
            port: Webhook监听端口
        """
        self._client = fhir_client
        self._on_update = on_patient_update
        self._host = host
        self._port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._notification_log: List[Dict[str, Any]] = []

    def start(self):
        """启动Webhook监听服务（后台线程）"""
        if self._server:
            return

        handler_cls = self._make_handler()
        self._server = HTTPServer((self._host, self._port), handler_cls)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
            name="FHIR-Subscription-Webhook")
        self._thread.start()
        LOGGER.info("FHIR Subscription Webhook 已启动: http://%s:%d",
                    self._host, self._port)

    def stop(self):
        """停止Webhook服务"""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        LOGGER.info("FHIR Subscription Webhook 已停止")

    def subscribe(self, topic: str, criteria: str, reason: str = "") -> bool:
        """向FHIR服务器创建订阅

        参数：
            topic: 话题标识（例如 "new-lab-result"）
            criteria: FHIR搜索表达式（例如 "Observation?code=http://loinc.org|640-4"）
            reason: 订阅原因说明
        """
        if not self._client:
            LOGGER.error("FHIRClient未配置，无法创建订阅")
            return False
        try:
            sub_resource = {
                "resourceType": "Subscription",
                "status": "requested",
                "reason": reason or f"tb_risk订阅：{topic}",
                "criteria": criteria,
                "channel": {
                    "type": "rest-hook",
                    "endpoint": f"http://{self._host}:{self._port}/fhir/webhook",
                    "payload": "application/fhir+json",
                },
            }
            result = self._client._request("POST", "Subscription", data=sub_resource)
            sub_id = result.get("id", "")
            self._subscriptions[topic] = {
                "id": sub_id,
                "criteria": criteria,
                "status": result.get("status", "requested"),
            }
            LOGGER.info("FHIR订阅创建成功: topic=%s id=%s criteria=%s",
                        topic, sub_id, criteria)
            return True
        except Exception as e:
            LOGGER.error("FHIR订阅创建失败: topic=%s error=%s", topic, e)
            return False

    def unsubscribe(self, topic: str) -> bool:
        """取消订阅"""
        sub = self._subscriptions.get(topic)
        if not sub or not sub.get("id"):
            return False
        try:
            self._client._request("DELETE", f"Subscription/{sub['id']}")
            del self._subscriptions[topic]
            LOGGER.info("FHIR订阅已取消: topic=%s", topic)
            return True
        except Exception as e:
            LOGGER.error("FHIR订阅取消失败: topic=%s error=%s", topic, e)
            return False

    def _handle_notification(self, payload: Dict[str, Any]):
        """处理收到的Webhook通知"""
        try:
            # FHIR Subscription通知通常包含SubscriptionStatus资源或直接是触发资源
            resource_type = payload.get("resourceType", "")
            patient_id = ""
            event_type = "update"

            if resource_type == "SubscriptionStatus":
                # R4B/R5 SubscriptionStatus
                events = payload.get("notificationEvent", []) or []
                for ev in events:
                    ref = ev.get("focus", {}).get("reference", "")
                    event_type = ev.get("eventType", "update") or "update"
                    if ref.startswith("Patient/"):
                        patient_id = ref.split("/", 1)[1]
            elif resource_type in ("Observation", "Condition", "DiagnosticReport",
                                   "MedicationStatement", "Encounter"):
                # 直接推送触发资源
                subj = payload.get("subject", {}) or {}
                ref = subj.get("reference", "")
                if ref.startswith("Patient/"):
                    patient_id = ref.split("/", 1)[1]
                event_type = f"new-{resource_type.lower()}"

            self._notification_log.append({
                "patient_id": patient_id,
                "event_type": event_type,
                "timestamp": timestamp_to_datetime("now").isoformat(),
            })

            if patient_id and self._on_update:
                LOGGER.info("FHIR通知触发：患者%s 事件%s", patient_id, event_type)
                self._on_update(patient_id, event_type)
        except Exception as e:
            LOGGER.error("处理FHIR通知失败: %s", e, exc_info=True)

    def _make_handler(self):
        """创建HTTP请求处理类（闭包访问self）"""
        parent = self

        class _WebhookHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/fhir/webhook":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(body) if body else {}
                    parent._handle_notification(payload)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"OK")
                except Exception as e:
                    LOGGER.error("Webhook处理异常: %s", e)
                    self.send_error(500, str(e))

            def log_message(self, fmt, *args):
                LOGGER.debug("Webhook: %s", fmt % args)

        return _WebhookHandler

    @property
    def subscriptions(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._subscriptions)

    @property
    def notification_log(self) -> List[Dict[str, Any]]:
        return list(self._notification_log)
