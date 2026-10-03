#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增量数据更新与事件驱动触发引擎

实现患者数据的基线建立、增量更新、事件驱动风险触发：

1. 基线建立：患者首次就诊时自动拉取历史数据建立基线
2. 增量更新：复诊/新检验/新诊断时自动增量更新特征
3. 风险重算：每次更新后自动重新计算风险评分
4. 事件驱动：LIS阳性/放射科疑似结核时自动触发风险评估并提醒医生
"""

import datetime
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger("tb_risk.health_interop.incremental")

# ============================================================================
# 事件类型定义
# ============================================================================


class EventType(Enum):
    """增量事件类型"""
    # 患者就诊事件
    PATIENT_FIRST_VISIT = "patient_first_visit"       # 首次就诊
    PATIENT_FOLLOW_UP = "patient_follow_up"           # 复诊
    PATIENT_DISCHARGE = "patient_discharge"           # 出院

    # 检验结果事件
    LAB_RESULT_NEW = "lab_result_new"                 # 新检验结果
    LAB_RESULT_CRITICAL = "lab_result_critical"       # 危急值
    SPUTUM_SMEAR_POSITIVE = "sputum_smear_positive"   # 痰涂片阳性
    XPERT_MTB_POSITIVE = "xpert_mtb_positive"         # Xpert MTB阳性
    XPERT_RIF_RESISTANT = "xpert_rif_resistant"       # 利福平耐药

    # 影像学事件
    IMAGING_SUSPECTED_TB = "imaging_suspected_tb"     # 影像疑似结核
    IMAGING_CAVITY_FOUND = "imaging_cavity_found"     # 发现空洞
    IMAGING_WORSENING = "imaging_worsening"           # 影像进展

    # 诊断事件
    DIAGNOSIS_CONFIRMED = "diagnosis_confirmed"        # 确诊
    DIAGNOSIS_CHANGED = "diagnosis_changed"           # 诊断变更
    COMORBIDITY_FOUND = "comorbidity_found"           # 发现合并症

    # 用药事件
    MEDICATION_STARTED = "medication_started"         # 开始用药
    MEDICATION_CHANGED = "medication_changed"         # 用药变更
    ADVERSE_REACTION = "adverse_reaction"             # 不良反应

    # 系统事件
    DATA_SYNC_COMPLETE = "data_sync_complete"         # 数据同步完成
    SCHEDULED_REASSESS = "scheduled_reassess"         # 定时重评估


# 事件优先级（用于决定处理顺序）
EVENT_PRIORITY = {
    EventType.SPUTUM_SMEAR_POSITIVE: 100,
    EventType.XPERT_MTB_POSITIVE: 100,
    EventType.XPERT_RIF_RESISTANT: 100,
    EventType.IMAGING_SUSPECTED_TB: 90,
    EventType.LAB_RESULT_CRITICAL: 90,
    EventType.DIAGNOSIS_CONFIRMED: 80,
    EventType.IMAGING_CAVITY_FOUND: 80,
    EventType.ADVERSE_REACTION: 80,
    EventType.IMAGING_WORSENING: 70,
    EventType.DIAGNOSIS_CHANGED: 60,
    EventType.COMORBIDITY_FOUND: 60,
    EventType.MEDICATION_CHANGED: 50,
    EventType.MEDICATION_STARTED: 50,
    EventType.PATIENT_FIRST_VISIT: 40,
    EventType.LAB_RESULT_NEW: 30,
    EventType.PATIENT_FOLLOW_UP: 20,
    EventType.DATA_SYNC_COMPLETE: 10,
    EventType.SCHEDULED_REASSESS: 5,
    EventType.PATIENT_DISCHARGE: 1,
}

# 高风险事件（需要立即通知医生）
HIGH_RISK_EVENTS = {
    EventType.SPUTUM_SMEAR_POSITIVE,
    EventType.XPERT_MTB_POSITIVE,
    EventType.XPERT_RIF_RESISTANT,
    EventType.IMAGING_SUSPECTED_TB,
    EventType.LAB_RESULT_CRITICAL,
    EventType.ADVERSE_REACTION,
}


# ============================================================================
# 数据类定义
# ============================================================================


@dataclass
class PatientEvent:
    """患者事件"""
    event_id: str
    patient_id: str
    event_type: EventType
    timestamp: datetime.datetime
    source: str                     # 事件来源（LIS/PACS/HIS/EMR等）
    data: Dict[str, Any] = field(default_factory=dict)  # 事件相关数据
    priority: int = 0
    processed: bool = False
    error: Optional[str] = None

    def __post_init__(self):
        if self.priority == 0:
            self.priority = EVENT_PRIORITY.get(self.event_type, 10)


@dataclass
class PatientSnapshot:
    """患者数据快照（基线/增量状态）"""
    patient_id: str
    baseline_established: bool = False
    baseline_time: Optional[datetime.datetime] = None
    last_update_time: Optional[datetime.datetime] = None
    last_risk_score: Optional[float] = None
    last_risk_level: str = "unknown"  # low/medium/high/critical
    update_count: int = 0
    features: Dict[str, Any] = field(default_factory=dict)
    source_data: Dict[str, Any] = field(default_factory=dict)
    event_history: List[Dict[str, Any]] = field(default_factory=list)
    pending_notifications: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Notification:
    """医生通知"""
    notification_id: str
    patient_id: str
    patient_name: str = ""
    event_type: EventType = EventType.DATA_SYNC_COMPLETE
    title: str = ""
    message: str = ""
    severity: str = "info"  # info/warning/critical
    risk_score: Optional[float] = None
    risk_level: str = ""
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    acknowledged: bool = False
    acknowledged_by: str = ""
    acknowledged_at: Optional[datetime.datetime] = None


# ============================================================================
# 事件处理器类型
# ============================================================================

EventHandler = Callable[[PatientEvent], Optional[Dict[str, Any]]]


# ============================================================================
# 增量更新引擎
# ============================================================================


class IncrementalEngine:
    """增量数据更新与事件驱动触发引擎

    核心功能：
    1. 管理患者数据快照（基线+增量）
    2. 事件注册与分发
    3. 增量更新特征
    4. 风险评分重算
    5. 通知医生
    """

    def __init__(self):
        self._snapshots: Dict[str, PatientSnapshot] = {}  # patient_id -> snapshot
        self._event_handlers: Dict[EventType, List[EventHandler]] = {}
        self._risk_calculator: Optional[Callable] = None
        self._notification_callback: Optional[Callable] = None
        self._data_fetcher: Optional[Callable] = None
        self._lock = threading.Lock()
        self._snapshots_lock = threading.RLock()
        self._event_queue: List[PatientEvent] = []
        self._processing = False
        self._processor_thread: Optional[threading.Thread] = None
        self._patient_cache: Dict[str, Dict[str, Any]] = {}

    # ==================== 配置 ====================

    def set_risk_calculator(self, calculator: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """设置风险评分计算函数

        参数：
            calculator: 输入特征字典，返回 {score, level, ...}
        """
        self._risk_calculator = calculator

    def set_notification_callback(self, callback: Callable[[Notification], None]):
        """设置通知回调函数

        参数：
            callback: 接收 Notification 对象并执行通知
        """
        self._notification_callback = callback

    def set_data_fetcher(self, fetcher: Callable[[str, Dict[str, Any]], Dict[str, Any]]):
        """设置数据拉取函数

        参数：
            fetcher: 输入(patient_id, options)，返回数据字典
        """
        self._data_fetcher = fetcher

    # ==================== 事件注册 ====================

    def register_handler(self, event_type: EventType, handler: EventHandler):
        """注册事件处理器

        参数：
            event_type: 事件类型
            handler: 处理函数，接收 PatientEvent，返回更新后的特征片段
        """
        self._event_handlers.setdefault(event_type, []).append(handler)
        LOGGER.info("注册事件处理器: %s -> %s", event_type.value, handler.__name__)

    def unregister_handler(self, event_type: EventType, handler: EventHandler):
        """注销事件处理器"""
        handlers = self._event_handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    # ==================== 事件处理 ====================

    def emit_event(self, patient_id: str, event_type: EventType,
                   data: Dict[str, Any], source: str = "") -> PatientEvent:
        """触发事件

        参数：
            patient_id: 患者ID
            event_type: 事件类型
            data: 事件相关数据
            source: 事件来源

        返回：
            PatientEvent
        """
        event = PatientEvent(
            event_id=f"{patient_id}_{event_type.value}_{int(time.time() * 1000)}",
            patient_id=patient_id,
            event_type=event_type,
            timestamp=datetime.datetime.now(),
            source=source or event_type.value,
            data=data,
        )

        with self._lock:
            self._event_queue.append(event)
            # 按优先级排序（在锁内执行，避免并发排序冲突）
            self._event_queue.sort(key=lambda e: e.priority, reverse=True)
            # 启动异步处理（在锁内执行 check-then-set，避免重复启动）
            self._ensure_processor_running()

        LOGGER.info("入队事件: [%s] %s (优先级=%d)",
                     patient_id, event_type.value, event.priority)

        return event

    def _ensure_processor_running(self):
        """确保事件处理器线程运行（调用方需持有 self._lock）"""
        if not self._processing:
            self._processing = True
            self._processor_thread = threading.Thread(
                target=self._process_events, daemon=True)
            self._processor_thread.start()

    def _process_events(self):
        """事件处理循环（异步线程）"""
        while True:
            with self._lock:
                if not self._event_queue:
                    self._processing = False
                    break
                event = self._event_queue.pop(0)

            try:
                self._handle_event(event)
            except Exception as e:
                LOGGER.error("处理事件失败: %s - %s", event.event_id, e)
                event.error = str(e)
            finally:
                event.processed = True

    def _handle_event(self, event: PatientEvent):
        """处理单个事件（线程安全）

        通知回调在锁外执行，避免死锁。
        """
        snapshot = self._get_or_create_snapshot(event.patient_id)
        pending_notifications = []

        with self._snapshots_lock:
            # 1. 执行事件处理器
            handlers = self._event_handlers.get(event.event_type, [])
            feature_updates = {}
            for handler in handlers:
                try:
                    result = handler(event)
                    if result:
                        feature_updates.update(result)
                except Exception as e:
                    LOGGER.error("事件处理器异常: %s - %s", handler.__name__, e)

            # 2. 更新快照特征
            if feature_updates:
                snapshot.features.update(feature_updates)
                snapshot.source_data[event.event_type.value] = event.data

            # 3. 记录事件历史
            snapshot.event_history.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "summary": self._summarize_event(event),
            })

            # 4. 更新时间和计数
            snapshot.last_update_time = event.timestamp
            snapshot.update_count += 1

            # 5. 如果是高风险事件，生成通知（仅构造，回调在锁外执行）
            if event.event_type in HIGH_RISK_EVENTS:
                notif = self._generate_notification(event, snapshot)
                if notif:
                    pending_notifications.append(notif)

            # 6. 重算风险评分
            self._recalculate_risk(event.patient_id, event, pending_notifications)

        # 锁外执行通知回调，避免死锁
        for notif in pending_notifications:
            if self._notification_callback:
                try:
                    self._notification_callback(notif)
                except Exception as e:
                    LOGGER.error("通知回调失败: %s", e)

    def _summarize_event(self, event: PatientEvent) -> str:
        """生成事件摘要"""
        summaries = {
            EventType.SPUTUM_SMEAR_POSITIVE: "痰涂片阳性结果",
            EventType.XPERT_MTB_POSITIVE: "Xpert MTB阳性",
            EventType.XPERT_RIF_RESISTANT: "利福平耐药",
            EventType.IMAGING_SUSPECTED_TB: "影像学疑似结核",
            EventType.IMAGING_CAVITY_FOUND: "影像发现空洞",
            EventType.DIAGNOSIS_CONFIRMED: "确诊结核病",
            EventType.PATIENT_FIRST_VISIT: "首次就诊，建立基线",
            EventType.PATIENT_FOLLOW_UP: "复诊",
            EventType.LAB_RESULT_NEW: "新检验结果",
            EventType.LAB_RESULT_CRITICAL: "危急值报告",
            EventType.MEDICATION_STARTED: "开始抗结核治疗",
            EventType.ADVERSE_REACTION: "药物不良反应",
        }
        return summaries.get(event.event_type, f"事件: {event.event_type.value}")

    # ==================== 基线建立 ====================

    def establish_baseline(self, patient_id: str,
                           patient_data: Optional[Dict[str, Any]] = None) -> PatientSnapshot:
        """建立患者基线（线程安全）

        在患者首次就诊时调用，自动拉取历史数据建立基线特征。

        参数：
            patient_id: 患者ID
            patient_data: 可选，已有患者数据（如从HIS查询）

        返回：
            PatientSnapshot
        """
        with self._snapshots_lock:
            snapshot = self._get_or_create_snapshot(patient_id)

            if snapshot.baseline_established:
                LOGGER.info("患者基线已存在: %s，跳过", patient_id)
                return snapshot

            LOGGER.info("建立患者基线: %s", patient_id)

            # 1. 拉取历史数据
            if patient_data:
                snapshot.features.update(patient_data)
            elif self._data_fetcher:
                try:
                    historical_data = self._data_fetcher(patient_id, {
                        "mode": "baseline",
                        "history_years": 5,
                    })
                    if historical_data:
                        snapshot.features.update(historical_data)
                except Exception as e:
                    LOGGER.error("拉取历史数据失败: %s - %s", patient_id, e)

            # 2. 标记基线已建立
            snapshot.baseline_established = True
            snapshot.baseline_time = datetime.datetime.now()

            # 3. 首次风险评分
            self._recalculate_risk(patient_id)

        # 4. 记录基线事件（锁外，emit_event 有自己的锁）
        self.emit_event(patient_id, EventType.PATIENT_FIRST_VISIT,
                        {"mode": "baseline", "features_count": len(snapshot.features)},
                        source="incremental_engine")

        LOGGER.info("患者基线建立完成: %s (特征数: %d)",
                     patient_id, len(snapshot.features))
        return snapshot

    def incremental_update(self, patient_id: str,
                           new_data: Dict[str, Any],
                           source: str = "") -> PatientSnapshot:
        """增量更新患者数据（线程安全）

        在复诊、新检验结果、新诊断时调用。

        参数：
            patient_id: 患者ID
            new_data: 新的数据片段
            source: 数据来源

        返回：
            PatientSnapshot
        """
        with self._snapshots_lock:
            snapshot = self._get_or_create_snapshot(patient_id)

            if not snapshot.baseline_established:
                # 基线未建立，先建立基线再增量更新
                LOGGER.info("基线未建立，先建立基线: %s", patient_id)
                if new_data:
                    snapshot.features.update(new_data)
                snapshot.baseline_established = True
                snapshot.baseline_time = datetime.datetime.now()
                snapshot.update_count += 1
                if source:
                    snapshot.source_data[source] = new_data
                self._recalculate_risk(patient_id)
            else:
                # 正常增量更新
                snapshot.features.update(new_data)
                snapshot.last_update_time = datetime.datetime.now()
                snapshot.update_count += 1
                if source:
                    snapshot.source_data[source] = new_data
                self._recalculate_risk(patient_id)

            LOGGER.info("增量更新完成: %s (来源: %s, 更新次数: %d)",
                         patient_id, source, snapshot.update_count)
            return snapshot

    # ==================== LIS/PACS事件触发 ====================

    def on_lab_result(self, patient_id: str, lab_result: Dict[str, Any]) -> Optional[PatientEvent]:
        """LIS检验结果到达时触发

        自动判断结果类型并触发相应事件。
        """
        test_code = lab_result.get("test_code", lab_result.get("loinc", ""))
        value = lab_result.get("value", lab_result.get("result", ""))
        flag = lab_result.get("abnormal_flag", "")

        # 判断事件类型
        event_type = EventType.LAB_RESULT_NEW
        event_data = {
            "test_code": test_code,
            "value": value,
            "test_name": lab_result.get("test_name", ""),
            "unit": lab_result.get("unit", ""),
            "lab_time": lab_result.get("lab_time", ""),
        }

        # 痰涂片阳性
        if "sputum_smear" in test_code.lower() or "涂片" in str(lab_result):
            if "阳性" in str(value) or "positive" in str(value).lower() or value in ("2", "positive"):
                event_type = EventType.SPUTUM_SMEAR_POSITIVE
                event_data["critical"] = True

        # Xpert MTB阳性
        elif "xpert" in test_code.lower() or "mtb" in test_code.lower():
            if "阳性" in str(value) or "positive" in str(value).lower() or "检出" in str(value):
                if "rif" in test_code.lower() or "耐药" in str(lab_result):
                    event_type = EventType.XPERT_RIF_RESISTANT
                else:
                    event_type = EventType.XPERT_MTB_POSITIVE
                event_data["critical"] = True

        # 危急值
        elif flag in ("HH", "LL", "critical", "critical_low", "critical_high"):
            event_type = EventType.LAB_RESULT_CRITICAL
            event_data["critical"] = True

        return self.emit_event(patient_id, event_type, event_data, source="LIS")

    def on_imaging_report(self, patient_id: str,
                          imaging_report: Dict[str, Any]) -> Optional[PatientEvent]:
        """PACS影像报告到达时触发

        自动分析报告内容并触发相应事件。
        """
        impression = imaging_report.get("impression", imaging_report.get("结论", ""))
        findings = imaging_report.get("findings", imaging_report.get("所见", ""))
        report_text = f"{impression} {findings}".lower()

        # 判断事件类型
        event_type = EventType.IMAGING_SUSPECTED_TB
        event_data = {
            "report_id": imaging_report.get("report_id", ""),
            "exam_date": imaging_report.get("exam_date", ""),
            "body_part": imaging_report.get("body_part", ""),
            "impression": impression,
        }

        # 疑似结核
        tb_keywords = ["结核", "肺结核", "结核病", "tb", "tuberculosis",
                       "疑似结核", "考虑结核", "不除外结核"]
        cavity_keywords = ["空洞", "cavity", "cavitation", "虫蚀样空洞"]
        worsening_keywords = ["进展", "加重", "增大", "增多", "恶化", "播散"]

        has_tb = any(kw in report_text for kw in tb_keywords)
        has_cavity = any(kw in report_text for kw in cavity_keywords)
        has_worsening = any(kw in report_text for kw in worsening_keywords)

        if has_cavity:
            event_type = EventType.IMAGING_CAVITY_FOUND
        elif has_worsening:
            event_type = EventType.IMAGING_WORSENING
        elif has_tb:
            event_type = EventType.IMAGING_SUSPECTED_TB

        return self.emit_event(patient_id, event_type, event_data, source="PACS")

    def on_diagnosis_update(self, patient_id: str,
                            diagnosis: Dict[str, Any]) -> List[Optional[PatientEvent]]:
        """诊断更新时触发

        支持同时触发多个事件（如确诊+合并症），避免事件类型被覆盖。
        """
        icd_code = diagnosis.get("code", diagnosis.get("icd10", ""))
        diagnosis_name = diagnosis.get("name", diagnosis.get("diagnosis", ""))

        events = []

        # 检查结核确诊
        if icd_code and (icd_code.startswith("A15") or icd_code.startswith("A16") or
                         icd_code.startswith("A17") or icd_code.startswith("A18") or
                         icd_code.startswith("A19")):
            events.append(self.emit_event(patient_id, EventType.DIAGNOSIS_CONFIRMED, {
                "code": icd_code,
                "name": diagnosis_name,
                "diagnosis_time": diagnosis.get("diagnosis_time", ""),
            }, source="HIS"))

        # 检查合并症（额外事件，不覆盖确诊事件）
        comorbidity_prefixes = ["E10", "E11", "E12", "E13", "E14", "B20", "B21",
                                "B22", "B23", "B24", "J62", "J65"]
        if any(icd_code.startswith(p) for p in comorbidity_prefixes):
            events.append(self.emit_event(patient_id, EventType.COMORBIDITY_FOUND, {
                "code": icd_code,
                "name": diagnosis_name,
                "diagnosis_time": diagnosis.get("diagnosis_time", ""),
            }, source="HIS"))

        # 未匹配任何特定类型时，发出默认诊断变更事件
        if not events:
            events.append(self.emit_event(patient_id, EventType.DIAGNOSIS_CHANGED, {
                "code": icd_code,
                "name": diagnosis_name,
                "diagnosis_time": diagnosis.get("diagnosis_time", ""),
            }, source="HIS"))

        return events

    # ==================== 风险评分 ====================

    def _recalculate_risk(self, patient_id: str,
                          trigger_event: Optional[PatientEvent] = None,
                          pending_notifications: Optional[List[Notification]] = None):
        """重算风险评分（线程安全）

        参数：
            pending_notifications: 可选，通知列表，用于收集风险升级通知（不必在锁内执行回调）
        """
        with self._snapshots_lock:
            snapshot = self._snapshots.get(patient_id)
            if not snapshot or not snapshot.features:
                return

            if self._risk_calculator:
                try:
                    risk_result = self._risk_calculator(snapshot.features)
                    snapshot.last_risk_score = risk_result.get("score")
                    snapshot.last_risk_level = risk_result.get("level", "unknown")

                    # 风险升级时生成通知（仅构造，不执行回调）
                    if trigger_event and risk_result.get("level") in ("high", "critical"):
                        notif = self._generate_notification(trigger_event, snapshot, risk_result)
                        if notif and pending_notifications is not None:
                            pending_notifications.append(notif)

                    LOGGER.info("风险评分重算: %s -> 分数=%.2f, 等级=%s",
                                 patient_id, snapshot.last_risk_score,
                                 snapshot.last_risk_level)
                except Exception as e:
                    LOGGER.error("风险评分计算失败: %s - %s", patient_id, e)

    # ==================== 通知管理 ====================

    def _generate_notification(self, event: PatientEvent,
                                snapshot: PatientSnapshot,
                                risk_result: Optional[Dict[str, Any]] = None):
        """生成医生通知"""
        severity = "info"
        title = ""
        message = ""

        if event.event_type == EventType.SPUTUM_SMEAR_POSITIVE:
            severity = "critical"
            title = "痰涂片阳性结果"
            message = f"患者 {event.patient_id} 痰涂片检查结果为阳性，请立即评估"
        elif event.event_type == EventType.XPERT_MTB_POSITIVE:
            severity = "critical"
            title = "Xpert MTB阳性"
            message = f"患者 {event.patient_id} Xpert MTB/RIF检测阳性，请确认诊断"
        elif event.event_type == EventType.XPERT_RIF_RESISTANT:
            severity = "critical"
            title = "利福平耐药"
            message = f"患者 {event.patient_id} 检测到利福平耐药，请调整治疗方案"
        elif event.event_type == EventType.IMAGING_SUSPECTED_TB:
            severity = "warning"
            title = "影像学疑似结核"
            message = f"患者 {event.patient_id} 影像学检查疑似结核，请进一步检查"
        elif event.event_type == EventType.IMAGING_CAVITY_FOUND:
            severity = "warning"
            title = "影像发现空洞"
            message = f"患者 {event.patient_id} 影像学检查发现空洞，提示活动性结核"
        elif event.event_type == EventType.LAB_RESULT_CRITICAL:
            severity = "critical"
            title = "检验危急值"
            message = f"患者 {event.patient_id} 检验结果出现危急值，请及时处理"
        elif event.event_type == EventType.ADVERSE_REACTION:
            severity = "warning"
            title = "药物不良反应"
            message = f"患者 {event.patient_id} 出现药物不良反应，请评估"
        elif event.event_type == EventType.DIAGNOSIS_CONFIRMED:
            severity = "warning"
            title = "结核病确诊"
            message = f"患者 {event.patient_id} 确诊结核病，请启动管理流程"
        else:
            severity = "info"
            title = "患者数据更新"
            message = f"患者 {event.patient_id} 有新数据更新，请查看"

        risk_score = risk_result.get("score") if risk_result else snapshot.last_risk_score
        risk_level = risk_result.get("level") if risk_result else snapshot.last_risk_level

        notification = Notification(
            notification_id=f"notif_{event.patient_id}_{int(time.time())}",
            patient_id=event.patient_id,
            event_type=event.event_type,
            title=title,
            message=message,
            severity=severity,
            risk_score=risk_score,
            risk_level=risk_level,
        )

        snapshot.pending_notifications.append({
            "notification_id": notification.notification_id,
            "title": title,
            "message": message,
            "severity": severity,
            "created_at": notification.created_at.isoformat(),
        })

        return notification  # 调用方在锁外执行回调

    # ==================== 快照管理 ====================

    def _get_or_create_snapshot(self, patient_id: str) -> PatientSnapshot:
        """获取或创建患者快照（线程安全）"""
        with self._snapshots_lock:
            if patient_id not in self._snapshots:
                self._snapshots[patient_id] = PatientSnapshot(patient_id=patient_id)
            return self._snapshots[patient_id]

    def get_snapshot(self, patient_id: str) -> Optional[PatientSnapshot]:
        """获取患者快照（线程安全，返回副本）"""
        with self._snapshots_lock:
            snapshot = self._snapshots.get(patient_id)
            if snapshot is None:
                return None
            # 返回副本防止外部修改
            return PatientSnapshot(
                patient_id=snapshot.patient_id,
                baseline_established=snapshot.baseline_established,
                baseline_time=snapshot.baseline_time,
                last_update_time=snapshot.last_update_time,
                last_risk_score=snapshot.last_risk_score,
                last_risk_level=snapshot.last_risk_level,
                update_count=snapshot.update_count,
                features=dict(snapshot.features),
                source_data=dict(snapshot.source_data),
                event_history=list(snapshot.event_history),
                pending_notifications=list(snapshot.pending_notifications),
            )

    def get_feature(self, patient_id: str, feature_name: str) -> Any:
        """获取患者某特征值（线程安全）"""
        with self._snapshots_lock:
            snapshot = self._snapshots.get(patient_id)
            if snapshot:
                return snapshot.features.get(feature_name)
            return None

    def get_all_features(self, patient_id: str) -> Dict[str, Any]:
        """获取患者所有特征（线程安全，返回副本）"""
        with self._snapshots_lock:
            snapshot = self._snapshots.get(patient_id)
            if snapshot:
                return dict(snapshot.features)
            return {}

    def get_pending_notifications(self, patient_id: str = "") -> List[Dict[str, Any]]:
        """获取待处理通知（线程安全，返回副本）"""
        with self._snapshots_lock:
            if patient_id:
                snapshot = self._snapshots.get(patient_id)
                if snapshot:
                    return list(snapshot.pending_notifications)
                return []

            all_notifications = []
            for snapshot in self._snapshots.values():
                all_notifications.extend(snapshot.pending_notifications)
            return list(all_notifications)

    def get_event_history(self, patient_id: str,
                          limit: int = 50) -> List[Dict[str, Any]]:
        """获取患者事件历史（线程安全，返回副本）"""
        with self._snapshots_lock:
            snapshot = self._snapshots.get(patient_id)
            if snapshot:
                return list(snapshot.event_history[-limit:])
            return []

    # ==================== 统计与管理 ====================

    def get_statistics(self) -> Dict[str, Any]:
        """获取引擎统计（线程安全）"""
        with self._lock:
            queue_size = len(self._event_queue)

        with self._snapshots_lock:
            total_patients = len(self._snapshots)
            baselines_established = sum(
                1 for s in self._snapshots.values() if s.baseline_established
            )
            total_updates = sum(s.update_count for s in self._snapshots.values())
            pending_notifs = sum(
                len(s.pending_notifications) for s in self._snapshots.values()
            )

            # 风险等级分布
            risk_distribution = {}
            for s in self._snapshots.values():
                level = s.last_risk_level
                risk_distribution[level] = risk_distribution.get(level, 0) + 1

        return {
            "total_patients": total_patients,
            "baselines_established": baselines_established,
            "total_updates": total_updates,
            "pending_notifications": pending_notifs,
            "event_queue_size": queue_size,
            "risk_distribution": risk_distribution,
            "registered_handlers": sum(
                len(h) for h in self._event_handlers.values()
            ),
        }

    def clear_patient_data(self, patient_id: str):
        """清除患者数据（线程安全）"""
        with self._snapshots_lock:
            self._snapshots.pop(patient_id, None)
        self._patient_cache.pop(patient_id, None)

    def reset(self):
        """重置引擎（线程安全）"""
        with self._snapshots_lock:
            self._snapshots.clear()
        self._patient_cache.clear()
        with self._lock:
            self._event_queue.clear()


# ============================================================================
# 便捷工厂函数
# ============================================================================

def create_default_engine() -> IncrementalEngine:
    """创建带默认配置的增量引擎"""
    engine = IncrementalEngine()

    # 注册默认事件处理器
    engine.register_handler(EventType.LAB_RESULT_NEW, _default_lab_handler)
    engine.register_handler(EventType.SPUTUM_SMEAR_POSITIVE, _default_sputum_handler)
    engine.register_handler(EventType.DIAGNOSIS_CONFIRMED, _default_diagnosis_handler)
    engine.register_handler(EventType.MEDICATION_STARTED, _default_medication_handler)

    return engine


def _default_lab_handler(event: PatientEvent) -> Dict[str, Any]:
    """默认检验结果处理器"""
    test_code = event.data.get("test_code", "")
    value = event.data.get("value", "")

    features = {}
    if "sputum_smear" in test_code.lower() or "涂片" in test_code:
        features["sputum_smear"] = "2" if "阳性" in str(value) or value in ("2", "positive") else "1"
    elif "xpert" in test_code.lower():
        features["xpert_mtb_rif"] = str(value)
    elif "esr" in test_code.lower() or "血沉" in test_code:
        features["esr"] = value
    elif "crp" in test_code.lower() or "c反应" in test_code or "c反应蛋白" in test_code:
        features["crp"] = value
    elif "血糖" in test_code or "glucose" in test_code.lower():
        features["blood_glucose"] = value

    return features


def _default_sputum_handler(event: PatientEvent) -> Dict[str, Any]:
    """默认痰涂片阳性处理器"""
    return {
        "sputum_smear": "2",
        "active_tb": "1",
        "sputum_positive_time": event.timestamp.isoformat(),
    }


def _default_diagnosis_handler(event: PatientEvent) -> Dict[str, Any]:
    """默认确诊处理器"""
    return {
        "active_tb": "1",
        "diagnosis_code": event.data.get("code", ""),
        "diagnosis_name": event.data.get("name", ""),
        "diagnosis_time": event.data.get("diagnosis_time",
                                          event.timestamp.isoformat()),
    }


def _default_medication_handler(event: PatientEvent) -> Dict[str, Any]:
    """默认用药处理器"""
    return {
        "treatment": "1",
        "treatment_start_time": event.timestamp.isoformat(),
    }