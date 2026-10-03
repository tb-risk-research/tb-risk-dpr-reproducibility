#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""传染病报告质量控制模块

提供传染病报告管理核心质控功能：
1. 漏报筛查：自动扫描HIS中结核诊断但未报卡的病例
2. 迟报预警：诊断后超时未报的迟报监测
3. 报卡质量统计：及时率、完整率、准确率、重报率统计报表

符合《传染病信息报告管理规范》和《医疗机构传染病报告管理要求》。
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .monitoring import AlertLevel

LOGGER = logging.getLogger("tb_risk.health_interop.quality_control")


# ============================================================================
# 数据源协议（与具体实现解耦）
# ============================================================================

class DiagnosisDataSourceProtocol(Protocol):
    """诊断记录数据源协议。
    
    QualityControlService通过此协议获取HIS诊断记录，
    支持db_direct直连、HL7消息订阅、FHIR Condition查询等多种数据源实现。
    """
    
    def get_tb_diagnoses(self, since: Optional[datetime.datetime] = None,
                         until: Optional[datetime.datetime] = None) -> List[DiagnosisRecord]:
        """获取指定时间范围内的结核相关诊断记录。
        
        参数：
            since: 开始时间（包含），None表示不限制
            until: 结束时间（包含），None表示到当前时间
        
        返回：
            List[DiagnosisRecord]: 诊断记录列表
        """
        ...


class ReportCardDataSourceProtocol(Protocol):
    """传染病报告卡数据源协议。
    
    QualityControlService通过此协议查询已报告的卡片数据，
    支持workflow数据库、本地存储、外部系统查询等实现。
    """
    
    def get_all_cards(self, since: Optional[datetime.datetime] = None,
                      until: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        """获取指定时间范围内的报告卡列表。
        
        每个卡片字典应包含：card_id, patient_id, status, created_at, 
        submitted_at, doctor_id, doctor_name, diagnosis, icd_code等字段。
        """
        ...
    
    def get_card_by_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """根据患者ID查询最新报告卡。"""
        ...


class NullDiagnosisDataSource:
    """空诊断数据源（降级模式），不返回任何诊断记录。"""
    
    def get_tb_diagnoses(self, since=None, until=None) -> List[DiagnosisRecord]:
        LOGGER.debug("使用NullDiagnosisDataSource，未配置诊断数据源，漏报筛查不可用")
        return []


class NullReportCardDataSource:
    """空报告卡数据源（降级模式），返回空列表。"""
    
    def get_all_cards(self, since=None, until=None) -> List[Dict[str, Any]]:
        LOGGER.debug("使用NullReportCardDataSource，未配置卡片数据源")
        return []
    
    def get_card_by_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        return None

# 迟报阈值（小时）：诊断后超过24小时未提交报告卡视为迟报
DEFAULT_LATE_REPORT_THRESHOLD_HOURS = 24
# 漏报筛查回溯天数：扫描最近N天内的诊断记录
DEFAULT_MISSED_REPORT_LOOKBACK_DAYS = 30
# 预警检查间隔（秒）
DEFAULT_CHECK_INTERVAL_SECONDS = 3600  # 每小时检查一次


class QCAlertType(Enum):
    """质控预警类型"""
    MISSED_REPORT = "missed_report"       # 漏报
    LATE_REPORT = "late_report"           # 迟报
    INCOMPLETE_CARD = "incomplete_card"   # 卡片信息不完整
    DUPLICATE_CARD = "duplicate_card"     # 重复报卡
    OVERDUE_REVIEW = "overdue_review"     # 审核超时


@dataclass
class QCAlert:
    """质控预警记录"""
    alert_id: str
    alert_type: QCAlertType
    level: AlertLevel
    patient_id: str
    patient_name: str
    doctor_id: str
    doctor_name: str
    diagnosis: str
    diagnosis_time: Optional[datetime.datetime]
    card_id: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    acknowledged: bool = False
    acknowledged_by: str = ""
    acknowledged_at: Optional[datetime.datetime] = None


@dataclass
class QCStatistics:
    """报卡质量统计"""
    period_start: datetime.datetime
    period_end: datetime.datetime
    total_cards: int = 0
    timely_count: int = 0          # 及时报告数
    late_count: int = 0            # 迟报数
    missed_count: int = 0          # 漏报数（期间发现的）
    complete_count: int = 0        # 信息完整数
    incomplete_count: int = 0      # 信息不全数
    duplicate_count: int = 0       # 重复报卡数
    rejected_count: int = 0        # 驳回重填数
    
    @property
    def timeliness_rate(self) -> float:
        """及时率 = 及时数/已完成报告数"""
        total_reported = self.timely_count + self.late_count
        return self.timely_count / total_reported if total_reported > 0 else 1.0
    
    @property
    def completeness_rate(self) -> float:
        """完整率"""
        return self.complete_count / self.total_cards if self.total_cards > 0 else 1.0
    
    @property
    def accuracy_rate(self) -> float:
        """准确率 = 1 - 驳回率"""
        return 1.0 - (self.rejected_count / self.total_cards if self.total_cards > 0 else 0)
    
    @property
    def duplicate_rate(self) -> float:
        """重报率"""
        return self.duplicate_count / self.total_cards if self.total_cards > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_cards": self.total_cards,
            "timely_count": self.timely_count,
            "late_count": self.late_count,
            "missed_count": self.missed_count,
            "complete_count": self.complete_count,
            "incomplete_count": self.incomplete_count,
            "duplicate_count": self.duplicate_count,
            "rejected_count": self.rejected_count,
            "timeliness_rate": round(self.timeliness_rate, 4),
            "completeness_rate": round(self.completeness_rate, 4),
            "accuracy_rate": round(self.accuracy_rate, 4),
            "duplicate_rate": round(self.duplicate_rate, 4),
        }


class DiagnosisRecord:
    """诊断记录（从HIS/LIS/PACS获取的诊断信息，用于比对）"""
    __slots__ = ("patient_id", "patient_name", "diagnosis", "icd_code",
                 "diagnosis_time", "doctor_id", "doctor_name", "department",
                 "source")
    
    def __init__(self, patient_id: str, diagnosis: str,
                 diagnosis_time: datetime.datetime, **kwargs):
        self.patient_id = patient_id
        self.patient_name = kwargs.get("patient_name", "")
        self.diagnosis = diagnosis
        self.icd_code = kwargs.get("icd_code", "")
        self.diagnosis_time = diagnosis_time
        self.doctor_id = kwargs.get("doctor_id", "")
        self.doctor_name = kwargs.get("doctor_name", "")
        self.department = kwargs.get("department", "")
        self.source = kwargs.get("source", "his")  # his/lis/pacs/manual


class QualityControlService:
    """传染病报告质控服务
    
    功能：
    - 漏报筛查：对比HIS诊断记录与已报告卡片，发现未报病例
    - 迟报监测：诊断后超过阈值未提交卡片触发预警
    - 质量统计：生成报卡及时率、完整率、准确率、重报率报表
    
    数据源解耦：通过DiagnosisDataSourceProtocol和ReportCardDataSourceProtocol
    接入不同数据源（db_direct直连、HL7订阅、FHIR查询等），未配置时使用Null实现优雅降级。
    """
    
    def __init__(self, db_manager=None, audit_logger=None,
                 diagnosis_source: DiagnosisDataSourceProtocol = None,
                 card_source: ReportCardDataSourceProtocol = None,
                 late_threshold_hours: int = DEFAULT_LATE_REPORT_THRESHOLD_HOURS,
                 missed_lookback_days: int = DEFAULT_MISSED_REPORT_LOOKBACK_DAYS):
        self.db_manager = db_manager
        self.audit_logger = audit_logger
        self.late_threshold_hours = late_threshold_hours
        self.missed_lookback_days = missed_lookback_days
        
        self._running = False
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._alerts: List[QCAlert] = []
        self._alerts_lock = threading.Lock()
        # Protocol数据源（优先使用）
        self._diagnosis_source_proto = diagnosis_source or NullDiagnosisDataSource()
        self._card_source_proto = card_source or NullReportCardDataSource()
        # Callable数据源（向后兼容）
        self._diagnosis_source: Optional[Callable[[], List[DiagnosisRecord]]] = None
        self._card_query_source: Optional[Callable[[], List[Dict[str, Any]]]] = None
        self._alert_callbacks: List[Callable[[QCAlert], None]] = []
        self._check_interval = DEFAULT_CHECK_INTERVAL_SECONDS
        
        self._ensure_qc_tables()
    
    def set_diagnosis_data_source(self, source: DiagnosisDataSourceProtocol):
        """设置诊断记录数据源（Protocol模式，支持db_direct/HL7/FHIR等实现）"""
        self._diagnosis_source_proto = source
    
    def set_card_data_source(self, source: ReportCardDataSourceProtocol):
        """设置报告卡数据源（Protocol模式）"""
        self._card_source_proto = source
    
    def set_diagnosis_source(self, source_func: Callable[[], List[DiagnosisRecord]]):
        """设置诊断记录数据源（Callable模式，向后兼容）
        
        source_func应返回List[DiagnosisRecord]
        """
        self._diagnosis_source = source_func
    
    def set_card_query_source(self, query_func: Callable[[], List[Dict[str, Any]]]):
        """设置已报卡查询数据源（Callable模式，向后兼容）
        
        query_func应返回卡片字典列表，每个字典需包含：
        card_id, patient_id, status, created_at, submitted_at, doctor_id, doctor_name...
        """
        self._card_query_source = query_func
    
    def register_alert_callback(self, callback: Callable[[QCAlert], None]):
        """注册预警回调，当产生新预警时调用"""
        self._alert_callbacks.append(callback)
    
    def _ensure_qc_tables(self):
        """确保质控预警表存在"""
        if not self.db_manager:
            return
        try:
            # 尝试用db_manager建表
            if hasattr(self.db_manager, 'execute'):
                self.db_manager.execute("""
                    CREATE TABLE IF NOT EXISTS qc_alerts (
                        alert_id TEXT PRIMARY KEY,
                        alert_type TEXT NOT NULL,
                        level TEXT NOT NULL,
                        patient_id TEXT NOT NULL,
                        patient_name TEXT DEFAULT '',
                        doctor_id TEXT DEFAULT '',
                        doctor_name TEXT DEFAULT '',
                        diagnosis TEXT DEFAULT '',
                        diagnosis_time REAL,
                        card_id TEXT DEFAULT '',
                        message TEXT DEFAULT '',
                        details TEXT DEFAULT '{}',
                        created_at REAL NOT NULL,
                        acknowledged INTEGER DEFAULT 0,
                        acknowledged_by TEXT DEFAULT '',
                        acknowledged_at REAL
                    )
                """)
                if hasattr(self.db_manager, 'commit'):
                    self.db_manager.commit()
        except Exception as e:
            LOGGER.debug("质控表创建失败（可能db_manager不支持execute）: %s", e)
    
    # ---- 漏报筛查 ----
    
    def _get_diagnoses(self, since: datetime.datetime = None) -> List[DiagnosisRecord]:
        """从配置的数据源获取诊断记录"""
        # 优先使用Callable数据源（向后兼容）
        if self._diagnosis_source:
            try:
                return self._diagnosis_source()
            except Exception as e:
                LOGGER.error("Callable诊断数据源失败: %s", e)
        # 使用Protocol数据源
        try:
            return self._diagnosis_source_proto.get_tb_diagnoses(since=since)
        except Exception as e:
            LOGGER.debug("Protocol诊断数据源获取失败: %s", e)
        return []
    
    def _get_existing_cards(self, since: datetime.datetime = None) -> List[Dict[str, Any]]:
        """从配置的数据源获取已报卡列表"""
        if self._card_query_source:
            try:
                return self._card_query_source()
            except Exception as e:
                LOGGER.error("Callable卡片数据源失败: %s", e)
        try:
            return self._card_source_proto.get_all_cards(since=since)
        except Exception as e:
            LOGGER.debug("Protocol卡片数据源获取失败: %s", e)
        return []
    
    def scan_missed_reports(self, diagnoses: List[DiagnosisRecord] = None,
                            existing_cards: List[Dict[str, Any]] = None) -> List[QCAlert]:
        """扫描漏报病例：有结核诊断但未创建/提交报告卡
        
        参数：
            diagnoses: 诊断记录列表；若None则使用配置的数据源
            existing_cards: 已报卡列表；若None则使用配置的数据源
        
        返回：
            漏报预警列表
        """
        cutoff_time = datetime.datetime.now() - datetime.timedelta(
            days=self.missed_lookback_days
        )
        if diagnoses is None:
            diagnoses = self._get_diagnoses(since=cutoff_time)
        if existing_cards is None:
            existing_cards = self._get_existing_cards(since=cutoff_time)
        
        diagnoses = diagnoses or []
        existing_cards = existing_cards or []
        
        # 提取已报卡患者集合
        reported_patients = set()
        card_patient_map = {}
        for card in existing_cards:
            pid = card.get("patient_id", "")
            if pid:
                reported_patients.add(pid)
                card_patient_map[pid] = card
        
        # 结核相关诊断关键词/ICD前缀
        tb_keywords = ["结核", "肺结核", "tb", "tuberculosis"]
        tb_icd_prefixes = ("A15", "A16", "A17", "A18", "A19")
        
        alerts = []
        for dx in diagnoses:
            # 只检查结核相关诊断
            is_tb = False
            if dx.icd_code:
                for prefix in tb_icd_prefixes:
                    if dx.icd_code.upper().startswith(prefix):
                        is_tb = True
                        break
            if not is_tb:
                dx_lower = dx.diagnosis.lower()
                for kw in tb_keywords:
                    if kw.lower() in dx_lower:
                        is_tb = True
                        break
            if not is_tb:
                continue
            
            # 只检查回溯期内的诊断
            if dx.diagnosis_time and dx.diagnosis_time < cutoff_time:
                continue
            
            # 排除陈旧性结核
            if "陈旧" in dx.diagnosis or "已愈" in dx.diagnosis or "old tb" in dx.diagnosis.lower():
                continue
            
            # 检查是否已报卡
            if dx.patient_id in reported_patients:
                continue
            
            # 创建漏报预警
            alert = QCAlert(
                alert_id=f"missed_{dx.patient_id}_{int(time.time()*1000)}",
                alert_type=QCAlertType.MISSED_REPORT,
                level=AlertLevel.CRITICAL,
                patient_id=dx.patient_id,
                patient_name=dx.patient_name,
                doctor_id=dx.doctor_id,
                doctor_name=dx.doctor_name,
                diagnosis=dx.diagnosis,
                diagnosis_time=dx.diagnosis_time,
                message=f"患者{dx.patient_name}诊断为{dx.diagnosis}但未报告传染病卡（漏报）",
                details={"department": dx.department, "source": dx.source},
            )
            alerts.append(alert)
            self._add_alert(alert)
        
        LOGGER.info("漏报筛查完成，发现%d例漏报", len(alerts))
        return alerts
    
    # ---- 迟报监测 ----
    
    def scan_late_reports(self, cards: List[Dict[str, Any]] = None) -> List[QCAlert]:
        """扫描迟报病例：创建卡片后超过阈值时间仍未提交
        
        参数：
            cards: 卡片列表；若None则使用配置的数据源
        """
        if cards is None:
            cards = self._get_existing_cards()
        
        cards = cards or []
        alerts = []
        now = datetime.datetime.now()
        
        for card in cards:
            status = card.get("status", "")
            # 只检查DRAFT/FILLED/PENDING_REVIEW状态
            from .workflow import CardStatus
            pending_statuses = {CardStatus.DRAFT.value if hasattr(CardStatus, 'DRAFT') else "DRAFT",
                              CardStatus.FILLED.value if hasattr(CardStatus, 'FILLED') else "FILLED",
                              CardStatus.PENDING_REVIEW.value if hasattr(CardStatus, 'PENDING_REVIEW') else "PENDING_REVIEW"}
            if status not in pending_statuses and status not in ("DRAFT", "FILLED", "PENDING_REVIEW"):
                continue
            
            # 获取诊断时间或卡片创建时间
            diag_time = card.get("diagnosis_time")
            created_at = card.get("created_at")
            reference_time = None
            for t in [diag_time, created_at]:
                if isinstance(t, datetime.datetime):
                    reference_time = t
                    break
                elif isinstance(t, (int, float)):
                    reference_time = datetime.datetime.fromtimestamp(t)
                    break
                elif isinstance(t, str):
                    try:
                        reference_time = datetime.datetime.fromisoformat(t)
                    except (ValueError, TypeError):
                        pass
                    if reference_time:
                        break
            
            if not reference_time:
                continue
            
            elapsed_hours = (now - reference_time).total_seconds() / 3600
            if elapsed_hours <= self.late_threshold_hours:
                continue
            
            # 24-48小时WARNING，超过48小时CRITICAL
            level = AlertLevel.WARNING if elapsed_hours <= 48 else AlertLevel.CRITICAL
            
            alert = QCAlert(
                alert_id=f"late_{card.get('card_id', '')}_{int(time.time()*1000)}",
                alert_type=QCAlertType.LATE_REPORT,
                level=level,
                patient_id=card.get("patient_id", ""),
                patient_name=card.get("patient_name", ""),
                doctor_id=card.get("creator_id", card.get("doctor_id", "")),
                doctor_name=card.get("creator_name", card.get("doctor_name", "")),
                diagnosis=card.get("diagnosis", "肺结核"),
                diagnosis_time=reference_time,
                card_id=card.get("card_id", ""),
                message=f"诊断后{elapsed_hours:.1f}小时仍未提交报告卡（迟报）",
                details={"elapsed_hours": round(elapsed_hours, 1),
                         "threshold_hours": self.late_threshold_hours,
                         "current_status": status},
            )
            alerts.append(alert)
            self._add_alert(alert)
        
        LOGGER.info("迟报监测完成，发现%d例迟报", len(alerts))
        return alerts
    
    # ---- 卡片完整性检查 ----
    
    def check_card_completeness(self, card_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """检查报告卡信息完整性
        
        返回 (is_complete, missing_fields)
        """
        required_fields = [
            "patient_id", "name", "gender", "age",
            "diagnosis", "diagnosis_date",
            "doctor_id", "doctor_name", "department",
            "address", "phone",
        ]
        missing = []
        for field in required_fields:
            val = card_data.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing.append(field)
        return (len(missing) == 0, missing)
    
    # ---- 重复报卡检查 ----
    
    def find_duplicate_cards(self, cards: List[Dict[str, Any]]) -> List[Tuple[Dict, Dict]]:
        """查找重复报卡（同一患者、同一诊断日期）
        
        返回重复卡片对列表
        """
        duplicates = []
        patient_date_map: Dict[Tuple[str, str], List[Dict]] = {}
        
        for card in cards:
            pid = card.get("patient_id", "")
            diag_date = card.get("diagnosis_date", "")
            if isinstance(diag_date, datetime.datetime):
                diag_date = diag_date.strftime("%Y-%m-%d")
            key = (pid, str(diag_date)[:10] if diag_date else "")
            if not pid or not diag_date:
                continue
            patient_date_map.setdefault(key, []).append(card)
        
        for key, card_list in patient_date_map.items():
            if len(card_list) > 1:
                for i in range(len(card_list)):
                    for j in range(i+1, len(card_list)):
                        duplicates.append((card_list[i], card_list[j]))
        
        return duplicates
    
    # ---- 质量统计 ----
    
    def generate_statistics(self, start_date: datetime.datetime = None,
                            end_date: datetime.datetime = None,
                            cards: List[Dict[str, Any]] = None) -> QCStatistics:
        """生成指定时间段内的报卡质量统计报表"""
        if end_date is None:
            end_date = datetime.datetime.now()
        if start_date is None:
            start_date = end_date - datetime.timedelta(days=30)
        
        if cards is None and self._card_query_source:
            try:
                cards = self._card_query_source()
            except Exception as e:
                LOGGER.error("获取卡片列表失败: %s", e)
                cards = []
        cards = cards or []
        
        stats = QCStatistics(period_start=start_date, period_end=end_date)
        
        for card in cards:
            # 过滤时间段外的卡片
            card_time = card.get("created_at") or card.get("submitted_at")
            if isinstance(card_time, (int, float)):
                card_dt = datetime.datetime.fromtimestamp(card_time)
            elif isinstance(card_time, str):
                try:
                    card_dt = datetime.datetime.fromisoformat(card_time)
                except (ValueError, TypeError):
                    continue
            elif isinstance(card_time, datetime.datetime):
                card_dt = card_time
            else:
                continue
            
            if card_dt < start_date or card_dt > end_date:
                continue
            
            stats.total_cards += 1
            
            # 及时性
            diag_time = card.get("diagnosis_time") or card.get("diagnosis_date")
            if diag_time:
                if isinstance(diag_time, (int, float)):
                    diag_dt = datetime.datetime.fromtimestamp(diag_time)
                elif isinstance(diag_time, str):
                    try:
                        diag_dt = datetime.datetime.fromisoformat(diag_time)
                    except (ValueError, TypeError):
                        diag_dt = None
                elif isinstance(diag_time, datetime.datetime):
                    diag_dt = diag_time
                else:
                    diag_dt = None
                
                submitted = card.get("submitted_at") or card.get("created_at")
                if diag_dt and submitted:
                    if isinstance(submitted, (int, float)):
                        sub_dt = datetime.datetime.fromtimestamp(submitted)
                    elif isinstance(submitted, str):
                        try:
                            sub_dt = datetime.datetime.fromisoformat(submitted)
                        except (ValueError, TypeError):
                            sub_dt = None
                    else:
                        sub_dt = submitted
                    
                    if sub_dt:
                        hours = (sub_dt - diag_dt).total_seconds() / 3600
                        if hours <= self.late_threshold_hours:
                            stats.timely_count += 1
                        else:
                            stats.late_count += 1
            
            # 完整性
            is_complete, _ = self.check_card_completeness(card)
            if is_complete:
                stats.complete_count += 1
            else:
                stats.incomplete_count += 1
            
            # 驳回次数统计
            if card.get("was_rejected") or card.get("rejected_count", 0) > 0:
                stats.rejected_count += 1
        
        # 统计重复报卡
        duplicates = self.find_duplicate_cards(cards)
        stats.duplicate_count = len(duplicates)
        
        return stats
    
    # ---- 预警管理 ----
    
    def _add_alert(self, alert: QCAlert):
        """添加预警并触发回调"""
        with self._alerts_lock:
            # 去重：同一患者同一类型不重复添加
            for existing in self._alerts:
                if (existing.patient_id == alert.patient_id and 
                    existing.alert_type == alert.alert_type and
                    not existing.acknowledged):
                    return
            self._alerts.append(alert)
        
        # 持久化到数据库
        self._save_alert(alert)
        
        # 触发回调
        for cb in self._alert_callbacks:
            try:
                cb(alert)
            except Exception as e:
                LOGGER.debug("预警回调执行失败: %s", e)
    
    def _save_alert(self, alert: QCAlert):
        """保存预警到数据库"""
        if not self.db_manager or not hasattr(self.db_manager, 'execute'):
            return
        try:
            import json
            diag_ts = alert.diagnosis_time.timestamp() if alert.diagnosis_time else None
            ack_ts = alert.acknowledged_at.timestamp() if alert.acknowledged_at else None
            self.db_manager.execute("""
                INSERT OR IGNORE INTO qc_alerts 
                (alert_id, alert_type, level, patient_id, patient_name,
                 doctor_id, doctor_name, diagnosis, diagnosis_time,
                 card_id, message, details, created_at,
                 acknowledged, acknowledged_by, acknowledged_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                alert.alert_id, alert.alert_type.value, alert.level.value,
                alert.patient_id, alert.patient_name,
                alert.doctor_id, alert.doctor_name, alert.diagnosis,
                diag_ts,
                alert.card_id, alert.message,
                json.dumps(alert.details, ensure_ascii=False),
                alert.created_at.timestamp(),
                1 if alert.acknowledged else 0,
                alert.acknowledged_by, ack_ts
            ])
            if hasattr(self.db_manager, 'commit'):
                self.db_manager.commit()
        except Exception as e:
            LOGGER.debug("保存预警失败: %s", e)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """确认预警（标记为已处理）"""
        with self._alerts_lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id and not alert.acknowledged:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = datetime.datetime.now()
                    self._save_alert(alert)
                    return True
        return False
    
    def get_pending_alerts(self, alert_type: QCAlertType = None) -> List[QCAlert]:
        """获取未处理的预警列表"""
        with self._alerts_lock:
            result = [a for a in self._alerts if not a.acknowledged]
            if alert_type:
                result = [a for a in result if a.alert_type == alert_type]
            return result
    
    def get_all_alerts(self, limit: int = 100) -> List[QCAlert]:
        """获取最近N条预警"""
        with self._alerts_lock:
            return sorted(self._alerts, key=lambda a: a.created_at, reverse=True)[:limit]
    
    # ---- 后台监控 ----
    
    def start_monitoring(self, interval_seconds: int = None):
        """启动后台监控线程（定期检查漏报和迟报）"""
        if self._running:
            return
        if interval_seconds:
            self._check_interval = interval_seconds
        
        self._running = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="QC-Monitor"
        )
        self._monitor_thread.start()
        LOGGER.info("质控监控已启动，检查间隔：%d秒", self._check_interval)
    
    def stop_monitoring(self):
        """停止后台监控"""
        self._running = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        LOGGER.info("质控监控已停止")
    
    def _monitor_loop(self):
        """后台监控循环"""
        while not self._stop_event.is_set():
            try:
                self.scan_missed_reports()
                self.scan_late_reports()
            except Exception as e:
                LOGGER.error("质控检查异常: %s", e, exc_info=True)
            self._stop_event.wait(self._check_interval)
    
    def run_once(self):
        """手动触发一次质控检查"""
        missed = self.scan_missed_reports()
        late = self.scan_late_reports()
        return {"missed_reports": len(missed), "late_reports": len(late)}


# 便捷函数：使用workflow数据库作为卡片源
def create_qc_service_for_workflow(db_path: str = "report_cards.db",
                                   audit_db_path: str = "audit.db",
                                   late_hours: int = 24) -> QualityControlService:
    """便捷创建绑定workflow数据库的QC服务"""
    from .audit import AuditLogger, get_default_auditor
    import sqlite3
    
    auditor = get_default_auditor(audit_db_path)
    
    class SimpleDBManager:
        def __init__(self, path):
            self.path = path
            self.conn = sqlite3.connect(path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._closed = False
        
        def execute(self, sql, params=None):
            if self._closed:
                raise RuntimeError("数据库连接已关闭")
            if params:
                return self.conn.execute(sql, params)
            return self.conn.execute(sql)
        
        def commit(self):
            if not self._closed:
                self.conn.commit()
        
        def close(self):
            if not self._closed:
                try:
                    self.conn.close()
                except Exception as _e:
                    LOGGER.debug("关闭SimpleDBManager数据库连接失败: %s", _e)
                self._closed = True
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
            return False
    
    db = SimpleDBManager(db_path)
    qc = QualityControlService(
        db_manager=db, audit_logger=auditor,
        late_threshold_hours=late_hours
    )
    
    def query_cards():
        try:
            rows = db.execute("SELECT * FROM report_cards").fetchall()
            return [dict(r) for r in rows]
        except Exception as _e:
            LOGGER.warning("查询报表卡片失败，返回空列表: %s", _e)
            return []
    
    qc.set_card_query_source(query_cards)
    
    # 将自动创建的 db 附加到 qc 对象，方便调用方管理生命周期
    qc._auto_db = db
    
    # 注册 atexit 钩子，确保进程退出时数据库连接被关闭
    import atexit
    atexit.register(db.close)
    
    return qc
