#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""医疗信息标准接口适配包

提供与医院信息系统（HIS/EMR/LIS/PACS）、疾控中心系统的标准化互操作能力，
覆盖HL7 FHIR R4、HL7 v2.x MLLP、数据库直连视图、LIS/PACS、疾控上报、
通用REST/WebService六类接口模式，并内置接口监控、数据质量校验、术语绑定、
患者匹配（EMPI）等基础设施。

模块结构：
- base.py:        适配器基类、常量、术语映射表、工具函数
- fhir/:          HL7 FHIR R4 客户端（按需查询+订阅通知+资源解析）
- hl7v2/:         HL7 v2.x MLLP 消息接收服务（ADT/ORM/ORU/MDM消息解析）
- db_direct/:     数据库直连适配器（只读视图、增量抽取、数据质量校验）
- lis_pacs/:      LIS检验/PACS影像报告接入（LOINC映射、影像征象NLP）
- cdc/:           疾控中心对接（传染病报告卡自动填报、回传数据接收）
- workflow/:      传染病报告卡多级审核工作流（状态机、审核记录、权限控制）
- messaging/:     可靠消息队列（确认、重试、死信队列、幂等处理）
- contact_tracing/: GNN接触网络与疾控数据双向联动（密接追踪、筛查任务、闭环）
- generic/:       通用WebService/RESTful适配器（认证、重试、熔断）
- monitoring/:    接口监控与告警（可用率、响应时间、数据质量指标）

设计原则：
1. 适配器模式：所有外部系统接入都实现 HealthcareAdapter 基类接口
2. 数据契约：输出统一为 schemas.PatientRecord + list[ContactRecord]
3. 容错优先：超时、重试、熔断、死信队列、脏数据隔离，绝不丢消息
4. 安全合规：只读权限、IP绑定、加密存储、脱敏日志、审计追踪
5. 术语绑定：所有字段绑定国标/国际编码体系，自动转码

版本：1.0.0
"""

__version__ = "1.0.0"

import logging as _logging

_logger = _logging.getLogger(__name__)


def _check_dependencies() -> dict:
    """检查可选依赖是否已安装，返回依赖状态字典并在日志中给出提示"""
    deps_status = {}
    install_hints = []

    optional_deps = [
        ("requests", "requests>=2.28.0", "增强HTTP支持（FHIR RESTful客户端推荐）"),
        ("lxml", "lxml>=4.9.0", "高性能XML/HL7 v2.x消息解析"),
        ("pymysql", "pymysql>=1.0.0", "MySQL/MariaDB数据库直连接入"),
        ("psycopg2", "psycopg2-binary>=2.9.0", "PostgreSQL数据库直连接入"),
        ("zeep", "zeep>=4.2.0", "完整SOAP WebService支持（WSDL解析）"),
    ]

    for mod_name, pip_name, desc in optional_deps:
        try:
            __import__(mod_name)
            deps_status[mod_name] = True
        except ImportError:
            deps_status[mod_name] = False
            install_hints.append(f"  - {desc}: pip install {pip_name}")

    missing = [m for m, ok in deps_status.items() if not ok]
    if missing:
        _logger.info(
            "health_interop可选依赖未完全安装（不影响核心功能）。"
            "如需启用对应接口，请安装缺失依赖：\n%s",
            "\n".join(install_hints) if install_hints else ""
        )
    else:
        _logger.debug("health_interop所有可选依赖均已安装。")

    return deps_status


_DEPENDENCIES = _check_dependencies()

from .base import (
    HealthcareAdapter,
    AdapterConfig,
    AdapterResult,
    PatientIdentifier,
    DataQualityIssue,
    TERMINOLOGY_SYSTEMS,
    FHIR_RESOURCE_MAPPING,
    LOINC_TB_PANELS,
    ICD10_TB_CODES,
    GENDER_MAPPING,
    timestamp_to_datetime,
    safe_parse_datetime,
    mask_sensitive,
    calculate_age,
)

from .monitoring import (
    InterfaceMonitor,
    CircuitBreaker,
    MetricsCollector,
    AlertLevel,
)

from .fhir import (
    FHIRClient,
    FHIRResourceParser,
    FHIRSubscriptionHandler,
    FHIRTermMapper,
)

from .hl7v2 import (
    MLLPServer,
    HL7MessageParser,
    HL7Message,
    HL7MessageQueue,
    HL7AckBuilder,
)

from .db_direct import DatabaseDirectAdapter
from .lis_pacs import LISPACSAdapter, LISPACSDecorator, LISMapper, RadiologyNLP
from .feature_mapper import FeatureMapper, map_adapter_result_to_predictor_input
from .cdc import (
    TBCardFiller, InfectiousDiseaseCard,
    CDCCallbackParser, CDCSecurity,
    CDCWebhookServer,
)
from .workflow import (
    ReviewWorkflow, CardStatus, UserRole, ReviewAction,
    ReviewRecord, WorkflowError,
)
from .messaging import (
    ReliableMessageQueue, ReliableMessage,
    MessageStatus, QueueType, MessageHandlerError,
)
from .contact_tracing import (
    ContactTracingManager, ContactRecord, ScreeningTask,
)
from .audit import (
    AuditLogger, AuditRecord, AuditOperationType,
)
from .quality_control import (
    QualityControlService, QCAlert, QCAlertType, QCStatistics,
    DiagnosisRecord, AlertLevel,
    DiagnosisDataSourceProtocol, ReportCardDataSourceProtocol,
    NullDiagnosisDataSource, NullReportCardDataSource,
)
from .generic import (
    RESTClient, SOAPClient,
    AuthProvider, NoAuth, APIKeyAuth,
    OAuth2ClientCredentials, SignatureAuth,
)
from .duplicate_check import (
    DuplicateChecker, DuplicateMatch,
    CDCQuarantineProtocol, NullCDCQuarantine,
    check_duplicate_before_submit,
)

from .etl_pipeline import (
    ETLPipeline, ETLConfig, ETLResult, DataSourceRecord,
)

from .feature_calculator import (
    FeatureCalculator, MissingValueImputer, FillRecord, CalculationResult,
    compute_bcg_status, compute_smoking_index, compute_bmi,
    extract_contact_history,
)

from .incremental_engine import (
    IncrementalEngine, EventType, PatientSnapshot, PatientEvent,
    Notification,
)

from .adapters import (
    DATA_SOURCE_TYPES,
    SyncStatus,
    SyncSchedule,
    DataSourceConfig,
    SyncRunRecord,
    BaseDataSourceAdapter,
    HISAdapter,
    LISAdapter,
    PACSAdapter,
    CDCDataSourceAdapter,
    AdapterRegistry,
)

from .sync_scheduler import SyncScheduler

__all__ = [
    # 基类与核心
    'HealthcareAdapter',
    'AdapterConfig',
    'AdapterResult',
    'PatientIdentifier',
    'DataQualityIssue',
    # 术语常量
    'TERMINOLOGY_SYSTEMS',
    'FHIR_RESOURCE_MAPPING',
    'LOINC_TB_PANELS',
    'ICD10_TB_CODES',
    'GENDER_MAPPING',
    # 工具函数
    'timestamp_to_datetime',
    'safe_parse_datetime',
    'mask_sensitive',
    'calculate_age',
    # 监控
    'InterfaceMonitor',
    'CircuitBreaker',
    'MetricsCollector',
    'AlertLevel',
    # FHIR
    'FHIRClient',
    'FHIRResourceParser',
    'FHIRSubscriptionHandler',
    'FHIRTermMapper',
    # HL7 v2.x
    'MLLPServer',
    'HL7MessageParser',
    'HL7Message',
    'HL7MessageQueue',
    'HL7AckBuilder',
    # 数据库直连
    'DatabaseDirectAdapter',
    # 特征映射（AdapterResult → tb_risk核心特征）
    'FeatureMapper',
    'map_adapter_result_to_predictor_input',
    # LIS/PACS
    'LISPACSAdapter',
    'LISPACSDecorator',
    'LISMapper',
    'RadiologyNLP',
    # 疾控
    'TBCardFiller',
    'InfectiousDiseaseCard',
    'CDCCallbackParser',
    'CDCSecurity',
    'CDCWebhookServer',
    # 审核工作流
    'ReviewWorkflow',
    'CardStatus',
    'UserRole',
    'ReviewAction',
    'ReviewRecord',
    'WorkflowError',
    # 可靠消息队列
    'ReliableMessageQueue',
    'ReliableMessage',
    'MessageStatus',
    'QueueType',
    'MessageHandlerError',
    # GNN-CDC联动
    'ContactTracingManager',
    'ContactRecord',
    'ScreeningTask',
    # 审计日志
    'AuditLogger',
    'AuditRecord',
    'AuditOperationType',
   # 质控管理
    'QualityControlService',
    'QCAlert',
    'QCAlertType',
    'QCStatistics',
    'DiagnosisRecord',
    'AlertLevel',
    'DiagnosisDataSourceProtocol',
    'ReportCardDataSourceProtocol',
    'NullDiagnosisDataSource',
    'NullReportCardDataSource',
    # 通用REST/SOAP
    'RESTClient',
    'SOAPClient',
    'AuthProvider',
    'NoAuth',
    'APIKeyAuth',
    'OAuth2ClientCredentials',
    'SignatureAuth',
    # 查重
    'DuplicateChecker',
    'DuplicateMatch',
    'CDCQuarantineProtocol',
    'NullCDCQuarantine',
    'check_duplicate_before_submit',
    # 统一入口
    'HealthInteropManager',
    'init_health_interop',
    # ETL 集成管线
    'ETLPipeline',
    'ETLConfig',
    'ETLResult',
    'DataSourceRecord',
    # 特征计算与缺失值填补
    'FeatureCalculator',
    'MissingValueImputer',
    'FillRecord',
    'CalculationResult',
    'compute_bcg_status',
    'compute_smoking_index',
    'compute_bmi',
    'extract_contact_history',
    # 增量引擎
    'IncrementalEngine',
    'EventType',
    'PatientSnapshot',
    'PatientEvent',
    'Notification',
    # 业务数据源标准适配器（适配器 + 配置）
    'DATA_SOURCE_TYPES',
    'SyncStatus',
    'SyncSchedule',
    'DataSourceConfig',
    'SyncRunRecord',
    'BaseDataSourceAdapter',
    'HISAdapter',
    'LISAdapter',
    'PACSAdapter',
    'CDCDataSourceAdapter',
    'AdapterRegistry',
    # 同步调度
    'SyncScheduler',
]


class _SQLiteDBManager:
    """ReviewWorkflow 使用的轻量 SQLite 数据库管理器

    提供 ReviewWorkflow 所需的 execute(sql[, params]) 接口：
    - DDL / 写操作返回空列表
    - 查询返回 dict 行列表（支持 len()、下标访问与迭代）
    """

    def __init__(self, path: str):
        import sqlite3
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._closed = False

    def execute(self, sql: str, params=None):
        if self._closed:
            raise RuntimeError("数据库连接已关闭")
        cur = self._conn.execute(sql, params) if params else self._conn.execute(sql)
        if sql.lstrip().upper().startswith(("SELECT", "PRAGMA")):
            return [dict(row) for row in cur.fetchall()]
        self._conn.commit()
        return []

    def close(self):
        if not self._closed:
            try:
                self._conn.close()
            except Exception as _e:
                _logger.debug("关闭 _SQLiteDBManager 数据库连接失败: %s", _e)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def _new_card_id() -> str:
    """生成唯一报告卡ID（备用）"""
    import uuid
    return "CARD-" + uuid.uuid4().hex[:12].upper()


class HealthInteropManager:
    """医疗互操作统一管理器
    
    自动创建和管理AuditLogger、QualityControlService等组件，
    实现依赖自动注入，使用方无需手动创建和传递auditor参数。
    
    使用方法：
        mgr = HealthInteropManager(data_dir="interop_data")
        workflow = mgr.create_review_workflow(db_path="cards.db")
        contact_tracer = mgr.create_contact_tracer()
        qc = mgr.create_quality_control()
    """
    
    def __init__(self, data_dir: str = "health_interop_data",
                 audit_db_path: str = None):
        import os
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.audit_db_path = audit_db_path or os.path.join(data_dir, "audit.db")
        self._auditor = None
        self._qc_service = None
    
    @property
    def auditor(self) -> 'AuditLogger':
        """获取共享的AuditLogger实例（懒加载）"""
        if self._auditor is None:
            from .audit import AuditLogger
            self._auditor = AuditLogger(self.audit_db_path)
        return self._auditor
    
    def create_review_workflow(self, card_id: str = None, 
                                db_path: str = None) -> 'ReviewWorkflow':
        """创建ReviewWorkflow，自动注入默认auditor

        参数：
            card_id: 报告卡ID（可选，缺省时自动生成唯一ID）
            db_path: SQLite数据库路径（可选，缺省使用 data_dir 下的
                     review_workflow.db；如不希望持久化可显式传 None 并配合
                     传入 db_manager 自行控制）

        返回：
            ReviewWorkflow: 已配置默认auditor与数据库持久化的工作流实例
        """
        import os
        from .workflow import ReviewWorkflow
        wf_db = db_path or os.path.join(self.data_dir, "review_workflow.db")
        wf = ReviewWorkflow(
            card_id=card_id or _new_card_id(),
            db_manager=_SQLiteDBManager(wf_db),
            auditor=self.auditor,
            auto_load=False,
        )
        return wf
    
    def create_contact_tracer(self, gnn_network=None) -> 'ContactTracingManager':
        """创建ContactTracingManager，自动注入默认auditor"""
        from .contact_tracing import ContactTracingManager
        return ContactTracingManager(auditor=self.auditor, gnn_network=gnn_network)
    
    def create_quality_control(self, diagnosis_source=None,
                                card_source=None,
                                late_threshold_hours: int = 24) -> 'QualityControlService':
        """创建QualityControlService，自动注入默认auditor"""
        qc = QualityControlService(
            audit_logger=self.auditor,
            diagnosis_source=diagnosis_source,
            card_source=card_source,
            late_threshold_hours=late_threshold_hours,
        )
        self._qc_service = qc
        return qc
    
    def shutdown(self):
        """优雅关闭所有组件"""
        if self._auditor:
            try:
                self._auditor.flush()
                self._auditor.close()
            except Exception as e:
                _logger.debug("关闭审计器失败: %s", e)


def init_health_interop(data_dir: str = "health_interop_data") -> HealthInteropManager:
    """便捷函数：创建HealthInteropManager实例
    
    使用方法：
        from health_interop import init_health_interop
        mgr = init_health_interop()
    """
    return HealthInteropManager(data_dir=data_dir)
