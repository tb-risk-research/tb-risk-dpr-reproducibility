#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""医疗信息互操作基类、常量与工具函数

定义所有医疗适配器的统一接口（HealthcareAdapter）、配置/结果数据类、
术语系统映射、特征-FHIR资源映射矩阵，以及脱敏、时间解析、年龄计算等通用工具。
"""

from __future__ import annotations

import abc
import datetime
import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("tb_risk.health_interop")


# ============================================================================
# 术语系统常量
# ============================================================================

class TERMINOLOGY_SYSTEMS:
    """术语编码系统 URI / 国标编号"""
    LOINC = "http://loinc.org"
    ICD10 = "http://hl7.org/fhir/sid/icd-10"
    ICD10_CM = "http://hl7.org/fhir/sid/icd-10-cm"
    SNOMED_CT = "http://snomed.info/sct"
    RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
    INTERNAL = "internal"               # 系统内部编码
    GB3304 = "urn:std:gb:3304"          # 中国各民族名称罗马字母拼写代码
    GB_T6565 = "urn:std:gb:t:6565"      # 职业分类与代码
    GB_T2260 = "urn:std:gb:t:2260"      # 中华人民共和国行政区划代码
    CVX = "http://hl7.org/fhir/sid/cvx" # 疫苗编码
    UCUM = "http://unitsofmeasure.org"  # 计量单位


# ============================================================================
# tb_risk 22维特征 → FHIR资源映射矩阵
# ============================================================================

FHIR_RESOURCE_MAPPING: Dict[str, Dict[str, Any]] = {
    # 患者基本信息
    "age": {"resource": "Patient", "path": "birthDate", "searchParam": "birthdate",
            "extractor": "extract_age_from_birthdate"},
    "gender": {"resource": "Patient", "path": "gender", "searchParam": "gender"},
    "ethnicity": {"resource": "Patient", "path": "extension[ethnicity]", "searchParam": None,
                  "terminology": TERMINOLOGY_SYSTEMS.GB3304},
    "occupation": {"resource": "Patient", "path": "extension[occupation]", "searchParam": None,
                   "terminology": TERMINOLOGY_SYSTEMS.GB_T6565},
    "district": {"resource": "Patient", "path": "address.district", "searchParam": "address",
                 "terminology": TERMINOLOGY_SYSTEMS.GB_T2260},
    # 症状与体征
    "cough": {"resource": "Condition|Observation", "path": "code", "searchParam": "code",
              "note": "咳嗽症状"},
    "fever": {"resource": "Condition|Observation", "path": "code", "searchParam": "code"},
    "hemoptysis": {"resource": "Condition|Observation", "path": "code", "searchParam": "code"},
    "weight_loss": {"resource": "Condition|Observation", "path": "code", "searchParam": "code"},
    "night_sweat": {"resource": "Condition|Observation", "path": "code", "searchParam": "code"},
    # 既往病史 / 合并症
    "diabetes": {"resource": "Condition", "path": "code", "searchParam": "code",
                 "icd10_prefixes": ["E10", "E11", "E12", "E13", "E14"]},
    "hiv": {"resource": "Condition", "path": "code", "searchParam": "code",
            "icd10_prefixes": ["B20", "B21", "B22", "B23", "B24", "Z21"]},
    "silicosis": {"resource": "Condition", "path": "code", "searchParam": "code",
                  "icd10_prefixes": ["J62", "J65"]},
    "history_tb": {"resource": "Condition", "path": "code", "searchParam": "code",
                   "icd10_prefixes": ["A15", "A16", "A17", "A18", "A19", "B90"]},
    # 检验结果
    "sputum_smear": {"resource": "Observation", "path": "value", "searchParam": "code",
                     "loinc_codes": ["640-4", "14473-1", "640-4"]},
    "sputum_culture": {"resource": "Observation", "path": "value", "searchParam": "code",
                       "loinc_codes": ["640-4", "11477-7"]},
    "xpert_mtb_rif": {"resource": "Observation", "path": "value", "searchParam": "code",
                      "loinc_codes": ["94504-3", "94505-0"]},
    "t_spot": {"resource": "Observation", "path": "value", "searchParam": "code",
               "loinc_codes": ["80403-4", "80404-2"]},
    "esr": {"resource": "Observation", "path": "valueQuantity", "searchParam": "code",
            "loinc_codes": ["4537-7", "30341-3"]},
    "crp": {"resource": "Observation", "path": "valueQuantity", "searchParam": "code",
            "loinc_codes": ["1988-5", "76485-7"]},
    "blood_glucose": {"resource": "Observation", "path": "valueQuantity", "searchParam": "code",
                      "loinc_codes": ["2345-7", "2339-0", "1558-6"]},
    # 影像
    "chest_xray": {"resource": "DiagnosticReport|ImagingStudy", "path": "conclusion",
                   "searchParam": "code", "nlp_required": True},
    "has_cavity": {"resource": "DiagnosticReport", "path": "conclusion", "searchParam": "code",
                   "nlp_required": True, "keywords": ["空洞", "cavity", "cavitation"]},
    # 用药
    "anti_tb_treatment": {"resource": "MedicationStatement", "path": "medicationCodeableConcept",
                          "searchParam": "code", "rxnorm_tbd": True},
}


# ============================================================================
# 结核相关LOINC检验项目面板（五大类）
# ============================================================================

LOINC_TB_PANELS: Dict[str, List[Dict[str, Any]]] = {
    "pathogen": [
        {"loinc": "640-4", "name": "痰涂片抗酸染色", "short": "涂片", "sample": "sputum"},
        {"loinc": "11477-7", "name": "痰结核分枝杆菌培养", "short": "培养", "sample": "sputum"},
        {"loinc": "14473-1", "name": "BACTEC液体培养", "short": "BACTEC", "sample": "sputum"},
        {"loinc": "43405-4", "name": "抗酸杆菌荧光染色", "short": "荧光涂片", "sample": "sputum"},
    ],
    "molecular": [
        {"loinc": "94504-3", "name": "Xpert MTB/RIF 结核分枝杆菌", "short": "Xpert_MTB", "sample": "sputum"},
        {"loinc": "94505-0", "name": "Xpert MTB/RIF 利福平耐药", "short": "Xpert_RIF", "sample": "sputum"},
        {"loinc": "94419-4", "name": "熔解曲线法结核耐药", "short": "熔解曲线", "sample": "sputum"},
        {"loinc": "82349-7", "name": "结核分枝杆菌测序", "short": "测序", "sample": "sputum"},
    ],
    "serology": [
        {"loinc": "80403-4", "name": "结核感染T细胞检测(T-SPOT)", "short": "T-SPOT", "sample": "blood"},
        {"loinc": "80404-2", "name": "γ干扰素释放试验(IGRA)", "short": "IGRA", "sample": "blood"},
        {"loinc": "22442-7", "name": "结核抗体", "short": "TB-Ab", "sample": "blood"},
        {"loinc": "40563-3", "name": "结核菌素皮试(PPD)", "short": "PPD", "sample": "skin"},
    ],
    "routine": [
        {"loinc": "4537-7", "name": "血沉(ESR)", "short": "ESR", "sample": "blood", "unit": "mm/h"},
        {"loinc": "1988-5", "name": "C反应蛋白(CRP)", "short": "CRP", "sample": "blood", "unit": "mg/L"},
        {"loinc": "2345-7", "name": "空腹血糖", "short": "GLU", "sample": "blood", "unit": "mmol/L"},
        {"loinc": "2339-0", "name": "随机血糖", "short": "GLU_random", "sample": "blood", "unit": "mmol/L"},
        {"loinc": "6690-2", "name": "白细胞计数(WBC)", "short": "WBC", "sample": "blood", "unit": "10^9/L"},
        {"loinc": "718-7", "name": "血红蛋白(HGB)", "short": "HGB", "sample": "blood", "unit": "g/L"},
        {"loinc": "2823-3", "name": "谷丙转氨酶(ALT)", "short": "ALT", "sample": "blood", "unit": "U/L"},
        {"loinc": "1920-8", "name": "谷草转氨酶(AST)", "short": "AST", "sample": "blood", "unit": "U/L"},
        {"loinc": "2160-0", "name": "肌酐(Cr)", "short": "Cr", "sample": "blood", "unit": "umol/L"},
    ],
    "infectious_screen": [
        {"loinc": "21470-1", "name": "HIV1/2抗体", "short": "HIV", "sample": "blood"},
        {"loinc": "16935-9", "name": "乙肝表面抗原(HBsAg)", "short": "HBsAg", "sample": "blood"},
        {"loinc": "22310-9", "name": "丙肝抗体(HCV-Ab)", "short": "HCV", "sample": "blood"},
        {"loinc": "31047-5", "name": "梅毒抗体(TP-Ab)", "short": "Syphilis", "sample": "blood"},
    ],
}


# ============================================================================
# ICD-10 结核相关诊断编码范围
# ============================================================================

ICD10_TB_CODES = {
    "A15": "肺结核，经细菌学和组织学证实",
    "A16": "肺结核，未经细菌学或组织学证实",
    "A17": "神经系统结核",
    "A18": "其他器官结核",
    "A19": "粟粒性结核",
    "B90": "结核病后遗症",
    "E10-E14": "糖尿病（合并症）",
    "B20-B24": "HIV病（合并症）",
    "Z21": "无症状HIV感染状态",
    "J62": "矽肺（合并症）",
    "J65": "与结核有关的尘肺",
}


# ============================================================================
# 性别映射（不同系统的性别编码 → 内部统一表示）
# ============================================================================

GENDER_MAPPING: Dict[str, str] = {
    # 中文
    "男": "male", "女": "female", "未知": "unknown", "未说明": "unknown",
    # FHIR
    "male": "male", "female": "female", "unknown": "unknown", "other": "other",
    # HL7 v2
    "M": "male", "F": "female", "O": "other", "U": "unknown", "A": "other",
    "m": "male", "f": "female",
    # 数字编码
    "1": "male", "2": "female", "0": "unknown", "9": "unknown",
}


# ============================================================================
# 异常标志映射
# ============================================================================

ABNORMAL_FLAGS = {
    # HL7 v2 OBX-8
    "L": "low", "H": "high", "LL": "critical_low", "HH": "critical_high",
    "<": "low", ">": "high", "A": "abnormal", "AA": "critical_abnormal",
    "N": "normal", "": "unknown",
    # FHIR Observation.interpretation
    "low": "low", "high": "high", "normal": "normal", "abnormal": "abnormal",
}


# ============================================================================
# 数据类定义
# ============================================================================

# TODO: [EMPI-001] 患者主索引匹配功能未实现
#  - 需支持跨系统患者ID匹配（门诊号/住院号/身份证号/医保号）
#  - 需实现重名识别、ID合并、相似度评分算法
#  - 需建立EMPI索引表，支持同一患者多系统数据聚合
@dataclass
class PatientIdentifier:
    """患者标识（EMPI统一ID）"""
    id_type: str = ""           # 标识类型：empi / patient_id / id_card / visit_id / mrn
    value: str = ""
    assigning_authority: str = ""  # 颁发机构（医院OID/名称）

    def __hash__(self):
        return hash((self.id_type, self.value, self.assigning_authority))


@dataclass
class AdapterConfig:
    """适配器通用配置"""
    # 连接
    base_url: str = ""
    port: int = 0
    username: str = ""
    password: str = ""          # 加密存储
    api_key: str = ""
    token: str = ""
    auth_type: str = "none"     # none / basic / oauth2 / api_key / signature

    # 超时与重试
    connect_timeout: int = 5
    read_timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 1.5  # 指数退避基数（秒）

    # 数据拉取
    history_years: int = 5      # 首次拉取历史数据年数
    poll_interval: int = 300    # 轮询间隔（秒）
    incremental: bool = True

    # 安全
    ssl_verify: bool = True
    ip_whitelist: List[str] = field(default_factory=list)
    encrypt_credentials: bool = True

    # 厂商特定参数
    vendor_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterResult:
    """适配器执行结果（统一输出契约）"""
    success: bool = False
    patient_info: Dict[str, Any] = field(default_factory=dict)
    family_contacts: List[Dict[str, Any]] = field(default_factory=list)
    social_contacts: List[Dict[str, Any]] = field(default_factory=list)
    lab_results: List[Dict[str, Any]] = field(default_factory=list)
    imaging_reports: List[Dict[str, Any]] = field(default_factory=list)
    diagnoses: List[Dict[str, Any]] = field(default_factory=list)
    medications: List[Dict[str, Any]] = field(default_factory=list)
    clinical_notes: List[Dict[str, Any]] = field(default_factory=list)

    # 增强特征（LIS/PACS装饰器等填充）
    risk_features: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    source_system: str = ""
    patient_id: str = ""
    last_sync_time: Optional[datetime.datetime] = None
    records_processed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unmapped_codes: List[Dict[str, str]] = field(default_factory=list)

    def to_patient_record(self):
        """转换为 schemas.PatientRecord"""
        from ..schemas import PatientRecord
        return PatientRecord.from_dict({
            "basic_info": self.patient_info,
            "meta": {
                "source_system": self.source_system,
                "patient_id": self.patient_id,
                "last_sync_time": self.last_sync_time.isoformat()
                    if self.last_sync_time else None,
            }
        })


@dataclass
class DataQualityIssue:
    """数据质量问题记录"""
    severity: str = "warning"   # error / warning / info
    field: str = ""
    value: Any = None
    message: str = ""
    record_id: str = ""


# ============================================================================
# 适配器抽象基类
# ============================================================================

class HealthcareAdapter(abc.ABC):
    """医疗信息适配器抽象基类

    所有外部系统接入都必须实现此接口，输出统一的 AdapterResult。
    """

    ADAPTER_TYPE: str = "generic"

    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self._logger = logging.getLogger(
            f"tb_risk.health_interop.{self.ADAPTER_TYPE}")
        self._monitor = None  # 由调用方注入 InterfaceMonitor
        self._last_sync_time: Optional[datetime.datetime] = None

    def set_monitor(self, monitor):
        """注入接口监控器"""
        self._monitor = monitor

    @abc.abstractmethod
    def connect(self) -> bool:
        """建立连接 / 认证，返回是否成功"""
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_patient(self, patient_id: str) -> AdapterResult:
        """按需查询单个患者数据（医生打开患者页面时调用）

        参数：
            patient_id: 患者ID（可以是EMPI ID、医院MRN等，子类自行解析）

        返回：
            AdapterResult
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_incremental(self, since: Optional[datetime.datetime] = None
                          ) -> List[AdapterResult]:
        """增量拉取（轮询或订阅推送触发时调用）

        参数：
            since: 拉取此时间之后的新数据；None则按配置拉取首次历史

        返回：
            多个患者的 AdapterResult 列表
        """
        raise NotImplementedError

    def disconnect(self):
        """断开连接 / 清理资源（默认空实现，子类可覆盖）"""
        pass

    def health_check(self) -> Tuple[bool, str]:
        """健康检查（用于监控），返回 (是否可用, 消息)"""
        return True, "OK"

    # ---- 通用辅助方法，子类可直接使用 ----

    def _record_success(self, endpoint: str, duration_ms: float, data_count: int = 0):
        """记录成功指标"""
        if self._monitor:
            self._monitor.record_success(
                self.ADAPTER_TYPE, endpoint, duration_ms, data_count)

    def _record_failure(self, endpoint: str, error: str, duration_ms: float = 0):
        """记录失败指标"""
        if self._monitor:
            self._monitor.record_failure(
                self.ADAPTER_TYPE, endpoint, error, duration_ms)

    def _log_unmapped_code(self, system: str, code: str, display: str = ""):
        """记录未映射的编码（供人工补充）"""
        self._logger.warning("未映射编码: system=%s code=%s display=%s",
                            system, code, display)


# ============================================================================
# 工具函数
# ============================================================================

def timestamp_to_datetime(ts: Any) -> Optional[datetime.datetime]:
    """将各种时间格式转换为 datetime（不抛异常，失败返回None）"""
    if ts is None or ts == "":
        return None
    # 已经是 datetime
    if isinstance(ts, datetime.datetime):
        return ts
    if isinstance(ts, datetime.date):
        return datetime.datetime.combine(ts, datetime.time.min)
    # Unix 时间戳（秒/毫秒）
    if isinstance(ts, (int, float)):
        if not ts > 0:
            # 非正时间戳无效（0/负值/NaN）。POSIX 上 fromtimestamp(-1)
            # 会"成功"返回 1969-12-31（Windows 才抛 OSError），
            # 必须在比较层显式拒绝以保证跨平台一致
            return None
        try:
            if ts > 1e12:  # 毫秒
                return datetime.datetime.fromtimestamp(ts / 1000.0)
            return datetime.datetime.fromtimestamp(ts)
        except (OverflowError, OSError, ValueError):
            return None
    # 字符串解析
    return safe_parse_datetime(str(ts))


_DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%Y%m%d%H%M%S",
    "%Y%m%d",
]


def safe_parse_datetime(s: str) -> Optional[datetime.datetime]:
    """安全解析日期时间字符串，支持多种常见格式"""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    # 先尝试 fromisoformat（Python 3.7+）
    try:
        # 处理 Z 结尾
        normalized = s.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        pass
    # 逐一尝试常见格式
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def calculate_age(birth_date: Any,
                  reference_date: Optional[datetime.datetime] = None) -> Optional[int]:
    """根据出生日期计算年龄（周岁）

    参数：
        birth_date: 出生日期（datetime/date/字符串均可）
        reference_date: 参考日期，默认今天
    """
    bd = timestamp_to_datetime(birth_date)
    if bd is None:
        return None
    ref = reference_date or datetime.datetime.now()
    try:
        age = ref.year - bd.year
        if (ref.month, ref.day) < (bd.month, bd.day):
            age -= 1
        if age < 0 or age > 120:
            return None  # 异常值
        return age
    except Exception:
        return None


def mask_sensitive(value: str, mask_char: str = "*",
                   keep_prefix: int = 1, keep_suffix: int = 1) -> str:
    """脱敏敏感信息（身份证号、手机号等）

    示例：mask_sensitive("110101199001011234") → "1101**********1234"
    """
    if value is None:
        return ""
    s = str(value)
    if len(s) <= keep_prefix + keep_suffix:
        return mask_char * len(s)
    return (s[:keep_prefix]
            + mask_char * (len(s) - keep_prefix - keep_suffix)
            + s[-keep_suffix:] if keep_suffix > 0 else "")


def hash_identifier(value: str, salt: str = "tb_risk_empi") -> str:
    """对患者标识做不可逆哈希（用于跨机构数据交换脱敏）"""
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()[:16]


def normalize_gender(raw: Any) -> str:
    """将各种性别表示统一为 'male'/'female'/'other'/'unknown'"""
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    return GENDER_MAPPING.get(s, GENDER_MAPPING.get(s.lower(), "unknown"))


def convert_unit(value: float, from_unit: str, to_unit: str) -> Optional[float]:
    """简单的单位换算（检验结果单位统一）

    覆盖常见单位：
    - 血糖：mg/dL ↔ mmol/L（换算系数 18.016）
    - 长度/体积：mm ↔ cm；mL ↔ L
    - 温度：℃ ↔ ℉
    """
    if from_unit == to_unit or not from_unit or not to_unit:
        return value

    conversions = {
        ("mg/dl", "mmol/l"): lambda v: v / 18.016,
        ("mmol/l", "mg/dl"): lambda v: v * 18.016,
        ("mg/l", "mg/dl"): lambda v: v / 10.0,
        ("mg/dl", "mg/l"): lambda v: v * 10.0,
        ("mm", "cm"): lambda v: v / 10.0,
        ("cm", "mm"): lambda v: v * 10.0,
        ("ml", "l"): lambda v: v / 1000.0,
        ("l", "ml"): lambda v: v * 1000.0,
        ("c", "f"): lambda v: v * 9.0 / 5.0 + 32.0,
        ("f", "c"): lambda v: (v - 32.0) * 5.0 / 9.0,
    }

    key = (from_unit.lower(), to_unit.lower())
    fn = conversions.get(key)
    if fn:
        try:
            return round(fn(float(value)), 4)
        except (TypeError, ValueError):
            return None
    return None


def parse_hl7_datetime(hl7_dt: str) -> Optional[datetime.datetime]:
    """解析HL7 v2日期时间格式（YYYYMMDDHHMMSS.SSSS[+-ZZZZ]）"""
    if not hl7_dt or not isinstance(hl7_dt, str):
        return None
    s = hl7_dt.strip()
    # 移除时区部分先解析基本时间
    tz_match = re.match(r"^(\d{4,14})(?:\.(\d+))?([+-]\d{4})?$", s)
    if not tz_match:
        return safe_parse_datetime(s)
    base = tz_match.group(1)
    frac = tz_match.group(2) or ""
    # 补齐到 14 位
    while len(base) < 14:
        base += "0" if len(base) in (4, 6) else "0"
        if len(base) > 14:
            break
    try:
        dt = datetime.datetime.strptime(base[:14], "%Y%m%d%H%M%S")
        if frac:
            micro = int(frac.ljust(6, "0")[:6])
            dt = dt.replace(microsecond=micro)
        return dt
    except ValueError:
        return safe_parse_datetime(s)
