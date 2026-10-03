#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""智能特征计算与缺失值处理引擎

很多特征不能直接从医院系统拿到原始值，需要根据原始数据自动计算：
1. BCG接种史：综合接种记录 + 卡痕检查结果
2. 吸烟指数（包年）：每日吸烟量 × 吸烟年数 ÷ 20
3. BMI：体重(kg) ÷ 身高(m)²
4. 密切接触史：从病历文本和传染病报告卡综合提取
5. 缺失值填补：基于医院数据分布的本土化填补策略，记录缺失率和填补方式
"""

import datetime
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("tb_risk.health_interop.feature_calculator")

# ============================================================================
# 缺失值填补策略常量
# ============================================================================

# 结核病高危人群医疗数据典型分布（基于中国结核病流行病学调查数据）
# 用于缺失值填补时的先验分布
DEFAULT_DISTRIBUTION: Dict[str, Dict[str, Any]] = {
    "age": {
        "mean": 42.5,
        "median": 43.0,
        "std": 18.2,
        "min": 0,
        "max": 110,
        "distribution": "normal",
        "description": "年龄（岁），基于中国结核病登记患者年龄分布",
    },
    "bmi": {
        "mean": 21.5,
        "median": 21.3,
        "std": 3.8,
        "min": 12,
        "max": 45,
        "distribution": "normal",
        "description": "BMI指数，结核病患者常偏低",
    },
    "smoking_years": {
        "mean": 15.0,
        "median": 10.0,
        "std": 12.0,
        "min": 0,
        "max": 70,
        "distribution": "right_skewed",
        "description": "吸烟年数",
    },
    "smoking_amount_per_day": {
        "mean": 15.0,
        "median": 10.0,
        "std": 10.0,
        "min": 0,
        "max": 60,
        "distribution": "right_skewed",
        "description": "每日吸烟量（支）",
    },
    "pack_years": {
        "mean": 15.0,
        "median": 10.0,
        "std": 15.0,
        "min": 0,
        "max": 120,
        "distribution": "right_skewed",
        "description": "吸烟指数（包年）",
    },
    "drinking_years": {
        "mean": 10.0,
        "median": 5.0,
        "std": 10.0,
        "min": 0,
        "max": 50,
        "distribution": "right_skewed",
        "description": "饮酒年数",
    },
    "height_cm": {
        "mean": 165.0,
        "median": 165.0,
        "std": 10.0,
        "min": 100,
        "max": 220,
        "distribution": "normal",
        "description": "身高（cm）",
    },
    "weight_kg": {
        "mean": 60.0,
        "median": 58.0,
        "std": 15.0,
        "min": 20,
        "max": 150,
        "distribution": "normal",
        "description": "体重（kg）",
    },
    "symptom_count": {
        "mean": 2.5,
        "median": 2.0,
        "std": 2.0,
        "min": 0,
        "max": 10,
        "distribution": "right_skewed",
        "description": "症状数量",
    },
    "delay_days": {
        "mean": 30.0,
        "median": 21.0,
        "std": 35.0,
        "min": 0,
        "max": 365,
        "distribution": "right_skewed",
        "description": "延迟就诊天数",
    },
    "contact_duration_months": {
        "mean": 6.0,
        "median": 3.0,
        "std": 8.0,
        "min": 0,
        "max": 60,
        "distribution": "right_skewed",
        "description": "密切接触时长（月）",
    },
}

# 缺失值填补策略
FILL_STRATEGIES = {
    "mean": "均值填补",
    "median": "中位数填补",
    "mode": "众数填补",
    "regression": "回归填补（基于其他相关字段）",
    "hotdeck": "热卡填补（同类患者相似值）",
    "default": "医学默认值填补",
    "flag": "标记为缺失，不填补",
    "ml": "机器学习模型预测填补",
}


@dataclass
class FillRecord:
    """缺失值填补记录"""
    field: str
    original_value: Any
    filled_value: Any
    strategy: str
    reason: str
    confidence: float = 0.5
    timestamp: str = ""


@dataclass
class CalculationResult:
    """特征计算结果"""
    features: Dict[str, Any] = field(default_factory=dict)
    fill_records: List[FillRecord] = field(default_factory=list)
    missing_rate: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    source_fields_used: Dict[str, List[str]] = field(default_factory=dict)


# ============================================================================
# BCG接种史判断
# ============================================================================

BCG_SCAR_KEYWORDS = ["有卡痕", "卡痕", "bcg scar", "scar", "卡介苗痕迹"]
BCG_VACCINE_KEYWORDS = ["卡介苗", "bcg", "卡介苗接种", "bcg vaccine", "bcg vaccination"]


def compute_bcg_status(
    vaccination_records: Optional[List[Dict[str, Any]]] = None,
    scar_check_result: Optional[str] = None,
    clinical_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """综合评估BCG接种史

    BCG接种史不能只看接种记录，还要结合卡痕检查结果。
    判断逻辑：
    1. 有卡痕 → 接种（高置信度）
    2. 有接种记录 → 接种（中高置信度）
    3. 无卡痕 + 无接种记录 → 未接种（中置信度）
    4. 无卡痕但有接种记录 → 可能接种（需结合年龄判断）
    5. 有卡痕但无接种记录 → 接种（卡痕为金标准）

    参数：
        vaccination_records: 接种记录列表 [{vaccine, date, ...}]
        scar_check_result: 卡痕检查结果（"有卡痕"/"无卡痕"/None）
        clinical_notes: 病历文本列表

    返回：
        dict: {
            "bcg_scar": "1" | "2",  # 1=有/接种, 2=无/未知
            "confidence": float,    # 置信度 0-1
            "evidence": [str],      # 证据列表
            "method": str,          # 判断方法
        }
    """
    evidence = []
    has_vaccination_record = False
    has_scar = False
    confidence = 0.0
    method = "unknown"

    # 1. 检查卡痕结果（金标准）
    if scar_check_result:
        scar_text = str(scar_check_result).strip().lower()
        for kw in BCG_SCAR_KEYWORDS:
            if kw in scar_text:
                has_scar = True
                evidence.append(f"卡痕检查: {scar_check_result}")
                break

    # 2. 检查接种记录
    if vaccination_records:
        for record in vaccination_records:
            vaccine_name = str(record.get("vaccine", record.get("name", ""))).lower()
            for kw in BCG_VACCINE_KEYWORDS:
                if kw in vaccine_name:
                    has_vaccination_record = True
                    evidence.append(f"接种记录: {record.get('vaccine', record.get('name', ''))}")
                    break

    # 3. 从病历文本中提取
    if clinical_notes:
        for note in clinical_notes:
            note_text = str(note).lower()
            for kw in BCG_SCAR_KEYWORDS + BCG_VACCINE_KEYWORDS:
                if kw in note_text:
                    evidence.append(f"病历记录: {note[:100]}")
                    if kw in BCG_SCAR_KEYWORDS:
                        has_scar = True
                    if kw in BCG_VACCINE_KEYWORDS:
                        has_vaccination_record = True
                    break

    # 4. 综合判断
    if has_scar:
        # 卡痕为金标准
        result = "1"  # 接种
        confidence = 0.95
        method = "scar_check"
        if has_vaccination_record:
            confidence = 0.99
            method = "scar_and_record"
    elif has_vaccination_record:
        result = "1"  # 接种
        confidence = 0.80
        method = "vaccination_record"
    else:
        # 无任何证据，保守假设为未接种
        result = "2"  # 未接种/未知
        confidence = 0.50
        method = "no_evidence"

    return {
        "bcg_scar": result,
        "confidence": confidence,
        "evidence": evidence,
        "method": method,
    }


# ============================================================================
# 吸烟指数计算
# ============================================================================


def compute_smoking_index(
    amount_per_day: Optional[float] = None,
    smoking_years: Optional[float] = None,
    amount_per_week: Optional[float] = None,
    unit: str = "cigarettes",
) -> Dict[str, Any]:
    """计算吸烟指数（包年）

    公式：
    - 包年 = 每日吸烟量(支) × 吸烟年数 ÷ 20
    - 或：包年 = 每周吸烟量(支) × 吸烟年数 ÷ 20 ÷ 52

    参数：
        amount_per_day: 每日吸烟量（支）
        smoking_years: 吸烟年数
        amount_per_week: 每周吸烟量（支）
        unit: 单位（cigarettes/支）

    返回：
        dict: {
            "pack_years": float,        # 包年
            "daily_amount": float,      # 每日吸烟量（支）
            "smoking_years": float,     # 吸烟年数
            "confidence": str,          # 置信度
            "calculation_method": str,  # 计算方法
        }
    """
    daily = None
    years = smoking_years
    confidence = "low"
    method = ""

    # 优先使用每日吸烟量
    if amount_per_day is not None and amount_per_day > 0:
        daily = float(amount_per_day)
        method = "daily_amount"
        confidence = "high"
    elif amount_per_week is not None and amount_per_week > 0:
        daily = float(amount_per_week) / 7.0
        method = "weekly_amount"
        confidence = "medium"

    if daily is None or daily <= 0:
        daily = 0
        years = 0
        confidence = "no_data"
        method = "no_data"

    if years is None or years <= 0:
        years = 0
        if daily > 0:
            # 有吸烟量但无年数，保守假设为1年
            years = 1
            confidence = "low"
            method = "amount_only"

    pack_years = daily * years / 20.0

    return {
        "pack_years": round(pack_years, 2),
        "daily_amount": round(daily, 1),
        "smoking_years": round(years, 1),
        "confidence": confidence,
        "calculation_method": method,
    }


# ============================================================================
# BMI计算
# ============================================================================


def compute_bmi(
    height_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
    height_m: Optional[float] = None,
    weight_jin: Optional[float] = None,
) -> Dict[str, Any]:
    """计算BMI指数

    公式：BMI = 体重(kg) / 身高(m)²

    参数：
        height_cm: 身高（cm）
        weight_kg: 体重（kg）
        height_m: 身高（m，替代参数）
        weight_jin: 体重（斤，替代参数）

    返回：
        dict: {
            "bmi": float,
            "height_cm": float,
            "weight_kg": float,
            "category": str,       # 分类
            "confidence": str,
        }
    """
    # 身高统一转为cm
    h = height_cm
    if h is None and height_m is not None:
        h = float(height_m) * 100

    # 体重统一转为kg
    w = weight_kg
    if w is None and weight_jin is not None:
        w = float(weight_jin) / 2.0

    confidence = "high"
    if h is None or w is None:
        return {
            "bmi": None,
            "height_cm": h,
            "weight_kg": w,
            "category": "unknown",
            "confidence": "no_data",
        }

    try:
        h = float(h)
        w = float(w)
    except (TypeError, ValueError):
        return {
            "bmi": None,
            "height_cm": h,
            "weight_kg": w,
            "category": "unknown",
            "confidence": "error",
        }

    if h <= 0 or w <= 0:
        return {
            "bmi": None,
            "height_cm": h,
            "weight_kg": w,
            "category": "invalid",
            "confidence": "error",
        }

    bmi = w / ((h / 100.0) ** 2)
    bmi = round(bmi, 1)

    # 分类
    if bmi < 18.5:
        category = "underweight"
    elif bmi < 24.0:
        category = "normal"
    elif bmi < 28.0:
        category = "overweight"
    else:
        category = "obese"

    return {
        "bmi": bmi,
        "height_cm": h,
        "weight_kg": w,
        "category": category,
        "confidence": confidence,
    }


# ============================================================================
# 密切接触史提取
# ============================================================================

# 接触类型关键字
CONTACT_TYPE_KEYWORDS = {
    "household_home": [
        "同住", "家人", "家庭", "夫妻", "配偶", "父母", "子女",
        "同床", "同居", "household", "family", "spouse", "parent",
    ],
    "household_out": [
        "同院", "同病房", "同病区", "陪护", "护理",
        "wardmate", "caregiver", "roommate",
    ],
    "social": [
        "同事", "同学", "朋友", "邻居", "同办公室", "同教室",
        "colleague", "classmate", "friend", "neighbor",
    ],
}

# 接触时长关键字
CONTACT_DURATION_PATTERNS = [
    (r"(\d+)\s*年", 12.0),      # N年 → N*12月
    (r"(\d+)\s*个月?", 1.0),     # N月 → N月
    (r"(\d+)\s*天", 1.0 / 30.0),  # N天 → N/30月
    (r"(\d+)\s*周", 0.25),       # N周 → N*0.25月
    (r"长期", 12.0),             # 长期 → 12月（默认）
    (r"短期", 1.0),              # 短期 → 1月
]


def extract_contact_history(
    contact_records: Optional[List[Dict[str, Any]]] = None,
    clinical_notes: Optional[List[str]] = None,
    report_card: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """综合提取密切接触史

    从病历文本、传染病报告卡、接触者记录中综合提取密切接触史。

    参数：
        contact_records: 接触者记录列表 [{name, relation, duration, ...}]
        clinical_notes: 病历文本列表
        report_card: 传染病报告卡数据

    返回：
        dict: {
            "has_contact": bool,           # 是否有密切接触
            "contact_type": str,           # 接触类型
            "contact_duration_months": float,  # 接触时长（月）
            "contact_with_tb": "1" | "2",  # 是否接触涂阳患者
            "contact_details": [str],      # 详细描述
            "confidence": str,             # 置信度
            "source": str,                 # 数据来源
        }
    """
    has_contact = False
    contact_type = "none"
    max_duration = 0.0
    contact_with_tb = "2"  # 默认否
    details = []
    source = "none"

    # 1. 从接触者记录中提取
    if contact_records:
        for record in contact_records:
            relation = str(record.get("relation", record.get("relationship", ""))).lower()
            duration = record.get("duration_months", record.get("duration", 0))
            tb_status = record.get("tb_status", record.get("sputum_smear", ""))

            # 判断接触类型
            for ctype, keywords in CONTACT_TYPE_KEYWORDS.items():
                for kw in keywords:
                    if kw in relation:
                        contact_type = ctype
                        break

            # 判断是否涂阳患者
            tb_str = str(tb_status).lower()
            if "阳" in tb_str or "positive" in tb_str or "涂阳" in tb_str:
                contact_with_tb = "1"

            # 持续时间
            try:
                d = float(duration)
                if d > max_duration:
                    max_duration = d
            except (TypeError, ValueError):
                pass

            has_contact = True
            details.append(
                f"接触者: {record.get('name', '未知')}, "
                f"关系: {relation}, "
                f"时长: {duration}月, "
                f"TB状态: {tb_status}"
            )
            source = "contact_record"

    # 2. 从病历文本中提取
    if clinical_notes:
        for note in clinical_notes:
            note_text = str(note)
            # 检查接触相关关键字
            for ctype, keywords in CONTACT_TYPE_KEYWORDS.items():
                for kw in keywords:
                    if kw in note_text.lower():
                        has_contact = True
                        if contact_type == "none":
                            contact_type = ctype
                        details.append(f"病历描述: {note_text[:200]}")
                        source = "clinical_note"
                        break

            # 提取接触时长
            for pattern, multiplier in CONTACT_DURATION_PATTERNS:
                match = re.search(pattern, note_text)
                if match:
                    try:
                        d = float(match.group(1)) * multiplier
                        if d > max_duration:
                            max_duration = d
                    except (ValueError, IndexError):
                        pass

            # 提取涂阳患者接触
            if re.search(r"涂阳|阳性.*接触|接触.*阳性|接触.*结核|结核.*接触", note_text):
                contact_with_tb = "1"
                has_contact = True

    # 3. 从传染病报告卡中提取
    if report_card:
        contact_history = report_card.get("contact_history", report_card.get("接触史", ""))
        if contact_history:
            contact_str = str(contact_history)
            for ctype, keywords in CONTACT_TYPE_KEYWORDS.items():
                for kw in keywords:
                    if kw in contact_str:
                        has_contact = True
                        if contact_type == "none":
                            contact_type = ctype
                        details.append(f"报告卡接触史: {contact_str[:200]}")
                        source = "report_card"
                        break
            if "阳" in contact_str or "positive" in contact_str:
                contact_with_tb = "1"

    confidence = "high" if source != "none" else "low"

    return {
        "has_contact": has_contact,
        "contact_type": contact_type if has_contact else "none",
        "contact_duration_months": max_duration,
        "contact_with_tb": contact_with_tb if has_contact else "2",
        "contact_details": details,
        "confidence": confidence,
        "source": source,
    }


# ============================================================================
# 缺失值填补引擎
# ============================================================================


class MissingValueImputer:
    """缺失值填补引擎

    基于医院数据分布的本土化缺失值填补策略：
    1. 优先使用医院自身数据分布的统计量
    2. 无医院数据时使用全国结核病先验分布
    3. 支持性别/年龄分层填补
    4. 记录缺失率和填补方式
    5. 可导出填补报告用于审计

    注意：fill_missing() 返回 (filled_data, fill_records) 元组，
    而非依赖内部可变状态，确保多次调用间统计一致性。
    """

    MAX_FILL_RECORDS = 10000  # 最大记录数，防止无界增长

    def __init__(self, hospital_distribution: Optional[Dict[str, Dict[str, Any]]] = None):
        self._hospital_dist = hospital_distribution or {}
        self._fill_records: List[FillRecord] = []
        self._field_stats: Dict[str, Dict[str, Any]] = {}
        self._missing_count: Dict[str, int] = {}
        self._total_count: Dict[str, int] = {}
        self._accumulated_records: List[FillRecord] = []  # 持续累积的填补记录（用于报告）

    def set_hospital_distribution(self, field: str, mean: float, std: float,
                                  median: float, min_val: float, max_val: float):
        """设置医院特定字段的数据分布"""
        self._hospital_dist[field] = {
            "mean": mean,
            "std": std,
            "median": median,
            "min": min_val,
            "max": max_val,
            "distribution": "normal",
        }

    def set_hospital_distributions(self, distributions: Dict[str, Dict[str, Any]]):
        """批量设置医院数据分布"""
        self._hospital_dist.update(distributions)

    def fill_missing(self, data: Dict[str, Any],
                     strategy: str = "auto",
                     record_fill: bool = True) -> Tuple[Dict[str, Any], List[FillRecord]]:
        """填补缺失值

        参数：
            data: 待填补的数据字典
            strategy: 填补策略
                "auto": 自动选择最佳策略
                "mean"/"median"/"default"/"flag"
            record_fill: 是否记录填补信息

        返回：
            tuple: (filled_data, fill_records)
                - filled_data: 填补后的数据
                - fill_records: 本次填补记录列表
        """
        result = dict(data)
        fill_records: List[FillRecord] = []

        for field, default_info in DEFAULT_DISTRIBUTION.items():
            value = result.get(field)
            is_missing = (value is None or value == "" or
                          (isinstance(value, float) and math.isnan(value)))

            # 统计缺失（持续累积，与 fill_records 分离）
            self._total_count[field] = self._total_count.get(field, 0) + 1
            if is_missing:
                self._missing_count[field] = self._missing_count.get(field, 0) + 1

            # 非缺失值，跳过
            if not is_missing:
                continue

            # 选择填补策略
            actual_strategy = strategy
            if actual_strategy == "auto":
                actual_strategy = self._select_strategy(field, data)

            # 执行填补
            filled_value = self._impute_value(field, actual_strategy, data)

            if filled_value is not None:
                result[field] = filled_value
                if record_fill:
                    record = FillRecord(
                        field=field,
                        original_value=value,
                        filled_value=filled_value,
                        strategy=actual_strategy,
                        reason=self._get_fill_reason(field, actual_strategy),
                        confidence=self._get_fill_confidence(field, actual_strategy),
                        timestamp=datetime.datetime.now().isoformat(),
                    )
                    fill_records.append(record)
                    # 累积记录（用于报告），达到上限时丢弃最旧记录
                    self._accumulated_records.append(record)
                    if len(self._accumulated_records) > self.MAX_FILL_RECORDS:
                        self._accumulated_records = self._accumulated_records[-self.MAX_FILL_RECORDS:]

        return result, fill_records

    def _select_strategy(self, field: str, context: Dict[str, Any]) -> str:
        """自动选择最佳填补策略"""
        # 优先使用医院分布
        if field in self._hospital_dist:
            return "mean"

        # 有相关字段可回归
        if field == "bmi" and context.get("height_cm") and context.get("weight_kg"):
            return "regression"
        if field == "pack_years":
            if context.get("smoking_amount_per_day") and context.get("smoking_years"):
                return "regression"
            return "default"

        # 二分类变量用众数
        if field in ("bcg_scar", "has_cavity", "contact_with_tb"):
            return "mode"

        # 连续变量用中位数
        if field in ("age", "bmi", "smoking_years", "smoking_amount_per_day",
                      "height_cm", "weight_kg"):
            return "median"

        # 默认用医学默认值
        return "default"

    def _impute_value(self, field: str, strategy: str,
                      context: Dict[str, Any]) -> Any:
        """执行具体填补"""
        # 回归填补（利用其他字段计算）
        if strategy == "regression":
            if field == "bmi":
                h = context.get("height_cm")
                w = context.get("weight_kg")
                if h and w:
                    try:
                        return round(float(w) / ((float(h) / 100) ** 2), 1)
                    except (ZeroDivisionError, TypeError, ValueError):
                        pass
            if field == "pack_years":
                daily = context.get("smoking_amount_per_day")
                years = context.get("smoking_years")
                if daily and years:
                    try:
                        return round(float(daily) * float(years) / 20.0, 2)
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
            # 回退到均值
            strategy = "mean"

        # 均值填补
        if strategy == "mean":
            dist = self._hospital_dist.get(field) or DEFAULT_DISTRIBUTION.get(field)
            if dist:
                return dist.get("mean")

        # 中位数填补
        if strategy == "median":
            dist = self._hospital_dist.get(field) or DEFAULT_DISTRIBUTION.get(field)
            if dist:
                return dist.get("median")

        # 众数填补（对分类变量）
        if strategy == "mode":
            categorical_defaults = {
                "bcg_scar": "2",
                "has_cavity": "1",
                "contact_with_tb": "2",
                "gender": 1,
            }
            return categorical_defaults.get(field)

        # 默认值填补
        if strategy == "default":
            defaults = {
                "age": 40,
                "bmi": 22.0,
                "smoking_years": 0,
                "smoking_amount_per_day": 0,
                "drinking_years": 0,
                "height_cm": 165,
                "weight_kg": 60,
                "pack_years": 0,
                "symptom_count": 0,
                "delay_days": 0,
                "contact_duration_months": 0,
                "bcg_scar": "2",
                "has_cavity": "1",
                "contact_with_tb": "2",
                "gender": 1,
            }
            return defaults.get(field)

        return None

    def _get_fill_reason(self, field: str, strategy: str) -> str:
        """获取填补原因说明"""
        messages = {
            "mean": f"字段'{field}'缺失，使用均值填补",
            "median": f"字段'{field}'缺失，使用中位数填补",
            "mode": f"字段'{field}'缺失，使用众数填补",
            "regression": f"字段'{field}'缺失，使用回归填补（基于相关字段计算）",
            "default": f"字段'{field}'缺失，使用医学默认值填补",
        }
        return messages.get(strategy, f"字段'{field}'缺失，使用{strategy}策略填补")

    def _get_fill_confidence(self, field: str, strategy: str) -> float:
        """获取填补置信度"""
        confidences = {
            "regression": 0.70,
            "mean": 0.50,
            "median": 0.50,
            "mode": 0.40,
            "default": 0.30,
        }
        return confidences.get(strategy, 0.50)

    def get_fill_report(self) -> Dict[str, Any]:
        """获取填补报告

        使用 _accumulated_records（持续累积的记录）而非 _fill_records，
        确保缺失率统计与填补记录保持一致。

        返回：
            dict: {
                "total_fills": int,          # 总填补次数
                "fill_records": [FillRecord], # 填补记录
                "missing_rates": {            # 各字段缺失率
                    "field": rate, ...
                },
                "strategy_usage": {           # 策略使用统计
                    "mean": count, ...
                },
            }
        """
        # 计算缺失率
        missing_rates = {}
        for field in DEFAULT_DISTRIBUTION:
            total = self._total_count.get(field, 0)
            missing = self._missing_count.get(field, 0)
            if total > 0:
                missing_rates[field] = round(missing / total * 100, 2)
            else:
                missing_rates[field] = 0.0

        # 策略使用统计（基于累积记录）
        strategy_usage = {}
        for record in self._accumulated_records:
            strategy_usage[record.strategy] = strategy_usage.get(record.strategy, 0) + 1

        return {
            "total_fills": len(self._accumulated_records),
            "fill_records": [
                {
                    "field": r.field,
                    "original_value": r.original_value,
                    "filled_value": r.filled_value,
                    "strategy": r.strategy,
                    "reason": r.reason,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp,
                }
                for r in self._accumulated_records
            ],
            "missing_rates": missing_rates,
            "strategy_usage": strategy_usage,
        }

    def get_missing_rate(self, field: str) -> float:
        """获取某字段的缺失率"""
        total = self._total_count.get(field, 0)
        missing = self._missing_count.get(field, 0)
        if total > 0:
            return round(missing / total * 100, 2)
        return 0.0

    def get_all_missing_rates(self) -> Dict[str, float]:
        """获取所有字段缺失率"""
        rates = {}
        for field in DEFAULT_DISTRIBUTION:
            rates[field] = self.get_missing_rate(field)
        return rates

    def reset_stats(self):
        """重置统计信息"""
        self._fill_records.clear()
        self._accumulated_records.clear()
        self._missing_count.clear()
        self._total_count.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """获取填补器统计"""
        return {
            "hospital_distribution_fields": len(self._hospital_dist),
            "total_fills": len(self._accumulated_records),
            "fields_tracked": len(DEFAULT_DISTRIBUTION),
            "fill_strategies": list(FILL_STRATEGIES.keys()),
        }


# ============================================================================
# 智能特征计算引擎（综合入口）
# ============================================================================


class FeatureCalculator:
    """智能特征计算引擎

    综合计算所有需要推导的特征，包括：
    - BCG接种史
    - 吸烟指数
    - BMI
    - 密切接触史
    - 缺失值填补
    """

    def __init__(self):
        self.imputer = MissingValueImputer()

    def calculate_all(
        self,
        patient_data: Dict[str, Any],
        vaccination_records: Optional[List[Dict[str, Any]]] = None,
        scar_check_result: Optional[str] = None,
        clinical_notes: Optional[List[str]] = None,
        contact_records: Optional[List[Dict[str, Any]]] = None,
        report_card: Optional[Dict[str, Any]] = None,
        fill_strategy: str = "auto",
    ) -> CalculationResult:
        """计算所有特征

        参数：
            patient_data: 患者原始数据（已映射为标准字段）
            vaccination_records: 接种记录
            scar_check_result: 卡痕检查结果
            clinical_notes: 病历文本
            contact_records: 接触者记录
            report_card: 传染病报告卡
            fill_strategy: 缺失值填补策略

        返回：
            CalculationResult
        """
        result = CalculationResult()
        features = dict(patient_data)
        warnings = []
        source_fields = {}

        # 1. 先填补原始缺失值，确保派生特征计算有输入数据
        filled, fill_records = self.imputer.fill_missing(features, strategy=fill_strategy)
        features = filled
        result.fill_records = fill_records

        # 2. BCG接种史
        if "bcg_scar" not in features or not features.get("bcg_scar"):
            bcg_result = compute_bcg_status(
                vaccination_records=vaccination_records,
                scar_check_result=scar_check_result,
                clinical_notes=clinical_notes,
            )
            features["bcg_scar"] = bcg_result["bcg_scar"]
            features["bcg_confidence"] = bcg_result["confidence"]
            features["bcg_method"] = bcg_result["method"]
            source_fields["bcg_scar"] = ["vaccination_records", "scar_check", "clinical_notes"]
            if bcg_result["evidence"]:
                features["bcg_evidence"] = "; ".join(bcg_result["evidence"])

        # 3. 吸烟指数（基于填补后的 smoking_years 和 smoking_amount_per_day）
        if "pack_years" not in features or not features.get("pack_years"):
            smoking_result = compute_smoking_index(
                amount_per_day=features.get("smoking_amount_per_day"),
                smoking_years=features.get("smoking_years"),
                amount_per_week=features.get("smoking_amount_per_week"),
            )
            features["pack_years"] = smoking_result["pack_years"]
            features["smoking_confidence"] = smoking_result["confidence"]
            source_fields["pack_years"] = ["smoking_amount_per_day", "smoking_years"]

        # 4. BMI（基于填补后的身高体重）
        if "bmi" not in features or not features.get("bmi"):
            bmi_result = compute_bmi(
                height_cm=features.get("height_cm"),
                weight_kg=features.get("weight_kg"),
            )
            features["bmi"] = bmi_result["bmi"]
            features["bmi_category"] = bmi_result["category"]
            source_fields["bmi"] = ["height_cm", "weight_kg"]
        elif "bmi_category" not in features and features.get("bmi"):
            # BMI was filled by imputer, compute category from filled value
            try:
                bmi_val = float(features["bmi"])
                if bmi_val < 18.5:
                    features["bmi_category"] = "underweight"
                elif bmi_val < 24.0:
                    features["bmi_category"] = "normal"
                elif bmi_val < 28.0:
                    features["bmi_category"] = "overweight"
                else:
                    features["bmi_category"] = "obese"
            except (TypeError, ValueError):
                pass

        # 5. 密切接触史
        contact_result = extract_contact_history(
            contact_records=contact_records,
            clinical_notes=clinical_notes,
            report_card=report_card,
        )
        features["has_contact"] = "1" if contact_result["has_contact"] else "2"
        features["contact_type"] = contact_result["contact_type"]
        features["contact_duration_months"] = contact_result["contact_duration_months"]
        features["contact_with_tb"] = contact_result["contact_with_tb"]
        features["contact_confidence"] = contact_result["confidence"]
        source_fields["contact"] = ["contact_records", "clinical_notes", "report_card"]

        # 6. 缺失率统计
        result.missing_rate = self.imputer.get_all_missing_rates()

        # 收集警告
        for record in result.fill_records:
            warnings.append(
                f"[{record.strategy}] {record.reason} → {record.filled_value}"
            )

        result.features = filled
        result.warnings = warnings
        result.source_fields_used = source_fields

        return result


# ============================================================================
# 便捷入口
# ============================================================================

def calculate_patient_features(
    patient_data: Dict[str, Any],
    vaccination_records: Optional[List[Dict[str, Any]]] = None,
    scar_check_result: Optional[str] = None,
    clinical_notes: Optional[List[str]] = None,
    contact_records: Optional[List[Dict[str, Any]]] = None,
    report_card: Optional[Dict[str, Any]] = None,
) -> CalculationResult:
    """便捷入口：计算患者所有特征

    参数：
        patient_data: 患者数据字典
        vaccination_records: 接种记录
        scar_check_result: 卡痕检查结果
        clinical_notes: 病历文本
        contact_records: 接触者记录
        report_card: 传染病报告卡

    返回：
        CalculationResult
    """
    calculator = FeatureCalculator()
    return calculator.calculate_all(
        patient_data=patient_data,
        vaccination_records=vaccination_records,
        scar_check_result=scar_check_result,
        clinical_notes=clinical_notes,
        contact_records=contact_records,
        report_card=report_card,
    )