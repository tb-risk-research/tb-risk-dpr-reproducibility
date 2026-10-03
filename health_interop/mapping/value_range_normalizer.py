#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检验值参考范围标准化

不同医院对同一检验项目的参考范围可能不同，需要统一标准化。
本模块提供：
1. 参考范围归一化：将不同医院参考范围映射到统一标准范围
2. 结果解释：根据参考范围判断结果是否异常
3. 异常程度量化：计算偏离程度
4. 多维度参考范围：支持按年龄、性别、孕周等分层
5. 自定义参考范围配置
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("tb_risk.health_interop.mapping.value_range")

# ============================================================================
# 结核相关检验项目标准参考范围
# ============================================================================

STANDARD_REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    # 痰涂片
    "sputum_smear": {
        "name": "痰涂片抗酸染色",
        "standard_unit": "定性",
        "normal": "阴性",
        "abnormal": "阳性",
        "interpretation": "qualitative",
        "categories": {
            "negative": {"label": "阴性", "value": "1"},
            "positive": {"label": "阳性", "value": "2"},
        },
    },
    # 痰培养
    "sputum_culture": {
        "name": "痰结核分枝杆菌培养",
        "standard_unit": "定性",
        "normal": "阴性",
        "abnormal": "阳性",
        "interpretation": "qualitative",
    },
    # Xpert MTB/RIF
    "xpert_mtb_rif": {
        "name": "Xpert MTB/RIF",
        "standard_unit": "定性",
        "normal": "MTB未检出",
        "abnormal": "MTB检出",
        "interpretation": "qualitative",
    },
    # T-SPOT/IGRA
    "t_spot": {
        "name": "结核感染T细胞检测",
        "standard_unit": "SFC/10⁶PBMC",
        "normal": {"low": 0, "high": 6},
        "borderline": {"low": 6, "high": 14},
        "abnormal": {"low": 14, "high": 9999},
        "interpretation": "numeric",
    },
    # ESR 血沉
    "esr": {
        "name": "血沉(ESR)",
        "standard_unit": "mm/h",
        "normal": {
            "default": {"low": 0, "high": 15},
            "by_gender": {
                "male": {"low": 0, "high": 15},
                "female": {"low": 0, "high": 20},
            },
            "by_age": {
                "over_50": {"low": 0, "high": 20},
                "over_60": {"low": 0, "high": 25},
            },
        },
        "interpretation": "numeric",
        "abnormal_direction": "high",
    },
    # CRP C反应蛋白
    "crp": {
        "name": "C反应蛋白(CRP)",
        "standard_unit": "mg/L",
        "normal": {"low": 0, "high": 5},
        "mild": {"low": 5, "high": 40},
        "moderate": {"low": 40, "high": 100},
        "severe": {"low": 100, "high": 9999},
        "interpretation": "numeric",
        "abnormal_direction": "high",
    },
    # 空腹血糖
    "blood_glucose_fasting": {
        "name": "空腹血糖",
        "standard_unit": "mmol/L",
        "normal": {"low": 3.9, "high": 6.1},
        "impaired": {"low": 6.1, "high": 7.0},
        "abnormal": {"low": 7.0, "high": 999},
        "interpretation": "numeric",
        "abnormal_direction": "high",
    },
    # 随机血糖
    "blood_glucose_random": {
        "name": "随机血糖",
        "standard_unit": "mmol/L",
        "normal": {"low": 0, "high": 11.1},
        "abnormal": {"low": 11.1, "high": 999},
        "interpretation": "numeric",
        "abnormal_direction": "high",
    },
    # WBC 白细胞
    "wbc": {
        "name": "白细胞计数",
        "standard_unit": "10⁹/L",
        "normal": {"low": 3.5, "high": 9.5},
        "interpretation": "numeric",
        "abnormal_direction": "both",
    },
    # HGB 血红蛋白
    "hgb": {
        "name": "血红蛋白",
        "standard_unit": "g/L",
        "normal": {
            "default": {"low": 120, "high": 160},
            "by_gender": {
                "male": {"low": 130, "high": 175},
                "female": {"low": 115, "high": 150},
            },
        },
        "interpretation": "numeric",
        "abnormal_direction": "both",
    },
    # ALT 谷丙转氨酶
    "alt": {
        "name": "谷丙转氨酶(ALT)",
        "standard_unit": "U/L",
        "normal": {"low": 0, "high": 40},
        "interpretation": "numeric",
        "abnormal_direction": "high",
    },
    # AST 谷草转氨酶
    "ast": {
        "name": "谷草转氨酶(AST)",
        "standard_unit": "U/L",
        "normal": {"low": 0, "high": 40},
        "interpretation": "numeric",
        "abnormal_direction": "high",
    },
    # Cr 肌酐
    "creatinine": {
        "name": "肌酐(Cr)",
        "standard_unit": "μmol/L",
        "normal": {
            "default": {"low": 44, "high": 104},
            "by_gender": {
                "male": {"low": 54, "high": 106},
                "female": {"low": 44, "high": 97},
            },
        },
        "interpretation": "numeric",
        "abnormal_direction": "high",
    },
    # BMI
    "bmi": {
        "name": "体重指数",
        "standard_unit": "kg/m²",
        "underweight": {"low": 0, "high": 18.5},
        "normal": {"low": 18.5, "high": 24.0},
        "overweight": {"low": 24.0, "high": 28.0},
        "obese": {"low": 28.0, "high": 999},
        "interpretation": "numeric",
        "abnormal_direction": "both",
    },
}

# 医院自定义参考范围（模拟不同医院的参考范围差异）
HOSPITAL_REFERENCE_RANGES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "hospital_a": {
        "esr": {"low": 0, "high": 20, "unit": "mm/h"},
        "crp": {"low": 0, "high": 8, "unit": "mg/L"},
        "wbc": {"low": 4.0, "high": 10.0, "unit": "10⁹/L"},
    },
    "hospital_b": {
        "esr": {"low": 0, "high": 15, "unit": "mm/h"},
        "crp": {"low": 0, "high": 6, "unit": "mg/L"},
        "alt": {"low": 0, "high": 50, "unit": "U/L"},
    },
    "hospital_c": {
        "esr": {"low": 0, "high": 25, "unit": "mm/h"},
        "crp": {"low": 0, "high": 10, "unit": "mg/L"},
        "creatinine": {"low": 50, "high": 120, "unit": "μmol/L"},
    },
}


class ValueRangeNormalizer:
    """检验值参考范围标准化

    将不同医院的检验结果参考范围映射到统一标准范围，
    并判断结果是否异常及异常程度。
    """

    def __init__(self):
        self._custom_ranges: Dict[str, Dict[str, Any]] = {}
        self._hospital_ranges: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # 加载医院预设范围
        for hospital_id, ranges in HOSPITAL_REFERENCE_RANGES.items():
            self._hospital_ranges[hospital_id] = dict(ranges)

    # ==================== 范围管理 ====================

    def get_standard_range(self, test_code: str) -> Optional[Dict[str, Any]]:
        """获取检验项目的标准参考范围"""
        return STANDARD_REFERENCE_RANGES.get(test_code)

    def get_all_standard_ranges(self) -> Dict[str, Dict[str, Any]]:
        """获取所有标准参考范围"""
        return dict(STANDARD_REFERENCE_RANGES)

    def set_hospital_range(self, hospital_id: str, test_code: str,
                           low: float, high: float, unit: str = ""):
        """设置医院的参考范围

        参数：
            hospital_id: 医院标识
            test_code: 检验项目编码
            low: 下限
            high: 上限
            unit: 单位
        """
        self._hospital_ranges.setdefault(hospital_id, {})[test_code] = {
            "low": low, "high": high, "unit": unit,
        }

    def get_hospital_range(self, hospital_id: str, test_code: str) -> Optional[Dict[str, Any]]:
        """获取医院的参考范围"""
        return self._hospital_ranges.get(hospital_id, {}).get(test_code)

    def set_custom_range(self, test_code: str, range_def: Dict[str, Any]):
        """设置自定义参考范围"""
        self._custom_ranges[test_code] = range_def

    # ==================== 核心标准化 ====================

    def normalize(self, value: Any, test_code: str, *,
                  hospital_id: str = "",
                  gender: str = "",
                  age: Optional[float] = None,
                  unit: str = "") -> Optional[Dict[str, Any]]:
        """标准化检验结果，判断是否异常

        参数：
            value: 检验值（数值或定性结果）
            test_code: 检验项目编码
            hospital_id: 医院标识（用于获取医院特定参考范围）
            gender: 性别（male/female，用于分层参考范围）
            age: 年龄（岁，用于年龄分层参考范围）
            unit: 值的单位

        返回：
            dict | None: {
                "value": 原始值,
                "standard_unit": 标准单位,
                "normalized": 标准化结果,
                "interpretation": 解释（normal/abnormal/critical等）,
                "severity": 异常程度（0-1，0为正常）,
                "category": 分类标签,
                "details": 详细说明,
            }
        """
        if value is None or value == "":
            return None

        # 获取标准范围定义
        range_def = self._custom_ranges.get(test_code) or STANDARD_REFERENCE_RANGES.get(test_code)
        if not range_def:
            LOGGER.warning("未知检验项目: %s", test_code)
            return None

        interpretation_type = range_def.get("interpretation", "qualitative")

        if interpretation_type == "qualitative":
            return self._normalize_qualitative(value, range_def)
        else:
            return self._normalize_numeric(value, test_code, range_def,
                                           hospital_id, gender, age)

    def _normalize_qualitative(self, value: Any, range_def: Dict[str, Any]) -> Dict[str, Any]:
        """处理定性结果"""
        str_val = str(value).strip().lower()

        # 分类映射
        categories = range_def.get("categories", {})
        for cat_key, cat_info in categories.items():
            if str_val == cat_info.get("value", "").lower() or str_val == cat_key.lower():
                return {
                    "value": value,
                    "standard_unit": range_def.get("standard_unit", ""),
                    "normalized": cat_info.get("value", value),
                    "interpretation": "abnormal" if cat_key == "positive" else "normal",
                    "severity": 1.0 if cat_key == "positive" else 0.0,
                    "category": cat_key,
                    "details": cat_info.get("label", str_val),
                }

        # 简单的阴性/阳性判断
        negative_keywords = ["阴性", "阴", "negative", "(-)", "未检出", "未查", "正常", "无"]
        positive_keywords = ["阳性", "阳", "positive", "(+)", "检出", "异常", "有"]

        for kw in negative_keywords:
            if kw in str_val:
                return {
                    "value": value,
                    "standard_unit": range_def.get("standard_unit", ""),
                    "normalized": range_def.get("normal", "阴性"),
                    "interpretation": "normal",
                    "severity": 0.0,
                    "category": "negative",
                    "details": "阴性/正常",
                }
        for kw in positive_keywords:
            if kw in str_val:
                return {
                    "value": value,
                    "standard_unit": range_def.get("standard_unit", ""),
                    "normalized": range_def.get("abnormal", "阳性"),
                    "interpretation": "abnormal",
                    "severity": 1.0,
                    "category": "positive",
                    "details": "阳性/异常",
                }

        return {
            "value": value,
            "standard_unit": range_def.get("standard_unit", ""),
            "normalized": value,
            "interpretation": "unknown",
            "severity": 0.0,
            "category": "unknown",
            "details": "无法判断",
        }

    def _normalize_numeric(self, value: Any, test_code: str,
                           range_def: Dict[str, Any],
                           hospital_id: str, gender: str,
                           age: Optional[float]) -> Dict[str, Any]:
        """处理数值型结果"""
        try:
            val = float(value)
        except (TypeError, ValueError):
            return {
                "value": value,
                "standard_unit": range_def.get("standard_unit", ""),
                "normalized": value,
                "interpretation": "error",
                "severity": 0.0,
                "category": "error",
                "details": "非数值型结果",
            }

        # 获取参考范围
        ref_range = self._get_applicable_range(test_code, range_def, hospital_id, gender, age)
        if not ref_range:
            return {
                "value": val,
                "standard_unit": range_def.get("standard_unit", ""),
                "normalized": val,
                "interpretation": "unknown",
                "severity": 0.0,
                "category": "unknown",
                "details": "无参考范围",
            }

        low = ref_range.get("low", 0)
        high = ref_range.get("high", 9999)
        direction = range_def.get("abnormal_direction", "high")

        # 判断是否异常及程度
        if low <= val <= high:
            # 正常范围内，计算偏离中心的程度
            mid = (low + high) / 2
            range_width = (high - low) / 2 if high > low else 1
            severity = abs(val - mid) / range_width * 0.5  # 正常范围内最大0.5
            return {
                "value": val,
                "standard_unit": range_def.get("standard_unit", ""),
                "normalized": val,
                "interpretation": "normal",
                "severity": round(severity, 4),
                "category": "normal",
                "details": f"正常范围 ({low}-{high})",
            }
        else:
            # 异常
            if direction == "high":
                if val < low:
                    # 偏低
                    severity = min(1.0, (low - val) / low)
                    return {
                        "value": val,
                        "standard_unit": range_def.get("standard_unit", ""),
                        "normalized": val,
                        "interpretation": "abnormal_low",
                        "severity": round(severity, 4),
                        "category": "low",
                        "details": f"低于正常范围 ({low}-{high})",
                    }
                else:
                    # 偏高
                    severity = min(1.0, (val - high) / high)
                    is_critical = severity >= 2.0
                    return {
                        "value": val,
                        "standard_unit": range_def.get("standard_unit", ""),
                        "normalized": val,
                        "interpretation": "critical" if is_critical else "abnormal_high",
                        "severity": round(severity, 4),
                        "category": "high",
                        "details": f"高于正常范围 ({low}-{high})",
                    }
            elif direction == "low":
                if val > high:
                    severity = min(1.0, (val - high) / high)
                    return {
                        "value": val,
                        "standard_unit": range_def.get("standard_unit", ""),
                        "normalized": val,
                        "interpretation": "abnormal_high",
                        "severity": round(severity, 4),
                        "category": "high",
                        "details": f"高于正常范围 ({low}-{high})",
                    }
                else:
                    severity = min(1.0, (low - val) / low)
                    is_critical = severity >= 2.0
                    return {
                        "value": val,
                        "standard_unit": range_def.get("standard_unit", ""),
                        "normalized": val,
                        "interpretation": "critical" if is_critical else "abnormal_low",
                        "severity": round(severity, 4),
                        "category": "low",
                        "details": f"低于正常范围 ({low}-{high})",
                    }
            else:  # both
                if val < low:
                    severity = min(1.0, (low - val) / low)
                    is_critical = severity >= 2.0
                    return {
                        "value": val,
                        "standard_unit": range_def.get("standard_unit", ""),
                        "normalized": val,
                        "interpretation": "critical" if is_critical else "abnormal_low",
                        "severity": round(severity, 4),
                        "category": "low",
                        "details": f"低于正常范围 ({low}-{high})",
                    }
                else:
                    severity = min(1.0, (val - high) / high)
                    is_critical = severity >= 2.0
                    return {
                        "value": val,
                        "standard_unit": range_def.get("standard_unit", ""),
                        "normalized": val,
                        "interpretation": "critical" if is_critical else "abnormal_high",
                        "severity": round(severity, 4),
                        "category": "high",
                        "details": f"高于正常范围 ({low}-{high})",
                    }

    def _get_applicable_range(self, test_code: str, range_def: Dict[str, Any],
                              hospital_id: str, gender: str,
                              age: Optional[float]) -> Optional[Dict[str, Any]]:
        """获取适用的参考范围（考虑医院、性别、年龄分层）"""
        # 1. 优先使用医院特定范围
        if hospital_id:
            hospital_range = self._hospital_ranges.get(hospital_id, {}).get(test_code)
            if hospital_range:
                return hospital_range

        # 2. 使用标准范围，考虑性别和年龄分层
        normal_range = range_def.get("normal", {})
        if isinstance(normal_range, dict) and "by_gender" in normal_range and gender:
            gender_range = normal_range["by_gender"].get(gender)
            if gender_range:
                return gender_range

        if isinstance(normal_range, dict) and "by_age" in normal_range and age is not None:
            for age_label, age_range in normal_range["by_age"].items():
                if age_label == "over_50" and age >= 50:
                    return age_range
                if age_label == "over_60" and age >= 60:
                    return age_range

        # 3. 返回默认范围
        if isinstance(normal_range, dict):
            default = normal_range.get("default", normal_range)
            if isinstance(default, dict) and "low" in default and "high" in default:
                return default
        return None

    # ==================== 批量标准化 ====================

    def normalize_batch(self, results: List[Dict[str, Any]], *,
                        hospital_id: str = "",
                        gender: str = "",
                        age: Optional[float] = None) -> List[Dict[str, Any]]:
        """批量标准化检验结果

        参数：
            results: [{test_code, value, ...}, ...]
            hospital_id, gender, age: 同上

        返回：
            list: 标准化后的结果列表
        """
        normalized = []
        for item in results:
            test_code = item.get("test_code", "")
            value = item.get("value")
            if test_code and value is not None:
                result = self.normalize(
                    value, test_code,
                    hospital_id=item.get("hospital_id", hospital_id),
                    gender=item.get("gender", gender),
                    age=item.get("age", age),
                    unit=item.get("unit", ""),
                )
                if result:
                    normalized.append({**item, "normalized": result})
                else:
                    normalized.append(item)
            else:
                normalized.append(item)
        return normalized

    # ==================== 统计 ====================

    def get_statistics(self) -> Dict[str, Any]:
        """获取标准化器统计"""
        return {
            "standard_ranges": len(STANDARD_REFERENCE_RANGES),
            "custom_ranges": len(self._custom_ranges),
            "hospital_range_groups": len(self._hospital_ranges),
            "total_hospital_ranges": sum(
                len(ranges) for ranges in self._hospital_ranges.values()
            ),
        }

    def get_hospital_list(self) -> List[str]:
        """获取已配置参考范围的医院列表"""
        return list(self._hospital_ranges.keys())