#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单位转换系统

支持 tb_risk 数据映射中常见的单位转换场景：
1. 年龄转换：月→岁、天→岁、周岁→虚岁
2. 吸烟量转换：支/天→包/年、支/周→包/年
3. 饮酒量转换：两/天→标准杯/周
4. 检验单位转换：mg/dL↔mmol/L、g/L↔g/dL
5. 身高体重转换：cm→m、斤→kg
6. 温度转换：°C↔°F
7. 自定义公式转换
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

LOGGER = logging.getLogger("tb_risk.health_interop.mapping.unit_converter")

# ============================================================================
# 预定义转换规则
# ============================================================================

CONVERSION_RULES: Dict[str, Dict[str, Any]] = {
    # ---- 年龄转换 ----
    "age_month_to_year": {
        "name": "月龄转岁",
        "from_unit": "月",
        "to_unit": "岁",
        "formula": "x / 12.0",
        "description": "将月龄转换为岁（x/12）",
        "precision": 1,
    },
    "age_day_to_year": {
        "name": "天数转岁",
        "from_unit": "天",
        "to_unit": "岁",
        "formula": "x / 365.25",
        "description": "将天数转换为岁（x/365.25）",
        "precision": 1,
    },
    "age_lunar_to_solar": {
        "name": "虚岁转周岁",
        "from_unit": "虚岁",
        "to_unit": "岁",
        "formula": "x - 1",
        "description": "将虚岁转换为周岁（x-1）",
        "precision": 0,
    },

    # ---- 吸烟量转换 ----
    "cig_per_day_to_pack_year": {
        "name": "支/天→包/年",
        "from_unit": "支/天",
        "to_unit": "包/年",
        "formula": "(x / 20.0) * smoking_years",
        "description": "每日吸烟量(支)×吸烟年数÷20",
        "precision": 1,
        "requires_fields": ["smoking_years"],
    },
    "cig_per_week_to_pack_year": {
        "name": "支/周→包/年",
        "from_unit": "支/周",
        "to_unit": "包/年",
        "formula": "(x / 20.0) * smoking_years / 52.0",
        "description": "每周吸烟量(支)×吸烟年数÷20÷52",
        "precision": 1,
        "requires_fields": ["smoking_years"],
    },
    "cig_per_day_to_per_day": {
        "name": "支/天→支/天（标准化）",
        "from_unit": "支/天",
        "to_unit": "支/天",
        "formula": "x",
        "precision": 0,
    },

    # ---- 饮酒量转换 ----
    "alcohol_liang_to_standard_drink": {
        "name": "两/天→标准杯/周",
        "from_unit": "两/天",
        "to_unit": "标准杯/周",
        "formula": "x * 7.0 * 0.5",
        "description": "1两≈0.5标准杯，每周=每日×7",
        "precision": 1,
    },
    "alcohol_ml_to_standard_drink": {
        "name": "mL/天→标准杯/周",
        "from_unit": "mL/天",
        "to_unit": "标准杯/周",
        "formula": "x * 7.0 / 30.0",
        "description": "1标准杯≈30mL纯酒精",
        "precision": 1,
    },

    # ---- 检验单位转换 ----
    "glucose_mgdL_to_mmolL": {
        "name": "血糖 mg/dL→mmol/L",
        "from_unit": "mg/dL",
        "to_unit": "mmol/L",
        "formula": "x / 18.018",
        "description": "血糖值÷18.018",
        "precision": 1,
    },
    "glucose_mmolL_to_mgdL": {
        "name": "血糖 mmol/L→mg/dL",
        "from_unit": "mmol/L",
        "to_unit": "mg/dL",
        "formula": "x * 18.018",
        "precision": 0,
    },
    "creatinine_mgdL_to_umolL": {
        "name": "肌酐 mg/dL→μmol/L",
        "from_unit": "mg/dL",
        "to_unit": "μmol/L",
        "formula": "x * 88.4",
        "precision": 0,
    },
    "creatinine_umolL_to_mgdL": {
        "name": "肌酐 μmol/L→mg/dL",
        "from_unit": "μmol/L",
        "to_unit": "mg/dL",
        "formula": "x / 88.4",
        "precision": 1,
    },
    "calcium_mgdL_to_mmolL": {
        "name": "血钙 mg/dL→mmol/L",
        "from_unit": "mg/dL",
        "to_unit": "mmol/L",
        "formula": "x / 4.008",
        "precision": 2,
    },
    "hb_gL_to_gdL": {
        "name": "血红蛋白 g/L→g/dL",
        "from_unit": "g/L",
        "to_unit": "g/dL",
        "formula": "x / 10.0",
        "precision": 1,
    },
    "hb_gdL_to_gL": {
        "name": "血红蛋白 g/dL→g/L",
        "from_unit": "g/dL",
        "to_unit": "g/L",
        "formula": "x * 10.0",
        "precision": 0,
    },
    "wbc_10e9L_to_count": {
        "name": "WBC 10⁹/L→个/μL",
        "from_unit": "10⁹/L",
        "to_unit": "个/μL",
        "formula": "x * 1000",
        "precision": 0,
    },

    # ---- 身高体重转换 ----
    "height_cm_to_m": {
        "name": "身高 cm→m",
        "from_unit": "cm",
        "to_unit": "m",
        "formula": "x / 100.0",
        "precision": 2,
    },
    "weight_jin_to_kg": {
        "name": "体重 斤→kg",
        "from_unit": "斤",
        "to_unit": "kg",
        "formula": "x / 2.0",
        "precision": 1,
    },
    "weight_kg_to_jin": {
        "name": "体重 kg→斤",
        "from_unit": "kg",
        "to_unit": "斤",
        "formula": "x * 2.0",
        "precision": 0,
    },
    "weight_lb_to_kg": {
        "name": "体重 lb→kg",
        "from_unit": "lb",
        "to_unit": "kg",
        "formula": "x * 0.453592",
        "precision": 1,
    },

    # ---- 温度转换 ----
    "temp_c_to_f": {
        "name": "°C→°F",
        "from_unit": "°C",
        "to_unit": "°F",
        "formula": "x * 9.0 / 5.0 + 32",
        "precision": 1,
    },
    "temp_f_to_c": {
        "name": "°F→°C",
        "from_unit": "°F",
        "to_unit": "°C",
        "formula": "(x - 32) * 5.0 / 9.0",
        "precision": 1,
    },

    # ---- 时间转换 ----
    "day_to_month": {
        "name": "天→月",
        "from_unit": "天",
        "to_unit": "月",
        "formula": "x / 30.0",
        "precision": 1,
    },
    "month_to_year": {
        "name": "月→年",
        "from_unit": "月",
        "to_unit": "年",
        "formula": "x / 12.0",
        "precision": 1,
    },
    "week_to_month": {
        "name": "周→月",
        "from_unit": "周",
        "to_unit": "月",
        "formula": "x / 4.35",
        "precision": 1,
    },
}

# 单位别名映射（用于自动检测）
UNIT_ALIASES: Dict[str, str] = {
    # 年龄
    "月": "月", "个月": "月", "month": "月", "months": "月", "mon": "月",
    "岁": "岁", "yr": "岁", "yrs": "岁",
    "天": "天", "日": "天", "day": "天", "days": "天",
    # 吸烟
    "支/天": "支/天", "支/日": "支/天", "cig/day": "支/天", "cigarettes/day": "支/天",
    "支/周": "支/周", "cig/week": "支/周", "cigarettes/week": "支/周",
    "包/年": "包/年", "pack/year": "包/年", "pack-years": "包/年", "py": "包/年",
    # 饮酒
    "两/天": "两/天", "两/日": "两/天", "ml/天": "mL/天", "ml": "mL/天",
    "标准杯/周": "标准杯/周", "drinks/week": "标准杯/周",
    # 血糖
    "mg/dl": "mg/dL", "mg/dL": "mg/dL", "mg%": "mg/dL",
    "mmol/l": "mmol/L", "mmol/L": "mmol/L", "mmo l/L": "mmol/L",
    # 身高体重
    "cm": "cm", "厘米": "cm", "公分": "cm",
    "m": "m", "米": "m",
    "kg": "kg", "公斤": "kg", "千克": "kg",
    "斤": "斤", "jin": "斤",
    "lb": "lb", "lbs": "lb", "磅": "lb",
    # 温度
    "°c": "°C", "℃": "°C", "c": "°C", "celsius": "°C",
    "°f": "°F", "℉": "°F", "f": "°F", "fahrenheit": "°F",
    # 时间
    "周": "周", "week": "周", "weeks": "周", "wk": "周",
    "月": "月", "month": "月", "months": "月",
    "年": "年", "year": "年", "years": "年",
}


class UnitConverter:
    """单位转换系统

    支持将医院系统使用的不同单位转换为 tb_risk 标准单位。
    支持自动检测单位、按规则转换、批量转换。
    """

    def __init__(self):
        self._custom_rules: Dict[str, Dict[str, Any]] = {}
        self._rule_cache: Dict[str, Callable] = {}

    # ==================== 规则管理 ====================

    def get_available_rules(self) -> List[Dict[str, Any]]:
        """获取所有可用转换规则"""
        rules = []
        for rule_id, rule in CONVERSION_RULES.items():
            rules.append({"id": rule_id, **rule})
        for rule_id, rule in self._custom_rules.items():
            rules.append({"id": rule_id, **rule, "custom": True})
        return rules

    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """获取指定转换规则"""
        rule = CONVERSION_RULES.get(rule_id)
        if rule:
            return {"id": rule_id, **rule}
        rule = self._custom_rules.get(rule_id)
        if rule:
            return {"id": rule_id, **rule, "custom": True}
        return None

    def add_rule(self, rule_id: str, name: str, from_unit: str, to_unit: str,
                 formula: str, precision: int = 2, description: str = "",
                 requires_fields: Optional[List[str]] = None) -> str:
        """添加自定义转换规则

        参数：
            rule_id: 规则标识符
            name: 规则名称
            from_unit: 源单位
            to_unit: 目标单位
            formula: 转换公式（x为输入值）
            precision: 精度（小数位数）
            description: 描述
            requires_fields: 公式依赖的其他字段名

        返回：
            str: 规则ID
        """
        rule = {
            "name": name,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "formula": formula,
            "precision": precision,
            "description": description or f"{from_unit}→{to_unit}",
        }
        if requires_fields:
            rule["requires_fields"] = requires_fields
        self._custom_rules[rule_id] = rule
        LOGGER.info("添加自定义转换规则: %s (%s→%s)", rule_id, from_unit, to_unit)
        return rule_id

    def remove_rule(self, rule_id: str) -> bool:
        """移除自定义转换规则"""
        if rule_id in self._custom_rules:
            del self._custom_rules[rule_id]
            self._rule_cache.pop(rule_id, None)
            return True
        return False

    # ==================== 核心转换 ====================

    def convert(self, value: Any, rule_id: str,
                context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """执行单位转换

        参数：
            value: 待转换的值
            rule_id: 转换规则ID
            context: 上下文数据（用于依赖其他字段的公式，如吸烟指数需要吸烟年数）

        返回：
            dict | None: {
                "value": 转换后的值,
                "unit": 目标单位,
                "precision": 精度,
                "rule_id": 规则ID,
                "rule_name": 规则名称,
            }
        """
        if value is None or value == "":
            return None

        # 查找规则
        rule = CONVERSION_RULES.get(rule_id) or self._custom_rules.get(rule_id)
        if not rule:
            LOGGER.warning("未知转换规则: %s", rule_id)
            return None

        try:
            val = float(value)
        except (TypeError, ValueError):
            LOGGER.warning("无法转换为数值: value=%s", value)
            return None

        # 获取公式
        formula = rule["formula"]
        context = context or {}

        # 检查依赖字段
        required_fields = rule.get("requires_fields", [])
        for field in required_fields:
            if field not in context:
                LOGGER.warning("转换规则 '%s' 依赖字段 '%s' 缺失", rule_id, field)
                return None

        try:
            # 构建求值环境：x + 所有上下文字段
            eval_env = {"x": val, "__builtins__": {}}
            # 为上下文中的数值字段添加变量
            for k, v in context.items():
                if isinstance(v, (int, float)):
                    eval_env[k] = v
                elif isinstance(v, str):
                    try:
                        eval_env[k] = float(v)
                    except (TypeError, ValueError):
                        eval_env[k] = v

            result = eval(formula, eval_env, eval_env)
            precision = rule.get("precision", 2)
            if precision is not None:
                result = round(result, precision)

            return {
                "value": result,
                "unit": rule.get("to_unit", ""),
                "precision": precision,
                "rule_id": rule_id,
                "rule_name": rule.get("name", ""),
            }
        except Exception as e:
            LOGGER.warning("转换失败: rule=%s, value=%s, error=%s", rule_id, value, e)
            return None

    def convert_batch(self, values: List[Tuple[Any, str]],
                      context: Optional[Dict[str, Any]] = None) -> List[Optional[Dict[str, Any]]]:
        """批量转换

        参数：
            values: [(value, rule_id), ...] 列表
            context: 共享上下文数据

        返回：
            list: 转换结果列表
        """
        return [self.convert(v, r, context) for v, r in values]

    # ==================== 自动检测与转换 ====================

    def detect_unit(self, value: str) -> Optional[str]:
        """从字符串中检测单位

        从"2岁"、"360mg/dL"等字符串中提取标准化单位名。
        """
        if not value or not isinstance(value, str):
            return None
        value = value.strip().lower()

        # 匹配常见模式：数字+单位
        patterns = [
            (r"(\d+\.?\d*)\s*岁", "岁"),
            (r"(\d+\.?\d*)\s*个月?", "月"),
            (r"(\d+\.?\d*)\s*月", "月"),
            (r"(\d+\.?\d*)\s*天", "天"),
            (r"(\d+\.?\d*)\s*日", "天"),
            (r"(\d+\.?\d*)\s*支/[天日]", "支/天"),
            (r"(\d+\.?\d*)\s*cigarettes?/(day|d)", "支/天"),
            (r"(\d+\.?\d*)\s*支/周", "支/周"),
            (r"(\d+\.?\d*)\s*mg/dl", "mg/dL"),
            (r"(\d+\.?\d*)\s*mmol/l", "mmol/L"),
            (r"(\d+\.?\d*)\s*cm", "cm"),
            (r"(\d+\.?\d*)\s*公斤", "kg"),
            (r"(\d+\.?\d*)\s*斤", "斤"),
            (r"(\d+\.?\d*)\s*°?c", "°C"),
            (r"(\d+\.?\d*)\s*°?f", "°F"),
        ]

        for pattern, unit in patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return unit
        return None

    def auto_convert(self, value: Any, from_unit: str, to_unit: str,
                     context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """自动查找并执行合适的转换规则

        参数：
            value: 待转换值
            from_unit: 源单位
            to_unit: 目标单位
            context: 上下文

        返回：
            dict | None: 转换结果
        """
        # 标准化单位
        from_unit = self._normalize_unit(from_unit)
        to_unit = self._normalize_unit(to_unit)

        if from_unit == to_unit:
            return {
                "value": float(value) if not isinstance(value, (int, float)) else value,
                "unit": to_unit,
                "rule_id": "identity",
                "rule_name": "无需转换",
            }

        # 查找匹配规则
        for rule_id, rule in {**CONVERSION_RULES, **self._custom_rules}.items():
            rule_from = self._normalize_unit(rule.get("from_unit", ""))
            rule_to = self._normalize_unit(rule.get("to_unit", ""))
            if rule_from == from_unit and rule_to == to_unit:
                return self.convert(value, rule_id, context)

        LOGGER.warning("未找到匹配转换规则: %s→%s", from_unit, to_unit)
        return None

    # ==================== 工具方法 ====================

    def _normalize_unit(self, unit: str) -> str:
        """标准化单位名称"""
        unit = unit.strip().lower()
        return UNIT_ALIASES.get(unit, unit)

    def list_conversions_between(self, from_unit: str, to_unit: str) -> List[Dict[str, Any]]:
        """列出两个单位之间的可用转换规则"""
        results = []
        from_norm = self._normalize_unit(from_unit)
        to_norm = self._normalize_unit(to_unit)
        for rule_id, rule in {**CONVERSION_RULES, **self._custom_rules}.items():
            r_from = self._normalize_unit(rule.get("from_unit", ""))
            r_to = self._normalize_unit(rule.get("to_unit", ""))
            if r_from == from_norm and r_to == to_norm:
                results.append({"id": rule_id, **rule})
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """获取转换器统计信息"""
        return {
            "builtin_rules": len(CONVERSION_RULES),
            "custom_rules": len(self._custom_rules),
            "total_rules": len(CONVERSION_RULES) + len(self._custom_rules),
            "unit_aliases": len(UNIT_ALIASES),
        }