#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核心字段映射引擎

支持：
1. 字段名映射（源字段 → 标准字段）
2. 单位转换（年龄月→岁、吸烟量转换等）
3. 编码映射（民族、职业、行政区划等）
4. 值范围映射（检验结果参考范围标准化）
5. Python 表达式转换
6. 条件映射（根据条件选择不同映射策略）
7. 映射结果验证与异常报告
"""

import ast
import copy
import datetime
import json
import logging
import operator
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

LOGGER = logging.getLogger("tb_risk.health_interop.mapping.engine")

# ============================================================================
# 安全表达式求值器（基于 AST 白名单，替代 eval）
# ============================================================================

# 允许的 AST 节点白名单
_ALLOWED_AST_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.FloorDiv,
    ast.Mod, ast.USub, ast.UAdd, ast.Compare, ast.Eq, ast.NotEq, ast.Lt,
    ast.Gt, ast.LtE, ast.GtE, ast.In, ast.NotIn, ast.List, ast.Tuple,
    ast.IfExp, ast.Attribute, ast.Slice, ast.keyword,
}

# 允许的内置函数白名单
_ALLOWED_BUILTINS = {
    "str": str, "int": int, "float": float, "round": round, "abs": abs,
    "min": min, "max": max, "len": len, "bool": bool, "list": list,
    "tuple": tuple, "dict": dict, "sum": sum, "pow": pow, "isinstance": isinstance,
    "type": type, "repr": repr, "ord": ord, "chr": chr,
}

# 不允许的属性访问路径（阻止沙箱逃逸）
_FORBIDDEN_ATTR_PREFIXES = ("__", "func_", "tb_", "gi_", "f_", "co_", "n_")


def _safe_eval(expr: str, variables: Dict[str, Any]) -> Any:
    """基于 AST 白名单的安全表达式求值

    替代 eval()，仅允许白名单内的 AST 节点和内置函数。
    通过 ast.parse() 解析表达式，人工验证 AST 节点类型，
    阻止 __class__、__subclasses__ 等沙箱逃逸路径。

    参数：
        expr: 表达式字符串（如 "round(float(x) / 12, 0)"）
        variables: 变量字典（如 {"x": 288}）

    返回：
        Any: 表达式求值结果

    抛出：
        ValueError: 表达式包含危险操作
        TypeError: 表达式语法错误
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {expr}") from e

    # 验证 AST 节点类型
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_AST_NODES:
            raise ValueError(
                f"不支持的表达式元素: {type(node).__name__} "
                f"(位置: {getattr(node, 'lineno', '?')}:{getattr(node, 'col_offset', '?')})"
            )
        # 阻止属性访问逃逸
        if isinstance(node, ast.Attribute):
            attr_name = node.attr
            if any(attr_name.startswith(prefix) for prefix in _FORBIDDEN_ATTR_PREFIXES):
                raise ValueError(f"不允许的属性访问: {attr_name}")

    # 编译并执行
    code = compile(tree, filename="<safe_eval>", mode="eval")
    safe_globals = {"__builtins__": _ALLOWED_BUILTINS}
    safe_locals = dict(variables)
    return eval(code, safe_globals, safe_locals)


# tb_risk 标准字段定义
STANDARD_FIELDS: Dict[str, Dict[str, Any]] = {
    # 患者基本信息
    "patient_name": {"type": "string", "required": False, "desc": "患者姓名"},
    "age": {"type": "int", "required": True, "desc": "年龄（岁）", "min": 0, "max": 120},
    "gender": {"type": "int", "required": True, "desc": "性别（0=女, 1=男）", "min": 0, "max": 1},
    "ethnicity": {"type": "string", "required": False, "desc": "民族编码"},
    "occupation": {"type": "string", "required": False, "desc": "职业"},
    "district": {"type": "string", "required": False, "desc": "行政区划编码"},
    # 临床指标
    "sputum_smear": {"type": "string", "required": True, "desc": "痰涂片结果（1=阴性, 2=阳性）"},
    "has_cavity": {"type": "string", "required": True, "desc": "有无空洞（1=无, 2=有）"},
    "active_tb": {"type": "string", "required": True, "desc": "活动性结核（1=是, 2=否）"},
    "treatment": {"type": "string", "required": True, "desc": "治疗状态（1=是, 2=否）"},
    "treatment_duration": {"type": "int", "required": False, "desc": "治疗时长（月）", "min": 0, "max": 24},
    "cough_freq": {"type": "int", "required": False, "desc": "咳嗽频率（次/小时）"},
    "symptoms": {"type": "string", "required": True, "desc": "症状严重度（1=无症状, 2=轻度, 3=中度, 4=重度）"},
    "delay_days": {"type": "int", "required": False, "desc": "延迟就诊天数"},
    "symptom_description": {"type": "string", "required": False, "desc": "症状描述"},
    # 暴露特征
    "ventilation": {"type": "string", "required": False, "desc": "通风条件（1-5）"},
    "family_living_conditions": {"type": "string", "required": False, "desc": "家庭居住条件（1-5）"},
    "flp_percentage": {"type": "float", "required": False, "desc": "家庭潜伏感染比例", "min": 0, "max": 100},
    "hrsp_percentage": {"type": "float", "required": False, "desc": "社会高风险人群LTBI比例", "min": 0, "max": 100},
    # 扩展特征
    "bcg_scar": {"type": "string", "required": False, "desc": "卡痕/BCG接种（1=有, 2=无）"},
    "smoking_years": {"type": "int", "required": False, "desc": "吸烟年数", "min": 0, "max": 80},
    "drinking_years": {"type": "int", "required": False, "desc": "饮酒年数", "min": 0, "max": 60},
    "bmi": {"type": "float", "required": False, "desc": "BMI指数", "min": 10, "max": 60},
    "height_cm": {"type": "float", "required": False, "desc": "身高（cm）"},
    "weight_kg": {"type": "float", "required": False, "desc": "体重（kg）"},
    "idu_history": {"type": "string", "required": False, "desc": "静脉吸毒史（1=是, 2=否）"},
    # 症状详情
    "symptom_cough": {"type": "string", "required": False, "desc": "咳嗽（1=有, 2=无）"},
    "symptom_sputum": {"type": "string", "required": False, "desc": "咳痰（1=有, 2=无）"},
    "symptom_hemoptysis": {"type": "string", "required": False, "desc": "咯血（1=有, 2=无）"},
    "symptom_fever": {"type": "string", "required": False, "desc": "发热（1=有, 2=无）"},
    "symptom_night_sweat": {"type": "string", "required": False, "desc": "盗汗（1=有, 2=无）"},
    "symptom_weight_loss": {"type": "string", "required": False, "desc": "体重下降（1=有, 2=无）"},
    "symptom_fatigue": {"type": "string", "required": False, "desc": "乏力（1=有, 2=无）"},
    # 合并症
    "past_tb_history": {"type": "string", "required": False, "desc": "既往结核史（1=有, 2=无）"},
    "comorbidity_diabetes": {"type": "string", "required": False, "desc": "糖尿病（1=有, 2=无）"},
    "comorbidity_hiv": {"type": "string", "required": False, "desc": "HIV（1=有, 2=无）"},
    "comorbidity_immunosuppression": {"type": "string", "required": False, "desc": "免疫抑制（1=有, 2=无）"},
}

# 标准字段列表（用于下拉选择）
STANDARD_FIELD_NAMES = sorted(STANDARD_FIELDS.keys())

# 预定义的转换函数
TRANSFORM_REGISTRY: Dict[str, Callable] = {}


def register_transform(name: str, func: Callable):
    """注册转换函数"""
    TRANSFORM_REGISTRY[name] = func


def get_transform(name: str) -> Optional[Callable]:
    """获取转换函数"""
    return TRANSFORM_REGISTRY.get(name)


class FieldMapping:
    """字段映射定义

    描述一个源字段到标准字段的完整映射规则。
    """

    def __init__(
        self,
        source_field: str,
        target_field: str,
        transforms: Optional[List[Dict[str, Any]]] = None,
        condition: Optional[Dict[str, Any]] = None,
        default_value: Any = None,
        description: str = "",
        confidence: float = 1.0,
    ):
        self.source_field = source_field
        self.target_field = target_field
        self.transforms = transforms or []
        self.condition = condition  # e.g. {"field": "age", "operator": ">", "value": 18}
        self.default_value = default_value
        self.description = description
        self.confidence = confidence  # 0-1，自动映射置信度

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_field": self.source_field,
            "target_field": self.target_field,
            "transforms": self.transforms,
            "condition": self.condition,
            "default_value": self.default_value,
            "description": self.description,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldMapping":
        return cls(
            source_field=data["source_field"],
            target_field=data["target_field"],
            transforms=data.get("transforms", []),
            condition=data.get("condition"),
            default_value=data.get("default_value"),
            description=data.get("description", ""),
            confidence=data.get("confidence", 1.0),
        )


class FieldMappingEngine:
    """核心字段映射引擎

    支持链式映射、单位转换、编码映射、条件映射、表达式计算。
    """

    def __init__(self):
        self._mappings: Dict[str, FieldMapping] = {}  # source_field -> mapping
        self._target_index: Dict[str, List[str]] = {}  # target_field -> [source_fields]
        self._unmapped_source: List[str] = []
        self._errors: List[Dict[str, Any]] = []
        self._warnings: List[Dict[str, Any]] = []
        self._unit_converter = None  # 延迟导入
        self._code_mapper = None

    # ==================== 映射配置 ====================

    def add_mapping(
        self,
        source_field: str,
        target_field: str,
        transforms: Optional[List[Dict[str, Any]]] = None,
        condition: Optional[Dict[str, Any]] = None,
        default_value: Any = None,
        description: str = "",
        confidence: float = 1.0,
    ):
        """添加字段映射

        参数：
            source_field: 源字段名（医院系统字段名）
            target_field: 目标字段名（tb_risk 标准字段名）
            transforms: 转换规则列表，每项：
                {"type": "unit_convert", "from": "月", "to": "岁", "formula": "x/12"}
                {"type": "code_map", "from_system": "hospital", "to_system": "standard", "mapping": {...}}
                {"type": "value_range", "from": {"low": 0, "high": 100}, "to": {"low": 0, "high": 10}}
                {"type": "expression", "expr": "float(x) * 0.1"}
                {"type": "text_extract", "pattern": r"(\\d+)"}
                {"type": "default", "value": "未知"}
            condition: 条件映射 {"field": "age", "operator": ">", "value": 60, "then_maps_to": ...}
            default_value: 默认值（源字段缺失时使用）
            confidence: 映射置信度（0-1）
        """
        mapping = FieldMapping(
            source_field=source_field,
            target_field=target_field,
            transforms=transforms or [],
            condition=condition,
            default_value=default_value,
            description=description,
            confidence=confidence,
        )
        self._mappings[source_field] = mapping
        self._target_index.setdefault(target_field, []).append(source_field)

    def add_mappings(self, mappings: List[Dict[str, Any]]):
        """批量添加映射"""
        for m in mappings:
            self.add_mapping(**m)

    def remove_mapping(self, source_field: str):
        """移除映射"""
        mapping = self._mappings.pop(source_field, None)
        if mapping:
            target = mapping.target_field
            sources = self._target_index.get(target, [])
            if source_field in sources:
                sources.remove(source_field)

    def clear(self):
        """清除所有映射"""
        self._mappings.clear()
        self._target_index.clear()

    def get_mapping(self, source_field: str) -> Optional[FieldMapping]:
        """获取映射"""
        return self._mappings.get(source_field)

    def get_mappings_for_target(self, target_field: str) -> List[FieldMapping]:
        """获取映射到某标准字段的所有映射"""
        source_fields = self._target_index.get(target_field, [])
        return [self._mappings[sf] for sf in source_fields if sf in self._mappings]

    def get_all_mappings(self) -> List[FieldMapping]:
        """获取所有映射"""
        return list(self._mappings.values())

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "mappings": [m.to_dict() for m in self._mappings.values()],
            "version": "1.0",
            "created_at": datetime.datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldMappingEngine":
        """反序列化"""
        engine = cls()
        for m_data in data.get("mappings", []):
            engine.add_mapping(**m_data)
        return engine

    # ==================== 执行映射 ====================

    def apply(
        self,
        record: Dict[str, Any],
        validate: bool = True,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """对单条记录执行映射

        参数：
            record: 源数据记录（医院系统字段名 → 值）
            validate: 是否验证必填字段
            strict: 严格模式（未映射字段抛出异常）

        返回：
            dict: 标准化后的记录（tb_risk 标准字段名 → 值）
        """
        self._errors.clear()
        self._warnings.clear()
        self._unmapped_source = []

        result = {}

        # 1. 执行已配置的映射
        for source_field, mapping in self._mappings.items():
            if source_field in record:
                raw_value = record[source_field]
                try:
                    transformed = self._apply_transforms(raw_value, mapping.transforms)
                    if transformed is not None:
                        result[mapping.target_field] = transformed
                except Exception as e:
                    self._errors.append({
                        "source_field": source_field,
                        "target_field": mapping.target_field,
                        "error": str(e),
                        "value": raw_value,
                    })
                    if strict:
                        raise
            elif mapping.default_value is not None:
                # 源字段缺失，使用默认值
                result[mapping.target_field] = mapping.default_value
                self._warnings.append({
                    "field": mapping.target_field,
                    "warning": f"源字段 '{source_field}' 缺失，使用默认值: {mapping.default_value}",
                })

        # 2. 记录未映射的源字段
        for field in record:
            if field not in self._mappings and not field.startswith("_"):
                self._unmapped_source.append(field)

        # 3. 验证必填字段
        if validate:
            validation_errors = self._validate_required(result)
            for err in validation_errors:
                self._errors.append(err)

        return result

    def apply_batch(
        self,
        records: List[Dict[str, Any]],
        validate: bool = True,
    ) -> List[Dict[str, Any]]:
        """批量执行映射"""
        return [self.apply(r, validate=validate) for r in records]

    # ==================== 转换引擎 ====================

    def _apply_transforms(self, value: Any, transforms: List[Dict[str, Any]]) -> Any:
        """应用转换链"""
        result = value
        for transform in transforms:
            t_type = transform.get("type", "")
            if t_type == "unit_convert":
                result = self._unit_convert(result, transform)
            elif t_type == "code_map":
                result = self._code_map(result, transform)
            elif t_type == "value_range":
                result = self._value_range_map(result, transform)
            elif t_type == "expression":
                result = self._eval_expression(result, transform)
            elif t_type == "text_extract":
                result = self._text_extract(result, transform)
            elif t_type == "default":
                result = self._apply_default(result, transform)
            elif t_type == "trim":
                result = str(result).strip() if result is not None else None
            elif t_type == "lower":
                result = str(result).lower() if result is not None else None
            elif t_type == "upper":
                result = str(result).upper() if result is not None else None
            elif t_type == "replace":
                result = self._apply_replace(result, transform)
            elif t_type == "round":
                if result is not None:
                    try:
                        decimals = transform.get("decimals", 0)
                        result = round(float(result), decimals)
                    except (TypeError, ValueError):
                        pass
            elif t_type == "custom":
                func_name = transform.get("function", "")
                func = get_transform(func_name)
                if func:
                    try:
                        result = func(result, **transform.get("params", {}))
                    except Exception as e:
                        LOGGER.warning("自定义转换函数 '%s' 执行失败: %s", func_name, e)
            if result is None:
                break
        return result

    def _unit_convert(self, value: Any, transform: Dict[str, Any]) -> Any:
        """单位转换（安全求值）"""
        if value is None or value == "":
            return None
        try:
            val = float(value)
            formula = transform.get("formula", "x")
            # 安全求值（基于 AST 白名单）
            result = _safe_eval(formula, {"x": val})
            return result
        except (TypeError, ValueError, SyntaxError, NameError) as e:
            LOGGER.warning("单位转换失败: value=%s, formula=%s, error=%s", value, transform.get("formula"), e)
            return value

    def _code_map(self, value: Any, transform: Dict[str, Any]) -> Any:
        """编码映射"""
        if value is None or value == "":
            return None
        mapping = transform.get("mapping", {})
        str_val = str(value).strip()
        # 直接映射
        if str_val in mapping:
            return mapping[str_val]
        # 大小写不敏感映射
        for k, v in mapping.items():
            if str_val.lower() == k.lower():
                return v
        # 子串映射
        if transform.get("fuzzy", False):
            for k, v in mapping.items():
                if k.lower() in str_val.lower() or str_val.lower() in k.lower():
                    return v
        # 未匹配
        default = transform.get("default", value)
        return default

    def _value_range_map(self, value: Any, transform: Dict[str, Any]) -> Any:
        """值范围映射（归一化）"""
        if value is None or value == "":
            return None
        try:
            val = float(value)
            from_range = transform.get("from", {})
            to_range = transform.get("to", {})
            from_low = from_range.get("low", 0)
            from_high = from_range.get("high", 100)
            to_low = to_range.get("low", 0)
            to_high = to_range.get("high", 10)

            if from_high == from_low:
                return to_low

            # 线性映射
            normalized = (val - from_low) / (from_high - from_low)
            result = to_low + normalized * (to_high - to_low)
            return round(result, transform.get("decimals", 2))
        except (TypeError, ValueError, ZeroDivisionError) as e:
            LOGGER.warning("值范围映射失败: %s", e)
            return value

    def _eval_expression(self, value: Any, transform: Dict[str, Any]) -> Any:
        """Python 表达式求值（安全求值，基于 AST 白名单）"""
        if value is None:
            return None
        try:
            expr = transform.get("expr", "x")
            result = _safe_eval(expr, {"x": value})
            return result
        except Exception as e:
            LOGGER.warning("表达式求值失败: expr=%s, value=%s, error=%s", transform.get("expr"), value, e)
            return value

    def _text_extract(self, value: Any, transform: Dict[str, Any]) -> Any:
        """文本正则提取"""
        if value is None or not isinstance(value, str):
            return None
        pattern = transform.get("pattern", "")
        try:
            match = re.search(pattern, value)
            if match:
                group = transform.get("group", 0)
                return match.group(group)
            return None
        except re.error as e:
            LOGGER.warning("正则提取失败: pattern=%s, error=%s", pattern, e)
            return value

    def _apply_default(self, value: Any, transform: Dict[str, Any]) -> Any:
        """默认值"""
        if value is None or value == "" or (isinstance(value, (int, float)) and value == 0 and transform.get("zero_as_missing", False)):
            return transform.get("value", value)
        return value

    def _apply_replace(self, value: Any, transform: Dict[str, Any]) -> Any:
        """字符串替换"""
        if value is None:
            return None
        old = transform.get("old", "")
        new = transform.get("new", "")
        if isinstance(value, str):
            return value.replace(old, new)
        return str(value).replace(old, new)

    # ==================== 验证 ====================

    def _validate_required(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """验证必填字段"""
        errors = []
        for field_name, field_def in STANDARD_FIELDS.items():
            if field_def.get("required", False):
                if field_name not in result or result[field_name] is None or result[field_name] == "":
                    errors.append({
                        "field": field_name,
                        "error": f"必填字段 '{field_name}' 缺失",
                        "severity": "error",
                    })
                elif field_def.get("type") == "int":
                    try:
                        val = int(result[field_name])
                        min_val = field_def.get("min")
                        max_val = field_def.get("max")
                        if min_val is not None and val < min_val:
                            errors.append({"field": field_name, "error": f"值 {val} 低于最小值 {min_val}", "severity": "warning"})
                        if max_val is not None and val > max_val:
                            errors.append({"field": field_name, "error": f"值 {val} 高于最大值 {max_val}", "severity": "warning"})
                    except (TypeError, ValueError):
                        errors.append({"field": field_name, "error": f"字段类型应为 int，实际值: {result[field_name]}", "severity": "error"})
        return errors

    def get_errors(self) -> List[Dict[str, Any]]:
        return self._errors

    def get_warnings(self) -> List[Dict[str, Any]]:
        return self._warnings

    def get_unmapped_fields(self) -> List[str]:
        return self._unmapped_source

    def get_report(self) -> Dict[str, Any]:
        """获取映射执行报告"""
        return {
            "total_mappings": len(self._mappings),
            "errors": self._errors,
            "warnings": self._warnings,
            "unmapped_source_fields": self._unmapped_source,
            "error_count": len(self._errors),
            "warning_count": len(self._warnings),
        }

    # ==================== 自动映射 ====================

    def auto_map(self, source_fields: List[str], similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """自动推断字段映射（基于字段名相似度）

        参数：
            source_fields: 源字段名列表
            similarity_threshold: 相似度阈值

        返回：
            list[dict]: 建议的映射列表
        """
        suggestions = []
        for sf in source_fields:
            best_match = None
            best_score = 0
            for tf_name, tf_def in STANDARD_FIELDS.items():
                score = self._field_similarity(sf, tf_name)
                if score > best_score and score >= similarity_threshold:
                    best_score = score
                    best_match = tf_name

            if best_match:
                suggestions.append({
                    "source_field": sf,
                    "target_field": best_match,
                    "confidence": round(best_score, 2),
                    "auto": True,
                })

        return suggestions

    @staticmethod
    def _field_similarity(field1: str, field2: str) -> float:
        """计算字段名相似度（基于字符重叠和拼音）"""
        s1 = field1.lower().replace("_", "").replace(" ", "").replace("-", "")
        s2 = field2.lower().replace("_", "").replace(" ", "").replace("-", "")

        if s1 == s2:
            return 1.0

        # 包含关系
        if s1 in s2 or s2 in s1:
            longer = max(len(s1), len(s2))
            shorter = min(len(s1), len(s2))
            return shorter / longer * 0.8

        # 字符重叠
        common = set(s1) & set(s2)
        if len(common) == 0:
            return 0.0
        overlap = len(common) / max(len(set(s1)), len(set(s2)))
        return overlap * 0.5

    @staticmethod
    def get_standard_fields() -> Dict[str, Dict[str, Any]]:
        """获取标准字段定义"""
        return dict(STANDARD_FIELDS)

    @staticmethod
    def get_standard_field_names() -> List[str]:
        """获取标准字段名列表"""
        return STANDARD_FIELD_NAMES