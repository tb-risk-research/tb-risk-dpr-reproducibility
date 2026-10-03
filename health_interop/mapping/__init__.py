#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化字段映射配置引擎

提供医院系统字段到 tb_risk 标准特征的完整映射能力：
1. FieldMappingEngine — 核心字段映射引擎
2. UnitConverter — 单位转换系统
3. CodeMapper — 编码/值映射
4. MappingTemplate — 映射模板管理与序列化
5. ValueRangeNormalizer — 检验值参考范围标准化

用法示例：
    from .field_mapping_engine import FieldMappingEngine

    engine = FieldMappingEngine()
    engine.add_mapping("患者姓名", "patient_name")
    engine.add_mapping("年龄", "age", transforms=[
        {"type": "unit_convert", "from": "月", "to": "岁", "formula": "x/12"}
    ])
    result = engine.apply({"患者姓名": "张三", "年龄": "360"})
"""

from .field_mapping_engine import FieldMappingEngine
from .unit_converter import UnitConverter
from .code_mapper import CodeMapper
from .template_manager import MappingTemplateManager
from .value_range_normalizer import ValueRangeNormalizer

__all__ = [
    "FieldMappingEngine",
    "UnitConverter",
    "CodeMapper",
    "MappingTemplateManager",
    "ValueRangeNormalizer",
]