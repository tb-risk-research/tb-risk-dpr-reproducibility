#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""医学术语标准编码体系

提供 ICD-10（疾病编码）、ICD-9-CM-3（手术操作编码）、LOINC（检验项目编码）、
SNOMED CT（临床术语）、国标版卫生信息数据元目录（GB/T 3304/6565/2260）等
标准术语的自动映射与查询能力。

核心入口：
    from .icd10_mapper import ICD10Mapper
    from .loinc_mapper import LOINCMapper
    from .diagnosis_matcher import DiagnosisMatcher

用法示例：
    mapper = ICD10Mapper()
    result = mapper.match_diagnosis("2型糖尿病")
    # {"code": "E11", "display": "2型糖尿病", "confidence": 0.95, ...}
"""

from .icd10_mapper import ICD10Mapper
from .icd9cm_mapper import ICD9CMMapper
from .loinc_mapper import LOINCMapper
from .snomed_mapper import SNOMEDMapper
from .gb_mapper import GBCodeMapper
from .diagnosis_matcher import DiagnosisMatcher

__all__ = [
    "ICD10Mapper",
    "ICD9CMMapper",
    "LOINCMapper",
    "SNOMEDMapper",
    "GBCodeMapper",
    "DiagnosisMatcher",
]