#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据导入增强包

提供统一的字段映射、文本抽取、缺失值填充、质量评分和审计能力。

版本：2.0.0
"""

__version__ = "2.0.0"

# ==============================================================================
# 常量
# ==============================================================================

from .conf import (
    CLINICAL_TEXT_MAP, DISTANCE_TEXT_MAP, SETTING_TEXT_MAP,
    RATING_TEXT_MAP, BOOL_TEXT_MAP,
    BOOLEAN_FIELDS, CLINICAL_FIELDS, RATING_FIELDS, NUMERIC_FIELDS,
    DEFAULT_VALUES, CONTACT_TYPE_OVERRIDES, SCENE_DEFAULTS,
    FIELD_RANGES, MISSING_VALUE_MARKER,
)

# ==============================================================================
# 同义词
# ==============================================================================

from .synonyms import (
    FIELD_SYNONYM_LIBRARY, SYNONYM_CONFIDENCE,
    fuzzy_match_column, normalize_name, dynamic_column_match,
    register_synonym_feedback, get_field_synonyms,
    get_field_synonyms_with_confidence,
)

# ==============================================================================
# 文本抽取
# ==============================================================================

from .text_parser import (
    parse_text_input, extract_fields_from_text, split_paragraphs,
    load_templates, detect_input_type,
)

# ==============================================================================
# 缺失值填充
# ==============================================================================

from .imputation import (
    impute_missing, impute_mode, impute_conditional_median,
    impute_scene_default, fill_record_defaults,
    mark_unfillable, fill_missing_fields_batch,
)

# ==============================================================================
# 质量评分
# ==============================================================================

from .scoring import DataQualityScorer, SOURCE_CONFIDENCE, QUALITY_THRESHOLDS

# ==============================================================================
# 审计
# ==============================================================================

from .audit import ImportAuditor

# ==============================================================================
# 管线编排
# ==============================================================================

from .pipeline import import_pipeline, quick_import, import_with_feedback, PipelineResult