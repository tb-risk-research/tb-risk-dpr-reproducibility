#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据导入统一常量配置

从 tb_risk.constants 导入文本映射常量（单一真值源），消除多处重复定义。

文献参考：
- HL7 FHIR R4 结核病资源模型
- 中国结核病临床数据字典 (2023)
"""

from ..constants import (
    CLINICAL_TEXT_MAP, DISTANCE_TEXT_MAP, SETTING_TEXT_MAP,
    RATING_TEXT_MAP, BOOL_TEXT_MAP,
)  # 重新导出供 data_io.__init__ 使用

# ==============================================================================
# 二、字段值域范围
# ==============================================================================

FIELD_RANGES = {
    'age':           {'min': 0,   'max': 120, 'type': 'int'},
    'member_age':    {'min': 0,   'max': 120, 'type': 'int'},
    'contact_age':   {'min': 0,   'max': 120, 'type': 'int'},
    'single_duration':  {'min': 0,   'max': 480,  'type': 'int'},
    'freq_density':  {'min': 0,   'max': 30,   'type': 'int'},
    'time_span':     {'min': 1,   'max': 52,   'type': 'int'},
    'ventilation':   {'min': 1,   'max': 5,    'type': 'int'},
    'symptoms':      {'min': 1,   'max': 4,    'type': 'int'},
    'cough_freq':    {'min': 0,   'max': 100,  'type': 'int'},
    'treatment_duration': {'min': 0, 'max': 24, 'type': 'int'},
    'delay_days':    {'min': 0,   'max': 365,  'type': 'int'},
    'flp_percentage': {'min': 0,  'max': 100,  'type': 'int'},
    'hrsp_percentage': {'min': 0, 'max': 100,  'type': 'int'},
    'family_living_conditions': {'min': 1, 'max': 5, 'type': 'int'},
    'cumulative_exposure': {'min': 0, 'max': float('inf'), 'type': 'float'},
    'sputum_smear':  {'min': 1,   'max': 2,    'type': 'int'},
    'has_cavity':    {'min': 1,   'max': 2,    'type': 'int'},
    'active_tb':     {'min': 1,   'max': 2,    'type': 'int'},
    'treatment':     {'min': 1,   'max': 2,    'type': 'int'},
}

# ==============================================================================
# 三、字段类型分类
# ==============================================================================

BOOLEAN_FIELDS = [
    'has_tb', 'has_symptoms', 'bcg_vaccine', 'past_illness', 'is_high_risk',
]

CLINICAL_FIELDS = [
    'sputum_smear', 'has_cavity', 'active_tb', 'treatment',
]

RATING_FIELDS = [
    'ventilation', 'family_living_conditions', 'symptoms',
]

NUMERIC_FIELDS = [
    'age', 'member_age', 'contact_age', 'single_duration', 'freq_density',
    'time_span', 'treatment_duration', 'cough_freq', 'delay_days',
    'flp_percentage', 'hrsp_percentage', 'cumulative_exposure',
]

# ==============================================================================
# 四、默认值
# ==============================================================================

DEFAULT_VALUES = {
    'single_duration': 60,
    'freq_density': 2,
    'time_span': 4,
    'has_tb': 0,
    'has_symptoms': 0,
    'bcg_vaccine': 0,
    'past_illness': 0,
    'past_illness_type': 'none',
    'contact_distance': 'medium',
    'ventilation': 3,
    'exposure_setting': 'general',
    'is_high_risk': 0,
}

# 按接触者类型的默认值覆盖
CONTACT_TYPE_OVERRIDES = {
    'family': {
        'freq_density': 14,
        'single_duration': 120,
        'contact_distance': 'close',
    },
    'social': {
        'freq_density': 2,
        'single_duration': 60,
        'contact_distance': 'medium',
    },
}

# 缺失值标记
MISSING_VALUE_MARKER = -999

# ==============================================================================
# 五、场景默认值
# ==============================================================================

SCENE_DEFAULTS = {
    'rural_family': {
        'name': '农村家庭',
        'patient': {
            'ventilation': 3,
            'family_living_conditions': 2,
            'flp_percentage': 10,
            'hrsp_percentage': 8,
        },
        'family_member': {
            'contact_distance': 'close',
            'ventilation': 3,
            'exposure_setting': 'general',
            'freq_density': 14,
            'single_duration': 120,
            'time_span': 4,
        },
        'social_contact': {
            'contact_distance': 'close',
            'ventilation': 3,
            'exposure_setting': 'general',
            'freq_density': 3,
            'single_duration': 60,
            'time_span': 4,
        },
    },
    'urban_workplace': {
        'name': '城市职场',
        'patient': {
            'ventilation': 4,
            'family_living_conditions': 3,
            'flp_percentage': 8,
            'hrsp_percentage': 15,
        },
        'family_member': {
            'contact_distance': 'close',
            'ventilation': 4,
            'exposure_setting': 'general',
            'freq_density': 14,
            'single_duration': 90,
            'time_span': 4,
        },
        'social_contact': {
            'contact_distance': 'medium',
            'ventilation': 4,
            'exposure_setting': 'closed',
            'freq_density': 5,
            'single_duration': 480,
            'time_span': 4,
        },
    },
    'school': {
        'name': '学校',
        'patient': {
            'ventilation': 3,
            'family_living_conditions': 3,
            'flp_percentage': 5,
            'hrsp_percentage': 12,
        },
        'family_member': {
            'contact_distance': 'close',
            'ventilation': 3,
            'exposure_setting': 'general',
            'freq_density': 14,
            'single_duration': 120,
            'time_span': 4,
        },
        'social_contact': {
            'contact_distance': 'very_close',
            'ventilation': 3,
            'exposure_setting': 'crowded',
            'freq_density': 5,
            'single_duration': 360,
            'time_span': 4,
        },
    },
    'oilfield_camp': {
        'name': '油田营地',
        'patient': {
            'ventilation': 2,        # 营地帐篷通风较差
            'family_living_conditions': 2,
            'flp_percentage': 12,    # 油田营地居住密度较高
            'hrsp_percentage': 10,
        },
        'family_member': {
            'contact_distance': 'close',
            'ventilation': 2,
            'exposure_setting': 'oilfield_camp',
            'freq_density': 21,      # 每日密切接触
            'single_duration': 180,  # 较高接触时长
            'time_span': 4,
        },
        'social_contact': {
            'contact_distance': 'close',
            'ventilation': 2,
            'exposure_setting': 'oilfield_camp',
            'freq_density': 7,
            'single_duration': 240,
            'time_span': 4,
        },
    },
}