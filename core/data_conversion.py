#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据转换工具函数

中文文本映射、类型转换、字段标准化、场景默认值等数据转换功能。
所有函数零依赖 tkinter 和 GUI。

文本映射常量从 tb_risk.constants 导入（单一真值源）。
"""

import math

from ..constants import (
    DISTANCE_TEXT_MAP,
    SETTING_TEXT_MAP,
    RATING_TEXT_MAP,
    BOOL_TEXT_MAP,
)

# 场景默认值
SCENE_DEFAULTS = {
    'family': {
        'contact_distance': 'close',
        'ventilation': 3,
        'exposure_setting': 'general',
        'freq_density': 14,
        'single_duration': 120,
        'time_span': 4,
    },
    'social': {
        'contact_distance': 'medium',
        'ventilation': 3,
        'exposure_setting': 'general',
        'freq_density': 2,
        'single_duration': 60,
        'time_span': 4,
    },
}


def convert_chinese_to_value(chinese_val, mapping):
    """将中文文本值转换为对应的枚举值

    参数：
        chinese_val: str — 中文文本值
        mapping: dict — 中文 → 枚举值映射表

    返回：
        Any: 映射后的值，若未找到则返回原值
    """
    if chinese_val is None:
        return None
    if isinstance(chinese_val, str):
        chinese_val = chinese_val.strip()
        # 精确匹配
        if chinese_val in mapping:
            return mapping[chinese_val]
        # 模糊匹配：检查是否包含关键词
        for key, value in mapping.items():
            if key in chinese_val or chinese_val in key:
                return value
    return chinese_val


def standardize_contact_fields(contact, contact_type='family'):
    """标准化接触者字段格式

    将中文字段值转换为英文枚举值，确保数据格式一致。

    参数：
        contact: dict — 接触者数据
        contact_type: str — 'family' 或 'social'

    返回：
        dict: 标准化后的数据
    """
    # 距离标准化
    if 'contact_distance' in contact:
        contact['contact_distance'] = convert_chinese_to_value(
            contact['contact_distance'], DISTANCE_TEXT_MAP)

    # 场景标准化
    if 'exposure_setting' in contact:
        contact['exposure_setting'] = convert_chinese_to_value(
            contact['exposure_setting'], SETTING_TEXT_MAP)

    # 通风条件标准化
    if 'ventilation' in contact:
        contact['ventilation'] = convert_chinese_to_value(
            contact['ventilation'], RATING_TEXT_MAP)

    # 布尔字段标准化
    for field in ['has_tb', 'has_symptoms', 'bcg_vaccine', 'past_illness',
                  'is_high_risk', 'idu_status']:
        if field in contact:
            converted = convert_chinese_to_value(contact[field], BOOL_TEXT_MAP)
            if converted is not None:
                contact[field] = converted

    return contact


def apply_scenario_defaults(contact, contact_type='family'):
    """应用场景默认值

    填充缺失字段为场景默认值。

    参数：
        contact: dict — 接触者数据
        contact_type: str — 'family' 或 'social'

    返回：
        dict: 填充后的数据
    """
    defaults = SCENE_DEFAULTS.get(contact_type, SCENE_DEFAULTS['family'])
    for key, default_val in defaults.items():
        if key not in contact or contact[key] is None:
            contact[key] = default_val
    return contact


def get_priority_and_recommendation(disease_probability):
    """根据疾病概率确定优先级和推荐筛查方案

    参数：
        disease_probability: float — 疾病概率 (%)

    返回：
        tuple[str, str]: (优先级, 推荐方案)
    """
    if disease_probability >= 80:
        return '极高', '立即进行胸部X线+分子生物学检测+痰培养，建议隔离观察'
    elif disease_probability >= 60:
        return '高', '立即进行胸部X线+IGRA联合筛查'
    elif disease_probability >= 40:
        return '较高', '建议进行胸部X线+PPD/IGRA筛查'
    elif disease_probability >= 20:
        return '中', '建议进行PPD/IGRA筛查'
    elif disease_probability >= 10:
        return '低', '选择性进行PPD/IGRA筛查'
    else:
        return '极低', '常规随访观察，有症状时及时就医'


def calculate_age_group(age):
    """计算年龄分组

    用于数据脱敏的年龄泛化。

    参数：
        age: float | int — 精确年龄

    返回：
        str: 年龄段标签
    """
    try:
        age = float(age)
    except (ValueError, TypeError):
        return '未知'
    if age < 6:
        return '0-5'
    elif age < 16:
        return '6-15'
    elif age < 36:
        return '16-35'
    elif age < 56:
        return '36-55'
    elif age < 66:
        return '56-65'
    else:
        return '65+'