#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""字段同义词库增强模块

提供三层分层同义词库、Levenshtein 模糊匹配和用户反馈学习机制。

参考来源：
- 中国结核病临床数据字典 (2023)
- HL7 FHIR R4 结核病资源模型
- WHO TB Data Dictionary
"""

import difflib
import json
import logging
import os
import re

LOGGER = logging.getLogger("tb_risk.data_io.synonyms")

# ==============================================================================
# 可选依赖：Levenshtein C 扩展
# ==============================================================================

try:
    from Levenshtein import distance as _lev_distance
    _LEVENSHTEIN_AVAILABLE = True
except ImportError:
    _LEVENSHTEIN_AVAILABLE = False

# ==============================================================================
# 用户反馈持久化路径
# ==============================================================================

_FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), 'synonym_feedback.json')


# ==============================================================================
# 一、同义词来源类型 → 置信度
# ==============================================================================

SYNONYM_CONFIDENCE = {
    'zh_clinical': 1.0,    # 中文临床俗称（如"涂阳""痰检阳性"）
    'en': 1.0,             # 英文标准名
    'zh_full': 0.95,       # 中文全称（如"患者年龄"）
    'en_field': 0.95,      # 英文字段全称（如"pulmonary_tuberculosis_positive"）
    'zh_short': 0.90,      # 中文简称（如"痰检"）
    'en_short': 0.90,      # 英文缩写（如"PTB+"）
    'en_abbr': 0.85,       # 英文缩写变体
    'zh_colloquial': 0.80, # 中文口语化表达
    'user_feedback': 0.95, # 用户反馈学习
}


# ==============================================================================
# 二、分层同义词库
# ==============================================================================

FIELD_SYNONYM_LIBRARY = {
    # 患者基本信息
    'age': [
        ('年龄', 'zh_clinical', 1.0),
        ('age', 'en', 1.0),
        ('patient_age', 'en_field', 0.95),
        ('患者年龄', 'zh_full', 0.95),
        ('周岁', 'zh_short', 0.85),
        ('岁数', 'zh_colloquial', 0.80),
        ('年纪', 'zh_colloquial', 0.80),
        ('Age', 'en', 0.95),
    ],
    'gender': [
        ('性别', 'zh_clinical', 1.0),
        ('gender', 'en', 1.0),
        ('sex', 'en_short', 0.90),
        ('男女性别', 'zh_full', 0.95),
        ('Gender', 'en', 0.95),
        ('Sex', 'en_short', 0.90),
    ],
    'ethnicity': [
        ('民族', 'zh_clinical', 1.0),
        ('ethnicity', 'en', 1.0),
        ('race', 'en_short', 0.85),
        ('种族', 'zh_full', 0.95),
        ('Ethnicity', 'en', 0.95),
    ],
    'occupation': [
        ('职业', 'zh_clinical', 1.0),
        ('occupation', 'en', 1.0),
        ('job', 'en_short', 0.85),
        ('工作', 'zh_short', 0.85),
        ('Occupation', 'en', 0.95),
    ],
    'district': [
        ('区县', 'zh_clinical', 1.0),
        ('district', 'en', 1.0),
        ('区域', 'zh_short', 0.85),
        ('地区', 'zh_short', 0.85),
        ('county', 'en_short', 0.85),
        ('所属区域', 'zh_full', 0.90),
        ('District', 'en', 0.95),
    ],

    # 临床字段
    'sputum_smear': [
        ('痰涂片', 'zh_clinical', 1.0),
        ('sputum_smear', 'en', 1.0),
        ('sputum', 'en_short', 0.90),
        ('smear', 'en_abbr', 0.85),
        ('涂片结果', 'zh_full', 0.95),
        ('痰检', 'zh_short', 0.90),
        ('痰检阳性', 'zh_clinical', 0.85),
        ('PTB+', 'en_abbr', 0.80),
        ('pulmonary_tuberculosis_positive', 'en_field', 0.85),
        ('SputumSmear', 'en', 0.95),
        ('Sputum', 'en_short', 0.90),
    ],
    'has_cavity': [
        ('空洞', 'zh_clinical', 1.0),
        ('has_cavity', 'en', 1.0),
        ('cavity', 'en_short', 0.90),
        ('有无空洞', 'zh_full', 0.95),
        ('肺部空洞', 'zh_full', 0.90),
        ('Cavity', 'en_short', 0.90),
        ('HasCavity', 'en', 0.95),
    ],
    'active_tb': [
        ('活动性', 'zh_clinical', 1.0),
        ('active_tb', 'en', 1.0),
        ('active', 'en_short', 0.85),
        ('活动性结核', 'zh_full', 0.95),
        ('是否活动性', 'zh_full', 0.90),
        ('ActiveTB', 'en', 0.95),
        ('Active', 'en_short', 0.85),
    ],
    'treatment': [
        ('治疗', 'zh_clinical', 1.0),
        ('treatment', 'en', 1.0),
        ('是否治疗', 'zh_full', 0.90),
        ('接受治疗', 'zh_full', 0.90),
        ('Treatment', 'en', 0.95),
        ('treated', 'en_short', 0.85),
    ],
    'treatment_duration': [
        ('治疗时长', 'zh_clinical', 1.0),
        ('treatment_duration', 'en', 1.0),
        ('治疗月数', 'zh_short', 0.90),
        ('治疗时间', 'zh_short', 0.85),
        ('疗程', 'zh_short', 0.85),
        ('TreatmentDuration', 'en', 0.95),
        ('treat_dur', 'en_abbr', 0.80),
    ],
    'cough_freq': [
        ('咳嗽', 'zh_clinical', 1.0),
        ('cough_freq', 'en', 1.0),
        ('cough', 'en_short', 0.90),
        ('咳嗽频率', 'zh_full', 0.95),
        ('cough_frequency', 'en_field', 0.90),
        ('咳嗽频次', 'zh_full', 0.90),
        ('CoughFreq', 'en', 0.95),
        ('Cough', 'en_short', 0.90),
    ],
    'symptoms': [
        ('症状', 'zh_clinical', 1.0),
        ('symptoms', 'en', 1.0),
        ('symptom', 'en_short', 0.90),
        ('症状严重', 'zh_full', 0.90),
        ('症状程度', 'zh_full', 0.90),
        ('症状表现', 'zh_full', 0.85),
        ('Symptoms', 'en', 0.95),
        ('Symptom', 'en_short', 0.90),
    ],
    'delay_days': [
        ('延迟', 'zh_clinical', 1.0),
        ('delay_days', 'en', 1.0),
        ('delay', 'en_short', 0.90),
        ('延迟就诊', 'zh_full', 0.95),
        ('就诊延迟', 'zh_full', 0.90),
        ('ftd', 'en_abbr', 0.85),
        ('DelayDays', 'en', 0.95),
        ('FTD', 'en_abbr', 0.85),
    ],
    'family_living_conditions': [
        ('居住条件', 'zh_clinical', 1.0),
        ('family_living_conditions', 'en', 1.0),
        ('family_living', 'en_short', 0.85),
        ('living_condition', 'en_field', 0.90),
        ('住房条件', 'zh_short', 0.90),
        ('居住环境', 'zh_short', 0.85),
        ('FamilyLivingConditions', 'en', 0.95),
    ],
    'flp_percentage': [
        ('家庭潜伏', 'zh_clinical', 1.0),
        ('flp_percentage', 'en', 1.0),
        ('flp', 'en_short', 0.90),
        ('family_latent', 'en_field', 0.90),
        ('潜伏感染比例', 'zh_full', 0.95),
        ('家庭潜伏率', 'zh_full', 0.90),
        ('FLP', 'en_short', 0.90),
    ],
    'hrsp_percentage': [
        ('社会高风险', 'zh_clinical', 1.0),
        ('hrsp_percentage', 'en', 1.0),
        ('hrsp', 'en_short', 0.90),
        ('high_risk_social', 'en_field', 0.90),
        ('高危人群比例', 'zh_full', 0.95),
        ('社会高危率', 'zh_full', 0.90),
        ('HRSP', 'en_short', 0.90),
    ],
    'ventilation': [
        ('通风', 'zh_clinical', 1.0),
        ('ventilation', 'en', 1.0),
        ('vent', 'en_short', 0.85),
        ('通风条件', 'zh_full', 0.95),
        ('空气流通', 'zh_full', 0.85),
        ('通风情况', 'zh_full', 0.85),
        ('Ventilation', 'en', 0.95),
        ('Vent', 'en_short', 0.85),
    ],

    # 接触者信息
    'single_duration': [
        ('接触时长', 'zh_clinical', 1.0),
        ('single_duration', 'en', 1.0),
        ('duration', 'en_short', 0.85),
        ('单次时长', 'zh_short', 0.90),
        ('接触时间', 'zh_short', 0.85),
        ('单次接触', 'zh_short', 0.85),
        ('SingleDuration', 'en', 0.95),
    ],
    'freq_density': [
        ('频次', 'zh_clinical', 1.0),
        ('freq_density', 'en', 1.0),
        ('freq', 'en_short', 0.90),
        ('frequency', 'en_field', 0.90),
        ('频率', 'zh_short', 0.85),
        ('每周频次', 'zh_full', 0.95),
        ('接触频率', 'zh_full', 0.90),
        ('每周次数', 'zh_full', 0.85),
        ('FreqDensity', 'en', 0.95),
    ],
    'time_span': [
        ('持续时间', 'zh_clinical', 1.0),
        ('time_span', 'en', 1.0),
        ('时间跨度', 'zh_full', 0.90),
        ('周期', 'zh_short', 0.85),
        ('周数', 'zh_short', 0.85),
        ('接触周期', 'zh_full', 0.90),
        ('span', 'en_short', 0.85),
        ('接触持续周期', 'zh_full', 0.90),
        ('TimeSpan', 'en', 0.95),
    ],
    'has_symptoms': [
        ('有症状', 'zh_clinical', 1.0),
        ('has_symptoms', 'en', 1.0),
        ('症状表现', 'zh_full', 0.85),
        ('是否有症状', 'zh_full', 0.90),
        ('出现症状', 'zh_full', 0.85),
        ('HasSymptoms', 'en', 0.95),
    ],
    'bcg_vaccine': [
        ('卡介苗', 'zh_clinical', 1.0),
        ('bcg_vaccine', 'en', 1.0),
        ('bcg', 'en_short', 0.90),
        ('疫苗', 'zh_short', 0.85),
        ('接种', 'zh_short', 0.80),
        ('BCG接种', 'zh_full', 0.95),
        ('卡介苗接种史', 'zh_full', 0.95),
        ('BCG', 'en_short', 0.90),
    ],
    'has_tb': [
        ('结核史', 'zh_clinical', 1.0),
        ('has_tb', 'en', 1.0),
        ('tb_history', 'en_field', 0.95),
        ('既往结核', 'zh_full', 0.95),
        ('结核病史', 'zh_full', 0.95),
        ('曾患结核', 'zh_full', 0.90),
        ('has_tuberculosis', 'en_field', 0.90),
        ('HasTB', 'en', 0.95),
    ],
    'past_illness': [
        ('慢性病', 'zh_clinical', 1.0),
        ('past_illness', 'en', 1.0),
        ('慢性疾病', 'zh_full', 0.95),
        ('其他疾病', 'zh_full', 0.90),
        ('合并症', 'zh_short', 0.90),
        ('基础疾病', 'zh_short', 0.90),
        ('other_illness', 'en_field', 0.85),
        ('病史', 'zh_short', 0.85),
        ('PastIllness', 'en', 0.95),
    ],
    'past_illness_type': [
        ('疾病类型', 'zh_clinical', 1.0),
        ('past_illness_type', 'en', 1.0),
        ('慢性病类型', 'zh_full', 0.90),
        ('合并症类型', 'zh_full', 0.90),
        ('illness_type', 'en_field', 0.90),
        ('病名', 'zh_short', 0.85),
        ('疾病种类', 'zh_full', 0.90),
        ('PastIllnessType', 'en', 0.95),
    ],
    'is_high_risk': [
        ('高危', 'zh_clinical', 1.0),
        ('is_high_risk', 'en', 1.0),
        ('high_risk', 'en_field', 0.95),
        ('高风险', 'zh_full', 0.95),
        ('高危人群', 'zh_full', 0.90),
        ('高危标识', 'zh_full', 0.85),
        ('是否高危人群', 'zh_full', 0.90),
        ('IsHighRisk', 'en', 0.95),
    ],
    'contact_distance': [
        ('距离', 'zh_clinical', 1.0),
        ('contact_distance', 'en', 1.0),
        ('distance', 'en_short', 0.90),
        ('接触距离', 'zh_full', 0.95),
        ('距离等级', 'zh_full', 0.85),
        ('ContactDistance', 'en', 0.95),
        ('Distance', 'en_short', 0.90),
    ],
    'exposure_setting': [
        ('场景', 'zh_clinical', 1.0),
        ('exposure_setting', 'en', 1.0),
        ('setting', 'en_short', 0.85),
        ('暴露场景', 'zh_full', 0.95),
        ('场所', 'zh_short', 0.85),
        ('环境', 'zh_short', 0.80),
        ('环境类型', 'zh_full', 0.85),
        ('ExposureSetting', 'en', 0.95),
        ('Setting', 'en_short', 0.85),
    ],

    # 标签与ID
    'is_confirmed': [
        ('确诊', 'zh_clinical', 1.0),
        ('is_confirmed', 'en', 1.0),
        ('确诊标识', 'zh_full', 0.90),
        ('是否确诊', 'zh_full', 0.90),
        ('诊断', 'zh_short', 0.85),
        ('diagnosed', 'en_field', 0.90),
        ('tb_outcome', 'en_field', 0.85),
        ('tb_diagnosed', 'en_field', 0.85),
        ('diagnosis', 'en_short', 0.85),
        ('label', 'en_short', 0.80),
        ('target', 'en_short', 0.80),
        ('IsConfirmed', 'en', 0.95),
    ],
    'record_id': [
        ('编号', 'zh_clinical', 1.0),
        ('record_id', 'en', 1.0),
        ('id', 'en_short', 0.90),
        ('序号', 'zh_short', 0.85),
        ('记录编号', 'zh_full', 0.90),
        ('样本编号', 'zh_full', 0.90),
        ('patient_id', 'en_field', 0.90),
        ('RecordID', 'en', 0.95),
    ],

    # 其他字段
    'cumulative_exposure': [
        ('累积暴露', 'zh_clinical', 1.0),
        ('cumulative_exposure', 'en', 1.0),
        ('暴露累积', 'zh_full', 0.90),
        ('总暴露时长', 'zh_full', 0.90),
        ('累计暴露', 'zh_full', 0.90),
        ('exposure_hours', 'en_field', 0.85),
        ('CumulativeExposure', 'en', 0.95),
    ],
    'scenario_type': [
        ('场景类型', 'zh_clinical', 1.0),
        ('scenario_type', 'en', 1.0),
        ('scenario', 'en_short', 0.90),
        ('场景', 'zh_short', 0.85),
        ('Scene', 'en_short', 0.85),
        ('ScenarioType', 'en', 0.95),
    ],
    'member_name': [
        ('姓名', 'zh_clinical', 1.0),
        ('member_name', 'en', 1.0),
        ('name', 'en_short', 0.90),
        ('名字', 'zh_short', 0.90),
        ('名称', 'zh_short', 0.85),
        ('成员姓名', 'zh_full', 0.95),
        ('家庭成员', 'zh_full', 0.85),
        ('MemberName', 'en', 0.95),
    ],
    'member_age': [
        ('成员年龄', 'zh_clinical', 1.0),
        ('member_age', 'en', 1.0),
        ('成员岁数', 'zh_full', 0.85),
        ('成员年纪', 'zh_full', 0.85),
        ('MemberAge', 'en', 0.95),
    ],
    'contact_name': [
        ('接触者姓名', 'zh_clinical', 1.0),
        ('contact_name', 'en', 1.0),
        ('接触人', 'zh_short', 0.90),
        ('ContactName', 'en', 0.95),
    ],
    'contact_age': [
        ('接触者年龄', 'zh_clinical', 1.0),
        ('contact_age', 'en', 1.0),
        ('接触人岁数', 'zh_full', 0.85),
        ('ContactAge', 'en', 0.95),
    ],
    'relationship': [
        ('关系', 'zh_clinical', 1.0),
        ('relationship', 'en', 1.0),
        ('与患者关系', 'zh_full', 0.95),
        ('与患者的关系', 'zh_full', 0.90),
        ('亲属关系', 'zh_full', 0.85),
        ('Relationship', 'en', 0.95),
    ],
    'origin_altitude': [
        ('原籍海拔', 'zh_clinical', 1.0),
        ('origin_altitude', 'en', 1.0),
        ('海拔', 'zh_short', 0.90),
        ('altitude', 'en_short', 0.90),
        ('Altitude', 'en_short', 0.90),
        ('OriginAltitude', 'en', 0.95),
    ],
    'workplace_type': [
        ('工作场景', 'zh_clinical', 1.0),
        ('workplace_type', 'en', 1.0),
        ('workplace', 'en_short', 0.90),
        ('工作场所', 'zh_full', 0.90),
        ('油田工作', 'zh_short', 0.85),
        ('Workplace', 'en_short', 0.90),
        ('WorkplaceType', 'en', 0.95),
    ],
    'idu_status': [
        ('吸毒史', 'zh_clinical', 1.0),
        ('idu_status', 'en', 1.0),
        ('注射吸毒', 'zh_full', 0.90),
        ('IDU', 'en_short', 0.90),
        ('idu', 'en_short', 0.90),
        ('IduStatus', 'en', 0.95),
    ],
    'months_since_migration': [
        ('迁入月份', 'zh_clinical', 1.0),
        ('months_since_migration', 'en', 1.0),
        ('迁入月数', 'zh_full', 0.90),
        ('migration', 'en_short', 0.85),
        ('Migration', 'en_short', 0.85),
        ('MonthsSinceMigration', 'en', 0.95),
    ],
}


# ==============================================================================
# 三、名称标准化
# ==============================================================================

def normalize_name(name):
    """标准化名称：去除空格、下划线、连字符、括号，转小写

    参数：
        name (str): 原始名称

    返回：
        str: 标准化后的名称
    """
    if not isinstance(name, str):
        return ''
    return re.sub(r'[\s_\-（）()]+', '', name).lower()


# ==============================================================================
# 四、Levenshtein 编辑距离
# ==============================================================================

def _levenshtein_distance(s1, s2):
    """计算编辑距离（Levenshtein distance）。

    优先使用 python-Levenshtein C 扩展，不可用时回退到纯 Python 实现。

    参数：
        s1 (str): 字符串1
        s2 (str): 字符串2

    返回：
        int: 编辑距离
    """
    if _LEVENSHTEIN_AVAILABLE:
        return _lev_distance(s1, s2)

    # 纯 Python 回退：Wagner-Fischer 算法（仅使用两行内存）
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # 插入 / 删除 / 替换
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# ==============================================================================
# 五、关键词包含匹配
# ==============================================================================

# 关键词优先级规则（用于模糊匹配的第二层）
_KEYWORD_PRIORITY = [
    (['确诊', 'diagnosed', 'diagnosis', 'tb_outcome', 'label', 'target'], 'is_confirmed', 3),
    (['年龄', 'age', '周岁'], 'age', 2),
    (['痰涂片', 'sputum', 'smear', 'PTB'], 'sputum_smear', 2),
    (['咳嗽', 'cough'], 'cough_freq', 2),
    (['症状', 'symptom'], 'symptoms', 2),
    (['卡介苗', 'bcg', '疫苗', 'vaccine'], 'bcg_vaccine', 2),
    (['结核史', 'tb_history', '既往结核'], 'has_tb', 2),
    (['慢性病', 'past_illness', '合并症', '基础疾病'], 'past_illness', 2),
    (['高危', 'high_risk', '高风险'], 'is_high_risk', 2),
    (['通风', 'ventilation', 'vent'], 'ventilation', 2),
    (['场景', 'setting', '场所'], 'exposure_setting', 2),
    (['距离', 'distance'], 'contact_distance', 2),
    (['空洞', 'cavity'], 'has_cavity', 2),
    (['活动性', 'active_tb', 'active'], 'active_tb', 2),
    (['治疗', 'treatment', 'treated'], 'treatment', 2),
    (['时长', 'duration', '时间'], 'single_duration', 1),
    (['频次', 'freq', '频率'], 'freq_density', 1),
    (['周期', 'time_span', '跨度', 'span'], 'time_span', 1),
    (['暴露', 'exposure', '累积'], 'cumulative_exposure', 1),
    (['编号', 'record_id', '序号', 'patient_id'], 'record_id', 1),
    (['民族', 'ethnicity', 'race'], 'ethnicity', 1),
    (['职业', 'occupation', 'job'], 'occupation', 1),
    (['性别', 'gender', 'sex'], 'gender', 1),
    (['区县', 'district', '区域', '地区', 'county'], 'district', 1),
    (['海拔', 'altitude'], 'origin_altitude', 1),
    (['吸毒', 'idu'], 'idu_status', 1),
    (['迁入', 'migration'], 'months_since_migration', 1),
]


def _keyword_match(col_name):
    """关键词包含匹配（双向包含）。

    参数：
        col_name (str): 原始列名

    返回：
        (standard_name, confidence) 或 (None, 0.0)
    """
    if not col_name:
        return None, 0.0

    col_lower = col_name.lower().strip()
    best_match = None
    best_priority = 0

    for keywords, field_name, priority in _KEYWORD_PRIORITY:
        for kw in keywords:
            if kw in col_lower or col_lower in kw:
                if priority > best_priority:
                    best_match = field_name
                    best_priority = priority
                break

    if best_match:
        confidence = 0.85 + best_priority * 0.05  # 优先级越高置信度越高
        return best_match, min(confidence, 0.95)

    return None, 0.0


# ==============================================================================
# 六、三阶模糊匹配主函数
# ==============================================================================

def fuzzy_match_column(col_name, threshold=2):
    """三阶匹配：精确匹配 → 关键词包含 → Levenshtein 模糊匹配

    参数：
        col_name (str): 原始列名
        threshold (int): Levenshtein 编辑距离阈值，默认 2（会根据候选词长度动态调整）

    返回：
        (standard_name, confidence, source) 或 (None, 0.0, None)
        - standard_name: 标准字段名
        - confidence: 匹配置信度 (0.0-1.0)
        - source: 匹配来源类型 ('exact' / 'keyword' / 'fuzzy_levenshtein' / 'fuzzy_difflib')
    """
    if not col_name or not isinstance(col_name, str):
        return None, 0.0, None

    col_normalized = normalize_name(col_name)

    # ---- 第一层：精确匹配 ----
    for standard_name, synonyms in FIELD_SYNONYM_LIBRARY.items():
        # 检查标准字段名
        if col_normalized == normalize_name(standard_name):
            return standard_name, 1.0, 'exact'
        # 检查所有别名
        for alias, src_type, confidence in synonyms:
            if col_normalized == normalize_name(alias):
                return standard_name, confidence, 'exact'

    # ---- 第二层：关键词包含匹配 ----
    kw_match, kw_confidence = _keyword_match(col_name)
    if kw_match:
        return kw_match, kw_confidence, 'keyword'

    # ---- 第三层：模糊匹配（动态阈值） ----
    # 3a. Levenshtein 编辑距离
    best_lev_match = None
    best_lev_distance = float('inf')
    best_lev_confidence = 0.0

    all_candidates = []
    for standard_name, synonyms in FIELD_SYNONYM_LIBRARY.items():
        all_candidates.append((standard_name, standard_name))
        for alias, src_type, _ in synonyms:
            all_candidates.append((standard_name, alias))

    for standard_name, candidate in all_candidates:
        norm_candidate = normalize_name(candidate)
        # 动态阈值：根据候选词长度调整
        effective_threshold = _compute_dynamic_threshold(col_normalized, norm_candidate, threshold)
        dist = _levenshtein_distance(col_normalized, norm_candidate)
        if dist <= effective_threshold and dist < best_lev_distance:
            best_lev_distance = dist
            best_lev_match = standard_name
            best_lev_confidence = 1.0 - dist / (effective_threshold + 1)

    if best_lev_match:
        # 编辑距离置信度范围 0.7-0.85
        best_lev_confidence = max(0.7, min(0.85, best_lev_confidence))
        return best_lev_match, best_lev_confidence, 'fuzzy_levenshtein'

    # 3b. difflib 补充匹配
    all_candidates_strs = []
    candidate_map = {}
    for standard_name, synonyms in FIELD_SYNONYM_LIBRARY.items():
        norm_std = normalize_name(standard_name)
        all_candidates_strs.append(norm_std)
        candidate_map[norm_std] = standard_name
        for alias, src_type, _ in synonyms:
            norm_alias = normalize_name(alias)
            all_candidates_strs.append(norm_alias)
            candidate_map[norm_alias] = standard_name

    matches = difflib.get_close_matches(col_normalized, all_candidates_strs, n=1, cutoff=0.8)
    if matches:
        matched = matches[0]
        standard_name = candidate_map.get(matched)
        if standard_name:
            # difflib 相似度作为置信度
            ratio = difflib.SequenceMatcher(None, col_normalized, matched).ratio()
            confidence = max(0.75, min(0.85, ratio))
            return standard_name, confidence, 'fuzzy_difflib'

    return None, 0.0, None


def _compute_dynamic_threshold(s1, s2, base_threshold=2):
    """根据候选词长度动态调整编辑距离阈值。

    规则：
    - 长度 ≤ 3 → 阈值 1（短词对编辑距离敏感）
    - 长度 ≤ 5 → 阈值 2
    - 长度 ≥ 8 → 阈值 3（长词可容忍更多编辑）
    - 其他 → 使用 base_threshold

    参数：
        s1 (str): 标准化后的输入字符串
        s2 (str): 标准化后的候选字符串
        base_threshold (int): 基础阈值

    返回：
        int: 调整后的阈值
    """
    min_len = min(len(s1), len(s2))
    if min_len <= 3:
        return 1
    if min_len <= 5:
        return 2
    if min_len >= 8:
        return 3
    return base_threshold


# ==============================================================================
# 七、用户反馈学习
# ==============================================================================

def register_synonym_feedback(original_name, standard_field):
    """用户确认匹配后，将原始列名写入同义词库并持久化到磁盘。

    参数：
        original_name (str): 用户输入的原始列名
        standard_field (str): 用户选择的标准字段名

    返回：
        bool: 是否成功写入
    """
    if standard_field not in FIELD_SYNONYM_LIBRARY:
        LOGGER.warning("标准字段 '%s' 不在同义词库中，无法注册反馈", standard_field)
        return False

    if not original_name or not isinstance(original_name, str):
        return False

    original_name = original_name.strip()

    # 检查是否已存在（避免重复）
    existing_aliases = [alias for alias, _, _ in FIELD_SYNONYM_LIBRARY[standard_field]]
    if normalize_name(original_name) in [normalize_name(a) for a in existing_aliases]:
        return False

    FIELD_SYNONYM_LIBRARY[standard_field].append(
        (original_name, 'user_feedback', 0.95)
    )
    LOGGER.info("用户反馈已注册: '%s' → '%s'", original_name, standard_field)

    # 持久化到磁盘
    _save_feedback_to_disk()
    return True


def _load_feedback_from_disk():
    """从磁盘加载用户反馈到同义词库。"""
    if not os.path.exists(_FEEDBACK_FILE):
        return

    try:
        with open(_FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        LOGGER.debug("加载用户反馈失败: %s", e)
        return

    loaded_count = 0
    for standard_field, entries in feedback_data.items():
        if standard_field not in FIELD_SYNONYM_LIBRARY:
            continue
        for entry in entries:
            alias = entry[0] if isinstance(entry, list) else entry.get('alias', '')
            if not alias:
                continue
            # 避免重复
            existing = [normalize_name(a) for a, _, _ in FIELD_SYNONYM_LIBRARY[standard_field]]
            if normalize_name(alias) in existing:
                continue
            FIELD_SYNONYM_LIBRARY[standard_field].append(
                (alias, 'user_feedback', 0.95)
            )
            loaded_count += 1

    if loaded_count > 0:
        LOGGER.info("从磁盘加载了 %d 条用户反馈同义词", loaded_count)


def _save_feedback_to_disk():
    """将用户反馈保存到磁盘。"""
    feedback_data = {}
    for field, synonyms in FIELD_SYNONYM_LIBRARY.items():
        user_entries = [
            [alias, src, conf]
            for alias, src, conf in synonyms
            if src == 'user_feedback'
        ]
        if user_entries:
            feedback_data[field] = user_entries

    if not feedback_data:
        return

    try:
        with open(_FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        LOGGER.debug("用户反馈已保存到 %s", _FEEDBACK_FILE)
    except IOError as e:
        LOGGER.warning("保存用户反馈失败: %s", e)


# 模块加载时自动加载持久化的用户反馈
_load_feedback_from_disk()


# ==============================================================================
# 八、向后兼容：委托到旧版 _FIELD_SYNONYMS 格式
# ==============================================================================

def get_field_synonyms():
    """获取字段同义词字典（只读副本，兼容旧版 io_utils._FIELD_SYNONYMS 格式）。

    返回：
        dict: {标准字段名: [别名列表]}
    """
    result = {}
    for field, synonyms in FIELD_SYNONYM_LIBRARY.items():
        result[field] = [alias for alias, _, _ in synonyms]
    return result


def get_field_synonyms_with_confidence():
    """获取字段同义词字典（含置信度信息）。

    返回：
        dict: {标准字段名: [(别名, 来源类型, 置信度), ...]}
    """
    import copy
    return copy.deepcopy(FIELD_SYNONYM_LIBRARY)


def generate_legacy_synonyms():
    """从 FIELD_SYNONYM_LIBRARY 生成兼容旧版 io_utils._FIELD_SYNONYMS 的格式。

    返回：
        dict: {标准字段名: [别名列表]}（与旧版 _FIELD_SYNONYMS 格式一致）
    """
    result = {}
    for field, synonyms in FIELD_SYNONYM_LIBRARY.items():
        result[field] = list(set(
            alias for alias, _, _ in synonyms
        ))
    return result


def dynamic_column_match(col_name):
    """向后兼容 io_utils.dynamic_column_match 的包装函数。

    使用新的三阶模糊匹配，但返回格式与旧版一致（仅返回标准字段名或 None）。

    参数：
        col_name (str): 原始列名

    返回：
        str|None: 匹配到的标准字段名，或 None
    """
    result = fuzzy_match_column(col_name)
    if result[0]:
        return result[0]
    return None