#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""患者评分模块（纯业务逻辑层）

将患者总体风险评分与个体风险等级计算从 RiskAssessmentService 中拆分出来，
形成独立的协作类。零依赖 tkinter 或任何 GUI 库。

职责：
- 患者信息/接触者预处理
- 患者总体风险评分（加权 + 症状/年龄/延迟加分 + Sigmoid 转换）
- 各维度风险等级与综合建议（FCI/SNC/FLP/HRSP/FTD）
- 患者类型风险乘数
- FCI / SNC 综合得分（家庭/社会接触强度）
"""

import copy
import logging
from typing import Any, Dict, List, Optional

from ..scoring.engine import ScoringEngine
from ..constants import (
    BASE_WEIGHTS,
    PATIENT_TYPE_ADJUSTMENTS,
    AGE_PROGRESSION,
    SPUTUM_SMEAR_POSITIVE_INCREASE, HAS_CAVITY_INCREASE, UNTREATED_INCREASE,
    SYMPTOMS_SEVERE_BONUS, COUGH_FREQ_HIGH_BONUS, COUGH_FREQ_MEDIUM_BONUS,
    COUGH_FREQ_LOW_BONUS, AGE_YOUNG_BONUS, AGE_ELDERLY_BONUS,
    DELAY_DAYS_SEVERE_BONUS, DELAY_DAYS_MEDIUM_BONUS, DELAY_DAYS_MILD_BONUS,
    INFECTION_PROB_HIGH_RISK, INFECTION_PROB_MEDIUM_RISK,
    RISK_THRESHOLDS,
    DEFAULT_DISTANCE_FACTOR, DEFAULT_VENTILATION_FACTOR, DEFAULT_SETTING_FACTOR,
    PATIENT_RISK_MULTIPLIER_CAP,
    SMEAR_CAVITY_ADDITIVE,
    LIVING_CONDITION_INTERCEPT, LIVING_CONDITION_SLOPE, LIVING_CONDITION_MIN_FACTOR,
    FCI_MAX_SCORE, FCI_DEFAULT_NO_FAMILY,
    HIGH_RISK_CONTACT_FACTOR,
    SNC_CONTACT_COUNT_DIVISOR, SNC_AVG_SCORE_WEIGHT, SNC_DIVISOR, SNC_MAX_SCORE,
    TOTAL_SCORE_CAP, FTD_NORMALIZATION_DAYS,
)
from .probability import (
    sigmoid_total_score, safe_float, safe_int,
    get_exposure_risk_score,
)
from .data_conversion import (
    convert_chinese_to_value,
    standardize_contact_fields,
    apply_scenario_defaults,
)
from .lab_evidence import (
    compute_infectivity_bonus,
    derive_lab_evidence_grade,
    is_mdr_record,
)

LOGGER = logging.getLogger("tb_risk.core")

# 模块级常量别名（与 AGE_PROGRESSION 同一真值源）
AGE_PROGRESSION_FACTORS = AGE_PROGRESSION


class PatientScorer:
    """患者风险评分器

    封装患者总体风险评分与各维度风险等级计算逻辑。
    """

    # 时间转换常数
    DAYS_PER_MONTH_AVG = 30.44
    DEFAULT_AGE = 30

    def __init__(self, engine=None):
        """初始化患者评分器

        参数：
            engine: ScoringEngine | None — 评分引擎实例（仅用于类型常量读取，可为 None）
        """
        self._engine = engine

    # ==================== 预处理 ====================

    def preprocess_patient(self, patient_info: Optional[Dict]) -> Dict:
        """预处理患者信息（深拷贝并标准化关键字段）"""
        if not patient_info:
            return {}
        info = copy.deepcopy(patient_info)
        if 'basic_info' in info:
            basic = info['basic_info']
            if 'treatment_duration' in basic:
                basic['treatment_duration'] = safe_float(
                    basic['treatment_duration'], 0.0)
        return info

    def preprocess_contacts(
        self, contacts: List[Dict], contact_type: str
    ) -> List[Dict]:
        """预处理接触者列表（标准化字段、应用场景默认、补全累积暴露）"""
        processed = []
        for contact in contacts:
            c = copy.deepcopy(contact)
            c = standardize_contact_fields(c, contact_type)
            c = apply_scenario_defaults(c, contact_type)
            if 'cumulative_exposure' not in c or c['cumulative_exposure'] is None:
                from .probability import calculate_cumulative_exposure
                c['cumulative_exposure'] = calculate_cumulative_exposure(
                    c.get('single_duration'),
                    c.get('freq_density'),
                    c.get('time_span'),
                    c.get('workplace_type', '非油田')
                )
            processed.append(c)
        return processed

    def get_treatment_days(self, patient_info: Dict) -> float:
        """从患者信息中提取治疗天数"""
        if patient_info and 'basic_info' in patient_info:
            treatment_months = safe_float(
                patient_info['basic_info'].get('treatment_duration', 0), 0.0)
            return treatment_months * self.DAYS_PER_MONTH_AVG
        return 0.0

    # ==================== 患者评分 ====================

    def compute_patient_score(self, patient_info: Dict) -> Dict[str, Any]:
        """计算患者总体风险评分

        返回：
            dict: {
                'total_score': float,
                'base_infection_probability': float,
                'weights': dict,
                'patient_risk_multiplier': float,
            }
        """
        if not patient_info:
            return {'total_score': 0.0, 'base_infection_probability': 0.0,
                    'weights': dict(BASE_WEIGHTS), 'patient_risk_multiplier': 1.0,
                    'infectivity_bonus': 0.0, 'lab_evidence_grade': 0,
                    'is_mdr': False}

        basic_info = patient_info.get('basic_info', {})

        # 提取患者评分因子
        fci_score = safe_float(patient_info.get('FCI', 0.0), 0.0)
        snc_score = safe_float(patient_info.get('SNC', 0.0), 0.0)
        flp_percentage = safe_float(basic_info.get('flp_percentage', 0.0), 0.0)
        hrsp_percentage = safe_float(basic_info.get('hrsp_percentage', 0.0), 0.0)
        ftd_days = safe_float(basic_info.get('delay_days', 0), 0.0)

        # 提取基本信息
        age = safe_float(basic_info.get('age', 30), 30.0)
        sputum_smear = safe_float(basic_info.get('sputum_smear', 1), 1.0)
        has_cavity = safe_float(basic_info.get('has_cavity', 1), 1.0)
        active_tb = safe_float(basic_info.get('active_tb', 1), 1.0)
        treatment = safe_float(basic_info.get('treatment', 2), 2.0)
        cough_freq = safe_float(basic_info.get('cough_freq', 0), 0.0)
        symptoms = safe_float(basic_info.get('symptoms', 2), 2.0)
        delay_days = safe_float(basic_info.get('delay_days', 0), 0.0)

        # 标准化编码
        sputum_smear_norm = 1 if sputum_smear == 2 else 0
        has_cavity_norm = 1 if has_cavity == 2 else 0
        untreated = (active_tb == 1 and treatment == 2)

        # 初始化权重（从 BASE_WEIGHTS 单一真值源）
        weights = dict(BASE_WEIGHTS)
        total_weight = sum(weights.values())
        if total_weight > 0:
            for param in weights:
                weights[param] = weights[param] / total_weight

        # 患者类型风险乘数
        patient_risk_multiplier = self.calculate_patient_risk_multiplier(
            sputum_smear_norm, has_cavity_norm, untreated)

        # 计算加权基础评分
        flp_normalized = max(0.0, min(flp_percentage / 100.0, 1.0))
        hrsp_normalized = max(0.0, min(hrsp_percentage / 100.0, 1.0))
        ftd_normalized = max(0.0, min(ftd_days / FTD_NORMALIZATION_DAYS, 1.0))

        _DIM_MAX = FCI_MAX_SCORE  # 单维度评分上限（0-10分制）
        base_score = (
            min(fci_score, _DIM_MAX) * weights['FCI'] +
            min(snc_score, _DIM_MAX) * weights['SNC'] +
            flp_normalized * _DIM_MAX * weights['FLP'] +
            hrsp_normalized * _DIM_MAX * weights['HRSP'] +
            ftd_normalized * _DIM_MAX * weights['FTD']
        )

        total_score = min(base_score * patient_risk_multiplier, TOTAL_SCORE_CAP)

        # 症状/咳嗽/年龄/延迟加分
        if symptoms >= 3:
            total_score = min(total_score + SYMPTOMS_SEVERE_BONUS, _DIM_MAX)
        if cough_freq >= 10:
            total_score = min(total_score + COUGH_FREQ_HIGH_BONUS, _DIM_MAX)
        elif cough_freq >= 5:
            total_score = min(total_score + COUGH_FREQ_MEDIUM_BONUS, _DIM_MAX)
        elif cough_freq >= 1:
            total_score = min(total_score + COUGH_FREQ_LOW_BONUS, _DIM_MAX)

        if 15 <= age <= 35:
            total_score = min(total_score + AGE_YOUNG_BONUS, _DIM_MAX)
        elif age > 65:
            total_score = min(total_score + AGE_ELDERLY_BONUS, _DIM_MAX)

        if active_tb == 1:
            if delay_days >= 30:
                total_score = min(total_score + DELAY_DAYS_SEVERE_BONUS, _DIM_MAX)
            elif delay_days >= 15:
                total_score = min(total_score + DELAY_DAYS_MEDIUM_BONUS, _DIM_MAX)
            elif delay_days >= 1:
                total_score = min(total_score + DELAY_DAYS_MILD_BONUS, _DIM_MAX)

        # Sigmoid 转换
        base_infection_probability = sigmoid_total_score(total_score)

        # 患者特征调整（实验室证据驱动）
        # 传染性加分 = 20 × bacillary_load(Ct)（平滑替换原二值涂阳 +20 逻辑）：
        #   - Ct 值分档连续映射（Ct<22→1.0 / 22-28→0.6 / >28→0.3）
        #   - Ct/涂片缺失时用影像学范围因子替代（cavity/tree_in_bud→0.8 ...）
        #   - 无实验室证据时回退旧逻辑（涂阳才加分）
        infectivity_bonus = compute_infectivity_bonus(basic_info)
        if infectivity_bonus > 0:
            base_infection_probability = min(
                base_infection_probability + infectivity_bonus,
                ScoringEngine.SIGMOID_CAP)
        if has_cavity_norm:
            base_infection_probability = min(
                base_infection_probability + HAS_CAVITY_INCREASE,
                ScoringEngine.SIGMOID_CAP)
        if untreated:
            base_infection_probability = min(
                base_infection_probability + UNTREATED_INCREASE,
                ScoringEngine.SIGMOID_CAP)

        # 自动推导的实验室证据等级（0-3）与 MDR 标记（供 GUI/导出标注）
        lab_evidence_grade = derive_lab_evidence_grade(basic_info)

        return {
            'total_score': total_score,
            'base_infection_probability': base_infection_probability,
            'weights': weights,
            'patient_risk_multiplier': patient_risk_multiplier,
            'infectivity_bonus': infectivity_bonus,
            'lab_evidence_grade': lab_evidence_grade,
            'is_mdr': is_mdr_record(basic_info),
        }

    def compute_individual_risks(
        self, patient_info: Dict, fci_score: float, snc_score: float,
        flp_percentage: float, hrsp_percentage: float, ftd_days: float,
        base_infection_probability: float,
    ) -> Dict[str, Any]:
        """计算各维度风险等级和综合建议

        返回：
            dict: {
                'individual_risks': {FCI/SNC/FLP/HRSP/FTD: {risk, suggestion}},
                'overall_risk': str,
                'overall_suggestion': str,
            }
        """
        risk_thresholds = RISK_THRESHOLDS

        # 综合风险等级
        if base_infection_probability >= INFECTION_PROB_HIGH_RISK:
            overall_risk = '高风险'
            overall_suggestion = '患者的家庭社会网络中存在大量潜在患者，需立即进行全面筛查和干预'
        elif base_infection_probability >= INFECTION_PROB_MEDIUM_RISK:
            overall_risk = '中风险'
            overall_suggestion = '患者的家庭社会网络中存在一定数量的潜在患者，建议尽快进行重点筛查'
        else:
            overall_risk = '低风险'
            overall_suggestion = '患者的家庭社会网络中潜在患者数量较少，可按常规流程进行监测'

        # FCI
        if fci_score >= risk_thresholds['FCI_high']:
            fci_risk, fci_suggestion = '高风险', '家庭内密切接触者感染风险高，建议全部进行结核菌素试验（PPD）或γ-干扰素释放试验（IGRA）筛查'
        elif fci_score >= risk_thresholds['FCI_medium']:
            fci_risk, fci_suggestion = '中风险', '家庭内接触者存在一定感染风险，建议对儿童、老人及免疫抑制者优先筛查'
        else:
            fci_risk, fci_suggestion = '低风险', '家庭内接触者感染风险较低，可按常规随访观察'

        # SNC
        if snc_score >= risk_thresholds['SNC_high']:
            snc_risk, snc_suggestion = '高风险', '患者社会接触广泛，传播风险高，建议对经常接触的人员进行筛查'
        elif snc_score >= risk_thresholds['SNC_medium']:
            snc_risk, snc_suggestion = '中风险', '患者有一定社会接触面，建议对密切接触者进行筛查'
        else:
            snc_risk, snc_suggestion = '低风险', '患者社会接触相对有限，传播风险较低，可重点关注密切接触者'

        # FLP
        if flp_percentage >= risk_thresholds['FLP_high']:
            flp_risk, flp_suggestion = '高风险', '家庭内潜伏感染率高，建议对所有家庭成员进行潜伏感染检测'
        elif flp_percentage >= risk_thresholds['FLP_medium']:
            flp_risk, flp_suggestion = '中风险', '家庭内潜伏感染率中等，建议对儿童和免疫抑制者进行检测'
        else:
            flp_risk, flp_suggestion = '低风险', '家庭内潜伏感染率较低，可选择性进行检测'

        # HRSP
        if hrsp_percentage >= risk_thresholds['HRSP_high']:
            hrsp_risk, hrsp_suggestion = '高风险', '社会接触中高风险人群比例高，建议对这些高风险者进行优先筛查'
        elif hrsp_percentage >= risk_thresholds['HRSP_medium']:
            hrsp_risk, hrsp_suggestion = '中风险', '社会接触中有一定比例的高风险人群，建议对其进行筛查'
        else:
            hrsp_risk, hrsp_suggestion = '低风险', '社会接触中高风险人群比例低，可按一般流程处理'

        # FTD
        if ftd_days >= risk_thresholds['FTD_high']:
            ftd_risk, ftd_suggestion = '高风险', '从症状出现到隔离的时间较长，家庭内传播风险较高，需加强筛查'
        elif ftd_days >= risk_thresholds['FTD_medium']:
            ftd_risk, ftd_suggestion = '中风险', '从症状出现到隔离的时间适中，存在一定的家庭内传播风险'
        else:
            ftd_risk, ftd_suggestion = '低风险', '从症状出现到隔离的时间短，家庭内传播风险相对较低，但仍需注意'

        return {
            'individual_risks': {
                'FCI': {'risk': fci_risk, 'suggestion': fci_suggestion},
                'SNC': {'risk': snc_risk, 'suggestion': snc_suggestion},
                'FLP': {'risk': flp_risk, 'suggestion': flp_suggestion},
                'HRSP': {'risk': hrsp_risk, 'suggestion': hrsp_suggestion},
                'FTD': {'risk': ftd_risk, 'suggestion': ftd_suggestion},
            },
            'overall_risk': overall_risk,
            'overall_suggestion': overall_suggestion,
        }

    def calculate_patient_risk_multiplier(self, sputum_smear, has_cavity, untreated):
        """计算患者类型的整体风险乘数（上限PATIENT_RISK_MULTIPLIER_CAP）"""
        base_risk = 1.0
        if sputum_smear == 1:
            base_risk += PATIENT_TYPE_ADJUSTMENTS['sputum_smear_positive']
        if has_cavity == 1:
            base_risk += PATIENT_TYPE_ADJUSTMENTS['has_cavity']
        if untreated:
            base_risk += PATIENT_TYPE_ADJUSTMENTS['untreated']
        if sputum_smear == 1 and has_cavity == 1:
            base_risk += SMEAR_CAVITY_ADDITIVE
        return min(base_risk, PATIENT_RISK_MULTIPLIER_CAP)

    # ==================== FCI / SNC 综合得分 ====================

    def calculate_fci_score(self, family_members, family_living_conditions):
        """计算家庭内密切接触强度（FCI）综合得分（0-10分）"""
        if not family_members:
            return FCI_DEFAULT_NO_FAMILY

        total_contact_score = 0.0
        for member in family_members:
            exposure_score = get_exposure_risk_score(
                member.get('cumulative_exposure', 0))
            contact_distance = convert_chinese_to_value(
                member.get('contact_distance', 'close'),
                dict(ScoringEngine.DISTANCE_FACTORS)
            )
            if contact_distance not in ScoringEngine.DISTANCE_FACTORS:
                contact_distance = 'close'
            distance_factor = ScoringEngine.DISTANCE_FACTORS.get(
                contact_distance, DEFAULT_DISTANCE_FACTOR)

            ventilation = safe_int(member.get('ventilation', 3), 3)
            ventilation_factor = ScoringEngine.VENTILATION_FACTORS.get(
                ventilation, DEFAULT_VENTILATION_FACTOR)

            member_contact_score = exposure_score * distance_factor * ventilation_factor
            total_contact_score += member_contact_score

        # 居住条件因子：线性模型 max(INTERCEPT + SLOPE * rating, MIN_FACTOR)
        # rating: 1(极差)~5(极好)
        living_condition_factor = (LIVING_CONDITION_INTERCEPT
                                   + LIVING_CONDITION_SLOPE * family_living_conditions)
        if family_living_conditions <= 2:
            living_condition_factor = max(living_condition_factor, LIVING_CONDITION_MIN_FACTOR)

        avg_member_score = (
            total_contact_score / len(family_members) if family_members else 0)
        fci_score = avg_member_score * living_condition_factor * FCI_MAX_SCORE
        return min(fci_score, FCI_MAX_SCORE)

    def calculate_snc_score(self, social_contacts):
        """计算社会接触网络中心性（SNC）综合得分（0-10分）"""
        if not social_contacts:
            return 0.0

        total_network_score = 0.0
        for contact in social_contacts:
            exposure_score = get_exposure_risk_score(
                contact.get('cumulative_exposure', 0))

            contact_distance = convert_chinese_to_value(
                contact.get('contact_distance', 'medium'),
                dict(ScoringEngine.DISTANCE_FACTORS)
            )
            if contact_distance not in ScoringEngine.DISTANCE_FACTORS:
                contact_distance = 'medium'
            distance_factor = ScoringEngine.DISTANCE_FACTORS.get(
                contact_distance, DEFAULT_DISTANCE_FACTOR)

            ventilation = safe_int(contact.get('ventilation', 3), 3)
            ventilation_factor = ScoringEngine.VENTILATION_FACTORS.get(
                ventilation, DEFAULT_VENTILATION_FACTOR)

            exposure_setting = convert_chinese_to_value(
                contact.get('exposure_setting', 'general'),
                dict(ScoringEngine.SETTING_FACTORS_BASE)
            )
            if exposure_setting not in ScoringEngine.SETTING_FACTORS_BASE:
                exposure_setting = 'general'
            setting_factor = ScoringEngine.SETTING_FACTORS_BASE.get(
                exposure_setting, DEFAULT_SETTING_FACTOR)

            high_risk_factor = HIGH_RISK_CONTACT_FACTOR if contact.get('is_high_risk', 0) == 1 else 1.0

            contact_network_score = (
                exposure_score * distance_factor
                * ventilation_factor * setting_factor
                * high_risk_factor
            )
            total_network_score += contact_network_score

        contact_count_score = min(len(social_contacts) / SNC_CONTACT_COUNT_DIVISOR, SNC_MAX_SCORE)
        avg_contact_score = (
            total_network_score / len(social_contacts) if social_contacts else 0)
        snc_score = (avg_contact_score * SNC_AVG_SCORE_WEIGHT + contact_count_score) / SNC_DIVISOR
        return min(snc_score, SNC_MAX_SCORE)
