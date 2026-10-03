#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接触者风险计算模块（纯业务逻辑层）

将接触者风险计算逻辑从 RiskAssessmentService 中拆分出来，形成独立的协作类。
零依赖 tkinter 或任何 GUI 库。

职责：
- 生成潜在患者列表（家庭/社会）
- 单个接触者的风险评分与概率计算（规则 + SEIR 融合 + 本土化增强）
- BCG 保护效力计算
- 克拉玛依本土化增强（感染/发病概率调整、建议文本增强）
- 环境配置构建（气候参数）
"""

import logging
from typing import Any, Dict, List, Optional

from ..scoring.engine import ScoringEngine
from ..utils import _is_yes
from ..schemas import ContactRiskResult  # 问题六-步1：出口 schema 约束
from ..constants import (
    AGE_PROGRESSION,
    BCG_PROTECTION_PARAMS,
    SYMPTOMS_PRESENT_BONUS,
    DEFAULT_DISTANCE_FACTOR, DEFAULT_VENTILATION_FACTOR, DEFAULT_SETTING_FACTOR,
    LATENT_BASELINE,  # 问题八-3：单一真值源
)
from .probability import (
    sigmoid_base_infection, safe_int,
    get_exposure_risk_score,
)
from .data_conversion import (
    convert_chinese_to_value,
    get_priority_and_recommendation,
)
from .seir_integration import fuse_probabilities
from .lab_evidence import (
    igra_progression_factor,
    derive_lab_evidence_grade,
    lab_evidence_grade_label,
    low_evidence_warning,
    is_mdr_record,
    # 双输出进展建模（P_infected × P_progress）
    latent_infection_probability,
    short_term_progression_probability,
)

LOGGER = logging.getLogger("tb_risk.core")

# 模块级常量别名（与 AGE_PROGRESSION 同一真值源）
AGE_PROGRESSION_FACTORS = AGE_PROGRESSION


class ContactRiskCalculator:
    """接触者风险计算器

    封装单个接触者风险评分与潜在患者列表生成逻辑。
    通过依赖注入接收 SEIRIntegration 协作器和本土化适配器。
    """

    # 时间/距离常数
    DEFAULT_CONTACT_DISTANCE_FAMILY = 'close'
    DEFAULT_CONTACT_DISTANCE_SOCIAL = 'medium'

    # 潜伏→发病基线（问题八-3：单一真值源，引用 constants.LATENT_BASELINE）
    LATENT_TO_ACTIVE_BASELINE = LATENT_BASELINE

    def __init__(self, localizer=None, contact_labels=None,
                 seir_integration=None, progress_callback=None):
        """初始化接触者风险计算器

        参数：
            localizer: KaramayLocalizer | None — 本土化适配器
            contact_labels: dict | None — 接触者标签 {_id: 0/1}
            seir_integration: SEIRIntegration | None — SEIR 协作器
            progress_callback: callable | None — 进度回调 (progress, message)
        """
        self._localizer = localizer
        self._use_localization = localizer is not None
        self._contact_labels = contact_labels or {}
        self._seir = seir_integration
        self._progress_callback = progress_callback

    def set_progress_callback(self, callback) -> None:
        """设置进度回调"""
        self._progress_callback = callback

    def set_contact_labels(self, contact_labels) -> None:
        """更新接触者标签（支持延迟初始化）"""
        self._contact_labels = contact_labels or {}

    def _report_progress(self, progress: int, message: str = "") -> None:
        """报告进度"""
        if self._progress_callback:
            try:
                self._progress_callback(progress, message)
            except Exception as e:
                LOGGER.debug("接触风险计算进度回调失败: %s", e)

    # ==================== 潜在患者生成 ====================

    def generate_potential_patients(
        self,
        patient_info: Dict,
        family_members: List[Dict],
        social_contacts: List[Dict],
        patient_treatment_days: float,
        individual_risks_result: Optional[Dict] = None,
    ) -> Dict[str, List[Dict]]:
        """生成潜在患者列表（家庭/社会）"""
        potential_patients = {'family': [], 'social': []}
        total_contacts = len(family_members) + len(social_contacts)
        processed = 0

        # 源病例 MDR 状态（genexpert_rif=resistant）：
        # MDR 传染窗口更长、治疗周期更长，按 MDR 拉长治疗传染性衰减曲线
        patient_basic = (patient_info or {}).get('basic_info', {})
        is_mdr = is_mdr_record(patient_basic)

        for fam_idx, member in enumerate(family_members):
            result = self.compute_single_contact_risk(
                member, patient_treatment_days,
                default_contact_distance=self.DEFAULT_CONTACT_DISTANCE_FAMILY,
                is_social_contact=False,
                is_mdr=is_mdr,
            )

            priority, recommendation = get_priority_and_recommendation(
                result['disease_probability'])

            if self._use_localization:
                recommendation = self.enhance_recommendation_karamay(
                    member, priority, recommendation)

            # 实验室证据等级与低证据告警（GUI/导出标注）
            lab_grade = result.get('lab_evidence_grade', 0)
            evidence_note = low_evidence_warning(member) or ''
            if evidence_note:
                recommendation = f"{recommendation}；{evidence_note}"

            is_diagnosed = member.get('diagnosed', False)
            contact_id = member.get('_id', None)
            if contact_id and not is_diagnosed:
                is_diagnosed = (self._contact_labels.get(contact_id, 0) == 1)

            name = member.get('name', '未知')
            rel = member.get('relationship', '家庭成员')
            age = member.get('age', 30)
            # 问题六：record_id 作为层间契约的稳定唯一标识。
            # 优先使用数据库 _id；缺失时回退到稳定的索引合成 id（同一评估内不变）。
            member_record_id = member.get('record_id') or contact_id or f"family_{fam_idx}"
            potential_patients['family'].append({
                '_id': contact_id,
                'record_id': member_record_id,
                'name': name,
                'relationship': rel,
                'age': age,
                'freq_density': member.get('freq_density', 0),
                'time_span': member.get('time_span', 0),
                'is_diagnosed': is_diagnosed,
                'priority': priority,
                'risk_score': result['risk_score'],
                'infection_probability': result['infection_probability'],
                'rule_infection_probability': result['rule_infection_probability'],
                'seir_infection_probability': result['seir_infection_probability'],
                'disease_probability': result['disease_probability'],
                'latent_infection_prob': result['latent_infection_prob'],
                'short_term_active_risk': result['short_term_active_risk'],
                'description': f"{rel}，{age}岁",
                'recommendation': recommendation,
                'lab_evidence_grade': lab_grade,
                'evidence_note': evidence_note,
                'ethnicity': member.get('ethnicity', '汉族'),
                'origin_altitude': member.get('origin_altitude', 350),
                'months_since_migration': member.get('months_since_migration', 0),
                'idu_status': member.get('idu_status', '否'),
                'workplace_type': member.get('workplace_type', '非油田'),
                'district': member.get('district', '克拉玛依区'),
            })

            processed += 1
            if total_contacts > 0:
                progress = 75 + int((processed / total_contacts) * 10)
                self._report_progress(progress,
                    f"分析接触者 {processed}/{total_contacts}...")

        for soc_idx, contact in enumerate(social_contacts):
            result = self.compute_single_contact_risk(
                contact, patient_treatment_days,
                default_contact_distance=self.DEFAULT_CONTACT_DISTANCE_SOCIAL,
                is_social_contact=True,
                is_mdr=is_mdr,
            )

            priority, recommendation = get_priority_and_recommendation(
                result['disease_probability'])

            if self._use_localization:
                recommendation = self.enhance_recommendation_karamay(
                    contact, priority, recommendation)

            is_diagnosed = contact.get('diagnosed', False)
            contact_id = contact.get('_id', None)
            if contact_id and not is_diagnosed:
                is_diagnosed = (self._contact_labels.get(contact_id, 0) == 1)

            contact_age_val = contact.get('age', None)
            # 问题六：record_id 作为层间契约的稳定唯一标识
            contact_record_id = contact.get('record_id') or contact_id or f"social_{soc_idx}"
            social_contact_info = {
                '_id': contact_id,
                'record_id': contact_record_id,
                'name': contact.get('name', '未知'),
                'age': contact_age_val,
                'freq_density': contact.get('freq_density', 0),
                'time_span': contact.get('time_span', 0),
                'is_diagnosed': is_diagnosed,
                'priority': priority,
                'risk_score': result['risk_score'],
                'infection_probability': result['infection_probability'],
                'rule_infection_probability': result['rule_infection_probability'],
                'seir_infection_probability': result['seir_infection_probability'],
                'disease_probability': result['disease_probability'],
                'latent_infection_prob': result['latent_infection_prob'],
                'short_term_active_risk': result['short_term_active_risk'],
                'description': '主要社会接触者',
                'recommendation': recommendation,
                'ethnicity': contact.get('ethnicity', '汉族'),
                'origin_altitude': contact.get('origin_altitude', 350),
                'months_since_migration': contact.get('months_since_migration', 0),
                'idu_status': contact.get('idu_status', '否'),
                'workplace_type': contact.get('workplace_type', '非油田'),
                'district': contact.get('district', '克拉玛依区'),
            }

            if contact_age_val is not None:
                social_contact_info['age'] = contact_age_val
                social_contact_info['description'] = (
                    f"{contact.get('name', '未知')}，{contact_age_val}岁")

            potential_patients['social'].append(social_contact_info)

            processed += 1
            if total_contacts > 0:
                progress = 75 + int((processed / total_contacts) * 10)
                self._report_progress(progress,
                    f"分析接触者 {processed}/{total_contacts}...")

        # 回退默认接触者生成：无实际接触者但高风险时生成默认条目
        if individual_risks_result:
            ind_risks = individual_risks_result.get('individual_risks', {})

            if not potential_patients['family'] and ind_risks.get('FCI', {}).get('risk', '') == '高风险':
                potential_patients['family'].append({
                    'record_id': 'default_family',
                    'name': '家庭成员', 'relationship': '家庭成员', 'age': 30,
                    'freq_density': 14, 'time_span': 4, 'is_diagnosed': False,
                    'priority': '高', 'description': '所有家庭成员',
                    'recommendation': '全部进行PPD/IGRA筛查和胸部X光检查',
                    'disease_probability': 50.0, 'infection_probability': 70.0
                })

            if not potential_patients['social']:
                snc_risk = ind_risks.get('SNC', {}).get('risk', '')
                hrsp_risk = ind_risks.get('HRSP', {}).get('risk', '')

                if snc_risk == '高风险':
                    potential_patients['social'].append({
                        'record_id': 'default_social_high',
                        'name': '主要社会接触者', 'age': 30, 'freq_density': 2,
                        'time_span': 4, 'is_diagnosed': False, 'priority': '高',
                        'description': '主要社会接触者', 'recommendation': '进行PPD/IGRA筛查',
                        'disease_probability': 40.0, 'infection_probability': 60.0
                    })
                elif hrsp_risk == '高风险':
                    potential_patients['social'].append({
                        'record_id': 'default_social_hrsp',
                        'name': '高风险社会接触者', 'age': 30, 'freq_density': 2,
                        'time_span': 4, 'is_diagnosed': False, 'priority': '高',
                        'description': '高风险社会接触者', 'recommendation': '进行PPD/IGRA筛查',
                        'disease_probability': 40.0, 'infection_probability': 60.0
                    })
                elif snc_risk == '中风险' or hrsp_risk == '中风险':
                    potential_patients['social'].append({
                        'record_id': 'default_social_medium',
                        'name': '部分社会接触者', 'age': 30, 'freq_density': 2,
                        'time_span': 4, 'is_diagnosed': False, 'priority': '中',
                        'description': '部分社会接触者', 'recommendation': '选择性进行PPD/IGRA筛查',
                        'disease_probability': 25.0, 'infection_probability': 40.0
                    })

        # 问题六-步1：出口 schema 约束。
        # 用 ContactRiskResult.from_dict() 包装每个接触者结果，使字段缺失/类型
        # 错误在 schema 层被捕获而非静默传播；to_dict() 保留裸 dict 形状以兼容
        # 既有下游消费者（summary 构建、GUI 表格、导出器、integrator 等）。
        # 这是 schemas.py 从"定义了没人用"到"在关键出口实际约束数据形状"的第一步。
        for ct in ('family', 'social'):
            wrapped = []
            for idx, item in enumerate(potential_patients.get(ct, [])):
                result_obj = ContactRiskResult.from_dict(
                    item, contact_type=ct, fallback_idx=idx)
                wrapped.append(result_obj.to_dict())
            potential_patients[ct] = wrapped

        return potential_patients

    # ==================== 单接触者风险 ====================

    def compute_single_contact_risk(
        self, contact: Dict, patient_treatment_days: float,
        default_contact_distance: str = 'medium',
        is_social_contact: bool = False,
        is_mdr: bool = False,
    ) -> Dict[str, float]:
        """计算单个接触者的风险

        副作用：会修改 contact 字典，添加 'bcg_note' 和 'risk_score' 键。

        参数：
            contact: 接触者记录
            patient_treatment_days: 源病例已治疗天数
            default_contact_distance: 默认接触距离
            is_social_contact: 是否社会接触者
            is_mdr: 源病例是否 MDR（genexpert_rif=resistant），
                用于拉长 SEIR 治疗传染性衰减曲线
        """
        risk_score = 0.0

        if contact.get('has_symptoms', 0) == 1:
            risk_score += SYMPTOMS_PRESENT_BONUS

        exposure_score = get_exposure_risk_score(
            contact.get('cumulative_exposure', 0))
        risk_score += exposure_score

        age = contact.get('age', 30)
        if age is None:
            age = 30
        try:
            age = float(age)
        except (TypeError, ValueError):
            age = 30

        if age < 5:
            risk_score += AGE_PROGRESSION_FACTORS['child_under_5'] / 2
        elif 5 <= age < 15:
            risk_score += AGE_PROGRESSION_FACTORS['child_5_14'] / 2
        elif 15 <= age <= 35:
            risk_score += AGE_PROGRESSION_FACTORS['young'] / 2
        elif 35 < age <= 65:
            risk_score += AGE_PROGRESSION_FACTORS.get('adult', 1.0) / 2
        elif age > 65:
            risk_score += AGE_PROGRESSION_FACTORS['elderly'] / 2

        is_immunosuppressed = False
        past_illness_type = str(contact.get('past_illness_type', 'none')).strip().lower()
        is_idu = _is_yes(contact.get('idu_status', '否'))
        if is_idu:
            is_immunosuppressed = True
        elif past_illness_type == 'hiv':
            is_immunosuppressed = True
        elif past_illness_type == 'immunosuppressants':
            is_immunosuppressed = True
        elif past_illness_type == 'diabetes':
            is_immunosuppressed = True
        elif _is_yes(contact.get('past_illness', 0)):
            is_immunosuppressed = True

        bcg_protection = 0.0
        bcg_note = ""
        contact_age_val = contact.get('age', 30)
        try:
            contact_age_val = float(contact_age_val)
        except (TypeError, ValueError):
            contact_age_val = 30.0
        bcg_vaccinated = _is_yes(contact.get('bcg_vaccine', 0))
        if not bcg_vaccinated:
            risk_score += ScoringEngine.NO_BCG_BONUS
            bcg_note = "未接种卡介苗（增加感染风险）"
        else:
            bcg_protection, bcg_note = self.calculate_bcg_protection(
                age=contact_age_val, bcg_vaccinated=True
            )
            if is_immunosuppressed:
                bcg_protection *= 0.5
                bcg_note += "（免疫状态低下，保护效果减弱）"

        if contact.get('has_tb', 0) == 1:
            risk_score += ScoringEngine.PAST_TB_BONUS

        if is_social_contact and contact.get('is_high_risk', 0) == 1 and not is_immunosuppressed:
            risk_score += 1.0

        contact['bcg_note'] = bcg_note
        contact['risk_score'] = risk_score

        # 距离/通风/场景因子
        contact_distance = convert_chinese_to_value(
            contact.get('contact_distance', default_contact_distance),
            dict(ScoringEngine.DISTANCE_FACTORS)
        )
        if contact_distance not in ScoringEngine.DISTANCE_FACTORS:
            contact_distance = default_contact_distance

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

        base_infection_prob = sigmoid_base_infection(risk_score)
        base_infection_prob *= (1 - bcg_protection)

        time_dependent_risk = self._seir.calculate_time_dependent_risk(
            contact, patient_treatment_days, is_mdr=is_mdr)
        rule_probability = (
            base_infection_prob * distance_factor
            * ventilation_factor * setting_factor
            * time_dependent_risk
        )
        rule_probability = min(rule_probability, 100.0)

        # SEIR 机制模型融合
        # 保留纯规则分(rule)与纯SEIR分(seir),供 integrator 三方向独立加权,
        # 避免将已融合的 infection_probability 当作"SEIR方向"再次加权导致SEIR贡献被双重计入
        p_seir = self._seir.compute_seir_probability(
            contact, patient_treatment_days, is_mdr=is_mdr)
        seir_infection_probability = rule_probability  # SEIR不可用时回退到规则分(保持旧行为)
        if p_seir is not None and p_seir > 0:
            w_seir = self._seir.compute_seir_confidence_weight()
            if w_seir > 0.01:
                p_rule = rule_probability / 100.0
                infection_probability = fuse_probabilities(
                    p_seir, p_rule, w_seir)
                seir_infection_probability = p_seir * 100.0  # 纯SEIR分(0-100)
            else:
                infection_probability = rule_probability
        else:
            infection_probability = rule_probability

        infection_probability = min(infection_probability, 100.0)

        # 进展为活动性结核的概率
        baseline_progress_rate = self.LATENT_TO_ACTIVE_BASELINE
        progression_multiplier = 1.0
        if is_idu:
            progression_multiplier = ScoringEngine.IMMUNO_FACTORS_BASE.get('idu', 10.0)
        elif past_illness_type == 'hiv':
            progression_multiplier = ScoringEngine.IMMUNO_FACTORS_BASE.get('hiv', 8.0)
        elif past_illness_type == 'immunosuppressants':
            progression_multiplier = ScoringEngine.IMMUNO_FACTORS_BASE.get(
                'immunosuppressants', 4.0)
        elif past_illness_type == 'diabetes':
            progression_multiplier = ScoringEngine.IMMUNO_FACTORS_BASE.get('diabetes', 2.5)
        elif _is_yes(contact.get('past_illness', 0)):
            progression_multiplier = ScoringEngine.IMMUNO_FACTORS_BASE.get('other', 1.8)

        age_factor = 1.0
        if contact_age_val < 5:
            age_factor = AGE_PROGRESSION_FACTORS['child_under_5']
        elif 5 <= contact_age_val < 15:
            age_factor = AGE_PROGRESSION_FACTORS['child_5_14']
        elif 15 <= contact_age_val <= 35:
            age_factor = AGE_PROGRESSION_FACTORS['young']
        elif 35 < contact_age_val <= 65:
            age_factor = AGE_PROGRESSION_FACTORS['adult']
        else:
            age_factor = AGE_PROGRESSION_FACTORS['elderly']

        combined_risk_factor = progression_multiplier * age_factor
        adjusted_progress_rate = min(
            baseline_progress_rate * (
                1 + (combined_risk_factor - 1) * ScoringEngine.PROGRESSION_SCALE_FACTOR
            ),
            ScoringEngine.MAX_PROGRESSION_RATE
        )
        disease_probability = max(0.0, min(
            infection_probability * adjusted_progress_rate,
            ScoringEngine.MAX_DISEASE_PROBABILITY
        ))

        # ─── 双输出进展建模（IGRA 感染门槛 / 时间衰减 / 生活方式） ───
        # 与 scoring/engine.py 的 compute_risk_score 对齐，输出两个临床更有用的量：
        #   latent_infection_prob  = P_infected：用 IGRA/TST 判定感染状态（未检测回退
        #                            暴露先验 infection_prob/100，保证与旧 exposure 推断连续）
        #   short_term_active_risk = P_infected × P_progress：给定已感染后的近期(1–2年)发病风险
        # P_progress 含时间衰减 × 年龄/免疫 × (1−BCG保护) × 生活方式 × 近期阳转放大。
        # 注：disease_probability 仍保留旧"固定基线 × 静态乘数"结构（供本土化增强的
        # base_progress 比例推算与既有下游兼容），新双输出作为附加临床量一并输出。
        exposure_prior = max(0.0, min(infection_probability / 100.0, 1.0))
        latent_infection_prob = latent_infection_probability(contact, exposure_prior)
        short_term_progress = short_term_progression_probability(
            contact,
            age_multiplier=combined_risk_factor,
            immuno_multiplier=1.0,          # 免疫乘数已并入 combined_risk_factor
            bcg_protective=bcg_protection,
            cap=ScoringEngine.MAX_PROGRESSION_RATE,
        )
        short_term_active_risk = latent_infection_prob * short_term_progress

        # 本土化增强
        if self._use_localization:
            for key, default in [
                ('ethnicity', '汉族'), ('origin_altitude', 350),
                ('months_since_migration', 0), ('idu_status', '否'),
                ('workplace_type', '非油田'), ('district', '克拉玛依区'),
            ]:
                if key not in contact:
                    contact[key] = default
            infection_probability, disease_probability = (
                self.apply_karamay_enhancements(
                    infection_probability, disease_probability, contact))

        return {
            'risk_score': risk_score,
            'infection_probability': infection_probability,
            'rule_infection_probability': rule_probability,
            'seir_infection_probability': seir_infection_probability,
            'disease_probability': disease_probability,
            'latent_infection_prob': latent_infection_prob,
            'short_term_active_risk': short_term_active_risk,
            'lab_evidence_grade': derive_lab_evidence_grade(contact),
        }

    # ==================== BCG 保护效力 ====================

    def calculate_bcg_protection(self, age, bcg_vaccinated=True, years_since_vaccination=None):
        """计算 BCG 接种的保护效力"""
        if not bcg_vaccinated:
            return 0.0, ""

        if years_since_vaccination is None:
            years_since_vaccination = max(0, age - 0.25)

        if age < 15:
            base_protection = (
                BCG_PROTECTION_PARAMS['child_severe_tb']
                + BCG_PROTECTION_PARAMS['child_pulmonary_tb']
            ) / 2
            bcg_note = "已接种卡介苗（儿童期，重症结核保护70-80%，肺结核保护50-60%）"
        else:
            base_protection = BCG_PROTECTION_PARAMS['adult_pulmonary_tb']
            bcg_note = "已接种卡介苗（成人期，肺结核保护效力约15%）"

        if years_since_vaccination <= 10:
            decay_factor = BCG_PROTECTION_PARAMS['decay_10_years']
            if years_since_vaccination > 0:
                bcg_note += f"（接种{int(years_since_vaccination)}年，保护效力衰减）"
        elif years_since_vaccination <= 20:
            decay_factor = BCG_PROTECTION_PARAMS['decay_10_20_years']
            bcg_note += f"（接种{int(years_since_vaccination)}年，保护效力显著衰减）"
        else:
            decay_factor = BCG_PROTECTION_PARAMS['decay_over_20_years']
            bcg_note += "（接种超过20年，基本无保护效力）"

        final_protection = base_protection * decay_factor
        return min(final_protection, 1.0), bcg_note

    # ==================== 本土化增强 ====================

    def apply_karamay_enhancements(self, infection_prob, disease_prob, contact_data):
        """应用克拉玛依本土化增强"""
        if not self._use_localization or self._localizer is None:
            return infection_prob, disease_prob

        kl = self._localizer
        if not kl.enabled:
            return infection_prob, disease_prob

        env = self.build_environment_config()
        pm10_val = env.get('pm10', 100)
        humidity_val = env.get('humidity', 45)
        month_val = env.get('month', 4)

        climate_beta = kl.climate.calculate_climate_multiplier(
            pm10=pm10_val, humidity=humidity_val,
            month=month_val, is_indoor=env.get('is_indoor', True))
        adj_infection = min(infection_prob * climate_beta, 100.0)

        eth = contact_data.get('ethnicity', contact_data.get('_ethnicity', '汉族'))
        eth_resolved = kl.ethnicity._resolve_ethnicity_key(eth)

        idu_raw = contact_data.get('idu_status', False)
        idu_bool = _is_yes(idu_raw)

        origin_alt = contact_data.get('origin_altitude')
        if origin_alt is not None:
            try:
                origin_alt = float(origin_alt)
            except (TypeError, ValueError):
                origin_alt = None

        months_since = contact_data.get('months_since_migration', 0)
        try:
            months_since = int(months_since) if months_since else 0
        except (TypeError, ValueError):
            months_since = 0

        combined = kl.get_combined_progression_multiplier(
            contact_data={'idu_status': idu_bool, 'ethnicity': eth_resolved},
            origin_altitude=origin_alt,
            months_since_migration=months_since,
            ethnicity=eth_resolved, strain_type='beijing')
        extra_mult = combined.get('multiplier', 1.0)

        if infection_prob > 0:
            base_progress = disease_prob / infection_prob
        else:
            base_progress = 0.0

        if base_progress >= self.LATENT_TO_ACTIVE_BASELINE and extra_mult != 1.0:
            clamped_progress = min(base_progress, 1.0)
            combined_original = (
                1.0 + (clamped_progress / self.LATENT_TO_ACTIVE_BASELINE - 1.0)
                / ScoringEngine.PROGRESSION_SCALE_FACTOR
            )
            combined_total = combined_original * extra_mult
            final_progress = min(
                self.LATENT_TO_ACTIVE_BASELINE * (
                    1.0 + (combined_total - 1.0) * ScoringEngine.PROGRESSION_SCALE_FACTOR
                ),
                ScoringEngine.MAX_PROGRESSION_RATE
            )
        else:
            final_progress = base_progress

        corrected_disease = min(
            adj_infection * final_progress,
            ScoringEngine.MAX_DISEASE_PROBABILITY
        )
        return adj_infection, corrected_disease

    def enhance_recommendation_karamay(self, contact, priority, base_recommendation):
        """通过克拉玛依本土化增强建议文本"""
        if not self._use_localization or self._localizer is None:
            return base_recommendation

        is_high_risk = (
            contact.get('is_high_risk', 0) == 1
            or _is_yes(contact.get('is_high_risk', '否'))
        )
        karamay_rec = self._localizer.enhance_recommendation({
            'priority': priority,
            'district': contact.get('district', '克拉玛依区'),
            'is_high_risk': is_high_risk,
            'workplace_type': contact.get('workplace_type', '非油田')
        })
        return f"{base_recommendation}；{karamay_rec}"

    # ==================== 环境配置 ====================

    def build_environment_config(self) -> Dict[str, Any]:
        """构建环境配置（气候参数等）"""
        cfg = {'pm10': 100, 'humidity': 45, 'month': 4, 'is_indoor': True}
        if self._use_localization and self._localizer:
            cfg.update(self._localizer.get_climate_defaults())
        return cfg
