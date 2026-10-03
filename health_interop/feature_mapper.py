#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FeatureMapper：将AdapterResult/PatientRecord映射为tb_risk核心特征字典

负责将医疗接口返回的结构化临床数据转换为tb_risk评分引擎所需的：
1. 患者基本信息字段（basic_info_vars）
2. 标准化接触者条目（family/social contacts）
3. 接触者ML特征字典（供extract_features使用）

处理内容：
- 人口学映射：年龄、性别
- 临床指标映射：痰涂片、空洞、治疗状态、症状严重度、咳嗽频率
- 检验结果映射：从LOINC编码的检验结果推导痰涂片/结核诊断/HIV/糖尿病状态
- 影像征象映射：从影像NLP结果推导空洞状态
- 诊断映射：从ICD-10编码推导活动性结核
- 用药映射：从抗结核用药推导治疗状态及时长
- 接触者映射：关系、年龄、接触特征
- 缺失值填充：所有字段均有合理医学默认值，不抛KeyError
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    AdapterResult, calculate_age, GENDER_MAPPING, ICD10_TB_CODES,
    timestamp_to_datetime, safe_parse_datetime, mask_sensitive,
)

LOGGER = logging.getLogger("tb_risk.health_interop.feature_mapper")

# ============================================================================
# 默认值（基于普通门诊初诊患者的保守假设）
# ============================================================================

PATIENT_DEFAULTS: Dict[str, Any] = {
    "age": 40,
    "gender": 1,  # 0=女, 1=男
    "sputum_smear": "1",  # 1=涂阴, 2=涂阳
    "has_cavity": "1",    # 1=无, 2=有
    "active_tb": "2",     # 1=是, 2=否（诊断前默认否，诊断ICD-10命中时覆盖）
    "treatment": "2",     # 1=是, 2=否
    "treatment_duration": 0,  # 月
    "cough_freq": 0,      # 次/小时
    "symptoms": "2",      # 1=无症状, 2=轻度, 3=中度, 4=重度
    "delay_days": 0,
    "ventilation": "3",   # 1-5
    "symptom_description": "",
    "family_living_conditions": "3",  # 1-5
    "flp_percentage": 5,
    "hrsp_percentage": 10,
    "patient_name": "",
    # 新增扩展字段
    "bcg_scar": "2",      # 1=有卡痕/接种, 2=无/未知
    "smoking_years": 0,   # 吸烟年数
    "drinking_years": 0,  # 饮酒年数
    "bmi": 22.0,          # BMI指数
    "height_cm": 0,       # 身高(cm)
    "weight_kg": 0,       # 体重(kg)
    "ethnicity": "",      # 民族
    "occupation": "",     # 职业
    "idu_history": "2",   # 静脉吸毒史 1=是, 2=否
    "contact_type": "none",  # 接触类型: household_home/household_out/social/none
    "contact_with_tb": "2",  # 是否接触涂阳患者 1=是, 2=否
    "contact_duration_months": 0,  # 密切接触时长(月)
    # ---- 新增：影像学详细征象 ----
    "imaging_infiltrate": "2",   # 浸润影 1=有, 2=无
    "imaging_nodule": "2",       # 结节影 1=有, 2=无
    "imaging_consolidation": "2", # 实变 1=有, 2=无
    "imaging_pleural_effusion": "2", # 胸腔积液 1=有, 2=无
    "imaging_upper_lobe": "2",   # 上叶病灶（结核好发部位）1=有, 2=无
    "imaging_bilateral": "2",    # 双肺病灶 1=有, 2=无
    "imaging_calcification": "2", # 钙化灶（陈旧性）1=有, 2=无
    # ---- 新增：症状特征 ----
    "symptom_cough": "2",        # 咳嗽 1=有, 2=无
    "symptom_sputum": "2",       # 咳痰 1=有, 2=无
    "symptom_hemoptysis": "2",   # 咯血 1=有, 2=无
    "symptom_fever": "2",        # 发热 1=有, 2=无
    "symptom_night_sweat": "2",  # 盗汗 1=有, 2=无
    "symptom_weight_loss": "2",  # 体重下降 1=有, 2=无
    "symptom_fatigue": "2",      # 乏力 1=有, 2=无
    "symptom_anorexia": "2",     # 食欲减退 1=有, 2=无
    "symptom_chest_pain": "2",   # 胸痛 1=有, 2=无
    "symptom_dyspnea": "2",      # 呼吸困难 1=有, 2=无
    "symptom_count": 0,          # 症状数量
    # ---- 新增：既往病史与合并症 ----
    "past_tb_history": "2",      # 既往结核病史 1=有, 2=无
    "comorbidity_diabetes": "2", # 糖尿病 1=有, 2=无
    "comorbidity_hiv": "2",      # HIV感染 1=有, 2=无
    "comorbidity_immunosuppression": "2", # 免疫抑制（激素/免疫抑制剂/移植）1=有, 2=无
    "comorbidity_copd": "2",     # 慢阻肺/矽肺 1=有, 2=无
    "comorbidity_ckd": "2",      # 慢性肾病 1=有, 2=无
    "comorbidity_liver": "2",    # 慢性肝病 1=有, 2=无
    "comorbidity_malnutrition": "2", # 营养不良 1=有, 2=无
}

# 高危职业关键字映射
HIGH_RISK_OCCUPATION_KEYWORDS = {
    "medical_staff": ["医生", "护士", "医护", "医院", "医务人员", "doctor", "nurse", "medical", "healthcare"],
    "miner": ["矿工", " mining", "mine", "coal", "矽肺"],
    "oilfield_worker": ["油田", "石油", "钻井", "oilfield", "oil", "drilling", "石油工人"],
    "prison_inmate": ["羁押", "监狱", "看守所", "在押", "prison", "inmate", "detention"],
    "migrant_worker": ["民工", "流动", "外出务工", "migrant", "worker"],
    "teacher": ["教师", "老师", "school", "teacher"],
    "student": ["学生", "school", "student", "pupil"],
}

# 民族编码映射（简化版，覆盖主要民族）
ETHNICITY_MAP = {
    "汉": "han", "汉族": "han", "han": "han",
    "维": "uyghur", "维吾尔": "uyghur", "维吾尔族": "uyghur", "uyghur": "uyghur",
    "哈": "kazakh", "哈萨克": "kazakh", "哈萨克族": "kazakh", "kazakh": "kazakh",
    "回": "hui", "回族": "hui", "hui": "hui",
    "蒙": "mongol", "蒙古": "mongol", "蒙古族": "mongol", "mongol": "mongol",
    "藏": "tibetan", "藏族": "tibetan", "tibetan": "tibetan",
    "苗": "miao", "苗族": "miao", "miao": "miao",
    "彝": "yi", "彝族": "yi", "yi": "yi",
    "壮": "zhuang", "壮族": "zhuang", "zhuang": "zhuang",
}

# BCG接种史关键字
BCG_KEYWORDS = ["bcg", "卡介苗", "卡痕", "接种", "vaccinated", "scar", "有卡痕", "已接种"]

# 静脉吸毒关键字
IDU_KEYWORDS = ["idu", "静脉吸毒", "注射吸毒", "iv drug", "injecting drug", "吸毒", "drug use"]

# Combo字段值映射（显示标签 → 实际值）
COMBO_VALUE_MAPS = {
    "sputum_smear": {"1": "1", "2": "2", "阴性": "1", "涂阴": "1", "阳性": "2", "涂阳": "2",
                     "negative": "1", "positive": "2"},
    "has_cavity": {"1": "1", "2": "2", "无": "1", "否": "1", "有": "2", "是": "2",
                   "no": "1", "yes": "2"},
    "active_tb": {"1": "1", "2": "2", "是": "1", "否": "2", "yes": "1", "no": "2",
                  "active": "1", "inactive": "2", "活动性": "1"},
    "treatment": {"1": "1", "2": "2", "是": "1", "否": "2", "yes": "1", "no": "2",
                  "治疗中": "1", "未治疗": "2", "on_treatment": "1"},
    "symptoms": {"1": "1", "2": "2", "3": "3", "4": "4",
                 "无症状": "1", "轻度": "2", "中度": "3", "重度": "4",
                 "none": "1", "mild": "2", "moderate": "3", "severe": "4"},
    "ventilation": {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
                    "极差": "1", "差": "2", "一般": "3", "好": "4", "极好": "5"},
    "family_living_conditions": {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
                                  "非常拥挤": "1", "拥挤": "2", "一般": "3", "宽敞": "4",
                                  "非常宽敞": "5"},
}

# 接触距离关键字映射
CONTACT_DISTANCE_KEYWORDS = {
    "very_close": ["同床", "陪护", "护理", "密切", "very_close", "intimate", "household"],
    "close": ["家人", "同事", "同桌", "同办公室", "close", "daily"],
    "medium": ["同教室", "同病房", "medium"],
    "far": ["同楼", "同层", "偶尔", "far", "casual"],
    "distant": ["短暂接触", "擦肩而过", "distant", "brief"],
}

# 暴露场景关键字映射
EXPOSURE_SETTING_KEYWORDS = {
    "crowded": ["监狱", "看守所", "拘留所", "crowded", "prison", "detention"],
    "closed": ["医院", "诊室", "密闭", "closed", "hospital", "clinic"],
    "oilfield_camp": ["油田", "矿区", "工棚", "集体宿舍", "oilfield", "camp", "dormitory"],
    "outdoor": ["户外", "室外", "露天", "outdoor", "open_air"],
    "general": ["一般", "普通", "general"],
}

# TB相关ICD-10编码前缀
TB_ICD10_PREFIXES = ("A15", "A16", "A17", "A18", "A19", "A15.", "A16.", "A17.", "A18.", "A19.")

# 抗结核药物关键字
TB_MEDICATION_KEYWORDS = [
    "异烟肼", "利福平", "吡嗪酰胺", "乙胺丁醇", "链霉素",
    "isoniazid", "rifampicin", "pyrazinamide", "ethambutol", "streptomycin",
    "INH", "RFP", "PZA", "EMB", "SM", "HRZE", "抗结核",
]

# ---- 影像学征象关键字 ----
IMAGING_FEATURE_KEYWORDS = {
    "cavity": ["空洞", "cavity", "cavitation", "薄壁空洞", "厚壁空洞", "虫蚀样空洞"],
    "infiltrate": ["浸润", "渗出", "斑片", "片状影", "infiltrate", "infiltration",
                   "patchy", "opacity", "磨玻璃", "ggo", "ground glass"],
    "nodule": ["结节", "nodule", "nodular", "小结节", "粟粒", "miliary"],
    "consolidation": ["实变", "consolidation", "肺实变", "大叶性", "小叶性"],
    "pleural_effusion": ["胸腔积液", "胸水", "pleural effusion", "effusion", "胸膜积液"],
    "upper_lobe": ["上叶", "尖段", "后段", "upper lobe", "apical", "右上肺", "左上肺",
                   "肺尖", "锁骨下"],
    "bilateral": ["双肺", "两侧", "bilateral", "both lungs"],
    "calcification": ["钙化", "calcification", "calcified", "陈旧性", "陈旧结核",
                      "纤维钙化", "硬结灶"],
}

# ---- 症状关键字（中英文） ----
SYMPTOM_KEYWORDS = {
    "cough": ["咳嗽", "咳", "cough", "coughing"],
    "sputum": ["咳痰", "痰", "sputum", "phlegm", "有痰"],
    "hemoptysis": ["咯血", "咳血", "痰中带血", "血痰", "hemoptysis", "bloody sputum"],
    "fever": ["发热", "发烧", "低热", "高热", "午后低热", "fever", "temperature", "pyrexia"],
    "night_sweat": ["盗汗", "夜间出汗", "night sweat", "night sweating"],
    "weight_loss": ["体重下降", "消瘦", "体重减轻", "weight loss", "emaciation", "wasting"],
    "fatigue": ["乏力", "疲乏", "疲倦", "fatigue", "weakness", "tiredness", "无力"],
    "anorexia": ["食欲减退", "纳差", "食欲不振", "厌食", "anorexia", "poor appetite", "loss of appetite"],
    "chest_pain": ["胸痛", "胸疼", "chest pain", "pleuritic pain"],
    "dyspnea": ["呼吸困难", "气促", "气短", "气喘", "dyspnea", "shortness of breath", "sob", "喘息"],
}

# ---- 既往病史/合并症关键字 ----
COMORBIDITY_KEYWORDS = {
    "past_tb_history": ["既往结核", "结核病史", "既往患结核", "old tb", "previous tb",
                        "结核复发", "复治", "retreatment", "history of tuberculosis"],
    "diabetes": ["糖尿病", "diabetes", "dm", "diabetes mellitus", "糖代谢异常"],
    "hiv": ["hiv", "艾滋病", "人类免疫缺陷", "aids", "hiv positive", "hiv感染"],
    "immunosuppression": ["免疫抑制", "激素治疗", "免疫抑制剂", "器官移植", "移植术后",
                          "immunosuppression", "steroid", "transplant", "化疗后", "长期使用激素"],
    "copd": ["慢阻肺", "copd", "慢性阻塞性肺", "矽肺", "尘肺", "silicosis", "肺气肿",
             "emphysema", "慢性支气管炎"],
    "ckd": ["慢性肾病", "肾功能不全", "ckd", "chronic kidney disease", "肾衰", "尿毒症",
            "renal failure", "透析"],
    "liver": ["慢性肝病", "肝硬化", "chronic liver disease", "cirrhosis", "乙肝", "丙肝",
              "hepatitis b", "hepatitis c", "hbv", "hcv"],
    "malnutrition": ["营养不良", "低蛋白", "malnutrition", "消瘦", "贫血", "anemia"],
}


class FeatureMapper:
    """将AdapterResult映射为tb_risk核心特征字典"""

    def __init__(self):
        self.warnings: List[str] = []
        self.unmapped_fields: List[str] = []

    def map_patient_basic_info(self, result: AdapterResult) -> Dict[str, Any]:
        """将AdapterResult映射为患者基本信息字典（可直接用于basic_info_vars）

        返回字典中的值类型：
        - Int字段：int
        - Combo/String字段：str（"1"-"5" 或 "1"/"2"）
        - Text字段：str
        """
        self.warnings.clear()
        self.unmapped_fields.clear()
        features = dict(PATIENT_DEFAULTS)
        pi = result.patient_info or {}

        # ---- 人口学 ----
        features["patient_name"] = (pi.get("name") or pi.get("patient_name") or "").strip()
        self._map_gender(features, pi)
        self._map_age(features, pi, result)

        # ---- 临床检验结果 ----
        self._map_lab_results(features, result.lab_results or [])

        # ---- 影像报告（增强版：提取更多征象） ----
        self._map_imaging(features, result.imaging_reports or [])

        # ---- 诊断 ----
        self._map_diagnoses(features, result.diagnoses or [])

        # ---- 用药 ----
        self._map_medications(features, result.medications or [])

        # ---- 症状提取（从临床笔记、主诉、现病史） ----
        clinical_texts = self._collect_clinical_texts(pi, result)
        self._map_symptoms(features, clinical_texts)

        # ---- 既往病史/合并症 ----
        self._map_comorbidities(features, clinical_texts, pi)

        # ---- 症状描述 ----
        if result.diagnoses or result.imaging_reports:
            symptom_parts = []
            for dx in (result.diagnoses or [])[:3]:
                desc = dx.get("display") or dx.get("description") or ""
                if desc:
                    symptom_parts.append(desc)
            if symptom_parts:
                features["symptom_description"] = "; ".join(symptom_parts)

        # ---- 扩展特征映射 ----
        self._map_extended_features(features, pi, result)

        # ---- 延迟就诊估算 ----
        # 如果诊断日期和症状开始日期都有，计算延迟天数
        delay = self._estimate_delay_days(pi, result.diagnoses or [])
        if delay is not None:
            features["delay_days"] = max(0, min(delay, 365))

        if self.warnings:
            LOGGER.info("FeatureMapper映射完成，%d条警告: %s",
                        len(self.warnings), "; ".join(self.warnings[:5]))

        return features

    def map_contacts(self, result: AdapterResult) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """将AdapterResult中的接触者映射为标准化接触者条目

        返回 (family_entries, social_entries)，每个条目包含
        name/age/relationship/contact_type以及ML特征字段。
        """
        family = []
        social = []

        for c in (result.family_contacts or []):
            entry = self._standardize_contact_entry(c, "family")
            family.append(entry)
        for c in (result.social_contacts or []):
            entry = self._standardize_contact_entry(c, "social")
            social.append(entry)

        return family, social

    def map_to_contact_feature_dict(self, contact_entry: Dict[str, Any],
                                     contact_type: str = "family") -> Dict[str, Any]:
        """将标准化接触者条目映射为供extract_features使用的特征字典

        返回字典包含：age, cumulative_exposure, has_symptoms, bcg_vaccine,
        has_tb, contact_distance, ventilation, is_high_risk, past_illness,
        exposure_setting, single_duration, freq_density, time_span, past_illness_type
        """
        return self._entry_to_feature_dict(contact_entry, contact_type)

    # ========================================================================
    # 私有映射方法
    # ========================================================================

    def _map_gender(self, features: Dict[str, Any], pi: Dict[str, Any]):
        gender_raw = (pi.get("gender") or pi.get("sex") or "").strip().lower()
        if not gender_raw:
            return
        # 尝试中文/英文/编码映射
        if gender_raw in ("male", "m", "男", "1"):
            features["gender"] = 1
        elif gender_raw in ("female", "f", "女", "0"):
            features["gender"] = 0
        elif gender_raw in GENDER_MAPPING:
            mapped = GENDER_MAPPING[gender_raw]
            features["gender"] = 1 if mapped == "male" else 0
        else:
            self.unmapped_fields.append(f"gender={gender_raw}")

    def _map_age(self, features: Dict[str, Any], pi: Dict[str, Any],
                  result: AdapterResult):
        age = pi.get("age")
        if age is not None:
            try:
                features["age"] = max(0, min(int(age), 120))
                return
            except (TypeError, ValueError):
                pass

        # 尝试从出生日期计算
        birth_date = pi.get("birth_date") or pi.get("birthDate") or pi.get("dob")
        if birth_date:
            bd = safe_parse_datetime(str(birth_date))
            if bd:
                age = calculate_age(bd)
                if age is not None:
                    features["age"] = max(0, min(age, 120))
                    return

    def _map_lab_results(self, features: Dict[str, Any], labs: List[Dict[str, Any]]):
        """从检验结果推导痰涂片、糖尿病、HIV状态"""
        has_smear_result = False
        has_hiv = False
        has_diabetes = False
        any_tb_positive = False

        for lab in labs:
            code = str(lab.get("code") or lab.get("test_code") or lab.get("loinc") or "").upper()
            name = str(lab.get("name") or lab.get("test_name") or lab.get("display") or lab.get("item_name") or "")
            value = lab.get("value")
            interpretation = str(lab.get("interpretation") or lab.get("abnormal_flag") or "").lower()
            abnormal = lab.get("abnormal", lab.get("is_abnormal", False))
            is_positive = lab.get("is_positive")

            # 痰涂片查找：AFB/抗酸杆菌/痰涂片
            if ("AFB" in code or "抗酸" in name or "痰涂片" in name or
                    "smear" in name.lower() or "acid-fast" in name.lower() or
                    "涂片" in name):
                has_smear_result = True
                if self._is_positive_result(value, interpretation, abnormal, is_positive):
                    features["sputum_smear"] = "2"
                    any_tb_positive = True
                else:
                    # 阴性结果不覆盖阳性结果
                    if features["sputum_smear"] != "2":
                        features["sputum_smear"] = "1"

            # Xpert MTB/RIF
            if ("XPERT" in code or "MTB" in code or "GeneXpert" in name.upper() or
                    "结核核酸" in name or "xpert" in name.lower()):
                if self._is_positive_result(value, interpretation, abnormal, is_positive):
                    any_tb_positive = True
                    features["sputum_smear"] = "2"

            # 痰培养
            if ("CULTURE" in code or "培养" in name or "culture" in name.lower()):
                if self._is_positive_result(value, interpretation, abnormal, is_positive):
                    any_tb_positive = True
                    features["sputum_smear"] = "2"

            # HIV
            if ("HIV" in code or "HIV" in name.upper() or "人类免疫缺陷" in name or
                    "艾滋病" in name or "hiv" in name.lower()):
                if self._is_positive_result(value, interpretation, abnormal, is_positive):
                    has_hiv = True

            # 糖尿病：空腹血糖 > 7.0 mmol/L 或 HbA1c >= 6.5%
            if ("GLUCOSE" in code or "血糖" in name or "GLU" in code or
                    "glucose" in name.lower() or "HbA1c" in code.upper() or
                    "糖化血红蛋白" in name):
                if self._is_diabetes_indicative(value, name, code):
                    has_diabetes = True

        if any_tb_positive:
            features["active_tb"] = "1"
        # 糖尿病/HIV 状态作为高风险因素
        # 注意：这些影响contact features中的is_high_risk/past_illness_type，
        # 而非患者本身的basic_info，此处仅记录供后续使用
        if has_hiv or has_diabetes:
            features["_has_comorbidity"] = True
            features["_comorbidities"] = {
                "hiv": has_hiv,
                "diabetes": has_diabetes,
            }

    def _map_imaging(self, features: Dict[str, Any], imaging: List[Dict[str, Any]]):
        """从影像报告提取多种征象：空洞、浸润、结节、实变、胸腔积液、上叶病灶、双侧、钙化"""
        imaging_findings = {
            "cavity": False,
            "infiltrate": False,
            "nodule": False,
            "consolidation": False,
            "pleural_effusion": False,
            "upper_lobe": False,
            "bilateral": False,
            "calcification": False,
        }
        
        for img in imaging:
            conclusion = str(img.get("conclusion") or img.get("finding") or
                            img.get("impression") or img.get("report_text") or
                            img.get("nlp_result") or "")
            if not conclusion:
                continue
            conclusion_lower = conclusion.lower()
            
            for finding, keywords in IMAGING_FEATURE_KEYWORDS.items():
                if imaging_findings[finding]:
                    continue  # 已找到，跳过
                for kw in keywords:
                    if kw.lower() in conclusion_lower or kw in conclusion:
                        # 排除否定修饰（"未见空洞"、"无胸腔积液"等）
                        negated = False
                        for neg in ["未见", "无", "未发现", "no ", "without ", "未见明显", "排除"]:
                            neg_pos = conclusion_lower.find(neg)
                            kw_pos = conclusion_lower.find(kw.lower())
                            if neg_pos >= 0 and 0 <= kw_pos - neg_pos < 15:
                                negated = True
                                break
                        if not negated:
                            imaging_findings[finding] = True
                        break
        
        # 设置各影像学特征
        if imaging_findings["cavity"]:
            features["has_cavity"] = "2"
        if imaging_findings["infiltrate"]:
            features["imaging_infiltrate"] = "1"
        if imaging_findings["nodule"]:
            features["imaging_nodule"] = "1"
        if imaging_findings["consolidation"]:
            features["imaging_consolidation"] = "1"
        if imaging_findings["pleural_effusion"]:
            features["imaging_pleural_effusion"] = "1"
        if imaging_findings["upper_lobe"]:
            features["imaging_upper_lobe"] = "1"
        if imaging_findings["bilateral"]:
            features["imaging_bilateral"] = "1"
        # 钙化/陈旧性 → 提示陈旧结核，活动性降低
        if imaging_findings["calcification"] and not any([
            imaging_findings["cavity"], imaging_findings["infiltrate"],
            imaging_findings["consolidation"]
        ]):
            # 仅有钙化而无活动征象，不标记为活动
            features["imaging_calcification"] = "1"

    def _map_diagnoses(self, features: Dict[str, Any], diagnoses: List[Dict[str, Any]]):
        """从诊断列表推导活动性结核状态"""
        for dx in diagnoses:
            code = str(dx.get("code") or "").upper()
            display = str(dx.get("display") or dx.get("description") or dx.get("diagnosis") or "")
            # ICD-10结核编码
            for prefix in TB_ICD10_PREFIXES:
                if code.startswith(prefix):
                    features["active_tb"] = "1"
                    self._infer_symptom_severity(features, display)
                    return
            # 中文诊断关键字
            if "结核" in display and ("活动性" in display or "活动期" in display):
                features["active_tb"] = "1"
                self._infer_symptom_severity(features, display)
                return
            if ("肺结核" in display or "结核性" in display) and "陈旧" not in display and "已愈" not in display:
                features["active_tb"] = "1"
                self._infer_symptom_severity(features, display)

    def _map_medications(self, features: Dict[str, Any], medications: List[Dict[str, Any]]):
        """从用药记录推导治疗状态"""
        on_tb_treatment = False
        earliest_start = None

        for med in medications:
            name = str(med.get("name") or med.get("medication_name") or
                       med.get("display") or "").lower()
            for kw in TB_MEDICATION_KEYWORDS:
                if kw.lower() in name:
                    on_tb_treatment = True
                    start = med.get("start_date") or med.get("effective_date")
                    if start:
                        sd = safe_parse_datetime(str(start))
                        if sd and (earliest_start is None or sd < earliest_start):
                            earliest_start = sd
                    break

        if on_tb_treatment:
            features["treatment"] = "1"
            # 计算治疗时长
            if earliest_start:
                now = datetime.datetime.now()
                months = (now.year - earliest_start.year) * 12 + (now.month - earliest_start.month)
                features["treatment_duration"] = max(0, min(months, 24))

    def _infer_symptom_severity(self, features: Dict[str, Any], diagnosis_text: str):
        """从诊断/症状文本推断症状严重程度"""
        text = diagnosis_text.lower()
        if any(kw in text for kw in ["重症", "呼吸衰竭", "大咯血", "severe", "respiratory failure"]):
            features["symptoms"] = "4"
        elif any(kw in text for kw in ["中度", "空洞", "moderate"]):
            if features["symptoms"] < "3":
                features["symptoms"] = "3"
        elif any(kw in text for kw in ["咳嗽", "低热", "盗汗", "cough", "fever", "night sweat"]):
            if features["symptoms"] < "2":
                features["symptoms"] = "2"

    def _estimate_delay_days(self, pi: Dict[str, Any],
                              diagnoses: List[Dict[str, Any]]) -> Optional[int]:
        """估算延迟就诊天数"""
        symptom_onset = pi.get("symptom_onset") or pi.get("onset_date")
        diagnosis_date = None
        for dx in diagnoses:
            d = dx.get("onset_date") or dx.get("recorded_date") or dx.get("date")
            if d:
                dd = safe_parse_datetime(str(d))
                if dd:
                    diagnosis_date = dd
                    break
        if symptom_onset and diagnosis_date:
            so = safe_parse_datetime(str(symptom_onset))
            if so:
                delta = diagnosis_date - so
                return max(0, delta.days)
        return None

    def _collect_clinical_texts(self, pi: Dict[str, Any], result: AdapterResult) -> List[str]:
        """收集所有可用的临床文本（主诉、现病史、既往史、临床笔记等）用于NLP提取"""
        texts = []
        
        # patient_info中的文本字段
        for field in ["chief_complaint", "present_illness", "chief complaint", 
                      "history_present_illness", "hpi", "主诉", "现病史",
                      "past_history", "既往史", "personal_history", "个人史",
                      "notes", "remark", "symptoms", "症状描述"]:
            val = pi.get(field)
            if val and isinstance(val, str) and val.strip():
                texts.append(val.strip())
        
        # clinical_notes（从AdapterResult）
        for note in (result.clinical_notes or []):
            if isinstance(note, dict):
                content = note.get("content") or note.get("text") or note.get("note")
                if content:
                    texts.append(str(content))
            elif isinstance(note, str):
                texts.append(note)
        
        return texts
    
    def _map_symptoms(self, features: Dict[str, Any], clinical_texts: List[str]):
        """从临床文本中提取症状特征（基于关键字，后续可接入专业NLP）"""
        if not clinical_texts:
            return
        
        full_text = " ".join(t.lower() for t in clinical_texts)
        symptom_count = 0
        
        for symptom_key, keywords in SYMPTOM_KEYWORDS.items():
            found = False
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in full_text:
                    # 简单否定检查
                    negated = False
                    for neg in ["无", "未见", "否认", "no ", "without ", "无明显", "不伴"]:
                        neg_pos = full_text.find(neg)
                        kw_pos = full_text.find(kw_lower)
                        if neg_pos >= 0 and 0 <= kw_pos - neg_pos < 12:
                            negated = True
                            break
                    if not negated:
                        found = True
                        break
            
            field_name = f"symptom_{symptom_key}"
            if found:
                features[field_name] = "1"
                symptom_count += 1
            else:
                features[field_name] = "2"
        
        features["symptom_count"] = symptom_count
        
        # 根据症状数量推断症状严重程度
        if symptom_count >= 5 or features.get("symptom_hemoptysis") == "1":
            features["symptoms"] = "4"  # 重度
        elif symptom_count >= 3:
            features["symptoms"] = "3"  # 中度
        elif symptom_count >= 1:
            if features["symptoms"] < "2":
                features["symptoms"] = "2"  # 轻度
    
    def _map_comorbidities(self, features: Dict[str, Any], clinical_texts: List[str],
                           pi: Dict[str, Any]):
        """从临床文本和patient_info中提取既往病史/合并症"""
        # 首先从检验结果映射糖尿病/HIV（已在_map_lab_results中标记）
        comorbidities = features.get("_comorbidities", {})
        if comorbidities.get("hiv"):
            features["comorbidity_hiv"] = "1"
        if comorbidities.get("diabetes"):
            features["comorbidity_diabetes"] = "1"
        
        # 从patient_info直接字段读取
        direct_fields = {
            "past_tb_history": ["past_tb", "tb_history", "previous_tb", "既往结核", "结核病史"],
            "diabetes": ["diabetes", "dm", "糖尿病史"],
            "hiv": ["hiv", "hiv_status", "aids", "HIV史"],
            "immunosuppression": ["immunosuppressed", "steroid_use", "transplant", "免疫抑制", "激素"],
            "copd": ["copd", "矽肺", "尘肺", "慢阻肺"],
            "ckd": ["ckd", "renal_disease", "chronic_kidney", "肾病", "肾衰"],
            "liver": ["liver_disease", "hepatitis", "肝硬化", "肝病"],
            "malnutrition": ["malnutrition", "低蛋白", "营养不良", "贫血"],
        }
        
        for comorbi, keys in direct_fields.items():
            field_name = f"comorbidity_{comorbi}" if comorbi != "past_tb_history" else "past_tb_history"
            if features.get(field_name) == "1":
                continue
            for key in keys:
                val = pi.get(key)
                if val is not None:
                    if isinstance(val, bool) and val:
                        features[field_name] = "1"
                        break
                    if isinstance(val, str) and val.strip().lower() in ("yes", "true", "1", "是", "有", "阳性"):
                        features[field_name] = "1"
                        break
        
        # 从临床文本NLP提取
        if clinical_texts:
            full_text = " ".join(t.lower() for t in clinical_texts)
            
            for comorbi, keywords in COMORBIDITY_KEYWORDS.items():
                field_name = f"comorbidity_{comorbi}" if comorbi != "past_tb_history" else "past_tb_history"
                if features.get(field_name) == "1":
                    continue
                for kw in keywords:
                    kw_lower = kw.lower()
                    if kw_lower in full_text:
                        # 否定检查
                        negated = False
                        for neg in ["否认", "无", "无 ", "no ", "without ", "既往无", "不伴"]:
                            neg_pos = full_text.find(neg)
                            kw_pos = full_text.find(kw_lower)
                            if neg_pos >= 0 and 0 <= kw_pos - neg_pos < 12:
                                negated = True
                                break
                        if not negated:
                            features[field_name] = "1"
                            break
    
    def _map_extended_features(self, features: Dict[str, Any],
                                pi: Dict[str, Any], result: AdapterResult):
        """映射扩展特征：BCG史、吸烟/饮酒、BMI、民族、职业、IDU、接触史分类"""
        # BCG接种史/卡痕
        self._map_bcg_history(features, pi)

        # 吸烟/饮酒史
        self._map_smoking_drinking(features, pi)

        # BMI计算
        self._map_bmi(features, pi, result.lab_results or [])

        # 民族编码
        self._map_ethnicity(features, pi)

        # 职业和高危职业识别
        self._map_occupation(features, pi)

        # 静脉吸毒史
        self._map_idu_history(features, pi)

        # 接触史分类
        self._map_contact_history(features, result)

        # T-SPOT/结核感染检测结果
        self._map_tb_infection_tests(features, result.lab_results or [])

    def _map_bcg_history(self, features: Dict[str, Any], pi: Dict[str, Any]):
        """映射BCG接种史"""
        # 直接字段
        bcg_val = pi.get("bcg_scar") or pi.get("bcg_vaccine") or pi.get("bcg")
        if bcg_val is not None:
            if isinstance(bcg_val, bool):
                features["bcg_scar"] = "1" if bcg_val else "2"
                return
            if isinstance(bcg_val, (int, float)):
                features["bcg_scar"] = "1" if bcg_val else "2"
                return
            if isinstance(bcg_val, str):
                v = bcg_val.lower().strip()
                if v in ("yes", "true", "1", "y", "是", "有", "阳性", "vaccinated"):
                    features["bcg_scar"] = "1"
                    return
                if v in ("no", "false", "0", "n", "否", "无", "阴性", "unvaccinated"):
                    features["bcg_scar"] = "2"
                    return

        # 从immunization/疫苗史中查找
        immunizations = pi.get("immunizations", [])
        if isinstance(immunizations, list):
            for imm in immunizations:
                if isinstance(imm, dict):
                    code = str(imm.get("code") or imm.get("vaccine_code") or "").lower()
                    name = str(imm.get("display") or imm.get("name") or "").lower()
                    if "bcg" in code or "卡介苗" in name or "bcg" in name:
                        features["bcg_scar"] = "1"
                        return
                elif isinstance(imm, str):
                    if "bcg" in imm.lower() or "卡介苗" in imm:
                        features["bcg_scar"] = "1"
                        return

        # 从notes/备注中查找
        notes = str(pi.get("notes") or pi.get("remark") or "")
        for kw in BCG_KEYWORDS:
            if kw in notes.lower():
                # 有"无卡痕"等否定词
                if any(neg in notes for neg in ["无卡痕", "未接种", "否认", "no bcg", "无bcg"]):
                    features["bcg_scar"] = "2"
                else:
                    features["bcg_scar"] = "1"
                break

    def _map_smoking_drinking(self, features: Dict[str, Any], pi: Dict[str, Any]):
        """映射吸烟/饮酒史"""
        # 吸烟年数
        smoke_years = pi.get("smoking_years") or pi.get("smoke_years") or pi.get("pack_years")
        if smoke_years is not None:
            try:
                features["smoking_years"] = max(0, min(int(float(smoke_years)), 80))
            except (TypeError, ValueError):
                pass
        else:
            # 从smoking_history推导
            smoke_hist = str(pi.get("smoking") or pi.get("smoking_history") or "").lower()
            if smoke_hist in ("never", "no", "否", "无", "不吸烟", "从未吸烟"):
                features["smoking_years"] = 0
            elif smoke_hist in ("current", "former", "是", "有", "吸烟", "current smoker"):
                # 有吸烟史但无具体年数，根据年龄估算一个保守值
                age = features.get("age", 40)
                features["smoking_years"] = max(0, age - 25)  # 假设25岁开始吸烟

        # 饮酒年数
        drink_years = pi.get("drinking_years") or pi.get("alcohol_years")
        if drink_years is not None:
            try:
                features["drinking_years"] = max(0, min(int(float(drink_years)), 60))
            except (TypeError, ValueError):
                pass
        else:
            drink_hist = str(pi.get("drinking") or pi.get("alcohol") or pi.get("drinking_history") or "").lower()
            if drink_hist in ("never", "no", "否", "无", "不饮酒", "不喝酒"):
                features["drinking_years"] = 0

    def _map_bmi(self, features: Dict[str, Any], pi: Dict[str, Any],
                  labs: List[Dict[str, Any]]):
        """计算BMI（身高体重）"""
        # 身高：优先从patient_info获取
        height = pi.get("height") or pi.get("height_cm")
        if height is not None:
            try:
                h = float(height)
                if h > 3:  # cm
                    features["height_cm"] = h
                else:  # meters
                    features["height_cm"] = h * 100
            except (TypeError, ValueError):
                pass

        # 体重：优先从patient_info获取
        weight = pi.get("weight") or pi.get("weight_kg")
        if weight is not None:
            try:
                w = float(weight)
                features["weight_kg"] = w
            except (TypeError, ValueError):
                pass

        # 如果patient_info中没有，从lab_results通过LOINC编码查找
        # LOINC: 8302-2=身高(cm), 29463-7=体重(kg)
        if features["height_cm"] == 0 or features["weight_kg"] == 0:
            for lab in labs:
                loinc = str(lab.get("loinc") or lab.get("code") or "").strip()
                name = str(lab.get("name") or lab.get("test_name") or "").lower()
                val = lab.get("value")
                val_num = None
                try:
                    val_num = float(val) if val is not None else None
                except (TypeError, ValueError):
                    pass

                if features["height_cm"] == 0 and val_num:
                    if loinc == "8302-2" or "身高" in name or "height" in name:
                        if val_num > 3:
                            features["height_cm"] = val_num
                        elif val_num > 0:
                            features["height_cm"] = val_num * 100
                if features["weight_kg"] == 0 and val_num:
                    if loinc == "29463-7" or "体重" in name or "weight" in name:
                        features["weight_kg"] = val_num

        # BMI
        bmi = pi.get("bmi")
        if bmi is not None:
            try:
                features["bmi"] = round(float(bmi), 1)
            except (TypeError, ValueError):
                pass
        elif features["height_cm"] > 0 and features["weight_kg"] > 0:
            h_m = features["height_cm"] / 100.0
            features["bmi"] = round(features["weight_kg"] / (h_m * h_m), 1)

        # 从检验结果中查找BMI（如体检报告）
        if features["bmi"] == 22.0:  # 默认值，未找到
            for lab in labs:
                name = str(lab.get("name") or lab.get("test_name") or "").lower()
                code = str(lab.get("code") or "").lower()
                if "bmi" in name or "体重指数" in name or "bmi" in code:
                    try:
                        val = float(lab.get("value", 0))
                        if 10 < val < 60:
                            features["bmi"] = round(val, 1)
                            break
                    except (TypeError, ValueError):
                        pass

    def _map_ethnicity(self, features: Dict[str, Any], pi: Dict[str, Any]):
        """映射民族编码"""
        eth_raw = pi.get("ethnicity") or pi.get("race") or pi.get("nation") or pi.get("民族")
        if not eth_raw:
            # 尝试FHIR extension
            extensions = pi.get("extension", [])
            if isinstance(extensions, list):
                for ext in extensions:
                    if isinstance(ext, dict):
                        url = str(ext.get("url", "")).lower()
                        if "ethnicity" in url or "race" in url or "民族" in url:
                            eth_raw = ext.get("valueString") or ext.get("valueCode")
                            break
        if eth_raw:
            eth_str = str(eth_raw).strip()
            for key, code in ETHNICITY_MAP.items():
                if key in eth_str:
                    features["ethnicity"] = code
                    return
            # 未匹配但有值，保存原值
            features["ethnicity"] = eth_str

    def _map_occupation(self, features: Dict[str, Any], pi: Dict[str, Any]):
        """映射职业，识别高危职业"""
        occ_raw = pi.get("occupation") or pi.get("job") or pi.get("profession") or pi.get("职业")
        if not occ_raw:
            # 尝试FHIR extension
            extensions = pi.get("extension", [])
            if isinstance(extensions, list):
                for ext in extensions:
                    if isinstance(ext, dict):
                        url = str(ext.get("url", "")).lower()
                        if "occupation" in url or "job" in url or "职业" in url:
                            occ_raw = ext.get("valueString") or ext.get("valueCode")
                            break
        if occ_raw:
            occ_str = str(occ_raw).strip().lower()
            features["occupation"] = occ_str
            # 高危职业标记（供is_high_risk特征使用）
            for occ_type, keywords in HIGH_RISK_OCCUPATION_KEYWORDS.items():
                for kw in keywords:
                    if kw.lower() in occ_str:
                        features["_high_risk_occupation"] = occ_type
                        return

    def _map_idu_history(self, features: Dict[str, Any], pi: Dict[str, Any]):
        """映射静脉吸毒史"""
        idu_val = pi.get("idu") or pi.get("idu_history") or pi.get("iv_drug_use") or pi.get("drug_use")
        if idu_val is not None:
            if isinstance(idu_val, bool):
                features["idu_history"] = "1" if idu_val else "2"
                return
            if isinstance(idu_val, str):
                v = idu_val.lower().strip()
                if v in ("yes", "true", "1", "y", "是", "有", "current", "former"):
                    features["idu_history"] = "1"
                    return
                if v in ("no", "false", "0", "n", "否", "无", "never"):
                    features["idu_history"] = "2"
                    return
        # 从social_history或notes查找
        social_hist = str(pi.get("social_history") or pi.get("notes") or "").lower()
        for kw in IDU_KEYWORDS:
            if kw.lower() in social_hist:
                if any(neg in social_hist for neg in ["否认", "无", "no ", "never"]):
                    features["idu_history"] = "2"
                else:
                    features["idu_history"] = "1"
                break

    def _map_contact_history(self, features: Dict[str, Any], result: AdapterResult):
        """映射接触史分类"""
        has_household = False
        has_outside = False
        has_smear_positive = False
        max_contact_months = 0

        family_contacts = result.family_contacts or []
        social_contacts = result.social_contacts or []
        all_contacts = family_contacts + social_contacts

        if family_contacts:
            has_household = True
            contact_type = "household"
        if social_contacts:
            has_outside = True
            if not has_household:
                contact_type = "social"

        for c in all_contacts:
            # 接触时长
            duration = c.get("contact_duration_months") or c.get("time_span")
            if duration:
                try:
                    months = int(duration)
                    if months > max_contact_months:
                        max_contact_months = months
                except (TypeError, ValueError):
                    pass
            # 涂阳接触
            rel = str(c.get("relationship") or c.get("contact_relation") or "").lower()
            note = str(c.get("notes") or "").lower()
            if any(kw in rel + " " + note for kw in ["涂阳", "菌阳", "排菌", "smear positive", "culture positive"]):
                has_smear_positive = True
            if c.get("has_tb"):
                has_smear_positive = True

        if has_household and has_outside:
            features["contact_type"] = "household_both"
        elif has_household:
            features["contact_type"] = "household_home"
        elif has_outside:
            features["contact_type"] = "social"
        else:
            features["contact_type"] = "none"

        features["contact_with_tb"] = "1" if (has_household or has_outside) else "2"
        features["contact_with_smear_positive"] = "1" if has_smear_positive else "2"
        features["contact_duration_months"] = max_contact_months

    def _map_tb_infection_tests(self, features: Dict[str, Any], labs: List[Dict[str, Any]]):
        """映射T-SPOT/结核感染T细胞检测等结果"""
        for lab in labs:
            code = str(lab.get("code") or lab.get("test_code") or "").upper()
            name = str(lab.get("name") or lab.get("test_name") or "").lower()
            value = lab.get("value")
            interpretation = str(lab.get("interpretation") or "").lower()

            # T-SPOT.TB / 结核感染T细胞检测
            if ("tspot" in name or "t-spot" in name or "tspot" in code.lower() or
                "结核感染" in name or "γ干扰素" in name or "igra" in name.lower()):
                is_pos = self._is_positive_result(value, interpretation, lab.get("abnormal", False))
                features["_tspot_positive"] = is_pos

            # PPD / 结核菌素试验
            if ("ppd" in name or "结核菌素" in name or "ppd" in code.lower() or
                "tuberculin" in name.lower()):
                is_pos = self._is_positive_result(value, interpretation, lab.get("abnormal", False))
                features["_ppd_positive"] = is_pos

    def _is_positive_result(self, value: Any, interpretation: str, abnormal: bool, is_positive: Any = None) -> bool:
        """判断检验结果是否为阳性"""
        # 直接的is_positive标志优先
        if is_positive is True or (isinstance(is_positive, str) and is_positive.lower() in ("true", "1", "yes", "positive")):
            return True
        if interpretation in ("positive", "pos", "abnormal", "a", "h", "high", "阳性", "异常"):
            return True
        if abnormal:
            return True
        if isinstance(value, str):
            v = value.lower().strip()
            if v in ("positive", "pos", "+", "++", "+++", "阳性", "异常", "detected"):
                return True
            # 包含阳性/+/detected等关键词
            if ("阳性" in v or "检出" in v or "+" in v or
                "detected" in v or "abnormal" in v or "reactive" in v):
                return True
        return False

    def _is_diabetes_indicative(self, value: Any, name: str, code: str) -> bool:
        """判断检验结果是否提示糖尿病"""
        try:
            if value is None:
                return False
            val = float(value)
            name_lower = name.lower()
            if "hba1c" in code.lower() or "糖化" in name:
                return val >= 6.5
            if "glucose" in name_lower or "血糖" in name or "glu" in code.lower():
                return val >= 7.0
        except (TypeError, ValueError):
            pass
        return False

    # ========================================================================
    # 接触者映射
    # ========================================================================

    def _standardize_contact_entry(self, c: Dict[str, Any],
                                    contact_type: str) -> Dict[str, Any]:
        """将适配器接触者记录标准化为GUI接触者条目格式"""
        entry = dict(c)

        # 姓名
        if "name" not in entry:
            entry["name"] = (c.get("name") or c.get("contact_name") or
                           c.get("member_name") or "")

        # 年龄
        age = c.get("age")
        if age is None and c.get("birth_date"):
            age = calculate_age(c["birth_date"])
        if age is not None:
            try:
                entry["age"] = str(int(age))
            except (TypeError, ValueError):
                entry["age"] = ""
        else:
            entry["age"] = ""

        # 关系
        if "relationship" not in entry:
            rel = c.get("contact_relation") or c.get("relation") or ""
            entry["relationship"] = rel
        entry["contact_type"] = contact_type

        # 设置ML特征默认值（供_extract_features使用）
        entry.setdefault("has_symptoms", 0)
        entry.setdefault("bcg_vaccine", 0)
        entry.setdefault("has_tb", 0)
        entry.setdefault("is_high_risk", 0)
        entry.setdefault("past_illness", 0)
        entry.setdefault("past_illness_type", "none")

        # 接触距离
        entry["contact_distance"] = self._infer_contact_distance(c, contact_type)

        # 通风条件
        vent = c.get("ventilation")
        if vent is not None:
            try:
                v = int(vent)
                entry["ventilation"] = str(max(1, min(v, 5)))
            except (TypeError, ValueError):
                entry["ventilation"] = "3"
        else:
            entry["ventilation"] = "3"

        # 暴露场景
        entry["exposure_setting"] = self._infer_exposure_setting(c)

        # 接触时长参数
        self._infer_contact_duration(entry, c, contact_type)

        return entry

    def _entry_to_feature_dict(self, entry: Dict[str, Any],
                                contact_type: str) -> Dict[str, Any]:
        """将GUI接触者条目转换为extract_features所需的特征字典"""
        def _yes(val):
            if isinstance(val, bool):
                return 1 if val else 0
            if isinstance(val, (int, float)):
                return 1 if val else 0
            if isinstance(val, str):
                return 1 if val.lower() in ("yes", "true", "1", "是", "y", "t") else 0
            return 0

        def _int(val, default):
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        age = _int(entry.get("age"), 30)
        cum_exp = _int(entry.get("cumulative_exposure", 0), 0)

        return {
            "age": age,
            "cumulative_exposure": cum_exp,
            "has_symptoms": _yes(entry.get("has_symptoms", 0)),
            "bcg_vaccine": _yes(entry.get("bcg_vaccine", 0)),
            "has_tb": _yes(entry.get("has_tb", 0)),
            "contact_distance": entry.get("contact_distance",
                                          "close" if contact_type == "family" else "medium"),
            "ventilation": _int(entry.get("ventilation", 3), 3),
            "is_high_risk": _yes(entry.get("is_high_risk", 0)),
            "past_illness": _yes(entry.get("past_illness", 0)),
            "exposure_setting": entry.get("exposure_setting", "general"),
            "single_duration": _int(entry.get("single_duration", 30), 30),
            "freq_density": _int(entry.get("freq_density",
                                           14 if contact_type == "family" else 2),
                                14 if contact_type == "family" else 2),
            "time_span": _int(entry.get("time_span", 4), 4),
            "past_illness_type": entry.get("past_illness_type", "none"),
        }

    def _infer_contact_distance(self, c: Dict[str, Any], contact_type: str) -> str:
        """推断接触距离等级"""
        rel = str(c.get("contact_relation") or c.get("relationship") or "").lower()
        note = str(c.get("notes") or c.get("contact_note") or "").lower()
        combined = rel + " " + note

        # 家庭默认close，配偶/子女very_close
        if contact_type == "family":
            if any(kw in combined for kw in ["配偶", "spouse", "夫妻", "同床"]):
                return "very_close"
            if any(kw in combined for kw in ["子女", "child", "son", "daughter"]):
                return "close"
            return "close"
        else:
            # 社会接触根据关键字判断
            for level, keywords in CONTACT_DISTANCE_KEYWORDS.items():
                for kw in keywords:
                    if kw in combined:
                        return level
            return "medium"

    def _infer_exposure_setting(self, c: Dict[str, Any]) -> str:
        """推断暴露场景"""
        text = str(c.get("notes") or c.get("setting") or c.get("contact_place") or "").lower()
        for setting, keywords in EXPOSURE_SETTING_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return setting
        return "general"

    def _infer_contact_duration(self, entry: Dict[str, Any], c: Dict[str, Any],
                                  contact_type: str):
        """推断接触时长参数（单次分钟数、频率、时间跨度月数）"""
        try:
            entry["single_duration"] = int(c.get("single_duration", 30))
        except (TypeError, ValueError):
            entry["single_duration"] = 30

        try:
            entry["freq_density"] = int(c.get("freq_density",
                                               14 if contact_type == "family" else 2))
        except (TypeError, ValueError):
            entry["freq_density"] = 14 if contact_type == "family" else 2

        try:
            entry["time_span"] = int(c.get("time_span", 4))
        except (TypeError, ValueError):
            entry["time_span"] = 4

        # 累积暴露时间（小时）= 单次分钟/60 × 频率（次/周）× 4周 × 月数
        entry["cumulative_exposure"] = round(
            entry["single_duration"] / 60.0 * entry["freq_density"] * 4 * entry["time_span"], 1
        )


def map_adapter_result_to_predictor_input(result: AdapterResult) -> Dict[str, Any]:
    """便捷函数：一步将AdapterResult映射为预测引擎输入

    返回字典包含：
        basic_info: Dict[str, Any] — 患者基本信息（可直接设置到basic_info_vars）
        family_contacts: List[Dict] — 标准化家庭接触者（含ML特征）
        social_contacts: List[Dict] — 标准化社会接触者（含ML特征）
        contact_features: List[Dict] — 所有接触者的ML特征字典列表
        warnings: List[str] — 映射警告
    """
    mapper = FeatureMapper()
    basic_info = mapper.map_patient_basic_info(result)
    family_entries, social_entries = mapper.map_contacts(result)

    contact_features = []
    for e in family_entries:
        contact_features.append(mapper.map_to_contact_feature_dict(e, "family"))
    for e in social_entries:
        contact_features.append(mapper.map_to_contact_feature_dict(e, "social"))

    return {
        "basic_info": basic_info,
        "family_contacts": family_entries,
        "social_contacts": social_entries,
        "contact_features": contact_features,
        "warnings": mapper.warnings,
        "unmapped_fields": mapper.unmapped_fields,
    }
