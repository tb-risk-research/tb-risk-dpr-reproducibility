#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FHIR 术语绑定与转码

支持：
- 编码体系识别（LOINC/ICD-10/SNOMED/GB国标）
- 编码映射：不同编码体系间自动转码
- 未映射编码记录（日志+人工补充）
- ICD-10结核编码细化分类（A15-A19, B90）
- LOINC结核相关检验项目映射
- GB/T国家标准数据字典（性别/婚姻/民族/职业）
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from ..base import TERMINOLOGY_SYSTEMS

LOGGER = logging.getLogger("tb_risk.health_interop.fhir.terminology")


# ============================================================================
# ICD-10 结核编码详细分类表
# ============================================================================

# ICD-10结核编码详细信息：
# {code: {name, site, bacteriology, cdc_category, is_pulmonary}}
# site: pulmonary/extrapulmonary/pleural/miliary/sequela
# bacteriology: positive/negative/unknown (病原学阳性/阴性/未查)
# cdc_category: 对应疾控分类

ICD10_TB_DETAIL: Dict[str, Dict[str, str]] = {
    # === A15: 呼吸道结核，经细菌学和组织学证实 ===
    "A15.0": {
        "name": "肺结核，痰涂片镜检发现抗酸杆菌",
        "site": "pulmonary", "bacteriology": "positive",
        "cdc_category": "confirmed", "tb_type_cdc": "涂阳",
    },
    "A15.1": {
        "name": "肺结核，仅经培养证实",
        "site": "pulmonary", "bacteriology": "positive",
        "cdc_category": "confirmed", "tb_type_cdc": "培阳",
    },
    "A15.2": {
        "name": "肺结核，组织学证实",
        "site": "pulmonary", "bacteriology": "positive",
        "cdc_category": "confirmed", "tb_type_cdc": "病原学阳性",
    },
    "A15.3": {
        "name": "肺结核，经未特指方法证实",
        "site": "pulmonary", "bacteriology": "positive",
        "cdc_category": "confirmed", "tb_type_cdc": "病原学阳性",
    },
    "A15.4": {
        "name": "肺淋巴结结核，经细菌学和组织学证实",
        "site": "extrapulmonary", "bacteriology": "positive",
        "cdc_category": "confirmed", "tb_type_cdc": "肺外结核",
    },
    "A15.5": {
        "name": "喉、气管和支气管结核，经细菌学和组织学证实",
        "site": "extrapulmonary", "bacteriology": "positive",
        "cdc_category": "confirmed", "tb_type_cdc": "肺外结核",
    },
    "A15.6": {
        "name": "结核性胸膜炎，经细菌学和组织学证实",
        "site": "pleural", "bacteriology": "positive",
        "cdc_category": "confirmed", "tb_type_cdc": "结核性胸膜炎",
    },
    "A15.7": {
        "name": "原发性呼吸道结核，经细菌学和组织学证实",
        "site": "pulmonary", "bacteriology": "positive",
        "cdc_category": "confirmed", "tb_type_cdc": "病原学阳性",
    },
    "A15.8": {
        "name": "其他呼吸道结核，经细菌学和组织学证实",
        "site": "pulmonary", "bacteriology": "positive",
        "cdc_category": "confirmed", "tb_type_cdc": "病原学阳性",
    },
    "A15.9": {
        "name": "未特指的呼吸道结核，经细菌学和组织学证实",
        "site": "pulmonary", "bacteriology": "positive",
        "cdc_category": "confirmed", "tb_type_cdc": "病原学阳性",
    },
    
    # === A16: 呼吸道结核，未经细菌学或组织学证实 ===
    "A16.0": {
        "name": "肺结核，细菌学和组织学检查阴性",
        "site": "pulmonary", "bacteriology": "negative",
        "cdc_category": "clinical", "tb_type_cdc": "涂阴",
    },
    "A16.1": {
        "name": "肺结核，未做细菌学和组织学检查",
        "site": "pulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "未痰检",
    },
    "A16.2": {
        "name": "肺结核，未提及细菌学或组织学证实",
        "site": "pulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "未痰检",
    },
    "A16.3": {
        "name": "肺淋巴结结核，未经细菌学或组织学证实",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "肺外结核",
    },
    "A16.4": {
        "name": "喉、气管和支气管结核，未经细菌学或组织学证实",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "肺外结核",
    },
    "A16.5": {
        "name": "结核性胸膜炎，未经细菌学或组织学证实",
        "site": "pleural", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "结核性胸膜炎",
    },
    "A16.7": {
        "name": "原发性呼吸道结核，未经细菌学或组织学证实",
        "site": "pulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "涂阴",
    },
    "A16.8": {
        "name": "其他呼吸道结核，未经细菌学或组织学证实",
        "site": "pulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "涂阴",
    },
    "A16.9": {
        "name": "未特指的呼吸道结核，未经细菌学或组织学证实",
        "site": "pulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "未痰检",
    },
    
    # === A17: 神经系统结核 ===
    "A17.0": {
        "name": "结核性脑膜炎",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "结核性脑膜炎",
    },
    "A17.1": {
        "name": "脑膜结核瘤",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "肺外结核",
    },
    "A17.8": {
        "name": "其他神经系统结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "肺外结核",
    },
    "A17.9": {
        "name": "未特指的神经系统结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "肺外结核",
    },
    
    # === A18: 其他器官结核 ===
    "A18.0": {
        "name": "骨和关节结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "骨结核",
    },
    "A18.01": {
        "name": "脊柱结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "骨结核",
    },
    "A18.02": {
        "name": "髋关节结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "骨结核",
    },
    "A18.03": {
        "name": "膝关节结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "骨结核",
    },
    "A18.1": {
        "name": "泌尿生殖系统结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "泌尿生殖系统结核",
    },
    "A18.10": {
        "name": "肾结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "泌尿生殖系统结核",
    },
    "A18.11": {
        "name": "附睾结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "泌尿生殖系统结核",
    },
    "A18.12": {
        "name": "输卵管结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "泌尿生殖系统结核",
    },
    "A18.2": {
        "name": "结核性周围淋巴结病",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "淋巴结核",
    },
    "A18.20": {
        "name": "颈淋巴结结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "淋巴结核",
    },
    "A18.3": {
        "name": "肠、腹膜和肠系膜腺结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "腹腔结核",
    },
    "A18.4": {
        "name": "皮肤和皮下组织结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "皮肤结核",
    },
    "A18.40": {
        "name": "寻常狼疮",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "皮肤结核",
    },
    "A18.5": {
        "name": "眼结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "眼结核",
    },
    "A18.6": {
        "name": "耳结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "肺外结核",
    },
    "A18.7": {
        "name": "肾上腺结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "肺外结核",
    },
    "A18.8": {
        "name": "其他特指器官结核",
        "site": "extrapulmonary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "肺外结核",
    },
    
    # === A19: 粟粒性结核 ===
    "A19.0": {
        "name": "急性粟粒性结核（单一部位）",
        "site": "miliary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "血行播散性肺结核",
    },
    "A19.1": {
        "name": "急性粟粒性结核（多部位）",
        "site": "miliary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "血行播散性肺结核",
    },
    "A19.2": {
        "name": "急性粟粒性结核（未特指）",
        "site": "miliary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "血行播散性肺结核",
    },
    "A19.8": {
        "name": "其他粟粒性结核",
        "site": "miliary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "血行播散性肺结核",
    },
    "A19.9": {
        "name": "未特指的粟粒性结核",
        "site": "miliary", "bacteriology": "unknown",
        "cdc_category": "clinical", "tb_type_cdc": "血行播散性肺结核",
    },
    
    # === B90: 结核后遗症 ===
    "B90.0": {
        "name": "中枢神经系统结核后遗症",
        "site": "sequela", "bacteriology": "unknown",
        "cdc_category": "other", "tb_type_cdc": "陈旧性结核",
    },
    "B90.1": {
        "name": "泌尿生殖系统结核后遗症",
        "site": "sequela", "bacteriology": "unknown",
        "cdc_category": "other", "tb_type_cdc": "陈旧性结核",
    },
    "B90.2": {
        "name": "骨和关节结核后遗症",
        "site": "sequela", "bacteriology": "unknown",
        "cdc_category": "other", "tb_type_cdc": "陈旧性结核",
    },
    "B90.8": {
        "name": "其他器官结核后遗症",
        "site": "sequela", "bacteriology": "unknown",
        "cdc_category": "other", "tb_type_cdc": "陈旧性结核",
    },
    "B90.9": {
        "name": "未特指的呼吸道和未特指结核后遗症",
        "site": "sequela", "bacteriology": "unknown",
        "cdc_category": "other", "tb_type_cdc": "陈旧性结核",
    },
}

# 章节级编码（未细化亚目时使用默认分类）
ICD10_TB_CHAPTER: Dict[str, Dict[str, str]] = {
    "A15": {"name": "呼吸道结核，经细菌学和组织学证实", "site": "pulmonary", "bacteriology": "positive", "cdc_category": "confirmed"},
    "A16": {"name": "呼吸道结核，未经细菌学或组织学证实", "site": "pulmonary", "bacteriology": "unknown", "cdc_category": "clinical"},
    "A17": {"name": "神经系统结核", "site": "extrapulmonary", "bacteriology": "unknown", "cdc_category": "clinical"},
    "A18": {"name": "其他器官结核", "site": "extrapulmonary", "bacteriology": "unknown", "cdc_category": "clinical"},
    "A19": {"name": "粟粒性结核", "site": "miliary", "bacteriology": "unknown", "cdc_category": "clinical"},
    "B90": {"name": "结核后遗症", "site": "sequela", "bacteriology": "unknown", "cdc_category": "other"},
}


# ============================================================================
# LOINC 结核相关检验项目编码映射
# ============================================================================

LOINC_TB_TESTS: Dict[str, Dict[str, str]] = {
    # 痰涂片抗酸杆菌
    "641-3": {"name": "抗酸杆菌染色(涂片)", "short_name": "痰涂片AFB", "specimen": "痰", "test_type": "smear"},
    "642-1": {"name": "抗酸杆菌染色(浓缩涂片)", "short_name": "浓缩涂片AFB", "specimen": "痰", "test_type": "smear"},
    "11269-0": {"name": "抗酸杆菌染色(支气管肺泡灌洗液)", "short_name": "BALF AFB", "specimen": "BALF", "test_type": "smear"},
    
    # 结核分枝杆菌培养
    "11476-1": {"name": "结核分枝杆菌培养", "short_name": "结核菌培养", "specimen": "痰", "test_type": "culture"},
    "11477-9": {"name": "结核分枝杆菌培养(液体培养基)", "short_name": "液体培养", "specimen": "痰", "test_type": "culture"},
    "11478-7": {"name": "结核分枝杆菌培养(固体培养基)", "short_name": "罗氏培养", "specimen": "痰", "test_type": "culture"},
    
    # 分子生物学检测
    "47480-0": {"name": "结核分枝杆菌rpoB基因和耐药突变检测(Xpert MTB/RIF)", "short_name": "Xpert MTB/RIF", "specimen": "痰", "test_type": "molecular"},
    "77412-1": {"name": "结核分枝杆菌核酸检测", "short_name": "TB-DNA", "specimen": "痰", "test_type": "molecular"},
    "93351-1": {"name": "结核分枝杆菌线性探针耐药检测(MTBDRplus)", "short_name": "HAIN LPA", "specimen": "分离株", "test_type": "molecular"},
    "94501-0": {"name": "结核分枝杆菌全基因组测序", "short_name": "TB WGS", "specimen": "分离株", "test_type": "molecular"},
    
    # γ-干扰素释放试验
    "72420-9": {"name": "结核分枝杆菌γ-干扰素释放试验(Quantiferon)", "short_name": "QFT", "specimen": "全血", "test_type": "igra"},
    "79898-9": {"name": "结核分枝杆菌γ-干扰素释放试验(T-SPOT.TB)", "short_name": "T-SPOT", "specimen": "全血", "test_type": "igra"},
    
    # 结核菌素皮肤试验
    "64720-1": {"name": "结核菌素皮肤试验(PPD)", "short_name": "PPD皮试", "specimen": "皮肤", "test_type": "skin_test"},
    
    # 结核抗体
    "22435-9": {"name": "结核分枝杆菌抗体(IgG)", "short_name": "TB-IgG", "specimen": "血清", "test_type": "antibody"},
    "22436-7": {"name": "结核分枝杆菌抗体(IgM)", "short_name": "TB-IgM", "specimen": "血清", "test_type": "antibody"},
    
    # 药敏试验
    "18851-8": {"name": "结核分枝杆菌对异烟肼耐药性", "short_name": "INH药敏", "specimen": "分离株", "test_type": "dst", "drug": "INH"},
    "18852-6": {"name": "结核分枝杆菌对利福平耐药性", "short_name": "RFP药敏", "specimen": "分离株", "test_type": "dst", "drug": "RFP"},
    "18853-4": {"name": "结核分枝杆菌对乙胺丁醇耐药性", "short_name": "EMB药敏", "specimen": "分离株", "test_type": "dst", "drug": "EMB"},
    "18854-2": {"name": "结核分枝杆菌对吡嗪酰胺耐药性", "short_name": "PZA药敏", "specimen": "分离株", "test_type": "dst", "drug": "PZA"},
    "18855-9": {"name": "结核分枝杆菌对链霉素耐药性", "short_name": "SM药敏", "specimen": "分离株", "test_type": "dst", "drug": "SM"},
    "33947-8": {"name": "结核分枝杆菌对氟喹诺酮类耐药性", "short_name": "FQ药敏", "specimen": "分离株", "test_type": "dst", "drug": "FQ"},
    "34556-6": {"name": "结核分枝杆菌对阿米卡星耐药性", "short_name": "AMK药敏", "specimen": "分离株", "test_type": "dst", "drug": "AMK"},
    
    # HIV检测（结核患者常规筛查）
    "21490-5": {"name": "人类免疫缺陷病毒1+2抗体", "short_name": "HIV-Ab", "specimen": "血清", "test_type": "hiv"},
    "73904-1": {"name": "HIV抗原/抗体联合检测", "short_name": "HIV Ag/Ab", "specimen": "血清", "test_type": "hiv"},
    "57939-2": {"name": "HIV-1 RNA载量", "short_name": "HIV VL", "specimen": "血浆", "test_type": "hiv"},
    
    # 炎症指标
    "4537-7": {"name": "红细胞沉降率(ESR)", "short_name": "血沉", "specimen": "全血", "test_type": "inflammatory"},
    "1988-5": {"name": "C反应蛋白(CRP)", "short_name": "CRP", "specimen": "血清", "test_type": "inflammatory"},
    
    # 影像学
    "30745-8": {"name": "胸部X线检查报告", "short_name": "胸片", "specimen": "影像", "test_type": "imaging"},
    "24627-8": {"name": "胸部CT检查报告", "short_name": "胸部CT", "specimen": "影像", "test_type": "imaging"},
}


# ============================================================================
# 国家标准数据字典 (GB/T)
# ============================================================================

# GB/T 2261.1-2003 个人基本信息分类与代码 第1部分：人的性别代码
GB_GENDER_CODES: Dict[str, str] = {
    "0": "未知性别",
    "1": "男",
    "2": "女",
    "9": "未说明性别",
    "male": "1",
    "female": "2",
    "男": "1",
    "女": "2",
}

# GB/T 2261.2-2003 个人基本信息分类与代码 第2部分：婚姻状况代码
GB_MARITAL_CODES: Dict[str, str] = {
    "10": "未婚",
    "20": "已婚",
    "21": "初婚",
    "22": "再婚",
    "23": "复婚",
    "30": "丧偶",
    "40": "离婚",
    "90": "未说明婚姻状况",
}

# GB/T 3304-1991 中国各民族名称的罗马字母拼写法和代码（主要民族）
GB_ETHNICITY_CODES: Dict[str, str] = {
    "01": "汉族",
    "02": "蒙古族",
    "03": "回族",
    "04": "藏族",
    "05": "维吾尔族",
    "06": "苗族",
    "07": "彝族",
    "08": "壮族",
    "09": "布依族",
    "10": "朝鲜族",
    "11": "满族",
    "12": "侗族",
    "13": "瑶族",
    "14": "白族",
    "15": "土家族",
    "16": "哈尼族",
    "17": "哈萨克族",
    "18": "傣族",
    "19": "黎族",
    "20": "傈僳族",
    "30": "佤族",
    "39": "柯尔克孜族",
    "49": "其他",
    "99": "未识别民族",
}

# GB/T 6565-2015 职业分类与代码（结核高危职业简化版）
GB_OCCUPATION_CODES: Dict[str, str] = {
    "1-00": "国家机关、党群组织、企业、事业单位负责人",
    "1-05": "医务人员",  # 高危
    "2-00": "专业技术人员",
    "2-02": "教学人员",
    "3-00": "办事人员和有关人员",
    "4-00": "商业、服务业人员",
    "5-00": "农、林、牧、渔、水利业生产人员",
    "5-04": "畜牧业生产人员",  # 高危
    "5-12": "家禽饲养人员",  # 高危
    "6-00": "生产、运输设备操作人员及有关人员",
    "6-17": "矿山作业人员",  # 高危：矿工/矽肺
    "6-23": "石油天然气开采人员",
    "7-00": "军人",
    "8-00": "学生",
    "9-00": "其他从业人员",
    "9-03": "无业/待业人员",
    "9-04": "离退休人员",
    "9-05": "自由职业者",
    "9-99": "不便分类的其他从业人员",
}

# 疾控传染病报告卡专用代码 - 病例分类
CDC_CASE_CLASSIFICATION_CODES: Dict[str, str] = {
    "1": "疑似病例",
    "2": "临床诊断病例",
    "3": "确诊病例",
    "4": "病原携带者",
    "5": "阳性检测",
}

# 疾控传染病报告卡专用代码 - 患者归属
CDC_PATIENT_AFFILIATION: Dict[str, str] = {
    "1": "本县区",
    "2": "本市其他县区",
    "3": "本省其他地市",
    "4": "其他省",
    "5": "港澳台",
    "6": "外籍",
}

# 疾控病例分类代码（结核专用）
TB_TYPE_CDC_CODES: Dict[str, str] = {
    "11": "涂阳",
    "12": "仅培阳",
    "13": "仅分子生物学阳性",
    "21": "涂阴培阴",
    "22": "未痰检",
    "31": "结核性胸膜炎",
    "32": "肺外结核（不含胸膜炎）",
    "33": "血行播散性结核",
}


class TBICD10Classifier:
    """ICD-10结核编码分类器
    
    提供ICD-10编码到结核分类的详细映射。
    """
    
    @staticmethod
    def classify(icd_code: str) -> Optional[Dict[str, str]]:
        """分类一个ICD-10编码。
        
        参数：
            icd_code: ICD-10编码（如"A15.0"、"B90"）
            
        返回：
            Dict: {name, site, bacteriology, cdc_category, tb_type_cdc}
            None: 非结核编码
        """
        if not icd_code:
            return None
        
        code = icd_code.strip().upper()
        
        # 先精确匹配亚目
        if code in ICD10_TB_DETAIL:
            return ICD10_TB_DETAIL[code].copy()
        
        # 匹配章节码（A15-A19, B90）
        chapter = code.split(".")[0]
        if chapter in ICD10_TB_CHAPTER:
            return ICD10_TB_CHAPTER[chapter].copy()
        
        return None
    
    @staticmethod
    def is_tb_code(icd_code: str) -> bool:
        """判断是否为结核相关ICD-10编码"""
        return TBICD10Classifier.classify(icd_code) is not None
    
    @staticmethod
    def is_active_tb(icd_code: str) -> bool:
        """判断是否为活动性结核（非后遗症B90）"""
        info = TBICD10Classifier.classify(icd_code)
        if not info:
            return False
        return info["site"] != "sequela"
    
    @staticmethod
    def is_pulmonary(icd_code: str) -> bool:
        """判断是否为肺结核（含胸膜结核、粟粒性结核）"""
        info = TBICD10Classifier.classify(icd_code)
        if not info:
            return False
        return info["site"] in ("pulmonary", "pleural", "miliary")
    
    @staticmethod
    def is_bacteriologically_positive(icd_code: str) -> bool:
        """判断是否为病原学阳性（A15类）"""
        info = TBICD10Classifier.classify(icd_code)
        if not info:
            return False
        return info["bacteriology"] == "positive"
    
    @staticmethod
    def get_cdc_tb_type(icd_code: str) -> str:
        """获取疾控报卡结核类型代码"""
        info = TBICD10Classifier.classify(icd_code)
        if not info:
            return ""
        return info.get("tb_type_cdc", "")


class FHIRTermMapper:
    """FHIR术语映射器"""

    def __init__(self):
        # 内置映射表：{(from_system, from_code): (to_system, to_code)}
        self._mappings: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._unmapped: List[Dict[str, str]] = []
        self._load_default_mappings()

    def _load_default_mappings(self):
        """加载内置映射（覆盖常见检验项目和诊断）"""
        # 性别编码
        self._mappings.update({
            ("http://hl7.org/fhir/administrative-gender", "male"): (TERMINOLOGY_SYSTEMS.INTERNAL, "male"),
            ("http://hl7.org/fhir/administrative-gender", "female"): (TERMINOLOGY_SYSTEMS.INTERNAL, "female"),
            ("http://hl7.org/fhir/administrative-gender", "other"): (TERMINOLOGY_SYSTEMS.INTERNAL, "other"),
            ("http://hl7.org/fhir/administrative-gender", "unknown"): (TERMINOLOGY_SYSTEMS.INTERNAL, "unknown"),
        })
        
        # FHIR性别 → GB/T 2261.1国标编码
        for fhir_code, gb_code in [("male", "1"), ("female", "2"), ("other", "0"), ("unknown", "9")]:
            self._mappings[("http://hl7.org/fhir/administrative-gender", fhir_code)] = (
                "http://www.stats.gov.cn/gbt/2261.1", gb_code)
        
        # ICD-10结核亚目编码映射到内部码（统一为章节码）
        for code, detail in ICD10_TB_DETAIL.items():
            chapter = code.split(".")[0]
            self._mappings[(TERMINOLOGY_SYSTEMS.ICD10, code)] = (TERMINOLOGY_SYSTEMS.INTERNAL, chapter)
            self._mappings[(TERMINOLOGY_SYSTEMS.ICD10_CM, code)] = (TERMINOLOGY_SYSTEMS.INTERNAL, chapter)

    def map_code(self, code: str, from_system: str, to_system: str) -> Optional[str]:
        """将编码从一个体系映射到另一个体系

        返回目标编码，若未映射则返回None并记录到未映射列表。
        """
        if not code or not from_system:
            return None
        if self._systems_match(from_system, to_system):
            return code
        key = (self._normalize_system(from_system), code)
        mapped = self._mappings.get(key)
        if mapped and self._systems_match(mapped[0], to_system):
            return mapped[1]
        self._unmapped.append({
            "from_system": from_system,
            "code": code,
            "to_system": to_system,
        })
        LOGGER.debug("未映射编码: %s@%s → %s", code, from_system, to_system)
        return None

    def classify_tb_icd10(self, icd_code: str) -> Optional[Dict[str, str]]:
        """分类ICD-10结核编码（返回详细信息）"""
        return TBICD10Classifier.classify(icd_code)

    def get_loinc_test_info(self, loinc_code: str) -> Optional[Dict[str, str]]:
        """获取LOINC检验项目详细信息"""
        return LOINC_TB_TESTS.get(loinc_code)

    def map_gender_to_gb(self, gender_code: str, from_system: str = None) -> str:
        """映射性别编码到GB/T 2261.1国标"""
        if gender_code in GB_GENDER_CODES and GB_GENDER_CODES[gender_code] in ("0", "1", "2", "9"):
            return GB_GENDER_CODES[gender_code]
        if gender_code in ("1", "2", "0", "9"):
            return gender_code
        mapped = self.map_code(gender_code, from_system or "http://hl7.org/fhir/administrative-gender",
                              "http://www.stats.gov.cn/gbt/2261.1")
        return mapped or "9"

    def register_mapping(self, from_system: str, from_code: str,
                         to_system: str, to_code: str):
        """注册人工补充的映射规则"""
        self._mappings[(self._normalize_system(from_system), from_code)] = (
            self._normalize_system(to_system), to_code)

    def get_unmapped_codes(self) -> List[Dict[str, str]]:
        """获取所有未映射的编码（供人工补充映射表）"""
        return list(self._unmapped)

    def clear_unmapped(self):
        """清空未映射记录"""
        self._unmapped.clear()

    @staticmethod
    def _normalize_system(system: str) -> str:
        """标准化术语系统URI"""
        if not system:
            return ""
        return system.rstrip("/").lower()

    @classmethod
    def _systems_match(cls, s1: str, s2: str) -> bool:
        return cls._normalize_system(s1) == cls._normalize_system(s2)
