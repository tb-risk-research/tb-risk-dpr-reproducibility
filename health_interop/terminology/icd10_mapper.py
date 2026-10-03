#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ICD-10 疾病编码映射引擎

核心功能：
1. 中文诊断文本 → ICD-10 编码自动匹配（含同义词、别名、古中医名）
2. 编码前缀匹配与范围判断（如 A15-A19 结核病相关编码）
3. 编码-显示名称双向查询
4. 结核病专病编码优先匹配（A15-A19/B90）
5. 合并症编码识别（糖尿病 E10-E14，HIV B20-B24，矽肺 J62）

数据来源：
    ICD-10 中文版（国标版 ICD-10，卫健委发布）
    内嵌常见疾病同义词库，无需外部依赖即可运行
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("tb_risk.health_interop.terminology.icd10")

# ============================================================================
# 结核病相关 ICD-10 编码（优先级最高）
# ============================================================================

TB_ICD10_CODES: Dict[str, Dict[str, Any]] = {
    "A15": {
        "display": "肺结核，经细菌学和组织学证实",
        "aliases": ["菌阳肺结核", "涂阳肺结核", "培阳肺结核", "细菌学阳性肺结核",
                     "肺结核"],
        "category": "active_tb",
        "priority": 10,
    },
    "A15.0": {
        "display": "肺结核，经显微镜检查证实",
        "aliases": ["涂阳肺结核", "痰涂片阳性肺结核", "菌阳肺结核"],
        "category": "active_tb",
        "parent": "A15",
        "priority": 10,
    },
    "A15.1": {
        "display": "肺结核，仅经培养证实",
        "aliases": ["培阳肺结核", "培养阳性肺结核"],
        "category": "active_tb",
        "parent": "A15",
        "priority": 10,
    },
    "A15.2": {
        "display": "肺结核，经组织学证实",
        "aliases": ["组织学证实肺结核", "病理证实肺结核"],
        "category": "active_tb",
        "parent": "A15",
        "priority": 10,
    },
    "A15.3": {
        "display": "肺结核，经未特指的方法证实",
        "aliases": [],
        "category": "active_tb",
        "parent": "A15",
        "priority": 10,
    },
    "A16": {
        "display": "肺结核，未经细菌学或组织学证实",
        "aliases": ["菌阴肺结核", "涂阴肺结核", "培阴肺结核", "临床诊断肺结核"],
        "category": "active_tb",
        "priority": 9,
    },
    "A16.0": {
        "display": "肺结核，细菌学和组织学检查阴性",
        "aliases": ["菌阴肺结核", "涂阴培阴"],
        "category": "active_tb",
        "parent": "A16",
        "priority": 9,
    },
    "A16.1": {
        "display": "肺结核，未做细菌学和组织学检查",
        "aliases": ["未查痰肺结核"],
        "category": "active_tb",
        "parent": "A16",
        "priority": 9,
    },
    "A16.2": {
        "display": "肺结核，未提及细菌学或组织学证实",
        "aliases": ["临床诊断肺结核", "影像学诊断肺结核"],
        "category": "active_tb",
        "parent": "A16",
        "priority": 9,
    },
    "A17": {
        "display": "神经系统结核",
        "aliases": ["结核性脑膜炎", "结脑", "脑结核", "脊髓结核", "神经结核"],
        "category": "active_tb",
        "priority": 9,
    },
    "A17.0": {
        "display": "结核性脑膜炎",
        "aliases": ["结脑", "结核性脑膜脑炎"],
        "category": "active_tb",
        "parent": "A17",
        "priority": 9,
    },
    "A18": {
        "display": "其他器官结核",
        "aliases": ["肺外结核", "淋巴结核", "骨结核", "肾结核", "肠结核",
                     "结核性腹膜炎", "结核性胸膜炎", "胸壁结核"],
        "category": "active_tb",
        "priority": 9,
    },
    "A18.0": {
        "display": "骨和关节结核",
        "aliases": ["骨结核", "脊柱结核", "关节结核", "Pott病"],
        "category": "active_tb",
        "parent": "A18",
        "priority": 9,
    },
    "A18.1": {
        "display": "泌尿生殖系统结核",
        "aliases": ["肾结核", "输尿管结核", "膀胱结核", "附睾结核", "输卵管结核"],
        "category": "active_tb",
        "parent": "A18",
        "priority": 9,
    },
    "A18.2": {
        "display": "结核性周围淋巴结病",
        "aliases": ["淋巴结核", "颈部淋巴结核", "瘰疬"],
        "category": "active_tb",
        "parent": "A18",
        "priority": 9,
    },
    "A18.3": {
        "display": "肠、腹膜和肠系膜淋巴结结核",
        "aliases": ["肠结核", "结核性腹膜炎", "肠系膜淋巴结核"],
        "category": "active_tb",
        "parent": "A18",
        "priority": 9,
    },
    "A18.4": {
        "display": "皮肤和皮下组织结核",
        "aliases": ["皮肤结核", "寻常狼疮"],
        "category": "active_tb",
        "parent": "A18",
        "priority": 9,
    },
    "A19": {
        "display": "粟粒性结核",
        "aliases": ["血行播散型肺结核", "粟粒性肺结核", "急性粟粒性结核"],
        "category": "active_tb",
        "priority": 10,
    },
    "B90": {
        "display": "结核病后遗症",
        "aliases": ["陈旧性肺结核", "陈旧结核", "结核钙化灶", "结核后遗症",
                     "纤维空洞型肺结核后遗症"],
        "category": "past_tb",
        "priority": 5,
    },
    "B90.0": {
        "display": "中枢神经系统结核后遗症",
        "aliases": ["结核性脑膜炎后遗症"],
        "category": "past_tb",
        "parent": "B90",
        "priority": 5,
    },
    "B90.9": {
        "display": "呼吸系统结核后遗症",
        "aliases": ["陈旧性肺结核", "肺结核钙化", "纤维化肺结核"],
        "category": "past_tb",
        "parent": "B90",
        "priority": 5,
    },
}

# ============================================================================
# 合并症相关 ICD-10 编码
# ============================================================================

COMORBIDITY_ICD10: Dict[str, Dict[str, Any]] = {
    # 糖尿病
    "E10": {
        "display": "1型糖尿病",
        "aliases": ["1型糖尿病", "胰岛素依赖型糖尿病", "青少年糖尿病", "I型糖尿病"],
        "comorbidity": "diabetes",
        "priority": 7,
    },
    "E11": {
        "display": "2型糖尿病",
        "aliases": ["2型糖尿病", "非胰岛素依赖型糖尿病", "成人发病型糖尿病",
                     "II型糖尿病", "糖尿病2型", "消渴症", "消渴病"],
        "comorbidity": "diabetes",
        "priority": 7,
    },
    "E12": {
        "display": "营养不良相关性糖尿病",
        "aliases": ["营养不良糖尿病"],
        "comorbidity": "diabetes",
        "priority": 7,
    },
    "E13": {
        "display": "其他特指糖尿病",
        "aliases": ["其他糖尿病", "继发性糖尿病"],
        "comorbidity": "diabetes",
        "priority": 7,
    },
    "E14": {
        "display": "未特指糖尿病",
        "aliases": ["糖尿病", "糖尿病（未特指型）", "血糖高"],
        "comorbidity": "diabetes",
        "priority": 6,
    },
    # HIV
    "B20": {
        "display": "人类免疫缺陷病毒病造成的传染病和寄生虫病",
        "aliases": ["HIV感染", "艾滋病", "AIDS"],
        "comorbidity": "hiv",
        "priority": 8,
    },
    "B21": {
        "display": "人类免疫缺陷病毒病造成的恶性肿瘤",
        "aliases": ["HIV相关恶性肿瘤"],
        "comorbidity": "hiv",
        "priority": 8,
    },
    "B22": {
        "display": "人类免疫缺陷病毒病造成的其他特指疾病",
        "aliases": ["HIV脑病", "HIV消耗综合征"],
        "comorbidity": "hiv",
        "priority": 8,
    },
    "B23": {
        "display": "人类免疫缺陷病毒病造成的其他情况",
        "aliases": ["HIV相关其他情况"],
        "comorbidity": "hiv",
        "priority": 8,
    },
    "B24": {
        "display": "未特指的人类免疫缺陷病毒病",
        "aliases": ["HIV", "HIV感染（未特指）", "人类免疫缺陷病毒感染"],
        "comorbidity": "hiv",
        "priority": 8,
    },
    "Z21": {
        "display": "无症状人类免疫缺陷病毒感染状态",
        "aliases": ["HIV携带者", "无症状HIV感染", "HIV阳性（无症状）"],
        "comorbidity": "hiv",
        "priority": 7,
    },
    # 矽肺/尘肺
    "J62": {
        "display": "矽肺",
        "aliases": ["矽肺", "硅沉着病", "硅肺", "silicosis"],
        "comorbidity": "copd",
        "priority": 6,
    },
    "J60": {
        "display": "煤工尘肺",
        "aliases": ["煤工尘肺", "煤肺", "炭肺"],
        "comorbidity": "copd",
        "priority": 6,
    },
    "J61": {
        "display": "石棉肺",
        "aliases": ["石棉肺", "asbestosis"],
        "comorbidity": "copd",
        "priority": 6,
    },
    "J63": {
        "display": "其他无机粉尘所致尘肺",
        "aliases": ["其他尘肺", "焊接工尘肺", "石墨尘肺"],
        "comorbidity": "copd",
        "priority": 6,
    },
    "J65": {
        "display": "与结核有关的尘肺",
        "aliases": ["结核合并尘肺", "矽肺结核", "尘肺结核"],
        "comorbidity": "copd",
        "priority": 7,
    },
}

# ============================================================================
# 常见症状/体征 ICD-10 编码（用于症状特征映射）
# ============================================================================

SYMPTOM_ICD10: Dict[str, Dict[str, Any]] = {
    "R05": {"display": "咳嗽", "aliases": ["咳嗽", "干咳", "慢性咳嗽", "cough"]},
    "R06.0": {"display": "呼吸困难", "aliases": ["呼吸困难", "气促", "气喘", "气短"]},
    "R04.2": {"display": "咯血", "aliases": ["咯血", "咳血", "痰中带血", "hemoptysis"]},
    "R50": {"display": "发热", "aliases": ["发热", "发烧", "高热", "低热", "午后低热"]},
    "R61": {"display": "盗汗", "aliases": ["盗汗", "夜间出汗", "night sweat"]},
    "R63.0": {"display": "食欲减退", "aliases": ["食欲减退", "纳差", "食欲不振", "厌食"]},
    "R63.4": {"display": "体重下降", "aliases": ["体重下降", "消瘦", "体重减轻", "wasting"]},
    "R53": {"display": "乏力", "aliases": ["乏力", "疲倦", "疲劳", "虚弱"]},
    "R07.4": {"display": "胸痛", "aliases": ["胸痛", "胸疼", "chest pain"]},
    "R09.1": {"display": "胸膜炎", "aliases": ["胸膜炎", "pleurisy"]},
}

# ============================================================================
# 中文诊断文本 → ICD-10 匹配规则
# ============================================================================

# 中文诊断同义词词典（覆盖 ICD-10 标准名称之外的常见表达）
DIAGNOSIS_SYNONYMS: Dict[str, List[str]] = {
    # 结核病
    "肺结核": ["肺TB", "TB肺", "肺结核病", "肺痨", "痨病", "肺部结核"],
    "活动性肺结核": ["活动期肺结核", "进展期肺结核", "浸润型肺结核"],
    "陈旧性肺结核": ["陈旧结核", "稳定期肺结核", "硬结钙化期", "纤维化肺结核"],
    "结核性胸膜炎": ["胸膜结核", "TB胸膜炎"],
    "淋巴结核": ["淋巴结结核", "颈部淋巴结核", "瘰疬", "老鼠疮"],
    "骨结核": ["脊柱结核", "Pott病", "结核性脊柱炎", "脊椎结核"],
    "肾结核": ["肾脏结核", "泌尿系结核"],
    "结核性脑膜炎": ["结脑", "TB脑膜炎", "脑膜结核"],
    "血行播散型肺结核": ["粟粒性肺结核", "粟粒性结核", "急性粟粒性TB", "播散型肺结核"],
    # 糖尿病
    "糖尿病": ["DM", "糖尿病症", "血糖升高", "糖代谢异常"],
    "2型糖尿病": ["T2DM", "II型糖尿病", "非胰岛素依赖型糖尿病", "成人发病型糖尿病",
                  "糖尿病2型", "2型糖尿", "消渴症", "消渴病"],
    "1型糖尿病": ["T1DM", "I型糖尿病", "胰岛素依赖型糖尿病", "青少年糖尿病"],
    # HIV
    "HIV感染": ["艾滋病", "AIDS", "获得性免疫缺陷综合征", "人类免疫缺陷病毒感染"],
    "HIV携带者": ["无症状HIV感染", "HIV阳性", "艾滋病病毒携带者"],
    # 矽肺
    "矽肺": ["硅沉着病", "硅肺", "silicosis", "砂肺"],
    "尘肺": ["煤工尘肺", "炭肺", "矿工尘肺"],
    # 免疫抑制
    "免疫抑制": ["免疫缺陷", "免疫低下", "免疫力低下", "免疫功能低下"],
    "长期使用激素": ["糖皮质激素治疗", "皮质激素治疗", "激素治疗中", "steroid therapy"],
    "器官移植术后": ["肾移植术后", "肝移植术后", "移植状态", "移植后"],
    "化疗后": ["化学治疗", "化疗中", "抗癌治疗", "肿瘤化疗"],
    "慢性肾病": ["CKD", "慢性肾功能不全", "肾衰竭", "尿毒症", "透析中"],
    "慢性肝病": ["肝硬化", "慢性乙肝", "慢性丙肝", "肝炎肝硬化", "肝纤维化"],
    "营养不良": ["低蛋白血症", "贫血", "消耗状态", "恶病质", "消瘦"],
    "慢阻肺": ["COPD", "慢性阻塞性肺疾病", "肺气肿", "慢性支气管炎"],
}

# 构建反向索引：所有别名 → 标准名称
_ALIAS_TO_STANDARD: Dict[str, str] = {}
for standard, aliases in DIAGNOSIS_SYNONYMS.items():
    for alias in aliases:
        _ALIAS_TO_STANDARD[alias] = standard

# ============================================================================
# 主类
# ============================================================================


class ICD10Mapper:
    """ICD-10 疾病编码映射引擎

    支持：
    - 中文诊断文本 → ICD-10 编码自动匹配
    - 编码前缀匹配（如 "A15" 匹配所有 A15.x）
    - 编码范围判断（如 A15-A19 结核病范围）
    - 合并症编码识别
    - 结核病专病编码优先匹配
    """

    # 合并所有编码库
    _ALL_CODES: Dict[str, Dict[str, Any]] = {}
    _ALL_CODES.update(TB_ICD10_CODES)
    _ALL_CODES.update(COMORBIDITY_ICD10)
    _ALL_CODES.update(SYMPTOM_ICD10)

    # 构建编码 → 显示名称映射
    _CODE_TO_DISPLAY: Dict[str, str] = {
        code: info["display"] for code, info in _ALL_CODES.items()
    }

    # 构建别名 → 编码映射
    _ALIAS_TO_CODE: Dict[str, str] = {}
    for code, info in _ALL_CODES.items():
        aliases = info.get("aliases", [])
        for alias in aliases:
            key = alias.lower().replace(" ", "").replace("　", "")
            # 仅当别名未映射或优先级更高时覆盖
            if key not in _ALIAS_TO_CODE:
                _ALIAS_TO_CODE[key] = code

    # 结核病四位数编码前缀（如 A15.0 → A15）
    _TB_PARENT_CODES: Dict[str, str] = {}
    for code in TB_ICD10_CODES:
        if "." in code:
            parent = code.split(".")[0]
            _TB_PARENT_CODES[code] = parent

    def __init__(self):
        self._last_match: Optional[Dict[str, Any]] = None

    # ==================== 核心匹配方法 ====================

    def match_diagnosis(self, diagnosis_text: str) -> Optional[Dict[str, Any]]:
        """将中文诊断文本匹配到 ICD-10 编码

        匹配优先级：
        1. 精确匹配 → 编码别名
        2. 精确匹配 → 诊断同义词
        3. 包含匹配（诊断文本含编码显示名称）
        4. 关键字匹配（结核/糖尿病/HIV 等关键字）
        5. 模糊匹配（fuzzy string matching）

        参数：
            diagnosis_text: 诊断文本（如 "2型糖尿病"、"肺结核"）

        返回：
            dict | None: {
                "code": "E11",
                "display": "2型糖尿病",
                "confidence": 0.95,
                "category": "comorbidity",
                "comorbidity": "diabetes",
                "match_type": "alias_exact"
            }
        """
        if not diagnosis_text or not isinstance(diagnosis_text, str):
            return None

        text = diagnosis_text.strip()
        if not text:
            return None

        # Step 1: 尝试直接作为 ICD-10 编码查询
        direct = self.lookup_by_code(text.upper())
        if direct:
            return {
                "code": text.upper(),
                "display": direct["display"],
                "confidence": 1.0,
                "category": self._get_category(text.upper()),
                "comorbidity": direct.get("comorbidity"),
                "match_type": "code_direct",
            }

        # Step 2: 别名精确匹配（标准化后）
        normalized = text.lower().replace(" ", "").replace("　", "")
        if normalized in self._ALIAS_TO_CODE:
            code = self._ALIAS_TO_CODE[normalized]
            info = self._ALL_CODES.get(code, {})
            return {
                "code": code,
                "display": info.get("display", ""),
                "confidence": 0.95,
                "category": self._get_category(code),
                "comorbidity": info.get("comorbidity"),
                "match_type": "alias_exact",
            }

        # Step 3: 诊断同义词匹配
        standard_name = _ALIAS_TO_STANDARD.get(normalized)
        if standard_name:
            return self._match_by_standard_name(standard_name, text)

        # Step 4: 包含匹配
        contain_match = self._match_by_contains(text)
        if contain_match:
            return contain_match

        # Step 5: 关键字匹配
        keyword_match = self._match_by_keyword(text)
        if keyword_match:
            return keyword_match

        # Step 6: 字符级模糊匹配（中文拼音/编辑距离）
        fuzzy_match = self._match_by_fuzzy(text)
        if fuzzy_match:
            return fuzzy_match

        return None

    def match_diagnoses_batch(self, diagnoses: List[str]) -> List[Dict[str, Any]]:
        """批量匹配多个诊断

        参数：
            diagnoses: 诊断文本列表

        返回：
            list[dict]: 匹配结果列表
        """
        results = []
        for dx in diagnoses:
            match = self.match_diagnosis(dx)
            if match:
                results.append(match)
        return results

    def is_tb_related(self, code: str) -> bool:
        """判断编码是否为结核病相关

        参数：
            code: ICD-10 编码（如 "A15.0"、"B90"）

        返回：
            bool
        """
        code = code.upper().strip()
        # 检查子编码（如 A15.0 → A15）
        if code in TB_ICD10_CODES:
            return True
        # 检查父编码前缀
        if code in self._TB_PARENT_CODES.values():
            return True
        # 检查编码范围
        for tb_code in TB_ICD10_CODES:
            if code.startswith(tb_code):
                return True
        return False

    def is_comorbidity(self, code: str) -> Optional[str]:
        """判断编码是否为合并症，返回合并症类型

        参数：
            code: ICD-10 编码

        返回：
            str | None: "diabetes" / "hiv" / "copd" / None
        """
        code = code.upper().strip()
        # 直接匹配
        info = COMORBIDITY_ICD10.get(code)
        if info:
            return info.get("comorbidity")

        # 前缀匹配（如 E11.9 → E11）
        if "." in code:
            parent = code.split(".")[0]
            parent_info = COMORBIDITY_ICD10.get(parent)
            if parent_info:
                return parent_info.get("comorbidity")

        # 前缀匹配（如 E10-E14）
        for comorbidity_code, info in COMORBIDITY_ICD10.items():
            if code.startswith(comorbidity_code):
                return info.get("comorbidity")

        return None

    def lookup_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        """通过编码查询显示名称

        参数：
            code: ICD-10 编码

        返回：
            dict | None
        """
        code = code.upper().strip()
        info = self._ALL_CODES.get(code)
        if info:
            return dict(info)
        # 尝试截断到三位数
        if "." in code:
            parent = code.split(".")[0]
            parent_info = self._ALL_CODES.get(parent)
            if parent_info:
                return dict(parent_info)
        return None

    def get_tb_codes(self) -> Dict[str, Dict[str, Any]]:
        """获取所有结核病相关编码

        返回：
            dict: {code: info}
        """
        return dict(TB_ICD10_CODES)

    def get_comorbidity_codes(self, comorbidity_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """获取合并症编码

        参数：
            comorbidity_type: "diabetes" / "hiv" / "copd" / None（全部）

        返回：
            dict
        """
        if comorbidity_type:
            return {
                code: info for code, info in COMORBIDITY_ICD10.items()
                if info.get("comorbidity") == comorbidity_type
            }
        return dict(COMORBIDITY_ICD10)

    def get_symptom_codes(self) -> Dict[str, Dict[str, Any]]:
        """获取症状相关编码"""
        return dict(SYMPTOM_ICD10)

    def extract_tb_info(self, diagnoses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从诊断列表中提取结核病相关信息

        参数：
            diagnoses: 诊断列表，每项含 code/display 字段

        返回：
            dict: {
                "has_active_tb": bool,
                "has_past_tb": bool,
                "tb_codes": [code, ...],
                "comorbidities": {"diabetes": bool, "hiv": bool, "copd": bool},
                "tb_diagnoses": [display, ...]
            }
        """
        result = {
            "has_active_tb": False,
            "has_past_tb": False,
            "tb_codes": [],
            "comorbidities": {"diabetes": False, "hiv": False, "copd": False},
            "tb_diagnoses": [],
        }

        for dx in diagnoses:
            code = str(dx.get("code", "")).upper().strip()
            display = str(dx.get("display", "") or dx.get("description", "") or "")

            # 尝试匹配
            matched = self.match_diagnosis(code) or self.match_diagnosis(display)
            if not matched:
                continue

            matched_code = matched.get("code", code)
            category = matched.get("category", "")
            comorbidity = matched.get("comorbidity")

            if category == "active_tb":
                result["has_active_tb"] = True
                result["tb_codes"].append(matched_code)
                if matched.get("display"):
                    result["tb_diagnoses"].append(matched["display"])
            elif category == "past_tb":
                result["has_past_tb"] = True
                result["tb_codes"].append(matched_code)

            if comorbidity in ("diabetes", "hiv", "copd"):
                result["comorbidities"][comorbidity] = True

            # 编码前缀匹配作为补充
            if not matched:
                for tb_code in TB_ICD10_CODES:
                    if code.startswith(tb_code):
                        result["has_active_tb"] = True
                        result["tb_codes"].append(code)
                        result["tb_diagnoses"].append(
                            TB_ICD10_CODES[tb_code]["display"])
                        break

        return result

    # ==================== 内部方法 ====================

    def _get_category(self, code: str) -> str:
        """获取编码所属分类"""
        if code in TB_ICD10_CODES or any(code.startswith(tc) for tc in TB_ICD10_CODES):
            info = TB_ICD10_CODES.get(code, {})
            return info.get("category", "active_tb")
        if code in COMORBIDITY_ICD10:
            return "comorbidity"
        if code in SYMPTOM_ICD10:
            return "symptom"
        return "unknown"

    def _match_by_standard_name(self, standard_name: str, original_text: str) -> Optional[Dict[str, Any]]:
        """通过标准名称匹配 ICD-10 编码"""
        # 在显示名称中查找
        for code, info in self._ALL_CODES.items():
            if info.get("display") == standard_name:
                comorbidity = info.get("comorbidity")
                return {
                    "code": code,
                    "display": info.get("display", ""),
                    "confidence": 0.92,
                    "category": self._get_category(code),
                    "comorbidity": comorbidity,
                    "match_type": "synonym_standard",
                }
        # 在别名中查找
        for code, info in self._ALL_CODES.items():
            if standard_name in info.get("aliases", []):
                comorbidity = info.get("comorbidity")
                return {
                    "code": code,
                    "display": info.get("display", ""),
                    "confidence": 0.90,
                    "category": self._get_category(code),
                    "comorbidity": comorbidity,
                    "match_type": "synonym_alias",
                }
        return None

    def _match_by_contains(self, text: str) -> Optional[Dict[str, Any]]:
        """包含匹配：诊断文本包含编码显示名称"""
        # 优先匹配结核病（最长匹配优先）
        for code, info in sorted(TB_ICD10_CODES.items(), key=lambda x: -len(x[1]["display"])):
            display = info["display"]
            if display and display in text:
                return {
                    "code": code,
                    "display": display,
                    "confidence": 0.85,
                    "category": info.get("category", "active_tb"),
                    "comorbidity": info.get("comorbidity"),
                    "match_type": "contains_display",
                }

        # 合并症匹配
        for code, info in sorted(COMORBIDITY_ICD10.items(), key=lambda x: -len(x[1]["display"])):
            display = info["display"]
            if display and display in text:
                return {
                    "code": code,
                    "display": display,
                    "confidence": 0.80,
                    "category": "comorbidity",
                    "comorbidity": info.get("comorbidity"),
                    "match_type": "contains_display",
                }

        # 别名包含匹配（按别名长度降序，确保最长、最具体的别名优先命中）
        alias_list = []
        for code, info in self._ALL_CODES.items():
            for alias in info.get("aliases", []):
                if alias:
                    alias_list.append((alias, code, info))
        alias_list.sort(key=lambda x: -len(x[0]))
        for alias, code, info in alias_list:
            if alias in text:
                return {
                    "code": code,
                    "display": info.get("display", ""),
                    "confidence": 0.75,
                    "category": self._get_category(code),
                    "comorbidity": info.get("comorbidity"),
                    "match_type": "contains_alias",
                }
        return None

    def _match_by_keyword(self, text: str) -> Optional[Dict[str, Any]]:
        """关键字匹配"""
        # 结核病关键字
        # 注意："肺结核"必须放在"结核"之前，确保精确匹配优先于泛化匹配
        tb_keywords = {
            "肺结核": ("A15", "肺结核，经细菌学和组织学证实"),
            "结核": ("A16", "肺结核，未经细菌学或组织学证实"),
            "TB": ("A15", "肺结核，经细菌学和组织学证实"),
            "结核病": ("A16", "肺结核，未经细菌学或组织学证实"),
            "肺痨": ("A16", "肺结核，未经细菌学或组织学证实"),
            "痨病": ("A16", "肺结核，未经细菌学或组织学证实"),
            "活动性结核": ("A16", "肺结核，未经细菌学或组织学证实"),
            "菌阳": ("A15", "肺结核，经细菌学和组织学证实"),
            "涂阳": ("A15.0", "肺结核，经显微镜检查证实"),
            "菌阴": ("A16", "肺结核，未经细菌学或组织学证实"),
            "粟粒": ("A19", "粟粒性结核"),
            "Pott": ("A18.0", "骨和关节结核"),
            "卡介苗": ("Z23", "需要接种疫苗"),
        }

        text_lower = text.lower()
        for keyword, (code, display) in tb_keywords.items():
            if keyword in text or keyword.lower() in text_lower:
                # 不同关键字使用不同置信度
                keyword_confidence = {
                    "肺结核": 0.90,
                    "TB": 0.75,
                    "菌阳": 0.85,
                    "涂阳": 0.85,
                    "菌阴": 0.80,
                    "粟粒": 0.85,
                }
                conf = keyword_confidence.get(keyword, 0.70)
                return {
                    "code": code,
                    "display": display,
                    "confidence": conf,
                    "category": "active_tb" if code.startswith("A") or code.startswith("B90") else "other",
                    "match_type": "keyword",
                }

        # 糖尿病关键字
        dm_keywords = ["糖尿病", "血糖", "消渴", "DM"]
        for kw in dm_keywords:
            if kw in text:
                return {
                    "code": "E14",
                    "display": "未特指糖尿病",
                    "confidence": 0.65,
                    "category": "comorbidity",
                    "comorbidity": "diabetes",
                    "match_type": "keyword",
                }

        # HIV 关键字
        hiv_keywords = ["HIV", "艾滋病", "AIDS", "人类免疫缺陷"]
        for kw in hiv_keywords:
            if kw in text or kw in text_lower:
                return {
                    "code": "B24",
                    "display": "未特指的人类免疫缺陷病毒病",
                    "confidence": 0.65,
                    "category": "comorbidity",
                    "comorbidity": "hiv",
                    "match_type": "keyword",
                }

        return None

    def _match_by_fuzzy(self, text: str) -> Optional[Dict[str, Any]]:
        """模糊匹配（基于字符级编辑距离和拼音）"""
        # 简化的模糊匹配：检查是否有任意别名包含在文本中或文本包含在别名中
        for code, info in sorted(self._ALL_CODES.items(), key=lambda x: -len(x[1].get("display", ""))):
            display = info.get("display", "")
            # 子串匹配（文本是显示名称的子串，或显示名称是文本的子串）
            if display and (display in text or text in display):
                return {
                    "code": code,
                    "display": display,
                    "confidence": 0.60,
                    "category": self._get_category(code),
                    "comorbidity": info.get("comorbidity"),
                    "match_type": "fuzzy_substring",
                }
            # 别名子串匹配
            for alias in info.get("aliases", []):
                if alias and (alias in text or text in alias):
                    return {
                        "code": code,
                        "display": display,
                        "confidence": 0.55,
                        "category": self._get_category(code),
                        "comorbidity": info.get("comorbidity"),
                        "match_type": "fuzzy_alias_substring",
                    }

        # 单字匹配（结核相关：任何含"核"字的诊断）
        if "核" in text:
            return {
                "code": "A16",
                "display": "肺结核，未经细菌学或组织学证实",
                "confidence": 0.40,
                "category": "active_tb",
                "match_type": "fuzzy_single_char",
            }

        return None