#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LOINC 检验项目编码映射引擎

核心功能：
1. 检验项目名称（中文/英文）→ LOINC 编码自动匹配
2. LOINC 编码反向查询
3. 结核病相关检验专项查询（痰涂片、Xpert、T-SPOT、培养等）
4. 检验结果单位转换与参考范围标准化
5. 检验项目分类（病原学/分子/血清学/常规/传染病筛查）

数据来源：
    LOINC® 数据库（Regenstrief Institute）
    内嵌 TB 相关核心 LOINC 编码及常见检验项目映射
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("tb_risk.health_interop.terminology.loinc")

# ============================================================================
# 结核病相关 LOINC 编码（五大类）
# ============================================================================

LOINC_TB_PANELS: Dict[str, List[Dict[str, Any]]] = {
    "pathogen": [
        {"loinc": "640-4", "name": "痰涂片抗酸染色", "short": "涂片", "sample": "sputum",
         "aliases": ["AFB smear", "抗酸杆菌涂片", "痰涂片找抗酸杆菌", "痰抗酸染色"]},
        {"loinc": "11477-7", "name": "痰结核分枝杆菌培养", "short": "培养", "sample": "sputum",
         "aliases": ["TB culture", "结核培养", "痰培养找结核菌", "分枝杆菌培养"]},
        {"loinc": "14473-1", "name": "BACTEC液体培养", "short": "BACTEC", "sample": "sputum",
         "aliases": ["BACTEC MGIT", "液体培养", "快速培养"]},
        {"loinc": "43405-4", "name": "抗酸杆菌荧光染色", "short": "荧光涂片", "sample": "sputum",
         "aliases": ["荧光抗酸", "金胺O染色", "auramine"]},
    ],
    "molecular": [
        {"loinc": "94504-3", "name": "Xpert MTB/RIF 结核分枝杆菌", "short": "Xpert_MTB", "sample": "sputum",
         "aliases": ["GeneXpert MTB", "Xpert MTB", "结核核酸快速检测"]},
        {"loinc": "94505-0", "name": "Xpert MTB/RIF 利福平耐药", "short": "Xpert_RIF", "sample": "sputum",
         "aliases": ["Xpert RIF", "利福平耐药检测", "RIF resistance"]},
        {"loinc": "94419-4", "name": "熔解曲线法结核耐药", "short": "熔解曲线", "sample": "sputum",
         "aliases": ["熔解曲线耐药", "melting curve", "结核耐药基因检测"]},
        {"loinc": "82349-7", "name": "结核分枝杆菌测序", "short": "测序", "sample": "sputum",
         "aliases": ["TB sequencing", "NGS结核", "结核二代测序"]},
    ],
    "serology": [
        {"loinc": "80403-4", "name": "结核感染T细胞检测(T-SPOT)", "short": "T-SPOT", "sample": "blood",
         "aliases": ["T-SPOT.TB", "TSPOT", "结核T细胞斑点", "ESAT-6/CFP-10"]},
        {"loinc": "80404-2", "name": "γ干扰素释放试验(IGRA)", "short": "IGRA", "sample": "blood",
         "aliases": ["IGRA", "QFT-GIT", "QuantiFERON", "γ干扰素"]},
        {"loinc": "22442-7", "name": "结核抗体", "short": "TB-Ab", "sample": "blood",
         "aliases": ["TB antibody", "结核抗体检测", "抗结核抗体"]},
        {"loinc": "40563-3", "name": "结核菌素皮试(PPD)", "short": "PPD", "sample": "skin",
         "aliases": ["PPD试验", "结核菌素试验", "OT试验", "Mantoux试验"]},
    ],
    "routine": [
        {"loinc": "4537-7", "name": "血沉(ESR)", "short": "ESR", "sample": "blood", "unit": "mm/h",
         "aliases": ["血沉", "ESR", "erythrocyte sedimentation rate"]},
        {"loinc": "1988-5", "name": "C反应蛋白(CRP)", "short": "CRP", "sample": "blood", "unit": "mg/L",
         "aliases": ["CRP", "超敏CRP", "hs-CRP", "C反应蛋白"]},
        {"loinc": "2345-7", "name": "空腹血糖", "short": "GLU", "sample": "blood", "unit": "mmol/L",
         "aliases": ["空腹血糖", "FBG", "fasting glucose"]},
        {"loinc": "2339-0", "name": "随机血糖", "short": "GLU_random", "sample": "blood", "unit": "mmol/L",
         "aliases": ["随机血糖", "血糖", "GLU", "glucose"]},
        {"loinc": "6690-2", "name": "白细胞计数(WBC)", "short": "WBC", "sample": "blood", "unit": "10^9/L",
         "aliases": ["白细胞", "WBC", "white blood cell"]},
        {"loinc": "718-7", "name": "血红蛋白(HGB)", "short": "HGB", "sample": "blood", "unit": "g/L",
         "aliases": ["血红蛋白", "HGB", "Hb", "hemoglobin"]},
        {"loinc": "2823-3", "name": "谷丙转氨酶(ALT)", "short": "ALT", "sample": "blood", "unit": "U/L",
         "aliases": ["ALT", "谷丙转氨酶", "GPT"]},
        {"loinc": "1920-8", "name": "谷草转氨酶(AST)", "short": "AST", "sample": "blood", "unit": "U/L",
         "aliases": ["AST", "谷草转氨酶", "GOT"]},
        {"loinc": "2160-0", "name": "肌酐(Cr)", "short": "Cr", "sample": "blood", "unit": "umol/L",
         "aliases": ["肌酐", "Cr", "creatinine"]},
        {"loinc": "2951-2", "name": "血尿素氮(BUN)", "short": "BUN", "sample": "blood", "unit": "mmol/L",
         "aliases": ["尿素氮", "BUN", "urea"]},
        {"loinc": "4544-3", "name": "血细胞比容(HCT)", "short": "HCT", "sample": "blood", "unit": "%",
         "aliases": ["红细胞压积", "HCT", "hematocrit"]},
    ],
    "infectious_screen": [
        {"loinc": "21470-1", "name": "HIV1/2抗体", "short": "HIV", "sample": "blood",
         "aliases": ["HIV抗体", "抗HIV", "AIDS抗体"]},
        {"loinc": "16935-9", "name": "乙肝表面抗原(HBsAg)", "short": "HBsAg", "sample": "blood",
         "aliases": ["乙肝表面抗原", "HBsAg", "乙肝两对半"]},
        {"loinc": "22310-9", "name": "丙肝抗体(HCV-Ab)", "short": "HCV", "sample": "blood",
         "aliases": ["丙肝抗体", "抗HCV", "HCV-Ab"]},
        {"loinc": "31047-5", "name": "梅毒抗体(TP-Ab)", "short": "Syphilis", "sample": "blood",
         "aliases": ["梅毒抗体", "TP-Ab", "RPR", "TPPA"]},
    ],
}

# 扁平化所有检验项目
_ALL_LOINC_ITEMS: List[Dict[str, Any]] = []
for panel, items in LOINC_TB_PANELS.items():
    for item in items:
        item["panel"] = panel
        _ALL_LOINC_ITEMS.append(item)

# 构建索引
_LOINC_BY_CODE: Dict[str, Dict[str, Any]] = {item["loinc"]: item for item in _ALL_LOINC_ITEMS}
_LOINC_BY_ALIAS: Dict[str, List[Dict[str, Any]]] = {}
for item in _ALL_LOINC_ITEMS:
    # 按名称
    key = item["name"].lower().replace(" ", "").replace("(", "").replace(")", "")
    _LOINC_BY_ALIAS.setdefault(key, []).append(item)
    # 按短名
    key_short = item["short"].lower().replace(" ", "")
    _LOINC_BY_ALIAS.setdefault(key_short, []).append(item)
    # 按别名
    for alias in item.get("aliases", []):
        key_alias = alias.lower().replace(" ", "").replace("(", "").replace(")", "")
        _LOINC_BY_ALIAS.setdefault(key_alias, []).append(item)

# ============================================================================
# 检验结果参考范围（按 LOINC 编码、性别、年龄分类）
# ============================================================================

LOINC_REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    "4537-7": {  # ESR
        "unit": "mm/h",
        "ranges": {
            "adult_male": {"low": 0, "high": 15},
            "adult_female": {"low": 0, "high": 20},
            "elderly": {"low": 0, "high": 30},
        },
        "critical": {"low": None, "high": 100},
    },
    "1988-5": {  # CRP
        "unit": "mg/L",
        "ranges": {
            "default": {"low": 0, "high": 5},
            "child": {"low": 0, "high": 10},
        },
        "critical": {"low": None, "high": 200},
    },
    "2345-7": {  # 空腹血糖
        "unit": "mmol/L",
        "ranges": {
            "default": {"low": 3.9, "high": 6.1},
            "diabetic": {"low": 3.9, "high": 7.0},
        },
        "critical": {"low": 2.8, "high": 25},
    },
    "2339-0": {  # 随机血糖
        "unit": "mmol/L",
        "ranges": {
            "default": {"low": 3.9, "high": 7.8},
        },
        "critical": {"low": 2.8, "high": 25},
    },
    "6690-2": {  # WBC
        "unit": "10^9/L",
        "ranges": {
            "default": {"low": 3.5, "high": 9.5},
            "child": {"low": 4.0, "high": 12.0},
        },
        "critical": {"low": 1.0, "high": 30},
    },
    "718-7": {  # HGB
        "unit": "g/L",
        "ranges": {
            "adult_male": {"low": 130, "high": 175},
            "adult_female": {"low": 115, "high": 150},
            "child": {"low": 110, "high": 160},
        },
        "critical": {"low": 60, "high": 200},
    },
    "2823-3": {  # ALT
        "unit": "U/L",
        "ranges": {
            "default": {"low": 0, "high": 40},
        },
        "critical": {"low": None, "high": 1000},
    },
    "1920-8": {  # AST
        "unit": "U/L",
        "ranges": {
            "default": {"low": 0, "high": 40},
        },
        "critical": {"low": None, "high": 1000},
    },
    "2160-0": {  # Cr
        "unit": "umol/L",
        "ranges": {
            "adult_male": {"low": 44, "high": 104},
            "adult_female": {"low": 44, "high": 84},
        },
        "critical": {"low": None, "high": 500},
    },
}


class LOINCMapper:
    """LOINC 检验项目编码映射引擎"""

    def __init__(self):
        self._last_match: Optional[Dict[str, Any]] = None

    # ==================== 查询方法 ====================

    def lookup_by_code(self, loinc_code: str) -> Optional[Dict[str, Any]]:
        """通过 LOINC 编码查询检验项目信息

        参数：
            loinc_code: LOINC 编码（如 "640-4"）

        返回：
            dict | None: {loinc, name, short, sample, panel, aliases}
        """
        loinc_code = loinc_code.strip()
        item = _LOINC_BY_CODE.get(loinc_code)
        if item:
            return dict(item)
        return None

    def match_test_name(self, test_name: str) -> Optional[Dict[str, Any]]:
        """将检验项目名称匹配到 LOINC 编码

        参数：
            test_name: 检验项目名称（中/英文，如 "痰涂片找抗酸杆菌"）

        返回：
            dict | None
        """
        if not test_name or not isinstance(test_name, str):
            return None

        name = test_name.strip()
        if not name:
            return None

        # Step 1: 直接 LOINC 编码查询
        if re.match(r"^\d{2,5}-\d$", name):
            direct = self.lookup_by_code(name)
            if direct:
                direct["confidence"] = 1.0
                direct["match_type"] = "code_direct"
                return direct

        # Step 2: 别名精确匹配
        normalized = name.lower().replace(" ", "").replace("(", "").replace(")", "")
        items = _LOINC_BY_ALIAS.get(normalized)
        if items:
            result = dict(items[0])
            result["confidence"] = 0.95
            result["match_type"] = "alias_exact"
            return result

        # Step 3: 包含匹配
        for item in _ALL_LOINC_ITEMS:
            # 检查名称
            if item["name"].lower() in name.lower() or name.lower() in item["name"].lower():
                result = dict(item)
                result["confidence"] = 0.80
                result["match_type"] = "contains_name"
                return result
            # 检查别名
            for alias in item.get("aliases", []):
                if alias.lower() in name.lower() or name.lower() in alias.lower():
                    result = dict(item)
                    result["confidence"] = 0.75
                    result["match_type"] = "contains_alias"
                    return result

        # Step 4: 关键字匹配
        keyword_match = self._match_by_keyword(name)
        if keyword_match:
            return keyword_match

        return None

    def match_test_batch(self, test_names: List[str]) -> List[Dict[str, Any]]:
        """批量匹配检验项目"""
        results = []
        for name in test_names:
            match = self.match_test_name(name)
            if match:
                results.append(match)
        return results

    # ==================== 分类查询 ====================

    def get_tb_panel(self, panel_name: str) -> List[Dict[str, Any]]:
        """获取结核病检验面板

        参数：
            panel_name: "pathogen" / "molecular" / "serology" / "routine" / "infectious_screen"

        返回：
            list[dict]
        """
        items = LOINC_TB_PANELS.get(panel_name, [])
        return [dict(item) for item in items]

    def get_all_tb_tests(self) -> List[Dict[str, Any]]:
        """获取所有结核病相关检验项目"""
        return [dict(item) for item in _ALL_LOINC_ITEMS]

    def get_tb_pathogen_tests(self) -> List[Dict[str, Any]]:
        """获取病原学检验项目（涂片、培养）"""
        return self.get_tb_panel("pathogen")

    def get_molecular_tests(self) -> List[Dict[str, Any]]:
        """获取分子生物学检验项目（Xpert、测序）"""
        return self.get_tb_panel("molecular")

    def get_serology_tests(self) -> List[Dict[str, Any]]:
        """获取血清学检验项目（T-SPOT、IGRA、PPD）"""
        return self.get_tb_panel("serology")

    def is_tb_test(self, loinc_code: str) -> bool:
        """判断是否为结核病相关检验"""
        return loinc_code.strip() in _LOINC_BY_CODE

    # ==================== 参考范围标准化 ====================

    def get_reference_range(self, loinc_code: str, gender: str = "default",
                            age: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """获取检验项目的参考范围

        参数：
            loinc_code: LOINC 编码
            gender: "male" / "female" / "default"
            age: 年龄（岁）

        返回：
            dict | None: {"low": float, "high": float, "unit": str, "critical_low": float, "critical_high": float}
        """
        ranges = LOINC_REFERENCE_RANGES.get(loinc_code.strip())
        if not ranges:
            return None

        unit = ranges.get("unit", "")
        critical = ranges.get("critical", {})

        # 按性别和年龄选择最佳匹配
        range_key = "default"
        if gender == "male":
            if age and age >= 65:
                range_key = "elderly"
            else:
                range_key = "adult_male"
        elif gender == "female":
            if age and age >= 65:
                range_key = "elderly"
            else:
                range_key = "adult_female"
        else:
            range_key = "default"

        # 尝试子键，回退到 default
        sub_ranges = ranges.get("ranges", {})
        selected = sub_ranges.get(range_key, sub_ranges.get("default"))

        if not selected:
            return None

        return {
            "low": selected["low"],
            "high": selected["high"],
            "unit": unit,
            "critical_low": critical.get("low"),
            "critical_high": critical.get("high"),
        }

    def is_abnormal(self, loinc_code: str, value: float, gender: str = "default",
                    age: Optional[int] = None) -> Optional[bool]:
        """判断检验结果是否异常

        参数：
            loinc_code: LOINC 编码
            value: 检验值
            gender: "male" / "female" / "default"
            age: 年龄

        返回：
            bool | None（参考范围不可用时返回 None）
        """
        range_info = self.get_reference_range(loinc_code, gender, age)
        if not range_info:
            return None
        return value < range_info["low"] or value > range_info["high"]

    def normalize_result(self, loinc_code: str, value: float, unit: str = "") -> Dict[str, Any]:
        """标准化检验结果（单位转换 + 异常判断）

        参数：
            loinc_code: LOINC 编码
            value: 原始值
            unit: 原始单位

        返回：
            dict: {value, unit, is_abnormal, normalized_value, normalized_unit}
        """
        result = {"original_value": value, "original_unit": unit}
        item = _LOINC_BY_CODE.get(loinc_code.strip())
        if item:
            target_unit = item.get("unit", unit)
            # 单位转换
            if unit and target_unit and unit.lower() != target_unit.lower():
                converted = self._convert_unit(value, unit, target_unit)
                if converted is not None:
                    result["normalized_value"] = converted
                    result["normalized_unit"] = target_unit
                else:
                    result["normalized_value"] = value
                    result["normalized_unit"] = unit
            else:
                result["normalized_value"] = value
                result["normalized_unit"] = unit or target_unit
        else:
            result["normalized_value"] = value
            result["normalized_unit"] = unit

        return result

    # ==================== 内部方法 ====================

    def _match_by_keyword(self, name: str) -> Optional[Dict[str, Any]]:
        """关键字匹配"""
        name_lower = name.lower()
        keywords = {
            "涂片": "640-4", "抗酸": "640-4", "afb": "640-4",
            "培养": "11477-7", "分枝杆菌": "11477-7",
            "xpert": "94504-3", "genexpert": "94504-3", "mtb": "94504-3",
            "利福平": "94505-0", "rif": "94505-0",
            "tspot": "80403-4", "t-spot": "80403-4", "t细胞": "80403-4",
            "igra": "80404-2", "quantiferon": "80404-2", "γ干扰素": "80404-2",
            "ppd": "40563-3", "结核菌素": "40563-3",
            "血沉": "4537-7", "esr": "4537-7",
            "crp": "1988-5", "c反应": "1988-5",
            "空腹血糖": "2345-7", "fbg": "2345-7",
            "hiv": "21470-1", "艾滋病": "21470-1",
            "hbsag": "16935-9", "乙肝": "16935-9",
        }

        for kw, loinc in keywords.items():
            if kw in name_lower:
                item = _LOINC_BY_CODE.get(loinc)
                if item:
                    result = dict(item)
                    result["confidence"] = 0.65
                    result["match_type"] = "keyword"
                    return result
        return None

    @staticmethod
    def _convert_unit(value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """单位转换"""
        conversions = {
            ("mg/dl", "mmol/l"): lambda v: v / 18.016,
            ("mmol/l", "mg/dl"): lambda v: v * 18.016,
            ("mg/l", "mg/dl"): lambda v: v / 10.0,
            ("mg/dl", "mg/l"): lambda v: v * 10.0,
            ("ug/ml", "mg/l"): lambda v: v,
            ("mg/l", "ug/ml"): lambda v: v,
        }
        key = (from_unit.lower().strip(), to_unit.lower().strip())
        fn = conversions.get(key)
        if fn:
            try:
                return round(fn(float(value)), 4)
            except (TypeError, ValueError):
                return None
        return None