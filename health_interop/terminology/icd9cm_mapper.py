#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ICD-9-CM-3 手术操作编码映射引擎

ICD-9-CM-3 是手术操作分类标准，用于：
1. 结核病相关手术操作编码映射
2. 诊断性操作（支气管镜、胸腔穿刺等）编码映射
3. 治疗性操作（肺叶切除、胸廓成形等）编码映射

注意：ICD-9-CM-3 仍是国内手术操作编码的主要标准，
ICD-10-PCS 尚未全面推广。
"""

import logging
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("tb_risk.health_interop.terminology.icd9cm")

# ICD-9-CM-3 结核病相关手术操作编码
ICD9CM_TB_CODES: Dict[str, Dict[str, Any]] = {
    # 诊断性操作
    "33.22": {"display": "纤维支气管镜检查", "type": "diagnostic",
              "aliases": ["支气管镜", "纤支镜", "气管镜", "bronchoscopy"]},
    "33.23": {"display": "支气管镜活检", "type": "diagnostic",
              "aliases": ["支气管镜活检", "纤支镜活检", "气管镜活检", "bronchoscopic biopsy"]},
    "33.24": {"display": "支气管镜刷检", "type": "diagnostic",
              "aliases": ["支气管镜刷检", "纤支镜刷检", "刷检"]},
    "33.25": {"display": "支气管肺泡灌洗", "type": "diagnostic",
              "aliases": ["支气管肺泡灌洗", "BAL", "灌洗", "bronchoalveolar lavage"]},
    "33.26": {"display": "经支气管针吸活检", "type": "diagnostic",
              "aliases": ["TBNA", "经支气管针吸", "纵隔穿刺"]},
    "33.27": {"display": "支气管镜超声引导活检", "type": "diagnostic",
              "aliases": ["EBUS-TBNA", "EBUS", "超声支气管镜"]},
    "34.91": {"display": "胸腔穿刺", "type": "diagnostic",
              "aliases": ["胸腔穿刺", "胸穿", "thoracentesis", "胸腔抽液"]},
    "34.92": {"display": "胸腔闭式引流", "type": "therapeutic",
              "aliases": ["胸腔闭式引流", "胸管引流", "chest tube"]},
    "34.24": {"display": "胸膜活检", "type": "diagnostic",
              "aliases": ["胸膜活检", "pleural biopsy"]},
    "34.25": {"display": "经皮肺穿刺活检", "type": "diagnostic",
              "aliases": ["肺穿刺", "经皮肺穿刺", "lung biopsy", "肺活检"]},
    # 治疗性操作
    "32.4": {"display": "肺叶切除术", "type": "therapeutic",
             "aliases": ["肺叶切除", "lobectomy", "肺叶切除术"]},
    "32.5": {"display": "全肺切除术", "type": "therapeutic",
             "aliases": ["全肺切除", "pneumonectomy", "全肺切除术"]},
    "32.6": {"display": "肺段切除术", "type": "therapeutic",
             "aliases": ["肺段切除", "segmentectomy", "肺段切除术"]},
    "32.9": {"display": "肺其他切除术", "type": "therapeutic",
             "aliases": ["肺楔形切除", "楔形切除", "wedge resection"]},
    "34.3": {"display": "胸膜剥脱术", "type": "therapeutic",
             "aliases": ["胸膜剥脱", "胸膜剥脱术", "decortication", "胸膜纤维板剥脱"]},
    "34.6": {"display": "胸廓成形术", "type": "therapeutic",
             "aliases": ["胸廓成形", "胸廓成形术", "thoracoplasty"]},
    "34.74": {"display": "胸膜固定术", "type": "therapeutic",
              "aliases": ["胸膜固定", "胸膜固定术", "pleurodesis"]},
    "34.99": {"display": "其他胸膜操作", "type": "therapeutic",
              "aliases": ["胸膜腔冲洗", "胸腔冲洗"]},
    "32.1": {"display": "支气管袖状切除术", "type": "therapeutic",
             "aliases": ["袖状切除", "支气管袖状", "sleeve resection"]},
    "40.11": {"display": "淋巴结活检", "type": "diagnostic",
              "aliases": ["淋巴结活检", "淋巴活检", "lymph node biopsy"]},
    "40.50": {"display": "淋巴结清扫术", "type": "therapeutic",
              "aliases": ["淋巴结清扫", "淋巴结清除", "lymph node dissection"]},
    "99.15": {"display": "卡介苗接种", "type": "preventive",
              "aliases": ["BCG接种", "卡介苗接种", "BCG vaccination"]},
}


class ICD9CMMapper:
    """ICD-9-CM-3 手术操作编码映射引擎"""

    def __init__(self):
        self._alias_map: Dict[str, str] = {}
        for code, info in ICD9CM_TB_CODES.items():
            self._alias_map[info["display"].lower().replace(" ", "")] = code
            for alias in info.get("aliases", []):
                self._alias_map[alias.lower().replace(" ", "")] = code

    def match(self, text: str) -> Optional[Dict[str, Any]]:
        """匹配手术操作编码

        参数：
            text: 操作名称（如"支气管镜活检"、"肺叶切除术"）

        返回：
            dict | None: {code, display, type, aliases}
        """
        if not text:
            return None
        text = text.strip()

        # 直接编码查询
        info = ICD9CM_TB_CODES.get(text)
        if info:
            result = dict(info)
            result["code"] = text
            result["confidence"] = 1.0
            return result

        # 别名匹配
        normalized = text.lower().replace(" ", "")
        code = self._alias_map.get(normalized)
        if code:
            info = ICD9CM_TB_CODES.get(code)
            if info:
                result = dict(info)
                result["code"] = code
                result["confidence"] = 0.95
                return result

        # 包含匹配
        for code, info in ICD9CM_TB_CODES.items():
            display = info["display"].lower()
            if display in text.lower() or text.lower() in display:
                result = dict(info)
                result["code"] = code
                result["confidence"] = 0.80
                return result
            for alias in info.get("aliases", []):
                if alias.lower() in text.lower() or text.lower() in alias.lower():
                    result = dict(info)
                    result["code"] = code
                    result["confidence"] = 0.75
                    return result

        return None

    def lookup_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        info = ICD9CM_TB_CODES.get(code.strip())
        if info:
            result = dict(info)
            result["code"] = code.strip()
            return result
        return None

    def get_diagnostic_procedures(self) -> List[Dict[str, Any]]:
        return [dict(info) for info in ICD9CM_TB_CODES.values() if info.get("type") == "diagnostic"]

    def get_therapeutic_procedures(self) -> List[Dict[str, Any]]:
        return [dict(info) for info in ICD9CM_TB_CODES.values() if info.get("type") == "therapeutic"]