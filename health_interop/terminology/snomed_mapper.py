#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SNOMED CT 临床术语编码映射引擎

SNOMED CT 是国际通用的临床术语标准，提供：
1. 症状/体征 → SNOMED CT 编码映射
2. 临床发现 → 编码映射
3. 操作/程序 → 编码映射
4. ICD-10 ↔ SNOMED CT 交叉映射

本模块内嵌结核病相关核心 SNOMED CT 编码。
"""

import logging
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("tb_risk.health_interop.terminology.snomed")

# SNOMED CT 结核病相关编码
SNOMED_TB_CODES: Dict[str, Dict[str, Any]] = {
    "154283005": {"display": "肺结核 (disorder)", "aliases": ["肺结核", "肺TB", "pulmonary tuberculosis"],
                  "icd10": "A15-A16"},
    "154342006": {"display": "活动性肺结核 (disorder)", "aliases": ["活动性肺结核", "active pulmonary tuberculosis"],
                  "icd10": "A15-A16"},
    "161539005": {"display": "陈旧性肺结核 (disorder)", "aliases": ["陈旧性肺结核", "陈旧肺结核"],
                  "icd10": "B90.9"},
    "186216006": {"display": "结核性脑膜炎 (disorder)", "aliases": ["结核性脑膜炎", "结脑"],
                  "icd10": "A17.0"},
    "186220004": {"display": "骨结核 (disorder)", "aliases": ["骨结核", "脊柱结核"],
                  "icd10": "A18.0"},
    "186222007": {"display": "肾结核 (disorder)", "aliases": ["肾结核", "泌尿系结核"],
                  "icd10": "A18.1"},
    "186224008": {"display": "淋巴结核 (disorder)", "aliases": ["淋巴结核", "淋巴结结核"],
                  "icd10": "A18.2"},
    "186226005": {"display": "肠结核 (disorder)", "aliases": ["肠结核", "结核性腹膜炎"],
                  "icd10": "A18.3"},
    "186228006": {"display": "粟粒性结核 (disorder)", "aliases": ["粟粒性结核", "粟粒性肺结核"],
                  "icd10": "A19"},
    "233604007": {"display": "肺炎结核 (disorder)", "aliases": ["肺炎结核", "结核性肺炎"]},
    "266469007": {"display": "咳嗽 (finding)", "aliases": ["咳嗽", "cough"], "icd10": "R05"},
    "267036007": {"display": "呼吸困难 (finding)", "aliases": ["呼吸困难", "dyspnea"], "icd10": "R06.0"},
    "274640006": {"display": "咯血 (finding)", "aliases": ["咯血", "hemoptysis"], "icd10": "R04.2"},
    "386661006": {"display": "发热 (finding)", "aliases": ["发热", "fever", "发烧"], "icd10": "R50"},
    "429840002": {"display": "盗汗 (finding)", "aliases": ["盗汗", "night sweat"], "icd10": "R61"},
    "79890006": {"display": "体重下降 (finding)", "aliases": ["体重下降", "weight loss", "消瘦"], "icd10": "R63.4"},
    "42255003": {"display": "乏力 (finding)", "aliases": ["乏力", "fatigue", "疲倦"], "icd10": "R53"},
    "298570002": {"display": "胸痛 (finding)", "aliases": ["胸痛", "chest pain"], "icd10": "R07.4"},
    "44054006": {"display": "2型糖尿病 (disorder)", "aliases": ["2型糖尿病", "type 2 diabetes", "T2DM"],
                "icd10": "E11"},
    "46635009": {"display": "1型糖尿病 (disorder)", "aliases": ["1型糖尿病", "type 1 diabetes", "T1DM"],
                "icd10": "E10"},
    "86406008": {"display": "HIV感染 (disorder)", "aliases": ["HIV感染", "HIV", "人类免疫缺陷病毒感染"],
                "icd10": "B20-B24"},
}


class SNOMEDMapper:
    """SNOMED CT 临床术语编码映射引擎"""

    def __init__(self):
        self._alias_map: Dict[str, str] = {}
        for code, info in SNOMED_TB_CODES.items():
            self._alias_map[info["display"].lower()] = code
            for alias in info.get("aliases", []):
                self._alias_map[alias.lower().replace(" ", "")] = code

    def match(self, text: str) -> Optional[Dict[str, Any]]:
        """匹配 SNOMED CT 编码

        参数：
            text: 临床术语文本

        返回：
            dict | None: {code, display, aliases, icd10}
        """
        if not text:
            return None
        text = text.strip()
        # 直接编码查询
        info = SNOMED_TB_CODES.get(text)
        if info:
            result = dict(info)
            result["code"] = text
            result["confidence"] = 1.0
            return result

        # 别名匹配
        normalized = text.lower().replace(" ", "")
        code = self._alias_map.get(normalized)
        if code:
            info = SNOMED_TB_CODES.get(code)
            if info:
                result = dict(info)
                result["code"] = code
                result["confidence"] = 0.95
                return result

        # 包含匹配
        for code, info in SNOMED_TB_CODES.items():
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
        """通过编码查询"""
        info = SNOMED_TB_CODES.get(code.strip())
        if info:
            result = dict(info)
            result["code"] = code.strip()
            return result
        return None

    def get_all_codes(self) -> Dict[str, Dict[str, Any]]:
        """获取所有 SNOMED CT 编码"""
        return dict(SNOMED_TB_CODES)