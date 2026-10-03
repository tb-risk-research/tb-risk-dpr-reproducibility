#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""国标编码映射引擎（GB/T 3304/6565/2260 等）

支持以下国标编码的映射：
1. GB/T 3304-1991：中国各民族名称的罗马字母拼写
2. GB/T 6565-2015：职业分类与代码
3. GB/T 2260-2007：中华人民共和国行政区划代码
4. 卫生信息数据元目录（WS 363-2011 等）
"""

import logging
import re
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("tb_risk.health_interop.terminology.gb")

# ============================================================================
# GB/T 3304-1991 民族编码
# ============================================================================

ETHNICITY_CODES: Dict[str, Dict[str, Any]] = {
    "01": {"code": "01", "roman": "Han", "name": "汉族", "aliases": ["汉", "汉族"]},
    "02": {"code": "02", "roman": "Mongol", "name": "蒙古族", "aliases": ["蒙", "蒙古族"]},
    "03": {"code": "03", "roman": "Hui", "name": "回族", "aliases": ["回", "回族"]},
    "04": {"code": "04", "roman": "Zang", "name": "藏族", "aliases": ["藏", "藏族"]},
    "05": {"code": "05", "roman": "Uyghur", "name": "维吾尔族", "aliases": ["维", "维吾尔族", "维吾尔"]},
    "06": {"code": "06", "roman": "Miao", "name": "苗族", "aliases": ["苗", "苗族"]},
    "07": {"code": "07", "roman": "Yi", "name": "彝族", "aliases": ["彝", "彝族"]},
    "08": {"code": "08", "roman": "Zhuang", "name": "壮族", "aliases": ["壮", "壮族"]},
    "09": {"code": "09", "roman": "Buyei", "name": "布依族", "aliases": ["布依", "布依族"]},
    "10": {"code": "10", "roman": "Korean", "name": "朝鲜族", "aliases": ["朝鲜", "朝鲜族"]},
    "11": {"code": "11", "roman": "Manchu", "name": "满族", "aliases": ["满", "满族"]},
    "12": {"code": "12", "roman": "Dong", "name": "侗族", "aliases": ["侗", "侗族"]},
    "13": {"code": "13", "roman": "Yao", "name": "瑶族", "aliases": ["瑶", "瑶族"]},
    "14": {"code": "14", "roman": "Bai", "name": "白族", "aliases": ["白", "白族"]},
    "15": {"code": "15", "roman": "Tujia", "name": "土家族", "aliases": ["土家", "土家族"]},
    "17": {"code": "17", "roman": "Kazakh", "name": "哈萨克族", "aliases": ["哈", "哈萨克", "哈萨克族"]},
    "29": {"code": "29", "roman": "Kirgiz", "name": "柯尔克孜族", "aliases": ["柯尔克孜", "柯尔克孜族"]},
    "31": {"code": "31", "roman": "Xibe", "name": "锡伯族", "aliases": ["锡伯", "锡伯族"]},
    "99": {"code": "99", "roman": "Other", "name": "其他", "aliases": ["其他", "其它"]},
}

# 别名 → 编码反向索引
_ETHNICITY_ALIAS_MAP: Dict[str, str] = {}
for code, info in ETHNICITY_CODES.items():
    for alias in info["aliases"]:
        _ETHNICITY_ALIAS_MAP[alias] = code
    _ETHNICITY_ALIAS_MAP[info["name"]] = code
    _ETHNICITY_ALIAS_MAP[info["roman"].lower()] = code

# ============================================================================
# GB/T 6565-2015 职业分类（简化版，覆盖结核病相关高风险职业）
# ============================================================================

OCCUPATION_CODES: Dict[str, Dict[str, Any]] = {
    "1": {"code": "1", "name": "国家机关、党群组织、企业、事业单位负责人", "level": "major"},
    "2": {"code": "2", "name": "专业技术人员", "level": "major",
          "sub": {
              "21": {"code": "21", "name": "科学研究人员"},
              "22": {"code": "22", "name": "工程技术人员"},
              "23": {"code": "23", "name": "农业技术人员"},
              "24": {"code": "24", "name": "卫生专业技术人员",
                     "aliases": ["医生", "护士", "医护", "医务人员", "医技", "药剂"]},
          }},
    "3": {"code": "3", "name": "办事人员和有关人员", "level": "major"},
    "4": {"code": "4", "name": "社会生产服务和生活服务人员", "level": "major"},
    "5": {"code": "5", "name": "农、林、牧、渔业生产及辅助人员", "level": "major"},
    "6": {"code": "6", "name": "生产制造及有关人员", "level": "major",
          "sub": {
              "61": {"code": "61", "name": "采矿人员",
                     "aliases": ["矿工", "采矿", "煤矿", "井下"]},
              "62": {"code": "62", "name": "石油和天然气开采与炼制人员",
                     "aliases": ["油田", "石油", "钻井", "采油", "炼油"]},
              "63": {"code": "63", "name": "金属冶炼和轧制人员"},
              "64": {"code": "64", "name": "建材生产人员",
                     "aliases": ["水泥", "石材", "粉尘"]},
          }},
    "7": {"code": "7", "name": "军人", "level": "major"},
    "8": {"code": "8", "name": "不便分类的其他从业人员", "level": "major"},
    "X": {"code": "X", "name": "学生", "level": "special",
          "aliases": ["学生", "学生", "school"]},
    "Y": {"code": "Y", "name": "无业/待业/失业", "level": "special",
          "aliases": ["无业", "待业", "失业", "退休", "离休"]},
}

# 职业别名 → 编码映射
_OCCUPATION_ALIAS_MAP: Dict[str, str] = {}
for major_code, major_info in OCCUPATION_CODES.items():
    for alias in major_info.get("aliases", []):
        _OCCUPATION_ALIAS_MAP[alias] = major_code
    for sub_code, sub_info in major_info.get("sub", {}).items():
        for alias in sub_info.get("aliases", []):
            _OCCUPATION_ALIAS_MAP[alias] = sub_code

# ============================================================================
# GB/T 2260-2007 行政区划代码（新疆、克拉玛依相关）
# ============================================================================

DIVISION_CODES: Dict[str, Dict[str, Any]] = {
    "650000": {"code": "650000", "name": "新疆维吾尔自治区", "level": "province"},
    "650200": {"code": "650200", "name": "克拉玛依市", "level": "city", "parent": "650000"},
    "650201": {"code": "650201", "name": "市辖区", "level": "district", "parent": "650200"},
    "650202": {"code": "650202", "name": "独山子区", "level": "district", "parent": "650200"},
    "650203": {"code": "650203", "name": "克拉玛依区", "level": "district", "parent": "650200"},
    "650204": {"code": "650204", "name": "白碱滩区", "level": "district", "parent": "650200"},
    "650205": {"code": "650205", "name": "乌尔禾区", "level": "district", "parent": "650200"},
    "650100": {"code": "650100", "name": "乌鲁木齐市", "level": "city", "parent": "650000"},
    "652800": {"code": "652800", "name": "巴音郭楞蒙古自治州", "level": "city", "parent": "650000"},
    "652900": {"code": "652900", "name": "阿克苏地区", "level": "city", "parent": "650000"},
    "653000": {"code": "653000", "name": "克孜勒苏柯尔克孜自治州", "level": "city", "parent": "650000"},
    "653100": {"code": "653100", "name": "喀什地区", "level": "city", "parent": "650000"},
    "653200": {"code": "653200", "name": "和田地区", "level": "city", "parent": "650000"},
    "654000": {"code": "654000", "name": "伊犁哈萨克自治州", "level": "city", "parent": "650000"},
}

# ============================================================================
# 卫生信息数据元目录（WS 363-2011 相关）
# ============================================================================

HEALTH_DATA_ELEMENTS: Dict[str, Dict[str, Any]] = {
    "DE02.01.001.00": {"name": "性别代码", "standard": "WS 363-2011", "code_system": "GB/T 2261.1"},
    "DE02.01.002.00": {"name": "出生日期", "standard": "WS 363-2011", "format": "YYYYMMDD"},
    "DE02.01.003.00": {"name": "年龄", "standard": "WS 363-2011", "unit": "岁"},
    "DE02.01.004.00": {"name": "民族代码", "standard": "WS 363-2011", "code_system": "GB/T 3304"},
    "DE02.01.005.00": {"name": "职业代码", "standard": "WS 363-2011", "code_system": "GB/T 6565"},
    "DE02.01.006.00": {"name": "婚姻状况代码", "standard": "WS 363-2011", "code_system": "GB/T 2261.2"},
    "DE02.01.007.00": {"name": "户口所在地代码", "standard": "WS 363-2011", "code_system": "GB/T 2260"},
    "DE02.01.008.00": {"name": "现住址代码", "standard": "WS 363-2011", "code_system": "GB/T 2260"},
    "DE04.01.001.00": {"name": "疾病诊断代码", "standard": "WS 363-2011", "code_system": "ICD-10"},
    "DE04.01.002.00": {"name": "疾病诊断名称", "standard": "WS 363-2011"},
    "DE04.50.001.00": {"name": "手术操作代码", "standard": "WS 363-2011", "code_system": "ICD-9-CM-3"},
    "DE05.01.001.00": {"name": "检验项目代码", "standard": "WS 363-2011", "code_system": "LOINC"},
    "DE05.01.002.00": {"name": "检验结果值", "standard": "WS 363-2011"},
    "DE05.01.003.00": {"name": "检验结果单位", "standard": "WS 363-2011"},
    "DE05.01.004.00": {"name": "检验结果异常标志", "standard": "WS 363-2011"},
    "DE06.00.001.00": {"name": "影像检查报告", "standard": "WS 363-2011"},
    "DE08.10.001.00": {"name": "吸烟史", "standard": "WS 363-2011"},
    "DE08.10.002.00": {"name": "饮酒史", "standard": "WS 363-2011"},
    "DE08.10.003.00": {"name": "过敏史", "standard": "WS 363-2011"},
    "DE08.10.004.00": {"name": "手术史", "standard": "WS 363-2011"},
    "DE08.10.005.00": {"name": "输血史", "standard": "WS 363-2011"},
    "DE09.00.001.00": {"name": "结核病接触史", "standard": "WS 363-2011"},
    "DE09.00.002.00": {"name": "卡介苗接种史", "standard": "WS 363-2011"},
}


class GBCodeMapper:
    """国标编码映射引擎"""

    # ==================== 民族编码 ====================

    def match_ethnicity(self, text: str) -> Optional[Dict[str, Any]]:
        """匹配民族编码

        参数：
            text: 民族名称（如"汉族"、"维"、"uyghur"）

        返回：
            dict | None: {code, name, roman}
        """
        if not text:
            return None
        text = text.strip()
        code = _ETHNICITY_ALIAS_MAP.get(text)
        if code:
            info = ETHNICITY_CODES.get(code)
            if info:
                return dict(info)
        # 尝试子串匹配
        for code, info in ETHNICITY_CODES.items():
            for alias in info["aliases"]:
                if alias in text or text in alias:
                    return dict(info)
        return None

    def get_all_ethnicities(self) -> List[Dict[str, Any]]:
        """获取所有民族编码"""
        return [dict(info) for info in ETHNICITY_CODES.values()]

    # ==================== 职业编码 ====================

    def match_occupation(self, text: str) -> Optional[Dict[str, Any]]:
        """匹配职业编码

        参数：
            text: 职业名称（如"医生"、"矿工"、"油田工人"）

        返回：
            dict | None: {code, name, level}
        """
        if not text:
            return None
        text = text.strip().lower()
        code = _OCCUPATION_ALIAS_MAP.get(text)
        if code:
            # 查找编码
            for major_code, major_info in OCCUPATION_CODES.items():
                if major_code == code:
                    return {"code": code, "name": major_info["name"], "level": major_info["level"]}
                for sub_code, sub_info in major_info.get("sub", {}).items():
                    if sub_code == code:
                        return {"code": code, "name": sub_info["name"], "level": "minor"}
        # 子串匹配
        for major_code, major_info in OCCUPATION_CODES.items():
            for alias in major_info.get("aliases", []):
                if alias.lower() in text:
                    return {"code": major_code, "name": major_info["name"], "level": major_info["level"]}
            for sub_code, sub_info in major_info.get("sub", {}).items():
                for alias in sub_info.get("aliases", []):
                    if alias.lower() in text:
                        return {"code": sub_code, "name": sub_info["name"], "level": "minor"}
        return None

    def is_high_risk_occupation(self, text: str) -> bool:
        """判断是否为结核病高风险职业"""
        if not text:
            return False
        text = text.lower()
        high_risk_keywords = [
            "矿", "油田", "石油", "钻井", "粉尘", "水泥", "医护",
            "医生", "护士", "医院", "羁押", "监狱", "看守所",
        ]
        return any(kw in text for kw in high_risk_keywords)

    # ==================== 行政区划编码 ====================

    def match_division(self, code_or_name: str) -> Optional[Dict[str, Any]]:
        """匹配行政区划编码

        参数：
            code_or_name: 编码（如"650203"）或名称（如"克拉玛依区"）

        返回：
            dict | None: {code, name, level, parent}
        """
        if not code_or_name:
            return None
        key = code_or_name.strip()
        # 直接编码匹配
        info = DIVISION_CODES.get(key)
        if info:
            return dict(info)
        # 名称匹配
        for code, info in DIVISION_CODES.items():
            if info["name"] == key:
                return dict(info)
        return None

    def get_child_divisions(self, parent_code: str) -> List[Dict[str, Any]]:
        """获取子级行政区划"""
        return [
            dict(info) for info in DIVISION_CODES.values()
            if info.get("parent") == parent_code
        ]

    # ==================== 卫生信息数据元 ====================

    def lookup_data_element(self, element_code: str) -> Optional[Dict[str, Any]]:
        """查询卫生信息数据元目录"""
        info = HEALTH_DATA_ELEMENTS.get(element_code)
        if info:
            return dict(info)
        return None

    def match_data_element_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """通过名称查找数据元"""
        name_lower = name.lower().strip()
        for code, info in HEALTH_DATA_ELEMENTS.items():
            if name_lower in info["name"].lower() or info["name"].lower() in name_lower:
                result = dict(info)
                result["code"] = code
                return result
        return None