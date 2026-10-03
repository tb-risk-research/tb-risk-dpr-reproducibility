#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LIS检验与PACS影像报告接入模块

LIS要点：
- 五大类结核相关检验项目映射（病原学/分子生物学/血清免疫/常规/感染筛查）
- 标本类型、方法学记录
- 历史结果时间线利用（阳性转归/治疗转阴追踪）

PACS要点：
- 第一阶段：结构化报告文本NLP（病变部位/性质/范围/播散/随访变化）
- 第二阶段：DICOM影像接入（预留接口）
- 放射科报告习惯用语覆盖
- 低置信度结果标记"待人工确认"
"""

from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import LOINC_TB_PANELS

LOGGER = logging.getLogger("tb_risk.health_interop.lis_pacs")


# ============================================================================
# LIS 检验项目映射
# ============================================================================

# 中文检验项目关键词 → 标准化检验项映射
LIS_KEYWORD_MAP: List[Tuple[str, Dict[str, Any]]] = [
    # 病原学
    ("痰涂片", {"category": "pathogen", "name": "痰涂片抗酸染色", "sample": "sputum"}),
    ("抗酸染色", {"category": "pathogen", "name": "痰涂片抗酸染色", "sample": "sputum"}),
    ("抗酸杆菌", {"category": "pathogen", "name": "痰涂片抗酸染色", "sample": "sputum"}),
    ("痰培养", {"category": "pathogen", "name": "痰结核分枝杆菌培养", "sample": "sputum"}),
    ("分枝杆菌培养", {"category": "pathogen", "name": "痰结核分枝杆菌培养", "sample": "sputum"}),
    ("BACTEC", {"category": "pathogen", "name": "BACTEC液体培养", "sample": "sputum"}),
    ("MGIT", {"category": "pathogen", "name": "BACTEC液体培养", "sample": "sputum"}),
    ("荧光染色", {"category": "pathogen", "name": "抗酸杆菌荧光染色", "sample": "sputum"}),
    # 分子生物学
    ("Xpert", {"category": "molecular", "name": "Xpert MTB/RIF", "sample": "sputum"}),
    ("MTB/RIF", {"category": "molecular", "name": "Xpert MTB/RIF", "sample": "sputum"}),
    ("利福平耐药", {"category": "molecular", "name": "Xpert MTB/RIF 利福平耐药", "sample": "sputum"}),
    ("rpoB", {"category": "molecular", "name": "rpoB基因突变检测", "sample": "sputum"}),
    ("熔解曲线", {"category": "molecular", "name": "熔解曲线法结核耐药", "sample": "sputum"}),
    ("基因测序", {"category": "molecular", "name": "结核分枝杆菌测序", "sample": "sputum"}),
    ("PCR", {"category": "molecular", "name": "结核PCR检测", "sample": "sputum"}),
    # 血清免疫
    ("T-SPOT", {"category": "serology", "name": "结核感染T细胞检测(T-SPOT)", "sample": "blood"}),
    ("TSPOT", {"category": "serology", "name": "结核感染T细胞检测(T-SPOT)", "sample": "blood"}),
    ("γ干扰素", {"category": "serology", "name": "γ干扰素释放试验(IGRA)", "sample": "blood"}),
    ("IGRA", {"category": "serology", "name": "γ干扰素释放试验(IGRA)", "sample": "blood"}),
    ("结核抗体", {"category": "serology", "name": "结核抗体", "sample": "blood"}),
    ("TB-Ab", {"category": "serology", "name": "结核抗体", "sample": "blood"}),
    ("PPD", {"category": "serology", "name": "结核菌素皮试(PPD)", "sample": "skin"}),
    ("结核菌素", {"category": "serology", "name": "结核菌素皮试(PPD)", "sample": "skin"}),
    # 常规
    ("血沉", {"category": "routine", "name": "血沉(ESR)", "sample": "blood", "unit": "mm/h"}),
    ("ESR", {"category": "routine", "name": "血沉(ESR)", "sample": "blood", "unit": "mm/h"}),
    ("C反应蛋白", {"category": "routine", "name": "C反应蛋白(CRP)", "sample": "blood", "unit": "mg/L"}),
    ("CRP", {"category": "routine", "name": "C反应蛋白(CRP)", "sample": "blood", "unit": "mg/L"}),
    ("空腹血糖", {"category": "routine", "name": "空腹血糖", "sample": "blood", "unit": "mmol/L"}),
    ("血糖", {"category": "routine", "name": "血糖", "sample": "blood", "unit": "mmol/L"}),
    ("GLU", {"category": "routine", "name": "血糖", "sample": "blood", "unit": "mmol/L"}),
    ("白细胞", {"category": "routine", "name": "白细胞计数(WBC)", "sample": "blood", "unit": "10^9/L"}),
    ("WBC", {"category": "routine", "name": "白细胞计数(WBC)", "sample": "blood", "unit": "10^9/L"}),
    ("血红蛋白", {"category": "routine", "name": "血红蛋白(HGB)", "sample": "blood", "unit": "g/L"}),
    ("HGB", {"category": "routine", "name": "血红蛋白(HGB)", "sample": "blood", "unit": "g/L"}),
    ("谷丙转氨酶", {"category": "routine", "name": "谷丙转氨酶(ALT)", "sample": "blood", "unit": "U/L"}),
    ("ALT", {"category": "routine", "name": "谷丙转氨酶(ALT)", "sample": "blood", "unit": "U/L"}),
    ("谷草转氨酶", {"category": "routine", "name": "谷草转氨酶(AST)", "sample": "blood", "unit": "U/L"}),
    ("AST", {"category": "routine", "name": "谷草转氨酶(AST)", "sample": "blood", "unit": "U/L"}),
    ("肌酐", {"category": "routine", "name": "肌酐(Cr)", "sample": "blood", "unit": "umol/L"}),
    # 感染筛查
    ("HIV", {"category": "infectious_screen", "name": "HIV1/2抗体", "sample": "blood"}),
    ("艾滋", {"category": "infectious_screen", "name": "HIV1/2抗体", "sample": "blood"}),
    ("乙肝表面抗原", {"category": "infectious_screen", "name": "乙肝表面抗原(HBsAg)", "sample": "blood"}),
    ("HBsAg", {"category": "infectious_screen", "name": "乙肝表面抗原(HBsAg)", "sample": "blood"}),
    ("丙肝抗体", {"category": "infectious_screen", "name": "丙肝抗体(HCV-Ab)", "sample": "blood"}),
    ("HCV", {"category": "infectious_screen", "name": "丙肝抗体(HCV-Ab)", "sample": "blood"}),
    ("梅毒", {"category": "infectious_screen", "name": "梅毒抗体(TP-Ab)", "sample": "blood"}),
    ("TP-Ab", {"category": "infectious_screen", "name": "梅毒抗体(TP-Ab)", "sample": "blood"}),
]

# 标本类型映射
SPECIMEN_MAP = {
    "痰": "sputum", "痰液": "sputum", "sputum": "sputum",
    "血": "blood", "血液": "blood", "blood": "blood", "血清": "blood", "血浆": "blood",
    "尿": "urine", "尿液": "urine", "urine": "urine",
    "脑脊液": "csf", "csf": "csf",
    "胸水": "pleural", "胸腔积液": "pleural",
    "腹水": "ascites",
    "支气管灌洗液": "bal", "BAL": "bal", "灌洗液": "bal",
    "脓液": "pus",
    "组织": "tissue",
    "皮肤": "skin",
}

# 阴阳性判定
POSITIVE_KEYWORDS = ("阳性", "+", "positive", "检出", "detected", "reactive",
                     "抗酸杆菌+", "杆菌+", "可见", "查到")
NEGATIVE_KEYWORDS = ("阴性", "-", "negative", "未检出", "not detected",
                     "non-reactive", "nonreactive", "未见", "未查到", "未发现")


@dataclass
class LabResultHistory:
    """检验结果历史（时间线）"""
    item_name: str = ""
    category: str = ""
    results: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_positive(self) -> bool:
        return any(r.get("is_positive") for r in self.results)

    @property
    def latest_positive_time(self) -> Optional[datetime.datetime]:
        pos = [r for r in self.results if r.get("is_positive") and r.get("effective_time")]
        if not pos:
            return None
        return max(r["effective_time"] for r in pos)

    @property
    def positive_count(self) -> int:
        return sum(1 for r in self.results if r.get("is_positive"))

    @property
    def has_seroconversion(self) -> bool:
        """是否有阳转（先阴后阳）或阴转（先阳后阴）"""
        sorted_results = sorted(
            [r for r in self.results if r.get("effective_time") and r.get("is_positive") is not None],
            key=lambda r: r["effective_time"])
        if len(sorted_results) < 2:
            return False
        first_val = sorted_results[0]["is_positive"]
        last_val = sorted_results[-1]["is_positive"]
        return first_val != last_val


class LISMapper:
    """LIS检验项目映射器"""

    def __init__(self):
        self._keyword_patterns = [
            (re.compile(re.escape(kw), re.IGNORECASE), info)
            for kw, info in LIS_KEYWORD_MAP
        ]

    def map_item(self, item_name: str, loinc_code: str = "",
                 item_code: str = "") -> Optional[Dict[str, Any]]:
        """将LIS检验项目名/LOINC编码映射为标准检验项

        返回 {"category", "name", "short", "loinc", "sample"} 或 None
        """
        # 1. 优先LOINC精确匹配
        if loinc_code:
            for category, items in LOINC_TB_PANELS.items():
                for item in items:
                    if item["loinc"] == loinc_code:
                        return {**item, "category": category, "match_method": "loinc"}
        # 2. 关键词匹配
        if item_name:
            for pattern, info in self._keyword_patterns:
                if pattern.search(item_name):
                    return {**info, "loinc": "", "short": info["name"][:8],
                            "match_method": "keyword"}
        return None

    @staticmethod
    def interpret_result(value: Any, item_name: str = "",
                         abnormal_flag: str = "",
                         reference_range: str = "") -> Optional[bool]:
        """解释检验结果阴阳性/异常性

        返回 True=阳性/异常, False=阴性/正常, None=无法判断
        """
        if value is None:
            return None
        s = str(value).strip().lower()
        # 先检查否定词（"未检出"、"未见"等优先于"检出"、"见"）
        for kw in NEGATIVE_KEYWORDS:
            if kw.lower() in s:
                return False
        # 直接关键词匹配
        for kw in POSITIVE_KEYWORDS:
            if kw.lower() in s:
                return True
        # 异常标志
        flag_upper = str(abnormal_flag).upper().strip()
        if flag_upper in ("H", "HH", "L", "LL", "A", "AA", "+", "POS"):
            return True
        if flag_upper in ("N", "NORMAL", "-", "NEG"):
            return False
        return None

    @staticmethod
    def detect_specimen(specimen_text: str, item_name: str = "") -> str:
        """检测标本类型"""
        text = (specimen_text or "").lower()
        for key, std in SPECIMEN_MAP.items():
            if key.lower() in text:
                return std
        # 从项目名推断
        name_lower = (item_name or "").lower()
        for key, std in SPECIMEN_MAP.items():
            if key.lower() in name_lower:
                return std
        return "unknown"

    def summarize_history(self, results: List[Dict],
                          lookback_months: int = 6) -> Dict[str, LabResultHistory]:
        """按检验项目汇总历史结果时间线

        返回 {item_key: LabResultHistory}，用于提取阳转/阴转等时间特征
        """
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(days=30 * lookback_months)
        histories: Dict[str, LabResultHistory] = {}
        for r in results:
            name = r.get("name", "")
            category = r.get("category", "")
            key = f"{category}:{name}"
            if key not in histories:
                histories[key] = LabResultHistory(item_name=name, category=category)
            eff_time = r.get("effective_time")
            if isinstance(eff_time, str):
                from .base import timestamp_to_datetime
                eff_time = timestamp_to_datetime(eff_time)
            if eff_time and eff_time < cutoff:
                continue  # 超过窗口期的不统计
            histories[key].results.append({**r, "effective_time": eff_time})
        return histories


# ============================================================================
# PACS 影像报告NLP
# ============================================================================

# 结核好发部位关键词
TB_SITE_KEYWORDS = {
    "apical_posterior": ["上叶尖后段", "尖后段", "apicoposterior", "上叶尖段", "上叶后段"],
    "dorsal": ["下叶背段", "背段", "dorsal segment", "superior segment"],
    "upper_lobe": ["上叶", "upper lobe"],
    "lower_lobe": ["下叶", "lower lobe"],
    "right_lung": ["右肺", "right lung"],
    "left_lung": ["左肺", "left lung"],
    "bilateral": ["双肺", "两肺", "bilateral", "both lungs"],
    "miliary": ["粟粒", "miliary", "弥漫", "diffuse"],
    "pleura": ["胸膜", "胸腔", "pleura", "pleural"],
    "hilar": ["肺门", "hilar", "hilum"],
    "mediastinum": ["纵隔", "mediastinal"],
}

# 病变性质关键词
LESION_KEYWORDS = {
    "patchy_opacity": ["斑片状阴影", "斑片影", "斑片", "patchy", "infiltrate", "exudate"],
    "nodule": ["结节影", "结节", "nodule", "nodular"],
    "cavity": ["空洞", "cavity", "cavitation", "cavitary", "壁空洞", "无壁空洞"],
    "fibrotic_streak": ["纤维条索影", "条索影", "条索灶", "fibrotic", "streak", "linear"],
    "calcification": ["钙化", "calcification", "calcified"],
    "consolidation": ["实变", "consolidation"],
    "pleural_effusion": ["胸腔积液", "胸水", "pleural effusion", "effusion"],
    "pleural_thickening": ["胸膜增厚", "pleural thickening"],
    "bronchiectasis": ["支气管扩张", "支扩", "bronchiectasis"],
    "lymphadenopathy": ["淋巴结肿大", "淋巴结增大", "lymphadenopathy"],
    "traction_bronchiectasis": ["牵拉性支扩", "traction bronchiectasis"],
}

# 播散征象关键词
DISSEMINATION_KEYWORDS = {
    "bronchogenic": ["支气管播散", "支气管扩散", "沿支气管分布", "tree-in-bud", "树芽征"],
    "hematogenous": ["血行播散", "血源播散", "血性播散", "粟粒性", "miliary spread"],
    "lymphangitic": ["淋巴管播散", "淋巴道播散"],
}

# 随访变化关键词
CHANGE_KEYWORDS = {
    "worsening": ["增大", "增多", "扩大", "进展", "加重", "increased", "enlarged", "worsening", "progressed"],
    "improving": ["吸收", "缩小", "减少", "好转", "改善", "decreased", "improved", "absorption", "resolving"],
    "stable": ["相仿", "无明显变化", "稳定", "stable", "no significant change"],
    "new": ["新出现", "新发", "新增", "new", "newly appeared"],
}

# 待排/疑似表述（置信度降低）
SUSPICIOUS_PHRASES = [
    "TB待排", "结核待排", "TB可能性大", "结核可能", "不排除结核", "可疑结核",
    "结核?", "tb?", "suggestive of tb", "rule out tb", "r/o tb", "tb待查",
    "抗酸染色阳性可能", "抗酸杆菌阳性可能",
]

# 否定词
NEGATION_WORDS = [
    "未见", "未发现", "无明显", "没有", "未见明显", "未显示", "未见异常", "无异常",
    "排除", "未及",
]


class RadiologyNLP:
    """影像报告NLP特征提取器

    提取结核相关影像征象：
    - 病变部位（上叶尖后段/下叶背段等结核好发部位）
    - 病变性质（斑片/结节/空洞/纤维条索/钙化/胸腔积液等）
    - 病灶范围（单肺/双肺、肺叶受累数）
    - 有无播散病灶（支气管播散/血行播散）
    - 随访变化（增大/增多/吸收/好转）
    """

    def __init__(self):
        self._site_patterns = self._compile(TB_SITE_KEYWORDS)
        self._lesion_patterns = self._compile(LESION_KEYWORDS)
        self._dissem_patterns = self._compile(DISSEMINATION_KEYWORDS)
        self._change_patterns = self._compile(CHANGE_KEYWORDS)
        self._suspicious_regex = re.compile(
            "|".join(re.escape(p) for p in SUSPICIOUS_PHRASES), re.IGNORECASE)
        self._negation_regex = re.compile(
            "|".join(re.escape(w) for w in NEGATION_WORDS))

    @staticmethod
    def _compile(keyword_dict: Dict[str, List[str]]):
        return {
            key: [re.compile(re.escape(kw), re.IGNORECASE) for kw in kws]
            for key, kws in keyword_dict.items()
        }

    @staticmethod
    def _find_keywords(text: str, patterns: Dict[str, List[re.Pattern]]) -> List[str]:
        found = []
        for key, pats in patterns.items():
            for p in pats:
                if p.search(text):
                    found.append(key)
                    break
        return found

    def _check_negation(self, text: str, feature_pos: int, window: int = 8) -> bool:
        """检查特征词附近是否有否定词（简单启发式）"""
        start = max(0, feature_pos - window * 2)
        end = min(len(text), feature_pos + window)
        context = text[start:end]
        return bool(self._negation_regex.search(context))

    def extract_features(self, report_text: str) -> Dict[str, Any]:
        """从放射报告文本中提取影像特征

        返回包含 sites/lesions/dissemination/changes/confidence/is_suspicious 的dict
        """
        text = report_text or ""
        text_lower = text.lower()

        # 基础命中
        sites = self._find_keywords(text, self._site_patterns)
        lesions = self._find_keywords(text, self._lesion_patterns)
        dissemination = self._find_keywords(text, self._dissem_patterns)
        changes = self._find_keywords(text, self._change_patterns)

        # 病灶范围估算
        lung_count = 0
        if "right_lung" in sites:
            lung_count += 1
        if "left_lung" in sites:
            lung_count += 1
        if "bilateral" in sites:
            lung_count = 2
        lobe_count = 0
        if "upper_lobe" in sites:
            lobe_count += 1
        if "lower_lobe" in sites:
            lobe_count += 1

        # 好发部位评分
        tb_predilection = (
            "apical_posterior" in sites or
            "dorsal" in sites or
            "miliary" in sites
        )

        # 置信度评估
        is_suspicious_tbd = bool(self._suspicious_regex.search(text))
        has_tb_keyword = any(
            k in text_lower
            for k in ("结核", "tb", "tuberculosis", "抗酸杆菌", "结核分枝杆菌",
                      "tb待排", "tb可能")
        )

        # 基础置信度
        confidence = 0.5
        if lesions:
            confidence += 0.15
        if tb_predilection:
            confidence += 0.15
        if "cavity" in lesions:
            confidence += 0.15
        if dissemination:
            confidence += 0.1
        if is_suspicious_tbd:
            confidence -= 0.1
        if not has_tb_keyword and not tb_predilection:
            confidence -= 0.15
        confidence = max(0.1, min(0.95, confidence))

        needs_manual_review = confidence < 0.4 or is_suspicious_tbd

        # 提示tb_risk特征字段
        features = {
            "sites": sites,
            "lesions": lesions,
            "dissemination": dissemination,
            "changes": changes,
            "lung_count": lung_count,
            "lobe_count": lobe_count,
            "tb_predilection_site": tb_predilection,
            "has_cavity": "cavity" in lesions,
            "has_pleural_effusion": "pleural_effusion" in lesions,
            "has_calcification": "calcification" in lesions,
            "has_fibrotic": "fibrotic_streak" in lesions,
            "is_miliary": "miliary" in sites,
            "has_dissemination": bool(dissemination),
            "bronchogenic_spread": "bronchogenic" in dissemination,
            "hematogenous_spread": "hematogenous" in dissemination,
            "worsening": "worsening" in changes or "new" in changes,
            "improving": "improving" in changes,
            "confidence": round(confidence, 2),
            "needs_manual_review": needs_manual_review,
            "is_suspicious": is_suspicious_tbd,
            "mentions_tb": has_tb_keyword,
        }
        return features


class LISPACSAdapter:
    """LIS/PACS数据接入适配器

    可作为其他适配器（FHIR/HL7v2/DB_Direct）的辅助模块使用，
    提供检验项目标准化、结果解释、历史时间线、影像NLP等能力。
    """

    def __init__(self):
        self.lis_mapper = LISMapper()
        self.rad_nlp = RadiologyNLP()

    def standardize_lab_result(self, raw_lab: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将原始检验记录标准化为tb_risk内部格式"""
        item_name = str(raw_lab.get("item_name", raw_lab.get("name", "")))
        loinc = str(raw_lab.get("loinc", raw_lab.get("loinc_code", "")))
        item_code = str(raw_lab.get("item_code", raw_lab.get("local_code", "")))
        value = raw_lab.get("value", raw_lab.get("result_value"))
        unit = str(raw_lab.get("unit", ""))
        flag = str(raw_lab.get("abnormal_flag", raw_lab.get("flag", "")))
        ref_range = str(raw_lab.get("reference_range", raw_lab.get("ref_range", "")))
        sample = str(raw_lab.get("specimen_type", raw_lab.get("sample_type", "")))

        mapped = self.lis_mapper.map_item(item_name, loinc, item_code)
        if not mapped:
            # 未映射的检验仍然保留，但标记为非结核核心项
            mapped = {
                "category": "other",
                "name": item_name or item_code,
                "short": (item_name or item_code)[:8],
                "sample": "",
                "match_method": "unmapped",
                "loinc": "",
            }

        is_positive = self.lis_mapper.interpret_result(value, item_name, flag, ref_range)
        specimen = self.lis_mapper.detect_specimen(sample, item_name)

        return {
            **raw_lab,
            "name": mapped["name"],
            "short_name": mapped.get("short", ""),
            "category": mapped["category"],
            "loinc": mapped.get("loinc", ""),
            "match_method": mapped.get("match_method", "raw"),
            "value": value,
            "unit": unit or mapped.get("unit", ""),
            "is_positive": is_positive,
            "is_abnormal": flag.upper() in ("H", "HH", "L", "LL", "A", "+") or is_positive is True,
            "specimen_type": specimen,
            "reference_range": ref_range,
        }

    def process_imaging_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """处理影像报告：执行NLP提取特征"""
        text = report.get("conclusion", "") or report.get("content", "") or ""
        title = report.get("title", "") or ""
        nlp_features = self.rad_nlp.extract_features(
            f"{title}\n{text}" if title else text)
        return {
            **report,
            "nlp_features": nlp_features,
            "nlp_processed": True,
            "needs_manual_review": nlp_features["needs_manual_review"],
        }

    def extract_risk_features_from_labs(self, lab_results: List[Dict]) -> Dict[str, Any]:
        """从历史检验结果中提取tb_risk所需风险特征

        返回的字段直接可用于填充22维特征中的相关字段。
        """
        std_results = []
        for r in lab_results:
            std = self.standardize_lab_result(r)
            if std:
                std_results.append(std)

        histories = self.lis_mapper.summarize_history(std_results)
        features = {
            "sputum_smear_positive": False,
            "sputum_culture_positive": False,
            "xpert_positive": False,
            "rif_resistant": False,
            "tspot_positive": False,
            "hiv_positive": False,
            "diabetes_risk": None,  # 血糖异常
            "crp_elevated": False,
            "esr_elevated": False,
            "positive_lookback_6m": False,
            "recent_positive_days": None,
            "seroconversion": False,
        }

        now = datetime.datetime.now()
        for key, hist in histories.items():
            if hist.category == "pathogen":
                if "涂片" in hist.item_name or "smear" in hist.item_name.lower():
                    features["sputum_smear_positive"] = hist.has_positive
                if "培养" in hist.item_name or "culture" in hist.item_name.lower():
                    features["sputum_culture_positive"] = hist.has_positive
                if hist.has_seroconversion:
                    features["seroconversion"] = True
            elif hist.category == "molecular":
                if "xpert" in hist.item_name.lower() or "rif" in hist.item_name.lower():
                    features["xpert_positive"] = hist.has_positive
                    # 利福平耐药（结果含"耐药"或rpoB突变）
                    for r in hist.results:
                        v = str(r.get("value", "")).lower()
                        if "耐药" in v or "mutation" in v or "mutant" in v or "rpo" in v:
                            features["rif_resistant"] = True
                            break
            elif hist.category == "serology":
                if "t-spot" in hist.item_name.lower() or "igra" in hist.item_name.lower():
                    features["tspot_positive"] = hist.has_positive
            elif hist.category == "infectious_screen":
                if "hiv" in hist.item_name.lower() or "艾滋" in hist.item_name:
                    features["hiv_positive"] = hist.has_positive
            elif hist.category == "routine":
                if "crp" in hist.item_name.lower() or "C反应" in hist.item_name:
                    features["crp_elevated"] = any(
                        r.get("is_abnormal") for r in hist.results)
                if "esr" in hist.item_name.lower() or "血沉" in hist.item_name:
                    features["esr_elevated"] = any(
                        r.get("is_abnormal") for r in hist.results)
                if "血糖" in hist.item_name or "glu" in hist.item_name.lower():
                    # 高血糖提示糖尿病风险
                    for r in hist.results:
                        try:
                            v = float(r.get("value", 0))
                            if v > 7.0:
                                features["diabetes_risk"] = True
                                break
                        except (TypeError, ValueError):
                            pass

            # 最近阳性时间
            if hist.has_positive and hist.latest_positive_time:
                days = (now - hist.latest_positive_time).days
                if features["recent_positive_days"] is None or days < features["recent_positive_days"]:
                    features["recent_positive_days"] = days
                if days <= 180:
                    features["positive_lookback_6m"] = True

        return features


class LISPACSDecorator:
    """LIS/PACS装饰器适配器

    包装任意HealthcareAdapter实例，在fetch_patient/fetch_incremental返回结果后
    自动增强：检验项目标准化、结果解释、历史时间线、影像NLP征象提取。
    """

    def __init__(self, base_adapter, enable_lis: bool = True, enable_pacs: bool = True):
        from .base import HealthcareAdapter
        if not isinstance(base_adapter, HealthcareAdapter):
            raise TypeError("base_adapter 必须是 HealthcareAdapter 实例")
        self._base = base_adapter
        self._lis_pacs = LISPACSAdapter()
        self.enable_lis = enable_lis
        self.enable_pacs = enable_pacs
        self.config = base_adapter.config
        self._connected = False

    def connect(self) -> bool:
        """委托给基础适配器"""
        ok = self._base.connect()
        self._connected = ok
        return ok

    def disconnect(self):
        self._base.disconnect()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def fetch_patient(self, patient_id: str) -> Any:
        """拉取患者后自动增强LIS/PACS处理"""
        from .base import AdapterResult
        result = self._base.fetch_patient(patient_id)
        if result.success:
            result = self._enhance_result(result)
        return result

    def fetch_incremental(self, since: Optional[datetime.datetime] = None) -> List[Any]:
        """增量拉取后自动增强每个结果"""
        results = self._base.fetch_incremental(since)
        return [self._enhance_result(r) for r in results]

    def _enhance_result(self, result: Any) -> Any:
        """对AdapterResult执行LIS/PACS增强处理"""
        # LIS检验标准化
        if self.enable_lis and result.lab_results:
            std_labs = []
            for lab in result.lab_results:
                std = self._lis_pacs.standardize_lab_result(lab)
                if std:
                    std_labs.append(std)
            result.lab_results = std_labs
            # 提取检验风险特征
            lis_features = self._lis_pacs.extract_risk_features_from_labs(std_labs)
            result.risk_features.update(lis_features)
            result.warnings.append("已启用LIS检验项目标准化映射")

        # PACS影像NLP
        if self.enable_pacs and result.imaging_reports:
            processed_imgs = []
            for img in result.imaging_reports:
                proc = self._lis_pacs.process_imaging_report(img)
                processed_imgs.append(proc)
            result.imaging_reports = processed_imgs
            # 从NLP特征提取影像风险标记
            img_features = {
                "has_cavity_any": False,
                "has_pleural_effusion_any": False,
                "tb_predilection_site_any": False,
                "bronchogenic_spread_any": False,
                "imaging_needs_review": False,
            }
            for img in processed_imgs:
                nlp = img.get("nlp_features", {})
                if nlp.get("has_cavity"):
                    img_features["has_cavity_any"] = True
                if nlp.get("has_pleural_effusion"):
                    img_features["has_pleural_effusion_any"] = True
                if nlp.get("tb_predilection_site"):
                    img_features["tb_predilection_site_any"] = True
                if nlp.get("bronchogenic_spread"):
                    img_features["bronchogenic_spread_any"] = True
                if nlp.get("needs_manual_review"):
                    img_features["imaging_needs_review"] = True
            result.risk_features.update(img_features)
            result.warnings.append("已启用PACS影像报告NLP征象提取")

        return result

    def __getattr__(self, name):
        """透传基础适配器的其他属性/方法"""
        return getattr(self._base, name)
