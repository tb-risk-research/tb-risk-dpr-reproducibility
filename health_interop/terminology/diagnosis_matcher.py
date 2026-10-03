#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文诊断文本 → ICD-10 智能匹配引擎（增强版）

整合 ICD10Mapper 的多种匹配策略，增加：
1. 多诊断文本拆分与并行匹配
2. 否定诊断排除（如"排除肺结核"、"无糖尿病"）
3. 诊断置信度加权综合评估
4. 传染病报告卡自动生成辅助
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .icd10_mapper import ICD10Mapper

LOGGER = logging.getLogger("tb_risk.health_interop.terminology.matcher")

# 否定诊断前缀
NEGATION_PREFIXES = [
    "排除", "除外", "无", "未见", "否认", "未发现", "不伴", "无明确",
    "无证据", "排除性", "待排除", "需排除", "不除外", "不排除",
    "no ", "without ", "exclude ", "rule out ", "r/o ",
]

# 诊断分隔符
DIAGNOSIS_SEPARATORS = [
    "；", ";", "。", "，", ",", "\n", "   ", "  ",
    "1.", "2.", "3.", "4.", "5.", "①", "②", "③",
]


class DiagnosisMatcher:
    """中文诊断文本 → ICD-10 智能匹配引擎

    支持：
    - 多诊断文本拆分（如"1.肺结核 2.2型糖尿病"）
    - 否定诊断排除
    - 置信度加权综合评估
    - 传染病报告卡辅助生成
    - 批量匹配与统计
    """

    def __init__(self):
        self._icd10 = ICD10Mapper()
        self._last_results: List[Dict[str, Any]] = []

    # ==================== 核心 API ====================

    def match(self, diagnosis_text: str) -> List[Dict[str, Any]]:
        """匹配诊断文本（自动拆分多诊断）

        参数：
            diagnosis_text: 诊断文本（如 "1.肺结核 2.2型糖尿病"）

        返回：
            list[dict]: 匹配结果列表，每项含：
                {code, display, confidence, category, match_type, is_negated, original_text}
        """
        if not diagnosis_text or not isinstance(diagnosis_text, str):
            return []

        text = diagnosis_text.strip()
        if not text:
            return []

        # 拆分多诊断
        sub_diagnoses = self._split_diagnoses(text)

        # 逐条匹配
        results = []
        seen_codes = set()
        for sub in sub_diagnoses:
            if not sub.strip():
                continue

            is_negated = self._is_negated(sub)
            clean_text = self._remove_negation_prefix(sub).strip()

            if not clean_text:
                continue

            match_result = self._icd10.match_diagnosis(clean_text)
            if match_result:
                code = match_result["code"]
                # 同编码去重，保留置信度高的
                if code in seen_codes:
                    for existing in results:
                        if existing["code"] == code and existing["confidence"] < match_result["confidence"]:
                            existing.update(match_result)
                            existing["is_negated"] = is_negated
                            existing["original_text"] = clean_text
                            break
                else:
                    match_result["is_negated"] = is_negated
                    match_result["original_text"] = clean_text
                    results.append(match_result)
                    seen_codes.add(code)

        self._last_results = results
        return results

    def match_batch(self, diagnosis_texts: List[str]) -> Dict[str, Any]:
        """批量匹配多条诊断文本（用于多个患者或多个诊断来源）

        参数：
            diagnosis_texts: 诊断文本列表

        返回：
            dict: {
                "total": int,
                "matched": int,
                "unmatched": int,
                "results": [list of match results],
                "tb_related": int,
                "comorbidities": {type: count}
            }
        """
        all_results = []
        matched_count = 0
        for text in diagnosis_texts:
            results = self.match(text)
            if results:
                matched_count += 1
            all_results.extend(results)

        # 统计
        tb_related = sum(1 for r in all_results if r.get("category") in ("active_tb", "past_tb") and not r.get("is_negated"))
        comorbidities = {"diabetes": 0, "hiv": 0, "copd": 0}
        for r in all_results:
            c = r.get("comorbidity")
            if c in comorbidities and not r.get("is_negated"):
                comorbidities[c] += 1

        return {
            "total": len(diagnosis_texts),
            "matched": matched_count,
            "unmatched": len(diagnosis_texts) - matched_count,
            "total_matches": len(all_results),
            "results": all_results,
            "tb_related": tb_related,
            "comorbidities": comorbidities,
        }

    def extract_tb_info(self, diagnosis_text: str) -> Dict[str, Any]:
        """从诊断文本提取结核病相关信息

        参数：
            diagnosis_text: 诊断文本

        返回：
            dict: {
                "has_active_tb": bool,
                "has_past_tb": bool,
                "tb_diagnoses": [str, ...],
                "comorbidities": {diabetes: bool, hiv: bool, copd: bool},
                "confidence": float
            }
        """
        results = self.match(diagnosis_text)
        info = {
            "has_active_tb": False,
            "has_past_tb": False,
            "tb_diagnoses": [],
            "comorbidities": {"diabetes": False, "hiv": False, "copd": False},
            "confidence": 0.0,
        }

        for r in results:
            if r.get("is_negated"):
                continue
            category = r.get("category")
            if category == "active_tb":
                info["has_active_tb"] = True
                info["tb_diagnoses"].append(r.get("display", ""))
                info["confidence"] = max(info["confidence"], r.get("confidence", 0))
            elif category == "past_tb":
                info["has_past_tb"] = True
                info["tb_diagnoses"].append(r.get("display", ""))
            comorbidity = r.get("comorbidity")
            if comorbidity in ("diabetes", "hiv", "copd"):
                info["comorbidities"][comorbidity] = True

        return info

    def generate_tb_report_card_text(self, diagnosis_text: str) -> Optional[str]:
        """根据诊断文本生成传染病报告卡辅助文本

        参数：
            diagnosis_text: 诊断文本

        返回：
            str | None: 报告卡文本
        """
        info = self.extract_tb_info(diagnosis_text)
        if not info["has_active_tb"] and not info["has_past_tb"]:
            return None

        lines = ["=== 结核病传染病报告卡辅助信息 ==="]
        if info["has_active_tb"]:
            lines.append("传染病类型: 活动性结核病")
            lines.append("报告类别: 疑似/确诊结核病")
            for dx in info["tb_diagnoses"]:
                if "活动" in dx or "A15" in dx or "A16" in dx:
                    lines.append(f"诊断依据: {dx}")
        if info["has_past_tb"]:
            lines.append("备注: 既往结核病史")
        if info["comorbidities"]["diabetes"]:
            lines.append("合并症: 糖尿病 (影响治疗方案选择)")
        if info["comorbidities"]["hiv"]:
            lines.append("合并症: HIV/AIDS (需双抗治疗)")
        if info["comorbidities"]["copd"]:
            lines.append("合并症: 尘肺/矽肺 (职业暴露高风险)")

        lines.append("================================")
        return "\n".join(lines)

    # ==================== 内部方法 ====================

    def _split_diagnoses(self, text: str) -> List[str]:
        """拆分多诊断文本"""
        # 先尝试按编号拆分
        parts = re.split(r"(?:\d+[\.\)、]|①|②|③|④|⑤)", text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) > 1:
            return parts

        # 按分隔符拆分
        for sep in DIAGNOSIS_SEPARATORS:
            if sep in text:
                parts = text.split(sep)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) > 1:
                    return parts

        return [text]

    def _is_negated(self, text: str) -> bool:
        """判断诊断是否为否定（排除性诊断）"""
        if not text:
            return False
        text_lower = text.lower().strip()
        for prefix in NEGATION_PREFIXES:
            if text_lower.startswith(prefix.lower()):
                return True
        return False

    def _remove_negation_prefix(self, text: str) -> str:
        """移除否定前缀"""
        for prefix in NEGATION_PREFIXES:
            if text.lower().startswith(prefix.lower()):
                return text[len(prefix):]
        return text