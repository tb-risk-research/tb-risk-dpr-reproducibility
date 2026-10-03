#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FHIR 资源解析器

将 Patient / Condition / Observation / MedicationStatement / DiagnosticReport 等
FHIR R4资源解析为tb_risk内部统一的患者信息、检验结果、诊断、用药、影像报告结构。
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

from ..base import (
    AdapterResult, LOINC_TB_PANELS, ICD10_TB_CODES,
    timestamp_to_datetime, calculate_age, normalize_gender,
    convert_unit, ABNORMAL_FLAGS,
)

LOGGER = logging.getLogger("tb_risk.health_interop.fhir.parser")


class FHIRResourceParser:
    """FHIR资源解析器"""

    def __init__(self):
        # LOINC编码 → 检验项目元数据（预构建快速查找表）
        self._loinc_map: Dict[str, Dict[str, Any]] = {}
        for category, items in LOINC_TB_PANELS.items():
            for item in items:
                self._loinc_map[item["loinc"]] = {**item, "category": category}

    # ---- Patient ----

    def parse_patient(self, resource: Dict, result: AdapterResult):
        """解析Patient资源，填充result.patient_info"""
        if not resource or resource.get("resourceType") != "Patient":
            return
        pi = result.patient_info

        # 标识符
        for ident in resource.get("identifier", []) or []:
            system = ident.get("system", "")
            value = ident.get("value", "")
            if "mrn" in system.lower() or "medical" in system.lower():
                pi.setdefault("medical_record_no", value)
            elif "id-card" in system.lower() or "idcard" in system.lower() or "身份证" in system:
                pi.setdefault("id_card", value)
            elif "empi" in system.lower():
                result.patient_id = value or result.patient_id

        # 姓名
        names = resource.get("name", []) or []
        if names:
            name_entry = names[0]
            given = " ".join(name_entry.get("given", []) or [])
            family = name_entry.get("family", "") or ""
            text = name_entry.get("text", "")
            pi["name"] = text or f"{family}{given}".strip() or "未知"

        # 性别
        gender = resource.get("gender")
        pi["gender"] = normalize_gender(gender)
        pi["gender_text"] = "男" if pi["gender"] == "male" else "女" if pi["gender"] == "female" else "未知"

        # 出生日期 → 年龄
        birth_date = resource.get("birthDate")
        if birth_date:
            age = calculate_age(birth_date)
            if age is not None:
                pi["age"] = age
            pi["birth_date"] = str(birth_date)

        # 联系电话
        telecom = resource.get("telecom", []) or []
        for t in telecom:
            if t.get("system") == "phone" and t.get("value"):
                pi["phone"] = t["value"]
                break

        # 地址
        addresses = resource.get("address", []) or []
        if addresses:
            addr = addresses[0]
            parts = [addr.get("line", [""])[0] if addr.get("line") else "",
                     addr.get("city", ""), addr.get("state", ""),
                     addr.get("district", ""), addr.get("postalCode", "")]
            pi["address"] = " ".join(p for p in parts if p)
            if addr.get("district"):
                pi["district"] = addr["district"]

        # 扩展字段（民族、职业、籍贯等国标字段）
        for ext in resource.get("extension", []) or []:
            url = ext.get("url", "")
            if not url:
                continue
            value = ext.get("valueCodeableConcept") or ext.get("valueString") or ext.get("valueCoding")
            if isinstance(value, dict) and value.get("coding"):
                code = value["coding"][0].get("code", "")
                display = value["coding"][0].get("display", "")
                if "ethnicity" in url.lower() or "nation" in url.lower():
                    pi["ethnicity"] = display or code
                elif "occupation" in url.lower():
                    pi["occupation"] = display or code
                elif "birth-place" in url.lower() or "native-place" in url.lower() or "籍贯" in url:
                    pi["native_place"] = display or code
            elif isinstance(value, str):
                if "ethnicity" in url.lower():
                    pi["ethnicity"] = value
                elif "occupation" in url.lower():
                    pi["occupation"] = value

        # 婚姻状况
        marital = resource.get("maritalStatus")
        if marital and marital.get("coding"):
            marital_code = marital["coding"][0].get("code", "")
            pi["marital_status"] = marital_code

    # ---- Condition ----

    def parse_condition(self, resource: Dict) -> Optional[Dict[str, Any]]:
        """解析Condition资源，返回诊断/合并症记录"""
        if not resource or resource.get("resourceType") != "Condition":
            return None
        codings = self._get_codings(resource.get("code", {}))
        if not codings:
            return None

        # 取第一个有效编码
        code_system = codings[0].get("system", "")
        code = codings[0].get("code", "")
        display = codings[0].get("display", "")

        onset = timestamp_to_datetime(
            resource.get("onsetDateTime") or resource.get("onsetPeriod", {}).get("start"))
        recorded = timestamp_to_datetime(resource.get("recordedDate"))
        abatement = timestamp_to_datetime(
            resource.get("abatementDateTime") or resource.get("abatementPeriod", {}).get("end"))

        # 临床状态
        clinical_status = ""
        cs = resource.get("clinicalStatus")
        if cs and cs.get("coding"):
            clinical_status = cs["coding"][0].get("code", "")

        is_tb = False
        is_comorbidity = False
        comorbidity_type = ""

        # 判断是否为结核相关诊断
        icd10_prefix = code[:3] if code and "icd-10" in code_system.lower() else ""
        if icd10_prefix in ICD10_TB_CODES or code in ICD10_TB_CODES:
            is_tb = True
            comorbidity_type = "tb"
        elif code and any(code.startswith(prefix) for prefix in ("E10", "E11", "E12", "E13", "E14")):
            is_comorbidity = True
            comorbidity_type = "diabetes"
        elif code and any(code.startswith(prefix) for prefix in ("B20", "B21", "B22", "B23", "B24", "Z21")):
            is_comorbidity = True
            comorbidity_type = "hiv"
        elif code and any(code.startswith(prefix) for prefix in ("J62", "J65")):
            is_comorbidity = True
            comorbidity_type = "silicosis"

        return {
            "code_system": code_system,
            "code": code,
            "display": display,
            "is_tb": is_tb,
            "comorbidity_type": comorbidity_type,
            "onset_date": onset.isoformat() if onset else None,
            "recorded_date": recorded.isoformat() if recorded else None,
            "abatement_date": abatement.isoformat() if abatement else None,
            "clinical_status": clinical_status,
            "is_active": clinical_status in ("active", "relapse", "recurrence", ""),
            "raw": resource,
        }

    # ---- Observation ----

    def parse_observation(self, resource: Dict, term_mapper=None) -> Optional[Dict[str, Any]]:
        """解析Observation资源，返回检验结果记录"""
        if not resource or resource.get("resourceType") != "Observation":
            return None
        codings = self._get_codings(resource.get("code", {}))
        if not codings:
            return None

        # 查找是否为结核相关LOINC编码
        matched_loinc = None
        for c in codings:
            sys = c.get("system", "")
            code = c.get("code", "")
            if "loinc" in sys.lower() and code in self._loinc_map:
                matched_loinc = self._loinc_map[code]
                break
        if not matched_loinc:
            # 通过术语映射器尝试转码
            if term_mapper:
                for c in codings:
                    mapped = term_mapper.map_code(c.get("code", ""), c.get("system", ""), "LOINC")
                    if mapped and mapped in self._loinc_map:
                        matched_loinc = self._loinc_map[mapped]
                        break
            if not matched_loinc:
                return None

        effective = timestamp_to_datetime(
            resource.get("effectiveDateTime") or resource.get("effectiveInstant")
            or resource.get("effectivePeriod", {}).get("start"))
        issued = timestamp_to_datetime(resource.get("issued"))

        # 结果值
        value = None
        unit = ""
        value_type = "unknown"
        if "valueQuantity" in resource:
            vq = resource["valueQuantity"]
            value = vq.get("value")
            unit = vq.get("unit") or vq.get("code", "")
            value_type = "quantity"
            # 单位换算到tb_risk内部标准单位
            target_unit = matched_loinc.get("unit", "")
            if target_unit and unit and unit != target_unit and value is not None:
                converted = convert_unit(float(value), unit, target_unit)
                if converted is not None:
                    value = converted
                    unit = target_unit
        elif "valueCodeableConcept" in resource:
            vcc = resource["valueCodeableConcept"]
            vc_codings = self._get_codings(vcc)
            if vc_codings:
                value = vc_codings[0].get("display") or vc_codings[0].get("code")
                value_type = "codeable"
            elif vcc.get("text"):
                value = vcc["text"]
                value_type = "codeable"
        elif "valueString" in resource:
            value = resource["valueString"]
            value_type = "string"
        elif "valueBoolean" in resource:
            value = resource["valueBoolean"]
            value_type = "boolean"

        # 异常标志
        interpretation = ""
        interp_list = resource.get("interpretation", []) or []
        if interp_list:
            for interp in interp_list:
                ic = self._get_codings(interp)
                if ic:
                    code = ic[0].get("code", "")
                    interpretation = ABNORMAL_FLAGS.get(code, code)
                    break

        # 参考范围
        ref_range = None
        rr_list = resource.get("referenceRange", []) or []
        if rr_list:
            rr = rr_list[0]
            ref_low = rr.get("low", {}).get("value") if rr.get("low") else None
            ref_high = rr.get("high", {}).get("value") if rr.get("high") else None
            ref_text = rr.get("text", "")
            ref_range = {"low": ref_low, "high": ref_high, "text": ref_text}

        # 标本类型
        specimen_type = matched_loinc.get("sample", "")
        specimen_ref = resource.get("specimen", {})
        specimen_display = ""
        if isinstance(specimen_ref, dict) and specimen_ref.get("display"):
            specimen_display = specimen_ref["display"]

        # 阴阳性判定（对于定性检验）
        is_positive = None
        if value_type == "codeable" and isinstance(value, str):
            v_low = value.lower()
            if any(k in v_low for k in ("阳性", "positive", "检出", "detected", "+", "reactive")):
                is_positive = True
            elif any(k in v_low for k in ("阴性", "negative", "未检出", "not detected", "-", "non-reactive", "nonreactive")):
                is_positive = False

        return {
            "loinc": matched_loinc["loinc"],
            "name": matched_loinc["name"],
            "short_name": matched_loinc["short"],
            "category": matched_loinc["category"],
            "value": value,
            "value_type": value_type,
            "unit": unit,
            "interpretation": interpretation,
            "is_abnormal": interpretation in ("low", "high", "critical_low", "critical_high", "abnormal", "critical_abnormal"),
            "is_positive": is_positive,
            "reference_range": ref_range,
            "specimen_type": specimen_type or specimen_display,
            "effective_time": effective.isoformat() if effective else None,
            "issued_time": issued.isoformat() if issued else None,
            "status": resource.get("status", ""),
        }

    # ---- MedicationStatement ----

    def parse_medication(self, resource: Dict) -> Optional[Dict[str, Any]]:
        """解析MedicationStatement资源，返回用药记录"""
        if not resource or resource.get("resourceType") != "MedicationStatement":
            return None
        med_cc = resource.get("medicationCodeableConcept", {})
        codings = self._get_codings(med_cc)
        if not codings and med_cc.get("text"):
            name = med_cc["text"]
            code_system = ""
            code = ""
        elif codings:
            code_system = codings[0].get("system", "")
            code = codings[0].get("code", "")
            name = codings[0].get("display", "")
        else:
            return None

        effective = resource.get("effectivePeriod", {})
        start = timestamp_to_datetime(effective.get("start"))
        end = timestamp_to_datetime(effective.get("end"))

        # 判断是否为抗结核药（简单关键词匹配，后续可接入RXNORM/ATC映射）
        is_anti_tb = False
        anti_tb_keywords = (
            "异烟肼", "利福平", "吡嗪酰胺", "乙胺丁醇", "链霉素",
            "isoniazid", "rifamp", "pyrazinamide", "ethambutol", "streptomycin",
            "inh", "rif", "pza", "emb", "sm",
            "利奈唑胺", "贝达喹啉", "莫西沙星", "左氧氟沙星",
        )
        name_lower = name.lower()
        if any(k.lower() in name_lower for k in anti_tb_keywords):
            is_anti_tb = True

        return {
            "code_system": code_system,
            "code": code,
            "name": name,
            "is_anti_tb": is_anti_tb,
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
            "status": resource.get("status", ""),
            "dosage": resource.get("dosage", []),
        }

    # ---- DiagnosticReport ----

    def parse_diagnostic_report(self, resource: Dict) -> Optional[Dict[str, Any]]:
        """解析DiagnosticReport资源，返回影像报告记录"""
        if not resource or resource.get("resourceType") != "DiagnosticReport":
            return None
        codings = self._get_codings(resource.get("code", {}))
        category_codings = []
        for cat in resource.get("category", []) or []:
            category_codings.extend(self._get_codings(cat))

        effective = timestamp_to_datetime(
            resource.get("effectiveDateTime") or resource.get("effectivePeriod", {}).get("start"))
        issued = timestamp_to_datetime(resource.get("issued"))

        # 结论文本（NLP待处理）
        conclusion = resource.get("conclusion", "") or ""
        if not conclusion:
            # 从presentedForm中取文本
            for pf in resource.get("presentedForm", []) or []:
                data = pf.get("data", "")
                if data:
                    import base64
                    try:
                        conclusion = base64.b64decode(data).decode("utf-8", errors="replace")
                    except Exception:
                        conclusion = data
                    break

        return {
            "report_id": resource.get("id", ""),
            "status": resource.get("status", ""),
            "category": [c.get("code", "") for c in category_codings],
            "conclusion": conclusion,
            "effective_time": effective.isoformat() if effective else None,
            "issued_time": issued.isoformat() if issued else None,
            "imaging_refs": [
                ref.get("reference", "") for ref in (resource.get("imagingStudy") or [])
            ] if isinstance(resource.get("imagingStudy"), list) else [],
            "nlp_processed": False,
            "nlp_features": {},
        }

    # ---- 工具方法 ----

    @staticmethod
    def _get_codings(codeable_concept: Dict) -> List[Dict]:
        """从CodeableConcept中提取coding列表（容错处理）"""
        if not codeable_concept or not isinstance(codeable_concept, dict):
            return []
        codings = codeable_concept.get("coding", []) or []
        return [c for c in codings if isinstance(c, dict)]
