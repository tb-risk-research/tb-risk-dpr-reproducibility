#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""疾控中心系统对接模块

对接路径：
1. 医疗机构侧：自动填报传染病报告卡（辅助公共卫生科审核，不直连大疫情网）
2. 区域平台侧：对接全民健康信息平台，获取治疗管理/密切接触者/流调数据回传

功能：
- TBCardFiller: 传染病报告卡自动填报
- CDCCallbackReceiver: 疾控回传数据接收（治疗/接触者/区域流行病数据）
- CDCSecurity: 数据加密/签名/脱敏
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import logging
import re
import ssl
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from .base import (
    mask_sensitive, hash_identifier, timestamp_to_datetime,
    calculate_age, normalize_gender,
)

LOGGER = logging.getLogger("tb_risk.health_interop.cdc")


# ============================================================================
# 传染病报告卡字段定义（国家《传染病报告卡》）
# ============================================================================

REPORT_CARD_REQUIRED_FIELDS = [
    # 患者基本信息
    "name", "gender", "age", "id_card", "phone",
    "current_address", "work_unit", "occupation",
    # 疾病信息
    "case_classification", "onset_date", "diagnosis_date",
    "diagnosis_basis", "pathogen_result",
    # 报告信息
    "report_unit", "report_doctor", "report_date",
]

CASE_CLASSIFICATIONS = {
    "suspected": "疑似病例",
    "clinical": "临床诊断病例",
    "confirmed": "确诊病例",
    "positive_detected": "病原携带者",
}

DIAGNOSIS_BASIS_OPTIONS = [
    "laboratory", "clinical", "epidemiological",
]


@dataclass
class InfectiousDiseaseCard:
    """传染病报告卡（结核）"""
    # 标识
    card_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str = ""
    is_repeat: bool = False          # 是否重报
    previous_card_id: str = ""       # 上次报告卡ID（复治/耐药关联）

    # 患者基本信息
    name: str = ""
    gender: str = ""                 # male/female
    gender_text: str = ""
    birth_date: str = ""
    age: Optional[int] = None
    id_card: str = ""                # 身份证号（加密存储/传输时脱敏）
    phone: str = ""
    current_address: str = ""       # 现住址
    permanent_address: str = ""     # 户籍地址
    work_unit: str = ""             # 工作单位
    occupation: str = ""
    nationality: str = "中国"
    ethnicity: str = ""
    education: str = ""
    marital_status: str = ""

    # 疾病信息
    disease_name: str = "肺结核"
    disease_code: str = "A15-A19"   # ICD-10
    case_classification: str = ""   # suspected/clinical/confirmed
    onset_date: str = ""            # 发病日期
    diagnosis_date: str = ""        # 诊断日期
    death_date: str = ""            # 死亡日期（如适用）
    diagnosis_basis: List[str] = field(default_factory=list)  # laboratory/clinical/epidemiological
    pathogen_result: str = ""       # 病原学检查结果
    drug_resistance: str = ""       # 耐药情况：敏感/单耐药/多耐药/耐多药/广泛耐药
    tb_type: str = ""              # 继发性/原发性/血行播散/结核性胸膜炎/其他
    treatment_category: str = ""    # 初治/复治

    # 报告信息
    report_unit: str = ""
    report_doctor: str = ""
    report_date: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))
    fill_date: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    status: str = "draft"           # draft/filled/submitted/accepted/rejected
    fill_source: str = "auto"       # auto/auto_with_manual/manual

    # 缺失字段（待医生补充）
    missing_fields: List[str] = field(default_factory=list)

    def validate(self) -> Tuple[bool, List[str]]:
        """校验必填字段，返回 (是否有效, 缺失字段列表)"""
        missing = []
        for f in REPORT_CARD_REQUIRED_FIELDS:
            val = getattr(self, f, None)
            if val is None or val == "" or val == []:
                missing.append(f)
        self.missing_fields = missing
        return len(missing) == 0, missing

    def to_dict(self, for_transmission: bool = False) -> Dict[str, Any]:
        """导出为字典，for_transmission=True时脱敏敏感字段"""
        d = {
            "card_id": self.card_id,
            "patient_id": self.patient_id,
            "name": mask_sensitive(self.name, keep_prefix=1, keep_suffix=0) if for_transmission else self.name,
            "gender": self.gender_text or self.gender,
            "age": self.age,
            "id_card": mask_sensitive(self.id_card, keep_prefix=3, keep_suffix=4) if for_transmission else self.id_card,
            "phone": mask_sensitive(self.phone, keep_prefix=3, keep_suffix=4) if for_transmission else self.phone,
            "current_address": self.current_address,
            "work_unit": self.work_unit,
            "occupation": self.occupation,
            "ethnicity": self.ethnicity,
            "case_classification": CASE_CLASSIFICATIONS.get(self.case_classification, self.case_classification),
            "onset_date": self.onset_date,
            "diagnosis_date": self.diagnosis_date,
            "diagnosis_basis": self.diagnosis_basis,
            "pathogen_result": self.pathogen_result,
            "drug_resistance": self.drug_resistance,
            "tb_type": self.tb_type,
            "treatment_category": self.treatment_category,
            "report_unit": self.report_unit,
            "report_doctor": mask_sensitive(self.report_doctor, keep_prefix=1, keep_suffix=0) if for_transmission else self.report_doctor,
            "report_date": self.report_date,
            "is_repeat": self.is_repeat,
            "missing_fields": self.missing_fields,
        }
        return d

    def to_cdc_xml(self) -> str:
        """导出为疾控标准XML格式（示例）"""
        ns = "http://www.chinacdc.cn/infectious/report"
        root = ET.Element("InfectiousDiseaseReport", xmlns=ns)
        ET.SubElement(root, "CardID").text = self.card_id
        ET.SubElement(root, "DiseaseCode").text = self.disease_code
        ET.SubElement(root, "DiseaseName").text = self.disease_name
        patient = ET.SubElement(root, "Patient")
        ET.SubElement(patient, "Name").text = mask_sensitive(self.name, keep_prefix=1, keep_suffix=0)
        ET.SubElement(patient, "Gender").text = self.gender_text
        if self.age is not None:
            ET.SubElement(patient, "Age").text = str(self.age)
        ET.SubElement(patient, "IDCard").text = mask_sensitive(self.id_card, keep_prefix=3, keep_suffix=4)
        diag = ET.SubElement(root, "Diagnosis")
        ET.SubElement(diag, "CaseClassification").text = CASE_CLASSIFICATIONS.get(
            self.case_classification, self.case_classification
        )
        ET.SubElement(diag, "OnsetDate").text = self.onset_date
        ET.SubElement(diag, "DiagnosisDate").text = self.diagnosis_date
        ET.SubElement(diag, "PathogenResult").text = self.pathogen_result
        ET.SubElement(diag, "DrugResistance").text = self.drug_resistance
        reporter = ET.SubElement(root, "Reporter")
        ET.SubElement(reporter, "ReportUnit").text = self.report_unit
        ET.SubElement(reporter, "ReportDoctor").text = mask_sensitive(self.report_doctor, keep_prefix=1, keep_suffix=0)
        ET.SubElement(reporter, "ReportDate").text = self.report_date
        return ET.tostring(root, encoding="unicode", xml_declaration=True)


# TODO: [GNN-CDC-001] 传染病报告卡与GNN接触网络双向联动未实现
#  - 发现确诊病例后，需自动提示追踪密切接触者
#  - 接触者筛查结果需回传更新GNN传播网络
#  - 需实现从GNN网络提取密切接触者列表、自动生成随访任务
#  - 需打通数据闭环：确诊→密接追踪→筛查→更新网络→风险重评估
class TBCardFiller:
    """传染病报告卡自动填报器

    从tb_risk内部数据（patient_info + lab_results + diagnoses + imaging）
    自动填充传染病报告卡字段。
    """

    def __init__(self):
        # 耐药关键词
        self._dr_keywords = {
            "sensitive": ["敏感", "sensitive", "无耐药", "药物敏感"],
            "mono": ["单耐药", "mono-resistant", "mono"],
            "poly": ["多耐药", "poly-resistant", "poly"],
            "mdr": ["耐多药", "MDR", "mdr", "multidrug-resistant", "multidrug"],
            "xdr": ["广泛耐药", "XDR", "xdr", "extensively"],
            "rifampicin_resistant": ["利福平耐药", "RR-TB", "RIF耐药", "rpoB突变"],
        }

    def fill_card(self, patient_info: Dict[str, Any],
                  lab_results: Optional[List[Dict]] = None,
                  diagnoses: Optional[List[Dict]] = None,
                  imaging_reports: Optional[List[Dict]] = None,
                  report_unit: str = "",
                  report_doctor: str = "") -> InfectiousDiseaseCard:
        """根据患者数据自动填卡"""
        card = InfectiousDiseaseCard()
        lab_results = lab_results or []
        diagnoses = diagnoses or []
        imaging_reports = imaging_reports or []

        # 患者基本信息
        card.patient_id = str(patient_info.get("patient_id",
                             patient_info.get("empi_id", "")))
        card.name = str(patient_info.get("name", ""))
        gender_raw = patient_info.get("gender", patient_info.get("gender_text", ""))
        card.gender = normalize_gender(gender_raw)
        card.gender_text = "男" if card.gender == "male" else "女" if card.gender == "female" else ""
        card.birth_date = str(patient_info.get("birth_date", ""))
        if patient_info.get("age") is not None:
            try:
                card.age = int(patient_info["age"])
            except (TypeError, ValueError):
                pass
        elif card.birth_date:
            card.age = calculate_age(card.birth_date)
        card.id_card = str(patient_info.get("id_card", ""))
        card.phone = str(patient_info.get("phone", ""))
        card.current_address = str(patient_info.get("address", ""))
        card.permanent_address = str(patient_info.get("permanent_address", ""))
        card.work_unit = str(patient_info.get("work_unit", patient_info.get("company", "")))
        card.occupation = str(patient_info.get("occupation", ""))
        card.ethnicity = str(patient_info.get("ethnicity", ""))

        # 报告信息
        card.report_unit = report_unit
        card.report_doctor = report_doctor

        # 诊断日期：取最早的诊断日期
        diag_dates = []
        has_tb_diag = False
        for d in diagnoses:
            code = str(d.get("code", ""))
            if code.startswith(("A15", "A16", "A17", "A18", "A19", "B90")):
                has_tb_diag = True
            dt = d.get("diagnosis_date") or d.get("onset_date") or d.get("recorded_date")
            if isinstance(dt, datetime.datetime):
                diag_dates.append(dt)
            elif dt:
                parsed = timestamp_to_datetime(dt)
                if parsed:
                    diag_dates.append(parsed)
        if diag_dates:
            card.diagnosis_date = min(diag_dates).strftime("%Y-%m-%d")

        # 发病日期：优先使用明确记录的症状出现时间
        # 注意：不使用"诊断前14天"这种不可靠的自动推断，避免医生误信
        # 如果没有明确发病日期，留空待医生补充，并加入missing_fields
        onset = patient_info.get("symptom_onset_date", "")
        if onset:
            card.onset_date = str(onset)
            card.fill_source = "auto"
        else:
            # 尝试从病历/主诉中提取最早症状时间（如果有）
            symptom_date = patient_info.get("earliest_symptom_date", "")
            if symptom_date:
                card.onset_date = str(symptom_date)
                card.fill_source = "auto_inferred"
            else:
                card.onset_date = ""
                card.fill_source = "auto"
                if "onset_date" not in card.missing_fields:
                    card.missing_fields.append("onset_date")

        # 病例分类判定：
        # - 有阳性病原学结果（涂片/培养/Xpert阳性）→ 确诊病例
        # - 影像典型+临床症状但无病原学 → 临床诊断
        # - 有疑似影像/症状 → 疑似
        pathogen_positive = False
        pathogen_results = []
        for lab in lab_results:
            name = str(lab.get("name", "")).lower()
            is_pos = lab.get("is_positive")
            cat = lab.get("category", "")
            if cat == "pathogen" or cat == "molecular" or any(
                    k in name for k in ("涂片", "培养", "xpert", "pcr", "smear", "culture")):
                if is_pos:
                    pathogen_positive = True
                if lab.get("value"):
                    pathogen_results.append(f"{lab.get('name','')}:{lab.get('value','')}")
            if cat == "molecular" and is_pos:
                # Xpert阳性可作为确诊
                pathogen_positive = True

        has_cavity = False
        has_tb_imaging = False
        for img in imaging_reports:
            nlp = img.get("nlp_features", {})
            if nlp.get("mentions_tb") or nlp.get("tb_predilection_site"):
                has_tb_imaging = True
            if nlp.get("has_cavity"):
                has_cavity = True

        # 既往结核病史
        history_tb = bool(patient_info.get("history_tb", False))

        if pathogen_positive:
            card.case_classification = "confirmed"
            card.diagnosis_basis = ["laboratory", "clinical"]
        elif has_tb_imaging or has_tb_diag:
            card.case_classification = "clinical"
            card.diagnosis_basis = ["clinical"]
        else:
            card.case_classification = "suspected"
            card.diagnosis_basis = ["clinical", "epidemiological"]

        card.pathogen_result = "; ".join(pathogen_results) if pathogen_results else (
            "病原学阳性" if pathogen_positive else "病原学阴性/待查")

        # 耐药判定
        dr_text = ""
        for lab in lab_results:
            v = str(lab.get("value", "")).lower()
            name = str(lab.get("name", "")).lower()
            if "耐药" in v or "rpo" in v or "mutation" in v or "mutant" in v:
                dr_text = v
                break
            if "xpert" in name and ("r" in str(lab.get("value", "")).lower() or "耐药" in str(lab.get("value", ""))):
                dr_text = "利福平耐药（Xpert检出）"
                break
        if dr_text:
            for dr_level, kws in self._dr_keywords.items():
                if any(kw in dr_text.lower() for kw in kws):
                    if dr_level == "rifampicin_resistant":
                        card.drug_resistance = "利福平耐药"
                    elif dr_level == "mdr":
                        card.drug_resistance = "耐多药"
                    elif dr_level == "xdr":
                        card.drug_resistance = "广泛耐药"
                    elif dr_level == "poly":
                        card.drug_resistance = "多耐药"
                    elif dr_level == "mono":
                        card.drug_resistance = "单耐药"
                    else:
                        card.drug_resistance = "敏感"
                    break
            if not card.drug_resistance:
                card.drug_resistance = "耐药（待进一步检测）"
        else:
            card.drug_resistance = "待检测/敏感"

        # 治疗分类：有既往结核病史则复治
        card.treatment_category = "复治" if history_tb else "初治"

        # 结核分型（简化判断：空洞=继发性，血行播散=miliary，影像+NLP）
        for img in imaging_reports:
            nlp = img.get("nlp_features", {})
            if nlp.get("is_miliary"):
                card.tb_type = "血行播散性肺结核"
                break
            if nlp.get("has_cavity"):
                card.tb_type = "继发性肺结核"
        if not card.tb_type and has_tb_imaging:
            card.tb_type = "继发性肺结核"  # 默认最常见

        card.status = "filled" if not card.validate()[1] else "draft"
        if card.missing_fields:
            card.fill_source = "auto"
        else:
            card.fill_source = "auto_complete"

        return card

    @staticmethod
    def check_duplicate(patient_info: Dict,
                        reported_cards: List[Dict]) -> Optional[Dict]:
        """查重：判断患者是否已报告过结核（避免重报）

        查重策略：身份证号精确匹配优先，其次姓名+年龄+地址模糊匹配
        """
        id_card = str(patient_info.get("id_card", "")).strip()
        name = str(patient_info.get("name", "")).strip()
        age = patient_info.get("age")
        if id_card and len(id_card) >= 15:
            for c in reported_cards:
                if str(c.get("id_card", "")).strip() == id_card:
                    return c
        # 姓名+年龄匹配
        if name:
            for c in reported_cards:
                if str(c.get("name", "")).strip() == name:
                    try:
                        c_age = int(c.get("age", -1))
                        p_age = int(age) if age is not None else -1
                        if p_age >= 0 and c_age >= 0 and abs(c_age - p_age) <= 2:
                            return c
                    except (TypeError, ValueError):
                        pass
        return None


class CDCCallbackParser:
    """疾控/区域平台回传数据解析"""
    
    # 否定修饰词（用于检测"未检出"、"无异常"等否定结构）
    NEGATION_WORDS = {"无", "未见", "未发现", "未检出", "未查到", "未找到", "排除", "阴性"}
    
    # 阳性关键词（临床检验报告常用表述）
    POSITIVE_KEYWORDS = {
        "positive", "confirmed", "tb_confirmed", "active_tb",
        "阳性", "确诊", "结核阳性", "涂阳", "培阳", "xpert阳性",
        "检出", "查到", "找到", "抗酸杆菌阳性", "afb阳性",
        "分枝杆菌阳性", "结核分枝杆菌阳性", "培养阳性", "抗酸染色阳性",
        "xpert检出", "tb检出", "抗酸杆菌+"
    }
    
    # 阴性关键词（临床检验报告常用表述）
    NEGATIVE_KEYWORDS = {
        "negative", "normal", "no_tb",
        "阴性", "正常", "未见异常", "无异常",
        "未检出", "未查到", "未找到", "抗酸杆菌阴性", "afb阴性",
        "未发现抗酸杆菌", "未见抗酸杆菌", "培养阴性", "抗酸染色阴性"
    }
    
    # 中性/待确认关键词（这些词不表示否定，只表示不确定）
    PENDING_KEYWORDS = {
        "pending", "suspected", "suspicious", "inconclusive",
        "待查", "待排", "疑似", "可疑", "建议复查", "随访",
        "待进一步检查", "待确认", "阴性待排", "可疑阳性", "弱阳性"
    }

    @staticmethod
    def classify_screening_result(text: str) -> str:
        """根据筛查结果文本精确分类，避免简单关键词匹配导致的误判。
        
        返回：
            "positive" - 阳性/确诊
            "negative" - 阴性/正常
            "pending" - 待确认/疑似/待查
        """
        if not text:
            return "pending"
        
        text_lower = text.lower().strip()
        
        # 1. 优先检测明确的待确认/可疑表述（优先于否定检测，因为这些词不应被当作否定）
        if any(kw in text_lower for kw in CDCCallbackParser.PENDING_KEYWORDS):
            # 但如果同时有明确的阳性词且没有否定修饰，还是算阳性
            has_positive = any(kw in text_lower for kw in CDCCallbackParser.POSITIVE_KEYWORDS)
            has_negation_before_positive = False
            # 简单位置检查：否定词是否出现在阳性词前面（如"未检出阳性"是阴性）
            for neg in CDCCallbackParser.NEGATION_WORDS:
                pos = text_lower.find(neg)
                if pos >= 0:
                    # 检查否定词后面是否跟着阳性相关词
                    after = text_lower[pos+len(neg):pos+len(neg)+10]
                    if any(pw in after for pw in ["阳性", "检出", "结核", "abnormal", "positive"]):
                        has_negation_before_positive = True
                        break
            if has_positive and not has_negation_before_positive:
                # 例如"可疑阳性"、"弱阳性"虽然有"可疑"但仍倾向阳性
                if any(kw in text_lower for kw in ["弱阳性", "可疑阳性"]):
                    return "positive"
                return "pending"
            return "pending"
        
        # 2. 检查否定修饰组合：否定词+异常/阳性词 → 阴性
        for neg in CDCCallbackParser.NEGATION_WORDS:
            neg_pos = text_lower.find(neg)
            if neg_pos >= 0:
                # 检查否定词后是否有异常/阳性相关词
                after_text = text_lower[neg_pos:neg_pos+15]
                if any(kw in after_text for kw in ["异常", "abnormal", "阳性", "positive",
                                                    "检出", "查到", "找到", "结核", "afb"]):
                    return "negative"
        
        # 3. 单独的否定词但没有后续阳性词（如直接说"阴性"）→ 阴性
        if any(kw in text_lower for kw in CDCCallbackParser.NEGATIVE_KEYWORDS):
            # 排除"弱阳性"等情况（已在PENDING中处理）
            if "弱阳性" not in text_lower and "可疑阳性" not in text_lower:
                return "negative"
        
        # 4. 检查明确阳性（排除否定后）
        if any(kw in text_lower for kw in CDCCallbackParser.POSITIVE_KEYWORDS):
            return "positive"
        
        # 5. 默认：无法判定 → 待确认
        return "pending"

    @staticmethod
    def parse_treatment_management(data: Dict[str, Any]) -> Dict[str, Any]:
        """解析治疗管理回传数据"""
        return {
            "patient_id": data.get("patient_id", ""),
            "treatment_plan": data.get("treatment_plan", ""),
            "treatment_start": timestamp_to_datetime(data.get("start_date")),
            "sputum_results": {
                "month2": data.get("month2_sputum"),
                "month5": data.get("month5_sputum"),
                "month6": data.get("month6_sputum"),
            },
            "outcome": data.get("outcome", ""),  # 治愈/完成疗程/死亡/失败/丢失/转出
            "outcome_date": timestamp_to_datetime(data.get("outcome_date")),
            "drug_resistance_screen": data.get("dr_screen", ""),
            "source": "cdc_treatment",
        }

    @staticmethod
    def parse_close_contacts(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析密切接触者回传数据"""
        contacts = []
        raw_list = data.get("contacts", data.get("close_contacts", []))
        for c in raw_list if isinstance(raw_list, list) else []:
            screening_text = str(c.get("screening_result", ""))
            result_class = CDCCallbackParser.classify_screening_result(screening_text)
            is_active = c.get("is_active_tb")
            # 如果未显式提供is_active_tb，根据精确分类推断
            if is_active is None:
                is_active = (result_class == "positive")
            contacts.append({
                "name": c.get("name", ""),
                "relationship": c.get("relationship", ""),
                "contact_type": c.get("contact_type", "cdc_contact"),
                "screening_result": screening_text,
                "screening_classification": result_class,
                "is_active_tb": bool(is_active),
                "screening_date": timestamp_to_datetime(c.get("screening_date")),
                "contact_id": c.get("contact_id", ""),
                "patient_id": c.get("patient_id") or data.get("source_patient_id", ""),
            })
        return contacts

    @staticmethod
    def parse_regional_epidemiology(data: Dict[str, Any]) -> Dict[str, Any]:
        """解析区域流行病学数据"""
        return {
            "district": data.get("district", ""),
            "incidence_rate_100k": data.get("incidence_per_100k", 0),
            "cluster_outbreaks": data.get("cluster_outbreaks", []),
            "regional_dr_rate": data.get("dr_rate", 0),
            "report_date": timestamp_to_datetime(data.get("report_date")),
        }


class CDCSecurity:
    """疾控数据交换安全工具：签名、加密、脱敏、防重放"""

    # 防重放时间窗口（秒），默认5分钟
    REPLAY_WINDOW_SECONDS = 300

    @staticmethod
    def sign(payload: Dict[str, Any], secret_key: str, timestamp: str = "") -> str:
        """生成HMAC-SHA256签名
        
        签名内容包含timestamp + body，防止截获后重放旧请求。
        如果提供timestamp，签名计算为 HMAC(timestamp + "." + body)；
        否则保持向后兼容，仅对body签名。
        """
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        if timestamp:
            signed_content = f"{timestamp}.{body}".encode("utf-8")
        else:
            signed_content = body.encode("utf-8")
        sig = hmac.new(secret_key.encode("utf-8"), signed_content,
                       hashlib.sha256).hexdigest()
        return sig

    @staticmethod
    def verify_signature(payload: Dict[str, Any], signature: str,
                         secret_key: str, timestamp: str = "") -> bool:
        """验证签名
        
        验证时同时尝试带timestamp和不带timestamp两种方式，保持向后兼容。
        """
        # 优先尝试带timestamp的新格式
        if timestamp:
            expected = CDCSecurity.sign(payload, secret_key, timestamp)
            if hmac.compare_digest(expected, signature):
                return True
        # 回退到旧格式（仅body），兼容未升级的发送方
        expected = CDCSecurity.sign(payload, secret_key)
        return hmac.compare_digest(expected, signature)
    
    @staticmethod
    def verify_timestamp(timestamp_str: str, 
                        window_seconds: int = None) -> bool:
        """验证请求时间戳是否在允许窗口内，防止重放攻击。
        
        参数：
            timestamp_str: Unix时间戳（秒或毫秒），支持字符串或数字
            window_seconds: 允许的时间偏差窗口，默认REPLAY_WINDOW_SECONDS(300秒/5分钟)
        
        返回：
            bool: 时间戳是否有效
        """
        import time
        if window_seconds is None:
            window_seconds = CDCSecurity.REPLAY_WINDOW_SECONDS
        
        if not timestamp_str:
            return False
        
        try:
            ts = float(timestamp_str)
            # 如果是毫秒级时间戳（13位），转换为秒
            if ts > 1e12:
                ts = ts / 1000.0
            now = time.time()
            return abs(now - ts) <= window_seconds
        except (ValueError, TypeError):
            return False

    @staticmethod
    def desensitize_for_transmission(data: Dict[str, Any]) -> Dict[str, Any]:
        """跨机构传输数据脱敏"""
        out = {}
        sensitive_fields = {"id_card", "phone", "name", "idcard", "mobile"}
        for k, v in data.items():
            if k in sensitive_fields and isinstance(v, str) and v:
                if k == "name":
                    out[k] = mask_sensitive(v, keep_prefix=1, keep_suffix=0)
                elif k in ("id_card", "idcard"):
                    out[k] = mask_sensitive(v, keep_prefix=3, keep_suffix=4)
                else:
                    out[k] = mask_sensitive(v, keep_prefix=3, keep_suffix=4)
            elif isinstance(v, dict):
                out[k] = CDCSecurity.desensitize_for_transmission(v)
            elif isinstance(v, list):
                out[k] = [CDCSecurity.desensitize_for_transmission(i)
                          if isinstance(i, dict) else i for i in v]
            else:
                out[k] = v
        return out


# ============================================================================
# CDC Webhook 回传接收服务器
# ============================================================================

class CDCWebhookServer:
    """轻量级CDC回传Webhook服务器

    使用Python内置http.server实现，监听指定端口接收疾控推送的JSON数据，
    验证HMAC-SHA256签名后通过CDCCallbackParser解析，触发回调通知。
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080,
                 secret_key: str = "", data_dir: str = "cdc_webhook",
                 ssl_certfile: str = "", ssl_keyfile: str = "",
                 require_timestamp: bool = False):
        self.host = host
        self.port = port
        self.secret_key = secret_key
        self.data_dir = data_dir
        # HTTPS/TLS配置
        self._ssl_certfile = ssl_certfile
        self._ssl_keyfile = ssl_keyfile
        self._https_enabled = bool(ssl_certfile and ssl_keyfile)
        # 是否要求时间戳防重放
        self._require_timestamp = require_timestamp
        # 如果启用HTTPS但未指定端口，使用标准HTTPS端口8443
        if self._https_enabled and port == 8080:
            self.port = 8443
        
        self._server = None
        self._thread = None
        self._running = False
        self._message_count = 0
        self._callbacks = {
            "treatment": [],
            "contacts": [],
            "epidemiology": [],
            "raw": [],
        }
        self._log_callback = None
        # 防重放：已处理的请求ID缓存（带过期时间）
        self._processed_nonces: Dict[str, float] = {}
        self._nonce_lock = None

    def set_log_callback(self, callback):
        """设置日志回调函数 callback(level: str, message: str)"""
        self._log_callback = callback

    def on(self, event_type: str, callback):
        """注册事件回调
        event_type: 'treatment'|'contacts'|'epidemiology'|'raw'
        callback: 接收解析后数据的函数
        """
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)

    def _log(self, level: str, message: str):
        if self._log_callback:
            try:
                self._log_callback(level, message)
            except Exception as e:
                LOGGER.debug("CDC日志回调失败: %s", e)
        LOGGER.log(
            logging.ERROR if level == 'error' else
            logging.WARNING if level == 'warning' else
            logging.INFO,
            "CDC Webhook: %s", message
        )

    def start(self) -> bool:
        """启动Webhook服务器（在后台线程运行）"""
        import os
        import json
        import threading
        import time
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from urllib.parse import urlparse

        # 初始化防重放锁
        if self._nonce_lock is None:
            self._nonce_lock = threading.Lock()
        
        # 清理过期nonce
        self._cleanup_expired_nonces()

        # 确保数据目录存在
        try:
            os.makedirs(self.data_dir, exist_ok=True)
        except Exception as e:
            self._log('error', f"无法创建数据目录: {e}")
            return False

        server_self = self

        class WebhookHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # 禁用默认日志

            def _send_json(self, status_code: int, data: dict):
                self.send_response(status_code)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

            def do_POST(self):
                parsed_path = urlparse(self.path)
                path = parsed_path.path

                # 只接受 /cdc/callback 路径
                if path not in ('/cdc/callback', '/webhook/cdc', '/callback'):
                    self._send_json(404, {"error": "Not found"})
                    return

                # 读取请求体
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length == 0 or content_length > 10 * 1024 * 1024:
                    self._send_json(400, {"error": "Invalid content length"})
                    return

                try:
                    body = self.rfile.read(content_length)
                    payload = json.loads(body.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    server_self._log('warning', f"无效JSON请求: {e}")
                    self._send_json(400, {"error": "Invalid JSON"})
                    return

                # 提取签名、时间戳、nonce
                signature = self.headers.get('X-CDC-Signature', '')
                timestamp = ""
                nonce = ""
                
                # 时间戳防重放验证
                if server_self._require_timestamp or server_self.secret_key:
                    timestamp = (self.headers.get('X-CDC-Timestamp', '') or 
                                 payload.get('timestamp', '') or 
                                 payload.get('ts', ''))
                    if timestamp:
                        if not CDCSecurity.verify_timestamp(timestamp):
                            server_self._log('warning', f"时间戳验证失败或过期: {timestamp}")
                            self._send_json(401, {"error": "Timestamp expired or invalid"})
                            return
                    elif server_self._require_timestamp:
                        server_self._log('warning', "缺少时间戳头 X-CDC-Timestamp")
                        self._send_json(401, {"error": "Missing timestamp"})
                        return
                
                # 验证签名（timestamp参与签名计算）
                if server_self.secret_key:
                    if not signature:
                        server_self._log('warning', "缺少签名头 X-CDC-Signature")
                        self._send_json(401, {"error": "Missing signature"})
                        return
                    if not CDCSecurity.verify_signature(payload, signature, 
                                                       server_self.secret_key, timestamp):
                        server_self._log('warning', "签名验证失败")
                        self._send_json(403, {"error": "Invalid signature"})
                        return
                
                # Nonce/请求ID防重放（在签名验证通过后执行）
                if server_self._require_timestamp or server_self.secret_key:
                    nonce = (self.headers.get('X-CDC-Nonce', '') or 
                             payload.get('nonce', '') or 
                             payload.get('request_id', '') or
                             signature)  # 用签名作为兜底nonce
                    if nonce:
                        with server_self._nonce_lock:
                            now = time.time()
                            if nonce in server_self._processed_nonces:
                                server_self._log('warning', f"重复请求(nonce={nonce[:16]}...)，可能为重放攻击")
                                self._send_json(409, {"error": "Duplicate request"})
                                return
                            server_self._processed_nonces[nonce] = now
                            # 清理过期nonce
                            server_self._cleanup_expired_nonces(now)

                # 确定数据类型
                data_type = payload.get('type', payload.get('data_type', 'raw'))
                server_self._message_count += 1

                # 原始数据回调
                for cb in server_self._callbacks['raw']:
                    try:
                        cb(payload)
                    except Exception as e:
                        server_self._log('error', f"raw回调异常: {e}")

                # 解析并分发
                parsed_data = None
                if data_type == 'treatment' or 'treatment_plan' in payload or 'outcome' in payload:
                    parsed_data = CDCCallbackParser.parse_treatment_management(payload)
                    event_type = 'treatment'
                    server_self._log('success', f"收到治疗管理回传 - 患者: {parsed_data.get('patient_id', 'N/A')}, 结局: {parsed_data.get('outcome', 'N/A')}")
                elif data_type == 'contacts' or 'contacts' in payload or 'close_contacts' in payload:
                    parsed_data = CDCCallbackParser.parse_close_contacts(payload)
                    event_type = 'contacts'
                    server_self._log('success', f"收到密切接触者回传 - 共 {len(parsed_data)} 人")
                elif data_type == 'epidemiology' or 'incidence_per_100k' in payload:
                    parsed_data = CDCCallbackParser.parse_regional_epidemiology(payload)
                    event_type = 'epidemiology'
                    server_self._log('success', f"收到区域流行病学数据 - 地区: {parsed_data.get('district', 'N/A')}")
                else:
                    event_type = 'raw'
                    server_self._log('info', f"收到未知类型回传数据，已作为raw事件分发")

                # 保存原始数据到文件
                try:
                    import time
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"{ts}_{server_self._message_count}.json"
                    fpath = os.path.join(server_self.data_dir, fname)
                    with open(fpath, 'w', encoding='utf-8') as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    server_self._log('warning', f"保存Webhook数据失败: {e}")

                # 触发对应回调
                if parsed_data is not None and event_type in server_self._callbacks:
                    for cb in server_self._callbacks[event_type]:
                        try:
                            cb(parsed_data)
                        except Exception as e:
                            server_self._log('error', f"{event_type}回调异常: {e}")

                self._send_json(200, {"status": "ok", "received": True})

        try:
            self._server = HTTPServer((self.host, self.port), WebhookHandler)
            
            # 如果启用HTTPS，包装socket
            if self._https_enabled:
                try:
                    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    context.load_cert_chain(certfile=self._ssl_certfile, keyfile=self._ssl_keyfile)
                    context.minimum_version = ssl.TLSVersion.TLSv1_2
                    self._server.socket = context.wrap_socket(
                        self._server.socket, server_side=True
                    )
                    self._log('info', "CDC Webhook已启用HTTPS/TLS加密")
                except Exception as ssl_err:
                    self._log('error', f"SSL证书加载失败: {ssl_err}")
                    return False
            
            proto = "HTTPS" if self._https_enabled else "HTTP"
            self._log('info', f"CDC {proto} Webhook服务器正在启动 {self.host}:{self.port}...")

            def run():
                try:
                    server_self._running = True
                    server_self._log('success', f"CDC {proto} Webhook服务已启动，监听 {server_self.host}:{server_self.port}")
                    server_self._server.serve_forever()
                except Exception as e:
                    server_self._log('error', f"Webhook服务异常: {e}")
                finally:
                    server_self._running = False

            self._thread = threading.Thread(target=run, daemon=True)
            self._thread.start()

            # 短暂等待确认服务启动
            import time
            time.sleep(0.3)
            return self._running

        except OSError as e:
            self._log('error', f"端口{self.port}被占用或无法绑定: {e}")
            return False
        except Exception as e:
            self._log('error', f"启动Webhook服务失败: {e}")
            return False

    def _cleanup_expired_nonces(self, now: float = None):
        """清理过期的nonce缓存，防止内存无限增长。"""
        import time
        if now is None:
            now = time.time()
        if not self._nonce_lock:
            return
        with self._nonce_lock:
            expired = [n for n, ts in self._processed_nonces.items() 
                      if now - ts > CDCSecurity.REPLAY_WINDOW_SECONDS * 2]
            for n in expired:
                del self._processed_nonces[n]

    def stop(self):
        """停止Webhook服务器"""
        self._running = False
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
                self._log('info', "CDC Webhook服务已停止")
            except Exception as e:
                self._log('warning', f"停止Webhook服务时异常: {e}")
        self._server = None
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running and self._server is not None

    @property
    def message_count(self) -> int:
        return self._message_count
