#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HL7 v2.x 消息解析器

支持解析：
- MSH段：消息头（类型、发送方、接收方、时间、控制ID、版本）
- PID段：患者标识（ID、姓名、性别、出生日期、身份证、联系方式、地址）
- ORC段：通用医嘱
- OBR段：检验申请
- OBX段：检验结果（支持多段拼接长文本/报告）
- TXA/TQ1等段
支持消息分段（Continuity消息）处理，长报告自动拼接。
"""

from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..base import (
    timestamp_to_datetime, normalize_gender, calculate_age,
    parse_hl7_datetime,
)

LOGGER = logging.getLogger("tb_risk.health_interop.hl7v2.parser")

# MLLP帧标记
MLLP_START_BLOCK = b"\x0b"
MLLP_END_BLOCK = b"\x1c\x0d"


@dataclass
class HL7Message:
    """解析后的HL7消息"""
    raw: str = ""
    message_type: str = ""       # ADT^A01, ORU^R01等
    trigger_event: str = ""
    sending_app: str = ""
    sending_facility: str = ""
    receiving_app: str = ""
    receiving_facility: str = ""
    message_time: Optional[datetime.datetime] = None
    message_control_id: str = ""
    version: str = ""
    segments: Dict[str, List[List[str]]] = field(default_factory=dict)
    # 便捷字段（解析后）
    patient_info: Dict[str, Any] = field(default_factory=dict)
    lab_results: List[Dict[str, Any]] = field(default_factory=list)
    clinical_notes: List[Dict[str, Any]] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)

    def get_segment(self, name: str, index: int = 0) -> Optional[List[str]]:
        """获取指定段（按索引）"""
        segs = self.segments.get(name, [])
        if 0 <= index < len(segs):
            return segs[index]
        return None

    def get_fields(self, seg_name: str, field_idx: int, seg_idx: int = 0) -> str:
        """获取指定段的指定字段值（1-based索引，返回第repeat）"""
        seg = self.get_segment(seg_name, seg_idx)
        if seg and field_idx < len(seg):
            return seg[field_idx]
        return ""

    def get_repeats(self, seg_name: str, field_idx: int, seg_idx: int = 0) -> List[str]:
        """获取字段所有repeat"""
        val = self.get_fields(seg_name, field_idx, seg_idx)
        if not val:
            return []
        return val.split("~")

    def get_component(self, seg_name: str, field_idx: int, comp_idx: int = 0,
                      seg_idx: int = 0) -> str:
        """获取字段的component（field_idx从0开始，comp_idx从0开始）"""
        val = self.get_fields(seg_name, field_idx, seg_idx)
        if not val:
            return ""
        # 只取第一个repeat的component
        first_repeat = val.split("~")[0]
        comps = first_repeat.split("^")
        if comp_idx < len(comps):
            return comps[comp_idx]
        return ""


class HL7MessageParser:
    """HL7 v2.x消息解析器"""

    def __init__(self):
        self._default_sep = {
            "field": "|",
            "component": "^",
            "repeat": "~",
            "escape": "\\",
            "subcomponent": "&",
        }

    def parse(self, raw_message: str) -> HL7Message:
        """解析HL7消息字符串为HL7Message对象"""
        msg = HL7Message(raw=raw_message)

        # 去除MLLP帧标记（如果有）
        text = raw_message.strip()
        if text.startswith("\x0b"):
            text = text[1:]
        if text.endswith("\x1c\x0d"):
            text = text[:-2]
        text = text.strip()

        if not text:
            msg.parse_errors.append("空消息")
            return msg

        # MSH段决定分隔符
        lines = text.split("\r") if "\r" in text else text.split("\n")
        lines = [l for l in lines if l.strip()]
        if not lines:
            msg.parse_errors.append("无有效段")
            return msg

        msh_line = None
        for line in lines:
            if line.startswith("MSH"):
                msh_line = line
                break
        if not msh_line:
            msg.parse_errors.append("缺少MSH段")
            return msg

        # 解析分隔符（MSH字段1是field分隔符之后跟着其他分隔符）
        field_sep = msh_line[3] if len(msh_line) > 3 else "|"
        if len(msh_line) > 4:
            comp_sep = msh_line[4]
            repeat_sep = msh_line[5] if len(msh_line) > 5 else "~"
            escape_sep = msh_line[6] if len(msh_line) > 6 else "\\"
            sub_sep = msh_line[7] if len(msh_line) > 7 else "&"
        else:
            comp_sep, repeat_sep, escape_sep, sub_sep = "^", "~", "\\", "&"
        separators = {
            "field": field_sep,
            "component": comp_sep,
            "repeat": repeat_sep,
            "escape": escape_sep,
            "subcomponent": sub_sep,
        }

        # 按段解析
        current_segments: Dict[str, List[List[str]]] = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(field_sep)
            seg_name = parts[0] if parts else ""
            if not seg_name:
                continue
            if seg_name not in current_segments:
                current_segments[seg_name] = []
            current_segments[seg_name].append(parts)

        msg.segments = current_segments

        # 解析MSH头
        self._parse_msh(msg, separators)
        # 解析PID
        if "PID" in current_segments:
            self._parse_pid(msg, separators)
        # 解析ORU消息的OBX段（检验结果）
        if msg.message_type.startswith("ORU"):
            self._parse_oru_obx(msg, separators)
        # 解析MDM消息的文档内容
        if msg.message_type.startswith("MDM"):
            self._parse_mdm_document(msg, separators)

        return msg

    def _parse_msh(self, msg: HL7Message, sep: Dict[str, str]):
        r"""解析MSH段
        注意：MSH特殊，parts[0]="MSH", parts[1]是编码字符(^~\&)，
        因为MSH-1是字段分隔符本身(|)，split后不占单独元素。
        所以 parts[n] 对应 MSH-(n+1)。
        """
        msh = msg.get_segment("MSH")
        if not msh:
            return
        # MSH-3 = SendingApp (index 2), MSH-4 = SendingFacility (index 3)
        msg.sending_app = msh[2] if len(msh) > 2 else ""
        msg.sending_facility = msh[3] if len(msh) > 3 else ""
        msg.receiving_app = msh[4] if len(msh) > 4 else ""
        msg.receiving_facility = msh[5] if len(msh) > 5 else ""

        # MSH-7 = DateTime (index 6)
        msg_time_raw = msh[6] if len(msh) > 6 else ""
        msg.message_time = parse_hl7_datetime(msg_time_raw)

        # MSH-9 = Message Type (index 8): MSG^TRIGGER^STRUCTURE
        msg_type_field = msh[8] if len(msh) > 8 else ""
        type_comps = msg_type_field.split(sep["component"])
        msg_code = type_comps[0] if len(type_comps) > 0 else ""
        trigger = type_comps[1] if len(type_comps) > 1 else ""
        msg.message_type = f"{msg_code}^{trigger}" if msg_code and trigger else msg_type_field
        msg.trigger_event = trigger

        # MSH-10 = Message Control ID (index 9)
        msg.message_control_id = msh[9] if len(msh) > 9 else ""
        # MSH-12 = Version ID (index 11)
        msg.version = msh[11] if len(msh) > 11 else ""

    def _parse_pid(self, msg: HL7Message, sep: Dict[str, str]):
        """解析PID患者标识段"""
        pid = msg.get_segment("PID")
        if not pid:
            return
        pi = msg.patient_info

        # PID-3 患者ID列表（多种类型：MRN/PID/身份证号/EMPI）
        id_fields = self._get_repeats_safe(pid, 3, sep)
        for idf in id_fields:
            comps = idf.split(sep["component"])
            id_value = comps[0] if len(comps) > 0 else ""
            id_type = comps[4] if len(comps) > 4 else ""
            assigning = comps[3] if len(comps) > 3 else ""
            if not id_value:
                continue
            type_upper = id_type.upper()
            if "MR" in type_upper or "MRN" in type_upper:
                pi.setdefault("medical_record_no", id_value)
            elif "NI" in type_upper or "PT" in type_upper or "PI" in type_upper:
                pi.setdefault("patient_id", id_value)
            elif "IDCARD" in type_upper or "ID" == type_upper or "身份证" in idf:
                pi.setdefault("id_card", id_value)
            elif "EMPI" in type_upper:
                pi.setdefault("empi_id", id_value)
                pi.setdefault("patient_id", id_value)
            else:
                pi.setdefault("patient_id", id_value)

        # PID-5 患者姓名（Family^Given^Middle）
        name_field = pid[5] if len(pid) > 5 else ""
        if name_field:
            name_comps = name_field.split(sep["component"])
            family = name_comps[0] if len(name_comps) > 0 else ""
            given = name_comps[1] if len(name_comps) > 1 else ""
            pi["name"] = (family + given) if family or given else name_field.replace(sep["component"], "")

        # PID-7 出生日期，PID-8 性别
        birth_raw = pid[7] if len(pid) > 7 else ""
        if birth_raw:
            bd = parse_hl7_datetime(birth_raw)
            if bd:
                pi["birth_date"] = bd.strftime("%Y-%m-%d")
                age = calculate_age(bd)
                if age is not None:
                    pi["age"] = age
        gender_raw = pid[8] if len(pid) > 8 else ""
        if gender_raw:
            pi["gender"] = normalize_gender(gender_raw)
            pi["gender_text"] = "男" if pi["gender"] == "male" else "女" if pi["gender"] == "female" else "未知"

        # PID-10 种族/民族, PID-11 地址
        race_field = pid[10] if len(pid) > 10 else ""
        if race_field:
            race_comps = race_field.split(sep["component"])
            pi["ethnicity"] = race_comps[0] if race_comps else race_field

        addr_field = pid[11] if len(pid) > 11 else ""
        if addr_field:
            addr_comps = addr_field.split(sep["component"])
            addr_parts = [c for c in (addr_comps + [""] * 8)[:9] if c]
            pi["address"] = " ".join(addr_parts)
            if len(addr_comps) >= 4:
                pi["district"] = addr_comps[3]
            if len(addr_comps) >= 3:
                pi["city"] = addr_comps[2]

        # PID-13 电话
        phone_repeats = self._get_repeats_safe(pid, 13, sep)
        for ph in phone_repeats:
            if ph:
                phone_comps = ph.split(sep["component"])
                phone_num = phone_comps[0] if phone_comps else ph
                if phone_num:
                    pi["phone"] = phone_num
                    break

    def _parse_oru_obx(self, msg: HL7Message, sep: Dict[str, str]):
        """解析ORU^R01消息中的OBX段（检验结果）"""
        obx_list = msg.segments.get("OBX", [])
        obr_list = msg.segments.get("OBR", [])
        # 先取OBR的申请信息（检验套餐、申请时间等）
        obr_info: Dict[str, Any] = {}
        if obr_list:
            obr = obr_list[0]
            obr_info["order_id"] = self._comp(obr, 2, 0, sep)
            obr_info["battery_code"] = self._comp(obr, 4, 0, sep)
            obr_info["battery_name"] = self._comp(obr, 4, 1, sep)
            obr_info["order_time"] = parse_hl7_datetime(
                obr[6] if len(obr) > 6 else "")
            obr_info["specimen_time"] = parse_hl7_datetime(
                obr[7] if len(obr) > 7 else "")

        pending_long_text: List[str] = []
        pending_meta: Dict[str, Any] = {}

        for obx in obx_list:
            if len(obx) < 7:
                continue
            try:
                set_id = obx[1] if len(obx) > 1 else ""
                value_type = obx[2] if len(obx) > 2 else ""  # NM/ST/CWE/CE/TX/FT/ED
                # OBX-3 检验项目代码
                obs_id = obx[3] if len(obx) > 3 else ""
                obs_comps = obs_id.split(sep["component"])
                obs_code = obs_comps[0] if obs_comps else ""
                obs_name = obs_comps[1] if len(obs_comps) > 1 else ""
                obs_coding = obs_comps[2] if len(obs_comps) > 2 else ""  # 编码系统（LN=LOINC）

                # OBX-5 结果值（可能跨多个OBX段）
                obs_value = obx[5] if len(obx) > 5 else ""
                # OBX-6 单位
                units_field = obx[6] if len(obx) > 6 else ""
                units_comps = units_field.split(sep["component"])
                unit = units_comps[0] if units_comps else ""

                # OBX-7 参考范围
                ref_range = obx[7] if len(obx) > 7 else ""
                # OBX-8 异常标志
                abnormal_flag = obx[8] if len(obx) > 8 else ""
                # OBX-11 结果状态（F=Final, P=Preliminary, C=Corrected）
                result_status = obx[11] if len(obx) > 11 else ""
                # OBX-14 观察时间
                obs_time_raw = obx[14] if len(obx) > 14 else ""
                obs_time = parse_hl7_datetime(obs_time_raw)
                # OBX-17 标本类型
                specimen_field = obx[17] if len(obx) > 17 else ""
                specimen = specimen_field.split(sep["component"])[0] if specimen_field else ""

                # 处理长文本分段（TX/FT类型，多个OBX段同set_id续接）
                if value_type in ("TX", "FT") and pending_long_text:
                    # 如果上一段在累积，先判断是否同set_id续接
                    if pending_meta.get("set_id") == set_id and pending_meta.get("value_type") in ("TX", "FT"):
                        pending_long_text.append(obs_value)
                        continue
                    else:
                        # 上个长文本结束了
                        self._finalize_long_text(msg, pending_long_text, pending_meta, sep)
                        pending_long_text = []
                        pending_meta = {}

                if value_type in ("TX", "FT", "ED"):
                    # 开始/继续长文本
                    pending_long_text.append(obs_value)
                    pending_meta = {
                        "set_id": set_id,
                        "value_type": value_type,
                        "obs_code": obs_code,
                        "obs_name": obs_name,
                        "obs_coding": obs_coding,
                        "unit": unit,
                        "abnormal_flag": abnormal_flag,
                        "result_status": result_status,
                        "obs_time": obs_time,
                        "specimen": specimen,
                        "obr": obr_info,
                    }
                    continue

                # 普通结果字段
                lab = {
                    "loinc": obs_code if "LN" in obs_coding.upper() else "",
                    "local_code": obs_code,
                    "name": obs_name or obs_code,
                    "value": obs_value,
                    "value_type": value_type,
                    "unit": unit,
                    "reference_range": ref_range,
                    "interpretation": abnormal_flag,
                    "is_abnormal": abnormal_flag in ("H", "L", "HH", "LL", "A", "AA"),
                    "is_positive": None,
                    "result_status": result_status,
                    "effective_time": obs_time.isoformat() if obs_time else None,
                    "specimen_type": specimen,
                    "order_id": obr_info.get("order_id", ""),
                }
                # 阴阳性判定
                v_low = obs_value.lower() if isinstance(obs_value, str) else ""
                if value_type in ("CE", "CWE", "ID", "ST"):
                    if any(k in v_low for k in ("阳性", "positive", "检出", "+", "reactive")):
                        lab["is_positive"] = True
                    elif any(k in v_low for k in ("阴性", "negative", "未检出", "-", "non-reactive")):
                        lab["is_positive"] = False
                msg.lab_results.append(lab)

            except Exception as e:
                msg.parse_errors.append(f"OBX段解析失败: {e}")
                LOGGER.debug("OBX段解析异常: %s", e, exc_info=True)

        # 处理最后一段长文本
        if pending_long_text:
            self._finalize_long_text(msg, pending_long_text, pending_meta, sep)

    def _finalize_long_text(self, msg: HL7Message, parts: List[str],
                            meta: Dict[str, Any], sep: Dict[str, str]):
        """将多段长文本拼接为完整文档/影像报告"""
        full_text = "\n".join(p for p in parts if p)
        if not full_text.strip():
            return
        # 判断是检验注释还是临床文档/影像报告
        name = meta.get("obs_name", "").lower()
        if any(k in name for k in ("impression", "report", "印象", "诊断", "报告", "conclusion")):
            msg.clinical_notes.append({
                "note_type": "radiology" if "放射" in name or "chest" in name or "影像" in name else "document",
                "title": meta.get("obs_name", ""),
                "content": full_text,
                "effective_time": meta["obs_time"].isoformat() if meta.get("obs_time") else None,
                "order_id": meta.get("obr", {}).get("order_id", ""),
                "nlp_pending": True,
            })
        else:
            msg.lab_results.append({
                "loinc": meta.get("obs_code", ""),
                "local_code": meta.get("obs_code", ""),
                "name": meta.get("obs_name", "报告"),
                "value": full_text,
                "value_type": "TX",
                "unit": meta.get("unit", ""),
                "interpretation": meta.get("abnormal_flag", ""),
                "result_status": meta.get("result_status", ""),
                "effective_time": meta["obs_time"].isoformat() if meta.get("obs_time") else None,
                "specimen_type": meta.get("specimen", ""),
                "order_id": meta.get("obr", {}).get("order_id", ""),
            })

    def _parse_mdm_document(self, msg: HL7Message, sep: Dict[str, str]):
        """解析MDM^T02病历文档消息（OBX或TX段内容）"""
        obx_list = msg.segments.get("OBX", [])
        txa_list = msg.segments.get("TXA", [])
        doc_title = ""
        doc_type = ""
        doc_time = None
        if txa_list:
            txa = txa_list[0]
            doc_type = self._comp(txa, 2, 0, sep)
            doc_title = self._comp(txa, 17, 0, sep) or self._comp(txa, 12, 0, sep)
            doc_time_raw = txa[4] if len(txa) > 4 else ""
            doc_time = parse_hl7_datetime(doc_time_raw) or msg.message_time

        # OBX段拼接
        doc_parts = []
        for obx in obx_list:
            if len(obx) > 5:
                v = obx[5]
                if v:
                    doc_parts.append(v)
        content = "\n".join(doc_parts)

        if content or doc_title:
            # 判断文档类型
            note_type = "clinical_note"
            title_lower = (doc_title or "").lower()
            if any(k in title_lower for k in ("放射", "影像", "x线", "ct", "mr", "胸片", "chest")):
                note_type = "radiology"
            elif any(k in title_lower for k in ("出院", "入院", "病历", "discharge", "admission")):
                note_type = "discharge" if "出院" in title_lower or "discharge" in title_lower else "admission"

            msg.clinical_notes.append({
                "note_type": note_type,
                "title": doc_title or doc_type or "病历文档",
                "content": content,
                "effective_time": doc_time.isoformat() if doc_time else None,
                "nlp_pending": True,
                "message_control_id": msg.message_control_id,
            })

    # ---- 工具方法 ----

    @staticmethod
    def _get_repeats_safe(seg: List[str], idx: int, sep: Dict[str, str]) -> List[str]:
        if not seg or idx >= len(seg):
            return []
        val = seg[idx]
        if not val:
            return []
        return val.split(sep["repeat"])

    @staticmethod
    def _comp(seg: List[str], field_idx: int, comp_idx: int = 0,
              sep: Optional[Dict[str, str]] = None) -> str:
        if not seg or field_idx >= len(seg):
            return ""
        val = seg[field_idx]
        if not val:
            return ""
        s = sep or {"component": "^", "repeat": "~"}
        first_repeat = val.split(s["repeat"])[0]
        comps = first_repeat.split(s["component"])
        return comps[comp_idx] if comp_idx < len(comps) else ""
