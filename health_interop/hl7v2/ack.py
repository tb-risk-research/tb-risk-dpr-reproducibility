#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HL7 ACK 消息构造器

根据处理结果构造AA（接受）、AR（拒绝）、AE（错误）三种ACK消息。
"""

from __future__ import annotations

import datetime
from typing import Optional


class HL7AckBuilder:
    """HL7 v2 ACK消息构造器"""

    # 应答码
    AA = "AA"  # Application Accept
    AR = "AR"  # Application Reject
    AE = "AE"  # Application Error

    @staticmethod
    def build(received_message, ack_code: str = "AA",
              error_message: str = "",
              sending_app: str = "TB_RISK",
              sending_facility: str = "TB_RISK_FACILITY") -> str:
        """构造ACK消息

        参数：
            received_message: 收到的 HL7Message 对象或原始消息字符串
            ack_code: AA / AR / AE
            error_message: 错误说明（AR/AE时使用）
            sending_app: 我方应用名
            sending_facility: 我方机构名
        """
        # 从接收消息中提取MSH字段用于构造ACK
        if hasattr(received_message, "segments"):
            msg = received_message
            msh = msg.get_segment("MSH") if hasattr(msg, "get_segment") else None
            if msh and len(msh) > 12:
                recv_app = msh[3] if len(msh) > 3 else ""
                recv_facility = msh[4] if len(msh) > 4 else ""
                recv_msg_control_id = msh[10] if len(msh) > 10 else ""
                version = msh[12] if len(msh) > 12 else "2.5"
                sep_field = msh[1] if len(msh) > 1 else "|"
                # MSH[2] 是 ^~\\&
                if len(msh) > 2:
                    other_seps = msh[2]
                else:
                    other_seps = "^~\\&"
            else:
                recv_app = recv_facility = recv_msg_control_id = ""
                version = "2.5"
                sep_field = "|"
                other_seps = "^~\\&"
        else:
            # 原始字符串，简单解析MSH
            raw = str(received_message)
            sep_field = "|"
            other_seps = "^~\\&"
            if raw.startswith("MSH") and len(raw) > 8:
                sep_field = raw[3]
                other_seps = raw[4:8] if len(raw) >= 8 else "^~\\&"
            parts = raw.split("\r")[0].split(sep_field) if "\r" in raw else raw.split(sep_field)
            recv_app = parts[3] if len(parts) > 3 else ""
            recv_facility = parts[4] if len(parts) > 4 else ""
            recv_msg_control_id = parts[10] if len(parts) > 10 else ""
            version = parts[12] if len(parts) > 12 else "2.5"

        comp = other_seps[0] if len(other_seps) > 0 else "^"
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        new_control_id = f"TB{now}"

        # 清理错误消息（不能包含段分隔符\r等）
        error_msg = error_message.replace("\r", " ").replace("\n", " ").strip()
        if len(error_msg) > 120:
            error_msg = error_msg[:120]

        # 构造MSH段
        msh_seg = (
            f"MSH{sep_field}{other_seps}{sep_field}"
            f"{sending_app}{sep_field}{sending_facility}{sep_field}"
            f"{recv_app}{sep_field}{recv_facility}{sep_field}"
            f"{now}{sep_field}{sep_field}ACK{comp}{recv_msg_control_id.split(comp)[0] if recv_msg_control_id else ''}{sep_field}"
            f"P{sep_field}{new_control_id}{sep_field}{sep_field}{version}"
        )
        # MSA段：MSA|AA|控制ID|文本
        msa_text = f"{error_msg}" if ack_code in ("AR", "AE") else ""
        msa_seg = f"MSA{sep_field}{ack_code}{sep_field}{recv_msg_control_id}{sep_field}{msa_text}"
        # ERR段（仅AR/AE）
        segments = [msh_seg, msa_seg]
        if ack_code in ("AR", "AE") and error_msg:
            err_seg = f"ERR{sep_field}{comp}{comp}{comp}{error_msg}"
            segments.append(err_seg)

        return "\r".join(segments) + "\r"

    @classmethod
    def accept(cls, msg) -> str:
        """构造AA接受ACK"""
        return cls.build(msg, cls.AA)

    @classmethod
    def reject(cls, msg, reason: str) -> str:
        """构造AR拒绝ACK"""
        return cls.build(msg, cls.AR, error_message=reason)

    @classmethod
    def error(cls, msg, reason: str) -> str:
        """构造AE错误ACK"""
        return cls.build(msg, cls.AE, error_message=reason)
