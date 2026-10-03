#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CDS Hooks（Clinical Decision Support Hooks）预警卡构建。

纯 Python（无 FastAPI 依赖）。将 tb_risk 评估结果转换为 CDS Hooks 标准
``Card`` 结构，使预警可直接嵌入医生电子病历（EMR）工作流。

CDS Hooks 规范（https://cds-hooks.org/）关键概念：
- *Hook*：EMR 中的事件点，如 ``patient-view``（医生打开患者页面时触发）、
  ``encounter-discharge``、``order-select`` 等。
- *Service*：一个可调用的决策支持服务，通过 ``GET /cds-services`` 发现。
- *Card*：服务返回的卡片，含 ``summary``/``detail``/``indicator``
  （info/warning/critical）/``suggestions``/``links``。

本模块构建 ``patient-view`` 挂钩的 TB 风险预警卡：当医生在 EMR 中打开
患者页面时，系统后台评估结核病风险并返回预警卡。风险等级与阈值对齐
``validation.threshold_spec`` / ``constants.DISEASE_PROB_*``。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("tb_risk.api.cds_hooks")

# CDS Hooks 服务发现标识
SERVICE_ID = "tb-risk-assessment"
SERVICE_HOOK = "patient-view"
SERVICE_TITLE = "结核病风险评估"
SERVICE_DESCRIPTION = (
    "基于 SEIR 传播动力学与机器学习模型，在医生打开患者页面时自动评估"
    "结核病风险，并给出筛查/转诊建议。"
)

# 风险等级 → CDS Hooks indicator 映射
_RISK_INDICATOR = {
    "低": "info",
    "中": "warning",
    "高": "critical",
    "较高": "critical",
    "极高": "critical",
    "低风险": "info",
    "中风险": "warning",
    "高风险": "critical",
    "极高风险": "critical",
}

# CDS Hooks 服务注册表（供 GET /cds-services 发现）
CDS_SERVICES = [
    {
        "hook": SERVICE_HOOK,
        "title": SERVICE_TITLE,
        "description": SERVICE_DESCRIPTION,
        "id": SERVICE_ID,
        "prefetch": {
            "patient": "Patient/{{context.patientId}}",
        },
    }
]


def indicator_for_risk(overall_risk: Optional[str]) -> str:
    """将风险等级映射为 CDS Hooks indicator（info/warning/critical）。"""
    if not overall_risk:
        return "info"
    return _RISK_INDICATOR.get(str(overall_risk).strip(), "info")


def _mask_name(name: Optional[str]) -> Optional[str]:
    """姓名脱敏：保留姓，其余用 * 代替。"""
    if not name:
        return name
    s = str(name)
    if len(s) <= 1:
        return s
    if len(s) == 2:
        return s[0] + "*"
    return s[0] + "*" * (len(s) - 2) + s[-1]


def summarize_result(assessment_result: Optional[Dict[str, Any]]) -> str:
    """从评估结果生成卡片摘要文本。"""
    if not assessment_result:
        return "风险评估执行失败，请检查输入数据。"
    overall = assessment_result.get("overall_risk", "未知")
    prob = assessment_result.get("base_infection_probability")
    prob_text = f"，感染概率约 {float(prob):.1f}%" if prob is not None else ""
    return f"结核病风险评估：{overall}风险{prob_text}"


def build_cdhooks_card(
    assessment_result: Optional[Dict[str, Any]],
    patient_id: Optional[str] = None,
    service_id: str = SERVICE_ID,
    mask_sensitive: bool = True,
) -> Dict[str, Any]:
    """将单个评估结果构建为一张 CDS Hooks Card。

    参数：
        assessment_result: ``RiskAssessmentService.assess()`` 返回的 dict
        patient_id: 患者 ID（用于卡片链路）
        service_id: 服务标识
        mask_sensitive: 是否对接触者姓名脱敏

    返回：
        dict: CDS Hooks Card
    """
    indicator = indicator_for_risk(
        (assessment_result or {}).get("overall_risk"))
    summary = summarize_result(assessment_result)
    detail = _build_detail(assessment_result)

    card: Dict[str, Any] = {
        "uuid": uuid.uuid4().hex,
        "summary": summary,
        "detail": detail,
        "indicator": indicator,
        "source": {
            "label": SERVICE_TITLE,
        },
        "service_id": service_id,  # 顶层透传，便于第三方快速识别
        "extension": {
            "service_id": service_id,
            "hook": SERVICE_HOOK,
        },
    }

    if patient_id:
        card["links"] = [
            {
                "label": "查看完整评估报告",
                "url": f"/reports?patient_id={patient_id}",
                "type": "smart",
            }
        ]

    # 高风险建议
    suggestions = _build_suggestions(assessment_result)
    if suggestions:
        card["suggestions"] = suggestions

    # 脱敏接触者名单（可选）
    if not mask_sensitive:
        card["extension"]["top_contacts"] = _top_contacts(assessment_result)
    else:
        card["extension"]["top_contacts"] = [
            {
                "rank": i + 1,
                "name": _mask_name(c.get("name")),
                "priority": c.get("priority"),
                "disease_probability": c.get("disease_probability"),
            }
            for i, c in enumerate(_top_contacts(assessment_result))
        ]

    return card


def build_cdhooks_response(
    assessment_result: Optional[Dict[str, Any]],
    patient_id: Optional[str] = None,
    service_id: str = SERVICE_ID,
    mask_sensitive: bool = True,
) -> Dict[str, Any]:
    """构建 CDS Hooks ``$apply`` 响应（``{cards: [...]}``）。

    当评估成功且风险等级 >= 中风险时返回预警卡；低风险时返回 info 卡。
    """
    cards = [build_cdhooks_card(
        assessment_result, patient_id, service_id, mask_sensitive)]
    return {"cards": cards}


def _build_detail(result: Optional[Dict[str, Any]]) -> str:
    """生成卡片详情文本（含摘要统计与建议）。"""
    if not result:
        return "缺少评估数据。"
    suggestion = result.get("overall_suggestion") or "建议结合临床资料进一步评估。"
    summary = result.get("summary") or {}
    total = summary.get("total_contacts", 0)
    high = summary.get("high_risk_count", 0)
    medium = summary.get("medium_risk_count", 0)
    lines = [
        suggestion,
        f"接触者：共 {total} 人（高风险 {high}，中风险 {medium}）。",
    ]
    return "\n".join(lines)


def _build_suggestions(result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """基于风险等级生成建议动作（suggestions）。"""
    if not result:
        return []
    overall = str(result.get("overall_risk", "低"))
    medium_codes = ("中", "中风险")
    high_codes = ("高", "较高", "高风险", "极高", "极高风险")
    if overall in high_codes:
        return [
            {
                "label": "建议立即进行结核病筛查",
                "actions": [
                    {
                        "type": "create",
                        "description": "胸部X线 + 痰涂片/分子检测（GeneXpert）",
                    }
                ],
            }
        ]
    if overall in medium_codes:
        return [
            {
                "label": "建议进行接触者筛查",
                "actions": [
                    {
                        "type": "create",
                        "description": "PPD/IGRA 结核菌素筛查",
                    }
                ],
            }
        ]
    return []


def _top_contacts(result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """提取高风险接触者（按疾病概率降序，取前 5）。"""
    if not result:
        return []
    pp = result.get("potential_patients") or {}
    contacts = list(pp.get("family", [])) + list(pp.get("social", []))
    high = [c for c in contacts if c.get("priority") in ("极高", "高", "较高")]
    high.sort(key=lambda c: c.get("disease_probability", 0), reverse=True)
    return high[:5]


__all__ = [
    "SERVICE_ID", "SERVICE_HOOK", "SERVICE_TITLE", "CDS_SERVICES",
    "indicator_for_risk", "build_cdhooks_card", "build_cdhooks_response",
]