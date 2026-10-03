#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API 请求/响应模型（Pydantic）。

仅当 FastAPI 已安装时导入（FastAPI 依赖 Pydantic）。请求体保持宽松
（``Dict``/``List[Dict]``），与 ``RiskAssessmentService.assess()`` 的
dict 契约一致，便于第三方按既有 CLI 配置格式直接调用。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContactInput(BaseModel):
    """单个接触者输入（宽松 dict，兼容既有 potential_patients 形状）。"""

    name: Optional[str] = Field(None, description="联系人姓名")
    relationship: Optional[str] = Field(None, description="关系")
    contact_distance: Optional[str] = Field(None, description="接触距离 close/medium/far")
    ventilation: Optional[str] = Field(None, description="通风条件 poor/average/good")
    exposure_setting: Optional[str] = Field(None, description="暴露场景")
    freq_density: Optional[float] = Field(None, description="每周接触频次")
    single_duration: Optional[float] = Field(None, description="单次接触时长(分钟)")
    time_span: Optional[float] = Field(None, description="持续周期(周)")
    age: Optional[int] = Field(None, description="年龄")
    # 透传任意扩展字段
    extra: Dict[str, Any] = Field(default_factory=dict, description="扩展字段")

    def to_dict(self) -> Dict[str, Any]:
        """序列化为普通 dict（合并 extra）。"""
        d = {
            "name": self.name,
            "relationship": self.relationship,
            "contact_distance": self.contact_distance,
            "ventilation": self.ventilation,
            "exposure_setting": self.exposure_setting,
            "freq_density": self.freq_density,
            "single_duration": self.single_duration,
            "time_span": self.time_span,
            "age": self.age,
        }
        d = {k: v for k, v in d.items() if v is not None}
        d.update(self.extra)
        return d


class AssessmentRequest(BaseModel):
    """单患者风险评估请求。"""

    patient_info: Dict[str, Any] = Field(
        default_factory=dict, description="患者信息（basic_info/FCI/SNC 等）")
    family_members: List[Dict[str, Any]] = Field(
        default_factory=list, description="家庭成员列表")
    social_contacts: List[Dict[str, Any]] = Field(
        default_factory=list, description="社会接触者列表")
    use_ml: bool = Field(False, description="是否启用 ML 预测")
    use_seir: bool = Field(False, description="是否启用 SEIR 模拟")


class BatchRecord(BaseModel):
    """批量评估中的单条记录。"""

    patient_info: Dict[str, Any] = Field(default_factory=dict)
    family_members: List[Dict[str, Any]] = Field(default_factory=list)
    social_contacts: List[Dict[str, Any]] = Field(default_factory=list)
    ref: Optional[str] = Field(None, description="业务引用标识（返回时透传）")


class BatchAssessmentRequest(BaseModel):
    """批量评估请求（异步执行）。"""

    records: List[BatchRecord] = Field(..., description="待评估记录列表")
    use_ml: bool = Field(False)
    use_seir: bool = Field(False)


class TrainRequest(BaseModel):
    """模型训练触发请求。"""

    n_samples: int = Field(2000, ge=10, le=100000, description="训练样本数")
    enable_hyperopt: bool = Field(False, description="是否启用超参数优化")
    save_path: Optional[str] = Field(None, description="模型保存路径")


class ReportRequest(BaseModel):
    """报告生成请求。"""

    result: Dict[str, Any] = Field(default_factory=dict, description="评估结果")
    format: str = Field("html", description="报告格式 html/json/text")
    ai_enhanced: bool = Field(False, description="是否启用 AI 摘要增强")


__all__ = [
    "ContactInput",
    "AssessmentRequest",
    "BatchRecord",
    "BatchAssessmentRequest",
    "TrainRequest",
    "ReportRequest",
]