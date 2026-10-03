#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HL7 FHIR R4 接口模块

提供：
- FHIRClient: FHIR R4 REST API 客户端（按需查询+增量拉取+OAuth2认证）
- FHIRSubscription: FHIR Subscription 订阅通知处理器（Webhook模式）
- FHIRResourceParser: FHIR资源解析器，将Bundle/Patient/Observation/Condition/
  DiagnosticReport/MedicationStatement等资源转换为tb_risk特征字段
- FHIRTermMapper: 术语绑定与转码（LOINC/ICD-10/SNOMED ↔ tb_risk内部编码）
"""

from .client import FHIRClient
from .parser import FHIRResourceParser
from .subscription import FHIRSubscriptionHandler
from .terminology import FHIRTermMapper

__all__ = [
    'FHIRClient',
    'FHIRResourceParser',
    'FHIRSubscriptionHandler',
    'FHIRTermMapper',
]
