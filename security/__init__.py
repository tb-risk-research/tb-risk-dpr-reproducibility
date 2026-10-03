#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据安全与合规模块。

对外导出核心 API：
- 角色：``Role`` / ``role_label`` / ``ROLE_LABELS``
- 脱敏：``apply_masking`` / ``mask_name`` / ``mask_id_card`` / ``mask_phone`` / ``mask_address``
- 配置：``DataSecurityConfig``
- 审计：``SecurityAuditor``
- 导出守卫：``ExportGuard``
- 聚合管理器：``DataSecurityManager``
- 合规说明：``compliance_summary``
"""

from .data_security import (
    Role,
    ROLE_LABELS,
    role_label,
    ROLE_FIELD_VISIBILITY,
    apply_masking,
    mask_name,
    mask_id_card,
    mask_phone,
    mask_address,
    mask_value,
    DataSecurityConfig,
    SecurityAuditor,
    ExportGuard,
    DataSecurityManager,
    compliance_summary,
)

__all__ = [
    "Role",
    "ROLE_LABELS",
    "role_label",
    "ROLE_FIELD_VISIBILITY",
    "apply_masking",
    "mask_name",
    "mask_id_card",
    "mask_phone",
    "mask_address",
    "mask_value",
    "DataSecurityConfig",
    "SecurityAuditor",
    "ExportGuard",
    "DataSecurityManager",
    "compliance_summary",
]
