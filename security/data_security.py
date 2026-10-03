#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据安全与合规核心模块

依据《中华人民共和国个人信息保护法》及卫生健康行业规范，为 tb_risk 提供：
- 角色权限控制：普通用户 / 评估员 / 管理员，不同角色可见字段不同。
- 强化脱敏：姓名打码、身份证部分隐藏、地址脱敏等，统一作用于导出流程。
- 本地优先存储：默认数据仅存本机，不联网上传。
- 关键操作审计：评估 / 导出 / 导入 写入审计日志（哈希链可追溯）。

设计要点：
- 底层导出函数（export_utils.*）保持纯净、向后兼容；
  脱敏在"应用层"统一入口（ExportGuard）执行，确保所有导出路径走同一套策略。
- 复用 health_interop 的 mask_sensitive 与 AuditLogger，不重复造轮子。
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

LOGGER = logging.getLogger("tb_risk.security")

# 配置目录与文件（跨平台，与 gui/app_config.py 一致）
SECURITY_DIR = os.path.join(os.path.expanduser('~'), '.tb_risk')
SECURITY_CONFIG_FILE = os.path.join(SECURITY_DIR, 'security.json')
SECURITY_AUDIT_DB = os.path.join(SECURITY_DIR, 'security_audit.db')

try:  # 复用 health_interop 的通用脱敏工具
    from ..health_interop.base import mask_sensitive as _mask_sensitive
except Exception:  # pragma: no cover - 降级为内部实现
    def _mask_sensitive(value, mask_char="*", keep_prefix=1, keep_suffix=1):
        if value is None:
            return ""
        s = str(value)
        if len(s) <= keep_prefix + keep_suffix:
            return mask_char * len(s)
        return s[:keep_prefix] + mask_char * (len(s) - keep_prefix - keep_suffix) + s[-keep_suffix:]


# ============================================================================
# 角色定义
# ============================================================================

class Role:
    """系统角色常量。"""
    NORMAL = "normal"      # 普通用户
    ASSESSOR = "assessor"  # 评估员
    ADMIN = "admin"        # 管理员

    ALL = (NORMAL, ASSESSOR, ADMIN)


ROLE_LABELS = {
    Role.NORMAL: "普通用户",
    Role.ASSESSOR: "评估员",
    Role.ADMIN: "管理员",
}


def role_label(role: str) -> str:
    """角色中文标签（未知角色回退为原值）。"""
    return ROLE_LABELS.get(role, role)


# ============================================================================
# 敏感字段策略
# ============================================================================

# 字段类别 -> 脱敏类型
_FIELD_KIND = {
    # 姓名类 -> 打码（保留首字）
    "name": "name", "member_name": "name", "contact_name": "name",
    "patient_name": "name", "report_doctor": "name",
    # 证件号 -> 部分隐藏（保留前3后4）
    "id_card": "id_card", "id_no": "id_card", "identity_card": "id_card",
    # 手机号 -> 部分隐藏（保留前3后4）
    "phone": "phone", "phone_number": "phone", "mobile": "phone",
    # 地址类 -> 脱敏（保留区/街道级，隐藏详细门牌）
    "address": "address", "current_address": "address",
    "permanent_address": "address", "home_address": "address",
    "work_unit": "address",
}

# 角色 -> {"hidden": 完全不可见字段集合, "masked": 脱敏字段集合}
ROLE_FIELD_VISIBILITY = {
    Role.NORMAL: {
        "hidden": {
            "id_card", "id_no", "identity_card",
            "phone", "phone_number", "mobile",
            "address", "current_address", "permanent_address",
            "home_address", "work_unit", "report_doctor",
        },
        "masked": {"name", "member_name", "contact_name", "patient_name"},
    },
    Role.ASSESSOR: {
        "hidden": {"id_card", "id_no", "identity_card"},
        "masked": {
            "name", "member_name", "contact_name", "patient_name", "report_doctor",
            "phone", "phone_number", "mobile",
            "address", "current_address", "permanent_address",
            "home_address", "work_unit",
        },
    },
    Role.ADMIN: {
        "hidden": set(),
        "masked": set(),
    },
}


def mask_name(value: Any) -> str:
    """姓名打码：保留首字，其余以 * 代替（如 张三 -> 张*）。"""
    s = str(value or "").strip()
    if not s:
        return s
    return s[0] + "*" * max(0, len(s) - 1)


def mask_id_card(value: Any) -> str:
    """身份证部分隐藏：保留前3后4（如 110101********1234）。"""
    return _mask_sensitive(value, keep_prefix=3, keep_suffix=4)


def mask_phone(value: Any) -> str:
    """手机号部分隐藏：保留前3后4。"""
    return _mask_sensitive(value, keep_prefix=3, keep_suffix=4)


def mask_address(value: Any) -> str:
    """地址脱敏：保留前两级行政区（区/街道级），隐藏详细门牌号。"""
    s = str(value or "").strip()
    if not s:
        return s
    if len(s) <= 6:
        return s[0] + "*" * max(0, len(s) - 1)
    return s[:6] + "*" * (len(s) - 6)


_MASKERS = {
    "name": mask_name,
    "id_card": mask_id_card,
    "phone": mask_phone,
    "address": mask_address,
}


def mask_value(value: Any, kind: str) -> str:
    """按脱敏类型对单个值脱敏。"""
    if value is None:
        return value
    fn = _MASKERS.get(kind)
    if fn is None:
        return str(value)
    return fn(value)


def role_policy(role: str) -> Dict[str, Set[str]]:
    """获取指定角色的字段策略（未知角色回退为普通用户策略，最严格）。"""
    return ROLE_FIELD_VISIBILITY.get(role, ROLE_FIELD_VISIBILITY[Role.NORMAL])


# ============================================================================
# 递归脱敏
# ============================================================================

def apply_masking(data: Any, role: str = Role.ASSESSOR) -> Any:
    """对数据结构按角色策略递归脱敏。

    处理规则：
    - 命中 "hidden" 集合的字段：从输出中删除（避免泄露）。
    - 命中 "masked" 集合的字段：按字段类别应用脱敏函数。
    - 其余字段：原样保留（深拷贝，不修改输入）。

    参数：
        data: 任意嵌套的 dict / list / 标量。
        role: 当前角色。

    返回：
        脱敏后的新对象；对非容器输入原样返回。
    """
    policy = role_policy(role)
    hidden = policy["hidden"]
    masked = policy["masked"]

    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if k in hidden:
                continue  # 完全隐藏
            if k in masked:
                kind = _FIELD_KIND.get(k, "name")
                # 容器类字段（列表/字典）递归处理而非整体打码
                if isinstance(v, (dict, list)):
                    out[k] = apply_masking(v, role)
                else:
                    out[k] = mask_value(v, kind)
            else:
                out[k] = apply_masking(v, role)
        return out
    elif isinstance(data, (list, tuple)):
        return [apply_masking(i, role) for i in data]
    else:
        return data


# ============================================================================
# 数据安全配置
# ============================================================================

@dataclass
class DataSecurityConfig:
    """数据安全配置（JSON 持久化到 ~/.tb_risk/security.json）。

    默认值遵循"安全优先"：本地优先存储、导出强制脱敏、关键操作审计开启。
    """
    local_only: bool = True                      # 默认本地优先存储
    network_upload_allowed: bool = False         # 不允许联网上传
    export_masking_enabled: bool = True          # 导出统一走强化脱敏
    audit_enabled: bool = True                   # 审计总开关
    log_assess: bool = True                      # 记录评估操作
    log_export: bool = True                      # 记录导出操作
    log_import: bool = True                      # 记录导入操作
    default_role: str = Role.ASSESSOR            # 默认角色
    current_role: str = Role.ASSESSOR            # 当前会话角色
    security_audit_db: str = SECURITY_AUDIT_DB   # 审计库路径

    def save(self, path: Optional[str] = None):
        """保存配置到 JSON 文件（原子写入）。"""
        path = path or SECURITY_CONFIG_FILE
        try:
            d = asdict(self)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("安全配置保存失败: %s", e)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "DataSecurityConfig":
        """从 JSON 文件加载配置；文件不存在或损坏时返回默认配置。"""
        path = path or SECURITY_CONFIG_FILE
        cfg = cls()
        if not os.path.exists(path):
            return cfg
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in asdict(cfg):
                    if key in data:
                        setattr(cfg, key, data[key])
            if cfg.current_role not in Role.ALL:
                cfg.current_role = cfg.default_role
        except Exception as e:
            LOGGER.warning("安全配置加载失败，使用默认: %s", e)
        return cfg


# ============================================================================
# 安全审计器（评估/导出/导入 关键操作）
# ============================================================================

class SecurityAuditor:
    """关键操作审计器。

    复用 health_interop.audit.AuditLogger（SQLite + 哈希链完整性校验）。
    仅当 config.audit_enabled 且对应操作开关开启时记录。
    """

    OPERATION_ASSESS = "ASSESS"
    OPERATION_EXPORT = "EXPORT"
    OPERATION_IMPORT = "IMPORT"
    OPERATION_CONFIG = "CONFIG"
    OPERATION_LOGIN = "LOGIN"

    def __init__(self, config: Optional[DataSecurityConfig] = None):
        self.config = config or DataSecurityConfig()
        self._logger = None
        self._lock_init = False

    def _get_logger(self):
        """懒初始化底层审计日志器。"""
        if not self._lock_init:
            try:
                from ..health_interop.audit import AuditLogger
                self._logger = AuditLogger(db_path=self.config.security_audit_db)
            except Exception as e:  # pragma: no cover
                LOGGER.warning("安全审计器初始化失败: %s", e)
                self._logger = None
            self._lock_init = True
        return self._logger

    def log_operation(self, operation: str, role: Optional[str] = None,
                      details: Any = None, user_id: str = "") -> bool:
        """记录关键操作。返回是否成功写入。"""
        if not self.config.audit_enabled:
            return False
        logger = self._get_logger()
        if logger is None:
            return False
        try:
            logger.log_operation(
                operation_type=operation,
                user_id=user_id or "system",
                user_role=role or self.config.current_role,
                details=details,
            )
            return True
        except Exception as e:  # pragma: no cover
            LOGGER.warning("审计写入失败: %s", e)
            return False

    def log_assess(self, role: Optional[str] = None, **kw) -> bool:
        if not self.config.log_assess:
            return False
        return self.log_operation(self.OPERATION_ASSESS, role, details=kw.get("details"))

    def log_export(self, fmt: str, filepath: str, role: Optional[str] = None) -> bool:
        if not self.config.log_export:
            return False
        return self.log_operation(
            self.OPERATION_EXPORT, role,
            details={"format": fmt, "filepath": os.path.basename(filepath)},
        )

    def log_import(self, source: str, record_count: int, role: Optional[str] = None) -> bool:
        if not self.config.log_import:
            return False
        return self.log_operation(
            self.OPERATION_IMPORT, role,
            details={"source": source, "record_count": record_count},
        )

    def recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """查询最近的关键操作审计记录。"""
        logger = self._get_logger()
        if logger is None:
            return []
        try:
            ops = {self.OPERATION_ASSESS, self.OPERATION_EXPORT,
                   self.OPERATION_IMPORT, self.OPERATION_CONFIG, self.OPERATION_LOGIN}
            records = logger.query(limit=limit)
            return [r.to_dict() for r in records if r.operation_type in ops]
        except Exception as e:  # pragma: no cover
            LOGGER.warning("查询安全审计失败: %s", e)
            return []

    def verify_integrity(self) -> Dict[str, Any]:
        """校验审计链完整性（防篡改）。"""
        logger = self._get_logger()
        if logger is None:
            return {"valid": False, "errors": ["审计器不可用"], "total": 0, "checked": 0}
        try:
            return logger.verify_integrity()
        except Exception as e:  # pragma: no cover
            return {"valid": False, "errors": [str(e)], "total": 0, "checked": 0}


# ============================================================================
# 统一导出守卫
# ============================================================================

class ExportGuard:
    """统一导出守卫：所有导出（PDF/Excel/JSON/CSV/TXT）走此入口完成脱敏与审计。

    用法：
        guard = ExportGuard(config)
        masked = guard.prepare(results)          # 得到脱敏后的数据
        guard.export("excel", results, path)     # 脱敏后委托底层导出器 + 写审计
    """

    def __init__(self, config: Optional[DataSecurityConfig] = None,
                 auditor: Optional[SecurityAuditor] = None):
        self.config = config or DataSecurityConfig()
        self.auditor = auditor or SecurityAuditor(self.config)

    def current_role(self) -> str:
        return self.config.current_role

    def prepare(self, data: Any, role: Optional[str] = None) -> Any:
        """对导出数据应用脱敏（基于当前角色与导出脱敏开关）。

        始终深拷贝，不修改原始数据。
        """
        cloned = copy.deepcopy(data)
        if not self.config.export_masking_enabled:
            return cloned
        return apply_masking(cloned, role or self.config.current_role)

    def export(self, fmt: str, results: Any, filepath: str,
               role: Optional[str] = None, patient_info: Any = None) -> bool:
        """统一导出入口：脱敏 -> 委托底层导出器 -> 写审计。

        参数：
            fmt: 'excel' / 'json' / 'pdf' / 'csv' / 'txt'
            results: 评估结果
            filepath: 输出路径
            role: 当前角色（默认用配置角色）
            patient_info: 患者信息（PDF 需要）

        返回：
            bool 是否成功。
        """
        fmt = (fmt or "").lower()
        masked = self.prepare(results, role)
        try:
            if fmt == "excel":
                from ..export_utils import export_to_excel as _fn
                ok = _fn(masked, filepath)
            elif fmt == "json":
                from ..export_utils import export_to_json as _fn
                ok = _fn(masked, filepath)
            elif fmt == "pdf":
                from ..export_utils import _export_pdf_report as _fn
                ok = _fn(masked, patient_info, filepath)
            elif fmt == "csv":
                ok = self._export_simple(masked, filepath, "csv")
            elif fmt == "txt":
                ok = self._export_simple(masked, filepath, "txt")
            else:
                LOGGER.warning("不支持的导出格式: %s", fmt)
                return False
        except Exception as e:  # pragma: no cover
            LOGGER.warning("导出失败(%s): %s", fmt, e, exc_info=True)
            return False

        if ok:
            self.auditor.log_export(fmt, filepath, role)
        return ok

    @staticmethod
    def _export_simple(data: Dict[str, Any], filepath: str, kind: str) -> bool:
        """极简 CSV/TXT 导出（供非 GUI 场景使用）。"""
        import datetime
        try:
            with open(filepath, "w", encoding="utf-8-sig" if kind == "csv" else "utf-8",
                      newline="" if kind == "csv" else None) as f:
                if kind == "csv":
                    import csv
                    w = csv.writer(f)
                    w.writerow(["综合风险等级", data.get("overall_risk", "N/A")])
                    w.writerow(["感染概率", f"{data.get('base_infection_probability', 0):.1f}%"])
                else:
                    f.write(f"综合风险等级: {data.get('overall_risk', 'N/A')}\n")
                    f.write(f"感染概率: {data.get('base_infection_probability', 0):.1f}%\n")
                    f.write(f"导出时间: {datetime.datetime.now().isoformat()}\n")
            return True
        except Exception as e:  # pragma: no cover
            LOGGER.warning("简单导出失败: %s", e)
            return False


# ============================================================================
# 统一管理器
# ============================================================================

class DataSecurityManager:
    """数据安全管理器：聚合配置 + 审计 + 导出守卫，作为应用层统一入口。"""

    def __init__(self, config: Optional[DataSecurityConfig] = None):
        self.config = config or DataSecurityConfig.load()
        self.auditor = SecurityAuditor(self.config)
        self.guard = ExportGuard(self.config, self.auditor)

    @property
    def current_role(self) -> str:
        return self.config.current_role

    def set_role(self, role: str) -> str:
        """切换当前角色并持久化。"""
        if role in Role.ALL:
            self.config.current_role = role
            self.config.save()
        return self.config.current_role

    def set_config(self, **kwargs) -> None:
        """批量更新配置并持久化。"""
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.config.save()


# ============================================================================
# 合规说明
# ============================================================================

def compliance_summary(config: Optional[DataSecurityConfig] = None) -> str:
    """生成数据合规说明文本（供界面展示与文档引用）。"""
    cfg = config or DataSecurityConfig()
    lines = [
        "《数据安全与合规》说明",
        "=" * 48,
        f"· 当前角色：{role_label(cfg.current_role)}（默认 {role_label(cfg.default_role)}）",
        f"· 本地优先存储：{'开启' if cfg.local_only else '关闭'}；联网上传：{'禁止' if not cfg.network_upload_allowed else '允许'}",
        f"· 导出强化脱敏：{'开启' if cfg.export_masking_enabled else '关闭'}",
        f"· 关键操作审计：{'开启' if cfg.audit_enabled else '关闭'}（评估/导出/导入）",
        "",
        "合规要点（《个人信息保护法》PIPL 及卫生健康行业规范）：",
        "1. 最小必要与目的相关：仅收集评估所需字段，导出默认脱敏。",
        "2. 敏感个人信息从严保护：医疗健康信息属敏感个人信息，仅评估员及以上可见。",
        "3. 本地优先、数据不出本机：默认不联网上传，满足医疗机构数据主权要求。",
        "4. 可追溯与审计：评估/导出/导入均写审计日志，哈希链防篡改。",
        "5. 权限最小化：不同角色可见字段不同，落实职责分离（SoD）。",
    ]
    return "\n".join(lines)
