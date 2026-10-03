#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据安全与合规模块单元测试。"""

import json
import os
import pytest

from tb_risk.security import (
    Role, role_label, ROLE_LABELS, apply_masking,
    mask_name, mask_id_card, mask_phone, mask_address, mask_value,
    DataSecurityConfig, SecurityAuditor, ExportGuard, DataSecurityManager,
    compliance_summary,
)


# ============================================================================
# 角色定义
# ============================================================================

class TestRole:
    def test_role_constants(self):
        assert Role.NORMAL == "normal"
        assert Role.ASSESSOR == "assessor"
        assert Role.ADMIN == "admin"
        assert set(Role.ALL) == {"normal", "assessor", "admin"}

    def test_role_labels(self):
        assert ROLE_LABELS[Role.NORMAL] == "普通用户"
        assert ROLE_LABELS[Role.ASSESSOR] == "评估员"
        assert ROLE_LABELS[Role.ADMIN] == "管理员"

    def test_role_label_unknown(self):
        assert role_label("unknown_role") == "unknown_role"


# ============================================================================
# 脱敏函数
# ============================================================================

class TestMaskFunctions:
    def test_mask_name(self):
        assert mask_name("张三") == "张*"
        assert mask_name("欧阳锋") == "欧**"
        assert mask_name("") == ""
        assert mask_name(None) == ""

    def test_mask_id_card(self):
        assert mask_id_card("110101199001011234") == "110***********1234"
        assert mask_id_card("") == ""
        assert mask_id_card(None) == ""

    def test_mask_phone(self):
        assert mask_phone("13800138000") == "138****8000"
        assert mask_phone(None) == ""

    def test_mask_address(self):
        # 超过6位保留前6位
        assert mask_address("克拉玛依区胜利路12号3栋") == "克拉玛依区胜*******"
        # 短地址只保留首字
        assert mask_address("某地") == "某*"
        assert mask_address("") == ""
        assert mask_address(None) == ""

    def test_mask_value_unknown_kind(self):
        assert mask_value("abc", "nonexistent") == "abc"


# ============================================================================
# 递归脱敏
# ============================================================================

class TestApplyMasking:
    def _sample(self):
        return {
            "patient_info": {
                "basic_info": {"age": 45},
                "name": "张伟",
                "id_card": "110101199001011234",
                "phone": "13800138000",
                "current_address": "克拉玛依区胜利路12号",
            },
            "family_results": [
                {"name": "李四", "age": 30, "disease_probability": 0.2},
            ],
            "overall_risk": "high",
            "base_infection_probability": 0.3,
        }

    def test_admin_unchanged(self):
        data = self._sample()
        out = apply_masking(data, Role.ADMIN)
        assert out == data
        assert out["patient_info"]["id_card"] == "110101199001011234"

    def test_assessor_masks_name_phone_address_hides_id(self):
        out = apply_masking(self._sample(), Role.ASSESSOR)
        # id_card 完全隐藏
        assert "id_card" not in out["patient_info"]
        # 姓名打码
        assert out["patient_info"]["name"] == "张*"
        assert out["family_results"][0]["name"] == "李*"
        # 电话/地址脱敏
        assert out["patient_info"]["phone"] == "138****8000"
        assert out["patient_info"]["current_address"] != "克拉玛依区胜利路12号"
        # 非敏感字段保留
        assert out["overall_risk"] == "high"
        assert out["patient_info"]["basic_info"]["age"] == 45

    def test_normal_hides_more(self):
        out = apply_masking(self._sample(), Role.NORMAL)
        pi = out["patient_info"]
        assert "id_card" not in pi
        assert "phone" not in pi
        assert "current_address" not in pi
        assert pi["name"] == "张*"

    def test_input_not_mutated(self):
        data = self._sample()
        apply_masking(data, Role.ASSESSOR)
        # 原始数据不应被修改
        assert data["patient_info"]["id_card"] == "110101199001011234"
        assert data["patient_info"]["name"] == "张伟"

    def test_scalar_returned(self):
        assert apply_masking("张三", Role.ASSESSOR) == "张三"
        assert apply_masking(None, Role.ASSESSOR) is None

    def test_unknown_role_falls_back_to_normal(self):
        out = apply_masking(self._sample(), "bogus")
        assert "id_card" not in out["patient_info"]


# ============================================================================
# 配置
# ============================================================================

class TestDataSecurityConfig:
    def test_defaults_secure(self):
        cfg = DataSecurityConfig()
        assert cfg.local_only is True
        assert cfg.network_upload_allowed is False
        assert cfg.export_masking_enabled is True
        assert cfg.audit_enabled is True
        assert cfg.current_role == Role.ASSESSOR

    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "sec.json")
        cfg = DataSecurityConfig(current_role=Role.ADMIN, local_only=False)
        cfg.save(path)
        loaded = DataSecurityConfig.load(path)
        assert loaded.current_role == Role.ADMIN
        assert loaded.local_only is False

    def test_load_missing_returns_default(self, tmp_path):
        path = str(tmp_path / "missing.json")
        loaded = DataSecurityConfig.load(path)
        assert loaded.current_role == Role.ASSESSOR

    def test_load_invalid_role_falls_back(self, tmp_path):
        path = str(tmp_path / "sec.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"current_role": "bogus", "default_role": Role.NORMAL}, f)
        loaded = DataSecurityConfig.load(path)
        assert loaded.current_role == Role.NORMAL

    def test_load_corrupt_returns_default(self, tmp_path):
        path = str(tmp_path / "sec.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{ not json ")
        cfg = DataSecurityConfig.load(path)
        assert cfg.current_role == Role.ASSESSOR


# ============================================================================
# 审计器
# ============================================================================

class TestSecurityAuditor:
    def _config(self, tmp_path):
        return DataSecurityConfig(security_audit_db=str(tmp_path / "audit.db"))

    def test_log_assess(self, tmp_path):
        cfg = self._config(tmp_path)
        auditor = SecurityAuditor(cfg)
        assert auditor.log_assess(role=Role.ADMIN, details="x") is True
        recs = auditor.recent()
        assert len(recs) == 1
        assert recs[0]["operation_type"] == "ASSESS"
        assert recs[0]["user_role"] == Role.ADMIN

    def test_log_export_import(self, tmp_path):
        cfg = self._config(tmp_path)
        auditor = SecurityAuditor(cfg)
        assert auditor.log_export("excel", "c:/tmp/rep.xlsx", Role.ASSESSOR) is True
        assert auditor.log_import("data.csv", 120, Role.ASSESSOR) is True
        recs = auditor.recent()
        ops = {r["operation_type"] for r in recs}
        assert ops == {"EXPORT", "IMPORT"}

    def test_audit_disabled(self, tmp_path):
        cfg = self._config(tmp_path)
        cfg.audit_enabled = False
        auditor = SecurityAuditor(cfg)
        assert auditor.log_assess() is False

    def test_log_operation_switch_off(self, tmp_path):
        cfg = self._config(tmp_path)
        cfg.log_export = False
        auditor = SecurityAuditor(cfg)
        assert auditor.log_export("json", "x.json") is False

    def test_integrity_valid(self, tmp_path):
        cfg = self._config(tmp_path)
        auditor = SecurityAuditor(cfg)
        auditor.log_assess(role=Role.ADMIN)
        auditor.log_export("pdf", "r.pdf", Role.ADMIN)
        result = auditor.verify_integrity()
        assert result["valid"] is True
        assert result["total"] == 2


# ============================================================================
# 导出守卫
# ============================================================================

class TestExportGuard:
    def test_prepare_masks(self, tmp_path):
        cfg = DataSecurityConfig(security_audit_db=str(tmp_path / "audit.db"))
        guard = ExportGuard(cfg)
        data = {"patient_info": {"name": "张伟", "id_card": "110101199001011234"}}
        masked = guard.prepare(data, role=Role.ASSESSOR)
        assert masked["patient_info"]["name"] == "张*"
        assert "id_card" not in masked["patient_info"]
        # 原始数据未变
        assert data["patient_info"]["id_card"] == "110101199001011234"

    def test_prepare_respects_masking_switch(self, tmp_path):
        cfg = DataSecurityConfig(security_audit_db=str(tmp_path / "audit.db"),
                                 export_masking_enabled=False)
        guard = ExportGuard(cfg)
        data = {"patient_info": {"name": "张伟", "id_card": "110101199001011234"}}
        masked = guard.prepare(data, role=Role.ASSESSOR)
        assert masked["patient_info"]["name"] == "张伟"
        assert masked["patient_info"]["id_card"] == "110101199001011234"

    def test_export_simple_csv(self, tmp_path):
        cfg = DataSecurityConfig(security_audit_db=str(tmp_path / "audit.db"))
        guard = ExportGuard(cfg)
        out = str(tmp_path / "out.csv")
        ok = guard.export("csv", {"overall_risk": "high", "base_infection_probability": 30.0},
                          out, role=Role.ADMIN)
        assert ok is True
        assert os.path.exists(out)
        recs = guard.auditor.recent()
        assert any(r["operation_type"] == "EXPORT" for r in recs)

    def test_export_unsupported_format(self, tmp_path):
        cfg = DataSecurityConfig()
        guard = ExportGuard(cfg)
        assert guard.export("bogus", {}, str(tmp_path / "x.bin")) is False


# ============================================================================
# 聚合管理器
# ============================================================================

class TestDataSecurityManager:
    def test_set_role_persists(self, tmp_path):
        cfg = DataSecurityConfig()
        path = str(tmp_path / "sec.json")
        cfg.save(path)
        mgr = DataSecurityManager(config=cfg)
        # 让配置落盘到临时路径以便断言
        mgr.set_role(Role.ADMIN)
        cfg.save(path)
        loaded = DataSecurityConfig.load(path)
        assert loaded.current_role == Role.ADMIN
        assert mgr.current_role == Role.ADMIN

    def test_set_invalid_role_ignored(self):
        cfg = DataSecurityConfig(current_role=Role.NORMAL)
        mgr = DataSecurityManager(config=cfg)
        mgr.set_role("bogus")
        assert mgr.current_role == Role.NORMAL

    def test_set_config(self):
        cfg = DataSecurityConfig()
        mgr = DataSecurityManager(config=cfg)
        mgr.set_config(local_only=False, network_upload_allowed=True)
        assert cfg.local_only is False
        assert cfg.network_upload_allowed is True

    def test_has_guard_and_auditor(self):
        mgr = DataSecurityManager(config=DataSecurityConfig())
        assert mgr.guard is not None
        assert mgr.auditor is not None


# ============================================================================
# 合规说明
# ============================================================================

class TestComplianceSummary:
    def test_returns_text(self):
        text = compliance_summary()
        assert "个人信息保护法" in text
        assert "本地优先" in text
        assert "审计" in text

    def test_reflects_config(self):
        cfg = DataSecurityConfig(current_role=Role.ADMIN, local_only=False)
        text = compliance_summary(cfg)
        assert "管理员" in text
