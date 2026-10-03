#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API 模块测试。

覆盖偶纲：
- 包导出与依赖检测
- 配置（默认/字典/环境变量/校验）
- 安全（API Key 认证 + 滑动窗口限流）
- 异步任务管理器（提交/状态/结果/清理）
- CDS Hooks 预警卡构建
- FastAPI 应用工厂（未安装 FastAPI 时可读报错）
"""

import os
import time

import pytest

from tb_risk.api import (
    APIConfig,
    APIKeyStore,
    RateLimiter,
    JobManager,
    STATUS_QUEUED, STATUS_RUNNING, STATUS_SUCCEEDED, STATUS_FAILED,
    build_cdhooks_card, build_cdhooks_response, indicator_for_risk,
    CDS_SERVICES, SERVICE_ID, SERVICE_HOOK,
    FASTAPI_AVAILABLE,
)


# ============================================================================
# 包导出与依赖检测
# ============================================================================

def test_api_exports():
    """核心公开 API 均可导入。"""
    from tb_risk.api import (
        APIConfig, APIKeyStore, RateLimiter, JobManager,
        build_cdhooks_card, build_cdhooks_response, indicator_for_risk,
        create_app, run_server,
    )
    assert APIConfig is not None
    assert APIKeyStore is not None
    assert RateLimiter is not None
    assert JobManager is not None
    assert callable(create_app)
    assert callable(run_server)


def test_check_dependencies():
    """依赖检测返回 dict 且含关键模块键。"""
    import tb_risk.api as api
    from tb_risk.api import _DEPENDENCIES
    assert isinstance(_DEPENDENCIES, dict)
    for mod in ("fastapi", "uvicorn", "pydantic"):
        assert mod in _DEPENDENCIES


def test_fastapi_availability_consistent():
    """FASTAPI_AVAILABLE 与依赖检测结果一致。"""
    import tb_risk.api as api
    assert api.FASTAPI_AVAILABLE == api._DEPENDENCIES.get("fastapi", False)


# ============================================================================
# 配置
# ============================================================================

def test_config_defaults():
    """默认配置具备合理初值。"""
    cfg = APIConfig.defaults()
    assert cfg.port == 8000
    assert cfg.host == "0.0.0.0"
    assert cfg.rate_limit_per_minute == 60
    assert cfg.batch_max_records == 500
    assert cfg.api_keys == []
    assert not cfg.auth_enabled  # 无 Key 时认证关闭


def test_config_property_auth_enabled():
    """auth_enabled 与 api_keys 一致。"""
    assert APIConfig(api_keys=["k1"]).auth_enabled
    assert not APIConfig(api_keys=[]).auth_enabled


@pytest.fixture()
def _no_api_env(monkeypatch):
    """清理 TB_RISK_API_* 环境变量，避免污染其他测试。"""
    for key in list(os.environ.keys()):
        if key.startswith("TB_RISK_API_"):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_config_from_dict(monkeypatch):
    """字典覆盖已知字段，未知字段进入 extra。"""
    cfg = APIConfig(api_keys=["k1"])
    cfg._apply_dict({
        "port": 9000,
        "rate_limit_per_minute": 30,
        "custom_field": 123,
    })
    assert cfg.port == 9000
    assert cfg.rate_limit_per_minute == 30
    assert cfg.extra["custom_field"] == 123


def test_config_from_env(monkeypatch, _no_api_env):
    """环境变量注入（含逗号分隔 API Keys）。"""
    monkeypatch.setenv("TB_RISK_API_PORT", "8080")
    monkeypatch.setenv("TB_RISK_API_KEYS", "k1,k2 , k3")
    monkeypatch.setenv("TB_RISK_API_MASK_SENSITIVE", "false")
    cfg = APIConfig.from_env()
    assert cfg.port == 8080
    assert cfg.api_keys == ["k1", "k2", "k3"]
    assert cfg.mask_sensitive is False


def test_config_validate_ok():
    """合法配置校验通过。"""
    APIConfig().validate()


def test_config_validate_port_invalid():
    """非法端口抛出 ValueError。"""
    with pytest.raises(ValueError):
        APIConfig(port=99999).validate()


def test_config_to_dict():
    """to_dict 序列化关键字段。"""
    d = APIConfig(api_keys=["k1"]).to_dict()
    assert d["auth_enabled"] is True
    assert d["port"] == 8000
    assert "api_keys" not in d  # 不泄露 Key 本身


# ============================================================================
# 安全：API Key 认证
# ============================================================================

def test_api_key_store_valid():
    """合法 Key 通过校验。"""
    store = APIKeyStore(["k1", "k2"])
    assert store.is_valid("X-API-Key", "k1")
    assert store.is_valid("X-API-Key", "k2")
    assert store.enabled


def test_api_key_store_invalid():
    """非法 Key 拒绝。"""
    store = APIKeyStore(["k1"])
    assert not store.is_valid("X-API-Key", "bad")
    assert not store.is_valid("X-API-Key", None)
    assert not store.is_valid("X-API-Key", ["k1", "bad"])  # 列表含非法


def test_api_key_store_disabled():
    """未配置 Key 时放行（认证关闭）。"""
    store = APIKeyStore([])
    assert store.is_valid("X-API-Key", "anything")
    assert not store.enabled


def test_api_key_store_add_remove():
    """动态增删 Key。"""
    store = APIKeyStore()
    store.add("k1")
    assert store.is_valid("X-API-Key", "k1")
    store.remove("k1")
    assert not store.is_valid("X-API-Key", "k1")


# ============================================================================
# 安全：滑动窗口限流
# ============================================================================

def test_rate_limiter_allows_under_limit():
    """未超限时全部放行。"""
    limiter = RateLimiter(limit=3, window_seconds=60)
    assert limiter.allow("c1")
    assert limiter.allow("c1")
    assert limiter.allow("c1")


def test_rate_limiter_blocks_over_limit():
    """超限时拒绝并计数正确。"""
    limiter = RateLimiter(limit=2, window_seconds=60)
    assert limiter.allow("c1")
    assert limiter.allow("c1")
    assert not limiter.allow("c1")
    assert limiter.get_count("c1") == 2


def test_rate_limiter_per_client_isolated():
    """不同 client 互不影响。"""
    limiter = RateLimiter(limit=1, window_seconds=60)
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")  # 另一 client 不受影响


def test_rate_limiter_zero_limit_disabled():
    """limit=0 表示不限流。"""
    limiter = RateLimiter(limit=0)
    for _ in range(100):
        assert limiter.allow("c1")


def test_rate_limiter_window_expiry(monkeypatch):
    """窗口过期后恢复放行。"""
    limiter = RateLimiter(limit=1, window_seconds=60)
    now = [1000.0]
    monkeypatch.setattr("tb_risk.api.security.time.monotonic", lambda: now[0])
    assert limiter.allow("c1")
    assert not limiter.allow("c1")
    # 时间前进 61 秒，窗口过期
    now[0] += 61
    assert limiter.allow("c1")


def test_rate_limiter_reset_and_prune():
    """reset 与 prune_expired 正确清理。"""
    limiter = RateLimiter(limit=1, window_seconds=60)
    limiter.allow("a")
    limiter.allow("b")
    assert limiter.get_count("a") == 1
    limiter.reset("a")
    assert limiter.get_count("a") == 0
    assert limiter.get_count("b") == 1
    limiter.reset()
    assert limiter.get_count("b") == 0


# ============================================================================
# 异步任务管理器
# ============================================================================

def _sleep_task(seconds):
    time.sleep(seconds)
    return {"done": True, "waited": seconds}


def test_job_manager_submit_and_result():
    """提交任务并获取成功结果。"""
    mgr = JobManager(max_workers=2)
    try:
        job_id = mgr.submit("test", _sleep_task, 0.01)
        assert isinstance(job_id, str) and job_id
        # 轮询直到完成
        result = None
        deadline = time.time() + 5
        while time.time() < deadline:
            info = mgr.get_result(job_id)
            if info and info["status"] == STATUS_SUCCEEDED:
                result = info
                break
            time.sleep(0.01)
        assert result is not None
        assert result["status"] == STATUS_SUCCEEDED
        assert result["result"] == {"done": True, "waited": 0.01}
    finally:
        mgr.shutdown()


def test_job_manager_failure():
    """任务异常时进入 failed 状态并记录错误。"""
    mgr = JobManager(max_workers=1)

    def _boom():
        raise ValueError("boom")

    try:
        job_id = mgr.submit("test", _boom)
        deadline = time.time() + 5
        while time.time() < deadline:
            info = mgr.get_result(job_id)
            if info and info["status"] == STATUS_FAILED:
                break
            time.sleep(0.01)
        assert info["status"] == STATUS_FAILED
        assert "boom" in info["error"]
    finally:
        mgr.shutdown(wait=False)


def test_job_manager_list_and_clear():
    """list_jobs 与 clear_finished 正确。"""
    mgr = JobManager(max_workers=2)
    try:
        jid = mgr.submit("test", _sleep_task, 0.01)
        time.sleep(0.1)
        jobs = mgr.list_jobs()
        assert any(j["id"] == jid for j in jobs)
        assert mgr.count() >= 1
        removed = mgr.clear_finished()
        assert removed >= 1
        assert mgr.count() == 0
    finally:
        mgr.shutdown(wait=False)


def test_job_manager_unknown_status():
    """查询不存在任务返回 None。"""
    mgr = JobManager(max_workers=1)
    try:
        assert mgr.get_status("nope") is None
        assert mgr.get_result("nope") is None
    finally:
        mgr.shutdown(wait=False)


# ============================================================================
# CDS Hooks 预警卡
# ============================================================================

def _sample_result(overall_risk="高"):
    """构造最小评估结果。"""
    return {
        "overall_risk": overall_risk,
        "base_infection_probability": 12.3,
        "overall_suggestion": "建议进行胸部X线+分子检测",
        "potential_patients": {
            "family": [
                {"name": "张三", "priority": "高",
                 "disease_probability": 20.0},
                {"name": "李四", "priority": "极高",
                 "disease_probability": 35.0},
            ],
            "social": [
                {"name": "王五", "priority": "低",
                 "disease_probability": 1.0},
            ],
        },
        "summary": {
            "total_contacts": 3,
            "high_risk_count": 2,
            "medium_risk_count": 0,
        },
    }


def test_indicator_for_risk_mapping():
    """风险等级 → indicator 映射。"""
    assert indicator_for_risk("低") == "info"
    assert indicator_for_risk("中") == "warning"
    assert indicator_for_risk("高") == "critical"
    assert indicator_for_risk("极高") == "critical"
    assert indicator_for_risk(None) == "info"
    assert indicator_for_risk("未知") == "info"


def test_cds_services_discovery():
    """服务发现注册表符合 CDS Hooks 结构。"""
    assert SERVICE_ID == "tb-risk-assessment"
    assert SERVICE_HOOK == "patient-view"
    assert len(CDS_SERVICES) == 1
    svc = CDS_SERVICES[0]
    assert svc["id"] == SERVICE_ID
    assert svc["hook"] == SERVICE_HOOK
    assert "label" in svc or "title" in svc


def test_build_cdhooks_card_high_risk():
    """高风险 → critical 卡 + 筛查建议 + 接触者脱敏。"""
    card = build_cdhooks_card(_sample_result("高"), patient_id="P1")
    assert card["indicator"] == "critical"
    assert "结核病风险评估" in card["summary"]
    assert card["suggestions"][0]["label"] == "建议立即进行结核病筛查"
    assert card["links"][0]["url"] == "/reports?patient_id=P1"
    # 姓名脱敏（mask_sensitive=True 默认）
    contacts = card["extension"]["top_contacts"]
    assert any(c["name"].endswith("*") for c in contacts)
    assert all("*" in (c["name"] or "") for c in contacts)


def test_build_cdhooks_card_no_mask():
    """mask_sensitive=False 时保留完整姓名。"""
    card = build_cdhooks_card(_sample_result("极高"), mask_sensitive=False)
    contacts = card["extension"]["top_contacts"]
    assert any(c["name"] == "张三" for c in contacts)


def test_build_cdhooks_card_low_risk():
    """低风险 → info 卡，无 critical 建议。"""
    card = build_cdhooks_card(_sample_result("低"))
    assert card["indicator"] == "info"
    assert "建议" not in str(card.get("suggestions") or [])


def test_build_cdhooks_response():
    """$apply 响应含 cards 列表。"""
    resp = build_cdhooks_response(_sample_result("中"), patient_id="P2")
    assert "cards" in resp
    assert len(resp["cards"]) == 1
    assert resp["cards"][0]["indicator"] == "warning"


def test_build_cdhooks_card_empty():
    """空结果仍返回结构完整的 info 卡。"""
    card = build_cdhooks_card(None)
    assert card["indicator"] == "info"
    assert card["service_id"] == SERVICE_ID  # extension 透传打平
    assert "失败" in card["summary"]


# ============================================================================
# FastAPI 应用工厂
# ============================================================================

def test_create_app_behavior():
    """create_app：FastAPI 可用则返回应用，否则抛可读 ImportError。"""
    from tb_risk.api import create_app
    if FASTAPI_AVAILABLE:
        app = create_app(APIConfig(api_keys=["k1"]))
        assert app is not None
        assert hasattr(app, "routes")
    else:
        with pytest.raises(ImportError) as exc:
            create_app()
        assert "FastAPI" in str(exc.value)