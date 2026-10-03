#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API FastAPI 应用工厂。

``create_app(config)`` 构建 FastAPI 应用，注册：
- 核心评估端点：单患者评估 / 批量评估（异步 job）/ 模型训练触发 / 模型状态 / 报告生成
- CDS Hooks 标准接口：``GET /cds-services`` 与 ``POST /cds-services/{id}/$apply``
- 安全：API Key 认证 + 滑动窗口限流
- 自动 OpenAPI/Swagger 文档（FastAPI 内置，``/docs``、``/redoc``、``/openapi.json``）

FastAPI 为可选依赖；未安装时 ``create_app`` 抛出 ImportError，并给出安装提示
（与 ops/errors.py 的"可读错误"风格一致）。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from .config import APIConfig
from .security import APIKeyStore, RateLimiter
from .background import JobManager, STATUS_SUCCEEDED, STATUS_FAILED

LOGGER = logging.getLogger("tb_risk.api.app")


def _import_fastapi():
    """惰性导入 FastAPI 相关模块，未安装时抛出带提示的 ImportError。"""
    try:
        from fastapi import FastAPI, Request, HTTPException  # noqa: F401
        from fastapi.responses import JSONResponse  # noqa: F401
        return FastAPI, Request, HTTPException, JSONResponse
    except ImportError as e:
        raise ImportError(
            "FastAPI 未安装，无法启动 REST API 服务。请执行："
            "pip install 'tb_risk[api]' 或 pip install fastapi uvicorn"
        ) from e


class _AppState:
    """应用共享状态（service / 安全 / 任务管理器）。"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.api_key_store = APIKeyStore(config.api_keys)
        self.rate_limiter = RateLimiter(
            limit=config.rate_limit_per_minute, window_seconds=60)
        self.job_manager = JobManager(max_workers=4)
        self._service = None
        self._service_lock = threading.Lock()

    def get_service(self):
        """惰性创建共享 RiskAssessmentService（用于单次评估/训练/状态）。"""
        with self._service_lock:
            if self._service is None:
                from tb_risk.core import RiskAssessmentService
                self._service = RiskAssessmentService(
                    enable_karamay=self.config.enable_karamay)
            return self._service

    def new_job_service(self):
        """为每个批量任务创建独立服务（隔离共享状态，避免并发竞态）。"""
        from tb_risk.core import RiskAssessmentService
        return RiskAssessmentService(enable_karamay=self.config.enable_karamay)

    def shutdown(self):
        self.job_manager.shutdown(wait=True)


def _auth_dependency(request):
    """认证 + 限流依赖（FastAPI dependency）。"""
    # 延迟导入 FastAPI 类型，仅用于类型注解无碍
    state = getattr(request.app.state, "state", None)
    if state is None:
        state = request.app.state
    config = state.config
    store = state.api_key_store
    limiter = state.rate_limiter

    header_name = config.api_key_header
    value = request.headers.get(header_name)
    if not store.is_valid(header_name, value):
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="无效或缺失 API Key")
    client = value if isinstance(value, str) else "anonymous"
    request.state.client = client
    if not limiter.allow(client):
        from fastapi import HTTPException
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    return client


def create_app(config: Optional[APIConfig] = None) -> Any:
    """构建 FastAPI 应用。

    参数：
        config: APIConfig，None 时从默认配置加载

    返回：
        FastAPI 应用实例

    抛出：
        ImportError: FastAPI 未安装
    """
    FastAPI, Request, HTTPException, JSONResponse = _import_fastapi()
    if config is None:
        config = APIConfig.load()
    config.validate()

    state = _AppState(config)

    app = FastAPI(
        title="tb_risk 结核病风险评估 REST API",
        description=(
            "基于 SEIR 传播动力学与机器学习模型的结核病风险评估服务。\n\n"
            "提供单患者评估、批量评估（异步）、模型训练与状态查询、报告生成，"
            "以及 CDS Hooks 标准接口（可直接嵌入 EMR 工作流）。\n\n"
            "认证：请在请求头携带 API Key（默认字段 X-API-Key）。"
        ),
        version="5.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.state = state

    # ---- 全局异常处理：统一返回结构化错误 ----
    @app.exception_handler(Exception)
    async def _handle_exception(request, exc):
        LOGGER.warning("REST API 异常: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "detail": "服务器内部错误"},
        )

    # ==================================================================
    # 系统状态
    # ==================================================================

    @app.get("/status", tags=["系统"])
    async def status(request: Request):
        """服务状态与配置信息（不含敏感 API Key）。"""
        cfg = state.config
        info = cfg.to_dict()
        info["api_keys_count"] = len(state.api_key_store)
        info["active_jobs"] = state.job_manager.count()
        info["ml_available"] = _ml_available()
        return info

    @app.get("/health", tags=["系统"])
    async def health(request: Request):
        """健康检查（供负载均衡/探针使用）。"""
        return {"status": "ok"}

    # ==================================================================
    # 单患者评估
    # ==================================================================

    @app.post("/assess", tags=["评估"], dependencies=[_auth_dependency])
    async def assess(request: Request, body: Dict[str, Any]):
        """单患者结核病风险评估。

        请求体：{ patient_info, family_members, social_contacts, use_ml, use_seir }
        """
        if not body.get("patient_info") and not body.get("family_members") \
                and not body.get("social_contacts"):
            raise HTTPException(status_code=422, detail="至少提供一项评估数据")
        service = state.get_service()
        result = service.assess(
            patient_info=body.get("patient_info") or {},
            family_members=body.get("family_members") or [],
            social_contacts=body.get("social_contacts") or [],
            use_ml=bool(body.get("use_ml", state.config.default_use_ml)),
            use_seir=bool(body.get("use_seir", state.config.default_use_seir)),
        )
        return _maybe_mask(result, state.config.mask_sensitive)

    # ==================================================================
    # 批量评估（异步）
    # ==================================================================

    @app.post("/assess/batch", tags=["评估"], dependencies=[_auth_dependency])
    async def assess_batch(request: Request, body: Dict[str, Any]):
        """批量评估（异步）。提交后返回 job_id，用 GET /jobs/{id} 轮询结果。"""
        records = body.get("records") or []
        if not records:
            raise HTTPException(status_code=422, detail="records 不能为空")
        if len(records) > state.config.batch_max_records:
            raise HTTPException(
                status_code=422,
                detail=f"records 数量 {len(records)} 超过上限 "
                       f"{state.config.batch_max_records}",
            )
        use_ml = bool(body.get("use_ml", state.config.default_use_ml))
        use_seir = bool(body.get("use_seir", state.config.default_use_seir))
        job_id = state.job_manager.submit(
            "batch_assess",
            _run_batch,
            records, use_ml, use_seir,
        )
        return {"job_id": job_id, "status": "queued", "total_records": len(records)}

    # ==================================================================
    # 异步任务查询
    # ==================================================================

    @app.get("/jobs/{job_id}", tags=["评估"], dependencies=[_auth_dependency])
    async def get_job(job_id: str):
        """查询异步任务状态与结果。"""
        info = state.job_manager.get_result(job_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
        if info["status"] == STATUS_FAILED:
            info["error"] = info.get("error")
        if info["status"] == STATUS_SUCCEEDED and info.get("result") is not None:
            info["result"] = _maybe_mask(info["result"], state.config.mask_sensitive)
        return info

    @app.get("/jobs", tags=["评估"], dependencies=[_auth_dependency])
    async def list_jobs():
        """列出最近的任务状态。"""
        return {"jobs": state.job_manager.list_jobs()}

    # ==================================================================
    # 模型训练与状态
    # ==================================================================

    @app.post("/train", tags=["模型"], dependencies=[_auth_dependency])
    async def train(request: Request, body: Dict[str, Any]):
        """触发 ML 模型训练（异步 job）。"""
        n_samples = int(body.get("n_samples", 2000))
        enable_hyperopt = bool(body.get("enable_hyperopt", False))
        save_path = body.get("save_path") or None
        job_id = state.job_manager.submit(
            "train",
            _run_train,
            n_samples, enable_hyperopt, save_path,
        )
        return {"job_id": job_id, "status": "queued"}

    @app.get("/model/status", tags=["模型"], dependencies=[_auth_dependency])
    async def model_status(request: Request):
        """查询模型训练状态与性能。"""
        service = state.get_service()
        predictor = service.ml_predictor
        return {
            "is_trained": bool(predictor.is_trained),
            "model_count": len(predictor.models),
            "model_performance": predictor.model_performance or {},
            "gnn_is_trained": bool(predictor.gnn_is_trained),
            "training_sample_count": predictor.training_sample_count,
            "use_real_data": predictor.use_real_data,
        }

    # ==================================================================
    # 报告生成
    # ==================================================================

    @app.post("/reports", tags=["报告"], dependencies=[_auth_dependency])
    async def generate_report(request: Request, body: Dict[str, Any]):
        """生成评估结果报告（html/json/text）。"""
        result = body.get("result") or {}
        fmt = body.get("format", "html")
        ai_enhanced = bool(body.get("ai_enhanced", False))
        if fmt not in ("html", "json", "text"):
            raise HTTPException(status_code=422, detail=f"不支持的格式: {fmt}")
        content = _generate_report(result, fmt, ai_enhanced)
        media = {"html": "text/html", "json": "application/json", "text": "text/plain"}[fmt]
        return JSONResponse(content=content, media_type=media)

    # ==================================================================
    # CDS Hooks 标准接口
    # ==================================================================

    @app.get("/cds-services", tags=["CDS Hooks"])
    async def cds_discovery():
        """CDS Hooks 服务发现（GET /cds-services）。"""
        from .cds_hooks import CDS_SERVICES
        return {"services": CDS_SERVICES}

    @app.post("/cds-services/{service_id}/$apply",
              tags=["CDS Hooks"], dependencies=[_auth_dependency])
    async def cds_apply(service_id: str, request: Request, body: Dict[str, Any]):
        """CDS Hooks $apply：执行服务并返回预警卡。

        请求体遵循 CDS Hooks 规范：{ hook, hookInstance, context, prefetch, ... }。
        从 ``context``（patientId 等）与 ``prefetch.patient`` 或内嵌
        ``tbRisk`` 数据构建患者信息，后台评估并返回 cards。
        """
        from .cds_hooks import SERVICE_ID, build_cdhooks_response
        if service_id != SERVICE_ID:
            raise HTTPException(status_code=404,
                                detail=f"未知 CDS 服务: {service_id}")
        patient_data = _extract_patient_from_cds_context(body)
        service = state.new_job_service()
        result = service.assess(
            patient_info=patient_data.get("patient_info") or {},
            family_members=patient_data.get("family_members") or [],
            social_contacts=patient_data.get("social_contacts") or [],
            use_ml=bool(patient_data.get("use_ml", state.config.default_use_ml)),
            use_seir=bool(patient_data.get("use_seir", state.config.default_use_seir)),
        )
        patient_id = (body.get("context") or {}).get("patientId")
        return build_cdhooks_response(
            result, patient_id=patient_id, mask_sensitive=state.config.mask_sensitive)

    return app


# ==================================================================
# 内部实现
# ==================================================================

def _run_batch(records: List[Dict[str, Any]], use_ml: bool, use_seir: bool) -> Dict[str, Any]:
    """在后台执行批量评估（每条独立 try，容忍单条失败）。"""
    from tb_risk.core import RiskAssessmentService
    service = RiskAssessmentService()
    results = []
    for rec in records:
        item = {"ref": rec.get("ref")}
        try:
            res = service.assess(
                patient_info=rec.get("patient_info") or {},
                family_members=rec.get("family_members") or [],
                social_contacts=rec.get("social_contacts") or [],
                use_ml=use_ml,
                use_seir=use_seir,
            )
            item["success"] = True
            item["summary"] = res.get("summary", {})
            item["overall_risk"] = res.get("overall_risk")
            item["patient_score"] = res.get("patient_score")
        except Exception as e:  # noqa: BLE001 - 单条失败不中断批量
            item["success"] = False
            item["error"] = str(e)
        results.append(item)
    return {
        "total": len(results),
        "success_count": sum(1 for r in results if r["success"]),
        "failed_count": sum(1 for r in results if not r["success"]),
        "results": results,
    }


def _run_train(n_samples: int, enable_hyperopt: bool, save_path: Optional[str]) -> Dict[str, Any]:
    """在后台执行模型训练。"""
    from tb_risk.core import RiskAssessmentService
    service = RiskAssessmentService()
    return service.train_ml_models(
        n_samples=n_samples, enable_hyperopt=enable_hyperopt, save_path=save_path)


def _generate_report(result: Dict[str, Any], fmt: str, ai_enhanced: bool) -> str:
    """委托 export_utils 生成报告。"""
    from tb_risk.export_utils.report_export import generate_assessment_report
    return generate_assessment_report(
        {"results": result}, format=fmt, ai_enhanced=ai_enhanced)


def _extract_patient_from_cds_context(body: Dict[str, Any]) -> Dict[str, Any]:
    """从 CDS Hooks 请求体中提取患者评估数据。

    支持两种来源（优先内嵌 tbRisk，其次 prefetch.patient + context）：
    1. ``body["context"]["extension"]["tbRisk"]``：调用方直接传入评估负载
    2. ``body["prefetch"]["patient"]``：FHIR Patient 资源（含 basic_info 等）
    """
    out = {}
    context = body.get("context") or {}
    ext = context.get("extension") or {}
    tb_risk = ext.get("tbRisk") or body.get("tbRisk")
    if isinstance(tb_risk, dict):
        out["patient_info"] = tb_risk.get("patient_info") or {}
        out["family_members"] = tb_risk.get("family_members") or []
        out["social_contacts"] = tb_risk.get("social_contacts") or []
        out["use_ml"] = bool(tb_risk.get("use_ml", False))
        out["use_seir"] = bool(tb_risk.get("use_seir", False))
        return out
    # 回退：从 prefetch.patient 提取 basic_info
    prefetch = body.get("prefetch") or {}
    patient = prefetch.get("patient")
    if isinstance(patient, dict):
        basic_info = {k: v for k, v in patient.items() if k != "resourceType"}
        out["patient_info"] = {"basic_info": basic_info}
    return out


def _mask_name(name) -> Optional[str]:
    """姓名脱敏：保留姓，其余用 * 代替。"""
    if not name:
        return name
    s = str(name)
    if len(s) <= 1:
        return s
    if len(s) == 2:
        return s[0] + "*"
    return s[0] + "*" * (len(s) - 2) + s[-1]


def _maybe_mask(result: Any, mask_sensitive: bool) -> Any:
    """根据配置对结果中的接触者姓名脱敏（深拷贝，不修改原对象）。"""
    if not mask_sensitive or not isinstance(result, dict):
        return result
    import copy
    result = copy.deepcopy(result)
    pp = result.get("potential_patients")
    if isinstance(pp, dict):
        for contacts in pp.values():
            if isinstance(contacts, list):
                for c in contacts:
                    if isinstance(c, dict) and c.get("name"):
                        c["name"] = _mask_name(c["name"])
    return result


def _ml_available() -> bool:
    """检测 ML 核心依赖是否可用。"""
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


__all__ = ["create_app", "_AppState"]