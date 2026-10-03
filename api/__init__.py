#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API 与服务化部署子包。

基于 FastAPI 提供核心评估端点（单患者/批量/模型训练/模型状态/报告生成），
支持异步处理、API Key 认证、滑动窗口限流、自动 OpenAPI/Swagger 文档，
并内置 CDS Hooks 标准接口，可将评估结果直接嵌入医生 EMR 工作流。

模块结构：
- config.py:      APIConfig 服务配置（文件/环境变量/默认三层覆盖）
- security.py:    APIKeyStore 认证 + RateLimiter 滑动窗口限流
- background.py:  JobManager 异步任务管理器（批量评估等耗时操作）
- cds_hooks.py:   CDS Hooks 预警卡构建（纯 Python，可独立测试）
- schemas.py:     Pydantic 请求/响应模型（需 FastAPI）
- app.py:         create_app() FastAPI 应用工厂
- main.py:        uvicorn 启动入口

设计原则：
1. 纯 Python 核心（config/security/background/cds_hooks）不依赖 FastAPI，
   可在未安装 FastAPI 的环境下导入与测试。
2. FastAPI 为可选依赖；未安装时 ``create_app``/``run_server`` 抛出可读
   ImportError（与 ops.errors 的"可读错误"风格一致）。
3. 批量评估走异步任务（job），避免同步阻塞；每个任务独立服务实例隔离状态。
"""

from __future__ import annotations

import logging as _logging

_LOGGER = _logging.getLogger(__name__)


def _check_dependencies() -> dict:
    """检查 FastAPI 服务所需依赖是否可用，返回状态字典。"""
    status = {}
    for mod in ("fastapi", "uvicorn", "pydantic"):
        try:
            __import__(mod)
            status[mod] = True
        except ImportError:
            status[mod] = False
    if not status.get("fastapi"):
        _LOGGER.info(
            "REST API 功能未完全启用（缺少 fastapi）。如需启用，请执行："
            "pip install 'tb_risk[api]'"
        )
    return status


_DEPENDENCIES = _check_dependencies()
FASTAPI_AVAILABLE = bool(_DEPENDENCIES.get("fastapi"))
UVICORN_AVAILABLE = bool(_DEPENDENCIES.get("uvicorn"))

# ---- 纯 Python 核心（始终可导入）----
from .config import APIConfig, DEFAULT_CONFIG_PATH  # noqa: E402
from .security import APIKeyStore, RateLimiter  # noqa: E402
from .background import (  # noqa: E402
    JobManager,
    STATUS_QUEUED, STATUS_RUNNING, STATUS_SUCCEEDED, STATUS_FAILED,
)
from .cds_hooks import (  # noqa: E402
    SERVICE_ID, SERVICE_HOOK, SERVICE_TITLE, CDS_SERVICES,
    build_cdhooks_card, build_cdhooks_response, indicator_for_risk,
)


def create_app(config=None):
    """构建 FastAPI 应用（FastAPI 未安装时抛出可读 ImportError）。

    参数：
        config: APIConfig | None

    返回：
        FastAPI 应用
    """
    from .app import create_app as _create_app
    return _create_app(config)


def run_server(config=None, reload: bool = False):
    """启动 REST API 服务（需 uvicorn）。"""
    from .main import run_server as _run_server
    return _run_server(config, reload=reload)


__all__ = [
    # 子模块
    "config", "security", "background", "cds_hooks",
    "schemas", "app", "main",
    # 依赖状态
    "FASTAPI_AVAILABLE", "UVICORN_AVAILABLE",
    # 配置
    "APIConfig", "DEFAULT_CONFIG_PATH",
    # 安全
    "APIKeyStore", "RateLimiter",
    # 异步任务
    "JobManager",
    "STATUS_QUEUED", "STATUS_RUNNING", "STATUS_SUCCEEDED", "STATUS_FAILED",
    # CDS Hooks
    "SERVICE_ID", "SERVICE_HOOK", "SERVICE_TITLE", "CDS_SERVICES",
    "build_cdhooks_card", "build_cdhooks_response", "indicator_for_risk",
    # 入口
    "create_app", "run_server",
]