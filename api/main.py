#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API 服务启动入口（uvicorn runner）。

用法：
    python -m tb_risk.api.main [--host HOST] [--port PORT] [--reload]
    tb-risk api [--host HOST] [--port PORT]   # 通过 CLI 调用

依赖：需要 FastAPI + uvicorn（``pip install 'tb_risk[api]'``）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from .config import APIConfig

LOGGER = logging.getLogger("tb_risk.api.main")


def _run_uvicorn(config: APIConfig, reload: bool = False) -> None:
    """启动 uvicorn 服务器（懒加载，未安装时给出可读错误）。"""
    try:
        import uvicorn
    except ImportError as e:
        sys.stderr.write(
            "uvicorn 未安装，无法启动 REST API 服务。请执行："
            "pip install 'tb_risk[api]' 或 pip install fastapi uvicorn\n"
        )
        raise SystemExit(1) from e

    # 单机计算优化：默认多 worker（= CPU 核数）并行分摊请求负载。
    # 但 uvicorn 的 reload 模式只支持单 worker（多 worker + reload 会抛错），
    # 开发热重载时强制降为 1，仅开发期受影响。
    workers = config.workers
    if reload and workers != 1:
        LOGGER.warning(
            "热重载模式与多 worker 不兼容，已将 workers 从 %d 降为 1", workers)
        workers = 1

    uvicorn.run(
        "tb_risk.api.app:create_app",
        host=config.host,
        port=config.port,
        workers=workers,
        log_level=config.log_level,
        reload=reload,
        factory=True,
    )


def run_server(config: Optional[APIConfig] = None, reload: bool = False) -> None:
    """启动 REST API 服务（程序化调用入口）。

    参数：
        config: APIConfig，None 时从默认配置加载
        reload: 是否启用热重载（开发用）
    """
    if config is None:
        config = APIConfig.load()
    config.validate()
    LOGGER.info("启动 tb_risk REST API 服务: http://%s:%s", config.host, config.port)
    _run_uvicorn(config, reload=reload)


def main(argv: Optional[list] = None) -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        prog="tb-risk-api",
        description="启动 tb_risk 结核病风险评估 REST API 服务",
    )
    parser.add_argument("--host", default=None, help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=None, help="监听端口（默认 8000）")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--workers", type=int, default=None, help="worker 数")
    parser.add_argument("--reload", action="store_true", help="热重载（开发）")
    args = parser.parse_args(argv)

    config = APIConfig.load(args.config)
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.workers:
        config.workers = args.workers
    run_server(config, reload=args.reload)


if __name__ == "__main__":
    main()