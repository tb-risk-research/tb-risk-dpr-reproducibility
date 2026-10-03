#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API 服务配置。

纯 Python（无 FastAPI 依赖），供 API 应用层与测试复用。
通过环境变量 / 配置文件 / 构造参数三层覆盖，遵循项目"策略注册"与
"外部配置优先，内嵌默认回退"的模式。

配置项：
- 服务监听：host / port / workers / log_level
- 认证：api_keys（API Key 列表，可通过环境变量注入）
- 限流：rate_limit_per_minute（每分钟请求上限，针对每个 API Key）
- 批量：batch_max_records（批量评估单次最大记录数）
- 响应：request_max_body_bytes（请求体大小上限）
- 安全：mask_sensitive（是否对结果中的姓名等字段脱敏）
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("tb_risk.api.config")


# 环境变量前缀，便于部署时统一注入
_ENV_PREFIX = "TB_RISK_API_"

# 默认配置路径（与项目 ~/.tb_risk 惯例一致）
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.expanduser("~"), ".tb_risk", "api_config.json")

# 对外公开的默认配置路径（与 __all__ / __init__ 导出保持一致）
DEFAULT_CONFIG_PATH = _DEFAULT_CONFIG_PATH


def _env(name: str, default: Any = None) -> Any:
    """从环境变量读取，前缀统一为 TB_RISK_API_。"""
    return os.environ.get(f"{_ENV_PREFIX}{name}", default)


def _env_bool(name: str, default: bool = False) -> bool:
    val = _env(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "是")


def _env_int(name: str, default: int) -> int:
    val = _env(name)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        LOGGER.warning("环境变量 %s%s 不是整数，使用默认值 %s", _ENV_PREFIX, name, default)
        return default


def default_workers() -> int:
    """默认 uvicorn worker 数 = CPU 核数（单机计算优化）。

    多进程并行：uvicorn 单进程只能吃满 1 个 CPU 核心，把 workers 从固定 1
    提升到 ``os.cpu_count()`` 可让并发请求分摊到所有核心。为控制内存与进程
    开销，上限设为 32（大核服务器也足够，避免无谓的资源占用）。

    部署时可显式覆盖：
    - 环境变量 ``TB_RISK_API_WORKERS``（或 JSON 配置文件的 ``workers`` 字段）
    - ``--workers N`` 命令行参数

    Returns:
        int: 1..32 的 worker 数
    """
    n = os.cpu_count() or 1
    return max(1, min(int(n), 32))


@dataclass
class APIConfig:
    """REST API 服务配置。

    字段说明见模块 docstring。构造优先级：
    1. 显式传入字段（代码/测试）
    2. 配置文件字段
    3. 环境变量（TB_RISK_API_*）
    4. 内嵌默认值
    """

    host: str = "0.0.0.0"
    port: int = 8000
    # 单机计算优化：默认 = CPU 核数（上限 32），多进程并行分摊请求负载
    workers: int = field(default_factory=default_workers)
    log_level: str = "info"

    # 认证：允许的 API Key 列表（空表示禁用认证，仅限本地/测试）
    api_keys: List[str] = field(default_factory=list)
    # 请求头中携带 API Key 的字段名（默认 X-API-Key）
    api_key_header: str = "X-API-Key"

    # 限流：每个 API Key 每分钟请求上限（0 表示不限流）
    rate_limit_per_minute: int = 60

    # 批量评估单次最大记录数
    batch_max_records: int = 500

    # 请求体大小上限（字节）
    request_max_body_bytes: int = 10 * 1024 * 1024

    # 是否对响应中的敏感字段（姓名等）脱敏
    mask_sensitive: bool = True

    # 是否启用克拉玛依本土化
    enable_karamay: bool = False

    # 是否启用 ML 预测（单患者评估默认）
    default_use_ml: bool = False
    # 是否启用 SEIR 模拟（单患者评估默认）
    default_use_seir: bool = False

    # 额外透传配置（部署时使用）
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    @classmethod
    def defaults(cls) -> "APIConfig":
        """纯默认配置（不读文件/环境变量）。"""
        return cls()

    @classmethod
    def from_env(cls) -> "APIConfig":
        """从环境变量构造（覆盖默认值）。"""
        cfg = cls()
        cfg.host = _env("HOST", cfg.host)
        cfg.port = _env_int("PORT", cfg.port)
        cfg.workers = _env_int("WORKERS", cfg.workers)
        cfg.log_level = _env("LOG_LEVEL", cfg.log_level)
        cfg.api_key_header = _env("KEY_HEADER", cfg.api_key_header)
        cfg.rate_limit_per_minute = _env_int("RATE_LIMIT", cfg.rate_limit_per_minute)
        cfg.batch_max_records = _env_int("BATCH_MAX", cfg.batch_max_records)
        cfg.request_max_body_bytes = _env_int("MAX_BODY_BYTES", cfg.request_max_body_bytes)
        cfg.mask_sensitive = _env_bool("MASK_SENSITIVE", cfg.mask_sensitive)
        cfg.enable_karamay = _env_bool("KARAMAY", cfg.enable_karamay)
        cfg.default_use_ml = _env_bool("USE_ML", cfg.default_use_ml)
        cfg.default_use_seir = _env_bool("USE_SEIR", cfg.default_use_seir)

        # API Key：支持逗号分隔注入（如 TB_RISK_API_KEYS="k1,k2"）
        raw_keys = _env("KEYS")
        if raw_keys:
            cfg.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        return cfg

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "APIConfig":
        """从 JSON 配置文件加载（不存在/失败时回退到默认）。"""
        cfg = cls.from_env()
        path = path or _DEFAULT_CONFIG_PATH
        if not os.path.exists(path):
            return cfg
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                LOGGER.warning("API 配置文件 %s 不是对象，忽略", path)
                return cfg
            return cfg._apply_dict(data)
        except (json.JSONDecodeError, OSError) as e:
            LOGGER.warning("API 配置文件加载失败 %s: %s", path, e)
            return cfg

    def _apply_dict(self, data: Dict[str, Any]) -> "APIConfig":
        """将配置字典应用到当前实例（仅覆盖已知字段）。"""
        for key, value in data.items():
            if key in ("extra",):
                self.extra.update(value or {})
                continue
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra[key] = value
        return self

    @classmethod
    def load(cls, path: Optional[str] = None) -> "APIConfig":
        """统一加载入口：文件 → 环境变量 → 默认。"""
        return cls.from_file(path)

    # ------------------------------------------------------------------
    # 便捷访问
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于状态接口/文档）。"""
        d = {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "log_level": self.log_level,
            "api_key_header": self.api_key_header,
            "auth_enabled": len(self.api_keys) > 0,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "batch_max_records": self.batch_max_records,
            "request_max_body_bytes": self.request_max_body_bytes,
            "mask_sensitive": self.mask_sensitive,
            "enable_karamay": self.enable_karamay,
            "default_use_ml": self.default_use_ml,
            "default_use_seir": self.default_use_seir,
        }
        d.update(self.extra)
        return d

    @property
    def auth_enabled(self) -> bool:
        """是否启用了 API Key 认证（配置了至少一个 Key）。"""
        return len(self.api_keys) > 0

    def validate(self) -> None:
        """校验配置合法性，非法时抛出 ValueError。"""
        if not (0 < self.port <= 65535):
            raise ValueError(f"port 越界: {self.port}")
        if self.workers < 1:
            raise ValueError(f"workers 必须 >= 1: {self.workers}")
        if self.rate_limit_per_minute < 0:
            raise ValueError(f"rate_limit_per_minute 不能为负: {self.rate_limit_per_minute}")
        if self.batch_max_records < 1:
            raise ValueError(f"batch_max_records 必须 >= 1: {self.batch_max_records}")
        if self.request_max_body_bytes < 1024:
            raise ValueError(f"request_max_body_bytes 过小: {self.request_max_body_bytes}")


__all__ = ["APIConfig", "default_workers", "DEFAULT_CONFIG_PATH"]