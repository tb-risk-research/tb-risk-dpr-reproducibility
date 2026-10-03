#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署与运维友好子包。

对外导出核心 API：
- 离线支持：``offline.status`` / ``offline.check_network`` / ``offline.ensure_cached``
- 热更新：``HotUpdateWatcher`` / ``RuleRegistry`` / ``DEFAULT_RULE_REGISTRY``
- 可读错误：``errors.classify_error`` / ``errors.format_error`` / ``errors.render_error_dialog_text``
- 部署打包：``deploy.pack_distribution``
"""

from . import errors, offline, hotupdate, deploy  # noqa: F401
from .errors import (
    classify_error, format_error, render_error_dialog_text,
    KIND_MISSING_DEP, KIND_CUDA_OOM, KIND_OOM, KIND_IMPORT,
    KIND_FILE, KIND_DISK, KIND_PERMISSION, KIND_NETWORK,
    KIND_CUDA_UNAVAILABLE, KIND_GENERIC,
)
from .offline import (
    check_network, status as offline_status,
    ensure_cached, status_text as offline_status_text,
    rules_ready, model_cache_ready,
)
from .hotupdate import (
    HotUpdateWatcher, RuleRegistry, DEFAULT_RULE_REGISTRY,
    infer_file_kind, DEFAULT_HOTUPDATE_DIR,
)
from .deploy import pack_distribution

__all__ = [
    # 子模块
    "errors", "offline", "hotupdate", "deploy",
    # 错误
    "classify_error", "format_error", "render_error_dialog_text",
    "KIND_MISSING_DEP", "KIND_CUDA_OOM", "KIND_OOM", "KIND_IMPORT",
    "KIND_FILE", "KIND_DISK", "KIND_PERMISSION", "KIND_NETWORK",
    "KIND_CUDA_UNAVAILABLE", "KIND_GENERIC",
    # 离线
    "check_network", "offline_status", "ensure_cached",
    "offline_status_text", "rules_ready", "model_cache_ready",
    # 热更新
    "HotUpdateWatcher", "RuleRegistry", "DEFAULT_RULE_REGISTRY",
    "infer_file_kind", "DEFAULT_HOTUPDATE_DIR",
    # 部署
    "pack_distribution",
]
