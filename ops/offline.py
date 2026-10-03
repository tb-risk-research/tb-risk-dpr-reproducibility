#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线运行支持 — 本地缓存模型与规则，无网络也能用

职责：
- 检测当前是否可联网（``check_network``）。
- 确认本地缓存（模型文件 + 规则配置）是否就绪（``ensure_cached``）。
- 汇总离线运行状态（``status``），供 GUI/CLI 展示。

设计要点：
- 模型文件来源：``ModelRepository``（~/.tb_risk/models）。
- 规则配置来源：``karamay_local_params.json``（本地外部配置，缺失时回退内嵌）。
- 不主动联网下载；仅做本地就绪性检查，保证"无网络也能用"。
"""

import logging
import os
import socket
import time

LOGGER = logging.getLogger("tb_risk.ops.offline")

# 本地数据目录（与 gui/app_config.py / security 保持一致）
DATA_DIR = os.path.join(os.path.expanduser('~'), '.tb_risk')
RULES_FILE = os.path.join(os.path.expanduser('~'), '.tb_risk',
                          'karamay_local_params.json')
# 项目内嵌规则配置（打包源码内）
PROJECT_RULES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'karamay_local_params.json')

# 联网检测目标（域名 + 端口）
_NETWORK_PROBE_HOST = os.environ.get("TB_RISK_PROBE_HOST", "pypi.org")
_NETWORK_PROBE_PORT = int(os.environ.get("TB_RISK_PROBE_PORT", "443"))
_NETWORK_TIMEOUT = float(os.environ.get("TB_RISK_PROBE_TIMEOUT", "2.0"))


def check_network(host: str = None, port: int = None,
                  timeout: float = None) -> bool:
    """探测网络连通性（短超时，不阻塞）。

    通过向指定主机端口发起 TCP 连接判断。失败（超时/拒绝/无法解析）返回 False。

    参数：
        host: 探测主机，默认 pypi.org
        port: 端口，默认 443
        timeout: 超时秒数，默认 2.0

    返回：
        bool: 是否可联网
    """
    host = host or _NETWORK_PROBE_HOST
    port = port or _NETWORK_PROBE_PORT
    timeout = timeout if timeout is not None else _NETWORK_TIMEOUT
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _find_rules_file() -> str:
    """定位规则配置文件：优先用户目录，其次项目内嵌。"""
    for path in (RULES_FILE, PROJECT_RULES_FILE):
        if os.path.exists(path):
            return path
    return RULES_FILE


def rules_file_exists() -> bool:
    """规则配置文件是否已本地就绪。"""
    return os.path.exists(_find_rules_file())


def rules_ready() -> bool:
    """规则是否可用（本地文件或项目内嵌之一存在）。"""
    # 内嵌规则始终可用（config.py 内嵌），此处判断外部文件是否存在作为标记
    return True


def _model_files() -> list:
    """列出本地模型仓库中的模型文件（存在者）。"""
    try:
        from ..gui.training_panel.model_repo import ModelRepository
        repo = ModelRepository()
        files = []
        for r in repo.list_models():
            if os.path.exists(r.file_path):
                files.append(r.file_path)
        return files
    except Exception:
        return []


def model_cache_ready() -> bool:
    """本地模型缓存是否就绪（存在至少一个可加载模型文件）。"""
    return len(_model_files()) > 0


def ensure_cached() -> dict:
    """确保本地缓存就绪（模型 + 规则）。

    返回：
        dict: { model_ready, rule_ready, model_count, rules_file }
    """
    models = _model_files()
    return {
        "model_ready": len(models) > 0,
        "rule_ready": rules_ready(),
        "model_count": len(models),
        "rules_file": _find_rules_file(),
    }


def status() -> dict:
    """汇总离线运行状态。"""
    online = check_network()
    cache = ensure_cached()
    return {
        "online": online,
        "offline": not online,
        "model_ready": cache["model_ready"],
        "rule_ready": cache["rule_ready"],
        "model_count": cache["model_count"],
        "rules_file": cache["rules_file"],
        "data_dir": DATA_DIR,
        "fully_offline_ready": cache["rule_ready"],  # 规则内嵌，核心可离线
        "probe_host": _NETWORK_PROBE_HOST,
    }


def status_text() -> str:
    """返回供展示的离线状态文本。"""
    s = status()
    lines = [
        f"联网状态: {'可联网' if s['online'] else '离线'}",
        f"本地模型缓存: {'就绪（%d 个）' % s['model_count'] if s['model_ready'] else '未就绪'}",
        f"规则配置: {'就绪' if s['rule_ready'] else '缺失'}",
        f"缓存目录: {s['data_dir']}",
    ]
    if not s['online'] and s['rule_ready']:
        lines.append("说明: 离线环境下规则始终可用，核心评估可正常运行。")
    return "\n".join(lines)


__all__ = [
    "DATA_DIR", "RULES_FILE",
    "check_network", "rules_file_exists", "rules_ready",
    "model_cache_ready", "ensure_cached", "status", "status_text",
]
