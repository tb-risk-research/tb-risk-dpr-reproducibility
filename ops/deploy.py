#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署打包 — 构建可分发的离线/在线安装包

把项目源码、依赖锁文件、Dockerfile、部署脚本、本地模型与配置一并打包成
zip 压缩包，供运维人员一键部署（在线环境可执行 Dockerfile / deploy.sh，
离线环境使用预打包产物离线安装）。

设计要点：
- 仅打包白名单内的文件/目录，避免混入 .venv、__pycache__、测试缓存等。
- 模型文件来自本地模型仓库（~/.tb_risk/models），随包分发以便离线直接加载。
- 纯标准库实现（zipfile/shutil），无第三方依赖。
"""

import logging
import os
import shutil
import tempfile
import zipfile

LOGGER = logging.getLogger("tb_risk.ops.deploy")

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 打包白名单：目录（递归）或文件（相对项目根）
PACKAGE_INCLUDE = [
    # 源码包
    "tb_risk",  # 主包（若存在）
    "ai", "cli", "core", "data_io", "export_utils", "gui",
    "health_interop", "karamay", "ml", "ops", "persistence",
    "scoring", "seir", "security", "validation",
    # 顶层模块
    "assessment.py", "config.py", "constants.py", "diagnose.py",
    "integrator.py", "io_utils.py", "main.py", "schemas.py", "utils.py",
    "requirements.lock", "pyproject.toml", "Dockerfile", "README.md",
    # 内嵌规则配置
    "karamay_local_params.json",
]

# 需要排除的目录名（os.walk 剪枝，避免遍历 .venv/.git 等大目录）
EXCLUDE_DIRS = {
    "__pycache__", ".pytest_cache", ".hypothesis", "catboost_info",
    "diagnostic_log.txt", ".venv", ".git", "tb_risk.egg-info",
    ".github", ".idea", "node_modules",
}


def _collect_local_models() -> list:
    """收集本地模型仓库中的模型文件路径。"""
    try:
        from ..gui.training_panel.model_repo import ModelRepository
        repo = ModelRepository()
        return [r.file_path for r in repo.list_models()
                if os.path.exists(r.file_path)]
    except Exception:
        return []


def _is_included(rel_path: str) -> bool:
    """判断相对路径是否在打包白名单内。

    统一使用 '/' 归一化相对路径与白名单项，保证跨平台（Windows 使用 '\\'）。
    """
    norm = rel_path.replace('\\', '/')
    for item in PACKAGE_INCLUDE:
        item_norm = item.replace('\\', '/')
        if item_norm.endswith('/') or '.' not in item.split('/')[-1]:
            # 目录白名单：路径开头匹配
            if norm == item_norm or norm.startswith(item_norm + '/'):
                return True
        else:
            # 文件白名单：精确匹配
            if norm == item_norm:
                return True
    return False


def _should_exclude(rel_path: str) -> bool:
    """排除目录判断。"""
    parts = rel_path.replace('\\', '/').split('/')
    return any(p in EXCLUDE_DIRS for p in parts) or \
        any(p.endswith('.pyc') for p in parts)


def pack_distribution(out_path: str = None, include_models: bool = True,
                      include_config: bool = True) -> str:
    """打包可分发的安装包（zip）。

    参数：
        out_path: 输出 zip 路径；None 时输出到 ~/.tb_risk/dist/tb_risk_dist.zip
        include_models: 是否包含本地模型仓库中的模型文件
        include_config: 是否包含本地规则配置（karamay_local_params.json）

    返回：
        str: 生成的压缩包路径
    """
    if out_path is None:
        dist_dir = os.path.join(os.path.expanduser('~'), '.tb_risk', 'dist')
        os.makedirs(dist_dir, exist_ok=True)
        out_path = os.path.join(dist_dir, "tb_risk_dist.zip")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # 收集要打包的模型文件（复制到临时目录）
    tmp_models = None
    model_files = _collect_local_models() if include_models else []

    with tempfile.TemporaryDirectory() as tmp:
        model_dir_in_pkg = "models"
        if model_files:
            tmp_models = os.path.join(tmp, model_dir_in_pkg)
            os.makedirs(tmp_models, exist_ok=True)
            for m in model_files:
                try:
                    shutil.copy2(m, tmp_models)
                except OSError as e:
                    LOGGER.debug("复制模型 %s 失败: %s", m, e)

        n_files = 0
        with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 打包源码白名单
            for root, dirs, files in os.walk(PROJECT_ROOT):
                # 剪枝排除目录
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for fname in files:
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, PROJECT_ROOT)
                    if not _is_included(rel) or _should_exclude(rel):
                        continue
                    zf.write(full, os.path.join("tb_risk_app", rel))
                    n_files += 1

            # 打包模型文件
            if tmp_models and os.path.isdir(tmp_models):
                for fname in os.listdir(tmp_models):
                    zf.write(os.path.join(tmp_models, fname),
                             os.path.join("tb_risk_app", "models", fname))
                    n_files += 1

            # 打包本地规则配置（外部覆盖文件）
            if include_config:
                rules_path = os.path.join(os.path.expanduser('~'),
                                          '.tb_risk',
                                          'karamay_local_params.json')
                if os.path.exists(rules_path):
                    zf.write(rules_path,
                             os.path.join("tb_risk_app", "local_config",
                                          "karamay_local_params.json"))
                    n_files += 1

        LOGGER.info("打包完成: %s (%d 个文件)", out_path, n_files)
    return out_path


__all__ = ["PROJECT_ROOT", "pack_distribution", "_collect_local_models"]
