#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练/运行异常 → 可读错误与恢复建议

把底层（torch / sklearn / 文件系统等）抛出的原始异常，映射为面向运维人员的
可读错误信息与可执行的恢复建议，便于用户在 GNN/SEIR/ML 训练失败时快速定位。

设计要点：
- 纯 Python 实现，无强依赖。
- ``classify_error(exc)`` 返回结构化结果：{ kind, message, suggestions }。
- ``format_error(exc)`` 返回拼接好的多行文本，便于直接展示。
"""

import logging
import re

LOGGER = logging.getLogger("tb_risk.ops.errors")


# 异常分类标识常量
KIND_MISSING_DEP = "missing_dependency"   # 依赖缺失
KIND_CUDA_OOM = "cuda_oom"                # GPU 显存不足
KIND_OOM = "oom"                          # 内存不足
KIND_IMPORT = "import_error"              # 导入错误（模块加载失败）
KIND_FILE = "file_not_found"              # 文件不存在
KIND_DISK = "disk_full"                   # 磁盘空间不足
KIND_PERMISSION = "permission_denied"     # 权限不足
KIND_NETWORK = "network_error"            # 网络异常
KIND_CUDA_UNAVAILABLE = "cuda_unavailable"  # CUDA 不可用
KIND_GENERIC = "generic"                  # 通用


def _match(text: str, patterns) -> bool:
    """判断文本是否命中任一（正则/子串）模式。"""
    for p in patterns:
        if isinstance(p, str):
            if p in text:
                return True
        else:
            try:
                if p.search(text):
                    return True
            except re.error:
                continue
    return False


# 各类异常的特征模式（子串或正则）
_CUDA_OOM_PATTERNS = [
    "out of memory",
    "cuda out of memory",
    "CUDA error: out of memory",
    "RuntimeError: CUDA out of memory",
]
_OOM_PATTERNS = [
    "MemoryError",
    "Cannot allocate memory",
    "Unable to allocate",
    "cannot allocate array",
]
_DISK_PATTERNS = [
    "no space left on device",
    "No space left on device",
    "disk full",
    "ENOSPC",
]
_PERMISSION_PATTERNS = [
    "permission denied",
    "Permission denied",
    "Access is denied",
    "EACCES",
]
_FILE_PATTERNS = [
    "No such file or directory",
    "FileNotFoundError",
    "not found",
]
_NETWORK_PATTERNS = [
    "timed out",
    "TimeoutError",
    "connection refused",
    "Connection refused",
    "ConnectionResetError",
    "Network is unreachable",
    "Name or service not known",
    "getaddrinfo failed",
    "SSL",
]
_IMPORT_PATTERNS = [
    "ImportError",
    "ModuleNotFoundError",
    "No module named",
]
_CUDA_UNAVAILABLE_PATTERNS = [
    "CUDA driver",
    "CUDA error",
    "no kernel image is available",
    "CUDA_VISIBLE_DEVICES",
]


def _classify_text(text: str) -> str:
    """基于异常文本归类异常类型。"""
    if _match(text, _CUDA_OOM_PATTERNS):
        return KIND_CUDA_OOM
    if _match(text, _OOM_PATTERNS):
        return KIND_OOM
    if _match(text, _DISK_PATTERNS):
        return KIND_DISK
    if _match(text, _PERMISSION_PATTERNS):
        return KIND_PERMISSION
    if _match(text, _FILE_PATTERNS):
        return KIND_FILE
    if _match(text, _NETWORK_PATTERNS):
        return KIND_NETWORK
    if _match(text, _IMPORT_PATTERNS):
        return KIND_IMPORT
    if _match(text, _CUDA_UNAVAILABLE_PATTERNS):
        return KIND_CUDA_UNAVAILABLE
    return KIND_GENERIC


# 各类型对应的可读消息与恢复建议
_SUGGESTIONS = {
    KIND_CUDA_OOM: (
        "GPU 显存不足，训练中断。",
        [
            "降低 batch_size（在训练参数中减小合成样本数或批次）",
            "关闭混合精度以外的显存占用：尝试减少隐藏层维度 hidden_dim",
            "关闭其他占用显存的程序后重试",
            "若显存始终不足，可在环境变量中设 CUDA_VISIBLE_DEVICES='' 强制使用 CPU",
        ],
    ),
    KIND_OOM: (
        "系统内存不足，训练中断。",
        [
            "减少合成数据样本数 n_samples（如从 2000 降到 500）",
            "关闭其他占用内存的应用程序",
            "增大系统交换分区/虚拟内存后重试",
        ],
    ),
    KIND_MISSING_DEP: (
        "缺少训练所需的第三方依赖。",
        [
            "运行 'tb-risk --install' 自动安装缺失依赖",
            "或手动执行：pip install -r requirements.lock",
            "若为离线环境，请使用离线安装包（见 docs/deployment_ops.md）",
        ],
    ),
    KIND_IMPORT: (
        "模块导入失败，可能是依赖缺失或版本不兼容。",
        [
            "运行 'tb-risk --check' 查看依赖状态",
            "确认 torch / torch-geometric 版本与项目要求匹配",
            "离线环境请使用预打包的 wheel 离线安装",
        ],
    ),
    KIND_FILE: (
        "训练所需的文件或数据不存在。",
        [
            "检查模型/数据文件路径是否存在",
            "确认已生成合成数据或已提供真实数据 CSV",
            "若为模型加载，确认文件扩展名为 .joblib/.pkl/.pth",
        ],
    ),
    KIND_DISK: (
        "磁盘空间不足，无法写入模型或缓存。",
        [
            "清理临时文件（%TEMP% 或 /tmp）与旧模型版本",
            "将 ~/.tb_risk 目录迁移到空间充足的磁盘",
            "删除模型仓库中的历史版本以释放空间",
        ],
    ),
    KIND_PERMISSION: (
        "文件写入/读取权限不足。",
        [
            "以管理员身份运行（Windows）或检查目录写权限（Linux/macOS）",
            "将 ~/.tb_risk 目录权限改为当前用户可写",
        ],
    ),
    KIND_NETWORK: (
        "网络请求失败（可能处于离线环境或网络不稳定）。",
        [
            "检查网络连接；离线环境请使用本地缓存与本地模型",
            "确认是否允许联网上传（数据安全配置默认禁止联网）",
            "本系统核心功能支持完全离线运行",
        ],
    ),
    KIND_CUDA_UNAVAILABLE: (
        "CUDA 运行环境不可用。",
        [
            "确认已安装与显卡匹配的 CUDA 驱动与 PyTorch CUDA 版本",
            "若无 GPU，可正常运行（将自动回退到 CPU 训练，速度较慢）",
            "可设置环境变量 CUDA_VISIBLE_DEVICES='' 强制使用 CPU",
        ],
    ),
    KIND_GENERIC: (
        "训练过程中发生未预期错误。",
        [
            "查看下方详细错误信息与日志",
            "减少训练规模（样本数/轮次）后重试",
            "若持续失败，请记录完整日志并联系运维人员",
        ],
    ),
}


def classify_error(exc: BaseException) -> dict:
    """将异常归类为结构化结果。

    返回：
        dict: { kind, message, suggestions(list), raw(str) }
    """
    text = str(exc)
    # 优先基于异常类型判断缺失依赖（ImportError/ModuleNotFoundError）
    if isinstance(exc, ImportError) or isinstance(exc, ModuleNotFoundError):  # noqa: F821
        kind = KIND_MISSING_DEP
    else:
        kind = _classify_text(text)

    message, suggestions = _SUGGESTIONS.get(
        kind, _SUGGESTIONS[KIND_GENERIC])

    # 对缺失依赖，尝试提取缺失的模块名
    if kind == KIND_MISSING_DEP:
        m = re.search(r"No module named ['\"]([^'\"]+)['\"]", text)
        if m:
            message = f"缺少依赖模块：{m.group(1)}。"

    return {
        "kind": kind,
        "message": message,
        "suggestions": list(suggestions),
        "raw": text,
    }


def format_error(exc: BaseException) -> str:
    """返回面向运维人员的可读多行错误文本。"""
    info = classify_error(exc)
    lines = [
        f"[{info['kind']}] {info['message']}",
        "",
        "恢复建议：",
    ]
    for s in info["suggestions"]:
        lines.append(f"  • {s}")
    lines.append("")
    lines.append("原始错误信息：")
    lines.append("  " + (info["raw"] or "(无)"))
    return "\n".join(lines)


def render_error_dialog_text(exc: BaseException) -> str:
    """渲染用于 GUI 错误弹窗的文本（message 在上，建议作为主内容）。"""
    info = classify_error(exc)
    lines = [info["message"], ""]
    if info["suggestions"]:
        lines.append("恢复建议：")
        lines.extend(f"  • {s}" for s in info["suggestions"])
    return "\n".join(lines)


__all__ = [
    "KIND_MISSING_DEP", "KIND_CUDA_OOM", "KIND_OOM", "KIND_IMPORT",
    "KIND_FILE", "KIND_DISK", "KIND_PERMISSION", "KIND_NETWORK",
    "KIND_CUDA_UNAVAILABLE", "KIND_GENERIC",
    "classify_error", "format_error", "render_error_dialog_text",
]
