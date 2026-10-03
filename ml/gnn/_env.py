#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN 深度学习环境分级检测与降级指引（问题三）。

统一判断当前进程运行在哪个 GNN 降级级别，供启动检测、GUI 状态提示与
推理回退决策使用，避免各处重复编写 try/except 与硬编码文案。

分级（tier）：
    tier 0 — 完整 GNN：torch + torch_geometric 均可用，可训练/加载完整 GNN。
    tier 1 — 仅 torch：无 PyG，但可复用 _layers.HeteroGATLayer 等手写层
             运行 HeteroTimeVaryingGNN / MultimodalGNN（torch 即可）。
    tier 2 — 仅 NumPy：无深度学习环境，使用 _numpy_gnp.NumPyGraphRiskPredictor
             做轻量图卷积（均值聚合 + PageRank 式传播），保证 GNN 分支始终有输出。
    tier 3 — 不可用：连 NumPy 都缺失（极端情况），GNN 分支退化为占位实现。

依赖声明（pyproject.toml）：
    pip install -e '.[gnn]'   # 安装 torch + torch-geometric → tier 0
    pip install torch         # 仅安装 torch          → tier 1
    无需安装任何深度学习包     # 内置 NumPy 回退         → tier 2
"""

try:
    import numpy as np  # noqa: F401  （numpy 为项目核心依赖，此处仅探测可用性）
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch  # noqa: F401
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import torch_geometric  # noqa: F401
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


# tier 编号 → 名称 / 说明 / 升级命令
GNN_TIER_INFO = {
    0: {
        'name': '完整 GNN',
        'description': 'torch + torch_geometric 均可用，可训练与加载完整图神经网络模型。',
    },
    1: {
        'name': '仅 torch（手写层）',
        'description': '无 torch_geometric；HeteroTimeVaryingGNN / MultimodalGNN 通过手写 '
                       'HeteroGATLayer 运行，但依赖 PyG 的框架模型（SEIRInformedGNN 等）不可用。',
    },
    2: {
        'name': '仅 NumPy（轻量图卷积）',
        'description': '无深度学习环境；GNN 分支回退到 NumPyGraphRiskPredictor 的轻量图卷积，'
                       '仍能利用接触网络结构输出风险，但精度低于完整 GNN。',
    },
    3: {
        'name': '不可用',
        'description': 'NumPy 不可用（极端环境）；GNN 分支退化为占位实现，由 ML + SEIR 集成兜底。',
    },
}

# tier → 建议安装命令
GNN_INSTALL_COMMANDS = {
    0: None,  # 已是最佳状态，无需升级
    1: "pip install -e '.[gnn]'",
    2: "pip install torch",
    3: "pip install numpy",
}


def gnn_degradation_level():
    """返回当前 GNN 环境的降级级别（0-3，数值越大能力越弱）。"""
    if PYTORCH_AVAILABLE and PYG_AVAILABLE:
        return 0
    if PYTORCH_AVAILABLE:
        return 1
    if NUMPY_AVAILABLE:
        return 2
    return 3


def gnn_degradation_name(level=None):
    """返回降级级别的中文名。level 缺省时自动检测。"""
    if level is None:
        level = gnn_degradation_level()
    return GNN_TIER_INFO.get(level, GNN_TIER_INFO[3])['name']


def gnn_tier_description(level=None):
    """返回降级级别的详细说明。level 缺省时自动检测。"""
    if level is None:
        level = gnn_degradation_level()
    return GNN_TIER_INFO.get(level, GNN_TIER_INFO[3])['description']


def gnn_install_guidance(level=None):
    """返回将当前级别升级到更高级别的安装指引（tier 0 返回 None）。"""
    if level is None:
        level = gnn_degradation_level()
    return GNN_INSTALL_COMMANDS.get(level)


def gnn_env_status():
    """汇总当前 GNN 环境状态（供启动检测与 GUI 展示）。

    返回：
        dict: {
            'level': int（0-3）,
            'name': str,
            'description': str,
            'torch_available': bool,
            'pyg_available': bool,
            'numpy_available': bool,
            'install_command': str | None,
            'full_gnn_ready': bool,   # 是否可训练/加载完整 GNN
            'branch_available': bool,  # GNN 分支是否有可用输出（tier 0-2）
        }
    """
    level = gnn_degradation_level()
    return {
        'level': level,
        'name': gnn_degradation_name(level),
        'description': gnn_tier_description(level),
        'torch_available': PYTORCH_AVAILABLE,
        'pyg_available': PYG_AVAILABLE,
        'numpy_available': NUMPY_AVAILABLE,
        'install_command': gnn_install_guidance(level),
        'full_gnn_ready': level == 0,
        'branch_available': level <= 2,
    }


def gnn_env_status_text():
    """人类可读的 GNN 环境状态文本（供启动提示）。"""
    st = gnn_env_status()
    line = "GNN 环境: [级别{}] {} — {}".format(st['level'], st['name'], st['description'])
    if st['install_command']:
        line += "\n  升级指引: {}".format(st['install_command'])
    return line
