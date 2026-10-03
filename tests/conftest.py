#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest 全局配置 — 随机种子固化、可选依赖检测与测试环境初始化

提供统一的依赖检测、pytest标记注册和命令行选项，各测试文件
无需各自重复实现依赖跳过逻辑。

用法：
    # 标记需要torch的测试
    @pytest.mark.requires_torch
    def test_ml_model():
        ...

    # 标记需要tkinter GUI的测试
    @pytest.mark.requires_tkinter
    def test_gui_dialog():
        ...

    # 命令行选项
    pytest --no-gui          # 跳过所有GUI测试
    pytest --no-ml           # 跳过所有ML/torch测试
    pytest --run-slow        # 运行标记为slow的测试（默认跳过）

文献支撑：
- Sandve GK et al. Ten simple rules for reproducible computational research.
  PLoS Comput Biol 9(10):e1003285, 2013.
- Wilson G et al. Best practices for scientific computing.
  PLoS Biol 12(1):e1001745, 2014.
"""

import os
import random
import sys
import warnings

import pytest

# ============================================================================
# 可选依赖统一检测（测试文件可直接从conftest导入，无需各自try/except）
# ============================================================================

# numpy（核心依赖，项目运行必需，这里仍做可用性标记）
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# PyTorch / ML 依赖
try:
    import torch
    HAS_TORCH = True
    PYTORCH_AVAILABLE = True  # 与项目命名统一
except ImportError:
    HAS_TORCH = False
    PYTORCH_AVAILABLE = False
    torch = None

# Tkinter GUI 依赖
try:
    import tkinter as tk
    # 尝试创建root窗口验证显示可用性（CI/无头环境可能导入成功但无法显示）
    try:
        root = tk.Tk()
        root.destroy()
        HAS_TKINTER = True
    except Exception:
        HAS_TKINTER = False
        tk = None
except ImportError:
    HAS_TKINTER = False
    tk = None

# SciPy（MCMC、统计推断需要）
try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy = None

# Matplotlib（图表测试需要）
try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    matplotlib = None

# ReportLab（PDF导出需要）
try:
    import reportlab
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    reportlab = None


# ============================================================================
# pytest 钩子：注册自定义标记和命令行选项
# ============================================================================

def pytest_configure(config):
    """pytest 启动钩子：注册标记、固化随机种子。"""
    # 注册自定义标记（防止pytest警告未知标记）
    config.addinivalue_line("markers", "requires_torch: 需要PyTorch才能运行的测试")
    config.addinivalue_line("markers", "requires_tkinter: 需要Tkinter GUI环境才能运行的测试")
    config.addinivalue_line("markers", "requires_scipy: 需要SciPy才能运行的测试")
    config.addinivalue_line("markers", "requires_matplotlib: 需要Matplotlib才能运行的测试")
    config.addinivalue_line("markers", "requires_reportlab: 需要ReportLab才能运行的测试")
    config.addinivalue_line("markers", "slow: 慢速测试，默认跳过，使用--run-slow启用")
    config.addinivalue_line("markers", "gui: GUI相关测试，使用--no-gui时跳过")
    config.addinivalue_line("markers", "ml: ML/torch相关测试，使用--no-ml时跳过")
    config.addinivalue_line("markers", "integration: 集成测试，需要外部资源")
    config.addinivalue_line("markers", "live: 调用真实外部API的测试（需要网络和密钥）")

    # 固化随机种子
    random.seed(42)
    if HAS_NUMPY:
        np.random.seed(42)
    if HAS_TORCH:
        torch.manual_seed(42)

    # 设置环境变量（影响子进程）
    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("TB_RISK_TEST_MODE", "1")

    # 使用非交互式matplotlib后端（CI/无头环境安全）
    if HAS_MATPLOTLIB:
        matplotlib.use('Agg', force=True)

    # 忽略已知的第三方库警告（减少测试输出噪音）
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    if HAS_TORCH:
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def pytest_unconfigure(config):
    """pytest 清理钩子。"""
    os.environ.pop("TB_RISK_TEST_MODE", None)


def pytest_addoption(parser):
    """添加命令行选项。"""
    parser.addoption(
        "--no-gui", action="store_true", default=False,
        help="跳过所有GUI相关测试（Tkinter界面测试）"
    )
    parser.addoption(
        "--no-ml", action="store_true", default=False,
        help="跳过所有ML/PyTorch相关测试"
    )
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="运行标记为slow的慢速测试（默认跳过）"
    )
    parser.addoption(
        "--run-live", action="store_true", default=False,
        help="运行调用真实外部API的测试（需要网络和API密钥）"
    )


def pytest_collection_modifyitems(config, items):
    """根据命令行选项和依赖可用性自动跳过测试。"""
    skip_no_gui = config.getoption("--no-gui")
    skip_no_ml = config.getoption("--no-ml")
    run_slow = config.getoption("--run-slow")
    run_live = config.getoption("--run-live")

    for item in items:
        markers = {m.name: m for m in item.iter_markers()}

        # --no-gui: 跳过requires_tkinter和gui标记的测试
        if skip_no_gui and ('requires_tkinter' in markers or 'gui' in markers):
            item.add_marker(pytest.mark.skip(reason="--no-gui: 跳过GUI测试"))
            continue

        # --no-ml: 跳过requires_torch和ml标记的测试
        if skip_no_ml and ('requires_torch' in markers or 'ml' in markers):
            item.add_marker(pytest.mark.skip(reason="--no-ml: 跳过ML测试"))
            continue

        # slow标记：默认跳过，--run-slow时运行
        if 'slow' in markers and not run_slow:
            item.add_marker(pytest.mark.skip(reason="慢速测试，使用--run-slow启用"))
            continue

        # live标记：默认跳过，--run-live时运行
        if 'live' in markers and not run_live:
            item.add_marker(pytest.mark.skip(reason="外部API测试，使用--run-live启用"))
            continue

        # 依赖不可用时跳过
        if 'requires_torch' in markers and not HAS_TORCH:
            item.add_marker(pytest.mark.skip(reason="PyTorch不可用"))
        if 'requires_tkinter' in markers and not HAS_TKINTER:
            item.add_marker(pytest.mark.skip(reason="Tkinter不可用"))
        if 'requires_scipy' in markers and not HAS_SCIPY:
            item.add_marker(pytest.mark.skip(reason="SciPy不可用"))
        if 'requires_matplotlib' in markers and not HAS_MATPLOTLIB:
            item.add_marker(pytest.mark.skip(reason="Matplotlib不可用"))
        if 'requires_reportlab' in markers and not HAS_REPORTLAB:
            item.add_marker(pytest.mark.skip(reason="ReportLab不可用"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_random_state():
    """每个测试前彻底重置随机状态，防止测试间状态泄漏。

    某些测试（如 HMC 采样器、ML 训练）会修改 PyTorch 的全局随机状态、
    CUDA 随机状态、autograd 异常检测状态或 torch.set_grad_enabled()，
    导致后续测试在特定随机种子下产生 NaN 值或类型错误。
    本 fixture 确保每个测试从相同的全局状态开始。

    文献: Sandve GK et al. (2013) PLoS Comput Biol 9(10):e1003285
    """
    random.seed(42)
    if HAS_NUMPY:
        np.random.seed(42)
    if HAS_TORCH:
        torch.manual_seed(42)
        # 重置 CUDA 随机状态（即使无 GPU 也无害，防止 GPU 测试泄漏状态）
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.cuda.empty_cache()
        # 确保 autograd 启用（HMC 测试需要梯度）
        torch.set_grad_enabled(True)
        # 清除异常检测状态
        torch.autograd.set_detect_anomaly(False)


# ============================================================================
# 供测试文件导入的便捷常量（保持向后兼容）
# ============================================================================

# 兼容旧命名
TORCH_AVAILABLE = HAS_TORCH
NUMPY_AVAILABLE = HAS_NUMPY
TKINTER_AVAILABLE = HAS_TKINTER
SCIPY_AVAILABLE = HAS_SCIPY
MATPLOTLIB_AVAILABLE = HAS_MATPLOTLIB
REPORTLAB_AVAILABLE = HAS_REPORTLAB
