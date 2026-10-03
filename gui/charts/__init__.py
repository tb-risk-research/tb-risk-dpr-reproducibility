#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 图表模块

ChartMixin 组合了 RiskChartMixin（风险分布图）、SEIRChartMixin（SEIR 动力学图）
和 NetworkChartMixin（ML/网络/因果分析图）。

子模块：
- _shared.py:         共享导入和常量
- risk_charts.py:     风险分布图（RiskChartMixin）
- seir_charts.py:     SEIR 传播动力学图（SEIRChartMixin）
- network_charts.py:  ML/网络/因果分析图（NetworkChartMixin）
"""

from ._shared import *  # noqa: F401,F403 — 提供所有子模块共享的导入和常量
from .risk_charts import RiskChartMixin
from .seir_charts import SEIRChartMixin
from .network_charts import NetworkChartMixin
from .interactive import InteractiveChartMixin
from .advanced_charts import AdvancedChartMixin


class ChartMixin(InteractiveChartMixin, RiskChartMixin, SEIRChartMixin, NetworkChartMixin, AdvancedChartMixin):
    """GUI 图表方法（组合交互基类 + 风险图 + SEIR图 + 网络/ML分析图 + 高级诊断图）"""
    pass