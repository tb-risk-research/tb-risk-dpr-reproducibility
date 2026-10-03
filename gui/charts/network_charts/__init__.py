#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 图表 - 网络/ML 分析图（NetworkChartMixin）

包含：ML结果展示、传统vs ML评分对比图、SHAP特征重要性图、模型性能对比图、
      校准曲线、个体SHAP解释、因果DAG可视化、反事实干预分析

文献支撑：
- She et al. (2024) 反事实分析框架
- Du et al. (2023) 结核病因果DAG

结构拆分为多个 Mixin 子模块，NetworkChartMixin 通过多继承组合所有 Mixin。
"""

from .._shared import *  # noqa: F401,F403 — 共享导入和常量

from ._ml_results import _MLResultsMixin
from ._ml_charts import _MLChartsMixin
from ._shap_charts import _SHAPChartsMixin
from ._causal_charts import _CausalChartsMixin
from ._counterfactual_charts import _CounterfactualMixin


class NetworkChartMixin(_MLResultsMixin, _MLChartsMixin, _SHAPChartsMixin,
                        _CausalChartsMixin, _CounterfactualMixin):
    """ML 模型分析、因果推断与反事实干预方法"""
    pass
