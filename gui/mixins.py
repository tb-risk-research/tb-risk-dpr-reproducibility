#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 聚合混合类 - 协调所有子Mixin"""

from .base import BaseMixin
from .tabs import TabMixin
from .charts import ChartMixin
from .training_panel import TrainingMixin
from .import_panel import DataImportMixin
from .results_panel import ResultsMixin
from .deploy_panel import DeployPanelMixin
from .wizard import WizardMixin


class GUIMixin(
    BaseMixin,
    TabMixin,
    ChartMixin,
    TrainingMixin,
    DataImportMixin,
    ResultsMixin,
    DeployPanelMixin,
    WizardMixin,
):
    """GUI界面方法混合类 — 聚合所有子Mixin

    所有方法分散在子模块中，通过多重继承组合。
    保持 from tb_risk.gui.mixins import GUIMixin 导入路径不变。
    """
    pass