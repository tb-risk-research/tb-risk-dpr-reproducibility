#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 向导模块 — 按功能域拆分为子 Mixin"""
from ._helpers import WizardHelpersMixin
from ._layout import WizardLayoutMixin
from ._steps import WizardStepsMixin


class WizardMixin(WizardHelpersMixin, WizardLayoutMixin, WizardStepsMixin):
    """GUI 向导方法混合类 — 聚合所有子 Mixin

    保持 from tb_risk.gui.wizard import WizardMixin 导入路径不变。
    """
    pass


__all__ = ['WizardMixin']
