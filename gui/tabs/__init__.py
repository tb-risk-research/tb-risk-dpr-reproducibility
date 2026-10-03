#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 选项卡模块 — 按功能域拆分为子 Mixin"""
from ._basic_info import BasicInfoTabMixin
from ._family import FamilyTabMixin
from ._social import SocialTabMixin
from ._misc import MiscTabMixin


class TabMixin(BasicInfoTabMixin, FamilyTabMixin, SocialTabMixin, MiscTabMixin):
    """GUI 选项卡方法混合类 — 聚合所有子 Mixin

    保持 from tb_risk.gui.tabs import TabMixin 导入路径不变。
    """
    pass


__all__ = ['TabMixin']
