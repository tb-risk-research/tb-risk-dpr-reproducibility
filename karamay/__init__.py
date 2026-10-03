#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
克拉玛依本土化适配器子模块 (KaramayLocalizer Submodules)

架构模式：策略注册中心 (Strategy Registration Center)
  - 每个子模块作为"插件"向外部暴露参数提供器和规则增强器
  - 核心系统仅需在初始化时注册适配器，无需分支判断
  - 所有差异封装在模块内部，实现高内聚、低耦合

作者：郭臻尧
版本：2.0（深化版）
"""

from .calibrator import LocalEpiCalibrator, KaramayCalibrator
from .oilfield import OilfieldExposureModel
from .idu import IDURiskModule
from .climate import ClimateTBInteraction
from .altitude import AltitudeAdaptation
from .ethnicity import EthnicityPathogenModule
from .policy import LocalPolicyEngine
from .localizer import KaramayLocalizer

__all__ = [
    'LocalEpiCalibrator',
    'KaramayCalibrator',
    'OilfieldExposureModel',
    'IDURiskModule',
    'ClimateTBInteraction',
    'AltitudeAdaptation',
    'EthnicityPathogenModule',
    'LocalPolicyEngine',
    'KaramayLocalizer',
]