#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三方向有机联动集成器 — 兼容性 re-export shim

实际实现已下沉至 core/integrator.py，由 RiskAssessmentService 持有实例。
此文件保留以兼容既有 `from tb_risk.integrator import ThreeDirectionIntegrator` 导入。
"""

from .core.integrator import ThreeDirectionIntegrator

__all__ = ['ThreeDirectionIntegrator']
