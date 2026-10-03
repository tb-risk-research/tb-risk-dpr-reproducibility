#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因果推断框架（Causal Inference Framework）

将 tb_risk 的可解释性从 SHAP 相关性分析升级为因果推断：
1. DAG（有向无环图）：显式建模变量间因果关系结构
2. PSM（倾向评分匹配）：评估干预的因果效应
3. 反事实分析：回答 what-if 干预问题
4. do-calculus：将观察关联转化为干预效应估计

四组件协同：DAG 提供结构假设 → do-calculus 基于结构识别调整集 →
PSM 与反事实分别从匹配与预测角度估计干预效应。

文献支撑：
- Pearl J. Causality, 2009. 因果阶梯（观察→干预→反事实）
- Hernán MA, Robins JM. Causal Inference: What If, 2020.
- Rosenbaum PR, Rubin DB. Biometrika, 1983.（PSM）
- Robins JM. Math Model, 1986.（g-computation）
"""
from .dag import CausalDAG, NodeType
from .tb_dag import build_tb_dag, get_default_adjustment_set
from .psm import PropensityScoreMatcher
from .counterfactual import CounterfactualAnalyzer
from .do_calculus import DoCalculusEstimator

__all__ = [
    'CausalDAG',
    'NodeType',
    'build_tb_dag',
    'get_default_adjustment_set',
    'PropensityScoreMatcher',
    'CounterfactualAnalyzer',
    'DoCalculusEstimator',
]
