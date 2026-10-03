#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流行病学计算：治疗传染性衰减、多层网络R₀

从 MLRiskPredictor 拆分出的独立函数模块。
所有函数无状态，通过参数传递数据。
"""
import copy
import datetime
import logging
import math
import os
import random

import numpy as np

from ...constants import (
    HOURS_PER_YEAR, BETA_FAMILY_ANNUAL, BETA_SOCIAL_ANNUAL, COMMUNITY_BASELINE_R0,
)

LOGGER = logging.getLogger("tb_risk.ml")

# 条件导入（与主类保持一致）
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 accuracy_score, precision_score, recall_score,
                                 f1_score, brier_score_loss)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

NUMPY_AVAILABLE = True



# ============================================================
# 以下函数从 MLRiskPredictor 的实例方法拆分而来
# 每个函数接收 predictor 实例作为第一个参数
# ============================================================

def calculate_treatment_infectivity_factor(predictor, weeks_since_treatment_start):
        """计算治疗阶段的传染性衰减因子
        
        文献支撑：
        - WHO 2024结核病治疗指南：痰涂片阳性患者治疗2周后传染性下降40%
        - CDC 2023：有效治疗6个月后传染性接近消除
        
        参数：
            weeks_since_treatment_start (int): 治疗开始后的周数
            
        返回：
            float: 传染性衰减因子（0-1之间）
        """
        if weeks_since_treatment_start < 2:
            return predictor.treatment_infectivity_decay['week_0']
        elif weeks_since_treatment_start < 8:
            return predictor.treatment_infectivity_decay['week_2']
        elif weeks_since_treatment_start < 24:
            return predictor.treatment_infectivity_decay['week_8']
        else:
            return predictor.treatment_infectivity_decay['week_24']
    

def calculate_multilayer_network_r0(predictor, family_contacts, social_contacts, community_baseline=COMMUNITY_BASELINE_R0):
        """计算三层网络的基本再生数R₀（下一代矩阵法）

        文献支撑：
        - Diekmann et al. (1990) "On the definition and the computation of
          the basic reproduction ratio R0" - 下一代矩阵法理论基础
        - Van den Driessche & Watmough (2002) "Reproduction numbers and
          sub-threshold endemic equilibria" - 谱半径计算
        - Rozan et al. (2025) 多层网络SIR模型，推导了每层网络的R₀贡献
        - Liu & Zhu (2025) 双层网络流行病阈值公式

        方法：
        构建 3×3 接触矩阵 K，其中 K[i][j] 表示从组 j 到组 i 的
        传播贡献。总 R₀ = ρ(K)，即 K 的谱半径（最大特征值）。

        参数：
            family_contacts (list): 家庭成员接触者列表
            social_contacts (list): 社会接触者列表
            community_baseline (float): 社区层基线传播概率（默认 COMMUNITY_BASELINE_R0=0.05）

        返回：
            dict: 包含各层R₀贡献、接触矩阵、谱半径总R₀的字典
        """
        # 家庭层R₀贡献（组内传播）
        r0_family = 0.0
        for contact in family_contacts:
            exposure_hours = contact.get('cumulative_exposure', 0)
            exposure_years = exposure_hours / HOURS_PER_YEAR
            r0_family += BETA_FAMILY_ANNUAL * exposure_years

        # 社会层R₀贡献（组内传播）
        r0_social = 0.0
        for contact in social_contacts:
            exposure_hours = contact.get('cumulative_exposure', 0)
            exposure_years = exposure_hours / HOURS_PER_YEAR
            r0_social += BETA_SOCIAL_ANNUAL * exposure_years

        # 社区层R₀贡献（固定基线）
        r0_community = community_baseline

        # === 下一代矩阵法：构建接触矩阵 K 并计算谱半径 ===
        # 文献：Diekmann et al. (1990); Van den Driessche & Watmough (2002)
        #
        # 接触矩阵结构（3×3，组别：family=0, social=1, community=2）：
        #   K[i][j] = 从组 j 到组 i 的传播贡献
        #
        # 对角线：组内传播（同质混合假设）
        # 非对角线：跨组传播（使用几何平均传播率 × 平均暴露量）
        #
        # 跨组耦合系数：family↔social 交叉传播约为同质传播的 10%
        # （家庭与社会接触者之间不存在长期共居关系）

        # 组内传播（对角线）
        K_ff = r0_family          # family → family
        K_ss = r0_social          # social → social
        K_cc = r0_community       # community → community

        # 跨组传播（非对角线）—— 基于几何平均传播率
        # family ↔ social：交叉耦合系数 0.1（非共居，接触频率较低）
        cross_coupling_fs = 0.1
        K_fs = cross_coupling_fs * np.sqrt(r0_family * r0_social) if r0_family > 0 and r0_social > 0 else 0.0
        K_sf = cross_coupling_fs * np.sqrt(r0_family * r0_social) if r0_family > 0 and r0_social > 0 else 0.0

        # family ↔ community：社区基线传播通过家庭接触者放大
        cross_coupling_fc = 0.05
        K_fc = cross_coupling_fc * np.sqrt(r0_family * r0_community) if r0_family > 0 else 0.0
        K_cf = cross_coupling_fc * np.sqrt(r0_family * r0_community) if r0_family > 0 else 0.0

        # social ↔ community
        cross_coupling_sc = 0.05
        K_sc = cross_coupling_sc * np.sqrt(r0_social * r0_community) if r0_social > 0 else 0.0
        K_cs = cross_coupling_sc * np.sqrt(r0_social * r0_community) if r0_social > 0 else 0.0

        # 构建 3×3 接触矩阵 K
        K = np.array([
            [K_ff, K_fs, K_fc],
            [K_sf, K_ss, K_sc],
            [K_cf, K_cs, K_cc],
        ], dtype=np.float64)

        # 计算谱半径 ρ(K) = max(|eigenvalues|)
        eigenvalues = np.linalg.eigvals(K)
        r0_spectral = float(np.max(np.abs(eigenvalues)))

        return {
            'R0_family': r0_family,
            'R0_social': r0_social,
            'R0_community': r0_community,
            'R0_total': r0_spectral,
            'contact_matrix': K.tolist(),
            'eigenvalues': [float(v.real) for v in eigenvalues],
            'spectral_radius': r0_spectral,
            'is_above_threshold': r0_spectral > 1.0,
            'method': 'next_generation_matrix_spectral_radius',
        }
    

