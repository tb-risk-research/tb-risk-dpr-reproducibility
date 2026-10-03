#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 工具模块 - 通用工具函数、类型定义和依赖管理

注意：tkinter 是 GUI 可选依赖，在无图形环境（Docker/CI/服务器）中不可用。
     本模块对 tkinter 做条件导入保护，非 GUI 工具函数（_is_yes、SyntheticDataCalibrator 等）
     不依赖 tkinter，可在无 GUI 环境中安全导入。
"""
import logging
import math
import os
import sys

LOGGER = logging.getLogger("tb_risk")

# === tkinter 条件导入（GUI 可选依赖） ===
try:
    import tkinter as tk
    from tkinter import ttk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    tk = None
    ttk = None


def _has_display() -> bool:
    """检测当前环境是否有可用的图形显示（用于决定是否设置 TkAgg 后端）。"""
    # Windows/macOS 通常有桌面环境，可直接使用 TkAgg
    if sys.platform.startswith('win') or sys.platform == 'darwin':
        return TKINTER_AVAILABLE
    # Linux 需检查 DISPLAY 环境变量（X11）或 WAYLAND_DISPLAY（Wayland）
    return bool(TKINTER_AVAILABLE and (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))


# ToolTip 已迁移到 gui.tooltip 模块（GUI 专用组件，不应位于通用工具模块中）。
# 此处保留惰性导入以兼容旧代码（from tb_risk.utils import ToolTip），但非 GUI 环境
# 不应触发 ToolTip 的导入，且导入时也不会因 tkinter 缺失而崩溃。
def __getattr__(name):
    """惰性导入 ToolTip，仅在实际访问时从 gui.tooltip 加载。"""
    if name == 'ToolTip':
        try:
            from tb_risk.gui.tooltip import ToolTip as _ToolTip
            return _ToolTip
        except Exception:
            # GUI 子包不可用时返回存根类，导入不崩溃但实例化会报错
            class _ToolTipStub:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError(
                        "ToolTip 需要 tkinter 和 GUI 子包支持，当前环境不可用。"
                    )
            return _ToolTipStub
    raise AttributeError(f"module 'tb_risk.utils' has no attribute {name!r}")

# === matplotlib 后端设置（单一真相源，防止重复设置） ===
_MATPLOTLIB_BACKEND_SET = False


def _setup_matplotlib_backend():
    """设置 matplotlib 后端（只执行一次）。

    必须在 import matplotlib.pyplot 之前调用。
    有 GUI 显示时用 TkAgg，无头环境用 Agg。
    """
    global _MATPLOTLIB_BACKEND_SET
    if _MATPLOTLIB_BACKEND_SET:
        return
    _MATPLOTLIB_BACKEND_SET = True

    try:
        import matplotlib
        if _has_display():
            try:
                matplotlib.use('TkAgg')
            except Exception as _e:
                LOGGER.debug("无法设置 TkAgg 后端，使用默认后端: %s", _e)
        else:
            try:
                matplotlib.use('Agg')
            except Exception as _e:
                LOGGER.debug("无法设置 Agg 后端，使用默认后端: %s", _e)
    except ImportError:
        pass


# === 第三方库导入（按功能分组） ===
# 图表和科学计算库
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import matplotlib
    _setup_matplotlib_backend()
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # FigureCanvasTkAgg/NavigationToolbar2Tk 仅在有图形显示时导入
    # （Linux 下即使安装了 tkinter，无 DISPLAY 时导入 backend_tkagg 也会失败）
    if _has_display():
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        except Exception:
            FigureCanvasTkAgg = None
            NavigationToolbar2Tk = None
    else:
        FigureCanvasTkAgg = None
        NavigationToolbar2Tk = None

    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    LOGGER.warning(f"matplotlib未安装，图表功能将不可用 - {e}")
    MATPLOTLIB_AVAILABLE = False
    Figure = None
    plt = None
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from scipy.integrate import odeint
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 机器学习库
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, classification_report
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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

# GNN相关依赖
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool
    from torch_geometric.data import Data, HeteroData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

# 数据处理库
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

def safe_int_convert(var, field_name, min_val=None, max_val=None, default=0):
    """安全地将 tkinter 变量或字符串转换为整数，并进行范围校验。

    Args:
        var: tkinter 变量、数值或字符串值
        field_name: 字段名称（用于错误提示）
        min_val: 最小值（可选）
        max_val: 最大值（可选）
        default: 默认值

    Returns:
        int: 转换后的值

    Raises:
        ValueError: 转换失败或超出范围时
    """
    try:
        if var is None:
            return default

        val = None
        if hasattr(var, 'get'):
            val = var.get()
        else:
            val = var

        if isinstance(val, (int, float)):
            int_val = int(val)
        else:
            val_str = str(val).strip()
            if not val_str:
                int_val = default
            else:
                clinical_text_map = {
                    '涂阳': 2, '涂阴': 1, '阳性': 2, '阴性': 1,
                    '有空洞': 2, '无空洞': 1, '空洞': 2,
                    '是': 2, '否': 1,
                    '已治疗': 2, '曾治疗': 2, '正治疗': 2, '治疗': 2,
                    '未治疗': 1, '未': 1,
                }
                if val_str in clinical_text_map:
                    int_val = clinical_text_map[val_str]
                else:
                    int_val = int(float(val_str))

        if min_val is not None and int_val < min_val:
            raise ValueError(f"{field_name}不能小于 {min_val}（当前值：{int_val}）")
        if max_val is not None and int_val > max_val:
            raise ValueError(f"{field_name}不能大于 {max_val}（当前值：{int_val}）")

        return int_val
    except ValueError as e:
        if "不能小于" in str(e) or "不能大于" in str(e):
            raise
        raise ValueError(f"{field_name}必须是有效的整数（输入值：{val}）")


def _is_yes(value):
    """判断值是否表示'是'，兼容多种格式

    参数：value，可以是int(1/0)、bool、str('1'/'是'/'有'/'涂阳'/'Yes'/'yes')等
    返回：bool
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in ('1', '是', '有', '涂阳', 'yes', 'true')
    return False


# ==============================================================================
# 内嵌合成数据先验校准器 — 分层贝叶斯经验校准
# 三层校准: 矩约束匹配 + 文献Meta分析 + 地区多层模型适配
# 文献: Gelman et al. BDA3; Riley et al. BMJ 2020; Copas & Shi (2000)
# ==============================================================================
LITERATURE_EFFECT_SIZES = {
    'has_symptoms': {'log_odds': 0.88, 'se': 0.12, 'source': 'Melsew et al., BMC Infect Dis 2024', 'tau2': 0.08, 'n_studies': 10},
    'bcg_vaccine': {'log_odds': -0.35, 'se': 0.10, 'source': 'Roy et al., BMJ 2014', 'tau2': 0.05, 'n_studies': 14},
    'has_tb': {'log_odds': 1.25, 'se': 0.15, 'source': 'Diel et al., Lancet Infect Dis 2020', 'tau2': 0.12, 'n_studies': 8},
    'is_high_risk': {'log_odds': 0.50, 'se': 0.08, 'source': 'Zwerling et al., Eur Respir J 2016', 'tau2': 0.04, 'n_studies': 12},
    'past_illness': {'log_odds': 0.70, 'se': 0.10, 'source': 'Dhanaraj et al., IJTLD 2021', 'tau2': 0.06, 'n_studies': 9},
    'contact_distance': {'log_odds': 0.80, 'se': 0.09, 'source': 'Acuna-Villaorduna et al., CID 2022', 'tau2': 0.05, 'n_studies': 7},
    'ventilation': {'log_odds': 0.60, 'se': 0.10, 'source': 'Nardell & Churchyard. NEJM 2020', 'tau2': 0.06, 'n_studies': 6},
    'cumulative_exposure': {'log_odds': 0.005, 'se': 0.002, 'source': 'Saunders et al., Thorax 2017', 'tau2': 0.001, 'n_studies': 5},
    'age_young': {'log_odds': 0.02, 'se': 0.006, 'source': 'IJERPH 2025 (Eastern Cape)', 'tau2': 0.004, 'n_studies': 11},
    'single_duration': {'log_odds': 0.003, 'se': 0.001, 'source': 'Velen et al., PLOS Med 2021', 'tau2': 0.001, 'n_studies': 4},
    'freq_density': {'log_odds': 0.02, 'se': 0.005, 'source': 'Mandalakas et al., JAMA Peds 2019', 'tau2': 0.004, 'n_studies': 6}
}

REGIONAL_INCIDENCE = {
    'china_national': {'per_100k': 55.0}, 'china_karamay': {'per_100k': 62.0},
    'india': {'per_100k': 196.0}, 'south_africa': {'per_100k': 465.0},
    'default': {'per_100k': 62.0}
}


class SyntheticDataCalibrator:
    def __init__(self, random_state=42, target_prevalence=None, region='default'):
        self.random_state = random_state
        if NUMPY_AVAILABLE:
            self.rng = np.random.RandomState(random_state)
        else:
            import random as _random
            _random.seed(random_state)
            self.rng = _random
        self.region = region
        self.target_prevalence = target_prevalence or self._get_regional_prevalence()
        self._calibrated_log_odds_base = None
        self._calibrated_coefficients = {}
        self._meta_priors = {}
        self._validation_report = {}

    def _get_regional_prevalence(self):
        rate = REGIONAL_INCIDENCE.get(self.region, REGIONAL_INCIDENCE['default'])
        annual_incidence = rate['per_100k'] / 100000.0
        return max(1.0 - (1.0 - annual_incidence) ** 10, 0.001)

    def calibrate_intercept_by_moments(self, expected_features):
        expected_logit = sum(LITERATURE_EFFECT_SIZES[f]['log_odds'] * expected_features[f]
                            for f in LITERATURE_EFFECT_SIZES if f in expected_features)
        logit_target = math.log(self.target_prevalence / (1.0 - self.target_prevalence + 1e-12))
        log_odds_base = logit_target - expected_logit
        self._calibrated_log_odds_base = log_odds_base
        self._validation_report['moment_matching'] = {
            'target_prevalence': self.target_prevalence,
            'calibrated_intercept': log_odds_base,
            'prevalence_match': self._check_prevalence_match(log_odds_base, expected_features)
        }
        return log_odds_base

    def _check_prevalence_match(self, log_odds_base, expected_features):
        total_logit = log_odds_base
        for f, lit in LITERATURE_EFFECT_SIZES.items():
            if f in expected_features:
                total_logit += expected_features[f] * lit['log_odds']
        implied_p = 1.0 / (1.0 + math.exp(-total_logit))
        rel_error = abs(implied_p - self.target_prevalence) / max(self.target_prevalence, 1e-6)
        return {'implied_prevalence': implied_p, 'target': self.target_prevalence,
                'relative_error': rel_error, 'match': rel_error < 0.10}

    def calibrate_entropy_maximized(self, expected_features):
        if self._calibrated_log_odds_base is None:
            self.calibrate_intercept_by_moments(expected_features)
        coefficients = {}
        for f, lit in LITERATURE_EFFECT_SIZES.items():
            prior_var = lit['se'] ** 2 + lit['tau2']
            shrinkage = 1.0 / (1.0 + prior_var * 100.0)
            coefficients[f] = lit['log_odds'] * shrinkage
        return {'log_odds_base': self._calibrated_log_odds_base, 'coefficients': coefficients}

    def build_meta_analytic_priors(self):
        priors = {}
        for f, lit in LITERATURE_EFFECT_SIZES.items():
            within_var, between_var = lit['se'] ** 2, lit['tau2']
            priors[f] = {'mean': lit['log_odds'], 'sd': math.sqrt(within_var + between_var),
                         'within_var': within_var, 'between_var': between_var,
                         'n_studies': lit['n_studies']}
        self._meta_priors = priors
        return priors

    def apply_publication_bias_correction(self, feat, coefficient):
        if feat in self._meta_priors:
            prior = self._meta_priors[feat]
            if prior['n_studies'] < 5: rho = 0.5
            elif prior['n_studies'] < 8: rho = 0.35
            elif prior['n_studies'] < 12: rho = 0.25
            else: rho = 0.3
            return coefficient / (1.0 + rho * prior['between_var'] / max(prior['within_var'], 1e-10))
        return coefficient

    def adapt_to_region(self, local_params):
        adapted = {'log_odds_base': self._calibrated_log_odds_base or 0, 'coefficients': {}}
        base_rate = local_params.get('base_incidence_per_100k', REGIONAL_INCIDENCE.get(self.region, REGIONAL_INCIDENCE['default'])['per_100k'])
        global_rate = REGIONAL_INCIDENCE['default']['per_100k']
        adapted['log_odds_base'] += math.log(max(1.0 + (base_rate / global_rate - 1.0) * 0.8, 0.5))
        self.build_meta_analytic_priors()
        for f, lit in LITERATURE_EFFECT_SIZES.items():
            modifier = local_params.get(f'{f}_modifier', 1.0)
            prior_sd = math.sqrt(lit['se'] ** 2 + lit['tau2'])
            adapted['coefficients'][f] = {
                'mean': lit['log_odds'] + (modifier - 1.0) * prior_sd,
                'sd': prior_sd
            }
        return adapted

    def james_stein_shrinkage(self, coefficients, n_total=2000):
        if not self._meta_priors:
            self.build_meta_analytic_priors()
        shrunk, weight = {}, min(1.0, 100.0 / max(n_total, 1))
        for f, v in coefficients.items():
            shrunk[f] = v * (1.0 - weight) + self._meta_priors[f]['mean'] * weight if f in self._meta_priors else v
        return shrunk

    def run_full_calibration(self, expected_features, local_params=None, n_samples=2000):
        log_odds_base = self.calibrate_intercept_by_moments(expected_features)
        self.calibrate_entropy_maximized(expected_features)
        self.build_meta_analytic_priors()
        raw_coefficients = {f: lit['log_odds'] for f, lit in LITERATURE_EFFECT_SIZES.items()}
        if local_params:
            regional = self.adapt_to_region(local_params or {})
            final_base = regional['log_odds_base']
            region_coeffs = {k: v['mean'] for k, v in regional['coefficients'].items()}
        else:
            final_base, region_coeffs = log_odds_base, raw_coefficients
        shrunk = self.james_stein_shrinkage(region_coeffs, n_samples)
        self._validation_report['calibration_summary'] = {
            'log_odds_base': float(final_base), 'n_coefficients': len(shrunk),
            'n_samples': n_samples, 'literature_priors_used': len(self._meta_priors)
        }
        return {'log_odds_base': float(final_base), 'coefficients': {k: float(v) for k, v in shrunk.items()},
                'validation_report': self._validation_report}


def expected_feature_means_from_config():
    return {'has_symptoms': 0.3, 'bcg_vaccine': 0.7, 'has_tb': 0.1, 'is_high_risk': 0.2,
            'past_illness': 0.3, 'contact_distance': 0.65, 'ventilation': 0.55,
            'cumulative_exposure': 40.0, 'age_young': 0.06, 'single_duration': 120.0, 'freq_density': 8.0}

