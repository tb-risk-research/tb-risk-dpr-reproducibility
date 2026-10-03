#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TB_Risk_Assessment - 结核病风险评估主类"""
import copy
import csv
import datetime
import json
import logging
import math
import os
import queue
import random
import re
import sys
import threading

from .utils import TKINTER_AVAILABLE  # tkinter 可用性标志（单一真相源）

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:
    tk = None
    ttk = None
    messagebox = None
    filedialog = None

# numpy/matplotlib 等可选依赖统一从 utils 导入（单一真相源，避免重复设置后端）
from .utils import (
    np, NUMPY_AVAILABLE,
    Figure, plt, FigureCanvasTkAgg, NavigationToolbar2Tk, MATPLOTLIB_AVAILABLE,
)
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
try:
    from torch_geometric.nn import GATConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    GATConv = None
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from . import LOGGER
from .utils import _is_yes  # TKINTER_AVAILABLE 已在上方导入
from .gui.tooltip import ToolTip
from .constants import (
    PAST_TB_BONUS as _PAST_TB_BONUS,
    CONTACT_TYPE_FAMILY, CONTACT_TYPE_SOCIAL,
    MODEL_TYPE_ENSEMBLE, MODEL_TYPE_LOGISTIC, MODEL_TYPE_RF,
    MODEL_TYPE_SVM, MODEL_TYPE_XGB, MODEL_TYPE_GNN,
    SCENARIO_TYPE_LIVING, SCENARIO_TYPE_WORK, SCENARIO_TYPE_TRIP,
    SCENARIO_TYPE_PARTY, SCENARIO_TYPE_CUSTOM,
    BOOL_YES, BOOL_NO,
    DISTANCE_VERY_CLOSE, DISTANCE_CLOSE, DISTANCE_MEDIUM, DISTANCE_FAR, DISTANCE_DISTANT,
    DISTANCE_VERY_CLOSE_CN, DISTANCE_CLOSE_CN, DISTANCE_MEDIUM_CN,
    DISTANCE_FAR_CN, DISTANCE_DISTANT_CN,
    SETTING_CROWDED, SETTING_CLOSED, SETTING_GENERAL, SETTING_OUTDOOR,
    SETTING_CROWDED_CN, SETTING_CLOSED_CN, SETTING_GENERAL_CN, SETTING_OUTDOOR_CN,
    ILLNESS_TYPE_HIV, ILLNESS_TYPE_DIABETES, ILLNESS_TYPE_IMMUNOSUPPRESSIVE,
    ILLNESS_TYPE_OTHER, ILLNESS_TYPE_NONE,
    # Phase 1: 业务参数统一到 constants.py
    BASE_WEIGHTS, PATIENT_TYPE_ADJUSTMENTS, TREATMENT_INFECTIVITY_FACTORS,
    BCG_PROTECTION_PARAMS, NO_BCG_BONUS as _NO_BCG_BONUS, LATENT_BASELINE as _LATENT_BASELINE,
    SYMPTOMS_SEVERE_BONUS, COUGH_FREQ_HIGH_BONUS, COUGH_FREQ_MEDIUM_BONUS,
    COUGH_FREQ_LOW_BONUS, AGE_YOUNG_BONUS, AGE_ELDERLY_BONUS,
    DELAY_DAYS_SEVERE_BONUS, DELAY_DAYS_MEDIUM_BONUS, DELAY_DAYS_MILD_BONUS,
    SYMPTOMS_PRESENT_BONUS as _SYMPTOMS_PRESENT_BONUS,
    MARK_HIGH_RISK_THRESHOLD as _MARK_HIGH_RISK_THRESHOLD,
    RISK_THRESHOLDS as _RISK_THRESHOLDS,
    DISEASE_PROB_VERY_HIGH as _DISEASE_PROB_VERY_HIGH,
    DISEASE_PROB_HIGH as _DISEASE_PROB_HIGH,
    DISEASE_PROB_MEDIUM as _DISEASE_PROB_MEDIUM,
    # Section V: 验证常量统一从 constants.py 导入（单一真相源）
    MIN_AGE as _MIN_AGE, MAX_AGE as _MAX_AGE,
    MIN_VENTILATION as _MIN_VENTILATION, MAX_VENTILATION as _MAX_VENTILATION,
    MAX_SINGLE_DURATION_MINUTES as _MAX_SINGLE_DURATION_MINUTES,
    MAX_TIME_SPAN_WEEKS as _MAX_TIME_SPAN_WEEKS,
    MAX_FAMILY_MEMBERS as _MAX_FAMILY_MEMBERS,
    MAX_SOCIAL_CONTACTS as _MAX_SOCIAL_CONTACTS,
    MAX_COUGH_FREQ as _MAX_COUGH_FREQ,
    MAX_TREATMENT_DURATION_MONTHS as _MAX_TREATMENT_DURATION_MONTHS,
    MAX_DELAY_DAYS as _MAX_DELAY_DAYS,
    MAX_FREQ_DENSITY as _MAX_FREQ_DENSITY,
    DEFAULT_FAMILY_FREQ as _DEFAULT_FAMILY_FREQ,
    DEFAULT_SOCIAL_FREQ as _DEFAULT_SOCIAL_FREQ,
)
from .scoring.engine import ScoringEngine
# 问题十-2：MLRiskPredictor 不再由 GUI 直接导入实例化，改由
# RiskAssessmentService.ml_predictor 暴露共享实例（单一真值源）
from .gui.dialog import ContactEditDialog
from .persistence.mixins import IOMixin
from .gui.mixins import GUIMixin
from .gui.ai_panel import AiAssistantMixin
from .gui.undo import UndoManager
from .gui.app_config import AppConfig, AutoSaveManager

# core 业务层导入（Phase 1.4 — 委托业务逻辑到 core/）
from .core import (
    RiskAssessmentService,
    calculate_cumulative_exposure,
    convert_chinese_to_value,
)

# data_io 统一常量（Phase 0 集成 — 消除重复定义）
try:
    from .data_io.conf import (
        DEFAULT_VALUES as _DV,
        DISTANCE_TEXT_MAP,
        SETTING_TEXT_MAP,
        SCENE_DEFAULTS as _SD,
        FIELD_RANGES,
        MISSING_VALUE_MARKER,
    )
except ImportError:
    _DV = {}
    DISTANCE_TEXT_MAP = {}
    SETTING_TEXT_MAP = {}
    _SD = {}
    FIELD_RANGES = {}
    MISSING_VALUE_MARKER = -999

# 克拉玛依本土化适配器（可选依赖）
try:
    from .karamay.localizer import KaramayLocalizer
    KARAMAY_LOCALIZER_AVAILABLE = True
except ImportError:
    KaramayLocalizer = None
    KARAMAY_LOCALIZER_AVAILABLE = False

# 克拉玛依验证器（可选依赖）
try:
    from .validation.validator import KaramayValidator
    KARAMAY_VALIDATOR_AVAILABLE = True
except ImportError:
    KaramayValidator = None
    KARAMAY_VALIDATOR_AVAILABLE = False

# SEIR 模型（可选依赖，需 numpy）
try:
    from .seir.stochastic import StochasticSEIRModel
    from .seir.bayesian import BayesianSEIRInference
    from .seir.posterior import PosteriorDrivenInfectivity
    SEIR_AVAILABLE = True
except ImportError:
    StochasticSEIRModel = None
    BayesianSEIRInference = None
    PosteriorDrivenInfectivity = None
    SEIR_AVAILABLE = False


class TB_Risk_Assessment(IOMixin, GUIMixin, AiAssistantMixin):
    """结核病风险评估类，用于分析结核病患者的家庭社会关系网络中的潜在患者风险"""
    
    # 类常量 — 从 constants.py 单一真值源引用（向后兼容，支持 TB_Risk_Assessment.CONST 访问）
    CONTACT_TYPE_FAMILY = CONTACT_TYPE_FAMILY
    CONTACT_TYPE_SOCIAL = CONTACT_TYPE_SOCIAL
    MODEL_TYPE_ENSEMBLE = MODEL_TYPE_ENSEMBLE
    MODEL_TYPE_LOGISTIC = MODEL_TYPE_LOGISTIC
    MODEL_TYPE_RF = MODEL_TYPE_RF
    MODEL_TYPE_SVM = MODEL_TYPE_SVM
    MODEL_TYPE_XGB = MODEL_TYPE_XGB
    MODEL_TYPE_GNN = MODEL_TYPE_GNN
    SCENARIO_TYPE_LIVING = SCENARIO_TYPE_LIVING
    SCENARIO_TYPE_WORK = SCENARIO_TYPE_WORK
    SCENARIO_TYPE_TRIP = SCENARIO_TYPE_TRIP
    SCENARIO_TYPE_PARTY = SCENARIO_TYPE_PARTY
    SCENARIO_TYPE_CUSTOM = SCENARIO_TYPE_CUSTOM
    BOOL_YES = BOOL_YES
    BOOL_NO = BOOL_NO
    DISTANCE_VERY_CLOSE = DISTANCE_VERY_CLOSE
    DISTANCE_CLOSE = DISTANCE_CLOSE
    DISTANCE_MEDIUM = DISTANCE_MEDIUM
    DISTANCE_FAR = DISTANCE_FAR
    DISTANCE_DISTANT = DISTANCE_DISTANT
    DISTANCE_VERY_CLOSE_CN = DISTANCE_VERY_CLOSE_CN
    DISTANCE_CLOSE_CN = DISTANCE_CLOSE_CN
    DISTANCE_MEDIUM_CN = DISTANCE_MEDIUM_CN
    DISTANCE_FAR_CN = DISTANCE_FAR_CN
    DISTANCE_DISTANT_CN = DISTANCE_DISTANT_CN
    SETTING_CROWDED = SETTING_CROWDED
    SETTING_CLOSED = SETTING_CLOSED
    SETTING_GENERAL = SETTING_GENERAL
    SETTING_OUTDOOR = SETTING_OUTDOOR
    SETTING_CROWDED_CN = SETTING_CROWDED_CN
    SETTING_CLOSED_CN = SETTING_CLOSED_CN
    SETTING_GENERAL_CN = SETTING_GENERAL_CN
    SETTING_OUTDOOR_CN = SETTING_OUTDOOR_CN
    ILLNESS_TYPE_HIV = ILLNESS_TYPE_HIV
    ILLNESS_TYPE_DIABETES = ILLNESS_TYPE_DIABETES
    ILLNESS_TYPE_IMMUNOSUPPRESSIVE = ILLNESS_TYPE_IMMUNOSUPPRESSIVE
    ILLNESS_TYPE_OTHER = ILLNESS_TYPE_OTHER
    ILLNESS_TYPE_NONE = ILLNESS_TYPE_NONE
    
    # 评分与阈值常量
    # 注：原 SIGMOID_COEFFICIENT=0.5 已删除（死代码且与 ScoringEngine.SIGMOID_COEFF=0.3
    # 冲突，易误导）。sigmoid 参数统一以 ScoringEngine 类属性为单一真值源。
    # 已移除未使用的 MAX_RISK_MULTIPLIER=10.0 和 SNC_MAX_CONTACT_COUNT=100（死代码）。
    INFECTION_PROB_MAX = ScoringEngine.SIGMOID_CAP  # 感染概率上限，与 ScoringEngine 单一真值源保持一致
    FTD_HIGH_THRESHOLD = 30  # 家庭传播延迟高风险阈值（天），延迟越长风险越高
    FTD_MEDIUM_THRESHOLD = 14  # 家庭传播延迟中风险阈值（天）
    # Section V: 验证常量从 constants.py 单一真相源引用（消除重复定义）
    MIN_AGE = _MIN_AGE
    MAX_AGE = _MAX_AGE
    MIN_VENTILATION = _MIN_VENTILATION
    MAX_VENTILATION = _MAX_VENTILATION
    MAX_SINGLE_DURATION_MINUTES = _MAX_SINGLE_DURATION_MINUTES
    MAX_TIME_SPAN_WEEKS = _MAX_TIME_SPAN_WEEKS
    MAX_FAMILY_MEMBERS = _MAX_FAMILY_MEMBERS
    MAX_SOCIAL_CONTACTS = _MAX_SOCIAL_CONTACTS
    MAX_COUGH_FREQ = _MAX_COUGH_FREQ
    MAX_TREATMENT_DURATION_MONTHS = _MAX_TREATMENT_DURATION_MONTHS
    MAX_DELAY_DAYS = _MAX_DELAY_DAYS
    MAX_FREQ_DENSITY = _MAX_FREQ_DENSITY
    DEFAULT_FAMILY_FREQ = _DEFAULT_FAMILY_FREQ
    DEFAULT_SOCIAL_FREQ = _DEFAULT_SOCIAL_FREQ
    
    # 风险评分调整常量（从 constants.py 单一真值源引用）
    SYMPTOMS_SEVERE_BONUS = SYMPTOMS_SEVERE_BONUS
    COUGH_FREQ_HIGH_BONUS = COUGH_FREQ_HIGH_BONUS
    COUGH_FREQ_MEDIUM_BONUS = COUGH_FREQ_MEDIUM_BONUS
    COUGH_FREQ_LOW_BONUS = COUGH_FREQ_LOW_BONUS
    AGE_YOUNG_BONUS = AGE_YOUNG_BONUS
    AGE_ELDERLY_BONUS = AGE_ELDERLY_BONUS
    DELAY_DAYS_SEVERE_BONUS = DELAY_DAYS_SEVERE_BONUS
    DELAY_DAYS_MEDIUM_BONUS = DELAY_DAYS_MEDIUM_BONUS
    DELAY_DAYS_MILD_BONUS = DELAY_DAYS_MILD_BONUS
    NO_BCG_VACCINE_BONUS = _NO_BCG_BONUS  # 未接种BCG加分（单一真值源：constants.NO_BCG_BONUS）
    SYMPTOMS_PRESENT_BONUS = _SYMPTOMS_PRESENT_BONUS  # 有症状加分
    PAST_TB_BONUS = _PAST_TB_BONUS  # 有结核病史加分
    MARK_HIGH_RISK_THRESHOLD = _MARK_HIGH_RISK_THRESHOLD  # 标记高风险确诊的阈值(%)

    # 未知输入时的回退因子（与 ScoringEngine 单一真值源保持一致）
    DEFAULT_DISTANCE_FACTOR = ScoringEngine.DEFAULT_DISTANCE_FACTOR
    DEFAULT_VENTILATION_FACTOR = ScoringEngine.DEFAULT_VENTILATION_FACTOR
    DEFAULT_SETTING_FACTOR = ScoringEngine.DEFAULT_SETTING_FACTOR

    # CSV列顺序常量（导出和导入统一使用，避免字段错位）
    FAMILY_CSV_COLUMNS = ['member_name', 'member_age', 'relationship', 'single_duration', 'freq_density',
                          'time_span', 'has_tb', 'has_symptoms', 'bcg_vaccine', 'past_illness',
                          'past_illness_type', 'contact_distance', 'ventilation', 'exposure_setting',
                          'ethnicity', 'origin_altitude', 'workplace_type', 'idu_status', 'district',
                          'months_since_migration']
    SOCIAL_CSV_COLUMNS = ['contact_name', 'contact_age', 'single_duration', 'freq_density', 'time_span',
                          'is_high_risk', 'has_symptoms', 'bcg_vaccine', 'has_tb', 'ventilation',
                          'contact_distance', 'exposure_setting', 'past_illness', 'past_illness_type',
                          'ethnicity', 'origin_altitude', 'workplace_type', 'idu_status', 'district',
                          'months_since_migration']
    
    # 智能列名映射字典（支持模糊匹配）
    COLUMN_MAPPING = {
        # 患者基本信息字段
        'scenario_type': ['scenario_type', '场景类型', '场景', 'scene', '情境', 'Scenario', 'SCENARIO'],
        'age': ['age', '年龄', '患者年龄', 'patient_age', '岁数', '年纪', 'Age', 'AGE', '病人年龄'],
        'sputum_smear': ['sputum_smear', '痰涂片', '痰涂片结果', 'sputum', '痰检', '涂片', 'Sputum', 'SPUTUM'],
        'has_cavity': ['has_cavity', '空洞', '是否有空洞', 'cavity', '肺部空洞', 'Cavity', 'CAVITY'],
        'active_tb': ['active_tb', '活动性结核', '是否活动性结核', 'active', '活动结核', 'Active', 'ACTIVE'],
        'treatment': ['treatment', '治疗', '是否接受治疗', '是否治疗', '治疗中', 'Treatment', 'TREATMENT'],
        'treatment_duration': ['treatment_duration', '治疗时长', '治疗时间', 'treat_dur', '治疗月数', '疗程', 'TreatmentDuration'],
        'cough_freq': ['cough_freq', '咳嗽频率', '每小时咳嗽次数', 'cough', '咳嗽', '咳嗽频次', 'Cough', 'COUGH'],
        'symptoms': ['symptoms', '症状', '症状严重程度', 'symptom', '症状等级', '症状表现', 'Symptoms', 'SYMPTOMS'],
        'delay_days': ['delay_days', '延迟就诊', '延迟天数', 'delay', '就诊延迟', 'Delay', 'DELAY'],
        'family_living_conditions': ['family_living_conditions', '家庭居住条件', '居住条件', 'living', '居住环境', 'LivingConditions'],
        'flp_percentage': ['flp_percentage', '家庭潜伏感染比例', 'flp', '家庭潜伏', '家庭潜伏率', 'FLP'],
        'hrsp_percentage': ['hrsp_percentage', '社会高风险人群比例', 'hrsp', '高风险比例', '社会高危率', 'HRSP'],
        'ventilation': ['ventilation', '通风', '通风条件', 'vent', '通风情况', 'Ventilation', 'VENTILATION'],
        # 家庭成员字段
        'member_name': ['member_name', '姓名', '名字', '名称', 'name', '成员姓名', '家庭成员', 'Name', 'NAME'],
        'member_age': ['member_age', '年龄', '成员年龄', 'age', '成员岁数', '成员年纪', 'MemberAge'],
        'relationship': ['relationship', '关系', '与患者关系', '与患者的关系', '亲属关系', 'Relationship'],
        'single_duration': ['single_duration', '单次接触时长', '单次接触', 'single', '单次时长', '接触时长', 'SingleDuration'],
        'freq_density': ['freq_density', '每周接触频次', '频次', '每周次数', 'frequency', 'freq', '接触频率', 'FreqDensity'],
        'time_span': ['time_span', '接触持续周期', '持续周期', '时间跨度', 'span', '接触时长', 'TimeSpan'],
        'has_tb': ['has_tb', '既往结核病史', '结核病史', 'tb_history', 'has_tuberculosis', '得过结核', '结核史', 'HasTB'],
        'has_symptoms': ['has_symptoms', '结核相关症状', '有症状', '症状', 'symptoms', '咳嗽', '低热', '盗汗', '乏力', 'HasSymptoms'],
        'bcg_vaccine': ['bcg_vaccine', '卡介苗接种史', '卡介苗', 'bcg', '疫苗', '接种史', 'BCG'],
        'past_illness': ['past_illness', '其他慢性疾病史', '其他疾病', '慢性病', 'other_illness', '病史', 'PastIllness'],
        'past_illness_type': ['past_illness_type', '疾病类型', '疾病种类', 'illness_type', '病名', 'PastIllnessType'],
        'contact_distance': ['contact_distance', '接触距离', '距离', 'distance', 'Distance'],
        'exposure_setting': ['exposure_setting', '暴露场景', '场景', 'setting', '环境', 'ExposureSetting'],
        'diagnosed': ['diagnosed', '是否确诊', '确诊', '诊断', 'tb diagnosed', '诊断结果', '确诊状态', 'Diagnosed'],
        'ethnicity': ['ethnicity', '民族', '种族', 'Ethnicity'],
        'origin_altitude': ['origin_altitude', '原籍海拔', '海拔', 'altitude', 'Altitude'],
        'workplace_type': ['workplace_type', '工作场景', '工作场所', 'workplace', 'Workplace', '油田工作'],
        'idu_status': ['idu_status', '吸毒史', '注射吸毒', 'IDU', 'idu', 'IduStatus'],
        'district': ['district', '所在城区', '城区', '行政区', 'District'],
        'months_since_migration': ['months_since_migration', '迁入月份', '迁入月数', 'migration', 'Migration'],
        # 社会接触者字段
        'contact_name': ['contact_name', '姓名', '名字', '名称', 'name', '接触者姓名', '接触人', 'ContactName'],
        'contact_age': ['contact_age', '年龄', '接触者年龄', 'age', '接触人岁数', 'ContactAge'],
        'is_high_risk': ['is_high_risk', '是否高危人群', '高危人群', 'high_risk', 'is_highrisk', '高危', 'IsHighRisk'],
        # 日期相关字段（新增）
        'start_date': ['start_date', '起始日期', '开始日期', '接触起始日期', 'StartDate'],
        'end_date': ['end_date', '结束日期', '接触结束日期', 'EndDate'],
        'contact_date': ['contact_date', '接触日期', 'ContactDate']
    }
    
    # 默认值字典（从 data_io.conf 导入，保留类属性别名以维持向后兼容）
    DEFAULT_VALUES = _DV

    # 时间转换常数
    DAYS_PER_MONTH_AVG = 30.44  # 平均每月天数（用于月→天转换）
    # 距离/场景映射（从 data_io.conf 导入，保留类属性别名）
    DISTANCE_MAPPING = DISTANCE_TEXT_MAP
    SETTING_MAPPING = SETTING_TEXT_MAP
    
    ILLNESS_TYPE_MAPPING = {
        "hiv": "hiv",
        "HIV": "hiv",
        "糖尿病": "diabetes",
        "免疫抑制剂": "immunosuppressants",
        "免疫抑制": "immunosuppressants",
        "其他": "other"
    }
    
    DISTANCE_MAPPING_REVERSE = {
        "very_close": "极近",
        "close": "近",
        "medium": "中等",
        "far": "远",
        "distant": "极远"
    }
    
    SETTING_MAPPING_REVERSE = {
        "crowded": "拥挤",
        "closed": "密闭",
        "general": "一般",
        "outdoor": "户外",
        "oilfield_camp": "油田营地"
    }
    
    ILLNESS_TYPE_MAPPING_REVERSE = {
        'hiv': 'HIV感染',
        'diabetes': '糖尿病',
        'immunosuppressants': '免疫抑制',
        'other': '其他'
    }
    
    def _create_treeview(self, parent, columns, col_widths, col_headers, height=8):
        """创建带滚动条的Treeview控件
        
        Args:
            parent: 父容器
            columns: 列名元组
            col_widths: 列宽字典 {列名: 宽度}
            col_headers: 列标题字典 {列名: 标题}
            height: Treeview高度
            
        Returns:
            tree: 创建好的Treeview对象
        """
        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True, padx=10, pady=5)
        
        tree = ttk.Treeview(container, columns=columns, show='headings', height=height)
        
        for col in columns:
            tree.heading(col, text=col_headers.get(col, col))
            tree.column(col, width=col_widths.get(col, 60), minwidth=40)
        
        v_scroll = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        h_scroll = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        return tree
    
    def _add_tooltip(self, widget, text):
        """为控件添加 tooltip"""
        ToolTip(widget, text)

    def _validate_numeric_field(self, widget, min_val, max_val, var, error_label=None):
        """实时校验数值字段，异常时红色边框和错误提示"""
        try:
            val = var.get()
            if isinstance(val, str):
                val = int(val)
            val = float(val)
            if val < min_val or val > max_val:
                raise ValueError(f"值应在{min_val}-{max_val}之间")
            widget.configure(foreground=self.COLORS['text'])
            if error_label:
                error_label.configure(text='')
            return True
        except (ValueError, TypeError, tk.TclError):
            widget.configure(foreground=self.COLORS['danger'])
            if error_label:
                error_label.configure(text=f"请输入{min_val}-{max_val}之间的数值")
            return False

    def _validate_required_field(self, widget, var, error_label=None):
        """校验必填字段"""
        try:
            val = var.get()
            if val is None or (isinstance(val, str) and not val.strip()):
                raise ValueError("必填字段为空")
            widget.configure(foreground=self.COLORS['text'])
            if error_label:
                error_label.configure(text='')
            return True
        except (ValueError, TypeError, tk.TclError):
            widget.configure(foreground=self.COLORS['danger'])
            if error_label:
                error_label.configure(text="此字段为必填项")
            return False

    def __init__(self, random_state=42):
        """初始化评估类，设置参数权重和风险阈值（基于最新医学文献）
        
        参数：
            random_state (int): 随机种子，确保可复现性
        """
        # 设置全局随机种子，确保可复现性
        self.random_state = random_state
        if NUMPY_AVAILABLE:
            np.random.seed(random_state)
        random.seed(random_state)
        try:
            os.environ['PYTHONHASHSEED'] = str(random_state)
        except (KeyError, ValueError, OSError):
            pass

        self.COLORS = {
            'primary': '#2c3e50', 'accent': '#3498db', 'success': '#2ecc71',
            'warning': '#f39c12', 'danger': '#e74c3c', 'bg_light': '#f8f9fa',
            'bg_card': '#ffffff', 'text': '#2c3e50', 'text_light': '#7f8c8d',
            'border': '#dee2e6',
        }

        # 基础参数权重（从 constants.py 单一真值源引用，.copy() 允许运行时局部覆盖）
        self.base_weights = BASE_WEIGHTS.copy()
        
        # 基于患者类型的权重调整系数（从 constants.py 单一真值源引用）
        self.patient_type_adjustments = PATIENT_TYPE_ADJUSTMENTS.copy()
        
        # 治疗阶段对传染性的调整因子（从 constants.py 单一真值源引用）
        self.treatment_infectivity_factors = TREATMENT_INFECTIVITY_FACTORS.copy()
        
        # 风险阈值（从 constants.py 单一真值源引用，.copy() 允许运行时局部覆盖）
        self.risk_thresholds = _RISK_THRESHOLDS.copy()
        
        # 存储患者信息
        self.patient_info = {}
        self.results = {}
        self.family_members = []
        self.social_contacts = []
        self.contact_count = 0  # 存储用户输入的社会接触者数量
        self.family_count = 0  # 存储用户输入的家庭成员数量
        
        # 接触者标签存储（用于ML模型训练）
        self.contact_labels = {}  # key: 接触者唯一标识, value: 0/1（是否确诊）
        self._contact_id_counter = 0  # 用于生成接触者唯一ID
        
        # 界面模式标志（策略模式：CLASSIC=传统界面，WIZARD=向导界面）
        self.UI_MODE_CLASSIC = 'classic'
        self.UI_MODE_WIZARD = 'wizard'
        self.ui_mode = None  # 初始化时设置为 None，在 init_gui 或 init_gui_wizard 中设置
        # 优先级六：UIModeAdapter 适配器（在 init_gui / init_gui_wizard 中由 create_adapter 创建）
        self.adapter = None
        
        # GUI 相关变量
        self.root = None
        self.notebook = None
        self.basic_info_vars = {}
        self.family_entries = []
        self.social_entries = []
        self.result_text = None
        self.canvas = None
        
        # 线程安全标志（threading.Event 消除读-改-写竞态）
        self.assessing = threading.Event()
        self._training = threading.Event()
        self._data_lock = threading.Lock()
        self._progress_queue = queue.Queue()
        self._poll_progress_after_id = None
        self._closing = False
        self._wizard_assessment_pending = False
        self._synthetic_warning_shown = False
        self._last_used_model_type = None
        self._assessment_error = None
        self._survival_trained = False
        self._multistate_trained = False

        # 数据安全层（Section IV）：撤销/重做 + 配置持久化 + 自动保存
        self.undo_manager = UndoManager(max_size=50)
        self.config = AppConfig.load()
        self.auto_save = AutoSaveManager(self, interval=self.config.autosave_interval)

        # 输入验证层（Section V）：三层验证工具
        from .gui.validators import FieldValidator
        self.field_validator = FieldValidator()
        # 机器学习风险分层模块 — 状态字段提前声明，实例引用延后从 service 获取
        self.ml_predictor = None
        self.ml_results = {}
        self.ml_results_lock = threading.Lock()  # 保护后台线程写入与主线程读取的竞态
        self._training_threads = []  # 跟踪后台训练线程，用于优雅退出时 join

        # 潜伏感染到发病的基线风险（终生约10%）[基于Houben et al., 2022，文献11]
        # 单一真值源：constants.LATENT_BASELINE
        self.latent_to_active_baseline = _LATENT_BASELINE

        # BCG接种保护效力参数（从 constants.py 单一真值源引用）
        self.bcg_protection_params = BCG_PROTECTION_PARAMS.copy()

        # 智能场景默认值配置（从 data_io.conf 导入，保留 karamay_localizer 更新逻辑）
        self.scenario_defaults = copy.deepcopy(_SD)

        # 当前选择的场景
        self.current_scenario = None

        # CSV 导入错误记录
        self.csv_import_errors = []

        # 克拉玛依本土化适配器（可选）
        # 问题十-3：localizer 创建统一由 service 层固化（create_default_localizer）
        # GUI 不再直接 KaramayLocalizer.get_instance()，而是通过 service 取引用，
        # 与 CLI 路径行为一致
        self.karamay_localizer = None
        self.use_karamay = False

        # ===== 核心业务服务（Phase 1.4 — 委托业务逻辑） =====
        # 问题十-2：service 构造提前至 ml_predictor 初始化之前，
        # GUI 不再自建 MLRiskPredictor，而是从 service 取同一实例引用，
        # 消除 GUI/CLI 双实例与训练状态不一致的问题。
        self._assessment_service = RiskAssessmentService(
            scoring_engine=None,  # 将在 SEIR 初始化后设置
            localizer=None,
            seir_param_uncertainty=None,
            contact_labels=self.contact_labels,
            ml_random_state=self.random_state,  # 透传 GUI 随机种子
            enable_karamay=KARAMAY_LOCALIZER_AVAILABLE,  # 问题十-3：统一由 service 创建
        )
        # 设置进度回调，让 core 服务的进度更新能传递到 GUI
        self._assessment_service.set_progress_callback(self._safe_progress_update)

        # 问题十-3：从 service 获取已固化的 localizer 引用（与 CLI 同源）
        self.karamay_localizer = self._assessment_service.get_localizer()
        if self.karamay_localizer is not None:
            try:
                self.use_karamay = True
                self.scenario_defaults.update(self.karamay_localizer.get_all_scenario_defaults())
            except (ImportError, RuntimeError, ValueError, AttributeError) as e:
                LOGGER.warning('克拉玛依适配器初始化失败，本地化功能禁用: %s', e,
                               exc_info=True)
                self.karamay_localizer = None
                self.use_karamay = False

        # 问题十-2：从 service 获取共享 ml_predictor 实例（单一真值源）
        # 若 ML 模块不可用则 service.ml_predictor 抛 ImportError，捕获后置 None
        if SKLEARN_AVAILABLE and NUMPY_AVAILABLE:
            try:
                self.ml_predictor = self._assessment_service.ml_predictor
            except (ImportError, RuntimeError, ValueError) as e:
                LOGGER.warning(
                    "MLRiskPredictor 初始化失败，ML 预测功能将不可用: %s", e,
                    exc_info=True)
                self.ml_predictor = None

        # ===== 集成权重优化器（Phase 3.2 — 从 assessment.py 迁移到 core） =====
        self._weight_optimizer = None  # 延迟初始化，在 _run_ml_predictions 中按需创建

        # ===== 方向一：随机SEIR + 贝叶斯推断模块初始化 =====
        self._init_seir_models()

        # 更新服务中的 SEIR 参数不确定性引用
        self._assessment_service.set_seir_param_uncertainty(self.seir_param_uncertainty)

        # ===== 方向二-五：高级功能（GNN/干预/多模态/不确定性）初始化 =====
        self._init_advanced_features()

        # ===== AI 助手基础设施（AiAssistantMixin 依赖） =====
        self._init_ai_assistant()

    @staticmethod
    def _to_bool(value):
        """将值转换为标准的0或1，兼容多种格式
        
        Args:
            value: 输入值，可以是int(1)、str('1'/'是'/'Yes'/'yes')等
        
        Returns:
            int: 0 或 1
        """
        return 1 if _is_yes(value) else 0
    
    def _add_import_error(self, msg):
        """添加导入错误信息
        
        Args:
            msg: 错误信息
        """
        self.csv_import_errors.append(msg)
    
    def get_patient_info(self):
        """获取完整的患者信息（委托到 cli.interactive.InteractivePatientCollector）

        CLI 交互式输入逻辑已迁移到 cli/interactive.py。
        此方法保留为委托包装器，负责 FCI/SNC 计算和状态同步。
        """
        from .cli.interactive import InteractivePatientCollector

        collector = InteractivePatientCollector()
        result = collector.collect_all()

        patient_info = result['patient_info']
        family_members = result['family_members']
        social_contacts = result['social_contacts']
        basic_info = patient_info.get('basic_info', {})

        # 计算FCI综合得分（结合家庭成员详细数据）[基于Floyd et al., 2023，文献5]
        patient_info['FCI'] = self._assessment_service.calculate_fci_score(
            family_members,
            basic_info.get('family_living_conditions', 3)
        )

        # 计算SNC综合得分（结合社会接触者详细数据）[基于Mills et al., 2008，文献6]
        patient_info['SNC'] = self._assessment_service.calculate_snc_score(social_contacts)

        # 计算累积暴露时长（填充 family_members 和 social_contacts 中 None 值）
        for m in family_members:
            if m.get('cumulative_exposure') is None:
                m['cumulative_exposure'] = calculate_cumulative_exposure(
                    m.get('single_duration', 0),
                    m.get('freq_density', self.DEFAULT_FAMILY_FREQ),
                    m.get('time_span', 4),
                    '非油田')
        for c in social_contacts:
            if c.get('cumulative_exposure') is None:
                c['cumulative_exposure'] = calculate_cumulative_exposure(
                    c.get('single_duration', 0),
                    c.get('freq_density', self.DEFAULT_SOCIAL_FREQ),
                    c.get('time_span', 4))

        # 同步到实例状态
        self.family_members = family_members
        self.social_contacts = social_contacts
        self.patient_info = patient_info

        return patient_info

    def assess_risk(self):
        """评估患者在家庭社会网络中的潜在传播风险（委托到 core 业务层）"""
        self.assessing.set()
        try:
            with self._data_lock:
                if not self.patient_info:
                    LOGGER.warning("请先调用get_patient_info()方法获取患者信息")
                    return

                # 线程安全：复制共享数据，避免后续操作中数据被其他线程修改
                patient_info_copy = dict(self.patient_info)
                family_members_copy = [dict(m) for m in self.family_members]
                social_contacts_copy = [dict(c) for c in self.social_contacts]

            self._safe_progress_update(72, "提取患者数据...")

            # 委托到 core 业务层执行完整评估
            result = self._assessment_service.assess(
                patient_info=patient_info_copy,
                family_members=family_members_copy,
                social_contacts=social_contacts_copy,
            )

            # 提取结果并填充到 GUI 兼容的 self.results 格式
            with self._data_lock:
                self.results = {
                    'individual_risks': result.get('individual_risks', {}),
                    'overall_risk': result.get('overall_risk', ''),
                    'overall_suggestion': result.get('overall_suggestion', ''),
                    'total_score': result.get('total_score', 0.0),
                    'base_infection_probability': result.get('base_infection_probability', 0.0),
                    'potential_patients': result.get('potential_patients', {'family': [], 'social': []}),
                }
                # 保存原始完整 result 供 AI 问答/报告生成使用
                # （包含 patient_score/summary/ml_results/seir_results 等字段）
                self.last_assessment_result = result

        except Exception as e:
            error_msg = f"风险评估过程中发生错误: {str(e)}"
            LOGGER.error(error_msg, exc_info=True)
            with self._data_lock:
                self.results = {}
                self._assessment_error = error_msg
        finally:
            self.assessing.clear()

    def _generate_potential_patients(self, patient_info_copy, family_members_copy, social_contacts_copy):
        """生成潜在患者判断（委托到 core 业务层）"""
        patient_treatment_days = 0
        if patient_info_copy and 'basic_info' in patient_info_copy:
            treatment_duration_months = self._safe_float(
                patient_info_copy['basic_info'].get('treatment_duration', 0), 0.0)
            patient_treatment_days = treatment_duration_months * self.DAYS_PER_MONTH_AVG

        return self._assessment_service.generate_potential_patients(
            patient_info_copy, family_members_copy, social_contacts_copy,
            patient_treatment_days)

    def _run_ml_predictions(self):
        """运行机器学习风险预测，与传统评分进行对比

        GUI 特定部分（合成数据警告、权重优化器初始化、RL 桥接）保留在此，
        核心预测循环委托到 core/ml_runner.py。
        """
        if self.ml_predictor is None:
            return

        # ---- 合成数据警告（GUI 特定） ----
        current_type = 'real' if self.ml_predictor.use_real_data else 'synthetic'
        if not self.ml_predictor.use_real_data and (
            not self._synthetic_warning_shown
            or self._last_used_model_type != current_type
        ):
            warning_msg = (
                "⚠️ 重要提示：\n\n"
                "当前机器学习模型基于合成数据训练，\n"
                "预测结果仅供参考，不能替代临床诊断！\n\n"
                "如需实际临床应用，请使用本地流行病学数据\n"
                "重新训练模型（ML标签页 -> 导入真实数据）。\n\n"
                "是否继续使用合成数据模型进行预测？"
            )
            if not messagebox.askyesno("合成数据模型警告", warning_msg):
                return
            self._synthetic_warning_shown = True
            self._last_used_model_type = current_type

        # ---- 训练模型 ----
        if not self.ml_predictor.is_trained:
            try:
                data_source = getattr(self, '_validation_data_path', None)
                self.ml_predictor.train_models(n_samples=2000, data_source=data_source)
            except (ValueError, RuntimeError, TypeError) as e:
                LOGGER.error("ML模型训练失败: %s", e, exc_info=True)
                return

        # ---- 设置患者上下文 ----
        if self.patient_info:
            basic_info = self.patient_info.get('basic_info', {})
            self.ml_predictor.set_patient_context(
                patient_ftd=basic_info.get('delay_days', 0),
                patient_cough_freq=basic_info.get('cough_freq', 0),
                contact_count=len(self.family_members) + len(self.social_contacts),
            )
        else:
            self.ml_predictor.set_patient_context(0, 0, 0)

        # ---- 三方向集成器：从 service 获取只读引用（GUI/CLI 共享同一实例）----
        # 统一 ML 集成行为：service 持有 integrator，GUI 取引用，
        # 权重优化器调优的同一实例亦被 CLI 路径使用
        self.integrator = self._assessment_service.get_integrator()

        # ---- 自动权重优化 ----
        if (self.integrator.weights_source == 'default'
            and self.ml_predictor is not None
            and hasattr(self.ml_predictor, 'model_performance')
            and self.ml_predictor.model_performance):
            if self._weight_optimizer is None:
                from .core.weight_optimizer import EnsembleWeightOptimizer
                self._weight_optimizer = EnsembleWeightOptimizer(
                    ml_predictor=self.ml_predictor,
                    integrator=self.integrator,
                    karamay_validator_ref=self,
                    karamay_localizer=getattr(self, 'karamay_localizer', None),
                    validation_data_path=getattr(self, '_validation_data_path', None),
                )
            self._weight_optimizer.ensure_validation()
            self._weight_optimizer.optimize_weights()

        # ---- 桥接 RL 引擎 ----
        if self.integrator.rl_ready and self.integrator.rl_engine is not None:
            self.ml_predictor.rl_engine = self.integrator.rl_engine

        # ---- 生存分析状态初始化 ----
        self._survival_trained = False
        self._multistate_trained = False

        # ---- 核心预测循环（委托到 core/ml_runner） ----
        from .core.ml_runner import run_ml_prediction_loop
        ml_results_local = run_ml_prediction_loop(
            assessment=self,
            ml_predictor=self.ml_predictor,
            integrator=self.integrator,
            results=self.results,
            data_lock=self._data_lock,
        )

        # ---- 原子性更新 ml_results ----
        if ml_results_local is not None:
            with self.ml_results_lock:
                self.ml_results = ml_results_local

    def _safe_int_convert(self, var, field_name, min_val=None, max_val=None, default=0):
        """安全地将 tkinter 变量或字符串转换为整数（委托到 utils.safe_int_convert）"""
        from .utils import safe_int_convert
        return safe_int_convert(var, field_name, min_val=min_val, max_val=max_val, default=default)
    
    def _update_progress(self, value, message):
        """更新进度条弹窗（仅限主线程调用），反映真实计算进度
        
        Args:
            value: 进度值（0-100）
            message: 进度描述消息
        """
        if self.root is None:
            return
        if hasattr(self, '_progress_popup_update') and self._progress_popup_update is not None:
            try:
                self._progress_popup_update(value, message)
            except (tk.TclError, RuntimeError, AttributeError):
                self._close_progress_popup()

    def _safe_progress_update(self, value, message):
        """线程安全的进度更新（可在任意线程中调用），通过队列传输到主线程"""
        try:
            self._progress_queue.put((value, message), timeout=0.5)
        except queue.Full:
            pass

    def _poll_progress_queue(self):
        """在主线程中轮询进度队列（通过root.after周期性调用）"""
        if self._closing or self.root is None:
            return
        try:
            while True:
                value, message = self._progress_queue.get_nowait()
                self._update_progress(value, message)
        except queue.Empty:
            pass
        if not self._closing and self.root is not None:
            self._poll_progress_after_id = self.root.after(100, self._poll_progress_queue)

    def _start_progress_polling(self):
        """启动进度队列轮询（必须在主线程创建root后调用）"""
        if self.root is None:
            return
        if self._poll_progress_after_id is not None:
            self.root.after_cancel(self._poll_progress_after_id)
        self._poll_progress_after_id = self.root.after(100, self._poll_progress_queue)

    def _stop_progress_polling(self):
        """停止进度队列轮询"""
        if self._poll_progress_after_id is not None and self.root is not None:
            self.root.after_cancel(self._poll_progress_after_id)
            self._poll_progress_after_id = None

    def _open_progress_popup(self, title="评估进度"):
        """打开弹窗式进度条对话框

        优先级一：同时保存 cancel_event 引用，供 assessment_task 在关键循环点检查。
        """
        if self.root is None:
            return
        self._close_progress_popup()
        popup, var, update_fn, close_fn = self._show_progress_dialog(title)
        self._progress_popup = popup
        self._progress_popup_var = var
        self._progress_popup_update = update_fn
        self._progress_popup_close = close_fn
        # 优先级一：保存 cancel_event 以便后台任务检查取消信号
        self._progress_cancel_event = getattr(popup, 'cancel_event', None)

    def _close_progress_popup(self):
        """关闭弹窗式进度条对话框"""
        try:
            if hasattr(self, '_progress_popup_close') and self._progress_popup_close is not None:
                self._progress_popup_close()
        except (tk.TclError, RuntimeError, AttributeError) as e:
            LOGGER.warning("关闭进度弹窗失败: %s", e, exc_info=True)
        self._progress_popup = None
        self._progress_popup_var = None
        self._progress_popup_update = None
        self._progress_popup_close = None
        # 优先级一：同时清空 cancel_event 引用
        self._progress_cancel_event = None

    def _reset_progress(self):
        """重置评估进度状态：清除assessing标志、关闭进度弹窗、停止轮询

        供评估流程中各种错误/取消/完成路径统一调用，防止进度弹窗残留。
        注意：延迟调度场景请使用 _schedule_delayed_progress_reset，
        以避免在延迟期间新评估启动时误关闭新弹窗。
        """
        try:
            # 清除评估中标志
            if hasattr(self, 'assessing'):
                try:
                    self.assessing.clear()
                except Exception:
                    pass
            # 关闭进度弹窗
            self._close_progress_popup()
            # 停止进度队列轮询
            self._stop_progress_polling()
            # 清空进度队列中残留消息
            if hasattr(self, '_progress_queue'):
                try:
                    while True:
                        self._progress_queue.get_nowait()
                except Exception:
                    pass
        except Exception as e:
            LOGGER.debug("_reset_progress 失败: %s", e)

    def _schedule_delayed_progress_reset(self, delay_ms=2000):
        """安全调度延迟的进度重置（防止新评估启动后旧回调误关新弹窗）

        通过捕获当前弹窗的close函数引用，在延迟到期时校验是否仍然是当前弹窗，
        避免旧回调关闭新评估的进度弹窗。
        """
        if self.root is None or self._closing:
            return
        captured_close = getattr(self, '_progress_popup_close', None)
        def _delayed_reset():
            try:
                # 若当前close函数仍是我们捕获的那个（无新弹窗），才执行重置
                if captured_close is not None and getattr(self, '_progress_popup_close', None) is captured_close:
                    self._reset_progress()
            except Exception:
                pass
        try:
            self.root.after(delay_ms, _delayed_reset)
        except (tk.TclError, RuntimeError):
            pass

    def _is_cancelled(self) -> bool:
        """优先级一：检查评估任务是否被用户取消

        供 assessment_task 在关键循环点调用，避免取消按钮无效。
        """
        cancel_event = getattr(self, '_progress_cancel_event', None)
        return cancel_event is not None and cancel_event.is_set()

    def _add_recent_file(self, file_path: str):
        """优先级一：将文件路径加入最近文件列表

        去重并限制为 10 条，保存到 config.recent_files。
        同时刷新最近文件菜单（若已创建）。
        """
        try:
            if not file_path or not hasattr(self, 'config'):
                return
            recent = getattr(self.config, 'recent_files', []) or []
            # 去重：先移除已存在的同名路径
            recent = [p for p in recent if p != file_path]
            recent.insert(0, file_path)
            # 限制为 10 条
            recent = recent[:10]
            self.config.recent_files = recent
            # 持久化保存
            try:
                self.config.save()
            except Exception:
                pass
            # 刷新最近文件菜单
            if hasattr(self, '_update_recent_files_menu'):
                self._update_recent_files_menu()
        except Exception:
            LOGGER.debug("更新最近文件列表失败", exc_info=True)

    def _check_autosave_recovery(self):
        """优先级一：启动时检查是否存在崩溃恢复数据

        在 init_gui 完成后、mainloop 之前调用。若存在 autosave 文件，
        弹出恢复确认对话框，用户确认后恢复所有 tkinter 变量状态和接触者列表。
        恢复操作包装为 ImportDataCommand 压入撤销栈，允许用户撤销恢复。
        """
        try:
            if not hasattr(self, 'auto_save'):
                return
            if not AutoSaveManager.has_autosave():
                return
            data = AutoSaveManager.load_autosave()
            if not data:
                return
            # 提取时间戳用于提示
            import os
            try:
                mtime = os.path.getmtime(
                    os.path.join(os.path.expanduser('~'), '.tb_risk', 'autosave.json'))
                from datetime import datetime
                timestamp_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                timestamp_str = '未知时间'
            family_count = len(data.get('family_entries', []))
            social_count = len(data.get('social_entries', []))
            msg = (
                f"检测到上次未正常关闭的会话数据（保存时间：{timestamp_str}）：\n"
                f"- 家庭成员：{family_count}人\n"
                f"- 社会接触者：{social_count}人\n\n"
                f"是否恢复该会话数据？"
            )
            if not messagebox.askyesno("恢复会话", msg, default='yes'):
                # 用户拒绝恢复，删除 autosave 文件
                AutoSaveManager.clear_autosave()
                return
            # 恢复接触者列表
            family_entries = data.get('family_entries', [])
            social_entries = data.get('social_entries', [])
            # 通过 ImportDataCommand 压入撤销栈（允许撤销恢复）
            if hasattr(self, 'undo_manager'):
                from .gui.undo import ImportDataCommand
                self.undo_manager.execute(
                    ImportDataCommand(self, family_entries, social_entries))
            else:
                self.family_entries = list(family_entries)
                self.social_entries = list(social_entries)
            # 恢复 contact_labels
            contact_labels = data.get('contact_labels', {})
            if contact_labels:
                self.contact_labels.update(contact_labels)
            # 恢复 basic_info_vars
            basic_vars = data.get('basic_info_vars', {})
            for key, value in basic_vars.items():
                if key in self.basic_info_vars and value is not None:
                    try:
                        var = self.basic_info_vars[key]
                        if isinstance(var, tk.IntVar):
                            var.set(int(value))
                        elif isinstance(var, tk.StringVar):
                            var.set(str(value))
                    except (ValueError, TypeError, tk.TclError):
                        pass
            # 同步 family_members / social_contacts
            self.family_members = list(self.family_entries)
            self.social_contacts = list(self.social_entries)
            # 刷新 Treeview
            if hasattr(self, '_update_family_treeview'):
                self._update_family_treeview()
            if hasattr(self, '_update_social_treeview'):
                self._update_social_treeview()
            if hasattr(self, '_update_fill_progress'):
                self._update_fill_progress()
            # 恢复后清除 autosave 文件（避免下次启动重复提示）
            AutoSaveManager.clear_autosave()
            LOGGER.info("已从 autosave 恢复会话（家庭 %d 人，社会 %d 人）",
                        family_count, social_count)
        except Exception:
            LOGGER.warning("会话恢复失败", exc_info=True)

    def _apply_karamay_enhancements(self, infection_prob, disease_prob, contact_data):
        """应用克拉玛依本土化增强（委托到 core）"""
        return self._assessment_service.apply_karamay_enhancements(
            infection_prob, disease_prob, contact_data)

    def _convert_chinese_to_value(self, chinese_val, mapping):
        """将中文界面值转换为内部英文值（委托到 core.data_conversion）"""
        return convert_chinese_to_value(chinese_val, mapping)

    def _build_environment_config(self):
        """构建环境配置（气候参数等），优先从KaramayLocalizer读取"""
        cfg = {'pm10': 100, 'humidity': 45, 'month': 4, 'is_indoor': True}
        if self.use_karamay and self.karamay_localizer:
            cfg.update(self.karamay_localizer.get_climate_defaults())
        return cfg

    @staticmethod
    def _safe_float(value, default=None):
        """安全转换为float，失败返回default"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_ventilation_int(ventilation, default=3):
        """安全地将通风值转换为整数，用于统一各评分路径的类型转换"""
        try:
            return int(ventilation)
        except (ValueError, TypeError):
            return default

    def _get_exposure_risk_score(self, cumulative_exposure):
        """根据累积暴露时长获取风险评分（委托到 ScoringEngine）"""
        try:
            cumulative_exposure = float(cumulative_exposure) if cumulative_exposure is not None else 0.0
        except (ValueError, TypeError):
            cumulative_exposure = 0.0
        return ScoringEngine._exposure_to_risk(cumulative_exposure)

    def _init_seir_models(self):
        """初始化随机SEIR + 贝叶斯推断模块（委托到 core/seir_initializer）

        解耦：core 返回 SEIRInitResult 数据对象，GUI 接收后赋值到 self，
        core 不再 mutate assessment 实例、不依赖其属性契约。
        """
        from .core.seir_initializer import init_seir_models

        # 从 karamay_localizer 解析 SEIR 种群规模（原 core 内部逻辑上移至调用方）
        try:
            if self.karamay_localizer is not None:
                seir_population = self.karamay_localizer._config.get(
                    'epidemiology.seir_population_size', 10000)
            else:
                seir_population = 10000
        except (AttributeError, ValueError, TypeError) as e:
            LOGGER.warning(
                "读取 SEIR 种群规模配置失败，使用默认值 10000: %s", e, exc_info=True)
            seir_population = 10000

        result = init_seir_models(
            seir_population=seir_population, random_state=self.random_state)
        self.stochastic_seir = result.stochastic_seir
        self.seir_inference = result.seir_inference
        self.posterior_infectivity = result.posterior_infectivity
        self.seir_param_uncertainty = result.seir_param_uncertainty

    def _init_advanced_features(self):
        """初始化高级功能模块（委托到 core/features_initializer）

        解耦：core 返回 AdvancedFeaturesInitResult 数据对象，GUI 接收后
        批量赋值到 self，core 不再 mutate assessment 实例。
        """
        from .core.features_initializer import init_advanced_features
        result = init_advanced_features()
        # 字段名与 assessment 属性一一对应，批量赋值
        self.__dict__.update(vars(result))

    def _init_ai_assistant(self):
        """初始化 AI 助手基础设施（AiAssistantMixin 依赖）

        - app_config: AppConfig 别名（ai_panel.py 通过 app_config 访问）
        - last_assessment_result: 最近一次评估结果（供 AI 问答/报告生成使用）
        - _ai_queue / _ai_active_jobs / _callbacks / _poll_timer_id:
          AiAssistantMixin 维护的运行时状态
        - 启动 queue 轮询（root=None 时安全返回，init_gui 后再次启动）

        该方法在 __init__ 末尾调用一次；不可重入。
        """
        # AppConfig 别名（self.config 是主入口，self.app_config 供 ai_panel 访问）
        self.app_config = self.config
        # 最近一次评估结果（初始为空 dict，assess_risk 完成后被覆盖）
        self.last_assessment_result = {}
        # AI queue 与回调
        self._ai_queue = queue.Queue()
        self._ai_active_jobs = set()
        # 默认回调：仅记录日志（GUI 主类可在 init_gui 后覆盖为 UI 更新函数）
        self._callbacks = {
            'answer': lambda msg: LOGGER.info("AI 回答就绪 (len=%d)",
                                              len(str(msg.get('content', '')))),
            'report': lambda msg: LOGGER.info("AI 报告就绪 (len=%d)",
                                              len(str(msg.get('content', '')))),
            'error': lambda msg: LOGGER.warning("AI 调用错误: %s",
                                                msg.get('content', '')),
        }
        self._poll_timer_id = None
        # 启动 queue 轮询（root=None 时 _schedule_next_poll 安全返回）
        try:
            self._start_ai_polling()
        except Exception as e:
            LOGGER.debug("AI queue 轮询启动失败（将在 init_gui 后重试）: %s", e)