#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 统一常量定义

所有评分/距离/通风/年龄/免疫/地理常量均在此单一真值源中定义，
其他模块通过 `from tb_risk.constants import ...` 引用，避免重复定义。

文献来源：
- WHO TB Technical Guidelines (2020): 症状权重、BCG 保护效力
- China CDC TB Annual Report (2023): 既往结核史风险倍数
- Andrews et al., JID 2012: 潜伏性结核进展率基线 10%/lifetime
- Escombe et al., PLoS Med 2009: 自然通风可降低TB传播风险>70%
- Marais et al., IJTLD 2011: <5岁进展风险为成人4倍
- Davies et al., JRSM 2006: >65岁进展风险翻倍
- Lawn et al., AIDS 2009: HIV感染者进展风险8倍
- Lange et al., Arthritis Rheum 2012: TNF-α抑制剂风险4倍
- Dooley & Chaisson, Lancet ID 2009: 糖尿病进展风险2.5倍
- Houben et al., PLoS Med 2016: 进展率上限
"""

# ==============================================================================
# 评分常数
# ==============================================================================
SYMPTOMS_BONUS = 4.0       # 有症状者风险加成（咳嗽≥2周为WHO筛查标准）
SYMPTOMS_PRESENT_BONUS = 4.0  # 接触者有症状加分（与SYMPTOMS_BONUS一致）
PAST_TB_BONUS = 3.5        # 既往结核史风险加成（复发风险，China CDC 2023）
NO_BCG_BONUS = 1.0         # 未接种BCG风险加成（WHO 2020: BCG保护效力~50%）

# ==============================================================================
# 接触距离因子
# ==============================================================================
DISTANCE_FACTORS = {
    'very_close': 1.0, 'close': 0.8, 'medium': 0.5, 'far': 0.25, 'distant': 0.1
}

# ==============================================================================
# 通风因子：1=完全密闭 ... 5=完全户外
# ==============================================================================
VENTILATION_FACTORS = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.3, 5: 0.15}

# ==============================================================================
# 暴露场景因子
# ==============================================================================
SETTING_FACTORS_BASE = {
    'crowded': 1.0, 'closed': 0.9, 'oilfield_camp': 0.85, 'general': 0.6, 'outdoor': 0.3
}

# ==============================================================================
# 年龄进展因子
# ==============================================================================
AGE_PROGRESSION = {
    'child_under_5': 4.0, 'child_5_14': 1.8, 'young': 1.2,
    'adult': 1.0, 'elderly': 2.0,
}

# ==============================================================================
# 免疫抑制进展因子
# ==============================================================================
IMMUNO_FACTORS_BASE = {
    'hiv': 8.0, 'immunosuppressants': 4.0, 'diabetes': 2.5, 'other': 1.8,
}

# ==============================================================================
# Sigmoid 感染概率参数
# P = cap / (1 + exp(-coeff * (risk - offset)))
#
# 注：coeff/offset/cap 属"风险分值→感染概率"剂量反应映射的经验系数（无单一
# 文献直接给出 0.3/10/95），其理论形式对应 Wells-Riley 方程的指数剂量反应
# 关系，拟合方法学见 Hosmer & Lemeshow 的 Logistic 回归。
# 文献来源：
#   - Wells WF. Airborne Contagion and Air Hygiene. 1955（气溶胶暴露-感染奠基）
#   - Riley RL et al. Am Rev Respir Dis 1978;117(5):893-908（Wells-Riley 方程
#     P = 1 - exp(-Iqpt/Q)，暴露剂量→感染概率的指数关系）
#   - Hosmer DW & Lemeshow S. Applied Logistic Regression（若需更严谨，可用本地
#     数据对 coeff/offset 做 Logistic 回归拟合，而非固定经验值）
# ==============================================================================
SIGMOID_COEFF = 0.3              # sigmoid 系数（risk→probability 变换斜率）
SIGMOID_OFFSET = 10.0            # sigmoid 偏移（中点风险分）
SIGMOID_CAP = 95.0               # sigmoid 上限 (%)
SIGMOID_CLIP_THRESHOLD = 700.0   # sigmoid 指数裁剪阈值（防止溢出）

# ==============================================================================
# 潜伏→活动性结核进展参数
#
# 标定说明（避免"年度 vs 终生"混用）：
# - LATENT_BASELINE 在早期版本被注释为"年度基线进展概率"，但文献中的 10%
#   实际是 LTBI 终生进展风险（Vynnycky & Fine 1997；Houben & Dodd 2016 约 5–10%），
#   且约一半病例在感染后 2 年内发病。直接按年度使用会系统性高估单年进展。
# - 自"双输出建模"（latent_infection_prob × short_term_active_risk）起，
#   LATENT_BASELINE 仅作旧的"固定基线 × 静态乘数"结构的保留参数（向后兼容），
#   新模型改用 LATENT_SHORT_TERM_BASELINE（2 年内短程基线，由终生 10% × 前 2 年
#   约一半推导，保守取 3%）配合 TIME_DECAY_FACTORS 时间衰减。
# ==============================================================================
LATENT_BASELINE = 0.10           # 旧结构年度基线进展概率（保留兼容；新模型不用）
PROGRESSION_SCALE_FACTOR = 0.15  # 旧结构进展率乘数→概率的缩放因子（保留兼容）
MAX_PROGRESSION_RATE = 0.30     # 旧结构进展率上限（保留兼容）

# --- 新：双输出进展建模（按 IGRA 判定是否感染 + 时间衰减） ---
LATENT_SHORT_TERM_BASELINE = 0.03   # 2 年内短程进展基线（终生 10% × 前 2 年约一半，保守取 3%）
TIME_DECAY_FACTORS = {              # 距感染时间(月) → 进展衰减（键为分档上界，取最接近的小档）
    6: 1.0,    # t ≤ 6 个月：近期感染/刚阳转，进展率高峰
    24: 0.5,   # 6 < t ≤ 24 个月：前 2 年，仍较高（约一半病例集中于此）
    60: 0.2,   # 24 < t ≤ 60 个月：2–5 年，显著下降
}
TIME_DECAY_LONG_TERM_FACTOR = 0.1   # t > 60 个月：远期再激活（约 200/10 万人年量级）
TIME_DECAY_DEFAULT_MONTHS = 24.0    # time_since_exposure_months 缺失时默认落入 0.5 档

# --- 新：是否已感染（LTBI 患病概率，P_infected） ---
IGRA_POSITIVE_PROB = 0.92         # 阳性 → 宣布已感染（T-SPOT/QFT 特异度约 97%，可信度高）
IGRA_NEGATIVE_REMAIN = 0.05       # 阴性 → 残留假阴性（IGRA 敏感度不足，不能完全排除）
IGRA_INDETERMINATE_PRIOR = 0.30   # 不确定 → 暴露先验缺省值
IGRA_NOT_DONE_PRIOR_CAP = 0.92    # 未检测 → 暴露先验上限（不超过阳性确认值）

# --- 新：生活方式与强化危险信号 ---
RECENT_CONVERSION_BOOST = 1.5      # 近期阳转：前置放大（新鲜感染进展风险最高，× (1 + 1.5)）
SMOKING_PROGRESSION_FACTOR = 1.2   # 吸烟：进展/复发风险轻度升高
BMI_PROGRESSION_FACTOR = {
    'low_underweight': 2.0,   # BMI < 18.5：低体重（免疫受损），风险升高
    'low_normal': 1.0,        # 18.5 ≤ BMI < 22：正常下缘，无修正
    'normal': 1.0,            # 22 ≤ BMI < 25：正常
    'overweight': 0.95,       # 25 ≤ BMI < 30：超重（轻度保护）
    'obese': 0.9,             # BMI ≥ 30：肥胖（复杂宿主免疫反应，风险略低）
}
BMI_DEFAULT = 22.0             # BMI 缺失时的默认值（落入 normal 档）
VITAMIN_D_PROGRESSION_FACTOR = {
    'deficient': 1.3,  # < 12 ng/mL：缺乏，免疫调节下降，风险升高
    'insufficient': 1.15,  # 12–20 ng/mL：不足
    'normal': 1.0,     # 20–40 ng/mL：正常
    'optimal': 0.95,   # > 40 ng/mL：充足，轻度保护
}
VITAMIN_D_DEFAULT = 30.0         # 维生素 D 缺失时默认值（落入 normal 档）

# ==============================================================================
# 累积暴露阈值
# ==============================================================================
EXPOSURE_SATURATION_HOURS = 80   # 累积暴露饱和阈值（小时）
EXPOSURE_HIGH_HOURS = 40         # 累积暴露高风险阈值（小时）
EXPOSURE_MEDIUM_HOURS = 15       # 累积暴露中风险阈值（小时）

# ==============================================================================
# 概率上限与风险阈值
# ==============================================================================
MAX_DISEASE_PROBABILITY = 50.0   # 疾病概率上限(%)
RISK_THRESHOLD_HIGH = 40.0       # 高风险标签阈值

# ==============================================================================
# 未知输入时的回退因子
# ==============================================================================
DEFAULT_DISTANCE_FACTOR = 0.5
DEFAULT_VENTILATION_FACTOR = 0.5
DEFAULT_SETTING_FACTOR = 0.6

# ==============================================================================
# Focal Loss 参数
# ==============================================================================
FOCAL_LOSS_ALPHA = 0.25          # Focal Loss alpha（正样本权重）
FOCAL_LOSS_GAMMA = 2.0           # Focal Loss gamma（难样本聚焦指数）

# ==============================================================================
# 地理与人口权重（与 config.py 保持一致）
# ==============================================================================
HIGH_ALTITUDE_THRESHOLD_M = 1500  # 高海拔阈值(m)

ETHNICITY_WEIGHTS = {
    'han': 0.75, 'uyghur': 0.15, 'kazakh': 0.05,
    'hui': 0.03, 'mongolian': 0.005, 'other': 0.015,
}

OCCUPATION_WEIGHTS = {
    'oilfield_worker': 0.30,    # 油田作业人员
    'office_worker': 0.25,      # 机关/办公室
    'service': 0.15,            # 服务业
    'student': 0.10,            # 学生
    'retired': 0.08,            # 退休
    'unemployed': 0.07,         # 无业
    'other': 0.05,              # 其他
}

DISTRICT_POPULATION_WEIGHTS = {
    '克拉玛依区': 0.45,
    '独山子区': 0.25,
    '白碱滩区': 0.20,
    '乌尔禾区': 0.10,
}

# ==============================================================================
# 月度气候基线（PM10 μg/m³、湿度 %）
# 与 config.py climate.monthly_profile 保持一致
# ==============================================================================
MONTHLY_PM10_BASELINE = {
    1: 130, 2: 120, 3: 180, 4: 160, 5: 120, 6: 70,
    7: 65, 8: 60, 9: 75, 10: 90, 11: 110, 12: 140
}

MONTHLY_HUMIDITY_BASELINE = {
    1: 62, 2: 58, 3: 35, 4: 30, 5: 28, 6: 32,
    7: 30, 8: 28, 9: 38, 10: 45, 11: 55, 12: 60
}

# ==============================================================================
# 文本映射常量（中文 ↔ 英文/整数）
# 所有模块通过 `from tb_risk.constants import ...` 引用，保证唯一真值源
# ==============================================================================

# 中文距离描述 → 英文枚举
DISTANCE_TEXT_MAP = {
    '极近': 'very_close', '极近 (<0.5 米)': 'very_close',
    '近': 'close', '近 (0.5-1 米)': 'close', '近距离': 'close',
    '中等': 'medium', '中等 (1-2 米)': 'medium', '中距离': 'medium',
    '远': 'far', '远 (2-3 米)': 'far', '远距离': 'far',
    '极远': 'distant', '极远 (>3 米)': 'distant',
}

# 中文场景描述 → 英文枚举
SETTING_TEXT_MAP = {
    '拥挤场所': 'crowded', '拥挤': 'crowded',
    '密闭场所': 'closed', '密闭': 'closed',
    '油田营地': 'oilfield_camp', '油田': 'oilfield_camp',
    '一般场所': 'general', '一般': 'general',
    '户外场所': 'outdoor', '户外': 'outdoor',
}

# 中文评级 → 整数
RATING_TEXT_MAP = {
    '极差': 1, '差': 2, '一般': 3, '好': 4, '极好': 5,
    '非常差': 1, '较差': 2, '中等': 3, '较好': 4, '非常好': 5,
}

# 中文布尔值 → 整数
BOOL_TEXT_MAP = {
    '是': 1, '否': 0,
    '有': 1, '无': 0,
    '已接种': 1, '未接种': 0,
    '曾患': 1, '未患': 0,
    '阳性': 1, '阴性': 0,
}

# 临床文本映射（痰涂片、空洞、治疗状态等）
CLINICAL_TEXT_MAP = {
    # 痰涂片
    '涂阳': 2, '涂阴': 1, '阳性': 2, '阴性': 1,
    # 空洞
    '有空洞': 2, '无空洞': 1, '空洞': 2,
    # 通用二值
    '是': 2, '否': 1,
    # 治疗状态
    '已治疗': 2, '曾治疗': 2, '正治疗': 2, '治疗': 2,
    '未治疗': 1, '未': 1,
}

# ==============================================================================
# 接触类型常量
# ==============================================================================
CONTACT_TYPE_FAMILY = 'family'
CONTACT_TYPE_SOCIAL = 'social'

# ==============================================================================
# 模型类型常量
# ==============================================================================
MODEL_TYPE_ENSEMBLE = 'ensemble'
MODEL_TYPE_LOGISTIC = 'logistic'
MODEL_TYPE_RF = 'random_forest'
MODEL_TYPE_SVM = 'svm'
MODEL_TYPE_XGB = 'xgb'
MODEL_TYPE_GNN = 'gnn'

# ==============================================================================
# 场景类型常量
# ==============================================================================
SCENARIO_TYPE_LIVING = 'living'
SCENARIO_TYPE_WORK = 'work'
SCENARIO_TYPE_TRIP = 'trip'
SCENARIO_TYPE_PARTY = 'party'
SCENARIO_TYPE_CUSTOM = 'custom'

# ==============================================================================
# 布尔值中文常量
# ==============================================================================
BOOL_YES = '是'
BOOL_NO = '否'

# ==============================================================================
# 接触距离常量（英文）
# ==============================================================================
DISTANCE_VERY_CLOSE = 'very_close'
DISTANCE_CLOSE = 'close'
DISTANCE_MEDIUM = 'medium'
DISTANCE_FAR = 'far'
DISTANCE_DISTANT = 'distant'

# ==============================================================================
# 接触距离常量（中文）
# ==============================================================================
DISTANCE_VERY_CLOSE_CN = '极近'
DISTANCE_CLOSE_CN = '近'
DISTANCE_MEDIUM_CN = '中等'
DISTANCE_FAR_CN = '远'
DISTANCE_DISTANT_CN = '极远'

# ==============================================================================
# 暴露场景常量（英文）
# ==============================================================================
SETTING_CROWDED = 'crowded'
SETTING_CLOSED = 'closed'
SETTING_GENERAL = 'general'
SETTING_OUTDOOR = 'outdoor'

# ==============================================================================
# 暴露场景常量（中文）
# ==============================================================================
SETTING_CROWDED_CN = '拥挤'
SETTING_CLOSED_CN = '密闭'
SETTING_GENERAL_CN = '一般'
SETTING_OUTDOOR_CN = '户外'

# ==============================================================================
# 疾病类型常量
# ==============================================================================
ILLNESS_TYPE_HIV = 'HIV'
ILLNESS_TYPE_DIABETES = 'diabetes'
ILLNESS_TYPE_IMMUNOSUPPRESSIVE = 'immunosuppressants'
ILLNESS_TYPE_OTHER = 'other'
ILLNESS_TYPE_NONE = 'none'

# ==============================================================================
# 基础评估权重（等权重分配，基于TRIPOD指南[文献20]推荐的可复用方法）
# 文献中未提供各因素的具体权重系数，等权重是最透明、最可复用的方法
# ==============================================================================
BASE_WEIGHTS = {
    'FCI': 0.20,  'SNC': 0.20,  'FLP': 0.20,  'HRSP': 0.20,  'FTD': 0.20,
}

# ==============================================================================
# 患者类型风险调整系数（基于Kik et al., 2021，加法模型避免风险估计膨胀）
# ==============================================================================
PATIENT_TYPE_ADJUSTMENTS = {
    'sputum_smear_positive': 1.2,
    'has_cavity': 0.8,
    'untreated': 0.6,
}

# ==============================================================================
# 治疗阶段传染性因子（基于Calderwood et al., 2021系统综述和meta分析）
# 大多数患者在治疗2周时仍保持培养阳性，挑战了"2周后即无传染性"的传统观点
# ==============================================================================
TREATMENT_INFECTIVITY_FACTORS = {
    'pre_treatment': 1.0,
    'early_treatment': 0.85,
    'mid_treatment': 0.60,
    'late_treatment': 0.30,
    'completed_treatment': 0.10,
}

# ==============================================================================
# BCG 接种保护效力参数（基于Trunz et al., 2021和Colditz et al., 1994）
# ==============================================================================
BCG_PROTECTION_PARAMS = {
    'child_severe_tb': 0.78,
    'child_pulmonary_tb': 0.55,
    'adult_pulmonary_tb': 0.15,
    'decay_10_years': 0.5,
    'decay_10_20_years': 0.3,
    'decay_over_20_years': 0.05,
}

# ==============================================================================
# 风险概率阈值
# ==============================================================================
DISEASE_PROB_VERY_HIGH = 15.0
DISEASE_PROB_HIGH = 8.0
DISEASE_PROB_MEDIUM = 3.0
INFECTION_PROB_HIGH_RISK = 60.0
INFECTION_PROB_MEDIUM_RISK = 30.0

# 概率分二分类切点（0-100 尺度，> 50 → risk_class/decision_class = 1）。
# L 级修复：原 50 分散硬编码于 scoring/architecture.py（2 处）与
# scoring/ml/evaluation.py（1 处），现为单一真值源。注意：这是展示级
# 粗二分类，与四档分级切点（8/15/3，见 DISEASE_PROB_*）语义不同。
DEFAULT_RISK_CLASS_THRESHOLD = 50.0

# 接触者年龄缺失时的特征填充默认值（岁）。
# L 级修复：原 30 分散硬编码于 scoring/ml/preprocessing.py（5 处）。
DEFAULT_CONTACT_AGE = 30.0

# ==============================================================================
# 涂阳/空洞患者感染概率调整
# ==============================================================================
SPUTUM_SMEAR_POSITIVE_INCREASE = 20.0
HAS_CAVITY_INCREASE = 15.0
UNTREATED_INCREASE = 10.0

# ==============================================================================
# 风险评分调整常量（症状/咳嗽/年龄/延迟加分）
# ==============================================================================
SYMPTOMS_SEVERE_BONUS = 1.0
COUGH_FREQ_HIGH_BONUS = 1.0
COUGH_FREQ_MEDIUM_BONUS = 0.7
COUGH_FREQ_LOW_BONUS = 0.3
AGE_YOUNG_BONUS = 0.8
AGE_ELDERLY_BONUS = 1.2
DELAY_DAYS_SEVERE_BONUS = 2.0
DELAY_DAYS_MEDIUM_BONUS = 1.0
DELAY_DAYS_MILD_BONUS = 0.5
MARK_HIGH_RISK_THRESHOLD = 40.0

# ==============================================================================
# 风险维度阈值（基于WHO 2024指南和中国WS 288-2023标准）
# ==============================================================================
RISK_THRESHOLDS = {
    'FCI_high': 5, 'FCI_medium': 3,
    'SNC_high': 30, 'SNC_medium': 15,
    'FLP_high': 15, 'FLP_medium': 5,
    'HRSP_high': 25, 'HRSP_medium': 10,
    'FTD_high': 30, 'FTD_medium': 14,
}

# ==============================================================================
# GUI / CLI 输入验证常量（Section V 单一真相源）
#
# 所有控件创建（Spinbox from_/to）、对话框验证、评估器验证、CLI 验证
# 必须从本区块导入，消除上下限不一致问题。
# 文献依据：
#   - MAX_AGE=120：中国人口寿命上限（国家统计局 2023）
#   - MAX_SINGLE_DURATION_MINUTES=480：单次接触 8 小时（睡眠+清醒时长合理上限）
#   - MAX_TIME_SPAN_WEEKS=52：一年 52 周
#   - MAX_FAMILY_MEMBERS=20：大家庭极端场景
#   - MAX_SOCIAL_CONTACTS=50：高负载接触者追踪上限（WHO 接触者调查指南）
#   - MAX_COUGH_FREQ=100：重症咳嗽上限（咳嗽监测研究）
#   - MAX_TREATMENT_DURATION_MONTHS=24：MDR-TB 长程治疗 24 个月（WHO 2020）
#   - MAX_DELAY_DAYS=365：延误就诊 1 年上限
#   - MAX_FREQ_DENSITY=30：每周 30 次（约每天 4 次）
# ==============================================================================
MIN_AGE = 0                       # 最小年龄
MAX_AGE = 120                     # 最大年龄
MIN_VENTILATION = 1               # 最小通风条件（1=完全密闭）
MAX_VENTILATION = 5               # 最大通风条件（5=完全户外）

MIN_SINGLE_DURATION_MINUTES = 0   # 单次接触时长下限（分钟）
MAX_SINGLE_DURATION_MINUTES = 480 # 单次接触时长上限（分钟）
MIN_TIME_SPAN_WEEKS = 1           # 接触持续周期下限（周）
MAX_TIME_SPAN_WEEKS = 52          # 接触持续周期上限（周）
MIN_FREQ_DENSITY = 0              # 每周接触频次下限
MAX_FREQ_DENSITY = 30             # 每周接触频次上限（次/周）

MAX_FAMILY_MEMBERS = 20           # 最大家庭成员数
MAX_SOCIAL_CONTACTS = 50          # 最大社会接触者数
MAX_COUGH_FREQ = 100              # 最大咳嗽频率（次/小时）
MIN_COUGH_FREQ = 0                # 最小咳嗽频率
MAX_TREATMENT_DURATION_MONTHS = 24  # 最大治疗时长（月）
MIN_TREATMENT_DURATION_MONTHS = 0   # 最小治疗时长（月）
MAX_DELAY_DAYS = 365              # 最大延迟就诊天数（天）
MIN_DELAY_DAYS = 0                # 最小延迟就诊天数（天）

PERCENTAGE_MIN = 0                # 百分比下限
PERCENTAGE_MAX = 100              # 百分比上限

DEFAULT_FAMILY_FREQ = 14          # 家庭成员默认每周接触频次（次/周，每天 2 次）
DEFAULT_SOCIAL_FREQ = 2           # 社会接触者默认每周接触频次（次/周）

# ==============================================================================
# 控件默认值（用于 Spinbox/Entry 初始化）
# ==============================================================================
DEFAULT_AGE = 30
DEFAULT_SINGLE_DURATION = 30
DEFAULT_TIME_SPAN = 4
DEFAULT_VENTILATION = 3
DEFAULT_TREATMENT_DURATION = 0
DEFAULT_COUGH_FREQ = 0
DEFAULT_DELAY_DAYS = 0
DEFAULT_FLP_PERCENTAGE = 0
DEFAULT_HRSP_PERCENTAGE = 0

# ==============================================================================
# SEIR 传播动力学常量
#
# 文献来源：
#   - Vynnycky & Fine, IJE 1997: TB自然史参数（潜伏期、传染期）
#   - Ragonnet et al., Nat Commun 2021: TB传播率估计
#   - Dowdy et al., BMC ID 2013: SEIR模型参数化
#   - WHO TB Guidelines 2023: 治疗阶段传染性下降时间线
# ==============================================================================

# --- 初始状态 ---
# 注：S(0)/E(0)/I(0)/R(0) 属初始条件假设（无特定文献直接给出），量级参考
# Ragonnet et al. 2021 的模型初始化设定（人群以易感为主、少量潜伏/感染）。
SEIR_INITIAL_S = 0.99             # 初始易感者比例 S(0)（≈99%易感，1%潜伏/感染）
SEIR_INITIAL_E = 0.01             # 初始潜伏者比例 E(0)（初始暴露/潜伏感染率）
SEIR_INITIAL_I = 0.0              # 初始感染者比例 I(0)
SEIR_INITIAL_R = 0.0              # 初始恢复者比例 R(0)
SEIR_INITIAL_PREVALENCE = 0.01    # 初始患病率 I/N（1%，与SEIR_INITIAL_E一致）

# --- 默认流行病学参数 ---
SEIR_DEFAULT_BETA = 0.3           # 默认传播率 β (有效接触数/天)
                                   # 文献：Ragonnet et al. 2021, TB R0≈2-4 → β≈0.2-0.5
                                   #（β 由 R0/传染期导出，0.3 为示例值）
SEIR_DEFAULT_BETA_POSTERIOR = 0.2 # 后验推断β默认均值（当MCMC未完成时的回退值）
SEIR_DEFAULT_SIGMA = 1.0 / 30.0   # 默认潜伏率 σ (/天)，平均潜伏期≈30天
                                   # 文献：Vynnycky & Fine 1997, TB潜伏期中位数~数月
SEIR_DEFAULT_GAMMA = 1.0 / 180.0  # 默认恢复率 γ (/天)，平均传染期≈180天
                                   # 文献：Vynnycky & Fine 1997（未经治疗TB自然史，
                                   # 传染期约6个月）；WHO 2023（治疗阶段划分）
SEIR_SIMULATION_DAYS = 365.0      # 默认模拟时长（天）=1年
SEIR_N_TRAJECTORIES = 10          # 默认随机模拟轨迹数
SEIR_RANDOM_SEED = 42             # SEIR随机模型默认种子（保证结果可复现）
                                  # 用户可通过配置 seir.random_seed 覆盖；
                                  # 该种子同时传播给 StochasticSEIRModel 和 MCMC 推断器

# --- MCMC收敛诊断阈值 ---
SEIR_MCMC_ESS_THRESHOLD = 400.0   # 有效样本量（ESS）阈值
                                   # 文献：Gelman et al., BDA 3rd ed. 推荐 ESS≥400

# --- 暴露→感染概率分段指数模型参数 ---
# P_infect(t) = cap·(1-exp(-rate_early·t)), t ≤ T_break
# P_infect(t) = early_risk + (1-early_risk)·(1-exp(-rate_late·(t-T_break))), t > T_break
# 其中 early_risk = cap·(1-exp(-rate_early·T_break))
EXPOSURE_EARLY_PHASE_CAP = 0.5    # 早期(≤T_break)感染概率上限（短时间接触最高50%感染风险）
EXPOSURE_EARLY_RATE = 0.05        # 早期指数速率 (/天)，时间常数=20天
EXPOSURE_LATE_RATE = 0.01         # 后期指数速率 (/天)，时间常数=100天
                                   # 解释：密切接触2周内风险快速上升，之后边际风险递减
EXPOSURE_PHASE_THRESHOLD_DAYS = 14  # 早/后期分界天数（2周），与传染性评估窗口一致

# --- 治疗阶段时间阈值（天） ---
# 与 TREATMENT_INFECTIVITY_FACTORS 中的阶段对应
TREATMENT_EARLY_THRESHOLD_DAYS = 14    # 早期治疗：0-14天
TREATMENT_MID_THRESHOLD_DAYS = 56      # 中期治疗：14-56天（≈8周）
TREATMENT_LATE_THRESHOLD_DAYS = 180    # 后期治疗：56-180天（≈6个月）
                                         # 文献：WHO 2023, 治疗2周后传染性显著下降，6个月完成治疗

# ==============================================================================
# 患者风险评分计算常量
#
# 文献来源：
#   - Kik et al., ERJ 2021: 涂阳/空洞患者传染性乘数
#   - WHO TB Guidelines 2023: 密切接触者风险分层
#   - 项目内部验证：居住条件评分与感染风险关联
# ==============================================================================

# --- 患者类型风险乘数 ---
PATIENT_RISK_MULTIPLIER_CAP = 3.0     # 患者风险乘数上限（防止风险叠加膨胀）
SMEAR_CAVITY_ADDITIVE = 0.3           # 涂阳+空洞叠加加成（涂阳且空洞时额外+0.3）
                                        # 文献：涂阳+空洞患者排菌量是单纯涂阳的2-3倍

# --- FCI（家庭接触强度）居住条件因子 ---
# living_condition_factor = max(INTERCEPT + SLOPE * rating, MIN_FACTOR)
# rating: 1(极差)~5(极好)
LIVING_CONDITION_INTERCEPT = 1.4      # 居住条件因子截距（极差居住条件→1.4倍风险）
LIVING_CONDITION_SLOPE = -0.16        # 居住条件因子斜率（每改善一级降低0.16）
LIVING_CONDITION_MIN_FACTOR = 1.1     # 居住条件因子下限（即使极好仍有基础风险）
FCI_MAX_SCORE = 10.0                  # FCI评分上限（0-10分）
FCI_DEFAULT_NO_FAMILY = 3.0           # 无家庭成员时FCI默认得分（中等偏低）

# --- SNC（社会接触网络）评分参数 ---
HIGH_RISK_CONTACT_FACTOR = 1.5        # 高风险接触者因子（高风险人群接触*1.5）
SNC_CONTACT_COUNT_DIVISOR = 5.0       # 接触数评分除数（5人→1.0, 50人→10.0）
SNC_AVG_SCORE_WEIGHT = 5.0            # SNC平均接触风险得分权重
SNC_COUNT_SCORE_WEIGHT = 1.0          # SNC接触数量得分权重
SNC_DIVISOR = 2.0                     # SNC总分合成除数（加权平均）
SNC_MAX_SCORE = 10.0                  # SNC评分上限（0-10分）

# --- 总评分合成参数 ---
TOTAL_SCORE_CAP = 30.0                # 基础分*患者乘数上限（30分）
FTD_NORMALIZATION_DAYS = 365.0        # FTD（就诊延迟）归一化分母（365天=1年）
FTD_MAX_CONTRIBUTION = 10.0           # FTD在加权评分中最大贡献（10分）
SNC_MAX_INDIVIDUAL_CONTRIBUTION = 10.0  # 单维度最大贡献（10分）

# ==============================================================================
# 实验室证据整合（三层证据：传染性 / 疾病状态 / 进展）
#
# 文献来源：
#   - 痰涂片与 GeneXpert Ct 值显著负相关：Xpert r≈-0.77、Ultra r≈-0.69，
#     全球中位 Ct≈20.1
#   - Ct 预测涂阳的截断值：Ct=25（敏感性 95%、特异性 65%）、Ct≈27.7（98%/48%）、
#     截断 28 对涂阳有较好预测价值
#   - 树芽征存在于约 72% 的活动性结核；空洞/树芽征/实变提示排菌与传染性高
#   - IGRA 在 BCG 接种人群中比 TST 更特异
#   - IGRA 敏感度（免疫健全成人）：T-SPOT 约 68%、QFT-GIT 约 52%，特异度约 97%；
#     HIV 感染者 T-SPOT 约 72%、QFT-GIT 约 61%
# ==============================================================================

# --- Ct 值 → 杆菌载量因子（bacillary_load，连续替代二值涂阳） ---
CT_BACILLARY_LOAD = {
    'high': 1.0,    # Ct < 22 → 高载量（≈涂阳 +++）
    'medium': 0.6,  # 22 ≤ Ct ≤ 28 → 中载量（≈涂阳 +/++）
    'low': 0.3,     # Ct > 28 → 低载量（≈涂阴）
}
CT_HIGH_THRESHOLD = 22.0     # Ct < 22 → 高载量
CT_MEDIUM_THRESHOLD = 28.0   # 22 ≤ Ct ≤ 28 → 中载量；>28 → 低载量

# --- 涂片分级（0–3+）→ 杆菌载量因子（与 Ct 分档量级对齐） ---
SMEAR_GRADE_LOAD = {
    0: 0.3,   # 涂阴
    1: 0.45,  # +
    2: 0.6,   # ++
    3: 1.0,   # +++
}

# --- 影像学范围因子（替代/修正杆菌载量，Ct/涂片缺失时使用） ---
IMAGING_EXTENT_FACTORS = {
    'cavity': 0.8,            # 空洞
    'tree_in_bud': 0.8,       # 树芽征（约 72% 活动性结核）
    'consolidation': 0.6,     # 大叶实变
    'miliary': 0.6,           # 粟粒
    'caseous_pneumonia': 0.6, # 干酪样肺炎
    'pleural_effusion': 0.4,  # 胸膜积液
    'nodule': 0.4,            # 仅结节（保守估计）
}

# --- IGRA 类型置信度权重（igra_type 作小的置信度权重） ---
IGRA_TYPE_WEIGHT = {
    't_spot': 1.0,        # T-SPOT.TB（敏感度最高，阴性结果更可靠）
    'quantiferon': 0.9,   # QuantiFERON（QFT-GIT）
    'tst': 0.7,           # TST（BCG 接种人群中特异度较低）
}
IGRA_TYPE_DEFAULT_WEIGHT = 0.9  # 未指定类型时的默认权重

# --- IGRA 感染状态 → 进展概率修正 ---
IGRA_NEGATIVE_SUPPRESS = 0.05        # 阴性：进展概率压到接近 0（未感染基本不进展）
IGRA_INDETERMINATE_FACTOR = 0.8      # 不确定：部分保留
# 阳性：视为已感染，启用完整 LATENT_BASELINE（×1.0，年龄/免疫因子在外层叠加）

# --- MDR（RIF 耐药）治疗传染性衰减曲线（按 MDR 拉长） ---
MDR_INFECTIVITY_LATE = 0.60         # MDR 后期治疗传染性（原 0.30，传染窗口更长）
MDR_INFECTIVITY_COMPLETED = 0.30    # MDR 完成治疗传染性（原 0.10）
MDR_TREATMENT_LATE_THRESHOLD_DAYS = 365     # MDR 后期治疗阈值（拉长至 1 年）
MDR_TREATMENT_COMPLETED_THRESHOLD_DAYS = 730  # MDR 完成治疗阈值（2 年，WHO 2020 长程 24 月）

# --- 实验室证据等级（0-3，缺失视为未检测） ---
LAB_EVIDENCE_GRADES = {
    0: '无实验室证据（基于症状/接触史推断）',
    1: '影像学证据',
    2: '涂片分级证据',
    3: 'GeneXpert（Ct+RIF）分子证据',
}

# --- 低证据不确定性放大（共形预测区间因子） ---
LOW_EVIDENCE_UNCERTAINTY_FACTORS = {
    0: 1.6,  # 无实验室证据：区间显著放大
    1: 1.3,  # 仅影像学证据
    2: 1.0,  # 涂片分级
    3: 1.0,  # GeneXpert 分子证据
}

# ==============================================================================
# 多层网络R₀分解参数（下一代矩阵法）
#
# 文献来源：
#   - Diekmann et al. (1990) 下一代矩阵法理论基础
#   - Van den Driessche & Watmough (2002) 谱半径计算
#   - Rozan et al. (2025) 多层网络SIR模型
#   - Liu & Zhu (2025) 双层网络流行病阈值公式
# ==============================================================================

HOURS_PER_YEAR = 8760.0               # 每年小时数（365*24=8760），用于暴露时长年换算
BETA_FAMILY_ANNUAL = 0.3              # 家庭层年传播系数（每年有效接触率）
                                       # 家庭密切接触传播概率高（涂阳+家庭=最高危）
BETA_SOCIAL_ANNUAL = 0.05             # 社会层年传播系数（每年有效接触率）
                                       # 社会接触传播概率低（接触时间短、通风较好）
COMMUNITY_BASELINE_R0 = 0.05          # 社区层R₀基线贡献
                                       # 社区环境随机接触的背景传播风险

# ==============================================================================
# 三方向集成器默认权重
#
# 文献：
#   - Breiman, MLJ 1996: Stacked generalization
#   - Wolpert, Neural Networks 1992: Stacked generalization
#   - Perrone & Cooper 1993: Ensemble averaging
#
# 数据依据（2026-08-22 集成层多 seed 评估，run 20260822_174335/174812，
# 机制 + 临床双标签基准、场景对齐 ML/GNN 模型）：
#   - 单纯形网格学习权重在两基准一致指向 GNN 0.7-0.85、ML 0-0.3、
#     SEIR 0-0.15（SEIR 个体排序价值有限，定位人群层预测，见
#     docs/model_boundary_clinical_validation.md §6）；
#   - 单方向：机制基准 ML 0.834±0.009 / GNN 0.775±0.014，临床基准
#     GNN 0.734±0.011 / ML 0.711±0.009（GNN 跨标签机制更稳健）；
#   - 重要警示：所有凸组合融合（固定/网格/秩变换/Platt/OOF L2 LR）
#     在两基准上均未显著超过最优单方向（DeLong p>0.05，5 seeds）。
#     权重取值在噪声区间内，本组值为跨基准折中，不是"最优解"；
#     部署时必须配合 ThreeDirectionIntegrator.set_fallback_policy()
#     护栏（验证集集成弱于最优单方向时自动降级）。
# ==============================================================================
DEFAULT_ENSEMBLE_WEIGHTS = {
    # 三方向部署默认（2026-08-23 更新，历史值 ml 0.35/seir 0.05/gnn 0.60）：
    #   - 旧值在临床基准为最差融合（0.718 vs 学权重 0.758，差 0.047）；
    #   - 新值 = 四成员冻结权重 'four' 的 ML 票位合并投影（ml+ml_clin→ml），
    #     配套 core/integrator.py 的症状富集手工门控：has_symptoms 接触者
    #     的 ML 票位由临床标签模型打分（跨域均值 0.797 vs 单模型最高 0.773，
    #     run 20260823_085303）；
    #   - SEIR 票权归零与 'four'/§6.2 结论一致（人群层定位，非个体排序）。
    'three': {'ml': 0.50, 'seir': 0.0, 'gnn': 0.50},          # 三方向：门控 ML 与 GNN 平权
    # 以下三组双方向权重为**未扫描验证的直觉默认值**（2026-09-05 地基
    # 审计标注）：'three'/'four' 有候选扫描冻结依据（见上），双方向组
    # 无对应实验——引用时不得表述为"学习所得"，部署走双方向时必须
    # 配套 set_fallback_policy() 护栏，并在获得真实数据后重扫。
    'two_ml_gnn': {'ml': 0.45, 'gnn': 0.55},                  # 双方向ML+GNN：近等权（未验证）
    'two_ml_seir': {'ml': 0.85, 'seir': 0.15},                # 双方向ML+SEIR：ML主导（未验证）
    'two_gnn_seir': {'gnn': 0.85, 'seir': 0.15},              # 双方向GNN+SEIR：GNN主导（未验证）
    # 四成员（双 ML 多样性架构，2026-08-22 冻结）：
    # 候选扫描 run 20260822_233053/234024，最终冻结评估 run
    # 20260823_001545/002506（bucket ensemble_eval_*_4m_v1）。
    # 5-seed 双基准 + 6 候选固定权重扫描（学与报分离，测试半区）：
    #   - 标签匹配的成员赢各自基准（机制 ml 0.835 / 临床 ml_clin 0.756），
    #     现行三成员权重在临床基准显著劣于最优单方向（-0.043, p=0.038）；
    #   - 本组是唯一在双基准都与最优单方向无显著差异的候选
    #     （机制 -0.013 p=0.32 / 临床 -0.015 p=0.18）——用机制基准噪声内
    #     的微小代价换跨标签族稳健性（现实标签族未知，待真实接触者
    #     数据裁决）；SEIR 票权归零（先验注入/门控交互实验均无增益，
    #     SEIR 定位人群层而非个体排序，见 §6.2）。
    'four': {'ml': 0.25, 'ml_clin': 0.25, 'gnn': 0.50, 'seir': 0.0},
}

# ==============================================================================
# 阳性率双口径（问题4：概率口径统一，单一真值源）
#
# 两套口径对应两类完全不同的筛查场景，PPV 换算、界面标注、训练报告
# 三处必须显式声明所用口径，严禁混用（见 PREVALENCE_CALIBER_MIXING_WARNING）：
#   - close_contact：密切接触人群筛查（本项目模型的目标场景）
#   - general_population：全人群普查（对照口径，仅供换算参考）
#
# 文献来源：
#   - Fox GJ et al., Lancet Infect Dis 2013;13(5):369-381: 家庭密切接触者
#     活动性结核检出率 3.1%（系统综述；项目 ml_training_anchors 锚点）
#   - 国内及亚洲密接筛查文献区间约 1-3%：四川 1.01%（家庭内 1.58%）、
#     上海静安 1.427%、印尼日惹 2.4% —— 2.85% 为项目半合成场景生成器
#     实际阳性率，落在文献区间上缘
#   - general_population 100/10万：WHO Global TB Report 2023 估算发病率
#     量级（全球 ~108/10万、中国 ~52/10万；克拉玛依本地口径为 121/10万
#     见 config.py，本地化部署以 config 为准，此处取通用量级）
# ==============================================================================
PREVALENCE_CALIBERS = {
    'close_contact': {
        'value': 0.0285,
        'display': '2.85%',
        'label': '密切接触人群（模型目标场景）',
        'source': ('Fox et al. Lancet Infect Dis 2013 家庭密接 3.1%；'
                   '国内密接筛查 1-3% 区间；项目半合成场景阳性率 2.85%'),
    },
    'general_population': {
        'value': 0.001,
        'display': '100/10万',
        'label': '全人群（对照口径，仅供换算参考）',
        'source': ('WHO Global TB Report 2023 发病率量级；'
                   '克拉玛依本地 121/10万（config.py）'),
    },
}

PREVALENCE_CALIBER_MIXING_WARNING = (
    '两套阳性率口径（密接人群 2.85% 与全人群 100/10万）不可混用：'
    'PPV 换算、界面展示与报告结论必须显式声明所用口径；'
    '在全人群口径下引用密接口径的绝对概率（或反之）属口径混用，'
    '会导致 PPV 相差数十倍量级的系统性误读。'
)

# ============================================================================
# 队列部署降级注册表（treats 降级判决缺陷5，2026-09-17；机制护栏 P1-c，2026-09-20）
# ============================================================================
# 单一真值源：哪些 v4 队列的 checkpoint 被降级为研究资产、禁止部署。
# 之前仅有文档约束（docs/model_boundary_clinical_validation.md §5），任何
# 站点仍可静默加载 treats checkpoint 做筛查决策；机制护栏后有三层执行：
#   1. 训练时（training.train_from_real_data）：队列命中 → predictor
#      .deployment_status='research_only' 落 checkpoint；
#   2. 加载时（persistence.load_model）：恢复 flag；旧档（无 flag）按
#      v4_cohort 回填补 flag——t7/t8 treats checkpoint 同样被拦；
#   3. 预测时（evaluation.predict_risk）：research_only 模型调用部署
#      预测 API 直接抛错（研究用途须显式 allow_research=True）。
RESEARCH_ONLY_COHORTS = {
    'treats': {
        'reason': ('判别 0.618 + 10% 筛查预算下操作点敏感度仅 14.5%——'
                   '每筛出 1 例真阳性需评估约 7 人且漏掉 85% 的感染者'
                   '在预算外，实用价值不足以支撑部署工具'),
        'evidence': {
            'auroc_cv': 0.6178,
            'sensitivity_at_10pct_budget': 0.145,
            'specificity': 0.941,
            'source': 't7 决策参考表（组感知 CV 口径）',
        },
        'retained_value': ('LTBI 族特征域审计基准 / 迁移学习小队列臂研究'
                           '对象 / v4 特征空间增益谱系对照（+0.079，CI '
                           '排除 0 的研究结论）'),
        'doc': 'docs/model_boundary_clinical_validation.md §5（降级判决）',
    },
}

RESEARCH_ONLY_DEPLOYMENT_ERROR = (
    '该模型为研究资产（队列 {cohort} 已降级，禁止部署）：{reason}。'
    '研究用途请显式传 allow_research=True；筛查决策不得引用该模型'
    '（判决见 docs/model_boundary_clinical_validation.md §5）。'
)