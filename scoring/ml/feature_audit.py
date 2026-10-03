#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""特征审计：相关矩阵 / VIF / 近常数 / 可推导性 + v2 精简特征集（问题3）

用户问题：22 维特征存在冗余与共线性（exposure_accumulation 与
cumulative_exposure 语义重复且 r≈0.3-0.4；single_duration/time_span/
freq_density 同属暴露时序家族），特征精简到 12-15 个，标准是
"每个特征都有独立临床含义且无法互相推导"。

本模块提供：
1. compute_feature_audit(df)：相关矩阵 + 高相关对（|r|>阈值）、
   VIF（手写 R² 回归法，无 statsmodels 依赖）、近常数列、
   可推导性检测（基展开重构：截距 + 各列 + 两两乘积（含自乘积），
   R²>0.999 即"可由其他列精确重构"）；
2. audit_training_csv(path)：对训练 CSV 完整审计 + 单变量 AUROC
   （检出无信号/常数特征）；
3. SELECTED_FEATURES_V2：22 → 15 精简特征集 + 逐项移除理由
   （FEATURE_REMOVAL_RATIONALE）。

文献：
- O'Brien RM. A Caution Regarding Rules of Thumb for Variance Inflation
  Factors. Quality & Quantity 2007;41:673-690（VIF>5 中度、>10 重度共线，
  阈值本身有争议，须结合相关结构解读）
- Riley RD et al. BMJ 2020;368:m443（TRIPOD：预测因子应有独立临床
  含义，避免同一信息重复计数导致解释失真）
- Lundberg SM et al. NeurIPS 2017（冗余特征破坏归因一致性）
"""

import logging

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

LOGGER = logging.getLogger("tb_risk.ml.feature_audit")

# 审计默认阈值
DEFAULT_CORR_THRESHOLD = 0.7     # |Pearson r| 超过即记高相关对
DEFAULT_VIF_THRESHOLD = 5.0      # VIF 超过即标记共线（O'Brien 2007）
DERIVABLE_R2_THRESHOLD = 0.999   # 基展开重构 R² 超过即"可推导"


# ============================================================================
# v2 精简特征集（22 → 15）
# ============================================================================

# 移除理由（单一真值源：审计结论 + 临床语义判断，2026-08-24 特征审计）
FEATURE_REMOVAL_RATIONALE = {
    'exposure_accumulation': (
        '可由 cumulative_exposure×time_span 精确重构（基展开 R²≈1，'
        '公式 (cum/80)×(span/10)），与 cumulative_exposure 语义重复'
        '（同为暴露剂量），用户点名的首要冗余'),
    'single_duration': (
        '暴露时序家族冗余：累积剂量 cumulative_exposure 已由'
        ' 单次时长×频次×周期 聚合（duration 是其组成部分），'
        '保留聚合剂量 + 时间窗两个临床含义最清晰的量'),
    'freq_density': (
        '暴露时序家族冗余：与 single_duration 同为累积剂量的组成部分，'
        '独立临床含义弱（"接触频繁"信息已进入累积暴露）'),
    'age_bcg_decay': (
        '可由 age×(1-bcg_vaccine) 精确重构（基展开 R²≈1），'
        '两主效应列已存在，交互不提供增量信息'),
    'highrisk_comorbid': (
        '可由 is_high_risk×past_illness 精确重构（基展开 R²≈1），'
        '两主效应列已存在，且 is_high_risk 本身就是共病聚合标志；'
        '实测场景 CSV 中 is_high_risk 全 0（dead feature），'
        '该交互列随之全 0——精简后由 is_high_risk 主效应承载该维度'),
    'immune_bcg': (
        '与 age_immuno 同属"免疫×BCG"交互家族，高度共线；'
        '保留临床证据更强的年龄×免疫（衰老免疫衰退）'),
    'cough_contact': (
        '依赖患者咳嗽频率与接触总数（源变量不在特征空间），'
        '语义混杂且与 has_symptoms 症状维度重复'),
}

# v2 精简特征集：11 基础 + 4 交互 = 15 维。
# 标准（用户约定）：每个特征都有独立临床含义且无法互相推导。
# 基础：人口学（age）/ 暴露剂量（cumulative_exposure）/ 时间窗
# （time_span）/ 症状（has_symptoms）/ BCG（bcg_vaccine）/ 既往 TB
# （has_tb）/ 距离（contact_distance_score）/ 通风（ventilation_score）/
# 高危（is_high_risk）/ 慢病史（past_illness）/ 场景
# （exposure_setting_score）；
# 交互：age_immuno（衰老×免疫）、dm_tb_synergy（糖尿病×TB 协同，
# IDCases 2025）、symptom_delay（症状×就诊延误）、age_diabetes
# （年龄×糖尿病）。
SELECTED_FEATURES_V2 = [
    # ---- 基础（11） ----
    'age',
    'cumulative_exposure',
    'time_span',
    'has_symptoms',
    'bcg_vaccine',
    'has_tb',
    'contact_distance_score',
    'ventilation_score',
    'is_high_risk',
    'past_illness',
    'exposure_setting_score',
    # ---- 交互（4） ----
    'age_immuno',
    'dm_tb_synergy',
    'symptom_delay',
    'age_diabetes',
]


# ============================================================================
# v3 特征集（15 个体 + 11 网络特征化，2026-08-25 第四轮 P3 定版）
# ============================================================================

# 网络特征化列（11 列，单一真值源 validation/combined_network.py 的
# NET_FEATURE_NAMES——此处为部署谱系副本，测试断言两侧一致防漂移）：
#   typed 5 列    分类型邻居暴露（type_exposure_*，边类型聚合）
#   window 5 列  分时间窗邻居暴露（window_exposure_*，时窗聚合）
#   pi 1 列       校准精简物理感染力（physics_lambda_calibrated，
#                 k̂·λ，λ 结构来自审计后物理——移除 3 个方向翻转乘子
#                 age_lt5/index_hiv/sleep_same_bed）
NETWORK_FEATURE_NAMES_V3 = [
    'type_exposure_household', 'type_exposure_workplace',
    'type_exposure_school', 'type_exposure_social',
    'type_exposure_casual',
    'window_exposure_0_2w', 'window_exposure_2_4w',
    'window_exposure_1_3m', 'window_exposure_3_6m',
    'window_exposure_6m_plus',
    'physics_lambda_calibrated',
]

# v3 = v2 精简个体集（15）+ 网络特征化（11）= 26 列。
#
# 组合机制消融结论（validation/combined_ablation.py，合成 20 种子）：
#   all − ind = +0.087（RF）/ +0.094（LGBM），realization 62%；
#   单机制 typed +0.061 / window +0.029 / pi +0.025；
#   协同项 −0.027（CI 不含零）→ 次可加（特征冗余主导）。
#   对外表述口径："+0.06~0.09（合成）"，不写三机制相加。
#
# 谱系边界（注册决策，第四轮 P3；推翻第三轮"NOT registered"暂缓，
# 用户最新指令优先，边界如下声明）：
#   - 合成验证层：feature_matrix(net, 'all') = 14 基础 + 11 网络列
#     （验证层个体基础列与部署 15 列是两套列名体系，共享的接口是
#     11 个网络特征列名）；
#   - 部署谱系（Kenya/HomeACF/部署 GUI）：typed/window 需图邻接
#     （边类型/时间窗分解），源字段不存在 → 补零死列，判别力
#     退化为 v2（Kenya 复验正是检验降级不劣化）；
#   - HomeACF/PACTS：physics_lambda_calibrated 可由冻结先验乘子
#     构造（精简子集），其余 10 列补零。
#   - 缺列补零 = 谱系降级而非错误（与 v2"不可识别即失败"的防错档
#     语义不同：网络列缺源字段是数据谱系的已知属性，训练内核
#     显式记录 dead 列审计，预测端同语义补零）。
SELECTED_FEATURES_V3 = list(SELECTED_FEATURES_V2) + list(NETWORK_FEATURE_NAMES_V3)


# ============================================================================
# 审计核心
# ============================================================================

def _near_constant_columns(df):
    """近常数列（std≈0 或单一取值）——训练中应剔除（dead feature）。"""
    result = []
    for col in df.columns:
        values = df[col].values
        if float(np.std(values)) < 1e-12:
            result.append(col)
            continue
        if df[col].nunique(dropna=False) <= 1:
            result.append(col)
    return result


def _vif_values(df):
    """手写 VIF（R² 回归法）：VIF_j = 1/(1-R²_j)。

    对每列用其余列（含截距）做最小二乘重构，避免 statsmodels 依赖。
    近常数列 VIF 记为 inf（完全共线）。
    """
    columns = list(df.columns)
    matrix = df.values.astype(float)
    n_rows = matrix.shape[0]
    vifs = {}
    for j, col in enumerate(columns):
        y = matrix[:, j]
        if float(np.std(y)) < 1e-12:
            vifs[col] = float('inf')
            continue
        X = np.delete(matrix, j, axis=1)
        if X.shape[1] == 0:
            vifs[col] = 1.0
            continue
        design = np.column_stack([np.ones(n_rows), X])
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ beta
        sse = float(((y - pred) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - sse / sst if sst > 0 else 1.0
        if r2 >= 1.0 - 1e-12:
            vifs[col] = float('inf')
        else:
            vifs[col] = 1.0 / (1.0 - r2)
    return vifs


def _high_correlation_pairs(df, threshold):
    """|Pearson r| > threshold 的列对（同时记录 Spearman 佐证秩相关）。"""
    corr = df.corr(method='pearson')
    try:
        spearman = df.corr(method='spearman')
    except Exception:  # noqa: BLE001 — 离散列 spearman 失败时降级
        spearman = None
    pairs = []
    columns = list(df.columns)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            a, b = columns[i], columns[j]
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) > threshold:
                entry = {
                    'a': a,
                    'b': b,
                    'pearson': round(float(r), 4),
                }
                if spearman is not None:
                    entry['spearman'] = round(
                        float(spearman.loc[a, b]), 4)
                pairs.append(entry)
    # 降序排序（最冗余的在前）
    pairs.sort(key=lambda p: abs(p['pearson']), reverse=True)
    return pairs, corr


def _derivable_features(df, r2_threshold=DERIVABLE_R2_THRESHOLD):
    """可推导性检测：基展开重构（截距 + 各列 + 两两乘积含自乘积）。

    某列能用其余列的线性-乘积基以 R²>0.999 重构 → "可由其他列推导"，
    精简特征集时应优先移除（信息重复计数）。乘积基覆盖本项目全部
    交互特征公式（如 exposure_accumulation = cum/80 × span/10、
    age_bcg_decay = age × (1-bcg) = age - age×bcg）。

    计算量：n 列 → 每列 O(n-1 + C(n-1,2)) 维基；列数大时仅当
    n_features ≤ 40 时执行（防组合爆炸）。
    """
    columns = list(df.columns)
    n_features = len(columns)
    derivable = []
    if n_features < 2 or n_features > 40:
        return derivable

    matrix = df.values.astype(float)
    n_rows = matrix.shape[0]
    for j, target in enumerate(columns):
        y = matrix[:, j]
        if float(np.std(y)) < 1e-12:
            continue
        other_idx = [k for k in range(n_features) if k != j]
        # 基：截距 + 各列 + 两两乘积（i<=j，含自乘积）
        design_cols = [np.ones(n_rows)]
        basis_names = ['截距']
        for k in other_idx:
            design_cols.append(matrix[:, k])
            basis_names.append(columns[k])
        for ii in range(len(other_idx)):
            for jj in range(ii, len(other_idx)):
                design_cols.append(
                    matrix[:, other_idx[ii]] * matrix[:, other_idx[jj]])
                basis_names.append(
                    f'{columns[other_idx[ii]]}*{columns[other_idx[jj]]}')
        design = np.column_stack(design_cols)
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ beta
        sst = float(((y - y.mean()) ** 2).sum())
        if sst <= 0:
            continue
        r2 = 1.0 - float(((y - pred) ** 2).sum()) / sst
        if r2 >= r2_threshold:
            nonzero = [basis_names[k]
                       for k in range(1, len(basis_names))
                       if abs(beta[k]) > 1e-8]
            derivable.append({
                'feature': target,
                'r2': round(float(r2), 6),
                'reconstruct_basis': nonzero,
            })
    return derivable


def compute_feature_audit(df, corr_threshold=DEFAULT_CORR_THRESHOLD,
                          vif_threshold=DEFAULT_VIF_THRESHOLD):
    """特征审计：相关矩阵 / 高相关对 / VIF / 近常数 / 可推导性。

    Args:
        df: 纯特征 DataFrame（不含标签列）
        corr_threshold: 高相关对阈值（|Pearson r|，默认 0.7）
        vif_threshold: 共线标记阈值（默认 5.0，O'Brien 2007）

    Returns:
        dict: {
            'n_samples', 'n_features',
            'correlation_matrix': {col: {col: r}}（Pearson），
            'high_corr_pairs': [{a, b, pearson, spearman?}]（降序），
            'vif': {col: vif}，'vif_flagged': [col...]（>阈值），
            'near_constant': [col...]，
            'derivable': [{feature, r2, reconstruct_basis}]，
        }

    Raises:
        ValueError: 空 DataFrame
    """
    if not PANDAS_AVAILABLE:
        raise ImportError('特征审计需要 pandas')
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError('审计输入 DataFrame 为空')

    df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')

    pairs, corr = _high_correlation_pairs(df, corr_threshold)
    vifs = _vif_values(df)
    audit = {
        'n_samples': int(df.shape[0]),
        'n_features': int(df.shape[1]),
        'correlation_matrix': corr.round(4).to_dict(),
        'high_corr_pairs': pairs,
        'vif': {k: (v if np.isfinite(v) else 1e9) for k, v in vifs.items()},
        'vif_flagged': [c for c, v in vifs.items() if v > vif_threshold],
        'near_constant': _near_constant_columns(df),
        'derivable': _derivable_features(df),
    }
    return audit


def audit_training_csv(csv_path, label_col='tb_outcome',
                       corr_threshold=DEFAULT_CORR_THRESHOLD,
                       vif_threshold=DEFAULT_VIF_THRESHOLD):
    """对训练 CSV 做完整特征审计（含单变量 AUROC）。

    单变量 AUROC 检出无信号/方向异常特征（如 Vietnam v1 的
    has_symptoms 天花板效应：univariate AUROC 0.501——项目既往教训：
    训练前必须验证特征变异与单变量信号）。

    Args:
        csv_path: 训练 CSV 路径（22 特征 + 标签列）
        label_col: 标签列名（默认 'tb_outcome'）

    Returns:
        compute_feature_audit 的完整结果 + univariate_auroc +
        positive_rate / csv_path / label_col
    """
    if not PANDAS_AVAILABLE:
        raise ImportError('特征审计需要 pandas')
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f'CSV 缺少标签列 {label_col}（现有列: '
                         f'{list(df.columns)}）')
    y = df[label_col].values
    features = df.drop(columns=[label_col])

    audit = compute_feature_audit(features, corr_threshold, vif_threshold)
    audit['label_col'] = label_col
    audit['csv_path'] = str(csv_path)
    audit['positive_rate'] = round(float(np.mean(y)), 6)

    if SKLEARN_AVAILABLE:
        univariate = {}
        for col in features.columns:
            try:
                univariate[col] = round(
                    float(roc_auc_score(y, features[col].values)), 4)
            except ValueError:
                univariate[col] = None
        audit['univariate_auroc'] = univariate
        # 无信号特征（单变量 AUROC ≈ 0.5，|auroc-0.5|<0.02）
        audit['no_signal_features'] = sorted(
            [c for c, a in univariate.items()
             if a is not None and abs(a - 0.5) < 0.02])
    return audit
