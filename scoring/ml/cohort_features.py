#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""族感知 v4 特征空间：per-cohort extras 构造的单一真值源（P1，2026-09-16）

背景（cohort_v4_features_20260915 定量确证）：22 维冻结特征空间在真实
队列近死（treats 1/22、brazil 2/22 活列），瓶颈在特征空间与终点的匹配
度而非训练方法。队列专属 extras（CRP 的 cad_score/crp_mgdl/症状组合、
Taiwan 的 QFT/CXR、TREATS 的 roomshare_n/hh_tb_contact、Peru 的先证
列、NHANES 的 bmi/smoking/income_pir、Brazil 的 ifng_genotype/ethnic）
配对 CV 增益 LR：taiwan +0.279 / crp +0.186 / brazil +0.109 / nhanes
+0.089 / treats +0.079 / peru +0.064（全部 bootstrap CI 排除 0）。

本模块是 V4_SPECS / 泄漏规则 / 哨兵清洗 / 缺失指示器的单一真值源：
  - 训练主管线（training.train_from_real_data feature_set='v4'）；
  - 评估脚本（data/run_cohort_v4_features.py）。
填补口径统一（缺陷1修复，2026-09-16）：CV 指标一律折内中位数填补
（训练折拟合填补器；主管线经 apply_v4_feature_space(return_raw=True)
+ Pipeline(SimpleImputer(median))，评估脚本经 _fold_fill），部署工件
用全数据中位数（fill_medians 随 checkpoint 落盘，部署端单样本构造
复用）——spec、列集与 CV 判决口径完全一致，两脚本数字可直接对照。

泄漏审计（按终点族结构规则，2026-09-15 定版）：
  - LTBI 族（nhanes/treats/brazil）排除 TST/IGRA 结果列（brazil
    tst_result_raw / group_raw 为终点定义性泄漏）；
  - 确诊/活动性 TB 族（kenya/crp/taiwan）排除培养/涂片结果列（taiwan
    culture_result / smear_cat）；
  - 影像分数（cad_score/cxr_*）与炎症标志（crp_mgdl）为合法协变量。

哨兵值（数据审计 2026-09-15）：crp_mgdl==999、cad_score==-1、
income_pir<1e-6（最小值 5.4e-79 为零替换产物）、brazil ethnic=='4'
语义可疑——置 NaN/指示器后再填补。

kenya v4 extras（P6，2026-09-15 原始数据再审）：回 S01 269 变量扫描，
新增 gender（Sex，单变量 0.596——原 ETL 唯一重大遗漏）/cxr 终判读
（xfindingall 三列，覆盖 99.1%，影像家族规则合法）/7 症状个别分辨率/
cough_weeks 时长梯度/treatment_sought/hiv_tested（14 条目 + 2 指示器
= 16 extras）。排除：细菌学结果族（终点定义）、痰标本级联族
（lab_sputum_requested 单变量 0.915，送检由症状+CXR 决定——巴基斯坦
教训同构）、现症治疗族（与患病终点同源）。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# ---- v4 增量列 spec：(out_col, src, mode, kwargs) ----
# mode:
#   binary_map  字符串/分类 → 是否等于 pos（缺失 → NaN）
#   unknown_ind '4'/缺失 → 1（语义可疑码与缺失合并指示器）
#   as_is       数值直传（NaN 保留待填补；可带 indicator）
#   num         数值 + 哨兵置 NaN（sentinel_eq / sentinel_lt）+
#               可选缺失指示器（indicator，缺失率 ≥1% 才生成）
#   onehot      分类 → dummies（base 为基类，不生成其列；类别全数据
#               动态探测并记入 audit，部署漂移由 feature_contract 兜底）
V4_SPECS: Dict[str, List[Tuple[str, str, str, dict]]] = {
    'nhanes': [
        ('gender_female', 'gender', 'binary_map', {'pos': 'f'}),
        ('bmi', 'bmi', 'num', {}),
        ('smoking', 'smoking', 'as_is', {}),
        ('born_us', 'born_us', 'as_is', {}),
        ('income_pir', 'income_pir', 'num',
         {'sentinel_lt': 1e-6, 'indicator': 'income_pir_missing'}),
        ('race_eth', 'race_eth', 'onehot', {'base': 1.0}),  # 5 级 → 4
    ],
    'treats': [
        ('gender_female', 'gender', 'binary_map', {'pos': 'f'}),
        ('smoking_status', 'smoking_status', 'onehot', {'base': 0}),  # 3→2
        ('roomshare_n', 'roomshare_n', 'as_is', {}),
        ('hhdens_quartile', 'hhdens_quartile', 'as_is', {}),
        ('hh_tb_contact', 'hh_tb_contact', 'num', {}),
        ('alcohol_use', 'alcohol_use', 'onehot', {'base': 0}),  # 4→3
        ('country_sa', 'country_code', 'binary_map', {'pos': 'SA'}),
    ],
    'crp': [
        ('gender_female', 'gender', 'binary_map', {'pos': 'f'}),
        ('cad_score', 'cad_score', 'num',
         {'sentinel_eq': -1.0, 'indicator': 'cad_score_missing'}),
        ('crp_mgdl', 'crp_mgdl', 'num',
         {'sentinel_eq': 999.0, 'indicator': 'crp_missing'}),
        ('symptom_cough', 'symptom_cough', 'as_is', {}),
        ('symptom_cough_weeks', 'symptom_cough_weeks', 'as_is', {}),
        ('symptom_fever', 'symptom_fever', 'as_is', {}),
        ('symptom_fever_weeks', 'symptom_fever_weeks', 'as_is', {}),
        ('symptom_chestpain', 'symptom_chestpain', 'as_is', {}),
        ('symptom_chestpain_weeks', 'symptom_chestpain_weeks', 'as_is', {}),
        ('symptom_nightsweats', 'symptom_nightsweats', 'as_is', {}),
        ('symptom_nightsweats_weeks', 'symptom_nightsweats_weeks', 'as_is', {}),
        ('symptom_weightlost', 'symptom_weightlost', 'as_is', {}),
        ('symptom_weightlost_weeks', 'symptom_weightlost_weeks', 'as_is', {}),
        ('tb_history', 'tb_history_raw', 'onehot', {'base': 'no'}),  # 3→2
        ('country_sa', 'country_code', 'binary_map', {'pos': 'SA'}),
    ],
    'taiwan': [
        ('gender_female', 'gender', 'binary_map', {'pos': 'f'}),
        ('bmi', 'bmi', 'num', {}),
        ('smoking', 'smoking', 'as_is', {}),
        ('comorbidity_any', 'comorbidity_any', 'as_is', {}),
        ('qft_tb_ag_nil', 'qft_tb_ag_nil', 'num', {}),
        ('qft_result_pos', 'qft_result', 'binary_map', {'pos': 'positive'}),
        ('cxr_score', 'cxr_score', 'num',
         {'indicator': 'cxr_missing'}),  # 17.1% 缺失，与 cxr_* 二值同源
        ('cxr_fibronodular', 'cxr_fibronodular', 'as_is',
         {'indicator': 'cxr_missing'}),
        ('cxr_cavitation', 'cxr_cavitation', 'as_is',
         {'indicator': 'cxr_missing'}),
        ('cxr_pleural_effusion', 'cxr_pleural_effusion', 'as_is',
         {'indicator': 'cxr_missing'}),
        ('ltbi_class', 'ltbi_class_raw', 'onehot', {'base': 1}),  # 3→2
    ],
    'brazil': [
        ('gender_female', 'gender', 'binary_map', {'pos': 'f'}),
        ('ifng_genotype', 'ifng_genotype', 'onehot', {'base': 'AA'}),
        ('ethnic_white', 'ethnic', 'binary_map', {'pos': 'WHITE'}),
        ('ethnic_unknown', 'ethnic', 'unknown_ind', {'unknown': ['4']}),
    ],
    'peru_mdr': [
        ('gender_female', 'gender', 'binary_map', {'pos': 'f'}),
        ('mdr_household', 'mdr_household', 'as_is', {}),
        ('index_smear_grade', 'index_smear_grade', 'as_is', {}),
        ('index_smear_missing', 'index_smear_missing', 'as_is', {}),
        ('index_cough_days', 'index_cough_days', 'num',
         {'indicator': 'index_cough_missing'}),
        ('index_hiv', 'index_hiv', 'as_is', {}),
    ],
    # kenya（P6 原始数据再审，2026-09-15）：14 条目 + 2 指示器 = 16
    # extras。cxr_* 为 ETL 派生契约列（xfindingall 终判读）；breathless
    # 问卷流条件缺失 26.6%、treatment_sought 未问缺失 62.5% → 指示器
    'kenya': [
        ('gender_female', 'gender', 'binary_map', {'pos': 'f'}),
        ('cxr_tb_suspect', 'cxr_tb_suspect', 'as_is', {}),
        ('cxr_abnormal_other', 'cxr_abnormal_other', 'as_is', {}),
        ('cxr_missing', 'cxr_missing', 'as_is', {}),
        ('symptom_bloodcough', 'symptom_bloodcough', 'as_is', {}),
        ('symptom_sputum', 'symptom_sputum', 'as_is', {}),
        ('symptom_chestpains', 'symptom_chestpains', 'as_is', {}),
        ('symptom_fever', 'symptom_fever', 'as_is', {}),
        ('symptom_fatigue', 'symptom_fatigue', 'as_is', {}),
        ('symptom_nightsweats', 'symptom_nightsweats', 'as_is', {}),
        ('symptom_breathless', 'symptom_breathless', 'as_is',
         {'indicator': 'symptom_breathless_missing'}),
        ('cough_weeks', 'cough_weeks', 'as_is', {}),
        ('treatment_sought', 'treatment_sought', 'as_is',
         {'indicator': 'treatment_sought_missing'}),
        ('hiv_tested', 'hiv_tested', 'as_is', {}),
    ],
}

# ---- 泄漏/冗余排除登记（单变量 AUROC 披露表用） ----
LEAK_EXCLUSIONS: Dict[str, Dict[str, str]] = {
    'brazil': {
        'tst_result_raw': 'leak_endpoint_defining（LTBI 由 TST 定义）',
        'group_raw': 'leak_endpoint_defining（TB/TST 分组即标签）',
        'past_illness_type': 'redundant（全为 none；22 维 past_illness 已捕捉）',
    },
    'taiwan': {
        'culture_result': 'leak_endpoint_defining（确诊终点由培养定义）',
        'smear_cat': 'leak_endpoint_defining（涂片分类为确诊依据）',
        'qft_nil': 'redundant（qft_tb_ag_nil + qft_result_pos 已代表）',
        'qft_tb_ag': 'redundant（qft_tb_ag_nil 是其标准化判定量）',
        'qft_t1': 'redundant（qft_tb_ag_nil + qft_result_pos 已代表）',
        'qft_t2': 'redundant（qft_tb_ag_nil + qft_result_pos 已代表）',
        'qft_cd8': 'redundant（qft_tb_ag_nil + qft_result_pos 已代表）',
        'past_illness_type': 'redundant（22 维 past_illness 已捕捉）',
    },
    'crp': {
        'crp_positive': 'redundant（crp_mgdl 阈值化派生）',
        'age_group': 'redundant（age 的粗化分组，22 维 age 已进）',
        'past_illness_type': 'redundant（22 维 past_illness 已捕捉）',
    },
    'nhanes': {
        'past_illness_type': 'redundant（22 维 past_illness 已捕捉）',
    },
    'treats': {
        'past_illness_type': 'redundant（全为 none；22 维 past_illness 已捕捉）',
    },
    'peru_mdr': {
        'past_illness_type': 'redundant（22 维 past_illness/is_high_risk 已捕捉）',
        'age_band': 'redundant（age 的粗化分组）',
        'follow_up_days': 'excluded_survival_only（时间轴列，属生存分析路径）',
    },
    # kenya（P6）：参照列由 ETL 发射进 CSV 供披露表计算单变量 AUROC
    'kenya': {
        'lab_smear_pos': 'leak_endpoint_defining（涂片阳性为终点组分）',
        'lab_sputum_requested': 'leak_cascade_embedded（送检由症状+CXR 决定，'
                                '单变量 AUROC 0.915——巴基斯坦同构证据）',
        'tb_current_treatment': 'leak_outcome_adjacent（现症治疗与患病终点同源）',
    },
}

# ---- 队列签名（列存在性探测；每队列取 2 个高特异列） ----
COHORT_SIGNATURES: Dict[str, List[str]] = {
    'nhanes': ['income_pir', 'race_eth'],
    'treats': ['roomshare_n', 'hhdens_quartile'],
    'crp': ['cad_score', 'crp_mgdl'],
    'taiwan': ['qft_tb_ag_nil', 'cxr_score'],
    'brazil': ['ifng_genotype', 'ethnic'],
    'peru_mdr': ['index_smear_grade', 'mdr_household'],
    # kenya（P6）：ETL 派生契约列，列名不与任何其他队列冲突
    'kenya': ['cxr_tb_suspect', 'symptom_bloodcough'],
}


def _collect_source_columns() -> frozenset:
    """全部 v4 spec 源列 + 签名列（列名契约，供编排层保护不被模糊改写）。"""
    srcs = set()
    for spec in V4_SPECS.values():
        for _out, src, _mode, _kw in spec:
            srcs.add(src)
    for sigs in COHORT_SIGNATURES.values():
        srcs.update(sigs)
    return frozenset(srcs)


# train_from_real_data 的 map_columns 保护集扩展用：这些列名是
# cohort spec 的契约（io_utils.map_columns 的模糊映射会把 ethnic→
# ethnicity、race_eth→ethnicity、index_smear_grade→sputum_smear、
# tb_history_raw→has_tb 等——轻则签名探测失败静默回退 v1，重则
# 与基础列改名碰撞产生重复列）
V4_SOURCE_COLUMNS = _collect_source_columns()

# 泄漏参照列（LEAK_EXCLUSIONS 键）同受保护（P6，2026-09-15）：kenya 的
# lab_smear_pos / lab_sputum_requested 会被模糊映射双双改写为
# sputum_smear（重复列碰撞 + 预验证噪声），tb_current_treatment →
# treatment。这些列名是披露表契约（F2 univariate_table 与
# kenya_dataset_meta.json leak_audit_reference_columns）
LEAK_REFERENCE_COLUMNS = frozenset(
    col for excl in LEAK_EXCLUSIONS.values() for col in excl)

# 二值列缺失填 0 后的指示器下限（与 num 模式一致：≥1% 才生成）
_INDICATOR_MIN_RATE = 0.01


def detect_cohort(df: pd.DataFrame) -> Optional[str]:
    """按列签名探测队列（不依赖文件名——编排层路径自由）。

    签名列全部存在 → 命中；未命中返回 None（调用方回退 v1 并披露）。
    """
    cols = set(df.columns)
    for cohort, sigs in COHORT_SIGNATURES.items():
        if all(s in cols for s in sigs):
            return cohort
    return None


def build_v4_extras(df: pd.DataFrame, cohort: str,
                    fill: str = 'median_full'
                    ) -> Tuple[List[str], np.ndarray, dict]:
    """构建 per-cohort v4 增量列（哨兵清洗 + 指示器 + 填补）。

    参数：
        df: 原始数据（22 维 + extras 源列）
        cohort: 队列名（V4_SPECS 键）
        fill: 填补口径——'median_full'（主管线：全数据中位数，缺失
              信息已由指示器承载）/ 'none'（NaN 保留，评估脚本折内
              自行填补）

    返回 (extra_cols, X_extra, audit)；audit 含哨兵/指示器/类别记录。
    """
    if cohort not in V4_SPECS:
        raise ValueError(f'队列 {cohort!r} 无 v4 spec（未识别队列）')
    spec = V4_SPECS[cohort]
    cols: List[str] = []
    arrays: List[np.ndarray] = []
    audit: dict = {'cohort': cohort, 'sentinels_applied': [],
                   'indicators_added': [], 'onehot_categories': {},
                   'fill': fill}
    for out_col, src, mode, kw in spec:
        s = df[src]
        if mode == 'binary_map':
            col = (s.astype(str).str.strip().str.lower() == kw['pos'])
            col = col.astype(float).where(s.notna(), np.nan)
        elif mode == 'unknown_ind':
            col = (s.isna() | s.astype(str).isin(kw['unknown'])).astype(float)
        elif mode == 'as_is':
            col = pd.to_numeric(s, errors='coerce')
        elif mode == 'num':
            col = pd.to_numeric(s, errors='coerce')
            if 'sentinel_eq' in kw:
                n_hit = int((col == kw['sentinel_eq']).sum())
                if n_hit:
                    audit['sentinels_applied'].append(
                        {'column': src, 'rule': f"=={kw['sentinel_eq']}",
                         'n': n_hit})
                col = col.mask(col == kw['sentinel_eq'], np.nan)
            if 'sentinel_lt' in kw:
                n_hit = int((col < kw['sentinel_lt']).sum())
                if n_hit:
                    audit['sentinels_applied'].append(
                        {'column': src, 'rule': f"<{kw['sentinel_lt']}",
                         'n': n_hit})
                col = col.mask(col < kw['sentinel_lt'], np.nan)
        elif mode == 'onehot':
            cats = sorted(pd.unique(s.dropna()), key=str)
            audit['onehot_categories'][src] = [str(c) for c in cats]
            for c in cats:
                if c == kw['base']:
                    continue
                # 下划线连接（非 [AT] 方括号）：主管线把 X 包成带名
                # DataFrame（H-ML3 特征契约需要真实列名），LightGBM
                # 拒绝特殊 JSON 字符（[]{}",:）——评估脚本
                # run_cohort_v4_features.py 喂 numpy 不触发校验故曾用
                # 方括号命名；模块为单一真值源，统一为 LGBM 安全命名
                cols.append(f'{out_col}_{c}')
                arrays.append((s == c).astype(float).to_numpy())
            continue
        else:
            raise ValueError(mode)
        # 缺失指示器（同名去重：taiwan cxr_missing 由多列共享）
        if 'indicator' in kw:
            miss = float(col.isna().mean())
            if (miss >= _INDICATOR_MIN_RATE or col.isna().any()) \
                    and kw['indicator'] not in cols:
                cols.append(kw['indicator'])
                arrays.append(col.isna().astype(float).to_numpy())
                audit['indicators_added'].append(
                    {'column': kw['indicator'], 'source': src,
                     'missing_rate': round(miss, 4)})
        cols.append(out_col)
        arrays.append(col.astype(float).to_numpy())
    X = np.column_stack(arrays) if arrays else np.zeros((len(df), 0))
    if fill == 'median_full' and X.size:
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        X = np.where(np.isnan(X), med, X)
        # 训练中位数按列名落 audit——部署端单样本构造（fill='given'）
        # 必须复用训练中位数，绝不在部署数据上重算
        audit['fill_medians'] = {c: float(m) for c, m in zip(cols, med)}
    audit['n_extra_cols'] = len(cols)
    return cols, X, audit


def apply_v4_feature_space(df: pd.DataFrame,
                           base_feature_names: List[str],
                           return_raw: bool = False
                           ) -> Tuple[Optional[np.ndarray], Optional[List[str]],
                                      dict]:
    """22 维基础 + per-cohort extras → v4 特征矩阵（训练主管线入口）。

    22 维沿用主管线口径（fillna(0)——与存量 v1 语义一致）；extras 走
    哨兵清洗 + 指示器 + 全数据中位数填补。队列未识别（合成/未知
    数据集/缺签名列）→ 返回 (None, None, {'cohort': None, ...})，
    调用方回退 v1 并披露（不冒认）。

    return_raw=True（缺陷1口径统一，2026-09-16）：额外返回未填补
    矩阵 X_raw（extras NaN 保留，base 仍 fillna(0)）——主管线 CV
    指标改折内中位填补（训练折拟合填补器，与 F2 评估脚本
    run_cohort_v4_features.py 判决口径一致），部署工件仍用全数据
    中位（fill_medians 随 checkpoint 落盘，部署端单样本构造复用）。
    返回 4 元组 (X, X_raw, names, audit)；未识别队列返回
    (None, None, None, audit)。
    """
    cohort = detect_cohort(df)
    if cohort is None or cohort not in V4_SPECS:
        audit = {'cohort': cohort,
                 'note': '未识别队列或无 extras → 回退 v1（22 维）'}
        return ((None, None, None, audit) if return_raw
                else (None, None, audit))
    # 一次构建 NaN 保留版：部署矩阵 = 全数据中位填补；raw 供 CV 折内填补
    extra_cols, X_extra_raw, audit = build_v4_extras(df, cohort, fill='none')
    base = df[base_feature_names].apply(
        pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=float)
    X_raw = np.hstack([base, X_extra_raw])
    if X_extra_raw.size:
        med = np.nanmedian(X_extra_raw, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        X_extra = np.where(np.isnan(X_extra_raw), med, X_extra_raw)
        # 训练中位数按列名落 audit——部署端单样本构造（fill='given'）
        # 必须复用训练中位数，绝不在部署数据上重算
        audit['fill_medians'] = {c: float(m) for c, m in zip(extra_cols, med)}
    else:
        X_extra = X_extra_raw
        audit['fill_medians'] = {}
    audit['fill'] = 'median_full'
    audit['n_extra_cols'] = len(extra_cols)
    X = np.hstack([base, X_extra])
    names = list(base_feature_names) + list(extra_cols)
    audit['n_base'] = len(base_feature_names)
    audit['n_total'] = len(names)
    return ((X, X_raw, names, audit) if return_raw
            else (X, names, audit))


def build_v4_extras_for_prediction(record: dict, cohort: str,
                                   expected_cols: List[str],
                                   fill_medians: Dict[str, float]
                                   ) -> Tuple[np.ndarray, dict]:
    """部署端单样本 extras 构造（列集严格对齐训练）。

    与训练口径（build_v4_extras fill='median_full'）的一致性契约：
      - 哨兵清洗 / 指示器 / onehot 走同一 spec；
      - 填补复用训练中位数（fill_medians，随 checkpoint 落盘）——
        绝不在部署单样本上重算中位数；
      - 列集按 expected_cols（训练时的 extras 列名，即
        predictor.v4_feature_names 去掉 22 维基础列）reindex：
        基类 onehot 列 / 未触发的指示器列补 0（语义 = 基类/未缺失），
        部署新出现的类别列丢弃（基类语义），spec 漂移缺失列补 0
        并记 audit（防御性，正常不应发生）。

    参数：
        record: 单样本原始字段字典（extras 源列名与训练 CSV 一致；
                缺失字段 → NaN → 训练中位数 + 指示器=1，与训练时
                "字段缺失"语义完全一致）
        cohort: 训练时探测的队列名（predictor.v4_cohort）
        expected_cols: 训练时的 extras 列名清单
        fill_medians: 训练时的逐列中位数（audit['fill_medians']）

    返回 (X_extra, audit)：X_extra 形状 (1, len(expected_cols))。
    """
    if cohort not in V4_SPECS:
        raise ValueError(f'队列 {cohort!r} 无 v4 spec')
    row = pd.DataFrame([dict(record)])
    # 源列补齐：记录缺整个字段（如无 gender/ifng_genotype）时按 NaN
    # 进入 spec——与训练时"字段缺失"语义完全一致（中位数 + 指示器=1），
    # 避免 build_v4_extras 的 df[src] 直接 KeyError
    for _out, src, _mode, _kw in V4_SPECS[cohort]:
        if src not in row.columns:
            row[src] = np.nan
    cols, X, audit = build_v4_extras(row, cohort, fill='none')
    # 训练中位数填补（NaN 保留到这里才填）
    if X.size:
        med = np.array([float(fill_medians.get(c, 0.0)) for c in cols])
        X = np.where(np.isnan(X), med, X)
    frame = pd.DataFrame(X, columns=cols)
    # 合法缺席列（非漂移）：单样本构造下，记录命中基类/其他类别时该
    # onehot 类别列不产出（语义=非该类 → 0），记录不缺失时指示器列不
    # 产出（语义=未缺失 → 0）——与训练口径逐值一致，不构成漂移
    conditional = set()
    for out_col, _src, mode, kw in V4_SPECS[cohort]:
        if mode == 'onehot':
            prefix = f'{out_col}_'
            conditional.update(
                c for c in expected_cols if c.startswith(prefix))
        if 'indicator' in kw:
            conditional.add(kw['indicator'])
    drift = [c for c in expected_cols
             if c not in cols and c not in conditional]
    if drift:
        audit['spec_drift_filled_zero'] = drift
        LOGGER.warning(
            "v4 部署列集漂移：%d 个训练列不在当前 spec 产物中（补 0）: %s",
            len(drift), drift[:4])
    frame = frame.reindex(columns=list(expected_cols), fill_value=0.0)
    return frame.to_numpy(dtype=float), audit
