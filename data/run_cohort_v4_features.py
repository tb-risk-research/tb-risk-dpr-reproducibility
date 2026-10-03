#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""族感知特征空间 v4：22 维 + 队列专属信息列（任务2，F2/S10）

背景（用户判读，2026-09-15）：组感知 CV 显示 22 维特征对部分队列判别
无信息（treats 0.438 低于随机），根因是特征空间与终点的错配——22 维中
11 列在真实队列是死的（合成场景产物），而各队列真正有信息的列被丢弃：
CRP 的 cad_score/crp_mgdl/症状组合、Taiwan 的 QFT/CXR、TREATS 的
roomshare_n/hh_tb_contact、Peru 的 index_smear_grade/index_cough_days、
NHANES 的 bmi/smoking。瓶颈不在训练方法，在特征空间。

设计：per-cohort v4 = 22 维 + 队列专属非泄漏 extras（精选以控共线），
配对 CV（LR + LightGBM 同折，与 22 维 v3 基线同折对比）→ v4−v3 AUROC
配对差 + cluster/个体 bootstrap CI。回答"被丢弃的队列信息列值多少"。

泄漏审计（按终点族的结构规则）：
  - LTBI 族（nhanes/treats/brazil）排除 TST/IGRA 结果列：
    brazil tst_result_raw（终点由 TST 定义）、group_raw（分组即标签）；
  - 确诊/活动性 TB 族（kenya/crp/taiwan）排除培养/涂片结果列：
    taiwan culture_result / smear_cat；
  - 影像分数（cad_score/cxr_*）与炎症标志（crp_mgdl）为合法协变量；
  - 冗余排除：crp_positive（crp_mgdl 阈值化派生）、qft_nil/t1/t2/cd8
    （qft_tb_ag_nil+qft_result_pos 已代表）、past_illness_type
    （22 维 past_illness 已捕捉）。
  - kenya（P6 原始数据再审）：细菌学结果族/痰标本级联族（送检由
    症状+CXR 决定，lab_sputum_requested 单变量 0.915——巴基斯坦同构）/
    现症治疗族排除；cxr 终判读（xfindingall，覆盖 99.1%）按影像
    家族规则为合法协变量。
被排除列全部进单变量 AUROC 披露表（含泄漏列——其"过好"的判别正是
泄漏证据）。

数据质量哨兵（审计发现）：
  - crp_mgdl == 999 → NaN（疑似缺失哨兵）；
  - income_pir < 1e-6 → NaN（最小值 5.4e-79 为零替换产物）；
  - cad_score == -1 → NaN（合法分数范围 0-100）；
  - brazil ethnic == '4' → 语义可疑，与缺失合并走指示器。

缺失处理（项目惯例）：数值列折内中位数填补；缺失率 ≥1% 的列加缺失
指示器；二值列缺失填 0 + 指示器（taiwan cxr_* 共享 cxr_missing）。
LR 折内 z 标准化；LGBM 用填补后原尺度（注册表默认参数）。

用法：
    python data/run_cohort_v4_features.py [--n-bootstrap 1000]
        [--folds 5] [--seed 42]
输出：
    data/processed/cohort_v4_features_<date>.json
"""
import argparse
import datetime
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import ensure_interaction_features  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import lightgbm as lgb  # noqa: E402

# ---- 队列注册表（与 run_multicohort_pooling.py 一致） ----
COHORTS = {
    'nhanes':   {'file': 'ml_training_nhanes_real.csv',  'group_col': None,
                 'endpoint_family': 'ltbi',       'endpoint': 'LTBI (IGRA)'},
    'treats':   {'file': 'ml_training_treats_real.csv',  'group_col': 'group_id',
                 'endpoint_family': 'ltbi',       'endpoint': 'LTBI (TST/IGRA)'},
    'crp':      {'file': 'ml_training_crp_real.csv',     'group_col': None,
                 'endpoint_family': 'confirmed_tb', 'endpoint': '确诊 TB'},
    'kenya':    {'file': 'kenya_ml_training.csv',        'group_col': None,
                 'endpoint_family': 'confirmed_tb', 'endpoint': '细菌学确诊 TB'},
    'taiwan':   {'file': 'ml_training_taiwan_real.csv',  'group_col': None,
                 'endpoint_family': 'confirmed_tb', 'endpoint': '活动性 TB'},
    'brazil':   {'file': 'ml_training_brazil_real.csv',  'group_col': None,
                 'endpoint_family': 'ltbi',       'endpoint': 'LTBI (TST)'},
    'peru_mdr': {'file': 'ml_training_peru_mdr_real.csv', 'group_col': 'family_id',
                 'endpoint_family': 'incident_tb', 'endpoint': 'incident TB（MDR 密切接触者）'},
}

# ---- v4 增量列 spec：(out_col, src, mode, kwargs) ----
# mode: binary_map / as_is / num / onehot
#   num: 数值列，哨兵置 NaN 后折内中位填补；kwargs 可带 sentinel_eq /
#        sentinel_lt / indicator（缺失指示器列名）
V4_SPECS = {
    'nhanes': [
        ('gender_female', 'gender', 'binary_map', {'pos': 'f'}),
        ('bmi', 'bmi', 'num', {}),
        ('smoking', 'smoking', 'as_is', {}),
        ('born_us', 'born_us', 'as_is', {}),
        ('income_pir', 'income_pir', 'num',
         {'sentinel_lt': 1e-6, 'indicator': 'income_pir_missing'}),
        ('race_eth', 'race_eth', 'onehot', {'base': 1.0}),  # 5 级 → 4 dummies
    ],
    'treats': [
        ('gender_female', 'gender', 'binary_map', {'pos': 'f'}),
        ('smoking_status', 'smoking_status', 'onehot', {'base': 0}),  # 3 级 → 2
        ('roomshare_n', 'roomshare_n', 'as_is', {}),
        ('hhdens_quartile', 'hhdens_quartile', 'as_is', {}),
        ('hh_tb_contact', 'hh_tb_contact', 'num', {}),
        ('alcohol_use', 'alcohol_use', 'onehot', {'base': 0}),  # 4 级 → 3
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
        ('tb_history', 'tb_history_raw', 'onehot',
         {'base': 'no'}),  # 3 级 → 2（既往治疗/正在治疗）
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
        ('cxr_fibronodular', 'cxr_fibronodular', 'as_is', {}),
        ('cxr_cavitation', 'cxr_cavitation', 'as_is', {}),
        ('cxr_pleural_effusion', 'cxr_pleural_effusion', 'as_is', {}),
        ('ltbi_class', 'ltbi_class_raw', 'onehot', {'base': 1}),  # 3 级 → 2
    ],
    'brazil': [
        ('gender_female', 'gender', 'binary_map', {'pos': 'f'}),
        ('ifng_genotype', 'ifng_genotype', 'onehot',
         {'base': 'AA'}),  # AA/TA/TT → 2 dummies
        ('ethnic_white', 'ethnic', 'binary_map', {'pos': 'WHITE'}),
        ('ethnic_unknown', 'ethnic', 'unknown_ind',
         {'unknown': ['4']}),  # '4' 语义可疑 + 缺失 → 指示器
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
    # extras；与 ml/cohort_features.py 单一真值源同步
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

# ---- 泄漏/冗余排除登记（全部进单变量 AUROC 披露表） ----
EXCLUSIONS = {
    'brazil': {
        'tst_result_raw': 'leak_endpoint_defining（LTBI 由 TST 定义）',
        'group_raw': 'leak_endpoint_defining（TB/TST 分组即标签）',
        'past_illness_type': 'redundant（全为 none；22 维 past_illness 已捕捉）',
        'gender': 'included（as gender_female）',
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
        'gender': 'included（as gender_female）',
    },
    'crp': {
        'crp_positive': 'redundant（crp_mgdl 阈值化派生）',
        'age_group': 'redundant（age 的粗化分组，22 维 age 已进）',
        'past_illness_type': 'redundant（22 维 past_illness 已捕捉）',
        'gender': 'included（as gender_female）',
    },
    'nhanes': {
        'past_illness_type': 'redundant（22 维 past_illness 已捕捉）',
        'gender': 'included（as gender_female）',
    },
    'treats': {
        'past_illness_type': 'redundant（全为 none；22 维 past_illness 已捕捉）',
        'gender': 'included（as gender_female）',
    },
    'peru_mdr': {
        'past_illness_type': 'redundant（22 维 past_illness/is_high_risk 已捕捉）',
        'age_band': 'redundant（age 的粗化分组）',
        'follow_up_days': 'excluded_survival_only（时间轴列，属 F4 生存分析）',
        'gender': 'included（as gender_female）',
    },
    # kenya（P6）：参照列由 ETL 发射进 CSV 供本披露表计算单变量 AUROC
    'kenya': {
        'lab_smear_pos': 'leak_endpoint_defining（涂片阳性为终点组分）',
        'lab_sputum_requested': 'leak_cascade_embedded（送检由症状+CXR 决定，'
                                '单变量 AUROC 0.915——巴基斯坦同构证据）',
        'tb_current_treatment': 'leak_outcome_adjacent（现症治疗与患病终点同源）',
        'gender': 'included（as gender_female）',
    },
}

# 二值列缺失填 0 后仍需指示器的列（缺失即信息：未测 ≠ 阴性）
BINARY_MISSING_IND = {
    'crp': {'crp_positive': 'crp_missing'},  # 仅披露用（列本身已冗余排除）
    'taiwan': {'cxr_fibronodular': 'cxr_missing',
               'cxr_cavitation': 'cxr_missing',
               'cxr_pleural_effusion': 'cxr_missing'},
}

LGBM_KWARGS = dict(n_estimators=300, num_leaves=31, learning_rate=0.1,
                   min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                   verbosity=-1, class_weight='balanced', random_state=42)
LR_KWARGS = dict(max_iter=2000, random_state=42)


def build_extras(df, cohort):
    """全数据级 extras 变换：哨兵置 NaN、onehot、binary_map、指示器。

    返回 (out_cols, X_extra)；数值列 NaN 保留（折内中位填补）。
    """
    spec = V4_SPECS[cohort]
    cols, arrays = [], []
    for out_col, src, mode, kw in spec:
        s = df[src]
        if mode == 'binary_map':
            col = (s.astype(str).str.strip().str.lower() == kw['pos'])
            col = col.astype(float).where(s.notna(), np.nan)
        elif mode == 'unknown_ind':  # '4'/缺失 → 1
            col = (s.isna() | s.astype(str).isin(kw['unknown'])).astype(float)
        elif mode == 'as_is':
            col = pd.to_numeric(s, errors='coerce')
        elif mode == 'num':
            col = pd.to_numeric(s, errors='coerce')
            if 'sentinel_eq' in kw:
                col = col.mask(col == kw['sentinel_eq'], np.nan)
            if 'sentinel_lt' in kw:
                col = col.mask(col < kw['sentinel_lt'], np.nan)
        elif mode == 'onehot':
            cats = pd.unique(s.dropna())
            base = kw['base']
            for c in sorted(cats, key=str):
                if c == base:
                    continue
                # 命名与 ml/cohort_features.py（单一真值源）同步：下划线
                # 连接——主管线把 X 包成带名 DataFrame（H-ML3 契约），
                # LightGBM 拒绝方括号等特殊 JSON 字符
                cols.append(f'{out_col}_{c}')
                arrays.append((s == c).astype(float).to_numpy())
            if 'indicator' in kw:  # onehot 缺失指示器（如有）
                cols.append(kw['indicator'])
                arrays.append(s.isna().astype(float).to_numpy())
            continue
        else:
            raise ValueError(mode)
        # 数值列 → 指示器 + NaN 保留
        if 'indicator' in kw:
            miss_rate = float(col.isna().mean())
            if miss_rate >= 0.01:
                cols.append(kw['indicator'])
                arrays.append(col.isna().astype(float).to_numpy())
            elif col.isna().any():
                cols.append(kw['indicator'])
                arrays.append(col.isna().astype(float).to_numpy())
        cols.append(out_col)
        arrays.append(col.astype(float).to_numpy())
    X = np.column_stack(arrays) if arrays else np.zeros((len(df), 0))
    return cols, X


def univariate_table(df, cohort, y):
    """单变量 AUROC 披露表：v4 spec 列 + 排除列（含泄漏列）。"""
    rows = []
    included_srcs = {src for _, src, _, _ in V4_SPECS.get(cohort, [])}
    # v4 spec 列
    for out_col, src, mode, kw in V4_SPECS.get(cohort, []):
        s = pd.to_numeric(_to_series(df, out_col, src, mode, kw),
                          errors='coerce')
        ok = s.notna()
        if ok.sum() < 10 or s[ok].std() < 1e-12:
            rows.append({'column': out_col, 'status': 'included',
                         'univar_auroc': None,
                         'note': '常数或样本不足'})
            continue
        a = float(roc_auc_score(y[ok.to_numpy()], s[ok]))
        rows.append({'column': out_col, 'status': 'included',
                     'univar_auroc': round(a, 4),
                     'n_avail': int(ok.sum())})
    # 排除列
    for col, reason in EXCLUSIONS.get(cohort, {}).items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce')
        ok = s.notna()
        if ok.sum() >= 10 and s[ok].std() > 1e-12:
            a = float(roc_auc_score(y[ok.to_numpy()], s[ok]))
            rows.append({'column': col, 'status': 'excluded',
                         'reason': reason, 'univar_auroc': round(a, 4),
                         'n_avail': int(ok.sum())})
        else:
            rows.append({'column': col, 'status': 'excluded',
                         'reason': reason, 'univar_auroc': None})
    return rows


def _to_series(df, out_col, src, mode, kw):
    """为单变量表重算 spec 列（哨兵处理后）——与 build_extras 同规则。"""
    s = df[src]
    if mode == 'binary_map':
        col = (s.astype(str).str.strip().str.lower() == kw['pos'])
        return col.astype(float).where(s.notna(), np.nan)
    if mode == 'unknown_ind':
        return (s.isna() | s.astype(str).isin(kw['unknown'])).astype(float)
    col = pd.to_numeric(s, errors='coerce')
    if mode == 'num':
        if 'sentinel_eq' in kw:
            col = col.mask(col == kw['sentinel_eq'], np.nan)
        if 'sentinel_lt' in kw:
            col = col.mask(col < kw['sentinel_lt'], np.nan)
    return col


def _fold_fill(X_tr, X_te):
    """折内中位数填补（训练折拟合；全 NaN 列填 0）。"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return (np.where(np.isnan(X_tr), med, X_tr),
            np.where(np.isnan(X_te), med, X_te))


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def _bootstrap_delta(y, s_v3, s_v4, groups, n_bootstrap, seed):
    """AUROC(v4) − AUROC(v3) 的重采样 CI（组感知队列按组，否则按个体）。"""
    n = len(y)
    rng = np.random.RandomState(seed)
    if groups is not None:
        uniq = np.unique(groups)
        by_g = {g: np.where(groups == g)[0] for g in uniq}
    deltas, a3s, a4s = [], [], []
    for _ in range(n_bootstrap):
        idx = (np.concatenate([by_g[g] for g in
                               rng.choice(uniq, size=len(uniq), replace=True)])
               if groups is not None else
               rng.randint(0, n, size=n))
        if len(np.unique(y[idx])) < 2:
            continue
        a3, a4 = _auc(y[idx], s_v3[idx]), _auc(y[idx], s_v4[idx])
        if a3 is None or a4 is None:
            continue
        deltas.append(a4 - a3)
        a3s.append(a3)
        a4s.append(a4)
    if not deltas:
        return None, None, None, 0
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return ([float(lo), float(hi)], float(np.mean(a3s)),
            float(np.mean(a4s)), len(deltas))


def run():
    parser = argparse.ArgumentParser(description='族感知 v4 特征空间配对 CV')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cohorts', type=str, default=None,
                        help='逗号分隔队列子集（冒烟用）')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    date_tag = datetime.date.today().strftime('%Y%m%d')
    out_path = args.out or os.path.join(
        HERE, 'processed', f'cohort_v4_features_{date_tag}.json')
    # LGBM sklearn 接口对 numpy 输入的 feature names 警告（无害，数值不变）
    warnings.filterwarnings(
        'ignore', message='X does not have valid feature names')

    sel = (args.cohorts.split(',') if args.cohorts
           else list(COHORTS))
    t_start = time.time()
    print('== 族感知 v4 特征空间（22 维 + 队列专属信息列）配对 CV ==')
    print(f'--folds={args.folds}, --n-bootstrap={args.n_bootstrap}, '
          f'--seed={args.seed}, cohorts={sel}')

    pred = MLRiskPredictor()
    results = {}
    for name in sel:
        meta = COHORTS[name]
        path = os.path.join(HERE, 'processed', meta['file'])
        df = pd.read_csv(path, encoding='utf-8-sig')
        df, missing = ensure_interaction_features(
            df, pred.INTERACTION_FEATURE_NAMES)
        X22 = df[list(pred.ALL_FEATURE_NAMES)].astype(float).to_numpy()
        y = df['tb_outcome'].astype(int).to_numpy()
        groups = (df[meta['group_col']].to_numpy()
                  if meta['group_col'] and meta['group_col'] in df.columns
                  else None)

        if name not in V4_SPECS or not V4_SPECS[name]:
            # 防御分支：7 队列全部有 spec 后不再触发（历史 kenya 跳过
            # 分支已由 P6 移除——kenya 现走下方通用路径）
            print(f'\n[{name}] 无 extra 列 → v4 = v3，跳过（n={len(y)}, '
                  f'pos={int(y.sum())}）')
            results[name] = {
                'n': int(len(y)), 'n_pos': int(y.sum()),
                'v4_note': '无 22 维之外的列：v4 与 v3 相同，跳过',
            }
            continue

        extra_cols, X_extra = build_extras(df, name)
        X_v4 = np.hstack([X22, X_extra])
        n22 = X22.shape[1]
        # 22 维在真实队列的活列数（对照披露）
        n_live22 = int((X22.std(axis=0) > 1e-12).sum())

        print(f'\n[{name}] n={len(y)}, pos={int(y.sum())} ({y.mean():.3f}), '
              f'22 维活列 {n_live22}/22，v4 增量 {len(extra_cols)} 列 '
              f'(总 {X_v4.shape[1]})')
        print(f'  extras: {extra_cols}')

        # ---- CV（同折：v3/v4 × LR/LGBM）----
        if groups is not None:
            cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                      random_state=args.seed)
            folds = list(cv.split(X_v4, y, groups))
            cv_desc = (f'StratifiedGroupKFold({args.folds}) by '
                       f'{meta["group_col"]}（{len(np.unique(groups))} 组）')
        else:
            cv = StratifiedKFold(n_splits=args.folds, shuffle=True,
                                 random_state=args.seed)
            folds = list(cv.split(X_v4, y))
            cv_desc = f'StratifiedKFold({args.folds})'
        oof = {k: np.full(len(y), np.nan) for k in
               ('lr_v3', 'lr_v4', 'lgbm_v3', 'lgbm_v4')}
        for tr, te in folds:
            # v3（22 维无 NaN）与 v4（extras 折内中位填补）分别准备
            X22tr, X22te = X22[tr], X22[te]
            Xe_tr, Xe_te = _fold_fill(X_extra[tr], X_extra[te])
            X4tr, X4te = np.hstack([X22tr, Xe_tr]), np.hstack([X22te, Xe_te])
            # LR：z 标准化（0 方差列除 1 保护）
            def _zs(a, b):
                mu, sd = a.mean(0), a.std(0)
                sd = np.where(sd < 1e-12, 1.0, sd)
                return (a - mu) / sd, (b - mu) / sd
            Z3tr, Z3te = _zs(X22tr, X22te)
            Z4tr, Z4te = _zs(X4tr, X4te)
            oof['lr_v3'][te] = LogisticRegression(**LR_KWARGS).fit(
                Z3tr, y[tr]).predict_proba(Z3te)[:, 1]
            oof['lr_v4'][te] = LogisticRegression(**LR_KWARGS).fit(
                Z4tr, y[tr]).predict_proba(Z4te)[:, 1]
            oof['lgbm_v3'][te] = lgb.LGBMClassifier(**LGBM_KWARGS).fit(
                X22tr, y[tr]).predict_proba(X22te)[:, 1]
            oof['lgbm_v4'][te] = lgb.LGBMClassifier(**LGBM_KWARGS).fit(
                X4tr, y[tr]).predict_proba(X4te)[:, 1]

        # ---- 指标 + 配对差 CI ----
        entry = {
            'n': int(len(y)), 'n_pos': int(y.sum()),
            'endpoint': meta['endpoint'],
            'n_live_22dim': n_live22,
            'v4_extra_cols': extra_cols,
            'cv': cv_desc,
            'models': {},
        }
        for mkey in ('lr', 'lgbm'):
            s3, s4 = oof[f'{mkey}_v3'], oof[f'{mkey}_v4']
            a3, a4 = _auc(y, s3), _auc(y, s4)
            d_ci, a3_bmean, a4_bmean, n_eff = _bootstrap_delta(
                y, s3, s4, groups, args.n_bootstrap, args.seed)
            entry['models'][mkey] = {
                'auroc_v3': round(a3, 4), 'auroc_v4': round(a4, 4),
                'delta_v4_minus_v3': round(a4 - a3, 4),
                'delta_ci95_bootstrap': ([round(v, 4) for v in d_ci]
                                         if d_ci else None),
                'delta_bootstrap_mean': round(a4_bmean - a3_bmean, 4),
                'n_bootstrap_effective': n_eff,
            }
            m = entry['models'][mkey]
            sig = (d_ci is not None
                   and (d_ci[0] > 0 or d_ci[1] < 0))
            print(f'  [{mkey}] v3={a3:.4f} → v4={a4:.4f} '
                  f'(Δ={a4-a3:+.4f}, CI95={m["delta_ci95_bootstrap"]}'
                  f'{", 显著" if sig else ""})')
        entry['univariate_audit'] = univariate_table(df, name, y)
        results[name] = entry

    # ---- 汇总输出 ----
    output = {
        'experiment': 'cohort_v4_family_aware_features',
        'date': date_tag,
        'runtime_seconds': round(time.time() - t_start, 1),
        'protocol': {
            'design': ('per-cohort v4 = 22 维 + 队列专属非泄漏 extras；'
                       'LR + LightGBM 同折配对 CV（v3/v4 同折），'
                       'v4−v3 AUROC 配对差 + 重采样 CI'),
            'models': {
                'lr': 'LogisticRegression(max_iter=2000)，折内 z 标准化',
                'lgbm': 'LGBMClassifier 注册表默认（300 树/balanced）',
            },
            'leak_rules': ('LTBI 族排除 TST/IGRA 结果列（brazil '
                           'tst_result_raw/group_raw）；确诊族排除培养/涂片'
                           '（taiwan culture_result/smear_cat）；kenya 排除'
                           '细菌学结果族/痰标本级联族（lab_sputum_requested '
                           '0.915）/现症治疗族；影像分数与炎症标志为合法'
                           '协变量；排除列全部进单变量 AUROC 披露表'),
            'sentinels': ('crp_mgdl==999→NaN；income_pir<1e-6→NaN；'
                          'cad_score==-1→NaN；brazil ethnic=="4"→未知指示器'),
            'missing': ('数值列折内中位填补；缺失≥1% 加指示器；taiwan '
                        'cxr_* 共享 cxr_missing；kenya breathless（26.6%）/'
                        'treatment_sought（62.5%）条件缺失走指示器'),
            'kenya_note': ('P6 原始数据再审（2026-09-15）：kenya 不再是零'
                           '增益队列——新增 16 extras（gender/cxr 终判读/'
                           '7 症状分辨率/cough_weeks/treatment_sought/'
                           'hiv_tested），历史封顶 13 列 0.73 的口径仅适用'
                           '于 P6 之前的 v3 基线'),
            'context': ('组感知 CV 诚实基线（multicohort_pooling_20260914 '
                        'arm_a）：nhanes 0.7289 / treats 0.4376 / crp 0.5805 '
                        '/ kenya 0.7678 / taiwan 0.5538 / brazil 0.6921 / '
                        'peru_mdr 0.6296。v3 列为本脚本同折复算（折内 z '
                        '标准化；arm_a 无标准化）——两者在 22 维近死队列'
                        '（treats 1/22、brazil 2/22 活列）上绝对值有差异'
                        '（treats 0.536 vs 0.438：标准化后死列归零、活列'
                        '权重更稳），其余队列一致（±0.02）。Δ(v4−v3) 为'
                        '同折同预处理配对差，不受此影响'),
        },
        'cohorts': results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n输出 → {out_path}（{time.time()-t_start:.0f}s）')

    # 控制台速览
    print('\n== 速览（v3 → v4 AUROC，配对同折）==')
    print(f'{"cohort":<10}{"LR v3":>8}{"LR v4":>8}{"Δ":>9}'
          f'{"LGBM v3":>9}{"LGBM v4":>9}{"Δ":>9}')
    for name in sel:
        if 'models' not in results.get(name, {}):
            print(f'{name:<10}（跳过：无 extra 列）')
            continue
        lr = results[name]['models']['lr']
        lb = results[name]['models']['lgbm']
        print(f'{name:<10}{lr["auroc_v3"]:>8.4f}{lr["auroc_v4"]:>8.4f}'
              f'{lr["delta_v4_minus_v3"]:>+9.4f}'
              f'{lb["auroc_v3"]:>9.4f}{lb["auroc_v4"]:>9.4f}'
              f'{lb["delta_v4_minus_v3"]:>+9.4f}')


if __name__ == '__main__':
    run()
