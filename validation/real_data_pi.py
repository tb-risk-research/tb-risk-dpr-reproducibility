#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据 PI 验证（P4b，2026-08-25）：马拉维 PACTS 家庭接触者试验。

【定位降级（2026-08-25 P4 复审）】本数据集从"PI 检验场"降级为
"方向性佐证"：主终点 sx_3m 与 3m 随访问卷同访采集（终点泄漏结构
性存在，防泄漏装载器只能保住合法性、救不了饱和），疾病终点 9 事件
不可判。够功效的 PI 检验已迁移至感染终点（real_data_infection.py
HomeACF，359 事件）。

统一对外叙事（四实验证据链，2026-08-25 第三轮定稿）——
"结构来自物理、参数来自数据、方向已被真实梯度验证"：
  ① 合成实验证明架构可学：时序机制 +0.032（temporal_ablation）、
     物理锚定救命级（pi−gnn +0.094 20/20）、特征化兑现 60-63%
     （combined_ablation）；
  ② 真实数据证明方向多数正确但刻度需校准：乘子审计 5/8 方向
     一致，效应量偏差 0.15~2.7 倍（lambda_calibration——
     share_bedroom ×2.72 被低估最重、cough_days ×0.36 高估；
     age_lt5/index_hiv/sleep_same_bed 方向翻转）；
  ③ 校准机制验证有效：特征级从"显著有害"（冻结 λ −0.0080 CI
     排除 0）修复为"不再有害"（校准 λ −0.0015 CI 含 0）；
  ④ 家庭聚集强信号存在但需时序建模兑现：同户先证 TST 率
     AUROC 0.715 > 全部学习臂 0.645，部署时序不可得 →
     时序留出两阶段筛查（household_temporal.py）把它转为
     可部署先证特征。
当前真实数据覆盖的仍是感染终点；进展终点（任务 B）在两个数据集
事件数均不足（9 与 7），等待 ERASE-TB 级别纵向队列。

数据：Kaswaswa/MacPherson et al. PLOS ONE 2022;17(5):e0269219
（patient-delivered household contact tracing, Blantyre Malawi;
家庭随机试验 PACTS vs 标准照护）。S1 Dataset（期刊补充材料，开放获取）：
838 名家庭接触者 / 197 户 / 指示病例涂片+培养状态可用。

科学问题（对齐 P1/P2 特征化阶梯 + PI 机制）：在真实稀疏终点上，
"物理感染力特征 λ"相对个体协变量、指示病例特征、暴露结构特征的
边际判别增益是否 > 0——即 PI 机制的工程结论（DGP 世界的 60-63%
兑现比例、+0.09 AUROC）在真实数据上是否方向成立。

两层终点（真实数据的结构性约束）：
  主终点 sx_3m（3 月症状筛查阳性，~13%）——但基线症状计数单变量
    AUROC 0.962：症状持久性主导，临床基线近乎饱和信号；阶梯问题
    仍合法（PI 在临床基线之上的边际），但预期增量为噪声级；
  辅终点 m3_tb_out（TB 结局，~9 事件）——疾病终点，单试验不可判，
    只报迁移点估计；
  物理方向检查 physics_direction_report——λ 四分位事件率梯度、
    涂片+/−、年龄 U 形、基线无症状亚群 λ 排序（部署目标层，
    事件极稀，仅方向证据）。

臂结构（嵌套特征集，主模型 = 冻结注册表 RF/LGBM）：
  ind       个体协变量（真基线文件）：年龄/年龄组、性别、<5 岁、
            教育、既往 TB 疑似、基线症状计数、试验臂、财富五分位
  index     + 指示病例特征：涂片阳性、HIV、年龄、性别、家庭规模
  exposure  + 暴露结构（P1 typed 特征化的真实数据版）：与指示病例
            关系、同住/过夜、接触场所/时长/频率
  pi        + 物理感染力 λ（1 列，单调于 Wells-Riley 式累积暴露）
  pi_only   纯 λ 排序（不学习）——物理排序本身的判别力

λ 构造（文献先验，冻结系数，不拟合）：
  λ_i = S(age_i) · I_index · C_i
  S(age)   宿主易感性 U 形先验：<5 岁 ×1.8、≥45 岁 ×1.6（儿童与
           老年进展率高；Marais 等儿童接触者进展数据）
  I_index  指示病例感染力：涂片+培养+ ×3.0（Grzybowski 传播研究
           涂片阳性传染比）、HIV+ ×1.3、年龄线性修正
  C_i      接触强度：同住 ×1.0 / 非同住 ×0.5

诚实边界：
  - 3m 文件 p 列为随访问卷（与终点同访，p30tbsus 单变量 AUROC
    0.993 ≈ 终点）——特征一律取基线文件（复合键连接），根除同访
    泄漏；
  - 终点 sx_3m 是分诊终点而非感染/发病金标准，且被基线临床状态
    饱和；疾病终点 9 事件不可判——本验证的真实信息量主要在"方向
    检查"与"饱和度陈述"，非效应量证明；
  - 涂片+/− 的 sx_3m 事件率 13.8% vs 12.9%——物理方向成立但信号
    弱（症状终点 + smear−culture+ 同样可传染）；
  - 指示病例症状时长缺失 → λ 无时间维（DGP 的窗口化 Λ(t_w) 不可
    复原），λ 为强度型而非轨迹型特征；
  - 感染终点（TST/IGRA）数据集（如 Adjobimey 2025 贝宁簇随机试
    验，Vivli PR00011500，需申请）才是 PI 验证的够功效路径；
  - 结论口径 = 单试验外部验证，非多数据集荟萃。

协议：household 分层分组 5 折 CV × 多种子（同折配对）；
户级 cluster bootstrap CI；逐种子 DeLong 配对。
"""

import json
import os
import time

import numpy as np
import pandas as pd

from .layer_ablation import delong_paired_test
from .threshold_spec import compute_auc

MODEL_KEYS = ('random_forest', 'lightgbm')
FEATURE_ARMS = ('ind', 'index', 'exposure', 'pi')

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(
    _REPO_ROOT, 'data', 'raw',
    'malawi_pacts_s002_x', 'S1_Publishing data set_2', '_Data')

# λ 文献先验（冻结，禁止用结局拟合——物理特征口径）
LAMBDA_PRIORS = {
    'susceptibility_lt5': 1.8,      # <5 岁宿主易感性倍数
    'susceptibility_ge45': 1.6,     # ≥45 岁宿主易感性倍数
    'index_smear_pos': 3.0,         # 涂片+培养+ 感染力倍数
    'index_hiv_pos': 1.3,           # 指示病例 HIV+ 感染力倍数
    'index_age_slope': 0.005,       # 指示病例年龄线性修正/岁
    'contact_lives_in': 1.0,        # 同住接触强度
    'contact_not_lives_in': 0.5,    # 非同住接触强度
}

_BASELINE_SYMPTOM_COLS = ('p16cgh', 'p17doc', 'p18fvr', 'p19nsa',
                          'p20nsb', 'p21wla', 'p22wlb')


def load_pacts_contacts(data_dir=None):
    """装载并清洗 PACTS 接触者队列（基线问卷 + 3 月随访终点）。

    关键防泄漏设计：3m 文件中的 p 列为随访问卷更新值（与终点同访，
    p30tbsus 单变量 AUROC 0.993 ≈ 终点本身），故一切特征一律取自
    contacts_baseline_rev.csv（真基线），终点与指示病例列取自
    contacts_threemonths_rev.csv；两表以复合键
    (id, p01apatid, p05age, p06sex) 连接（两文件内均唯一）。

    Returns:
        pd.DataFrame: 一行一接触者；'id' 列 = 指示病例（户）簇键。
    """
    data_dir = data_dir or _DATA_DIR
    b = pd.read_csv(os.path.join(data_dir, 'contacts_baseline_rev.csv'),
                    low_memory=False)
    t = pd.read_csv(os.path.join(data_dir, 'contacts_threemonths_rev.csv'),
                    low_memory=False)
    idx = pd.read_csv(os.path.join(data_dir, 'indexcases.csv'),
                      low_memory=False)
    hh = pd.read_csv(os.path.join(data_dir, 'index_baseline.csv'),
                     low_memory=False)
    hh = hh[['id', 'hh_size', 'hh_children', 'hh_u5']]

    key = ['id', 'p01apatid', 'p05age', 'p06sex']
    base_cols = key + ['p15edu', 'p10rel', 'p07lvn', 'p08nght', 'p35tcs',
                       'p36plac', 'p41tbcls'] + list(_BASELINE_SYMPTOM_COLS) \
        + ['p30tbsus']
    bm = b[~b.duplicated(key)][base_cols].rename(
        columns={c: c + '_b' for c in base_cols[4:]})

    t = t[pd.to_numeric(t['sx_3m'], errors='coerce').notna()]
    df = t.merge(bm, on=key, how='left',
                 suffixes=('', '_dup'), indicator=True)
    # 未匹配基线问卷的行必须剔除：哑元派生会把 NaN 变成全 0（假数据）
    df = df[df['_merge'] == 'both'].drop(columns=['_merge'])
    df = df.merge(idx, on='id', how='left', suffixes=('_contact', '_idx'))
    df = df.merge(hh, on='id', how='left')
    df = _derive_features(df)
    return df.reset_index(drop=True)


def _derive_features(df):
    """派生建模列（全部为基线可观测，无随访泄漏）。"""
    out = df.copy()

    # ---- 个体协变量（ind 臂）----
    out['contact_age'] = pd.to_numeric(out['p05age'], errors='coerce')
    out['age_lt5'] = (out['contact_age'] < 5).astype(float)
    out['age_ge45'] = (out['contact_age'] >= 45).astype(float)
    out['contact_sex_m'] = (pd.to_numeric(
        out['p06sex'], errors='coerce') == 1).astype(float)
    out['educ_no_school'] = (pd.to_numeric(
        out['p15edu_b'], errors='coerce') == 1).astype(float)
    out['educ_primary'] = (pd.to_numeric(
        out['p15edu_b'], errors='coerce') == 2).astype(float)
    out['prior_tb_suspected'] = (pd.to_numeric(
        out['p30tbsus_b'], errors='coerce') == 1).astype(float)
    sym = np.zeros(len(out))
    for col in _BASELINE_SYMPTOM_COLS:
        sym += (pd.to_numeric(out[col + '_b'], errors='coerce')
                == 1).astype(int)
    out['baseline_symptom_count'] = sym
    out['arm_intervention'] = (
        out['group_contact'] == 'Intervention_Pacts').astype(float)
    for short, level in (('wealth_a_Les', 'a_Least poor'),
                         ('wealth_b_Less', 'b_Less poor than average'),
                         ('wealth_c_Aver', 'c_Average'),
                         ('wealth_d_Poor', 'd_Poorer than average')):
        out[short] = (out['wealth_quintile'] == level).astype(float)

    # ---- 指示病例特征（index 臂）----
    out['idx_smear_pos'] = out['tbtype_idx'].astype(str).str.startswith(
        'Smr +ve').astype(float)
    out['idx_hiv_pos'] = (out['hiv_idx'] == 'b_HIV-positive').astype(float)
    out['idx_hiv_unknown'] = (out['hiv_idx'] == 'c_Unknown').astype(float)
    out['idx_age'] = pd.to_numeric(out['age_idx'], errors='coerce')
    out['idx_sex_m'] = (pd.to_numeric(
        out['sex_idx'], errors='coerce') == 1).astype(float)
    out['hh_size_f'] = pd.to_numeric(out['hh_size'], errors='coerce')
    out['hh_u5_f'] = pd.to_numeric(out['hh_u5'], errors='coerce')

    # ---- 暴露结构（exposure 臂，P1 typed 特征化的真实数据版）----
    rel = pd.to_numeric(out['p10rel_b'], errors='coerce')
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        out[f'rel_{int(v)}'] = (rel == v).astype(float)
    out['lives_in_household'] = (pd.to_numeric(
        out['p07lvn_b'], errors='coerce') == 1).astype(float)
    out['nights_shared'] = (pd.to_numeric(
        out['p08nght_b'], errors='coerce') == 1).astype(float)
    tcs = pd.to_numeric(out['p35tcs_b'], errors='coerce')
    for v in (1.0, 2.0, 3.0):
        out[f'tcs_{int(v)}'] = (tcs == v).astype(float)
    plac = pd.to_numeric(out['p36plac_b'], errors='coerce')
    for v in (8.0, 9.0):
        out[f'plac_{int(v)}'] = (plac == v).astype(float)
    cls = pd.to_numeric(out['p41tbcls_b'], errors='coerce')
    for v in (3.0, 9.0):
        out[f'tbcls_{int(v)}'] = (cls == v).astype(float)
    # 户内聚合（真实数据的"邻居聚合"：同户接触者环境压力）
    grp = out.groupby('id')
    out['hh_n_contacts'] = grp['id'].transform('size').astype(float)
    out['hh_symptom_share'] = grp['baseline_symptom_count'].transform(
        'mean').astype(float)

    # ---- 物理感染力 λ（pi 臂，冻结文献先验）----
    out['lam'] = build_lambda_real(out)

    # ---- 终点 ----
    out['sx_3m'] = pd.to_numeric(out['sx_3m'], errors='coerce')
    out['m3_tb_out'] = pd.to_numeric(out['m3_tb_out'], errors='coerce')
    return out


def build_lambda_real(df):
    """Wells-Riley 式乘性感染力：S(host) · I(index) · C(contact)。

    系数全部冻结（LAMBDA_PRIORS），禁止与结局拟合——纯物理排序口径。
    """
    p = LAMBDA_PRIORS
    s_host = 1.0 + p['susceptibility_lt5'] * df['age_lt5'] \
        + p['susceptibility_ge45'] * df['age_ge45']
    idx_age_centered = (df['idx_age'].fillna(35.0) - 35.0).clip(-20, 40)
    i_index = (p['index_smear_pos'] ** df['idx_smear_pos']) \
        * (p['index_hiv_pos'] ** df['idx_hiv_pos']) \
        * np.exp(p['index_age_slope'] * idx_age_centered)
    c_expo = p['contact_lives_in'] * df['lives_in_household'] \
        + p['contact_not_lives_in'] * (1.0 - df['lives_in_household'])
    return (s_host * i_index * c_expo).astype(float)


FEATURE_SETS = {
    'ind': [
        'contact_age', 'age_lt5', 'age_ge45', 'contact_sex_m',
        'educ_no_school', 'educ_primary', 'prior_tb_suspected',
        'baseline_symptom_count', 'arm_intervention',
        'wealth_a_Les', 'wealth_b_Less', 'wealth_c_Aver',
        'wealth_d_Poor',
    ],
    'index': [
        'idx_smear_pos', 'idx_hiv_pos', 'idx_hiv_unknown',
        'idx_age', 'idx_sex_m', 'hh_size_f', 'hh_u5_f',
    ],
    'exposure': [
        'rel_1', 'rel_2', 'rel_3', 'rel_4', 'rel_5',
        'lives_in_household', 'nights_shared',
        'tcs_1', 'tcs_2', 'tcs_3', 'plac_8', 'plac_9',
        'tbcls_3', 'tbcls_9',
        'hh_n_contacts', 'hh_symptom_share',
    ],
    'pi': ['lam'],
}

_OUTCOME_COLS = ('sx_3m', 'm3_tb_out', 'micro_confirmed',
                 's3_micro_confirmed', 'on_tb_rx', 'contact3m_smear',
                 'contact3m_culture', 'contact3m_xpert')


def feature_columns(arm):
    """臂的完整特征列（嵌套：ind ⊂ index ⊂ exposure ⊂ pi）。"""
    cols = []
    for a in FEATURE_ARMS[:FEATURE_ARMS.index(arm) + 1]:
        cols.extend(FEATURE_SETS[a])
    return cols


def physics_direction_report(df):
    """物理方向性检查（无模型，纯描述性证据）。

    Returns:
        dict: λ 四分位事件率、涂片/年龄梯度、基线无症状亚群 λ 排序。
    """
    lam = df['lam'].to_numpy(dtype=float)
    y = df['sx_3m'].astype(int).to_numpy()
    y_tb = df['m3_tb_out'].fillna(0).astype(int).to_numpy()

    q = np.quantile(lam, [0.25, 0.5, 0.75])
    stratum = np.digitize(lam, q)  # 0..3 = λ 四分位层
    lam_strata = {
        f'q{i+1}': {
            'n': int((stratum == i).sum()),
            'sx_3m_rate': float(y[stratum == i].mean()),
            'm3_tb_events': int(y_tb[stratum == i].sum()),
        } for i in range(4)}

    asym = df['baseline_symptom_count'].to_numpy() == 0
    out = {
        'lambda_quartile_sx_3m': lam_strata,
        'smear_gradient': {
            'smear_pos': {'n': int((df['idx_smear_pos'] == 1).sum()),
                          'sx_3m_rate': float(y[df['idx_smear_pos'] == 1].mean())},
            'smear_neg': {'n': int((df['idx_smear_pos'] == 0).sum()),
                          'sx_3m_rate': float(y[df['idx_smear_pos'] == 0].mean())},
        },
        'age_gradient': {
            band: {'n': int(m.sum()), 'sx_3m_rate': float(y[m].mean())}
            for band, m in (
                ('lt5', df['age_lt5'] == 1),
                ('mid', (df['age_lt5'] == 0) & (df['age_ge45'] == 0)),
                ('ge45', df['age_ge45'] == 1))
        },
        'asymptomatic_subcohort': {
            'n': int(asym.sum()),
            'sx_3m_events': int(y[asym].sum()),
            'lam_auroc': (float(compute_auc(lam[asym], y[asym]))
                          if y[asym].sum() >= 2 else None),
            'note': '部署目标层（基线无症状）；事件极稀，仅方向证据',
        },
    }
    return out


def _make_model(model_key, seed):
    """按冻结注册表超参构造主模型（与 combined_ablation 同口径）。"""
    from ..scoring.ml.training import MODEL_REGISTRY
    spec = MODEL_REGISTRY[model_key]
    params = dict(spec.default_params)
    params[spec.random_state_param] = int(seed)
    return spec.model_cls(**params)


def _group_cv_indices(groups, y, n_splits=5, seed=0):
    """户分层分组 CV：折间户不相交、每折标签比例近似。

    Returns:
        list[(train_idx, test_idx)]
    """
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
    folds = []
    for tr, te in cv.split(np.zeros(len(y)), y, groups=groups):
        folds.append((np.asarray(tr), np.asarray(te)))
    return folds


def _pr_auc(y, scores):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, scores))


def _recall_at_budget(scores, y, budget=0.25):
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    n_top = max(1, int(np.ceil(budget * len(y))))
    top = np.argsort(-scores)[:n_top]
    return float(y[top].sum() / max(y.sum(), 1))


def run_real_data_ablation_once(seed=0, n_splits=5, model_keys=MODEL_KEYS,
                                df=None):
    """单种子真实数据阶梯消融：户分组 CV → 池化 OOF 预测。

    Returns:
        dict: arms（pi_only + 各特征臂×模型 + 各臂对 m3_tb_out 的迁移
        AUROC）、y、groups、oof 分数（供配对检验与 cluster bootstrap）。
    """
    if df is None:
        df = load_pacts_contacts()
    y = df['sx_3m'].astype(int).to_numpy()
    y_tb = df['m3_tb_out'].fillna(0).astype(int).to_numpy()
    groups = df['id'].to_numpy()
    folds = _group_cv_indices(groups, y, n_splits=n_splits, seed=seed)

    arms = {}
    lam = df['lam'].to_numpy(dtype=float)
    arms['pi_only'] = {'oof': lam}

    for arm in FEATURE_ARMS:
        cols = feature_columns(arm)
        X = df[cols].to_numpy(dtype=float)
        for mk in model_keys:
            oof = np.zeros(len(y))
            for tr, te in folds:
                model = _make_model(mk, seed)
                model.fit(X[tr], y[tr])
                oof[te] = model.predict_proba(X[te])[:, 1]
            arms[f'{arm}:{mk}'] = {'oof': oof}

    result = {'seed': seed, 'y': y, 'y_tb': y_tb, 'groups': groups,
              'arms': {}}
    for name, a in arms.items():
        oof = a['oof']
        entry = {
            'auroc': float(compute_auc(oof, y)),
            'pr_auc': _pr_auc(y, oof),
            'recall_at_budget': _recall_at_budget(oof, y),
            'auroc_m3_tb_out': float(compute_auc(oof, y_tb)),
        }
        result['arms'][name] = entry
    result['oof'] = {name: a['oof'] for name, a in arms.items()}
    result['design'] = {
        'n': int(len(y)), 'n_events': int(y.sum()),
        'n_households': int(pd.Series(groups).nunique()),
        'n_splits': n_splits,
        'endpoint': 'sx_3m（3 月随访症状筛查阳性）',
        'secondary': 'm3_tb_out（9 事件，迁移评估）',
    }
    return result


def _cluster_bootstrap_delta(oof_a, oof_b, y, groups, n_bootstrap=2000,
                             seed=0):
    """户级 cluster bootstrap：重采样户 → ΔAUROC 分布 → CI 与 P(Δ>0)。"""
    rng = np.random.RandomState(seed)
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=int)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    deltas = []
    aurocs_a, aurocs_b = [], []
    for _ in range(n_bootstrap):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in pick])
        ya, sa, sb = y[rows], oof_a[rows], oof_b[rows]
        if ya.sum() == 0 or ya.sum() == len(ya):
            continue
        ua = float(compute_auc(sa, ya))
        ub = float(compute_auc(sb, ya))
        aurocs_a.append(ua)
        aurocs_b.append(ub)
        deltas.append(ua - ub)
    deltas = np.array(deltas)
    lo, hi = float(np.percentile(deltas, 2.5)), \
        float(np.percentile(deltas, 97.5))
    return {
        'mean': float(np.mean(deltas)),
        'bootstrap_ci': [lo, hi],
        'p_positive': float(np.mean(deltas > 0.0)),
        'ci_excludes_zero': bool(lo > 0.0 or hi < 0.0),
    }


def run_multi_seed_real_data(n_seeds=20, seed_start=0, n_splits=5,
                             n_bootstrap=2000, df=None):
    """多种子阶梯消融 + 户级 cluster bootstrap CI + DeLong 汇总。"""
    if df is None:
        df = load_pacts_contacts()

    per_seed = []
    pooled_oof = {}
    for s in range(seed_start, seed_start + n_seeds):
        rep = run_real_data_ablation_once(seed=s, n_splits=n_splits, df=df)
        per_seed.append({
            'seed': rep['seed'],
            'arms': rep['arms'],
            'delong': {
                f'{a}_vs_{b}': delong_paired_test(
                    rep['y'], rep['oof'][a], rep['oof'][b])
                for a, b in (
                    ('index:random_forest', 'ind:random_forest'),
                    ('exposure:random_forest', 'index:random_forest'),
                    ('pi:random_forest', 'exposure:random_forest'),
                    ('pi:random_forest', 'ind:random_forest'),
                    ('pi:lightgbm', 'exposure:lightgbm'),
                    ('pi_only', 'ind:random_forest'),
                )
            },
        })
        for name, oof in rep['oof'].items():
            pooled_oof.setdefault(name, []).append(oof)

    design = rep['design']
    arm_names = list(per_seed[0]['arms'].keys())

    arm_summary = {}
    for name in arm_names:
        vals = {m: [r['arms'][name][m] for r in per_seed]
                for m in ('auroc', 'pr_auc', 'recall_at_budget',
                          'auroc_m3_tb_out')}
        arm_summary[name] = {
            'mean_auroc': float(np.mean(vals['auroc'])),
            'sd_auroc': float(np.std(vals['auroc'])),
            'mean_pr_auc': float(np.mean(vals['pr_auc'])),
            'mean_recall_at_budget': float(np.mean(vals['recall_at_budget'])),
            'mean_auroc_m3_tb_out': float(np.mean(vals['auroc_m3_tb_out'])),
        }

    # 末种子 OOF 上的户级 cluster bootstrap（效应量主口径）
    y = rep['y']
    groups = rep['groups']
    ladder_summary = {}
    contrasts = (
        ('index:random_forest', 'ind:random_forest'),
        ('exposure:random_forest', 'index:random_forest'),
        ('pi:random_forest', 'exposure:random_forest'),
        ('pi:random_forest', 'ind:random_forest'),
        ('pi:lightgbm', 'exposure:lightgbm'),
        ('pi_only', 'ind:random_forest'),
    )
    for a, b in contrasts:
        ladder_summary[f'{a}_minus_{b}'] = _cluster_bootstrap_delta(
            rep['oof'][a], rep['oof'][b], y, groups,
            n_bootstrap=n_bootstrap, seed=seed_start)

    delong_summary = {}
    for key in per_seed[0]['delong']:
        ps = [float(r['delong'][key]['p_value'])
              for r in per_seed]
        delong_summary[key] = {
            'mean_p': float(np.mean(ps)),
            'median_p': float(np.median(ps)),
            'frac_p_lt_0.05': float(np.mean(np.array(ps) < 0.05)),
        }

    pi_gain = ladder_summary['pi:random_forest_minus_exposure:random_forest']
    out = {
        'design': {
            'name': 'real_data_pi_validation_v1',
            'source': 'PLOS ONE 2022;17(5):e0269219 S1 Dataset（PACTS '
                      '马拉维家庭接触者试验，开放获取）',
            'n': design['n'], 'n_events': design['n_events'],
            'n_households': design['n_households'],
            'endpoint': design['endpoint'],
            'secondary': design['secondary'],
            'n_seeds': n_seeds, 'seed_start': seed_start,
            'cv': f'StratifiedGroupKFold({n_splits}) by household',
            'ci': f'户级 cluster bootstrap ×{n_bootstrap}（末种子 OOF）',
            'models': '冻结注册表 RF/LGBM（scoring/ml/training.py）',
            'lambda_priors': LAMBDA_PRIORS,
            'feature_arms': {a: feature_columns(a) for a in FEATURE_ARMS},
        },
        'physics_direction': physics_direction_report(df),
        'arm_summary': arm_summary,
        'ladder_summary': ladder_summary,
        'delong_summary': delong_summary,
        'seeds': per_seed,
        'conclusion': {
            'pi_feature_gain': {
                'mean': pi_gain['mean'],
                'ci': pi_gain['bootstrap_ci'],
                'significant': pi_gain['ci_excludes_zero'],
            },
            'physics_only_vs_learned_baseline': {
                k: ladder_summary['pi_only_minus_ind:random_forest'][k]
                for k in ('mean', 'bootstrap_ci', 'ci_excludes_zero')},
        },
    }
    return out


def save_result(out, path=None):
    """归档 JSON（含 design/arm/ladder/delong/conclusion）。"""
    if path is None:
        stamp = time.strftime('%Y%m%d')
        path = os.path.join(_REPO_ROOT, 'data', 'processed',
                            f'real_data_pi_validation_{stamp}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return path
