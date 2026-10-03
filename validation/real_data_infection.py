#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据感染终点 PI 验证（P4b-2，2026-08-25）：南非 HomeACF 队列。

数据：petermacp/tstsa（github，HomeACF 研究分析数据）
homeacf_tstsa.rda：2985 名家庭接触者 / 924 户（Mangaung + Capricorn
两站点）；每行已合并指示病例特征（_i 后缀）。

科学问题（对齐 real_data_pi.py 的 PACTS 验证）：
PACTS 的症状终点被基线临床状态饱和、疾病终点 9 事件不可判——
感染终点（TST 阳性）才是够功效的 PI 物理验证路径。本数据集：
  终点 tst_pos10 = TST 硬结直径 ≥10 mm（359/2725 ≈ 13.2% 事件，
  为 PACTS 主终点事件的 3.4 倍）；敏感性口径 ≥5 mm（458 事件）。

相对 PACTS 的两处机制升级：
  1. λ 首次获得时间维——指示病例咳嗽天数 coughdays_i（中位 30 天，
     范围 0-365）：I_index 含 exp(coughdays) 累积暴露项（PACTS 的
     指示病例症状时长缺失，λ 只能做强度型）；
  2. 宿主 HIV 可用（hivfinal_h）——S(host) 增加免疫抑制先验；
  3. 暴露分级可用（timespent_h 三级 / sharebedroom / sleepsamebed）
     ——C(contact) 从二值同住升级为分级接触强度。

λ 构造（文献先验，冻结系数，禁止用终点拟合）：
  λ_i = S(age_i, hiv_i) · I(index) · C_i
  S(host)   1 + 1.8·(<5 岁) + 1.6·(≥45 岁)，HIV+ ×1.3
            （与 PACTS 验证同源 U 形先验；HIV 修正取 Andrews 接触
            者追踪荟萃的进展/感染风险比量级）
  I(index)  exp(0.01·coughdays) × 3.0^(涂片+) × 1.3^(指示病例 HIV+)
            涂片 83% 未做（Xpert 时代队列）——涂片项仅在已知子集
            生效，咳嗽时长承载主要时间维；咳嗽坡度 0.01/天 =
            30 天 ×1.35、90 天 ×2.46、365 天 ×3.78（Verver 传染性
            持续时间研究的量级）
  C(ctc)    timespent 三级 0.5/1.0/1.5（偶尔/部分/大部分时间同处）
            × 1.2^(共用卧室) × 1.5^(同床)

诚实边界：
  - TST 阳性 = LTBI（感染），非 TB 发病；BCG 接种与环境中分枝
    杆菌可致假阳性（<10 岁 BCG 效应更强）——方向性验证口径；
  - TST 读数缺失 8.7%（2725/2985）：按 tstdiam 非缺失的完全
    病例分析，未做缺失机制检验（MAR 假设未验证）；
  - 特征层缺失的中性化：咳嗽天数缺失（9 行）填中位 30 天、
    BMI 缺失（16 行）填中位、共处时长缺失（3 行）归中等级
    ——物理口径下给中性值而非剔除；
  - smear/culture 多数未做、Xpert+ 占 93%（入组标准即确诊 TB）
    ——指示病例感染力的细菌学维度近乎无变异，λ 的 I 项主要由
    咳嗽天数与 HIV 驱动；
  - 单队列（南非两站点）、非随机抽样：结论口径 = 感染终点上的
    方向与效应量外部验证，非多数据集荟萃；
  - λ 系数全部冻结：pi 臂检验的是"文献先验构造的物理排序"是否
    携带学习模型未消化的信息，而非拟合后的上限。

臂结构（嵌套，主模型 = 冻结注册表 RF/LGBM，与 PACTS 验证同口径）：
  ind       接触者协变量：年龄/年龄组、性别、HIV、BMI、吸烟、
            糖尿病、就业、站点、户内接触者数
  index     + 指示病例：咳嗽天数、涂片状态、Xpert、HIV、年龄、
            性别、死亡
  exposure  + 暴露分级：共处时长三级、共用卧室、同床、关系、
            同 airspace
  pi        + λ（1 列）
  pi_only   纯 λ 排序（不学习）

协议：户（record_id）分层分组 5 折 CV × 多种子（同折配对）；
户级 cluster bootstrap CI；逐种子 DeLong 配对。
"""

import json
import os
import time

import numpy as np
import pandas as pd

from .layer_ablation import delong_paired_test
from .real_data_pi import (
    MODEL_KEYS,
    _cluster_bootstrap_delta,
    _group_cv_indices,
    _make_model,
    _pr_auc,
    _recall_at_budget,
)
from .threshold_spec import compute_auc

FEATURE_ARMS = ('ind', 'index', 'exposure', 'pi')

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RDA_PATH = os.path.join(_REPO_ROOT, 'data', 'raw', 'homeacf_tstsa.rda')

# λ 文献先验（冻结，禁止用终点拟合——物理特征口径）
LAMBDA_PRIORS_INF = {
    'susceptibility_lt5': 1.8,      # <5 岁宿主易感性加项（同 PACTS）
    'susceptibility_ge45': 1.6,     # ≥45 岁宿主易感性加项（同 PACTS）
    'host_hiv_pos': 1.3,            # 接触者 HIV+ 易感性倍数
    'index_cough_slope': 0.01,      # 咳嗽天数指数坡度/天（时间维）
    'index_smear_pos': 3.0,         # 涂片+ 感染力倍数（同 PACTS）
    'index_hiv_pos': 1.3,           # 指示病例 HIV+ 感染力倍数（同 PACTS）
    'timespent_low': 0.5,           # 偶尔同处接触强度
    'timespent_mid': 1.0,           # 部分时间同处
    'timespent_high': 1.5,          # 大部分时间同处
    'share_bedroom': 1.2,           # 共用卧室倍数
    'sleep_same_bed': 1.5,          # 同床倍数
}

# 咳嗽天数缺失（9/2985）填中位 30 天——中性物理值，docstring 已声明
_COUGH_MEDIAN_FILL = 30.0


def load_homeacf_contacts(rda_path=None):
    """装载 HomeACF 接触者队列（rdata 解析 → 特征派生 → 终点定义）。

    Returns:
        pd.DataFrame: 一行一接触者（tstdiam 非缺失的完全病例）；
        'record_id' 列 = 户（指示病例）簇键。
    """
    rda_path = rda_path or _RDA_PATH
    import rdata
    with warnings_catch():
        conv = rdata.conversion.convert(rdata.parser.parse_file(rda_path))
    df = conv['tstsa'].copy()
    df.columns = [str(c) for c in df.columns]
    return _derive_features(df)


class warnings_catch:
    """抑制 rdata 对 tbl_df/haven_labelled 构造器缺失的回退警告。"""

    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter('ignore', UserWarning)
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def _derive_features(df):
    """派生建模列（基线可观测）+ 终点。"""
    out = df.copy()

    # ---- 终点（感染：TST 硬结直径）----
    diam = pd.to_numeric(out['tstdiam_h'], errors='coerce')
    out = out[diam.notna()].copy()
    diam = pd.to_numeric(out['tstdiam_h'], errors='coerce')
    out['tst_pos10'] = (diam >= 10).astype(int)
    out['tst_pos5'] = (diam >= 5).astype(int)
    out['tst_diam'] = diam.astype(float)

    # ---- 个体协变量（ind 臂）----
    out['contact_age'] = pd.to_numeric(out['ageyears_h'], errors='coerce')
    out['age_lt5'] = (out['contact_age'] < 5).astype(float)
    out['age_ge45'] = (out['contact_age'] >= 45).astype(float)
    out['contact_sex_m'] = (out['sex_h'] == 'Male').astype(float)
    out['hiv_pos_h'] = (out['hivfinal_h'] == 'b) HIV positive').astype(float)
    out['hiv_unknown_h'] = (
        out['hivfinal_h'] == 'c) HIV unknown').astype(float)
    out['bmi_h'] = pd.to_numeric(out['bmi_h'], errors='coerce')
    # BMI 缺失（16/2725）填中位——中性值，docstring 已声明
    out['bmi_h'] = out['bmi_h'].fillna(out['bmi_h'].median())
    out['smoke_ever_h'] = out['smoke_h'].astype(str).str.contains(
        'smoked', regex=False).astype(float)
    out['diabetes_h_f'] = (out['diabetes_h'] == 'Yes').astype(float)
    out['site_capricorn'] = (out['site'] == 'Capricorn').astype(float)
    out['hh_n_contacts'] = pd.to_numeric(
        out['num_contacts'], errors='coerce').astype(float)

    # ---- 指示病例特征（index 臂）----
    out['idx_coughdays'] = pd.to_numeric(
        out['coughdays_i'], errors='coerce').fillna(_COUGH_MEDIAN_FILL)
    out['idx_smear_pos'] = (
        out['smear_i'] == 'b) Smear positive').astype(float)
    out['idx_smear_known'] = out['smear_i'].isin(
        ['a) Smear negative', 'b) Smear positive']).astype(float)
    out['idx_xpert_pos'] = (
        out['xpert_i'] == 'b) Xpert positive').astype(float)
    out['idx_hiv_pos'] = (out['hiv_i'] == 'b) HIV-positive').astype(float)
    out['idx_age'] = pd.to_numeric(out['ageyears_i'], errors='coerce')
    out['idx_sex_m'] = (out['sex_i'] == 'Male').astype(float)
    out['idx_dead'] = (out['dead_i'] == 'b) Deceased').astype(float)

    # ---- 暴露分级（exposure 臂）----
    ts = out['timespent_h'].astype(str)
    out['ts_low'] = ts.str.startswith('a)').astype(float)
    out['ts_high'] = ts.str.startswith('c)').astype(float)
    # 缺失（3/2985）与"部分时间"同归中等级——λ 取中性强度
    out['ts_mid'] = 1.0 - out['ts_low'] - out['ts_high']
    out['share_bedroom'] = (out['sharebedroom_h'] == 'Yes').astype(float)
    out['sleep_same_bed'] = (out['sleepsamebed_h'] == 'Yes').astype(float)
    out['airspace_shared'] = (out['airspace_h'] == 'Yes').astype(float)
    rel = out['relationship_h'].astype(str)
    for key, pat in (('rel_child', 'Child'), ('rel_spouse', 'Spouse'),
                     ('rel_sibling', 'Brother/sister'),
                     ('rel_parent', 'Parent')):
        out[key] = rel.eq(pat).astype(float)

    # ---- 物理感染力 λ（pi 臂，冻结文献先验）----
    out['lam'] = build_lambda_infection(out)

    return out.reset_index(drop=True)


def build_lambda_infection(df):
    """乘性感染力：S(age, hiv) · I(coughdays, smear, hiv) · C(分级接触)。

    系数全部冻结（LAMBDA_PRIORS_INF），禁止与终点拟合。
    """
    p = LAMBDA_PRIORS_INF
    s_host = (1.0 + p['susceptibility_lt5'] * df['age_lt5']
              + p['susceptibility_ge45'] * df['age_ge45']) \
        * (p['host_hiv_pos'] ** df['hiv_pos_h'])
    i_index = np.exp(p['index_cough_slope'] * df['idx_coughdays'].clip(
        0, 365)) \
        * (p['index_smear_pos'] ** df['idx_smear_pos']) \
        * (p['index_hiv_pos'] ** df['idx_hiv_pos'])
    c_expo = (p['timespent_low'] * df['ts_low']
              + p['timespent_mid'] * df['ts_mid']
              + p['timespent_high'] * df['ts_high']) \
        * (p['share_bedroom'] ** df['share_bedroom']) \
        * (p['sleep_same_bed'] ** df['sleep_same_bed'])
    return (s_host * i_index * c_expo).astype(float)


FEATURE_SETS = {
    'ind': [
        'contact_age', 'age_lt5', 'age_ge45', 'contact_sex_m',
        'hiv_pos_h', 'hiv_unknown_h', 'bmi_h', 'smoke_ever_h',
        'diabetes_h_f', 'site_capricorn', 'hh_n_contacts',
    ],
    'index': [
        'idx_coughdays', 'idx_smear_pos', 'idx_smear_known',
        'idx_xpert_pos', 'idx_hiv_pos', 'idx_age', 'idx_sex_m',
        'idx_dead',
    ],
    'exposure': [
        'ts_low', 'ts_mid', 'ts_high', 'share_bedroom', 'sleep_same_bed',
        'airspace_shared', 'rel_child', 'rel_spouse', 'rel_sibling',
        'rel_parent',
    ],
    'pi': ['lam'],
}


def feature_columns(arm):
    """臂的完整特征列（嵌套：ind ⊂ index ⊂ exposure ⊂ pi）。"""
    cols = []
    for a in FEATURE_ARMS[:FEATURE_ARMS.index(arm) + 1]:
        cols.extend(FEATURE_SETS[a])
    return cols


def physics_direction_report_inf(df):
    """物理方向性检查（无模型，纯描述性证据）。

    Returns:
        dict: λ 四分位 TST 阳性率、咳嗽时长梯度、涂片梯度（已知
        子集）、年龄梯度、宿主 HIV 梯度。
    """
    lam = df['lam'].to_numpy(dtype=float)
    y = df['tst_pos10'].astype(int).to_numpy()

    def rate(mask):
        m = np.asarray(mask, dtype=bool)
        return {'n': int(m.sum()),
                'tst_pos10_rate': float(y[m].mean()) if m.any() else None}

    q = np.quantile(lam, [0.25, 0.5, 0.75])
    stratum = np.digitize(lam, q)
    lam_strata = {f'q{i+1}': rate(stratum == i) for i in range(4)}

    cd = df['idx_coughdays'].to_numpy(dtype=float)
    return {
        'lambda_quartile_tst_pos10': lam_strata,
        'coughdays_gradient': {
            'lt14': rate(cd < 14),
            '14_60': rate((cd >= 14) & (cd <= 60)),
            'gt60': rate(cd > 60),
        },
        'smear_gradient_known': {
            'smear_pos': rate((df['idx_smear_pos'] == 1).to_numpy()),
            'smear_neg': rate(((df['idx_smear_known'] == 1)
                               & (df['idx_smear_pos'] == 0)).to_numpy()),
            'note': '涂片已知仅 ~17% 子集；83% not done 不进梯度',
        },
        'age_gradient': {
            'lt5': rate((df['age_lt5'] == 1).to_numpy()),
            'mid': rate(((df['age_lt5'] == 0)
                         & (df['age_ge45'] == 0)).to_numpy()),
            'ge45': rate((df['age_ge45'] == 1).to_numpy()),
        },
        'host_hiv_gradient': {
            'hiv_pos': rate((df['hiv_pos_h'] == 1).to_numpy()),
            'hiv_neg': rate(((df['hiv_pos_h'] == 0)
                             & (df['hiv_unknown_h'] == 0)).to_numpy()),
        },
        'lam_auroc_overall': float(compute_auc(lam, y)),
    }


def run_infection_ablation_once(seed=0, n_splits=5, model_keys=MODEL_KEYS,
                                df=None):
    """单种子感染终点阶梯消融：户分组 CV → 池化 OOF。"""
    if df is None:
        df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    y5 = df['tst_pos5'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    folds = _group_cv_indices(groups, y, n_splits=n_splits, seed=seed)

    arms = {}
    lam = df['lam'].to_numpy(dtype=float)
    arms['pi_only'] = {'oof': lam}

    for arm in FEATURE_ARMS:
        cols = feature_columns(arm)
        X = df[cols].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0)  # RF/LGBM 注册表无 NaN 处理
        for mk in model_keys:
            oof = np.zeros(len(y))
            for tr, te in folds:
                model = _make_model(mk, seed)
                model.fit(X[tr], y[tr])
                oof[te] = model.predict_proba(X[te])[:, 1]
            arms[f'{arm}:{mk}'] = {'oof': oof}

    result = {'seed': seed, 'y': y, 'y5': y5, 'groups': groups,
              'arms': {}}
    for name, a in arms.items():
        oof = a['oof']
        result['arms'][name] = {
            'auroc': float(compute_auc(oof, y)),
            'pr_auc': _pr_auc(y, oof),
            'recall_at_budget': _recall_at_budget(oof, y),
            'auroc_tst_pos5': float(compute_auc(oof, y5)),
        }
    result['oof'] = {name: a['oof'] for name, a in arms.items()}
    result['design'] = {
        'n': int(len(y)), 'n_events': int(y.sum()),
        'n_events_pos5': int(y5.sum()),
        'n_households': int(pd.Series(groups).nunique()),
        'n_splits': n_splits,
        'endpoint': 'tst_pos10 = TST 硬结直径 ≥10 mm（LTBI 感染终点）',
        'secondary': 'tst_pos5（≥5 mm 敏感性口径）',
    }
    return result


def run_multi_seed_infection(n_seeds=20, seed_start=0, n_splits=5,
                             n_bootstrap=2000, df=None):
    """多种子阶梯消融 + 户级 cluster bootstrap CI + DeLong 汇总。"""
    if df is None:
        df = load_homeacf_contacts()

    per_seed = []
    for s in range(seed_start, seed_start + n_seeds):
        rep = run_infection_ablation_once(seed=s, n_splits=n_splits,
                                          df=df)
        per_seed.append({
            'seed': rep['seed'],
            'arms': rep['arms'],
            'delong': {
                f'{a}_vs_{b}': delong_paired_test(
                    rep['y'], rep['oof'][a], rep['oof'][b])
                for a, b in _CONTRASTS
            },
        })

    design = rep['design']
    arm_names = list(per_seed[0]['arms'].keys())

    arm_summary = {}
    for name in arm_names:
        vals = {m: [r['arms'][name][m] for r in per_seed]
                for m in ('auroc', 'pr_auc', 'recall_at_budget',
                          'auroc_tst_pos5')}
        arm_summary[name] = {
            'mean_auroc': float(np.mean(vals['auroc'])),
            'sd_auroc': float(np.std(vals['auroc'])),
            'mean_pr_auc': float(np.mean(vals['pr_auc'])),
            'mean_recall_at_budget': float(np.mean(vals['recall_at_budget'])),
            'mean_auroc_tst_pos5': float(np.mean(vals['auroc_tst_pos5'])),
        }

    # 末种子 OOF 上的户级 cluster bootstrap（效应量主口径）
    y = rep['y']
    groups = rep['groups']
    ladder_summary = {
        f'{a}_minus_{b}': _cluster_bootstrap_delta(
            rep['oof'][a], rep['oof'][b], y, groups,
            n_bootstrap=n_bootstrap, seed=seed_start)
        for a, b in _CONTRASTS
    }

    delong_summary = {}
    for key in per_seed[0]['delong']:
        ps = [float(r['delong'][key]['p_value']) for r in per_seed]
        delong_summary[key] = {
            'mean_p': float(np.mean(ps)),
            'median_p': float(np.median(ps)),
            'frac_p_lt_0.05': float(np.mean(np.array(ps) < 0.05)),
        }

    pi_gain = ladder_summary['pi:random_forest_minus_exposure:random_forest']
    return {
        'design': {
            'name': 'real_data_infection_validation_v1',
            'source': 'HomeACF（github petermacp/tstsa，南非 Mangaung + '
                      'Capricorn 家庭接触者队列）',
            'n': design['n'], 'n_events': design['n_events'],
            'n_events_pos5': design['n_events_pos5'],
            'n_households': design['n_households'],
            'endpoint': design['endpoint'],
            'secondary': design['secondary'],
            'n_seeds': n_seeds, 'seed_start': seed_start,
            'cv': f'StratifiedGroupKFold({n_splits}) by household',
            'ci': f'户级 cluster bootstrap ×{n_bootstrap}（末种子 OOF）',
            'models': '冻结注册表 RF/LGBM（scoring/ml/training.py）',
            'lambda_priors': LAMBDA_PRIORS_INF,
            'feature_arms': {a: feature_columns(a) for a in FEATURE_ARMS},
        },
        'physics_direction': physics_direction_report_inf(df),
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
            'pi_vs_ind_total': {
                k: ladder_summary[
                    'pi:random_forest_minus_ind:random_forest'][k]
                for k in ('mean', 'bootstrap_ci', 'ci_excludes_zero')},
            'physics_only_vs_learned_baseline': {
                k: ladder_summary[
                    'pi_only_minus_ind:random_forest'][k]
                for k in ('mean', 'bootstrap_ci', 'ci_excludes_zero')},
        },
    }


_CONTRASTS = (
    ('index:random_forest', 'ind:random_forest'),
    ('exposure:random_forest', 'index:random_forest'),
    ('pi:random_forest', 'exposure:random_forest'),
    ('pi:random_forest', 'ind:random_forest'),
    ('pi:lightgbm', 'exposure:lightgbm'),
    ('pi_only', 'ind:random_forest'),
)


def save_result(out, path=None):
    """归档 JSON。"""
    if path is None:
        stamp = time.strftime('%Y%m%d')
        path = os.path.join(_REPO_ROOT, 'data', 'processed',
                            f'real_data_infection_validation_{stamp}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return path
