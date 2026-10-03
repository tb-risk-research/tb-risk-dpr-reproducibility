#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-6（第九轮）：跨人群迁移衰减曲线——巴西训练 → 各国队列测试。

问题：方法学论文缺外部验证维度——「换部署地区掉多少」。

设计（轻量诚实版）：
  共同特征空间 = age / sex_male / hiv（四数据集可构造的最大交集；
  SINAN 处理管道未保留症状列 → 症状不进交集，边界如实归档）
  源模型   ：SINAN 三特征 LGBM（train FILE_YEAR≤12，全因死亡终点）
  参照     ：巴西 far test（18-19）AUROC = 内部时间外推
  迁移臂   ：同一模型直接给各国队列打分（零样本迁移，不重训）
  本地臂   ：各国三特征 LGBM 5 折 CV（同终点内的人群对照）
  衰减     ：本地 AUROC − 迁移 AUROC（同队列、同终点、同特征——
             纯人群迁移成本）；巴西参照 vs 各国 = 人群 + 终点联合差异

队列与终点（终点异质，caveat 必读）：
  Kenya 患病率调查：tb_outcome = 涂片/Xpert/培养任一阳性（n_pos=336）
  Malawi PACTS    ：sx_3m（3 月随访症状；接触者队列）
  Vietnam PROVE_TB：keybl_MTBcat.f = PTB-Positive MTB（Xpert 确诊）
  Mongolia MDR    ：排除——接触者检测结果表仅 70 行（sputum+ 5、
                    TST+ 13），事件量不足以稳定估计 AUROC
  TB Portals      ：排除——数据未入 data/raw

归档：data/processed/cross_population_transfer_20260826.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

PROC = os.path.join(BASE, 'tb_risk', 'data', 'processed')
RAW = os.path.join(BASE, 'tb_risk', 'data', 'raw')
OUT = os.path.join(PROC, 'cross_population_transfer_20260826.json')

COMMON_FEATS = ['age', 'sex_male', 'hiv']
TRAIN_MAX_YEAR, FAR_MIN_YEAR, FAR_MAX_YEAR = 12, 18, 19
SEED = 42


def _lgbm():
    return LGBMClassifier(n_estimators=400, learning_rate=0.06,
                          num_leaves=63, subsample=0.8,
                          colsample_bytree=0.8, n_jobs=-1,
                          random_state=SEED, verbose=-1)


# ---------------------------------------------------------------- SINAN 源
def load_sinan_common():
    """SINAN 共同特征 + 全因死亡终点（SITUA_ENCE∈{3,4}，人群 {1,3,4}）。"""
    d = pd.read_parquet(
        os.path.join(PROC, 'sinan', 'tubebra_2001_2019.parquet'),
        columns=['AGE_YEARS', 'CS_SEXO', 'AGRAVAIDS', 'SITUA_ENCE',
                 'FILE_YEAR'])
    d = d[d['SITUA_ENCE'].astype(str).isin(['1', '3', '4'])].copy()
    y = d['SITUA_ENCE'].astype(str).isin(['3', '4']).astype(int).values
    X = pd.DataFrame({
        'age': np.clip(pd.to_numeric(d['AGE_YEARS'], errors='coerce')
                       .fillna(d['AGE_YEARS'].median()).values, 0, 100),
        'sex_male': (pd.to_numeric(d['CS_SEXO'], errors='coerce') == 1)
                    .astype(float).values,
        'hiv': pd.to_numeric(d['AGRAVAIDS'], errors='coerce')
               .fillna(0).clip(0, 1).astype(float).values,
    })
    year = pd.to_numeric(d['FILE_YEAR'], errors='coerce').values
    return X, y, year


# ---------------------------------------------------------------- 队列
def load_kenya():
    """Kenya 患病率调查（共同特征 + 细菌学确诊终点）。"""
    df = pd.read_csv(os.path.join(RAW, 'kenya_prevalence', 'S01.csv'),
                     low_memory=False)
    df = df[df['Coughing'].notna()].copy()

    def is_pos(s):
        return (s.astype(str).str.strip().str.upper() == 'POS')

    y = (is_pos(df['Smearpositive'].fillna(''))
         | is_pos(df['Xpertpositive'].fillna(''))
         | is_pos(df['Culturepositive'].fillna(''))).astype(int).values
    X = pd.DataFrame({
        'age': pd.to_numeric(df['Age'], errors='coerce')
                .fillna(pd.to_numeric(df['Age'], errors='coerce').median())
                .values,
        'sex_male': (df['Sex'].astype(str).str.strip().str.upper()
                      == 'M').astype(float).values,
        'hiv': (pd.to_numeric(df['HIVStattus'], errors='coerce')
                .fillna(0) > 0).astype(float).values,
    })
    return X, y


def load_pacts():
    """马拉维 PACTS 接触者（共同特征 + 3 月症状终点；接触者无 HIV 检测）。"""
    from tb_risk.validation.real_data_pi import load_pacts_contacts
    df = load_pacts_contacts(data_dir=os.path.join(
        RAW, 'malawi_pacts_s002_x', 'S1_Publishing data set_2', '_Data'))
    y = pd.to_numeric(df['sx_3m'], errors='coerce').values
    ok = np.isfinite(y)
    X = pd.DataFrame({
        'age': pd.to_numeric(df['contact_age'], errors='coerce')
                .fillna(df['contact_age'].median()).values,
        'sex_male': pd.to_numeric(df['contact_sex_m'], errors='coerce')
                     .fillna(0).values,
        # 接触者自身 HIV 未检测（指示病例 HIV 属另一特征）——按阴性
        'hiv': np.zeros(len(df)),
    })
    return X[ok], y[ok].astype(int)


def load_vietnam():
    """越南 PROVE_TB（共同特征 + Xpert MTB 确诊终点；HIV 全阴性）。"""
    x = pd.ExcelFile(os.path.join(RAW, 'vietnam_lam', 'lam_paper_data.xlsx'))
    d = pd.read_excel(x, sheet_name=1)
    y = (d['keybl_MTBcat.f'].astype(str).str.strip()
         == 'PTB- Positive, MTB').astype(int).values
    X = pd.DataFrame({
        'age': pd.to_numeric(d['char_age'], errors='coerce')
                .fillna(d['char_age'].median()).values,
        'sex_male': (d['char_gender.factor'].astype(str).str.strip()
                     == 'Male').astype(float).values,
        'hiv': (d['keybl_hiv.f'].astype(str).str.strip()
                == 'Positive').astype(float).values,
    })
    return X, y


def local_cv_auroc(X, y):
    """队列本地三特征 LGBM 5 折 CV AUROC（同终点内的人群对照）。"""
    aucs = []
    for tri, tei in StratifiedKFold(5, shuffle=True,
                                    random_state=SEED).split(X, y):
        mdl = _lgbm()
        mdl.fit(X.iloc[tri], y[tri])
        aucs.append(roc_auc_score(y[tei],
                                   mdl.predict_proba(X.iloc[tei])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    t0 = time.time()
    print('=== P3-6 跨人群迁移衰减：巴西训练 → 各国队列测试 ===')
    if not HAS_LGBM:
        raise SystemExit('需要 lightgbm')

    # 1) 源模型：巴西 SINAN 共同特征（内部 far test 参照）
    Xs, ys, year = load_sinan_common()
    tr = year <= TRAIN_MAX_YEAR
    te = (year >= FAR_MIN_YEAR) & (year <= FAR_MAX_YEAR)
    src = _lgbm()
    src.fit(Xs[tr], ys[tr])
    ref_auc = roc_auc_score(ys[te], src.predict_proba(Xs[te])[:, 1])
    print('[BR far test 参照] n=%d rate %.3f | 三特征 AUROC %.4f'
          % (te.sum(), ys[te].mean(), ref_auc))

    # 2) 各国队列：迁移（零样本）+ 本地 CV
    cohorts = [('Kenya 患病率调查（确诊终点）', load_kenya),
               ('Malawi PACTS（3 月症状终点）', load_pacts),
               ('Vietnam PROVE_TB（Xpert 确诊终点）', load_vietnam)]
    rows = []
    for name, loader in cohorts:
        X, y = loader()
        if y.sum() < 20 or (1 - y).sum() < 20:
            print('[%s] 事件量不足，跳过' % name)
            continue
        transfer_auc = roc_auc_score(y, src.predict_proba(X)[:, 1])
        local_auc, local_std = local_cv_auroc(X, y)
        rows.append({
            'cohort': name,
            'n': int(len(y)), 'n_pos': int(y.sum()),
            'event_rate': float(y.mean()),
            'transfer_auroc': float(transfer_auc),
            'local_cv_auroc': local_auc,
            'local_cv_std': local_std,
            'population_decay': float(local_auc - transfer_auc),
        })
        print('[%s] n=%d pos=%d rate %.3f | 迁移 %.4f | 本地 %.4f±%.3f'
              ' | 人群衰减 %.4f'
              % (name, len(y), y.sum(), y.mean(), transfer_auc,
                 local_auc, local_std, local_auc - transfer_auc))

    res = {
        'date': '2026-08-26',
        'experiment': 'cross_population_transfer_v1',
        'question': '换部署地区掉多少——巴西训练的模型在各国队列的迁移衰减',
        'design': {
            'source': 'SINAN 三特征 LGBM（train FILE_YEAR≤12，全因死亡）',
            'common_features': COMMON_FEATS,
            'feature_boundary': 'SINAN 处理管道未保留症状列 → 症状不进交集；'
                                'PACTS 接触者无 HIV 检测（按阴性）；'
                                'Vietnam HIV 全阴性（特征无变异）',
            'transfer': '零样本迁移（源模型直接打分，不重训）',
            'local': '各国三特征 LGBM 5 折 CV（同终点人群对照）',
            'population_decay': 'local_cv_auroc - transfer_auroc'
                                '（同队列同终点同特征，纯人群迁移成本）',
            'seed': SEED,
        },
        'br_reference': {
            'cohort': 'SINAN far test 2018-19（全因死亡）',
            'n': int(te.sum()), 'event_rate': float(ys[te].mean()),
            'auroc': float(ref_auc),
            'note': '内部时间外推参照——与各国迁移 AUROC 的差'
                    '混合了人群与终点两层差异',
        },
        'cohorts': rows,
        'excluded': [
            {'cohort': 'Mongolia MDR 接触者',
             'reason': '检测结果表仅 70 行（sputum 阳性 5、TST 阳性 13），'
                       '事件量不足以稳定估计 AUROC'},
            {'cohort': 'TB Portals',
             'reason': '数据未入 data/raw（未下载）'},
        ],
        'caveats': [
            '终点异质：巴西=死亡（9.7%）、Kenya=患病率确诊（0.5%）、'
            'PACTS=3 月症状、Vietnam=Xpert 确诊（43.5%）——'
            'br_reference 与各国迁移 AUROC 的差不能解读为纯人群衰减',
            '共同特征仅 age/sex_male/hiv 三维——判别力远低于全特征模型'
            '（BR 三特征 AUROC %.3f vs 159 维 0.8135），本实验测的是'
            '特征交集上的相对迁移成本，不是部署级绝对水平' % ref_auc,
            '迁移方向依赖终点语义重叠：死亡终点的 age/hiv 系数'
            '（age↑risk）与确诊终点（症状主导人群）方向部分相反，'
            '负迁移（迁移 < 0.5）本身是跨终点部署风险的经验证据',
        ],
        'has_lgbm': HAS_LGBM,
        'runtime_s': round(time.time() - t0, 1),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
