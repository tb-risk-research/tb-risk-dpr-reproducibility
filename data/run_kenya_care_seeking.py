#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-1（第九轮）：目标人群对齐——case-population 框架下量化
「已通报人群训练 vs 未就诊人群部署」的采样偏差。

问题：SINAN 0.83 的 AUROC 训练于已确诊通报人群——而部署目标是
社区未就诊人群中的未诊断 TB。未就诊者在通报训练集中完全不可见。
Kenya 2016 全国患病率调查（126,389 筛查，社区抽样）是唯一覆盖
「准未就诊人群」的真实数据源。

三个问题（全部观测性、无 PU 概念滑步）：
  A. 系统漏检率直接估计：调查确诊病例中「未就诊/未在治」比例
     ——被动就诊漏掉了多少（患病率调查的核心输出，与官方
     ~40% 估计对照）；
  B. 漏检病例特征谱：未就诊确诊 vs 已发现确诊在年龄/症状/HIV
     上的分布差异（采样偏差的方向）；
  C. 未就诊高风险池：全人群症状风险模型 top-decile 中未就诊
     者的规模（主动筛查该捞谁——部署视角的 yield 语言）。

归档：data/processed/kenya_case_population_20260826.json
"""
import json
import os
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw', 'kenya_prevalence')
PROC = os.path.join(HERE, 'processed')
OUT = os.path.join(PROC, 'kenya_case_population_20260826.json')

SYMPTOM_COLS = ['Coughing', 'Sputum', 'BloodCough', 'ChestPains', 'Fever',
                'Fatigue', 'WeightLoss', 'NightSweats', 'BreatheShortness']


def main():
    t0 = time.time()
    print('=== P0-1 目标人群对齐：case-population 采样偏差量化 ===')
    df = pd.read_csv(os.path.join(RAW, 'S01.csv'), low_memory=False)
    n_screened = len(df)
    df = df[df['Coughing'].notna()].copy()
    n_analyzed = len(df)

    def is_pos(series):
        return (series.astype(str).str.strip().str.upper() == 'POS')

    y = (is_pos(df['Smearpositive'].fillna(''))
         | is_pos(df['Xpertpositive'].fillna(''))
         | is_pos(df['Culturepositive'].fillna(''))).astype(int).values

    # 系统可见性：患病率调查的 gap 定义——调查发现的病例中
    # 「从未被系统治疗过」的比例。注意 ParticipantCareSeekingDone
    # 与 IsTakingTBDrugs 是调查流程字段（99.9% 常数 / 大量缺失），
    # 不能作为「自行就诊史」，已弃用（诚实记录）。
    treated_prior = (
        (pd.to_numeric(df['TBEverBeenTreated'], errors='coerce')
         .fillna(0) > 0)
        | (pd.to_numeric(df['AntiTBTreatmentBefore'], errors='coerce')
           .fillna(0) > 0)).values
    taking = pd.to_numeric(df['IsTakingTBDrugs'], errors='coerce').fillna(0)
    on_treat = taking.values > 0
    treated = treated_prior.astype(int)

    # 特征（同 process_kenya_data 口径 + 症状计数）
    age = pd.to_numeric(df['Age'], errors='coerce').fillna(0).values
    sym = pd.DataFrame({
        c: pd.to_numeric(df[c], errors='coerce').fillna(0)
        for c in SYMPTOM_COLS})
    n_sym = sym.gt(0).sum(axis=1).values
    has_sym = (n_sym > 0).astype(int)
    hiv = (pd.to_numeric(df['HIVStattus'], errors='coerce')
           .fillna(0) > 0).astype(int)

    # ---------- A. 系统漏检率直接估计 ----------
    pos = y == 1
    n_pos = int(pos.sum())
    miss = pos & ~treated_prior & ~on_treat   # 确诊且系统从未接触
    found = pos & (treated_prior | on_treat)  # 确诊且系统已知
    n_miss, n_found = int(miss.sum()), int(found.sum())
    miss_rate = n_miss / max(n_pos, 1)

    print('A. 确诊 %d 例：系统从未接触（从未治疗且未在治）%d（%.1f%%）'
          % (n_pos, n_miss, 100 * miss_rate))

    # ---------- B. 漏检 vs 已发现病例特征谱 ----------
    def profile(mask):
        return {'n': int(mask.sum()),
                'age_median': float(np.median(age[mask])),
                'n_symptoms_median': float(np.median(n_sym[mask])),
                'symptomatic_pct': round(100 * float(has_sym[mask].mean()), 1),
                'hiv_pct': round(100 * float(hiv[mask].mean()), 1),
                'past_tb_pct': round(100 * float(treated[mask].mean()), 1)}

    prof_miss, prof_found = profile(miss), profile(found)
    print('B. 漏检病例：年龄中位 %.0f / 症状中位 %.0f / 有症状 %.1f%% / HIV %.1f%%'
          % (prof_miss['age_median'], prof_miss['n_symptoms_median'],
             prof_miss['symptomatic_pct'], prof_miss['hiv_pct']))
    print('   已发现：年龄中位 %.0f / 症状中位 %.0f / 有症状 %.1f%% / HIV %.1f%%'
          % (prof_found['age_median'], prof_found['n_symptoms_median'],
             prof_found['symptomatic_pct'], prof_found['hiv_pct']))

    # 漏检概率模型（确诊者内：未发现 vs 已发现——小样本，LR 稳妥）
    sel_model = {'available': False}
    if n_found >= 10 and n_miss >= 10:
        Xs = np.column_stack([age[pos], n_sym[pos], hiv[pos]])
        ys = miss[pos].astype(int)          # 1 = 漏检
        n_folds = min(3, int(ys.sum()), int((1 - ys).sum()))
        if n_folds >= 2:
            oof = np.zeros(len(ys))
            for tri, tei in StratifiedKFold(n_folds, shuffle=True,
                                            random_state=42).split(Xs, ys):
                m = LogisticRegression(max_iter=2000).fit(Xs[tri], ys[tri])
                oof[tei] = m.predict_proba(Xs[tei])[:, 1]
            auc = float(roc_auc_score(ys, oof))
            lr = LogisticRegression(max_iter=2000).fit(Xs, ys)
            sel_model = {
                'available': True,
                'auroc_miss_vs_found': auc,
                'coefs': dict(zip(['age', 'n_symptoms', 'hiv'],
                                  [round(float(c), 4) for c in lr.coef_[0]])),
                'interpretation': '漏检可由年龄/症状/HIV 部分预测'
                                  '（选择偏差非随机）'
                if auc > 0.6 else '漏检与年龄/症状/HIV 近似独立'
                                  '（选择偏差近似随机，特征外推风险低）'}
            print('   漏检 vs 已发现 AUROC %.3f | 系数 %s'
                  % (auc, sel_model['coefs']))

    # ---------- C. 未就诊高风险池（部署 yield 语言） ----------
    X = np.column_stack([age, n_sym, hiv, treated]).astype(float)
    oof = np.zeros(n_analyzed)
    if HAS_LGBM and n_pos >= 30:
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for tri, tei in skf.split(X, y):
            m = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                               num_leaves=15, random_state=42, verbose=-1)
            m.fit(X[tri], y[tri])
            oof[tei] = m.predict_proba(X[tei])[:, 1]
        auroc = float(roc_auc_score(y, oof))
        order = np.argsort(-oof)
        dec = int(round(n_analyzed * 0.1))
        top = order[:dec]
        top_pos = int(y[top].sum())
        top_missed = int(miss[top].sum())
        pool = {
            'auroc_oof': auroc,
            'yield_top10pct': round(top_pos / max(n_pos, 1), 4),
            'top10_n': dec,
            'top10_positives': top_pos,
            'top10_never_treated_pos': top_missed,
            'note': 'top-decile 风险分层中的「系统从未接触」确诊病例'
                    '——主动筛查（ACF）可及、被动就诊永远不可及的'
                    '增益池',
        }
        print('C. 全人群 OOF AUROC %.4f | top-10%% 捕获 %.1f%% 确诊，'
              '其中系统从未接触 %d 人'
              % (auroc, 100 * pool['yield_top10pct'], top_missed))
    else:
        pool = {'available': False, 'reason': 'n_pos<30 或无 LGBM'}

    res = {
        'date': '2026-08-26',
        'experiment': 'kenya_case_population_v1',
        'question': '已通报人群训练 vs 未就诊人群部署——采样偏差多大、'
                    '方向如何、未就诊高风险池有多少',
        'data': {'n_screened': n_screened, 'n_analyzed': n_analyzed,
                 'n_positive': n_pos,
                 'prevalence': round(n_pos / n_analyzed, 6)},
        'A_missed_by_system': {
            'n_missed': n_miss, 'n_found': n_found, 'n_total_pos': n_pos,
            'miss_rate': round(miss_rate, 4),
            'definition': 'missed = 确诊 且 TBEverBeenTreated=0 且'
                          ' AntiTBTreatmentBefore=0 且未在治'
                          '（系统从未接触；ParticipantCareSeekingDone 为'
                          '调查流程字段 99.9% 常数，已弃用）',
            'official_reference': 'Kenya WHO 2016 估计系统检出缺口 ~40%'
                                  '（本直接估计含亚临床培养阳性，'
                                  '口径更宽故更高）',
        },
        'B_missed_profile': {'missed': prof_miss, 'found': prof_found,
                             'selection_model': sel_model},
        'C_high_risk_pool': pool,
        'implication': (
            'SINAN（已通报人群）训练的死亡模型不可直接外推到未就诊'
            '人群；部署「找患者」应使用覆盖未就诊者的患病率调查特征'
            '（症状/年龄/HIV）——即本实验 C 部分的模型形态；'
            'ERASE-TB 前瞻队列将提供进展终点的未就诊覆盖。'),
        'has_lgbm': HAS_LGBM,
        'runtime_s': round(time.time() - t0, 1),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
