#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1（第六轮）：SINAN 老年（65+）亚组专项——塌方修复实验，2026-08-26。

背景：规模训练实验（sinan_scale_training_20260826.json）亚组分析
显示 65+ AUROC 0.670 为全亚组最低（15-49 岁 0.855），老年是
残余误差质量。面向中国部署场景（老年 TB 负担高）有直接政策意义。

三臂设计（T1 时间外推协议，全程与规模实验一致）：
  (a) all_age   ：全年龄训练（2001-2012）→ 65+ far 测试（18-19）
                  ——基线复现，应≈0.67
  (b) elderly   ：65+ 单独训练 → 65+ far 测试
  (c) elderly_cr：65+ 单独训练 + 竞争死亡风险交互特征 → 65+ far 测试

竞争死亡风险特征（c 臂新增 13 维）：
  - AGE_SQ（年龄平方，捕获老年段内非线性死亡风险陡增）
  - AGE × 11 个合并症二值（AIDS/酒精/糖尿病/其他疾病/毒品/
    吸烟/监狱/流浪/健康服务/移民/政府救济）——合并症的死亡
    权重随年龄变化（老年合并症→非 TB 竞争死亡通路的代理）
  - AGE × 痰涂阳性（老年涂阳的临床意义不同）

校准（65+ far 测试集，各臂最优模型）：十分位可靠性曲线
（预测概率 vs 观测率）+ Brier + ECE（期望校准误差）。

归档：data/processed/sinan_elderly_20260826.json
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

from scipy import sparse  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score,  # noqa: E402
                             brier_score_loss, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

PROC = os.path.join(BASE, 'tb_risk', 'data', 'processed')
OUT = os.path.join(PROC, 'sinan_elderly_20260826.json')
SEED = 42
ELDERLY_MIN = 65

# 与 run_sinan_scale_training.py 一致的特征定义
BIN_AGRAV = ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
             'AGRAVDROGA', 'AGRAVTABAC', 'POP_LIBER', 'POP_RUA',
             'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV']
CAT_FEATS = ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'FORMA', 'EXTRAPU1_N',
             'RAIOX_TORA', 'TESTE_TUBE', 'CULTURA_ES', 'HISTOPATOL',
             'TEST_MOLEC', 'BACILOSC_E', 'CS_GESTANT']
NUM_FEATS = ['AGE_YEARS', 'NU_CONTATO']
CR_INTERACT = BIN_AGRAV + ['SMEAR_POS']   # 竞争风险交互对象


def load_analysis():
    d = pd.read_parquet(
        os.path.join(PROC, 'sinan', 'tubebra_2001_2019.parquet'),
        columns=BIN_AGRAV + CAT_FEATS + ['AGE_YEARS', 'NU_CONTATO',
                                         'DT_NOTIFIC_Y', 'NU_ANO',
                                         'SITUA_ENCE'])
    d = d[d['SITUA_ENCE'].astype(str).isin(['1', '4'])].copy()
    y = (d['SITUA_ENCE'].astype(str) == '4').astype(int).values
    d['year'] = d['DT_NOTIFIC_Y'].fillna(d['NU_ANO'].astype(float))
    return d, y


def build_features(d, with_cr):
    """返回 (X 稀疏, feat_names)。with_cr=True 追加 13 维竞争风险特征。

    注意：矩阵在全量分析集上构造一次、下游行掩码切分
    （encoder 全量 fit，与 run_sinan_scale_training 口径一致，
    避免训练/测试分别 fit 的列错位）。
    """
    age = np.clip(pd.to_numeric(d['AGE_YEARS'], errors='coerce')
                  .fillna(d['AGE_YEARS'].median()).values, 0, 100)
    contato = np.clip(pd.to_numeric(d['NU_CONTATO'], errors='coerce')
                      .fillna(0).values, 0, 10)
    num = np.column_stack([age, contato]).astype(np.float32)
    cat_df = pd.DataFrame({
        c: d[c].astype(str).str.strip().replace('', 'UNK')
        for c in BIN_AGRAV + CAT_FEATS})
    enc = OneHotEncoder(handle_unknown='ignore')
    Z = enc.fit_transform(cat_df).astype(np.float32)
    blocks = [sparse.csr_matrix(num), Z]
    names = list(NUM_FEATS) + list(enc.get_feature_names_out())
    if with_cr:
        # 竞争死亡风险交互块
        age_c = age.astype(np.float32)
        smear_pos = pd.Series(d['BACILOSC_E'].astype(str).str.strip()
                               ).isin(['2', '3', '4']).values.astype(np.float32)
        inter = [age_c ** 2]
        inter_names = ['AGE_SQ']
        for c in BIN_AGRAV:
            b = (cat_df[c] == '1').values.astype(np.float32)
            inter.append(age_c * b)
            inter_names.append('AGE_X_' + c)
        inter.append(age_c * smear_pos)
        inter_names.append('AGE_X_SMEAR_POS')
        blocks.append(sparse.csr_matrix(
            np.column_stack(inter).astype(np.float32)))
        names += inter_names
    X = sparse.hstack(blocks, format='csr')
    return X, names


def make_models():
    models = {'LR': LogisticRegression(max_iter=2000, C=1.0,
                                       solver='lbfgs')}
    models['RF'] = RandomForestClassifier(
        n_estimators=80, min_samples_leaf=20, n_jobs=-1,
        random_state=SEED)
    if HAS_XGB:
        models['XGB'] = XGBClassifier(
            n_estimators=300, learning_rate=0.08, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, tree_method='hist',
            n_jobs=-1, random_state=SEED, eval_metric='logloss',
            verbosity=0)
    if HAS_LGBM:
        models['LGBM'] = LGBMClassifier(
            n_estimators=400, learning_rate=0.06, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=SEED, verbose=-1)
    if HAS_CAT:
        models['CAT'] = CatBoostClassifier(
            iterations=300, learning_rate=0.08, depth=6,
            random_seed=SEED, verbose=0, allow_writing_files=False,
            thread_count=os.cpu_count() or 4)
    return models


def eval_preds(y, p):
    return {'n': int(len(y)), 'events': int(y.sum()),
            'auroc': float(roc_auc_score(y, p)),
            'auprc': float(average_precision_score(y, p)),
            'brier': float(brier_score_loss(y, p))}


def calibration_curve(y, p, n_bins=10):
    """十分位可靠性曲线 + ECE。"""
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    bins = np.clip(np.searchsorted(qs, p, side='right') - 1, 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        mp, obs = float(p[m].mean()), float(y[m].mean())
        w = float(m.mean())
        ece += w * abs(mp - obs)
        rows.append({'bin': b, 'n': int(m.sum()),
                     'mean_predicted': round(mp, 4),
                     'observed_rate': round(obs, 4)})
    return {'bins': rows, 'ece': round(float(ece), 4),
            'brier': round(float(brier_score_loss(y, p)), 4),
            'n': int(len(y)), 'events': int(y.sum())}


def main():
    t0 = time.time()
    d, y = load_analysis()
    year = d['year'].values
    age = np.clip(pd.to_numeric(d['AGE_YEARS'], errors='coerce')
                  .fillna(d['AGE_YEARS'].median()).values, 0, 100)
    elderly = age >= ELDERLY_MIN
    print('analysis %d rows; elderly(65+) %d (%.1f%%), death %.2f%%'
          % (len(y), elderly.sum(), 100 * elderly.mean(),
             100 * y[elderly].mean()))

    # T1 时间外推（同规模实验）
    tr = (year >= 2001) & (year <= 2012)
    tf = (year >= 2018) & (year <= 2019)
    te = elderly & tf
    res = {'date': '2026-08-26', 'experiment': 'sinan_elderly_v1',
           'models_available': {'xgb': HAS_XGB, 'lgbm': HAS_LGBM,
                                 'cat': HAS_CAT},
           'arms': {
               'all_age': '全年龄 2001-12 训练 → 65+ 18-19 测试（基线）',
               'elderly': '65+ 2001-12 训练 → 65+ 18-19 测试',
               'elderly_cr': '65+ + 竞争死亡风险交互特征训练 → '
                             '65+ 18-19 测试'},
           'cr_features': ['AGE_SQ'] + ['AGE_X_' + c for c in BIN_AGRAV]
                         + ['AGE_X_SMEAR_POS'],
           'data': {
               'n_analysis': int(len(y)),
               'n_elderly': int(elderly.sum()),
               'elderly_death_pct': round(100 * float(y[elderly].mean()), 2),
               'train_elderly_n': int((elderly & tr).sum()),
               'test_elderly_65p_far_n': int(te.sum()),
               'test_events': int(y[te].sum())},
           'results': {}, 'calibration': {}}

    tr_all = tr
    tr_el = elderly & tr
    # 全量矩阵一次构造（with_cr 两个版本），行掩码切分
    X_base, _ = build_features(d, with_cr=False)
    X_cr, _ = build_features(d, with_cr=True)
    X_te = {'all_age': X_base[te], 'elderly': X_base[te],
            'elderly_cr': X_cr[te]}
    X_tr = {'all_age': X_base[tr_all], 'elderly': X_base[tr_el],
            'elderly_cr': X_cr[tr_el]}
    y_tr = {'all_age': y[tr_all], 'elderly': y[tr_el],
            'elderly_cr': y[tr_el]}

    preds = {}
    for arm_name in ('all_age', 'elderly', 'elderly_cr'):
        res['results'][arm_name] = {}
        preds[arm_name] = {}
        for mname, m in make_models().items():
            t1 = time.time()
            m.fit(X_tr[arm_name], y_tr[arm_name])
            p = m.predict_proba(X_te[arm_name])[:, 1]
            preds[arm_name][mname] = p
            entry = eval_preds(y[te], p)
            entry['train_time_s'] = round(time.time() - t1, 1)
            res['results'][arm_name][mname] = entry
            print('%-11s %-4s %5.1fs auroc=%.4f brier=%.4f'
                  % (arm_name, mname, entry['train_time_s'],
                     entry['auroc'], entry['brier']))

    # 校准曲线：每臂最优模型（按 far AUROC），直接复用缓存预测
    for arm_name in res['results']:
        best = max(res['results'][arm_name].items(),
                   key=lambda kv: kv[1]['auroc'])
        res['calibration'][arm_name] = {
            'model': best[0],
            **calibration_curve(y[te], preds[arm_name][best[0]])}
        print('calibration %s (%s): ECE=%.4f Brier=%.4f'
              % (arm_name, best[0], res['calibration'][arm_name]['ece'],
                 res['calibration'][arm_name]['brier']))

    # 汇总判定：老年专项净增量（c 臂 vs a 臂，逐模型）
    delta = {}
    for mname in res['results']['all_age']:
        if mname in res['results']['elderly_cr']:
            delta[mname] = round(
                res['results']['elderly_cr'][mname]['auroc']
                - res['results']['all_age'][mname]['auroc'], 4)
    res['delta_elderly_cr_vs_all_age'] = delta
    best_gain = max(delta.values()) if delta else None
    res['verdict'] = {
        'best_elderly_cr_gain': best_gain,
        'note': ('65+ 塌方（0.670）经老年专项训练与竞争风险交互的'
                 '修复幅度见 delta_elderly_cr_vs_all_age；校准曲线'
                 '供老年筛查阈值政策参考')}

    res['total_seconds'] = round(time.time() - t0, 1)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('saved', OUT)
    print('delta elderly_cr vs all_age:', delta)
    return res


if __name__ == '__main__':
    main()
