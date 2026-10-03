#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1（第七轮）：老年任务「换终点」实验——终点属性 vs 人群属性，
2026-08-26。

背景：round-6 老年专项证明 65+ 的死亡二分类判别塌方（AUROC 0.670）
不可经专项训练或竞争风险交互修复（诚实阴性）→ 塌方可能是
任务属性（死亡终点在老年人群本就低信号）。本轮换终点检验：
判别力是否在更 TB 特异的结局上恢复。

标签字典（run_sinan_label_dictionary.py 本轮裁定）：
  1 治愈 / 2 放弃治疗（失访）/ 3 TB 死亡 / 4 非 TB 死亡 / 5 转出
  （round-5 的 y=4「TB 死亡」假设已更正：4 实为非 TB 死亡，
   2001-05 段因值 3 未启用含全因死亡成分。）

二分类终点组（T1 时间外推：train 2001-12 → far test 18-19）：
  E1  非 TB 死亡（4 vs 1）            ——旧终点基线复现（应≈0.67）
  E2  全因死亡（{3,4} vs 1）           ——修正后的稳健死亡终点
  E3  TB 死亡（3 vs 1）               ——真 TB 死亡（2006+ 语义一致，
                                          2001-05 段值 3 未启用的时期
                                          漂移如实记录）
  E4  不良结局（{2,3,4,5} vs 1）       ——WHO 标准复合终点
  E5  治愈 vs 失访+转出（{2,5} vs 1，非死亡人群内）
  E6  治愈 vs 失访（2 vs 1）
  E7  治愈 vs 转出（5 vs 1）
两个人群：all_age（参照，检验塌方是否老年特异）与 65+。

多类别（softmax，LGBM）：5 类 {1,2,3,4,5}，每类 One-vs-Rest
AUROC + macro——「多任务头」版本，检验类别间判别结构。

模型：LR + LGBM（5 模型成本高；round-6 已证模型间差异 <0.003）。
统计：far 测试集行级 bootstrap（500 次）AUROC 95% CI（LGBM）。

归档：data/processed/sinan_endpoint_switch_20260826.json
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

PROC = os.path.join(BASE, 'tb_risk', 'data', 'processed')
OUT = os.path.join(PROC, 'sinan_endpoint_switch_20260826.json')
SEED = 42
ELDERLY_MIN = 65
N_BOOT = 500

BIN_AGRAV = ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
             'AGRAVDROGA', 'AGRAVTABAC', 'POP_LIBER', 'POP_RUA',
             'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV']
CAT_FEATS = ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'FORMA', 'EXTRAPU1_N',
             'RAIOX_TORA', 'TESTE_TUBE', 'CULTURA_ES', 'HISTOPATOL',
             'TEST_MOLEC', 'BACILOSC_E', 'CS_GESTANT']

CLASS_NAMES = {'1': 'cured', '2': 'lost', '3': 'tb_death',
               '4': 'non_tb_death', '5': 'transfer'}

# 终点组定义：(名称, 人群类别集, 阳性类别集, 说明)
ENDPOINTS = [
    ('E1_death_nontb', ['1', '4'], ['4'],
     '非 TB 死亡（旧终点基线复现；round-5/6 的 y=4）'),
    ('E2_death_all', ['1', '3', '4'], ['3', '4'],
     '全因死亡（修正后的稳健死亡终点）'),
    ('E3_death_tb', ['1', '3'], ['3'],
     'TB 死亡（2006+ 语义一致；2001-05 段值 3 未启用，训练分布'
     '含时期漂移）'),
    ('E4_unfavorable', ['1', '2', '3', '4', '5'], ['2', '3', '4', '5'],
     '不良结局（WHO 复合终点：非治愈）'),
    ('E5_cure_vs_lost_transfer', ['1', '2', '5'], ['2', '5'],
     '治愈 vs 失访+转出（非死亡人群内，用户指定的 TB 特异终点）'),
    ('E6_cure_vs_lost', ['1', '2'], ['2'], '治愈 vs 失访'),
    ('E7_cure_vs_transfer', ['1', '5'], ['5'], '治愈 vs 转出'),
]


def load_analysis():
    d = pd.read_parquet(
        os.path.join(PROC, 'sinan', 'tubebra_2001_2019.parquet'),
        columns=BIN_AGRAV + CAT_FEATS + ['AGE_YEARS', 'NU_CONTATO',
                                         'DT_NOTIFIC_Y', 'NU_ANO',
                                         'SITUA_ENCE'])
    d['s'] = d['SITUA_ENCE'].astype(str).str.strip()
    d = d[d['s'].isin(['1', '2', '3', '4', '5'])].copy()
    d['year'] = d['DT_NOTIFIC_Y'].fillna(
        pd.to_numeric(d['NU_ANO'], errors='coerce'))
    return d


def build_features(d):
    """159 维基线特征（同 run_sinan_scale_training 口径）。
    矩阵在全量上构造一次，下游行掩码切分。"""
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
    X = sparse.hstack([sparse.csr_matrix(num), Z], format='csr')
    return X


def make_models(seed=SEED):
    models = {'LR': LogisticRegression(max_iter=2000, C=1.0,
                                       solver='lbfgs', random_state=seed)}
    if HAS_LGBM:
        models['LGBM'] = LGBMClassifier(
            n_estimators=400, learning_rate=0.06, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=seed, verbose=-1)
    return models


def boot_auroc_ci(y, p, n_boot=N_BOOT, seed=SEED):
    """行级 bootstrap AUROC 95% CI（单类重采样跳过）。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals.append(roc_auc_score(yb, p[idx]))
    if not vals:
        return [None, None]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def main():
    t0 = time.time()
    d = load_analysis()
    year = d['year'].values
    age = np.clip(pd.to_numeric(d['AGE_YEARS'], errors='coerce')
                  .fillna(d['AGE_YEARS'].median()).values, 0, 100)
    elderly = age >= ELDERLY_MIN
    s = d['s'].values
    print('analysis set %d rows (5 outcome classes), 65+ %.1f%%'
          % (len(s), 100 * elderly.mean()))
    print('class distribution:',
          {CLASS_NAMES[k]: int((s == k).sum()) for k in '12345'})

    X = build_features(d)
    del d
    n = len(s)

    res = {'date': '2026-08-26',
           'experiment': 'sinan_endpoint_switch_v1',
           'question': '65+ 死亡判别塌方是任务属性还是人群属性——'
                       '换终点能否恢复判别力',
           'label_dictionary': 'sinan_label_dictionary_20260826.json '
                              '（1 治愈/2 失访/3 TB死亡/4 非TB死亡/5 转出；'
                              'round-5 的 y=4 假设已更正）',
           'models_available': {'lgbm': HAS_LGBM},
           'endpoints': {name: {'population': pop, 'positive': pos,
                                'note': note}
                         for name, pop, pos, note in ENDPOINTS},
           'data': {'n_analysis_5class': int(n),
                    'elderly_n': int(elderly.sum()),
                    'class_counts': {CLASS_NAMES[k]: int((s == k).sum())
                                     for k in '12345'}},
           'results': {}, 'multiclass': {}}

    tr_years = (year >= 2001) & (year <= 2012)
    te_years = (year >= 2018) & (year <= 2019)

    # ---------- 二分类终点组 ----------
    for name, pop, pos, note in ENDPOINTS:
        pop_mask = np.isin(s, pop)
        y_all = np.isin(s, pos).astype(int)
        res['results'][name] = {}
        for pop_name, pop_base in (('all_age', np.ones(n, dtype=bool)),
                                   ('elderly_65p', elderly)):
            mask = pop_mask & pop_base
            tr = tr_years & mask
            te = te_years & mask
            if te.sum() < 200 or y_all[te].sum() < 20:
                continue
            res['results'][name][pop_name] = {}
            for mname, m in make_models().items():
                t1 = time.time()
                m.fit(X[tr], y_all[tr])
                p = m.predict_proba(X[te])[:, 1]
                entry = {'n': int(te.sum()),
                         'events': int(y_all[te].sum()),
                         'event_rate': round(float(y_all[te].mean()), 4),
                         'auroc': round(float(roc_auc_score(y_all[te], p)), 4),
                         'train_time_s': round(time.time() - t1, 1)}
                if mname == 'LGBM':
                    entry['auroc_ci95_boot'] = boot_auroc_ci(y_all[te], p)
                    preds_cache = p
                res['results'][name][pop_name][mname] = entry
            print('%-24s %-12s LR %.4f | LGBM %.4f (n=%d, ev=%.1f%%)'
                  % (name, pop_name,
                     res['results'][name][pop_name]['LR']['auroc'],
                     res['results'][name][pop_name]['LGBM']['auroc'],
                     te.sum(), 100 * y_all[te].mean()))

    # ---------- 多类别（softmax 多任务头） ----------
    for pop_name, pop_base in (('all_age', np.ones(n, dtype=bool)),
                               ('elderly_65p', elderly)):
        tr = tr_years & pop_base
        te = te_years & pop_base
        classes = sorted(set(s[te]) | set(s[tr]))
        ymap = {c: i for i, c in enumerate(classes)}
        y_tr = np.array([ymap[c] for c in s[tr]])
        y_te = np.array([ymap[c] for c in s[te]])
        entry = {'n_test': int(te.sum()), 'classes': classes}
        if HAS_LGBM:
            m = LGBMClassifier(
                n_estimators=400, learning_rate=0.06, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                random_state=SEED, verbose=-1)
            m.fit(X[tr], y_tr)
            P = m.predict_proba(X[te])
            per_class = {}
            aucs = []
            for i, c in enumerate(classes):
                yb = (y_te == i).astype(int)
                if yb.min() == yb.max():
                    continue
                a = float(roc_auc_score(yb, P[:, i]))
                per_class[CLASS_NAMES[c]] = {
                    'n': int(yb.sum()),
                    'auroc_ovr': round(a, 4)}
                aucs.append(a)
            entry['lgbm'] = {'per_class_ovr': per_class,
                             'macro_auroc': round(float(np.mean(aucs)), 4)}
            print('multiclass %s: macro OvR AUROC %.4f  %s'
                  % (pop_name, entry['lgbm']['macro_auroc'],
                     {k: v['auroc_ovr'] for k, v in per_class.items()}))
        res['multiclass'][pop_name] = entry

    # ---------- 核心判定 ----------
    def auroc_of(ep, pop_name, model='LGBM'):
        try:
            return res['results'][ep][pop_name][model]['auroc']
        except KeyError:
            return None

    e1 = auroc_of('E1_death_nontb', 'elderly_65p')
    e5 = auroc_of('E5_cure_vs_lost_transfer', 'elderly_65p')
    e5a = auroc_of('E5_cure_vs_lost_transfer', 'all_age')
    e1a = auroc_of('E1_death_nontb', 'all_age')
    res['verdict'] = {
        'elderly_E1_death_nontb': e1,
        'elderly_E5_cure_vs_lost_transfer': e5,
        'all_age_E1_death_nontb': e1a,
        'all_age_E5_cure_vs_lost_transfer': e5a,
        'note': '对比 E5 vs E1（65+）：判别力若恢复（显著更高），'
                '塌方为终点属性；E5 在 all_age 与 65+ 间的差距'
                '刻画人群残余差异'}

    res['total_seconds'] = round(time.time() - t0, 1)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('saved', OUT)
    print('verdict:', json.dumps(res['verdict'], ensure_ascii=False))
    return res


if __name__ == '__main__':
    main()
