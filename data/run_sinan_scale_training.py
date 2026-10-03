#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大规模真实个体数据训练 + 外部验证实验（第五轮，2026-08-26）。

数据：
1. 主力：SINAN 巴西全国结核通报 2001-2019（~150 万个体级记录，
   DataSUS TUBEBR01-19.dbc，纯 Python blast 解析管线）。
   任务：诊断时点特征 → 治疗终末 TB 死亡（SITUA_ENCE=4）vs 治愈（=1）。
   外部验证两个轴：
   - T1 时间外推：train 2001-2012 / val 2013-2015 /
     test-near 2016-2017 / test-far 2018-2019；
   - T2 空间外推：训练排除东南大区（SP/RJ/MG/ES，全时段 2001-2017），
     test = 东南大区 2018-2019。
2. 对比基准：巴基斯坦儿科 TB 筛查 5,880 例（Diagnosed 标签，
   5 模型 5 折 CV 小样本对照）+ 效应方向一致性检验。
3. 蒙古 MDR 接触者登记（66 例）：派生结局描述性关联（不建模）。

模型：项目 5 基模型 LR / RF / XGB / LGBM / CAT（缺失自动降级如实记录）。

归档：data/processed/sinan_scale_training_20260826.json
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
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
OUT = os.path.join(PROC, 'sinan_scale_training_20260826.json')
SEED = 42

# 诊断时点可得特征（排除治疗中信息防泄漏：TRAT_SUPER/DOENCA_TRA/
# SITUA_9_M/SITUA_12_M/BACILOSC_1-6/药物列等）
BIN_AGRAV = ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
             'AGRAVDROGA', 'AGRAVTABAC', 'POP_LIBER', 'POP_RUA',
             'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV']
CAT_FEATS = ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'FORMA', 'EXTRAPU1_N',
             'RAIOX_TORA', 'TESTE_TUBE', 'CULTURA_ES', 'HISTOPATOL',
             'TEST_MOLEC', 'BACILOSC_E', 'CS_GESTANT']
NUM_FEATS = ['AGE_YEARS', 'NU_CONTATO']


def load_sinan():
    return pd.read_parquet(os.path.join(PROC, 'sinan',
                                        'tubebra_2001_2019.parquet'))


def build_matrix(df):
    """SINAN → (X 稀疏, y, meta)。y=1 为 TB 死亡。"""
    d = df.copy()
    d['year'] = d['DT_NOTIFIC_Y'].fillna(d['NU_ANO'].astype(float))
    d = d[d['SITUA_ENCE'].astype(str).isin(['1', '4'])].copy()
    y = (d['SITUA_ENCE'].astype(str) == '4').astype(int).values
    uf = d['SG_UF_NOT'].astype(str).str.strip()
    region = uf.str[0].map({'1': 'N', '2': 'NE', '3': 'SE', '4': 'S',
                            '5': 'CO'}).fillna('UNK').values

    num = np.column_stack([
        np.clip(pd.to_numeric(d['AGE_YEARS'], errors='coerce')
                .fillna(d['AGE_YEARS'].median()).values, 0, 100),
        np.clip(pd.to_numeric(d['NU_CONTATO'], errors='coerce')
                .fillna(0).values, 0, 10)]).astype(np.float32)
    cat_df = pd.DataFrame({
        c: d[c].astype(str).str.strip().replace('', 'UNK')
        for c in BIN_AGRAV + CAT_FEATS})
    enc = OneHotEncoder(handle_unknown='ignore')
    Z = enc.fit_transform(cat_df).astype(np.float32)
    X = sparse.hstack([sparse.csr_matrix(num), Z], format='csr')
    feat_names = list(NUM_FEATS) + list(enc.get_feature_names_out())
    meta = pd.DataFrame({
        'year': d['year'].values, 'region': region,
        'AGE_YEARS': num[:, 0],
        'SEX': d['CS_SEXO'].astype(str).values,
        'SMEAR': d['BACILOSC_E'].astype(str).values,
        'AIDS': d['AGRAVAIDS'].astype(str).values,
        'UF': uf.values})
    return X, y, meta, feat_names


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


def subgroup_auroc(name, meta, y, p):
    """亚组 AUROC（T1 far 测试集上）。"""
    out = {}
    masks = {
        'age_0_14': meta['AGE_YEARS'] < 15,
        'age_15_49': (meta['AGE_YEARS'] >= 15) & (meta['AGE_YEARS'] < 50),
        'age_50_64': (meta['AGE_YEARS'] >= 50) & (meta['AGE_YEARS'] < 65),
        'age_65p': meta['AGE_YEARS'] >= 65,
        'male': meta['SEX'] == 'M', 'female': meta['SEX'] == 'F',
        'smear_pos': meta['SMEAR'].isin(['2', '3', '4']),
        'smear_neg': meta['SMEAR'] == '1',
        'aids': meta['AIDS'] == '1', 'no_aids': meta['AIDS'] != '1',
    }
    for k, m in masks.items():
        yy, pp = y[m.values], p[m.values]
        if len(yy) and yy.sum() > 50 and yy.mean() < 1:
            out[k] = {'n': int(len(yy)), 'events': int(yy.sum()),
                      'auroc': float(roc_auc_score(yy, pp))}
    return out


def run_sinan(res):
    t0 = time.time()
    raw = load_sinan()
    X, y, meta, feat_names = build_matrix(raw)
    del raw
    res['data'] = {'n_total': int(len(y)),
                   'events_tb_death': int(y.sum()),
                   'prevalence_pct': round(float(y.mean() * 100), 2),
                   'n_features': int(X.shape[1]),
                   'age_median': float(np.median(meta['AGE_YEARS'])),
                   'year_range': [float(meta['year'].min()),
                                  float(meta['year'].max())]}
    print('analysis set: %d rows, %d events (%.2f%%), %d features'
          % (len(y), y.sum(), y.mean() * 100, X.shape[1]))

    year = meta['year'].values
    region = meta['region'].values

    # ---------- T1 时间外推 ----------
    tr = (year >= 2001) & (year <= 2012)
    va = (year >= 2013) & (year <= 2015)
    tn = (year >= 2016) & (year <= 2017)
    tf = (year >= 2018) & (year <= 2019)
    res['T1_split'] = {k: int(v.sum()) for k, v in
                       [('train_01_12', tr), ('val_13_15', va),
                        ('test_near_16_17', tn), ('test_far_18_19', tf)]}
    res['T1_temporal'] = {}
    best_far = None
    best_far_auroc = -1.0
    for name, m in make_models().items():
        t1 = time.time()
        m.fit(X[tr], y[tr])
        entry = {}
        for split, mask in [('val', va), ('near', tn), ('far', tf)]:
            p = m.predict_proba(X[mask])[:, 1]
            entry[split] = eval_preds(y[mask], p)
            if split == 'far' and entry['far']['auroc'] > best_far_auroc:
                best_far_auroc = entry['far']['auroc']
                best_far = (name, m, p, mask)
        res['T1_temporal'][name] = entry
        print('T1 %-4s %5.1fs auroc val/near/far = %.4f/%.4f/%.4f'
              % (name, time.time() - t1, entry['val']['auroc'],
                 entry['near']['auroc'], entry['far']['auroc']))
        res['T1_temporal'][name]['train_time_s'] = round(time.time() - t1, 1)

    # 亚组（far 测试集，最优树模型 + LR）
    name, m, p, mask = best_far
    res['subgroup'][name + '_far'] = subgroup_auroc(
        name, meta[mask].reset_index(drop=True), y[mask], p)

    # ---------- T2 空间外推 ----------
    se = region == 'SE'
    tr2 = (~se) & (year >= 2001) & (year <= 2017)
    te2 = se & (year >= 2018) & (year <= 2019)
    res['T2_split'] = {'train_nonSE_01_17': int(tr2.sum()),
                       'test_SE_18_19': int(te2.sum())}
    res['T2_spatial'] = {}
    for name, m in make_models().items():
        t1 = time.time()
        m.fit(X[tr2], y[tr2])
        p = m.predict_proba(X[te2])[:, 1]
        res['T2_spatial'][name] = eval_preds(y[te2], p)
        res['T2_spatial'][name]['train_time_s'] = round(time.time() - t1, 1)
        print('T2 %-4s %5.1fs auroc=%.4f'
              % (name, time.time() - t1, res['T2_spatial'][name]['auroc']))

    # ---------- 特征重要性（XGB gain，T1 训练集重训轻量版） ----------
    if HAS_XGB:
        xg = XGBClassifier(n_estimators=200, learning_rate=0.08, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8,
                           tree_method='hist', n_jobs=-1, random_state=SEED,
                           eval_metric='logloss', verbosity=0)
        xg.fit(X[tr], y[tr])
        imp = xg.feature_importances_
        order = np.argsort(imp)[::-1][:25]
        res['feature_importance']['xgb_gain_top25'] = [
            {'feature': feat_names[i], 'gain': round(float(imp[i]), 5)}
            for i in order]
    res['sinan_total_seconds'] = round(time.time() - t0, 1)


def run_pakistan(res):
    """巴基斯坦儿科筛查 5 模型 5 折 CV + 方向一致性。"""
    path = os.path.join(BASE, 'tb_risk', 'data', 'raw', 'pakistan_child_tb',
                        'Dataset 30Dec2021.tab')
    df = pd.read_csv(path, sep='\t')
    y = df['Diagnosed'].values
    feats = ['Gender', 'Age_years', 'Fever', 'Weightloss', 'Cough',
             'Cough_duration', 'BCG_scar', 'family_history',
             'Family_member_TB', 'Family_sputum_pos', 'Rural_Facility']
    X = df[feats].copy()
    X['Gender'] = (X['Gender'] == 'M').astype(float)
    X = X.fillna(-1.0).astype(float)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res['pakistan'] = {'n': int(len(y)), 'events': int(y.sum()),
                       'prevalence_pct': round(float(y.mean() * 100), 1),
                       'features': feats, 'cv_auroc': {},
                       # PACTS 先例同构的终点泄漏警示（第六轮 P2 补）：
                       # 防止未来误引 0.97 作为性能证据
                       'interpretation': 'directional_reference_only',
                       'endpoint_leakage_caveat': (
                           '「Diagnosed」标签部分由诊断流程特征嵌入：'
                           'family_history r≈0.78 / Family_member_TB / '
                           'Family_sputum_pos 为接触调查转诊路径（确诊'
                           '依据回填），与 PACTS 基线症状终点泄漏同构'
                           '（见 PACTS demoted 先例）。AUROC 0.96-0.98 '
                           '属诊断流程饱和上限，不可与真实前筛查人群'
                           '性能可比，仅作效应方向对照，禁止作为模型'
                           '性能证据引用。')}
    for name, m in make_models().items():
        try:
            p = cross_val_predict(m, X.values, y, cv=cv,
                                  method='predict_proba')[:, 1]
            res['pakistan']['cv_auroc'][name] = {
                'auroc': float(roc_auc_score(y, p)),
                'auprc': float(average_precision_score(y, p))}
            print('PK %-4s auroc=%.4f' % (name, roc_auc_score(y, p)))
        except Exception as e:
            res['pakistan']['cv_auroc'][name] = {'error': str(e)}

    # 方向一致性：单变量效应方向（Spearman/简单 LR z 值符号）
    from scipy.stats import pointbiserialr
    dirn = {}
    for c in feats:
        if c == 'Gender':
            v = X[c].values
        else:
            v = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median())
        ok = ~np.isnan(v)
        r, pv = pointbiserialr(y[ok], v[ok])
        dirn[c] = {'r': round(float(r), 4), 'p': float(pv)}
    res['pakistan']['univariate_direction'] = dirn


def run_mongolia(res):
    """蒙古 66 接触者：派生结局描述性关联。"""
    try:
        path = os.path.join(BASE, 'tb_risk', 'data', 'raw', 'mongolia_mdr',
                            'Additional File 2 - Contact eregistry data.xlsx')
        c = pd.read_excel(path)
        c = c[pd.to_numeric(c['gender'], errors='coerce').isna()]
        t3 = pd.read_excel(
            os.path.join(BASE, 'tb_risk', 'data', 'raw', 'mongolia_mdr',
                         'Additional file 3 - Contact eregistry test '
                         'results data.xlsx'))
        sput = t3[t3['appt'] == 'initial'].set_index('DRTB_registry_number')
        # 派生：任何时点痰阳或 CXR 异常
        agg = t3.groupby('DRTB_registry_number').agg(
            sput_pos=('sputum_results-sputum2',
                      lambda s: (s == 'positive').any()),
            cxr_abn=('cxrnormal', lambda s: (s == 'abnormal').any()))
        c = c.set_index('DRTB_registry_number')
        c = c.join(agg)
        c['tb_outcome'] = (c['sput_pos'] | c['cxr_abn']).astype(int)
        n = int(c['tb_outcome'].notna().sum())
        ev = int(c['tb_outcome'].sum())
        prev = round(float(c['tb_outcome'].mean() * 100), 1)
        res['mongolia'] = {
            'n_contacts': n, 'derived_events': ev,
            'prevalence_pct': prev,
            'note': '样本量不支持建模，仅描述性记录'}
        print('MN n=%d events=%d (%.1f%%)' % (n, ev, prev))
    except Exception as e:
        res['mongolia'] = {'error': str(e)}


def run():
    res = {'date': '2026-08-26', 'experiment': 'sinan_scale_training_v1',
           'models_available': {'xgb': HAS_XGB, 'lgbm': HAS_LGBM,
                                'cat': HAS_CAT},
           'T1_temporal': {}, 'T2_spatial': {}, 'subgroup': {},
           'feature_importance': {}, 'pakistan': {}, 'mongolia': {}}
    run_sinan(res)
    run_pakistan(res)
    run_mongolia(res)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('saved', OUT)
    return res


if __name__ == '__main__':
    run()
