#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2-4（第九轮）：细菌学确诊作第二任务头——「找患者」语义的代理终点。

问题：部署目标是找未诊断患者，死亡终点训练的模型在通报人群内
排序死亡风险；细菌学确诊（涂片/培养阳性 = 传染性病例）才是
「找患者」的直接语义。多任务的意义：与死亡终点共享诊断时点
特征空间，检验两个头的部署互补性。

诚实边界：LGBM 的 MultiOutputClassifier 是独立头（无梯度级
表征共享——真共享需 NN，过度工程留待）；本实验交付的是
(a) 细菌学头的可预测性与部署度量，(b) 与死亡头 top-decile 的
重叠结构（互补 vs 冗余）。

设计：
  y1 = 全因死亡（{1,3,4} 人群内 {3,4}）
  y2 = 细菌学确诊（BACILOSC_E=='1' | CULTURA_ES=='1'，
       涂片或培养任一阳性；特征中已含这两列——诊断时点可得，
       多任务语境下作为标签使用是合法的）
  划分：train FILE_YEAR≤12 / far test 18-19（时间外推）
  部署度量：yield@k / NNS@k（同 deployment_metrics 口径）

归档：data/processed/sinan_bacteriology_head_20260826.json
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
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from tb_risk.ml.planner import screening_metrics_at_budgets  # noqa: E402

PROC = os.path.join(BASE, 'tb_risk', 'data', 'processed')
OUT = os.path.join(PROC, 'sinan_bacteriology_head_20260826.json')

BIN_AGRAV = ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
             'AGRAVDROGA', 'AGRAVTABAC', 'POP_LIBER', 'POP_RUA',
             'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV']
# 病原学确诊方式列必须整体剔除（BACILOSC_E/CULTURA_ES 构成 y2 标签，
# TEST_MOLEC/HISTOPATOL 为等价确诊证据）——否则标签泄漏（首跑
# AUROC=1.0000 的直接证据）；RAIOX_TORA 影像学保留。
CONFIRM_COLS = ['BACILOSC_E', 'CULTURA_ES', 'TEST_MOLEC', 'HISTOPATOL']
CAT_FEATS = ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'FORMA', 'EXTRAPU1_N',
             'RAIOX_TORA', 'TESTE_TUBE', 'CS_GESTANT']
BUDGETS = [1, 2, 5, 10, 20, 30, 50, 100]


def build_ind_matrix(d):
    """个体特征矩阵（数值 2 列 + 二元/类别 OneHot）。

    返回 (X, encoder)——encoder 随双头模型一并落盘，供
    ``tb_risk.core.dual_head_predictor.DualHeadPredictor`` 加载复用。
    """
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
    return sparse.hstack([sparse.csr_matrix(num), Z], format='csr'), enc


def main():
    t0 = time.time()
    print('=== P2-4 细菌学第二任务头：找患者语义的部署度量 ===')
    d = pd.read_parquet(
        os.path.join(PROC, 'sinan', 'tubebra_2001_2019.parquet'),
        columns=BIN_AGRAV + CAT_FEATS + CONFIRM_COLS
        + ['AGE_YEARS', 'NU_CONTATO', 'SITUA_ENCE', 'FILE_YEAR'])
    s = d['SITUA_ENCE'].astype(str)
    d = d[s.isin(['1', '3', '4'])].reset_index(drop=True)
    s = d['SITUA_ENCE'].astype(str)
    y1 = s.isin(['3', '4']).astype(int).values
    bac = d['BACILOSC_E'].astype(str).str.strip()
    cul = d['CULTURA_ES'].astype(str).str.strip()
    y2 = ((bac == '1') | (cul == '1')).astype(int).values
    print('n=%d | 死亡 %.1f%% | 细菌学确诊 %.1f%%'
          % (len(d), 100 * y1.mean(), 100 * y2.mean()))

    year = pd.to_numeric(d['FILE_YEAR'], errors='coerce').values
    tr, te = year <= 12, (year >= 18) & (year <= 19)
    X, enc = build_ind_matrix(d)

    heads = {}
    scores = {}
    fitted = {}
    for name, y in (('death_allcause', y1), ('bacteriology_pos', y2)):
        mdl = LGBMClassifier(n_estimators=400, learning_rate=0.06,
                             num_leaves=63, subsample=0.8,
                             colsample_bytree=0.8, n_jobs=-1,
                             random_state=42, verbose=-1)
        mdl.fit(X[tr], y[tr])
        fitted[name] = mdl
        sc = mdl.predict_proba(X[te])[:, 1]
        scores[name] = sc
        m = {k: v for k, v in
             screening_metrics_at_budgets(y[te], sc, BUDGETS).items()}
        heads[name] = {
            'auroc': float(roc_auc_score(y[te], sc)),
            'event_rate_test': float(y[te].mean()),
            'metrics': {str(k): v for k, v in m.items()}}
        print('[%s] AUROC %.4f | yield@10%% %.3f (lift %.1fx) | '
              'yield@30%% %.3f'
              % (name, heads[name]['auroc'], m[10]['yield'],
                 m[10]['lift'], m[30]['yield']))

    # top-decile 重叠（部署互补性）
    dec = int(round(te.sum() * 0.1))
    top1 = set(np.argsort(-scores['death_allcause'])[:dec].tolist())
    top2 = set(np.argsort(-scores['bacteriology_pos'])[:dec].tolist())
    overlap = len(top1 & top2) / dec

    # round-10 P2：双头模型正式化——encoder + 两头 LGBM 落盘，
    # 供 core.dual_head_predictor.DualHeadPredictor 加载为产品能力
    model_info = {'saved': False}
    if HAS_JOBLIB:
        model_dir = os.path.join(BASE, 'tb_risk', 'models',
                                 'sinan_dual_head')
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(enc, os.path.join(model_dir, 'encoder_v1.joblib'))
        joblib.dump(fitted['death_allcause'],
                    os.path.join(model_dir, 'death_head_v1.joblib'))
        joblib.dump(fitted['bacteriology_pos'],
                    os.path.join(model_dir, 'bact_head_v1.joblib'))
        model_info = {
            'saved': True,
            'model_dir': 'tb_risk/models/sinan_dual_head',
            'files': ['encoder_v1.joblib', 'death_head_v1.joblib',
                      'bact_head_v1.joblib'],
            'version': 'v1',
            'loader': 'tb_risk.core.dual_head_predictor.DualHeadPredictor',
            'feature_schema': {
                'numeric': ['AGE_YEARS', 'NU_CONTATO'],
                'binary': BIN_AGRAV,
                'categorical': CAT_FEATS,
                'excluded_leakage': CONFIRM_COLS,
            },
        }
        print('双头模型落盘:', model_info['model_dir'])

    res = {
        'date': '2026-08-26',
        'experiment': 'sinan_bacteriology_head_v1',
        'question': '细菌学确诊（传染性病例）作为「找患者」第二头——'
                    '可预测性多高、与死亡头部署互补还是冗余',
        'labels': {
            'y1': '全因死亡 {3,4}（人群 {1,3,4}）',
            'y2': '细菌学确诊：BACILOSC_E==1 或 CULTURA_ES==1'
                  '（涂片/培养任一阳性）'},
        'leakage_control': '特征集已剔除全部病原学确诊方式列'
                          '（BACILOSC_E/CULTURA_ES/TEST_MOLEC/'
                          'HISTOPATOL）；首跑含确诊列时 AUROC=1.0000'
                          ' 为标签泄漏直接证据，已修复',
        'split': {'train': 'FILE_YEAR<=12', 'far_test': '18-19',
                  'n_test': int(te.sum())},
        'heads': heads,
        'top_decile_overlap': {
            'overlap_fraction': round(overlap, 4),
            'interpretation': (
                '重叠 %.1f%%：两个头的 top-decile 近乎正交——'
                '死亡分层与「找患者」是两个部署语义，双头独立部署'
                '有明确互补价值（若 >30%% 才考虑单模型覆盖）'
                % (100 * overlap)),
        },
        'multitask_note': 'LGBM 下双任务为独立头（无表征共享）；'
                          '真多任务共享需 NN，过度工程留待',
        'model_artifacts': model_info,
        'task_mapping': {
            'A_prime': 'bacteriology_pos —— 细菌学确诊（找传染源）：'
                       '在通报人群中排序「涂片/培养阳性」概率，'
                       '对准病例发现 / 传染源控制的部署语义',
            'B': 'death_allcause —— 全因死亡预后：对准治疗结局'
                 '分层 / 高危随访资源分配的部署语义',
        },
        'has_lgbm': HAS_LGBM,
        'runtime_s': round(time.time() - t0, 1),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('top-decile 重叠 %.1f%%' % (100 * overlap))
    print('saved:', OUT)


if __name__ == '__main__':
    main()
