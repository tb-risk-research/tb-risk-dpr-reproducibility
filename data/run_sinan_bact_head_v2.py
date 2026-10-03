#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1-3（第十一轮）：细菌学头 v2——症状列可行性裁定 + 扩展特征重跑。

round-12 P1-2 增补：v2_extended 双头模型落盘为产品工件 v2
（encoder/death_head/bact_head_v2.joblib + feature_schema 含
HIV/OCC_TRUNC 与职业 top-30 词表），供 DualHeadPredictor
MODEL_VERSION='v2' 加载——GUI 双头输出随之切换到 HIV/职业扩展版。

用户问题：细菌学头 AUROC 0.728、lift 1.36-1.44（基率 57% 排序天花板
低）——补 AUPRC，并评估从 SINAN 原始表恢复症状列的可行性。

可行性侦察结论（2026-08-26，TUBEBR19.dbc 94 字段全集实锤）：
  症状列（TOSSE/FEBRE/SINTOM/PERDA_PESO 等）**不存在**于 TUBEBR
  全国合并库——SINAN 全国导出把临床表收缩到诊断/治疗/结局块，
  症状块只在地方级 SINAN 个案库里。恢复不可行，属数据侧边界。

替代路径（本轮实验）：94 字段中有两个未利用的合法诊断时点特征：
  HIV      —— HIV 检测结果（1=阳 2=阴 3=未做 4=等待），与 AGRAVAIDS
              （AIDS 合并症标帜）不同源，诊断时点可得
  ID_OCUPA_N —— 职业编码（CBO，频次截断 top-30 + OTHER）
排除（标签/未来信息）：
  BACILOSC_1-6 / BACILOS_E2 —— 随访涂片（结局期 + 与标签同源）
  CULTURA_OU —— 培养结果（与 CULTURA_ES 同源）

设计：
  v2 特征 = v1 全部 + HIV + ID_OCUPA_N（OneHot top-30）
  对照 = v1（44 列中建模相关 21 列）零改动重跑
  双头（死亡/细菌学）+ AUROC + AUPRC（基率 57% 下 AUROC 信息量
  不足，AUPRC 才是排序质量的诚实度量）
  期望管理：HIV/职业若带来增益，量级预期 +0.01-0.03（症状块缺位
  的主体信息缺口无法替代）；无增益则如实归档「分诊工具，非诊断」

归档：data/processed/sinan_bact_head_v2_20260826.json
"""
import json
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from scipy import sparse  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
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

from tb_risk.data.dbc_reader import read_dbc  # noqa: E402
from tb_risk.ml.planner import screening_metrics_at_budgets  # noqa: E402

RAW_DIR = os.path.join(BASE, 'tb_risk', 'data', 'raw', 'sinan')
OUT = os.path.join(BASE, 'tb_risk', 'data', 'processed',
                   'sinan_bact_head_v2_20260826.json')

# v1 建模列（run_sinan_bacteriology_head.py 单一口径）
V1_COLS = (['AGE_YEARS', 'NU_CONTATO', 'SITUA_ENCE', 'FILE_YEAR',
            'BACILOSC_E', 'CULTURA_ES']
           + ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
              'AGRAVDROGA', 'AGRAVTABAC', 'POP_LIBER', 'POP_RUA',
              'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV']
           + ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'FORMA', 'EXTRAPU1_N',
              'RAIOX_TORA', 'TESTE_TUBE', 'CS_GESTANT'])
# v2 新增（原始 DBC 读，parquet 未含）
NEW_COLS = ['HIV', 'ID_OCUPA_N']
# 年龄解码需要的原始列
READ_COLS = sorted(set(V1_COLS + NEW_COLS + ['NU_IDADE_N']))

CONFIRM_COLS = ['BACILOSC_E', 'CULTURA_ES', 'TEST_MOLEC', 'HISTOPATOL']
BUDGETS = [1, 2, 5, 10, 20, 30, 50, 100]
OCC_TOP_K = 30


def decode_age(nu_idade):
    """SINAN 年龄编码 → 岁（process_sinan.py 同一逻辑）。"""
    nu = np.asarray(nu_idade, dtype=np.float64)
    out = np.full(nu.shape, np.nan)
    m = (nu >= 4000) & (nu <= 4130)
    out[m] = nu[m] - 4000
    m = (nu >= 3000) & (nu < 4000)
    out[m] = (nu[m] - 3000) / 12.0
    m = (nu >= 2000) & (nu < 3000)
    out[m] = (nu[m] - 2000) / 365.25
    m = (nu >= 1000) & (nu < 2000)
    out[m] = (nu[m] - 1000) / 8766.0
    m = (nu > 0) & (nu < 1000)
    out[m] = nu[m]
    return out


def load_sinan_v2():
    """重解析 19 个 DBC（v1 列 + HIV + ID_OCUPA_N）。"""
    frames = []
    for fn in sorted(f for f in os.listdir(RAW_DIR)
                     if f.endswith('.dbc')):
        df, hdr = read_dbc(os.path.join(RAW_DIR, fn), columns=READ_COLS)
        df['FILE_YEAR'] = int(fn.replace('TUBEBR', '')
                              .replace('.dbc', ''))
        frames.append(df)
        print('parsed', fn, len(df))
    d = pd.concat(frames, ignore_index=True)
    d['AGE_YEARS'] = decode_age(d['NU_IDADE_N'].values)
    return d


def build_matrix(d, use_new):
    """特征矩阵：v1 口径 +（可选）HIV/职业 OneHot。

    返回 (X, enc, occ_top)——encoder 与职业词表随 v2 模型一并落盘
    （occ_top 为 None 当 use_new=False）。
    """
    num = np.column_stack([
        np.clip(pd.to_numeric(d['AGE_YEARS'], errors='coerce')
                .fillna(d['AGE_YEARS'].median()).values, 0, 100),
        np.clip(pd.to_numeric(d['NU_CONTATO'], errors='coerce')
                .fillna(0).values, 0, 10)]).astype(np.float32)
    v1_bin = ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
              'AGRAVDROGA', 'AGRAVTABAC', 'POP_LIBER', 'POP_RUA',
              'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV']
    v1_cat = ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'FORMA', 'EXTRAPU1_N',
              'RAIOX_TORA', 'TESTE_TUBE', 'CS_GESTANT']
    cat = v1_bin + v1_cat
    occ_top = None
    if use_new:
        # HIV：1=阳 2=阴 3=未做 4=等待（保留全值域做 OneHot）
        cat = cat + ['HIV']
        # 职业：频次截断 top-K + OTHER（CBO 编码数百类）
        occ = d['ID_OCUPA_N'].astype(str).str.strip()
        top = {o for o, _ in Counter(occ).most_common(OCC_TOP_K)}
        occ_top = sorted(top)
        cat = cat + ['OCC_TRUNC']
        d = d.assign(OCC_TRUNC=occ.where(occ.isin(top), 'OTHER'))
    cat_df = pd.DataFrame({
        c: d[c].astype(str).str.strip().replace('', 'UNK')
        for c in cat})
    enc = OneHotEncoder(handle_unknown='ignore')
    Z = enc.fit_transform(cat_df).astype(np.float32)
    return sparse.hstack([sparse.csr_matrix(num), Z],
                         format='csr'), enc, occ_top


def main():
    t0 = time.time()
    print('=== P1-3 细菌学头 v2：症状列裁定 + HIV/职业扩展 ===')
    if not HAS_LGBM:
        raise SystemExit('需要 lightgbm')

    d = load_sinan_v2()
    s = d['SITUA_ENCE'].astype(str)
    d = d[s.isin(['1', '3', '4'])].reset_index(drop=True)
    s = d['SITUA_ENCE'].astype(str)
    y1 = s.isin(['3', '4']).astype(int).values
    bac = d['BACILOSC_E'].astype(str).str.strip()
    cul = d['CULTURA_ES'].astype(str).str.strip()
    y2 = ((bac == '1') | (cul == '1')).astype(int).values
    year = pd.to_numeric(d['FILE_YEAR'], errors='coerce').values
    tr, te = year <= 12, (year >= 18) & (year <= 19)
    print('n=%d | 死亡 %.1f%% | 细菌学 %.1f%%'
          % (len(d), 100 * y1.mean(), 100 * y2.mean()))

    results = {}
    scores = {}
    v2_fitted = {}
    v2_enc = None
    v2_occ_top = None
    for tag, use_new in (('v1_baseline', False), ('v2_extended', True)):
        X, enc, occ_top = build_matrix(d, use_new)
        head_res = {}
        for name, y in (('death_allcause', y1),
                        ('bacteriology_pos', y2)):
            mdl = LGBMClassifier(n_estimators=400, learning_rate=0.06,
                                 num_leaves=63, subsample=0.8,
                                 colsample_bytree=0.8, n_jobs=-1,
                                 random_state=42, verbose=-1)
            mdl.fit(X[tr], y[tr])
            sc = mdl.predict_proba(X[te])[:, 1]
            if tag == 'v1_baseline':
                scores.setdefault(name, {})
                scores[name][tag] = sc
            else:
                scores[name][tag] = sc
                v2_fitted[name] = mdl
                v2_enc, v2_occ_top = enc, occ_top
            m = screening_metrics_at_budgets(y[te], sc, BUDGETS)
            head_res[name] = {
                'auroc': float(roc_auc_score(y[te], sc)),
                'auprc': float(average_precision_score(y[te], sc)),
                'event_rate_test': float(y[te].mean()),
                'yield10': m[10]['yield'],
                'lift10': m[10]['lift'],
            }
            print('[%s/%s] AUROC %.4f | AUPRC %.4f | yield@10%% %.3f'
                  % (tag, name, head_res[name]['auroc'],
                     head_res[name]['auprc'], m[10]['yield']))
        results[tag] = head_res

    # 双头 top-decile 重叠（v2）
    dec = int(round(te.sum() * 0.1))
    t1 = set(np.argsort(-scores['death_allcause']['v2_extended'])
             [:dec].tolist())
    t2 = set(np.argsort(-scores['bacteriology_pos']['v2_extended'])
             [:dec].tolist())
    overlap = len(t1 & t2) / dec

    def _delta(arm, head, key):
        return (results['v2_extended'][head][key]
                - results['v1_baseline'][head][key])

    # round-12 P1-2：v2 双头模型落盘为产品工件（encoder + 职业
    # 词表 + 两头 LGBM），供 DualHeadPredictor MODEL_VERSION='v2'
    model_info_v2 = {'saved': False}
    if HAS_JOBLIB and v2_enc is not None:
        model_dir = os.path.join(BASE, 'tb_risk', 'models',
                                 'sinan_dual_head')
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(v2_enc, os.path.join(model_dir, 'encoder_v2.joblib'))
        joblib.dump(v2_fitted['death_allcause'],
                    os.path.join(model_dir, 'death_head_v2.joblib'))
        joblib.dump(v2_fitted['bacteriology_pos'],
                    os.path.join(model_dir, 'bact_head_v2.joblib'))
        model_info_v2 = {
            'saved': True,
            'model_dir': 'tb_risk/models/sinan_dual_head',
            'files': ['encoder_v2.joblib', 'death_head_v2.joblib',
                      'bact_head_v2.joblib'],
            'version': 'v2',
            'loader': 'tb_risk.core.dual_head_predictor'
                      '.DualHeadPredictor',
            'feature_schema': {
                'numeric': ['AGE_YEARS', 'NU_CONTATO'],
                'binary': ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE',
                           'AGRAVDOENC', 'AGRAVDROGA', 'AGRAVTABAC',
                           'POP_LIBER', 'POP_RUA', 'POP_SAUDE',
                           'POP_IMIG', 'BENEF_GOV'],
                'categorical': ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N',
                                'FORMA', 'EXTRAPU1_N', 'RAIOX_TORA',
                                'TESTE_TUBE', 'CS_GESTANT', 'HIV',
                                'OCC_TRUNC'],
                'raw_input_extra': ['ID_OCUPA_N'],
                'excluded_leakage': ['BACILOSC_E', 'CULTURA_ES',
                                     'TEST_MOLEC', 'HISTOPATOL'],
                'occupation_top_k': OCC_TOP_K,
                'occupation_top': v2_occ_top,
            },
        }
        print('v2 双头模型落盘:', model_info_v2['model_dir'])

    res = {
        'date': '2026-08-26',
        'experiment': 'sinan_bact_head_v2',
        'question': '症状列恢复可行性裁定 + HIV/职业扩展特征能否推高'
                    '细菌学头（基率 57% 的排序天花板）',
        'symptom_feasibility': {
            'verdict': 'INFEASIBLE',
            'evidence': 'TUBEBR19.dbc 94 字段全集实锤：无 TOSSE/FEBRE/'
                        'SINTOM/PERDA_PESO 等症状列——SINAN 全国合并库'
                        '不含症状块（症状表只在地方级个案库）；'
                        'parquet 44 列是 process_sinan.py 显式筛选，'
                        '但即使全集读入也无症状列可恢复',
            'implication': '症状信息缺位是 TUBEBR 数据侧边界；细菌学头'
                           '维持「分诊工具，非诊断」定位（基率 57% 决定'
                           '排序天花板，AUPRC 报告为准）',
        },
        'new_features': {
            'HIV': 'HIV 检测结果（1=阳/2=阴/3=未做/4=等待）——与 '
                   'AGRAVAIDS（AIDS 合并症）不同源的诊断时点特征',
            'ID_OCUPA_N': '职业编码（CBO，top-%d + OTHER）' % OCC_TOP_K,
            'excluded_leakage': ['BACILOSC_1-6', 'BACILOS_E2',
                                 'CULTURA_OU',
                                 '随访涂片/培养结果——结局期数据'],
        },
        'split': {'train': 'FILE_YEAR<=12', 'far_test': '18-19',
                  'n_test': int(te.sum())},
        'results': results,
        'v2_gain': {
            'death_auroc': _delta(None, 'death_allcause', 'auroc'),
            'bact_auroc': _delta(None, 'bacteriology_pos', 'auroc'),
            'bact_auprc': _delta(None, 'bacteriology_pos', 'auprc'),
            'bact_yield10': _delta(None, 'bacteriology_pos', 'yield10'),
        },
        'top_decile_overlap_v2': round(overlap, 4),
        'model_artifacts_v2': model_info_v2,
        'note': 'v1_baseline 为零改动重跑（read_dbc 重解析，与 '
                'sinan_bacteriology_head_20260826.json 数值应确定性'
                '一致——列子集解析等价）',
        'runtime_s': round(time.time() - t0, 1),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('v2 增益: 死亡 AUROC %+.4f | 细菌学 AUROC %+.4f | '
          'AUPRC %+.4f | yield@10%% %+.4f'
          % (res['v2_gain']['death_auroc'],
             res['v2_gain']['bact_auroc'],
             res['v2_gain']['bact_auprc'],
             res['v2_gain']['bact_yield10']))
    print('saved:', OUT)


if __name__ == '__main__':
    main()
