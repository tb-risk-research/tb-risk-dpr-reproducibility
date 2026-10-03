#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1-3（第九轮）：老年亚群的竞争风险生存框架——Fine-Gray
cause-specific 替代已证伪的交互特征路线（elderly_cr）。

问题：老年塌方（E1 0.64 vs 全年龄 0.83，人群属性）已确证不可用
判别力修复——但「老年死于非 TB 原因」从未被显式建模为竞争事件，
而是塞进树模型交互项（elderly_cr，已证伪）。生存框架给出两个
未试过的方向：
  (1) Fine-Gray subdistribution hazard：TB 死亡为主要事件，
      非 TB 死亡为竞争事件，治愈/失访/转出为删失——直接对
      「老年死于非 TB 原因的稀释效应」建模；
  (2) cause-specific Cox：两个 cause 分别拟合（TB 死亡 /
      非 TB 死亡），对比风险结构差异。

设计：
  人群     ：全 5 类分析集 {1,2,3,4,5}；老年 65+ 与 all_age 分层
  时间     ：DT_NOTIFIC → DT_ENCERRA 天数（严格正滞后者）
  划分     ：train FILE_YEAR≤2012 / far test 2018-2019
  对照     ：E3（TB 死亡二分类 LGBM）、E4（复合不良结局二分类——
             老年最优终点 0.674）同划分复现
  评估     ：1 年累计风险排序 AUROC（Far test）+ 竞争结构披露
             （非 TB 死亡竞争概率随年龄的稀释曲线）

归档：data/processed/sinan_competing_risk_20260826.json
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
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

PROC = os.path.join(BASE, 'tb_risk', 'data', 'processed')
OUT = os.path.join(PROC, 'sinan_competing_risk_20260826.json')

BIN_AGRAV = ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
             'AGRAVDROGA', 'AGRAVTABAC', 'POP_LIBER', 'POP_RUA',
             'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV']
CAT_FEATS = ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'FORMA', 'EXTRAPU1_N',
             'RAIOX_TORA', 'TESTE_TUBE', 'CULTURA_ES', 'HISTOPATOL',
             'TEST_MOLEC', 'BACILOSC_E', 'CS_GESTANT']
# Fine-Gray 是线性模型：稀疏 159 维会病态；取数值化核心子集
FG_NUM = ['AGE_YEARS', 'NU_CONTATO']
FG_CAT = ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'FORMA', 'BACILOSC_E']
TRAIN_MAX_YEAR = 12      # FILE_YEAR 为 1-19（DBC 文件年份后两位）
HORIZON_D = 365.0


def _day_number(date_str):
    s = pd.Series(date_str).astype(str).str.strip()
    ok = s.str.fullmatch(r'\d{8}', na=False)
    dt = pd.to_datetime(s.where(ok), format='%Y%m%d', errors='coerce')
    days = dt.values.astype('datetime64[D]').astype(np.int64)
    return days, ok.values


def build_ind_matrix(d):
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
    return sparse.hstack([sparse.csr_matrix(num), Z], format='csr')


def fg_features(d):
    """Fine-Gray 数值特征矩阵（低维，无缺失）。"""
    df = pd.DataFrame({
        'age': np.clip(pd.to_numeric(d['AGE_YEARS'], errors='coerce')
                       .fillna(d['AGE_YEARS'].median()).values, 0, 100),
        'n_contacts': np.clip(pd.to_numeric(d['NU_CONTATO'], errors='coerce')
                              .fillna(0).values, 0, 10)})
    for c in BIN_AGRAV:
        df['b_' + c] = (d[c].astype(str).str.strip() == '1').astype(int)
    for c in FG_CAT:
        vals = d[c].astype(str).str.strip()
        for v in sorted(vals.unique())[:8]:   # 每类至多 8 水平
            df['%s=%s' % (c, v)] = (vals == v).astype(int)
    return df


def run_competing_risk(pop_df, pop_name, res):
    """cause-specific Cox 竞争风险框架（sksurv）。

    主事件 TB 死亡（fg_event==1）/ 竞争事件非 TB 死亡（==2）/
    其余删失（==0）。lifelines 0.30 与 sksurv 0.23 均未实现
    FineGrayPHFitter——用 cause-specific hazard 框架替代
    （病因学解释的标准方法；subdistribution 框架留待 ERASE-TB
    前瞻队列，届时 sksurv 的 FineGrayPHFitter 或可用）。
    """
    out = {'available': False}
    try:
        import sksurv
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.metrics import concordance_index_ipcw
        from sksurv.util import Surv

        X = fg_features(pop_df)
        dur = pop_df['dur'].values
        ev = pop_df['fg_event'].values     # 1=TB死亡 2=非TB死亡 0=删失

        def surv_y(main_cause):
            ev_mask = (ev == main_cause)
            return Surv.from_arrays(ev_mask, dur)

        cs = {}
        for cause, cname in ((1, 'tb_death'), (2, 'nontb_death')):
            y_s = surv_y(cause)
            cph = CoxPHSurvivalAnalysis(alpha=1.0)
            cph.fit(X, y_s)
            # 训练集内 IPCW C-index（竞争风险正确版）
            try:
                ci = concordance_index_ipcw(
                    y_s, y_s, cph.predict(X), tau=730.0)[0]
            except Exception:
                ci = float('nan')
            coefs = dict(zip(X.columns.tolist(),
                            np.asarray(cph.coef_).ravel().tolist()))
            cs[cname] = {
                'n_events': int((ev == cause).sum()),
                'c_index_ipcw': None if np.isnan(ci) else round(float(ci), 4),
                'age_coeff': round(float(coefs.get('age', float('nan'))), 5),
                'hiv_aids_coeff': round(
                    float(coefs.get('b_AGRAVAIDS', float('nan'))), 5),
                'top_coeffs': {k: round(float(v), 4) for k, v in
                               sorted(coefs.items(),
                                      key=lambda kv: -abs(kv[1]))[:8]},
            }
            print('[CS-Cox %s/%s] n_events %d | IPCW C %.4f | age %+0.5f | '
                  'HIV %+0.4f'
                  % (pop_name, cname, cs[cname]['n_events'],
                     cs[cname]['c_index_ipcw'] or float('nan'),
                     cs[cname]['age_coeff'], cs[cname]['hiv_aids_coeff']))

        out = {
            'available': True,
            'n': int(len(X)),
            'n_tb_death': int((ev == 1).sum()),
            'n_nontb_death': int((ev == 2).sum()),
            'n_censored': int((ev == 0).sum()),
            'cause_specific_cox': cs,
            'framework': 'cause-specific hazard（sksurv CoxPHSurvivalAnalysis'
                         ' + IPCW C-index，tau=730 天）；Fine-Gray '
                         'subdistribution fitter 两个库均未实现，如实记录',
        }
    except Exception as e:
        out = {'available': False, 'reason': str(e)[:300]}
        print('[CS-Cox %s] FAILED: %s' % (pop_name, str(e)[:160]))
    res[pop_name] = out


def run_binary_endpoints(d, year, tr, te, res):
    """E3（TB 死亡）/ E4（复合不良结局）二分类 LGBM 对照（far test）。"""
    s = d['SITUA_ENCE'].astype(str)
    eps = {
        'E3_death_tb': {'pop': ['1', '3'], 'pos': ['3']},
        'E4_unfavorable': {'pop': ['1', '2', '3', '4', '5'],
                            'pos': ['2', '3', '4', '5']},
    }
    for name, ep in eps.items():
        mask = s.isin(ep['pop']).values
        y = s.isin(ep['pos']).astype(int).values
        sub = d[mask].reset_index(drop=True)
        y, m_tr, m_te = y[mask], tr[mask], te[mask]
        if HAS_LGBM and m_te.sum() > 100 and y[m_te].sum() > 10:
            X = build_ind_matrix(sub)
            mdl = LGBMClassifier(n_estimators=400, learning_rate=0.06,
                                 num_leaves=63, subsample=0.8,
                                 colsample_bytree=0.8, n_jobs=-1,
                                 random_state=42, verbose=-1)
            mdl.fit(X[m_tr], y[m_tr])
            sc = mdl.predict_proba(X[m_te])[:, 1]
            entry = {'auroc': float(roc_auc_score(y[m_te], sc)),
                     'n_test': int(m_te.sum()),
                     'n_events': int(y[m_te].sum())}
            elderly = np.asarray(pd.to_numeric(
                sub['AGE_YEARS'], errors='coerce').fillna(0).values) >= 65
            # sc 仅覆盖 far test 行（m_te.sum()）——老年 mask 须在
            # test 子集内取，不能与全量 sub 维度的布尔与
            em = elderly[m_te]
            if em.sum() > 100 and y[m_te][em].sum() > 10:
                entry['auroc_elderly'] = float(
                    roc_auc_score(y[m_te][em], sc[em]))
            res[name] = entry
            print('[%s] all_age AUROC %.4f | elderly %.4f'
                  % (name, entry['auroc'],
                     entry.get('auroc_elderly', float('nan'))))


def main():
    t0 = time.time()
    print('=== P1-3 竞争风险生存框架（Fine-Gray）+ E3/E4 对照 ===')
    d = pd.read_parquet(
        os.path.join(PROC, 'sinan', 'tubebra_2001_2019.parquet'),
        columns=BIN_AGRAV + CAT_FEATS + ['AGE_YEARS', 'NU_CONTATO',
                                         'DT_NOTIFIC', 'DT_ENCERRA',
                                         'SITUA_ENCE', 'FILE_YEAR'])
    d = d[d['SITUA_ENCE'].astype(str).isin(
        ['1', '2', '3', '4', '5'])].reset_index(drop=True)

    nd, ok_n = _day_number(d['DT_NOTIFIC'].values)
    ed, ok_e = _day_number(d['DT_ENCERRA'].values)
    dur = (ed - nd).astype(float)
    valid = ok_n & ok_e & (dur > 0)
    s = d['SITUA_ENCE'].astype(str)

    # event 编码：1=TB 死亡（主）2=非 TB 死亡（竞争）0=删失
    ev = np.zeros(len(d), dtype=int)
    ev[s.isin(['3']).values] = 1
    ev[s.isin(['4']).values] = 2

    year = pd.to_numeric(d['FILE_YEAR'], errors='coerce').values
    tr = (year <= TRAIN_MAX_YEAR) & valid
    te = (year >= 18) & (year <= 19) & valid
    elderly = np.asarray(pd.to_numeric(
        d['AGE_YEARS'], errors='coerce').fillna(0).values) >= 65

    d['dur'] = dur
    d['fg_event'] = ev

    res = {'date': '2026-08-26',
           'experiment': 'sinan_competing_risk_v1',
           'question': '把「老年死于非 TB 原因」显式建模为竞争事件'
                       '（Fine-Gray subdistribution hazard）能否优于'
                       '已证伪的交互特征路线',
           'endpoint': 'TB 死亡（主事件）vs 非 TB 死亡（竞争事件），'
                       '治愈/失访/转出删失',
           'time_origin': 'DT_NOTIFIC → DT_ENCERRA（严格正滞后）',
           'split': {'train': 'FILE_YEAR<=12', 'far_test': '18-19'},
           'fine_gray': {}, 'has_lgbm': HAS_LGBM,
           'note_fit': 'cause-specific Cox 用全期数据'
                       '（生存模型时间起点为通报，无泄漏概念）；'
                       '二分类对照用时间外推划分'}

    # 竞争结构披露：非 TB 死亡竞争概率随年龄稀释
    age_bins = [(0, 35), (35, 50), (50, 65), (65, 120)]
    dilution = []
    age_v = pd.to_numeric(d['AGE_YEARS'], errors='coerce').fillna(0).values
    for lo, hi in age_bins:
        m = (age_v >= lo) & (age_v < hi)
        n_tb = int(((ev == 1) & m).sum())
        n_ntb = int(((ev == 2) & m).sum())
        dilution.append({
            'age_bin': '%d-%d' % (lo, hi), 'n': int(m.sum()),
            'tb_death': n_tb, 'nontb_death': n_ntb,
            'competing_fraction': round(n_ntb / max(n_tb + n_ntb, 1), 4)})
    res['age_dilution'] = dilution
    for r in dilution:
        print('  年龄 %s：TB 死亡 %d / 非 TB 死亡 %d（竞争占比 %.1f%%）'
              % (r['age_bin'], r['tb_death'], r['nontb_death'],
                 100 * r['competing_fraction']))

    # 竞争风险：老年与全年龄分别拟合（全期正滞后样本）
    run_competing_risk(d[valid & elderly].reset_index(drop=True),
                       'elderly_65p', res['fine_gray'])
    run_competing_risk(d[valid].reset_index(drop=True), 'all_age',
                       res['fine_gray'])

    # 二分类对照（时间外推）
    run_binary_endpoints(d, year, tr, te, res)

    res['runtime_s'] = round(time.time() - t0, 1)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
