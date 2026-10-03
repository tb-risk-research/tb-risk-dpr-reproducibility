#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-2（第九轮）：SINAN 部署度量切换——AUROC → yield@k / NNS / PPV@预算。

问题：模型不动，把评估目标从判别力（AUROC）切换到疾控真实决策语言
（筛查前 k% 能捞到多少病人、每确诊一例要筛多少人）。ml/planner.py 已
有 10%-100% 筛查预算的动作骨架，本实验把评估指标接上去。

设计：
  数据     ：SINAN 全因死亡修正终点（SITUA_ENCE∈{3,4}，人群 {1,3,4}）
  划分     ：train = FILE_YEAR≤2012 / far test = 2018-2019（T1 时间外推
             同款——部署度量的可信前提是模型未见过未来）
  两臂     ：ind（159 维个体基线）/ ind_all（+市级时序先证块）
             —— 直接回答「加先证后检测效果优化多少」
  模型     ：LGBM（单模型，时间外推单折；部署场景不用 CV）
  指标     ：yield@k / NNS@k / PPV@k / lift（k=1,2,5,10,20,30,50,100%）
             + AUROC/AUPRC 顺带 + 随机基线（yield@k=k%）
  分层     ：all_age 与 elderly_65p（老年塌方在部署语言下的真实代价）
  纯函数   ：tb_risk.ml.planner.screening_metrics_at_budgets（GUI 共用）

round-11 P1-1 追加：老年双终点报告——全因死亡（预后分层语义）
与 E3 TB 死亡（传染控制语义）并列。老年全因死亡 63% 为非 TB
（round-9 竞争风险归档），单看全因会把模型无法干预的死亡计入
NNS 效率；E3 臂给传染控制语义下的真实成绩。

归档：data/processed/sinan_deployment_metrics_20260826.json
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
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from tb_risk.ml.planner import screening_metrics_at_budgets  # noqa: E402

PROC = os.path.join(BASE, 'tb_risk', 'data', 'processed')
OUT = os.path.join(PROC, 'sinan_deployment_metrics_20260826.json')

BIN_AGRAV = ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
             'AGRAVDROGA', 'AGRAVTABAC', 'POP_LIBER', 'POP_RUA',
             'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV']
CAT_FEATS = ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'FORMA', 'EXTRAPU1_N',
             'RAIOX_TORA', 'TESTE_TUBE', 'CULTURA_ES', 'HISTOPATOL',
             'TEST_MOLEC', 'BACILOSC_E', 'CS_GESTANT']
OUTCOME_BLOCK = ['prior_n', 'prior_death', 'prior_rate']
DENSITY_BLOCK = ['notif_1y', 'log_notif_1y']
K_SHRINK = 2.0
MISSING_ENCERRA_SENTINEL = 10 ** 9
TRAIN_MAX_YEAR = 12      # FILE_YEAR 为 1-19（DBC 文件年份后两位）
FAR_MIN_YEAR, FAR_MAX_YEAR = 18, 19
BUDGETS = [1, 2, 5, 10, 20, 30, 50, 100]


def _day_number(date_str):
    """YYYYMMDD 字符串 → 天数整数（datetime64[D] 基准）。"""
    s = pd.Series(date_str).astype(str).str.strip()
    ok = s.str.fullmatch(r'\d{8}', na=False)
    dt = pd.to_datetime(s.where(ok), format='%Y%m%d', errors='coerce')
    days = dt.values.astype('datetime64[D]').astype(np.int64)
    return days, ok.values


def build_prior_features(mun, notif_day, enc_day, yv, base_rate):
    """市级时序先证特征（部署合法版本，同 run_sinan_temporal_prior v2）。"""
    n = len(yv)
    prior_n = np.zeros(n, dtype=np.int64)
    prior_death = np.zeros(n, dtype=np.int64)
    notif_1y = np.zeros(n, dtype=np.int64)

    um, gidx = np.unique(mun, return_inverse=True)
    order = np.argsort(gidx, kind='stable')
    bounds = np.searchsorted(gidx[order], np.arange(len(um) + 1))

    for g in range(len(um)):
        rows = order[bounds[g]:bounds[g + 1]]
        t = notif_day[rows]
        o = np.argsort(t, kind='stable')
        rows, t = rows[o], t[o]
        hi = np.searchsorted(t, t, side='right')
        lo = np.searchsorted(t, t - 365, side='left')
        notif_1y[rows] = hi - lo - 1
        e = enc_day[rows]
        yd = yv[rows]
        v = (e < MISSING_ENCERRA_SENTINEL // 2) & (e > t)
        ev, ydv = e[v], yd[v]
        if len(ev):
            oe = np.argsort(ev, kind='stable')
            ev, ydv = ev[oe], ydv[oe]
            pos = np.searchsorted(ev, t, side='right')
            prior_n[rows] = pos
            cd = np.concatenate(([0], np.cumsum(ydv)))
            prior_death[rows] = cd[pos]

    out = pd.DataFrame({'prior_n': prior_n, 'prior_death': prior_death,
                        'notif_1y': notif_1y})
    out['prior_rate'] = ((out['prior_death'] + K_SHRINK * base_rate)
                          / (out['prior_n'] + K_SHRINK))
    out['log_notif_1y'] = np.log1p(out['notif_1y'].values)
    return out


def build_ind_matrix(d):
    """个体基线特征（159 维，同规模训练实验口径）。"""
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


def main():
    t0 = time.time()
    print('=== P0-2 部署度量切换：yield@k / NNS / PPV@预算（时间外推 far test）===')

    d = pd.read_parquet(
        os.path.join(PROC, 'sinan', 'tubebra_2001_2019.parquet'),
        columns=BIN_AGRAV + CAT_FEATS + ['AGE_YEARS', 'NU_CONTATO',
                                         'ID_MUNICIP', 'DT_NOTIFIC',
                                         'DT_ENCERRA', 'SITUA_ENCE',
                                         'FILE_YEAR'])
    d = d[d['SITUA_ENCE'].astype(str).isin(['1', '3', '4'])].copy()
    s = d['SITUA_ENCE'].astype(str)
    y = s.isin(['3', '4']).astype(int).values      # 全因死亡（预后分层）
    y_e3 = (s == '3').astype(int).values           # E3 TB 死亡（传染控制）
    base_rate = float(y.mean())
    print('analysis set %d, all-cause death %.2f%%' % (len(d), 100 * base_rate))

    # 先证特征（全量构造）
    nd, ok_n = _day_number(d['DT_NOTIFIC'].values)
    ed, ok_e = _day_number(d['DT_ENCERRA'].values)
    ed[~ok_e] = MISSING_ENCERRA_SENTINEL
    mun = d['ID_MUNICIP'].astype(str).values
    pf = build_prior_features(mun, nd, ed, y, base_rate)

    X = build_ind_matrix(d)
    P = sparse.csr_matrix(pf[OUTCOME_BLOCK + DENSITY_BLOCK]
                          .astype(np.float32).values)
    year = pd.to_numeric(d['FILE_YEAR'], errors='coerce').values

    tr = (year <= TRAIN_MAX_YEAR) & ok_n
    te = (year >= FAR_MIN_YEAR) & (year <= FAR_MAX_YEAR)
    print('train %d (%.1f%% events) / far test %d (%.1f%% events)'
          % (tr.sum(), 100 * y[tr].mean(), te.sum(), 100 * y[te].mean()))

    elderly = np.asarray(
        pd.to_numeric(d['AGE_YEARS'], errors='coerce').fillna(0).values) >= 65

    arms = {'ind': X, 'ind_all': sparse.hstack([X, P], format='csr')}
    res = {'date': '2026-08-26',
           'experiment': 'sinan_deployment_metrics_v1',
           'question': '模型判别力在疾控预算语言下值多少——'
                       '筛查前 k% 捕获多少死亡、每例需筛多少人',
           'endpoint': '全因死亡 SITUA_ENCE∈{3,4}（round-7 修正终点）',
           'split': {'train': 'FILE_YEAR<=12', 'far_test': '18-19',
                     'n_train': int(tr.sum()), 'n_test': int(te.sum()),
                     'test_event_rate': float(y[te].mean())},
           'budgets_pct': BUDGETS,
           'random_baseline_note': '随机筛查 yield@k = k%（全部 lift 的分母）',
           'arms': {}, 'elderly_65p': {}, 'has_lgbm': HAS_LGBM}

    for name, Xa in arms.items():
        mdl = LGBMClassifier(n_estimators=400, learning_rate=0.06,
                             num_leaves=63, subsample=0.8,
                             colsample_bytree=0.8, n_jobs=-1,
                             random_state=42, verbose=-1)
        mdl.fit(Xa[tr], y[tr])
        sc = mdl.predict_proba(Xa[te])[:, 1]
        entry = {
            'auroc': float(roc_auc_score(y[te], sc)),
            'auprc': float(average_precision_score(y[te], sc)),
            'metrics': {str(k): v for k, v in
                        screening_metrics_at_budgets(y[te], sc,
                                                     BUDGETS).items()},
        }
        res['arms'][name] = entry
        m = entry['metrics']
        print('[%s] AUROC %.4f | yield@10%% %.3f (lift %.1fx) | '
              'yield@30%% %.3f | NNS@10%% %.1f'
              % (name, entry['auroc'], m['10']['yield'], m['10']['lift'],
                 m['30']['yield'], m['10']['nns']))

    # 老年分层（用 ind_all 臂——部署最优配置）
    mdl = LGBMClassifier(n_estimators=400, learning_rate=0.06,
                         num_leaves=63, subsample=0.8,
                         colsample_bytree=0.8, n_jobs=-1,
                         random_state=42, verbose=-1)
    Xall = arms['ind_all']
    mdl.fit(Xall[tr], y[tr])
    sc = mdl.predict_proba(Xall[te])[:, 1]
    for gname, mask in [('elderly_65p', elderly[te]),
                        ('adult_lt65', ~elderly[te])]:
        if mask.sum() < 50 or y[te][mask].sum() < 10:
            continue
        sc_g, y_g = sc[mask], y[te][mask]
        n_events_g = int(y_g.sum())
        # 分龄阈值策略（round-10 P3）：组内 top-k% 的分数阈值 +
        # 预期检出/漏检——「预算 X 时该筛哪些人」的操作答案
        thresholds = {}
        for k in BUDGETS:
            m_size = max(1, int(round(len(sc_g) * k / 100.0)))
            thr = float(np.sort(sc_g)[::-1][min(m_size - 1,
                                                len(sc_g) - 1)])
            caught = int((sc_g >= thr).sum() and
                         int(y_g[sc_g >= thr].sum()))
            thresholds[str(k)] = {
                'threshold': round(thr, 4),
                'screen_n': m_size,
                'caught': caught,
                'missed': n_events_g - caught,
                'yield': round(caught / n_events_g, 4),
            }
        res['elderly_65p'][gname] = {
            'n': int(mask.sum()),
            'event_rate': float(y_g.mean()),
            'auroc': float(roc_auc_score(y_g, sc_g)),
            'metrics': {str(k): v for k, v in
                        screening_metrics_at_budgets(y_g, sc_g,
                                                      BUDGETS).items()},
            'age_thresholds': thresholds,
        }
        m = res['elderly_65p'][gname]['metrics']
        t10 = thresholds['10']
        print('[%s] n=%d rate %.3f | AUROC %.4f | yield@10%% %.3f | '
              'thr@10%% %.3f (检出 %d / 漏检 %d)'
              % (gname, mask.sum(), y_g.mean(),
                 res['elderly_65p'][gname]['auroc'], m['10']['yield'],
                 t10['threshold'], t10['caught'], t10['missed']))

    # ---- round-11 P1-1：老年双终点报告（全因 vs E3 TB 死亡并列）----
    # 老年全因死亡 63% 为非 TB（round-9 竞争风险归档），全因 NNS
    # 效率有相当部分来自模型无法干预的死亡；E3 是传染控制语义的
    # 真终点。同款 ind_all 特征 + 同款划分，仅换标签。
    mdl_e3 = LGBMClassifier(n_estimators=400, learning_rate=0.06,
                            num_leaves=63, subsample=0.8,
                            colsample_bytree=0.8, n_jobs=-1,
                            random_state=42, verbose=-1)
    mdl_e3.fit(Xall[tr], y_e3[tr])
    sc_e3 = mdl_e3.predict_proba(Xall[te])[:, 1]
    y_e3_te = y_e3[te]

    def _subgroup(y_g, sc_g):
        """组内部署度量 + 分龄阈值（与全因老年分层同口径）。"""
        n_events_g = int(y_g.sum())
        thresholds = {}
        for k in BUDGETS:
            m_size = max(1, int(round(len(sc_g) * k / 100.0)))
            thr = float(np.sort(sc_g)[::-1][min(m_size - 1,
                                                len(sc_g) - 1)])
            caught = int(y_g[sc_g >= thr].sum())
            thresholds[str(k)] = {
                'threshold': round(thr, 4),
                'screen_n': m_size,
                'caught': caught,
                'missed': n_events_g - caught,
                'yield': round(caught / max(n_events_g, 1), 4),
            }
        return {
            'n': int(len(y_g)),
            'event_rate': float(y_g.mean()),
            'n_events': n_events_g,
            'auroc': float(roc_auc_score(y_g, sc_g)),
            'metrics': {str(k): v for k, v in
                        screening_metrics_at_budgets(y_g, sc_g,
                                                     BUDGETS).items()},
            'age_thresholds': thresholds,
        }

    e3_subgroups = {}
    for gname, gmask in [('elderly_65p', elderly[te]),
                         ('adult_lt65', ~elderly[te])]:
        if gmask.sum() < 50 or y_e3_te[gmask].sum() < 10:
            e3_subgroups[gname] = {'reason': 'insufficient events'}
            continue
        e3_subgroups[gname] = _subgroup(y_e3_te[gmask], sc_e3[gmask])
        g = e3_subgroups[gname]
        print('[E3/%s] n=%d rate %.3f | AUROC %.4f | yield@10%% %.3f | '
              'thr@10%% %.3f (检出 %d / 漏检 %d)'
              % (gname, g['n'], g['event_rate'], g['auroc'],
                 g['metrics']['10']['yield'],
                 g['age_thresholds']['10']['threshold'],
                 g['age_thresholds']['10']['caught'],
                 g['age_thresholds']['10']['missed']))

    res['elderly_dual_endpoint'] = {
        'endpoint_semantics': {
            'allcause_death': 'SITUA_ENCE∈{3,4}——预后分层语义（含非 TB 死亡）',
            'e3_tb_death': "SITUA_ENCE=='3'——传染控制语义（TB 死亡）",
        },
        'features': 'ind_all（个体基线 + 市级时序先证）',
        'split': 'train FILE_YEAR<=12 / far test 18-19（同主实验）',
        'elderly_non_tb_share': 0.63,
        'elderly_non_tb_source': 'sinan_competing_risk_20260826.json '
                                 '(cause_specific_cox + age_dilution)',
        'arms': {
            'allcause_death': {
                'overall_auroc': res['arms']['ind_all']['auroc'],
                'subgroups': res['elderly_65p'],
            },
            'e3_tb_death': {
                'overall_auroc': float(roc_auc_score(y_e3_te, sc_e3)),
                'subgroups': e3_subgroups,
            },
        },
        'paper_note': '老年全因死亡 63% 为非 TB；老年组高 NNS 效率有'
                      '相当部分来自模型无法干预的死亡（HIV 系数在老年段'
                      '符号反转：TB 死亡 −0.247 / 非 TB 死亡 +0.974）。'
                      '双终点并列报告是纠偏最直接的方式；若目标是 TB '
                      '死亡，应以 E3 臂成绩为准。',
    }
    print('[E3/overall] AUROC %.4f (vs 全因 %.4f)'
          % (res['elderly_dual_endpoint']['arms']['e3_tb_death']
             ['overall_auroc'],
             res['elderly_dual_endpoint']['arms']['allcause_death']
             ['overall_auroc']))

    # 两臂检测效果对比（回答「先证是否优化检测」）
    a, b = res['arms']['ind']['metrics'], res['arms']['ind_all']['metrics']
    res['prior_gain'] = {
        'auroc_delta': res['arms']['ind_all']['auroc']
                       - res['arms']['ind']['auroc'],
        'yield_at': {k: {'ind': a[k]['yield'], 'ind_all': b[k]['yield'],
                         'delta': b[k]['yield'] - a[k]['yield']}
                     for k in ('1', '5', '10', '30')},
        'note': '时间外推 far test（2018-19）单折 LGBM，种子 42；'
                '部署增益以 yield@k 差值报告（CI 见时序先证 v2 归档的'
                '统计检验，AUROC +0.0026 显著）',
    }
    print('prior AUROC delta %+.4f | yield@10%% delta %+.4f'
          % (res['prior_gain']['auroc_delta'],
             res['prior_gain']['yield_at']['10']['delta']))

    res['runtime_s'] = round(time.time() - t0, 1)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
