#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-3（第十二轮）：SINAN 前瞻证据链——walk-forward 月度批次评估。

问题（2026-09-04 证据缺口审查）：部署证据全部为回顾性一次性评估
（train FILE_YEAR≤12 → far test 2018-19 池化单折），缺少「冻结模型
在按时间顺序到达的新批次上」的前瞻表现证据；yield@k 是离线指标，
无操作化定义与真值回链协议，无法证明「兑现」。

设计（与 v1 部署度量完全同口径，保证可比）：
  数据     ：SINAN 全因死亡修正终点（SITUA_ENCE∈{3,4}，人群 {1,3,4}）
  特征臂   ：ind（159 维个体基线）/ ind_all（+市级时序先证块）——
             编码器与先证特征构建单源复用 run_sinan_deployment_metrics
             （严格正滞后 notif_j < enc_j ≤ notif_i 的合法性逻辑不复制）
  模型     ：LGBM（400 树 / lr 0.06 / 63 叶 / 种子 42，与 v1 逐参数一致）
  批次轴   ：DT_NOTIFIC 通知月，2013-01..2019-12 共 84 个月度批次
             （逐日批次单批 ~22 事件无统计意义；月批是「按到达顺序
             滚动」的可行近似）
  两臂     ：
    frozen  冻结臂（影子模式回放 = 严格前瞻主证据）——train
            FILE_YEAR≤12（与 v1 逐字节同过滤），冻结后对 2013-2019
            逐批打分；
    refit   年度刷新臂——打分年 Y 用 FILE_YEAR<Y 且 notif_year<Y
            重训（双重过滤防边界泄漏：FILE_YEAR-13 文件里 2014 年
            才通知的个案不进 2014 年训练集），Y=2013 复用冻结模型。
  每批次   ：yield@k / NNS@10 / PPV@10 / AUROC（Hanley-McNeil CI）/
            Brier / ECE / PSI（vs 2013 基线）/ 预期-实际 z 值
  真值回链 ：realized yield@10 在批次月末 +{90,180,365} 天已结案
            （enc≤审查点 且 enc>notif）子集上计算——影子模式在真实
            审查时点可见的「兑现」口径，并列 closure_rate 量化回流
            选择偏倚（早结案 ≠ 全体）。
  一致性   ：各批 yield@10 vs 离线承诺值（v1 far-test 池化
            ind 0.3855 / ind_all 0.3979），|z|≤1.96 命中率与
            连续 2 批 z<−1.96 告警。

诚实边界：
  - v1 far test 含通知日期不可解析个案，本实验批次宇宙按通知月
    剔除——对账差异由此产生（reconciliation_vs_v1 块并列报告）；
  - refit 臂月初批次存在数周文件可用性边界（FILE_YEAR<Y 的文件在
    Y 年初未必完全到位）；frozen 臂无此问题，是严格前瞻主证据；
  - yield SE 为 n_pos 二项近似（top-k 选择相关性忽略）；
  - PSI 参考分布 = 2013 自然年池化分数（冻结模型口径）；refit 臂
    PSI 同时含模型刷新与人群漂移两种效应；
  - 事件率 8.6%→10.3% 逐年上行本身即人群漂移——校准漂移（ECE）
    与分数漂移（PSI）按批次给出，正是池化评估掩盖的部分；
  - 历史模拟 ≠ 实时影子运行：真实到达流、数据质量与回流延迟
    只能由试点验证（协议见 docs/deployment_ops.md §11）。

归档：data/processed/sinan_walkforward_20260904.json
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
# 先证合法性逻辑单源复用（项目硬约束：严格正滞后 notif_j<enc_j≤notif_i）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy import sparse  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

from run_sinan_deployment_metrics import (  # noqa: E402
    BIN_AGRAV,
    BUDGETS,
    CAT_FEATS,
    DENSITY_BLOCK,
    MISSING_ENCERRA_SENTINEL,
    OUTCOME_BLOCK,
    TRAIN_MAX_YEAR,
    _day_number,
    build_ind_matrix,
    build_prior_features,
)
from tb_risk.ml.planner import screening_metrics_at_budgets  # noqa: E402
from tb_risk.scoring.ml.deployment_metrics import (  # noqa: E402
    calibration_error,
    population_stability_index,
)

PROC = os.path.join(BASE, 'tb_risk', 'data', 'processed')
OUT = os.path.join(PROC, 'sinan_walkforward_20260904.json')

WINDOW_FIRST, WINDOW_LAST = '2013-01', '2019-12'
EVAL_MIN_YEAR, EVAL_MAX_YEAR = 13, 19
HORIZONS = (90, 180, 365)
K_PRIMARY = 10
# v1 far-test 池化承诺值（sinan_deployment_metrics_20260826.json）
EXPECTED_FROM_V1 = {
    'ind': {'yield_at_10': 0.3854977468250717, 'auroc': 0.8135136350328167},
    'ind_all': {'yield_at_10': 0.3978560699167008, 'auroc': 0.8176772770951724},
}
Z_CRIT = 1.96


def _make_model():
    """LGBM 配置与 v1 逐参数一致（可比性前提）。"""
    return LGBMClassifier(n_estimators=400, learning_rate=0.06,
                          num_leaves=63, subsample=0.8,
                          colsample_bytree=0.8, n_jobs=-1,
                          random_state=42, verbose=-1)


def _auroc_hm_ci(y, s, z=Z_CRIT):
    """AUROC + Hanley-McNeil 95% CI（闭式，批次级轻量）。"""
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return None, None
    a = float(roc_auc_score(y, s))
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a)
                  + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    return a, [a - z * se, a + z * se]


def _month_end_day(month):
    """'YYYY-MM' → 月末天数（datetime64[D] 基准，与 _day_number 同纪元）。"""
    y, m = int(month[:4]), int(month[5:7])
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    first_next = np.datetime64('%04d-%02d-01' % (ny, nm), 'D')
    return int(first_next.astype(np.int64)) - 1


def _pooled_record(y_s, sc_s):
    """池化（年度/全期/对账）指标记录。"""
    metrics = screening_metrics_at_budgets(y_s, sc_s, BUDGETS)
    auroc, ci = _auroc_hm_ci(y_s, sc_s)
    cal = calibration_error(y_s, sc_s)
    return {
        'n': int(len(y_s)),
        'n_events': int(y_s.sum()),
        'event_rate': float(y_s.mean()),
        'metrics': {str(k): v for k, v in metrics.items()},
        'auroc': auroc,
        'auroc_ci': ci,
        'brier': cal['brier'],
        'ece': cal['ece'],
    }


def _realized_at(y_b, sc_b, nd_b, ed_b, ok_e_b, review_day, k=K_PRIMARY):
    """审查时点已结案子集上的 realized yield@k（真值回链模拟）。

    eligible = 已结案（enc 有效）且 enc ≤ 审查点 且严格正滞后
    （enc > notif）——影子模式在该审查点可对账的全部个案。
    """
    n = len(y_b)
    elig = ok_e_b & (ed_b <= review_day) & (ed_b > nd_b)
    n_elig = int(elig.sum())
    out = {'n_eligible': n_elig,
           'closure_rate': (n_elig / n) if n else None,
           'yield_at_%d' % k: None}
    if n_elig and y_b[elig].sum() > 0:
        m = screening_metrics_at_budgets(y_b[elig], sc_b[elig], [k])
        out['yield_at_%d' % k] = m[k]['yield']
    return out


def _batch_record(y_b, sc_b, nd_b, ed_b, ok_e_b, review_days, psi_ref):
    """单批次单臂单特征记录。"""
    metrics = screening_metrics_at_budgets(y_b, sc_b, BUDGETS)
    auroc, ci = _auroc_hm_ci(y_b, sc_b)
    cal = calibration_error(y_b, sc_b)
    n_pos = int(y_b.sum())
    rec = {
        'n': int(len(y_b)),
        'n_events': n_pos,
        'event_rate': float(y_b.mean()) if len(y_b) else None,
        'metrics': {str(k): v for k, v in metrics.items()},
        'auroc': auroc,
        'auroc_ci': ci,
        'brier': cal['brier'],
        'ece': cal['ece'],
        'psi': population_stability_index(psi_ref, sc_b),
        'realized': {str(h): _realized_at(y_b, sc_b, nd_b, ed_b, ok_e_b,
                                          review_days[h])
                     for h in HORIZONS},
    }
    return rec


def _closure_stats(nd_b, ed_b, ok_e_b):
    """批次级结案统计（臂无关）：滞后分布 + 各时限结案份额。"""
    pos_lag = ok_e_b & (ed_b > nd_b)
    neg_lag = ok_e_b & (ed_b <= nd_b)
    lag = (ed_b[pos_lag] - nd_b[pos_lag]) if pos_lag.any() \
        else np.array([], dtype=np.int64)
    n = len(nd_b)
    closed_within = {}
    for h in HORIZONS:
        m_h = ok_e_b & (ed_b > nd_b) & (ed_b <= nd_b + h)
        closed_within[str(h)] = (int(m_h.sum()) / n) if n else None
    return {
        'n_valid_enc': int(ok_e_b.sum()),
        'n_negative_lag': int(neg_lag.sum()),
        'median_lag_days': float(np.median(lag)) if len(lag) else None,
        'closed_within_days': closed_within,
    }


def _z_vs_expected(y10, expected, n_pos):
    """批次 yield@10 相对离线承诺值的 z（n_pos 二项近似）。"""
    if n_pos == 0 or y10 is None:
        return None
    se = np.sqrt(expected * (1.0 - expected) / n_pos)
    return (y10 - expected) / se


def _attach_z(batch_recs, expected):
    """按时间序给每批记录附 z 与一致性标记。"""
    for _, rec in batch_recs:
        y10 = rec['metrics'][str(K_PRIMARY)]['yield']
        rec['z_vs_expected'] = _z_vs_expected(y10, expected,
                                              rec['n_events'])
    return batch_recs


def _alert_summary(batch_recs):
    """告警汇总：PSI 两级 / ECE / z 连续越界 / CI 命中率。"""
    months = [m for m, _ in batch_recs]
    psis = [r['psi'] for _, r in batch_recs]
    zs = [r['z_vs_expected'] for _, r in batch_recs]
    eces = [r['ece'] for _, r in batch_recs]
    consecutive = [months[i] for i in range(1, len(zs))
                   if zs[i] is not None and zs[i - 1] is not None
                   and zs[i] < -Z_CRIT and zs[i - 1] < -Z_CRIT]
    z_valid = [z for z in zs if z is not None]
    return {
        'n_batches': len(months),
        'psi_warn_gt_0p10': [m for m, p in zip(months, psis)
                             if p is not None and p > 0.10],
        'psi_drift_gt_0p25': [m for m, p in zip(months, psis)
                              if p is not None and p > 0.25],
        'ece_alert_gt_0p05': [m for m, e in zip(months, eces)
                              if e is not None and e > 0.05],
        'yield_below_expected_2_consecutive': consecutive,
        'within_ci_rate': (float(np.mean([abs(z) <= Z_CRIT
                                          for z in z_valid]))
                           if z_valid else None),
    }


def main():
    t0 = time.time()
    print('=== P0-3 前瞻证据链：walk-forward 月度批次（冻结 + 年度刷新）===')

    d = pd.read_parquet(
        os.path.join(PROC, 'sinan', 'tubebra_2001_2019.parquet'),
        columns=BIN_AGRAV + CAT_FEATS + ['AGE_YEARS', 'NU_CONTATO',
                                         'ID_MUNICIP', 'DT_NOTIFIC',
                                         'DT_ENCERRA', 'SITUA_ENCE',
                                         'FILE_YEAR'])
    d = d[d['SITUA_ENCE'].astype(str).isin(['1', '3', '4'])].copy()
    s = d['SITUA_ENCE'].astype(str)
    y = s.isin(['3', '4']).astype(int).values
    base_rate = float(y.mean())
    print('analysis set %d, all-cause death %.2f%%' % (len(d), 100 * base_rate))

    # ---- 时间轴（与 v1 相同的解析器）----
    nd, ok_n = _day_number(d['DT_NOTIFIC'].values)
    ed, ok_e = _day_number(d['DT_ENCERRA'].values)
    ed[~ok_e] = MISSING_ENCERRA_SENTINEL
    mun = d['ID_MUNICIP'].astype(str).values
    year = pd.to_numeric(d['FILE_YEAR'], errors='coerce').values

    s_str = d['DT_NOTIFIC'].astype(str).str.strip()
    # 8 位日期 'YYYYMMDD' → 'YYYY-MM' 月键（固定宽度，字符串序 = 时间序）
    month_arr = np.where(
        ok_n, (s_str.str[:4] + '-' + s_str.str[4:6]).values, '')
    notif_year = np.full(len(d), -1, dtype=int)
    ny_vals = pd.to_numeric(s_str.str[:4], errors='coerce').values
    notif_year[ok_n] = ny_vals[ok_n].astype(int)

    in_window = ok_n & (month_arr >= WINDOW_FIRST) & (month_arr <= WINDOW_LAST)
    universe = ((year >= EVAL_MIN_YEAR) & (year <= EVAL_MAX_YEAR)
                & in_window)
    n_eval_files = int(((year >= EVAL_MIN_YEAR) & (year <= EVAL_MAX_YEAR)).sum())
    n_out_of_window = int(((year >= EVAL_MIN_YEAR)
                           & (year <= EVAL_MAX_YEAR) & ok_n
                           & ~in_window).sum())
    n_no_notif = int(((year >= EVAL_MIN_YEAR)
                      & (year <= EVAL_MAX_YEAR) & ~ok_n).sum())
    print('batch universe %d / FILE_YEAR 13-19 全量 %d '
          '（窗口外通知日期 %d，不可解析 %d）'
          % (universe.sum(), n_eval_files, n_out_of_window, n_no_notif))

    # ---- 特征（先证合法性单源复用 v1）----
    pf = build_prior_features(mun, nd, ed, y, base_rate)
    X = build_ind_matrix(d)
    P = sparse.csr_matrix(pf[OUTCOME_BLOCK + DENSITY_BLOCK]
                          .astype(np.float32).values)
    arms_X = {'ind': X, 'ind_all': sparse.hstack([X, P], format='csr')}

    trA = (year <= TRAIN_MAX_YEAR) & ok_n      # 与 v1 逐字节同过滤
    print('frozen train %d（FILE_YEAR<=12 & ok_notif）' % trA.sum())

    u_idx = np.flatnonzero(universe)
    y_u, nd_u, ed_u, ok_e_u = y[u_idx], nd[u_idx], ed[u_idx], ok_e[u_idx]
    month_u = month_arr[u_idx]
    nyear_u = notif_year[u_idx]
    months = sorted(set(month_u.tolist()))
    print('batches %d（%s .. %s）' % (len(months), months[0], months[-1]))

    # ---- 两臂打分（notif_year 用全年纪元；FILE_YEAR 码 = 年份−2000）----
    scores = {}
    psi_ref = {}
    for feat, Xa in arms_X.items():
        t1 = time.time()
        mA = _make_model()
        mA.fit(Xa[trA], y[trA])
        Xu = Xa[u_idx]
        sc_frozen = mA.predict_proba(Xu)[:, 1].astype(np.float64)
        scores[('frozen', feat)] = sc_frozen
        psi_ref[feat] = sc_frozen[nyear_u == 2013]  # 2013 基线（冻结口径）
        print('[%s] frozen model trained (%.1fs)' % (feat, time.time() - t1))

        sc_refit = np.empty(len(u_idx), dtype=np.float64)
        sc_refit[nyear_u == 2013] = sc_frozen[nyear_u == 2013]
        for Y in range(2014, 2020):
            t1 = time.time()
            trB = (year < Y - 2000) & ok_n & (notif_year < Y)
            mB = _make_model()
            mB.fit(Xa[trB], y[trB])
            mU = nyear_u == Y
            sc_refit[mU] = mB.predict_proba(Xu[mU])[:, 1]
            print('[%s] refit year %d: train %d, score %d (%.1fs)'
                  % (feat, Y, trB.sum(), mU.sum(), time.time() - t1))
        scores[('refit', feat)] = sc_refit

    # ---- 批次级记录 ----
    month_review = {m: {h: _month_end_day(m) + h for h in HORIZONS}
                    for m in months}
    batches = {}
    arm_batches = {(a, f): [] for a in ('frozen', 'refit')
                   for f in arms_X}
    for month in months:
        sel = month_u == month
        y_b, nd_b, ed_b, ok_e_b = y_u[sel], nd_u[sel], ed_u[sel], ok_e_u[sel]
        rec_m = {'closure': _closure_stats(nd_b, ed_b, ok_e_b)}
        for arm in ('frozen', 'refit'):
            for feat in arms_X:
                sc_b = scores[(arm, feat)][sel]
                rec = _batch_record(y_b, sc_b, nd_b, ed_b, ok_e_b,
                                    month_review[month], psi_ref[feat])
                rec_m.setdefault(arm, {})[feat] = rec
                arm_batches[(arm, feat)].append((month, rec))
        batches[month] = rec_m

    for (arm, feat), recs in arm_batches.items():
        _attach_z(recs, EXPECTED_FROM_V1[feat]['yield_at_10'])

    # ---- 年度 / 全期池化 ----
    yearly, overall = {}, {}
    for arm in ('frozen', 'refit'):
        for feat in arms_X:
            sc = scores[(arm, feat)]
            yearly[(arm, feat)] = {
                str(Y): _pooled_record(y_u[nyear_u == Y],
                                       sc[nyear_u == Y])
                for Y in range(2013, 2020)}
            overall[(arm, feat)] = _pooled_record(y_u, sc)

    # ---- staleness：刷新买回多少（同notif年两臂池化 yield@10 差）----
    staleness = {}
    for feat in arms_X:
        staleness[feat] = {
            str(Y): {
                'frozen_y10': yearly[('frozen', feat)][str(Y)]
                ['metrics'][str(K_PRIMARY)]['yield'],
                'refit_y10': yearly[('refit', feat)][str(Y)]
                ['metrics'][str(K_PRIMARY)]['yield'],
                'delta': (yearly[('refit', feat)][str(Y)]
                          ['metrics'][str(K_PRIMARY)]['yield']
                          - yearly[('frozen', feat)][str(Y)]
                          ['metrics'][str(K_PRIMARY)]['yield']),
            } for Y in range(2014, 2020)}

    # ---- 对账 vs v1（2018-19 池化，frozen = v1 同模型语义）----
    far = np.isin(nyear_u, (2018, 2019))
    reconciliation = {}
    for feat in arms_X:
        rec = _pooled_record(y_u[far], scores[('frozen', feat)][far])
        v1 = EXPECTED_FROM_V1[feat]
        reconciliation[feat] = {
            'batch_universe_2018_19': rec,
            'v1_archive_far_test': v1,
            'delta_yield_at_10': (rec['metrics'][str(K_PRIMARY)]['yield']
                                  - v1['yield_at_10']),
            'delta_auroc': (rec['auroc'] - v1['auroc']) if rec['auroc']
            else None,
            'note': 'v1 far test 含通知日期不可解析个案（本宇宙按月剔除），'
                    '差异由此产生；模型/过滤/种子逐字节一致',
        }

    # ---- 告警汇总 ----
    alerts = {('%s/%s' % k): _alert_summary(v)
              for k, v in arm_batches.items()}

    # ---- 归档 ----
    res = {
        'date': '2026-09-04',
        'experiment': 'sinan_walkforward_v1',
        'question': '冻结模型在按时间顺序到达的月度批次上 yield@k 是否'
                    '兑现离线承诺值；漂移与刷新代价几何',
        'endpoint': '全因死亡 SITUA_ENCE∈{3,4}（round-7 修正终点）',
        'protocol': {
            'batch_axis': 'DT_NOTIFIC 通知月，%s..%s（%d 批）'
                          % (WINDOW_FIRST, WINDOW_LAST, len(months)),
            'arm_frozen': 'train FILE_YEAR<=12 & ok_notif（与 v1 逐字节'
                          '同过滤），冻结打分 2013-2019',
            'arm_refit': '打分年 Y: train FILE_YEAR<Y & notif_year<Y '
                         '& ok_notif；Y=2013 复用冻结模型',
            'leakage_rules': [
                '训练/评估按 FILE_YEAR 划分（13-19 评估宇宙与 ≤12 训练'
                '集不相交）',
                'refit 双重过滤：FILE_YEAR-13 文件里 2014+ 才通知的个案'
                '不进 2014+ 训练集',
                '先证特征逐案点时合法：notif_j < enc_j ≤ notif_i'
                '（单源复用 run_sinan_deployment_metrics.build_prior_features）',
            ],
            'horizon_review': '批次月末 +{%d,%d,%d} 天，已结案'
                              '（enc≤审查点 且 enc>notif）子集' % HORIZONS,
            'psi_reference': '2013 自然年池化分数（冻结模型口径）；'
                             'refit 臂 PSI 含模型刷新+人群漂移双效应',
            'expected_yield_source': 'sinan_deployment_metrics_20260826.json '
                                     'far-test 池化 yield@10（离线承诺值）',
            'yield_se': 'n_pos 二项近似（top-k 选择相关性忽略）',
            'boundaries': [
                '历史模拟 ≠ 实时影子运行（真实到达流/数据质量/回流'
                '延迟需试点验证）',
                'refit 臂月初批次存在数周文件可用性边界；frozen 臂为'
                '严格前瞻主证据',
                '编码器词表与 AGE 中位数在全分析集拟合（沿用 v1 口径'
                '保可比；不含标签信息）',
            ],
        },
        'design': {
            'n_analysis': int(len(d)),
            'event_rate': base_rate,
            'n_frozen_train': int(trA.sum()),
            'n_eval_file_13_19': n_eval_files,
            'n_batch_universe': int(universe.sum()),
            'n_out_of_window_notif': n_out_of_window,
            'n_unparseable_notif': n_no_notif,
            'budgets_pct': BUDGETS,
            'horizons_days': list(HORIZONS),
        },
        'batches': batches,
        'yearly': {'%s/%s' % k: v for k, v in yearly.items()},
        'overall': {'%s/%s' % k: v for k, v in overall.items()},
        'staleness_refit_minus_frozen': staleness,
        'reconciliation_vs_v1': reconciliation,
        'alerts': alerts,
    }
    res['runtime_s'] = round(time.time() - t0, 1)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # ---- 摘要输出 ----
    for feat in arms_X:
        o = overall[('frozen', feat)]
        y13 = yearly[('frozen', feat)]['2013']
        y19 = yearly[('frozen', feat)]['2019']
        al = alerts['frozen/%s' % feat]
        r = reconciliation[feat]
        print('[frozen/%s] overall yield@10%% %.4f (AUROC %.4f) | '
              '2013 %.4f -> 2019 %.4f | PSI drift %d 批 | CI 命中 %.0f%%'
              % (feat,
                 o['metrics'][str(K_PRIMARY)]['yield'], o['auroc'],
                 y13['metrics'][str(K_PRIMARY)]['yield'],
                 y19['metrics'][str(K_PRIMARY)]['yield'],
                 len(al['psi_drift_gt_0p25']),
                 100 * (al['within_ci_rate'] or 0)))
        print('           对账 v1: yield@10%% %+.4f / AUROC %+.4f'
              % (r['delta_yield_at_10'], r['delta_auroc'] or 0.0))
    for feat in arms_X:
        st = staleness[feat]
        gains = [st[str(Y)]['delta'] for Y in range(2014, 2020)]
        print('[refit/%s] 刷新增益 yield@10%%（2014-2019）: %s'
              % (feat, ' '.join('%+.4f' % g for g in gains)))
    # 真值回链偏倚摘要（frozen/ind_all，180 天）
    gaps = [b['frozen']['ind_all']['realized']['180']['yield_at_10']
            - b['frozen']['ind_all']['metrics'][str(K_PRIMARY)]['yield']
            for b in batches.values()]
    gaps = [g for g in gaps if g is not None]
    cr = [b['frozen']['ind_all']['realized']['180']['closure_rate']
          for b in batches.values()]
    print('真值回链（frozen/ind_all, 180d）：realized−full yield@10%% '
          '均值 %+.4f | 批次 closure_rate 均值 %.3f'
          % (np.mean(gaps), np.mean(cr)))
    print('saved: %s (%.1fs)' % (OUT, time.time() - t0))
    return res


if __name__ == '__main__':
    main()
