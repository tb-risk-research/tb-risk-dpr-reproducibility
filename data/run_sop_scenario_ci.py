#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""级联情景 CI 补充（2026-09-28，评审回应第二步收尾）。

battery（run_sop_robustness_battery.py，20260927）B③ 已给出两阶段级联
网格（q×K×首波模式）点估计，本脚本补齐三项缺口：

  (a) 主格点（q=5%, K=10%）户级 cluster bootstrap CI——级联 vs 静态、
      LR 版级联 vs 静态、级联 vs LR 级联、lr_eb 静态 vs 静态；
  (b) dep-LR 级联：完整"简单规则流程"对照——首波 LR exposure top-q、
      第二波 σ(logit(p_lr)+7(r̂−π̂))，与 dep-RF（RF 基概率）同构，
      回答"简单透明的家庭信息更新规则接入全流程是否同好"；
  (c) 三情景框架（同信息同预算）：
        same_day    同日集中检查：结果同日返回，对当日排序不可用
                    → 退化为 exposure 静态 top-K（battery exp_static）；
        batched     分批检查、结果即时返回可用 → 级联 exposure-top 首波；
        batched_rnd 分批检查但首波随机选择 → 级联 random 首波；
        random      未靶向筛查期望 = K；
        free_evidence（参考上界）= RF(ep) 静态 top-K（先证信息免费口径）。

假设披露（HomeACF 无真实筛查日期，故取两端包络）：
  - 分批即时返回情景假设首波结果在第二波开始前全部返回，无延迟、
    无缺失；同日情景假设结果当日不可用于排序。真实流程（未知返回
    滞后）介于两者之间，读数为此包络而非实测流程。

协议：与 battery 逐位同构（run_seed_std 3 锚臂 + LR 臂重建 data 与
B④ 相同构造）；20 种子点估计 + 末种子户级 cluster bootstrap ×2000。

用法：
    python data/run_sop_scenario_ci.py
输出：
    data/processed/sop_scenario_ci_homeacf_20260928.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_sop_robustness_battery as bat   # noqa: E402
import run_sop_deepening as rsd             # noqa: E402 (dep_scores)
from tb_risk.validation.household_temporal import (  # noqa: E402
    assign_screening_order,
    prior_features,
)
from tb_risk.validation.real_data_infection import (  # noqa: E402
    feature_columns as inf_feature_columns,
    load_homeacf_contacts,
)

OUT = os.path.join(HERE, 'processed',
                   'sop_scenario_ci_homeacf_%s.json' % time.strftime('%Y%m%d'))

Q_MAIN = 0.05
K_GRID = (0.10, 0.20, 0.30, 0.50)
K_MAIN = 0.10
N_SEEDS = 20
N_BOOTSTRAP = 2000


def cascade_once_lr(oof_lr, y, groups, pi_row, q, K, w1_mode, rng):
    """LR 版级联（简单规则全流程）：首波 LR top-q，第二波
    σ(logit(p_lr)+w(r̂−π̂))。与 bat.cascade_once 唯一差异 = 基概率列。"""
    n = len(y)
    n_events = int(y.sum())
    m1 = max(1, int(np.ceil(q * n)))
    w1 = np.zeros(n, dtype=bool)
    if w1_mode == 'exposure_top':
        w1[np.argsort(-oof_lr, kind='mergesort')[:m1]] = True
    else:
        w1[rng.choice(n, size=m1, replace=False)] = True

    sub = pd.DataFrame({'g': groups[w1], 'y': y[w1]})
    gn = sub.groupby('g').size()
    gp = sub.groupby('g')['y'].sum()
    n1 = pd.Series(groups).map(gn).fillna(0.0).to_numpy(dtype=float)
    pos1 = pd.Series(groups).map(gp).fillna(0.0).to_numpy(dtype=float)

    score2 = rsd.dep_scores(oof_lr, n1, pos1, pi_row, bat.DEP_K, bat.DEP_W)

    total_budget = max(1, int(np.ceil(K * n)))
    b2 = total_budget - m1
    captured = float(y[w1].sum())
    w2 = np.zeros(n, dtype=bool)
    if b2 > 0:
        rem = np.where(~w1)[0]
        pick = rem[np.argsort(-score2[rem], kind='mergesort')[:b2]]
        w2[pick] = True
        captured += float(y[w2].sum())
    n_screened = m1 + int(w2.sum())
    capture = captured / n_events
    nns = n_screened / captured if captured > 0 else float('nan')
    return capture, nns, n_screened


def static_capture(oof_scores, y, K):
    """一次性 top-⌈K·n⌉ 排序捕获率（同日/静态/免费信息口径）。"""
    n = len(y)
    n_events = int(y.sum())
    k = max(1, int(np.ceil(K * n)))
    top = np.argsort(-oof_scores, kind='mergesort')[:k]
    return float(y[top].sum()) / n_events


def run_seed_full(seed, df, y, groups):
    """run_seed_std + LR 臂（与 battery B④ 构造逐位一致）。"""
    r = bat.run_seed_std(seed, df, y, groups)
    order = assign_screening_order(df, seed=seed, mode='random')
    pf = prior_features(df, order)
    data = pd.concat([df.reset_index(drop=True), pf], axis=1)
    data['prior_rate_eb'] = (r['prior_pos'] + r['k_row'] * r['pi_row']) \
        / (r['prior_n'] + r['k_row'])
    cols_exp = inf_feature_columns('exposure')
    r['oof']['lr_exposure'] = bat.lr_oof(cols_exp, r['folds'], y, data, seed)
    r['oof']['lr_exposure_eb'] = bat.lr_oof(
        cols_exp + ['prior_rate_eb'], r['folds'], y, data, seed)
    return r


def main():
    t0 = time.time()
    df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    print('HomeACF: n=%d events=%d households=%d' % (
        len(y), int(y.sum()), len(np.unique(groups))))

    std = []
    for s in range(N_SEEDS):
        std.append(run_seed_full(s, df, y, groups))
        print('seed %2d done (%ds)' % (s, time.time() - t0))

    # ---- 情景网格（20 种子均值 + 逐种子）----
    grid = {}
    per_seed = {}
    for K in K_GRID:
        rows = {k: [] for k in (
            'same_day_static', 'batched_dep_rf', 'batched_dep_lr',
            'batched_random_first', 'free_evidence_ep_static',
            'lr_eb_static')}
        nns_rows = {k: [] for k in ('same_day_static', 'batched_dep_rf',
                                    'batched_dep_lr')}
        for r in std:
            rng = np.random.default_rng(1000 + r['seed'])
            o = r['oof']
            c_dep = bat.cascade_once(o['exposure'], y, groups,
                                     r['pi_row'], Q_MAIN, K,
                                     'exposure_top', rng)
            c_lr = cascade_once_lr(o['lr_exposure'], y, groups,
                                    r['pi_row'], Q_MAIN, K,
                                    'exposure_top', rng)
            c_rnd = bat.cascade_once(o['exposure'], y, groups,
                                     r['pi_row'], Q_MAIN, K,
                                     'random', rng)
            rows['same_day_static'].append(
                static_capture(o['exposure'], y, K))
            rows['batched_dep_rf'].append(c_dep[0])
            rows['batched_dep_lr'].append(c_lr[0])
            rows['batched_random_first'].append(c_rnd[0])
            rows['free_evidence_ep_static'].append(
                static_capture(o['exposure_prior'], y, K))
            rows['lr_eb_static'].append(
                static_capture(o['lr_exposure_eb'], y, K))
            nns_rows['same_day_static'].append(
                max(1, int(np.ceil(K * len(y)))) / max(
                    static_capture(o['exposure'], y, K)
                    * float(y.sum()), 1e-12))
            nns_rows['batched_dep_rf'].append(c_dep[1])
            nns_rows['batched_dep_lr'].append(c_lr[1])
        grid['K_%s' % K] = {
            k: float(np.mean(v)) for k, v in rows.items()}
        grid['K_%s' % K]['random_expected'] = K
        grid['K_%s' % K]['nns'] = {
            k: float(np.mean(v)) for k, v in nns_rows.items()}
        per_seed['K_%s' % K] = {
            'batched_dep_rf_minus_static': [
                a - b for a, b in zip(rows['batched_dep_rf'],
                                      rows['same_day_static'])],
            'batched_dep_lr_minus_static': [
                a - b for a, b in zip(rows['batched_dep_lr'],
                                      rows['same_day_static'])],
            'batched_dep_rf_minus_dep_lr': [
                a - b for a, b in zip(rows['batched_dep_rf'],
                                      rows['batched_dep_lr'])],
        }
        print('grid K=%.2f done (%ds)' % (K, time.time() - t0))

    # ---- 末种子户级 cluster bootstrap（主格点 q=5%, K=10%）----
    last = std[-1]
    hh = np.unique(groups)
    member_idx = {h: np.where(groups == h)[0] for h in hh}

    o = last['oof']
    exp_full, lr_full, lreb_full = (o['exposure'], o['lr_exposure'],
                                    o['lr_exposure_eb'])
    pi_full = last['pi_row']

    d_names = ('cascade_vs_static', 'lr_cascade_vs_static',
               'cascade_vs_lr_cascade', 'lr_eb_static_vs_static')
    deltas = {k: [] for k in d_names}
    for b in range(N_BOOTSTRAP):
        rng_b = np.random.default_rng(50000 + b)
        hh_s = rng_b.choice(hh, size=len(hh), replace=True)
        idx = np.concatenate([member_idx[h] for h in hh_s])
        yb, gb = y[idx], groups[idx]
        exp_b, lr_b, lreb_b = exp_full[idx], lr_full[idx], lreb_full[idx]
        pi_b = pi_full[idx]
        c_dep = bat.cascade_once(exp_b, yb, gb, pi_b, Q_MAIN, K_MAIN,
                                 'exposure_top', rng_b)[0]
        c_lr = cascade_once_lr(lr_b, yb, gb, pi_b, Q_MAIN, K_MAIN,
                               'exposure_top', rng_b)[0]
        c_st = static_capture(exp_b, yb, K_MAIN)
        c_lreb = static_capture(lreb_b, yb, K_MAIN)
        deltas['cascade_vs_static'].append(c_dep - c_st)
        deltas['lr_cascade_vs_static'].append(c_lr - c_st)
        deltas['cascade_vs_lr_cascade'].append(c_dep - c_lr)
        deltas['lr_eb_static_vs_static'].append(c_lreb - c_st)
        if (b + 1) % 500 == 0:
            print('bootstrap %d/%d (%ds)' % (b + 1, N_BOOTSTRAP,
                                             time.time() - t0))

    ci = {}
    for k, v in deltas.items():
        v = np.asarray(v)
        ci[k] = {
            'mean': float(v.mean()),
            'ci95': [float(np.percentile(v, 2.5)),
                     float(np.percentile(v, 97.5))],
            'excludes_zero': bool(np.percentile(v, 2.5) > 0
                                  or np.percentile(v, 97.5) < 0)}

    ps = per_seed['K_%s' % K_MAIN]
    result = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'sop_scenario_ci',
        'design': {
            'q_main': Q_MAIN, 'k_grid': list(K_GRID), 'k_main': K_MAIN,
            'n_seeds': N_SEEDS, 'n_bootstrap': N_BOOTSTRAP,
            'dep_w': bat.DEP_W, 'dep_k': bat.DEP_K,
            'scenarios': {
                'same_day_static': '同日集中检查：结果当日不可用于排序，'
                                   '退化为 exposure 静态 top-K',
                'batched_dep_rf': '分批 + 结果即时返回；第二波基概率=RF '
                                  'exposure（部署公式口径）',
                'batched_dep_lr': '分批 + 结果即时返回；第二波基概率=LR '
                                  'exposure（简单规则全流程）',
                'batched_random_first': '分批但首波随机选择',
                'free_evidence_ep_static': '参考上界：RF(ep) 静态 top-K'
                                           '（先证信息免费）',
                'lr_eb_static': '简单规则信息免费口径：LR+EB 率静态 top-K'},
            'assumptions': [
                'HomeACF 无真实筛查日期；分批情景假设首波结果在第二波'
                '开始前全部返回（无延迟、无缺失）',
                '同日情景假设结果当日不可用于排序（同日批量检测包络下端）',
                '真实流程（未知返回滞后）介于同日与即时返回两情景之间',
                '全部读数为模拟情景，非实测筛查流程'],
        },
        'grid': grid,
        'per_seed_share': {
            k: float(np.mean([d > 0 for d in v]))
            for k, v in ps.items()},
        'ci_main': ci,
        'runtime_sec': round(time.time() - t0, 1),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print('归档:', OUT)

    print('\n=== 主格点 q=%.0f%% K=%.0f%% 户级 bootstrap CI ==='
          % (Q_MAIN * 100, K_MAIN * 100))
    for k in d_names:
        print('  %-28s Δ=%+.4f CI[%.4f,%.4f] %s'
              % (k, ci[k]['mean'], ci[k]['ci95'][0], ci[k]['ci95'][1],
                 'excl-zero' if ci[k]['excludes_zero'] else 'incl-zero'))
    print('per-seed share:', {k: round(v, 2)
                              for k, v in result['per_seed_share'].items()})


if __name__ == '__main__':
    main()
