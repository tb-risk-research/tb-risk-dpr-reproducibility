#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同队列同终点同模型对照组（2026-09-29，JCE v1(1) 外审第 2 点补强）。

审稿意见：GA2（症状终点天花板 0.993）、GA3（跨队列/终点拼接的 ~30 倍
衰减）、GA1b（oracle 用真实结局）不能排除主要替代解释；要求同一队列、
同一终点、同一模型下的对照，并报告各对照究竟改变了什么。本脚本在
HomeACF 主队列（TST 终点、同一冻结 RF、折内无泄漏管线）上补四组：

B1a 全局结果置换：个体结局全局置换（保边际）。
    改变了什么：prior_pos/prior_rate 聚合的是失去户结构的置换结局；
    prior_n（户内位置）与 exposure 块不变；两臂在同一置换结局上重训。
    判据（预固定）：Δ̄_a 的 CI 含 0 → 增益被摧毁，与"增益来自户结局
    结构"一致；若增益存活 → 增益并非来自户结局结构。

B1b 户内结果置换（保户总量）：结局仅在户内置换。
    改变了什么：每户结局总量与户率不变（户聚集保留），户内成员-结局
    对应与累计轨迹重排；prior_n 与 exposure 块不变。
    判据：Δ̄_b ≥ 0.5×Δ̄_main 且 CI_b 不含 0 → 仅户总量信息即可保留
    至少一半增益。

B1c 信息量配平 past/future 窗口：k=1,2 相邻窗口（过去 k 个 vs 未来
    k 个同户成员结局），两侧聚合人数上限相同、结构镜像（行 t 两侧
    窗口满员时各 k 人）。
    判据：Δ̄(past_k − future_k) 的 CI 含 0（k=1,2）→ 匹配信息量下
    无顺序特异超额。
    副读数：两侧窗口均满员子集上两臂 AUROC（描述性）。

B1d 同队列伪粒度衰减：随机合并 k∈{1,2,4,8} 户为伪户，prior 块、
    筛查顺序、CV 分组全部按伪户重构；k=1 位级复现主对照（同折同
    顺序同特征 → gate）。
    判据：Δ̄(k) 点估计随 k 单调不增（1→2→4→8）→ 同队列剂量反应
    支持 P2，不依赖跨队列拼接。

管线：run_sop_unified_ci_leakfree 的折内口径（BMI 折内中位、
prior_rate/future_rate 折内回退）；估计量同统一口径（Δ̄ = 20 种子
逐种子配对差均值 + 户级 cluster bootstrap CI ×2000，条件于折外
预测，重抽真户；bootstrap_seed=90000 与 unified 70000、battery 0、
scenario 50000 区分）。

用法：
    python data/run_samecohort_controls.py           # 全量
    python data/run_samecohort_controls.py smoke     # 2 种子 × 200 冒烟
输出：
    data/processed/samecohort_controls_homeacf_YYYYMMDD.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_sop_robustness_battery as bat     # noqa: E402
import run_sop_unified_ci_leakfree as lf     # noqa: E402  (rf_oof_lf 等)
import run_sop_future_placebo as futp        # noqa: E402  (COLS_FUT)

from tb_risk.validation.household_temporal import (  # noqa: E402
    assign_screening_order,
    prior_features,
)
from tb_risk.validation.real_data_infection import (  # noqa: E402
    feature_columns as inf_feature_columns,
    load_homeacf_contacts,
)
from tb_risk.validation.real_data_pi import _group_cv_indices  # noqa: E402

SMOKE = (len(sys.argv) > 1 and sys.argv[1] == 'smoke')
N_SEEDS = 2 if SMOKE else 20
N_BOOTSTRAP = 200 if SMOKE else 2000
BOOT_SEED = 90000
KS_WINDOW = (1, 2)
KS_MERGE = (1, 2, 4, 8)
GATE_TOL = 0.0005
PERM_AUROC_LO, PERM_AUROC_HI = 0.35, 0.70   # 置换臂数据完好性软门

OUT = os.path.join(HERE, 'processed',
                   'samecohort_controls_homeacf_%s.json'
                   % time.strftime('%Y%m%d'))
LEAKFREE_REF = os.path.join(
    HERE, 'processed', 'sop_unified_ci_leakfree_homeacf_20260929.json')


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(np.asarray(y, dtype=int),
                                    np.asarray(s, dtype=float)))
    except ValueError:
        return float('nan')


# ------------------------------------------------------------ 特征构造 --
def windowed_block(y, groups, order_pos, k, direction):
    """k-窗口块（past: 位置 [t−k,t)；future: (t,t+k]），命名与主块一致。

    k=∞ 时 past 退化为 prior_features、future 退化为 future_features。
    回退率（n=0 行）置 NaN，由 rf_oof_lf 折内填补。
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    order_pos = np.asarray(order_pos, dtype=float)
    w_n = np.zeros(len(y))
    w_pos = np.zeros(len(y))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        srt = idx[np.argsort(order_pos[idx], kind='stable')]
        yh = y[srt]
        m = len(srt)
        cs = np.concatenate([[0.0], np.cumsum(yh)])
        t = np.arange(m)
        if direction == 'past':
            cnt = np.minimum(t, k)
            w_n[srt] = cnt
            w_pos[srt] = cs[t] - cs[t - cnt]
        else:
            cnt = np.minimum(m - 1 - t, k)
            w_n[srt] = cnt
            w_pos[srt] = cs[t + 1 + cnt] - cs[t + 1]
    prefix = 'prior' if direction == 'past' else 'future'
    rate = np.where(w_n > 0, w_pos / np.maximum(w_n, 1.0), np.nan)
    return pd.DataFrame({
        prefix + '_n': w_n,
        prefix + '_pos': w_pos,
        prefix + '_rate': rate,
        prefix + '_screened': (w_n > 0).astype(float),
    })


def pseudo_group_vec(groups, k, rng):
    """随机合并 k 户为伪户（k=1 恒等）。"""
    if k == 1:
        return np.asarray(groups).copy()
    hh = np.unique(groups)
    hh_sh = hh[rng.permutation(len(hh))]
    mapping = {h: 'ps%06d' % (j // k) for j, h in enumerate(hh_sh)}
    return np.array([mapping[h] for h in groups])


def permute_outcome(y, groups, rng, within_household):
    """结局置换：全局（破坏户结构）或户内（保户总量）。"""
    y = np.asarray(y, dtype=float).copy()
    if not within_household:
        return y[rng.permutation(len(y))]
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) > 1:
            y[idx] = y[idx[rng.permutation(len(idx))]]
    return y


# ------------------------------------------------------------ 拟合臂 --
def _data_raw(df, bmi_raw, blocks):
    """df 原始 BMI + 附加块（prior_rate/future_rate 回退位置 NaN）。"""
    df_raw = df.copy()
    df_raw['bmi_h'] = bmi_raw
    data = pd.concat([df_raw.reset_index(drop=True)] + blocks, axis=1)
    for pre in ('prior', 'future'):
        n_col, p_col, r_col = pre + '_n', pre + '_pos', pre + '_rate'
        if n_col in data.columns:
            n_ = data[n_col].to_numpy(dtype=float)
            p_ = data[p_col].to_numpy(dtype=float)
            data[r_col] = np.where(n_ > 0,
                                   p_ / np.maximum(n_, 1.0), np.nan)
    return data


def _prior_from(df, order):
    pf = prior_features(df, order)
    # prior_features 的回退率先置 NaN（折内化在 rf_oof_lf 内完成）
    n_ = pf['prior_n'].to_numpy(dtype=float)
    p_ = pf['prior_pos'].to_numpy(dtype=float)
    pf['prior_rate'] = np.where(n_ > 0, p_ / np.maximum(n_, 1.0), np.nan)
    return pf


# ------------------------------------------------------------ 主流程 --
def main():
    t0 = time.time()
    df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    bmi_raw = bat.load_raw_bmi(df)
    cols_exp = inf_feature_columns('exposure')
    print('HomeACF: n=%d events=%d households=%d'
          % (len(y), int(y.sum()), len(np.unique(groups))))

    # ---------- B1a / B1b：结果置换（折取自原结局，逐位同构） ----------
    b1_store = {}
    for tag, within in (('b1a_global_perm', False),
                        ('b1b_within_hh_perm', True)):
        rows = []
        for s in range(N_SEEDS):
            rng = np.random.default_rng(
                (51000 if not within else 52000) + s)
            folds = _group_cv_indices(groups, y,
                                      n_splits=bat.N_SPLITS, seed=s)
            order = assign_screening_order(df, seed=s, mode='random')
            y_p = permute_outcome(y, groups, rng, within)
            df_p = df.copy()
            df_p['tst_pos10'] = y_p
            pf = _prior_from(df_p, order)
            data = _data_raw(df, bmi_raw, [pf])
            o_exp = lf.rf_oof_lf(cols_exp, folds, y_p, data, s)
            o_ep = lf.rf_oof_lf(cols_exp + bat.COLS_PRIOR, folds, y_p,
                                data, s)
            rows.append({
                'seed': s, 'y': y_p,
                'exp': o_exp, 'ep': o_ep,
                'auroc_exp': _auroc(y_p, o_exp),
                'auroc_ep': _auroc(y_p, o_ep),
            })
            print('%s seed %2d: exp=%.4f ep=%.4f (%ds)'
                  % (tag, s, rows[-1]['auroc_exp'], rows[-1]['auroc_ep'],
                     time.time() - t0))
        b1_store[tag] = rows

    # ---------- B1c：信息量配平 past/future 窗口 ----------
    b1c = {'k%d' % k: {'past': [], 'fut': []} for k in KS_WINDOW}
    b1c_exp = []
    b1c_masks = {'k%d' % k: [] for k in KS_WINDOW}
    for s in range(N_SEEDS):
        folds = _group_cv_indices(groups, y,
                                  n_splits=bat.N_SPLITS, seed=s)
        order = assign_screening_order(df, seed=s, mode='random')
        op = order.to_numpy(dtype=float)
        data_exp = _data_raw(df, bmi_raw, [])
        o_exp = lf.rf_oof_lf(cols_exp, folds, y, data_exp, s)
        b1c_exp.append(o_exp)
        for k in KS_WINDOW:
            blk_p = windowed_block(y, groups, op, k, 'past')
            blk_f = windowed_block(y, groups, op, k, 'future')
            data_p = _data_raw(df, bmi_raw, [blk_p])
            data_f = _data_raw(df, bmi_raw, [blk_f])
            o_p = lf.rf_oof_lf(cols_exp + bat.COLS_PRIOR, folds, y,
                               data_p, s)
            o_f = lf.rf_oof_lf(cols_exp + futp.COLS_FUT, folds, y,
                               data_f, s)
            b1c['k%d' % k]['past'].append(o_p)
            b1c['k%d' % k]['fut'].append(o_f)
            # 逐种子掩码：两侧窗口均满员（各 k 人）的行
            b1c_masks['k%d' % k].append(
                (blk_p['prior_n'].to_numpy() >= k)
                & (blk_f['future_n'].to_numpy() >= k))
        print('B1c seed %2d fitted (%ds)' % (s, time.time() - t0))

    # ---------- B1d：伪粒度衰减（k=1 位级复现主对照） ----------
    b1d = {'k%d' % k: {'exp': [], 'ep': []} for k in KS_MERGE}
    for s in range(N_SEEDS):
        for k in KS_MERGE:
            if k == 1:
                pg = pseudo_group_vec(groups, 1, None)
            else:
                rng = np.random.default_rng(53000 + 10 * k + s)
                pg = pseudo_group_vec(groups, k, rng)
            df_ps = df.copy()
            df_ps['record_id'] = pg
            folds = _group_cv_indices(pg, y,
                                      n_splits=bat.N_SPLITS, seed=s)
            order = assign_screening_order(df_ps, seed=s, mode='random')
            pf = _prior_from(df_ps, order)
            data = _data_raw(df, bmi_raw, [pf])
            o_exp = lf.rf_oof_lf(cols_exp, folds, y, data, s)
            o_ep = lf.rf_oof_lf(cols_exp + bat.COLS_PRIOR, folds, y,
                                data, s)
            b1d['k%d' % k]['exp'].append(o_exp)
            b1d['k%d' % k]['ep'].append(o_ep)
        print('B1d seed %2d fitted (%ds)' % (s, time.time() - t0))

    # ---------- gate：B1d k=1 与 B1c exposure 对照 leakfree 归档 ----------
    gates = {'b1d_k1_vs_leakfree': {}, 'b1c_exp_vs_leakfree': {},
             'perm_arm_sanity': {}}
    ok = True
    with open(LEAKFREE_REF, encoding='utf-8') as f:
        ref = json.load(f)
    ref_ep = ref['arm_stats']['exposure_prior']['per_seed']
    ref_exp = ref['arm_stats']['exposure']['per_seed']
    for s in range(N_SEEDS):
        mine = _auroc(y, b1d['k1']['ep'][s]) - _auroc(
            y, b1d['k1']['exp'][s])
        refd = ref_ep[s] - ref_exp[s]
        gates['b1d_k1_vs_leakfree']['seed%d' % s] = {
            'this': round(mine, 6), 'ref': round(refd, 6),
            'passed': bool(abs(mine - refd) <= GATE_TOL)}
        ok &= abs(mine - refd) <= GATE_TOL
        mine_e = _auroc(y, b1c_exp[s])
        gates['b1c_exp_vs_leakfree']['seed%d' % s] = {
            'this': round(mine_e, 6), 'ref': ref_exp[s],
            'passed': bool(abs(mine_e - ref_exp[s]) <= GATE_TOL)}
        ok &= abs(mine_e - ref_exp[s]) <= GATE_TOL
    for tag in ('b1a_global_perm', 'b1b_within_hh_perm'):
        vals = [r['auroc_exp'] for r in b1_store[tag]]
        in_range = all(PERM_AUROC_LO < v < PERM_AUROC_HI for v in vals)
        gates['perm_arm_sanity'][tag] = {
            'auroc_exp_range': [round(min(vals), 4), round(max(vals), 4)],
            'passed': bool(in_range)}
        ok &= in_range
    print('gates:', {k: all(v.get('passed', True)
                            for v in g.values())
                     for k, g in gates.items()})
    if not ok:
        raise SystemExit('同队列对照 gate 失败，中止归档')

    # ---------- 读数 + 统一 bootstrap ----------
    hh = np.unique(groups)
    member_idx = {h: np.where(groups == h)[0] for h in hh}

    def seed_deltas(score_a_list, score_b_list, y_list=None):
        if y_list is None:
            y_list = [y] * len(score_a_list)
        return np.array([_auroc(y_list[s], score_a_list[s])
                         - _auroc(y_list[s], score_b_list[s])
                         for s in range(len(score_a_list))])

    def boot_ci(score_a_list, score_b_list, y_list=None):
        if y_list is None:
            y_list = [y] * len(score_a_list)
        vals = []
        for b in range(N_BOOTSTRAP):
            rng_b = np.random.default_rng(BOOT_SEED + b)
            hh_s = rng_b.choice(hh, size=len(hh), replace=True)
            idx = np.concatenate([member_idx[h] for h in hh_s])
            ds = [_auroc(y_list[s][idx], score_a_list[s][idx])
                  - _auroc(y_list[s][idx], score_b_list[s][idx])
                  for s in range(len(score_a_list))]
            m = np.nanmean(ds) if np.isfinite(np.nanmean(ds)) else float('nan')
            vals.append(m)
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def eff(deltas, ci):
        return {'per_seed': [round(float(x), 4) for x in deltas],
                'mean': float(deltas.mean()), 'sd': float(deltas.std()),
                'share_positive': float((deltas > 0).mean()),
                'ci95_unified': ci,
                'ci_excludes_zero': (bool(ci[0] > 0 or ci[1] < 0)
                                     if ci else None)}

    results = {'b1a_global_perm': {}, 'b1b_within_hh_perm': {},
               'b1c_matched_windows': {}, 'b1d_pseudo_granularity': {}}

    d_main = seed_deltas(b1d['k1']['ep'], b1d['k1']['exp'])
    main_eff = eff(d_main, boot_ci(b1d['k1']['ep'], b1d['k1']['exp']))
    results['b1d_pseudo_granularity']['k1'] = main_eff
    print('main contrast (B1d k=1): Δ̄=%+.4f CI[%+.4f,%+.4f]'
          % (main_eff['mean'], *main_eff['ci95_unified']))

    for tag in ('b1a_global_perm', 'b1b_within_hh_perm'):
        rows = b1_store[tag]
        yl = [r['y'] for r in rows]
        d = seed_deltas([r['ep'] for r in rows], [r['exp'] for r in rows],
                        yl)
        ci = boot_ci([r['ep'] for r in rows], [r['exp'] for r in rows], yl)
        e = eff(d, ci)
        e['retention_vs_main'] = float(e['mean'] / main_eff['mean'])
        results[tag] = e
        print('%s: Δ̄=%+.4f CI[%+.4f,%+.4f] retention=%.2f'
              % (tag, e['mean'], *e['ci95_unified'],
                 e['retention_vs_main']))

    for k in KS_WINDOW:
        key = 'k%d' % k
        past, fut = b1c[key]['past'], b1c[key]['fut']
        d_pf = seed_deltas(past, fut)
        ci_pf = boot_ci(past, fut)
        e_pf = eff(d_pf, ci_pf)
        d_pe = seed_deltas(past, b1c_exp)
        e_pe = eff(d_pe, None)
        d_fe = seed_deltas(fut, b1c_exp)
        e_fe = eff(d_fe, None)
        masks_s = b1c_masks[key]
        sub = {
            'n_rows_mean': float(np.mean(
                [int(m.sum()) for m in masks_s])),
            'n_events_mean': float(np.mean(
                [int(y[m].sum()) for m in masks_s])),
            'auroc_past_matched': float(np.mean(
                [_auroc(y[masks_s[s]], past[s][masks_s[s]])
                 for s in range(N_SEEDS)])),
            'auroc_fut_matched': float(np.mean(
                [_auroc(y[masks_s[s]], fut[s][masks_s[s]])
                 for s in range(N_SEEDS)])),
            'auroc_exp_matched': float(np.mean(
                [_auroc(y[masks_s[s]], b1c_exp[s][masks_s[s]])
                 for s in range(N_SEEDS)])),
        }
        results['b1c_matched_windows'][key] = {
            'past_minus_fut': e_pf, 'past_minus_exp': e_pe,
            'fut_minus_exp': e_fe, 'matched_subset': sub}
        print('B1c %s: past−fut Δ̄=%+.4f CI[%+.4f,%+.4f]; '
              'matched-subset AUROC past=%.4f fut=%.4f exp=%.4f (n≈%d)'
              % (key, e_pf['mean'], *e_pf['ci95_unified'],
                 sub['auroc_past_matched'], sub['auroc_fut_matched'],
                 sub['auroc_exp_matched'], int(sub['n_rows_mean'])))

    for k in KS_MERGE[1:]:
        key = 'k%d' % k
        d = seed_deltas(b1d[key]['ep'], b1d[key]['exp'])
        ci = boot_ci(b1d[key]['ep'], b1d[key]['exp'])
        e = eff(d, ci)
        e['ratio_vs_k1'] = float(e['mean'] / main_eff['mean'])
        results['b1d_pseudo_granularity'][key] = e
        print('B1d %s: Δ̄=%+.4f CI[%+.4f,%+.4f] ratio=%.3f'
              % (key, e['mean'], *e['ci95_unified'], e['ratio_vs_k1']))

    # ---------- 预固定判据 ----------
    a = results['b1a_global_perm']
    b = results['b1b_within_hh_perm']
    verdicts = {
        'b1a_gain_destroyed': bool(not a['ci_excludes_zero']),
        'b1b_household_totals_retain': bool(
            b['mean'] >= 0.5 * main_eff['mean'] and b['ci_excludes_zero']),
        'b1c_no_order_excess_at_matched_info': bool(all(
            not results['b1c_matched_windows']['k%d' % k][
                'past_minus_fut']['ci_excludes_zero']
            for k in KS_WINDOW)),
        'b1d_monotone_decay': bool(all(
            results['b1d_pseudo_granularity']['k%d' % k1]['mean']
            >= results['b1d_pseudo_granularity']['k%d' % k2]['mean']
            for k1, k2 in zip(KS_MERGE[:-1], KS_MERGE[1:]))),
    }

    result = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'samecohort_controls',
        'status': 'JCE v1(1) 外审第 2 点：同队列同终点同模型对照'
                  '（事后补做，判据预固定于本脚本）',
        'design': {
            'pipeline': 'run_sop_unified_ci_leakfree 折内口径'
                        '（BMI 折内中位 + 回退率折内）',
            'estimator': 'Δ̄ = mean over 20 seeds of per-seed paired '
                         'differences; CI = household-cluster bootstrap '
                         '×2000 conditional on OOF predictions '
                         '(bootstrap_seed=90000)',
            'n_seeds': N_SEEDS, 'n_bootstrap': N_BOOTSTRAP,
            'ks_window': list(KS_WINDOW), 'ks_merge': list(KS_MERGE),
            'criteria_prefixed': {
                'b1a': '全局结果置换下 Δ̄ CI 含 0 → 增益被摧毁',
                'b1b': 'Δ̄_b ≥ 0.5×Δ̄_main 且 CI 不含 0 → 户总量保留'
                       '至少一半增益',
                'b1c': 'k=1,2 的 past−future Δ̄ CI 均含 0 → 匹配信息量'
                       '下无顺序特异超额',
                'b1d': 'Δ̄(k) 点估计随 k=1→2→4→8 单调不增 → 同队列'
                       '剂量反应支持 P2',
            },
            'changed_what': {
                'b1a': '结局全局置换：破坏户结构与个体关联，保留边际；'
                       'prior_n 与 exposure 块不变',
                'b1b': '结局户内置换：保留户总量/户率，重排户内成员-结局'
                       '对应与累计轨迹；prior_n 与 exposure 块不变',
                'b1c': '过去/未来窗口各限 k 人：信息量上限配平、结构'
                       '镜像；对照两侧均在同一折/顺序/模型下重训',
                'b1d': '伪户合并 k 户：prior 聚合、筛查顺序、CV 分组均'
                       '按伪户重构；个体特征与结局不变',
            },
            'gates': gates,
            'leakfree_ref': os.path.basename(LEAKFREE_REF),
        },
        'results': results,
        'verdicts': verdicts,
        'smoke': SMOKE,
        'runtime_sec': round(time.time() - t0, 1),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print('归档:', OUT)
    print('verdicts:', verdicts)


if __name__ == '__main__':
    main()
