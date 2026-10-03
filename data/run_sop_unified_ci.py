#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一口径 CI 重算（2026-09-28，外审 v0.1(2) 第一优先项）。

背景：v0.1(2) 外审指出全稿系统性混用两种估计量——点估计 = 20 种子
（CV 重播种 × 随机筛查顺序）均值，而配对 CI = 末种子 pooled OOF 户级
cluster bootstrap。摘要因此出现 0.714−0.644=0.070 却报 Δ+0.059、级联
Δ2.7pp 却配末种子 +0.9pp CI 的口径错位。本脚本一次性修复为**单一
主口径**：

  效应量  Δ̄ = mean_s [AUROC_s(臂A) − AUROC_s(臂B)]   （20 种子逐种子
           配对差的均值；级联/DCA/Brier 同构：逐种子配对差 → 均值）
  置信区间  对 Δ̄ 本身做户级 cluster bootstrap ×2000：每次重抽 877 户
           （有放回），同一重抽样本套用于全部 20 种子，逐种子在重抽行
           上重算配对差再取 20 种子均值；百分位 2.5/97.5。

末种子 pooled 读数不再入正文（旧值归档于 superseded 字段以供追溯）；
逐种子符号份额（R5a 纪律）保留为稳定性描述。

覆盖（全部为稿件正文带 CI 的引用量，逐项同口径重算）：
  A. AUROC 域：ep−exp（主）、ep−ind、prior_only−ind、LR+EB−LR（简单
     对照）、ep−LR+EB（RF 溢价）、fut−exp / fut−prior（placebo）、
     dep_future−dep_fixed（部署公式镜像）、6 亚组 ep−exp；
  B. 级联域（q=5%, K=10%）：RF 级联−静态、LR 级联−静态、LR+EB 免费
     口径−静态、RF 级联−LR 级联；
  C. DCA 域（crossfit Platt 主读数，cal 同队列为副）：7 阈值 ΔNB
     全网格 + 曲线均值；
  D. Brier 域（crossfit 主）：ΔBS + Murphy RES/REL 分量；
  E. 新增亚组预算读数（外审 P2 要求）：各亚组在 K=10% 下的捕获率、
     漏检数、O:E 比、ECE（描述性，20 种子均值）。

锚臂复现 gate（与 battery/placebo 同款）：ind 0.6428 / exposure 0.6441 /
prior_only 0.6443 / exposure_prior 0.7142 / dep_fixed 0.7202 /
exposure_future 0.7100，|Δ|<0.002 方可归档。

用法：
    python data/run_sop_unified_ci.py           # 全量 20 种子 × 2000 bootstrap
    python data/run_sop_unified_ci.py smoke     # 2 种子 × 200 冒烟
输出：
    data/processed/sop_unified_ci_homeacf_YYYYMMDD.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_sop_robustness_battery as bat   # noqa: E402  (lr_oof/platt_crossfit/常数)
import run_sop_deepening as rsd             # noqa: E402  (run_seed/rf_oof/dep_scores)
import run_sop_future_placebo as fut        # noqa: E402  (future_features/COLS_FUT)
import run_sop_clinical_layer as rcl        # noqa: E402  (platt/net_benefit/brier_decomp/PT_GRID)
import run_sop_scenario_ci as scn           # noqa: E402  (cascade_once_lr/static_capture)

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
BOOT_SEED = 70000                      # 与 scenario_ci 的 50000、battery 的 0 区分
Q_MAIN, K_MAIN = 0.05, 0.10
ANCHOR = {'ind': 0.6428, 'exposure': 0.6441, 'prior_only': 0.6443,
          'exposure_prior': 0.7142, 'dep_fixed': 0.7202,
          'exposure_future': 0.7100}

OUT = os.path.join(HERE, 'processed',
                   'sop_unified_ci_homeacf_%s.json' % time.strftime('%Y%m%d'))


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(np.asarray(y, dtype=int),
                                    np.asarray(s, dtype=float)))
    except ValueError:                 # 退化重抽（单类）→ NaN，nanmean 处理
        return float('nan')


# ------------------------------------------------------------ 单种子全臂 --
def seed_full(seed, df, y, groups):
    """rsd.run_seed（v1 六臂，锚 gate 位位复现）+ 未来/LR/Platt 臂。

    folds/pi_row/k_row 重算路径与 battery run_seed_std 逐位一致（锚 gate
    验证过等价）；dep_future 按 placebo 镜像口径（pi_true 基率）。
    """
    r = rsd.run_seed(seed, df, y, groups)
    folds = _group_cv_indices(groups, y, n_splits=bat.N_SPLITS, seed=seed)
    order = assign_screening_order(df, seed=seed, mode='random')
    pf = prior_features(df, order)
    ff = fut.future_features(df, order)

    prior_n = r['prior_n'] if 'prior_n' in r else pf['prior_n'].to_numpy(float)
    prior_pos = (r['prior_pos'] if 'prior_pos' in r
                 else pf['prior_pos'].to_numpy(float))
    pi_row = r['pi_row'] if 'pi_row' in r else None
    k_row = r['k_row'] if 'k_row' in r else None
    if pi_row is None or k_row is None:      # 与 run_seed_std 相同的重算路径
        fold_of = np.zeros(len(y), dtype=int)
        eb_params = []
        for f, (tr, te) in enumerate(folds):
            fold_of[te] = f
            eb_params.append(rsd.eb_shrinkage_params(y[tr], groups[tr]))
        pi_row = np.array([eb_params[f][0] for f in fold_of])
        k_row = np.array([eb_params[f][1] for f in fold_of])

    data = pd.concat([df.reset_index(drop=True), pf], axis=1)
    data['prior_rate_eb'] = (prior_pos + k_row * pi_row) / (prior_n + k_row)
    data_fut = pd.concat([df.reset_index(drop=True), ff], axis=1)
    cols_exp = inf_feature_columns('exposure')

    oof = r['oof']
    oof['exposure_future'] = rsd.rf_oof(
        cols_exp + fut.COLS_FUT, folds, y, data_fut, seed)
    oof['dep_future'] = rsd.dep_scores(
        oof['exposure'], ff['future_n'].to_numpy(float),
        ff['future_pos'].to_numpy(float), r['pi_true'],
        rsd.DEP_K, rsd.DEP_W)
    oof['lr_exposure'] = bat.lr_oof(cols_exp, folds, y, data, seed)
    oof['lr_exposure_eb'] = bat.lr_oof(
        cols_exp + ['prior_rate_eb'], folds, y, data, seed)

    oof['p_cross_ep'] = bat.platt_crossfit(oof['exposure_prior'], y, folds)
    oof['p_cross_exp'] = bat.platt_crossfit(oof['exposure'], y, folds)
    oof['p_cross_dep'] = bat.platt_crossfit(oof['dep_fixed'], y, folds)
    oof['p_cal_ep'] = rcl.platt(oof['exposure_prior'], y)
    oof['p_cal_exp'] = rcl.platt(oof['exposure'], y)

    r['folds'] = folds
    r['pi_row'] = pi_row
    return r


# ------------------------------------------------------ 亚组预算读数工具 --
def picked_static(scores, K):
    n = len(scores)
    k = max(1, int(np.ceil(K * n)))
    picked = np.zeros(n, dtype=bool)
    picked[np.argsort(-np.asarray(scores), kind='mergesort')[:k]] = True
    return picked


def picked_cascade_rf(oof_exp, y, groups, pi_row, q, K):
    """bat.cascade_once 的选择集版本（exposure_top 模式，确定性）。"""
    n = len(y)
    m1 = max(1, int(np.ceil(q * n)))
    w1 = np.zeros(n, dtype=bool)
    w1[np.argsort(-oof_exp, kind='mergesort')[:m1]] = True
    sub = pd.DataFrame({'g': groups[w1], 'y': y[w1]})
    gn = sub.groupby('g').size()
    gp = sub.groupby('g')['y'].sum()
    n1 = pd.Series(groups).map(gn).fillna(0.0).to_numpy(dtype=float)
    pos1 = pd.Series(groups).map(gp).fillna(0.0).to_numpy(dtype=float)
    score2 = rsd.dep_scores(oof_exp, n1, pos1, pi_row, bat.DEP_K, bat.DEP_W)
    w2 = np.zeros(n, dtype=bool)
    b2 = max(1, int(np.ceil(K * n))) - m1
    if b2 > 0:
        rem = np.where(~w1)[0]
        w2[rem[np.argsort(-score2[rem], kind='mergesort')[:b2]]] = True
    return w1 | w2


def picked_cascade_lr(oof_lr, y, groups, pi_row, q, K):
    """scn.cascade_once_lr 的选择集版本（LR 基概率，确定性）。"""
    n = len(y)
    m1 = max(1, int(np.ceil(q * n)))
    w1 = np.zeros(n, dtype=bool)
    w1[np.argsort(-oof_lr, kind='mergesort')[:m1]] = True
    sub = pd.DataFrame({'g': groups[w1], 'y': y[w1]})
    gn = sub.groupby('g').size()
    gp = sub.groupby('g')['y'].sum()
    n1 = pd.Series(groups).map(gn).fillna(0.0).to_numpy(dtype=float)
    pos1 = pd.Series(groups).map(gp).fillna(0.0).to_numpy(dtype=float)
    score2 = rsd.dep_scores(oof_lr, n1, pos1, pi_row, bat.DEP_K, bat.DEP_W)
    w2 = np.zeros(n, dtype=bool)
    b2 = max(1, int(np.ceil(K * n))) - m1
    if b2 > 0:
        rem = np.where(~w1)[0]
        w2[rem[np.argsort(-score2[rem], kind='mergesort')[:b2]]] = True
    return w1 | w2


def ece_10bin(y, p, n_bins=10):
    """等分位 10 箱 ECE（与 brier_decomp 同切箱方式）。"""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return float('nan')
    order = np.argsort(p, kind='mergesort')
    m = n // n_bins
    ece = 0.0
    for b in range(n_bins):
        rows = order[b * m: (b + 1) * m] if b < n_bins - 1 else order[b * m:]
        if len(rows) == 0:
            continue
        ece += len(rows) * abs(float(p[rows].mean()) - float(y[rows].mean()))
    return ece / n


def main():
    t0 = time.time()
    df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    print('HomeACF: n=%d events=%d households=%d' % (
        len(y), int(y.sum()), len(np.unique(groups))))

    masks = {
        'age_lt15': df['contact_age'].to_numpy() < 15,
        'age_ge15': df['contact_age'].to_numpy() >= 15,
        'hiv_pos': df['hiv_pos_h'].to_numpy() == 1,
        'hiv_neg': df['hiv_pos_h'].to_numpy() == 0,
        'sex_m': df['contact_sex_m'].to_numpy() == 1,
        'sex_f': df['contact_sex_m'].to_numpy() == 0,
    }
    masks = {k: np.asarray(v, dtype=bool) for k, v in masks.items()}

    seeds = []
    for s in range(N_SEEDS):
        seeds.append(seed_full(s, df, y, groups))
        print('seed %2d fitted (%ds)' % (s, time.time() - t0))

    # ---- 锚臂 gate：逐种子位级核验（placebo 归档 per_seed 表）----
    arm_names = ['ind', 'exposure', 'prior_only', 'exposure_prior',
                 'dep_eb', 'dep_fixed', 'exposure_future', 'dep_future',
                 'lr_exposure', 'lr_exposure_eb']
    arm_stats = {}
    for a in arm_names:
        vals = np.array([_auroc(y, r['oof'][a]) for r in seeds])
        arm_stats[a] = {'mean': float(vals.mean()), 'sd': float(vals.std()),
                        'per_seed': [round(float(v), 4) for v in vals]}
    placebo_path = os.path.join(HERE, 'processed',
                                'sop_future_placebo_homeacf_20260925.json')
    gate = {'per_seed_check': {}, 'mean_check': {}}
    ok = True
    if os.path.exists(placebo_path):
        with open(placebo_path, encoding='utf-8') as f:
            arc = json.load(f)
        ps_arc = {r['seed']: r for r in arc['per_seed_arm_auroc']}
        for r in seeds:
            s = r['seed']
            if s not in ps_arc:
                continue
            row = {}
            for a in ANCHOR:
                this = arm_stats[a]['per_seed'][s]
                exp = ps_arc[s][a]
                row[a] = {'this': this, 'archive': exp,
                          'passed': bool(abs(this - exp) <= 0.0005)}
                ok &= row[a]['passed']
            gate['per_seed_check'][s] = row
            print('seed %2d per-seed gate: %s' % (
                s, {a: v['passed'] for a, v in row.items()}))
    if not SMOKE:                     # 均值门仅全量模式有效（N=20）
        for a, target in ANCHOR.items():
            d = arm_stats[a]['mean'] - target
            gate['mean_check'][a] = {
                'this_run': round(arm_stats[a]['mean'], 4),
                'anchor': target, 'diff': round(float(d), 4),
                'passed': bool(abs(d) < 0.002)}
            ok &= abs(d) < 0.002
        print('mean gate:', {k: v['passed']
                              for k, v in gate['mean_check'].items()})
    if not ok:
        raise SystemExit('锚臂复现失败，中止归档')

    # ---- 全队列效应（20 种子逐种子配对差 → 均值）----
    au_contrasts = [
        ('ep_minus_exp', 'exposure_prior', 'exposure', None),
        ('ep_minus_ind', 'exposure_prior', 'ind', None),
        ('prior_only_minus_ind', 'prior_only', 'ind', None),
        ('lr_eb_minus_lr', 'lr_exposure_eb', 'lr_exposure', None),
        ('ep_minus_lr_eb', 'exposure_prior', 'lr_exposure_eb', None),
        ('fut_minus_exp', 'exposure_future', 'exposure', None),
        ('fut_minus_prior', 'exposure_future', 'exposure_prior', None),
        ('dep_fut_minus_dep_fixed', 'dep_future', 'dep_fixed', None),
        ('sub_age_lt15', 'exposure_prior', 'exposure', 'age_lt15'),
        ('sub_age_ge15', 'exposure_prior', 'exposure', 'age_ge15'),
        ('sub_hiv_pos', 'exposure_prior', 'exposure', 'hiv_pos'),
        ('sub_hiv_neg', 'exposure_prior', 'exposure', 'hiv_neg'),
        ('sub_sex_m', 'exposure_prior', 'exposure', 'sex_m'),
        ('sub_sex_f', 'exposure_prior', 'exposure', 'sex_f'),
    ]

    def per_seed_au_delta(r, a, b, m):
        if m is None:
            return _auroc(y, r['oof'][a]) - _auroc(y, r['oof'][b])
        return (_auroc(y[masks[m]], r['oof'][a][masks[m]])
                - _auroc(y[masks[m]], r['oof'][b][masks[m]]))

    effects = {}
    for name, a, b, m in au_contrasts:
        d = np.array([per_seed_au_delta(r, a, b, m) for r in seeds])
        effects[name] = {
            'arms': [a, b], 'mask': m,
            'mean': float(d.mean()), 'sd': float(d.std()),
            'share_positive': float((d > 0).mean()),
            'min': float(d.min()), 'max': float(d.max()),
        }
    print('AUROC-domain effects done (%ds)' % (time.time() - t0))

    # ---- 级联/静态 全队列逐种子配对差 ----
    def per_seed_captures(r):
        rng = np.random.default_rng(1000 + r['seed'])
        o = r['oof']
        c_rf = bat.cascade_once(o['exposure'], y, groups, r['pi_row'],
                                Q_MAIN, K_MAIN, 'exposure_top', rng)[0]
        c_lr = scn.cascade_once_lr(o['lr_exposure'], y, groups,
                                   r['pi_row'], Q_MAIN, K_MAIN,
                                   'exposure_top', rng)[0]
        c_st = scn.static_capture(o['exposure'], y, K_MAIN)
        c_lreb = scn.static_capture(o['lr_exposure_eb'], y, K_MAIN)
        c_ep = scn.static_capture(o['exposure_prior'], y, K_MAIN)
        return {'c_rf': c_rf, 'c_lr': c_lr, 'c_st': c_st,
                'c_lreb': c_lreb, 'c_ep': c_ep}

    caps = [per_seed_captures(r) for r in seeds]
    cas_contrasts = [
        ('rf_cascade_minus_static', 'c_rf', 'c_st'),
        ('lr_cascade_minus_static', 'c_lr', 'c_st'),
        ('lr_eb_static_minus_static', 'c_lreb', 'c_st'),
        ('rf_minus_lr_cascade', 'c_rf', 'c_lr'),
    ]
    for name, ka, kb in cas_contrasts:
        d = np.array([c[ka] - c[kb] for c in caps])
        effects[name] = {
            'arms': [ka, kb], 'mask': None,
            'mean': float(d.mean()), 'sd': float(d.std()),
            'share_positive': float((d > 0).mean()),
            'min': float(d.min()), 'max': float(d.max()),
        }
    print('cascade-domain effects done (%ds)' % (time.time() - t0))

    # ---- DCA / Brier 全队列效应（crossfit 主，cal 副）----
    pt_grid = list(rcl.PT_GRID)
    dca_eff = {v: {'crossfit': {}, 'cal': {}} for v in pt_grid}
    for pt in pt_grid:
        d_cr = np.array([rcl.net_benefit(y, r['oof']['p_cross_ep'], pt)
                         - rcl.net_benefit(y, r['oof']['p_cross_exp'], pt)
                         for r in seeds])
        d_ca = np.array([rcl.net_benefit(y, r['oof']['p_cal_ep'], pt)
                         - rcl.net_benefit(y, r['oof']['p_cal_exp'], pt)
                         for r in seeds])
        for tag, d in (('crossfit', d_cr), ('cal', d_ca)):
            dca_eff[pt][tag] = {
                'mean': float(d.mean()), 'sd': float(d.std()),
                'share_positive': float((d > 0).mean()),
                'min': float(d.min()), 'max': float(d.max()),
            }
    brier_cr = np.array([float(np.mean(
        (r['oof']['p_cross_ep'] - y) ** 2
        - (r['oof']['p_cross_exp'] - y) ** 2)) for r in seeds])
    brier_ca = np.array([float(np.mean(
        (r['oof']['p_cal_ep'] - y) ** 2
        - (r['oof']['p_cal_exp'] - y) ** 2)) for r in seeds])
    brier_eff = {
        'crossfit': {'mean': float(brier_cr.mean()),
                     'sd': float(brier_cr.std()),
                     'share_positive': float((brier_cr < 0).mean()),
                     'min': float(brier_cr.min()),
                     'max': float(brier_cr.max())},
        'cal': {'mean': float(brier_ca.mean()), 'sd': float(brier_ca.std()),
                'share_positive': float((brier_ca < 0).mean()),
                'min': float(brier_ca.min()), 'max': float(brier_ca.max())},
    }
    # Brier 分量与曲线均值（点估计）
    brier_detail = {}
    for tag, ka, kb in (('crossfit', 'p_cross_ep', 'p_cross_exp'),
                       ('cal', 'p_cal_ep', 'p_cal_exp')):
        row = {}
        for lbl, k in (('ep', ka), ('exp', kb)):
            bs = [float(np.mean((r['oof'][k] - y) ** 2)) for r in seeds]
            dec = [rcl.brier_decomp(y, r['oof'][k]) for r in seeds]
            row[lbl] = {
                'brier_mean': float(np.mean(bs)),
                'brier_sd': float(np.std(bs)),
                'res_mean': float(np.mean([d['res'] for d in dec])),
                'rel_mean': float(np.mean([d['rel'] for d in dec])),
                'disp_mean': float(np.mean([d['disp'] for d in dec])),
            }
        brier_detail[tag] = row
    nb_curves = {str(pt): {} for pt in pt_grid}
    for pt in pt_grid:
        row = {'treat_all': rcl.nb_treat_all(y, pt)}
        for tag, ka, kd in (('crossfit', 'p_cross_ep', 'p_cross_dep'),
                            ('cal', 'p_cal_ep', None)):
            row[tag] = {
                'ep': float(np.mean([rcl.net_benefit(
                    y, r['oof'][ka], pt) for r in seeds])),
                'exp': float(np.mean([rcl.net_benefit(
                    y, r['oof']['p_cross_exp' if tag == 'crossfit'
                                else 'p_cal_exp'], pt) for r in seeds])),
            }
            if kd is not None:
                row[tag]['dep'] = float(np.mean([rcl.net_benefit(
                    y, r['oof'][kd], pt) for r in seeds]))
        nb_curves[str(pt)] = row
    print('DCA/Brier effects done (%ds)' % (time.time() - t0))

    # ---- 统一 bootstrap：Δ̄ 的户级 cluster CI ----
    hh = np.unique(groups)
    member_idx = {h: np.where(groups == h)[0] for h in hh}
    boot = {name: [] for name, _, _, _ in au_contrasts}
    for name, _, _ in cas_contrasts:
        boot[name] = []
    boot_dca = {pt: {'crossfit': [], 'cal': []} for pt in pt_grid}
    boot_brier = {'crossfit': [], 'cal': []}

    for b in range(N_BOOTSTRAP):
        rng_b = np.random.default_rng(BOOT_SEED + b)
        hh_s = rng_b.choice(hh, size=len(hh), replace=True)
        idx = np.concatenate([member_idx[h] for h in hh_s])
        yb, gb = y[idx], groups[idx]
        for name, a, bm, m in au_contrasts:
            if m is None:
                vals = [_auroc(yb, r['oof'][a][idx])
                        - _auroc(yb, r['oof'][bm][idx]) for r in seeds]
            else:
                mi = masks[m][idx]
                vals = [_auroc(yb[mi], r['oof'][a][idx][mi])
                        - _auroc(yb[mi], r['oof'][bm][idx][mi])
                        for r in seeds]
            boot[name].append(float(np.nanmean(vals))
                              if np.isfinite(np.nanmean(vals)) else float('nan'))
        # 级联逐种子差 → 20 种子均值（同一重抽样本套用于全部种子）
        cas_rows = []
        for r in seeds:
            pi_b = r['pi_row'][idx]
            o = {k: r['oof'][k][idx] for k in
                 ('exposure', 'lr_exposure', 'lr_exposure_eb')}
            c_rf = bat.cascade_once(o['exposure'], yb, gb, pi_b,
                                    Q_MAIN, K_MAIN, 'exposure_top', rng_b)[0]
            c_lr = scn.cascade_once_lr(o['lr_exposure'], yb, gb, pi_b,
                                       Q_MAIN, K_MAIN, 'exposure_top',
                                       rng_b)[0]
            c_st = scn.static_capture(o['exposure'], yb, K_MAIN)
            c_lreb = scn.static_capture(o['lr_exposure_eb'], yb, K_MAIN)
            cas_rows.append((c_rf, c_lr, c_st, c_lreb))
        col = {'c_rf': 0, 'c_lr': 1, 'c_st': 2, 'c_lreb': 3}
        for name, ka, kb in cas_contrasts:
            vals = [row[col[ka]] - row[col[kb]] for row in cas_rows]
            boot[name].append(float(np.mean(vals)))
        for pt in pt_grid:
            v_cr = [rcl.net_benefit(yb, r['oof']['p_cross_ep'][idx], pt)
                    - rcl.net_benefit(yb, r['oof']['p_cross_exp'][idx], pt)
                    for r in seeds]
            v_ca = [rcl.net_benefit(yb, r['oof']['p_cal_ep'][idx], pt)
                    - rcl.net_benefit(yb, r['oof']['p_cal_exp'][idx], pt)
                    for r in seeds]
            boot_dca[pt]['crossfit'].append(float(np.mean(v_cr)))
            boot_dca[pt]['cal'].append(float(np.mean(v_ca)))
        v_bc = [float(np.mean((r['oof']['p_cross_ep'][idx] - yb) ** 2
                              - (r['oof']['p_cross_exp'][idx] - yb) ** 2))
                for r in seeds]
        v_ba = [float(np.mean((r['oof']['p_cal_ep'][idx] - yb) ** 2
                              - (r['oof']['p_cal_exp'][idx] - yb) ** 2))
                for r in seeds]
        boot_brier['crossfit'].append(float(np.mean(v_bc)))
        boot_brier['cal'].append(float(np.mean(v_ba)))
        if (b + 1) % 100 == 0:
            print('bootstrap %d/%d (%ds)' % (b + 1, N_BOOTSTRAP,
                                             time.time() - t0))

    def ci_of(vals):
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
        if len(v) == 0:
            return None
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    for name in effects:
        effects[name]['ci95_unified'] = ci_of(boot[name])
        lo, hi = effects[name]['ci95_unified'] or [float('nan')] * 2
        effects[name]['ci_excludes_zero'] = bool(lo > 0 or hi < 0)
    dca_out = {}
    for pt in pt_grid:
        dca_out[str(pt)] = {
            'threshold': pt,
            'delta_crossfit': {
                **dca_eff[pt]['crossfit'],
                'ci95_unified': ci_of(boot_dca[pt]['crossfit'])},
            'delta_cal': {
                **dca_eff[pt]['cal'],
                'ci95_unified': ci_of(boot_dca[pt]['cal'])},
            'nb_curves': nb_curves[str(pt)],
        }
        for tag in ('crossfit', 'cal'):
            lo, hi = dca_out[str(pt)]['delta_%s' % tag]['ci95_unified']
            dca_out[str(pt)]['delta_%s' % tag]['ci_excludes_zero'] = bool(
                lo > 0 or hi < 0)
    for tag in ('crossfit', 'cal'):
        brier_eff[tag]['ci95_unified'] = ci_of(boot_brier[tag])
        lo, hi = brier_eff[tag]['ci95_unified']
        brier_eff[tag]['ci_excludes_zero'] = bool(lo > 0 or hi < 0)
    print('bootstrap done (%ds)' % (time.time() - t0))

    # ---- 亚组预算读数（K=10%：捕获/漏检/O:E/ECE/AUROC，描述性）----
    subgroup_table = {}
    for mname, m in masks.items():
        n_ev_m = int(y[m].sum())
        row = {'n': int(m.sum()), 'n_events': n_ev_m}
        for arm_lbl in ('ep_static', 'exp_static', 'cascade_rf',
                        'cascade_lr'):
            caps_m, missed_m = [], []
            for r in seeds:
                rng = np.random.default_rng(1000 + r['seed'])
                if arm_lbl == 'ep_static':
                    picked = picked_static(r['oof']['exposure_prior'], K_MAIN)
                elif arm_lbl == 'exp_static':
                    picked = picked_static(r['oof']['exposure'], K_MAIN)
                elif arm_lbl == 'cascade_rf':
                    picked = picked_cascade_rf(r['oof']['exposure'], y,
                                               groups, r['pi_row'],
                                               Q_MAIN, K_MAIN)
                else:
                    picked = picked_cascade_lr(r['oof']['lr_exposure'], y,
                                               groups, r['pi_row'],
                                               Q_MAIN, K_MAIN)
                captured = float(y[picked & m].sum())
                caps_m.append(captured / n_ev_m if n_ev_m else float('nan'))
                missed_m.append(n_ev_m - captured)
            row[arm_lbl] = {
                'capture_mean': float(np.mean(caps_m)),
                'missed_mean': float(np.mean(missed_m)),
            }
        oe = [float(np.mean(r['oof']['p_cross_ep'][m])
                    / max(np.mean(y[m]), 1e-12)) for r in seeds]
        ece = [ece_10bin(y[m], r['oof']['p_cross_ep'][m]) for r in seeds]
        row['oe_ratio_crossfit_ep'] = float(np.mean(oe))
        row['ece_crossfit_ep'] = float(np.mean(ece))
        # 亚组两臂 AUROC 均值（与 Δ̄ 同估计量族）
        row['auroc_ep'] = float(np.mean(
            [_auroc(y[m], r['oof']['exposure_prior'][m]) for r in seeds]))
        row['auroc_exp'] = float(np.mean(
            [_auroc(y[m], r['oof']['exposure'][m]) for r in seeds]))
        subgroup_table[mname] = row
    print('subgroup table done (%ds)' % (time.time() - t0))

    result = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'sop_unified_ci',
        'status': 'v0.1(2) 外审第一优先项：全稿统一主口径'
                  '（20 种子均值配对差 + 匹配户级 bootstrap CI）；'
                  '末种子 pooled 读数降级为补充',
        'design': {
            'estimator': 'Δ̄ = mean over 20 seeds of per-seed paired '
                         'differences; CI = household-cluster bootstrap of '
                         'Δ̄ (resample 877 households with replacement, '
                         'same resample applied to all 20 seeds, '
                         'percentile 2.5/97.5)',
            'n_seeds': N_SEEDS, 'n_bootstrap': N_BOOTSTRAP,
            'bootstrap_seed': BOOT_SEED,
            'q_main': Q_MAIN, 'k_main': K_MAIN,
            'dca_pt_grid': pt_grid,
            'dca_primary_variant': 'crossfit Platt（训练折拟合，测试折出值）',
            'anchor_gate': gate,
            'superseded_note': '旧口径（点估计=20 种子均值、CI=末种子 '
                               'pooled 户级 bootstrap）读数保留于 '
                               'sop_future_placebo_homeacf_20260925.json '
                               '与 sop_scenario_ci_homeacf_20260928.json，'
                               'v0.2 稿起不再入正文',
        },
        'arm_stats': arm_stats,
        'effects': effects,
        'dca': dca_out,
        'brier': {'delta': brier_eff, 'detail': brier_detail},
        'subgroup_budget_table': subgroup_table,
        'smoke': SMOKE,
        'runtime_sec': round(time.time() - t0, 1),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print('归档:', OUT)

    print('\n=== 统一口径主读数（20 种子均值 Δ̄ + 匹配户级 CI）===')
    for name in ('ep_minus_exp', 'ep_minus_ind', 'prior_only_minus_ind',
                 'lr_eb_minus_lr', 'ep_minus_lr_eb', 'fut_minus_exp',
                 'fut_minus_prior', 'dep_fut_minus_dep_fixed',
                 'rf_cascade_minus_static', 'lr_cascade_minus_static',
                 'lr_eb_static_minus_static', 'rf_minus_lr_cascade'):
        e = effects[name]
        print('  %-26s Δ̄=%+.4f CI[%+.4f,%+.4f] share=%.2f' % (
            name, e['mean'], e['ci95_unified'][0], e['ci95_unified'][1],
            e['share_positive']))
    for mname in ('sub_age_lt15', 'sub_age_ge15', 'sub_hiv_pos',
                  'sub_hiv_neg', 'sub_sex_m', 'sub_sex_f'):
        e = effects[mname]
        print('  %-26s Δ̄=%+.4f CI[%+.4f,%+.4f] share=%.2f' % (
            mname, e['mean'], e['ci95_unified'][0], e['ci95_unified'][1],
            e['share_positive']))


if __name__ == '__main__':
    main()
