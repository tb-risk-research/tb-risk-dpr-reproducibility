#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无泄漏主口径统一重算（2026-09-29，JCE v1(1) 外审附带项：折内化升主分析）。

背景：v1 管线四处使用全队列估计——
  (a) BMI 缺失 16 行以全队列中位（20.2）填补；
  (b) prior_rate 在 prior_n=0 行（户内首位）回退全队列阳性率；
  (c) future_rate 在 future_n=0 行（户内末位）回退全队列阳性率；
  (d) dep_fixed / dep_future 的 π̂ 用全队列率（prior_n=0 行 r̂=π̂ 完全
      抵消，π 仅经收缩项 (pos−nπ)/(n+k) 影响 n>0 行）。
battery B①（sop_robustness_battery_homeacf_20260927.json）已验证 (a)+(b)
折内化后 3 主臂均值位级不变。本脚本把折内管线升为**主口径**，覆盖
unified CI 全部读数（10 臂 + 18 效应 + DCA + Brier + 级联 + 亚组预算），
估计量与 run_sop_unified_ci.py 完全一致（Δ̄ = 20 种子逐种子配对差均值；
CI = Δ̄ 的户级 cluster bootstrap ×2000，bootstrap_seed=70000 与 v1 相同
→ 重抽样本逐位相同，两归档严格配对可差分）。

折内化规则（每折用训练折统计量）：
  bmi_h NaN → 训练折中位；prior_rate/future_rate NaN（n=0 回退位）→
  训练折阳性率；其余 NaN → 0（v1 nan_to_num 口径）。
  dep_fixed/dep_future 的 π̂ → 训练折阳性率（逐行向量）。
  prior_rate_eb / dep_eb / 级联 π̂ 本就折内（v1 已合规，不动）。
  Platt：crossfit 主读数本就折内；cal 副读数为同队列入档口径（不动）。

gate：3 主臂对照 battery B① 折内臂（tol 0.002，硬门）；全 10 臂对照
v1 unified 归档均值（非 dep 臂 tol 0.005；dep_fixed/dep_future 因 π̂ 改
折内允许 tol 0.010；均为硬门）+ 逐种子差 MAD（描述性）。

v1 全队列回退版读数保留于 sop_unified_ci_homeacf_20260928.json，降级为
敏感性归档；本归档为主口径。

用法：
    python data/run_sop_unified_ci_leakfree.py           # 全量 20 种子 × 2000
    python data/run_sop_unified_ci_leakfree.py smoke     # 2 种子 × 200 冒烟
输出：
    data/processed/sop_unified_ci_leakfree_homeacf_YYYYMMDD.json
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
import run_sop_deepening as rsd             # noqa: E402
import run_sop_future_placebo as fut        # noqa: E402
import run_sop_clinical_layer as rcl        # noqa: E402
import run_sop_scenario_ci as scn           # noqa: E402
import run_sop_unified_ci as u              # noqa: E402  (helpers + 常量)

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
BOOT_SEED = u.BOOT_SEED                    # 70000，与 v1 相同 → 配对重抽
Q_MAIN, K_MAIN = u.Q_MAIN, u.K_MAIN        # 0.05, 0.10

OUT = os.path.join(HERE, 'processed',
                   'sop_unified_ci_leakfree_homeacf_%s.json'
                   % time.strftime('%Y%m%d'))
V1_REF = os.path.join(HERE, 'processed',
                      'sop_unified_ci_homeacf_20260928.json')
BAT_REF = os.path.join(HERE, 'processed',
                       'sop_robustness_battery_homeacf_20260927.json')

ARM_NAMES = ['ind', 'exposure', 'prior_only', 'exposure_prior',
             'dep_eb', 'dep_fixed', 'exposure_future', 'dep_future',
             'lr_exposure', 'lr_exposure_eb']

# 折内填补的目标列：bmi_h→训练折中位；回退率列→训练折阳性率
_IMPUTE_BMI = 'bmi_h'
_IMPUTE_RATE = ('prior_rate', 'future_rate')


# ------------------------------------------------------ 折内 OOF 生成器 --
def _foldclean_matrix(cols, folds, y, data_raw):
    """逐折生成填补后的特征矩阵列表（训练折统计量填补，v1 其余口径）。"""
    X_all = data_raw[cols].to_numpy(dtype=float)
    idx_special = [(j, c) for j, c in enumerate(cols)
                   if c == _IMPUTE_BMI or c in _IMPUTE_RATE]
    mats = []
    for tr, te in folds:
        Xf = X_all.copy()
        for j, c in idx_special:
            if c == _IMPUTE_BMI:
                fill = float(np.nanmedian(X_all[tr, j]))
            else:
                fill = float(y[tr].mean())
            col = Xf[:, j]
            col[np.isnan(col)] = fill
        Xf = np.nan_to_num(Xf, nan=0.0)
        mats.append((Xf, tr, te))
    return mats


def rf_oof_lf(cols, folds, y, data_raw, seed):
    """折内合法 RF OOF（BMI 中位与回退率均由训练折估计）。"""
    oof = np.zeros(len(y))
    for Xf, tr, te in _foldclean_matrix(cols, folds, y, data_raw):
        m = bat._make_model('random_forest', seed)
        m.fit(Xf[tr], y[tr])
        oof[te] = m.predict_proba(Xf[te])[:, 1]
    return oof


def lr_oof_lf(cols, folds, y, data_raw, seed):
    """折内合法 LR OOF（标准化 L2，同 bat.lr_oof 口径 + 折内填补）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros(len(y))
    for Xf, tr, te in _foldclean_matrix(cols, folds, y, data_raw):
        m = Pipeline([
            ('sc', StandardScaler()),
            ('lr', LogisticRegression(C=1.0, max_iter=2000))])
        m.fit(Xf[tr], y[tr])
        oof[te] = m.predict_proba(Xf[te])[:, 1]
    return oof


# ------------------------------------------------ 单种子全臂（折内口径） --
def seed_full_lf(seed, df, y, groups, bmi_raw):
    """10 臂 + Platt，全部折内合法（与 u.seed_full 同结构）。

    与 v1 的差异仅四处：BMI 折内中位、prior_rate 折内回退、future_rate
    折内回退、dep_fixed/dep_future 的 π̂ 折内率。folds/order/EB/种子与
    v1 逐位相同。
    """
    folds = _group_cv_indices(groups, y, n_splits=bat.N_SPLITS, seed=seed)
    order = assign_screening_order(df, seed=seed, mode='random')
    pf = prior_features(df, order)
    ff = fut.future_features(df, order)

    prior_n = pf['prior_n'].to_numpy(dtype=float)
    prior_pos = pf['prior_pos'].to_numpy(dtype=float)
    fut_n = ff['future_n'].to_numpy(dtype=float)
    fut_pos = ff['future_pos'].to_numpy(dtype=float)

    fold_of = np.zeros(len(y), dtype=int)
    eb_params = []
    pi_fold_row = np.zeros(len(y))
    for f, (tr, te) in enumerate(folds):
        fold_of[te] = f
        eb_params.append(rsd.eb_shrinkage_params(y[tr], groups[tr]))
        pi_fold_row[te] = float(y[tr].mean())
    pi_row = np.array([eb_params[f][0] for f in fold_of])
    k_row = np.array([eb_params[f][1] for f in fold_of])

    df_raw = df.copy()
    df_raw['bmi_h'] = bmi_raw
    data_raw = pd.concat([df_raw.reset_index(drop=True), pf, ff], axis=1)
    # n=0 回退位置 NaN → 折内填补；n>0 行与 v1 逐位相同
    data_raw['prior_rate'] = np.where(
        prior_n > 0, prior_pos / np.maximum(prior_n, 1.0), np.nan)
    data_raw['future_rate'] = np.where(
        fut_n > 0, fut_pos / np.maximum(fut_n, 1.0), np.nan)
    data_raw['prior_rate_eb'] = (prior_pos + k_row * pi_row) / (prior_n + k_row)

    cols_ind = inf_feature_columns('ind')
    cols_exp = inf_feature_columns('exposure')

    oof = {
        'ind': rf_oof_lf(cols_ind, folds, y, data_raw, seed),
        'exposure': rf_oof_lf(cols_exp, folds, y, data_raw, seed),
        'prior_only': rf_oof_lf(bat.COLS_PRIOR, folds, y, data_raw, seed),
        'exposure_prior': rf_oof_lf(cols_exp + bat.COLS_PRIOR, folds, y,
                                    data_raw, seed),
        'exposure_future': rf_oof_lf(cols_exp + fut.COLS_FUT, folds, y,
                                     data_raw, seed),
        'lr_exposure': lr_oof_lf(cols_exp, folds, y, data_raw, seed),
        'lr_exposure_eb': lr_oof_lf(cols_exp + ['prior_rate_eb'], folds, y,
                                    data_raw, seed),
    }
    # 部署公式臂（p_base = 折内 exposure OOF；π̂ 折内率）
    oof['dep_fixed'] = rsd.dep_scores(
        oof['exposure'], prior_n, prior_pos, pi_fold_row,
        rsd.DEP_K, rsd.DEP_W)
    oof['dep_eb'] = rsd.dep_scores(
        oof['exposure'], prior_n, prior_pos, pi_row, k_row, rsd.DEP_W)
    oof['dep_future'] = rsd.dep_scores(
        oof['exposure'], fut_n, fut_pos, pi_fold_row,
        rsd.DEP_K, rsd.DEP_W)

    oof['p_cross_ep'] = bat.platt_crossfit(oof['exposure_prior'], y, folds)
    oof['p_cross_exp'] = bat.platt_crossfit(oof['exposure'], y, folds)
    oof['p_cross_dep'] = bat.platt_crossfit(oof['dep_fixed'], y, folds)
    oof['p_cal_ep'] = rcl.platt(oof['exposure_prior'], y)
    oof['p_cal_exp'] = rcl.platt(oof['exposure'], y)

    return {
        'seed': seed, 'y': y, 'groups': groups, 'oof': oof, 'folds': folds,
        'pi_row': pi_row, 'k_row': k_row,
        'prior_n': prior_n, 'prior_pos': prior_pos,
    }


# ------------------------------------------------------------------ main --
def main():
    t0 = time.time()
    df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    bmi_raw = bat.load_raw_bmi(df)
    print('HomeACF: n=%d events=%d households=%d (bmi missing %d)'
          % (len(y), int(y.sum()), len(np.unique(groups)),
             int(np.isnan(bmi_raw).sum())))

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
        seeds.append(seed_full_lf(s, df, y, groups, bmi_raw))
        print('seed %2d fitted (%ds)' % (s, time.time() - t0))

    # ---- gate：battery B① 折内臂（硬）+ v1 unified 均值差（硬）----
    arm_stats = {}
    for a in ARM_NAMES:
        vals = np.array([u._auroc(y, r['oof'][a]) for r in seeds])
        arm_stats[a] = {'mean': float(vals.mean()), 'sd': float(vals.std()),
                        'per_seed': [round(float(v), 4) for v in vals]}
    gate = {'battery_b1_foldclean_check': {}, 'v1_mean_diff': {},
            'v1_per_seed_mad': {}}
    ok = True
    with open(BAT_REF, encoding='utf-8') as f:
        b1_arms = json.load(f)[
            'b1_leakage_sensitivity']['arm_mean_auroc']
    for a in ('ind', 'exposure', 'exposure_prior'):
        d = arm_stats[a]['mean'] - b1_arms[a]
        gate['battery_b1_foldclean_check'][a] = {
            'this_run': round(arm_stats[a]['mean'], 4),
            'battery_b1': b1_arms[a], 'diff': round(float(d), 4),
            'passed': bool(abs(d) < 0.002)}
        if not SMOKE:            # 均值门仅全量模式有效（N=20 对 N=20）
            ok &= abs(d) < 0.002
    with open(V1_REF, encoding='utf-8') as f:
        v1 = json.load(f)
    for a in ARM_NAMES:
        v1m = v1['arm_stats'][a]['mean']
        d = arm_stats[a]['mean'] - v1m
        tol = 0.010 if a in ('dep_fixed', 'dep_future') else 0.005
        gate['v1_mean_diff'][a] = {
            'this_run': round(arm_stats[a]['mean'], 4),
            'v1_unified': round(v1m, 4), 'diff': round(float(d), 4),
            'tol': tol, 'passed': bool(abs(d) < tol)}
        if not SMOKE:            # smoke 为 2 种子均值，与 20 种子参考不可比
            ok &= abs(d) < tol
            mad = float(np.mean(np.abs(
                np.array(arm_stats[a]['per_seed'])
                - np.array(v1['arm_stats'][a]['per_seed']))))
            gate['v1_per_seed_mad'][a] = round(mad, 5)
    print('gate battery_b1:', {k: v['passed']
                               for k, v in gate['battery_b1_foldclean_check'].items()})
    print('gate v1_mean_diff:', {k: v['passed']
                                 for k, v in gate['v1_mean_diff'].items()})
    if not ok:
        raise SystemExit('折内臂 gate 失败，中止归档')

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
            return u._auroc(y, r['oof'][a]) - u._auroc(y, r['oof'][b])
        return (u._auroc(y[masks[m]], r['oof'][a][masks[m]])
                - u._auroc(y[masks[m]], r['oof'][b][masks[m]]))

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
                'min': float(brier_ca.min()),
                'max': float(brier_ca.max())},
    }
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

    # ---- 统一 bootstrap：Δ̄ 的户级 cluster CI（seed=70000 与 v1 配对）----
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
                vals = [u._auroc(yb, r['oof'][a][idx])
                        - u._auroc(yb, r['oof'][bm][idx]) for r in seeds]
            else:
                mi = masks[m][idx]
                vals = [u._auroc(yb[mi], r['oof'][a][idx][mi])
                        - u._auroc(yb[mi], r['oof'][bm][idx][mi])
                        for r in seeds]
            boot[name].append(float(np.nanmean(vals))
                              if np.isfinite(np.nanmean(vals)) else float('nan'))
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
                    picked = u.picked_static(r['oof']['exposure_prior'],
                                             K_MAIN)
                elif arm_lbl == 'exp_static':
                    picked = u.picked_static(r['oof']['exposure'], K_MAIN)
                elif arm_lbl == 'cascade_rf':
                    picked = u.picked_cascade_rf(r['oof']['exposure'], y,
                                                 groups, r['pi_row'],
                                                 Q_MAIN, K_MAIN)
                else:
                    picked = u.picked_cascade_lr(r['oof']['lr_exposure'], y,
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
        ece = [u.ece_10bin(y[m], r['oof']['p_cross_ep'][m]) for r in seeds]
        row['oe_ratio_crossfit_ep'] = float(np.mean(oe))
        row['ece_crossfit_ep'] = float(np.mean(ece))
        row['auroc_ep'] = float(np.mean(
            [u._auroc(y[m], r['oof']['exposure_prior'][m]) for r in seeds]))
        row['auroc_exp'] = float(np.mean(
            [u._auroc(y[m], r['oof']['exposure'][m]) for r in seeds]))
        subgroup_table[mname] = row
    print('subgroup table done (%ds)' % (time.time() - t0))

    result = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'sop_unified_ci_leakfree',
        'status': 'JCE v1(1) 外审附带项：折内管线升主口径'
                  '（BMI 折内中位 + prior/future 回退率折内 + dep π̂ 折内），'
                  '估计量与 sop_unified_ci_homeacf_20260928.json 完全一致',
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
            'leakfree_changes': [
                'bmi_h NaN → 训练折中位（v1：全队列中位 20.2，16 行）',
                'prior_rate 回退（prior_n=0）→ 训练折阳性率（v1：全队列率）',
                'future_rate 回退（future_n=0）→ 训练折阳性率（v1：全队列率）',
                'dep_fixed/dep_future 的 π̂ → 训练折阳性率（v1：全队列率；'
                'prior_n=0 行 r̂=π̂ 抵消，π 仅经收缩项影响 n>0 行）',
                'prior_rate_eb/dep_eb/级联 π̂/crossfit Platt：v1 已折内，不动',
            ],
            'anchor_gate': gate,
            'superseded_note': 'v1 全队列回退版（sop_unified_ci_homeacf_'
                               '20260928.json）降级为敏感性归档；'
                               'bootstrap_seed 相同 → 两归档重抽样本配对，'
                               '差分即折内化的纯效应',
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

    print('\n=== 折内主口径读数（20 种子均值 Δ̄ + 匹配户级 CI）===')
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
