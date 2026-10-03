#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOP 稳健性电池（2026-09-27）：外审响应 B①-B④ 四项重跑，一脚本一归档。

背景：v0.1 稿外审提出四项可计算批评，本脚本逐项以同口径重跑回应——

B①（预处理泄漏敏感性）：v1 管线两处非折内预处理——(a) BMI 缺失
  （16/2725）在 loader 内以**全队列中位**填补（分折前）；(b)
  prior_rate 在 prior_n=0 行回退到**全队列标签率**（y.mean()）。
  两处均为"用全队列（含测试折）信息进入特征"的口径缺陷。修复口径：
  逐折用训练折 BMI 中位 / 训练折事件率回填，其余与 v1 逐位相同
  （同 folds 同 screening order 同 RF 种子），直接对照主增益
  Δ(exposure_prior − exposure) 是否稳健。

B②（校准独立性）：clinical layer 的 Platt 重校准为**同队列**
  （全 OOF 拟合 → 同批评价），DCA/Brier 读数有乐观风险。修复口径：
  **交叉拟合 Platt**——同种子同折结构下，训练折拟合仿射参数、测试折
  出校准预测（户不相交 → 校准器从未见过该户）。报告 crossfit 与
  same-cohort 两层 Brier/DCA，差值 = 乐观量；预期（v1 docstring 已
  预声明）：乐观集中在 REL/NB 水平，臂间 ΔNB 方向稳健。

B③（级联全成本核算）：v1 capture 口径 = 静态 top-k 排序（假设先证
  信息免费），未计入"先查一部分人才能获得信息"的检查名额成本。
  修复口径：两阶段级联模拟——首波按 exposure 排序 top-q%（q∈{2,5,10}%，
  另设随机首波敏感性），首波结果按部署公式产生先证信息
  r̂=(pos₁+2π̂)/(n₁+2)（π̂=该行所属测试折的训练折率，k=2 冻结），
  第二波预算 = K·n − 首波名额（K∈{10,20,30,50}%），对剩余者按
  σ(logit(p_exp)+7·(r̂−π̂)) 重排取满预算。capture 计入首波检出与
  首波成本，对照同一总预算 K 下的 exposure 静态 top-K 与
  exposure_prior 静态 top-K（后者为排序指标，非成本核算协议），
  及随机靶向期望 K。

B④（简单模型公平对照）：审稿问"增益来自新增信息还是 SOP 的 RF 编码
  方式"。对照臂：L2 logistic（折内标准化）on cols_exp + 家庭 EB 收缩率
  prior_rate_eb（(pos+k̂π̂)/(n+k̂)，折内 EB 参数——同 deepening 口径）。
  判读：(a) lr_exposure_eb − lr_exposure ≈ RF 增益 → 增益主体为信息；
  (b) exposure_prior:RF − lr_exposure_eb ≈ 0（CI 含 0）→ RF/四列编码
  无额外溢价（与 deepening H-I1 加性栈证据互证）。

协议（与 v1/deepening 逐位同构）：HomeACF tst_pos10，n=2725/359 事件/
877 户，StratifiedGroupKFold(5) by record_id × 20 种子（0-19），random
筛查顺序，冻结注册表 RF；点估计 = 20 种子均值，配对 CI = 末种子 OOF
户级 cluster bootstrap ×2000；逐种子 Δ 四栏（R5a 纪律）。锚臂复现
gate：ind/exposure/exposure_prior 均值 |Δ|<0.002 vs v1 归档。

诚实边界：
  - 本电池为**事后稳健性分析**（post-hoc，回应外审），非预注册分析；
    预注册协议不覆盖 B①-B④ 任何一项（协议 grep placebo/DCA 零匹配
    的同款披露纪律）；
  - B① 修复的是"折内合法性"，不改变 v1 归档有效性的主张口径——
    v1 数字为"全队列预处理口径"的既成事实，本电池给出"折内口径"
    的平行数字与差值；
  - B② crossfit Platt 与部署"本地重校准"语义不同（回顾层无独立外
    样本），它修正的是评价层乐观性，不替代部署重校准；
  - B③ 级联为**回顾模拟**：首波 exposure-top / random 两种顺序假设，
    真实筛查顺序未记录（v1 同款边界）；预算语义 = 检查名额（非花费/
    QALY）；
  - B④ LR 用 v1 预处理口径（BMI 已填补）与标准管线对照，未叠加 B①
    折内化（隔离变量：B④ 只回答编码 vs 信息）。

用法：
    python data/run_sop_robustness_battery.py          # 全量 20 种子
    python data/run_sop_robustness_battery.py smoke    # 2 种子×200 bootstrap 冒烟
输出：
    data/processed/sop_robustness_battery_homeacf_YYYYMMDD.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))
sys.path.insert(0, HERE)

import run_sop_deepening as rsd  # noqa: E402  (_logit/eb/dep_scores/rf_oof)
import run_sop_seir_bridge as b1  # noqa: E402  (capture_at_k 口径同一性)
import run_sop_clinical_layer as rcl  # noqa: E402  (brier/DCA/Platt 工具)

from tb_risk.validation.household_temporal import (  # noqa: E402
    assign_screening_order,
    prior_features,
)
from tb_risk.validation.real_data_infection import (  # noqa: E402
    _RDA_PATH,
    feature_columns as inf_feature_columns,
    load_homeacf_contacts,
    warnings_catch,
)
from tb_risk.validation.real_data_pi import (  # noqa: E402
    _cluster_bootstrap_delta,
    _group_cv_indices,
    _make_model,
)

SMOKE = (len(sys.argv) > 1 and sys.argv[1] == 'smoke')
N_SEEDS = 2 if SMOKE else 20
N_BOOTSTRAP = 200 if SMOKE else 2000
SEED_START = 0
N_SPLITS = 5
DEP_W, DEP_K = 7.0, 2.0            # v1 网格校准（temporal_deployment 冻结）
Q_GRID = (0.02, 0.05, 0.10)        # 首波比例
K_GRID = (0.10, 0.20, 0.30, 0.50)  # 总预算
W1_MODES = ('exposure_top', 'random')
PT_GRID = rcl.PT_GRID
ANCHOR = {'ind': 0.6428, 'exposure': 0.6441, 'exposure_prior': 0.7142}
COLS_PRIOR = ['prior_n', 'prior_pos', 'prior_rate', 'prior_screened']

OUT = os.path.join(
    HERE, 'processed',
    'sop_robustness_battery_homeacf_%s.json' % time.strftime('%Y%m%d'))


# --------------------------------------------------------------- 标准管线 --

def run_seed_std(seed, df, y, groups):
    """单种子标准 OOF（v1 口径逐位复刻，3 锚臂 + EB 参数 + 折表）。

    与 deepening run_seed 的 ind/exposure/exposure_prior 三臂逐位同构
    （同 folds/同 order/同 rf_oof），锚 gate |Δ|<0.002 验证。
    """
    folds = _group_cv_indices(groups, y, n_splits=N_SPLITS, seed=seed)
    order = assign_screening_order(df, seed=seed, mode='random')
    pf = prior_features(df, order)
    data = pd.concat([df.reset_index(drop=True), pf], axis=1)

    fold_of = np.zeros(len(y), dtype=int)
    eb_params = []
    for f, (tr, te) in enumerate(folds):
        fold_of[te] = f
        eb_params.append(rsd.eb_shrinkage_params(y[tr], groups[tr]))
    pi_row = np.array([eb_params[f][0] for f in fold_of])
    k_row = np.array([eb_params[f][1] for f in fold_of])

    cols_ind = inf_feature_columns('ind')
    cols_exp = inf_feature_columns('exposure')
    oof = {
        'ind': rsd.rf_oof(cols_ind, folds, y, data, seed),
        'exposure': rsd.rf_oof(cols_exp, folds, y, data, seed),
        'exposure_prior': rsd.rf_oof(cols_exp + COLS_PRIOR, folds, y,
                                      data, seed),
    }
    return {
        'seed': seed, 'y': y, 'groups': groups, 'oof': oof, 'folds': folds,
        'pi_row': pi_row, 'k_row': k_row,
        'prior_n': pf['prior_n'].to_numpy(dtype=float),
        'prior_pos': pf['prior_pos'].to_numpy(dtype=float),
    }


# ------------------------------------------------- B① 折内预处理管线 --

def rf_oof_foldclean(cols, folds, y, data_raw, seed):
    """折内合法 RF OOF：BMI 中位与 base_rate 均由训练折估计。

    data_raw 与 v1 data 的唯一差异：(a) bmi_h 保留 NaN（原始值）；
    (b) prior_rate 在 prior_n=0 行为 NaN（回退率由折内填补）。其余列
    逐位相同；非目标列的 NaN 仍按 v1 口径 nan_to_num(0.0)。
    """
    X_all = data_raw[cols].to_numpy(dtype=float)
    bmi_j = cols.index('bmi_h') if 'bmi_h' in cols else None
    pr_j = cols.index('prior_rate') if 'prior_rate' in cols else None
    oof = np.zeros(len(y))
    for tr, te in folds:
        Xf = X_all.copy()
        if bmi_j is not None:
            med = float(np.nanmedian(X_all[tr, bmi_j]))
            Xf[np.isnan(Xf[:, bmi_j]), bmi_j] = med
        if pr_j is not None:
            base_rate = float(y[tr].mean())
            Xf[np.isnan(Xf[:, pr_j]), pr_j] = base_rate
        Xf = np.nan_to_num(Xf, nan=0.0)
        m = _make_model('random_forest', seed)
        m.fit(Xf[tr], y[tr])
        oof[te] = m.predict_proba(Xf[te])[:, 1]
    return oof


def run_seed_foldclean(seed, df, y, groups, bmi_raw):
    """单种子折内口径 3 臂（其余与标准管线逐位相同）。"""
    folds = _group_cv_indices(groups, y, n_splits=N_SPLITS, seed=seed)
    order = assign_screening_order(df, seed=seed, mode='random')
    pf = prior_features(df, order)
    prior_n = pf['prior_n'].to_numpy(dtype=float)
    prior_pos = pf['prior_pos'].to_numpy(dtype=float)

    df_raw = df.copy()
    df_raw['bmi_h'] = bmi_raw
    data_raw = pd.concat([df_raw.reset_index(drop=True), pf], axis=1)
    # prior_n=0 行回退率置 NaN → 折内填补；prior_n>0 行与 v1 逐位相同
    data_raw['prior_rate'] = np.where(
        prior_n > 0, prior_pos / np.maximum(prior_n, 1.0), np.nan)

    cols_ind = inf_feature_columns('ind')
    cols_exp = inf_feature_columns('exposure')
    oof = {
        'ind': rf_oof_foldclean(cols_ind, folds, y, data_raw, seed),
        'exposure': rf_oof_foldclean(cols_exp, folds, y, data_raw, seed),
        'exposure_prior': rf_oof_foldclean(
            cols_exp + COLS_PRIOR, folds, y, data_raw, seed),
    }
    return {'seed': seed, 'y': y, 'groups': groups, 'oof': oof}


def load_raw_bmi(df):
    """重解析 rda 取原始 BMI（NaN 保留），与 loader 过滤逐位对齐。"""
    import rdata
    with warnings_catch():
        conv = rdata.conversion.convert(
            rdata.parser.parse_file(_RDA_PATH))
    raw = conv['tstsa'].copy()
    raw.columns = [str(c) for c in raw.columns]
    diam = pd.to_numeric(raw['tstdiam_h'], errors='coerce')
    mask = diam.notna().to_numpy()
    bmi_raw = pd.to_numeric(raw['bmi_h'], errors='coerce')[
        mask].to_numpy(dtype=float)
    assert len(bmi_raw) == len(df), \
        '原始 BMI 与装载队列行数不一致：%d vs %d' % (len(bmi_raw), len(df))
    return bmi_raw


# --------------------------------------------------- B② 交叉拟合 Platt --

def platt_crossfit(p, y, folds):
    """交叉拟合 Platt：训练折拟合仿射参数，测试折出校准预测。

    与 rcl.platt 同为 logit 域 2 参数仿射（C=1e6），唯一差异是折内
    拟合——校准器从未见过测试行/测试户（户不相交折结构）。
    """
    from sklearn.linear_model import LogisticRegression
    x = rsd._logit(p)
    out = np.zeros(len(y))
    for tr, te in folds:
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(x[tr].reshape(-1, 1), y[tr])
        out[te] = lr.predict_proba(x[te].reshape(-1, 1))[:, 1]
    return out


# --------------------------------------------------------- B③ 级联 --

def cascade_once(oof_exp, y, groups, pi_row, q, K, w1_mode, rng):
    """两阶段级联单次：返回 (capture, nns, n_screened, n_with_info)。

    首波 top-q%（exposure 或随机）→ 结果经部署公式产生先证信息 →
    第二波按 σ(logit+7(r̂−π̂)) 重排取满总预算 K·n。capture 计入首波
    检出与首波名额成本。
    """
    n = len(y)
    n_events = int(y.sum())
    m1 = max(1, int(np.ceil(q * n)))
    w1 = np.zeros(n, dtype=bool)
    if w1_mode == 'exposure_top':
        w1[np.argsort(-oof_exp, kind='mergesort')[:m1]] = True
    else:
        w1[rng.choice(n, size=m1, replace=False)] = True

    sub = pd.DataFrame({'g': groups[w1], 'y': y[w1]})
    gn = sub.groupby('g').size()
    gp = sub.groupby('g')['y'].sum()
    n1 = pd.Series(groups).map(gn).fillna(0.0).to_numpy(dtype=float)
    pos1 = pd.Series(groups).map(gp).fillna(0.0).to_numpy(dtype=float)

    score2 = rsd.dep_scores(oof_exp, n1, pos1, pi_row, DEP_K, DEP_W)

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
    # 首波信息覆盖：第二波中获得同户先证信息（n1>0）的人数占比
    with_info = float((n1[~w1] > 0).mean()) if (~w1).any() else 0.0
    return capture, nns, n_screened, with_info


# ------------------------------------------------- B④ LR 简单模型对照 --

def lr_oof(cols, folds, y, data, seed):
    """折内标准化 L2 logistic OOF（简单对照臂，v1 预处理口径）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.nan_to_num(data[cols].to_numpy(dtype=float), nan=0.0)
    oof = np.zeros(len(y))
    for tr, te in folds:
        m = Pipeline([
            ('sc', StandardScaler()),
            ('lr', LogisticRegression(C=1.0, max_iter=2000))])
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


# ------------------------------------------------------------- 工具 --

def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y, dtype=int),
                               np.asarray(s, dtype=float)))


def arm_means(seeds, arms):
    return {a: round(float(np.mean(
        [_auroc(r['y'], r['oof'][a]) for r in seeds])), 4) for a in arms}


def per_seed_four_col(seeds, arm_a, arm_b):
    vals = np.asarray([_auroc(r['y'], r['oof'][arm_a])
                       - _auroc(r['y'], r['oof'][arm_b]) for r in seeds])
    return {'mean': round(float(vals.mean()), 4),
            'sd': round(float(vals.std()), 4),
            'min': round(float(vals.min()), 4),
            'max': round(float(vals.max()), 4),
            'share_positive': round(float((vals > 0).mean()), 3)}


def bootstrap_delta_last(seeds, arm_a, arm_b):
    last = seeds[-1]
    cb = _cluster_bootstrap_delta(
        last['oof'][arm_a], last['oof'][arm_b], last['y'], last['groups'],
        n_bootstrap=N_BOOTSTRAP, seed=SEED_START)
    return {'mean': round(cb['mean'], 4),
            'bootstrap_ci': [round(v, 4) for v in cb['bootstrap_ci']],
            'ci_excludes_zero': cb['ci_excludes_zero']}


# ------------------------------------------------------------- 主流程 --

def main():
    t0 = time.time()
    df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    n_events = int(y.sum())
    print('HomeACF: n=%d events=%d households=%d  π=%.4f'
          % (len(y), n_events, len(np.unique(groups)), y.mean()))

    bmi_raw = load_raw_bmi(df)
    n_bmi_missing = int(np.isnan(bmi_raw).sum())
    bmi_med_full = float(np.nanmedian(bmi_raw))
    print('B① 原始 BMI 缺失 %d/%d（全队列中位 %.2f，v1 口径）'
          % (n_bmi_missing, len(bmi_raw), bmi_med_full))

    # ---- 标准管线（20 种子）----
    std = []
    for s in range(SEED_START, SEED_START + N_SEEDS):
        std.append(run_seed_std(s, df, y, groups))
        print('std seed %2d done (%.0fs)' % (s, time.time() - t0))

    anchor_gate = {}
    for a, expect in ANCHOR.items():
        v = float(np.mean([_auroc(y, r['oof'][a]) for r in std]))
        anchor_gate[a] = {'this_run': round(v, 4), 'v1_archive': expect,
                          'reproduced': bool(abs(v - expect) < 0.002)}
    print('=== 锚臂 gate ===')
    for a, g in anchor_gate.items():
        print('  %-16s %.4f vs %.4f  %s'
              % (a, g['this_run'], g['v1_archive'],
                 'OK' if g['reproduced'] else 'FAIL'))

    # ================================================== B① 折内敏感性 ====
    print('\n=== B① 折内 BMI 中位 + 折内 base_rate（20 种子重跑）===')
    fc = []
    for s in range(SEED_START, SEED_START + N_SEEDS):
        fc.append(run_seed_foldclean(s, df, y, groups, bmi_raw))
        if (s - SEED_START) % 5 == 4 or SMOKE:
            print('  foldclean seed %2d done (%.0fs)'
                  % (s, time.time() - t0))
    b1_arm = arm_means(fc, ('ind', 'exposure', 'exposure_prior'))
    b1_delta = per_seed_four_col(fc, 'exposure_prior', 'exposure')
    b1_ci = bootstrap_delta_last(fc, 'exposure_prior', 'exposure')
    print('  折内口径臂均值:', b1_arm)
    print('  Δ(ep−e) 逐种子: %+.4f (share+ %.2f)  末种子 CI [%+.4f,%+.4f]'
          % (b1_delta['mean'], b1_delta['share_positive'],
             b1_ci['bootstrap_ci'][0], b1_ci['bootstrap_ci'][1]))
    print('  v1 参照: 臂 %s  Δ 末种子 bootstrap mean 0.0585 CI '
          '[0.0341, 0.0827]' % {a: ANCHOR[a] for a in ANCHOR})

    # ================================================ B② 交叉拟合校准 ====
    print('\n=== B② 交叉拟合 Platt（cal 乐观性修正）===')
    for r in std:
        r['cal_cf'] = {a: platt_crossfit(r['oof'][a], r['y'], r['folds'])
                       for a in ('exposure', 'exposure_prior')}
        r['cal_same'] = {a: rcl.platt(r['oof'][a], r['y'])
                         for a in ('exposure', 'exposure_prior')}

    b2 = {'brier': {}, 'dca': {}, 'optimism': {}}
    last = std[-1]
    for layer, key in (('cal_cf', 'crossfit'), ('cal_same', 'same_cohort')):
        ent = {}
        for a in ('exposure', 'exposure_prior'):
            per = [rcl.brier_decomp(y, r[layer][a]) for r in std]
            ent[a] = {k: round(float(np.mean([d[k] for d in per])), 6)
                      for k in ('brier', 'unc', 'res', 'rel', 'disp',
                                'scaled_brier')}
        # ΔBS(ep−e)：20 种子均值 + 末种子配对户级 bootstrap CI
        bs_p = (last[layer]['exposure_prior'] - y) ** 2
        bs_e = (last[layer]['exposure'] - y) ** 2
        ci = rcl.cluster_boot_mean_delta(bs_p, bs_e, groups,
                                         N_BOOTSTRAP, SEED_START)
        ent['delta_bs_ep_minus_e'] = {
            'per_seed': rcl.four_col(
                [rcl.brier_decomp(y, r[layer]['exposure_prior'])['brier']
                 - rcl.brier_decomp(y, r[layer]['exposure'])['brier']
                 for r in std]),
            'last_seed_ci': {'mean': round(ci['mean'], 6),
                             'bootstrap_ci': [round(v, 6)
                                              for v in ci['bootstrap_ci']],
                             'ci_excludes_zero': ci['ci_excludes_zero']}}
        b2['brier'][key] = ent

        dca = {}
        for pt in PT_GRID:
            row = {'nb_treat_all': round(rcl.nb_treat_all(y, pt), 5)}
            for a in ('exposure', 'exposure_prior'):
                nbs = [rcl.net_benefit(y, r[layer][a], pt) for r in std]
                row['nb_%s' % a] = {'mean': round(float(np.mean(nbs)), 5),
                                    'sd': round(float(np.std(nbs)), 5)}
            dnb = rcl.cluster_boot_mean_delta(
                rcl.nb_contrib(y, last[layer]['exposure_prior'], pt),
                rcl.nb_contrib(y, last[layer]['exposure'], pt),
                groups, N_BOOTSTRAP, SEED_START)
            row['delta_nb_ep_minus_e'] = {
                'last_seed_ci': {'mean': round(dnb['mean'], 5),
                                 'bootstrap_ci': [round(v, 5) for v in
                                                  dnb['bootstrap_ci']],
                                 'ci_excludes_zero':
                                     dnb['ci_excludes_zero']},
                'per_seed': rcl.four_col(
                    [rcl.net_benefit(y, r[layer]['exposure_prior'], pt)
                     - rcl.net_benefit(y, r[layer]['exposure'], pt)
                     for r in std])}
            dca['pt_%.3f' % pt] = row
        b2['dca'][key] = dca

    # 乐观量 = same − crossfit（ep 臂主读数层）
    b2['optimism']['bs_ep'] = rcl.four_col(
        [rcl.brier_decomp(y, r['cal_same']['exposure_prior'])['brier']
         - rcl.brier_decomp(y, r['cal_cf']['exposure_prior'])['brier']
         for r in std])
    b2['optimism']['rel_ep'] = {
        'same_cohort': b2['brier']['same_cohort']['exposure_prior']['rel'],
        'crossfit': b2['brier']['crossfit']['exposure_prior']['rel']}
    b2['optimism']['nb_ep'] = rcl.four_col(
        [rcl.net_benefit(y, r['cal_same']['exposure_prior'], pt)
         - rcl.net_benefit(y, r['cal_cf']['exposure_prior'], pt)
         for r in std for pt in (0.132,)])
    for key in ('crossfit', 'same_cohort'):
        e = b2['brier'][key]
        print('  Brier[%s] e=%.5f ep=%.5f  Δ(ep−e)=%+.5f share+=%.2f'
              % (key, e['exposure']['brier'],
                 e['exposure_prior']['brier'],
                 e['delta_bs_ep_minus_e']['per_seed']['mean'],
                 e['delta_bs_ep_minus_e']['per_seed']['share_positive']))
    for pt in PT_GRID:
        r_cf = b2['dca']['crossfit']['pt_%.3f' % pt]
        r_sm = b2['dca']['same_cohort']['pt_%.3f' % pt]
        print('  pt=%.3f  NB_cf ep=%+.4f e=%+.4f  CI_cf[%+.4f,%+.4f]  '
              'NB_same ep=%+.4f' % (
                  pt, r_cf['nb_exposure_prior']['mean'],
                  r_cf['nb_exposure']['mean'],
                  r_cf['delta_nb_ep_minus_e']['last_seed_ci']
                  ['bootstrap_ci'][0],
                  r_cf['delta_nb_ep_minus_e']['last_seed_ci']
                  ['bootstrap_ci'][1],
                  r_sm['nb_exposure_prior']['mean']))

    # ==================================================== B③ 级联 ====
    print('\n=== B③ 两阶段级联（首波成本计入总预算 K）===')
    n = len(y)
    b3 = {}
    for w1_mode in W1_MODES:
        for q in Q_GRID:
            for K in K_GRID:
                if K <= q:
                    continue
                caps, nnss, winfos = [], [], []
                cap_exp, cap_ep = [], []
                d_casc_exp, d_casc_ep = [], []
                for r in std:
                    rng = np.random.RandomState(10000 + r['seed'])
                    cap, nns, _, winfo = cascade_once(
                        r['oof']['exposure'], y, groups, r['pi_row'],
                        q, K, w1_mode, rng)
                    ce = b1.capture_at_k(r['oof']['exposure'], y, K)
                    cp = b1.capture_at_k(r['oof']['exposure_prior'],
                                          y, K)
                    caps.append(cap); nnss.append(nns)
                    winfos.append(winfo)
                    cap_exp.append(ce); cap_ep.append(cp)
                    d_casc_exp.append(cap - ce)
                    d_casc_ep.append(cap - cp)
                key = '%s_q%.2f_K%.2f' % (w1_mode, q, K)
                b3[key] = {
                    'q': q, 'K': K, 'w1_mode': w1_mode,
                    'n_budget_total': int(np.ceil(K * n)),
                    'capture_cascade': {
                        'mean': round(float(np.mean(caps)), 4),
                        'sd': round(float(np.std(caps)), 4)},
                    'nns_cascade': round(float(np.mean(nnss)), 2),
                    'wave2_with_prior_info_share': round(
                        float(np.mean(winfos)), 4),
                    'capture_exp_static': {
                        'mean': round(float(np.mean(cap_exp)), 4),
                        'sd': round(float(np.std(cap_exp)), 4)},
                    'capture_ep_static': {
                        'mean': round(float(np.mean(cap_ep)), 4),
                        'sd': round(float(np.std(cap_ep)), 4)},
                    'capture_random_expected': round(K, 4),
                    'nns_random': round(n / n_events, 2),
                    'delta_casc_minus_exp_static': _four(d_casc_exp),
                    'delta_casc_minus_ep_static': _four(d_casc_ep),
                }
    for w1_mode in W1_MODES:
        print('  [%s]' % w1_mode)
        for key, v in b3.items():
            if not key.startswith(w1_mode):
                continue
            print('    q=%.2f K=%.2f  capture=%+.4f  exp_static=%+.4f  '
                  'ep_static=%+.4f  Δ(casc−exp)=%+.4f share+=%.2f  NNS=%.2f'
                  % (v['q'], v['K'],
                     v['capture_cascade']['mean'],
                     v['capture_exp_static']['mean'],
                     v['capture_ep_static']['mean'],
                     v['delta_casc_minus_exp_static']['mean'],
                     v['delta_casc_minus_exp_static']['share_positive'],
                     v['nns_cascade']))

    # ================================================ B④ LR 对照臂 ====
    print('\n=== B④ 简单模型对照（L2 LR + 家庭 EB 率）===')
    cols_exp = inf_feature_columns('exposure')
    for r in std:
        seed = r['seed']
        order = assign_screening_order(df, seed=seed, mode='random')
        pf = prior_features(df, order)
        data = pd.concat([df.reset_index(drop=True), pf], axis=1)
        data['prior_rate_eb'] = (
            r['prior_pos'] + r['k_row'] * r['pi_row']) \
            / (r['prior_n'] + r['k_row'])
        r['oof']['lr_exposure'] = lr_oof(
            cols_exp, r['folds'], y, data, seed)
        r['oof']['lr_exposure_eb'] = lr_oof(
            cols_exp + ['prior_rate_eb'], r['folds'], y, data, seed)

    b4_arms = arm_means(std, ('lr_exposure', 'lr_exposure_eb',
                              'exposure', 'exposure_prior'))
    b4_c1 = {'per_seed': per_seed_four_col(std, 'lr_exposure_eb',
                                           'lr_exposure'),
             'last_seed': bootstrap_delta_last(std, 'lr_exposure_eb',
                                               'lr_exposure')}
    b4_c2 = {'per_seed': per_seed_four_col(std, 'exposure_prior',
                                           'lr_exposure_eb'),
             'last_seed': bootstrap_delta_last(std, 'exposure_prior',
                                               'lr_exposure_eb')}
    b4_c3 = {'per_seed': per_seed_four_col(std, 'exposure_prior',
                                           'exposure'),
             'last_seed': bootstrap_delta_last(std, 'exposure_prior',
                                               'exposure')}
    print('  臂均值:', b4_arms)
    print('  C1 lr_eb−lr_exp=%+.4f (share+ %.2f)  CI [%+.4f,%+.4f]'
          % (b4_c1['per_seed']['mean'], b4_c1['per_seed']['share_positive'],
             b4_c1['last_seed']['bootstrap_ci'][0],
             b4_c1['last_seed']['bootstrap_ci'][1]))
    print('  C2 RF(ep)−lr_eb=%+.4f (share+ %.2f)  CI [%+.4f,%+.4f]'
          % (b4_c2['per_seed']['mean'], b4_c2['per_seed']['share_positive'],
             b4_c2['last_seed']['bootstrap_ci'][0],
             b4_c2['last_seed']['bootstrap_ci'][1]))

    # ======================================================== 归档 ====
    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'sop_robustness_battery_homeacf_v1',
        'status': '外审响应稳健性电池（post-hoc，非预注册）：B① 折内'
                  '预处理敏感性 / B② 交叉拟合校准 / B③ 级联全成本 / '
                  'B④ 简单模型对照',
        'design': {
            'source': 'HomeACF（github petermacp/tstsa）',
            'endpoint': 'tst_pos10（TST≥10mm LTBI，基线横断面阳性）',
            'n': int(n), 'n_events': n_events,
            'n_households': int(len(np.unique(groups))),
            'n_seeds': N_SEEDS, 'seed_start': SEED_START,
            'smoke': bool(SMOKE),
            'cv': 'StratifiedGroupKFold(%d) by household' % N_SPLITS,
            'ci': '户级 cluster bootstrap ×%d（末种子 OOF/逐行贡献）'
                  % N_BOOTSTRAP,
            'dep_formula': 'σ(logit(p_exp)+%.1f·(r̂−π̂)), '
                           'r̂=(pos+k·π̂)/(n+k), k=%.1f（v1 网格冻结）'
                           % (DEP_W, DEP_K),
            'eb': 'k̂=(1−ICC)/ICC（训练折 MoM），π̂=训练折率（deepening '
                  '同款）',
            'anchor_gate': anchor_gate,
        },
        'b1_leakage_sensitivity': {
            'design_note': '折内 BMI 中位（缺失 %d/%d）+ 折内 base_rate'
                           '（prior_n=0 行）替代全队列估计；folds/order/'
                           'RF 种子与 v1 逐位相同' % (
                               n_bmi_missing, len(bmi_raw)),
            'n_bmi_missing': n_bmi_missing,
            'bmi_median_full_cohort': round(bmi_med_full, 2),
            'arm_mean_auroc': b1_arm,
            'v1_archive': ANCHOR,
            'delta_ep_minus_exposure': {
                'per_seed': b1_delta, 'last_seed_bootstrap': b1_ci},
            'v1_reference_delta': {
                'mean': 0.0585, 'bootstrap_ci': [0.0341, 0.0827]},
        },
        'b2_crossfit_calibration': b2,
        'b3_cascade': b3,
        'b4_lr_comparator': {
            'design_note': 'L2 LR（折内标准化，C=1.0）on cols_exp[+EB '
                           'prior_rate]；v1 预处理口径；判读：C1 同信息'
                           '简单模型增益 ≈ RF 增益 → 增益主体为信息；'
                           'C2 RF 编码溢价 ≈ 0 → 编码无额外价值',
            'arm_mean_auroc': b4_arms,
            'c1_lr_eb_minus_lr_exposure': b4_c1,
            'c2_rf_ep_minus_lr_eb': b4_c2,
            'c3_rf_ep_minus_rf_exposure': b4_c3,
        },
        'runtime_sec': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


def _four(vals):
    vals = np.asarray(vals, dtype=float)
    return {'mean': round(float(vals.mean()), 4),
            'sd': round(float(vals.std()), 4),
            'min': round(float(vals.min()), 4),
            'max': round(float(vals.max()), 4),
            'share_positive': round(float((vals > 0).mean()), 3)}


if __name__ == '__main__':
    main()
