#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""λ 联合校准 + 乘子审计 + 户级聚合（P1/P2/P3，2026-08-25）。

P1（最高优先，机制已就绪）：把 SEIR 联合学习（实验 1，seir_joint.py）
的可微管线搬到 HomeACF——λ 的每个乘子系数在 log 空间参数化、以文献
先验为起点，训练折上 BCE 损失 + Adam 优化（凸），测试折评估。
三臂裁决：
  frozen λ   手排文献系数（real_data_infection，AUROC 0.529）
  calibrated λ   结构来自物理、参数来自数据
  学习基线   ind/index/exposure RF（冻结注册表）
验收标准（用户原文）：校准后 λ 的 AUROC 从 0.529 显著抬升，且作为
特征不再产生负增益（exposure+cal λ − exposure ≥ 0，CI 下界不再 < 0）。

乘子分解（结构来自物理；smear 因 83% 缺失自主集移除，见 P2）：
  log λ = θ0 + θ_lt5·lt5 + θ_ge45·ge45 + θ_hiv·HIV_h
          + θ_cough·coughdays + θ_idxhiv·HIV_idx
          + θ_ts·log(timespent) + θ_bedroom·共用卧室 + θ_bed·同床
  起点即先验：θ_lt5=log(2.8)（1+1.8）、θ_ge45=log(2.6)、θ_hiv=log(1.3)、
  θ_cough=0.01/天、θ_idxhiv=log(1.3)、θ_ts=1.0、θ_bedroom=log(1.2)、
  θ_bed=log(1.5)——校准 = "先验出发的数据修正"，系数可直接回读为
  log-倍数（P2 审计口径）。

P2（物理先验体检报告）：逐乘子审计表——先验方向/先验刻度 vs 数据
方向（OOF 单乘子 AUROC + 单变量系数）/数据刻度（校准系数均值±sd，
折×种子聚合）/可检验性（smear known 仅 ~17% → untestable，从主 λ
移除并在 known 子集内做敏感性）。

P3（HomeACF 升格为暴露通路第一个真实载体）：户级聚合特征检验
channel_agg 发现在真实 LTBI 终点上是否成立——
  hh_mean_oof   户内 OOF 预测均值（分组 CV 保证同户同折、训练不
                含同户 → 无标签泄漏，部署可用）
  hh_in_cohort  户内队列观测接触者数
  hh_tst_rate_loo  户内 TST 阳性率 leave-one-out（标签聚合——
                科学证据口径：家庭聚集信号强度；部署不可用，
                docstring 与结果 JSON 显式声明）
臂：exposure（基线）vs exposure + hh_agg（+hh_mean_oof/hh_in_cohort）；
loo 聚集度单变量 AUROC 单独报告（聚集存在性证据）。

诚实边界：
  - 校准 λ 的"显著抬升"是相对 frozen λ 的排序改进，终点仍是
    LTBI（TST≥10mm）而非 TB 发病；
  - 校准系数的可辨识性受乘子间共线与效应稀释限制（每户共享指示
    病例变量 → 有效样本量 = 户级 877 而非行级 2725；户级 cluster
    bootstrap 已部分吸收，但系数 sd 为折-种子层面的欠保守估计）；
  - hh_tst_rate_loo 用了同户他人终点——绝不进部署臂；
  - smear 移除改变了 frozen 与 calibrated 的可比性：主对比双方均
    不含 smear（frozen 臂复算 smear-free 版本），敏感性臂报 known
    子集内 smear 方向。

第三轮 v2（2026-08-25 追加）：λ 精简为数据一致子集——移除乘子审计
三个方向翻转项（age_lt5/index_hiv/sleep_same_bed，见 _REMOVED_
COMPONENTS），保留方向一致 5 乘子（SIMPLIFIED_COMPONENTS）并增设
frozen/cal 精简臂与特征级精简臂。20 种子诚实判定：精简校准 λ
0.5742 vs 全乘子校准 0.5700（+0.0048，CI 含零——未达"显著超过"
验收线）；但 frozen 精简 0.5740 ≈ cal 精简 0.5742：移除翻转乘子
贡献了几乎全部提升（vs frozen 全乘子 0.557 → +0.017），可微刻度
校准在方向一致子集上边际增益 ≈ 0。结论：方向修正 > 刻度校准；
λ 物理信号弱（0.574）仍是主约束，真实增量在时序家庭先证特征
（household_temporal.py：exposure+prior 0.7142，+0.0585 显著）。

协议：StratifiedGroupKFold(5) by household × 多种子；户级 cluster
bootstrap CI；逐种子 DeLong（复用 real_data_pi 协议骨架）。
"""

import json
import os
import time

import numpy as np
import pandas as pd

from .layer_ablation import delong_paired_test
from .real_data_infection import (
    FEATURE_SETS as INF_FEATURE_SETS,
    _COUGH_MEDIAN_FILL,
    feature_columns as inf_feature_columns,
    load_homeacf_contacts,
)
from .real_data_pi import (
    MODEL_KEYS,
    _cluster_bootstrap_delta,
    _group_cv_indices,
    _make_model,
    _pr_auc,
    _recall_at_budget,
)
from .threshold_spec import compute_auc

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- P1 乘子分解 ----------------------------------------------------
# 主集（smear 移除）；起点 = 文献先验（log 空间）
MULTIPLIER_COMPONENTS = (
    'age_lt5',        # 哑元；先验 θ=log(2.8)
    'age_ge45',       # 哑元；先验 θ=log(2.6)
    'host_hiv',       # 哑元；先验 θ=log(1.3)
    'cough_days',     # 数值 0-365；先验 θ=0.01/天
    'index_hiv',      # 哑元；先验 θ=log(1.3)
    'log_ts',         # log(0.5/1.0/1.5)；先验权重 1.0
    'share_bedroom',  # 哑元；先验 θ=log(1.2)
    'sleep_same_bed', # 哑元；先验 θ=log(1.5)
)

PRIOR_START = {
    'age_lt5': np.log(2.8),
    'age_ge45': np.log(2.6),
    'host_hiv': np.log(1.3),
    'cough_days': 0.01,
    'index_hiv': np.log(1.3),
    'log_ts': 1.0,
    'share_bedroom': np.log(1.2),
    'sleep_same_bed': np.log(1.5),
}

# P2（第三轮）：λ 精简为"数据一致子集"——乘子审计发现 3 个方向
# 翻转项（age_lt5 / index_hiv / sleep_same_bed，校准系数为负），
# 从 λ 中移除；保留 5 个方向一致乘子，刻度由校准管线在训练折上
# 学习（"结构来自审计后的物理，参数来自数据"）。
SIMPLIFIED_COMPONENTS = (
    'cough_days',      # 先验 + / 数据 +（单变量 AUROC 0.544 最强）
    'host_hiv',        # 先验 + / 数据 +（刻度 1.78×）
    'share_bedroom',   # 先验 + / 数据 +（刻度 2.72× 被低估最重）
    'age_ge45',        # 先验 + / 数据 +（弱）
    'log_ts',          # 先验 + / 数据 +
)
_REMOVED_COMPONENTS = ('age_lt5', 'index_hiv', 'sleep_same_bed')

_PRIOR_LITERATURE = {
    'age_lt5': {'value': '×2.8（1+1.8，Marais 儿童进展）',
                'direction': '+'},
    'age_ge45': {'value': '×2.6（1+1.6，老年进展）', 'direction': '+'},
    'host_hiv': {'value': '×1.3（Andrews 荟萃量级）', 'direction': '+'},
    'cough_days': {'value': 'exp(0.01·天)（Verver 传染持续）',
                   'direction': '+'},
    'index_hiv': {'value': '×1.3', 'direction': '+'},
    'log_ts': {'value': '0.5/1.0/1.5 三级', 'direction': '+'},
    'share_bedroom': {'value': '×1.2', 'direction': '+'},
    'sleep_same_bed': {'value': '×1.5', 'direction': '+'},
    'smear': {'value': '×3.0（Grzybowski）', 'direction': '+',
              'note': '83% not done → 自主 λ 移除'},
}


def multiplier_matrix(df, include_smear=False, components=None):
    """乘子分量矩阵（结构列 + 可选 smear 敏感性列）。

    Args:
        components: 分量子集（P2 精简 λ 用 SIMPLIFIED_COMPONENTS）；
            None = 全部分量。

    Returns:
        (pd.DataFrame 分量矩阵, list[str] 分量名)
    """
    C = pd.DataFrame(index=df.index)
    C['age_lt5'] = df['age_lt5']
    C['age_ge45'] = df['age_ge45']
    C['host_hiv'] = df['hiv_pos_h']
    C['cough_days'] = df['idx_coughdays'].clip(0, 365)
    C['index_hiv'] = df['idx_hiv_pos']
    # 接触强度：先验三级 0.5/1.0/1.5 的 log（缺失已归中等级 → log(1.0)=0）
    ts_int = (0.5 * df['ts_low'] + 1.0 * df['ts_mid']
              + 1.5 * df['ts_high'])
    C['log_ts'] = np.log(ts_int.to_numpy(dtype=float))
    C['share_bedroom'] = df['share_bedroom']
    C['sleep_same_bed'] = df['sleep_same_bed']
    cols = list(MULTIPLIER_COMPONENTS)
    if include_smear:
        C['smear'] = df['idx_smear_pos']
        cols = cols + ['smear']
    if components is not None:
        cols = [c for c in components if c in cols]
    return C, cols


def calibrate_lambda_torch(C, y, train_idx, seed=0, epochs=1500,
                           lr=0.03, l2=1e-4):
    """可微校准（实验 1 管线）：BCE + Adam，log 空间乘子系数。

    系数以文献先验为起点（"先验出发的数据修正"），凸损失无局部
    极小；torch.set_num_threads(1) + manual_seed 保证确定性。

    Returns:
        dict: {'intercept': float, 分量名: float, 'loss_trace': list}
    """
    if torch is None:  # pragma: no cover
        raise ImportError('torch 不可用')
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))

    cols = list(C.columns)
    y = np.asarray(y, dtype=float)          # 防 pandas Series 索引歧义
    train_idx = np.asarray(train_idx, dtype=np.int64)
    X = torch.tensor(C[cols].to_numpy(dtype=np.float64),
                     dtype=torch.float64)
    yv = torch.tensor(np.asarray(y, dtype=float), dtype=torch.float64)
    tr = torch.as_tensor(np.asarray(train_idx, dtype=np.int64))

    start = [float(np.log(np.clip(np.mean(y[tr]), 0.02, 0.5) /
                          (1 - np.clip(np.mean(y[tr]), 0.02, 0.5))))]
    start += [float(PRIOR_START.get(c, 0.0)) for c in cols]
    params = [torch.tensor(v, dtype=torch.float64, requires_grad=True)
              for v in start]
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(params, lr=lr, weight_decay=l2)

    loss_trace = []
    for _ in range(epochs):
        opt.zero_grad()
        logit = params[0] + sum(
            params[j + 1] * X[:, j] for j in range(len(cols)))
        loss = bce(logit[tr], yv[tr])
        loss.backward()
        opt.step()
        loss_trace.append(float(loss.detach()))

    out = {'intercept': float(params[0].detach())}
    for j, c in enumerate(cols):
        out[c] = float(params[j + 1].detach())
    out['loss_trace'] = loss_trace[-5:]
    return out


def _oof_calibrated_lambda(df, y, folds, seed, include_smear=False,
                           components=None):
    """逐折训练校准、测试折打分 → 校准 λ 的 OOF logit + 系数表。"""
    C, cols = multiplier_matrix(df, include_smear=include_smear,
                                components=components)
    oof = np.zeros(len(y))
    coefs = []
    for tr, te in folds:
        fit = calibrate_lambda_torch(
            C.iloc[tr].reset_index(drop=True), y[tr],
            np.arange(len(tr)), seed=seed)
        b = np.array([fit[c] for c in cols], dtype=float)
        oof[te] = fit['intercept'] + \
            C.iloc[te][cols].to_numpy(dtype=float) @ b
        coefs.append({c: fit[c] for c in cols} | {'intercept': fit['intercept']})
    return oof, coefs, cols


# ---- P1/P3 三臂 + 户聚合阶梯 ----------------------------------------

def household_features(df, y, oof_ind_rf):
    """P3 户级聚合特征（防泄漏口径）。

    hh_mean_oof：户内 OOF 预测均值——分组 CV 下同户全在同一折、
    训练不含任何同户行，故均值不含自身或同户标签信息（部署可用）。
    hh_in_cohort：户内队列观测接触者数。
    hh_tst_rate_loo：户内 TST 阳性率 leave-one-out——用同户他人终点，
    仅作聚集存在性证据，不入部署臂。
    """
    g = df['record_id']
    hh_mean = pd.Series(oof_ind_rf).groupby(g).transform('mean')
    hh_n = g.groupby(g).transform('size').astype(float)
    y_s = pd.Series(y)
    hh_sum = y_s.groupby(g).transform('sum')
    loo = (hh_sum - y_s) / (hh_n - 1.0).clip(lower=1.0)
    loo = loo.where(hh_n > 1.0, other=float(np.mean(y)))  # 独户填全局率
    return pd.DataFrame({
        'record_id': g,
        'hh_mean_oof': hh_mean.to_numpy(dtype=float),
        'hh_in_cohort': hh_n.to_numpy(dtype=float),
        'hh_tst_rate_loo': loo.to_numpy(dtype=float),
    })


def run_calibration_ablation_once(seed=0, n_splits=5, df=None):
    """单种子：P1 三臂 + P3 户聚合臂，全部 OOF。"""
    if df is None:
        df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    folds = _group_cv_indices(groups, y, n_splits=n_splits, seed=seed)

    # ---- P1：校准 λ（OOF，smear-free）+ frozen λ（smear-free 复算）----
    cal_logit, cal_coefs, cal_cols = _oof_calibrated_lambda(
        df, y, folds, seed, include_smear=False)
    # frozen λ 复算（与 build_lambda_infection 同结构但移除 smear 项，
    # 保证与校准臂可比）
    C0, cols0 = multiplier_matrix(df, include_smear=False)
    frozen_logit = np.log(np.mean(y) / (1 - np.mean(y))) + sum(
        PRIOR_START[c] * C0[c].to_numpy(dtype=float) for c in cols0)

    # ---- P2：精简 λ（5 个数据一致乘子）----
    cal_logit_s, cal_coefs_s, cal_cols_s = _oof_calibrated_lambda(
        df, y, folds, seed, components=SIMPLIFIED_COMPONENTS)
    base_logit = np.log(np.mean(y) / (1 - np.mean(y)))
    frozen_logit_s = base_logit + sum(
        PRIOR_START[c] * C0[c].to_numpy(dtype=float)
        for c in SIMPLIFIED_COMPONENTS)

    # ---- 学习臂（冻结注册表 RF）----
    def _rf_oof(cols):
        X = np.nan_to_num(df[cols].to_numpy(dtype=float), nan=0.0)
        oof = np.zeros(len(y))
        for tr, te in folds:
            m = _make_model('random_forest', seed)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        return oof

    oof_ind = _rf_oof(inf_feature_columns('ind'))
    oof_exposure = _rf_oof(inf_feature_columns('exposure'))

    # ---- P3：户聚合（OOF ind 预测做户均值）----
    hh = household_features(df, y, oof_ind)
    cols_hh = list(inf_feature_columns('exposure')) + \
        ['hh_mean_oof', 'hh_in_cohort']
    df_hh = pd.concat(
        [df.reset_index(drop=True), hh[['hh_mean_oof', 'hh_in_cohort']]],
        axis=1)
    oof_exposure_hh = _rf_oof_df(df_hh, cols_hh, y, folds, seed)

    arms = {
        'frozen_pi_only': frozen_logit,
        'cal_pi_only': cal_logit,
        'frozen_simplified_pi_only': frozen_logit_s,
        'cal_simplified_pi_only': cal_logit_s,
        'ind:RF': oof_ind,
        'exposure:RF': oof_exposure,
        'exposure_hh:RF': oof_exposure_hh,
    }
    # 特征级：exposure + cal λ（cal λ OOF 作为 1 列特征）
    df_cal = pd.concat([df_hh.reset_index(drop=True),
                        pd.DataFrame({'cal_lambda': cal_logit})], axis=1)
    cols_calpi = list(inf_feature_columns('exposure')) + ['cal_lambda']
    arms['exposure_calpi:RF'] = _rf_oof_df(df_cal, cols_calpi, y, folds,
                                           seed)
    # P2 特征级：exposure + 精简校准 λ
    df_cals = pd.concat([df_hh.reset_index(drop=True),
                         pd.DataFrame(
                             {'cal_lambda_simplified': cal_logit_s})],
                        axis=1)
    cols_cals = list(inf_feature_columns('exposure')) + \
        ['cal_lambda_simplified']
    arms['exposure_calsimpl_pi:RF'] = _rf_oof_df(df_cals, cols_cals, y,
                                                 folds, seed)

    result = {'seed': seed, 'y': y, 'groups': groups, 'arms': {},
              'cal_coefs': cal_coefs, 'cal_cols': cal_cols,
              'cal_coefs_simplified': cal_coefs_s,
              'cal_cols_simplified': cal_cols_s}
    for name, oof in arms.items():
        result['arms'][name] = {
            'auroc': float(compute_auc(oof, y)),
            'pr_auc': _pr_auc(y, oof),
            'recall_at_budget': _recall_at_budget(oof, y),
        }
    # P3 描述性证据：loo 聚集度单变量（科学口径，部署不可用）
    result['arms']['hh_tst_rate_loo_only'] = {
        'auroc': float(compute_auc(hh['hh_tst_rate_loo'], y)),
        'pr_auc': _pr_auc(y, hh['hh_tst_rate_loo']),
        'recall_at_budget': _recall_at_budget(hh['hh_tst_rate_loo'], y),
        'note': '标签聚合（同户他人终点）——聚集存在性证据，部署不可用',
    }
    result['oof'] = arms
    result['design'] = {
        'n': int(len(y)), 'n_events': int(y.sum()),
        'n_households': int(pd.Series(groups).nunique()),
        'n_splits': n_splits, 'endpoint': 'tst_pos10（TST≥10mm LTBI）',
    }
    return result


def _rf_oof_df(frame, cols, y, folds, seed):
    X = np.nan_to_num(frame[cols].to_numpy(dtype=float), nan=0.0)
    oof = np.zeros(len(y))
    for tr, te in folds:
        m = _make_model('random_forest', seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


# ---- P2 乘子审计 -----------------------------------------------------

def multiplier_audit(df, y, folds, cal_coefs, cal_cols):
    """先验-数据对照表（P2）：方向 / 刻度 / 可检验性。

    每乘子：先验（文献值+方向）vs 数据方向（OOF 单乘子 AUROC、
    单变量系数符号）vs 数据刻度（校准系数均值±sd，折层聚合）。
    smear：known 子集内单变量检验（untestable 主口径）。
    """
    rows = []
    coef_df = pd.DataFrame(cal_coefs)
    C_full, _ = multiplier_matrix(df, include_smear=False)
    from sklearn.linear_model import LogisticRegression
    for c in cal_cols:
        xv = (df[c] if c in df.columns else C_full[c]).to_numpy(dtype=float)
        oof = np.zeros(len(y))
        for tr, te in folds:
            xm, ym = xv[tr], y[tr]
            if xm.std() == 0:
                oof[te] = float(np.mean(ym))
                continue
            # 单变量逻辑打分（OOF，方向与排序证据）
            lr = LogisticRegression(C=1.0, max_iter=1000)
            lr.fit(xm.reshape(-1, 1), ym)
            oof[te] = lr.predict_proba(xv[te].reshape(-1, 1))[:, 1]
        prior = _PRIOR_LITERATURE[c]
        co = coef_df[c]
        cal_mean, cal_sd = float(co.mean()), float(co.std())
        prior_theta = PRIOR_START.get(c)
        data_dir = '+' if cal_mean > 0 else ('-' if cal_mean < 0 else '0')
        # 刻度比：校准/先验（θ 的相对修正；哑元与 log_ts 可直接比）
        scale_ratio = (cal_mean / prior_theta
                       if prior_theta not in (None, 0.0) else None)
        rows.append({
            'multiplier': c,
            'prior_value': prior['value'],
            'prior_direction': prior['direction'],
            'univariate_oof_auroc': float(compute_auc(oof, y)),
            'calibrated_coef': cal_mean,
            'calibrated_sd_across_folds': cal_sd,
            'data_direction': data_dir,
            'direction_agrees': bool(
                (prior['direction'] == '+') == (cal_mean > 0)),
            'scale_ratio_cal_over_prior': scale_ratio,
            'testable': True,
        })
    # smear：known 子集敏感性
    known = df['idx_smear_known'] == 1
    smear_row = {
        'multiplier': 'smear',
        'prior_value': _PRIOR_LITERATURE['smear']['value'],
        'prior_direction': '+',
        'testable': False,
        'note': '83% not done → 自主 λ 移除；下行 known 子集敏感性',
    }
    if known.sum() >= 50 and y[known.to_numpy()].sum() >= 10:
        xk = df.loc[known, 'idx_smear_pos'].to_numpy(dtype=float)
        oofk = np.zeros(int(known.sum()))
        yk = y[known.to_numpy()]
        # known 子集内 3 折 OOF（户分组）
        sub_groups = df.loc[known, 'record_id'].to_numpy()
        sub_folds = _group_cv_indices(sub_groups, yk, n_splits=3, seed=0)
        for tr, te in sub_folds:
            lr2 = LogisticRegression(C=1.0, max_iter=1000)
            lr2.fit(xk[tr].reshape(-1, 1), yk[tr])
            oofk[te] = lr2.predict_proba(xk[te].reshape(-1, 1))[:, 1]
        smear_row['known_subset'] = {
            'n': int(known.sum()),
            'n_events': int(yk.sum()),
            'smear_pos_rate_pos10': float(
                yk[xk == 1].mean()) if (xk == 1).any() else None,
            'smear_neg_rate_pos10': float(
                yk[xk == 0].mean()) if (xk == 0).any() else None,
            'univariate_oof_auroc_known': float(compute_auc(oofk, yk)),
        }
    rows.append(smear_row)
    return rows


# ---- 多种子汇总 -------------------------------------------------------

_CONTRASTS = (
    ('cal_pi_only', 'frozen_pi_only'),        # P1 验收 1：校准抬升
    ('exposure_calpi:RF', 'exposure:RF'),     # P1 验收 2：特征级不再负增益
    ('exposure_hh:RF', 'exposure:RF'),        # 户聚合增益
    ('cal_pi_only', 'ind:RF'),                # 校准 λ vs 学习基线
    ('cal_simplified_pi_only', 'cal_pi_only'),        # P2 验收：精简增益
    ('cal_simplified_pi_only', 'frozen_simplified_pi_only'),  # 精简后校准
    ('exposure_calsimpl_pi:RF', 'exposure:RF'),        # 精简 λ 特征级
)


def run_multi_seed_calibration(n_seeds=20, seed_start=0, n_splits=5,
                               n_bootstrap=2000, df=None):
    """多种子：三臂 + 户聚合 + 审计表 + cluster bootstrap。"""
    if df is None:
        df = load_homeacf_contacts()
    y0 = df['tst_pos10'].astype(int).to_numpy()
    groups0 = df['record_id'].to_numpy()
    folds0 = _group_cv_indices(groups0, y0, n_splits=n_splits, seed=0)

    per_seed = []
    last = None
    for s in range(seed_start, seed_start + n_seeds):
        rep = run_calibration_ablation_once(seed=s, n_splits=n_splits,
                                            df=df)
        per_seed.append({
            'seed': rep['seed'], 'arms': rep['arms'],
            'cal_coefs': rep['cal_coefs'],
            'delong': {
                f'{a}_vs_{b}': delong_paired_test(
                    rep['y'], rep['oof'][a], rep['oof'][b])
                for a, b in _CONTRASTS
            },
        })
        last = rep
    # 末种子 OOF 上的户级 cluster bootstrap（效应量主口径）
    y, groups = last['y'], last['groups']
    ladder = {
        f'{a}_minus_{b}': _cluster_bootstrap_delta(
            last['oof'][a], last['oof'][b], y, groups,
            n_bootstrap=n_bootstrap, seed=seed_start)
        for a, b in _CONTRASTS
    }

    arm_summary = {}
    for name in per_seed[0]['arms']:
        vals = [r['arms'][name]['auroc'] for r in per_seed]
        prs = [r['arms'][name]['pr_auc'] for r in per_seed]
        arm_summary[name] = {
            'mean_auroc': float(np.mean(vals)),
            'sd_auroc': float(np.std(vals)),
            'mean_pr_auc': float(np.mean(prs)),
            'note': per_seed[0]['arms'][name].get('note'),
        }

    # 审计：全部种子×折的校准系数聚合
    all_coefs = []
    for r in per_seed:
        all_coefs.extend(r['cal_coefs'])
    audit = multiplier_audit(df, y0, folds0, all_coefs, last['cal_cols'])

    lift = ladder['cal_pi_only_minus_frozen_pi_only']
    noharm = ladder['exposure_calpi:RF_minus_exposure:RF']
    hh_gain = ladder['exposure_hh:RF_minus_exposure:RF']
    simpl_lift = ladder['cal_simplified_pi_only_minus_cal_pi_only']
    simpl_no_harm = ladder['exposure_calsimpl_pi:RF_minus_exposure:RF']
    return {
        'design': {
            'name': 'lambda_calibration_homeacf_v2',
            'source': 'HomeACF（github petermacp/tstsa）',
            'endpoint': 'tst_pos10（TST≥10mm LTBI）',
            'n': int(last['design']['n']),
            'n_events': int(last['design']['n_events']),
            'n_households': int(last['design']['n_households']),
            'n_seeds': n_seeds, 'seed_start': seed_start,
            'cv': f'StratifiedGroupKFold({n_splits}) by household',
            'ci': f'户级 cluster bootstrap ×{n_bootstrap}（末种子 OOF）',
            'pipeline': 'torch BCEWithLogitsLoss + Adam（seir_joint 实'
                        '验 1 管线），乘子系数 log 空间、先验为起点',
            'prior_start': {k: float(v) for k, v in PRIOR_START.items()},
            'simplified_components': list(SIMPLIFIED_COMPONENTS),
            'removed_components': list(_REMOVED_COMPONENTS),
            'acceptance': {
                'P1a_auclift_vs_frozen': {
                    'mean': lift['mean'], 'ci': lift['bootstrap_ci'],
                    'significant': lift['ci_excludes_zero']},
                'P1b_feature_no_harm': {
                    'mean': noharm['mean'], 'ci': noharm['bootstrap_ci'],
                    'not_significantly_negative': bool(
                        noharm['bootstrap_ci'][1] >= 0.0)},
                'P2_simplified_lift_vs_full_cal': {
                    'mean': simpl_lift['mean'],
                    'ci': simpl_lift['bootstrap_ci'],
                    'significant': simpl_lift['ci_excludes_zero']},
                'P2_simplified_feature_no_harm': {
                    'mean': simpl_no_harm['mean'],
                    'ci': simpl_no_harm['bootstrap_ci'],
                    'not_significantly_negative': bool(
                        simpl_no_harm['bootstrap_ci'][1] >= 0.0)},
            },
        },
        'arm_summary': arm_summary,
        'ladder_summary': ladder,
        'multiplier_audit': audit,
        'seeds': [
            {'seed': r['seed'],
             'arms': r['arms'],
             'delong': r['delong']} for r in per_seed],
        'conclusion': {
            'cal_lift_vs_frozen': {
                'mean': lift['mean'], 'ci': lift['bootstrap_ci'],
                'significant': lift['ci_excludes_zero']},
            'calpi_feature_gain_vs_exposure': {
                'mean': noharm['mean'], 'ci': noharm['bootstrap_ci'],
                'no_longer_harmful': bool(noharm['bootstrap_ci'][1] >= 0.0)},
            'household_agg_gain': {
                'mean': hh_gain['mean'], 'ci': hh_gain['bootstrap_ci'],
                'significant': hh_gain['ci_excludes_zero']},
            'simplified_cal_lift_vs_full_cal': {
                'mean': simpl_lift['mean'], 'ci': simpl_lift['bootstrap_ci'],
                'significant': simpl_lift['ci_excludes_zero']},
            'simplified_feature_no_harm': {
                'mean': simpl_no_harm['mean'],
                'ci': simpl_no_harm['bootstrap_ci'],
                'no_harm': bool(simpl_no_harm['bootstrap_ci'][1] >= 0.0)},
        },
    }


def save_result(out, path=None):
    """归档 JSON。"""
    if path is None:
        stamp = time.strftime('%Y%m%d')
        path = os.path.join(_REPO_ROOT, 'data', 'processed',
                            f'lambda_calibration_homeacf_{stamp}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return path
