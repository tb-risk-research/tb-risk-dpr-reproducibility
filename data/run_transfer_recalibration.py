#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1-2（第十一轮，第十二/十三轮升级）：本地重校准最小样本量曲线。

背景（cross_population_transfer_20260826.json）：巴西三特征死亡模型
零样本迁移到 Vietnam PROVE_TB（Xpert 确诊终点）AUROC 0.4258 < 0.5
——死亡终点的 age 方向（age↑risk）在确诊终点人群反转，属跨终点
负迁移。Kenya 患病率调查迁移 0.5659 vs 本地 0.6108（+4.5pp 缺口）。

设计演进：
  v1 LR(trans_score)      —— logit 单调变换，保序/翻序二选一；
                             Vietnam「恢复」0.573≈1−0.426 纯翻序；
                             Kenya null 是设计产物（缺陷对照，弃用）
  v2 加性残差 logit+LR(age,sex,hiv) —— 注入本地特征但迁移 logit
                             权重固定 1：反信号洗不掉（Vietnam
                             0.463）；对齐场景恢复真实但慢
  v3 stack LR(logit,age,sex,hiv) —— 迁移 logit 降级为特征，权重
                             可学习（含负）：可部署配方
  v4 stack+先验（round-13） —— v3 基础上对 β_logit 加 N(1, σ²)
                             高斯先验：事件稀缺时先验主导退化为 v2
                             （β≈1 = 加性先验 + 本地特征），事件充足
                             时数据主导退化为 v3——把「选 v2 还是
                             v3」的决策树坍缩成单一公式

round-13 P1-1 网格外推：N∈{2000, 5000, 10000}（Kenya ≈11/27/53
阳性）——v2 臂在 N=1000 边界 0.590 仍在爬升、距恢复线 0.601 差
0.011，正衰减场景（最常见情形）的「要多少本地标签」是配方表
唯一空格，本轮补齐。

round-13 P1-2 方向裁定成本：配方按「零样本 AUROC 是否 <0.5」
条件化，但条件本身在本地标注上估计——每档 N 记录裁定错误率
（N 抽样 → 组内经验 AUROC<0.5 判反向 → 与全队列真值对比），
输出「裁定错误率 vs N」曲线 = runbook 自动化的前提。

N ∈ {100, 200, 500, 1000, 2000, 5000, 10000}，每档 20 次随机
重采样（分层采样，测中位 AUROC 与 IQR；N 上限 = 队列 80%）。

诚实边界：Vietnam n=611（事件 266）——2000+ 档自动截断；
Kenya 全队列仅 336 事件，N=10000 采样仅≈53 阳性——低基率
场景的裁定与恢复都受事件稀缺支配。

归档：data/processed/transfer_recalibration_20260826.json（v4 覆盖）
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

from scipy.optimize import minimize  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from tb_risk.data.run_cross_population_transfer import (  # noqa: E402
    load_kenya, load_vietnam, load_sinan_common, local_cv_auroc)

OUT = os.path.join(BASE, 'tb_risk', 'data', 'processed',
                   'transfer_recalibration_20260826.json')
NS = [100, 200, 500, 1000, 2000, 5000, 10000]
N_REPEATS = 20
SEED = 42
PRIOR_MU = 1.0    # v4 β_logit 先验均值（= v2 的固定权重）
PRIOR_SIGMA = 1.0  # v4 先验标准差（弱先验：事件充足时数据主导）


def _to_logit(score):
    score = np.clip(score, 1e-6, 1 - 1e-6)
    return np.log(score / (1 - score))


def _sigmoid(eta):
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))


def fit_residual_score(trans_score_train, y_train):
    """v1 残差层（缺陷对照）：logit_local = logit_transfer + LR(trans_score)。"""
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(_to_logit(trans_score_train).reshape(-1, 1), y_train)
    return lr


def fit_residual_features(X_local_train, y_train):
    """v2 加性残差：logit_local = logit_transfer + LR(age,sex,hiv)。"""
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X_local_train, y_train)
    return lr


def _stack_design(logit_col, X_local):
    return np.column_stack([logit_col,
                            np.asarray(X_local, dtype=np.float64)])


def fit_stack(X_local_train, logit_train, y_train):
    """v3 stack（可部署配方）：logit_local = LR(logit_transfer,age,sex,hiv)。"""
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(_stack_design(logit_train, X_local_train), y_train)
    return lr


def fit_stack_prior(X_local_train, logit_train, y_train,
                    mu=PRIOR_MU, sigma=PRIOR_SIGMA):
    """v4 统一配方：v3 stack + β_logit ~ N(mu, sigma²) 高斯先验。

    精确惩罚 MLE（scipy L-BFGS-B，非数据增强近似）：
      min_θ Σ softplus(η_i) − y_i·η_i + (β_logit − mu)² / (2σ²)
    事件稀缺时先验主导 → β_logit≈mu → 等价 v2 加性形式；
    事件充足时数据主导 → 等价 v3 stack。单一公式覆盖两场景。
    """
    Z = np.column_stack([np.ones(len(logit_train)), logit_train,
                         np.asarray(X_local_train, dtype=np.float64)])
    y = np.asarray(y_train, dtype=np.float64)

    def nll(theta):
        eta = Z @ theta
        return (float(np.sum(np.logaddexp(0.0, eta) - y * eta))
                + (theta[1] - mu) ** 2 / (2 * sigma ** 2))

    theta0 = np.zeros(Z.shape[1])
    theta0[1] = mu
    res = minimize(nll, theta0, method='L-BFGS-B')
    return res.x


def apply_stack_prior(theta, X_local, logit_all):
    Z = np.column_stack([np.ones(len(logit_all)), logit_all,
                         np.asarray(X_local, dtype=np.float64)])
    return _sigmoid(Z @ theta)


def recal_curve(X, y, src_model, n_repeats=N_REPEATS, seed=SEED):
    """恢复曲线 + 方向裁定错误率：zero_shot / local_N / 四臂微调。"""
    rng = np.random.default_rng(seed)
    trans_all = src_model.predict_proba(X)[:, 1]
    zero_shot = float(roc_auc_score(y, trans_all))
    logit_all = _to_logit(trans_all)
    truth_reversed = zero_shot < 0.5
    n_max = int(0.8 * len(y))
    ns = [n for n in NS if n <= n_max]
    rows = []
    for n in ns:
        fus_s, fus_f, fus_st, fus_p, lts = [], [], [], [], []
        ruling_err = []
        for rep in range(n_repeats):
            # 分层重采样（保持事件率）
            idx_pos = np.flatnonzero(y == 1)
            idx_neg = np.flatnonzero(y == 0)
            k_pos = max(2, int(round(n * y.mean())))
            k_neg = n - k_pos
            if k_pos > len(idx_pos) or k_neg > len(idx_neg):
                continue
            tr_i = np.concatenate([
                rng.choice(idx_pos, k_pos, replace=False),
                rng.choice(idx_neg, k_neg, replace=False)])
            te_i = np.setdiff1d(np.arange(len(y)), tr_i)
            if y[te_i].sum() < 5 or (1 - y[te_i]).sum() < 5:
                continue
            # (0) 方向裁定：组内经验 AUROC<0.5 判反向 vs 全队列真值
            emp_auc = roc_auc_score(y[tr_i], trans_all[tr_i])
            ruling_err.append((emp_auc < 0.5) != truth_reversed)
            # (a) v1 残差微调（LR(trans_score)，缺陷对照）
            lr_s = fit_residual_score(trans_all[tr_i], y[tr_i])
            sc_s = lr_s.predict_proba(
                logit_all[te_i].reshape(-1, 1))[:, 1]
            fus_s.append(roc_auc_score(y[te_i], sc_s))
            # (b) v2 加性残差（对齐场景的诚实形式）
            lr_f = fit_residual_features(X.iloc[tr_i], y[tr_i])
            sc_f = lr_f.predict_proba(X.iloc[te_i])[:, 1]
            fus_f.append(roc_auc_score(
                y[te_i], _to_logit(sc_f) + logit_all[te_i]))
            # (c) v3 stack（可部署配方）
            lr_st = fit_stack(X.iloc[tr_i], logit_all[tr_i], y[tr_i])
            fus_st.append(roc_auc_score(
                y[te_i], lr_st.predict_proba(
                    _stack_design(logit_all[te_i], X.iloc[te_i]))[:, 1]))
            # (d) v4 stack+先验（统一配方：稀缺退 v2 / 充足退 v3）
            theta = fit_stack_prior(X.iloc[tr_i], logit_all[tr_i],
                                    y[tr_i])
            fus_p.append(roc_auc_score(
                y[te_i], apply_stack_prior(theta, X.iloc[te_i],
                                           logit_all[te_i])))
            # (e) 纯本地（同样本三特征 LR）
            lrm = LogisticRegression(max_iter=1000)
            lrm.fit(X.iloc[tr_i], y[tr_i])
            lts.append(roc_auc_score(
                y[te_i], lrm.predict_proba(X.iloc[te_i])[:, 1]))

        def _pack(arr):
            if not arr:
                return None, None
            return (float(np.median(arr)),
                    [float(np.percentile(arr, 25)),
                     float(np.percentile(arr, 75))])

        m_s, i_s = _pack(fus_s)
        m_f, i_f = _pack(fus_f)
        m_st, i_st = _pack(fus_st)
        m_p, i_p = _pack(fus_p)
        m_l, i_l = _pack(lts)
        rows.append({
            'n_local': n,
            'n_repeats': len(fus_p),
            'n_events_median': (float(np.median(
                [int(round(n * y.mean()))])) if fus_p else None),
            'finetune_score_median': m_s, 'finetune_score_iqr': i_s,
            'finetune_features_median': m_f, 'finetune_features_iqr': i_f,
            'stack_median': m_st, 'stack_iqr': i_st,
            'stack_prior_median': m_p, 'stack_prior_iqr': i_p,
            'local_only_median': m_l, 'local_only_iqr': i_l,
            'ruling_error_rate': (float(np.mean(ruling_err))
                                  if ruling_err else None),
        })
        print('  N=%5d | v1 %.4f | v2 %.4f | v3 %.4f | v4先验 %.4f | '
              '纯本地 %.4f | 裁定错误率 %.0f%%（%d 次）'
              % (n, m_s or 0, m_f or 0, m_st or 0, m_p or 0, m_l or 0,
                 100 * (rows[-1]['ruling_error_rate'] or 0), len(fus_p)))
    return zero_shot, truth_reversed, ns, rows


def main():
    t0 = time.time()
    print('=== P1 重校准曲线 v4：网格外推 + 方向裁定成本 + 统一配方 ===')
    if not HAS_LGBM:
        raise SystemExit('需要 lightgbm')

    # 源模型（与迁移实验完全一致）
    Xs, ys, year = load_sinan_common()
    tr = year <= 12
    src = LGBMClassifier(n_estimators=400, learning_rate=0.06,
                         num_leaves=63, subsample=0.8,
                         colsample_bytree=0.8, n_jobs=-1,
                         random_state=SEED, verbose=-1)
    src.fit(Xs[tr], ys[tr])

    cohorts = []
    for name, loader in (('Kenya 患病率调查', load_kenya),
                         ('Vietnam PROVE_TB', load_vietnam)):
        X, y = loader()
        print('[%s] n=%d pos=%d rate %.3f' % (name, len(y), y.sum(),
                                             y.mean()))
        zero_shot, truth_reversed, ns, rows = recal_curve(X, y, src)
        full_auc, full_std = local_cv_auroc(X, y)

        def _recover(key):
            for r in rows:
                v = r.get(key)
                if v is not None and v >= full_auc - 0.01:
                    return r['n_local']
            return None

        cohorts.append({
            'cohort': name,
            'n': int(len(y)), 'event_rate': float(y.mean()),
            'zero_shot_auroc': zero_shot,
            'direction_truth': 'reversed' if truth_reversed else 'aligned',
            'full_local_cv_auroc': full_auc,
            'full_local_cv_std': full_std,
            'curve': rows,
            'n_to_recover_stack': _recover('stack_median'),
            'n_to_recover_stack_prior': _recover('stack_prior_median'),
            'n_to_recover_features': _recover('finetune_features_median'),
            'n_to_recover_score_v1': _recover('finetune_score_median'),
            'n_max_sampled': int(0.8 * len(y)),
        })
        c = cohorts[-1]
        print('[%s] zero-shot %.4f（%s）| 全量本地 %.4f | 恢复 N：'
              'v4先验=%s / v3stack=%s / v2加性=%s / v1分数=%s'
              % (name, zero_shot, c['direction_truth'], full_auc,
                 c['n_to_recover_stack_prior'],
                 c['n_to_recover_stack'],
                 c['n_to_recover_features'],
                 c['n_to_recover_score_v1']))

    res = {
        'date': '2026-09-03',
        'experiment': 'transfer_recalibration_v4',
        'question': '迁移配方最后两个未知数（round-13）：(1) 正衰减场景'
                    '（对齐站，最常见）要多少本地标签——Kenya 网格外推 '
                    'N∈{2000,5000,10000}；(2) 方向条件（零样本 AUROC<0.5）'
                    '在本地标注上估计需要多少样本才可靠——裁定错误率 vs N '
                    '曲线；(3) 统一配方 v4（stack+β_logit 先验）能否把'
                    '决策树坍缩成单一公式',
        'design': {
            'source': 'SINAN 三特征 LGBM（死亡终点，train≤12）',
            'arms': {
                'v1_score': 'logit + LR(trans_score)——单调变换缺陷对照',
                'v2_additive': 'logit + LR(age,sex,hiv)——对齐场景诚实形式'
                               '（迁移权重固定 1，反信号不可恢复）',
                'v3_stack': 'LR(logit, age,sex,hiv)——可部署配方',
                'v4_stack_prior': 'v3 + β_logit~N(%.1f, %.1f²) 先验'
                                  '（精确惩罚 MLE）——稀缺退 v2 / '
                                  '充足退 v3 的统一公式'
                                  % (PRIOR_MU, PRIOR_SIGMA),
                'direction_ruling': 'N 抽样 → 组内经验 AUROC<0.5 判反向 →'
                                    ' 对比全队列真值 → 裁定错误率',
            },
            'local_only': '同 N 样本三特征 LR（无迁移对照）',
            'full_local': '全量 5 折 CV LGBM（恢复上限参照）',
            'ns': NS, 'n_repeats': N_REPEATS, 'seed': SEED,
            'sampling': '分层重采样（保持事件率），N 上限 = 队列 80%',
            'recovery_rule': '各臂中位 AUROC ≥ 全量本地 CV − 0.01',
        },
        'cohorts': cohorts,
        'red_line_note': 'Vietnam 为负迁移场景（zero-shot 0.4258<0.5）；'
                         '本曲线回答四个问题：(1) 对齐站恢复 N（Kenya '
                         '外推网格）；(2) 方向裁定可靠样本量（错误率 '
                         'curve）；(3) v4 统一配方是否可行；(4) v1/v2 '
                         '设计产物与反信号失败的实证对照',
        'runtime_s': round(time.time() - t0, 1),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
