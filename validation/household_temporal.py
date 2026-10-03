#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序留出家庭传播模型（P1，2026-08-25 第三轮）：把"不可部署的
强信号"转成"可部署的强信号"。

背景（上轮 P3 的关键发现）：同户他人 TST 阳性率（leave-one-out）
单变量 AUROC 0.715，超过全部学习臂（0.643~0.645）——家庭聚集
真实存在且很强，但部署时序上不可得（预测一个尚未筛查的人时，
不知道同户筛查结果）。这是合成实验"邻居确诊密度"信息不对称
的真实版本。

解法（用户规格）：模拟真实筛查时序——户内接触者按筛查顺序排列，
第 k 位筛查者只允许使用"已筛查成员（位置 < k）的结果"作特征：
同户先证者 TST 阳性 → 后筛查者风险上调。两阶段筛查语义：
第一轮查出阳性户，第二轮在户内按风险排序。

时序特征（对户 h 内位置 k 的接触者）：
  prior_n        户内已筛查人数（k）
  prior_pos      已筛查者中 TST 阳性数
  prior_rate     已筛查者阳性率（prior_n=0 填全局基线率）
  prior_screened 已有先证信息哑元

筛查顺序协议（真实顺序未记录，随机为部署期望主口径）：
  random  户内均匀随机（主口径——任意筛查顺序下的期望性能，
          多种子平均吸收排序随机性）
  oracle  阳性优先（两阶段策略上界——先查阳性者使后查者获得
          最强先证信息；敏感性参考）
  anti    阴性优先（下界；敏感性参考）

防泄漏论证（协议核心）：
  prior 特征是同户他人标签的函数；StratifiedGroupKFold 按户分组
  → 同户永远同折 → 测试折户的先证标签从未进入训练；训练折户的
  先证标签训练时可见 = 部署时序合法（已筛查结果是历史）。
  时序合法性 = 部署合法性，二者由同一构造保证。

臂结构（主模型 = 冻结注册表 RF，同 real_data_infection 口径）：
  ind:RF            个体基线（复现 0.643）
  exposure:RF       + 指示病例 + 暴露分级（复现 0.644）
  prior_only:RF    纯时序先证特征（4 列）——先证信号本体
  exposure_prior:RF exposure + 时序先证（主验收臂）
  hh_loo_only       leave-one-out 同户率（0.715 上界参考，
                    部署不可用——非对比主线，只作天花板锚定）

验收（用户原文口径）：时序约束下的家庭先证特征（只允许用已筛查
成员结果）显著超过个体基线 0.643——即 exposure_prior −
ind:RF（及 exposure:RF）的户级 cluster bootstrap CI 排除 0 且
为正。

诚实边界：
  - 真实筛查顺序未记录，random 排序是"期望部署"而非实际协议；
    oracle/anti 给出顺序敏感性包络；
  - prior 特征依赖标签 → 不可用于 λ 物理族（λ 是先验构造），
    只进学习臂；
  - 单队列（HomeACF）LTBI 终点，非 TB 发病；
  - 两阶段筛查的成本口径（第一轮全筛）不在本验证范围。
"""

import json
import os
import time

import numpy as np
import pandas as pd

from .layer_ablation import delong_paired_test
from .real_data_infection import (
    feature_columns as inf_feature_columns,
    load_homeacf_contacts,
)
from .real_data_pi import (
    _cluster_bootstrap_delta,
    _group_cv_indices,
    _make_model,
    _pr_auc,
    _recall_at_budget,
)
from .threshold_spec import compute_auc

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORDER_MODES = ('random', 'oracle', 'anti')


def assign_screening_order(df, seed=0, mode='random'):
    """户内筛查顺序（位置 0 最先筛查）。

    random = 均匀随机（部署期望主口径）；oracle = 阳性优先
    （两阶段上界）；anti = 阴性优先（下界）。

    Returns:
        pd.Series: 每行户内位置（0-based，户内无重复）。
    """
    if mode not in ORDER_MODES:
        raise ValueError('mode 必须是 %s 之一' % (ORDER_MODES,))
    rng = np.random.RandomState(seed)
    y = df['tst_pos10'].to_numpy()
    pos = pd.Series(index=df.index, dtype=float)
    for _, sub in df.groupby('record_id'):
        n = len(sub)
        if n == 1:
            pos.loc[sub.index] = 0
            continue
        if mode == 'random':
            perm = rng.permutation(n)
        elif mode == 'oracle':
            # 阳性 key 小 → argsort 排前；同状态内随机断平
            key = (1 - y[sub.index.to_numpy()]) * 1e6 + rng.rand(n)
            perm = np.argsort(key)
        else:  # anti
            key = y[sub.index.to_numpy()] * 1e6 + rng.rand(n)
            perm = np.argsort(key)
        pos.loc[sub.index] = perm
    return pos


def prior_features(df, order_pos):
    """已筛查同户成员结果聚合（时序留出——只含位置更小者）。

    Returns:
        pd.DataFrame: prior_n / prior_pos / prior_rate /
        prior_screened 四列（与 lambda_calibration.household_features
        的 loo 同族，但仅聚合位置更小的成员 = 部署时序可用）。
    """
    y = df['tst_pos10'].to_numpy(dtype=float)
    hh = df['record_id'].to_numpy()
    pos = order_pos.to_numpy(dtype=float)
    n_rows = len(df)
    prior_n = np.zeros(n_rows)
    prior_pos = np.zeros(n_rows)
    for hh_id in np.unique(hh):
        idx_local = np.where(hh == hh_id)[0]
        order = np.argsort(pos[idx_local], kind='stable')
        sorted_idx = idx_local[order]
        yh = y[sorted_idx]
        k = len(yh)
        prior_n[sorted_idx] = np.arange(k)
        prior_pos[sorted_idx] = np.concatenate(
            [np.zeros(1), np.cumsum(yh)[:-1]])
    base_rate = float(y.mean())
    prior_rate = np.where(prior_n > 0,
                          prior_pos / np.maximum(prior_n, 1.0),
                          base_rate)
    return pd.DataFrame({
        'prior_n': prior_n,
        'prior_pos': prior_pos,
        'prior_rate': prior_rate,
        'prior_screened': (prior_n > 0).astype(float),
    })


def _loo_household_rate(df):
    """leave-one-out 同户 TST 阳性率（上界参考，部署不可用）。"""
    y = df['tst_pos10'].astype(float).to_numpy()
    hh = pd.Series(df['record_id'].to_numpy())
    y_s = pd.Series(y)
    hh_sum = y_s.groupby(hh).transform('sum')
    hh_n = hh.groupby(hh).transform('size').astype(float)
    loo = (hh_sum - y_s) / (hh_n - 1.0).clip(lower=1.0)
    return loo.where(hh_n > 1.0, other=float(np.mean(y))).to_numpy()


def run_temporal_household_once(seed=0, n_splits=5, order_mode='random',
                                df=None):
    """单种子：时序留出臂 × 户分组 CV，全部 OOF。"""
    if df is None:
        df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    folds = _group_cv_indices(groups, y, n_splits=n_splits, seed=seed)

    order = assign_screening_order(df, seed=seed, mode=order_mode)
    pf = prior_features(df, order)

    def _rf_oof(cols):
        X = np.nan_to_num(
            pd.concat([df.reset_index(drop=True), pf],
                      axis=1)[cols].to_numpy(dtype=float), nan=0.0)
        oof = np.zeros(len(y))
        for tr, te in folds:
            m = _make_model('random_forest', seed)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        return oof

    cols_ind = inf_feature_columns('ind')
    cols_exp = inf_feature_columns('exposure')
    cols_prior = ['prior_n', 'prior_pos', 'prior_rate', 'prior_screened']

    arms = {
        'ind:RF': _rf_oof(cols_ind),
        'exposure:RF': _rf_oof(cols_exp),
        'prior_only:RF': _rf_oof(cols_prior),
        'exposure_prior:RF': _rf_oof(cols_exp + cols_prior),
    }
    result = {'seed': seed, 'order_mode': order_mode, 'y': y,
              'groups': groups, 'arms': {}, 'oof': arms,
              'design': {
                  'n': int(len(y)), 'n_events': int(y.sum()),
                  'n_households': int(pd.Series(groups).nunique()),
                  'n_splits': n_splits, 'order_mode': order_mode,
                  'endpoint': 'tst_pos10（TST≥10mm LTBI）',
              }}
    for name, oof in arms.items():
        result['arms'][name] = {
            'auroc': float(compute_auc(oof, y)),
            'pr_auc': _pr_auc(y, oof),
            'recall_at_budget': _recall_at_budget(oof, y),
        }
    loo = _loo_household_rate(df)
    result['arms']['hh_loo_only'] = {
        'auroc': float(compute_auc(loo, y)),
        'pr_auc': _pr_auc(y, loo),
        'recall_at_budget': _recall_at_budget(loo, y),
        'note': 'leave-one-out 同户率——部署不可用上界锚定',
    }
    return result


_CONTRASTS = (
    ('exposure_prior:RF', 'ind:RF'),        # P1 验收：vs 个体基线
    ('exposure_prior:RF', 'exposure:RF'),   # P1 验收：vs 最强学习臂
    ('prior_only:RF', 'ind:RF'),            # 纯先证信号本体
)


def run_multi_seed_temporal(n_seeds=20, seed_start=0, n_splits=5,
                            n_bootstrap=2000, order_mode='random',
                            df=None, sensitivity_seeds=3):
    """多种子 + 户级 cluster bootstrap + 顺序敏感性包络。"""
    if df is None:
        df = load_homeacf_contacts()

    per_seed = []
    last = None
    for s in range(seed_start, seed_start + n_seeds):
        rep = run_temporal_household_once(seed=s, n_splits=n_splits,
                                          order_mode=order_mode, df=df)
        per_seed.append({
            'seed': rep['seed'], 'arms': rep['arms'],
            'delong': {
                f'{a}_vs_{b}': delong_paired_test(
                    rep['y'], rep['oof'][a], rep['oof'][b])
                for a, b in _CONTRASTS
            },
        })
        last = rep
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

    # 顺序敏感性包络（oracle 上界 / anti 下界，少量种子）
    sensitivity = {}
    for mode in ('oracle', 'anti'):
        acc = [run_temporal_household_once(seed=s, order_mode=mode, df=df)
               ['arms']['exposure_prior:RF']['auroc']
               for s in range(seed_start, seed_start + sensitivity_seeds)]
        sensitivity[mode] = {
            'exposure_prior_auroc_mean': float(np.mean(acc)),
            'n_seeds': sensitivity_seeds,
        }

    acc_ind = ladder['exposure_prior:RF_minus_ind:RF']
    acc_exp = ladder['exposure_prior:RF_minus_exposure:RF']
    return {
        'design': {
            'name': 'household_temporal_screening_v1',
            'source': 'HomeACF（github petermacp/tstsa）',
            'endpoint': 'tst_pos10（TST≥10mm LTBI）',
            'n': last['design']['n'],
            'n_events': last['design']['n_events'],
            'n_households': last['design']['n_households'],
            'n_seeds': n_seeds, 'seed_start': seed_start,
            'order_mode': order_mode,
            'cv': f'StratifiedGroupKFold({n_splits}) by household',
            'ci': f'户级 cluster bootstrap ×{n_bootstrap}（末种子 OOF）',
            'leakage_argument': 'prior 特征=同户位置更小者标签的函数；'
                                '户分组 CV 同户同折 → 测试折先证标签'
                                '从未进训练；已筛查结果=部署时序合法',
            'acceptance': {
                'vs_ind_baseline': {
                    'mean': acc_ind['mean'], 'ci': acc_ind['bootstrap_ci'],
                    'significant_positive': bool(
                        acc_ind['ci_excludes_zero']
                        and acc_ind['mean'] > 0)},
                'vs_exposure_baseline': {
                    'mean': acc_exp['mean'], 'ci': acc_exp['bootstrap_ci'],
                    'significant_positive': bool(
                        acc_exp['ci_excludes_zero']
                        and acc_exp['mean'] > 0)},
            },
        },
        'arm_summary': arm_summary,
        'ladder_summary': ladder,
        'order_sensitivity': sensitivity,
        'seeds': per_seed,
    }


def save_result(out, path=None):
    """归档 JSON。"""
    if path is None:
        stamp = time.strftime('%Y%m%d')
        path = os.path.join(_REPO_ROOT, 'data', 'processed',
                            f'household_temporal_screening_{stamp}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return path
