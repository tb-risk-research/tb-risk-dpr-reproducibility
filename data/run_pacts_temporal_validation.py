#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序家庭先验在 PACTS（马拉维接触者队列）上的可复验性核查与复验
（2026-08-25 第四轮 P4a）。

背景：HomeACF（南非）时序先验 +0.058 AUROC 是单一数据集结果
（household_temporal_screening_v1）。用户要求核查马拉维 PACTS
数据的户结构并复验。

== 户结构核查（先决问题）==
PACTS 的 'id'（指示病例）即户簇键，同户接触者共享：
  804 接触者 / 192 户 / 户均 4.19 人 / ≥2 人户覆盖 98% 接触者
  （比 HomeACF 877 户均 3.1 更大）——户结构存在且充足。

== 先证变量的谱系适配（关键边界）==
HomeACF 用 TST 阳性（筛查时点即时结果，48-72h 可读）作先证；
PACTS **无 TST/IGRA 检测**，仅有基线症状问卷（p16cgh-p22wlb，
t0）+ 3 月随访终点（sx_3m，t1）。可部署且时序合法的先证口径 =
**同户已筛查者的基线症状状态**（基线在前、终点在后）。诚实边界：
  - 先证是"症状"而非"感染检测"（信号本体弱于 TST：同户基线症状
    LOO 与 sx_3m 相关仅 0.221，而 HomeACF TST LOO AUROC 0.715）；
  - 本复验检验的是"序贯筛查中户内先证信息更新"**机制**的跨队列
    成立性，不是 HomeACF 数值的复制。

== 臂结构（与 household_temporal 严格同构）==
主模型 = 冻结注册表 RF，StratifiedGroupKFold 户分组 CV（id），
随机筛查顺序（部署期望口径），20 种子 + 末种子户级 cluster
bootstrap：
  ind:RF             个体基线（13 列）
  exposure:RF        + 指示病例 + 暴露分级（31 列）
  prior_only:RF      纯时序先证特征（4 列，参考）
  exposure_prior:RF  exposure + 时序先证（主验收臂）
  hh_loo_only        同户他人基线症状 LOO 率（部署不可用上界锚定）

时序特征（户内位置 k 的接触者，只聚合位置 < k 的成员基线症状）：
  prior_n / prior_sx / prior_rate / prior_screened

产物（只追加不覆盖）：
- data/processed/pacts_temporal_household_20260825.json
"""

import json
import os
import sys
import time

# 注意：插入 repo 的父目录（Desktop）而非 repo 本身——杂散
# tb_risk/tb_risk 命名空间目录会在后者情况下遮蔽真包。
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tb_risk.validation.real_data_pi import (  # noqa: E402
    _cluster_bootstrap_delta,
    _group_cv_indices,
    _make_model,
    _pr_auc,
    _recall_at_budget,
    feature_columns as pacts_feature_columns,
    load_pacts_contacts,
)
from tb_risk.validation.threshold_spec import compute_auc  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'processed')
OUT_JSON = os.path.join(OUT_DIR, 'pacts_temporal_household_20260825.json')

PRIOR_COLS = ['prior_n', 'prior_sx', 'prior_rate', 'prior_screened']


def _assign_random_order(df, seed):
    """户内均匀随机筛查顺序（部署期望主口径；位置 0 最先筛查）。"""
    rng = np.random.RandomState(seed)
    pos = pd.Series(index=df.index, dtype=float)
    for _, sub in df.groupby('id'):
        n = len(sub)
        pos.loc[sub.index] = (rng.permutation(n) if n > 1
                               else np.zeros(n))
    return pos


def _prior_features(df, order_pos):
    """已筛查同户成员**基线症状**聚合（时序留出——只含位置更小者）。

    先证 = base_sx（基线症状哑元，t0 问卷可读），与 household_temporal.
    prior_features 同构（那聚合 tst_pos10）。
    """
    sx = df['base_sx'].to_numpy(dtype=float)
    hh = df['id'].to_numpy()
    pos = order_pos.to_numpy(dtype=float)
    n_rows = len(df)
    prior_n = np.zeros(n_rows)
    prior_sx = np.zeros(n_rows)
    for hh_id in np.unique(hh):
        idx_local = np.where(hh == hh_id)[0]
        order = np.argsort(pos[idx_local], kind='stable')
        sorted_idx = idx_local[order]
        sh = sx[sorted_idx]
        k = len(sh)
        prior_n[sorted_idx] = np.arange(k)
        prior_sx[sorted_idx] = np.concatenate(
            [np.zeros(1), np.cumsum(sh)[:-1]])
    base_rate = float(sx.mean())
    prior_rate = np.where(prior_n > 0,
                          prior_sx / np.maximum(prior_n, 1.0),
                          base_rate)
    return pd.DataFrame({
        'prior_n': prior_n,
        'prior_sx': prior_sx,
        'prior_rate': prior_rate,
        'prior_screened': (prior_n > 0).astype(float),
    })


def _loo_base_sx_rate(df):
    """leave-one-out 同户基线症状率（部署不可用上界锚定）。"""
    sx = pd.Series(df['base_sx'].to_numpy(dtype=float))
    hh = pd.Series(df['id'].to_numpy())
    hh_sum = sx.groupby(hh).transform('sum')
    hh_n = hh.groupby(hh).transform('size').astype(float)
    loo = (hh_sum - sx) / (hh_n - 1.0).clip(lower=1.0)
    return loo.where(hh_n > 1.0, other=float(sx.mean())).to_numpy()


def run_pacts_temporal_once(seed, n_splits, df):
    """单种子：时序留出臂 × 户分组 CV，全部 OOF。"""
    y = df['sx_3m'].astype(int).to_numpy()
    groups = df['id'].to_numpy()
    folds = _group_cv_indices(groups, y, n_splits=n_splits, seed=seed)

    order = _assign_random_order(df, seed)
    pf = _prior_features(df, order)

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

    cols_ind = pacts_feature_columns('ind')
    cols_exp = pacts_feature_columns('exposure')
    arms = {
        'ind:RF': _rf_oof(cols_ind),
        'exposure:RF': _rf_oof(cols_exp),
        'prior_only:RF': _rf_oof(PRIOR_COLS),
        'exposure_prior:RF': _rf_oof(cols_exp + PRIOR_COLS),
    }
    loo = _loo_base_sx_rate(df)
    result = {'seed': seed, 'y': y, 'groups': groups,
              'oof': arms, 'loo': loo, 'arms': {}}
    for name, oof in arms.items():
        result['arms'][name] = {
            'auroc': float(compute_auc(oof, y)),
            'pr_auc': _pr_auc(y, oof),
            'recall_at_budget': _recall_at_budget(oof, y),
        }
    result['arms']['hh_loo_only'] = {
        'auroc': float(compute_auc(loo, y)),
        'pr_auc': _pr_auc(y, loo),
        'note': 'leave-one-out 同户基线症状率——部署不可用上界锚定',
    }
    return result


def main(n_seeds=20, seed_start=0, n_splits=5, n_bootstrap=2000):
    print('=' * 70)
    print('PACTS 时序先验复验：户结构核查 + 基线症状先证口径')
    print('=' * 70)
    df = load_pacts_contacts()
    # 基线症状哑元（先证变量：t0 问卷可读，先于 sx_3m 终点）
    df['base_sx'] = (df['baseline_symptom_count'] > 0).astype(float)
    y = df['sx_3m'].astype(int).to_numpy()
    sizes = df.groupby('id').size()
    print(f'户结构: {len(df)} 接触者 / {len(sizes)} 户 / 户均 '
          f'{sizes.mean():.2f} 人 / ≥2 人户覆盖 '
          f'{sizes[sizes >= 2].sum() / sizes.sum():.1%} 接触者')
    print(f'终点: sx_3m 阳性率 {y.mean():.3f}（{int(y.sum())} 事件）；'
          f'先证 = 基线症状（PACTS 无 TST/IGRA）')

    per_seed = []
    t0 = time.time()
    for i in range(n_seeds):
        rep = run_pacts_temporal_once(seed_start + i, n_splits, df)
        per_seed.append(rep)
        print(f'  seed {rep["seed"]:2d}: '
              + ' '.join(f'{k}={v["auroc"]:.4f}'
                         for k, v in rep['arms'].items()))
    print(f'{n_seeds} 种子完成 ({time.time() - t0:.1f}s)')

    summary = {}
    for arm in ('ind:RF', 'exposure:RF', 'prior_only:RF',
                'exposure_prior:RF', 'hh_loo_only'):
        vals = [r['arms'][arm]['auroc'] for r in per_seed]
        pvs = [r['arms'][arm]['pr_auc'] for r in per_seed]
        summary[arm] = {
            'mean_auroc': round(float(np.mean(vals)), 4),
            'sd_across_seeds': round(float(np.std(vals)), 4),
            'mean_pr_auc': round(float(np.mean(pvs)), 4),
        }

    last = per_seed[-1]
    deltas = {}
    for a, b in (('exposure_prior:RF', 'ind:RF'),
                 ('exposure_prior:RF', 'exposure:RF'),
                 ('prior_only:RF', 'ind:RF')):
        deltas[f'{a} - {b}'] = _cluster_bootstrap_delta(
            last['oof'][a], last['oof'][b], last['y'], last['groups'],
            n_bootstrap=n_bootstrap, seed=last['seed'])

    gains = {k: round(float(np.mean(
        [r['arms']['exposure_prior:RF']['auroc'] - r['arms'][b]['auroc']
         for r in per_seed])), 4)
        for k, b in (('vs_ind', 'ind:RF'), ('vs_exposure', 'exposure:RF'))}

    out = {
        'design': {
            'dataset': 'PACTS（马拉维接触者队列，终点 sx_3m 3 月症状）',
            'household_structure': {
                'n_contacts': int(len(df)), 'n_households': int(len(sizes)),
                'mean_household_size': round(float(sizes.mean()), 2),
                'frac_contacts_in_multi_person_hh': round(
                    float(sizes[sizes >= 2].sum() / sizes.sum()), 4),
                'household_key': "id（指示病例簇键）",
            },
            'prior_variable': (
                '同户已筛查者基线症状（base_sx，t0 问卷可读）——PACTS '
                '无 TST/IGRA，HomeACF 的 TST 先证口径不可复制；本复验'
                '检验机制（序贯先证更新）而非数值（先证信号本体弱于'
                '感染检测：同户基线症状 LOO 与 sx_3m 相关 0.221）'),
            'endpoint': 'sx_3m',
            'model': '冻结注册表 RF（MODEL_REGISTRY，只注入种子）',
            'cv': 'StratifiedGroupKFold 户分组（id）',
            'order': 'random（部署期望主口径）',
            'n_seeds': n_seeds, 'seed_start': seed_start,
            'n_splits': n_splits, 'n_bootstrap': n_bootstrap,
            'protocol': '与 household_temporal_screening_v1 严格同构',
        },
        'arms_summary': summary,
        'deltas_cluster_bootstrap': deltas,
        'per_seed': [
            {'seed': r['seed'],
             'auroc': {k: v['auroc'] for k, v in r['arms'].items()}}
            for r in per_seed],
        'verdict': {
            'gain_vs_ind': gains['vs_ind'],
            'gain_vs_exposure': gains['vs_exposure'],
            'ci_excludes_zero_vs_ind': bool(
                deltas['exposure_prior:RF - ind:RF']
                ['ci_excludes_zero']),
            'ci_excludes_zero_vs_exposure': bool(
                deltas['exposure_prior:RF - exposure:RF']
                ['ci_excludes_zero']),
            'mechanism_replicated': bool(
                gains['vs_ind'] > 0 and
                deltas['exposure_prior:RF - ind:RF']
                ['ci_excludes_zero']),
            'interpretation': '',
        },
    }
    v = out['verdict']
    ind_ceiling = summary['ind:RF']['mean_auroc'] >= 0.95
    v['ind_baseline_ceiling'] = bool(ind_ceiling)
    if v['mechanism_replicated']:
        v['interpretation'] = (
            '时序先证机制在 PACTS 上复现（exposure+prior vs ind 显著'
            f"为正，{v['gain_vs_ind']:+.4f}）")
    elif ind_ceiling:
        # 既有归档（real_data_pi_validation_20260825.json）同口径
        # ind:RF = 0.9928：sx_3m 终点主要为基线症状的持续，个体
        # 基线特征（含自身基线症状）已近完美预测——时序先验的
        # 增益空间结构性不存在。
        v['verdict_type'] = 'inconclusive_ceiling'
        v['interpretation'] = (
            'PACTS sx_3m 终点天花板效应（ind:RF '
            f"{summary['ind:RF']['mean_auroc']:.3f}——基线症状≈终点，"
            '与既有归档 real_data_pi_validation_20260825 同值），时序'
            '先验增益空间结构性不存在；复验不可裁决（inconclusive），'
            '非机制否定——机制裁决移交 ERASE-TB（感染检测先证 + '
            '进展终点）')
    elif v['gain_vs_ind'] > 0:
        v['verdict_type'] = 'positive_not_significant'
        v['interpretation'] = (
            '时序先证方向为正但不显著（症状先证信号弱——PACTS 无 '
            '感染检测，机制复现结论为"弱阳性不显著"）')
    else:
        v['verdict_type'] = 'failed_on_this_proxy'
        v['interpretation'] = (
            '时序先证在 PACTS 基线症状先证口径下无正增益——机制'
            '复现失败于该口径（不排除 TST 口径下成立，需 ERASE-TB '
            '类带感染检测的数据裁决）')

    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print('\n'.join(f'[{k}] AUROC={v["mean_auroc"]:.4f} PR-AUC='
                    f'{v["mean_pr_auc"]:.4f}'
                    for k, v in summary.items()))
    for k, d in deltas.items():
        ci = d['bootstrap_ci']
        print(f'{k}: {d["mean"]:+.4f} CI [{ci[0]:+.4f}, {ci[1]:+.4f}] '
              f'p_positive={d["p_positive"]:.3f}')
    print(f"判定: {v['interpretation']}")
    print(f'已保存: {OUT_JSON}')
    return out


if __name__ == '__main__':
    main()
