#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合机制消融：特征化阶梯（P1）+ 收益可加性检验（P2）。

臂结构（用户 P1/P2 规格："主模型走特征化，GNN 分支只保留时序+PI，
类型维度放弃端到端注意力"）：

  ┌─────────────────────────────────────────────────────────────────┐
  │ seir      纯物理：k̂·λ 排序（默认参数平坦 Λ_ref，部署口径）     │
  │ ind       个体基线：14 维基础特征（宿主 + 自身边聚合，RF/LGBM） │
  │ typed     + 分类型邻居暴露 5 列（P1 特征化）                    │
  │ window    + 分时间窗邻居暴露 5 列（P1 特征化）                  │
  │ pi        + k̂·λ 校准物理感染力 1 列                             │
  │ all       + 全部 11 列网络特征（v3 口径）                       │
  │ gnn_pi    GNN 分支参考：PI-GNN（源池化残差，读 s̄；类型/窗口     │
  │           不进 GNN——产品方向：类型走特征化）                   │
  │ oracle    真值 ν 排序（三机制全知上界）                         │
  └─────────────────────────────────────────────────────────────────┘

主模型 = 冻结注册表超参（scoring/ml/training.py MODEL_REGISTRY 的
random_forest / lightgbm default_params，逐字面复用——P1"直接加进
冻结的主模型特征集"的口径；只换特征列，不动模型超参）。

归因阶梯（P2 核心问题："收益是否可加"）：
  typed/window/pi − ind    单机制特征化的边际收益；
  all − ind                组合收益；
  synergy = (all−ind) − Σ(单机制−ind)
                           协同项：>0 超可加（互补），<0 亚可加
                           （特征间冗余/信息重叠），≈0 可加；
  realization = (all−ind)/(oracle−ind)
                           特征化兑现的可学间隙比例（P1"大头"主张）；
  gnn_pi − all             GNN 分支 vs 特征化主模型（产品路线裁决）。

协议沿用三层消融：分层 50/50 半区（接触者空间）、k̂ 只用训练半区、
逐种子 DeLong 配对 + 种子级 bootstrap CI。

诚实边界：DGP 三重信息不对称为构造性设定；"特征化拿到大头"的
结论口径 = 本组合机制世界内的兑现比例，非真实世界承诺。
"""

import numpy as np

from .layer_ablation import delong_paired_test
from .threshold_spec import compute_auc
from .combined_network import FEATURE_SETS
from .pi_network import calibrate_physics_k


MODEL_KEYS = ('random_forest', 'lightgbm')
FEATURE_ARMS = ('ind', 'typed', 'window', 'pi', 'all')


def _make_model(model_key, seed):
    """按冻结注册表超参构造主模型（只注入种子，其余逐字面复用）。"""
    from ..scoring.ml.training import MODEL_REGISTRY
    spec = MODEL_REGISTRY[model_key]
    params = dict(spec.default_params)
    params[spec.random_state_param] = int(seed)
    return spec.model_cls(**params)


def _stratified_split(labels, rng):
    """分层 50/50：训练/测试半区两类比例一致（接触者空间）。"""
    labels = np.asarray(labels, dtype=int)
    train_idx, test_idx = [], []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        train_idx.extend(idx[:half].tolist())
        test_idx.extend(idx[half:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def _recall_at_budget(scores, y, budget=0.25):
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    n_top = max(1, int(np.ceil(budget * len(y))))
    top = np.argsort(-scores)[:n_top]
    return float(y[top].sum() / max(y.sum(), 1))


def run_combined_ablation_once(n_contacts=400, seed=42, target_rate=0.25,
                               budget=0.25, gnn_epochs=800,
                               model_keys=MODEL_KEYS, skip_gnn=False):
    """单次特征化阶梯消融（一个种子 = 一个网络 + 一次训练）。

    Returns:
        dict: arms（seir / 各特征臂×模型 / gnn_pi / oracle 的
        auroc / recall / brier + test_scores）、ladder（逐模型阶梯）、
        additivity（synergy / realization）、delong
    """
    from .combined_network import build_combined_network, feature_matrix
    from ..ml.gnn.pi_gnn import train_pi_gnn

    net = build_combined_network(n_contacts=n_contacts,
                                 target_rate=target_rate,
                                 random_state=seed)
    labels = np.asarray(net['labels'], dtype=int)
    rng = np.random.RandomState(seed)
    train_idx, test_idx = _stratified_split(labels, rng)

    # 物理校准 k̂（只用训练半区；seir 臂与 pi 特征列共用）
    k_hat = calibrate_physics_k(net, labels, train_idx)

    arms = {}
    lam_t = k_hat * np.asarray(net['lam'], dtype=float)
    arms['seir'] = {'scores': lam_t, 'p': 1.0 - np.exp(-lam_t)}

    for fs in FEATURE_ARMS:
        X, _ = feature_matrix(net, feature_set=fs,
                              k_hat=k_hat if fs in ('pi', 'all') else None)
        for mk in model_keys:
            model = _make_model(mk, seed)
            model.fit(X[train_idx], labels[train_idx])
            p = model.predict_proba(X[test_idx])[:, 1]
            arms[f'{fs}:{mk}'] = {'p_test': p}

    if not skip_gnn:
        r = train_pi_gnn(net, labels, train_idx, mode='feature',
                         epochs=gnn_epochs, seed=seed)
        arms['gnn_pi'] = {'scores': r['score'], 'p': r['p']}

    arms['oracle_nu'] = {
        'scores': np.asarray(net['nu'], dtype=float),
        'p': 1.0 - np.exp(-net['k_calibration']
                          * np.asarray(net['nu'], dtype=float))}

    # ---- 统一评估（特征臂只有 test 半区预测；排序臂全量取 test）----
    y_test = labels[test_idx]
    for name, arm in arms.items():
        if 'p_test' in arm:                      # 主模型特征臂
            s = arm.pop('p_test')
            arm['test_scores'] = np.asarray(s, dtype=float).tolist()
            p_test = np.asarray(s, dtype=float)
        else:
            s = arm['scores'][test_idx]
            arm['test_scores'] = s.tolist()
            p_test = np.asarray(arm['p'], dtype=float)[test_idx]
            del arm['scores'], arm['p']
        arm['auroc'] = compute_auc(arm['test_scores'], y_test.tolist())
        arm['recall_at_budget'] = _recall_at_budget(s, y_test, budget)
        arm['brier'] = float(np.mean((p_test - y_test) ** 2))

    # ---- 阶梯 + 可加性（逐模型）----
    ladder = {}
    for mk in model_keys:
        ind = arms[f'ind:{mk}']['auroc']
        singles = {fs: arms[f'{fs}:{mk}']['auroc'] - ind
                   for fs in ('typed', 'window', 'pi')}
        for fs, d in singles.items():
            ladder[f'{mk}:{fs}_minus_ind'] = d
        all_gain = arms[f'all:{mk}']['auroc'] - ind
        ladder[f'{mk}:all_minus_ind'] = all_gain
        best_single = max(singles, key=singles.get)
        ladder[f'{mk}:all_minus_best_single'] = (
            arms[f'all:{mk}']['auroc'] - arms[f'{best_single}:{mk}']['auroc'])
        ladder[f'{mk}:synergy_all_minus_sum_singles'] = (
            all_gain - sum(singles.values()))
        oracle_gain = arms['oracle_nu']['auroc'] - ind
        ladder[f'{mk}:oracle_minus_ind'] = oracle_gain
        ladder[f'{mk}:oracle_minus_all'] = (
            arms['oracle_nu']['auroc'] - arms[f'all:{mk}']['auroc'])
        ladder[f'{mk}:realization_ratio'] = (
            float(all_gain / oracle_gain) if oracle_gain > 1e-9 else
            float('nan'))
        if not skip_gnn:
            ladder[f'{mk}:gnn_pi_minus_all'] = (
                arms['gnn_pi']['auroc'] - arms[f'all:{mk}']['auroc'])

    def _pair(a, b):
        return delong_paired_test(y_test.tolist(),
                                  arms[a]['test_scores'],
                                  arms[b]['test_scores'])

    delong = {}
    for mk in model_keys:
        for fs in ('typed', 'window', 'pi', 'all'):
            delong[f'{fs}_vs_ind[{mk}]'] = _pair(f'{fs}:{mk}', f'ind:{mk}')
        if not skip_gnn:
            delong[f'all_vs_gnn_pi[{mk}]'] = _pair(f'all:{mk}', 'gnn_pi')

    return {
        'seed': seed,
        'n_contacts': n_contacts,
        'target_rate': target_rate,
        'budget': budget,
        'k_hat': float(k_hat),
        'n_test': int(len(test_idx)),
        'test_labels': y_test.tolist(),
        'arms': arms,
        'ladder': ladder,
        'delong': delong,
    }


_DELONG_MAP = {}
for _mk in MODEL_KEYS:
    for _fs in ('typed', 'window', 'pi', 'all'):
        _DELONG_MAP[f'{_mk}:{_fs}_minus_ind'] = f'{_fs}_vs_ind[{_mk}]'
    _DELONG_MAP[f'{_mk}:all_minus_best_single'] = f'all_vs_ind[{_mk}]'
    _DELONG_MAP[f'{_mk}:gnn_pi_minus_all'] = f'all_vs_gnn_pi[{_mk}]'


def run_multi_seed_combined_ablation(n_contacts=400, n_seeds=20,
                                     seed_start=101, target_rate=0.25,
                                     budget=0.25, gnn_epochs=800,
                                     model_keys=MODEL_KEYS, skip_gnn=False,
                                     n_bootstrap=2000):
    """多种子特征化阶梯消融：效应量 + 种子级 bootstrap CI + DeLong 汇总。"""
    model_keys = tuple(model_keys)
    arm_keys = ([f'{fs}:{mk}' for fs in FEATURE_ARMS for mk in model_keys]
                + (['seir', 'gnn_pi', 'oracle_nu'] if not skip_gnn
                   else ['seir', 'oracle_nu']))
    ladder_keys = [f'{mk}:{k}' for mk in model_keys for k in (
        'typed_minus_ind', 'window_minus_ind', 'pi_minus_ind',
        'all_minus_ind', 'all_minus_best_single',
        'synergy_all_minus_sum_singles', 'oracle_minus_ind',
        'oracle_minus_all', 'realization_ratio')
        ] + ([] if skip_gnn else
             [f'{mk}:gnn_pi_minus_all' for mk in model_keys])
    seeds_reports = []
    for i in range(n_seeds):
        rep = run_combined_ablation_once(
            n_contacts=n_contacts, seed=seed_start + i,
            target_rate=target_rate, budget=budget,
            gnn_epochs=gnn_epochs, model_keys=model_keys, skip_gnn=skip_gnn)
        seeds_reports.append({
            'seed': rep['seed'],
            'auroc': {k: rep['arms'][k]['auroc'] for k in arm_keys},
            'recall_at_budget': {k: rep['arms'][k]['recall_at_budget']
                                 for k in arm_keys},
            'brier': {k: rep['arms'][k]['brier'] for k in arm_keys},
            'ladder': rep['ladder'],
            'delong_p': {k: v['p_value'] for k, v in rep['delong'].items()},
            'positive_rate': float(np.mean(rep['test_labels'])),
        })

    rng = np.random.RandomState(seed_start)

    def _boot_ci(values):
        values = np.asarray(values, dtype=float)
        means = [values[rng.randint(0, len(values), len(values))].mean()
                 for _ in range(n_bootstrap)]
        return [float(np.percentile(means, 2.5)),
                float(np.percentile(means, 97.5))]

    arm_summary = {}
    for k in arm_keys:
        entry = {}
        for metric in ('auroc', 'recall_at_budget', 'brier'):
            vals = [s[metric][k] for s in seeds_reports]
            entry['mean_' + metric] = float(np.mean(vals))
            entry['bootstrap_ci_' + metric] = _boot_ci(vals)
        arm_summary[k] = entry

    ladder_summary = {}
    for k in ladder_keys:
        vals = [s['ladder'][k] for s in seeds_reports]
        ci = _boot_ci(vals)
        entry = {
            'mean': float(np.nanmean(vals)),
            'bootstrap_ci': ci,
            'positive_seeds': int(sum(v > 0 for v in vals
                                      if not np.isnan(v))),
            'ci_excludes_zero': bool((ci[0] > 0.0) or (ci[1] < 0.0)),
        }
        pair = _DELONG_MAP.get(k)
        if pair:
            p_vals = [s['delong_p'][pair] for s in seeds_reports
                      if pair in s['delong_p']]
            if p_vals:
                entry['delong_significant_frac'] = float(np.mean(
                    [p < 0.05 for p in p_vals]))
        ladder_summary[k] = entry

    # ---- P2 结论：可加性 + 兑现比例（逐模型）----
    additivity = {}
    for mk in model_keys:
        all_gain = ladder_summary[f'{mk}:all_minus_ind']
        syn = ladder_summary[f'{mk}:synergy_all_minus_sum_singles']
        real = ladder_summary[f'{mk}:realization_ratio']
        featureization_wins = bool(
            all_gain['mean'] > 0.0 and all_gain['bootstrap_ci'][0] > 0.0)
        if syn['bootstrap_ci'][0] > 0.0:
            verdict = 'super_additive（互补协同）'
        elif syn['bootstrap_ci'][1] < 0.0:
            verdict = 'sub_additive（特征冗余主导）'
        else:
            verdict = 'additive_within_noise（可加）'
        additivity[mk] = {
            'all_minus_ind': all_gain,
            'synergy': syn,
            'realization_ratio': real,
            'featureization_wins': featureization_wins,
            'additivity_verdict': verdict,
        }

    return {
        'design': {
            'n_contacts': n_contacts, 'n_seeds': n_seeds,
            'seed_start': seed_start, 'target_rate': target_rate,
            'budget': budget, 'gnn_epochs': gnn_epochs,
            'model_keys': list(model_keys), 'skip_gnn': skip_gnn,
            'frozen_model_params':
                'scoring/ml/training.py MODEL_REGISTRY default_params'
                '（random_forest / lightgbm，只注入种子）',
            'split': 'stratified 50/50 half-split（接触者空间；'
                     '训练半区拟合/校准，测试半区报告）',
            'dgp': 'combined_mechanism_network_v1',
            'metrics': 'AUROC（判别力）+ recall@budget（追踪效率）'
                       '+ Brier（校准）',
        },
        'seeds': seeds_reports,
        'arm_summary': arm_summary,
        'ladder_summary': ladder_summary,
        'additivity': additivity,
    }
