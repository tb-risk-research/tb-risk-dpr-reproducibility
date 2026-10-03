#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序接触网络五臂消融：时序 GNN 的机制可行性验证。

消融阶梯（隔离每一层贡献，回答"第二层：时序 GNN（接触时间维度）"）：

  ┌────────────────────────────────────────────────────────────────┐
  │ arm1 individual    个体特征 logistic（宿主 + time_span 总量等   │
  │                    无时序聚合）                                  │
  │ arm2 static_agg    + 跨窗等权邻居均值聚合（现行网络层语义：     │
  │                    所有接触不分时间一样权重——时序盲）           │
  │ arm3 window_agg    + 5 窗独立邻居均值（时序区分可学上界）       │
  │ arm4 stgnn         每窗 GAT + 窗间 GRU 端到端（时序感知）       │
  │ arm5 stgnn_merged  同结构时序盲对照（全窗并入单一静态子图，     │
  │                    窗标注抹除）                                  │
  └────────────────────────────────────────────────────────────────┘

  归因：
    arm2 − arm1   聚合本身的价值（预期 ≈ 0：跨窗均值聚合无时间区分）
    arm3 − arm2   时序区分信息的可学上界（构造性信息不对称 → 应 > 0）
    arm4 − arm1   端到端 STGNN 的总增量
    arm4 − arm5   **时序感知 vs 时序盲**（核心机制检验：时间切片 +
                  窗间状态传递的净贡献）

协议（沿用 typed_ablation / hybrid_validation / P3 教训）：
  - 分层 50/50 半区（接触者空间）：训练半区拟合、测试半区报告；
  - 五臂同一 split；个体特征预算一致（arm3/4/5 的增量来自时序/
    图信息，特征预算差异即机制本身）；
  - Bernoulli 概率抽签标签（非 top-k 排序选病）；
  - 逐种子 DeLong 配对检验 + 种子级 bootstrap CI（复用 layer_ablation
    的 delong_paired_test）。

诚实边界：时间信息不对称由 DGP v1 构造保证（边特征全窗口同分布 +
窗口指派独立于一切个体可观测量 + 时间差异只在衰减权重与窗口标注）。
本实验回答"机制能否兑现"，不回答"真实世界必然存在"（需带接触
时间戳的真实数据裁决）。
"""

import numpy as np

from .layer_ablation import delong_paired_test
from .threshold_spec import compute_auc
from .temporal_network import (
    build_temporal_network, individual_features, static_agg_features,
    window_agg_features,
)


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


def _fit_logistic(X, y, train_idx):
    """在训练半区拟合 logistic 回归，返回全接触者分数（0-1）。"""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(X[train_idx], y[train_idx])
    return model.predict_proba(X)[:, 1].tolist()


def run_temporal_ablation_once(n_contacts=400, seed=42, epochs=500,
                               target_rate=0.25, stgnn_seed=None,
                               stgnn_merged_seed=None):
    """单次五臂消融（一个种子 = 一个网络 + 一次训练）。

    Args:
        n_contacts: 接触者数（另有 M 个指示病例节点参与图传播）
        seed: DGP 种子（同时决定分层 split 与 STGNN 默认种子）
        epochs: STGNN 训练轮数（验证值 500）
        target_rate: 目标阳性率（k 校准）
        stgnn_seed / stgnn_merged_seed: STGNN 异质/时序盲臂的 torch
            种子（默认 = seed，保证两臂同初始化流）

    Returns:
        dict: arms（各臂 auroc + test_scores）、test_labels、ladder
        （阶梯差值）、delong（逐对检验）
    """
    from ..ml.gnn.temporal_gnn import train_temporal_gnn

    net = build_temporal_network(n_contacts=n_contacts,
                                 target_rate=target_rate,
                                 random_state=seed)
    labels = np.asarray(net['labels'], dtype=int)
    rng = np.random.RandomState(seed)
    train_idx, test_idx = _stratified_split(labels, rng)

    Xi = individual_features(net)
    Xs = static_agg_features(net)
    Xw = window_agg_features(net)

    arms = {}
    arms['individual'] = {
        'test_scores': _fit_logistic(Xi, labels, train_idx)}
    arms['static_agg'] = {
        'test_scores': _fit_logistic(Xs, labels, train_idx)}
    arms['window_agg'] = {
        'test_scores': _fit_logistic(Xw, labels, train_idx)}
    arms['stgnn'] = {
        'test_scores': train_temporal_gnn(
            net, labels, train_contacts=train_idx, epochs=epochs,
            seed=(seed if stgnn_seed is None else stgnn_seed))['scores']}
    arms['stgnn_merged'] = {
        'test_scores': train_temporal_gnn(
            net, labels, train_contacts=train_idx, epochs=epochs,
            seed=(seed if stgnn_merged_seed is None else stgnn_merged_seed),
            windows_merged=True)['scores']}

    y_test = labels[test_idx].tolist()
    for name, arm in arms.items():
        s = np.asarray(arm['test_scores'], dtype=float)[test_idx]
        arm['test_scores'] = s.tolist()
        arm['auroc'] = compute_auc(s.tolist(), y_test)

    ladder = {
        'static_minus_individual':
            arms['static_agg']['auroc'] - arms['individual']['auroc'],
        'window_minus_static':
            arms['window_agg']['auroc'] - arms['static_agg']['auroc'],
        'stgnn_minus_individual':
            arms['stgnn']['auroc'] - arms['individual']['auroc'],
        'stgnn_minus_static_agg':
            arms['stgnn']['auroc'] - arms['static_agg']['auroc'],
        'stgnn_minus_stgnn_merged':
            arms['stgnn']['auroc'] - arms['stgnn_merged']['auroc'],
    }

    def _pair(a, b):
        return delong_paired_test(
            y_test, arms[a]['test_scores'], arms[b]['test_scores'])

    delong = {
        'stgnn_vs_individual': _pair('stgnn', 'individual'),
        'stgnn_vs_static_agg': _pair('stgnn', 'static_agg'),
        'stgnn_vs_window_agg': _pair('stgnn', 'window_agg'),
        'stgnn_vs_stgnn_merged': _pair('stgnn', 'stgnn_merged'),
    }

    return {
        'seed': seed,
        'n_contacts': n_contacts,
        'n_index_cases': int(net['M']),
        'target_rate': target_rate,
        'n_test': int(len(test_idx)),
        'test_labels': y_test,
        'arms': arms,
        'ladder': ladder,
        'delong': delong,
    }


def run_multi_seed_temporal_ablation(n_contacts=400, n_seeds=20,
                                     seed_start=101, epochs=500,
                                     target_rate=0.25, n_bootstrap=2000):
    """多种子五臂消融：效应量 + 种子级 bootstrap CI + DeLong 汇总。

    Returns:
        dict: seeds（逐种子五臂 AUROC + 阶梯）、每臂均值、每级阶梯的
        均值/CI/DeLong 显著计数、conclusion
    """
    arm_keys = ['individual', 'static_agg', 'window_agg',
                'stgnn', 'stgnn_merged']
    ladder_keys = ['static_minus_individual', 'window_minus_static',
                   'stgnn_minus_individual', 'stgnn_minus_static_agg',
                   'stgnn_minus_stgnn_merged']
    seeds_reports = []
    for i in range(n_seeds):
        rep = run_temporal_ablation_once(
            n_contacts=n_contacts, seed=seed_start + i, epochs=epochs,
            target_rate=target_rate)
        seeds_reports.append({
            'seed': rep['seed'],
            'auroc': {k: rep['arms'][k]['auroc'] for k in arm_keys},
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
        vals = [s['auroc'][k] for s in seeds_reports]
        arm_summary[k] = {
            'mean_auroc': float(np.mean(vals)),
            'bootstrap_ci': _boot_ci(vals),
        }

    ladder_summary = {}
    delong_map = {
        'static_minus_individual': 'stgnn_vs_individual',
        'window_minus_static': 'stgnn_vs_static_agg',
        'stgnn_minus_individual': 'stgnn_vs_individual',
        'stgnn_minus_static_agg': 'stgnn_vs_static_agg',
        'stgnn_minus_stgnn_merged': 'stgnn_vs_stgnn_merged',
    }
    for k in ladder_keys:
        vals = [s['ladder'][k] for s in seeds_reports]
        p_vals = [s['delong_p'][delong_map[k]] for s in seeds_reports]
        ci = _boot_ci(vals)
        ladder_summary[k] = {
            'mean': float(np.mean(vals)),
            'bootstrap_ci': ci,
            'positive_seeds': int(sum(v > 0 for v in vals)),
            'delong_significant_frac': float(np.mean(
                [p < 0.05 for p in p_vals])),
            'ci_excludes_zero': bool(
                (ci[0] > 0.0) or (ci[1] < 0.0)),
        }

    # 核心结论：两级机制检验
    core = ladder_summary['stgnn_minus_stgnn_merged']
    network_gain = ladder_summary['stgnn_minus_individual']
    temporal_mechanism_contributes = bool(
        core['mean'] > 0.0 and core['bootstrap_ci'][0] > 0.0)
    network_increment_exists = bool(
        network_gain['mean'] > 0.0
        and network_gain['bootstrap_ci'][0] > 0.0)

    return {
        'design': {
            'n_contacts': n_contacts, 'n_seeds': n_seeds,
            'epochs': epochs, 'target_rate': target_rate,
            'split': 'stratified 50/50 half-split（接触者空间；'
                     '训练半区拟合，测试半区报告）',
            'label_mechanism':
                'Bernoulli 抽签 p = 1-exp(-k·Σdecay(window)·intensity·'
                'infectivity(src)×宿主乘数)（DGP v1：时间切片 + 衰减'
                '权重信息不对称，半衰期 8 周）',
            'dgp': 'temporal_contact_network_v1',
        },
        'seeds': seeds_reports,
        'arm_summary': arm_summary,
        'ladder_summary': ladder_summary,
        'network_increment_exists': network_increment_exists,
        'temporal_mechanism_contributes': temporal_mechanism_contributes,
        'conclusion': (
            '网络层增量存在（STGNN−个体 ΔAUROC 均值 '
            f"{network_gain['mean']:.4f}，CI "
            f"[{network_gain['bootstrap_ci'][0]:.4f}, "
            f"{network_gain['bootstrap_ci'][1]:.4f}]）；"
            + ('时序感知显著优于时序盲对照（ΔAUROC 均值 '
               f"{core['mean']:.4f}，CI [{core['bootstrap_ci'][0]:.4f}, "
               f"{core['bootstrap_ci'][1]:.4f}]）"
               if temporal_mechanism_contributes else
               '时序感知方向为正但 CI 含 0（ΔAUROC 均值 '
               f"{core['mean']:.4f}，CI [{core['bootstrap_ci'][0]:.4f}, "
               f"{core['bootstrap_ci'][1]:.4f}]）")
            + '——机制可行性口径（时间信息不对称为 DGP 构造性设定）'),
    }
