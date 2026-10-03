#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""类型化边网络五臂消融：异质边注意力的机制可行性验证。

消融阶梯（隔离每一层贡献，回答"第一层：异质边 + 注意力"）：

  ┌────────────────────────────────────────────────────────────────┐
  │ arm1 individual   个体特征 logistic（宿主 + 无类型暴露聚合）    │
  │ arm2 mean_agg     + 等权邻居均值聚合（现行网络层语义）          │
  │ arm3 channel_agg  + 5 类型独立邻居均值（类型区分可学上界）      │
  │ arm4 gat_hetero   2层 HeteroGAT 端到端（类型感知注意力）        │
  │ arm5 gat_homo     同结构对照（5 类边并入单通道，无类型区分）    │
  └────────────────────────────────────────────────────────────────┘

  归因：
    arm2 − arm1  聚合本身的价值（预期 ≈ 0：均值聚合无类型区分）
    arm3 − arm2  类型区分信息的可学上界（构造性信息不对称 → 应 > 0）
    arm4 − arm1  端到端 GAT 的总增量
    arm4 − arm5  **类型感知注意力 vs 无类型注意力**（核心机制检验）

协议（沿用 hybrid_validation / P3 教训）：
  - 分层 50/50 半区（接触者空间）：训练半区拟合、测试半区报告
    （全量训练+全量评估会得到 RF/GAT 记忆化 AUROC=1.0 假象）；
  - 五臂同一 split；个体特征预算一致（arm3/4/5 的增量来自类型/
    图信息，特征预算差异即机制本身）；
  - Bernoulli 概率抽签标签（非 top-k 排序选病）；
  - 逐种子 DeLong 配对检验 + 种子级 bootstrap CI（复用 layer_ablation
    的 delong_paired_test）。

诚实边界：信息不对称由 DGP v3 构造保证（边特征全类型同分布 +
类型差异只在 β 与边类型标注 + 传染源 infectivity 只在图）。本实验
回答"机制能否兑现"，不回答"真实世界必然存在"。
"""

import numpy as np

from .layer_ablation import delong_paired_test
from .threshold_spec import compute_auc
from .typed_network import (
    build_typed_network, individual_features, mean_agg_features,
    channel_agg_features,
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


def run_typed_ablation_once(n_contacts=400, seed=42, epochs=500,
                            target_rate=0.25, gat_seed=None,
                            gat_homo_seed=None):
    """单次五臂消融（一个种子 = 一个网络 + 一次训练）。

    Args:
        n_contacts: 接触者数（另有 M 个指示病例节点参与图传播）
        seed: DGP 种子（同时决定分层 split 与 GAT 默认种子）
        epochs: GAT 训练轮数（验证值 500）
        target_rate: 目标阳性率（k 校准）
        gat_seed / gat_homo_seed: GAT 异质/同质臂的 torch 种子
            （默认 = seed，保证两臂同初始化流）

    Returns:
        dict: arms（各臂 auroc + test_scores）、test_labels、ladder
        （阶梯差值）、delong（逐对检验）
    """
    from ..ml.gnn.typed_gat import train_typed_gat

    net = build_typed_network(n_contacts=n_contacts,
                              target_rate=target_rate, random_state=seed)
    labels = np.asarray(net['labels'], dtype=int)
    rng = np.random.RandomState(seed)
    train_idx, test_idx = _stratified_split(labels, rng)

    Xi = individual_features(net)
    Xm = mean_agg_features(net)
    Xc = channel_agg_features(net)

    arms = {}
    arms['individual'] = {
        'test_scores': _fit_logistic(Xi, labels, train_idx)}
    arms['mean_agg'] = {
        'test_scores': _fit_logistic(Xm, labels, train_idx)}
    arms['channel_agg'] = {
        'test_scores': _fit_logistic(Xc, labels, train_idx)}
    arms['gat_hetero'] = {
        'test_scores': train_typed_gat(
            net, labels, train_contacts=train_idx, epochs=epochs,
            seed=(seed if gat_seed is None else gat_seed))['scores']}
    arms['gat_homo'] = {
        'test_scores': train_typed_gat(
            net, labels, train_contacts=train_idx, epochs=epochs,
            seed=(seed if gat_homo_seed is None else gat_homo_seed),
            edge_types_merged=True)['scores']}

    y_test = labels[test_idx].tolist()
    for name, arm in arms.items():
        s = np.asarray(arm['test_scores'], dtype=float)[test_idx]
        arm['test_scores'] = s.tolist()
        arm['auroc'] = compute_auc(s.tolist(), y_test)

    ladder = {
        'mean_minus_individual':
            arms['mean_agg']['auroc'] - arms['individual']['auroc'],
        'channel_minus_mean':
            arms['channel_agg']['auroc'] - arms['mean_agg']['auroc'],
        'gat_hetero_minus_individual':
            arms['gat_hetero']['auroc'] - arms['individual']['auroc'],
        'gat_hetero_minus_mean_agg':
            arms['gat_hetero']['auroc'] - arms['mean_agg']['auroc'],
        'gat_hetero_minus_gat_homo':
            arms['gat_hetero']['auroc'] - arms['gat_homo']['auroc'],
    }

    def _pair(a, b):
        return delong_paired_test(
            y_test, arms[a]['test_scores'], arms[b]['test_scores'])

    delong = {
        'gat_hetero_vs_individual': _pair('gat_hetero', 'individual'),
        'gat_hetero_vs_mean_agg': _pair('gat_hetero', 'mean_agg'),
        'gat_hetero_vs_channel_agg': _pair('gat_hetero', 'channel_agg'),
        'gat_hetero_vs_gat_homo': _pair('gat_hetero', 'gat_homo'),
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


def run_multi_seed_typed_ablation(n_contacts=400, n_seeds=20, seed_start=101,
                                  epochs=500, target_rate=0.25,
                                  n_bootstrap=2000):
    """多种子五臂消融：效应量 + 种子级 bootstrap CI + DeLong 汇总。

    Returns:
        dict: seeds（逐种子五臂 AUROC + 阶梯）、每臂均值、每级阶梯的
        均值/CI/DeLong 显著计数、conclusion
    """
    arm_keys = ['individual', 'mean_agg', 'channel_agg',
                'gat_hetero', 'gat_homo']
    ladder_keys = ['mean_minus_individual', 'channel_minus_mean',
                   'gat_hetero_minus_individual',
                   'gat_hetero_minus_mean_agg',
                   'gat_hetero_minus_gat_homo']
    seeds_reports = []
    for i in range(n_seeds):
        rep = run_typed_ablation_once(
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
        'mean_minus_individual': 'gat_hetero_vs_individual',
        'channel_minus_mean': 'gat_hetero_vs_mean_agg',
        'gat_hetero_minus_individual': 'gat_hetero_vs_individual',
        'gat_hetero_minus_mean_agg': 'gat_hetero_vs_mean_agg',
        'gat_hetero_minus_gat_homo': 'gat_hetero_vs_gat_homo',
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
    core = ladder_summary['gat_hetero_minus_gat_homo']
    network_gain = ladder_summary['gat_hetero_minus_individual']
    type_attention_contributes = bool(
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
                'Bernoulli 抽签 p = 1-exp(-k·Σβ_type·intensity·'
                'infectivity(src)×宿主乘数)（DGP v3：传染源节点 + '
                '边特征全类型同分布）',
            'dgp': 'typed_contact_network_v3',
        },
        'seeds': seeds_reports,
        'arm_summary': arm_summary,
        'ladder_summary': ladder_summary,
        'network_increment_exists': network_increment_exists,
        'type_attention_contributes': type_attention_contributes,
        'conclusion': (
            '网络层增量存在（GAT−个体 ΔAUROC 均值 '
            f"{network_gain['mean']:.4f}，CI "
            f"[{network_gain['bootstrap_ci'][0]:.4f}, "
            f"{network_gain['bootstrap_ci'][1]:.4f}]）；"
            + ('类型感知注意力显著优于无类型注意力（ΔAUROC 均值 '
               f"{core['mean']:.4f}，CI [{core['bootstrap_ci'][0]:.4f}, "
               f"{core['bootstrap_ci'][1]:.4f}]）"
               if type_attention_contributes else
               '类型感知注意力方向为正但 CI 含 0（ΔAUROC 均值 '
               f"{core['mean']:.4f}，CI [{core['bootstrap_ci'][0]:.4f}, "
               f"{core['bootstrap_ci'][1]:.4f}]）")
            + '——机制可行性口径（信息不对称为 DGP 构造性设定）'),
    }
