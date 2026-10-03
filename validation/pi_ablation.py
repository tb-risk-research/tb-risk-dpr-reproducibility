#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PI 接触网络四臂消融：物理信息 GNN 的机制可行性验证。

消融阶梯（用户"PI-GNN"规格的四组对照 + oracle 参考）：

  ┌────────────────────────────────────────────────────────────────┐
  │ arm1 seir             纯 SEIR：k̂·λ 排序（冻结物理 + 单参数     │
  │                       校准，无学习）                            │
  │ arm2 gnn              纯 GNN：14 维基础特征端到端（无物理）    │
  │ arm3 pi_feature       PI 特征级：物理特征拼接 + 残差融合结构   │
  │                       （p = 1−exp(−(λ̃+δ))，仅 L_label）       │
  │ arm4 pi_feature_loss  PI 特征+损失级：+ L_bounded/L_consistency │
  │ ref  oracle_nu        真值 ν 排序（DGP 内部，可达上界参考）    │
  └────────────────────────────────────────────────────────────────┘

  归因（用户规格："比较追踪效率和判别力"）：
    arm3 − arm1   **残差学习检验（核心）**：物理锚定下 GNN 残差能否
                  兑现为增量（修正均匀传染源假设）
    arm3 − arm2   **物理锚定检验**：物理先验 vs 从零学（小样本效益）
    arm4 − arm3   损失级注入的净贡献（物理一致性约束）
    oracle − arm1 可达上界（DGP 残差空间）

  指标：判别力（AUROC）+ 追踪效率（recall@25% 预算——按分数降序
  筛查前 25% 接触者的检出率）+ 校准（Brier，p 尺度）。

协议（沿用 typed/temporal ablation）：
  - 分层 50/50 半区（接触者空间）；四臂同一 split；
  - k̂/模型拟合只用训练半区；测试半区报告全部指标；
  - 逐种子 DeLong 配对检验（AUROC）+ 种子级 bootstrap CI。

诚实边界：物理公式精确已知 + 残差（s̄）图可读均为 DGP 构造性
设定。本实验回答"物理锚定 + 有界残差能否兑现为判别/追踪增量"，
不回答"真实世界残差规模是否如此"。
"""

import numpy as np

from .layer_ablation import delong_paired_test
from .threshold_spec import compute_auc
from .pi_network import (
    build_pi_network, physics_lambda, calibrate_physics_k,
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


def _recall_at_budget(scores, y, budget=0.25):
    """追踪效率：按分数降序筛查前 budget 比例接触者的阳性检出率。"""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    n_top = max(1, int(np.ceil(budget * len(y))))
    top = np.argsort(-scores)[:n_top]
    return float(y[top].sum() / max(y.sum(), 1))


def run_pi_ablation_once(n_contacts=400, seed=42, pi_epochs=800,
                         gnn_epochs=500, target_rate=0.25, budget=0.25):
    """单次四臂消融（一个种子 = 一个网络 + 一次训练）。

    pi_epochs/gnn_epochs 分别为 PI 臂（v11 验证值 800，lr=0.05
    线性读出）与纯 GNN 臂（第一层沿用值 500，lr=0.02）的训练
    轮数——两臂超参各取其验证配置，消融只比较结构/信息集差异。

    Returns:
        dict: arms（各臂 auroc / recall_at_budget / brier +
        test_scores）、test_labels、ladder（AUROC 阶梯差）、
        recall_ladder、delong（逐对检验）
    """
    from ..ml.gnn.pi_gnn import train_pi_gnn, train_pure_gnn

    net = build_pi_network(n_contacts=n_contacts, target_rate=target_rate,
                           random_state=seed)
    labels = np.asarray(net['labels'], dtype=int)
    rng = np.random.RandomState(seed)
    train_idx, test_idx = _stratified_split(labels, rng)

    # arm1 纯 SEIR：冻结物理 + k̂ 校准（训练半区）
    k_hat = calibrate_physics_k(net, labels, train_idx)
    lam_t = k_hat * physics_lambda(net)

    arms = {}
    arms['seir'] = {'scores': lam_t, 'p': 1.0 - np.exp(-lam_t)}
    r = train_pure_gnn(net, labels, train_idx, epochs=gnn_epochs,
                       seed=seed)
    arms['gnn'] = {'scores': r['score'], 'p': r['p']}
    r = train_pi_gnn(net, labels, train_idx, mode='feature',
                     epochs=pi_epochs, seed=seed)
    arms['pi_feature'] = {'scores': r['score'], 'p': r['p'],
                          'delta_t_abs_mean': float(
                              np.abs(r['delta_t']).mean())}
    r = train_pi_gnn(net, labels, train_idx, mode='feature_loss',
                     epochs=pi_epochs, seed=seed)
    arms['pi_feature_loss'] = {'scores': r['score'], 'p': r['p'],
                               'delta_t_abs_mean': float(
                                   np.abs(r['delta_t']).mean())}
    arms['oracle_nu'] = {
        'scores': np.asarray(net['nu'], dtype=float),
        'p': 1.0 - np.exp(-net['k_calibration'] * np.asarray(net['nu'],
                                                             dtype=float))}

    y_test = labels[test_idx]
    for name, arm in arms.items():
        s = arm['scores'][test_idx]
        arm['test_scores'] = s.tolist()
        arm['auroc'] = compute_auc(s.tolist(), y_test.tolist())
        arm['recall_at_budget'] = _recall_at_budget(s, y_test, budget)
        arm['brier'] = float(np.mean(
            (np.asarray(arm['p'], dtype=float)[test_idx] - y_test) ** 2))
        del arm['scores'], arm['p']

    ladder = {
        'pi_feature_minus_seir':
            arms['pi_feature']['auroc'] - arms['seir']['auroc'],
        'pi_feature_minus_gnn':
            arms['pi_feature']['auroc'] - arms['gnn']['auroc'],
        'pi_feature_loss_minus_seir':
            arms['pi_feature_loss']['auroc'] - arms['seir']['auroc'],
        'pi_feature_loss_minus_gnn':
            arms['pi_feature_loss']['auroc'] - arms['gnn']['auroc'],
        'pi_feature_loss_minus_pi_feature':
            arms['pi_feature_loss']['auroc'] - arms['pi_feature']['auroc'],
        'oracle_minus_seir':
            arms['oracle_nu']['auroc'] - arms['seir']['auroc'],
    }
    recall_ladder = {
        'pi_feature_minus_seir':
            arms['pi_feature']['recall_at_budget']
            - arms['seir']['recall_at_budget'],
        'pi_feature_minus_gnn':
            arms['pi_feature']['recall_at_budget']
            - arms['gnn']['recall_at_budget'],
        'pi_feature_loss_minus_seir':
            arms['pi_feature_loss']['recall_at_budget']
            - arms['seir']['recall_at_budget'],
    }

    def _pair(a, b):
        return delong_paired_test(
            y_test.tolist(), arms[a]['test_scores'], arms[b]['test_scores'])

    delong = {
        'pi_feature_vs_seir': _pair('pi_feature', 'seir'),
        'pi_feature_vs_gnn': _pair('pi_feature', 'gnn'),
        'pi_feature_loss_vs_seir': _pair('pi_feature_loss', 'seir'),
        'pi_feature_loss_vs_gnn': _pair('pi_feature_loss', 'gnn'),
        'pi_feature_loss_vs_pi_feature':
            _pair('pi_feature_loss', 'pi_feature'),
    }

    return {
        'seed': seed,
        'n_contacts': n_contacts,
        'n_index_cases': int(net['M']),
        'target_rate': target_rate,
        'budget': budget,
        'n_test': int(len(test_idx)),
        'test_labels': y_test.tolist(),
        'arms': arms,
        'ladder': ladder,
        'recall_ladder': recall_ladder,
        'delong': delong,
    }


def run_multi_seed_pi_ablation(n_contacts=400, n_seeds=20, seed_start=101,
                               pi_epochs=800, gnn_epochs=500,
                               target_rate=0.25, budget=0.25,
                               n_bootstrap=2000):
    """多种子四臂消融：效应量 + 种子级 bootstrap CI + DeLong 汇总。

    Returns:
        dict: seeds（逐种子各臂指标 + 阶梯）、每臂指标均值/CI、
        每级阶梯均值/CI/显著计数、conclusion
    """
    arm_keys = ['seir', 'gnn', 'pi_feature', 'pi_feature_loss', 'oracle_nu']
    ladder_keys = ['pi_feature_minus_seir', 'pi_feature_minus_gnn',
                   'pi_feature_loss_minus_seir', 'pi_feature_loss_minus_gnn',
                   'pi_feature_loss_minus_pi_feature', 'oracle_minus_seir']
    delong_map = {
        'pi_feature_minus_seir': 'pi_feature_vs_seir',
        'pi_feature_minus_gnn': 'pi_feature_vs_gnn',
        'pi_feature_loss_minus_seir': 'pi_feature_loss_vs_seir',
        'pi_feature_loss_minus_gnn': 'pi_feature_loss_vs_gnn',
        'pi_feature_loss_minus_pi_feature':
            'pi_feature_loss_vs_pi_feature',
        'oracle_minus_seir': 'pi_feature_vs_seir',   # 参考项无独立检验
    }
    seeds_reports = []
    for i in range(n_seeds):
        rep = run_pi_ablation_once(
            n_contacts=n_contacts, seed=seed_start + i,
            pi_epochs=pi_epochs, gnn_epochs=gnn_epochs,
            target_rate=target_rate, budget=budget)
        seeds_reports.append({
            'seed': rep['seed'],
            'auroc': {k: rep['arms'][k]['auroc'] for k in arm_keys},
            'recall_at_budget': {k: rep['arms'][k]['recall_at_budget']
                                 for k in arm_keys},
            'brier': {k: rep['arms'][k]['brier'] for k in arm_keys},
            'delta_t_abs_mean': {
                k: rep['arms'][k]['delta_t_abs_mean']
                for k in ('pi_feature', 'pi_feature_loss')},
            'ladder': rep['ladder'],
            'recall_ladder': rep['recall_ladder'],
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
        p_vals = [s['delong_p'][delong_map[k]] for s in seeds_reports]
        ci = _boot_ci(vals)
        ladder_summary[k] = {
            'mean': float(np.mean(vals)),
            'bootstrap_ci': ci,
            'positive_seeds': int(sum(v > 0 for v in vals)),
            'delong_significant_frac': float(np.mean(
                [p < 0.05 for p in p_vals])),
            'ci_excludes_zero': bool((ci[0] > 0.0) or (ci[1] < 0.0)),
        }
    recall_ladder_summary = {}
    for k, vals in (
            ('pi_feature_minus_seir',
             [s['recall_ladder']['pi_feature_minus_seir']
              for s in seeds_reports]),
            ('pi_feature_minus_gnn',
             [s['recall_ladder']['pi_feature_minus_gnn']
              for s in seeds_reports]),
            ('pi_feature_loss_minus_seir',
             [s['recall_ladder']['pi_feature_loss_minus_seir']
              for s in seeds_reports])):
        ci = _boot_ci(vals)
        recall_ladder_summary[k] = {
            'mean': float(np.mean(vals)),
            'bootstrap_ci': ci,
            'positive_seeds': int(sum(v > 0 for v in vals)),
            'ci_excludes_zero': bool((ci[0] > 0.0) or (ci[1] < 0.0)),
        }

    # 核心结论：两级机制检验
    residual = ladder_summary['pi_feature_minus_seir']
    anchor = ladder_summary['pi_feature_minus_gnn']
    residual_learning_works = bool(
        residual['mean'] > 0.0 and residual['bootstrap_ci'][0] > 0.0)
    physics_anchor_helps = bool(
        anchor['mean'] > 0.0 and anchor['bootstrap_ci'][0] > 0.0)

    return {
        'design': {
            'n_contacts': n_contacts, 'n_seeds': n_seeds,
            'pi_epochs': pi_epochs, 'gnn_epochs': gnn_epochs,
            'target_rate': target_rate,
            'budget': budget,
            'split': 'stratified 50/50 half-split（接触者空间；'
                     '训练半区拟合/校准，测试半区报告）',
            'label_mechanism':
                'Bernoulli 抽签 p = 1-exp(-k·host·Λ·corr·s̄)（DGP v1：'
                '物理 λ = host·Λ·corr 个体可观测推导，残差 s̄ 图可读）',
            'dgp': 'pi_contact_network_v1',
            'metrics': 'AUROC（判别力）+ recall@budget（追踪效率）'
                       '+ Brier（校准）',
        },
        'seeds': seeds_reports,
        'arm_summary': arm_summary,
        'ladder_summary': ladder_summary,
        'recall_ladder_summary': recall_ladder_summary,
        'residual_learning_works': residual_learning_works,
        'physics_anchor_helps': physics_anchor_helps,
        'conclusion': (
            '残差学习检验：PI 特征级 − 纯 SEIR ΔAUROC 均值 '
            f"{residual['mean']:.4f}，CI "
            f"[{residual['bootstrap_ci'][0]:.4f}, "
            f"{residual['bootstrap_ci'][1]:.4f}]，"
            f"{residual['positive_seeds']}/{n_seeds} 种子为正；"
            f"{'通过' if residual_learning_works else '未通过'}。"
            '物理锚定检验：PI 特征级 − 纯 GNN ΔAUROC 均值 '
            f"{anchor['mean']:.4f}，CI "
            f"[{anchor['bootstrap_ci'][0]:.4f}, "
            f"{anchor['bootstrap_ci'][1]:.4f}]，"
            f"{'通过' if physics_anchor_helps else '未通过'}"
            '——机制可行性口径（物理精确已知 + 残差图可读均为 DGP '
            '构造性设定）'),
    }
