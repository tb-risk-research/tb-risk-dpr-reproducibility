#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机制×图×时序端到端联合 vs 三成员堆叠消融（P5，2026-09-04）。

用户问题："PI-GNN / STGNN / temporal GNN 并列成员、集成只追平不
显著超越的天花板，能否用单一模型内的物理信息联合（SEIR ODE 嵌入
GNN 消息传递 + 时序先验可学习）突破？"

十臂（combined DGP——三层机制共存的唯一测试台；协议沿用
seir_joint / combined_ablation：分层 50/50 半区、训练半区拟合/
校准、测试半区报告、逐种子 DeLong + 种子级 bootstrap）：

  ┌───────────────────────────────────────────────────────────────┐
  │ seir_default    纯物理基线：k̂·λ（默认参数平坦 Λ_ref）        │
  │ features_lgbm   部署特征化：14+11 列 + 冻结 LGBM（现行最优    │
  │                 可部署臂，combined_ablation 复现）            │
  │ member_pi       成员 1：PI-GNN（源池化残差，v11 参数化）      │
  │ member_typed    成员 2：TypedGAT（类型注意力，本模块适配      │
  │                 combined 网——类型边从 membership 重建）       │
  │ member_temporal 成员 3：TemporalGNN（窗切片+GRU，本模块适配   │
  │                 ——窗子图从 membership 重建）                  │
  │ ensemble_mean   三成员概率等权平均（"事后加权"代表）          │
  │ seir_joint      P4a 特征级联合（2 参数，机制无类型/图结构）   │
  │ joint           端到端联合（12 机制参数：SEIR×类型×衰减×源   │
  │                 消息传递，本实验主体）                        │
  │ joint_res       joint + 有界乘性残差头（PI 哲学鲁棒性检查）   │
  │ oracle_nu       真值 ν（含不可观测 intens——不可达上界）      │
  └───────────────────────────────────────────────────────────────┘

阶梯（核心裁决）：
  joint − ensemble_mean          用户核心问题：联合 vs 堆叠
  joint − features_lgbm          联合 vs 现行可部署最优
  joint − seir_joint             图类型×源消息结构的增量（vs 特征级）
  joint_res − joint              残差头增量（intens 缺失可否部分回收）
  ensemble − best_member         验证"集成只追平"前提
  joint − best_alternative       天花板突破检验（逐种子最优替代臂，
                                 含特征化/成员/集成/P4a——保守口径）
  oracle − joint                 联合模型到上界的残余间隙

参数恢复（机制学习是否学到真机制，尺度不变量）：
  β̂/β*、Î0/I0* 相对误差；Λ̂ 形状误差；类型/衰减相对文献初值漂移
  （本 DGP 真值=文献值——漂移应≈0，学习不应破坏正确的先验）。

成员适配的诚实边界：
  - combined 网的 ef 为扁平同分布边特征，类型/窗口标注只在
    membership 结构中——typed/temporal 成员的图输入由 membership
    重建（合法图可读信息；PI-GNN 原生兼容无需适配）；
  - temporal 成员的窗子图只含 membership 边（同簇互连边无窗标注，
    不进窗切片——信息少于原生 temporal DGP，成员臂口径如实偏保守）；
  - 各成员沿用其模块验证超参（PI-GNN 800ep/lr0.05、typed/temporal
    500ep/lr0.02/wd3e-3）——对成员最有利的公平口径。

诚实边界（DGP 世界）：结论口径 = "本组合机制世界内，端到端联合
能否突破三成员堆叠与特征化天花板"，非真实世界承诺。

P5 主实验结果（n=400 × 20 种子，seed 101-120，
joint_ablation_multiseed_20260904.json，总耗时 55 分钟）：

  AUROC：joint 0.9161 > joint_res 0.9154 > seir_joint 0.8854 >
         features_lgbm 0.8699 > member_pi 0.8526 > ensemble_mean
         0.8416 > seir_default 0.8270 > member_typed 0.8180 >
         member_temporal 0.7678（oracle_nu 上界 0.9301）

  阶梯裁决（种子级 bootstrap CI + 逐种子 DeLong）：
  * joint − ensemble_mean      Δ=+0.0745 CI[+0.0584,+0.0900]
                              19/20 正、DeLong 85%——联合显著超越堆叠
  * joint − features_lgbm      Δ=+0.0462 CI[+0.0355,+0.0582] 20/20 正
  * joint − seir_joint         Δ=+0.0307 CI[+0.0244,+0.0373] 20/20 正
    ——图类型×源消息结构在特征级联合之上仍有显著增量
  * joint − best_alternative   Δ=+0.0256 CI[+0.0184,+0.0332] 19/20 正
    ——逐种子最优替代臂（seir_joint 16 次 / lgbm 3 / ensemble 1）
      口径下仍显著，天花板突破成立
  * joint_res − joint          Δ=−0.0007 CI 含零、DeLong 0%
    ——残差头无增量：机制结构已充分刻画 DGP（intens 缺失由
      源传染性 ŝ 消息通道吸收，无需额外容量）
  * ensemble − best_member     Δ=−0.0170 CI[−0.0270,−0.0075]
    ——"集成只追平（实为劣于最优成员）"前提在 combined 世界复现
  * oracle − joint             Δ=+0.0140，兑现率 86.3%（CI 不含零
    的上界差距）——端到端联合回收了大部分可学信号

  参数恢复（尺度不变量）：β̂ 相对误差 0.43、Î0 0.62、Λ̂ 形状误差
  0.24、Λ̂·d̂ 形状误差 0.40、类型/衰减漂移 L2 ≈ 0.5——个体参数
  恢复中等（尺度不可分性：Λ·decay 乘积稳定但分量漂移），但预测
  增量不受影响（消息使用的是乘积）。

  结论：在三层机制共存的 DGP 内，"SEIR ODE 嵌入消息传递 + 类型
  权重 + 时序衰减 + 源传染性"的单一模型端到端联合，稳定显著超越
  三成员等权集成（+0.075）、最强特征基线（+0.046）、特征级 SEIR
  联合（+0.031）与逐种子最优替代臂（+0.026），兑现 oracle 上界
  86%——"集成只追平"的天花板可被端到端机制联合突破。
"""

import numpy as np

from .layer_ablation import delong_paired_test
from .threshold_spec import compute_auc

ARMS = ('seir_default', 'features_lgbm', 'member_pi', 'member_typed',
        'member_temporal', 'ensemble_mean', 'seir_joint', 'joint',
        'joint_res', 'oracle_nu')


# ==============================================================================
# 成员适配器：combined 网 → typed / temporal 成员图输入
# ==============================================================================

def to_typed_member_graph(net):
    """combined 网 → TypedGATNet 输入（类型边从 membership 重建）。

    边集（全部双向、FEAT_SCALE 归一化）：
      - membership 边（源↔接触者）：类型 = 簇类型；
      - 同簇互连边（接触者↔接触者）：类型 = 簇类型（接触者同属
        多簇时进入多个类型通道——并行类型边，GAT 语义合法）。
    节点特征：pi_network 14 维（全体成员统一可观测口径）。
    """
    import torch

    from ..validation.pi_network import (EDGE_FEATURE_DIM, FEAT_SCALE,
                                         node_features)
    from ..validation.typed_network import EDGE_TYPE_IDS

    M = int(net['M'])
    ef = net['ef']
    X = torch.tensor(node_features(net), dtype=torch.float32)
    edges = {t: [] for t in EDGE_TYPE_IDS}
    # membership 边
    for i, mems in enumerate(net['memberships']):
        for (src, t, _w, _inten, _s) in mems:
            u, v = int(src), M + i
            key = (min(u, v), max(u, v))
            edges[t].append((u, v, ef[key]))
    # 同簇互连边（按簇分组重建）
    by_src = {}
    for i, mems in enumerate(net['memberships']):
        for (src, t, _w, _inten, _s) in mems:
            by_src.setdefault(src, (t, []))[1].append(i)
    for src, (t, members) in by_src.items():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = min(members[a], members[b]), max(members[a],
                                                        members[b])
                key = (M + i, M + j)
                edges[t].append((M + i, M + j, ef[key]))

    ei, ea = [], []
    for t in EDGE_TYPE_IDS:
        lst = edges[t]
        if not lst:
            ei.append(torch.zeros((2, 0), dtype=torch.long))
            ea.append(torch.zeros((0, EDGE_FEATURE_DIM),
                                  dtype=torch.float32))
            continue
        arr = np.array([[u, v] for (u, v, _f) in lst], dtype=np.int64)
        f = np.array([_f for (_u, _v, _f) in lst],
                     dtype=np.float64) / FEAT_SCALE
        ei.append(torch.tensor(
            np.concatenate([arr, arr[:, ::-1]], axis=0).T,
            dtype=torch.long))
        ea.append(torch.tensor(
            np.concatenate([f, f], axis=0), dtype=torch.float32))
    return {'x': X, 'edge_indices': ei, 'edge_attrs': ea}


def to_temporal_member_graph(net):
    """combined 网 → TemporalGNN 输入（窗子图从 membership 重建）。

    窗序列最旧→最新（窗 id 4→0，GRU 末步 = 当前风险）；每窗子图
    只含该窗的 membership 边（同簇互连边无窗标注，不进窗切片）。
    节点特征：pi_network 14 维。
    """
    import torch

    from ..validation.pi_network import (EDGE_FEATURE_DIM, FEAT_SCALE,
                                         node_features)
    from ..validation.temporal_network import NUM_WINDOWS

    M = int(net['M'])
    ef = net['ef']
    X = torch.tensor(node_features(net), dtype=torch.float32)
    windows, window_ids = [], []
    for w in range(NUM_WINDOWS - 1, -1, -1):        # 最旧 → 最新
        lst = []
        for i, mems in enumerate(net['memberships']):
            for (src, _t, mw, _inten, _s) in mems:
                if int(mw) != w:
                    continue
                u, v = int(src), M + i
                lst.append((u, v, ef[(min(u, v), max(u, v))]))
        if lst:
            arr = np.array([[u, v] for (u, v, _f) in lst], dtype=np.int64)
            f = np.array([_f for (_u, _v, _f) in lst],
                         dtype=np.float64) / FEAT_SCALE
            ei = torch.tensor(
                np.concatenate([arr, arr[:, ::-1]], axis=0).T,
                dtype=torch.long)
            ea = torch.tensor(
                np.concatenate([f, f], axis=0), dtype=torch.float32)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
            ea = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)
        windows.append({'edge_index': ei, 'edge_attr': ea})
        window_ids.append(w)
    return {'x': X, 'windows': windows, 'window_ids': window_ids}


def train_member_typed(net, labels, train_contacts, epochs=500, lr=0.02,
                       weight_decay=3e-3, hidden_dim=32, seed=0):
    """TypedGAT 成员臂（协议镜像 train_typed_gnn 原生封装）。"""
    import torch

    from ..ml.gnn.typed_gat import TypedGATNet

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    g = to_typed_member_graph(net)
    M = int(net['M'])
    y = torch.tensor(np.asarray(labels, dtype=np.float32))
    tr_nodes = torch.tensor(np.asarray(train_contacts, dtype=np.int64)
                            + M)
    model = TypedGATNet(node_dim=g['x'].shape[1], hidden_dim=hidden_dim,
                        num_edge_types=len(g['edge_indices']))
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(g['x'], g['edge_indices'], g['edge_attrs'])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[tr_nodes], y[torch.as_tensor(
                np.asarray(train_contacts, dtype=np.int64))])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(g['x'], g['edge_indices'], g['edge_attrs'])
    p = torch.sigmoid(logits)[M:].numpy().astype(np.float64)
    return {'p': p, 'scores': p.tolist()}


def train_member_temporal(net, labels, train_contacts, epochs=500,
                          lr=0.02, weight_decay=3e-3, hidden_dim=32,
                          seed=0):
    """TemporalGNN 成员臂（协议镜像 train_temporal_gnn 原生封装）。"""
    import torch

    from ..ml.gnn.temporal_gnn import TemporalGNN

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    g = to_temporal_member_graph(net)
    M = int(net['M'])
    y = torch.tensor(np.asarray(labels, dtype=np.float32))
    tr_contacts = torch.tensor(np.asarray(train_contacts, dtype=np.int64))
    tr_nodes = tr_contacts + M
    model = TemporalGNN(node_dim=g['x'].shape[1], hidden_dim=hidden_dim,
                        num_windows=len(g['windows']))
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        curve = model(g['x'], g['windows'])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            curve[-1][tr_nodes], y[tr_contacts])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        curve = model(g['x'], g['windows'])
    p = torch.sigmoid(curve[-1])[M:].numpy().astype(np.float64)
    return {'p': p, 'scores': p.tolist()}


# ==============================================================================
# 单种子消融
# ==============================================================================

def _stratified_split(labels, rng):
    """分层 50/50（与 seir_joint / combined_ablation 同实现）。"""
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


def run_joint_ablation_once(n_contacts=400, seed=42, target_rate=0.25,
                            budget=0.25, joint_epochs=1200, lr=0.03,
                            member_epochs=500, pi_epochs=800,
                            prior_lambda=1e-3):
    """单种子十臂消融（combined DGP 一个网络 + 全臂拟合）。"""
    from ..ml.gnn.joint_mech_gnn import train_joint_mech_gnn
    from ..ml.gnn.pi_gnn import train_pi_gnn
    from .combined_ablation import _make_model
    from .combined_network import (build_combined_network, feature_matrix)
    from .pi_network import calibrate_physics_k
    from .seir_joint import learn_windowed_seir
    from .temporal_network import DECAY_WEIGHTS
    from .typed_network import BETA_BY_TYPE, EDGE_TYPE_IDS

    net = build_combined_network(n_contacts=n_contacts,
                                 target_rate=target_rate,
                                 random_state=seed)
    labels = np.asarray(net['labels'], dtype=int)
    rng = np.random.RandomState(seed)
    train_idx, test_idx = _stratified_split(labels, rng)

    # ---- 物理基线 + 部署特征化 ----
    k_hat = calibrate_physics_k(net, labels, train_idx)
    lam_t = k_hat * np.asarray(net['lam'], dtype=float)
    X_all, _ = feature_matrix(net, feature_set='all', k_hat=k_hat)
    lgbm = _make_model('lightgbm', seed)
    lgbm.fit(X_all[train_idx], labels[train_idx])

    # ---- 三成员 ----
    r_pi = train_pi_gnn(net, labels, train_idx, mode='feature',
                        epochs=pi_epochs, seed=seed)
    r_typed = train_member_typed(net, labels, train_idx,
                                 epochs=member_epochs, seed=seed)
    r_temporal = train_member_temporal(net, labels, train_idx,
                                       epochs=member_epochs, seed=seed)

    # ---- P4a 特征级联合 + 端到端联合 ----
    r_sj = learn_windowed_seir(net, labels, train_idx, mode='seir',
                               epochs=joint_epochs, lr=lr, seed=seed)
    r_joint = train_joint_mech_gnn(net, labels, train_idx,
                                   epochs=joint_epochs, lr=lr,
                                   prior_lambda=prior_lambda, seed=seed)
    r_jres = train_joint_mech_gnn(net, labels, train_idx, residual=True,
                                  epochs=joint_epochs, lr=lr,
                                  prior_lambda=prior_lambda, seed=seed)

    arms = {
        'seir_default': {'scores': lam_t,
                         'p': 1.0 - np.exp(-lam_t)},
        'features_lgbm': {
            'scores': lgbm.predict_proba(X_all)[:, 1],
            'p': lgbm.predict_proba(X_all)[:, 1]},
        'member_pi': {'scores': np.asarray(r_pi['score']),
                      'p': np.asarray(r_pi['p'])},
        'member_typed': {'scores': np.asarray(r_typed['scores']),
                         'p': np.asarray(r_typed['p'])},
        'member_temporal': {'scores': np.asarray(r_temporal['scores']),
                            'p': np.asarray(r_temporal['p'])},
        'ensemble_mean': None,          # 下方填充
        'seir_joint': {'scores': np.asarray(r_sj['score']),
                       'p': np.asarray(r_sj['p'])},
        'joint': {'scores': np.asarray(r_joint['score']),
                  'p': np.asarray(r_joint['p'])},
        'joint_res': {'scores': np.asarray(r_jres['score']),
                      'p': np.asarray(r_jres['p'])},
        'oracle_nu': {
            'scores': np.asarray(net['nu'], dtype=float),
            'p': 1.0 - np.exp(-net['k_calibration']
                              * np.asarray(net['nu'], dtype=float))},
    }
    p_mean = (arms['member_pi']['p'] + arms['member_typed']['p']
              + arms['member_temporal']['p']) / 3.0
    arms['ensemble_mean'] = {'scores': p_mean, 'p': p_mean}

    # ---- 统一评估（测试半区）----
    y_test = labels[test_idx]
    for name, arm in arms.items():
        s = np.asarray(arm['scores'], dtype=float)[test_idx]
        p_test = np.asarray(arm['p'], dtype=float)[test_idx]
        arm['test_scores'] = s.tolist()
        arm['auroc'] = compute_auc(arm['test_scores'], y_test.tolist())
        arm['recall_at_budget'] = _recall_at_budget(s, y_test, budget)
        arm['brier'] = float(np.mean((p_test - y_test) ** 2))
        del arm['scores'], arm['p']

    # ---- 阶梯 ----
    a = {k: v['auroc'] for k, v in arms.items()}
    members = ('member_pi', 'member_typed', 'member_temporal')
    best_member = max(members, key=lambda k: a[k])
    alternatives = ('features_lgbm',) + members + ('ensemble_mean',
                                                   'seir_joint')
    best_alt = max(alternatives, key=lambda k: a[k])
    ladder = {
        'joint_minus_ensemble': a['joint'] - a['ensemble_mean'],
        'joint_minus_features_lgbm': a['joint'] - a['features_lgbm'],
        'joint_minus_seir_joint': a['joint'] - a['seir_joint'],
        'joint_res_minus_joint': a['joint_res'] - a['joint'],
        'ensemble_minus_best_member': a['ensemble_mean'] - a[best_member],
        'ensemble_minus_features_lgbm':
            a['ensemble_mean'] - a['features_lgbm'],
        'joint_minus_best_alternative': a['joint'] - a[best_alt],
        'best_alternative_arm': best_alt,
        'oracle_minus_joint': a['oracle_nu'] - a['joint'],
        'oracle_minus_features_lgbm':
            a['oracle_nu'] - a['features_lgbm'],
        'joint_minus_default': a['joint'] - a['seir_default'],
        'seir_joint_minus_default': a['seir_joint'] - a['seir_default'],
        'default_minus_features_lgbm':
            a['seir_default'] - a['features_lgbm'],
    }
    gap = a['oracle_nu'] - a['seir_default']
    ladder['joint_realization_ratio'] = (
        float(ladder['joint_minus_default'] / gap)
        if abs(gap) > 1e-9 else float('nan'))

    # ---- 参数恢复（尺度不变量：Λ 形状误差不受 β↔k 缩放影响）----
    seir = net['seir_info']
    lam_hat = np.asarray(r_joint['lam_w_hat'], dtype=float)
    lam_true = np.asarray(seir['lam_w_truth'], dtype=float)
    beta_t_hat = np.asarray(r_joint['params']['beta_t_hat'])
    beta_t_lit = np.asarray([BETA_BY_TYPE[t] for t in EDGE_TYPE_IDS])
    decay_hat = np.asarray(r_joint['params']['decay_hat'])
    decay_lit = np.asarray(DECAY_WEIGHTS, dtype=float)
    param_recovery = {
        'beta_rel_err': abs(r_joint['params']['beta_hat']
                            / seir['beta_truth'] - 1.0),
        'i0_rel_err': abs(r_joint['params']['I0_hat']
                          / seir['I0_truth'] - 1.0),
        'lam_w_shape_err': float(np.sqrt(np.mean(
            (lam_hat / lam_hat.mean() - lam_true / lam_true.mean()) ** 2))),
        # 可辨识量：消息中只出现乘积 Λ̂·d̂（分量间可交换漂移，
        # 逐分量恢复受乘积简并限制——产品形状才是数学契约）
        'force_shape_err': float(np.sqrt(np.mean(
            ((lam_hat * decay_hat) / (lam_hat * decay_hat).mean()
             - (lam_true * decay_lit) / (lam_true * decay_lit).mean())
            ** 2))),
        'type_drift_l2': float(np.sqrt(np.mean(
            np.log(beta_t_hat / beta_t_lit) ** 2))),
        'decay_drift_l2': float(np.sqrt(np.mean(
            np.log(decay_hat / decay_lit) ** 2))),
        'joint_loss_drop': r_joint['loss_first'] - r_joint['loss_last'],
    }

    def _pair(x, ykey):
        return delong_paired_test(y_test.tolist(),
                                  arms[x]['test_scores'],
                                  arms[ykey]['test_scores'])

    delong = {
        'joint_vs_ensemble': _pair('joint', 'ensemble_mean'),
        'joint_vs_features_lgbm': _pair('joint', 'features_lgbm'),
        'joint_vs_seir_joint': _pair('joint', 'seir_joint'),
        'joint_res_vs_joint': _pair('joint_res', 'joint'),
        'ensemble_vs_member_pi': _pair('ensemble_mean', 'member_pi'),
        'ensemble_vs_best_member': _pair('ensemble_mean', best_member),
        'joint_vs_oracle': _pair('joint', 'oracle_nu'),
    }

    return {
        'seed': seed,
        'n_contacts': n_contacts,
        'target_rate': target_rate,
        'budget': budget,
        'n_test': int(len(test_idx)),
        'test_labels': y_test.tolist(),
        'arms': arms,
        'ladder': ladder,
        'param_recovery': param_recovery,
        'delong': delong,
        'joint_params': r_joint['params'],
    }


# ==============================================================================
# 多种子汇总
# ==============================================================================

_NUMERIC_LADDER = (
    'joint_minus_ensemble', 'joint_minus_features_lgbm',
    'joint_minus_seir_joint', 'joint_res_minus_joint',
    'ensemble_minus_best_member', 'ensemble_minus_features_lgbm',
    'joint_minus_best_alternative', 'oracle_minus_joint',
    'oracle_minus_features_lgbm', 'joint_minus_default',
    'seir_joint_minus_default', 'default_minus_features_lgbm',
    'joint_realization_ratio')

_DELONG_MAP = {
    'joint_minus_ensemble': 'joint_vs_ensemble',
    'joint_minus_features_lgbm': 'joint_vs_features_lgbm',
    'joint_minus_seir_joint': 'joint_vs_seir_joint',
    'joint_res_minus_joint': 'joint_res_vs_joint',
    'ensemble_minus_best_member': 'ensemble_vs_best_member',
}


def run_multi_seed_joint_ablation(n_contacts=400, n_seeds=20,
                                  seed_start=101, target_rate=0.25,
                                  budget=0.25, joint_epochs=1200, lr=0.03,
                                  member_epochs=500, pi_epochs=800,
                                  prior_lambda=1e-3, n_bootstrap=2000):
    """多种子十臂消融：效应量 + 种子级 bootstrap CI + DeLong 汇总。"""
    seeds_reports = []
    for i in range(n_seeds):
        rep = run_joint_ablation_once(
            n_contacts=n_contacts, seed=seed_start + i,
            target_rate=target_rate, budget=budget,
            joint_epochs=joint_epochs, lr=lr, member_epochs=member_epochs,
            pi_epochs=pi_epochs, prior_lambda=prior_lambda)
        seeds_reports.append({
            'seed': rep['seed'],
            'auroc': {k: rep['arms'][k]['auroc'] for k in ARMS},
            'recall_at_budget': {k: rep['arms'][k]['recall_at_budget']
                                 for k in ARMS},
            'brier': {k: rep['arms'][k]['brier'] for k in ARMS},
            'ladder': rep['ladder'],
            'param_recovery': rep['param_recovery'],
            'delong_p': {k: v['p_value'] for k, v in rep['delong'].items()},
            'positive_rate': float(np.mean(rep['test_labels'])),
            'best_alternative_arm': rep['ladder']['best_alternative_arm'],
        })

    rng = np.random.RandomState(seed_start)

    def _boot_ci(values):
        values = np.asarray(values, dtype=float)
        means = [values[rng.randint(0, len(values), len(values))].mean()
                 for _ in range(n_bootstrap)]
        return [float(np.percentile(means, 2.5)),
                float(np.percentile(means, 97.5))]

    arm_summary = {}
    for k in ARMS:
        entry = {}
        for metric in ('auroc', 'recall_at_budget', 'brier'):
            vals = [s[metric][k] for s in seeds_reports]
            entry['mean_' + metric] = float(np.mean(vals))
            entry['bootstrap_ci_' + metric] = _boot_ci(vals)
        arm_summary[k] = entry

    ladder_summary = {}
    for k in _NUMERIC_LADDER:
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
            p_vals = [s['delong_p'][pair] for s in seeds_reports]
            entry['delong_significant_frac'] = float(np.mean(
                [p < 0.05 for p in p_vals]))
        ladder_summary[k] = entry
    ladder_summary['best_alternative_arm_counts'] = {
        arm: int(sum(1 for s in seeds_reports
                     if s['best_alternative_arm'] == arm))
        for arm in set(s['best_alternative_arm'] for s in seeds_reports)}

    recovery_summary = {}
    for k in seeds_reports[0]['param_recovery']:
        vals = [s['param_recovery'][k] for s in seeds_reports]
        recovery_summary[k] = {'mean': float(np.mean(vals)),
                               'bootstrap_ci': _boot_ci(vals)}

    # ---- P5 结论 ----
    je = ladder_summary['joint_minus_ensemble']
    jf = ladder_summary['joint_minus_features_lgbm']
    jb = ladder_summary['joint_minus_best_alternative']
    em = ladder_summary['ensemble_minus_best_member']
    joint_beats_ensemble = bool(je['mean'] > 0 and je['bootstrap_ci'][0] > 0)
    joint_beats_features = bool(jf['mean'] > 0 and jf['bootstrap_ci'][0] > 0)
    breaks_ceiling = bool(jb['mean'] > 0 and jb['bootstrap_ci'][0] > 0)
    if em['bootstrap_ci'][0] > 0:
        ensemble_verdict = 'ensemble_beats_best_member（集成反超）'
    elif em['bootstrap_ci'][1] < 0:
        ensemble_verdict = 'ensemble_below_best_member（集成劣于最优成员）'
    else:
        ensemble_verdict = 'ensemble_ties_best_member（只追平——用户前提成立）'
    conclusion = {
        'joint_beats_ensemble': joint_beats_ensemble,
        'joint_beats_features_lgbm': joint_beats_features,
        'joint_breaks_ceiling_vs_all_alternatives': breaks_ceiling,
        'ensemble_verdict': ensemble_verdict,
        'joint_realization_ratio': ladder_summary['joint_realization_ratio'],
        'param_recovery': {
            'beta_rel_err': recovery_summary['beta_rel_err']['mean'],
            'i0_rel_err': recovery_summary['i0_rel_err']['mean'],
            'lam_w_shape_err': recovery_summary['lam_w_shape_err']['mean'],
        },
    }

    return {
        'design': {
            'name': 'joint_mech_ablation_v1',
            'n_contacts': n_contacts, 'n_seeds': n_seeds,
            'seed_start': seed_start, 'target_rate': target_rate,
            'budget': budget,
            'joint': 'JointMechGNN：Λ̂(RK4,δβ/δI0)×β̂_t×d̂_w×ŝ 逐边'
                     '消息 + host·corr 固定物理 + k 交替校准 '
                     '（12 机制参数 + prior_lambda=%.0e 先验收缩）'
                     % prior_lambda,
            'arms': list(ARMS),
            'dgp': 'combined_mechanism_network_v1（类型×时序×PI 三机制'
                   '共存；类型/衰减真值=文献值，SEIR 真值抖动 ±20%/±50%'
                   ' + 窗口化 Λ vs 平坦 Λ_ref 失配）',
            'ensemble': '三成员概率等权平均（事后加权代表）',
            'split': 'stratified 50/50 half-split（接触者空间）',
            'metrics': 'AUROC + recall@budget + Brier',
        },
        'seeds': seeds_reports,
        'arm_summary': arm_summary,
        'ladder_summary': ladder_summary,
        'param_recovery_summary': recovery_summary,
        'conclusion': conclusion,
    }
