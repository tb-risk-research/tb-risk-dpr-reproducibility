#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端联合 GNN（joint_mech_gnn）与十臂消融协议测试（P5）。

锁定契约：
  1. 消息传递闭式一致：初始参数下 ν̂ = host·corr·Σ Λ_default(w)·
     β_t·decay_w·s（逐边乘积结构的数学正确性）；
  2. 孤立接触者 ν̂=0 → p≈0（物理正确：无暴露无感染）；
  3. 确定性 + 损失下降；
  4. 成员适配器边记账（typed/temporal 从 membership 重建的边数守恒）；
  5. 十臂消融冒烟（小 n、小预算：臂齐全、AUROC ∈ [0,1]、阶梯键齐全）。
"""
import unittest

import numpy as np

from tb_risk.validation.combined_network import build_combined_network
from tb_risk.validation.joint_ablation import (
    to_temporal_member_graph,
    to_typed_member_graph,
)
from tb_risk.validation.pi_network import node_features
from tb_risk.validation.temporal_network import DECAY_WEIGHTS
from tb_risk.validation.typed_network import BETA_BY_TYPE, EDGE_TYPE_IDS

try:
    import torch  # noqa: F401
    from tb_risk.ml.gnn.joint_mech_gnn import (
        JointMechGNN,
        membership_tensors,
        train_joint_mech_gnn,
    )
    TORCH = True
except ImportError:  # pragma: no cover
    TORCH = False


class TestJointMechGNN(unittest.TestCase):

    @unittest.skipUnless(TORCH, 'PyTorch 不可用')
    def test_message_passing_closed_form(self):
        """初始参数下 ν̂ 与闭式逐边乘积一致（P4a 默认参数窗口化 Λ）。"""
        net = build_combined_network(n_contacts=60, random_state=7)
        g = membership_tensors(net)
        model = JointMechGNN(
            np.asarray([BETA_BY_TYPE[t] for t in EDGE_TYPE_IDS]),
            np.asarray(DECAY_WEIGHTS, dtype=float))
        with torch.no_grad():
            _, nu_hat, _ = model(
                g['mem_contact'], g['mem_src'], g['mem_type'],
                g['mem_win'], g['log_infectivity'], g['host'],
                g['corr'], x_nodes=g['x_nodes'], k=1.0)
        # 闭式：host·corr·Σ Λ_default(w)·β_t·decay_w·exp(log_s_src)
        lam_w = np.asarray(net['seir_info']['lam_w_default'], dtype=float)
        X = node_features(net)
        M = net['M']
        nu_ref = np.zeros(len(net['labels']))
        for i, mems in enumerate(net['memberships']):
            acc = 0.0
            for (src, t, w, _inten, _s) in mems:
                acc += (lam_w[w] * BETA_BY_TYPE[t] * DECAY_WEIGHTS[w]
                        * float(np.exp(X[int(src), 1])))
            nu_ref[i] = (net['nodes'][M + i]['host_multiplier']
                         * net['corr'][i] * acc)
        np.testing.assert_allclose(nu_hat.numpy(), nu_ref, rtol=1e-9,
                                   atol=1e-15)

    @unittest.skipUnless(TORCH, 'PyTorch 不可用')
    def test_isolated_contact_zero_risk(self):
        """无 membership 接触者：ν̂=0、p≈0。"""
        net = build_combined_network(n_contacts=60, random_state=7)
        g = membership_tensors(net)
        model = JointMechGNN(
            np.asarray([BETA_BY_TYPE[t] for t in EDGE_TYPE_IDS]),
            np.asarray(DECAY_WEIGHTS, dtype=float))
        with torch.no_grad():
            logit, nu, _ = model(
                g['mem_contact'], g['mem_src'], g['mem_type'],
                g['mem_win'], g['log_infectivity'], g['host'],
                g['corr'], x_nodes=g['x_nodes'], k=1.0)
        isolated = [i for i, mems in enumerate(net['memberships'])
                    if not mems]
        self.assertTrue(len(isolated) > 0, '测试网应有孤立接触者')
        np.testing.assert_array_equal(nu.numpy()[isolated], 0.0)
        self.assertTrue(np.all(logit.numpy()[isolated] < -15.0))

    @unittest.skipUnless(TORCH, 'PyTorch 不可用')
    def test_training_deterministic_and_loss_drops(self):
        """同种子两次训练逐位一致；损失下降（可微链路通）。"""
        from tb_risk.validation.joint_ablation import _stratified_split
        net = build_combined_network(n_contacts=60, random_state=7)
        labels = np.asarray(net['labels'])
        train_idx, _ = _stratified_split(labels,
                                         np.random.RandomState(0))
        r1 = train_joint_mech_gnn(net, labels, train_idx, epochs=150,
                                  seed=3, k_refit_every=50)
        r2 = train_joint_mech_gnn(net, labels, train_idx, epochs=150,
                                  seed=3, k_refit_every=50)
        np.testing.assert_array_equal(r1['score'], r2['score'])
        self.assertLess(r1['loss_last'], r1['loss_first'])
        self.assertGreater(r1['loss_first'], 0.1)   # 非退化起点

    @unittest.skipUnless(TORCH, 'PyTorch 不可用')
    def test_residual_variant_runs(self):
        """joint_res 变体可训练且输出有限。"""
        net = build_combined_network(n_contacts=60, random_state=7)
        labels = np.asarray(net['labels'])
        r = train_joint_mech_gnn(net, labels, np.arange(30), epochs=60,
                                 residual=True, seed=3)
        self.assertTrue(r['residual'])
        self.assertTrue(np.all(np.isfinite(r['score'])))
        self.assertTrue(np.all((r['p'] >= 0) & (r['p'] <= 1)))


class TestMemberAdapters(unittest.TestCase):

    def test_typed_adapter_edge_accounting(self):
        """typed 适配器：membership 边全数入图（双向）。"""
        net = build_combined_network(n_contacts=60, random_state=7)
        g = to_typed_member_graph(net)
        n_mem = sum(len(m) for m in net['memberships'])
        # 有向计数：membership 边贡献 2×n_mem；同簇互连边另计
        total = sum(e.shape[1] for e in g['edge_indices'])
        self.assertGreaterEqual(total, 2 * n_mem)
        self.assertEqual(len(g['edge_indices']), len(EDGE_TYPE_IDS))
        self.assertEqual(g['x'].shape[0], len(net['nodes']))
        self.assertEqual(g['x'].shape[1], 14)

    def test_temporal_adapter_windows(self):
        """temporal 适配器：5 窗、membership 边全数入图、每边恰一窗。"""
        net = build_combined_network(n_contacts=60, random_state=7)
        g = to_temporal_member_graph(net)
        n_mem = sum(len(m) for m in net['memberships'])
        self.assertEqual(len(g['windows']), 5)
        self.assertEqual(g['window_ids'], [4, 3, 2, 1, 0])  # 最旧→最新
        total = sum(w['edge_index'].shape[1] for w in g['windows'])
        self.assertEqual(total, 2 * n_mem)
        self.assertEqual(g['x'].shape[1], 14)


class TestAblationSmoke(unittest.TestCase):

    @unittest.skipUnless(TORCH, 'PyTorch 不可用')
    def test_run_once_smoke(self):
        """十臂消融冒烟（小 n、小预算）：臂齐全、指标合法。"""
        from tb_risk.validation.joint_ablation import (
            ARMS, run_joint_ablation_once)
        rep = run_joint_ablation_once(n_contacts=60, seed=42,
                                      joint_epochs=100,
                                      member_epochs=30, pi_epochs=60)
        self.assertEqual(set(rep['arms'].keys()), set(ARMS))
        for arm in ARMS:
            auc = rep['arms'][arm]['auroc']
            self.assertTrue(0.0 <= auc <= 1.0, arm)
        for key in ('joint_minus_ensemble', 'joint_minus_features_lgbm',
                    'joint_minus_best_alternative', 'oracle_minus_joint'):
            self.assertIn(key, rep['ladder'])
        self.assertIn('beta_rel_err', rep['param_recovery'])


if __name__ == '__main__':
    unittest.main()
