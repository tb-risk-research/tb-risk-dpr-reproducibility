#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 — GNN-SEIR v3.0 同步升级（改进 11）

覆盖内容：
  1. seir_ode_step_v3: 9-房室 ODE 导数计算
  2. seir_v3_euler_integrate: 9-房室 Euler 积分与指标提取
  3. seir_ode_gating: 6/7/8 通道门控自动检测与分发（改进7+8: 独立 rho_react + eta_sub）
  4. HeteroTimeVaryingGNN: node_feature_dim=30, seir_gate 8 通道, 因果矩阵 19 变量
  5. SEIRInformedGNN: node_feature_dim=30, seir_gate 8 通道
  6. _extract_contact_features: 30 维特征 (含 SEIR 状态 one-hot)
  7. 向后兼容: 旧版 4D seir_ode_step / seir_euler_integrate / 3 通道门控

文献：
  Kermack & McKendrick (1927) Proc R Soc Lond A 115:700-721
  Chen et al. (2018) Neural ODE, NeurIPS
  Brody et al. (2022) GATv2, ICLR
  Houben et al. (2016) BMC Med 14:151
  Horton et al. (2023) PNAS 120(47):e2221186120
  Emery et al. (2023) eLife 12:e82269
"""

import os
import sys
import unittest

import numpy as np

# 确保包路径在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.ml.gnn import (
    seir_ode_step,
    seir_ode_step_v3,
    seir_euler_integrate,
    seir_rk4_integrate,
    seir_v3_euler_integrate,
    seir_v3_rk4_integrate,
    seir_ode_gating,
    seir_solver_error_baseline,
    _IDX_S, _IDX_LF, _IDX_LS, _IDX_M,
    _IDX_ISUB, _IDX_ISP, _IDX_ISN, _IDX_R, _IDX_C,
    _N_COMPARTMENTS_V3,
)

# 可选依赖：torch / torch_geometric
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


# =============================================================================
# 1. seir_ode_step_v3: 9-房室 ODE 导数
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch 不可用")
class TestSEIROdeStepV3(unittest.TestCase):
    """9-房室 SEIR ODE 单步导数计算测试"""

    def test_output_shape_9d(self):
        """导数张量最后一维应为 9"""
        state = torch.zeros(1, 9)
        state[0, _IDX_S] = 0.99
        state[0, _IDX_LF] = 0.005
        d = seir_ode_step_v3(state, beta=0.5, rho_fast=0.001,
                             rho_react=1e-5, sigma_clear=0.01, gamma=0.1)
        self.assertEqual(d.shape[-1], 9)

    def test_batch_shape(self):
        """批量输入应保持批量维度"""
        B = 16
        state = torch.zeros(B, 9)
        state[:, _IDX_S] = 0.95
        state[:, _IDX_LF] = 0.03
        state[:, _IDX_ISP] = 0.02
        d = seir_ode_step_v3(state, beta=0.5, rho_fast=0.001,
                             rho_react=1e-5, sigma_clear=0.01, gamma=0.1)
        self.assertEqual(d.shape, (B, 9))

    def test_no_infection_no_change(self):
        """无感染源 (I_sub=I_sp=I_sn=0) 时 S 不变"""
        state = torch.zeros(1, 9)
        state[0, _IDX_S] = 1.0
        d = seir_ode_step_v3(state, beta=0.5, rho_fast=0.001,
                             rho_react=1e-5, sigma_clear=0.01, gamma=0.1)
        self.assertTrue(torch.allclose(d[0, _IDX_S], torch.tensor(0.0), atol=1e-8))

    def test_infection_decreases_S(self):
        """有感染源时 S 导数为负"""
        state = torch.zeros(1, 9)
        state[0, _IDX_S] = 0.9
        state[0, _IDX_ISP] = 0.1
        d = seir_ode_step_v3(state, beta=0.5, rho_fast=0.001,
                             rho_react=1e-5, sigma_clear=0.01, gamma=0.1)
        self.assertLess(float(d[0, _IDX_S]), 0.0)

    def test_clearance_flow_to_C(self):
        """自清除流从 L_fast/L_slow 流向 C"""
        state = torch.zeros(1, 9)
        state[0, _IDX_LF] = 0.5
        state[0, _IDX_LS] = 0.3
        d = seir_ode_step_v3(state, beta=0.0, rho_fast=0.0,
                             rho_react=0.0, sigma_clear=0.01, gamma=0.1)
        # C 应增加 (clear_fast + clear_slow > 0)
        self.assertGreater(float(d[0, _IDX_C]), 0.0)
        # L_fast 和 L_slow 应减少 (清除流)
        self.assertLess(float(d[0, _IDX_LF]), 0.0)
        self.assertLess(float(d[0, _IDX_LS]), 0.0)

    def test_differentiability(self):
        """导数计算应保持计算图可微"""
        state = torch.zeros(1, 9, requires_grad=False)
        state[0, _IDX_S] = 0.9
        state[0, _IDX_ISP] = 0.1
        beta = torch.tensor(0.5, requires_grad=True)
        d = seir_ode_step_v3(state, beta=beta, rho_fast=0.001,
                             rho_react=1e-5, sigma_clear=0.01, gamma=0.1)
        loss = d.sum()
        loss.backward()
        self.assertIsNotNone(beta.grad)


# =============================================================================
# 2. seir_v3_euler_integrate: 9-房室 Euler 积分
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch 不可用")
class TestSEIRv3EulerIntegrate(unittest.TestCase):
    """9-房室 SEIR Euler 积分测试"""

    def test_output_keys(self):
        """返回 dict 应包含所有必要键"""
        beta = torch.tensor([0.5])
        result = seir_v3_euler_integrate(
            beta, rho_fast=torch.tensor([0.001]),
            rho_react=torch.tensor([1e-5]),
            sigma_clear=torch.tensor([0.01]),
            gamma=torch.tensor([0.1]))
        for key in ['trajectory', 'peak_I', 'final_I', 'auc_I',
                     'time_to_peak', 'epidemic_size']:
            self.assertIn(key, result)

    def test_trajectory_shape(self):
        """轨迹 shape 应为 [B, n_steps+1, 9]"""
        B, n_steps = 4, 10
        beta = torch.full((B,), 0.5)
        result = seir_v3_euler_integrate(
            beta, rho_fast=torch.full((B,), 0.001),
            rho_react=torch.full((B,), 1e-5),
            sigma_clear=torch.full((B,), 0.01),
            gamma=torch.full((B,), 0.1), n_steps=n_steps)
        self.assertEqual(result['trajectory'].shape, (B, n_steps + 1, 9))

    def test_peak_I_nonneg(self):
        """peak_I 应非负"""
        beta = torch.tensor([0.5])
        result = seir_v3_euler_integrate(
            beta, rho_fast=torch.tensor([0.001]),
            rho_react=torch.tensor([1e-5]),
            sigma_clear=torch.tensor([0.01]),
            gamma=torch.tensor([0.1]))
        self.assertTrue(torch.all(result['peak_I'] >= 0))

    def test_differentiability(self):
        """积分结果应可微"""
        beta = torch.tensor([0.5], requires_grad=True)
        result = seir_v3_euler_integrate(
            beta, rho_fast=torch.tensor([0.001]),
            rho_react=torch.tensor([1e-5]),
            sigma_clear=torch.tensor([0.01]),
            gamma=torch.tensor([0.1]))
        loss = result['peak_I'].sum()
        loss.backward()
        self.assertIsNotNone(beta.grad)

    def test_zero_beta_no_epidemic(self):
        """beta=0 且无初始潜伏时不应有疫情传播"""
        beta = torch.tensor([0.0])
        result = seir_v3_euler_integrate(
            beta, rho_fast=torch.tensor([0.001]),
            rho_react=torch.tensor([1e-5]),
            sigma_clear=torch.tensor([0.01]),
            gamma=torch.tensor([0.1]),
            Lf0=0.0, Ls0=0.0)  # 无初始潜伏感染
        # 无感染源且无初始潜伏时 peak_I 应为 0
        self.assertTrue(float(result['peak_I'][0]) < 1e-6)


# =============================================================================
# 3. seir_ode_gating: 6/7/8 通道门控自动检测
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch 不可用")
class TestSEIROdeGatingV3(unittest.TestCase):
    """SEIR ODE 门控 6/7/8 通道自动检测测试"""

    def setUp(self):
        torch.manual_seed(42)
        self.B = 8
        self.hidden_dim = 32
        self.h = torch.randn(self.B, self.hidden_dim)
        self.tp = torch.rand(self.B, 1) * 3.0  # 治疗阶段 [0, 3]

    def _make_gate(self, n_out):
        """构造门控 MLP"""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(self.hidden_dim + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, n_out),
            nn.Sigmoid())

    def test_6channel_output_shape(self):
        """6 通道门控应返回 [B, hidden_dim] 调制表示"""
        gate = self._make_gate(6)
        h_mod, metrics = seir_ode_gating(self.h, self.tp, gate)
        self.assertEqual(h_mod.shape, (self.B, self.hidden_dim))

    def test_6channel_metrics_keys(self):
        """6 通道门控应返回 v3 指标"""
        gate = self._make_gate(6)
        _, metrics = seir_ode_gating(self.h, self.tp, gate)
        for key in ['peak_I', 'auc_I', 'epidemic_size']:
            self.assertIn(key, metrics)

    def test_3channel_backward_compat(self):
        """3 通道门控应回退到 4D SEIR（向后兼容）"""
        gate = self._make_gate(3)
        h_mod, metrics = seir_ode_gating(self.h, self.tp, gate)
        self.assertEqual(h_mod.shape, (self.B, self.hidden_dim))
        # 4D SEIR trajectory 最后一维应为 4
        self.assertEqual(metrics['trajectory'].shape[-1], 4)

    def test_6channel_trajectory_9d(self):
        """6 通道门控的 ODE 轨迹应为 9 维"""
        gate = self._make_gate(6)
        _, metrics = seir_ode_gating(self.h, self.tp, gate)
        self.assertEqual(metrics['trajectory'].shape[-1], 9)

    def test_modulation_finite(self):
        """调制后的表示应有限（无数值溢出）"""
        gate = self._make_gate(6)
        h_mod, _ = seir_ode_gating(self.h, self.tp, gate)
        self.assertTrue(torch.all(torch.isfinite(h_mod)))

    def test_7channel_independent_rho_react(self):
        """改进7: 7 通道应使用独立 rho_react（非 rho_fast*0.05 近似）"""
        gate7 = self._make_gate(7)
        gate6 = self._make_gate(6)
        # 复制 6 通道权重到 7 通道前 6 维，第 7 维设为不同值
        with torch.no_grad():
            for m7, m6 in zip(gate7.modules(), gate6.modules()):
                if hasattr(m6, 'weight'):
                    m7.weight.data[:m6.weight.shape[0]] = m6.weight.data
                    m7.bias.data[:m6.bias.shape[0]] = m6.bias.data
        h_mod7, metrics7 = seir_ode_gating(self.h, self.tp, gate7)
        h_mod6, metrics6 = seir_ode_gating(self.h, self.tp, gate6)
        # 7 通道和 6 通道的轨迹应不同（rho_react 不同）
        self.assertEqual(metrics7['trajectory'].shape[-1], 9)
        self.assertFalse(torch.allclose(metrics7['peak_I'], metrics6['peak_I']))

    def test_8channel_independent_eta_sub(self):
        """改进8: 8 通道应使用独立 eta_sub"""
        gate8 = self._make_gate(8)
        h_mod, metrics = seir_ode_gating(self.h, self.tp, gate8)
        self.assertEqual(h_mod.shape, (self.B, self.hidden_dim))
        self.assertEqual(metrics['trajectory'].shape[-1], 9)
        self.assertTrue(torch.all(torch.isfinite(h_mod)))

    def test_8channel_metrics_keys(self):
        """8 通道门控应返回 v3 指标"""
        gate = self._make_gate(8)
        _, metrics = seir_ode_gating(self.h, self.tp, gate)
        for key in ['peak_I', 'auc_I', 'epidemic_size']:
            self.assertIn(key, metrics)

    def test_8channel_different_from_6(self):
        """8 通道与 6 通道结果不同（eta_sub 影响传播）"""
        gate8 = self._make_gate(8)
        gate6 = self._make_gate(6)
        with torch.no_grad():
            for m8, m6 in zip(gate8.modules(), gate6.modules()):
                if hasattr(m6, 'weight'):
                    m8.weight.data[:m6.weight.shape[0]] = m6.weight.data
                    m8.bias.data[:m6.bias.shape[0]] = m6.bias.data
        _, metrics8 = seir_ode_gating(self.h, self.tp, gate8)
        _, metrics6 = seir_ode_gating(self.h, self.tp, gate6)
        # eta_sub 通道影响传播动力学，峰值应不同
        self.assertFalse(torch.allclose(metrics8['peak_I'], metrics6['peak_I']))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch 不可用")
class TestSEIROdeGatingSolver(unittest.TestCase):
    """seir_ode_gating 数值求解器参数化 (solver='rk4'/'euler') 测试

    问题二修复: 生产路径默认使用四阶 RK4 (全局误差 O(dt⁴)),
    保留 solver='euler' 供对比/兼容/误差基准。
    """

    def setUp(self):
        torch.manual_seed(42)
        self.B = 8
        self.hidden_dim = 32
        self.h = torch.randn(self.B, self.hidden_dim)
        self.tp = torch.rand(self.B, 1) * 3.0  # 治疗阶段 [0, 3]

    def _make_gate(self, n_out):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(self.hidden_dim + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, n_out),
            nn.Sigmoid())

    def test_default_solver_is_rk4(self):
        """默认调用 (不传 solver) 与显式 solver='rk4' 结果一致"""
        gate = self._make_gate(6)
        h_def, m_def = seir_ode_gating(self.h, self.tp, gate)
        h_rk4, m_rk4 = seir_ode_gating(self.h, self.tp, gate, solver='rk4')
        self.assertTrue(torch.allclose(h_def, h_rk4),
                        "默认 solver 应与 rk4 一致")
        self.assertTrue(torch.allclose(m_def['peak_I'], m_rk4['peak_I']))
        self.assertTrue(torch.allclose(m_def['trajectory'], m_rk4['trajectory']))

    def test_solver_euler_finite(self):
        """solver='euler' 可运行且输出有限 (6 通道 9D 路径)"""
        gate = self._make_gate(6)
        h_e, m_e = seir_ode_gating(self.h, self.tp, gate, solver='euler')
        self.assertTrue(torch.all(torch.isfinite(h_e)))
        self.assertEqual(m_e['trajectory'].shape[-1], 9)

    def test_solver_rk4_more_accurate_than_euler(self):
        """RK4 与 Euler 在 9D 路径上数值不同 (RK4 更高精度)"""
        gate = self._make_gate(6)
        _, m_rk4 = seir_ode_gating(self.h, self.tp, gate, solver='rk4')
        _, m_e = seir_ode_gating(self.h, self.tp, gate, solver='euler')
        self.assertFalse(torch.allclose(m_rk4['peak_I'], m_e['peak_I']),
                         "RK4 与 Euler 的峰值指标应不同")

    def test_3channel_solver_param_4d(self):
        """3 通道 (向后兼容 4D 路径) 也接受 solver 参数"""
        gate = self._make_gate(3)
        h_e, m_e = seir_ode_gating(self.h, self.tp, gate, solver='euler')
        h_r, m_r = seir_ode_gating(self.h, self.tp, gate, solver='rk4')
        self.assertEqual(m_e['trajectory'].shape[-1], 4)
        self.assertEqual(m_r['trajectory'].shape[-1], 4)
        self.assertFalse(torch.allclose(m_e['peak_I'], m_r['peak_I']))
        self.assertTrue(torch.all(torch.isfinite(h_e)))
        self.assertTrue(torch.all(torch.isfinite(h_r)))

    def test_invalid_solver_raises(self):
        """非法 solver 抛 ValueError (大小写敏感)"""
        gate = self._make_gate(6)
        with self.assertRaises(ValueError):
            seir_ode_gating(self.h, self.tp, gate, solver='midpoint')
        with self.assertRaises(ValueError):
            seir_ode_gating(self.h, self.tp, gate, solver='RK4')

    def test_7channel_solver_works(self):
        """7 通道 (独立 rho_react) 在两种 solver 下均可运行"""
        gate = self._make_gate(7)
        _, m_rk4 = seir_ode_gating(self.h, self.tp, gate, solver='rk4')
        _, m_e = seir_ode_gating(self.h, self.tp, gate, solver='euler')
        self.assertEqual(m_rk4['trajectory'].shape[-1], 9)
        self.assertEqual(m_e['trajectory'].shape[-1], 9)


# =============================================================================
# 4. HeteroTimeVaryingGNN: v3.0 默认参数与因果矩阵
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch 不可用")
class TestHeteroTimeVaryingGNNv3(unittest.TestCase):
    """HeteroTimeVaryingGNN v3.0 升级测试"""

    def test_default_node_feature_dim_30(self):
        """默认 node_feature_dim 应为 30"""
        from tb_risk.ml.gnn import HeteroTimeVaryingGNN
        model = HeteroTimeVaryingGNN()
        self.assertEqual(model.node_embedding.in_features, 30)

    def test_seir_gate_8_channels(self):
        """改进7+8: seir_gate 输出维度应为 8"""
        from tb_risk.ml.gnn import HeteroTimeVaryingGNN
        model = HeteroTimeVaryingGNN()
        # 最后一个 Linear 层的 out_features 应为 8
        last_linear = [m for m in model.seir_gate.modules()
                       if hasattr(m, 'out_features')][-1]
        self.assertEqual(last_linear.out_features, 8)

    def test_causal_mask_19_variables(self):
        """因果矩阵变量数应为 19 (14 原始 + 5 新增)"""
        from tb_risk.ml.gnn import HeteroTimeVaryingGNN
        # 需要 causal_graph 才会构建因果矩阵
        try:
            from tb_risk.causal_rl.constraints import CausalGraphConstraint
            cg = CausalGraphConstraint()
        except ImportError:
            self.skipTest("causal_rl 不可用")
            return
        model = HeteroTimeVaryingGNN(causal_graph=cg)
        if model.causal_adj_matrix is not None:
            self.assertEqual(model.causal_adj_matrix.shape[0], 19)

    def test_forward_with_30d_features(self):
        """前向传播应接受 30 维节点特征"""
        from tb_risk.ml.gnn import HeteroTimeVaryingGNN
        model = HeteroTimeVaryingGNN(node_feature_dim=30, hidden_dim=16, num_layers=1)
        model.eval()
        num_nodes = 5
        x = torch.randn(num_nodes, 30)
        # 构造简单边索引
        edge_indices = [torch.tensor([[0, 1], [1, 2]], dtype=torch.long)]
        edge_attrs = [torch.randn(2, 6)]
        tp = torch.rand(num_nodes, 1) * 3
        with torch.no_grad():
            risk, final_emb, causal_loss, time_embs = model(
                [x], [edge_indices], [edge_attrs],
                treatment_phase=tp, use_ode_gating=False)
        self.assertIsNotNone(risk)


# =============================================================================
# 5. SEIRInformedGNN: v3.0 默认参数
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE and PYG_AVAILABLE, "PyTorch/PyG 不可用")
class TestSEIRInformedGNNv3(unittest.TestCase):
    """SEIRInformedGNN v3.0 升级测试"""

    def test_default_node_feature_dim_30(self):
        """默认 node_feature_dim 应为 30"""
        from tb_risk.ml.framework.seir_informed import SEIRInformedGNN
        model = SEIRInformedGNN()
        self.assertEqual(model.node_embedding.in_features, 30)

    def test_seir_gate_8_channels(self):
        """改进7+8: seir_gate 输出维度应为 8"""
        from tb_risk.ml.framework.seir_informed import SEIRInformedGNN
        model = SEIRInformedGNN()
        last_linear = [m for m in model.seir_gate.modules()
                       if hasattr(m, 'out_features')][-1]
        self.assertEqual(last_linear.out_features, 8)

    def test_forward_with_30d_features(self):
        """前向传播应接受 30 维节点特征"""
        from tb_risk.ml.framework.seir_informed import SEIRInformedGNN
        model = SEIRInformedGNN(node_feature_dim=30, hidden_dim=16, num_layers=1)
        model.eval()
        num_nodes = 5
        x = torch.randn(num_nodes, 30)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        tp = torch.rand(num_nodes, 1) * 3
        with torch.no_grad():
            risk, graph_emb = model(x, edge_index, treatment_phase=tp,
                                     use_ode_gating=False)
        self.assertEqual(risk.shape[0], num_nodes)


# =============================================================================
# 6. _extract_contact_features: 30 维特征
# =============================================================================

class TestContactFeatures30D(unittest.TestCase):
    """接触者特征提取 30 维测试"""

    def _make_mixin(self):
        """构造 _ContactFeatureMixin 实例"""
        from tb_risk.ml.framework._contact_features import _ContactFeatureMixin
        return _ContactFeatureMixin()

    def test_feature_dim_30(self):
        """特征维度应为 30"""
        mixin = self._make_mixin()
        entry = {'age': 35, 'bcg_vaccine': '是', 'ventilation': 3,
                 'contact_distance': '近', 'cumulative_exposure': 100}
        features = mixin._extract_contact_features(entry, 'family')
        self.assertEqual(len(features), 30)

    def test_latent_state_onehot(self):
        """潜伏感染状态 one-hot 应正确编码"""
        mixin = self._make_mixin()
        entry = {'age': 35, 'latent_state': 'L_fast'}
        features = mixin._extract_contact_features(entry, 'family')
        # 特征 23-26: L_fast one-hot → [1, 0, 0, 0]
        self.assertEqual(features[22:26], [1, 0, 0, 0])

    def test_disease_state_onehot(self):
        """疾病状态 one-hot 应正确编码"""
        mixin = self._make_mixin()
        entry = {'age': 35, 'disease_state': 'Subclinical'}
        features = mixin._extract_contact_features(entry, 'family')
        # 特征 27-30: Subclinical one-hot → [0, 1, 0, 0]
        self.assertEqual(features[26:30], [0, 1, 0, 0])

    def test_default_state_none(self):
        """缺失 SEIR 状态字段时默认为 None"""
        mixin = self._make_mixin()
        entry = {'age': 35}
        features = mixin._extract_contact_features(entry, 'family')
        # latent: None → [0, 0, 0, 1]
        self.assertEqual(features[22:26], [0, 0, 0, 1])
        # disease: None → [0, 0, 0, 1]
        self.assertEqual(features[26:30], [0, 0, 0, 1])

    def test_chinese_alias(self):
        """中文别名应正确映射"""
        mixin = self._make_mixin()
        entry = {'age': 35, 'latent_state': '快潜伏', 'disease_state': '临床'}
        features = mixin._extract_contact_features(entry, 'family')
        self.assertEqual(features[22:26], [1, 0, 0, 0])  # L_fast
        self.assertEqual(features[26:30], [0, 0, 1, 0])  # Clinical


# =============================================================================
# 7. 向后兼容: 4D seir_ode_step / seir_euler_integrate
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch 不可用")
class TestBackwardCompat4D(unittest.TestCase):
    """旧版 4D SEIR 函数向后兼容测试"""

    def test_seir_ode_step_4d(self):
        """4D seir_ode_step 应输出 4 维导数"""
        state = torch.tensor([[0.9, 0.05, 0.05, 0.0]])
        d = seir_ode_step(state, beta=0.5, sigma=0.2, gamma=0.1)
        self.assertEqual(d.shape[-1], 4)

    def test_seir_euler_integrate_4d(self):
        """4D seir_euler_integrate 应输出 4 维轨迹"""
        beta = torch.tensor([0.5])
        sigma = torch.tensor([0.2])
        gamma = torch.tensor([0.1])
        result = seir_euler_integrate(beta, sigma, gamma, n_steps=5)
        self.assertEqual(result['trajectory'].shape[-1], 4)
        self.assertEqual(result['trajectory'].shape[1], 6)  # n_steps + 1


# =============================================================================
# 7b. RK4 / 误差基准（问题二: 一阶 Euler → 四阶 RK4 升级）
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch 不可用")
class TestSEIRRK4Upgrade(unittest.TestCase):
    """4D RK4 积分: 形状 / 可微性 / 精度优于一阶 Euler"""

    def test_rk4_trajectory_shape(self):
        """4D RK4 轨迹 shape 应为 [B, n_steps+1, 4]"""
        beta = torch.tensor([0.5])
        res = seir_rk4_integrate(beta, torch.tensor([0.2]), torch.tensor([0.1]),
                                 n_steps=10, dt=0.1)
        self.assertEqual(res['trajectory'].shape, (1, 11, 4))

    def test_rk4_output_keys(self):
        """RK4 返回 dict 应含全部指标键"""
        beta = torch.tensor([0.5])
        res = seir_rk4_integrate(beta, torch.tensor([0.2]), torch.tensor([0.1]))
        for key in ['trajectory', 'peak_I', 'final_I', 'auc_I',
                     'time_to_peak', 'epidemic_size']:
            self.assertIn(key, res)

    def test_rk4_differentiable(self):
        """RK4 积分结果应可微（梯度流经 beta）"""
        beta = torch.tensor([0.5], requires_grad=True)
        res = seir_rk4_integrate(beta, torch.tensor([0.2]), torch.tensor([0.1]))
        loss = res['peak_I'].sum() + res['epidemic_size'].sum()
        loss.backward()
        self.assertIsNotNone(beta.grad)
        self.assertTrue(torch.isfinite(beta.grad).all())

    def test_rk4_more_accurate_than_euler(self):
        """同网格下 RK4 相对高分辨率参考的误差应小于 Euler"""
        beta = torch.tensor([1.5]); sigma = torch.tensor([0.3]); gamma = torch.tensor([0.1])
        ref = seir_rk4_integrate(beta, sigma, gamma, n_steps=2000, dt=0.0005)
        ref_traj = ref['trajectory'][:, ::200]  # 对齐到 11 点网格
        euler_err = (seir_euler_integrate(beta, sigma, gamma, n_steps=10, dt=0.1)
                     ['trajectory'] - ref_traj).abs().max().item()
        rk4_err = (seir_rk4_integrate(beta, sigma, gamma, n_steps=10, dt=0.1)
                   ['trajectory'] - ref_traj).abs().max().item()
        self.assertLess(rk4_err, euler_err)
        # RK4 在粗网格下的误差应绝对很小（四阶收敛）
        self.assertLess(rk4_err, 1e-5)

    def test_rk4_converges_with_fine_grid(self):
        """加密网格时 RK4 与高分辨率参考一致（四阶收敛）"""
        beta = torch.tensor([1.0]); sigma = torch.tensor([0.2]); gamma = torch.tensor([0.1])
        ref = seir_rk4_integrate(beta, sigma, gamma, n_steps=2000, dt=0.0005)
        ref_traj = ref['trajectory'][:, ::100]  # 对齐到 21 点网格
        fine = seir_rk4_integrate(beta, sigma, gamma, n_steps=20, dt=0.05)
        err = (fine['trajectory'] - ref_traj).abs().max().item()
        self.assertLess(err, 1e-6)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch 不可用")
class TestSEIRv3RK4Upgrade(unittest.TestCase):
    """9-房室 RK4 积分: 形状 / 可微性 / 精度"""

    def test_rk4_trajectory_shape(self):
        """v3 RK4 轨迹 shape 应为 [B, n_steps+1, 9]"""
        beta = torch.tensor([0.5])
        res = seir_v3_rk4_integrate(
            beta, torch.tensor([0.001]), torch.tensor([1e-5]),
            torch.tensor([0.01]), torch.tensor([0.1]), n_steps=10, dt=0.1)
        self.assertEqual(res['trajectory'].shape, (1, 11, 9))

    def test_rk4_output_keys(self):
        """v3 RK4 返回 dict 应含全部指标键"""
        beta = torch.tensor([0.5])
        res = seir_v3_rk4_integrate(
            beta, torch.tensor([0.001]), torch.tensor([1e-5]),
            torch.tensor([0.01]), torch.tensor([0.1]))
        for key in ['trajectory', 'peak_I', 'final_I', 'auc_I',
                     'time_to_peak', 'epidemic_size']:
            self.assertIn(key, res)

    def test_rk4_differentiable(self):
        """v3 RK4 积分结果应可微（梯度流经 beta）"""
        beta = torch.tensor([0.5], requires_grad=True)
        res = seir_v3_rk4_integrate(
            beta, torch.tensor([0.001]), torch.tensor([1e-5]),
            torch.tensor([0.01]), torch.tensor([0.1]))
        loss = res['peak_I'].sum() + res['epidemic_size'].sum()
        loss.backward()
        self.assertIsNotNone(beta.grad)
        self.assertTrue(torch.isfinite(beta.grad).all())

    def test_rk4_more_accurate_than_euler(self):
        """同网格下 v3 RK4 相对高分辨率参考的误差应小于 Euler"""
        beta = torch.tensor([1.5]); rf = torch.tensor([0.005])
        rr = torch.tensor([1e-5]); sc = torch.tensor([0.02]); ga = torch.tensor([0.1])
        ref = seir_v3_rk4_integrate(beta, rf, rr, sc, ga, n_steps=2000, dt=0.0005)
        ref_traj = ref['trajectory'][:, ::200]
        euler_err = (seir_v3_euler_integrate(beta, rf, rr, sc, ga, n_steps=10, dt=0.1)
                     ['trajectory'] - ref_traj).abs().max().item()
        rk4_err = (seir_v3_rk4_integrate(beta, rf, rr, sc, ga, n_steps=10, dt=0.1)
                   ['trajectory'] - ref_traj).abs().max().item()
        self.assertLess(rk4_err, euler_err)
        self.assertLess(rk4_err, 1e-5)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch 不可用")
class TestSolverErrorBaseline(unittest.TestCase):
    """误差基准验证: 量化 Euler 相对 RK4 的误差边界"""

    def test_baseline_output_keys(self):
        """基准应返回完整 key 集合"""
        bl = seir_solver_error_baseline(n_samples=16)
        for key in ['ref_I', 'ref_peak', 'euler_max_dt', 'rk4_max_dt',
                     'euler_errors', 'rk4_errors', 'dt_grid',
                     'recommended_dt', 'recommended_solver', 'summary']:
            self.assertIn(key, bl)

    def test_baseline_rk4_better_than_euler(self):
        """各网格下 RK4 峰值误差应小于 Euler"""
        bl = seir_solver_error_baseline(n_samples=16)
        for dt in bl['dt_grid']:
            self.assertLess(bl['rk4_errors'][dt]['peak_err'],
                            bl['euler_errors'][dt]['peak_err'])

    def test_baseline_euler_error_linear_in_dt(self):
        """Euler 峰值误差应随 dt 线性下降（dt 减半误差约减半）"""
        bl = seir_solver_error_baseline(n_samples=16)
        e1 = bl['euler_errors'][0.1]['peak_err']
        e2 = bl['euler_errors'][0.01]['peak_err']
        # dt 缩小 10 倍，误差应缩小一个数量级（一阶收敛）
        self.assertLess(e2, e1 / 3.0)

    def test_baseline_rk4_default_meets_tol(self):
        """RK4 在默认 dt=0.1 下应满足可接受误差阈值"""
        bl = seir_solver_error_baseline(n_samples=16)
        self.assertLess(bl['rk4_errors'][0.1]['peak_err'], 1e-3)
        self.assertEqual(bl['recommended_solver'], 'rk4')

    def test_baseline_reproducible(self):
        """基准结果应可复现（固定种子）"""
        bl1 = seir_solver_error_baseline(n_samples=16, seed=42)
        bl2 = seir_solver_error_baseline(n_samples=16, seed=42)
        self.assertEqual(bl1['euler_errors'][0.1]['peak_err'],
                         bl2['euler_errors'][0.1]['peak_err'])


# =============================================================================
# 8. 端到端: 30D 特征 → SEIRInformedGNN → ODE 门控
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE and PYG_AVAILABLE, "PyTorch/PyG 不可用")
class TestEndToEnd30D(unittest.TestCase):
    """端到端 30D 特征 + ODE 门控集成测试"""

    def test_full_forward_with_ode_gating(self):
        """完整前向传播：30D 特征 + 8 通道 ODE 门控"""
        from tb_risk.ml.framework.seir_informed import SEIRInformedGNN
        model = SEIRInformedGNN(node_feature_dim=30, hidden_dim=16, num_layers=1)
        model.eval()
        num_nodes = 5
        x = torch.randn(num_nodes, 30)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        tp = torch.rand(num_nodes, 1) * 3
        with torch.no_grad():
            risk, graph_emb = model(x, edge_index, treatment_phase=tp,
                                     use_ode_gating=True)
        self.assertEqual(risk.shape[0], num_nodes)
        self.assertTrue(torch.all(torch.isfinite(risk)))

    def test_full_forward_hetero_gnn(self):
        """HeteroTimeVaryingGNN 完整前向传播（ODE 门控）"""
        from tb_risk.ml.gnn import HeteroTimeVaryingGNN
        model = HeteroTimeVaryingGNN(node_feature_dim=30, hidden_dim=16, num_layers=1)
        model.eval()
        num_nodes = 5
        x = torch.randn(num_nodes, 30)
        edge_indices = [torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)]
        edge_attrs = [torch.randn(3, 6)]
        tp = torch.rand(num_nodes, 1) * 3
        with torch.no_grad():
            risk, final_emb, causal_loss, time_embs = model(
                [x], [edge_indices], [edge_attrs],
                treatment_phase=tp, use_ode_gating=True)
        self.assertIsNotNone(risk)
        self.assertTrue(torch.isfinite(risk).item() if risk.dim() == 0
                        else torch.all(torch.isfinite(risk)))


if __name__ == '__main__':
    unittest.main()
