#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PI 接触网络 DGP v1 + PI-GNN 消融测试。

覆盖三个模块：
  1. validation/pi_network.py —— 物理信息接触网络 DGP v1
     （SEIR 宏观 Λ + Wells-Riley 暴露修正 + 源传染性残差）；
  2. ml/gnn/pi_gnn.py —— 物理锚定 + 源池化残差融合网络
     （源专属均值聚合 + 线性读出 + 乘性融合 z = λ̃·exp(a·t)）；
  3. validation/pi_ablation.py —— 四臂消融阶梯
     （纯 SEIR / 纯 GNN / PI 特征级 / PI 特征+损失级 + oracle）。

核心科学主张（用户"第三层：PI-GNN 把 SEIR 先验注入网络层"）：
  已知用方程、未知用学习——SEIR 理论感染力 λ 给主排序，GNN 残差
  只修动物理模型无法刻画的传染源异质性 s̄（只有图可读）。
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)

from tb_risk.validation.pi_network import (  # noqa: E402
    SEIR_COMMUNITY_SPEC, PI_NETWORK_SPEC, BASE_FEATURE_DIM,
    EDGE_FEATURE_DIM, SOURCE_INFECTIVITY_SIGMA, community_seir_force,
    exposure_correction, build_pi_network, node_features,
    individual_features, physics_lambda, calibrate_physics_k, to_pi_graph,
)


# ==============================================================================
# 宏观：冻结社区 SEIR
# ==============================================================================

class TestCommunitySEIR:
    """宏观 SEIR 模块（RK4 积分 + 种子间抖动）。"""

    def test_spec_frozen_literature_anchors(self):
        """文献锚点：R0=β/γ=2.5、潜伏 8 周（IGRA 窗口）、传染期 10 周。"""
        s = SEIR_COMMUNITY_SPEC
        assert abs(s['beta'] / s['gamma'] - 2.5) < 1e-9
        assert abs(1.0 / s['sigma'] - 8.0) < 1e-9
        assert abs(1.0 / s['gamma'] - 10.0) < 1e-9
        assert s['solver'] == 'rk4'

    def test_force_positive_and_deterministic(self):
        """Λ_ref > 0；同 seed 重建一致。"""
        rng_a = np.random.RandomState(3)
        rng_b = np.random.RandomState(3)
        lam_a, info_a = community_seir_force(rng_a)
        lam_b, info_b = community_seir_force(rng_b)
        assert lam_a > 0.0
        assert lam_a == lam_b
        assert info_a == info_b
        assert info_a['t_ref_weeks'] == 8.0

    def test_jitter_varies_across_seeds(self):
        """β/I0 抖动：不同 seed 的 Λ 有差异（社区流行强度异质）。"""
        lams = {community_seir_force(np.random.RandomState(s))[0]
                for s in (1, 2, 3, 4, 5)}
        assert len(lams) > 1

    def test_rk4_mirror_torch_ode(self):
        """numpy RK4 与 ml/gnn/_seir_ode.py torch 版交叉一致（< 1e-6）。"""
        torch = pytest.importorskip('torch')
        from tb_risk.ml.gnn._seir_ode import seir_rk4_integrate
        spec = SEIR_COMMUNITY_SPEC
        from tb_risk.validation.pi_network import _seir_rk4_numpy
        np_traj = _seir_rk4_numpy(spec['beta'], spec['sigma'],
                                  spec['gamma'], spec['n_steps'],
                                  spec['dt'], spec['S0'], spec['E0'],
                                  spec['I0'], spec['R0'])
        out = seir_rk4_integrate(
            torch.tensor(spec['beta']), torch.tensor(spec['sigma']),
            torch.tensor(spec['gamma']), n_steps=spec['n_steps'],
            dt=spec['dt'], S0=spec['S0'], E0=spec['E0'],
            I0=spec['I0'], R0=spec['R0'])
        t_traj = out['trajectory'][0].numpy()   # [n_steps+1, 4]
        assert np_traj.shape == t_traj.shape
        assert np.abs(np_traj - t_traj).max() < 1e-6


# ==============================================================================
# 中观：感染力映射层（Wells-Riley 暴露修正）
# ==============================================================================

class TestExposureCorrection:
    """暴露修正因子的机制方向（Wells-Riley 乘积）。"""

    def test_monotone_in_frequency_and_duration(self):
        """频率/时长 ↑ → 暴露剂量 ↑（传播剂量的机制方向）。"""
        assert exposure_correction(20, 100, 1.0, 3) \
            > exposure_correction(5, 100, 1.0, 3)
        assert exposure_correction(20, 100, 1.0, 3) \
            > exposure_correction(20, 30, 1.0, 3)

    def test_monotone_against_distance_and_ventilation(self):
        """距离 ↑ / 通风差 ↑（数值大）→ 暴露剂量 ↓。"""
        assert exposure_correction(20, 100, 0.0, 3) \
            > exposure_correction(20, 100, 3.0, 3)
        assert exposure_correction(20, 100, 0.0, 1) \
            > exposure_correction(20, 100, 0.0, 5)

    def test_prox_ladder_decays(self):
        """近距离衰减阶梯单调递减。"""
        from tb_risk.validation.pi_network import PROX_BY_DIST
        assert all(PROX_BY_DIST[i] > PROX_BY_DIST[i + 1]
                   for i in range(3))


# ==============================================================================
# DGP v1：结构 + 信息分解
# ==============================================================================

class TestPINetworkDGP:
    """PI 接触网络 DGP 结构与三层桥接分解。"""

    def test_spec_registered(self):
        assert PI_NETWORK_SPEC['name'] == 'pi_contact_network_v1'
        assert 'host' in PI_NETWORK_SPEC['decomposition']
        assert PI_NETWORK_SPEC['source_infectivity'].startswith(
            '对数正态')

    def test_structure_and_fields(self):
        """前 M 指示病例 + n 接触者；边端点合法、无自环；字段完整。"""
        net = build_pi_network(n_contacts=120, random_state=7)
        M = net['M']
        nodes = net['nodes']
        assert M == net['n_clusters'] >= 2
        assert len(nodes) == M + 120
        assert all(n['is_index_case'] == 1 for n in nodes[:M])
        assert all(n['is_index_case'] == 0 for n in nodes[M:])
        assert all(n['infectivity'] > 0.0 for n in nodes[:M])
        assert all(n['infectivity'] == 0.0 for n in nodes[M:])
        for (u, v), feat in net['ef'].items():
            assert u != v, '自环'
            assert 0 <= u < len(nodes) and 0 <= v < len(nodes)
            assert len(feat) == EDGE_FEATURE_DIM
        for node in nodes:
            for field in ('lam_seir', 'nu_true', 's_bar',
                          'host_multiplier', 'n_memberships', 'age'):
                assert field in node

    def test_decomposition_identity(self):
        """三层桥接恒等式：λ = host·Λ·corr；ν = λ·s̄（逐接触者）。"""
        net = build_pi_network(n_contacts=200, random_state=11)
        mults = np.array([n['host_multiplier']
                          for n in net['nodes'][net['M']:]])
        expected_lam = mults * net['lam_community'] * net['corr']
        assert np.allclose(expected_lam, net['lam'], rtol=1e-12)
        assert np.allclose(net['lam'] * net['s_bar'], net['nu'],
                           rtol=1e-12)

    def test_residual_invisible_to_individual_features(self):
        """残差 s̄ 与个体基础特征无秩相关（信息不对称：只有图可读）。"""
        from scipy.stats import spearmanr
        net = build_pi_network(n_contacts=300, random_state=13)
        X = individual_features(net)
        rho, p = spearmanr(X[:, 5], net['s_bar'])   # 任一基础维
        assert abs(rho) < 0.2

    def test_source_infectivity_mean_anchored(self):
        """对数正态传染性 E[s]=1（归一化基准）且跨一个数量级。"""
        net = build_pi_network(n_contacts=400, random_state=17)
        inf = np.array([n['infectivity'] for n in net['nodes'][:net['M']]])
        assert abs(inf.mean() - 1.0) < 0.35
        assert np.log(inf.max() / inf.min()) > 2.0   # ln 跨度 > 2（一个数量级）
        assert SOURCE_INFECTIVITY_SIGMA == 0.7

    def test_label_calibration(self):
        """k 二分校准：标签阳性率 ≈ target_rate。"""
        net = build_pi_network(n_contacts=400, target_rate=0.25,
                               random_state=19)
        rate = float(np.mean(net['labels']))
        assert abs(rate - 0.25) < 0.06
        p = 1.0 - np.exp(-net['k_calibration'] * net['nu'])
        assert abs(p.mean() - 0.25) < 1e-6

    def test_determinism(self):
        a = build_pi_network(n_contacts=80, random_state=23)
        b = build_pi_network(n_contacts=80, random_state=23)
        assert a['labels'].tolist() == b['labels'].tolist()
        assert a['M'] == b['M']
        assert np.allclose(a['lam'], b['lam'])
        assert a['ef'].keys() == b['ef'].keys()

    def test_oracle_gap_positive(self):
        """oracle 间隙存在：AUROC(ν) > AUROC(λ)（残差修正可兑现）。"""
        from tb_risk.validation.threshold_spec import compute_auc
        net = build_pi_network(n_contacts=400, random_state=29)
        y = net['labels'].tolist()
        a_lam = compute_auc(net['lam'].tolist(), y)
        a_nu = compute_auc(net['nu'].tolist(), y)
        assert a_nu > a_lam


# ==============================================================================
# 特征与图构造
# ==============================================================================

class TestFeaturesAndGraph:
    """特征口径（三臂公平预算）与 torch 图张量。"""

    def test_node_features_layout(self):
        """14 维；源行第 1 维 = log 传染性、接触者行 = 0。"""
        net = build_pi_network(n_contacts=100, random_state=31)
        X = node_features(net)
        assert X.shape == (net['M'] + 100, BASE_FEATURE_DIM)
        assert BASE_FEATURE_DIM == 14
        M = net['M']
        assert np.all(X[M:, 1] == 0.0)
        expected = np.log([max(n['infectivity'], 1e-3)
                           for n in net['nodes'][:M]])
        assert np.allclose(X[:M, 1], expected)
        assert np.all(X[:, 0] == np.concatenate(
            [np.ones(M), np.zeros(100)]))

    def test_physics_lambda_and_calibration(self):
        """λ 提取一致；k̂ 校准训练半区检出率。"""
        net = build_pi_network(n_contacts=150, random_state=37)
        lam = physics_lambda(net)
        assert np.allclose(lam, net['lam'])
        y = np.asarray(net['labels'], dtype=float)
        tr = np.arange(0, 150, 2)
        k = calibrate_physics_k(net, y, tr)
        assert k > 0.0
        rate = float((1.0 - np.exp(-k * lam[tr])).mean())
        assert abs(rate - y[tr].mean()) < 1e-4

    def test_to_pi_graph_tensor_shapes(self):
        """include_physics：17 维 + λ̃ = k̂·λ + 双向边 + 归一化边特征。"""
        torch = pytest.importorskip('torch')
        net = build_pi_network(n_contacts=100, random_state=41)
        y = np.asarray(net['labels'], dtype=float)
        k = calibrate_physics_k(net, y, np.arange(0, 100, 2))
        g = to_pi_graph(net, include_physics=True, k_hat=k)
        N = net['M'] + 100
        assert g['x'].shape == (N, 17)
        assert g['x'].dtype == torch.float32
        assert g['edge_index'].shape[0] == 2
        assert g['edge_index'].shape[1] % 2 == 0      # 双向
        assert g['edge_attr'].shape == (g['edge_index'].shape[1], 6)
        assert float(g['lam_tilde'][:net['M']].abs().sum()) == 0.0
        assert np.allclose(g['lam_tilde'][net['M']:].numpy(),
                           k * net['lam'])
        # 物理特征 3 维：λ̃、Λ·1e3、传染源密度
        assert np.allclose(g['x'][:, 14].numpy(),
                           g['lam_tilde'].numpy())

    def test_to_pi_graph_base_has_no_physics(self):
        torch = pytest.importorskip('torch')
        net = build_pi_network(n_contacts=60, random_state=43)
        g = to_pi_graph(net)
        assert g['x'].shape[1] == BASE_FEATURE_DIM
        assert g['lam_tilde'] is None

    def test_include_physics_requires_k(self):
        with pytest.raises(ValueError):
            to_pi_graph(build_pi_network(n_contacts=40, random_state=1),
                        include_physics=True, k_hat=None)


# ==============================================================================
# PI-GNN：源池化残差融合
# ==============================================================================

class TestPIGNNNet:
    """源专属池化 + 线性读出 + 乘性融合的结构性质。"""

    @pytest.fixture()
    def small_graph(self):
        """手工小图：2 源 3 接触者，源池化答案可手算。"""
        torch = pytest.importorskip('torch')
        # x 列 1 = log inf：源 0 → log(2)，源 1 → log(0.5)
        x = torch.zeros(5, 4)
        x[0, 0], x[0, 1] = 1.0, float(np.log(2.0))
        x[1, 0], x[1, 1] = 1.0, float(np.log(0.5))
        x[2:, 2] = 0.5                      # 接触者特征
        # 边：源0-接触者0、源0-接触者1、源1-接触者1、接触者0-接触者2（互连）
        ei = torch.tensor([[0, 2, 0, 3, 1, 3, 2, 4],
                           [2, 0, 3, 0, 3, 1, 4, 2]], dtype=torch.long)
        lam_t = torch.tensor([0.0, 0.0, 0.5, 0.3, 0.0])
        return x, ei, lam_t

    def test_source_pool_correctness(self, small_graph):
        """池化只含指示病例边；孤立/互连邻居不计入分母。"""
        torch = pytest.importorskip('torch')
        from tb_risk.ml.gnn.pi_gnn import PIGNNNet
        x, ei, _ = small_graph
        msg = PIGNNNet.source_pool(x, ei, num_index_cases=2)
        # 接触者 0：只源 0 → [1, log2, 0, 0]
        assert torch.allclose(msg[2], x[0])
        # 接触者 1：源 0 + 源 1 均值
        assert torch.allclose(msg[3], (x[0] + x[1]) / 2)
        # 接触者 2：邻居只有接触者 0（互连）→ 零向量
        assert float(msg[4].abs().sum()) == 0.0
        # 指示病例自身不被池化（无源→源边）
        assert float(msg[:2].abs().sum()) == 0.0

    def test_forward_bounded_ratio_and_probability_identity(self,
                                                            small_graph):
        """z/λ̃ ∈ [e^−a, e^a]；sigmoid(logit) = 1 − exp(−z)。"""
        torch = pytest.importorskip('torch')
        from tb_risk.ml.gnn.pi_gnn import PIGNNNet
        torch.manual_seed(0)
        x, ei, lam_t = small_graph
        model = PIGNNNet(x.shape[1], residual_scale=1.0)
        logit, z, t = model(x, ei, lam_t, num_index_cases=2)
        assert float(t.detach().abs().max()) <= 1.0
        pos = lam_t[2:] > 0.0                    # λ̃>0 节点才有定义
        ratio = (z[2:] / lam_t[2:]).detach()[pos]
        assert float(ratio.min()) >= np.exp(-1.0) - 1e-6
        assert float(ratio.max()) <= np.exp(1.0) + 1e-6
        # 概率恒等式：p = sigmoid(logit) = 1 − exp(−z)
        p = torch.sigmoid(logit)
        assert torch.allclose(p, 1.0 - torch.exp(-z), atol=1e-6)
        # λ̃ = 0 节点（源/孤立）logit 有限（clamp 保护）
        assert torch.isfinite(logit).all()

    def test_readout_variants(self, small_graph):
        torch = pytest.importorskip('torch')
        from tb_risk.ml.gnn.pi_gnn import PIGNNNet
        x, ei, lam_t = small_graph
        PIGNNNet(x.shape[1], readout='mlp', hidden_dim=8)(x, ei, lam_t, 2)
        with pytest.raises(ValueError):
            PIGNNNet(x.shape[1], readout='bogus')

    def test_physics_consistency_loss(self):
        torch = pytest.importorskip('torch')
        from tb_risk.ml.gnn.pi_gnn import physics_consistency_loss
        z = torch.tensor([0.5, 1.0, 2.0])
        t = torch.tensor([0.1, -0.2, 0.3])
        lam = torch.tensor([0.4, 0.9, 2.2])
        pc = physics_consistency_loss(z, t, lam, rho_min=0.5)
        assert abs(float(pc['bounded']) - float(t.pow(2).mean())) < 1e-9
        assert float(pc['consistency']) >= 0.0
        # 高相关不罚、低相关罚（relu 下界形式）
        pc_hi = physics_consistency_loss(lam, t, lam, rho_min=0.5)
        assert float(pc_hi['consistency']) == 0.0
        pc_lo = physics_consistency_loss(-lam, t, lam, rho_min=0.5)
        assert float(pc_lo['consistency']) > 0.0

    def test_train_pi_gnn_deterministic_and_modes(self):
        """同 seed 逐位一致；两种 mode 输出 shape 正确。"""
        pytest.importorskip('torch')
        from tb_risk.ml.gnn.pi_gnn import train_pi_gnn
        net = build_pi_network(n_contacts=80, random_state=47)
        y = np.asarray(net['labels'], dtype=float)
        tr = np.arange(0, 80, 2)
        a = train_pi_gnn(net, y, tr, mode='feature', epochs=50, seed=5)
        b = train_pi_gnn(net, y, tr, mode='feature', epochs=50, seed=5)
        assert np.array_equal(a['score'], b['score'])
        assert a['score'].shape == (80,)
        assert a['p'].shape == (80,)
        assert a['delta_t'].shape == (80,)
        assert a['k_hat'] > 0.0
        c = train_pi_gnn(net, y, tr, mode='feature_loss', epochs=50,
                         seed=5)
        assert c['mode'] == 'feature_loss'
        assert np.isfinite(c['score']).all()
        # 训练不动点保护：零残差初始化下 p 与物理臂接近量级
        assert ((a['p'] >= 0.0) & (a['p'] <= 1.0)).all()

    def test_train_pure_gnn(self):
        pytest.importorskip('torch')
        from tb_risk.ml.gnn.pi_gnn import train_pure_gnn
        net = build_pi_network(n_contacts=80, random_state=53)
        y = np.asarray(net['labels'], dtype=float)
        tr = np.arange(0, 80, 2)
        a = train_pure_gnn(net, y, tr, epochs=50, seed=5)
        b = train_pure_gnn(net, y, tr, epochs=50, seed=5)
        assert np.array_equal(a['score'], b['score'])
        assert a['score'].shape == (80,)


# ==============================================================================
# 四臂消融
# ==============================================================================

class TestPIAblation:
    """四臂消融协议（分层半区 + 三指标 + DeLong）。"""

    def test_stratified_split(self):
        from tb_risk.validation.pi_ablation import _stratified_split
        y = np.array([0] * 30 + [1] * 10)
        rng = np.random.RandomState(2)
        tr, te = _stratified_split(y, rng)
        assert len(tr) + len(te) == 40
        assert not set(tr.tolist()) & set(te.tolist())
        assert abs(y[tr].mean() - y[te].mean()) < 0.15

    def test_recall_at_budget_hand_example(self):
        from tb_risk.validation.pi_ablation import _recall_at_budget
        y = np.array([1, 0, 1, 0, 0, 0, 0, 0])   # 2 阳性
        scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
        # 预算 25% → top2 = 分数前二 → 含 1 个阳性
        assert _recall_at_budget(scores, y, budget=0.25) == 0.5
        assert _recall_at_budget(scores, y, budget=0.5) == 1.0

    def test_run_once_structure(self):
        """小规模冒烟：arms 齐全、阶梯恒等式、概率合法。"""
        pytest.importorskip('torch')
        from tb_risk.validation.pi_ablation import run_pi_ablation_once
        rep = run_pi_ablation_once(n_contacts=100, seed=61,
                                   pi_epochs=100, gnn_epochs=60)
        assert set(rep['arms']) == {'seir', 'gnn', 'pi_feature',
                                    'pi_feature_loss', 'oracle_nu'}
        for name, arm in rep['arms'].items():
            assert 0.0 <= arm['auroc'] <= 1.0
            assert 0.0 <= arm['recall_at_budget'] <= 1.0
            assert arm['brier'] >= 0.0
            assert len(arm['test_scores']) == rep['n_test']
        # 阶梯恒等式（AUROC 差 = ladder 项）
        lad = rep['ladder']
        arms = rep['arms']
        assert abs(lad['pi_feature_minus_seir']
                   - (arms['pi_feature']['auroc']
                      - arms['seir']['auroc'])) < 1e-12
        assert abs(lad['oracle_minus_seir']
                   - (arms['oracle_nu']['auroc']
                      - arms['seir']['auroc'])) < 1e-12
        # delta_t 有界（tanh）
        for k in ('pi_feature', 'pi_feature_loss'):
            assert 0.0 <= rep['arms'][k]['delta_t_abs_mean'] <= 1.0
        # DeLong 键齐全
        assert set(rep['delong']) == {
            'pi_feature_vs_seir', 'pi_feature_vs_gnn',
            'pi_feature_loss_vs_seir', 'pi_feature_loss_vs_gnn',
            'pi_feature_loss_vs_pi_feature'}

    def test_run_multi_seed_summary(self):
        """多种子汇总结构 + 结论字段。"""
        pytest.importorskip('torch')
        from tb_risk.validation.pi_ablation import (
            run_multi_seed_pi_ablation)
        rep = run_multi_seed_pi_ablation(
            n_contacts=80, n_seeds=2, seed_start=71,
            pi_epochs=80, gnn_epochs=40, n_bootstrap=200)
        assert rep['design']['n_seeds'] == 2
        assert len(rep['seeds']) == 2
        for k in ('seir', 'gnn', 'pi_feature', 'pi_feature_loss',
                  'oracle_nu'):
            entry = rep['arm_summary'][k]
            for m in ('auroc', 'recall_at_budget', 'brier'):
                assert 'mean_' + m in entry
                ci = entry['bootstrap_ci_' + m]
                assert ci[0] <= ci[1]
        for k, ls in rep['ladder_summary'].items():
            assert ls['positive_seeds'] <= 2
            assert isinstance(ls['ci_excludes_zero'], bool)
        assert isinstance(rep['residual_learning_works'], bool)
        assert isinstance(rep['physics_anchor_helps'], bool)
        assert '残差学习检验' in rep['conclusion']


# ==============================================================================
# P3：PI 注入路径样本量自适应
# ==============================================================================

class TestPIInjectionMode:
    """select_pi_injection_mode 边界 + train_pi_gnn mode='auto' 解析。

    产品化依据（用户 P3 规格 + 2026-08-25 正式消融）：n=400 特征级
    +0.0038 占优、n=200 相当、n=100 损失级反超 +0.0083——交叉点
    100~200 之间取 150 为切换阈值。
    """

    def test_threshold_boundaries(self):
        """< 150 → 损失级；≥ 150 → 特征级；负数拒绝；阈值可配。"""
        from tb_risk.ml.gnn.pi_gnn import (
            select_pi_injection_mode, PI_INJECTION_SWITCH_THRESHOLD,
            PI_INJECTION_EVIDENCE)
        assert PI_INJECTION_SWITCH_THRESHOLD == 150
        # 证据链三档齐全（n400/n200/n100）
        assert set(PI_INJECTION_EVIDENCE) >= {
            'n400_feature_minus_loss', 'n200_feature_minus_loss',
            'n100_feature_minus_loss'}
        assert select_pi_injection_mode(149) == 'feature_loss'
        assert select_pi_injection_mode(150) == 'feature'
        assert select_pi_injection_mode(0) == 'feature_loss'
        assert select_pi_injection_mode(400) == 'feature'
        # 自定义阈值（可测试性）
        assert select_pi_injection_mode(99, threshold=100) == 'feature_loss'
        assert select_pi_injection_mode(100, threshold=100) == 'feature'
        with pytest.raises(ValueError):
            select_pi_injection_mode(-1)

    def test_train_pi_gnn_auto_resolves_by_n(self):
        """mode='auto'：n=100 解析为 feature_loss，n=160 为 feature。

        返回的 mode 是解析后的实际路径（非 'auto'）；显式非法
        mode 立即抛 ValueError（训练开始前）。
        """
        pytest.importorskip('torch')
        from tb_risk.ml.gnn.pi_gnn import train_pi_gnn
        # n=100 < 150 → 损失级（小样本正则路径）
        net_s = build_pi_network(n_contacts=100, random_state=91)
        y_s = np.asarray(net_s['labels'], dtype=float)
        tr_s = np.arange(0, 100, 2)
        r_s = train_pi_gnn(net_s, y_s, tr_s, mode='auto', epochs=30, seed=3)
        assert r_s['mode'] == 'feature_loss'
        assert np.isfinite(r_s['score']).all()
        # n=160 ≥ 150 → 特征级（常规路径）
        net_l = build_pi_network(n_contacts=160, random_state=92)
        y_l = np.asarray(net_l['labels'], dtype=float)
        tr_l = np.arange(0, 160, 2)
        r_l = train_pi_gnn(net_l, y_l, tr_l, mode='auto', epochs=30, seed=3)
        assert r_l['mode'] == 'feature'
        assert np.isfinite(r_l['score']).all()
        # 显式 mode 非法值在训练前拒绝
        with pytest.raises(ValueError):
            train_pi_gnn(net_s, y_s, tr_s, mode='bogus', epochs=5)
