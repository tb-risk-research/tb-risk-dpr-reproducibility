#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序接触网络 DGP v1 + 时序 GNN（STGNN）消融测试。

覆盖三个模块：
  1. validation/temporal_network.py —— 时序接触网络 DGP v1
     （时间切片 + 衰减权重 + 信息不对称标签机制）；
  2. ml/gnn/temporal_gnn.py —— 轻量时序 GNN
     （每窗 HeteroGAT 聚合 + 窗间 GRU 状态传递）；
  3. validation/temporal_ablation.py —— 五臂消融阶梯
     （个体基线 / 静态聚合 / 分窗聚合 / STGNN / 时序盲对照）。

核心科学主张（用户"第二层：时序 GNN（接触时间维度）"）：
  个体特征里的 time_span（接触持续时间）是粗粒度总量——100 小时
  的接触集中在最近两周还是分散在半年里，传播风险完全不同，而
  这一"近期密集接触"时序模式只存在于按时间切片的图里。
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)

from tb_risk.validation.temporal_network import (  # noqa: E402
    WINDOW_SPEC, PROFILE_SPEC, DECAY_WEIGHTS, DECAY_HALF_LIFE_WEEKS,
    TEMPORAL_NETWORK_SPEC, NUM_WINDOWS, build_temporal_network,
    node_features, individual_features, static_agg_features,
    window_agg_features, to_stgnn_graph,
)


# ==============================================================================
# DGP v1：时序接触网络（时间切片 + 衰减权重信息不对称）
# ==============================================================================

class TestTemporalNetworkDGP:
    """DGP v1 结构与时间信息不对称机制。"""

    def test_window_spec_five_windows_with_decay(self):
        """5 个时间窗（用户规格）+ 单调衰减（文献锚定）。"""
        assert NUM_WINDOWS == 5
        labels = [s['label'] for s in WINDOW_SPEC]
        assert labels == ['0-2周', '2-4周', '1-3月', '3-6月', '6月+']
        # 衰减单调递减（窗 id 0 最近 → 4 最远）
        d = DECAY_WEIGHTS
        assert all(d[i] > d[i + 1] for i in range(4))
        # 锚点：半衰期 8 周（IGRA 窗口期）——1-3 月窗（中点 8 周）
        # 衰减 ≈ 0.5
        assert abs(d[2] - 0.5) < 1e-9
        # 近/远对比 > 15x（时序画像的信号强度保障）
        assert d[0] / d[4] > 15.0
        # 每窗 spec 含文献引用（单一真值源登记）
        assert all(s.get('reference') for s in WINDOW_SPEC)
        assert DECAY_HALF_LIFE_WEEKS == 8.0
        assert TEMPORAL_NETWORK_SPEC['name'] == 'temporal_contact_network_v1'

    def test_profile_spec_three_patterns(self):
        """时序画像三档：recent / spread / old（用户"集中近期 vs
        分散"对比的构造载体），概率和为 1。"""
        assert set(PROFILE_SPEC) == {'recent', 'spread', 'old'}
        total = sum(p['prob'] for p in PROFILE_SPEC.values())
        assert abs(total - 1.0) < 1e-9
        # recent 落近窗、old 落远窗、spread 全窗
        assert max(PROFILE_SPEC['recent']['windows']) <= 1
        assert min(PROFILE_SPEC['old']['windows']) >= 3
        assert len(PROFILE_SPEC['spread']['windows']) == NUM_WINDOWS

    def test_index_case_nodes(self):
        """v1 结构沿用 v3：前 M 节点为指示病例（传染源）。"""
        net = build_temporal_network(n_contacts=120, random_state=7)
        M = net['M']
        nodes = net['nodes']
        assert M == net['n_clusters'] >= 2
        assert len(nodes) == M + 120
        assert all(n['is_index_case'] == 1 for n in nodes[:M])
        assert all(n['is_index_case'] == 0 for n in nodes[M:])
        # 传染性 0-1，只存在于指示病例节点
        assert all(0.0 < n['infectivity'] < 1.0 for n in nodes[:M])
        assert all(n['infectivity'] == 0.0 for n in nodes[M:])
        # 标签只对接触者
        assert len(net['labels']) == 120

    def test_build_network_structure_and_alignment(self):
        """边端点合法、无自环、窗桶齐全、节点字段完整。"""
        net = build_temporal_network(n_contacts=300, random_state=9)
        n_total = len(net['nodes'])
        assert set(net['ef'].keys()) == set(range(NUM_WINDOWS))
        n_nonempty = sum(len(net['ef'][w]) > 0 for w in net['ef'])
        assert n_nonempty >= 4          # 至少 4 窗有边
        total_edges = 0
        for w in net['ef']:
            for (u, v) in net['ef'][w]:
                assert u != v, '自环'
                assert 0 <= u < n_total and 0 <= v < n_total
                assert len(net['ef'][w][(u, v)]) == 6
                total_edges += 1
        assert total_edges > 0
        # 节点字段含宿主、个体可观测总量与真值暴露
        for node in (net['nodes'][0], net['nodes'][net['M']]):
            for field in ('age', 'has_symptoms', 'bcg_vaccine',
                          'past_illness_type', 'cumulative_exposure',
                          'temporal_exposure', 'host_multiplier',
                          'profile'):
                assert field in node

    def test_determinism(self):
        """同 seed 重建逐字段一致。"""
        a = build_temporal_network(n_contacts=60, random_state=11)
        b = build_temporal_network(n_contacts=60, random_state=11)
        assert a['labels'].tolist() == b['labels'].tolist()
        assert a['M'] == b['M']
        assert a['profiles'] == b['profiles']
        for na, nb in zip(a['nodes'], b['nodes']):
            assert na == nb
        assert a['ef'] == b['ef']

    def test_positive_rate_calibration(self):
        """二分校准：实际阳性率 ≈ 目标患病率（Bernoulli 抽样波动内）。"""
        net = build_temporal_network(n_contacts=600, random_state=3,
                                     target_rate=0.25)
        rate = float(np.mean(net['labels']))
        # sd ≈ sqrt(0.25*0.75/600) ≈ 0.018；0.05 ≈ 2.8σ
        assert abs(rate - 0.25) < 0.05

    def test_info_asymmetry_exists(self):
        """时间信息不对称存在性：累积总量是不完美代理。

        temporal_exposure（真实风险驱动，含衰减）与
        cumulative_exposure（个体可观测，无衰减）相关但
        corr < 0.9；且 temporal 的排序能力（AUROC）强于
        cumulative——时间加权构成只存在于切片图。
        （Pearson 不适用教训沿用：p 饱和 + 重尾使线相关性
        系统性低估判别力，以 AUROC 口径断言。）
        """
        from tb_risk.validation.threshold_spec import compute_auc
        net = build_temporal_network(n_contacts=500, random_state=5)
        contacts = net['nodes'][net['M']:]
        temporal = [n['temporal_exposure'] for n in contacts]
        cum = [n['cumulative_exposure'] for n in contacts]
        labels = net['labels'].tolist()
        r = float(np.corrcoef(cum, temporal)[0, 1])
        assert 0.3 < r < 0.9, f'不对称构造失效: corr={r}'
        assert compute_auc(temporal, labels) > compute_auc(cum, labels)

    def test_profile_mechanism_recent_riskier_than_old(self):
        """构造性检验：同等总暴露量下，近期画像的风险更高。

        temporal/cum = 贡献的衰减加权均值（"有效暴露比值"）——
        recent 画像应显著高于 old 画像（用户"100 小时集中在
        最近两周 vs 分散半年"的直接后果）。
        """
        net = build_temporal_network(n_contacts=600, random_state=13)
        contacts = net['nodes'][net['M']:]
        temporal = np.array([n['temporal_exposure'] for n in contacts])
        cum = np.array([n['cumulative_exposure'] for n in contacts])
        prof = np.array(net['profiles'])
        mask = cum > 1e-9
        ratio = {}
        for p in ('recent', 'spread', 'old'):
            m = mask & (prof == p)
            assert m.sum() >= 10
            ratio[p] = float((temporal[m] / cum[m]).mean())
        assert ratio['recent'] > ratio['spread'] > ratio['old']
        # 近/远有效暴露比值差距 > 3x（信号强度保障）
        assert ratio['recent'] / ratio['old'] > 3.0

    def test_profile_not_inferable_from_individual_features(self):
        """v2 教训回归：个体特征不得成为时序画像的代理。

        窗口指派独立于边特征/宿主/强度——用个体特征预测画像
        （recent vs old 二分类）应接近随机（AUROC 上界留裕量）。
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        net = build_temporal_network(n_contacts=600, random_state=19)
        prof = np.array(net['profiles'])
        mask = np.isin(prof, ['recent', 'old'])
        Xi = individual_features(net)[mask]
        y = (prof[mask] == 'recent').astype(int)
        if y.sum() == 0 or y.sum() == len(y):   # pragma: no cover
            pytest.skip('画像退化')
        tr = np.arange(0, len(y), 2)
        lr = LogisticRegression(max_iter=2000).fit(Xi[tr], y[tr])
        auroc = roc_auc_score(y, lr.predict_proba(Xi)[:, 1])
        assert auroc < 0.65, f'个体特征可推断画像（v2 失效）: {auroc:.3f}'

    def test_host_multiplier_signal_preserved(self):
        """v3 语义保持：宿主乘数与标签正相关（单一真值源）。"""
        net = build_temporal_network(n_contacts=600, random_state=17)
        contacts = net['nodes'][net['M']:]
        mults = np.array([n['host_multiplier'] for n in contacts])
        labels = np.asarray(net['labels'])
        assert mults[labels == 1].mean() > mults[labels == 0].mean()
        assert float(np.corrcoef(mults, labels)[0, 1]) > 0.0


# ==============================================================================
# 特征构造（消融臂的公平特征预算）
# ==============================================================================

class TestFeatureConstruction:
    """个体 / 静态聚合 / 分窗聚合特征。"""

    @pytest.fixture(scope='class')
    def net(self):
        return build_temporal_network(n_contacts=150, random_state=21)

    def test_node_features_shape(self, net):
        """全图节点特征：指示病例 + 接触者，12 维同构。"""
        X = node_features(net)
        assert X.shape == (net['M'] + 150, 12)
        assert np.all(np.isfinite(X))
        # 接触者行前两维（指示病例标记/传染性）恒 0——信息不对称上界
        assert np.all(X[net['M']:, :2] == 0.0)
        # 指示病例行携带传染性（只有图可读）
        assert np.all(X[:net['M'], 1] > 0.0)

    def test_individual_features_shape(self, net):
        """个体特征：接触者矩阵（信息上界基线，无时序信息）。"""
        X = individual_features(net)
        assert X.shape == (150, 12)
        assert np.all(np.isfinite(X))

    def test_static_agg_extends_individual(self, net):
        """静态聚合：个体特征 + 跨窗等权邻居均值（时序盲）。"""
        Xi = individual_features(net)
        Xs = static_agg_features(net)
        assert Xs.shape == (150, 2 * 12)
        assert np.allclose(Xs[:, :12], Xi)

    def test_window_agg_differs_from_static(self, net):
        """分窗聚合与静态聚合必然不同（时序区分）。

        若二者等价（分窗 = 静态块平铺 5 次），说明时间区分信息
        未进入特征——机制构造失败。
        """
        Xs = static_agg_features(net)
        Xw = window_agg_features(net)
        assert Xw.shape == (150, 12 + NUM_WINDOWS * 12)
        assert np.allclose(Xw[:, :12], Xs[:, :12])
        assert np.abs(Xw[:, 12:]).sum() > 0
        assert not np.allclose(Xw[:, 12:], np.tile(Xs[:, 12:], (1, NUM_WINDOWS)))

    def test_to_stgnn_graph_alignment(self, net):
        """STGNN 图：5 窗、最旧→最新排序、双向边、张量对齐。"""
        torch = pytest.importorskip('torch')
        g = to_stgnn_graph(net)
        x, wins, wids = g['x'], g['windows'], g['window_ids']
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] == net['M'] + 150
        assert x.shape[1] == 12
        assert len(wins) == NUM_WINDOWS
        # 窗序列最旧 → 最新（GRU 最后一步 = 最新状态）
        assert wids == [4, 3, 2, 1, 0]
        for w in wins:
            assert w['edge_index'].shape[1] == w['edge_attr'].shape[0]
            assert w['edge_attr'].shape[1] == 6
            # 双向边（无向接触语义：GAT 聚合需要入边对称）
            pairs = set(map(tuple, w['edge_index'].t().tolist()))
            assert all((b, a) in pairs for a, b in pairs)

    def test_to_stgnn_graph_edge_conservation(self, net):
        """各窗边数之和 = ef 总条目（逐窗张量不丢边）。"""
        g = to_stgnn_graph(net)
        n_packed = sum(w['edge_index'].shape[1] // 2 for w in g['windows'])
        n_src = sum(len(net['ef'][w]) for w in range(NUM_WINDOWS))
        assert n_packed == n_src

    def test_to_stgnn_graph_merged_control(self, net):
        """时序盲对照：全窗并入单窗，边数 = 跨窗去重并集。"""
        torch = pytest.importorskip('torch')
        g_seq = to_stgnn_graph(net)
        g_merge = to_stgnn_graph(net, windows_merged=True)
        assert len(g_merge['windows']) == 1
        w = g_merge['windows'][0]
        n_merge = w['edge_index'].shape[1] // 2
        n_union = len({k for win in range(NUM_WINDOWS)
                       for k in net['ef'][win]})
        assert n_merge == n_union
        # 跨窗重复对已去重（≤ 逐窗边数总和）
        assert n_merge <= sum(len(net['ef'][win])
                              for win in range(NUM_WINDOWS))
        assert torch.equal(g_merge['x'], g_seq['x'])


# ==============================================================================
# 轻量时序 GNN（每窗聚合 + 窗间 GRU）
# ==============================================================================

class TestTemporalGNNModel:
    """TemporalGNN 训练与结构规格。"""

    @pytest.fixture(scope='class')
    def net(self):
        return build_temporal_network(n_contacts=150, random_state=23,
                                      target_rate=0.25)

    def test_model_spec_shared_gat_and_gru(self, net):
        """结构规格：窗间参数共享的单 GAT 层 + GRU + 时序风险曲线。"""
        pytest.importorskip('torch')
        from tb_risk.ml.gnn.temporal_gnn import TemporalGNN
        model = TemporalGNN(node_dim=12)
        assert model.num_windows == 5
        assert hasattr(model, 'gru') and isinstance(
            model.gru, __import__('torch').nn.GRU)
        # 单一窗聚合层（参数跨窗共享）
        assert not isinstance(model.window_gat, __import__('torch').nn.ModuleList)

    def test_forward_risk_curve_shape(self, net):
        """前向输出：[W, N] 时序风险曲线（logits），末行 = 最新窗。"""
        torch = pytest.importorskip('torch')
        from tb_risk.ml.gnn.temporal_gnn import TemporalGNN
        g = to_stgnn_graph(net)
        model = TemporalGNN(node_dim=g['x'].shape[1])
        with torch.no_grad():
            curve = model(g['x'], g['windows'])
        assert curve.shape == (NUM_WINDOWS, net['M'] + 150)
        assert torch.all(torch.isfinite(curve))

    def test_train_loss_decreases_and_deterministic(self, net):
        """训练损失下降 + 同 seed 确定性 + 输出仅接触者 + 曲线形状。"""
        pytest.importorskip('torch')
        from tb_risk.ml.gnn.temporal_gnn import train_temporal_gnn
        labels = np.asarray(net['labels'])
        train_idx = np.arange(0, 150, 2)   # 接触者空间
        out_a = train_temporal_gnn(net, labels, train_contacts=train_idx,
                                   epochs=120, seed=99)
        assert out_a['final_loss'] < out_a['initial_loss'] * 0.8
        scores = np.asarray(out_a['scores'])
        assert len(scores) == 150          # 仅接触者（无指示病例）
        assert np.all((scores >= 0) & (scores <= 1))
        # 时序风险曲线：[W, n_contacts]，末行 = 当前风险（scores）
        curve = np.asarray(out_a['risk_curve'])
        assert curve.shape == (NUM_WINDOWS, 150)
        assert np.allclose(curve[-1], scores, atol=1e-6)
        out_b = train_temporal_gnn(net, labels, train_contacts=train_idx,
                                   epochs=120, seed=99)
        assert np.allclose(out_a['scores'], out_b['scores'], atol=1e-6)

    def test_merged_control_same_protocol(self, net):
        """时序盲对照：windows_merged=True 单窗、损失仍可下降。"""
        pytest.importorskip('torch')
        from tb_risk.ml.gnn.temporal_gnn import train_temporal_gnn
        labels = np.asarray(net['labels'])
        train_idx = np.arange(0, 150, 2)
        out = train_temporal_gnn(net, labels, train_contacts=train_idx,
                                 epochs=120, seed=99, windows_merged=True)
        assert out['windows_merged'] is True
        assert len(out['scores']) == 150
        # 单窗：时序风险曲线只有 1 步
        assert np.asarray(out['risk_curve']).shape == (1, 150)
        assert out['final_loss'] < out['initial_loss']


# ==============================================================================
# 五臂消融阶梯
# ==============================================================================

class TestTemporalAblation:
    """消融阶梯结构 + 构造性机制检验。"""

    @pytest.fixture(scope='class')
    def report(self):
        from tb_risk.validation.temporal_ablation import run_temporal_ablation_once
        return run_temporal_ablation_once(n_contacts=200, seed=31,
                                          epochs=150, target_rate=0.25)

    def test_five_arms_structure(self, report):
        """五臂齐全：individual / static_agg / window_agg / stgnn /
        stgnn_merged。"""
        arms = report['arms']
        for key in ('individual', 'static_agg', 'window_agg',
                    'stgnn', 'stgnn_merged'):
            assert key in arms
            assert 0.0 <= arms[key]['auroc'] <= 1.0
        assert report['ladder']['window_minus_static'] is not None
        assert report['ladder']['stgnn_minus_stgnn_merged'] is not None

    def test_scores_aligned_with_labels(self, report):
        """测试半区分数与标签同长（DeLong 配对前提）。"""
        y = report['test_labels']
        for arm in report['arms'].values():
            assert len(arm['test_scores']) == len(y)

    def test_delong_pairs_available(self, report):
        """逐对 DeLong：STGNN vs 其余臂。"""
        d = report['delong']
        assert 'stgnn_vs_static_agg' in d
        assert 'stgnn_vs_window_agg' in d
        assert 'stgnn_vs_stgnn_merged' in d
        assert 0.0 <= d['stgnn_vs_stgnn_merged']['p_value'] <= 1.0

    def test_constructive_mechanism_check(self):
        """构造性机制检验（时间信息不对称存在时，多种子均值口径）：
        时序区分（window_agg）应优于静态聚合；端到端 STGNN 应优于
        个体基线。这是"机制可行性"验证而非自然发现——DGP 构造
        保证了不对称存在（单种子有波动，取 3 种子均值）。

        注：n_contacts=400 与正式消融协议一致——STGNN 的 GRU
        容量在小样本（n=300 训练半区 150）下会记忆化训练集，
        n≥400 时 8/8 种子方向正确（2026-08-25 扫描实证）。
        """
        from tb_risk.validation.temporal_ablation import run_temporal_ablation_once
        window_minus_static, stgnn_minus_ind = [], []
        for seed in (101, 102, 103):
            rep = run_temporal_ablation_once(n_contacts=400, seed=seed,
                                             epochs=300, target_rate=0.25)
            window_minus_static.append(rep['ladder']['window_minus_static'])
            stgnn_minus_ind.append(
                rep['ladder']['stgnn_minus_individual'])
        assert float(np.mean(window_minus_static)) > 0
        assert float(np.mean(stgnn_minus_ind)) > 0
