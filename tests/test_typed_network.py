#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""类型化边网络 DGP v3 + 异质注意力（GAT）消融测试。

覆盖三个模块：
  1. validation/typed_network.py —— 类型化接触网络 DGP v3
     （传染源节点 + 同分布边特征 + 信息不对称标签机制）；
  2. ml/gnn/typed_gat.py —— 轻量类型感知 GAT
     （2 层 HeteroGATLayer + 残差，接入现有骨架）；
  3. validation/typed_ablation.py —— 五臂消融阶梯
     （个体基线 / 均值聚合 / 分通道聚合 / GAT异质 / GAT同质对照）。

核心科学主张（用户"第一层：异质边 + 注意力"）：
  个体特征只能观测"累积暴露总量"，看不到"接触类型构成"（家庭密接
  vs 偶遇的传播风险差一个数量级）——信息不对称由网络层利用。
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)

from tb_risk.validation.typed_network import (  # noqa: E402
    EDGE_TYPE_SPEC, TYPED_NETWORK_SPEC, build_typed_network,
    node_features, individual_features, mean_agg_features,
    channel_agg_features, to_gat_graph,
)


# ==============================================================================
# DGP v3：类型化接触网络（传染源节点 + 同分布边特征）
# ==============================================================================

class TestTypedNetworkDGP:
    """DGP v3 结构与信息不对称机制。"""

    def test_edge_type_spec_five_types_with_beta_gradient(self):
        """5 种边类型 + β 传播梯度：家庭 >> 同事/同学 > 社会 >> 偶遇。"""
        ids = [e['id'] for e in EDGE_TYPE_SPEC]
        assert set(ids) == {'household', 'workplace', 'school', 'social', 'casual'}
        beta = {e['id']: e['base_transmission'] for e in EDGE_TYPE_SPEC}
        assert beta['household'] > beta['workplace'] >= beta['school']
        assert beta['school'] > beta['social'] > beta['casual']
        # 文献锚点：家庭 vs 偶遇传播率比值 > 10（Fox 2013 密接 30-40% vs 社区 <5%）
        assert beta['household'] / beta['casual'] > 10.0
        # 每条 spec 含文献引用（单一真值源登记）
        assert all(e.get('reference') for e in EDGE_TYPE_SPEC)
        # v3 规格：同分布边特征 + 传染源节点
        assert TYPED_NETWORK_SPEC['name'] == 'typed_contact_network_v3'

    def test_index_case_nodes(self):
        """v3 结构：前 M 节点为指示病例（传染源，infectivity 只在图）。"""
        net = build_typed_network(n_contacts=120, random_state=7)
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
        # 绝大多数指示病例在图中有关联边（作为传染源参与传播）
        src_ids = set(range(M))
        touched = set()
        for t in net['ef']:
            for (u, v) in net['ef'][t]:
                if u in src_ids:
                    touched.add(u)
                if v in src_ids:
                    touched.add(v)
        assert len(touched) >= M // 2

    def test_build_network_structure_and_alignment(self):
        """边端点合法、无自环、类型通道齐全、节点字段完整。"""
        net = build_typed_network(n_contacts=300, random_state=9)
        n_total = len(net['nodes'])
        M = net['M']
        # 至少 4/5 类型通道有边（n=300 时类型覆盖概率 > 98%）
        n_nonempty = sum(len(net['ef'][t]) > 0 for t in net['ef'])
        assert n_nonempty >= 4
        total_edges = 0
        for t in net['ef']:
            for (u, v) in net['ef'][t]:
                assert u != v, '自环'
                assert 0 <= u < n_total and 0 <= v < n_total
                assert len(net['ef'][t][(u, v)]) == 6
                total_edges += 1
        assert total_edges > 0
        # 节点字段含宿主与个体可观测暴露特征
        for node in (net['nodes'][0], net['nodes'][M]):
            for field in ('age', 'has_symptoms', 'bcg_vaccine',
                          'past_illness_type', 'cumulative_exposure',
                          'typed_exposure', 'host_multiplier'):
                assert field in node

    def test_determinism(self):
        """同 seed 重建逐字段一致。"""
        a = build_typed_network(n_contacts=60, random_state=11)
        b = build_typed_network(n_contacts=60, random_state=11)
        assert a['labels'].tolist() == b['labels'].tolist()
        assert a['M'] == b['M']
        for na, nb in zip(a['nodes'], b['nodes']):
            assert na == nb
        assert a['ef'] == b['ef']

    def test_positive_rate_calibration(self):
        """二分校准：实际阳性率 ≈ 目标患病率（Bernoulli 抽样波动内）。"""
        net = build_typed_network(n_contacts=600, random_state=3,
                                  target_rate=0.25)
        rate = float(np.mean(net['labels']))
        # sd ≈ sqrt(0.25*0.75/600) ≈ 0.018；0.05 ≈ 2.8σ
        assert abs(rate - 0.25) < 0.05

    def test_info_asymmetry_exists(self):
        """信息不对称存在性：累积暴露量是不完美代理。

        typed_exposure（真实风险驱动，含 β）与 cumulative_exposure
        （个体可观测，无 β）相关但相关系数 < 0.9；且 typed 的排序
        能力（AUROC，与下游任务一致的度量）强于 cumulative——
        β 加权构成只存在于图。
        （注：Pearson 相关不适用——p 饱和与 typed 重尾使线相关性
        系统性低估 typed 的判别力，实证 5 种子 AUROC(typed) >
        AUROC(cum) 稳定成立。）
        """
        from tb_risk.validation.threshold_spec import compute_auc
        net = build_typed_network(n_contacts=500, random_state=5)
        contacts = net['nodes'][net['M']:]
        typed = [n['typed_exposure'] for n in contacts]
        cum = [n['cumulative_exposure'] for n in contacts]
        labels = net['labels'].tolist()
        r = float(np.corrcoef(cum, typed)[0, 1])
        assert 0.3 < r < 0.9, f'不对称构造失效: corr={r}'
        # 真实驱动因子的排序能力必须更强（AUROC 口径）
        assert compute_auc(typed, labels) > compute_auc(cum, labels)

    def test_household_dominant_higher_risk_than_casual(self):
        """构造性检验：同等总接触量下家庭暴露占比高者 β 加权风险更高。

        typed/cum = 贡献的 β 加权均值（"有效传播率"）——家庭主导者
        应显著高于无家庭暴露者（信息不对称的直接后果）。
        """
        net = build_typed_network(n_contacts=600, random_state=13)
        contacts = net['nodes'][net['M']:]
        typed = np.array([n['typed_exposure'] for n in contacts])
        cum = np.array([n['cumulative_exposure'] for n in contacts])
        rows = net['node_exposure_by_type']
        hh_share = np.array([
            rows[i].get('household', 0.0) / max(cum[i], 1e-9)
            for i in range(len(cum))])
        mask = cum > 1e-9
        hh_share, typed, cum = hh_share[mask], typed[mask], cum[mask]
        med = float(np.median(hh_share))
        hi = typed[hh_share > med] / cum[hh_share > med]
        lo = typed[hh_share <= med] / cum[hh_share <= med]
        assert len(hi) >= 10 and len(lo) >= 10
        # 同等总接触量下家庭主导者有效传播率显著更高
        assert hi.mean() > lo.mean() * 1.5

    def test_host_multiplier_signal_preserved(self):
        """v3 语义保持：宿主乘数与标签正相关（单一真值源）。"""
        net = build_typed_network(n_contacts=600, random_state=17)
        contacts = net['nodes'][net['M']:]
        mults = np.array([n['host_multiplier'] for n in contacts])
        labels = np.asarray(net['labels'])
        assert mults[labels == 1].mean() > mults[labels == 0].mean()
        assert float(np.corrcoef(mults, labels)[0, 1]) > 0.0


# ==============================================================================
# 特征构造（消融臂的公平特征预算）
# ==============================================================================

class TestFeatureConstruction:
    """个体 / 均值聚合 / 分通道聚合特征。"""

    @pytest.fixture(scope='class')
    def net(self):
        return build_typed_network(n_contacts=150, random_state=21)

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
        """个体特征：接触者矩阵（信息上界基线）。"""
        X = individual_features(net)
        assert X.shape == (150, 12)
        assert np.all(np.isfinite(X))

    def test_mean_agg_extends_individual(self, net):
        """均值聚合：个体特征 + 等权邻居均值（无类型信息）。"""
        Xi = individual_features(net)
        Xm = mean_agg_features(net)
        assert Xm.shape == (150, 2 * 12)
        # 前半部分 = 个体特征（拼接语义）
        assert np.allclose(Xm[:, :12], Xi)

    def test_channel_agg_differs_from_mean(self, net):
        """分通道聚合与均值聚合必然不同（类型区分）。

        若二者等价（分通道 = 均值块平铺 5 次），说明类型区分信息
        未进入特征——机制构造失败。
        """
        Xm = mean_agg_features(net)
        Xc = channel_agg_features(net)
        assert Xc.shape == (150, 12 + 5 * 12)
        assert np.allclose(Xc[:, :12], Xm[:, :12])
        assert np.abs(Xc[:, 12:]).sum() > 0
        assert not np.allclose(Xc[:, 12:], np.tile(Xm[:, 12:], (1, 5)))

    def test_to_gat_graph_alignment(self, net):
        """GAT 图：5 类型通道索引/特征对齐 + 双向 + torch 张量。"""
        torch = pytest.importorskip('torch')
        g = to_gat_graph(net)
        x, eidx, eattr = g['x'], g['edge_indices'], g['edge_attrs']
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] == net['M'] + 150
        assert x.shape[1] == 12
        assert len(eidx) == 5 and len(eattr) == 5
        for k in range(5):
            assert eidx[k].shape[1] == eattr[k].shape[0]
            assert eattr[k].shape[1] == 6
            # 双向边（无向接触语义：GAT 聚合需要入边对称）
            pairs = set(map(tuple, eidx[k].t().tolist()))
            assert all((b, a) in pairs for a, b in pairs)

    def test_to_gat_graph_merged_control(self, net):
        """同质对照：5 类边并入单通道，边数守恒（同结构对照前提）。"""
        torch = pytest.importorskip('torch')
        g_hetero = to_gat_graph(net)
        g_merge = to_gat_graph(net, edge_types_merged=True)
        assert len(g_merge['edge_indices']) == 1
        assert len(g_merge['edge_attrs']) == 1
        n_hetero = sum(e.shape[1] for e in g_hetero['edge_indices'])
        assert g_merge['edge_indices'][0].shape[1] == n_hetero
        assert g_merge['edge_attrs'][0].shape[0] == n_hetero
        assert torch.equal(g_merge['x'], g_hetero['x'])


# ==============================================================================
# 轻量 GAT（2 层 + 残差，复用 HeteroGATLayer）
# ==============================================================================

class TestTypedGATModel:
    """TypedGATNet 训练与结构规格。"""

    @pytest.fixture(scope='class')
    def net(self):
        return build_typed_network(n_contacts=150, random_state=23,
                                   target_rate=0.25)

    def test_model_spec_two_layers_residual(self, net):
        """用户规格：2 层 + 残差连接 + 残差门控。"""
        pytest.importorskip('torch')
        from tb_risk.ml.gnn.typed_gat import TypedGATNet
        model = TypedGATNet(node_dim=12)
        assert len(model.gat_layers) == 2
        assert model.residual is True
        assert model.num_edge_types == 5
        assert len(model.residual_gates) == 2

    def test_train_loss_decreases_and_deterministic(self, net):
        """训练损失下降 + 同 seed 确定性 + 输出仅接触者。"""
        pytest.importorskip('torch')
        from tb_risk.ml.gnn.typed_gat import train_typed_gat
        labels = np.asarray(net['labels'])
        train_idx = np.arange(0, 150, 2)   # 接触者空间
        out_a = train_typed_gat(net, labels, train_contacts=train_idx,
                                epochs=120, seed=99)
        assert out_a['final_loss'] < out_a['initial_loss'] * 0.8
        assert len(out_a['scores']) == 150   # 仅接触者（无指示病例）
        assert np.all((np.asarray(out_a['scores']) >= 0)
                      & (np.asarray(out_a['scores']) <= 1))
        out_b = train_typed_gat(net, labels, train_contacts=train_idx,
                                epochs=120, seed=99)
        assert np.allclose(out_a['scores'], out_b['scores'], atol=1e-6)

    def test_merged_control_same_protocol(self, net):
        """同质对照：edge_types_merged=True 单通道、损失仍可下降。"""
        pytest.importorskip('torch')
        from tb_risk.ml.gnn.typed_gat import train_typed_gat
        labels = np.asarray(net['labels'])
        train_idx = np.arange(0, 150, 2)
        out = train_typed_gat(net, labels, train_contacts=train_idx,
                              epochs=120, seed=99, edge_types_merged=True)
        assert out['edge_types_merged'] is True
        assert len(out['scores']) == 150
        assert out['final_loss'] < out['initial_loss']


# ==============================================================================
# 五臂消融阶梯
# ==============================================================================

class TestTypedAblation:
    """消融阶梯结构 + 构造性机制检验。"""

    @pytest.fixture(scope='class')
    def report(self):
        from tb_risk.validation.typed_ablation import run_typed_ablation_once
        return run_typed_ablation_once(n_contacts=200, seed=31,
                                       epochs=150, target_rate=0.25)

    def test_five_arms_structure(self, report):
        """五臂齐全：individual / mean_agg / channel_agg / gat_hetero /
        gat_homo。"""
        arms = report['arms']
        for key in ('individual', 'mean_agg', 'channel_agg',
                    'gat_hetero', 'gat_homo'):
            assert key in arms
            assert 0.0 <= arms[key]['auroc'] <= 1.0
        assert report['ladder']['channel_minus_mean'] is not None
        assert report['ladder']['gat_hetero_minus_gat_homo'] is not None

    def test_scores_aligned_with_labels(self, report):
        """测试半区分数与标签同长（DeLong 配对前提）。"""
        y = report['test_labels']
        for arm in report['arms'].values():
            assert len(arm['test_scores']) == len(y)

    def test_delong_pairs_available(self, report):
        """逐对 DeLong：GAT异质 vs 其余臂。"""
        d = report['delong']
        assert 'gat_hetero_vs_mean_agg' in d
        assert 'gat_hetero_vs_channel_agg' in d
        assert 'gat_hetero_vs_gat_homo' in d
        assert 0.0 <= d['gat_hetero_vs_gat_homo']['p_value'] <= 1.0

    def test_constructive_mechanism_check(self):
        """构造性机制检验（信息不对称存在时，多种子均值口径）：
        类型区分（channel_agg）应优于均值聚合；端到端 GAT 应优于
        个体基线。这是"机制可行性"验证而非自然发现——DGP 构造
        保证了不对称存在（单种子有波动，取 3 种子均值）。
        """
        from tb_risk.validation.typed_ablation import run_typed_ablation_once
        channel_minus_mean, gat_minus_ind = [], []
        for seed in (31, 32, 33):
            rep = run_typed_ablation_once(n_contacts=300, seed=seed,
                                          epochs=300, target_rate=0.25)
            channel_minus_mean.append(rep['ladder']['channel_minus_mean'])
            gat_minus_ind.append(
                rep['ladder']['gat_hetero_minus_individual'])
        assert float(np.mean(channel_minus_mean)) > 0
        assert float(np.mean(gat_minus_ind)) > 0
