#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合机制 DGP v1 + 特征化消融测试（用户 P1/P2）。

覆盖两个模块：
  1. validation/combined_network.py —— 类型 × 时序 × PI 三机制合一
     DGP（窗口时点 Λ(t_w) + 特征化列 type/window exposure）；
  2. validation/combined_ablation.py —— 特征化阶梯消融
     （ind/+typed/+window/+pi/+all，冻结注册表 RF/LGBM + 可加性）。

核心科学主张（用户 P1/P2）：
  "channel_agg > gat_hetero 的发现指明低复杂度路径——分类型/分时间窗
  邻居聚合特征直接加进冻结的主模型特征集，零深度学习依赖、部署
  友好、可解释，大概率拿到机制收益的大头"；"类型通道 + 时序窗口 +
  PI 物理特征能否叠加？组合 vs 单机制消融确认收益是否可加"。
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)

from tb_risk.validation.combined_network import (  # noqa: E402
    AUDIT_CALIBRATED_PRIORS, COMBINED_SEIR_SPEC, COMBINED_NETWORK_SPEC,
    community_seir_window_forces, build_combined_network,
    feature_matrix, feature_dim, NET_FEATURE_NAMES,
    TYPE_EXPOSURE_NAMES, WINDOW_EXPOSURE_NAMES, PHYSICS_LAMBDA_NAME,
    FEATURE_SETS,
)
from tb_risk.validation.threshold_spec import compute_auc  # noqa: E402


def _auc(scores, labels):
    return compute_auc(list(map(float, scores)), list(map(int, labels)))


# ==============================================================================
# v3.1 修订依据：真实数据校准刻度（HomeACF 乘子审计）
# ==============================================================================

class TestAuditCalibratedPriors:
    """P3：先验-数据对照表常量契约（防意外篡改/删除）。"""

    def test_scale_fields_present_and_finite(self):
        for k in ('cough_slope_per_day', 'host_hiv_log',
                  'share_bedroom_log', 'age_ge45_log', 'log_ts_weight'):
            v = AUDIT_CALIBRATED_PRIORS[k]
            assert np.isfinite(v) and v != 0.0

    def test_meta_contract(self):
        m = AUDIT_CALIBRATED_PRIORS['audit_meta']
        assert m['n'] == 2725 and m['n_events'] == 359
        assert m['direction_agreement'] == '5/8'
        assert set(m['removed_flipped']) == {
            'age_lt5', 'index_hiv', 'sleep_same_bed'}
        assert m['removed_untestable'] == ['smear']

    def test_removed_flipped_not_in_scale_fields(self):
        """方向翻转乘子不得再以正刻度出现在校准先验里。"""
        scale = {k for k in AUDIT_CALIBRATED_PRIORS
                 if k != 'audit_meta'}
        for flipped in AUDIT_CALIBRATED_PRIORS['audit_meta'][
                'removed_flipped']:
            assert not any(flipped in k for k in scale)


# ==============================================================================
# 宏观 SEIR：窗口时点社区感染力
# ==============================================================================

class TestCommunitySEIRWindowForces:

    def test_window_forces_shapes_positive(self):
        rng = np.random.RandomState(42)
        out = community_seir_window_forces(rng)
        assert len(out['lam_w_truth']) == 5
        assert len(out['lam_w_default']) == 5
        assert all(v > 0 for v in out['lam_w_truth'])
        assert out['lam_ref_default'] > 0
        assert len(out['lam_trajectory_truth']) == COMBINED_SEIR_SPEC[
            'n_steps'] + 1

    def test_default_deterministic_truth_jittered(self):
        """默认轨迹两次一致；真值参数相对默认有抖动（可学习失配源）。"""
        a = community_seir_window_forces(np.random.RandomState(7))
        b = community_seir_window_forces(np.random.RandomState(7))
        assert a['lam_w_default'] == b['lam_w_default']
        assert a['lam_ref_default'] == b['lam_ref_default']
        # 抖动存在：真值 β 偏离默认（42 号种子下实测非零）
        c = community_seir_window_forces(np.random.RandomState(42))
        assert c['beta_truth'] != COMBINED_SEIR_SPEC['beta']

    def test_window_mid_coverage(self):
        """轨迹 40 周，覆盖最远窗中点 36 周。"""
        weeks = COMBINED_SEIR_SPEC['n_steps'] * COMBINED_SEIR_SPEC['dt']
        assert weeks >= max(COMBINED_SEIR_SPEC['window_mid_weeks'])


# ==============================================================================
# DGP v1：结构 / 分解恒等式 / 信息不对称
# ==============================================================================

class TestCombinedNetworkDGP:

    def test_determinism(self):
        a = build_combined_network(n_contacts=120, random_state=42)
        b = build_combined_network(n_contacts=120, random_state=42)
        np.testing.assert_array_equal(a['labels'], b['labels'])
        np.testing.assert_array_equal(a['nu'], b['nu'])
        np.testing.assert_array_equal(a['lam'], b['lam'])

    def test_shapes_and_label_rate(self):
        net = build_combined_network(n_contacts=200, target_rate=0.25,
                                     random_state=42)
        n, M = 200, net['M']
        assert M == max(2, n // 5)
        assert len(net['labels']) == n
        assert net['exposure_by_type'].shape == (n, 5)
        assert net['exposure_by_window'].shape == (n, 5)
        assert 0.10 < net['labels'].mean() < 0.40

    def test_decomposition_identity(self):
        """ν = host·corr·mech 逐位成立（三机制乘性合一的恒等式）。"""
        net = build_combined_network(n_contacts=150, random_state=42)
        mults = np.array([nd['host_multiplier']
                          for nd in net['nodes'][net['M']:]])
        recomputed = mults * net['corr'] * net['mech']
        np.testing.assert_allclose(recomputed, net['nu'], rtol=1e-12)

    def test_exposure_decomposition_consistency(self):
        """Σ_t E_t == Σ_w E_w == Σ成员 intens·s（同一总量的两个分解）。"""
        net = build_combined_network(n_contacts=150, random_state=42)
        by_type = net['exposure_by_type'].sum(axis=1)
        by_win = net['exposure_by_window'].sum(axis=1)
        manual = np.array([sum(m[3] * m[4] for m in ms)
                           for ms in net['memberships']])
        np.testing.assert_allclose(by_type, by_win, rtol=1e-12)
        np.testing.assert_allclose(by_type, manual, rtol=1e-12)

    def test_typed_columns_match_manual(self):
        net = build_combined_network(n_contacts=150, random_state=3)
        for i in (0, 17, 63, 149):
            for k, (src, ctype, w, inten, s) in enumerate(
                    net['memberships'][i]):
                col = TYPE_EXPOSURE_NAMES.index('type_exposure_' + ctype)
                assert net['exposure_by_type'][i, col] > 0 or inten * s == 0

    def test_oracle_gap_exists(self):
        """真值 ν 显著高于物理基线 λ（三机制信息在 mech 里）。"""
        net = build_combined_network(n_contacts=300, random_state=42)
        gap = _auc(net['nu'], net['labels']) - _auc(net['lam'],
                                                    net['labels'])
        assert gap > 0.02

    def test_no_proxy_leak_individual_features(self):
        """个体基础特征不能重构三机制**构成载体**（v2 教训回归检验）。

        载体 = 时序画像 / 类型暴露份额 / 源传染性 s̄——三者 AUROC
        均 < 0.75（弱部分推断，非完美代理；typed v2 教训的"无空间"
        口径为 0.95+）。总量类信息（度数→成员数→mech 量级）与容量
        饱和带来的部分类型份额推断属部署现实（n_edges 可观测，对齐
        typed v3 的 cum 先例）；剩余空间的判据 = oracle 间隙存在
        （test_oracle_gap_exists，实测 > 0.13）。
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        net = build_combined_network(n_contacts=400, random_state=42)
        X, _ = feature_matrix(net, feature_set='ind')
        rng = np.random.RandomState(0)

        def _auc_cv(y_bin):
            y_bin = np.asarray(y_bin, dtype=int)
            if y_bin.sum() < 5 or (1 - y_bin).sum() < 5:
                return 0.5
            tr, te = [], []
            for cls in (0, 1):
                idx = np.where(y_bin == cls)[0]
                rng.shuffle(idx)
                half = len(idx) // 2
                tr.extend(idx[:half]); te.extend(idx[half:])
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000))
            clf.fit(X[tr], y_bin[tr])
            return _auc(clf.predict_proba(X[te])[:, 1], y_bin[te])

        profiles = np.array([1 if p == 'old' else 0
                             for p in net['profiles']])
        assert _auc_cv(profiles) < 0.75
        tot = net['exposure_by_type'].sum(axis=1)
        hh_share = np.where(tot > 0,
                            net['exposure_by_type'][:, 0]
                            / np.maximum(tot, 1e-9), 0.0)
        assert _auc_cv((hh_share > 0.5).astype(int)) < 0.75
        hi_s = (net['s_bar'] >= np.percentile(net['s_bar'], 75)).astype(int)
        assert _auc_cv(hi_s) < 0.75

    def test_pi_graph_compat(self):
        """组合网络与 PI 图转换兼容（GNN 分支参考臂前提）。"""
        torch = pytest.importorskip('torch')
        from tb_risk.validation.pi_network import to_pi_graph, physics_lambda
        net = build_combined_network(n_contacts=120, random_state=42)
        g = to_pi_graph(net)
        assert g['x'].shape == (net['M'] + 120, 14)
        g2 = to_pi_graph(net, include_physics=True, k_hat=1.0)
        assert g2['x'].shape[1] == 17
        assert g2['lam_tilde'].shape[0] == net['M'] + 120
        np.testing.assert_allclose(
            np.asarray(physics_lambda(net)), net['lam'], rtol=1e-12)
        assert g['edge_index'].shape[1] == 2 * len(net['ef'])


# ==============================================================================
# P1 特征化：矩阵 / 列名 / 维度
# ==============================================================================

class TestFeatureization:

    @pytest.mark.parametrize('fs', FEATURE_SETS)
    def test_matrix_shapes_and_names(self, fs):
        net = build_combined_network(n_contacts=100, random_state=42)
        X, names = feature_matrix(net, feature_set=fs,
                                  k_hat=2.0 if fs in ('pi', 'all')
                                  else None)
        assert X.shape == (100, feature_dim(fs))
        assert len(names) == X.shape[1]
        assert len(set(names)) == len(names)
        assert np.isfinite(X).all()

    def test_net_feature_names_registry(self):
        assert len(NET_FEATURE_NAMES) == 11
        assert NET_FEATURE_NAMES == (TYPE_EXPOSURE_NAMES
                                     + WINDOW_EXPOSURE_NAMES
                                     + [PHYSICS_LAMBDA_NAME])

    def test_pi_requires_k_hat(self):
        net = build_combined_network(n_contacts=60, random_state=42)
        with pytest.raises(ValueError):
            feature_matrix(net, feature_set='pi')

    def test_invalid_feature_set(self):
        net = build_combined_network(n_contacts=60, random_state=42)
        with pytest.raises(ValueError):
            feature_matrix(net, feature_set='wrong')

    def test_pi_column_is_calibrated_lambda(self):
        net = build_combined_network(n_contacts=100, random_state=42)
        X, names = feature_matrix(net, feature_set='pi', k_hat=3.5)
        col = names.index(PHYSICS_LAMBDA_NAME)
        np.testing.assert_allclose(X[:, col], 3.5 * net['lam'], rtol=1e-12)


# ==============================================================================
# 消融（冒烟）：冻结超参 + 阶梯 + 可加性
# ==============================================================================

class TestCombinedAblation:

    def test_frozen_model_params_from_registry(self):
        from tb_risk.validation.combined_ablation import _make_model
        rf = _make_model('random_forest', seed=42)
        p = rf.get_params()
        assert p['n_estimators'] == 300 and p['max_depth'] == 7
        assert p['class_weight'] == 'balanced'

    def test_once_smoke_no_gnn(self):
        from tb_risk.validation.combined_ablation import (
            run_combined_ablation_once)
        rep = run_combined_ablation_once(
            n_contacts=120, seed=42, skip_gnn=True,
            model_keys=('random_forest',))
        for arm in ('seir', 'ind:random_forest', 'typed:random_forest',
                    'window:random_forest', 'pi:random_forest',
                    'all:random_forest', 'oracle_nu'):
            a = rep['arms'][arm]
            assert 0.0 <= a['auroc'] <= 1.0
            assert 0.0 <= a['recall_at_budget'] <= 1.0
            assert np.isfinite(a['brier'])
        for key in ('random_forest:all_minus_ind',
                    'random_forest:synergy_all_minus_sum_singles',
                    'random_forest:realization_ratio'):
            assert key in rep['ladder']
        assert 'gnn_pi' not in rep['arms']

    def test_once_smoke_with_gnn(self):
        torch = pytest.importorskip('torch')
        from tb_risk.validation.combined_ablation import (
            run_combined_ablation_once)
        rep = run_combined_ablation_once(
            n_contacts=120, seed=42, gnn_epochs=60,
            model_keys=('random_forest',))
        assert 'gnn_pi' in rep['arms']
        assert 'random_forest:gnn_pi_minus_all' in rep['ladder']
        assert 'all_vs_gnn_pi[random_forest]' in rep['delong']
        assert torch is not None

    def test_once_deterministic(self):
        from tb_risk.validation.combined_ablation import (
            run_combined_ablation_once)
        a = run_combined_ablation_once(n_contacts=100, seed=7, skip_gnn=True,
                                       model_keys=('random_forest',))
        b = run_combined_ablation_once(n_contacts=100, seed=7, skip_gnn=True,
                                       model_keys=('random_forest',))
        assert a['arms']['all:random_forest']['auroc'] == \
            b['arms']['all:random_forest']['auroc']

    def test_multi_seed_smoke(self):
        from tb_risk.validation.combined_ablation import (
            run_multi_seed_combined_ablation)
        out = run_multi_seed_combined_ablation(
            n_contacts=100, n_seeds=2, seed_start=1, skip_gnn=True,
            model_keys=('random_forest',), n_bootstrap=50)
        assert len(out['seeds']) == 2
        assert out['additivity']['random_forest'][
            'additivity_verdict'] in (
            'super_additive（互补协同）',
            'sub_additive（特征冗余主导）',
            'additive_within_noise（可加）')
        assert 'all:random_forest' in out['arm_summary']
