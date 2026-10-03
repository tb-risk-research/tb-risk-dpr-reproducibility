#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序家庭先验评分服务测试（第四轮 P1a/P1b）。

覆盖：
  1. 先证特征四列（与 household_temporal 实验同构）；
  2. 贝叶斯收缩户级率（n=0 → 基线；n→∞ → 经验率）；
  3. logit 风险更新（首筛查者回退 / 阳性上调 / 全阴下调 / 单调性）；
  4. 序贯筛查状态机（无未来泄漏：已筛成员冻结，证据只增）；
  5. 离线重放 trace（首筛查者策略 / 检出曲线）；
  6. 三层架构第 2 层接线（screened_results 优先于 GNN 残差路径）。
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)

from tb_risk.scoring.temporal_household import (  # noqa: E402
    DEFAULT_TEMPORAL_SHRINKAGE, DEFAULT_TEMPORAL_WEIGHT,
    FIRST_SCREENER_POLICIES, HouseholdScreeningState,
    PRIOR_FEATURE_NAMES, household_rate_estimate,
    sequential_screening_trace, temporal_increment,
    temporal_prior_features, temporal_risk_update,
)
from tb_risk.scoring.architecture import (  # noqa: E402
    ThreeLayerArchitecture, compute_network_increment,
)


# ==============================================================================
# 1. 先证特征
# ==============================================================================

class TestPriorFeatures:

    def test_feature_names_contract(self):
        assert PRIOR_FEATURE_NAMES == (
            'prior_n', 'prior_pos', 'prior_rate', 'prior_screened')

    def test_empty_screened_falls_back_to_base_rate(self):
        """首筛查者：n=0 → rate 回退基线、screened=0。"""
        f = temporal_prior_features([], base_rate=0.13)
        assert f['prior_n'] == 0 and f['prior_pos'] == 0
        assert f['prior_rate'] == 0.13
        assert f['prior_screened'] == 0.0

    def test_counts_and_rate(self):
        f = temporal_prior_features([1, 0, 1], base_rate=0.13)
        assert f['prior_n'] == 3 and f['prior_pos'] == 2
        assert f['prior_rate'] == pytest.approx(2 / 3)
        assert f['prior_screened'] == 1.0

    def test_boolean_inputs_coerced(self):
        f = temporal_prior_features([True, False, True], base_rate=0.1)
        assert f['prior_pos'] == 2


# ==============================================================================
# 2. 贝叶斯收缩率
# ==============================================================================

class TestHouseholdRateEstimate:

    def test_no_evidence_equals_base(self):
        assert household_rate_estimate(
            [], 0.2) == pytest.approx(0.2)

    def test_shrinkage_interpolates(self):
        r = household_rate_estimate([1], 0.2, shrinkage=2.0)
        # (1 + 2×0.2)/(1+2) = 1.4/3
        assert r == pytest.approx(1.4 / 3.0)

    def test_more_evidence_moves_to_empirical(self):
        r1 = household_rate_estimate([1], 0.2, shrinkage=2.0)
        r5 = household_rate_estimate([1] * 5, 0.2, shrinkage=2.0)
        assert r5 > r1
        assert r5 == pytest.approx((5 + 0.4) / 7.0)

    def test_negative_shrinkage_raises(self):
        with pytest.raises(ValueError):
            household_rate_estimate([1], 0.2, shrinkage=-1.0)


# ==============================================================================
# 3. logit 风险更新
# ==============================================================================

class TestTemporalRiskUpdate:

    def test_first_screener_returns_base_unchanged(self):
        """用户规格：首筛查者回退个体基线。"""
        for p in (0.0, 5.0, 30.0, 80.0, 100.0):
            assert temporal_risk_update(
                p, [], 0.13) == pytest.approx(p)

    def test_positive_household_raises_risk(self):
        p0, p1 = 30.0, temporal_risk_update(30.0, [1, 1], 0.13)
        assert p1 > p0

    def test_all_negative_household_lowers_risk(self):
        p0, p1 = 30.0, temporal_risk_update(30.0, [0, 0, 0], 0.13)
        assert p1 < p0

    def test_more_positives_stronger_uplift(self):
        p1 = temporal_risk_update(30.0, [1], 0.13)
        p2 = temporal_risk_update(30.0, [1, 1], 0.13)
        p3 = temporal_risk_update(30.0, [1, 1, 1], 0.13)
        assert p3 > p2 > p1

    def test_output_clipped_to_0_100(self):
        assert temporal_risk_update(100.0, [1, 1, 1], 0.13) <= 100.0
        assert temporal_risk_update(0.0, [0, 0, 0], 0.13) >= 0.0

    def test_increment_signs(self):
        assert temporal_increment(30.0, [], 0.13) == 0.0
        assert temporal_increment(30.0, [1], 0.13) > 0.0
        assert temporal_increment(30.0, [0, 0], 0.13) < 0.0

    def test_zero_weight_is_identity(self):
        """w=0 → 无更新（校准下界锚定）。"""
        assert temporal_risk_update(
            30.0, [1, 1], 0.13, weight=0.0) == pytest.approx(30.0)

    def test_order_preserved_within_household(self):
        """同户统一位移 → 户内排序保持基线序（诚实边界，防回归）。"""
        bases = [10.0, 40.0, 70.0]
        updated = [temporal_risk_update(b, [1], 0.13) for b in bases]
        assert updated == sorted(updated)

    def test_invalid_base_rate_raises(self):
        with pytest.raises(ValueError):
            temporal_risk_update(30.0, [1], 1.5)


# ==============================================================================
# 4. 序贯筛查状态机
# ==============================================================================

class TestHouseholdScreeningState:

    def test_state_flow_and_rerank(self):
        state = HouseholdScreeningState(
            base_scores=[20.0, 50.0, 35.0],
            names=['甲', '乙', '丙'], base_rate=0.13)
        # 首位 = 当前（=基线）风险最高的成员
        assert state.next_to_screen() == 1
        rec = state.record_result(1, 1)   # 乙筛查阳性
        assert rec['result'] == 1
        # 阳性后：其余成员风险上调
        scores = state.current_scores()
        assert scores[0] > 20.0 and scores[2] > 35.0
        # 已筛成员冻结为基线
        assert scores[1] == pytest.approx(50.0)
        assert state.screened_results() == [1]

    def test_double_screening_raises(self):
        state = HouseholdScreeningState([50.0], base_rate=0.1)
        state.record_result(0, 0)
        with pytest.raises(ValueError):
            state.record_result(0, 1)

    def test_next_returns_none_when_done(self):
        state = HouseholdScreeningState([10.0, 20.0], base_rate=0.1)
        state.record_result(0, 0)
        state.record_result(1, 1)
        assert state.next_to_screen() is None

    def test_reset_restores_first_screener_fallback(self):
        state = HouseholdScreeningState([10.0, 20.0], base_rate=0.1)
        state.record_result(0, 1)
        state.reset()
        assert state.screened_results() == []
        assert state.current_scores() == [10.0, 20.0]

    def test_empty_household_raises(self):
        with pytest.raises(ValueError):
            HouseholdScreeningState([], base_rate=0.1)

    def test_names_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            HouseholdScreeningState([10.0], names=['甲', '乙'],
                                    base_rate=0.1)

    def test_no_future_leak_in_evidence(self):
        """证据只含已登记结果：翻转未筛成员"未来标签"不影响当前风险。"""
        state = HouseholdScreeningState(
            [20.0, 50.0, 35.0], base_rate=0.13)
        state.record_result(1, 1)
        s_before = state.current_scores()
        # 成员 0/2 尚未筛查——其"真实标签"如何都不改变证据
        state.results[0] = 1   # 恶意注入未筛成员结果（不设 flag）
        assert state.current_scores() == s_before


# ==============================================================================
# 5. 离线重放
# ==============================================================================

class TestSequentialTrace:

    def test_trace_completes_and_finds_all(self):
        base = [60.0, 20.0, 40.0, 10.0]
        labels = [1, 0, 1, 0]
        t = sequential_screening_trace(base, labels, base_rate=0.5)
        assert t['n_screens'] == 4
        assert t['n_positive_found'] == 2
        assert t['detection_curve'][-1] == 2
        # 每步筛查的是当时风险最高的未筛成员（基线序：0→2→1→3）
        assert t['screen_order'] == [0, 2, 1, 3]

    def test_cross_household_rerank_accelerates_detection(self):
        """阳性户成员整体上移（户间重排）→ 检出早于静态基线序。"""
        base = [60.0, 42.0, 45.0, 44.0]
        labels = [1, 1, 0, 0]
        hh = ['A', 'A', 'B', 'B']
        t = sequential_screening_trace(base, labels, households=hh,
                                       base_rate=0.5)
        # 静态基线序 60,45,44,42 → 第二例阳性要到第 4 步才检出
        static_curve = [1, 1, 1, 2]
        # 序贯：第 1 步 A 户 60（阳性）→ A 户 42 上调越过 B 户 45/44
        # → 第 2 步即检出第二例阳性
        assert t['steps'][1]['screened_idx'] == 1
        assert t['detection_curve'][1] == 2 > static_curve[1]
        assert t['n_households'] == 2

    def test_evidence_never_leaks_own_result(self):
        """已筛成员冻结为基线——自身结果不进入自身风险。"""
        base = [50.0, 20.0]
        labels = [1, 1]
        t = sequential_screening_trace(base, labels, base_rate=0.5)
        # 成员 0 第 1 步被筛（基线最高），其"风险"即基线本身
        assert t['steps'][0]['risk_before'] == pytest.approx(50.0)

    def test_first_policy_random_is_seeded(self):
        base = [10.0, 20.0, 30.0]
        labels = [0, 0, 0]
        t1 = sequential_screening_trace(base, labels, base_rate=0.5,
                                        first_policy='random', seed=3)
        t2 = sequential_screening_trace(base, labels, base_rate=0.5,
                                         first_policy='random', seed=3)
        assert t1['screen_order'] == t2['screen_order']

    def test_first_policy_highest_risk_picks_max(self):
        t = sequential_screening_trace(
            [10.0, 20.0, 30.0], [0, 0, 0], base_rate=0.5)
        assert t['steps'][0]['screened_idx'] == 2

    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError):
            sequential_screening_trace([10.0], [0], base_rate=0.5,
                                       first_policy='oracle')

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            sequential_screening_trace([10.0, 20.0], [0], base_rate=0.5)


# ==============================================================================
# 6. 三层架构第 2 层接线（P1b）
# ==============================================================================

class TestArchitectureWiring:

    def test_compute_network_increment_temporal_priority(self):
        """screened_results 提供时 → 第 2 层走时序先证路径。"""
        out = compute_network_increment(
            ml_predictor=None, tb_assessment=None, contact_data={},
            contact_type='family', p_base=30.0,
            screened_results=[1, 1], household_base_rate=0.13)
        assert out['backend'] == 'temporal_household'
        assert out['network_increment'] > 0.0
        assert out['network_aware'] is True
        assert out['residual_learning'] is True
        pf = out['prior_features']
        assert pf['prior_n'] == 2.0 and pf['prior_pos'] == 2.0

    def test_temporal_first_screener_no_increment(self):
        out = compute_network_increment(
            ml_predictor=None, tb_assessment=None, contact_data={},
            contact_type='family', p_base=30.0,
            screened_results=[], household_base_rate=0.13)
        assert out['network_increment'] == 0.0
        assert out['network_aware'] is False

    def test_temporal_requires_base_rate(self):
        with pytest.raises(ValueError):
            compute_network_increment(
                ml_predictor=None, tb_assessment=None, contact_data={},
                contact_type='family', p_base=30.0,
                screened_results=[1], household_base_rate=None)

    def test_gnn_path_unchanged_without_screened_results(self):
        """无筛查状态 → 原 GNN/NumPy 残差路径（回归保护）。"""
        out = compute_network_increment(
            ml_predictor=None, tb_assessment=None, contact_data={},
            contact_type='family', p_base=30.0)
        assert 'backend' in out
        assert 'network_increment' in out

    def test_pipeline_passes_screened_results(self):
        """run_pipeline 全流水线透传筛查状态到第 2 层。"""
        arch = ThreeLayerArchitecture()
        report = arch.run_pipeline(
            ml_predictor=None, tb_assessment=None,
            contact_data={'record_id': 'c1'}, contact_type='family',
            run_intervention=False,
            screened_results=[1], household_base_rate=0.13)
        l2 = report['layers']['layer2_network']
        assert l2['backend'] == 'temporal_household'
        assert report['summary']['network_increment'] > 0.0

    def test_gated_integration_temporal_full_gate(self):
        """时序路径门控系数为 1（证据收缩已内建，不再按接触数衰减）。"""
        arch = ThreeLayerArchitecture()
        report = arch.run_pipeline(
            ml_predictor=None, tb_assessment=None,
            contact_data={}, contact_type='family',
            run_intervention=False,
            screened_results=[1], household_base_rate=0.13)
        assert report['decision']['gating'] == 1.0
        # 门控不衰减增量（区别于 GNN 残差的接触数门控）
        assert report['decision']['gated_network_increment'] == \
            pytest.approx(report['summary']['network_increment'])
