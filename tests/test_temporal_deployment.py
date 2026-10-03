#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序先验部署口径决策分析测试（P2，2026-08-25）。"""

import numpy as np
import pandas as pd
import pytest

from tb_risk.scoring.temporal_household import (
    sequential_screening_trace,
    temporal_risk_update,
)
from tb_risk.validation.temporal_deployment import (
    _budget_recall,
    calibrate_weight_shrinkage,
    screening_workpoint,
    sequential_replay_fast,
    temporal_holdout_scores,
    run_deployment_analysis_once,
    run_multi_seed_deployment,
)

_IND_COLS = [
    'contact_age', 'age_lt5', 'age_ge45', 'contact_sex_m',
    'hiv_pos_h', 'hiv_unknown_h', 'bmi_h', 'smoke_ever_h',
    'diabetes_h_f', 'site_capricorn', 'hh_n_contacts',
]


def _toy_households():
    """3 户 10 人：分数与标签户内相关（手算可控）。"""
    base = [0.30, 0.25, 0.20, 0.15,  # hh_a：基线偏高
            0.55, 0.45, 0.10,        # hh_b：混杂
            0.08, 0.06, 0.05]       # hh_c：基线偏低
    res = [1, 1, 0, 0,
           1, 0, 0,
           0, 0, 1]  # hh_c 一例阳性（顺序敏感性用）
    hh = ['hh_a'] * 4 + ['hh_b'] * 3 + ['hh_c'] * 3
    return base, res, hh


def _synth_df(n_hh=8, size=4, seed=0):
    """合成 HomeACF 同构 df（ind 特征 + 户聚集标签）。"""
    rng = np.random.RandomState(seed)
    rows = []
    for h in range(n_hh):
        hh_rate = 0.15 + 0.7 * rng.rand()          # 户内聚集率
        for _ in range(size):
            age = rng.randint(1, 70)
            rows.append({
                'record_id': f'hh_{h}',
                'tst_pos10': int(rng.rand() < hh_rate),
                'contact_age': float(age),
                'age_lt5': float(age < 5),
                'age_ge45': float(age >= 45),
                'contact_sex_m': float(rng.rand() < 0.5),
                'hiv_pos_h': float(rng.rand() < 0.1),
                'hiv_unknown_h': float(rng.rand() < 0.2),
                'bmi_h': 18.0 + 6.0 * rng.rand(),
                'smoke_ever_h': float(rng.rand() < 0.3),
                'diabetes_h_f': float(rng.rand() < 0.1),
                'site_capricorn': float(h % 2),
                'hh_n_contacts': float(size),
            })
    return pd.DataFrame(rows)


class TestSequentialReplayFast:
    """向量化序贯重放：与参考实现同语义 + 停止阈值。"""

    def test_parity_with_reference_highest_risk(self):
        """highest_risk：与 sequential_screening_trace 逐步一致。"""
        base, res, hh = _toy_households()
        fast = sequential_replay_fast(base, res, hh, base_rate=0.3,
                                      weight=7.0, shrinkage=2.0,
                                      first_policy='highest_risk', seed=3)
        ref = sequential_screening_trace(
            [b * 100.0 for b in base], res, hh, base_rate=0.3,
            weight=7.0, shrinkage=2.0, first_policy='highest_risk', seed=3)
        assert fast['screen_order'] == ref['screen_order']
        assert fast['detection_curve'] == ref['detection_curve']
        # 更新前风险同尺度换算一致（0-1 vs 0-100）
        for a, b in zip(fast['risk_at_screen'], ref['steps']):
            assert a == pytest.approx(b['risk_before'] / 100.0, abs=1e-9)

    def test_parity_with_reference_random(self):
        """random 首筛：同种子下与参考实现一致。"""
        base, res, hh = _toy_households()
        fast = sequential_replay_fast(base, res, hh, base_rate=0.3,
                                      weight=9.0, shrinkage=2.0,
                                      first_policy='random', seed=11)
        ref = sequential_screening_trace(
            [b * 100.0 for b in base], res, hh, base_rate=0.3,
            weight=9.0, shrinkage=2.0, first_policy='random', seed=11)
        assert fast['screen_order'] == ref['screen_order']
        assert fast['detection_curve'] == ref['detection_curve']

    def test_default_single_household(self):
        """households=None → 全部同一户。"""
        base, res, _ = _toy_households()
        out = sequential_replay_fast(base, res, None, base_rate=0.3,
                                     weight=7.0, shrinkage=2.0)
        assert out['n_households'] == 1
        assert out['n_screens'] == len(res)
        assert out['n_positive_found'] == sum(res)

    def test_validation_errors(self):
        base, res, hh = _toy_households()
        with pytest.raises(ValueError):
            sequential_replay_fast(base[:-1], res, hh)
        with pytest.raises(ValueError):
            sequential_replay_fast(base, res, hh[:5])
        with pytest.raises(ValueError):
            sequential_replay_fast(base, res, hh, first_policy='bogus')

    def test_stop_threshold(self):
        """前瞻停止：阈值高 → 提前停；None → 全量。"""
        base, res, hh = _toy_households()
        full = sequential_replay_fast(base, res, hh, base_rate=0.3,
                                      weight=7.0, shrinkage=2.0)
        assert full['n_screens'] == len(res)
        assert not full['stopped_by_threshold']
        # 首筛者（基线最高 0.55）更新前分数 = 0.55 → 阈值略高即停
        stopped = sequential_replay_fast(base, res, hh, base_rate=0.3,
                                         weight=7.0, shrinkage=2.0,
                                         stop_threshold=0.56)
        assert stopped['stopped_by_threshold']
        assert stopped['n_screens'] == 0


class TestTemporalHoldoutScores:
    """时序留出分数（固定顺序口径）。"""

    def test_first_screener_falls_back_to_base(self):
        """prior_n=0 → 分数 = 个体基线（首筛者回退）。"""
        base = [0.2, 0.5]
        res = [1, 1]
        hh = ['h'] * 2
        s = temporal_holdout_scores(base, res, hh, [0.0, 1.0],
                                     base_rate=0.3, weight=9.0,
                                     shrinkage=2.0)
        assert s[0] == pytest.approx(0.2, abs=1e-9)

    def test_positive_prior_raises_negative_lowers(self):
        """同户先证阳性 → 上调；全阴 → 下调。"""
        pi = 0.3
        s_pos = temporal_holdout_scores([0.3, 0.3], [1, 1], ['h'] * 2,
                                        [0.0, 1.0], pi, 9.0, 2.0)
        s_neg = temporal_holdout_scores([0.3, 0.3], [0, 0], ['h'] * 2,
                                         [0.0, 1.0], pi, 9.0, 2.0)
        assert s_pos[1] > 0.3
        assert s_neg[1] < 0.3

    def test_parity_with_temporal_risk_update(self):
        """与评分服务逐点一致（0-1 vs 0-100 尺度换算）。"""
        pi = 0.25
        base = [0.4, 0.2]
        res = [1, 0]
        s = temporal_holdout_scores(base, res, ['h'] * 2, [0.0, 1.0],
                                    pi, w := 11.0, 3.0)
        ref = temporal_risk_update(20.0, [1], pi, weight=w, shrinkage=3.0)
        assert s[1] == pytest.approx(ref / 100.0, abs=1e-9)

    def test_zero_weight_returns_base(self):
        base = [0.3, 0.5, 0.7]
        s = temporal_holdout_scores(base, [1, 0, 1], ['h'] * 3,
                                    [0.0, 1.0, 2.0], 0.3, 0.0, 2.0)
        assert s == pytest.approx(base, abs=1e-9)


class TestScreeningWorkpoint:
    """目标检出率工作点（停止口径 A）。"""

    def test_perfect_order(self):
        """完美排序：80% 检出需 ceil(0.8×n_pos) 步。"""
        curve = [1, 2, 3, 3, 3]  # 3 阳性，前 3 步全中
        wp = screening_workpoint(curve, n_pos=3, sensitivity_target=0.8)
        assert wp['n_screens'] == 3
        assert wp['tp'] == 3
        assert wp['reached']
        assert wp['nns'] == pytest.approx(1.0)

    def test_unreached_falls_to_full(self):
        curve = [0, 0, 0, 0]
        wp = screening_workpoint(curve, n_pos=2, sensitivity_target=0.8)
        assert not wp['reached']
        assert wp['n_screens'] == 4
        assert wp['sensitivity'] == 0.0
        assert wp['nns'] == float('inf')

    def test_no_positive_raises(self):
        with pytest.raises(ValueError):
            screening_workpoint([0, 0], n_pos=0)


class TestCalibrateWeightShrinkage:
    """(w, k) 网格校准（合成数据 + 轻量路径）。"""

    def test_grid_contract(self):
        df = _synth_df(n_hh=6, size=4, seed=1)
        w_grid = (0.0, 5.0)
        k_grid = (1.0, 4.0)
        cal = calibrate_weight_shrinkage(df=df, seed=0, n_splits=3,
                                         w_grid=w_grid, k_grid=k_grid,
                                         rf_reference=False)
        assert cal['best_w'] in w_grid
        assert cal['best_k'] in k_grid
        assert len(cal['grid']) == len(w_grid) * len(k_grid)
        assert cal['rf_reference_auroc'] is None
        best = max(g['auroc'] for g in cal['grid'])
        assert cal['best_auroc'] == pytest.approx(best)

    def test_w0_equals_base(self):
        """w=0 → 时序分数退化为基线（AUROC 不变）。"""
        df = _synth_df(n_hh=6, size=4, seed=2)
        cal = calibrate_weight_shrinkage(df=df, seed=0, n_splits=3,
                                         w_grid=(0.0, 5.0),
                                         k_grid=(2.0,), rf_reference=False)
        w0 = [g for g in cal['grid'] if g['w'] == 0.0][0]
        assert w0['auroc'] == pytest.approx(cal['base_auroc'], abs=1e-9)


class TestDeploymentAnalysis:
    """单/多种子部署分析（合成 df，无 rda 依赖）。"""

    @pytest.fixture(scope='class')
    def df(self):
        return _synth_df(n_hh=8, size=4, seed=3)

    def test_once_contract(self, df):
        rep = run_deployment_analysis_once(seed=0, df=df, weight=7.0,
                                           shrinkage=2.0, n_splits=3,
                                           sensitivity_target=0.8)
        assert set(rep['auroc']) == {'static', 'temporal_random'}
        assert set(rep['order_envelope']) == {'oracle', 'anti'}
        assert set(rep['sequential']) == {'random', 'highest_risk'}
        for policy in ('random', 'highest_risk'):
            node = rep['sequential'][policy]
            assert set(node['target']) == {
                'n_screens', 'tp', 'sensitivity', 'nns', 'reached'}
            assert set(node['prospect']) == {
                'n_screens', 'tp', 'sensitivity', 'nns',
                'stopped_by_threshold'}
        # 序贯与静态同锚（阈值来自静态工作点）
        assert rep['sequential']['random']['prospect']['n_screens'] \
            <= len(rep['y'])
        # 口径 A：序贯不超过全量
        assert rep['static']['target']['n_screens'] <= len(rep['y'])

    def test_once_deterministic(self, df):
        r1 = run_deployment_analysis_once(seed=5, df=df, weight=7.0,
                                          shrinkage=2.0, n_splits=3)
        r2 = run_deployment_analysis_once(seed=5, df=df, weight=7.0,
                                          shrinkage=2.0, n_splits=3)
        assert r1['auroc'] == r2['auroc']
        assert (r1['oof']['p_base'] == r2['oof']['p_base']).all()
        assert r1['sequential']['random']['target'] == \
            r2['sequential']['random']['target']

    def test_multi_seed_smoke(self, df):
        out = run_multi_seed_deployment(n_seeds=2, seed_start=0,
                                        n_splits=3, df=df,
                                        weight=7.0, shrinkage=2.0,
                                        calibrate=False, n_bootstrap=100)
        assert out['design']['n_seeds'] == 2
        assert len(out['seeds']) == 2
        assert out['design']['calibrated'] is False
        for key in ('static', 'seq_random', 'seq_highest_risk'):
            assert key in out['target_summary']
            assert key in out['prospect_summary']
        assert 'temporal_vs_static_bootstrap' in out['auroc_summary']
        assert len(out['budget_recall_curve']['budgets']) == 21
        # static 曲线 = 完整排序 → 100% 预算召回 1.0；
        # seq 曲线受口径 B 前瞻阈值提前终止 → 末点召回 ≤ 1（合法）
        for curve in ('static', 'seq_random', 'seq_highest_risk'):
            vals = out['budget_recall_curve'][curve]
            assert len(vals) == 21
            assert vals == sorted(vals)  # 检出曲线单调
            assert 0.0 <= vals[-1] <= 1.0
        assert out['budget_recall_curve']['static'][-1] == \
            pytest.approx(1.0)
        # 验收结构
        acc = out['design']['acceptance']
        assert 'temporal_vs_static_auroc' in acc
        assert 'seq_screens_below_static_win_rate' in acc
        assert 'first_hit_highest_above_base' in acc

    def test_multi_seed_calibration_path(self, df):
        """calibrate=True + 小网格：w/k 来自校准并写入 design。"""
        out = run_multi_seed_deployment(
            n_seeds=1, seed_start=0, n_splits=3, df=df,
            w_grid=(0.0, 7.0), k_grid=(2.0,), calibrate=True,
            n_bootstrap=50, rf_reference=False)
        assert out['design']['calibrated'] is True
        assert out['calibration']['best_w'] in (0.0, 7.0)
        assert out['design']['weight'] == out['calibration']['best_w']


class TestBudgetRecall:
    """预算-召回采样。"""

    def test_sampling_monotone(self):
        y = np.array([1, 0, 1, 0] * 5)          # 10 阳性
        curve = list(range(1, 11)) + [10] * 10  # 完美检出：前 10 步全中
        vals = _budget_recall(curve, y)
        assert vals[0] == 0.0
        assert vals[-1] == pytest.approx(1.0)
        assert vals == sorted(vals)

    def test_budget_zero_is_zero(self):
        y = np.array([1, 1, 0])
        vals = _budget_recall([1, 2, 2], y)
        assert vals[0] == 0.0
        assert vals[-1] == pytest.approx(1.0)
