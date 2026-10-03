#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序留出家庭传播模型测试（P1，2026-08-25）。"""

import os

import numpy as np
import pandas as pd
import pytest

from tb_risk.validation.real_data_infection import _RDA_PATH
from tb_risk.validation.household_temporal import (
    ORDER_MODES, assign_screening_order, prior_features,
    run_multi_seed_temporal, run_temporal_household_once,
)

_HAVE_DATA = os.path.isfile(_RDA_PATH)
needs_data = pytest.mark.skipif(not _HAVE_DATA,
                                reason='HomeACF .rda 未下载')


def _toy_df():
    """3 户 7 人：标签模式户内相关（h1 聚集、h2 混合、h3 独居）。"""
    return pd.DataFrame({
        'record_id': ['h1'] * 3 + ['h2'] * 3 + ['h3'],
        'tst_pos10': [1, 1, 0, 1, 0, 0, 1],
        'x': np.arange(7, dtype=float),
    })


class TestScreeningOrder:
    """户内筛查顺序协议。"""

    def test_random_deterministic_and_complete(self):
        df = _toy_df()
        p1 = assign_screening_order(df, seed=5)
        p2 = assign_screening_order(df, seed=5)
        assert p1.equals(p2)
        for _, sub in df.groupby('record_id'):
            assert sorted(p1.loc[sub.index]) == list(range(len(sub)))

    def test_oracle_puts_positive_first(self):
        df = _toy_df()
        pos = assign_screening_order(df, seed=0, mode='oracle')
        for _, sub in df.groupby('record_id'):
            if len(sub) < 2:
                continue
            first = sub.index[0]
            assert pos.loc[first] == 0 or sub['tst_pos10'].iloc[
                int(pos.loc[first])] == 1
            # 户内任一阳性者的位置 ≤ 任一阴性者（阳性优先）
            pp = pos.loc[sub.index[sub['tst_pos10'] == 1]]
            pn = pos.loc[sub.index[sub['tst_pos10'] == 0]]
            if len(pp) and len(pn):
                assert pp.max() <= pn.min()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            assign_screening_order(_toy_df(), mode='bogus')

    def test_modes_cover(self):
        assert ORDER_MODES == ('random', 'oracle', 'anti')


class TestPriorFeatures:
    """时序先证特征：手算 + 防泄漏构造断言。"""

    def test_handcheck(self):
        """户 h1 三人按位置 0/1/2：prior_n=0/1/2，prior_pos 累计。"""
        df = _toy_df()
        order = pd.Series([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
                          index=df.index)  # h1: [1,1,0] h2: [1,0,0]
        pf = prior_features(df, order)
        base = df['tst_pos10'].mean()
        # h1 位置 0：无先证 → 全局率
        assert pf['prior_n'].iloc[0] == 0
        assert pf['prior_rate'].iloc[0] == pytest.approx(base)
        # h1 位置 1：先证 1 人阳性 → rate=1
        assert pf['prior_n'].iloc[1] == 1
        assert pf['prior_pos'].iloc[1] == 1
        assert pf['prior_rate'].iloc[1] == pytest.approx(1.0)
        # h1 位置 2：先证 2 人（均阳性）→ rate=1.0
        assert pf['prior_n'].iloc[2] == 2
        assert pf['prior_pos'].iloc[2] == 2
        assert pf['prior_rate'].iloc[2] == pytest.approx(1.0)
        # h3 独居：prior_n=0 → 全局率
        assert pf['prior_n'].iloc[6] == 0
        assert pf['prior_rate'].iloc[6] == pytest.approx(base)

    def test_no_future_leak(self):
        """翻转位置 k 的标签：位置 < k 者的 prior 不变；位置 > k 者变。"""
        df = _toy_df()
        order = pd.Series([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
                          index=df.index)
        pf0 = prior_features(df, order)
        df2 = df.copy()
        df2.loc[df2.index[1], 'tst_pos10'] = 0  # h1 位置 1 翻转
        pf1 = prior_features(df2, order)
        # 位置 0（未来者标签不影响先证）
        assert pf1['prior_pos'].iloc[0] == pf0['prior_pos'].iloc[0]
        assert pf1['prior_n'].iloc[0] == pf0['prior_n'].iloc[0]
        # 位置 2（其先证含位置 1）改变
        assert pf1['prior_pos'].iloc[2] != pf0['prior_pos'].iloc[2]
        # 其他户（h2/h3）完全不受影响
        assert pf1['prior_pos'].iloc[3:].equals(pf0['prior_pos'].iloc[3:])

    def test_prior_pos_bounded_by_prior_n(self):
        df = _toy_df()
        order = assign_screening_order(df, seed=1)
        pf = prior_features(df, order)
        assert (pf['prior_pos'] <= pf['prior_n']).all()
        assert pf['prior_screened'].equals((pf['prior_n'] > 0).astype(float))


@needs_data
class TestTemporalAblation:
    """单/多种子契约 + 上界锚定。"""

    def test_once_deterministic(self):
        r1 = run_temporal_household_once(seed=3)
        r2 = run_temporal_household_once(seed=3)
        for k in r1['arms']:
            assert r1['arms'][k]['auroc'] == pytest.approx(
                r2['arms'][k]['auroc'], abs=1e-12)

    def test_arm_contract(self):
        r = run_temporal_household_once(seed=0)
        expected = {'ind:RF', 'exposure:RF', 'prior_only:RF',
                    'exposure_prior:RF', 'hh_loo_only'}
        assert set(r['arms']) == expected
        for name, a in r['arms'].items():
            assert np.isfinite(a['auroc']), name
        # 上界锚定：loo ≥ 时序版（信息量单调）
        assert r['arms']['hh_loo_only']['auroc'] >= \
            r['arms']['exposure_prior:RF']['auroc'] - 0.01

    def test_order_envelope(self):
        """顺序包络（3 种子平均）：prior_only 纯先证信号单调
        oracle > random > anti；exp_prior 主臂 oracle 不差于 random。
        （exp_prior 上 anti≈random：先证信号对顺序方向不极端敏感，
        敏感性证据本身入结果 JSON。）"""
        acc = {m: [] for m in ORDER_MODES}
        acc_exp = {m: [] for m in ORDER_MODES}
        for s in (2, 3, 4):
            for m in ORDER_MODES:
                r = run_temporal_household_once(seed=s, order_mode=m)
                acc[m].append(r['arms']['prior_only:RF']['auroc'])
                acc_exp[m].append(
                    r['arms']['exposure_prior:RF']['auroc'])
        assert np.mean(acc['oracle']) > np.mean(acc['random'])
        assert np.mean(acc['random']) > np.mean(acc['anti'])
        assert np.mean(acc_exp['oracle']) >= np.mean(acc_exp['random'])

    def test_multi_seed_smoke(self):
        out = run_multi_seed_temporal(n_seeds=2, seed_start=11,
                                      n_bootstrap=100, sensitivity_seeds=1)
        assert out['design']['n_seeds'] == 2
        assert len(out['seeds']) == 2
        assert set(out['order_sensitivity']) == {'oracle', 'anti'}
        for k, s in out['ladder_summary'].items():
            assert s['bootstrap_ci'][0] <= s['mean'] <= \
                s['bootstrap_ci'][1]
        assert set(out['design']['acceptance']) == {
            'vs_ind_baseline', 'vs_exposure_baseline'}
