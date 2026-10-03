#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据感染终点验证模块测试（P4b-2，2026-08-25）。

数据缺失时自动跳过（homeacf_tstsa.rda 不随仓库分发）。
"""

import os

import numpy as np
import pytest

from tb_risk.validation.real_data_infection import (
    LAMBDA_PRIORS_INF, FEATURE_ARMS, FEATURE_SETS, _RDA_PATH,
    _group_cv_indices, build_lambda_infection, feature_columns,
    load_homeacf_contacts, run_infection_ablation_once,
    run_multi_seed_infection,
)

_HAVE_DATA = os.path.isfile(_RDA_PATH)

needs_data = pytest.mark.skipif(
    not _HAVE_DATA, reason='HomeACF .rda 未下载')


class TestLambdaPhysicsInf:
    """λ 物理构造：冻结先验、可手算、方向性（含咳嗽时间维）。"""

    def test_priors_frozen_and_positive(self):
        assert set(LAMBDA_PRIORS_INF) == {
            'susceptibility_lt5', 'susceptibility_ge45', 'host_hiv_pos',
            'index_cough_slope', 'index_smear_pos', 'index_hiv_pos',
            'timespent_low', 'timespent_mid', 'timespent_high',
            'share_bedroom', 'sleep_same_bed'}
        assert all(v > 0 for v in LAMBDA_PRIORS_INF.values())

    def test_lambda_handcheck(self):
        """S(U 形+HIV) × I(exp 咳嗽×涂片) × C(分级×卧室×同床) 手算。"""
        import pandas as pd
        df = pd.DataFrame({
            'age_lt5': [1.0], 'age_ge45': [0.0], 'hiv_pos_h': [1.0],
            'idx_coughdays': [30.0], 'idx_smear_pos': [1.0],
            'idx_hiv_pos': [0.0],
            'ts_low': [0.0], 'ts_mid': [0.0], 'ts_high': [1.0],
            'share_bedroom': [1.0], 'sleep_same_bed': [0.0],
        })
        lam = build_lambda_infection(df)
        # S = (1+1.8)·1.3 = 3.64；I = exp(0.3)·3.0；C = 1.5·1.2
        expect = 3.64 * np.exp(0.30) * 3.0 * 1.5 * 1.2
        assert lam.iloc[0] == pytest.approx(expect)

    def test_lambda_monotone_in_coughdays_and_exposure(self):
        """咳嗽时长 ↑ 与接触强度 ↑ 均单调推高 λ（时间维先验）。"""
        import pandas as pd
        base = {
            'age_lt5': [0.0, 0.0], 'age_ge45': [0.0, 0.0],
            'hiv_pos_h': [0.0, 0.0],
            'idx_coughdays': [90.0, 10.0], 'idx_smear_pos': [0.0, 1.0],
            'idx_hiv_pos': [0.0, 0.0],
            'ts_low': [0.0, 1.0], 'ts_mid': [0.0, 0.0],
            'ts_high': [1.0, 0.0], 'share_bedroom': [1.0, 0.0],
            'sleep_same_bed': [1.0, 0.0],
        }
        lam = build_lambda_infection(pd.DataFrame(base))
        assert lam.iloc[0] > lam.iloc[1]  # 长咳+高暴露 > 短咳+低暴露


class TestFeatureContractInf:
    """特征臂契约：嵌套结构 + 终点列不入特征。"""

    def test_arms_nested(self):
        for arm in FEATURE_ARMS:
            cols = feature_columns(arm)
            assert len(cols) == len(set(cols))
        assert set(FEATURE_SETS['ind']) <= set(feature_columns('pi'))

    def test_outcome_columns_not_features(self):
        forbidden = {'tst_diam', 'tst_pos10', 'tst_pos5', 'tstdiam_h',
                     'tstdone_h', 'tstread_h'}
        for arm in FEATURE_ARMS:
            for col in FEATURE_SETS[arm]:
                assert col not in forbidden


@needs_data
class TestLoaderAndCVInf:
    """装载契约 + 户分组 CV。"""

    @pytest.fixture(scope='class')
    def df(self):
        return load_homeacf_contacts()

    def test_load_contract(self, df):
        assert len(df) == 2725          # tstdiam 非缺失完全病例
        assert int(df['tst_pos10'].sum()) == 359
        assert int(df['tst_pos5'].sum()) == 458
        assert df['record_id'].nunique() > 800  # ~924 户
        for arm in FEATURE_ARMS:
            assert not df[feature_columns(arm)].isna().any().any()

    def test_lambda_available_and_positive(self, df):
        assert (df['lam'] > 0).all()
        assert df['lam'].std() > 0

    def test_coughdays_filled_not_missing(self, df):
        """咳嗽时长缺失填中位 30（物理中性值）——不得留 NaN。"""
        assert df['idx_coughdays'].notna().all()
        assert df['idx_coughdays'].between(0, 365).all()

    def test_group_cv_disjoint_and_complete(self, df):
        y = df['tst_pos10'].astype(int).to_numpy()
        groups = df['record_id'].to_numpy()
        folds = _group_cv_indices(groups, y, n_splits=5, seed=0)
        seen = np.zeros(len(y), dtype=int)
        for tr, te in folds:
            assert set(groups[tr]).isdisjoint(set(groups[te]))
            seen[te] += 1
        assert (seen == 1).all()
        for _, te in folds:
            assert y[te].sum() >= 10  # 每折测试半区含足量事件


@needs_data
class TestInfectionAblation:
    """单种子 + 多种子契约与确定性。"""

    def test_once_deterministic(self):
        r1 = run_infection_ablation_once(seed=3)
        r2 = run_infection_ablation_once(seed=3)
        for k in r1['arms']:
            assert r1['arms'][k]['auroc'] == \
                pytest.approx(r2['arms'][k]['auroc'], abs=1e-12)

    def test_once_contract(self):
        r = run_infection_ablation_once(seed=0)
        assert set(r['design']) >= {'n', 'n_events', 'n_events_pos5',
                                    'n_households', 'n_splits', 'endpoint'}
        expected = (['pi_only']
                    + [f'{a}:{m}' for a in FEATURE_ARMS
                       for m in ('random_forest', 'lightgbm')])
        assert set(r['arms']) == set(expected)
        for name, a in r['arms'].items():
            assert np.isfinite(a['auroc'])
            assert np.isfinite(a['pr_auc'])
            assert np.isfinite(a['auroc_tst_pos5'])

    def test_multi_seed_smoke(self):
        out = run_multi_seed_infection(n_seeds=2, seed_start=7,
                                       n_bootstrap=100)
        assert out['design']['n_seeds'] == 2
        assert len(out['seeds']) == 2
        for key in ('pi:random_forest_minus_exposure:random_forest',
                    'pi:random_forest_minus_ind:random_forest',
                    'pi_only_minus_ind:random_forest'):
            s = out['ladder_summary'][key]
            assert set(s) >= {'mean', 'bootstrap_ci', 'ci_excludes_zero'}
            assert s['bootstrap_ci'][0] <= s['mean'] <= s['bootstrap_ci'][1]
        # 物理方向报告四分位层覆盖全部样本
        q = out['physics_direction']['lambda_quartile_tst_pos10']
        assert sum(v['n'] for v in q.values()) == out['design']['n']
