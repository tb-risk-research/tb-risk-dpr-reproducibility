#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据 PI 验证模块测试（P4b，2026-08-25）。

数据缺失时自动跳过（原始 CSV 不随仓库分发）。
"""

import os

import numpy as np
import pytest

from tb_risk.validation.real_data_pi import (
    LAMBDA_PRIORS, FEATURE_ARMS, FEATURE_SETS, _DATA_DIR,
    _OUTCOME_COLS, _group_cv_indices, build_lambda_real,
    feature_columns, load_pacts_contacts,
    run_real_data_ablation_once, run_multi_seed_real_data,
)

_HAVE_DATA = os.path.isfile(
    os.path.join(_DATA_DIR, 'contacts_threemonths_rev.csv'))

needs_data = pytest.mark.skipif(
    not _HAVE_DATA, reason='PACTS 原始数据未下载')


class TestLambdaPhysics:
    """λ 物理构造：冻结先验、可手算、方向性。"""

    def test_priors_frozen_and_positive(self):
        assert set(LAMBDA_PRIORS) == {
            'susceptibility_lt5', 'susceptibility_ge45',
            'index_smear_pos', 'index_hiv_pos', 'index_age_slope',
            'contact_lives_in', 'contact_not_lives_in'}
        assert all(v > 0 for v in LAMBDA_PRIORS.values())

    def test_lambda_handcheck(self):
        """U 形易感性 × 涂片感染力 × 同住强度，逐项手算。"""
        import pandas as pd
        df = pd.DataFrame({
            'age_lt5': [1.0], 'age_ge45': [0.0],
            'idx_smear_pos': [1.0], 'idx_hiv_pos': [0.0],
            'idx_age': [35.0], 'lives_in_household': [1.0],
        })
        lam = build_lambda_real(df)
        # S = 1+1.8 = 2.8；I = 3.0·exp(0) = 3.0；C = 1.0
        assert lam.iloc[0] == pytest.approx(2.8 * 3.0 * 1.0)

    def test_lambda_monotone_in_index_infectiousness(self):
        import pandas as pd
        base = {'age_lt5': [0.0, 0.0], 'age_ge45': [1.0, 1.0],
                'idx_smear_pos': [1.0, 0.0], 'idx_hiv_pos': [1.0, 0.0],
                'idx_age': [40.0, 40.0],
                'lives_in_household': [1.0, 0.0]}
        lam = build_lambda_real(pd.DataFrame(base))
        assert lam.iloc[0] > lam.iloc[1]  # 涂片+/HIV+/同住 → 更大 λ


class TestFeatureContract:
    """特征臂契约：嵌套结构 + 无随访泄漏。"""

    def test_arms_nested(self):
        for arm in FEATURE_ARMS:
            cols = feature_columns(arm)
            assert len(cols) == len(set(cols))  # 无重复列
        assert set(FEATURE_SETS['ind']) <= set(feature_columns('pi'))

    def test_no_followup_leakage(self):
        """特征列不得包含任何随访结局列（3 月检验/治疗/结局）。"""
        forbidden = set(_OUTCOME_COLS)
        for arm in FEATURE_ARMS:
            for col in FEATURE_SETS[arm]:
                assert col not in forbidden
                assert not col.startswith('contact3m')


@needs_data
class TestLoaderAndCV:
    """装载契约 + 户分组 CV。"""

    @pytest.fixture(scope='class')
    def df(self):
        return load_pacts_contacts()

    def test_load_contract(self, df):
        # 838 接触者 − 2 终点缺失 − 32 无基线问卷匹配（防哑元假数据回归）
        assert len(df) == 804
        assert df['sx_3m'].notna().all()
        assert int(df['sx_3m'].sum()) == 107
        assert df['id'].nunique() == 192
        for arm in FEATURE_ARMS:
            assert not df[feature_columns(arm)].isna().any().any()

    def test_baseline_features_from_baseline_file(self, df):
        """同访泄漏防回归：基线症状必须取自 _b 列（基线问卷）。"""
        import pandas as pd
        syms = ['p16cgh', 'p17doc', 'p18fvr', 'p19nsa',
                'p20nsb', 'p21wla', 'p22wlb']
        sym_b = df[[c + '_b' for c in syms]].apply(
            pd.to_numeric, errors='coerce').eq(1).sum(axis=1)
        assert (sym_b == df['baseline_symptom_count']).all()
        # 反向锚定：随访问卷列（t 文件同名列）与基线列确有差异——
        # 若特征误用随访问卷列，上面的一致性不会成立
        sym_t = df[syms].apply(pd.to_numeric, errors='coerce').eq(1).sum(axis=1)
        assert not bool((sym_t == sym_b).all())

    def test_lambda_available_and_positive(self, df):
        assert (df['lam'] > 0).all()
        assert df['lam'].std() > 0  # 有区分度

    def test_group_cv_disjoint_and_complete(self, df):
        y = df['sx_3m'].astype(int).to_numpy()
        groups = df['id'].to_numpy()
        folds = _group_cv_indices(groups, y, n_splits=5, seed=0)
        seen = np.zeros(len(y), dtype=int)
        for tr, te in folds:
            assert set(groups[tr]).isdisjoint(set(groups[te]))
            seen[te] += 1
        assert (seen == 1).all()  # 每行恰被测一次
        # 分层：每折测试半区均含事件
        for _, te in folds:
            assert y[te].sum() >= 5


@needs_data
class TestRealDataAblation:
    """单种子 + 多种子契约与确定性。"""

    def test_once_deterministic(self):
        r1 = run_real_data_ablation_once(seed=3)
        r2 = run_real_data_ablation_once(seed=3)
        for k in r1['arms']:
            assert r1['arms'][k]['auroc'] == \
                pytest.approx(r2['arms'][k]['auroc'], abs=1e-12)

    def test_once_contract(self):
        r = run_real_data_ablation_once(seed=0)
        assert set(r['design']) >= {'n', 'n_events', 'n_households',
                                    'n_splits', 'endpoint'}
        expected = (['pi_only']
                    + [f'{a}:{m}' for a in FEATURE_ARMS
                       for m in ('random_forest', 'lightgbm')])
        assert set(r['arms']) == set(expected)
        for name, a in r['arms'].items():
            assert 0.4 < a['auroc'] < 1.0, name
            assert np.isfinite(a['pr_auc'])
        assert np.isfinite(r['arms']['ind:random_forest'][
            'auroc_m3_tb_out'])

    def test_multi_seed_smoke(self):
        out = run_multi_seed_real_data(n_seeds=2, seed_start=7,
                                       n_bootstrap=100)
        assert out['design']['n_seeds'] == 2
        assert len(out['seeds']) == 2
        for key in ('pi:random_forest_minus_exposure:random_forest',
                    'pi_only_minus_ind:random_forest'):
            s = out['ladder_summary'][key]
            assert set(s) >= {'mean', 'bootstrap_ci', 'ci_excludes_zero'}
            assert s['bootstrap_ci'][0] <= s['mean'] <= s['bootstrap_ci'][1]
        assert out['conclusion']['pi_feature_gain']['mean'] < 1.0
