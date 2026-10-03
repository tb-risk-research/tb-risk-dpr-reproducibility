#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""λ 联合校准 + 乘子审计 + 户聚合模块测试（P1/P2/P3，2026-08-25）。

数据缺失时自动跳过（homeacf_tstsa.rda 不随仓库分发）。
"""

import os

import numpy as np
import pandas as pd
import pytest

from tb_risk.validation.real_data_infection import _RDA_PATH
from tb_risk.validation.lambda_calibration import (
    MULTIPLIER_COMPONENTS, PRIOR_START, SIMPLIFIED_COMPONENTS,
    calibrate_lambda_torch, household_features, multiplier_audit,
    multiplier_matrix, run_calibration_ablation_once,
    run_multi_seed_calibration,
)

_HAVE_DATA = os.path.isfile(_RDA_PATH)
needs_data = pytest.mark.skipif(not _HAVE_DATA,
                                reason='HomeACF .rda 未下载')


class TestMultiplierMatrix:
    """乘子分量契约。"""

    def test_components_frozen(self):
        assert MULTIPLIER_COMPONENTS == (
            'age_lt5', 'age_ge45', 'host_hiv', 'cough_days', 'index_hiv',
            'log_ts', 'share_bedroom', 'sleep_same_bed')
        assert 'smear' not in MULTIPLIER_COMPONENTS  # P2：83% 缺失移除
        assert all(np.isfinite(v) for v in PRIOR_START.values())

    def test_prior_start_is_literature(self):
        """起点 = 文献先验（log 空间）。"""
        assert PRIOR_START['age_lt5'] == pytest.approx(np.log(2.8))
        assert PRIOR_START['cough_days'] == pytest.approx(0.01)
        assert PRIOR_START['log_ts'] == pytest.approx(1.0)

    def test_simplified_subset_contract(self):
        """P2：精简乘子集 = 5 个数据一致项，移除项全在反向名单。"""
        assert SIMPLIFIED_COMPONENTS == (
            'cough_days', 'host_hiv', 'share_bedroom', 'age_ge45',
            'log_ts')
        assert set(SIMPLIFIED_COMPONENTS) <= set(MULTIPLIER_COMPONENTS)
        assert set(SIMPLIFIED_COMPONENTS).isdisjoint(
            {'age_lt5', 'index_hiv', 'sleep_same_bed'})

    def test_multiplier_matrix_component_subset(self):
        """components 子集下矩阵仍含全列、返回列名受控。"""
        df = pd.DataFrame({
            'age_lt5': [0.0, 1.0], 'age_ge45': [0.0, 0.0],
            'hiv_pos_h': [1.0, 0.0], 'idx_coughdays': [10.0, 90.0],
            'idx_hiv_pos': [0.0, 1.0], 'ts_low': [1.0, 0.0],
            'ts_mid': [0.0, 0.0], 'ts_high': [0.0, 1.0],
            'share_bedroom': [1.0, 0.0], 'sleep_same_bed': [0.0, 1.0],
        })
        C, cols = multiplier_matrix(df, components=SIMPLIFIED_COMPONENTS)
        assert cols == list(SIMPLIFIED_COMPONENTS)
        assert len(C) == 2
        C_full, cols_full = multiplier_matrix(df)
        assert cols_full == list(MULTIPLIER_COMPONENTS)


class TestCalibrateTorch:
    """可微校准：确定性 + 合成恢复。"""

    def _toy_C(self, n=400, seed=0):
        rng = np.random.RandomState(seed)
        C = pd.DataFrame({
            'age_lt5': (rng.rand(n) < 0.15).astype(float),
            'age_ge45': (rng.rand(n) < 0.2).astype(float),
            'host_hiv': (rng.rand(n) < 0.1).astype(float),
            'cough_days': rng.randint(0, 90, n).astype(float),
            'index_hiv': (rng.rand(n) < 0.5).astype(float),
            'log_ts': rng.choice([-0.69, 0.0, 0.4], n),
            'share_bedroom': (rng.rand(n) < 0.2).astype(float),
            'sleep_same_bed': (rng.rand(n) < 0.05).astype(float),
        })
        return C

    def test_deterministic(self):
        C = self._toy_C()
        y = (np.random.RandomState(1).rand(len(C)) < 0.2).astype(int)
        f1 = calibrate_lambda_torch(C, y, np.arange(len(y)), seed=3)
        f2 = calibrate_lambda_torch(C, y, np.arange(len(y)), seed=3)
        for k in f1:
            if k == 'loss_trace':
                continue
            assert f1[k] == pytest.approx(f2[k], abs=1e-10)

    def test_recovers_direction_on_synthetic(self):
        """标签由 cough + host_hiv 驱动 → 校准系数方向正确且显著非零。"""
        C = self._toy_C(n=600, seed=2)
        logit = -2.5 + 0.03 * C['cough_days'] + 1.0 * C['host_hiv']
        p = 1 / (1 + np.exp(-logit))
        y = (np.random.RandomState(3).rand(len(C)) < p).astype(int)
        fit = calibrate_lambda_torch(C, y, np.arange(len(y)), seed=0)
        assert fit['cough_days'] > 0.015          # 方向 + 接近真值 0.03
        assert fit['host_hiv'] > 0.5              # 方向 + 量级正确
        assert abs(fit['sleep_same_bed']
                   - PRIOR_START['sleep_same_bed']) < 1.5  # 无信号→回先验


@needs_data
class TestCalibrationAblation:
    """三臂 + 户聚合：契约、防泄漏、验收。"""

    @pytest.fixture(scope='class')
    def rep(self):
        return run_calibration_ablation_once(seed=0)

    @pytest.fixture(scope='class')
    def df(self):
        from tb_risk.validation.real_data_infection import (
            load_homeacf_contacts)
        return load_homeacf_contacts()

    def test_arm_contract(self, rep):
        expected = {'frozen_pi_only', 'cal_pi_only',
                    'frozen_simplified_pi_only', 'cal_simplified_pi_only',
                    'ind:RF', 'exposure:RF', 'exposure_hh:RF',
                    'exposure_calpi:RF', 'exposure_calsimpl_pi:RF',
                    'hh_tst_rate_loo_only'}
        assert set(rep['arms']) == expected
        for name, a in rep['arms'].items():
            assert np.isfinite(a['auroc']), name

    def test_simplified_direction(self, rep):
        """P2 方向性：精简校准 λ 排序不低于全乘子校准（去噪方向）。"""
        assert rep['arms']['cal_simplified_pi_only']['auroc'] >= \
            rep['arms']['cal_pi_only']['auroc'] - 0.005

    def test_cal_beats_frozen_direction(self, rep):
        """P1 验收方向：校准 λ 排序不低于 frozen（逐种子口径）。"""
        assert rep['arms']['cal_pi_only']['auroc'] >= \
            rep['arms']['frozen_pi_only']['auroc'] - 0.005

    def test_cal_oof_not_full_fit(self, df):
        """校准 OOF 防泄漏：测试折分数必须来自训练折系数
        （全拟合系数 ≠ 折级系数 → OOF 与全拟合打分不同）。"""
        y = df['tst_pos10'].astype(int).to_numpy()
        C, cols = multiplier_matrix(df)
        full = calibrate_lambda_torch(C, y, np.arange(len(y)), seed=0)
        b_full = np.array([full[c] for c in cols])
        s_full = full['intercept'] + C.to_numpy(dtype=float) @ b_full
        rep = run_calibration_ablation_once(seed=0, df=df)
        assert not np.allclose(rep['oof']['cal_pi_only'], s_full)

    def test_household_features_no_label_leak(self, df):
        """hh_mean_oof 用 OOF 预测（非标签）；loo 独户填全局率。"""
        y = df['tst_pos10'].astype(int).to_numpy()
        oof_ind = np.random.RandomState(0).rand(len(df))
        hh = household_features(df, y, oof_ind)
        # OOF 均值恒在 [0,1]，独户 loo = 全局率
        assert hh['hh_mean_oof'].between(0, 1).all()
        solo = df['record_id'].map(
            df['record_id'].value_counts()) == 1
        assert np.allclose(
            hh.loc[solo, 'hh_tst_rate_loo'], np.mean(y))

    def test_cal_coefs_recorded(self, rep):
        assert len(rep['cal_coefs']) == 5          # 每折一份
        for co in rep['cal_coefs']:
            assert set(co) == set(rep['cal_cols']) | {'intercept'}

    def test_multiplier_audit_table(self, df):
        """P2 审计表：主乘子可检验 + smear 不可检验。"""
        from tb_risk.validation.real_data_pi import _group_cv_indices
        y = df['tst_pos10'].astype(int).to_numpy()
        folds = _group_cv_indices(df['record_id'].to_numpy(), y,
                                  n_splits=5, seed=0)
        rep = run_calibration_ablation_once(seed=0, df=df)
        rows = multiplier_audit(df, y, folds, rep['cal_coefs'],
                                rep['cal_cols'])
        by = {r['multiplier']: r for r in rows}
        assert set(by) == set(MULTIPLIER_COMPONENTS) | {'smear'}
        assert by['smear']['testable'] is False
        assert 'known_subset' in by['smear']
        for c in MULTIPLIER_COMPONENTS:
            r = by[c]
            assert r['testable'] is True
            assert 'prior_value' in r and 'data_direction' in r
            assert 0.4 < r['univariate_oof_auroc'] < 0.8

    def test_multi_seed_smoke(self):
        out = run_multi_seed_calibration(n_seeds=2, seed_start=11,
                                         n_bootstrap=100)
        assert out['design']['n_seeds'] == 2
        assert len(out['seeds']) == 2
        acc = out['design']['acceptance']
        assert set(acc) == {'P1a_auclift_vs_frozen',
                            'P1b_feature_no_harm',
                            'P2_simplified_lift_vs_full_cal',
                            'P2_simplified_feature_no_harm'}
        for k, s in out['ladder_summary'].items():
            assert s['bootstrap_ci'][0] <= s['mean'] <= \
                s['bootstrap_ci'][1]
