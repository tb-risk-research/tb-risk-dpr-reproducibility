#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务1（2026-09-14）：主训练路径组感知交叉验证。

背景：原全链路随机 StratifiedKFold，TREATS group_id（14 社区）/
Peru family_id（688 户）的户/社区成员同时进训练与验证折 →
AUROC 偏乐观、超参选择失真。本测试覆盖对齐后的行为契约：

1. 组路径：StratifiedGroupKFold 折间组不相交；entry 记录
   cv_protocol='stratified_group_kfold' / group_column / n_groups /
   池化 OOF bootstrap CI（AUROC_ci95/AUPRC_ci95）；
2. 无组路径：完全维持存量语义（stratified_kfold、逐折均值、
   无 CI 键值），向后兼容；
3. train_from_real_data：组列自动探测命中 / 显式指定缺失时
   error_type='missing_group_column' 响亮失败 / 无组列回退；
4. cluster_bootstrap_metric_ci：CI 包含点估计、lo < hi、退化防护。
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import (  # noqa: E402
    _train_all_registered_models,
    ensure_interaction_features,
    GROUP_COLUMN_CANDIDATES,
)
from tb_risk.scoring.ml.validation import cluster_bootstrap_metric_ci  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402


def _toy_matrix(n=300, seed=42):
    """22 列带列名 DataFrame + 弱信号标签（快速训练用）。"""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 22)
    y = (X[:, 0] + 0.5 * X[:, 3] + rng.normal(0, 0.3, n) > 1.1).astype(int)
    names = list(MLRiskPredictor.ALL_FEATURE_NAMES)
    return pd.DataFrame(X, columns=names), y


def _toy_groups(n=300, n_groups=30, seed=7):
    """模拟户结构：每户 10 人，户内标签相关（信号由组携带）。"""
    rng = np.random.RandomState(seed)
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    return groups[:n]


class TestGroupAwareKernel:

    def test_group_path_protocol_and_ci(self):
        """组路径：协议/组数/CI 键齐全，CI 合理。"""
        X, y = _toy_matrix()
        groups = _toy_groups(len(y))
        pred = MLRiskPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42, data_source_label='real',
            enable_calibration=False, model_keys=['random_forest'],
            groups=groups, group_column_name='family_id')
        assert ok
        entry = pred.model_performance['random_forest']
        assert entry['cv_protocol'] == 'stratified_group_kfold'
        assert entry['group_column'] == 'family_id'
        assert entry['n_groups'] == len(np.unique(groups))
        ci = entry['AUROC_ci95']
        assert ci is not None
        assert ci[0] < ci[1]
        # CI 应覆盖点估计附近（2.5/97.5 分位区间含点估计是常态，
        # 至少点估计不落在区间外的极端反例上）
        assert ci[0] - 0.05 <= entry['AUROC'] <= ci[1] + 0.05
        assert entry['n_bootstrap_effective'] > 0

    def test_folds_disjoint_by_group(self):
        """StratifiedGroupKFold 契约：任何组不同时出现在训练/验证折。"""
        rng = np.random.RandomState(0)
        n = 400
        y = rng.binomial(1, 0.3, n)
        groups = rng.randint(0, 40, n)  # 组大小不均（更严苛）
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in cv.split(np.zeros(n), y, groups=groups):
            assert not set(groups[tr]) & set(groups[te])

    def test_no_group_path_unchanged(self):
        """无组路径：存量语义（协议名/无 CI 值）。"""
        X, y = _toy_matrix(n=200, seed=1)
        pred = MLRiskPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42, data_source_label='real',
            enable_calibration=False, model_keys=['random_forest'])
        assert ok
        entry = pred.model_performance['random_forest']
        assert entry['cv_protocol'] == 'stratified_kfold'
        assert entry['group_column'] is None
        assert entry['n_groups'] is None
        assert entry.get('AUROC_ci95') is None

    def test_fold_clamp_when_few_groups(self):
        """组数 < cv_folds：钳制折数不崩溃。"""
        X, y = _toy_matrix(n=150, seed=2)
        groups = np.repeat(np.arange(3), 50)  # 仅 3 组
        pred = MLRiskPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42, data_source_label='real',
            enable_calibration=False, cv_folds=5,
            model_keys=['logistic'], groups=groups)
        assert ok
        entry = pred.model_performance['logistic']
        assert entry['cv_protocol'] == 'stratified_group_kfold'
        assert entry['n_groups'] == 3

    def test_hyperopt_group_path(self):
        """超参搜索组路径可跑通（网格小样本快）。"""
        X, y = _toy_matrix(n=200, seed=3)
        groups = _toy_groups(len(y), n_groups=20)
        pred = MLRiskPredictor()
        ok = _train_all_registered_models(
            pred, X, y, random_state=42, data_source_label='real',
            enable_calibration=False, enable_hyperopt=True,
            model_keys=['logistic'], groups=groups)
        assert ok
        assert pred.model_performance['logistic']['cv_protocol'] == \
            'stratified_group_kfold'


class TestTrainFromRealDataGroupColumn:

    def _make_csv(self, tmp_path, with_group=True, group_col='family_id',
                  n=250, seed=11):
        X, y = _toy_matrix(n=n, seed=seed)
        df = X.copy()
        df['tb_outcome'] = y
        if with_group:
            df[group_col] = _toy_groups(n, n_groups=25, seed=seed)
        csv = tmp_path / f'toy_{with_group}_{group_col}.csv'
        df.to_csv(csv, index=False)
        return str(csv), df

    def test_auto_detect_family_id(self, tmp_path):
        """自动探测 family_id → 组感知协议 + 结果携带组信息。"""
        csv, _ = self._make_csv(tmp_path)
        pred = MLRiskPredictor()
        res = pred.train_from_real_data(csv, target_column='tb_outcome')
        assert res.get('success')
        assert res['cv_protocol'] == 'stratified_group_kfold'
        assert res['group_column'] == 'family_id'
        assert res['n_groups'] == 25
        entry = next(iter(pred.model_performance.values()))
        assert entry['cv_protocol'] == 'stratified_group_kfold'

    def test_explicit_missing_group_column_fails(self, tmp_path):
        """显式指定不存在的组列 → 响亮失败（不静默回退）。"""
        csv, _ = self._make_csv(tmp_path)
        pred = MLRiskPredictor()
        res = pred.train_from_real_data(
            csv, target_column='tb_outcome', group_column='nonexistent_col')
        assert not res.get('success')
        assert res.get('error_type') == 'missing_group_column'

    def test_no_group_column_falls_back(self, tmp_path):
        """无组列 CSV → 随机分层 CV（正确回退，非组感知）。"""
        csv, _ = self._make_csv(tmp_path, with_group=False)
        pred = MLRiskPredictor()
        res = pred.train_from_real_data(csv, target_column='tb_outcome')
        assert res.get('success')
        assert res['cv_protocol'] == 'stratified_kfold'
        assert res['group_column'] is None

    def test_explicit_group_column_used(self, tmp_path):
        """显式指定候选列（group_id）→ 使用该列。"""
        csv, _ = self._make_csv(tmp_path, group_col='group_id')
        pred = MLRiskPredictor()
        res = pred.train_from_real_data(
            csv, target_column='tb_outcome', group_column='group_id')
        assert res.get('success')
        assert res['group_column'] == 'group_id'


class TestClusterBootstrapMetricCI:

    def test_ci_contains_point(self):
        """CI 合理性：lo < hi，点估计在区间内（连续指标常态）。"""
        rng = np.random.RandomState(5)
        n = 300
        y = rng.binomial(1, 0.3, n)
        scores = np.clip(y * 0.4 + rng.rand(n) * 0.6, 0, 1)
        groups = np.repeat(np.arange(30), 10)
        out = cluster_bootstrap_metric_ci(scores, y, groups,
                                          n_bootstrap=200, seed=0)
        assert out['point'] is not None
        assert out['ci95'][0] < out['ci95'][1]
        assert out['ci95'][0] <= out['point'] <= out['ci95'][1]
        assert out['n_effective'] == 200

    def test_degenerate_inputs(self):
        """单类 y / 单组 → point 尽量给、CI 为 None 不崩溃。"""
        y = np.zeros(50, dtype=int)
        scores = np.random.RandomState(0).rand(50)
        groups = np.repeat(np.arange(5), 10)
        out = cluster_bootstrap_metric_ci(scores, y, groups)
        assert out['ci95'] is None

    def test_length_mismatch_raises(self):
        y = np.zeros(10, dtype=int)
        with pytest.raises(ValueError):
            cluster_bootstrap_metric_ci(np.zeros(9), y, np.zeros(10))


class TestEnsureInteractionFeatures:

    def test_missing_cols_filled(self):
        """13 基础列 → 9 交互列自动补齐。"""
        rng = np.random.RandomState(3)
        n = 60
        base_cols = list(MLRiskPredictor.FEATURE_NAMES)
        df = pd.DataFrame({
            'age': rng.randint(1, 80, n),
            'cumulative_exposure': rng.rand(n) * 100,
            'has_symptoms': rng.binomial(1, 0.3, n),
            'bcg_vaccine': rng.binomial(1, 0.9, n),
            'has_tb': rng.binomial(1, 0.05, n),
            'contact_distance_score': rng.rand(n),
            'ventilation_score': rng.rand(n),
            'is_high_risk': rng.binomial(1, 0.2, n),
            'past_illness': rng.binomial(1, 0.2, n),
            'exposure_setting_score': rng.rand(n),
            'single_duration': rng.randint(5, 400, n),
            'freq_density': rng.randint(1, 30, n),
            'time_span': rng.randint(1, 52, n),
        })
        assert set(base_cols) == set(df.columns)
        inter = list(MLRiskPredictor.INTERACTION_FEATURE_NAMES)
        df2, missing = ensure_interaction_features(df, inter)
        assert set(missing) == set(inter)
        for c in inter:
            assert c in df2.columns
            assert df2[c].notna().all()

    def test_noop_when_present(self):
        """全部交互列已存在 → 不动原框。"""
        rng = np.random.RandomState(4)
        df = pd.DataFrame({'a': [1, 2]})
        for c in MLRiskPredictor.INTERACTION_FEATURE_NAMES:
            df[c] = rng.rand(2)
        df2, missing = ensure_interaction_features(
            df, list(MLRiskPredictor.INTERACTION_FEATURE_NAMES))
        assert missing == []
        assert list(df2.columns) == list(df.columns)


def test_group_column_candidates():
    """候选列顺序：family_id（Peru 户）优先于 group_id（TREATS 社区）。"""
    assert GROUP_COLUMN_CANDIDATES[0] == 'family_id'
    assert 'group_raw' not in GROUP_COLUMN_CANDIDATES  # 泄漏排除列不在候选


# ============================================================================
# 任务2：多队列部分池化（CohortMixedModel）
# ============================================================================

def _toy_cohorts(n_per=200, n_cohorts=3, seed=0, n_features=4):
    """三队列合成数据：共享斜率 + 队列基线率差异（随机截距结构）。"""
    rng = np.random.RandomState(seed)
    beta = np.array([1.2, -0.8, 0.5, 0.3])[:n_features]
    Xs, ys, cs = [], [], []
    base = {'a': 0.3, 'b': 0.05, 'c': 0.7}
    for i, name in enumerate(['a', 'b', 'c'][:n_cohorts]):
        X = rng.randn(n_per, n_features)
        logit = -1.0 + (2.0 * (base[name] - 0.4)) + X @ beta
        y = (rng.rand(n_per) < 1 / (1 + np.exp(-logit))).astype(int)
        Xs.append(X)
        ys.append(y)
        cs.append(np.full(n_per, name))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(cs)


class TestCohortMixedModel:

    def setup_method(self):
        pytest.importorskip('statsmodels')
        from tb_risk.scoring.ml.partial_pooling import CohortMixedModel
        self.cls = CohortMixedModel

    def test_fit_deterministic_with_seed(self):
        """同数据 + 同 random_state 两次拟合 → 参数完全一致。

        回归 statsmodels fit_vb/fit_map 未播种全局 RNG 的非确定性
        （bayes_mixed_glm.py L402/L759；brazil LOCO 两次 AUROC
        0.40 vs 0.59 的实测教训）。
        """
        X, y, c = _toy_cohorts()
        m1 = self.cls().fit(X, y, c, random_state=7, n_starts=2)
        m2 = self.cls().fit(X, y, c, random_state=7, n_starts=2)
        assert m1.fit_method == m2.fit_method
        assert np.allclose(m1.fe_mean, m2.fe_mean)
        assert m1.vc_mean == m2.vc_mean

    def test_predict_modes_known_vs_zero_shot(self):
        """已知队列加 BLUP；未知队列 = 总体截距（零样本）。"""
        X, y, c = _toy_cohorts()
        m = self.cls().fit(X, y, c, random_state=0, n_starts=2)
        p_known = m.predict_proba(X, cohort='a')
        p_zero = m.predict_proba(X, cohort=None)
        p_unknown = m.predict_proba(X, cohort='never_seen')
        # 已知队列：logit 差恒等于 BLUP
        blup = m.vc_mean['a']
        logit_known = np.log(p_known / (1 - p_known))
        logit_zero = np.log(p_zero / (1 - p_zero))
        assert np.allclose(logit_known - logit_zero, blup, atol=1e-8)
        assert np.allclose(p_unknown, p_zero)  # 未知 → 总体截距
        # 零样本判别：合成数据共享斜率 → 排序应显著优于随机
        from sklearn.metrics import roc_auc_score
        assert roc_auc_score(y, p_zero) > 0.6

    def test_recalibrate_intercept_freezes_slopes(self):
        """本地重校准：只动截距，斜率冻结，队列内排序不变。"""
        X, y, c = _toy_cohorts()
        m = self.cls().fit(X, y, c, random_state=0, n_starts=2)
        fe_before = m.fe_mean.copy()
        mask = c == 'a'
        XA, yA = X[mask][:80], y[mask][:80]
        XB, yB = X[mask][80:], y[mask][80:]
        new_a = m.recalibrate_intercept_locally(XA, yA)
        assert np.isfinite(new_a)
        assert np.allclose(m.fe_mean, fe_before)  # 斜率未动
        # 评估半区：重校准前后 logit 差为常数（= 截距位移）
        logit_recal = np.log(m.predict_proba(XB, cohort=None) /
                             (1 - m.predict_proba(XB, cohort=None)))
        m.local_intercept = None  # 临时还原总体截距口径
        logit_zero = np.log(m.predict_proba(XB, cohort=None) /
                            (1 - m.predict_proba(XB, cohort=None)))
        assert np.allclose(logit_recal - logit_zero,
                           new_a - fe_before[0], atol=1e-8)
        # 秩不变：AUROC 完全一致
        from sklearn.metrics import roc_auc_score
        m.local_intercept = new_a
        p_r = m.predict_proba(XB, cohort=None)
        m.local_intercept = None
        p_z = m.predict_proba(XB, cohort=None)
        assert roc_auc_score(yB, p_r) == roc_auc_score(yB, p_z)

    def test_summary_and_validation(self):
        X, y, c = _toy_cohorts()
        m = self.cls(feature_names=['x1', 'x2', 'x3', 'x4'])
        m.fit(X, y, c, random_state=0, n_starts=1)
        s = m.summary_dict()
        assert s['fitted'] is True
        assert set(s['cohort_blups']) == {'a', 'b', 'c'}
        assert s['random_intercept_sd'] > 0
        assert 'endpoint_heterogeneity_note' in s
        # 输入校验
        with pytest.raises(ValueError):
            m.fit(X[:10], y[:10], c[:10])          # n < 20
        with pytest.raises(ValueError):
            m.fit(X, np.zeros(len(y), int), c)     # 单类标签
        with pytest.raises(ValueError):
            m.fit(X[:-1], y, c)                    # 长度不一致
