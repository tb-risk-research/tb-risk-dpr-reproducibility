#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5（2026-09-16）：Peru 生存分析转正（survival_cox）回归测试

覆盖：
1. Schoenfeld PH 检验（Grambsch-Therneau score test）：
   - 零假设校准：PH 数据全局拒绝率 ≤ 15%（5% 名义，不过度拒绝）；
   - 功效：分段指数逆变换注入穿越（时间相依）效应 → 全局 + 逐变量检出；
   - 退化与校验：事件 <5 优雅返回 / 未知 transform / log 变换要求 t>0 /
     输出结构契约。
2. F4 数值机器：zfit 零方差保护 / fit_mask 零方差+共线剔除映射 /
   harrell_c 手工算例 / window_label 删失感知窗语义。
3. CoxSurvivalModel：退化列（零方差/完全共线）拟合防护、剔除列零贡献、
   predict 维度校验、β 符号恢复、to_dict/from_dict 往返逐位一致 +
   JSON 可序列化 + _res 不落盘。
4. train_survival_model：返回结构 / groups 感知 CV 披露 / 判别恢复
   （模拟信号 Harrell C > 0.6）/ too_few_events 门（事件 <10）。
5. train_from_real_data 生存分支：follow_up_days 在场 → survival_model
   挂载 + predict_risk 报 survival_cox（含单调性）+ checkpoint 往返；
   无时间列不挂载 + 陈旧挂载清理。

依据：F4 peru_survival_20260915（B−A Harrell C +0.0491）；PH 检验模拟
验证（零假设 3/100 ≈ 5% 名义；穿越效应 ±0.9/−1.2 检出 50/50）；
单一真值源 ml/survival_cox.py。
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import train_from_real_data  # noqa: E402
from tb_risk.scoring.ml.survival_cox import (  # noqa: E402
    STATSMODELS_SURVIVAL_AVAILABLE, CoxSurvivalModel, cox_fit, fit_mask,
    harrell_c, schoenfeld_ph_test, train_survival_model, window_label, zfit,
)

pytestmark = pytest.mark.skipif(
    not STATSMODELS_SURVIVAL_AVAILABLE,
    reason='statsmodels 不可用，生存路径整体跳过')

BASE13 = [
    'age', 'cumulative_exposure', 'has_symptoms', 'bcg_vaccine',
    'has_tb', 'contact_distance_score', 'ventilation_score',
    'is_high_risk', 'past_illness', 'exposure_setting_score',
    'single_duration', 'freq_density', 'time_span',
]


# ============================================================================
# 模拟数据构造
# ============================================================================

def _sim_ph(n=300, beta=(0.7, 0.5, -0.4), seed=42):
    """PH 满足数据：指数基线 + 固定 β，均匀删失。

    返回 (t, e, X)；X 三列（连续 + 两二元）。
    """
    rng = np.random.RandomState(seed)
    X = np.column_stack([
        rng.randn(n),
        rng.randint(0, 2, n).astype(float),
        rng.randint(0, 2, n).astype(float),
    ])
    lam = 0.004 * np.exp(X @ np.asarray(beta, float))
    t_event = -np.log(rng.rand(n)) / lam
    t_cens = rng.uniform(50, 600, n)
    t = np.minimum(t_event, t_cens)
    e = (t_event <= t_cens).astype(int)
    return t, e, X


def _sim_crossing(n=300, tau=180.0, b_early=0.9, b_late=-1.2, seed=42):
    """时间相依（穿越）效应：分段指数逆变换精确采样。

    H(t) 分段线性（τ 前后 hazard 各含一个 β），解 −log(u) = H(t)：
    τ 前解得则取之，否则从 τ 起按晚期 hazard 续解。β 早期 +0.9 /
    晚期 −1.2 → 效应在 τ 翻转，PH 假设被破坏。
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(n)
    h_e = 0.004 * np.exp(b_early * x)
    h_l = 0.004 * np.exp(b_late * x)
    u = rng.rand(n)
    neg_log_u = -np.log(u)
    t_early = neg_log_u / h_e
    t = np.where(t_early < tau, t_early,
                 tau + (neg_log_u - h_e * tau) / h_l)
    t_cens = rng.uniform(60, 900, n)
    obs = np.minimum(t, t_cens)
    e = (t <= t_cens).astype(int)
    return obs, e, x


def _fit_and_ph(t, e, X, names, transform='log'):
    """生产口径小助手：z 标准化 → Cox(BFGS) → PH 检验。"""
    mu, sd = zfit(X)
    res = cox_fit(t, (X - mu) / sd, e)
    return schoenfeld_ph_test(res, t, e, names, transform=transform)


# ============================================================================
# 1. Schoenfeld PH 检验
# ============================================================================

class TestSchoenfeldPH:

    def test_null_calibration(self):
        """PH 数据全局拒绝率 ≤ 15%（5% 名义；固定种子确定性）。"""
        n_reps, rejections = 60, 0
        for rep in range(n_reps):
            t, e, X = _sim_ph(n=220, seed=1000 + rep)
            out = _fit_and_ph(t, e, X, ['a', 'b', 'c'])
            if out['global']['p_value'] < 0.05:
                rejections += 1
        assert rejections / n_reps <= 0.15, (
            f'PH 数据拒绝率 {rejections}/{n_reps} 过高（检验过激进）')

    def test_crossing_effect_detected(self):
        """穿越效应（β 早 +0.9 / 晚 −1.2）→ 全局 + 逐变量高检出。"""
        n_reps, glob, per_var = 20, 0, 0
        for rep in range(n_reps):
            t, e, x = _sim_crossing(seed=2000 + rep)
            out = _fit_and_ph(t, e, x.reshape(-1, 1), ['x'])
            if out['global']['p_value'] < 0.05:
                glob += 1
            if out['per_variable'][0]['p_value'] < 0.05:
                per_var += 1
        assert glob >= n_reps - 2, f'全局检出 {glob}/{n_reps}'
        assert per_var >= n_reps - 2, f'逐变量检出 {per_var}/{n_reps}'

    def test_too_few_events_degrades_gracefully(self):
        rng = np.random.RandomState(0)
        n = 30
        X = rng.randn(n, 2)
        t = np.full(n, 100.0)
        e = np.zeros(n, int)
        e[:3] = 1
        t[:3] = [50.0, 80.0, 120.0]
        res = cox_fit(t, X, e)
        out = schoenfeld_ph_test(res, t, e, ['a', 'b'])
        assert out['global'] is None
        assert out['per_variable'] == []
        assert '事件数不足' in out['note']

    def test_unknown_transform_raises(self):
        t, e, X = _sim_ph(n=120, seed=5)
        with pytest.raises(ValueError, match='未知 transform'):
            _fit_and_ph(t, e, X, ['a', 'b', 'c'], transform='km')

    def test_log_transform_requires_positive_event_times(self):
        t, e, X = _sim_ph(n=120, seed=6)
        mu, sd = zfit(X)
        res = cox_fit(t, (X - mu) / sd, e)
        t_bad = t.copy()
        e_bad = e.copy()
        t_bad[0], e_bad[0] = 0.0, 1  # 事件时间为 0
        with pytest.raises(ValueError, match='log'):
            schoenfeld_ph_test(res, t_bad, e_bad, ['a', 'b', 'c'],
                               transform='log')

    def test_output_structure(self):
        t, e, X = _sim_ph(n=150, seed=7)
        out = _fit_and_ph(t, e, X, ['a', 'b', 'c'])
        assert out['transform'] == 'log'
        assert out['n_events'] == int(e.sum())
        assert out['global']['df'] == 3
        assert 0.0 <= out['global']['p_value'] <= 1.0
        assert len(out['per_variable']) == 3
        assert {v['feature'] for v in out['per_variable']} == {'a', 'b', 'c'}
        assert 0 <= out['n_violated_0.05'] <= 3


# ============================================================================
# 2. F4 数值机器（单元）
# ============================================================================

class TestNumericMachinery:

    def test_zfit_zero_variance_guard(self):
        X = np.array([[1.0, 2.0, 5.0],
                      [3.0, 2.0, 7.0],
                      [5.0, 2.0, 9.0]])
        mu, sd = zfit(X)
        assert sd[1] == 1.0          # 零方差列除 1 保护
        assert sd[0] > 0 and sd[2] > 0

    def test_fit_mask_drops_zero_variance_and_collinear(self):
        rng = np.random.RandomState(1)
        n = 200
        X = np.column_stack([
            rng.randn(n),                       # 0 live
            2 * rng.randn(n) + 1,               # 占位（不共线）
            np.full(n, 3.0),                    # 2 零方差
            rng.randint(0, 2, n).astype(float),  # 3 live
        ])
        # 重构列 1 与列 0 完全共线
        X[:, 1] = 2 * X[:, 0] + 1
        mask, dropped = fit_mask(X)
        assert mask.tolist() == [True, False, False, True]
        assert dropped == {1: 0}   # 共线映射列序 → 保留列序；零方差不进 dropped

    def test_harrell_c_hand_case(self):
        t, e = [1, 2, 3, 4], [1, 0, 1, 0]
        # 事件 i=0（t=1）与 t>1 的三人比：risk 5<10✓ 1<10✓ 100>10✗；
        # 事件 i=2（t=3）与 t>3 比：100>1 ✗ → 2/4 = 0.5
        assert harrell_c(t, e, [10, 5, 1, 100]) == pytest.approx(0.5)
        # 完美排序（事件越早风险越高）→ 1.0
        assert harrell_c(t, e, [4, 3, 2, 1]) == pytest.approx(1.0)

    def test_window_label_censoring_aware_semantics(self):
        t, e = [100, 200, 100, 300], [1, 1, 0, 0]
        keep, y = window_label(t, e, 180)
        # 行0 窗内事件 → 留，y=1；行1 窗后事件 → 留，y=0（窗后按非事件计）；
        # 行2 窗前删失 → 移出风险集；行3 窗后删失 → 留，y=0
        assert keep.tolist() == [True, True, False, True]
        assert y.tolist() == [1, 0, 0]


# ============================================================================
# 3. CoxSurvivalModel（部署工件）
# ============================================================================

class TestCoxSurvivalModelDegenerate:

    def _fit_degenerate(self):
        """列 0 live / 列 1 与 0 完全共线 / 列 2 零方差 / 列 3 live 二元。"""
        rng = np.random.RandomState(3)
        n = 250
        x0 = rng.randn(n)
        X = np.column_stack([
            x0,
            2 * x0 + 1,
            np.full(n, 7.0),
            rng.randint(0, 2, n).astype(float),
        ])
        lam = 0.005 * np.exp(0.8 * x0 + 0.5 * X[:, 3])
        t_ev = -np.log(rng.rand(n)) / lam
        t_c = rng.uniform(50, 500, n)
        t, e = np.minimum(t_ev, t_c), (t_ev <= t_c).astype(int)
        m = CoxSurvivalModel()
        m.fit(t, e, X, ['x0', 'dup', 'const', 'bin'],
              time_column='follow_up_days', cohort='test')
        return m, X, t, e

    def test_fit_survives_degenerate_columns(self):
        m, X, t, e = self._fit_degenerate()
        assert len(m.beta) == 4                      # 全列长
        assert m.beta[2] == 0.0                      # 零方差列系数 0
        assert m.mask.tolist() == [True, False, False, True]
        assert m.dropped == {'dup': 'x0'}            # 名字口径共线映射
        assert m.n_samples == 250 and m.n_events == int(e.sum())
        assert m.time_column == 'follow_up_days' and m.cohort == 'test'

    def test_dropped_columns_contribute_zero(self):
        m, X, t, e = self._fit_degenerate()
        X2 = X.copy()
        X2[:, 1] = X2[:, 1] * 3 + 10   # 改共线（剔除）列
        X2[:, 2] = 99.0                # 改零方差（剔除）列
        np.testing.assert_array_equal(m.predict(X), m.predict(X2))

    def test_predict_dimension_check(self):
        m, X, t, e = self._fit_degenerate()
        with pytest.raises(ValueError, match='特征维度'):
            m.predict(X[:, :3])

    def test_beta_sign_recovery(self):
        m, X, t, e = self._fit_degenerate()
        assert m.beta[0] > 0 and m.beta[3] > 0   # 真实 β=(0.8, _, _, 0.5)

    def test_results_property_in_memory(self):
        m, X, t, e = self._fit_degenerate()
        assert m.results is not None              # 拟合后内存态可用
        assert m.results.params.shape[0] == 2     # 掩码后 2 列


class TestCoxSurvivalModelRoundtrip:

    def _fitted(self):
        t, e, X = _sim_ph(n=250, seed=8)
        m = CoxSurvivalModel()
        m.fit(t, e, X, ['a', 'b', 'c'],
              time_column='follow_up_days', cohort='rt')
        return m, X

    def test_roundtrip_identical_predictions(self):
        m, X = self._fitted()
        m2 = CoxSurvivalModel.from_dict(m.to_dict())
        np.testing.assert_array_equal(m.predict(X), m2.predict(X))
        assert m2.feature_names == m.feature_names
        assert m2.dropped == m.dropped
        assert m2.time_column == 'follow_up_days'
        assert m2.cohort == 'rt'
        assert m2.results is None                 # _res 不落盘

    def test_to_dict_json_serializable(self):
        m, _ = self._fitted()
        json.dumps(m.to_dict())                   # 不抛即通过


# ============================================================================
# 4. train_survival_model（主管线入口）
# ============================================================================

class TestTrainSurvivalModel:

    def test_success_structure_and_discrimination(self):
        t, e, X = _sim_ph(n=400, seed=9)
        groups = np.repeat(np.arange(40), 10)     # 40 户 → 组感知 CV
        out = train_survival_model(
            t, e, X, ['a', 'b', 'c'], groups=groups,
            time_column='follow_up_days', cohort='sim')
        assert out['success'] is True
        assert isinstance(out['model'], CoxSurvivalModel)
        assert out['n_samples'] == 400
        assert out['n_events'] == int(e.sum())
        assert out['n_features'] == 3
        assert out['n_features_fitted'] == 3
        # 判别恢复：真实信号 β=(0.7, 0.5, -0.4)
        assert out['harrell_c_oof'] > 0.6
        for w in ('auroc_180d', 'auroc_365d'):
            assert out['fixed_window'][w]['point'] > 0.55
        # 组感知 CV 披露
        assert 'StratifiedGroupKFold' in out['protocol']['cv']
        # PH 检验在返回结构中
        assert out['ph_test']['global'] is not None
        # HR 表：全 live + HR > 0
        live = [r for r in out['hr_table'] if r['status'] == 'live']
        assert len(live) == 3
        assert all(r['HR_per_sd'] > 0 for r in live)
        # EPV 口径
        assert out['events_per_variable'] == pytest.approx(
            out['n_events'] / out['n_features_fitted'])

    def test_too_few_events_gate(self):
        rng = np.random.RandomState(2)
        n = 60
        X = rng.randn(n, 2)
        t = rng.uniform(10, 400, n)
        e = np.zeros(n, int)
        e[:6] = 1                                  # 6 < 10
        out = train_survival_model(t, e, X, ['a', 'b'])
        assert out['success'] is False
        assert out['error_type'] == 'too_few_events'
        assert 'model' not in out


# ============================================================================
# 5. train_from_real_data 生存分支（集成）
# ============================================================================

def _survival_frame(n=300, n_families=30, seed=11):
    """合成时间-事件 CSV：13 维基础 + family_id + follow_up_days + tb_outcome。

    生存终点由 age / is_high_risk / cumulative_exposure 驱动（指数基线 +
    均匀删失）；tb_outcome = 事件指示（分类六模型仍可作对照训练）。
    无任何队列签名列 → v1（22 维，A 版直通路径）。
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'age': rng.randint(15, 85, n),
        'cumulative_exposure': rng.uniform(0, 800, n),
        'has_symptoms': rng.randint(0, 2, n),
        'bcg_vaccine': rng.randint(0, 2, n),
        'has_tb': rng.randint(0, 2, n),
        'contact_distance_score': rng.uniform(0, 10, n),
        'ventilation_score': rng.uniform(0, 10, n),
        'is_high_risk': rng.randint(0, 2, n),
        'past_illness': rng.randint(0, 2, n),
        'exposure_setting_score': rng.uniform(0, 10, n),
        'single_duration': rng.uniform(10, 300, n),
        'freq_density': rng.uniform(1, 20, n),
        'time_span': rng.uniform(1, 52, n),
    })
    eta = (0.03 * (df['age'] - 45) + 0.7 * df['is_high_risk']
           + 0.002 * df['cumulative_exposure'])
    lam = 0.0035 * np.exp(eta)
    t_event = -np.log(rng.rand(n)) / lam
    t_cens = rng.uniform(60, 700, n)
    df['follow_up_days'] = np.maximum(
        1, np.round(np.minimum(t_event, t_cens))).astype(int)
    df['tb_outcome'] = (t_event <= t_cens).astype(int)
    assert df['tb_outcome'].sum() >= 30, '合成事件数不足（种子异常）'
    df['family_id'] = np.repeat(np.arange(n_families), n // n_families)
    return df


def _contact(**extra):
    base = {
        'age': 40, 'cumulative_exposure': 300, 'has_symptoms': 0,
        'bcg_vaccine': 1, 'has_tb': 1, 'contact_distance_score': 5,
        'ventilation_score': 5, 'is_high_risk': 0, 'past_illness': 0,
        'exposure_setting_score': 5, 'single_duration': 120,
        'freq_density': 8, 'time_span': 12,
    }
    base.update(extra)
    return base


@pytest.fixture(scope='module')
def surv_trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('surv_p5')
    csv = tmp / 'sim_surv.csv'
    _survival_frame().to_csv(csv, index=False)
    predictor = MLRiskPredictor()
    result = train_from_real_data(predictor, str(csv),
                                  target_column='tb_outcome')
    return predictor, result, tmp


class TestPipelineSurvivalBranch:

    def test_branch_mounts_model(self, surv_trained):
        predictor, result, _ = surv_trained
        assert result['success'] is True
        surv = result['survival']
        assert surv['success'] is True
        assert surv['n_events'] >= 10
        assert 0.5 < surv['harrell_c_oof'] <= 1.0
        assert surv['fixed_window']['auroc_180d']['point'] is not None
        m = predictor.survival_model
        assert m is not None
        assert m.time_column == 'follow_up_days'
        assert set(m.feature_names) == set(predictor.ALL_FEATURE_NAMES)

    def test_predict_risk_reports_survival_cox(self, surv_trained):
        from tb_risk.scoring.ml.evaluation import predict_risk
        predictor, _, _ = surv_trained
        out = predict_risk(predictor, _contact(), contact_type='family')
        assert out is not None
        sc = out.get('survival_cox')
        assert sc is not None, f'predict_risk 未报 survival_cox: {list(out)}'
        assert isinstance(sc['risk_score'], float)
        assert sc['time_column'] == 'follow_up_days'

    def test_predict_risk_monotone_in_risk_factors(self, surv_trained):
        from tb_risk.scoring.ml.evaluation import predict_risk
        predictor, _, _ = surv_trained
        hi = predict_risk(predictor, _contact(
            age=75, is_high_risk=1, cumulative_exposure=700),
            contact_type='family')['survival_cox']['risk_score']
        lo = predict_risk(predictor, _contact(
            age=22, is_high_risk=0, cumulative_exposure=10),
            contact_type='family')['survival_cox']['risk_score']
        assert hi > lo, (hi, lo)

    def test_checkpoint_roundtrip(self, surv_trained):
        predictor, _, tmp = surv_trained
        ckpt = os.path.join(str(tmp), 'surv_ckpt.joblib')
        assert predictor.save_model(ckpt)
        p2 = MLRiskPredictor()
        assert p2.load_model(ckpt)
        m2 = getattr(p2, 'survival_model', None)
        assert m2 is not None, 'survival_model 未随 checkpoint 恢复'
        assert m2.time_column == 'follow_up_days'
        assert len(m2.feature_names) == len(predictor.ALL_FEATURE_NAMES)
        from tb_risk.scoring.ml.evaluation import predict_risk
        out = predict_risk(p2, _contact(), contact_type='family')
        assert out.get('survival_cox') is not None

    def test_no_time_column_no_mount(self, tmp_path):
        csv = tmp_path / 'no_time.csv'
        _survival_frame(n=150, n_families=15, seed=77).drop(
            columns=['follow_up_days']).to_csv(csv, index=False)
        predictor = MLRiskPredictor()
        result = train_from_real_data(predictor, str(csv),
                                      target_column='tb_outcome')
        assert result['success'] is True
        assert 'survival' not in result
        assert not hasattr(predictor, 'survival_model')

    def test_stale_mount_cleaned_on_retrain_without_time(self, tmp_path):
        """带时间列训练挂载后，同一 predictor 重训无时间列数据 → 清理。"""
        csv_t = tmp_path / 'with_time.csv'
        _survival_frame(n=150, n_families=15, seed=78).to_csv(
            csv_t, index=False)
        csv_n = tmp_path / 'no_time.csv'
        _survival_frame(n=150, n_families=15, seed=79).drop(
            columns=['follow_up_days']).to_csv(csv_n, index=False)
        predictor = MLRiskPredictor()
        r1 = train_from_real_data(predictor, str(csv_t),
                                  target_column='tb_outcome')
        assert r1['survival']['success'] is True
        assert hasattr(predictor, 'survival_model')
        r2 = train_from_real_data(predictor, str(csv_n),
                                  target_column='tb_outcome')
        assert r2['success'] is True
        assert 'survival' not in r2
        assert not hasattr(predictor, 'survival_model')


# ============================================================================
# 分层 Cox 重拟合（缺陷2，2026-09-17：PH 违反处置）
# ============================================================================

class TestStratifiedRefit:

    def test_harrell_c_stratified_only_within_pairs(self):
        """分层 C 只统计同层对：手工构造两层，全对集与同层对集可分辨。"""
        t = np.array([10., 20., 30., 40.])
        e = np.array([1, 0, 1, 0])
        risk = np.array([2., 1., 1., 2.])
        strata = np.array(['a', 'a', 'b', 'b'])
        # 全对集：可比 4 对（10 vs 20/30/40 + 30 vs 40），一致 2.5
        assert harrell_c(t, e, risk) == pytest.approx(2.5 / 4)
        # 同层：a 层 1 对一致，b 层 1 对不一致 → 0.5
        from tb_risk.scoring.ml.survival_cox import harrell_c_stratified
        assert harrell_c_stratified(t, e, risk, strata) == pytest.approx(0.5)

    def test_harrell_c_stratified_row_mismatch_raises(self):
        from tb_risk.scoring.ml.survival_cox import harrell_c_stratified
        with pytest.raises(ValueError):
            harrell_c_stratified([1., 2.], [1, 0], [0.5, 0.6], ['a'])

    def test_refit_on_ph_data_reports_delta(self):
        """PH 满足合成数据上重拟合可运行：结构字段齐全、ΔC 有限。"""
        from tb_risk.scoring.ml.survival_cox import stratified_cox_refit
        t, e, X = _sim_ph(n=300, seed=42)
        names = ['x0', 'v1', 'v2']
        out = stratified_cox_refit(t, e, X, names, strata_vars=['v1'],
                                   folds=3, seed=42)
        assert out['success'] is True
        assert out['strata_vars'] == ['v1']
        assert out['n_strata'] == 2
        assert sum(s['n'] for s in out['per_stratum_events']) == 300
        assert sum(s['n_events'] for s in out['per_stratum_events']) \
            == int(e.sum())
        assert np.isfinite(out['c_static_within_strata_pairs'])
        assert np.isfinite(out['c_stratified_within_strata_pairs'])
        assert out['delta_c'] == pytest.approx(
            out['c_stratified_within_strata_pairs']
            - out['c_static_within_strata_pairs'])
        assert 'decision_rule' in out['protocol']
        # PH 满足数据上分层不应破坏判别（容差放宽：n=300 抽样噪声）
        assert out['c_stratified_within_strata_pairs'] > 0.5

    def test_refit_drops_collinear_with_stratum_var(self):
        """与分层变量完全共线的协变量入场会使层内零方差 → 预过滤剔除。"""
        from tb_risk.scoring.ml.survival_cox import stratified_cox_refit
        t, e, X0 = _sim_ph(n=300, seed=7)
        # v_dup 与 v1 完全共线（peru highrisk_comorbid↔past_illness 同构）
        X = np.column_stack([X0, X0[:, 1]])
        names = ['x0', 'v1', 'v2', 'v_dup']
        out = stratified_cox_refit(t, e, X, names, strata_vars=['v1'],
                                   folds=3, seed=42)
        assert out['success'] is True
        dropped = [d['feature'] for d in out['strata_collinear_dropped']]
        assert 'v_dup' in dropped

    def test_refit_rejects_nonbinary_strata_var(self):
        from tb_risk.scoring.ml.survival_cox import stratified_cox_refit
        t, e, X = _sim_ph(n=200, seed=3)
        with pytest.raises(ValueError, match='非二元'):
            stratified_cox_refit(t, e, X, ['x0', 'v1', 'v2'],
                                 strata_vars=['x0'], folds=3)

    def test_refit_rejects_unknown_strata_var(self):
        from tb_risk.scoring.ml.survival_cox import stratified_cox_refit
        t, e, X = _sim_ph(n=200, seed=3)
        with pytest.raises(ValueError, match='不在特征名中'):
            stratified_cox_refit(t, e, X, ['x0', 'v1', 'v2'],
                                 strata_vars=['nope'], folds=3)
