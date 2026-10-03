#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题4：类别不平衡与概率口径 — 单元测试

覆盖：
1. 阳性率双口径常量（密接人群 2.85% vs 全人群 100/10万）单一真值源；
2. PPV 贝叶斯换算（按目标人群阳性率）；
3. 截断点操作特性（敏感度/特异度）与决策参考表构建；
4. 训练流程接入的真实数据校准（Platt/isotonic 自动选择 + Brier/AUROC 前后对照）；
5. 训练日志自动携带双口径标注。

文献：
- Bayes PPV = sens·π / (sens·π + (1-spec)·(1-π))（标准筛查换算）
- Platt J. (1999)；Zadrozny & Elkan (2002)：小样本用 sigmoid，
  大样本 isotonic（项目硬约束：阳性样本 < 200 用 sigmoid；
  2026-09-05 地基审计由 500 对齐记忆约束）
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk import constants as C  # noqa: E402
from tb_risk.core import ppv as ppv_mod  # noqa: E402


# ============================================================================
# 1. 阳性率双口径常量（单一真值源）
# ============================================================================

class TestPrevalenceCalibers:
    """两套阳性率口径常量：密接人群 vs 全人群。"""

    def test_two_calibers_defined(self):
        calibers = C.PREVALENCE_CALIBERS
        assert set(calibers.keys()) == {'close_contact', 'general_population'}

    def test_close_contact_value(self):
        cc = C.PREVALENCE_CALIBERS['close_contact']
        assert cc['value'] == pytest.approx(0.0285)
        assert cc['display'] == '2.85%'
        assert 'label' in cc and 'source' in cc

    def test_general_population_value(self):
        gp = C.PREVALENCE_CALIBERS['general_population']
        assert gp['value'] == pytest.approx(0.001)      # 100/10万
        assert gp['display'] == '100/10万'
        assert 'label' in gp and 'source' in gp

    def test_mixing_warning_exists(self):
        assert isinstance(C.PREVALENCE_CALIBER_MIXING_WARNING, str)
        assert '混用' in C.PREVALENCE_CALIBER_MIXING_WARNING


# ============================================================================
# 2. PPV 贝叶斯换算
# ============================================================================

class TestPpvAtCutoff:
    """PPV = sens·π / (sens·π + (1-spec)·(1-π))。"""

    def test_known_value(self):
        # sens=0.9, spec=0.9, π=0.0285:
        # PPV = 0.9*0.0285 / (0.9*0.0285 + 0.1*0.9715)
        num = 0.9 * 0.0285
        den = num + 0.1 * (1 - 0.0285)
        assert ppv_mod.ppv_at_cutoff(0.9, 0.9, 0.0285) == pytest.approx(num / den)

    def test_perfect_classifier(self):
        assert ppv_mod.ppv_at_cutoff(1.0, 1.0, 0.001) == pytest.approx(1.0)

    def test_useless_classifier_returns_prevalence(self):
        # sens=spec（无判别力）→ PPV=π
        assert ppv_mod.ppv_at_cutoff(0.5, 0.5, 0.0285) == pytest.approx(0.0285)

    def test_lower_prevalence_lower_ppv(self):
        ppv_cc = ppv_mod.ppv_at_cutoff(0.9, 0.9, 0.0285)
        ppv_gp = ppv_mod.ppv_at_cutoff(0.9, 0.9, 0.001)
        assert ppv_gp < ppv_cc

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            ppv_mod.ppv_at_cutoff(1.5, 0.9, 0.01)
        with pytest.raises(ValueError):
            ppv_mod.ppv_at_cutoff(0.9, -0.1, 0.01)
        with pytest.raises(ValueError):
            ppv_mod.ppv_at_cutoff(0.9, 0.9, 0.0)
        with pytest.raises(ValueError):
            ppv_mod.ppv_at_cutoff(0.9, 0.9, 1.0)


# ============================================================================
# 3. 截断点操作特性与决策参考表
# ============================================================================

class TestOperatingPoint:
    """排序+截断点：top-k% 截断的敏感度/特异度。"""

    def test_perfect_separation(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        op = ppv_mod.operating_point(scores, labels, top_fraction=0.5)
        assert op['sensitivity'] == pytest.approx(1.0)
        assert op['specificity'] == pytest.approx(1.0)
        assert op['threshold'] == pytest.approx(0.2)  # top-50% 覆盖到 0.2

    def test_random_scores(self):
        rng = np.random.RandomState(42)
        scores = rng.rand(200).tolist()
        labels = rng.randint(0, 2, 200).tolist()
        op = ppv_mod.operating_point(scores, labels, top_fraction=0.1)
        assert 0.0 <= op['sensitivity'] <= 1.0
        assert 0.0 <= op['specificity'] <= 1.0
        assert op['n_flagged'] == 20

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            ppv_mod.operating_point([0.5], [1], top_fraction=1.5)


class TestDecisionReference:
    """决策参考表：双口径 PPV + 口径不可混用提示。"""

    def _scores_labels(self):
        rng = np.random.RandomState(0)
        n = 400
        scores = rng.rand(n)
        labels = (scores + rng.normal(0, 0.4, n) > 0.8).astype(int).tolist()
        return scores.tolist(), labels

    def test_structure(self):
        scores, labels = self._scores_labels()
        ref = ppv_mod.build_decision_reference(scores, labels, top_fraction=0.1)
        assert ref['decision_form'] == 'ranking_cutoff'
        assert 'operating_point' in ref
        assert set(ref['ppv'].keys()) == {'close_contact', 'general_population'}
        for key in ('close_contact', 'general_population'):
            block = ref['ppv'][key]
            assert block['prevalence'] == pytest.approx(
                C.PREVALENCE_CALIBERS[key]['value'])
            assert 0.0 <= block['ppv'] <= 1.0
        assert '不可混用' in ref['note']

    def test_ppv_consistent_with_bayes(self):
        scores, labels = self._scores_labels()
        ref = ppv_mod.build_decision_reference(scores, labels, top_fraction=0.1)
        op = ref['operating_point']
        expected = ppv_mod.ppv_at_cutoff(
            op['sensitivity'], op['specificity'], 0.0285)
        assert ref['ppv']['close_contact']['ppv'] == pytest.approx(expected, rel=1e-6)


class TestDecisionReferenceLocalCaliber:
    """第三口径（缺陷1，2026-09-17）：训练队列实际阳性率 PPV。

    固定参照口径（密接 2.85%）在低患病率队列上高估 PPV 4-5 倍
    （kenya 实际 0.53%：密接口径 21% vs 本地口径约 4.6%）。
    """

    def _scores_labels(self):
        rng = np.random.RandomState(1)
        n = 600
        scores = rng.rand(n)
        labels = (scores + rng.normal(0, 0.35, n) > 0.75).astype(int).tolist()
        return scores.tolist(), labels

    def test_local_caliber_block_added(self):
        scores, labels = self._scores_labels()
        ref = ppv_mod.build_decision_reference(
            scores, labels, top_fraction=0.1,
            local_prevalence=0.0053, local_label='kenya 训练样本阳性率')
        assert set(ref['ppv'].keys()) == {
            'close_contact', 'general_population', 'cohort_local'}
        lc = ref['ppv']['cohort_local']
        assert lc['prevalence'] == pytest.approx(0.0053)
        assert lc['label'] == 'kenya 训练样本阳性率'
        op = ref['operating_point']
        expected = ppv_mod.ppv_at_cutoff(
            op['sensitivity'], op['specificity'], 0.0053)
        assert lc['ppv'] == pytest.approx(expected, rel=1e-6)
        # 缺陷1 判读锚：低患病率队列上固定参照口径高估 PPV
        assert lc['ppv'] < ref['ppv']['close_contact']['ppv']
        assert 'cohort_local' in ref['note']
        assert '首要读数' in lc['note']

    def test_no_local_caliber_without_param(self):
        scores, labels = self._scores_labels()
        ref = ppv_mod.build_decision_reference(scores, labels, top_fraction=0.1)
        assert 'cohort_local' not in ref['ppv']
        assert 'cohort_local' not in ref['note']

    def test_invalid_local_prevalence_raises(self):
        scores, labels = self._scores_labels()
        with pytest.raises(ValueError):
            ppv_mod.build_decision_reference(
                scores, labels, local_prevalence=0.0)
        with pytest.raises(ValueError):
            ppv_mod.build_decision_reference(
                scores, labels, local_prevalence=1.0)

    def test_summary_includes_local_segment(self):
        scores, labels = self._scores_labels()
        ref = ppv_mod.build_decision_reference(
            scores, labels, local_prevalence=0.0053)
        summary = ppv_mod.format_decision_reference_summary(ref)
        assert '队列实际(0.53%)' in summary
        ref2 = ppv_mod.build_decision_reference(scores, labels)
        assert '队列实际' not in ppv_mod.format_decision_reference_summary(ref2)


# ============================================================================
# 4. 真实数据校准接入训练流程
# ============================================================================

class _StubPredictor:
    """最小 predictor 桩：models / model_performance / is_trained。"""

    ALL_FEATURE_NAMES = [f'f{i}' for i in range(5)]

    def __init__(self, models):
        self.models = dict(models)
        self.model_performance = {
            k: {'name': k, 'AUROC': 0.8, 'AUPRC': 0.3}
            for k in models
        }
        self.is_trained = True
        self.calibrated_models = None
        self._is_calibrated = False
        self._calibration_method = None


def _make_data(n=600, pos_rate=0.15, seed=42):
    """合成可分数据：阳性样本分数系统性偏高。"""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 5)
    logit = 3.0 * X[:, 0] + 2.0 * X[:, 1] - 2.5 + rng.normal(0, 0.5, n)
    y = (logit > 0).astype(int)
    if y.sum() < 10:  # 保证有足够阳性
        y[:10] = 1
    return X, y


@pytest.fixture()
def tiny_models():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    X, y = _make_data()
    rf = RandomForestClassifier(n_estimators=10, max_depth=3,
                                random_state=0, class_weight='balanced')
    rf.fit(X, y)
    gb = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=0)
    gb.fit(X, y)
    return {'random_forest': rf, 'gradient_boosting': gb}, X, y


class TestCalibrateModelsOnData:
    """calibrate_models_on_data：用真实训练数据（而非合成数据）校准。"""

    def test_calibrates_and_records_metrics(self, tiny_models):
        from tb_risk.scoring.ml.calibration import calibrate_models_on_data
        models, X, y = tiny_models
        pred = _StubPredictor(models)
        ok = calibrate_models_on_data(pred, X, y, method='auto',
                                      random_state=42)
        assert ok is True
        assert pred._is_calibrated is True
        assert set(pred.calibrated_models.keys()) == set(models.keys())
        for key in models:
            cal = pred.model_performance[key]['calibration']
            # 硬约束：Brier 与 AUROC 前后都记录（检出判别力损失）
            for field in ('method', 'n_positive', 'brier_before', 'brier_after',
                          'auroc_before', 'auroc_after'):
                assert field in cal, f'{key} 缺少 {field}'
            assert 0.0 <= cal['auroc_before'] <= 1.0
            assert 0.0 <= cal['auroc_after'] <= 1.0
            assert cal['brier_before'] >= 0.0

    def test_auto_method_selection(self, tiny_models):
        """阳性 < 200 → Platt(sigmoid)；≥ 200 → isotonic（项目硬约束；
        2026-09-05 地基审计：阈值由 500 对齐记忆约束 200）。"""
        from sklearn.ensemble import RandomForestClassifier
        from tb_risk.scoring.ml.calibration import calibrate_models_on_data
        X, y = _make_data(n=120)  # 55 阳性 < 200
        assert y.sum() < 200
        rf = RandomForestClassifier(n_estimators=10, max_depth=3,
                                    random_state=0, class_weight='balanced')
        rf.fit(X, y)
        pred = _StubPredictor({'random_forest': rf})
        calibrate_models_on_data(pred, X, y, method='auto', random_state=42)
        cal = pred.model_performance['random_forest']['calibration']
        assert cal['method'] == 'sigmoid'
        assert cal['n_positive'] < 200

    def test_auto_isotonic_in_200_499_range(self, tiny_models):
        """地基审计回归锁定：200-499 阳性区间旧代码误用 sigmoid，
        对齐约束（<200 才用 sigmoid）后应为 isotonic。"""
        from tb_risk.scoring.ml.calibration import calibrate_models_on_data
        models, X, y = tiny_models  # 311 阳性 ∈ [200, 500)
        assert 200 <= y.sum() < 500
        pred = _StubPredictor(models)
        calibrate_models_on_data(pred, X, y, method='auto', random_state=42)
        cal = pred.model_performance['random_forest']['calibration']
        assert cal['method'] == 'isotonic'

    def test_auto_isotonic_when_many_positives(self, tiny_models):
        from tb_risk.scoring.ml.calibration import calibrate_models_on_data
        models, _, _ = tiny_models
        rng = np.random.RandomState(1)
        X = rng.rand(1200, 5)
        y = np.zeros(1200, dtype=int)
        y[:600] = 1  # 600 阳性 → isotonic
        pred = _StubPredictor(models)
        calibrate_models_on_data(pred, X, y, method='auto', random_state=42)
        cal = pred.model_performance['random_forest']['calibration']
        assert cal['method'] == 'isotonic'

    def test_returns_eval_arrays_for_decision_reference(self, tiny_models):
        """校准时保留泄漏无关的测试半区概率，供决策参考表使用。"""
        from tb_risk.scoring.ml.calibration import calibrate_models_on_data
        models, X, y = tiny_models
        pred = _StubPredictor(models)
        calibrate_models_on_data(pred, X, y, method='auto', random_state=42)
        ev = getattr(pred, 'calibration_eval', None)
        assert isinstance(ev, dict) and ev
        entry = ev['random_forest']
        assert len(entry['y_test']) == len(entry['p_after'])
        assert entry['leak_free'] is True


class TestCalibrationSameCaliberBaseline:
    """缺陷2回归（2026-09-16）：校准退化拦截的 before/after 必须同口径。

    旧实现 before = 全量数据训练的基模型直接预测 eval 半区
    （eval 样本已入训练，in-sample 乐观口径），记忆化树模型
    AUROC 虚高 0.03-0.08，乐观差距被误判为校准退化——kenya
    v4 主管线 run 实测：五个树模型全部被误拦截（auroc_before
    0.97-0.9995 全部高于 CV OOF，auroc_after 与 CV OOF 一致），
    logistic 因 in-sample≈OOF 未触发。修复后 before 为 cal 半区
    K 折重拟合未校准克隆的平均预测（复刻 CalibratedClassifierCV
    ensemble=True 内部折结构），两臂仅差校准器。
    """

    def _memorizing_rf_data(self):
        """强信号数据 + 深树 RF：in-sample AUROC ≈ 1.0（记忆化前提）。"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        rng = np.random.RandomState(7)
        n = 1000
        X = rng.rand(n, 5)
        logit = 4.0 * X[:, 0] - 2.0 + rng.normal(0, 0.8, n)
        y = (logit > 0).astype(int)
        assert 200 <= y.sum()  # ≥200 阳性 → isotonic 分支
        rf = RandomForestClassifier(n_estimators=60, random_state=0,
                                    class_weight='balanced')
        rf.fit(X, y)
        in_sample = roc_auc_score(y, rf.predict_proba(X)[:, 1])
        return rf, X, y, in_sample

    def test_before_is_leak_free_not_in_sample(self):
        """记忆化 RF 的 before 不得是 in-sample 口径（≈1.0）。"""
        from tb_risk.scoring.ml.calibration import calibrate_models_on_data
        rf, X, y, in_sample = self._memorizing_rf_data()
        assert in_sample > 0.99  # 前提坐实：该模型确实记忆化

        pred = _StubPredictor({'random_forest': rf})
        ok = calibrate_models_on_data(pred, X, y, method='auto',
                                      random_state=42)
        assert ok is True
        cal = pred.model_performance['random_forest']['calibration']
        assert cal['before_caliber'] == 'cal_half_kfold_leak_free'
        # 泄漏无关基线：远离 in-sample 水平（旧实现在此返回 ≈1.0）
        assert cal['auroc_before'] < in_sample - 0.02
        # 同口径对照下校准不构成退化（差值在容差内）→ 不误拦截
        assert cal['calibration_degraded'] is False
        # 部署的是校准器包裹而非回退原模型
        assert pred.calibrated_models['random_forest'] is not rf

    def test_interceptor_still_fires_on_genuine_degradation(
            self, monkeypatch):
        """真退化仍拦截：auroc_after 低于 before 超容差 → 回退原模型。"""
        from sklearn.ensemble import RandomForestClassifier
        from tb_risk.scoring.ml import calibration as cal_mod
        from tb_risk.scoring.ml.calibration import calibrate_models_on_data
        rng = np.random.RandomState(3)
        n = 600
        X = rng.rand(n, 5)
        y = (2.0 * X[:, 0] + rng.normal(0, 0.8, n) > 0.9).astype(int)
        if y.sum() < 10:
            y[:15] = 1
        rf = RandomForestClassifier(n_estimators=10, max_depth=3,
                                    random_state=0)
        rf.fit(X, y)
        pred = _StubPredictor({'random_forest': rf})

        calls = {'n': 0}

        def fake_auc(y_true, y_score, *a, **kw):
            calls['n'] += 1
            # 代码顺序：第 1 次 = auroc_before，第 2 次 = auroc_after
            return 0.90 if calls['n'] == 1 else 0.80

        monkeypatch.setattr(cal_mod, 'roc_auc_score', fake_auc)
        ok = calibrate_models_on_data(pred, X, y, method='auto',
                                      random_state=42)
        assert ok is True
        cal = pred.model_performance['random_forest']['calibration']
        assert cal['auroc_before'] == pytest.approx(0.90)
        assert cal['auroc_after'] == pytest.approx(0.80)
        assert cal['calibration_degraded'] is True
        assert pred.calibrated_models['random_forest'] is rf  # 回退原模型


class TestCalibrationFeatureNameWarningCleanup:
    """P1-a（2026-09-20）：校准路径的 LGBM 特征名告警清理。

    旧实现 ndarray 直入 CalibratedClassifierCV：LGBM 在 ndarray 拟合时
    写入自动特征名（Column_N），内部折预测 + calibrated.predict_proba
    各触发一条告警（实测每模型 10 条/5 折）。修复：校准输入统一转带
    列名的 DataFrame 再切分（列名优先序 X.columns > 模型
    feature_names_in_ > x0..xN 生成名），两臂同口径、数值不变。
    """

    def _lgbm_setup(self, seed=5, n=400, as_dataframe=False):
        from lightgbm import LGBMClassifier
        rng = np.random.RandomState(seed)
        X = rng.rand(n, 5)
        logit = 3.0 * X[:, 0] - 1.5 + rng.normal(0, 0.6, n)
        y = (logit > 0).astype(int)
        if y.sum() < 10 or (1 - y).sum() < 10:
            y[:20] = 1
            y[-20:] = 0
        model = LGBMClassifier(n_estimators=30, verbose=-1,
                               class_weight='balanced')
        if as_dataframe:
            import pandas as pd
            X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
            model.fit(X_df, y)
            return model, X_df, y
        model.fit(X, y)
        return model, X, y

    def _warn_count(self, fn):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fn()
            return [x for x in w if 'feature name' in str(x.message)]

    def test_ndarray_input_no_warnings(self):
        from tb_risk.scoring.ml.calibration import calibrate_models_on_data
        model, X, y = self._lgbm_setup()
        pred = _StubPredictor({'lightgbm': model})
        hits = self._warn_count(
            lambda: calibrate_models_on_data(pred, X, y, method='auto',
                                             random_state=42))
        assert hits == [], [str(h.message) for h in hits]
        assert pred._is_calibrated is True

    def test_dataframe_input_no_warnings(self):
        from tb_risk.scoring.ml.calibration import calibrate_models_on_data
        model, X, y = self._lgbm_setup(as_dataframe=True)
        pred = _StubPredictor({'lightgbm': model})
        hits = self._warn_count(
            lambda: calibrate_models_on_data(pred, X, y, method='auto',
                                             random_state=42))
        assert hits == [], [str(h.message) for h in hits]

    def test_dataframe_wrap_values_unchanged(self):
        """ndarray→DataFrame 包装不改变校准前后指标（特征名不参与计算）。"""
        from tb_risk.scoring.ml.calibration import calibrate_models_on_data
        results = {}
        for as_df in (False, True):
            model, X, y = self._lgbm_setup(seed=9, as_dataframe=as_df)
            pred = _StubPredictor({'lightgbm': model})
            calibrate_models_on_data(pred, X, y, method='auto',
                                     random_state=42)
            cal = pred.model_performance['lightgbm']['calibration']
            results[as_df] = (cal['auroc_before'], cal['auroc_after'],
                              cal['brier_before'], cal['brier_after'])
        for a, b in zip(results[False], results[True]):
            assert a == pytest.approx(b, abs=1e-12)


class TestCalibrationBeforeArmFoldStructure:
    """P1-d（2026-09-20）：校准拦截对照口径复核——锁定 before 臂
    精确复刻 CalibratedClassifierCV(ensemble=True) 内部折结构。

    sklearn 1.6.1 源码（calibration.py::fit）：ensemble=True 对
    cv.split(X, y) 的每个 (train, test) 折 clone 基模型在 train 上
    重拟合、校准器拟合于 test 折预测，predict_proba 对 K 个校准器
    取平均。before 臂 = 同一 cv 折重拟合未校准克隆取平均 → 两臂仅差
    校准器。本测试手工复刻该结构，断言 auroc_before 逐位一致；若库
    代码改变折结构/切分/平均方式（口径漂移），数值立即失配。
    """

    def test_before_arm_matches_manual_replication(self):
        import pandas as pd
        from sklearn.base import clone
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import (StratifiedKFold,
                                             StratifiedShuffleSplit)
        from tb_risk.scoring.ml.calibration import calibrate_models_on_data

        rng = np.random.RandomState(13)
        n = 600
        X = rng.rand(n, 5)
        logit = 3.0 * X[:, 0] + 2.0 * X[:, 1] - 3.0 + rng.normal(0, 0.6, n)
        y = (logit > 0).astype(int)
        X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        rf = RandomForestClassifier(n_estimators=30, random_state=0,
                                    class_weight='balanced')
        rf.fit(X_df, y)
        pred = _StubPredictor({'random_forest': rf})
        assert calibrate_models_on_data(pred, X_df, y, method='auto',
                                        random_state=42) is True
        cal = pred.model_performance['random_forest']['calibration']
        assert cal['before_caliber'] == 'cal_half_kfold_leak_free'

        # 手工复刻（与库代码逐步同构：同 split、同 cv 折、同平均）
        X_arr = np.asarray(X_df)
        y_arr = np.asarray(y).astype(int)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5,
                                          random_state=42)
        cal_idx, eval_idx = next(splitter.split(X_arr, y_arr))
        X_cal = X_df.iloc[cal_idx]
        y_cal, y_eval = y_arr[cal_idx], y_arr[eval_idx]
        X_eval = X_df.iloc[eval_idx]
        min_class = int(min(np.bincount(y_cal)))
        n_splits = max(2, min(5, min_class))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=42)
        p_manual = np.zeros(len(y_eval), dtype=float)
        n_folds = 0
        for fold_tr, _ in cv.split(X_cal, y_cal):
            ref = clone(rf)
            ref.fit(X_cal.iloc[fold_tr], y_cal[fold_tr])
            p_manual += ref.predict_proba(X_eval)[:, 1]
            n_folds += 1
        p_manual /= max(n_folds, 1)

        assert roc_auc_score(y_eval, p_manual) == pytest.approx(
            cal['auroc_before'], abs=1e-12)
        # 泄漏无关结构性前提：cal/eval 半区不相交，且折训练索引是
        # cal 半区内部位置（< len(y_cal)）——eval 半区行从不进入拟合
        assert not set(cal_idx.tolist()) & set(eval_idx.tolist())
        assert int(fold_tr.max()) < len(y_cal)


# ============================================================================
# 5. 训练日志自动携带双口径标注
# ============================================================================

class TestTrainingLogCalibers:
    """TrainingLogger.log 自动写入 prevalence_calibers（训练报告口径统一）。"""

    def test_log_stamps_calibers(self, tmp_path, monkeypatch):
        monkeypatch.setenv('TB_RISK_ARCHIVE_DIR', str(tmp_path))
        from tb_risk.gui.training_panel.training_log import TrainingLogger
        logger = TrainingLogger()
        entry = logger.log(model_type='ml', params={'n_samples': 100},
                           metrics={'AUROC': 0.8}, status='success',
                           dataset={'source': 'test'})
        stamped = entry.prevalence_calibers
        assert set(stamped.keys()) == {'close_contact', 'general_population'}
        assert stamped['close_contact']['value'] == pytest.approx(0.0285)
        assert stamped['general_population']['value'] == pytest.approx(0.001)

        # 归档 JSON 也包含
        import json
        result_files = list((tmp_path / 'results').glob('*.json'))
        assert result_files, '训练档案未写入 results/'
        with open(result_files[0], 'r', encoding='utf-8') as f:
            archived = json.load(f)
        assert 'prevalence_calibers' in archived

    def test_old_records_without_field_still_load(self):
        from tb_risk.gui.training_panel.training_log import TrainingLogEntry
        entry = TrainingLogEntry.from_dict({'timestamp': 'x', 'model_type': 'ml'})
        assert entry.prevalence_calibers == {}


# ============================================================================
# 6. 输出侧：排序分位与决策参考摘要（界面/报告共用纯函数）
# ============================================================================

class TestPercentileRanks:
    """percentile_ranks：分数 → 群内分位（排序决策形式的数据基础）。"""

    def test_distinct_scores(self):
        # [0.1, 0.2, 0.8, 0.9] → [25, 50, 75, 100]（最高分 = 100%）
        ranks = ppv_mod.percentile_ranks([0.1, 0.2, 0.8, 0.9])
        assert ranks == [25.0, 50.0, 75.0, 100.0]

    def test_tied_scores_share_percentile(self):
        ranks = ppv_mod.percentile_ranks([0.5, 0.5, 0.9])
        # 两个 0.5 并列：占第 1、2 秩 → 平均秩 1.5 → 50%
        assert ranks[0] == pytest.approx(50.0)
        assert ranks[1] == pytest.approx(50.0)
        assert ranks[2] == pytest.approx(100.0)

    def test_preserves_order_and_range(self):
        rng = np.random.RandomState(7)
        scores = rng.rand(50).tolist()
        ranks = ppv_mod.percentile_ranks(scores)
        assert len(ranks) == 50
        assert all(0.0 < r <= 100.0 for r in ranks)
        # 分位与分数单调一致（严格不等号下方向相同）
        for i in range(50):
            for j in range(50):
                if scores[i] < scores[j]:
                    assert ranks[i] <= ranks[j]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ppv_mod.percentile_ranks([])


class TestFormatDecisionReferenceSummary:
    """format_decision_reference_summary：界面/报告一行的中文决策口径摘要。"""

    def _reference(self):
        scores, labels = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05], \
                         [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
        return ppv_mod.build_decision_reference(scores, labels, top_fraction=0.2)

    def test_summary_contains_core_elements(self):
        summary = ppv_mod.format_decision_reference_summary(self._reference())
        # 决策形式 + 截断覆盖 + 双口径 PPV + 不可混用警示
        assert '排序' in summary
        assert '10' in summary or '20' in summary  # 覆盖人数信息
        assert '2.85%' in summary
        assert '100/10万' in summary
        assert '不可混用' in summary

    def test_summary_is_single_line(self):
        summary = ppv_mod.format_decision_reference_summary(self._reference())
        assert '\n' not in summary

    def test_summary_includes_operating_point(self):
        summary = ppv_mod.format_decision_reference_summary(self._reference())
        # 敏感度出现在摘要中（操作特性）
        ref = self._reference()
        sens = ref['operating_point']['sensitivity']
        assert f'{sens:.0%}' in summary
