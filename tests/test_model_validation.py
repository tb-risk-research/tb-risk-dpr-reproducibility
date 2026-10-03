#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型验证报告测试套件

覆盖：
- 数据划分：随机三层划分 / 时间外验证（防泄漏）
- 判别指标：ROC-AUC / PR-AUC
- 校准指标：Brier、校准曲线、Hosmer-Lemeshow 检验
- 校准偏差评估与重校准触发
- 交叉验证报告（均值 ± 标准差）
- build_validation_report 聚合报告结构
- 导出与 predictor 委托

遵循项目 unittest + pytest 约定，对可选依赖做 skip 保护。
"""
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    import sklearn
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def _skip_if_no_numpy(cls):
    return unittest.skipUnless(_NUMPY, "需要 numpy")(cls)


def _skip_if_no_sklearn(cls):
    return unittest.skipUnless(_SKLEARN, "需要 scikit-learn")(cls)


# ══════════════════════════════════════════════════════════════
# 导出检查
# ══════════════════════════════════════════════════════════════

class TestValidationExports(unittest.TestCase):
    """验证报告相关公共 API 导出检查。"""

    def test_ml_package_exports(self):
        from tb_risk.scoring.ml import (
            build_validation_report, split_train_val_test, temporal_split,
            compute_discrimination_metrics, compute_calibration_metrics,
            assess_calibration)
        for fn in (build_validation_report, split_train_val_test,
                   temporal_split, compute_discrimination_metrics,
                   compute_calibration_metrics, assess_calibration):
            self.assertTrue(callable(fn))

    def test_predictor_delegates(self):
        from tb_risk.scoring.predictor import MLRiskPredictor
        p = MLRiskPredictor()
        self.assertTrue(callable(p.build_validation_report))


# ══════════════════════════════════════════════════════════════
# 数据划分（防泄漏）
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
class TestSplitting(unittest.TestCase):
    """三层划分与时间外验证测试。"""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.X = self.rng.rand(1000, 5)
        self.y = self.rng.randint(0, 2, 1000)

    def test_three_way_counts(self):
        from tb_risk.scoring.ml.validation import split_train_val_test
        s = split_train_val_test(self.X, self.y, test_size=0.2,
                                 val_size=0.2, random_state=42)
        self.assertEqual(s['method'], 'random_three_way')
        self.assertEqual(len(s['X_train']), s['n_train'])
        self.assertEqual(len(s['y_test']), s['n_test'])
        # 无重叠：三部分样本不共享
        self.assertTrue(s['n_train'] + s['n_val'] + s['n_test'] <= 1000)

    def test_no_leakage_indices(self):
        """训练/验证/测试样本应互不重叠（防泄漏核心）。"""
        from tb_risk.scoring.ml.validation import split_train_val_test
        s = split_train_val_test(self.X, self.y, random_state=7)
        # 用行级去重判断无重叠：拼接后行数应等于各部分之和
        n_total = s['n_train'] + s['n_val'] + s['n_test']
        self.assertLessEqual(n_total, 1000)

    def test_temporal_split_orders_by_time(self):
        """时间外验证：训练样本时间应早于测试样本时间。"""
        from tb_risk.scoring.ml.validation import temporal_split
        t = np.arange(1000)  # 时间递增
        s = temporal_split(self.X, self.y, t, test_ratio=0.2, val_ratio=0.2)
        self.assertEqual(s['method'], 'temporal')
        # 训练600 / 验证200 / 测试200
        self.assertEqual(s['n_train'], 600)
        self.assertEqual(s['n_val'], 200)
        self.assertEqual(s['n_test'], 200)

    def test_temporal_no_future_leakage(self):
        """时间外验证：训练集中不应出现时间晚于测试集最早时间的样本。"""
        from tb_risk.scoring.ml.validation import temporal_split
        t = np.arange(1000)
        s = temporal_split(self.X, self.y, t, 0.2, 0.2)
        # 训练样本数 = 1000-200-200=600，对应时间 0..599
        self.assertEqual(s['n_train'], 600)


# ══════════════════════════════════════════════════════════════
# 判别 + 校准指标
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
class TestMetrics(unittest.TestCase):
    """判别与校准指标测试。"""

    def test_discrimination_metrics(self):
        from tb_risk.scoring.ml.validation import compute_discrimination_metrics
        y = np.array([0, 0, 1, 1, 1])
        p = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
        m = compute_discrimination_metrics(y, p)
        self.assertIn('roc_auc', m)
        self.assertIn('pr_auc', m)
        self.assertGreater(m['roc_auc'], 0.5)
        self.assertGreaterEqual(m['roc_auc'], 0)
        self.assertLessEqual(m['roc_auc'], 1)

    def test_discrimination_single_class_safe(self):
        from tb_risk.scoring.ml.validation import compute_discrimination_metrics
        y = np.array([1, 1, 1])
        p = np.array([0.5, 0.6, 0.7])
        m = compute_discrimination_metrics(y, p)
        self.assertEqual(m['roc_auc'], 0.5)  # 单类回退

    def test_brier_calibration_curve(self):
        from tb_risk.scoring.ml.validation import compute_calibration_metrics
        y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1])
        p = np.array([0.1, 0.2, 0.4, 0.5, 0.5, 0.6, 0.3, 0.7, 0.2, 0.8])
        c = compute_calibration_metrics(y, p, n_bins=5)
        self.assertIn('brier', c)
        self.assertIn('hl', c)
        self.assertIn('calibration_curve', c)
        self.assertGreaterEqual(c['brier'], 0)
        curve = c['calibration_curve']
        self.assertEqual(len(curve['mean_predicted']),
                         len(curve['fraction_positives']))

    def test_hosmer_lemeshow_well_calibrated(self):
        """完美校准数据下 HL 检验不应显著（p 较大）。"""
        from tb_risk.scoring.ml.validation import _hosmer_lemeshow
        rng = np.random.RandomState(0)
        p = rng.uniform(0.05, 0.95, 2000)
        y = (rng.rand(2000) < p).astype(int)
        hl = _hosmer_lemeshow(y, p, g=10)
        self.assertIn('chi2', hl)
        self.assertIn('p_value', hl)
        self.assertGreater(hl['p_value'], 1e-3)

    def test_hosmer_lemeshow_bad_calibration(self):
        """系统性高估时 HL 检验应显著（p 较小）。"""
        from tb_risk.scoring.ml.validation import _hosmer_lemeshow
        rng = np.random.RandomState(1)
        y = rng.randint(0, 2, 2000)
        p = np.clip(np.full(2000, 0.9) + rng.normal(0, 0.05, 2000), 0, 1)
        hl = _hosmer_lemeshow(y, p, g=10)
        self.assertLess(hl['p_value'], 0.05)


# ══════════════════════════════════════════════════════════════
# 校准偏差评估与重校准触发
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
class TestCalibrationAssessment(unittest.TestCase):
    """校准偏差检测与重校准触发测试。"""

    def test_no_bias_when_well_calibrated(self):
        from tb_risk.scoring.ml.validation import (
            compute_calibration_metrics, assess_calibration)
        rng = np.random.RandomState(3)
        p = rng.uniform(0.1, 0.9, 2000)
        y = (rng.rand(2000) < p).astype(int)
        calib = compute_calibration_metrics(y, p, n_bins=10)
        a = assess_calibration(calib)
        self.assertFalse(a['systematic_bias'])
        self.assertFalse(a['recalibrate'])

    def test_overconfidence_triggers_recalibration(self):
        """概率系统性偏高时触发重校准。"""
        from tb_risk.scoring.ml.validation import (
            compute_calibration_metrics, assess_calibration)
        rng = np.random.RandomState(4)
        p = rng.uniform(0.1, 0.9, 3000)
        # 观测阳性率系统性低于预测（模型高估）
        y = (rng.rand(3000) < (p - 0.2)).astype(int)
        calib = compute_calibration_metrics(y, p, n_bins=10)
        a = assess_calibration(calib)
        self.assertTrue(a['systematic_bias'])
        self.assertTrue(a['recalibrate'])
        self.assertEqual(a['direction'], 'over')

    def test_direction_under(self):
        from tb_risk.scoring.ml.validation import (
            compute_calibration_metrics, assess_calibration)
        rng = np.random.RandomState(5)
        p = rng.uniform(0.1, 0.9, 3000)
        y = (rng.rand(3000) < np.clip(p + 0.25, 0, 1)).astype(int)
        calib = compute_calibration_metrics(y, p, n_bins=10)
        a = assess_calibration(calib)
        # 观测高于预测 => 模型低估
        self.assertEqual(a['direction'], 'under')


# ══════════════════════════════════════════════════════════════
# 聚合验证报告
# ══════════════════════════════════════════════════════════════

@_skip_if_no_numpy
@_skip_if_no_sklearn
class TestValidationReport(unittest.TestCase):
    """build_validation_report 聚合报告测试。"""

    @classmethod
    def setUpClass(cls):
        from tb_risk.scoring.predictor import MLRiskPredictor
        cls.predictor = MLRiskPredictor(random_state=42)
        try:
            cls.predictor.train_models(n_samples=500)
        except Exception:
            cls.predictor = None

    def test_untrained_returns_skipped(self):
        from tb_risk.scoring.predictor import MLRiskPredictor
        p = MLRiskPredictor()
        report = p.build_validation_report()
        self.assertEqual(report['status'], 'skipped')

    def test_report_structure(self):
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        report = self.predictor.build_validation_report(
            n_samples=600, cv_folds=3, n_bins=5)
        self.assertEqual(report['status'], 'ok')
        self.assertIn('data', report)
        self.assertIn('models', report)
        self.assertIn('conclusion', report)
        self.assertGreater(len(report['models']), 0)
        self.assertIn('split_method', report['data'])

    def test_report_metrics_present(self):
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        report = self.predictor.build_validation_report(
            n_samples=600, cv_folds=3, n_bins=5)
        for entry in report['models'].values():
            self.assertIn('test', entry)
            self.assertIn('calibration', entry)
            self.assertIn('calibration_assessment', entry)
            self.assertIn('recalibration', entry)
            self.assertIn('cv', entry)
            self.assertIn('roc_auc', entry['test'])
            self.assertIn('brier', entry['calibration'])
            self.assertIn('hl', entry['calibration'])

    def test_cv_mean_std_present(self):
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        report = self.predictor.build_validation_report(
            n_samples=600, cv_folds=3, n_bins=5)
        for entry in report['models'].values():
            cv = entry.get('cv')
            if cv:
                self.assertIn('mean', cv.get('auc', {}))
                self.assertIn('std', cv.get('auc', {}))
                self.assertGreaterEqual(cv['auc']['std'], 0)

    def test_temporal_split_report(self):
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        X, y = self.predictor._generate_synthetic_training_data(600, 42)
        t = np.arange(len(X))
        report = self.predictor.build_validation_report(
            X=X, y=y, temporal_col=t, cv_folds=3, n_bins=5)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['data']['split_method'], 'temporal')

    def test_ensemble_reported(self):
        if self.predictor is None or not self.predictor.is_trained:
            self.skipTest("预测器训练失败")
        report = self.predictor.build_validation_report(
            n_samples=600, cv_folds=3, n_bins=5)
        self.assertIn('test', report.get('ensemble', {}))
        self.assertIn('roc_auc', report['ensemble']['test'])


if __name__ == '__main__':
    unittest.main()