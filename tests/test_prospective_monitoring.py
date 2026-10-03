#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""前瞻监控纯函数测试（round-12：walk-forward 批次监控）。

``tb_risk.scoring.ml.deployment_metrics`` 的 PSI / ECE 是前瞻部署
证据链（影子运行 → 批次对账 → 告警）的数学契约，必须锁定：

  PSI —— 同分布 ≈ 0；分布平移 > 0.25（风控惯例漂移线）；
         空输入 / n_bins<2 报 ValueError；常量分数不崩溃；
  ECE —— 完美校准（score=y 退化）= 0；系统性高估 = 偏移量；
         与 Brier 同源（同一 (s, y) 输入）。
"""
import json
import os
import unittest

import numpy as np

from tb_risk.scoring.ml.deployment_metrics import (
    calibration_error,
    population_stability_index,
)


class TestPopulationStabilityIndex(unittest.TestCase):

    def test_identical_samples_zero(self):
        """完全相同的两组分数 → PSI 精确为 0。"""
        s = np.linspace(0.0, 1.0, 500)
        self.assertLess(population_stability_index(s, s), 1e-12)

    def test_same_distribution_near_zero(self):
        """同分布独立采样（大样本）→ PSI 仅剩统计噪声（<0.02）。"""
        rng = np.random.default_rng(42)
        ref = rng.random(20000)
        new = rng.random(20000)
        self.assertLess(population_stability_index(ref, new), 0.02)

    def test_shifted_distribution_trips_drift_line(self):
        """均值平移 1 个标准差 → PSI > 0.25（漂移告警线）。"""
        rng = np.random.default_rng(42)
        ref = rng.normal(0.0, 1.0, 20000)
        shifted = rng.normal(1.0, 1.0, 20000)
        self.assertGreater(population_stability_index(ref, shifted), 0.25)

    def test_constant_scores_do_not_crash(self):
        """常量分数（分位数全部并列）→ 退化为单箱，PSI = 0。"""
        c = np.full(100, 0.37)
        self.assertEqual(population_stability_index(c, c), 0.0)

    def test_invalid_inputs_raise(self):
        """空输入 / n_bins<2 → ValueError。"""
        with self.assertRaises(ValueError):
            population_stability_index([], [0.1, 0.2])
        with self.assertRaises(ValueError):
            population_stability_index([0.1], [])
        with self.assertRaises(ValueError):
            population_stability_index([0.1, 0.2], [0.3, 0.4], n_bins=1)


class TestCalibrationError(unittest.TestCase):

    def test_perfect_degenerate_calibration(self):
        """score = y（0/1 退化）→ ECE = 0、Brier = 0。"""
        y = np.array([1, 1, 0, 0, 0])
        s = y.astype(float)
        res = calibration_error(y, s)
        self.assertEqual(res['ece'], 0.0)
        self.assertEqual(res['brier'], 0.0)

    def test_systematic_overestimate(self):
        """全阴样本 + 恒定 0.2 分数 → ECE = 0.2（= 偏移量）。"""
        res = calibration_error([0, 0, 0, 0], [0.2, 0.2, 0.2, 0.2])
        self.assertAlmostEqual(res['ece'], 0.2)
        self.assertAlmostEqual(res['brier'], 0.04)

    def test_known_small_case(self):
        """4 样本 25% 阳性率、全 0.5 分数 → ECE = Brier = 0.25。"""
        res = calibration_error([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(res['ece'], 0.25)
        self.assertAlmostEqual(res['brier'], 0.25)

    def test_structure_and_edge_bin(self):
        """返回结构齐全；score=1.0 落入最后一箱不越界。"""
        res = calibration_error([1, 0], [1.0, 0.05])
        for key in ('ece', 'brier', 'n', 'n_pos', 'bin_centers',
                    'bin_counts', 'bin_pos_rates'):
            self.assertIn(key, res)
        self.assertEqual(res['n'], 2)
        self.assertEqual(res['n_pos'], 1)
        self.assertEqual(len(res['bin_counts']), 10)

    def test_brier_consistency_with_direct_formula(self):
        """Brier 与直接公式 np.mean((s-y)**2) 一致。"""
        rng = np.random.default_rng(7)
        s = rng.random(1000)
        y = (rng.random(1000) < s).astype(int)
        res = calibration_error(y, s)
        self.assertAlmostEqual(res['brier'],
                               float(np.mean((s - y) ** 2)), places=12)

    def test_invalid_inputs_raise(self):
        """空输入 / 长度不一致 → ValueError。"""
        with self.assertRaises(ValueError):
            calibration_error([], [])
        with self.assertRaises(ValueError):
            calibration_error([0, 1], [0.5])


ARCHIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), 'data', 'processed', 'sinan_walkforward_20260904.json')


@unittest.skipUnless(os.path.exists(ARCHIVE),
                     'walk-forward 归档不存在（先跑 run_sinan_walkforward.py）')
class TestWalkforwardGuardrailCalibration(unittest.TestCase):
    """§11.5 护栏分级与通道敏度校准契约（2026-09-05 修订）。

    锁定 walk-forward 84 月回放中 PSI/ECE 告警数的通道分解——文档
    声明的任何数字变动（重跑归档 / 改阈值 / 改特征臂定义）都必须
    显式更新 docs/deployment_ops.md §11.5，不允许静默漂移：
      - PSI 运行参照 = ind 臂：frozen 反事实 7 警告 / 0 漂移级；
      - ind_all PSI 为诊断通道（先证块放大，44 警告但同源，
        逐月相关 >=0.85——放大的不是独立信号）；
      - ECE 护栏盯部署工件 ind_all（21 警告，全部在
        2015-04..2017-12）；ind 参照臂 ECE 零越线；
      - 年度刷新政体 null 带：refit 两通道全部零告警，
        ind 月度 PSI max <=0.05、ind_all <=0.07，ECE max <=0.03。
    """

    @classmethod
    def setUpClass(cls):
        with open(ARCHIVE, encoding='utf-8') as f:
            cls.data = json.load(f)
        cls.alerts = cls.data['alerts']
        cls.months = sorted(cls.data['batches'].keys())

    def _monthly(self, arm, feat, metric):
        return np.array([self.data['batches'][m][arm][feat][metric]
                         for m in self.months])

    def test_ind_reference_psi_warning_count(self):
        """PSI 运行参照（ind）：frozen 反事实恰好 7 月警告、0 月漂移级。"""
        self.assertEqual(
            len(self.alerts['frozen/ind']['psi_warn_gt_0p10']), 7)
        self.assertEqual(
            len(self.alerts['frozen/ind']['psi_drift_gt_0p25']), 0)

    def test_ind_all_psi_is_diagnostic_amplified_same_source(self):
        """ind_all PSI 44 月常驻告警但与 ind 同源（相关 >=0.85）。"""
        self.assertEqual(
            len(self.alerts['frozen/ind_all']['psi_warn_gt_0p10']), 44)
        psi_ind = self._monthly('frozen', 'ind', 'psi')
        psi_iall = self._monthly('frozen', 'ind_all', 'psi')
        corr = float(np.corrcoef(psi_ind, psi_iall)[0, 1])
        self.assertGreaterEqual(corr, 0.85)
        # 放大量级：frozen 期 ind_all 月均约为 ind 的 1.5-2.5 倍
        ratio = float(psi_iall.mean() / psi_ind.mean())
        self.assertGreaterEqual(ratio, 1.5)
        self.assertLessEqual(ratio, 2.5)

    def test_ece_guardrail_on_deployed_artifact_only(self):
        """ECE 告警集中在部署工件 ind_all（21 月，2015-04..2017-12），
        ind 参照臂零越线——ECE 盯参照臂会漏报。"""
        self.assertEqual(
            len(self.alerts['frozen/ind_all']['ece_alert_gt_0p05']), 21)
        self.assertEqual(
            len(self.alerts['frozen/ind']['ece_alert_gt_0p05']), 0)
        em = [m for m in self.months
              if self.data['batches'][m]['frozen']['ind_all']['ece'] > 0.05]
        self.assertGreaterEqual(min(em), '2015-04')
        self.assertLessEqual(max(em), '2017-12')

    def test_refit_regime_null_band(self):
        """年度刷新政体（运行常态）：两通道零告警，null 带远低于警告线。"""
        for key in ('refit/ind', 'refit/ind_all'):
            self.assertEqual(len(self.alerts[key]['psi_warn_gt_0p10']), 0)
            self.assertEqual(len(self.alerts[key]['psi_drift_gt_0p25']), 0)
            self.assertEqual(len(self.alerts[key]['ece_alert_gt_0p05']), 0)
        self.assertLessEqual(self._monthly('refit', 'ind', 'psi').max(),
                             0.05)
        self.assertLessEqual(self._monthly('refit', 'ind_all', 'psi').max(),
                             0.07)
        for feat in ('ind', 'ind_all'):
            self.assertLessEqual(
                self._monthly('refit', feat, 'ece').max(), 0.03)

    def test_threshold_constants_match_protocol(self):
        """代码常量与 §11.5 协议线一致（单一来源）。"""
        from tb_risk.scoring.ml.deployment_metrics import (
            ECE_WARN, PSI_DRIFT, PSI_WARN,
        )
        self.assertEqual(PSI_WARN, 0.10)
        self.assertEqual(PSI_DRIFT, 0.25)
        self.assertEqual(ECE_WARN, 0.05)


if __name__ == '__main__':
    unittest.main()
