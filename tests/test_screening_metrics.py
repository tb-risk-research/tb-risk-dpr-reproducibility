#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署度量纯函数测试（round-9 P0-2：AUROC → yield@k / NNS / PPV）。

``tb_risk.ml.planner.screening_metrics_at_budgets`` 是 GUI 部署度量
标签页与 SINAN 部署实验脚本共用的评估接口，其数学契约必须锁定：

  yield@k —— top-k 捕获阳性占全部阳性比例（随机基线 = k%）
  nns     —— 每捕获一例阳性需筛查人数 = m / caught
  ppv     —— top-k 内阳性率 = caught / m
  lift    —— yield / (k/100)
  yield@100 = 1.0（满预算必捕获全部）
"""
import unittest

import numpy as np

from tb_risk.ml.planner import (DEFAULT_BUDGETS, screening_metrics_at_budgets,
                                yield_at_k)


class TestYieldAtK(unittest.TestCase):

    def test_perfect_ranking_full_capture(self):
        """完美排序：前 50% 分数恰好覆盖全部阳性 → yield@50 = 1.0。"""
        y = np.array([1] * 5 + [0] * 5)
        scores = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        self.assertEqual(yield_at_k(y, scores, 50), 1.0)

    def test_uninformative_scores_near_random_baseline(self):
        """无信息分数：大样本下 yield@k ≈ k%（随机基线）。"""
        rng = np.random.default_rng(42)
        n = 20000
        y = (rng.random(n) < 0.1).astype(int)
        scores = rng.random(n)
        yk = yield_at_k(y, scores, 10)
        self.assertAlmostEqual(yk, 0.10, delta=0.01)

    def test_empty_or_no_positives(self):
        """边界：空数组 / 无阳性 → 0.0 不抛异常。"""
        self.assertEqual(yield_at_k([], [], 10), 0.0)
        self.assertEqual(yield_at_k([0, 0, 0], [1.0, 2.0, 3.0], 10), 0.0)

    def test_budget_rounding_minimum_one_sample(self):
        """小样本：预算不足 1 人时至少筛 1 人（yield > 0）。"""
        y = [1, 0]
        scores = [5.0, 1.0]
        self.assertEqual(yield_at_k(y, scores, 1), 1.0)


class TestScreeningMetricsAtBudgets(unittest.TestCase):

    def _case(self):
        """确定性样例：10 人、3 例阳性、分数完美分层。"""
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        scores = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
        return y, scores

    def test_structure_and_defaults(self):
        """默认预算档位齐全，每档四指标。"""
        out = screening_metrics_at_budgets(*self._case())
        self.assertEqual(tuple(sorted(out.keys())), tuple(sorted(DEFAULT_BUDGETS)))
        for m in out.values():
            for key in ('yield', 'nns', 'ppv', 'lift'):
                self.assertIn(key, m)

    def test_math_consistency(self):
        """四指标两两一致性：nns=1/ppv，yield=caught/n_pos，lift=yield/(k/100)。"""
        y, scores = self._case()
        n, n_pos = len(y), int(y.sum())
        out = screening_metrics_at_budgets(y, scores, budgets=(10, 30, 50))
        for k, m in out.items():
            m_size = max(1, int(round(n * k / 100.0)))
            caught = round(m['yield'] * n_pos)
            self.assertAlmostEqual(m['ppv'], caught / m_size, places=9)
            self.assertAlmostEqual(m['nns'], 1.0 / m['ppv'], places=9)
            self.assertAlmostEqual(m['lift'], m['yield'] / (k / 100.0),
                                   places=9)

    def test_full_budget_captures_everything(self):
        """满预算：yield=1.0，nns=n/n_pos，ppv=n_pos/n。"""
        y, scores = self._case()
        n, n_pos = len(y), int(y.sum())
        m = screening_metrics_at_budgets(y, scores, budgets=(100,))[100]
        self.assertEqual(m['yield'], 1.0)
        self.assertAlmostEqual(m['nns'], n / n_pos, places=9)
        self.assertAlmostEqual(m['ppv'], n_pos / n, places=9)
        self.assertEqual(m['lift'], 1.0)

    def test_perfect_top_decile(self):
        """完美样例的 top-10%：yield=1/3，ppv=1，nns=1。"""
        y, scores = self._case()
        m = screening_metrics_at_budgets(y, scores, budgets=(10,))[10]
        self.assertAlmostEqual(m['yield'], 1 / 3, places=9)
        self.assertEqual(m['ppv'], 1.0)
        self.assertEqual(m['nns'], 1.0)
        self.assertAlmostEqual(m['lift'], (1 / 3) / 0.10, places=9)

    def test_zero_catch_returns_none_nns(self):
        """top-k 无阳性时 nns=None（不产生 inf），其余指标有值。"""
        y = np.array([0, 0, 0, 0, 1])
        scores = np.array([9.0, 8.0, 7.0, 6.0, 1.0])
        m = screening_metrics_at_budgets(y, scores, budgets=(20,))[20]
        self.assertEqual(m['yield'], 0.0)
        self.assertIsNone(m['nns'])
        self.assertEqual(m['ppv'], 0.0)

    def test_degenerate_inputs(self):
        """空数组 / 无阳性 / 非法预算 → 全 None 不抛异常。"""
        for y, s in (([], []), ([0, 0], [1.0, 2.0])):
            out = screening_metrics_at_budgets(y, s, budgets=(10, 50))
            for m in out.values():
                self.assertIsNone(m['yield'])
                self.assertIsNone(m['nns'])
        out = screening_metrics_at_budgets([1, 0], [1.0, 2.0], budgets=(0,))
        self.assertIsNone(out[0]['yield'])

    def test_budget_over_100_capped(self):
        """k>100 时 top-m 封顶到 n，yield 不超过 1.0。"""
        y, scores = self._case()
        m = screening_metrics_at_budgets(y, scores, budgets=(150,))[150]
        self.assertEqual(m['yield'], 1.0)


if __name__ == '__main__':
    unittest.main()
