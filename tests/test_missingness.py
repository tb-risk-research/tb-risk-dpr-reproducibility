#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成数据增强 + 缺失值插补 + 缺失敏感性分析 测试套件

覆盖：
1. 合成数据增强（generate_augmented_contact_dataset）
2. 缺失值插补（impute_contact_dataset）
3. 缺失敏感性分析（analyze_missingness / find_boundary_missing_rate）
4. 包导出完整性
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


# ============================================================================
# 1. 合成数据增强
# ============================================================================

class TestGenerateAugmentedDataset(unittest.TestCase):
    """generate_augmented_contact_dataset 行为测试。"""

    def setUp(self):
        from tb_risk.validation.missingness import generate_augmented_contact_dataset
        self.generate = generate_augmented_contact_dataset

    def test_generates_correct_count(self):
        recs = self.generate(n_samples=100, random_state=1)
        self.assertEqual(len(recs), 100)

    def test_contact_type_assigned(self):
        recs = self.generate(n_samples=50, random_state=2)
        for r in recs:
            self.assertIn('contact_type', r)
            self.assertIn(r['contact_type'], ('family', 'social'))

    def test_has_is_confirmed_label(self):
        recs = self.generate(n_samples=50, random_state=3)
        for r in recs:
            self.assertIn('is_confirmed', r)
            self.assertIn(r['is_confirmed'], (0, 1))

    def test_record_id_unique(self):
        recs = self.generate(n_samples=100, random_state=4)
        ids = [r['record_id'] for r in recs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_has_required_fields(self):
        required = {'age', 'bcg_vaccine', 'has_symptoms', 'cumulative_exposure',
                    'single_duration', 'freq_density', 'time_span', 'ventilation',
                    'contact_distance', 'exposure_setting', 'is_confirmed'}
        recs = self.generate(n_samples=20, random_state=5)
        for r in recs:
            for field in required:
                self.assertIn(field, r, f"缺少字段: {field}")

    def test_deterministic(self):
        a = self.generate(n_samples=50, random_state=6)
        b = self.generate(n_samples=50, random_state=6)
        self.assertEqual(a, b)

    def test_contact_type_ratio(self):
        # ratio=0.0 → 全部 social
        recs = self.generate(n_samples=30, contact_type_ratio=0.0, random_state=7)
        self.assertTrue(all(r['contact_type'] == 'social' for r in recs))
        # ratio=1.0 → 全部 family
        recs = self.generate(n_samples=30, contact_type_ratio=1.0, random_state=7)
        self.assertTrue(all(r['contact_type'] == 'family' for r in recs))

    def test_exported(self):
        from tb_risk.validation import generate_augmented_contact_dataset
        self.assertTrue(callable(generate_augmented_contact_dataset))


# ============================================================================
# 2. 缺失值插补
# ============================================================================

class TestImputeContactDataset(unittest.TestCase):
    """impute_contact_dataset 行为测试。"""

    def setUp(self):
        from tb_risk.validation.missingness import (
            generate_augmented_contact_dataset, impute_contact_dataset)
        self.records = generate_augmented_contact_dataset(n_samples=50, random_state=10)
        self.impute = impute_contact_dataset

    def test_no_missing_returns_same(self):
        recs, audit = self.impute([dict(r) for r in self.records])
        self.assertEqual(len(recs), len(self.records))
        for field, info in audit.items():
            self.assertEqual(info['filled'], 0)
            self.assertEqual(info['missing_before'], 0)

    def test_fills_introduced_missing(self):
        import random
        rng = random.Random(1)
        modified = [dict(r) for r in self.records]
        for r in modified:
            if rng.random() < 0.3:
                r['has_symptoms'] = None
            if rng.random() < 0.3:
                r['bcg_vaccine'] = None
        recs, audit = self.impute(modified)
        # 所有记录的非缺失字段应被填充
        for r in recs:
            self.assertIsNotNone(r.get('has_symptoms'))
            self.assertIsNotNone(r.get('bcg_vaccine'))
        # 审计应有记录
        for field in ('has_symptoms', 'bcg_vaccine'):
            self.assertIn(field, audit)
            self.assertGreaterEqual(audit[field]['filled'], 0)
            self.assertGreaterEqual(audit[field]['missing_before'], 0)

    def test_specific_field_set(self):
        import random
        rng = random.Random(2)
        modified = [dict(r) for r in self.records]
        for r in modified:
            if rng.random() < 0.5:
                r['age'] = None
        recs, audit = self.impute(modified, fields=['age'])
        self.assertIn('age', audit)
        self.assertNotIn('has_symptoms', audit)
        for r in recs:
            self.assertIsNotNone(r.get('age'))

    def test_exported(self):
        from tb_risk.validation import impute_contact_dataset
        self.assertTrue(callable(impute_contact_dataset))


# ============================================================================
# 3. 缺失敏感性分析
# ============================================================================

class TestAnalyzeMissingness(unittest.TestCase):
    """analyze_missingness 行为测试。"""

    def setUp(self):
        from tb_risk.validation.missingness import (
            generate_augmented_contact_dataset, analyze_missingness)
        self.records = generate_augmented_contact_dataset(n_samples=200, random_state=20)
        self.analyze = analyze_missingness

    def test_baseline_structure(self):
        res = self.analyze(self.records, n_trials=2, random_state=21)
        self.assertIn('baseline', res)
        self.assertIn('rates', res)
        self.assertIn('fields', res)
        self.assertEqual(res['baseline']['n'], 200)
        self.assertGreater(res['baseline']['mean'], 0.0)

    def test_rate_monotonic(self):
        """缺失率越高，shift 应单调非递减（或至少不下降）。"""
        res = self.analyze(self.records, n_trials=3, random_state=22)
        shifts = [r['mean_abs_shift'] for r in res['rates']]
        for i in range(1, len(shifts)):
            self.assertGreaterEqual(shifts[i], shifts[i - 1] - 0.02)

    def test_correlation_monotonic(self):
        """缺失率越高，相关性应单调非增。"""
        res = self.analyze(self.records, n_trials=3, random_state=23)
        corrs = [r['correlation'] for r in res['rates']]
        for i in range(1, len(corrs)):
            self.assertLessEqual(corrs[i], corrs[i - 1] + 0.02)

    def test_rate_0_has_no_shift(self):
        res = self.analyze(self.records, n_trials=2, random_state=24)
        self.assertEqual(res['rates'][0]['mean_abs_shift'], 0.0)
        self.assertEqual(res['rates'][0]['grade_flip_rate'], 0.0)
        self.assertAlmostEqual(res['rates'][0]['correlation'], 1.0, places=6)

    def test_empty_records(self):
        res = self.analyze([], random_state=25)
        self.assertEqual(res['baseline']['n'], 0)
        self.assertEqual(res['rates'], [])
        self.assertIsNone(res['boundary'])

    def test_custom_predict_fn(self):
        def constant_predictor(record):
            return 50.0
        res = self.analyze(self.records, predict_fn=constant_predictor, n_trials=2, random_state=26)
        self.assertEqual(res['baseline']['mean'], 50.0)
        self.assertEqual(res['predict'], 'constant_predictor')

    def test_boundary_reasonable(self):
        res = self.analyze(self.records, n_trials=3, random_state=27)
        if res['boundary'] is not None:
            self.assertIn('missing_rate', res['boundary'])
            self.assertGreaterEqual(res['boundary']['missing_rate'], 0.0)

    def test_exported(self):
        from tb_risk.validation import analyze_missingness, find_boundary_missing_rate
        self.assertTrue(callable(analyze_missingness))
        self.assertTrue(callable(find_boundary_missing_rate))


# ============================================================================
# 4. 包导出完整性
# ============================================================================

class TestExports(unittest.TestCase):
    """validation 包导出完整性。"""

    def test_default_constants_exported(self):
        from tb_risk.validation import DEFAULT_SENSITIVITY_FIELDS, DEFAULT_MISSING_RATES
        self.assertGreater(len(DEFAULT_SENSITIVITY_FIELDS), 5)
        self.assertGreater(len(DEFAULT_MISSING_RATES), 3)

    def test_all_public_api_in_all(self):
        from tb_risk.validation import (
            generate_augmented_contact_dataset,
            impute_contact_dataset,
            analyze_missingness,
            find_boundary_missing_rate,
            DEFAULT_SENSITIVITY_FIELDS,
            DEFAULT_MISSING_RATES,
        )
        self.assertTrue(callable(generate_augmented_contact_dataset))
        self.assertTrue(callable(impute_contact_dataset))
        self.assertTrue(callable(analyze_missingness))
        self.assertTrue(callable(find_boundary_missing_rate))


if __name__ == '__main__':
    unittest.main()