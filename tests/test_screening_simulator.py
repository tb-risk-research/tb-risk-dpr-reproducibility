#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - 筛查数据模拟器单元测试

针对 scoring/simulator.py 的 ScreeningDataSimulator 做行为级测试：
- 合成数据生成（字段完整性 / 样本数 / 确定性与可复现性）
- 传播链标签分配（SEIR 传播链，打破循环论证）
- 真实数据加载 / 合成回退 / 批量导入
- 抽样辅助函数（年龄 / PM10 / 湿度 / 加权选择）
"""
import os
import sys
import unittest

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


REQUIRED_FIELDS = {
    '_id', 'district', 'ethnicity', 'occupation', 'age', 'gender',
    'single_duration', 'freq_density', 'time_span', 'has_symptoms',
    'bcg_vaccine', 'has_tb', 'ventilation', 'contact_distance',
    'exposure_setting', 'past_illness', 'past_illness_type', 'is_high_risk',
    'is_oilfield', 'crew_type', 'idu_status', 'origin_altitude',
    'months_since_migration', 'pm10', 'humidity', 'month',
    'cumulative_exposure',
}


class TestGenerateDataset(unittest.TestCase):
    """合成数据生成测试。"""

    def setUp(self):
        from tb_risk.scoring.simulator import ScreeningDataSimulator
        self.sim = ScreeningDataSimulator(random_state=42)

    def test_sample_count(self):
        records = self.sim.generate_dataset(n_samples=100)
        self.assertEqual(len(records), 100)

    def test_required_fields_present(self):
        records = self.sim.generate_dataset(n_samples=50)
        for rec in records:
            for field in REQUIRED_FIELDS:
                self.assertIn(field, rec, f"记录缺少字段: {field}")

    def test_deterministic_with_seed(self):
        from tb_risk.scoring.simulator import ScreeningDataSimulator
        a = ScreeningDataSimulator(random_state=123).generate_dataset(n_samples=200)
        b = ScreeningDataSimulator(random_state=123).generate_dataset(n_samples=200)
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        from tb_risk.scoring.simulator import ScreeningDataSimulator
        a = ScreeningDataSimulator(random_state=1).generate_dataset(n_samples=200)
        b = ScreeningDataSimulator(random_state=2).generate_dataset(n_samples=200)
        self.assertNotEqual(a, b)

    def test_age_in_valid_range(self):
        records = self.sim.generate_dataset(n_samples=200)
        for rec in records:
            self.assertGreaterEqual(rec['age'], 0)
            self.assertLess(rec['age'], 100)

    def test_cumulative_exposure_nonnegative(self):
        records = self.sim.generate_dataset(n_samples=200)
        for rec in records:
            self.assertGreaterEqual(rec['cumulative_exposure'], 0.0)

    def test_include_labels_false(self):
        records = self.sim.generate_dataset(n_samples=50, include_labels=False)
        for rec in records:
            self.assertNotIn('is_confirmed', rec)

    def test_exposure_setting_valid(self):
        records = self.sim.generate_dataset(n_samples=300)
        valid = set(self.sim.EXPOSURE_SETTINGS)
        for rec in records:
            self.assertIn(rec['exposure_setting'], valid)


class TestTransmissionChainLabels(unittest.TestCase):
    """传播链标签分配测试。"""

    def setUp(self):
        from tb_risk.scoring.simulator import ScreeningDataSimulator
        self.sim = ScreeningDataSimulator(random_state=42)

    def test_labels_are_binary(self):
        records = self.sim.generate_dataset(n_samples=2000)
        for rec in records:
            self.assertIn(rec['is_confirmed'], (0, 1))

    def test_has_confirmed_cases(self):
        records = self.sim.generate_dataset(n_samples=3000)
        confirmed = sum(1 for r in records if r['is_confirmed'] == 1)
        self.assertGreater(confirmed, 0, "传播链应产生至少一个确诊病例")

    def test_rate_is_low_not_degenerate(self):
        # 避免出现"全部确诊"的退化场景
        records = self.sim.generate_dataset(n_samples=3000)
        rate = sum(1 for r in records if r['is_confirmed'] == 1) / len(records)
        self.assertLess(rate, 0.10, f"检出率应远低于10%，实际 {rate:.4f}")

    def test_confirm_count_reasonable_scale(self):
        # 检出率量级应接近 121/10 万（同一数量级内，允许传播链放大）
        records = self.sim.generate_dataset(n_samples=5000)
        confirmed = sum(1 for r in records if r['is_confirmed'] == 1)
        rate_per_100k = confirmed / len(records) * 100000
        self.assertLess(rate_per_100k, 5000, f"检出率量级异常: {rate_per_100k:.0f}/10万")


class TestLoadAndGenerate(unittest.TestCase):
    """真实数据加载 / 合成回退测试。"""

    def setUp(self):
        from tb_risk.scoring.simulator import ScreeningDataSimulator
        self.sim = ScreeningDataSimulator(random_state=42)

    def test_load_or_generate_without_file_returns_synthetic(self):
        records, meta = self.sim.load_or_generate(filepath=None, n_samples=100)
        self.assertEqual(meta['data_source'], 'synthetic')
        self.assertEqual(len(records), 100)
        for rec in records:
            self.assertEqual(rec.get('_data_source'), 'synthetic')

    def test_load_from_file_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.sim.load_from_file('/nonexistent/path/data.csv')

    def test_import_batch_mode_append(self):
        existing = [{'record_id': 'a', 'is_confirmed': 0}]
        # 用合成数据文件回退场景：无文件时应抛 FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.sim.import_batch('/nonexistent/file.csv', mode='append',
                                  existing_records=existing)


class TestAuxiliarySampling(unittest.TestCase):
    """抽样辅助函数测试。"""

    def setUp(self):
        from tb_risk.scoring.simulator import ScreeningDataSimulator
        self.SimClass = ScreeningDataSimulator
        self.sim = ScreeningDataSimulator(random_state=42)

    def test_weighted_choice(self):
        weights = {'a': 1, 'b': 0}
        result = self.SimClass._weighted_choice(weights, self.sim.rng)
        self.assertEqual(result, 'a')

    def test_sample_age_bounds(self):
        for _ in range(500):
            age = self.SimClass._sample_age(self.sim.rng)
            self.assertGreaterEqual(age, 0)
            self.assertLess(age, 100)

    def test_sample_pm10_nonnegative(self):
        for month in range(1, 13):
            val = self.SimClass._sample_pm10(month, self.sim.rng)
            self.assertGreaterEqual(val, 0.0, f"month={month}")

    def test_sample_humidity_clamped(self):
        for month in range(1, 13):
            val = self.SimClass._sample_humidity(month, self.sim.rng)
            self.assertGreaterEqual(val, 5.0)
            self.assertLessEqual(val, 95.0)


class TestDeprecatedAssignLabel(unittest.TestCase):
    """旧版标签分配（已弃用，保留兼容）测试。"""

    def test_assign_label_runs(self):
        from tb_risk.scoring.simulator import ScreeningDataSimulator
        sim = ScreeningDataSimulator(random_state=42)
        record = sim._generate_one_record(0)
        label = sim._assign_label(record, n_confirmed=5, total_n=100)
        self.assertIsInstance(label, bool)


if __name__ == '__main__':
    unittest.main()
