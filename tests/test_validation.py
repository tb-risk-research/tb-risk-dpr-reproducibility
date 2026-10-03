#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - validation 模块专项测试 (V4)

覆盖：
- KaramayValidator 外部数据加载
- BacktestEngine 时间分割回测
- AblationAnalyzer 统计检验 + AUROC
- NationalModelComparator BCa + Wilcoxon
"""

import sys
import os
import unittest
import tempfile
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class TestKaramayValidatorExternalData(unittest.TestCase):
    """V1: KaramayValidator 外部数据加载接口测试"""

    @classmethod
    def setUpClass(cls):
        from tb_risk.validation.validator import KaramayValidator
        cls.Validator = KaramayValidator

    def setUp(self):
        self.validator = self.Validator(localizer=None, random_state=42)

    def _make_sample_records(self, n=50):
        """生成模拟外部数据记录"""
        records = []
        for i in range(n):
            records.append({
                'record_id': f'ext_{i:04d}',
                'is_confirmed': 1 if i < 5 else 0,
                'age': 30 + i % 40,
                'cough_freq': i % 5,
                'sputum_smear': 1 + (i % 2),
                'has_cavity': 1 + (i % 3 == 0),
                'treatment': 2 if i < 3 else 1,
                'ftd': 10 + i % 20,
                'symptoms': '咳嗽' if i % 2 == 0 else '发热',
                'exposure_hours': 100 + i * 10,
            })
        return records

    def test_load_json_external_data(self):
        """测试从 JSON 文件加载外部数据"""
        records = self._make_sample_records(30)
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False,
                encoding='utf-8') as f:
            json.dump({'records': records}, f)
            tmp_path = f.name

        try:
            loaded = self.validator.load_external_data(
                tmp_path, record_key='records', verbose=False)
            self.assertEqual(len(loaded), 30)
            self.assertTrue(self.validator._external_data_loaded)
            n_confirmed = sum(1 for r in loaded if r.get('is_confirmed', 0))
            self.assertEqual(n_confirmed, 5)
        finally:
            os.unlink(tmp_path)

    def test_load_csv_external_data(self):
        """测试从 CSV 文件加载外部数据"""
        import csv
        records = self._make_sample_records(20)
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.csv', delete=False,
                encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
            tmp_path = f.name

        try:
            loaded = self.validator.load_external_data(
                tmp_path, verbose=False)
            self.assertEqual(len(loaded), 20)
        finally:
            os.unlink(tmp_path)

    def test_load_jsonl_external_data(self):
        """测试从 JSONL 文件加载外部数据"""
        records = self._make_sample_records(15)
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.jsonl', delete=False,
                encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            tmp_path = f.name

        try:
            loaded = self.validator.load_external_data(
                tmp_path, verbose=False)
            self.assertEqual(len(loaded), 15)
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_file_raises(self):
        """测试加载不存在的文件时抛出异常"""
        with self.assertRaises(FileNotFoundError):
            self.validator.load_external_data(
                'nonexistent_file.json', verbose=False)

    def test_unsupported_format_raises(self):
        """测试不支持的格式时抛出异常"""
        with self.assertRaises(ValueError):
            self.validator.load_external_data(
                'data.parquet', verbose=False)

    def test_normalize_records_missing_label(self):
        """测试标准化缺失标签字段的记录"""
        records = [{'record_id': '001', 'age': 30}]
        normalized = self.validator._normalize_records(
            records, 'is_confirmed', 'record_id')
        self.assertEqual(normalized[0]['is_confirmed'], 0)


class TestBacktestTimeSplit(unittest.TestCase):
    """BacktestEngine 时间分割回测测试"""

    def test_time_split_available(self):
        """测试时间分割方法存在"""
        from tb_risk.validation.backtest import BacktestEngine
        engine = BacktestEngine(localizer=None, use_localization=True)
        self.assertTrue(hasattr(engine, 'run_backtest_with_time_split'))

    def test_time_split_param_validation(self):
        """测试时间分割参数验证"""
        from tb_risk.validation.backtest import BacktestEngine
        engine = BacktestEngine(localizer=None, use_localization=True)
        # 空记录应返回空结果
        result = engine.run_backtest_with_time_split(
            [], time_field='date', train_ratio=0.7)
        self.assertEqual(result.get('dataset_size', 0), 0)


class TestAblationAnalyzerMetrics(unittest.TestCase):
    """AblationAnalyzer 统计检验 + AUROC 测试"""

    def test_ablation_has_auroc(self):
        """测试消融实验报告包含 AUROC"""
        from tb_risk.validation.ablation import AblationAnalyzer
        analyzer = AblationAnalyzer(localizer=None)
        self.assertTrue(hasattr(analyzer, 'run_ablation_with_metrics'))

    def test_evaluate_with_engine_returns_metrics(self):
        """测试 _evaluate_with_engine 返回分类指标"""
        from tb_risk.validation.ablation import AblationAnalyzer
        analyzer = AblationAnalyzer(localizer=None)
        # 空记录测试
        result = analyzer._evaluate_with_engine(None, [])
        self.assertIn('mean_disease_prob', result)
        self.assertIn('auroc', result)


class TestNationalModelComparator(unittest.TestCase):
    """NationalModelComparator BCa + Wilcoxon 测试"""

    def test_bca_params_computation(self):
        """测试 BCa 参数计算"""
        import numpy as np
        from tb_risk.validation.comparator import NationalModelComparator

        # 模拟 bootstrap 统计量
        np.random.seed(42)
        bootstrap_stats = np.random.normal(0.05, 0.02, 1000)
        point_estimate = 0.05

        bias_corr, acceleration = NationalModelComparator._compute_bca_params(
            bootstrap_stats, point_estimate, n=500)

        self.assertIsNotNone(bias_corr)
        self.assertIsNotNone(acceleration)


if __name__ == '__main__':
    unittest.main()