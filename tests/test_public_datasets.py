#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - 公开数据集目录按任务匹配（改进三-第二步）

针对 validation/public_datasets.py：
- 目录完整性（个体/接触者/传播三层）
- 按任务 / 层级过滤与优先级排序（任务 A/B 优先、任务 C 靠后）
- 任务外部验证覆盖（A/B 可验证，C 仅方法学参考）
- 目录查询健壮性（未知 id 抛 KeyError）
"""
import os
import sys
import unittest

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class TestCatalogCompleteness(unittest.TestCase):
    """目录完整性测试。"""

    def test_has_all_known_datasets(self):
        from tb_risk.validation import PUBLIC_DATASETS
        ids = {d['id'] for d in PUBLIC_DATASETS}
        for expected in ['tb_portals', 'smh_tb', 'erase_tb',
                         'china_ccdc_school_2024', 'bc_wgs_clusters']:
            self.assertIn(expected, ids)

    def test_covers_all_three_layers(self):
        from tb_risk.validation import (PUBLIC_DATASETS, LAYER_INDIVIDUAL,
                                         LAYER_CONTACT, LAYER_TRANSMISSION)
        layers = {d['layer'] for d in PUBLIC_DATASETS}
        self.assertIn(LAYER_INDIVIDUAL, layers)
        self.assertIn(LAYER_CONTACT, layers)
        self.assertIn(LAYER_TRANSMISSION, layers)

    def test_priority_labels_cover_all(self):
        from tb_risk.validation import PUBLIC_DATASETS
        for d in PUBLIC_DATASETS:
            self.assertIn('priority', d)
            self.assertGreaterEqual(d['priority'], 1)
            self.assertIn('tasks', d)


class TestFiltering(unittest.TestCase):
    """任务 / 层级过滤测试。"""

    def test_task_a_only_priority1(self):
        from tb_risk.validation import list_public_datasets
        datasets = list_public_datasets(task='A')
        self.assertTrue(datasets)
        for d in datasets:
            self.assertIn('A', d['tasks'])
            self.assertEqual(d['priority'], 1, "任务 A 数据集应为优先接入")

    def test_sorted_by_priority(self):
        from tb_risk.validation import list_public_datasets
        datasets = list_public_datasets()
        prios = [d['priority'] for d in datasets]
        self.assertEqual(prios, sorted(prios))

    def test_filter_by_layer(self):
        from tb_risk.validation import (list_public_datasets, LAYER_CONTACT)
        datasets = list_public_datasets(layer=LAYER_CONTACT)
        self.assertTrue(datasets)
        for d in datasets:
            self.assertEqual(d['layer'], LAYER_CONTACT)


class TestCoverage(unittest.TestCase):
    """任务外部验证覆盖测试。"""

    def test_ab_external_validatable(self):
        from tb_risk.validation import dataset_coverage_by_task
        cov = dataset_coverage_by_task()
        self.assertTrue(cov['A']['external_validatable'])
        self.assertTrue(cov['B']['external_validatable'])

    def test_c_methodology_only(self):
        from tb_risk.validation import dataset_coverage_by_task
        cov = dataset_coverage_by_task()
        self.assertFalse(cov['C']['external_validatable'])
        self.assertTrue(cov['C']['methodology_only'])

    def test_access_order_ab_before_c(self):
        from tb_risk.validation import recommended_access_order
        order = recommended_access_order()
        prios = [o['priority'] for o in order]
        self.assertEqual(prios, [1, 2, 3])


class TestLookup(unittest.TestCase):
    """目录查询健壮性测试。"""

    def test_get_existing(self):
        from tb_risk.validation import get_public_dataset
        ds = get_public_dataset('tb_portals')
        self.assertEqual(ds['id'], 'tb_portals')
        self.assertEqual(ds['access'], 'open')

    def test_get_unknown_raises(self):
        from tb_risk.validation import get_public_dataset
        with self.assertRaises(KeyError):
            get_public_dataset('does_not_exist')


if __name__ == '__main__':
    unittest.main()
