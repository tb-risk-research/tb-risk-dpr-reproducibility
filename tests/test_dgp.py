#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - 可复现 DGP 文档与分布对照校验（改进三-第一步）

针对 validation/dgp.py：
- 字段分布规范（FIELD_SPECS 覆盖 simulator 全部字段）
- 关键变量分布对照校验（观测 vs 规范权重）
- 患病率/检出率对照（vs 官方 121/10 万）
- 可复现性校验（同种子逐字段一致）
- DGP Markdown 文档渲染
"""
import os
import sys
import unittest

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class TestFieldSpecs(unittest.TestCase):
    """字段分布规范覆盖性测试。"""

    def test_covers_all_simulator_fields(self):
        from tb_risk.validation import field_specs_by_name
        spec_map = field_specs_by_name()
        required = [
            'district', 'ethnicity', 'occupation', 'age', 'gender',
            'exposure_setting', 'contact_distance', 'single_duration',
            'freq_density', 'time_span', 'cumulative_exposure',
            'has_symptoms', 'bcg_vaccine', 'has_tb', 'past_illness',
            'idu_status', 'ventilation', 'origin_altitude',
            'months_since_migration', 'pm10', 'humidity',
        ]
        for f in required:
            self.assertIn(f, spec_map, f"DGP 规范缺少 simulator 字段: {f}")

    def test_every_spec_has_distribution_and_source(self):
        from tb_risk.validation import FIELD_SPECS
        for s in FIELD_SPECS:
            self.assertIn('field', s)
            self.assertIn('distribution', s)
            self.assertIn('source', s)
            self.assertIn('kind', s)
            if s['kind'] == 'categorical':
                self.assertTrue(s.get('parameters'), f"分类字段 {s['field']} 缺参数")

    def test_transmission_spec_has_rerf_and_generations(self):
        from tb_risk.validation import TRANSMISSION_SPEC
        self.assertIn('reproduction_number', TRANSMISSION_SPEC)
        self.assertIn('generations', TRANSMISSION_SPEC)
        self.assertIn('seed_selection', TRANSMISSION_SPEC)


class TestDistributionValidation(unittest.TestCase):
    """关键变量分布 / 患病率对照校验测试。"""

    def test_distributions_pass_at_scale(self):
        from tb_risk.validation import build_dgp_report
        rep = build_dgp_report(n_samples=12000, random_state=42)
        dists = rep['validation']['distributions']
        self.assertEqual(len(dists), len(rep['field_specs']))
        fails = [v['field'] for v in dists if not v['pass']]
        self.assertEqual(fails, [], f"分布校验失败字段: {fails}")

    def test_prevalence_within_tolerance(self):
        from tb_risk.validation import validate_prevalence
        from tb_risk.scoring.simulator import ScreeningDataSimulator
        records = ScreeningDataSimulator(random_state=42).generate_dataset(
            n_samples=12000)
        pv = validate_prevalence(records)
        self.assertTrue(pv['pass'])
        self.assertGreater(pv['n_confirmed'], 0)
        self.assertAlmostEqual(pv['target_per_100k'], 121.0, places=1)

    def test_validate_empty_records_raises(self):
        from tb_risk.validation import validate_prevalence
        with self.assertRaises(ValueError):
            validate_prevalence([])

    def test_literature_check_marks_not_applicable(self):
        from tb_risk.validation import validate_against_literature
        from tb_risk.scoring.simulator import ScreeningDataSimulator
        records = ScreeningDataSimulator(random_state=42).generate_dataset(
            n_samples=3000)
        res = validate_against_literature(records)
        entries = res['entries']
        self.assertTrue(any(e['status'] == 'pass' for e in entries))
        self.assertTrue(any(e['status'] == 'not_applicable' for e in entries))


class TestReproducibility(unittest.TestCase):
    """可复现性校验测试。"""

    def test_same_seed_identical(self):
        from tb_risk.validation import reproducibility_check
        res = reproducibility_check(n_samples=500, random_state=7)
        self.assertTrue(res['identical'])
        self.assertEqual(res['random_state'], 7)


class TestRenderDgpDocument(unittest.TestCase):
    """DGP Markdown 文档渲染测试。"""

    def test_document_contains_required_sections(self):
        from tb_risk.validation import render_dgp_document
        doc = render_dgp_document(n_samples=5000, random_state=42)
        for section in [
            '## 1. 全局参数',
            '## 2. 字段分布规范',
            '## 3. 标签生成机制',
            '## 4. 随机种子',
            '## 5. 分布对照校验',
            '## 6. 任务标签口径',
            '## 7. 文献参考表',
            '## 8. 文献对照结果',
        ]:
            self.assertIn(section, doc, f"文档缺少章节: {section}")

    def test_document_records_seed_and_rate(self):
        from tb_risk.validation import render_dgp_document
        doc = render_dgp_document(n_samples=5000, random_state=42)
        self.assertIn('随机种子', doc)
        self.assertIn('121', doc)

    def test_document_annotates_prevalence_calibers(self):
        """问题4：DGP 文档统一标注密接/全人群双口径，杜绝混用。"""
        from tb_risk.validation import render_dgp_document
        doc = render_dgp_document(n_samples=5000, random_state=42)
        # 双口径并列（来自 constants.PREVALENCE_CALIBERS 单一真值源）
        self.assertIn('2.85%', doc)
        self.assertIn('100/10万', doc)
        self.assertIn('密接', doc)
        self.assertIn('全人群', doc)
        # 不可混用警示
        self.assertIn('不可混用', doc)


if __name__ == '__main__':
    unittest.main()
