#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阈值与规范对照 / 约登指数重切 / 临床分级对照 测试套件

覆盖：
- 阈值-规范对照表结构与等级映射
- 风险等级切分函数（极高/高/中/低）
- 约登指数最优切点（含数据不足/无变异边界）
- 真实转归重切阈值
- 临床标准等级推导（涂阳/涂阴驱动）
- 模型等级 vs 临床标准一致性/差异分析（一致率、Kappa、混淆矩阵、高估/低估）
- 端到端 compare_model_vs_clinical
- 包导出

遵循项目 unittest + pytest 约定（无 numpy 强依赖，纯 Python 实现）。
"""
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════
# 导出检查
# ══════════════════════════════════════════════════════════════

class TestThresholdSpecExports(unittest.TestCase):
    def test_package_exports(self):
        from tb_risk.validation import (
            RISK_GRADES, THRESHOLD_SPEC_TABLE, grade_risk,
            build_threshold_spec_report, youden_optimal_threshold,
            refit_risk_thresholds, clinical_grade_from_record,
            ClinicalGradeComparator, compare_model_vs_clinical)
        self.assertEqual(len(RISK_GRADES), 4)
        self.assertTrue(callable(grade_risk))
        self.assertTrue(callable(build_threshold_spec_report))
        self.assertTrue(callable(youden_optimal_threshold))
        self.assertTrue(callable(compare_model_vs_clinical))

    def test_module_exports(self):
        from tb_risk.validation.threshold_spec import (
            THRESHOLD_SPEC_TABLE, DEFAULT_VERY_HIGH, DEFAULT_HIGH,
            DEFAULT_MEDIUM, SMEAR_CLINICAL_GRADE)
        self.assertEqual(DEFAULT_VERY_HIGH, 15.0)
        self.assertEqual(DEFAULT_HIGH, 8.0)
        self.assertEqual(DEFAULT_MEDIUM, 3.0)
        self.assertIn('sputum_smear_positive', SMEAR_CLINICAL_GRADE)


# ══════════════════════════════════════════════════════════════
# 阈值-规范对照表
# ══════════════════════════════════════════════════════════════

class TestThresholdSpecTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tb_risk.validation.threshold_spec import THRESHOLD_SPEC_TABLE
        cls.table = THRESHOLD_SPEC_TABLE

    def test_table_has_four_grades_in_order(self):
        grades = [r['grade'] for r in self.table]
        self.assertEqual(grades, ['低风险', '中风险', '高风险', '极高风险'])

    def test_every_row_has_authoritative_reference(self):
        for row in self.table:
            self.assertIn('clinical_basis', row)
            self.assertIn('who_priority', row)
            self.assertIn('reference', row)
            self.assertTrue(row['reference'])

    def test_references_mention_who_and_smear(self):
        text = ' '.join(r['reference'] for r in self.table)
        self.assertIn('WHO', text)
        self.assertIn('WS 288', text)

    def test_get_threshold_spec(self):
        from tb_risk.validation.threshold_spec import get_threshold_spec
        row = get_threshold_spec('极高风险')
        self.assertIsNotNone(row)
        self.assertEqual(row['code'], 'very_high')
        self.assertIsNone(get_threshold_spec('未知等级'))

    def test_build_report_structure(self):
        from tb_risk.validation.threshold_spec import build_threshold_spec_report
        report = build_threshold_spec_report()
        self.assertEqual(report['n_grades'], 4)
        self.assertEqual(report['thresholds']['very_high'], 15.0)
        self.assertEqual(report['thresholds']['high'], 8.0)
        self.assertEqual(report['thresholds']['medium'], 3.0)
        for r in report['grades']:
            self.assertIn('range', r)
            self.assertIn('min_prob', r)


# ══════════════════════════════════════════════════════════════
# 风险等级切分
# ══════════════════════════════════════════════════════════════

class TestGradeRisk(unittest.TestCase):
    def test_grading_default_thresholds(self):
        from tb_risk.validation.threshold_spec import grade_risk
        self.assertEqual(grade_risk(20.0), '极高风险')
        self.assertEqual(grade_risk(15.0), '极高风险')  # 边界含等号
        self.assertEqual(grade_risk(10.0), '高风险')
        self.assertEqual(grade_risk(8.0), '高风险')
        self.assertEqual(grade_risk(5.0), '中风险')
        self.assertEqual(grade_risk(3.0), '中风险')
        self.assertEqual(grade_risk(1.0), '低风险')
        self.assertEqual(grade_risk(0.0), '低风险')

    def test_grading_custom_thresholds(self):
        from tb_risk.validation.threshold_spec import grade_risk
        self.assertEqual(grade_risk(90, 80, 50, 20), '极高风险')
        self.assertEqual(grade_risk(60, 80, 50, 20), '高风险')
        self.assertEqual(grade_risk(30, 80, 50, 20), '中风险')
        self.assertEqual(grade_risk(5, 80, 50, 20), '低风险')


# ══════════════════════════════════════════════════════════════
# 约登指数
# ══════════════════════════════════════════════════════════════

class TestYoudenThreshold(unittest.TestCase):
    def test_separated_distributions(self):
        """病例概率高、非病例概率低，最优切点应落在两者之间。"""
        from tb_risk.validation.threshold_spec import youden_optimal_threshold
        probs = [2.0, 3.0, 4.0, 5.0, 30.0, 35.0, 40.0, 45.0]
        outcomes = [0, 0, 0, 0, 1, 1, 1, 1]
        opt = youden_optimal_threshold(probs, outcomes)
        self.assertIsNotNone(opt)
        self.assertTrue(5.0 <= opt['threshold'] <= 30.0)
        self.assertAlmostEqual(opt['sensitivity'], 1.0, places=6)
        self.assertAlmostEqual(opt['specificity'], 1.0, places=6)
        self.assertAlmostEqual(opt['youden_index'], 1.0, places=6)

    def test_imperfect_separation(self):
        """存在交叉时，约登指数应 < 1 且为正。"""
        from tb_risk.validation.threshold_spec import youden_optimal_threshold
        probs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        outcomes = [0, 0, 1, 0, 1, 1, 0, 1]
        opt = youden_optimal_threshold(probs, outcomes)
        self.assertIsNotNone(opt)
        self.assertGreaterEqual(opt['youden_index'], 0.0)
        self.assertLessEqual(opt['youden_index'], 1.0)

    def test_insufficient_data(self):
        from tb_risk.validation.threshold_spec import youden_optimal_threshold
        self.assertIsNone(youden_optimal_threshold([], []))
        self.assertIsNone(youden_optimal_threshold([1.0], [0]))

    def test_no_label_variation(self):
        """全正或全负样本无法计算，应返回 None。"""
        from tb_risk.validation.threshold_spec import youden_optimal_threshold
        self.assertIsNone(youden_optimal_threshold([1, 2, 3], [0, 0, 0]))
        self.assertIsNone(youden_optimal_threshold([1, 2, 3], [1, 1, 1]))

    def test_length_mismatch(self):
        from tb_risk.validation.threshold_spec import youden_optimal_threshold
        self.assertIsNone(youden_optimal_threshold([1, 2, 3], [0, 1]))


class TestRefitThresholds(unittest.TestCase):
    def test_refit_returns_recommendations(self):
        from tb_risk.validation.threshold_spec import refit_risk_thresholds
        probs = [2.0, 3.0, 4.0, 5.0, 30.0, 35.0, 40.0, 45.0]
        outcomes = [0, 0, 0, 0, 1, 1, 1, 1]
        report = refit_risk_thresholds(probs, outcomes)
        self.assertEqual(report['original_thresholds']['very_high'], 15.0)
        self.assertIsNotNone(report['recommended_thresholds'])
        self.assertIn('medium', report['recommended_thresholds'])
        self.assertIn('youden_optimal', report)
        self.assertIn('method', report)

    def test_refit_insufficient(self):
        from tb_risk.validation.threshold_spec import refit_risk_thresholds
        report = refit_risk_thresholds([], [])
        self.assertIsNone(report['recommended_thresholds'])
        self.assertIsNone(report['youden_optimal'])


# ══════════════════════════════════════════════════════════════
# 临床标准等级推导
# ══════════════════════════════════════════════════════════════

class TestClinicalGradeFromRecord(unittest.TestCase):
    def test_smear_positive_very_high(self):
        from tb_risk.validation.threshold_spec import clinical_grade_from_record
        self.assertEqual(clinical_grade_from_record(
            {'sputum_smear': 2}), '极高风险')
        self.assertEqual(clinical_grade_from_record(
            {'sputum_smear': '涂阳'}), '极高风险')
        self.assertEqual(clinical_grade_from_record(
            {'sputum_smear': '阳性'}), '极高风险')

    def test_cavity_high(self):
        from tb_risk.validation.threshold_spec import clinical_grade_from_record
        self.assertEqual(clinical_grade_from_record(
            {'sputum_smear': 1, 'has_cavity': 2}), '高风险')

    def test_hiv_high(self):
        from tb_risk.validation.threshold_spec import clinical_grade_from_record
        self.assertEqual(clinical_grade_from_record(
            {'illness_type': 'HIV'}), '高风险')

    def test_child_under5_high(self):
        from tb_risk.validation.threshold_spec import clinical_grade_from_record
        self.assertEqual(clinical_grade_from_record({'age': 3}), '高风险')

    def test_smear_negative_medium(self):
        from tb_risk.validation.threshold_spec import clinical_grade_from_record
        self.assertEqual(clinical_grade_from_record(
            {'sputum_smear': 1}), '中风险')

    def test_missing_low(self):
        from tb_risk.validation.threshold_spec import clinical_grade_from_record
        self.assertEqual(clinical_grade_from_record({'age': 40}), '低风险')


# ══════════════════════════════════════════════════════════════
# 临床分级对照
# ══════════════════════════════════════════════════════════════

class TestClinicalGradeComparator(unittest.TestCase):
    def _make_records(self):
        return [
            {'sputum_smear': 2, 'age': 40},   # 临床极高
            {'sputum_smear': 1, 'age': 40},   # 临床中
            {'sputum_smear': 1, 'age': 40},   # 临床中
            {'sputum_smear': 1, 'age': 3},    # 临床高(<5)
            {'age': 40},                       # 临床低
        ]

    def test_compare_structure(self):
        from tb_risk.validation.threshold_spec import (
            ClinicalGradeComparator, grade_risk, clinical_grade_from_record)
        records = self._make_records()
        probs = [20.0, 5.0, 5.0, 10.0, 1.0]
        model_grades = [grade_risk(p) for p in probs]
        clinical_grades = [clinical_grade_from_record(r) for r in records]
        comp = ClinicalGradeComparator().compare(model_grades, clinical_grades)
        self.assertEqual(comp['n'], 5)
        self.assertIn('accuracy', comp)
        self.assertIn('cohens_kappa', comp)
        self.assertIn('confusion_matrix', comp)
        self.assertIn('overestimation_rate', comp)
        self.assertIn('underestimation_rate', comp)
        self.assertIn('conclusion', comp)
        self.assertEqual(comp['overestimation_count'] + comp['underestimation_count']
                         + comp['agreement_count'], 5)

    def test_perfect_agreement(self):
        from tb_risk.validation.threshold_spec import ClinicalGradeComparator
        grades = ['低风险', '中风险', '高风险', '极高风险']
        comp = ClinicalGradeComparator().compare(grades, grades)
        self.assertAlmostEqual(comp['accuracy'], 1.0, places=6)
        self.assertAlmostEqual(comp['cohens_kappa'], 1.0, places=6)
        self.assertEqual(comp['overestimation_count'], 0)
        self.assertEqual(comp['underestimation_count'], 0)

    def test_overestimation_detected(self):
        from tb_risk.validation.threshold_spec import ClinicalGradeComparator
        model = ['极高风险', '极高风险', '极高风险']
        clin = ['高风险', '中风险', '低风险']
        comp = ClinicalGradeComparator().compare(model, clin)
        self.assertEqual(comp['overestimation_count'], 3)
        self.assertEqual(comp['underestimation_count'], 0)
        self.assertGreater(comp['overestimation_rate'], 0.9)

    def test_underestimation_detected(self):
        from tb_risk.validation.threshold_spec import ClinicalGradeComparator
        model = ['低风险', '低风险', '低风险']
        clin = ['中风险', '高风险', '极高风险']
        comp = ClinicalGradeComparator().compare(model, clin)
        self.assertEqual(comp['underestimation_count'], 3)
        self.assertEqual(comp['overestimation_count'], 0)

    def test_length_mismatch_raises(self):
        from tb_risk.validation.threshold_spec import ClinicalGradeComparator
        with self.assertRaises(ValueError):
            ClinicalGradeComparator().compare(['低风险'], ['低风险', '中风险'])


# ══════════════════════════════════════════════════════════════
# 端到端
# ══════════════════════════════════════════════════════════════

class TestCompareEndToEnd(unittest.TestCase):
    def test_compare_model_vs_clinical(self):
        from tb_risk.validation.threshold_spec import compare_model_vs_clinical
        records = [
            {'sputum_smear': 2, 'age': 40, 'is_confirmed': 1},
            {'sputum_smear': 1, 'age': 40, 'is_confirmed': 0},
            {'sputum_smear': 1, 'age': 40, 'is_confirmed': 0},
            {'sputum_smear': 1, 'age': 3, 'is_confirmed': 1},
            {'age': 40, 'is_confirmed': 0},
        ]
        probs = [20.0, 5.0, 5.0, 10.0, 1.0]
        report = compare_model_vs_clinical(records, probs)
        self.assertEqual(report['n_records'], 5)
        self.assertIn('model_vs_clinical', report)
        self.assertIn('threshold_refit', report)
        self.assertIsNotNone(report['threshold_refit']['youden_optimal'])

    def test_custom_grade_fn(self):
        from tb_risk.validation.threshold_spec import compare_model_vs_clinical
        records = [
            {'sputum_smear': 2, 'is_confirmed': 1},
            {'sputum_smear': 1, 'is_confirmed': 0},
            {'sputum_smear': 1, 'is_confirmed': 0},
            {'age': 40, 'is_confirmed': 1},
        ]
        probs = [25.0, 5.0, 5.0, 12.0]
        report = compare_model_vs_clinical(records, probs,
                                           grade_fn=lambda p: '高风险')
        self.assertEqual(report['model_vs_clinical']['n'], 4)
        self.assertEqual(
            report['model_vs_clinical']['model_distribution']['高风险'], 4)


# ══════════════════════════════════════════════════════════════
# 期望校准误差（ECE）对比
# ══════════════════════════════════════════════════════════════

class TestExpectedCalibrationError(unittest.TestCase):
    def test_perfect_calibration(self):
        """预测与观测完全一致时 ECE = 0。"""
        from tb_risk.validation.threshold_spec import expected_calibration_error
        # 箱内预测均值等于观测频率 → 逐箱绝对误差为 0
        res = expected_calibration_error([0.0, 0.0, 1.0, 1.0],
                                         [0, 0, 1, 1])
        self.assertAlmostEqual(res['ece'], 0.0, places=4)
        self.assertEqual(res['n'], 4)

    def test_percent_auto_normalized(self):
        """百分比输入自动归一到 [0,1]。"""
        from tb_risk.validation.threshold_spec import expected_calibration_error
        # [10%, 90%] → [0.1, 0.9]，与小数输入等价的 ECE
        res = expected_calibration_error([10.0, 90.0], [0, 1])
        ref = expected_calibration_error([0.1, 0.9], [0, 1])
        self.assertAlmostEqual(res['ece'], ref['ece'], places=4)
        self.assertEqual(res['n'], 2)

    def test_insufficient_data(self):
        from tb_risk.validation.threshold_spec import expected_calibration_error
        self.assertIn('error', expected_calibration_error([], []))
        self.assertIn('error', expected_calibration_error([0.5], [0]))

    def test_ece_structure(self):
        from tb_risk.validation.threshold_spec import expected_calibration_error
        res = expected_calibration_error(
            [0.2, 0.8, 0.3, 0.9, 0.6, 0.4],
            [0, 1, 0, 1, 1, 0])
        self.assertIn('ece', res)
        self.assertIn('bin_details', res)
        self.assertTrue(0.0 <= res['ece'] <= 1.0)
        self.assertEqual(res['n'], 6)


class TestCompareCalibration(unittest.TestCase):
    def test_integrated_better_than_pure_symptom(self):
        """实验室整合版概率更接近真实转归 → ECE 更低。"""
        from tb_risk.validation.threshold_spec import compare_calibration
        pure = [0.10, 0.60, 0.40, 0.95, 0.30, 0.70]   # 症状版：偏差大
        lab = [0.15, 0.75, 0.35, 0.90, 0.25, 0.80]    # 实验室版：更准
        outcomes = [0, 1, 0, 1, 0, 1]
        report = compare_calibration(pure, lab, outcomes)
        self.assertTrue(report['improved'])
        self.assertGreater(report['ece_reduction'], 0.0)
        self.assertIn('pure_symptom', report)
        self.assertIn('lab_integrated', report)
        self.assertIn('conclusion', report)
        self.assertLess(report['lab_integrated']['ece'],
                        report['pure_symptom']['ece'])

    def test_insufficient_data(self):
        from tb_risk.validation.threshold_spec import compare_calibration
        report = compare_calibration([], [], [])
        self.assertEqual(report['improved'], None)
        self.assertIn('数据不足', report['conclusion'])


# ══════════════════════════════════════════════════════════════
# 近期进展分层：IGRA 状态分层 AUC 验证（双输出 vs 固定基线）
# ══════════════════════════════════════════════════════════════

class TestIgraStratifiedDiscrimination(unittest.TestCase):
    """测试按 IGRA 状态分层的 AUC 区分度验证。"""

    def test_compute_auc_perfect_discrimination(self):
        """完全区分时 AUC=1.0。"""
        from tb_risk.validation.threshold_spec import compute_auc
        probs = [10, 20, 30, 40, 50]
        outs = [0, 0, 0, 1, 1]
        auc = compute_auc(probs, outs)
        self.assertAlmostEqual(auc, 1.0, places=4)

    def test_compute_auc_random_discrimination(self):
        """随机猜测时 AUC≈0.5。"""
        from tb_risk.validation.threshold_spec import compute_auc
        probs = [50, 50, 50, 50]
        outs = [0, 0, 1, 1]
        auc = compute_auc(probs, outs)
        self.assertAlmostEqual(auc, 0.5, places=4)

    def test_compute_auc_insufficient_data(self):
        """样本不足或无标签变异 → None。"""
        from tb_risk.validation.threshold_spec import compute_auc
        self.assertIsNone(compute_auc([], []))
        self.assertIsNone(compute_auc([10, 20], [0, 0]))  # 全 0
        self.assertIsNone(compute_auc([10, 20], [1, 1]))  # 全 1

    def test_auc_standard_error_calculation(self):
        """AUC 标准误计算正确（Hanley & McNeil 1982 公式）。"""
        from tb_risk.validation.threshold_spec import auc_standard_error
        se = auc_standard_error(0.8, 20, 80)
        self.assertGreater(se, 0.0)
        self.assertLess(se, 0.1)

    def test_igra_stratified_comparison_end_to_end(self):
        """端到端：对比双输出 vs 固定基线在已知分层下的区分度。"""
        from tb_risk.validation.threshold_spec import compare_individualized_vs_fixed_baseline
        records = [
            {'igra_result': 'positive', 'age': 30, 'past_illness_type': 'none', 'is_confirmed': 1},
            {'igra_result': 'positive', 'age': 30, 'past_illness_type': 'none', 'is_confirmed': 1},
            {'igra_result': 'positive', 'age': 30, 'past_illness_type': 'none', 'is_confirmed': 0},
            {'igra_result': 'negative', 'age': 30, 'past_illness_type': 'none', 'is_confirmed': 0},
            {'igra_result': 'negative', 'age': 30, 'past_illness_type': 'none', 'is_confirmed': 0},
            {'igra_result': 'negative', 'age': 30, 'past_illness_type': 'none', 'is_confirmed': 1},
            {'igra_result': 'not_done', 'age': 30, 'past_illness_type': 'none', 'is_confirmed': 0},
            {'igra_result': 'not_done', 'age': 30, 'past_illness_type': 'none', 'is_confirmed': 1},
        ]
        report = compare_individualized_vs_fixed_baseline(records)
        self.assertEqual(report['n_records'], 8)
        self.assertEqual(report['n_pos'], 4)
        self.assertEqual(report['n_neg'], 4)
        self.assertIn('overall_auc_individualized', report)
        self.assertIsNotNone(report['overall_auc_fixed_baseline'])
        self.assertIn('strata', report)
        self.assertIn('positive', report['strata'])
        self.assertIn('negative', report['strata'])
        self.assertIn('conclusion', report)


if __name__ == '__main__':
    unittest.main()
