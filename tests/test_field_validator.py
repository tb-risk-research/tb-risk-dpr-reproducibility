#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - Section V FieldValidator 专项测试

覆盖三层验证体系：
  - 第一层：纯验证函数（validate_int / validate_range / validate_non_empty 等）
  - 第二层：交叉验证（cross_validate_smear_symptoms）
  - 第三层：表单级预检（validate_form / validate_contact_counts）
  - 错误状态管理（get_errors / has_errors / clear_all_errors / get_first_error）
  - FIELD_RULES / COUNT_RULES 常量一致性
"""

import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class TestFieldValidatorInt(unittest.TestCase):
    """validate_int 整数验证"""

    def setUp(self):
        from tb_risk.gui.validators import FieldValidator
        self.validator = FieldValidator()

    def test_valid_int_in_range(self):
        ok, msg = self.validator.validate_int(50, '年龄', 0, 120)
        self.assertTrue(ok)
        self.assertEqual(msg, '')

    def test_valid_int_no_range(self):
        ok, msg = self.validator.validate_int(42, '计数')
        self.assertTrue(ok)

    def test_int_below_min(self):
        ok, msg = self.validator.validate_int(-5, '年龄', 0, 120)
        self.assertFalse(ok)
        self.assertIn('不能小于', msg)
        self.assertIn('0', msg)

    def test_int_above_max(self):
        ok, msg = self.validator.validate_int(150, '年龄', 0, 120)
        self.assertFalse(ok)
        self.assertIn('不能大于', msg)
        self.assertIn('120', msg)

    def test_string_int_valid(self):
        ok, msg = self.validator.validate_int('30', '年龄', 0, 120)
        self.assertTrue(ok)

    def test_string_float_integer_valid(self):
        """浮点字符串 '30.0' 应被接受为整数 30"""
        ok, msg = self.validator.validate_int('30.0', '年龄', 0, 120)
        self.assertTrue(ok)

    def test_string_float_non_integer_invalid(self):
        ok, msg = self.validator.validate_int('30.5', '年龄', 0, 120)
        self.assertFalse(ok)
        self.assertIn('必须是整数', msg)

    def test_non_numeric_string_invalid(self):
        ok, msg = self.validator.validate_int('abc', '年龄')
        self.assertFalse(ok)
        self.assertIn('必须是整数', msg)

    def test_empty_string_invalid(self):
        ok, msg = self.validator.validate_int('', '年龄')
        self.assertFalse(ok)
        self.assertIn('不能为空', msg)

    def test_none_invalid(self):
        ok, msg = self.validator.validate_int(None, '年龄')
        self.assertFalse(ok)
        self.assertIn('不能为空', msg)

    def test_float_value_integer_valid(self):
        ok, msg = self.validator.validate_int(30.0, '年龄', 0, 120)
        self.assertTrue(ok)

    def test_float_value_non_integer_invalid(self):
        ok, msg = self.validator.validate_int(30.5, '年龄', 0, 120)
        self.assertFalse(ok)

    def test_bool_invalid(self):
        """布尔值是 int 子类但应视为非法"""
        ok, msg = self.validator.validate_int(True, '年龄')
        self.assertFalse(ok)
        self.assertIn('布尔值', msg)

    def test_tkinter_like_object(self):
        """模拟 tkinter 变量（带 get 方法）"""
        class FakeVar:
            def get(self):
                return 25
        ok, msg = self.validator.validate_int(FakeVar(), '年龄', 0, 120)
        self.assertTrue(ok)


class TestFieldValidatorRange(unittest.TestCase):
    """validate_range 范围检查"""

    def setUp(self):
        from tb_risk.gui.validators import FieldValidator
        self.validator = FieldValidator()

    def test_valid_range(self):
        ok, _ = self.validator.validate_range(50, 0, 100, '百分比')
        self.assertTrue(ok)

    def test_below_min(self):
        ok, msg = self.validator.validate_range(-1, 0, 100, '百分比')
        self.assertFalse(ok)
        self.assertIn('不能小于', msg)

    def test_above_max(self):
        ok, msg = self.validator.validate_range(101, 0, 100, '百分比')
        self.assertFalse(ok)
        self.assertIn('不能大于', msg)

    def test_string_numeric(self):
        ok, _ = self.validator.validate_range('45.5', 0, 100, '数值')
        self.assertTrue(ok)

    def test_non_numeric(self):
        ok, msg = self.validator.validate_range('abc', 0, 100, '数值')
        self.assertFalse(ok)
        self.assertIn('必须是数值', msg)


class TestFieldValidatorNonEmpty(unittest.TestCase):
    """validate_non_empty 非空检查"""

    def setUp(self):
        from tb_risk.gui.validators import FieldValidator
        self.validator = FieldValidator()

    def test_non_empty_string(self):
        ok, _ = self.validator.validate_non_empty('hello', '名称')
        self.assertTrue(ok)

    def test_empty_string(self):
        ok, msg = self.validator.validate_non_empty('', '名称')
        self.assertFalse(ok)
        self.assertIn('不能为空', msg)

    def test_whitespace_only(self):
        ok, msg = self.validator.validate_non_empty('   ', '名称')
        self.assertFalse(ok)

    def test_none(self):
        ok, msg = self.validator.validate_non_empty(None, '名称')
        self.assertFalse(ok)

    def test_empty_list(self):
        ok, msg = self.validator.validate_non_empty([], '列表')
        self.assertFalse(ok)

    def test_non_empty_list(self):
        ok, _ = self.validator.validate_non_empty([1, 2], '列表')
        self.assertTrue(ok)


class TestFieldValidatorPercentage(unittest.TestCase):
    """validate_percentage 百分比验证"""

    def setUp(self):
        from tb_risk.gui.validators import FieldValidator
        self.validator = FieldValidator()

    def test_valid_percentage(self):
        ok, _ = self.validator.validate_percentage(50, '感染比例')
        self.assertTrue(ok)

    def test_zero_valid(self):
        ok, _ = self.validator.validate_percentage(0, '感染比例')
        self.assertTrue(ok)

    def test_100_valid(self):
        ok, _ = self.validator.validate_percentage(100, '感染比例')
        self.assertTrue(ok)

    def test_above_100_invalid(self):
        ok, msg = self.validator.validate_percentage(101, '感染比例')
        self.assertFalse(ok)
        self.assertIn('100', msg)

    def test_negative_invalid(self):
        ok, msg = self.validator.validate_percentage(-1, '感染比例')
        self.assertFalse(ok)


class TestFieldValidatorCount(unittest.TestCase):
    """validate_count 数量上限检查"""

    def setUp(self):
        from tb_risk.gui.validators import FieldValidator
        self.validator = FieldValidator()

    def test_valid_count(self):
        ok, _ = self.validator.validate_count(10, 20, '家庭成员数')
        self.assertTrue(ok)

    def test_at_max(self):
        ok, _ = self.validator.validate_count(20, 20, '家庭成员数')
        self.assertTrue(ok)

    def test_above_max(self):
        ok, msg = self.validator.validate_count(25, 20, '家庭成员数')
        self.assertFalse(ok)
        self.assertIn('不能超过', msg)
        self.assertIn('20', msg)

    def test_negative_invalid(self):
        ok, msg = self.validator.validate_count(-1, 20, '家庭成员数')
        self.assertFalse(ok)
        self.assertIn('负数', msg)


class TestCrossValidation(unittest.TestCase):
    """交叉验证"""

    def setUp(self):
        from tb_risk.gui.validators import FieldValidator
        self.validator = FieldValidator()

    def test_smear_positive_with_symptoms(self):
        """涂阳+有症状：无警告"""
        entry = {'has_tb': '是', 'sputum_smear': '涂阳', 'has_symptoms': '是'}
        ok, msg = self.validator.cross_validate_smear_symptoms(entry)
        self.assertTrue(ok)
        self.assertEqual(msg, '')

    def test_smear_positive_without_symptoms_warns(self):
        """涂阳+无症状：软警告（is_valid=True 但带提示）"""
        entry = {'has_tb': '是', 'sputum_smear': '涂阳', 'has_symptoms': '否'}
        ok, msg = self.validator.cross_validate_smear_symptoms(entry)
        self.assertTrue(ok)  # 软警告不阻塞
        self.assertIn('提示', msg)

    def test_smear_negative_no_warning(self):
        entry = {'has_tb': '是', 'sputum_smear': '涂阴', 'has_symptoms': '否'}
        ok, msg = self.validator.cross_validate_smear_symptoms(entry)
        self.assertTrue(ok)
        self.assertEqual(msg, '')

    def test_no_tb_no_warning(self):
        entry = {'has_tb': '否', 'sputum_smear': '涂阳', 'has_symptoms': '否'}
        ok, msg = self.validator.cross_validate_smear_symptoms(entry)
        self.assertTrue(ok)
        self.assertEqual(msg, '')


class TestFormValidation(unittest.TestCase):
    """表单级预检"""

    def setUp(self):
        from tb_risk.gui.validators import FieldValidator
        self.validator = FieldValidator()

    def test_all_valid(self):
        checks = [
            {'field': 'age', 'value': 30, 'rules': ['int'], 'min': 0, 'max': 120},
            {'field': 'flp', 'value': 50, 'rules': ['int', 'range'], 'min': 0, 'max': 100},
        ]
        ok, errors = self.validator.validate_form(checks)
        self.assertTrue(ok)
        self.assertEqual(len(errors), 0)

    def test_one_invalid(self):
        checks = [
            {'field': 'age', 'value': 30, 'rules': ['int'], 'min': 0, 'max': 120},
            {'field': 'flp', 'value': 150, 'rules': ['int'], 'min': 0, 'max': 100},
        ]
        ok, errors = self.validator.validate_form(checks)
        self.assertFalse(ok)
        self.assertIn('flp', errors)
        self.assertIn('不能大于', errors['flp'])

    def test_multiple_invalid(self):
        checks = [
            {'field': 'age', 'value': -5, 'rules': ['int'], 'min': 0, 'max': 120},
            {'field': 'flp', 'value': 150, 'rules': ['int'], 'min': 0, 'max': 100},
        ]
        ok, errors = self.validator.validate_form(checks)
        self.assertFalse(ok)
        self.assertIn('age', errors)
        self.assertIn('flp', errors)

    def test_non_empty_rule(self):
        checks = [
            {'field': 'name', 'value': '', 'rules': ['non_empty']},
        ]
        ok, errors = self.validator.validate_form(checks)
        self.assertFalse(ok)
        self.assertIn('name', errors)

    def test_percentage_rule(self):
        checks = [
            {'field': 'pct', 'value': 50, 'rules': ['percentage']},
        ]
        ok, _ = self.validator.validate_form(checks)
        self.assertTrue(ok)

    def test_field_rules_auto_fill_min_max(self):
        """FIELD_RULES 自动补全 min/max"""
        checks = [
            {'field': 'age', 'value': 30, 'rules': ['int']},  # 不传 min/max
        ]
        ok, _ = self.validator.validate_form(checks)
        self.assertTrue(ok)

    def test_field_rules_auto_fill_invalid(self):
        checks = [
            {'field': 'age', 'value': 200, 'rules': ['int']},  # 超出 MAX_AGE
        ]
        ok, errors = self.validator.validate_form(checks)
        self.assertFalse(ok)
        self.assertIn('age', errors)

    def test_label_override(self):
        checks = [
            {'field': 'age', 'value': -5, 'rules': ['int'], 'label': '患者年龄'},
        ]
        ok, errors = self.validator.validate_form(checks)
        self.assertFalse(ok)
        self.assertIn('患者年龄', errors['age'])


class TestContactCountValidation(unittest.TestCase):
    """接触者数量上限检查"""

    def setUp(self):
        from tb_risk.gui.validators import FieldValidator
        self.validator = FieldValidator()

    def test_both_valid(self):
        ok, errors = self.validator.validate_contact_counts(10, 20)
        self.assertTrue(ok)

    def test_family_exceeds(self):
        ok, errors = self.validator.validate_contact_counts(25, 10)
        self.assertFalse(ok)
        self.assertIn('family_count', errors)

    def test_social_exceeds(self):
        ok, errors = self.validator.validate_contact_counts(10, 60)
        self.assertFalse(ok)
        self.assertIn('social_count', errors)

    def test_both_exceed(self):
        ok, errors = self.validator.validate_contact_counts(25, 60)
        self.assertFalse(ok)
        self.assertIn('family_count', errors)
        self.assertIn('social_count', errors)


class TestErrorStateManagement(unittest.TestCase):
    """错误状态管理"""

    def setUp(self):
        from tb_risk.gui.validators import FieldValidator
        self.validator = FieldValidator()

    def test_initial_no_errors(self):
        self.assertFalse(self.validator.has_errors())
        self.assertEqual(len(self.validator.get_errors()), 0)

    def test_get_first_error_none(self):
        self.assertIsNone(self.validator.get_first_error())

    def test_errors_after_form_validation(self):
        checks = [
            {'field': 'age', 'value': -5, 'rules': ['int'], 'min': 0, 'max': 120},
        ]
        self.validator.validate_form(checks)
        self.assertTrue(self.validator.has_errors())
        self.assertIn('age', self.validator.get_errors())
        self.assertIsNotNone(self.validator.get_first_error())

    def test_clear_all_errors(self):
        checks = [
            {'field': 'age', 'value': -5, 'rules': ['int'], 'min': 0, 'max': 120},
            {'field': 'flp', 'value': 150, 'rules': ['int'], 'min': 0, 'max': 100},
        ]
        self.validator.validate_form(checks)
        self.assertTrue(self.validator.has_errors())
        self.validator.clear_all_errors()
        self.assertFalse(self.validator.has_errors())
        self.assertEqual(len(self.validator.get_errors()), 0)

    def test_get_errors_returns_copy(self):
        """get_errors 返回副本，修改不影响内部状态"""
        checks = [
            {'field': 'age', 'value': -5, 'rules': ['int'], 'min': 0, 'max': 120},
        ]
        self.validator.validate_form(checks)
        errors = self.validator.get_errors()
        errors['new'] = 'injected'
        self.assertNotIn('new', self.validator.get_errors())


class TestFieldRulesConstants(unittest.TestCase):
    """FIELD_RULES / COUNT_RULES 常量一致性"""

    def test_field_rules_contains_age(self):
        from tb_risk.gui.validators import FIELD_RULES
        self.assertIn('age', FIELD_RULES)

    def test_field_rules_age_range_matches_constants(self):
        from tb_risk.gui.validators import FIELD_RULES
        from tb_risk.constants import MIN_AGE, MAX_AGE
        rule_min, rule_max, is_int, label = FIELD_RULES['age']
        self.assertEqual(rule_min, MIN_AGE)
        self.assertEqual(rule_max, MAX_AGE)
        self.assertTrue(is_int)

    def test_field_rules_contains_all_expected(self):
        from tb_risk.gui.validators import FIELD_RULES
        expected = {'age', 'single_duration', 'freq_density', 'time_span',
                    'ventilation', 'cough_freq', 'treatment_duration',
                    'delay_days', 'flp_percentage', 'hrsp_percentage'}
        self.assertEqual(set(FIELD_RULES.keys()), expected)

    def test_count_rules_matches_constants(self):
        from tb_risk.gui.validators import COUNT_RULES
        from tb_risk.constants import MAX_FAMILY_MEMBERS, MAX_SOCIAL_CONTACTS
        self.assertEqual(COUNT_RULES['family_count'][0], MAX_FAMILY_MEMBERS)
        self.assertEqual(COUNT_RULES['social_count'][0], MAX_SOCIAL_CONTACTS)

    def test_all_field_rules_is_int_true(self):
        from tb_risk.gui.validators import FIELD_RULES
        for field, (_, _, is_int, _) in FIELD_RULES.items():
            self.assertTrue(is_int, f'{field} 应为整数验证')


class TestConstantsSingleSource(unittest.TestCase):
    """验证 constants.py 为唯一真相源"""

    def test_assessment_imports_from_constants(self):
        """assessment.py 类属性应与 constants.py 一致"""
        from tb_risk.constants import (
            MAX_AGE, MAX_SINGLE_DURATION_MINUTES, MAX_FAMILY_MEMBERS,
            MAX_SOCIAL_CONTACTS, MAX_COUGH_FREQ, MAX_DELAY_DAYS,
            MAX_FREQ_DENSITY, MAX_TIME_SPAN_WEEKS, MAX_VENTILATION,
            MIN_VENTILATION, MIN_AGE,
        )
        # 验证 assessment.py 中的类属性值与 constants.py 一致
        # 通过导入检查不抛异常即可（类属性赋值在模块加载时完成）
        import tb_risk.assessment as asm
        self.assertEqual(asm.TB_Risk_Assessment.MAX_AGE, MAX_AGE)
        self.assertEqual(asm.TB_Risk_Assessment.MIN_AGE, MIN_AGE)
        self.assertEqual(asm.TB_Risk_Assessment.MAX_SINGLE_DURATION_MINUTES, MAX_SINGLE_DURATION_MINUTES)
        self.assertEqual(asm.TB_Risk_Assessment.MAX_FAMILY_MEMBERS, MAX_FAMILY_MEMBERS)
        self.assertEqual(asm.TB_Risk_Assessment.MAX_SOCIAL_CONTACTS, MAX_SOCIAL_CONTACTS)
        self.assertEqual(asm.TB_Risk_Assessment.MAX_COUGH_FREQ, MAX_COUGH_FREQ)
        self.assertEqual(asm.TB_Risk_Assessment.MAX_DELAY_DAYS, MAX_DELAY_DAYS)
        self.assertEqual(asm.TB_Risk_Assessment.MAX_FREQ_DENSITY, MAX_FREQ_DENSITY)
        self.assertEqual(asm.TB_Risk_Assessment.MAX_TIME_SPAN_WEEKS, MAX_TIME_SPAN_WEEKS)
        self.assertEqual(asm.TB_Risk_Assessment.MAX_VENTILATION, MAX_VENTILATION)
        self.assertEqual(asm.TB_Risk_Assessment.MIN_VENTILATION, MIN_VENTILATION)

    def test_cli_imports_from_constants(self):
        """cli/interactive.py 类属性应与 constants.py 一致"""
        from tb_risk.constants import (
            MAX_AGE, MAX_SINGLE_DURATION_MINUTES, MAX_FAMILY_MEMBERS,
            MAX_SOCIAL_CONTACTS, MAX_COUGH_FREQ, MAX_DELAY_DAYS,
            MAX_FREQ_DENSITY, MAX_TIME_SPAN_WEEKS,
        )
        from tb_risk.cli.interactive import InteractivePatientCollector
        self.assertEqual(InteractivePatientCollector.MAX_AGE, MAX_AGE)
        self.assertEqual(InteractivePatientCollector.MAX_SINGLE_DURATION_MINUTES, MAX_SINGLE_DURATION_MINUTES)
        self.assertEqual(InteractivePatientCollector.MAX_FAMILY_MEMBERS, MAX_FAMILY_MEMBERS)
        self.assertEqual(InteractivePatientCollector.MAX_SOCIAL_CONTACTS, MAX_SOCIAL_CONTACTS)
        self.assertEqual(InteractivePatientCollector.MAX_COUGH_FREQ, MAX_COUGH_FREQ)
        self.assertEqual(InteractivePatientCollector.MAX_DELAY_DAYS, MAX_DELAY_DAYS)
        self.assertEqual(InteractivePatientCollector.MAX_FREQ_DENSITY, MAX_FREQ_DENSITY)
        self.assertEqual(InteractivePatientCollector.MAX_TIME_SPAN_WEEKS, MAX_TIME_SPAN_WEEKS)


if __name__ == '__main__':
    unittest.main()
