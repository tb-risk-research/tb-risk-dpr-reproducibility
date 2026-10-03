#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - 核心业务层测试 (RiskAssessmentService)
"""

import sys
import os
import unittest

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class TestRiskAssessmentService(unittest.TestCase):
    """RiskAssessmentService 端到端测试"""

    _SAMPLE_PATIENT = {
        'basic_info': {
            'sputum_smear': 1,
            'has_cavity': 1,
            'cough_freq': 8,
            'symptoms': 2,
            'delay_days': 10,
            'family_living_conditions': 3,
            'flp_percentage': 10,
            'hrsp_percentage': 15,
            'treatment_duration': 2,
        }
    }

    _SAMPLE_FAMILY = [
        {
            'name': '家庭成员A',
            'relationship': '配偶',
            'age': 35,
            'contact_distance': 'close',
            'ventilation': 3,
            'exposure_setting': 'general',
            'freq_density': 14,
            'single_duration': 120,
            'time_span': 4,
            'has_symptoms': 0,
            'has_tb': 0,
            'bcg_vaccine': 1,
            'ethnicity': '汉族',
            'origin_altitude': 350,
        },
        {
            'name': '家庭成员B',
            'relationship': '子女',
            'age': 8,
            'contact_distance': 'close',
            'ventilation': 3,
            'exposure_setting': 'general',
            'freq_density': 14,
            'single_duration': 60,
            'time_span': 4,
            'has_symptoms': 0,
            'has_tb': 0,
            'bcg_vaccine': 1,
            'ethnicity': '汉族',
            'origin_altitude': 350,
        },
    ]

    _SAMPLE_SOCIAL = [
        {
            'name': '同事A',
            'relationship': '同事',
            'age': 42,
            'contact_distance': 'medium',
            'ventilation': 3,
            'exposure_setting': 'general',
            'freq_density': 5,
            'single_duration': 60,
            'time_span': 4,
            'has_symptoms': 0,
            'has_tb': 0,
            'bcg_vaccine': 0,
            'ethnicity': '汉族',
            'origin_altitude': 350,
        },
    ]

    def test_service_import(self):
        """测试 RiskAssessmentService 可导入"""
        from tb_risk.core import RiskAssessmentService
        service = RiskAssessmentService()
        self.assertIsNotNone(service)

    def test_assess_with_empty_data(self):
        """测试评估空数据不会崩溃"""
        from tb_risk.core import RiskAssessmentService
        service = RiskAssessmentService()
        result = service.assess()
        self.assertIn('potential_patients', result)
        self.assertIn('summary', result)
        self.assertIn('patient_score', result)
        self.assertEqual(result['patient_score'], 0.0)
        self.assertEqual(len(result['potential_patients']['family']), 0)
        self.assertEqual(len(result['potential_patients']['social']), 0)

    def test_assess_with_patient_only(self):
        """测试仅患者数据的评估"""
        from tb_risk.core import RiskAssessmentService
        service = RiskAssessmentService()
        result = service.assess(patient_info=self._SAMPLE_PATIENT)
        self.assertIn('patient_score', result)
        self.assertGreater(result['patient_score'], 0.0,
                           "有症状的患者评分应 > 0")

    def test_assess_with_full_data(self):
        """测试完整患者+接触者评估"""
        from tb_risk.core import RiskAssessmentService
        service = RiskAssessmentService()
        result = service.assess(
            patient_info=self._SAMPLE_PATIENT,
            family_members=self._SAMPLE_FAMILY,
            social_contacts=self._SAMPLE_SOCIAL,
        )
        self.assertIn('potential_patients', result)
        family_pp = result['potential_patients']['family']
        social_pp = result['potential_patients']['social']
        self.assertEqual(len(family_pp), 2, "应有 2 个家庭潜在患者")
        self.assertEqual(len(social_pp), 1, "应有 1 个社会潜在患者")

        # 验证每个潜在患者有正确的字段
        for pp in family_pp + social_pp:
            self.assertIn('name', pp)
            self.assertIn('risk_score', pp)
            self.assertIn('infection_probability', pp)
            self.assertIn('disease_probability', pp)
            self.assertIn('priority', pp)
            self.assertIn('recommendation', pp)
            # 概率应在 [0, 100] 区间
            self.assertGreaterEqual(pp['infection_probability'], 0.0)
            self.assertLessEqual(pp['infection_probability'], 100.0)
            self.assertGreaterEqual(pp['disease_probability'], 0.0)
            self.assertLessEqual(pp['disease_probability'], 100.0)

    def test_assess_result_structure(self):
        """测试评估结果结构完整性"""
        from tb_risk.core import RiskAssessmentService
        service = RiskAssessmentService()
        result = service.assess(
            patient_info=self._SAMPLE_PATIENT,
            family_members=self._SAMPLE_FAMILY,
        )
        required_keys = {'potential_patients', 'summary', 'patient_score',
                         'ml_results', 'seir_results'}
        for key in required_keys:
            self.assertIn(key, result, f"缺少结果键: {key}")

    def test_get_last_results(self):
        """测试获取最近评估结果"""
        from tb_risk.core import RiskAssessmentService
        service = RiskAssessmentService()
        service.assess(patient_info=self._SAMPLE_PATIENT)
        last = service.get_last_results()
        self.assertIn('patient_score', last)
        self.assertGreater(last['patient_score'], 0.0)

    def test_get_last_summary(self):
        """测试获取最近评估摘要"""
        from tb_risk.core import RiskAssessmentService
        service = RiskAssessmentService()
        service.assess(patient_info=self._SAMPLE_PATIENT)
        summary = service.get_last_summary()
        self.assertIsNotNone(summary)

    def test_public_api_methods(self):
        """测试公开 API 方法均可调用"""
        from tb_risk.core import RiskAssessmentService
        service = RiskAssessmentService()

        # calculate_time_dependent_risk
        risk = service.calculate_time_dependent_risk(
            {'time_span': 4}, patient_treatment_days=0)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)

        # calculate_bcg_protection
        protection, note = service.calculate_bcg_protection(age=10, bcg_vaccinated=True)
        self.assertGreater(protection, 0.0)
        self.assertIsInstance(note, str)

        no_protection, _ = service.calculate_bcg_protection(age=10, bcg_vaccinated=False)
        self.assertEqual(no_protection, 0.0)

        # calculate_fci_score
        fci = service.calculate_fci_score(self._SAMPLE_FAMILY, family_living_conditions=3)
        self.assertGreaterEqual(fci, 0.0)
        self.assertLessEqual(fci, 10.0)

        # calculate_snc_score
        snc = service.calculate_snc_score(self._SAMPLE_SOCIAL)
        self.assertGreaterEqual(snc, 0.0)
        self.assertLessEqual(snc, 10.0)

        # calculate_patient_risk_multiplier
        multiplier = service.calculate_patient_risk_multiplier(
            sputum_smear=1, has_cavity=1, untreated=True)
        self.assertGreater(multiplier, 1.0)

    def test_contact_labels(self):
        """测试接触者标签对诊断状态的影响"""
        from tb_risk.core import RiskAssessmentService
        contact_labels = {'member_1': 1}  # 标记为已诊断
        family = [
            {
                '_id': 'member_1',
                'name': '已诊断成员',
                'relationship': '配偶',
                'age': 35,
                'contact_distance': 'close',
                'ventilation': 3,
                'exposure_setting': 'general',
                'freq_density': 14,
                'single_duration': 120,
                'time_span': 4,
            },
        ]
        service = RiskAssessmentService(contact_labels=contact_labels)
        result = service.assess(family_members=family)
        pp = result['potential_patients']['family'][0]
        self.assertTrue(pp['is_diagnosed'],
                        "标记为已诊断的接触者应显示 is_diagnosed=True")


class TestProbabilityFunctions(unittest.TestCase):
    """core/probability.py 概率函数边界测试"""

    def test_sigmoid_base_infection_low_input(self):
        """测试 sigmoid 在低风险评分下的行为"""
        from tb_risk.core import sigmoid_base_infection
        prob = sigmoid_base_infection(0.0)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLess(prob, 50.0, "零风险评分下感染概率应较低")

    def test_sigmoid_base_infection_high_input(self):
        """测试 sigmoid 在高风险评分下的行为"""
        from tb_risk.core import sigmoid_base_infection
        prob = sigmoid_base_infection(100.0)
        self.assertGreater(prob, 50.0, "高风险评分下感染概率应较高")

    def test_sigmoid_base_infection_negative(self):
        """测试 sigmoid 处理负数输入"""
        from tb_risk.core import sigmoid_base_infection
        prob = sigmoid_base_infection(-10.0)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLess(prob, 100.0)

    def test_sigmoid_total_score_bounds(self):
        """测试 sigmoid_total_score 输出在 [0, 95] 区间"""
        from tb_risk.core import sigmoid_total_score
        for x in [-100, -10, 0, 10, 50, 100, 1000]:
            prob = sigmoid_total_score(x)
            self.assertGreaterEqual(prob, 0.0, f"x={x}: prob={prob}")
            self.assertLessEqual(prob, 95.0, f"x={x}: prob={prob}")

    def test_exposure_to_infection_probability_zero(self):
        """测试暴露时长为 0 时感染概率为 0"""
        from tb_risk.core import exposure_to_infection_probability
        prob = exposure_to_infection_probability(0.0)
        self.assertEqual(prob, 0.0)

    def test_exposure_to_infection_probability_large(self):
        """测试暴露时长极大时感染概率接近 1.0"""
        from tb_risk.core import exposure_to_infection_probability
        prob = exposure_to_infection_probability(10000)
        self.assertGreater(prob, 0.99, "暴露 10000 小时感染概率应接近 1.0")
        self.assertLessEqual(prob, 1.0)

    def test_exposure_to_infection_probability_negative(self):
        """测试暴露时长为负时返回 0"""
        from tb_risk.core import exposure_to_infection_probability
        prob = exposure_to_infection_probability(-10)
        self.assertEqual(prob, 0.0)

    def test_safe_float(self):
        """测试 safe_float 转换"""
        from tb_risk.core import safe_float
        self.assertEqual(safe_float("3.14"), 3.14)
        self.assertEqual(safe_float(42), 42.0)
        self.assertIsNone(safe_float("abc"))
        self.assertIsNone(safe_float(None))
        self.assertEqual(safe_float("abc", 0.0), 0.0)

    def test_safe_int(self):
        """测试 safe_int 转换"""
        from tb_risk.core import safe_int
        self.assertEqual(safe_int("5"), 5)
        self.assertEqual(safe_int(3.7), 3)
        self.assertIsNone(safe_int("abc"))
        self.assertEqual(safe_int("abc", 0), 0)

    def test_calculate_cumulative_exposure(self):
        """测试累积暴露计算"""
        from tb_risk.core import calculate_cumulative_exposure
        # 120分钟 × 5次/周 × 4周 = 2400分钟 = 40小时
        exp = calculate_cumulative_exposure(120, 5, 4)
        self.assertAlmostEqual(exp, 40.0, places=1)

    def test_calculate_cumulative_exposure_zero(self):
        """测试零输入的累积暴露"""
        from tb_risk.core import calculate_cumulative_exposure
        self.assertEqual(calculate_cumulative_exposure(0, 0, 0), 0.0)
        self.assertEqual(calculate_cumulative_exposure(None, None, None), 0.0)

    def test_calculate_cumulative_exposure_oilfield(self):
        """测试油田场景的累积暴露（修正系数 0.70）"""
        from tb_risk.core import calculate_cumulative_exposure
        exp_normal = calculate_cumulative_exposure(120, 5, 4, workplace_type='非油田')
        exp_oilfield = calculate_cumulative_exposure(120, 5, 4, workplace_type='油田')
        self.assertAlmostEqual(exp_oilfield, exp_normal * 0.70, places=1)


class TestDataConversion(unittest.TestCase):
    """core/data_conversion.py 数据转换函数测试"""

    def test_convert_chinese_to_value_exact(self):
        """测试中文精确匹配转换"""
        from tb_risk.core import convert_chinese_to_value, DISTANCE_TEXT_MAP
        result = convert_chinese_to_value('近', DISTANCE_TEXT_MAP)
        self.assertEqual(result, 'close')

    def test_convert_chinese_to_value_fuzzy(self):
        """测试中文模糊匹配转换"""
        from tb_risk.core import convert_chinese_to_value, SETTING_TEXT_MAP
        result = convert_chinese_to_value('拥挤', SETTING_TEXT_MAP)
        self.assertEqual(result, 'crowded')

    def test_convert_chinese_to_value_not_found(self):
        """测试未找到映射时返回原值"""
        from tb_risk.core import convert_chinese_to_value, DISTANCE_TEXT_MAP
        result = convert_chinese_to_value('xyz_unknown', DISTANCE_TEXT_MAP)
        self.assertEqual(result, 'xyz_unknown')

    def test_convert_chinese_to_value_none(self):
        """测试 None 输入"""
        from tb_risk.core import convert_chinese_to_value, DISTANCE_TEXT_MAP
        result = convert_chinese_to_value(None, DISTANCE_TEXT_MAP)
        self.assertIsNone(result)

    def test_standardize_contact_fields(self):
        """测试接触者字段标准化"""
        from tb_risk.core import standardize_contact_fields
        contact = {
            'contact_distance': '近',
            'exposure_setting': '拥挤场所',
            'ventilation': '好',
            'has_tb': '否',
            'bcg_vaccine': '已接种',
        }
        standardized = standardize_contact_fields(contact, 'family')
        self.assertEqual(standardized['contact_distance'], 'close')
        self.assertEqual(standardized['exposure_setting'], 'crowded')
        self.assertEqual(standardized['ventilation'], 4)
        self.assertEqual(standardized['has_tb'], 0)
        self.assertEqual(standardized['bcg_vaccine'], 1)

    def test_apply_scenario_defaults(self):
        """测试场景默认值填充"""
        from tb_risk.core import apply_scenario_defaults
        contact = {'name': '测试'}
        filled = apply_scenario_defaults(contact, 'family')
        self.assertEqual(filled['contact_distance'], 'close')
        self.assertEqual(filled['ventilation'], 3)
        self.assertEqual(filled['freq_density'], 14)

    def test_get_priority_and_recommendation(self):
        """测试优先级和推荐方案"""
        from tb_risk.core import get_priority_and_recommendation
        priority, rec = get_priority_and_recommendation(85.0)
        self.assertEqual(priority, '极高')
        self.assertIn('分子生物学', rec)

        priority, rec = get_priority_and_recommendation(5.0)
        self.assertEqual(priority, '极低')
        self.assertIn('随访', rec)

    def test_get_priority_boundaries(self):
        """测试优先级各分档边界"""
        from tb_risk.core import get_priority_and_recommendation
        self.assertEqual(get_priority_and_recommendation(80)[0], '极高')
        self.assertEqual(get_priority_and_recommendation(79.99)[0], '高')
        self.assertEqual(get_priority_and_recommendation(60)[0], '高')
        self.assertEqual(get_priority_and_recommendation(59.99)[0], '较高')
        self.assertEqual(get_priority_and_recommendation(40)[0], '较高')
        self.assertEqual(get_priority_and_recommendation(39.99)[0], '中')
        self.assertEqual(get_priority_and_recommendation(20)[0], '中')
        self.assertEqual(get_priority_and_recommendation(19.99)[0], '低')
        self.assertEqual(get_priority_and_recommendation(10)[0], '低')
        self.assertEqual(get_priority_and_recommendation(9.99)[0], '极低')

    def test_calculate_age_group(self):
        """测试年龄分组泛化"""
        from tb_risk.core.data_conversion import calculate_age_group
        self.assertEqual(calculate_age_group(3), '0-5')
        self.assertEqual(calculate_age_group(10), '6-15')
        self.assertEqual(calculate_age_group(20), '16-35')
        self.assertEqual(calculate_age_group(40), '36-55')
        self.assertEqual(calculate_age_group(60), '56-65')
        self.assertEqual(calculate_age_group(80), '65+')
        self.assertEqual(calculate_age_group(6), '6-15')
        self.assertEqual(calculate_age_group(5), '0-5')
        self.assertEqual(calculate_age_group('abc'), '未知')
        self.assertEqual(calculate_age_group(None), '未知')

    def test_get_exposure_risk_score(self):
        """测试暴露风险评分分级"""
        from tb_risk.core import get_exposure_risk_score
        # 无暴露 / 极小暴露 → 0
        self.assertEqual(get_exposure_risk_score(0), 0)
        self.assertEqual(get_exposure_risk_score(None), 0)
        # 无效输入 → 0
        self.assertEqual(get_exposure_risk_score('abc'), 0)
        # 非负且在合法分档内
        for hours in (0, 10, 50, 100, 500, 1000):
            score = get_exposure_risk_score(hours)
            self.assertIn(score, (0, 1, 2, 3, 4), f"hours={hours}")
        # 单调不减
        prev = 0
        for hours in (0, 20, 40, 80, 160, 320, 640):
            cur = get_exposure_risk_score(hours)
            self.assertGreaterEqual(cur, prev, f"hours={hours}")
            prev = cur
        # 极大暴露达到最高档
        self.assertEqual(get_exposure_risk_score(1e9), 4)


class TestGUICLIConsistency(unittest.TestCase):
    """GUI/CLI 结果一致性验证测试

    验证 GUI 路径（TB_Risk_Assessment.assess_risk）和 CLI 路径
    （RiskAssessmentService.assess）对相同输入产生相同结果。

    文献支撑：
    - 当两条路径合并到同一 core 实现后，风险概率应完全一致。
    - 浮点运算路径可能存在微小差异，设置容差阈值 0.01。
    """

    _PATIENT = {
        'basic_info': {
            'sputum_smear': 1,
            'has_cavity': 1,
            'cough_freq': 8,
            'symptoms': 2,
            'delay_days': 10,
            'family_living_conditions': 3,
            'flp_percentage': 10,
            'hrsp_percentage': 15,
            'treatment_duration': 2,
        }
    }

    _FAMILY = [
        {
            'name': '家庭成员A',
            'relationship': '配偶',
            'age': 35,
            'contact_distance': 'close',
            'ventilation': 3,
            'exposure_setting': 'general',
            'freq_density': 14,
            'single_duration': 120,
            'time_span': 4,
            'has_symptoms': 0,
            'has_tb': 0,
            'bcg_vaccine': 1,
        },
        {
            'name': '家庭成员B',
            'relationship': '子女',
            'age': 8,
            'contact_distance': 'close',
            'ventilation': 3,
            'exposure_setting': 'general',
            'freq_density': 14,
            'single_duration': 60,
            'time_span': 4,
            'has_symptoms': 0,
            'has_tb': 0,
            'bcg_vaccine': 1,
        },
    ]

    _SOCIAL = [
        {
            'name': '同事A',
            'relationship': '同事',
            'age': 42,
            'contact_distance': 'medium',
            'ventilation': 3,
            'exposure_setting': 'general',
            'freq_density': 5,
            'single_duration': 60,
            'time_span': 4,
            'has_symptoms': 0,
            'has_tb': 0,
            'bcg_vaccine': 0,
        },
        {
            'name': '邻居B',
            'relationship': '邻居',
            'age': 55,
            'contact_distance': 'far',
            'ventilation': 4,
            'exposure_setting': 'outdoor',
            'freq_density': 2,
            'single_duration': 30,
            'time_span': 2,
            'has_symptoms': 0,
            'has_tb': 0,
            'bcg_vaccine': 1,
        },
    ]

    def _assert_contact_consistency(self, gui_contacts, cli_contacts, label):
        """验证两组接触者数据的一致性（概率容差 0.01，优先级完全一致）"""
        self.assertEqual(len(gui_contacts), len(cli_contacts),
                         f"{label}数量应一致: GUI={len(gui_contacts)}, CLI={len(cli_contacts)}")
        for i, (g, c) in enumerate(zip(gui_contacts, cli_contacts)):
            with self.subTest(contact=f"{label}[{i}] {g.get('name', '?')}"):
                self.assertEqual(g.get('name'), c.get('name'),
                                 f"名称不一致: GUI={g.get('name')}, CLI={c.get('name')}")
                self.assertAlmostEqual(
                    g.get('infection_probability', 0),
                    c.get('infection_probability', 0),
                    delta=0.01,
                    msg=(f"感染概率不一致: GUI={g.get('infection_probability')}, "
                         f"CLI={c.get('infection_probability')}")
                )
                self.assertAlmostEqual(
                    g.get('disease_probability', 0),
                    c.get('disease_probability', 0),
                    delta=0.01,
                    msg=(f"发病概率不一致: GUI={g.get('disease_probability')}, "
                         f"CLI={c.get('disease_probability')}")
                )
                self.assertEqual(
                    g.get('priority'), c.get('priority'),
                    f"优先级分类不一致: GUI={g.get('priority')}, CLI={c.get('priority')}"
                )

    def test_gui_cli_full_consistency(self):
        """验证完整数据（患者+2家庭+2社会）的 GUI/CLI 结果一致性"""
        from tb_risk.core import RiskAssessmentService
        from tb_risk.assessment import TB_Risk_Assessment

        # ---- CLI 路径 ----
        cli_service = RiskAssessmentService()
        cli_result = cli_service.assess(
            patient_info=self._PATIENT,
            family_members=self._FAMILY,
            social_contacts=self._SOCIAL,
        )

        # ---- GUI 路径 ----
        gui_app = TB_Risk_Assessment(random_state=42)
        gui_app.patient_info = self._PATIENT
        gui_app.family_members = self._FAMILY
        gui_app.social_contacts = self._SOCIAL
        gui_app.assess_risk()

        gui_pp = gui_app.results.get('potential_patients', {})
        cli_pp = cli_result.get('potential_patients', {})

        # 比较家庭接触者
        self._assert_contact_consistency(
            gui_pp.get('family', []), cli_pp.get('family', []), '家庭接触者')

        # 比较社会接触者
        self._assert_contact_consistency(
            gui_pp.get('social', []), cli_pp.get('social', []), '社会接触者')

    def test_gui_cli_patient_score_consistency(self):
        """验证仅患者数据（无接触者）的 GUI/CLI 患者评分一致"""
        from tb_risk.core import RiskAssessmentService
        from tb_risk.assessment import TB_Risk_Assessment

        cli_service = RiskAssessmentService()
        cli_result = cli_service.assess(patient_info=self._PATIENT)

        gui_app = TB_Risk_Assessment(random_state=42)
        gui_app.patient_info = self._PATIENT
        gui_app.family_members = []
        gui_app.social_contacts = []
        gui_app.assess_risk()

        cli_score = cli_result.get('patient_score', 0)
        gui_score = gui_app.results.get('total_score', 0)
        self.assertAlmostEqual(
            gui_score, cli_score, delta=0.01,
            msg=f"患者评分不一致: GUI={gui_score}, CLI={cli_score}")

    @pytest.mark.slow
    def test_gui_cli_empty_data_consistency(self):
        """验证空数据评估的 GUI/CLI 结果一致性"""
        from tb_risk.core import RiskAssessmentService
        from tb_risk.assessment import TB_Risk_Assessment

        cli_service = RiskAssessmentService()
        cli_result = cli_service.assess()

        gui_app = TB_Risk_Assessment(random_state=42)
        gui_app.patient_info = {}
        gui_app.family_members = []
        gui_app.social_contacts = []
        gui_app.assess_risk()

        cli_score = cli_result.get('patient_score', 0)
        gui_score = gui_app.results.get('total_score', 0)
        self.assertEqual(gui_score, cli_score,
                         f"空数据患者评分应一致: GUI={gui_score}, CLI={cli_score}")
        self.assertEqual(len(gui_app.results.get('potential_patients', {}).get('family', [])),
                         len(cli_result.get('potential_patients', {}).get('family', [])))
        self.assertEqual(len(gui_app.results.get('potential_patients', {}).get('social', [])),
                         len(cli_result.get('potential_patients', {}).get('social', [])))


if __name__ == '__main__':
    unittest.main()