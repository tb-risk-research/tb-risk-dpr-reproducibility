#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - 核心风险评估模型协作类单元测试

针对 core/ 子包中仅被端到端测试间接覆盖的协作类做直接单元测试：
- PatientScorer            （患者评分 / FCI / SNC / 个体风险等级）
- ContactRiskCalculator    （接触者风险 / BCG 保护 / 潜在患者生成）
- SEIRIntegration          （时间依赖风险 / SEIR 概率 / 置信度权重 / 融合）
- fuse_probabilities       （SEIR/规则概率 log 空间融合边界）
- EnsembleWeightOptimizer  （集成权重优化器编排）
- init_seir_models / init_advanced_features（初始化器）
- run_ml_prediction_loop   （ML 预测运行器）

遵循项目 unittest + pytest 约定，对可选依赖做 skip 保护。
"""
import os
import sys
import threading
import unittest

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


def _make_family_member(**overrides):
    """构造标准家庭接触者样本。"""
    base = {
        'name': '家庭成员A', 'relationship': '配偶', 'age': 35,
        'contact_distance': 'close', 'ventilation': 3,
        'exposure_setting': 'general', 'freq_density': 14,
        'single_duration': 120, 'time_span': 4, 'cumulative_exposure': 40,
        'has_symptoms': 0, 'has_tb': 0, 'bcg_vaccine': 1,
        'ethnicity': '汉族', 'origin_altitude': 350,
    }
    base.update(overrides)
    return base


def _make_social_contact(**overrides):
    """构造标准社会接触者样本。"""
    base = {
        'name': '同事A', 'relationship': '同事', 'age': 42,
        'contact_distance': 'medium', 'ventilation': 3,
        'exposure_setting': 'general', 'freq_density': 5,
        'single_duration': 60, 'time_span': 4, 'cumulative_exposure': 10,
        'has_symptoms': 0, 'has_tb': 0, 'bcg_vaccine': 0,
        'ethnicity': '汉族', 'origin_altitude': 350,
    }
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════
# PatientScorer
# ══════════════════════════════════════════════════════════════

class TestPatientScorerPreprocess(unittest.TestCase):
    """患者/接触者预处理测试。"""

    def setUp(self):
        from tb_risk.core.patient_scorer import PatientScorer
        self.scorer = PatientScorer()

    def test_preprocess_patient_none(self):
        self.assertEqual(self.scorer.preprocess_patient(None), {})
        self.assertEqual(self.scorer.preprocess_patient({}), {})

    def test_preprocess_patient_treatment_duration_normalized(self):
        info = {'basic_info': {'treatment_duration': '2'}}
        out = self.scorer.preprocess_patient(info)
        self.assertEqual(out['basic_info']['treatment_duration'], 2.0)

    def test_preprocess_patient_deep_copy(self):
        info = {'basic_info': {'treatment_duration': '2'}}
        out = self.scorer.preprocess_patient(info)
        out['basic_info']['treatment_duration'] = 99.0
        # 深拷贝：原字典不受影响
        self.assertEqual(info['basic_info']['treatment_duration'], '2')

    def test_preprocess_contacts_adds_cumulative_exposure(self):
        member = _make_family_member()
        # 移除累积暴露，验证自动补全
        member.pop('cumulative_exposure', None)
        processed = self.scorer.preprocess_contacts([member], 'family')
        self.assertIn('cumulative_exposure', processed[0])
        self.assertGreater(processed[0]['cumulative_exposure'], 0.0)

    def test_preprocess_contacts_standardizes_chinese(self):
        member = {
            'contact_distance': '近', 'exposure_setting': '拥挤场所',
            'ventilation': '好', 'bcg_vaccine': '已接种',
        }
        processed = self.scorer.preprocess_contacts([member], 'family')
        self.assertEqual(processed[0]['contact_distance'], 'close')
        self.assertEqual(processed[0]['exposure_setting'], 'crowded')

    def test_get_treatment_days(self):
        days = self.scorer.get_treatment_days(
            {'basic_info': {'treatment_duration': 2}})
        self.assertAlmostEqual(days, 2 * self.scorer.DAYS_PER_MONTH_AVG, places=6)

    def test_get_treatment_days_empty(self):
        self.assertEqual(self.scorer.get_treatment_days({}), 0.0)
        self.assertEqual(self.scorer.get_treatment_days(None), 0.0)


class TestPatientScorerScoring(unittest.TestCase):
    """患者总体评分测试。"""

    def setUp(self):
        from tb_risk.core.patient_scorer import PatientScorer
        self.scorer = PatientScorer()

    def test_compute_patient_score_empty(self):
        result = self.scorer.compute_patient_score({})
        self.assertEqual(result['total_score'], 0.0)
        self.assertEqual(result['base_infection_probability'], 0.0)
        self.assertEqual(result['patient_risk_multiplier'], 1.0)
        self.assertIn('weights', result)

    def test_compute_patient_score_full_bounds(self):
        info = {
            'FCI': 8.0, 'SNC': 6.0,
            'basic_info': {
                'age': 30, 'sputum_smear': 2, 'has_cavity': 2,
                'active_tb': 1, 'treatment': 1,
                'cough_freq': 10, 'symptoms': 3, 'delay_days': 40,
                'flp_percentage': 80, 'hrsp_percentage': 70,
            },
        }
        result = self.scorer.compute_patient_score(info)
        # 总评分非负，且感染概率在合法区间
        self.assertGreaterEqual(result['total_score'], 0.0)
        self.assertLessEqual(result['total_score'], 100.0)
        self.assertGreaterEqual(result['base_infection_probability'], 0.0)
        self.assertLessEqual(result['base_infection_probability'], 100.0)
        self.assertGreaterEqual(result['patient_risk_multiplier'], 1.0)

    def test_compute_patient_score_weights_normalized(self):
        info = {
            'FCI': 5.0, 'SNC': 5.0,
            'basic_info': {'age': 40},
        }
        result = self.scorer.compute_patient_score(info)
        weights = result['weights']
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        for dim in ('FCI', 'SNC', 'FLP', 'HRSP', 'FTD'):
            self.assertIn(dim, weights)

    def test_compute_patient_score_increases_with_risk(self):
        low = {'FCI': 1.0, 'basic_info': {'age': 40}}
        high = {'FCI': 9.0, 'basic_info': {'age': 40}}
        low_prob = self.scorer.compute_patient_score(low)['base_infection_probability']
        high_prob = self.scorer.compute_patient_score(high)['base_infection_probability']
        self.assertGreater(high_prob, low_prob)


class TestPatientScorerIndividualRisks(unittest.TestCase):
    """各维度风险等级判定测试。"""

    def setUp(self):
        from tb_risk.core.patient_scorer import PatientScorer
        self.scorer = PatientScorer()

    def test_overall_risk_high(self):
        result = self.scorer.compute_individual_risks(
            {}, fci_score=0, snc_score=0, flp_percentage=0,
            hrsp_percentage=0, ftd_days=0,
            base_infection_probability=80.0)
        self.assertEqual(result['overall_risk'], '高风险')
        self.assertIn('立即', result['overall_suggestion'])

    def test_overall_risk_medium(self):
        result = self.scorer.compute_individual_risks(
            {}, fci_score=0, snc_score=0, flp_percentage=0,
            hrsp_percentage=0, ftd_days=0,
            base_infection_probability=40.0)
        self.assertEqual(result['overall_risk'], '中风险')

    def test_overall_risk_low(self):
        result = self.scorer.compute_individual_risks(
            {}, fci_score=0, snc_score=0, flp_percentage=0,
            hrsp_percentage=0, ftd_days=0,
            base_infection_probability=5.0)
        self.assertEqual(result['overall_risk'], '低风险')

    def test_all_dimensions_present(self):
        result = self.scorer.compute_individual_risks(
            {}, fci_score=0, snc_score=0, flp_percentage=0,
            hrsp_percentage=0, ftd_days=0,
            base_infection_probability=0.0)
        for dim in ('FCI', 'SNC', 'FLP', 'HRSP', 'FTD'):
            self.assertIn(dim, result['individual_risks'])
            item = result['individual_risks'][dim]
            self.assertIn('risk', item)
            self.assertIn('suggestion', item)

    def test_fci_high_threshold(self):
        result = self.scorer.compute_individual_risks(
            {}, fci_score=9.0, snc_score=0, flp_percentage=0,
            hrsp_percentage=0, ftd_days=0,
            base_infection_probability=0.0)
        self.assertEqual(result['individual_risks']['FCI']['risk'], '高风险')

    def test_ftd_medium_threshold(self):
        result = self.scorer.compute_individual_risks(
            {}, fci_score=0, snc_score=0, flp_percentage=0,
            hrsp_percentage=0, ftd_days=20,
            base_infection_probability=0.0)
        self.assertEqual(result['individual_risks']['FTD']['risk'], '中风险')


class TestPatientScorerFciSnc(unittest.TestCase):
    """FCI / SNC 综合得分测试。"""

    def setUp(self):
        from tb_risk.core.patient_scorer import PatientScorer
        self.scorer = PatientScorer()

    def test_calculate_fci_score_empty(self):
        from tb_risk.constants import FCI_DEFAULT_NO_FAMILY
        self.assertEqual(
            self.scorer.calculate_fci_score([], family_living_conditions=3),
            FCI_DEFAULT_NO_FAMILY)

    def test_calculate_fci_score_bounds(self):
        members = [_make_family_member() for _ in range(3)]
        score = self.scorer.calculate_fci_score(
            members, family_living_conditions=3)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)

    def test_calculate_fci_score_increases_with_exposure(self):
        low = self.scorer.calculate_fci_score(
            [_make_family_member(cumulative_exposure=5)], family_living_conditions=3)
        high = self.scorer.calculate_fci_score(
            [_make_family_member(cumulative_exposure=200)], family_living_conditions=3)
        self.assertGreater(high, low)

    def test_calculate_snc_score_empty(self):
        self.assertEqual(self.scorer.calculate_snc_score([]), 0.0)

    def test_calculate_snc_score_bounds(self):
        contacts = [_make_social_contact() for _ in range(4)]
        score = self.scorer.calculate_snc_score(contacts)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)

    def test_calculate_snc_score_high_risk_bonus(self):
        normal = self.scorer.calculate_snc_score(
            [_make_social_contact(is_high_risk=0)])
        high = self.scorer.calculate_snc_score(
            [_make_social_contact(is_high_risk=1)])
        self.assertGreater(high, normal)

    def test_calculate_patient_risk_multiplier_none(self):
        self.assertEqual(
            self.scorer.calculate_patient_risk_multiplier(0, 0, False), 1.0)

    def test_calculate_patient_risk_multiplier_capped(self):
        from tb_risk.constants import PATIENT_RISK_MULTIPLIER_CAP
        m = self.scorer.calculate_patient_risk_multiplier(1, 1, True)
        self.assertLessEqual(m, PATIENT_RISK_MULTIPLIER_CAP)
        self.assertGreater(m, 1.0)


# ══════════════════════════════════════════════════════════════
# ContactRiskCalculator
# ══════════════════════════════════════════════════════════════

class TestContactRiskCalculatorBCG(unittest.TestCase):
    """BCG 保护效力测试。"""

    def setUp(self):
        from tb_risk.core.contact_risk_calculator import ContactRiskCalculator
        from tb_risk.core.seir_integration import SEIRIntegration
        # 计算单个接触者风险依赖 SEIR 协作器，注入默认实例
        self.calc = ContactRiskCalculator(seir_integration=SEIRIntegration())

    def test_not_vaccinated_zero(self):
        protection, note = self.calc.calculate_bcg_protection(
            age=10, bcg_vaccinated=False)
        self.assertEqual(protection, 0.0)
        self.assertEqual(note, "")

    def test_child_protection_in_range(self):
        protection, note = self.calc.calculate_bcg_protection(age=5, bcg_vaccinated=True)
        self.assertGreater(protection, 0.0)
        self.assertLessEqual(protection, 1.0)
        self.assertIn('儿童期', note)

    def test_adult_protection_lower(self):
        child_prot, _ = self.calc.calculate_bcg_protection(age=5, bcg_vaccinated=True)
        adult_prot, note = self.calc.calculate_bcg_protection(age=40, bcg_vaccinated=True)
        self.assertLess(adult_prot, child_prot)
        self.assertIn('成人期', note)

    def test_protection_decays_over_time(self):
        young = self.calc.calculate_bcg_protection(
            age=40, bcg_vaccinated=True, years_since_vaccination=5)[0]
        old = self.calc.calculate_bcg_protection(
            age=40, bcg_vaccinated=True, years_since_vaccination=25)[0]
        self.assertLess(old, young)

    def test_explicit_years_since_vaccination(self):
        protection, note = self.calc.calculate_bcg_protection(
            age=30, bcg_vaccinated=True, years_since_vaccination=15)
        self.assertIn('显著衰减', note)


class TestContactRiskCalculatorSingleRisk(unittest.TestCase):
    """单个接触者风险计算测试。"""

    def setUp(self):
        from tb_risk.core.contact_risk_calculator import ContactRiskCalculator
        from tb_risk.core.seir_integration import SEIRIntegration
        # 计算单个接触者风险依赖 SEIR 协作器，注入默认实例
        self.calc = ContactRiskCalculator(seir_integration=SEIRIntegration())

    def test_returns_all_keys(self):
        contact = _make_family_member()
        result = self.calc.compute_single_contact_risk(contact, patient_treatment_days=0)
        for key in ('risk_score', 'infection_probability', 'rule_infection_probability',
                    'seir_infection_probability', 'disease_probability'):
            self.assertIn(key, result)

    def test_probabilities_in_range(self):
        contact = _make_social_contact()
        result = self.calc.compute_single_contact_risk(contact, patient_treatment_days=0)
        for key in ('infection_probability', 'disease_probability',
                    'rule_infection_probability'):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 100.0)

    def test_mutates_contact(self):
        contact = _make_family_member()
        self.calc.compute_single_contact_risk(contact, patient_treatment_days=0)
        self.assertIn('bcg_note', contact)
        self.assertIn('risk_score', contact)

    def test_symptoms_increase_risk_score(self):
        no_sym = self.calc.compute_single_contact_risk(
            _make_family_member(has_symptoms=0), patient_treatment_days=0)
        sym = self.calc.compute_single_contact_risk(
            _make_family_member(has_symptoms=1), patient_treatment_days=0)
        self.assertGreater(sym['risk_score'], no_sym['risk_score'])

    def test_invalid_age_does_not_crash(self):
        contact = _make_family_member(age='abc')
        result = self.calc.compute_single_contact_risk(contact, patient_treatment_days=0)
        self.assertIsNotNone(result['risk_score'])


class TestContactRiskCalculatorGenerate(unittest.TestCase):
    """潜在患者列表生成测试。"""

    def setUp(self):
        from tb_risk.core.contact_risk_calculator import ContactRiskCalculator
        from tb_risk.core.seir_integration import SEIRIntegration
        # 计算单个接触者风险依赖 SEIR 协作器，注入默认实例
        self.calc = ContactRiskCalculator(seir_integration=SEIRIntegration())

    def test_generate_family_and_social(self):
        result = self.calc.generate_potential_patients(
            {}, [_make_family_member()], [_make_social_contact()],
            patient_treatment_days=0)
        self.assertEqual(len(result['family']), 1)
        self.assertEqual(len(result['social']), 1)

    def test_generated_entry_has_required_fields(self):
        result = self.calc.generate_potential_patients(
            {}, [_make_family_member()], [_make_social_contact()],
            patient_treatment_days=0)
        for pp in result['family'] + result['social']:
            for key in ('record_id', 'name', 'infection_probability',
                        'disease_probability', 'priority', 'recommendation'):
                self.assertIn(key, pp)

    def test_record_id_synthesized_from_index(self):
        member = _make_family_member()
        member.pop('_id', None)
        member.pop('record_id', None)
        result = self.calc.generate_potential_patients(
            {}, [member], [], patient_treatment_days=0)
        self.assertTrue(result['family'][0]['record_id'].startswith('family_'))

    def test_default_family_fallback_when_high_fci(self):
        individual_risks = {
            'individual_risks': {'FCI': {'risk': '高风险'}},
        }
        result = self.calc.generate_potential_patients(
            {}, [], [], patient_treatment_days=0,
            individual_risks_result=individual_risks)
        self.assertEqual(len(result['family']), 1)
        self.assertEqual(result['family'][0]['record_id'], 'default_family')

    def test_no_default_family_when_low_fci(self):
        individual_risks = {
            'individual_risks': {'FCI': {'risk': '低风险'}},
        }
        result = self.calc.generate_potential_patients(
            {}, [], [], patient_treatment_days=0,
            individual_risks_result=individual_risks)
        self.assertEqual(len(result['family']), 0)

    def test_default_social_medium_fallback(self):
        individual_risks = {
            'individual_risks': {
                'FCI': {'risk': '低风险'},
                'SNC': {'risk': '中风险'},
                'HRSP': {'risk': '低风险'},
            },
        }
        result = self.calc.generate_potential_patients(
            {}, [], [], patient_treatment_days=0,
            individual_risks_result=individual_risks)
        self.assertEqual(len(result['social']), 1)
        self.assertEqual(result['social'][0]['record_id'], 'default_social_medium')

    def test_build_environment_config_defaults(self):
        cfg = self.calc.build_environment_config()
        self.assertIn('pm10', cfg)
        self.assertIn('humidity', cfg)
        self.assertIn('month', cfg)

    def test_enhance_recommendation_without_localizer(self):
        rec = self.calc.enhance_recommendation_karamay(
            _make_social_contact(), '高', '基础建议')
        self.assertEqual(rec, '基础建议')


# ══════════════════════════════════════════════════════════════
# SEIRIntegration & fuse_probabilities
# ══════════════════════════════════════════════════════════════

class TestFuseProbabilities(unittest.TestCase):
    """SEIR/规则概率融合边界测试。"""

    def test_identical_probs(self):
        from tb_risk.core import fuse_probabilities
        fused = fuse_probabilities(0.4, 0.4, 0.5)
        self.assertAlmostEqual(fused, 40.0, places=6)

    def test_weight_zero_uses_rule_only(self):
        from tb_risk.core import fuse_probabilities
        fused = fuse_probabilities(0.9, 0.2, 0.0)
        self.assertAlmostEqual(fused, 20.0, places=6)

    def test_weight_one_uses_seir_only(self):
        from tb_risk.core import fuse_probabilities
        fused = fuse_probabilities(0.9, 0.2, 1.0)
        self.assertAlmostEqual(fused, 90.0, places=6)

    def test_geometric_mean_property(self):
        """w=0.5 时应为几何平均：sqrt(p_seir * p_rule)。"""
        from tb_risk.core import fuse_probabilities
        import math
        fused = fuse_probabilities(0.8, 0.2, 0.5)
        expected = math.sqrt(0.8 * 0.2) * 100.0
        self.assertAlmostEqual(fused, expected, places=6)

    def test_capped_at_100(self):
        from tb_risk.core import fuse_probabilities
        fused = fuse_probabilities(1.0, 1.0, 0.5)
        self.assertLessEqual(fused, 100.0)


class TestSEIRIntegration(unittest.TestCase):
    """SEIR 集成器测试。"""

    def setUp(self):
        from tb_risk.core.seir_integration import SEIRIntegration
        self.integration = SEIRIntegration()

    def test_time_dependent_risk_bounds(self):
        for days in (0, 5, 20, 60, 200, 500):
            risk = self.integration.calculate_time_dependent_risk(
                {'time_span': 4}, patient_treatment_days=days)
            self.assertGreaterEqual(risk, 0.0, f"days={days}")
            self.assertLessEqual(risk, 1.0, f"days={days}")

    def test_time_dependent_risk_error_returns_zero(self):
        risk = self.integration.calculate_time_dependent_risk(
            {'time_span': None}, patient_treatment_days=None)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)

    def test_compute_seir_probability_none_without_uncertainty(self):
        self.assertIsNone(
            self.integration.compute_seir_probability(
                _make_family_member(), patient_treatment_days=0))

    def test_compute_seir_confidence_weight_zero_without_uncertainty(self):
        self.assertEqual(self.integration.compute_seir_confidence_weight(), 0.0)

    def test_compute_seir_probability_with_mock_uncertainty(self):
        from tb_risk.core.seir_integration import SEIRIntegration

        class _MockUncertainty:
            inference_completed = True
            posterior_summary = {'beta': {'mean': 0.3}}
            mcmc_diagnostics = {'r_hat': [1.0], 'ess': [500]}

        integration = SEIRIntegration(seir_param_uncertainty=_MockUncertainty())
        p = integration.compute_seir_probability(
            _make_family_member(), patient_treatment_days=0)
        self.assertIsNotNone(p)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_compute_seir_confidence_weight_with_mock(self):
        from tb_risk.core.seir_integration import SEIRIntegration

        class _MockUncertainty:
            inference_completed = True
            posterior_summary = {'beta': {'mean': 0.3}}
            mcmc_diagnostics = {'r_hat': [1.0], 'ess': [500]}

        integration = SEIRIntegration(seir_param_uncertainty=_MockUncertainty())
        w = integration.compute_seir_confidence_weight()
        self.assertGreaterEqual(w, 0.0)
        self.assertLessEqual(w, 1.0)
        self.assertGreater(w, 0.0)

    def test_run_seir_simulation_no_crash(self):
        result = self.integration.run_seir_simulation(
            {}, [_make_family_member()], [_make_social_contact()])
        # SEIR 可用则返回结构，否则返回 None（均不崩溃）
        if result is not None:
            self.assertIn('seir_trajectories', result)


class TestSEIRSimulationCache(unittest.TestCase):
    """SEIR 模拟结果 LRU 缓存（增量与缓存计算）测试。

    覆盖:
      1. _SEIRSimulationCache 基本 put/get
      2. LRU 淘汰策略
      3. 统计信息 (hits/misses/hit_rate)
      4. _sim_cache_key 确定性 / 参数敏感 / 字典顺序无关
      5. run_seir_simulation 缓存命中复用与 use_cache 绕过
    """

    def setUp(self):
        from tb_risk.core.seir_integration import SEIRIntegration
        self.integration = SEIRIntegration()

    def test_put_get(self):
        """put 后 get 命中, 未命中返回 None"""
        from tb_risk.core.seir_integration import _SEIRSimulationCache
        cache = _SEIRSimulationCache(maxsize=10)
        self.assertIsNone(cache.get('k'))
        cache.put('k', {'value': 1})
        self.assertEqual(cache.get('k'), {'value': 1})

    def test_lru_eviction(self):
        """超出 maxsize 时淘汰最久未访问"""
        from tb_risk.core.seir_integration import _SEIRSimulationCache
        cache = _SEIRSimulationCache(maxsize=2)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.get('a')  # a 最近访问
        cache.put('c', 3)  # 淘汰 b
        self.assertIsNotNone(cache.get('a'))
        self.assertIsNone(cache.get('b'))
        self.assertIsNotNone(cache.get('c'))

    def test_stats(self):
        """hits/misses/hit_rate 统计正确"""
        from tb_risk.core.seir_integration import _SEIRSimulationCache
        cache = _SEIRSimulationCache(maxsize=4)
        cache.put('a', 1)
        cache.get('a')   # hit
        cache.get('b')   # miss
        stats = cache.stats()
        self.assertEqual(stats['size'], 1)
        self.assertEqual(stats['maxsize'], 4)
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertAlmostEqual(stats['hit_rate'], 0.5)

    def test_sim_cache_key_deterministic(self):
        """相同接触网络与参数 → 相同 MD5 键"""
        from tb_risk.core.seir_integration import _sim_cache_key
        patient = {'name': 'p', 'age': 30}
        family = [{'name': 'f', 'age': 20}]
        social = [{'name': 's', 'age': 40}]
        params = {'beta': 0.3, 'sigma': 0.1, 'gamma': 0.05}
        k1 = _sim_cache_key(patient, family, social, params)
        k2 = _sim_cache_key(dict(patient), list(family),
                            list(social), dict(params))
        self.assertEqual(k1, k2)
        self.assertEqual(len(k1), 32)  # MD5 hex

    def test_sim_cache_key_dict_order_insensitive(self):
        """参数字典顺序无关 (dict 按键排序后哈希)"""
        from tb_risk.core.seir_integration import _sim_cache_key
        k1 = _sim_cache_key({}, [], [],
                            {'beta': 0.3, 'gamma': 0.05, 'sigma': 0.1})
        k2 = _sim_cache_key({}, [], [],
                            {'gamma': 0.05, 'beta': 0.3, 'sigma': 0.1})
        self.assertEqual(k1, k2)

    def test_sim_cache_key_changes_with_params(self):
        """参数变化 → 键变化 (不同参数组合不复用结果)"""
        from tb_risk.core.seir_integration import _sim_cache_key
        k1 = _sim_cache_key({}, [], [], {'beta': 0.3})
        k2 = _sim_cache_key({}, [], [], {'beta': 0.4})
        self.assertNotEqual(k1, k2)

    def test_run_seir_simulation_reuses_cached(self):
        """相同接触网络 + 参数再次调用时命中缓存 (返回同一对象)"""
        patient = {'name': 'p'}
        family = [_make_family_member()]
        social = [_make_social_contact()]
        result1 = self.integration.run_seir_simulation(patient, family, social)
        if result1 is None:
            self.skipTest("SEIR 模块不可用")
        stats_before = self.integration.simulation_cache_stats()
        result2 = self.integration.run_seir_simulation(patient, family, social)
        self.assertIs(result2, result1)  # 同一缓存对象, 未重新模拟
        stats_after = self.integration.simulation_cache_stats()
        self.assertGreater(stats_after['hits'], stats_before['hits'])
        self.assertEqual(stats_after['misses'], stats_before['misses'])

    def test_run_seir_simulation_use_cache_false_bypasses(self):
        """use_cache=False 时绕过缓存 (重新计算且不写入)"""
        patient = {'name': 'p'}
        family = [_make_family_member()]
        social = [_make_social_contact()]
        result1 = self.integration.run_seir_simulation(patient, family, social)
        if result1 is None:
            self.skipTest("SEIR 模块不可用")
        stats_before = self.integration.simulation_cache_stats()
        result2 = self.integration.run_seir_simulation(
            patient, family, social, use_cache=False)
        self.assertIsNot(result2, result1)  # 重新计算, 新对象
        stats_after = self.integration.simulation_cache_stats()
        # 绕过时既不命中也不写入 (hits/misses/size 均不变)
        self.assertEqual(stats_after, stats_before)

    def test_simulation_cache_stats_structure(self):
        """simulation_cache_stats 返回完整统计结构"""
        stats = self.integration.simulation_cache_stats()
        for key in ('size', 'maxsize', 'hits', 'misses', 'hit_rate'):
            self.assertIn(key, stats)
        self.assertGreater(stats['maxsize'], 0)


# ══════════════════════════════════════════════════════════════
# EnsembleWeightOptimizer
# ══════════════════════════════════════════════════════════════

class TestEnsembleWeightOptimizer(unittest.TestCase):
    """集成权重优化器测试。"""

    def test_karamay_validator_property_none(self):
        from tb_risk.core.weight_optimizer import EnsembleWeightOptimizer
        opt = EnsembleWeightOptimizer()
        self.assertIsNone(opt.karamay_validator)

    def test_ensure_validation_no_localizer(self):
        """无本土化模块时自动编排应安全跳过。"""
        from tb_risk.core.weight_optimizer import EnsembleWeightOptimizer
        opt = EnsembleWeightOptimizer(karamay_localizer=None)
        opt.ensure_validation()  # 不应抛异常
        self.assertTrue(opt._validation_auto_triggered)

    def test_ensure_validation_existing_validator(self):
        from tb_risk.core.weight_optimizer import EnsembleWeightOptimizer
        ref = type('_Ref', (), {})()
        ref.karamay_validator = object()
        opt = EnsembleWeightOptimizer(karamay_validator_ref=ref)
        opt.ensure_validation()
        self.assertTrue(opt._validation_auto_triggered)

    def test_optimize_weights_no_data_keeps_defaults(self):
        """无可用 AUROC 数据时优化应安全返回（不崩溃）。"""
        from tb_risk.core.weight_optimizer import EnsembleWeightOptimizer
        opt = EnsembleWeightOptimizer()
        opt.optimize_weights()  # 不应抛异常
        self.assertTrue(True)


# ══════════════════════════════════════════════════════════════
# 初始化器
# ══════════════════════════════════════════════════════════════

class TestInitializers(unittest.TestCase):
    """core 初始化器测试。"""

    def test_init_advanced_features_defaults(self):
        from tb_risk.core import init_advanced_features, AdvancedFeaturesInitResult
        result = init_advanced_features()
        self.assertIsInstance(result, AdvancedFeaturesInitResult)
        # 未启用功能时相关字段应为 None / False
        self.assertFalse(result._uncertainty_ready)
        self.assertIsNone(result.hetero_gnn)
        self.assertFalse(result._multimodal_ready)
        self.assertFalse(result._intervention_optimizer_ready)

    def test_init_seir_models_returns_result(self):
        from tb_risk.core import init_seir_models, SEIRInitResult
        result = init_seir_models(seir_population=10000, random_state=42)
        self.assertIsInstance(result, SEIRInitResult)
        # SEIR 可用时 stochastic_seir 非 None；不可用时为 None（均不崩溃）
        self.assertTrue(hasattr(result, 'stochastic_seir'))
        self.assertTrue(hasattr(result, 'seir_inference'))
        # 统一框架别名：seir_param_uncertainty 与 seir_inference 应指向同一实例或同为 None
        self.assertTrue(
            result.seir_param_uncertainty is result.seir_inference
            or (result.seir_param_uncertainty is None
                and result.seir_inference is None))


# ══════════════════════════════════════════════════════════════
# run_ml_prediction_loop
# ══════════════════════════════════════════════════════════════

class _MockPredictor:
    """轻量 ML 预测器桩，避免真实训练（保持测试快速）。"""

    def __init__(self, trained=True):
        self.is_trained = trained
        self.gnn_is_trained = False
        self.model_performance = {'rf': {'AUROC': 0.8}}
        self.training_sample_count = 100

    def predict_risk(self, contact, contact_type):
        return {'ensemble': {'risk_probability': 60.0, 'risk_class': 1}}

    def predict_gnn_risk(self, assessment, contact, contact_type):
        return None

    def compute_gnn_explanation(self, assessment, contact, contact_type):
        return None

    def compute_shap_values(self, contacts, types):
        return {
            'feature_importance': [], 'feature_names': [],
            'feature_descriptions': {}, 'shap_values': [],
            'feature_matrix': [],
        }

    def compute_contact_attributions(self, contacts, types):
        return [None] * len(contacts)


class _StubIntegrator:
    def integrate_predictions(self, ml_predictor, assessment, contact, contact_type):
        return {'ensemble': {'risk_probability': 55.0, 'risk_class': 1}}


class TestMLRunner(unittest.TestCase):
    """run_ml_prediction_loop 测试。"""

    def _run(self, predictor, results):
        from tb_risk.core.ml_runner import run_ml_prediction_loop
        return run_ml_prediction_loop(None, predictor, _StubIntegrator(),
                                      results, threading.Lock())

    def test_none_predictor_returns_none(self):
        self.assertIsNone(self._run(None, {}))

    def test_untrained_predictor_returns_none(self):
        self.assertIsNone(self._run(_MockPredictor(trained=False), {}))

    def test_returns_structure(self):
        results = {
            'potential_patients': {
                'family': [_make_family_member()],
                'social': [_make_social_contact()],
            }
        }
        out = self._run(_MockPredictor(), results)
        self.assertIsNotNone(out)
        self.assertIn('family', out)
        self.assertIn('social', out)
        self.assertIn('model_performance', out)
        self.assertIn('gnn', out)
        self.assertIn('ensemble', out)
        self.assertEqual(len(out['family']), 1)
        self.assertEqual(len(out['social']), 1)

    def test_returns_empty_for_no_potential_patients(self):
        out = self._run(_MockPredictor(), {})
        self.assertIsNotNone(out)
        self.assertEqual(out['family'], [])
        self.assertEqual(out['social'], [])


if __name__ == '__main__':
    unittest.main()
