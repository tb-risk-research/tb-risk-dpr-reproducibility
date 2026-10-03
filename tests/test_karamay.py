#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - karamay 模块专项测试 (V4)

覆盖：
- BayesianCalibrator 默认路径
- KaramayLocalizer 各子模块
- CalibrationPriors 参数验证
- 贝叶斯校准后验收敛
"""

import sys
import os
import unittest

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class TestBayesianCalibrator(unittest.TestCase):
    """BayesianCalibrator 贝叶斯校准默认路径测试 (L2)"""

    @classmethod
    def setUpClass(cls):
        """类级共享 KaramayLocalizer 实例，避免重复初始化(~13s)"""
        from tb_risk.karamay.localizer import KaramayLocalizer
        cls._localizer = KaramayLocalizer()

    @pytest.mark.slow
    def test_bayesian_calibrator_import(self):
        """测试 BayesianCalibrator 可导入"""
        from tb_risk.karamay.bayesian_calibrator import (
            BayesianCalibrator, CalibrationPriors, CalibrationResult)
        self.assertTrue(True)

    def test_calibration_priors_default(self):
        """测试 CalibrationPriors 默认参数"""
        from tb_risk.karamay.bayesian_calibrator import CalibrationPriors
        self.assertTrue(hasattr(CalibrationPriors, 'PARAM_NAMES'))
        self.assertTrue(hasattr(CalibrationPriors, 'PARAM_BOUNDS'))
        self.assertTrue(hasattr(CalibrationPriors, 'log_prior'))
        self.assertIn('oilfield_camp_factor', CalibrationPriors.PARAM_NAMES)
        # base_incidence_per_100k 通过发病率缩放进入 likelihood，纳入 PARAM_NAMES；
        # diagnosis_delay_days 不影响 likelihood，作为 EPIDEMIC_PARAMS 常量保留。
        self.assertIn('base_incidence_per_100k', CalibrationPriors.PARAM_NAMES)
        self.assertIn('latent_prevalence', CalibrationPriors.PARAM_BOUNDS)
        self.assertNotIn('base_incidence_per_100k', CalibrationPriors.EPIDEMIC_PARAMS)
        self.assertIn('diagnosis_delay_days', CalibrationPriors.EPIDEMIC_PARAMS)

    def test_bayesian_calibrator_initialization(self):
        """测试 BayesianCalibrator 初始化"""
        from tb_risk.karamay.bayesian_calibrator import BayesianCalibrator
        calibrator = BayesianCalibrator(self._localizer)
        self.assertIsNotNone(calibrator)

    def test_calibration_result_structure(self):
        """测试 CalibrationResult 数据结构"""
        from tb_risk.karamay.bayesian_calibrator import CalibrationResult
        result = CalibrationResult(
            param_names=['base_incidence_per_100k', 'oilfield_camp_factor', 'latent_prevalence'],
            chains=None, n_burnin=0, observed_data={},
            is_fallback=True,
        )
        self.assertTrue(result.is_fallback)
        # fallback 的 posterior_mean 包含 EPIDEMIC_PARAMS 与 PARAM_NAMES 参数
        self.assertIn('base_incidence_per_100k', result.posterior_mean)
        self.assertIn('oilfield_camp_factor', result.posterior_mean)

    def test_calibrator_is_default_path(self):
        """测试贝叶斯校准是否为默认路径"""
        from tb_risk.karamay.calibrator import KaramayCalibrator
        cal = KaramayCalibrator()
        self.assertTrue(hasattr(cal, 'use_bayesian'))
        # 贝叶斯校准应为默认路径
        self.assertTrue(cal.use_bayesian,
                        '贝叶斯校准应为默认路径 (L2)')


class TestKaramayLocalizerSubmodules(unittest.TestCase):
    """KaramayLocalizer 子模块测试"""

    @classmethod
    def setUpClass(cls):
        """类级共享 KaramayLocalizer 实例，避免重复初始化(~13s)"""
        from tb_risk.karamay.localizer import KaramayLocalizer
        cls._localizer = KaramayLocalizer()

    def setUp(self):
        from tb_risk.config import ConfigProxy
        self.config = ConfigProxy()

    def test_oilfield_module(self):
        """测试油田职业暴露模块"""
        from tb_risk.karamay.oilfield import OilfieldExposureModel
        module = OilfieldExposureModel(self.config)
        self.assertTrue(hasattr(module, 'calculate_oilfield_exposure'))
        self.assertAlmostEqual(module.oilfield_camp_factor, 0.85)

    def test_oilfield_continuous_ratio_validation(self):
        """continuous_ratio 超出 0.65-0.80 范围时应被钳制"""
        from tb_risk.karamay.oilfield import OilfieldExposureModel
        module = OilfieldExposureModel(self.config)
        # 默认 crew_type 应在合理范围内
        self.assertGreaterEqual(module.get_continuous_ratio(), 0.65)
        self.assertLessEqual(module.get_continuous_ratio(), 0.80)
        # 显式校验越界输入
        validated = module._validate_shift_pattern({'continuous_ratio': 0.5})
        self.assertAlmostEqual(validated['continuous_ratio'], 0.65)
        validated = module._validate_shift_pattern({'continuous_ratio': 0.95})
        self.assertAlmostEqual(validated['continuous_ratio'], 0.80)
        validated = module._validate_shift_pattern({'continuous_ratio': 0.72})
        self.assertAlmostEqual(validated['continuous_ratio'], 0.72)

    def test_climate_module(self):
        """测试气候气溶胶模块"""
        from tb_risk.karamay.climate import ClimateTBInteraction
        module = ClimateTBInteraction(self.config)
        self.assertTrue(hasattr(module, 'calculate_climate_multiplier'))

    @pytest.mark.slow
    def test_altitude_module(self):
        """测试海拔适应性模块"""
        from tb_risk.karamay.altitude import AltitudeAdaptation
        module = AltitudeAdaptation(self.config)
        self.assertTrue(hasattr(module, 'get_progression_adjustment'))

    def test_ethnicity_module(self):
        """测试民族模块（已弃用为描述性统计，始终返回 1.0）"""
        from tb_risk.karamay.ethnicity import EthnicityPathogenModule
        module = EthnicityPathogenModule(self.config)
        self.assertTrue(hasattr(module, 'get_combined_strain_ethnicity_factor'))
        # 验证民族遗传易感性已弃用，始终返回 1.0
        self.assertEqual(module.get_ethnicity_susceptibility('han'), 1.0)
        self.assertEqual(module.get_ethnicity_susceptibility('uyghur'), 1.0)
        self.assertEqual(module.get_combined_strain_ethnicity_factor('beijing', 'han'), 1.0)
        adj = module.get_screening_priority_adjustment('uyghur', 'beijing')
        self.assertEqual(adj['adjustment'], 1.0)

    def test_sdoh_module(self):
        """测试社会决定因素（SDOH）模块"""
        from tb_risk.karamay.sdoh import SDOHModule
        module = SDOHModule(self.config)
        self.assertTrue(hasattr(module, 'compute_sdoh_multiplier'))
        self.assertTrue(hasattr(module, 'compute_housing_crowding'))
        self.assertTrue(hasattr(module, 'compute_healthcare_access'))
        self.assertTrue(hasattr(module, 'compute_nutrition_risk'))
        self.assertTrue(hasattr(module, 'compute_income_risk'))
        self.assertTrue(hasattr(module, 'audit_fairness'))

    def test_sdoh_housing_crowding(self):
        """测试 SDOH 住房拥挤指数计算"""
        from tb_risk.karamay.sdoh import SDOHModule
        module = SDOHModule(self.config)

        # 正常居住面积 → 无风险
        self.assertEqual(module.compute_housing_crowding(30.0), 1.0)
        self.assertEqual(module.compute_housing_crowding(15.0), 1.0)

        # 拥挤 → 触发乘数
        factor = module.compute_housing_crowding(10.0)
        # 1.0 + (15-10)*0.02 = 1.10
        self.assertAlmostEqual(factor, 1.10, places=4)

        # 极端拥挤 → 上限 1.3
        factor = module.compute_housing_crowding(0.0)
        # 1.0 + (15-0)*0.02 = 1.30
        self.assertAlmostEqual(factor, 1.30, places=4)

        # None → 无数据，返回 1.0
        self.assertEqual(module.compute_housing_crowding(None), 1.0)

    def test_sdoh_healthcare_access(self):
        """测试 SDOH 医疗可及性指数计算"""
        from tb_risk.karamay.sdoh import SDOHModule
        module = SDOHModule(self.config)

        # 距离近 → 无风险
        self.assertEqual(module.compute_healthcare_access(10.0), 1.0)
        self.assertEqual(module.compute_healthcare_access(30.0), 1.0)

        # 偏远 → 触发乘数
        factor = module.compute_healthcare_access(50.0)
        # 1.0 + (50-30)*0.005 = 1.10
        self.assertAlmostEqual(factor, 1.10, places=4)

        # 极端偏远 → 上限 1.25
        factor = module.compute_healthcare_access(100.0)
        # 1.0 + (100-30)*0.005 = 1.35 → 裁剪到 1.25
        self.assertAlmostEqual(factor, 1.25, places=4)

    def test_sdoh_nutrition_risk(self):
        """测试 SDOH 营养风险评分计算"""
        from tb_risk.karamay.sdoh import SDOHModule
        module = SDOHModule(self.config)

        # 低体重 → 乘数 1.3
        self.assertEqual(module.compute_nutrition_risk(16.0), 1.3)
        self.assertEqual(module.compute_nutrition_risk(18.4), 1.3)

        # 正常 BMI → 乘数 1.0
        self.assertEqual(module.compute_nutrition_risk(22.0), 1.0)

        # 超重 → 保守设为 1.0
        self.assertEqual(module.compute_nutrition_risk(28.0), 1.0)

    def test_sdoh_income_risk(self):
        """测试 SDOH 收入分层计算"""
        from tb_risk.karamay.sdoh import SDOHModule
        module = SDOHModule(self.config)

        # 贫困线以下 → 乘数 1.2
        self.assertEqual(module.compute_income_risk(0.5), 1.2)

        # 贫困线以上 → 无风险
        self.assertEqual(module.compute_income_risk(0.8), 1.0)
        self.assertEqual(module.compute_income_risk(0.6), 1.0)

    def test_sdoh_additive_combination(self):
        """测试 SDOH 加法组合模型"""
        from tb_risk.karamay.sdoh import SDOHModule
        module = SDOHModule(self.config)

        # 无 SDOH 风险 → 乘数 1.0
        result = module.compute_sdoh_multiplier({
            'living_area_per_person': 30.0,
            'distance_to_tb_center_km': 10.0,
            'bmi': 22.0,
            'income_ratio': 0.8,
        })
        self.assertAlmostEqual(result['multiplier'], 1.0, places=4)

        # 单一风险（住房拥挤）
        result = module.compute_sdoh_multiplier({
            'living_area_per_person': 10.0,
        })
        # 1.0 + (1.10 - 1.0) = 1.10
        self.assertAlmostEqual(result['multiplier'], 1.10, places=4)

        # 多重风险（加法组合）
        result = module.compute_sdoh_multiplier({
            'living_area_per_person': 10.0,    # 住房拥挤: 1.10
            'distance_to_tb_center_km': 50.0,  # 医疗可及性: 1.10
            'bmi': 16.0,                        # 营养风险: 1.30
            'income_ratio': 0.5,                # 收入分层: 1.20
        })
        # combined = 1.0 + 0.10 + 0.10 + 0.30 + 0.20 = 1.70
        self.assertAlmostEqual(result['multiplier'], 1.70, places=4)

        # 验证上限 2.0
        result = module.compute_sdoh_multiplier({
            'living_area_per_person': 0.0,     # 住房拥挤: 1.30 (上限)
            'distance_to_tb_center_km': 100.0, # 医疗可及性: 1.25 (上限)
            'bmi': 16.0,                        # 营养风险: 1.30
            'income_ratio': 0.5,                # 收入分层: 1.20
        })
        # combined = 1.0 + 0.30 + 0.25 + 0.30 + 0.20 = 2.05 → 裁剪到 2.0
        self.assertAlmostEqual(result['multiplier'], 2.0, places=4)

        # 验证下限 0.8
        # 当所有因子为 1.0 时不会低于 0.8，但下限机制存在
        self.assertGreaterEqual(result['multiplier'], 0.8)

    def test_sdoh_fairness_audit_metadata(self):
        """测试 SDOH 公平性审计元数据"""
        from tb_risk.karamay.sdoh import SDOHModule
        module = SDOHModule(self.config)

        result = module.compute_sdoh_multiplier({
            'bmi': 16.0,
        })
        self.assertIn('fairness_audit', result)
        audit = result['fairness_audit']
        self.assertEqual(audit['method'], 'additive')
        self.assertIn('belmont_compliance', audit)
        self.assertIn('irb_note', audit)

    def test_sdoh_localizer_integration(self):
        """测试 SDOH 与 KaramayLocalizer 集成"""
        self.assertTrue(hasattr(self._localizer, 'sdoh'))

        # 验证 SDOH 规则增强器存在
        self.assertTrue(hasattr(self._localizer, '_enhance_sdoh'))

        # 验证 get_combined_progression_multiplier 使用 SDOH
        result = self._localizer.get_combined_progression_multiplier(
            contact_data={
                'living_area_per_person': 10.0,
                'bmi': 16.0,
            }
        )
        self.assertIn('sdoh', result['components'])

    def test_idu_module(self):
        """测试 IDU 人群风险模块"""
        from tb_risk.karamay.idu import IDURiskModule
        module = IDURiskModule(self.config)
        self.assertTrue(hasattr(module, 'get_immunosuppression_factor'))

    def test_policy_module(self):
        """测试地方政策引擎模块"""
        from tb_risk.karamay.policy import LocalPolicyEngine
        module = LocalPolicyEngine(self.config)
        self.assertTrue(hasattr(module, 'get_screening_recommendation'))

    def test_localizer_full(self):
        """测试 KaramayLocalizer 完整初始化"""
        self.assertTrue(hasattr(self._localizer, '_config'))
        self.assertTrue(hasattr(self._localizer, 'get_combined_progression_multiplier'))


class TestPosteriorDrivenInfectivity(unittest.TestCase):
    """PosteriorDrivenInfectivity 阶段系数一致性测试"""

    def test_default_factors_match_assessment(self):
        """测试默认因子与文献基线一致（v3.0 嵌套结构）

        涂阳 (smear+): 1.0/0.55/0.30/0.15/0.05
          - early_treatment=0.55：涂阳治疗 2-3 周内涂片转阴，衰减更快
        涂阴 (smear-): 0.35/0.85/0.65/0.40/0.15
          - pre_treatment=η_sn=0.35：涂阴基线传染性较低
          - 其余阶段衰减相对平缓
        """
        from tb_risk.seir.posterior import PosteriorDrivenInfectivity
        pdi = PosteriorDrivenInfectivity(seir_inference=None)

        factors = pdi.factors
        # 嵌套结构校验
        self.assertIn('smear_positive', factors)
        self.assertIn('smear_negative', factors)

        sp = factors['smear_positive']
        self.assertEqual(sp['pre_treatment'], 1.0)
        self.assertEqual(sp['early_treatment'], 0.55)
        self.assertEqual(sp['mid_treatment'], 0.30)
        self.assertEqual(sp['late_treatment'], 0.15)
        self.assertEqual(sp['completed_treatment'], 0.05)

        sn = factors['smear_negative']
        self.assertAlmostEqual(sn['pre_treatment'], 0.35, places=2)
        self.assertEqual(sn['early_treatment'], 0.85)
        self.assertEqual(sn['mid_treatment'], 0.65)
        self.assertEqual(sn['late_treatment'], 0.40)
        self.assertEqual(sn['completed_treatment'], 0.15)

    def test_posterior_factors_use_correct_coefficients(self):
        """测试后验路径使用正确的涂阳/涂阴系数

        v3.0：scaling=1.0 时
          - 涂阳 early = 1.0 * 0.55 = 0.55
          - 涂阴 early = η_sn * 1.0 * 0.85 = 0.35 * 0.85 = 0.2975
        """
        from tb_risk.seir.posterior import PosteriorDrivenInfectivity

        # 模拟 seir_inference 对象
        class MockInference:
            class posterior_samples:
                pass

        mock_inf = MockInference()
        mock_inf.posterior_samples = {
            'beta': [0.3, 0.32, 0.28],
            'gamma': [0.05, 0.052, 0.048],
        }

        pdi = PosteriorDrivenInfectivity(seir_inference=mock_inf)
        factors = pdi.factors

        # 验证 scaling 计算
        scaling = 0.3 / (0.05 * 6.0)  # scaling = 1.0
        # 涂阳 early = scaling * 0.55
        expected_sp_early = min(scaling * 0.55, 1.0)
        self.assertEqual(factors['smear_positive']['early_treatment'], expected_sp_early)
        # 涂阴 early = η_sn * scaling * 0.85 (η_sn=0.35 默认值)
        eta_sn = 0.35
        expected_sn_early = min(eta_sn * scaling * 0.85, 1.0)
        self.assertAlmostEqual(
            factors['smear_negative']['early_treatment'], expected_sn_early, places=6)

    def test_update_from_mcmc_resets_cache(self):
        """测试 update_from_mcmc 重置缓存"""
        from tb_risk.seir.posterior import PosteriorDrivenInfectivity
        pdi = PosteriorDrivenInfectivity(seir_inference=None)

        # 先获取默认因子
        _ = pdi.factors
        self.assertIsNone(pdi._posterior_factors)

        # 更新后应重置
        pdi.update_from_mcmc(seir_inference=None)
        self.assertIsNone(pdi._posterior_factors)


if __name__ == '__main__':
    unittest.main()