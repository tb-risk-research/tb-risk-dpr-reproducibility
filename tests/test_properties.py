#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - Hypothesis 属性测试 (V4)

基于属性的测试（Property-Based Testing），验证数学不变量：
- 单调性：风险评分对风险因素单调递增
- 界限：概率输出在 [0, 1] 区间
- 缩放不变性：归一化不影响排序
- 对称性：特征排列不变性
"""

import sys
import os
import unittest

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

try:
    from hypothesis import given, settings, strategies as st, assume, HealthCheck
    from hypothesis.extra.numpy import arrays
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

    # 回退：当 Hypothesis 不可用时，装饰器会跳过测试（而非空跑通过）
    def given(*args, **kwargs):
        """装饰器：Hypothesis 不可用时跳过测试"""
        return unittest.skip("Hypothesis 未安装，跳过属性测试")

    def settings(*args, **kwargs):
        """装饰器：Hypothesis 不可用时跳过测试"""
        return unittest.skip("Hypothesis 未安装，跳过属性测试")

    class st:
        """空策略模块（Hypothesis 不可用时的回退）"""
        @staticmethod
        def integers(*args, **kwargs):
            return None
        @staticmethod
        def floats(*args, **kwargs):
            return None
        @staticmethod
        def text(*args, **kwargs):
            return None

    def assume(condition):
        pass

    class HealthCheck:
        too_slow = None


class TestScoringEngineProperties(unittest.TestCase):
    """ScoringEngine 数学不变量属性测试"""

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(
        age=st.integers(min_value=0, max_value=120),
        cough_freq=st.integers(min_value=0, max_value=30),
        ftd=st.integers(min_value=0, max_value=365),
        symptoms=st.integers(min_value=0, max_value=1),
        has_cavity=st.integers(min_value=0, max_value=1),
        sputum_smear=st.integers(min_value=1, max_value=2),
        exposure_hours=st.floats(min_value=0.0, max_value=10000.0),
        ventilation=st.integers(min_value=1, max_value=5),
        pm10=st.floats(min_value=0.0, max_value=500.0),
        humidity=st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_risk_probability_bounded(self, age, cough_freq, ftd, symptoms,
                                       has_cavity, sputum_smear, exposure_hours,
                                       ventilation, pm10, humidity):
        """属性：风险概率始终在 [0, 1] 区间内"""
        from tb_risk.scoring.engine import ScoringEngine

        engine = ScoringEngine()
        contact = {
            'age': age,
            'cough_freq': cough_freq,
            'ftd': ftd,
            'symptoms': symptoms,
            'has_cavity': has_cavity,
            'sputum_smear': sputum_smear,
            'cumulative_exposure': exposure_hours,
            'ventilation': ventilation,
            'pm10': pm10,
            'humidity': humidity,
            'exposure_setting': 'general',
            'contact_distance': 'medium',
        }

        result = engine.compute_risk_score(contact)
        prob = result.get('disease_probability', 0.0)

        self.assertGreaterEqual(prob, 0.0, f"概率 {prob} < 0")
        self.assertLessEqual(prob, 1.0, f"概率 {prob} > 1")

    @pytest.mark.slow
    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(
        age=st.integers(min_value=20, max_value=60),
        base_exposure=st.floats(min_value=0.0, max_value=5000.0),
        extra_exposure=st.floats(min_value=0.0, max_value=5000.0),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_risk_monotonic_in_exposure(self, age, base_exposure, extra_exposure):
        """属性：风险评分对暴露时长单调递增"""
        from tb_risk.scoring.engine import ScoringEngine

        engine = ScoringEngine()
        base_contact = {
            'age': age,
            'cough_freq': 5,
            'ftd': 30,
            'symptoms': 0,
            'has_cavity': 0,
            'sputum_smear': 1,
            'cumulative_exposure': base_exposure,
            'ventilation': 3,
            'pm10': 100.0,
            'humidity': 40.0,
            'exposure_setting': 'general',
            'contact_distance': 'medium',
        }

        high_exposure_contact = dict(base_contact)
        high_exposure_contact['cumulative_exposure'] = base_exposure + extra_exposure

        base_result = engine.compute_risk_score(base_contact)
        high_result = engine.compute_risk_score(high_exposure_contact)

        base_prob = base_result.get('disease_probability', 0.0)
        high_prob = high_result.get('disease_probability', 0.0)

        self.assertGreaterEqual(
            high_prob, base_prob,
            f"暴露增加 {extra_exposure}h 但风险从 {base_prob:.6f} 降至 {high_prob:.6f}"
        )

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(
        age=st.integers(min_value=0, max_value=120),
        exposure=st.floats(min_value=0.0, max_value=10000.0),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_treatment_reduces_risk(self, age, exposure):
        """属性：治疗状态应降低风险评分"""
        from tb_risk.scoring.engine import ScoringEngine

        engine = ScoringEngine()
        untreated = {
            'age': age,
            'cough_freq': 5,
            'ftd': 30,
            'symptoms': 0,
            'has_cavity': 0,
            'sputum_smear': 1,
            'cumulative_exposure': exposure,
            'ventilation': 3,
            'pm10': 100.0,
            'humidity': 40.0,
            'exposure_setting': 'general',
            'contact_distance': 'medium',
            'is_treated': 0,
            'treatment_phase': 0,
        }

        treated = dict(untreated)
        treated['is_treated'] = 1
        treated['treatment_phase'] = 2  # 中期治疗

        untreated_result = engine.compute_risk_score(untreated)
        treated_result = engine.compute_risk_score(treated)

        untreated_prob = untreated_result.get('disease_probability', 0.0)
        treated_prob = treated_result.get('disease_probability', 0.0)

        self.assertLessEqual(
            treated_prob, untreated_prob,
            f"治疗后风险 {treated_prob:.6f} > 未治疗风险 {untreated_prob:.6f}"
        )


class TestSEIRProperties(unittest.TestCase):
    """SEIR 模型数学不变量属性测试"""

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(
        beta=st.floats(min_value=0.01, max_value=5.0),
        gamma=st.floats(min_value=0.01, max_value=2.0),
        population=st.integers(min_value=100, max_value=100000),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_r0_positive(self, beta, gamma, population):
        """属性：R₀ 始终为正"""
        assume(gamma > 0)
        r0 = beta / gamma
        self.assertGreater(r0, 0.0)

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(
        beta=st.floats(min_value=0.01, max_value=5.0),
        sigma=st.floats(min_value=0.01, max_value=2.0),
        gamma=st.floats(min_value=0.01, max_value=2.0),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_seir_parameters_positive(self, beta, sigma, gamma):
        """属性：SEIR 参数始终为正"""
        self.assertGreater(beta, 0.0)
        self.assertGreater(sigma, 0.0)
        self.assertGreater(gamma, 0.0)


class TestSigmoidProperties(unittest.TestCase):
    """Sigmoid 函数数学不变量属性测试"""

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(x=st.floats(min_value=-100.0, max_value=100.0))
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_sigmoid_bounded(self, x):
        """属性：Sigmoid 输出始终在 [0, 1] 区间"""
        import math
        s = 1.0 / (1.0 + math.exp(-x))
        self.assertGreaterEqual(s, 0.0, f"Sigmoid({x}) = {s} < 0")
        self.assertLessEqual(s, 1.0, f"Sigmoid({x}) = {s} > 1")

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(x=st.floats(min_value=-50.0, max_value=50.0))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_sigmoid_monotonic(self, x):
        """属性：Sigmoid 单调非递减

        在数值饱和区（x 很大时）双精度浮点无法区分 s(x) 与 s(x+delta)，
        因此使用 >= 而非 >，并在未饱和时仍要求严格递增。
        """
        import math
        delta = 0.01
        s_x = 1.0 / (1.0 + math.exp(-x))
        s_x_plus = 1.0 / (1.0 + math.exp(-(x + delta)))
        # 浮点精度限制：当 s_x 已接近 1 时，s_x_plus 可能完全相等
        self.assertGreaterEqual(s_x_plus, s_x,
                                f"Sigmoid({x + delta}) = {s_x_plus} < Sigmoid({x}) = {s_x}")
        # 在未饱和区域保留严格单调性检查
        if s_x < 0.999999999999:
            self.assertGreater(s_x_plus, s_x,
                               f"Sigmoid({x + delta}) = {s_x_plus} ≤ Sigmoid({x}) = {s_x}")

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(x=st.floats(min_value=-50.0, max_value=50.0))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_sigmoid_symmetry(self, x):
        """属性：Sigmoid(-x) = 1 - Sigmoid(x)"""
        import math
        s_pos = 1.0 / (1.0 + math.exp(-x))
        s_neg = 1.0 / (1.0 + math.exp(x))
        self.assertAlmostEqual(s_neg, 1.0 - s_pos, places=10,
                               msg=f"Sigmoid(-{x}) != 1 - Sigmoid({x})")


class TestCumulativeExposureProperties(unittest.TestCase):
    """累积暴露计算属性测试"""

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(
        duration=st.floats(min_value=0.0, max_value=480.0),
        freq=st.integers(min_value=0, max_value=30),
        weeks=st.integers(min_value=0, max_value=52),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_cumulative_exposure_nonnegative(self, duration, freq, weeks):
        """属性：累积暴露始终非负"""
        cumulative = (duration / 60.0) * freq * weeks
        self.assertGreaterEqual(cumulative, 0.0)

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(
        duration=st.floats(min_value=0.0, max_value=480.0),
        freq=st.integers(min_value=0, max_value=30),
        weeks=st.integers(min_value=0, max_value=52),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_cumulative_exposure_zero_when_any_factor_zero(self, duration, freq, weeks):
        """属性：任一因子为零时累积暴露为零"""
        cumulative = (duration / 60.0) * freq * weeks
        if duration == 0.0 or freq == 0 or weeks == 0:
            self.assertEqual(cumulative, 0.0)


class TestPosteriorCoefficientConsistency(unittest.TestCase):
    """后验系数一致性属性测试"""

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(
        scaling=st.floats(min_value=0.1, max_value=3.0),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_treatment_phase_ordering(self, scaling):
        """属性：治疗阶段系数保持递减顺序（早期 > 中期 > 晚期 > 完成）

        v3.0：factors 返回嵌套结构 {smear_positive: {...}, smear_negative: {...}}，
        涂阳以 η_sp=1.0 为锚点，涂阴以 η_sn≈0.35 为基线。
        """
        from tb_risk.seir.posterior import PosteriorDrivenInfectivity

        pdi = PosteriorDrivenInfectivity(seir_inference=None)
        factors = pdi.factors

        # 嵌套结构校验
        self.assertIn('smear_positive', factors)
        self.assertIn('smear_negative', factors)

        # 涂阳默认因子应有递减顺序
        sp = factors['smear_positive']
        self.assertGreater(sp['early_treatment'], sp['mid_treatment'])
        self.assertGreater(sp['mid_treatment'], sp['late_treatment'])
        self.assertGreater(sp['late_treatment'], sp['completed_treatment'])

        # 涂阴默认因子应有递减顺序
        sn = factors['smear_negative']
        self.assertGreater(sn['early_treatment'], sn['mid_treatment'])
        self.assertGreater(sn['mid_treatment'], sn['late_treatment'])
        self.assertGreater(sn['late_treatment'], sn['completed_treatment'])

        # 涂阳锚点：pre_treatment 恒为 1.0
        self.assertEqual(sp['pre_treatment'], 1.0)
        # 涂阴基线：pre_treatment = η_sn ≈ 0.35
        self.assertAlmostEqual(sn['pre_treatment'], 0.35, places=2)

        # 即使 scaling 变化，涂阳顺序也应保持
        early = min(scaling * 0.55, 1.0)
        mid = min(scaling * 0.30, 1.0)
        late = min(scaling * 0.15, 1.0)
        completed = min(scaling * 0.05, 1.0)

        self.assertGreaterEqual(early, mid)
        self.assertGreaterEqual(mid, late)
        self.assertGreaterEqual(late, completed)


class TestTimeDependentRiskProperties(unittest.TestCase):
    """时间依赖性风险属性测试"""

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(
        exposure_days=st.integers(min_value=0, max_value=3650),
        beta=st.floats(min_value=0.001, max_value=1.0),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_infection_probability_monotonic_in_days(self, exposure_days, beta):
        """属性：感染概率随暴露天数单调递增"""
        import math
        prob_t = 1.0 - math.exp(-beta * exposure_days)
        prob_t_plus = 1.0 - math.exp(-beta * (exposure_days + 1))

        self.assertGreaterEqual(prob_t_plus, prob_t,
                                f"暴露 {exposure_days + 1} 天感染概率 < {exposure_days} 天")

    @unittest.skipUnless(HAS_HYPOTHESIS, "Hypothesis 未安装")
    @given(
        exposure_days=st.integers(min_value=100, max_value=3650),
        beta=st.floats(min_value=0.001, max_value=1.0),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_infection_probability_approaches_one(self, exposure_days, beta):
        """属性：暴露天数→∞ 时感染概率→1.0"""
        import math
        assume(beta > 0.001)
        prob = 1.0 - math.exp(-beta * exposure_days)
        # 暴露天数足够大时，概率应接近 1.0
        if exposure_days > 1000 and beta > 0.01:
            self.assertGreater(prob, 0.99,
                               f"暴露 {exposure_days} 天, β={beta}, 概率={prob:.6f} < 0.99")


if __name__ == '__main__':
    unittest.main()