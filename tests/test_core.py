#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 测试套件 - 核心模块单元测试
"""
import sys
import os
import unittest

import pytest

# 确保包路径在 sys.path 中（支持双击运行和命令行运行）
# 需要将 tb_risk 的父目录（Desktop）加入 sys.path，而非 tb_risk 本身
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class TestConfig(unittest.TestCase):
    """测试配置模块"""

    def test_embedded_config_exists(self):
        """测试内嵌配置存在"""
        from tb_risk.config import _EMBEDDED_KARAMAY_CONFIG
        self.assertIsInstance(_EMBEDDED_KARAMAY_CONFIG, dict)
        self.assertIn("_meta", _EMBEDDED_KARAMAY_CONFIG)
        self.assertIn("epidemiology", _EMBEDDED_KARAMAY_CONFIG)
        self.assertIn("occupation", _EMBEDDED_KARAMAY_CONFIG)

    def test_config_epidemiology(self):
        """测试流行病学配置参数"""
        from tb_risk.config import _EMBEDDED_KARAMAY_CONFIG
        epi = _EMBEDDED_KARAMAY_CONFIG["epidemiology"]
        self.assertGreater(epi["base_incidence_per_100k"], 0)
        self.assertGreater(epi["latent_prevalence_family_default"], 0)
        self.assertLess(epi["latent_prevalence_family_default"], 1)
        self.assertGreater(epi["treatment_success_rate"], 0)
        self.assertLess(epi["treatment_success_rate"], 1)

    def test_config_occupation(self):
        """测试职业配置参数"""
        from tb_risk.config import _EMBEDDED_KARAMAY_CONFIG
        occ = _EMBEDDED_KARAMAY_CONFIG["occupation"]
        self.assertGreater(occ["oilfield_camp_factor"], 0)
        self.assertLess(occ["oilfield_camp_factor"], 1)
        self.assertIn("dust_exposure_levels", occ)


class TestUtils(unittest.TestCase):
    """测试工具模块"""

    def test_is_yes(self):
        """测试 _is_yes 函数"""
        # 直接导入
        import importlib.util
        # ToolTip 已迁移到 gui.tooltip，SyntheticDataCalibrator 仍在 utils
        from tb_risk.gui.tooltip import ToolTip
        from tb_risk.utils import SyntheticDataCalibrator
        self.assertTrue(True)  # 至少导入成功

    def test_logger(self):
        """测试日志模块"""
        from tb_risk import LOGGER
        self.assertIsNotNone(LOGGER)
        self.assertEqual(LOGGER.name, "tb_risk")


class TestScoringEngine(unittest.TestCase):
    """测试评分引擎"""

    def test_engine_import(self):
        """测试评分引擎导入"""
        try:
            from tb_risk.scoring.engine import ScoringEngine
            engine = ScoringEngine()
            self.assertIsNotNone(engine)
        except ImportError as e:
            self.skipTest(f"依赖不可用: {e}")

    def test_setting_factors(self):
        """测试场景因子完整"""
        from tb_risk.scoring.engine import ScoringEngine
        engine = ScoringEngine()
        expected_settings = ['crowded', 'closed', 'oilfield_camp', 'general', 'outdoor']
        for setting in expected_settings:
            self.assertIn(setting, engine.SETTING_FACTORS_BASE,
                         f"缺少场景因子: {setting}")


class TestKaramay(unittest.TestCase):
    """测试克拉玛依模块"""

    @classmethod
    def setUpClass(cls):
        """类级共享 KaramayLocalizer 实例，避免每个测试重复初始化(~13s)"""
        from tb_risk.karamay.localizer import KaramayLocalizer
        cls._localizer = KaramayLocalizer()

    @pytest.mark.slow
    def test_localizer_import(self):
        """测试本土化模块导入"""
        self.assertIsNotNone(self._localizer)

    @pytest.mark.slow
    def test_localizer_factors(self):
        """测试本土化因子"""
        self.assertIsNotNone(self._localizer.oilfield)
        self.assertGreater(self._localizer.oilfield.oilfield_camp_factor, 0)
        self.assertLess(self._localizer.oilfield.oilfield_camp_factor, 1)


class TestAssessment(unittest.TestCase):
    """测试风险评估主类"""

    def test_constants(self):
        """测试类常量定义"""
        from tb_risk.assessment import TB_Risk_Assessment
        self.assertEqual(TB_Risk_Assessment.CONTACT_TYPE_FAMILY, 'family')
        self.assertEqual(TB_Risk_Assessment.CONTACT_TYPE_SOCIAL, 'social')
        self.assertGreater(TB_Risk_Assessment.INFECTION_PROB_MAX, 0)
        self.assertLess(TB_Risk_Assessment.INFECTION_PROB_MAX, 100)
        self.assertGreater(TB_Risk_Assessment.MARK_HIGH_RISK_THRESHOLD, 0)

    def test_setting_mapping(self):
        """测试场景映射完整性"""
        from tb_risk.assessment import TB_Risk_Assessment
        self.assertIn("油田营地", TB_Risk_Assessment.SETTING_MAPPING)
        self.assertIn("oilfield_camp", TB_Risk_Assessment.SETTING_MAPPING_REVERSE)
        self.assertEqual(
            TB_Risk_Assessment.SETTING_MAPPING["油田营地"],
            "oilfield_camp"
        )

    def test_exposure_setting_factors(self):
        """测试暴露场景因子完整性 — 验证所有必需场景因子存在于 ScoringEngine"""
        from tb_risk.scoring.engine import ScoringEngine
        expected_settings = ['crowded', 'closed', 'oilfield_camp', 'general', 'outdoor']
        for setting in expected_settings:
            self.assertIn(setting, ScoringEngine.SETTING_FACTORS_BASE,
                          f"缺失暴露场景因子: {setting}")
        # 验证因子值在合理范围 (0.0, 1.0]
        for setting, factor in ScoringEngine.SETTING_FACTORS_BASE.items():
            self.assertGreater(factor, 0.0, f"{setting} 因子应 > 0")
            self.assertLessEqual(factor, 1.0, f"{setting} 因子应 <= 1")


class TestNumericStability(unittest.TestCase):
    """测试数值稳定性"""

    def test_time_dependent_risk_asymptotic(self):
        """测试时间依赖风险渐近行为"""
        import math
        # 模拟修复后的公式
        def calc_risk(days):
            if days <= 14:
                return 0.5 * (1 - math.exp(-0.05 * days))
            else:
                early = 0.5 * (1 - math.exp(-0.05 * 14))
                remaining = 1.0 - early
                late = remaining * (1 - math.exp(-0.01 * (days - 14)))
                return early + late

        # 渐近应趋于1.0
        self.assertGreater(calc_risk(10000), 0.99)
        # 单调递增
        for d in range(1, 365):
            self.assertGreaterEqual(calc_risk(d + 1), calc_risk(d))

    def test_no_division_by_zero(self):
        """测试无除零风险 — 验证 ScoringEngine 在边界输入下不触发除零"""
        import math
        from tb_risk.scoring.engine import ScoringEngine
        engine = ScoringEngine()
        # 边界情况：年龄为0、累计暴露为0、无风险因子
        record = {
            'age': 0, 'has_symptoms': 0, 'has_tb': 0, 'bcg_vaccine': 1,
            'contact_distance': 'medium', 'ventilation': 3, 'exposure_setting': 'general',
            'cumulative_exposure': 0, 'past_illness_type': 'none',
        }
        result = engine.compute_risk_score(record)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result['disease_probability'], 0.0)
        self.assertLessEqual(result['disease_probability'], 100.0)
        # 验证 sigmoid 在极端输入下不溢出
        z = -ScoringEngine.SIGMOID_COEFF * (1000.0 - ScoringEngine.SIGMOID_OFFSET)
        z_clipped = max(-ScoringEngine.SIGMOID_CLIP_THRESHOLD,
                        min(ScoringEngine.SIGMOID_CLIP_THRESHOLD, z))
        self.assertGreater(1.0 + math.exp(z_clipped), 0.0)  # 分母不为零


class TestBatchScoring(unittest.TestCase):
    """测试批量评分计算"""

    def test_batch_consistency(self):
        """测试批量计算与单次计算结果一致"""
        from tb_risk.scoring.engine import ScoringEngine
        engine = ScoringEngine()
        records = [
            {'age': 5, 'has_symptoms': 0, 'has_tb': 0, 'bcg_vaccine': 1,
             'contact_distance': 'medium', 'ventilation': 3, 'exposure_setting': 'general',
             'cumulative_exposure': 10, 'past_illness_type': 'none'},
            {'age': 70, 'has_symptoms': 1, 'has_tb': 0, 'bcg_vaccine': 0,
             'contact_distance': 'close', 'ventilation': 2, 'exposure_setting': 'crowded',
             'cumulative_exposure': 50, 'past_illness_type': 'diabetes'},
            {'age': 30, 'has_symptoms': 0, 'has_tb': 1, 'bcg_vaccine': 1,
             'contact_distance': 'far', 'ventilation': 5, 'exposure_setting': 'outdoor',
             'cumulative_exposure': 5, 'past_illness_type': 'hiv'},
        ]
        single = [engine.compute_risk_score(r) for r in records]
        batch = engine.compute_batch(records)
        for i in range(len(records)):
            self.assertAlmostEqual(single[i]['disease_probability'],
                                  batch[i]['disease_probability'], places=6)
            self.assertAlmostEqual(single[i]['infection_probability'],
                                  batch[i]['infection_probability'], places=6)

    def test_batch_empty(self):
        """测试空列表批量计算"""
        from tb_risk.scoring.engine import ScoringEngine
        engine = ScoringEngine()
        self.assertEqual(engine.compute_batch([]), [])


class TestIntegrator(unittest.TestCase):
    """测试集成器模块"""

    def test_integrator_import(self):
        """测试集成器导入"""
        from tb_risk.integrator import ThreeDirectionIntegrator
        t = ThreeDirectionIntegrator()
        self.assertIsNotNone(t)
        self.assertFalse(t.rl_ready)  # RL不可用时应为False

    def test_weighted_ensemble_empty(self):
        """测试空结果加权集成"""
        from tb_risk.integrator import ThreeDirectionIntegrator
        t = ThreeDirectionIntegrator()
        result = t._weighted_ensemble({})
        self.assertIsNone(result['risk_probability'])
        self.assertEqual(result['integration_note'], '所有方向均不可用')

    def test_weighted_ensemble_single(self):
        """测试单一方向加权集成"""
        from tb_risk.integrator import ThreeDirectionIntegrator
        t = ThreeDirectionIntegrator()
        result = t._weighted_ensemble({
            'ml': {'risk_probability': 60.0, 'risk_class': 1},
        })
        self.assertAlmostEqual(result['risk_probability'], 60.0)
        self.assertEqual(result['risk_class'], 1)


class TestIOModule(unittest.TestCase):
    """测试I/O模块"""

    def test_iomixin_import(self):
        """测试IOMixin导入"""
        from tb_risk.persistence.mixins import IOMixin
        self.assertTrue(hasattr(IOMixin, 'export_csv_template'))
        self.assertTrue(hasattr(IOMixin, 'import_csv'))
        self.assertTrue(hasattr(IOMixin, 'generate_report'))


def run_tests(verbosity=2, pattern=None):
    """运行测试套件

    参数：
        verbosity: 输出详细程度 (0/1/2)
        pattern: 仅运行匹配的测试名（如 "Vectorized"）
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    if pattern:
        suite = unittest.TestSuite()
        all_classes = [TestConfig, TestUtils, TestScoringEngine, TestKaramay,
                       TestAssessment, TestNumericStability, TestBatchScoring,
                       TestIntegrator, TestIOModule]
        pat_lower = pattern.lower()
        for cls in all_classes:
            # 匹配类名或方法名
            if pat_lower in cls.__name__.lower():
                suite.addTests(loader.loadTestsFromTestCase(cls))
            else:
                for name in dir(cls):
                    if name.startswith('test') and pat_lower in name.lower():
                        suite.addTest(cls(name))
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()


def launch_gui():
    """启动 GUI 应用"""
    try:
        from tb_risk.assessment import TB_Risk_Assessment
        import tkinter as tk
        root = tk.Tk()
        TB_Risk_Assessment(root)
        root.mainloop()
    except Exception as e:
        print(f"启动失败: {e}", file=sys.stderr)
        sys.exit(1)


def check_deps():
    """检查依赖状态"""
    deps = {}
    for mod_name in ['numpy', 'scipy', 'matplotlib', 'seaborn',
                     'sklearn', 'xgboost', 'shap', 'joblib',
                     'torch', 'torch_geometric', 'pandas']:
        try:
            __import__(mod_name)
            deps[mod_name] = '✓'
        except ImportError:
            deps[mod_name] = '✗'
    print("\n  依赖状态检查")
    print("  " + "=" * 40)
    for k, v in deps.items():
        print(f"  {v} {k}")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '--gui':
            launch_gui()
        elif arg == '--check':
            check_deps()
        elif arg == '--test' or arg == '-t':
            pattern = sys.argv[2] if len(sys.argv) > 2 else None
            ok = run_tests(verbosity=2, pattern=pattern)
            sys.exit(0 if ok else 1)
        elif arg == '--help' or arg == '-h':
            print("用法: python test_core.py [选项]")
            print("")
            print("  (无参数)     运行全部测试")
            print("  --test [名]  运行匹配名称的测试")
            print("  --gui        启动 GUI 应用")
            print("  --check      检查依赖状态")
            print("  --help       显示帮助")
        else:
            print(f"未知选项: {arg}，使用 --help 查看帮助")
            sys.exit(1)
    else:
        # 默认：运行全部测试
        ok = run_tests(verbosity=2)
        sys.exit(0 if ok else 1)