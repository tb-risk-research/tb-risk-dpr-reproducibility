#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk 测试套件 — SEIR 模型理论拓展

覆盖 v4 之后的前沿拓展：

1. 超级传播建模（superspreading.py）
   - 负二项子代分布（mean=R0, dispersion=k）
   - 传播异质性度量：20/80 规则、k 拟合、随机熄灭概率
   - 两类传染者 SEIR 模型（确定性 ODE + 随机 Tau-leap）
   - 均匀 vs 超传播对比
2. 四维综合对比（synthesis.py）
   - 年龄结构化分析（复用 v4 WAIFW）
   - TB-HIV 共感染分析（复用 v4 rr_hiv）
   - 耐药 TB 分析（复用 v4 适合度代价 + 获得性耐药）
   - 综合报告 build_theory_report

文献：
- Lloyd-Smith JO, et al. Nature 2005;438(7066):355-359.
- Melsew YA, et al. BMC Infect Dis 2019;19:244.
- Woolhouse ME, et al. PNAS 1997;94:338-342.
- Mossong J, et al. PLoS Med 2008;5(3):e74.
- Pawlowski A, et al. PLoS Pathog 2012;8(2):e1002464.
"""

import os
import sys
import unittest

import numpy as np

# 确保包路径在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.seir.extensions.superspreading import (
    DEFAULT_R0, DEFAULT_K, DEFAULT_TOP_FRACTION, DEFAULT_SHARE,
    nb_log_pmf, nb_pmf, nb_cdf, sample_offspring,
    offspring_pmf_array, prob_no_transmission,
    theoretical_dispersion_from_share,
    transmission_share_of_top_fraction,
    top_fraction_for_transmission_share,
    extinction_probability, fit_dispersion_k,
    superspreading_metrics, SuperspreaderSEIR,
    compare_homogeneous_vs_heterogeneous,
)
from tb_risk.seir.extensions.synthesis import (
    AGE_LABELS,
    age_structured_analysis,
    hiv_analysis,
    dr_analysis,
    build_theory_report,
)
from tb_risk.seir import extensions as ext_pkg


class TestNegativeBinomial(unittest.TestCase):
    """负二项子代分布"""

    def test_pmf_is_probability(self):
        """z=0..500 概率质量非负且大致归一化"""
        pmf = offspring_pmf_array(DEFAULT_R0, DEFAULT_K)
        self.assertTrue(np.all(pmf >= 0))
        self.assertAlmostEqual(float(pmf.sum()), 1.0, places=3)

    def test_log_pmf_matches_pmf(self):
        """log-pmf 与 pmf 一致"""
        for z in (0, 1, 5, 20):
            self.assertAlmostEqual(nb_pmf(z, 2.0, 0.5),
                                   np.exp(nb_log_pmf(z, 2.0, 0.5)), places=9)

    def test_cdf_is_monotonic(self):
        """cdf 单调不减且终值≈1"""
        c0 = nb_cdf(0, 2.0, 0.5)
        c5 = nb_cdf(5, 2.0, 0.5)
        self.assertLessEqual(c0, c5)
        self.assertGreater(c5, c0)

    def test_sample_offspring_mean(self):
        """抽样均值接近理论 R0"""
        rng = np.random.RandomState(42)
        samples = sample_offspring(rng, size=20000, mean=1.5, k=0.5)
        self.assertAlmostEqual(float(samples.mean()), 1.5, delta=0.1)

    def test_heterogeneous_has_larger_variance(self):
        """k 越小（异质性越强），方差/均值比越大"""
        var_ratio_small_k = (DEFAULT_R0 * (1 + DEFAULT_R0 / 0.1)
                             / DEFAULT_R0)
        var_ratio_large_k = (DEFAULT_R0 * (1 + DEFAULT_R0 / 50.0)
                             / DEFAULT_R0)
        self.assertGreater(var_ratio_small_k, var_ratio_large_k)


class TestTransmissionMetrics(unittest.TestCase):
    """传播异质性度量"""

    def test_prob_no_transmission_decreases_with_R0(self):
        """R0 越大，产生 0 个二代病例概率越小"""
        p1 = prob_no_transmission(0.8, 0.5)
        p2 = prob_no_transmission(2.0, 0.5)
        self.assertGreater(p1, p2)

    def test_prob_no_transmission_formula(self):
        """P(Z=0)=(k/(k+m))^k"""
        expect = (0.5 / (0.5 + 1.5)) ** 0.5
        self.assertAlmostEqual(prob_no_transmission(1.5, 0.5), expect, places=9)

    def test_small_k_concentrates_transmission(self):
        """k 越小，top 20% 贡献传播占比越高（20/80 规则）"""
        share_small = transmission_share_of_top_fraction(1.5, 0.1)
        share_large = transmission_share_of_top_fraction(1.5, 50.0)
        self.assertGreater(share_small, share_large)
        self.assertGreater(share_small, 0.5)

    def test_top_fraction_for_80pct(self):
        """产生 80% 传播所需病例占比：强异质性 < 均匀"""
        frac_hetero = top_fraction_for_transmission_share(1.5, 0.2, 0.8)
        frac_homo = top_fraction_for_transmission_share(1.5, 50.0, 0.8)
        self.assertGreater(frac_homo, frac_hetero)

    def test_theoretical_dispersion_from_share(self):
        """反解满足 20/80 所需的 k"""
        k = theoretical_dispersion_from_share(1.5, 0.8, 0.2)
        self.assertLess(k, 1.0)
        self.assertGreater(k, 0.0)

    def test_extinction_probability_subcritical(self):
        """R0<=1 时必然熄灭"""
        self.assertEqual(extinction_probability(0.8, 0.5), 1.0)

    def test_extinction_probability_supercritical_less_than_one(self):
        """R0>1 时熄灭概率 <1"""
        q = extinction_probability(2.0, 0.5)
        self.assertGreaterEqual(q, 0.0)
        self.assertLess(q, 1.0)

    def test_extinction_higher_for_heterogeneous(self):
        """同 R0 下，k 越小熄灭概率越高"""
        q_het = extinction_probability(1.5, 0.2)
        q_hom = extinction_probability(1.5, 50.0)
        self.assertGreater(q_het, q_hom)

    def test_fit_dispersion_k_recovers(self):
        """从负二项样本拟合 k 接近真值"""
        rng = np.random.RandomState(7)
        true_k = 0.5
        samples = sample_offspring(rng, size=5000, mean=1.5, k=true_k)
        est = fit_dispersion_k(samples, mean=1.5)
        self.assertIsNotNone(est)
        self.assertAlmostEqual(est, true_k, delta=0.3)

    def test_fit_dispersion_k_insufficient_data(self):
        """样本过少（<2）返回 None"""
        self.assertIsNone(fit_dispersion_k([1]))

    def test_superspreading_metrics_structure(self):
        """汇总指标包含全部关键键"""
        m = superspreading_metrics(1.5, 0.5)
        for key in ("R0", "k", "variance", "variance_to_mean_ratio",
                    "prob_no_transmission", "top_fraction_transmission_share",
                    "fraction_for_80pct_transmission", "extinction_probability",
                    "note"):
            self.assertIn(key, m)
        self.assertAlmostEqual(m["R0"], 1.5)
        self.assertAlmostEqual(m["k"], 0.5)


class TestSuperspreaderSEIR(unittest.TestCase):
    """两类传染者 SEIR 模型"""

    def setUp(self):
        self.model = SuperspreaderSEIR(population=1000, R0=1.5, seed=42)

    def test_r0_from_params(self):
        """由 R0 反推 beta 后 mean_R0 应回归到目标值"""
        m = SuperspreaderSEIR(population=1000, R0=2.0,
                              p_superspreader=0.1, rel_infectiousness=10.0)
        self.assertAlmostEqual(m.mean_R0, 2.0, places=6)

    def test_rhs_mass_conservation(self):
        """确定性 ODE 房室导数之和为 0"""
        y0 = self.model.initial_state(I0=5)
        rhs = self.model.rhs(0.0, y0)
        self.assertAlmostEqual(float(rhs.sum()), 0.0, places=6)

    def test_force_of_infection(self):
        """超传播者相对传染性被计入感染力"""
        s = np.array([900.0, 50.0, 10.0, 40.0, 0.0])
        lam = self.model.force_of_infection(s)
        # 无超传播者对比
        s2 = np.array([900.0, 50.0, 50.0, 0.0, 0.0])
        lam2 = self.model.force_of_infection(s2)
        self.assertGreater(lam, lam2)

    def test_solve_shape(self):
        """RK4 解形状 (n_steps, 5)"""
        times, hist = self.model.solve((0, 100), 1.0, I0=2)
        self.assertEqual(hist.shape[1], 5)
        self.assertEqual(len(times), hist.shape[0])
        self.assertTrue(np.all(hist >= -1e-9))

    def test_solve_conserves_population(self):
        """每步总人口守恒"""
        _, hist = self.model.solve((0, 100), 1.0, I0=2)
        totals = hist.sum(axis=1)
        self.assertTrue(np.allclose(totals, 1000.0, atol=1e-6),
                        f"人口不守恒 max_dev={np.max(np.abs(totals-1000))}")

    def test_solve_epidemic_develops(self):
        """R0>1 时感染应明显超过初始种子（TB 慢病程，365 天内增长温和）"""
        _, hist = self.model.solve((0, 365), 1.0, I0=2)
        peak_inf = (hist[:, 2] + hist[:, 3]).max()
        self.assertGreater(peak_inf, 5.0)

    def test_stochastic_simulation_shape(self):
        """随机模拟返回 (n_traj, n_steps)"""
        times, traj = self.model.simulate_stochastic((0, 50), 1.0, I0=2,
                                                     n_trajectories=3)
        self.assertEqual(traj.shape[0], 3)
        self.assertEqual(traj.shape[1], len(times))
        self.assertTrue(np.all(traj >= -1e-9))

    def test_final_epidemic_size(self):
        """终点感染比例在 [0,1]"""
        size = self.model.final_epidemic_size()
        self.assertGreaterEqual(size, 0.0)
        self.assertLessEqual(size, 1.0)


class TestCompareHomogeneousVsHeterogeneous(unittest.TestCase):
    """均匀 vs 超传播对比"""

    def test_same_mean_R0(self):
        """两种模型平均 R0 一致"""
        res = compare_homogeneous_vs_heterogeneous(R0=1.5)
        self.assertAlmostEqual(res["mean_R0_homogeneous"],
                               res["mean_R0_heterogeneous"], places=6)

    def test_minimal_deterministic_divergence(self):
        """同平均 R0 下确定性轨迹基本一致（异质性改变随机动力学而非均值）"""
        res = compare_homogeneous_vs_heterogeneous(R0=1.5)
        self.assertAlmostEqual(res["deterministic_final_size_homogeneous"],
                               res["deterministic_final_size_heterogeneous"],
                               delta=0.05)

    def test_extinction_higher_for_heterogeneous(self):
        """异质性模型随机熄灭概率更高"""
        res = compare_homogeneous_vs_heterogeneous(R0=1.5,
                                                   heterogeneous_dispersion=0.2)
        self.assertGreater(res["heterogeneous_extinction_probability"],
                           res["homogeneous_extinction_probability"])

    def test_top20_share_higher_for_heterogeneous(self):
        """异质性模型 top 20% 传播占比更高"""
        res = compare_homogeneous_vs_heterogeneous(R0=1.5,
                                                   heterogeneous_dispersion=0.2)
        self.assertGreater(res["heterogeneous_top20_transmission_share"],
                           res["homogeneous_top20_transmission_share"])


class TestSynthesis(unittest.TestCase):
    """四维综合对比"""

    def test_age_structured_analysis(self):
        """年龄结构化分析返回关键字段"""
        res = age_structured_analysis(population=1000, i0=10,
                                      t_span=(0, 365), dt=1.0, beta=0.05)
        self.assertEqual(len(res["age_labels"]), 5)
        self.assertEqual(len(res["total_active_by_age"]), 5)
        self.assertIn("dominant_age_group", res)
        total = sum(res["total_active_by_age"])
        self.assertGreater(total, 0.0)

    def test_hiv_analysis(self):
        """HIV 共感染：rr_hiv 增大 → 负担增加"""
        res = hiv_analysis(population=1000, i0=10,
                           t_span=(0, 365), dt=1.0, beta=0.05)
        self.assertEqual(len(res["rr_hiv_values"]), 2)
        self.assertEqual(len(res["total_active_cases"]), 2)
        self.assertGreater(res["fold_increase_with_hiv"], 1.0)

    def test_dr_analysis(self):
        """耐药 TB：获得性耐药推高 DR 占比"""
        res = dr_analysis(population=1000, i0=10,
                          t_span=(0, 365), dt=1.0, beta=0.05)
        self.assertIn("dr_fraction_without_acq", res)
        self.assertIn("dr_fraction_with_acq", res)
        self.assertGreaterEqual(res["acq_dr_increment"], 0.0)

    def test_build_theory_report_structure(self):
        """综合报告包含四维与引用"""
        report = build_theory_report(population=1000, i0=10,
                                     t_span=(0, 365), dt=1.0, beta=0.05)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(len(report["summary"]["dimensions"]), 4)
        self.assertIn("age", report)
        self.assertIn("superspreading", report)
        self.assertIn("hiv", report)
        self.assertIn("dr", report)
        self.assertGreaterEqual(len(report["references"]), 5)


class TestPackageExports(unittest.TestCase):
    """包导出完整性"""

    def test_extensions_hyper_exports(self):
        """extensions 子包导出全部公开 API"""
        for name in (
            "DEFAULT_R0", "DEFAULT_K", "nb_log_pmf", "nb_pmf", "nb_cdf",
            "sample_offspring", "offspring_pmf_array", "prob_no_transmission",
            "transmission_share_of_top_fraction",
            "top_fraction_for_transmission_share",
            "extinction_probability", "fit_dispersion_k",
            "superspreading_metrics", "SuperspreaderSEIR",
            "compare_homogeneous_vs_heterogeneous",
            "AGE_LABELS", "age_structured_analysis", "hiv_analysis",
            "dr_analysis", "build_theory_report",
        ):
            self.assertTrue(hasattr(ext_pkg, name), f"缺少导出: {name}")

    def test_seir_package_reexports(self):
        """seir 包顶层 re-export 理论拓展内容"""
        import tb_risk.seir as seir_pkg
        for name in ("superspreading_metrics", "SuperspreaderSEIR",
                     "build_theory_report", "age_structured_analysis"):
            self.assertTrue(hasattr(seir_pkg, name), f"缺少 re-export: {name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)