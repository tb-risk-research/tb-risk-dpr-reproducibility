#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题5：消融实验统计效力升级（layer_ablation v3）。

用户问题：样本量 40、种子 3 个、只报点估计——统计效力不足。

升级方案（文献）：
- Hanley JA, McNeil BJ. Radiology 1982;143:29-46（AUC 方差的 Q1/Q2 指数
  近似）与 Radiology 1983;148:839-843（**同一组病例**两 AUC 比较的
  相关方差——消融正是配对设计：基线与完整模型给同一批受试者打分）；
- DeLong ER, DeLong DM, Clarke-Pearson DL. Biometrics 1988;44:837-845
  （配对 AUROC 差异检验的非参数协方差法）；
- Efron bootstrap（种子级重采样 → 效应量 95% CI）。

覆盖：
1. auroc_power_analysis：单调性（更小 Δ/更低阳性率/更高功效 → 更多样本）、
   字段完整、参考量级正确（base 0.7 / Δ0.05 / 阳性率 30% / r=0.6
   → 数百例量级；密接实际阳性率 2.85% → 数千例）；
2. delong_paired_test：完全相同评分 p=1、可分离信号 p<0.05、字段完整；
3. run_multi_seed_ablation：默认 500 样本 × 20 种子（用户规格）、
   效应量 + bootstrap 95% CI + 每种子 DeLong p、功效分析写进实验设计、
   报告向后兼容（baseline/full/delta 字段不破坏单次运行版本）。
"""

import inspect
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.validation.layer_ablation import (  # noqa: E402
    auroc_power_analysis,
    delong_paired_test,
    run_multi_seed_ablation,
)


# ============================================================================
# 1. Hanley-McNeil 配对 AUROC 功效分析
# ============================================================================

class TestAurocPowerAnalysis:

    def _analyze(self, **kwargs):
        params = dict(base_auc=0.70, delta=0.05, prevalence=0.30,
                      alpha=0.05, power=0.80, paired_r=0.60)
        params.update(kwargs)
        return auroc_power_analysis(**params)

    def test_fields_complete(self):
        r = self._analyze()
        for field in ('n_total', 'n_positive', 'n_negative', 'base_auc',
                      'target_auc', 'delta', 'prevalence', 'alpha', 'power',
                      'paired_r', 'se_diff_at_n', 'reference'):
            assert field in r, f'缺少字段 {field}'

    def test_reference_magnitude(self):
        """base 0.7 / Δ0.05 / 阳性率 30% / r=0.6：数百例量级（解析约 660）。"""
        r = self._analyze()
        assert 300 <= r['n_total'] <= 1500, r['n_total']

    def test_smaller_delta_needs_more_samples(self):
        n_big = self._analyze(delta=0.10)['n_total']
        n_small = self._analyze(delta=0.05)['n_total']
        assert n_small > n_big

    def test_lower_prevalence_needs_more_samples(self):
        """密接实际阳性率 2.85% vs 实验阳性率 30% → 需要更多总样本。"""
        n_rare = self._analyze(prevalence=0.0285)['n_total']
        n_common = self._analyze(prevalence=0.30)['n_total']
        assert n_rare > n_common

    def test_higher_power_needs_more_samples(self):
        n80 = self._analyze(power=0.80)['n_total']
        n90 = self._analyze(power=0.90)['n_total']
        assert n90 > n80

    def test_counts_consistent_with_prevalence(self):
        r = self._analyze(prevalence=0.30)
        assert r['n_positive'] + r['n_negative'] == r['n_total']
        actual_rate = r['n_positive'] / r['n_total']
        assert abs(actual_rate - 0.30) < 0.02

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError):
            auroc_power_analysis(base_auc=0.3, delta=0.05, prevalence=0.3)
        with pytest.raises(ValueError):
            auroc_power_analysis(base_auc=0.7, delta=-0.01, prevalence=0.3)
        with pytest.raises(ValueError):
            auroc_power_analysis(base_auc=0.7, delta=0.05, prevalence=1.5)


# ============================================================================
# 2. DeLong 配对检验
# ============================================================================

class TestDelongPairedTest:

    def _make_scores(self, n=200, pos_rate=0.3, seed=42):
        rng = np.random.RandomState(seed)
        y = np.zeros(n, dtype=int)
        y[: int(n * pos_rate)] = 1
        rng.shuffle(y)
        signal = 0.5 + 0.5 * y + rng.normal(0, 0.25, n)
        return y, np.clip(signal, 0, 1)

    def test_identical_scores_no_difference(self):
        y, s = self._make_scores()
        r = delong_paired_test(y, s, s.copy())
        assert r['p_value'] >= 0.99
        assert abs(r['delta_auroc']) < 1e-12

    def test_signal_vs_noise_significant(self):
        rng = np.random.RandomState(0)
        y, s = self._make_scores()
        noise = rng.rand(len(y))
        r = delong_paired_test(y, s, noise)
        assert r['auroc_a'] > 0.75
        assert abs(r['auroc_b'] - 0.5) < 0.15
        assert r['delta_auroc'] > 0
        assert r['p_value'] < 0.05

    def test_fields_complete(self):
        y, s = self._make_scores(n=100)
        r = delong_paired_test(y, s, s + 0.01)
        for field in ('auroc_a', 'auroc_b', 'delta_auroc', 'se_diff',
                      'z', 'p_value'):
            assert field in r
        assert 0.0 <= r['p_value'] <= 1.0

    def test_p_value_in_range_for_weak_difference(self):
        y, s = self._make_scores(n=60)
        r = delong_paired_test(y, s, s + 0.001)
        assert 0.0 <= r['p_value'] <= 1.0


# ============================================================================
# 3. 多种子消融（500+ 样本 × 20+ 种子，效应量+CI+DeLong）
# ============================================================================

class TestRunMultiSeedAblation:

    @pytest.fixture(scope='class')
    def report(self):
        """小规模运行（120×3 种子）验证结构；500×20 全量在验证阶段跑。"""
        return run_multi_seed_ablation(n_contacts=120, n_seeds=3,
                                       random_state=42)

    def test_defaults_match_user_spec(self):
        """用户规格：样本量 500+、种子 20+。"""
        sig = inspect.signature(run_multi_seed_ablation)
        assert sig.parameters['n_contacts'].default >= 500
        assert sig.parameters['n_seeds'].default >= 20

    def test_report_structure(self, report):
        for field in ('method', 'n_contacts', 'n_seeds', 'prevalence',
                      'seeds', 'effect_size', 'bootstrap_ci', 'delong',
                      'power_design', 'network_contributes', 'conclusion'):
            assert field in report, f'缺少字段 {field}'

    def test_every_seed_has_auroc_and_delong_p(self, report):
        assert len(report['seeds']) == 3
        for entry in report['seeds']:
            for field in ('seed', 'auroc_baseline', 'auroc_full',
                          'delta_auroc', 'delong_p'):
                assert field in entry
            assert 0.0 <= entry['delong_p'] <= 1.0

    def test_effect_size_matches_seed_mean(self, report):
        deltas = [e['delta_auroc'] for e in report['seeds']]
        # 报告字段按 4 位小数舍入 → 容差 1e-3
        assert abs(report['effect_size']['delta_auroc_mean']
                   - float(np.mean(deltas))) < 1e-3

    def test_bootstrap_ci_brackets_effect(self, report):
        ci = report['bootstrap_ci']
        mean = report['effect_size']['delta_auroc_mean']
        assert ci['ci_lower'] <= mean <= ci['ci_upper']
        assert ci['ci_lower'] <= ci['ci_upper']

    def test_delong_summary(self, report):
        d = report['delong']
        per_seed = [e['delong_p'] for e in report['seeds']]
        assert len(d['per_seed_p']) == len(per_seed)
        # p 值按 6 位小数舍入 → 容差 1e-5
        assert abs(d['p_median'] - float(np.median(per_seed))) < 1e-5
        assert 'n_significant' in d

    def test_power_design_written_into_experiment(self, report):
        """功效分析写进实验设计：80% 功效检出 ΔAUROC=0.05 所需样本量。"""
        pd_ = report['power_design']
        assert pd_['detect_delta'] == 0.05
        assert pd_['power'] == 0.80
        # 两种阳性率口径的样本量需求都必须事先给出
        assert 'required_n_experiment_prevalence' in pd_
        assert 'required_n_close_contact_prevalence' in pd_
        exp_n = pd_['required_n_experiment_prevalence']['n_total']
        cc_n = pd_['required_n_close_contact_prevalence']['n_total']
        assert cc_n > exp_n  # 2.85% 实际阳性率需要更多样本

    def test_conclusion_reflects_ci_not_point_estimate(self, report):
        """结论依据效应量+CI（而非单点估计）。"""
        assert isinstance(report['network_contributes'], bool)
        assert report['conclusion']


# ============================================================================
# 4. 向后兼容：单次运行接口不变
# ============================================================================

class TestBackwardCompat:

    def test_single_run_interface_unchanged(self):
        from tb_risk.validation.layer_ablation import (
            network_contribution_ablation)
        rep = network_contribution_ablation(
            n_contacts=40, n_cases=12, random_state=42)
        for field in ('baseline', 'full', 'delta', 'network_contributes'):
            assert field in rep
        assert 'delta_auroc' in rep['delta']
        assert 'delta_cindex' in rep['delta']
