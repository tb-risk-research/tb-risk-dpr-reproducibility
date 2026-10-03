#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1：真实患病率口径的部署评估 — 决策指标模块

背景（用户 2026-08-24 指令）：场景训练数据阳性率 20.7%，Kenya 真实
0.53%——患病率断层 40 倍，阈值和概率不可直接迁移。此前验证只报
AUROC；筛查部署的真正决策指标是 AUPRC、PPV@固定敏感度、NNS
（每筛查 N 人发现 1 例）。

本测试覆盖 scoring/ml/deployment_metrics.py：
1. logit_shift：先验校正（Saerens 2002）——排序保持、恒等变换、
   目标均值收敛、数值边界裁剪；
2. ppv_at_sensitivity：敏感度约束下的最优工作点（满足敏感度的
   最高阈值 → PPV 最大 / NNS 最小）、并列分数、敏感度不可达；
3. screening_metrics：组合指标（AUROC/AUPRC/PPV/NNS/Brier）。

文献：
- Saerens M, Latinne P, Decaestecker C. Adjusting the Outputs of a
  Classifier to New a Priori Probabilities: A Simple Procedure.
  Neural Computation 2002;14:21-41（logit 先验平移）
- Trevethan R. Sensitivity, Specificity, and Predictive Values.
  Educ Health 2017;30:36-38（PPV 与患病率的关系）
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.scoring.ml.deployment_metrics import (  # noqa: E402
    logit_shift,
    ppv_at_sensitivity,
    screening_metrics,
)


# ============================================================================
# 1. logit_shift（先验校正）
# ============================================================================

class TestLogitShift:

    def test_identity_when_same_prevalence(self):
        """源/目标患病率相同 → 概率不变（恒等）。"""
        p = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
        out = logit_shift(p, source_prevalence=0.2, target_prevalence=0.2)
        np.testing.assert_allclose(out, p, atol=1e-12)

    def test_rank_preserved(self):
        """先验平移是严格单调变换 → 排序不变（AUROC 不受影响）。"""
        rng = np.random.RandomState(42)
        p = rng.rand(200)
        out = logit_shift(p, source_prevalence=0.207, target_prevalence=0.0053)
        np.testing.assert_array_equal(np.argsort(out), np.argsort(p))

    def test_mean_converges_to_target(self):
        """校正后概率均值 ≈ 目标患病率（logit 均值平移性质）。"""
        rng = np.random.RandomState(0)
        p = np.clip(rng.normal(0.207, 0.15, 100000), 1e-4, 1 - 1e-4)
        out = logit_shift(p, source_prevalence=0.207, target_prevalence=0.0053)
        assert abs(out.mean() - 0.0053) < 0.002

    def test_clip_to_valid_range(self):
        """0/1 端点数值安全（裁剪到 (eps, 1-eps)），无 nan/inf。"""
        p = np.array([0.0, 1.0, 0.5])
        out = logit_shift(p, source_prevalence=0.2, target_prevalence=0.01)
        assert np.all(np.isfinite(out))
        assert np.all(out > 0.0) and np.all(out < 1.0)

    def test_shift_direction(self):
        """目标患病率更低 → 概率整体下移。"""
        p = np.array([0.3, 0.6])
        out = logit_shift(p, source_prevalence=0.5, target_prevalence=0.05)
        assert np.all(out < p)


# ============================================================================
# 2. ppv_at_sensitivity
# ============================================================================

class TestPpvAtSensitivity:

    def test_hand_computed_example(self):
        """手工小例：5 阳性/95 阴性，构造排序后精确断言工作点。

        分数设计（降序）：rank1-5 = [阳,阴,阳,阳,阳]（前 5 名 4 阳
        1 阴），rank6 = 阴，随后第 5 个阳性在 rank7（低分），其余低分
        阴性。敏感度 80% → 需捕获 4/5 阳性 → 阈值取 rank5
        （满足 sens>=0.8 的最高阈值）→ TP=4, FP=1, PPV=0.8, NNS=1.25。
        """
        y = np.zeros(100, dtype=int)
        scores = np.zeros(100)
        # 降序放置：rank1-5: [阳,阴,阳,阳,阳]; rank6: 阴; rank7: 阳
        order = [(0, 1, 0.99), (1, 0, 0.98), (2, 1, 0.97),
                 (3, 1, 0.96), (4, 1, 0.95), (5, 0, 0.94), (6, 1, 0.50)]
        for idx, label, sc in order:
            y[idx] = label
            scores[idx] = sc
        res = ppv_at_sensitivity(y, scores, sensitivity_target=0.80)
        assert res['sensitivity'] == pytest.approx(0.80)
        assert res['tp'] == 4
        assert res['fp'] == 1
        assert res['ppv'] == pytest.approx(4 / 5)
        assert res['nns'] == pytest.approx(5 / 4)
        assert res['n_flagged'] == 5

    def test_perfect_classifier(self):
        """完美排序：sens 80% 工作点 PPV=1，NNS=1。"""
        y = np.array([1] * 10 + [0] * 90)
        scores = np.concatenate([np.linspace(0.99, 0.90, 10),
                                 np.linspace(0.5, 0.1, 90)])
        res = ppv_at_sensitivity(y, scores, sensitivity_target=0.80)
        assert res['ppv'] == pytest.approx(1.0)
        assert res['nns'] == pytest.approx(1.0)

    def test_unreachable_sensitivity_flags_all(self):
        """敏感度目标不可达（阳性数少于 ceil(0.8*n_pos)）→ 全量筛查
        并显式标记 reached=False。"""
        y = np.array([1, 0, 0, 0])
        scores = np.array([0.1, 0.9, 0.8, 0.7])
        res = ppv_at_sensitivity(y, scores, sensitivity_target=0.80)
        assert res['reached'] is False
        assert res['n_flagged'] == 4  # 退化为全量筛查
        assert res['sensitivity'] == pytest.approx(1.0)

    def test_tied_scores(self):
        """并列分数按组进入（同分组要么全筛查要么全不筛查）。"""
        y = np.array([1, 1, 0, 0, 0, 0])
        scores = np.array([0.9, 0.9, 0.9, 0.5, 0.5, 0.5])
        res = ppv_at_sensitivity(y, scores, sensitivity_target=0.50)
        # 阈值取 0.9 组：TP=2, FP=1 → sens=1.0（并列组不可拆）, PPV=2/3
        assert res['tp'] == 2
        assert res['ppv'] == pytest.approx(2 / 3)

    def test_no_positives_raises(self):
        """无阳性样本 → ValueError（工作点无定义）。"""
        with pytest.raises(ValueError):
            ppv_at_sensitivity(np.zeros(10, dtype=int), np.arange(10) / 10.0)


# ============================================================================
# 3. screening_metrics（组合指标）
# ============================================================================

class TestScreeningMetrics:

    def _synthetic(self, n=20000, prevalence=0.005, seed=7):
        rng = np.random.RandomState(seed)
        n_pos = int(round(n * prevalence))
        y = np.zeros(n, dtype=int)
        y[:n_pos] = 1
        # AUROC≈0.70 的分数：阳性 N(0.55,1)，阴性 N(0,1)
        scores = np.concatenate([
            rng.normal(0.55, 1.0, n_pos), rng.normal(0.0, 1.0, n - n_pos)])
        return y, scores

    def test_auprc_and_auroc_reported(self):
        y, scores = self._synthetic()
        res = screening_metrics(y, scores, sensitivity_target=0.80)
        assert 0.65 < res['auroc'] < 0.75
        assert 0.004 < res['auprc'] < 0.05
        assert res['base_rate'] == pytest.approx(0.005)
        assert res['ppv_at_sensitivity']['sensitivity'] >= 0.80

    def test_nns_consistency(self):
        """NNS = 1/PPV，且优于随机筛查基线（随机 NNS = 1/患病率）。"""
        y, scores = self._synthetic()
        res = screening_metrics(y, scores, sensitivity_target=0.80)
        ppv = res['ppv_at_sensitivity']['ppv']
        assert res['nns'] == pytest.approx(1.0 / ppv)
        assert res['nns'] < 1.0 / res['base_rate']  # 有判别力 → 优于随机

    def test_logit_shift_calibration_path(self):
        """给定 source/target 患病率 → 输出校正后 Brier + 校正幅度。

        分数构造为"20.7% 患病率人群上校准的概率"（均值≈0.207）——
        在 0.5% 部署人群上严重过报；logit 平移后应显著收敛。
        """
        y, _ = self._synthetic()
        rng = np.random.RandomState(7)
        n_pos = int(y.sum())
        # AUROC≈0.70 的 logit 分数 + 平移到 20.7% 概率口径
        logit_scores = np.concatenate([
            rng.normal(0.55, 1.0, n_pos), rng.normal(0.0, 1.0, len(y) - n_pos)])
        scores = 1.0 / (1.0 + np.exp(-(logit_scores
                                       + np.log(0.207 / 0.793))))
        # 构造自检：概率均值在 20% 口径量级（logistic-normal 的 Jensen
        # 效应使均值略高于 0.207，约 0.24——不影响"过报于 0.5% 人群"的
        # 演示语义）
        assert 0.18 < scores.mean() < 0.30
        res = screening_metrics(y, scores, source_prevalence=0.207,
                                target_prevalence=0.005,
                                sensitivity_target=0.80)
        assert 'brier_shifted' in res
        # 20.7% 口径的原始概率在 0.5% 人群上严重过报 →
        # logit 平移后 Brier 必须显著下降
        assert res['brier_shifted'] < res['brier_raw']
        assert 'mean_prob_shifted' in res
        # 均值收敛到目标患病率量级（logit shift 平移 logit 均值，
        # sigmoid 非线性下概率均值有 Jensen 偏差，0.008 vs 0.005 同量级）
        assert 0.002 < res['mean_prob_shifted'] < 0.012

    def test_no_prevalence_given_skips_calibration(self):
        y, scores = self._synthetic(n=2000, prevalence=0.02)
        res = screening_metrics(y, scores)
        assert 'brier_raw' in res
        assert 'brier_shifted' not in res
