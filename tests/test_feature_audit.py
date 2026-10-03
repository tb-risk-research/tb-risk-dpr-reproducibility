#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题3：特征冗余与共线性 — 特征审计模块与 v2 精简特征集

覆盖：
1. compute_feature_audit：相关矩阵 / 高相关对 / VIF / 近常数列 / 可推导性；
2. audit_training_csv：对真实训练 CSV 的完整审计（含单变量 AUROC）；
3. SELECTED_FEATURES_V2：22 → 15 精简特征集（每个特征独立临床含义且
   无法互相推导），移除项逐一给出理由。

文献：
- 多重共线性诊断：O'Brien RM. A Caution Regarding Rules of Thumb for
  Variance Inflation Factors. Quality & Quantity 2007;41:673-690
  （VIF 阈值的争议；常用参考 VIF>5 中度、>10 重度）
- 冗余特征与树模型解释失真：Lundberg SM et al. A Unified Approach to
  Interpreting Model Predictions. NeurIPS 2017（一致性公理）
- 临床预测模型特征精简原则：Riley RD et al. BMJ 2020;368:m443
  （TRIPOD 系列：预测因子应有独立临床含义，避免信息重复计数）
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.scoring.ml.feature_audit import (  # noqa: E402
    compute_feature_audit,
    audit_training_csv,
    SELECTED_FEATURES_V2,
    FEATURE_REMOVAL_RATIONALE,
)

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
_ALIGNED_CSV = os.path.join(_DATA_DIR, 'ml_training_scenario_aligned.csv')


# ============================================================================
# 1. compute_feature_audit
# ============================================================================

class TestComputeFeatureAudit:

    def _df_with_known_structure(self):
        rng = np.random.RandomState(42)
        n = 500
        x = rng.rand(n)
        return pd.DataFrame({
            'x': x,
            'y': 2.0 * x + rng.normal(0, 0.01, n),   # 与 x 高相关
            'z': x * (2.0 * x),                       # 可由 x 推导
            'const': np.ones(n),                      # 近常数
            'w': rng.rand(n),                         # 独立
        })

    def test_returns_core_structure(self):
        df = self._df_with_known_structure()
        audit = compute_feature_audit(df)
        assert audit['n_samples'] == 500
        assert audit['n_features'] == 5
        for key in ('high_corr_pairs', 'vif', 'near_constant', 'derivable'):
            assert key in audit

    def test_detects_high_correlation_pair(self):
        audit = compute_feature_audit(self._df_with_known_structure())
        pairs = {(p['a'], p['b']) for p in audit['high_corr_pairs']}
        assert ('x', 'y') in pairs or ('y', 'x') in pairs

    def test_detects_near_constant_column(self):
        audit = compute_feature_audit(self._df_with_known_structure())
        assert 'const' in audit['near_constant']

    def test_detects_derivable_column(self):
        audit = compute_feature_audit(self._df_with_known_structure())
        derivable = {d['feature'] for d in audit['derivable']}
        assert 'z' in derivable

    def test_vif_flags_collinearity(self):
        rng = np.random.RandomState(0)
        n = 400
        x1 = rng.rand(n)
        df = pd.DataFrame({
            'x1': x1,
            'x2': x1 + rng.normal(0, 0.001, n),   # 与 x1 近共线
            'x3': rng.rand(n),                    # 独立
        })
        audit = compute_feature_audit(df)
        assert audit['vif']['x1'] > 5.0
        assert audit['vif']['x2'] > 5.0
        assert audit['vif']['x3'] < 5.0

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            compute_feature_audit(pd.DataFrame())


# ============================================================================
# 2. audit_training_csv（真实训练数据）
# ============================================================================

@pytest.mark.skipif(not os.path.exists(_ALIGNED_CSV),
                    reason='场景对齐训练 CSV 不存在')
class TestAuditTrainingCsv:

    def test_audit_real_csv(self):
        audit = audit_training_csv(_ALIGNED_CSV, label_col='tb_outcome')
        assert audit['n_samples'] > 1000
        assert audit['n_features'] == 22
        for key in ('high_corr_pairs', 'vif', 'near_constant',
                    'derivable', 'univariate_auroc'):
            assert key in audit

    def test_exposure_accumulation_is_derivable(self):
        """用户点名的冗余：exposure_accumulation 应被可推导性检查捕获。"""
        audit = audit_training_csv(_ALIGNED_CSV, label_col='tb_outcome')
        derivable = {d['feature'] for d in audit['derivable']}
        assert 'exposure_accumulation' in derivable

    def test_univariate_auroc_in_range(self):
        audit = audit_training_csv(_ALIGNED_CSV, label_col='tb_outcome')
        assert len(audit['univariate_auroc']) == 22
        for feat, auroc in audit['univariate_auroc'].items():
            assert 0.0 <= auroc <= 1.0

    def test_interaction_derivable_flags(self):
        """可精确重构的交互 age_bcg_decay 应被检出；
        highrisk_comorbid = is_high_risk × past_illness（宿主通路 v2
        重新生成后两列均有变异）→ 从近常数移入可推导报告。"""
        audit = audit_training_csv(_ALIGNED_CSV, label_col='tb_outcome')
        derivable = {d['feature'] for d in audit['derivable']}
        assert 'age_bcg_decay' in derivable
        assert 'exposure_accumulation' in derivable
        # 宿主通路 v2（2026-08-24）：is_high_righ 复活后该交互可由
        # 基础特征两两乘积精确重构
        assert 'highrisk_comorbid' in derivable
        # 六个曾全零的列全部脱离近常数报告
        near_constant = set(audit['near_constant'])
        for col in ('has_tb', 'is_high_risk', 'dm_tb_synergy',
                    'symptom_delay', 'cough_contact', 'highrisk_comorbid'):
            assert col not in near_constant, f'{col} 不应再是近常数列'


# ============================================================================
# 3. v2 精简特征集（22 → 15）
# ============================================================================

class TestSelectedFeaturesV2:

    def test_v2_has_15_features(self):
        assert len(SELECTED_FEATURES_V2) == 15

    def test_v2_subset_of_v1(self):
        from tb_risk.scoring.predictor import MLRiskPredictor
        assert set(SELECTED_FEATURES_V2) <= set(MLRiskPredictor.ALL_FEATURE_NAMES)

    def test_user_named_merges_applied(self):
        # exposure_accumulation 并入 cumulative_exposure（语义重复，用户点名）
        assert 'cumulative_exposure' in SELECTED_FEATURES_V2
        assert 'exposure_accumulation' not in SELECTED_FEATURES_V2
        # 暴露时序保留临床含义最清晰的组合：累积剂量 + 时间窗
        assert 'time_span' in SELECTED_FEATURES_V2
        assert 'single_duration' not in SELECTED_FEATURES_V2
        assert 'freq_density' not in SELECTED_FEATURES_V2

    def test_removal_rationale_covers_all_removed(self):
        from tb_risk.scoring.predictor import MLRiskPredictor
        removed = set(MLRiskPredictor.ALL_FEATURE_NAMES) - set(SELECTED_FEATURES_V2)
        assert set(FEATURE_REMOVAL_RATIONALE.keys()) == removed
        for feature, reason in FEATURE_REMOVAL_RATIONALE.items():
            assert isinstance(reason, str) and len(reason) > 10

    def test_keeps_core_clinical_dimensions(self):
        """每个保留特征对应独立临床维度（人口学/剂量/症状/免疫/环境）。"""
        for feat in ('age', 'has_symptoms', 'bcg_vaccine', 'has_tb',
                     'contact_distance_score', 'ventilation_score',
                     'is_high_risk', 'past_illness',
                     'exposure_setting_score', 'age_immuno',
                     'dm_tb_synergy', 'age_diabetes'):
            assert feat in SELECTED_FEATURES_V2, f'{feat} 应保留'
