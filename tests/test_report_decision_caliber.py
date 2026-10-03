#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题4（输出侧·训练报告）：决策口径段（排序+截断点+双口径 PPV）

_generate_markdown_report 必须包含：
1. "决策口径"独立章节；
2. 阳性率双口径表（密接 2.85% / 全人群 100/10万，来自单一真值源）；
3. "不可混用"警示；
4. 各模型 decision_reference（若有）的排序+截断点摘要。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tb_risk.export_utils.report_export import _generate_markdown_report  # noqa: E402
from tb_risk.core.ppv import (  # noqa: E402
    build_decision_reference, format_decision_reference_summary)


class _ReportStubPredictor:
    """满足 _generate_markdown_report 协议的最小桩。"""

    FEATURE_NAMES = ['age', 'cumulative_exposure']
    INTERACTION_FEATURE_NAMES = ['age_diabetes']
    ALL_FEATURE_NAMES = FEATURE_NAMES + INTERACTION_FEATURE_NAMES
    FEATURE_DESCRIPTIONS = {'age': '年龄', 'cumulative_exposure': '累积暴露'}
    INTERACTION_FEATURE_DESCRIPTIONS = {'age_diabetes': '年龄×糖尿病'}

    def __init__(self, with_decision_reference=True):
        self.is_trained = True
        self.training_sample_count = 100
        self.use_real_data = True
        self.shap_explainer = None
        self.last_shap_values = None
        self.model_performance = {
            'random_forest': {
                'name': '随机森林', 'AUROC': 0.85, 'AUPRC': 0.4,
                'calibration': {
                    'method': 'sigmoid', 'n_positive': 90,
                    'brier_before': 0.12, 'brier_after': 0.10,
                    'auroc_before': 0.85, 'auroc_after': 0.84,
                },
            },
        }
        if with_decision_reference:
            scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
            labels = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
            self.model_performance['random_forest'][
                'decision_reference'] = build_decision_reference(
                scores, labels, top_fraction=0.2)

    def compute_bootstrap_ci(self, n_bootstrap=50):
        return {}

    def compute_calibration(self, n_bins=10, random_state=42):
        return {}


# ============================================================================
# 决策口径段
# ============================================================================

class TestDecisionCaliberSection:

    def test_report_has_decision_caliber_section(self):
        md = _generate_markdown_report(_ReportStubPredictor())
        assert '决策口径' in md

    def test_report_lists_both_calibers(self):
        md = _generate_markdown_report(_ReportStubPredictor())
        assert '2.85%' in md
        assert '100/10万' in md
        assert '密接' in md
        assert '全人群' in md

    def test_report_carries_mixing_warning(self):
        md = _generate_markdown_report(_ReportStubPredictor())
        assert '不可混用' in md

    def test_report_includes_ranking_decision_form(self):
        md = _generate_markdown_report(_ReportStubPredictor())
        assert '排序' in md
        # 决策参考摘要格式（截断点信息）
        assert '截断' in md or '覆盖' in md

    def test_report_without_decision_reference_still_has_calibers(self):
        md = _generate_markdown_report(
            _ReportStubPredictor(with_decision_reference=False))
        assert '2.85%' in md
        assert '100/10万' in md
        assert '不可混用' in md

    def test_real_data_calibration_metrics_rendered(self):
        """真实数据校准的 Brier/AUROC 前后对照进报告（非合成数据校准）。"""
        md = _generate_markdown_report(_ReportStubPredictor())
        assert 'Brier' in md
        assert '校准前' in md or 'before' in md.lower()

    def test_summary_helper_matches_report_fragment(self):
        pred = _ReportStubPredictor()
        ref = pred.model_performance['random_forest']['decision_reference']
        summary = format_decision_reference_summary(ref)
        md = _generate_markdown_report(pred)
        assert summary in md
