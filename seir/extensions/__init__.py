#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEIR 模型理论拓展子包。

把结核病传播建模的前沿拓展整合为可发表的形式：

- ``superspreading``：超级传播者建模（负二项子代分布、k 拟合、20/80 规则、
  随机熄灭概率、超传播者 SEIR 模型）。这是相对于 v4（年龄×HIV×耐药 270D）
  的**新增**维度。
- ``synthesis``：四维综合对比（年龄 / 超级传播 / TB-HIV / 耐药），复用 v4
  内置的三维分层，输出可直接入论文的结构化报告。

说明：年龄结构化、TB-HIV 共感染、耐药 TB 三个维度已由
``tb_risk.seir.stochastic.StochasticSEIRModel``（v4，270D）内置实现；
本包聚焦于把超级传播这一缺失维度补齐，并将四维统一为理论拓展框架。
"""

from .superspreading import (
    DEFAULT_R0, DEFAULT_K,
    DEFAULT_TOP_FRACTION, DEFAULT_SHARE,
    DEFAULT_LATENT_PERIOD, DEFAULT_INFECTIOUS_PERIOD,
    DEFAULT_P_SUPERSPREADER, DEFAULT_REL_INFECTIOUSNESS,
    nb_log_pmf, nb_pmf, nb_cdf, sample_offspring,
    offspring_pmf_array, prob_no_transmission,
    theoretical_dispersion_from_share,
    transmission_share_of_top_fraction,
    top_fraction_for_transmission_share,
    extinction_probability, fit_dispersion_k,
    superspreading_metrics, SuperspreaderSEIR,
    compare_homogeneous_vs_heterogeneous,
)
from .synthesis import (
    AGE_LABELS,
    age_structured_analysis,
    hiv_analysis,
    dr_analysis,
    build_theory_report,
)

__all__ = [
    # superspreading
    "DEFAULT_R0", "DEFAULT_K", "DEFAULT_TOP_FRACTION", "DEFAULT_SHARE",
    "DEFAULT_LATENT_PERIOD", "DEFAULT_INFECTIOUS_PERIOD",
    "DEFAULT_P_SUPERSPREADER", "DEFAULT_REL_INFECTIOUSNESS",
    "nb_log_pmf", "nb_pmf", "nb_cdf", "sample_offspring",
    "offspring_pmf_array", "prob_no_transmission",
    "theoretical_dispersion_from_share",
    "transmission_share_of_top_fraction",
    "top_fraction_for_transmission_share",
    "extinction_probability", "fit_dispersion_k",
    "superspreading_metrics", "SuperspreaderSEIR",
    "compare_homogeneous_vs_heterogeneous",
    # synthesis
    "AGE_LABELS",
    "age_structured_analysis",
    "hiv_analysis",
    "dr_analysis",
    "build_theory_report",
]