"""seir 子包：SEIR 传播动力学与贝叶斯推断公共门面。

问题三（统一贝叶斯框架）：本模块作为 SEIR 相关功能的唯一公共入口，
统一各消费者的导入路径。推荐用法：

    from tb_risk.seir import get_inference_engine

    Engine = get_inference_engine('v4')  # 返回 BayesianInferenceV4 类
    engine = Engine(seir_model)

避免直接从 seir.uncertainty 或 seir.inference._base 等内部模块导入。
"""

# 公共门面：贝叶斯推断引擎工厂
from .inference import get_inference_engine

# 常用类再导出（方便 `from tb_risk.seir import BayesianSEIRInference`）
from .bayesian import BayesianSEIRInference
from .posterior import PosteriorDrivenInfectivity

# 理论拓展子包（超级传播建模 + 四维综合对比）
from .extensions import (
    # superspreading
    DEFAULT_R0, DEFAULT_K, DEFAULT_TOP_FRACTION, DEFAULT_SHARE,
    nb_log_pmf, nb_pmf, nb_cdf, sample_offspring,
    offspring_pmf_array, prob_no_transmission,
    theoretical_dispersion_from_share,
    transmission_share_of_top_fraction,
    top_fraction_for_transmission_share,
    extinction_probability, fit_dispersion_k,
    superspreading_metrics, SuperspreaderSEIR,
    compare_homogeneous_vs_heterogeneous,
    # synthesis
    AGE_LABELS, age_structured_analysis, hiv_analysis, dr_analysis,
    build_theory_report,
)

# 社区干预反事实模拟（三层递进架构 · 第 3 层：SEIR 只做干预反事实）
from .intervention import (
    InterventionCounterfactualSimulator,
    simulate_intervention_effects,
)

__all__ = [
    'get_inference_engine',
    'BayesianSEIRInference',
    'PosteriorDrivenInfectivity',
    # super spreading
    'DEFAULT_R0', 'DEFAULT_K', 'DEFAULT_TOP_FRACTION', 'DEFAULT_SHARE',
    'nb_log_pmf', 'nb_pmf', 'nb_cdf', 'sample_offspring',
    'offspring_pmf_array', 'prob_no_transmission',
    'theoretical_dispersion_from_share',
    'transmission_share_of_top_fraction',
    'top_fraction_for_transmission_share',
    'extinction_probability', 'fit_dispersion_k',
    'superspreading_metrics', 'SuperspreaderSEIR',
    'compare_homogeneous_vs_heterogeneous',
    # synthesis
    'AGE_LABELS', 'age_structured_analysis', 'hiv_analysis', 'dr_analysis',
    'build_theory_report',
    # 社区干预反事实模拟（第 3 层）
    'InterventionCounterfactualSimulator',
    'simulate_intervention_effects',
]
