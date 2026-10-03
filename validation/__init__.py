from .backtest import BacktestEngine
from .sensitivity import MorrisSensitivityAnalyzer
from .ablation import AblationAnalyzer
from .comparator import NationalModelComparator
from .validator import KaramayValidator
from .historical_calibration import HistoricalCalibration
from .threshold_spec import (
    RISK_GRADES,
    THRESHOLD_SPEC_TABLE,
    DEFAULT_VERY_HIGH,
    DEFAULT_HIGH,
    DEFAULT_MEDIUM,
    grade_risk,
    get_threshold_spec,
    build_threshold_spec_report,
    youden_optimal_threshold,
    refit_risk_thresholds,
    clinical_grade_from_record,
    ClinicalGradeComparator,
    compare_model_vs_clinical,
    # 近期进展分层区分度验证（双输出 vs 固定基线）
    IGRA_STRATA_KEYS,
    compute_auc,
    auc_standard_error,
    igra_stratified_discrimination,
    compare_individualized_vs_fixed_baseline,
)
from .causal import (
    CausalDAG,
    NodeType,
    build_tb_dag,
    get_default_adjustment_set,
    PropensityScoreMatcher,
    CounterfactualAnalyzer,
    DoCalculusEstimator,
)
from .missingness import (
    generate_augmented_contact_dataset,
    impute_contact_dataset,
    analyze_missingness,
    find_boundary_missing_rate,
    DEFAULT_SENSITIVITY_FIELDS,
    DEFAULT_MISSING_RATES,
)
# 三层递进架构 · 网络增强层消融（有/无 GNN 层 ΔAUROC / ΔC-index）
from .layer_ablation import (
    compute_c_index,
    build_synthetic_network,
    LayerAblationAnalyzer,
    network_contribution_ablation,
)
# 任务分解与临床终点定义（横断面筛查 A / 纵向进展 B / 网络传播 C）
from .task_decomposition import (
    TaskType,
    EndPoint,
    TASK_ENDPOINT_MAP,
    evaluate_task_a,
    evaluate_task_b,
    evaluate_task_c,
    assign_task_labels,
    evaluate_task,
)
# 改进三：可复现 DGP 文档 + 分布对照校验（合成数据规范化）
from .dgp import (
    DEFAULT_RANDOM_STATE,
    DGP_PARAMETERS,
    FIELD_SPECS,
    TRANSMISSION_SPEC,
    LITERATURE_REFERENCES,
    field_specs_by_name,
    validate_synthetic_vs_spec,
    validate_prevalence,
    validate_against_literature,
    reproducibility_check,
    build_dgp_report,
    render_dgp_document,
)
# 改进三：公开数据集目录按任务匹配
from .public_datasets import (
    LAYER_INDIVIDUAL,
    LAYER_CONTACT,
    LAYER_TRANSMISSION,
    PUBLIC_DATASETS,
    list_public_datasets,
    get_public_dataset,
    dataset_coverage_by_task,
    recommended_access_order,
    summarize_public_datasets,
)
# 改进三：公开数据验证策略（A/B 外部验证 / C 过渡验证 + 合作数据需求）
from .public_data_strategy import (
    EXTERNAL_VALIDATION_PLAN,
    COOPERATION_DATA_NEEDS,
    transition_validation_steps,
    cooperation_data_needs,
    top_cooperation_data_need,
    build_validation_strategy,
    render_strategy_document,
)

__all__ = [
    'BacktestEngine',
    'MorrisSensitivityAnalyzer',
    'AblationAnalyzer',
    'NationalModelComparator',
    'KaramayValidator',
    'HistoricalCalibration',
    'RISK_GRADES',
    'THRESHOLD_SPEC_TABLE',
    'DEFAULT_VERY_HIGH',
    'DEFAULT_HIGH',
    'DEFAULT_MEDIUM',
    'grade_risk',
    'get_threshold_spec',
    'build_threshold_spec_report',
    'youden_optimal_threshold',
    'refit_risk_thresholds',
    'clinical_grade_from_record',
    'ClinicalGradeComparator',
    'compare_model_vs_clinical',
    'IGRA_STRATA_KEYS',
    'compute_auc',
    'auc_standard_error',
    'igra_stratified_discrimination',
    'compare_individualized_vs_fixed_baseline',
    'CausalDAG',
    'NodeType',
    'build_tb_dag',
    'get_default_adjustment_set',
    'PropensityScoreMatcher',
    'CounterfactualAnalyzer',
    'DoCalculusEstimator',
    # data augmentation + imputation + missingness sensitivity
    'generate_augmented_contact_dataset',
    'impute_contact_dataset',
    'analyze_missingness',
    'find_boundary_missing_rate',
    'DEFAULT_SENSITIVITY_FIELDS',
    'DEFAULT_MISSING_RATES',
    # 网络增强层消融
    'compute_c_index',
    'build_synthetic_network',
    'LayerAblationAnalyzer',
    'network_contribution_ablation',
    # 任务分解与临床终点
    'TaskType',
    'EndPoint',
    'TASK_ENDPOINT_MAP',
    'evaluate_task_a',
    'evaluate_task_b',
    'evaluate_task_c',
    'assign_task_labels',
    'evaluate_task',
    # 改进三：可复现 DGP 文档 + 分布对照校验
    'DEFAULT_RANDOM_STATE',
    'DGP_PARAMETERS',
    'FIELD_SPECS',
    'TRANSMISSION_SPEC',
    'LITERATURE_REFERENCES',
    'field_specs_by_name',
    'validate_synthetic_vs_spec',
    'validate_prevalence',
    'validate_against_literature',
    'reproducibility_check',
    'build_dgp_report',
    'render_dgp_document',
    # 改进三：公开数据集目录
    'LAYER_INDIVIDUAL',
    'LAYER_CONTACT',
    'LAYER_TRANSMISSION',
    'PUBLIC_DATASETS',
    'list_public_datasets',
    'get_public_dataset',
    'dataset_coverage_by_task',
    'recommended_access_order',
    'summarize_public_datasets',
    # 改进三：公开数据验证策略
    'EXTERNAL_VALIDATION_PLAN',
    'COOPERATION_DATA_NEEDS',
    'transition_validation_steps',
    'cooperation_data_needs',
    'top_cooperation_data_need',
    'build_validation_strategy',
    'render_strategy_document',
]