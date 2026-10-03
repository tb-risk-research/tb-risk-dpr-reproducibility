#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML风险预测器子模块 — 按职责分离的独立模块

每个子模块包含无状态函数，通过参数传递数据。
MLRiskPredictor 主类保留编排逻辑，委托给子模块执行。
"""

from .training import (
    SKLEARN_AVAILABLE,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    CATBOOST_AVAILABLE,
    SHAP_AVAILABLE,
)

from .preprocessing import (
    extract_features,
    extract_features_with_interactions,
    _generate_synthetic_training_data,
    _extract_features_from_records,
    set_patient_context,
)

from .training import (
    train_models,
    _train_models_from_arrays,
    train_from_arrays,
    train_from_real_data,
    _train_models_for_mode,
    _train_with_fallback,
    _interruptible_grid_search,
)

from .evaluation import (
    compare_real_vs_synthetic,
    _evaluate_training_mode,
    _bootstrap_mode_comparison,
    _wilcoxon_mode_comparison,
    _ensemble_predict,
    _safe_roc_auc_score,
    _safe_average_precision_score,
    _wilcoxon_signed_rank,
    _generate_comparison_recommendation,
    compute_bootstrap_ci,
    _compute_gnn_metrics,
    predict_risk,
    _normalize_illness_type,
)

from .calibration import (
    compute_calibration,
    calibrate_models,
    train_stacking_ensemble,
)

from .shap_analysis import compute_shap_values, compute_contact_attributions

from .validation import (
    build_validation_report,
    split_train_val_test,
    temporal_split,
    compute_discrimination_metrics,
    compute_calibration_metrics,
    assess_calibration,
)

from .persistence import (
    save_model,
    load_model,
    save_gnn_model,
    load_gnn_model,
    _validate_gnn_checkpoint,
)

from .transfer_learning import (
    WHO_TB_BURDEN_CATALOG,
    DEFAULT_PRETRAIN_REGION,
    TRANSFER_STAGE_UNTRAINED,
    TRANSFER_STAGE_PRETRAINED,
    TRANSFER_STAGE_FINETUNED,
    list_pretrain_regions,
    generate_pretraining_dataset,
    pretrain_models,
    finetune_models,
    compare_finetune_vs_from_scratch,
    summarize_transfer_status,
)

from .gnn_training import (
    train_gnn,
    train_stgnn,
    predict_gnn_risk,
    predict_time_aware_gnn_risk,
    predict_gnn_increment,
    compute_gnn_explanation,
    _generate_synthetic_graphs,
    _focal_loss,
    _train_gnn_once,
    _evaluate_gnn,
)

from .graph_conversion import (
    _convert_hetero_to_homo,
    _find_target_node_idx,
)

from .survival import train_survival_multistate

from .epidemiology import (
    calculate_treatment_infectivity_factor,
    calculate_multilayer_network_r0,
)

# 大规模图网络处理（gnn_scalable 子包全量导出）
try:
    from .gnn_scalable import (
        PYTORCH_AVAILABLE as GNN_SCALABLE_PYTORCH_AVAILABLE,
        PYG_AVAILABLE as GNN_SCALABLE_PYG_AVAILABLE,
        NEIGHBOR_LOADER_AVAILABLE as GNN_NEIGHBOR_LOADER_AVAILABLE,
        DYNAMIC_QUANT_AVAILABLE as GNN_DYNAMIC_QUANT_AVAILABLE,
        sample_k_hop_subgraph,
        manual_k_hop_subgraph,
        build_neighbor_sampler,
        NeighborBatch,
        MinibatchConfig,
        train_gnn_minibatch,
        predict_target_k_hop,
        predict_subgraph,
        quantize_model_fp16,
        quantize_model_int8,
        predict_quantized,
        measure_inference_latency,
        GNNStudentMLP,
        distill_gnn_to_mlp,
        predict_with_mlp,
        add_new_node,
        incremental_predict_new_node,
    )
    GNN_SCALABLE_AVAILABLE = True
except ImportError:  # pragma: no cover - 依赖缺失时仅符号为 None
    GNN_SCALABLE_AVAILABLE = False
    sample_k_hop_subgraph = None
    manual_k_hop_subgraph = None
    build_neighbor_sampler = None
    NeighborBatch = None
    MinibatchConfig = None
    train_gnn_minibatch = None
    predict_target_k_hop = None
    predict_subgraph = None
    quantize_model_fp16 = None
    quantize_model_int8 = None
    predict_quantized = None
    measure_inference_latency = None
    GNNStudentMLP = None
    distill_gnn_to_mlp = None
    predict_with_mlp = None
    add_new_node = None
    incremental_predict_new_node = None

__all__ = [
    # 可用性标志
    'SKLEARN_AVAILABLE',
    'XGBOOST_AVAILABLE',
    'LIGHTGBM_AVAILABLE',
    'CATBOOST_AVAILABLE',
    'SHAP_AVAILABLE',
    # preprocessing
    'extract_features',
    'extract_features_with_interactions',
    '_generate_synthetic_training_data',
    '_extract_features_from_records',
    'set_patient_context',
    # training
    'train_models',
    '_train_models_from_arrays',
    'train_from_arrays',
    'train_from_real_data',
    '_train_models_for_mode',
    '_train_with_fallback',
    '_interruptible_grid_search',
    # evaluation
    'compare_real_vs_synthetic',
    '_evaluate_training_mode',
    '_bootstrap_mode_comparison',
    '_wilcoxon_mode_comparison',
    '_ensemble_predict',
    '_safe_roc_auc_score',
    '_safe_average_precision_score',
    '_wilcoxon_signed_rank',
    '_generate_comparison_recommendation',
    'compute_bootstrap_ci',
    '_compute_gnn_metrics',
    'predict_risk',
    '_normalize_illness_type',
    # calibration
    'compute_calibration',
    'calibrate_models',
    'train_stacking_ensemble',
    # shap
    'compute_shap_values',
    'compute_contact_attributions',
    # validation
    'build_validation_report',
    'split_train_val_test',
    'temporal_split',
    'compute_discrimination_metrics',
    'compute_calibration_metrics',
    'assess_calibration',
    # persistence
    'save_model',
    'load_model',
    'save_gnn_model',
    'load_gnn_model',
    '_validate_gnn_checkpoint',
    # transfer learning（迁移学习与预训练）
    'WHO_TB_BURDEN_CATALOG',
    'DEFAULT_PRETRAIN_REGION',
    'TRANSFER_STAGE_UNTRAINED',
    'TRANSFER_STAGE_PRETRAINED',
    'TRANSFER_STAGE_FINETUNED',
    'list_pretrain_regions',
    'generate_pretraining_dataset',
    'pretrain_models',
    'finetune_models',
    'compare_finetune_vs_from_scratch',
    'summarize_transfer_status',
    # gnn
    'train_gnn',
    'train_stgnn',
    'predict_gnn_risk',
    'predict_time_aware_gnn_risk',
    'predict_gnn_increment',
    'compute_gnn_explanation',
    '_generate_synthetic_graphs',
    '_focal_loss',
    '_train_gnn_once',
    '_evaluate_gnn',
    # graph
    '_convert_hetero_to_homo',
    '_find_target_node_idx',
    # survival
    'train_survival_multistate',
    # epidemiology
    'calculate_treatment_infectivity_factor',
    'calculate_multilayer_network_r0',
    # gnn_scalable（大规模图网络处理）
    'GNN_SCALABLE_AVAILABLE',
    'GNN_SCALABLE_PYTORCH_AVAILABLE',
    'GNN_SCALABLE_PYG_AVAILABLE',
    'GNN_NEIGHBOR_LOADER_AVAILABLE',
    'GNN_DYNAMIC_QUANT_AVAILABLE',
    'sample_k_hop_subgraph',
    'manual_k_hop_subgraph',
    'build_neighbor_sampler',
    'NeighborBatch',
    'MinibatchConfig',
    'train_gnn_minibatch',
    'predict_target_k_hop',
    'predict_subgraph',
    'quantize_model_fp16',
    'quantize_model_int8',
    'predict_quantized',
    'measure_inference_latency',
    'GNNStudentMLP',
    'distill_gnn_to_mlp',
    'predict_with_mlp',
    'add_new_node',
    'incremental_predict_new_node',
]