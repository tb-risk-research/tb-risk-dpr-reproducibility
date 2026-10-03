"""
机器学习风险分层预测器

基于文献支撑：
- 东开普省研究（IJERPH, 2025, 22(12), 1823）：梯度提升AUROC=0.65, AUPRC=0.76
- 新疆研究（Research Square, 2025）：风险诺模图AUC=0.839
- GBM模型（BMC Infect Dis, 2025, 25, 1512）：AUC=0.831, 特异性85.5%

使用随机森林、梯度提升、XGBoost、LightGBM和CatBoost五种异构树模型
替代sigmoid评分函数，并集成SHAP值分析增强模型可解释性。
"""

import copy
import logging
import os
import random

import numpy as np

LOGGER = logging.getLogger("tb_risk.ml")

# --- 条件导入：scikit-learn ---
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 accuracy_score, precision_score, recall_score,
                                 f1_score, brier_score_loss)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- 条件导入：XGBoost ---
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

# --- 条件导入：LightGBM ---
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

# --- 条件导入：CatBoost ---
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

# --- 条件导入：SHAP ---
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# --- 条件导入：joblib ---
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# --- 条件导入：pandas ---
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# --- 条件导入：PyTorch ---
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# --- 条件导入：PyTorch Geometric ---
try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

# --- 内部模块条件导入 ---
# 注意：ml/framework.py 中未定义 PIGNN 类，因此不导入它，
# 以免 ImportError 连带把 HeterogeneousTBNetwork/SEIRInformedGNN/SEIRTimeAwareGNN 也置为 None。
try:
    from ..ml.framework import HeterogeneousTBNetwork, SEIRInformedGNN, SEIRTimeAwareGNN
except ImportError:
    HeterogeneousTBNetwork = None
    SEIRInformedGNN = None
    SEIRTimeAwareGNN = None

# PIGNN 类尚未实现，标记为不可用（与 gui/mixins.py 保持一致）
PIGNN = None
PIGNN_AVAILABLE = False

# 生存分析模块（ml/survival.py 不存在，设为 None 待后续实现）
SurvivalPredictor = None
MultiStateSurvivalPredictor = None
SURVIVAL_BAYESIAN_AVAILABLE = False
MULTISTATE_AVAILABLE = False

try:
    from ..utils import SyntheticDataCalibrator, expected_feature_means_from_config
except ImportError:
    SyntheticDataCalibrator = None
    expected_feature_means_from_config = None
    LOGGER.warning("SyntheticDataCalibrator 导入失败，合成数据校准不可用")

NUMPY_AVAILABLE = True


def _log_ml_model_availability():
    """输出ML模型依赖可用性提示，告知用户哪些可选模型可启用"""
    available = []
    unavailable = []
    checks = [
        ("随机森林 (sklearn)", SKLEARN_AVAILABLE, "scikit-learn"),
        ("XGBoost", XGBOOST_AVAILABLE, "xgboost"),
        ("LightGBM", LIGHTGBM_AVAILABLE, "lightgbm"),
        ("CatBoost", CATBOOST_AVAILABLE, "catboost"),
    ]
    for name, ok, pkg in checks:
        if ok:
            available.append(name)
        else:
            unavailable.append(f"{name} (pip install {pkg} 可启用)")
    if available:
        LOGGER.info("ML预测器已加载模型: %s", ", ".join(available))
    if unavailable:
        LOGGER.info("以下ML模型因依赖未安装暂不可用（不影响其他模型）: %s",
                    "; ".join(unavailable))


_log_ml_model_availability()

# 从 utils 统一导入 _is_yes，提供fallback确保模块可加载
try:
    from ..utils import _is_yes
except ImportError:
    def _is_yes(val) -> bool:
        """fallback实现：判断是否为truthy值。"""
        if val is None:
            return False
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val != 0
        if isinstance(val, str):
            return val.strip().lower() in ("yes", "y", "true", "1", "是", "有", "阳性")
        return bool(val)
    LOGGER.debug("_is_yes 相对导入失败，使用本地fallback实现")

# 问题八-1：评分映射表单一真值源 — 从 constants.py 导入，消除字面量副本。
# 原 DISTANCE/VENTILATION/SETTING_SCORE_MAP 是手动同步的 ScoringEngine 副本，
# 无导入关系，公式改动需同步两处。现统一引用 constants.DISTANCE_FACTORS 等。
try:
    from ..constants import (
        DISTANCE_FACTORS as _DISTANCE_SCORE_MAP_SRC,
        VENTILATION_FACTORS as _VENTILATION_SCORE_MAP_SRC,
        SETTING_FACTORS_BASE as _SETTING_SCORE_MAP_SRC,
    )
except ImportError:
    # fallback默认值，保证模块可加载
    _DISTANCE_SCORE_MAP_SRC = {"close": 1.0, "medium": 0.7, "far": 0.3}
    _VENTILATION_SCORE_MAP_SRC = {"poor": 1.0, "average": 0.7, "good": 0.4}
    _SETTING_SCORE_MAP_SRC = {"hospital": 1.0, "clinic": 0.7, "home": 0.5, "public": 0.4}
    LOGGER.debug("constants 相对导入失败，使用默认评分因子fallback")



# 子模块导入（委托给 scoring/ml/ 子包）
# 使用 _impl 别名避免与实例方法名冲突
# 使用 try-except 保护，确保模块在任何导入路径下都能加载
_ML_SUBMODULES_AVAILABLE = False
_impl_extract_features = None
_impl_extract_features_with_interactions = None
_impl_generate_synthetic_training_data = None
_impl_extract_features_from_records = None
_impl_set_patient_context = None
_impl_train_models = None
_impl_train_models_from_arrays = None
_impl_train_from_arrays = None
_impl_train_from_real_data = None
_impl_train_models_for_mode = None
_impl_train_with_fallback = None
_impl_interruptible_grid_search = None
_impl_compare_real_vs_synthetic = None
_impl_evaluate_training_mode = None
_impl_bootstrap_mode_comparison = None
_impl_wilcoxon_mode_comparison = None
_impl_ensemble_predict = None
_impl_safe_roc_auc_score = None
_impl_safe_average_precision_score = None
_impl_wilcoxon_signed_rank = None
_impl_generate_comparison_recommendation = None
_impl_compute_bootstrap_ci = None
_impl_compute_gnn_metrics = None
_impl_predict_risk = None
_impl_normalize_illness_type = None
_impl_compute_calibration = None
_impl_calibrate_models = None
_impl_train_stacking_ensemble = None
_impl_compute_shap_values = None
_impl_compute_contact_attributions = None
_impl_build_validation_report = None
_impl_save_model = None
_impl_load_model = None
_impl_save_gnn_model = None
_impl_load_gnn_model = None
_impl_validate_gnn_checkpoint = None
_impl_pretrain_models = None
_impl_finetune_models = None
_impl_compare_finetune_vs_from_scratch = None
_impl_summarize_transfer_status = None
_impl_generate_pretraining_dataset = None
_impl_train_gnn = None
_impl_train_pignn = None
_impl_train_stgnn = None
_impl_predict_gnn_risk = None
_impl_predict_gnn_increment = None
_impl_compute_gnn_explanation = None
_impl_predict_time_aware_gnn_risk = None
_impl_generate_synthetic_graphs = None
_impl_focal_loss = None
_impl_train_gnn_once = None
_impl_evaluate_gnn = None
_impl_convert_hetero_to_homo = None
_impl_find_target_node_idx = None
_impl_train_survival_multistate = None
_impl_calculate_treatment_infectivity_factor = None
_impl_calculate_multilayer_network_r0 = None

try:
    from .ml.preprocessing import (
        extract_features as _impl_extract_features,
        extract_features_with_interactions as _impl_extract_features_with_interactions,
        _generate_synthetic_training_data as _impl_generate_synthetic_training_data,
        _extract_features_from_records as _impl_extract_features_from_records,
        set_patient_context as _impl_set_patient_context,
    )
    from .ml.training import (
        train_models as _impl_train_models,
        _train_models_from_arrays as _impl_train_models_from_arrays,
        train_from_arrays as _impl_train_from_arrays,
        train_from_real_data as _impl_train_from_real_data,
        _train_models_for_mode as _impl_train_models_for_mode,
        _train_with_fallback as _impl_train_with_fallback,
        _interruptible_grid_search as _impl_interruptible_grid_search,
    )
    from .ml.evaluation import (
        compare_real_vs_synthetic as _impl_compare_real_vs_synthetic,
        _evaluate_training_mode as _impl_evaluate_training_mode,
        _bootstrap_mode_comparison as _impl_bootstrap_mode_comparison,
        _wilcoxon_mode_comparison as _impl_wilcoxon_mode_comparison,
        _ensemble_predict as _impl_ensemble_predict,
        _safe_roc_auc_score as _impl_safe_roc_auc_score,
        _safe_average_precision_score as _impl_safe_average_precision_score,
        _wilcoxon_signed_rank as _impl_wilcoxon_signed_rank,
        _generate_comparison_recommendation as _impl_generate_comparison_recommendation,
        compute_bootstrap_ci as _impl_compute_bootstrap_ci,
        _compute_gnn_metrics as _impl_compute_gnn_metrics,
        predict_risk as _impl_predict_risk,
        _normalize_illness_type as _impl_normalize_illness_type,
    )
    from .ml.calibration import (
        compute_calibration as _impl_compute_calibration,
        calibrate_models as _impl_calibrate_models,
        train_stacking_ensemble as _impl_train_stacking_ensemble,
    )
    from .ml.shap_analysis import compute_shap_values as _impl_compute_shap_values
    from .ml.shap_analysis import compute_contact_attributions as _impl_compute_contact_attributions
    from .ml.validation import build_validation_report as _impl_build_validation_report
    from .ml.persistence import (
        save_model as _impl_save_model,
        load_model as _impl_load_model,
        save_gnn_model as _impl_save_gnn_model,
        load_gnn_model as _impl_load_gnn_model,
        _validate_gnn_checkpoint as _impl_validate_gnn_checkpoint,
    )
    from .ml.transfer_learning import (
        pretrain_models as _impl_pretrain_models,
        finetune_models as _impl_finetune_models,
        compare_finetune_vs_from_scratch as _impl_compare_finetune_vs_from_scratch,
        summarize_transfer_status as _impl_summarize_transfer_status,
        generate_pretraining_dataset as _impl_generate_pretraining_dataset,
    )
    from .ml.gnn_training import (
        train_gnn as _impl_train_gnn,
        train_pignn as _impl_train_pignn,
        train_stgnn as _impl_train_stgnn,
        predict_gnn_risk as _impl_predict_gnn_risk,
        predict_time_aware_gnn_risk as _impl_predict_time_aware_gnn_risk,
        predict_gnn_increment as _impl_predict_gnn_increment,
        compute_gnn_explanation as _impl_compute_gnn_explanation,
        _generate_synthetic_graphs as _impl_generate_synthetic_graphs,
        _focal_loss as _impl_focal_loss,
        _train_gnn_once as _impl_train_gnn_once,
        _evaluate_gnn as _impl_evaluate_gnn,
    )
    from .ml.graph_conversion import (
        _convert_hetero_to_homo as _impl_convert_hetero_to_homo,
        _find_target_node_idx as _impl_find_target_node_idx,
    )
    from .ml.survival import train_survival_multistate as _impl_train_survival_multistate
    from .ml.epidemiology import (
        calculate_treatment_infectivity_factor as _impl_calculate_treatment_infectivity_factor,
        calculate_multilayer_network_r0 as _impl_calculate_multilayer_network_r0,
    )
    _ML_SUBMODULES_AVAILABLE = True
except ImportError as e:
    _ML_SUBMODULES_AVAILABLE = False
    LOGGER.debug("scoring.ml 子模块导入失败，ML训练/评估功能降级不可用: %s", e)


# --- 大规模图网络处理（gnn_scalable）委托符号 ---
# 这些是独立函数（不依赖 predictor 实例），此处仅做符号绑定供委托方法调用。
try:
    from .ml.gnn_scalable import (
        MinibatchConfig as _scalable_MinibatchConfig,
        train_gnn_minibatch as _scalable_train_gnn_minibatch,
        predict_target_k_hop as _scalable_predict_target_k_hop,
        predict_subgraph as _scalable_predict_subgraph,
        sample_k_hop_subgraph as _scalable_sample_k_hop_subgraph,
        quantize_model_fp16 as _scalable_quantize_model_fp16,
        quantize_model_int8 as _scalable_quantize_model_int8,
        predict_quantized as _scalable_predict_quantized,
        measure_inference_latency as _scalable_measure_inference_latency,
        distill_gnn_to_mlp as _scalable_distill_gnn_to_mlp,
        predict_with_mlp as _scalable_predict_with_mlp,
        add_new_node as _scalable_add_new_node,
        incremental_predict_new_node as _scalable_incremental_predict_new_node,
    )
    _GNN_SCALABLE_AVAILABLE = True
except ImportError:
    _GNN_SCALABLE_AVAILABLE = False
    _scalable_MinibatchConfig = None
    _scalable_train_gnn_minibatch = None
    _scalable_predict_target_k_hop = None
    _scalable_predict_subgraph = None
    _scalable_sample_k_hop_subgraph = None
    _scalable_quantize_model_fp16 = None
    _scalable_quantize_model_int8 = None
    _scalable_predict_quantized = None
    _scalable_measure_inference_latency = None
    _scalable_distill_gnn_to_mlp = None
    _scalable_predict_with_mlp = None
    _scalable_add_new_node = None
    _scalable_incremental_predict_new_node = None


class MLRiskPredictor:
    """机器学习风险分层预测器

    巨文件拆分后，所有方法委托给 scoring/ml/ 子包中的独立函数。
    保持公开API不变，内部实现对外部调用者透明。
    """

    FEATURE_NAMES = [
        'age', 'cumulative_exposure', 'has_symptoms', 'bcg_vaccine',
        'has_tb', 'contact_distance_score', 'ventilation_score',
        'is_high_risk', 'past_illness', 'exposure_setting_score',
        'single_duration', 'freq_density', 'time_span'
    ]
    
    FEATURE_DESCRIPTIONS = {
        'age': '年龄',
        'cumulative_exposure': '累积暴露时长(小时)',
        'has_symptoms': '有症状(0/1)',
        'bcg_vaccine': '卡介苗接种(0/1)',
        'has_tb': '既往结核史(0/1)',
        'contact_distance_score': '接触距离评分',
        'ventilation_score': '通风条件评分',
        'is_high_risk': '高危人群(0/1)',
        'past_illness': '慢性病史(0/1)',
        'exposure_setting_score': '暴露场景评分',
        'single_duration': '单次接触时长(分钟)',
        'freq_density': '每周接触频次',
        'time_span': '持续周期(周)'
    }
    
    INTERACTION_FEATURE_NAMES = [
        'age_immuno', 'dm_tb_synergy', 'age_bcg_decay', 'symptom_delay',
        'cough_contact', 'highrisk_comorbid', 'immune_bcg', 'exposure_accumulation', 'age_diabetes'
    ]
    
    INTERACTION_FEATURE_DESCRIPTIONS = {
        'age_immuno': '年龄×免疫状态交互',
        'dm_tb_synergy': '糖尿病×结核病史交互',
        'age_bcg_decay': '年龄×未接种BCG交互',
        'symptom_delay': '症状×延迟就诊交互',
        'cough_contact': '咳嗽频率×接触人数交互',
        'highrisk_comorbid': '高危人群×合并症交互',
        'immune_bcg': '未接种BCG×免疫抑制交互',
        'exposure_accumulation': '累积暴露×持续周期交互',
        'age_diabetes': '年龄×糖尿病交互'
    }
    
    ALL_FEATURE_NAMES = list(FEATURE_NAMES) + INTERACTION_FEATURE_NAMES
    ALL_FEATURE_DESCRIPTIONS = {**FEATURE_DESCRIPTIONS, **INTERACTION_FEATURE_DESCRIPTIONS}
    
    # 映射字典常量（问题八-1：单一真值源，从 constants.py 导入）
    # 与 ScoringEngine.DISTANCE_FACTORS / VENTILATION_FACTORS /
    # SETTING_FACTORS_BASE 同源，确保 ML 特征工程与评分引擎使用同一套口径
    DISTANCE_SCORE_MAP = _DISTANCE_SCORE_MAP_SRC
    VENTILATION_SCORE_MAP = _VENTILATION_SCORE_MAP_SRC
    SETTING_SCORE_MAP = _SETTING_SCORE_MAP_SRC
    
    # GNN相关常量
    EDGE_FEATURE_DIM = 6  # 边特征维度（6维，避免硬编码）
    NODE_FEATURE_DIM = 22  # 节点特征维度

    # GNN checkpoint 预期键结构（schema 校验用）
    _GNN_CHECKPOINT_SCHEMA = {
        'model_state_dict': (dict, type(None)),
        'gnn_is_trained': bool,
        'model_performance': dict,
        'hidden_dim': int,
        'num_layers': int,
        'node_feature_dim': int,
        'edge_feature_dim': int,
        'schema_version': str,
        'model_type': str,
    }
    


    def __init__(self, random_state=42):
        self.models = {}
        self.model_performance = {}
        self.is_trained = False
        self._is_calibrated = False       # 概率校准标志
        self._calibration_method = None   # 校准方法 (isotonic/sigmoid)
        self.calibrated_models = {}       # 校准后的模型
        self._is_stacking_trained = False # Stacking 集成标志
        self.stacking_meta_model = None   # Stacking 元模型
        self.stacking_weights = {}        # 元模型权重
        self.stacking_model_names = []    # 基模型名称顺序
        self.training_sample_count = 0
        self.use_real_data = False
        self.shap_explainer = None
        self.last_shap_values = None
        self.last_feature_matrix = None
        self.patient_ftd = 0
        self.patient_cough_freq = 0
        self.contact_count = 0

        # 迁移学习与预训练版本信息（预训练/微调阶段追溯）
        self.transfer_stage = 'untrained'
        self.transfer_info = {}

        # P4（标签代际，2026-08-24）：训练时由 _train_all_registered_models
        # 写入（'v3-host-pathway' / 'clinical-rule-v1' / None 未声明），
        # 随 checkpoint 落盘；expected_label_generation 由加载方设置以
        # 启用代际校验（不匹配 → load_model 拒绝，防错代加载）
        self.label_generation = None
        self.expected_label_generation = None
        
        # 统一随机种子，确保可复现性
        self.random_state = random_state
        
        # 设置所有随机种子
        if NUMPY_AVAILABLE:
            self._np_rng = np.random.RandomState(random_state)
        random.seed(random_state)
        try:
            # 注意：PYTHONHASHSEED 必须在解释器启动前设置，运行时设置无效
            # 保留此代码仅用于文档目的，实际不生效
            os.environ['PYTHONHASHSEED'] = str(random_state)
        except (KeyError, ValueError, OSError):
            pass
        
        # 如果PyTorch可用，设置PyTorch随机种子
        if PYTORCH_AVAILABLE:
            import torch
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        # 缓存合成训练数据，避免重复生成
        self._cached_training_data = None
        self._cached_training_params = None
        
        # GNN相关属性
        self.gnn_model = None
        self.stgnn_is_trained = False  # 时空GNN训练状态
        self.time_aware_gnn_model = None  # 时间感知GNN模型
        self.gnn_network_builder = None
        self.gnn_is_trained = False
        
        # 训练停止标志
        self._stop_training = False
        
        # 治疗阶段传染性衰减参数（文献来源：WHO 2024指南）
        self.treatment_infectivity_decay = {
            'week_0': 0.85,      # 治疗前/开始时：85%传染性
            'week_2': 0.60,      # 治疗2周后：60%传染性
            'week_8': 0.30,      # 治疗8周后：30%传染性
            'week_24': 0.10      # 治疗24周后：10%传染性（基本无传染性）
        }

        # 贝叶斯生存分析预测器（集成DeepHit/DeepSurv + MC Dropout）
        self.survival_predictor = None
        self._survival_ready = False
        if SURVIVAL_BAYESIAN_AVAILABLE:
            try:
                self.survival_predictor = SurvivalPredictor(
                    input_dim=22, num_time_bins=52, mc_samples=50, use_deephit=False)
                self._survival_ready = True
            except (ValueError, RuntimeError, OSError):
                self._survival_ready = False

        # 竞争风险多结局预测器
        self.multi_state_predictor = None
        self._multistate_ready = False
        if MULTISTATE_AVAILABLE:
            try:
                self.multi_state_predictor = MultiStateSurvivalPredictor(
                    input_dim=22, num_time_bins=52, num_causes=4,
                    hidden_dim=96, mc_samples=50)
                self._multistate_ready = True
            except (ValueError, RuntimeError, OSError):
                self._multistate_ready = False

        # RL干预引擎引用（由ThreeDirectionIntegrator设置）
        self.rl_engine = None

    # ============================================================
    # 训练停止控制（供 GUI 异步训练中断；training.py 调用 reset_stop_flag）
    # ============================================================

    def reset_stop_flag(self):
        """重置训练停止标志（训练开始前由 training.py 调用）。"""
        self._stop_training = False

    def request_stop(self):
        """请求停止训练（供 GUI 停止按钮异步调用）。"""
        self._stop_training = True


    # ============================================================
    # 委托方法（内部调用 scoring/ml/ 子包中的独立函数）
    # ============================================================

    def calculate_treatment_infectivity_factor(self, *args, **kwargs):
        return _impl_calculate_treatment_infectivity_factor(self, *args, **kwargs)

    def calculate_multilayer_network_r0(self, *args, **kwargs):
        return _impl_calculate_multilayer_network_r0(self, *args, **kwargs)

    def _extract_features_from_records(self, *args, **kwargs):
        return _impl_extract_features_from_records(self, *args, **kwargs)

    def _generate_synthetic_training_data(self, *args, **kwargs):
        return _impl_generate_synthetic_training_data(self, *args, **kwargs)

    def _interruptible_grid_search(self, *args, **kwargs):
        return _impl_interruptible_grid_search(self, *args, **kwargs)

    def _train_with_fallback(self, *args, **kwargs):
        return _impl_train_with_fallback(self, *args, **kwargs)

    def train_models(self, *args, **kwargs):
        return _impl_train_models(self, *args, **kwargs)

    def _train_models_from_arrays(self, *args, **kwargs):
        return _impl_train_models_from_arrays(self, *args, **kwargs)

    def compare_real_vs_synthetic(self, *args, **kwargs):
        return _impl_compare_real_vs_synthetic(self, *args, **kwargs)

    def _evaluate_training_mode(self, *args, **kwargs):
        return _impl_evaluate_training_mode(self, *args, **kwargs)

    def _bootstrap_mode_comparison(self, *args, **kwargs):
        return _impl_bootstrap_mode_comparison(self, *args, **kwargs)

    def _safe_roc_auc_score(self, y_true, y_score):
        """安全计算 ROC AUC（单一类别等情况返回 None 而非抛异常）

        委托 ml/evaluation._safe_roc_auc_score；evaluation.py 与
        gnn_training/training_pignn.py 以 predictor._safe_roc_auc_score 形式调用。
        """
        if _impl_safe_roc_auc_score is None:
            return None
        return _impl_safe_roc_auc_score(y_true, y_score)

    def _safe_average_precision_score(self, y_true, y_score):
        """安全计算 AUPRC（委托 ml/evaluation，边界情况返回 None）"""
        if _impl_safe_average_precision_score is None:
            return None
        return _impl_safe_average_precision_score(y_true, y_score)

    def _wilcoxon_signed_rank(self, *args, **kwargs):
        """Wilcoxon 符号秩检验（委托 ml/evaluation）"""
        if _impl_wilcoxon_signed_rank is None:
            return None
        return _impl_wilcoxon_signed_rank(*args, **kwargs)

    def _ensemble_predict(self, *args, **kwargs):
        """模型集成预测（委托 ml/evaluation）"""
        if _impl_ensemble_predict is None:
            return None
        return _impl_ensemble_predict(self, *args, **kwargs)

    def _wilcoxon_mode_comparison(self, *args, **kwargs):
        return _impl_wilcoxon_mode_comparison(self, *args, **kwargs)

    def _train_models_for_mode(self, *args, **kwargs):
        return _impl_train_models_for_mode(self, *args, **kwargs)

    def _generate_comparison_recommendation(self, *args, **kwargs):
        return _impl_generate_comparison_recommendation(self, *args, **kwargs)

    def train_from_arrays(self, *args, **kwargs):
        return _impl_train_from_arrays(self, *args, **kwargs)

    def extract_features(self, *args, **kwargs):
        return _impl_extract_features(self, *args, **kwargs)

    def extract_features_with_interactions(self, *args, **kwargs):
        return _impl_extract_features_with_interactions(self, *args, **kwargs)

    def _normalize_illness_type(self, illness_type):
        # 模块级 _impl_normalize_illness_type 不接收 self，仅接收 illness_type；
        # 绑定为方法供 preprocessing.extract_features_with_interactions 调用
        return _impl_normalize_illness_type(illness_type)

    def set_patient_context(self, *args, **kwargs):
        return _impl_set_patient_context(self, *args, **kwargs)

    def predict_risk(self, *args, **kwargs):
        return _impl_predict_risk(self, *args, **kwargs)

    def compute_calibration(self, *args, **kwargs):
        return _impl_compute_calibration(self, *args, **kwargs)

    def calibrate_models(self, *args, **kwargs):
        return _impl_calibrate_models(self, *args, **kwargs)

    def train_stacking_ensemble(self, *args, **kwargs):
        return _impl_train_stacking_ensemble(self, *args, **kwargs)

    def compute_shap_values(self, *args, **kwargs):
        return _impl_compute_shap_values(self, *args, **kwargs)

    def compute_contact_attributions(self, *args, **kwargs):
        return _impl_compute_contact_attributions(self, *args, **kwargs)

    def build_validation_report(self, *args, **kwargs):
        return _impl_build_validation_report(self, *args, **kwargs)

    def save_model(self, *args, **kwargs):
        return _impl_save_model(self, *args, **kwargs)

    def load_model(self, *args, **kwargs):
        return _impl_load_model(self, *args, **kwargs)

    # ============ 迁移学习与预训练（transfer learning） ============
    def generate_pretraining_dataset(self, *args, **kwargs):
        """按国家/WHO 公开统计生成预训练数据集（22 维特征矩阵 + 标签 + 元信息）。"""
        return _impl_generate_pretraining_dataset(self, *args, **kwargs)

    def pretrain_models(self, *args, **kwargs):
        """用国家监测 / WHO 公开数据预训练所有可用 ML 模型。"""
        return _impl_pretrain_models(self, *args, **kwargs)

    def finetune_models(self, *args, **kwargs):
        """用本地小样本对预训练模型微调（warm_start/init_model 延续预训练知识）。"""
        return _impl_finetune_models(self, *args, **kwargs)

    def compare_finetune_vs_from_scratch(self, *args, **kwargs):
        """对比"预训练+微调" vs "本地从零训练"在小样本上的泛化能力。"""
        return _impl_compare_finetune_vs_from_scratch(self, *args, **kwargs)

    def summarize_transfer_status(self):
        """汇总当前模型的迁移状态（阶段/预训练源/微调信息）。"""
        return _impl_summarize_transfer_status(self)

    def compute_bootstrap_ci(self, *args, **kwargs):
        return _impl_compute_bootstrap_ci(self, *args, **kwargs)

    def train_survival_multistate(self, *args, **kwargs):
        return _impl_train_survival_multistate(self, *args, **kwargs)

    def train_gnn(self, *args, **kwargs):
        return _impl_train_gnn(self, *args, **kwargs)

    def train_pignn(self, *args, **kwargs):
        return _impl_train_pignn(self, *args, **kwargs)

    def _generate_synthetic_graphs(self, *args, **kwargs):
        return _impl_generate_synthetic_graphs(self, *args, **kwargs)

    def _focal_loss(self, *args, **kwargs):
        return _impl_focal_loss(self, *args, **kwargs)

    def _train_gnn_once(self, *args, **kwargs):
        return _impl_train_gnn_once(self, *args, **kwargs)

    def _evaluate_gnn(self, *args, **kwargs):
        return _impl_evaluate_gnn(self, *args, **kwargs)

    def _compute_gnn_metrics(self, *args, **kwargs):
        return _impl_compute_gnn_metrics(self, *args, **kwargs)

    def predict_gnn_risk(self, *args, **kwargs):
        return _impl_predict_gnn_risk(self, *args, **kwargs)

    def predict_gnn_increment(self, *args, **kwargs):
        """第 2 层：GNN 残差增量（网络增强层）。

        网络增量 = 网络感知风险 − 个体基线概率 P_base。
        """
        if _impl_predict_gnn_increment is None:
            return None
        return _impl_predict_gnn_increment(self, *args, **kwargs)

    def compute_gnn_explanation(self, *args, **kwargs):
        return _impl_compute_gnn_explanation(self, *args, **kwargs)

    def _convert_hetero_to_homo(self, *args, **kwargs):
        return _impl_convert_hetero_to_homo(self, *args, **kwargs)

    def _find_target_node_idx(self, *args, **kwargs):
        return _impl_find_target_node_idx(self, *args, **kwargs)

    # ============ 大规模图网络处理（gnn_scalable 委托）============
    @property
    def gnn_scalable_available(self):
        """大规模图规模化能力是否可用（torch + torch_geometric）。"""
        return _GNN_SCALABLE_AVAILABLE

    def train_gnn_minibatch(self, model, large_data, config=None, **kwargs):
        """节点级 mini-batch GNN 训练（GraphSAGE 范式，整图不加载进显存）。"""
        if _scalable_train_gnn_minibatch is None:
            return {"success": False, "backend": "unavailable"}
        return _scalable_train_gnn_minibatch(model, large_data, config, **kwargs)

    def predict_gnn_k_hop(self, model, large_data, target_idx, num_hops=2, **kwargs):
        """对目标患者仅在其 k-hop 邻域子图上推理（不跑全图）。"""
        if _scalable_predict_target_k_hop is None:
            return None
        return _scalable_predict_target_k_hop(model, large_data, target_idx,
                                              num_hops, **kwargs)

    def quantize_gnn_model(self, model, quant_type='fp16'):
        """FP16 / INT8 量化副本，返回 (量化模型, backend)。"""
        if quant_type == 'int8':
            if _scalable_quantize_model_int8 is None:
                return model, 'fp32'
            return _scalable_quantize_model_int8(model)
        if _scalable_quantize_model_fp16 is None:
            return model, 'fp32'
        return _scalable_quantize_model_fp16(model), 'fp16'

    def measure_gnn_inference_latency(self, model, large_data, **kwargs):
        """测量 k-hop 子图推理延迟（含统计百分位）。"""
        if _scalable_measure_inference_latency is None:
            return None
        return _scalable_measure_inference_latency(model, large_data, **kwargs)

    def distill_gnn_to_mlp(self, teacher, large_data, **kwargs):
        """用大 GNN 蒸馏出轻量 MLP 学生（毫秒级单样本推理）。"""
        if _scalable_distill_gnn_to_mlp is None:
            return {"success": False}
        return _scalable_distill_gnn_to_mlp(teacher, large_data, **kwargs)

    def predict_with_gnn_student(self, student, node_features):
        """用蒸馏出的 MLP 学生仅基于节点特征预测风险（无图结构）。"""
        if _scalable_predict_with_mlp is None:
            return None
        return _scalable_predict_with_mlp(student, node_features)

    def add_contact_node(self, large_data, new_feature, neighbor_indices,
                         edge_attr=None):
        """向接触网络增量追加一个新接触者节点（原图不变）。"""
        if _scalable_add_new_node is None:
            return None
        return _scalable_add_new_node(large_data, new_feature,
                                      neighbor_indices, edge_attr)

    def incremental_predict_new_contact(self, model, large_data, new_feature,
                                        neighbor_indices, num_hops=2, **kwargs):
        """新增接触者增量推理：仅局部重算子图，而非全图重算。"""
        if _scalable_incremental_predict_new_node is None:
            return None
        return _scalable_incremental_predict_new_node(
            model, large_data, new_feature, neighbor_indices, num_hops, **kwargs)

    def train_from_real_data(self, *args, **kwargs):
        if _impl_train_from_real_data is None:
            return {
                "success": False,
                "error_type": "dependency_missing",
                "diagnostics": "scikit-learn 未安装，无法执行训练",
                "suggestions": ["pip install scikit-learn"],
            }
        return _impl_train_from_real_data(self, *args, **kwargs)

    def save_gnn_model(self, *args, **kwargs):
        return _impl_save_gnn_model(self, *args, **kwargs)

    def _validate_gnn_checkpoint(self, *args, **kwargs):
        return _impl_validate_gnn_checkpoint(self, *args, **kwargs)

    def load_gnn_model(self, *args, **kwargs):
        return _impl_load_gnn_model(self, *args, **kwargs)

    def train_stgnn(self, *args, **kwargs):
        return _impl_train_stgnn(self, *args, **kwargs)

    def predict_time_aware_gnn_risk(self, *args, **kwargs):
        return _impl_predict_time_aware_gnn_risk(self, *args, **kwargs)
