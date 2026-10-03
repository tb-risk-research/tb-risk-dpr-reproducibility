#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型持久化：ML模型和GNN模型的保存/加载

从 MLRiskPredictor 拆分出的独立函数模块。
所有函数无状态，通过参数传递数据。
"""
import copy
import datetime
import logging
import math
import os
import pickle
import random

import numpy as np

LOGGER = logging.getLogger("tb_risk.ml")

# 条件导入（与主类保持一致）
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 accuracy_score, precision_score, recall_score,
                                 f1_score, brier_score_loss)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    sklearn = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

NUMPY_AVAILABLE = True

# --- 内部模块条件导入 ---
try:
    from ...ml.framework import HeterogeneousTBNetwork, SEIRInformedGNN, SEIRTimeAwareGNN
except ImportError:
    HeterogeneousTBNetwork = None
    SEIRInformedGNN = None
    SEIRTimeAwareGNN = None



# ============================================================
# 以下函数从 MLRiskPredictor 的实例方法拆分而来
# 每个函数接收 predictor 实例作为第一个参数
# ============================================================

def save_model(predictor, filepath):
        """保存训练好的模型到文件
        
        参数：
            filepath (str): 保存路径，建议使用.pkl或.joblib扩展名
        
        返回：
            bool: 保存是否成功
        """
        if not JOBLIB_AVAILABLE:
            LOGGER.warning("joblib未安装，无法保存模型")
            return False
        
        if not predictor.is_trained:
            LOGGER.warning("模型未训练，无法保存")
            return False
        
        try:
            model_data = {
                'models': predictor.models,
                'model_performance': predictor.model_performance,
                'is_trained': predictor.is_trained,
                'training_sample_count': predictor.training_sample_count,
                'use_real_data': predictor.use_real_data,
                'feature_names': predictor.ALL_FEATURE_NAMES,
                'feature_descriptions': predictor.ALL_FEATURE_DESCRIPTIONS,
                'interaction_feature_names': predictor.INTERACTION_FEATURE_NAMES,
                'interaction_feature_descriptions': predictor.INTERACTION_FEATURE_DESCRIPTIONS,
                'version': '2.0'
            }

            # 迁移学习版本信息（预训练/微调阶段追溯；无则写 untrained 空值）
            model_data['transfer_stage'] = getattr(predictor, 'transfer_stage', 'untrained')
            model_data['transfer_info'] = getattr(predictor, 'transfer_info', {}) or {}
            # P4（标签代际，2026-08-24）：标签机制代际随 checkpoint 落盘
            #（'v3-host-pathway' / 'clinical-rule-v1' / None 未声明），
            # 供 load_model 代际校验，防止标签换代后静默加载错代模型
            model_data['label_generation'] = getattr(
                predictor, 'label_generation', None)
            # H-ML3（特征契约，训练/部署契约统一）：feature_set（训练
            # 声明的特征集版本）+ feature_contract（模型实际拟合的真实
            # 列名，取首个记录非 Column_N 名的模型）随档落盘。加载端
            # 与当前代码的审计契约比对，防止 v2/v3 档案特征空间与审计
            # 结论脱钩后被静默部署。
            model_data['feature_set'] = getattr(predictor, 'feature_set', 'v1')
            _contract_names = None
            for _m in (predictor.models or {}).values():
                _fn = getattr(_m, 'feature_names_in_', None)
                if _fn is not None and not any(
                        str(_x).startswith('Column_') for _x in _fn):
                    _contract_names = [str(_x) for _x in _fn]
                    break
            model_data['feature_contract'] = _contract_names
            # v4（P1，2026-09-16）：族感知特征空间三件套随档落盘——
            # 队列名（列集 per-cohort，跨队列不可比）+ 特征名清单 +
            # 训练中位数（部署端单样本构造复用，绝不在部署数据上重算）。
            # v4 列集无全局 canonical 契约可比，加载端仅恢复不校验；
            # 预测端按模型 feature_names_in_ + 此三件套构造输入。
            if model_data['feature_set'] == 'v4':
                model_data['v4_cohort'] = getattr(predictor, 'v4_cohort', None)
                model_data['v4_feature_names'] = list(
                    getattr(predictor, 'v4_feature_names', []) or [])
                model_data['v4_fill_medians'] = dict(
                    getattr(predictor, 'v4_fill_medians', {}) or {})
            # P2（小队列部署模型路由）：路由决策随档落盘（preferred_key /
            # 规则 / 触发原因 / CV 披露表）——部署端 predict_risk 报
            # preferred_model 用；旧档无此键 = 未路由（兼容）
            if getattr(predictor, 'deployment_model', None):
                model_data['deployment_model'] = dict(
                    predictor.deployment_model)
            # P1-c（treats 降级机制护栏，2026-09-20）：降级标记随档落盘
            # （research_only 队列的 checkpoint 永久携带，加载端恢复 +
            # predict_risk 拦截）；未降级档无此键（兼容）
            if getattr(predictor, 'deployment_status', None) == 'research_only':
                model_data['deployment_status'] = 'research_only'
                model_data['research_only_reason'] = getattr(
                    predictor, 'research_only_reason', '')
            # P5（生存路径转正，2026-09-16）：Cox PH 部署工件随档落盘——
            # follow_up_days 数据（peru_mdr）的 canonical 模型。系数/
            # 掩码/标准化参数完整携带预测所需信息（底层 PHRegResults
            # 不落盘）；旧档无此键 = 未走生存路径（兼容）
            _surv = getattr(predictor, 'survival_model', None)
            if _surv is not None:
                try:
                    model_data['survival_model'] = _surv.to_dict()
                except Exception as e:
                    LOGGER.warning('生存模型序列化失败（跳过落盘）: %s', e)

            if SKLEARN_AVAILABLE:
                try:
                    model_data['sklearn_version'] = sklearn.__version__
                except (ImportError, AttributeError):
                    pass

            if XGBOOST_AVAILABLE:
                try:
                    model_data['xgboost_version'] = xgb.__version__
                except (ImportError, AttributeError):
                    pass

            if LIGHTGBM_AVAILABLE:
                try:
                    model_data['lightgbm_version'] = lgb.__version__
                except (ImportError, AttributeError):
                    pass

            if CATBOOST_AVAILABLE:
                try:
                    model_data['catboost_version'] = cb.__version__
                except (ImportError, AttributeError):
                    pass

            if JOBLIB_AVAILABLE:
                try:
                    model_data['joblib_version'] = joblib.__version__
                except (ImportError, AttributeError):
                    pass
            
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            LOGGER.warning("模型保存失败: %s", e, exc_info=True)
            return False


def load_model(predictor, filepath):
        """从文件加载训练好的模型
        
        参数：
            filepath (str): 模型文件路径
        
        返回：
            bool: 加载是否成功
        """
        if not JOBLIB_AVAILABLE:
            LOGGER.warning("joblib未安装，无法加载模型")
            return False
        
        # 安全检查：仅允许 .joblib/.pkl 扩展名，防止加载任意文件
        allowed_exts = ('.joblib', '.pkl', '.pickle')
        if not filepath.lower().endswith(allowed_exts):
            LOGGER.warning("拒绝加载非模型文件: %s (仅支持 %s 格式)", filepath, allowed_exts)
            return False
        
        try:
            # 安全警告：joblib.load 底层使用 pickle，仅加载可信来源的模型文件
            LOGGER.info(f"加载模型文件: {filepath} (注意：pickle 反序列化，仅加载可信来源)")
            model_data = joblib.load(filepath)
            
            if not isinstance(model_data, dict) or 'models' not in model_data:
                LOGGER.warning("无效的模型文件格式")
                return False
            
            warnings = []
            if SKLEARN_AVAILABLE and 'sklearn_version' in model_data:
                try:
                    saved_version = model_data['sklearn_version']
                    current_version = sklearn.__version__
                    if saved_version != current_version:
                        warnings.append(f"sklearn: 保存时 {saved_version}, 当前 {current_version}")
                except (KeyError, AttributeError):
                    pass
            
            if XGBOOST_AVAILABLE and 'xgboost_version' in model_data:
                try:
                    saved_version = model_data['xgboost_version']
                    current_version = xgb.__version__
                    if saved_version != current_version:
                        warnings.append(f"xgboost: 保存时 {saved_version}, 当前 {current_version}")
                except (KeyError, AttributeError):
                    pass
            
            if LIGHTGBM_AVAILABLE and 'lightgbm_version' in model_data:
                try:
                    saved_version = model_data['lightgbm_version']
                    current_version = lgb.__version__
                    if saved_version != current_version:
                        warnings.append(f"lightgbm: 保存时 {saved_version}, 当前 {current_version}")
                except (KeyError, AttributeError):
                    pass
            
            if CATBOOST_AVAILABLE and 'catboost_version' in model_data:
                try:
                    saved_version = model_data['catboost_version']
                    current_version = cb.__version__
                    if saved_version != current_version:
                        warnings.append(f"catboost: 保存时 {saved_version}, 当前 {current_version}")
                except (KeyError, AttributeError):
                    pass
            
            if JOBLIB_AVAILABLE and 'joblib_version' in model_data:
                try:
                    saved_version = model_data['joblib_version']
                    current_version = joblib.__version__
                    if saved_version != current_version:
                        warnings.append(f"joblib: 保存时 {saved_version}, 当前 {current_version}")
                except (KeyError, AttributeError):
                    pass
            
            if warnings:
                LOGGER.warning("模型库版本不匹配：%s", "; ".join(warnings))
            
            # P4（标签代际校验，2026-08-24）：调用方设置
            # predictor.expected_label_generation（非 None）且 checkpoint
            # 已声明代际（旧代文件为 None/缺失 → 放行 + warning）时，
            # 不匹配 → 拒绝加载（best.json 跨代比较失效后，这是防止
            # 静默加载错代模型的最后一道闸）
            checkpoint_generation = model_data.get('label_generation', None)
            expected = getattr(predictor, 'expected_label_generation', None)
            if (expected is not None and checkpoint_generation is not None
                    and checkpoint_generation != expected):
                LOGGER.error(
                    "标签代际不匹配，拒绝加载: checkpoint 声明 %r，"
                    "调用方期望 %r（%s）",
                    checkpoint_generation, expected, filepath)
                return False
            if expected is not None and checkpoint_generation is None:
                LOGGER.warning(
                    "checkpoint 未声明标签代际（旧代文件），跳过代际校验: %s",
                    filepath)

            predictor.models = model_data['models']
            predictor.model_performance = model_data.get('model_performance', {})
            predictor.is_trained = model_data.get('is_trained', True)
            predictor.training_sample_count = model_data.get('training_sample_count', 0)
            predictor.use_real_data = model_data.get('use_real_data', False)
            predictor.label_generation = checkpoint_generation

            # H-ML3（特征契约恢复 + 校验）：feature_set / feature_contract
            # 恢复到 predictor；checkpoint 声明 v2/v3 时与当前代码的审计
            # 契约（feature_audit.SELECTED_FEATURES_V2/V3）比对——
            # 不一致说明该档训练时的特征集定义已被改动（跨代码版本旧档
            # /手改契约后训练），告警但不拒载（旧档兼容），由预测端
            # _check_feature_contract 持续追溯。
            ck_feature_set = model_data.get('feature_set', None)
            ck_feature_contract = model_data.get('feature_contract', None)
            predictor.feature_set = ck_feature_set or 'v1'
            predictor.feature_contract = ck_feature_contract
            if ck_feature_set in ('v2', 'v3'):
                try:
                    from .feature_audit import (SELECTED_FEATURES_V2,
                                                SELECTED_FEATURES_V3)
                    _canonical = (SELECTED_FEATURES_V2 if ck_feature_set == 'v2'
                                  else SELECTED_FEATURES_V3)
                    if ck_feature_contract is not None and \
                            set(ck_feature_contract) != set(_canonical):
                        _diff = sorted(
                            set(ck_feature_contract) ^ set(_canonical))
                        LOGGER.warning(
                            "特征契约漂移（H-ML3）：checkpoint 声明 "
                            "feature_set=%r，但其记录的 %d 维特征契约与"
                            "当前代码的 %s 审计契约不一致（对称差 %d 项: "
                            "%s...）。该档训练时的特征集定义已被改动，"
                            "部署语义与审计结论脱钩，建议重训。",
                            ck_feature_set, len(ck_feature_contract),
                            ck_feature_set, len(_diff), _diff[:4])
                    elif ck_feature_contract is None:
                        LOGGER.warning(
                            "checkpoint 声明 feature_set=%r 但未记录"
                            "feature_contract（旧版保存路径），跳过契约"
                            "比对；预测端仍会做名集校验", ck_feature_set)
                except ImportError:
                    pass
            elif ck_feature_set == 'v4':
                # v4（P1）：三件套恢复（队列/特征名/训练中位数）。列集
                # per-cohort 动态，无全局 canonical 可比——仅做完整性
                # 检查（有模型却无三件套 = 异常保存路径），不拒载。
                predictor.v4_cohort = model_data.get('v4_cohort')
                predictor.v4_feature_names = (
                    model_data.get('v4_feature_names') or [])
                predictor.v4_fill_medians = (
                    model_data.get('v4_fill_medians') or {})
                if not predictor.v4_feature_names and predictor.models:
                    LOGGER.warning(
                        "checkpoint 声明 feature_set='v4' 但未记录"
                        "v4_feature_names（异常保存路径）——预测端 v4 "
                        "构造将回退按模型 feature_names_in_ 尺寸校验失败"
                        "并逐模型告警，建议重训")
            elif ck_feature_set is None:
                LOGGER.warning(
                    "checkpoint 未声明 feature_set（旧版文件，按 v1 22 维"
                    "语义加载）；预测端 _model_input_frame 按模型自身"
                    "feature_names_in_ 选列并做契约校验")

            # 迁移学习版本信息恢复（旧版模型文件无此字段时保持默认）
            predictor.transfer_stage = model_data.get('transfer_stage', 'untrained')
            predictor.transfer_info = model_data.get('transfer_info', {}) or {}

            # P2（小队列部署模型路由）恢复：旧档无此键 = 未路由（兼容，
            # predict_risk 不报 preferred_model 字段）
            if model_data.get('deployment_model'):
                predictor.deployment_model = model_data['deployment_model']

            # P1-c（treats 降级机制护栏，2026-09-20）：降级标记恢复 +
            # 旧档回填。新档（训练时命中 RESEARCH_ONLY_COHORTS）直接
            # 恢复 deployment_status；旧档（t7 treats checkpoint 等，
            # 保存早于护栏落地）按 checkpoint 自带的 v4_cohort 回填补
            # flag——两层合并后任何 treats checkpoint 加载即带
            # research_only 标记，predict_risk 部署入口拦截。
            try:
                from ...constants import RESEARCH_ONLY_COHORTS
            except ImportError:  # pragma: no cover - constants 始终在场
                RESEARCH_ONLY_COHORTS = {}
            _status = model_data.get('deployment_status')
            _ck_cohort = model_data.get('v4_cohort')
            if _status == 'research_only' \
                    or _ck_cohort in RESEARCH_ONLY_COHORTS:
                predictor.deployment_status = 'research_only'
                predictor.research_only_reason = (
                    model_data.get('research_only_reason')
                    or (RESEARCH_ONLY_COHORTS.get(_ck_cohort, {})
                        .get('reason', '')))
                if _status != 'research_only':
                    LOGGER.warning(
                        '旧档回填：checkpoint 队列 %s 命中降级注册表'
                        '（research_only，保存早于护栏落地）——'
                        'predict_risk 部署入口将拦截（研究用途显式 '
                        'allow_research=True）', _ck_cohort)
            elif hasattr(predictor, 'deployment_status'):
                # 未降级档：清 predictor 上的陈旧标记（同一实例先后
                # 加载多档时不残留）
                delattr(predictor, 'deployment_status')
                if hasattr(predictor, 'research_only_reason'):
                    delattr(predictor, 'research_only_reason')

            # P5（生存路径转正）恢复：旧档无此键 = 未走生存路径（兼容，
            # predict_risk 不报 survival_cox 字段）
            if model_data.get('survival_model'):
                try:
                    from .survival_cox import CoxSurvivalModel
                    predictor.survival_model = CoxSurvivalModel.from_dict(
                        model_data['survival_model'])
                except Exception as e:
                    LOGGER.warning('生存模型恢复失败（忽略）: %s', e)

            # SHAP解释器初始化（带次优模型回退）
            if SHAP_AVAILABLE and len(predictor.models) > 0:
                sorted_models = sorted(
                    predictor.model_performance.items(),
                    key=lambda x: x[1].get('AUROC', 0),
                    reverse=True
                )
                for model_name, perf in sorted_models:
                    if model_name not in predictor.models:
                        continue
                    try:
                        predictor.shap_explainer = shap.TreeExplainer(predictor.models[model_name])
                        LOGGER.info("SHAP解释器初始化成功，使用模型: %s (AUROC=%.4f)",
                                   perf.get('name', model_name), perf.get('AUROC', 0))
                        break
                    except Exception as e:
                        LOGGER.warning("SHAP解释器使用 %s 创建失败，尝试下一个模型: %s",
                                      perf.get('name', model_name), e)
                        continue
                else:
                    LOGGER.warning("所有模型均无法创建SHAP解释器，特征重要性分析将不可用")
                    predictor.shap_explainer = None
            
            return True
        except Exception as e:
            LOGGER.warning("模型加载失败: %s", e, exc_info=True)
            return False


def save_gnn_model(predictor, filepath):
        """保存GNN模型到文件（文献支撑：PyTorch官方最佳实践）

        参数：
            filepath: 保存文件路径

        返回：
            bool: 是否成功
        """
        if not (PYTORCH_AVAILABLE and PYG_AVAILABLE) or not predictor.gnn_is_trained:
            return False

        try:
            import torch

            # 提取模型架构信息（回退默认与架构约束一致：num_layers=1，
            # 单层聚合防过平滑——见 training_gnn.py 约束注记）
            gnn_perf = predictor.model_performance.get('gnn', {})
            hidden_dim = 64
            num_layers = 1
            if predictor.gnn_model is not None:
                hidden_dim = getattr(predictor.gnn_model, 'hidden_dim', 64)
                num_layers = getattr(predictor.gnn_model, 'num_layers', 1)

            # numpy 标量 → Python 原生类型（根治 PyTorch 2.6 weights_only
            # 对 checkpoint 元数据的加载失败；训练指标多来自 numpy float）
            def _to_native(obj):
                if isinstance(obj, dict):
                    return {k: _to_native(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return type(obj)(_to_native(v) for v in obj)
                if isinstance(obj, np.generic):
                    return obj.item()
                return obj

            save_dict = {
                'model_state_dict': predictor.gnn_model.state_dict() if predictor.gnn_model is not None else None,
                'gnn_is_trained': bool(predictor.gnn_is_trained),
                'model_performance': _to_native(gnn_perf),
                # 模型架构元数据（用于加载时重建和 schema 校验）
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                # 从首层权重推断真实输入维度（训练时按数据实际维度建模，
                # 如 M7 度数特征加入后为 23 维；写死 NODE_FEATURE_DIM
                # 常量会导致加载时 size mismatch）
                'node_feature_dim': (
                    int(predictor.gnn_model.node_embedding.weight.shape[1])
                    if predictor.gnn_model is not None
                    else predictor.NODE_FEATURE_DIM),
                'edge_feature_dim': predictor.EDGE_FEATURE_DIM,
                'schema_version': '2.0',
                'model_type': 'SEIRInformedGNN',
            }

            torch.save(save_dict, filepath)
            LOGGER.info("GNN模型已保存到: %s", filepath)
            return True

        except Exception as e:
            LOGGER.warning("保存GNN模型失败: %s", e, exc_info=True)
            return False


def _validate_gnn_checkpoint(predictor, checkpoint):
        """校验 GNN checkpoint 字典的键结构是否符合预期 schema

        参数：
            checkpoint (dict): 加载的 checkpoint 字典

        返回：
            tuple[bool, str]: (是否有效, 错误信息)
        """
        if not isinstance(checkpoint, dict):
            return False, "checkpoint 不是字典类型"

        missing = []
        wrong_type = []
        for key, expected_types in predictor._GNN_CHECKPOINT_SCHEMA.items():
            if key not in checkpoint:
                missing.append(key)
            elif not isinstance(checkpoint[key], expected_types):
                wrong_type.append(
                    f"{key}: 期望 {expected_types}, 实际 {type(checkpoint[key]).__name__}")

        errors = []
        if missing:
            errors.append(f"缺少必需键: {missing}")
        if wrong_type:
            errors.append(f"类型不匹配: {wrong_type}")

        if errors:
            return False, "; ".join(errors)
        return True, ""


def load_gnn_model(predictor, filepath):
        """从文件加载GNN模型（含安全检查）

        安全措施：
        1. 扩展名白名单（仅 .pth/.pt）
        2. torch.load 使用 weights_only=True（PyTorch 2.0+）
        3. checkpoint 字典 schema 校验

        参数：
            filepath: 模型文件路径

        返回：
            bool: 是否成功
        """
        if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
            return False

        try:
            import torch

            # 检查文件是否存在
            if not os.path.exists(filepath):
                LOGGER.warning("GNN模型文件不存在: %s", filepath)
                return False

            # 安全检查1：扩展名白名单
            allowed_exts = ('.pth', '.pt')
            if not filepath.lower().endswith(allowed_exts):
                LOGGER.warning("拒绝加载非GNN模型文件: %s (仅支持 %s 格式)", filepath, allowed_exts)
                return False

            # 安全检查2：torch.load 使用 weights_only=True（PyTorch 2.0+）
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            try:
                # PyTorch 2.0+ 支持 weights_only 参数，限制反序列化范围
                checkpoint = torch.load(filepath, map_location=device, weights_only=True)
            except TypeError:
                # PyTorch < 2.0 不支持 weights_only，回退到普通加载但记录警告
                LOGGER.warning("PyTorch 版本 < 2.0，weights_only 不可用。"
                             "torch.load 将使用默认 pickle 反序列化，"
                             "请确保模型文件来源可信。"
                             "建议升级到 PyTorch 2.0+ 以获得安全增强。")
                checkpoint = torch.load(filepath, map_location=device)
            except pickle.UnpicklingError as e:
                # PyTorch 2.6+：checkpoint 元数据（model_performance 中的
                # numpy 标量/dtype）不在 weights_only 默认白名单。这些是
                # 纯数据容器（无代码执行面），allowlist 后重试（修复
                # "自己保存的模型自己加载不了"）；仍失败才放弃。
                import numpy as _np
                _allow = [_np.dtype]
                _core = getattr(_np, '_core', None) or getattr(_np, 'core', None)
                if _core is not None and hasattr(_core, 'multiarray'):
                    _allow.append(_core.multiarray.scalar)
                _dtypes_mod = getattr(_np, 'dtypes', None)
                if _dtypes_mod is not None:
                    _allow += [getattr(_dtypes_mod, _n) for _n in dir(_dtypes_mod)
                               if _n.endswith('DType')]
                try:
                    torch.serialization.add_safe_globals(_allow)
                    checkpoint = torch.load(
                        filepath, map_location=device, weights_only=True)
                except Exception:
                    LOGGER.warning("GNN checkpoint 含不支持的 numpy 对象"
                                   "（weights_only 加载失败）: %s", filepath)
                    return False

            # 安全检查3：checkpoint schema 校验
            is_valid, error_msg = predictor._validate_gnn_checkpoint(checkpoint)
            if not is_valid:
                LOGGER.warning("GNN checkpoint schema 校验失败: %s", error_msg)
                return False

            # 获取模型配置
            hidden_dim = checkpoint.get('hidden_dim', 64)
            num_layers = checkpoint.get('num_layers', 3)
            node_feature_dim = checkpoint.get('node_feature_dim', predictor.NODE_FEATURE_DIM)

            # 历史 checkpoint 的维度元数据偏差修正：保存端曾写死
            # NODE_FEATURE_DIM 常量（22），而 M7 生成器加入度数特征后
            # 实际输入为 23 维。以 state_dict 首层权重的真实形状为准。
            _sd = checkpoint.get('model_state_dict') or {}
            if 'node_embedding.weight' in _sd:
                _real_dim = int(_sd['node_embedding.weight'].shape[1])
                if _real_dim != node_feature_dim:
                    LOGGER.warning(
                        "checkpoint node_feature_dim 元数据 %d 与 state_dict "
                        "实际 %d 不一致，以 state_dict 为准",
                        node_feature_dim, _real_dim)
                    node_feature_dim = _real_dim

            # 创建并加载模型
            predictor.gnn_model = SEIRInformedGNN(
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(device)

            if checkpoint.get('model_state_dict') is not None:
                predictor.gnn_model.load_state_dict(checkpoint['model_state_dict'])

            predictor.gnn_is_trained = checkpoint.get('gnn_is_trained', False)
            predictor.gnn_network_builder = HeterogeneousTBNetwork()

            if 'gnn' not in predictor.model_performance:
                predictor.model_performance['gnn'] = checkpoint.get('model_performance', {})

            LOGGER.info("GNN模型已加载: %s (schema=%s, hidden=%d, layers=%d)",
                       filepath,
                       checkpoint.get('schema_version', 'unknown'),
                       hidden_dim, num_layers)
            return True

        except Exception as e:
            LOGGER.warning("加载GNN模型失败: %s", e, exc_info=True)
            return False


