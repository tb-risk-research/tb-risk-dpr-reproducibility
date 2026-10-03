#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型评估：AUROC/AUPRC/Brier、Bootstrap CI、Wilcoxon检验、风险预测

从 MLRiskPredictor 拆分出的独立函数模块。
所有函数无状态，通过参数传递数据。
"""
import copy
import datetime
import logging
import math
import os
import random

import numpy as np

from ...constants import DEFAULT_RISK_CLASS_THRESHOLD

LOGGER = logging.getLogger("tb_risk.ml")

# 条件导入（与主类保持一致）
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 accuracy_score, precision_score, recall_score,
                                 f1_score, brier_score_loss)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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



from ...utils import _is_yes

# ============================================================
# 以下函数从 MLRiskPredictor 的实例方法拆分而来
# 每个函数接收 predictor 实例作为第一个参数
# ============================================================

def compare_real_vs_synthetic(predictor, real_data_path=None, n_samples=2000,
                                   test_size=0.3, n_bootstrap=500, random_state=42):
        """真实数据与合成数据性能对比实验

        在相同评估指标下分别训练纯合成数据、纯真实数据、真实加合成增强
        三种模式，用 Bootstrap BCa 计算 95% 置信区间，用 Wilcoxon 符号秩检验
        判断差异显著性，为合成数据训练的可接受性提供统计学证据。

        参数：
            real_data_path (str|None): 真实数据文件路径。为 None 时跳过对比。
            n_samples (int): 总样本量，默认 2000
            test_size (float): 测试集比例，默认 0.3
            n_bootstrap (int): Bootstrap 重采样次数，默认 500
            random_state (int): 随机种子

        返回：
            dict: 结构化对比报告，包含以下键：
                - 'status': 'ok'/'skipped'/'error'
                - 'test_size': int，测试集样本量
                - 'modes': dict，各训练模式的性能指标
                - 'bootstrap_ci': dict，各模式各指标的 95% Bootstrap BCa CI
                - 'wilcoxon': dict，成对 Wilcoxon 检验结果
                - 'recommendation': str，基于统计证据的建议
        """
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

        report = {
            'status': 'skipped',
            'timestamp': datetime.datetime.now().isoformat(),
            'test_size': 0,
            'modes': {},
            'bootstrap_ci': {},
            'wilcoxon': {},
            'recommendation': '未执行对比实验',
        }

        if not real_data_path or not os.path.exists(real_data_path):
            report['recommendation'] = '未提供真实数据路径，无法执行对比实验。'
            if real_data_path:
                LOGGER.warning("真实数据文件不存在: %s", real_data_path)
            return report

        try:
            from .simulator import ScreeningDataSimulator
            simulator = ScreeningDataSimulator(random_state=random_state)
            records, meta = simulator.load_or_generate(
                filepath=real_data_path, n_samples=n_samples, include_labels=True
            )
            X_real, y_real = predictor._extract_features_from_records(records)

            n_real = len(records)
            if n_real < 50:
                report['status'] = 'error'
                report['recommendation'] = (
                    f'真实数据仅有 {n_real} 条（需要 ≥50），'
                    '无法执行有统计意义的对比实验。'
                )
                return report

            LOGGER.info("对比实验：加载 %d 条真实数据", n_real)

            X_train, X_test, y_train, y_test = train_test_split(
                X_real, y_real, test_size=test_size, random_state=random_state,
                stratify=y_real if len(np.unique(y_real)) > 1 else None
            )
            report['test_size'] = len(X_test)

            X_syn, y_syn = predictor._generate_synthetic_training_data(
                n_samples=n_samples, random_state=random_state
            )

            training_sets = {
                'synthetic': {'X': X_syn, 'y': y_syn, 'label': '纯合成数据'},
                'real': {'X': X_train, 'y': y_train, 'label': '纯真实数据'},
                'mixed': {
                    'X': np.vstack([X_train, X_syn]),
                    'y': np.hstack([y_train, y_syn]),
                    'label': '真实+合成增强',
                },
            }

            model_keys = ['random_forest', 'gradient_boosting']
            if XGBOOST_AVAILABLE:
                model_keys.append('xgboost')
            if LIGHTGBM_AVAILABLE:
                model_keys.append('lightgbm')
            if CATBOOST_AVAILABLE:
                model_keys.append('catboost')

            metrics_by_mode = {}
            for mode_name, ts in training_sets.items():
                metrics_by_mode[mode_name] = predictor._evaluate_training_mode(
                    X_train=ts['X'], y_train=ts['y'],
                    X_test=X_test, y_test=y_test,
                    model_keys=model_keys,
                    random_state=random_state,
                    label=ts['label'],
                )
            report['modes'] = metrics_by_mode

            report['bootstrap_ci'] = predictor._bootstrap_mode_comparison(
                X_test=X_test, y_test=y_test,
                training_sets=training_sets,
                model_keys=model_keys,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )

            report['wilcoxon'] = predictor._wilcoxon_mode_comparison(
                X_test=X_test, y_test=y_test,
                training_sets=training_sets,
                model_keys=model_keys,
                random_state=random_state,
            )

            report['recommendation'] = predictor._generate_comparison_recommendation(
                report['modes'], report['wilcoxon']
            )
            report['status'] = 'ok'

            LOGGER.info(
                "对比实验完成: 合成 AUROC=%.4f, 真实 AUROC=%.4f, 混合 AUROC=%.4f",
                metrics_by_mode['synthetic'].get('auroc', 0),
                metrics_by_mode['real'].get('auroc', 0),
                metrics_by_mode['mixed'].get('auroc', 0),
            )

        except Exception as e:
            report['status'] = 'error'
            report['recommendation'] = f'对比实验失败: {e}'
            LOGGER.warning("对比实验失败: %s", e, exc_info=True)

        return report


def _evaluate_training_mode(predictor, X_train, y_train, X_test, y_test,
                                 model_keys, random_state, label=''):
        """训练单个模式并评估性能指标（AUROC, AUPRC, Brier Score）"""
        import numpy as np
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

        # 安全计算 AUROC：当 y_test 只有一类时（如 Bootstrap 重采样极端情况），
        # roc_auc_score 会触发 UndefinedMetricWarning 或 ValueError。
        def _safe_auroc(y_true, y_score):
            unique = np.unique(y_true)
            if len(unique) < 2:
                return 0.5  # 单类时无法区分，返回随机猜测基线
            return float(roc_auc_score(y_true, y_score))

        def _safe_auprc(y_true, y_score):
            unique = np.unique(y_true)
            if len(unique) < 2:
                return float(np.mean(y_true))  # 单类时返回正类比例
            return float(average_precision_score(y_true, y_score))

        ensemble_pred = np.zeros(len(X_test), dtype=float)
        n_models_trained = 0
        model_aurocs = {}

        # 复用主训练路径的快速训练器（注册表默认参数 + 统一类不平衡
        # 加权：XGBoost scale_pos_weight / GB balanced sample_weight /
        # RF·LGBM·CatBoost 注册表 class_weight）。
        # 此前手工构造模型漏掉 GB/XGB 的不平衡处理，导致
        # compare_real_vs_synthetic 的 Wilcoxon 比较建立在弱化
        # 不平衡处理的模型上，与主模型不可比——统一后两条路径同构。
        from .training import _train_models_quick
        trained_models = _train_models_quick(
            model_keys, X_train, y_train, random_state)

        for key, model in trained_models.items():
            try:
                proba = model.predict_proba(
                    _aligned_feature_frame(model, X_test))[:, 1]
                ensemble_pred += proba
                n_models_trained += 1
                model_aurocs[key] = _safe_auroc(y_test, proba)
            except Exception as e:
                LOGGER.warning("[%s] %s 预测失败: %s", label, key, e)

        if n_models_trained == 0:
            return {'auroc': 0, 'auprc': 0, 'brier': 1, 'n_models': 0, 'label': label}

        ensemble_pred /= n_models_trained
        return {
            'auroc': _safe_auroc(y_test, ensemble_pred),
            'auprc': _safe_auprc(y_test, ensemble_pred),
            'brier': float(brier_score_loss(y_test, ensemble_pred)),
            'n_models': n_models_trained,
            'model_aurocs': model_aurocs,
            'label': label,
            'train_size': len(X_train),
        }


def _bootstrap_mode_comparison(predictor, X_test, y_test, training_sets,
                                    model_keys, n_bootstrap=500, random_state=42):
        """Bootstrap BCa 置信区间计算（AUROC, AUPRC, Brier Score）"""
        import numpy as np
        rng = np.random.RandomState(random_state)
        n_test = len(X_test)
        metrics = ['auroc', 'auprc', 'brier']
        ci_results = {}

        for mode_name in training_sets:
            ci_results[mode_name] = {}
            for metric in metrics:
                bootstrap_samples = np.zeros(n_bootstrap)
                for b in range(n_bootstrap):
                    idx = rng.choice(n_test, size=n_test, replace=True)
                    eval_result = predictor._evaluate_training_mode(
                        X_train=training_sets[mode_name]['X'],
                        y_train=training_sets[mode_name]['y'],
                        X_test=X_test[idx], y_test=y_test[idx],
                        model_keys=model_keys,
                        random_state=random_state + b,
                        label=f'{mode_name}_bootstrap_{b}',
                    )
                    bootstrap_samples[b] = eval_result.get(metric, 0)

                mean_val = float(np.mean(bootstrap_samples))
                ci_results[mode_name][metric] = {
                    'mean': mean_val,
                    'median': float(np.median(bootstrap_samples)),
                    'std': float(np.std(bootstrap_samples, ddof=1)),
                    'ci_95_lower': float(np.percentile(bootstrap_samples, 2.5)),
                    'ci_95_upper': float(np.percentile(bootstrap_samples, 97.5)),
                    'n_bootstrap': n_bootstrap,
                }
        return ci_results


def _wilcoxon_mode_comparison(predictor, X_test, y_test, training_sets,
                                   model_keys, random_state=42):
        """成对 Wilcoxon 符号秩检验（合成 vs 真实, 合成 vs 混合, 真实 vs 混合）"""
        import numpy as np
        wilcoxon_results = {}
        mode_names = list(training_sets.keys())
        pairs = [(mode_names[i], mode_names[j])
                 for i in range(len(mode_names))
                 for j in range(i + 1, len(mode_names))]

        for mode_a, mode_b in pairs:
            try:
                models_a = predictor._train_models_for_mode(
                    training_sets[mode_a]['X'], training_sets[mode_a]['y'],
                    model_keys, random_state
                )
                models_b = predictor._train_models_for_mode(
                    training_sets[mode_b]['X'], training_sets[mode_b]['y'],
                    model_keys, random_state
                )

                pred_a = predictor._ensemble_predict(models_a, X_test)
                pred_b = predictor._ensemble_predict(models_b, X_test)

                err_a = (y_test - pred_a) ** 2
                err_b = (y_test - pred_b) ** 2
                diff = err_a - err_b

                stat, p_value = predictor._wilcoxon_signed_rank(diff)

                wilcoxon_results[f'{mode_a}_vs_{mode_b}'] = {
                    'label': f'{training_sets[mode_a]["label"]} vs {training_sets[mode_b]["label"]}',
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'significant_at_0_05': p_value < 0.05,
                    'mean_diff': float(np.mean(diff)),
                    'n_pairs': len(diff),
                }
            except Exception as e:
                LOGGER.warning("Wilcoxon 检验 %s vs %s 失败: %s", mode_a, mode_b, e)
                wilcoxon_results[f'{mode_a}_vs_{mode_b}'] = {
                    'label': f'{training_sets[mode_a]["label"]} vs {training_sets[mode_b]["label"]}',
                    'error': str(e),
                }
        return wilcoxon_results


def _generate_comparison_recommendation(predictor, modes, wilcoxon):
        """基于对比实验结果生成统计建议"""
        if not modes or not wilcoxon:
            return '对比实验数据不足，无法生成建议。'

        syn_auroc = modes.get('synthetic', {}).get('auroc', 0)
        real_auroc = modes.get('real', {}).get('auroc', 0)
        mixed_auroc = modes.get('mixed', {}).get('auroc', 0)

        syn_vs_real = wilcoxon.get('synthetic_vs_real', {})
        sig_05 = syn_vs_real.get('significant_at_0_05', False)

        parts = [
            f'纯合成 AUROC={syn_auroc:.4f}, 纯真实 AUROC={real_auroc:.4f}, '
            f'真实+合成 AUROC={mixed_auroc:.4f}'
        ]

        if sig_05:
            parts.append(
                '合成数据与真实数据之间存在显著差异 (p<0.05)，'
                '建议优先使用真实数据或混合增强模式训练'
            )
        else:
            parts.append(
                '合成数据与真实数据之间无显著差异 (p≥0.05)，'
                '合成数据可作为真实数据不足时的有效替代'
            )

        if mixed_auroc > real_auroc and mixed_auroc > syn_auroc:
            parts.append('混合增强模式性能最优，推荐使用真实+合成数据联合训练')
        elif real_auroc >= mixed_auroc and real_auroc >= syn_auroc:
            parts.append('纯真实数据性能最优，建议优先收集真实数据')
        else:
            parts.append('纯合成数据性能最优，合成数据质量良好')

        return '；'.join(parts)


def compute_bootstrap_ci(predictor, n_bootstrap=100, random_state=42,
                         X=None, y=None):
        """使用bootstrap方法计算模型性能的95%置信区间

        M 级审计修复：原实现固定用合成数据算 CI——合成数据上的 CI 只
        反映"复现评分引擎"的方差，不构成真实泛化性能证据（项目硬
        约束），且与真实数据校准路径不同源。现支持显式传入真实数据
        (X, y)；未传时保留合成回退但打 data_source='synthetic' 标签
        并告警，调用方引用 CI 时必须核对口径。

        参数：
            n_bootstrap (int): bootstrap重采样次数，默认100
            random_state (int): 随机种子
            X (array-like|None): 评估特征矩阵（真实数据优先；None 回退合成）
            y (array-like|None): 评估标签（与 X 配对）

        返回：
            dict: 各模型的AUROC和AUPRC的95%置信区间
                （每项含 data_source: 'provided'/'synthetic' 口径标签）
        """
        if not SKLEARN_AVAILABLE or not predictor.is_trained:
            return None

        try:
            if X is not None and y is not None:
                X = np.asarray(X)
                y = np.asarray(y)
                data_source = 'provided'
            else:
                X, y = predictor._generate_synthetic_training_data(
                    2000, random_state)
                data_source = 'synthetic'
                LOGGER.warning(
                    'compute_bootstrap_ci 未传入真实数据，回退合成数据'
                    '（口径=data_source:synthetic）——该 CI 只反映复现'
                    '评分引擎的方差，不得作为真实泛化性能证据引用')
            rng = np.random.RandomState(random_state)

            ci_results = {}

            for model_name, model in predictor.models.items():
                auroc_scores = []
                auprc_scores = []

                for _ in range(n_bootstrap):
                    indices = rng.choice(len(X), size=len(X), replace=True)
                    X_boot = X[indices]
                    y_boot = y[indices]

                    if len(np.unique(y_boot)) < 2:
                        continue

                    try:
                        y_prob = model.predict_proba(
                            _aligned_feature_frame(model, X_boot))[:, 1]
                        auroc_scores.append(predictor._safe_roc_auc_score(y_boot, y_prob))
                        auprc_scores.append(predictor._safe_average_precision_score(y_boot, y_prob))
                    except (ValueError, RuntimeError, IndexError) as e:
                        LOGGER.debug('Bootstrap样本预测失败，跳过: %s', e)
                        continue

                if auroc_scores:
                    ci_results[model_name] = {
                        'AUROC_mean': np.mean(auroc_scores),
                        'AUROC_ci_lower': np.percentile(auroc_scores, 2.5),
                        'AUROC_ci_upper': np.percentile(auroc_scores, 97.5),
                        'AUPRC_mean': np.mean(auprc_scores),
                        'AUPRC_ci_lower': np.percentile(auprc_scores, 2.5),
                        'AUPRC_ci_upper': np.percentile(auprc_scores, 97.5),
                        'name': predictor.model_performance.get(model_name, {}).get('name', model_name),
                        'data_source': data_source,
                    }

            return ci_results

        except Exception as e:
            LOGGER.warning("Bootstrap置信区间计算失败: %s", e, exc_info=True)
            return None


def _compute_gnn_metrics(predictor, preds, labels):
        """计算完整的GNN评估指标

        阈值依赖指标（Accuracy/Precision/Recall/F1）使用约登指数最优切点
        而非固定 0.5：阳性率低（~4%）时 0.5 阈值必然全阴性，F1=0 只是
        阈值假象而非模型无判别力。约登指数在 J = Sens + Spec - 1 最大处
        重切（Youden 1950），复用 validation/threshold_spec 现成实现。
        """
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss

        from ...validation.threshold_spec import youden_optimal_threshold

        auroc = predictor._safe_roc_auc_score(labels, preds)
        auprc = predictor._safe_average_precision_score(labels, preds)

        opt = youden_optimal_threshold(
            [float(p) for p in np.asarray(preds).ravel()],
            [int(l) for l in np.asarray(labels).ravel()],
        )
        threshold = float(opt['threshold']) if opt else 0.5
        youden_index = float(opt['youden_index']) if opt else None

        pred_classes = (np.asarray(preds).ravel() >= threshold).astype(int)
        labels_flat = np.asarray(labels).ravel()
        
        try:
            accuracy = accuracy_score(labels_flat, pred_classes)
        except (ValueError, RuntimeError, ImportError):
            accuracy = 0.5

        try:
            precision = precision_score(labels_flat, pred_classes, zero_division=0)
        except (ValueError, RuntimeError, ImportError):
            precision = 0.0

        try:
            recall = recall_score(labels_flat, pred_classes, zero_division=0)
        except (ValueError, RuntimeError, ImportError):
            recall = 0.0

        try:
            f1 = f1_score(labels_flat, pred_classes, zero_division=0)
        except (ValueError, RuntimeError, ImportError):
            f1 = 0.0

        try:
            brier = brier_score_loss(labels_flat, np.asarray(preds).ravel())
        except (ValueError, RuntimeError, ImportError):
            brier = 0.25

        # L 级修复：positive_rate 进正式指标（原仅训练日志有，评估口径缺失
        # 会使 AUROC/AUPRC 对比失去患病率上下文——合成 4% vs 真实数据患病率
        # 不同时阈值依赖指标不可比）
        positive_rate = float(np.mean(labels_flat)) if len(labels_flat) else 0.0

        return {
            'auroc': auroc,
            'auprc': auprc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'brier': brier,
            'threshold': threshold,
            'youden_index': youden_index,
            'positive_rate': positive_rate,
        }
    

def _aligned_feature_frame(model, features):
    """把 numpy 特征矩阵按模型拟合时记录的特征名包装成 DataFrame。

    LightGBM 的 sklearn 包装器即使以 numpy 拟合也会把自动生成的特征名
    （Column_0..N）写入 feature_names_in_，此后任何 numpy 预测都会触发
    sklearn 的 "X does not have valid feature names" UserWarning（实测
    2026-08-24：GUI 训练 CV 内部与部署 predict_risk 均中招）。此处用
    模型自己记录的名字包装预测输入：名称精确匹配 → 警告消除，数值与
    列顺序完全不变；模型未记录特征名或维度不符时原样返回（不改变行为）。
    兼容两类 checkpoint：numpy 拟合的旧档（Column_N 名）与 DataFrame
    拟合的新档（真实特征名）。
    """
    if not PANDAS_AVAILABLE:
        return features
    names = getattr(model, 'feature_names_in_', None)
    if names is None or not hasattr(features, 'shape'):
        return features
    try:
        if len(names) != features.shape[1]:
            return features
        return pd.DataFrame(np.asarray(features), columns=list(names))
    except Exception:
        return features


# H-ML3（特征契约告警缓存）：按"名集"去重告警，避免预测热路径逐次刷屏。
# 键 = 排序后的特征名元组（有界：一个进程内只有少数几个特征集变体）。
_CONTRACT_WARNED = set()


def _check_feature_contract(model, names, all_names):
    """校验模型特征名集是否与当前代码的审计契约一致（H-ML3）。

    v2（15 维）/ v3（26 维）特征集是特征审计的**结论**（哪些列进模型），
    不是任意可变配置。预测端若只按 feature_names_in_ 动态选列，一个用
    非契约特征集训练的模型（手改 SELECTED_FEATURES_V2 后训练、或跨
    代码版本训练的旧档）会被静默适配，审计结论被绕过。此处对维度
    命中已知契约的模型做名集等值校验：

      - 15 名 == SELECTED_FEATURES_V2（集合等值）→ v2 契约通过
      - 26 名 == SELECTED_FEATURES_V3 → v3 契约通过
      - 22 名 == all_names（部署端 22 维全集）→ v1 契约通过
      - 维度命中但名集不等 → 契约违规告警（每名集一次）
      - Column_N 自动名 / 其他维度 → 旧 numpy 档，跳过（行为不变）

    不中断预测（旧档兼容优先），但违规模型的使用可被日志追溯。
    """
    n = len(names)
    if any(str(_x).startswith('Column_') for _x in names):
        return  # 旧 numpy 拟合档（自动名），无契约可校验
    if n == 22 and all_names is not None:
        canonical, label = list(all_names), 'v1'
    elif n in (15, 26):
        try:
            from .feature_audit import (SELECTED_FEATURES_V2,
                                        SELECTED_FEATURES_V3,
                                        NETWORK_FEATURE_NAMES_V3)
        except Exception:
            return
        if n == 15:
            canonical, label = SELECTED_FEATURES_V2, 'v2'
        else:
            # 26 维歧义防护（P1，2026-09-16）：22 + 4 extras 的 v4 列集
            # （如 brazil）同样是 26 维——仅当全集外列与 v3 网络列有
            # 交集（v3 家族特征，含拼写错的漂移档）时才按 v3 契约校验；
            # 无交集（v4 per-cohort extras，无全局 canonical）跳过
            if all_names is not None:
                _extra = set(names) - set(all_names)
                if not (_extra & set(NETWORK_FEATURE_NAMES_V3)):
                    return
            canonical, label = SELECTED_FEATURES_V3, 'v3'
    else:
        return  # 其他维度：非三代契约谱系，不校验
    key = tuple(sorted(str(_x) for _x in names))
    if key in _CONTRACT_WARNED:
        return
    if set(names) != set(canonical):
        _CONTRACT_WARNED.add(key)
        mismatched = sorted(set(names) ^ set(canonical))
        LOGGER.warning(
            "特征契约违规（H-ML3）：模型 %s 记录 %d 维特征集与当前代码"
            "的 %s 审计契约不一致（对称差 %d 项: %s...）。预测端仍按模型"
            "记录选列，但该模型训练时的特征空间已脱离审计结论——"
            "请核对训练代码版本或重训。",
            type(model).__name__, n, label, len(mismatched),
            mismatched[:4])


def _model_input_frame(model, features, all_names):
    """按模型自身记录的特征名构造预测输入（v1/v2/v3 三代特征集统一入口）。

    在 _aligned_feature_frame（维度一致即按名包装）之上增加**列子集对齐**：
    v2 精简特征集（2026-08-24 特征审计，22→15）训练的模型记录 15 个真实
    特征名，而部署端 extract_features_with_interactions 仍产出 22 维向量
    ——此处按名从全集矩阵选列，模型与特征空间解耦。

    四类 checkpoint 行为：
    1. v1 全名档（22 名）→ 等价于 _aligned_feature_frame（按名包装）；
    2. v2 子集档（15 名 ⊆ all_names）→ 从 22 列矩阵按名选 15 列
       （列顺序按模型记录，数值取自对应位置）；
    3. v3 网络特征档（26 名，11 网络列 ∉ all_names）→ 个体列按名选 +
       网络列补零（谱系降级部署语义：网络列源字段仅合成验证层存在，
       部署谱系缺列 = 死列，与训练端 _select_feature_columns_v3 同
       语义——v3 模型在 22 维部署向量上的行为 = v2）；
    4. 旧 numpy 档（无 feature_names_in_ / Column_N 自动名）→
       维度一致按名包装或原样返回（行为不变）。

    H-ML3（特征契约校验）：选列前先校验模型名集与当前代码的审计契约
    （feature_audit.SELECTED_FEATURES_V2 / V3 / v1 22 维全集）一致。
    若不一致（手改特征集 / 换代码版本训练的旧档），预测端静默适配会
    绕过审计结论——此处按名集等值校验并告警（每个名集只告警一次，
    不中断预测：旧 checkpoint 兼容性优先，但违规使用可被日志追溯）。
    """
    if not PANDAS_AVAILABLE:
        return features
    names = getattr(model, 'feature_names_in_', None)
    if names is None:
        return features
    arr = np.asarray(features)
    if arr.ndim != 2:
        return features
    names = list(names)
    _check_feature_contract(model, names, all_names)
    try:
        # 全集矩阵（列数 = all_names）+ 模型真实名（v1 全名或 v2 子集）
        if arr.shape[1] == len(all_names) and set(names) <= set(all_names):
            frame = pd.DataFrame(arr, columns=list(all_names))
            return frame[names]
        # v3 谱系感知降级：模型名部分在全集、部分不在——**仅限 v3 网络
        # 特征列**（合成验证层专属列，部署谱系天然缺失，补零 = 与训练端
        # _select_feature_columns_v3 同语义）。v4 extras（cad_score/QFT
        # 等真实部署信号列）误入此路径会被静默补零 → 部署端静默降级
        # （P1 2026-09-16 收紧）：非网络列不补零，落回原样返回，由
        # 调用方响亮失败（维度不匹配异常）而非拿到静默错值。
        if arr.shape[1] == len(all_names) and \
                set(names) & set(all_names):
            _extra = set(names) - set(all_names)
            try:
                from .feature_audit import NETWORK_FEATURE_NAMES_V3
                _v3_only = _extra <= set(NETWORK_FEATURE_NAMES_V3)
            except ImportError:
                _v3_only = True  # feature_audit 不可用：维持存量 v3 语义
            if _v3_only:
                frame = pd.DataFrame(arr, columns=list(all_names))
                for col in names:
                    if col not in all_names:
                        frame[col] = 0.0
                return frame[names]
        # 维度一致：旧档（Column_N）或训练一致矩阵
        if len(names) == arr.shape[1]:
            return pd.DataFrame(arr, columns=names)
        return features
    except Exception:
        return features


def _model_feature_names(model):
    """模型记录的特征名列表（无记录返回 []）。

    sklearn/XGBoost/LightGBM 写 sklearn 约定的 ``feature_names_in_``
    （ndarray，不可用 ``or`` 真值合并——数组真值歧义）；CatBoost 只写
    自有 ``feature_names_`` 不写约定名。双属性显式判 None 读取
    （同 transfer_learning 的读法）。
    """
    fn = getattr(model, 'feature_names_in_', None)
    if fn is None:
        fn = getattr(model, 'feature_names_', None)
    return [str(x) for x in fn] if fn is not None else []


def _v4_row(predictor, contact_data, base_features, names):
    """v4 列集的单样本构造（extras 从原始字段 + 训练中位数，列序对齐）。

    共享核心（分类模型 _v4_input_frame 与生存模型 survival_cox 输入
    同一构造）：extras 由 cohort spec（单一真值源 ml/cohort_features.py）
    从 contact_data 原始字段构造——哨兵清洗/指示器/onehot 同训练口径，
    缺失值用训练中位数（predictor.v4_fill_medians，随 checkpoint 落盘，
    绝不在部署单样本上重算），列集按训练清单 reindex（基类 onehot 列
    补 0、部署新类别列丢弃 = 基类语义）。

    返回 DataFrame（按 names 列序）；None = 名集不匹配 v4 三件套或
    构造失败，调用方自行回退。
    """
    if not PANDAS_AVAILABLE:
        return None
    v4_names = list(getattr(predictor, 'v4_feature_names', []) or [])
    if not v4_names or set(names) != set(v4_names):
        return None
    cohort = getattr(predictor, 'v4_cohort', None)
    if not cohort:
        LOGGER.warning(
            "v4 列集输入构造要求 predictor.v4_cohort（异常 "
            "checkpoint？），无法构造 extras")
        return None
    try:
        from .cohort_features import build_v4_extras_for_prediction
        base_names = list(predictor.ALL_FEATURE_NAMES)
        extras_expected = [c for c in v4_names
                           if c not in set(base_names)]
        X_extra, _audit = build_v4_extras_for_prediction(
            contact_data, cohort, extras_expected,
            getattr(predictor, 'v4_fill_medians', {}) or {})
        base = np.asarray(base_features, dtype=float).reshape(1, -1)
        if base.shape[1] != len(base_names):
            return None
        X = np.hstack([base, X_extra])
        return pd.DataFrame(X, columns=v4_names)[list(names)]
    except Exception as e:
        LOGGER.warning("v4 预测输入构造失败（%s）", e)
        return None


def _v4_input_frame(predictor, model, contact_data, base_features):
    """v4（族感知）模型的预测输入：22 维基础 + extras 严格对齐训练列集。

    判定：模型 feature_names_in_ 与 predictor.v4_feature_names 名集
    相同 → v4 模型（混合谱系场景下 v1 模型名集不匹配，自动落回
    通用路径）。构造逻辑见 _v4_row（共享核心）。

    返回 DataFrame（按模型记录列序）；None = 非 v4 模型或三件套缺失，
    调用方回退 _model_input_frame 通用路径（该路径对 v4 extras 不补零，
    维度不匹配将响亮失败而非静默错值）。
    """
    if not PANDAS_AVAILABLE:
        return None
    names = _model_feature_names(model)
    if not names:
        return None
    return _v4_row(predictor, contact_data, base_features, names)


def predict_risk(predictor, contact_data, contact_type='family',
                 patient_context=None, allow_research=False):
        """预测接触者的结核病风险概率

        参数：
            contact_data (dict): 接触者数据字典
            contact_type (str): 'family' 或 'social'
            patient_context (dict): 可选的患者上下文信息，包含
                patient_ftd, patient_cough_freq, contact_count。
                若提供则临时覆盖实例属性。
            allow_research (bool): 显式放行研究资产模型（P1-c 护栏，
                2026-09-20）。treats 等降级队列的 checkpoint（部署
                价值不足，判决见 docs/model_boundary_clinical_validation
                .md §5）默认禁止走部署预测入口——本方法被 GUI/api/
                deploy_services 等部署面共用，在此单点拦截；研究用途
                （复验/审计脚本）须显式传 True，误部署则抛错。

        返回：
            dict: 各模型的预测概率和综合评估，键为模型名，值为包含
                  risk_probability(float)、risk_class(int)、model_name(str)的字典；
                  另有'ensemble'键提供集成评估结果
        """
        # P1-c（treats 降级机制护栏）：research_only 模型（降级队列
        # checkpoint，训练时标记或加载时旧档回填）禁止走部署预测入口
        if getattr(predictor, 'deployment_status', None) == 'research_only' \
                and not allow_research:
            from ...constants import (RESEARCH_ONLY_DEPLOYMENT_ERROR,
                                      RESEARCH_ONLY_COHORTS)
            _cohort = getattr(predictor, 'v4_cohort', None) or '未知队列'
            raise ValueError(RESEARCH_ONLY_DEPLOYMENT_ERROR.format(
                cohort=_cohort,
                reason=getattr(
                    predictor, 'research_only_reason',
                    RESEARCH_ONLY_COHORTS.get(_cohort, {}).get('reason',
                                                               ''))))
        if not predictor.is_trained:
            return None
        p_ftd = patient_context.get('patient_ftd', predictor.patient_ftd) if patient_context else predictor.patient_ftd
        p_cough = patient_context.get('patient_cough_freq', predictor.patient_cough_freq) if patient_context else predictor.patient_cough_freq
        p_count = patient_context.get('contact_count', predictor.contact_count) if patient_context else predictor.contact_count
        
        features = predictor.extract_features_with_interactions(
            contact_data, contact_type,
            patient_ftd=p_ftd, patient_cough_freq=p_cough, contact_count=p_count
        )
        
        predictions = {}
        for model_name, model in predictor.models.items():
            try:
                # v4（族感知）模型优先走专用构造（extras 从 contact_data
                # 原始字段 + 训练中位数）；非 v4 模型/构造失败 → 通用路径
                X_pred = _v4_input_frame(predictor, model, contact_data,
                                          features)
                if X_pred is None:
                    X_pred = _model_input_frame(model, features,
                                                predictor.ALL_FEATURE_NAMES)
                prob = model.predict_proba(X_pred)[0]
                predictions[model_name] = {
                    'risk_probability': float(prob[1]) * 100,
                    'risk_class': int(model.predict(X_pred)[0]),
                    'model_name': predictor.model_performance[model_name]['name'],
                    'model_name_en': predictor.model_performance[model_name]['name_en']
                }
            except (ValueError, RuntimeError, OSError, IndexError) as e:
                LOGGER.warning('模型 %s 预测失败，使用零值回退: %s', model_name, e)
                predictions[model_name] = {
                    'risk_probability': 0.0,
                    'risk_class': 0,
                    'model_name': predictor.model_performance.get(model_name, {}).get('name', model_name),
                    'model_name_en': predictor.model_performance.get(model_name, {}).get('name_en', model_name)
                }
        
        if predictions:
            # Stacking 集成（优先）或简单平均
            if predictor._is_stacking_trained and predictor.stacking_meta_model is not None:
                try:
                    # 按基模型顺序收集预测概率（0-1 范围）
                    base_probs = np.array([
                        predictions[name]['risk_probability'] / 100.0
                        for name in predictor.stacking_model_names
                        if name in predictions
                    ]).reshape(1, -1)
                    stacking_prob = predictor.stacking_meta_model.predict_proba(base_probs)[0, 1]
                    ensemble_prob = float(stacking_prob) * 100.0
                    ensemble_method = 'stacking_logreg'
                except (ValueError, RuntimeError, OSError, IndexError) as e:
                    LOGGER.warning("Stacking 预测失败，回退到简单平均: %s", e)
                    avg_prob = np.mean([p['risk_probability'] for p in predictions.values()])
                    ensemble_prob = float(avg_prob)
                    ensemble_method = 'mean_fallback'
            else:
                avg_prob = np.mean([p['risk_probability'] for p in predictions.values()])
                ensemble_prob = float(avg_prob)
                ensemble_method = 'mean'

            best_model_name = max(predictor.model_performance, key=lambda k: predictor.model_performance[k]['AUROC'])
            best_prob = predictions.get(best_model_name, {}).get('risk_probability', 0.0)
            predictions['ensemble'] = {
                'risk_probability': float(ensemble_prob),
                'risk_class': 1 if ensemble_prob > DEFAULT_RISK_CLASS_THRESHOLD else 0,
                'best_model': predictor.model_performance[best_model_name]['name'],
                'best_model_probability': float(best_prob),
                'best_model_auc': predictor.model_performance[best_model_name]['AUROC'],
                'ensemble_method': ensemble_method,
            }
            # P2（小队列部署模型路由）：训练时路由决策（n<1000 或 v4
            # extras 在场 → LR 优先）随 checkpoint 恢复——部署端报
            # preferred_model；best_model 保留原始 max-CV 口径供对照
            _selector = getattr(predictor, 'deployment_model', None)
            if _selector and _selector.get('preferred_key') in predictions:
                _pk = _selector['preferred_key']
                predictions['ensemble']['preferred_model'] = \
                    predictor.model_performance[_pk]['name']
                predictions['ensemble']['preferred_model_probability'] = \
                    predictions[_pk]['risk_probability']
                predictions['ensemble']['preferred_model_rule'] = \
                    _selector.get('rule')
            if predictor._is_stacking_trained:
                predictions['ensemble']['stacking_weights'] = predictor.stacking_weights

        # P5（生存路径转正，2026-09-16）：Cox PH 线性预测子——
        # follow_up_days 数据（peru_mdr）的 canonical 模型随 checkpoint
        # 恢复；输入构造与分类 v4 模型同一核心（_v4_row：extras 从
        # 原始字段 + 训练中位数），22 维（A 版）走基础特征直通。
        _surv = getattr(predictor, 'survival_model', None)
        if _surv is not None:
            try:
                _s_names = list(getattr(_surv, 'feature_names', []) or [])
                _base_names = list(predictor.ALL_FEATURE_NAMES)
                _X_s = None
                if set(_s_names) == set(_base_names):
                    # A 版（22 维）：基础特征直通
                    _X_s = pd.DataFrame(
                        np.asarray(features, dtype=float).reshape(1, -1),
                        columns=_base_names)[_s_names]
                else:
                    # B 版（v4 列集）：extras 单样本构造（训练中位数）
                    _X_s = _v4_row(predictor, contact_data, features,
                                   _s_names)
                if _X_s is not None:
                    _score = float(np.asarray(
                        _surv.predict(_X_s.to_numpy(dtype=float)),
                        dtype=float).ravel()[0])
                    predictions['survival_cox'] = {
                        'risk_score': _score,
                        'model_name': 'Cox PH（生存路径）',
                        'interpretation': (
                            'Cox 线性预测子（log-HR，相对训练队列参照）：'
                            '越高风险越高；排序分值而非绝对概率——'
                            '分窗绝对风险需基线累积风险表，判别部署用'
                            '排序即可'),
                        'time_column': getattr(_surv, 'time_column', None),
                        'cohort': getattr(_surv, 'cohort', None),
                    }
            except Exception as e:
                LOGGER.warning('生存模型预测失败（跳过 survival_cox）: %s', e)

        return predictions


# ============================================================
# 静态方法（从 MLRiskPredictor 拆分而来，不依赖 predictor 实例）
# ============================================================

def _ensemble_predict(models, X):
    """集成预测（简单平均）"""
    import numpy as np
    if not models:
        return np.zeros(len(X))
    preds = np.zeros(len(X))
    for model in models.values():
        preds += model.predict_proba(
            _aligned_feature_frame(model, X))[:, 1]
    return preds / len(models)


def _safe_roc_auc_score(y_true, y_score):
    """安全计算 AUROC：单类时返回 0.5 而非触发 UndefinedMetricWarning"""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision_score(y_true, y_score):
    """安全计算 AUPRC：单类时返回正类比例而非触发 UndefinedMetricWarning"""
    import numpy as np
    from sklearn.metrics import average_precision_score
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_score))


def _wilcoxon_signed_rank(diff):
    """Wilcoxon 符号秩检验（无外部依赖实现，正态近似）"""
    import numpy as np
    diff = diff[np.abs(diff) > 1e-10]
    n = len(diff)
    if n == 0:
        return 0.0, 1.0

    abs_diff = np.abs(diff)
    ranks = np.zeros(n)
    order = np.argsort(abs_diff)
    i = 0
    while i < n:
        j = i
        while j < n and abs_diff[order[j]] == abs_diff[order[i]]:
            j += 1
        rank_val = (i + j + 2) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank_val
        i = j

    W_plus = np.sum(ranks[diff > 0])
    W_minus = np.sum(ranks[diff < 0])
    W = min(W_plus, W_minus)
    mean_W = n * (n + 1) / 4.0
    unique_vals, counts = np.unique(abs_diff, return_counts=True)
    tie_correction = np.sum(counts * (counts ** 2 - 1)) / 48.0
    std_W = np.sqrt(max(n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction, 1e-10))
    z = (W - mean_W) / std_W
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + np.erf(np.abs(z) / np.sqrt(2))))

    return float(W), float(p_value)


def _normalize_illness_type(illness_type):
    """将中文/英文疾病类型统一转换为标准英文键

    参数：
        illness_type (str): 原始疾病类型字符串，支持中英文

    返回：
        str: 标准化的英文键（'none', 'hiv', 'diabetes', 'immunosuppressants', 'other'）
    """
    if illness_type is None:
        return 'none'
    illness_type = str(illness_type).strip().lower()
    mapping = {
        'none': 'none', '无': 'none',
        'hiv': 'hiv', '艾滋病': 'hiv',
        'diabetes': 'diabetes', '糖尿病': 'diabetes', 'dm': 'diabetes',
        'immunosuppressants': 'immunosuppressants', '免疫抑制': 'immunosuppressants',
        '免疫抑制剂': 'immunosuppressants',
        'other': 'other', '其他': 'other', '其它': 'other'
    }
    return mapping.get(illness_type, 'other')
    

