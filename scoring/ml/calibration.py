#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""概率校准：Isotonic/Sigmoid校准、Stacking集成

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



# ============================================================
# 以下函数从 MLRiskPredictor 的实例方法拆分而来
# 每个函数接收 predictor 实例作为第一个参数
# ============================================================

def compute_calibration(predictor, n_bins=10, random_state=42):
        """计算模型校准曲线和Brier评分
        
        校准曲线反映预测概率与实际频率的一致性，
        Brier评分衡量概率预测的准确性（越低越好）。
        
        参数：
            n_bins (int): 校准曲线分箱数，默认10
            random_state (int): 随机种子
        
        返回：
            dict: 各模型的校准数据，包含mean_predicted_value、
                  fraction_of_positives、brier_score
        """
        if not SKLEARN_AVAILABLE or not predictor.is_trained:
            return None
        
        try:
            from sklearn.calibration import calibration_curve
            
            X, y = predictor._generate_synthetic_training_data(2000, random_state)
            
            calibration_results = {}
            for model_name, model in predictor.models.items():
                try:
                    y_prob = model.predict_proba(X)[:, 1]
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y, y_prob, n_bins=n_bins, strategy='uniform'
                    )
                    brier = brier_score_loss(y, y_prob)
                    
                    calibration_results[model_name] = {
                        'mean_predicted_value': mean_predicted_value,
                        'fraction_of_positives': fraction_of_positives,
                        'brier_score': brier,
                        'name': predictor.model_performance.get(model_name, {}).get('name', model_name),
                        'name_en': predictor.model_performance.get(model_name, {}).get('name_en', model_name)
                    }
                except (ValueError, RuntimeError, OSError) as e:
                    LOGGER.warning('模型 %s 校准曲线计算失败: %s', model_name, e)
                    continue

            return calibration_results

        except Exception as e:
            LOGGER.warning("校准曲线计算失败: %s", e, exc_info=True)
            return None


def calibrate_models(predictor, n_samples=2000, random_state=42, method='auto'):
        """合成数据概率校准（H-ML4 遗留入口，委托 calibrate_models_on_data）

        H-ML4（API 对齐）修复：旧实现默认 isotonic + 全量合成数据拟合，
        无方法自动选择（阳性 < 200 应走 sigmoid 的项目硬约束）、无分层
        对半切分的泄漏无关评估、无校准退化拦截，与项目实际校准路径
        calibrate_models_on_data 行为分叉，易误用。现本入口仅生成合成
        数据后委托 calibrate_models_on_data，两条路径语义统一：

          - method='auto' 默认（阳性 < 200 → sigmoid，否则 isotonic）
          - 分层对半切分泄漏无关评估（brier/auroc 前后留档）
          - AUROC 退化拦截（降幅 > CALIBRATION_DEGRADATION_DELTA 回退）
          - calibration_eval 留档（决策参考表输入）

        注意（项目硬约束）：合成数据校准结果不得作为性能证据；生产部署
        应优先用真实数据调用 calibrate_models_on_data。

        参数：
            n_samples (int): 合成校准数据样本量，默认2000
            random_state (int): 随机种子
            method (str): 校准方法，'auto'（默认）/ 'sigmoid' / 'isotonic'

        返回：
            bool: 校准是否成功

        文献：
            Platt J. (1999) Probabilistic outputs for SVMs.
            Zadrozny B, Elkan C. (2002) Transforming classifier scores.
        """
        if not SKLEARN_AVAILABLE or not predictor.is_trained:
            return False

        LOGGER.warning(
            'calibrate_models（合成数据遗留入口，H-ML4 对齐）：校准基于'
            '合成数据，校准质量不构成性能证据；规范路径是'
            'calibrate_models_on_data（真实数据）。本入口已对齐 auto 方法'
            '选择 + 校准退化拦截语义。')

        try:
            X, y = predictor._generate_synthetic_training_data(
                n_samples, random_state)
        except Exception as e:
            LOGGER.warning('合成校准数据生成失败: %s', e)
            return False

        return calibrate_models_on_data(predictor, X, y, method=method,
                                        random_state=random_state)


def train_stacking_ensemble(predictor, n_samples=2000, random_state=42,
                            X=None, y=None):
        """训练 Stacking 集成（Out-of-Fold 预测 + Logistic Regression 元模型）

        流程：
        1. 对每个基模型使用 5 折交叉验证生成 OOF 预测
        2. 将 OOF 预测作为元特征，训练 Logistic Regression 元模型
        3. 元模型自动学习各基模型的最优权重

        M 级审计修复：原实现固定用合成数据训练元模型，与真实数据校准
        路径（calibrate_models_on_data）不同源。现支持显式传入真实数据
        (X, y)；未传时保留合成回退但记录 _stacking_data_source 标签
        并告警——合成口径的 stacking 权重只反映复现评分引擎，不得
        作为真实泛化证据引用。

        文献：
            Wolpert DH. (1992) Stacked generalization. Neural Networks 5(2):241-259.
            Breiman L. (1996) Stacked regressions. Machine Learning 24(1):49-64.

        参数：
            n_samples (int): 合成训练数据样本量（仅 X/y 未传时使用）
            random_state (int): 随机种子
            X (array-like|None): 真实训练特征矩阵（优先）
            y (array-like|None): 真实训练标签（与 X 配对）

        返回：
            bool: 训练是否成功
        """
        if not SKLEARN_AVAILABLE or not predictor.is_trained:
            return False

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_predict

            if X is not None and y is not None:
                X = np.asarray(X)
                y = np.asarray(y)
                data_source = 'provided'
            else:
                X, y = predictor._generate_synthetic_training_data(
                    n_samples, random_state)
                data_source = 'synthetic'
                LOGGER.warning(
                    'train_stacking_ensemble 未传入真实数据，回退合成'
                    '数据训练元模型（口径=synthetic）——该 stacking 权重'
                    '只反映复现评分引擎，不构成真实泛化性能证据')
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

            # 第一阶段：为每个基模型生成 OOF 预测
            oof_predictions = np.zeros((X.shape[0], len(predictor.models)))
            model_names = list(predictor.models.keys())

            for i, model_name in enumerate(model_names):
                model = predictor.models[model_name]
                try:
                    # 使用校准后的模型（如果有）或原始模型
                    base_model = predictor.calibrated_models.get(
                        model_name, model) if hasattr(predictor, 'calibrated_models') else model
                    oof_pred = cross_val_predict(
                        base_model, X, y, cv=cv, method='predict_proba'
                    )[:, 1]  # 取正类概率
                    oof_predictions[:, i] = oof_pred
                except (ValueError, RuntimeError, OSError) as e:
                    LOGGER.warning("模型 %s OOF 预测失败: %s", model_name, e)
                    # 回退：使用模型自身预测
                    oof_predictions[:, i] = model.predict_proba(X)[:, 1]

            # 第二阶段：训练 Logistic Regression 元模型
            predictor.stacking_meta_model = LogisticRegression(
                penalty='l2', C=1.0, solver='lbfgs',
                max_iter=1000, random_state=random_state
            )
            predictor.stacking_meta_model.fit(oof_predictions, y)

            # 记录元模型权重（逻辑回归系数归一化）
            coef = predictor.stacking_meta_model.coef_[0]
            coef_abs = np.abs(coef)
            if coef_abs.sum() > 0:
                predictor.stacking_weights = {
                    name: float(coef[i] / coef_abs.sum())
                    for i, name in enumerate(model_names)
                }
            else:
                predictor.stacking_weights = {
                    name: 1.0 / len(model_names)
                    for name in model_names
                }

            predictor.stacking_model_names = model_names
            predictor._is_stacking_trained = True
            predictor._stacking_data_source = data_source
            LOGGER.debug("Stacking 集成训练完成，权重: %s (数据口径: %s)",
                         predictor.stacking_weights, data_source)
            return True

        except Exception as e:
            LOGGER.warning("Stacking 集成训练失败: %s", e)
            return False


# ============================================================
# 问题4（校准侧）：真实数据校准，接入训练流程
# ============================================================

# 校准方法自动选择的阳性样本阈值（项目硬约束：阳性 < 200 用 sigmoid，
# 避免 isotonic 在极端不平衡下塌缩为常数预测；2026-09-05 地基审计：
# 此前为 500 与约束不符，200-499 区间误用 sigmoid，已对齐记忆约束）
CALIBRATION_SIGMOID_MAX_POSITIVES = 200

# 校准退化拦截阈值（AUROC 绝对降幅）：排序能力退化超过该值时
# 回退未校准模型并打标，防止静默性能坍缩（校准改善概率质量，
# 但 CalibratedClassifierCV 内部 CV 重拟合可能损害排序）。
CALIBRATION_DEGRADATION_DELTA = 0.01


def calibrate_models_on_data(predictor, X, y, method='auto', random_state=42):
    """用真实训练数据对基模型做概率校准（问题4：校准侧接入训练流程）

    与 calibrate_models（合成数据）不同，本函数使用真实训练数据：

    1. 分层对半切分：校准半区训练 CalibratedClassifierCV，另一半区
       泄漏无关地评估校准前后的 Brier 与 AUROC（校准可能损失判别力，
       前后都记录以检出）。前后对照同口径（缺陷2修复 2026-09-16）：
       before = cal 半区按同一 cv 折重拟合的未校准克隆的 K 折平均
       预测（复刻 CalibratedClassifierCV ensemble=True 的内部折
       结构），after = 校准器包裹同一折结构的 K 折平均——两臂对
       eval 半区均泄漏无关，差异只来自校准器本身。旧实现 before
       用全量数据训练的基模型直接预测 eval 半区（eval 样本已入训
       练，in-sample 乐观口径），树模型记忆化使 AUROC 虚高
       0.03-0.08，乐观差距被误判为校准退化（kenya run 2026-09-16
       实测：五个树模型全部被误拦截，auroc_after 与 CV OOF 一致，
       logistic 因 in-sample≈OOF 未触发）。
    2. method='auto'：阳性样本 < 200 → sigmoid（Platt）；≥ 200 →
       isotonic（项目硬约束，Platt 在小样本更稳健）；
    3. 每模型把校准元信息写入
       predictor.model_performance[name]['calibration']：
       {method, n_positive, brier_before/after, auroc_before/after}；
    4. 保留评估半区概率 predictor.calibration_eval[name] =
       {y_test, p_after, leak_free: True}，供决策参考表
       （core.ppv.build_decision_reference）使用。

    Args:
        predictor: MLRiskPredictor（或满足 models/model_performance/
                   is_trained 协议的对象）
        X: 特征矩阵（numpy / DataFrame）
        y: 标签（0/1）
        method: 'auto' / 'sigmoid' / 'isotonic'
        random_state: 随机种子（项目全局约定 42）

    Returns:
        bool: 校准是否成功

    文献：
        Platt J. (1999) Probabilistic outputs for SVMs.
        Zadrozny B, Elkan C. (2002) Transforming classifier scores.
        Niculescu-Mizil A, Caruana R. (2005) ICML: 小样本 sigmoid
        优于 isotonic（isotonic 在数据不足时过拟合）。
    """
    if not SKLEARN_AVAILABLE or not predictor.is_trained:
        return False

    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedShuffleSplit

        # 特征名对齐（LightGBM checkpoint 记录 feature_names_in_ 时
        # 避免 numpy 预测触发 sklearn 警告；见 evaluation._aligned_feature_frame）
        try:
            from .evaluation import _aligned_feature_frame
        except ImportError:
            _aligned_feature_frame = None

        def _predict_positive(model, X_block):
            if _aligned_feature_frame is not None:
                try:
                    return model.predict_proba(
                        _aligned_feature_frame(model, X_block))[:, 1]
                except Exception:
                    pass
            return model.predict_proba(X_block)[:, 1]

        X_arr = np.asarray(X)
        y_arr = np.asarray(y).astype(int)
        n_positive = int(y_arr.sum())
        if n_positive == 0 or n_positive == len(y_arr):
            LOGGER.warning('校准需要两类样本，跳过（阳性 %s/%s）',
                           n_positive, len(y_arr))
            return False

        if method == 'auto':
            method = ('sigmoid' if n_positive < CALIBRATION_SIGMOID_MAX_POSITIVES
                      else 'isotonic')
        elif method not in ('sigmoid', 'isotonic'):
            raise ValueError(f'未知校准方法: {method}（须为 auto/sigmoid/isotonic）')

        # P1-a（特征名告警清理，2026-09-20）：校准输入统一转带列名的
        # DataFrame 再切分。旧实现 ndarray 直入：LGBM sklearn 包装器在
        # ndarray 拟合时写入自动特征名（Column_N），CalibratedClassifierCV
        # 内部折预测 + calibrated.predict_proba 各触发一条告警（实测每模型
        # 10 条/5 折）。列名优先序：X 自带列名 > 基模型 feature_names_in_
        # （同维）> x0..xN 生成名；特征名不参与数值计算，before/after 两臂
        # 结果逐位不变。pandas 不可用时维持 ndarray（告警仍在，行为不变）。
        X_named = X_arr
        if PANDAS_AVAILABLE and X_arr.ndim == 2:
            if hasattr(X, 'columns') and X.shape[1] == X_arr.shape[1]:
                cols = list(X.columns)
            else:
                cols = None
                for _m in getattr(predictor, 'models', {}).values():
                    _names = getattr(_m, 'feature_names_in_', None)
                    if _names is not None and len(_names) == X_arr.shape[1]:
                        cols = list(_names)
                        break
                if cols is None:
                    cols = [f'x{i}' for i in range(X_arr.shape[1])]
            X_named = pd.DataFrame(X_arr, columns=cols)

        # 分层对半切分：校准半区 / 泄漏无关评估半区
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=random_state)
        cal_idx, eval_idx = next(splitter.split(X_arr, y_arr))
        X_cal, y_cal = X_named.iloc[cal_idx], y_arr[cal_idx]
        X_eval, y_eval = X_named.iloc[eval_idx], y_arr[eval_idx]

        if len(set(y_eval.tolist())) < 2:
            LOGGER.warning('评估半区只有单类，无法计算 AUROC，跳过校准')
            return False

        # 交叉验证折数受校准半区少数类样本数约束
        min_class = int(min(np.bincount(y_cal)))
        n_splits = max(2, min(5, min_class))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=random_state)

        predictor.calibrated_models = {}
        predictor.calibration_eval = {}
        for model_name, model in predictor.models.items():
            try:
                calibrated = CalibratedClassifierCV(
                    model, method=method, cv=cv)
                calibrated.fit(X_cal, y_cal)

                # 泄漏无关未校准基线（缺陷2修复）：CalibratedClassifierCV
                # (ensemble=True) 在 cal 半区按 cv 折重拟合 K 个基模型并
                # 对预测取平均；对照基线用同一 cv 折结构重拟合未校准克隆
                # 取平均——before/after 两臂仅差校准器，对 eval 半区同口径
                # 泄漏无关。旧实现用全量数据训练的基模型（含 eval 样本）
                # 直接预测，in-sample 乐观 AUROC 把树模型记忆化差距误判
                # 为校准退化。
                # 口径注记：GB 的 balanced sample_weight 是 fit 时参数，
                # 克隆重拟合与 CalibratedClassifierCV 内部重拟合均不带
                # （两臂一致，不影响前后可比性）；RF/LGBM/CatBoost 的
                # class_weight 与 XGBoost 的 scale_pos_weight 是构造参数，
                # clone 保留。
                from sklearn.base import clone
                p_before = np.zeros(len(y_eval), dtype=float)
                n_folds_used = 0
                for fold_tr, _ in cv.split(X_cal, y_cal):
                    ref = clone(model)
                    ref.fit(X_cal.iloc[fold_tr], y_cal[fold_tr])
                    p_before += _predict_positive(ref, X_eval)
                    n_folds_used += 1
                p_before /= max(n_folds_used, 1)
                p_after = calibrated.predict_proba(X_eval)[:, 1]

                perf = predictor.model_performance.setdefault(model_name, {})
                auroc_before = float(roc_auc_score(y_eval, p_before))
                auroc_after = float(roc_auc_score(y_eval, p_after))

                # 校准退化拦截：AUROC 降幅超过容差时回退未校准模型
                # （防止静默性能坍缩——只记录不拦截等于把退化放行到部署）
                degraded = (auroc_after <
                            auroc_before - CALIBRATION_DEGRADATION_DELTA)
                perf['calibration'] = {
                    'method': method,
                    'n_positive': n_positive,
                    'n_samples': int(len(y_arr)),
                    'before_caliber': 'cal_half_kfold_leak_free',
                    'brier_before': float(brier_score_loss(y_eval, p_before)),
                    'brier_after': float(brier_score_loss(y_eval, p_after)),
                    'auroc_before': auroc_before,
                    'auroc_after': auroc_after,
                    'leak_free_eval': True,
                    'calibration_degraded': degraded,
                    'degradation_delta': CALIBRATION_DEGRADATION_DELTA,
                }
                if degraded:
                    LOGGER.warning(
                        '模型 %s 校准退化拦截（同口径对照 cal 半区 K 折基线）：'
                        'AUROC %.4f→%.4f（降幅 %.4f > '
                        '容差 %.4f），回退未校准模型',
                        model_name, auroc_before, auroc_after,
                        auroc_before - auroc_after,
                        CALIBRATION_DEGRADATION_DELTA)
                    predictor.calibrated_models[model_name] = model
                else:
                    predictor.calibrated_models[model_name] = calibrated
                # 评估半区概率留档：供决策参考表（排序+截断点+双口径 PPV）使用
                predictor.calibration_eval[model_name] = {
                    'y_test': y_eval.tolist(),
                    'p_after': p_after.tolist(),
                    'p_before': p_before.tolist(),
                    'leak_free': True,
                }
                LOGGER.debug('模型 %s 真实数据校准完成 (method=%s, '
                             'Brier %.4f→%.4f, AUROC %.4f→%.4f)',
                             model_name, method,
                             perf['calibration']['brier_before'],
                             perf['calibration']['brier_after'],
                             perf['calibration']['auroc_before'],
                             perf['calibration']['auroc_after'])
            except (ValueError, RuntimeError, OSError) as e:
                LOGGER.warning('模型 %s 真实数据校准失败: %s', model_name, e)
                predictor.calibrated_models[model_name] = model  # 回退未校准
                continue

        predictor._is_calibrated = True
        predictor._calibration_method = method
        return True

    except Exception as e:
        LOGGER.warning('真实数据概率校准失败: %s', e)
        return False
    

