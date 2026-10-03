#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML 预测运行器 — 封装机器学习预测循环逻辑

将 ML 预测、GNN 预测、集成预测和 SHAP 分析的核心循环从 GUI 控制器中
分离出来，使 core 层可独立运行 ML 预测而不依赖 tkinter。

调用方式：
    from .core.ml_runner import run_ml_prediction_loop
    ml_results = run_ml_prediction_loop(assessment, ml_predictor, integrator, results, data_lock)
"""

import copy
import logging

LOGGER = logging.getLogger(__name__)


def run_ml_prediction_loop(assessment, ml_predictor, integrator, results, data_lock):
    """对 results 中的所有接触者运行 ML/GNN/Ensemble 预测和 SHAP 分析。

    解耦说明：assessment 在此仅作为 GNN 图构建(build_from_assessment)与
    integrator 回调(_get_seir_results 读取 results)的只读运行时上下文，
    本函数不对其做任何初始化式 mutate。完整窄化为独立 GNNContext 数据对象
    需级联重构 build_from_assessment，暂不在本次解耦范围内。

    Args:
        assessment: 评估上下文（只读，供 GNN 图构建与 integrator 回调使用）
        ml_predictor: MLRiskPredictor 实例
        integrator: ThreeDirectionIntegrator 实例
        results: 评估结果字典（包含 potential_patients）
        data_lock: threading.Lock，用于原子性读取 results

    Returns:
        dict: ml_results_local 字典，包含 family/social/gnn/ensemble/shap_analysis
              若 ml_predictor 为 None 或未训练则返回 None
    """
    if ml_predictor is None:
        return None

    if not ml_predictor.is_trained:
        return None

    ml_results_local = {
        'family': [],
        'social': [],
        'model_performance': dict(ml_predictor.model_performance),
        'training_sample_count': ml_predictor.training_sample_count,
        'gnn': {'family': [], 'social': []},
        'ensemble': {'family': [], 'social': []},
        'attribution': {'family': [], 'social': []},
    }

    # 原子性读取 results 快照，避免与评估线程写入竞态
    with data_lock:
        results_snapshot = copy.deepcopy(results) if results else {}

    if not results_snapshot or 'potential_patients' not in results_snapshot:
        return ml_results_local

    potential = results_snapshot['potential_patients']

    # ---- 家庭接触者预测 ----
    for member in potential.get('family', []):
        try:
            ml_pred = ml_predictor.predict_risk(member, 'family')
            if ml_pred:
                ml_results_local['family'].append({
                    'name': member.get('name', '未知'),
                    'traditional_prob': member.get('disease_probability', 0),
                    'ml_predictions': ml_pred,
                })

            # GNN 预测
            try:
                gnn_pred = ml_predictor.predict_gnn_risk(assessment, member, 'family')
                if gnn_pred:
                    ml_results_local['gnn']['family'].append({
                        'name': member.get('name', '未知'),
                        'traditional_prob': member.get('disease_probability', 0),
                        'gnn_prediction': gnn_pred,
                    })
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                LOGGER.error("GNN预测失败: %s", e, exc_info=True)

            # GNN 图归因（哪些节点/边影响判定最大）
            gnn_explanation = None
            if ml_predictor.gnn_is_trained:
                try:
                    gnn_explanation = ml_predictor.compute_gnn_explanation(
                        assessment, member, 'family')
                except (ValueError, RuntimeError, TypeError, KeyError) as e:
                    LOGGER.error("GNN归因失败: %s", e, exc_info=True)

            # 逐因素 SHAP 归因（延迟到统一阶段计算，此处先占位）
            ml_results_local['attribution']['family'].append({
                'name': member.get('name', '未知'),
                'type': 'family',
                'shap_contributions': None,
                'gnn_explanation': gnn_explanation,
            })

            # 三方向集成预测
            try:
                ensemble_pred = integrator.integrate_predictions(
                    ml_predictor, assessment, member, 'family')
                if ensemble_pred and 'ensemble' in ensemble_pred:
                    ml_results_local['ensemble']['family'].append({
                        'name': member.get('name', '未知'),
                        'traditional_prob': member.get('disease_probability', 0),
                        'ensemble_predictions': ensemble_pred,
                    })
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                LOGGER.error("集成预测失败: %s", e, exc_info=True)
        except (ValueError, RuntimeError, TypeError, KeyError) as e:
            LOGGER.error("家庭接触者预测失败: %s", e, exc_info=True)

    # ---- 社会接触者预测 ----
    for contact in potential.get('social', []):
        try:
            ml_pred = ml_predictor.predict_risk(contact, 'social')
            if ml_pred:
                ml_results_local['social'].append({
                    'name': contact.get('name', '未知'),
                    'traditional_prob': contact.get('disease_probability', 0),
                    'ml_predictions': ml_pred,
                })

            # GNN 预测
            try:
                gnn_pred = ml_predictor.predict_gnn_risk(assessment, contact, 'social')
                if gnn_pred:
                    ml_results_local['gnn']['social'].append({
                        'name': contact.get('name', '未知'),
                        'traditional_prob': contact.get('disease_probability', 0),
                        'gnn_prediction': gnn_pred,
                    })
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                LOGGER.error("GNN预测失败: %s", e, exc_info=True)

            # GNN 图归因（哪些节点/边影响判定最大）
            gnn_explanation = None
            if ml_predictor.gnn_is_trained:
                try:
                    gnn_explanation = ml_predictor.compute_gnn_explanation(
                        assessment, contact, 'social')
                except (ValueError, RuntimeError, TypeError, KeyError) as e:
                    LOGGER.error("GNN归因失败: %s", e, exc_info=True)

            # 逐因素 SHAP 归因（延迟到统一阶段计算，此处先占位）
            ml_results_local['attribution']['social'].append({
                'name': contact.get('name', '未知'),
                'type': 'social',
                'shap_contributions': None,
                'gnn_explanation': gnn_explanation,
            })

            # 三方向集成预测
            try:
                ensemble_pred = integrator.integrate_predictions(
                    ml_predictor, assessment, contact, 'social')
                if ensemble_pred and 'ensemble' in ensemble_pred:
                    ml_results_local['ensemble']['social'].append({
                        'name': contact.get('name', '未知'),
                        'traditional_prob': contact.get('disease_probability', 0),
                        'ensemble_predictions': ensemble_pred,
                    })
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                LOGGER.error("集成预测失败: %s", e, exc_info=True)
        except (ValueError, RuntimeError, TypeError, KeyError) as e:
            LOGGER.error("社会接触者预测失败: %s", e, exc_info=True)

    # ---- SHAP 分析 ----
    all_contacts = []
    all_types = []
    for member in potential.get('family', []):
        all_contacts.append(member)
        all_types.append('family')
    for contact in potential.get('social', []):
        all_contacts.append(contact)
        all_types.append('social')

    if all_contacts:
        try:
            shap_result = ml_predictor.compute_shap_values(all_contacts, all_types)
            if shap_result:
                ml_results_local['shap_analysis'] = {
                    'feature_importance': shap_result['feature_importance'],
                    'feature_names': shap_result['feature_names'],
                    'feature_descriptions': shap_result['feature_descriptions'],
                    'shap_values': shap_result['shap_values'],
                    'feature_matrix': shap_result['feature_matrix'],
                }
            # 逐接触者 SHAP 因素归因（与 all_contacts 顺序对齐）
            attributions = ml_predictor.compute_contact_attributions(all_contacts, all_types)
            if attributions:
                attr_list = ml_results_local['attribution']['family'] + \
                    ml_results_local['attribution']['social']
                for idx, contributions in enumerate(attributions):
                    if idx < len(attr_list):
                        attr_list[idx]['shap_contributions'] = contributions
        except (ValueError, RuntimeError, TypeError, KeyError) as e:
            LOGGER.error("SHAP分析失败: %s", e, exc_info=True)

    return ml_results_local