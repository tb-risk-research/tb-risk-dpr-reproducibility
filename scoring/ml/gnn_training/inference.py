#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN训练 — 推理子模块。

包含 GNN 与时间感知 GNN 的风险预测函数。
从原 scoring/ml/gnn_training.py 拆分而来，不修改业务逻辑。
"""
from ._common import LOGGER, PYTORCH_AVAILABLE, PYG_AVAILABLE


def _numpy_fallback_predict(predictor, tb_assessment, contact_data, contact_type,
                            reason):
    """完整 GNN 不可用时的 NumPy 轻量图卷积回退（问题三）。

    无 torch/PyG 或模型未训练时，用纯 NumPy 图卷积保证 GNN 分支始终有
    可用输出（backend='numpy'），返回结构与完整 GNN 结果一致，便于上层
    集成器直接消费。
    """
    try:
        from ....ml.gnn import NumPyGraphRiskPredictor
    except ImportError:
        LOGGER.warning("NumPy 图卷积回退模块不可用，GNN 分支无输出")
        return None

    family_entries = getattr(tb_assessment, 'family_entries', None) or []
    social_entries = getattr(tb_assessment, 'social_entries', None) or []
    patient_info = getattr(tb_assessment, 'patient_info', None)

    predictor_np = NumPyGraphRiskPredictor()
    if family_entries or social_entries:
        result = predictor_np.predict_network_risk(
            contact_data, contact_type, patient_info,
            family_entries, social_entries)
    else:
        result = predictor_np.predict_contact_risk(contact_data, contact_type)

    if result is None:
        return None
    result['degradation'] = reason
    return result


def predict_gnn_risk(predictor, tb_assessment, contact_data, contact_type='family'):
        """使用GNN模型预测接触者风险（文献支撑：Catarcione Pinto et al., BRACIS 2025）

        参数：
            tb_assessment: 结核病风险评估实例
            contact_data (dict): 接触者数据
            contact_type (str): 接触者类型

        返回：
            dict: GNN预测结果
        """
        if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
            # 问题三：无深度学习环境时回退到 NumPy 轻量图卷积，保证 GNN 分支有输出
            reason = ('pyg_missing' if PYTORCH_AVAILABLE else 'torch_missing')
            return _numpy_fallback_predict(predictor, tb_assessment, contact_data,
                                           contact_type, reason)

        if not predictor.gnn_is_trained:
            # 问题三：torch/PyG 可用但模型未训练时，同样回退 NumPy 兜底
            return _numpy_fallback_predict(predictor, tb_assessment, contact_data,
                                           contact_type, 'not_trained')

        try:
            import torch

            # 步骤1: 从当前评估构建异质图
            hetero_data = predictor.gnn_network_builder.build_from_assessment(tb_assessment)

            # 步骤2: 将异质图转换为同构图（兼容现有GNN架构
            homo_data = predictor._convert_hetero_to_homo(hetero_data)

            # 步骤3: 找出目标接触者在同构图中的节点索引
            target_idx = predictor._find_target_node_idx(
                contact_data,
                homo_data,
                contact_type,
                tb_assessment
            )

            if target_idx is None:
                LOGGER.error("未找到目标接触者节点")
                return None

            # 步骤4: 执行GNN前向传播（支持edge_attr）
            predictor.gnn_model.eval()
            device = next(predictor.gnn_model.parameters()).device

            with torch.no_grad():
                homo_data = homo_data.to(device)
                if hasattr(homo_data, 'edge_attr') and homo_data.edge_attr is not None:
                    risk_scores, _ = predictor.gnn_model(
                        homo_data.x,
                        homo_data.edge_index,
                        edge_attr=homo_data.edge_attr
                    )
                else:
                    risk_scores, _ = predictor.gnn_model(homo_data.x, homo_data.edge_index)

            # 步骤5: 提取目标节点的风险概率
            gnn_risk = float(risk_scores[target_idx, 0].cpu().numpy()) * 100.0
            gnn_risk = max(0.0, min(100.0, gnn_risk))

            # 步骤6: 构建返回结果
            risk_class = 1 if gnn_risk > 50 else 0

            return {
                'risk_probability': float(gnn_risk),
                'risk_class': risk_class,
                'model_name': 'GNN图神经网络',
                'model_name_en': 'GNN Graph Neural Network',
                'network_aware': True,
                'node_idx': target_idx,
                'gnn_used': True
            }

        except Exception as e:
            LOGGER.warning("GNN预测失败: %s", e, exc_info=True)
            return None


def predict_time_aware_gnn_risk(predictor, tb_assessment, contact_data, contact_type='family'):
        """
        使用时间感知GNN预测接触者风险

        特别功能：
        - 支持治疗时间衰减
        - 考虑时间序列特征
        - 基于SEIR动力学的时间注意力

        返回：
            dict: 包含风险预测和风险曲线
        """
        if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
            return None

        if not hasattr(predictor, 'time_aware_gnn_model') or not hasattr(predictor, 'stgnn_is_trained') or not predictor.stgnn_is_trained:
            return None

        try:
            import torch

            hetero_data = predictor.gnn_network_builder.build_from_assessment(tb_assessment)
            homo_data = predictor._convert_hetero_to_homo(hetero_data)

            target_idx = predictor._find_target_node_idx(
                contact_data, homo_data, contact_type, tb_assessment
            )

            if target_idx is None:
                LOGGER.error("未找到目标接触者节点")
                return None

            predictor.time_aware_gnn_model.eval()
            device = next(predictor.time_aware_gnn_model.parameters()).device

            # 计算治疗时间（基于患者信息）
            treatment_week = torch.zeros(homo_data.x.shape[0], 1, device=device)
            if hasattr(tb_assessment, 'patient_info') and tb_assessment.patient_info:
                ftd = tb_assessment.patient_info.get('basic_info', {}).get('ftd', 0)
                treatment_week[:, 0] = max(0.0, float(ftd))

            with torch.no_grad():
                homo_data = homo_data.to(device)
                treatment_week = treatment_week.to(device)

                if hasattr(homo_data, 'edge_attr') and homo_data.edge_attr is not None:
                    risk_scores, _ = predictor.time_aware_gnn_model(
                        homo_data.x, homo_data.edge_index,
                        treatment_week=treatment_week,
                        edge_attr=homo_data.edge_attr
                    )
                else:
                    risk_scores, _ = predictor.time_aware_gnn_model(
                        homo_data.x, homo_data.edge_index,
                        treatment_week=treatment_week
                    )

            gnn_risk = float(risk_scores[target_idx, 0].cpu().numpy()) * 100.0
            gnn_risk = max(0.0, min(100.0, gnn_risk))

            risk_class = 1 if gnn_risk > 50 else 0

            return {
                'risk_probability': float(gnn_risk),
                'risk_class': risk_class,
                'model_name': '时空图神经网络(ST-GNN)',
                'model_name_en': 'Spatio-Temporal Graph Neural Network',
                'time_aware': True,
                'seir_informed': True,
                'node_idx': target_idx
            }

        except Exception as e:
            LOGGER.warning("时间感知GNN预测失败: %s", e, exc_info=True)
            return None


def predict_gnn_increment(predictor, tb_assessment, contact_data,
                          contact_type='family', p_base=None):
    """第 2 层：GNN 残差增量推理（网络增强层）。

    残差语义：``网络增量 = 网络感知风险 − 个体基线概率 P_base``。
    网络只负责解释"接触结构带来的额外风险"，与个体层不再重复计分。
    无网络结构（network_aware=False）时增量自然趋于 0。

    参数：
        predictor: MLRiskPredictor（提供 predict_gnn_risk / predict_risk）
        tb_assessment: 风险评估实例
        contact_data (dict): 目标接触者数据
        contact_type (str): 'family' / 'social'
        p_base (float|None): 0-100 个体基线概率；None 时用 ML 第 1 层计算

    返回：
        dict:
            - 'baseline_probability': 0-100 个体基线
            - 'network_risk': 0-100 网络感知风险
            - 'network_increment': 网络增量（残差）
            - 'network_aware': 是否使用了网络结构
            - 'backend': 'pytorch' / 'numpy' / 'fallback'
    """
    # 第 1 层：个体基线概率（未传入时自行计算）
    if p_base is None:
        base = 30.0
        try:
            pred = predictor.predict_risk(contact_data, contact_type)
            if pred and 'ensemble' in pred:
                base = float(pred['ensemble'].get('risk_probability', 30.0))
        except Exception as e:
            LOGGER.debug("第 2 层 P_base 计算失败，使用默认 30.0: %s", e)
        p_base = base
    p_base = float(max(0.0, min(100.0, p_base)))

    # 第 2 层：网络感知风险（PyG 完整路径或 NumPy 回退）
    gnn_fn = getattr(predictor, 'predict_gnn_risk', None)
    network_risk = None
    backend = 'fallback'
    network_aware = False
    if gnn_fn is not None:
        try:
            result = gnn_fn(tb_assessment, contact_data, contact_type)
            if result is not None and result.get('risk_probability') is not None:
                network_risk = float(result['risk_probability'])
                backend = result.get('backend', 'pytorch')
                network_aware = bool(result.get('network_aware', False))
        except Exception as e:
            LOGGER.debug("GNN 残差推理异常，回退: %s", e)

    if network_risk is None:
        # 纯 NumPy 兜底（确定性）
        try:
            from ....ml.gnn import NumPyGraphRiskPredictor
            family_entries = getattr(tb_assessment, 'family_entries', None) or []
            social_entries = getattr(tb_assessment, 'social_entries', None) or []
            patient_info = getattr(tb_assessment, 'patient_info', None)
            np_predictor = NumPyGraphRiskPredictor()
            if family_entries or social_entries:
                result = np_predictor.predict_network_risk(
                    contact_data, contact_type, patient_info,
                    family_entries, social_entries, p_base=p_base)
            else:
                result = np_predictor.predict_contact_risk(
                    contact_data, contact_type, p_base=p_base)
            if result is not None:
                network_risk = float(result['risk_probability'])
                backend = 'numpy'
                network_aware = bool(result.get('network_aware', False))
        except Exception as e:
            LOGGER.debug("NumPy 残差兜底失败: %s", e)

    if network_risk is None:
        network_risk = p_base
        backend = 'fallback'

    return {
        'baseline_probability': p_base,
        'network_risk': float(max(0.0, min(100.0, network_risk))),
        'network_increment': float(max(0.0, min(100.0, network_risk))) - p_base,
        'network_aware': network_aware,
        'backend': backend,
        'residual_learning': True,
    }
