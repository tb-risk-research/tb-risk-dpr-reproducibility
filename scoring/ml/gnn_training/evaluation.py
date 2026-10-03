#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN训练 — 评估子模块。

包含GNN模型评估函数（验证集评估、指标计算）。
从原 scoring/ml/gnn_training.py 拆分而来，不修改业务逻辑。
"""
import numpy as np

try:
    import torch
except ImportError:
    pass


def _evaluate_gnn(predictor, model, data_loader, device):
        """
        评估GNN模型（修复：使用mask忽略患者节点；支持edge_attr；传递batch索引）

        文献支撑：Yang et al. 2021; Catarcione Pinto et al. 2025
        """
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)

                # 调用forward，根据是否有edge_attr选择方式，并传递batch索引
                if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                    risk_scores, _ = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
                else:
                    risk_scores, _ = model(batch.x, batch.edge_index, batch=batch.batch)

                # 应用mask，只收集接触者节点的预测和标签
                if hasattr(batch, 'mask'):
                    mask = batch.mask
                    pred = risk_scores[mask].cpu().numpy()
                    label = batch.y[mask].cpu().numpy()
                else:
                    pred = risk_scores.cpu().numpy()
                    label = batch.y.cpu().numpy()

                all_preds.extend(pred)
                all_labels.extend(label)

        return predictor._compute_gnn_metrics(np.array(all_preds), np.array(all_labels))
