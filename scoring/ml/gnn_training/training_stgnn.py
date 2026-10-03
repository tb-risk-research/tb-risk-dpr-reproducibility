#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN训练 — ST-GNN训练循环子模块。

包含时空图神经网络（SEIRTimeAwareGNN）的训练流程。
从原 scoring/ml/gnn_training.py 拆分而来，不修改业务逻辑。
"""
import copy

import numpy as np

from ._common import (
    LOGGER,
    PYTORCH_AVAILABLE,
    PYG_AVAILABLE,
    SEIRTimeAwareGNN,
    HeterogeneousTBNetwork,
)


def train_stgnn(predictor, n_samples=2000, random_state=42, n_epochs=50,
                    learning_rate=0.001, time_steps=52, dropout=0.3, log_callback=None):
        """
        训练时空图神经网络（ST-GNN）

        核心创新：
        - 方案一：时序编码 + 时间感知GNN
        - 方案二：时间卷积 + 时空GNN
        - 方案三：SEIR动力学驱动的时间注意力

        文献支撑：
        - ST-GCN: Yan et al., 2018
        - ASTGNN: Guo et al., 2021
        - DCRNN: Li et al., 2018
        - SEIR-informed GNN: Zhang et al., 2025

        参数：
            n_samples (int): 合成训练样本数
            random_state (int): 随机种子
            n_epochs (int): 训练轮数
            learning_rate (float): 学习率
            time_steps (int): 时间步数（默认52周）
            dropout (float): Dropout比例
            log_callback (callable): 日志回调函数

        返回：
            bool: 训练是否成功
        """
        def _log(msg):
            if log_callback:
                log_callback(msg)
            else:
                LOGGER.info(msg)

        if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
            return False

        try:
            import torch
            import torch.nn.functional as F
            from torch_geometric.loader import DataLoader

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            _log(f"使用设备: {device}")

            # 初始化时间感知GNN模型和网络构建器
            # num_layers=1：与项目架构约束一致（单层聚合防过平滑，
            # 见 training_gnn.py 约束注记）
            hidden_dim = 64
            num_layers = 1
            predictor.time_aware_gnn_model = SEIRTimeAwareGNN(
                node_feature_dim=30,
                edge_feature_dim=6,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
            predictor.gnn_network_builder = HeterogeneousTBNetwork()
            predictor.time_steps = time_steps

            # 生成合成图数据（用于训练）
            _log("生成合成图数据...")
            train_graphs, val_graphs, test_graphs = predictor._generate_synthetic_graphs(
                n_graphs=n_samples, random_state=random_state, device=device
            )

            train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
            test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)

            # 开始训练
            _log("开始训练时间感知GNN...")
            optimizer = torch.optim.AdamW(predictor.time_aware_gnn_model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

            best_val_auroc = 0.0
            best_model_state = None
            patience_counter = 0
            early_stop_patience = 10

            for epoch in range(n_epochs):
                predictor.time_aware_gnn_model.train()
                train_loss = 0.0
                train_preds = []
                train_labels = []

                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()

                    # 前向传播（使用时间感知GNN）
                    # 为每个图生成合成的治疗时间
                    treatment_week = torch.rand(batch.x.shape[0], 1, device=device) * predictor.time_steps

                    if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                        risk_scores, _ = predictor.time_aware_gnn_model(
                            batch.x, batch.edge_index,
                            treatment_week=treatment_week, edge_attr=batch.edge_attr
                        )
                    else:
                        risk_scores, _ = predictor.time_aware_gnn_model(
                            batch.x, batch.edge_index,
                            treatment_week=treatment_week
                        )

                    # 计算损失
                    if hasattr(batch, 'mask'):
                        mask = batch.mask
                        pred = risk_scores.squeeze()[mask]
                        target = batch.y[mask]
                    else:
                        pred = risk_scores.squeeze()
                        target = batch.y

                    loss = F.binary_cross_entropy(pred, target.float())

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(predictor.time_aware_gnn_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    if hasattr(batch, 'mask'):
                        pred_np = risk_scores[mask].detach().cpu().numpy()
                        label_np = batch.y[mask].detach().cpu().numpy()
                    else:
                        pred_np = risk_scores.detach().cpu().numpy()
                        label_np = batch.y.detach().cpu().numpy()

                    train_loss += loss.item() * batch.num_graphs
                    train_preds.extend(pred_np)
                    train_labels.extend(label_np)

                train_loss /= len(train_loader.dataset)

                # 验证
                val_metrics = predictor._evaluate_gnn(predictor.time_aware_gnn_model, val_loader, device)

                scheduler.step(val_metrics['auroc'])

                if val_metrics['auroc'] > best_val_auroc:
                    best_val_auroc = val_metrics['auroc']
                    best_model_state = copy.deepcopy(predictor.time_aware_gnn_model.state_dict())
                    patience_counter = 0
                    _log(f"Epoch {epoch+1}: 新的最佳模型! val_auroc={best_val_auroc:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        _log(f"早停触发，{early_stop_patience}轮无改进")
                        break

                if (epoch + 1) % 10 == 0:
                    train_metrics = predictor._compute_gnn_metrics(
                        np.array(train_preds), np.array(train_labels)
                    )
                    _log(f"Epoch {epoch+1}/{n_epochs}: "
                          f"train_loss={train_loss:.4f}, "
                          f"train_auroc={train_metrics['auroc']:.4f}, "
                          f"val_auroc={val_metrics['auroc']:.4f}, "
                          f"val_auprc={val_metrics['auprc']:.4f}")

            if best_model_state is not None:
                predictor.time_aware_gnn_model.load_state_dict(best_model_state)

            # 最终测试评估
            test_metrics = predictor._evaluate_gnn(predictor.time_aware_gnn_model, test_loader, device)
            _log("\n测试集性能:")
            for metric_name, metric_value in test_metrics.items():
                _log(f"  {metric_name}: {metric_value:.4f}")

            predictor.model_performance['stgnn'] = {
                'AUROC': test_metrics['auroc'],
                'AUPRC': test_metrics['auprc'],
                'Accuracy': test_metrics['accuracy'],
                'Precision': test_metrics['precision'],
                'Recall': test_metrics['recall'],
                'F1': test_metrics['f1'],
                'Brier': test_metrics['brier'],
                'name': '时空图神经网络(ST-GNN)',
                'name_en': 'Spatio-Temporal Graph Neural Network',
                'time_steps': time_steps,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': learning_rate
            }

            predictor.stgnn_is_trained = True
            return True

        except Exception as e:
            LOGGER.warning("ST-GNN训练失败: %s", e, exc_info=True)
            return False
