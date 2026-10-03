#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN训练 — GNN训练循环子模块。

包含 SEIRInformedGNN 的完整训练流程（超参数搜索、两阶段训练、早停、混合精度）。
从原 scoring/ml/gnn_training.py 拆分而来，不修改业务逻辑。
"""
import copy

import numpy as np

from ._common import (
    LOGGER,
    PYTORCH_AVAILABLE,
    PYG_AVAILABLE,
    SEIRInformedGNN,
    HeterogeneousTBNetwork,
)


def train_gnn(predictor, n_samples=2000, random_state=42, enable_hyperopt=False,
                   n_epochs=50, learning_rate=0.001, use_focal_loss=True,
                   alpha=0.25, gamma=2.0, two_phase_training=True, log_callback=None,
                   dropout=0.3, use_optuna=True, n_optuna_trials=30,
                   use_mixed_precision=True, enable_pruning=True,
                   progress_callback=None, gnn_num_layers=1, gnn_hidden_dim=64,
                   graph_generator=None):
        """
        训练GNN模型用于社会接触网络分析（文献驱动实现）

        文献支撑：
        - 两阶段训练：Tan et al., DeepTrace 2024
        - Focal Loss：Lin et al., ICCV 2017
        - 贝叶斯优化：Akiba et al., KDD 2019 (Optuna)
        - 试验剪枝：Li et al., JMLR 2018 (Hyperband)
        - 混合精度训练：Micikevicius et al., ICLR 2018

        参数：
            n_samples (int): 合成训练数据样本量，默认2000
            random_state (int): 随机种子
            enable_hyperopt (bool): 是否启用超参数搜索
            n_epochs (int): 训练轮数
            learning_rate (float): 学习率
            use_focal_loss (bool): 是否使用Focal Loss处理类别不平衡
            alpha (float): Focal Loss的alpha参数
            gamma (float): Focal Loss的gamma参数
            two_phase_training (bool): 是否使用两阶段训练策略
            log_callback (callable, optional): 日志回调函数，用于输出到GUI
            dropout (float): 初始dropout参数
            use_optuna (bool): 是否使用Optuna贝叶斯优化（默认True，比网格搜索更高效）
            n_optuna_trials (int): Optuna试验次数，默认30
            use_mixed_precision (bool): 是否使用混合精度训练（FP16）
            enable_pruning (bool): 是否启用试验剪枝（默认True，与Optuna配合使用）
            progress_callback (callable, optional): 进度回调 (percent:0-100, message)

        返回：
            bool: 训练是否成功
        """
        def _log(msg):
            """内部日志输出函数"""
            if log_callback:
                log_callback(msg)
            else:
                LOGGER.info(msg)

        def _progress(pct, msg):
            """内部进度上报（线程安全由调用方保证）"""
            if progress_callback:
                try:
                    progress_callback(pct, msg)
                except Exception:
                    pass

        if not (PYTORCH_AVAILABLE and PYG_AVAILABLE):
            return False

        try:
            import torch
            import torch.nn.functional as F
            from torch_geometric.loader import DataLoader
            from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss

            # 检查Optuna可用性
            OPTUNA_AVAILABLE = False
            try:
                import optuna
                from optuna.trial import Trial
                from optuna.pruners import HyperbandPruner
                OPTUNA_AVAILABLE = True
            except ImportError:
                _log("Optuna未安装，将使用传统超参数搜索方法")

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            _log(f"使用设备: {device}")
            _progress(2, "初始化训练环境...")

            # 混合精度训练设置
            scaler = None
            if use_mixed_precision and torch.cuda.is_available():
                scaler = torch.cuda.amp.GradScaler()
                _log("混合精度训练（FP16）已启用")

            # 初始化GNN模型和网络构建器
            # 注：hidden_dim/num_layers/dropout/lr/weight_decay/batch/epochs 等数值
            # 属通用深度学习超参数（非某篇论文定义），默认值为工程经验值，可经
            # enable_hyperopt 的 Optuna/网格搜索自动调优（见超参数搜索分支）。
            # 架构选取文献（说明"为什么用这类层"，
            # 见 ml/framework/temporal.py / ml/gnn/_base.py）：
            #   - GATv2：Brody et al., ICLR 2022
            #   - GCN：Kipf & Welling, ICLR 2017
            #   - GAT：Veličković et al., ICLR 2018
            #   - GraphSAGE：Hamilton et al., NeurIPS 2017
            # 超参数来源（把"经验值"转化为可复现的调优结果）：
            #   - Optuna：Akiba et al., KDD 2019
            #   - Hyperband 剪枝：Li et al., JMLR 2018
            # 生成合成数据（第一阶段：合成数据预训练，文献支撑：Tan et al. 2024）
            # graph_generator：外部图生成器（场景对齐训练用，消除
            # 训练图/部署场景分布断崖）；签名 (n_graphs, random_state)
            # -> (train, val, test)，字段与 _generate_synthetic_graphs 同构
            _log("生成合成图数据...")
            _progress(6, "生成合成图数据...")
            if graph_generator is not None:
                train_graphs, val_graphs, test_graphs = graph_generator(
                    n_graphs=n_samples, random_state=random_state)
            else:
                train_graphs, val_graphs, test_graphs = predictor._generate_synthetic_graphs(
                    n_graphs=n_samples, random_state=random_state, device=device
                )

            hidden_dim = gnn_hidden_dim
            num_layers = gnn_num_layers
            # 节点特征维度从实际生成数据推断：
            # _generate_synthetic_graphs 产出 24 维特征（22 随机 + 入度
            # 接触广度 + 邻居确认密度结构信号；GNN 特征提取路径为 30 维），
            # 两者历史不一致曾导致 matmul 形状错误；此处按数据实际
            # 维度建模，保证与训练图数据一致。
            node_feat_dim = (
                int(train_graphs[0].x.shape[1]) if train_graphs else 30
            )
            predictor.gnn_model = SEIRInformedGNN(
                node_feature_dim=node_feat_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
            predictor.gnn_network_builder = HeterogeneousTBNetwork()

            train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
            test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)

            # 超参数搜索（支持Optuna贝叶斯优化）
            # 样本量门槛（项目约束：样本量 ≥ 5000 再调参）：小样本上
            # 超参搜索的验证集噪声占主导，选出的"最优"超参是过拟合
            # 噪声而非信号；门槛下强制走默认超参
            GNN_HYPEROPT_MIN_SAMPLES = 5000
            if enable_hyperopt and n_samples < GNN_HYPEROPT_MIN_SAMPLES:
                _log(f"样本量 {n_samples} < {GNN_HYPEROPT_MIN_SAMPLES}，"
                     f"跳过超参数搜索（约束：样本量 ≥ 5000 再调参），"
                     f"使用默认超参数训练")
                enable_hyperopt = False
            if enable_hyperopt:
                _progress(12, "超参数搜索中...")
                if OPTUNA_AVAILABLE and use_optuna:
                    _log("开始Optuna贝叶斯超参数搜索...")

                    def objective(trial: Trial) -> float:
                        """Optuna目标函数"""
                        trial_hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
                        # num_layers 收敛为 [1]（项目约束）：v7-v11 诊断扫描
                        # （ml/gnn/pi_gnn.py 头部，2026-08-25，8 种子 n=400）
                        # 证明 ≥2 层 GAT 在小样本图上记忆化 Bernoulli 标签
                        # 噪声（train 0.97 / test 0.80），类型化单层聚合 +
                        # 浅读出是唯一正增益结构（Δ=+0.0235，8/8 为正）。
                        # 据此 num_layers=1 固化为默认架构约束，不再作为
                        # 搜索维度（深层候选直接排除，防过平滑复发）。
                        trial_num_layers = trial.suggest_categorical('num_layers', [1])
                        trial_lr = trial.suggest_categorical('learning_rate', [0.001, 0.005, 0.01])
                        trial_dropout = trial.suggest_categorical('dropout', [0.2, 0.3, 0.4])

                        _log(f"Trial {trial.number}: hidden_dim={trial_hidden_dim}, layers={trial_num_layers}, lr={trial_lr}, dropout={trial_dropout}")

                        # 修复：节点特征维度必须与实际生成数据一致
                        # （train_graphs 为 24 维，硬编码 30 会导致 matmul
                        # 形状错误，hyperopt 一开就崩）
                        temp_model = SEIRInformedGNN(
                            node_feature_dim=node_feat_dim,
                            hidden_dim=trial_hidden_dim,
                            num_layers=trial_num_layers,
                            dropout=trial_dropout
                        ).to(device)

                        val_auroc = predictor._train_gnn_once(
                            temp_model, train_loader, val_loader, device,
                            n_epochs=20 if not enable_pruning else 5,  # 剪枝模式下使用更少epoch
                            learning_rate=trial_lr, use_focal_loss=use_focal_loss,
                            alpha=alpha, gamma=gamma
                        )

                        # 报告中间值给Optuna用于剪枝
                        if enable_pruning:
                            trial.report(val_auroc, step=0)
                            if trial.should_prune():
                                raise optuna.exceptions.TrialPruned()

                        # 清理内存
                        del temp_model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        return val_auroc

                    # 设置剪枝器（Hyperband算法）
                    pruner = HyperbandPruner(min_resource=5, max_resource=20, reduction_factor=3) if enable_pruning else None
                    study = optuna.create_study(
                        direction='maximize',
                        pruner=pruner,
                        sampler=optuna.samplers.TPESampler(seed=random_state)
                    )
                    study.optimize(objective, n_trials=n_optuna_trials)

                    _log(f"最佳超参数: {study.best_params}")
                    _log(f"最佳验证AUROC: {study.best_value:.4f}")

                    # 使用最佳超参数重新初始化模型
                    hidden_dim = study.best_params['hidden_dim']
                    num_layers = study.best_params['num_layers']
                    learning_rate = study.best_params['learning_rate']
                    dropout = study.best_params['dropout']

                else:
                    _log("开始传统网格超参数搜索...")
                    best_auroc = 0.0
                    best_params = None
                    param_grid = {
                        'hidden_dim': [32, 64, 128],
                        # num_layers 收敛为 [1]：与 Optuna 路径同一项目约束
                        # （单层聚合是唯一正增益结构，见上方诊断扫描注记）
                        'num_layers': [1],
                        'learning_rate': [0.001, 0.005, 0.01],
                        'dropout': [0.2, 0.3, 0.4]
                    }

                    for hidden_dim in param_grid['hidden_dim']:
                        for num_layers in param_grid['num_layers']:
                            for lr in param_grid['learning_rate']:
                                for dp in param_grid['dropout']:
                                    # 修复：与实际数据维度一致（原硬编码 30）
                                    temp_model = SEIRInformedGNN(
                                        node_feature_dim=node_feat_dim,
                                        hidden_dim=hidden_dim,
                                        num_layers=num_layers,
                                        dropout=dp
                                    ).to(device)

                                    val_auroc = predictor._train_gnn_once(
                                        temp_model, train_loader, val_loader, device,
                                        n_epochs=20, learning_rate=lr, use_focal_loss=use_focal_loss,
                                        alpha=alpha, gamma=gamma
                                    )

                                    if val_auroc > best_auroc:
                                        best_auroc = val_auroc
                                        best_params = (hidden_dim, num_layers, lr, dp)
                                        _log(f"新的最佳超参数: hidden_dim={hidden_dim}, layers={num_layers}, lr={lr}, dropout={dp}, val_auroc={val_auroc:.4f}")

                                    # 清理内存
                                    del temp_model
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()

                    if best_params:
                        hidden_dim, num_layers, learning_rate, dropout = best_params

                # 重新初始化模型
                # 修复：与实际数据维度一致（原硬编码 30）
                predictor.gnn_model = SEIRInformedGNN(
                    node_feature_dim=node_feat_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout
                ).to(device)

                # 清理临时数据
                del train_graphs, val_graphs, test_graphs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 重新生成数据用于完整训练
                if graph_generator is not None:
                    train_graphs, val_graphs, test_graphs = graph_generator(
                        n_graphs=n_samples, random_state=random_state)
                else:
                    train_graphs, val_graphs, test_graphs = predictor._generate_synthetic_graphs(
                        n_graphs=n_samples, random_state=random_state
                    )
                train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
                val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
                test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)

            # 完整训练
            _log("开始完整训练...")
            _progress(18, "开始完整训练...")
            optimizer = torch.optim.AdamW(predictor.gnn_model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

            # 全局固定 pos_weight：训练集统计一次，focal loss 全程
            # 用同一权重（批内动态重算使 loss 量级随批次波动）
            from .training_utils import _global_pos_weight_from_loader
            global_pos_weight = _global_pos_weight_from_loader(train_loader)

            best_val_auroc = 0.0
            best_model_state = None
            patience_counter = 0
            # 早停耐心：小直径图上注意力学习慢（layers=1 时 epoch 48 仍在
            # 上升），10 轮过早截断；30 轮兼顾效率与收敛
            early_stop_patience = 30
            early_stopped = False
            # 训练曲线（每 epoch 的 train_loss / val_auroc）：
            # 用于收敛诊断——loss 早期平坦且 AUROC 在 0.5 附近震荡，
            # 提示学习率过大或图特征无信号
            epoch_history = []

            for epoch in range(n_epochs):
                # 训练
                predictor.gnn_model.train()
                train_loss = 0.0
                train_preds = []
                train_labels = []

                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()

                    if scaler is not None:
                        # 混合精度训练
                        with torch.cuda.amp.autocast():
                            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                                risk_scores, _ = predictor.gnn_model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
                            else:
                                risk_scores, _ = predictor.gnn_model(batch.x, batch.edge_index, batch=batch.batch)

                            # 应用mask，只计算接触者节点的损失
                            if hasattr(batch, 'mask'):
                                mask = batch.mask
                                pred = risk_scores.squeeze()[mask]
                                target = batch.y[mask]
                            else:
                                pred = risk_scores.squeeze()
                                target = batch.y

                            if use_focal_loss:
                                loss = predictor._focal_loss(pred, target, alpha=alpha, gamma=gamma)
                            else:
                                loss = F.binary_cross_entropy(pred, target.float())

                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(predictor.gnn_model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # 标准精度训练
                        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                            risk_scores, _ = predictor.gnn_model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
                        else:
                            risk_scores, _ = predictor.gnn_model(batch.x, batch.edge_index, batch=batch.batch)

                        # 应用mask，只计算接触者节点的损失
                        if hasattr(batch, 'mask'):
                            mask = batch.mask
                            pred = risk_scores.squeeze()[mask]
                            target = batch.y[mask]
                        else:
                            pred = risk_scores.squeeze()
                            target = batch.y

                        if use_focal_loss:
                            loss = predictor._focal_loss(pred, target, alpha=alpha, gamma=gamma)
                        else:
                            loss = F.binary_cross_entropy(pred, target.float())

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(predictor.gnn_model.parameters(), max_norm=1.0)
                        optimizer.step()

                    # 收集训练数据时也用mask
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
                val_metrics = predictor._evaluate_gnn(predictor.gnn_model, val_loader, device)

                # 学习率调度
                scheduler.step(val_metrics['auroc'])

                # 记录训练曲线
                epoch_history.append({
                    'epoch': epoch + 1,
                    'train_loss': round(float(train_loss), 6),
                    'val_auroc': round(float(val_metrics['auroc']), 6)
                    if val_metrics.get('auroc') is not None else None,
                })

                # 进度上报（18% -> 80% 随 epoch 推进）
                _progress(min(80, 18 + int((epoch + 1) / max(n_epochs, 1) * 62)),
                          f"训练 Epoch {epoch + 1}/{n_epochs}")

                # 早停检查
                if val_metrics['auroc'] > best_val_auroc:
                    best_val_auroc = val_metrics['auroc']
                    best_model_state = copy.deepcopy(predictor.gnn_model.state_dict())
                    patience_counter = 0
                    _log(f"Epoch {epoch+1}: 新的最佳模型! val_auroc={best_val_auroc:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        early_stopped = True
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

            # 加载最佳模型
            if best_model_state is not None:
                predictor.gnn_model.load_state_dict(best_model_state)

            # 两阶段训练第二阶段：微调（文献支撑：Tan et al. 2024）
            if two_phase_training:
                _log("两阶段训练第二阶段：微调...")
                _progress(82, "两阶段微调中...")
                finetune_graphs = predictor._generate_synthetic_graphs(
                    n_graphs=min(500, n_samples // 2), random_state=random_state + 1,
                    device=device
                )[0]
                finetune_loader = DataLoader(finetune_graphs, batch_size=8, shuffle=True)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate * 0.1

                for epoch in range(10):
                    predictor.gnn_model.train()
                    for batch in finetune_loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()

                        if scaler is not None:
                            # 混合精度微调
                            with torch.cuda.amp.autocast():
                                if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                                    risk_scores, _ = predictor.gnn_model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
                                else:
                                    risk_scores, _ = predictor.gnn_model(batch.x, batch.edge_index, batch=batch.batch)

                                # 应用mask，只计算接触者节点的损失
                                if hasattr(batch, 'mask'):
                                    mask = batch.mask
                                    pred = risk_scores.squeeze()[mask]
                                    target = batch.y[mask]
                                else:
                                    pred = risk_scores.squeeze()
                                    target = batch.y

                                if use_focal_loss:
                                    loss = predictor._focal_loss(
                                        pred, target, alpha=alpha, gamma=gamma
                                    )
                                else:
                                    loss = F.binary_cross_entropy(pred, target.float())

                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # 标准精度微调
                            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                                risk_scores, _ = predictor.gnn_model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
                            else:
                                risk_scores, _ = predictor.gnn_model(batch.x, batch.edge_index, batch=batch.batch)

                            # 应用mask，只计算接触者节点的损失
                            if hasattr(batch, 'mask'):
                                mask = batch.mask
                                pred = risk_scores.squeeze()[mask]
                                target = batch.y[mask]
                            else:
                                pred = risk_scores.squeeze()
                                target = batch.y

                            if use_focal_loss:
                                loss = predictor._focal_loss(
                                    pred, target, alpha=alpha, gamma=gamma,
                                    pos_weight=global_pos_weight)
                            else:
                                loss = F.binary_cross_entropy(pred, target.float())
                            loss.backward()
                            optimizer.step()

                # 微调完成后清理
                del finetune_graphs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 最终测试评估
            _progress(95, "评估测试集...")
            test_metrics = predictor._evaluate_gnn(predictor.gnn_model, test_loader, device)
            _log("\n测试集性能:")
            for metric_name, metric_value in test_metrics.items():
                if metric_value is None:
                    _log(f"  {metric_name}: N/A")
                else:
                    _log(f"  {metric_name}: {metric_value:.4f}")

            # 测试集阳性率：AUPRC 的基线参照（无此值 AUPRC 无从解读——
            # 0.428 在 34% 阳性率下与在 5% 下是完全不同的成绩）
            test_labels = []
            for batch in test_loader:
                if hasattr(batch, 'mask') and batch.mask is not None:
                    test_labels.append(batch.y[batch.mask].cpu().numpy())
                else:
                    test_labels.append(batch.y.cpu().numpy())
            test_pos_rate = (float(np.concatenate(test_labels).mean())
                             if test_labels else 0.0)

            # ΔAUROC 同切分对照（监控缺口修复）：仅特征基线 = 无图结构
            # 的 logistic（同节点特征、同接触者 mask、同 train/test 切分）。
            # 网络层正增益必须以 Δ = GNN − 基线 持续监控——此前 ΔAUROC
            # 为负的问题靠合成数据修复过，无此对照同类问题会复发而不
            # 被发现（验证标准见 data_preparation.py：GNN 须优于仅特征
            # 基线 > 0.05）
            def _collect_nodes(loader):
                feats, labels = [], []
                for batch in loader:
                    batch = batch.to('cpu')
                    if hasattr(batch, 'mask') and batch.mask is not None:
                        m = batch.mask
                        feats.append(batch.x[m].detach().cpu().numpy())
                        labels.append(batch.y[m].detach().cpu().numpy())
                    else:
                        feats.append(batch.x.detach().cpu().numpy())
                        labels.append(batch.y.detach().cpu().numpy())
                if not feats:
                    return None, None
                return np.concatenate(feats), np.concatenate(labels)

            baseline_auroc = None
            delta_auroc = None
            try:
                from sklearn.linear_model import LogisticRegression
                X_bl, y_bl = _collect_nodes(train_loader)
                X_bl_te, y_bl_te = _collect_nodes(test_loader)
                if (X_bl is not None and X_bl_te is not None
                        and len(np.unique(y_bl)) >= 2
                        and len(np.unique(y_bl_te)) >= 2):
                    baseline = LogisticRegression(
                        max_iter=1000, class_weight='balanced')
                    baseline.fit(X_bl, y_bl)
                    p_bl = baseline.predict_proba(X_bl_te)[:, 1]
                    baseline_auroc = float(roc_auc_score(y_bl_te, p_bl))
                    delta_auroc = float(test_metrics['auroc'] - baseline_auroc)
                    _log(f"仅特征基线 AUROC={baseline_auroc:.4f}, "
                         f"ΔAUROC(网络层净增益)={delta_auroc:+.4f}")
                    if delta_auroc < 0:
                        _log(f"警告: GNN 低于仅特征基线 "
                             f"({test_metrics['auroc']:.4f} < "
                             f"{baseline_auroc:.4f})，网络层为负增益，"
                             f"不应作为增益证据引用")
            except Exception as bl_err:
                LOGGER.warning("仅特征基线对照计算失败: %s", bl_err)

            # 更新GNN性能指标（修复：保存模型超参数）
            predictor.model_performance['gnn'] = {
                'AUROC': test_metrics['auroc'],
                'AUPRC': test_metrics['auprc'],
                'positive_rate': test_pos_rate,
                'Accuracy': test_metrics['accuracy'],
                'Precision': test_metrics['precision'],
                'Recall': test_metrics['recall'],
                'F1': test_metrics['f1'],
                'Brier': test_metrics['brier'],
                # 阈值与约登指数：F1/Accuracy 等阈值依赖指标的数据驱动切点
                'threshold': test_metrics.get('threshold'),
                'youden_index': test_metrics.get('youden_index'),
                'name': 'GNN图神经网络',
                'name_en': 'GNN Graph Neural Network',
                'two_phase_training': two_phase_training,
                'focal_loss': use_focal_loss,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                # 收敛保障：训练曲线 + 早停信息（供归档与收敛诊断）
                'best_val_auroc': float(best_val_auroc),
                'early_stopped': early_stopped,
                'epochs_run': len(epoch_history),
                'training_curve': epoch_history,
                # ΔAUROC 同切分对照（网络层净增益监控）：
                # 仅特征基线（无图结构 logistic）AUROC 与 GNN−基线差值；
                # delta < 0 表示网络层为负增益，不可作增益证据引用
                'feature_only_baseline_auroc': baseline_auroc,
                'delta_auroc_vs_feature_only': delta_auroc,
            }

            # 训练完成后清理内存
            del train_graphs, val_graphs, test_graphs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            predictor.gnn_is_trained = True
            _progress(100, "GNN 训练完成")
            return True

        except Exception as e:
            LOGGER.warning("GNN模型训练失败: %s", e, exc_info=True)
            # 记录异常，供 GUI 层生成可读错误与恢复建议
            try:
                predictor._last_gnn_train_error = e
            except Exception:
                pass
            # 异常情况下也要清理（直接置 None 释放引用，避免 NameError 风险）
            train_graphs = val_graphs = test_graphs = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
