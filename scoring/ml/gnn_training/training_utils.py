#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN训练 — 训练辅助子模块。

包含损失函数（Focal Loss）和单次训练循环辅助函数。
从原 scoring/ml/gnn_training.py 拆分而来，不修改业务逻辑。
"""


def _global_pos_weight_from_loader(train_loader):
        """从训练集统计全局 pos_weight = n_neg/n_pos（一次计算，批间固定）

        批内动态 pos_weight 的问题：小 batch 的正样本数波动大（0-3 个），
        n_neg/n_pos 逐批跳变，loss 量级随批次波动且与全局不平衡度偏离。
        训练开始统计一次全局值，focal loss 全程用同一权重。
        """
        n_pos = 0.0
        n_neg = 0.0
        for batch in train_loader:
            if hasattr(batch, 'mask') and batch.mask is not None:
                t = batch.y[batch.mask]
            else:
                t = batch.y
            n_pos += float(t.sum())
            n_neg += float((1.0 - t).sum())
        return max(n_neg / max(n_pos, 1.0), 1.0)


def _focal_loss(predictor, preds, targets, alpha=0.25, gamma=2.0,
                adaptive_pos_weight=True, pos_weight=None):
        """
        Focal Loss用于处理类别不平衡（文献支撑：Lin et al., ICCV 2017）

        FL = -alpha_t · (1 - pt)^gamma · log(pt)

        v2 修正（GNN 全阴性退化修复）：
        - 原实现的 alpha 仅作全局缩放，对正负类无差别加权，等于没有
          类别权重——阳性样本极少（~4%）时模型学"恒输出阴性"即可压低
          loss，导致 F1=0、AUROC<0.5。
        - 现按标准 focal 实现 alpha_t：正类 alpha、负类 1-alpha。
        - pos_weight 显式传入时（推荐：训练集统计一次的全局值，见
          _global_pos_weight_from_loader）批间固定，loss 量级稳定；
          alpha_t 置 1 避免与 pos_weight 双重加权。
        - pos_weight=None 且 adaptive_pos_weight=True 时按批内不平衡度
          动态加权（旧行为，仅向后兼容保留——小 batch 下波动大）。
        """
        import torch
        import torch.nn.functional as F

        preds = preds.clamp(min=1e-7, max=1 - 1e-7)
        targets = targets.float()

        if pos_weight is not None:
            # 全局固定 pos_weight（训练集统计一次）：消除批间动态波动
            weight = torch.where(
                targets == 1,
                torch.full_like(targets, float(pos_weight)),
                torch.ones_like(targets))
            # 均值归一化：保持 n_neg:n_pos 相对权重比，约束整体损失量级
            weight = weight / weight.mean()
            bce = F.binary_cross_entropy(
                preds, targets, reduction='none', weight=weight)
            alpha_t = torch.ones_like(targets)
        elif adaptive_pos_weight:
            n_pos = targets.sum()
            n_neg = (1.0 - targets).sum()
            pos_weight = n_neg / n_pos.clamp(min=1.0)
            weight = torch.where(
                targets == 1, pos_weight, torch.ones_like(targets))
            # 均值归一化：保持 n_neg:n_pos 的相对权重比，但约束整体
            # 损失量级，避免大 pos_weight 放大梯度导致训练不稳
            weight = weight / weight.mean()
            bce = F.binary_cross_entropy(
                preds, targets, reduction='none', weight=weight)
            alpha_t = torch.ones_like(targets)
        else:
            bce = F.binary_cross_entropy(preds, targets, reduction='none')
            alpha_t = torch.where(
                targets == 1,
                torch.full_like(targets, alpha),
                torch.full_like(targets, 1.0 - alpha))

        pt = torch.where(targets == 1, preds, 1 - preds)
        loss = alpha_t * (1 - pt) ** gamma * bce

        return loss.mean()


def _train_gnn_once(predictor, model, train_loader, val_loader, device,
                        n_epochs=20, learning_rate=0.001, use_focal_loss=True,
                        alpha=0.25, gamma=2.0, early_stop_patience=7):
        """单次GNN训练（修复：使用mask忽略患者节点；支持edge_attr；支持focal_loss参数；传递batch索引）

        M 级审计修复（早停）：原实现固定跑满 n_epochs——超参搜索的
        Optuna/GridSearch 每个候选都全量训练，验证 AUROC 早已平台化的
        候选浪费整个 epoch 预算（网格 27 配置 × 20 epoch 无一早退）。
        现与主训练循环一致加 patience 早停（默认 7：搜索 trial 预算
        20 epoch 下的经验值——前 3-5 epoch 陡升，之后平台化）。

        文献支撑：Zhu et al., GAST 2024 (PLOS ONE)
        """
        import torch
        import torch.nn.functional as F

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_auroc = 0.0
        patience_counter = 0

        # 全局固定 pos_weight：训练集统计一次（批内动态重算会使
        # loss 量级随批次波动，见 _focal_loss docstring）
        global_pos_weight = _global_pos_weight_from_loader(train_loader)

        for epoch in range(n_epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                # 调用forward，根据是否有edge_attr选择方式，并传递batch索引
                if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                    risk_scores, _ = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
                else:
                    risk_scores, _ = model(batch.x, batch.edge_index, batch=batch.batch)

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

            val_metrics = predictor._evaluate_gnn(model, val_loader, device)
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                patience_counter = 0
            else:
                patience_counter += 1
                if early_stop_patience > 0 and \
                        patience_counter >= early_stop_patience:
                    break

        return best_auroc
