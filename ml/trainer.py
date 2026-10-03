"""
训练器模块

包含三阶段训练器和时序多模态GNN训练器。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreePhaseTrainer:
    """三阶段训练：同构预训练 → 异质微调 → 时序聚合"""
    
    def __init__(self, base_gnn, hetero_gnn, causal_graph=None, device='cpu'):
        self.base_gnn = base_gnn
        self.hetero_gnn = hetero_gnn
        self.causal_graph = causal_graph
        self.device = device
        self.phase = 0
        self.training_history = {'phase_1': [], 'phase_2': [], 'phase_3': []}
    
    def train_phase_1(self, dataloader, n_epochs=30, lr=0.001, log_callback=None):
        optimizer = torch.optim.Adam(self.base_gnn.parameters(), lr=lr)
        self.base_gnn.to(self.device)
        self.phase = 1
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                if hasattr(batch, 'edge_attr'):
                    risk, _ = self.base_gnn(batch.x, batch.edge_index,
                                           edge_attr=batch.edge_attr, batch=batch.batch)
                else:
                    risk, _ = self.base_gnn(batch.x, batch.edge_index, batch=batch.batch)
                loss = F.binary_cross_entropy(risk.squeeze(), batch.y.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            self.training_history['phase_1'].append(avg_loss)
            if log_callback:
                log_callback(f"[Phase 1] Epoch {epoch+1}/{n_epochs}, Loss={avg_loss:.4f}")
    
    def train_phase_2(self, hetero_graphs, n_epochs=30, lr=0.0005, log_callback=None):
        optimizer = torch.optim.Adam(self.hetero_gnn.parameters(), lr=lr)
        self.hetero_gnn.to(self.device)
        self.phase = 2
        nf, ei, ea, y_true = hetero_graphs
        # 将输入张量移动到训练设备
        nf = nf.to(self.device) if isinstance(nf, torch.Tensor) else nf
        ei = ei.to(self.device) if isinstance(ei, torch.Tensor) else ei
        ea = ea.to(self.device) if isinstance(ea, torch.Tensor) else ea
        y_true = y_true.to(self.device) if isinstance(y_true, torch.Tensor) else y_true
        for epoch in range(n_epochs):
            # HeteroTimeVaryingGNN.forward 返回 4 个值；旧代码按 3 个解包会抛 ValueError
            risk, emb, causal_loss, _ = self.hetero_gnn.forward([nf], ei, ea)
            pred_loss = F.binary_cross_entropy(risk, y_true)
            total_loss = pred_loss + self.hetero_gnn.causal_loss_weight * causal_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            self.training_history['phase_2'].append(total_loss.item())
            if log_callback:
                log_callback(f"[Phase 2] Epoch {epoch+1}/{n_epochs}, "
                           f"Pred={pred_loss.item():.4f}, Causal={causal_loss.item():.4f}")
    
    def train_phase_3(self, time_snapshots, n_epochs=20, lr=0.0003, log_callback=None):
        optimizer = torch.optim.Adam(self.hetero_gnn.parameters(), lr=lr)
        self.hetero_gnn.to(self.device)
        self.phase = 3
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for snap_batch in time_snapshots:
                nf_list, ei_list, ea_list = [], [], []
                target = None
                if isinstance(snap_batch, dict):
                    snaps = snap_batch.get('snapshots', [snap_batch])
                    target = snap_batch.get('y', None)
                else:
                    snaps = snap_batch
                for snap in snaps:
                    nf = snap['node_features']
                    ei = snap['edge_indices']
                    ea = snap['edge_attrs']
                    nf_list.append(nf.to(self.device) if isinstance(nf, torch.Tensor) else nf)
                    ei_list.append(ei.to(self.device) if isinstance(ei, torch.Tensor) else ei)
                    ea_list.append(ea.to(self.device) if isinstance(ea, torch.Tensor) else ea)
                    if target is None:
                        target = snap.get('y', None)
                risk, emb, causal_loss, time_embs = self.hetero_gnn(nf_list, ei_list, ea_list)
                if target is None:
                    target = torch.zeros_like(risk)
                else:
                    target = target.to(self.device) if isinstance(target, torch.Tensor) else \
                             torch.tensor(target, dtype=torch.float32, device=risk.device)
                pred_loss = F.binary_cross_entropy(risk, target)
                temporal_smooth = self._temporal_smoothness_loss(time_embs)
                total_loss = pred_loss + self.hetero_gnn.causal_loss_weight * causal_loss + 0.01 * temporal_smooth
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
            avg_loss = epoch_loss / max(len(time_snapshots), 1)
            self.training_history['phase_3'].append(avg_loss)
            if log_callback:
                log_callback(f"[Phase 3] Epoch {epoch+1}/{n_epochs}, Loss={avg_loss:.4f}")

    def _temporal_smoothness_loss(self, embeddings):
        if len(embeddings) <= 1:
            return torch.tensor(0.0, device=embeddings[0].device if embeddings else 'cpu')
        loss = torch.tensor(0.0, device=embeddings[0].device)
        for i in range(len(embeddings) - 1):
            loss += F.mse_loss(embeddings[i], embeddings[i + 1])
        return loss / (len(embeddings) - 1)
    
    def get_training_summary(self):
        summary = {'current_phase': self.phase}
        for phase, history in self.training_history.items():
            if history:
                summary[f'{phase}_final_loss'] = history[-1]
                summary[f'{phase}_min_loss'] = min(history)
                summary[f'{phase}_epochs'] = len(history)
        return summary


class TemporalTrainer:
    """时序多模态GNN训练器

    支持按时间顺序分批训练，结合物理约束和迁移学习。
    文献：预训练+微调、时序交叉验证
    """

    def __init__(self, multimodal_gnn, climate_interaction=None,
                 graph_builder=None, data_access=None, device='cpu'):
        self.model = multimodal_gnn
        self.climate = climate_interaction
        self.graph_builder = graph_builder
        self.data_access = data_access
        self.device = device
        self.model.to(device)
        self.training_history = {'loss': [], 'val_loss': [], 'epoch': []}

    def prepare_temporal_dataset(self, start_date, n_weeks, patient_features,
                                  contact_features_list, behavior_overrides=None):
        """准备时序数据集

        参数：
            start_date: 起始日期
            n_weeks: 周数
            patient_features: 患者特征
            contact_features_list: 各接触者特征列表
            behavior_overrides: 行为覆盖

        返回：
            list: 快照序列, list: 环境序列, list: 标签
        """
        if self.graph_builder is None:
            from .graph import EnhancedTemporalGraphBuilder
            self.graph_builder = EnhancedTemporalGraphBuilder(
                data_access=self.data_access,
                climate_interaction=self.climate,
                static_node_features=patient_features,
                n_social=len(contact_features_list) if contact_features_list else 10)

        import datetime as dt_lib
        base = dt_lib.datetime.strptime(start_date, '%Y-%m-%d') if isinstance(
            start_date, str) else start_date

        snapshots = []
        env_sequences = []
        labels = []

        for w in range(n_weeks):
            week_dates = [base + dt_lib.timedelta(days=w * 7 + d)
                         for d in range(7)]
            week_snaps = []
            week_envs = []

            for day_date in week_dates:
                snap = self.graph_builder.build_snapshot(day_date)
                week_snaps.append(snap)

                env_data = None
                if self.data_access is not None:
                    env_data = self.data_access.get_environment_data(day_date)

                env_vec = np.array([
                    min(1.0, env_data['pm10'] / 300.0) if env_data else 0.3,
                    env_data['humidity'] / 100.0 if env_data else 0.5,
                    (env_data['temp'] + 20.0) / 60.0 if env_data else 0.5,
                    min(1.0, env_data['wind_speed'] / 15.0) if env_data else 0.25,
                ], dtype=np.float32)
                week_envs.append(env_vec)

            snapshots.append(week_snaps[-1])
            env_sequences.append(np.stack(week_envs).mean(axis=0))
            labels.append(0.0 if w < n_weeks * 0.6 else
                          min(0.5, (w - n_weeks * 0.6) / (n_weeks * 0.4)))

        return snapshots, np.array(env_sequences), np.array(labels)

    def train_temporal(self, snapshot_sequence, env_sequence, labels,
                        n_epochs=50, lr=0.001, batch_size=4,
                        val_split=0.2, log_callback=None):
        """时序训练

        参数：
            snapshot_sequence: 图快照序列
            env_sequence: 环境特征序列 [T, 4]
            labels: 标签 [T]
            n_epochs: 训练轮数
            lr: 学习率
            batch_size: 批大小
            val_split: 验证集比例
            log_callback: 日志回调

        返回：
            dict: 训练摘要
        """
        n_samples = len(snapshot_sequence)
        n_val = max(1, int(n_samples * val_split))
        n_train = n_samples - n_val

        train_snaps = snapshot_sequence[:n_train]
        train_envs = env_sequence[:n_train]
        train_labels = labels[:n_train]

        val_snaps = snapshot_sequence[n_train:]
        val_envs = env_sequence[n_train:]
        val_labels = labels[n_train:]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # MultimodalGNN 各 forward 路径最终都对输出做了 torch.sigmoid，因此损失函数
        # 不能再用 BCEWithLogitsLoss（会再次 sigmoid）。改为 BCELoss 与模型输出一致。
        criterion = nn.BCELoss()

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0

            for i in range(0, n_train, batch_size):
                end_i = min(i + batch_size, n_train)
                batch_loss = 0.0

                for j in range(i, end_i):
                    context = train_snaps[:j+1]
                    env_ctx = torch.tensor(train_envs[:j+1], dtype=torch.float32,
                                           device=self.device)
                    if env_ctx.dim() == 2:
                        env_ctx = env_ctx.unsqueeze(0)

                    risk = self.model(context, env_sequence=env_ctx)

                    target = torch.tensor([[train_labels[j]]], dtype=torch.float32,
                                          device=self.device)
                    loss = criterion(risk, target)
                    batch_loss += loss

                batch_loss = batch_loss / (end_i - i)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                total_loss += float(batch_loss.item())

            avg_train_loss = total_loss / max(1, (n_train + batch_size - 1) // batch_size)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for j in range(n_val):
                    context = val_snaps[:j+1]
                    env_ctx = torch.tensor(val_envs[:j+1], dtype=torch.float32,
                                           device=self.device)
                    if env_ctx.dim() == 2:
                        env_ctx = env_ctx.unsqueeze(0)
                    risk = self.model(context, env_sequence=env_ctx)
                    target = torch.tensor([[val_labels[j]]], dtype=torch.float32,
                                          device=self.device)
                    val_loss += float(criterion(risk, target).item())

            avg_val_loss = val_loss / max(1, n_val)

            self.training_history['loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['epoch'].append(epoch + 1)

            if log_callback and (epoch + 1) % 10 == 0:
                log_callback(
                    f"Epoch {epoch+1}/{n_epochs}: "
                    f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # 真正的收敛检测：检查最近 5 个 epoch 的验证损失变化是否低于阈值
        # 文献：Prechelt (1998), "Early stopping - but when?"
        patience = 5
        min_delta = 1e-4
        val_losses = self.training_history.get('val_loss', [])
        converged = False
        if len(val_losses) >= patience + 1:
            recent = val_losses[-(patience + 1):]
            changes = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
            converged = all(c < min_delta for c in changes)

        return {
            'n_epochs': n_epochs,
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'converged': converged,
        }