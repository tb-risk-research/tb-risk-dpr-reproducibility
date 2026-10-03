"""深度集成不确定性估计

训练多个独立种子的同质模型，通过预测分布估计不确定性。

文献：[3] Lakshminarayanan et al. NeurIPS 2017
"""

import os
import numpy as np

from ._common import (LOGGER, PYTORCH_AVAILABLE, JOBLIB_AVAILABLE,
                      torch, nn, joblib, QuantileRegressionForest)


class DeepEnsemblePredictor:
    """深度集成不确定性估计

    训练多个独立种子的同质模型，通过预测分布估计不确定性。

    文献：[3] Lakshminarayanan et al. NeurIPS 2017
    """

    def __init__(self, base_model_class=None, model_kwargs=None,
                 n_ensembles=7, device='cpu'):
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("DeepEnsemblePredictor 需要 PyTorch，但未安装 torch")
        self.base_model_class = base_model_class or nn.Linear
        self.model_kwargs = model_kwargs or {}
        self.n_ensembles = n_ensembles
        self.device = device

        self.models = []
        self.is_trained = False
        self.ensemble_stats = {}
        self.device = torch.device(device) if device is not None else torch.device('cpu')

    def initialize_models(self, input_dim=22, output_dim=1):
        """初始化集成模型（v2.0：使用 base_model_class 或默认 MLP）。

        若构造函数传入 base_model_class，则使用它创建各集成成员；
        否则回退到默认 MLP 架构（与主模型对齐）。
        """
        self.models = []
        for i in range(self.n_ensembles):
            if PYTORCH_AVAILABLE:
                torch.manual_seed(42 + i * 137)

            if self.base_model_class is not None and self.base_model_class is not nn.Linear:
                # 使用用户指定的 base_model_class（如 XGBoost 包装器、分位数回归森林等）
                try:
                    model = self.base_model_class(
                        input_dim=input_dim, output_dim=output_dim,
                        **self.model_kwargs)
                except TypeError:
                    model = self.base_model_class(**self.model_kwargs)
            else:
                # 默认 MLP 架构（与主模型架构对齐）
                model = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, output_dim),
                )
            self.models.append(model)
        return self.models

    def train_ensemble(self, X, y, n_epochs=30, lr=0.001,
                        bootstrap_ratio=0.8, log_callback=None):
        """训练集成模型（bootstrap采样+不同初始化）"""
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32).reshape(-1, 1)
        n_samples = len(X_np)

        if not PYTORCH_AVAILABLE:
            return False

        criterion = nn.BCEWithLogitsLoss()

        histories = []
        for i, model in enumerate(self.models):
            bs_idx = np.random.choice(n_samples,
                                       size=int(n_samples * bootstrap_ratio),
                                       replace=True)
            X_bs = torch.tensor(X_np[bs_idx], device=self.device)
            y_bs = torch.tensor(y_np[bs_idx], device=self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model = model.to(self.device)
            model.train()
            losses = []

            for epoch in range(n_epochs):
                optimizer.zero_grad()
                logits = model(X_bs)
                loss = criterion(logits, y_bs)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            histories.append({'model_idx': i, 'final_loss': losses[-1]})

            if log_callback and (i + 1) % 2 == 0:
                log_callback(f"[Ensemble] Model {i+1}/{self.n_ensembles} trained")

        self.is_trained = True
        self.ensemble_stats['training_histories'] = histories
        return True

    def predict_with_uncertainty(self, X):
        """集成预测并估计不确定性

        返回：
            dict: {
                'mean': 均值预测,
                'std': 认知不确定性(标准差),
                'predictions': 所有模型预测,
                'ci_90': 90%置信区间,
                'disagreement': 模型间分歧度
            }
        """
        if not self.is_trained or not PYTORCH_AVAILABLE:
            return {'error': '模型未训练或torch不可用'}

        X_t = torch.tensor(np.array(X, dtype=np.float32), device=self.device)

        predictions = []
        with torch.no_grad():
            for model in self.models:
                model = model.to(self.device)
                model.eval()
                logits = model(X_t)
                prob = torch.sigmoid(logits).squeeze().cpu()
                predictions.append(prob.numpy())

        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0, ddof=1)
        ci_90_lower = np.percentile(predictions, 5, axis=0)
        ci_90_upper = np.percentile(predictions, 95, axis=0)

        ci_width = ci_90_upper - ci_90_lower

        return {
            'mean': mean.tolist() if np.isscalar(mean) else mean.tolist(),
            'std': std.tolist() if np.isscalar(std) else std.tolist(),
            'predictions': [p.tolist() for p in predictions],
            'ci_90': [ci_90_lower.tolist() if np.isscalar(ci_90_lower)
                       else ci_90_lower.tolist(),
                      ci_90_upper.tolist() if np.isscalar(ci_90_upper)
                       else ci_90_upper.tolist()],
            'ci_width': ci_width.tolist() if np.isscalar(ci_width)
                         else ci_width.tolist(),
            'disagreement': float(std.mean()) if hasattr(std, 'mean') else float(std),
            'ensemble_size': self.n_ensembles,
        }

    # ========== 持久化方法 ==========

    def save(self, filepath):
        """保存深度集成模型到文件

        将所有成员模型序列化为单个 joblib 文件，记录版本信息。
        支持训练中断后恢复——保存训练历史以支持增量训练。

        参数：
            filepath (str): 保存路径，建议使用 .joblib 扩展名

        返回：
            bool: 保存是否成功
        """
        if not JOBLIB_AVAILABLE:
            LOGGER.warning("joblib 未安装，无法保存 DeepEnsemble 模型")
            return False
        if not self.is_trained:
            LOGGER.warning("DeepEnsemble 模型未训练，无法保存")
            return False

        allowed_exts = ('.joblib', '.pkl', '.pickle')
        if not filepath.lower().endswith(allowed_exts):
            LOGGER.warning("拒绝保存到非模型文件扩展名: %s (仅支持 %s)", filepath, allowed_exts)
            return False

        try:
            # 收集所有成员模型的状态字典
            member_states = []
            for i, model in enumerate(self.models):
                if hasattr(model, 'state_dict'):
                    # 提取模型架构信息（用于加载时重建）
                    arch_info = {}
                    state = model.state_dict()
                    # 从 state_dict 推断 input_dim 和 output_dim
                    # 遍历所有 weight 层，取最后一层的 output_dim（即模型最终输出维度）
                    for key in state:
                        if 'weight' in key and len(state[key].shape) == 2:
                            arch_info['output_dim'] = state[key].shape[0]
                            arch_info['input_dim'] = state[key].shape[1]
                            # 不 break，继续遍历以获取最后一层的信息
                    # 同时记录首层的 input_dim（模型实际输入维度）
                    for key in state:
                        if 'weight' in key and len(state[key].shape) == 2:
                            arch_info['input_dim'] = state[key].shape[1]
                            break
                    member_states.append({
                        'index': i,
                        'state_dict': state,
                        'type': type(model).__name__,
                        'arch_info': arch_info,
                    })
                elif hasattr(model, '_model') and model._model is not None:
                    # sklearn 包装器（如 QuantileRegressionForest）
                    member_states.append({
                        'index': i,
                        'sklearn_model': model._model,
                        'type': type(model).__name__,
                        'input_dim': getattr(model, 'input_dim', None),
                        'output_dim': getattr(model, 'output_dim', None),
                        'n_estimators': getattr(model, 'n_estimators', None),
                        'max_depth': getattr(model, 'max_depth', None),
                        'learning_rate': getattr(model, 'learning_rate', None),
                        'alpha': getattr(model, 'alpha', None),
                    })
                else:
                    # nn.Sequential 等无 state_dict 的模型，尝试用 pickle
                    member_states.append({
                        'index': i,
                        'raw_model': model,
                        'type': type(model).__name__,
                    })

            save_data = {
                'version': '2.0',
                'model_type': 'DeepEnsemblePredictor',
                'n_ensembles': self.n_ensembles,
                'is_trained': self.is_trained,
                'ensemble_stats': self.ensemble_stats,
                'member_states': member_states,
                'base_model_class_name': self.base_model_class.__name__,
                'model_kwargs': self.model_kwargs,
            }

            joblib.dump(save_data, filepath)
            LOGGER.info("DeepEnsemble 模型已保存到: %s (成员数=%d)", filepath, self.n_ensembles)
            return True

        except Exception as e:
            LOGGER.warning("DeepEnsemble 模型保存失败: %s", e, exc_info=True)
            return False

    def load(self, filepath, device=None):
        """从文件加载深度集成模型

        恢复所有成员模型及其训练状态，支持跨设备加载。

        参数：
            filepath (str): 模型文件路径
            device (str|None): 目标设备，None 则自动检测

        返回：
            bool: 加载是否成功
        """
        if not JOBLIB_AVAILABLE:
            LOGGER.warning("joblib 未安装，无法加载 DeepEnsemble 模型")
            return False

        allowed_exts = ('.joblib', '.pkl', '.pickle')
        if not filepath.lower().endswith(allowed_exts):
            LOGGER.warning("拒绝加载非模型文件: %s (仅支持 %s)", filepath, allowed_exts)
            return False

        if not os.path.exists(filepath):
            LOGGER.warning("DeepEnsemble 模型文件不存在: %s", filepath)
            return False

        try:
            LOGGER.info("加载 DeepEnsemble 模型: %s", filepath)
            save_data = joblib.load(filepath)

            if not isinstance(save_data, dict) or 'model_type' not in save_data:
                LOGGER.warning("无效的 DeepEnsemble 模型文件格式")
                return False

            if save_data.get('model_type') != 'DeepEnsemblePredictor':
                LOGGER.warning("模型类型不匹配: 期望 DeepEnsemblePredictor, 实际 %s",
                             save_data.get('model_type'))
                return False

            # 验证版本兼容性
            saved_version = save_data.get('version', 'unknown')
            LOGGER.info("模型版本: %s", saved_version)

            if device is None:
                device = 'cuda' if PYTORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
            self.device = torch.device(device) if device is not None else torch.device('cpu')

            # 恢复成员模型
            self.models = []
            for member in save_data['member_states']:
                if 'state_dict' in member:
                    # PyTorch 模型：使用保存的架构信息重建
                    arch_info = member.get('arch_info', {})
                    model = self._rebuild_single_model(arch_info=arch_info)
                    model.load_state_dict(member['state_dict'])
                    model = model.to(self.device)
                    self.models.append(model)
                elif 'sklearn_model' in member:
                    # sklearn 包装器模型
                    model = QuantileRegressionForest(
                        input_dim=member.get('input_dim', 22),
                        output_dim=member.get('output_dim', 1),
                        n_estimators=member.get('n_estimators', 100),
                        max_depth=member.get('max_depth', 5),
                        learning_rate=member.get('learning_rate', 0.1),
                        alpha=member.get('alpha', 0.5),
                        device=device,
                    )
                    model._model = member['sklearn_model']
                    model._is_fitted = True
                    self.models.append(model)
                elif 'raw_model' in member:
                    self.models.append(member['raw_model'])

            self.n_ensembles = save_data.get('n_ensembles', len(self.models))
            self.is_trained = save_data.get('is_trained', True)
            self.ensemble_stats = save_data.get('ensemble_stats', {})

            LOGGER.info("DeepEnsemble 模型加载成功: %d 个成员", len(self.models))
            return True

        except Exception as e:
            LOGGER.warning("DeepEnsemble 模型加载失败: %s", e, exc_info=True)
            return False

    def _rebuild_single_model(self, arch_info=None):
        """重建单个成员模型架构（用于加载 state_dict）

        参数：
            arch_info (dict|None): {'input_dim': int, 'output_dim': int}，
                从保存的 state_dict 推断的架构信息
        """
        input_dim = arch_info.get('input_dim', 22) if arch_info else 22
        output_dim = arch_info.get('output_dim', 1) if arch_info else 1

        if self.base_model_class is not None and self.base_model_class is not nn.Linear:
            try:
                return self.base_model_class(
                    input_dim=input_dim, output_dim=output_dim,
                    **self.model_kwargs)
            except TypeError:
                return self.base_model_class(**self.model_kwargs)
        else:
            # 默认 MLP 架构（使用推断的 input_dim）
            return nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, output_dim),
            )

    def save_checkpoint(self, filepath, epoch, member_idx, loss):
        """保存训练检查点（支持中断后恢复）

        参数：
            filepath (str): 检查点文件路径
            epoch (int): 当前 epoch
            member_idx (int): 当前训练的成员索引
            loss (float): 当前损失值
        """
        if not JOBLIB_AVAILABLE:
            return False
        try:
            ckpt = {
                'version': '2.0',
                'checkpoint_type': 'DeepEnsemble',
                'epoch': epoch,
                'member_idx': member_idx,
                'loss': loss,
                'n_ensembles': self.n_ensembles,
                'trained_models': [],
            }
            for i, model in enumerate(self.models):
                if hasattr(model, 'state_dict'):
                    ckpt['trained_models'].append({
                        'index': i,
                        'state_dict': model.state_dict(),
                    })
                elif hasattr(model, '_model') and model._model is not None:
                    ckpt['trained_models'].append({
                        'index': i,
                        'sklearn_model': model._model,
                    })
            joblib.dump(ckpt, filepath)
            return True
        except Exception as e:
            LOGGER.warning("检查点保存失败: %s", e, exc_info=True)
            return False

    def load_checkpoint(self, filepath):
        """从检查点恢复训练状态

        返回：
            dict|None: {'epoch', 'member_idx', 'loss'} 或 None
        """
        if not JOBLIB_AVAILABLE or not os.path.exists(filepath):
            return None
        try:
            ckpt = joblib.load(filepath)
            if ckpt.get('checkpoint_type') != 'DeepEnsemble':
                return None

            # 恢复已训练的模型
            for trained in ckpt.get('trained_models', []):
                idx = trained['index']
                if idx < len(self.models):
                    if 'state_dict' in trained:
                        if hasattr(self.models[idx], 'load_state_dict'):
                            self.models[idx].load_state_dict(trained['state_dict'])
                    elif 'sklearn_model' in trained:
                        if hasattr(self.models[idx], '_model'):
                            self.models[idx]._model = trained['sklearn_model']
                            self.models[idx]._is_fitted = True

            return {
                'epoch': ckpt['epoch'],
                'member_idx': ckpt['member_idx'],
                'loss': ckpt.get('loss', 0),
            }
        except Exception as e:
            LOGGER.warning("检查点加载失败: %s", e, exc_info=True)
            return None
