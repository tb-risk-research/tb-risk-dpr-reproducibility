"""SWAG (Stochastic Weight Averaging Gaussian) 权重后验估计

收集 SGD 后期轨迹，构建权重的低秩高斯近似。

文献：[7] Maddox et al. NeurIPS 2019
"""

import os
import numpy as np

from ._common import (LOGGER, PYTORCH_AVAILABLE, JOBLIB_AVAILABLE,
                      torch, joblib)


class SWAGEstimator:
    """SWAG (Stochastic Weight Averaging Gaussian) 权重后验估计

    收集SGD后期轨迹，构建权重的低秩高斯近似。

    文献：[7] Maddox et al. NeurIPS 2019
    """

    def __init__(self, model, n_models=20, max_rank=5,
                 update_freq=1, scale=0.5):
        self.base_model = model
        self.n_models = n_models
        self.max_rank = max_rank
        self.update_freq = update_freq
        self.scale = scale

        self.swag_mean = {}
        self.swag_dev = {}
        self.swag_var = {}
        self.n_collected = 0
        self.collection_complete = False

    def collect_model(self, model):
        """收集当前权重快照，更新SWAG统计量"""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            w = param.detach().cpu()

            if name not in self.swag_mean:
                self.swag_mean[name] = w.clone().flatten().numpy()
                self.swag_dev[name] = np.zeros((self.max_rank,
                                                 w.numel()), dtype=np.float32)
                self.swag_var[name] = np.zeros(w.numel(), dtype=np.float32)
            else:
                w_np = w.flatten().numpy()
                prev_mean = self.swag_mean[name].copy()

                self.swag_mean[name] = (self.n_collected * prev_mean + w_np) / (
                    self.n_collected + 1)

                dev = (w_np - self.swag_mean[name])
                rank = (self.n_collected - 1) % self.max_rank
                if self.n_collected > 0:
                    self.swag_dev[name][rank] = dev

                delta = w_np - prev_mean
                # Welford 在线方差算法：var += (w - prev_mean) * (w - new_mean)
                # 而非 delta²，后者会导致方差估计有偏
                new_mean = self.swag_mean[name]
                delta2 = delta * (w_np - new_mean)
                self.swag_var[name] = ((self.n_collected - 1) * self.swag_var[name] +
                                       delta2) / max(self.n_collected, 1)

        self.n_collected += 1
        if self.n_collected >= self.n_models:
            self.collection_complete = True

    def sample_weights(self):
        """从SWAG近似后验中采样权重

        返回：
            dict: {name: sampled_tensor}，若 torch 不可用则返回 None
        """
        if not PYTORCH_AVAILABLE:
            return None
        if self.n_collected < 2:
            return None

        sampled = {}
        for name in self.swag_mean:
            mean = self.swag_mean[name]
            dev_mat = self.swag_dev[name]
            var_diag = self.swag_var.get(name, np.ones_like(mean))
            effective_rank = min(self.max_rank, self.n_collected // 2)

            z1 = np.random.randn(effective_rank).astype(np.float32) * \
                 self.scale / np.sqrt(2)
            z2 = np.random.randn(len(mean)).astype(np.float32) * \
                 self.scale / np.sqrt(2)

            dev_component = np.zeros(len(mean), dtype=np.float32)
            for r in range(effective_rank):
                dev_component += z1[r] * dev_mat[r]

            noise_component = np.sqrt(np.maximum(var_diag, 1e-6)) * z2

            sampled_w = mean + dev_component + noise_component
            sampled[name] = torch.tensor(
                sampled_w, dtype=torch.float32,
                device=next(self.base_model.parameters()).device
                if any(self.base_model.parameters()) else torch.device('cpu'))

        return sampled

    def predict_mc(self, x, edge_index, n_samples=20):
        """使用SWAG后验采样进行MC预测

        返回：
            dict: {'mean', 'std', 'ci_95', 'swag_variance'}
        """
        if not self.collection_complete or not PYTORCH_AVAILABLE:
            return {'error': 'SWAG采样尚未完成'}

        original_state = {}
        for n, p in self.base_model.named_parameters():
            if p.requires_grad:
                original_state[n] = p.data.clone()

        predictions = []
        device = next(self.base_model.parameters()).device
        x = x.to(device) if hasattr(x, 'to') else x
        edge_index = edge_index.to(device) if hasattr(edge_index, 'to') else edge_index
        with torch.no_grad():
            for _ in range(n_samples):
                sampled_weights = self.sample_weights()
                if sampled_weights is None:
                    break
                for n, p in self.base_model.named_parameters():
                    if n in sampled_weights:
                        p.data.copy_(sampled_weights[n])

                self.base_model.eval()
                try:
                    risk, _ = self.base_model(x, edge_index)
                except Exception as e:
                    LOGGER.warning("SWAG MC 调用 forward(x, edge_index) 失败: %s", e)
                    risk = self.base_model(x, edge_index)
                predictions.append(risk.squeeze().cpu().numpy())

        for n, p in self.base_model.named_parameters():
            if n in original_state:
                p.data.copy_(original_state[n])

        if len(predictions) == 0:
            return {'error': '无有效预测'}

        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0, ddof=1)

        return {
            'mean': float(mean) if np.isscalar(mean) else mean.tolist(),
            'std': float(std) if np.isscalar(std) else std.tolist(),
            'ci_95': [float(np.percentile(predictions, 2.5, axis=0)),
                      float(np.percentile(predictions, 97.5, axis=0))],
            'swag_variance': float(np.var(predictions)),
            'n_swag_samples': len(predictions),
        }

    # ========== 持久化方法 ==========

    def save(self, filepath):
        """保存 SWAG 估计器到文件

        保存协方差矩阵（swag_mean, swag_dev, swag_var）、
        贝叶斯线性回归参数和收集状态，使模型重启后可恢复。

        参数：
            filepath (str): 保存路径，建议使用 .joblib 扩展名

        返回：
            bool: 保存是否成功
        """
        if not JOBLIB_AVAILABLE:
            LOGGER.warning("joblib 未安装，无法保存 SWAG 估计器")
            return False
        if not self.collection_complete:
            LOGGER.warning("SWAG 采样尚未完成，无法保存")
            return False

        allowed_exts = ('.joblib', '.pkl', '.pickle')
        if not filepath.lower().endswith(allowed_exts):
            LOGGER.warning("拒绝保存到非模型文件扩展名: %s (仅支持 %s)", filepath, allowed_exts)
            return False

        try:
            # 保存 base_model 的 state_dict（如果是 PyTorch 模型）
            base_state = None
            if hasattr(self.base_model, 'state_dict'):
                base_state = self.base_model.state_dict()

            save_data = {
                'version': '2.0',
                'model_type': 'SWAGEstimator',
                'n_models': self.n_models,
                'max_rank': self.max_rank,
                'update_freq': self.update_freq,
                'scale': self.scale,
                'swag_mean': self.swag_mean,
                'swag_dev': self.swag_dev,
                'swag_var': self.swag_var,
                'n_collected': self.n_collected,
                'collection_complete': self.collection_complete,
                'base_model_state': base_state,
            }
            joblib.dump(save_data, filepath)
            LOGGER.info("SWAG 估计器已保存到: %s (rank=%d, samples=%d)",
                      filepath, self.max_rank, self.n_collected)
            return True
        except Exception as e:
            LOGGER.warning("SWAG 估计器保存失败: %s", e, exc_info=True)
            return False

    def load(self, filepath, base_model=None):
        """从文件加载 SWAG 估计器

        恢复协方差矩阵和贝叶斯参数，使模型重启后可继续采样。

        参数：
            filepath (str): 模型文件路径
            base_model (nn.Module|None): 基础模型实例，None 则使用当前 base_model

        返回：
            bool: 加载是否成功
        """
        if not JOBLIB_AVAILABLE:
            LOGGER.warning("joblib 未安装，无法加载 SWAG 估计器")
            return False

        allowed_exts = ('.joblib', '.pkl', '.pickle')
        if not filepath.lower().endswith(allowed_exts):
            LOGGER.warning("拒绝加载非模型文件: %s (仅支持 %s)", filepath, allowed_exts)
            return False

        if not os.path.exists(filepath):
            LOGGER.warning("SWAG 估计器文件不存在: %s", filepath)
            return False

        try:
            LOGGER.info("加载 SWAG 估计器: %s", filepath)
            save_data = joblib.load(filepath)

            if not isinstance(save_data, dict) or save_data.get('model_type') != 'SWAGEstimator':
                LOGGER.warning("无效的 SWAG 估计器文件格式")
                return False

            self.n_models = save_data.get('n_models', 20)
            self.max_rank = save_data.get('max_rank', 5)
            self.update_freq = save_data.get('update_freq', 1)
            self.scale = save_data.get('scale', 0.5)
            self.swag_mean = save_data.get('swag_mean', {})
            self.swag_dev = save_data.get('swag_dev', {})
            self.swag_var = save_data.get('swag_var', {})
            self.n_collected = save_data.get('n_collected', 0)
            self.collection_complete = save_data.get('collection_complete', False)

            # 恢复 base_model 的 state_dict
            if base_model is not None:
                self.base_model = base_model
            base_state = save_data.get('base_model_state')
            if base_state is not None and hasattr(self.base_model, 'load_state_dict'):
                self.base_model.load_state_dict(base_state)

            LOGGER.info("SWAG 估计器加载成功: rank=%d, samples=%d",
                      self.max_rank, self.n_collected)
            return True
        except Exception as e:
            LOGGER.warning("SWAG 估计器加载失败: %s", e, exc_info=True)
            return False
