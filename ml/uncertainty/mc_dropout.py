"""MC Dropout 不确定性量化

保持 Dropout 训练时开启，T 次随机前向传播，用方差近似不确定性。

文献：[6] Gal & Ghahramani, ICML 2016
"""

import numpy as np

from ._common import LOGGER, PYTORCH_AVAILABLE, torch


class MCDropoutGNNUncertainty:
    """MC Dropout GNN不确定性量化

    保持Dropout训练时开启，T次随机前向传播，用方差近似不确定性。

    文献：[6] Gal & Ghahramani, ICML 2016
    """

    def __init__(self, gnn_model, n_samples=30):
        self.gnn_model = gnn_model
        self.n_samples = n_samples
        self.mc_history = []

    def predict_mc(self, x, edge_index, edge_attr=None):
        """MC Dropout预测

        返回：
            dict: {'mean', 'variance', 'std', 'samples',
                   'aleatoric_variance', 'epistemic_variance'}
        """
        if not PYTORCH_AVAILABLE:
            return {'error': 'PyTorch不可用'}

        self.gnn_model.train()
        device = next(self.gnn_model.parameters()).device
        x = x.to(device) if hasattr(x, 'to') else x
        edge_index = edge_index.to(device) if hasattr(edge_index, 'to') else edge_index
        if edge_attr is not None:
            edge_attr = edge_attr.to(device) if hasattr(edge_attr, 'to') else edge_attr

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                try:
                    risk, _ = self.gnn_model(x, edge_index)
                except Exception as e:
                    LOGGER.warning("MC Dropout 调用 forward(x, edge_index) 失败: %s", e)
                    try:
                        risk = self.gnn_model(x, edge_index,
                                              edge_attr=edge_attr)
                    except Exception as e2:
                        LOGGER.warning("MC Dropout 调用 forward(x, edge_index, edge_attr=...) 失败: %s", e2)
                        try:
                            risk = self.gnn_model(x, edge_index)
                        except Exception as e3:
                            LOGGER.error("MC Dropout 三次调用均失败: %s", e3)
                            return {'error': f'MC Dropout 三次调用均失败: {e3}'}
                predictions.append(risk.squeeze().cpu().numpy())

        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        variance = predictions.var(axis=0, ddof=1)
        std = np.sqrt(variance + 1e-10)

        ci_95_lower = np.percentile(predictions, 2.5, axis=0)
        ci_95_upper = np.percentile(predictions, 97.5, axis=0)

        return {
            'mean': float(mean) if np.isscalar(mean) else mean.tolist(),
            'std': float(std) if np.isscalar(std) else std.tolist(),
            'variance': float(variance) if np.isscalar(variance)
                        else variance.tolist(),
            'ci_95': [float(ci_95_lower) if np.isscalar(ci_95_lower)
                       else ci_95_lower.tolist(),
                      float(ci_95_upper) if np.isscalar(ci_95_upper)
                       else ci_95_upper.tolist()],
            'n_samples': self.n_samples,
            'coefficient_of_variation': (float(std / (mean + 1e-10))
                                         if np.isscalar(std) else (std / (mean + 1e-10)).tolist()),
            'uncertainty_level': ('high' if float(std) > 0.15 else
                                  ('medium' if float(std) > 0.07 else 'low'))
                                 if np.isscalar(std) else
                                 (['high' if s > 0.15 else ('medium' if s > 0.07 else 'low')
                                   for s in np.asarray(std).ravel()]),
        }

    def predict_batch_mc(self, X_list, edge_index_list):
        """批量MC Dropout预测"""
        results = []
        for x, ei in zip(X_list, edge_index_list):
            result = self.predict_mc(x, ei)
            results.append(result)
        return results

    def uncertainty_decomposition(self, x, edge_index, y_true=None):
        """不确定性分解：偶然 vs 认知

        对每个节点分别估计不确定性，分类为数据噪声(aleatoric)
        还是模型不确定(epistemic)。
        """
        mc_result = self.predict_mc(x, edge_index)

        epistemic = float(mc_result['variance'])
        aleatoric = 0.0

        if y_true is not None:
            y_true_np = np.array(y_true).flatten()
            residual_var = float(np.var(y_true_np - mc_result['mean']))
            aleatoric = max(0.0, residual_var - epistemic)

        total = epistemic + aleatoric

        return {
            'total_variance': total,
            'aleatoric_variance': aleatoric,
            'epistemic_variance': epistemic,
            'epistemic_ratio': epistemic / (total + 1e-10),
            'aleatoric_ratio': aleatoric / (total + 1e-10),
            'dominant_source': 'epistemic' if epistemic > aleatoric
                               else 'aleatoric',
            'needs_more_data': epistemic > aleatoric and epistemic > 0.05,
        }
