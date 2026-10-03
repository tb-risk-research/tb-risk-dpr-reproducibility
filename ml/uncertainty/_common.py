"""不确定性量化模块 - 共享基础设施

包含模块级常量、条件导入、日志器与基础模型类。
供子模块 ensemble / conformal / mc_dropout / swag / fusion 复用。

文献：
  [3] Lakshminarayanan et al. NeurIPS 2017
  [4] Barber et al. 2021
  [5] Adaptive CP via Bayesian UW 2026
  [6] Gal & Ghahramani, ICML 2016
  [7] Maddox et al. NeurIPS 2019
"""

import logging
import numpy as np

LOGGER = logging.getLogger("tb_risk.ml")

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    nn = None
    torch = None

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

# 当 torch 不可用时，使用 object 作为占位基类，避免模块加载阶段
# 触发 AttributeError: 'NoneType' object has no attribute 'Module'。
# 实际使用 QuantileRegressionForest 仍需 torch（见 __init__ 检查）。
_BaseModule = nn.Module if PYTORCH_AVAILABLE else object


class QuantileRegressionForest(_BaseModule):
    """分位数回归森林（PyTorch 包装器，用于 DeepEnsemble 集成）

    将 sklearn 的 GradientBoostingRegressor 包装为 PyTorch nn.Module，
    支持分位数损失函数，可作为 DeepEnsemblePredictor 的 base_model_class。

    文献：
    - Meinshausen N. "Quantile Regression Forests." JMLR, 2006.
    - Athey S, Tibshirani J, Wager S. "Generalized Random Forests." AOS, 2019.

    参数：
        input_dim: 输入特征维度
        output_dim: 输出维度（默认 1）
        n_estimators: 树的数量（默认 100）
        max_depth: 树的最大深度（默认 5）
        learning_rate: 学习率（默认 0.1）
        alpha: 分位数水平（默认 0.5，即中位数回归）
    """

    def __init__(self, input_dim=22, output_dim=1, n_estimators=100,
                 max_depth=5, learning_rate=0.1, alpha=0.5, device='cpu'):
        if not PYTORCH_AVAILABLE:
            raise RuntimeError(
                "QuantileRegressionForest 需要 PyTorch，但未安装 torch")
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self._model = None
        self._is_fitted = False

        # 注册一个虚拟参数以确保与 PyTorch 生态兼容
        self.dummy_param = nn.Parameter(
            torch.zeros(1, device=self.device), requires_grad=False)

    def forward(self, x):
        """前向传播：使用分位数回归森林预测。

        若模型尚未拟合，返回零张量作为占位符。
        """
        if not self._is_fitted or self._model is None:
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)

        x_np = x.detach().cpu().numpy()
        preds = self._model.predict(x_np)
        return torch.tensor(preds, dtype=torch.float32, device=x.device).reshape(-1, self.output_dim)

    def fit(self, X, y):
        """拟合分位数回归森林。

        参数：
            X: np.ndarray, shape (n_samples, input_dim)
            y: np.ndarray, shape (n_samples,)
        """
        from sklearn.ensemble import GradientBoostingRegressor

        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32).ravel()

        self._model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            loss='quantile',
            alpha=self.alpha,
            random_state=42,
        )
        self._model.fit(X_np, y_np)
        self._is_fitted = True
        return self
