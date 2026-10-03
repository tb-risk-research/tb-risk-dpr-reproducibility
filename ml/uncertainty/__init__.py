"""不确定性量化模块

包含深度集成、共形预测、MC Dropout、SWAG、不确定性融合与可视化。

子模块：
  - _common: 共享基础设施（日志器、条件导入、基础模型类 QuantileRegressionForest）
  - ensemble: 深度集成预测器（DeepEnsemblePredictor）
  - conformal: 共形预测（ConformalPredictor）
  - mc_dropout: MC Dropout 不确定性量化（MCDropoutGNNUncertainty）
  - swag: SWAG 权重后验估计（SWAGEstimator）
  - fusion: 不确定性融合与可视化（UncertaintyFusionEngine, UncertaintyVisualizer）

文献：
  [3] Lakshminarayanan et al. NeurIPS 2017
  [4] Barber et al. 2021
  [5] Adaptive CP via Bayesian UW 2026
  [6] Gal & Ghahramani, ICML 2016
  [7] Maddox et al. NeurIPS 2019
"""

from ._common import QuantileRegressionForest
from .ensemble import DeepEnsemblePredictor
from .conformal import ConformalPredictor
from .mc_dropout import MCDropoutGNNUncertainty
from .swag import SWAGEstimator
from .fusion import UncertaintyFusionEngine, UncertaintyVisualizer

__all__ = [
    "QuantileRegressionForest",
    "DeepEnsemblePredictor",
    "ConformalPredictor",
    "MCDropoutGNNUncertainty",
    "SWAGEstimator",
    "UncertaintyFusionEngine",
    "UncertaintyVisualizer",
]
