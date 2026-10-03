"""贝叶斯 MCMC 参数推断引擎（薄再导出模块，向后兼容）。

历史版本：本模块原为单文件实现 v1/v2/v3/v4 四套平行方法。现按模型代际
拆分至 ``seir.inference`` 子包：

- ``BaseBayesianInference``           共享工具基类（_neg_binom_loglik / compute_r0_posterior / compute_hdi）
- ``BayesianInferenceV1``             4-房室 Metropolis-Hastings
- ``BayesianInferenceV2``             7-房室 Metropolis-Hastings
- ``BayesianInferenceV3``             9-房室 Metropolis-Hastings
- ``BayesianInferenceV4``             年龄×HIV×耐药分层，集成 HMC（最新版）
- ``get_inference_engine(version)``   工厂函数

向后兼容：
  ``from tb_risk.seir.bayesian import BayesianSEIRInference`` 继续可用，
  默认指向最新版 ``BayesianInferenceV4``（集成 HMC）。
  ``from tb_risk.seir.bayesian import HMCSampler`` 同样继续可用。
"""

from .inference import (
    BaseBayesianInference,
    BayesianInferenceV1,
    BayesianInferenceV2,
    BayesianInferenceV3,
    BayesianInferenceV4,
    get_inference_engine,
)
# 从 hmc_sampler 模块重新导出，保持 ``from tb_risk.seir.bayesian import HMCSampler``
# 等现有导入路径有效（拆分自原 bayesian.py）
from .hmc_sampler import HMCSampler, _safe_rhat_converged

# 向后兼容别名：BayesianSEIRInference 指向最新版 (v4，集成 HMC)
BayesianSEIRInference = BayesianInferenceV4

__all__ = [
    'BayesianSEIRInference',
    'BaseBayesianInference',
    'BayesianInferenceV1',
    'BayesianInferenceV2',
    'BayesianInferenceV3',
    'BayesianInferenceV4',
    'get_inference_engine',
    'HMCSampler',
    '_safe_rhat_converged',
]
