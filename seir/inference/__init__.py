"""seir.inference 子包：按模型代际组织的贝叶斯推断引擎。

导出：
  BaseBayesianInference      共享工具基类
  BayesianInferenceV1/V2/V3/V4  各代际推断类（链式继承）
  get_inference_engine(version)  工厂函数

版本说明：
  V1: 4-房室 SEIR + 自适应 MH-MCMC（已弃用）
  V2: 7-房室 SEIR + 自适应 MH-MCMC（已弃用）
  V3: 7-房室 SEIR + 多链并行 MCMC（维护模式）
  V4: 7-房室 SEIR + HMC (autodiff) + RK4 ODE（推荐版本）
"""

import warnings

from ._base import BaseBayesianInference
from .v1_inference import BayesianInferenceV1
from .v2_inference import BayesianInferenceV2
from .v3_inference import BayesianInferenceV3
from .v4_inference import BayesianInferenceV4

_ENGINES = {
    'v1': BayesianInferenceV1,
    'v2': BayesianInferenceV2,
    'v3': BayesianInferenceV3,
    'v4': BayesianInferenceV4,
}

_DEPRECATED_VERSIONS = {'v1', 'v2'}
_MAINTENANCE_VERSIONS = {'v3'}


def get_inference_engine(version='v4'):
    """按模型代际返回对应的推断引擎类。

    参数:
        version: 'v1' / 'v2' / 'v3' / 'v4'（默认 'v4'，最新版集成 HMC）

    返回:
        对应代际的推断类（未实例化）

    抛出:
        ValueError: 未知版本号
    """
    key = str(version).lower()
    if key not in _ENGINES:
        raise ValueError(
            f"未知推断引擎版本: {version!r}，可选: {sorted(_ENGINES)}")
    if key in _DEPRECATED_VERSIONS:
        warnings.warn(
            f"BayesianInference{key.upper()} 已弃用，建议使用 V4 版本（HMC+autodiff，精度与效率均更优）。"
            f"V1/V2 仅保留用于结果复现，不再接收功能更新。",
            DeprecationWarning,
            stacklevel=2,
        )
    elif key in _MAINTENANCE_VERSIONS:
        warnings.warn(
            "BayesianInferenceV3 处于维护模式，推荐迁移到 V4 版本（HMC+autodiff）。",
            FutureWarning,
            stacklevel=2,
        )
    return _ENGINES[key]


__all__ = [
    'BaseBayesianInference',
    'BayesianInferenceV1',
    'BayesianInferenceV2',
    'BayesianInferenceV3',
    'BayesianInferenceV4',
    'get_inference_engine',
]
