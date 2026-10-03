"""SEIR机制模型参数不确定性量化（已弃用）

.. deprecated:: 问题三
   本子包与 ``seir/inference/`` 提供等价的贝叶斯推断能力，但缺少 HMC 可微
   ODE 支撑与完善的收敛诊断。新代码应使用 ``seir/inference/`` 入口：

       from tb_risk.seir import get_inference_engine
       Engine = get_inference_engine('v4')  # BayesianInferenceV4

   ``SEIRParameterUncertainty`` 暂保留以兼容 GUI 训练面板的 UQ 工作流
   （``gui/training_panel/ml_tab/_uncertainty.py``），后续将统一迁移。

使用MCMC/NUTS对SEIR参数进行贝叶斯推断，
生成后验预测分布和R0置信度。

**v3.0 7-房室模型参数先验**（文献校准）：
  参数         旧值（流感尺度）       TB 7-房室值               先验分布
  ρ_fast (快进展)  1.0/day          ~0.000274/day (0.1/年)    Gamma(2, 0.05/365)
  ρ_react (慢再激活) —               ~1.1e-5/day (0.004/年)  Gamma(2, 0.002/365)
  σ_clear (自清除)  —               ~0.00548/day (2.0/年)    Gamma(2, 1/365)
  γ (恢复率)       2.0/day          0.05/day (传染期~20天)     Gamma(2, 0.025)

文献：
  Rosato C et al. (2022); Chakraborty T et al. (2025)
  Vynnycky E, Fine PEM. (1997) Epidemiol Rev 19(2).
  Ragonnet R et al. (2021) PLoS Comput Biol.
  TIME模型: Houben RMGJ et al. BMC Med. 2016;14:56 (PMID: 27012808)
  快进展率: Andrews JR et al. Nat Rev Dis Primers. 2012;8(1):10710
  自清除: Horton KC et al. PNAS. 2023;120(47):e2221186120 (PMID: 37963250)
  亚临床TB: Emery JC et al. eLife. 2023;12:e82469 (PMID: 38109277)

模块结构（拆分自原 monolithic seir/uncertainty.py，业务逻辑不变）：
  - _core.py            : 参数先验、初始化、对数先验 + v3/v4 固定参数常量（核心 Mixin）
  - _mcmc.py            : 对数似然 + Metropolis-Hastings 自适应 MCMC 采样（Mixin）
  - _diagnostics.py     : R-hat 与 ESS 收敛诊断（Mixin）
  - _posterior.py       : v1 后验预测分布（Mixin）
  - _posterior_v3.py    : v3 9-房室后验预测分布 + R0 + ODE 轨迹（Mixin）
  - _posterior_v4.py    : v4 270D 分层后验预测分布 + NGM R0 + ODE 轨迹（Mixin）
  - _warnings.py        : 高置信度传播风险警告（Mixin）
  - __init__.py (本文件): 组合 Mixin 为单一类 SEIRParameterUncertainty + 再导出
"""

import warnings

from ._core import _UncertaintyCore
from ._mcmc import _MCMCMixin
from ._diagnostics import _DiagnosticsMixin
from ._posterior import _PosteriorPredictiveMixin
from ._posterior_v3 import _PosteriorV3Mixin
from ._posterior_v4 import _PosteriorV4Mixin
from ._warnings import _WarningsMixin


class SEIRParameterUncertainty(_UncertaintyCore,
                               _MCMCMixin,
                               _DiagnosticsMixin,
                               _PosteriorPredictiveMixin,
                               _PosteriorV3Mixin,
                               _PosteriorV4Mixin,
                               _WarningsMixin):
    """SEIR机制模型参数不确定性量化（已弃用，见模块文档字符串）

    使用MCMC/NUTS对SEIR参数进行贝叶斯推断，
    生成后验预测分布和R0置信度。

    文献：[1] Rosato et al. 2022, [2] Chakraborty et al. 2025

    通过组合各功能 Mixin 保持单一类（所有 self.* 调用无需修改）。
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "seir.uncertainty.SEIRParameterUncertainty 已弃用，"
            "请改用 seir.inference.get_inference_engine('v4') "
            "（BayesianInferenceV4，集成 HMC + autodiff 可微 ODE）。"
            "GUI 训练面板 UQ 工作流暂保留此入口，后续将统一迁移。",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# 模块级再导出，保持 ``from tb_risk.seir.uncertainty import SEIR_PARAM_PRIORS`` 可用
SEIR_PARAM_PRIORS = SEIRParameterUncertainty.SEIR_PARAM_PRIORS

__all__ = ['SEIRParameterUncertainty', 'SEIR_PARAM_PRIORS']
