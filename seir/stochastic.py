"""
随机SEIR模型 (Stochastic SEIR Model) — 7-房室扩展版

v2.0 改进（三个维度）：
  1. 潜伏感染快/慢分层：E → L_fast + L_slow
     - L_fast: 近期感染（~2年内），高进展率 ρ_fast
     - L_slow: 远期潜伏（>2年），低再激活率 ρ_react
     - 转换率 ρ_conv: L_fast → L_slow
  2. 自清除建模：增加 Cleared (C) 房室
     - 自清除率 σ_clear: L_fast, L_slow → C
     - 再感染易感性 β_reinf: C → L_fast
  3. 亚临床TB建模：I → I_subclinical + I_clinical
     - I_sub: 传染但无症状，相对传染性 η_sub
     - I_clin: 传染且有症状，基准传染性 η_clin=1.0

房室结构（7维）：
  [S, L_fast, L_slow, I_sub, I_clin, R, C]

文献：
  TIME模型快/慢分层: Houben RMGJ et al. BMC Med. 2016;14:56 (PMID: 27012808)
  快进展率: Andrews JR et al. Nat Rev Dis Primers. 2012;8(1):10710
  慢再激活率: Vynnycky E, Fine PEM. Epidemiol Rev. 1997;19(2):183-201
  自清除: Horton KC et al. PNAS. 2023;120(47):e2221186120 (PMID: 37963250)
  亚临床TB传染性: Emery JC et al. eLife. 2023;12:e82469 (PMID: 38109277)
  CLE标度律: Gillespie DT. (2000) J Chem Phys 113(1):297-306

模块结构（拆分自原 monolithic 文件，业务逻辑不变）：
  - _stochastic_common.py     : 房室索引与文献默认参数（共享常量）
  - _stochastic_v2.py          : 7-房室 v2 方法（Mixin）
  - _stochastic_v3.py          : 9-房室 v3 方法（Mixin）
  - _stochastic_v4.py          : 270维 v4 年龄×HIV×耐药方法（Mixin）
  - stochastic.py (本文件)     : 组合 Mixin + 公共 API re-export
"""

import numpy as np

# 共享常量与默认参数（re-export 以保持 `from tb_risk.seir.stochastic import XXX` 有效）
from ._stochastic_common import *  # noqa: F401,F403
from ._stochastic_common import __all__ as _common_all
from ._stochastic_v2 import StochasticSEIRMixinV2
from ._stochastic_v3 import StochasticSEIRMixinV3
from ._stochastic_v4 import StochasticSEIRMixinV4


class StochasticSEIRModel(StochasticSEIRMixinV2,
                          StochasticSEIRMixinV3,
                          StochasticSEIRMixinV4):
    """随机SEIR模型 — 9-房室扩展版（v3.0）

    v3.0 新增改进：
      4. 疾病状态波动：M 房室 + 回退流 (ω_reg_M, ω_reg_sub)
      5. 涂阳/涂阴分层：I_clin → I_sp + I_sn (不同死亡率/自愈率/传染性)
      6. 内源性/外源性再感染分离：β_exo 参数

    房室结构（9维）：
      [S, L_fast, L_slow, M, I_sub, I_sp, I_sn, R, C]

    支持连续时间马尔可夫链(CTMC)、随机微分方程(SDE)和确定性ODE。
    向后兼容旧版4/7-房室API。
    """

    def __init__(self, population=100, noise_scale=0.05, seed=42):
        """
        参数：
            population: 总人口数
            noise_scale: SDE噪声幅度（0=确定性ODE）
            seed: 随机数种子（默认42，保证可复现；None使用numpy全局状态）
        """
        self.population = population
        self.noise_scale = noise_scale
        self.rng = np.random.RandomState(seed)


__all__ = ["StochasticSEIRModel"] + list(_common_all)
