"""
后验驱动的治疗阶段传染性因子

基于贝叶斯MCMC后验推断，动态计算治疗各阶段的传染性因子。

v3.0 更新：涂阳/涂阴分层
=========================

将 `factors` 属性返回值从单一字典改为嵌套结构：

    {
        'smear_positive': {  # 涂阳 TB
            'pre_treatment': 1.0,         # 锚点 η_sp = 1.0
            'early_treatment': 0.55,      # 2-3 周内涂片转阴，衰减更快
            'mid_treatment':   0.30,
            'late_treatment':  0.15,
            'completed_treatment': 0.05,
        },
        'smear_negative': {  # 涂阴 TB
            'pre_treatment': 0.35,        # 基线 η_sn ≈ 0.35
            'early_treatment': 0.85,      # 基线已低，衰减相对平缓
            'mid_treatment':   0.65,
            'late_treatment':  0.40,
            'completed_treatment': 0.15,
        },
    }

依据：
  - 涂阳患者治疗开始后涂片通常在 2-3 周内转阴，因此 early_treatment 阶段
    传染性衰减更快（Ragonnet et al. 2021, CID）。
  - 涂阴患者传染性基线已低（η_sn = 0.3-0.5），治疗阶段衰减相对平缓。
  - 涂阳/涂阴自然史参数：μ_sp=0.389/yr, r_sp=0.231/yr, μ_sn=0.025/yr,
    r_sn=0.130/yr；涂阳未治疗平均持续 1.57 年，涂阴 5.35 年。

文献：
  Ragonnet R, Flegg JA, Brilleman SL, et al.
  "Revisiting the Natural History of Pulmonary Tuberculosis: A Bayesian
  Estimation of Natural Recovery and Mortality Rates."
  Clin Infect Dis. 2021;73(1):e88-e96 (PMID: 32766718)
"""

import numpy as np

# 问题八-4：涂阴相对传染性默认值单一真值源 — 从 seir._stochastic_common 导入，
# 消除本地字面量副本（原注释"与 stochastic.py 一致"但无导入关系）。
from ._stochastic_common import DEFAULT_ETA_SN

# 涂阳阶段相对系数（相对于 pre_treatment=1.0 锚点）
# 涂阳治疗开始后涂片通常在 2-3 周内转阴，early 阶段衰减更快
_SMEAR_POSITIVE_BASE = {
    'pre_treatment': 1.0,         # 锚点：未治疗涂阳传染性 η_sp = 1.0
    'early_treatment': 0.55,      # 2-3周内涂片转阴 → 快速衰减
    'mid_treatment': 0.30,
    'late_treatment': 0.15,
    'completed_treatment': 0.05,
}

# 涂阴阶段相对系数（pre_treatment = η_sn 基线）
# 涂阴基线传染性已低，治疗阶段衰减相对平缓
_SMEAR_NEGATIVE_BASE = {
    'pre_treatment': DEFAULT_ETA_SN,   # 涂阴基线 η_sn ≈ 0.35
    'early_treatment': 0.85,           # 相对 pre 缓慢衰减
    'mid_treatment': 0.65,
    'late_treatment': 0.40,
    'completed_treatment': 0.15,
}


class PosteriorDrivenInfectivity:
    """后验驱动的治疗阶段传染性因子

    v3.0：返回嵌套结构 ``{smear_positive: {...}, smear_negative: {...}}``，
    分别给出涂阳和涂阴 TB 在各治疗阶段的相对传染性系数。

    涂阳以 ``η_sp = 1.0`` 为锚点，``pre_treatment`` 恒为 1.0；
    涂阴以 ``η_sn`` 为基线，``pre_treatment = η_sn``（默认 0.35）。

    当贝叶斯推断完成后，``scaling = β/(γ·6)`` 用于按 R0 调整衰减幅度，
    ``η_sn`` 可从后验样本 ``posterior_samples['eta_sn']`` 取均值覆盖默认值。
    """

    def __init__(self, seir_inference=None):
        self.seir_inference = seir_inference
        self._posterior_factors = None

    @property
    def factors(self):
        """返回嵌套字典 ``{smear_positive: {...}, smear_negative: {...}}``"""
        if self._posterior_factors is not None:
            return self._posterior_factors

        if self.seir_inference and self.seir_inference.posterior_samples:
            beta_post = self.seir_inference.posterior_samples.get('beta')
            gamma_post = self.seir_inference.posterior_samples.get('gamma')
            if beta_post is not None and gamma_post is not None:
                if len(beta_post) == 0 or len(gamma_post) == 0:
                    return self._default_factors()
                beta_mean = float(np.mean(beta_post))
                gamma_mean = float(np.mean(gamma_post))
                if np.isnan(beta_mean) or np.isnan(gamma_mean):
                    return self._default_factors()
                scaling = beta_mean / (gamma_mean * 6.0) if gamma_mean > 0 else 1.0

                # η_sn：若后验可用则取均值，否则使用默认值
                eta_sn_post = self.seir_inference.posterior_samples.get('eta_sn')
                if eta_sn_post is not None and len(eta_sn_post) > 0:
                    eta_sn_mean = float(np.mean(eta_sn_post))
                    eta_sn = eta_sn_mean if not np.isnan(eta_sn_mean) else DEFAULT_ETA_SN
                else:
                    eta_sn = DEFAULT_ETA_SN

                # TB尺度：以 pre_treatment 为锚点，其余阶段按 scaling 缩放，
                # 与 assessment.py 保持一致的衰减逻辑：
                #   R0≈4 → scaling≈0.67
                #   R0≈6 → scaling=1.0
                # 涂阳：pre_treatment 恒为 1.0；其余阶段 = scaling * base
                sp_factors = {
                    'pre_treatment': 1.0,  # 锚点：未治疗涂阳传染性恒为 1.0
                    'early_treatment': min(scaling * _SMEAR_POSITIVE_BASE['early_treatment'], 1.0),
                    'mid_treatment': min(scaling * _SMEAR_POSITIVE_BASE['mid_treatment'], 1.0),
                    'late_treatment': min(scaling * _SMEAR_POSITIVE_BASE['late_treatment'], 1.0),
                    'completed_treatment': min(scaling * _SMEAR_POSITIVE_BASE['completed_treatment'], 1.0),
                }
                # 涂阴：pre_treatment = η_sn；其余阶段 = η_sn * scaling * base
                sn_factors = {
                    'pre_treatment': eta_sn,  # 涂阴基线传染性
                    'early_treatment': min(eta_sn * scaling * _SMEAR_NEGATIVE_BASE['early_treatment'], 1.0),
                    'mid_treatment': min(eta_sn * scaling * _SMEAR_NEGATIVE_BASE['mid_treatment'], 1.0),
                    'late_treatment': min(eta_sn * scaling * _SMEAR_NEGATIVE_BASE['late_treatment'], 1.0),
                    'completed_treatment': min(eta_sn * scaling * _SMEAR_NEGATIVE_BASE['completed_treatment'], 1.0),
                }
                self._posterior_factors = {
                    'smear_positive': sp_factors,
                    'smear_negative': sn_factors,
                }
                return self._posterior_factors

        return self._default_factors()

    @staticmethod
    def _default_factors():
        """默认嵌套结构（无后验时使用文献基线值）"""
        return {
            'smear_positive': dict(_SMEAR_POSITIVE_BASE),
            'smear_negative': dict(_SMEAR_NEGATIVE_BASE),
        }

    def update_from_mcmc(self, seir_inference):
        """从 MCMC 后验更新传染性因子（清除缓存，下次访问时重算）"""
        self.seir_inference = seir_inference
        self._posterior_factors = None
