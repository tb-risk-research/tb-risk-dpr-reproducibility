#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
维度一：本土流行病学参数校准器（深化版）

支持两种校准模式：
1. 贝叶斯校准（优先）：使用 MCMC 推断参数后验分布
2. 点估计回退：当 scipy/numpy 不可用时使用硬编码公式
"""

# 全国结核病平均发病率（每10万人），数据来源：WHO Global TB Report 2024 中国数据
# 用于本土化校准的基线参考值，计算本地发病率相对全国的偏移量
DEFAULT_NATIONAL_INCIDENCE_PER_100K = 62.0

import math

from ..config import ConfigProxy

try:
    from .bayesian_calibrator import BayesianCalibrator
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False


class LocalEpiCalibrator:
    """维度一：本土流行病学参数校准器（深化版）

    输入：无（内部从配置读取）
    输出：校正参数包

    核心参数：
    - base_incidence: 121/10万（2024年筛查检出率）
    - latent_prevalence_family: 8-12%（本地IGRA调查）
    - spatial_correlation σ²ω: 1.806（Liu et al., 2025空间模型）
    - diagnosis_delay_median: 29天（就诊延迟中位数）

    对核心系统的影响：
    - MLRiskPredictor._generate_synthetic_training_data:
      flp_percentage分布和log_odds截距项使用本地参数
    - TB_Risk_Assessment.assess_risk:
      base_infection_probability基线参考检出率调整
    """

    def __init__(self, config):
        self._cfg = config
        self._bayesian_calibrator = None
        self._calibration_result = None

    @staticmethod
    def _safe_float(value, default=None):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @property
    def base_incidence_per_100k(self):
        return self._cfg.get('epidemiology.base_incidence_per_100k', 121.0)

    @property
    def latent_prevalence_family_min(self):
        return self._cfg.get('epidemiology.latent_prevalence_family_min', 0.08)

    @property
    def latent_prevalence_family_max(self):
        return self._cfg.get('epidemiology.latent_prevalence_family_max', 0.12)

    @property
    def latent_prevalence_family_default(self):
        return self._cfg.get('epidemiology.latent_prevalence_family_default', 0.10)

    @property
    def hrsp_percentage_default(self):
        return self._cfg.get('epidemiology.hrsp_percentage_default', 18.0)

    @property
    def spatial_correlation_sigma2_omega(self):
        return self._cfg.get('epidemiology.spatial_correlation_sigma2_omega', 1.806)

    @property
    def diagnosis_delay_median_days(self):
        return self._cfg.get('epidemiology.diagnosis_delay_median_days', 29)

    @property
    def treatment_success_rate(self):
        return self._cfg.get('epidemiology.treatment_success_rate', 0.92)

    @property
    def dr_resistance_rate(self):
        return self._cfg.get('epidemiology.dr_resistance_rate', 0.08)

    def get_age_distribution(self):
        """返回本土年龄分布权重"""
        d = self._cfg.get_dict('epidemiology.local_age_distribution')
        if not d:
            d = {
                'child_under_5': 0.03, 'child_5_14': 0.05,
                'young_adult': 0.45, 'adult': 0.35, 'elderly': 0.12
            }
        return d

    def calibrate_synthetic_params(self):
        """生成用于校准合成数据生成的参数包

        返回一个字典，包含替换默认值的本土化参数。
        MLRiskPredictor._generate_synthetic_training_data 中：
          - flp_percentage的随机分布参数替换
          - log_odds截距项参考本地发病率偏移
        """
        return {
            'flp_percentage_min': self.latent_prevalence_family_min * 100,
            'flp_percentage_max': self.latent_prevalence_family_max * 100,
            'flp_percentage_default': self.latent_prevalence_family_default * 100,
            'hrsp_percentage_default': self.hrsp_percentage_default,
            'base_incidence_per_100k': self.base_incidence_per_100k,
            'diagnosis_delay_median': self.diagnosis_delay_median_days,
            'dr_resistance_rate': self.dr_resistance_rate,
            'spatial_correlation_sigma2_omega': self.spatial_correlation_sigma2_omega,
            'age_distribution': self.get_age_distribution(),
            'treatment_success_rate': self.treatment_success_rate
        }

    def get_calibrated_parameters(self):
        """获取校准后的参数包（兼容 KaramayCalibrator 回退路径）。

        LocalEpiCalibrator 的核心参数化方法是 calibrate_synthetic_params，
        但 KaramayCalibrator 在 run_calibration 回退路径中调用
        get_calibrated_parameters()。补充此方法以消除 AttributeError。
        """
        return self.calibrate_synthetic_params()

    def get_infection_probability_baseline_offset(self):
        """计算感染概率基线偏移量（贝叶斯校准优先，回退到硬编码公式）。

        基于本土发病率与默认参数之间的差异，返回偏移量 (%)。

        贝叶斯模式：使用后验分布的 base_incidence_per_100k 均值，
        通过对数比计算偏移，比线性比更稳健。

        文献支撑：
        - Kennedy & O'Hagan (2001): Bayesian calibration of computer models
        - Gelman et al. (2013): Bayesian Data Analysis, 3rd ed.
        """
        # 优先使用贝叶斯校准结果
        if self._calibration_result is not None and not self._calibration_result.is_fallback:
            posterior_mean = self._calibration_result.posterior_mean.get(
                'base_incidence_per_100k', 121.0)
            default_rate = DEFAULT_NATIONAL_INCIDENCE_PER_100K
            if default_rate <= 0:
                return 0.0
            log_offset = math.log(max(posterior_mean, 1.0) / default_rate) * 5.0
            return max(-10.0, min(15.0, log_offset))

        # 回退：硬编码公式（保留用于向后兼容和 scipy 不可用场景）
        local_rate = self.base_incidence_per_100k
        default_rate = DEFAULT_NATIONAL_INCIDENCE_PER_100K
        if default_rate <= 0:
            return 0.0
        offset = (local_rate / default_rate - 1.0) * 5.0
        return max(-10.0, min(15.0, offset))

    def run_bayesian_calibration(self, localizer=None, observed_data=None,
                                  n_iterations=4000, n_burnin=2000,
                                  n_chains=4, random_seed=42):
        """执行贝叶斯校准（需要 KaramayLocalizer 实例）。

        参数：
            localizer: KaramayLocalizer 实例（必需）
            observed_data: dict, 区县名到观测阳性率(%)的映射
            n_iterations: int, 每条链迭代次数
            n_burnin: int, burn-in 迭代数
            n_chains: int, 并行链数
            random_seed: int, 随机种子

        返回：
            CalibrationResult or None: 校准结果
        """
        if not BAYESIAN_AVAILABLE:
            import logging
            logging.getLogger("tb_risk.karamay").warning(
                "bayesian_calibrator 模块不可用，跳过贝叶斯校准")
            return None

        if localizer is None:
            import logging
            logging.getLogger("tb_risk.karamay").warning(
                "需要 KaramayLocalizer 实例才能执行贝叶斯校准")
            return None

        try:
            self._bayesian_calibrator = BayesianCalibrator(localizer)
            self._calibration_result = self._bayesian_calibrator.calibrate(
                observed_data=observed_data,
                n_iterations=n_iterations,
                n_burnin=n_burnin,
                n_chains=n_chains,
                random_seed=random_seed,
            )
            return self._calibration_result
        except Exception as e:
            import logging
            logging.getLogger("tb_risk.karamay").warning(
                "贝叶斯校准失败: %s，回退到点估计", e)
            self._calibration_result = None
            return None

    def get_posterior_parameter(self, name, default=None):
        """获取参数的后验均值（贝叶斯校准后）。

        参数：
            name: str, 参数名
            default: 默认值（校准未运行时返回）

        返回：
            float: 后验均值或默认值
        """
        if self._calibration_result is not None and not self._calibration_result.is_fallback:
            return self._calibration_result.posterior_mean.get(name, default)
        return default

    def get_posterior_credible_interval(self, name):
        """获取参数的 95% 可信区间（贝叶斯校准后）。

        返回：
            tuple[float, float] or None: (lower, upper) 或 None
        """
        if self._calibration_result is not None and not self._calibration_result.is_fallback:
            return self._calibration_result.posterior_ci.get(name)
        return None

    def get_scenario_defaults(self):
        """返回场景默认值中的本土化患者参数"""
        return {
            'flp_percentage': self.latent_prevalence_family_default * 100,
            'hrsp_percentage': self.hrsp_percentage_default,
            'ventilation': 3,
            'family_living_conditions': 2
        }


class KaramayCalibrator:
    """克拉玛依校准器顶层接口（L2：贝叶斯校准为默认路径）

    封装 LocalEpiCalibrator，提供统一的校准接口。
    默认使用贝叶斯 MCMC 校准，在不可用时自动回退到点估计。

    参数：
        config: ConfigProxy 配置对象
        use_bayesian: bool, 是否使用贝叶斯校准（默认 True）
    """

    def __init__(self, config=None, use_bayesian=True):
        self.use_bayesian = use_bayesian
        self._config = config
        self._calibrator = None
        if config is not None:
            self._calibrator = LocalEpiCalibrator(config)

    @property
    def calibrator(self):
        if self._calibrator is None:
            self._calibrator = LocalEpiCalibrator(ConfigProxy())
        return self._calibrator

    def run_calibration(self, localizer=None, observed_data=None, **kwargs):
        """执行校准（优先贝叶斯路径）。

        参数：
            localizer: KaramayLocalizer 实例
            observed_data: 观测数据
            **kwargs: 传递给 run_bayesian_calibration 的参数

        返回：
            CalibrationResult or dict: 校准结果
        """
        if self.use_bayesian:
            result = self.calibrator.run_bayesian_calibration(
                localizer=localizer, observed_data=observed_data, **kwargs)
            if result is not None:
                return result
            # 贝叶斯失败，回退到点估计
            import logging
            logging.getLogger("tb_risk.karamay").warning(
                "贝叶斯校准失败，回退到点估计校准")
        return self.calibrator.get_calibrated_parameters()

    def get_calibrated_parameters(self):
        """获取校准后的参数"""
        return self.calibrator.get_calibrated_parameters()

    def get_base_incidence(self):
        """获取基线发病率"""
        return self.calibrator.base_incidence_per_100k