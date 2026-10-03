#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
维度一：贝叶斯校准器 — NUTS/自适应MCMC 驱动的本土化参数推断

文献支撑：
- Kennedy MC, O'Hagan A. Bayesian calibration of computer models.
  JRSS-B 63(3):425-464, 2001.
- Higdon D et al. Combining field data and computer simulations for
  calibration. SIAM J Sci Comput, 2004.
- Gelman A et al. Bayesian Data Analysis (3rd ed). CRC Press, 2013.

核心思想：
将 ScoringEngine 视为"计算机模型"（computer model），给定参数 θ，
输出预测的筛查阳性率。通过贝叶斯推断，将观测数据（区县级筛查阳性率）
与先验知识结合，得到参数的后验分布。

与现有 calibrator.py 的关系：
- 本模块替换 get_infection_probability_baseline_offset 的硬编码公式
- 输出后验分布而非点估计，支持不确定性传播
- 向后兼容：后验均值可作为点估计供现有代码使用
"""

# 全国结核病平均发病率（每10万人），数据来源：WHO Global TB Report 2024 中国数据
# 用于本土化校准的基线参考值，与 calibrator.py 中定义一致
DEFAULT_NATIONAL_INCIDENCE_PER_100K = 62.0

import math
import logging
import random

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from scipy import stats as sps
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    sps = None

from ..scoring.engine import ScoringEngine

logger = logging.getLogger("tb_risk.karamay.bayesian_calibrator")


# ==================== 先验分布定义 ====================

class CalibrationPriors:
    """贝叶斯校准的先验分布定义。

    所有先验基于文献或本地数据设定：
    - base_incidence: 克拉玛依卫健委 2024 年报告，121/10万
    - oilfield_camp_factor: 介于 general(0.6) 和 closed(0.9) 之间
    - latent_prevalence: 本地 IGRA 调查，家庭潜伏感染 8-12%
    - diagnosis_delay: 中位数 29 天，对数正态适应右偏分布
    - SDOH 系数：基于 WHO SDOH Framework (Lönnroth 2009)，
      Normal(0, 0.5) 先验，中心化在零效应，允许数据驱动学习
    """

    # 参数名称列表（固定顺序）
    # 仅保留实际影响 ScoringEngine 预测阳性率的参数。
    # base_incidence_per_100k 通过 _compute_rates_from_records 中的发病率缩放
    # 进入 likelihood，因此重新纳入推断；diagnosis_delay_days 不影响 likelihood，
    # 仍作为 EPIDEMIC_PARAMS 常量保留。
    # SDOH 系数（v3.0）：通过 _compute_rates_from_records 中的个体风险乘数
    # 进入 likelihood，使后验更新能学习 SDOH 对基线感染概率的影响。
    # 文献：WHO SDOH Framework (Lönnroth et al., 2009); Belmont Report 1979
    PARAM_NAMES = [
        'base_incidence_per_100k',
        'oilfield_camp_factor',
        'latent_prevalence',
        'sdoh_housing_coefficient',    # 居住面积对感染风险的影响
        'sdoh_access_coefficient',     # 就医距离对感染风险的影响
        'sdoh_bmi_coefficient',        # BMI 对感染风险的影响
        'sdoh_income_coefficient',     # 收入水平对感染风险的影响
    ]

    # 辅助常量：不影响似然的流行病学参数（用于 calibrate_synthetic_params / 报告）
    EPIDEMIC_PARAMS = {
        'diagnosis_delay_days': 29.0,
    }

    # 参数边界（用于变换到无约束空间）
    PARAM_BOUNDS = {
        'base_incidence_per_100k': (50.0, 300.0),
        'oilfield_camp_factor': (0.3, 1.0),
        'latent_prevalence': (0.02, 0.30),
        'sdoh_housing_coefficient': (-2.0, 2.0),
        'sdoh_access_coefficient': (-2.0, 2.0),
        'sdoh_bmi_coefficient': (-2.0, 2.0),
        'sdoh_income_coefficient': (-2.0, 2.0),
    }

    # 先验标准差（用于初始 proposal 协方差的对角缩放）
    # 后验参数量纲差异极大（base_incidence ~121±30 与 latent_prevalence
    # ~0.03 同在一向量），单位矩阵提议天然低效：对前者步长过小（接受率
    # 虚高、混合极慢），对后者步长过大。按各参数先验 std 归一化后再提议
    # （Gelman et al., BDA3, Ch.11 建议）。
    # 数值来源：TruncatedNormal(121,30)；0.7×Beta(4,2) std≈0.125；
    # Beta(8,72) std≈0.033；Normal(0, 0.5)。
    PRIOR_STD = {
        'base_incidence_per_100k': 30.0,
        'oilfield_camp_factor': 0.125,
        'latent_prevalence': 0.033,
        'sdoh_housing_coefficient': 0.5,
        'sdoh_access_coefficient': 0.5,
        'sdoh_bmi_coefficient': 0.5,
        'sdoh_income_coefficient': 0.5,
    }

    @staticmethod
    def log_prior(params_dict):
        """计算对数先验概率密度。

        参数：
            params_dict: dict, 参数字典

        返回：
            float: 对数先验密度（未归一化）
        """
        lp = 0.0

        # base_incidence_per_100k ~ TruncatedNormal(121, 30) 在 [50, 300]
        base = params_dict.get('base_incidence_per_100k', 121.0)
        if 50.0 < base < 300.0:
            lp += sps.norm.logpdf(base, loc=121.0, scale=30.0)
        else:
            return -float('inf')

        # oilfield_camp_factor ~ Beta(4, 2) 变换到 [0.3, 1.0]
        of = params_dict.get('oilfield_camp_factor', 0.85)
        of_scaled = (of - 0.3) / 0.7  # 缩放到 [0, 1]
        if 0 < of_scaled < 1:
            lp += sps.beta.logpdf(of_scaled, a=4, b=2) - math.log(0.7)
        else:
            return -float('inf')

        # latent_prevalence ~ Beta(8, 72)
        lp_val = params_dict.get('latent_prevalence', 0.10)
        if 0 < lp_val < 1:
            lp += sps.beta.logpdf(lp_val, a=8, b=72)
        else:
            return -float('inf')

        # SDOH 系数 ~ Normal(0, 0.5)：中心化在零效应，允许数据驱动学习
        # 文献：WHO SDOH Framework (Lönnroth et al., 2009)
        for sdoh_name in ['sdoh_housing_coefficient', 'sdoh_access_coefficient',
                          'sdoh_bmi_coefficient', 'sdoh_income_coefficient']:
            sdoh_val = params_dict.get(sdoh_name, 0.0)
            if -2.0 < sdoh_val < 2.0:
                lp += sps.norm.logpdf(sdoh_val, loc=0.0, scale=0.5)
            else:
                return -float('inf')

        return lp

    @staticmethod
    def sample_prior(random_state=None):
        """从先验分布中采样一组参数。

        参数：
            random_state: random.Random, np.random.RandomState, or None

        返回：
            dict: 参数名到采样值的映射
        """
        rng = random_state if random_state is not None else random.Random()

        # 检测是否为 numpy RandomState
        is_numpy_rng = hasattr(rng, 'random_sample')

        def _sample_normal(mean, std):
            """按可用库采样正态分布并裁剪到合理范围"""
            if is_numpy_rng:
                return float(rng.normal(mean, std))
            # random.Random 没有正态分布，使用纯 Python 回退
            return random.gauss(mean, std)

        if SCIPY_AVAILABLE and NUMPY_AVAILABLE:
            if is_numpy_rng:
                # numpy RandomState 使用 scipy 分布采样
                base = float(sps.truncnorm.rvs(
                    (50.0 - 121.0) / 30.0, (300.0 - 121.0) / 30.0,
                    loc=121.0, scale=30.0, random_state=rng))
                of_scaled = float(sps.beta.rvs(a=4, b=2, random_state=rng))
                of = 0.3 + 0.7 * of_scaled
                lp_val = float(sps.beta.rvs(a=8, b=72, random_state=rng))
            else:
                # random.Random 使用其内置方法
                base = _sample_normal(121.0, 30.0)
                of = 0.3 + 0.7 * rng.betavariate(4, 2)
                lp_val = rng.betavariate(8, 72)
        elif is_numpy_rng:
            # 无 scipy 时的 numpy 近似
            base = float(rng.normal(121.0, 30.0))
            of = 0.3 + 0.7 * float(rng.beta(4, 2))
            lp_val = float(rng.beta(8, 72))
        else:
            # 纯 Python 回退
            base = random.gauss(121.0, 30.0)
            of = 0.3 + 0.7 * random.betavariate(4, 2)
            lp_val = random.betavariate(8, 72)

        return {
            'base_incidence_per_100k': max(50.0, min(300.0, base)),
            'oilfield_camp_factor': max(0.3, min(1.0, of)),
            'latent_prevalence': max(0.02, min(0.30, lp_val)),
            'sdoh_housing_coefficient': max(-2.0, min(2.0, _sample_normal(0.0, 0.5))),
            'sdoh_access_coefficient': max(-2.0, min(2.0, _sample_normal(0.0, 0.5))),
            'sdoh_bmi_coefficient': max(-2.0, min(2.0, _sample_normal(0.0, 0.5))),
            'sdoh_income_coefficient': max(-2.0, min(2.0, _sample_normal(0.0, 0.5))),
        }


# ==================== 贝叶斯校准器 ====================

class BayesianCalibrator:
    """贝叶斯校准器：使用 MCMC 推断本土化参数的后验分布。

    替代 calibrator.py 中 get_infection_probability_baseline_offset 的硬编码公式。

    用法：
        calibrator = BayesianCalibrator(localizer)
        result = calibrator.calibrate(observed_data)
        # result.posterior_mean['base_incidence_per_100k']  # 后验均值
        # result.posterior_ci['base_incidence_per_100k']     # 95% 可信区间
    """

    def __init__(self, localizer):
        """
        参数：
            localizer: KaramayLocalizer 实例
        """
        self.localizer = localizer
        self.engine = ScoringEngine(localizer=localizer, use_localization=True)
        self._result = None
        self._chain_acceptance = {}  # 链索引到接受率的映射

    def calibrate(self, observed_data=None, n_iterations=4000, n_burnin=2000,
                  n_chains=4, random_seed=42):
        """执行贝叶斯校准。

        参数：
            observed_data: dict, 键为区县名，值为观测筛查阳性率(%)
                          若为 None，使用默认合成数据
            n_iterations: int, 每条链的迭代次数
            n_burnin: int, burn-in 迭代次数
            n_chains: int, 并行链数
            random_seed: int, 随机种子

        返回：
            CalibrationResult: 包含后验分布、诊断信息的结果对象
        """
        if observed_data is None:
            observed_data = self._default_observed_data()

        if not SCIPY_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("scipy/numpy 不可用，回退到点估计模式")
            return self._fallback_point_estimate()

        # 多链并行运行
        all_chains = []
        self._chain_acceptance = {}
        for chain_idx in range(n_chains):
            chain_seed = random_seed + chain_idx * 1000
            chain, accept_rate = self._run_single_chain(
                observed_data, n_iterations, n_burnin,
                np.random.RandomState(chain_seed)
            )
            all_chains.append(chain)
            self._chain_acceptance[chain_idx] = accept_rate

        # 构建结果
        result = CalibrationResult(
            param_names=CalibrationPriors.PARAM_NAMES,
            chains=all_chains,
            n_burnin=n_burnin,
            observed_data=observed_data,
        )
        # 汇总各链接受率
        if all_chains:
            result.acceptance_rate = sum(
                self._chain_acceptance.get(i, 0.0) for i in range(n_chains)
            ) / n_chains
        self._result = result
        return result

    def _run_single_chain(self, observed_data, n_iterations, n_burnin, rng):
        """运行单条 MCMC 链（自适应 Metropolis-Hastings）。

        实现 Haario (2001) 自适应方案：
        - 初始 proposal 协方差 = diag(先验 std²)（量纲归一化），
          而非与参数量纲失配的单位矩阵×0.01
        - burn-in 期间每步收集链样本，每 200 步且样本数≥100 时
          用最近 500 个样本的协方差更新 proposal_cov
        - burn-in 后冻结 proposal_cov（保持 Markov 性质）

        v2 修正（低接受率根因）：原实现每 100 步才收集 1 个样本点，
        burnin=1000 时凑不满 10 个点，协方差从未被更新过——提议分布
        始终是与后验几何失配的 0.01·I，导致接受率 ~9%（健康区间
        0.2-0.5，Roberts & Rosenthal 2001）。
        """
        n_params = len(CalibrationPriors.PARAM_NAMES)
        param_names = CalibrationPriors.PARAM_NAMES

        # 初始化：从先验采样
        current = CalibrationPriors.sample_prior(rng)
        current_log_post = self._log_posterior(current, observed_data)

        # 初始化 proposal 协方差：按先验 std 归一化（量纲匹配）
        prior_std = np.array(
            [CalibrationPriors.PRIOR_STD[name] for name in param_names])
        proposal_cov = np.diag(prior_std ** 2)
        proposal_scale = 2.38 / math.sqrt(n_params)  # 最优缩放 (Gelman 1997)

        # 存储链（burn-in 后的样本）
        chain = np.zeros((n_iterations - n_burnin, n_params))
        accepted = 0
        total = 0

        # 自适应期间收集样本用于更新协方差（每步收集，滑窗估计）
        adaptive_samples = []
        ADAPTIVE_UPDATE_EVERY = 200   # 更新间隔
        ADAPTIVE_MIN_SAMPLES = 100    # 触发更新的最小样本数
        ADAPTIVE_WINDOW = 500         # 协方差估计滑窗长度

        for iteration in range(n_iterations):
            # 提案：在当前点添加高斯噪声
            current_vec = np.array([current[name] for name in param_names])
            proposal_vec = rng.multivariate_normal(
                current_vec, proposal_cov * proposal_scale
            )

            # 构建提案参数字典
            proposal = {}
            for i, name in enumerate(param_names):
                lo, hi = CalibrationPriors.PARAM_BOUNDS[name]
                proposal[name] = max(lo, min(hi, proposal_vec[i]))

            # Metropolis 接受/拒绝
            proposal_log_post = self._log_posterior(proposal, observed_data)
            log_ratio = proposal_log_post - current_log_post

            if log_ratio > 0 or math.log(max(rng.random_sample(), 1e-300)) < log_ratio:
                current = proposal
                current_log_post = proposal_log_post
                accepted += 1

            total += 1

            # 存储 burn-in 后的样本
            if iteration >= n_burnin:
                chain_idx = iteration - n_burnin
                chain[chain_idx] = np.array([current[name] for name in param_names])

            # 自适应更新 proposal 协方差（仅 burn-in 期间，Haario 2001）
            if iteration < n_burnin:
                adaptive_samples.append(
                    np.array([current[name] for name in param_names]))
                if (iteration > 0
                        and iteration % ADAPTIVE_UPDATE_EVERY == 0
                        and len(adaptive_samples) >= ADAPTIVE_MIN_SAMPLES):
                    stacked = np.array(adaptive_samples[-ADAPTIVE_WINDOW:])
                    cov = np.cov(stacked, rowvar=False)
                    # 正则化：确保协方差矩阵非奇异且量级有下界
                    cov += np.diag(prior_std ** 2) * 1e-4
                    if np.all(np.isfinite(cov)):
                        proposal_cov = cov

        # 计算接受率
        accept_rate = accepted / max(total, 1)
        logger.debug("Chain acceptance rate: %.3f", accept_rate)

        return chain, accept_rate

    def _log_posterior(self, params, observed_data):
        """计算对数后验密度（先验 + 似然）。

        参数：
            params: dict, 参数名到值的映射
            observed_data: dict, 区县名到观测阳性率(%)

        返回：
            float: 对数后验密度
        """
        # 先验
        lp = CalibrationPriors.log_prior(params)
        if lp == -float('inf') or math.isnan(lp):
            return -float('inf')

        # 似然：各区县观测阳性率 ~ Normal(predicted_rate, sigma_obs)
        ll = self._log_likelihood(params, observed_data)
        if ll == -float('inf') or math.isnan(ll):
            return -float('inf')

        return lp + ll

    def _log_likelihood(self, params, observed_data):
        """计算对数似然。

        将 ScoringEngine 作为计算机模型：
        - 给定 θ，生成代表性接触者记录
        - 运行 ScoringEngine 计算预测阳性率
        - 似然：observed_rate ~ Normal(predicted_rate, sigma_obs)
        """
        # 配置 ScoringEngine 参数
        self._configure_engine(params)

        # 使用预生成的记录（避免每次似然评估重新生成）
        if not hasattr(self, '_cached_records') or self._cached_records is None:
            self._cached_records = self._generate_representative_records(params)

        predicted_rates = self._compute_predicted_rates_cached(params)

        if not SCIPY_AVAILABLE:
            # 无 scipy 时用简易 MSE
            ll = 0.0
            for district, obs_rate in observed_data.items():
                pred_rate = predicted_rates.get(district, predicted_rates.get('default', 5.0))
                sigma_obs = max(obs_rate * 0.1, 0.5)  # 10% 观测变异，下限 0.5%
                ll += -0.5 * ((obs_rate - pred_rate) / sigma_obs) ** 2
            return ll

        # 正态似然
        ll = 0.0
        for district, obs_rate in observed_data.items():
            pred_rate = predicted_rates.get(district, predicted_rates.get('default', 5.0))
            sigma_obs = max(obs_rate * 0.1, 0.5)
            ll += sps.norm.logpdf(obs_rate, loc=pred_rate, scale=sigma_obs)

        return ll

    def _configure_engine(self, params):
        """用参数配置 ScoringEngine。

        参数覆盖确保 ScoringEngine 使用校准参数而非 localizer 默认值。
        """
        overrides = {
            'base_incidence_per_100k': params.get('base_incidence_per_100k', 121.0),
            'oilfield_camp_factor': params.get('oilfield_camp_factor', 0.85),
            'LATENT_BASELINE': params.get('latent_prevalence', 0.10),
            'SETTING_FACTORS_BASE': dict(self.engine.SETTING_FACTORS_BASE),
        }
        overrides['SETTING_FACTORS_BASE']['oilfield_camp'] = params.get(
            'oilfield_camp_factor', 0.85)

        self.engine.apply_param_overrides(overrides)

    def _compute_predicted_rates(self, params):
        """使用当前参数配置，通过 ScoringEngine 计算各区县的预测阳性率。

        返回：
            dict: 区县名到预测阳性率(%)的映射
        """
        # 生成代表性接触者记录（每个区县）
        representative_records = self._generate_representative_records(params)

        return self._compute_rates_from_records(representative_records, params)

    def _compute_predicted_rates_cached(self, params):
        """使用预生成的记录计算预测阳性率（性能优化）。"""
        return self._compute_rates_from_records(self._cached_records, params)

    def _compute_rates_from_records(self, records_dict, params=None):
        """从记录字典计算各区县预测阳性率。

        参数：
            records_dict: dict, 区县名到记录列表的映射
            params: dict, 当前 MCMC 参数（用于发病率缩放和 SDOH 调整）

        返回：
            dict: 区县名到预测阳性率(%)的映射

        SDOH 调整（v3.0）：
            对每条记录读取 SDOH 字段（living_area_per_person、
            distance_to_tb_center_km、bmi、income_ratio），通过
            log-linear 乘数调整个体基线感染概率：
            
            sdoh_multiplier = exp(
                Σ coeff_i × (feature_i - μ_i) / σ_i
            )
            
            系数从 params 中获取，先验中心化在零（无效应），
            使后验更新能数据驱动地学习 SDOH 的影响。
            文献：WHO SDOH Framework (Lönnroth et al., 2009)
        """
        import math
        params = params or {}
        base_incidence = params.get('base_incidence_per_100k', 121.0)
        incidence_scale = base_incidence / 121.0

        # SDOH 系数（默认 0.0 = 无效应）
        sdoh_coeffs = {
            'housing': params.get('sdoh_housing_coefficient', 0.0),
            'access': params.get('sdoh_access_coefficient', 0.0),
            'bmi': params.get('sdoh_bmi_coefficient', 0.0),
            'income': params.get('sdoh_income_coefficient', 0.0),
        }

        predicted = {}
        for district, records in records_dict.items():
            if not records:
                predicted[district] = 0.0
                continue

            total_prob = 0.0
            for rec in records:
                result = self.engine.compute_risk_score(rec)
                prob = result.get('disease_probability', 0.0)

                # SDOH 乘数：log-linear 调整个体基线感染概率
                # 特征归一化（中心化 + 缩放），使系数可比较
                living_area = rec.get('living_area_per_person', 20.0)
                distance = rec.get('distance_to_tb_center_km', 10.0)
                bmi = rec.get('bmi', 22.0)
                income = rec.get('income_ratio', 1.0)

                sdoh_log_mult = (
                    sdoh_coeffs['housing'] * (living_area - 20.0) / 10.0 +
                    sdoh_coeffs['access'] * (distance - 10.0) / 10.0 +
                    sdoh_coeffs['bmi'] * (bmi - 22.0) / 3.0 +
                    sdoh_coeffs['income'] * (income - 1.0) / 0.5
                )
                sdoh_mult = math.exp(sdoh_log_mult)

                total_prob += prob * sdoh_mult

            predicted[district] = (
                total_prob / max(len(records), 1)
            ) * incidence_scale

        return predicted

    def _generate_representative_records(self, params):
        """生成代表性接触者记录（用于计算机模型评估）。

        每个区县生成一组涵盖不同风险特征的合成记录。
        实际部署时应替换为本地真实数据。

        参数：
            params: dict, 校准参数

        返回：
            dict: 区县名到记录列表的映射
        """
        base_incidence = params.get('base_incidence_per_100k', 121.0)
        diagnosis_delay = params.get('diagnosis_delay_days', 29.0)

        # 基础记录模板
        records = {
            'karamay': [],    # 克拉玛依区
            'dushanzi': [],   # 独山子区
            'baijiantan': [], # 白碱滩区
            'urho': [],       # 乌尔禾区
        }

        # 各区县基础发病率偏移（反映空间异质性）
        district_offsets = {
            'karamay': 1.0,      # 城区：基准
            'dushanzi': 0.85,    # 独山子：石化产业，相对较低
            'baijiantan': 1.10,  # 白碱滩：油田区域，略高
            'urho': 0.90,        # 乌尔禾：旅游区，流动人口
        }

        # 为每个区县生成记录
        n_records_per_district = 50
        rng = random.Random(42)

        for district in records:
            for _ in range(n_records_per_district):
                rec = self._make_synthetic_record(
                    rng, base_incidence, diagnosis_delay,
                    district_offsets.get(district, 1.0)
                )
                records[district].append(rec)

        return records

    @staticmethod
    def _make_synthetic_record(rng, base_incidence, diagnosis_delay, district_offset):
        """生成一条合成接触者记录。

        覆盖不同风险特征组合，确保计算机模型输出有区分度。
        """
        # 基本人口学特征
        age = rng.choice([3, 8, 16, 28, 42, 58, 72])
        settings = ['general', 'closed', 'oilfield_camp', 'crowded', 'outdoor']
        distances = ['very_close', 'close', 'medium', 'far', 'distant']

        return {
            'age': age,
            'has_symptoms': rng.choice([0, 1]),
            'has_tb': rng.choice([0, 0, 0, 1]),  # 75% 无既往史
            'bcg_vaccine': rng.choice([0, 1, 1]),  # 67% 已接种
            'exposure_setting': rng.choice(settings),
            'contact_distance': rng.choice(distances),
            'ventilation': rng.randint(1, 5),
            'cumulative_exposure': rng.choice([0, 8, 20, 50, 100]),
            'past_illness_type': rng.choice(['none', 'none', 'none', 'diabetes', 'other']),
            'idu_status': rng.choice([False, False, False, True]),
            'ethnicity': rng.choice(['han', 'han', 'han', 'uyghur', 'kazakh']),
            'pm10': rng.choice([80, 100, 120, 150, 200]),
            'humidity': rng.choice([10, 20, 30, 40, 50]),
            'origin_altitude': rng.choice([None, 500, 800, 1500, 2500, 3500]),
            'months_since_migration': rng.choice([0, 6, 12, 24, 60]),
            # SDOH 字段（v3.0 已实现）：参与贝叶斯校准的协变量建模
            # 这些字段在 _compute_rates_from_records 中通过 log-linear 乘数
            # 调整个体基线感染概率，系数通过 PARAM_NAMES 中的 SDOH 参数
            # 经 MCMC 后验更新学习。先验中心化在零（无效应），允许数据驱动。
            # 文献：WHO SDOH Framework (Lönnroth et al., 2009); Belmont Report 1979
            'living_area_per_person': round(rng.uniform(10, 30), 1),
            'distance_to_tb_center_km': round(rng.uniform(0, 50), 1),
            'bmi': round(rng.gauss(22, 3), 1),
            'income_ratio': round(rng.uniform(0.3, 1.5), 2),
        }

    @staticmethod
    def _default_observed_data():
        """默认观测数据（克拉玛依各区县筛查阳性率，%）。

        基于克拉玛依卫健委 2024 年报告估算。
        实际部署时应替换为真实数据。
        """
        return {
            'karamay': 5.2,     # 克拉玛依区：约 5.2%
            'dushanzi': 3.8,    # 独山子区：约 3.8%
            'baijiantan': 6.1,  # 白碱滩区：约 6.1%
            'urho': 4.5,        # 乌尔禾区：约 4.5%
        }

    def _fallback_point_estimate(self):
        """当 scipy/numpy 不可用时的回退点估计。"""
        return CalibrationResult(
            param_names=CalibrationPriors.PARAM_NAMES,
            chains=None,
            n_burnin=0,
            observed_data=self._default_observed_data(),
            is_fallback=True,
        )

    def get_infection_probability_baseline_offset(self):
        """计算感染概率基线偏移量（替代 calibrator.py 中的硬编码公式）。

        基于后验分布的 base_incidence_per_100k 均值，计算相对于
        默认发病率 62/10万 的对数偏移。

        返回：
            float: 基线偏移量 (%)
        """
        if self._result is None or self._result.is_fallback:
            # 回退到原始公式
            local_rate = 121.0
            default_rate = DEFAULT_NATIONAL_INCIDENCE_PER_100K
            if default_rate <= 0:
                return 0.0
            return max(-10.0, min(15.0, (local_rate / default_rate - 1.0) * 5.0))

        # 使用后验均值
        posterior_mean = self._result.posterior_mean.get(
            'base_incidence_per_100k', 121.0)
        default_rate = DEFAULT_NATIONAL_INCIDENCE_PER_100K
        if default_rate <= 0:
            return 0.0

        # 对数偏移：使用 log-ratio 而非线性 ratio，更稳健
        log_offset = math.log(max(posterior_mean, 1.0) / default_rate) * 5.0
        return max(-10.0, min(15.0, log_offset))


# ==================== 校准结果 ====================

class CalibrationResult:
    """贝叶斯校准结果，包含后验分布和诊断信息。"""

    def __init__(self, param_names, chains, n_burnin, observed_data,
                 is_fallback=False):
        """
        参数：
            param_names: list[str], 参数名列表
            chains: list[np.ndarray] or None, 各链的后验样本
            n_burnin: int, burn-in 迭代数
            observed_data: dict, 观测数据
            is_fallback: bool, 是否为回退点估计
        """
        self.param_names = param_names
        self.chains = chains
        self.n_burnin = n_burnin
        self.observed_data = observed_data
        self.is_fallback = is_fallback

        if is_fallback or chains is None:
            self._compute_fallback()
        else:
            self._compute_posterior()

    def _compute_fallback(self):
        """回退模式：使用先验均值作为点估计。"""
        self.posterior_mean = {
            'base_incidence_per_100k': 121.0,
            'oilfield_camp_factor': 0.85,
            'latent_prevalence': 0.10,
            'diagnosis_delay_days': 29.0,
            'sdoh_housing_coefficient': 0.0,
            'sdoh_access_coefficient': 0.0,
            'sdoh_bmi_coefficient': 0.0,
            'sdoh_income_coefficient': 0.0,
        }
        self.posterior_median = dict(self.posterior_mean)
        self.posterior_std = {k: 0.0 for k in self.posterior_mean}
        self.posterior_ci = {k: (v, v) for k, v in self.posterior_mean.items()}
        self.r_hat = {k: 1.0 for k in self.posterior_mean}
        self.ess = {k: 0 for k in self.posterior_mean}
        self.acceptance_rate = 0.0
        self.converged = False

    def _compute_posterior(self):
        """从 MCMC 链计算后验统计量。"""
        if not NUMPY_AVAILABLE or not SCIPY_AVAILABLE:
            self._compute_fallback()
            return

        # 防御：burn-in 等于迭代次数或链为空时无有效样本
        if (not self.chains or
                self.chains[0].size == 0 or
                len(self.chains[0]) == 0):
            import logging
            logging.getLogger("tb_risk.karamay.bayesian_calibrator").warning(
                "MCMC 链为空（n_burnin 可能等于 n_iterations），使用后验均值回退")
            self._compute_fallback()
            return

        # 合并所有链（burn-in 后的样本）
        all_samples = np.concatenate(self.chains, axis=0)

        self.posterior_mean = {}
        self.posterior_median = {}
        self.posterior_std = {}
        self.posterior_ci = {}
        self.r_hat = {}
        self.ess = {}

        for i, name in enumerate(self.param_names):
            samples = all_samples[:, i]
            self.posterior_mean[name] = float(np.mean(samples))
            self.posterior_median[name] = float(np.median(samples))
            self.posterior_std[name] = float(np.std(samples))
            self.posterior_ci[name] = (
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            )
            self.r_hat[name] = self._compute_r_hat_single(name, i)
            self.ess[name] = self._compute_ess_single(i)

        # 收敛判断
        self.converged = all(
            r < 1.1 for r in self.r_hat.values()
        ) and all(
            e > 100 for e in self.ess.values()
        )

        # 接受率
        self.acceptance_rate = 0.0  # 在链级别已记录

    def _compute_r_hat_single(self, name, param_idx):
        """计算单个参数的 rank-normalized R-hat (Vehtari 2021)。

        使用 rank-normalized 方法，比传统 R-hat 对厚尾分布更稳健。
        """
        if self.chains is None or len(self.chains) < 2:
            return 1.0

        try:
            chain_vals = [c[:, param_idx] for c in self.chains]
            # 合并所有链的值
            all_vals = np.concatenate(chain_vals)
            n_total = len(all_vals)

            # Rank normalization (Vehtari 2021)
            ranks = np.argsort(np.argsort(all_vals)) + 1.0  # 1-based
            blom_offset = 3.0 / 8.0
            blom_denom = n_total + 0.25
            z_all = sps.norm.ppf((ranks - blom_offset) / blom_denom)

            # 分割回各链
            z_chains = []
            start = 0
            for c in chain_vals:
                n = len(c)
                z_chains.append(z_all[start:start + n])
                start += n

            # 计算 R-hat
            chain_means = np.array([np.mean(z) for z in z_chains])
            chain_vars = np.array([np.var(z, ddof=1) for z in z_chains])

            n_samples = len(z_chains[0])
            m = len(z_chains)
            overall_mean = np.mean(chain_means)

            B = n_samples / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
            W = np.mean(chain_vars)

            var_plus = ((n_samples - 1) / n_samples) * W + B / n_samples
            r_hat = np.sqrt(var_plus / max(W, 1e-10))

            return float(min(r_hat, 10.0))
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as e:
            import logging
            logging.getLogger("tb_risk.karamay.bayesian_calibrator").warning(
                "R-hat 计算失败: %s", e, exc_info=True)
            return 1.0

    def _compute_ess_single(self, param_idx):
        """计算单个参数的有效样本量（ESS）。

        使用自相关方法估计有效样本量。
        """
        if self.chains is None:
            return 0

        try:
            all_samples = np.concatenate([c[:, param_idx] for c in self.chains])
            n = len(all_samples)
            if n < 2:
                return n

            # 自相关估计
            mean = np.mean(all_samples)
            var = np.var(all_samples)
            if var < 1e-15:
                return n

            # 计算自相关，截断到 N/2 或首次负相关
            max_lag = min(n // 2, 100)
            rho = np.zeros(max_lag)
            for lag in range(1, max_lag + 1):
                rho[lag - 1] = np.mean(
                    (all_samples[:n - lag] - mean) * (all_samples[lag:] - mean)
                ) / var
                if rho[lag - 1] < 0:
                    break

            # ESS = N / (1 + 2 * sum(rho))
            tau = 1.0 + 2.0 * np.sum(rho[rho > 0])
            ess = n / max(tau, 1.0)
            return int(ess)
        except (ValueError, ZeroDivisionError) as e:
            import logging
            logging.getLogger("tb_risk.karamay.bayesian_calibrator").warning(
                "ESS 计算失败: %s", e, exc_info=True)
            return 0

    def summary(self):
        """生成校准结果摘要。"""
        lines = ["=== 贝叶斯校准结果 ==="]
        lines.append(f"收敛: {'是' if self.converged else '否'}")
        lines.append(f"接受率: {self.acceptance_rate:.3f}")
        lines.append("")

        lines.append("参数后验分布:")
        lines.append(f"{'参数':<28} {'均值':>8} {'中位数':>8} {'95% CI':>20} {'R-hat':>6} {'ESS':>6}")
        lines.append("-" * 80)

        for name in self.param_names:
            ci = self.posterior_ci.get(name, (0, 0))
            lines.append(
                f"{name:<28} {self.posterior_mean.get(name, 0):>8.1f} "
                f"{self.posterior_median.get(name, 0):>8.1f} "
                f"[{ci[0]:>7.1f}, {ci[1]:>7.1f}] "
                f"{self.r_hat.get(name, 0):>6.3f} "
                f"{self.ess.get(name, 0):>6d}"
            )

        return "\n".join(lines)

    def get_posterior_sample(self, n=1):
        """从后验分布中抽样 n 组参数。

        返回：
            list[dict]: 参数样本列表
        """
        if self.chains is None or not NUMPY_AVAILABLE:
            return [self.posterior_mean] * n

        all_samples = np.concatenate(self.chains, axis=0)
        n_total = len(all_samples)
        indices = np.random.choice(n_total, size=min(n, n_total), replace=False)

        samples = []
        for idx in indices:
            sample = {}
            for i, name in enumerate(self.param_names):
                sample[name] = float(all_samples[idx, i])
            samples.append(sample)

        return samples