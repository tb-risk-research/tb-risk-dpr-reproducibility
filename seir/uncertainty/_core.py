"""SEIR 参数不确定性量化 — 核心模块

参数先验、初始化、对数先验密度，以及 v3/v4 固定模型参数常量。
拆分自原 seir/uncertainty.py（业务逻辑不变）。
"""

import numpy as np

try:
    import scipy.stats as sps
    SCIPY_STATS_AVAILABLE = True
except ImportError:
    SCIPY_STATS_AVAILABLE = False


class _UncertaintyCore:
    """核心: 参数先验、初始化、对数先验密度 + v3/v4 固定参数常量"""

    SEIR_PARAM_PRIORS = {
        'beta': {'dist': 'beta', 'a': 2.0, 'b': 8.0,
                 'name': '传播率', 'unit': 'day⁻¹'},
        'rho_fast': {'dist': 'gamma', 'shape': 2.0, 'scale': 0.05 / 365.0,
                     'name': '快进展率', 'unit': 'day⁻¹'},
        'rho_react': {'dist': 'gamma', 'shape': 2.0, 'scale': 0.002 / 365.0,
                      'name': '慢再激活率', 'unit': 'day⁻¹'},
        'sigma_clear': {'dist': 'gamma', 'shape': 2.0, 'scale': 1.0 / 365.0,
                        'name': '自清除率', 'unit': 'day⁻¹'},
        'gamma': {'dist': 'gamma', 'shape': 2.0, 'scale': 0.025,
                  'name': '恢复率', 'unit': 'day⁻¹'},
        'eta_sub': {'dist': 'beta', 'a': 2.0, 'b': 5.0,
                    'name': '亚临床相对传染性', 'unit': ''},
        # v3.0 新增参数
        'omega_reg_m': {'dist': 'gamma', 'shape': 2.0, 'scale': 0.5 / 365.0,
                        'name': 'M→L_slow 回退率', 'unit': 'day⁻¹'},
        'omega_reg_sub': {'dist': 'gamma', 'shape': 2.0, 'scale': 0.5 / 365.0,
                          'name': 'I_sub→M 回退率', 'unit': 'day⁻¹'},
        'beta_exo': {'dist': 'beta', 'a': 2.0, 'b': 4.0,
                     'name': '外源再感染易感性', 'unit': ''},
        'eta_sn': {'dist': 'beta', 'a': 2.0, 'b': 5.0,
                   'name': '涂阴相对传染性', 'unit': ''},
        # v4.0 年龄×HIV×耐药分层新参数
        'rr_hiv': {'dist': 'gamma', 'shape': 2.0, 'scale': 10.0,
                   'name': 'HIV+再激活相对风险', 'unit': ''},
        'art_reduction': {'dist': 'beta', 'a': 13.0, 'b': 7.0,
                          'name': 'ART降低TB风险比例', 'unit': ''},
        'hiv_infection_rate': {'dist': 'gamma', 'shape': 2.0, 'scale': 5e-4,
                               'name': '外生HIV感染率', 'unit': 'day⁻¹'},
        'art_initiation_rate': {'dist': 'gamma', 'shape': 2.0, 'scale': 1.4e-4,
                                'name': 'ART启动率', 'unit': 'day⁻¹'},
        'dr_fitness_cost': {'dist': 'beta', 'a': 6.0, 'b': 30.0,
                            'name': 'DR株适合度代价', 'unit': ''},
        'p_acq': {'dist': 'beta', 'a': 3.0, 'b': 90.0,
                  'name': '获得性耐药概率', 'unit': ''},
        'social_factor': {'dist': 'gamma', 'shape': 3.0, 'scale': 0.5,
                          'name': '社会接触因子', 'unit': ''},
        'household_factor': {'dist': 'gamma', 'shape': 4.0, 'scale': 0.5,
                             'name': '家庭接触因子', 'unit': ''},
        'k': {'dist': 'gamma', 'shape': 2.0, 'scale': 0.1,
              'name': '过度离散参数', 'unit': ''},
    }

    def __init__(self, stochastic_seir_model=None, population=10000,
                 random_seed=42):
        self.stochastic_seir = stochastic_seir_model
        self.population = population
        self.random_seed = random_seed
        # 使用独立 RandomState，避免多实例/多链共享全局 np.random 状态，
        # 确保可复现性并满足 MCMC 独立链假设。
        self.rng = np.random.RandomState(random_seed)

        self.posterior_samples = None
        self.posterior_summary = None
        self.r0_posterior = None
        self.inference_completed = False
        self.mcmc_diagnostics = {}

    def _log_prior(self, params):
        """计算参数先验对数密度

        v3.0 7-房室模型支撑域（per day）：
          β:          0.001 < β < 0.95
          ρ_fast:     0.00001 < ρ_fast < 0.01       (~0.004~3.65/年)
          ρ_react:    1e-7 < ρ_react < 0.0001        (~0.00004~0.036/年)
          σ_clear:    0.0001 < σ_clear < 0.05        (~0.036~18/年)
          γ:          0.005 < γ < 0.2                (5~200天传染期)
          η_sub:      0.01 < η_sub < 5.0

        v4.0 年龄×HIV×耐药分层新增支撑域 (Pawlowski 2012, Karmakar 2022)：
          RR_HIV:              5.0 < RR_HIV < 50.0          (均值≈20)
          ART_reduction:       0.20 < ART_red < 0.90        (均值≈0.65)
          HIV_infection_rate:  1e-6 < rate < 0.01           (均值≈0.001)
          ART_initiation_rate: 1e-5 < rate < 1e-2           (均值≈2.7e-4)
          DR_fitness_cost:     0.02 < cost < 0.40           (均值≈0.15)
          p_acq:               0.005 < p_acq < 0.10         (均值≈0.032)
        """
        # 上界约束检查
        beta_val = params.get('beta')
        rho_fast_val = params.get('rho_fast')
        rho_react_val = params.get('rho_react')
        sigma_clear_val = params.get('sigma_clear')
        gamma_val = params.get('gamma')
        eta_sub_val = params.get('eta_sub')

        if beta_val is not None and not (0.001 < beta_val < 0.95):
            return -np.inf
        if rho_fast_val is not None and not (0.00001 < rho_fast_val < 0.01):
            return -np.inf
        if rho_react_val is not None and not (1e-7 < rho_react_val < 0.0001):
            return -np.inf
        if sigma_clear_val is not None and not (0.0001 < sigma_clear_val < 0.05):
            return -np.inf
        if gamma_val is not None and not (0.005 < gamma_val < 0.2):
            return -np.inf
        if eta_sub_val is not None and not (0.01 < eta_sub_val < 5.0):
            return -np.inf

        # v3.0 新增参数支撑域
        omega_reg_m_val = params.get('omega_reg_m')
        omega_reg_sub_val = params.get('omega_reg_sub')
        beta_exo_val = params.get('beta_exo')
        eta_sn_val = params.get('eta_sn')

        if omega_reg_m_val is not None and not (0.0001 < omega_reg_m_val < 0.05):
            return -np.inf
        if omega_reg_sub_val is not None and not (0.0001 < omega_reg_sub_val < 0.05):
            return -np.inf
        if beta_exo_val is not None and not (0.01 < beta_exo_val < 0.95):
            return -np.inf
        if eta_sn_val is not None and not (0.01 < eta_sn_val < 1.0):
            return -np.inf

        # v4.0 新增参数支撑域 (Pawlowski 2012, Karmakar 2022)
        rr_hiv_val = params.get('rr_hiv')
        art_reduction_val = params.get('art_reduction')
        hiv_infection_rate_val = params.get('hiv_infection_rate')
        art_initiation_rate_val = params.get('art_initiation_rate')
        dr_fitness_cost_val = params.get('dr_fitness_cost')
        p_acq_val = params.get('p_acq')

        if rr_hiv_val is not None and not (5.0 < rr_hiv_val < 50.0):
            return -np.inf
        if art_reduction_val is not None and not (0.20 < art_reduction_val < 0.90):
            return -np.inf
        if hiv_infection_rate_val is not None and \
                not (1e-6 < hiv_infection_rate_val < 0.01):
            return -np.inf
        if art_initiation_rate_val is not None and \
                not (1e-5 < art_initiation_rate_val < 1e-2):
            return -np.inf
        if dr_fitness_cost_val is not None and \
                not (0.02 < dr_fitness_cost_val < 0.40):
            return -np.inf
        if p_acq_val is not None and not (0.005 < p_acq_val < 0.10):
            return -np.inf

        lp = 0.0
        for key, prior in self.SEIR_PARAM_PRIORS.items():
            val = params.get(key, None)
            if val is None:
                continue
            if prior['dist'] == 'beta':
                if 0 < val < 1:
                    if SCIPY_STATS_AVAILABLE:
                        lp += sps.beta.logpdf(val, prior['a'], prior['b'])
                    else:
                        lp += (prior['a'] - 1) * np.log(val + 1e-10) + \
                              (prior['b'] - 1) * np.log(1 - val + 1e-10)
                elif key == 'eta_sub':
                    # eta_sub 缩放到 [0, 1] 用于 Beta 先验
                    x = val / 5.0
                    if 0 < x < 1:
                        if SCIPY_STATS_AVAILABLE:
                            lp += sps.beta.logpdf(x, prior['a'], prior['b'])
                        else:
                            lp += (prior['a'] - 1) * np.log(x + 1e-10) + \
                                  (prior['b'] - 1) * np.log(1 - x + 1e-10)
                    else:
                        return -np.inf
                else:
                    return -np.inf
            elif prior['dist'] == 'gamma':
                if val > 0:
                    if SCIPY_STATS_AVAILABLE:
                        lp += sps.gamma.logpdf(
                            val, prior['shape'], scale=prior['scale'])
                    else:
                        lp += (prior['shape'] - 1) * np.log(val + 1e-10) - \
                              val / prior['scale']
                else:
                    return -np.inf
        return lp

    # v3.0 9-房室固定参数（与 _stochastic_v3.py / bayesian.py 一致）
    _V3_FIXED = {
        'rho_conv': 1.0 / 365.0, 'p_clin': 0.332, 'rho_prog': 0.05,
        'p_m': 0.5, 'rho_min': 0.5 / 365.0, 'p_sp': 0.55,
        'r_sp': 0.231 / 365.0, 'mu_sp': 0.389 / 365.0,
        'r_sn': 0.130 / 365.0, 'mu_sn': 0.025 / 365.0,
        'p_sn2sp': 0.1 / 365.0, 'beta_reinf': 0.4,
    }
    # v3.0 默认相对传染性
    _V3_DEFAULT_ETA_SUB = 1.0
    _V3_DEFAULT_ETA_SN = 0.35

    # v4.0 固定参数（与 _stochastic_v4.py / autodiff.py 一致）
    _V4_FIXED = {
        'rho_conv': 1.0 / 365.0, 'rho_prog': 0.05,
        'p_clin': 0.332, 'p_m': 0.5, 'rho_min': 0.5 / 365.0,
        'p_sp': 0.55,
        'r_sp': 0.231 / 365.0, 'mu_sp': 0.389 / 365.0,
        'r_sn': 0.130 / 365.0, 'mu_sn': 0.025 / 365.0,
        'p_sn2sp': 0.05 / 365.0, 'beta_reinf': 0.4,
        'eta_sub': 1.0,
    }
