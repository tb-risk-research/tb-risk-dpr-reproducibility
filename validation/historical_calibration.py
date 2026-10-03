"""
历史数据验证模块（改进10）

验证 v3.0 9-房室 SEIR 模型是否能复现文献中的结核病自然史关键指标：

1. 涂阳/涂阴 TB 未治疗持续时间（Ragonnet et al. 2021, Clin Infect Dis）
   - 涂阳: ~1.57 年
   - 涂阴: ~5.35 年
2. 10 年自清除率 ~92%（Horton et al. 2023, PNAS）
3. 感染后进展为传染性 TB 的比例 ~7.9%（Andrews et al. 2012）

文献：
  Ragonnet R et al. (2021) Clin Infect Dis 73(1):e88-e96
  Horton KC et al. (2023) PNAS 120(47):e2221186120
  Andrews JR et al. (2012) Nat Rev Dis Primers 8(1):10710
  Emery JC et al. (2023) eLife 12:e82469
"""

import logging

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

LOGGER = logging.getLogger("tb_risk.validation.historical_calibration")

try:
    from scipy.integrate import odeint
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# 文献参考值
LITERATURE = {
    'smear_positive_duration_years': 1.57,   # Ragonnet 2021
    'smear_negative_duration_years': 5.35,   # Ragonnet 2021
    'self_clearance_10yr_pct': 92.0,         # Horton 2023
    'progression_to_infectious_pct': 7.9,    # Andrews 2012
}


# v3.0 9-房室默认参数（per day，与 _stochastic_v3.py 一致）
# 要点E: 调整 p_sn2sp 和 sigma_clear 使默认值更接近文献值
#   - p_sn2sp: 0.1/365 → 0.05/365，涂阴持续时间 3.92→4.88 年 (文献 5.35)
#   - sigma_clear: 2.0/365 → 1.0/365，10 年自清除率避免接近 100% (文献 92%)
_V3_PARAMS = {
    'rho_fast': 0.1 / 365.0,
    'rho_conv': 1.0 / 365.0,
    'rho_react': 0.004 / 365.0,
    'sigma_clear': 1.0 / 365.0,   # 要点E: 2.0→1.0
    'rho_prog': 0.05,
    'gamma': 0.05,
    'p_clin': 0.332,
    'p_m': 0.5,
    'rho_min': 0.5 / 365.0,
    'p_sp': 0.55,
    'r_sp': 0.231 / 365.0,
    'mu_sp': 0.389 / 365.0,
    'r_sn': 0.130 / 365.0,
    'mu_sn': 0.025 / 365.0,
    'p_sn2sp': 0.05 / 365.0,      # 要点E: 0.1→0.05
    'beta_reinf': 0.4,
    'beta_exo': 0.33,
    'eta_sub': 1.0,
    'eta_sn': 0.35,
}


class HistoricalCalibration:
    """历史数据校准验证（改进10）

    验证 v3.0 9-房室 SEIR 模型参数是否能复现文献中报告的 TB 自然史指标。
    """

    def __init__(self, params=None):
        if not NUMPY_AVAILABLE:
            raise ImportError("HistoricalCalibration 需要 numpy 支持")
        self.params = dict(_V3_PARAMS)
        if params:
            self.params.update(params)

    def compute_tb_durations(self):
        """计算涂阳/涂阴 TB 平均持续时间（年）

        涂阳: 1/(r_sp + μ_sp) — 恢复或死亡的总流出率
        涂阴: 1/(r_sn + μ_sn + p_sn2sp) — 恢复、死亡或转阳的总流出率

        返回:
            dict: {'smear_positive_years', 'smear_negative_years', 'literature_sp', 'literature_sn'}
        """
        p = self.params
        sp_outflow = p['r_sp'] + p['mu_sp']
        sn_outflow = p['r_sn'] + p['mu_sn'] + p['p_sn2sp']

        sp_duration = 1.0 / max(sp_outflow, 1e-10) / 365.0  # 转换为年
        sn_duration = 1.0 / max(sn_outflow, 1e-10) / 365.0

        return {
            'smear_positive_years': sp_duration,
            'smear_negative_years': sn_duration,
            'literature_sp': LITERATURE['smear_positive_duration_years'],
            'literature_sn': LITERATURE['smear_negative_duration_years'],
            'sp_ratio': sp_duration / LITERATURE['smear_positive_duration_years'],
            'sn_ratio': sn_duration / LITERATURE['smear_negative_duration_years'],
        }

    def _seir_ode_v3(self, y, t, beta, rho_fast, rho_react, sigma_clear, gamma, N):
        """v3 9-房室 ODE 右侧（无治疗干预）"""
        p = self.params
        S, Lf, Ls, M, Isub, Isp, Isn, R, C = y
        lam = beta * (p['eta_sub'] * Isub + Isp + p['eta_sn'] * Isn) / max(N, 1e-10)

        infection = lam * S
        reinfection_C = p['beta_reinf'] * lam * C
        reinfection_R = p['beta_reinf'] * lam * R
        exo_reinfection = p['beta_exo'] * lam * Ls

        fast_to_min = rho_fast * p['p_m'] * Lf
        fast_to_sub = rho_fast * (1 - p['p_m']) * (1 - p['p_clin']) * Lf
        fast_to_sp = rho_fast * (1 - p['p_m']) * p['p_clin'] * Lf
        fast_to_slow = p['rho_conv'] * Lf
        clear_fast = sigma_clear * Lf

        slow_to_min = rho_react * p['p_m'] * Ls
        slow_to_sub = rho_react * (1 - p['p_m']) * (1 - p['p_clin']) * Ls
        slow_to_sp = rho_react * (1 - p['p_m']) * p['p_clin'] * Ls
        clear_slow = sigma_clear * Ls

        min_to_sub = p['rho_min'] * M
        reg_m = p.get('omega_reg_m', 1.0 / 365.0) * M
        reg_sub = p.get('omega_reg_sub', 1.0 / 365.0) * Isub

        sub_to_sp = p['rho_prog'] * p['p_sp'] * Isub
        sub_to_sn = p['rho_prog'] * (1 - p['p_sp']) * Isub
        sn_to_sp = p['p_sn2sp'] * Isn
        sp_out = (p['r_sp'] + p['mu_sp']) * Isp
        sn_out = (p['r_sn'] + p['mu_sn']) * Isn

        dS = -infection
        dLf = (infection + reinfection_C + reinfection_R + exo_reinfection
               - fast_to_min - fast_to_sub - fast_to_sp - fast_to_slow - clear_fast)
        dLs = (fast_to_slow + reg_m
               - slow_to_min - slow_to_sub - slow_to_sp - clear_slow - exo_reinfection)
        dM = (fast_to_min + slow_to_min + reg_sub - min_to_sub - reg_m)
        dIsub = (fast_to_sub + slow_to_sub + min_to_sub
                 - sub_to_sp - sub_to_sn - reg_sub)
        dIsp = (fast_to_sp + slow_to_sp + sub_to_sp + sn_to_sp - sp_out)
        dIsn = (sub_to_sn - sn_to_sp - sn_out)
        dR = (sp_out + sn_out - reinfection_R)
        dC = (clear_fast + clear_slow - reinfection_C)
        return [dS, dLf, dLs, dM, dIsub, dIsp, dIsn, dR, dC]

    def compute_self_clearance_rate(self, years=10, population=10000,
                                     initial_infected=100, beta=0.0):
        """计算 N 年自清除率（Horton 2023: ~92%）

        在无传播条件（β=0）下跟踪初始感染队列，计算 L_fast+L_slow 中
        流入 C（自清除）的比例。

        要点E: rho_fast/rho_react/sigma_clear/gamma 从 ``self.params`` 读取，
        支持通过 ``HistoricalCalibration(params={...})`` 覆盖，便于校准后验证。

        参数:
            years: 跟踪年数
            population: 总人口
            initial_infected: 初始感染人数
            beta: 传播率（0 = 无传播，纯队列研究）

        返回:
            dict: {'clearance_pct', 'literature_pct', 'ratio'}
        """
        p = self.params
        N = population
        I0 = initial_infected
        # 初始状态: 大部分 S，少量 Lf/Ls
        Lf0 = int(I0 * 0.3)
        Ls0 = I0 - Lf0
        S0 = N - I0
        y0 = [float(S0), float(Lf0), float(Ls0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 要点E: 从 params 读取（支持覆盖）
        rho_fast = p.get('rho_fast', 0.1 / 365.0)
        rho_react = p.get('rho_react', 0.004 / 365.0)
        sigma_clear = p.get('sigma_clear', 1.0 / 365.0)
        gamma = p.get('gamma', 0.05)

        n_days = int(years * 365)
        t = np.linspace(0, years * 365, n_days)

        if SCIPY_AVAILABLE:
            sol = odeint(self._seir_ode_v3, y0, t,
                         args=(beta, rho_fast, rho_react, sigma_clear, gamma, N))
        else:
            # Euler 回退
            sol = np.zeros((n_days, 9))
            sol[0] = y0
            dt = t[1] - t[0]
            for i in range(1, n_days):
                dy = self._seir_ode_v3(sol[i - 1], t[i - 1],
                                        beta, rho_fast, rho_react,
                                        sigma_clear, gamma, N)
                sol[i] = sol[i - 1] + np.array(dy) * dt

        # 初始 Lf+Ls 总量
        initial_latent = Lf0 + Ls0
        # 最终 C 房室总量（自清除累积）
        final_C = sol[-1, 8]
        # 自清除比例
        clearance_pct = (final_C / max(initial_latent, 1)) * 100.0

        return {
            'clearance_pct': clearance_pct,
            'literature_pct': LITERATURE['self_clearance_10yr_pct'],
            'ratio': clearance_pct / LITERATURE['self_clearance_10yr_pct'],
            'years': years,
            'final_C': float(final_C),
            'initial_latent': initial_latent,
        }

    def compute_progression_rate(self, years=10, population=10000,
                                  initial_infected=100, beta=0.0):
        """计算感染后进展为传染性 TB 的比例（Andrews 2012: ~7.9%）

        在无传播条件（β=0）下跟踪初始感染队列，计算最终进入
        I_sub+I_sp+I_sn+R（活动性 TB 累积）的比例。

        要点E: rho_fast/rho_react/sigma_clear/gamma 从 ``self.params`` 读取。

        返回:
            dict: {'progression_pct', 'literature_pct', 'ratio'}
        """
        p = self.params
        N = population
        I0 = initial_infected
        Lf0 = int(I0 * 0.3)
        Ls0 = I0 - Lf0
        S0 = N - I0
        y0 = [float(S0), float(Lf0), float(Ls0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 要点E: 从 params 读取
        rho_fast = p.get('rho_fast', 0.1 / 365.0)
        rho_react = p.get('rho_react', 0.004 / 365.0)
        sigma_clear = p.get('sigma_clear', 1.0 / 365.0)
        gamma = p.get('gamma', 0.05)

        n_days = int(years * 365)
        t = np.linspace(0, years * 365, n_days)

        if SCIPY_AVAILABLE:
            sol = odeint(self._seir_ode_v3, y0, t,
                         args=(beta, rho_fast, rho_react, sigma_clear, gamma, N))
        else:
            sol = np.zeros((n_days, 9))
            sol[0] = y0
            dt = t[1] - t[0]
            for i in range(1, n_days):
                dy = self._seir_ode_v3(sol[i - 1], t[i - 1],
                                        beta, rho_fast, rho_react,
                                        sigma_clear, gamma, N)
                sol[i] = sol[i - 1] + np.array(dy) * dt

        # 累积进展为活动性 TB = 最终 (I_sub + I_sp + I_sn + R)
        # R 包含恢复的 TB 个体（死亡的不在 R 中，但当前模型将死亡合并到 R）
        final_infectious = (sol[-1, 4] + sol[-1, 5] + sol[-1, 6] + sol[-1, 7])
        initial_latent = Lf0 + Ls0
        progression_pct = (final_infectious / max(initial_latent, 1)) * 100.0

        return {
            'progression_pct': progression_pct,
            'literature_pct': LITERATURE['progression_to_infectious_pct'],
            'ratio': progression_pct / LITERATURE['progression_to_infectious_pct'],
            'years': years,
            'final_infectious': float(final_infectious),
            'initial_latent': initial_latent,
        }

    def run_all_validations(self):
        """运行全部历史数据验证

        返回:
            dict: 所有验证结果汇总
        """
        durations = self.compute_tb_durations()
        clearance = self.compute_self_clearance_rate()
        progression = self.compute_progression_rate()

        return {
            'tb_durations': durations,
            'self_clearance': clearance,
            'progression': progression,
            'literature': LITERATURE,
            'all_passed': (
                0.5 < durations['sp_ratio'] < 2.0
                and 0.5 < durations['sn_ratio'] < 2.0
                and 0.5 < clearance['ratio'] < 2.0
                and 0.3 < progression['ratio'] < 3.0
            ),
        }

    def calibrated_validation(self, posterior_samples, years=10,
                               n_subsample=100, seed=42):
        """要点E: 使用 MCMC 后验样本进行校准后验证

        接受 ``run_mcmc_v3`` / ``run_mcmc_v4`` 产生的后验样本，使用后验均值
        （而非默认先验均值）运行验证，并报告后验预测区间与文献值的吻合度。

        参数:
            posterior_samples: dict, 参数名 → 1D 数组（后验样本）
                支持的参数名（与 _V3_PARAMS 键一致）:
                  beta, rho_fast, rho_react, sigma_clear, rho_prog, gamma,
                  p_clin, p_m, rho_min, p_sp, r_sp, mu_sp, r_sn, mu_sn,
                  p_sn2sp, beta_reinf, beta_exo, eta_sub, eta_sn, rho_conv,
                  omega_reg_m, omega_reg_sub
            years: 自清除/进展跟踪年数
            n_subsample: 从后验中采样的参数组数（计算后验预测区间）
            seed: 随机种子

        返回:
            dict: {
                'posterior_mean_validation': run_all_validations() 格式,
                'posterior_predictive': {
                    'tb_durations_sp': [lo, median, hi],
                    'tb_durations_sn': [lo, median, hi],
                    'clearance_pct':   [lo, median, hi],
                    'progression_pct': [lo, median, hi],
                },
                'n_samples': int,
                'n_effective': int,
            }
        """
        rng = np.random.RandomState(seed)

        # 收集后验中可用的参数名（与 _V3_PARAMS 键交集）
        supported = set(_V3_PARAMS.keys()) | {'omega_reg_m', 'omega_reg_sub'}
        available = {k: np.asarray(v) for k, v in posterior_samples.items()
                     if k in supported and len(np.asarray(v)) > 0}
        if not available:
            raise ValueError("posterior_samples 中无可用参数（需包含 _V3_PARAMS 键）")

        n_total = min(len(next(iter(available.values()))), n_subsample)
        if n_total <= 0:
            raise ValueError("后验样本数为 0")

        # 1. 后验均值验证
        mean_params = {k: float(np.mean(v)) for k, v in available.items()}
        hc_mean = HistoricalCalibration(params=mean_params)
        mean_validation = hc_mean.run_all_validations()

        # 2. 后验预测区间（采样 n_total 组参数，分别计算指标）
        sp_durations = []
        sn_durations = []
        clearances = []
        progressions = []

        n_pool = len(next(iter(available.values())))
        indices = rng.choice(n_pool, size=min(n_total, n_pool), replace=False)

        for idx in indices:
            sample_params = {k: float(v[idx]) for k, v in available.items()}
            try:
                hc_i = HistoricalCalibration(params=sample_params)
                dur = hc_i.compute_tb_durations()
                clr = hc_i.compute_self_clearance_rate(years=years)
                prg = hc_i.compute_progression_rate(years=years)
                sp_durations.append(dur['smear_positive_years'])
                sn_durations.append(dur['smear_negative_years'])
                clearances.append(clr['clearance_pct'])
                progressions.append(prg['progression_pct'])
            except Exception as _e:
                # 跳过数值异常的样本（如流出率为 0）
                LOGGER.debug("跳过数值异常的历史校准样本 params=%s: %s", sample_params, _e)
                continue

        def _pctl(arr):
            if not arr:
                return [float('nan')] * 3
            a = np.asarray(arr, dtype=float)
            return [float(np.percentile(a, 2.5)),
                    float(np.percentile(a, 50.0)),
                    float(np.percentile(a, 97.5))]

        posterior_predictive = {
            'tb_durations_sp': _pctl(sp_durations),
            'tb_durations_sn': _pctl(sn_durations),
            'clearance_pct': _pctl(clearances),
            'progression_pct': _pctl(progressions),
        }

        return {
            'posterior_mean_validation': mean_validation,
            'posterior_predictive': posterior_predictive,
            'n_samples': n_total,
            'n_effective': len(sp_durations),
            'literature': LITERATURE,
        }
