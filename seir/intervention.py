#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三层递进架构 — 第 3 层：社区干预反事实模拟（SEIR 只做人群层模拟）。

在"个体基础层 → 网络增强层 → 社区干预层"三层架构中，SEIR 不再承担
"给个体打分"的角色，而是给定第 1、2 层识别出的高风险个体集合，模拟不同
干预策略下社区 6-24 个月的病例数变化，输出**人群层面**的干预效果曲线
（病例数 / 二代病例 / 可避免病例），与个体/网络层在语义上彻底解耦。

支持的干预策略：
- ``preventive_treatment``  高风险接触者预防性治疗：对高风险潜伏感染者用药，
  直接降低其再激活进展率（ρ_react 下调，β_reinf 对已治疗者下调）。
- ``shortened_screening``   缩短筛查间隔：更早发现病例 → 缩短传染期 →
  提高涂阳/涂阴的移除（治疗）速率并小幅降低有效传播率。
- ``combined``              上述两者组合。

实现基于 ``StochasticSEIRModel``（v4，270D 年龄×HIV×耐药分层），基线场景与
各干预场景共享同一初始状态，仅参数不同 —— 即"反事实对照"：
  干预效果 = 基线结果 − 干预场景结果（同一随机种子、同一初始状态）。

文献：
  WHO (2024) 接触者预防性治疗 (TPT) 推荐 — 对高风险接触者减少潜伏进展
  Houben & Dodd (2016) BMC Med — 潜伏感染池规模与再激活负担
  Dowdy et al. (2013) PLoS Med — 筛查频率对病例发现与传播的影响
"""

import numpy as np

from ._stochastic_common import (
    N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3,
    IDX_S_V3, IDX_LF_V3, IDX_LS_V3, IDX_M_V3, IDX_ISUB_V3,
    IDX_ISP_V3, IDX_ISN_V3, IDX_R_V3, IDX_C_V3,
    DEFAULT_RHO_FAST, DEFAULT_RHO_CONV, DEFAULT_RHO_REACT,
    DEFAULT_SIGMA_CLEAR, DEFAULT_RHO_PROG, DEFAULT_ETA_SUB,
    DEFAULT_P_CLIN, DEFAULT_BETA_REINF, DEFAULT_RHO_MIN, DEFAULT_P_M,
    DEFAULT_OMEGA_REG_M, DEFAULT_OMEGA_REG_SUB, DEFAULT_P_SP,
    DEFAULT_MU_SP, DEFAULT_R_SP, DEFAULT_MU_SN, DEFAULT_R_SN,
    DEFAULT_P_SN2SP, DEFAULT_ETA_SN, DEFAULT_BETA_EXO,
    DEFAULT_WAIFW, DEFAULT_AGE_PROGRESSION,
    DEFAULT_RR_HIV, DEFAULT_ART_REDUCTION,
    DEFAULT_HIV_INFECTION_RATE, DEFAULT_ART_INITIATION_RATE,
    DEFAULT_DR_FITNESS_COST, DEFAULT_P_ACQ,
)
from .stochastic import StochasticSEIRModel

# 干预策略参数（默认值，可被 simulate_intervention_effects 参数覆盖）
DEFAULT_PT_EFFICACY = 0.60       # 预防性治疗对再激活进展的相对效果
DEFAULT_PT_REINF_REDUCTION = 0.5  # 预防性治疗后再感染易感性的相对下调
DEFAULT_SS_DETECTION_BOOST = 1.5  # 缩短筛查间隔后移除/治疗速率放大倍数
DEFAULT_SS_BETA_REDUCTION = 0.2   # 缩短筛查间隔后有效传播率相对下调

# 年龄结构权重（近似克拉玛依/城市人口结构，归一化后使用）
_AGE_FRAC = np.array([0.14, 0.16, 0.42, 0.17, 0.11], dtype=float)
# HIV 分层权重：HIV- / HIV+未治疗 / HIV+ART
_HIV_FRAC = np.array([0.95, 0.03, 0.02], dtype=float)
# 耐药分层权重：DS / DR
_DR_FRAC = np.array([0.97, 0.03], dtype=float)


def build_v4_initial_state(n_population=10000, random_state=42):
    """构造 v4（270D 年龄×HIV×耐药分层）初始状态。

    基线比例：S≈SEIR_INITIAL_S、Lf/Ls 潜伏、少量活动性病例（作为传播种子）、
    R/C 恢复/清除。各分层按年龄/HIV/耐药权重分配，总人口守恒到 n_population。

    参数：
        n_population (int): 社区总人口（默认 10000）
        random_state (int): 随机种子（分层内小数舍入的确定性来源）

    返回：
        np.ndarray[5, 3, 2, 9]: 270D 初始状态
    """
    rng = np.random.RandomState(random_state)
    age_frac = _AGE_FRAC / _AGE_FRAC.sum()
    hiv_frac = _HIV_FRAC / _HIV_FRAC.sum()
    dr_frac = _DR_FRAC / _DR_FRAC.sum()

    # 房室比例（与 constants SEIR_INITIAL_* 语义对齐）
    s_frac = 0.99
    lf_frac = 0.004
    ls_frac = 0.02
    m_frac = 0.0004
    isub_frac = 0.0004
    isp_frac = 0.0003
    isn_frac = 0.0002
    r_frac = 0.0010
    c_frac = 0.0050
    # 归一化保证总人口守恒
    comp_fracs = np.array([s_frac, lf_frac, ls_frac, m_frac, isub_frac,
                           isp_frac, isn_frac, r_frac, c_frac])
    comp_fracs = comp_fracs / comp_fracs.sum()

    # 权重张量 [age, hiv, dr]
    w = (age_frac[:, None, None] * hiv_frac[None, :, None]
         * dr_frac[None, None, :])
    w = w / w.sum()

    # 确定性分配（按分层权重切分总人口；余数计入最大层，保证守恒）
    state = np.zeros((N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3),
                     dtype=float)
    counts = np.floor(w * n_population).astype(int)
    remainder = n_population - int(counts.sum())
    if remainder > 0:
        flat = counts.reshape(-1)
        idx = int(np.argmax(flat))
        flat[idx] += remainder
        counts = flat.reshape(counts.shape)
    for c, frac in enumerate(comp_fracs):
        state[:, :, :, c] = counts * frac
    # 保证 S 不随随机性波动：S 取余量（其余房室取整后 S 补足）
    state[:, :, :, IDX_S_V3] = counts - state[:, :, :, 1:].sum(axis=-1)
    state = np.maximum(state, 0)
    return state


def _aggregate_active(state_flat):
    """将单步 270D 状态展平向量聚合为活动性病例总数（I_sp + I_sn）。

    参数：
        state_flat: np.ndarray[270]

    返回：
        float: 活动性病例总数
    """
    st = np.asarray(state_flat, dtype=float).reshape(
        N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
    return float(st[:, :, :, IDX_ISP_V3].sum()
                 + st[:, :, :, IDX_ISN_V3].sum())


def _aggregate_susceptible(state_flat):
    """返回 S 房室总数（用于估计累计新感染 S(0) − S(T)）。"""
    st = np.asarray(state_flat, dtype=float).reshape(
        N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3)
    return float(st[:, :, :, IDX_S_V3].sum())


def _default_v4_params(**overrides):
    """收集 v4 默认参数 dict，供 simulate_sde_v4 展开。"""
    params = dict(
        rho_fast=DEFAULT_RHO_FAST, rho_conv=DEFAULT_RHO_CONV,
        rho_react=DEFAULT_RHO_REACT, sigma_clear=DEFAULT_SIGMA_CLEAR,
        rho_prog=DEFAULT_RHO_PROG, eta_sub=DEFAULT_ETA_SUB,
        p_clin=DEFAULT_P_CLIN, beta_reinf=DEFAULT_BETA_REINF,
        rho_min=DEFAULT_RHO_MIN, p_m=DEFAULT_P_M,
        omega_reg_m=DEFAULT_OMEGA_REG_M, omega_reg_sub=DEFAULT_OMEGA_REG_SUB,
        p_sp=DEFAULT_P_SP, mu_sp=DEFAULT_MU_SP, r_sp=DEFAULT_R_SP,
        mu_sn=DEFAULT_MU_SN, r_sn=DEFAULT_R_SN, p_sn2sp=DEFAULT_P_SN2SP,
        eta_sn=DEFAULT_ETA_SN, beta_exo=DEFAULT_BETA_EXO,
        waifw=DEFAULT_WAIFW, age_progression=DEFAULT_AGE_PROGRESSION,
        rr_hiv=DEFAULT_RR_HIV, art_reduction=DEFAULT_ART_REDUCTION,
        hiv_infection_rate=DEFAULT_HIV_INFECTION_RATE,
        art_initiation_rate=DEFAULT_ART_INITIATION_RATE,
        dr_fitness_cost=DEFAULT_DR_FITNESS_COST, p_acq=DEFAULT_P_ACQ,
    )
    params.update(overrides)
    return params


def _coverage_from_high_risk(high_risk_contacts, coverage=None):
    """由高风险个体数量推算干预覆盖率（0-1）。

    覆盖率随高风险个体数单调上升并封顶 0.9；显式传入 coverage 时直接返回。
    """
    if coverage is not None:
        return float(np.clip(coverage, 0.0, 0.9))
    n_high = 0
    if isinstance(high_risk_contacts, (list, tuple)):
        n_high = len(high_risk_contacts)
    elif isinstance(high_risk_contacts, dict):
        n_high = len(high_risk_contacts)
    return float(np.clip(0.15 + n_high / 200.0, 0.0, 0.9))


def _trapz(y, x):
    """数值梯形积分（兼容 numpy <2.0 的 np.trapz 与 >=2.0 的 np.trapezoid）。"""
    trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    if trapezoid is None:
        # 纯 Python 回退
        total = 0.0
        for i in range(1, len(x)):
            total += (y[i] + y[i - 1]) * (x[i] - x[i - 1]) / 2.0
        return total
    return trapezoid(y, x)


class InterventionCounterfactualSimulator:
    """社区干预反事实模拟器（第 3 层）。

    以 v4 SEIR 为引擎，在**同一初始状态、同一随机种子**下运行基线场景与
    各干预场景，比较活动性病例数曲线，量化"可避免病例"。

    参数：
        population (int): 社区总人口
        noise_scale (float): SDE 噪声幅度（0 = 确定性 ODE）
        seed (int): 随机种子（保证基线/干预同源可复现）
    """

    # 干预策略清单（含中文标签）
    STRATEGIES = {
        'preventive_treatment': '高风险接触者预防性治疗',
        'shortened_screening': '缩短筛查间隔',
        'combined': '预防性治疗 + 缩短筛查间隔（组合）',
    }

    def __init__(self, population=10000, noise_scale=0.0, seed=42):
        self.population = int(population)
        self.noise_scale = float(noise_scale)
        self.seed = int(seed)

    def _run(self, initial_state, t_horizon_days, dt, beta, params):
        """运行一次 v4 模拟并返回 (times, active_curve, new_infections)。

        new_infections = S(0) − S(T)，即水平内累计新感染数。
        """
        model = StochasticSEIRModel(
            population=self.population, noise_scale=self.noise_scale,
            seed=self.seed)
        t_span = (0.0, float(t_horizon_days))
        times, traj, _ = model.simulate_sde_v4(
            initial_state, t_span, dt, beta, **params)
        n_steps = traj.shape[0]
        active = np.array([_aggregate_active(traj[i]) for i in range(n_steps)])
        s0 = _aggregate_susceptible(initial_state)
        sT = _aggregate_susceptible(traj[-1])
        return times, active, max(0.0, s0 - sT)

    def _strategy_params(self, base_params, strategy, coverage):
        """根据策略生成反事实场景的参数覆盖。"""
        if strategy == 'preventive_treatment':
            # 预防性治疗：高风险潜伏池再激活率下调 + 治疗后再感染易感性下调
            return dict(
                rho_react=base_params['rho_react']
                * (1.0 - coverage * DEFAULT_PT_EFFICACY),
                beta_reinf=base_params['beta_reinf']
                * (1.0 - coverage * DEFAULT_PT_REINF_REDUCTION),
            )
        if strategy == 'shortened_screening':
            # 缩短筛查间隔：更早发现 → 缩短传染期（移除/治疗速率放大）
            boost = 1.0 + coverage * DEFAULT_SS_DETECTION_BOOST
            return dict(
                r_sp=base_params['r_sp'] * boost,
                r_sn=base_params['r_sn'] * boost,
                mu_sp=base_params['mu_sp'] * boost,
                mu_sn=base_params['mu_sn'] * boost,
                beta_reinf=base_params['beta_reinf']
                * (1.0 - coverage * DEFAULT_SS_BETA_REDUCTION),
            )
        if strategy == 'combined':
            pt = self._strategy_params(base_params, 'preventive_treatment', coverage)
            ss = self._strategy_params(base_params, 'shortened_screening', coverage)
            merged = dict(base_params)
            merged.update(pt)
            merged.update(ss)
            # 避免重复缩放 beta_reinf：取预防性治疗覆盖值
            return merged
        raise ValueError(f"未知干预策略 '{strategy}'，可选: {list(self.STRATEGIES)}")

    def simulate(self, high_risk_contacts=None, initial_state=None,
                 strategies=None, t_horizon_days=730, dt=7.0, beta=None,
                 coverage=None, random_state=None):
        """运行基线 + 各干预策略的反事实模拟。

        参数：
            high_risk_contacts: 第 1/2 层识别的高风险个体集合（list/dict），
                仅用于推算覆盖率（coverage 显式传入时忽略）
            initial_state: v4 初始状态（None 时用 build_v4_initial_state）
            strategies: 干预策略列表（None 时用全部三种）
            t_horizon_days: 模拟水平（默认 730 天 = 24 个月）
            dt: 时间步长（默认 7 天）
            beta: 有效传播率（默认 0.3，与 SEIR_DEFAULT_BETA 一致）
            coverage: 干预覆盖率（0-1；None 时由高风险人数推算）
            random_state: 随机种子（None 时用 self.seed）

        返回：
            dict: 人群层干预效果报告（各策略曲线 + 汇总指标）
        """
        if beta is None:
            beta = 0.3
        if strategies is None:
            strategies = list(self.STRATEGIES)
        seed = self.seed if random_state is None else int(random_state)
        cov = _coverage_from_high_risk(high_risk_contacts, coverage)

        if initial_state is None:
            initial_state = build_v4_initial_state(
                n_population=self.population, random_state=seed)

        base_params = _default_v4_params()
        # 基线场景
        b_times, b_active, b_new_inf = self._run(
            initial_state, t_horizon_days, dt, beta, base_params)

        report = {
            'level': 3,
            'layer': '社区干预层 (SEIR 反事实模拟)',
            'population': self.population,
            't_horizon_days': t_horizon_days,
            'dt_days': dt,
            'coverage': cov,
            'n_high_risk': _coverage_count(high_risk_contacts),
            'baseline': {
                'cases_curve': b_active.tolist(),
                'times': b_times.tolist(),
                'peak_cases': float(np.max(b_active)),
                'final_cases': float(b_active[-1]),
                'cumulative_case_days': float(_trapz(b_active, b_times)),
                'new_infections': b_new_inf,
            },
            'strategies': {},
            'summary': {},
        }

        for strat in strategies:
            if strat not in self.STRATEGIES:
                continue
            params = self._strategy_params(base_params, strat, cov)
            _, s_active, s_new_inf = self._run(
                initial_state, t_horizon_days, dt, beta, params)
            cum_b = report['baseline']['cumulative_case_days']
            cum_s = float(_trapz(s_active, b_times))
            avoidable = max(0.0, cum_b - cum_s)
            averted = avoidable / cum_b if cum_b > 0 else 0.0
            report['strategies'][strat] = {
                'label': self.STRATEGIES[strat],
                'cases_curve': s_active.tolist(),
                'peak_cases': float(np.max(s_active)),
                'final_cases': float(s_active[-1]),
                'cumulative_case_days': cum_s,
                'new_infections': s_new_inf,
                'avoidable_case_days': avoidable,
                'averted_fraction': float(averted),
                'averted_percent': float(averted * 100.0),
            }

        # 汇总：推荐最优策略
        if report['strategies']:
            best = max(report['strategies'].items(),
                       key=lambda kv: kv[1]['averted_fraction'])
            report['summary'] = {
                'best_strategy': best[0],
                'best_strategy_label': best[1]['label'],
                'best_averted_percent': best[1]['averted_percent'],
                'best_avoidable_case_days': best[1]['avoidable_case_days'],
                'recommendation': (
                    f"最优干预策略为「{best[1]['label']}」，在 {t_horizon_days:.0f} 天"
                    f"水平内预计可避免 {best[1]['avoidable_case_days']:.1f} 个病例·天"
                    f"（相对下降 {best[1]['averted_percent']:.1f}%）"),
            }
        return report


def _coverage_count(high_risk_contacts):
    """统计高风险个体数量（list/dict 兼容）。"""
    if isinstance(high_risk_contacts, (list, tuple)):
        return len(high_risk_contacts)
    if isinstance(high_risk_contacts, dict):
        return len(high_risk_contacts)
    return 0


def simulate_intervention_effects(high_risk_contacts=None, initial_state=None,
                                  strategies=None, t_horizon_days=730, dt=7.0,
                                  beta=None, coverage=None, population=10000,
                                  noise_scale=0.0, random_state=42):
    """模块级便捷入口（幂等、确定性），见 InterventionCounterfactualSimulator。"""
    simulator = InterventionCounterfactualSimulator(
        population=population, noise_scale=noise_scale, seed=random_state)
    return simulator.simulate(
        high_risk_contacts=high_risk_contacts, initial_state=initial_state,
        strategies=strategies, t_horizon_days=t_horizon_days, dt=dt,
        beta=beta, coverage=coverage, random_state=random_state)
