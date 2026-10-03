#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEIR 理论拓展综合对比（Synthesis）

将结核病建模的前沿拓展整合为可发表的四维对比框架：

1. 年龄结构化（Age structure）      — 复用 v4 的 WAIFW + 年龄特异进展率
2. 超级传播（Superspreading）      — 见 superspreading.py（负二项子代分布）
3. TB-HIV 共感染（HIV co-infection）— 复用 v4 的 rr_hiv / ART 分层
4. 耐药结核病（DR-TB）             — 复用 v4 的适合度代价 + 获得性耐药

本模块作为"综合对比层"，把 v4 已内置的三维（年龄/HIV/耐药）与新增的超级传播
维度统一起来，输出可直接入论文的结构化结果（``build_theory_report``）。

文献依据：
- Mossong J et al. (2008). Social contacts and mixing patterns relevant to the
  spread of infectious diseases. PLoS Med 5(3):e74.（POLYMOD 接触矩阵）
- Marais BJ et al. (2011). Childhood pulmonary TB: old wisdom and new challenges.
  IJTLD 15(11):1478-1485.（年龄特异进展风险）
- Davies PD (2006). TB in the elderly. JRSM 99(6):298-303.（老年再激活）
- Pawlowski A et al. (2012). TB and HIV co-infection. PLoS Pathog 8(2):e1002464.
- Houben RMGJ et al. (2016). TIME Impact. BMC Med 14:56.（HIV/ART 分层）
- Karmakar M et al. (2022). Drug-resistant TB expansion. BMC Infect Dis 22:82.
- Lloyd-Smith JO et al. (2005). Nature 438:355-359.（超级传播）
- Melsew YA et al. (2019). BMC Infect Dis 19:244.（TB 超级传播）

依赖：numpy（可选调用 v4 时需 tb_risk.seir.stochastic，惰性导入避免循环依赖）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .superspreading import (
    DEFAULT_R0, DEFAULT_K,
    superspreading_metrics,
    compare_homogeneous_vs_heterogeneous,
    extinction_probability,
)

LOGGER = logging.getLogger("tb_risk.seir.extensions.synthesis")


# ============================================================================
# 复用 v4 的辅助（惰性导入，避免与 seir/__init__ 循环依赖）
# ============================================================================

def _v4():
    """惰性导入 v4 模型常量与 RHS。"""
    from tb_risk.seir.stochastic import (  # noqa: F401
        N_AGE_V4, N_HIV_V4, N_DR_V4, N_COMPARTMENTS_V3,
        IDX_ISP_V3, IDX_ISN_V3,
        DEFAULT_WAIFW, DEFAULT_AGE_PROGRESSION,
        DEFAULT_RR_HIV, DEFAULT_ART_REDUCTION,
        DEFAULT_DR_FITNESS_COST, DEFAULT_P_ACQ,
        # v3 房室默认参数（RHS 关键字参数）
        DEFAULT_RHO_FAST, DEFAULT_RHO_CONV, DEFAULT_RHO_REACT,
        DEFAULT_SIGMA_CLEAR, DEFAULT_RHO_PROG, DEFAULT_ETA_SUB,
        DEFAULT_P_CLIN, DEFAULT_BETA_REINF, DEFAULT_RHO_MIN, DEFAULT_P_M,
        DEFAULT_OMEGA_REG_M, DEFAULT_OMEGA_REG_SUB, DEFAULT_P_SP,
        DEFAULT_MU_SP, DEFAULT_R_SP, DEFAULT_MU_SN, DEFAULT_R_SN,
        DEFAULT_P_SN2SP, DEFAULT_ETA_SN, DEFAULT_BETA_EXO,
        StochasticSEIRModel,
    )
    return {
        "N_AGE_V4": N_AGE_V4, "N_HIV_V4": N_HIV_V4, "N_DR_V4": N_DR_V4,
        "N_COMP": N_COMPARTMENTS_V3,
        "IDX_ISP": IDX_ISP_V3, "IDX_ISN": IDX_ISN_V3,
        "WAIFW": DEFAULT_WAIFW, "AGE_PROG": DEFAULT_AGE_PROGRESSION,
        "RR_HIV": DEFAULT_RR_HIV, "ART_RED": DEFAULT_ART_REDUCTION,
        "DR_FIT": DEFAULT_DR_FITNESS_COST, "P_ACQ": DEFAULT_P_ACQ,
        "rhs": StochasticSEIRModel().deterministic_rhs_v4,
        # v3 房室默认参数（供 RHS 调用）
        "RHO_FAST": DEFAULT_RHO_FAST, "RHO_CONV": DEFAULT_RHO_CONV,
        "RHO_REACT": DEFAULT_RHO_REACT, "SIGMA_CLEAR": DEFAULT_SIGMA_CLEAR,
        "RHO_PROG": DEFAULT_RHO_PROG, "ETA_SUB": DEFAULT_ETA_SUB,
        "P_CLIN": DEFAULT_P_CLIN, "BETA_REINF": DEFAULT_BETA_REINF,
        "RHO_MIN": DEFAULT_RHO_MIN, "P_M": DEFAULT_P_M,
        "OMEGA_REG_M": DEFAULT_OMEGA_REG_M,
        "OMEGA_REG_SUB": DEFAULT_OMEGA_REG_SUB, "P_SP": DEFAULT_P_SP,
        "MU_SP": DEFAULT_MU_SP, "R_SP": DEFAULT_R_SP,
        "MU_SN": DEFAULT_MU_SN, "R_SN": DEFAULT_R_SN,
        "P_SN2SP": DEFAULT_P_SN2SP, "ETA_SN": DEFAULT_ETA_SN,
        "BETA_EXO": DEFAULT_BETA_EXO,
    }


def _base_params() -> dict:
    """返回 v3 房室默认参数 dict（供 RHS 关键字调用）。"""
    v = _v4()
    return {
        "rho_fast": v["RHO_FAST"], "rho_conv": v["RHO_CONV"],
        "rho_react": v["RHO_REACT"], "sigma_clear": v["SIGMA_CLEAR"],
        "rho_prog": v["RHO_PROG"], "eta_sub": v["ETA_SUB"],
        "p_clin": v["P_CLIN"], "beta_reinf": v["BETA_REINF"],
        "rho_min": v["RHO_MIN"], "p_m": v["P_M"],
        "omega_reg_m": v["OMEGA_REG_M"],
        "omega_reg_sub": v["OMEGA_REG_SUB"], "p_sp": v["P_SP"],
        "mu_sp": v["MU_SP"], "r_sp": v["R_SP"],
        "mu_sn": v["MU_SN"], "r_sn": v["R_SN"],
        "p_sn2sp": v["P_SN2SP"], "eta_sn": v["ETA_SN"],
        "beta_exo": v["BETA_EXO"],
    }


AGE_LABELS = ["0-4", "5-14", "15-49", "50-64", "65+"]


def _make_v4_initial_state(population: int = 1000, i0: int = 10) -> np.ndarray:
    """构造 270D 初始状态 (5 age, 3 HIV, 2 DR, 9 comp)。"""
    v = _v4()
    N_AGE, N_HIV, N_DR, NC = v["N_AGE_V4"], v["N_HIV_V4"], v["N_DR_V4"], v["N_COMP"]
    state = np.zeros((N_AGE, N_HIV, N_DR, NC))
    age_dist = np.array([0.06, 0.12, 0.55, 0.18, 0.09])
    n_inf = i0
    # 初始感染分到 I_sub / I_sp / I_sn
    isub0 = int(n_inf * 0.5)
    isp0 = int(n_inf * 0.3)
    isn0 = max(0, n_inf - isub0 - isp0)
    lf_total = min(i0 * 30, population - i0)
    for a in range(N_AGE):
        n_a = int(population * age_dist[a])
        s_a = int(n_a * 0.95)
        lf_a = int(lf_total * age_dist[a])
        state[a, 0, 0, 0] = max(s_a - lf_a, 0)   # S
        state[a, 0, 0, 1] = lf_a                 # L_fast
        # 初始传染者放在 15-49 岁（age=2）HIV-/DS 层
    state[2, 0, 0, 4] = isub0
    state[2, 0, 0, 5] = isp0
    state[2, 0, 0, 6] = isn0
    return state


def _rk4(rhs, y0: np.ndarray, t_span, dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """RK4 ODE 积分器（state 展平为 1D）。"""
    t0, t1 = t_span
    n_steps = max(1, int((t1 - t0) / dt) + 1)
    times = np.linspace(t0, t1, n_steps)
    y = np.asarray(y0, dtype=float).flatten().copy()
    hist = np.zeros((n_steps, y.shape[0]))
    hist[0] = y
    for i in range(1, n_steps):
        t = times[i - 1]
        h = dt
        k1 = np.asarray(rhs(y, t, **kwargs), dtype=float)
        k2 = np.asarray(rhs(y + h / 2 * k1, t + h / 2, **kwargs), dtype=float)
        k3 = np.asarray(rhs(y + h / 2 * k2, t + h / 2, **kwargs), dtype=float)
        k4 = np.asarray(rhs(y + h * k3, t + h, **kwargs), dtype=float)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        y = np.maximum(y, 0)
        hist[i] = y
    return times, hist


def _active_tb_by_age(state_flat: np.ndarray) -> np.ndarray:
    """返回每个年龄组的活动性 TB（I_sp + I_sn）人数，(5,)。"""
    v = _v4()
    st = np.asarray(state_flat, dtype=float).reshape(
        v["N_AGE_V4"], v["N_HIV_V4"], v["N_DR_V4"], v["N_COMP"])
    active = (st[:, :, :, v["IDX_ISP"]] + st[:, :, :, v["IDX_ISN"]]).sum(axis=(1, 2))
    return active


# ============================================================================
# 各维度分析
# ============================================================================

def age_structured_analysis(population: int = 1000, i0: int = 10,
                            t_span=(0.0, 365.0 * 3), dt: float = 1.0,
                            beta: float = 0.05) -> dict:
    """年龄结构化分析：WAIFW 混合 + 年龄特异进展 → 各年龄组活动性 TB 负担。

    返回：
        dict: {age_labels, age_distribution, total_active_by_age,
               peak_active_by_age, dominant_age, note}
    """
    v = _v4()
    y0 = _make_v4_initial_state(population, i0)
    base = _base_params()
    times, hist = _rk4(v["rhs"], y0, t_span, dt, beta=beta,
                       waifw=v["WAIFW"], age_progression=v["AGE_PROG"],
                       **base)
    active_by_age = np.array([_active_tb_by_age(row) for row in hist])
    total = active_by_age.sum(axis=0)
    peak = active_by_age.max(axis=0)
    dom_idx = int(np.argmax(total))
    return {
        "age_labels": list(AGE_LABELS),
        "age_distribution": [0.06, 0.12, 0.55, 0.18, 0.09],
        "total_active_by_age": [float(x) for x in total],
        "peak_active_by_age": [float(x) for x in peak],
        "dominant_age_group": AGE_LABELS[dom_idx],
        "note": (
            f"WAIFW 同配混合 + 年龄特异进展率下，活动性 TB 负担最高的年龄组为"
            f" {AGE_LABELS[dom_idx]}（青壮年接触强度高、老年再激活风险高）。"
            "对克拉玛依油田营地这类明确年龄段封闭社区，年龄结构决定传播骨架。"
        ),
    }


def hiv_analysis(population: int = 1000, i0: int = 10,
                 t_span=(0.0, 365.0 * 3), dt: float = 1.0,
                 beta: float = 0.05) -> dict:
    """TB-HIV 共感染分析：HIV 再激活倍数 rr_hiv 对活动性 TB 负担的影响。

    对比 rr_hiv=1（无 HIV 效应）与默认 rr_hiv=20（Pawlowski 2012）。

    返回：
        dict: {rr_hiv_values, total_active, fold_increase, note}
    """
    v = _v4()
    y0 = _make_v4_initial_state(population, i0)
    base = _base_params()
    results = []
    for rr in (1.0, v["RR_HIV"]):
        kwargs = dict(base, rr_hiv=rr, art_reduction=v["ART_RED"])
        _, hist = _rk4(v["rhs"], y0, t_span, dt, beta=beta,
                       waifw=v["WAIFW"], age_progression=v["AGE_PROG"], **kwargs)
        total = np.array([_active_tb_by_age(row).sum() for row in hist]).sum()
        results.append(float(total))
    fold = results[1] / max(results[0], 1e-12)
    return {
        "rr_hiv_values": [1.0, v["RR_HIV"]],
        "total_active_cases": results,
        "fold_increase_with_hiv": float(fold),
        "note": (
            f"HIV 再激活倍数从 1 增至 {v['RR_HIV']:.0f} 后，研究期内活动性 TB "
            f"总负担增大约 {fold:.2f} 倍，凸显 TB-HIV 共感染在新疆高发情景下的"
            "公共卫生意义。"
        ),
    }


def dr_analysis(population: int = 1000, i0: int = 10,
                t_span=(0.0, 365.0 * 3), dt: float = 1.0,
                beta: float = 0.05) -> dict:
    """耐药 TB 分析：适合度代价 + 获得性耐药 → DR 株占比演化。

    对比 p_acq=0（无获得性耐药）与默认 p_acq=0.03。

    返回：
        dict: {dr_fraction_without_acq, dr_fraction_with_acq,
               acq_dr_fraction, note}
    """
    v = _v4()
    y0 = _make_v4_initial_state(population, i0)
    base = _base_params()

    def _final_dr_fraction(p_acq):
        kwargs = dict(base, dr_fitness_cost=v["DR_FIT"], p_acq=p_acq)
        _, hist = _rk4(v["rhs"], y0, t_span, dt, beta=beta,
                       waifw=v["WAIFW"], age_progression=v["AGE_PROG"], **kwargs)
        st = hist[-1].reshape(v["N_AGE_V4"], v["N_HIV_V4"], v["N_DR_V4"], v["N_COMP"])
        active = st[:, :, :, v["IDX_ISP"]] + st[:, :, :, v["IDX_ISN"]]
        ds = active[:, :, 0].sum()
        dr = active[:, :, 1].sum()
        return float(dr / max(ds + dr, 1e-12)), float(dr), float(ds + dr)

    frac_no, dr_no, _tot_no = _final_dr_fraction(0.0)
    frac_acq, dr_acq, _tot_acq = _final_dr_fraction(v["P_ACQ"])
    return {
        "dr_fraction_without_acq": float(frac_no),
        "dr_fraction_with_acq": float(frac_acq),
        "acq_dr_increment": float(frac_acq - frac_no),
        "note": (
            f"考虑获得性耐药（p_acq={v['P_ACQ']}）后，研究期末活动性 TB 中"
            f"DR 株占比由 {frac_no*100:.2f}% 升至 {frac_acq*100:.2f}%。"
            "DR 株适合度代价（传染性降 15%）与获得性耐药共同决定耐药传播链。"
        ),
    }


# ============================================================================
# 综合报告
# ============================================================================

def build_theory_report(population: int = 1000, i0: int = 10,
                        t_span=(0.0, 365.0 * 3), dt: float = 1.0,
                        beta: float = 0.05,
                        superspread_R0: float = DEFAULT_R0,
                        superspread_k: float = DEFAULT_K) -> dict:
    """构建 SEIR 理论拓展综合报告（面向论文）。

    整合四个维度：
    - 年龄结构化（age）
    - 超级传播（superspreading）
    - TB-HIV 共感染（hiv）
    - 耐药 TB（dr）

    返回：
        dict: {summary, age, superspreading, hiv, dr, references, status}
    """
    try:
        age = age_structured_analysis(population, i0, t_span, dt, beta)
    except Exception as e:  # v4 不可用时降级
        LOGGER.warning("年龄结构化分析失败，降级: %s", e)
        age = {"status": "skipped", "error": str(e)}
    try:
        hiv = hiv_analysis(population, i0, t_span, dt, beta)
    except Exception as e:
        LOGGER.warning("HIV 分析失败，降级: %s", e)
        hiv = {"status": "skipped", "error": str(e)}
    try:
        dr = dr_analysis(population, i0, t_span, dt, beta)
    except Exception as e:
        LOGGER.warning("耐药 TB 分析失败，降级: %s", e)
        dr = {"status": "skipped", "error": str(e)}

    ss = superspreading_metrics(superspread_R0, superspread_k)
    ss_compare = compare_homogeneous_vs_heterogeneous(
        population=population, R0=superspread_R0)

    return {
        "summary": {
            "title": "SEIR 模型理论拓展综合报告",
            "dimensions": ["年龄结构化", "超级传播", "TB-HIV 共感染", "耐药 TB"],
            "population": int(population),
            "R0": float(superspread_R0),
            "dispersion_k": float(superspread_k),
        },
        "age": age,
        "superspreading": {
            "metrics": ss,
            "homogeneous_vs_heterogeneous": ss_compare,
        },
        "hiv": hiv,
        "dr": dr,
        "references": [
            "Mossong J, et al. PLoS Med 2008;5(3):e74.",
            "Marais BJ, et al. Int J Tuberc Lung Dis 2011;15(11):1478-1485.",
            "Davies PD. J R Soc Med 2006;99(6):298-303.",
            "Pawlowski A, et al. PLoS Pathog 2012;8(2):e1002464.",
            "Houben RMGJ, et al. BMC Med 2016;14:56.",
            "Karmakar M, et al. BMC Infect Dis 2022;22:82.",
            "Lloyd-Smith JO, et al. Nature 2005;438(7066):355-359.",
            "Melsew YA, et al. BMC Infect Dis 2019;19:244.",
        ],
        "status": "ok",
    }


__all__ = [
    "AGE_LABELS",
    "age_structured_analysis",
    "hiv_analysis",
    "dr_analysis",
    "build_theory_report",
]