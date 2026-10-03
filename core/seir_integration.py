#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEIR 集成模块（纯业务逻辑层）

将 SEIR 传播动力学相关的概率计算与模拟从 RiskAssessmentService 中拆分出来，
形成独立的协作类。零依赖 tkinter 或任何 GUI 库。

职责：
- 时间依赖性基础风险系数计算（基于治疗状态与暴露时间）
- 从 SEIR 后验推断感染概率 P_seir
- SEIR 推断的置信度权重 w_seir
- SEIR 随机模拟
- SEIR/规则概率融合（log 空间）
"""

import hashlib
import logging
import math
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

from ..constants import (
    SEIR_INITIAL_PREVALENCE,
    SEIR_DEFAULT_BETA,
    SEIR_DEFAULT_BETA_POSTERIOR,
    SEIR_DEFAULT_SIGMA,
    SEIR_DEFAULT_GAMMA,
    SEIR_SIMULATION_DAYS,
    SEIR_N_TRAJECTORIES,
    SEIR_MCMC_ESS_THRESHOLD,
    SEIR_RANDOM_SEED,
    SEIR_INITIAL_S, SEIR_INITIAL_E, SEIR_INITIAL_I, SEIR_INITIAL_R,
    EXPOSURE_EARLY_PHASE_CAP,
    EXPOSURE_EARLY_RATE,
    EXPOSURE_LATE_RATE,
    EXPOSURE_PHASE_THRESHOLD_DAYS,
)
from .lab_evidence import mdr_treatment_infectivity

LOGGER = logging.getLogger("tb_risk.core")


def _stable_repr(obj):
    """将对象规范化为确定性可哈希结构 (用于缓存键)

    - dict: 按键排序 (键先转 str 保证可排序), 与插入顺序无关
    - list/tuple: 逐元素递归
    - numpy.ndarray: shape + dtype + tobytes (内容敏感)
    - 标量/None: (类型名, repr)
    - 其他: repr (尽力而为)

    保证: 相同内容 → 相同返回结构 → 相同 MD5 键。
    """
    if isinstance(obj, dict):
        return ('dict', tuple(
            (_stable_repr(k), _stable_repr(obj[k]))
            for k in sorted(obj, key=lambda x: str(x))))
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, tuple(_stable_repr(x) for x in obj))
    if isinstance(obj, np.ndarray):
        return ('ndarray', obj.shape, str(obj.dtype), obj.tobytes())
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return ('scalar', type(obj).__name__, repr(obj))
    return ('obj', repr(obj))


def _sim_cache_key(patient_info, family_members, social_contacts, params):
    """构建 SEIR 模拟缓存的 MD5 键 (接触网络 + 参数组合)

    将接触网络 (patient_info + family_members + social_contacts) 与模拟
    参数规范化为确定性字节并哈希。相同接触网络与参数组合 → 相同键,
    从而在 :meth:`SEIRIntegration.run_seir_simulation` 中复用模拟结果。

    Args:
        patient_info: dict 患者信息
        family_members: list[dict] 家庭成员
        social_contacts: list[dict] 社会接触者
        params: dict 模拟参数 (initial_state/t_span/dt/beta/sigma/gamma/
            n_trajectories/seed 等标量)

    Returns:
        str: 32 字符 MD5 十六进制
    """
    payload = (
        _stable_repr(patient_info),
        _stable_repr(family_members),
        _stable_repr(social_contacts),
        _stable_repr(params),
    )
    return hashlib.md5(repr(payload).encode('utf-8')).hexdigest()


class _SEIRSimulationCache:
    """SEIR 模拟结果 LRU 缓存 (线程安全)

    增量与缓存计算: GNN 已有增量推理, 这里将 SEIR 模拟结果也纳入缓存 —
    相同接触网络与参数组合的模拟 (随机轨迹) 结果按 MD5 键复用, 避免
    重复的 ODE/SDE 随机模拟。LRU 淘汰 + threading.Lock 保护, 与
    ``seir.likelihood_cache.LikelihoodCache`` 的设计一致。

    Args:
        maxsize: 最大条目数 (默认 64, 超过后 LRU 淘汰)
    """

    def __init__(self, maxsize=64):
        self.maxsize = max(int(maxsize), 1)
        self._store = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key):
        """查询缓存 (未命中返回 None)"""
        with self._lock:
            if key in self._store:
                # LRU: 移到末尾 (最近访问)
                value = self._store.pop(key)
                self._store[key] = value
                self._hits += 1
                return value
            self._misses += 1
            return None

    def put(self, key, value):
        """存入缓存 (已存在则更新为最近访问)"""
        with self._lock:
            if key in self._store:
                self._store.pop(key)
            while len(self._store) >= self.maxsize:
                self._store.popitem(last=False)
            self._store[key] = value

    def stats(self):
        """返回缓存统计 (size/maxsize/hits/misses/hit_rate)"""
        with self._lock:
            total = self._hits + self._misses
            return {
                'size': len(self._store),
                'maxsize': self.maxsize,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self._hits / total if total > 0 else 0.0,
            }


def fuse_probabilities(p_seir, p_rule, w_seir):
    """在 log 空间中融合 SEIR 概率和规则概率，返回 (0-100)

    参数：
        p_seir: float — SEIR 推断的感染概率 (0-1)
        p_rule: float — 规则引擎的感染概率 (0-1)
        w_seir: float — SEIR 置信度权重 ∈ [0, 1]

    返回：
        float: 融合后的感染概率 (0-100)
    """
    w_rule = 1.0 - w_seir
    eps = 1e-10
    p_seir_safe = max(p_seir, eps)
    p_rule_safe = max(p_rule, eps)
    # log 空间加权：log(p) 的凸组合，等价于 p_seir^w_seir * p_rule^w_rule
    log_fused = w_seir * math.log(p_seir_safe) + w_rule * math.log(p_rule_safe)
    p_fused = math.exp(log_fused)
    return min(p_fused * 100.0, 100.0)


class SEIRIntegration:
    """SEIR 传播动力学集成器

    封装所有与 SEIR 模型相关的概率计算与模拟逻辑。
    通过依赖注入接收 SEIRParameterUncertainty 实例。
    """

    def __init__(self, seir_param_uncertainty=None,
                 simulation_cache_maxsize=64):
        """初始化 SEIR 集成器

        参数：
            seir_param_uncertainty: SEIRParameterUncertainty | None — SEIR 参数不确定性实例
            simulation_cache_maxsize: int — SEIR 模拟结果 LRU 缓存条目数
                (增量与缓存计算: 相同接触网络与参数组合的模拟结果复用;
                需要跳过缓存时用 ``run_seir_simulation(use_cache=False)``)
        """
        self._seir_param_uncertainty = seir_param_uncertainty
        self._sim_cache = _SEIRSimulationCache(
            maxsize=simulation_cache_maxsize)

    def set_param_uncertainty(self, seir_param_uncertainty) -> None:
        """设置 SEIR 参数不确定性实例（支持延迟初始化）"""
        self._seir_param_uncertainty = seir_param_uncertainty

    def calculate_time_dependent_risk(self, contact, patient_treatment_days,
                                      is_mdr=False):
        """计算基于 SEIR 模型的时间依赖性基础风险系数（0-1）

        仅返回基于治疗状态和暴露时间的风险系数。

        参数：
            contact: 接触者记录
            patient_treatment_days: 患者已治疗天数
            is_mdr: 源病例是否 MDR（genexpert_rif=resistant）。
                MDR 时治疗传染性衰减曲线拉长（见 mdr_treatment_infectivity）。
        """
        try:
            patient_treatment_days = (
                int(patient_treatment_days)
                if patient_treatment_days is not None else 0
            )
            infectivity = mdr_treatment_infectivity(
                patient_treatment_days, is_mdr=bool(is_mdr))

            time_span = int(contact.get('time_span', 4)) if contact.get('time_span') is not None else 4
            exposure_days = time_span * 7

            if exposure_days <= EXPOSURE_PHASE_THRESHOLD_DAYS:
                exposure_to_infectious_prob = EXPOSURE_EARLY_PHASE_CAP * (1 - math.exp(-EXPOSURE_EARLY_RATE * exposure_days))
            else:
                early_risk = EXPOSURE_EARLY_PHASE_CAP * (1 - math.exp(-EXPOSURE_EARLY_RATE * EXPOSURE_PHASE_THRESHOLD_DAYS))
                remaining_risk = 1.0 - early_risk
                late_risk = remaining_risk * (1 - math.exp(-EXPOSURE_LATE_RATE * (exposure_days - EXPOSURE_PHASE_THRESHOLD_DAYS)))
                exposure_to_infectious_prob = early_risk + late_risk

            base_risk = infectivity * exposure_to_infectious_prob
            return min(max(base_risk, 0.0), 1.0)
        except (ValueError, TypeError, KeyError) as e:
            LOGGER.warning("计算时间依赖风险时发生错误: %s", e, exc_info=True)
            return 0.0

    def compute_seir_probability(self, contact, patient_treatment_days,
                                 is_mdr=False):
        """从 SEIR 后验推断计算感染概率 P_seir (0-1)

        参数：
            contact: 接触者记录
            patient_treatment_days: 患者已治疗天数
            is_mdr: 源病例是否 MDR（genexpert_rif=resistant）。
                MDR 时治疗传染性衰减曲线拉长（见 mdr_treatment_infectivity）。
        """
        if self._seir_param_uncertainty is None:
            return None
        if not self._seir_param_uncertainty.inference_completed:
            return None

        summary = self._seir_param_uncertainty.posterior_summary
        if summary is None:
            return None

        try:
            beta_mean = summary.get('beta', {}).get('mean', SEIR_DEFAULT_BETA_POSTERIOR)

            single_duration_min = float(contact.get('single_duration', 60))
            freq_density = float(contact.get('freq_density', 2))
            time_span = float(contact.get('time_span', 4))

            contact_intensity = (single_duration_min * freq_density) / (7 * 24 * 60)
            delta_t = time_span * 7
            I_over_N = SEIR_INITIAL_PREVALENCE

            infectivity = mdr_treatment_infectivity(
                patient_treatment_days, is_mdr=bool(is_mdr))

            effective_beta = beta_mean * infectivity
            lambda_seir = effective_beta * I_over_N * contact_intensity * delta_t
            p_seir = 1.0 - math.exp(-max(lambda_seir, 0.0))
            return min(p_seir, 1.0)
        except (ValueError, TypeError, ZeroDivisionError) as e:
            LOGGER.debug("SEIR 概率计算失败: %s", e)
            return None

    def compute_seir_confidence_weight(self):
        """计算 SEIR 推断的置信度权重 w_seir ∈ [0, 1]"""
        if self._seir_param_uncertainty is None:
            return 0.0

        diagnostics = self._seir_param_uncertainty.mcmc_diagnostics
        if not diagnostics:
            return 0.0

        r_hat_list = diagnostics.get('r_hat', [])
        ess_list = diagnostics.get('ess', [])

        if not r_hat_list or not ess_list:
            return 0.0

        ess_ratio = min([min(ess / SEIR_MCMC_ESS_THRESHOLD, 1.0) for ess in ess_list]) if ess_list else 0.0
        r_hat_max = max(r_hat_list) if r_hat_list else 1.0
        r_hat_factor = 1.0 / max(r_hat_max, 1.0)

        w_seir = ess_ratio * r_hat_factor
        return min(max(w_seir, 0.0), 1.0)

    def run_seir_simulation(
        self, patient_info: Dict,
        family_members: List[Dict],
        social_contacts: List[Dict],
        beta: Optional[float] = None,
        sigma: Optional[float] = None,
        gamma: Optional[float] = None,
        n_trajectories: Optional[int] = None,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """运行 SEIR 模拟（可选）

        增量与缓存计算: 相同接触网络 (patient_info + family_members +
        social_contacts) 与参数组合 (beta/sigma/gamma/n_trajectories 等)
        的模拟结果按 MD5 键缓存复用 (LRU 淘汰, 线程安全), 避免重复的
        随机模拟。统计见 :meth:`simulation_cache_stats`。

        Args:
            patient_info: 患者信息
            family_members: 家庭成员列表
            social_contacts: 社会接触者列表
            beta: 传播率 (None 用 SEIR_DEFAULT_BETA)
            sigma: 潜伏→传染进展率 (None 用 SEIR_DEFAULT_SIGMA)
            gamma: 恢复率 (None 用 SEIR_DEFAULT_GAMMA)
            n_trajectories: 轨迹数 (None 用 SEIR_N_TRAJECTORIES)
            use_cache: 是否启用结果缓存 (默认 True)

        Returns:
            dict {'seir_trajectories': ...} | None
        """
        beta = SEIR_DEFAULT_BETA if beta is None else float(beta)
        sigma = SEIR_DEFAULT_SIGMA if sigma is None else float(sigma)
        gamma = SEIR_DEFAULT_GAMMA if gamma is None else float(gamma)
        n_trajectories = (SEIR_N_TRAJECTORIES if n_trajectories is None
                          else int(n_trajectories))
        initial_state = (SEIR_INITIAL_S, SEIR_INITIAL_E,
                         SEIR_INITIAL_I, SEIR_INITIAL_R)
        t_span = (0.0, SEIR_SIMULATION_DAYS)

        params = {
            'beta': beta, 'sigma': sigma, 'gamma': gamma,
            'n_trajectories': n_trajectories,
            'initial_state': initial_state, 't_span': t_span,
            'dt': 1.0, 'seed': SEIR_RANDOM_SEED,
        }

        try:
            cache_key = _sim_cache_key(
                patient_info, family_members, social_contacts, params)
            if use_cache:
                cached = self._sim_cache.get(cache_key)
                if cached is not None:
                    return cached

            from ..seir.stochastic import StochasticSEIRModel
            model = StochasticSEIRModel(
                population=len(family_members) + len(social_contacts) + 1,
                seed=SEIR_RANDOM_SEED,
            )
            result = model.simulate_multiple(
                initial_state=initial_state,
                t_span=t_span,
                dt=1.0,
                beta=beta,
                sigma=sigma,
                gamma=gamma,
                n_trajectories=n_trajectories,
            )
            payload = {'seir_trajectories': result}
            if use_cache:
                self._sim_cache.put(cache_key, payload)
            return payload
        except ImportError:
            LOGGER.info("SEIR 模块不可用，跳过模拟")
            return None
        except Exception as e:
            LOGGER.warning("SEIR 模拟失败: %s", e)
            return None

    def simulation_cache_stats(self) -> Dict[str, Any]:
        """返回 SEIR 模拟结果缓存统计

        Returns:
            dict: {
                'size': 当前条目数,
                'maxsize': 最大条目数,
                'hits': 命中次数,
                'misses': 未命中次数,
                'hit_rate': 命中率 (0-1),
            }
        """
        return self._sim_cache.stats()
