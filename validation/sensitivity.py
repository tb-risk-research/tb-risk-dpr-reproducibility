#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局敏感性分析模块 — Morris 筛选 + Sobol 方差分解

文献支撑：
- Morris MD. Factorial sampling plans for preliminary computational
  experiments. Technometrics 33(2):161-174, 1991.
- Sobol IM. Global sensitivity indices for nonlinear mathematical models.
  Math Comput Simul 55(1-3):271-280, 2001.
- Saltelli A et al. Variance based sensitivity analysis of model output.
  Comput Phys Commun 181(2):259-270, 2010.
- Iwanaga T et al. SALib: an open-source Python library for sensitivity
  analysis. JOSS, 2022.

架构改进（v2.0）：
- 消除 _compute_dp_with_params 中约 100 行代码重复
- 通过 ScoringEngine.apply_param_overrides() 注入参数
- 敏感性分析直接评估真实评分引擎，而非复制逻辑
- Morris 作为快速筛选（~150 次调用），Sobol 作为精细分析（~20480 次调用）
"""

import math
import random
import logging

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..scoring.engine import ScoringEngine

logger = logging.getLogger("tb_risk.validation.sensitivity")


# ==================== 参数定义 ====================

# 敏感性分析参数定义（9 个）。
#
# ⚠️ 已移除 base_incidence_per_100k 与 centralized_trt_rate（原 11 个中的 2 个）：
# 本分析通过 ScoringEngine.compute_risk_score 评估每接触者风险，而这两个参数
# 在该函数中无任何消费点（apply_param_overrides 不处理，compute_risk_score 不读取），
# 导致 Morris μ*/Sobol S1/ST 恒为 0，产生"关键流行病学参数不显著"的误导。
#   - base_incidence_per_100k：本土发病率。接触者追踪场景下感染源（指标病例）已知，
#     发病率已隐含在"是否暴露于已知 TB 病例"中，不进入每接触者评分。
#   - centralized_trt_rate：集中收治率。属人群级传播控制（SEIR 侧），不影响个体接触评分。
# 若需分析这两个参数，应在 SEIR/人群级模型上进行，而非 ScoringEngine。
# 格式：(name, lower_bound, upper_bound, category)
SOBOL_PARAM_DEFS = [
    ('oilfield_camp_factor',    0.50,  1.00,  'exposure'),
    ('idu_combined_factor',     4.0,   20.0,  'morbidity'),
    ('pm10_effect_per_50ug',    0.05,  0.35,  'climate'),
    ('humidity_effect_per_10',  0.03,  0.25,  'climate'),
    ('altitude_base_additive',  1.00,  1.50,  'altitude'),
    ('altitude_per_km_additive', 0.03, 0.25,  'altitude'),
    ('altitude_threshold_m',    800,   2500,  'altitude'),
    ('sdoh_housing_factor',     1.00,  1.30,  'sdoh'),
    ('sdoh_access_factor',      1.00,  1.25,  'sdoh'),
]

# Morris 筛选参数（与 Sobol 参数一致，兼容旧接口）
MORRIS_PARAM_DEFS = [
    ('oilfield_camp_factor',    0.50,  1.00, 'exposure'),
    ('idu_combined_factor',     4.0,   20.0, 'morbidity'),
    ('pm10_effect_per_50ug',    0.05,  0.35, 'climate'),
    ('humidity_effect_per_10',  0.03,  0.25, 'climate'),
    ('altitude_base_additive',  1.00,  1.50, 'altitude'),
    ('altitude_per_km_additive', 0.03, 0.25, 'altitude'),
    ('altitude_threshold_m',    800,   2500, 'altitude'),
    ('sdoh_housing_factor',     1.00,  1.30, 'sdoh'),
    ('sdoh_access_factor',      1.00,  1.25, 'sdoh'),
]

# Morris 参数名到 Sobol 参数名的映射
MORRIS_TO_SOBOL_NAME = {
    'oilfield_camp_factor': 'oilfield_camp_factor',
    'idu_combined_factor': 'idu_combined_factor',
    'pm10_effect_per_50ug': 'pm10_effect_per_50ug',
    'humidity_effect_per_10': 'humidity_effect_per_10',
    'altitude_base_additive': 'altitude_base_additive',
    'altitude_per_km_additive': 'altitude_per_km_additive',
    'altitude_threshold_m': 'altitude_threshold_m',
    'sdoh_housing_factor': 'sdoh_housing_factor',
    'sdoh_access_factor': 'sdoh_access_factor',
}

# 参数中文标签
PARAM_LABELS = {
    'oilfield_camp_factor':      '油田营地暴露因子',
    'idu_combined_factor':       'IDU组合免疫抑制因子',
    'pm10_effect_per_50ug':      'PM10效应/50μg',
    'humidity_effect_per_10':    '干旱湿度效应/10%',
    'altitude_base_additive':    '海拔基线加数',
    'altitude_per_km_additive':  '海拔每公里加数',
    'altitude_threshold_m':      '高海拔阈值(m)',
    'sdoh_housing_factor':       'SDOH住房拥挤因子',
    'sdoh_access_factor':        'SDOH医疗可及性因子',
}


# ==================== 共享工具函数 ====================

def _create_engine_with_overrides(localizer, param_dict):
    """创建配置了参数覆盖的 ScoringEngine 实例。

    参数：
        localizer: KaramayLocalizer 实例
        param_dict: dict, 参数名到值的映射

    返回：
        ScoringEngine: 配置了参数覆盖的引擎实例
    """
    engine = ScoringEngine(localizer=localizer, use_localization=True)

    # 构建覆盖字典
    overrides = {}

    # 直接可覆盖的 ScoringEngine 参数
    for key in ['oilfield_camp_factor', 'idu_combined_factor',
                'pm10_effect_per_50ug', 'humidity_effect_per_10',
                'altitude_base_additive', 'altitude_per_km_additive',
                'altitude_threshold_m']:
        if key in param_dict:
            overrides[key] = param_dict[key]

    # SDOH 参数：将住房拥挤和医疗可及性因子组合为 sdoh_combined_factor
    # 加法模型：combined = 1.0 + (housing - 1.0) + (access - 1.0)
    sdoh_housing = param_dict.get('sdoh_housing_factor', 1.0)
    sdoh_access = param_dict.get('sdoh_access_factor', 1.0)
    if 'sdoh_housing_factor' in param_dict or 'sdoh_access_factor' in param_dict:
        sdoh_combined = 1.0 + (sdoh_housing - 1.0) + (sdoh_access - 1.0)
        sdoh_combined = max(0.8, min(2.0, sdoh_combined))
        overrides['sdoh_combined_factor'] = sdoh_combined

    # SETTING_FACTORS_BASE 覆盖（oilfield_camp）
    if 'oilfield_camp_factor' in param_dict:
        overrides['SETTING_FACTORS_BASE'] = dict(engine.SETTING_FACTORS_BASE)
        overrides['SETTING_FACTORS_BASE']['oilfield_camp'] = param_dict[
            'oilfield_camp_factor']

    engine.apply_param_overrides(overrides)
    return engine


def _generate_sample_records(rng_seed=42, n_records=500):
    """生成代表性样本记录（用于敏感性分析评估）。

    参数：
        rng_seed: int, 随机种子
        n_records: int, 记录数量

    返回：
        list[dict]: 接触者记录列表
    """
    rng = random.Random(rng_seed)
    records = []

    settings = ['general', 'closed', 'oilfield_camp', 'crowded', 'outdoor']
    distances = ['very_close', 'close', 'medium', 'far', 'distant']
    illness_types = ['none', 'none', 'none', 'diabetes', 'other', 'hiv']
    ethnicities = ['han', 'han', 'han', 'han', 'uyghur', 'kazakh']

    for _ in range(n_records):
        records.append({
            'age': rng.choice([3, 8, 16, 25, 42, 58, 72]),
            'has_symptoms': rng.choice([0, 0, 1]),
            'has_tb': rng.choice([0, 0, 0, 1]),
            'bcg_vaccine': rng.choice([0, 1, 1]),
            'exposure_setting': rng.choice(settings),
            'contact_distance': rng.choice(distances),
            'ventilation': rng.randint(1, 5),
            'cumulative_exposure': rng.choice([0, 10, 25, 50, 80, 120]),
            'past_illness_type': rng.choice(illness_types),
            'idu_status': rng.choice([False, False, False, True]),
            'ethnicity': rng.choice(ethnicities),
            'pm10': rng.choice([80, 100, 120, 150, 200]),
            'humidity': rng.choice([10, 20, 30, 40, 50]),
            'origin_altitude': rng.choice([None, None, 500, 1500, 2500, 3500]),
            'months_since_migration': rng.choice([0, 6, 12, 24, 60]),
        })

    return records


# ==================== Morris 筛选分析器 ====================

class MorrisSensitivityAnalyzer:
    """莫里斯筛选法参数敏感性分析器（v2.0 重构版）。

    改进：
    - 不再复制 ScoringEngine 逻辑（消除 ~100 行代码重复）
    - 通过 ScoringEngine.apply_param_overrides() 注入参数
    - 评估的是真实评分引擎，而非副本

    指标：
    - μ* (mu_star): 基效应绝对值的均值，衡量参数总体影响
    - σ (sigma): 基效应的标准差，衡量非线性和交互效应

    参考文献：
    - Morris MD, Technometrics, 1991
    - Campolongo et al., Environ Modell Softw, 2007
    """

    def __init__(self, localizer, n_trajectories=15, n_levels=5):
        """
        参数：
            localizer: KaramayLocalizer 实例
            n_trajectories: int, Morris 轨迹数
            n_levels: int, 参数离散化级别数
        """
        self.localizer = localizer
        self.n_trajectories = n_trajectories
        self.n_levels = n_levels
        self._param_defs = MORRIS_PARAM_DEFS

    def run_analysis(self, records=None, n_sample_ids=500):
        """运行 Morris 筛选分析。

        参数：
            records: list[dict] or None, 接触者记录（None 时自动生成）
            n_sample_ids: int, 最大记录数

        返回：
            dict: 包含参数敏感性排名和诊断信息
        """
        if records is None:
            records = _generate_sample_records(n_records=min(n_sample_ids, 500))

        if len(records) > n_sample_ids:
            rng = random.Random(42)
            records = rng.sample(records, n_sample_ids)

        k = len(self._param_defs)
        all_elementary_effects = [[] for _ in range(k)]

        for traj_idx in range(self.n_trajectories):
            traj = self._generate_trajectory(k, seed_offset=traj_idx)
            param_sets = self._trajectory_to_params(traj)

            prev_output = None
            for step_idx, params_step in enumerate(param_sets):
                output = self._evaluate_with_engine(params_step, records)

                if prev_output is not None:
                    scale = 1.0 / max(1, self.n_levels - 1)
                    for j in range(k):
                        if traj[step_idx - 1][j] != traj[step_idx][j]:
                            ee = (output - prev_output) / (scale + 1e-9)
                            all_elementary_effects[j].append(ee)

                prev_output = output

        # 计算统计量
        result = []
        for i, (pdef, effects) in enumerate(
                zip(self._param_defs, all_elementary_effects)):
            if not effects:
                result.append({
                    'name': pdef[0], 'label': PARAM_LABELS.get(pdef[0], pdef[0]),
                    'mu': 0, 'mu_star': 0, 'sigma': 0,
                    'range': [pdef[1], pdef[2]], 'n_effects': 0,
                })
                continue

            if NUMPY_AVAILABLE:
                arr = np.array(effects)
                mu = float(np.mean(arr))
                mu_star = float(np.mean(np.abs(arr)))
                sigma = float(np.std(arr))
            else:
                mu = sum(effects) / len(effects)
                mu_star = sum(abs(e) for e in effects) / len(effects)
                sq = sum((e - mu) ** 2 for e in effects) / len(effects)
                sigma = math.sqrt(max(sq, 0))

            result.append({
                'name': pdef[0], 'label': PARAM_LABELS.get(pdef[0], pdef[0]),
                'mu': mu, 'mu_star': mu_star, 'sigma': sigma,
                'range': [pdef[1], pdef[2]], 'n_effects': len(effects),
            })

        result.sort(key=lambda x: x['mu_star'], reverse=True)

        return {
            'parameters': result,
            'n_trajectories': self.n_trajectories,
            'n_levels': self.n_levels,
            'n_parameters': k,
            'morris_note': 'mu_star:总体影响, sigma:非线性/交互效应; '
                           'v2.0: 通过 ScoringEngine 参数注入评估',
        }

    def _evaluate_with_engine(self, param_values, records):
        """使用 ScoringEngine（参数覆盖）评估一组参数下的平均风险。

        替代旧版 _compute_dp_with_params 的 ~100 行代码重复。
        """
        param_dict = {}
        for (pdef, value) in zip(self._param_defs, param_values):
            name = pdef[0]
            # 映射 Morris 参数名到 ScoringEngine 参数名
            sobol_name = MORRIS_TO_SOBOL_NAME.get(name, name)
            param_dict[sobol_name] = value

        engine = _create_engine_with_overrides(self.localizer, param_dict)

        total_risk = 0.0
        for rec in records:
            result = engine.compute_risk_score(rec)
            total_risk += result.get('disease_probability', 0.0)

        return total_risk / max(len(records), 1)

    def _generate_trajectory(self, k, seed_offset=0):
        """生成 Morris 轨迹（OAT 设计）。"""
        rng = random.Random(42 + seed_offset * 9973)
        base = [rng.randint(0, self.n_levels - 1) for _ in range(k)]
        perm = list(range(k))
        rng.shuffle(perm)
        delta = max(self.n_levels // 2, 1)

        trajectory = [list(base)]
        for j in perm:
            new = list(trajectory[-1])
            if rng.random() < 0.5:
                new[j] = min(new[j] + delta, self.n_levels - 1)
            else:
                new[j] = max(new[j] - delta, 0)
            trajectory.append(new)
        return trajectory

    def _trajectory_to_params(self, trajectory):
        """将轨迹转换为参数值。"""
        return [
            [self._level_to_value(p[1], p[2], traj_step[j])
             for j, p in enumerate(self._param_defs)]
            for traj_step in trajectory
        ]

    def _level_to_value(self, lo, hi, level):
        """将离散级别映射到参数值。"""
        if self.n_levels <= 1:
            return (lo + hi) / 2.0
        return lo + (level / (self.n_levels - 1)) * (hi - lo)


# ==================== Sobol 全局敏感性分析器 ====================

class SobolSensitivityAnalyzer:
    """Sobol 全局敏感性分析器（Saltelli 采样）。

    计算一阶指数 S_i 和总阶指数 S_Ti：
    - S_i: 参数 i 的主效应方差贡献占总方差的比例
    - S_Ti: 参数 i 的总效应（含交互），衡量该参数在所有交互中的总贡献

    若 S_Ti ≈ S_i：参数无显著交互
    若 S_Ti >> S_i：参数主要通过交互起作用

    文献支撑：
    - Sobol IM, Math Comput Simul, 2001
    - Saltelli A et al., Comput Phys Commun, 2010
    """

    def __init__(self, localizer, n_base=1024, random_seed=42):
        """
        参数：
            localizer: KaramayLocalizer 实例
            n_base: int, Saltelli 采样的基础样本量 N
                    总计算次数 = N × (2p + 2)，p=9 时为 20480（N=1024）
            random_seed: int, 随机种子
        """
        self.localizer = localizer
        self.n_base = n_base
        self.random_seed = random_seed
        self._param_defs = SOBOL_PARAM_DEFS
        self._p = len(self._param_defs)

    def run_analysis(self, records=None, n_sample_ids=500, n_bootstrap=100):
        """运行 Sobol 全局敏感性分析。

        参数：
            records: list[dict] or None, 接触者记录
            n_sample_ids: int, 最大记录数
            n_bootstrap: int, Bootstrap 重采样次数（用于置信区间）

        返回：
            dict: 包含 Sobol 指数和诊断信息
        """
        if not NUMPY_AVAILABLE:
            logger.warning("numpy 不可用，Sobol 分析需要 numpy")
            return self._empty_result("numpy 不可用")

        if records is None:
            records = _generate_sample_records(n_records=min(n_sample_ids, 500))

        if len(records) > n_sample_ids:
            rng = np.random.RandomState(self.random_seed)
            indices = rng.choice(len(records), size=n_sample_ids, replace=False)
            records = [records[i] for i in indices]

        rng = np.random.RandomState(self.random_seed)

        # Step 1: Saltelli 采样
        # 生成两个独立样本矩阵 A 和 B，各 N×p
        A = self._saltelli_sample(rng, self.n_base)
        B = self._saltelli_sample(rng, self.n_base)

        # 计算 A 和 B 所有行的模型输出
        f_A = self._evaluate_matrix(A, records)
        f_B = self._evaluate_matrix(B, records)

        # Step 2: 计算 Sobol 指数
        total_variance = float(np.var(np.concatenate([f_A, f_B])))

        if total_variance < 1e-15:
            return self._empty_result("总方差接近零，无法计算 Sobol 指数")

        S1 = np.zeros(self._p)
        ST = np.zeros(self._p)

        for i in range(self._p):
            # 构建 C_i 矩阵：A 的第 i 列替换为 B 的第 i 列
            C_i = A.copy()
            C_i[:, i] = B[:, i]

            f_Ci = self._evaluate_matrix(C_i, records)

            # Jansen (1999) 估计器（数值更稳定）
            # S_i = 1 - (1/(2N)) * sum(f_B - f_Ci)^2 / Var(Y)
            S1[i] = 1.0 - 0.5 * np.mean((f_B - f_Ci) ** 2) / total_variance

            # S_Ti = (1/(2N)) * sum(f_A - f_Ci)^2 / Var(Y)
            ST[i] = 0.5 * np.mean((f_A - f_Ci) ** 2) / total_variance

        # 确保 Sobol 指数在合理范围
        S1 = np.clip(S1, 0.0, 1.0)
        ST = np.clip(ST, 0.0, 1.0)

        # S_Ti 必须 ≥ S_i
        ST = np.maximum(ST, S1)

        # Step 3: Bootstrap 置信区间
        ci_first = np.zeros((self._p, 2))
        ci_total = np.zeros((self._p, 2))

        if n_bootstrap > 0:
            ci_first, ci_total = self._bootstrap_ci(
                A, B, f_A, f_B, records, n_bootstrap, rng)

        # Step 4: 构建结果
        params_result = []
        for i, pdef in enumerate(self._param_defs):
            params_result.append({
                'name': pdef[0],
                'label': PARAM_LABELS.get(pdef[0], pdef[0]),
                'S1': float(S1[i]),
                'ST': float(ST[i]),
                'S1_ci': [float(ci_first[i, 0]), float(ci_first[i, 1])],
                'ST_ci': [float(ci_total[i, 0]), float(ci_total[i, 1])],
                'range': [pdef[1], pdef[2]],
                'is_significant': ci_total[i, 0] > 0,
                'has_interaction': (ST[i] - S1[i]) > 0.05,
            })

        params_result.sort(key=lambda x: x['ST'], reverse=True)

        # 诊断
        sum_S1 = float(np.sum(S1))
        sum_ST = float(np.sum(ST))

        return {
            'parameters': params_result,
            'n_base': self.n_base,
            'n_parameters': self._p,
            'n_evaluations': self.n_base * (2 * self._p + 2),
            'total_variance': float(total_variance),
            'sum_S1': sum_S1,
            'sum_ST': sum_ST,
            'sobol_note': (
                f'S1: 主效应, ST: 总效应(含交互); '
                f'ΣS1={sum_S1:.3f} (≤1.0 表示交互弱); '
                f'ST-S1>0.05 表示显著交互; '
                f'CI不跨0 表示参数显著'
            ),
        }

    def _saltelli_sample(self, rng, n_samples):
        """Saltelli 采样：在 [0, 1]^p 超立方体中生成准随机样本。

        使用 Sobol 序列近似（通过随机排列模拟）。
        实际部署时建议使用 SALib 的 saltelli.sample() 获取真正的 Sobol 序列。
        """
        # 使用 Latin Hypercube 近似（比纯随机更均匀）
        p = self._p
        samples = np.zeros((n_samples, p))

        for j in range(p):
            # 分层采样
            perm = rng.permutation(n_samples)
            samples[:, j] = (perm + rng.random_sample(n_samples)) / n_samples

        # 缩放到参数边界
        for j, pdef in enumerate(self._param_defs):
            lo, hi = pdef[1], pdef[2]
            samples[:, j] = lo + samples[:, j] * (hi - lo)

        return samples

    def _evaluate_matrix(self, matrix, records):
        """评估参数矩阵中每一行的模型输出。

        参数：
            matrix: np.ndarray, shape (N, p)
            records: list[dict], 接触者记录

        返回：
            np.ndarray: shape (N,), 每行的平均疾病概率
        """
        n = matrix.shape[0]
        outputs = np.zeros(n)

        for i in range(n):
            param_dict = {}
            for j, pdef in enumerate(self._param_defs):
                param_dict[pdef[0]] = float(matrix[i, j])

            engine = _create_engine_with_overrides(self.localizer, param_dict)

            total_risk = 0.0
            for rec in records:
                result = engine.compute_risk_score(rec)
                total_risk += result.get('disease_probability', 0.0)

            outputs[i] = total_risk / max(len(records), 1)

        return outputs

    def _bootstrap_ci(self, A, B, f_A, f_B, records, n_bootstrap, rng):
        """Bootstrap 计算 Sobol 指数的置信区间。

        通过重采样 A 和 B 的行，重新计算 Sobol 指数。
        """
        N = self.n_base
        p = self._p
        S1_bootstrap = np.zeros((n_bootstrap, p))
        ST_bootstrap = np.zeros((n_bootstrap, p))

        for k in range(n_bootstrap):
            # Bootstrap 重采样索引
            indices = rng.randint(0, N, size=N)

            f_A_boot = f_A[indices]
            f_B_boot = f_B[indices]

            total_var = float(np.var(np.concatenate([f_A_boot, f_B_boot])))
            if total_var < 1e-15:
                continue

            for i in range(p):
                C_i = A[indices].copy()
                C_i[:, i] = B[indices, i]
                f_Ci = self._evaluate_matrix(C_i, records)

                S1_bootstrap[k, i] = 1.0 - 0.5 * np.mean(
                    (f_B_boot - f_Ci) ** 2) / total_var
                ST_bootstrap[k, i] = 0.5 * np.mean(
                    (f_A_boot - f_Ci) ** 2) / total_var

        # 2.5% 和 97.5% 分位数
        ci_first = np.zeros((p, 2))
        ci_total = np.zeros((p, 2))

        for i in range(p):
            ci_first[i, 0] = np.percentile(
                np.clip(S1_bootstrap[:, i], 0, 1), 2.5)
            ci_first[i, 1] = np.percentile(
                np.clip(S1_bootstrap[:, i], 0, 1), 97.5)
            ci_total[i, 0] = np.percentile(
                np.clip(ST_bootstrap[:, i], 0, 1), 2.5)
            ci_total[i, 1] = np.percentile(
                np.clip(ST_bootstrap[:, i], 0, 1), 97.5)

        return ci_first, ci_total

    def _empty_result(self, reason=""):
        """返回空结果。"""
        return {
            'parameters': [],
            'n_base': self.n_base,
            'n_parameters': self._p,
            'n_evaluations': 0,
            'total_variance': 0.0,
            'sum_S1': 0.0,
            'sum_ST': 0.0,
            'sobol_note': f'Sobol 分析未执行: {reason}',
        }

    def run_from_morris_top(self, morris_result, records=None,
                            top_k=5, n_base=512):
        """Morris 快速筛选 + Sobol 精细分析。

        先用 Morris 筛选出 top_k 个重要参数，再对它们做 Sobol 分析。

        参数：
            morris_result: dict, Morris 分析结果
            records: list[dict], 接触者记录
            top_k: int, 选取前 k 个重要参数
            n_base: int, Sobol 基础样本量

        返回：
            dict: Sobol 分析结果（仅包含 top_k 参数）
        """
        top_params = [p['name'] for p in morris_result['parameters'][:top_k]]

        # 筛选参数定义
        filtered_defs = [d for d in SOBOL_PARAM_DEFS if d[0] in top_params]

        # 创建临时分析器
        analyzer = SobolSensitivityAnalyzer(
            self.localizer, n_base=n_base, random_seed=self.random_seed)
        analyzer._param_defs = filtered_defs
        analyzer._p = len(filtered_defs)

        return analyzer.run_analysis(records=records)