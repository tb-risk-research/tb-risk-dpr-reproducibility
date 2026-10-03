#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可复现的合成数据生成过程（DGP）文档与分布对照校验。

改进三-第一步：把现有合成方法规范化为**可复现的 DGP 文档**，直接回应评审
"可参考性不强"的批评。本模块不做新的数据生成，而是把
``ScreeningDataSimulator`` 的合成口径逐字段文档化，并提供对照校验：

1. ``FIELD_SPECS`` / ``DGP_PARAMETERS`` / ``TRANSMISSION_SPEC``
   —— 结构化 DGP 规范（每个字段的分布与参数、全局参数、标签生成机制）；
2. ``LITERATURE_REFERENCES`` —— 参数来源与文献/官方公报引用；
3. ``build_dgp_report()`` —— 生成数据 + 分布对照校验，汇总为完整文档 dict；
4. ``render_dgp_document()`` —— 渲染为可发布的 Markdown 文档；
5. ``validate_synthetic_vs_spec()`` / ``validate_prevalence()`` /
   ``validate_against_literature()`` / ``reproducibility_check()``
   —— 合成 vs 真实分布对照校验 + 可复现性校验。

对照校验（"合成数据与真实数据分布的对照校验"）：
- 关键变量分布：分类字段观测比例 vs 规范权重（容差内比对）；
  连续字段均值/取值范围 vs 规范；
- 患病率/检出率：观测检出率 vs 官方 121/10 万；
- 接触者感染率等流行病学量：与文献值比对；当前合成数据（任务 A 口径）不可
  直接计算的项（需任务 B/C 感染标签或网络层 DGP）显式标注 ``not_applicable``，
  避免过度声称。

设计原则：
- 单一真值源：字段参数从 ``tb_risk.constants`` 与
  ``ScreeningDataSimulator`` 常量导入，本模块只做文档化 + 校验，不重新定义
  口径（防止规范与实际实现漂移）；
- 诚实标注：合成数据当前只产出任务 A 标签（``is_confirmed``，活动性结核
  确诊）；任务 B（感染后进展）与任务 C（网络传播）标签需由公开数据
  （见 ``validation.public_datasets``）或网络层 DGP 补充。
"""

import logging

from tb_risk.constants import (
    HIGH_ALTITUDE_THRESHOLD_M,
    ETHNICITY_WEIGHTS,
    OCCUPATION_WEIGHTS,
    DISTRICT_POPULATION_WEIGHTS,
    MONTHLY_PM10_BASELINE,
    MONTHLY_HUMIDITY_BASELINE,
    PREVALENCE_CALIBERS,
    PREVALENCE_CALIBER_MIXING_WARNING,
)

LOGGER = logging.getLogger("tb_risk.validation.dgp")


# ==============================================================================
# 全局参数与容差
# ==============================================================================

# 默认随机种子（与 ScreeningDataSimulator 默认一致，保证同种子完全可复现）
DEFAULT_RANDOM_STATE = 42

# 分类字段绝对比例容差（对照校验）
CATEGORICAL_TOLERANCE = 0.05
# 连续字段均值相对容差（对照校验）
CONTINUOUS_MEAN_TOLERANCE = 0.20
# 检出率容差（/10 万）——覆盖小样本二项噪声（48683 人期望 59 例，σ≈8）
PREVALENCE_TOLERANCE_PER_100K = 60.0

DGP_PARAMETERS = {
    'population': '2024 年克拉玛依市重点人群结核病筛查',
    'target_total_screened': 48683,
    'target_confirmed': 59,
    'target_rate_per_100k': 121.0,
    'random_seed': DEFAULT_RANDOM_STATE,
    'rng_backend': 'random.Random(seed)',
    'generator': 'tb_risk.scoring.simulator.ScreeningDataSimulator',
    'source': '克拉玛依市卫健委 2024 年筛查公报（48,683 人筛查，59 例确诊，检出率约 121/10 万）',
}


# ==============================================================================
# 字段分布规范（逐字段：分布 / 参数 / 来源）—— 与 simulator 实现逐字段对齐
# ==============================================================================

def _annual_avg(baseline: dict) -> float:
    """月度基线年均值（月份均匀抽样时的期望均值）。"""
    return sum(baseline.values()) / max(len(baseline), 1)


FIELD_SPECS = [
    # ---- 人口学 ----
    {
        'field': 'district', 'role': '人口学', 'kind': 'categorical',
        'distribution': '分类加权抽样（weighted choice）',
        'parameters': dict(DISTRICT_POPULATION_WEIGHTS),
        'target_key': 'DISTRICT_POPULATION_WEIGHTS',
        'source': '克拉玛依市行政区划人口结构（官方公报）',
    },
    {
        'field': 'ethnicity', 'role': '人口学', 'kind': 'categorical',
        'distribution': '分类加权抽样（weighted choice）',
        'parameters': dict(ETHNICITY_WEIGHTS),
        'target_key': 'ETHNICITY_WEIGHTS',
        'source': '新疆/克拉玛依民族构成（统计公报）',
    },
    {
        'field': 'occupation', 'role': '人口学', 'kind': 'categorical',
        'distribution': '分类加权抽样（weighted choice）',
        'parameters': dict(OCCUPATION_WEIGHTS),
        'target_key': 'OCCUPATION_WEIGHTS',
        'source': '克拉玛依油田产业人口结构（官方公报）',
    },
    {
        'field': 'age', 'role': '人口学', 'kind': 'continuous',
        'distribution': '分段均匀（piecewise uniform）',
        'parameters': {'segments': [
            {'range': '(0,5)', 'weight': 0.05},
            {'range': '(5,15)', 'weight': 0.05},
            {'range': '(15,35)', 'weight': 0.45},
            {'range': '(35,65)', 'weight': 0.30},
            {'range': '(66,90)', 'weight': 0.15},
        ]},
        'expected_mean': 38.6, 'range': (0, 100),
        'source': '油田青年人口为主，符合克拉玛依人口结构',
    },
    {
        'field': 'gender', 'role': '人口学', 'kind': 'categorical',
        'distribution': 'Bernoulli(p=0.55)',
        'parameters': {'male': 0.55, 'female': 0.45},
        'source': '油田就业男性占比偏高',
    },
    # ---- 接触史 / 暴露 ----
    {
        'field': 'exposure_setting', 'role': '接触史', 'kind': 'categorical',
        'distribution': '分类加权抽样（weighted choice）',
        'parameters': {
            'general': 0.35, 'closed': 0.20, 'crowded': 0.15,
            'outdoor': 0.15, 'oilfield_camp': 0.15,
        },
        'target_key': 'EXPOSURE_SETTINGS/WEIGHTS',
        'source': '重点人群暴露场景构成（专家校准）',
    },
    {
        'field': 'contact_distance', 'role': '接触史', 'kind': 'categorical',
        'distribution': '分类加权抽样（weighted choice）',
        'parameters': {
            'very_close': 0.10, 'close': 0.30, 'medium': 0.35,
            'far': 0.15, 'distant': 0.10,
        },
        'target_key': 'CONTACT_DISTANCES/DISTANCE_WEIGHTS',
        'source': '接触距离分布（家庭/社会接触调查经验值）',
    },
    {
        'field': 'single_duration', 'role': '接触史', 'kind': 'continuous',
        'distribution': '三角分布 Triangular(5, 480, mode=60)（分钟/次）',
        'parameters': {'low': 5, 'high': 480, 'mode': 60},
        'expected_mean': (5 + 480 + 60) / 3.0, 'range': (5, 480),
        'source': '单次接触时长经验分布（多数为短时接触）',
    },
    {
        'field': 'freq_density', 'role': '接触史', 'kind': 'continuous',
        'distribution': '三角分布 Triangular(0, 30, mode)（次/周）；'
                        '油田作业人员 mode=21，其余 mode=14',
        'parameters': {'low': 0, 'high': 30, 'mode_oilfield': 21, 'mode_general': 14},
        'range': (0, 30),
        'source': '接触频次密度（油田高强度作业环境）',
    },
    {
        'field': 'time_span', 'role': '接触史', 'kind': 'continuous',
        'distribution': '三角分布 Triangular(1, 12, mode=4)（月）',
        'parameters': {'low': 1, 'high': 12, 'mode': 4},
        'expected_mean': (1 + 12 + 4) / 3.0, 'range': (1, 12),
        'source': '接触时间跨度（多数为短期共事/共居）',
    },
    {
        'field': 'cumulative_exposure', 'role': '接触史', 'kind': 'continuous',
        'distribution': '派生量：单次时长(h)×频次(次/周)×周数（time_span×4.33）',
        'parameters': {'formula': 'single_duration/60 × freq_density × time_span × 4.33'},
        'range': (0, None),
        'source': '累积暴露强度（接触评估核心量）',
    },
    # ---- 临床 ----
    {
        'field': 'has_symptoms', 'role': '临床', 'kind': 'categorical',
        'distribution': 'Bernoulli(p=0.08)',
        'parameters': {0: 0.92, 1: 0.08},
        'source': '重点人群有症状比例（筛查现场经验值）',
    },
    {
        'field': 'bcg_vaccine', 'role': '临床', 'kind': 'categorical',
        'distribution': 'Bernoulli(p=0.85)',
        'parameters': {0: 0.15, 1: 0.85},
        'source': '卡介苗接种率（中国新生儿常规接种，>85%）',
    },
    {
        'field': 'has_tb', 'role': '临床', 'kind': 'categorical',
        'distribution': 'Bernoulli(p=0.02)（既往结核病史）',
        'parameters': {0: 0.98, 1: 0.02},
        'source': '既往结核病史比例（流行病学经验值）',
    },
    {
        'field': 'past_illness', 'role': '临床', 'kind': 'categorical',
        'distribution': 'Bernoulli(p=0.12)；类型 ∈ {HIV, diabetes, '
                        'immunosuppressants, other}',
        'parameters': {0: 0.88, 1: 0.12},
        'source': '合并症/免疫抑制风险人群比例',
    },
    {
        'field': 'idu_status', 'role': '临床', 'kind': 'categorical',
        'distribution': 'Bernoulli(p=0.01)（静脉吸毒）',
        'parameters': {0: 0.99, 1: 0.01},
        'source': '高危行为人群比例（低流行假设）',
    },
    {
        'field': 'ventilation', 'role': '临床', 'kind': 'continuous',
        'distribution': '三角分布 Triangular(1, 5, mode=3)（通风评分）',
        'parameters': {'low': 1, 'high': 5, 'mode': 3},
        'expected_mean': (1 + 5 + 3) / 3.0, 'range': (1, 5),
        'source': '通风条件评分（1=差, 5=好）',
    },
    # ---- 环境 ----
    {
        'field': 'origin_altitude', 'role': '环境', 'kind': 'continuous',
        'distribution': '15% 概率存在：Uniform(600, 3500)（米）',
        'parameters': {'present_rate': 0.15, 'low': 600, 'high': 3500},
        'present_rate': 0.15, 'range': (600, 3500),
        'source': '高海拔迁入人群比例（>1500m 视为高海拔风险，'
                  f'阈值 {HIGH_ALTITUDE_THRESHOLD_M}m）',
    },
    {
        'field': 'months_since_migration', 'role': '环境', 'kind': 'continuous',
        'distribution': '无迁入为 0；迁入人群 Uniform(0, 120)（月）',
        'parameters': {'low': 0, 'high': 120, 'migration_rate': 0.15},
        'expected_mean': 0.15 * 60.0, 'range': (0, 120),
        'source': '迁入后时长（高海拔迁入 24 月内风险叠加）',
    },
    {
        'field': 'pm10', 'role': '环境', 'kind': 'continuous',
        'distribution': 'Normal(月度基线, 0.3×基线)（μg/m³），下界 0',
        'parameters': {'baseline_by_month': dict(MONTHLY_PM10_BASELINE),
                       'cv': 0.3},
        'expected_mean': _annual_avg(MONTHLY_PM10_BASELINE), 'range': (0, None),
        'source': '克拉玛依月度 PM10 气候基线（tb_risk.constants 单一真值源）',
    },
    {
        'field': 'humidity', 'role': '环境', 'kind': 'continuous',
        'distribution': 'Normal(月度基线, σ=8)（%），截断 [5, 95]',
        'parameters': {'baseline_by_month': dict(MONTHLY_HUMIDITY_BASELINE),
                       'sigma': 8},
        'expected_mean': _annual_avg(MONTHLY_HUMIDITY_BASELINE), 'range': (5, 95),
        'source': '克拉玛依月度湿度气候基线（tb_risk.constants 单一真值源）',
    },
]


def field_specs_by_name() -> dict:
    """字段名 → 分布规范 映射（便于程序化查询）。"""
    return {s['field']: s for s in FIELD_SPECS}


# ==============================================================================
# 标签生成机制（SEIR 传播链）规范
# ==============================================================================

TRANSMISSION_SPEC = {
    'mechanism': '独立 SEIR 传播链（与评分模型信息源解耦，打破循环论证）',
    'seed_selection': {
        'description': '仅基于人口统计学因素选择种子病例（年龄 / 既往 TB 史），'
                       '不依赖症状/暴露等评分模型特征',
        'age_risk': {
            '0-5岁': 0.02, '5-15岁': 0.01, '15-35岁': 0.015,
            '35-55岁': 0.02, '55-100岁': 0.03,
        },
        'prior_tb_multiplier': 2.0,
        'sampling_control': '当期望确诊数超过目标 3 倍时随机下采样种子，'
                            '使最终检出率贴近 121/10 万',
        'source': 'TB 年龄别发病风险（幼儿/老年免疫风险更高；WHO 2024）',
    },
    'reproduction_number': {
        'R_effective': 1.0,
        'range': '0.8–1.2',
        'contact_count_distribution': 'Gaussian(R_eff × gen_factor, σ=0.5)',
        'generation_decay': 0.3,
        'generation_factor': 'max(0.1, 1 − 0.3 × (代际−1))',
        'source': 'TB 家庭/社区传播二代病例数文献估计'
                  '（Fox et al. 2013; WHO 接触者追踪指南）',
    },
    'generations': {
        'max': 3,
        'note': '种子→家庭接触者→社会接触者→社区接触，传播概率随代际衰减',
    },
    'contact_network_geometry': {
        'contact_index_sampling': 'seed_idx + N(0, spread)，spread = max(5, n//100)，'
                                  '对 n 取环（模拟局部接触网络）',
        'avoid_repeat': '已确诊个体不重复入选（max_attempts=20）',
    },
    'target_detection_rate': '121/10 万',
    'label_field': 'is_confirmed（任务 A：活动性结核确诊，二元）',
}


# ==============================================================================
# 标签机制版本登记表（P3：DGP v3 统一——代码库全部标签机制的单一真值源索引）
# ==============================================================================
# 背景：宿主通路修复后代码库曾并存两套标签机制（场景生成器 v3 vs
# layer_ablation 旧 top-k），消融与部署场景的标签口径漂移。本登记表把
# 代码库全部标签生成机制显式版本化并互相引用真值源；checkpoint 元数据
# 的 label_generation 字段（P4）据此校验加载代际。

LABEL_MECHANISM_VERSIONS = [
    {
        'id': 'taskA-seir-chain-v1',
        'generator': 'tb_risk.scoring.simulator.ScreeningDataSimulator',
        'label_mechanism': '独立 SEIR 传播链（人口学种子选择 + R_eff 代际衰减，'
                           '详见 TRANSMISSION_SPEC）',
        'source_of_truth': 'validation/dgp.py TRANSMISSION_SPEC',
    },
    {
        'id': 'v3-host-pathway',
        'generator': 'data/evaluate_ensemble.py generate_scenarios'
                     '(label_mode="mechanistic")',
        'label_mechanism': 'p = 1 - exp(-infectivity × intensity × '
                           'host_susceptibility_multiplier)——宿主通路 v3：'
                           '文献校准的年龄U型/症状/共病RR/BCG衰减乘数',
        'source_of_truth': 'validation/host_susceptibility.py '
                           'HOST_SUSCEPTIBILITY_SPEC（宿主乘数单一真值源）',
    },
    {
        'id': 'clinical-rule-v1',
        'generator': 'data/evaluate_ensemble.py generate_scenarios'
                     '(label_mode="clinical")',
        'label_mechanism': '线性打分 + sigmoid 临床规则标签（与机制标签错开'
                           '训练的多样性工程）',
        'source_of_truth': 'data/evaluate_ensemble.py _clinical_rule_prob',
    },
    {
        'id': 'network-ablation-topk-v3',
        'generator': 'validation/layer_ablation.py build_synthetic_network',
        'label_mechanism': '综合风险（个体可观测 + 簇隐藏）× 宿主易感性乘数，'
                           '排序取 top-k 为病例（2026-08-24 起与场景 DGP 同源）',
        'source_of_truth': 'validation/host_susceptibility.py（宿主乘数与 '
                           'v3-host-pathway 共用同一真值源）',
    },
]


# ==============================================================================
# 文献参考表（参数来源 + 可计算性标注）
# ==============================================================================

LITERATURE_REFERENCES = [
    {
        'id': 'krmy_2024',
        'quantity': '活动性结核检出率（重点人群筛查）',
        'value': '121/10 万（48,683 人筛查 59 例确诊）',
        'range': '121/10 万',
        'source': '克拉玛依市卫健委 2024 年重点人群筛查公报',
        'computable_from_base_dgp': True,
    },
    {
        'id': 'who_china_2023',
        'quantity': '中国结核病发病率',
        'value': '52/10 万（2023 年估算）',
        'range': '—',
        'source': 'WHO Global TB Report 2023',
        'computable_from_base_dgp': False,
        'note': '国家层面发病率，用于预训练人群校准（transfer_learning），'
                '非合成数据直接可计算量',
    },
    {
        'id': 'hh_infection',
        'quantity': '家庭接触者结核感染率（IGRA 阳性）',
        'value': '约 30%',
        'range': '20%–50%',
        'source': 'ERASE-TB 队列（坦桑尼亚/莫桑比克/津巴布韦 1905 名家庭接触者）'
                  '；Fox et al. 2013 综述',
        'computable_from_base_dgp': False,
        'note': '任务 B 口径标签（感染状态），当前合成数据（任务 A）无法直接计算；'
                '由公开数据（public_datasets 接触者层）或网络层 DGP 补充后校验',
    },
    {
        'id': 'school_latent_2024',
        'quantity': '学校密切接触者潜伏感染率（河南/广州）',
        'value': '见原文（6,893 名密切接触者 ML 建模）',
        'range': '—',
        'source': '中国疾控周报（China CDC Weekly）2024 年学校密切接触者研究',
        'computable_from_base_dgp': False,
        'note': '任务 B 口径参考队列；需联系作者获取或复现其方法',
    },
    {
        'id': 'ltbi_reactivation',
        'quantity': '潜伏感染再激活率',
        'value': '终生约 10%，前 2 年集中（约一半）',
        'range': '—',
        'source': 'Vynnycky & Fine 1997; Houben & Dodd 2016 (BMC Med)',
        'computable_from_base_dgp': False,
        'note': '任务 B 随访终点参数（时间-事件），由 SEIR 层标定',
    },
    {
        'id': 'r_eff',
        'quantity': '有效繁殖数 R_eff',
        'value': '0.8–1.2',
        'range': '0.8–1.2',
        'source': 'TB 传播动力学文献估计（家庭/社区接触）',
        'computable_from_base_dgp': False,
        'note': '传播链过程量（任务 C 口径）；需网络层过程数据或敏感性分析校验',
    },
]


# ==============================================================================
# 分布对照校验
# ==============================================================================

def _category_proportions(records, field):
    """统计字段的观测比例分布。"""
    n = len(records)
    counts = {}
    for rec in records:
        key = rec.get(field)
        counts[key] = counts.get(key, 0) + 1
    return {k: v / max(n, 1) for k, v in counts.items()}


def _mean(values):
    return sum(values) / max(len(values), 1)


def validate_synthetic_vs_spec(records):
    """关键变量分布对照校验：观测分布 vs DGP 规范权重（容差内比对）。

    参数：
        records (list[dict]): 合成数据集

    返回：
        list[dict]: 逐字段校验结果（observed / target / max_delta / pass）
    """
    n = len(records)
    if n == 0:
        raise ValueError("records 为空，无法校验")

    results = []
    for spec in FIELD_SPECS:
        field = spec['field']
        entry = {'field': field, 'kind': spec.get('kind'),
                 'distribution': spec['distribution']}

        if spec.get('kind') == 'categorical':
            target = spec['parameters']
            observed = _category_proportions(records, field)
            max_delta = 0.0
            discrepancies = []
            for key, weight in target.items():
                obs = observed.get(key, 0.0)
                delta = abs(obs - weight)
                max_delta = max(max_delta, delta)
                if delta > CATEGORICAL_TOLERANCE:
                    discrepancies.append({'category': key, 'target': weight,
                                          'observed': round(obs, 4),
                                          'delta': round(delta, 4)})
            entry.update({
                'n_observed': n,
                'max_abs_delta': round(max_delta, 4),
                'tolerance': CATEGORICAL_TOLERANCE,
                'pass': not discrepancies,
                'discrepancies': discrepancies,
            })

        else:  # continuous
            present_rate = spec.get('present_rate')
            values = [float(rec[field]) for rec in records
                      if rec.get(field) is not None]
            checks = []

            # 存在率（允许部分样本缺失的字段，如 origin_altitude）
            if present_rate is not None:
                rate = len(values) / max(n, 1)
                ok_rate = abs(rate - present_rate) <= CATEGORICAL_TOLERANCE
                checks.append({'check': 'present_rate', 'target': present_rate,
                               'observed': round(rate, 4), 'pass': ok_rate})
                if not ok_rate:
                    LOGGER.warning('DGP 校验失败: %s 存在率 %.3f vs 目标 %.3f',
                                   field, rate, present_rate)

            if values:
                lo, hi = spec.get('range', (None, None))
                vmin, vmax = min(values), max(values)
                ok_range = True
                if lo is not None and vmin < lo - CATEGORICAL_TOLERANCE:
                    ok_range = False
                if hi is not None and vmax > hi + CATEGORICAL_TOLERANCE:
                    ok_range = False
                checks.append({'check': 'range', 'target': (lo, hi),
                               'observed': (round(vmin, 1), round(vmax, 1)),
                               'pass': ok_range})

                if spec.get('expected_mean') is not None:
                    exp_mean = spec['expected_mean']
                    obs_mean = _mean(values)
                    rel_err = abs(obs_mean - exp_mean) / max(abs(exp_mean), 1e-9)
                    ok_mean = rel_err <= CONTINUOUS_MEAN_TOLERANCE
                    checks.append({'check': 'mean',
                                   'target': round(exp_mean, 1),
                                   'observed': round(obs_mean, 1),
                                   'relative_error': round(rel_err, 4),
                                   'tolerance': CONTINUOUS_MEAN_TOLERANCE,
                                   'pass': ok_mean})
            entry.update({'checks': checks, 'pass': all(c['pass'] for c in checks)})

        results.append(entry)

    return results


def validate_prevalence(records, target_per_100k=None):
    """患病率/检出率对照校验：观测检出率 vs 官方 121/10 万。

    参数：
        records (list[dict]): 合成数据集（须含 is_confirmed 标签）
        target_per_100k (float|None): 目标检出率，None 用官方 121/10 万

    返回：
        dict: 检出率校验结果
    """
    if target_per_100k is None:
        target_per_100k = float(DGP_PARAMETERS['target_rate_per_100k'])

    n = len(records)
    if n == 0:
        raise ValueError("records 为空，无法校验")

    confirmed = sum(1 for r in records if r.get('is_confirmed', 0))
    observed = confirmed / n * 1e5
    delta = abs(observed - target_per_100k)

    return {
        'n': n,
        'n_confirmed': confirmed,
        'observed_per_100k': round(observed, 1),
        'target_per_100k': target_per_100k,
        'tolerance_per_100k': PREVALENCE_TOLERANCE_PER_100K,
        'abs_delta_per_100k': round(delta, 1),
        'pass': delta <= PREVALENCE_TOLERANCE_PER_100K,
        'source': DGP_PARAMETERS['source'],
    }


def validate_against_literature(records):
    """文献参考量对照：可计算的量做比对，不可计算的量明确标注。

    参数：
        records (list[dict]): 合成数据集

    返回：
        dict: {'entries': list[dict], 'prevalence_check': dict}
    """
    prevalence = validate_prevalence(records)
    entries = []
    for ref in LITERATURE_REFERENCES:
        if ref.get('computable_from_base_dgp'):
            entries.append({
                'id': ref['id'],
                'quantity': ref['quantity'],
                'target': ref['value'],
                'observed': f"{prevalence['observed_per_100k']}/10 万",
                'status': 'pass' if prevalence['pass'] else 'fail',
                'source': ref['source'],
            })
        else:
            entries.append({
                'id': ref['id'],
                'quantity': ref['quantity'],
                'target': ref['value'],
                'source': ref['source'],
                'status': 'not_applicable',
                'note': ref.get(
                    'note',
                    '当前合成数据（任务 A 口径）无法直接计算，'
                    '需任务 B/C 标签或网络层 DGP 补充'),
            })
    return {'entries': entries, 'prevalence_check': prevalence}


def reproducibility_check(n_samples=500, random_state=DEFAULT_RANDOM_STATE):
    """可复现性校验：同种子两次生成逐字段完全一致。

    参数：
        n_samples (int): 校验样本量
        random_state (int): 随机种子

    返回：
        dict: {'identical': bool, 'n_samples': int, 'random_state': int,
               'generator': str}
    """
    from ..scoring.simulator import ScreeningDataSimulator

    a = ScreeningDataSimulator(random_state=random_state).generate_dataset(
        n_samples=n_samples, include_labels=True)
    b = ScreeningDataSimulator(random_state=random_state).generate_dataset(
        n_samples=n_samples, include_labels=True)
    return {
        'identical': a == b,
        'n_samples': int(n_samples),
        'random_state': int(random_state),
        'generator': 'ScreeningDataSimulator(random.Random(seed))',
    }


# ==============================================================================
# DGP 文档构建与渲染
# ==============================================================================

def build_dgp_report(n_samples=None, random_state=DEFAULT_RANDOM_STATE,
                     records=None):
    """构建完整 DGP 文档（结构化 dict：规范 + 生成元信息 + 对照校验）。

    参数：
        n_samples (int|None): 生成样本数；None 使用官方总筛查量 48,683
        random_state (int): 随机种子（文档记录并用于生成）
        records (list[dict]|None): 预生成的合成数据；提供则跳过重新生成

    返回：
        dict: 完整 DGP 文档
    """
    from ..scoring.simulator import ScreeningDataSimulator

    if records is None:
        sim = ScreeningDataSimulator(random_state=random_state)
        records = sim.generate_dataset(n_samples=n_samples, include_labels=True)
    n = len(records)

    report = {
        'document_title': '合成数据生成过程（DGP）可复现文档',
        'dgp_parameters': {**DGP_PARAMETERS, 'random_seed': int(random_state)},
        'field_specs': FIELD_SPECS,
        'transmission_spec': TRANSMISSION_SPEC,
        'label_mechanism_versions': LABEL_MECHANISM_VERSIONS,
        'literature_references': LITERATURE_REFERENCES,
        'generation': {
            'n_samples': n,
            'random_seed': int(random_state),
            'generator': DGP_PARAMETERS['generator'],
            'rng_backend': DGP_PARAMETERS['rng_backend'],
        },
        'validation': {
            'distributions': validate_synthetic_vs_spec(records),
            'prevalence': validate_prevalence(records),
            'literature': validate_against_literature(records),
        },
    }
    report['overall_pass'] = (
        all(v['pass'] for v in report['validation']['distributions'])
        and report['validation']['prevalence']['pass']
    )
    return report


def _fmt_field_table(specs) -> list:
    """把字段规范渲染为 Markdown 表格行。"""
    header = ['| 字段 | 类型 | 分布与参数 | 参数来源 |',
              '| --- | --- | --- | --- |']
    rows = list(header)
    for s in specs:
        params = s.get('parameters', {})
        if isinstance(params, dict) and 'baseline_by_month' in params:
            params = {k: v for k, v in params.items()
                      if k != 'baseline_by_month'}
        if s.get('kind') == 'categorical':
            dist = f"{s['distribution']}: " + '; '.join(
                f"{k}={v}" for k, v in s['parameters'].items())
        else:
            dist = s['distribution']
            extra = []
            if 'expected_mean' in s:
                extra.append(f"期望均值={s['expected_mean']}")
            if 'range' in s and s['range'] != (0, None):
                extra.append(f"范围={s['range']}")
            if extra:
                dist = f"{dist}（{'；'.join(extra)}）"
        rows.append(
            f"| `{s['field']}` | {s.get('role', '')} | {dist} | "
            f"{s.get('source', '')} |")
    return rows


def render_dgp_document(report=None, n_samples=None,
                        random_state=DEFAULT_RANDOM_STATE, records=None):
    """把 DGP 文档渲染为可发布的 Markdown 文本。

    参数：
        report (dict|None): build_dgp_report 的结果；None 则先构建
        n_samples / random_state / records: 构建参数（report 为 None 时生效）

    返回：
        str: Markdown 文档
    """
    if report is None:
        report = build_dgp_report(n_samples=n_samples,
                                  random_state=random_state, records=records)

    lines = []
    ap = report['dgp_parameters']
    lines.append(f"# {report['document_title']}")
    lines.append('')
    lines.append('> 目标：把合成数据生成过程规范化为可复现的文档，'
                 '回应评审"可参考性不强"的批评。')
    lines.append('> 原则：字段参数以 `tb_risk.constants` 与 '
                 '`ScreeningDataSimulator` 为单一真值源，本文档只做'
                 '文档化 + 校验，不重新定义口径。')
    lines.append('')

    # 1. 全局参数
    lines.append('## 1. 全局参数')
    lines.append('')
    lines.append('| 参数 | 值 |')
    lines.append('| --- | --- |')
    lines.append(f"| 人群 | {ap['population']} |")
    lines.append(f"| 总筛查量 | {ap['target_total_screened']} |")
    lines.append(f"| 确诊数 | {ap['target_confirmed']} |")
    lines.append(f"| 目标检出率 | {ap['target_rate_per_100k']}/10 万 |")
    lines.append(f"| 随机种子 | `{ap['random_seed']}` |")
    lines.append(f"| 随机后端 | `{ap['rng_backend']}` |")
    lines.append(f"| 生成器 | `{ap['generator']}` |")
    lines.append(f"| 参数来源 | {ap['source']} |")
    lines.append('')

    # 1.1 阳性率双口径标注（问题4：与训练报告、GUI 界面三处统一）
    lines.append('### 阳性率口径（双口径标注，不可混用）')
    lines.append('')
    lines.append('| 口径 | 阳性率 | 适用场景 | 来源 |')
    lines.append('| --- | --- | --- | --- |')
    for caliber in PREVALENCE_CALIBERS.values():
        lines.append(f"| {caliber['label']} | {caliber['display']} "
                     f"| PPV 换算与筛查决策 | {caliber['source']} |")
    lines.append('')
    lines.append(f'> **口径警示**：{PREVALENCE_CALIBER_MIXING_WARNING}')
    lines.append('')

    # 2. 字段分布规范
    lines.append('## 2. 字段分布规范（逐字段：分布 / 参数 / 来源）')
    lines.append('')
    lines.extend(_fmt_field_table(report['field_specs']))
    lines.append('')

    # 3. 标签生成机制
    ts = report['transmission_spec']
    lines.append('## 3. 标签生成机制（SEIR 传播链）')
    lines.append('')
    lines.append(f"- **机制**：{ts['mechanism']}。")
    ss = ts['seed_selection']
    lines.append(f"- **种子选择**：{ss['description']}。")
    lines.append("  - 年龄别种子风险：" + '；'.join(
        f"{k}→{v}" for k, v in ss['age_risk'].items()))
    lines.append(f"  - 既往 TB 史倍率：×{ss['prior_tb_multiplier']}；"
                 f"采样控制：{ss['sampling_control']}。")
    lines.append(f"  - 来源：{ss['source']}。")
    rn = ts['reproduction_number']
    lines.append(f"- **繁殖数 R_eff**：{rn['R_effective']}（文献范围 "
                 f"{rn['range']}），接触数 {rn['contact_count_distribution']}，"
                 f"代际衰减 {rn['generation_decay']}，"
                 f"代际因子 {rn['generation_factor']}。")
    lines.append(f"  - 来源：{rn['source']}。")
    g = ts['generations']
    lines.append(f"- **代际数**：最大 {g['max']}。{g['note']}。")
    geo = ts['contact_network_geometry']
    lines.append(f"- **接触网络几何**：{geo['contact_index_sampling']}；"
                 f"{geo['avoid_repeat']}。")
    lines.append(f"- **目标检出率**：{ts['target_detection_rate']}；"
                 f"标签：{ts['label_field']}。")
    lines.append('')

    # 3.1 标签机制版本登记表（P3：单一真值源索引，checkpoint 代际校验依据）
    lines.append('### 3.1 标签机制版本登记表（LABEL_MECHANISM_VERSIONS）')
    lines.append('')
    lines.append('| ID | 生成器 | 标签机制 | 真值源 |')
    lines.append('| --- | --- | --- | --- |')
    for e in report.get('label_mechanism_versions', []):
        lines.append(f"| `{e['id']}` | {e['generator']} | "
                     f"{e['label_mechanism']} | {e['source_of_truth']} |")
    lines.append('')
    lines.append('> checkpoint 元数据的 `label_generation` 字段（P4）以本表'
                 '为代际校验依据；跨代 AUROC 不可比，best.json 刷新需显式'
                 '冻结决策。')
    lines.append('')

    # 4. 随机种子与可复现性
    lines.append('## 4. 随机种子与可复现性')
    lines.append('')
    lines.append(f"- 默认种子：`{DEFAULT_RANDOM_STATE}`；"
                 '同一 `(random_state, n_samples)` 生成逐字段完全一致。')
    lines.append('- 可复现性由 `reproducibility_check()` 校验'
                 '（同种子两次生成逐字段比对）。')
    lines.append('')

    # 5. 分布对照校验结果
    val = report['validation']
    lines.append('## 5. 分布对照校验结果')
    lines.append('')
    lines.append(f"- 总体通过：**{'是' if report['overall_pass'] else '否'}**")
    lines.append('')
    dist_fail = [v for v in val['distributions'] if not v['pass']]
    if dist_fail:
        lines.append(f"- 字段校验通过 {len(val['distributions']) - len(dist_fail)}/"
                     f"{len(val['distributions'])}，失败字段：")
        for v in dist_fail:
            lines.append(f"  - `{v['field']}`：{v.get('discrepancies', v.get('checks'))}")
    else:
        lines.append(f"- 关键变量分布：{len(val['distributions'])} 个字段全部在容差内。")
    lines.append('')
    pv = val['prevalence']
    lines.append(f"- **检出率校验**：观测 {pv['observed_per_100k']}/10 万 "
                 f"（{pv['n_confirmed']}/{pv['n']}）vs 目标 "
                 f"{pv['target_per_100k']}/10 万，绝对差 "
                 f"{pv['abs_delta_per_100k']}/10 万，"
                 f"容差 {pv['tolerance_per_100k']}/10 万 → "
                 f"**{'通过' if pv['pass'] else '不通过'}**。")
    lines.append('')

    # 6. 任务标签口径
    lines.append('## 6. 任务标签口径（A/B/C 与公开数据衔接）')
    lines.append('')
    lines.append('- **任务 A（横断面筛查）**：`is_confirmed` —— 当前合成数据'
                 '可直接产出并校验（检出率 121/10 万）。')
    lines.append('- **任务 B（感染后进展）**：需 `infection_status` / '
                 '`progression` 随访标签 —— 由公开接触者队列'
                 '（如 ERASE-TB，见 `validation.public_datasets`）校准。')
    lines.append('- **任务 C（网络传播）**：需接触网络结构与传播簇标签 —— '
                 '公开数据稀缺，采用"合成网络 + 文献参数校准 + 敏感性分析"'
                 '过渡验证。')
    lines.append('- 三任务标签不可混用（同一批数据，任务 A 用基线筛查结果、'
                 '任务 B 用随访发病、任务 C 用网络传播结构）。')
    lines.append('')

    # 7. 文献参考表
    lines.append('## 7. 文献参考表（参数来源）')
    lines.append('')
    lines.append('| ID | 参考量 | 参考值 | 来源 | 当前可计算性 |')
    lines.append('| --- | --- | --- | --- | --- |')
    for ref in report['literature_references']:
        comp = ('可直接计算' if ref.get('computable_from_base_dgp')
                else '需任务 B/C 标签或网络层 DGP')
        lines.append(f"| `{ref['id']}` | {ref['quantity']} | "
                     f"{ref['value']} | {ref['source']} | {comp} |")
    lines.append('')

    # 8. 文献对照结果
    lit = val['literature']
    lines.append('## 8. 文献对照结果')
    lines.append('')
    for e in lit['entries']:
        if e['status'] == 'not_applicable':
            lines.append(f"- `{e['id']}` **暂不可计算**：{e['note']}。")
        else:
            lines.append(f"- `{e['id']}` **{e['status']}**："
                         f"观测 {e['observed']} vs 参考 {e['target']}。")
    lines.append('')

    return '\n'.join(lines)


__all__ = [
    'DEFAULT_RANDOM_STATE',
    'CATEGORICAL_TOLERANCE',
    'CONTINUOUS_MEAN_TOLERANCE',
    'PREVALENCE_TOLERANCE_PER_100K',
    'DGP_PARAMETERS',
    'FIELD_SPECS',
    'TRANSMISSION_SPEC',
    'LABEL_MECHANISM_VERSIONS',
    'LITERATURE_REFERENCES',
    'field_specs_by_name',
    'validate_synthetic_vs_spec',
    'validate_prevalence',
    'validate_against_literature',
    'reproducibility_check',
    'build_dgp_report',
    'render_dgp_document',
]
