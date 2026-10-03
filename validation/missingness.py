#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成数据增强 + 缺失值插补 + 缺失敏感性分析

三阶段闭环，明确"数据缺失对预测结果的边界影响"：

1. **合成数据增强**：复用 ``ScreeningDataSimulator``（基于独立 SEIR 传播链分配标签，
   打破循环论证）生成细粒度个体接触样本，扩充训练/评估样本。
2. **缺失值插补**：复用 ``data_io.imputation`` 的众数/条件中位数/场景默认值策略，
   对缺失字段补齐并记录填充审计。
3. **缺失敏感性分析**：在完整基线上一级级提高字段缺失率，插补后重新预测，
   量化风险分数偏移、风险等级翻转率等指标，找出"预测结果可接受退化"的缺失率边界。

核心思路：敏感性分析必须用**同一套插补策略**处理人工引入的缺失，才有实际意义——
它刻画的是"真实场景下数据不完整时，插补后预测与完整数据预测的偏差"。
"""

import logging
import math
import random

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy 缺失时回退
    NUMPY_AVAILABLE = False
    np = None

from ..scoring.engine import ScoringEngine

LOGGER = logging.getLogger("tb_risk.validation.missingness")

# 默认缺失敏感性分析字段（细粒度接触记录的关键预测字段）
DEFAULT_SENSITIVITY_FIELDS = [
    'has_symptoms', 'has_tb', 'bcg_vaccine', 'cumulative_exposure',
    'single_duration', 'freq_density', 'time_span', 'ventilation',
    'contact_distance', 'exposure_setting', 'past_illness_type',
    'age',
]

# 默认缺失率扫描网格
DEFAULT_MISSING_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

# 边界判定阈值
DEFAULT_MAX_MEAN_SHIFT = 5.0    # 平均风险绝对偏移（0-100 分制）上限
DEFAULT_MAX_GRADE_FLIP = 0.10   # 风险等级翻转率上限


def _default_predict_fn(record):
    """默认预测函数：ScoringEngine 综合风险评分（独立于 ML 模型）。

    返回 0-100 的疾病概率。缺失/异常时回退 0。
    """
    engine = ScoringEngine()
    try:
        result = engine.compute_risk_score(record)
        return float(result.get('disease_probability', 0.0))
    except Exception as e:  # pragma: no cover - 防御性保护
        LOGGER.warning("ScoringEngine 预测失败: %s", e)
        return 0.0


def generate_augmented_contact_dataset(n_samples=500, contact_type_ratio=0.5,
                                       random_state=42):
    """基于 SEIR 传播链模拟生成细粒度个体接触样本（合成数据增强）。

    复用 ``ScreeningDataSimulator``：其标签由独立 SEIR 传播链分配（打破循环论证），
    生成的人口学/接触/临床/环境字段与真实筛查口径一致。

    参数：
        n_samples (int): 生成个体接触样本数
        contact_type_ratio (float): 家庭接触者占比 (0-1)，其余为社会接触者
        random_state (int): 随机种子

    返回：
        list[dict]: 每条记录含 contact_type/_id/record_id 及全部预测字段
    """
    from ..scoring.simulator import ScreeningDataSimulator

    sim = ScreeningDataSimulator(random_state=random_state)
    records = sim.generate_dataset(n_samples=n_samples, include_labels=True)

    processed = []
    for idx, rec in enumerate(records):
        contact_type = 'family' if random.Random(random_state).random() < contact_type_ratio else 'social'
        enriched = dict(rec)
        enriched['contact_type'] = contact_type
        enriched['record_id'] = f"syn_{random_state}_{idx}"
        processed.append(enriched)
    return processed


def impute_contact_dataset(records, fields=None, strategy='auto'):
    """对接触记录做缺失值插补（复用 data_io.imputation）。

    参数：
        records (list[dict]): 接触记录列表
        fields (list[str]|None): 待插补字段，None 则取 DEFAULT_SENSITIVITY_FIELDS
        strategy (str): 插补策略（'auto'/'mode'/'conditional_median'/'median'/...）

    返回：
        (list[dict], dict): (插补后的记录, 填充审计报告)
        audit = {field: {'filled': int, 'strategy': str, 'value': any, 'missing_before': int}}
    """
    from ..data_io.imputation import impute_missing

    if fields is None:
        fields = DEFAULT_SENSITIVITY_FIELDS

    audit = {}
    for field in fields:
        missing_before = sum(
            1 for r in records if r.get(field) is None or r.get(field) == ''
        )
        filled, strategy_used, fill_value = impute_missing(
            records, field, strategy=strategy)
        audit[field] = {
            'filled': filled,
            'strategy': strategy_used,
            'value': fill_value,
            'missing_before': missing_before,
        }
    return records, audit


def _introduce_missingness(records, fields, missing_rate, rng):
    """按缺失率随机置空部分 (record, field) 值，模拟字段缺失。

    返回：新列表（不修改原记录），缺失位置置为 None。
    """
    if missing_rate <= 0.0:
        return [dict(r) for r in records]

    missing_records = []
    for rec in records:
        missing_rec = dict(rec)
        for field in fields:
            if field not in missing_rec:
                continue
            if rng.random() < missing_rate:
                missing_rec[field] = None
        missing_records.append(missing_rec)
    return missing_records


def _predict_all(records, predict_fn):
    """对记录列表批量预测，返回 0-100 风险数组。"""
    return [predict_fn(r) for r in records]


def _grade_flip_rate(baseline, imputed, bounds=(3.0, 8.0, 15.0)):
    """风险等级翻转率：四级分级（低/中/高/极高）下等级发生变化的记录占比。

    分级阈值与 constants DISEASE_PROB_* / threshold_spec 对齐。
    """
    if len(baseline) != len(imputed) or not baseline:
        return 0.0

    def grade(p):
        if p >= bounds[2]:
            return 3
        if p >= bounds[1]:
            return 2
        if p >= bounds[0]:
            return 1
        return 0

    flips = sum(1 for b, i in zip(baseline, imputed) if grade(b) != grade(i))
    return flips / len(baseline)


def _correlation(a, b):
    """Pearson 相关系数（numpy 不可用时回退朴实现）。"""
    if not a or len(a) != len(b):
        return 0.0
    if NUMPY_AVAILABLE:
        return float(np.corrcoef(a, b)[0, 1])
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def analyze_missingness(records, fields=None, missing_rates=DEFAULT_MISSING_RATES,
                        n_trials=5, predict_fn=None, random_state=42,
                        grade_bounds=(3.0, 8.0, 15.0)):
    """缺失敏感性分析主入口。

    步骤：
    1. 对完整记录计算基线风险；
    2. 对每个缺失率，多轮随机引入缺失 → 插补 → 预测；
    3. 汇总各轮指标（平均绝对偏移/最大偏移/等级翻转率/相关性）。

    参数：
        records (list[dict]): 完整接触记录列表
        fields (list[str]|None): 参与缺失模拟的字段
        missing_rates (tuple): 缺失率网格（含 0.0 作为基线）
        n_trials (int): 每个缺失率的随机试验轮数
        predict_fn (callable|None): 单条记录 → 0-100 风险；None 用 ScoringEngine
        random_state (int): 随机种子
        grade_bounds (tuple): 低/中/高/极高 分级阈值

    返回：
        dict: {
            'baseline': {'mean': float, 'std': float, 'n': int},
            'rates': [ { 'missing_rate': float, 'mean_abs_shift': float,
                         'max_abs_shift': float, 'mean_error_pct': float,
                         'grade_flip_rate': float, 'correlation': float,
                         'median_abs_shift': float }, ... ],
            'boundary': { ... } | None,
            'fields': list[str],
            'predict': predict_fn_name,
        }
    """
    if not records:
        return {'baseline': {'n': 0}, 'rates': [], 'boundary': None,
                'fields': fields or [], 'predict': _predict_fn_name(predict_fn)}

    if predict_fn is None:
        predict_fn = _default_predict_fn
    if fields is None:
        fields = DEFAULT_SENSITIVITY_FIELDS

    rng = random.Random(random_state)

    # 基线：完整数据预测
    baseline_risks = _predict_all(records, predict_fn)
    baseline_mean = sum(baseline_risks) / len(baseline_risks)
    baseline_std = _std(baseline_risks)

    rate_results = []
    for rate in missing_rates:
        if rate <= 0.0:
            rate_results.append(_rate_summary(
                rate, baseline_risks, baseline_risks, grade_bounds))
            continue

        shift_acc = []
        grade_flip_acc = []
        corr_acc = []
        for trial in range(n_trials):
            trial_rng = random.Random(random_state * 1000 + trial * 7 + int(rate * 100))
            missing_records = _introduce_missingness(records, fields, rate, trial_rng)
            imputed_records, _ = impute_contact_dataset(missing_records, fields)
            imputed_risks = _predict_all(imputed_records, predict_fn)

            shifts = [abs(b - i) for b, i in zip(baseline_risks, imputed_risks)]
            shift_acc.extend(shifts)
            grade_flip_acc.append(_grade_flip_rate(baseline_risks, imputed_risks, grade_bounds))
            corr_acc.append(_correlation(baseline_risks, imputed_risks))

        rate_results.append({
            'missing_rate': rate,
            'mean_abs_shift': _mean(shift_acc),
            'median_abs_shift': _median(shift_acc),
            'max_abs_shift': max(shift_acc) if shift_acc else 0.0,
            'mean_error_pct': _mean(shift_acc) / (baseline_mean + 1e-9),
            'grade_flip_rate': _mean(grade_flip_acc),
            'correlation': _mean(corr_acc),
        })

    boundary = find_boundary_missing_rate(
        rate_results, DEFAULT_MAX_MEAN_SHIFT, DEFAULT_MAX_GRADE_FLIP)

    return {
        'baseline': {'mean': baseline_mean, 'std': baseline_std, 'n': len(records)},
        'rates': rate_results,
        'boundary': boundary,
        'fields': fields,
        'predict': _predict_fn_name(predict_fn),
    }


def find_boundary_missing_rate(rate_results, max_mean_shift=DEFAULT_MAX_MEAN_SHIFT,
                               max_grade_flip=DEFAULT_MAX_GRADE_FLIP):
    """确定"可接受退化"的缺失率边界。

    从高到低扫描，返回第一个满足以下条件的缺失率：
        mean_abs_shift <= max_mean_shift 且 grade_flip_rate <= max_grade_flip
    （即在该缺失率及以下，预测退化仍在可接受范围内）。

    返回：
        dict|None: {'missing_rate': float, 'mean_abs_shift': float,
                    'grade_flip_rate': float, 'max_allowed_mean_shift': float,
                    'max_allowed_grade_flip': float}
    """
    valid = [r for r in rate_results
             if r['mean_abs_shift'] <= max_mean_shift
             and r['grade_flip_rate'] <= max_grade_flip]
    if not valid:
        return None
    # 取可接受区间内的最大缺失率（最"激进"仍可接受的点）
    best = max(valid, key=lambda r: r['missing_rate'])
    return {
        'missing_rate': best['missing_rate'],
        'mean_abs_shift': best['mean_abs_shift'],
        'grade_flip_rate': best['grade_flip_rate'],
        'max_allowed_mean_shift': max_mean_shift,
        'max_allowed_grade_flip': max_grade_flip,
    }


def _predict_fn_name(predict_fn):
    return getattr(predict_fn, '__name__', type(predict_fn).__name__)


def _rate_summary(rate, baseline, imputed, grade_bounds):
    shifts = [abs(b - i) for b, i in zip(baseline, imputed)]
    return {
        'missing_rate': rate,
        'mean_abs_shift': _mean(shifts),
        'median_abs_shift': _median(shifts),
        'max_abs_shift': max(shifts) if shifts else 0.0,
        'mean_error_pct': 0.0,
        'grade_flip_rate': _grade_flip_rate(baseline, imputed, grade_bounds),
        'correlation': _correlation(baseline, imputed),
    }


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _std(xs):
    if not len(xs) > 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))