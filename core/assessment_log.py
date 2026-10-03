#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估日志与分布漂移监控（core 层，纯 Python，不依赖 tkinter）。

部署期漂移监控的数据基座：每次风险评估（单次或批量）把"去标识化"的
接触者特征摘要与最终评分追加到 ``assessment_log.jsonl``（只追加、不覆盖，
与训练记录同一工程约定）。漂移监控视图基于该日志计算：

  - 数值特征 PSI（Population Stability Index，5 分位箱）
  - 布尔特征阳性率偏移
  - 评分分布 PSI（模型输出本身的漂移）

 PSI 判读惯例（银行业评分卡标准）：
  - < 0.10  分布稳定
  - 0.10 ~ 0.25  轻度漂移（关注）
  - >= 0.25  显著漂移（告警）

所有函数防御性返回 dict（``available=False`` 时带 ``reason``），绝不向上抛异常。
"""

import json
import logging
import math
import os
import time

LOGGER = logging.getLogger("tb_risk.core.assessment_log")

LOG_FILENAME = 'assessment_log.jsonl'

# 漂移监控的数值特征（与 22 维 ML 特征空间的基础维度对齐）
NUMERIC_FEATURES = (
    'age', 'single_duration', 'freq_density', 'time_span',
    'cumulative_exposure', 'ventilation',
)
# 布尔特征（按阳性率偏移监控）
BOOL_FEATURES = ('has_symptoms', 'bcg_vaccine', 'is_high_risk')
# 评分字段（模型输出漂移）
SCORE_FEATURES = ('score', 'ml_prob')

FEATURE_LABELS = {
    'age': '年龄',
    'single_duration': '单次接触时长',
    'freq_density': '接触频率',
    'time_span': '接触周期',
    'cumulative_exposure': '累计暴露',
    'ventilation': '通风条件',
    'has_symptoms': '有症状比例',
    'bcg_vaccine': '卡介苗接种比例',
    'is_high_risk': '高危人群比例',
    'score': '最终评分分布',
    'ml_prob': 'ML 评分分布',
}

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

_LEVEL_STABLE = '稳定'
_LEVEL_MODERATE = '轻度漂移'
_LEVEL_SIGNIFICANT = '显著漂移'


def _default_base_dir():
    """日志根目录：优先 TB_RISK_ARCHIVE_DIR（沙箱/自定义部署），否则 ~/.tb_risk。"""
    env_dir = os.environ.get("TB_RISK_ARCHIVE_DIR", "").strip()
    if env_dir:
        return env_dir
    return os.path.join(os.path.expanduser('~'), '.tb_risk')


def assessment_log_path(base_dir=None):
    """评估日志文件路径。"""
    return os.path.join(base_dir or _default_base_dir(), LOG_FILENAME)


# ==============================================================================
# 记录构建与追加
# ==============================================================================

def build_assessment_record(contact, score=None, ml_prob=None, source='single'):
    """从接触者 entry 构建一条去标识化评估日志记录。

    参数：
        contact (dict): 接触者数据（姓名字段会被剔除，不落盘）
        score (float|None): 最终风险评分（0-100，三层联合/集成/轻量评分）
        ml_prob (float|None): ML 方向评分（0-100，未参与时 None）
        source (str): 'single'（单次评估）或 'batch'（批量评估）

    返回：
        dict: 可 JSON 序列化的日志记录
    """
    from ..utils import _is_yes

    def _num(key):
        v = contact.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _bool01(key):
        return 1 if _is_yes(contact.get(key, 0)) else 0

    rec = {
        'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': str(source),
        'contact_type': ('family' if contact.get('contact_type') == 'family'
                         else contact.get('_contact_type') or 'social'),
        'age': _num('age'),
        'single_duration': _num('single_duration'),
        'freq_density': _num('freq_density'),
        'time_span': _num('time_span'),
        'cumulative_exposure': _num('cumulative_exposure'),
        'ventilation': _num('ventilation'),
        'has_symptoms': _bool01('has_symptoms'),
        'bcg_vaccine': _bool01('bcg_vaccine'),
        'is_high_risk': _bool01('is_high_risk'),
        'score': float(score) if isinstance(score, (int, float)) else None,
        'ml_prob': float(ml_prob) if isinstance(ml_prob, (int, float)) else None,
    }
    return rec


def append_assessment_records(records, base_dir=None):
    """追加评估记录到日志文件（只追加不覆盖；单行 JSON，损坏不影响他行）。

    参数：
        records (list[dict]): build_assessment_record 产出的记录列表
        base_dir (str|None): 日志根目录（None 时自动解析）

    返回：
        int: 实际写入的记录数
    """
    if not records:
        return 0
    path = assessment_log_path(base_dir)
    written = 0
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            for rec in records:
                try:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    written += 1
                except (TypeError, ValueError) as e:
                    LOGGER.debug("跳过不可序列化的评估记录: %s", e)
    except OSError as e:
        LOGGER.debug("评估日志写入失败（不阻断主流程）: %s", e)
    return written


def read_assessment_log(base_dir=None, limit=None):
    """读取评估日志（按时间正序，跳过损坏行）。

    返回：
        list[dict]: 日志记录列表（可能为空）
    """
    path = assessment_log_path(base_dir)
    if not os.path.exists(path):
        return []
    records = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    if limit is not None and limit > 0:
        records = records[-limit:]
    return records


# ==============================================================================
# PSI 计算
# ==============================================================================

def compute_psi(baseline, current, n_bins=5):
    """计算两个数值样本间的 PSI（Population Stability Index）。

    以基线样本的分位数建箱，比较当前样本在各箱中的占比偏移。
    样本过少或基线无变异（常数列）时返回 None。

    注意 PSI 的小样本噪声底：期望噪声 ≈ (n_bins-1)/n_current
    （如 5 箱 + 当前窗口 50 条 → 噪声底约 0.08，接近 0.10 关注线），
    因此默认 5 箱并要求当前窗口 ≥ 50。判读时应结合报告中的
    ``noise_floor`` 字段。

    参数：
        baseline (list[float]): 基线窗口样本
        current (list[float]): 当前窗口样本
        n_bins (int): 分箱数（默认 5，抗小样本噪声）

    返回：
        float|None: PSI 值；不可计算时 None
    """
    base = sorted(float(v) for v in baseline if isinstance(v, (int, float)))
    cur = [float(v) for v in current if isinstance(v, (int, float))]
    if len(base) < 10 or len(cur) < 5:
        return None
    if base[-1] - base[0] <= 1e-12:
        # 基线常数列：当前同为该常数则稳定，否则视为显著漂移
        same = all(abs(v - base[0]) <= 1e-12 for v in cur)
        return 0.0 if same else PSI_SIGNIFICANT

    # 基线分位数建箱（去重边界）
    edges = []
    for i in range(1, n_bins):
        q = base[min(len(base) - 1, int(i * len(base) / n_bins))]
        if not edges or q > edges[-1]:
            edges.append(q)
    if len(edges) < 1:
        return None

    def _distribution(samples):
        counts = [0] * (len(edges) + 1)
        for v in samples:
            idx = 0
            while idx < len(edges) and v > edges[idx]:
                idx += 1
            counts[idx] += 1
        total = float(len(samples))
        return [c / total for c in counts]

    eps = 1e-4
    base_dist = _distribution(base)
    cur_dist = _distribution(cur)
    psi = 0.0
    for b, c in zip(base_dist, cur_dist):
        b = max(b, eps)
        c = max(c, eps)
        psi += (c - b) * math.log(c / b)
    return psi


def psi_level(psi):
    """PSI 值 → 漂移等级（稳定/轻度漂移/显著漂移）。"""
    if psi is None:
        return '不可计算'
    if psi >= PSI_SIGNIFICANT:
        return _LEVEL_SIGNIFICANT
    if psi >= PSI_MODERATE:
        return _LEVEL_MODERATE
    return _LEVEL_STABLE


# ==============================================================================
# 漂移报告
# ==============================================================================

def compute_drift_report(records=None, base_dir=None,
                         baseline_n=100, current_n=50, n_bins=5):
    """基于评估日志计算分布漂移报告。

    基线窗口取最早的 baseline_n 条记录，当前窗口取最新的 current_n 条；
    两窗口不重叠（日志不足时降低要求并说明原因）。

    参数：
        records (list[dict]|None): 已读取的日志记录（None 时自动读取）
        base_dir (str|None): 日志根目录
        baseline_n (int): 基线窗口大小（默认 100）
        current_n (int): 当前窗口大小（默认 50 — PSI 5 箱时噪声底约
            0.08，窗口再小则噪声淹没信号，见 compute_psi 文档）
        n_bins (int): PSI 分箱数（默认 5，抗小样本噪声）

    返回：
        dict: 漂移报告（available=False 时带 reason）；``noise_floor``
            字段给出当前窗口下的 PSI 期望噪声底 (n_bins-1)/n_current，
            判读接近 0.10 关注线的读数时应结合该值
    """
    if records is None:
        records = read_assessment_log(base_dir)
    baseline_n = max(int(baseline_n), 10)
    current_n = max(int(current_n), 5)

    if len(records) < baseline_n + current_n:
        return {
            'available': False,
            'reason': (f"评估日志记录不足（当前 {len(records)} 条，"
                       f"需要 ≥ {baseline_n + current_n} 条：基线 {baseline_n} + "
                       f"当前 {current_n}）。随评估/批量评估的持续使用自动积累。"),
            'n_records': len(records),
        }

    baseline = records[:baseline_n]
    current = records[-current_n:]

    def _values(recs, key):
        return [r.get(key) for r in recs
                if isinstance(r.get(key), (int, float))]

    features = []
    alerts = []
    worst = _LEVEL_STABLE

    for key in NUMERIC_FEATURES + SCORE_FEATURES:
        base_vals = _values(baseline, key)
        cur_vals = _values(current, key)
        psi = compute_psi(base_vals, cur_vals, n_bins=n_bins)
        level = psi_level(psi)
        features.append({
            'name': key,
            'label': FEATURE_LABELS.get(key, key),
            'kind': 'score' if key in SCORE_FEATURES else 'numeric',
            'psi': psi,
            'level': level,
            'base_mean': (sum(base_vals) / len(base_vals)) if base_vals else None,
            'cur_mean': (sum(cur_vals) / len(cur_vals)) if cur_vals else None,
        })
        if level == _LEVEL_SIGNIFICANT:
            worst = _LEVEL_SIGNIFICANT
            alerts.append(f"{FEATURE_LABELS.get(key, key)} 显著漂移（PSI={psi:.3f}）")
        elif level == _LEVEL_MODERATE and worst != _LEVEL_SIGNIFICANT:
            worst = _LEVEL_MODERATE

    categorical = []
    for key in BOOL_FEATURES:
        base_vals = _values(baseline, key)
        cur_vals = _values(current, key)
        if not base_vals or not cur_vals:
            continue
        base_rate = sum(base_vals) / len(base_vals)
        cur_rate = sum(cur_vals) / len(cur_vals)
        delta = cur_rate - base_rate
        # 阳性率绝对偏移 ≥ 0.15 视为显著（类别型变量的实用阈值）
        level = (_LEVEL_SIGNIFICANT if abs(delta) >= 0.15
                 else (_LEVEL_MODERATE if abs(delta) >= 0.05 else _LEVEL_STABLE))
        categorical.append({
            'name': key,
            'label': FEATURE_LABELS.get(key, key),
            'base_rate': base_rate,
            'cur_rate': cur_rate,
            'delta': delta,
            'level': level,
        })
        if level == _LEVEL_SIGNIFICANT:
            worst = _LEVEL_SIGNIFICANT
            alerts.append(
                f"{FEATURE_LABELS.get(key, key)} 阳性率偏移 {delta:+.1%}")
        elif level == _LEVEL_MODERATE and worst != _LEVEL_SIGNIFICANT:
            worst = _LEVEL_MODERATE

    return {
        'available': True,
        'n_records': len(records),
        'n_baseline': len(baseline),
        'n_current': len(current),
        'n_bins': n_bins,
        'noise_floor': round((n_bins - 1) / max(len(current), 1), 3),
        'baseline_window': (baseline[0].get('ts', ''),
                            baseline[-1].get('ts', '')),
        'current_window': (current[0].get('ts', ''),
                           current[-1].get('ts', '')),
        'features': features,
        'categorical': categorical,
        'overall_level': worst,
        'alerts': alerts,
    }


__all__ = [
    'assessment_log_path',
    'build_assessment_record',
    'append_assessment_records',
    'read_assessment_log',
    'compute_psi',
    'psi_level',
    'compute_drift_report',
    'NUMERIC_FEATURES',
    'BOOL_FEATURES',
    'SCORE_FEATURES',
    'FEATURE_LABELS',
    'PSI_MODERATE',
    'PSI_SIGNIFICANT',
]
