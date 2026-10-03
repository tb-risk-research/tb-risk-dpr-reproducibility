#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨基准（cross-benchmark）稳健性汇总：把"域未知条件下的集成价值"
做成正式归档指标。

背景：
    集成层冻结评估在两个标签族基准（机制/临床解耦）上各自运行 5 seeds。
    单看任一基准，集成只是"追平最优单方向"（DeLong p>0.05）；但把两基准
    视为"未知域"的两个抽样，学权重融合的平均/最差域性能超过任何单一
    模型——这是"集成为什么存在"的跨域稳健性证据（2026-08-23 用户指示，
    当前证据能支撑的最强表述）。

逻辑：
    1. 扫描 training_archive/results/（TrainingLogger 归档目录），
       按 dataset.source 分桶，每桶取最新 status=success 的 ensemble 运行；
    2. 对每个方法（单方向 + 融合法）计算：
       - mech_auroc / clin_auroc：两基准 5-seed 均值；
       - cross_benchmark_mean：域未知条件下的平均性能；
       - cross_benchmark_min：最差域性能（跨域稳健性下界）；
       - std 传播：mean_std = sqrt((s_mech^2 + s_clin^2) / 2)（独立近似）；
    3. 归档为新 run（model_type='ensemble_summary'，追加不覆盖），
       bucket ensemble_cross_benchmark_4m_v1。

用法：
    python data/summarize_cross_benchmark.py
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)

MECH_BUCKET = 'ensemble_eval_mechanistic_4m_v1'
CLIN_BUCKET = 'ensemble_eval_clinical_4m_v1'
SUMMARY_BUCKET = 'ensemble_cross_benchmark_4m_v1'

# 方法 -> 归档 metrics 中的 AUROC 键（历史键名大小写不统一，显式映射）
METHOD_KEYS = {
    # 单方向
    'ml_only': 'ml_only_AUROC',
    'ml_clin_only': 'ml_clin_only_AUROC',
    'seir_only': 'seir_only_AUROC',
    'gnn_only': 'gnn_only_AUROC',
    # 固定权重融合（部署形态）
    'ens_three': 'ens_three_AUROC',
    'ens_four': 'ens_four_AUROC',
    # 学权重融合（训练半区学习、测试半区报告）
    'grid4_best': 'grid4_best_auroc',
    'rank4_grid_best': 'rank4_grid_best_auroc',
    'cal4_grid_best': 'cal4_grid_best_auroc',
    'seir_prior': 'seir_prior_auroc',
    'lr_stack': 'lr_stack_auroc',
    'lr_gated': 'lr_gated_auroc',
}
SINGLE_METHODS = ('ml_only', 'ml_clin_only', 'seir_only', 'gnn_only')
FUSION_METHODS = tuple(m for m in METHOD_KEYS if m not in SINGLE_METHODS)


def _load_latest_success(results_dir, bucket, model_type='ensemble'):
    """读取指定 bucket 最新一条 success 运行的完整归档 JSON。

    历史教训：标签必须从实际加载的归档推断，不硬编码。
    """
    candidates = []
    if not os.path.isdir(results_dir):
        raise SystemExit(f"[错误] 归档目录不存在: {results_dir}")
    for fn in os.listdir(results_dir):
        if not fn.endswith('.json'):
            continue
        path = os.path.join(results_dir, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get('status') != 'success':
            continue
        if rec.get('model_type') != model_type:
            continue
        src = str((rec.get('dataset') or {}).get('source') or '')
        if src != bucket:
            continue
        candidates.append(rec)
    if not candidates:
        raise SystemExit(f"[错误] bucket '{bucket}' 无 success 归档")
    candidates.sort(key=lambda r: str(r.get('run_id') or r.get('timestamp')))
    return candidates[-1]


def _propagate_std(s1, s2):
    """两独立 5-seed 均值之平均的 std 传播（缺项时返回 None）。"""
    if s1 is None or s2 is None:
        return None
    return ((float(s1) ** 2 + float(s2) ** 2) / 2.0) ** 0.5


def summarize(mech_rec, clin_rec):
    """两基准 -> 每方法跨域 mean/min + headline（对外结论数字）。"""
    methods = {}
    for name, key in METHOD_KEYS.items():
        m_mech = (mech_rec.get('metrics') or {}).get(key)
        m_clin = (clin_rec.get('metrics') or {}).get(key)
        s_mech = (mech_rec.get('metrics') or {}).get(f'{key}_std')
        s_clin = (clin_rec.get('metrics') or {}).get(f'{key}_std')
        if m_mech is None or m_clin is None:
            continue  # 该方法在任一基准缺失（如旧版本归档）
        mean = (float(m_mech) + float(m_clin)) / 2.0
        methods[name] = {
            'mech_auroc': float(m_mech),
            'clin_auroc': float(m_clin),
            'cross_benchmark_mean': mean,
            'cross_benchmark_min': min(float(m_mech), float(m_clin)),
            'cross_benchmark_mean_std': _propagate_std(s_mech, s_clin),
        }

    singles = {k: v for k, v in methods.items() if k in SINGLE_METHODS}
    fusions = {k: v for k, v in methods.items() if k in FUSION_METHODS}
    if not singles or not fusions:
        raise SystemExit("[错误] 方法表不完整（单方向或融合法缺失）")

    # tie-break：mean 相同时取最差域更高者（更稳健）
    best_fusion = max(fusions, key=lambda k: (
        fusions[k]['cross_benchmark_mean'], fusions[k]['cross_benchmark_min']))
    best_single = max(singles, key=lambda k: (
        singles[k]['cross_benchmark_mean'], singles[k]['cross_benchmark_min']))
    headline = {
        'best_fusion_method': best_fusion,
        'best_fusion_cross_mean': round(fusions[best_fusion]['cross_benchmark_mean'], 4),
        'best_fusion_cross_min': round(fusions[best_fusion]['cross_benchmark_min'], 4),
        'best_single_method': best_single,
        'best_single_cross_mean': round(singles[best_single]['cross_benchmark_mean'], 4),
        'best_single_cross_min': round(singles[best_single]['cross_benchmark_min'], 4),
        'margin_mean': round(
            fusions[best_fusion]['cross_benchmark_mean']
            - singles[best_single]['cross_benchmark_mean'], 4),
        'margin_min': round(
            fusions[best_fusion]['cross_benchmark_min']
            - singles[best_single]['cross_benchmark_min'], 4),
    }
    return methods, headline


def archive(methods, headline, mech_rec, clin_rec):
    """TrainingLogger 归档（model_type='ensemble_summary'，追加不覆盖）。"""
    from train_with_public_data import get_logger

    metrics = {}
    for name, m in methods.items():
        metrics[f'{name}_mech_auroc'] = round(m['mech_auroc'], 4)
        metrics[f'{name}_clin_auroc'] = round(m['clin_auroc'], 4)
        metrics[f'{name}_cross_mean'] = round(m['cross_benchmark_mean'], 4)
        metrics[f'{name}_cross_min'] = round(m['cross_benchmark_min'], 4)
        if m['cross_benchmark_mean_std'] is not None:
            metrics[f'{name}_cross_mean_std'] = round(
                m['cross_benchmark_mean_std'], 4)
    metrics.update(headline)

    params = {
        'mech_run_id': mech_rec.get('run_id'),
        'clin_run_id': clin_rec.get('run_id'),
        'mech_bucket': MECH_BUCKET,
        'clin_bucket': CLIN_BUCKET,
        'n_methods': len(methods),
        'std_propagation': 'sqrt((s_mech^2 + s_clin^2)/2), independent approx',
        'definition': {
            'cross_benchmark_mean': '两基准 5-seed AUROC 均值的平均（域未知平均性能）',
            'cross_benchmark_min': '两基准 5-seed AUROC 均值的较小者（最差域性能）',
        },
    }
    dataset = {
        'source': SUMMARY_BUCKET,
        'n_samples': 0,
        'real_data': False,
        'provenance': (
            'cross-benchmark robustness summary aggregated from frozen '
            'ensemble_eval runs (mechanistic + clinical-decoupled labels); '
            'no new scoring, pure aggregation of archived 5-seed means'),
    }
    logger = get_logger()
    entry = logger.log(
        model_type='ensemble_summary', params=params, metrics=metrics,
        training_duration=0.0, status='success', dataset=dataset)
    print(f"[归档] run_id={entry.run_id}（bucket={SUMMARY_BUCKET}）")
    return metrics


def _print_report(methods, headline):
    print("\n===== 跨域稳健性（域未知 = 机制/临床双标签族基准，各 5 seeds）=====")
    print(f"{'方法':<18}{'机制':>8}{'临床':>8}{'跨域均值':>10}{'最差域':>9}")
    for name in METHOD_KEYS:
        m = methods.get(name)
        if not m:
            continue
        std = m['cross_benchmark_mean_std']
        std_s = f"±{std:.3f}" if std is not None else ''
        print(f"{name:<18}{m['mech_auroc']:>8.4f}{m['clin_auroc']:>8.4f}"
              f"{m['cross_benchmark_mean']:>10.4f}{m['cross_benchmark_min']:>9.4f}"
              f"  {std_s}")
    print("\n----- 对外结论（headline）-----")
    print(f"学权重融合最高跨域均值: {headline['best_fusion_method']} "
          f"{headline['best_fusion_cross_mean']}"
          f"（最差域 {headline['best_fusion_cross_min']}）")
    print(f"单一模型最高跨域均值: {headline['best_single_method']} "
          f"{headline['best_single_cross_mean']}"
          f"（最差域 {headline['best_single_cross_min']}）")
    print(f"平均性能优势: +{headline['margin_mean']:.3f} | "
          f"最差域优势: +{headline['margin_min']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--no-archive', action='store_true',
                    help='只打印汇总，不写入归档')
    args = ap.parse_args()

    from train_with_public_data import get_logger
    logger = get_logger()

    mech_rec = _load_latest_success(logger.results_dir, MECH_BUCKET)
    clin_rec = _load_latest_success(logger.results_dir, CLIN_BUCKET)
    print(f"[源运行] 机制: {mech_rec.get('run_id')} | "
          f"临床: {clin_rec.get('run_id')}")

    methods, headline = summarize(mech_rec, clin_rec)
    _print_report(methods, headline)

    if not args.no_archive:
        archive(methods, headline, mech_rec, clin_rec)


if __name__ == '__main__':
    main()
