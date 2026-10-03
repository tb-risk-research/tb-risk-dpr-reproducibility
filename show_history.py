#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练历史查询 — 读取训练档案总表，按时间倒序展示每次训练的指标与提升

用法（在项目根目录执行）：
    python show_history.py                  # 全部历史（倒序）
    python show_history.py --type ml        # 只看 ML 训练记录
    python show_history.py --type gnn       # 只看 GNN 训练记录
    python show_history.py --limit 10       # 只看最近 10 条
    python show_history.py --best           # 只看当前历史最佳
    python show_history.py --stats          # 统计摘要

数据来源（由 gui/training_panel/training_log.py 写入）：
    ~/.tb_risk/training_log.jsonl   总表（每次训练追加一行）
    ~/.tb_risk/best.json            按模型类型的历史最佳
    ~/.tb_risk/results/             每次训练的完整档案
"""

import argparse
import os
import sys

# 以 tb_risk 包形式导入（gui 包内使用相对导入，必须挂在 tb_risk 命名空间下）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PROJECT_ROOT)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from tb_risk.gui.training_panel.training_log import (  # noqa: E402
    TrainingLogger, PRIMARY_METRIC,
)

STATUS_LABELS = {'success': '成功', 'failed': '失败', 'cancelled': '取消'}


def _fmt_delta(value):
    """格式化差值：正数带 +，None 显示 -"""
    if value is None:
        return '-'
    return f'{value:+.4f}'


def _fmt_duration(seconds):
    if seconds <= 0:
        return '-'
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes = int(seconds // 60)
    return f'{minutes}m{int(seconds % 60)}s'


def print_history(entries):
    """按时间倒序打印训练记录表"""
    if not entries:
        print('（暂无训练记录）')
        return

    header = (f'{"时间":<19} {"类型":<4} {"状态":<4} '
              f'{PRIMARY_METRIC:>7} {"vs上次":>8} {"vs最佳":>8} {"耗时":>7}  '
              f'run_id')
    print(header)
    print('-' * max(len(header), 100))

    for e in entries:
        status = STATUS_LABELS.get(e.status, e.status)
        auroc = e.primary_metric_value()
        auroc_str = f'{auroc:.4f}' if auroc is not None else '-'
        comp = e.comparison or {}
        vs_last = _fmt_delta(comp.get('vs_last'))
        vs_best = _fmt_delta(comp.get('vs_best'))
        duration = _fmt_duration(e.training_duration)
        new_best_flag = ' ★新最佳' if comp.get('is_new_best') else ''

        dataset = e.dataset or {}
        dataset_str = ''
        if dataset.get('source'):
            dataset_str = f"  数据集={dataset['source']}"
            if dataset.get('n_samples'):
                dataset_str += f"({dataset['n_samples']}样本)"

        print(f'{e.timestamp:<19} {e.model_type:<4} {status:<4} '
              f'{auroc_str:>7} {vs_last:>8} {vs_best:>8} {duration:>7}  '
              f'{e.run_id or "-"}{new_best_flag}{dataset_str}')

        if e.status == 'failed' and e.error_message:
            print(f'    └ 失败原因: {e.error_message[:120]}')


def print_best(logger, model_type=None):
    """打印当前历史最佳"""
    types = [model_type] if model_type else ['ml', 'gnn']
    found = False
    for mt in types:
        best = logger.get_best(mt)
        if not best:
            continue
        found = True
        metric = best.get('metrics', {}).get(PRIMARY_METRIC)
        print(f'[{mt}] 历史最佳 {PRIMARY_METRIC}={metric:.4f}  '
              f'run_id={best.get("run_id", "-")}  '
              f'时间={best.get("timestamp", "-")}')
        model_path = best.get('model_path', '')
        if model_path:
            exists = os.path.exists(model_path)
            print(f'      权重快照: {model_path}'
                  f'{"" if exists else "（文件已不存在）"}')
    if not found:
        print('（暂无历史最佳记录）')


def print_stats(logger):
    """打印统计摘要"""
    stats = logger.get_stats()
    print(f'总训练次数: {stats["total_count"]}  '
          f'成功: {stats["success_count"]}  '
          f'失败: {stats["failure_count"]}  '
          f'平均耗时: {_fmt_duration(stats["avg_duration"])}')
    if stats['last_training_time']:
        print(f'最近一次训练: {stats["last_training_time"]}')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='查询训练历史与指标提升（读取 ~/.tb_risk 训练档案）')
    parser.add_argument('--type', choices=['ml', 'gnn'], default=None,
                        help='筛选模型类型（默认全部）')
    parser.add_argument('--limit', type=int, default=None,
                        help='最多显示的记录数（默认全部）')
    parser.add_argument('--best', action='store_true',
                        help='只显示当前历史最佳')
    parser.add_argument('--stats', action='store_true',
                        help='显示统计摘要')
    args = parser.parse_args(argv)

    logger = TrainingLogger()

    if args.best:
        print_best(logger, args.type)
        return 0

    entries = logger.read_history(limit=args.limit, model_type=args.type)
    print_history(entries)

    if args.stats:
        print()
        print_stats(logger)
    if args.type is None:
        print()
        print_best(logger)
    return 0


if __name__ == '__main__':
    sys.exit(main())
