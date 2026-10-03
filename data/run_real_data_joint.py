#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据端到端联合复验正式运行（P6，2026-09-05）。

主实验：HomeACF（n=2725，359 事件，877 户）为主判据；
        PACTS（n=804，终点饱和）作结构退化记录。
六臂 pi_only / logit_lin / joint_frozen / joint_free_w / joint /
best_existing（逐种子最优 RF/LGBM 特征臂），seed 0-19 × 5 折
户分组 CV；归档 data/processed/real_data_joint_{dataset}_{date}.json。
"""

import json
import os
import sys
import time

# __file__ = <repo>/tb_risk/data/run_real_data_joint.py；import tb_risk.*
# 需要包根的父目录（repo 根）在 sys.path 上，故三层 dirname。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from tb_risk.validation.real_data_joint import (  # noqa: E402
    run_real_joint_multi_seed,
)

STAMP = time.strftime('%Y%m%d')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'processed')


def summarize(out):
    d = out['design']
    print('\n===== P6 真实复验（%s, n=%d, %d 事件, %d 户, %d 种子）=====' % (
        d['dataset'], d['n'], d['n_events'], d['n_households'],
        d['n_seeds']))
    for k, s in out['arm_summary'].items():
        print('  %-14s AUROC=%.4f±%.4f  pr=%.4f  r@25%%=%.3f  brier=%.4f'
              % (k, s['mean_auroc'], s['sd_auroc'], s['mean_pr_auc'],
                 s['mean_recall_at_budget'], s['mean_brier']))
    print()
    for key, s in out['ladder_summary'].items():
        star = '*' if s.get('ci_excludes_zero') else ' '
        dl = out['delong_summary'].get(
            key.replace('_minus_', '_vs_'), {})
        dls = ('  DeLong p<0.05 %.0f%%'
               % (100 * dl['frac_p_lt_0.05'])) if dl else ''
        print('  %s %-34s Δ=%+.4f CI=[%+.4f,%+.4f] P(>0)=%.2f%s' % (
            star, key, s['mean'], s['bootstrap_ci'][0],
            s['bootstrap_ci'][1], s['p_positive'], dls))
    print('\n  best_existing 臂分布: %s' % out['best_existing_arm_counts'])
    c = out['conclusion']
    print('  >> joint − best_existing = %+.4f CI=%s excl0=%s' % (
        c['joint_minus_best_existing']['mean'],
        ['%+.4f' % v for v in
         c['joint_minus_best_existing']['bootstrap_ci']],
        c['joint_minus_best_existing']['ci_excludes_zero']))
    print('  >> joint − pi_only      = %+.4f CI=%s excl0=%s' % (
        c['joint_minus_pi_only']['mean'],
        ['%+.4f' % v for v in
         c['joint_minus_pi_only']['bootstrap_ci']],
        c['joint_minus_pi_only']['ci_excludes_zero']))


def save(out, dataset):
    path = os.path.join(
        OUT_DIR, f'real_data_joint_{dataset}_{STAMP}.json')
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'归档 → {os.path.abspath(path)}')


if __name__ == '__main__':
    t0 = time.time()
    only = sys.argv[1] if len(sys.argv) > 1 else 'both'
    datasets = ('homeacf', 'pacts') if only == 'both' else (only,)
    for ds in datasets:
        print('\n>>>>>> dataset = %s' % ds, flush=True)
        out = run_real_joint_multi_seed(dataset=ds, n_seeds=20,
                                        seed_start=0)
        summarize(out)
        save(out, ds)
    print('\n总耗时 %.1f 分钟' % ((time.time() - t0) / 60.0))
