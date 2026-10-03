#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEIR 参数联合学习消融正式运行（用户 P4a，2026-08-25）。

主实验：n=400 × 20 种子（seed 101-120），五臂
seir_default / seir_joint / free_windows / oracle_window / oracle_nu；
归档 data/processed/seir_joint_ablation_multiseed_20260825.json。
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tb_risk.validation.seir_joint import (  # noqa: E402
    run_multi_seed_seir_joint_ablation, ARMS,
)

STAMP = time.strftime('%Y%m%d')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed',
                   f'seir_joint_ablation_multiseed_{STAMP}.json')


def summarize(out):
    print('\n===== P4a 主实验（n=%d, %d 种子）=====' % (
        out['design']['n_contacts'], out['design']['n_seeds']))
    for k in ARMS:
        s = out['arm_summary'][k]
        print('  %-16s AUROC=%.4f  recall@25%%=%.3f  brier=%.4f' % (
            k, s['mean_auroc'], s['mean_recall_at_budget'], s['mean_brier']))
    print()
    for key, s in out['ladder_summary'].items():
        star = '*' if s.get('ci_excludes_zero') else ' '
        print('  %s %-32s Δ=%+.4f CI=[%+.4f,%+.4f] P(>0)=%d/%d' % (
            star, key, s['mean'], s['bootstrap_ci'][0],
            s['bootstrap_ci'][1], s['positive_seeds'],
            out['design']['n_seeds']))
    print('\n参数恢复（种子均值）：')
    for key, s in out['param_recovery_summary'].items():
        print('  %-22s mean=%.4f CI=[%.4f,%.4f]' % (
            key, s['mean'], s['bootstrap_ci'][0], s['bootstrap_ci'][1]))
    c = out['conclusion']
    print('\n  >> joint_learning_helps=%s' % c['joint_learning_helps'])
    print('  >> structure_prior_verdict=%s' % c['structure_prior_verdict'])
    print('  >> recovery_ratio=%.1f%%' % (
        100.0 * c['recovery_ratio']['mean']))


def save(out):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'\n归档 → {os.path.abspath(OUT)}')


if __name__ == '__main__':
    t0 = time.time()
    out = run_multi_seed_seir_joint_ablation(
        n_contacts=400, n_seeds=20, seed_start=101, joint_epochs=1200)
    summarize(out)
    save(out)
    print('总耗时 %.1f 分钟' % ((time.time() - t0) / 60.0))
