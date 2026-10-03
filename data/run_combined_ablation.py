#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合机制特征化消融正式运行（用户 P1/P2，2026-08-25）。

主实验：n=400 × 20 种子（seed 101-120），特征阶梯
ind/+typed/+window/+pi/+all（冻结注册表 RF/LGBM）+ seir/gnn_pi/oracle
参考臂；归档 data/processed/combined_ablation_multiseed_20260825.json。
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tb_risk.validation.combined_ablation import (  # noqa: E402
    run_multi_seed_combined_ablation, MODEL_KEYS, FEATURE_ARMS,
)

STAMP = time.strftime('%Y%m%d')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed',
                   f'combined_ablation_multiseed_{STAMP}.json')


def summarize(out):
    print('\n===== 主实验（n=%d, %d 种子）=====' % (
        out['design']['n_contacts'], out['design']['n_seeds']))
    arm_keys = ([f'{fs}:{mk}' for fs in FEATURE_ARMS for mk in MODEL_KEYS]
                + ['seir', 'gnn_pi', 'oracle_nu'])
    for k in arm_keys:
        s = out['arm_summary'][k]
        print('  %-26s AUROC=%.4f  recall@25%%=%.3f  brier=%.4f' % (
            k, s['mean_auroc'], s['mean_recall_at_budget'], s['mean_brier']))
    print()
    for mk in MODEL_KEYS:
        for key in ('typed_minus_ind', 'window_minus_ind', 'pi_minus_ind',
                    'all_minus_ind', 'all_minus_best_single',
                    'synergy_all_minus_sum_singles', 'oracle_minus_ind',
                    'realization_ratio', 'gnn_pi_minus_all'):
            lk = f'{mk}:{key}'
            if lk not in out['ladder_summary']:
                continue
            s = out['ladder_summary'][lk]
            star = '*' if s.get('ci_excludes_zero') else ' '
            print('  %s %-42s Δ=%+.4f CI=[%+.4f,%+.4f] P(>0)=%d/%d' % (
                star, lk, s['mean'], s['bootstrap_ci'][0],
                s['bootstrap_ci'][1], s['positive_seeds'],
                out['design']['n_seeds']))
        a = out['additivity'][mk]
        print('  >> %s: wins=%s verdict=%s realization=%.1f%%' % (
            mk, a['featureization_wins'], a['additivity_verdict'],
            100.0 * a['realization_ratio']['mean']))


def save(out):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'\n归档 → {os.path.abspath(OUT)}')


if __name__ == '__main__':
    t0 = time.time()
    out = run_multi_seed_combined_ablation(
        n_contacts=400, n_seeds=20, seed_start=101, gnn_epochs=800)
    summarize(out)
    save(out)
    print('总耗时 %.1f 分钟' % ((time.time() - t0) / 60.0))
