#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机制×图×时序端到端联合消融正式运行（P5，2026-09-04）。

用户问题：SEIR ODE 嵌入 GNN 消息传递 + 时序先验可学习的单一联合
模型，能否突破"三成员集成只追平"的天花板。

主实验：n=400 × 20 种子（seed 101-120，与 P4a/seir_joint 同种子
区间），十臂 seir_default / features_lgbm / member_{pi,typed,
temporal} / ensemble_mean / seir_joint / joint / joint_res / oracle_nu；
归档 data/processed/joint_ablation_multiseed_20260904.json。
"""

import json
import os
import sys
import time

# __file__ = <repo>/tb_risk/data/run_joint_ablation.py；import tb_risk.*
# 需要包根的父目录（repo 根）在 sys.path 上，故三层 dirname。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from tb_risk.validation.joint_ablation import (  # noqa: E402
    ARMS,
    run_multi_seed_joint_ablation,
)

STAMP = time.strftime('%Y%m%d')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed',
                   f'joint_ablation_multiseed_{STAMP}.json')


def summarize(out):
    print('\n===== P5 主实验（n=%d, %d 种子）=====' % (
        out['design']['n_contacts'], out['design']['n_seeds']))
    for k in ARMS:
        s = out['arm_summary'][k]
        print('  %-16s AUROC=%.4f  recall@25%%=%.3f  brier=%.4f' % (
            k, s['mean_auroc'], s['mean_recall_at_budget'], s['mean_brier']))
    print()
    for key, s in out['ladder_summary'].items():
        if not isinstance(s, dict) or 'mean' not in s:
            print('  %s: %s' % (key, s))
            continue
        star = '*' if s.get('ci_excludes_zero') else ' '
        delong = ''
        if 'delong_significant_frac' in s:
            delong = '  DeLong %.0f%%' % (
                100.0 * s['delong_significant_frac'])
        print('  %s %-36s Δ=%+.4f CI=[%+.4f,%+.4f] P(>0)=%d/%d%s' % (
            star, key, s['mean'], s['bootstrap_ci'][0],
            s['bootstrap_ci'][1], s['positive_seeds'],
            out['design']['n_seeds'], delong))
    print('\n参数恢复（种子均值）：')
    for key, s in out['param_recovery_summary'].items():
        print('  %-22s mean=%.4f CI=[%.4f,%.4f]' % (
            key, s['mean'], s['bootstrap_ci'][0], s['bootstrap_ci'][1]))
    c = out['conclusion']
    print('\n  >> joint_beats_ensemble=%s' % c['joint_beats_ensemble'])
    print('  >> joint_beats_features_lgbm=%s'
          % c['joint_beats_features_lgbm'])
    print('  >> joint_breaks_ceiling=%s'
          % c['joint_breaks_ceiling_vs_all_alternatives'])
    print('  >> ensemble_verdict=%s' % c['ensemble_verdict'])
    print('  >> joint_realization=%.1f%%' % (
        100.0 * c['joint_realization_ratio']['mean']))


def save(out):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'\n归档 → {os.path.abspath(OUT)}')


if __name__ == '__main__':
    t0 = time.time()
    out = run_multi_seed_joint_ablation(
        n_contacts=400, n_seeds=20, seed_start=101, joint_epochs=1200)
    summarize(out)
    save(out)
    print('总耗时 %.1f 分钟' % ((time.time() - t0) / 60.0))
