#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据感染终点 PI 验证正式运行（用户 P4b-2，2026-08-25）。

南非 HomeACF 队列（github petermacp/tstsa）：2725 接触者 / 877 户 /
终点 tst_pos10（359 事件，LTBI 感染终点）；household 分层分组 5 折
CV × 20 种子；归档 data/processed/real_data_infection_validation_*.json。
"""

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from tb_risk.validation.real_data_infection import (  # noqa: E402
    run_multi_seed_infection, save_result,
)


def summarize(out):
    d = out['design']
    print('\n===== P4b-2 感染终点 PI 验证 HomeACF（n=%d, %d 户, '
          '%d 事件, %d 种子）=====' % (
              d['n'], d['n_households'], d['n_events'], d['n_seeds']))
    print('终点：', d['endpoint'])
    for k, s in out['arm_summary'].items():
        print('  %-26s AUROC=%.4f±%.3f PR=%.4f R@25%%=%.3f  '
              '≥5mm迁移=%.3f' % (
                  k, s['mean_auroc'], s['sd_auroc'], s['mean_pr_auc'],
                  s['mean_recall_at_budget'], s['mean_auroc_tst_pos5']))
    print()
    for k, s in out['ladder_summary'].items():
        star = '*' if s.get('ci_excludes_zero') else ' '
        print('  %s %-56s Δ=%+.4f CI=[%+.4f,%+.4f] P(>0)=%.2f' % (
            star, k, s['mean'], s['bootstrap_ci'][0],
            s['bootstrap_ci'][1], s['p_positive']))
    pd_ = out['physics_direction']
    print('\n物理方向检查：')
    for q, v in pd_['lambda_quartile_tst_pos10'].items():
        print('  λ %s: n=%d tst_pos10=%.3f' % (q, v['n'],
                                                v['tst_pos10_rate']))
    for g, v in pd_['coughdays_gradient'].items():
        print('  咳嗽 %s: n=%d rate=%.3f' % (g, v['n'],
                                             v['tst_pos10_rate']))
    sg = pd_['smear_gradient_known']
    print('  涂片已知子集 +/−: %.3f vs %.3f' % (
        sg['smear_pos']['tst_pos10_rate'],
        sg['smear_neg']['tst_pos10_rate']))
    print('  年龄 <5/中/≥45: %.3f / %.3f / %.3f' % (
        pd_['age_gradient']['lt5']['tst_pos10_rate'],
        pd_['age_gradient']['mid']['tst_pos10_rate'],
        pd_['age_gradient']['ge45']['tst_pos10_rate']))
    print('  宿主 HIV+/−: %.3f vs %.3f' % (
        pd_['host_hiv_gradient']['hiv_pos']['tst_pos10_rate'],
        pd_['host_hiv_gradient']['hiv_neg']['tst_pos10_rate']))
    print('  λ 整体 AUROC: %.4f' % pd_['lam_auroc_overall'])
    print('\nDeLong（多种子中位 p / P(p<0.05)）:')
    for k, s in out['delong_summary'].items():
        print('  %-56s p_med=%.3f frac<0.05=%.2f' % (
            k, s['median_p'], s['frac_p_lt_0.05']))


if __name__ == '__main__':
    t0 = time.time()
    out = run_multi_seed_infection(n_seeds=20, seed_start=0, n_splits=5,
                                   n_bootstrap=2000)
    summarize(out)
    path = save_result(out)
    print('\n归档 → %s' % os.path.abspath(path))
    print('总耗时 %.1f 分钟' % ((time.time() - t0) / 60.0))
