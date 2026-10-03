#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时序留出家庭传播模型正式运行（P1，2026-08-25 第三轮）。

HomeACF：20 种子 × StratifiedGroupKFold(5) by household × 户级
cluster bootstrap 2000（random 排序主口径）+ oracle/anti 敏感性
包络（3 种子）；归档 data/processed/household_temporal_*.json。
"""

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from tb_risk.validation.household_temporal import (  # noqa: E402
    run_multi_seed_temporal, save_result,
)


def summarize(out):
    d = out['design']
    print('\n===== 时序留出家庭传播模型 HomeACF（n=%d, %d 户, %d 事件, '
          '%d 种子, order=%s）=====' % (
              d['n'], d['n_households'], d['n_events'], d['n_seeds'],
              d['order_mode']))
    print('防泄漏：', d['leakage_argument'])
    for k, s in out['arm_summary'].items():
        print('  %-22s AUROC=%.4f±%.3f PR=%.4f%s' % (
            k, s['mean_auroc'], s['sd_auroc'], s['mean_pr_auc'],
            '  [%s]' % s['note'] if s.get('note') else ''))
    print()
    for k, s in out['ladder_summary'].items():
        star = '*' if s.get('ci_excludes_zero') else ' '
        print('  %s %-38s Δ=%+.4f CI=[%+.4f,%+.4f] P(>0)=%.2f' % (
            star, k, s['mean'], s['bootstrap_ci'][0],
            s['bootstrap_ci'][1], s['p_positive']))
    print('\n验收（P1）：')
    for k, v in d['acceptance'].items():
        print('  %s: mean=%+.4f ci=%s sig+=%s' % (
            k, v['mean'], v['ci'], v['significant_positive']))
    print('\n顺序敏感性包络（exposure_prior AUROC，%d 种子）:' %
          out['order_sensitivity']['oracle']['n_seeds'])
    for m, s in out['order_sensitivity'].items():
        print('  %-7s %.4f' % (m, s['exposure_prior_auroc_mean']))


if __name__ == '__main__':
    t0 = time.time()
    out = run_multi_seed_temporal(n_seeds=20, seed_start=0, n_splits=5,
                                  n_bootstrap=2000, order_mode='random',
                                  sensitivity_seeds=3)
    path = save_result(out)
    print('归档 → %s' % os.path.abspath(path))
    summarize(out)
    print('总耗时 %.1f 分钟' % ((time.time() - t0) / 60.0))
