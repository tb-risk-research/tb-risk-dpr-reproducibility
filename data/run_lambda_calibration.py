#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""λ 联合校准正式运行（P1/P2/P3，2026-08-25）。

HomeACF：20 种子 × StratifiedGroupKFold(5) by household × 户级
cluster bootstrap 2000；归档 data/processed/lambda_calibration_*.json。
"""

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from tb_risk.validation.lambda_calibration import (  # noqa: E402
    run_multi_seed_calibration, save_result,
)


def summarize(out):
    d = out['design']
    print('\n===== λ 联合校准 HomeACF（n=%d, %d 户, %d 事件, %d 种子）====='
          % (d['n'], d['n_households'], d['n_events'], d['n_seeds']))
    print('管线：', d['pipeline'])
    for k, s in out['arm_summary'].items():
        print('  %-22s AUROC=%.4f±%.3f PR=%.4f%s' % (
            k, s['mean_auroc'], s['sd_auroc'], s['mean_pr_auc'],
            '  [%s]' % s['note'] if s.get('note') else ''))
    print()
    for k, s in out['ladder_summary'].items():
        star = '*' if s.get('ci_excludes_zero') else ' '
        print('  %s %-42s Δ=%+.4f CI=[%+.4f,%+.4f] P(>0)=%.2f' % (
            star, k, s['mean'], s['bootstrap_ci'][0],
            s['bootstrap_ci'][1], s['p_positive']))
    print('\n验收（P1/P2）：')
    for k, v in d['acceptance'].items():
        print('  %s: %s' % (k, v))
    print('\n乘子审计表（P2，先验 vs 数据）：')
    print('  %-15s %-8s %-8s %-10s %-9s %s' % (
        'multiplier', 'prior_dir', 'data_dir', 'uniAUROC',
        'cal_coef', '刻度比'))
    for r in out['multiplier_audit']:
        if not r.get('testable'):
            ks = r.get('known_subset', {})
            print('  %-15s 不可检验（%s）known: n=%s ev=%s '
                  'smear+率=%s vs smear−率=%s AUROC=%s' % (
                      r['multiplier'], r.get('note', ''),
                      ks.get('n'), ks.get('n_events'),
                      ks.get('smear_pos_rate_pos10'),
                      ks.get('smear_neg_rate_pos10'),
                      ks.get('univariate_oof_auroc_known')))
            continue
        print('  %-15s %-8s %-8s %-10.4f %-9.4f %.2f' % (
            r['multiplier'], r['prior_direction'], r['data_direction'],
            r['univariate_oof_auroc'], r['calibrated_coef'],
            r['scale_ratio_cal_over_prior'] or float('nan')))
    print('\nDeLong（多种子中位 p / P(p<0.05)）:')
    import numpy as np
    for k in out['seeds'][0]['delong'].keys():
        ps = [r['delong'][k] for r in out['seeds']]
        print('  %-42s p_med=%.3f frac<0.05=%.2f' % (
            k, float(np.median([p['p_value'] for p in ps])),
            float(np.mean([p['p_value'] < 0.05 for p in ps]))))


if __name__ == '__main__':
    t0 = time.time()
    out = run_multi_seed_calibration(n_seeds=20, seed_start=0,
                                     n_splits=5, n_bootstrap=2000)
    path = save_result(out)
    print('归档 → %s' % os.path.abspath(path))
    summarize(out)
    print('总耗时 %.1f 分钟' % ((time.time() - t0) / 60.0))
