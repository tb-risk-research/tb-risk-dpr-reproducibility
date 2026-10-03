#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据 PI 验证正式运行（用户 P4b，2026-08-25）。

马拉维 PACTS 试验 S1 Dataset（PLOS ONE 补充材料）：804 接触者 /
192 户 / 主终点 sx_3m（107 事件）；household 分层分组 5 折 CV ×
20 种子；归档 data/processed/real_data_pi_validation_20260825.json。
"""

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from tb_risk.validation.real_data_pi import (  # noqa: E402
    run_multi_seed_real_data, save_result,
)


def summarize(out):
    d = out['design']
    print('\n===== P4b 真实数据 PI 验证（n=%d, %d 户, %d 事件, %d 种子）=====' % (
        d['n'], d['n_households'], d['n_events'], d['n_seeds']))
    print('终点：', d['endpoint'])
    for k, s in out['arm_summary'].items():
        print('  %-26s AUROC=%.4f±%.3f PR=%.4f R@25%%=%.3f  疾病终点迁移=%.3f' % (
            k, s['mean_auroc'], s['sd_auroc'], s['mean_pr_auc'],
            s['mean_recall_at_budget'], s['mean_auroc_m3_tb_out']))
    print()
    for k, s in out['ladder_summary'].items():
        star = '*' if s.get('ci_excludes_zero') else ' '
        print('  %s %-56s Δ=%+.4f CI=[%+.4f,%+.4f] P(>0)=%.2f' % (
            star, k, s['mean'], s['bootstrap_ci'][0],
            s['bootstrap_ci'][1], s['p_positive']))
    pd_ = out['physics_direction']
    print('\n物理方向检查：')
    for q, v in pd_['lambda_quartile_sx_3m'].items():
        print('  λ %s: n=%d sx_3m=%.3f 疾病事件=%d' % (
            q, v['n'], v['sx_3m_rate'], v['m3_tb_events']))
    print('  涂片+/−: %.3f vs %.3f' % (
        pd_['smear_gradient']['smear_pos']['sx_3m_rate'],
        pd_['smear_gradient']['smear_neg']['sx_3m_rate']))
    print('  年龄 <5/中/≥45: %.3f / %.3f / %.3f' % (
        pd_['age_gradient']['lt5']['sx_3m_rate'],
        pd_['age_gradient']['mid']['sx_3m_rate'],
        pd_['age_gradient']['ge45']['sx_3m_rate']))
    a = pd_['asymptomatic_subcohort']
    print('  基线无症状亚群: n=%d 事件=%d λ AUROC=%s' % (
        a['n'], a['sx_3m_events'],
        'null' if a['lam_auroc'] is None else '%.4f' % a['lam_auroc']))
    print('\nDeLong（多种子中位 p / P(p<0.05)）:')
    for k, s in out['delong_summary'].items():
        print('  %-56s p_med=%.3f frac<0.05=%.2f' % (
            k, s['median_p'], s['frac_p_lt_0.05']))


if __name__ == '__main__':
    t0 = time.time()
    out = run_multi_seed_real_data(n_seeds=20, seed_start=0, n_splits=5,
                                   n_bootstrap=2000)
    summarize(out)
    path = save_result(out)
    print('\n归档 → %s' % os.path.abspath(path))
    print('总耗时 %.1f 分钟' % ((time.time() - t0) / 60.0))
