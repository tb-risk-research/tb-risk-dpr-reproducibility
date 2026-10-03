#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机制价值来源受控消融（P0.2，2026-09-05）。

用户问题："+0.074 归因到物理结构还是多模态融合？固定物理（不学
δβ/δI0）vs 学机制 vs 纯表征。"

设计：三臂同起点控制变量（初始消息完全相同，验证 mech==free==
frozen at init——见 tests）：
  joint            全可学（SEIR 2 + 类型 5 + 衰减 5 = 12 机制参数）
  joint_frozen     SEIR 冻结（文献默认 Λ），类型/衰减可学——
                   隔离"学 ODE 参数"的贡献
  joint_free_msg   自由 type×window 消息表（25 参数，无 SEIR/
                   类型/衰减因子分解），初值=机制乘积——隔离
                   "物理因子分解形式"的贡献（多模态融合但无物理）

参照（P5 归档 joint_ablation_multiseed_20260904.json，同种子同网络
可配对）：ensemble_mean / features_lgbm / member_* / seir_joint /
oracle_nu。joint 臂重跑并断言与归档逐位一致（确定性审计）。

阶梯：
  joint − joint_frozen        学 ODE 参数的价值
  joint − joint_free_msg      物理因子分解的价值（负 = 自由表更好）
  joint_free_msg − ensemble   图结构×消息传递（无物理）vs 堆叠
  joint_free_msg − features_lgbm  同信息量级对照
归档参照：joint − ensemble = +0.0745（P5）。

协议：seeds 101-120 × n=400，分层 50/50，训练半区拟合、测试半区
报告；种子级 bootstrap CI + 逐种子 DeLong。
"""

import json
import os
import sys
import time

# __file__ = <repo>/tb_risk/data/run_joint_attribution.py；三层 dirname
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from tb_risk.ml.gnn.joint_mech_gnn import train_joint_mech_gnn  # noqa: E402
from tb_risk.validation.combined_network import (  # noqa: E402
    build_combined_network,
)
from tb_risk.validation.layer_ablation import delong_paired_test  # noqa: E402
from tb_risk.validation.seir_joint import _stratified_split  # noqa: E402
from tb_risk.validation.threshold_spec import compute_auc  # noqa: E402

STAMP = time.strftime('%Y%m%d')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'processed', f'joint_attribution_multiseed_{STAMP}.json')
ARCHIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'processed',
                       'joint_ablation_multiseed_20260904.json')

ARMS = ('joint', 'joint_frozen', 'joint_free_msg')
ARM_KWARGS = {
    'joint': {},
    'joint_frozen': {'seir_mode': 'frozen'},
    'joint_free_msg': {'msg_mode': 'free'},
}
DELONG_PAIRS = (
    ('joint', 'joint_frozen'),
    ('joint', 'joint_free_msg'),
    ('joint_free_msg', 'joint_frozen'),
)


def run_once(seed, joint_epochs=1200, lr=0.03):
    net = build_combined_network(n_contacts=400, target_rate=0.25,
                                 random_state=seed)
    labels = np.asarray(net['labels'], dtype=int)
    rng = np.random.RandomState(seed)
    train_idx, test_idx = _stratified_split(labels, rng)
    y_test = labels[test_idx]

    arms = {}
    for arm, kw in ARM_KWARGS.items():
        r = train_joint_mech_gnn(net, labels, train_idx,
                                 epochs=joint_epochs, lr=lr,
                                 prior_lambda=1e-3, seed=seed, **kw)
        s = r['score'][test_idx]
        arms[arm] = {
            'test_scores': s.tolist(),
            'auroc': float(compute_auc(s, y_test)),
            'loss_drop': r['loss_first'] - r['loss_last'],
        }
    delong = {
        f'{a}_vs_{b}': delong_paired_test(
            y_test, arms[a]['test_scores'], arms[b]['test_scores'])
        for a, b in DELONG_PAIRS}
    return {'seed': seed, 'arms': arms, 'delong': delong,
            'positive_rate': float(y_test.mean())}


def main():
    with open(ARCHIVE, encoding='utf-8') as f:
        archive = json.load(f)
    arch_by_seed = {s['seed']: s for s in archive['seeds']}
    assert set(arch_by_seed) == set(range(101, 121)), '归档种子不匹配'

    t0 = time.time()
    reports = []
    for i, seed in enumerate(range(101, 121)):
        rep = run_once(seed)
        # 确定性审计：joint 臂必须与 P5 归档逐位一致
        a_fresh = rep['arms']['joint']['auroc']
        a_arch = arch_by_seed[seed]['auroc']['joint']
        assert abs(a_fresh - a_arch) < 1e-12, \
            f'seed {seed}: joint AUROC 复算 {a_fresh} != 归档 {a_arch}'
        reports.append(rep)
        print('  seed %d: joint %.4f frozen %.4f free_msg %.4f (%.0fs)'
              % (seed, a_fresh,
                 rep['arms']['joint_frozen']['auroc'],
                 rep['arms']['joint_free_msg']['auroc'],
                 time.time() - t0), flush=True)

    rng = np.random.RandomState(101)

    def _boot(vals):
        vals = np.asarray(vals, dtype=float)
        means = [vals[rng.randint(0, len(vals), len(vals))].mean()
                 for _ in range(2000)]
        return [float(np.percentile(means, 2.5)),
                float(np.percentile(means, 97.5))]

    arm_summary = {}
    for arm in ARMS:
        vals = [r['arms'][arm]['auroc'] for r in reports]
        arm_summary[arm] = {'mean': float(np.mean(vals)),
                            'bootstrap_ci': _boot(vals)}

    # 与归档臂的配对阶梯（同种子同网络——AUROC 配对合法）
    ref_arms = ('ensemble_mean', 'features_lgbm', 'seir_joint',
                'member_pi', 'oracle_nu')
    ladder = {}
    contrasts = [
        ('joint', 'joint_frozen'), ('joint', 'joint_free_msg'),
        ('joint_free_msg', 'joint_frozen'),
        ('joint_free_msg', 'ensemble_mean'),
        ('joint_free_msg', 'features_lgbm'),
        ('joint_free_msg', 'seir_joint'),
        ('joint_free_msg', 'member_pi'),
        ('oracle_nu', 'joint_free_msg'),
    ]
    for a, b in contrasts:
        if a in ARMS:
            va = [r['arms'][a]['auroc'] for r in reports]
        else:
            va = [arch_by_seed[r['seed']]['auroc'][a] for r in reports]
        if b in ARMS:
            vb = [r['arms'][b]['auroc'] for r in reports]
        else:
            vb = [arch_by_seed[r['seed']]['auroc'][b] for r in reports]
        delta = [x - y for x, y in zip(va, vb)]
        ci = _boot(delta)
        ladder[f'{a}_minus_{b}'] = {
            'mean': float(np.mean(delta)),
            'bootstrap_ci': ci,
            'positive_seeds': int(sum(d > 0 for d in delta)),
            'ci_excludes_zero': bool(ci[0] > 0 or ci[1] < 0),
        }

    delong_summary = {}
    for key in reports[0]['delong']:
        ps = [float(r['delong'][key]['p_value']) for r in reports]
        delong_summary[key] = {
            'mean_p': float(np.mean(ps)),
            'frac_p_lt_0.05': float(np.mean(np.array(ps) < 0.05)),
        }

    out = {
        'design': {
            'name': 'joint_attribution_v1',
            'n_contacts': 400, 'n_seeds': 20, 'seeds': '101-120',
            'joint_epochs': 1200,
            'arms': {
                'joint': '12 机制参数全可学（P5 主体）',
                'joint_frozen': 'SEIR 冻结，类型/衰减可学（学 ODE 归因）',
                'joint_free_msg': '自由 25 参数 type×window 消息表，'
                                  '初值=机制乘积（物理形式归因）',
            },
            'determinism_audit': 'joint 臂与 P5 归档逐位一致（断言）',
            'reference_archive': os.path.basename(ARCHIVE),
        },
        'arm_summary': arm_summary,
        'ladder_summary': ladder,
        'delong_summary': delong_summary,
        'seeds': reports,
    }

    print('\n===== P0.2 机制价值来源归因（20 种子）=====')
    for arm in ARMS:
        s = arm_summary[arm]
        print('  %-14s AUROC=%.4f CI=[%.4f,%.4f]' % (
            arm, s['mean'], s['bootstrap_ci'][0], s['bootstrap_ci'][1]))
    print()
    for key, s in ladder.items():
        star = '*' if s['ci_excludes_zero'] else ' '
        dl = delong_summary.get(key.replace('_minus_', '_vs_'))
        dls = (' DeLong %.0f%%' % (100 * dl['frac_p_lt_0.05'])) if dl else ''
        print('  %s %-36s Δ=%+.4f CI=[%+.4f,%+.4f] P(>0)=%d/20%s' % (
            star, key, s['mean'], s['bootstrap_ci'][0],
            s['bootstrap_ci'][1], s['positive_seeds'], dls))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'\n归档 → {os.path.abspath(OUT)}')
    print('总耗时 %.1f 分钟' % ((time.time() - t0) / 60.0))


if __name__ == '__main__':
    main()
