#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOP P4 闭合实验（第一层，2026-09-25）：严格未来先证块 placebo
（C1 破坏实验）。

命题背景（model_boundary_clinical_validation.md §8.4-P4，命名形式化）：
  P4（合法性命题）："破坏 C1（使用同窗/未来结局）时增益消失或反转"——
  此前仅有 SINAN walk-forward 84 月回测的间接反事实（PSI 通道对照）；
  本脚本在主证据基座（HomeACF，+0.0585 CI[+0.034,+0.083]）上**直接实验**
  闭合 P4。

构造：先证块的时序镜像——future_n/future_pos/future_rate/future_screened
只聚合**位置严格更大**（更晚筛查）的同户成员结局（prior_features 的
镜像：位置 k 的先证块聚合位置 <k，未来块聚合位置 >k）。同一种子、同一
随机顺序（random 口径 = 部署期望主口径）、同一户分组 CV——唯一差异是
块的时间方向。

预声明判读（先于运行，用户规格原文）：
  H-P4a（placebo 持存）：Δ(exposure_future − exposure) > 0 且末种子
     cluster bootstrap CI 排除零 → 增益在 C1 破坏下仍存在 → **增益
     来自同户聚集信号而非时序信息**；
  H-P4b（顺序不变性）：Δ(exposure_future − exposure_prior) ≈ 0（CI
     含零）→ 未来块 ≈ 过去块 → C1 的角色 = 部署可行性闸门（未来结局
     部署时不可得），而非信号源；SOP 命名合法性保留于部署语义（序贯
     揭示），机制归属改写为"同户聚集，经时序序贯以可部署形式提取"；
  H-P4c（顺序特异性成分，反读数如实报告）：若 Δ(exposure_future −
     exposure_prior) CI 排除零 → 存在顺序特异性成分，按符号与量级
     如实报告，不作辩护。

结构预期（诚实披露）：random 顺序下，未来块是过去块在**同一随机顺序**
内的可交换镜像（补集子集）——纯聚集零假设下 Δ_future ≈ Δ_past 为近
必然（对称性论证），v1 归档的顺序极端包络（oracle 0.7430 vs anti
0.7186）已示顺序在极端下有二阶效应。本实验的价值不在"意外"，而在：
(i) 把 P4 的反事实从间接证据升级为主证据基座上的直接实验读数；
(ii) 对"顺序特异性成分"给出配对定量分割（Δ(future−prior) 的符号、
量级与 CI）。

臂（镜像三臂 + 复用 run_seed 锚臂）：
  future_only:RF      4 列未来块（prior_only 镜像）
  exposure_future:RF  exposure + 未来块（exposure_prior 镜像，主臂）
  dep_future          部署公式镜像 σ(logit(p_exp)+w(r̂_fut−π))，w=7/k=2
                      冻结（镜像对称性优先于最优性）

锚臂复现 gate：ind/exposure/prior_only/exposure_prior 逐臂 |Δ|<0.002
vs v1 归档（0.6428/0.6441/0.6443/0.7142）。

主对比（末种子 OOF 户级 cluster bootstrap ×2000 + 逐种子四栏 R5a）：
  exposure_future − exposure      placebo 增益（C1 破坏）
  exposure_prior − exposure       合法增益复现（+0.0585 锚）
  exposure_future − exposure_prior 顺序不变性读数（H-P4b/c 主判据）
  future_only − prior_only        纯块镜像
  dep_future − dep_static         部署公式 placebo（dep_static=exposure）
  dep_future − dep_fixed          部署公式镜像 Δ

诚实边界：
  - 单队列（HomeACF LTBI 终点）；random 顺序为部署期望口径，非实际
    筛查协议（真实顺序未记录）；
  - placebo 读数限于**池化 AUROC 通道**（P4 主张所在层）；户内自排除
    反相关（§8.11 H-I3）对称存在于未来块，不作户内分解；
  - dep_future 用同一 w=7（未对未来块重校准）——镜像对称性优先；
  - 结论口径为"机制与结构"（P4 闭合），跨人群迁移归 ERASE-TB 前瞻外验。

用法：
    python data/run_sop_future_placebo.py
输出：
    data/processed/sop_future_placebo_homeacf_YYYYMMDD.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))
sys.path.insert(0, HERE)

import run_sop_deepening as rsd  # noqa: E402

from tb_risk.validation.real_data_infection import (  # noqa: E402
    feature_columns as inf_feature_columns,
    load_homeacf_contacts,
)
from tb_risk.validation.real_data_pi import (  # noqa: E402
    _cluster_bootstrap_delta,
    _group_cv_indices,
)
from tb_risk.validation.household_temporal import (  # noqa: E402
    assign_screening_order,
    prior_features,
)

OUT = os.path.join(HERE, 'processed',
                   'sop_future_placebo_homeacf_%s.json'
                   % time.strftime('%Y%m%d'))

N_BOOTSTRAP = 2000
COLS_FUT = ['future_n', 'future_pos', 'future_rate', 'future_screened']
ANCHOR = {'ind': 0.6428, 'exposure': 0.6441,
          'prior_only': 0.6443, 'exposure_prior': 0.7142}


def _auroc(y, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y, dtype=int),
                               np.asarray(scores, dtype=float)))


def future_features(df, order_pos):
    """先证块的时序镜像：只聚合**位置严格更大**（更晚筛查）的同户结局。

    位置 t 的未来块 = 位置 t+1..m−1 的成员（严格未来，无平局）。
    prior_features 聚合位置 <t，本函数聚合位置 >t——同一顺序的补集。
    """
    y = df['tst_pos10'].to_numpy(dtype=float)
    hh = df['record_id'].to_numpy()
    pos = order_pos.to_numpy(dtype=float)
    n_rows = len(df)
    fut_n = np.zeros(n_rows)
    fut_pos = np.zeros(n_rows)
    for hh_id in np.unique(hh):
        idx_local = np.where(hh == hh_id)[0]
        order = np.argsort(pos[idx_local], kind='stable')
        sorted_idx = idx_local[order]
        yh = y[sorted_idx]
        k = len(yh)
        cs = np.cumsum(yh)
        # 位置 t：未来数 = k−1−t；未来阳性 = cs[k−1] − cs[t]
        fut_pos[sorted_idx] = cs[-1] - cs
        fut_n[sorted_idx] = np.arange(k - 1, -1, -1)
    base_rate = float(y.mean())
    fut_rate = np.where(fut_n > 0,
                        fut_pos / np.maximum(fut_n, 1.0),
                        base_rate)
    return pd.DataFrame({
        'future_n': fut_n,
        'future_pos': fut_pos,
        'future_rate': fut_rate,
        'future_screened': (fut_n > 0).astype(float),
    })


def run_seed_with_future(seed, df, y, groups):
    """run_seed（锚臂 + 部署公式）+ 未来镜像三臂，同折同顺序。"""
    r = rsd.run_seed(seed, df, y, groups)
    folds = _group_cv_indices(groups, y, n_splits=rsd.N_SPLITS, seed=seed)
    order = assign_screening_order(df, seed=seed, mode='random')
    ff = future_features(df, order)
    data = pd.concat([df.reset_index(drop=True), ff], axis=1)

    fut_n = ff['future_n'].to_numpy(dtype=float)
    fut_pos = ff['future_pos'].to_numpy(dtype=float)

    cols_exp = inf_feature_columns('exposure')
    oof = dict(r['oof'])
    oof['future_only'] = rsd.rf_oof(COLS_FUT, folds, y, data, seed)
    oof['exposure_future'] = rsd.rf_oof(
        cols_exp + COLS_FUT, folds, y, data, seed)
    oof['dep_future'] = rsd.dep_scores(
        oof['exposure'], fut_n, fut_pos, r['pi_true'],
        rsd.DEP_K, rsd.DEP_W)
    return {'seed': seed, 'oof': oof,
            'fut_n': fut_n, 'fut_pos': fut_pos}


def sanity_future_mirror(df):
    """镜像对称性自检：future_n+prior_n ≡ 户规模−1；两块阳性互补。"""
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    order = assign_screening_order(df, seed=0, mode='random')
    pf = prior_features(df, order)
    ff = future_features(df, order)
    hh_n = pd.Series(1, index=groups).groupby(groups).transform('size')
    tot_n = pf['prior_n'].to_numpy() + ff['future_n'].to_numpy()
    tot_pos = pf['prior_pos'].to_numpy() + ff['future_pos'].to_numpy()
    y_tot = pd.Series(y).groupby(groups).transform('sum').to_numpy()
    assert np.allclose(tot_n + 1, hh_n.to_numpy()), 'future_n 镜像断裂'
    assert np.allclose(tot_pos + y, y_tot), 'future_pos 镜像断裂'
    return True


def main():
    t0 = time.time()
    df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    print('HomeACF: n=%d events=%d households=%d' % (
        len(y), int(y.sum()), len(np.unique(groups))))
    assert sanity_future_mirror(df), '镜像自检失败'
    print('镜像自检 OK：prior_n+future_n ≡ 户规模−1，'
          'prior_pos+future_pos ≡ 户阳性−自身')

    seeds_results = []
    for s in range(rsd.SEED_START, rsd.SEED_START + rsd.N_SEEDS):
        r = run_seed_with_future(s, df, y, groups)
        r['y'] = y
        r['groups'] = groups
        seeds_results.append(r)
        print('seed %2d: exp_fut=%.4f fut_only=%.4f dep_fut=%.4f '
              ' (%.0fs)' % (
                  s, _auroc(y, r['oof']['exposure_future']),
                  _auroc(y, r['oof']['future_only']),
                  _auroc(y, r['oof']['dep_future']), time.time() - t0))

    # ---- 锚臂复现 gate ----
    anchor_gate = {}
    for a, expect in ANCHOR.items():
        vals = [_auroc(r['y'], r['oof'][a]) for r in seeds_results]
        v = float(np.mean(vals))
        anchor_gate[a] = {'this_run': round(v, 4), 'v1_archive': expect,
                          'reproduced': bool(abs(v - expect) < 0.002)}
        if not anchor_gate[a]['reproduced']:
            print('[WARN] anchor %s: %.4f vs %.4f 未复现！' % (a, v, expect))
    print('\n=== 锚臂复现 gate ===')
    for a, g in anchor_gate.items():
        print('  %-16s %.4f vs %.4f  %s' % (
            a, g['this_run'], g['v1_archive'],
            'OK' if g['reproduced'] else 'FAIL'))

    # ---- 主对比 ----
    contrast_pairs = [
        ('exposure_future', 'exposure', 'H-P4a：placebo 增益（C1 破坏）'),
        ('exposure_prior', 'exposure', '合法增益复现（+0.0585 锚）'),
        ('exposure_future', 'exposure_prior',
         'H-P4b/c：顺序不变性读数（主判据）'),
        ('future_only', 'prior_only', '纯块镜像'),
        ('dep_future', 'dep_static', '部署公式 placebo'),
        ('dep_future', 'dep_fixed', '部署公式镜像 Δ'),
    ]
    last = seeds_results[-1]
    ladder = {}
    for a, b, note in contrast_pairs:
        cb = _cluster_bootstrap_delta(last['oof'][a], last['oof'][b],
                                      last['y'], last['groups'],
                                      n_bootstrap=N_BOOTSTRAP,
                                      seed=rsd.SEED_START)
        ladder['%s_minus_%s' % (a, b)] = {
            'mean': cb['mean'], 'bootstrap_ci': cb['bootstrap_ci'],
            'p_positive': cb['p_positive'],
            'ci_excludes_zero': cb['ci_excludes_zero'],
            'per_seed_delta': rsd.per_seed_delta(seeds_results, a, b),
            'note': note,
        }
        ps = ladder['%s_minus_%s' % (a, b)]['per_seed_delta']
        print('%-40s %+0.4f CI [%+0.4f,%+0.4f] per-seed share+=%.2f'
              % ('%s−%s' % (a, b), cb['mean'], *cb['bootstrap_ci'],
                 ps['share_positive']))

    # ---- 臂汇总（20 种子均值）----
    arm_mean = {}
    for a in ('ind', 'exposure', 'prior_only', 'exposure_prior',
              'future_only', 'exposure_future', 'dep_static',
              'dep_fixed', 'dep_future'):
        vals = [_auroc(r['y'], r['oof'][a]) for r in seeds_results]
        arm_mean[a] = {'mean': round(float(np.mean(vals)), 4),
                      'sd': round(float(np.std(vals)), 4)}

    # ---- 汇总归档 ----
    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'sop_future_placebo_homeacf_v1',
        'status': 'P4 闭合实验（预声明判读 H-P4a/b/c）；结论口径=机制与'
                  '结构，跨人群迁移归 ERASE-TB',
        'design': {
            'source': 'HomeACF（github petermacp/tstsa）',
            'endpoint': 'tst_pos10（TST≥10mm LTBI）',
            'n': int(len(y)), 'n_events': int(y.sum()),
            'n_households': int(len(np.unique(groups))),
            'n_seeds': rsd.N_SEEDS, 'seed_start': rsd.SEED_START,
            'cv': 'StratifiedGroupKFold(%d) by household' % rsd.N_SPLITS,
            'ci': '户级 cluster bootstrap ×%d（末种子 OOF）' % N_BOOTSTRAP,
            'order_mode': 'random（部署期望口径；与 v1 同顺序同种子）',
            'placebo_construction': 'future 块 = 位置严格更大同户成员'
                                    '结局聚合（prior_features 时序镜像；'
                                    '自检：prior_n+future_n≡户规模−1，'
                                    'prior_pos+future_pos≡户阳性−自身）',
            'dep_formula': 'σ(logit(p_exp)+w(r̂_fut−π)), w=%.1f, k=%.1f '
                           '（镜像对称性优先，未重校准）' % (
                               rsd.DEP_W, rsd.DEP_K),
            'anchor_gate': anchor_gate,
            'structure_expectation': 'random 顺序下未来块为过去块的可交换'
                                     '镜像（纯聚集零假设下 Δ_fut≈Δ_past '
                                     '近必然）；实验价值=直接读数+顺序特异'
                                     '性成分配对分割（顺序极端包络见 v1：'
                                     'oracle 0.7430 vs anti 0.7186）',
        },
        'hypotheses': {
            'H-P4a': 'Δ(exposure_future−exposure)>0 且 CI 排除零 → '
                     '增益来自同户聚集信号而非时序信息（placebo 持存）',
            'H-P4b': 'Δ(exposure_future−exposure_prior)≈0（CI 含零）→ '
                     'C1=部署可行性闸门而非信号源',
            'H-P4c': '若顺序不变性 Δ CI 排除零 → 存在顺序特异性成分，'
                     '按符号量级如实报告',
        },
        'arm_summary': arm_mean,
        'ladder': ladder,
        'per_seed_arm_auroc': [
            {'seed': r['seed'],
             **{a: round(_auroc(r['y'], r['oof'][a]), 4)
                for a in ('ind', 'exposure', 'prior_only',
                          'exposure_prior', 'future_only',
                          'exposure_future', 'dep_fixed', 'dep_future')}}
            for r in seeds_results],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
