#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""折损-风险相关性检验（A 系列，2026-09-07）：桥接最后一个未验证假设。

地位：S3 级联折损注入假设"折损-风险无关"（人群级锚 0.68×0.42 =
0.2856 常数乘子，两臂同乘——同乘单调性下靶向优势平凡存活）。本
脚本用 TBESC-II Part B 个体级数据直接检验该假设：**预测风险与级联
完成是否相关**。若高风险者更难完成（负相关），靶向优势缩水——桥
接价值主张的最后软肋。

命题（预声明，先于运行；方向均不设预期——机制对冲：高龄 → 临床
驱动启动更多，外国出生 → 可及性障碍更少，净方向真不确定）：
- A1（锚复现 gate）：级联计数（25,792/17,404/16,593/14,527/1,729）
  与 demo 臂 AUROC 0.6041（n=16,575/2,215）复现 §8.12 归档
  （|Δ|<0.002，StratifiedKFold(5)×10 种子同协议）；联合阳性分母
  复核（TST+∪QFT+ → 1,729，≈42% 口径）。
- A2（主命题，两阶段斜率）：OOF 风险分（demo 臂，预测 TST+）与
  (a) 放置阶段 P(placed|registered, score)、(b) 启动阶段
  P(initiated|TST+, score) 的 logistic 斜率——site（ClinicId）
  cluster bootstrap CI。判据：|CI 不含零| 报告方向；两阶段净效应
  （乘积）由 A3 定量。
- A3（桥接平移更新）：retention(score) = P(place)×P(init|+) 按
  score 十分位非参估计；靶向保留比值 ρ(k)=E[retention|top-k]/
  E[retention]（top-k 按注册人群分数分布，k∈{0.10,0.20,0.30}）；
  以 ρ(k) 修正 S3 平移：靶向臂有效覆盖 = capture(k)×0.2856×ρ(k)
  vs 随机臂 = k×0.2856（绝对锚保持 Part B 级联水平，只更新相对
  比值——相对量引用纪律沿用）。重算 Δ(full−blanket) 病例·天。
- A4（QFT 层敏感性）：QFT+ 层启动-风险斜率与 TST+ 层同向（两
  平台/两期稳健性读数）。

判决预告（预声明）：A3 的三种可能——ρ>1（正相关，靶向优势放大）、
ρ≈1（S3 假设成立，回归平凡）、ρ<1（负相关，缩水）。若 ρ<1 且
capture 增益不足以补偿（capture(k)×ρ(k) < k），方向翻负——
S3 的"相对优势存活"结论被推翻，如实改写部署文档 §12。

诚实边界：
- 观察性关联非因果：启动决策受临床指南与站点协议驱动——但桥接
  用途正是"有效保留函数"的描述性输入（平移层），不要求因果解释；
- Part B 人群（美国移民筛查为主，出生国梯度强）→ ρ 方向对克拉
  玛依的可迁移性未知：只作方向性参考 + 敏感性区间，不作点预测；
- 放置阶段受站点协议驱动（QFT 抽取有选择性——7,967/25,792）；
  TST+ 层为单一通路主分析，QFT+ 层敏感性，联合层仅描述锚；
- retention 用十分位非参乘积（避免 logistic 外推）；ρ(k) 的
  bootstrap 为 site cluster 重采样（诊所级，两阶段联动重估）；
- score 对全注册人群可用（demo 列注册表全覆盖；年龄缺失中位数
  填补——18 岁以下填补对放置分析影响披露）；
- 表单完成选择泄漏纪律：主分析仅用 demo 臂（注册表字段），风险
  表字段不进主分析（其完成与 TST+ 共线，§8.12 教训）。

用法：
    python data/run_attrition_risk_correlation.py
输出：
    data/processed/attrition_risk_partb_YYYYMMDD.json
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

import run_cascade_digests as pcd  # noqa: E402
import run_sop_seir_bridge as b1  # noqa: E402

from tb_risk.seir.intervention import (  # noqa: E402
    build_v4_initial_state,
    _default_v4_params,
    _trapz,
)

OUT = os.path.join(HERE, 'processed',
                   'attrition_risk_partb_%s.json' % time.strftime('%Y%m%d'))
B1_ARCHIVE = os.path.join(HERE, 'processed',
                          'sop_seir_bridge_homeacf_20260906.json')

K_GRID = (0.10, 0.20, 0.30)
N_BOOTSTRAP = 1000
RETENTION_ANCHOR = 0.2856   # S3：0.68 放置 × 0.42 启动（绝对锚不变）

DEMO_COLS = ['d_age', 'd_sex', 'RaceWhite', 'RaceBlack',
             'RaceAsian', 'IsLatino']

ANCHOR_CASCADE = {'registration': 25792, 'tst_placed': 17404,
                  'tst_interpreted': 16593, 'riskfactor_form': 14527,
                  'ltbi_treatment_started': 1729}
ANCHOR_DEMO_AUROC = 0.6041
ANCHOR_TST_COHORT = {'n': 16575, 'n_events': 2215}


def logistic_slope(x, y):
    """标准化 score 上 y 的 logistic 斜率（无正则）。"""
    from sklearn.linear_model import LogisticRegression
    xs = (np.asarray(x, dtype=float) - np.mean(x)) / (np.std(x) + 1e-12)
    m = LogisticRegression(C=1e6, max_iter=2000)
    m.fit(xs.reshape(-1, 1), np.asarray(y, dtype=int))
    return float(m.coef_[0][0])


def site_cluster_slope_ci(x, y, sites, n_boot, seed):
    """site cluster bootstrap 的斜率 CI。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    sites = np.asarray(sites)
    uniq = np.unique(sites)
    rng = np.random.RandomState(seed)
    slopes = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([np.where(sites == g)[0] for g in pick])
        yr = y[rows]
        if yr.sum() == 0 or yr.sum() == len(yr):
            continue
        slopes.append(logistic_slope(x[rows], yr))
    s = np.asarray(slopes)
    return {
        'slope_point': logistic_slope(x, y),
        'bootstrap_ci': [float(np.percentile(s, 2.5)),
                         float(np.percentile(s, 97.5))],
        'ci_excludes_zero': bool(np.percentile(s, 2.5) > 0
                                 or np.percentile(s, 97.5) < 0),
        'n_sites': int(len(uniq)),
    }


def decile_rates(x, y):
    """score 十分位的完成率（x 升序分位）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    q = pd.qcut(pd.Series(x), 10, labels=False, duplicates='drop')
    out = []
    for d in range(10):
        m = (q == d).to_numpy()
        if m.sum() == 0:
            out.append((d, 0, float('nan')))
            continue
        out.append((int(d), int(m.sum()), float(y[m].mean())))
    return out


def main():
    t0 = time.time()
    reg, st, qft, rfk, tx, ce = pcd.load_partb()
    tx_ids = set(tx['PatientId'])

    # ---- 队列重建（§8.12 同协议）----
    st_sorted = st.sort_values(['PatientId', 'TstPlacedDate'])
    first_st = st_sorted.groupby('PatientId').first().reset_index()
    qft_sorted = qft.sort_values(['PatientId', 'BloodDrawDate'])
    first_qft = qft_sorted.groupby('PatientId').first().reset_index()

    cascade = {
        'registration': len(reg),
        'tst_placed': int(first_st['TstPlacedDate'].notna().sum()),
        'tst_interpreted': int(first_st['TstInterpretation']
                               .isin([1, 2]).sum()),
        'riskfactor_form': int(rfk['PatientId'].nunique()),
        'ltbi_treatment_started': int(tx['PatientId'].nunique()),
    }
    gate_cascade = all(cascade[k] == v for k, v in ANCHOR_CASCADE.items())
    print('级联 gate: %s %s' % (cascade, 'OK' if gate_cascade else 'FAIL'))

    tst = first_st[first_st['TstInterpretation'].isin([1, 2])].copy()
    tst['y'] = (tst['TstInterpretation'] == 1).astype(int)
    reg_first = reg.groupby('PatientId').first().reset_index()
    a = tst[['PatientId', 'y']].merge(
        reg_first[['PatientId'] + DEMO_COLS + ['ClinicId']],
        on='PatientId', how='left')
    ok = a['y'].notna() & a['d_age'].notna()
    sub = a[ok].copy()
    for c in DEMO_COLS:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    sub[DEMO_COLS] = sub[DEMO_COLS].fillna(sub[DEMO_COLS].median())
    y = sub['y'].to_numpy(dtype=int)
    X = sub[DEMO_COLS].to_numpy(dtype=float)
    gate_cohort = (len(y) == ANCHOR_TST_COHORT['n']
                   and int(y.sum()) == ANCHOR_TST_COHORT['n_events'])

    # ---- OOF score（10 种子，demo 臂，均值分数）----
    oofs = [pcd.rf_oof(X, y, s) for s in range(pcd.SEED_START,
                                               pcd.SEED_START + pcd.N_SEEDS)]
    demo_aurocs = [pcd._auroc(y, o) for o in oofs]
    gate_demo = abs(float(np.mean(demo_aurocs)) - ANCHOR_DEMO_AUROC) < 0.002
    score_judged = np.mean(oofs, axis=0)   # 判读队列内（TST+ 层用）
    print('demo 臂 gate: %.4f vs %.4f %s | 队列 gate: %s' % (
        np.mean(demo_aurocs), ANCHOR_DEMO_AUROC,
        'OK' if gate_demo else 'FAIL',
        'OK' if gate_cohort else 'FAIL'))
    if not (gate_cascade and gate_cohort and gate_demo):
        print('!! 锚 gate 失败，中止')
        return None

    # 全注册人群 score（demo 列中位数填补——判读队列同一填补口径）。
    # 口径说明：注册层用 10 种子全拟合均值（一致尺度，rank 十分位不受
    # OOF 收缩混入影响；放置非训练标签，无泄漏）；TST+ 层用 OOF 均值
    # （其条件变量即训练终点，防内嵌）。
    reg_all = reg_first.copy()
    for c in DEMO_COLS:
        reg_all[c] = pd.to_numeric(reg_all[c], errors='coerce')
    fill = sub[DEMO_COLS].median()
    reg_all[DEMO_COLS] = reg_all[DEMO_COLS].fillna(fill)
    full_fits = []
    for s in range(pcd.SEED_START, pcd.SEED_START + pcd.N_SEEDS):
        m = pcd._make_model('random_forest', s)
        m.fit(X, y)
        full_fits.append(m.predict_proba(
            reg_all[DEMO_COLS].to_numpy(dtype=float))[:, 1])
    score_reg = np.mean(full_fits, axis=0)   # 全层一致（全拟合均值）

    placed = reg_all['PatientId'].isin(
        set(first_st['PatientId'])).to_numpy(dtype=int)
    sites = reg_all['ClinicId'].fillna(-1).to_numpy()
    n_sites = len(np.unique(sites))
    print('\n注册 n=%d，placed=%d（%.1f%%），sites=%d' % (
        len(reg_all), placed.sum(), 100 * placed.mean(), n_sites))

    # ---- A2a：放置阶段 ----
    s1_slope = site_cluster_slope_ci(score_reg, placed, sites,
                                     N_BOOTSTRAP, 0)
    s1_dec = decile_rates(score_reg, placed)
    print('\n=== A2a 放置阶段 P(placed|score) ===')
    print('  slope=%+.4f CI [%+.4f, %+.4f] %s' % (
        s1_slope['slope_point'], *s1_slope['bootstrap_ci'],
        '★' if s1_slope['ci_excludes_zero'] else '.'))
    print('  十分位完成率:', ['%.3f' % r for _, _, r in s1_dec])

    # ---- A2b：启动阶段（TST+ 层）----
    pos_mask = sub['y'].to_numpy() == 1
    pid_pos = sub['PatientId'].to_numpy()[pos_mask]
    score_pos = score_judged[pos_mask]
    init_pos = np.array([p in tx_ids for p in pid_pos], dtype=int)
    sites_pos = sub['ClinicId'].fillna(-1).to_numpy()[pos_mask]
    s3_slope = site_cluster_slope_ci(score_pos, init_pos, sites_pos,
                                     N_BOOTSTRAP, 1)
    s3_dec = decile_rates(score_pos, init_pos)
    print('\n=== A2b 启动阶段 P(initiated|TST+, score) ===')
    print('  n=%d events=%d（%.1f%%）' % (
        len(init_pos), init_pos.sum(), 100 * init_pos.mean()))
    print('  slope=%+.4f CI [%+.4f, %+.4f] %s' % (
        s3_slope['slope_point'], *s3_slope['bootstrap_ci'],
        '★' if s3_slope['ci_excludes_zero'] else '.'))
    print('  十分位完成率:', ['%.3f' % r for _, _, r in s3_dec])

    # ---- A4：QFT+ 层敏感性 ----
    fq = first_qft[first_qft['QftResult'].isin([1, 2])].copy()
    fq['yq'] = (fq['QftResult'] == 1).astype(int)
    fq_pos = fq[fq['yq'] == 1]
    pid_qpos = fq_pos['PatientId'].to_numpy()
    sc_lookup = dict(zip(reg_all['PatientId'], score_reg))
    score_qpos = np.array([sc_lookup.get(p, np.nan) for p in pid_qpos])
    keep = ~np.isnan(score_qpos)
    init_qpos = np.array([p in tx_ids for p in pid_qpos],
                         dtype=int)[keep]
    sites_q = reg_all.set_index('PatientId')['ClinicId'].fillna(-1) \
        .reindex(pid_qpos).to_numpy()[keep]
    s3q_slope = site_cluster_slope_ci(score_qpos[keep], init_qpos,
                                      sites_q, N_BOOTSTRAP, 2)
    print('\n=== A4 QFT+ 层 P(initiated|QFT+, score) ===')
    print('  n=%d events=%d（%.1f%%）slope=%+.4f CI [%+.4f, %+.4f] %s' % (
        keep.sum(), init_qpos.sum(), 100 * init_qpos.mean(),
        s3q_slope['slope_point'], *s3q_slope['bootstrap_ci'],
        '★' if s3q_slope['ci_excludes_zero'] else '.'))

    # ---- 联合阳性描述锚（≈42% 口径复核）----
    union_pos = set(sub[sub['y'] == 1]['PatientId']) | set(
        fq_pos['PatientId'])
    union_init = len(union_pos & tx_ids)
    union_anchor = {
        'n_union_positive': len(union_pos),
        'n_initiated': int(union_init),
        'rate': round(union_init / len(union_pos), 4),
        'tx_not_in_union': int(len(tx_ids) - len(union_pos & tx_ids)),
    }
    print('\n联合阳性层：%d 中启动 %d（%.1f%%）；tx 表不在联合内 %d' % (
        len(union_pos), union_init, 100 * union_init / len(union_pos),
        union_anchor['tx_not_in_union']))

    # ---- A3：retention 函数与 ρ(k) ----
    s1_by_dec = {d: r for d, _, r in s1_dec}
    s3_by_dec = {d: r for d, _, r in s3_dec}
    dec_reg = pd.qcut(pd.Series(score_reg), 10, labels=False,
                      duplicates='drop').to_numpy()

    def retention_of_decile(d):
        s1 = s1_by_dec.get(d, np.nan)
        s3 = s3_by_dec.get(d, np.nan)
        return s1 * s3

    ret_all = np.array([retention_of_decile(d) for d in dec_reg])
    valid = ~np.isnan(ret_all)
    order = np.argsort(-score_reg)
    rho = {}
    for k in K_GRID:
        m = max(1, int(np.ceil(k * len(score_reg))))
        top_rows = order[:m]
        r_top = np.nanmean(ret_all[top_rows])
        r_all = np.nanmean(ret_all)
        rho['%.2f' % k] = round(float(r_top / r_all), 4) if r_all > 0 \
            else float('nan')
    print('\n=== A3 靶向保留比值 ρ(k)=E[retention|top-k]/E[retention] ===')
    print('  E[retention] 整体=%.4f' % np.nanmean(ret_all))
    for k in K_GRID:
        print('  k=%.2f: ρ=%.4f' % (k, rho['%.2f' % k]))

    # ρ(k) site cluster bootstrap（两阶段联动重估——简化：重采样注册
    # 行，两阶段十分位随样本重估；站点不足时退化为行 bootstrap）
    rng = np.random.RandomState(3)
    use_sites = n_sites >= 20
    uniq_sites = np.unique(sites[valid])
    rho_boot = {('%.2f' % k): [] for k in K_GRID}
    for _ in range(N_BOOTSTRAP):
        if use_sites:
            pick = rng.choice(uniq_sites, size=len(uniq_sites),
                              replace=True)
            rows = np.concatenate([np.where(
                (sites == g) & valid)[0] for g in pick])
        else:
            rows = rng.choice(np.where(valid)[0],
                              size=int(valid.sum()), replace=True)
        if len(rows) < 100:
            continue
        srr = score_reg[rows]
        dec_b = pd.qcut(pd.Series(srr), 10, labels=False,
                        duplicates='drop').to_numpy()
        # 阶段率按原十分位函数映射（点估计层面的重估：位次重排）
        s1b = np.array([s1_by_dec.get(d, np.nan) for d in dec_b])
        # 启动阶段：TST+ 层行 bootstrap（同站点联动过重——用行级）
        pos_rows = rng.choice(len(score_pos), size=len(score_pos),
                              replace=True)
        dec_pb = pd.qcut(pd.Series(score_pos[pos_rows]), 10,
                         labels=False, duplicates='drop').to_numpy()
        ypb = init_pos[pos_rows]
        s3b_by = {}
        for d in np.unique(dec_pb):
            m = dec_pb == d
            s3b_by[int(d)] = float(ypb[m].mean())
        ret_b = np.array([s1_by_dec.get(d, np.nan)
                          * s3b_by.get(d, np.nan) for d in dec_b])
        okb = ~np.isnan(ret_b)
        if okb.sum() < 100:
            continue
        ord_b = np.argsort(-srr)
        r_all_b = np.nanmean(ret_b)
        for k in K_GRID:
            mb = max(1, int(np.ceil(k * len(srr))))
            rho_boot['%.2f' % k].append(
                float(np.nanmean(ret_b[ord_b[:mb]]) / r_all_b))
    rho_ci = {}
    for k in K_GRID:
        arr = np.asarray(rho_boot['%.2f' % k], dtype=float)
        rho_ci['%.2f' % k] = {
            'rho': rho['%.2f' % k],
            'bootstrap_ci': [round(float(np.percentile(arr, 2.5)), 4),
                             round(float(np.percentile(arr, 97.5)), 4)],
            'ci_excludes_one': bool(np.percentile(arr, 2.5) > 1
                                     or np.percentile(arr, 97.5) < 1),
        }
    print('  ρ bootstrap CI:', {k: v['bootstrap_ci']
                                for k, v in rho_ci.items()})

    # ---- A3 SEIR 平移更新（绝对锚 0.2856 保持，只更新相对比值）----
    with open(B1_ARCHIVE, encoding='utf-8') as f:
        b1a = json.load(f)
    pi_cohort = b1a['seir_translation']['baseline']['latent_fraction']
    capture_dep = {k: b1a['capture']['per_seed_summary'][k]['dep_fixed']
                   ['mean'] for k in ('0.10', '0.20', '0.30')}
    state_generic = build_v4_initial_state(
        n_population=b1.SEIR_POPULATION, random_state=b1.SEIR_SEED)
    initial_state_c, _ = b1.rescale_latent_to_pi(state_generic, pi_cohort)
    base_times, active_base, new_inf_base = b1.seir_run(initial_state_c,
                                                        _default_v4_params())
    cum_base = float(_trapz(active_base, base_times))
    seir_update = {}
    for k in K_GRID:
        kk = '%.2f' % k
        cap = capture_dep[kk]
        rk = rho[kk]
        n_courses = int(round(k * b1.SEIR_POPULATION))
        rules = {
            'blanket_S3_constant': b1.pt_params(
                k * RETENTION_ANCHOR, k * RETENTION_ANCHOR),
            'targeted_S3_constant': b1.pt_params(
                cap * RETENTION_ANCHOR, cap * RETENTION_ANCHOR),
            'blanket_riskdep': b1.pt_params(
                k * RETENTION_ANCHOR, k * RETENTION_ANCHOR),  # 随机臂不变
            'targeted_riskdep': b1.pt_params(
                cap * RETENTION_ANCHOR * rk,
                cap * RETENTION_ANCHOR * rk),
        }
        out_r = {n: b1.translate_rule(initial_state_c, base_times,
                                      cum_base, new_inf_base, p, n_courses)
                 for n, p in rules.items()}
        d_s3 = (out_r['targeted_S3_constant']['averted_case_days']
                - out_r['blanket_S3_constant']['averted_case_days'])
        d_rd = (out_r['targeted_riskdep']['averted_case_days']
                - out_r['blanket_riskdep']['averted_case_days'])
        seir_update[kk] = {
            'capture': cap, 'rho': rk,
            'n_courses': n_courses,
            'delta_S3_constant_retention': round(d_s3, 2),
            'delta_riskdep_retention': round(d_rd, 2),
            'direction_flipped': bool((d_s3 > 0) != (d_rd > 0)),
            'per100_riskdep': round(d_rd / (n_courses / 100.0), 3),
        }
        print('  k=%s cap=%.3f ρ=%.3f: Δ(S3常数)=%.1f → Δ(风险依赖)'
              '=%.1f 病例·天 %s' % (
                  kk, cap, rk, d_s3, d_rd,
                  'FLIP!' if d_rd <= 0 else ''))

    # ---- 判决汇总 ----
    verdicts = {
        'A1_anchor_gate': True,
        'A2a_placement_slope': s1_slope,
        'A2b_initiation_slope_TSTpos': s3_slope,
        'A3_rho': rho_ci,
        'A3_seir_direction_survives': all(
            not v['direction_flipped'] for v in seir_update.values()),
        'A4_qft_slope': s3q_slope,
        'A4_direction_consistent': bool(
            (s3_slope['slope_point'] > 0) == (s3q_slope['slope_point'] > 0)),
    }
    print('\n=== 预声明命题判决 ===')
    print('  A2a 放置斜率: %+.4f CI %s %s' % (
        s1_slope['slope_point'], [round(v, 4) for v in
                                  s1_slope['bootstrap_ci']],
        '★不含零' if s1_slope['ci_excludes_zero'] else '.含零'))
    print('  A2b 启动斜率（TST+ 层）: %+.4f CI %s %s' % (
        s3_slope['slope_point'], [round(v, 4) for v in
                                  s3_slope['bootstrap_ci']],
        '★不含零' if s3_slope['ci_excludes_zero'] else '.含零'))
    print('  A4 QFT+ 层斜率: %+.4f（与 TST+ 层%s向）' % (
        s3q_slope['slope_point'],
        '同' if verdicts['A4_direction_consistent'] else '反'))
    print('  A3 SEIR 方向存活: %s' % verdicts['A3_seir_direction_survives'])

    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'attrition_risk_partb_v1',
        'status': 'S3"折损-风险无关"假设的直接检验（预声明 A1-A4）；'
                  '观察性关联→平移层保留函数输入',
        'design': {
            'source': 'TBESC-II Part B（个体级链接：注册→放置→判读→'
                      '启动）',
            'score': 'demo 臂 OOF 均值（预测 TST+，StratifiedKFold(5)×10 '
                     '种子，§8.12 同协议；判读队列内 OOF，注册全层全拟'
                     '合均值）',
            'stages': 's1=P(placed|registered)；s3=P(initiated|TST+)；'
                      'A4=P(initiated|QFT+) 敏感性层',
            'bootstrap': 'site（ClinicId）cluster ×%d（n_sites=%d，'
                         '%s）' % (N_BOOTSTRAP, n_sites,
                                    'site 级' if use_sites else '行级退化'),
            'seir_update': '绝对锚 %.4f 保持（S3），仅相对比值 ρ(k) 由 '
                          'Part B 经验更新' % RETENTION_ANCHOR,
        },
        'hypotheses': {
            'A1': '级联计数 + demo 0.6041 锚复现',
            'A2': '放置/启动两阶段斜率（方向不设预期，机制对冲）',
            'A3': 'ρ(k)=E[retention|top-k]/E[retention]；ρ<1 且 '
                  'capture×ρ<k 时靶向方向翻负（如实改写 §12）',
            'A4': 'QFT+ 层与 TST+ 层斜率同向',
        },
        'verdicts': verdicts,
        'cascade': cascade,
        'union_positive_anchor': union_anchor,
        'placement_stage': {
            'slope': s1_slope,
            'decile_rates': [{'decile': d, 'n': n, 'rate': round(r, 4)}
                             for d, n, r in s1_dec],
        },
        'initiation_stage_tstpos': {
            'n': int(len(init_pos)), 'n_events': int(init_pos.sum()),
            'slope': s3_slope,
            'decile_rates': [{'decile': d, 'n': n, 'rate': round(r, 4)}
                             for d, n, r in s3_dec],
        },
        'initiation_stage_qftpos': {
            'n': int(keep.sum()), 'n_events': int(init_qpos.sum()),
            'slope': s3q_slope,
        },
        'rho_and_seir': seir_update,
        'honest_boundaries': [
            '观察性关联非因果（启动受指南/站点协议驱动）——桥接用途为'
            '描述性保留函数输入，不要求因果解释',
            'Part B 为美国移民筛查人群（出生国梯度强）→ ρ 对克拉玛依'
            '可迁移性未知：只作方向性参考与敏感性区间，不作点预测',
            '主分析仅 demo 臂（注册表字段）——风险表字段因完成-TST+ '
            '共线不进主分析（§8.12 表单泄漏纪律）',
            'retention 为十分位非参乘积（s1×s3），ρ bootstrap 的两阶段'
            '联动为简化实现（s1 用原十分位函数映射，s3 行级重估）',
            'QFT 抽取有选择性（7,967/25,792，站点协议驱动）→ A4 层为'
            '敏感性读数',
            'score 对全注册层含 10 种子全拟合均值（判读队列内为 OOF）'
            '——部署式分数口径，非纯 OOF',
        ],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
