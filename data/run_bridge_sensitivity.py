#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOP×SEIR 桥接敏感性补强（B1-S，2026-09-07）：PT 效力网格 + 级联折损 + k 下探。

地位：B1 桥接（sop_seir_bridge_homeacf_20260906.json）的三个低成本确定性
补强——不重跑 SEIR 校准、不改 seir/ 库，仅扩网格与注入乘子。

命题（预声明，先于运行）：
- S1（k 下探 0.05，资源极稀缺窗口）：capture 富集倍数随 k 减小单调增大
  （k=0.05 富集 ≥ k=0.10 的 2.89×）；Δ(dep−ind) @k=0.05 末种子户级
  cluster bootstrap CI 不含零。
- S2（PT 效力网格，唯一未测的文献锚）：efficacy ∈ {0.5, 0.6, 0.75, 0.9}
  下 Δ(full−blanket) > 0 全网格全 k（方向稳健）；每 100 人次避免量随
  efficacy 单调增（剂量-响应）。β 保守口径在极值点（0.5/0.9）须同向。
- S3（级联折损注入，Part B 锚）：以皮测放置 0.68 × 治疗启动 0.42 =
  0.2856 为有效覆盖乘子（计划 k → 实际完成 0.2856k，所有通道 cov 同乘）。
  判据 Δ(full−blanket) > 0 全 k 在同乘单调性下**平凡成立**（capture>k ⟹
  capture×ret > k×ret）——本项价值不在判据，在：①绝对量缩减幅度的量化
  （预期 ~0.29× 近似线性）；②"折损-风险无关"假设的披露（若折损与风险
  负相关则靶向优势进一步缩水，正相关则扩大——数据不支持分层折损估计）。

协议：复用 run_sop_deepening.run_seed 20 种子 OOF（锚臂 gate |Δ|<0.002
对 §8.11 归档：ind 0.6428 / exposure 0.6441 / prior_only 0.6443 /
exposure_prior 0.7142 / dep_fixed 0.7202）；SEIR 腿沿用 B1 接触人群
镜像初态（潜伏池重标定到队列 π，年龄/HIV 边际 v4 默认）、确定性 ODE、
成对反事实。经验内容在 capture 曲线（单队列 OOF），平移层零经验内容。

级联锚口径：TBESC-II Part B（用户指定锚：皮测放置 68% / 治疗启动 42%）；
注意为人群级折损（无风险分层信息），注入语义=折损与风险无关。

用法：
    python data/run_bridge_sensitivity.py
输出：
    data/processed/bridge_sensitivity_homeacf_YYYYMMDD.json
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))
sys.path.insert(0, HERE)

import run_sop_deepening as rsd  # noqa: E402
import run_sop_seir_bridge as b1  # noqa: E402

from tb_risk.validation.real_data_infection import (  # noqa: E402
    load_homeacf_contacts,
)
from tb_risk.seir.intervention import (  # noqa: E402
    build_v4_initial_state,
    _default_v4_params,
    _trapz,
    DEFAULT_PT_EFFICACY,
    DEFAULT_PT_REINF_REDUCTION,
)

OUT = os.path.join(HERE, 'processed',
                   'bridge_sensitivity_homeacf_%s.json' % time.strftime(
                       '%Y%m%d'))

K_GRID = (0.05, 0.10, 0.20, 0.30, 0.50)
PT_EFF_GRID = (0.50, 0.60, 0.75, 0.90)
CASCADE_PLACE = 0.68     # Part B 皮测放置完成率（用户指定锚）
CASCADE_TREAT = 0.42     # Part B 治疗启动率（用户指定锚）
RETENTION = CASCADE_PLACE * CASCADE_TREAT   # 0.2856
CAPTURE_ARMS = ('dep_fixed', 'ind', 'exposure_prior')
SEIR_BETA = b1.SEIR_BETA


def pt_params_cov(cov_rho, cov_beta, efficacy=DEFAULT_PT_EFFICACY):
    """PT 参数覆盖（显式 efficacy；通道缩放因子=各自 cov×各自系数）。

    ρ_react ← ρ0(1−cov_rho×efficacy)；β_reinf ← β0(1−cov_beta×0.5)。
    级联折损经 cov 侧注入（调用方传 cov×retention）。
    """
    p = _default_v4_params()
    p['rho_react'] = p['rho_react'] * (1.0 - cov_rho * efficacy)
    p['beta_reinf'] = p['beta_reinf'] * (
        1.0 - cov_beta * DEFAULT_PT_REINF_REDUCTION)
    return p


def main():
    t0 = time.time()
    df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    print('HomeACF: n=%d events=%d households=%d' % (
        len(y), int(y.sum()), len(np.unique(groups))))

    # ---- OOF（复用 §8.11 run_seed，逐位一致）----
    seeds_results = []
    for s in range(rsd.SEED_START, rsd.SEED_START + rsd.N_SEEDS):
        r = rsd.run_seed(s, df, y, groups)
        r['y'] = y
        r['groups'] = groups
        seeds_results.append(r)
        if s % 5 == 0:
            print('seed %2d done (%.0fs)' % (s, time.time() - t0))
    last = seeds_results[-1]
    y_l, g_l = last['y'], last['groups']
    oof_l = last['oof']
    pi_cohort = float(last['pi_true'])

    anchor_expect = {'ind': 0.6428, 'exposure': 0.6441,
                     'prior_only': 0.6443, 'exposure_prior': 0.7142,
                     'dep_fixed': 0.7202}
    anchor_gate = {}
    for a, expect in anchor_expect.items():
        v = float(np.mean([rsd._auroc(r['y'], r['oof'][a])
                           for r in seeds_results]))
        anchor_gate[a] = {'this_run': round(v, 4), 'archive': expect,
                          'reproduced': bool(abs(v - expect) < 0.002)}
    ok = all(g['reproduced'] for g in anchor_gate.values())
    print('\n锚臂 gate: %s' % ('OK' if ok else 'FAIL'))
    if not ok:
        print('!! 锚臂 gate 失败，中止')
        return None

    # ---- S1：capture 曲线（k 含 0.05）----
    capture_per_seed = {k: {a: [] for a in CAPTURE_ARMS} for k in K_GRID}
    for r in seeds_results:
        for k in K_GRID:
            for a in CAPTURE_ARMS:
                capture_per_seed[k][a].append(
                    b1.capture_at_k(r['oof'][a], r['y'], k))
    capture_summary = {}
    for k in K_GRID:
        entry = {}
        for a in CAPTURE_ARMS:
            arr = np.asarray(capture_per_seed[k][a], dtype=float)
            entry[a] = {
                'mean': round(float(np.mean(arr)), 4),
                'sd': round(float(np.std(arr)), 4),
                'min': round(float(np.min(arr)), 4),
                'max': round(float(np.max(arr)), 4),
                'share_exceeds_k': round(float((arr > k).mean()), 3),
                'enrichment_vs_random': round(float(np.mean(arr)) / k, 3),
            }
        capture_summary['%.2f' % k] = entry
    print('\n=== S1 capture(k)（20 种子均值）===')
    for k in K_GRID:
        e = capture_summary['%.2f' % k]
        print('  k=%.2f  dep=%.4f (x%.2f)  ind=%.4f (x%.2f)' % (
            k, e['dep_fixed']['mean'],
            e['dep_fixed']['enrichment_vs_random'],
            e['ind']['mean'], e['ind']['enrichment_vs_random']))

    boot_005 = b1.capture_bootstrap(oof_l, y_l, g_l, 0.05,
                                    rsd.N_BOOTSTRAP, rsd.SEED_START)
    print('  k=0.05 bootstrap: dep−ind %+0.4f CI [%+0.4f,%+0.4f] %s | '
          'dep−rand %+0.4f CI [%+0.4f,%+0.4f] %s' % (
              boot_005['dep_minus_ind']['mean'],
              *boot_005['dep_minus_ind']['bootstrap_ci'],
              '★' if boot_005['dep_minus_ind']['ci_excludes_zero'] else '.',
              boot_005['dep_minus_random']['mean'],
              *boot_005['dep_minus_random']['bootstrap_ci'],
              '★' if boot_005['dep_minus_random'][
                  'ci_excludes_zero'] else '.'))

    # ---- SEIR 基线（B1 同款镜像初态）----
    state_generic = build_v4_initial_state(n_population=b1.SEIR_POPULATION,
                                            random_state=b1.SEIR_SEED)
    initial_state_c, _ = b1.rescale_latent_to_pi(state_generic, pi_cohort)
    base_times, active_base, new_inf_base = b1.seir_run(initial_state_c,
                                                        _default_v4_params())
    cum_base = float(_trapz(active_base, base_times))
    print('\nSEIR 基线：累计 %.0f 病例·天（镜像初态 π=%.3f）' % (
        cum_base, pi_cohort))

    def run_rules(k, cap_dep, cap_ind, efficacy, retention=1.0):
        """同预算下三规则反事实（retention 折损经 cov 侧注入）。"""
        n_courses = int(round(k * b1.SEIR_POPULATION))
        rules = {
            'blanket_random': pt_params_cov(
                k * retention, k * retention, efficacy),
            'host_targeted': pt_params_cov(
                cap_ind * retention, cap_ind * retention, efficacy),
            'full_targeted': pt_params_cov(
                cap_dep * retention, cap_dep * retention, efficacy),
        }
        out = {}
        for name, params in rules.items():
            out[name] = b1.translate_rule(
                initial_state_c, base_times, cum_base, new_inf_base,
                params, n_courses)
        return n_courses, out

    # ---- S2：PT 效力网格 ----
    eff_grid = {}
    for k in K_GRID:
        cap_dep = capture_summary['%.2f' % k]['dep_fixed']['mean']
        cap_ind = capture_summary['%.2f' % k]['ind']['mean']
        per_eff = {}
        for eff in PT_EFF_GRID:
            n_courses, out = run_rules(k, cap_dep, cap_ind, eff)
            d_fb = (out['full_targeted']['averted_case_days']
                    - out['blanket_random']['averted_case_days'])
            entry = {
                'averted_blanket': round(
                    out['blanket_random']['averted_case_days'], 2),
                'averted_host': round(
                    out['host_targeted']['averted_case_days'], 2),
                'averted_full': round(
                    out['full_targeted']['averted_case_days'], 2),
                'delta_full_minus_blanket': round(d_fb, 2),
                'delta_full_minus_host': round(
                    out['full_targeted']['averted_case_days']
                    - out['host_targeted']['averted_case_days'], 2),
                'per_100_courses_full_minus_blanket': round(
                    d_fb / (n_courses / 100.0), 3),
                'direction_positive': bool(d_fb > 0),
            }
            # 保守 β 口径（β 通道按 k，非 capture）极值点检查
            if eff in (0.50, 0.90):
                p_cons = pt_params_cov(cap_dep, k, eff)
                r_cons = b1.translate_rule(
                    initial_state_c, base_times, cum_base, new_inf_base,
                    p_cons, n_courses)
                entry['conservative_beta_delta'] = round(
                    r_cons['averted_case_days']
                    - out['blanket_random']['averted_case_days'], 2)
            per_eff['%.2f' % eff] = entry
        eff_grid['%.2f' % k] = per_eff
    print('\n=== S2 PT 效力网格（Δ(full−blanket) 每 100 人次避免病例·天）===')
    for k in K_GRID:
        row = eff_grid['%.2f' % k]
        cells = []
        for e in ('0.50', '0.60', '0.75', '0.90'):
            cells.append('eff=%s %.1f%s' % (
                e, row[e]['per_100_courses_full_minus_blanket'],
                '' if row[e]['direction_positive'] else ' (!!NEG)'))
        print('  k=%.2f: %s' % (k, ' | '.join(cells)))

    # ---- S3：级联折损注入 ----
    cascade = {}
    for k in K_GRID:
        cap_dep = capture_summary['%.2f' % k]['dep_fixed']['mean']
        cap_ind = capture_summary['%.2f' % k]['ind']['mean']
        n_courses, out = run_rules(k, cap_dep, cap_ind,
                                   DEFAULT_PT_EFFICACY, retention=RETENTION)
        n_eff = int(round(n_courses * RETENTION))
        d_fb = (out['full_targeted']['averted_case_days']
                - out['blanket_random']['averted_case_days'])
        # 无折损对照（eff=0.6 基点）
        _, out0 = run_rules(k, cap_dep, cap_ind, DEFAULT_PT_EFFICACY)
        d0 = (out0['full_targeted']['averted_case_days']
              - out0['blanket_random']['averted_case_days'])
        cascade['%.2f' % k] = {
            'n_courses_planned': n_courses,
            'n_courses_effective': n_eff,
            'averted_blanket': round(
                out['blanket_random']['averted_case_days'], 2),
            'averted_full': round(
                out['full_targeted']['averted_case_days'], 2),
            'delta_full_minus_blanket': round(d_fb, 2),
            'delta_full_minus_blanket_noretention': round(d0, 2),
            'retention_ratio_of_delta': round(
                d_fb / d0, 3) if abs(d0) > 1e-9 else None,
            'direction_positive': bool(d_fb > 0),
        }
    print('\n=== S3 级联折损（retention=%.4f：%.2f×%.2f）Δ(full−blanket) '
          '病例·天 ===' % (RETENTION, CASCADE_PLACE, CASCADE_TREAT))
    for k in K_GRID:
        e = cascade['%.2f' % k]
        print('  k=%.2f: Δ=%0.1f（无折损 %0.1f，比值 %.3f）%s' % (
            k, e['delta_full_minus_blanket'],
            e['delta_full_minus_blanket_noretention'],
            e['retention_ratio_of_delta'],
            '★' if e['direction_positive'] else ' (!!NEG)'))

    # ---- 判决汇总 ----
    s1_monotone = (capture_summary['0.05']['dep_fixed'][
                       'enrichment_vs_random']
                   >= capture_summary['0.10']['dep_fixed'][
                       'enrichment_vs_random'])
    s1_ci = boot_005['dep_minus_ind']['ci_excludes_zero']
    s2_all_pos = all(
        eff_grid['%.2f' % k]['%.2f' % e]['direction_positive']
        for k in K_GRID for e in PT_EFF_GRID)
    s2_cons_pos = all(
        eff_grid['%.2f' % k]['%.2f' % e]['conservative_beta_delta'] > 0
        for k in K_GRID for e in (0.50, 0.90))
    s2_monotone_eff = all(
        eff_grid['%.2f' % k]['%.2f' % e][
            'per_100_courses_full_minus_blanket']
        <= eff_grid['%.2f' % k]['%.2f' % e2][
            'per_100_courses_full_minus_blanket']
        for k in K_GRID
        for e, e2 in zip(PT_EFF_GRID[:-1], PT_EFF_GRID[1:]))
    s3_all_pos = all(
        cascade['%.2f' % k]['direction_positive'] for k in K_GRID)
    verdicts = {
        'S1_enrichment_monotone_at_005': bool(s1_monotone),
        'S1_dep_minus_ind_ci_excludes_zero_at_005': bool(s1_ci),
        'S2_direction_positive_all_grid': bool(s2_all_pos),
        'S2_conservative_beta_positive_extremes': bool(s2_cons_pos),
        'S2_per100_monotone_in_efficacy': bool(s2_monotone_eff),
        'S3_direction_positive_all_k': bool(s3_all_pos),
        'S3_note': '同乘单调性下判据平凡；价值在绝对量缩减与假设披露',
    }
    print('\n=== 预声明命题判决 ===')
    for kk, vv in verdicts.items():
        print('  %s: %s' % (kk, vv))

    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'bridge_sensitivity_homeacf_v1',
        'status': 'B1 桥接敏感性补强（预声明 S1/S2/S3）；确定性平移层，'
                  '经验内容在 capture 曲线（HomeACF 单队列 OOF）',
        'design': {
            'base': 'B1 桥接（sop_seir_bridge_homeacf_20260906.json）',
            'anchor_gate': anchor_gate,
            'k_grid': list(K_GRID),
            'pt_efficacy_grid': list(PT_EFF_GRID),
            'cascade_anchor': {
                'source': 'TBESC-II Part B（用户指定锚）',
                'placement_rate': CASCADE_PLACE,
                'treatment_initiation_rate': CASCADE_TREAT,
                'retention': round(RETENTION, 4),
                'semantics': '有效覆盖 = 计划覆盖 × retention（人群级，'
                             '折损-风险无关假设）',
            },
            'seir': '沿用 B1：镜像初态（π=%.3f 重标定）、确定性 ODE、'
                    '730 天、β=0.3、成对反事实' % pi_cohort,
        },
        'hypotheses': {
            'S1': 'k=0.05 富集 ≥ k=0.10（单调）；Δ(dep−ind)@0.05 CI 不含零',
            'S2': 'Δ(full−blanket)>0 全效力网格全 k；每100人次随效力单调；'
                  '保守 β 极值点同向',
            'S3': 'retention=0.2856 下 Δ>0 全 k（平凡，同乘单调）；绝对量'
                  '缩减幅度 ~retention',
        },
        'verdicts': verdicts,
        'capture_summary': capture_summary,
        'capture_bootstrap_k005': {
            'dep_minus_ind': boot_005['dep_minus_ind'],
            'dep_minus_random': boot_005['dep_minus_random'],
        },
        'pt_efficacy_grid': eff_grid,
        'cascade_retention': cascade,
        'honest_boundaries': [
            '级联折损为人群级锚（无风险分层信息）——折损-风险无关假设；'
            '若折损与风险负相关，靶向优势进一步缩水',
            'S3 判据在同乘单调性下平凡成立（capture>k ⟹ capture×ret>k×ret），'
            '价值在绝对量缩减量化与假设披露',
            '绝对水平仍为通用参数场景（β=0.3、初态仅潜伏池重标定、年龄/HIV'
            '边际 v4 默认）——只可引相对量与对比结构',
            'k=0.50 已饱和（B1 边界沿用）；k=0.05 点位 bootstrap 含天花板'
            '效应的镜像披露（capture≤1）',
        ],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
