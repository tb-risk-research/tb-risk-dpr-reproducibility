#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ERASE-TB 外验预注册把握度计算（2026-09-05；v2 同日增补；v3 2026-09-06）

目的：为 SOP（时序结局先证，model_boundary_clinical_validation.md §8）
的 ERASE-TB 前瞻外验预注册提供检出 ΔAUROC 所需事件数的把握度曲线。

方法（经验锚缩放，非参数假设）：
  - 锚 = HomeACF 判决性实验（household_temporal_screening_20260825.json）：
    n=2725 / 359 事件 / 户级 cluster bootstrap ×2000 的配对差 CI
    [+0.0312, +0.0870] → SE_diff 锚 = CI 宽 / (2×1.96)；
  - 缩放律：配对差 SE ∝ √(1/n1 + 1/n0)（Hanley-McNeil 信息量），
    事件分数 π 固定时 1/n1 + 1/n0 = 1/(n1·(1−π))（闭式）；
  - 检验水准：双侧 α=0.05（与全部既有归档的 "95% CI 不含零" 判定
    口径一致）；把握度 80% / 90%。

规划场景（ERASE-TB 文献口径：786 户 / 2,109 名家庭接触者，
IGRA 阳性率 ~30%）：
  - 主分析（感染终点）：IGRA 阳性，π≈0.30；
  - 次分析（病程终点）：TB 发病，π≈2%-5%（把握度受限，如实标注）。

v2 增补（协议修订 v1.1 R4，数据接触前；v1 场景数值不变、只增不改）：
  - π 敏感性：主分析终点 π ∈ {0.20, 0.30, 0.50}——π 是单一最大规划
    假设（事件数随 π 线性缩放，触发线随 π 移动，见 trigger_line_by_pi）；
  - 锚迁移 caveat：规划 Δ=0.058 为 HomeACF TST 终点效应，迁移至 IGRA
    含 assay 迁移假设（IGRA 特异性更高/无 BCG 交叉，户内聚类信号
    强度可能不同），不可从现有数据检验；
  - SE 缩放律 caveat：锚 SE 含 20 种子训练方差分量，√(1/n1+1/n0)
    只缩放采样分量——种子分量不随 n 缩放，planning 略偏乐观。

v3 增补（协议修订 v1.2 R5，数据接触前；v1/v2 场景数值不变、只增不改）：
  - 锚 SE 种子分解：锚归档含 20 逐种子配对差（exposure_prior − ind），
    全部 >0（无符号翻转）；种子分量 SE = SD(逐种子 Δ)/√20，与
    bootstrap 采样分量 SE 平方和合并为保守合并 SE；
  - 触发线只收紧：合并 SE 口径下重算 required_events，主分析
    π=0.30 触发线由 211 上抬（见 trigger_lines_v3）；
  - 动机：CXR 覆盖模拟 P2 种子稳健性腿已实证"单折分种子方向翻转"
    （seed 13 富集 0.85×，data/processed/cxr_coverage_simulation_
    20260906.json）——"单种子不稳健"是本项目方法论事实，锚 SE 须
    显式分解后在数据接触前入档。

v1 输出（不变）：data/processed/erase_tb_prereg_power_20260905.json
v2 输出（不变）：data/processed/erase_tb_prereg_power_20260905_v2.json
v3 输出（本脚本现行）：data/processed/erase_tb_prereg_power_20260906_v3.json
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'processed', 'erase_tb_prereg_power_20260906_v3.json')

Z_ALPHA = 1.959964  # 双侧 5%
Z_POWER = {0.80: 0.841621, 0.90: 1.281552}

# ---- 锚（从归档 JSON 读取，防止手抄错）----
ANCHOR_PATH = os.path.join(HERE, 'processed',
                           'household_temporal_screening_20260825.json')


def _hm_marginal_se(auroc, n_pos, n_neg):
    """Hanley-McNeil 单评分 AUROC 标准误（仅作锚合理性旁证）。"""
    q1 = auroc / (2.0 - auroc)
    q2 = 2.0 * auroc * auroc / (1.0 + auroc)
    var = (auroc * (1 - auroc)
           + (n_pos - 1) * (q1 - auroc ** 2)
           + (n_neg - 1) * (q2 - auroc ** 2)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def main():
    with open(ANCHOR_PATH, encoding='utf-8') as f:
        anchor_doc = json.load(f)
    acc = anchor_doc['design']['acceptance']['vs_ind_baseline']
    ci_lo, ci_hi = acc['ci']
    se_anchor = (ci_hi - ci_lo) / (2.0 * Z_ALPHA)
    n1_a = int(anchor_doc['design']['n_events'])          # 359
    n0_a = int(anchor_doc['design']['n'] - n1_a)          # 2366
    K = 1.0 / n1_a + 1.0 / n0_a
    auroc_full = anchor_doc['arm_summary']['exposure_prior:RF']['mean_auroc']

    hm = _hm_marginal_se(auroc_full, n1_a, n0_a)

    def se_diff(n_pos, pi):
        """配对差 SE（经验锚缩放；n_neg = n_pos(1-π)/π）。"""
        info = 1.0 / (n_pos * (1.0 - pi))
        return se_anchor * math.sqrt(info / K)

    def detectable_delta(n_pos, pi, power):
        return (Z_ALPHA + Z_POWER[power]) * se_diff(n_pos, pi)

    def required_events(delta, pi, power):
        """检出 delta 所需阳性事件数（事件分数 π 下）。"""
        se_t = delta / (Z_ALPHA + Z_POWER[power])
        info_req = K * (se_t / se_anchor) ** 2
        n_pos = 1.0 / (info_req * (1.0 - pi))
        return math.ceil(n_pos)

    def achieved_power(delta, n_pos, pi):
        return float(0.5 * (1.0 + math.erf(
            (delta / se_diff(n_pos, pi) - Z_ALPHA) / math.sqrt(2.0))))

    # ---- 场景 ----
    N_COHORT = 2109
    scenarios = {}
    for pi, key in ((0.30, 'igra_primary_pi30'),
                    (0.20, 'igra_pi_sens_20'),
                    (0.50, 'igra_pi_sens_50'),
                    (0.05, 'progression_pi5'),
                    (0.02, 'progression_pi2')):
        n_pos = int(round(N_COHORT * pi))
        scenarios[key] = {
            'n_total': N_COHORT, 'event_fraction': pi, 'n_events': n_pos,
            'se_diff': round(se_diff(n_pos, pi), 6),
            'detectable_delta_80pct': round(detectable_delta(n_pos, pi, 0.80), 4),
            'detectable_delta_90pct': round(detectable_delta(n_pos, pi, 0.90), 4),
            'power_at_planning_delta_058': round(
                achieved_power(0.058, n_pos, pi), 4),
        }

    # ---- v2：主分析 π 敏感性 + 随 π 移动的判决触发线 ----
    # 触发线口径：required_events(Δ=0.058, π, 80%)——事件数阈值本身随 π
    # 移动（π 大 → 同事件数下阴性更少 → 信息更少 → 需要更多事件）；
    # 达到该事件数所需样本量 = 触发线 / π。
    pi_sensitivity = []
    for pi in (0.20, 0.30, 0.50):
        trig = required_events(0.058, pi, 0.80)
        pi_sensitivity.append({
            'event_fraction': pi,
            'n_events_full_cohort': int(round(N_COHORT * pi)),
            'power_at_planning_delta_058': round(
                achieved_power(0.058, int(round(N_COHORT * pi)), pi), 4),
            'trigger_line_events_80pct': trig,
            'n_total_needed_for_trigger': int(math.ceil(trig / pi)),
        })

    # ---- 把握度曲线（主分析 π=0.30）----
    curve = []
    for n_pos in (50, 75, 100, 150, 210, 300, 421, 633, 800, 1000):
        curve.append({
            'n_events': n_pos,
            'n_total_at_pi30': int(round(n_pos / 0.30)),
            'se_diff': round(se_diff(n_pos, 0.30), 6),
            'detectable_delta_80pct': round(detectable_delta(n_pos, 0.30, 0.80), 4),
            'detectable_delta_90pct': round(detectable_delta(n_pos, 0.30, 0.90), 4),
        })

    # ---- 检出目标 Δ 所需事件数 ----
    required = {}
    for delta in (0.058, 0.03, 0.02):
        dkey = f'delta_{delta:.3f}'
        required[dkey] = {}
        for power in (0.80, 0.90):
            pkey = f'power_{int(power * 100)}'
            required[dkey][pkey] = {
                'pi30_events': required_events(delta, 0.30, power),
                'pi132_events': required_events(delta, 0.1319, power),
            }

    # ---- 主分析最小事件数（预注册触发线）----
    # 检出规划值 Δ=0.058（HomeACF 户级效应）@80% 把握度、π=0.30
    min_events_primary = required_events(0.058, 0.30, 0.80)

    # ---- v3：锚 SE 种子分解（协议修订 v1.2 R5，数据接触前）----
    # 锚归档逐种子条目：per-seed 配对差 Δ_s = auroc(exposure_prior) −
    # auroc(ind)。种子分量 SE = SD(Δ_s)/√n_seeds（种子均值的训练方差）；
    # bootstrap 采样分量 = se_anchor（户级重采样、分数固定，不含训练
    # 随机性）；保守合并 SE = √(采样² + 种子²)。
    seed_entries = anchor_doc['seeds']
    seed_deltas = np.array([
        s['arms']['exposure_prior:RF']['auroc'] - s['arms']['ind:RF']['auroc']
        for s in seed_entries])
    n_seeds_anchor = len(seed_deltas)
    seed_component_se = float(seed_deltas.std(ddof=1) / math.sqrt(n_seeds_anchor))
    combined_se_anchor = math.sqrt(se_anchor ** 2 + seed_component_se ** 2)
    seed_flip_count = int((seed_deltas <= 0).sum())

    def required_events_v3(delta, pi, power, se_a):
        se_t = delta / (Z_ALPHA + Z_POWER[power])
        info_req = K * (se_t / se_a) ** 2
        return math.ceil(1.0 / (info_req * (1.0 - pi)))

    def detectable_delta_v3(n_pos, pi, power, se_a):
        info = 1.0 / (n_pos * (1.0 - pi))
        se_d = se_a * math.sqrt(info / K)
        return (Z_ALPHA + Z_POWER[power]) * se_d

    trigger_lines_v3 = []
    for pi in (0.20, 0.30, 0.50):
        trig_v2 = required_events(0.058, pi, 0.80)
        trig_v3 = required_events_v3(0.058, pi, 0.80, combined_se_anchor)
        trigger_lines_v3.append({
            'event_fraction': pi,
            'trigger_line_events_v2_sampling_only': trig_v2,
            'trigger_line_events_v3_combined': trig_v3,
            'only_tightening': trig_v3 >= trig_v2,
        })

    min_events_primary_v3 = required_events_v3(
        0.058, 0.30, 0.80, combined_se_anchor)
    detect_633_v3 = detectable_delta_v3(633, 0.30, 0.80, combined_se_anchor)
    detect_633_v2 = detectable_delta(633, 0.30, 0.80)

    seed_decomposition = {
        'anchor_seeds': n_seeds_anchor,
        'per_seed_delta_min': round(float(seed_deltas.min()), 6),
        'per_seed_delta_max': round(float(seed_deltas.max()), 6),
        'per_seed_delta_mean': round(float(seed_deltas.mean()), 6),
        'seeds_with_sign_flip': seed_flip_count,
        'sampling_component_se': round(se_anchor, 6),
        'seed_component_se': round(seed_component_se, 6),
        'seed_to_sampling_ratio': round(seed_component_se / se_anchor, 4),
        'combined_se': round(combined_se_anchor, 6),
        'aggregation_note': (
            '逐种子 Δ 均值 %.4f ≠ 池化 OOF 判决均值 %.4f（池化口径的 '
            'AUROC 聚合非线性）——种子分量按逐种子离散度（SD=%.4f）'
            '度量，与均值口径差无关' % (
                float(seed_deltas.mean()), float(acc['mean']),
                float(seed_deltas.std(ddof=1)))),
        'interpretation': (
            '锚 SE 种子稳健：20 逐种子配对差全部 >0（无符号翻转），种子'
            '分量 %.4f 仅为采样分量 %.4f 的 %.0f%%，合并 SE %.4f 较采样'
            '分量仅上浮 %.1f%%——v1/v2 锚对种子变异稳健，v2 缩放律 '
            'caveat 的种子分量在此量化收口'
            % (seed_component_se, se_anchor,
               100 * seed_component_se / se_anchor, combined_se_anchor,
               100 * (combined_se_anchor / se_anchor - 1))),
    }

    out = {
        'date': '2026-09-06',
        'experiment': 'erase_tb_prereg_power_v3',
        'v3_changes': (
            '协议修订 v1.2 R5（数据接触前）：锚 SE 种子分解（20 逐种子'
            '配对差全部 >0 无翻转；种子分量 = SD/√20 与 bootstrap 采样'
            '分量平方和合并）+ 合并 SE 口径触发线（只收紧：π=0.30 触发'
            '线 %d→%d）；v1/v2 场景数值逐位不变（归档保留于 '
            'erase_tb_prereg_power_20260905.json / _v2.json）'
            % (min_events_primary, min_events_primary_v3)),
        'v2_changes': '协议修订 v1.1 R4（数据接触前）：新增主分析 π 敏感性'
                      '（igra_pi_sens_20/50 + pi_sensitivity）与锚迁移/种子'
                      '分量双 caveat；v1 场景数值不变（igra_primary_pi30 / '
                      'progression_* 与 v1 逐位一致）；v1 归档保留于 '
                      'erase_tb_prereg_power_20260905.json',
        'purpose': 'SOP（时序结局先证）ERASE-TB 前瞻外验预注册把握度',
        'method': {
            'anchor': 'household_temporal_screening_20260825.json '
                      '（HomeACF 判决性实验，vs_ind_baseline 配对差）',
            'se_diff_anchor': round(se_anchor, 6),
            'anchor_n_events': n1_a,
            'anchor_n_total': int(anchor_doc['design']['n']),
            'anchor_delta_mean': round(acc['mean'], 6),
            'anchor_ci': [round(ci_lo, 6), round(ci_hi, 6)],
            'scaling': 'SE ∝ sqrt(1/n1 + 1/n0)；π 固定时闭式 '
                       '1/n1+1/n0 = 1/(n1(1-π))',
            'test': '双侧 α=0.05（与归档 "95% CI 不含零" 判定口径一致）',
            'hm_marginal_se_sanity': round(hm, 6),
            'hm_note': 'Hanley-McNeil 单评分边际 SE（锚 A=%.4f）仅作旁证：'
                       '配对差 SE（%.6f）< 边际 SE 符合同集配对正相关预期'
                       % (auroc_full, se_anchor),
        },
        'planning_assumptions': {
            'cohort': 'ERASE-TB 文献口径 786 户 / 2,109 名家庭接触者',
            'igra_prevalence': 0.30,
            'planning_delta': 0.058,
            'planning_delta_source': 'HomeACF 户级判决性效应（§8.2），'
                                     '粒度衰减提示前瞻队列应预期 ≤ 该值',
            'caveat': '缩放律假设 ERASE-TB 的户内相关结构与 HomeACF '
                      '同量级；把握度按主分析（IGRA 感染终点）计，'
                      'G1 时序粒度门通过为前提',
            'assay_migration_caveat': '规划 Δ=0.058 为 HomeACF TST 终点'
                      '效应，迁移至 IGRA 含 assay 迁移假设（IGRA 特异性'
                      '更高/无 BCG 交叉，户内聚类信号强度可能不同），'
                      '不可从现有数据检验——故判决门槛以事件数触发线'
                      '（prereg_minimum_events_primary）为主口径，'
                      '规划 Δ 仅用于功效预期',
            'seed_variance_caveat': '锚 SE 含 20 种子训练方差分量，'
                      'sqrt(1/n1+1/n0) 缩放只缩放采样分量，种子分量不随 '
                      'n 缩放——planning 略偏乐观；事件数充裕场景（主分析）'
                      '影响可忽略，欠功效场景（次分析 B）实际功效可能'
                      '低于表值',
            'pi_sensitivity_note': 'π=0.30 为单一最大规划假设：事件数随 '
                      'π 线性缩放，判决触发线随 π 移动（pi_sensitivity.'
                      'trigger_line_events_80pct：π=0.20 → 更低、π=0.50 → '
                      '更高），达到触发线所需样本量 = 触发线/π',
        },
        'scenarios': scenarios,
        'pi_sensitivity': pi_sensitivity,
        'power_curve_pi30': curve,
        'required_events': required,
        'prereg_minimum_events_primary': min_events_primary,
        'prereg_minimum_note': '主分析（IGRA 终点）阳性事件 ≥ %d 才进入'
                               '判决口径（检出 Δ=0.058 @80%%/双侧5%%，'
                               'π=0.30）；不足则只报 CI 不判决'
                               % min_events_primary,
        'v3_seed_decomposition': seed_decomposition,
        'v3_trigger_lines': {
            'method': ('合并 SE（采样² + 种子²）口径重算 required_events'
                       '（Δ=0.058 @80%/双侧5%）；只收紧不放松'),
            'rows': trigger_lines_v3,
            'primary_pi30_sampling_only': min_events_primary,
            'primary_pi30_combined': min_events_primary_v3,
            'detectable_delta_633events_pi30': {
                'v2_sampling_only': round(detect_633_v2, 4),
                'v3_combined': round(detect_633_v3, 4),
            },
            'ruling': ('R5 后判决触发线 = max(%d, required_events(0.058, '
                       'π̂, 0.80) 以合并 SE 计)——协议 v1.1 R4 的自适应'
                       '规则锚 SE 从采样分量升级为合并 SE，只收紧'
                       % min_events_primary_v3),
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print('=== 锚 ===')
    print('SE_diff 锚 = %.6f（HomeACF CI 宽/3.92）' % se_anchor)
    print('HM 边际 SE 旁证 = %.6f' % hm)
    print()
    print('=== 场景（检出 Δ @80%/90% 双侧5%）===')
    for k, v in scenarios.items():
        print('%-22s 事件 %4d：80%% Δ≥%.4f / 90%% Δ≥%.4f / '
              'Δ=0.058 功效 %.1f%%'
              % (k, v['n_events'], v['detectable_delta_80pct'],
                 v['detectable_delta_90pct'],
                 100 * v['power_at_planning_delta_058']))
    print()
    print('=== 所需事件数 ===')
    for dkey, pv in required.items():
        for pkey, v in pv.items():
            print('%s %s: π30 → %d 事件; π13.2 → %d 事件'
                  % (dkey, pkey, v['pi30_events'], v['pi132_events']))
    print()
    print('=== v2：主分析 π 敏感性（全队列 2109 人）===')
    for row in pi_sensitivity:
        print('π=%.2f → %4d 事件，Δ=0.058 功效 %.1f%%，'
              '触发线 %d 事件（需 ≥%d 人）'
              % (row['event_fraction'], row['n_events_full_cohort'],
                 100 * row['power_at_planning_delta_058'],
                 row['trigger_line_events_80pct'],
                 row['n_total_needed_for_trigger']))
    print()
    print('主分析最小事件数（预注册触发线，π=0.30）= %d' % min_events_primary)
    print()
    print('=== v3：锚 SE 种子分解（协议 v1.2 R5，数据接触前）===')
    print('锚 20 逐种子配对差：min %.4f / max %.4f / 翻转 %d 个'
          % (seed_deltas.min(), seed_deltas.max(), seed_flip_count))
    print('采样分量 SE=%.6f | 种子分量 SE=%.6f（%.0f%%）| 合并 SE=%.6f（+%.1f%%）'
          % (se_anchor, seed_component_se,
             100 * seed_component_se / se_anchor, combined_se_anchor,
             100 * (combined_se_anchor / se_anchor - 1)))
    print(seed_decomposition['interpretation'])
    for row in trigger_lines_v3:
        print('π=%.2f 触发线：v2 采样口径 %d → v3 合并口径 %d（%s）'
              % (row['event_fraction'],
                 row['trigger_line_events_v2_sampling_only'],
                 row['trigger_line_events_v3_combined'],
                 '只收紧' if row['only_tightening'] else '异常：放松'))
    print('主分析 633 事件可检出 Δ：v2 %.4f → v3 %.4f'
          % (detect_633_v2, detect_633_v3))
    print('归档 →', OUT)


if __name__ == '__main__':
    main()
