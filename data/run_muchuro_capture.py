#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOP×SEIR 桥接第二队列（B1-M，2026-09-07）：Muchuro capture 曲线 + 桥接重跑。

地位：B1 桥接全部押在 HomeACF 单队列上，本脚本以 Muchuro（Uganda 家庭
接触者，n=352/59 户，IGRA 终点阳 32.7%，tbindex_id 户键）做**队列稳健性
检验**。原清单指定 Aibana（12,648 例）经核实 27 列无户键（§8.12 纠偏
记录），先证块不可构造，故以 Muchuro 替代——它是数据在手、带户键、
感染终点的唯一可用第二队列。

命题（预声明，先于运行）：
- M1（锚复现 gate）：host/exp/full 三臂 pooled AUROC 复现 §8.9 归档
  （RF：0.5887/0.6585/0.7247，|Δ|<0.005——同 _make_model、同折内中位数
  填补、同 StratifiedGroupKFold(5)×20 种子平均 OOF）。
- M2（capture 队列稳健性，主命题）：部署公式（w=7 跨队列迁移、k=2、
  π=队列率）OOF 的 capture(k)=P(top-k|y=1) 满足
  (a) 20 种子均值 capture_dep(k) > k，∀k∈{0.10,0.20,0.30}；
  (b) capture_dep(k) > capture_host(k) 同 k（对宿主臂，即现行优先层
      特征族）——**方向性为主判据**：n=352/59 户的小样本下 CI 不含零
      不作硬判据，如实报告。
- M3（prior 块增量读数）：RF(full+prior) − RF(full) AUROC 方向性
  （IGRA 终点 + 户级分组满足 P5 条件 a/b/d；小户均值 prior_n 小，
  增益预期小于 HomeACF，无定量预期）。
- M4（SEIR 平移，方向性）：以 Muchuro capture 数值注入 B1 平移层
  （镜像初态 π=0.327 重标定、PT 效力 0.60），Δ(full_targeted −
  blanket_random) > 0 ∀k∈{0.10,0.20,0.30}——平移层零经验内容，
  方向由 capture 顺序决定（平凡单调，披露口径）。

诚实边界：
- w=7 为 HomeACF 网格校准的跨队列迁移，未在 Muchuro 本地重校准——
  检验的是"部署配方跨队列迁移"而非"每队列最优"；k=2 固定 + π=队列率
  （部署者可估计）；dep_eb（折内 EB k̂/π̂）作敏感性臂；
- 小样本（59 户）：capture bootstrap CI 宽，方向性为主判据；k=0.50
  点位近饱和仅披露；
- IGRA 终点阳 32.7% → SEIR 镜像初态潜伏池 13.9× 重标定（S 池 0.99→
  ~0.68），绝对水平为设定场景不可作预测引用，仅相对量；
- Muchuro 无 HIV+ 大层（宿主臂 HIV 稀疏），分层读数不做。

用法：
    python data/run_muchuro_capture.py
输出：
    data/processed/muchuro_capture_bridge_YYYYMMDD.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))
sys.path.insert(0, HERE)

import run_sop_deepening as rsd  # noqa: E402
import run_sop_seir_bridge as b1  # noqa: E402
import run_uga_dual_exposure_gradient as uga  # noqa: E402

from tb_risk.validation.real_data_pi import _make_model  # noqa: E402
from tb_risk.validation.household_temporal import (  # noqa: E402
    assign_screening_order,
    prior_features,
)
from tb_risk.seir.intervention import (  # noqa: E402
    build_v4_initial_state,
    _default_v4_params,
    _trapz,
    DEFAULT_PT_EFFICACY,
)

OUT = os.path.join(HERE, 'processed',
                   'muchuro_capture_bridge_%s.json' % time.strftime('%Y%m%d'))

K_GRID = (0.10, 0.20, 0.30, 0.50)
N_SEEDS = 20
N_SPLITS = 5
N_BOOTSTRAP = 2000
DEP_W = 7.0
DEP_K = 2.0
PROB_EPS = 1e-6
COLS_PRIOR = ['prior_n', 'prior_pos', 'prior_rate', 'prior_screened']

# 锚臂 pooled AUROC（§8.9 归档 RF 臂）
ANCHOR = {'host': 0.5887, 'exp': 0.6585, 'full': 0.7247}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _auroc(y, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y, dtype=int),
                               np.asarray(scores, dtype=float)))


def fit_arm_oof(X, y, groups, seed):
    """单臂单种子组感知 OOF（折内中位数填补，§8.9 同款）。"""
    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=seed)
    for tr, te in cv.split(X, y, groups):
        med = np.nanmedian(X[tr], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(X[tr]), med, X[tr])
        Xte = np.where(np.isnan(X[te]), med, X[te])
        m = _make_model('random_forest', seed)
        m.fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
    return oof


def capture_bootstrap_mu(oof, y, groups, k, n_boot, seed,
                         arms=('dep_fixed', 'host', 'full')):
    """Muchuro 版 capture bootstrap（B1 函数硬编码 HomeACF 臂名，本地实现）。

    随机规则同重采样内配对；Δ(dep−host) 与 Δ(dep−random) 输出。
    """
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    caps = {a: [] for a in arms}
    caps_rand = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in pick])
        yr = y[rows]
        if yr.sum() == 0:
            continue
        for a in arms:
            caps[a].append(b1.capture_at_k(oof[a][rows], yr, k))
        caps_rand.append(b1.random_capture_at_k(yr, k, rng))
    out = {}
    for a in arms:
        arr = np.asarray(caps[a], dtype=float)
        out[a] = {
            'mean': float(np.mean(arr)),
            'bootstrap_ci': [float(np.percentile(arr, 2.5)),
                             float(np.percentile(arr, 97.5))],
        }
    arr_r = np.asarray(caps_rand, dtype=float)
    out['random'] = {
        'mean': float(np.mean(arr_r)),
        'bootstrap_ci': [float(np.percentile(arr_r, 2.5)),
                         float(np.percentile(arr_r, 97.5))],
    }
    for name, (a, b) in {'dep_minus_host': ('dep_fixed', 'host'),
                         'dep_minus_full': ('dep_fixed', 'full')}.items():
        deltas = np.asarray(caps[a]) - np.asarray(caps[b])
        out[name] = b1._ci_block(deltas)
    deltas = np.asarray(caps['dep_fixed']) - arr_r
    out['dep_minus_random'] = b1._ci_block(deltas)
    return out


def main():
    t0 = time.time()
    d = uga.load_muchuro()
    d = d.dropna(subset=['y']).reset_index(drop=True)
    y = d['y'].astype(int).to_numpy()
    groups = d['group'].to_numpy()
    pi_cohort = float(y.mean())
    print('Muchuro: n=%d events=%d households=%d  π=%.4f' % (
        len(y), int(y.sum()), len(np.unique(groups)), pi_cohort))

    # prior block 构造（重命名复用 household_temporal 接口）
    hh = pd.DataFrame({'record_id': groups, 'tst_pos10': y})
    arms_cols = {
        'host': uga.MU_ARMS['host'],
        'exp': uga.MU_ARMS['exp'],
        'full': uga.MU_ARMS['full'],
    }
    # 稳定顺序的特征列并集
    feat_cols = []
    seen = set()
    for c in sum(arms_cols.values(), []):
        if c not in seen:
            seen.add(c)
            feat_cols.append(c)
    X_all = d[feat_cols].to_numpy(dtype=float)
    col_ix = {c: i for i, c in enumerate(feat_cols)}

    def X_arm(arm):
        return X_all[:, [col_ix[c] for c in arms_cols[arm]]]

    seeds_oof = []
    for s in range(N_SEEDS):
        order = assign_screening_order(hh, seed=s, mode='random')
        pf = prior_features(hh, order)
        prior_n = pf['prior_n'].to_numpy(dtype=float)
        prior_pos = pf['prior_pos'].to_numpy(dtype=float)
        X_full_prior = np.column_stack([
            X_arm('full'), prior_n, prior_pos,
            pf['prior_rate'].to_numpy(dtype=float),
            pf['prior_screened'].to_numpy(dtype=float)])
        oof = {
            'host': fit_arm_oof(X_arm('host'), y, groups, s),
            'exp': fit_arm_oof(X_arm('exp'), y, groups, s),
            'full': fit_arm_oof(X_arm('full'), y, groups, s),
            'full_prior': fit_arm_oof(X_full_prior, y, groups, s),
        }
        r_hat = (prior_pos + DEP_K * pi_cohort) / (prior_n + DEP_K)
        oof['dep_fixed'] = np.clip(
            _sigmoid(_logit(oof['full']) + DEP_W * (r_hat - pi_cohort)),
            PROB_EPS, 1.0 - PROB_EPS)
        seeds_oof.append({
            'seed': s, 'oof': oof,
            'prior_n': prior_n, 'prior_pos': prior_pos,
        })
        if s % 5 == 0:
            print('seed %2d done (%.0fs)' % (s, time.time() - t0))

    # ---- M1 锚 gate（20 种子平均 OOF = pooled 口径）----
    avg = {a: np.mean([r['oof'][a] for r in seeds_oof], axis=0)
           for a in ('host', 'exp', 'full')}
    anchor_gate = {}
    for a, expect in ANCHOR.items():
        v = _auroc(y, avg[a])
        anchor_gate[a] = {'this_run': round(v, 4), 'archive': expect,
                          'reproduced': bool(abs(v - expect) < 0.005)}
    print('\n=== M1 锚 gate（pooled AUROC vs §8.9）===')
    for a, g in anchor_gate.items():
        print('  %-5s %.4f vs %.4f  %s' % (
            a, g['this_run'], g['archive'],
            'OK' if g['reproduced'] else 'FAIL'))
    if not all(g['reproduced'] for g in anchor_gate.values()):
        print('!! 锚 gate 失败，中止')
        return None

    # ---- M3 prior 增量（AUROC）----
    m3_deltas = [_auroc(y, r['oof']['full_prior']) - _auroc(y, r['oof']['full'])
                 for r in seeds_oof]
    m3 = {
        'delta_full_prior_minus_full_mean': round(float(np.mean(m3_deltas)), 4),
        'per_seed_min': round(float(np.min(m3_deltas)), 4),
        'per_seed_max': round(float(np.max(m3_deltas)), 4),
        'share_positive': round(float((np.asarray(m3_deltas) > 0).mean()), 3),
    }
    print('\n=== M3 prior 块增量（逐种子 ΔAUROC full_prior−full）===')
    print('  mean %+.4f  share+ %.2f  [%+.4f, %+.4f]' % (
        m3['delta_full_prior_minus_full_mean'], m3['share_positive'],
        m3['per_seed_min'], m3['per_seed_max']))

    # ---- M2 capture 曲线（逐种子）----
    capture_arms = ('dep_fixed', 'host', 'full', 'full_prior')
    capture_per_seed = {k: {a: [] for a in capture_arms} for k in K_GRID}
    for r in seeds_oof:
        for k in K_GRID:
            for a in capture_arms:
                capture_per_seed[k][a].append(
                    b1.capture_at_k(r['oof'][a], y, k))
    capture_summary = {}
    for k in K_GRID:
        entry = {}
        for a in capture_arms:
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
    print('\n=== M2 capture(k)（20 种子均值，random=k）===')
    for k in K_GRID:
        e = capture_summary['%.2f' % k]
        print('  k=%.2f  dep=%.3f (x%.2f)  host=%.3f (x%.2f)  '
              'full=%.3f (x%.2f)' % (
                  k, e['dep_fixed']['mean'],
                  e['dep_fixed']['enrichment_vs_random'],
                  e['host']['mean'], e['host']['enrichment_vs_random'],
                  e['full']['mean'], e['full']['enrichment_vs_random']))

    # ---- M2 bootstrap（末种子，户级 cluster ×2000）----
    last = seeds_oof[-1]['oof']
    capture_boot = {}
    for k in (0.10, 0.20, 0.30):
        capture_boot['%.2f' % k] = capture_bootstrap_mu(
            last, y, groups, k, N_BOOTSTRAP, 0)
    print('\n=== M2 末种子 bootstrap（Δ capture，59 户 cluster ×%d）==='
          % N_BOOTSTRAP)
    for k in (0.10, 0.20, 0.30):
        e = capture_boot['%.2f' % k]
        print('  k=%.2f  dep−host %+0.4f CI [%+0.4f,%+0.4f] %s | '
              'dep−rand %+0.4f CI [%+0.4f,%+0.4f] %s' % (
                  k,
                  e['dep_minus_host']['mean'],
                  *e['dep_minus_host']['bootstrap_ci'],
                  '★' if e['dep_minus_host']['ci_excludes_zero'] else '.',
                  e['dep_minus_random']['mean'],
                  *e['dep_minus_random']['bootstrap_ci'],
                  '★' if e['dep_minus_random'][
                      'ci_excludes_zero'] else '.'))

    # ---- M4 SEIR 平移（Muchuro capture 注入 B1 平移层）----
    state_generic = build_v4_initial_state(n_population=b1.SEIR_POPULATION,
                                            random_state=b1.SEIR_SEED)
    initial_state_c, _ = b1.rescale_latent_to_pi(state_generic, pi_cohort)
    base_times, active_base, new_inf_base = b1.seir_run(initial_state_c,
                                                        _default_v4_params())
    cum_base = float(_trapz(active_base, base_times))
    seir_per_k = {}
    for k in (0.10, 0.20, 0.30):
        cap_dep = capture_summary['%.2f' % k]['dep_fixed']['mean']
        n_courses = int(round(k * b1.SEIR_POPULATION))
        rules = {
            'blanket_random': b1.pt_params(k, k),
            'host_targeted': b1.pt_params(
                capture_summary['%.2f' % k]['host']['mean'],
                capture_summary['%.2f' % k]['host']['mean']),
            'full_targeted': b1.pt_params(cap_dep, cap_dep),
        }
        rule_out = {name: b1.translate_rule(
            initial_state_c, base_times, cum_base, new_inf_base, params,
            n_courses) for name, params in rules.items()}
        d_fb = (rule_out['full_targeted']['averted_case_days']
                - rule_out['blanket_random']['averted_case_days'])
        d_fh = (rule_out['full_targeted']['averted_case_days']
                - rule_out['host_targeted']['averted_case_days'])
        seir_per_k['%.2f' % k] = {
            'n_courses': n_courses,
            'capture_dep_used': round(cap_dep, 4),
            'rules': rule_out,
            'delta_full_minus_blanket': round(d_fb, 2),
            'delta_full_minus_host': round(d_fh, 2),
            'per_100_courses_full_minus_blanket': round(
                d_fb / (n_courses / 100.0), 3),
            'direction_positive': bool(d_fb > 0),
        }
    print('\n=== M4 SEIR 平移（镜像初态 π=%.3f，PT 效力 %.2f）===' % (
        pi_cohort, DEFAULT_PT_EFFICACY))
    for k in (0.10, 0.20, 0.30):
        e = seir_per_k['%.2f' % k]
        print('  k=%.2f: blanket %.1f | host %.1f | full %.1f 病例·天；'
              'Δ(full−blanket)=%.1f（每100人次 %.1f）' % (
                  k, e['rules']['blanket_random']['averted_case_days'],
                  e['rules']['host_targeted']['averted_case_days'],
                  e['rules']['full_targeted']['averted_case_days'],
                  e['delta_full_minus_blanket'],
                  e['per_100_courses_full_minus_blanket']))

    # ---- 判决汇总 ----
    m2a = all(capture_summary['%.2f' % k]['dep_fixed']['mean'] > k
              for k in K_GRID)
    m2b = all(capture_summary['%.2f' % k]['dep_fixed']['mean']
              > capture_summary['%.2f' % k]['host']['mean']
              for k in K_GRID)
    m4 = all(seir_per_k['%.2f' % k]['direction_positive']
             for k in (0.10, 0.20, 0.30))
    verdicts = {
        'M1_anchor_gate': True,
        'M2a_dep_capture_exceeds_k_all': bool(m2a),
        'M2b_dep_capture_exceeds_host_all': bool(m2b),
        'M3_prior_increment': m3,
        'M4_seir_direction_positive': bool(m4),
    }
    print('\n=== 预声明命题判决 ===')
    for kk, vv in verdicts.items():
        print('  %s: %s' % (kk, vv))

    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'muchuro_capture_bridge_v1',
        'status': 'B1 桥接第二队列稳健性检验（预声明 M1-M4）；方向性为主'
                  '判据（59 户小样本），CI 如实报告不作硬判据',
        'design': {
            'source': 'Muchuro（PLOS GPH S1，n=352/59 户，IGRA 32.7%）',
            'replaces': '清单原指定 Aibana（12,648 例）——27 列无户键，'
                        '先证块不可构造（§8.12 纠偏），改用 Muchuro',
            'endpoint': 'igra_pos（IGRA 感染终点）',
            'n': int(len(y)), 'n_events': int(y.sum()),
            'n_households': int(len(np.unique(groups))),
            'cv': 'StratifiedGroupKFold(%d) by tbindex_id × %d 种子，'
                  '折内中位数填补（§8.9 同款）' % (N_SPLITS, N_SEEDS),
            'dep_formula': 'σ(logit(p_full)+w·(r̂−π))，w=%.1f（HomeACF '
                           '网格校准跨队列迁移，未本地重校准），k=%.0f，'
                           'π=队列率 %.4f' % (DEP_W, DEP_K, pi_cohort),
            'capture_definition': 'capture(k)=P(top-k|y=1)',
            'seir': 'B1 平移层同款：镜像初态（潜伏池重标定到 π=%.3f）、'
                    '确定性 ODE、730 天、β=0.3、PT 效力 0.60'
                    % pi_cohort,
        },
        'hypotheses': {
            'M1': 'host/exp/full 复现 §8.9（|Δ|<0.005）',
            'M2': 'capture_dep>k ∀k 且 >capture_host（方向性）',
            'M3': 'RF(full+prior)−RF(full) 方向性读数',
            'M4': 'SEIR Δ(full−blanket)>0 ∀k≤0.30（平凡单调，披露）',
        },
        'verdicts': verdicts,
        'anchor_gate': anchor_gate,
        'capture_summary': capture_summary,
        'capture_bootstrap_last_seed': capture_boot,
        'seir_translation': {
            'baseline_cumulative_case_days': round(cum_base, 2),
            'per_k': seir_per_k,
        },
        'honest_boundaries': [
            'w=7 跨队列迁移未本地重校准——检验的是部署配方迁移而非每队列'
            '最优；k=2 固定 + π=队列率（部署者可估计）',
            'n=352/59 户小样本：bootstrap CI 宽，方向性为主判据；k=0.50 '
            '点位近饱和仅披露',
            'IGRA 阳 32.7% → SEIR 镜像初态 13.9× 潜伏池重标定——绝对水平'
            '为设定场景，仅相对量可引',
            'M4 平移层零经验内容（capture>k ⟹ 靶向强度序保持，平凡单调）',
        ],
        'per_seed_capture': [
            {'seed': r['seed'],
             **{'%.2f' % k: {a: round(capture_per_seed[k][a][i], 4)
                             for a in capture_arms}
                for k in K_GRID}}
            for i, r in enumerate(seeds_oof)
        ],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
