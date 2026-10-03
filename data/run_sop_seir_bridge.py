#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOP×SEIR 桥接原型（B1，2026-09-07）：个体风险分层 → 干预反事实的靶向效率平移。

地位：打通"个体风险模型（谁该干预）↔ SEIR 干预反事实（干预多大效果）"
两个此前孤立的模块。HomeACF 感染终点 SOP 评分（唯一已判决真实增益，
exposure_prior 0.7142 vs ind 0.6428，+0.0585）在 AUROC 层的增量，本脚本
将其翻译到**决策口径**（靶向捕获率）与**干预经济学口径**（每 100 TPT
人次避免病例·天）。

命题（预声明，先于运行）：
- H-B1（靶向富集，经验内容所在）：部署公式 OOF 评分的 top-k 靶向层对
  TST+ 的捕获率 capture(k)=P(top-k|y=1) 满足
  (a) 20 种子均值 capture(k) > k，∀k∈{0.10,0.20,0.30,0.50}；
  (b) 末种子户级 cluster bootstrap 下 Δ(capture_dep−capture_ind) 在
      k∈{0.10,0.20,0.30} 的 CI 不含零——SOP 增量在召回口径上兑现为
      靶向富集（相对宿主臂，即现行 WHO 优先层所用特征族）。
- H-B2（SEIR 平移，确定性翻译层）：同预算（TPT 人次 = k×N）下，
  ρ_react 通道按 capture(k) 缩放的靶向规则比按 k 缩放的随机覆盖规则
  避免更多病例·天（∀k）；Δ(full−host) 为 SOP 增量的干预经济学平移。
  判据方向性为主——平移层经验内容为 0，全部在 capture 曲线。
- H-B3（π 通道边界，方向 2 反向桥）：以社区 SEIR 初始状态潜伏池比例
  （Lf+Ls≈2.4%）作为部署公式 π̂ 的实测增益保留率。预期：5.5× 幅度
  越过 H-K2 容忍带（低估 2.6× 保留 93%），叠加构造错位（TST+ 含远端
  感染/已清除者，Lf+Ls 仅计可进展潜伏池）→ 社区 SEIR 状态不能替代
  队列 π̂；反向桥（SEIR→π）必须经接触人群重标定，非免费午餐。

协议（与 §8.11 逐位同构）：复用 run_sop_deepening.run_seed 全臂 OOF
（HomeACF tst_pos10，StratifiedGroupKFold(5) by record_id × 20 种子，
random 筛查顺序，冻结注册表 RF）；锚臂 gate |Δ|<0.002
（ind 0.6428 / exposure 0.6441 / prior_only 0.6443 /
exposure_prior 0.7142；dep_fixed 0.7202 取 §8.11 归档）。

SEIR 平移设计（不修改 seir/ 库）：
- 初始状态 = build_v4_initial_state 经潜伏池重标定（Lf+Ls → 队列 π，
  S 补偿守恒；年龄/HIV/耐药边际取 v4 默认）——"接触人群镜像"场景；
- PT 通道解耦：ρ_react×(1−efficacy·capture)（治疗的是捕获的潜伏池
  份额——靶向语义核心）；β_reinf 双口径带：主行按 capture（潜伏中心
  语义）、保守行按 k（全部治疗者语义）——结论须两口径同向；
- 基线/干预同种子同初态成对反事实（复刻 intervention.py 的
  _run/_strategy_params 公式，仅解耦两通道缩放因子）；
- 人口 10000、水平 730 天、dt=7、β=0.3、确定性（noise_scale=0）。

诚实边界：
- SEIR 腿是确定性平移层：干预参数为文献/默认锚（PT 效力 0.60），非
  本地校准；capture 曲线的经验内容全部来自单队列 OOF；
- 初始状态仅重标定潜伏池比例，年龄/HIV 边际非队列镜像——影响绝对
  水平，靶向对比（同初态成对）大体抵消边际选择；
- 横断面队列无时序 → history matching 正流程不适用（H-B3 用初始状态
  潜伏池口径替代，构造错位须随行披露）；
- 单队列 LTBI 终点；k=0.50 接近 capture 天花板（π=0.132），该点位
  仅作饱和披露；random 规则的 capture=k 为期望口径（bootstrap 内
  配对随机抽层，含其抽样方差）。

用法：
    python data/run_sop_seir_bridge.py
输出：
    data/processed/sop_seir_bridge_homeacf_YYYYMMDD.json
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))
sys.path.insert(0, HERE)  # 复用 run_sop_deepening 的逐位一致机制

import run_sop_deepening as rsd  # noqa: E402

from tb_risk.validation.real_data_infection import (  # noqa: E402
    load_homeacf_contacts,
)
from tb_risk.seir.intervention import (  # noqa: E402
    build_v4_initial_state,
    _default_v4_params,
    _aggregate_active,
    _aggregate_susceptible,
    _trapz,
    DEFAULT_PT_EFFICACY,
    DEFAULT_PT_REINF_REDUCTION,
)
from tb_risk.seir.stochastic import StochasticSEIRModel  # noqa: E402
from tb_risk.seir._stochastic_common import (  # noqa: E402
    IDX_S_V3, IDX_LF_V3, IDX_LS_V3,
)

OUT = os.path.join(HERE, 'processed',
                   'sop_seir_bridge_homeacf_%s.json' % time.strftime('%Y%m%d'))

K_GRID = (0.10, 0.20, 0.30, 0.50)
BAND_K = 0.20                    # capture CI → SEIR 平移带的代表点位
SEIR_POPULATION = 10000
SEIR_HORIZON_DAYS = 730.0
SEIR_DT = 7.0
SEIR_BETA = 0.3
SEIR_SEED = 42
CAPTURE_ARMS = ('dep_fixed', 'ind', 'exposure_prior')  # 部署公式/宿主/RF 参考
PROB_EPS = 1e-6


# ---------------------------------------------------------------- capture --

def topk_mask(scores, k):
    """top-k 层隶属（按分数降序取前 ⌈k·n⌉ 行；并列按稳定序截断）。"""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    m = max(1, int(np.ceil(k * n)))
    order = np.argsort(-scores, kind='mergesort')
    mask = np.zeros(n, dtype=bool)
    mask[order[:m]] = True
    return mask


def capture_at_k(scores, y, k):
    """capture(k) = P(top-k | y=1)：top-k 层捕获的阳性占全部阳性比例。"""
    y = np.asarray(y, dtype=int)
    pos = y == 1
    if pos.sum() == 0:
        return float('nan')
    mask = topk_mask(scores, k)
    return float(pos[mask].sum() / pos.sum())


def random_capture_at_k(y, k, rng):
    """随机靶向的配对实现（同层规模均匀抽层，含抽样方差）。"""
    y = np.asarray(y, dtype=int)
    pos = y == 1
    if pos.sum() == 0:
        return float('nan')
    mask = rng.rand(len(y)) < k
    if not mask.any():
        mask[rng.randint(len(y))] = True
    return float(pos[mask].sum() / pos.sum())


def capture_bootstrap(oof, y, groups, k, n_boot, seed):
    """末种子户级 cluster bootstrap：重采样户后户内重算 top-k 与 capture。

    随机规则与分数规则在同一重采样内配对（同层规模均匀抽层），
    Δ(capture_dep−capture_random) 含随机靶向自身方差。
    """
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    caps = {a: [] for a in CAPTURE_ARMS}
    caps_rand = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in pick])
        yr = y[rows]
        if yr.sum() == 0:
            continue
        for a in CAPTURE_ARMS:
            caps[a].append(capture_at_k(oof[a][rows], yr, k))
        caps_rand.append(random_capture_at_k(yr, k, rng))
    out = {}
    for a in CAPTURE_ARMS:
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
    contrasts = {
        'dep_minus_ind': ('dep_fixed', 'ind'),
        'dep_minus_exposure_prior': ('dep_fixed', 'exposure_prior'),
    }
    for name, (a, b) in contrasts.items():
        deltas = np.asarray(caps[a]) - np.asarray(caps[b])
        out[name] = _ci_block(deltas)
    deltas = np.asarray(caps['dep_fixed']) - arr_r
    out['dep_minus_random'] = _ci_block(deltas)
    return out


def _ci_block(deltas):
    deltas = np.asarray(deltas, dtype=float)
    return {
        'mean': float(np.mean(deltas)),
        'bootstrap_ci': [float(np.percentile(deltas, 2.5)),
                         float(np.percentile(deltas, 97.5))],
        'ci_excludes_zero': bool(np.percentile(deltas, 2.5) > 0
                                 or np.percentile(deltas, 97.5) < 0),
    }


# ------------------------------------------------------------------- SEIR --

def rescale_latent_to_pi(state, pi_target):
    """潜伏池重标定：Lf/Ls 等比放大到 π_target，S 逐层补偿守恒。"""
    st = np.array(state, dtype=float, copy=True)
    lat = st[:, :, :, IDX_LF_V3].sum() + st[:, :, :, IDX_LS_V3].sum()
    latent_frac = float(lat / st.sum())
    factor = pi_target / latent_frac
    excess = np.zeros_like(st[:, :, :, IDX_S_V3])
    for idx in (IDX_LF_V3, IDX_LS_V3):
        excess = excess + st[:, :, :, idx] * (factor - 1.0)
        st[:, :, :, idx] = st[:, :, :, idx] * factor
    st[:, :, :, IDX_S_V3] = np.maximum(
        st[:, :, :, IDX_S_V3] - excess, 0.0)
    return st, latent_frac


def seir_run(initial_state, params, population=SEIR_POPULATION,
             t_horizon=SEIR_HORIZON_DAYS, dt=SEIR_DT, beta=SEIR_BETA,
             seed=SEIR_SEED):
    """单次 v4 模拟（复刻 intervention._run：确定性 ODE，返回活动曲线）。"""
    model = StochasticSEIRModel(population=population, noise_scale=0.0,
                                seed=seed)
    times, traj, _ = model.simulate_sde_v4(
        initial_state, (0.0, float(t_horizon)), dt, beta, **params)
    n_steps = traj.shape[0]
    active = np.array([_aggregate_active(traj[i]) for i in range(n_steps)])
    s0 = _aggregate_susceptible(initial_state)
    sT = _aggregate_susceptible(traj[-1])
    return times, active, max(0.0, s0 - sT)


def pt_params(cov_rho, cov_beta):
    """PT 参数覆盖（复刻 _strategy_params 公式，解耦两通道缩放因子）。"""
    p = _default_v4_params()
    p['rho_react'] = p['rho_react'] * (1.0 - cov_rho * DEFAULT_PT_EFFICACY)
    p['beta_reinf'] = p['beta_reinf'] * (
        1.0 - cov_beta * DEFAULT_PT_REINF_REDUCTION)
    return p


def translate_rule(initial_state, times, base_cum, base_new_inf, params,
                   n_courses):
    """单规则反事实 → 避免负担与每 100 TPT 人次口径。"""
    _, active, new_inf = seir_run(initial_state, params)
    cum = float(_trapz(active, times))
    averted = max(0.0, base_cum - cum)
    return {
        'cumulative_case_days': cum,
        'final_active': float(active[-1]),
        'peak_active': float(np.max(active)),
        'new_infections': new_inf,
        'averted_case_days': averted,
        'averted_fraction': float(averted / base_cum) if base_cum > 0 else 0.0,
        'averted_new_infections': max(0.0, base_new_inf - new_inf),
        'averted_case_days_per_100_courses': float(
            averted / (n_courses / 100.0)) if n_courses > 0 else 0.0,
    }


# -------------------------------------------------------------------- 主流程

def main():
    t0 = time.time()
    df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    print('HomeACF: n=%d events=%d households=%d' % (
        len(y), int(y.sum()), len(np.unique(groups))))

    # ---- 腿 1：全臂 OOF（复用 §8.11 run_seed，逐位一致）----
    seeds_results = []
    for s in range(rsd.SEED_START, rsd.SEED_START + rsd.N_SEEDS):
        r = rsd.run_seed(s, df, y, groups)
        r['y'] = y
        r['groups'] = groups
        seeds_results.append(r)
        print('seed %2d done (%.0fs)' % (s, time.time() - t0))
    last = seeds_results[-1]
    y_l, g_l = last['y'], last['groups']
    oof_l = last['oof']
    pi_cohort = float(last['pi_true'])

    # ---- 锚臂复现 gate ----
    anchor_expect = {'ind': 0.6428, 'exposure': 0.6441,
                     'prior_only': 0.6443, 'exposure_prior': 0.7142,
                     'dep_fixed': 0.7202}
    anchor_gate = {}
    for a, expect in anchor_expect.items():
        v = float(np.mean([rsd._auroc(r['y'], r['oof'][a])
                           for r in seeds_results]))
        anchor_gate[a] = {'this_run': round(v, 4), 'archive': expect,
                          'reproduced': bool(abs(v - expect) < 0.002)}
    print('\n=== 锚臂复现 gate ===')
    for a, g in anchor_gate.items():
        print('  %-16s %.4f vs %.4f  %s' % (
            a, g['this_run'], g['archive'],
            'OK' if g['reproduced'] else 'FAIL'))
    if not all(g['reproduced'] for g in anchor_gate.values()):
        print('!! 锚臂 gate 失败，中止（不产出归档）')
        return None

    # ---- H-B1：capture 曲线（逐种子 + 末种子 bootstrap）----
    capture_per_seed = {k: {a: [] for a in CAPTURE_ARMS} for k in K_GRID}
    for r in seeds_results:
        for k in K_GRID:
            for a in CAPTURE_ARMS:
                capture_per_seed[k][a].append(
                    capture_at_k(r['oof'][a], r['y'], k))
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
                'enrichment_vs_random': round(
                    float(np.mean(arr)) / k, 3),
            }
        capture_summary['%.2f' % k] = entry
    print('\n=== H-B1 capture(k)（20 种子均值，random=k）===')
    for k in K_GRID:
        e = capture_summary['%.2f' % k]
        print('  k=%.2f  dep=%.3f (x%.2f)  ind=%.3f (x%.2f)  '
              'exp_prior=%.3f (x%.2f)' % (
                  k, e['dep_fixed']['mean'],
                  e['dep_fixed']['enrichment_vs_random'],
                  e['ind']['mean'], e['ind']['enrichment_vs_random'],
                  e['exposure_prior']['mean'],
                  e['exposure_prior']['enrichment_vs_random']))

    capture_boot = {}
    for k in K_GRID:
        capture_boot['%.2f' % k] = capture_bootstrap(
            oof_l, y_l, g_l, k, rsd.N_BOOTSTRAP, rsd.SEED_START)
    print('\n=== H-B1 末种子 bootstrap（Δ capture，户级 cluster ×%d）==='
          % rsd.N_BOOTSTRAP)
    for k in K_GRID:
        e = capture_boot['%.2f' % k]
        print('  k=%.2f  dep−ind %+0.4f CI [%+0.4f,%+0.4f] %s | '
              'dep−rand %+0.4f CI [%+0.4f,%+0.4f] %s' % (
                  k,
                  e['dep_minus_ind']['mean'], *e['dep_minus_ind']['bootstrap_ci'],
                  '★' if e['dep_minus_ind']['ci_excludes_zero'] else '.',
                  e['dep_minus_random']['mean'],
                  *e['dep_minus_random']['bootstrap_ci'],
                  '★' if e['dep_minus_random']['ci_excludes_zero'] else '.'))

    # ---- H-B3：π 通道边界（SEIR 初始状态潜伏池 → 部署 π̂）----
    state_generic = build_v4_initial_state(n_population=SEIR_POPULATION,
                                            random_state=SEIR_SEED)
    latent_frac_generic = float(
        (state_generic[:, :, :, IDX_LF_V3].sum()
         + state_generic[:, :, :, IDX_LS_V3].sum()) / state_generic.sum())
    pi_seir = round(latent_frac_generic, 4)

    retentions, gains_pi_seir, gains_true = [], [], []
    for r in seeds_results:
        s_pi = rsd.dep_scores(r['oof']['exposure'], r['prior_n'],
                              r['prior_pos'], pi_seir, rsd.DEP_K, rsd.DEP_W)
        auc_pi = rsd._auroc(r['y'], s_pi)
        auc_static = rsd._auroc(r['y'], r['oof']['dep_static'])
        auc_true = rsd._auroc(r['y'], r['oof']['dep_fixed'])
        g_pi, g_true = auc_pi - auc_static, auc_true - auc_static
        gains_pi_seir.append(g_pi)
        gains_true.append(g_true)
        if abs(g_true) > 1e-9:
            retentions.append(g_pi / g_true)
    gains_pi_050 = [
        rsd._auroc(r['y'], r['oof']['dep_pi_0.050'])
        - rsd._auroc(r['y'], r['oof']['dep_static'])
        for r in seeds_results]
    pi_channel = {
        'pi_cohort_true': round(pi_cohort, 4),
        'pi_seir_generic_state': pi_seir,
        'mismatch_ratio_underestimate': round(pi_cohort / pi_seir, 2),
        'gain_retention_at_pi_seir': {
            'mean': round(float(np.mean(retentions)), 4),
            'min': round(float(np.min(retentions)), 4),
            'max': round(float(np.max(retentions)), 4),
        },
        'gain_vs_static_at_pi_seir': {
            'mean': round(float(np.mean(gains_pi_seir)), 4),
            'share_positive': round(
                float((np.asarray(gains_pi_seir) > 0).mean()), 3),
        },
        'gain_vs_static_at_pi_true': {
            'mean': round(float(np.mean(gains_true)), 4)},
        'gain_vs_static_at_pi_050_hk2_grid': {
            'mean': round(float(np.mean(gains_pi_050)), 4)},
        'construct_note': 'TST+ 含远端感染/已清除者；SEIR Lf+Ls 仅计可进展'
                          '潜伏池——两者构造错位，幅度 5.5× 越过 H-K2 容忍带',
    }
    print('\n=== H-B3 π 通道（SEIR 初始状态 π=%.4f vs 队列 π=%.4f，'
          '低估 %.1f×）===' % (
              pi_seir, pi_cohort, pi_cohort / pi_seir))
    print('  增益保留率 @π_SEIR: %.3f  (gain %+0.4f vs true %+0.4f; '
          'π=0.050 网格点 %+0.4f)' % (
              np.mean(retentions), np.mean(gains_pi_seir),
              np.mean(gains_true), np.mean(gains_pi_050)))

    # ---- 腿 2：SEIR 平移（确定性反事实，接触人群镜像初态）----
    initial_state_c, _ = rescale_latent_to_pi(state_generic, pi_cohort)
    base_params = _default_v4_params()
    base_times, active_base, new_inf_base = seir_run(initial_state_c,
                                                     base_params)
    cum_base = float(_trapz(active_base, base_times))
    print('\n=== 腿 2 SEIR 平移（接触人群镜像初态，π=%.3f，人口 %d，'
          '水平 %.0f 天）===' % (pi_cohort, SEIR_POPULATION,
                                 SEIR_HORIZON_DAYS))
    print('  基线：累计 %.0f 病例·天，峰值 %.1f，新感染 %.0f' % (
        cum_base, float(np.max(active_base)), new_inf_base))

    seir_per_k = {}
    for k in K_GRID:
        cap_dep = capture_summary['%.2f' % k]['dep_fixed']['mean']
        cap_ind = capture_summary['%.2f' % k]['ind']['mean']
        n_courses = int(round(k * SEIR_POPULATION))
        rules = {
            'blanket_random': pt_params(k, k),
            'host_targeted': pt_params(cap_ind, cap_ind),
            'full_targeted': pt_params(cap_dep, cap_dep),
            'full_targeted_conservative_beta': pt_params(cap_dep, k),
        }
        rule_out = {}
        for name, params in rules.items():
            rule_out[name] = translate_rule(
                initial_state_c, base_times, cum_base, new_inf_base, params,
                n_courses)
        d_full_blanket = (rule_out['full_targeted']['averted_case_days']
                          - rule_out['blanket_random']['averted_case_days'])
        d_full_host = (rule_out['full_targeted']['averted_case_days']
                       - rule_out['host_targeted']['averted_case_days'])
        d_cons_blanket = (
            rule_out['full_targeted_conservative_beta']['averted_case_days']
            - rule_out['blanket_random']['averted_case_days'])
        seir_per_k['%.2f' % k] = {
            'n_courses': n_courses,
            'capture_dep_used': round(cap_dep, 4),
            'capture_ind_used': round(cap_ind, 4),
            'rules': rule_out,
            'delta_full_minus_blanket_case_days': round(d_full_blanket, 2),
            'delta_full_minus_host_case_days': round(d_full_host, 2),
            'delta_conservative_minus_blanket_case_days': round(
                d_cons_blanket, 2),
            'delta_full_minus_blanket_per_100_courses': round(
                d_full_blanket / (n_courses / 100.0), 3),
            'delta_full_minus_host_per_100_courses': round(
                d_full_host / (n_courses / 100.0), 3),
        }
        e = seir_per_k['%.2f' % k]
        print('  k=%.2f（%d 人次）: blanket 避免 %.1f | host %.1f | '
              'full %.1f | full(保守β) %.1f 病例·天；'
              'Δ(full−blanket)=%.1f（每100人次 %.2f） '
              'Δ(full−host)=%.1f' % (
                  k, n_courses,
                  rule_out['blanket_random']['averted_case_days'],
                  rule_out['host_targeted']['averted_case_days'],
                  rule_out['full_targeted']['averted_case_days'],
                  rule_out['full_targeted_conservative_beta'][
                      'averted_case_days'],
                  d_full_blanket,
                  e['delta_full_minus_blanket_per_100_courses'],
                  d_full_host))

    # ---- capture CI → SEIR 平移带（代表点 k=0.20）----
    cap_ci = capture_boot['%.2f' % BAND_K]['dep_fixed']['bootstrap_ci']
    n_courses_band = int(round(BAND_K * SEIR_POPULATION))
    band_lo = translate_rule(
        initial_state_c, base_times, cum_base, new_inf_base,
        pt_params(cap_ci[0], cap_ci[0]), n_courses_band)
    band_hi = translate_rule(
        initial_state_c, base_times, cum_base, new_inf_base,
        pt_params(cap_ci[1], cap_ci[1]), n_courses_band)
    blanket_k = seir_per_k['%.2f' % BAND_K]['rules']['blanket_random']
    seir_band = {
        'k': BAND_K,
        'capture_ci': [round(v, 4) for v in cap_ci],
        'averted_case_days_range': [
            round(band_lo['averted_case_days'], 2),
            round(band_hi['averted_case_days'], 2)],
        'delta_minus_blanket_range': [
            round(band_lo['averted_case_days']
                  - blanket_k['averted_case_days'], 2),
            round(band_hi['averted_case_days']
                  - blanket_k['averted_case_days'], 2)],
    }
    print('\n=== 平移带（k=%.2f，capture CI → Δ(full−blanket) 病例·天 '
          '[%.1f, %.1f]）===' % (
              BAND_K, seir_band['delta_minus_blanket_range'][0],
              seir_band['delta_minus_blanket_range'][1]))

    # ---- 汇总归档 ----
    hb1_dep_exceeds_k = all(
        capture_summary['%.2f' % k]['dep_fixed']['mean'] > k for k in K_GRID)
    hb1_ci = {k: capture_boot['%.2f' % k]['dep_minus_ind'][
        'ci_excludes_zero'] for k in K_GRID}
    hb2_direction = all(
        seir_per_k['%.2f' % k]['delta_full_minus_blanket_case_days'] > 0
        for k in K_GRID) and all(
        seir_per_k['%.2f' % k]['delta_conservative_minus_blanket_case_days']
        > 0 for k in K_GRID)
    verdicts = {
        'H-B1a_capture_exceeds_k_all_k': bool(hb1_dep_exceeds_k),
        'H-B1b_dep_minus_ind_ci_excludes_zero': hb1_ci,
        'H-B2_full_beats_blanket_both_calibres': bool(hb2_direction),
        'H-B3_retention_at_pi_seir': pi_channel['gain_retention_at_pi_seir'],
    }
    print('\n=== 预声明命题判决 ===')
    print('  H-B1a capture>k ∀k: %s' % hb1_dep_exceeds_k)
    print('  H-B1b Δ(dep−ind) CI 不含零: %s' % hb1_ci)
    print('  H-B2 full>blanket（主+保守口径同向）: %s' % hb2_direction)
    print('  H-B3 增益保留率 @π_SEIR: %s' % pi_channel[
        'gain_retention_at_pi_seir'])

    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'sop_seir_bridge_homeacf_v1',
        'status': '桥接原型（预声明 H-B1/H-B2/H-B3）；SEIR 腿为确定性'
                  '平移层，经验内容在 capture 曲线（HomeACF OOF）',
        'design': {
            'source': 'HomeACF（github petermacp/tstsa）',
            'endpoint': 'tst_pos10（TST≥10mm LTBI）',
            'n': int(len(y_l)), 'n_events': int(y_l.sum()),
            'n_households': int(len(np.unique(g_l))),
            'n_seeds': rsd.N_SEEDS, 'seed_start': rsd.SEED_START,
            'oof': '复用 run_sop_deepening.run_seed（§8.11 逐位一致）',
            'anchor_gate': anchor_gate,
            'capture_definition': 'capture(k)=P(top-k|y=1)（召回/捕获率，'
                                  'top-k 按 OOF 分数层内排序）',
            'k_grid': list(K_GRID),
            'seir': {
                'population': SEIR_POPULATION,
                't_horizon_days': SEIR_HORIZON_DAYS,
                'dt_days': SEIR_DT,
                'beta': SEIR_BETA,
                'noise': 'deterministic (noise_scale=0)',
                'initial_state': 'build_v4_initial_state 潜伏池重标定到'
                                 '队列 π（Lf/Ls 等比、S 补偿；年龄/HIV/'
                                 '耐药边际 v4 默认）',
                'strategy': 'preventive_treatment（效力 %.2f，再感染下调 '
                            '%.2f，通道解耦：ρ_react←capture，β_reinf 双'
                            '口径带）' % (DEFAULT_PT_EFFICACY,
                                          DEFAULT_PT_REINF_REDUCTION),
                'pairing': '基线/干预同种子同初态成对反事实',
            },
        },
        'hypotheses': {
            'H-B1': 'capture(k)>k ∀k 且 Δ(dep−ind) CI 不含零 @k≤0.30',
            'H-B2': '同预算下靶向（ρ_react←capture）> 随机覆盖（←k）'
                    '避免病例·天，主/保守口径同向',
            'H-B3': 'SEIR 社区状态 π̂ 的增益保留率 + 构造错位 → 不可替代'
                    '队列 π̂',
        },
        'verdicts': verdicts,
        'capture': {
            'per_seed_summary': capture_summary,
            'last_seed_bootstrap': capture_boot,
        },
        'seir_translation': {
            'baseline': {
                'cumulative_case_days': round(cum_base, 2),
                'peak_active': round(float(np.max(active_base)), 2),
                'new_infections': round(new_inf_base, 2),
                'latent_fraction': round(pi_cohort, 4),
            },
            'per_k': seir_per_k,
            'capture_ci_band': seir_band,
        },
        'pi_channel_boundary': pi_channel,
        'honest_boundaries': [
            'SEIR 腿是确定性平移层：干预参数为文献/默认锚（PT 效力 0.60、'
            'β=0.3）非本地校准；经验内容全部在 capture 曲线（单队列 OOF）',
            '初始状态仅重标定潜伏池比例，年龄/HIV 边际为 v4 默认非队列镜像'
            '——绝对水平受影响，成对靶向对比大体抵消边际选择',
            'β_reinf 通道口径带：主行按 capture（潜伏中心语义）、保守行按 k'
            '（全部治疗者语义）；结论须两口径同向（见 verdicts）',
            '横断面队列无时序 → history matching 正流程不适用；H-B3 用初始'
            '状态潜伏池口径替代，构造错位随行披露',
            'k=0.50 接近 capture 天花板（π=0.132），该点位仅作饱和披露',
            'random 规则 capture=k 为期望口径；bootstrap 内配对随机抽层含其'
            '抽样方差',
        ],
        'per_seed_capture': [
            {'seed': r['seed'],
             **{'%.2f' % k: {a: round(capture_per_seed[k][a][i], 4)
                             for a in CAPTURE_ARMS}
                for k in K_GRID}}
            for i, r in enumerate(seeds_results)
        ],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
