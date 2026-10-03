#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOP 临床意义补口层（第一层，2026-09-25）：DCA net benefit + Brier 分解
+ capture/NNS 同表。

地位：时序先证增益的判别层证据已归档（household_temporal_screening_v1：
exposure_prior 0.7142 vs exposure 0.6441，Δ=+0.0585 户级 cluster bootstrap
CI [+0.034, +0.083]）。审稿必问"AUROC 改善的临床意义是什么"——本脚本以
**决策后果层**回答，三件套同脚本产出：

1. DCA（Vickers net benefit）：NB(pt) = TP/n − FP/n·pt/(1−pt)，
   pt 网格横跨基线率 π=0.132（下方/上方各点位），参照策略 treat-all /
   treat-none（NB≡0）；
2. Brier 评分 + Murphy 分解（BS = UNC − RES + REL）+ scaled Brier
   （skill = 1 − BS/UNC）——校准（REL）与分辨（RES）通道定位；
3. capture(k)/NNS(k) 同表（与 muchuro_capture/S3 级联口径逐字一致：
   capture(k) = P(top-k|y=1)，NNS(k) = top-k 层每检出 1 例 TST+ 所需
   筛查数 = k·n/检出数；NNS_random = n/n_events）。

口径（与 v1/deepening 逐位同构）：HomeACF tst_pos10，20 种子（0-19）
随机筛查顺序，StratifiedGroupKFold(5) by record_id，冻结注册表 RF 逐行
池化 OOF；点估计 = 20 种子均值，配对 CI = 末种子 OOF 户级 cluster
bootstrap ×2000；逐种子 Δ 四栏（R5a 纪律）。锚臂复现 gate（sanity）：
ind/exposure/prior_only/exposure_prior 逐臂 |Δ|<0.002 vs v1 归档。

臂与两层校准结构（seed-0 冒烟后定型）：
  原始臂：exposure（对照）/ exposure_prior（SOP 主臂）/ dep_fixed
  （部署公式 w=7,k=2）/ dep_eb（EB 变体）——冒烟发现 RF 原始 OOF 概率
  系统性过自信（中位预测 ~0.36 vs π=0.132，decile REL 0.06-0.07，
  scaled Brier 为负），低阈值 DCA 退化（uptake ≈98%，NB≈treat-all）；
  校准臂：cal_*（同队列逐种子 Platt 重校准，logit 域 2 参数仿射）——
  DCA/Brier 主读数在 cal_* 上（部署语义：项目硬规则"部署前本地重校准"
  的回顾层对应物；与 w=7 网格校准同款"同队列校准"边界）；原始臂分解
  作为校准缺陷披露层（REL 0.06-0.07 = 强制重校准教义的定量证据）。
  capture/NNS 为秩不变量（Platt 单调）——原始/校准同值，单块报告。

预声明判读（先于运行）：
  H-C1（临床净获益，主判据在 cal 臂）：∃pt∈网格，NB(cal_ep) −
     NB(cal_e) > 0 且末种子配对 cluster bootstrap CI 排除零——AUROC
     增益转化为可陈述的决策净获益；NB(cal_ep) − NB(treat_all) 给出
     "建模筛查 vs 全筛"的净获益差；
  H-C2（Brier 改善，RES 通道）：BS(cal_ep) < BS(cal_e)（方向性 +
     逐种子 share 报告）；分解预期改善走 RES（分辨）通道——先证块
     注入的是排序信息；校准层修复 REL 后残余差异应集中在 RES；
  H-C3（靶向效率，秩不变量）：capture(ep, k) > capture(e, k)
     ∀k∈{0.10, 0.20, 0.30, 0.50}（方向性主判据），NNS 富集
     NNS_random/NNS(k) > 1 且高于 exposure 臂——直接对接 S3 级联
     "放置 ≥0.70"目标的靶向层读数。

诚实边界：
  - 本层为**事后补口**（post-hoc addendum，回应预期审稿意见），非预注册
    分析——预注册协议（ERASE-TB）不覆盖 HomeACF 回顾层的 DCA/Brier；
  - DCA 把"阳性决策"框定为"给予预防性治疗候选资格"（模型预测 ≥ pt →
    进入干预候选）；pt/(1−pt) 为治疗一名未感染者的相对害权重。NB 差
    不依赖治疗效力假设，但**不折算治疗获益的绝对量**（无效力参数注入，
    不作 QALY/每预防病例数声明）；NNS 口径 = 每检出 1 例 TST+ 所需
    筛查数，非每预防 1 例发病所需数；
  - Platt 重校准为**同队列**（逐种子 2 参数，logit 域）——与 v1 网格
    w=7 / EB π̂ 同款边界（同队列校准，无独立外样本），部署时由本地
    重校准替换；校准层读数的乐观性集中在 REL（≈机器零），臂间 ΔNB
    受影响有限（秩保留）；
  - 单队列回顾设计（HomeACF LTBI 终点），跨人群结论归 ERASE-TB 前瞻
    外验。

用法：
    python data/run_sop_clinical_layer.py
输出：
    data/processed/sop_clinical_layer_homeacf_YYYYMMDD.json
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
import run_sop_seir_bridge as b1  # noqa: E402  （capture_at_k 定义同一性）

from tb_risk.validation.real_data_infection import (  # noqa: E402
    load_homeacf_contacts,
)
from tb_risk.validation.real_data_pi import (  # noqa: E402
    _cluster_bootstrap_delta,
)

OUT = os.path.join(HERE, 'processed',
                   'sop_clinical_layer_homeacf_%s.json'
                   % time.strftime('%Y%m%d'))

N_BOOTSTRAP = 2000
N_BINS_BRIER = 10          # Murphy 分解等分位箱数
PT_GRID = (0.05, 0.08, 0.10, 0.132, 0.15, 0.20, 0.30)
K_GRID = (0.10, 0.20, 0.30, 0.50)   # 与 muchuro_capture 一致
ARMS = ('exposure', 'exposure_prior', 'dep_fixed', 'dep_eb')
CAL = {a: 'cal_' + a for a in ARMS}   # 校准臂命名
ANCHOR = {'ind': 0.6428, 'exposure': 0.6441,
          'prior_only': 0.6443, 'exposure_prior': 0.7142}


def _auroc(y, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y, dtype=int),
                               np.asarray(scores, dtype=float)))


# ------------------------------------------------------------------ Brier --

def brier_decomp(y, p, n_bins=N_BINS_BRIER):
    """Brier + Murphy 分解（等分位箱，按排名切箱）。

    粗箱下严格恒等式为 BS = UNC − RES + REL + DISP（DISP = 箱内预测
    离散 (1/n)Σ(p_i−p̄_b)²——Murphy 原式在"预测值即箱值"时 DISP=0；
    连续概率 + 粗箱必须显式报告 DISP，否则恒等式不闭合）。逐位披露，
    identity_check 应为机器精度零。
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    bs = float(np.mean((p - y) ** 2))
    pi = float(y.mean())
    unc = pi * (1.0 - pi)
    order = np.argsort(p, kind='mergesort')
    m = n // n_bins
    res = 0.0
    rel = 0.0
    disp = 0.0
    for b in range(n_bins):
        rows = order[b * m: (b + 1) * m] if b < n_bins - 1 \
            else order[b * m:]
        nb = len(rows)
        if nb == 0:
            continue
        p_bar = float(p[rows].mean())
        phi = float(y[rows].mean())
        res += nb * (phi - pi) ** 2
        rel += nb * (p_bar - phi) ** 2
        disp += float(((p[rows] - p_bar) ** 2).sum())
    res /= n
    rel /= n
    disp /= n
    scaled = 1.0 - bs / unc if unc > 0 else float('nan')
    return {'brier': bs, 'unc': unc, 'res': res, 'rel': rel, 'disp': disp,
            'scaled_brier': scaled,
            'identity_check': unc - res + rel + disp - bs}


# --------------------------------------------------------------------- DCA --

def nb_contrib(y, p, pt):
    """net benefit 逐行贡献：y·s − (1−y)·(pt/(1−pt))·s，s = 1(p≥pt)。

    NB = mean(contrib)；cluster bootstrap 直接重采样户后取均值（配对）。
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    s = (p >= pt).astype(float)
    odds = pt / (1.0 - pt)
    return y * s - (1.0 - y) * odds * s


def net_benefit(y, p, pt):
    return float(nb_contrib(y, p, pt).mean())


def nb_treat_all(y, pt):
    """treat-all 策略 NB（不依赖模型）：π − (1−π)·pt/(1−pt)。"""
    y = np.asarray(y, dtype=float)
    pi = float(y.mean())
    return pi - (1.0 - pi) * (pt / (1.0 - pt))


def platt(p, y):
    """同队列 Platt 重校准：logit 域 2 参数仿射（σ(a·logit(p)+b)）。

    单调变换 → AUROC/capture/NNS 秩不变量；仅修复概率刻度（DCA/Brier
    用）。与 v1 网格 w=7 同款"同队列校准"边界（诚实披露，见 docstring）。
    """
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    x = np.log(np.clip(np.asarray(p, dtype=float),
                       1e-6, 1 - 1e-6) / (1 - np.clip(
                           np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)))
    lr.fit(x.reshape(-1, 1), np.asarray(y, dtype=int))
    return lr.predict_proba(x.reshape(-1, 1))[:, 1]


def cluster_boot_mean_delta(contrib_a, contrib_b, groups,
                            n_boot, seed):
    """配对 Δ（NB/BS 类逐行贡献均值差）的户级 cluster bootstrap。"""
    rng = np.random.RandomState(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in pick])
        deltas.append(float(contrib_a[rows].mean()
                            - contrib_b[rows].mean()))
    deltas = np.asarray(deltas)
    return {
        'mean': float(np.mean(deltas)),
        'bootstrap_ci': [float(np.percentile(deltas, 2.5)),
                         float(np.percentile(deltas, 97.5))],
        'ci_excludes_zero': bool(np.percentile(deltas, 2.5) > 0
                                 or np.percentile(deltas, 97.5) < 0),
    }


# ----------------------------------------------------------- capture / NNS --

def nns_at_k(scores, y, k):
    """top-k 层每检出 1 例所需筛查数 = k·n/检出数；检出 0 → None。"""
    mask = b1.topk_mask(scores, k)
    hit = int(np.asarray(y)[mask].sum())
    if hit == 0:
        return None
    return float(mask.sum()) / hit


def four_col(values):
    """R5a 逐种子四栏。"""
    v = np.asarray(values, dtype=float)
    return {'mean': float(np.mean(v)), 'sd': float(np.std(v)),
            'min': float(np.min(v)), 'max': float(np.max(v)),
            'share_positive': float((v > 0).mean())}


def main():
    t0 = time.time()
    df = load_homeacf_contacts()
    y = df['tst_pos10'].astype(int).to_numpy()
    groups = df['record_id'].to_numpy()
    pi_true = float(y.mean())
    print('HomeACF: n=%d events=%d households=%d  π=%.4f' % (
        len(y), int(y.sum()), len(np.unique(groups)), pi_true))

    # ---- 20 种子 OOF（复用 deepening 管线，含锚臂）----
    seeds = []
    for s in range(rsd.SEED_START, rsd.SEED_START + rsd.N_SEEDS):
        r = rsd.run_seed(s, df, y, groups)
        seeds.append({'seed': s, 'oof': r['oof'], 'pi_true': r['pi_true']})
        print('seed %2d done (%.0fs)' % (s, time.time() - t0))

    # ---- 锚臂复现 gate ----
    anchor_gate = {}
    for a, expect in ANCHOR.items():
        vals = [_auroc(y, r['oof'][a]) for r in seeds]
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

    last = seeds[-1]

    # ---- 同队列逐种子 Platt 重校准（cal 臂）----
    for r in seeds:
        r['cal'] = {a: platt(r['oof'][a], y) for a in ARMS}

    # ---- Brier + Murphy 分解（cal 主读数；raw 为校准缺陷披露）----
    brier = {'cal': {}, 'raw': {}}
    for layer, getp in (('cal', lambda r, a: r['cal'][a]),
                        ('raw', lambda r, a: r['oof'][a])):
        for a in ARMS:
            per = [brier_decomp(y, getp(r, a)) for r in seeds]
            entry = {}
            for key in ('brier', 'unc', 'res', 'rel', 'disp',
                        'scaled_brier'):
                vals = [d[key] for d in per]
                entry[key] = {'mean': round(float(np.mean(vals)), 6),
                              'sd': round(float(np.std(vals)), 6)}
            entry['identity_check_max_abs'] = round(float(np.max(
                [abs(d['identity_check']) for d in per])), 9)
            brier[layer][a] = entry
    brier['delta_cal_ep_minus_e'] = four_col(
        [brier_decomp(y, r['cal']['exposure_prior'])['brier']
         - brier_decomp(y, r['cal']['exposure'])['brier'] for r in seeds])
    brier['delta_raw_ep_minus_e'] = four_col(
        [brier_decomp(y, r['oof']['exposure_prior'])['brier']
         - brier_decomp(y, r['oof']['exposure'])['brier'] for r in seeds])
    # 末种子配对 CI（逐行贡献 = (p−y)²；cal 层）
    bs_p = (last['cal']['exposure_prior'] - y) ** 2
    bs_e = (last['cal']['exposure'] - y) ** 2
    brier['delta_cal_ep_minus_e']['last_seed_ci'] = \
        cluster_boot_mean_delta(bs_p, bs_e, groups,
                                N_BOOTSTRAP, rsd.SEED_START)
    print('\n=== Brier（20 种子均值；cal 主读数 / raw 披露）===')
    for layer in ('cal', 'raw'):
        for a in ARMS:
            e = brier[layer][a]
            print('  %s %-16s BS=%.5f (REL=%.5f RES=%.5f DISP=%.5f '
                  'UNC=%.5f) scaled=%+.4f' % (
                      layer, a, e['brier']['mean'], e['rel']['mean'],
                      e['res']['mean'], e['disp']['mean'],
                      e['unc']['mean'], e['scaled_brier']['mean']))
    d = brier['delta_cal_ep_minus_e']
    print('  ΔBS_cal(ep−e) mean=%+.5f share+=%.2f CI[%+.5f,%+.5f]' % (
        d['mean'], d['share_positive'], d['last_seed_ci']['bootstrap_ci'][0],
        d['last_seed_ci']['bootstrap_ci'][1]))

    # ---- DCA（cal 臂主读数；阈值网格 × 臂 × 参照）----
    dca = {'cal': {}, 'raw': {}}
    for layer, getp in (('cal', lambda r, a: r['cal'][a]),
                        ('raw', lambda r, a: r['oof'][a])):
        for pt in PT_GRID:
            row = {'threshold': pt,
                   'nb_treat_all': round(nb_treat_all(y, pt), 5)}
            for a in ARMS:
                nbs = [net_benefit(y, getp(r, a), pt) for r in seeds]
                row['nb_%s' % a] = {
                    'mean': round(float(np.mean(nbs)), 5),
                    'sd': round(float(np.std(nbs)), 5)}
            # 同表读数：该阈值下筛查比例 / capture / NNS（末种子）
            for a in ARMS:
                p_last = getp(last, a)
                s_mask = p_last >= pt
                row['readout_%s' % a] = {
                    'uptake_last_seed': round(float(s_mask.mean()), 4),
                    'capture_last_seed': (
                        round(float(y[s_mask].sum() / max(y.sum(), 1)), 4)
                        if s_mask.any() else 0.0),
                    'nns_last_seed': (
                        round(float(s_mask.sum() / y[s_mask].sum()), 2)
                        if y[s_mask].sum() > 0 else None),
                }
            # 末种子配对 ΔNB CI（ep vs e；ep vs treat_all）+ 逐种子四栏
            row['delta_nb_ep_minus_e'] = cluster_boot_mean_delta(
                nb_contrib(y, getp(last, 'exposure_prior'), pt),
                nb_contrib(y, getp(last, 'exposure'), pt),
                groups, N_BOOTSTRAP, rsd.SEED_START)
            row['delta_nb_ep_minus_treat_all'] = cluster_boot_mean_delta(
                nb_contrib(y, getp(last, 'exposure_prior'), pt),
                nb_contrib(y, np.ones(len(y)), pt),  # treat-all 逐行
                groups, N_BOOTSTRAP, rsd.SEED_START)
            row['delta_nb_ep_minus_e_per_seed'] = four_col(
                [net_benefit(y, getp(r, 'exposure_prior'), pt)
                 - net_benefit(y, getp(r, 'exposure'), pt) for r in seeds])
            dca[layer]['pt_%.3f' % pt] = row
    print('\n=== DCA net benefit（cal 臂，20 种子均值）===')
    print('  pt     NB_ep     NB_e     NB_all   Δ(ep−e) CI   '
          'uptake_ep  capture_ep  NNS_ep')
    for pt in PT_GRID:
        r = dca['cal']['pt_%.3f' % pt]
        dn = r['delta_nb_ep_minus_e']
        ro = r['readout_exposure_prior']
        print('  %.3f  %+.4f  %+.4f  %+.4f  [%+.4f,%+.4f]  %.3f  '
              '%.3f  %s' % (
                  pt, r['nb_exposure_prior']['mean'],
                  r['nb_exposure']['mean'], r['nb_treat_all'],
                  dn['bootstrap_ci'][0], dn['bootstrap_ci'][1],
                  ro['uptake_last_seed'], ro['capture_last_seed'],
                  ro['nns_last_seed']))

    # ---- capture / NNS（k 网格，与 muchuro 口径一致）----
    cap = {}
    for k in K_GRID:
        row = {'k': k, 'nns_random': round(len(y) / float(y.sum()), 3)}
        for a in ARMS:
            caps = [b1.capture_at_k(r['oof'][a], y, k) for r in seeds]
            row['capture_%s' % a] = {
                'mean': round(float(np.mean(caps)), 4),
                'sd': round(float(np.std(caps)), 4)}
            nns = [nns_at_k(r['oof'][a], y, k) for r in seeds]
            nns = [v for v in nns if v is not None]
            row['nns_%s' % a] = round(float(np.mean(nns)), 3) if nns \
                else None
            row['enrichment_%s' % a] = (
                round(row['nns_random'] / row['nns_%s' % a], 3)
                if row['nns_%s' % a] else None)
        row['delta_capture_ep_minus_e'] = four_col(
            [b1.capture_at_k(r['oof']['exposure_prior'], y, k)
             - b1.capture_at_k(r['oof']['exposure'], y, k)
             for r in seeds])
        cap['k_%.2f' % k] = row
    print('\n=== capture/NNS（k 网格，20 种子均值；NNS_random=%.2f）==='
          % (len(y) / float(y.sum())))
    print('  k     cap_ep  cap_e  cap_dep  NNS_ep  NNS_e  NNS_dep  '
          'rich_ep')
    for k in K_GRID:
        r = cap['k_%.2f' % k]
        print('  %.2f  %.4f  %.4f  %.4f    %.2f  %.2f  %.2f  %.2f' % (
            k, r['capture_exposure_prior']['mean'],
            r['capture_exposure']['mean'],
            r['capture_dep_fixed']['mean'],
            r['nns_exposure_prior'], r['nns_exposure'],
            r['nns_dep_fixed'], r['enrichment_exposure_prior']))

    # ---- 汇总归档 ----
    out = {
        'date': time.strftime('%Y-%m-%d'),
        'name': 'sop_clinical_layer_homeacf_v1',
        'status': '事后补口（post-hoc addendum，回应预期审稿：AUROC 增益'
                  '的临床意义）；判读预声明 H-C1/H-C2/H-C3',
        'design': {
            'source': 'HomeACF（github petermacp/tstsa）',
            'endpoint': 'tst_pos10（TST≥10mm LTBI）',
            'n': int(len(y)), 'n_events': int(y.sum()),
            'n_households': int(len(np.unique(groups))),
            'pi_true': round(pi_true, 4),
            'n_seeds': rsd.N_SEEDS, 'seed_start': rsd.SEED_START,
            'cv': 'StratifiedGroupKFold(%d) by household' % rsd.N_SPLITS,
            'ci': '户级 cluster bootstrap ×%d（末种子 OOF，配对）'
                  % N_BOOTSTRAP,
            'dca_formula': 'NB(pt)=TP/n−FP/n·pt/(1−pt)；决策=预防治疗'
                           '候选资格',
            'brier_bins': '等分位 %d 箱（严格恒等式 BS=UNC−RES+REL+DISP，'
                          'DISP=箱内预测离散）' % N_BINS_BRIER,
            'nns_def': '每检出 1 例 TST+ 所需筛查数；NNS_random=n/n_events',
            'calibration': 'cal 臂 = 同队列逐种子 Platt（logit 域 2 参数'
                           '仿射，σ(a·logit(p)+b)）；与 v1 网格 w=7 同款'
                           '"同队列校准"边界，部署时由本地重校准替换；'
                           'raw 臂 REL 0.06-0.07 为强制重校准教义的定量'
                           '证据（披露层）',
            'rank_invariance': 'capture/NNS 与 AUROC 为秩不变量（Platt '
                               '单调），raw/cal 同值',
            'pt_grid': list(PT_GRID), 'k_grid': list(K_GRID),
            'anchor_gate': anchor_gate,
        },
        'hypotheses': {
            'H-C1': '∃pt: NB(cal_ep)−NB(cal_e)>0 且配对 CI 排除零'
                    '（临床净获益，主判据在 cal 臂）',
            'H-C2': 'BS(cal_ep)<BS(cal_e)，改善走 RES 通道（Brier 分解）',
            'H-C3': 'capture(ep,k)>capture(e,k) ∀k；NNS 富集>随机'
                    '（靶向效率，秩不变量）',
        },
        'brier': brier,
        'dca': dca,
        'capture_nns': cap,
        'per_seed_arm_auroc': [
            {'seed': r['seed'],
             **{a: round(_auroc(y, r['oof'][a]), 4) for a in ARMS}}
            for r in seeds],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n归档: %s (%.0fs)' % (OUT, time.time() - t0))
    return out


if __name__ == '__main__':
    main()
