#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""胸片覆盖下探成本-效果模拟（政策参数决策证据腿，2026-09-06；r2 扩展）

问题：克拉玛依接触者胸片触发面（高风险 + 中风险重点人群，见
karamay/policy.py get_screening_recommendation）窄于国内规范基线
（≥15 岁密接症状+LTBI+胸片三件套同筛）。胸片覆盖应下探到哪个
风险分层，边际检出/读片负担最优？

三位置裁决（crp_karamay_schema_expansion_memo.md §3 D2 + §7）：
- 接触者主线 22 维：胸片/CAD 不入模（D2 分支甲，2026-09-05 冻结）；
- 分诊分支：CAD 连续分数条件入模（D2 分支乙，四条件全满足方可）；
- 检测动作覆盖面：政策参数，由本模拟的边际检出曲线 + 盈亏平衡
  查找表裁决——正确问题不是"要不要全员加"，而是"覆盖下探到哪个
  风险分层时边际检出/百张胸片跌破可接受线"。

r2 扩展（2026-09-06 第二轮，基础节未改动、数值与 r1 逐位一致）：
- P0-a 下探曲线 bootstrap CI：患者级 1000 次重采样（seed 42），
  每层边际检出/百张与富集倍数的 95% 百分位区间；层间配对比较
  判定 40% 层反常富集 / 底部 10% 层回升是噪声还是信号；
- P0-b 排序器质量敏感性臂：对 OOF 分数做保序标签混合（rank 归一
  + 标签混合，二分搜索混合系数命中目标 AUROC），在
  {实测, 0.65, 0.70, 0.80} 下重放下探曲线——量化"排序器每
  +0.05 AUROC → 富集/检出/每例检出省多少张胸片"；
- P1 本地盈亏平衡查找表：本地患病率 {0.5%, 1%, 2%, 3%} ×
  可接受胸片/检出 {20, 50, 100} → 推荐覆盖层（点估计与 CI 下界
  保守口径），附 0.70/0.80 合成排序器对照。

方法（基础节，r1 已归档）：
- CRP 域（LSHTM ACF 验证子集 n=1777 / 阳性 117，赞比亚+南非，
  全部 ≥15 岁）：22 维基线特征 5 折 OOF（seed 42，LR+LGBM，
  与 2026-09-05 归档 run 同口径）做风险排序——排序不含 cad/crp
  （时序前置性：基线评分先于胸片，D2 分支甲同一纪律）；
- 检出模型：文献校准主口径（WHO 2021 结核筛查指南密接口径：
  症状+胸片联合灵敏度 0.85 vs 仅症状 0.53 → 覆盖一个分层的
  检出增量 = Δsens × 该分层病例数）；另设保守口径 Δsens=0.20；
- CAD≥40 数据口径只作上限披露（验证规则循环：CAD≥40 是送检
  资格规则，备忘录 §2.1）；
- Kenya 2016 全国患病率调查（n=63,050 / 确诊 336）：症状→确诊
  映射，量化"仅症状筛查的结构性盲区"（无症状确诊病例比例）。

已知偏倚（引用本模拟数字时必须随行）：
1. CRP 域为验证子集（患病率 6.6% 被富集），绝对边际检出数不可
   外推——政策迁移用"富集曲线"（分层病例浓度 vs 覆盖比例）×
   本地患病率参数化：本地边际检出/百张 = Δsens × 本地患病率 ×
   富集倍数 × 100；
2. CAD≥40 是验证资格规则 → 数据口径灵敏度（~94%）只作上限；
3. Kenya XRayExam 字段语义不明（=1 者 97% 有症状、仅 58.7%
   触发痰检），不可作"胸片异常"代理——Kenya 只贡献症状→确诊
   分解，不贡献胸片检出分解；
4. Kenya 无症状确诊病例 90 例全部有痰检记录（SputumRequestDone=1）
   → 存在未记录的送检通道，26.8% 为"记录症状口径"的估计而非
   严格下界；
5. 排序模型 AUROC ~0.62（与 A0 归档一致，弱排序器）——富集曲线
   是"当前模型可实现"而非"理论上限"；
6. Δsens 文献校准均匀施加于所有分层（层特异性检出率不可测，
   无层间检出异质性数据）——层间差异全部来自排序富集，检出端
   无层间自由度；
7. 对照策略的"检出/百张"为覆盖集均值口径，曲线各层为边际带
   口径——两量纲并存，对照值不可与单层边际值直接比较；
8. bootstrap CI 条件于 seed 42 单次 OOF 排序器（患者级重采样
   不含模型训练随机性）；保序标签混合为合成变换（病例谱固定、
   排序器改进表现为分数-标签对齐度提升，不模拟特征空间变化），
   0.65/0.70/0.80 曲线为合成对照、非实测预测。

用法：
    python data/run_cxr_coverage_simulation.py
输出：
    data/processed/cxr_coverage_simulation_20260906.json（r2 原位
    再生：基础节与 r1 逐位一致，扩展节追加，revision 字段留痕）
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)

PROC = os.path.join(HERE, "processed")
CRP_SRC = os.path.join(PROC, "ml_training_crp_real.csv")
KENYA_SRC = os.path.join(HERE, "raw", "kenya_prevalence", "S01.csv")
OUT = os.path.join(PROC, "cxr_coverage_simulation_20260906.json")

# 文献校准参数（WHO 2021 结核筛查指南，家庭接触者筛查口径；
# 外部点估计，引用时随行来源标注）
SENS_SYMPTOM_ONLY = 0.53
SENS_SYMPTOM_CXR = 0.85
DELTA_SENS_CONSERVATIVE = 0.20

# 部署风险分层阈值（validation/threshold_spec.py 单一真值源：
# 极高 ≥15% / 高 ≥8% / 中 ≥3%）
TH_HIGH, TH_MEDIUM = 0.08, 0.03

TIERS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_BOOT = 1000
# P2：折分种子稳健性——split seed 变异、模型 seed 固定 42
SEEDS = [42, 7, 13, 99, 2020]

BASE_FEATURES = ['age', 'cumulative_exposure', 'has_symptoms',
                 'bcg_vaccine', 'has_tb', 'contact_distance_score',
                 'ventilation_score', 'is_high_risk', 'past_illness',
                 'exposure_setting_score', 'single_duration',
                 'freq_density', 'time_span']
INTERACTIONS = ['age_immuno', 'dm_tb_synergy', 'age_bcg_decay',
                'symptom_delay', 'cough_contact', 'highrisk_comorbid',
                'immune_bcg', 'exposure_accumulation', 'age_diabetes']

KENYA_SYMPTOMS = ['Coughing', 'Sputum', 'BloodCough', 'ChestPains',
                  'Fever', 'Fatigue', 'WeightLoss', 'NightSweats',
                  'BreatheShortness']


def build_features(df):
    """22 维特征（复用 scoring/ml/training.py L1317-1352 同一逻辑：
    past_illness_type → 免疫抑制分；delay/cough/contact 列缺省时
    与生产管线同默认值）。"""
    hiv = df['past_illness_type'].isin(['hiv', 'HIV']).astype(float)
    diabetes = df['past_illness_type'].isin(
        ['diabetes', '糖尿病']).astype(float)
    immunosupp = df['past_illness_type'].isin(
        ['immunosuppressants', '免疫抑制']).astype(float)
    immuno = (hiv * 2.0 + diabetes * 1.0 + immunosupp * 1.5
              + df['past_illness'].astype(float) * 0.5)
    X = df[BASE_FEATURES].astype(float).copy()
    X['age_immuno'] = df['age'].astype(float) * immuno
    X['dm_tb_synergy'] = diabetes * df['has_tb'].astype(float)
    X['age_bcg_decay'] = df['age'].astype(float) * (1 - df['bcg_vaccine'].astype(float))
    X['symptom_delay'] = df['has_symptoms'].astype(float) * 0.0  # delay 缺省 0，同生产管线
    X['cough_contact'] = 0.0 * (5 / 10)                          # cough 缺省 0，同生产管线
    X['highrisk_comorbid'] = df['is_high_risk'].astype(float) * df['past_illness'].astype(float)
    X['immune_bcg'] = (1 - df['bcg_vaccine'].astype(float)) * immuno
    X['exposure_accumulation'] = (df['cumulative_exposure'].astype(float) / 80) * (df['time_span'].astype(float) / 10)
    X['age_diabetes'] = df['age'].astype(float) * diabetes
    return X[BASE_FEATURES + INTERACTIONS].values


def oof_scores(X, y):
    """5 折 OOF（seed 42）——LR 与 LGBM 双排序器，报告两者，
    取 AUROC 更高者作覆盖曲线主排序（弱排序器口径，与 A0 一致）。"""
    scores, aucs = {}, {}
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for name, mk in [('LR', lambda: LogisticRegression(max_iter=2000))] + (
            [('LGBM', lambda: LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=15,
                random_state=42, verbose=-1))] if HAS_LGBM else []):
        oof = np.zeros(len(y))
        for tri, tei in skf.split(X, y):
            m = mk().fit(X[tri], y[tri])
            oof[tei] = m.predict_proba(X[tei])[:, 1]
        scores[name] = oof
        aucs[name] = float(roc_auc_score(y, oof))
    best = max(aucs, key=aucs.get)
    return scores[best], best, aucs


def descent_curve(score, y, delta_sens):
    """覆盖下探曲线（模型排序 top-k% 逐层边际统计；r1 基础节原逻辑）。"""
    prev_domain = float(y.mean())
    order = np.argsort(-score)
    n = len(y)
    curve, prev_cum, det_cum, n_cum = [], 0.0, 0.0, 0
    for pct in TIERS:
        k = int(round(n * pct / 100))
        idx = order[:k]
        pos_tier = int(y[idx].sum()) - prev_cum
        n_tier = k - n_cum
        det_tier = delta_sens * pos_tier
        det_tier_cons = DELTA_SENS_CONSERVATIVE * pos_tier
        row = {
            'coverage_pct': pct,
            'n_cxr_cum': k, 'cases_cum': int(prev_cum + pos_tier),
            'marginal_n_cxr': int(n_tier),
            'marginal_cases': int(pos_tier),
            'marginal_prevalence': round(pos_tier / n_tier, 4) if n_tier else None,
            'enrichment': round((pos_tier / n_tier) / prev_domain, 2) if n_tier and prev_domain else None,
            'marginal_det_per_100_cxr': round(det_tier / n_tier * 100, 2) if n_tier else None,
            'marginal_det_per_100_cxr_conservative': round(det_tier_cons / n_tier * 100, 2) if n_tier else None,
            'cum_det': round(det_cum + det_tier, 1),
            'cxr_per_marginal_det': round(n_tier / det_tier, 1) if det_tier > 0 else None,
        }
        curve.append(row)
        prev_cum += pos_tier
        det_cum += det_tier
        n_cum = k
    return curve


def bootstrap_ci(score, y, delta_sens, n_boot=N_BOOT, seed=42):
    """P0-a：患者级 bootstrap（条件于固定 OOF 排序器）。

    每次重采样内重算分层（分位数边界随样本重排），给出每层
    边际检出/百张与富集倍数的 95% 百分位区间；层间配对比较
    （同一次重采样内）判定非单调反常是噪声还是信号。
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    det_samples = {t: [] for t in TIERS}
    enr_samples = {t: [] for t in TIERS}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        y_b, s_b = y[idx], score[idx]
        order_b = np.argsort(-s_b)
        dom_prev_b = float(y_b.mean())
        prev_pos, prev_n = 0, 0
        for t in TIERS:
            k = int(round(n * t / 100))
            sel = order_b[:k]
            pos_cum = int(y_b[sel].sum())
            m_cases = pos_cum - prev_pos
            m_n = k - prev_n
            det_samples[t].append(delta_sens * m_cases / m_n * 100)
            enr_samples[t].append(
                (m_cases / m_n) / dom_prev_b if dom_prev_b > 0 else np.nan)
            prev_pos, prev_n = pos_cum, k
    ci = {}
    for t in TIERS:
        det = np.asarray(det_samples[t], dtype=float)
        enr = np.asarray(enr_samples[t], dtype=float)
        ci[t] = {
            'det_per_100_ci95': [round(float(np.nanpercentile(det, 2.5)), 2),
                                 round(float(np.nanpercentile(det, 97.5)), 2)],
            'enrichment_ci95': [round(float(np.nanpercentile(enr, 2.5)), 2),
                                round(float(np.nanpercentile(enr, 97.5)), 2)],
            'p_enrichment_below_1': round(float(np.mean(enr < 1.0)), 3),
        }
    e10 = np.asarray(enr_samples[10]); e30 = np.asarray(enr_samples[30])
    e40 = np.asarray(enr_samples[40]); e50 = np.asarray(enr_samples[50])
    e80 = np.asarray(enr_samples[80]); e100 = np.asarray(enr_samples[100])
    anomaly = {
        'p_enr40_gt_enr10': round(float(np.mean(e40 > e10)), 3),
        'p_enr40_gt_enr30': round(float(np.mean(e40 > e30)), 3),
        'p_enr40_gt_enr50': round(float(np.mean(e40 > e50)), 3),
        'p_bottom_gt_tier80': round(float(np.mean(e100 > e80)), 3),
        'threshold': '层间配对单侧 P>0.95 判信号，否则噪声',
    }
    anomaly['verdict_40pct'] = ('signal' if anomaly['p_enr40_gt_enr10'] > 0.95
                                else 'noise')
    anomaly['verdict_bottom_rebound'] = (
        'signal' if anomaly['p_bottom_gt_tier80'] > 0.95 else 'noise')
    return ci, anomaly


def morph_to_auroc(score, y, target):
    """P0-b：保序标签混合——rank 归一后与标签按系数 α 混合，
    二分搜索 α 命中目标 AUROC。合成变换：病例谱固定，排序器
    改进表现为分数-标签对齐度提升，不模拟特征空间变化。"""
    r = pd.Series(score).rank().values / len(score)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        a = (lo + hi) / 2
        if roc_auc_score(y, (1 - a) * r + a * y) < target:
            lo = a
        else:
            hi = a
    a = (lo + hi) / 2
    s2 = (1 - a) * r + a * y
    return s2, round(a, 4), round(float(roc_auc_score(y, s2)), 4)


def sensitivity_arm(score, y, delta_sens, measured_auc):
    """P0-b：各 AUROC 水平（实测 + 合成目标）下重放下探曲线。"""
    levels = {}
    for label, target in [('measured', None), ('auroc_0.65', 0.65),
                          ('auroc_0.70', 0.70), ('auroc_0.80', 0.80)]:
        if target is None:
            s, alpha, auc = score, 0.0, round(measured_auc, 4)
        else:
            s, alpha, auc = morph_to_auroc(score, y, target)
        rows = descent_curve(s, y, delta_sens)
        enr = {r['coverage_pct']: r['enrichment'] for r in rows}
        top = rows[0]
        pos_total = int(y.sum())
        levels[label] = {
            'target_auroc': target, 'blend_alpha': alpha,
            'achieved_auroc': auc,
            'is_synthetic': target is not None,
            'top10pct': {
                'enrichment': top['enrichment'],
                'det_per_100': top['marginal_det_per_100_cxr'],
                'cxr_per_det': top['cxr_per_marginal_det'],
                'case_capture': round(top['marginal_cases'] / pos_total, 3),
            },
            'deepest_tier_enrichment_ge_1': max(
                [t for t in TIERS if enr[t] is not None and enr[t] >= 1.0],
                default=None),
            'deepest_tier_enrichment_ge_1.5': max(
                [t for t in TIERS if enr[t] is not None and enr[t] >= 1.5],
                default=None),
            'curve': rows,
        }
    return levels


def breakeven_lookup(curve_rows, ci=None, delta_sens=None):
    """P1：本地盈亏平衡查找表。

    边际带准则：推荐覆盖 = 边际检出/百张 ≥ 100/C 的最深分层；
    本地边际检出/百张 = Δsens × p_local × enrichment(t) × 100
    → required_enrichment = 1/(Δsens·p·C)。
    conservative 列用 bootstrap 富集 CI 下界（仅实测排序器水平）。
    """
    if delta_sens is None:
        delta_sens = SENS_SYMPTOM_CXR - SENS_SYMPTOM_ONLY
    enr = {r['coverage_pct']: r['enrichment'] for r in curve_rows}
    enr_lo = ({t: ci[t]['enrichment_ci95'][0] for t in TIERS}
              if ci is not None else None)
    rows = []
    for p in (0.005, 0.01, 0.02, 0.03):
        for c in (20, 50, 100):
            req = 1.0 / (delta_sens * p * c)
            rec_pt = max([t for t in TIERS
                          if enr[t] is not None and enr[t] >= req],
                         default=None)
            rec_cons = (max([t for t in TIERS
                             if enr_lo[t] is not None and enr_lo[t] >= req],
                            default=None)
                        if enr_lo is not None else None)
            holes = None
            if rec_pt is not None:
                bad = [t for t in TIERS
                       if t < rec_pt and enr[t] is not None and enr[t] < req]
                holes = bad if bad else None
            rows.append({
                'prevalence_local': p,
                'acceptable_cxr_per_det': c,
                'required_enrichment': round(req, 2),
                'recommended_coverage_point': rec_pt,
                'recommended_coverage_conservative': rec_cons,
                'nonmonotonic_holes': holes,
            })
    return rows


def seed_robustness(X, y, delta_sens):
    """P2：OOF 折分种子稳健性。

    split seed 变异（SEEDS）、模型 seed 固定 42——隔离数据划分
    随机性；各种子按主口径规则取 AUROC 更高排序器，报告 AUROC
    与 top-10% 富集的种子间范围，与 bootstrap CI 对照判定
    seed 42 是否可作代表值。
    """
    rows = []
    for sd in SEEDS:
        scores, aucs = {}, {}
        skf = StratifiedKFold(5, shuffle=True, random_state=sd)
        for name, mk in [('LR', lambda: LogisticRegression(max_iter=2000))] + (
                [('LGBM', lambda: LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, num_leaves=15,
                    random_state=42, verbose=-1))] if HAS_LGBM else []):
            oof = np.zeros(len(y))
            for tri, tei in skf.split(X, y):
                m = mk().fit(X[tri], y[tri])
                oof[tei] = m.predict_proba(X[tei])[:, 1]
            scores[name] = oof
            aucs[name] = float(roc_auc_score(y, oof))
        best = max(aucs, key=aucs.get)
        c = descent_curve(scores[best], y, delta_sens)
        rows.append({
            'split_seed': sd,
            'best_ranker': best,
            'auroc_lr': round(aucs.get('LR'), 4),
            'auroc_lgbm': round(aucs.get('LGBM'), 4) if HAS_LGBM else None,
            'oof_auroc': round(aucs[best], 4),
            'top10_enrichment': c[0]['enrichment'],
            'deepest_tier_enrichment_ge_1': max(
                [r['coverage_pct'] for r in c
                 if r['enrichment'] is not None and r['enrichment'] >= 1.0],
                default=None),
        })
    enr = [r['top10_enrichment'] for r in rows]
    auc_list = [r['oof_auroc'] for r in rows]
    spread = round(max(enr) - min(enr), 2)
    summary = {
        'top10_enrichment_range': [min(enr), max(enr)],
        'top10_enrichment_spread': spread,
        'oof_auroc_range': [min(auc_list), max(auc_list)],
        'note': ('split seed 变异、模型 seed 固定 42（隔离折分随机性）；'
                 '各种子按主口径规则取 AUROC 更高排序器'),
    }
    return rows, summary


def main():
    # ---------- 腿 1：CRP 域覆盖下探曲线（r1 基础节） ----------
    df = pd.read_csv(CRP_SRC)
    y = df['tb_outcome'].astype(int).values
    X = build_features(df)
    score, best_ranker, aucs = oof_scores(X, y)
    prev_domain = float(y.mean())
    n = len(y)
    pos_total = int(y.sum())

    # 数据口径上限：CAD≥40 在病例中的阳性率（验证规则循环，仅上限）
    cad_obs = df['cad_score'].notna().values
    cad_pos_among_cases = float(
        ((df['cad_score'] >= 40) & (df['tb_outcome'] == 1)).sum()
        / max(int((cad_obs & (y == 1)).sum()), 1))

    delta_sens = SENS_SYMPTOM_CXR - SENS_SYMPTOM_ONLY  # 0.32 主口径

    curve = descent_curve(score, y, delta_sens)

    # 对照策略（非模型排序）
    comparators = []
    for label, mask in [
        ('is_high_risk_flag(重点人群代理)', df['is_high_risk'].values == 1),
        ('deploy_threshold>=0.08(高风险层)', score >= TH_HIGH),
        ('deploy_threshold>=0.03(中风险+)', score >= TH_MEDIUM),
        ('symptom_only(仅症状人群)', df['has_symptoms'].values == 1),
    ]:
        m = np.asarray(mask)
        pos = int(y[m].sum())
        comparators.append({
            'policy': label,
            'n_cxr': int(m.sum()),
            'cases': pos,
            'prevalence': round(pos / max(int(m.sum()), 1), 4),
            'enrichment': round((pos / max(int(m.sum()), 1)) / prev_domain, 2),
            'marginal_det_per_100_cxr': round(delta_sens * pos / max(int(m.sum()), 1) * 100, 2),
        })

    # 模型定向 vs 无定向（随机覆盖 = 域患病率）浓缩指标
    top10 = curve[0]
    random_det_per_100 = round(delta_sens * prev_domain * 100, 2)
    model_advantage = round(top10['marginal_det_per_100_cxr'] / random_det_per_100, 1) if random_det_per_100 else None

    # ---------- 腿 3（P0-a）：bootstrap CI ----------
    ci, anomaly = bootstrap_ci(score, y, delta_sens)
    first_sig_below_untargeted = next(
        (t for t in TIERS if ci[t]['enrichment_ci95'][1] < 1.0), None)

    # ---------- 腿 4（P0-b）：排序器质量敏感性 ----------
    sens_levels = sensitivity_arm(score, y, delta_sens, aucs[best_ranker])

    # ---------- 腿 5（P1）：本地盈亏平衡查找表 ----------
    lookup_measured = breakeven_lookup(curve, ci=ci)
    lookup_d02 = breakeven_lookup(curve, delta_sens=DELTA_SENS_CONSERVATIVE)
    lookup_070 = breakeven_lookup(sens_levels['auroc_0.70']['curve'])
    lookup_080 = breakeven_lookup(sens_levels['auroc_0.80']['curve'])

    # ---------- 腿 6（P2）：折分种子稳健性 ----------
    seed_rows, seed_summary = seed_robustness(X, y, delta_sens)
    seed_ci = ci[10]['enrichment_ci95']
    seed_summary['seed42_representative'] = bool(
        seed_summary['top10_enrichment_spread']
        <= round(seed_ci[1] - seed_ci[0], 2))
    seed_summary['comparison'] = (
        f"top-10% 富集种子间极差 {seed_summary['top10_enrichment_spread']}"
        f" vs bootstrap CI 宽度 {round(seed_ci[1] - seed_ci[0], 2)}"
        f"（{seed_ci}）→ seed 42 "
        f"{'可' if seed_summary['seed42_representative'] else '不可'}"
        '作为代表值引用')

    # ---------- 腿 2：Kenya 症状→确诊（仅症状筛查盲区） ----------
    kdf = pd.read_csv(KENYA_SRC, low_memory=False)
    kdf = kdf[kdf['Coughing'].notna()].copy()

    def is_pos(s):
        return (s.astype(str).str.strip().str.upper() == 'POS')

    ky = (is_pos(kdf['Smearpositive'].fillna(''))
          | is_pos(kdf['Xpertpositive'].fillna(''))
          | is_pos(kdf['Culturepositive'].fillna(''))).astype(int).values
    nsym = np.zeros(len(kdf))
    for c in KENYA_SYMPTOMS:
        nsym += (pd.to_numeric(kdf[c], errors='coerce').fillna(0) > 0).astype(int).values
    asym = nsym == 0
    cases = ky == 1
    n_asym_cases = int((cases & asym).sum())
    n_sym_cases = int((cases & ~asym).sum())

    kenya = {
        'n_analyzed': int(len(kdf)), 'n_cases': int(cases.sum()),
        'asymptomatic_cases': n_asym_cases,
        'asymptomatic_case_fraction': round(n_asym_cases / max(int(cases.sum()), 1), 4),
        'prevalence_symptomatic': round(float(ky[~asym].mean()), 5),
        'prevalence_asymptomatic': round(float(ky[asym].mean()), 5),
        'symptom_enrichment': round(float(ky[~asym].mean()) / max(float(ky[asym].mean()), 1e-9), 1),
        'symptom_screen_miss_upper': (
            '记录症状口径下 26.8% 确诊病例无症状——仅症状筛查的结构性盲区；'
            '与 WHO 85% vs 53% 口径方向一致（症状单筛漏 ~47%），数值为'
            '本调查记录口径'),
        'xray_field_disclosure': (
            'XRayExam 字段语义不明（=1 者 97% 有症状、仅 58.7% 触发痰检、'
            '90 例无症状确诊全部 =0），不可作胸片异常代理；Kenya 只贡献'
            '症状→确诊分解'),
        'eligibility_disclosure': (
            '90 例无症状确诊全部 SputumRequestDone=1 → 存在未记录送检'
            '通道，无症状比例非严格下界'),
    }

    # ---------- 汇总 ----------
    res = {
        'date': '2026-09-06',
        'revision': ('r2.1 2026-09-06：bootstrap CI（P0-a）/排序器敏感性'
                     '（P0-b）/盈亏平衡查找表（P1）/折分种子稳健性'
                     '（P2）；基础节（曲线/对照/Kenya/点估计裁决）'
                     '与 r1 逐位一致'),
        'question': ('接触者胸片覆盖从高风险层逐级下探，边际检出/'
                     '百张胸片在哪个分层跌破可接受线？（政策参数，'
                     '非模型入模问题——D2 分支甲维持不入模）'),
        'crp_leg': {
            'dataset': 'ml_training_crp_real.csv（LSHTM ACF 验证子集）',
            'n': int(n), 'n_positive': pos_total,
            'prevalence_domain': round(prev_domain, 4),
            'ranker': best_ranker, 'ranker_oof_auroc': {k: round(v, 4) for k, v in aucs.items()},
            'ranking_features': '22 维基线（不含 cad/crp，时序前置性）',
            'detection_model': {
                'primary': f'文献校准：Δsens={delta_sens:.2f}'
                           f'（症状+胸片 {SENS_SYMPTOM_CXR} vs 仅症状 {SENS_SYMPTOM_ONLY}，'
                           'WHO 2021 密接筛查口径）',
                'conservative': f'Δsens={DELTA_SENS_CONSERVATIVE:.2f}',
                'data_derived_upper_bound': f'CAD≥40 病例阳性率 '
                                            f'{cad_pos_among_cases:.2f}'
                                            '（验证规则循环，仅上限）',
                'uniform_across_tiers_disclosure': (
                    'Δsens 均匀施加于所有分层；层特异性检出率不可测，'
                    '层间差异全部来自排序富集'),
            },
            'coverage_curve_model_ranked': curve,
            'comparators': comparators,
            'comparator_dimension_note': (
                '对照策略的检出/百张为覆盖集均值口径，曲线各层为边际带'
                '口径——两量纲并存，对照值不可与单层边际值直接比较'),
            'comparator_bias_disclosure': (
                'CRP 验证子集资格规则=症状或CAD≥40 → 子集内"仅症状"'
                '对照的富集被选择压平（0.0664≈域基线 0.0658），不可解读为'
                '"症状无定向价值"；症状富集的诚实来源是 Kenya 全人群口径'
                '（症状阳性患病率 ~1.03% vs 无症状 ~0.23%，约 4.5×）'),
            'curve_nonmonotonic_disclosure': (
                '曲线非单调（40% 层富集高于 30% 层；底部 10% 层反常回升'
                '至 11 例）——弱排序器（OOF AUROC 0.605）的真实噪声，'
                '不作平滑处理，按原样报告；单点"跌破阈值"读数因此为'
                '区间性参考而非精确拐点'),
            'bootstrap_ci': {
                'method': (f'患者级 bootstrap {N_BOOT} 次（RandomState(42)，'
                           '分层边界随每次重采样重排）；95% 百分位区间；'
                           '条件于 seed 42 单次 OOF 排序器，不含模型训练'
                           '随机性'),
                'per_tier': {str(t): ci[t] for t in TIERS},
                'anomaly_tests': anomaly,
                'first_tier_significantly_below_untargeted': (
                    first_sig_below_untargeted),
            },
            'ranker_sensitivity': {
                'method': ('保序标签混合：OOF 分数 rank 归一后与标签按 α '
                           '混合，二分搜索命中目标 AUROC；合成变换——'
                           '病例谱固定、排序器改进表现为分数-标签对齐度'
                           '提升，不模拟特征空间变化；0.65/0.70/0.80 '
                           '曲线为合成对照、非实测预测'),
                'levels': sens_levels,
            },
            'untargeted_det_per_100_cxr': random_det_per_100,
            'model_top10_vs_untargeted_advantage': model_advantage,
            'seed_robustness': {
                'method': ('OOF 折分种子 {SEEDS}（模型 seed 固定 42，'
                           '隔离折分随机性）；各种子按主口径规则取 '
                           'AUROC 更高排序器').replace('{SEEDS}', str(SEEDS)),
                'per_seed': seed_rows,
                'summary': seed_summary,
            },
            'transfer_rule': ('绝对数不可外推（验证子集患病率 6.6% 被富集）；'
                              '本地迁移：边际检出/百张 = Δsens × 本地接触者'
                              '患病率 × enrichment(t) × 100'),
        },
        'breakeven_lookup': {
            'method': ('边际带准则：推荐覆盖 = 边际检出/百张 ≥ 100/C 的'
                       '最深分层（C=可接受胸片/检出）；required_enrichment '
                       '= 1/(Δsens·p·C)；conservative 列用 bootstrap 富集 '
                       'CI 下界（仅实测排序器水平）'),
            'primary_delta_sens': delta_sens,
            'measured_ranker': lookup_measured,
            'conservative_delta_sens_020': lookup_d02,
            'synthetic_auroc_070_point': lookup_070,
            'synthetic_auroc_080_point': lookup_080,
            'note': ('0.70/0.80 为 P0-b 合成排序器对照（点估计）；'
                     'nonmonotonic_holes 标注推荐层之前未达标的中间层'
                     '（点估计口径的噪声产物，判定见 bootstrap_ci.'
                     'anomaly_tests）'),
        },
        'kenya_leg': kenya,
        'scope': ('扩展对象限于 ≥15 岁成人接触者（CRP 域全员 ≥15 可整体'
                  '对齐）；<15 岁维持症状/TST 强阳/IGRA 阳性→胸片的触发式'
                  '（辐射与国内共识双重考虑）'),
        'loop_precheck': ('胸片/CAD 结果不得反向参与"谁做分子检测"的资格'
                          '决定，否则复刻 CRP 域验证规则循环，本地数据从此'
                          '不可用于诚实评估'),
        'verdict': None,  # 运行后按曲线填
    }

    # 决策读数：主口径下边际检出/百张跌破 1.0 与跌破无定向基线的分层
    first_below_1 = next((r['coverage_pct'] for r in curve
                          if r['marginal_det_per_100_cxr'] is not None
                          and r['marginal_det_per_100_cxr'] < 1.0), None)
    first_below_untargeted = next(
        (r['coverage_pct'] for r in curve
         if r['marginal_det_per_100_cxr'] is not None
         and r['marginal_det_per_100_cxr'] < random_det_per_100), None)
    res['verdict'] = {
        'first_tier_below_1_per_100': first_below_1,
        'first_tier_below_untargeted': first_below_untargeted,
        'first_tier_significantly_below_untargeted': (
            first_sig_below_untargeted),
        'tier60_p_enrichment_below_1': ci[60]['p_enrichment_below_1'],
        'anomaly_40pct': anomaly['verdict_40pct'],
        'anomaly_bottom_rebound': anomaly['verdict_bottom_rebound'],
        'note': ('跌破无定向基线 = 模型排序不再提供定向价值，继续下探'
                 '等同随机覆盖；点估计读数与 CI 读数均基于富集域，'
                 '迁移按 transfer_rule 参数化'),
    }

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # 控制台决策表（政策层可读）
    print(f"排序器：{best_ranker} OOF AUROC "
          f"{ {k: round(v, 3) for k, v in aucs.items()} }（弱排序器口径）")
    print(f"域患病率 {prev_domain:.3%} | 无定向基线 "
          f"{random_det_per_100}/百张 | top10 优势 ×{model_advantage}")
    print(f"数据口径上限（循环，仅上限）：CAD≥40 病例阳性率 "
          f"{cad_pos_among_cases:.1%}")
    print("\n模型排序覆盖下探曲线（主口径 Δsens=0.32 / 保守 0.20）：")
    print(f"{'覆盖':>4} {'累计胸片':>8} {'边际病例':>8} {'边际患病率':>10} "
          f"{'富集':>6} {'检出/百张':>10} {'保守':>8} {'胸片/检出':>10}")
    for r in curve:
        print(f"{r['coverage_pct']:>4}% {r['n_cxr_cum']:>8} "
              f"{r['marginal_cases']:>8} {r['marginal_prevalence']:>10.4f} "
              f"{r['enrichment']:>6} {r['marginal_det_per_100_cxr']:>10} "
              f"{r['marginal_det_per_100_cxr_conservative']:>8} "
              f"{r['cxr_per_marginal_det']:>10}")
    print("\n对照策略：")
    for c in comparators:
        print(f"  {c['policy']}: n={c['n_cxr']} 病例={c['cases']} "
              f"患病率={c['prevalence']:.4f} 检出/百张={c['marginal_det_per_100_cxr']}")

    # ---------- P0-a：bootstrap CI 控制台 ----------
    print(f"\nP0-a bootstrap CI（患者级 {N_BOOT} 次，条件于 seed 42 OOF 排序器）：")
    print(f"{'覆盖':>4} {'检出/百张 CI95':>18} {'富集 CI95':>16} {'P(富集<1)':>10}")
    for t in TIERS:
        c = ci[t]
        print(f"{t:>4}% [{c['det_per_100_ci95'][0]:>6.2f},{c['det_per_100_ci95'][1]:>6.2f}]"
              f"      [{c['enrichment_ci95'][0]:>5.2f},{c['enrichment_ci95'][1]:>5.2f}]"
              f"      {c['p_enrichment_below_1']:>10.3f}")
    print(f"  40% 层反常富集判定：{anomaly['verdict_40pct']}"
          f"（P(enr40>enr10)={anomaly['p_enr40_gt_enr10']}, "
          f"P(enr40>enr30)={anomaly['p_enr40_gt_enr30']}, "
          f"P(enr40>enr50)={anomaly['p_enr40_gt_enr50']}）")
    print(f"  底部 10% 层回升判定：{anomaly['verdict_bottom_rebound']}"
          f"（P(bottom>80%层)={anomaly['p_bottom_gt_tier80']}）")
    print(f"  统计显著低于无定向的最浅层：{first_sig_below_untargeted}%"
          f"（富集 CI 上界 < 1）")

    # ---------- P0-b：敏感性控制台 ----------
    print("\nP0-b 排序器质量敏感性（合成保序标签混合，非实测预测）：")
    print(f"{'AUROC':>7} {'α混合':>7} {'top10捕获':>9} {'top10富集':>9} "
          f"{'top10检出/百张':>13} {'top10胸片/检出':>13} {'富集≥1最深':>10} {'富集≥1.5最深':>11}")
    for label in ['measured', 'auroc_0.65', 'auroc_0.70', 'auroc_0.80']:
        lv = sens_levels[label]
        t10 = lv['top10pct']
        print(f"{lv['achieved_auroc']:>7} {lv['blend_alpha']:>7} "
              f"{t10['case_capture']:>9.3f} {t10['enrichment']:>9} "
              f"{t10['det_per_100']:>13} {t10['cxr_per_det']:>13} "
              f"{str(lv['deepest_tier_enrichment_ge_1']) + '%':>10} "
              f"{str(lv['deepest_tier_enrichment_ge_1.5']) + '%':>11}")

    # ---------- P1：查找表控制台 ----------
    print("\nP1 本地盈亏平衡查找表（Δsens=0.32，实测排序器，"
          "推荐覆盖层：点估计/CI下界保守；—=不可行）：")
    print(f"{'p_local':>8} | {'C=20':>12} {'C=50':>12} {'C=100':>12}")
    for p in (0.005, 0.01, 0.02, 0.03):
        cells = []
        for c in (20, 50, 100):
            row = next(r for r in lookup_measured
                       if r['prevalence_local'] == p
                       and r['acceptable_cxr_per_det'] == c)
            pt = f"{row['recommended_coverage_point']}%" if row['recommended_coverage_point'] else '—'
            cons = (f"{row['recommended_coverage_conservative']}%"
                    if row['recommended_coverage_conservative'] else '—')
            cells.append(f"{pt}/{cons:>5}")
        print(f"{p:>8.3f} | {cells[0]:>12} {cells[1]:>12} {cells[2]:>12}")
    for label, lut in [('auroc_0.70', lookup_070), ('auroc_0.80', lookup_080)]:
        n_viable = sum(1 for r in lut if r['recommended_coverage_point'])
        print(f"  合成 {label} 对照（点估计）：可行单元格 {n_viable}/12")

    # ---------- P2：种子稳健性控制台 ----------
    print(f"\nP2 折分种子稳健性（模型 seed 固定 42，主口径排序器选择规则）：")
    print(f"{'seed':>6} {'排序器':>6} {'AUROC':>8} {'top10富集':>10} {'富集≥1最深':>10}")
    for r in seed_rows:
        print(f"{r['split_seed']:>6} {r['best_ranker']:>6} {r['oof_auroc']:>8} "
              f"{r['top10_enrichment']:>10} "
              f"{str(r['deepest_tier_enrichment_ge_1']) + '%':>10}")
    print(f"  top-10% 富集种子间范围 {seed_summary['top10_enrichment_range']}"
          f"（极差 {seed_summary['top10_enrichment_spread']}）"
          f" | OOF AUROC 范围 {seed_summary['oof_auroc_range']}")
    print(f"  {seed_summary['comparison']}")

    print(f"\nKenya：确诊 {kenya['n_cases']} 例，无症状 {kenya['asymptomatic_cases']} 例"
          f"（{kenya['asymptomatic_case_fraction']:.1%}）——仅症状筛查盲区")
    print(f"\n[OK] {OUT}")


if __name__ == '__main__':
    main()
