#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""越南 LAM 外部验证 v2：死特征修复后重验 + 暴露映射 + 谱系再校准。

数据：PROVE_TB Vietnam（Hoa NB, Fajans M, et al. 2024, PLOS Glob Public Health
4:e0003891；Harvard Dataverse DOI 10.7910/DVN/AOL0LP，CC0 1.0）。
745 名 HIV 阴性有症状成人，MRS 标签（Xpert Ultra + MGIT 培养，
404 阴 / 341 阳）。

v2（2026-08-22 晚，死特征修复后）：
- 场景生成器 _sample_contact 已补 cumulative_exposure（部署同款公式
  duration/60*freq*span），两个 scenario 模型已重训（185119/185411）；
- 新增暴露映射变体：int_lvnowtb（家中现患 TB，45/745）→ 家庭典型
  接触暴露档案（生成器 family 参数中位数：freq=19/月、dur=270min、
  span=13 月、close、vent=3、closed，cumulative≈1108.5 h），
  其余行暴露保持 0/中性——检验激活后的暴露特征能否承接家庭暴露信号；
- 新增谱系再校准（50/50 切分 seed 42，校准半区拟合、测试半区报告，
  无泄露）：logistic 重校准（Platt，截距+斜率）与患病率先验修正
  （训练先验→验证域先验的 odds 重标定，目标先验取校准半区经验率）。
  两者均为单调变换：修复 Brier/均值概率（阈值可用性），AUROC 不变
  ——判别力上限仍由特征层决定。

验证对象：
- kenya_hyperopt : 164601（Kenya 真实患病率调查，训练内 0.7537）
- scenario_mech  : 185119（场景对齐·机制标签·暴露修复后，训练内 0.8395）
- scenario_clin  : 185411（场景对齐·临床标签·暴露修复后，训练内 0.7612）

归档：TrainingLogger(model_type='ml')，bucket external_vietnam_lam_mrs，
追加不覆盖（与 v1 run 20260822_183644 同桶可比）。

用法：
    python data/validate_vietnam_lam.py [--bootstrap 1000]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)
sys.path.insert(0, HERE)

MODELS_DIR = os.path.join(HERE, "training_archive", "models")
DATA_PATH = os.path.join(HERE, "raw", "vietnam_lam", "lam_paper_data.xlsx")
MAIN_SHEET = "PROVE_TB_VN_publicdataset5.12.2"

MODEL_SPECS = {
    "kenya_hyperopt": "best_ml_20260822_164601_ns63050_cv5.joblib",
    # v2：死特征修复后重训的 checkpoint
    "scenario_mech": "best_ml_20260822_185119_ns12476_cv5.joblib",
    "scenario_clin": "best_ml_20260822_185411_ns12476_cv5.joblib",
}
# 各模型训练集阳性率（先验，供患病率修正；来自归档 run 记录）
TRAIN_PRIORS = {
    "kenya_hyperopt": 336 / 63050,    # 0.0053
    "scenario_mech": 2721 / 12476,    # 0.2181
    "scenario_clin": 3324 / 12476,    # 0.2664
}

SYMPTOM_COLS = ["cough_c", "feverchill_c", "nitesweat_c",
                "wtloss_c", "lossapp_c", "fatigue_c"]

# 家庭典型接触暴露档案：场景生成器 family 参数区间的中位数
# （freq 8-29→19，dur 60-479→270，span 1-25→13；close/vent3/closed
# 为 family 众数档），cumulative = 270/60*19*13 ≈ 1108.5 小时。
FAMILY_EXPOSURE_PROFILE = {
    "cumulative_exposure": 19 * 270 / 60.0 * 13,
    "freq_density": 19,
    "single_duration": 270,
    "time_span": 13,
    "contact_distance": "close",
    "ventilation": 3,
    "exposure_setting": "closed",
}


# ---------------------------------------------------------------------------
# 数据加载与映射
# ---------------------------------------------------------------------------

def load_records():
    """读取主 sheet → (行列表, 元信息)。

    每行含两套 contact dict（主映射 / 暴露映射）与两套标签。
    """
    import pandas as pd

    df = pd.read_excel(DATA_PATH, sheet_name=MAIN_SHEET)
    assert len(df) == 745, f"行数异常: {len(df)}"

    records = []
    n_sym_missing = 0
    n_exposed = 0
    for _, row in df.iterrows():
        age = float(row["char_age"])
        if np.isnan(age):
            continue
        sym_vals = [row.get(c, np.nan) for c in SYMPTOM_COLS]
        n_sym_missing += int(sum(np.isnan(v) for v in sym_vals))
        has_symptoms = int(any(not np.isnan(v) and v > 0 for v in sym_vals))

        # int_lvnowtb：家中现患 TB（Yes=45，No=692，Dont know=8→按 No）
        lvnow = str(row.get("int_lvnowtb.factor", "No"))
        exposed = int(lvnow.strip().startswith("Yes"))
        n_exposed += exposed

        def _contact(expo):
            c = {
                "age": int(age),
                "has_symptoms": has_symptoms,
                "bcg_vaccine": 1,
                "has_tb": 0,
                "past_illness": 0,
                "past_illness_type": "none",
                "is_high_risk": int(age >= 65),
                "cumulative_exposure": 0,
                "single_duration": 0,
                "freq_density": 0,
                "time_span": 0,
                "contact_distance": "medium",
                "ventilation": 3,
                "exposure_setting": "general",
            }
            if expo:
                c.update(FAMILY_EXPOSURE_PROFILE)
            return c

        records.append({
            "contact": _contact(False),        # 主映射（与 v1 可比）
            "contact_expo": _contact(exposed),  # 暴露映射变体
            "y_mrs": int(row["MRSref"]),
            "y_crs": int(row["CRSref"]),
        })

    meta = {
        "n": len(records),
        "n_mrs_pos": int(sum(r["y_mrs"] for r in records)),
        "n_crs_pos": int(sum(r["y_crs"] for r in records)),
        "n_sym_missing": n_sym_missing,
        "n_exposed": n_exposed,
    }
    return records, meta


# ---------------------------------------------------------------------------
# 打分与指标
# ---------------------------------------------------------------------------

def score_model(ml_path, records, contact_key="contact"):
    """加载一个 ML checkpoint，对全部行打分（ensemble，0-1 概率）。"""
    from tb_risk.scoring.predictor import MLRiskPredictor

    predictor = MLRiskPredictor()
    assert predictor.load_model(ml_path), f"模型加载失败: {ml_path}"
    assert predictor.is_trained, "is_trained=False"

    ens = []
    for r in records:
        pred = predictor.predict_risk(r[contact_key], "family")
        assert pred and "ensemble" in pred, "predict_risk 返回异常"
        ens.append(float(pred["ensemble"]["risk_probability"]) / 100.0)
    return np.array(ens)


def _metrics(y, p):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss)
    y, p = np.asarray(y), np.asarray(p)
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "mean_pred": float(p.mean()),
    }


def _auroc_bootstrap_ci(y, p, n_boot=1000, seed=42):
    """AUROC 的 percentile bootstrap 95% CI。"""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, p[idx]))
    if not aucs:
        return None, None
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def _delong_test(y, p1, p2):
    """配对 DeLong 检验（复用 evaluate_ensemble 实现）。"""
    from evaluate_ensemble import _delong_test as _impl
    return _impl(y, p1, p2)


# ---------------------------------------------------------------------------
# 谱系再校准（50/50 无泄露切分）
# ---------------------------------------------------------------------------

def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def recalibrate(y, p, train_prior, seed=42):
    """logistic 重校准 + 患病率先验修正。

    50/50 切分（seed 固定）：校准半区拟合 Platt 截距/斜率与目标先验，
    测试半区统一报告。两法均为单调变换——修复校准（Brier/均值概率），
    AUROC 严格不变，判别力上限不受影响。

    返回测试半区指标 dict。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score

    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    half = len(y) // 2
    cal_i, test_i = idx[:half], idx[half:]

    # 1) logistic 重校准（Platt）：logit(p) → y
    lr = LogisticRegression(C=1e6)          # 近似无正则的 MLE
    lr.fit(_logit(p[cal_i]).reshape(-1, 1), y[cal_i])
    p_recal = lr.predict_proba(_logit(p[test_i]).reshape(-1, 1))[:, 1]

    # 2) 患病率先验修正：训练先验 → 目标先验的 odds 重标定
    #    目标先验取校准半区经验率（不触碰测试半区标签）
    pi_t = float(train_prior)
    pi_v = float(y[cal_i].mean())
    odds = p[test_i] / (1 - p[test_i])
    odds *= (pi_v / (1 - pi_v)) / (pi_t / (1 - pi_t))
    p_prior = odds / (1 + odds)

    return {
        "n_test": int(len(test_i)),
        "target_prior": pi_v,
        "brier_raw": float(brier_score_loss(y[test_i], p[test_i])),
        "brier_recal": float(brier_score_loss(y[test_i], p_recal)),
        "brier_prior": float(brier_score_loss(y[test_i], p_prior)),
        "mean_raw": float(p[test_i].mean()),
        "mean_recal": float(p_recal.mean()),
        "mean_prior": float(p_prior.mean()),
        "auroc_raw": float(roc_auc_score(y[test_i], p[test_i])),
        "auroc_recal": float(roc_auc_score(y[test_i], p_recal)),
        "platt_slope": float(lr.coef_[0][0]),
        "platt_intercept": float(lr.intercept_[0]),
    }


# ---------------------------------------------------------------------------
# 报告与归档
# ---------------------------------------------------------------------------

def _print_table(title, rows, pos_rate):
    print(f"\n=== {title} ===")
    print(f"{'模型':<16}{'AUROC':>8}{'AUPRC':>8}{'lift':>7}"
          f"{'Brier':>8}{'均值p':>8}")
    for name, m in rows:
        lift = m["AUPRC"] / pos_rate
        print(f"{name:<16}{m['AUROC']:>8.4f}{m['AUPRC']:>8.4f}{lift:>7.2f}"
              f"{m['Brier']:>8.4f}{m['mean_pred']:>8.4f}")


def archive(metrics, params, duration):
    """TrainingLogger 归档（model_type='ml'，bucket 同 v1，追加不覆盖）。"""
    from train_with_public_data import get_logger

    dataset_info = {
        "source": "external_vietnam_lam_mrs",
        "n_samples": params["n_samples"],
        "real_data": True,
        "provenance": (
            "PROVE_TB Vietnam (Hoa NB et al. 2024, PLOS Glob Public Health "
            "4:e0003891; Harvard Dataverse DOI 10.7910/DVN/AOL0LP, CC0 1.0); "
            "745 HIV-negative symptomatic adults, MRS reference standard. "
            "v2 re-validation after dead-feature fix (cumulative_exposure "
            "now varies in scenario training); exposure variant maps "
            "int_lvnowtb household exposure; spectrum-shift recalibration "
            "(Platt + prevalence prior correction, 50/50 leak-free split)"),
    }
    logger = get_logger()
    entry = logger.log(
        model_type="ml", params=params, metrics=metrics,
        training_duration=duration, status="success", dataset=dataset_info)
    print(f"[归档] run_id={entry.run_id}（bucket=external_vietnam_lam_mrs）")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="越南 LAM 外部验证 v2")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="bootstrap 次数（0 关闭 CI）")
    args = parser.parse_args()

    t0 = time.time()
    records, meta = load_records()
    y_mrs = np.array([r["y_mrs"] for r in records])
    y_crs = np.array([r["y_crs"] for r in records])
    pos_mrs = float(y_mrs.mean())
    pos_crs = float(y_crs.mean())
    print(f"[数据] n={meta['n']}  MRS 阳 {meta['n_mrs_pos']}"
          f"（{pos_mrs:.3f}）  CRS 阳 {meta['n_crs_pos']}"
          f"（{pos_crs:.3f}）  家庭暴露 {meta['n_exposed']}")
    print(f"       症状字段缺失 {meta['n_sym_missing']} 个单元格（nan→0）")

    # 家庭暴露单变量信号上限（int_lvnowtb 本身的判别力）
    expo = np.array([r["contact_expo"]["cumulative_exposure"] > 0
                     for r in records], dtype=int)
    from sklearn.metrics import roc_auc_score
    print(f"[单变量] int_lvnowtb（家庭暴露，n={expo.sum()}）"
          f" AUROC={roc_auc_score(y_mrs, expo):.4f}")

    # --- 主映射 + MRS：三模型对照（与 v1 可比）---
    scores = {}
    for tag, fname in MODEL_SPECS.items():
        path = os.path.join(MODELS_DIR, fname)
        t = time.time()
        scores[tag] = score_model(path, records, "contact")
        print(f"[打分] {tag}: {fname}（{time.time() - t:.1f}s）")

    rows = []
    for tag in MODEL_SPECS:
        m = _metrics(y_mrs, scores[tag])
        rows.append((tag, m))
    _print_table("主映射 × MRS（与 v1 run 183644 可比）", rows, pos_mrs)

    # bootstrap CI
    ci = {}
    if args.bootstrap > 0:
        for tag in MODEL_SPECS:
            lo, hi = _auroc_bootstrap_ci(
                y_mrs, scores[tag], args.bootstrap, seed=42)
            ci[tag] = (lo, hi)
        print("\n[bootstrap 95% CI]（1000 次，seed 42）")
        for tag, (lo, hi) in ci.items():
            print(f"  {tag:<16}{lo:.4f} - {hi:.4f}")

    # 配对 DeLong（三模型互比）
    delong = {}
    print("\n[配对 DeLong]（主映射 × MRS）")
    for a, b in [("kenya_hyperopt", "scenario_mech"),
                 ("kenya_hyperopt", "scenario_clin"),
                 ("scenario_mech", "scenario_clin")]:
        d = _delong_test(y_mrs, scores[a], scores[b])
        delong[f"{a}_vs_{b}"] = d
        if d:
            print(f"  {a} vs {b}: Δ={d['auc1'] - d['auc2']:+.4f}"
                  f"  z={d['z']:+.3f}  p={d['p_value']:.4f}")

    # --- 暴露映射变体：新模型 + int_lvnowtb → 家庭典型暴露 ---
    expo_rows = []
    scores_expo = {}
    for tag, fname in MODEL_SPECS.items():
        path = os.path.join(MODELS_DIR, fname)
        scores_expo[tag] = score_model(path, records, "contact_expo")
        m = _metrics(y_mrs, scores_expo[tag])
        expo_rows.append((tag, m))
    _print_table("暴露映射变体（int_lvnowtb→家庭典型暴露）× MRS",
                 expo_rows, pos_mrs)

    # --- 谱系再校准（暴露映射变体上，50/50 无泄露）---
    print("\n[谱系再校准]（暴露变体分数，校准半区拟合/测试半区报告，seed 42）")
    print(f"{'模型':<16}{'Brier原始':>10}{'Brier重校':>10}{'Brier先验':>10}"
          f"{'均值p原始':>10}{'均值p重校':>10}")
    recal = {}
    for tag in MODEL_SPECS:
        recal[tag] = recalibrate(y_mrs, scores_expo[tag],
                                 TRAIN_PRIORS[tag], seed=42)
        r = recal[tag]
        print(f"{tag:<16}{r['brier_raw']:>10.4f}{r['brier_recal']:>10.4f}"
              f"{r['brier_prior']:>10.4f}{r['mean_raw']:>10.4f}"
              f"{r['mean_recal']:>10.4f}")
        print(f"{'':<16}（AUROC 原始 {r['auroc_raw']:.4f} / 重校 "
              f"{r['auroc_recal']:.4f}；斜率>0 时排序不变，斜率<0 时翻转"
              f"=分数与结局无稳定方向关联的诊断信号；Platt 斜率 "
              f"{r['platt_slope']:.2f} 截距 {r['platt_intercept']:.2f}）")

    # --- CRS 标签（暴露变体，第二金标稳健性）---
    crs_rows = []
    for tag in MODEL_SPECS:
        m = _metrics(y_crs, scores_expo[tag])
        crs_rows.append((tag, m))
    _print_table("暴露映射变体 × CRS 标签（复合金标）", crs_rows, pos_crs)

    duration = time.time() - t0

    # --- 归档 ---
    kenya = rows[0][1]
    metrics = {
        "AUROC": kenya["AUROC"],          # 主指标：Kenya 模型 MRS 外推
        "AUPRC": kenya["AUPRC"],
        "Brier": kenya["Brier"],
        "positive_rate": pos_mrs,
        "auprc_lift": kenya["AUPRC"] / pos_mrs,
        "mean_pred": kenya["mean_pred"],
        "int_lvnowtb_univariate_AUROC": float(roc_auc_score(y_mrs, expo)),
    }
    if ci.get("kenya_hyperopt"):
        metrics["auroc_ci_low"] = ci["kenya_hyperopt"][0]
        metrics["auroc_ci_high"] = ci["kenya_hyperopt"][1]
    for tag in ("scenario_mech", "scenario_clin"):
        m = next(m for t, m in rows if t == tag)
        metrics[f"{tag}_AUROC"] = m["AUROC"]
        if ci.get(tag):
            metrics[f"{tag}_ci_low"] = ci[tag][0]
            metrics[f"{tag}_ci_high"] = ci[tag][1]
    for k, d in delong.items():
        if d:
            metrics[f"delong_{k}_p"] = d["p_value"]
            metrics[f"delong_{k}_delta"] = d["auc1"] - d["auc2"]
    for tag, m in expo_rows:
        metrics[f"expo_{tag}_AUROC"] = m["AUROC"]
        metrics[f"expo_{tag}_AUPRC"] = m["AUPRC"]
    for tag, r in recal.items():
        for k in ("brier_raw", "brier_recal", "brier_prior",
                  "mean_raw", "mean_recal", "platt_slope",
                  "platt_intercept", "auroc_raw", "auroc_recal"):
            metrics[f"recal_{tag}_{k}"] = r[k]
    for tag, m in crs_rows:
        metrics[f"crs_expo_{tag}_AUROC"] = m["AUROC"]

    params = {
        "n_samples": meta["n"],
        "label": "MRSref (Xpert Ultra + MGIT culture)",
        "label_positive": meta["n_mrs_pos"],
        "n_household_exposed": meta["n_exposed"],
        "models": {t: f for t, f in MODEL_SPECS.items()},
        "dead_feature_fix": (
            "cumulative_exposure added to _sample_contact (deployment "
            "formula duration/60*freq*span); scenario models retrained "
            "on CSVs with exposure variation (was constant 0 in all "
            "training sets)"),
        "bootstrap": args.bootstrap,
        "bootstrap_seed": 42,
        "mapping": {
            "age": "char_age",
            "has_symptoms": "any of 6 symptoms (nan->0)",
            "is_high_risk": "age>=65 (HIV all negative)",
            "bcg_vaccine": "1 (Vietnam universal BCG)",
            "past_illness": "0 (no prior-TB field)",
            "primary_exposure": "0/neutral (v1-comparable)",
            "exposure_variant": "int_lvnowtb Yes -> family-typical "
                                "profile (freq19/dur270/span13/close/"
                                "vent3/closed, cumulative~1108.5h)",
        },
        "recalibration": (
            "50/50 split seed 42; Platt logistic recalibration and "
            "prevalence prior correction (train prior -> calibration-"
            "half empirical prior); both monotone -> AUROC invariant, "
            "Brier/mean-prob reported on held-out half"),
        "train_priors": {t: round(p, 4) for t, p in TRAIN_PRIORS.items()},
        "limitations": (
            "spectrum shift: symptomatic clinic cohort 45.8% positive vs "
            "Kenya prevalence survey 0.53% / scenario 21.8%; has_symptoms "
            "99.3% constant (ceiling effect); age direction reversed "
            "(univariate AUROC 0.414)"),
    }
    archive(metrics, params, duration)

    # --- 结论提示 ---
    print("\n=== 判读要点 ===")
    kenya_e = next(m for t, m in expo_rows if t == "kenya_hyperopt")
    mech_e = next(m for t, m in expo_rows if t == "scenario_mech")
    print(f"主映射：Kenya {kenya['AUROC']:.4f} / scenario_mech "
          f"{rows[1][1]['AUROC']:.4f}（v1 对照：0.537/0.497）")
    print(f"暴露映射：Kenya {kenya_e['AUROC']:.4f} / scenario_mech "
          f"{mech_e['AUROC']:.4f}——家庭暴露 45 例的信号承接情况")
    print(f"再校准：Brier 与均值概率的修复幅度见上表；AUROC 不变"
          "（单调变换），判别力上限仍由特征层决定。")


if __name__ == "__main__":
    main()
