#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CDC TBESC-II Part A × 终点依赖 2×2 扩展（§8.7 扩展，2026-09-06）

数据：CDC TBESC-II Part A（22,020×209，2012-2017 入组前瞻随访；
定向审计 VPN 复测轮解封）。研究全称（SODA 元数据）："Comparison of
the Tuberculin Skin Test and Interferon-Gamma Release Assays in
Diagnosing Infection with M. tuberculosis and Prediction of
Progression to Tuberculosis Disease"——官方研究目的即感染诊断与
进展预测双终点，与 §8.6/§8.7 终点依赖图景直接对口。

**终点定义（官方数据字典破译后自建，字典归档
raw/cdc/partA_data_dictionary_2025_02_04.xlsx）**：
- 感染 = QFT 主检（AllCombinedLabs d_main_qft==1）qft_result==1；
  对照 = qft_result==2（阴性）；3/4（不定/无效）排除；人群排除全部
  TB 病例（d_case_status 1/2/3）。注意：**现患感染口径**（累积暴露
  结果，非 §8.7 的转阳新感染），ltbi_pos 官方定义为"因阳性检测被
  转诊"（入组原因）不可作终点——已破译排除；
- 病程 = d_case_status==1（官方 Incident 新发 TB，42 例）；现患
  （==2，87）与不确定（==3，20）从人群排除；对照 = 0。

**特征臂（个体级，CDC 无指示病例关联键 → 暴露臂重构为"个体级暴露
压力"，正好补 §8.7 反转腿缺的个体级暴露变异）**：
- EXP：close_contact（接触史）、close_contact_hh（户内）+缺失指示、
  close_contact_time（强度）+缺失指示、tb_inc（出生国 WHO 结核
  发病率/10万，2012-2017 中位，MDG_0000000020）、d_years_us
  （在美年数组距=暴露衰减）、travel；
- HOST：d_age_cat、d_sex、foreign_born、hiv、diabetes、smoked_100、
  alc_rate、drug_use、homeless、corr_fac、bcg；
- HOST_FULL = HOST + BMI（§8.7 营养块对应物）；
- FULL = EXP + HOST_FULL；
- FULL_BASEINF = FULL + 基线 QFT 主检结果（病程终点臂，§8.7 的
  full_baseinf 对应物）。

**可证伪预期（预声明，建模前）**：
P1 感染终点暴露压力臂显著贡献（full−host_full > 0）——个体级暴露
   变异存在（国家负担 3→548/10万 跨 180 倍、在美年数、接触强度），
   与 §8.7 感染反转腿（户级常量弱代理）形成设计对照；
P2 病程终点宿主臂 ≥ 暴露压力臂（host_full−exp ≥ 0）；
P3 基线感染状态对病程增量 > 0（§8.7 full_baseinf−full = +0.081
   CI[+0.062,+0.099] 的跨数据集复现尝试）；
P4 BMI 不对称（病程 >0 / 感染 ≈0，§8.7 营养块 +0.036/−0.006 与
   Aibana 2016 Cox HR 0.48）在低负担筛查人群的探索格。

**协议（与 §8.7 同构）**：StratifiedKFold(5) × 20 种子（个体分层，
无户键/簇键 → 组感知乐观偏差披露同 §8.6 −0.037 量级）；逐行平均
池化 OOF；个体 bootstrap ×2000 + DeLong（次）+ 逐种子 Δ 四栏
（R5a 纪律）；RF 主口径 + LR 敏感性（冻结注册表 _make_model）；
折内中位数填补 + 结构性缺失指示列（close_contact_hh/time 83%
跳答缺失）。

**敏感性层**：
- progression_symptom：FULL_SYM = FULL + 7 基线症状 + abnormal_cxr
  （症状=早期疾病状态非宿主易感性，仅病程终点，作 CDC 特有参照）；
- infection_tst：感染终点换 TST 主检 ≥10mm（平台敏感性，仅 exp/
  host_base 两臂看不对称方向）。

**地位**：探索性方向分析（公开数据、原文已发表多篇 TBESC 结果，
不重复其主张；判别层终点依赖结构检验）。归档
data/processed/tbesc2_partA_endpoint_dependence_20260906.json。

用法：
    python data/run_tbesc2_partA_endpoint_dependence.py
"""
import json
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from tb_risk.validation.real_data_pi import _make_model, _pr_auc  # noqa: E402
from tb_risk.validation.layer_ablation import delong_paired_test  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

ZIP_PATH = os.path.join(HERE, "raw", "cdc", "tbesc2_partA_tst_igra.zip")
BURDEN_CSV = os.path.join(HERE, "raw", "who", "gho_tb_incidence_per_100k.csv")
OUT = os.path.join(HERE, "processed", "tbesc2_partA_endpoint_dependence_20260906.json")

N_SEEDS = 20
N_SPLITS = 5
N_BOOTSTRAP = 2000
PRIMARY = "random_forest"


def _fast_auc(y, scores):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def load_zip(name):
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open(name) as f:
            return pd.read_csv(f, sep="|", low_memory=False, encoding="latin-1")


def build_frame():
    m = load_zip("PartA03062025/PartAMain.csv")
    labs = load_zip("PartA03062025/AllCombinedLabs.csv")
    burden = pd.read_csv(BURDEN_CSV, index_col=0)["tb_inc_per_100k"]

    d = pd.DataFrame(index=m.index)
    # ---- HOST ----
    d["age_cat"] = m["d_age_cat"].astype(float)
    d["sex_male"] = m["d_sex"].map({1: 0.0, 2: 1.0})
    d["foreign_born"] = (m["birth_country"].astype(str).str.upper() != "USA").astype(float)
    for col, name in [("hiv", "hiv"), ("diabetes", "diabetes"),
                      ("smoked_100", "smoked_100"), ("alc_rate", "alc_rate"),
                      ("drug_use", "drug_use"), ("homeless", "homeless"),
                      ("corr_fac", "corr_fac"), ("bcg", "bcg"),
                      ("travel", "travel")]:
        v = pd.to_numeric(m[col], errors="coerce")
        d[name] = v.replace(99, np.nan)
    # BMI（哨兵值清洗）
    w = pd.to_numeric(m["weight_lb"], errors="coerce")
    h = pd.to_numeric(m["height_in"], errors="coerce")
    w = w.where((w >= 40) & (w <= 800))
    h = h.where((h >= 20) & (h <= 90))
    d["bmi"] = (w * 0.4536) / ((h * 0.0254) ** 2)
    d["bmi"] = d["bmi"].where((d["bmi"] >= 10) & (d["bmi"] <= 80))
    # ---- EXP ----
    d["close_contact"] = pd.to_numeric(m["close_contact"], errors="coerce").replace(99, np.nan)
    hh = pd.to_numeric(m["close_contact_hh"], errors="coerce").replace(99, np.nan)
    ct = pd.to_numeric(m["close_contact_time"], errors="coerce").replace(99, np.nan)
    d["close_contact_hh"] = hh
    d["close_contact_hh_missing"] = hh.isna().astype(float)
    d["close_contact_time"] = ct
    d["close_contact_time_missing"] = ct.isna().astype(float)
    d["years_us"] = pd.to_numeric(m["d_years_us"], errors="coerce").replace(99, np.nan)
    d["tb_inc"] = m["birth_country"].astype(str).str.upper().map(burden)
    # ---- 症状（敏感性层） ----
    for col in ["cough", "cough_blood", "fever", "night_sweats",
                "weight_loss", "chest_pain", "tired"]:
        d[col] = pd.to_numeric(m[col], errors="coerce").replace(99, np.nan)
    d["abnormal_cxr"] = pd.to_numeric(m["abnormal_cxr"], errors="coerce").replace(99, np.nan)

    # ---- 终点 ----
    d["d_case_status"] = pd.to_numeric(m["d_case_status"], errors="coerce")
    qft_main = labs[(labs["d_main_qft"] == 1) & labs["qft_result"].notna()]
    qft_res = qft_main.groupby("studyid")["qft_result"].first()
    d["qft_main"] = m["studyid"].map(qft_res)  # 1=pos 2=neg 3/4=ind/inv
    tst_main = labs[(labs["d_main_tst"] == 1) & labs["tst_result"].notna()]
    tst_mm = tst_main.groupby("studyid")["tst_result"].first()
    d["tst_main_mm"] = m["studyid"].map(tst_mm)
    return d


ARMS = {
    "exp": ["close_contact", "close_contact_hh", "close_contact_hh_missing",
            "close_contact_time", "close_contact_time_missing", "tb_inc",
            "years_us", "travel"],
    "host_base": ["age_cat", "sex_male", "foreign_born", "hiv", "diabetes",
                  "smoked_100", "alc_rate", "drug_use", "homeless",
                  "corr_fac", "bcg"],
    "host_full": ["age_cat", "sex_male", "foreign_born", "hiv", "diabetes",
                  "smoked_100", "alc_rate", "drug_use", "homeless",
                  "corr_fac", "bcg", "bmi"],
    "full": (["close_contact", "close_contact_hh", "close_contact_hh_missing",
              "close_contact_time", "close_contact_time_missing", "tb_inc",
              "years_us", "travel", "age_cat", "sex_male", "foreign_born",
              "hiv", "diabetes", "smoked_100", "alc_rate", "drug_use",
              "homeless", "corr_fac", "bcg", "bmi"]),
    "full_baseinf": (["close_contact", "close_contact_hh",
                      "close_contact_hh_missing", "close_contact_time",
                      "close_contact_time_missing", "tb_inc", "years_us",
                      "travel", "age_cat", "sex_male", "foreign_born",
                      "hiv", "diabetes", "smoked_100", "alc_rate",
                      "drug_use", "homeless", "corr_fac", "bcg", "bmi",
                      "qft_pos_baseline", "qft_baseline_missing"]),
    "full_sym": (["close_contact", "close_contact_hh",
                  "close_contact_hh_missing", "close_contact_time",
                  "close_contact_time_missing", "tb_inc", "years_us",
                  "travel", "age_cat", "sex_male", "foreign_born",
                  "hiv", "diabetes", "smoked_100", "alc_rate",
                  "drug_use", "homeless", "corr_fac", "bcg", "bmi",
                  "qft_pos_baseline", "qft_baseline_missing",
                  "cough", "cough_blood", "fever", "night_sweats",
                  "weight_loss", "chest_pain", "tired", "abnormal_cxr"]),
}

CONTRASTS_INFECTION = [
    ("exp", "host_base"),
    ("host_full", "host_base"),
    ("full", "exp"),
    ("full", "host_full"),
]
CONTRASTS_PROGRESSION = [
    ("host_base", "exp"),
    ("host_full", "host_base"),
    ("full", "host_full"),
    ("full_baseinf", "full"),
    ("full_sym", "full_baseinf"),
]


def run_arms(dfeat, y, arm_keys, model_keys=("random_forest", "logistic")):
    X_all = {arm: dfeat[ARMS[arm]].to_numpy(dtype=float) for arm in arm_keys}
    oof = {arm: {} for arm in arm_keys}
    per_seed = {arm: {} for arm in arm_keys}
    for arm in arm_keys:
        X = X_all[arm]
        for mk in model_keys:
            oof_acc = np.zeros(len(y))
            aurocs = []
            for s in range(N_SEEDS):
                skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                      random_state=s)
                oof_s = np.zeros(len(y))
                for tr, te in skf.split(X, y):
                    med = np.nanmedian(X[tr], axis=0)
                    med = np.where(np.isnan(med), 0.0, med)
                    Xtr = np.where(np.isnan(X[tr]), med, X[tr])
                    Xte = np.where(np.isnan(X[te]), med, X[te])
                    model = _make_model(mk, s)
                    model.fit(Xtr, y[tr])
                    oof_s[te] = model.predict_proba(Xte)[:, 1]
                oof_acc += oof_s
                aurocs.append(_fast_auc(y, oof_s))
            oof[arm][mk] = oof_acc / N_SEEDS
            per_seed[arm][mk] = aurocs
    return {"y": y, "oof": oof, "per_seed": per_seed}


def _individual_bootstrap(oof_p, y, contrasts, n_bootstrap=N_BOOTSTRAP, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y)
    acc = {k: [] for k in oof_p}
    dacc = {f"{a}_minus_{b}": [] for a, b in contrasts}
    for _ in range(n_bootstrap):
        rows = rng.randint(0, n, size=n)
        yr = y[rows]
        if yr.sum() == 0 or yr.sum() == len(yr):
            continue
        auc = {k: _fast_auc(yr, v[rows]) for k, v in oof_p.items()}
        for k, a in auc.items():
            acc[k].append(a)
        for a, b in contrasts:
            dacc[f"{a}_minus_{b}"].append(auc[a] - auc[b])
    levels = {}
    for k, v in acc.items():
        levels[k] = {"mean": round(float(np.mean(v)), 4),
                     "ci": [round(float(np.percentile(v, 2.5)), 4),
                            round(float(np.percentile(v, 97.5)), 4)]}
    deltas = {}
    for k, v in dacc.items():
        v = np.array(v)
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        deltas[k] = {"mean": round(float(np.mean(v)), 4),
                     "bootstrap_ci": [round(lo, 4), round(hi, 4)],
                     "p_positive": round(float(np.mean(v > 0.0)), 3),
                     "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0)}
    return levels, deltas


def summarize_run(run, contrasts, primary=PRIMARY):
    y = run["y"]
    arm_summary = {}
    for arm in run["oof"]:
        for mk, oof in run["oof"][arm].items():
            aurocs = run["per_seed"][arm][mk]
            arm_summary[f"{arm}:{mk}"] = {
                "pooled_auroc": round(_fast_auc(y, oof), 4),
                "seed_mean_auroc": round(float(np.mean(aurocs)), 4),
                "seed_sd_auroc": round(float(np.std(aurocs)), 4),
                "seed_min_auroc": round(float(np.min(aurocs)), 4),
                "seed_max_auroc": round(float(np.max(aurocs)), 4),
                "pr_auc": round(_pr_auc(y, oof), 4),
            }
    oof_p = {arm: run["oof"][arm][primary] for arm in run["oof"]}
    levels, deltas = _individual_bootstrap(oof_p, y, contrasts)
    for arm in levels:
        arm_summary[f"{arm}:{primary}"]["bootstrap_ci"] = levels[arm]["ci"]
    ladder = {}
    for a, b in contrasts:
        key = f"{a}_minus_{b}"
        sa = np.array(run["per_seed"][a][primary])
        sb = np.array(run["per_seed"][b][primary])
        sd = sa - sb
        try:
            dl = delong_paired_test(y, oof_p[a], oof_p[b])
            delong = {"delta_auroc": round(dl["delta_auroc"], 4),
                      "p_value": round(dl["p_value"], 4)}
        except Exception as e:  # noqa: BLE001
            delong = {"error": str(e)[:120]}
        ladder[key] = {**deltas[key],
                       "per_seed_delta": {
                           "mean": round(float(np.mean(sd)), 4),
                           "sd": round(float(np.std(sd)), 4),
                           "min": round(float(np.min(sd)), 4),
                           "max": round(float(np.max(sd)), 4),
                           "share_positive": round(float(np.mean(sd > 0.0)), 3)},
                       "delong": delong}
    return arm_summary, ladder


def main():
    t0 = time.time()
    d = build_frame()

    # ---- 基线感染状态特征（full_baseinf 臂） ----
    d["qft_pos_baseline"] = (d["qft_main"] == 1).astype(float).where(
        d["qft_main"].notna())
    d["qft_baseline_missing"] = d["qft_main"].isna().astype(float)

    # ---- 感染终点人群 ----
    inf_pop = d["d_case_status"].isin([0]) & d["qft_main"].isin([1, 2])
    inf_y = (d.loc[inf_pop, "qft_main"] == 1).to_numpy(dtype=int)
    # ---- 病程终点人群（排现患/不确定，case=Incident） ----
    prog_pop = d["d_case_status"].isin([0, 1])
    prog_y = (d.loc[prog_pop, "d_case_status"] == 1).to_numpy(dtype=int)
    # ---- TST 敏感性终点 ----
    tst_pop = d["d_case_status"].isin([0]) & d["tst_main_mm"].notna()
    tst_y = (d.loc[tst_pop, "tst_main_mm"] >= 10).to_numpy(dtype=int)

    print("infection: n=%d ev=%d (%.1f%%)" % (
        inf_pop.sum(), inf_y.sum(), 100 * inf_y.mean()))
    print("progression: n=%d ev=%d (%.2f%%)" % (
        prog_pop.sum(), prog_y.sum(), 100 * prog_y.mean()))
    print("infection_tst: n=%d ev=%d (%.1f%%)" % (
        tst_pop.sum(), tst_y.sum(), 100 * tst_y.mean()))

    inf_arms = ["exp", "host_base", "host_full", "full"]
    prog_arms = ["exp", "host_base", "host_full", "full", "full_baseinf",
                 "full_sym"]
    tst_arms = ["exp", "host_base"]

    results = {}
    for name, pop, y, arms, contrasts in [
        ("infection", inf_pop, inf_y, inf_arms, CONTRASTS_INFECTION),
        ("progression", prog_pop, prog_y, prog_arms, CONTRASTS_PROGRESSION),
        ("infection_tst", tst_pop, tst_y, tst_arms,
         [("exp", "host_base")]),
    ]:
        print("[%s] running..." % name)
        dsub = d.loc[pop].reset_index(drop=True)
        run = run_arms(dsub, y, arms)
        arm_summary, ladder = summarize_run(run, contrasts)
        results[name] = {"n": int(pop.sum()), "n_events": int(y.sum()),
                         "arm_summary": arm_summary, "ladder": ladder}
        print("[%s] done (%.1f min)" % (name, (time.time() - t0) / 60.0))

    inf_l = results["infection"]["ladder"]
    prog_l = results["progression"]["ladder"]
    inf_a = results["infection"]["arm_summary"]
    prog_a = results["progression"]["arm_summary"]

    asymmetry = {
        "core_cells": {
            "infection:exp": inf_a[f"exp:{PRIMARY}"]["pooled_auroc"],
            "infection:host_base": inf_a[f"host_base:{PRIMARY}"]["pooled_auroc"],
            "infection:host_full": inf_a[f"host_full:{PRIMARY}"]["pooled_auroc"],
            "progression:exp": prog_a[f"exp:{PRIMARY}"]["pooled_auroc"],
            "progression:host_base": prog_a[f"host_base:{PRIMARY}"]["pooled_auroc"],
            "progression:host_full": prog_a[f"host_full:{PRIMARY}"]["pooled_auroc"],
            "progression:full_baseinf": prog_a[f"full_baseinf:{PRIMARY}"]["pooled_auroc"],
            "progression:full_sym": prog_a[f"full_sym:{PRIMARY}"]["pooled_auroc"],
        },
        "P1_exposure_increment_infection_full_minus_host_full":
            inf_l["full_minus_host_full"],
        "P2_host_minus_exp_progression": prog_l["host_base_minus_exp"],
        "P3_baseline_infection_increment_progression":
            prog_l["full_baseinf_minus_full"],
        "P4a_bmi_increment_infection": inf_l["host_full_minus_host_base"],
        "P4b_bmi_increment_progression": prog_l["host_full_minus_host_base"],
        "symptom_increment_progression_fullsym_minus_baseinf":
            prog_l["full_sym_minus_full_baseinf"],
        "readme": ("预声明 P1-P4 见脚本 docstring；全部方向级读数，"
                   "CI 横跨零只可写未证伪（TRIPOD 23a 纪律）"),
    }

    max_auroc = max(
        v["pooled_auroc"]
        for pop in ("infection", "progression")
        for k, v in results[pop]["arm_summary"].items()
        if k.endswith(f":{PRIMARY}"))

    out = {
        "date": "2026-09-06",
        "name": "tbesc2_partA_endpoint_dependence_v1",
        "status": ("探索性方向分析（预声明）：CDC TBESC-II Part A 判别层"
                   "终点依赖 2×2 扩展；个体级暴露压力臂（补 §8.7 户级常量"
                   "弱代理的设计对照）；只报方向 + CI"),
        "design": {
            "source": ("CDC data.cdc.gov 5hpj-p74g（TBESC-II Part A，"
                       "22,020×209 + 18 附表；定向审计 VPN 复测轮解封）；"
                       "官方数据字典 partA_data_dictionary_2025_02_04.xlsx "
                       "（SODA 附件通道）+ User Guide PDF 已归档 raw/cdc/"),
            "study_official_name": ("Comparison of the Tuberculin Skin Test "
                                    "and IGRAs in Diagnosing Infection with "
                                    "M. tuberculosis and Prediction of "
                                    "Progression to TB Disease"),
            "endpoint_definitions": {
                "infection": ("QFT 主检（d_main_qft==1）qft_result==1 阳 / "
                              "==2 阴；3/4 排除；人群排除全部 TB 病例。"
                              "现患感染口径（累积暴露结果）——与 §8.7 的"
                              "转阳（新感染）口径不同，披露"),
                "progression": ("d_case_status==1 官方 Incident（新发 TB）"
                                "42 例；现患==2（87）与不确定==3（20）"
                                "从人群排除；对照==0"),
                "infection_tst": "TST 主检 ≥10mm（平台敏感性层）",
                "ltbi_pos_warning": ("官方定义'因阳性检测被转诊'（入组原因）"
                                     "非感染判定——已破译排除，不可引用为终点"),
            },
            "cv": (f"StratifiedKFold({N_SPLITS}) × {N_SEEDS} 种子（个体"
                   "分层非组感知，乐观偏差披露同 §8.6 −0.037 量级）"),
            "pooling": "逐行平均 20 种子 OOF",
            "inference": (f"个体 bootstrap ×{N_BOOTSTRAP}（无簇键→非 "
                          "cluster，CI 偏乐观披露）+ DeLong（次）+ 逐种子 Δ"
                          " 四栏（R5a 纪律）"),
            "models": "RF（主）+ LR（敏感性），冻结注册表超参（_make_model）",
            "missing_data": ("折内中位数填补 + 结构性缺失指示列"
                             "（close_contact_hh/time 83% 跳答缺失）；"
                             "BMI 哨兵值清洗（weight 40-800lb / height "
                             "20-90in / BMI 10-80）"),
            "feature_arms": ARMS,
            "exposure_arm_construct": ("个体级暴露压力：出生国 WHO 结核发病率"
                                       "（MDG_0000000020，2012-2017 中位，197 "
                                       "国）+ 在美年数（暴露衰减）+ 密切接触"
                                       "史/户内/强度——CDC 无指示病例关联键，"
                                       "户内共享暴露时序腿不可测（裁决权仍在 "
                                       "ERASE-TB）"),
            "gate_checks": {
                "g1_analog": "不可计算（无指示病例关联键）",
                "g3_analog_max_auroc": round(max_auroc, 4),
                "g3_threshold": 0.99,
                "g3_verdict": "通过（远未饱和）" if max_auroc < 0.99 else "检查",
            },
            "preregistered_predictions": [
                "P1 感染终点 full−host_full > 0（个体级暴露变异贡献）",
                "P2 病程终点 host_base−exp ≥ 0（宿主主导病程）",
                "P3 病程终点 full_baseinf−full > 0（§8.7 +0.081 复现尝试）",
                "P4 BMI 增量：病程 >0 / 感染 ≈0（§8.7 不对称探索格）",
            ],
        },
        "infection": results["infection"],
        "progression": results["progression"],
        "sensitivity": {"infection_tst": results["infection_tst"]},
        "endpoint_asymmetry": asymmetry,
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print()
    print("===== 主读数（RF 池化 AUROC [个体 bootstrap 95%% CI]）=====")
    for pop in ("infection", "progression"):
        print("[%s] n=%d ev=%d" % (pop, out[pop]["n"], out[pop]["n_events"]))
        for arm in inf_arms if pop == "infection" else prog_arms:
            s = out[pop]["arm_summary"][f"{arm}:{PRIMARY}"]
            print("  %-14s %.4f CI%s" % (
                arm, s["pooled_auroc"], s.get("bootstrap_ci", [])))
        for k, v in out[pop]["ladder"].items():
            star = "*" if v.get("ci_excludes_zero") else " "
            print("  %s %-30s Δ=%+.4f CI=[%+.4f,%+.4f] P(>0)=%.3f "
                  "seed_share=%.2f DeLong p=%.4f" % (
                      star, k, v["mean"], v["bootstrap_ci"][0],
                      v["bootstrap_ci"][1], v["p_positive"],
                      v["per_seed_delta"]["share_positive"],
                      v["delong"].get("p_value", -1)))
    print()
    print("===== 预声明 P1-P4 读数 =====")
    for key in ("P1_exposure_increment_infection_full_minus_host_full",
                "P2_host_minus_exp_progression",
                "P3_baseline_infection_increment_progression",
                "P4a_bmi_increment_infection",
                "P4b_bmi_increment_progression"):
        v = asymmetry[key]
        print("  %-55s Δ=%+.4f CI=%s" % (key[:55], v["mean"],
                                          v["bootstrap_ci"]))
    print("\n归档 → %s" % OUT)
    print("总耗时 %.1f 分钟" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
