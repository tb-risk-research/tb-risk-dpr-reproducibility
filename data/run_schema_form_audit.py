#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主线 22 维 schema 特征形态审计（§8.9 反哺，2026-09-06）

设计地位：§8.7/§8.9 的"暴露排序力形态分层"发现对主线模型有一个
未兑现的直接预测——22 维中暴露族特征哪些是**个体间变异**形态
（接触时长/亲近度，承载个体排序信号），哪些是**户级常量**形态
（指示病例特征/共享环境，个体排序力弱）。本脚本用三个带户键的
真实队列实证：逐特征计算**户内方差占比**（within-household
variance share，仅多成员户）+ **单变量排序力**（AUROC + 户级
cluster bootstrap 95% CI；二值特征另报 2×2 Wald OR）。

队列与终点：
- HomeACF（南非，TST 感染 tst_pos10，户键 record_id）——唯一同队列
  含双形态（个体暴露梯度 vs 指示病例特征）的感染终点队列；
- Muchuro（乌干达，IGRA 感染，户键 tbindex_id）——仅个体形态
  （prox/tw/reln/ncontacts），供个体形态证据；
- Peru MDR（秘鲁，病程 tb_outcome，户键 family_id）——主线 schema
  直接映射队列，**形态混装读数**：cumulative_exposure 槽位装入
  指示病例痰涂片等级（户级常量，meta 已披露），contact_distance_
  score 装入共享卧室（个体形态）——同一队列内的天然形态对照。

预声明预期（建模前写入，A1 为度量构造性验证）：
- A1：户级常量形态特征 within_share = 0（ANOVA 度量正确性检查）；
- A2：感染终点上（HomeACF），个体变异暴露形态的平均 |AUROC−0.5|
  > 户级常量形态特征（形态 × 排序贡献的主体检验）；
- A3：Peru MDR（病程终点）上，cumulative_exposure（户级常量装入）
  单变量 ≈ null（meta 已载 0.5123）——若 contact_distance_score
  （个体形态）排序力更高，形态预测在病程终点同向。

口径纪律：AUROC 原值报出（含方向，保护性特征 <0.5 如实呈现），
形态层汇总用 |AUROC−0.5|；CI = 户级 cluster bootstrap ×1000
（seed 42，对户级常量特征防止个体级重采样高估精度）；OR 仅二值
特征（2×2 Wald，零格 Haldane-Anscombe 0.5 校正，个体级口径披露）；
within_share 仅多成员户（单成员户无户内变异可言）。

地位：探索性方向分析（公开数据）；不改 schema、不改部署——
输出为特征工程方向（P2 优化目标定位）。

用法：
    python data/run_schema_form_audit.py
输出：
    data/processed/schema_form_audit_20260906.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from tb_risk.validation.real_data_infection import (  # noqa: E402
    load_homeacf_contacts)

OUT = os.path.join(HERE, "processed", "schema_form_audit_20260906.json")
MU_XLSX = os.path.join(HERE, "raw", "figshare",
                       "uganda_household_igra_s001.xlsx")
PERU_CSV = os.path.join(HERE, "processed", "ml_training_peru_mdr_real.csv")

N_BOOTSTRAP = 1000
SEED = 42

PROX_MAP = {
    "Sleeps in a different house": 1.0,
    "Sleeps in a different room, same house": 2.0,
    "Sleeps in different bed but same room": 3.0,
    "Sleeps in the same bed": 4.0,
}
TW_MAP = {
    "Not every day": 1.0,
    "Everyday,<50% of the day": 2.0,
    "Everyday, >=50% of the day": 3.0,
}


# ---------------------------------------------------------------- 加载
def load_muchuro():
    mu = pd.read_excel(MU_XLSX)
    d = pd.DataFrame(index=mu.index)
    d["prox"] = mu["prox2index"].map(PROX_MAP)
    d["tw"] = mu["tw_tbindex"].map(TW_MAP)
    d["reln_parent_sibling"] = (
        mu["reln_w_index"] == "Parent/Sibling").astype(float)
    d["reln_spouse"] = (mu["reln_w_index"] == "Spouse").astype(float)
    d["reln_others"] = (mu["reln_w_index"] == "Others").astype(float)
    d["ncontacts"] = pd.to_numeric(mu["ncontacts"], errors="coerce")
    d["age"] = pd.to_numeric(mu["ageofcontact"], errors="coerce")
    d["sex_male"] = (mu["gender"] == "Male").astype(float)
    d["smoke"] = mu["smk_status"].map(
        {"Never smoker": 0.0, "Ex-smoker": 1.0, "Current smoker": 2.0})
    d["hiv"] = mu["hivstatus"].map({"Negative": 0.0, "Positive": 1.0})
    d["bcg"] = (mu["bcgscar"] == "Yes").astype(float)
    d["cluster"] = mu["tbindex_id"]
    d["y"] = pd.to_numeric(mu["igra_pos"], errors="coerce")
    return d


def load_peru():
    df = pd.read_csv(PERU_CSV)
    d = pd.DataFrame(index=df.index)
    for c in ["age", "has_tb", "past_illness",
              "contact_distance_score", "cumulative_exposure",
              "exposure_setting_score", "has_symptoms",
              "ventilation_score", "single_duration", "freq_density",
              "time_span", "bcg_vaccine", "mdr_household",
              "index_smear_grade", "index_cough_days", "index_hiv",
              "tb_outcome"]:
        d[c] = pd.to_numeric(df[c], errors="coerce")
    # gender 存为 'm'/'f' 字符串（meta 声称 1/0 编码，CSV 实际为字符串）
    d["gender"] = (df["gender"].astype(str).str.lower() == "m").astype(
        float).where(df["gender"].notna())
    d["cluster"] = df["family_id"].astype(str)
    d["y"] = d.pop("tb_outcome")
    return d


def load_homeacf():
    df = load_homeacf_contacts()
    d = pd.DataFrame(index=df.index)
    # 暴露个体梯度（ts 三级 → 序）
    d["ts_ord"] = 2.0 - df["ts_low"] + df["ts_high"]
    for c in ["share_bedroom", "sleep_same_bed", "airspace_shared",
              "rel_child", "rel_spouse", "rel_sibling", "rel_parent",
              "idx_smear_pos", "idx_xpert_pos", "idx_hiv_pos",
              "idx_coughdays", "idx_dead", "hh_n_contacts",
              "contact_age", "contact_sex_m", "hiv_pos_h", "bmi_h",
              "smoke_ever_h", "diabetes_h_f", "tst_pos10"]:
        d[c] = pd.to_numeric(df[c], errors="coerce")
    d["cluster"] = df["record_id"].astype(str)
    d["y"] = d.pop("tst_pos10")
    return d


# ------------------------------------------------------------ 度量
def within_share(x, clusters):
    """户内方差占比（仅多成员户）：SS_within / SS_total。

    户级常量特征 → 0；纯个体特征 → 接近 1。
    """
    ok = x.notna()
    x, g = x[ok].values, clusters[ok].values
    df_ = pd.DataFrame({"x": x, "g": g})
    sizes = df_.groupby("g")["x"].transform("size")
    multi = df_[sizes > 1]
    if len(multi) < 2:
        return float("nan"), int(len(x))
    ss_total = float(((multi["x"] - multi["x"].mean()) ** 2).sum())
    if ss_total == 0:
        return float("nan"), int(len(multi))
    within = multi.groupby("g")["x"].transform(
        lambda v: (v - v.mean()) ** 2)
    ss_within = float(within.sum())
    return ss_within / ss_total, int(len(multi))


def auroc_ci(x, y, clusters, n_boot=N_BOOTSTRAP, seed=SEED):
    """单变量 AUROC + 户级 cluster bootstrap 95% CI。"""
    ok = x.notna() & y.notna()
    x, y, g = x[ok].values, y[ok].values, clusters[ok].values
    n = int(ok.sum())
    if n == 0 or len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return float("nan"), [float("nan"), float("nan")], n
    auc = float(roc_auc_score(y, x))
    rng = np.random.RandomState(seed)
    g_arr = np.asarray(g, dtype=object)
    uniq = np.unique(g_arr)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        parts = []
        for c in pick:
            parts.append(np.where(g_arr == c)[0])
        idx = np.concatenate(parts)
        yb, xb = y[idx], x[idx]
        if len(np.unique(yb)) < 2 or len(np.unique(xb)) < 2:
            continue
        boots.append(roc_auc_score(yb, xb))
    if len(boots) < n_boot * 0.5:
        ci = [float("nan"), float("nan")]
    else:
        ci = [float(np.percentile(boots, 2.5)),
              float(np.percentile(boots, 97.5))]
    return auc, ci, n


def odd_ratio(x, y):
    """二值特征 2×2 Wald OR（零格 0.5 校正；个体级口径披露）。"""
    ok = x.notna() & y.notna()
    x, y = x[ok].values, y[ok].values
    if len(np.unique(x)) != 2 or len(np.unique(y)) != 2:
        return None
    xv, yv = (x == x.max()).astype(int), (y == 1).astype(int)
    a = int(((xv == 1) & (yv == 1)).sum())
    b = int(((xv == 1) & (yv == 0)).sum())
    c = int(((xv == 0) & (yv == 1)).sum())
    dd = int(((xv == 0) & (yv == 0)).sum())
    if min(a, b, c, dd) == 0:
        a, b, c, dd = a + 0.5, b + 0.5, c + 0.5, dd + 0.5
        corr = True
    else:
        corr = False
    ln_or = np.log((a * dd) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / dd)
    return {
        "or": float(np.exp(ln_or)),
        "ci95": [float(np.exp(ln_or - 1.96 * se)),
                 float(np.exp(ln_or + 1.96 * se))],
        "table": {"x1y1": int(((xv == 1) & (yv == 1)).sum()),
                  "x1y0": int(((xv == 1) & (yv == 0)).sum()),
                  "x0y1": int(((xv == 0) & (yv == 1)).sum()),
                  "x0y0": int(((xv == 0) & (yv == 0)).sum())},
        "zero_cell_corrected": corr,
    }


# ------------------------------------------------------------ 注册表
FEATURES = {
    "homeacf": {
        "endpoint": "TST 感染（tst_pos10，≥10mm）",
        "features": {
            # 暴露·个体梯度形态
            "ts_ord": ("个体梯度", "与指示病例共处时长 3 级"),
            "share_bedroom": ("个体梯度", "共享卧室"),
            "sleep_same_bed": ("个体梯度", "同床睡"),
            "airspace_shared": ("个体梯度", "共享空气空间"),
            "rel_child": ("个体梯度", "关系=子女"),
            "rel_spouse": ("个体梯度", "关系=配偶"),
            "rel_sibling": ("个体梯度", "关系=兄弟姐妹"),
            "rel_parent": ("个体梯度", "关系=父母"),
            # 暴露·户级常量形态（指示病例/户特征）
            "idx_smear_pos": ("户级常量", "指示病例涂片阳性"),
            "idx_xpert_pos": ("户级常量", "指示病例 Xpert 阳性"),
            "idx_hiv_pos": ("户级常量", "指示病例 HIV 阳性"),
            "idx_coughdays": ("户级常量", "指示病例咳嗽天数"),
            "idx_dead": ("户级常量", "指示病例死亡"),
            "hh_n_contacts": ("户级常量", "户内接触者数"),
            # 宿主对照
            "contact_age": ("宿主对照", "年龄"),
            "contact_sex_m": ("宿主对照", "男性"),
            "hiv_pos_h": ("宿主对照", "接触者 HIV 阳性"),
            "bmi_h": ("宿主对照", "BMI"),
            "smoke_ever_h": ("宿主对照", "曾吸烟"),
            "diabetes_h_f": ("宿主对照", "糖尿病"),
        },
    },
    "muchuro": {
        "endpoint": "IGRA 感染（igra_pos）",
        "features": {
            "prox": ("个体梯度", "亲近度 4 级（同床→异屋）"),
            "tw": ("个体梯度", "共处时长 3 级"),
            "reln_parent_sibling": ("个体梯度", "关系=父母/兄弟姐妹"),
            "reln_spouse": ("个体梯度", "关系=配偶"),
            "reln_others": ("个体梯度", "关系=其他"),
            "ncontacts": ("个体梯度", "接触者人数"),
            "age": ("宿主对照", "年龄"),
            "sex_male": ("宿主对照", "男性"),
            "smoke": ("宿主对照", "吸烟 3 级"),
            "hiv": ("宿主对照", "HIV 阳性"),
            "bcg": ("宿主对照", "BCG 瘢痕"),
        },
    },
    "peru_mdr": {
        "endpoint": "病程（tb_outcome=Incident TB）",
        "features": {
            # 主线 schema 槽位映射（meta 披露的形态混装）
            "contact_distance_score": ("个体梯度", "主线槽=接触距离；装入共享卧室 0/1"),
            "cumulative_exposure": ("户级常量", "主线槽=累积暴露时长；装入指示病例痰涂片等级 0-3"),
            "exposure_setting_score": ("户级常量", "主线槽=暴露场景；装入 MDR 户 0/1"),
            "mdr_household": ("户级常量", "MDR 户"),
            "index_smear_grade": ("户级常量", "指示病例痰涂片等级（原始列）"),
            "index_cough_days": ("户级常量", "指示病例咳嗽天数"),
            "index_hiv": ("户级常量", "指示病例 HIV"),
            "age": ("宿主对照", "年龄（带中点）"),
            "gender": ("宿主对照", "男性"),
            "has_tb": ("宿主对照", "既往 TB 史"),
            "past_illness": ("宿主对照", "HIV|糖尿病"),
            # 置0/置常数槽位（收集缺口展示）
            "has_symptoms": ("置0未采集", "主线槽=症状；置0"),
            "ventilation_score": ("置0未采集", "主线槽=通风；置0"),
            "single_duration": ("置0未采集", "主线槽=单次时长；置0"),
            "freq_density": ("置0未采集", "主线槽=频次；置0"),
            "time_span": ("置0未采集", "主线槽=持续周期；置0"),
            "bcg_vaccine": ("置常数", "主线槽=BCG；置1（普种推断）"),
        },
    },
}

# 主线 22 维 schema 形态分类（静态分析，predictor.py 单一真值源）
SCHEMA_FORM = {
    "exposure_individual": [
        "cumulative_exposure", "contact_distance_score", "single_duration",
        "freq_density", "time_span"],
    "exposure_environment_constant": [
        "ventilation_score", "exposure_setting_score"],
    "host_individual": [
        "age", "has_symptoms", "bcg_vaccine", "has_tb", "is_high_risk",
        "past_illness"],
    "interaction_individual": [
        "age_immuno", "dm_tb_synergy", "age_bcg_decay", "symptom_delay",
        "cough_contact", "highrisk_comorbid", "immune_bcg",
        "exposure_accumulation", "age_diabetes"],
    "note": ("暴露族 7 维中 5 维为个体形态、2 维为环境常量倾向形态；"
             "宿主 6 维与交互 9 维均为个体形态（由个体特征派生）。"
             "cough_contact 含暴露分量但仍为接触者级。"),
}


# ------------------------------------------------------------ 主流程
def audit_dataset(name, d, registry):
    y = d["y"]
    cluster = d["cluster"]
    rows = []
    for feat, (form, desc) in registry["features"].items():
        if feat not in d.columns:
            rows.append({"feature": feat, "form": form, "desc": desc,
                         "error": "column missing"})
            continue
        x = d[feat]
        ws, n_multi = within_share(x, cluster)
        auc, ci, n = auroc_ci(x, y, cluster)
        orr = odd_ratio(x, y) if form != "置0未采集" else None
        row = {
            "feature": feat, "form": form, "desc": desc,
            "n": n, "n_multi_member": n_multi,
            "within_share": None if np.isnan(ws) else round(ws, 4),
            "auroc": None if np.isnan(auc) else round(auc, 4),
            "auroc_ci95": [None if np.isnan(v) else round(v, 4)
                           for v in ci],
        }
        if orr is not None:
            row["or"] = {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in orr.items()}
        rows.append(row)
    # 形态层汇总：平均 |AUROC−0.5|
    summary = {}
    for form in ["个体梯度", "户级常量", "宿主对照"]:
        mags = [abs(r["auroc"] - 0.5) for r in rows
                if r["form"] == form and r["auroc"] is not None]
        if mags:
            summary[form] = {
                "n_features": len(mags),
                "mean_abs_auc_minus_half": round(float(np.mean(mags)), 4),
                "max_abs_auc_minus_half": round(float(np.max(mags)), 4),
            }
    return {"endpoint": registry["endpoint"], "features": rows,
            "form_summary": summary}


def main():
    t0 = time.time()
    loaders = {
        "homeacf": (load_homeacf, FEATURES["homeacf"]),
        "muchuro": (load_muchuro, FEATURES["muchuro"]),
        "peru_mdr": (load_peru, FEATURES["peru_mdr"]),
    }
    results = {}
    for name, (fn, reg) in loaders.items():
        d = fn()
        d = d[d["y"].notna()]
        results[name] = audit_dataset(name, d, reg)
        results[name]["n"] = int(len(d))
        results[name]["n_events"] = int(d["y"].sum())
        results[name]["n_clusters"] = int(d["cluster"].nunique())
        print(f"[{name}] n={len(d)} ev={int(d['y'].sum())} "
              f"clusters={d['cluster'].nunique()}")

    # -------- 预期读数 --------
    ha = results["homeacf"]
    pm = results["peru_mdr"]

    # A1：户级常量形态特征 within_share == 0（容差 1e-9，浮点）
    a1_rows = [r for ds in results.values() for r in ds["features"]
               if r["form"] == "户级常量" and "within_share" in r
               and r["within_share"] is not None]
    a1_pass = all(r["within_share"] < 1e-6 for r in a1_rows) if a1_rows \
        else None

    # A2：HomeACF 感染终点，个体梯度形态平均 |AUROC−0.5| > 户级常量
    s = ha["form_summary"]
    a2_pass = (s.get("个体梯度", {}).get("mean_abs_auc_minus_half", -1)
               > s.get("户级常量", {}).get("mean_abs_auc_minus_half", -1)) \
        if "个体梯度" in s and "户级常量" in s else None

    # A3：Peru MDR——cumulative_exposure（户级常量装入）vs
    # contact_distance_score（个体形态）单变量排序对比
    pm_feats = {r["feature"]: r for r in pm["features"]}
    ce = pm_feats.get("cumulative_exposure", {}).get("auroc")
    cd = pm_feats.get("contact_distance_score", {}).get("auroc")
    a3 = None
    if ce is not None and cd is not None:
        a3 = {
            "cumulative_exposure_auroc": ce,
            "contact_distance_score_auroc": cd,
            "individual_form_ranks_higher":
                abs(cd - 0.5) > abs(ce - 0.5),
        }

    out = {
        "date": "2026-09-06",
        "name": "schema_form_audit_v1",
        "status": "探索性方向分析（预声明 A1-A3）：主线 22 维 schema "
                  "暴露特征形态 × 排序贡献实证；不改 schema 不改部署，"
                  "输出为特征工程方向定位",
        "design": {
            "motivation": "§8.9 暴露排序力形态分层（个体梯度 3-6× 于户级"
                          "常量增量）对主线 22 维暴露族 7 维的直接预测",
            "metric_definitions": {
                "within_share": "SS_within/SS_total（ANOVA by 户键，仅多成"
                                "员户）；户级常量→0，纯个体→→1",
                "auroc": "单变量 AUROC 原值（含方向；形态层汇总用 "
                         "|AUROC−0.5|）",
                "auroc_ci95": "户级 cluster bootstrap ×1000（seed 42）——"
                              "对户级常量特征防止个体级重采样高估精度",
                "or": "二值特征 2×2 Wald OR（零格 0.5 校正；个体级口径"
                      "非 cluster，CI 偏窄披露）",
            },
            "predictions_predeclared": {
                "A1": "户级常量形态特征 within_share = 0（构造性验证）",
                "A2": "HomeACF 感染终点：个体梯度形态平均 |AUROC−0.5| > "
                      "户级常量形态",
                "A3": "Peru MDR：cumulative_exposure（指示病例涂片等级装"
                      "入，户级常量）单变量 ≈ null（meta 已载 0.5123）；"
                      "contact_distance_score（共享卧室，个体形态）排序"
                      "力对比待读",
            },
            "inputs": {
                "homeacf": "raw/homeacf_tstsa.rda via "
                           "validation.real_data_infection."
                           "load_homeacf_contacts",
                "muchuro": "raw/figshare/"
                           "uganda_household_igra_s001.xlsx",
                "peru_mdr": "processed/ml_training_peru_mdr_real.csv "
                            "（meta：peru_mdr_dataset_meta.json 形态混装"
                            "披露）",
            },
        },
        "schema_form_22": SCHEMA_FORM,
        "results": results,
        "predictions_verdicts": {
            "A1_within_share_zero_for_constants": a1_pass,
            "A2_homeacf_individual_beats_constant": a2_pass,
            "A3_peru_form_contrast": a3,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\narchived -> {OUT}")

    # 控制台速览
    for ds in ["homeacf", "muchuro", "peru_mdr"]:
        r = results[ds]
        print(f"\n=== {ds}（{r['endpoint']}） n={r['n']} "
              f"ev={r['n_events']} clusters={r['n_clusters']}")
        for row in r["features"]:
            auc = row.get("auroc")
            ws = row.get("within_share")
            print(f"  [{row['form']}] {row['feature']:<26} "
                  f"ws={ws if ws is not None else '—'} "
                  f"auc={auc if auc is not None else '—'} "
                  f"ci={row.get('auroc_ci95')}")
        print(f"  form_summary: {r['form_summary']}")
    print(f"\nA1={a1_pass} A2={a2_pass}")
    print(f"A3={a3}")


if __name__ == "__main__":
    import time  # noqa: E402
    main()
