#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table 1：公开数据域多队列基线特征表（TRIPOD 条目 20b）

产出：
1. processed/table1_public_cohorts.csv —— 长表（机器可读，逐特征逐队列）
2. docs/table1_public_cohorts.md —— 稿件格式（含特征真实性脚注）

科学诚信设计（审稿关键）：
- 每个特征在每个队列标注真实性来源：R=真实变异 / M=组中点映射 /
  Z=置0(未采集) / C=置常数(推断) / S=半合成
- 真实性标注自动从各队列 meta 的 disclosure 字段提取（crp/taiwan/brazil
  等 meta 已含"真实特征/置0特征/置常数特征"清单），fallback 手动表
- 年龄 CRP 为组中点、semi_synthetic 为合成——脚注显式声明
- 终点行注明终点类型（LTBI 感染 / 活动性 TB / 细菌学确诊），防止
  跨终点直接比较的误读

用法：
    python data/build_table1.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "processed")
DOCS = os.path.join(os.path.dirname(HERE), "docs")

# 队列注册：名称 / csv / meta / 终点域（infection=LTBI；disease=活动性/确诊）
COHORTS = [
    ("Semi-synthetic (anchored)", "ml_training_semi_synthetic_anchored.csv",
     "ml_training_anchors.json", "disease", "合成"),
    ("NHANES 2011-12 (USA)", "ml_training_nhanes_real.csv",
     "nhanes_dataset_meta.json", "infection", "高收入低负担"),
    ("TREATS (ZMB/ZAF)", "ml_training_treats_real.csv",
     "treats_dataset_meta.json", "infection", "非洲高负担社区"),
    ("CRP screening (ZMB/ZAF)", "ml_training_crp_real.csv",
     "crp_dataset_meta.json", "disease", "非洲 ACF 筛查"),
    ("Kenya survey 2016", "kenya_ml_training.csv",
     "kenya_dataset_meta.json", "disease", "全国患病率调查"),
    ("Taiwan NTUH", "ml_training_taiwan_real.csv",
     "taiwan_dataset_meta.json", "disease", "转诊评估（亚洲）"),
    ("Brazil IGRA", "ml_training_brazil_real.csv",
     "brazil_dataset_meta.json", "infection", "门诊横断面"),
    ("Peru MDR household (PER)", "ml_training_peru_mdr_real.csv",
     "peru_mdr_dataset_meta.json", "disease", "家庭接触前瞻（美洲）"),
]

# 自动提取后的真实性覆盖（peru meta 的真实特征写法为"age(带中点)"，
# 自动提取的"组中点"模式不匹配，会误标 R；带中点=年龄带→组中点映射）
TRUTH_OVERRIDE = {
    ("Peru MDR household (PER)", "age"): "M",
}

# 手动真实性 fallback（meta 读取失败时用；键 = (队列简称, 特征)）
MANUAL_TRUTH = {
    ("semi", "age"): "S", ("semi", "gender"): "S",
    ("semi", "has_symptoms"): "S", ("semi", "past_illness"): "S",
    ("semi", "is_high_risk"): "S", ("semi", "bcg_vaccine"): "S",
    ("semi", "contact_exposure"): "S",
    ("nhanes", "age"): "R", ("nhanes", "gender"): "R",
    ("nhanes", "has_symptoms"): "R", ("nhanes", "past_illness"): "R",
    ("nhanes", "is_high_risk"): "R", ("nhanes", "bcg_vaccine"): "R",
    ("nhanes", "contact_exposure"): "Z",
    ("treats", "age"): "R", ("treats", "gender"): "R",
    ("treats", "has_symptoms"): "R", ("treats", "past_illness"): "R",
    ("treats", "is_high_risk"): "R", ("treats", "bcg_vaccine"): "R",
    ("treats", "contact_exposure"): "R",
    ("crp", "age"): "M", ("crp", "gender"): "R",
    ("crp", "has_symptoms"): "R", ("crp", "past_illness"): "R",
    ("crp", "is_high_risk"): "R", ("crp", "bcg_vaccine"): "C",
    ("crp", "contact_exposure"): "Z",
    ("kenya", "age"): "R", ("kenya", "gender"): "Z",
    ("kenya", "has_symptoms"): "R", ("kenya", "past_illness"): "R",
    ("kenya", "is_high_risk"): "R", ("kenya", "bcg_vaccine"): "R",
    ("kenya", "contact_exposure"): "R",
    ("taiwan", "age"): "R", ("taiwan", "gender"): "R",
    ("taiwan", "has_symptoms"): "Z", ("taiwan", "past_illness"): "R",
    ("taiwan", "is_high_risk"): "R", ("taiwan", "bcg_vaccine"): "C",
    ("taiwan", "contact_exposure"): "Z",
    ("brazil", "age"): "R", ("brazil", "gender"): "R",
    ("brazil", "has_symptoms"): "Z", ("brazil", "past_illness"): "Z",
    ("brazil", "is_high_risk"): "R", ("brazil", "bcg_vaccine"): "C",
    ("brazil", "contact_exposure"): "Z",
    ("peru_mdr", "age"): "M", ("peru_mdr", "gender"): "R",
    ("peru_mdr", "has_symptoms"): "Z", ("peru_mdr", "past_illness"): "R",
    ("peru_mdr", "is_high_risk"): "R", ("peru_mdr", "bcg_vaccine"): "C",
    ("peru_mdr", "contact_exposure"): "R",
}

SHORT = {c[0]: k for k, c in zip(
    ["semi", "nhanes", "treats", "crp", "kenya", "taiwan", "brazil",
     "peru_mdr"],
    COHORTS)}

FEATURES = [
    ("age", "年龄, 岁, mean ± SD"),
    ("gender", "女性, n (%)"),
    ("has_symptoms", "任一 TB 症状, n (%)"),
    ("past_illness", "既往史/合并症, n (%)"),
    ("is_high_risk", "高危 (≥65 或合并症), n (%)"),
    ("bcg_vaccine", "BCG 接种, n (%)"),
    ("contact_exposure", "接触暴露评分 (累积), mean ± SD"),
]

TRUTH_LABEL = {"R": "真实变异", "M": "组中点映射", "Z": "置0 (未采集)",
               "C": "置常数 (推断)", "S": "半合成"}


def load_truth_flags(name, meta):
    """从 meta disclosure 自动提取特征真实性（失败 fallback 手动表）"""
    flags = {}
    short = SHORT[name]
    disc = (meta or {}).get("disclosure", {}) if isinstance(meta, dict) else {}
    if not isinstance(disc, dict):
        disc = {}  # 部分 meta 的 disclosure 是纯文本描述（如 anchors）
    real = disc.get("真实特征", [])
    zero = disc.get("置0特征(未采集)", []) + disc.get("置0特征", [])
    const = disc.get("置常数特征", [])
    for feat, _ in FEATURES:
        blob = " ".join(str(x) for x in real)
        if any(feat in str(x) or (feat == "gender" and ("sex" in str(x).lower() or "性别" in str(x)))
               for x in real):
            flags[feat] = "M" if (feat == "age" and "组中点" in blob) else "R"
        elif any(feat in str(x) or (feat == "gender" and "sex" in str(x).lower())
                 for x in zero):
            flags[feat] = "Z"
        elif any(feat in str(x) or (feat == "gender" and "sex" in str(x).lower())
                 for x in const):
            flags[feat] = "C"
        else:
            flags[feat] = MANUAL_TRUTH.get((short, feat), "R")
    return flags


def fmt_cell(d, feat, n):
    if feat == "age":
        if d["age"].notna().sum() == 0:
            return "—"
        return f"{d['age'].mean():.1f} ± {d['age'].std():.1f}"
    if feat == "gender":
        if "gender" not in d.columns or d["gender"].isna().all():
            return "未采集"
        fem = (d["gender"].astype(str).str.lower() == "f").sum()
        return f"{fem} ({fem / n * 100:.1f})"
    if feat == "contact_exposure":
        col = "cumulative_exposure"
        if col not in d.columns:
            return "—"
        s = d[col]
        if s.std() == 0:
            return f"{s.mean():.2f} (常数)"
        return f"{s.mean():.2f} ± {s.std():.2f}"
    col = feat
    if col not in d.columns:
        return "—"
    pos = (pd.to_numeric(d[col], errors="coerce") == 1).sum()
    return f"{pos} ({pos / n * 100:.1f})"


def main():
    rows = []
    cohort_meta = {}
    for name, csv_fn, meta_fn, endpoint, setting in COHORTS:
        csv_path = os.path.join(PROC, csv_fn)
        if not os.path.exists(csv_path):
            print(f"[跳过] {csv_fn} 不存在")
            continue
        d = pd.read_csv(csv_path)
        n = len(d)
        meta = None
        meta_path = os.path.join(PROC, meta_fn)
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        flags = load_truth_flags(name, meta)
        flags.update({f: t for (nm, f), t in TRUTH_OVERRIDE.items()
                      if nm == name})
        n_pos = int(d["tb_outcome"].sum())
        pos_rate = n_pos / n * 100 if n else 0
        cohort_meta[name] = {
            "n": n, "endpoint": endpoint, "setting": setting,
            "n_events": n_pos, "event_rate": round(pos_rate, 1),
            "outcome_def": (meta or {}).get("outcome", ""),
            "flags": flags,
        }
        for feat, label in FEATURES:
            rows.append({
                "cohort": name, "feature": label, "feature_key": feat,
                "value": fmt_cell(d, feat, n),
                "truth": TRUTH_LABEL[flags[feat]],
                "n": n,
            })
        rows.append({
            "cohort": name, "feature": "终点事件, n (%)",
            "feature_key": "outcome",
            "value": f"{n_pos} ({pos_rate:.1f})",
            "truth": "终点", "n": n,
        })

    out_csv = pd.DataFrame(rows)
    csv_path = os.path.join(PROC, "table1_public_cohorts.csv")
    out_csv.to_csv(csv_path, index=False)

    # ---- markdown 稿件表 ----
    names = [c[0] for c in COHORTS if c[0] in cohort_meta]
    md = ["# Table 1. 公开数据域多队列基线特征\n"]
    md.append("特征真实性标注：†=组中点映射，‡=置0（该队列未采集），"
              "§=置常数（人群推断），¶=半合成；未标注=真实变异。\n")
    md.append("终点域：**I**=潜伏感染（LTBI/IGRA）；**D**=活动性/确诊 TB。"
              "感染与患病终点的事件率不可直接比较。\n")
    header = "| 特征 | " + " | ".join(
        f"{n}<br>(N={cohort_meta[n]['n']}, "
        f"{'**I**' if cohort_meta[n]['endpoint'] == 'infection' else '**D**'})"
        for n in names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    md += ["", header, sep]
    feat_rows = {}
    for r in rows:
        feat_rows.setdefault(r["feature"], {})[r["cohort"]] = r
    for _, label in FEATURES:
        cells = []
        for n in names:
            r = feat_rows[label][n]
            mark = ""
            if r["truth"] == "组中点映射":
                mark = "†"
            elif r["truth"] == "置0 (未采集)":
                mark = "‡"
            elif r["truth"].startswith("置常数"):
                mark = "§"
            elif r["truth"] == "半合成":
                mark = "¶"
            cells.append(f"{r['value']}{mark}")
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    # 终点行（标签不在 FEATURES 的 value 对里）
    end_label = "终点事件, n (%)"
    cells = []
    for n in names:
        r = feat_rows[end_label][n]
        cells.append(r["value"])
    md.append(f"| {end_label} | " + " | ".join(cells) + " |")
    md += ["", "## 队列出处与终点定义", ""]
    for n in names:
        cm = cohort_meta[n]
        md.append(f"- **{n}**（{cm['setting']}，N={cm['n']}，"
                  f"事件 {cm['n_events']} = {cm['event_rate']}%）："
                  f"{cm['outcome_def'] or '见对应 meta json'}")
    md += ["", "## 特征真实性明细（防误读）", ""]
    for n in names:
        fl = cohort_meta[n]["flags"]
        nonreal = {k: v for k, v in fl.items() if v != "R"}
        if nonreal:
            md.append(f"- {n}: " + "; ".join(
                f"{k}={TRUTH_LABEL[v]}" for k, v in nonreal.items()))
    md_path = os.path.join(DOCS, "table1_public_cohorts.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[OK] {csv_path} ({len(rows)} 行)")
    print(f"[OK] {md_path} ({len(names)} 队列 × {len(FEATURES) + 1} 行)")


if __name__ == "__main__":
    main()
