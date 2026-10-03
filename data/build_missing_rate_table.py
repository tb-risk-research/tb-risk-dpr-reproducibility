#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""各队列逐变量缺失率表（TRIPOD+AI 条目 11，补充 Table）

产出：
1. processed/table_missing_rates.csv —— 长表（机器可读）
2. docs/table_missing_rates.md —— 稿件格式

与 build_table1.py 同源（同一 COHORTS 注册表、同一 processed CSV），
特征真实性标注（R/M/Z/C/S）随行复用：防止把"置0/置常数"的
结构性缺失误读为真实观测（审稿关键——缺失率表若不带真实性标注，
Z/C 队列的 0% 缺失会被误读为高质量真实数据）。

缺失定义：列缺失（NaN）或空字符串；列不存在计为 100% 缺失
（结构性未采集，与真实性标注 Z 一致时在表中标注）。

用法：
    python data/build_missing_rate_table.py
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_table1 import (  # noqa: E402
    COHORTS, DOCS, FEATURES, PROC, SHORT, TRUTH_LABEL, TRUTH_OVERRIDE,
    load_truth_flags,
)

# 特征 → 实际列名（与 build_table1.fmt_cell 的映射一致）
COL_MAP = {"contact_exposure": "cumulative_exposure"}
# 队列 meta 路径（提取真实性标注用）
META_BY_NAME = {c[0]: c[2] for c in COHORTS}


def col_missing_stats(d, feat):
    """返回 (n_total, n_missing, col_present)。缺失 = NaN 或空串。"""
    col = COL_MAP.get(feat, feat)
    n = len(d)
    if col not in d.columns:
        return n, n, False
    s = d[col]
    isna = s.isna() | (s.astype(str).str.strip() == "")
    return n, int(isna.sum()), True


def main():
    rows = []
    cohort_flags = {}
    for name, csv_fn, meta_fn, endpoint, setting in COHORTS:
        csv_path = os.path.join(PROC, csv_fn)
        if not os.path.exists(csv_path):
            print(f"[跳过] {csv_fn} 不存在")
            continue
        d = pd.read_csv(csv_path)
        meta = None
        meta_path = os.path.join(PROC, meta_fn)
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        flags = load_truth_flags(name, meta)
        flags.update({f: t for (nm, f), t in TRUTH_OVERRIDE.items()
                      if nm == name})
        cohort_flags[name] = flags
        for feat, label in FEATURES:
            n_total, n_miss, col_present = col_missing_stats(d, feat)
            rows.append({
                "cohort": name,
                "feature_key": feat,
                "feature": label,
                "n_total": n_total,
                "n_missing": n_miss,
                "pct_missing": round(n_miss / n_total * 100, 1)
                if n_total else float("nan"),
                "col_present": col_present,
                "truth": flags[feat],
            })
        # 终点列缺失（终点缺失 = 标签缺失样本）
        n_total, n_miss, col_present = col_missing_stats(d, "tb_outcome")
        rows.append({
            "cohort": name, "feature_key": "outcome",
            "feature": "终点事件", "n_total": n_total,
            "n_missing": n_miss,
            "pct_missing": round(n_miss / n_total * 100, 1)
            if n_total else float("nan"),
            "col_present": col_present, "truth": "outcome",
        })

    out = pd.DataFrame(rows)
    csv_out = os.path.join(PROC, "table_missing_rates.csv")
    out.to_csv(csv_out, index=False, encoding="utf-8-sig")

    # ---- markdown 稿件表 ----
    names = [c[0] for c in COHORTS if c[0] in cohort_flags]
    mark_of = {"M": "†", "Z": "‡", "C": "§", "S": "¶"}
    md = ["# 补充 Table. 各队列逐变量缺失率（TRIPOD+AI 条目 11）\n"]
    md.append(
        "缺失定义：列缺失（NaN）或空字符串；列不存在计为 100% 缺失。"
        "真实性标注沿用 Table 1：†=组中点映射，‡=置0（未采集），"
        "§=置常数（人群推断），¶=半合成。**带 ‡/§/¶ 的 0% 缺失表示"
        "结构性构造值而非真实观测**；真缺失（真实性 R/M 且缺失率>0）"
        "为数据质量口径。\n")
    md.append(
        "口径说明：本表统计**建模矩阵**（processed 训练 CSV，构造完整"
        "病例分析）；**原始采集层**的缺失在方法节声明——HomeACF TST "
        "读数缺失 8.7%（2725/2985，完全病例分析）、PACTS 指示病例咳嗽"
        "时长缺失致单窗退化、SINAN 负滞后排除约 2.8%（协议 §7）。\n")
    header = ("| 特征 | " + " | ".join(names) + " |")
    sep = "|---" * (len(names) + 1) + "|"
    md += ["", header, sep]
    by_feat = {}
    for r in rows:
        by_feat.setdefault(r["feature"], {})[r["cohort"]] = r
    for _, label in FEATURES + [("tb_outcome", "终点事件")]:
        key = "tb_outcome" if label == "终点事件" else None
        src_label = "终点事件" if label == "终点事件" else label
        if src_label not in by_feat:
            continue
        cells = []
        for n in names:
            r = by_feat[src_label][n]
            if not r["col_present"]:
                cells.append("— (无列)")
                continue
            mark = mark_of.get(r["truth"], "")
            cells.append(f"{r['pct_missing']:.1f}%{mark}")
        md.append(f"| {label} 缺失率 | " + " | ".join(cells) + " |")
    md += ["", "## 真缺失明细（真实性 R/M 且缺失率 > 0 的单元格）", ""]
    n_true = 0
    for r in rows:
        if r["truth"] in ("R", "M") and r["pct_missing"] > 0:
            n_true += 1
            md.append(f"- {r['cohort']} / {r['feature']}: "
                      f"{r['n_missing']}/{r['n_total']} "
                      f"({r['pct_missing']}%)")
    if n_true == 0:
        md.append("- 无（全部 R/M 特征在全部队列无缺失）")
    md += ["", "## 结构性缺失明细（Z/C/S 标注的构造值）", ""]
    for n in names:
        fl = cohort_flags[n]
        struct = [k for k, v in fl.items()
                  if v in ("Z", "C", "S")]
        if struct:
            md.append(f"- {n}: " + ", ".join(
                f"{k}={TRUTH_LABEL[fl[k]]}" for k in struct))
    md_path = os.path.join(DOCS, "table_missing_rates.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[OK] {csv_out} ({len(rows)} 行)")
    print(f"[OK] {md_path} ({len(names)} 队列 × "
          f"{len(FEATURES) + 1} 行; 真缺失单元格 {n_true})")


if __name__ == "__main__":
    main()
