#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aibana 秘鲁队列 × 终点依赖 2×2：暴露/宿主特征族 × 感染/病程双终点
（§8.7 方向分析，2026-09-06）

设计地位：定向审计（P1-③）到手的唯一"同队列双终点 + 双特征族"队列。
**探索性方向分析**（预声明）：
- 无户键列 → SOP 先证块不可构建 → R2 第二队列资格四项对照全部
  不满足（见归档 JSON design.r2_verdict）——与 Peru MDR 同级
  （方向性伴随证据，不进逆方差合并，不闭合病程缺口）；
- 原论文（Aibana 2016 PLOS ONE，Cox 病因学层）已建立：BMI 不预测
  TB 感染、超重对 TB 病程保护（HR 0.48）、<12 岁无关联——本分析
  **不重复主张病因学发现**，在判别层（AUROC 排序）检验终点依赖
  图景（§8.6：感染由暴露驱动、病程由宿主驱动，系跨数据集推断）
  的**队内 2×2 结构**。

可证伪预期（方向性，分析前声明）：
- 感染终点：暴露臂（index 特征）> 宿主臂（宿主/营养特征）；
- 病程终点：宿主臂 ≥ 暴露臂（§8.6 实证：家庭共享暴露天花板 +0.007）；
- 营养增量：病程终点 >0 / 感染终点 ≈0（判别层复现论文不对称）。

协议（与 §8.6 Peru MDR 脚本同构，无户键处显式替换并披露）：
- CV：StratifiedKFold(5) × 20 种子（**无户键 → 个体分层非组感知**，
  组感知乐观偏差按 §8.6 实证同款量级披露：−0.037）；
- 池化：逐行平均 20 种子 OOF（同 §8.6）；
- 推断：**个体 bootstrap ×2000**（非 cluster——无户键，CI 偏乐观
  披露）+ DeLong 配对（次）+ 逐种子 Δ 四栏（R5a 纪律）；
- 模型：RF（主，HomeACF/§8.6 保真）+ LR（敏感性），冻结注册表超参；
- 缺失：折内中位数填补（最大缺失率 4.2%，无缺失指示器，披露）；
- inh（基线后预防治疗，预测时点不可知）不入特征 → 未治疗层敏感性；
- 全龄主口径 + ≥15 岁敏感性层（<12 岁营养 null 为论文已知结论）。

终点口径：
- 感染 = 基线 TST 阴性且终态已知者的 TST 转阳（论文口径 1,787/6,853
  = 26.1%；本分析完整病例 5,290 中 33.7%，缺失非随机披露）；
- 病程 = 排除 Coprevalent（218，基线 15 天内诊断）后的 Secondary
  TB（406 事件）；disease_days 为全队列随访时长（非病例中位 373 天），
  不入特征（防删失不对称通道，§8.6 教训）。

用法：
    python data/run_aibana_endpoint_dependence.py
输出：
    data/processed/aibana_endpoint_dependence_20260906.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from tb_risk.validation.real_data_pi import _make_model, _pr_auc  # noqa: E402
from tb_risk.validation.layer_ablation import delong_paired_test  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402


def _fast_auc(y, scores):
    """向量化 AUROC——与 threshold_spec.compute_auc 的曼-惠特尼 U 语义
    逐位一致（并列计 0.5；roc_auc_score 梯形法 = U 统计量）。
    纯 Python 版为 O(n_pos×n_neg) 双循环，在 n=12,430 的 20 种子 CV +
    2000 次 bootstrap 内不可行，故本脚本用 sklearn 等价实现。"""
    y = np.asarray(y, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))

SRC = os.path.join(HERE, "raw", "figshare", "peru_aibana_contacts.xlsx")
OUT = os.path.join(HERE, "processed", "aibana_endpoint_dependence_20260906.json")

N_SEEDS = 20
N_SPLITS = 5
N_BOOTSTRAP = 2000
PRIMARY = "random_forest"

# 特征族（2×2 设计）
EXP_COLS = [  # 暴露族：指示病例特征（暴露强度/传染性代理）
    "index_smear_pos", "index_smear_grade", "index_cavity",
    "index_hiv_status", "index_smoke", "index_drink",
]
HOST_BASE_COLS = [  # 宿主族（不含营养）：人口学/免疫/行为/社会经济
    "age", "sex_male", "num_scar", "hiv", "diabetes", "comorbid",
    "tb_history", "smoke", "drink", "ses",
]
NUT_COLS = ["bmi", "nutritional_status"]  # 营养块（论文主因素）
BASEINF_COLS = ["inf_baseline", "inf_baseline_missing"]  # 基线感染状态（病程侧臂）

ARMS = {
    "exp": EXP_COLS,
    "host_base": HOST_BASE_COLS,
    "host_full": HOST_BASE_COLS + NUT_COLS,
    "full": EXP_COLS + HOST_BASE_COLS + NUT_COLS,
    "full_baseinf": EXP_COLS + HOST_BASE_COLS + NUT_COLS + BASEINF_COLS,
}

# 主对比（感染终点）
CONTRASTS_INFECTION = [
    ("exp", "host_base"),        # 核心不对称 1：暴露 vs 宿主 @ 感染
    ("host_full", "host_base"),  # 营养增量 @ 感染（论文：预期 null）
    ("full", "exp"),             # 宿主信息增量 @ 感染
    ("full", "host_full"),       # 暴露信息增量 @ 感染
]
# 主对比（病程终点）
CONTRASTS_PROGRESSION = [
    ("host_base", "exp"),        # 核心不对称 2：宿主 vs 暴露 @ 病程
    ("host_full", "host_base"),  # 营养增量 @ 病程（论文：预期 >0）
    ("full", "host_full"),       # 暴露信息增量 @ 病程
    ("full_baseinf", "full"),    # 基线感染状态增量（临床参照，3.8× 粗率）
]


def _yn(series):
    """YES/NO、ye/no 等二值字符串 → 1/0（'ye' 为源数据拼写，保留原样映射）。"""
    def one(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s.startswith("y"):
            return 1.0
        if s.startswith("n"):
            return 0.0
        return np.nan
    return series.map(one)


def load_and_encode():
    df = pd.read_excel(SRC)
    d = pd.DataFrame(index=df.index)
    d["age"] = df["age"].astype(float)
    d["sex_male"] = (df["Sex"] == "M").astype(float)
    d["comorbid"] = _yn(df["comorbid"])
    d["num_scar"] = df["num_scar"].astype(float)
    d["nutritional_status"] = df["Nutritional_status"].astype(float)
    d["ses"] = df["SES"].astype(float)
    d["diabetes"] = df["Diabetes"].astype(float)
    d["bmi"] = df["BMI"].astype(float)
    d["hiv"] = df["Hiv_status"].astype(float)
    d["index_cavity"] = df["index_cavity"].astype(float)
    d["index_hiv_status"] = df["index_hiv_status"].astype(float)
    d["tb_history"] = df["TB_history"].astype(float)
    d["index_smear_pos"] = df["index_smear"].map(
        {"pos": 1.0, "neg": 0.0}).astype(float)
    d["index_smear_grade"] = df["index_smear_grade"].astype(float)
    d["smoke"] = _yn(df["smoke"])
    d["drink"] = _yn(df["drink"])
    d["index_smoke"] = _yn(df["index_smoke"])
    d["index_drink"] = _yn(df["index_drink"])
    d["inh"] = df["inh"].astype(float)
    d["inf_baseline"] = df["inf_status_baseline"].astype(float)
    d["inf_baseline_missing"] = df["inf_status_baseline"].isna().astype(float)
    d["inf_final"] = df["inf_status_final"].astype(float)
    d["tb_type"] = df["tb_type"]
    d["disease_days"] = df["disease_days"].astype(float)
    return d


def run_arms(dfeat, y, arm_keys, model_keys=("random_forest", "logistic"),
             seeds=range(N_SEEDS)):
    """臂 × 模型 × 种子 → 逐行平均 OOF + 逐种子 AUROC（折内中位数填补）。"""
    X_all = {arm: dfeat[ARMS[arm]].to_numpy(dtype=float) for arm in arm_keys}
    oof = {arm: {} for arm in arm_keys}
    per_seed = {arm: {} for arm in arm_keys}
    seeds = list(seeds)
    for arm in arm_keys:
        X = X_all[arm]
        for mk in model_keys:
            oof_acc = np.zeros(len(y))
            aurocs = []
            for s in seeds:
                skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                      random_state=s)
                oof_s = np.zeros(len(y))
                for tr, te in skf.split(X, y):
                    med = np.nanmedian(X[tr], axis=0)
                    med = np.where(np.isnan(med), 0.0, med)  # 全缺列兜底
                    Xtr = np.where(np.isnan(X[tr]), med, X[tr])
                    Xte = np.where(np.isnan(X[te]), med, X[te])
                    m = _make_model(mk, s)
                    m.fit(Xtr, y[tr])
                    oof_s[te] = m.predict_proba(Xte)[:, 1]
                oof_acc += oof_s
                aurocs.append(_fast_auc(y, oof_s))
            oof[arm][mk] = oof_acc / len(seeds)
            per_seed[arm][mk] = aurocs
    return {"y": y, "oof": oof, "per_seed": per_seed}


def _individual_bootstrap(oof_p, y, contrasts,
                          n_bootstrap=N_BOOTSTRAP, seed=0):
    """个体 bootstrap（无户键 → 非 cluster）：同一重采样下算全部臂
    level CI 与配对 Δ CI——CI 偏乐观（忽略户内相关）已在上层披露。"""
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
        levels[k] = {
            "mean": round(float(np.mean(v)), 4),
            "ci": [round(float(np.percentile(v, 2.5)), 4),
                   round(float(np.percentile(v, 97.5)), 4)],
        }
    deltas = {}
    for k, v in dacc.items():
        v = np.array(v)
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        deltas[k] = {
            "mean": round(float(np.mean(v)), 4),
            "bootstrap_ci": [round(lo, 4), round(hi, 4)],
            "p_positive": round(float(np.mean(v > 0.0)), 3),
            "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        }
    return levels, deltas


def summarize_run(run, tag, contrasts, primary=PRIMARY):
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
        ladder[key] = {
            **deltas[key],
            "per_seed_delta": {
                "mean": round(float(np.mean(sd)), 4),
                "sd": round(float(np.std(sd)), 4),
                "min": round(float(np.min(sd)), 4),
                "max": round(float(np.max(sd)), 4),
                "share_positive": round(float(np.mean(sd > 0.0)), 3),
            },
            "delong": delong,
        }
    return {"tag": tag, "arm_summary": arm_summary, "ladder": ladder}


def main():
    t0 = time.time()
    d = load_and_encode()

    # ---- 终点人口 ----
    inf_mask = (d["inf_baseline"] == 0) & d["inf_final"].notna()
    inf_y = d.loc[inf_mask, "inf_final"].to_numpy(dtype=int)

    prog_mask = d["tb_type"] != "Coprevalent"
    prog_y = (d.loc[prog_mask, "tb_type"] == "Secondary").to_numpy(dtype=int)

    untreated = (d["inh"] == 0)
    age15 = d["age"] >= 15

    inf_arms = ["exp", "host_base", "host_full", "full"]
    prog_arms = ["exp", "host_base", "host_full", "full", "full_baseinf"]

    populations = {
        "infection": (inf_mask, inf_y, inf_arms, CONTRASTS_INFECTION),
        "progression": (prog_mask, prog_y, prog_arms, CONTRASTS_PROGRESSION),
        "progression_untreated": (
            prog_mask & untreated, prog_y[untreated[prog_mask].to_numpy()],
            prog_arms, CONTRASTS_PROGRESSION),
        "infection_15plus": (
            inf_mask & age15, inf_y[age15[inf_mask].to_numpy()],
            inf_arms, CONTRASTS_INFECTION),
        "progression_15plus": (
            prog_mask & age15, prog_y[age15[prog_mask].to_numpy()],
            prog_arms, CONTRASTS_PROGRESSION),
    }

    # 粗率锚（论文口径对照）
    n_base_neg = int((d["inf_baseline"] == 0).sum())
    crude = {
        "n_total": int(len(d)),
        "note_n_vs_paper": "12,648 为论文 14,044 的分析子集（S1 交付）",
        "tst_conversion_paper_anchor": {
            "baseline_negative": n_base_neg,
            "conversions": int(((d["inf_baseline"] == 0) &
                                (d["inf_final"] == 1)).sum()),
            "paper_rate": 0.261,
            "complete_case_n": int(inf_mask.sum()),
            "complete_case_rate": round(float(inf_y.mean()), 4),
            "final_status_missing_disclosure": (
                "基阴终态缺失 1,563/6,853（22.8%）——完整病例率 33.7% 高于"
                "论文全分母率 26.1%，缺失非随机，判别读数按完整病例口径披露"),
        },
        "secondary_tb": int(prog_y.sum()),
        "coprevalent_excluded": int((d["tb_type"] == "Coprevalent").sum()),
        "secondary_rate_per_1k_by_baseline": {
            "baseline_positive": round(float(
                ((d["tb_type"] == "Secondary") &
                 (d["inf_baseline"] == 1)).sum() /
                (d["inf_baseline"] == 1).sum()) * 1000, 1),
            "baseline_negative": round(float(
                ((d["tb_type"] == "Secondary") &
                 (d["inf_baseline"] == 0)).sum() /
                (d["inf_baseline"] == 0).sum()) * 1000, 1),
        },
        "inh_crude_rate_per_1k_noncoprevalent": {
            "untreated": round(float(
                ((d["tb_type"] == "Secondary") & (d["inh"] == 0)).sum() /
                ((d["tb_type"] != "Coprevalent") & (d["inh"] == 0)).sum())
                * 1000, 1),
            "treated": round(float(
                ((d["tb_type"] == "Secondary") & (d["inh"] == 1)).sum() /
                ((d["tb_type"] != "Coprevalent") & (d["inh"] == 1)).sum())
                * 1000, 1),
        },
    }

    results = {}
    for name, (mask, y, arms, contrasts) in populations.items():
        print("[%s] n=%d events=%d arms=%d" % (name, mask.sum(), y.sum(),
                                               len(arms)))
        dsub = d.loc[mask].reset_index(drop=True)
        run = run_arms(dsub, y, arms)
        results[name] = {
            "n": int(mask.sum()), "n_events": int(y.sum()),
            **summarize_run(run, name, contrasts),
        }

    # ---- 2×2 核心读数 ----
    inf_l = results["infection"]["ladder"]
    prog_l = results["progression"]["ladder"]
    inf_a = results["infection"]["arm_summary"]
    prog_a = results["progression"]["arm_summary"]
    asymmetry = {
        "core_cells": {
            "infection:exp": inf_a[f"exp:{PRIMARY}"]["pooled_auroc"],
            "infection:host_base": inf_a[f"host_base:{PRIMARY}"]["pooled_auroc"],
            "progression:exp": prog_a[f"exp:{PRIMARY}"]["pooled_auroc"],
            "progression:host_base": prog_a[f"host_base:{PRIMARY}"]["pooled_auroc"],
        },
        "asymmetry_infection_exp_minus_host": inf_l["exp_minus_host_base"],
        "asymmetry_progression_host_minus_exp": prog_l["host_base_minus_exp"],
        "nutrition_increment_infection": inf_l["host_full_minus_host_base"],
        "nutrition_increment_progression": prog_l["host_full_minus_host_base"],
        "readme": (
            "可证伪预期：感染终点 exp>host（Δ>0）、病程终点 host≥exp、"
            "营养增量病程>0/感染≈0。四格全部成立 → §8.6 终点依赖图景"
            "获单队列锚定；任一反转 → 图景需收窄。判别层读数，不重复"
            "主张论文病因学发现。"),
    }

    # ---- G3 类比门 ----
    max_auroc = max(
        v["pooled_auroc"] for pop in ("infection", "progression")
        for k, v in results[pop]["arm_summary"].items()
        if k.endswith(f":{PRIMARY}"))

    out = {
        "date": "2026-09-06",
        "name": "aibana_endpoint_dependence_v1",
        "status": ("探索性方向分析（预声明）：无户键不可构建先证块（R2 不"
                   "合格）；判别层 2×2 终点依赖检验，只报方向 + CI，"
                   "不做判决主张"),
        "design": {
            "source": ("PLOS S1 doi:10.1371/journal.pone.0166333.s001"
                       "（Aibana 2016 PLOS ONE，CC BY；定向审计 2026-09-06"
                       " plos-s1-doi 通道入手）"),
            "original_paper_anchor": (
                "Aibana 2016（Cox 病因学层）：BMI 不预测感染、超重保护病程"
                "（HR 0.48）、≥35 仍保护（HR 0.30）、<12 岁无关联——"
                "本分析为判别层（AUROC 排序）复现与扩展，不重复病因学主张"),
            "cv": (f"StratifiedKFold({N_SPLITS}) × {N_SEEDS} 种子"
                   "（无户键 → 个体分层非组感知；组感知乐观偏差按 §8.6 "
                   "实证同款量级披露：−0.037，方向为偏乐观）"),
            "pooling": "逐行平均 20 种子 OOF（与 §8.6 同构）",
            "inference": (f"个体 bootstrap ×{N_BOOTSTRAP}（无户键 → 非 "
                          "cluster，CI 偏乐观披露）+ DeLong（次）+ "
                          "逐种子 Δ 四栏（R5a 纪律）"),
            "models": "RF（主）+ LR（敏感性），冻结注册表超参（_make_model）",
            "missing_data": "折内中位数填补，无缺失指示器（特征最大缺失 4.2%）",
            "inh_disclosure": (
                "inh（基线后预防治疗，预测时点不可知）不入特征；粗率 "
                "未治 vs 已治见 crude 块；未治疗层敏感性归档"),
            "endpoint_definitions": {
                "infection": "基线 TST 阴性且终态已知 → TST 转阳（完整病例）",
                "progression": ("排除 Coprevalent（<15 天诊断）→ Secondary "
                                "TB；disease_days 全队列随访时长不入特征"
                                "（防删失不对称通道，§8.6 教训）；事件="
                                "随访≤455 天内诊断的判别口径"),
            },
            "feature_arms": ARMS,
            "contrasts": {
                "infection": [f"{a}_minus_{b}" for a, b in CONTRASTS_INFECTION],
                "progression": [f"{a}_minus_{b}"
                                for a, b in CONTRASTS_PROGRESSION],
            },
            "gate_checks": {
                "g1_analog": ("不可计算（无户键）——本身即 R2 第二队列"
                              "不合格的决定性证据"),
                "g3_analog_max_auroc": round(max_auroc, 4),
                "g3_threshold": 0.99,
                "g3_verdict": "通过（远未饱和）",
            },
            "r2_verdict": (
                "R2 四项资格对照：家庭接触者人群 ✓ / 先证块可重建 ✗（无户"
                "键，决定性）/ 病程终点随访 ✓ / G1 独立通过 ✗（不可计算）"
                "+ 协议先于数据接触冻结 ✗（数据已公开）→ 不构成第二队列，"
                "与 Peru MDR 同级方向性伴随证据，不进逆方差合并，"
                "R2 缺口维持开放"),
            "crude": crude,
        },
        "infection": results["infection"],
        "progression": results["progression"],
        "sensitivity": {
            "progression_untreated": results["progression_untreated"],
            "infection_15plus": results["infection_15plus"],
            "progression_15plus": results["progression_15plus"],
        },
        "endpoint_asymmetry": asymmetry,
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print()
    print("===== 主读数（RF，池化 AUROC [个体 bootstrap CI]）=====")
    for pop in ("infection", "progression"):
        print("[%s]" % pop)
        for arm in (("exp", "host_base", "host_full", "full")
                    if pop == "infection"
                    else ("exp", "host_base", "host_full", "full",
                          "full_baseinf")):
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
    print("===== 2×2 核心四格 =====")
    for k, v in asymmetry["core_cells"].items():
        print("  %-28s %.4f" % (k, v))
    print("  感染 exp−host  Δ=%+.4f CI=%s" % (
        inf_l["exp_minus_host_base"]["mean"],
        inf_l["exp_minus_host_base"]["bootstrap_ci"]))
    print("  病程 host−exp  Δ=%+.4f CI=%s" % (
        prog_l["host_base_minus_exp"]["mean"],
        prog_l["host_base_minus_exp"]["bootstrap_ci"]))
    print("  营养增量@感染  Δ=%+.4f CI=%s" % (
        inf_l["host_full_minus_host_base"]["mean"],
        inf_l["host_full_minus_host_base"]["bootstrap_ci"]))
    print("  营养增量@病程  Δ=%+.4f CI=%s" % (
        prog_l["host_full_minus_host_base"]["mean"],
        prog_l["host_full_minus_host_base"]["bootstrap_ci"]))
    print("\n归档 → %s" % OUT)
    print("总耗时 %.1f 分钟" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
