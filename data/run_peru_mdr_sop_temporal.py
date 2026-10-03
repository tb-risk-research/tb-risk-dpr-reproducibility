#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Peru MDR × SOP：核心机制的前瞻病程终点首测（P0-a，2026-09-05）

设计地位：ERASE-TB 之前唯一可填补"前瞻 + 病程"开放缺口的数据
（协议 v1.1 R2 多队列 meta 第二队列候选）。预声明为**方向分析**：
仅 32 事件拥有非空先证块（< 25-50 下限），只报方向 + CI，
不做判决主张——与分析前 §8 P1-P4 命题对表。

协议（与 HomeACF 判决性实验严格同构，validation/household_temporal.py）：
- CV：StratifiedGroupKFold(5) by family_id × 20 种子（折分配随机化）；
  池化 = 逐行平均 20 种子 OOF → 单一 AUROC / DeLong / cluster bootstrap；
- 推断：DeLong 配对（次）+ 户级 cluster bootstrap ×2000（主）；
- 模型：RF（主，HomeACF 保真）+ LR（敏感性）——冻结注册表超参。

时序轴（与 HomeACF 的关键差异，C1 构造）：
- 本数据无逐人日期，唯一时序轴 = Follow Up Time（病例=确诊时间，
  非病例=删失时间）；
- 先证块严格不等式：户内接触者 j 满足 f_j < f_i 才进入 i 的先证块
  （同日/同随访时长并列者互不进入——8,234 户内并列对被排除）；
- 指示病例不进先证块（其状态已由 index 特征承载，先证块只含
  接触者 incident 诊断）。

已知混淆通道（本脚本显式分解）：
- 时序轴由结局定义（病例中位随访 153 天 vs 非病例 424 天）→
  prior_n/prior_screened（纯时间结构，无事件内容）携带删失不对称；
- time_control 臂（base + prior_n + prior_screened）量化该通道；
- sop_mech 臂（base + prior_pos + 静态分母率）为去混淆机制臂；
- sop_full 臂（base + 四列协议块）为 HomeACF 协议保真臂。

门检（分析前执行并归档）：
- G1 类比门：prior_n≥1 覆盖率 19.9% < ERASE-TB 预注册 G1 阈值 40%
  → 按同款规则降级为"机制不可检验级"，支持方向分析预声明；
- C4 饱和门：基线臂 AUROC（远低于 0.99，预期不触发）。

用法：
    python data/run_peru_mdr_sop_temporal.py
输出：
    data/processed/peru_mdr_sop_temporal_20260905.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from tb_risk.validation.real_data_pi import (  # noqa: E402
    _cluster_bootstrap_delta, _group_cv_indices, _make_model, _pr_auc,
)
from tb_risk.validation.threshold_spec import compute_auc  # noqa: E402
from tb_risk.validation.layer_ablation import delong_paired_test  # noqa: E402
from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SRC = os.path.join(HERE, "processed", "ml_training_peru_mdr_real.csv")
OUT = os.path.join(HERE, "processed", "peru_mdr_sop_temporal_20260905.json")

N_SEEDS = 20
N_SPLITS = 5
N_BOOTSTRAP = 2000

# 主对比（±SOP 与分解）
CONTRASTS = (
    ("sop_full", "base"),        # 协议保真 ±SOP（主对比）
    ("sop_mech", "base"),        # 去混淆机制臂
    ("time_control", "base"),    # 删失通道控制臂（无事件内容）
    ("prior_only", "base"),      # 纯先证信号本体（§8.2 同款）
)


def build_prior_block(df):
    """严格不等式先证块 + 上界锚（hh_loo）。

    Returns: (prior_df, stats) —— prior 列直接拼接在 df 索引上。
    """
    y = df["tb_outcome"].astype(int).to_numpy()
    f = df["follow_up_days"].astype(float).to_numpy()
    hh = df["family_id"].to_numpy()
    n = len(df)
    base_rate = float(y.mean())

    prior_n = np.zeros(n)
    prior_pos = np.zeros(n)
    hh_size = np.zeros(n)
    hh_pos = np.zeros(n)
    for hh_id in np.unique(hh):
        idx = np.where(hh == hh_id)[0]
        fh, yh = f[idx], y[idx]
        hh_size[idx] = len(idx)
        hh_pos[idx] = yh.sum()
        # 严格不等式：f_j < f_i（并列排除）
        for k in idx:
            mask = (fh < f[k])
            prior_n[k] = mask.sum()
            prior_pos[k] = (mask & (yh == 1)).sum()

    prior_rate = np.where(prior_n > 0,
                          prior_pos / np.maximum(prior_n, 1.0), base_rate)
    # 机制-only 静态分母率（分母=户规模，不含 f_i 依赖）
    prior_pos_static_rate = prior_pos / np.maximum(hh_size - 1.0, 1.0)
    # 上界锚：leave-one-out 同户率（部署不可用——无视时序）
    hh_loo = np.where(hh_size > 1.0,
                      (hh_pos - y) / np.maximum(hh_size - 1.0, 1.0),
                      base_rate)

    prior_df = pd.DataFrame({
        "prior_n": prior_n,
        "prior_pos": prior_pos,
        "prior_rate": prior_rate,
        "prior_screened": (prior_n > 0).astype(float),
        "prior_pos_static_rate": prior_pos_static_rate,
    })
    ties = int(sum(int(s * (s - 1) / 2)
                   for s in df.groupby("family_id")["follow_up_days"]
                   .value_counts().values))
    stats = {
        "time_axis": ("Follow Up Time（病例=确诊时间中位 153 天，"
                      "非病例=删失时间中位 424 天）——时序轴由结局定义，"
                      "删失通道由 time_control 臂分解"),
        "strict_inequality": "f_j < f_i（户内并列 8,234 对被排除）",
        "base_rate": round(base_rate, 4),
        "prior_n_ge1": int((prior_n >= 1).sum()),
        "prior_n_ge1_rate": round(float((prior_n >= 1).mean()), 4),
        "prior_pos_ge1": int((prior_pos >= 1).sum()),
        "events_with_prior": int(((prior_pos >= 1) & (y == 1)).sum()),
        "nonevents_with_prior": int(((prior_pos >= 1) & (y == 0)).sum()),
        "crude_rate_with_prior": round(float(
            y[prior_pos >= 1].mean()), 4),
        "crude_rate_without_prior": round(float(
            y[prior_pos == 0].mean()), 4),
        "intra_hh_tied_pairs_excluded": ties,
        "hh_loo_auroc": None,  # 主循环后回填
    }
    return prior_df, stats, hh_loo


def add_interaction_features(d):
    """复算 9 维交互特征（与 train_from_real_data 完全同款逻辑，
    training.py L1317-L1356；CSV 只存 13 基础列）。"""
    hiv = (d["past_illness_type"].isin(["hiv", "HIV"])).astype(float)
    diabetes = (d["past_illness_type"].isin(
        ["diabetes", "糖尿病"])).astype(float)
    immunosupp = (d["past_illness_type"].isin(
        ["immunosuppressants", "免疫抑制"])).astype(float)
    immuno_score = (hiv * 2.0 + diabetes * 1.0 + immunosupp * 1.5 +
                    d["past_illness"].astype(float) * 0.5)
    age = d["age"].astype(float)
    out = pd.DataFrame(index=d.index)
    out["age_immuno"] = age * immuno_score
    out["dm_tb_synergy"] = diabetes * d["has_tb"].astype(float)
    out["age_bcg_decay"] = age * (1 - d["bcg_vaccine"].astype(float))
    delay = d.get("delay_days", d.get("ftd", 0))
    delay = delay.astype(float) if hasattr(delay, "astype") else float(delay)
    out["symptom_delay"] = d["has_symptoms"].astype(float) * np.minimum(
        delay / 30, 1.0)
    cough = d.get("cough_frequency", d.get("cough_freq", 0))
    cough = cough.astype(float) if hasattr(cough, "astype") else float(cough)
    contacts = d.get("contact_count", 5)
    contacts = (contacts.astype(float) if hasattr(contacts, "astype")
                else float(contacts))
    out["cough_contact"] = np.minimum(cough / 20, 1.0) * (contacts / 10)
    out["highrisk_comorbid"] = (d["is_high_risk"].astype(float) *
                                d["past_illness"].astype(float))
    out["immune_bcg"] = (1 - d["bcg_vaccine"].astype(float)) * immuno_score
    ce = d["cumulative_exposure"].astype(float)
    ts = d["time_span"].astype(float)
    out["exposure_accumulation"] = (ce / 80) * (ts / 10)
    out["age_diabetes"] = age * diabetes
    return out


def run_arms(df, prior_df, model_keys=("random_forest", "logistic"),
             seeds=range(N_SEEDS), subset=None):
    """全部臂 × 模型 × 种子 → 每行平均 OOF（池化）+ 逐种子 AUROC。

    subset: 布尔掩码（MDR/敏感分层次分析用）。
    """
    if subset is None:
        subset = np.ones(len(df), dtype=bool)
    d = df.loc[subset].reset_index(drop=True)
    p = prior_df.loc[subset].reset_index(drop=True)
    y = d["tb_outcome"].astype(int).to_numpy()
    groups = d["family_id"].to_numpy()

    base_cols = list(MLRiskPredictor.FEATURE_NAMES)
    inter = add_interaction_features(d)
    prior4 = ["prior_n", "prior_pos", "prior_rate", "prior_screened"]
    arms = {
        "base": base_cols + list(inter.columns),
        "sop_full": base_cols + list(inter.columns) + prior4,
        "sop_mech": base_cols + list(inter.columns) +
                    ["prior_pos", "prior_pos_static_rate"],
        "time_control": base_cols + list(inter.columns) +
                        ["prior_n", "prior_screened"],
        "prior_only": prior4,
    }
    X_all = pd.concat(
        [pd.concat([d[base_cols].apply(pd.to_numeric, errors="coerce")
                    .fillna(0), inter], axis=1), p],
        axis=1)

    oof = {}          # arm -> {model -> per-row mean OOF}
    per_seed = {}     # arm -> {model -> [auroc per seed]}
    for arm, cols in arms.items():
        X = X_all[cols].to_numpy(dtype=float)
        for mk in model_keys:
            oof_acc = np.zeros(len(y))
            aurocs = []
            for s in seeds:
                folds = _group_cv_indices(groups, y, n_splits=N_SPLITS,
                                          seed=s)
                oof_s = np.zeros(len(y))
                for tr, te in folds:
                    m = _make_model(mk, s)
                    m.fit(X[tr], y[tr])
                    oof_s[te] = m.predict_proba(X[te])[:, 1]
                oof_acc += oof_s
                aurocs.append(float(compute_auc(oof_s, y)))
            oof.setdefault(arm, {})[mk] = oof_acc / len(list(seeds))
            per_seed.setdefault(arm, {})[mk] = aurocs
    return {"y": y, "groups": groups, "oof": oof, "per_seed": per_seed,
            "arms": arms}


def summarize_run(run, tag, primary="random_forest"):
    """臂汇总 + 主对比（DeLong + 户级 cluster bootstrap）。"""
    y, groups = run["y"], run["groups"]
    arm_summary = {}
    for arm in run["oof"]:
        for mk, oof in run["oof"][arm].items():
            aurocs = run["per_seed"][arm][mk]
            arm_summary[f"{arm}:{mk}"] = {
                "pooled_auroc": round(float(compute_auc(oof, y)), 4),
                "seed_mean_auroc": round(float(np.mean(aurocs)), 4),
                "seed_sd_auroc": round(float(np.std(aurocs)), 4),
                "pr_auc": round(_pr_auc(y, oof), 4),
            }
    ladder = {}
    delong = {}
    for a, b in CONTRASTS:
        oa, ob = run["oof"][a][primary], run["oof"][b][primary]
        key = f"{a}_minus_{b}"
        cb = _cluster_bootstrap_delta(oa, ob, y, groups,
                                      n_bootstrap=N_BOOTSTRAP, seed=0)
        ladder[key] = cb
        try:
            dl = delong_paired_test(y, oa, ob)
            delong[key] = {
                "delta_auroc": round(dl["delta_auroc"], 4),
                "p_value": round(dl["p_value"], 4),
            }
        except Exception as e:  # noqa: BLE001
            delong[key] = {"error": str(e)}
    return {"tag": tag, "arm_summary": arm_summary,
            "ladder": ladder, "delong": delong}


def main():
    t0 = time.time()
    df = pd.read_csv(SRC)
    prior_df, stats, hh_loo = build_prior_block(df)
    y = df["tb_outcome"].astype(int).to_numpy()
    stats["hh_loo_auroc"] = round(float(compute_auc(hh_loo, y)), 4)

    print("===== 门检（分析前）=====")
    print("G1 类比：prior_n≥1 覆盖率 %.1f%%（ERASE-TB 阈值 40%% → %s）" % (
        100 * stats["prior_n_ge1_rate"],
        "通过" if stats["prior_n_ge1_rate"] >= 0.40 else "未通过→方向分析级"))
    print("事件数：先证块非空事件 %d / 149（< 25-50 下限 → 方向分析预声明）"
          % stats["events_with_prior"])
    print("粗关联：prior_pos≥1 组阳性率 %.4f vs 无先证组 %.4f" % (
        stats["crude_rate_with_prior"], stats["crude_rate_without_prior"]))
    print("上界锚 hh_loo AUROC: %.4f" % stats["hh_loo_auroc"])
    print()

    # 主分析（全队列）
    run = run_arms(df, prior_df)
    main_summary = summarize_run(run, "all")

    # 次分析：MDR vs 敏感分层
    strata = {}
    mdr = df["mdr_household"].to_numpy() == 1
    for name, mask in (("mdr", mdr), ("sensitive", ~mdr)):
        sub = df.loc[mask]
        n_hh = sub["family_id"].nunique()
        n_ev = int(sub["tb_outcome"].sum())
        if n_ev < 10 or n_hh < N_SPLITS:
            strata[name] = {"skipped": f"事件 {n_ev} / 户 {n_hh} 不足"}
            continue
        r = run_arms(df, prior_df, seeds=range(N_SEEDS), subset=mask)
        strata[name] = {
            "n": int(mask.sum()), "n_events": n_ev,
            "n_households": int(n_hh),
            **summarize_run(r, name),
        }

    out = {
        "date": "2026-09-05",
        "name": "peru_mdr_sop_temporal_v1",
        "status": ("方向分析（预声明）：32 事件拥有非空先证块，"
                   "低于 25-50 下限——只报方向 + CI，不做判决主张"),
        "design": {
            "source": ("Dryad Peru MDR household cohort "
                       "doi:10.5061/dryad.br760（CC0，Grandjean 2016）"),
            "endpoint": "tb_outcome = Incident TB Disease（前瞻病程终点）",
            "n": len(df), "n_events": int(y.sum()),
            "n_households": int(df["family_id"].nunique()),
            "cv": f"StratifiedGroupKFold({N_SPLITS}) by family_id",
            "n_seeds": N_SEEDS,
            "pooling": "逐行平均 20 种子 OOF（池化）",
            "inference": (f"户级 cluster bootstrap ×{N_BOOTSTRAP}（主）"
                          " + DeLong 配对（次）"),
            "models": "RF（主，HomeACF 保真）+ LR（敏感性），冻结注册表超参",
            "leakage_argument": (
                "先证块=同户 f_j<f_i 严格更早接触者的结局函数；户分组 CV "
                "同户同折 → 测试折先证标签从未进训练；指示病例不进先证块"),
            "confounding_disclosure": (
                "时序轴由结局定义（病例随访中位 153 天 vs 非病例 424 天），"
                "prior_n/prior_screened 携带删失不对称通道——time_control "
                "臂量化该通道，sop_mech 臂（静态分母）为去混淆估计"),
            "gate_checks": {
                "g1_analog_coverage": stats["prior_n_ge1_rate"],
                "g1_threshold_erasetb": 0.40,
                "g1_verdict": ("未通过 → 按预注册 G1 同款规则为"
                               "机制不可检验级，支持方向分析预声明"),
                "c4_baseline_auroc": None,  # 主循环后回填
            },
            "prior_block_stats": stats,
            "contrasts": [f"{a}_minus_{b}" for a, b in CONTRASTS],
        },
        "main": main_summary,
        "strata": strata,
    }
    # C4 回填
    out["design"]["gate_checks"]["c4_baseline_auroc"] = \
        main_summary["arm_summary"]["base:random_forest"]["pooled_auroc"]

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("===== 主分析（全队列，RF）=====")
    for k in ("base", "sop_full", "sop_mech", "time_control", "prior_only"):
        s = main_summary["arm_summary"][f"{k}:random_forest"]
        print("  %-14s pooled AUROC=%.4f (seed %.4f±%.4f) PR=%.4f" % (
            k, s["pooled_auroc"], s["seed_mean_auroc"],
            s["seed_sd_auroc"], s["pr_auc"]))
    print()
    for k, v in main_summary["ladder"].items():
        star = "*" if v.get("ci_excludes_zero") else " "
        print("  %s %-28s Δ=%+.4f CI=[%+.4f,%+.4f] P(>0)=%.3f DeLong p=%.4f" % (
            star, k, v["mean"], v["bootstrap_ci"][0], v["bootstrap_ci"][1],
            v["p_positive"], main_summary["delong"][k].get("p_value", -1)))
    print()
    for name in ("mdr", "sensitive"):
        s = strata.get(name, {})
        if "ladder" not in s:
            print("  [%s] %s" % (name, s.get("skipped", "跳过")))
            continue
        for k, v in s["ladder"].items():
            if k.startswith(("sop_full", "sop_mech")):
                print("  [%s] %-28s Δ=%+.4f CI=[%+.4f,%+.4f]" % (
                    name, k, v["mean"], v["bootstrap_ci"][0],
                    v["bootstrap_ci"][1]))
    print("\n归档 → %s" % OUT)
    print("总耗时 %.1f 分钟" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
