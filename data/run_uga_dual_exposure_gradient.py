#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""乌干达双队列 × 个体级暴露梯度剂量响应（§8.6 暴露代理腿，2026-09-06）

设计地位：§8.7 反转腿的针对性对照——Aibana 暴露臂（指示病例特征）
为户级常量、个体排序力弱（宿主 0.664 ≫ 暴露 0.549）；乌干达双队列
携带**个体级暴露梯度**（与指示病例的共处时长/亲近度/关系），检验
暴露梯度在个体感染排序中是否携带信号（§8.6"感染由暴露驱动"腿的
强度维度；时序维度仍归 ERASE-TB）。

队列：
- Muchuro 2022（PLOS GPH，n=352/59 指示病例户，IGRA 阳 115/352
  = 32.7%）：prox2index（亲近度 4 级）+ tw_tbindex（共处时长 3 级）
  + reln_w_index（关系）+ ncontacts；**有指示病例键 tbindex_id →
  StratifiedGroupKFold 组感知 CV + 户级 cluster bootstrap**（本
  分析线程首个组感知队列）；
- Mayito 2023（PLOS ONE，n=202 坎帕拉成人接触者，QFT-Plus 阳
  105/192 = 54.7%，排除 10 indeterminate）：closeness（4 级）+
  intensity + bacilliload + coughdura + lastcontact + contactplace
  + relation + contactno；无簇键 → 个体分层 CV + 个体 bootstrap。

可证伪预期（预声明）：
- U1 full−host > 0 于两队列（个体级暴露梯度携带感染排序增量——
  与 §8.7 户级常量臂的设计对照）；
- U2 Muchuro exp 臂单独 AUROC > 0.5（bootstrap CI 下界 > 0.5）。

声明时点披露：剂量响应粗率交叉表（Muchuro prox2index 23.4%→
57.1% 等）已于勘察阶段查看——剂量响应为方向已知，本分析的主要
新增检验是**判别层臂间对比**（exp/host/full AUROC 与 Δ）。

排除：Mayito IGRA 定量列（tb1_ag/tb2_ag，终点决定性——QFT 判定
即由其导出，§8.6 Peru MDR 终点泄漏教训）；Mayito treatment（时点
不明）；Muchuro igra_exp/pickone_idx（调查管理列）。

协议：与 §8.7 同构——RF 主 + LR 敏感性（冻结注册表）、20 种子 ×
5 折、逐行平均池化、bootstrap×2000（Muchuro 户级 cluster、Mayito
个体）+ DeLong（次）+ 逐种子 Δ 四栏（R5a 纪律）、折内中位数填补。
地位：探索性方向分析（公开数据、两文原文已发表暴露-感染关联的
流行病学分析，本分析不重复其病因学主张，检验判别层结构）。

用法：
    python data/run_uga_dual_exposure_gradient.py
输出：
    data/processed/uga_dual_exposure_gradient_20260906.json
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
from sklearn.model_selection import (StratifiedGroupKFold,  # noqa: E402
                                     StratifiedKFold)

MU_XLSX = os.path.join(HERE, "raw", "figshare", "uganda_household_igra_s001.xlsx")
MY_XLSX = os.path.join(HERE, "raw", "dryad", "uganda_qftplus_contacts.xlsx")
OUT = os.path.join(HERE, "processed", "uga_dual_exposure_gradient_20260906.json")

N_SEEDS = 20
N_SPLITS = 5
N_BOOTSTRAP = 2000
PRIMARY = "random_forest"

PROX_MAP = {  # 亲近度 → 暴露强度序（1=最低）
    "Sleeps in a different house": 1.0,
    "Sleeps in a different room, same house": 2.0,
    "Sleeps in different bed but same room": 3.0,
    "Sleeps in the same bed": 4.0,
}
TW_MAP = {  # 共处时长 → 序（1=最低）
    "Not every day": 1.0,
    "Everyday,<50% of the day": 2.0,
    "Everyday, >=50% of the day": 3.0,
}


def _fast_auc(y, scores):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def load_muchuro():
    mu = pd.read_excel(MU_XLSX)
    d = pd.DataFrame(index=mu.index)
    d["prox"] = mu["prox2index"].map(PROX_MAP)
    d["tw"] = mu["tw_tbindex"].map(TW_MAP)
    d["reln_parent_sibling"] = (mu["reln_w_index"] == "Parent/Sibling").astype(float)
    d["reln_spouse"] = (mu["reln_w_index"] == "Spouse").astype(float)
    d["reln_others"] = (mu["reln_w_index"] == "Others").astype(float)
    d["ncontacts"] = pd.to_numeric(mu["ncontacts"], errors="coerce")
    d["age"] = pd.to_numeric(mu["ageofcontact"], errors="coerce")
    d["sex_male"] = (mu["gender"] == "Male").astype(float)
    d["educ"] = mu["educ"].map({"Primary": 1.0, "Post-primary": 2.0})
    d["employment_agri"] = (mu["employment"] == "Agriculture").astype(float)
    d["employment_business"] = (mu["employment"] == "Business").astype(float)
    d["employment_other"] = mu["employment"].isin(
        ["Casual labourer", "Formal employment"]).astype(float)
    d["smoke"] = mu["smk_status"].map(
        {"Never smoker": 0.0, "Ex-smoker": 1.0, "Current smoker": 2.0})
    d["hiv"] = mu["hivstatus"].map({"Negative": 0.0, "Positive": 1.0})
    d["bcg"] = (mu["bcgscar"] == "Yes").astype(float)
    d["livewithsmoker"] = (mu["livewithsmoker"] == "Yes").astype(float)
    d["group"] = mu["tbindex_id"]
    d["y"] = pd.to_numeric(mu["igra_pos"], errors="coerce")
    return d


def load_mayito():
    my = pd.read_excel(MY_XLSX)
    v1 = my[my["visit no."] == 1].copy()
    v1 = v1[v1["igrastatus"].isin([0, 1])]
    d = pd.DataFrame(index=v1.index)
    for src, name in [("closeness", "closeness"), ("intensity", "intensity"),
                      ("bacilliload", "bacilliload"), ("coughdura", "coughdura"),
                      ("lastcontact", "lastcontact"), ("contactplace", "contactplace"),
                      ("relation", "relation"), ("contactno", "contactno")]:
        v = pd.to_numeric(v1[src], errors="coerce")
        d[name] = v.replace(9, np.nan)
    d["age"] = pd.to_numeric(v1["age (years)"], errors="coerce")
    d["sex_male"] = (pd.to_numeric(v1["gender"], errors="coerce") == 1).astype(float)
    for src, name in [("education", "education"), ("occupation", "occupation"),
                      ("medcond", "medcond")]:
        v = pd.to_numeric(v1[src], errors="coerce")
        d[name] = v.replace(9, np.nan)
    d["hiv"] = pd.to_numeric(v1["hivstatus"], errors="coerce")
    d["bcg"] = pd.to_numeric(v1["bcgscar"], errors="coerce")
    d["smoke"] = pd.to_numeric(v1["smokingstatus"], errors="coerce").replace(9, np.nan)
    d["livecsmoker"] = pd.to_numeric(v1["livecsmoker"], errors="coerce")
    d["y"] = pd.to_numeric(v1["igrastatus"], errors="coerce").astype(int)
    return d


MU_ARMS = {
    "exp": ["prox", "tw", "reln_parent_sibling", "reln_spouse", "reln_others",
            "ncontacts"],
    "host": ["age", "sex_male", "educ", "employment_agri", "employment_business",
             "employment_other", "smoke", "hiv", "bcg", "livewithsmoker"],
    "full": ["prox", "tw", "reln_parent_sibling", "reln_spouse", "reln_others",
             "ncontacts", "age", "sex_male", "educ", "employment_agri",
             "employment_business", "employment_other", "smoke", "hiv", "bcg",
             "livewithsmoker"],
}
MY_ARMS = {
    "exp": ["closeness", "intensity", "bacilliload", "coughdura", "lastcontact",
            "contactplace", "relation", "contactno"],
    "host": ["age", "sex_male", "education", "occupation", "medcond", "hiv",
             "bcg", "smoke", "livecsmoker"],
    "full": ["closeness", "intensity", "bacilliload", "coughdura", "lastcontact",
             "contactplace", "relation", "contactno", "age", "sex_male",
             "education", "occupation", "medcond", "hiv", "bcg", "smoke",
             "livecsmoker"],
}
CONTRASTS = [("exp", "host"), ("full", "host"), ("full", "exp")]


def run_cv(dfeat, y, arms, arms_dict, groups=None):
    """groups 给定 → StratifiedGroupKFold（组感知）；否则 StratifiedKFold。"""
    X_all = {a: dfeat[arms_dict[a]].to_numpy(dtype=float) for a in arms}
    oof = {a: {} for a in arms}
    per_seed = {a: {} for a in arms}
    for arm in arms:
        X = X_all[arm]
        for mk in ("random_forest", "logistic"):
            oof_acc = np.zeros(len(y))
            aurocs = []
            for s in range(N_SEEDS):
                if groups is not None:
                    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                              random_state=s)
                    splits = cv.split(X, y, groups)
                else:
                    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                          random_state=s)
                    splits = skf.split(X, y)
                oof_s = np.zeros(len(y))
                for tr, te in splits:
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


def bootstrap(oof_p, y, contrasts, clusters=None, n_bootstrap=N_BOOTSTRAP):
    """clusters 给定 → 户级 cluster bootstrap；否则个体。"""
    rng = np.random.RandomState(0)
    n = len(y)
    if clusters is not None:
        uniq = pd.unique(clusters)
        idx_by_c = {c: np.where(clusters == c)[0] for c in uniq}
        draw = lambda: np.concatenate(
            [idx_by_c[c] for c in rng.choice(uniq, size=len(uniq),
                                             replace=True)])
    else:
        draw = lambda: rng.randint(0, n, size=n)
    acc = {k: [] for k in oof_p}
    dacc = {f"{a}_minus_{b}": [] for a, b in contrasts}
    for _ in range(n_bootstrap):
        rows = draw()
        yr = y[rows]
        if yr.sum() == 0 or yr.sum() == len(yr):
            continue
        auc = {k: _fast_auc(yr, v[rows]) for k, v in oof_p.items()}
        for k, a in auc.items():
            acc[k].append(a)
        for a, b in contrasts:
            dacc[f"{a}_minus_{b}"].append(auc[a] - auc[b])
    levels = {k: {"mean": round(float(np.mean(v)), 4),
                  "ci": [round(float(np.percentile(v, 2.5)), 4),
                         round(float(np.percentile(v, 97.5)), 4)]}
              for k, v in acc.items()}
    deltas = {}
    for k, v in dacc.items():
        v = np.array(v)
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        deltas[k] = {"mean": round(float(np.mean(v)), 4),
                     "bootstrap_ci": [round(lo, 4), round(hi, 4)],
                     "p_positive": round(float(np.mean(v > 0.0)), 3),
                     "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0)}
    return levels, deltas


def summarize(run, contrasts, tag, clusters):
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
    oof_p = {arm: run["oof"][arm][PRIMARY] for arm in run["oof"]}
    levels, deltas = bootstrap(oof_p, y, contrasts, clusters=clusters)
    for arm in levels:
        arm_summary[f"{arm}:{PRIMARY}"]["bootstrap_ci"] = levels[arm]["ci"]
    ladder = {}
    for a, b in contrasts:
        key = f"{a}_minus_{b}"
        sa = np.array(run["per_seed"][a][PRIMARY])
        sb = np.array(run["per_seed"][b][PRIMARY])
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
    return {"tag": tag, "arm_summary": arm_summary, "ladder": ladder}


def icc_binary(y, groups):
    """单因素随机效应 ICC（组均值法近似，小样本披露为近似）。"""
    df = pd.DataFrame({"y": y, "g": groups})
    g = df.groupby("g")["y"].agg(["mean", "count"])
    grand = df["y"].mean()
    k = g["count"].mean()
    msb = (g["count"] * (g["mean"] - grand) ** 2).sum() / (len(g) - 1)
    msw = df.merge(g, left_on="g", right_index=True).apply(
        lambda r: (r["y"] - r["mean"]) ** 2, axis=1).sum() / (len(df) - len(g))
    var_b = max((msb - msw) / k, 0.0)
    return round(float((var_b) / (var_b + msw + 1e-12)), 4)


def main():
    t0 = time.time()

    # ============ Muchuro（组感知） ============
    mu = load_muchuro()
    mu_y = mu["y"].to_numpy(dtype=int)
    mu_g = mu["group"].to_numpy()
    run = run_cv(mu, mu_y, ["exp", "host", "full"], MU_ARMS, groups=mu_g)
    muchuro = summarize(run, CONTRASTS, "muchuro_group_aware", clusters=mu_g)
    muchuro["n"] = int(len(mu))
    muchuro["n_events"] = int(mu_y.sum())
    muchuro["n_households"] = int(pd.unique(mu_g).size)
    muchuro["icc_igra_within_household"] = icc_binary(mu_y, mu_g)
    # 剂量响应（勘察已看，归档留档）
    mu2 = mu.copy()
    mu2["prox_label"] = pd.read_excel(MU_XLSX)["prox2index"]
    dr = {}
    for lvl, sub in mu2.groupby("prox_label"):
        dr[str(lvl)[:40]] = {"n": int(len(sub)),
                             "igra_pos_rate": round(float(sub["y"].mean()), 4)}
    muchuro["dose_response_prox"] = dr
    tw_lbl = pd.read_excel(MU_XLSX)["tw_tbindex"]
    dr2 = {}
    for lvl, sub in mu.assign(tw_label=tw_lbl.values).groupby("tw_label"):
        dr2[str(lvl)[:40]] = {"n": int(len(sub)),
                              "igra_pos_rate": round(float(sub["y"].mean()), 4)}
    muchuro["dose_response_tw"] = dr2

    # ============ Mayito（个体级） ============
    my = load_mayito()
    my_y = my["y"].to_numpy(dtype=int)
    run2 = run_cv(my, my_y, ["exp", "host", "full"], MY_ARMS, groups=None)
    mayito = summarize(run2, CONTRASTS, "mayito_individual", clusters=None)
    mayito["n"] = int(len(my))
    mayito["n_events"] = int(my_y.sum())
    my2 = my.copy()
    my2["close_lbl"] = pd.read_excel(MY_XLSX).loc[
        my2.index, "closeness"]
    dr3 = {}
    for lvl, sub in my2.groupby("closeness"):
        dr3[str(lvl)] = {"n": int(len(sub)),
                         "qft_pos_rate": round(float(sub["y"].mean()), 4)}
    mayito["dose_response_closeness"] = dr3
    dr4 = {}
    for lvl, sub in my2.groupby("bacilliload"):
        dr4[str(lvl)] = {"n": int(len(sub)),
                         "qft_pos_rate": round(float(sub["y"].mean()), 4)}
    mayito["dose_response_bacilliload"] = dr4

    # ============ 跨队列方向读数 ============
    mu_l = muchuro["ladder"]
    my_l = mayito["ladder"]
    cross = {
        "U1_full_minus_host": {
            "muchuro": mu_l["full_minus_host"],
            "mayito": my_l["full_minus_host"],
            "direction_agree": bool(np.sign(mu_l["full_minus_host"]["mean"]) ==
                                    np.sign(my_l["full_minus_host"]["mean"])),
        },
        "U2_exp_arm_muchuro": muchuro["arm_summary"][f"exp:{PRIMARY}"],
        "exp_minus_host": {
            "muchuro": mu_l["exp_minus_host"],
            "mayito": my_l["exp_minus_host"],
            "direction_agree": bool(np.sign(mu_l["exp_minus_host"]["mean"]) ==
                                    np.sign(my_l["exp_minus_host"]["mean"])),
        },
        "readme": ("U1/U2 见脚本 docstring；剂量响应粗率勘察先于声明查看"
                   "（方向已知），判别层臂间对比为主要新增检验"),
    }

    out = {
        "date": "2026-09-06",
        "name": "uga_dual_exposure_gradient_v1",
        "status": ("探索性方向分析（预声明）：个体级暴露梯度的感染排序"
                   "信号——§8.7 户级常量反转腿的设计对照；§8.6 暴露腿的"
                   "强度维度检验（时序维度仍归 ERASE-TB）"),
        "design": {
            "sources": {
                "muchuro": ("PLOS GPH 10.1371/journal.pgph.0000197 S1 "
                            "（n=352/59 户，IGRA 阳 32.7%）；PLOS S1 通道"
                            "（定向审计 VPN 复测轮）"),
                "mayito": ("Dryad doi:10.5061/dryad.k3j9kd5bg（Mayito 2023 "
                           "PLOS ONE，n=202→192 排除 10 indeterminate，"
                           "QFT-Plus 阳 54.7%）；浏览器通道"),
            },
            "cv": ("Muchuro：StratifiedGroupKFold(5) by tbindex_id × 20 种子"
                   "（组感知，本线程首个）+ 户级 cluster bootstrap（59 簇）；"
                   "Mayito：StratifiedKFold(5) × 20 种子 + 个体 bootstrap"),
            "models": "RF（主）+ LR（敏感性），冻结注册表（_make_model）",
            "missing_data": "折内中位数填补（Mayito 9 编码→缺失）",
            "exclusions": ("Mayito IGRA 定量列 tb1_ag/tb2_ag（终点决定性泄漏，"
                           "§8.6 教训）+ treatment（时点不明）；Muchuro "
                           "igra_exp/pickone_idx（调查管理列）"),
            "predictions": ["U1 full−host > 0 两队列", "U2 Muchuro exp CI 下界>0.5"],
        },
        "muchuro": muchuro,
        "mayito": mayito,
        "cross_cohort": cross,
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("===== Muchuro（组感知，n=%d，ev=%d，%d 户，ICC=%.3f）=====" %
          (muchuro["n"], muchuro["n_events"], muchuro["n_households"],
           muchuro["icc_igra_within_household"]))
    for arm in ("exp", "host", "full"):
        s = muchuro["arm_summary"][f"{arm}:{PRIMARY}"]
        print("  %-6s %.4f CI%s" % (arm, s["pooled_auroc"],
                                    s.get("bootstrap_ci", [])))
    for k, v in muchuro["ladder"].items():
        star = "*" if v.get("ci_excludes_zero") else " "
        print("  %s %-16s Δ=%+.4f CI=%s seed_share=%.2f DeLong p=%.4f" % (
            star, k, v["mean"], v["bootstrap_ci"],
            v["per_seed_delta"]["share_positive"],
            v["delong"].get("p_value", -1)))
    print()
    print("===== Mayito（个体，n=%d，ev=%d）=====" %
          (mayito["n"], mayito["n_events"]))
    for arm in ("exp", "host", "full"):
        s = mayito["arm_summary"][f"{arm}:{PRIMARY}"]
        print("  %-6s %.4f CI%s" % (arm, s["pooled_auroc"],
                                    s.get("bootstrap_ci", [])))
    for k, v in mayito["ladder"].items():
        star = "*" if v.get("ci_excludes_zero") else " "
        print("  %s %-16s Δ=%+.4f CI=%s seed_share=%.2f DeLong p=%.4f" % (
            star, k, v["mean"], v["bootstrap_ci"],
            v["per_seed_delta"]["share_positive"],
            v["delong"].get("p_value", -1)))
    print()
    print("===== 跨队列 U1/U2 =====")
    print("  U1 full−host: Muchuro %+.4f %s | Mayito %+.4f %s | 方向一致 %s" % (
        mu_l["full_minus_host"]["mean"], mu_l["full_minus_host"]["bootstrap_ci"],
        my_l["full_minus_host"]["mean"], my_l["full_minus_host"]["bootstrap_ci"],
        cross["U1_full_minus_host"]["direction_agree"]))
    print("  U2 Muchuro exp: %.4f CI=%s（下界>0.5: %s）" % (
        muchuro["arm_summary"][f"exp:{PRIMARY}"]["pooled_auroc"],
        muchuro["arm_summary"][f"exp:{PRIMARY}"]["bootstrap_ci"],
        muchuro["arm_summary"][f"exp:{PRIMARY}"]["bootstrap_ci"][0] > 0.5))
    print("\n归档 → %s" % OUT)
    print("耗时 %.1f 分钟" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
