#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CRP 域 schema 扩展侧臂实验（P1-a 决策备忘录证据腿，2026-09-05）

问题：22 维部署 schema 是否应纳入 CRP / 胸片（CAD）特征？
证据腿：在 CRP 域（LSHTM 社区筛查开放版验证子集，与克拉玛依分诊
场景同构）上做受控侧臂——22 维基线 vs 22 维+CRP vs 22 维+CAD，
量化"schema 扩展后可显著提升"（crp_dataset_meta.json 预告）的具体幅度。

方法学要点：
- 完整复用 train_from_real_data 管线（map_columns→交互特征→六模型
  注册表→5 折 CV，seed 42），仅实例级扩展 ALL_FEATURE_NAMES 追加
  侧臂列——与 2026-09-05 归档 run（AUROC 0.622，XGBoost）完全同径，
  A0 基线臂即复现性检查；
- availability 缺失指示器（§6.5 分诊分支模态纪律："一律带"）：
  crp_missing/cad_missing 随连续值入模；另设"仅缺失指示器"控制臂，
  分解增益来自检验值本身还是缺失模式（验证选择通道）；
- 部署形态臂：crp_positive 二值（LISPACSAdapter.extract_risk_features_
  from_labs 实际交付 crp_elevated 二值，非连续值）。

已知偏倚（引用侧臂数字时必须随行）：
1. 验证规则循环：CAD≥40 是进入验证子集的资格规则之一（sputum_elig
   = 症状或 CAD≥40），+CAD 臂在验证子集内评估存在"特征参与样本
   选择"的部分循环——A2/A3 的 Δ 只作上限解读，不作部署预期；
2. 任务定义：本实验为"分诊确认"任务（已送检人群中识别确诊），
   非接触者基线筛查任务；
3. CRP 缺失 26.2%（466/1777）且缺失组阳性率 8.8% vs 观测组 5.8%
   （CRP 开单本身有选择）——缺失指示器控制臂因此必要。

用法：
    python data/run_crp_schema_expansion_arms.py
输出：
    data/processed/crp_schema_expansion_arms_20260905.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DESKTOP = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DESKTOP)

PROC = os.path.join(HERE, "processed")
SRC = os.path.join(PROC, "ml_training_crp_real.csv")
OUT = os.path.join(PROC, "crp_schema_expansion_arms_20260905.json")

import pandas as pd  # noqa: E402

from tb_risk.scoring.predictor import MLRiskPredictor  # noqa: E402
from tb_risk.scoring.ml.training import train_from_real_data  # noqa: E402

# 侧臂定义：arm 名 → 追加列（instance 级 ALL_FEATURE_NAMES 扩展）
ARMS = {
    "A0_baseline_22": [],
    "A1_crp_continuous": ["crp_mgdl", "crp_missing"],
    "A1b_crp_binary_deploy": ["crp_positive", "crp_missing"],
    "A1c_crp_missing_only": ["crp_missing"],
    "A2_cad": ["cad_score", "cad_missing"],
    "A2c_cad_missing_only": ["cad_missing"],
    "A3_crp_and_cad": ["crp_mgdl", "crp_missing", "cad_score", "cad_missing"],
    "A3b_binary_deploy_full": ["crp_positive", "crp_missing",
                               "cad_score", "cad_missing"],
}


def main():
    df = pd.read_csv(SRC)
    df["crp_missing"] = df["crp_mgdl"].isna().astype(int)
    df["cad_missing"] = df["cad_score"].isna().astype(int)
    n_pos = int(df["tb_outcome"].sum())

    base_names = list(MLRiskPredictor.ALL_FEATURE_NAMES)
    results = {
        "date": "2026-09-05",
        "question": ("22 维部署 schema 是否纳入 CRP/CAD？"
                     "（P1-a CRP/Karamay schema 扩展决策备忘录证据腿）"),
        "dataset": {
            "source": "ml_training_crp_real.csv",
            "n": len(df), "n_positive": n_pos,
            "task": "分诊确认（验证子集内识别确诊），非接触者基线筛查",
            "crp_missing": int(df["crp_mgdl"].isna().sum()),
            "cad_missing": int(df["cad_score"].isna().sum()),
            "crp_missing_positive_rate": round(
                float(df.loc[df["crp_mgdl"].isna(), "tb_outcome"].mean()), 4),
            "crp_observed_positive_rate": round(
                float(df.loc[df["crp_mgdl"].notna(), "tb_outcome"].mean()), 4),
        },
        "protocol": ("train_from_real_data 全管线复用（map_columns→交互"
                     "特征→六模型注册表→5 折 CV，seed 42）；"
                     "instance 级 ALL_FEATURE_NAMES 扩展追加侧臂列；"
                     "A0 为 2026-09-05 归档 run 0.622 的复现性检查"),
        "known_biases": [
            "验证规则循环：CAD≥40 是验证资格规则（sputum_elig=症状或"
            "CAD≥40），A2/A3/A3b 的 +CAD 增幅只作上限解读",
            "CRP 缺失 26.2% 且开具有选择（缺失组阳性率 8.8% vs 5.8%），"
            "A1c 控制臂分解该通道",
            "部署适配器（LISPACSAdapter）交付二值 crp_elevated，"
            "A1b/A3b 为部署形态臂",
        ],
        "arms": {},
    }

    for arm, extra in ARMS.items():
        tmp = os.path.join(PROC, f"_crp_arm_{arm}.csv")
        df.to_csv(tmp, index=False)
        try:
            p = MLRiskPredictor()
            p.ALL_FEATURE_NAMES = base_names + list(extra)
            res = train_from_real_data(p, tmp)
            per_model = {
                k: {"AUROC": round(v["AUROC"], 4),
                    "AUPRC": round(v["AUPRC"], 4)}
                for k, v in p.model_performance.items()
            }
            best_key, best = max(per_model.items(),
                                 key=lambda kv: kv[1]["AUROC"])
            results["arms"][arm] = {
                "extra_columns": extra,
                "n_features": len(base_names) + len(extra),
                "success": bool(res["success"]),
                "per_model": per_model,
                "best_model": best_key,
                "best_AUROC": best["AUROC"],
                "best_AUPRC": best["AUPRC"],
            }
            print(f"[{arm}] best={best_key} AUROC={best['AUROC']:.4f} "
                  f"AUPRC={best['AUPRC']:.4f}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # 与基线的差值（引用口径：最优基模型 vs 基线最优基模型）
    base_best = results["arms"]["A0_baseline_22"]["best_AUROC"]
    for arm, r in results["arms"].items():
        r["delta_vs_A0"] = round(r["best_AUROC"] - base_best, 4)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] {OUT}")


if __name__ == "__main__":
    main()
