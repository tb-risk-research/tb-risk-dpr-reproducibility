#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""肯尼亚 2016 全国结核病患病率调查 → ML 训练数据集

数据来源（免注册公开下载）：
  Kenya National Tuberculosis Prevalence Survey 2016
 民主共和国？不——肯尼亚共和国卫生部调查，发表：
  Enos Masini et al., PLOS ONE 2019, doi:10.1371/journal.pone.0209098
  S1（个体级 CSV，126,389 筛查对象 × 269 变量）+ S2（数据字典）

标签定义（患病率调查标准）：
  tb_outcome = 涂片 / Xpert MTB-RIF / 培养任一细菌学阳性（POS）
  未送检或阴性 → 0（保守假设：未检出 = 非病例；误分类朝向稀释效应，
  高估的 AUROC 会被该稀释部分抵消，结论保守可信）

特征映射（项目 13 基础特征语义对齐）：
  age                ← Age
  has_symptoms       ← 9 症状任一（咳嗽/咳痰/咯血/胸痛/发热/乏力/
                        消瘦/盗汗/气短）
  has_tb             ← TBEverBeenTreated 或 AntiTBTreatmentBefore
  past_illness       ← HIVStattus==1（HIV 感染）
  bcg_vaccine        ← 1.0（肯尼亚新生儿 BCG 普种，WHO 覆盖率 >99%）
  is_high_risk       ← HIV 阳性 或 年龄 ≥65
  7 个接触暴露特征    ← 0.0（横断面调查无接触者追踪暴露信息，
                        常数列树模型自动忽略；语义诚实：该数据集
                        提供人群/症状/合并症信号，不提供暴露信号）

v4 extras 源列（P6，2026-09-15 原始数据再审——回 S01 269 变量扫未纳入列）：
  gender             ← Sex（M/F）小写化（男性 TB 患病率显著偏高，
                        单变量 AUROC 0.596——原 ETL 唯一重大遗漏）
  cxr_tb_suspect     ← xfindingall==2（终判读 AbNormal Suggestive of
                        TB；clinician1+2+国家放射学家一致判读，
                        分析集覆盖 99.1%——按 v4 家族规则影像为合法
                        协变量，taiwan cxr_score / crp cad_score 先例）
  cxr_abnormal_other ← xfindingall==3（AbNormal 非 TB 提示）
  cxr_missing        ← xfindingall 缺失指示器
  symptom_*（7 列）   ← BloodCough/Sputum/ChestPains/Fever/Fatigue/
                        NightSweats/BreatheShortness 个别症状分辨率
                       （原 ETL 坍缩为 has_symptoms 一个二值；
                        breathless 问卷流条件缺失 26.6% → 指示器）
  cough_weeks        ← CoughingWeeksNumber（咳嗽时长梯度，禁二值化）
  treatment_sought   ← TreatmentSought（1=就诊 2=未就诊 0=未问→NaN）
  hiv_tested         ← HIVStattus 非空（past_illness 把"未检"与
                        "阴性"同为 0，此列消歧选择通道）
泄漏审计参照列（进 CSV 仅供排除登记表计算单变量 AUROC，
  绝不进任何 spec）：
  lab_smear_pos      ← Smearpositive POS（终点定义性）
  lab_sputum_requested ← SputumRequestDone（送检级联嵌入——
                        单变量 AUROC 0.915，巴基斯坦同构证据）
  tb_current_treatment ← TBCurrentTreatment（现症治疗与终点同源）

P6 排除登记（原始列不进 CSV 的完整理由见 kenya_dataset_meta.json）：
  细菌学结果族（Smear/MTB/RIF/Culture/_2/smur 组合/TempResult/
  AllLabResult/Smearpositive/Xpertpositive/Culturepositive/
  TBSuspect*）——终点定义；痰标本级联族（SputumRequestDone/
  First/SecondSputumCollected/SputumCollected_1/2/Container*/
  req_*/sample_*/volume/quality/transit_time/status 等）——
  送检由症状+CXR 决定（诊断流程嵌入特征，巴基斯坦教训同构）；
  现症治疗族（TBCurrentTreatment/CurrentTBXrayExaminations/
  CurrentTBSputumExaminations/CurrentTreatmentFromDate）——
  与患病终点同源；弱信号列（Schooling 0.482/LungDiseaseSymptoms
  0.525/SymptomsPast 0.521/XRayExamPast 0.506/SputumExamPast
  0.519/MaritalStatus/TimeNeededToNearestHealthCenter）——
  单变量 |AUROC-0.5|<0.03 不入 spec。

用法：
    python data/process_kenya_data.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw", "kenya_prevalence")
PROC = os.path.join(HERE, "processed")

SYMPTOM_COLS = [
    "Coughing", "Sputum", "BloodCough", "ChestPains", "Fever",
    "Fatigue", "WeightLoss", "NightSweats", "BreatheShortness",
]


def build_kenya_ml_dataset():
    src = os.path.join(RAW, "S01.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(f"缺少原始数据: {src}（先运行下载步骤）")

    df = pd.read_csv(src, low_memory=False)
    n_screened = len(df)

    # 分析集：完成个体问卷者（症状字段非空 = 15+ 岁问卷对象）
    df = df[df["Coughing"].notna()].copy()
    n_analyzed = len(df)

    # 标签：任一细菌学阳性（CON = 培养污染，不算阳性）
    def is_pos(series):
        return (series.astype(str).str.strip().str.upper() == "POS")

    tb_outcome = (
        is_pos(df["Smearpositive"].fillna(""))
        | is_pos(df["Xpertpositive"].fillna(""))
        | is_pos(df["Culturepositive"].fillna(""))
    ).astype(int)
    n_pos = int(tb_outcome.sum())

    # 特征映射
    out = pd.DataFrame()
    out["age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0)
    sym = pd.DataFrame({
        c: pd.to_numeric(df[c], errors="coerce").fillna(0) for c in SYMPTOM_COLS
    })
    out["has_symptoms"] = (sym.max(axis=1) > 0).astype(float)
    out["has_tb"] = (
        (pd.to_numeric(df["TBEverBeenTreated"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(df["AntiTBTreatmentBefore"], errors="coerce").fillna(0) > 0)
    ).astype(float)
    hiv = pd.to_numeric(df["HIVStattus"], errors="coerce").fillna(0)
    out["past_illness"] = (hiv > 0).astype(float)
    out["bcg_vaccine"] = 1.0  # 肯尼亚新生儿 BCG 普种（WHO >99%）
    out["is_high_risk"] = ((hiv > 0) | (out["age"] >= 65)).astype(float)
    # 接触暴露特征：横断面调查无对应信息（诚实置 0，非缺失填充）
    for col in ("cumulative_exposure", "contact_distance_score",
                "ventilation_score", "exposure_setting_score",
                "single_duration", "freq_density", "time_span"):
        out[col] = 0.0

    # ---- v4 extras 源列（P6 原始数据再审）----
    # 性别：男性患病率显著偏高（单变量 AUROC 0.596）——原 ETL 遗漏
    out["gender"] = df["Sex"].astype(str).str.strip().str.lower()
    # CXR 终判读（clinician1+2+国家放射学家一致；分析集覆盖 99.1%）：
    # 1=Normal / 2=AbNormal Suggestive of TB / 3=AbNormal Not suggestive
    xf = pd.to_numeric(df["xfindingall"], errors="coerce")
    out["cxr_tb_suspect"] = (xf == 2).astype(float).where(xf.notna())
    out["cxr_abnormal_other"] = (xf == 3).astype(float).where(xf.notna())
    out["cxr_missing"] = xf.isna().astype(float)
    # 个别症状分辨率（原 ETL 坍缩为 has_symptoms 一个二值）
    for out_col, src in (
            ("symptom_bloodcough", "BloodCough"),
            ("symptom_sputum", "Sputum"),
            ("symptom_chestpains", "ChestPains"),
            ("symptom_fever", "Fever"),
            ("symptom_fatigue", "Fatigue"),
            ("symptom_nightsweats", "NightSweats"),
            ("symptom_breathless", "BreatheShortness")):
        out[out_col] = pd.to_numeric(df[src], errors="coerce")
    # 咳嗽时长梯度（0=无长咳；1-35 周梯度，禁二值化）
    out["cough_weeks"] = pd.to_numeric(
        df["CoughingWeeksNumber"], errors="coerce")
    # 症状就诊行为（1=就诊 / 2=未就诊 / 0=无症状未问→NaN）
    ts = pd.to_numeric(df["TreatmentSought"], errors="coerce")
    out["treatment_sought"] = ts.map({1.0: 1.0, 2.0: 0.0})
    # HIV 检测选择通道（past_illness 把"未检"与"阴性"同为 0，此列消歧）
    out["hiv_tested"] = df["HIVStattus"].notna().astype(float)

    # ---- 泄漏审计参照列（仅排除登记表用，绝不进 spec）----
    out["lab_smear_pos"] = is_pos(
        df["Smearpositive"].fillna("")).astype(float).where(
            df["Smearpositive"].notna())
    out["lab_sputum_requested"] = pd.to_numeric(
        df["SputumRequestDone"], errors="coerce").fillna(0.0)
    out["tb_current_treatment"] = pd.to_numeric(
        df["TBCurrentTreatment"], errors="coerce")

    out["tb_outcome"] = tb_outcome.values

    os.makedirs(PROC, exist_ok=True)
    csv_path = os.path.join(PROC, "kenya_ml_training.csv")
    out.to_csv(csv_path, index=False)

    meta = {
        "source": "Kenya National TB Prevalence Survey 2016",
        "reference": "doi:10.1371/journal.pone.0209098 (PLOS ONE 2019)",
        "access": "public, no registration (PLOS Supporting Information S1/S2)",
        "n_screened": n_screened,
        "n_analyzed": n_analyzed,
        "n_positive": n_pos,
        "positive_rate": round(n_pos / n_analyzed, 6),
        "label": "tb_outcome = smear/Xpert/culture any POS (bacteriologically confirmed TB)",
        "feature_map": {
            "age": "Age",
            "has_symptoms": "any of 9 symptoms",
            "has_tb": "TBEverBeenTreated | AntiTBTreatmentBefore",
            "past_illness": "HIVStattus==1",
            "bcg_vaccine": "constant 1.0 (universal neonatal BCG, WHO >99%)",
            "is_high_risk": "HIV+ or age>=65",
            "exposure_features": "constant 0.0 (cross-sectional survey has no contact-tracing exposure data)",
            "gender": "Sex (lowercased m/f; univariate AUROC 0.596)",
            "cxr_tb_suspect": "xfindingall==2 (final CXR read suggestive of TB; 99.1% coverage)",
            "cxr_abnormal_other": "xfindingall==3 (abnormal, not suggestive of TB)",
            "cxr_missing": "xfindingall missing indicator",
            "symptom_bloodcough": "BloodCough",
            "symptom_sputum": "Sputum",
            "symptom_chestpains": "ChestPains",
            "symptom_fever": "Fever",
            "symptom_fatigue": "Fatigue",
            "symptom_nightsweats": "NightSweats",
            "symptom_breathless": "BreatheShortness (26.6% conditional missing -> indicator)",
            "cough_weeks": "CoughingWeeksNumber (gradient, not dichotomized)",
            "treatment_sought": "TreatmentSought (1=sought, 2=not, 0=not asked -> NaN)",
            "hiv_tested": "HIVStattus notna (disambiguates untested vs negative)",
        },
        "leak_audit_reference_columns": {
            "lab_smear_pos": "Smearpositive POS (endpoint-defining; disclosure only)",
            "lab_sputum_requested": "SputumRequestDone (cascade-embedded, univariate AUROC 0.915; disclosure only)",
            "tb_current_treatment": "TBCurrentTreatment (concurrent with prevalent TB; disclosure only)",
        },
        "p6_exclusions": {
            "bacteriology_family": "Smear/MTB/RIF/Culture/Smearpositive/Xpertpositive/Culturepositive/TBSuspect* (endpoint-defining)",
            "sputum_cascade_family": "SputumRequestDone/First/SecondSputumCollected/Container*/req_*/sample_*/volume/quality (diagnostic cascade embedded in features)",
            "current_treatment_family": "TBCurrentTreatment/CurrentTBXrayExaminations/CurrentTBSputumExaminations/CurrentTreatmentFromDate (same-source as prevalence endpoint)",
            "weak_signal_columns": "Schooling 0.482 / LungDiseaseSymptoms 0.525 / SymptomsPast 0.521 / XRayExamPast 0.506 / SputumExamPast 0.519 / MaritalStatus / TimeNeededToNearestHealthCenter (|AUROC-0.5|<0.03, not in spec)",
        },
        "disclosure": "undetected cases classified negative (dilution bias, conservative)",
    }
    meta_path = os.path.join(PROC, "kenya_dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"分析集: {n_analyzed} / {n_screened}（完成个体问卷者）")
    print(f"细菌学确诊 TB: {n_pos} ({n_pos/n_analyzed:.3%})")
    print(f"输出: {csv_path}")
    print(f"元数据: {meta_path}")
    return csv_path


if __name__ == "__main__":
    build_kenya_ml_dataset()
