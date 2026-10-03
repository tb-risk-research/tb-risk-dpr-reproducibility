#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开数据处理器：raw/ → processed/（三个模型模块的直接输入）

产出：
1. processed/seir_observed_china_2000_2024.csv
   WHO GHO 中国时序（发病/患病/死亡/RR-MDR/通报/治疗成功率）+ 世行人口
   → SEIR 拟合与贝叶斯标定的观测数据
2. processed/china_contact_matrices.npz + contact_mixing_summary.json
   Prem et al. (2017) 中国接触矩阵（家庭/学校/工作/社区, 85 年龄组）
   → GNN 家庭/社会两层网络的接触强度校准（BETA_FAMILY/BETA_SOCIAL 相对比）
3. processed/ml_training_semi_synthetic_anchored.csv (+ anchors.json)
   个体级半合成队列：**重要披露** —— 公开个体级结核临床数据（TB Portals 等）
   均需注册授权（实测 401），本数据集为"公开参数锚定"的半合成队列：
   - 年龄/性别结构 ← WHO GHO 中国 2021 按年龄性别通报分布
   - 基线 2 年进展率 3.1% ← Fox et al. Lancet Infect Dis 2013（家庭接触者）
   - 免疫抑制乘数 ← 项目 constants.IMMUNO_FACTORS_BASE（文献溯源见该文件）
   - BCG 覆盖率 99% ← WHO 中国估计
   - 糖尿病患病率 ~12%（成人）← 公开流行病学估计
   标签列 tb_outcome；13 基础特征 + 9 交互特征（公式与
   scoring/ml/training.py::train_from_real_data 完全一致）

用法：
    python data/process_public_data.py
"""
import csv
import datetime
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
PROC = os.path.join(HERE, "processed")


def read_gho(name):
    path = os.path.join(RAW, "who_gho", name)
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["TimeDim"])] = float(row["NumericValue"])
    return out


# ---------------------------------------------------------------------------
# 1. SEIR 观测时序
# ---------------------------------------------------------------------------
def build_seir_observed():
    inc = read_gho("TB_e_inc_num_CHN.csv")
    prev = read_gho("TB_e_prev_num_CHN.csv")
    mort = read_gho("TB_e_mort_exc_tbhiv_num_CHN.csv")
    rr = read_gho("TB_e_inc_rr_num_CHN.csv")
    hiv = read_gho("TB_e_inc_tbhiv_100k_CHN.csv")
    tsr = read_gho("TB_c_new_tsr_CHN.csv")

    pop = {}
    with open(os.path.join(RAW, "worldbank", "CHN_SP_POP_TOTL.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["value"]:
                pop[int(row["year"])] = float(row["value"])

    years = sorted(set(inc) | set(prev) | set(mort))
    out_path = os.path.join(PROC, "seir_observed_china_2000_2024.csv")
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "est_incident_cases", "est_prevalent_cases",
                    "est_deaths_excl_hiv", "est_rr_mdr_incident",
                    "hiv_pos_incidence_per_100k", "treatment_success_rate",
                    "population", "incidence_per_100k"])
        for y in years:
            if y < 2000:
                continue
            p = pop.get(y, pop.get(2023))
            rate = inc.get(y, inc.get(y)) / p * 1e5 if (inc.get(y) and p) else ""
            w.writerow([y, inc.get(y, ""), prev.get(y, ""), mort.get(y, ""),
                        rr.get(y, ""), hiv.get(y, ""), tsr.get(y, ""), p, rate])
            n += 1
    print(f"[OK] seir_observed_china_2000_2024.csv ({n} 年)")
    return out_path


# ---------------------------------------------------------------------------
# 2. 中国接触矩阵（GNN 家庭/社会分层校准）
# ---------------------------------------------------------------------------
# Prem/Mistry 接触矩阵口径（mobs-lab/mixing-patterns README）：
# - 分场景 F^k_ij 为"人均接触概率"，需乘以该场景的平均有效接触数权重
#   （household 4.11 / school 11.41 / work 8.07 / community 2.79）
# - 总矩阵 M_ij 为加权线性和，即人均接触数
# - 文献：Mistry et al. Nat Commun 12:323 (2021)；Prem et al. PLOS Comput Biol (2017)
SETTING_WEIGHTS = {"household": 4.11, "school": 11.41, "work": 8.07, "community": 2.79}


def build_contact_matrices():
    settings = ["household", "school", "work", "community"]
    arrays = {}
    summary = {"source": "Prem et al. 2017 / Mistry et al. 2021 (mobs-lab/mixing-patterns)",
               "country": "China", "age_groups": 85, "license": "CC BY 4.0",
               "setting_weights": SETTING_WEIGHTS,
               "note": "分场景矩阵为接触概率，按官方权重换算为接触数；文件无表头"}

    def load(fname):
        # 该库 CSV 无表头行（85x85 纯数值）
        return np.loadtxt(os.path.join(RAW, "contact_matrices", fname),
                          delimiter=",")

    overall = load("China_country_level_M_overall_contact_matrix_85.csv")
    assert overall.shape == (85, 85), f"总矩阵形状异常: {overall.shape}"

    setting_contacts = {}
    for s in settings:
        m = load(f"China_country_level_F_{s}_setting_85.csv")
        assert m.shape == (85, 85), f"{s} 矩阵形状异常: {m.shape}"
        arrays[s] = m
        setting_contacts[s] = m.sum(axis=1) * SETTING_WEIGHTS[s]

    overall_mean_contacts = overall.sum(axis=1)
    household = setting_contacts["household"]
    social = sum(setting_contacts[s] for s in ["school", "work", "community"])

    summary["avg_contacts_overall"] = float(overall_mean_contacts.mean())
    summary["avg_contacts_by_setting"] = {
        s: float(setting_contacts[s].mean()) for s in settings}
    summary["avg_contacts_household"] = float(household.mean())
    summary["avg_contacts_social"] = float(social.mean())
    summary["family_share_of_contacts"] = float(
        household.mean() / overall_mean_contacts.mean())
    summary["social_share_of_contacts"] = float(
        social.mean() / overall_mean_contacts.mean())
    # 分年龄段（5 岁一组，共 17 组：0-4 ... 80+）
    bands = {}
    for start in range(0, 85, 5):
        end = min(start + 5, 85)
        bands[f"{start}-{end - 1}"] = {
            "overall": float(overall_mean_contacts[start:end].mean()),
            "household": float(household[start:end].mean()),
            "social": float(social[start:end].mean()),
        }
    summary["by_age_band"] = bands
    summary["gnn_calibration_hint"] = {
        "BETA_FAMILY_vs_BETA_SOCIAL": (
            f"家庭层接触占比 {summary['family_share_of_contacts']:.2f}，"
            f"社会层 {summary['social_share_of_contacts']:.2f}；"
            "GNN 合成网络的家庭/社会边密度建议按此比例校准"),
    }

    npz_path = os.path.join(PROC, "china_contact_matrices.npz")
    np.savez_compressed(npz_path,
                        overall=overall,
                        **{f"{s}": arrays[s] for s in settings})
    json_path = os.path.join(PROC, "contact_mixing_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] china_contact_matrices.npz + contact_mixing_summary.json "
          f"(家庭占比 {summary['family_share_of_contacts']:.2f}, "
          f"家庭 {summary['avg_contacts_household']:.1f} / "
          f"社会 {summary['avg_contacts_social']:.1f} 接触/人)")
    return npz_path


# ---------------------------------------------------------------------------
# 3. ML 半合成锚定队列
# ---------------------------------------------------------------------------
ANCHORS = {
    "household_contact_2y_tb": {
        "value": 0.031,
        "source": "Fox et al. Lancet Infect Dis 2013（家庭接触者结核累计发病率 Meta）"},
    "immuno_factors": {
        "value": {"hiv": 8.0, "immunosuppressants": 4.0, "diabetes": 2.5, "other": 1.8},
        "source": "项目 constants.IMMUNO_FACTORS_BASE（文献溯源见 constants.py）"},
    "bcg_coverage": {"value": 0.99, "source": "WHO 中国 BCG 覆盖率估计"},
    "diabetes_prevalence_adult": {"value": 0.12, "source": "中国成人糖尿病患病率公开估计"},
    "hiv_prevalence": {"value": 0.001, "source": "WHO GHO 中国 HIV 阳性结核发病率极低"},
    "retreatment_share": {"value": 0.045, "source": "WHO GHO 中国复治占比估计"},
    "age_sex_structure": {
        "value": "WHO GHO TB_notif_num_agesex CHN 2021",
        "source": "data/raw/who_gho/TB_notif_num_agesex_CHN.csv"},
    "smoking_progression": {
        "value": 1.2, "source": "constants.SMOKING_PROGRESSION_FACTOR"},
    "bmi_factors": {"value": "constants.BMI_PROGRESSION_FACTOR", "source": "constants.py"},
}


def _age_sex_weights():
    """从 GHO 2021 年龄性别通报分布构造抽样权重（带年龄段中点年龄）"""
    path = os.path.join(RAW, "who_gho", "TB_notif_num_agesex_CHN.csv")
    bands = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["TimeDim"] != "2021" or row["Dim1"] in ("SEX_UNK", ""):
                continue
            age = row["Dim2"].replace("AGEGROUP_", "")
            try:
                val = float(row["NumericValue"])
            except ValueError:
                continue
            bands.setdefault(age, {"m": 0.0, "f": 0.0})
            if row["Dim1"] == "SEX_MLE":
                bands[age]["m"] += val
            elif row["Dim1"] == "SEX_FMLE":
                bands[age]["f"] += val
    age_map = {
        "YEARSUNDER5": 3, "YEARS05-14": 10, "YEARS15-24": 20,
        "YEARS25-34": 30, "YEARS35-44": 40, "YEARS45-54": 50,
        "YEARS55-64": 60, "YEARS65PLUS": 72,
    }
    weights, ages = [], []
    for band, cnt in bands.items():
        if band not in age_map:
            continue
        total = cnt["m"] + cnt["f"]
        if total <= 0:
            continue
        ages.append(age_map[band] + np.random.uniform(-4, 4))
        weights.append(total)
    w = np.array(weights, dtype=float)
    return ages, w / w.sum()


def build_ml_dataset(n_samples=3000, random_state=42):
    rng = np.random.RandomState(random_state)
    ages, age_w = _age_sex_weights()

    rows = []
    for i in range(n_samples):
        # --- 个体特征（公开锚定）---
        age = float(rng.choice(ages, p=age_w) + rng.uniform(-3, 3))
        age = float(np.clip(age, 1, 95))
        bcg = 1 if rng.random() < ANCHORS["bcg_coverage"]["value"] else 0
        has_tb = 1 if rng.random() < ANCHORS["retreatment_share"]["value"] else 0

        r = rng.random()
        if r < ANCHORS["hiv_prevalence"]["value"]:
            illness_type = "hiv"
        elif r < ANCHORS["hiv_prevalence"]["value"] + 0.005:
            illness_type = "immunosuppressants"
        elif r < ANCHORS["hiv_prevalence"]["value"] + 0.005 + ANCHORS["diabetes_prevalence_adult"]["value"]:
            illness_type = "diabetes"
        elif r < 0.18:
            illness_type = "other"
        else:
            illness_type = "none"
        past_illness = 0 if illness_type == "none" else 1
        is_high_risk = int(age >= 65 or past_illness or has_tb)

        # --- 接触/暴露特征（家庭接触者队列场景）---
        # 累积暴露剂量服从剂量-反应：长时/近距离/密闭 → 高剂量
        freq_density = float(np.clip(rng.gamma(2.0, 3.0), 1, 20))          # 每周接触频次
        single_duration = float(np.clip(rng.lognormal(4.6, 0.5), 10, 720))  # 单次分钟
        time_span = float(np.clip(rng.gamma(6.0, 4.0), 2, 52))              # 持续周
        cumulative_exposure = freq_density * (single_duration / 60.0) * time_span  # 小时
        contact_distance_score = float(np.clip(rng.beta(2, 2), 0, 1))       # 越高越近
        ventilation_score = float(np.clip(rng.beta(2, 3), 0, 1))            # 越高越差
        exposure_setting_score = float(np.clip(rng.beta(2, 2) * 0.6
                                               + contact_distance_score * 0.4, 0, 1))

        # 症状与就诊延迟
        symptom_prob = 0.15 + 0.35 * exposure_setting_score
        has_symptoms = int(rng.random() < symptom_prob)
        delay_days = float(np.clip(rng.lognormal(3.4, 0.8), 0, 180))
        cough_frequency = float(np.clip(rng.gamma(2, 8), 0, 60))
        contact_count = int(np.clip(rng.poisson(6), 1, 30))

        # --- 标签：公开锚定的剂量-反应 × 风险乘数 ---
        immuno = ANCHORS["immuno_factors"]["value"]
        mult = immuno.get(illness_type if illness_type != "none" else "other", 1.0) \
            if past_illness else 1.0
        mult *= 1.2 if (age >= 40 and rng.random() < 0.5) else 1.0          # 吸烟代理（中老年部分吸烟）
        mult *= 1.3 if age >= 65 else 1.0
        mult *= 1.5 if has_tb else 1.0
        # Wells-Riley 式暴露剂量反应：P = 1 - exp(-q·D)
        dose = (cumulative_exposure / 500.0) * (0.5 + contact_distance_score) \
            * (0.6 + ventilation_score) * (0.5 + exposure_setting_score)
        dose_reaction = 1.0 - np.exp(-0.9 * dose)
        p_tb = ANCHORS["household_contact_2y_tb"]["value"] * mult * (0.5 + dose_reaction)
        p_tb = float(np.clip(p_tb, 0.002, 0.55))
        tb_outcome = int(rng.random() < p_tb)
        # 症状与结局相关（病例更可能有症状）
        if tb_outcome and not has_symptoms and rng.random() < 0.6:
            has_symptoms = 1

        rows.append({
            "age": round(age, 1), "cumulative_exposure": round(cumulative_exposure, 1),
            "has_symptoms": has_symptoms, "bcg_vaccine": bcg, "has_tb": has_tb,
            "contact_distance_score": round(contact_distance_score, 3),
            "ventilation_score": round(ventilation_score, 3),
            "is_high_risk": is_high_risk, "past_illness": past_illness,
            "exposure_setting_score": round(exposure_setting_score, 3),
            "single_duration": round(single_duration, 1),
            "freq_density": round(freq_density, 2), "time_span": round(time_span, 1),
            # 交互特征（公式 = train_from_real_data 的自动计算）
            "past_illness_type": illness_type,
            "delay_days": delay_days, "cough_frequency": cough_frequency,
            "contact_count": contact_count,
            "tb_outcome": tb_outcome,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    # 交互特征（与 scoring/ml/training.py L945-977 完全一致的公式）
    hiv = (df["past_illness_type"] == "hiv").astype(float)
    diabetes = (df["past_illness_type"] == "diabetes").astype(float)
    imm = (df["past_illness_type"] == "immunosuppressants").astype(float)
    iss = hiv * 2.0 + diabetes * 1.0 + imm * 1.5 + df["past_illness"].astype(float) * 0.5
    df["age_immuno"] = df["age"] * iss
    df["dm_tb_synergy"] = diabetes * df["has_tb"]
    df["age_bcg_decay"] = df["age"] * (1 - df["bcg_vaccine"])
    df["symptom_delay"] = df["has_symptoms"] * np.minimum(df["delay_days"] / 30, 1.0)
    df["cough_contact"] = np.minimum(df["cough_frequency"] / 20, 1.0) * (df["contact_count"] / 10)
    df["highrisk_comorbid"] = df["is_high_risk"] * df["past_illness"]
    df["immune_bcg"] = (1 - df["bcg_vaccine"]) * iss
    df["exposure_accumulation"] = (df["cumulative_exposure"] / 80) * (df["time_span"] / 10)
    df["age_diabetes"] = df["age"] * diabetes

    out_path = os.path.join(PROC, "ml_training_semi_synthetic_anchored.csv")
    keep = ["age", "cumulative_exposure", "has_symptoms", "bcg_vaccine", "has_tb",
            "contact_distance_score", "ventilation_score", "is_high_risk",
            "past_illness", "exposure_setting_score", "single_duration",
            "freq_density", "time_span", "age_immuno", "dm_tb_synergy",
            "age_bcg_decay", "symptom_delay", "cough_contact", "highrisk_comorbid",
            "immune_bcg", "exposure_accumulation", "age_diabetes", "tb_outcome"]
    df[keep].to_csv(out_path, index=False)

    pos = int(df["tb_outcome"].sum())
    anchors_out = dict(ANCHORS)
    anchors_out["dataset"] = {
        "n_samples": n_samples, "n_positive": pos,
        "positive_rate": round(pos / n_samples, 4),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "disclosure": ("半合成队列：公开参数锚定，非真实个体数据；"
                       "TB Portals 个体数据注册后可用 data/download_public_datasets.py "
                       "--only tbportals 拉取并替换"),
        "random_state": random_state,
    }
    with open(os.path.join(PROC, "ml_training_anchors.json"), "w", encoding="utf-8") as f:
        json.dump(anchors_out, f, ensure_ascii=False, indent=2)
    print(f"[OK] ml_training_semi_synthetic_anchored.csv "
          f"({n_samples} 行, 阳性 {pos} = {pos / n_samples:.1%})")
    return out_path


def main():
    os.makedirs(PROC, exist_ok=True)
    build_seir_observed()
    build_contact_matrices()
    build_ml_dataset(n_samples=10000)
    build_pm25_dataset()
    build_nhanes_dataset()
    build_treats_dataset()
    build_crp_dataset()
    build_cape_town_contacts()
    # 2026-09-05 全球搜寻新增（Mendeley 台湾/巴西 + WHO TME + OWID 协变量）
    build_taiwan_dataset()
    build_brazil_igra_dataset()
    build_tb_covariates_timeseries()
    # 2026-09-05 P0-a 恢复新增（Dryad 秘鲁 MDR 家庭接触者前瞻队列）
    build_peru_mdr_dataset()


# ---------------------------------------------------------------------------
# 4. PM2.5 环境因子（OWID/State of Global Air，世行下线后的替代源）
# ---------------------------------------------------------------------------
def build_pm25_dataset():
    import csv as _csv

    src = os.path.join(RAW, "owid", "pm25-air-pollution.csv")
    if not os.path.exists(src):
        print("[跳过] pm25-air-pollution.csv 不存在")
        return
    china_rows, latest_by_country = [], {}
    # OWID 该数据集的值列名（人口加权 PM2.5 年均暴露）
    VAL_COL = "Outdoor air pollution exposure (population-weighted PM2.5)"
    with open(src, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            if not row.get("Year") or not row.get(VAL_COL):
                continue
            try:
                year = int(row["Year"])
                val = float(row[VAL_COL])
            except (TypeError, ValueError):
                continue
            entity = row["Entity"]
            if entity == "China":
                china_rows.append((year, val))
            code = row.get("Code", "")
            if code and (code not in latest_by_country or year > latest_by_country[code][0]):
                latest_by_country[code] = (year, val, entity)

    china_rows.sort()
    out_path = os.path.join(PROC, "pm25_china_timeseries.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["year", "pm25_mean_exposure_ugm3"])
        for y, v in china_rows:
            w.writerow([y, v])

    summary = {
        "source": "OWID grapher pm25-air-pollution（State of Global Air / HEI），CC BY 4.0",
        "worldbank_note": ("世行 EN.ATM.PM25.PC.MC.ZS 已下线"
                           "（data.worldbank.org 'no longer available'，2026-08 实测）"),
        "china_latest": dict(zip(["year", "pm25"], china_rows[-1])) if china_rows else None,
        "who_guideline_2021_annual": 5.0,
        "peers_latest": {
            code: {"year": y, "pm25": v, "entity": e}
            for code, (y, v, e) in latest_by_country.items()
            if code in ("CHN", "USA", "IND", "JPN", "KOR", "DEU", "World")},
    }
    with open(os.path.join(PROC, "pm25_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] pm25_china_timeseries.csv ({len(china_rows)} 年) "
          f"最新: {china_rows[-1] if china_rows else '无'}")


# ---------------------------------------------------------------------------
# 5. NHANES 2011-2012 真实个体级训练集（LTBI 结局）
# ---------------------------------------------------------------------------
def build_nhanes_dataset():
    """NHANES 2011-2012 → 真实个体级 LTBI 数据集

    标签: QuantiFERON Gold in-Tube (IGRA) 官方判定 LBXTBIN==1（潜伏结核感染）
    样本: 6 岁以上接受 IGRA 检测的参与者（约 7800 人，无注册公开数据）

    特征映射披露（诚实标注 NHANES 未采集的字段）：
    - age ← RIDAGEYR（真实）
    - past_illness/past_illness_type ← DIQ010 糖尿病 + MCQ220 癌症（真实）
    - is_high_risk ← 年龄≥65 或 past_illness（真实派生）
    - bcg_vaccine=0（美国常规不接种 BCG，全 0 无变异）
    - has_tb=0（社区筛查人群，无活动性 TB 诊断）
    - 接触暴露 7 列 =0（NHANES 未采集：cumulative_exposure/
      contact_distance_score/ventilation_score/exposure_setting_score/
      single_duration/freq_density/time_span）
    - 附加真实列（模型 22 特征之外，供后续扩展）：gender/bmi/smoking/
      born_us/income_pir/race
    交互特征由 train_from_real_data 自动计算。
    """
    import pandas as pd

    ndir = os.path.join(RAW, "nhanes")

    def rd(fn):
        return pd.read_sas(os.path.join(ndir, fn), format="xport")

    # TB_G = "Tuberculosis - Quantiferon_In_Gold"（官方文档标题，LBXTBIN 为
    # TB coded result）。注意：TST_G.xpt 实为血清睾酮（TeSTosterone）检测，
    # 与结核无关，不用于本数据集。
    igra = rd("TB_G.xpt")[["SEQN", "LBXTBA", "LBXTBN", "LBXTBIN", "LBXTBM"]]
    demo = rd("DEMO_G.xpt")[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1",
                             "DMDBORN4", "INDFMPIR"]]
    bmx = rd("BMX_G.xpt")[["SEQN", "BMXBMI"]]
    diq = rd("DIQ_G.xpt")[["SEQN", "DIQ010"]]
    smq = rd("SMQ_G.xpt")[["SEQN", "SMQ040"]]
    mcq = rd("MCQ_G.xpt")[["SEQN", "MCQ220"]]

    df = (igra.merge(demo, on="SEQN", how="left")
              .merge(bmx, on="SEQN", how="left")
              .merge(diq, on="SEQN", how="left")
              .merge(smq, on="SEQN", how="left")
              .merge(mcq, on="SEQN", how="left"))

    # 标签：IGRA 官方判定（1=阳性 2=阴性 3=不确定→剔除）
    n0 = len(df)
    df = df[df["LBXTBIN"].isin([1.0, 2.0])].copy()
    df["tb_outcome"] = (df["LBXTBIN"] == 1.0).astype(int)
    igra_pos = df["tb_outcome"].mean()

    # 真实协变量
    df["age"] = df["RIDAGEYR"].astype(float)
    df["past_illness_type"] = "none"
    df.loc[df["DIQ010"] == 1.0, "past_illness_type"] = "diabetes"
    df.loc[df["MCQ220"] == 1.0, "past_illness_type"] = \
        df.loc[df["MCQ220"] == 1.0, "past_illness_type"].map(
            lambda x: x if x == "diabetes" else "other")
    df["past_illness"] = (df["past_illness_type"] != "none").astype(int)
    df["is_high_risk"] = ((df["age"] >= 65) | (df["past_illness"] == 1)).astype(int)

    # NHANES 未采集的接触暴露列（诚实置 0，见 docstring 披露）
    for col in ["cumulative_exposure", "has_symptoms", "bcg_vaccine", "has_tb",
                "contact_distance_score", "ventilation_score",
                "exposure_setting_score", "single_duration",
                "freq_density", "time_span"]:
        df[col] = 0

    # 附加真实列（22 特征外，供扩展分析）
    df["gender"] = df["RIAGENDR"].map({1.0: "m", 2.0: "f"})
    df["bmi"] = df["BMXBMI"]
    df["smoking"] = df["SMQ040"].isin([1.0, 2.0]).astype(int)
    df["born_us"] = (df["DMDBORN4"] == 1.0).astype(int)
    df["income_pir"] = df["INDFMPIR"]
    df["race_eth"] = df["RIDRETH1"]

    keep = ["age", "cumulative_exposure", "has_symptoms", "bcg_vaccine", "has_tb",
            "contact_distance_score", "ventilation_score", "is_high_risk",
            "past_illness", "exposure_setting_score", "single_duration",
            "freq_density", "time_span", "past_illness_type",
            "gender", "bmi", "smoking", "born_us", "income_pir",
            "race_eth", "tb_outcome"]
    out = df[keep].dropna(subset=["age"])
    out_path = os.path.join(PROC, "ml_training_nhanes_real.csv")
    out.to_csv(out_path, index=False)

    meta = {
        "source": "CDC NCHS NHANES 2011-2012（Public Domain，无需注册的公开微观据）",
        "outcome": "LTBI = QuantiFERON Gold in-Tube 官方判定阳性（LBXTBIN==1）",
        "n_total_igra_tested": n0,
        "n_analyzed": len(out),
        "n_positive": int(out["tb_outcome"].sum()),
        "positive_rate_igra": round(igra_pos, 4),
        "note": ("NHANES 的 TST_G.xpt 实为血清睾酮检测（TeSTosterone）而非结核菌素"
                 "试验，已排除；LTBI 标签以 QuantiFERON 官方判定为准"),
        "disclosure": {
            "真实特征": ["age", "past_illness(糖尿病/癌症)", "is_high_risk",
                        "gender", "bmi", "smoking", "born_us", "income_pir"],
            "置0特征(NHANES未采集)": ["接触暴露7列", "has_symptoms", "has_tb"],
            "置0特征(人群特性)": ["bcg_vaccine(美国不接种)"],
            "适用性提示": ("该数据集训练的模型反映美国社区人群 LTBI 风险结构；"
                        "用于中国人群需域适配（迁移学习场景参照）")},
        "variable_map": {
            "age": "RIDAGEYR", "tb_outcome": "LBXTBIN==1 (QuantiFERON GIT)",
            "diabetes": "DIQ010==1", "cancer": "MCQ220==1",
            "bmi": "BMXBMI", "smoking": "SMQ040∈{1,2}",
            "born_us": "DMDBORN4==1", "income_pir": "INDFMPIR"},
    }
    with open(os.path.join(PROC, "nhanes_dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] ml_training_nhanes_real.csv ({len(out)} 行, "
          f"IGRA 阳性 {int(out['tb_outcome'].sum())} = {igra_pos:.1%})")


# ---------------------------------------------------------------------------
# 6. TREATS 队列（赞比亚+南非 14 社区 15-24 岁，真实 LTBI 标签）
# ---------------------------------------------------------------------------
def build_treats_dataset():
    """TREATS-ic baseline → 真实个体级 LTBI 数据集（高负担社区人群）

    来源: LSHTM Data Compass DOI 10.17037/DATA.00003627（CC BY 4.0，无需注册）
    样本: 4529 名 15-24 岁青少年，赞比亚(7 社区)+南非(7 社区)社区队列
    标签: qft = QuantiFERON-TB Gold Plus 官方判定（0.35 IU/ml 截断）

    特征映射披露（诚实标注 TREATS 未采集/受限的字段）：
    - age ← agegroup 组中点(15.5/17.5/19.5/21.5/23.5)（真实但粗化）
    - 接触暴露 7 列 =0（TREATS 的 hhcontact 为 0-3 序数，与 schema 的
      连续接触剂量构造语义不同，不做伪造映射；梯度 46.3%→62.1% 记录于 meta）
    - past_illness=0 / past_illness_type='none'（开放 CSV 无共病数据；
      HIV 状态文件受访问控制 401）
    - is_high_risk=0（队列 15-24 岁且无共病记录，按公式全 0）
    - bcg_vaccine=1（南非/赞比亚均新生儿普种，WHO 估计；常数无个体变异）
    - has_tb=0（社区感染调查，无活动性 TB 诊断列）
    - 附加真实列（22 特征外）: sex/smoking(0-2)/roomshare/hhdens/
      hhcontact(0-3)/alcohol(0-3)/country/community(group_id 供组感知划分)
    """
    import pandas as pd

    src = os.path.join(RAW, "lshtm", "TREATS-ic-baseline-dataset_open.csv")
    df = pd.read_csv(src)

    n0 = len(df)
    # 标签：qft 0/1 官方判定（探测确认零缺失）
    df = df[df["qft"].isin([0, 1])].copy()
    df["tb_outcome"] = df["qft"].astype(int)
    pos_rate = df["tb_outcome"].mean()

    # schema 特征（诚实映射，见 docstring）
    age_mid = {1: 15.5, 2: 17.5, 3: 19.5, 4: 21.5, 5: 23.5}
    df["age"] = df["agegroup"].map(age_mid).astype(float)
    for col in ["cumulative_exposure", "has_symptoms", "has_tb",
                "contact_distance_score", "ventilation_score",
                "exposure_setting_score", "single_duration",
                "freq_density", "time_span"]:
        df[col] = 0
    df["bcg_vaccine"] = 1
    df["past_illness"] = 0
    df["past_illness_type"] = "none"
    df["is_high_risk"] = 0

    # 附加真实列
    df["gender"] = df["sex"].map({1: "m", 2: "f"})
    df["smoking_status"] = df["smoking"]          # 0=从不 1=已戒 2=当前
    df["roomshare_n"] = df["roomshare"]           # 同屋人数 0-5
    df["hhdens_quartile"] = df["hhdens"]          # 家庭密度分位 1-4
    df["hh_tb_contact"] = df["hhcontact"]         # 家庭接触抗结核治疗 0-3(36 NaN)
    df["alcohol_use"] = df["alcohol"]             # 0=从不 1=每月 2=2-4次/月 3=5+次/月
    df["country_code"] = df["country"]            # SA / Z
    df["group_id"] = df["community"]              # 14 社区（组感知划分用）

    keep = ["age", "cumulative_exposure", "has_symptoms", "bcg_vaccine", "has_tb",
            "contact_distance_score", "ventilation_score", "is_high_risk",
            "past_illness", "exposure_setting_score", "single_duration",
            "freq_density", "time_span", "past_illness_type",
            "gender", "smoking_status", "roomshare_n", "hhdens_quartile",
            "hh_tb_contact", "alcohol_use", "country_code", "group_id",
            "tb_outcome"]
    out = df[keep]
    out_path = os.path.join(PROC, "ml_training_treats_real.csv")
    out.to_csv(out_path, index=False)

    by_country = df.groupby("country")["tb_outcome"].mean().round(4).to_dict()
    by_contact = (df.dropna(subset=["hhcontact"])
                    .groupby("hhcontact")["tb_outcome"].mean().round(4).to_dict())
    meta = {
        "source": ("LSHTM Data Compass TREATS 队列 DOI 10.17037/DATA.00003627 "
                   "（CC BY 4.0，无需注册）"),
        "outcome": "LTBI = QuantiFERON-TB Gold Plus 官方判定阳性（qft==1，0.35 IU/ml）",
        "n_total": n0,
        "n_analyzed": len(out),
        "n_positive": int(out["tb_outcome"].sum()),
        "positive_rate_qft": round(pos_rate, 4),
        "positive_rate_by_country": by_country,
        "positive_rate_by_hhcontact": by_contact,
        "note": ("高负担社区人群真实 LTBI 数据（vs NHANES 美国低负担社区）；"
                 "recentremote 伴发数据(DOI .00003879)为 1343 人 QFT-Plus 连续读数"
                 "（含删失字符串值），未纳入训练"),
        "disclosure": {
            "真实特征": ["age(组中点粗化)", "gender", "smoking_status",
                        "roomshare_n", "hhdens_quartile", "hh_tb_contact",
                        "alcohol_use", "country/community"],
            "置0特征(未采集或语义不符)": ["接触暴露7列", "has_symptoms", "has_tb"],
            "置0特征(数据受限)": ["past_illness(HIV 状态 401 受控，无共病列)",
                                "is_high_risk(年龄15-24且无共病记录)"],
            "置常数特征": ["bcg_vaccine=1(两国新生儿普种推断，无个体变异)"],
            "组感知划分": "group_id=community（14 社区），真实数据训练必须 groups= 防社区泄漏",
            "适用性提示": ("22 维 schema 仅 age 携带真实个体变异，训练性能反映"
                        "schema 对该队列的可提取信息（~0.55 AUROC 量级），"
                        "非 TREATS 信息上限；hhcontact 梯度 46.3%→62.1% 需"
                        "schema 扩展后方可入模")},
        "variable_map": {
            "age": "agegroup 组中点(1..5 → 15.5..23.5)",
            "tb_outcome": "qft==1 (QuantiFERON Gold Plus)",
            "gender": "sex(1=m,2=f)", "smoking_status": "smoking(0/1/2)",
            "roomshare_n": "roomshare(0-5)", "hhdens_quartile": "hhdens(1-4)",
            "hh_tb_contact": "hhcontact(0=无 1=既往 2=当前 3=两者)",
            "alcohol_use": "alcohol(0-3)", "group_id": "community(SA14..Z12)"},
    }
    with open(os.path.join(PROC, "treats_dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] ml_training_treats_real.csv ({len(out)} 行, "
          f"QFT 阳性 {int(out['tb_outcome'].sum())} = {pos_rate:.1%})")


# ---------------------------------------------------------------------------
# 7. CRP 社区筛查队列（赞比亚+南非，真实细菌学确诊标签）
# ---------------------------------------------------------------------------
def build_crp_dataset():
    """LSHTM CRP 社区 TB 筛查开放版 → 真实个体级筛查分诊数据集

    来源: LSHTM Data Compass DOI 10.17037/DATA.00005150（CC BY 4.0，开放版）
    设计: 社区主动病例发现(ACF)筛查，赞比亚+南非，2019-02~08
    标签: Xpertculture = Xpert 或培养任一阳性（细菌学确诊）

    验证偏倚披露（关键）：
    - 9588 人接受筛查，仅 1777 人进入验证（sputum_elig = 症状或 CAD≥40）
    - 训练集 = 验证子集（"分诊确认"任务：在已送检人群中识别确诊），
      非全人群患病率任务；外推到未送检人群需重新校准
    特征映射披露：
    - age ← age_group 组中点(20/30/40/50/60；≥55 开区间取 60)
    - has_symptoms ← symp_elig（TREATS 定义，真实）
    - past_illness ← tb_history=="history of tb treatment"（type=other）
    - has_tb=0（现症 TB 是结局本体，不作特征）
    - 接触暴露 7 列 =0（社区筛查未采集）
    - bcg_vaccine=1（两国新生儿普种推断，常数无个体变异）
    - 附加真实列: CAD_score/crp_value(mg/dl)/五类症状及周数/sex/
      age_group/tb_history/country
    - 无聚类变量（村/社区 ID 未开放），按独立个体处理（局限记录于 meta）
    """
    import pandas as pd

    src = os.path.join(RAW, "lshtm", "CRP_dataset_open.csv")
    df = pd.read_csv(src)

    n_screened = len(df)
    # 验证子集：Xpertculture 非缺失（进入 Xpert/培养验证者）
    ver = df[df["Xpertculture"].notna()].copy()
    n_verified = len(ver)
    ver["tb_outcome"] = (ver["Xpertculture"] == 1.0).astype(int)
    pos_rate = ver["tb_outcome"].mean()

    # schema 特征（诚实映射，见 docstring）
    age_mid = {1: 20.0, 2: 30.0, 3: 40.0, 4: 50.0, 5: 60.0}
    ver["age"] = ver["age_group"].map(age_mid).astype(float)
    ver["has_symptoms"] = ver["symp_elig"].astype(int)
    ver["past_illness"] = (
        ver["tb_history"] == "history of tb treatment").astype(int)
    ver["past_illness_type"] = "none"
    ver.loc[ver["past_illness"] == 1, "past_illness_type"] = "other"
    ver["is_high_risk"] = ((ver["age"] >= 65) |
                           (ver["past_illness"] == 1)).astype(int)
    ver["has_tb"] = 0
    ver["bcg_vaccine"] = 1
    for col in ["cumulative_exposure", "contact_distance_score",
                "ventilation_score", "exposure_setting_score",
                "single_duration", "freq_density", "time_span"]:
        ver[col] = 0

    # 附加真实列
    ver["gender"] = ver["sex"].map({"F": "f", "M": "m"})
    ver["cad_score"] = pd.to_numeric(ver["CAD_score"], errors="coerce")
    ver["crp_mgdl"] = pd.to_numeric(ver["crp_value"], errors="coerce")
    ver["crp_positive"] = ver["crp_positive"]
    for s in ["cough", "fever", "chestpain", "nightsweats", "weightlost"]:
        ver[f"symptom_{s}"] = (ver[f"symp_{s}"] == 1).astype(int)
        ver[f"symptom_{s}_weeks"] = ver[f"symp_{s}_weeks"]
    ver["tb_history_raw"] = ver["tb_history"]
    ver["country_code"] = ver["country"]

    keep = ["age", "cumulative_exposure", "has_symptoms", "bcg_vaccine", "has_tb",
            "contact_distance_score", "ventilation_score", "is_high_risk",
            "past_illness", "exposure_setting_score", "single_duration",
            "freq_density", "time_span", "past_illness_type",
            "gender", "cad_score", "crp_mgdl", "crp_positive",
            "symptom_cough", "symptom_cough_weeks", "symptom_fever",
            "symptom_fever_weeks", "symptom_chestpain",
            "symptom_chestpain_weeks", "symptom_nightsweats",
            "symptom_nightsweats_weeks", "symptom_weightlost",
            "symptom_weightlost_weeks", "age_group", "tb_history_raw",
            "country_code", "tb_outcome"]
    out = ver[keep]
    out_path = os.path.join(PROC, "ml_training_crp_real.csv")
    out.to_csv(out_path, index=False)

    # 分诊特征的单变量判别（供 meta 记录）
    from sklearn.metrics import roc_auc_score
    uni = {}
    for c in ["cad_score", "crp_mgdl"]:
        sub = ver[[c, "tb_outcome"]].dropna()
        if sub[c].nunique() > 2 and sub["tb_outcome"].nunique() == 2:
            uni[c] = round(float(roc_auc_score(sub["tb_outcome"], sub[c])), 4)
    uni["has_symptoms"] = round(float(
        ver.groupby("has_symptoms")["tb_outcome"].mean().diff().iloc[-1]), 4)

    meta = {
        "source": ("LSHTM Data Compass CRP 社区筛查开放版 "
                   "DOI 10.17037/DATA.00005150（CC BY 4.0）"),
        "outcome": "tb_outcome = Xpertculture==1（Xpert 或培养任一阳性，细菌学确诊）",
        "n_screened": n_screened,
        "n_verified": n_verified,
        "n_analyzed": len(out),
        "n_positive": int(out["tb_outcome"].sum()),
        "positive_rate_verified": round(pos_rate, 4),
        "verification_bias_disclosure": (
            "验证规则 sputum_elig = 症状(TREATS定义) 或 CAD≥40；9588 筛查中仅 "
            f"{n_verified} 进入验证。本训练集为验证子集（分诊确认任务），"
            "患病率 6.6% 为验证子集患病率而非人群患病率"),
        "univariate_discrimination": uni,
        "note": ("真实 ACF 筛查分诊数据（赞比亚+南非），与本项目克拉玛依部署"
                 "场景同构；CAD/CRP 为 schema 外真实分诊特征"),
        "disclosure": {
            "真实特征": ["age(组中点)", "has_symptoms(symp_elig)",
                        "past_illness(既往抗结核治疗史)",
                        "gender", "cad_score", "crp_mgdl", "五类症状及周数"],
            "置0特征(未采集)": ["接触暴露7列"],
            "置0特征(结局本体)": ["has_tb(现症TB即结局)"],
            "置常数特征": ["bcg_vaccine=1(两国普种推断)"],
            "聚类局限": "无村/社区级聚类ID，按独立个体划分（无法组感知CV）",
            "适用性提示": ("22 维 schema 可用真实变异 = age/has_symptoms/"
                        "past_illness；CAD_score(单变量 AUROC 见 uni)与 CRP 为"
                        "schema 外强特征，schema 扩展后可显著提升")},
        "variable_map": {
            "age": "age_group 组中点(1..5 → 20/30/40/50/60)",
            "tb_outcome": "Xpertculture==1",
            "has_symptoms": "symp_elig(TREATS 症状定义)",
            "past_illness": "tb_history=='history of tb treatment'",
            "cad_score": "CAD_score(CAD 读片 0-100)",
            "crp_mgdl": "crp_value(POC CRP, mg/dl)"},
    }
    with open(os.path.join(PROC, "crp_dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] ml_training_crp_real.csv ({len(out)} 行验证子集, "
          f"确诊 {int(out['tb_outcome'].sum())} = {pos_rate:.1%}; "
          f"单变量 AUROC: {uni})")


# ---------------------------------------------------------------------------
# 8. Cape Town 社会接触调查（经验性接触模式，GNN 层校准参照）
# ---------------------------------------------------------------------------
def build_cape_town_contacts():
    """Cape Town 接触调查 → 经验性接触统计（对照 Prem 合成矩阵）

    来源: LSHTM Data Compass DOI 10.17037/DATA.00002756（开放）
    价值: 非洲高 TB 负担城市人群的经验接触模式（家庭/密切/建筑/交通），
         为 GNN 家庭/社会层强度提供经验锚点（Prem 中国矩阵为合成值）
    """
    import zipfile

    import pandas as pd

    zp = os.path.join(RAW, "lshtm", "Cape_Town_social_contact_data.zip")
    exdir = os.path.join(RAW, "lshtm", "cape_town_contacts")
    if not os.path.exists(zp):
        print("[跳过] Cape_Town_social_contact_data.zip 不存在")
        return
    if not os.path.exists(exdir):
        with zipfile.ZipFile(zp) as z:
            z.extractall(exdir)

    hh = pd.read_csv(os.path.join(exdir, "Household_data.csv"))
    cc = pd.read_csv(os.path.join(exdir, "Close_contacts_data.csv"))
    main = pd.read_csv(os.path.join(exdir, "Main_data.csv"))

    hh_size = main["hhSize"] if "hhSize" in main.columns else None
    summary = {
        "source": ("LSHTM Data Compass Cape Town 社会接触调查 "
                   "DOI 10.17037/DATA.00002756（CC BY 4.0）"),
        "n_participants": int(len(main)),
        "household": {
            "n_rows": int(len(hh)),
            "mean_members_per_participant": round(
                float(hh.groupby("id").size().mean()), 2),
            "mean_household_size_selfreport": round(
                float(hh_size.mean()), 2) if hh_size is not None else None,
        },
        "close_contacts": {
            "n_rows": int(len(cc)),
            "mean_reported_total": round(
                float(pd.to_numeric(cc["contactstotal"], errors="coerce").mean()), 1),
            "mean_non_household": round(
                float(pd.to_numeric(cc["contactsnonhhnumber"], errors="coerce").mean()), 1),
        },
        "usage_note": ("经验性非洲城市接触模式；与 processed/"
                     "contact_mixing_summary.json 的 Prem 中国合成矩阵对照，"
                     "GNN 家庭/社会边强度跨人群迁移时作敏感度参照"),
    }
    out_path = os.path.join(PROC, "cape_town_contact_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] cape_town_contact_summary.json "
          f"(n={summary['n_participants']}, "
          f"家庭均员 {summary['household']['mean_members_per_participant']}, "
          f"密切接触报告均值 {summary['close_contacts']['mean_reported_total']})")


# ---------------------------------------------------------------------------
# 9. 台湾 QFT-Plus/活动性 TB 双结局队列（Mendeley，2026-09-05 全球搜寻新增）
# ---------------------------------------------------------------------------
def build_taiwan_dataset():
    """台湾 NTUH QFT-Plus 队列 → 真实个体级活动性 TB 数据集

    来源: Mendeley Data doi:10.17632/457bkx9p6n.2（CC BY 4.0，
         Wang Jann-Yuan, NTUH 2021）
    设计: 双队列——129 人活动性 TB 评估组（TB/culture/smear/CXR 完整）
         + 207 人 LTBI 组（TB 结局缺失）
    标签: tb_outcome = TB ∈ {1(确诊), 2(临床诊断)}，TB==0 为阴性；
         仅用 TB 列非缺失的评估组（n=129）

    特征映射披露：
    - age/bmi/Smoking/合并症 7 列为真实值（评估组无缺失）
    - gender ← sex（1/2 编码，按台湾常规 1=M/2=F 推断，meta 披露）
    - past_illness ← dm/esrd/cancer/Cirrhosis/COPD/Steroid/Autoimmune 任一；
      past_illness_type 优先映射 diabetes（schema 免疫抑制权重最高）
    - has_tb=0（现症 TB 是结局本体）
    - bcg_vaccine=1（台湾 1965 起新生儿普种，推断常数）
    - 接触暴露 7 列=0（未采集）
    - 附加真实列: QFT-Plus 定量 6 列 + cxr_score/三影像征象 + LTBI 分层
    - QFT 定量与感染同源但终点是活动性 TB（感染→发病链条），临床合理
    """
    import pandas as pd

    src = os.path.join(RAW, "mendeley", "taiwan_qftplus.csv")
    if not os.path.exists(src):
        print("[跳过] taiwan_qftplus.csv 不存在")
        return
    df = pd.read_csv(src)

    ev = df[df["TB"].notna()].copy()
    ev["tb_outcome"] = ev["TB"].isin([1, 2]).astype(int)

    ev["age"] = ev["age"].astype(float)
    ev["gender"] = ev["sex"].map({1: "m", 2: "f"})
    ev["has_symptoms"] = 0
    comorb_cols = ["dm", "esrd", "cancer", "Cirrhosis", "COPD",
                   "Steroid_User", "Autoimmune_Disease"]
    ev["past_illness"] = (ev[comorb_cols].fillna(0).sum(axis=1) > 0).astype(int)
    ev["past_illness_type"] = "none"
    ev.loc[ev["dm"] == 1, "past_illness_type"] = "diabetes"
    other = (ev["past_illness"] == 1) & (ev["past_illness_type"] == "none")
    ev.loc[other, "past_illness_type"] = "other"
    ev["is_high_risk"] = ((ev["age"] >= 65) |
                          (ev["past_illness"] == 1)).astype(int)
    ev["has_tb"] = 0
    ev["bcg_vaccine"] = 1
    for col in ["cumulative_exposure", "contact_distance_score",
                "ventilation_score", "exposure_setting_score",
                "single_duration", "freq_density", "time_span"]:
        ev[col] = 0

    ev["qft_nil"] = ev["Nil"]
    ev["qft_tb_ag"] = ev["TB_Ag"]
    ev["qft_tb_ag_nil"] = ev["TB_Ag_nil"]
    ev["qft_t1"] = ev["T1"]
    ev["qft_t2"] = ev["T2"]
    ev["qft_cd8"] = ev["CD8"]
    ev["qft_result"] = ev["New_Result"].str.lower()
    ev["cxr_score"] = ev["cxr_score"]
    ev["cxr_fibronodular"] = ev["Fibronodular"]
    ev["cxr_cavitation"] = ev["Cavitation"]
    ev["cxr_pleural_effusion"] = ev["pleural_effusion"]
    ev["bmi"] = ev["bmi"]
    ev["smoking"] = ev["Smoking"]
    ev["comorbidity_any"] = ev["comorbidity"].fillna(0)
    ev["culture_result"] = ev["culture"]
    ev["smear_cat"] = ev["smear_cat"]
    ev["ltbi_class_raw"] = ev["LTBI"]

    keep = ["age", "cumulative_exposure", "has_symptoms", "bcg_vaccine",
            "has_tb", "contact_distance_score", "ventilation_score",
            "is_high_risk", "past_illness", "exposure_setting_score",
            "single_duration", "freq_density", "time_span",
            "past_illness_type",
            "gender", "bmi", "smoking", "comorbidity_any",
            "qft_nil", "qft_tb_ag", "qft_tb_ag_nil", "qft_t1", "qft_t2",
            "qft_cd8", "qft_result", "cxr_score", "cxr_fibronodular",
            "cxr_cavitation", "cxr_pleural_effusion",
            "culture_result", "smear_cat", "ltbi_class_raw", "tb_outcome"]
    out = ev[keep]
    out_path = os.path.join(PROC, "ml_training_taiwan_real.csv")
    out.to_csv(out_path, index=False)

    from sklearn.metrics import roc_auc_score
    uni = {}
    for c in ["cxr_score", "qft_tb_ag_nil", "qft_cd8", "bmi", "age"]:
        sub = out[[c, "tb_outcome"]].dropna()
        if sub[c].nunique() > 2:
            uni[c] = round(float(roc_auc_score(sub["tb_outcome"], sub[c])), 4)

    meta = {
        "source": ("Mendeley Data 台湾 NTUH QFT-Plus 队列 "
                   "doi:10.17632/457bkx9p6n.2（CC BY 4.0）"),
        "outcome": "tb_outcome = TB∈{1确诊, 2临床诊断}；评估组=TB列非缺失者",
        "n_total": len(df), "n_evaluated": len(out),
        "n_positive": int(out["tb_outcome"].sum()),
        "positive_rate": round(float(out["tb_outcome"].mean()), 4),
        "univariate_discrimination": uni,
        "disclosure": {
            "真实特征": ["age", "bmi", "smoking", "gender(编码推断1=M/2=F)",
                        "合并症7列(dm/esrd/cancer/Cirrhosis/COPD/"
                        "Steroid/Autoimmune)", "QFT-Plus定量6列",
                        "cxr_score+三影像征象", "culture/smear"],
            "置0特征(未采集)": ["has_symptoms", "接触暴露7列"],
            "置0特征(结局本体)": ["has_tb"],
            "置常数特征": ["bcg_vaccine=1(台湾新生儿普种推断)"],
            "同源性说明": ("QFT 定量与感染状态同源（New_Result 由 TB_Ag_nil "
                        "判读），但终点为活动性 TB（感染→发病链条），"
                        "临床时序上 QFT 先于确诊可用"),
            "双队列结构": ("全 336 人含 207 人 LTBI 组（TB 结局缺失，未纳入"
                        "本数据集）；LTBI 任务可用 QFT 全量另行构造"),
        },
        "variable_map": {
            "age": "age(真实连续)", "tb_outcome": "TB∈{1,2}",
            "past_illness": "合并症7列任一",
            "cxr_score": "cxr_score(CXR 异常评分1-13)",
            "qft_tb_ag_nil": "TB_Ag-Nil(QFT-Plus 抗原管差值 IU/ml)"},
    }
    with open(os.path.join(PROC, "taiwan_dataset_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] ml_training_taiwan_real.csv ({len(out)} 行评估组, "
          f"阳性 {int(out['tb_outcome'].sum())} = "
          f"{out['tb_outcome'].mean():.1%}; 单变量 AUROC: {uni})")


# ---------------------------------------------------------------------------
# 10. 巴西 IGRA/TST/IFNG 基因型队列（Mendeley）
# ---------------------------------------------------------------------------
def build_brazil_igra_dataset():
    """巴西东北部 IGRA 队列 → 真实个体级 LTBI 数据集

    来源: Mendeley Data doi:10.17632/mfcwvxwrhc.1（CC BY 4.0，
         Carneiro et al. 2018）
    设计: 三组——TB(76 活动性)/TST+(71 LTBI)/TST-(59 对照)
    标签: tb_outcome = IGRA result==POSITIVE（LTBI 感染）
         排除 INDETERMINATE(3)/缺失(1) → n=202

    特征映射披露（防泄漏关键）：
    - 不使用 IFN-γ 定量 7 列（与 IGRA 结局同源——IGRA 即 IFN-γ 检测）
    - 不使用 Group（TST+/TST- 按定义编码 TST 结果）
    - 不使用 TST result（与 IGRA 高度相关但属另一检测，作记录列）
    - 宿主特征仅 age/sex/ethnic/IFNG+874 基因型——判别力弱是诚实结果
    - bcg_vaccine=1（巴西新生儿普种推断）；接触 7 列=0；past_illness=0
    """
    import pandas as pd

    src = os.path.join(RAW, "mendeley", "brazil_igra_ifng.xlsx")
    if not os.path.exists(src):
        print("[跳过] brazil_igra_ifng.xlsx 不存在")
        return
    df = pd.read_excel(src)
    df = df.replace(".", pd.NA)

    keep_rows = df["IGRA result"].isin(["POSITIVE", "NEGATIVE"])
    ev = df[keep_rows].copy()
    ev["tb_outcome"] = (ev["IGRA result"] == "POSITIVE").astype(int)

    ev["age"] = ev["Age"].astype(float)
    ev["gender"] = ev["Sex"].map({"F": "f", "M": "m"})
    ev["has_symptoms"] = 0
    ev["past_illness"] = 0
    ev["past_illness_type"] = "none"
    ev["is_high_risk"] = (ev["age"] >= 65).astype(int)
    ev["has_tb"] = 0
    ev["bcg_vaccine"] = 1
    for col in ["cumulative_exposure", "contact_distance_score",
                "ventilation_score", "exposure_setting_score",
                "single_duration", "freq_density", "time_span"]:
        ev[col] = 0

    ev["ifng_genotype"] = ev["IFNG+874"]
    ev["ethnic"] = ev["Ethnic background"]
    ev["group_raw"] = ev["Group"]
    ev["tst_result_raw"] = ev["TST result"]

    keep = ["age", "cumulative_exposure", "has_symptoms", "bcg_vaccine",
            "has_tb", "contact_distance_score", "ventilation_score",
            "is_high_risk", "past_illness", "exposure_setting_score",
            "single_duration", "freq_density", "time_span",
            "past_illness_type",
            "gender", "ifng_genotype", "ethnic", "group_raw",
            "tst_result_raw", "tb_outcome"]
    out = ev[keep]
    out_path = os.path.join(PROC, "ml_training_brazil_real.csv")
    out.to_csv(out_path, index=False)

    # 单变量：基因型 TT(高IFN-γ) vs AA 的 LTBI 率差 + age
    geno = out.groupby("ifng_genotype")["tb_outcome"].agg(["mean", "count"])
    tst_igra = None
    both = out[out["tst_result_raw"].isin(["POSITIVE", "NEGATIVE"])]
    if len(both) > 20:
        tst_igra = {
            "n_both_tests": len(both),
            "concordance": round(float(
                (both["tst_result_raw"].eq(
                    both["tb_outcome"].map({1: "POSITIVE", 0: "NEGATIVE"}))
                 ).mean()), 4),
        }

    meta = {
        "source": ("Mendeley Data 巴西东北部 IGRA 队列 "
                   "doi:10.17632/mfcwvxwrhc.1（CC BY 4.0, Carneiro 2018）"),
        "outcome": "tb_outcome = IGRA result==POSITIVE（LTBI）",
        "n_total": len(df), "n_analyzed": len(out),
        "n_positive": int(out["tb_outcome"].sum()),
        "positive_rate": round(float(out["tb_outcome"].mean()), 4),
        "genotype_ltbi_rate": {
            k: [round(float(r["mean"]), 4), int(r["count"])]
            for k, r in geno.iterrows()},
        "tst_igra_agreement": tst_igra,
        "disclosure": {
            "防泄漏排除": ["IFN-γ定量7列(与IGRA同源)", "Group(编码TST定义)",
                          "TST result(另一切片检测,仅记录)"],
            "真实特征": ["age", "gender", "ifng_genotype(AA/TA/TT)", "ethnic"],
            "置0特征": ["has_symptoms/past_illness(未采集)",
                       "接触暴露7列", "has_tb(结局本体)"],
            "置常数特征": ["bcg_vaccine=1(巴西新生儿普种推断)"],
            "诚实预期": ("仅宿主特征(age/sex/基因型/ethnic)预测 LTBI 的判别力"
                       "有限——LTBI 主要由暴露决定而暴露未采集，"
                       "本数据集价值在 TST/IGRA 不一致 + 宿主遗传维度"),
        },
        "variable_map": {
            "age": "Age(真实)", "tb_outcome": "IGRA result==POSITIVE",
            "ifng_genotype": "IFNG+874 A>T 多态性(AA/TA/TT)"},
    }
    with open(os.path.join(PROC, "brazil_dataset_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] ml_training_brazil_real.csv ({len(out)} 行, "
          f"LTBI 阳性 {int(out['tb_outcome'].sum())} = "
          f"{out['tb_outcome'].mean():.1%}; 基因型 LTBI 率: "
          f"{ {k: round(float(r['mean']),3) for k, r in geno.iterrows()} })")


# ---------------------------------------------------------------------------
# 11. 秘鲁 MDR vs 敏感 TB 家庭接触者前瞻队列（Dryad，感染→病程终点）
# ---------------------------------------------------------------------------
def build_peru_mdr_dataset():
    """秘鲁 MDR vs 敏感 TB 家庭接触者 3 年前瞻队列 → 真实个体级活动性 TB 数据集

    来源: Dryad doi:10.5061/dryad.br760（CC0，Grandjean 2016 PLOS Medicine）
    设计: 双队列——MDR TB 指示病例家庭接触者 + 敏感 TB 指示病例家庭接触者，
         前瞻随访约 3 年（Follow Up Time 1-962 天）
    标签: tb_outcome = Incident TB Disease（3 年内发生活动性 TB，149/3406）

    特征映射披露：
    - age ← Contact Age 年龄带编码 1-9 的带中点 (c-1)*10+5
      （Grandjean 2016 Table 2 年龄分组 0-10…>70；带内均匀假设）
    - gender ← Contact Sex（1/0；1=M/0=F 编码推断，与论文 Table 1
      男性占比 49.4% 吻合）
    - has_tb ← Contact Previous TB History（缺省 21 例置 0）
    - past_illness ← Contact HIV | Contact Diabetes；
      past_illness_type 优先映射 hiv（schema 免疫抑制权重最高）
    - contact_distance_score ← Contact-Index Roomshare（共享卧室=1，
      近距离家庭暴露的真实 0/1 测量；缺省 26 例置 0）
    - cumulative_exposure ← Index Sputum Smear Grade 0-3
      （指示病例传染性剂量代理；缺省 106 例置 0）
    - exposure_setting_score ← MDR household（MDR=1，SENS=0；
      暴露源类型而非物理场景评分，见 meta 披露）
    - bcg_vaccine=1（秘鲁新生儿 BCG 普种，WHO 估计覆盖率 >90%，推断常数）
    - has_symptoms/ventilation_score/single_duration/freq_density/time_span=0
      （基线未采集）
    - 附加真实列: family_id/mdr_household/index_smear_grade/
      index_cough_days/index_hiv/age_band/follow_up_days
    - 家庭聚类: 149 阳性分布于 116/688 户；train_from_real_data 的 CV
      不感知家庭分组，AUROC 点估计存在家庭内相关导致的乐观偏差；
      family_id 列保留供组感知（GroupKFold）重分析
    """
    import pandas as pd

    src = os.path.join(RAW, "dryad", "peru_mdr_household_incident.xlsx")
    if not os.path.exists(src):
        print("[跳过] peru_mdr_household_incident.xlsx 不存在")
        return
    df = pd.read_excel(src)

    # 指示病例行（Individual Code=1 或 Contact 字段全缺）不进入接触者队列
    contacts = df[df["Contact Age"].notna() &
                  df["Contact Sex"].notna()].copy()
    contacts = contacts[contacts["Incident TB Disease"].notna()].copy()

    # 年龄带 1-9 → 带中点（0-10→5, 10-20→15, …, 80+→85）
    contacts["age"] = (contacts["Contact Age"].astype(float) - 1) * 10 + 5
    contacts["gender"] = contacts["Contact Sex"].map({1: "m", 0: "f"})
    contacts["tb_outcome"] = contacts["Incident TB Disease"].astype(int)

    # 宿主特征
    contacts["has_hiv"] = contacts["Contact HIV"].fillna(0).astype(int)
    contacts["has_diabetes"] = contacts["Contact Diabetes"].fillna(0).astype(int)
    contacts["past_illness"] = ((contacts["has_hiv"] == 1) |
                                (contacts["has_diabetes"] == 1)).astype(int)
    contacts["past_illness_type"] = "none"
    contacts.loc[contacts["has_diabetes"] == 1,
                 "past_illness_type"] = "diabetes"
    contacts.loc[contacts["has_hiv"] == 1, "past_illness_type"] = "hiv"
    contacts["has_tb"] = contacts["Contact Previous TB History"].fillna(
        0).astype(int)
    contacts["is_high_risk"] = ((contacts["age"] >= 65) |
                                (contacts["past_illness"] == 1)).astype(int)
    contacts["bcg_vaccine"] = 1
    contacts["has_symptoms"] = 0
    contacts["ventilation_score"] = 0
    contacts["single_duration"] = 0
    contacts["freq_density"] = 0
    contacts["time_span"] = 0

    # 暴露特征（真实测量，见 docstring 披露）
    contacts["contact_distance_score"] = contacts[
        "Contact-Index Roomshare"].fillna(0).astype(float)
    contacts["cumulative_exposure"] = contacts[
        "Index Sputum Smear Grade"].fillna(0).astype(float)
    contacts["exposure_setting_score"] = (
        contacts["MDR or Sensitive Household"] == "MDR").astype(float)

    # 附加披露列（原始语义，不进 22 维特征）
    contacts["family_id"] = contacts["Family Code"]
    contacts["mdr_household"] = contacts["exposure_setting_score"]
    contacts["index_smear_grade"] = contacts[
        "Index Sputum Smear Grade"].fillna(0)
    contacts["index_smear_missing"] = contacts[
        "Index Sputum Smear Grade"].isna().astype(int)
    contacts["index_cough_days"] = contacts["Index Cough Duration (days)"]
    contacts["index_hiv"] = contacts["Index HIV Status"]
    contacts["age_band"] = contacts["Contact Age"].astype(int)
    contacts["follow_up_days"] = contacts["Follow Up Time"]

    keep = ["age", "cumulative_exposure", "has_symptoms", "bcg_vaccine",
            "has_tb", "contact_distance_score", "ventilation_score",
            "is_high_risk", "past_illness", "exposure_setting_score",
            "single_duration", "freq_density", "time_span",
            "past_illness_type", "gender",
            "family_id", "mdr_household", "index_smear_grade",
            "index_smear_missing", "index_cough_days", "index_hiv",
            "age_band", "follow_up_days", "tb_outcome"]
    out = contacts[keep]
    out_path = os.path.join(PROC, "ml_training_peru_mdr_real.csv")
    out.to_csv(out_path, index=False)

    from sklearn.metrics import roc_auc_score
    uni = {}
    for c in ["age", "cumulative_exposure", "exposure_setting_score",
              "contact_distance_score"]:
        sub = out[[c, "tb_outcome"]].dropna()
        if sub[c].nunique() > 2:
            uni[c] = round(float(
                roc_auc_score(sub["tb_outcome"], sub[c])), 4)
    # 组感知对照：同户成员结局相关性（家庭聚类的量化披露）
    fam_pos_rate = out.groupby("family_id")["tb_outcome"].mean()
    n_pos_families = int((fam_pos_rate > 0).sum())

    meta = {
        "source": ("Dryad 秘鲁 MDR vs 敏感 TB 家庭接触者 3 年前瞻队列 "
                   "doi:10.5061/dryad.br760（CC0，Grandjean 2016）"),
        "outcome": "tb_outcome = Incident TB Disease（前瞻发生活动性 TB）",
        "n_total": len(df), "n_contacts": len(out),
        "n_positive": int(out["tb_outcome"].sum()),
        "positive_rate": round(float(out["tb_outcome"].mean()), 4),
        "n_families": int(out["family_id"].nunique()),
        "n_families_with_positive": n_pos_families,
        "univariate_discrimination": uni,
        "variable_map": {
            "age": "Contact Age 带编码 1-9 → 带中点 (c-1)*10+5",
            "gender": "Contact Sex 1=M/0=F（编码推断，与论文男性占比吻合）",
            "tb_outcome": "Incident TB Disease",
            "has_tb": "Contact Previous TB History（缺省 21→0）",
            "past_illness": "Contact HIV | Contact Diabetes",
            "contact_distance_score": "Contact-Index Roomshare（0/1，缺省 26→0）",
            "cumulative_exposure": "Index Sputum Smear Grade 0-3（缺省 106→0）",
            "exposure_setting_score": "MDR household=1 / SENS=0",
        },
        "disclosure": {
            "真实特征": ["age(带中点)", "gender(编码推断)",
                        "has_tb(既往TB史)", "past_illness(HIV/糖尿病)",
                        "contact_distance_score(共享卧室)",
                        "cumulative_exposure(指示病例痰涂片等级)",
                        "exposure_setting_score(MDR暴露)"],
            "置0特征(未采集)": ["has_symptoms", "ventilation_score",
                              "single_duration", "freq_density", "time_span"],
            "置常数特征": ["bcg_vaccine=1(秘鲁新生儿普种推断)"],
            "家庭聚类": (f"149 阳性分布于 {n_pos_families}/"
                        f"{out['family_id'].nunique()} 户；"
                        "train_from_real_data 的 CV 不感知家庭分组，"
                        "AUROC 点估计存在家庭内相关导致的乐观偏差；"
                        "family_id 列保留供组感知（GroupKFold）重分析"),
            "量纲说明": ("cumulative_exposure 语义为小时，此处装入痰涂片"
                        "等级 0-3（序保留，树模型不受量纲影响）；"
                        "exposure_setting_score 语义为场景评分，"
                        "此处装入 MDR 暴露源 0/1"),
        },
    }
    with open(os.path.join(PROC, "peru_mdr_dataset_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] ml_training_peru_mdr_real.csv "
          f"({len(out)} 行, 阳性 {int(out['tb_outcome'].sum())}, "
          f"{out['family_id'].nunique()} 户)")


# ---------------------------------------------------------------------------
# 12. TB 协变量时序（WHO TME + OWID，SEIR 外部驱动/协变量分析）
# ---------------------------------------------------------------------------
def build_tb_covariates_timeseries():
    """WHO TME 负担估算 + OWID 四协变量 → 中国多变量时序表

    来源: WHO 全球 TB 报告 TME CSV（estimates，与 GHO 指标 API 不同源）
         + OWID（HIV/糖尿病/吸烟/营养不良，CC BY）
    产出: tb_covariates_china_timeseries.csv（年份 × 8+ 列）
    用途: SEIR 机制模型的外部协变量/敏感性分析（TB 进展的宿主风险
         因素人群水平轨迹）
    """
    import pandas as pd

    est_path = os.path.join(RAW, "who_tme", "who_tme_estimates.csv")
    if not os.path.exists(est_path):
        print("[跳过] who_tme_estimates.csv 不存在")
        return
    est = pd.read_csv(est_path, encoding="utf-8-sig")
    chn = est[est["iso3"] == "CHN"][[
        "year", "e_inc_num", "e_mort_num", "e_inc_100k",
        "e_mort_100k"]].copy() if "iso3" in est.columns else pd.DataFrame()

    owid_files = {
        "hiv_prev_adult": "share-of-the-population-infected-with-hiv.csv",
        "diabetes_prev": "diabetes-prevalence.csv",
        "smoking_prev": "share-of-adults-who-smoke.csv",
        "undernourishment": "prevalence-of-undernourishment.csv",
    }
    cov_frames = []
    for col, fname in owid_files.items():
        p = os.path.join(RAW, "owid", fname)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        # OWID grapher 格式: Entity,Code,Year,{value_col}
        val_col = [c for c in d.columns
                   if c not in ("Entity", "Code", "Year")][0]
        d = d[(d["Entity"] == "China") & d["Year"].notna()]
        cov_frames.append(d[["Year", val_col]].rename(
            columns={"Year": "year", val_col: col}))
    if cov_frames:
        cov = cov_frames[0]
        for c in cov_frames[1:]:
            cov = cov.merge(c, on="year", how="outer")
    else:
        cov = pd.DataFrame(columns=["year"])

    out = chn.merge(cov, on="year", how="left").sort_values("year")
    out_path = os.path.join(PROC, "tb_covariates_china_timeseries.csv")
    out.to_csv(out_path, index=False)
    meta = {
        "source": ("WHO Global TB Report TME estimates CSV + OWID grapher "
                   "(HIV/DM/吸烟/营养不良)，2026-09-05 新增"),
        "n_years": len(out),
        "year_range": [int(out["year"].min()), int(out["year"].max())],
        "columns": list(out.columns),
        "usage_note": ("中国 TB 负担(WHO TME 口径,与 GHO 交叉核验) + 四大"
                     "宿主风险协变量的年度轨迹；SEIR 协变量敏感性分析用"),
    }
    with open(os.path.join(PROC, "tb_covariates_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] tb_covariates_china_timeseries.csv "
          f"({len(out)} 年 × {len(out.columns)} 列)")


if __name__ == "__main__":
    main()
