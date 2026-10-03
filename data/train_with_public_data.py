#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开数据训练编排：ML / GNN / SEIR 三模块训练 + TrainingLogger 归档

数据来源（data/processed/，由 process_public_data.py 生成）：
- ML : ml_training_semi_synthetic_anchored.csv（公开参数锚定半合成队列，见 anchors.json）
- GNN: 合成图 + Prem 中国接触矩阵校准说明（contact_mixing_summary.json）
- SEIR: WHO GHO 中国时序（seir_observed_china_2000_2024.csv）作为外部参照

工程约定：
- 所有训练出口（成功/失败）写 TrainingLogger（追加，不覆盖）
- run_id: 日期时间+关键超参；指标主键 AUROC
- 刷新历史最佳时通过 model_saver 回调保存权重快照

用法：
    python data/train_with_public_data.py --models ml
    python data/train_with_public_data.py --models gnn --gnn-samples 300 --gnn-epochs 30
    python data/train_with_public_data.py --models seir --mcmc-iters 2000
    python data/train_with_public_data.py --models ml,gnn,seir
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)          # .../tb_risk
DESKTOP = os.path.dirname(PROJECT_ROOT)       # tb_risk 包的父目录
sys.path.insert(0, DESKTOP)

PROC = os.path.join(HERE, "processed")


ARCHIVE_FALLBACK = os.path.join(HERE, "training_archive")


def get_logger():
    """TrainingLogger：优先项目约定的 ~/.tb_risk；沙箱不可写时回退项目内档案目录。

    回退目录结构与 ~/.tb_risk 完全一致（training_log.jsonl / best.json /
    results/ / models/），事后可直接整体合并回 ~/.tb_risk。

    沙箱注意事项：trae-sandbox 会把"触碰 ~/.tb_risk 下任何文件"判定为违规
    并直接终止进程（探测文件也不行），try/except 无法捕获。因此提供
    环境变量 TB_RISK_ARCHIVE_DIR：设置后完全不触碰 ~/.tb_risk，
    直接使用指定目录（沙箱内运行时设为 data/training_archive）。
    """
    from tb_risk.gui.training_panel.training_log import TrainingLogger
    env_dir = os.environ.get("TB_RISK_ARCHIVE_DIR", "").strip()
    if env_dir:
        return TrainingLogger(base_dir=env_dir)
    default_base = os.path.join(os.path.expanduser("~"), ".tb_risk")
    try:
        os.makedirs(default_base, exist_ok=True)
        probe = os.path.join(default_base, ".write_probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return TrainingLogger()
    except OSError as e:
        print(f"[警告] ~/.tb_risk 不可写（{type(e).__name__}），"
              f"训练档案暂存回退目录: {ARCHIVE_FALLBACK}（可事后合并回 ~/.tb_risk）")
        return TrainingLogger(base_dir=ARCHIVE_FALLBACK)


def _ml_metrics(predictor):
    """镜像 GUI _extract_ml_training_metrics：取各模型最优 AUROC 套件"""
    perf = getattr(predictor, "model_performance", {}) or {}
    candidates = []
    for key, info in perf.items():
        if not isinstance(info, dict) or not isinstance(info.get("AUROC"), (int, float)):
            continue
        candidates.append((float(info["AUROC"]), key, info))
    if not candidates:
        return {}
    _, best_key, best_info = max(candidates, key=lambda x: x[0])
    metrics = {}
    for name in ("AUROC", "AUPRC", "Brier", "Accuracy", "F1"):
        v = best_info.get(name)
        if isinstance(v, (int, float)):
            metrics[name] = float(v)
    metrics["best_model"] = best_info.get("name", best_key)
    metrics["n_models_evaluated"] = len(candidates)
    return metrics


def _ml_per_model_metrics(predictor):
    """全模型指标入档：5 个基模型各自 AUROC/AUPRC，判断是集成弱还是个别拖后腿"""
    perf = getattr(predictor, "model_performance", {}) or {}
    per_model = {}
    for key, info in perf.items():
        if not isinstance(info, dict) or not isinstance(info.get("AUROC"), (int, float)):
            continue
        per_model[key] = {
            "name": info.get("name", key),
            "AUROC": round(float(info["AUROC"]), 4),
            "AUPRC": round(float(info["AUPRC"]), 4) if isinstance(info.get("AUPRC"), (int, float)) else None,
        }
    return per_model


def _ml_isotonic_calibration(predictor, csv_path):
    """对 5 个基模型做概率校准（Platt 1999; Zadrozny & Elkan 2002）

    v3（2026-08-22，isotonic 常数塌缩修复）：
    - 方法按阳性数自适应：isotonic 是非参数校准，每个校准折需要足够
      阳性支撑。Kenya 0.53% 不平衡（336 阳性，CalibratedClassifierCV
      内部 cv=5 → 每折 ~67 阳性）下 isotonic 塌缩为常数预测器，
      五个模型校准后 Brier 全部 = π(1-π) = 0.0053（常数下界）——
      "Brier 改善"只是学会了全预测阴性，判别力被完全摧毁。
      规则：阳性 ≥500 → isotonic；100–499 → sigmoid（Platt，2 参数，
      极少阳性下远比 isotonic 稳健）；<100 → 跳过校准。
    - 退化检测：out-of-fold AUROC before/after 同时入档；after 相对
      before 跌幅 > 0.02 或绝对值 < 0.55 → 判定退化并拒绝采用该校准器
      （保留原始模型）。Brier 单指标会掩盖塌缩（Brier 变好、模型却废了），
      判别力必须作为采用门槛。

    返回：
        (calibrated_models, summary) — 校准器字典与
        {method, n_positive, {model: {brier/auroc_before/after, degraded}}}
        摘要；不可用时返回 ({}, {})
    """
    try:
        import numpy as np
        import pandas as pd
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import brier_score_loss
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        return {}, {}

    if not getattr(predictor, "is_trained", False) or not predictor.models:
        return {}, {}

    try:
        df = pd.read_csv(csv_path)
        feature_names = getattr(predictor, "ALL_FEATURE_NAMES", None)
        if feature_names is None or "tb_outcome" not in df.columns:
            return {}, {}
        # 数据集只含 13 基础特征时（如 kenya/nhanes 原始输出），补齐 9 个
        # 交互特征列——公式与 scoring/ml/training.py::train_from_real_data
        # 的自动计算段保持一致（修改任一处须同步另一处）
        if any(f not in df.columns for f in predictor.INTERACTION_FEATURE_NAMES):
            if 'past_illness_type' not in df.columns:
                df['past_illness_type'] = 'none'
                df.loc[df['past_illness'] == 1, 'past_illness_type'] = 'other'
            hiv = (df['past_illness_type'].isin(['hiv', 'HIV'])).astype(float)
            diabetes = (df['past_illness_type'].isin(['diabetes', '糖尿病'])).astype(float)
            immunosupp = (df['past_illness_type'].isin(['immunosuppressants', '免疫抑制'])).astype(float)
            immuno_score = (hiv * 2.0 + diabetes * 1.0 + immunosupp * 1.5
                            + df['past_illness'].astype(float) * 0.5)
            df['age_immuno'] = df['age'].astype(float) * immuno_score
            df['dm_tb_synergy'] = diabetes * df['has_tb'].astype(float)
            df['age_bcg_decay'] = df['age'].astype(float) * (1 - df['bcg_vaccine'].astype(float))
            delay = df.get('delay_days', df.get('ftd', 0))
            delay = delay.astype(float) if hasattr(delay, 'astype') else float(delay)
            df['symptom_delay'] = df['has_symptoms'].astype(float) * np.minimum(delay / 30, 1.0)
            cough = df.get('cough_frequency', df.get('cough_freq', 0))
            cough = cough.astype(float) if hasattr(cough, 'astype') else float(cough)
            contacts = df.get('contact_count', 5)
            contacts = contacts.astype(float) if hasattr(contacts, 'astype') else float(contacts)
            df['cough_contact'] = np.minimum(cough / 20, 1.0) * (contacts / 10)
            df['highrisk_comorbid'] = df['is_high_risk'].astype(float) * df['past_illness'].astype(float)
            df['immune_bcg'] = (1 - df['bcg_vaccine'].astype(float)) * immuno_score
            ce = df.get('cumulative_exposure', 0)
            ce = ce.astype(float) if hasattr(ce, 'astype') else float(ce)
            ts = df.get('time_span', 4)
            ts = ts.astype(float) if hasattr(ts, 'astype') else float(ts)
            df['exposure_accumulation'] = (ce / 80) * (ts / 10)
            df['age_diabetes'] = df['age'].astype(float) * diabetes
        X = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0).values
        y = pd.to_numeric(df["tb_outcome"], errors="coerce").fillna(0).values.astype(int)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        from sklearn.metrics import roc_auc_score

        # v3：方法按阳性数自适应（见 docstring）
        n_pos = int(np.asarray(y).sum())
        if n_pos >= 500:
            method = "isotonic"
        elif n_pos >= 100:
            method = "sigmoid"
        else:
            print(f"     [校准跳过] 阳性数 {n_pos} < 100，无法支撑任何校准器")
            predictor.calibrated_models = {}
            predictor._is_calibrated = False
            predictor._calibration_method = None
            return {}, {"skipped": f"n_positive={n_pos} < 100"}

        calibrated = {}
        summary = {"method": method, "n_positive": n_pos}
        for key, model in predictor.models.items():
            try:
                # out-of-fold 原始概率 → Brier + AUROC（判别力基线）
                oof_raw = cross_val_predict(
                    model, X, y, cv=cv, method="predict_proba")[:, 1]
                brier_before = float(brier_score_loss(y, oof_raw))
                auroc_before = float(roc_auc_score(y, oof_raw))

                cal = CalibratedClassifierCV(model, method=method, cv=5)
                cal.fit(X, y)
                oof_cal = cross_val_predict(
                    CalibratedClassifierCV(model, method=method, cv=5),
                    X, y, cv=cv, method="predict_proba")[:, 1]
                brier_after = float(brier_score_loss(y, oof_cal))
                auroc_after = float(roc_auc_score(y, oof_cal))

                # 退化检测：判别力是采用门槛，Brier 单指标会掩盖
                # "全预测阴性"式塌缩（Brier 变好、模型却废了）
                degraded = bool(auroc_after < auroc_before - 0.02
                                or auroc_after < 0.55)
                if not degraded:
                    calibrated[key] = cal
                summary[key] = {
                    "brier_before": round(brier_before, 4),
                    "brier_after": round(brier_after, 4),
                    "auroc_before": round(auroc_before, 4),
                    "auroc_after": round(auroc_after, 4),
                    "degraded": degraded,
                }
            except Exception as e:  # 单模型校准失败不阻断其余模型
                print(f"     [校准跳过] {key}: {type(e).__name__}: {e}")

        predictor.calibrated_models = calibrated
        predictor._is_calibrated = bool(calibrated)
        predictor._calibration_method = method if calibrated else None
        return calibrated, summary
    except Exception as e:
        print(f"     [校准失败] {type(e).__name__}: {e}")
        return {}, {}


def train_ml(args):
    dataset = getattr(args, "ml_dataset", "semi_synthetic")
    # 组感知 CV 组列映射（任务1，2026-09-14）：有户/社区结构的队列显式
    # 指定组列（train_from_real_data 也会自动探测，这里显式化便于审计）；
    # 其余队列无组结构，维持随机分层 CV（正确行为，非遗漏）。
    group_column = None
    if dataset == "nhanes":
        csv_path = os.path.join(PROC, "ml_training_nhanes_real.csv")
        meta_path = os.path.join(PROC, "nhanes_dataset_meta.json")
        label = "NHANES 真实个体数据（IGRA LTBI 标签）"
        real_data = True
    elif dataset == "treats":
        csv_path = os.path.join(PROC, "ml_training_treats_real.csv")
        meta_path = os.path.join(PROC, "treats_dataset_meta.json")
        label = "TREATS 队列真实个体数据（赞比亚+南非 14 社区，QFT-Plus LTBI 标签）"
        real_data = True
        group_column = "group_id"  # 14 社区（社区内特征/标签相关）
    elif dataset == "crp":
        csv_path = os.path.join(PROC, "ml_training_crp_real.csv")
        meta_path = os.path.join(PROC, "crp_dataset_meta.json")
        label = ("LSHTM CRP 社区筛查（赞比亚+南非 ACF，验证子集 "
                 "Xpert/培养确诊标签）")
        real_data = True
    elif dataset == "kenya":
        csv_path = os.path.join(PROC, "kenya_ml_training.csv")
        meta_path = os.path.join(PROC, "kenya_dataset_meta.json")
        label = "肯尼亚 2016 全国 TB 患病率调查（细菌学确诊标签）"
        real_data = True
    elif dataset == "taiwan":
        csv_path = os.path.join(PROC, "ml_training_taiwan_real.csv")
        meta_path = os.path.join(PROC, "taiwan_dataset_meta.json")
        label = ("台湾 NTUH QFT-Plus 队列评估组（活动性 TB 标签，"
                 "QFT 定量 + CXR 评分 + 合并症）")
        real_data = True
    elif dataset == "brazil":
        csv_path = os.path.join(PROC, "ml_training_brazil_real.csv")
        meta_path = os.path.join(PROC, "brazil_dataset_meta.json")
        label = ("巴西东北部 IGRA 队列（LTBI 标签，宿主特征 + "
                 "IFNG+874 基因型；IFN-γ 定量防泄漏排除）")
        real_data = True
    elif dataset == "peru_mdr":
        csv_path = os.path.join(PROC, "ml_training_peru_mdr_real.csv")
        meta_path = os.path.join(PROC, "peru_mdr_dataset_meta.json")
        label = ("秘鲁 MDR vs 敏感 TB 家庭接触者 3 年前瞻队列"
                 "（感染→病程终点：incident TB，户级聚类）")
        real_data = True
        group_column = "family_id"  # 688 户（户内接触者强相关，防泄漏）
    elif dataset == "scenario":
        # 场景对齐训练：部署特征管线 + 机制标签（GNN 场景对齐同一配方）。
        # 背景：Kenya/半合成 ML 在部署场景评估 ml_only 仅 0.47/0.62
        # （域迁移断崖）；本数据集特征与部署打分路径完全一致。
        # 训练 seed 2026，与集成评估 seed 42 数据不重叠但分布一致。
        # --ml-scenario-label clinical：成员多样性工程——ML 用临床
        # 规则标签训练（与 GNN 的机制标签错开），降低成员错误相关。
        from evaluate_ensemble import generate_scenario_ml_csv
        scen_label = getattr(args, "ml_scenario_label", "mechanistic")
        if scen_label == "clinical":
            csv_path = os.path.join(
                PROC, "ml_training_scenario_clinical.csv")
            meta_path = os.path.join(
                PROC, "ml_training_scenario_clinical_meta.json")
        else:
            csv_path = os.path.join(PROC, "ml_training_scenario_aligned.csv")
            meta_path = os.path.join(PROC, "ml_training_scenario_meta.json")
        n_gen = generate_scenario_ml_csv(
            2500, 2026, csv_path, meta_path, label_mode=scen_label)
        print(f"  [场景生成] {n_gen[1]} 个接触者样本"
              f"（label={scen_label}）-> {os.path.basename(csv_path)}")
        label = (f"部署场景对齐（评估同款场景构造 + "
                 f"{'临床规则' if scen_label == 'clinical' else '机制传播'}标签）")
        real_data = False
    else:
        csv_path = os.path.join(PROC, "ml_training_semi_synthetic_anchored.csv")
        meta_path = os.path.join(PROC, "ml_training_anchors.json")
        label = "公开参数锚定半合成队列"
        real_data = False
    print(f"=== ML 训练（{label}） ===")
    from tb_risk.scoring.predictor import MLRiskPredictor

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    predictor = MLRiskPredictor()
    # P4（标签代际，2026-08-24）：场景谱系显式声明标签机制代际，
    # 随 checkpoint 落盘供 load_model 代际校验（防错代加载）
    label_generation = None
    if dataset == "scenario":
        label_generation = ("clinical-rule-v1" if scen_label == "clinical"
                            else "v3-host-pathway")
    start = time.time()
    result = predictor.train_from_real_data(
        csv_path, target_column="tb_outcome",
        enable_hyperopt=getattr(args, "ml_hyperopt", False),
        label_generation=label_generation,
        group_column=group_column)
    duration = time.time() - start

    dataset_info = {
        "source": os.path.basename(csv_path),
        "source_path": csv_path,
        "n_samples": result.get("n_samples", 0),
        "n_positive": result.get("n_positive", 0),
        "positive_rate": result.get("positive_rate", 0.0),
        "real_data": real_data,
        "provenance": (
            "CDC NCHS NHANES 2011-2012 public microdata; LTBI outcome = "
            "QuantiFERON Gold in-Tube official interpretation (LBXTBIN==1); "
            "variable map & disclosure in nhanes_dataset_meta.json"
            if dataset == "nhanes" else
            "LSHTM Data Compass TREATS cohort (DOI 10.17037/DATA.00003627, "
            "CC BY 4.0, no registration); Zambia+South Africa 14 communities, "
            "ages 15-24; LTBI outcome = QuantiFERON Gold Plus official "
            "interpretation (qft==1); variable map & disclosure in "
            "treats_dataset_meta.json"
            if dataset == "treats" else
            "LSHTM Data Compass CRP community screening open version "
            "(DOI 10.17037/DATA.00005150, CC BY 4.0); Zambia+South Africa "
            "ACF 2019; verified subset (sputum_elig = symptoms or CAD>=40); "
            "outcome = Xpert or culture positive; verification-bias "
            "disclosure in crp_dataset_meta.json"
            if dataset == "crp" else
            "Kenya National TB Prevalence Survey 2016 (PLOS ONE "
            "doi:10.1371/journal.pone.0209098, public S1/S2); outcome = "
            "bacteriologically confirmed TB (smear/Xpert/culture any POS); "
            "variable map & disclosure in kenya_dataset_meta.json"
            if dataset == "kenya" else
            "Mendeley Data Taiwan NTUH QFT-Plus cohort (doi:10.17632/"
            "457bkx9p6n.2, CC BY 4.0); evaluated subset (TB outcome "
            "non-missing, n=129); outcome = active TB (confirmed 1 / "
            "clinical 2); features incl. QFT-Plus quantitative, CXR "
            "score, 7 comorbidities; disclosure in taiwan_dataset_meta.json"
            if dataset == "taiwan" else
            "Mendeley Data Brazil IGRA cohort (doi:10.17632/mfcwvxwrhc.1, "
            "CC BY 4.0, Carneiro 2018); outcome = IGRA positive (LTBI); "
            "leakage exclusion: IFN-gamma quantitative columns and Group "
            "(encodes TST) not used as features; host features + IFNG+874 "
            "genotype; disclosure in brazil_dataset_meta.json"
            if dataset == "brazil" else
            "Dryad Peru MDR vs drug-susceptible TB household contact "
            "cohort (doi:10.5061/dryad.br760, CC0, Grandjean 2016); "
            "3-year prospective follow-up; outcome = incident active TB "
            "(n=149/3406); features incl. age band midpoint, gender, "
            "diabetes, HIV, previous TB, bedroom sharing, index smear "
            "grade, MDR exposure; household clustering (688 families) "
            "disclosed — CV is not group-aware, family_id column retained "
            "for GroupKFold reanalysis; disclosure in "
            "peru_mdr_dataset_meta.json"
            if dataset == "peru_mdr" else
            "deployment-form scenario contacts; same generator & feature "
            "pipeline as ensemble evaluation (seed=2026, "
            f"{getattr(args, 'ml_scenario_label', 'mechanistic')} "
            "labels); disclosure in ml_training_scenario_meta.json"
            if dataset == "scenario" else
            "public-anchored semi-synthetic cohort; "
            "anchors & citations in ml_training_anchors.json"),
    }
    params = {"n_samples": result.get("n_samples", 0),
              "data_source": os.path.basename(csv_path),
              "ml_cv_folds": getattr(predictor, "cv_folds", 5),
              # 组感知协议入档（任务1）：档案可审计本次 AUROC 的 CV 协议
              "cv_protocol": result.get("cv_protocol"),
              "group_column": result.get("group_column"),
              "n_groups": result.get("n_groups")}

    logger = get_logger()
    if result.get("success"):
        metrics = _ml_metrics(predictor)
        print(f"[DEBUG] ml_hyperopt={getattr(args, 'ml_hyperopt', 'MISSING')} "
              f"perf_keys={list((getattr(predictor, 'model_performance', {}) or {}).keys())}")
        # 全模型指标入档：判断是集成弱还是个别基模型拖后腿
        per_model = _ml_per_model_metrics(predictor)
        if per_model:
            params["per_model"] = per_model
        # isotonic 概率校准：out-of-fold Brier 前后对比
        calibrated, cal_summary = _ml_isotonic_calibration(predictor, csv_path)
        if cal_summary:
            params["isotonic_calibration"] = cal_summary
        if getattr(args, "ml_hyperopt", False):
            params["enable_hyperopt"] = True
        entry = logger.log(
            model_type="ml", params=params, metrics=metrics,
            training_duration=duration, status="success",
            dataset=dataset_info,
            model_saver=(lambda p: bool(predictor.save_model(p)))
            if hasattr(predictor, "save_model") else None)
        print(f"[OK] ML 训练成功 run_id={entry.run_id}")
        for k, v in metrics.items():
            print(f"     {k}: {v}")
        if per_model:
            print("     各基模型 AUROC/AUPRC:")
            for key, info in per_model.items():
                print(f"       {info['name']}: AUROC={info['AUROC']}, AUPRC={info['AUPRC']}")
        if cal_summary:
            if "method" in cal_summary:
                print(f"     [校准] 方法={cal_summary['method']}, "
                      f"阳性={cal_summary['n_positive']}")
            for key, s in cal_summary.items():
                if not isinstance(s, dict):
                    continue
                flag = " [退化→弃用]" if s.get("degraded") else ""
                print(f"     [校准] {key}: "
                      f"AUROC {s['auroc_before']}→{s['auroc_after']}, "
                      f"Brier {s['brier_before']}→{s['brier_after']}{flag}")
    else:
        entry = logger.log(
            model_type="ml", params=params, training_duration=duration,
            status="failed",
            error_message=result.get("diagnostics", "unknown"),
            dataset=dataset_info)
        print(f"[FAIL] ML 训练失败: {result.get('diagnostics')}")
        print(f"       run_id={entry.run_id}")
    return result.get("success", False)


def train_gnn(args):
    print("=== GNN 训练（合成图 + Prem 中国接触矩阵校准参照） ===")
    from tb_risk.scoring.predictor import MLRiskPredictor

    summary_path = os.path.join(PROC, "contact_mixing_summary.json")
    contact_note = ""
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        contact_note = (f"family_share={s['family_share_of_contacts']:.2f}, "
                        f"social_share={s['social_share_of_contacts']:.2f}")

    predictor = MLRiskPredictor()
    params = {
        "gnn_learning_rate": 0.001,
        "gnn_n_epochs": args.gnn_epochs,
        "n_samples": args.gnn_samples,
        "gnn_enable_hyperopt": args.gnn_hyperopt,
        "gnn_optuna_trials": args.gnn_trials,
        "gnn_num_layers": args.gnn_layers,
        "gnn_hidden_dim": args.gnn_hidden_dim,
        "gnn_dropout": args.gnn_dropout,
        "contact_matrix_calibration": contact_note,
    }
    start = time.time()
    # 场景对齐训练：训练图直接复用集成评估的场景构造逻辑
    # （同一特征编码/图结构/标签机制），消除训练/部署分布断崖
    # （此前训练内 AUROC 0.657 → 部署场景 0.533 的域迁移损失）。
    # 训练 seed 固定 2026，与评估 seed 42 数据不重叠但分布一致。
    graph_generator = None
    if getattr(args, "gnn_scenario_align", False):
        from evaluate_ensemble import generate_scenario_graphs
        graph_generator = generate_scenario_graphs
        params["scenario_align"] = True
        params["scenario_train_seed"] = 2026
    result = predictor.train_gnn(
        n_samples=args.gnn_samples,
        random_state=2026 if graph_generator is not None else 42,
        enable_hyperopt=args.gnn_hyperopt,
        n_epochs=args.gnn_epochs,
        learning_rate=0.001,
        use_focal_loss=True,
        two_phase_training=True,
        use_optuna=True,
        n_optuna_trials=args.gnn_trials,
        gnn_num_layers=args.gnn_layers,
        gnn_hidden_dim=args.gnn_hidden_dim,
        dropout=args.gnn_dropout,
        use_mixed_precision=False,   # CPU 训练
        log_callback=lambda m: print(f"     [gnn] {m}"),
        graph_generator=graph_generator,
    )
    duration = time.time() - start

    dataset_info = {
        "source": "scenario_aligned_graphs" if graph_generator
        else "synthetic_graphs",
        "n_samples": args.gnn_samples,
        "real_data": False,
        "provenance": ("deployment-form scenario graphs (same generator as "
                       "ensemble evaluation; mechanistic labels)"
                       if graph_generator else
                       "synthetic contact graphs; layer intensity referenced to "
                       "Prem/Mistry China contact matrices (see "
                       "processed/contact_mixing_summary.json)"),
    }
    perf = getattr(predictor, "model_performance", {}) or {}
    info = perf.get("gnn") or {}
    metrics = {}
    for name in ("AUROC", "AUPRC", "Brier", "Accuracy", "F1"):
        v = info.get(name)
        if isinstance(v, (int, float)):
            metrics[name] = float(v)
    # 约登阈值与收敛诊断（v2：区分"阈值假象"与"真失败"）
    for name in ("threshold", "youden_index", "best_val_auroc",
                 "early_stopped", "epochs_run", "positive_rate"):
        v = info.get(name)
        if isinstance(v, (int, float, bool)):
            metrics[f"gnn_{name}"] = float(v)
    curve = info.get("training_curve")
    if isinstance(curve, list) and curve:
        idx = [0, len(curve) // 2, len(curve) - 1]
        metrics["gnn_training_curve_anchor"] = [curve[i] for i in idx]

    logger = get_logger()
    if result:
        entry = logger.log(
            model_type="gnn", params=params, metrics=metrics,
            training_duration=duration, status="success",
            dataset=dataset_info,
            model_saver=(lambda p: bool(predictor.save_gnn_model(p)))
            if hasattr(predictor, "save_gnn_model") else None)
        print(f"[OK] GNN 训练成功 run_id={entry.run_id}")
        for k, v in metrics.items():
            print(f"     {k}: {v}")
    else:
        err = getattr(predictor, "_last_gnn_train_error", None)
        entry = logger.log(
            model_type="gnn", params=params, training_duration=duration,
            status="failed", error_message=str(err) if err else "unknown",
            dataset=dataset_info)
        print(f"[FAIL] GNN 训练失败 run_id={entry.run_id}")
    return bool(result)


def train_seir(args):
    print("=== SEIR 贝叶斯标定（WHO GHO 中国时序作外部参照） ===")
    import numpy as np
    from tb_risk.karamay.localizer import KaramayLocalizer
    from tb_risk.karamay.bayesian_calibrator import BayesianCalibrator

    # WHO 中国发病率（外部参照，写入日志做归因）
    who_ref = {}
    obs_path = os.path.join(PROC, "seir_observed_china_2000_2024.csv")
    with open(obs_path, encoding="utf-8") as f:
        rows = [r for r in __import__("csv").DictReader(f)]
    for r in rows:
        if r["year"] in ("2020", "2023", "2024"):
            who_ref[r["year"]] = {
                "incidence_per_100k": round(float(r["incidence_per_100k"]), 2),
                "est_incident_cases": int(float(r["est_incident_cases"])),
            }

    localizer = KaramayLocalizer()
    calibrator = BayesianCalibrator(localizer)
    params = {
        "mcmc_iterations": args.mcmc_iters,
        "mcmc_burnin": args.mcmc_iters // 2,
        "mcmc_chains": args.mcmc_chains,
        "who_china_reference": who_ref,
        "prior_base_incidence_note": (
            "先验中心 121/10万（克拉玛依 2024 卫健委）；WHO 中国 2023 约 "
            f"{who_ref.get('2023', {}).get('incidence_per_100k', '?')}/10万 "
            "作为全国外部参照（新疆高于全国约 2-3 倍）"),
    }
    start = time.time()
    result = calibrator.calibrate(
        n_iterations=args.mcmc_iters,
        n_burnin=args.mcmc_iters // 2,
        n_chains=args.mcmc_chains,
        random_seed=42)
    duration = time.time() - start

    summary = result.summary() if hasattr(result, "summary") else {}
    metrics = {}
    if isinstance(summary, dict):
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                metrics[f"seir_{k}"] = float(v)
    metrics["acceptance_rate"] = float(getattr(result, "acceptance_rate", 0) or 0)
    # 收敛诊断：接受率合格 ≠ 收敛，还需 R-hat<1.1 且 ESS>100（Vehtari 2021）
    r_hat = getattr(result, "r_hat", None) or {}
    ess = getattr(result, "ess", None) or {}
    if r_hat:
        metrics["max_rhat"] = float(max(r_hat.values()))
    if ess:
        metrics["min_ess"] = float(min(ess.values()))
    metrics["converged"] = bool(
        metrics.get("max_rhat", 99) < 1.1 and metrics.get("min_ess", 0) > 100)

    logger = get_logger()
    dataset_info = {
        "source": "seir_observed_china_2000_2024.csv (external reference)",
        "source_path": obs_path,
        "n_samples": len(rows),
        "real_data": True,   # WHO GHO 真实时序（作外部参照）
        "provenance": ("WHO GHO China TB time series 2000-2024 (TB_e_inc_num 等); "
                       "校准目标为区县筛查阳性率（默认克拉玛依区县数据）"),
    }
    entry = logger.log(
        model_type="seir", params=params, metrics=metrics,
        training_duration=duration, status="success", dataset=dataset_info)
    print(f"[OK] SEIR 标定完成 run_id={entry.run_id}")
    pm = getattr(result, "posterior_mean", {}) or {}
    for k in ("base_incidence_per_100k", "latent_prevalence", "oilfield_camp_factor"):
        if k in pm:
            print(f"     后验均值 {k}: {pm[k]:.3f}")
    print(f"     接受率: {metrics.get('acceptance_rate', 0):.2%}")
    if "max_rhat" in metrics:
        print(f"     收敛: max_R-hat={metrics['max_rhat']:.3f} "
              f"min_ESS={metrics.get('min_ess', 0):.0f} "
              f"converged={metrics['converged']}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="ml,gnn,seir",
                        help="逗号分隔: ml,gnn,seir")
    parser.add_argument("--ml-dataset", default="semi_synthetic",
                        choices=["semi_synthetic", "nhanes", "treats",
                                 "crp", "kenya", "taiwan", "brazil",
                                 "peru_mdr", "scenario"],
                        help="ML 训练数据集：nhanes=美国真实个体数据(IGRA LTBI)；"
                             "treats=赞比亚+南非社区队列(QFT-Plus LTBI)；"
                             "crp=赞比亚+南非社区ACF筛查(细菌学确诊,验证子集)；"
                             "kenya=肯尼亚患病率调查(细菌学确诊)；"
                             "taiwan=台湾NTUH评估组(活动性TB,QFT+CXR)；"
                             "brazil=巴西IGRA队列(LTBI,含IFNG基因型)；"
                             "peru_mdr=秘鲁MDR家庭接触者3年前瞻队列"
                             "(incident TB,感染→病程终点)；"
                             "scenario=部署场景对齐(消除域迁移断崖)")
    parser.add_argument("--ml-scenario-label",
                        choices=["mechanistic", "clinical"], default="mechanistic",
                        help="scenario 数据集标签机制：clinical=临床规则"
                             "（成员多样性工程——与 GNN 的机制标签错开，"
                             "降低集成成员错误相关性）")
    parser.add_argument("--gnn-samples", type=int, default=2000)
    parser.add_argument("--gnn-epochs", type=int, default=200)
    parser.add_argument("--gnn-trials", type=int, default=8)
    parser.add_argument("--gnn-layers", type=int, default=3,
                        help="GNN 层数（网格搜索显示 2 层优于默认 3 层）")
    parser.add_argument("--gnn-hidden-dim", type=int, default=64,
                        help="GNN 隐藏维度")
    parser.add_argument("--gnn-dropout", type=float, default=0.3,
                        help="GNN dropout 率")
    parser.add_argument("--gnn-hyperopt", action="store_true", default=False)
    parser.add_argument("--gnn-scenario-align", action="store_true", default=False,
                        help="GNN 场景对齐训练：训练图复用集成评估的"
                             "场景构造逻辑（同特征编码/图结构/标签机制），"
                             "消除训练/部署分布断崖")
    parser.add_argument("--mcmc-iters", type=int, default=8000)
    parser.add_argument("--mcmc-chains", type=int, default=4)
    parser.add_argument("--ml-hyperopt", action="store_true", default=False,
                        help="ML 5 基模型逐个网格搜索（默认关）")
    args = parser.parse_args()

    selected = [s.strip() for s in args.models.split(",") if s.strip()]
    outcomes = {}
    if "ml" in selected:
        outcomes["ml"] = train_ml(args)
    if "gnn" in selected:
        outcomes["gnn"] = train_gnn(args)
    if "seir" in selected:
        outcomes["seir"] = train_seir(args)

    print("\n=== 汇总 ===")
    for k, v in outcomes.items():
        print(f"  {k}: {'成功' if v else '失败'}")


if __name__ == "__main__":
    main()
