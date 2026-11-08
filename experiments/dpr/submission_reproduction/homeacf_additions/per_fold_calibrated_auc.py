"""Per-fold AUROC for the nested-calibration analysis (v0.20 revision).

Mirrors experiments/dpr/run_nested_calibration_v1.py exactly (same folds,
same inner Platt fitting, same frozen RF settings) but records AUROC within
each outer fold before and after calibration. The pooled OOF AUROC reported
in v0.19 concatenates five fold-specific monotone transforms on a common
score axis; this script quantifies the within-fold ranking that the pooled
number obscures.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.special import logit

REPRO = os.path.abspath(os.environ.get("DPR_REPRO_ROOT", os.getcwd()))
sys.path.insert(0, os.path.join(REPRO, "data"))

import run_sop_unified_ci_leakfree as lf  # noqa: E402
import run_sop_robustness_battery as bat  # noqa: E402

OUT = os.path.join(REPRO, "data", "processed", "dpr_v020_additions", "per_fold_calibrated_auc.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ARMS = ["exposure", "exposure_prior"]


def main():
    start = time.time()
    df = lf.load_homeacf_contacts()
    y = df.tst_pos10.to_numpy(dtype=int)
    g = df.record_id.to_numpy()
    bmi = bat.load_raw_bmi(df)

    rows = []
    for seed in range(N_SEEDS):
        order = lf.assign_screening_order(df, seed=seed, mode="random")
        pf = lf.prior_features(df, order)
        data = df.copy()
        data["bmi_h"] = bmi
        data = pd.concat([data.reset_index(drop=True), pf], axis=1)
        data["prior_rate"] = np.where(
            pf.prior_n > 0, pf.prior_pos / np.maximum(pf.prior_n, 1), np.nan
        )
        outer = lf._group_cv_indices(g, y, n_splits=5, seed=seed)
        pooled = {a: {"raw": np.zeros(len(y)), "cal": np.zeros(len(y))} for a in ARMS}
        fold_records = []
        for f, (tr, te) in enumerate(outer):
            assert not set(g[tr]) & set(g[te])
            inner_seed = 100000 + seed * 5 + f
            inner = lf._group_cv_indices(g[tr], y[tr], n_splits=4, seed=inner_seed)
            for arm, name in enumerate(ARMS):
                cols = lf.inf_feature_columns("exposure") + (bat.COLS_PRIOR if arm else [])
                inner_oof = lf.rf_oof_lf(cols, inner, y[tr], data.iloc[tr], seed)
                calibrator = LogisticRegression(C=1e6, max_iter=2000)
                calibrator.fit(logit(np.clip(inner_oof, 1e-6, 1 - 1e-6))[:, None], y[tr])
                X, _, _ = lf._foldclean_matrix(cols, [(tr, te)], y, data)[0]
                model = bat._make_model("random_forest", seed)
                model.fit(X[tr], y[tr])
                p = model.predict_proba(X[te])[:, 1]
                cp = calibrator.predict_proba(
                    logit(np.clip(p, 1e-6, 1 - 1e-6))[:, None]
                )[:, 1]
                pooled[name]["raw"][te] = p
                pooled[name]["cal"][te] = cp
                fold_records.append(
                    {
                        "seed": seed,
                        "fold": f,
                        "arm": name,
                        "n": int(len(te)),
                        "events": int(y[te].sum()),
                        "auroc_raw": float(roc_auc_score(y[te], p)),
                        "auroc_cal": float(roc_auc_score(y[te], cp)),
                    }
                )
            print(f"seed {seed} fold {f + 1}/5", flush=True)
        row = {"seed": seed, "fold_records": fold_records}
        for name in ARMS:
            row[name] = {
                "pooled_auroc_raw": float(roc_auc_score(y, pooled[name]["raw"])),
                "pooled_auroc_cal": float(roc_auc_score(y, pooled[name]["cal"])),
                "mean_fold_auroc_raw": float(
                    np.mean([r["auroc_raw"] for r in fold_records if r["arm"] == name])
                ),
                "mean_fold_auroc_cal": float(
                    np.mean([r["auroc_cal"] for r in fold_records if r["arm"] == name])
                ),
            }
        rows.append(row)

    summary = {}
    for name in ARMS:
        for kind in ["pooled_auroc_raw", "pooled_auroc_cal", "mean_fold_auroc_raw", "mean_fold_auroc_cal"]:
            vals = [r[name][kind] for r in rows]
            summary[f"{name}__{kind}"] = {
                "mean": float(np.mean(vals)),
                "seed_min": float(np.min(vals)),
                "seed_max": float(np.max(vals)),
            }
    for kind in ["auroc_raw", "auroc_cal"]:
        vals = [r["auroc_raw"] for row in rows for r in row["fold_records"] if r["arm"] == "exposure_prior"]
        summary[f"fold_level_exposure_prior__{kind}"] = {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_folds": len(vals),
        }
    for kind in ["auroc_raw", "auroc_cal"]:
        vals = [r["auroc_raw"] for row in rows for r in row["fold_records"] if r["arm"] == "exposure"]
        summary[f"fold_level_exposure__{kind}"] = {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_folds": len(vals),
        }

    out = {
        "status": "complete",
        "seeds": N_SEEDS,
        "n": int(len(y)),
        "events": int(y.sum()),
        "households": int(len(np.unique(g))),
        "design": "Same outer 5-fold household CV, inner 4-fold Platt, frozen RF of run_nested_calibration_v1.py",
        "summary": summary,
        "rows": rows,
        "runtime_seconds": time.time() - start,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
