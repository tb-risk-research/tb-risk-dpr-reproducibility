"""Corrected-coding parallel analysis with discrimination and budget capture.

Same cohort, households, folds, seeds, frozen RF settings and SOP construction
as experiments/dpr/run_feature_audit_sensitivity_v1.py (scenario='coding').
Adds capture at the same budget fractions used in the manuscript so that the
corrected-coding analysis can be reported alongside the frozen primary
analysis rather than only as an AUROC sensitivity.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPRO = os.path.abspath(os.environ.get("DPR_REPRO_ROOT", os.getcwd()))
sys.path.insert(0, os.path.join(REPRO, "data"))

import run_sop_unified_ci_leakfree as lf  # noqa: E402
import run_sop_robustness_battery as bat  # noqa: E402
import run_homeacf_tst_missingness_ipw as e1  # noqa: E402

OUTDIR = os.path.join(REPRO, "data", "processed", "dpr_v020_additions", "coding_parallel")
os.makedirs(OUTDIR, exist_ok=True)
e1.RDA = str(bat._RDA_PATH)
BUDGETS = (0.05, 0.10, 0.20, 0.30)
EXTRA = [
    "diabetes_missing",
    "airspace_missing",
    "timespent_missing",
    "relationship_missing",
    "index_hiv_unknown",
    "index_xpert_not_done",
]


def capture_metrics(y, score, q):
    n = max(1, int(np.ceil(len(y) * q)))
    ix = np.argsort(-score)[:n]
    captured = int(y[ix].sum())
    total = int(y.sum())
    return {
        "capture_rate": captured / total,
        "candidates": captured,
        "tests": n,
        "tests_per_captured": (n / captured) if captured else None,
    }


def one(seed):
    start = time.time()
    df = lf.load_homeacf_contacts()
    raw = e1.raw_data()
    raw = raw.loc[raw.tst_observed.eq(1)].reset_index(drop=True)
    assert np.array_equal(raw.contact_id.to_numpy(), df.contact_id.to_numpy())
    y = df.tst_pos10.to_numpy(int)
    g = df.record_id.to_numpy()
    cols = lf.inf_feature_columns("exposure")
    pf = lf.prior_features(df, lf.assign_screening_order(df, seed=seed, mode="random"))
    data = df.copy()
    data["bmi_h"] = bat.load_raw_bmi(df)
    data = pd.concat([data.reset_index(drop=True), pf], axis=1)
    data["prior_rate"] = np.where(pf.prior_n > 0, pf.prior_pos / np.maximum(pf.prior_n, 1), np.nan)

    data["rel_parent"] = raw.relationship_h.eq("Parent/Parent-in-law").astype(float)
    data["hiv_unknown_h"] = (
        raw.hivfinal_h.eq("c) HIV unknown") | raw.hivfinal_h.isna()
    ).astype(float)
    for new, source in zip(
        EXTRA[:4], ["diabetes_h", "airspace_h", "timespent_h", "relationship_h"]
    ):
        data[new] = raw[source].isna().astype(float)
    data["index_hiv_unknown"] = raw.hiv_i.eq("c) HIV-unknown").astype(float)
    data["index_xpert_not_done"] = raw.xpert_i.eq("c) Xpert not done").astype(float)
    cols = cols + EXTRA

    folds = lf._group_cv_indices(g, y, n_splits=5, seed=seed)
    for tr, te in folds:
        assert not set(g[tr]) & set(g[te])
    scores = np.array(
        [lf.rf_oof_lf(cs, folds, y, data, seed) for cs in [cols, cols + bat.COLS_PRIOR]]
    )
    np.savez_compressed(os.path.join(OUTDIR, f"coding_{seed:02d}.npz"), scores=scores, y=y, g=g)
    arms = []
    for s in scores:
        arms.append(
            {
                "auroc": float(roc_auc_score(y, s)),
                "auprc": float(average_precision_score(y, s)),
                "budgets": {str(q): capture_metrics(y, s, q) for q in BUDGETS},
            }
        )
    rec = {
        "seed": seed,
        "base": arms[0],
        "sop": arms[1],
        "delta_auroc": arms[1]["auroc"] - arms[0]["auroc"],
        "seconds": time.time() - start,
    }
    with open(os.path.join(OUTDIR, f"coding_{seed:02d}.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    return seed


def bootstrap_auc(scores, y, groups, multiplicities):
    """Tie-weighted cluster bootstrap AUC, identical to the archived routine."""
    out = np.empty((len(multiplicities), scores.shape[1]))
    for j in range(scores.shape[1]):
        order = np.argsort(scores[:, j], kind="mergesort")
        ss = scores[order, j]
        start = np.r_[0, np.flatnonzero(np.diff(ss) != 0) + 1]
        weights = multiplicities[:, groups[order]].astype(float)
        pos = np.add.reduceat(weights * y[order], start, axis=1)
        neg = np.add.reduceat(weights * (1 - y[order]), start, axis=1)
        den = pos.sum(axis=1) * neg.sum(axis=1)
        out[:, j] = (pos * (np.cumsum(neg, axis=1) - 0.5 * neg)).sum(axis=1) / den
    return out


def finish():
    import hashlib

    def sha(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest()

    rows = [
        json.load(open(os.path.join(OUTDIR, f"coding_{s:02d}.json"), encoding="utf-8"))
        for s in range(20)
    ]
    zs = [
        np.load(os.path.join(OUTDIR, f"coding_{s:02d}.npz"), allow_pickle=True)
        for s in range(20)
    ]
    y = zs[0]["y"]
    g = zs[0]["g"]
    _, gid = np.unique(g, return_inverse=True)
    h = gid.max() + 1
    draws = np.random.default_rng(91208).multinomial(h, np.full(h, 1 / h), size=2000)

    boots = np.array([bootstrap_auc(zz["scores"].T, y, gid, draws) for zz in zs]).mean(axis=0)
    delta_ci = np.quantile(boots[:, 1] - boots[:, 0], [0.025, 0.975]).tolist()

    summary = {}
    for arm in ("base", "sop"):
        for key in ("auroc", "auprc"):
            vals = [r[arm][key] for r in rows]
            summary[f"{arm}_{key}"] = {"mean": float(np.mean(vals)), "seed_range": [min(vals), max(vals)]}
        for q in BUDGETS:
            vals = [r[arm]["budgets"][str(q)]["capture_rate"] for r in rows]
            summary[f"{arm}_capture_q{q}"] = {
                "mean": float(np.mean(vals)),
                "seed_range": [min(vals), max(vals)],
            }
    summary["delta_auroc"] = {
        "mean": float(np.mean([r["delta_auroc"] for r in rows])),
        "conditional_ci95": delta_ci,
        "positive_seeds": int(sum(r["delta_auroc"] > 0 for r in rows)),
        "seed_range": [min(r["delta_auroc"] for r in rows), max(r["delta_auroc"] for r in rows)],
    }
    for q in BUDGETS:
        vals = [
            r["sop"]["budgets"][str(q)]["capture_rate"]
            - r["base"]["budgets"][str(q)]["capture_rate"]
            for r in rows
        ]
        summary[f"delta_capture_q{q}"] = {"mean": float(np.mean(vals)), "seed_range": [min(vals), max(vals)]}
    out = {
        "status": "complete",
        "n": int(len(y)),
        "events": int(y.sum()),
        "households": int(h),
        "seeds": 20,
        "scenario": "coding-corrected baseline columns, same folds/orders/RF settings as primary",
        "summary": summary,
        "limits": [
            "Conditional fixed-prediction intervals; no full refit.",
            "Coding correction and the availability sensitivity were not combined.",
            "Capture assumes earlier household results are available without acquisition cost.",
            "LR and calibration were not recomputed under corrected coding.",
        ],
    }
    with open(os.path.join(OUTDIR, "coding_parallel_result.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    start = time.time()
    todo = [s for s in range(20) if not os.path.exists(os.path.join(OUTDIR, f"coding_{s:02d}.json"))]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one, s) for s in todo]
        for f in as_completed(futures):
            print("done", f.result(), flush=True)
    finish()
    print("elapsed", time.time() - start)


if __name__ == "__main__":
    main()
