# Analysis reproduction for manuscript v0.20

This folder supplements the fixed analytical archive with the two summaries added in the
v0.20 revision.

- `per_fold_calibrated_auc.py` recomputes AUROC within each outer fold before and after
  nested Platt calibration. Platt scaling is monotone within a fold, so per-fold AUROC is
  unchanged; the script quantifies the within-fold values that the pooled out-of-fold
  figure obscures.
- `coding_parallel_capture.py` recomputes the corrected-coding analysis (parent-category
  fix, empty values coded as unknown, six missingness indicators) and adds capture at the
  5, 10, 20 and 30 percent budgets with tests per positive captured.

Both scripts call the frozen pipeline of the archived reproducibility project. Run from the project root or set
`DPR_REPRO_ROOT` to its path. Outputs are written under
`data/processed/dpr_v020_additions/` in that project. Aggregate result JSON files are included. Participant-level prediction caches
and the resampling arrays are omitted, consistent with the data-availability statement.
