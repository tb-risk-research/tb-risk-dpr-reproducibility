# DPR submission reproduction additions

This directory supports the v0.22 manuscript. It contains the HomeACF analyses added in v0.20 and the exploratory Brazilian evaluation added in v0.21. The v0.22 edits concern manuscript wording and reporting.

- [HomeACF additions](homeacf_additions/) contains the corrected-coding and per-fold calibration summaries and their scripts. The scripts resolve inputs and outputs from the project root.
- [Brazil analysis plan](brazil_external/protocol.md) and [aggregate outputs](brazil_external/results/) document the independent evaluation of reduced models fitted and frozen in HomeACF. The principal comparison uses logistic regression with and without an empirical-Bayes household rate, under simulated earlier-result availability. The paired AUROC difference was +0.095 (conditional 95% CI −0.012 to +0.206). The models were not recalibrated in Brazil.
- [v0.19 displays](../presentation_v019/) contains the earlier table and figure reproduction materials.
- [Zenodo 0.17-MIT-r1](https://doi.org/10.5281/zenodo.23153355) contains the frozen base analysis. The additional files in this directory postdate that archive.

## Reproduce the Brazilian analysis

From the repository root, prepare the [DPR Python environment](../requirements-dpr.txt) and install `openpyxl` for the source workbook. The evaluation was run with Python 3.12.14, numpy 1.26.4, pandas 2.2.3 and scikit-learn 1.5.2. Use a fresh clone or extraction for each rerun.

```powershell
$env:DPR_ROOT = (Get-Location).Path
$brazil = 'experiments/dpr/submission_reproduction/brazil_external'
$out = 'data/processed/dpr_external_validation_20261005'
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item "$brazil/protocol.md" "$out/protocol.md"
python "$brazil/acquire_dpr_external_20261005.py"
python "$brazil/run_dpr_external_20261005.py"
```

The acquisition script verifies both pinned source files against SHA-256. Compare the resulting aggregates with the supplied results. The recorded rerun reproduced the earlier predictions and summaries. The published artifact contains aggregate results only. Generated participant-level files remain in the local run directory. See the [Brazil source dataset](https://doi.org/10.1371/journal.pone.0206384.s002) and the [repository licensing notes](../../../LICENSING.md) for attribution and terms.
