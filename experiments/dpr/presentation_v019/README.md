# v0.19 display reproduction

This supplementary software bundle redraws the five v0.19 main figures and the selected-rule supplementary figure from accepted aggregate analysis outputs. It does not train, tune, recalibrate or resample models, and it does not claim to reproduce all underlying analyses.

Requirements: Python 3 with NumPy and Matplotlib. The verified local drawing environment versions are recorded in verification.json. A fresh installation, Linux execution and complete scientific pipeline regeneration have not been tested for this revision.

Run from any directory: `python /path/to/presentation_reproduction/build_figures.py`. Inputs are resolved relative to the script. Output is written to `regenerated/figures`. Optionally set DPR_FIGURE_OUTPUT to a different output directory. Existing S1-S8 snapshots are copied; the original cohort flow is retained as an input. The supplied S10 is an unchanged copy of that original flow.

The CSV directory exports all four main tables and all 15 supplementary table blocks without changing their contents. Input JSON and source figure checksums are listed in manifest.json; original first-party terms are MIT and third-party terms remain unchanged.

Underlying analytical software archive: https://doi.org/10.5281/zenodo.23153355 (0.17-MIT-r1). Source project https://github.com/tiande888/tb_risk, base commit 3cc3c82f42dbf1e2bb8c3c4209e078a99db7c5c7. Development repository https://github.com/tb-risk-research/tb-risk-dpr-reproducibility. The v0.19 presentation revision is supplied here; it is not part of that immutable Zenodo snapshot.

No participant-level raw data, prediction caches, ethics originals or personal credentials are included. The main analytical repository documents pinned HomeACF acquisition and SHA-256 validation.
