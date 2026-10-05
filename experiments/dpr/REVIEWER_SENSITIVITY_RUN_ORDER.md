# Additional reviewer sensitivities

Current accepted order/grid/timeline outputs are in evidence/publication_v017/analysis. Original execution sources are in reviewer_sources; their absolute paths are historical and are not portable entry points.

For a future full regeneration, use the existing isolated delivery run-all sequence first. In that generated source snapshot, copy evidence/historical_gates/temporal_deployment_20260825.json to data/processed/temporal_deployment_20260825.json. The same-information run supplies dpr_same_information_baseline_v1/seed_*.npz; the budget run supplies homeacf_dpr_budget_audit_v2.npz and JSON. These participant-level caches are deliberately NOT distributed. They must be regenerated from the pinned input.

Then run experiments/dpr/run_reviewer_sensitivities.py (60 fits across deterministic orders, plus parameter-surface calculation). Run experiments/dpr/run_reviewer_interval_harmonization_v012.py --project . afterwards to produce the accepted 2000-resample conditional interval procedure. The first script initially computes 1000-resample intervals; this is a documented intermediate, not the manuscript's final interval count. No automated best-rule selection or independent policy validation is introduced.

The new path adapters were syntax/dependency checked only; their full modeling sequence was NOT rerun in this round. Prior fits remain the evidence for the accepted outputs. Publication export is an immutable snapshot check, not regenerated statistical graphics. Historical plotting sources are provided, but some require manuscript-version files and path configuration; automatic complete regeneration of every publication diagram is not claimed.
