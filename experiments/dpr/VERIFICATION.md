# Delivery verification, 2026-10-03

The curated source was copied into an extraction containing only the release-manifest files. A separate Python 3.12.14 venv with no system site-packages was created. Existing tested dependency distributions and their metadata were copied offline; pip check passed. The full project built as a wheel, installed, and imported outside the source directory. This is not a verified fresh installation from PyPI, a Linux validation, or an independent laboratory replication.

The extracted entry point fetched the pinned original HomeACF file over the network and verified its SHA-256. Fresh model fits passed all six selected check families: primary seeds 0/19; seven-arm same-information seed 0; nested calibration seed 0; budget metrics for seeds 0/19; A1/E1 first raw-household refits; D1 first LR dataset for four corner configurations. Comparisons use aggregate metrics, score sums and squared sums, not shipped participant predictions. Complete repetitions of all experiments were not rerun during handoff.

All 3,708 synthetic journal records were reconstructed, with 37 cell summaries, labels and Monte Carlo intervals checked. Seven figure PNGs matched the accepted figures pixel for pixel, and corresponding PDFs were generated. Fourteen accepted table snapshots were exported and compared cell by cell; this does not claim automatic manuscript-table rebuilding after refitting.

Seven runner contract tests passed, including overwrite protection, missing dependencies and failed/partial result rejection. Every packaged Python source passed syntax parsing. Full-run source entry points and dependency ordering were audited; tests do not imply that all original GUI, GNN, Cox or unrelated dataset modules have been independently validated.

The observed RuntimeWarning was traced to scikit-learn 1.5.2 _assert_all_finite calling sum on missing-value input before imputation. Observation probabilities were finite and inside (0,1), and selected IPW/refit outputs reproduced the aggregate reference within stated tolerances. Warnings were retained, not suppressed or presented as failed scientific results.

The private local audit folder retains setup, result, diagnostic and source-hash records. These checks do not establish clinical implementation, restore unavailable timestamps or identify causal mechanisms. No commit, push, public archive or submission was performed. Project-root copyright/permission notice remains an owner-confirmation item before public release; the existing pyproject declares MIT and original HomeACF notices are included.

## Independent-repository retargeting (2026-10-03)

The delivery target is now a standalone DPR paper-reproduction repository, with source https://github.com/tiande888/tb_risk and base 3cc3c82f42dbf1e2bb8c3c4209e078a99db7c5c7. Documentation and curated upload scope changed. Executable analysis code, installed-dependency pins, experiment protocols required for runtime and accepted results are retained byte for byte; previous selected fits remain valid evidence. No training was repeated. New checks cover manifest/dependency closure, CLI audit, snapshot construction and final ZIP hashes, not additional scientific replication.

Current local Git history and remote configuration remain unchanged; no new remote is created and no history is published. Historical source logs/drafts remain local but are excluded from the curated code ZIP. Full-history public upload requires an explicit history-scope review. Public code licence notice remains an owner-confirmation item. See VALIDATION_STATUS.json and INDEPENDENT_HANDOFF.md.

## Private team-repository upload (2026-10-03)

The preceding sections are historical verification records. The user has now authorised a clean initial commit to https://github.com/tb-risk-research/tb-risk-dpr-reproducibility (private). Source Git history and remotes remain intact locally. This step changes delivery metadata only; no scientific fitting is repeated. Upload checks cover the curated file tree, source hashes, CLI audit, commit inventory and remote commit identity. GitHub Actions is disabled. Public release, project permission scope and DOI remain pending.
