# DPR external transport assessment — frozen before model fitting

This is an exploratory addition to the existing DPR study, not a registered
prospective study, not the OSF ESRU5 external-validation protocol, and not a
validation of the original 29-predictor model. Aggregate sample sizes, outcome
counts, and variable availability were examined before this plan was written.
No prediction results from these new analyses had been examined.

## Question and cohorts

Does a source-trained, frozen portable representation using assumed earlier
household TST results improve prediction of baseline TST >=10 mm in an
independently collected Brazilian household cohort?

- Source: HomeACF public tstsa extract, South Africa; retain observed TST only.
- External target: Acuña-Villaorduña et al., PLOS ONE 2018,
  doi:10.1371/journal.pone.0206384, supporting file S2, HHCs sheet.
  Use ppd1 >=10 mm, indexid as household, contactid as participant.
  Do not use ppd2, IGRA, outcome classifications, or aerosol/culture results
  as predictors. The extract has 230 contacts, 48 households, 154 positives;
  missing-TST contacts excluded upstream cannot be reconstructed.
- The target lacks most original predictors. The four shared baseline
  variables are contact age, contact male sex, index age, index male sex.
  This reduced source-trained model is a separately specified portable
  extension. No coefficient, imputation statistic, or EB parameter is fitted
  on Brazil in the frozen external evaluation.

## Models and ordering

Primary comparison: LR baseline versus LR baseline plus empirical-Bayes
household positivity rate. Representation comparison: LR plus four prior
variables (number, positives, raw rate, any-result indicator). RF analogues
are secondary, with original RF defaults: 300 trees, depth 7, split 5,
leaf 2, balanced class weights. LR uses source-trained median imputation,
standardization, C=1, unbalanced L2 logistic regression, max_iter=2000.
No tuning, target-based model selection, or target calibration in primary
evaluation. EB parameters follow the existing run_sop_deepening.py estimator.

Train six models once on all source observations with simulated household
ordering seed 0. Freeze source medians, prevalence, EB strength and coefficients.
Apply each frozen model under 20 prespecified target random household orders
(seeds 0–19); average metrics, not predictions, across orders. Structured
sensitivity: contact age ascending, descending, and frozen LR baseline risk
descending; ties use stable participant input order, missing age uses source
median. Only results at strictly earlier positions enter each prediction;
the target's own or future outcome is never a predictor for that target.
Availability of earlier results is assumed, not verified by real timestamps.

## Metrics and uncertainty

Primary: external delta AUROC, LR+EB minus LR baseline, averaged across the
20 random orders; two-sided 95% paired household-bootstrap percentile interval,
2,000 draws, seed 1052026. A positive estimate with lower bound >0 is evidence
of improved external discrimination for this portable extension. Otherwise
record the uncertainty or deterioration, without deleting the cohort.
RF results and representation comparisons are secondary; no multiplicity-adjusted
confirmatory claims are made. Intervals condition on the frozen source-trained
models and the fixed ordering set; they do not cover refitting in HomeACF.

Also report AUROC, average precision (AUPRC), Brier score, log loss, ECE
(10 equal-width bins), calibration intercept/slope as evaluation diagnostics,
mean predicted risk, prevalence, and free-evidence captures at 10/20/30%.
Free-evidence scenarios do not charge acquisition of earlier results, so are
scenario upper bounds, not deployable strategies. Ties at a budget boundary
receive fractional allocation. Calibration fits are diagnostics only and do
not replace the frozen model's predictions.

Sensitivity: complete-baseline-covariate subset without refitting. Separately
evaluate LR arms by household-grouped 5-fold CV over 20 seeds in HomeACF and
Brazil. These are local internal validations, never called external validation.
Train-fold preprocessing and EB parameters only; all members of each household
remain in one fold. External and internal results remain separately labelled.

## Data gates and exclusions

Check identity uniqueness, household definitions, coding dictionary, mm units,
outcome completeness, admissible baseline variables and feature leakage.
Do not interpret this small target as adequately powered definitive validation.
McIntosh 2019 Brazil/Uganda supporting files combine baseline and follow-up
positivity and do not expose separate baseline measurements; exclude them
from this same-endpoint exercise. Carabayllo has only 73 measured TSTs/13
positives, age categories and missing shared variables; exclude for endpoint
availability and model compatibility. India CTRIUMPH was not accessible as
immediate public individual-level data. Selection is based on compatibility,
not the prediction results. No manuscript is overwritten or remotely published.

## Reproducibility and scope

Save code, source URLs and SHA-256 hashes, frozen model objects, local prediction
caches, software versions, progress, aggregate tables, bootstrap draws/results,
and an honest limitations report. Raw and participant-level files stay local;
desktop handoff contains aggregates and a verified acquisition procedure.
Original source attribution and data-license links remain attached. This plan
has a timestamped hash before fitting; it is not formal preregistration.
