# Frozen experiment dependency order

| Order | Task | Inputs beyond raw HomeACF | Scale / restart |
|---|---|---|---|
| 1 | cohort | None | roster, category and missingness audit |
| 2 | primary | two historical aggregate regression gates supplied | 20 seeds, 10 arms, 2,000 conditional resamples; restarts fitting |
| 3 | controls | freshly regenerated primary | 20 seeds; permutations, windows and pseudo-groups; restarts fitting |
| 4 | fullrefit | None | 300 raw-household prediction refits; restarts fitting |
| 5 | availability | None | 20 seeds, excludes index death; restarts fitting |
| 6 | missingness | None | raw observation model and 20 prediction seeds; restarts fitting |
| 7 | ipw-fullrefit | None | 300 observation + prediction refits; checkpoints are progress, not automatic resume |
| 8 | operating | None | 20-seed AUPRC/capture/NNS; restarts fitting |
| 9 | calibration | regenerated primary | 20 x 5 outer x 4 inner household folds; resumes matching local checkpoint |
| 10 | same-information | newly generated calibration NPZ | 20 seeds x 7 arms; resumes completed seed files |
| 11 | coding-sensitivity | None | 2 scenarios x 20 seeds; resumes completed seed/scenario files |
| 12 | budget | None | 20 RF/LR seeds, 2,000 conditional resamples; local generated cache permitted |
| 13 | simulation | generators, frozen protocol | 37 x 100 LR + 8 RF; resumes synthetic journal |
| 14 | simulation-report | completed synthetic journal/metadata | rebuild labels, counts, MC intervals; does not edit manuscripts |
| 15 | performance | primary, missingness, IPW fullrefit, operating, calibration | aggregate result assembly |
| 16 | dca | nested calibration summary | exploratory summary assembly |
| 17 | figures | regenerated six result families | seven PDF/PNG figures |

Use one consistent --run-dir for a complete sequence. Script-generated filenames may contain today's date; the wrapper adds the canonical historical-name alias required by older downstream scripts. This is a filename mapping, not a fabricated historical execution date. Original runtime/provenance metadata stays intact.

Long full runs can take many hours or days on a CPU. The exact full-pipeline total was not timed in the new delivery environment. Historic per-task elapsed values are recorded in evidence/results and the delivery validation report. Progress logs distinguish completed checks from complete scientific replications. Do not launch run-all merely to export accepted figures.
