# Phase 15 — Experimental Findings

## 1. Objective
Compare the existing baseline, graph-profile, cross-modal physiological-motion consistency (CMPC), and combined feature configurations using the completed validation, group cross-validation, and feature-ablation results.

## 2. Validation results

| Feature configuration | Features | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Baseline | 60 | 0.7734 | 0.6753 |
| Graph profile | 45 | 0.6250 | 0.6104 |
| CMPC | 16 | 0.4922 | 0.5168 |
| CMPC reduced | 13 | 0.4922 | 0.5168 |
| Baseline + CMPC | 76 | 0.7109 | 0.6473 |
| Baseline + CMPC reduced | 73 | 0.7109 | 0.6473 |

The validation set contains 16 videos: 8 real and 8 fake. The model was trained on 34 genuine training videos, according to the existing experiment output.

## 3. Group cross-validation results

| Feature configuration | Mean ROC-AUC | ROC-AUC SD | Mean PR-AUC |
|---|---:|---:|---:|
| Baseline | 0.5809 | 0.2874 | 0.7157 |
| Graph profile | 0.4412 | 0.3154 | 0.6275 |
| CMPC | 0.5515 | 0.2146 | 0.6863 |
| CMPC reduced | 0.5515 | 0.2146 | 0.6863 |
| Baseline + CMPC | 0.5956 | 0.2893 | 0.7304 |
| Baseline + CMPC reduced | 0.5956 | 0.2893 | 0.7304 |

The existing group-CV output contains 17 held-out groups, each with four videos (two real and two fake). Results vary substantially between groups, so the mean metrics should be interpreted cautiously.

## 4. Feature-ablation observations

The existing ablation output shows identical group-level ROC-AUC and PR-AUC values for the 16-feature and 13-feature CMPC configurations. The reduced 12-feature configuration differs on some groups. This suggests that removing three features did not change the recorded metrics in this experiment; it does not establish that those features are universally unnecessary.

## 5. Interpretation

- The 60-feature baseline has the strongest ROC-AUC on the existing validation set.
- Adding CMPC features does not improve validation ROC-AUC in this experiment.
- The combined feature configuration has a slightly higher mean group-CV ROC-AUC than the baseline, but this difference is small relative to the observed fold variability.
- CMPC alone performs close to chance on validation.
- These results do not establish a robust generalization improvement from CMPC.

## 6. Limitations

- The validation set is small.
- Each held-out CV group contains only four videos, making per-group metrics sensitive to individual predictions.
- Group-CV results show substantial variability.
- The existing metrics alone do not establish statistical significance or performance on unseen datasets.
- Results should not be generalized beyond the evaluated data without further independent testing.

## 7. Decision for the next phase

Carry the 60-feature baseline forward as the leading candidate for Phase 16 because it has the strongest existing validation ROC-AUC. This is a provisional model-selection decision, not a claim that it is universally optimal.

Select the decision threshold using validation data only. Keep the reserved Phase 12D final-test set and its existing result untouched; do not use them for threshold selection.

## 8. Reproducibility

This report summarizes existing CSV outputs and does not rerun experiments. Preserve the original experiment files alongside this report.
