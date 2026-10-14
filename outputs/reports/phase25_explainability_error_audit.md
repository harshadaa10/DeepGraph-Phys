# Phase 25: Explainability and Error Audit

## Objective
Inspect the saved Phase 23 Random Forest feature importances and analyze
Phase 24 grouped outer-fold predictions. No model refitting was performed
by the audit script.

## Model and data scope
- Model: phase23_motion_only_random_forest
- Model training rows: 68
- Training groups: 17
- Features: 30 motion features
- Error analysis: 68 training-set out-of-fold predictions
- Outer folds: 4
- Reserved final-test split: not used

## Top five built-in feature importances

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | motion_high_ratio_std | 0.1323 |
| 2 | motion_low_ratio_std | 0.1228 |
| 3 | motion_spectral_centroid_std | 0.0822 |
| 4 | motion_smoothness_std | 0.0510 |
| 5 | motion_smoothness_max | 0.0497 |

These importances describe how the saved Random Forest uses its features.
They do not establish causation. Correlated features can affect importance
rankings, and importance alone does not explain an individual prediction.

## Out-of-fold error summary

| Metric | Count |
|---|---:|
| Total videos | 68 |
| Correct predictions | 55 |
| Incorrect predictions | 13 |
| True negatives | 27 |
| False positives | 7 |
| False negatives | 6 |
| True positives | 28 |

Out-of-fold accuracy was 80.9%. This is a small grouped evaluation and
should not be interpreted as a guarantee of performance on external data.

## Error observations
- Seven real videos were incorrectly classified as fake.
- Six fake videos were incorrectly classified as real.
- Several errors were close to their selected fold thresholds.
- Other errors had scores well away from their thresholds, so threshold
  changes alone are unlikely to fix every mistake.
- Multiple errors occurred in group_01 and group_05. This is an observation
  for future investigation, not proof of a specific cause.

## Limitations and safeguards
- Built-in feature importance is not causal explanation.
- The dataset contains only 68 training videos across 17 groups.
- Out-of-fold predictions are useful for analysis but have uncertainty.
- Do not tune the model or threshold to individual errors from this report.
- The reserved final-test split remains untouched.
- The protected final-test evaluation script remains untouched.
- SHAP explanations were not generated.

## Generated audit artifacts
- outputs/experiments/phase25_random_forest_feature_importance.csv
- outputs/experiments/phase25_oof_error_analysis.csv
- outputs/experiments/phase25_error_summary.csv
- outputs/experiments/phase25_error_types.csv
- outputs/experiments/phase25_fold_error_summary.csv

## Phase 25 status
The initial feature-importance and error audit completed successfully.
The findings are descriptive and do not justify further model tuning by
themselves.
