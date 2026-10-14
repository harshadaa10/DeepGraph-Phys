# Phase 23 — Provisional Model Selection

## Selected candidate

- Model: Random Forest classifier
- Feature set: 30 motion features
- Estimators: 300 trees
- Minimum samples per leaf: 2
- Class weighting: balanced
- Random state: 42
- Model artifact: `models/phase23_motion_only_rf.joblib`

## Evidence supporting selection

The motion-only model was selected provisionally using the Phase 21–22 experiments.

| Evaluation | Model | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| Group-held-out training CV | Baseline, 60 features | 0.868 | 0.860 |
| Group-held-out training CV | Motion-only, 30 features | 0.898 | 0.900 |
| Group-held-out training CV | Top 5 motion features | 0.841 | 0.746 |
| Fixed validation, 16 videos | Baseline, 60 features | 0.891 | 0.913 |
| Fixed validation, 16 videos | Motion-only, 30 features | 0.906 | 0.931 |
| Fixed validation, 16 videos | Top 5 motion features | 0.859 | 0.914 |

The motion-only model performed best among these candidates in the reported comparisons. The small validation set and limited number of groups mean this is provisional evidence, not proof of generalization.

## Training and preprocessing

- Fitted on the 68 training videos across 17 groups.
- Real labels are encoded as 0 and fake labels as 1.
- Missing-value medians are calculated from training data only.
- Random Forest settings match the Phase 22 motion ablation experiment.
- The 16 validation rows are not used to fit the saved model.
- The reserved final-test split was not used for this model selection or training step.

## Threshold and limitations

The saved bundle deliberately has no decision threshold. Probability calibration and threshold selection remain pending in Phase 24.

The reported metrics are from the existing Phase 21–22 experiments; they are not fresh evaluations of the saved bundle. The final-test set remains reserved and must not be used for model selection or threshold tuning.

## Artifact verification

The saved bundle was reloaded successfully and verified to contain:
- 30 unique feature names
- A fitted `RandomForestClassifier`
- 300 estimators
- No decision threshold

## Status

Phase 23 model bundle created and verified. Confidence calibration, evidence-based explanations, application integration, and final reporting remain pending.
