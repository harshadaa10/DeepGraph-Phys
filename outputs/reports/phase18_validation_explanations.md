# Phase 18 — Validation Prediction Explanations

## Objective
Create transparent, descriptive explanations for baseline_60 predictions using validation data without accessing the reserved test set.

## Model and data
- Model bundle: `models/phase16_baseline_60.joblib`
- Model: RBF One-Class SVM
- Features: the exact 60 features saved in the bundle
- Validation videos: 16
- Threshold: -0.0821457543
- Prediction rule: anomaly score >= threshold means fake; otherwise real
- Scaling: the saved StandardScaler was reused without refitting.

## Outputs
- `scripts/explain_phase18_validation.py`
- `outputs/experiments/phase18_validation_predictions.csv`
- `outputs/experiments/phase18_validation_feature_deviations.csv`
- `outputs/figures/phase18_validation_feature_deviations.png`

## Explanation method
For each validation video, the script records its anomaly score, prediction, true label, and whether the prediction matches the label.

It also calculates the standardized feature values using the saved scaler and ranks features by their absolute standardized deviation. A visualization summarizes the mean absolute deviation of each feature across validation videos.

## Interpretation limits
These deviations describe how unusual a feature value is relative to the scaler's training reference. They are not direct One-Class SVM feature contributions, causal explanations, or proof that a particular feature drove a prediction. Correlated features may also make individual deviations difficult to interpret.

The aggregate visualization describes unusual feature values across the validation set; it is not a ranking of features by predictive importance.

## Data protection
Only rows assigned to the validation split were analyzed. The reserved test split was not evaluated, and the frozen model and threshold were not changed.

## Reproducibility
Run from the repository root with the project's virtual environment activated:

`python scripts/explain_phase18_validation.py`

## Status
Phase 18 validation explanation artifacts were generated successfully. Any conclusions about generalization must await the separately controlled final evaluation.
