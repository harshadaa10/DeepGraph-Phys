# Phase 17 — Final Evaluation Protocol

## 1. Purpose
Define a fixed, reproducible final evaluation protocol for DeepGraph-Phys without using the reserved test set for model selection or threshold tuning.

## 2. Frozen model
- Model artifact: `models/phase16_baseline_60.joblib`
- Model: baseline_60 One-Class SVM
- Kernel: RBF
- Gamma: scale
- Nu: 0.10
- Feature count: 60
- Training data: genuine videos from the training split only
- Preprocessing: the saved StandardScaler fitted on genuine training data

## 3. Frozen decision threshold
- Threshold: -0.08214575
- Anomaly score: negative One-Class SVM decision function
- Prediction rule: anomaly score >= threshold means fake; otherwise real
- Threshold source: validation data only
- The threshold must not be changed after observing final test results.

## 4. Reserved test set
The feature table contains 100 rows:
- Training: 68 videos (34 real, 34 fake)
- Validation: 16 videos (8 real, 8 fake)
- Test: 16 videos (8 real, 8 fake)

Only rows assigned to the test split may be used for the final evaluation. Test labels must not be used for training, feature selection, scaling, threshold selection, or other model tuning.

## 5. Evaluation procedure
When the authorized final evaluation is performed:
1. Load the frozen model bundle.
2. Select only the reserved test-split rows.
3. Use the exact 60 feature columns saved in the bundle, in the saved order.
4. Apply the saved scaler and model without fitting or updating either on test data.
5. Calculate anomaly scores and apply the frozen threshold.
6. Report ROC-AUC, PR-AUC, accuracy, precision, recall, F1, and the confusion matrix, where calculable.
7. Record the test sample count and class counts.
8. Preserve the resulting metrics and explain limitations honestly.

## 6. Leakage prevention
- Do not retrain on validation or test rows.
- Do not fit preprocessing on test rows.
- Do not change features or threshold based on test performance.
- Do not repeatedly evaluate the test set during development.
- Any model changes after final evaluation require a new, clearly documented evaluation protocol.

## 7. Current status
This document records the intended protocol; it does not execute the final evaluation. Phase 16 used 16 validation videos to select the threshold and did not use reserved test rows. The final test evaluation remains a separate, controlled step.

## 8. Known limitation
The validation set is small, so the selected threshold may not generalize. Final-test metrics must be reported as an independent evaluation and must not be used to retroactively claim that the threshold was selected without test influence unless the protocol was followed.

## 9. Frozen artifacts
- Model: `models/phase16_baseline_60.joblib`
- Threshold results: `outputs/experiments/phase16_threshold_results.csv`
- Validation scores: `outputs/experiments/phase16_validation_scores.csv`
- Model and threshold report: `outputs/reports/phase16_model_and_threshold.md`
