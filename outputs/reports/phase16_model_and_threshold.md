# Phase 16 — Model and Threshold Selection

## Objective
Select a provisional anomaly-score threshold using validation data and save a reusable baseline model bundle.

## Selected model
- Model: `baseline_60`
- Algorithm: RBF One-Class SVM
- Parameters: `kernel=rbf`, `gamma=scale`, `nu=0.10`
- Features: 60 existing baseline motion and physiology features
- Training data: 34 genuine videos from the training split
- Preprocessing: StandardScaler fitted on genuine training features only

## Validation-selected threshold
- Threshold: `-0.08214575`
- Scoring convention: higher anomaly scores indicate greater suspicion of a fake video
- Prediction rule: score >= threshold => fake; otherwise real
- Validation videos: 16 (8 real, 8 fake)
- F1: 0.8421
- Precision: 0.7273
- Recall: 1.0000
- Accuracy: 0.8125

## Interpretation and limitations
The selected threshold achieved 100% recall on this small validation set, but precision was 72.73%. Some genuine validation videos may therefore be incorrectly flagged as fake. The threshold is provisional and may be sensitive to this small validation sample.

These metrics describe validation performance, not independent final-test performance. No reserved Phase 12D test rows were used to select the threshold or train the saved bundle.

## Saved artifacts
- `scripts/select_phase16_threshold.py`
- `scripts/save_phase16_model_bundle.py`
- `outputs/experiments/phase16_validation_scores.csv`
- `outputs/experiments/phase16_threshold_results.csv`
- `models/phase16_baseline_60.joblib`

The model bundle contains the fitted scaler, fitted One-Class SVM, exact feature names, threshold, and training metadata.

## Next phase
Phase 17 will preserve and document the final evaluation protocol and reserved test set. Do not use the reserved test set to adjust the threshold or model.
