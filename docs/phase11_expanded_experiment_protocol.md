# Phase 11 — Expanded Experiment Protocol

## 1. Purpose

This document freezes the experimental protocol for the expanded DeepGraph-Phys experiment.

The purpose is to evaluate whether the proposed graph-based physiological and facial-motion features provide useful information for video deepfake anomaly detection compared with the existing baseline feature set.

The expanded experiment is intended to provide stronger evidence than the initial 40-video pilot experiment.

---

## 2. Dataset

Dataset:

**FaceForensics++**

Current expanded subset:

* 100 videos total
* 50 real videos
* 50 Deepfakes videos
* Manipulation type: Deepfakes
* Compression: c23

The dataset is currently a single-manipulation, single-compression experimental subset.

Therefore, results from this experiment must not be described as proof of generalization to all deepfake manipulation methods or datasets.

---

## 3. Group Construction

The expanded dataset contains 25 identity groups.

Each group contains four videos:

1. Original video from identity A
2. Original video from identity B
3. Deepfake A → B
4. Deepfake B → A

Therefore:

* 25 groups
* 4 videos per group
* 100 total videos

The grouping structure is used to prevent related videos from appearing across different dataset splits.

---

## 4. Dataset Splits

The dataset is divided at the group level.

| Split      | Groups |  Videos |   Real |   Fake |
| ---------- | -----: | ------: | -----: | -----: |
| Train      |     17 |      68 |     34 |     34 |
| Validation |      4 |      16 |      8 |      8 |
| Test       |      4 |      16 |      8 |      8 |
| **Total**  | **25** | **100** | **50** | **50** |

No group is shared between train, validation, and test.

Identity overlap between splits is also prohibited.

---

## 5. Feature Sets

Three feature configurations will be compared.

### 5.1 Baseline Features

The baseline representation contains:

**60 features**

These are the existing physiological and facial-motion video-level features.

### 5.2 Graph Profile Features

The proposed graph representation contains:

**45 features**

Composition:

* 35 motion graph signal features
* 10 physiological graph signal features

The graph features summarize graph-frequency behavior across the facial-region graph.

### 5.3 Combined Features

The combined representation contains:

**105 features**

Calculated as:

60 baseline + 45 graph-profile = 105 features.

The same videos and same train/validation/test splits must be used for all three representations.

---

## 6. Feature Quality Requirements

Before model evaluation:

* All feature values must be numeric.
* No NaN values are permitted.
* No infinite values are permitted.
* Every feature row must correspond to exactly one manifest video.
* Video names must be unique.
* Group, split, and label metadata must match the dataset manifest.

Phase 11E confirmed that these requirements are satisfied for the current 100-video dataset.

---

## 7. Training Protocol

The experiment uses a genuine-only anomaly-detection setting.

For model fitting:

* Only genuine/real training videos are used.
* Fake training videos are not used for fitting the anomaly detector.
* Validation and test data remain separate from model fitting.

For the expanded dataset:

* Total training videos: 68
* Genuine training videos: 34
* Fake training videos: 34
* Model fitting uses only the 34 genuine training videos.

---

## 8. Feature Scaling

A `StandardScaler` is used.

The scaler must be fitted only on the genuine training videos.

The fitted scaler is then applied to:

* genuine training data
* validation data
* test data

Validation and test data must never be used when fitting the scaler.

---

## 9. Anomaly Detection Model

The primary anomaly detector is:

**One-Class Support Vector Machine (One-Class SVM)**

Configuration:

* Kernel: RBF
* `gamma = "scale"`
* `nu = 0.10`

The model is fitted only on the scaled genuine training data.

The anomaly score is defined as:

**anomaly score = -decision_function(X)**

Therefore:

* Higher score → more anomalous
* Lower score → more similar to the learned genuine distribution

---

## 10. Feature Comparisons

The following three configurations will be evaluated under the same model protocol:

1. `baseline_60`
2. `graph_profile_45`
3. `combined_105`

The purpose is to determine whether the graph representation provides additional discriminative information relative to the baseline.

The comparison must use the same:

* training videos
* validation videos
* test videos
* scaling procedure
* anomaly detector
* scoring method
* evaluation metrics

---

## 11. Validation Protocol

Validation data will be used for model comparison and analysis before the final test evaluation.

Primary validation metrics:

* ROC-AUC
* PR-AUC

Secondary metrics may include:

* F1-score
* Accuracy
* Precision
* Recall
* TPR/FPR at a selected threshold

Threshold-dependent metrics must clearly state how the threshold was selected.

The validation set may be used to select the preferred feature representation or model configuration.

---

## 12. Group-Based Cross-Validation

Where computationally practical, group-based cross-validation may be performed on the training portion.

Groups must remain intact during every fold.

No related identity group may be split between the training and validation portions of an individual fold.

The anomaly detector must be refitted independently within each fold using only genuine training samples from that fold.

Because each group contains only four videos, fold-level estimates may have high variance and should be reported with appropriate caution.

---

## 13. Test-Set Lock

The expanded test set is considered locked until the feature representation and model protocol have been finalized.

The test set must not be repeatedly used for:

* feature selection
* hyperparameter tuning
* threshold tuning
* model selection
* debugging model behavior

The final test evaluation should be performed only after the experimental protocol is frozen.

Once the final test evaluation is performed, its results should be treated as the final held-out measurement for this experiment.

---

## 14. Primary Research Questions

The expanded experiment will investigate:

### RQ1

Can genuine-only anomaly detection distinguish Deepfake videos from genuine videos using the existing baseline features?

### RQ2

Do graph-based physiological and facial-motion features provide useful information for Deepfake anomaly detection?

### RQ3

Does combining the baseline and graph representations improve performance compared with either representation alone?

### RQ4

How stable are the results under group-disjoint evaluation?

---

## 15. Important Interpretation Rule

The experiment is designed to compare feature representations under a controlled protocol.

A higher ROC-AUC on the expanded test set does not by itself prove that the method will generalize to:

* unseen manipulation methods
* unseen datasets
* different compression levels
* different recording conditions
* real-world social-media videos

Such claims require additional cross-dataset and cross-manipulation experiments.

---

## 16. Current Dataset Limitations

The current expanded experiment has several limitations:

1. Only one manipulation method is currently included: Deepfakes.
2. Only c23 compression is currently used.
3. The dataset contains 100 videos.
4. The number of identity groups is 25.
5. The experiment is still relatively small compared with the complete FaceForensics++ dataset.
6. External-dataset generalization has not yet been evaluated.

These limitations must be reported honestly in the research paper.

---

## 17. Reproducibility

The following artifacts define the expanded experiment:

### Dataset manifest

`data/features/phase11c_expanded_manifest.csv`

### Extracted feature dataset

`data/features/phase11d_expanded_features.csv`

### Dataset/feature audit

`scripts/audit_phase11e_features.py`

### Feature extraction script

`scripts/build_phase11d_expanded_features.py`

### Feature pipeline

`src/features/video_pipeline.py`

### Graph profile extraction

`src/features/graph_profile_features.py`

The experiment should be reproducible using the committed scripts and frozen protocol.

---

## 18. Test Evaluation Rule

The expanded test evaluation will be performed only after:

1. Feature extraction is frozen.
2. Feature-quality auditing is complete.
3. The experiment protocol is frozen.
4. Validation/model-selection decisions are complete.
5. No further test-driven tuning is planned.

The final test results will then be recorded separately from validation results.

---

## 19. Status

Phase 11A — Dataset expansion strategy: COMPLETE

Phase 11B — FaceForensics++ expanded download: COMPLETE

Phase 11C — Group/identity manifest: COMPLETE

Phase 11D — Expanded feature extraction: COMPLETE

Phase 11E — Feature quality and leakage audit: COMPLETE

Phase 11F — Expanded experiment protocol: FROZEN

Next stage:

**Expanded validation/model evaluation**
