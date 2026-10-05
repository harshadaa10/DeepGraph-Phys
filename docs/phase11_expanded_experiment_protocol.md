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
---

# Phase 11I Exploratory Model-Selection Amendment

## Purpose

Phase 11I evaluated whether the graph-profile features provide useful and complementary information beyond the 60-dimensional baseline feature representation.

This analysis was performed using the expanded FaceForensics++ training partition only.

The validation and reserved test partitions were not used.

## Exploratory Comparisons

Four feature configurations were compared using training-group Leave-One-Group-Out cross-validation:

1. `baseline_60`
2. `graph_profile_45`
3. `reduced_graph`
4. `baseline_60 + reduced_graph`

The reduced graph representation was selected independently inside each CV fold using a deterministic correlation filter with an absolute correlation threshold of 0.90.

This ensured that graph-feature selection did not use the held-out group.

All configurations used the same genuine-only One-Class SVM protocol:

- StandardScaler fitted only on genuine training videos
- One-Class SVM with RBF kernel
- `gamma="scale"`
- `nu=0.10`
- anomaly score = negative decision function
- held-out group evaluation using ROC-AUC and PR-AUC

## Phase 11I Results

| Feature configuration | Mean ROC-AUC | ROC-AUC SD | Mean PR-AUC | PR-AUC SD |
|---|---:|---:|---:|---:|
| `baseline_60` | 0.5809 | 0.2874 | 0.7157 | 0.2002 |
| `graph_profile_45` | 0.4412 | 0.3154 | 0.6275 | 0.1855 |
| `reduced_graph` | 0.5147 | 0.2775 | 0.6569 | 0.1828 |
| `baseline_60 + reduced_graph` | 0.5662 | 0.2848 | 0.6961 | 0.1937 |

## Interpretation

The original 45-dimensional graph profile showed substantial redundancy.

Correlation-based reduction decreased the graph representation to approximately 13�17 features per fold and improved graph-only performance relative to the unreduced graph profile.

However, the reduced graph representation remained below the baseline model.

Adding the reduced graph features to the baseline also did not improve performance. The combined reduced representation achieved a mean ROC-AUC of 0.5662 compared with 0.5809 for the baseline.

Therefore, the Phase 11I training-group analysis provides no evidence that the current graph-profile representation adds complementary predictive value to the baseline feature set under the frozen genuine-only anomaly-detection protocol.

## Model Selection Decision

For the next evaluation stage, `baseline_60` is selected as the primary candidate.

The graph-profile experiments are retained as an exploratory research finding rather than being discarded.

This decision is based on training-group cross-validation only.

The validation partition remains reserved for the next model-selection checkpoint.

The test partition remains completely locked and has not been used during Phase 11I.

## Research Interpretation

The graph-based representation should not be presented as empirically superior in the current 100-video pilot experiment.

Instead, the results demonstrate that:

- graph features can contain class-direction differences;
- the original graph profile contains substantial redundancy;
- redundancy reduction improves the graph-only representation somewhat;
- the current graph representation does not outperform the baseline;
- adding the current graph representation does not improve the baseline under the present experimental conditions.

This is treated as a negative/diagnostic finding and is retained for transparent reporting.

## Scope Limitation

These conclusions are specific to the current pilot experiment:

- FaceForensics++ dataset
- 100 videos
- 50 real and 50 fake
- Deepfakes manipulation
- c23 compression
- 25 identity groups
- 17 training groups used for exploratory CV

They do not establish that graph-based physiological or motion features are universally ineffective for deepfake detection.

Future work may investigate larger datasets, additional manipulation types, alternative graph constructions, or stronger graph representations.

## Reproducibility Artifacts

Phase 11I analysis scripts:

- `scripts/diagnose_phase11i_graph_features.py`
- `scripts/analyze_phase11i_group_stability.py`
- `scripts/analyze_phase11i_redundancy.py`
- `scripts/evaluate_phase11i_reduced_graph_cv.py`
- `scripts/evaluate_phase11i_combined_reduced_cv.py`

Earlier expanded validation and group-CV scripts:

- `scripts/evaluate_phase11g_validation.py`
- `scripts/evaluate_phase11h_group_cv.py`

Phase 11I result artifacts are stored under:

`outputs/experiments/`

The next step is validation evaluation of the selected `baseline_60` candidate. The reserved test set remains untouched until the validation/model-selection decision is finalized.
