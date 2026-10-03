# DeepGraph-Phys Experiment Protocol

## Phase 10B — Experiment Protocol Freeze

### 1. Dataset

Dataset:
FaceForensics++

Local experimental subset:
- 40 videos total
- 20 real videos
- 20 fake videos
- Manipulation: Deepfakes
- Compression: c23

The current dataset is a controlled pilot subset and does not represent all
FaceForensics++ manipulation methods or compression settings.

Source structure:

- Real:
  `data/datasets/FaceForensics++/original_sequences/youtube/c23/videos/`

- Fake:
  `data/datasets/FaceForensics++/manipulated_sequences/Deepfakes/c23/videos/`

The feature dataset is:

`data/features/ffpp_features.csv`

The graph feature dataset is:

`data/features/ffpp_graph_features.csv`

---

## 2. Group Construction

Each identity-pair group contains four videos:

1. Original video of identity A
2. Original video of identity B
3. Deepfake A <- B
4. Deepfake B <- A

Total:
- 10 groups
- 4 videos per group
- 40 videos

The two directional manipulations are kept within the same identity-pair
group to prevent related videos from being distributed across different
evaluation splits.

---

## 3. Dataset Split

Groups are split rather than individual videos.

### Training
- Groups: 1–6
- Videos: 24
- Real: 12
- Fake: 12

### Validation
- Groups: 7–8
- Videos: 8
- Real: 4
- Fake: 4

### Test
- Groups: 9–10
- Videos: 8
- Real: 4
- Fake: 4

No group is shared between training, validation, and test sets.

The test set remains locked until the experimental protocol and model
configuration are finalized.

---

## 4. Feature Sets

Three feature configurations are compared.

### Baseline feature set

60 features.

The baseline contains the existing motion and physiological
graph-signal-processing-derived feature representation.

### Graph profile

45 graph-specific features.

The graph profile contains:
- 35 motion features
- 10 physiological features

### Combined feature set

105 features:

60 baseline features + 45 graph-profile features.

All three feature configurations use the same video identities and
group-disjoint dataset splits.

---

## 5. Feature Quality Requirements

Before model training:

- Feature values must be finite.
- Duplicate video identities must not exist.
- Feature/video metadata must remain aligned.
- Group assignments must remain unchanged.
- Source videos must correspond to feature rows.

The Phase 10A source audit verified:

- 40 feature rows
- 40 unique video names
- 0 missing source files

---

## 6. Scaling

StandardScaler is fitted only using the genuine training videos.

The fitted scaler is then applied to validation and test data.

No validation or test information is used to fit the scaler.

---

## 7. Genuine-Only Anomaly Detection

The primary detection setting is genuine-only anomaly detection.

The model is trained using genuine training videos only.

Fake videos are not used as positive training examples.

The anomaly score is defined so that:

> higher score = more anomalous

This allows the system to detect deviations from the learned genuine
physiological/motion representation.

---

## 8. One-Class SVM

Primary model:

One-Class SVM with:

- Kernel: RBF
- gamma: `scale`
- nu: `0.10`

Training:
- genuine training videos only

Preprocessing:
- StandardScaler fitted on genuine training videos only

Scoring:
- anomaly score = negative One-Class SVM decision function

Higher anomaly scores indicate greater deviation from the genuine model.

---

## 9. Validation

Validation data is used for model/feature comparison before final test
evaluation.

Primary comparison metric:

- ROC-AUC

Validation results from Phase 9A are considered preliminary because the
validation set contains only 8 videos:

- 4 real
- 4 fake

Previously observed validation AUC:

| Feature set | Validation ROC-AUC |
|---|---:|
| Baseline 60 | 0.625 |
| Graph profile 45 | 0.6875 |
| Combined 105 | 0.750 |

These values are preliminary and must not be interpreted as evidence of
generalization performance.

---

## 10. Group-Aware Cross-Validation

Phase 9B used Leave-One-Group-Out cross-validation on the 24 training
videos.

Training groups:
- 6 groups

Each fold holds out one complete identity-pair group.

This prevents videos belonging to the same identity pair from appearing
on both sides of a fold.

Because each group contains only:
- 2 real videos
- 2 fake videos

individual fold AUC values are coarse and unstable.

Therefore, group-CV results are treated as diagnostic evidence rather than
as definitive performance estimates.

---

## 11. Test-Set Lock

The test set consists of groups 9–10.

The test set must not be used for:

- feature selection
- model selection
- hyperparameter tuning
- threshold selection
- protocol modification
- repeated experimentation

The test set is evaluated only after the experimental protocol and
model configuration have been frozen.

---

## 12. Research Limitations

The current experiment has important limitations:

1. Only 40 videos are available in the local experimental subset.
2. Only the Deepfakes manipulation method is available.
3. Only c23 compression is represented.
4. Only 10 identity-pair groups are available.
5. The validation set contains only 8 videos.
6. Group-CV folds contain only 4 videos each.
7. The current experiment therefore provides controlled pilot evidence
   rather than broad evidence of real-world deepfake generalization.

Claims about generalization must remain consistent with these limitations.

---

## 13. Reproducibility Rule

After this protocol is frozen:

- dataset splits must not be changed based on test results
- feature definitions must not be changed based on test results
- model configuration must not be changed based on test results
- all final test results must be reported exactly as obtained

Any later methodological change must be treated as a new experiment.

---

## 14. Phase 10A Verification

Dataset/source verification completed before protocol freeze.

Verified:

- 20 real source videos
- 20 fake source videos
- 40 feature rows
- 40 unique video names
- 0 missing source files
- 10 identity-pair groups
- group-disjoint train/validation/test split