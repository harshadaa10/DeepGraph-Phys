# Phase 9A — Controlled Feature-Set Comparison

## Research question

Do graph-derived features provide useful information alongside the existing facial physiology and motion features for deepfake video detection?

## Dataset and split

- Total videos: 40
- Training: 24 videos (12 real, 12 fake)
- Validation: 8 videos (4 real, 4 fake)
- Test: 8 videos (4 real, 4 fake)
- The reserved test split was not loaded for evaluation.

## Compared feature sets

| Feature set | Feature count | Description |
|---|---:|---|
| Baseline | 60 | Existing facial physiology and motion features |
| Graph profile | 45 | Graph signal processing features |
| Combined | 105 | Baseline and graph profile features concatenated |

The feature tables were aligned by video identity. The Phase 9 audit found no duplicate video names, matching identity/split metadata, and no overlapping feature names.

## Evaluation protocol

All three feature sets were evaluated using the same preliminary one-class SVM protocol:

- RBF kernel
- `gamma="scale"`
- `nu=0.10`
- StandardScaler fitted only on genuine training videos
- One-class SVM fitted only on genuine training videos
- Anomaly score: negative decision function; larger values indicate greater anomaly
- Validation ROC-AUC used for comparison

No classification threshold was selected using fake validation labels.

## Preliminary validation results

| Feature set | Validation ROC-AUC |
|---|---:|
| Baseline (60 features) | 0.6250 |
| Graph profile (45 features) | 0.6875 |
| Combined (105 features) | 0.7500 |

## Interpretation and limitations

On this validation split, the combined feature set achieved the highest ROC-AUC among the three evaluated feature sets. The graph-profile-only feature set also had a higher validation ROC-AUC than the baseline-only feature set.

These observations are preliminary. The validation set contains only four real and four fake videos, so the results are highly sensitive to individual examples and do not establish a reliable or statistically significant improvement.

ROC-AUC measures ranking performance; it is not classification accuracy. The comparison does not establish a final decision threshold or demonstrate generalization to unseen datasets.

The combined feature set has 105 dimensions but only 12 genuine training videos. This high-dimensional, small-sample setting is an additional limitation.

The reserved test split remains unevaluated. Further validation with a larger, appropriately grouped dataset is needed before making claims about generalizable performance.

## Current conclusion

The results provide an initial reason to investigate whether graph-derived features contain complementary information. They do not yet justify claiming that graph features improve deepfake detection reliably or selecting the combined feature set as the final model.



## Phase 9B — Training-only group-wise cross-validation

### Procedure

A Leave-One-Group-Out cross-validation experiment was run using only the 24 training videos across six groups.

For each fold:
- One complete group was held out (two real and two fake videos).
- The scaler was fitted only on genuine videos from the other training groups.
- The one-class SVM was fitted only on those scaled genuine training videos.
- The held-out group was scored using the negative decision function.
- The reserved validation and test splits were not used in this cross-validation experiment.

### Results

| Feature set | Mean fold ROC-AUC | Standard deviation across folds |
|---|---:|---:|
| Baseline (60 features) | 0.4792 | 0.1517 |
| Graph profile (45 features) | 0.5417 | 0.3033 |
| Combined (105 features) | 0.4792 | 0.2094 |

Individual fold AUCs varied substantially. For the graph-profile model, fold AUCs ranged from 0.0 to 1.0.

### Interpretation and limitations

The graph-profile model had a slightly higher mean fold ROC-AUC in this experiment, but its fold-to-fold variation was large. The combined model's mean fold ROC-AUC was the same as the baseline model's mean.

Each held-out fold contained only four videos, so fold AUCs are coarse and sensitive to individual examples. The results do not establish that graph features improve detection reliably.

The pooled out-of-fold ROC-AUC values are reported by the experiment script, but they combine scores from separately fitted models. Since score scales may differ between folds, pooled AUC should be interpreted cautiously. Mean and standard deviation of the per-fold AUCs are the primary descriptive summaries here.

The experiment used only six training groups and 12 genuine training videos in total. It is exploratory and does not establish generalization to other identities, datasets, or manipulation methods. The reserved test split remains unevaluated.

### Current conclusion

The group-wise experiment does not provide consistent evidence that graph-derived features improve performance over the baseline. The current dataset is too small to support a strong feature-set selection claim. A larger, appropriately grouped dataset and a pre-specified evaluation protocol are needed for more reliable conclusions.