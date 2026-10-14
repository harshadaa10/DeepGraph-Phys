# DeepGraph-Phys - Phase 22 Feature Ablation Report

## Objective

Investigate whether motion features provide a useful signal for distinguishing
real and fake videos, using the existing training and validation splits.

## Data boundaries

- Training split: 68 videos across 17 groups.
- Fixed validation split: 16 videos.
- The reserved final-test split was not used in these experiments.
- Cross-validation used GroupKFold to hold out complete training groups.
- Missing-value replacements in cross-validation were learned from each
  fold's training portion only.

## Experiment 1 - Fixed validation comparison

The validation set contains only 16 videos, so its results are provisional.

| model | roc_auc | pr_auc | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.781 | 0.851 | 0.750 | 0.750 | 0.750 | 0.750 |
| rbf_svm | 0.820 | 0.876 | 0.562 | 0.545 | 0.750 | 0.632 |
| random_forest | 0.891 | 0.913 | 0.750 | 0.700 | 0.875 | 0.778 |

## Experiment 2 - Feature-set ablation on validation

| model | feature_count | roc_auc | pr_auc | accuracy | precision | recall | f1 | real_correct | real_flagged_fake | fake_missed | fake_detected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_60 | 60 | 0.891 | 0.913 | 0.750 | 0.700 | 0.875 | 0.778 | 5 | 3 | 1 | 7 |
| motion_only | 30 | 0.906 | 0.931 | 0.812 | 0.778 | 0.875 | 0.824 | 6 | 2 | 1 | 7 |
| top_5_motion | 5 | 0.859 | 0.914 | 0.750 | 0.700 | 0.875 | 0.778 | 5 | 3 | 1 | 7 |

## Experiment 3 - Group-held-out cross-validation

| model | feature_count | pooled_oof_roc_auc | pooled_oof_pr_auc | pooled_oof_accuracy | pooled_oof_precision | pooled_oof_recall | pooled_oof_f1 | mean_fold_roc_auc | fold_roc_auc_sd | mean_fold_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_60 | 60 | 0.868 | 0.860 | 0.794 | 0.794 | 0.794 | 0.794 | 0.912 | 0.119 | 0.941 |
| motion_only | 30 | 0.898 | 0.900 | 0.809 | 0.800 | 0.824 | 0.812 | 0.926 | 0.114 | 0.951 |
| top_5_motion | 5 | 0.841 | 0.746 | 0.794 | 0.763 | 0.853 | 0.806 | 0.838 | 0.226 | 0.877 |

## Experiment 4 - Spectral-centroid ablation

| model | feature_count | pooled_oof_roc_auc | pooled_oof_pr_auc | pooled_oof_accuracy | pooled_oof_precision | pooled_oof_recall | pooled_oof_f1 | mean_fold_roc_auc | fold_roc_auc_sd | mean_fold_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| motion_30 | 30 | 0.898 | 0.900 | 0.809 | 0.800 | 0.824 | 0.812 | 0.926 | 0.114 | 0.951 |
| motion_without_spectral_centroid | 25 | 0.897 | 0.904 | 0.794 | 0.778 | 0.824 | 0.800 | 0.897 | 0.150 | 0.936 |

## Current interpretation

1. The 30-feature motion-only model is the leading candidate in the experiments
   conducted so far.
2. On fixed validation, motion-only achieved ROC-AUC 0.906 and PR-AUC 0.931.
3. In group-held-out cross-validation, motion-only achieved pooled ROC-AUC
   0.898 and PR-AUC 0.900, compared with 0.868 and 0.860 for the 60-feature
   baseline.
4. The five-feature model did not consistently match the 30-feature model.
5. Removing spectral-centroid features produced mixed results. The evidence
   does not currently justify removing them.
6. These repeated experiments reuse the same dataset and are not independent
   confirmations. The small validation set and limited number of groups mean
   performance estimates remain uncertain.

## Decision

Retain all 30 motion features as the provisional Phase 23 candidate. Do not
describe this model as conclusively superior or as proven to generalize to
real-world deepfakes.

## Limitations and safeguards

- No threshold optimization was performed in this report.
- Cross-validation predictions are out-of-fold training predictions, not
  independent final-test results.
- The reserved final-test split must remain untouched during model selection.
- The current results do not establish causality or rule out all dataset biases.

## Reproducibility

This report was generated from the existing CSV experiment summaries. It does
not retrain models or alter the source feature dataset.
