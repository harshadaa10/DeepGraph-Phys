from pathlib import Path
import pandas as pd

EXPERIMENTS = Path("outputs/experiments")
REPORTS = Path("outputs/reports")
REPORTS.mkdir(parents=True, exist_ok=True)

files = {
    "validation": EXPERIMENTS / "phase21_validation_summary.csv",
    "motion_ablation": EXPERIMENTS / "phase22_motion_ablation_summary.csv",
    "group_cv": EXPERIMENTS / "phase22_motion_group_cv_summary.csv",
    "centroid_ablation": EXPERIMENTS / "phase22_spectral_centroid_ablation_summary.csv",
}

for name, path in files.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name} results file: {path}"
        )

data = {name: pd.read_csv(path) for name, path in files.items()}

def make_table(frame, columns):
    columns = [c for c in columns if c in frame.columns]
    if not columns:
        return "_No compatible columns found._"

    view = frame[columns].copy()

    def format_value(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float,)):
            return f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in view.iterrows():
        lines.append(
            "| " + " | ".join(
                format_value(row[column]) for column in columns
            ) + " |"
        )

    return "\n".join(lines)

validation = data["validation"]
motion_ablation = data["motion_ablation"]
group_cv = data["group_cv"]
centroid_ablation = data["centroid_ablation"]

# Build a single machine-readable comparison from the saved experiments.
records = []

def add_records(experiment_name, frame, metrics):
    for _, row in frame.iterrows():
        record = {
            "experiment": experiment_name,
            "model": row.get("model", "unknown"),
            "feature_count": row.get("feature_count", None),
        }
        for metric in metrics:
            if metric in frame.columns:
                record[metric] = row[metric]
        records.append(record)

add_records(
    "phase21_validation",
    validation,
    ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"],
)
add_records(
    "phase22_validation_ablation",
    motion_ablation,
    ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1",
     "real_correct", "real_flagged_fake", "fake_missed", "fake_detected"],
)
add_records(
    "phase22_group_cv",
    group_cv,
    ["pooled_oof_roc_auc", "pooled_oof_pr_auc",
     "pooled_oof_accuracy", "pooled_oof_precision",
     "pooled_oof_recall", "pooled_oof_f1",
     "mean_fold_roc_auc", "fold_roc_auc_sd", "mean_fold_pr_auc"],
)
add_records(
    "phase22_centroid_ablation",
    centroid_ablation,
    ["pooled_oof_roc_auc", "pooled_oof_pr_auc",
     "pooled_oof_accuracy", "pooled_oof_precision",
     "pooled_oof_recall", "pooled_oof_f1",
     "mean_fold_roc_auc", "fold_roc_auc_sd", "mean_fold_pr_auc"],
)

comparison = pd.DataFrame(records)
csv_path = EXPERIMENTS / "phase22_consolidated_comparison.csv"
comparison.to_csv(csv_path, index=False)

report = f"""# DeepGraph-Phys - Phase 22 Feature Ablation Report

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

{make_table(validation, ['model', 'roc_auc', 'pr_auc', 'accuracy',
                          'precision', 'recall', 'f1'])}

## Experiment 2 - Feature-set ablation on validation

{make_table(motion_ablation, ['model', 'feature_count', 'roc_auc', 'pr_auc',
                              'accuracy', 'precision', 'recall', 'f1',
                              'real_correct', 'real_flagged_fake',
                              'fake_missed', 'fake_detected'])}

## Experiment 3 - Group-held-out cross-validation

{make_table(group_cv, ['model', 'feature_count', 'pooled_oof_roc_auc',
                       'pooled_oof_pr_auc', 'pooled_oof_accuracy',
                       'pooled_oof_precision', 'pooled_oof_recall',
                       'pooled_oof_f1', 'mean_fold_roc_auc',
                       'fold_roc_auc_sd', 'mean_fold_pr_auc'])}

## Experiment 4 - Spectral-centroid ablation

{make_table(centroid_ablation, ['model', 'feature_count',
                                'pooled_oof_roc_auc',
                                'pooled_oof_pr_auc',
                                'pooled_oof_accuracy',
                                'pooled_oof_precision',
                                'pooled_oof_recall',
                                'pooled_oof_f1',
                                'mean_fold_roc_auc',
                                'fold_roc_auc_sd',
                                'mean_fold_pr_auc'])}

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
"""

report_path = REPORTS / "phase22_feature_ablation_report.md"
report_path.write_text(report, encoding="utf-8")

print("Phase 22 consolidation complete.")
print("Report:", report_path)
print("Comparison CSV:", csv_path)
print("\nConsolidated experiment rows:", len(comparison))
print("\nReport preview:")
print(report[:2500])
