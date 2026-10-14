
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "features" / "phase11d_expanded_features.csv"
OUTPUT_DIR = ROOT / "outputs" / "experiments"

RESULTS_PATH = OUTPUT_DIR / "phase21_supervised_group_cv_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "phase21_supervised_group_cv_summary.csv"

QC_COLUMNS = {
    "fps",
    "frames_processed",
    "rgb_detected_frames",
    "face_detection_rate",
    "rppg_samples",
    "rppg_segment_count",
    "shortest_rppg_segment",
    "longest_rppg_segment",
    "interpolated_frames",
    "interpolation_rate",
    "motion_samples",
    "graph_samples",
    "consecutive_graph_transitions",
    "graph_gap_transitions",
    "largest_graph_frame_difference",
    "synchronized_interpolated_frames",
    "synchronized_interpolation_rate",
    "feature_count",
}

METADATA_COLUMNS = {
    "video",
    "label",
    "group_id",
    "split",
    "source_path",
    "identity_a",
    "identity_b",
}


def main():
    print("=" * 70)
    print("PHASE 21 - SUPERVISED GROUP CROSS-VALIDATION")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    # Restrict all modeling to the original training split.
    train_df = df.loc[df["split"].eq("train")].copy()

    if len(train_df) != 68:
        raise ValueError(
            f"Expected 68 training videos, found {len(train_df)}"
        )

    if train_df["video"].duplicated().any():
        raise ValueError("Duplicate video IDs in training data")

    if not set(train_df["label"].unique()).issubset({"real", "fake"}):
        raise ValueError("Unexpected class label")

    feature_columns = [
        col for col in df.columns
        if (
            col.startswith("motion_")
            or col.startswith("physiology_")
        )
        and col not in QC_COLUMNS
        and col not in METADATA_COLUMNS
        and not col.startswith("motion_gsp_")
        and not col.startswith("physiology_gsp_")
    ]

    if len(feature_columns) != 60:
        raise ValueError(
            f"Expected 60 baseline features, found {len(feature_columns)}"
        )

    X = train_df[feature_columns].to_numpy(dtype=float)
    y = train_df["label"].eq("fake").astype(int).to_numpy()
    groups = train_df["group_id"].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinity")

    unique_groups = np.unique(groups)

    if len(unique_groups) != 17:
        raise ValueError(
            f"Expected 17 training groups, found {len(unique_groups)}"
        )

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            ),
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced",
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    splitter = GroupKFold(n_splits=len(unique_groups))
    all_results = []
    summaries = []

    print(f"Training videos: {len(train_df)}")
    print(f"Real videos: {(y == 0).sum()}")
    print(f"Fake videos: {(y == 1).sum()}")
    print(f"Groups/folds: {len(unique_groups)}")
    print(f"Features: {len(feature_columns)}")
    print("Validation rows used: 0")
    print("Reserved test rows used: 0")

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")

        oof_scores = np.full(len(train_df), np.nan)
        oof_predictions = np.full(len(train_df), -1, dtype=int)
        fold_aucs = []

        for fold, (train_idx, heldout_idx) in enumerate(
            splitter.split(X, y, groups),
            start=1,
        ):
            train_groups = set(groups[train_idx])
            heldout_groups = set(groups[heldout_idx])

            if train_groups & heldout_groups:
                raise ValueError("Group leakage detected")

            # Every training group is held out exactly once.
            X_fold_train = X[train_idx]
            y_fold_train = y[train_idx]
            X_fold_test = X[heldout_idx]
            y_fold_test = y[heldout_idx]

            if len(np.unique(y_fold_train)) != 2:
                raise ValueError("Training fold is missing a class")

            if len(np.unique(y_fold_test)) != 2:
                raise ValueError("Held-out fold is missing a class")

            model.fit(X_fold_train, y_fold_train)

            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_fold_test)
                predictions = (scores >= 0).astype(int)
            else:
                scores = model.predict_proba(X_fold_test)[:, 1]
                predictions = (scores >= 0.5).astype(int)

            oof_scores[heldout_idx] = scores
            oof_predictions[heldout_idx] = predictions

            fold_auc = roc_auc_score(y_fold_test, scores)
            fold_aucs.append(fold_auc)

            all_results.append({
                "model": model_name,
                "fold": fold,
                "heldout_group": next(iter(heldout_groups)),
                "heldout_videos": len(heldout_idx),
                "roc_auc": fold_auc,
                "pr_auc": average_precision_score(y_fold_test, scores),
            })

            print(
                f"Fold {fold:02d}/17 | "
                f"Group {next(iter(heldout_groups))} | "
                f"ROC-AUC={fold_auc:.3f}"
            )

        if np.isnan(oof_scores).any() or (oof_predictions < 0).any():
            raise ValueError("Some training videos lack out-of-fold predictions")

        # Pooled metrics use one prediction per training video.
        # Thresholds are model defaults, not validation-tuned thresholds.
        summary = {
            "model": model_name,
            "videos": len(train_df),
            "features": len(feature_columns),
            "folds": len(unique_groups),
            "pooled_oof_roc_auc": roc_auc_score(y, oof_scores),
            "pooled_oof_pr_auc": average_precision_score(y, oof_scores),
            "pooled_oof_accuracy": accuracy_score(y, oof_predictions),
            "pooled_oof_precision": precision_score(
                y, oof_predictions, zero_division=0
            ),
            "pooled_oof_recall": recall_score(
                y, oof_predictions, zero_division=0
            ),
            "pooled_oof_f1": f1_score(
                y, oof_predictions, zero_division=0
            ),
            "mean_fold_roc_auc": float(np.mean(fold_aucs)),
            "fold_roc_auc_sd": float(np.std(fold_aucs, ddof=0)),
        }
        summaries.append(summary)

        print(
            f"Pooled OOF ROC-AUC: {summary['pooled_oof_roc_auc']:.4f}"
        )
        print(
            f"Pooled OOF PR-AUC:  {summary['pooled_oof_pr_auc']:.4f}"
        )
        print(
            f"Pooled OOF F1:      {summary['pooled_oof_f1']:.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)
    pd.DataFrame(summaries).to_csv(SUMMARY_PATH, index=False)

    print("\n=== SUPERVISED MODEL COMPARISON ===")
    print(
        pd.DataFrame(summaries)[
            [
                "model",
                "pooled_oof_roc_auc",
                "pooled_oof_pr_auc",
                "pooled_oof_f1",
                "mean_fold_roc_auc",
                "fold_roc_auc_sd",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    print(f"\nDetailed results: {RESULTS_PATH}")
    print(f"Summary results:  {SUMMARY_PATH}")
    print("Only the original 68 training videos were used.")


if __name__ == "__main__":
    main()
