
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "phase23_motion_only_rf.joblib"
PREDICTIONS_PATH = (
    ROOT
    / "outputs"
    / "experiments"
    / "phase24_nested_group_threshold_predictions.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "experiments"
REPORT_DIR = ROOT / "outputs" / "reports"


def main():
    print("Phase 25: Explainability and error audit")
    print("-" * 55)

    # -----------------------------------------------------
    # 1. Load the existing model bundle without refitting.
    # -----------------------------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Phase 23 model bundle not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    model = bundle.get("model")
    feature_columns = bundle.get("feature_columns", [])

    if model is None:
        raise ValueError("No model was found in the saved bundle.")

    if not feature_columns:
        raise ValueError("The model bundle has no feature columns.")

    if len(feature_columns) != 30:
        raise ValueError(
            f"Expected 30 motion features; found {len(feature_columns)}."
        )

    # -----------------------------------------------------
    # 2. Extract built-in Random Forest feature importance.
    # -----------------------------------------------------
    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            "The saved model does not expose feature_importances_."
        )

    importances = np.asarray(model.feature_importances_, dtype=float)

    if len(importances) != len(feature_columns):
        raise ValueError(
            "Feature-importance count does not match feature count."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    ).sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(
        0,
        "rank",
        np.arange(1, len(importance_df) + 1),
    )

    # -----------------------------------------------------
    # 3. Load Phase 24 outer-fold predictions.
    #    These are predictions for held-out training groups.
    # -----------------------------------------------------
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Phase 24 predictions not found. "
            "Complete Phase 24 before running this audit."
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "video_id",
        "group_id",
        "true_label",
        "fake_score",
        "selected_threshold",
        "outer_fold",
        "predicted_label_tuned",
        "predicted_label_fixed_050",
    }

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"Phase 24 predictions are missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(predictions) != 68:
        raise ValueError(
            f"Expected 68 Phase 24 prediction rows; found "
            f"{len(predictions)}."
        )

    if predictions["group_id"].nunique() != 17:
        raise ValueError(
            "Expected predictions from 17 groups."
        )

    if predictions["video_id"].duplicated().any():
        raise ValueError(
            "Duplicate video identifiers found in predictions."
        )

    # Convert labels and predictions to integers.
    y_true = predictions["true_label"].astype(int).to_numpy()
    y_pred = (
        predictions["predicted_label_tuned"].astype(int).to_numpy()
    )

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Unexpected true-label values.")

    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError("Unexpected prediction values.")

    # -----------------------------------------------------
    # 4. Create an error table.
    #    Label mapping: real = 0, fake = 1.
    # -----------------------------------------------------
    error_df = predictions.copy()

    error_df["actual_class"] = error_df["true_label"].map(
        {0: "real", 1: "fake"}
    )
    error_df["predicted_class"] = error_df[
        "predicted_label_tuned"
    ].map(
        {0: "real", 1: "fake"}
    )

    error_df["correct"] = (
        error_df["true_label"]
        == error_df["predicted_label_tuned"]
    )

    error_df["error_type"] = np.select(
        [
            (error_df["true_label"] == 0)
            & (error_df["predicted_label_tuned"] == 1),
            (error_df["true_label"] == 1)
            & (error_df["predicted_label_tuned"] == 0),
        ],
        [
            "false_positive_real_flagged_fake",
            "false_negative_fake_missed",
        ],
        default="correct",
    )

    # Absolute distance from the threshold selected for that fold.
    error_df["distance_from_threshold"] = np.abs(
        error_df["fake_score"] - error_df["selected_threshold"]
    )

    error_df = error_df.sort_values(
        ["correct", "distance_from_threshold"],
        ascending=[True, True],
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # 5. Summarize confusion-matrix counts and error patterns.
    # -----------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    error_summary = pd.DataFrame(
        [
            {
                "total_videos": len(predictions),
                "correct_predictions": int((y_true == y_pred).sum()),
                "incorrect_predictions": int((y_true != y_pred).sum()),
                "true_negative_real_correct": int(tn),
                "false_positive_real_flagged_fake": int(fp),
                "false_negative_fake_missed": int(fn),
                "true_positive_fake_correct": int(tp),
                "groups": int(predictions["group_id"].nunique()),
                "outer_folds": int(predictions["outer_fold"].nunique()),
            }
        ]
    )

    error_types = (
        error_df.groupby(
            ["actual_class", "predicted_class", "error_type"],
            dropna=False,
        )
        .size()
        .reset_index(name="video_count")
    )

    fold_errors = (
        error_df.groupby("outer_fold")
        .agg(
            videos=("video_id", "count"),
            incorrect_predictions=("correct", lambda values: int((~values).sum())),
            mean_score=("fake_score", "mean"),
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # 6. Save the audit outputs.
    # -----------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    importance_path = (
        OUTPUT_DIR / "phase25_random_forest_feature_importance.csv"
    )
    errors_path = OUTPUT_DIR / "phase25_oof_error_analysis.csv"
    summary_path = OUTPUT_DIR / "phase25_error_summary.csv"
    types_path = OUTPUT_DIR / "phase25_error_types.csv"
    fold_path = OUTPUT_DIR / "phase25_fold_error_summary.csv"

    importance_df.to_csv(importance_path, index=False)
    error_df.to_csv(errors_path, index=False)
    error_summary.to_csv(summary_path, index=False)
    error_types.to_csv(types_path, index=False)
    fold_errors.to_csv(fold_path, index=False)

    # -----------------------------------------------------
    # 7. Print a concise audit summary.
    # -----------------------------------------------------
    print("Model:", bundle.get("model_name"))
    print("Model training rows:", bundle.get("training_rows"))
    print("Model training groups:", bundle.get("training_group_count"))
    print("Features audited:", len(feature_columns))

    print("\n=== TOP 10 BUILT-IN FEATURE IMPORTANCES ===")
    print(
        importance_df.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n=== OUT-OF-FOLD ERROR SUMMARY ===")
    print(error_summary.to_string(index=False))

    print("\n=== ERROR TYPES ===")
    print(error_types.to_string(index=False))

    print("\n=== ERRORS BY OUTER FOLD ===")
    print(fold_errors.to_string(index=False))

    print("\n=== MISCLASSIFIED VIDEOS ===")
    mistakes = error_df.loc[
        ~error_df["correct"],
        [
            "video_id",
            "group_id",
            "outer_fold",
            "actual_class",
            "predicted_class",
            "fake_score",
            "selected_threshold",
            "distance_from_threshold",
            "error_type",
        ],
    ]

    if mistakes.empty:
        print("No misclassified videos in the Phase 24 predictions.")
    else:
        print(mistakes.to_string(index=False))

    print("\nSaved audit files:")
    print(importance_path)
    print(errors_path)
    print(summary_path)
    print(types_path)
    print(fold_path)

    print(
        "\nInterpretation note: built-in Random Forest importance "
        "describes model reliance, not causation. Correlated features "
        "can share or distort importance."
    )
    print(
        "This audit uses saved training-only outer-fold predictions. "
        "It does not use the reserved final-test split."
    )


if __name__ == "__main__":
    main()
