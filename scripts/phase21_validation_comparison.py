
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "features" / "phase11d_expanded_features.csv"
OUTPUT_DIR = ROOT / "outputs" / "experiments"
PREDICTIONS_PATH = OUTPUT_DIR / "phase21_validation_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "phase21_validation_summary.csv"

QC_COLUMNS = {
    "fps", "frames_processed", "rgb_detected_frames",
    "face_detection_rate", "rppg_samples", "rppg_segment_count",
    "shortest_rppg_segment", "longest_rppg_segment",
    "interpolated_frames", "interpolation_rate", "motion_samples",
    "graph_samples", "consecutive_graph_transitions",
    "graph_gap_transitions", "largest_graph_frame_difference",
    "synchronized_interpolated_frames",
    "synchronized_interpolation_rate", "feature_count",
}

METADATA_COLUMNS = {
    "video", "label", "group_id", "split", "source_path",
    "identity_a", "identity_b",
}


def main():
    df = pd.read_csv(DATA_PATH)

    train = df[df["split"].eq("train")].copy()
    valid = df[df["split"].eq("validation")].copy()

    if len(train) != 68 or len(valid) != 16:
        raise ValueError(
            f"Expected 68 training and 16 validation rows; "
            f"got {len(train)} and {len(valid)}"
        )

    if set(train["group_id"]) & set(valid["group_id"]):
        raise ValueError("Training/validation group leakage detected")

    features = [
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

    if len(features) != 60:
        raise ValueError(f"Expected 60 features, found {len(features)}")

    X_train = train[features].to_numpy(dtype=float)
    X_valid = valid[features].to_numpy(dtype=float)
    y_train = train["label"].eq("fake").astype(int).to_numpy()
    y_valid = valid["label"].eq("fake").astype(int).to_numpy()

    if not np.isfinite(X_train).all() or not np.isfinite(X_valid).all():
        raise ValueError("Feature data contains NaN or infinity")

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=5000, random_state=42,
            ),
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    predictions = []
    summaries = []

    print("PHASE 21 — FIXED VALIDATION COMPARISON")
    print(f"Training rows: {len(train)}; validation rows: {len(valid)}")
    print(f"Validation real: {(y_valid == 0).sum()}")
    print(f"Validation fake: {(y_valid == 1).sum()}")
    print("Thresholds are model defaults; no validation tuning.")

    for name, model in models.items():
        model.fit(X_train, y_train)

        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_valid)
            predicted = (scores >= 0).astype(int)
        else:
            scores = model.predict_proba(X_valid)[:, 1]
            predicted = (scores >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_valid, predicted, labels=[0, 1]
        ).ravel()

        metrics = {
            "model": name,
            "train_videos": len(train),
            "validation_videos": len(valid),
            "roc_auc": roc_auc_score(y_valid, scores),
            "pr_auc": average_precision_score(y_valid, scores),
            "accuracy": accuracy_score(y_valid, predicted),
            "precision": precision_score(
                y_valid, predicted, zero_division=0
            ),
            "recall": recall_score(
                y_valid, predicted, zero_division=0
            ),
            "f1": f1_score(y_valid, predicted, zero_division=0),
            "true_real": int(tn),
            "false_fake": int(fp),
            "missed_fake": int(fn),
            "detected_fake": int(tp),
        }
        summaries.append(metrics)

        print(f"\n--- {name} ---")
        print(
            f"ROC-AUC={metrics['roc_auc']:.3f} | "
            f"PR-AUC={metrics['pr_auc']:.3f} | "
            f"F1={metrics['f1']:.3f}"
        )
        print(
            f"Real correctly identified: {tn}/8 | "
            f"Real incorrectly flagged fake: {fp}/8"
        )
        print(
            f"Fake missed: {fn}/8 | Fake detected: {tp}/8"
        )

        for i, (_, row) in enumerate(valid.iterrows()):
            predictions.append({
                "model": name,
                "video": row["video"],
                "group_id": row["group_id"],
                "actual_label": row["label"],
                "predicted_label": "fake" if predicted[i] else "real",
                "model_score": float(scores[i]),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(predictions).to_csv(PREDICTIONS_PATH, index=False)

    print("\n=== VALIDATION SUMMARY ===")
    print(
        pd.DataFrame(summaries)[
            ["model", "roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print(f"\nSummary saved: {SUMMARY_PATH}")
    print(f"Per-video predictions saved: {PREDICTIONS_PATH}")
    print("Reserved final-test data were not used.")


if __name__ == "__main__":
    main()
