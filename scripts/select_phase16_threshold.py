from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "features" / "phase11d_expanded_features.csv"
OUTPUT_DIR = ROOT / "outputs" / "experiments"
SCORES_PATH = OUTPUT_DIR / "phase16_validation_scores.csv"
THRESHOLD_PATH = OUTPUT_DIR / "phase16_threshold_results.csv"

QC_METADATA_COLUMNS = {
    "fps",
    "frames_processed",
    "rgb_detected_frames",
    "face_detection_rate",
    "rppg_samples",
    "rppg_segment_count",
    "shortest_rppg_segment",
    "longest_rppg_segment",
    "rppg_interpolated_samples",
    "rppg_interpolation_rate",
    "motion_samples",
    "motion_segment_count",
    "shortest_motion_segment",
    "longest_motion_segment",
    "motion_interpolated_samples",
    "motion_interpolation_rate",
    "synchronized_samples",
    "synchronized_frame_count",
    "synchronized_interpolated_frames",
    "synchronized_interpolation_rate",
}

def get_existing_feature_columns(df):
    graph_features = [
        col for col in df.columns
        if col.startswith("motion_gsp_")
        or col.startswith("physiology_gsp_")
    ]
    baseline_features = [
        col for col in df.columns
        if (
            col.startswith("motion_")
            or col.startswith("physiology_")
        )
        and col not in QC_METADATA_COLUMNS
        and col not in graph_features
    ]
    if len(baseline_features) != 60:
        raise ValueError(
            f"Expected 60 baseline features, got {len(baseline_features)}"
        )
    return baseline_features



def main():
    df = pd.read_csv(FEATURES_PATH)

    # Reuse the existing 60-feature baseline definition.
    feature_columns = get_existing_feature_columns(df)

    if len(feature_columns) != 60:
        raise ValueError(
            f"Expected 60 baseline features; found {len(feature_columns)}. "
            "Stop rather than silently changing the model."
        )

    # Select rows by split BEFORE accessing labels for evaluation.
    train_df = df.loc[df["split"].eq("train")].copy()
    val_df = df.loc[df["split"].eq("validation")].copy()

    if len(train_df) != 68 or len(val_df) != 16:
        raise ValueError(
            f"Unexpected split sizes: train={len(train_df)}, "
            f"validation={len(val_df)}"
        )

    if set(train_df["group_id"]) & set(val_df["group_id"]):
        raise ValueError("Group leakage detected between train and validation.")

    # Only genuine training videos fit the model.
    genuine_train = train_df.loc[train_df["label"].eq("real")]
    if len(genuine_train) != 34:
        raise ValueError(
            f"Expected 34 genuine training videos; found {len(genuine_train)}"
        )

    X_train = genuine_train[feature_columns].to_numpy(dtype=float)
    X_val = val_df[feature_columns].to_numpy(dtype=float)

    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Non-finite feature values detected.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.10)
    model.fit(X_train_scaled)

    # Higher score = more suspicious / more likely fake.
    scores = -model.decision_function(X_val_scaled).ravel()
    y_true = val_df["label"].eq("fake").astype(int).to_numpy()

    # Candidate thresholds are midpoints between distinct validation scores.
    unique_scores = np.unique(scores)
    candidates = np.concatenate((
        [np.nextafter(unique_scores[0], -np.inf)],
        (unique_scores[:-1] + unique_scores[1:]) / 2,
        [np.nextafter(unique_scores[-1], np.inf)],
    ))

    # Choose the threshold maximizing validation F1.
    # If tied, prefer the threshold with higher recall, then higher precision.
    rows = []
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, predictions)),
        })

    metrics = pd.DataFrame(rows)
    best = metrics.sort_values(
        ["f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
        kind="stable",
    ).iloc[0]
    threshold = float(best["threshold"])
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_output = val_df[
        ["video", "group_id", "label", "split"]
    ].copy()
    score_output["anomaly_score"] = scores
    score_output["predicted_label"] = np.where(predictions == 1, "fake", "real")
    score_output.to_csv(SCORES_PATH, index=False)

    result = pd.DataFrame([{
        "model": "baseline_60",
        "feature_count": len(feature_columns),
        "genuine_training_videos": len(genuine_train),
        "validation_videos": len(val_df),
        "threshold": threshold,
        "threshold_rule": "maximize validation F1; ties prefer recall, then precision",
        "validation_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "validation_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "validation_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "validation_accuracy": float(accuracy_score(y_true, predictions)),
        "true_real": int(tn),
        "false_fake": int(fp),
        "missed_fake": int(fn),
        "detected_fake": int(tp),
        "reserved_test_rows_used": 0,
    }])
    result.to_csv(THRESHOLD_PATH, index=False)

    print("PHASE 16 — VALIDATION THRESHOLD SELECTION")
    print(f"Model: baseline_60 ({len(feature_columns)} features)")
    print(f"Genuine training videos: {len(genuine_train)}")
    print(f"Validation videos: {len(val_df)} (8 real, 8 fake)")
    print(f"Selected threshold: {threshold:.8f}")
    print(f"Validation F1: {result.loc[0, 'validation_f1']:.4f}")
    print(f"Precision: {result.loc[0, 'validation_precision']:.4f}")
    print(f"Recall: {result.loc[0, 'validation_recall']:.4f}")
    print(f"Accuracy: {result.loc[0, 'validation_accuracy']:.4f}")
    print(f"Saved validation scores: {SCORES_PATH}")
    print(f"Saved threshold results: {THRESHOLD_PATH}")
    print("Reserved test rows used: 0")


if __name__ == "__main__":
    main()
