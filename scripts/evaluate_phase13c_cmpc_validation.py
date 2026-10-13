
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "phase13c_cmpc_validation_results.csv"
)

BASELINE_PATH = DATA_DIR / "phase11d_expanded_features.csv"
CMPC_PATH = DATA_DIR / "phase13c_cmpc_features.csv"

QC_METADATA_COLUMNS = {
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
    "graph_feature_count",
}

SEGMENT_STD_FEATURES = [
    "cmpc_forehead_segment_std",
    "cmpc_left_cheek_segment_std",
    "cmpc_right_cheek_segment_std",
]


def get_existing_feature_columns(df):
    """Reuse the project's existing baseline and graph definitions."""

    graph_features = [
        col
        for col in df.columns
        if col.startswith("motion_gsp_")
        or col.startswith("physiology_gsp_")
    ]

    baseline_features = [
        col
        for col in df.columns
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

    if len(graph_features) != 45:
        raise ValueError(
            f"Expected 45 graph features, got {len(graph_features)}"
        )

    return baseline_features, graph_features


def evaluate_feature_set(train_df, validation_df, feature_columns, name):
    """Fit on genuine training videos; score validation videos only."""

    genuine_train = train_df.loc[train_df["label"].eq("real")]

    X_train = genuine_train[feature_columns].to_numpy(dtype=float)
    X_validation = validation_df[feature_columns].to_numpy(dtype=float)

    if not np.isfinite(X_train).all():
        raise ValueError(f"Non-finite training values in {name}")

    if not np.isfinite(X_validation).all():
        raise ValueError(f"Non-finite validation values in {name}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.10,
    )
    model.fit(X_train_scaled)

    # Larger scores indicate greater anomaly / fake likelihood.
    scores = -model.decision_function(X_validation_scaled)
    y_true = validation_df["label"].eq("fake").astype(int).to_numpy()

    return {
        "feature_set": name,
        "feature_count": len(feature_columns),
        "genuine_train_videos": len(genuine_train),
        "validation_videos": len(validation_df),
        "validation_real": int((y_true == 0).sum()),
        "validation_fake": int((y_true == 1).sum()),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "real_mean_anomaly_score": float(scores[y_true == 0].mean()),
        "fake_mean_anomaly_score": float(scores[y_true == 1].mean()),
    }


def main():
    print("=" * 70)
    print("PHASE 13C - CMPC VALIDATION COMPARISON")
    print("=" * 70)

    for path in [BASELINE_PATH, CMPC_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")

    baseline_df = pd.read_csv(BASELINE_PATH)
    cmpc_df = pd.read_csv(CMPC_PATH)

    metadata = ["video", "group_id", "label", "split", "source_path"]

    for column in metadata:
        if column not in baseline_df.columns or column not in cmpc_df.columns:
            raise ValueError(f"Missing alignment column: {column}")

    if baseline_df["video"].duplicated().any():
        raise ValueError("Duplicate video IDs in baseline dataset")

    if cmpc_df["video"].duplicated().any():
        raise ValueError("Duplicate video IDs in CMPC dataset")

    # Align CMPC rows to the baseline dataset by video identity.
    aligned = baseline_df[metadata].merge(
        cmpc_df[metadata + [
            c for c in cmpc_df.columns if c.startswith("cmpc_")
        ]],
        on="video",
        how="left",
        suffixes=("_baseline", "_cmpc"),
        validate="one_to_one",
        indicator=True,
    )

    if not aligned["_merge"].eq("both").all():
        raise ValueError("Baseline and CMPC video sets do not match")

    for column in ["group_id", "label", "split", "source_path"]:
        if not (
            aligned[f"{column}_baseline"].astype(str).to_numpy()
            == aligned[f"{column}_cmpc"].astype(str).to_numpy()
        ).all():
            raise ValueError(f"Baseline/CMPC mismatch in {column}")

    cmpc_features = [
        col for col in cmpc_df.columns if col.startswith("cmpc_")
    ]

    if len(cmpc_features) != 16:
        raise ValueError(f"Expected 16 CMPC features, got {len(cmpc_features)}")

    missing_reduced = set(SEGMENT_STD_FEATURES) - set(cmpc_features)
    if missing_reduced:
        raise ValueError(
            f"Cannot build CMPC-13; missing features: {sorted(missing_reduced)}"
        )

    baseline_features, graph_features = get_existing_feature_columns(
        baseline_df
    )

    # Merge feature columns using the stable video ID.
    feature_df = baseline_df[metadata + baseline_features + graph_features].merge(
        cmpc_df[["video"] + cmpc_features],
        on="video",
        how="left",
        validate="one_to_one",
    )

    train_df = feature_df.loc[feature_df["split"].eq("train")].copy()
    validation_df = feature_df.loc[
        feature_df["split"].eq("validation")
    ].copy()

    if len(train_df) != 68 or len(validation_df) != 16:
        raise ValueError(
            f"Unexpected split sizes: train={len(train_df)}, "
            f"validation={len(validation_df)}"
        )

    if set(train_df["group_id"]) & set(validation_df["group_id"]):
        raise ValueError("Group leakage detected between train and validation")

    cmpc_reduced = [
        col for col in cmpc_features if col not in SEGMENT_STD_FEATURES
    ]

    feature_sets = [
        ("baseline_60", baseline_features),
        ("graph_profile_45", graph_features),
        ("cmpc_16", cmpc_features),
        ("cmpc_13", cmpc_reduced),
        ("baseline_plus_cmpc_76", baseline_features + cmpc_features),
        ("baseline_plus_cmpc_73", baseline_features + cmpc_reduced),
    ]

    print(f"Training videos: {len(train_df)}")
    print(f"Genuine training videos: {(train_df['label'] == 'real').sum()}")
    print(f"Validation videos: {len(validation_df)}")
    print(f"Baseline features: {len(baseline_features)}")
    print(f"Graph-profile features: {len(graph_features)}")
    print(f"CMPC features: {len(cmpc_features)}")
    print("Reserved test rows used for evaluation: 0")
    print("\nRunning experiments...")

    results = []

    for name, columns in feature_sets:
        result = evaluate_feature_set(
            train_df,
            validation_df,
            columns,
            name,
        )
        results.append(result)

        print(
            f"{name:26s} | features={result['feature_count']:3d} "
            f"| ROC-AUC={result['roc_auc']:.4f} "
            f"| PR-AUC={result['pr_auc']:.4f}"
        )

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(
        results_df[
            ["feature_set", "feature_count", "roc_auc", "pr_auc",
             "real_mean_anomaly_score", "fake_mean_anomaly_score"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print(f"\nSaved results: {OUTPUT_PATH}")
    print("The reserved test split was not used for evaluation.")
    print("Reminder: validation contains only 8 real and 8 fake videos.")


if __name__ == "__main__":
    main()
