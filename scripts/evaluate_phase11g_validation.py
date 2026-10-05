from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "phase11d_expanded_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "phase11g_validation_results.csv"
)


QC_METADATA_COLUMNS = {
    "video",
    "label",
    "group_id",
    "split",
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


EXPECTED_BASELINE_FEATURES = 60
EXPECTED_GRAPH_FEATURES = 45


def get_feature_columns(df):
    """Identify the frozen baseline and graph feature sets."""

    graph_features = [
        column
        for column in df.columns
        if column.startswith("motion_gsp_")
        or column.startswith("physiology_gsp_")
    ]

    baseline_features = [
        column
        for column in df.columns
        if (
            column.startswith("motion_")
            or column.startswith("physiology_")
        )
        and column not in QC_METADATA_COLUMNS
        and column not in graph_features
    ]

    if len(baseline_features) != EXPECTED_BASELINE_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_BASELINE_FEATURES} baseline features, "
            f"found {len(baseline_features)}"
        )

    if len(graph_features) != EXPECTED_GRAPH_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_FEATURES} graph features, "
            f"found {len(graph_features)}"
        )

    return baseline_features, graph_features


def evaluate_feature_set(
    df,
    feature_columns,
    feature_name,
):
    """Fit on genuine training videos and evaluate on validation."""

    train_mask = df["split"] == "train"
    validation_mask = df["split"] == "validation"

    genuine_train_mask = train_mask & (df["label"] == "real")

    X_train = df.loc[genuine_train_mask, feature_columns].to_numpy(
        dtype=float
    )

    X_validation = df.loc[
        validation_mask, feature_columns
    ].to_numpy(dtype=float)

    y_validation = (
        df.loc[validation_mask, "label"]
        .eq("fake")
        .astype(int)
        .to_numpy()
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.10,
    )

    model.fit(X_train_scaled)

    anomaly_scores = -model.decision_function(
        X_validation_scaled
    )

    roc_auc = roc_auc_score(
        y_validation,
        anomaly_scores,
    )

    pr_auc = average_precision_score(
        y_validation,
        anomaly_scores,
    )

    return {
        "feature_set": feature_name,
        "train_videos": int(train_mask.sum()),
        "genuine_train_videos": int(genuine_train_mask.sum()),
        "validation_videos": int(validation_mask.sum()),
        "validation_real": int(
            (validation_mask & (df["label"] == "real")).sum()
        ),
        "validation_fake": int(
            (validation_mask & (df["label"] == "fake")).sum()
        ),
        "feature_count": len(feature_columns),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def main():
    print("=" * 70)
    print("PHASE 11G - EXPANDED VALIDATION EVALUATION")
    print("=" * 70)

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_PATH}"
        )

    df = pd.read_csv(FEATURE_PATH)

    print("\nDataset:")
    print(f"Rows: {len(df)}")

    baseline_features, graph_features = get_feature_columns(df)

    combined_features = baseline_features + graph_features

    print(f"Baseline features: {len(baseline_features)}")
    print(f"Graph-profile features: {len(graph_features)}")
    print(f"Combined features: {len(combined_features)}")

    print("\nFrozen split:")
    print(df["split"].value_counts())

    if set(df["split"]) != {"train", "validation", "test"}:
        raise ValueError("Unexpected split values.")

    # Important:
    # The test rows are intentionally never passed to the evaluator.
    validation_results = []

    validation_results.append(
        evaluate_feature_set(
            df,
            baseline_features,
            "baseline_60",
        )
    )

    validation_results.append(
        evaluate_feature_set(
            df,
            graph_features,
            "graph_profile_45",
        )
    )

    validation_results.append(
        evaluate_feature_set(
            df,
            combined_features,
            "combined_105",
        )
    )

    results_df = pd.DataFrame(validation_results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    for _, row in results_df.iterrows():
        print(
            f"{row['feature_set']}: "
            f"ROC-AUC = {row['roc_auc']:.4f}, "
            f"PR-AUC = {row['pr_auc']:.4f}"
        )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print("\nTest set was not evaluated.")
    print("Phase 11G validation evaluation completed.")


if __name__ == "__main__":
    main()