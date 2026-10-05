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
    / "phase11h_group_cv_results.csv"
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


def evaluate_one_feature_set(
    df,
    feature_columns,
    feature_name,
    training_groups,
    heldout_group,
):
    """Train on genuine videos from remaining groups and evaluate one held-out group."""

    train_mask = (
        (df["split"] == "train")
        & (df["group_id"].isin(training_groups))
    )

    heldout_mask = (
        (df["split"] == "train")
        & (df["group_id"] == heldout_group)
    )

    genuine_train_mask = (
        train_mask
        & (df["label"] == "real")
    )

    X_train = df.loc[
        genuine_train_mask,
        feature_columns,
    ].to_numpy(dtype=float)

    X_heldout = df.loc[
        heldout_mask,
        feature_columns,
    ].to_numpy(dtype=float)

    y_heldout = (
        df.loc[
            heldout_mask,
            "label",
        ]
        .eq("fake")
        .astype(int)
        .to_numpy()
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_heldout_scaled = scaler.transform(X_heldout)

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.10,
    )

    model.fit(X_train_scaled)

    anomaly_scores = -model.decision_function(
        X_heldout_scaled
    )

    roc_auc = roc_auc_score(
        y_heldout,
        anomaly_scores,
    )

    pr_auc = average_precision_score(
        y_heldout,
        anomaly_scores,
    )

    return {
        "feature_set": feature_name,
        "heldout_group": heldout_group,
        "training_groups": len(training_groups),
        "genuine_training_videos": int(
            genuine_train_mask.sum()
        ),
        "heldout_videos": int(
            heldout_mask.sum()
        ),
        "heldout_real": int(
            (
                heldout_mask
                & (df["label"] == "real")
            ).sum()
        ),
        "heldout_fake": int(
            (
                heldout_mask
                & (df["label"] == "fake")
            ).sum()
        ),
        "feature_count": len(feature_columns),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def main():
    print("=" * 70)
    print("PHASE 11H - EXPANDED TRAINING-GROUP CROSS-VALIDATION")
    print("=" * 70)

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_PATH}"
        )

    df = pd.read_csv(FEATURE_PATH)

    print("\nDataset:")
    print(f"Rows: {len(df)}")

    baseline_features, graph_features = get_feature_columns(df)

    combined_features = (
        baseline_features + graph_features
    )

    print(f"Baseline features: {len(baseline_features)}")
    print(f"Graph-profile features: {len(graph_features)}")
    print(f"Combined features: {len(combined_features)}")

    print("\nFrozen split:")
    print(df["split"].value_counts())

    # Only training groups are used for LOGO cross-validation.
    train_df = df[df["split"] == "train"].copy()

    training_groups = sorted(
        train_df["group_id"].unique()
    )

    print(
        f"\nTraining groups available for CV: "
        f"{len(training_groups)}"
    )

    print(
        "Validation and test groups are excluded."
    )

    all_results = []

    feature_sets = {
        "baseline_60": baseline_features,
        "graph_profile_45": graph_features,
        "combined_105": combined_features,
    }

    for fold_number, heldout_group in enumerate(
        training_groups,
        start=1,
    ):
        remaining_groups = [
            group
            for group in training_groups
            if group != heldout_group
        ]

        print(
            f"\nFold {fold_number}/{len(training_groups)}"
        )
        print(
            f"Held-out group: {heldout_group}"
        )

        for feature_name, feature_columns in feature_sets.items():

            result = evaluate_one_feature_set(
                df=df,
                feature_columns=feature_columns,
                feature_name=feature_name,
                training_groups=remaining_groups,
                heldout_group=heldout_group,
            )

            result["fold"] = fold_number

            all_results.append(result)

            print(
                f"  {feature_name}: "
                f"ROC-AUC={result['roc_auc']:.4f}, "
                f"PR-AUC={result['pr_auc']:.4f}"
            )

    results_df = pd.DataFrame(all_results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("GROUP-CV SUMMARY")
    print("=" * 70)

    for feature_name in feature_sets:

        subset = results_df[
            results_df["feature_set"] == feature_name
        ]

        print(f"\n{feature_name}")

        print(
            f"Mean ROC-AUC: "
            f"{subset['roc_auc'].mean():.4f}"
        )

        print(
            f"ROC-AUC SD: "
            f"{subset['roc_auc'].std(ddof=0):.4f}"
        )

        print(
            f"Mean PR-AUC: "
            f"{subset['pr_auc'].mean():.4f}"
        )

        print(
            f"PR-AUC SD: "
            f"{subset['pr_auc'].std(ddof=0):.4f}"
        )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print(
        "\nValidation and test sets were not evaluated."
    )

    print(
        "Phase 11H group cross-validation completed."
    )


if __name__ == "__main__":
    main()