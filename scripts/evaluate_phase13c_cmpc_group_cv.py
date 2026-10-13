
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments"

BASELINE_PATH = DATA_DIR / "phase11d_expanded_features.csv"
CMPC_PATH = DATA_DIR / "phase13c_cmpc_features.csv"

RESULTS_PATH = (
    OUTPUT_DIR / "phase13c_cmpc_group_cv_results.csv"
)
SUMMARY_PATH = (
    OUTPUT_DIR / "phase13c_cmpc_group_cv_summary.csv"
)

SEGMENT_STD_FEATURES = [
    "cmpc_forehead_segment_std",
    "cmpc_left_cheek_segment_std",
    "cmpc_right_cheek_segment_std",
]

METADATA_COLUMNS = {
    "video",
    "group_id",
    "identity_a",
    "identity_b",
    "label",
    "source_path",
    "split",
}

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


def get_baseline_and_graph_features(df):
    graph_features = [
        col
        for col in df.columns
        if col.startswith("motion_gsp_")
        or col.startswith("physiology_gsp_")
    ]

    baseline_features = [
        col
        for col in df.columns
        if col.startswith("motion_")
        or col.startswith("physiology_")
    ]

    baseline_features = [
        col
        for col in baseline_features
        if col not in QC_METADATA_COLUMNS
        and col not in graph_features
    ]

    if len(baseline_features) != 60:
        raise ValueError(
            f"Expected 60 baseline features, "
            f"found {len(baseline_features)}"
        )

    if len(graph_features) != 45:
        raise ValueError(
            f"Expected 45 graph features, "
            f"found {len(graph_features)}"
        )

    return baseline_features, graph_features


def validate_and_merge_datasets(baseline_df, cmpc_df):
    metadata = [
        "video",
        "group_id",
        "label",
        "split",
        "source_path",
    ]

    for column in metadata:
        if column not in baseline_df.columns:
            raise ValueError(
                f"Baseline dataset is missing {column}"
            )

        if column not in cmpc_df.columns:
            raise ValueError(
                f"CMPC dataset is missing {column}"
            )

    if baseline_df["video"].duplicated().any():
        raise ValueError("Duplicate video IDs in baseline data")

    if cmpc_df["video"].duplicated().any():
        raise ValueError("Duplicate video IDs in CMPC data")

    cmpc_features = [
        col for col in cmpc_df.columns
        if col.startswith("cmpc_")
    ]

    if len(cmpc_features) != 16:
        raise ValueError(
            f"Expected 16 CMPC features, "
            f"found {len(cmpc_features)}"
        )

    aligned = baseline_df[metadata].merge(
        cmpc_df[metadata + cmpc_features],
        on="video",
        how="outer",
        suffixes=("_baseline", "_cmpc"),
        validate="one_to_one",
        indicator=True,
    )

    if not aligned["_merge"].eq("both").all():
        raise ValueError(
            "Baseline and CMPC video IDs do not match"
        )

    for column in metadata:
        if column == "video":
            continue

        left = aligned[f"{column}_baseline"].astype(str)
        right = aligned[f"{column}_cmpc"].astype(str)

        if not left.equals(right):
            raise ValueError(
                f"Baseline/CMPC metadata mismatch: {column}"
            )

    # Use the original baseline split and metadata.
    merged = baseline_df.merge(
        cmpc_df[["video"] + cmpc_features],
        on="video",
        how="left",
        validate="one_to_one",
    )

    if merged[cmpc_features].isna().any().any():
        raise ValueError("Missing CMPC feature values after merge")

    return merged, cmpc_features


def evaluate_fold(fold_train, fold_test, feature_columns):
    # The model learns only from genuine videos in this fold.
    genuine_train = fold_train[
        fold_train["label"].eq("real")
    ]

    X_train = genuine_train[feature_columns].to_numpy(
        dtype=float
    )
    X_test = fold_test[feature_columns].to_numpy(
        dtype=float
    )

    if len(genuine_train) == 0:
        raise ValueError("No genuine training videos in fold")

    if not np.isfinite(X_train).all():
        raise ValueError("Non-finite values in training features")

    if not np.isfinite(X_test).all():
        raise ValueError("Non-finite values in held-out features")

    y_test = (
        fold_test["label"].eq("fake").astype(int).to_numpy()
    )

    if len(np.unique(y_test)) != 2:
        raise ValueError(
            "Held-out group must contain both real and fake videos"
        )

    # Fit scaling only on genuine training videos.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.10,
    )
    model.fit(X_train_scaled)

    # Larger anomaly scores mean more suspicious.
    scores = -model.decision_function(X_test_scaled)

    return {
        "genuine_train_videos": len(genuine_train),
        "held_out_videos": len(fold_test),
        "held_out_real": int((y_test == 0).sum()),
        "held_out_fake": int((y_test == 1).sum()),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
    }


def main():
    print("=" * 72)
    print("PHASE 13C - GROUP-HELD-OUT CMPC CROSS-VALIDATION")
    print("=" * 72)

    for path in [BASELINE_PATH, CMPC_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {path}"
            )

    baseline_df = pd.read_csv(BASELINE_PATH)
    cmpc_df = pd.read_csv(CMPC_PATH)

    df, cmpc_features = validate_and_merge_datasets(
        baseline_df,
        cmpc_df,
    )

    baseline_features, graph_features = (
        get_baseline_and_graph_features(df)
    )

    missing_segment_features = (
        set(SEGMENT_STD_FEATURES) - set(cmpc_features)
    )
    if missing_segment_features:
        raise ValueError(
            "Missing segment-std features: "
            f"{sorted(missing_segment_features)}"
        )

    cmpc_reduced = [
        col for col in cmpc_features
        if col not in SEGMENT_STD_FEATURES
    ]

    feature_sets = [
        ("baseline_60", baseline_features),
        ("graph_profile_45", graph_features),
        ("cmpc_16", cmpc_features),
        ("cmpc_13", cmpc_reduced),
        (
            "baseline_plus_cmpc_76",
            baseline_features + cmpc_features,
        ),
        (
            "baseline_plus_cmpc_73",
            baseline_features + cmpc_reduced,
        ),
    ]

    # Restrict evaluation to the original training split.
    # Validation and reserved test rows are never evaluated.
    train_df = df[df["split"].eq("train")].copy()

    if len(train_df) != 68:
        raise ValueError(
            f"Expected 68 training videos, found {len(train_df)}"
        )

    groups = sorted(train_df["group_id"].unique())

    if len(groups) != 17:
        raise ValueError(
            f"Expected 17 training groups, found {len(groups)}"
        )

    print(f"Training videos: {len(train_df)}")
    print(f"Training groups/folds: {len(groups)}")
    print(f"Genuine training videos overall: "
          f"{train_df['label'].eq('real').sum()}")
    print(f"Baseline features: {len(baseline_features)}")
    print(f"Graph-profile features: {len(graph_features)}")
    print(f"CMPC features: {len(cmpc_features)}")
    print("Validation videos evaluated: 0")
    print("Reserved test videos evaluated: 0")

    results = []

    # Each group becomes the held-out fold exactly once.
    for fold_number, held_out_group in enumerate(groups, start=1):
        fold_train = train_df[
            train_df["group_id"] != held_out_group
        ].copy()

        fold_test = train_df[
            train_df["group_id"] == held_out_group
        ].copy()

        if set(fold_train["group_id"]) & set(fold_test["group_id"]):
            raise ValueError("Group leakage detected")

        print(
            f"\nFold {fold_number:02d}/{len(groups)} "
            f"- held-out group: {held_out_group}"
        )
        print(
            f"Held-out videos: {len(fold_test)}; "
            f"real={fold_test['label'].eq('real').sum()}, "
            f"fake={fold_test['label'].eq('fake').sum()}"
        )

        for feature_name, columns in feature_sets:
            metrics = evaluate_fold(
                fold_train,
                fold_test,
                columns,
            )

            row = {
                "fold": fold_number,
                "held_out_group": held_out_group,
                "feature_set": feature_name,
                "feature_count": len(columns),
                **metrics,
            }
            results.append(row)

            print(
                f"  {feature_name:27s} "
                f"ROC-AUC={metrics['roc_auc']:.4f} "
                f"PR-AUC={metrics['pr_auc']:.4f}"
            )

    results_df = pd.DataFrame(results)

    summary_df = (
        results_df.groupby("feature_set", sort=False)
        .agg(
            folds=("fold", "count"),
            feature_count=("feature_count", "first"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
        )
        .reset_index()
    )

    # Use population standard deviation for consistency
    # with the earlier Phase 11I reports.
    population_std = (
        results_df.groupby("feature_set", sort=False)
        .agg(
            roc_auc_sd=("roc_auc", lambda x: x.std(ddof=0)),
            pr_auc_sd=("pr_auc", lambda x: x.std(ddof=0)),
        )
        .reset_index()
    )

    summary_df = summary_df.drop(
        columns=["std_roc_auc", "std_pr_auc"]
    ).merge(
        population_std,
        on="feature_set",
        validate="one_to_one",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n" + "=" * 72)
    print("GROUP CROSS-VALIDATION SUMMARY")
    print("=" * 72)
    print(
        summary_df[
            [
                "feature_set",
                "feature_count",
                "folds",
                "mean_roc_auc",
                "roc_auc_sd",
                "mean_pr_auc",
                "pr_auc_sd",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nDetailed results: {RESULTS_PATH}")
    print(f"Summary results:  {SUMMARY_PATH}")
    print("Only the 68 original training videos were evaluated.")
    print("Validation and reserved test splits were not evaluated.")


if __name__ == "__main__":
    main()
