from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "phase11d_expanded_features.csv"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "phase11c_expanded_manifest.csv"
)


EXPECTED_BASELINE_FEATURES = 60
EXPECTED_GRAPH_FEATURES = 45
EXPECTED_TOTAL_FEATURES = 105

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


def fail(message):
    raise ValueError(f"FAIL: {message}")


def main():
    print("=" * 70)
    print("PHASE 11E - EXPANDED FEATURE QUALITY & LEAKAGE AUDIT")
    print("=" * 70)

    if not FEATURE_PATH.exists():
        fail(f"Feature file not found: {FEATURE_PATH}")

    if not MANIFEST_PATH.exists():
        fail(f"Manifest file not found: {MANIFEST_PATH}")

    df = pd.read_csv(FEATURE_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    print("\n1. BASIC DATASET CHECK")
    print("-" * 70)

    print(f"Feature rows: {len(df)}")
    print(f"Manifest rows: {len(manifest)}")

    if len(df) != 100:
        fail(f"Expected 100 feature rows, found {len(df)}")

    if len(manifest) != 100:
        fail(f"Expected 100 manifest rows, found {len(manifest)}")

    if df["video"].nunique() != 100:
        fail("Duplicate video names found in feature dataset.")

    print("Unique videos: PASS")

    print("\n2. FEATURE COUNT CHECK")
    print("-" * 70)

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

    print(f"Baseline features: {len(baseline_features)}")
    print(f"Graph-profile features: {len(graph_features)}")
    print(f"Total ML features: {len(baseline_features) + len(graph_features)}")

    if len(baseline_features) != EXPECTED_BASELINE_FEATURES:
        fail(
            f"Expected {EXPECTED_BASELINE_FEATURES} baseline features, "
            f"found {len(baseline_features)}"
        )

    if len(graph_features) != EXPECTED_GRAPH_FEATURES:
        fail(
            f"Expected {EXPECTED_GRAPH_FEATURES} graph features, "
            f"found {len(graph_features)}"
        )

    if len(baseline_features) + len(graph_features) != EXPECTED_TOTAL_FEATURES:
        fail("Total ML feature count is incorrect.")

    print("Feature count: PASS")

    print("\n3. NUMERIC / FINITE VALUE CHECK")
    print("-" * 70)

    feature_columns = baseline_features + graph_features

    numeric_features = df[feature_columns]

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        fail(f"Non-numeric feature columns found: {non_numeric}")

    if numeric_features.isna().any().any():
        missing_columns = numeric_features.columns[
            numeric_features.isna().any()
        ].tolist()
        fail(f"NaN values found in: {missing_columns}")

    if np.isinf(numeric_features.to_numpy()).any():
        fail("Infinite values found in ML features.")

    print("Numeric features: PASS")
    print("NaN check: PASS")
    print("Infinity check: PASS")

    print("\n4. LABEL / SPLIT CHECK")
    print("-" * 70)

    expected_labels = {"real", "fake"}
    actual_labels = set(df["label"].astype(str))

    print("Labels:")
    print(df["label"].value_counts())

    if actual_labels != expected_labels:
        fail(
            f"Expected labels {expected_labels}, "
            f"found {actual_labels}"
        )

    expected_splits = {"train", "validation", "test"}
    actual_splits = set(df["split"].astype(str))

    print("\nSplits:")
    print(df["split"].value_counts())

    if actual_splits != expected_splits:
        fail(
            f"Expected splits {expected_splits}, "
            f"found {actual_splits}"
        )

    print("Label/split values: PASS")

    print("\n5. GROUP LEAKAGE CHECK")
    print("-" * 70)

    train_groups = set(df.loc[df["split"] == "train", "group_id"])
    validation_groups = set(
        df.loc[df["split"] == "validation", "group_id"]
    )
    test_groups = set(df.loc[df["split"] == "test", "group_id"])

    print(f"Train groups: {len(train_groups)}")
    print(f"Validation groups: {len(validation_groups)}")
    print(f"Test groups: {len(test_groups)}")

    if train_groups & validation_groups:
        fail("Train/validation group overlap detected.")

    if train_groups & test_groups:
        fail("Train/test group overlap detected.")

    if validation_groups & test_groups:
        fail("Validation/test group overlap detected.")

    print("Group leakage: PASS")

    print("\n6. IDENTITY LEAKAGE CHECK")
    print("-" * 70)

    identity_columns = {"identity_a", "identity_b"}

    missing_identity_columns = identity_columns - set(manifest.columns)

    if missing_identity_columns:
        fail(
            "Manifest is missing identity columns: "
            f"{missing_identity_columns}"
        )

    merged = df.merge(
        manifest[
            [
                "video",
                "group_id",
                "split",
                "identity_a",
                "identity_b",
            ]
        ],
        on="video",
        how="left",
        suffixes=("_features", "_manifest"),
        validate="one_to_one",
    )

    if merged["identity_a_manifest"].isna().any():
      fail("Some feature videos are missing manifest identity information.")

    for split_a, split_b in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:
      ids_a = set(
        pd.concat(
            [
                merged.loc[
                    merged["split_features"] == split_a,
                    "identity_a_manifest",
                ],
                merged.loc[
                    merged["split_features"] == split_a,
                    "identity_b_manifest",
                ],
            ]
        )
    )

    ids_b = set(
        pd.concat(
            [
                merged.loc[
                    merged["split_features"] == split_b,
                    "identity_a_manifest",
                ],
                merged.loc[
                    merged["split_features"] == split_b,
                    "identity_b_manifest",
                ],
            ]
        )
    )

    overlap = ids_a & ids_b

    print(
        f"{split_a} vs {split_b} identity overlap: "
        f"{sorted(overlap)}"
    )

    if overlap:
        fail(
            f"Identity leakage detected between "
            f"{split_a} and {split_b}: {sorted(overlap)}"
        )

    print("Identity leakage: PASS")

    print("\n7. FEATURE / MANIFEST CONSISTENCY")
    print("-" * 70)

    if set(df["video"]) != set(manifest["video"]):
        missing_in_features = set(manifest["video"]) - set(df["video"])
        missing_in_manifest = set(df["video"]) - set(manifest["video"])

        fail(
            "Feature/manifest video mismatch. "
            f"Missing in features: {missing_in_features}; "
            f"Missing in manifest: {missing_in_manifest}"
        )

    consistency_columns = [
        "group_id",
        "split",
        "label",
    ]

    for column in consistency_columns:
        feature_values = (
            df.set_index("video")[column]
            .astype(str)
            .sort_index()
        )

        manifest_values = (
            manifest.set_index("video")[column]
            .astype(str)
            .sort_index()
        )

        if not feature_values.equals(manifest_values):
            fail(
                f"Feature/manifest mismatch detected in column: {column}"
            )

    print("Video identity consistency: PASS")
    print("Group consistency: PASS")
    print("Split consistency: PASS")
    print("Label consistency: PASS")

    print("\n8. FINAL SUMMARY")
    print("=" * 70)

    print(f"Rows: {len(df)}")
    print(f"Baseline features: {len(baseline_features)}")
    print(f"Graph-profile features: {len(graph_features)}")
    print(f"Total ML features: {len(feature_columns)}")

    print("\nSplit counts:")
    print(df["split"].value_counts())

    print("\nLabel counts:")
    print(df["label"].value_counts())

    print("\nAll Phase 11E audits: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()