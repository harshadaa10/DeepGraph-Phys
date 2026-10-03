from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "phase11c_expanded_manifest.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "phase11d_expanded_features.csv"
)


# =========================================================
# Import the validated DeepGraph-Phys pipeline
# =========================================================

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.features.video_pipeline import (
    process_video_to_features,
)


# =========================================================
# Expected manifest columns
# =========================================================

MANIFEST_COLUMNS = [
    "group_id",
    "identity_a",
    "identity_b",
    "video",
    "label",
    "source_path",
    "split",
]


# =========================================================
# Feature validation helper
# =========================================================

def validate_feature_dictionary(
    features,
    expected_count,
    feature_type,
):
    """
    Validate one feature dictionary.
    """

    if not isinstance(
        features,
        dict,
    ):
        raise ValueError(
            f"{feature_type} features must be a dictionary."
        )

    if len(features) != expected_count:
        raise ValueError(
            f"{feature_type} feature count mismatch: "
            f"received {len(features)}, "
            f"expected {expected_count}."
        )

    values = np.asarray(
        list(features.values()),
        dtype=np.float64,
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            f"{feature_type} features contain "
            "NaN or infinite values."
        )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("Phase 11D - Expanded Feature Extraction")
    print("=" * 70)

    # -----------------------------------------------------
    # Check required files
    # -----------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found:\n{MODEL_PATH}"
        )

    # -----------------------------------------------------
    # Load manifest
    # -----------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Manifest rows: {len(manifest)}"
    )

    missing_columns = [
        column
        for column in MANIFEST_COLUMNS
        if column not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(
            "Manifest is missing columns: "
            f"{missing_columns}"
        )

    if len(manifest) != 100:
        raise ValueError(
            "Expected exactly 100 videos "
            f"in expanded manifest, "
            f"received {len(manifest)}."
        )

    if manifest["video"].nunique() != 100:
        raise ValueError(
            "Manifest does not contain "
            "100 unique videos."
        )

    # -----------------------------------------------------
    # Resume-safe loading
    # -----------------------------------------------------

    records = []

    processed_videos = set()

    if OUTPUT_PATH.exists():

        existing = pd.read_csv(
            OUTPUT_PATH
        )

        if "video" in existing.columns:

            processed_videos = set(
                existing["video"].astype(str)
            )

            records = existing.to_dict(
                orient="records"
            )

            print(
                f"Existing partial results found: "
                f"{len(records)} videos"
            )

        else:

            print(
                "Existing output does not contain "
                "the required video column. "
                "Starting fresh."
            )

    total_videos = len(manifest)

    # -----------------------------------------------------
    # Process each video
    # -----------------------------------------------------

    for index, row in manifest.iterrows():

        video_name = str(
            row["video"]
        )

        # Skip videos already successfully processed.
        if video_name in processed_videos:

            print(
                f"[{index + 1}/{total_videos}] "
                f"{video_name} - already processed, skipping"
            )

            continue

        video_path = (
            PROJECT_ROOT
            / row["source_path"]
        )

        print()
        print("-" * 70)
        print(
            f"[{index + 1}/{total_videos}] "
            f"{video_name}"
        )

        print(
            f"Group: {row['group_id']} | "
            f"Split: {row['split']} | "
            f"Label: {row['label']}"
        )

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found:\n{video_path}"
            )

        # -------------------------------------------------
        # Run existing validated pipeline
        # -------------------------------------------------

        (
            baseline_features,
            graph_features,
            metadata,
        ) = process_video_to_features(
            video_path=video_path,
            model_path=MODEL_PATH,
            include_graph_profile=True,
        )

        # -------------------------------------------------
        # Validate baseline features
        # -------------------------------------------------

        validate_feature_dictionary(
            baseline_features,
            expected_count=60,
            feature_type="Baseline",
        )

        # -------------------------------------------------
        # Validate graph features
        # -------------------------------------------------

        validate_feature_dictionary(
            graph_features,
            expected_count=45,
            feature_type="Graph-profile",
        )

        # -------------------------------------------------
        # Create record
        # -------------------------------------------------

        record = {}

        # Experimental metadata
        record["group_id"] = row["group_id"]
        record["identity_a"] = row["identity_a"]
        record["identity_b"] = row["identity_b"]
        record["video"] = row["video"]
        record["label"] = row["label"]
        record["source_path"] = row["source_path"]
        record["split"] = row["split"]

        # -------------------------------------------------
        # 60 baseline features
        # -------------------------------------------------

        for name, value in baseline_features.items():
            record[name] = value

        # -------------------------------------------------
        # 45 graph-profile features
        # -------------------------------------------------

        for name, value in graph_features.items():
            record[name] = value

        # -------------------------------------------------
        # Quality-control metadata
        # -------------------------------------------------

        record["fps"] = metadata["fps"]

        record["frames_processed"] = (
            metadata["frames_processed"]
        )

        record["rgb_detected_frames"] = (
            metadata["rgb_detected_frames"]
        )

        record["face_detection_rate"] = (
            metadata["face_detection_rate"]
        )

        record["rppg_samples"] = (
            metadata["rppg_samples"]
        )

        record["rppg_segment_count"] = (
            metadata["rppg_segment_count"]
        )

        record["shortest_rppg_segment"] = (
            metadata["shortest_rppg_segment"]
        )

        record["longest_rppg_segment"] = (
            metadata["longest_rppg_segment"]
        )

        record["interpolated_frames"] = (
            metadata["interpolated_frames"]
        )

        record["interpolation_rate"] = (
            metadata["interpolation_rate"]
        )

        record["motion_samples"] = (
            metadata["motion_samples"]
        )

        record["graph_samples"] = (
            metadata["graph_samples"]
        )

        record["consecutive_graph_transitions"] = (
            metadata["consecutive_graph_transitions"]
        )

        record["graph_gap_transitions"] = (
            metadata["graph_gap_transitions"]
        )

        record["largest_graph_frame_difference"] = (
            metadata["largest_graph_frame_difference"]
        )

        record["synchronized_interpolated_frames"] = (
            metadata[
                "synchronized_interpolated_frames"
            ]
        )

        record["synchronized_interpolation_rate"] = (
            metadata[
                "synchronized_interpolation_rate"
            ]
        )

        # -------------------------------------------------
        # Add successful record
        # -------------------------------------------------

        records.append(
            record
        )

        processed_videos.add(
            video_name
        )

        # -------------------------------------------------
        # Save progress immediately
        # -------------------------------------------------

        partial_dataframe = pd.DataFrame(
            records
        )

        partial_dataframe.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        print(
            "Baseline features: 60"
        )

        print(
            "Graph-profile features: 45"
        )

        print(
            "Status: PASS"
        )

        print(
            f"Progress saved: "
            f"{len(records)}/{total_videos}"
        )

    # =====================================================
    # Create final dataframe
    # =====================================================

    dataframe = pd.DataFrame(
        records
    )

    # =====================================================
    # Final dataset validation
    # =====================================================

    if len(dataframe) != 100:
        raise ValueError(
            "Final dataset does not contain "
            f"100 rows. Received {len(dataframe)}."
        )

    # -----------------------------------------------------
    # QC metadata must NOT be counted as ML features
    # -----------------------------------------------------

    QC_METADATA_COLUMNS = {
        "motion_samples",
        "graph_samples",
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
        "consecutive_graph_transitions",
        "graph_gap_transitions",
        "largest_graph_frame_difference",
        "synchronized_interpolated_frames",
        "synchronized_interpolation_rate",
    }

    # -----------------------------------------------------
    # Identify graph-profile features first
    # -----------------------------------------------------

    graph_feature_columns = [
        column
        for column in dataframe.columns
        if column.startswith(
            "motion_gsp_"
        )
        or column.startswith(
            "physiology_gsp_"
        )
    ]

    # -----------------------------------------------------
    # Identify baseline features
    # -----------------------------------------------------

    baseline_feature_columns = [
        column
        for column in dataframe.columns
        if (
            (
                column.startswith("motion_")
                or column.startswith("physiology_")
            )
            and column not in QC_METADATA_COLUMNS
            and column not in graph_feature_columns
        )
    ]

    # -----------------------------------------------------
    # Validate feature counts
    # -----------------------------------------------------

    if len(baseline_feature_columns) != 60:
        raise ValueError(
            "Expected 60 baseline features, "
            f"received {len(baseline_feature_columns)}."
        )

    if len(graph_feature_columns) != 45:
        raise ValueError(
            "Expected 45 graph-profile features, "
            f"received {len(graph_feature_columns)}."
        )

    all_ml_features = (
        baseline_feature_columns
        + graph_feature_columns
    )

    if len(all_ml_features) != 105:
        raise ValueError(
            "Expected 105 total ML features, "
            f"received {len(all_ml_features)}."
        )

    # -----------------------------------------------------
    # Check finite values
    # -----------------------------------------------------

    feature_matrix = dataframe[
        all_ml_features
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        feature_matrix
    ).all():
        raise ValueError(
            "Final ML feature matrix contains "
            "NaN or infinite values."
        )

    # -----------------------------------------------------
    # Check metadata
    # -----------------------------------------------------

    if dataframe["video"].nunique() != 100:
        raise ValueError(
            "Final dataset contains duplicate videos."
        )

    if dataframe["group_id"].nunique() != 25:
        raise ValueError(
            "Expected 25 groups, "
            f"received {dataframe['group_id'].nunique()}."
        )

    # =====================================================
    # Print final summary
    # =====================================================

    print()
    print("=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Rows: {len(dataframe)}"
    )

    print(
        f"Baseline features: "
        f"{len(baseline_feature_columns)}"
    )

    print(
        f"Graph-profile features: "
        f"{len(graph_feature_columns)}"
    )

    print(
        f"Total ML features: "
        f"{len(all_ml_features)}"
    )

    print()
    print("Split counts:")

    print(
        dataframe["split"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Label counts:")

    print(
        dataframe["label"]
        .value_counts()
        .sort_index()
    )

    # =====================================================
    # Save final dataset
    # =====================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )

    print()
    print(
        "Phase 11D extraction completed successfully."
    )


if __name__ == "__main__":
    main()

