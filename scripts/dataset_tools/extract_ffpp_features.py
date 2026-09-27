from pathlib import Path
import argparse
import sys
import traceback

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.features.video_pipeline import (
    process_video_to_features,
)


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ffpp_manifest_splits.csv"
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
    / "ffpp_features.csv"
)

FAILURE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "ffpp_feature_failures.csv"
)


MANIFEST_COLUMNS = [
    "video_name",
    "video_path",
    "label",
    "dataset",
    "manipulation",
    "source_id",
    "target_id",
    "group_id",
    "split",
]


QC_COLUMNS = [
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
]


def load_manifest():
    """
    Load and validate the FF++ manifest.
    """

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
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

    if manifest["video_name"].duplicated().any():
        duplicates = (
            manifest.loc[
                manifest[
                    "video_name"
                ].duplicated(
                    keep=False
                ),
                "video_name",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate video names found "
            f"in manifest: {duplicates}"
        )

    return manifest


def load_completed_videos():
    """
    Read the existing feature dataset so a
    stopped batch can resume without recomputing
    completed videos.
    """

    if not OUTPUT_PATH.exists():
        return set()

    dataframe = pd.read_csv(
        OUTPUT_PATH
    )

    if (
        "video_name"
        not in dataframe.columns
    ):
        raise ValueError(
            "Existing FF++ feature file does "
            "not contain video_name."
        )

    return set(
        dataframe[
            "video_name"
        ].astype(str)
    )


def save_success_record(
    record,
):
    """
    Append one successful video immediately.

    This makes extraction resume-safe.
    """

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        [record]
    )

    write_header = (
        not OUTPUT_PATH.exists()
        or OUTPUT_PATH.stat().st_size == 0
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        mode="a",
        header=write_header,
        index=False,
    )


def save_failure_record(
    record,
):
    """
    Append one failed video immediately.
    """

    FAILURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        [record]
    )

    write_header = (
        not FAILURE_PATH.exists()
        or FAILURE_PATH.stat().st_size == 0
    )

    dataframe.to_csv(
        FAILURE_PATH,
        mode="a",
        header=write_header,
        index=False,
    )


def resolve_video_path(
    manifest_video_path,
):
    """
    Resolve the relative path stored in the
    FF++ manifest against the project root.
    """

    path = Path(
        str(manifest_video_path)
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path


def validate_features(
    features,
):
    """
    Validate the final ML feature vector.
    """

    if len(features) != 60:
        raise ValueError(
            "Expected 60 ML features, "
            f"received {len(features)}."
        )

    values = np.asarray(
        list(
            features.values()
        ),
        dtype=np.float64,
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Feature vector contains "
            "NaN or infinite values."
        )


def process_manifest_row(
    row,
):
    """
    Process one FF++ manifest row.
    """

    video_path = resolve_video_path(
        row["video_path"]
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    features, metadata = (
        process_video_to_features(
            video_path=video_path,
            model_path=MODEL_PATH,
        )
    )

    validate_features(
        features
    )

    # ------------------------------------------
    # Dataset identity / split information
    # ------------------------------------------

    record = {
        "video_name":
            row["video_name"],

        "video_path":
            row["video_path"],

        "label":
            row["label"],

        "dataset":
            row["dataset"],

        "manipulation":
            row["manipulation"],

        "source_id":
            row["source_id"],

        "target_id":
            row["target_id"],

        "group_id":
            row["group_id"],

        "split":
            row["split"],
    }

    # ------------------------------------------
    # QC metadata
    # ------------------------------------------

    for column in QC_COLUMNS:

        record[column] = (
            metadata.get(
                column,
                np.nan,
            )
        )

    # ------------------------------------------
    # Exactly 60 model features
    # ------------------------------------------

    record.update(
        features
    )

    return record


def validate_saved_dataset():
    """
    Validate the accumulated FF++ feature CSV.
    """

    if not OUTPUT_PATH.exists():
        return

    dataframe = pd.read_csv(
        OUTPUT_PATH
    )

    if len(dataframe) == 0:
        return

    if dataframe[
        "video_name"
    ].duplicated().any():
        raise ValueError(
            "Duplicate videos detected in "
            "FF++ feature dataset."
        )

    non_feature_columns = (
        MANIFEST_COLUMNS
        + QC_COLUMNS
    )

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in non_feature_columns
    ]

    if len(feature_columns) != 60:
        raise ValueError(
            "Saved dataset does not contain "
            "exactly 60 ML features. "
            f"Found {len(feature_columns)}."
        )

    feature_values = (
        dataframe[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "Saved FF++ dataset contains "
            "invalid feature values."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe FF++ feature "
            "extraction for DeepGraph-Phys."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of NEW videos "
            "to process in this run."
        ),
    )

    args = parser.parse_args()

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise ValueError(
            "--limit must be greater than 0."
        )

    print(
        "\nDeepGraph-Phys — "
        "FF++ Batch Feature Extraction"
    )

    print("-" * 65)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "MediaPipe model not found: "
            f"{MODEL_PATH}"
        )

    manifest = load_manifest()

    completed_videos = (
        load_completed_videos()
    )

    print(
        "Manifest videos:",
        len(manifest)
    )

    print(
        "Already completed:",
        len(completed_videos)
    )

    remaining_count = int(
        (
            ~manifest[
                "video_name"
            ].astype(str).isin(
                completed_videos
            )
        ).sum()
    )

    print(
        "Remaining:",
        remaining_count
    )

    if args.limit is not None:
        print(
            "New-video limit:",
            args.limit
        )

    processed_this_run = 0
    successful_this_run = 0
    failed_this_run = 0

    for manifest_index, row in (
        manifest.iterrows()
    ):

        video_name = str(
            row["video_name"]
        )

        if video_name in completed_videos:
            continue

        if (
            args.limit is not None
            and processed_this_run
            >= args.limit
        ):
            break

        processed_this_run += 1

        print(
            "\n"
            + "=" * 65
        )

        print(
            f"Batch item: "
            f"{manifest_index + 1}/"
            f"{len(manifest)}"
        )

        print(
            "Video:",
            video_name
        )

        print(
            "Label:",
            row["label"]
        )

        print(
            "Split:",
            row["split"]
        )

        print(
            "Group:",
            row["group_id"]
        )

        try:

            record = (
                process_manifest_row(
                    row
                )
            )

            save_success_record(
                record
            )

            completed_videos.add(
                video_name
            )

            successful_this_run += 1

            print(
                "\nSUCCESS"
            )

            print(
                "Features:",
                record[
                    "feature_count"
                ]
            )

            print(
                "Face detection rate:",
                f"{record['face_detection_rate']:.4f}"
            )

            print(
                "rPPG segments:",
                record[
                    "rppg_segment_count"
                ]
            )

            print(
                "Graph samples:",
                record[
                    "graph_samples"
                ]
            )

        except Exception as error:

            failed_this_run += 1

            failure_record = {
                "video_name":
                    video_name,

                "video_path":
                    row["video_path"],

                "label":
                    row["label"],

                "split":
                    row["split"],

                "group_id":
                    row["group_id"],

                "error_type":
                    type(error).__name__,

                "error":
                    str(error),
            }

            save_failure_record(
                failure_record
            )

            print(
                "\nFAILED"
            )

            print(
                type(error).__name__,
                ":",
                error
            )

            print(
                "\nTraceback:"
            )

            traceback.print_exc()

    # ==========================================
    # Validate everything saved so far
    # ==========================================

    validate_saved_dataset()

    final_completed = (
        load_completed_videos()
    )

    print(
        "\n"
        + "=" * 65
    )

    print(
        "Run summary"
    )

    print(
        "Processed this run:",
        processed_this_run
    )

    print(
        "Successful this run:",
        successful_this_run
    )

    print(
        "Failed this run:",
        failed_this_run
    )

    print(
        "Total completed:",
        len(final_completed)
    )

    print(
        "Total manifest videos:",
        len(manifest)
    )

    print(
        "Remaining:",
        len(manifest)
        - len(final_completed)
    )

    if OUTPUT_PATH.exists():

        print(
            "\nFeature dataset:"
        )

        print(
            OUTPUT_PATH
        )

    if FAILURE_PATH.exists():

        print(
            "\nFailure log:"
        )

        print(
            FAILURE_PATH
        )

    print(
        "\nBatch extraction checkpoint "
        "completed."
    )


if __name__ == "__main__":
    main()