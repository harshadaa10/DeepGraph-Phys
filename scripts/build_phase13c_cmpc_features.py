
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "features"
    / "phase11c_expanded_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "features"
    / "phase13c_cmpc_features.csv"
)

FAILURES_PATH = (
    PROJECT_ROOT / "data" / "features"
    / "phase13c_cmpc_failures.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "models" / "mediapipe"
    / "face_landmarker.task"
)

sys.path.insert(0, str(PROJECT_ROOT))

from src.physiology.video_rgb_signals import (
    extract_video_rgb_signals,
)
from src.physiology.rppg_pipeline import (
    generate_rppg_dataframe,
)
from src.motion.video_motion import (
    extract_video_motion,
)
from src.graph.dynamic_graph import (
    create_dynamic_graph_data,
)
from src.features.cross_modal_graph import (
    extract_cross_modal_graph_features,
)


EXPECTED_FEATURE_COUNT = 16

IDENTITY_COLUMNS = [
    "group_id",
    "identity_a",
    "identity_b",
    "video",
    "label",
    "source_path",
    "split",
]


def extract_cmpc_for_video(video_path):
    """Extract CMPC features and basic QC metadata."""

    rgb_dataframe, rgb_metadata = (
        extract_video_rgb_signals(
            video_path,
            MODEL_PATH,
        )
    )

    fps = float(rgb_metadata["fps"])

    rppg_dataframe = generate_rppg_dataframe(
        rgb_dataframe,
        fps,
        max_gap_frames=6,
    )

    motion_dataframe, motion_metadata = (
        extract_video_motion(
            video_path,
            MODEL_PATH,
        )
    )

    dynamic_data = create_dynamic_graph_data(
        rppg_dataframe,
        motion_dataframe,
    )

    synchronized = dynamic_data["dataframe"]

    if synchronized.empty:
        raise ValueError(
            "No synchronized motion/rPPG samples."
        )

    # Confirm metadata needed by CMPC survived synchronization.
    required_columns = {
        "frame",
        "segment_id",
        "interpolated",
        "forehead_rppg",
        "left_cheek_rppg",
        "right_cheek_rppg",
        "forehead_motion",
        "left_cheek_motion",
        "right_cheek_motion",
    }

    missing = required_columns - set(synchronized.columns)

    if missing:
        raise ValueError(
            f"Synchronized data missing columns: {sorted(missing)}"
        )

    features = extract_cross_modal_graph_features(
        synchronized
    )

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} CMPC features; "
            f"received {len(features)}."
        )

    feature_values = np.asarray(
        list(features.values()),
        dtype=np.float64,
    )

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "CMPC features contain NaN or infinite values."
        )

    interpolated = synchronized["interpolated"].astype(bool)

    qc = {
        "fps": fps,
        "rgb_samples": len(rgb_dataframe),
        "rppg_samples": len(rppg_dataframe),
        "motion_samples": len(motion_dataframe),
        "graph_samples": len(synchronized),
        "rppg_segment_count": int(
            rppg_dataframe["segment_id"].nunique()
        ),
        "synchronized_interpolated_samples": int(
            interpolated.sum()
        ),
        "synchronized_interpolation_rate": float(
            interpolated.mean()
        ),
    }

    return features, qc


def save_progress(records, failures):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(records).to_csv(
        OUTPUT_PATH,
        index=False,
    )

    pd.DataFrame(
        failures,
        columns=["video", "source_path", "error"],
    ).to_csv(
        FAILURES_PATH,
        index=False,
    )


def main():
    print("=" * 70)
    print("Phase 13C — CMPC Feature Dataset")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MODEL_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    missing_columns = (
        set(IDENTITY_COLUMNS) - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing_columns)}"
        )

    if (
        len(manifest) != 100
        or manifest["video"].nunique() != 100
    ):
        raise ValueError(
            "Expected exactly 100 unique manifest videos."
        )

    if manifest[IDENTITY_COLUMNS].isna().any().any():
        raise ValueError(
            "Manifest contains missing identity or split metadata."
        )

    # Resume from prior successful results, if present.
    records = []
    processed = set()

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)

        if not set(IDENTITY_COLUMNS).issubset(existing.columns):
            raise ValueError(
                "Existing CMPC output has an unexpected schema. "
                "Back it up and inspect it before continuing."
            )

        if existing["video"].duplicated().any():
            raise ValueError(
                "Existing CMPC output has duplicate video rows."
            )

        manifest_videos = set(manifest["video"].astype(str))
        existing_videos = set(existing["video"].astype(str))

        if not existing_videos.issubset(manifest_videos):
            raise ValueError(
                "Existing CMPC output contains videos "
                "not present in the current manifest."
            )

        records = existing.to_dict(orient="records")
        processed = existing_videos

        print(
            f"Resuming: {len(processed)} videos already saved."
        )

    failures = []

    if FAILURES_PATH.exists():
        old_failures = pd.read_csv(FAILURES_PATH)

        if {"video", "source_path", "error"}.issubset(
            old_failures.columns
        ):
            failures = old_failures.to_dict(orient="records")

    # Retry previous failures, but don't repeat successful videos.
    for index, row in manifest.iterrows():
        video_name = str(row["video"])

        if video_name in processed:
            print(
                f"[{index + 1}/100] {video_name}: already saved"
            )
            continue

        source_path = str(row["source_path"])
        video_path = PROJECT_ROOT / source_path

        print()
        print("-" * 70)
        print(
            f"[{index + 1}/100] {video_name} | "
            f"{row['split']} | {row['label']} | "
            f"{row['group_id']}"
        )

        try:
            if not video_path.exists():
                raise FileNotFoundError(
                    f"Video not found: {video_path}"
                )

            features, qc = extract_cmpc_for_video(
                video_path
            )

            record = {
                column: row[column]
                for column in IDENTITY_COLUMNS
            }

            # Prefix predictive features for safe identification.
            record.update(features)
            record.update(qc)

            records.append(record)
            processed.add(video_name)

            # Remove an older failure entry if this retry succeeded.
            failures = [
                failure
                for failure in failures
                if failure.get("video") != video_name
            ]

            save_progress(records, failures)

            print(
                f"PASS: {len(features)} CMPC features. "
                f"Saved {len(records)}/100 successful videos."
            )

        except Exception as error:
            failures = [
                failure
                for failure in failures
                if failure.get("video") != video_name
            ]

            failures.append({
                "video": video_name,
                "source_path": source_path,
                "error": str(error),
            })

            save_progress(records, failures)

            print(f"FAILED: {error}")
            print("Failure recorded; continuing to next video.")

    # Final validation of successful records.
    if not records:
        save_progress(records, failures)
        raise RuntimeError(
            "No videos were processed successfully."
        )

    result = pd.DataFrame(records)

    feature_columns = [
        column
        for column in result.columns
        if column.startswith("cmpc_")
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} predictive features, "
            f"found {len(feature_columns)}."
        )

    feature_matrix = result[feature_columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(feature_matrix).all():
        raise ValueError(
            "Final CMPC feature matrix contains invalid values."
        )

    if result["video"].duplicated().any():
        raise ValueError(
            "Final CMPC dataset contains duplicate videos."
        )

    # Check saved metadata still agrees with the manifest.
    expected = manifest[
        ["video", "group_id", "label", "split", "source_path"]
    ].copy()

    actual = result[
        ["video", "group_id", "label", "split", "source_path"]
    ].copy()

    check = expected.merge(
        actual,
        on="video",
        how="inner",
        suffixes=("_expected", "_actual"),
        validate="one_to_one",
    )

    for column in [
        "group_id",
        "label",
        "split",
        "source_path",
    ]:
        if not (
            check[f"{column}_expected"].astype(str).to_numpy()
            == check[f"{column}_actual"].astype(str).to_numpy()
        ).all():
            raise ValueError(
                f"Manifest metadata mismatch for {column}."
            )

    print()
    print("=" * 70)
    print("CMPC EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Successful videos: {len(result)}/100")
    print(f"Failed videos: {len(failures)}")
    print(f"Predictive CMPC features: {len(feature_columns)}")

    print("\nRows per split and label:")
    print(
        result.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nSaved dataset:")
    print(OUTPUT_PATH)

    print("\nSaved failure report:")
    print(FAILURES_PATH)

    if len(result) != 100 or failures:
        print(
            "\nWARNING: Extraction is incomplete. "
            "Resolve failures and rerun this script to resume."
        )
    else:
        print(
            "\nSTATUS: All 100 videos processed successfully."
        )


if __name__ == "__main__":
    main()
