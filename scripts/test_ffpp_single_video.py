from pathlib import Path
import sys
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.features.video_pipeline import process_video_to_features


VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "FaceForensics++"
    / "original_sequences"
    / "youtube"
    / "c23"
    / "videos"
    / "033.mp4"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)

MAX_FRAMES = 300


def main():

    print(
        "\nDeepGraph-Phys — "
        "FaceForensics++ Single Video Test"
    )

    print("-" * 65)

    print("Video:")
    print(VIDEO_PATH)

    print(
        "\nMaximum frames:",
        MAX_FRAMES,
    )

    features, metadata = (
        process_video_to_features(
            video_path=str(VIDEO_PATH),
            model_path=str(MODEL_PATH),
            max_frames=MAX_FRAMES,
        )
    )

    print("\nMetadata:")

    for key, value in metadata.items():
        print(
            f"{key}: {value}"
        )

    feature_values = np.array(
        list(features.values()),
        dtype=float,
    )

    print("\nValidation checks:")

    print(
        "Feature count = 60:",
        len(features) == 60,
    )

    print(
        "All features finite:",
        np.isfinite(
            feature_values
        ).all(),
    )

    print(
        "No NaN values:",
        not np.isnan(
            feature_values
        ).any(),
    )

    print(
        "Graph samples > 0:",
        metadata["graph_samples"] > 0,
    )

    print(
        "\nSelected features:"
    )

    selected = [
        "motion_smoothness_mean",
        "motion_temporal_mean",
        "motion_high_ratio_mean",
        "physiology_mean_correlation",
        "physiology_mean_abs_correlation",
        "physiology_smoothness_mean",
        "physiology_temporal_mean",
        "physiology_high_ratio_mean",
    ]

    for name in selected:

        if name in features:
            print(
                f"{name}: "
                f"{features[name]:.8f}"
            )

    print(
        "\nFaceForensics++ single-video "
        "pipeline test completed."
    )


if __name__ == "__main__":
    main()