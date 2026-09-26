from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.features.video_pipeline import (
    process_video_to_features,
)


# ==============================================
# Select the FF++ video to test
# ==============================================

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


def main():

    print(
        "\nDeepGraph-Phys — "
        "FF++ End-to-End Video Pipeline Test"
    )

    print("-" * 65)

    print(
        "Video:"
    )

    print(
        VIDEO_PATH
    )

    # ==========================================
    # File validation
    # ==========================================

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: "
            f"{MODEL_PATH}"
        )

    # ==========================================
    # Run complete pipeline
    # ==========================================

    print(
        "\nRunning complete DeepGraph-Phys "
        "pipeline..."
    )

    features, metadata = (
        process_video_to_features(
            VIDEO_PATH,
            MODEL_PATH,
        )
    )

    # ==========================================
    # Print metadata
    # ==========================================

    print(
        "\nPipeline metadata:"
    )

    for key, value in (
        metadata.items()
    ):

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key:40s}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{key:40s}: "
                f"{value}"
            )

    # ==========================================
    # Feature values
    # ==========================================

    feature_values = np.asarray(
        list(
            features.values()
        ),
        dtype=np.float64,
    )

    # ==========================================
    # Generic validation
    # ==========================================

    fps_valid = (
        np.isfinite(
            metadata["fps"]
        )
        and metadata["fps"] > 0
    )

    frames_processed_valid = (
        metadata[
            "frames_processed"
        ] is not None
        and metadata[
            "frames_processed"
        ] > 0
    )

    rgb_frames_valid = (
        metadata[
            "rgb_detected_frames"
        ] > 0
    )

    detection_rate_valid = (
        metadata[
            "face_detection_rate"
        ] is not None
        and 0.0
        <= metadata[
            "face_detection_rate"
        ]
        <= 1.0
    )

    rppg_samples_valid = (
        metadata[
            "rppg_samples"
        ] > 0
    )

    interpolation_rate_valid = (
        0.0
        <= metadata[
            "interpolation_rate"
        ]
        <= 1.0
    )

    motion_samples_valid = (
        metadata[
            "motion_samples"
        ] > 0
    )

    graph_samples_valid = (
        metadata[
            "graph_samples"
        ] > 0
    )

    synchronized_rate_valid = (
        0.0
        <= metadata[
            "synchronized_interpolation_rate"
        ]
        <= 1.0
    )

    exactly_60_features = (
        len(features) == 60
    )

    metadata_feature_count_correct = (
        metadata[
            "feature_count"
        ] == 60
    )

    all_features_finite = (
        np.isfinite(
            feature_values
        ).all()
    )

    # ==========================================
    # Print validation
    # ==========================================

    print(
        "\nValidation checks:"
    )

    print(
        "FPS valid:",
        fps_valid
    )

    print(
        "Frames processed available:",
        frames_processed_valid
    )

    print(
        "RGB face frames available:",
        rgb_frames_valid
    )

    print(
        "Face detection rate valid:",
        detection_rate_valid
    )

    print(
        "rPPG samples available:",
        rppg_samples_valid
    )

    print(
        "Interpolation rate valid:",
        interpolation_rate_valid
    )

    print(
        "Motion samples available:",
        motion_samples_valid
    )

    print(
        "Graph samples available:",
        graph_samples_valid
    )

    print(
        "Synchronized interpolation rate valid:",
        synchronized_rate_valid
    )

    print(
        "Exactly 60 features:",
        exactly_60_features
    )

    print(
        "Metadata feature count = 60:",
        metadata_feature_count_correct
    )

    print(
        "All feature values finite:",
        all_features_finite
    )

    # ==========================================
    # Overall result
    # ==========================================

    all_checks_passed = all(
        [
            fps_valid,
            frames_processed_valid,
            rgb_frames_valid,
            detection_rate_valid,
            rppg_samples_valid,
            interpolation_rate_valid,
            motion_samples_valid,
            graph_samples_valid,
            synchronized_rate_valid,
            exactly_60_features,
            metadata_feature_count_correct,
            all_features_finite,
        ]
    )

    print(
        "\nAll validation checks passed:",
        all_checks_passed
    )

    # ==========================================
    # Feature preview
    # ==========================================

    print(
        "\nFirst 10 features:"
    )

    for index, (
        feature_name,
        value,
    ) in enumerate(
        features.items()
    ):

        if index >= 10:
            break

        print(
            f"{feature_name:42s} "
            f"{value:.8f}"
        )

    print(
        "\nFF++ end-to-end video pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()