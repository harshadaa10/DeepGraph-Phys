from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.physiology.video_rgb_signals import (
    extract_video_rgb_signals,
)


# ==============================================
# FaceForensics++ Deepfake test video
# ==============================================

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "FaceForensics++"
    / "manipulated_sequences"
    / "Deepfakes"
    / "c23"
    / "videos"
    / "033_097.mp4"
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
    / "sample_fake_rgb.csv"
)


def main():

    print(
        "\nDeepGraph-Phys — "
        "FF++ Fake RGB Signal Extraction"
    )

    print("-" * 60)

    print(
        "Processing:"
    )

    print(
        VIDEO_PATH
    )

    # ==========================================
    # Validate files
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
    # Extract RGB signals
    # ==========================================

    dataframe, metadata = (
        extract_video_rgb_signals(
            VIDEO_PATH,
            MODEL_PATH,
        )
    )

    # ==========================================
    # Video information
    # ==========================================

    fps = float(
        metadata["fps"]
    )

    frames_processed = int(
        metadata[
            "frames_processed"
        ]
    )

    frames_with_face = int(
        metadata[
            "frames_with_face"
        ]
    )

    face_detection_rate = float(
        metadata[
            "face_detection_rate"
        ]
    )

    print(
        "\nVideo information:"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        "Frames processed:",
        frames_processed
    )

    print(
        "Frames with detected face:",
        frames_with_face
    )

    print(
        f"Face detection rate: "
        f"{face_detection_rate * 100:.2f}%"
    )

    # ==========================================
    # Signal information
    # ==========================================

    print(
        "\nSignal table shape:"
    )

    print(
        dataframe.shape
    )

    print(
        "\nColumns:"
    )

    for column in (
        dataframe.columns
    ):

        print(
            " -",
            column
        )

    print(
        "\nFirst five rows:"
    )

    print(
        dataframe.head().to_string(
            index=False
        )
    )

    # ==========================================
    # Basic validation
    # ==========================================

    rgb_columns = [
        column
        for column
        in dataframe.columns
        if column.endswith(
            (
                "_R",
                "_G",
                "_B",
            )
        )
    ]

    fps_valid = (
        np.isfinite(fps)
        and fps > 0
    )

    frames_valid = (
        frames_processed > 0
    )

    detections_valid = (
        frames_with_face > 0
    )

    detection_rate_valid = (
        0.0
        <= face_detection_rate
        <= 1.0
    )

    dataframe_count_correct = (
        len(dataframe)
        == frames_with_face
    )

    rgb_values_finite = (
        len(rgb_columns) > 0
        and np.isfinite(
            dataframe[
                rgb_columns
            ].values
        ).all()
    )

    frame_numbers_unique = (
        dataframe[
            "frame"
        ].is_unique
    )

    frame_numbers_monotonic = (
        dataframe[
            "frame"
        ].is_monotonic_increasing
    )

    print(
        "\nValidation checks:"
    )

    print(
        "FPS valid:",
        fps_valid
    )

    print(
        "Frames processed valid:",
        frames_valid
    )

    print(
        "Face detections available:",
        detections_valid
    )

    print(
        "Face detection rate valid:",
        detection_rate_valid
    )

    print(
        "Dataframe rows match detections:",
        dataframe_count_correct
    )

    print(
        "RGB values finite:",
        rgb_values_finite
    )

    print(
        "Frame numbers unique:",
        frame_numbers_unique
    )

    print(
        "Frame numbers increasing:",
        frame_numbers_monotonic
    )

    all_checks_passed = all(
        [
            fps_valid,
            frames_valid,
            detections_valid,
            detection_rate_valid,
            dataframe_count_correct,
            rgb_values_finite,
            frame_numbers_unique,
            frame_numbers_monotonic,
        ]
    )

    print(
        "\nAll validation checks passed:",
        all_checks_passed
    )

    # ==========================================
    # Save RGB dataframe
    # ==========================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nRGB signals saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nFake RGB signal extraction "
        "completed successfully."
    )


if __name__ == "__main__":
    main()