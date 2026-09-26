from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.motion.video_motion import (
    extract_video_motion,
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

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_fake_motion.csv"
)

OUTPUT_PLOT = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "fake_motion_signals.png"
)


def main():

    print(
        "\nDeepGraph-Phys — "
        "FF++ Fake Facial Motion Test"
    )

    print("-" * 60)

    print(
        "Processing:"
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
    # Extract motion
    # ==========================================

    dataframe, metadata = (
        extract_video_motion(
            VIDEO_PATH,
            MODEL_PATH,
        )
    )

    fps = float(
        metadata["fps"]
    )

    frames_processed = int(
        metadata["frames_processed"]
    )

    frames_with_face = int(
        metadata["frames_with_face"]
    )

    motion_samples = len(
        dataframe
    )

    face_detection_rate = (
        frames_with_face
        / frames_processed
        if frames_processed > 0
        else 0.0
    )

    # ==========================================
    # Basic information
    # ==========================================

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
        "Frames with face:",
        frames_with_face
    )

    print(
        f"Face detection rate: "
        f"{face_detection_rate * 100:.2f}%"
    )

    print(
        "Motion samples:",
        motion_samples
    )

    print(
        "\nMotion table shape:",
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

    # ==========================================
    # Motion columns
    # ==========================================

    motion_columns = [
        column
        for column
        in dataframe.columns
        if column.endswith(
            "_motion"
        )
    ]

    print(
        "\nMean regional motion:"
    )

    for column in (
        motion_columns
    ):

        print(
            f"{column:20s} "
            f"{dataframe[column].mean():.6f}"
        )

    # ==========================================
    # Frame continuity diagnostics
    # ==========================================

    frames = (
        dataframe["frame"]
        .astype(int)
        .to_numpy()
    )

    if len(frames) > 1:

        frame_differences = (
            np.diff(frames)
        )

        consecutive_transitions = int(
            np.sum(
                frame_differences == 1
            )
        )

        nonconsecutive_transitions = int(
            np.sum(
                frame_differences > 1
            )
        )

        largest_frame_difference = int(
            np.max(
                frame_differences
            )
        )

    else:

        consecutive_transitions = 0
        nonconsecutive_transitions = 0
        largest_frame_difference = 0

    print(
        "\nMotion frame continuity:"
    )

    if motion_samples > 0:

        print(
            "First motion frame:",
            frames[0]
        )

        print(
            "Last motion frame:",
            frames[-1]
        )

    print(
        "Consecutive transitions:",
        consecutive_transitions
    )

    print(
        "Non-consecutive transitions:",
        nonconsecutive_transitions
    )

    print(
        "Largest motion-frame difference:",
        largest_frame_difference
    )

    # ==========================================
    # Validation
    # ==========================================

    fps_valid = (
        np.isfinite(fps)
        and fps > 0
    )

    frames_processed_valid = (
        frames_processed > 0
    )

    face_frames_valid = (
        frames_with_face > 0
    )

    motion_samples_valid = (
        motion_samples > 0
    )

    expected_max_motion_samples = max(
        frames_with_face - 1,
        0,
    )

    motion_count_valid = (
        motion_samples
        <= expected_max_motion_samples
    )

    motion_values_finite = (
        len(motion_columns) > 0
        and np.isfinite(
            dataframe[
                motion_columns
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
        frames_processed_valid
    )

    print(
        "Face frames available:",
        face_frames_valid
    )

    print(
        "Motion samples available:",
        motion_samples_valid
    )

    print(
        "Motion sample count plausible:",
        motion_count_valid
    )

    print(
        "Motion values finite:",
        motion_values_finite
    )

    print(
        "Motion frame IDs unique:",
        frame_numbers_unique
    )

    print(
        "Motion frame IDs increasing:",
        frame_numbers_monotonic
    )

    all_checks_passed = all(
        [
            fps_valid,
            frames_processed_valid,
            face_frames_valid,
            motion_samples_valid,
            motion_count_valid,
            motion_values_finite,
            frame_numbers_unique,
            frame_numbers_monotonic,
        ]
    )

    print(
        "\nAll validation checks passed:",
        all_checks_passed
    )

    # ==========================================
    # Save CSV
    # ==========================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ==========================================
    # Visualization
    # ==========================================

    plt.figure(
        figsize=(12, 6)
    )

    time = dataframe[
        "time_seconds"
    ]

    for region in [
        "forehead",
        "left_cheek",
        "right_cheek",
        "left_eye",
        "right_eye",
        "nose",
        "mouth",
    ]:

        plt.plot(
            time,
            dataframe[
                f"{region}_motion"
            ],
            label=region,
            alpha=0.8,
        )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Normalized landmark displacement"
    )

    plt.title(
        "FF++ Deepfake Regional "
        "Facial Motion Signals"
    )

    plt.legend(
        ncol=2
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    OUTPUT_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_PLOT,
        dpi=150,
    )

    plt.close()

    print(
        "\nSaved motion CSV:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        "\nSaved motion plot:"
    )

    print(
        OUTPUT_PLOT
    )

    print(
        "\nFake facial motion extraction "
        "completed successfully."
    )


if __name__ == "__main__":
    main()