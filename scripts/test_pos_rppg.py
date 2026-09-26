from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.physiology.rppg_pipeline import (
    generate_rppg_dataframe,
)


RGB_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_fake_rgb.csv"
)

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

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_fake_rppg.csv"
)

OUTPUT_PLOT = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "fake_pos_rppg.png"
)


RPPG_COLUMNS = [
    "forehead_rppg",
    "left_cheek_rppg",
    "right_cheek_rppg",
]


def main():

    print(
        "\nDeepGraph-Phys — "
        "Segment-Aware Fake POS rPPG Test"
    )

    print("-" * 65)

    # ==========================================
    # Load RGB
    # ==========================================

    if not RGB_PATH.exists():
        raise FileNotFoundError(
            f"RGB CSV not found: {RGB_PATH}"
        )

    rgb_dataframe = pd.read_csv(
        RGB_PATH
    )

    print(
        "RGB data loaded:",
        rgb_dataframe.shape
    )

    # ==========================================
    # Read true source FPS
    # ==========================================

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: "
            f"{VIDEO_PATH}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_video_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    print(
        f"Source video FPS: "
        f"{fps:.2f}"
    )

    print(
        "Source video frames:",
        total_video_frames
    )

    # ==========================================
    # Generate segment-aware rPPG
    # ==========================================

    print(
        "\nGenerating segment-aware "
        "POS rPPG signals..."
    )

    rppg_dataframe = (
        generate_rppg_dataframe(
            rgb_dataframe,
            fps=fps,
            max_gap_frames=6,
            window_seconds=1.6,
        )
    )

    print(
        "\nrPPG table shape:",
        rppg_dataframe.shape
    )

    print(
        "\nColumns:"
    )

    for column in (
        rppg_dataframe.columns
    ):

        print(
            " -",
            column
        )

    # ==========================================
    # Segment information
    # ==========================================

    segment_counts = (
        rppg_dataframe[
            "segment_id"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nValid POS segments:",
        len(segment_counts)
    )

    print(
        "\nSegment lengths:"
    )

    for (
        segment_id,
        count,
    ) in segment_counts.items():

        segment_rows = (
            rppg_dataframe[
                rppg_dataframe[
                    "segment_id"
                ]
                == segment_id
            ]
        )

        first_frame = int(
            segment_rows[
                "frame"
            ].iloc[0]
        )

        last_frame = int(
            segment_rows[
                "frame"
            ].iloc[-1]
        )

        print(
            f"Segment {segment_id}: "
            f"{count} samples | "
            f"frames "
            f"{first_frame}–"
            f"{last_frame}"
        )

    # ==========================================
    # Interpolation information
    # ==========================================

    interpolated_frames = int(
        rppg_dataframe[
            "interpolated"
        ].sum()
    )

    interpolation_rate = (
        interpolated_frames
        / len(rppg_dataframe)
        if len(rppg_dataframe) > 0
        else 0.0
    )

    print(
        "\nInterpolated rPPG frames:",
        interpolated_frames
    )

    print(
        f"Interpolation rate: "
        f"{interpolation_rate * 100:.2f}%"
    )

    # ==========================================
    # rPPG statistics
    # ==========================================

    print(
        "\nrPPG statistics:"
    )

    for column in (
        RPPG_COLUMNS
    ):

        print(
            f"{column:22s} "
            f"mean="
            f"{rppg_dataframe[column].mean():.6f} "
            f"std="
            f"{rppg_dataframe[column].std():.6f}"
        )

    print(
        "\nCross-region correlation:"
    )

    print(
        rppg_dataframe[
            RPPG_COLUMNS
        ].corr().to_string()
    )

    # ==========================================
    # Validation
    # ==========================================

    fps_valid = (
        np.isfinite(fps)
        and fps > 0
    )

    samples_available = (
        len(rppg_dataframe) > 0
    )

    multiple_segments_available = (
        len(segment_counts) >= 1
    )

    segment_lengths_valid = (
        segment_counts >=
        int(
            round(
                1.6 * fps
            )
        )
    ).all()

    frames_unique = (
        rppg_dataframe[
            "frame"
        ].is_unique
    )

    frames_monotonic = (
        rppg_dataframe[
            "frame"
        ].is_monotonic_increasing
    )

    all_rppg_finite = (
        np.isfinite(
            rppg_dataframe[
                RPPG_COLUMNS
            ].values
        ).all()
    )

    segment_ids_valid = (
        rppg_dataframe[
            "segment_id"
        ]
        .notna()
        .all()
    )

    print(
        "\nValidation checks:"
    )

    print(
        "Source FPS valid:",
        fps_valid
    )

    print(
        "rPPG samples available:",
        samples_available
    )

    print(
        "Valid POS segments available:",
        multiple_segments_available
    )

    print(
        "Every segment >= POS window:",
        segment_lengths_valid
    )

    print(
        "Frame IDs unique:",
        frames_unique
    )

    print(
        "Frame IDs increasing:",
        frames_monotonic
    )

    print(
        "All rPPG finite:",
        all_rppg_finite
    )

    print(
        "Segment IDs valid:",
        segment_ids_valid
    )

    all_checks_passed = all(
        [
            fps_valid,
            samples_available,
            multiple_segments_available,
            segment_lengths_valid,
            frames_unique,
            frames_monotonic,
            all_rppg_finite,
            segment_ids_valid,
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

    rppg_dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        "\nSaved rPPG CSV:"
    )

    print(
        OUTPUT_CSV
    )

    # ==========================================
    # Plot
    # ==========================================

    plt.figure(
        figsize=(12, 6)
    )

    for column in (
        RPPG_COLUMNS
    ):

        plt.plot(
            rppg_dataframe[
                "time_seconds"
            ],
            rppg_dataframe[
                column
            ],
            label=column,
            alpha=0.8,
        )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Normalized POS rPPG"
    )

    plt.title(
        "FF++ Deepfake "
        "Segment-Aware POS rPPG"
    )

    plt.legend()

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
        "\nSaved rPPG plot:"
    )

    print(
        OUTPUT_PLOT
    )

    print(
        "\nSegment-aware POS rPPG "
        "test completed."
    )


if __name__ == "__main__":
    main()