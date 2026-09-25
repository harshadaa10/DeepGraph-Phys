from pathlib import Path
import sys

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


VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "real"
    / "sample_real.mp4"
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
    / "sample_real_motion.csv"
)

OUTPUT_PLOT = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "real_motion_signals.png"
)


def main():

    print(
        "\nDeepGraph-Phys — Facial Motion Test"
    )

    print("-" * 50)

    dataframe, metadata = (
        extract_video_motion(
            VIDEO_PATH,
            MODEL_PATH,
        )
    )

    print(
        f"FPS: {metadata['fps']:.2f}"
    )

    print(
        "Frames processed:",
        metadata["frames_processed"]
    )

    print(
        "Frames with face:",
        metadata["frames_with_face"]
    )

    print(
        "Motion samples:",
        len(dataframe)
    )

    print(
        "\nMotion table shape:",
        dataframe.shape
    )

    print("\nColumns:")

    for column in dataframe.columns:
        print(" -", column)

    print("\nMean regional motion:")

    motion_columns = [
        column
        for column in dataframe.columns
        if column.endswith("_motion")
    ]

    for column in motion_columns:

        print(
            f"{column:20s} "
            f"{dataframe[column].mean():.6f}"
        )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ------------------------------
    # Visualization
    # ------------------------------

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
        "Regional Facial Motion Signals"
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
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PLOT,
        dpi=150
    )

    plt.close()

    print("\nSaved motion CSV:")
    print(OUTPUT_CSV)

    print("\nSaved motion plot:")
    print(OUTPUT_PLOT)

    print(
        "\nFacial motion extraction completed."
    )


if __name__ == "__main__":
    main()