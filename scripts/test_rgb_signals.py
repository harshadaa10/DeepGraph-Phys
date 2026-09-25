from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.physiology.video_rgb_signals import (
    extract_video_rgb_signals,
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_real_rgb.csv"
)


def main():

    print(
        "\nDeepGraph-Phys — Raw RGB Signal Extraction"
    )

    print("-" * 55)

    print("Processing:")
    print(VIDEO_PATH)

    dataframe, metadata = (
        extract_video_rgb_signals(
            VIDEO_PATH,
            MODEL_PATH
        )
    )

    print("\nVideo information")

    print(
        f"FPS: {metadata['fps']:.2f}"
    )

    print(
        "Frames processed:",
        metadata["total_frames_processed"]
    )

    print(
        "Frames with detected face:",
        metadata["frames_with_face"]
    )

    detection_rate = (
        metadata["frames_with_face"]
        / metadata["total_frames_processed"]
        * 100
        if metadata["total_frames_processed"] > 0
        else 0
    )

    print(
        f"Face detection rate: "
        f"{detection_rate:.2f}%"
    )

    print("\nSignal table shape:")
    print(dataframe.shape)

    print("\nColumns:")

    for column in dataframe.columns:
        print(" -", column)

    print("\nFirst five rows:")

    print(
        dataframe.head().to_string(
            index=False
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nRGB signals saved to:")
    print(OUTPUT_PATH)

    print(
        "\nRaw RGB signal extraction completed."
    )


if __name__ == "__main__":
    main()