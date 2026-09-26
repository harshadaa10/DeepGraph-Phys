from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.preprocessing.face_landmarker import (
    FaceLandmarkDetector,
    draw_landmarks,
)


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)


def load_middle_frame(video_path):

    capture = cv2.VideoCapture(
        str(video_path)
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    middle_frame = total_frames // 2

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        middle_frame,
    )

    success, frame = capture.read()

    capture.release()

    if not success:
        raise RuntimeError(
            "Could not read middle frame."
        )

    return frame, middle_frame


def main():

    print(
        "\nDeepGraph-Phys — "
        "Face Crop Diagnostic"
    )

    print("-" * 60)

    frame, frame_number = load_middle_frame(
        VIDEO_PATH
    )

    height, width = frame.shape[:2]

    print(
        "Frame number:",
        frame_number,
    )

    print(
        "Original resolution:",
        f"{width}x{height}",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Candidate central crops.
    #
    # Coordinates are fractions of the
    # original frame:
    #
    # x1, y1, x2, y2

    crop_configs = {
        "wide_center": (
            0.20,
            0.00,
            0.65,
            0.75,
        ),

        "medium_center": (
            0.28,
            0.02,
            0.58,
            0.58,
        ),

        "tight_center": (
            0.34,
            0.04,
            0.54,
            0.48,
        ),
    }

    detector = FaceLandmarkDetector(
        str(MODEL_PATH)
    )

    for crop_name, (
        x1_ratio,
        y1_ratio,
        x2_ratio,
        y2_ratio,
    ) in crop_configs.items():

        x1 = int(
            width * x1_ratio
        )

        y1 = int(
            height * y1_ratio
        )

        x2 = int(
            width * x2_ratio
        )

        y2 = int(
            height * y2_ratio
        )

        crop = frame[
            y1:y2,
            x1:x2
        ]

        print(
            f"\nCrop: {crop_name}"
        )

        print(
            "Coordinates:",
            x1,
            y1,
            x2,
            y2,
        )

        print(
            "Crop shape:",
            crop.shape,
        )

        crop_path = (
            OUTPUT_DIR
            / f"ffpp_033_{crop_name}.jpg"
        )

        cv2.imwrite(
            str(crop_path),
            crop,
        )

        landmarks = detector.detect(
            crop
        )

        detected = (
            landmarks is not None
        )

        print(
            "Face detected:",
            detected,
        )

        if detected:

            print(
                "Landmarks:",
                len(landmarks),
            )

            annotated = draw_landmarks(
                crop,
                landmarks,
            )

            output_path = (
                OUTPUT_DIR
                / (
                    "ffpp_033_"
                    f"{crop_name}_landmarks.jpg"
                )
            )

            cv2.imwrite(
                str(output_path),
                annotated,
            )

            print(
                "Landmark image:",
                output_path,
            )

    detector.close()

    print(
        "\nFace crop diagnostic completed."
    )


if __name__ == "__main__":
    main()