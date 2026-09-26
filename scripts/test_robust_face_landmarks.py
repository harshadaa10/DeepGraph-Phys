from pathlib import Path
import sys

import cv2


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.preprocessing.robust_face_landmarker import (
    RobustFaceLandmarkDetector,
)

from src.preprocessing.face_landmarker import (
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "ffpp_033_robust_landmarks.jpg"
)


def main():

    print(
        "\nDeepGraph-Phys — "
        "Robust Landmark Test"
    )

    print("-" * 60)

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    middle_frame = (
        total_frames // 2
    )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        middle_frame,
    )

    success, frame = (
        capture.read()
    )

    capture.release()

    if not success:

        raise RuntimeError(
            "Could not read video frame."
        )

    print(
        "Frame:",
        middle_frame,
    )

    print(
        "Resolution:",
        frame.shape[1],
        "x",
        frame.shape[0],
    )

    detector = (
        RobustFaceLandmarkDetector(
            str(MODEL_PATH)
        )
    )

    landmarks = (
        detector.detect(
            frame
        )
    )

    detector.close()

    print(
        "\nFace detected:",
        landmarks is not None,
    )

    if landmarks is None:

        return

    print(
        "Landmarks:",
        len(landmarks),
    )

    x_values = [
        landmark.x
        for landmark in landmarks
    ]

    y_values = [
        landmark.y
        for landmark in landmarks
    ]

    print(
        "\nMapped coordinate range:"
    )

    print(
        "X:",
        min(x_values),
        "to",
        max(x_values),
    )

    print(
        "Y:",
        min(y_values),
        "to",
        max(y_values),
    )

    valid_coordinates = all(
        0.0 <= landmark.x <= 1.0
        and
        0.0 <= landmark.y <= 1.0
        for landmark in landmarks
    )

    print(
        "\nValidation checks:"
    )

    print(
        "Face detected:",
        landmarks is not None,
    )

    print(
        "478 landmarks:",
        len(landmarks) == 478,
    )

    print(
        "Coordinates inside frame:",
        valid_coordinates,
    )

    annotated = draw_landmarks(
        frame,
        landmarks,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(OUTPUT_PATH),
        annotated,
    )

    print(
        "\nSaved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()