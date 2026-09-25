from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.preprocessing.face_landmarker import FaceLandmarkDetector

from src.preprocessing.face_regions import (
    get_all_region_polygons,
    draw_regions,
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
    / "outputs"
    / "figures"
    / "real_face_regions.jpg"
)


def get_middle_frame(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    middle_frame = total_frames // 2

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        middle_frame
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise ValueError(
            "Could not read middle frame."
        )

    return frame


def main():

    print("\nDeepGraph-Phys — Facial ROI Test")
    print("-" * 50)

    frame = get_middle_frame(VIDEO_PATH)

    print("Frame loaded.")
    print("Frame shape:", frame.shape)

    detector = FaceLandmarkDetector(
        MODEL_PATH
    )

    print("Detecting facial landmarks...")

    landmarks = detector.detect(frame)

    if landmarks is None:

        print("No face detected.")

        detector.close()

        return

    print(
        f"Detected {len(landmarks)} landmarks."
    )

    print("Creating facial regions...")

    regions = get_all_region_polygons(
        landmarks,
        frame.shape
    )

    print(
        f"Created {len(regions)} regions."
    )

    for region_name in regions:

        print(" -", region_name)

    output = draw_regions(
        frame,
        regions
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(OUTPUT_PATH),
        output
    )

    detector.close()

    print("\nROI visualization saved to:")
    print(OUTPUT_PATH)

    print(
        "\nFacial ROI test completed successfully."
    )


if __name__ == "__main__":
    main()