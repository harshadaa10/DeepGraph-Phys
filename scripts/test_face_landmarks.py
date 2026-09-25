from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.preprocessing.face_landmarker import (
    FaceLandmarkDetector,
    draw_landmarks,
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
    / "real_face_landmarks.jpg"
)


def get_middle_frame(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    middle_frame_index = total_frames // 2

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        middle_frame_index
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise ValueError("Could not read middle video frame.")

    return frame


def main():

    print("\nDeepGraph-Phys — Face Landmark Test")
    print("-" * 50)

    print("Loading middle video frame...")

    frame = get_middle_frame(VIDEO_PATH)

    print("Frame loaded.")
    print("Frame shape:", frame.shape)

    print("\nLoading MediaPipe Face Landmarker...")

    detector = FaceLandmarkDetector(MODEL_PATH)

    print("Detecting facial landmarks...")

    landmarks = detector.detect(frame)

    if landmarks is None:

        print("No face detected.")

        detector.close()
        return

    print(f"Face detected.")
    print(f"Number of landmarks: {len(landmarks)}")

    output = draw_landmarks(
        frame,
        landmarks
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

    print("\nLandmark image saved to:")
    print(OUTPUT_PATH)

    print("\nFace landmark test completed successfully.")


if __name__ == "__main__":
    main()