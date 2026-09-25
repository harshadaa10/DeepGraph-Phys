from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceLandmarkDetector:
    """
    Detect facial landmarks using MediaPipe Face Landmarker.
    """

    def __init__(self, model_path):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found: {self.model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(self.model_path)
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame):
        """
        Detect facial landmarks from an OpenCV BGR frame.
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]

    def close(self):
        self.detector.close()


def draw_landmarks(frame, landmarks):
    """
    Draw detected facial landmarks on an OpenCV frame.
    """

    output = frame.copy()

    height, width = output.shape[:2]

    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(
                output,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

    return output