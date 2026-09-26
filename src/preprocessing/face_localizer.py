from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class FaceBox:
    """
    Bounding box of a detected face.
    Coordinates refer to the original frame.
    """

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self):
        return self.x + self.width

    @property
    def y2(self):
        return self.y + self.height

    @property
    def area(self):
        return self.width * self.height


class PrimaryFaceLocalizer:
    """
    Lightweight face localization using OpenCV's
    Haar cascade.

    This stage only finds a candidate face region.
    MediaPipe is still used for precise landmarks.
    """

    def __init__(self):

        project_root = Path(__file__).resolve().parents[2]

        cascade_path = (
        project_root
        / "models"
        / "opencv"
        / "haarcascade_frontalface_default.xml"
      )
        if not cascade_path.exists():
         raise FileNotFoundError(
            f"Haar cascade not found: "
            f"{cascade_path}"
        )
         
        self.detector = cv2.CascadeClassifier(
        str(cascade_path)
        )

        if self.detector.empty():
            raise RuntimeError(
                "Could not load OpenCV Haar "
                "face cascade."
            )

    def detect_faces(self, frame):
        """
        Return all detected faces as FaceBox objects.
        """

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.equalizeHist(gray)

        detections = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        faces = []

        for x, y, width, height in detections:

            faces.append(
                FaceBox(
                    x=int(x),
                    y=int(y),
                    width=int(width),
                    height=int(height),
                )
            )

        return faces

    def select_primary_face(
        self,
        frame,
        faces,
    ):
        """
        Select the primary face.

        Score considers:
        1. face size
        2. distance from frame center

        This helps prefer a large central subject
        over small background/display faces.
        """

        if not faces:
            return None

        frame_height, frame_width = (
            frame.shape[:2]
        )

        frame_center_x = (
            frame_width / 2.0
        )

        frame_center_y = (
            frame_height / 2.0
        )

        frame_area = (
            frame_width * frame_height
        )

        max_distance = (
            (
                frame_center_x ** 2
                + frame_center_y ** 2
            )
            ** 0.5
        )

        best_face = None
        best_score = float("-inf")

        for face in faces:

            face_center_x = (
                face.x
                + face.width / 2.0
            )

            face_center_y = (
                face.y
                + face.height / 2.0
            )

            distance = (
                (
                    (
                        face_center_x
                        - frame_center_x
                    )
                    ** 2
                    +
                    (
                        face_center_y
                        - frame_center_y
                    )
                    ** 2
                )
                ** 0.5
            )

            normalized_distance = (
                distance / max_distance
            )

            normalized_area = (
                face.area / frame_area
            )

            # Larger faces are preferred.
            # Faces closer to the frame center
            # receive an additional preference.
            score = (
                normalized_area
                - 0.05
                * normalized_distance
            )

            if score > best_score:

                best_score = score
                best_face = face

        return best_face

    def detect_primary_face(
        self,
        frame,
    ):
        """
        Detect faces and return the selected
        primary face.
        """

        faces = self.detect_faces(frame)

        return self.select_primary_face(
            frame,
            faces,
        )


def expand_face_box(
    face,
    frame_shape,
    margin=0.70,
):
    """
    Expand a face box to include forehead,
    cheeks, jaw, and surrounding facial context.

    The expanded coordinates remain in the
    original-frame coordinate system.
    """

    frame_height, frame_width = (
        frame_shape[:2]
    )

    extra_x = int(
        face.width * margin
    )

    extra_y = int(
        face.height * margin
    )

    x1 = max(
        0,
        face.x - extra_x,
    )

    y1 = max(
        0,
        face.y - extra_y,
    )

    x2 = min(
        frame_width,
        face.x2 + extra_x,
    )

    y2 = min(
        frame_height,
        face.y2 + extra_y,
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def crop_face_region(
    frame,
    face,
    margin=0.70,
):
    """
    Return expanded face crop and its
    original-frame coordinates.
    """

    x1, y1, x2, y2 = expand_face_box(
        face,
        frame.shape,
        margin=margin,
    )

    crop = frame[
        y1:y2,
        x1:x2
    ].copy()

    return (
        crop,
        (x1, y1, x2, y2),
    )