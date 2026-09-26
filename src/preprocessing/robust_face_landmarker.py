from src.preprocessing.face_localizer import (
    PrimaryFaceLocalizer,
    crop_face_region,
)

from src.preprocessing.face_landmarker import (
    FaceLandmarkDetector,
    map_landmarks_to_frame,
)


class RobustFaceLandmarkDetector:
    """
    Robust face landmark detection for full-scene videos.

    Strategy:

    1. Try MediaPipe directly on the full frame.

    2. If direct detection fails:
       - localize candidate faces
       - select primary face
       - crop with margin
       - run MediaPipe on crop
       - map landmarks back to original-frame coordinates

    This preserves compatibility with the existing
    DeepGraph-Phys ROI and motion pipelines.
    """

    def __init__(
        self,
        model_path,
        crop_margin=0.70,
    ):

        self.landmarker = (
            FaceLandmarkDetector(
                model_path
            )
        )

        self.localizer = (
            PrimaryFaceLocalizer()
        )

        self.crop_margin = (
            crop_margin
        )

    def detect(self, frame):
        """
        Return landmarks normalized relative
        to the ORIGINAL frame.
        """

        # --------------------------------------
        # Attempt 1:
        # MediaPipe directly on full frame
        # --------------------------------------

        landmarks = (
            self.landmarker.detect(
                frame
            )
        )

        if landmarks is not None:

            return landmarks

        # --------------------------------------
        # Attempt 2:
        # Localize primary face first
        # --------------------------------------

        primary_face = (
            self.localizer.detect_primary_face(
                frame
            )
        )

        if primary_face is None:

            return None

        crop, coordinates = (
            crop_face_region(
                frame,
                primary_face,
                margin=self.crop_margin,
            )
        )

        if (
            crop is None
            or crop.size == 0
        ):
            return None

        crop_landmarks = (
            self.landmarker.detect(
                crop
            )
        )

        if crop_landmarks is None:

            return None

        # --------------------------------------
        # Convert crop landmarks back into
        # original-frame coordinates.
        # --------------------------------------

        mapped_landmarks = (
            map_landmarks_to_frame(
                crop_landmarks,
                coordinates,
                frame.shape,
            )
        )

        return mapped_landmarks

    def close(self):

        self.landmarker.close()