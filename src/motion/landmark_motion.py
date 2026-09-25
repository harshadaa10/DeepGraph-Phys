import numpy as np

from src.preprocessing.face_regions import (
    REGION_LANDMARKS,
)


MOTION_REGIONS = [
    "forehead",
    "left_cheek",
    "right_cheek",
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
]


def landmarks_to_array(landmarks):
    """
    Convert MediaPipe landmarks into an
    Nx2 array of normalized x/y coordinates.
    """

    return np.array(
        [
            [
                landmark.x,
                landmark.y
            ]
            for landmark in landmarks
        ],
        dtype=np.float64
    )


def calculate_face_scale(points):
    """
    Estimate face size using the spatial extent
    of all facial landmarks.
    """

    x_range = (
        np.max(points[:, 0])
        - np.min(points[:, 0])
    )

    y_range = (
        np.max(points[:, 1])
        - np.min(points[:, 1])
    )

    face_scale = np.sqrt(
        x_range ** 2
        + y_range ** 2
    )

    return max(
        face_scale,
        1e-8
    )


def calculate_region_motion(
    previous_points,
    current_points,
    landmark_indices,
):
    """
    Calculate average landmark displacement
    for one facial region.
    """

    previous_region = previous_points[
        landmark_indices
    ]

    current_region = current_points[
        landmark_indices
    ]

    displacement = (
        current_region
        - previous_region
    )

    magnitude = np.linalg.norm(
        displacement,
        axis=1
    )

    return np.mean(
        magnitude
    )


def calculate_all_region_motion(
    previous_landmarks,
    current_landmarks,
):
    """
    Calculate normalized motion for all
    defined facial regions.
    """

    previous_points = landmarks_to_array(
        previous_landmarks
    )

    current_points = landmarks_to_array(
        current_landmarks
    )

    face_scale = calculate_face_scale(
        current_points
    )

    motion = {}

    for region in MOTION_REGIONS:

        indices = REGION_LANDMARKS[
            region
        ]

        raw_motion = calculate_region_motion(
            previous_points,
            current_points,
            indices,
        )

        normalized_motion = (
            raw_motion / face_scale
        )

        motion[region] = (
            normalized_motion
        )

    return motion