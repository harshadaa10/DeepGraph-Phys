import cv2
import numpy as np


# MediaPipe Face Landmarker indices used to define
# approximate facial regions for the initial prototype.
REGION_LANDMARKS = {
    "forehead": [
        10, 67, 69, 104, 108,
        151, 337, 299, 297, 338
    ],

    "left_cheek": [
        50, 101, 205, 206,
        207, 187, 123, 116
    ],

    "right_cheek": [
        280, 330, 425, 426,
        427, 411, 352, 345
    ],

    "left_eye": [
        33, 160, 158, 133,
        153, 144
    ],

    "right_eye": [
        362, 385, 387, 263,
        373, 380
    ],

    "nose": [
        168, 197, 5, 4,
        1, 2, 98, 327
    ],

    "mouth": [
        61, 40, 37, 0,
        267, 270, 291,
        321, 314, 17,
        84, 91
    ],
}


def landmarks_to_points(landmarks, frame_shape):
    """
    Convert normalized MediaPipe landmarks into pixel coordinates.
    """

    height, width = frame_shape[:2]

    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    return points


def get_region_polygon(points, landmark_indices):
    """
    Create a convex polygon from selected landmark indices.
    """

    selected_points = np.array(
        [points[index] for index in landmark_indices],
        dtype=np.int32
    )

    hull = cv2.convexHull(selected_points)

    return hull


def get_all_region_polygons(landmarks, frame_shape):
    """
    Generate polygons for all defined facial regions.
    """

    points = landmarks_to_points(
        landmarks,
        frame_shape
    )

    regions = {}

    for region_name, indices in REGION_LANDMARKS.items():

        polygon = get_region_polygon(
            points,
            indices
        )

        regions[region_name] = polygon

    return regions


def draw_regions(frame, regions):
    """
    Draw facial region polygons and labels.
    """

    output = frame.copy()

    for region_name, polygon in regions.items():

        cv2.polylines(
            output,
            [polygon],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

        x, y, width, height = cv2.boundingRect(polygon)

        cv2.putText(
            output,
            region_name,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return output