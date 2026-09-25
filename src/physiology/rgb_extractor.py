import cv2
import numpy as np


RPPG_REGIONS = [
    "forehead",
    "left_cheek",
    "right_cheek",
]


def extract_mean_rgb(frame, polygon):
    """
    Calculate the mean RGB value inside a facial ROI polygon.

    Parameters
    ----------
    frame : numpy.ndarray
        OpenCV BGR image.

    polygon : numpy.ndarray
        Polygon describing the facial ROI.

    Returns
    -------
    numpy.ndarray
        Mean RGB values in the order [R, G, B].
    """

    mask = np.zeros(
        frame.shape[:2],
        dtype=np.uint8
    )

    cv2.fillConvexPoly(
        mask,
        polygon,
        255
    )

    mean_bgr = cv2.mean(
        frame,
        mask=mask
    )[:3]

    mean_rgb = np.array(
        [
            mean_bgr[2],
            mean_bgr[1],
            mean_bgr[0]
        ],
        dtype=np.float64
    )

    return mean_rgb


def extract_region_rgb(frame, regions):
    """
    Extract mean RGB values from all rPPG facial regions.
    """

    rgb_values = {}

    for region_name in RPPG_REGIONS:

        if region_name not in regions:
            continue

        rgb_values[region_name] = extract_mean_rgb(
            frame,
            regions[region_name]
        )

    return rgb_values