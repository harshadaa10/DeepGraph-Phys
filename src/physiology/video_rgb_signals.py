from pathlib import Path

import cv2
import pandas as pd

from src.preprocessing.face_landmarker import FaceLandmarkDetector
from src.preprocessing.face_regions import get_all_region_polygons

from src.physiology.rgb_extractor import (
    RPPG_REGIONS,
    extract_region_rgb,
)


def extract_video_rgb_signals(
    video_path,
    model_path,
    max_frames=None
):
    """
    Extract temporal RGB signals from selected facial regions
    across an entire video.
    """

    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    detector = FaceLandmarkDetector(
        model_path
    )

    records = []

    frame_index = 0
    detected_frames = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if max_frames is not None:
            if frame_index >= max_frames:
                break

        landmarks = detector.detect(frame)

        if landmarks is not None:

            regions = get_all_region_polygons(
                landmarks,
                frame.shape
            )

            rgb_values = extract_region_rgb(
                frame,
                regions
            )

            record = {
                "frame": frame_index,
                "time_seconds": (
                    frame_index / fps
                    if fps > 0
                    else 0
                ),
            }

            for region_name in RPPG_REGIONS:

                if region_name in rgb_values:

                    r, g, b = rgb_values[
                        region_name
                    ]

                    record[
                        f"{region_name}_R"
                    ] = r

                    record[
                        f"{region_name}_G"
                    ] = g

                    record[
                        f"{region_name}_B"
                    ] = b

            records.append(record)

            detected_frames += 1

        frame_index += 1

    cap.release()
    detector.close()

    dataframe = pd.DataFrame(records)

    metadata = {
        "fps": fps,
        "total_frames_processed": frame_index,
        "frames_with_face": detected_frames,
    }

    return dataframe, metadata