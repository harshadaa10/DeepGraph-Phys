from pathlib import Path

import cv2
import pandas as pd

from src.preprocessing.robust_face_landmarker import (
    RobustFaceLandmarkDetector,
)

from src.preprocessing.face_regions import (
    get_all_region_polygons,
)

from src.physiology.rgb_extractor import (
    RPPG_REGIONS,
    extract_region_rgb,
)


def extract_video_rgb_signals(
    video_path,
    model_path,
    max_frames=None,
):
    """
    Extract temporal RGB signals from selected facial
    regions across a video.

    Returns
    -------
    dataframe : pandas.DataFrame
        RGB values for face-detected frames.

    metadata : dict
        Video and face-detection quality information.
    """

    video_path = Path(
        video_path
    )

    # ==========================================
    # 1. Open video
    # ==========================================

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if fps <= 0:
        cap.release()

        raise ValueError(
            f"Invalid FPS for video: {video_path}"
        )

    # ==========================================
    # 2. Create robust face detector
    # ==========================================

    detector = (
        RobustFaceLandmarkDetector(
            model_path
        )
    )

    records = []

    frame_index = 0
    detected_frames = 0

    # ==========================================
    # 3. Process video frames
    # ==========================================

    try:

        while True:

            # Stop before reading an extra frame
            # when a frame limit is requested.
            if (
                max_frames is not None
                and frame_index >= max_frames
            ):
                break

            success, frame = (
                cap.read()
            )

            if not success:
                break

            landmarks = (
                detector.detect(
                    frame
                )
            )

            if landmarks is not None:

                regions = (
                    get_all_region_polygons(
                        landmarks,
                        frame.shape,
                    )
                )

                rgb_values = (
                    extract_region_rgb(
                        frame,
                        regions,
                    )
                )

                record = {
                    "frame":
                        frame_index,

                    "time_seconds":
                        frame_index / fps,
                }

                for region_name in (
                    RPPG_REGIONS
                ):

                    if (
                        region_name
                        in rgb_values
                    ):

                        r, g, b = (
                            rgb_values[
                                region_name
                            ]
                        )

                        record[
                            f"{region_name}_R"
                        ] = r

                        record[
                            f"{region_name}_G"
                        ] = g

                        record[
                            f"{region_name}_B"
                        ] = b

                records.append(
                    record
                )

                detected_frames += 1

            frame_index += 1

    finally:

        cap.release()
        detector.close()

    # ==========================================
    # 4. Build dataframe
    # ==========================================

    dataframe = pd.DataFrame(
        records
    )

    # ==========================================
    # 5. Quality-control metadata
    # ==========================================

    if frame_index > 0:

        face_detection_rate = (
            detected_frames
            / frame_index
        )

    else:

        face_detection_rate = 0.0

    metadata = {
        "fps":
            fps,

        "frames_processed":
            frame_index,

        "frames_with_face":
            detected_frames,

        "face_detection_rate":
            face_detection_rate,
    }

    return (
        dataframe,
        metadata,
    )