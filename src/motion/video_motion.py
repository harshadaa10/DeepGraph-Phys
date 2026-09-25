from pathlib import Path

import cv2
import pandas as pd

from src.preprocessing.face_landmarker import (
    FaceLandmarkDetector,
)

from src.motion.landmark_motion import (
    MOTION_REGIONS,
    calculate_all_region_motion,
)


def extract_video_motion(
    video_path,
    model_path,
    max_frames=None,
):
    """
    Extract region-wise facial landmark motion
    across a video.
    """

    video_path = Path(
        video_path
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    detector = FaceLandmarkDetector(
        model_path
    )

    records = []

    previous_landmarks = None

    frame_index = 0
    detected_frames = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if (
            max_frames is not None
            and frame_index >= max_frames
        ):
            break

        landmarks = detector.detect(
            frame
        )

        if landmarks is not None:

            detected_frames += 1

            if previous_landmarks is not None:

                motion = (
                    calculate_all_region_motion(
                        previous_landmarks,
                        landmarks,
                    )
                )

                record = {
                    "frame": frame_index,
                    "time_seconds": (
                        frame_index / fps
                        if fps > 0
                        else 0
                    ),
                }

                for region in MOTION_REGIONS:

                    record[
                        f"{region}_motion"
                    ] = motion[region]

                records.append(
                    record
                )

            previous_landmarks = landmarks

        else:
            # Do not calculate displacement across
            # a gap where the face was not detected.
            previous_landmarks = None

        frame_index += 1

    cap.release()

    detector.close()

    dataframe = pd.DataFrame(
        records
    )

    metadata = {
        "fps": fps,
        "frames_processed": frame_index,
        "frames_with_face": detected_frames,
    }

    return dataframe, metadata