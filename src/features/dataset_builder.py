from pathlib import Path

import numpy as np
import pandas as pd

from src.features.video_pipeline import (
    process_video_to_features,
)


SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}


def find_videos(
    folder_path,
):
    """
    Find supported video files inside a folder.
    """

    folder_path = Path(
        folder_path
    )

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {folder_path}"
        )

    videos = []

    for path in folder_path.iterdir():

        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            videos.append(
                path
            )

    return sorted(
        videos
    )


def process_video_folder(
    folder_path,
    label,
    model_path,
    max_frames=None,
):
    """
    Process every video inside one class folder.

    Returns
    -------
    records : list
        Successfully extracted feature rows.

    failures : list
        Videos that could not be processed.
    """

    videos = find_videos(
        folder_path
    )

    records = []
    failures = []

    print(
        f"\nFound {len(videos)} "
        f"{label} video(s)."
    )

    for index, video_path in enumerate(
        videos,
        start=1,
    ):

        print(
            f"\n[{index}/{len(videos)}] "
            f"Processing {label}: "
            f"{video_path.name}"
        )

        try:

            features, metadata = (
                process_video_to_features(
                    video_path,
                    model_path,
                    max_frames=max_frames,
                )
            )

            feature_values = np.array(
                list(
                    features.values()
                ),
                dtype=np.float64,
            )

            if not np.isfinite(
                feature_values
            ).all():

                raise ValueError(
                    "Extracted feature vector "
                    "contains invalid values."
                )

            record = {
                "video_name":
                    video_path.name,

                "label":
                    label,

                "fps":
                    metadata["fps"],

                "graph_samples":
                    metadata[
                        "graph_samples"
                    ],

                **features,
            }

            records.append(
                record
            )

            print(
                "Success — "
                f"{len(features)} features"
            )

        except Exception as error:

            failures.append(
                {
                    "video_name":
                        video_path.name,

                    "label":
                        label,

                    "error":
                        str(error),
                }
            )

            print(
                "FAILED:"
            )

            print(
                error
            )

    return (
        records,
        failures,
    )


def build_feature_dataset(
    real_folder,
    fake_folder,
    model_path,
    max_frames=None,
):
    """
    Build one feature dataset from real and
    fake video folders.
    """

    real_records, real_failures = (
        process_video_folder(
            real_folder,
            "real",
            model_path,
            max_frames=max_frames,
        )
    )

    fake_records, fake_failures = (
        process_video_folder(
            fake_folder,
            "fake",
            model_path,
            max_frames=max_frames,
        )
    )

    records = (
        real_records
        + fake_records
    )

    failures = (
        real_failures
        + fake_failures
    )

    dataframe = pd.DataFrame(
        records
    )

    failures_dataframe = pd.DataFrame(
        failures
    )

    return (
        dataframe,
        failures_dataframe,
    )