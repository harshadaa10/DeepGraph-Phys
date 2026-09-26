from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


REAL_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "FaceForensics++"
    / "original_sequences"
    / "youtube"
    / "c23"
    / "videos"
)

FAKE_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "FaceForensics++"
    / "manipulated_sequences"
    / "Deepfakes"
    / "c23"
    / "videos"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ffpp_manifest.csv"
)


def main():

    print(
        "\nDeepGraph-Phys — FaceForensics++ Manifest"
    )

    print("-" * 60)

    records = []

    # ==========================================
    # REAL VIDEOS
    # ==========================================

    real_videos = sorted(
        REAL_FOLDER.glob("*.mp4")
    )

    for video_path in real_videos:

        source_id = (
            video_path.stem
        )

        records.append(
            {
                "video_name":
                    video_path.name,

                "video_path":
                    str(
                        video_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "label":
                    "real",

                "dataset":
                    "FaceForensics++",

                "manipulation":
                    "original",

                "source_id":
                    source_id,

                "target_id":
                    "",
            }
        )

    # ==========================================
    # FAKE VIDEOS
    # ==========================================

    fake_videos = sorted(
        FAKE_FOLDER.glob("*.mp4")
    )

    for video_path in fake_videos:

        parts = (
            video_path.stem.split("_")
        )

        if len(parts) != 2:

            print(
                "Skipping unexpected filename:",
                video_path.name,
            )

            continue

        target_id = parts[0]
        source_id = parts[1]

        records.append(
            {
                "video_name":
                    video_path.name,

                "video_path":
                    str(
                        video_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "label":
                    "fake",

                "dataset":
                    "FaceForensics++",

                "manipulation":
                    "Deepfakes",

                "source_id":
                    source_id,

                "target_id":
                    target_id,
            }
        )

    # ==========================================
    # DATAFRAME
    # ==========================================

    dataframe = pd.DataFrame(
        records
    )

    print(
        "Real videos:",
        len(real_videos)
    )

    print(
        "Fake videos:",
        len(fake_videos)
    )

    print(
        "Total manifest rows:",
        len(dataframe)
    )

    print(
        "\nClass counts:"
    )

    print(
        dataframe[
            "label"
        ].value_counts()
    )

    # ==========================================
    # VALIDATION
    # ==========================================

    real_ids = set(
        dataframe.loc[
            dataframe["label"] == "real",
            "source_id",
        ]
    )

    fake_source_ids = set(
        dataframe.loc[
            dataframe["label"] == "fake",
            "source_id",
        ]
    )

    fake_target_ids = set(
        dataframe.loc[
            dataframe["label"] == "fake",
            "target_id",
        ]
    )

    fake_all_ids = (
        fake_source_ids
        | fake_target_ids
    )

    print(
        "\nValidation checks:"
    )

    print(
        "Exactly 20 real videos:",
        len(real_videos) == 20
    )

    print(
        "Exactly 20 fake videos:",
        len(fake_videos) == 20
    )

    print(
        "Exactly 40 manifest rows:",
        len(dataframe) == 40
    )

    print(
        "All fake filenames parsed:",
        len(
            dataframe[
                dataframe["label"] == "fake"
            ]
        ) == len(fake_videos)
    )

    print(
        "All fake IDs represented "
        "in real subset:",
        fake_all_ids.issubset(
            real_ids
        )
    )

    # ==========================================
    # SAVE
    # ==========================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nManifest saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nFirst 10 rows:"
    )

    print(
        dataframe.head(10).to_string(
            index=False
        )
    )

    print(
        "\nFaceForensics++ manifest "
        "created successfully."
    )


if __name__ == "__main__":
    main()