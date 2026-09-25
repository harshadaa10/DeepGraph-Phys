from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.features.dataset_builder import (
    build_feature_dataset,
)


REAL_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "real"
)

FAKE_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fake"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "deepgraph_features.csv"
)

FAILURE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "feature_extraction_failures.csv"
)


def main():

    print(
        "\nDeepGraph-Phys — Feature Dataset Builder"
    )

    print("-" * 60)

    dataframe, failures = (
        build_feature_dataset(
            REAL_FOLDER,
            FAKE_FOLDER,
            MODEL_PATH,
        )
    )

    # ------------------------------------------
    # Dataset summary
    # ------------------------------------------

    print(
        "\nDataset summary:"
    )

    print(
        "Successful videos:",
        len(dataframe)
    )

    print(
        "Failed videos:",
        len(failures)
    )

    if len(dataframe) == 0:

        print(
            "\nNo valid videos were processed."
        )

        return

    print(
        "\nClass counts:"
    )

    print(
        dataframe[
            "label"
        ].value_counts()
    )

    print(
        "\nDataset shape:",
        dataframe.shape
    )

    # ------------------------------------------
    # Feature validation
    # ------------------------------------------

    metadata_columns = [
        "video_name",
        "label",
        "fps",
        "graph_samples",
    ]

    feature_columns = [
        column
        for column
        in dataframe.columns
        if column
        not in metadata_columns
    ]

    feature_values = (
        dataframe[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    print(
        "\nNumber of ML features:",
        len(
            feature_columns
        )
    )

    print(
        "\nValidation checks:"
    )

    print(
        "Exactly 60 ML features:",
        len(
            feature_columns
        ) == 60
    )

    print(
        "All feature values finite:",
        np.isfinite(
            feature_values
        ).all()
    )

    print(
        "No NaN feature values:",
        not np.isnan(
            feature_values
        ).any()
    )

    # ------------------------------------------
    # Save dataset
    # ------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nFeature dataset saved to:"
    )

    print(
        OUTPUT_PATH
    )

    # ------------------------------------------
    # Save failures if any
    # ------------------------------------------

    if len(failures) > 0:

        FAILURE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        failures.to_csv(
            FAILURE_PATH,
            index=False,
        )

        print(
            "\nFailure report saved to:"
        )

        print(
            FAILURE_PATH
        )

    print(
        "\nFeature dataset generation completed."
    )


if __name__ == "__main__":
    main()