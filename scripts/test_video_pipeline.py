from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.features.video_pipeline import (
    process_video_to_features,
)


VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "real"
    / "sample_real.mp4"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)


def main():

    print(
        "\nDeepGraph-Phys — End-to-End Video Pipeline Test"
    )

    print("-" * 60)

    print(
        "Video:"
    )

    print(
        VIDEO_PATH
    )

    # ------------------------------------------
    # Run complete pipeline
    # ------------------------------------------

    features, metadata = (
        process_video_to_features(
            VIDEO_PATH,
            MODEL_PATH,
        )
    )

    # ------------------------------------------
    # Metadata
    # ------------------------------------------

    print(
        "\nPipeline metadata:"
    )

    for key, value in (
        metadata.items()
    ):

        print(
            f"{key:20s}: {value}"
        )

    # ------------------------------------------
    # Validation
    # ------------------------------------------

    feature_values = np.array(
        list(
            features.values()
        ),
        dtype=np.float64,
    )

    print(
        "\nValidation checks:"
    )

    print(
        "Exactly 60 features:",
        len(features) == 60
    )

    print(
        "All values finite:",
        np.isfinite(
            feature_values
        ).all()
    )

    print(
        "No NaN values:",
        not np.isnan(
            feature_values
        ).any()
    )

    print(
        "No infinite values:",
        not np.isinf(
            feature_values
        ).any()
    )

    print(
        "Graph samples available:",
        metadata[
            "graph_samples"
        ] > 0
    )

    # ------------------------------------------
    # Preview
    # ------------------------------------------

    print(
        "\nFirst 10 features:"
    )

    for index, (
        feature_name,
        value,
    ) in enumerate(
        features.items()
    ):

        if index >= 10:
            break

        print(
            f"{feature_name:42s} "
            f"{value:.8f}"
        )

    print(
        "\nEnd-to-end video pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()