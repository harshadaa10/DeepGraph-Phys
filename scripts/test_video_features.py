from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.graph.dynamic_graph import (
    create_dynamic_graph_data,
)

from src.features.video_features import (
    extract_video_features,
)


RPPG_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_real_rppg.csv"
)

MOTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_real_motion.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_real_video_features.csv"
)


def main():

    print(
        "\nDeepGraph-Phys — Final Video Feature Test"
    )

    print("-" * 60)

    # ------------------------------------------
    # Load data
    # ------------------------------------------

    rppg_dataframe = pd.read_csv(
        RPPG_PATH
    )

    motion_dataframe = pd.read_csv(
        MOTION_PATH
    )

    # ------------------------------------------
    # Dynamic graph
    # ------------------------------------------

    dynamic_data = (
        create_dynamic_graph_data(
            rppg_dataframe,
            motion_dataframe,
        )
    )

    # ------------------------------------------
    # Extract final features
    # ------------------------------------------

    features = extract_video_features(
        dynamic_data
    )

    print(
        "Number of extracted features:",
        len(features)
    )

    print(
        "\nExtracted video-level features:"
    )

    for feature_name in sorted(
        features.keys()
    ):

        print(
            f"{feature_name:42s} "
            f"{features[feature_name]:.8f}"
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
        "Feature vector not empty:",
        len(features) > 0
    )

    print(
        "All feature values finite:",
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

    # ------------------------------------------
    # Save one-row feature CSV
    # ------------------------------------------

    output_dataframe = pd.DataFrame(
        [
            {
                "video_name":
                    "sample_real.mp4",
                "label":
                    "real",
                **features,
            }
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nFeature CSV saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nCSV shape:",
        output_dataframe.shape
    )

    print(
        "\nFinal video feature extraction "
        "completed successfully."
    )


if __name__ == "__main__":
    main()