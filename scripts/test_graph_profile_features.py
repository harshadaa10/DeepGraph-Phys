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


from src.graph.facial_graph import (
    build_facial_graph,
)

from src.features.graph_profile_features import (
    GRAPH_PROFILE_STATISTICS,
    EXPECTED_GRAPH_PROFILE_FEATURES,
    extract_graph_profile_features,
)


def main():

    print(
        "\nDeepGraph-Phys — "
        "45-D Graph Profile Feature Test"
    )

    print("=" * 70)

    frames = np.array(
        [10, 11, 12, 15, 16],
        dtype=np.int64,
    )

    motion = np.array(
        [
            [
                0.10, 0.12, 0.11,
                0.08, 0.09, 0.10,
                0.11,
            ],
            [
                0.11, 0.13, 0.12,
                0.09, 0.10, 0.11,
                0.12,
            ],
            [
                0.12, 0.14, 0.13,
                0.10, 0.11, 0.12,
                0.13,
            ],
            [
                0.20, 0.08, 0.16,
                0.07, 0.18, 0.09,
                0.15,
            ],
            [
                0.21, 0.09, 0.17,
                0.08, 0.19, 0.10,
                0.16,
            ],
        ],
        dtype=np.float64,
    )

    synchronized = pd.DataFrame(
        {
            "frame":
                frames,

            "forehead_rppg":
                [
                    0.10,
                    0.20,
                    0.30,
                    -0.10,
                    -0.20,
                ],

            "left_cheek_rppg":
                [
                    0.11,
                    0.19,
                    0.31,
                    -0.08,
                    -0.21,
                ],

            "right_cheek_rppg":
                [
                    0.09,
                    0.21,
                    0.29,
                    -0.12,
                    -0.19,
                ],
        }
    )

    dynamic_data = {
        "graph":
            build_facial_graph(),

        "dataframe":
            synchronized,

        "motion":
            motion,

        "frames":
            frames,
    }

    features = (
        extract_graph_profile_features(
            dynamic_data
        )
    )

    feature_names = list(
        features.keys()
    )

    feature_values = np.asarray(
        list(
            features.values()
        ),
        dtype=np.float64,
    )

    motion_features = [
        name
        for name
        in feature_names
        if name.startswith(
            "motion_gsp_"
        )
    ]

    physiology_features = [
        name
        for name
        in feature_names
        if name.startswith(
            "physiology_gsp_"
        )
    ]

    print(
        "\nFeature count:",
        len(features),
    )

    print(
        "Motion features:",
        len(
            motion_features
        ),
    )

    print(
        "Physiology features:",
        len(
            physiology_features
        ),
    )

    print(
        "\nStatistics per eigenspace:",
        GRAPH_PROFILE_STATISTICS,
    )

    print(
        "\nFirst 10 features:"
    )

    for name in (
        feature_names[:10]
    ):
        print(
            f"{name}: "
            f"{features[name]:.8f}"
        )

    print(
        "\nLast 10 features:"
    )

    for name in (
        feature_names[-10:]
    ):
        print(
            f"{name}: "
            f"{features[name]:.8f}"
        )

    # ==========================================
    # Expected feature names
    # ==========================================

    expected_names = []

    for group_index in range(
        7
    ):
        for statistic in (
            GRAPH_PROFILE_STATISTICS
        ):
            expected_names.append(
                f"motion_gsp_g"
                f"{group_index}_"
                f"{statistic}"
            )

    for group_index in range(
        2
    ):
        for statistic in (
            GRAPH_PROFILE_STATISTICS
        ):
            expected_names.append(
                f"physiology_gsp_g"
                f"{group_index}_"
                f"{statistic}"
            )

    # ==========================================
    # Validation
    # ==========================================

    print(
        "\nValidation checks:"
    )

    print(
        "Exactly 45 features:",
        len(features)
        == EXPECTED_GRAPH_PROFILE_FEATURES,
    )

    print(
        "Exactly 35 motion features:",
        len(
            motion_features
        ) == 35,
    )

    print(
        "Exactly 10 physiology features:",
        len(
            physiology_features
        ) == 10,
    )

    print(
        "All features finite:",
        np.isfinite(
            feature_values
        ).all(),
    )

    print(
        "Feature names exactly expected:",
        feature_names
        == expected_names,
    )

    print(
        "No duplicate feature names:",
        len(
            set(
                feature_names
            )
        )
        == len(
            feature_names
        ),
    )

    print(
        "All profile statistics bounded [0, 1]:",
        np.all(
            feature_values
            >= -1e-12
        )
        and np.all(
            feature_values
            <= 1.0 + 1e-12
        ),
    )


if __name__ == "__main__":
    main()