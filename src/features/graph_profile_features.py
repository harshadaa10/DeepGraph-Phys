import numpy as np

from src.features.graph_spectral_profile import (
    extract_graph_spectral_profiles,
)


GRAPH_PROFILE_STATISTICS = [
    "mean",
    "std",
    "median",
    "q25",
    "q75",
]


EXPECTED_GRAPH_PROFILE_FEATURES = 45


def summarize_profile_column(
    values,
):
    """
    Summarize one graph spectral eigenspace
    across time.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "Profile values must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            "Profile contains no temporal samples."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Profile contains NaN or infinite values."
        )

    return {
        "mean":
            float(
                np.mean(values)
            ),

        "std":
            float(
                np.std(values)
            ),

        "median":
            float(
                np.median(values)
            ),

        "q25":
            float(
                np.percentile(
                    values,
                    25,
                )
            ),

        "q75":
            float(
                np.percentile(
                    values,
                    75,
                )
            ),
    }


def summarize_grouped_profile(
    grouped_profile,
    prefix,
):
    """
    Convert a T x G grouped spectral profile
    into fixed-size video-level features.

    Five temporal statistics are generated
    for every graph eigenspace.
    """

    grouped_profile = np.asarray(
        grouped_profile,
        dtype=np.float64,
    )

    if grouped_profile.ndim != 2:
        raise ValueError(
            "grouped_profile must have shape (T, G)."
        )

    if grouped_profile.shape[0] == 0:
        raise ValueError(
            "grouped_profile contains no samples."
        )

    if grouped_profile.shape[1] == 0:
        raise ValueError(
            "grouped_profile contains no "
            "eigenvalue groups."
        )

    if not np.isfinite(
        grouped_profile
    ).all():
        raise ValueError(
            "grouped_profile contains NaN or "
            "infinite values."
        )

    features = {}

    num_groups = (
        grouped_profile.shape[1]
    )

    for group_index in range(
        num_groups
    ):

        summary = (
            summarize_profile_column(
                grouped_profile[
                    :,
                    group_index
                ]
            )
        )

        for statistic in (
            GRAPH_PROFILE_STATISTICS
        ):

            feature_name = (
                f"{prefix}_g"
                f"{group_index}_"
                f"{statistic}"
            )

            features[
                feature_name
            ] = summary[
                statistic
            ]

    return features


def extract_graph_profile_features(
    dynamic_data,
):
    """
    Generate the fixed-size DeepGraph-Phys
    graph spectral consistency representation.

    Motion graph:
        7 eigenspaces x 5 statistics = 35

    Physiology graph:
        2 eigenspaces x 5 statistics = 10

    Total:
        45 graph-aware video-level features.
    """

    profiles = (
        extract_graph_spectral_profiles(
            dynamic_data
        )
    )

    motion_profile = (
        profiles[
            "motion"
        ][
            "grouped_profile"
        ]
    )

    physiology_profile = (
        profiles[
            "physiology"
        ][
            "grouped_profile"
        ]
    )

    # Current anatomical facial graph
    # should have seven distinct eigenspaces.
    if motion_profile.shape[1] != 7:
        raise ValueError(
            "Expected 7 motion eigenspaces, "
            f"received {motion_profile.shape[1]}."
        )

    # Current 3-node complete physiology graph
    # has eigenvalues [0, 3, 3], therefore
    # two distinct eigenspaces.
    if physiology_profile.shape[1] != 2:
        raise ValueError(
            "Expected 2 physiology eigenspaces, "
            f"received "
            f"{physiology_profile.shape[1]}."
        )

    features = {}

    motion_features = (
        summarize_grouped_profile(
            motion_profile,
            prefix="motion_gsp",
        )
    )

    physiology_features = (
        summarize_grouped_profile(
            physiology_profile,
            prefix="physiology_gsp",
        )
    )

    features.update(
        motion_features
    )

    features.update(
        physiology_features
    )

    if len(features) != (
        EXPECTED_GRAPH_PROFILE_FEATURES
    ):
        raise ValueError(
            "Unexpected graph-profile feature "
            f"count: {len(features)}. "
            "Expected 45."
        )

    feature_values = np.asarray(
        list(
            features.values()
        ),
        dtype=np.float64,
    )

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "Graph-profile feature vector "
            "contains NaN or infinite values."
        )

    return features