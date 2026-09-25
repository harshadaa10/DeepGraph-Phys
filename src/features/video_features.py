import numpy as np

from src.graph.facial_graph import (
    get_adjacency_matrix,
    get_laplacian_matrix,
)

from src.graph.physiology_graph import (
    build_physiology_graph,
    get_physiology_adjacency,
    get_physiology_laplacian,
)

from src.features.gsp_features import (
    graph_smoothness_series,
    graph_fourier_basis,
    graph_fourier_transform_series,
    spectral_energy_features,
    temporal_graph_variation,
)

from src.features.physiology_features import (
    build_physiology_matrix,
    physiology_correlation_matrix,
    mean_pairwise_correlation,
    mean_absolute_pairwise_correlation,
)


def summarize_signal(values):
    """
    Convert a temporal feature sequence into
    video-level summary statistics.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def add_summary(
    feature_dictionary,
    prefix,
    values,
):
    """
    Add summary statistics to a flat
    feature dictionary.
    """

    summary = summarize_signal(
        values
    )

    for statistic, value in summary.items():

        feature_dictionary[
            f"{prefix}_{statistic}"
        ] = value


def extract_video_features(
    dynamic_data,
):
    """
    Extract one complete video-level feature
    vector from synchronized motion and
    physiology graph signals.
    """

    graph = dynamic_data[
        "graph"
    ]

    motion = dynamic_data[
        "motion"
    ]

    synchronized = dynamic_data[
        "dataframe"
    ]

    features = {}

    # ==================================================
    # 1. MOTION GRAPH
    # ==================================================

    adjacency = get_adjacency_matrix(
        graph
    )

    motion_laplacian = (
        get_laplacian_matrix(
            adjacency
        )
    )

    # ------------------------------------------
    # Motion graph smoothness
    # ------------------------------------------

    motion_smoothness = (
        graph_smoothness_series(
            motion,
            motion_laplacian,
        )
    )

    add_summary(
        features,
        "motion_smoothness",
        motion_smoothness,
    )

    # ------------------------------------------
    # Motion temporal variation
    # ------------------------------------------

    motion_temporal = (
        temporal_graph_variation(
            motion
        )
    )

    add_summary(
        features,
        "motion_temporal",
        motion_temporal,
    )

    # ------------------------------------------
    # Motion graph Fourier features
    # ------------------------------------------

    (
        motion_eigenvalues,
        motion_eigenvectors,
    ) = graph_fourier_basis(
        motion_laplacian
    )

    motion_coefficients = (
        graph_fourier_transform_series(
            motion,
            motion_eigenvectors,
        )
    )

    motion_spectral = (
        spectral_energy_features(
            motion_coefficients,
            motion_eigenvalues,
            low_frequency_count=3,
        )
    )

    add_summary(
        features,
        "motion_total_energy",
        motion_spectral[
            "total_energy"
        ],
    )

    add_summary(
        features,
        "motion_low_ratio",
        motion_spectral[
            "low_ratio"
        ],
    )

    add_summary(
        features,
        "motion_high_ratio",
        motion_spectral[
            "high_ratio"
        ],
    )

    add_summary(
        features,
        "motion_spectral_centroid",
        motion_spectral[
            "spectral_centroid"
        ],
    )

    # ==================================================
    # 2. PHYSIOLOGY GRAPH
    # ==================================================

    physiology = (
        build_physiology_matrix(
            synchronized
        )
    )

    physiology_graph = (
        build_physiology_graph()
    )

    physiology_adjacency = (
        get_physiology_adjacency(
            physiology_graph
        )
    )

    physiology_laplacian = (
        get_physiology_laplacian(
            physiology_adjacency
        )
    )

    # ------------------------------------------
    # Physiology correlation
    # ------------------------------------------

    correlation = (
        physiology_correlation_matrix(
            physiology
        )
    )

    features[
        "physiology_mean_correlation"
    ] = mean_pairwise_correlation(
        correlation
    )

    features[
        "physiology_mean_abs_correlation"
    ] = (
        mean_absolute_pairwise_correlation(
            correlation
        )
    )

    features[
        "physiology_forehead_left_corr"
    ] = float(
        correlation[0, 1]
    )

    features[
        "physiology_forehead_right_corr"
    ] = float(
        correlation[0, 2]
    )

    features[
        "physiology_left_right_corr"
    ] = float(
        correlation[1, 2]
    )

    # ------------------------------------------
    # Physiology graph smoothness
    # ------------------------------------------

    physiology_smoothness = (
        graph_smoothness_series(
            physiology,
            physiology_laplacian,
        )
    )

    add_summary(
        features,
        "physiology_smoothness",
        physiology_smoothness,
    )

    # ------------------------------------------
    # Physiology temporal variation
    # ------------------------------------------

    physiology_temporal = (
        temporal_graph_variation(
            physiology
        )
    )

    add_summary(
        features,
        "physiology_temporal",
        physiology_temporal,
    )

    # ------------------------------------------
    # Physiology GFT
    # ------------------------------------------

    (
        physiology_eigenvalues,
        physiology_eigenvectors,
    ) = graph_fourier_basis(
        physiology_laplacian
    )

    physiology_coefficients = (
        graph_fourier_transform_series(
            physiology,
            physiology_eigenvectors,
        )
    )

    physiology_spectral = (
        spectral_energy_features(
            physiology_coefficients,
            physiology_eigenvalues,
            low_frequency_count=1,
        )
    )

    add_summary(
        features,
        "physiology_total_energy",
        physiology_spectral[
            "total_energy"
        ],
    )

    add_summary(
        features,
        "physiology_high_ratio",
        physiology_spectral[
            "high_ratio"
        ],
    )

    add_summary(
        features,
        "physiology_spectral_centroid",
        physiology_spectral[
            "spectral_centroid"
        ],
    )

    return features