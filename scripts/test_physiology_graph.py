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

from src.graph.physiology_graph import (
    PHYSIOLOGY_NODES,
    build_physiology_graph,
    get_physiology_adjacency,
    get_physiology_laplacian,
)

from src.features.physiology_features import (
    build_physiology_matrix,
    physiology_correlation_matrix,
    mean_pairwise_correlation,
    mean_absolute_pairwise_correlation,
)

from src.features.gsp_features import (
    graph_smoothness_series,
    graph_fourier_basis,
    graph_fourier_transform_series,
    spectral_energy_features,
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


def main():

    print(
        "\nDeepGraph-Phys — Physiology Graph Test"
    )

    print("-" * 55)

    # ------------------------------------------
    # Load synchronized data
    # ------------------------------------------

    rppg_dataframe = pd.read_csv(
        RPPG_PATH
    )

    motion_dataframe = pd.read_csv(
        MOTION_PATH
    )

    dynamic_data = create_dynamic_graph_data(
        rppg_dataframe,
        motion_dataframe,
    )

    synchronized = dynamic_data[
        "dataframe"
    ]

    # ------------------------------------------
    # Build physiology signal matrix
    # ------------------------------------------

    physiology = build_physiology_matrix(
        synchronized
    )

    print(
        "Physiology matrix shape:",
        physiology.shape
    )

    print(
        "\nPhysiology node order:"
    )

    print(
        PHYSIOLOGY_NODES
    )

    # ------------------------------------------
    # Build physiology graph
    # ------------------------------------------

    graph = build_physiology_graph()

    adjacency = get_physiology_adjacency(
        graph
    )

    laplacian = get_physiology_laplacian(
        adjacency
    )

    print(
        "\nNumber of nodes:",
        graph.number_of_nodes()
    )

    print(
        "Number of edges:",
        graph.number_of_edges()
    )

    print(
        "\nAdjacency matrix:"
    )

    print(
        adjacency.astype(int)
    )

    print(
        "\nLaplacian matrix:"
    )

    print(
        laplacian.astype(int)
    )

    # ------------------------------------------
    # Cross-region correlation
    # ------------------------------------------

    correlation = (
        physiology_correlation_matrix(
            physiology
        )
    )

    print(
        "\nCross-region rPPG correlation:"
    )

    print(
        np.round(
            correlation,
            3
        )
    )

    signed_correlation = (
        mean_pairwise_correlation(
            correlation
        )
    )

    absolute_correlation = (
        mean_absolute_pairwise_correlation(
            correlation
        )
    )

    print(
        "\nMean pairwise correlation:"
    )

    print(
        f"{signed_correlation:.6f}"
    )

    print(
        "Mean absolute pairwise correlation:"
    )

    print(
        f"{absolute_correlation:.6f}"
    )

    # ------------------------------------------
    # Physiological graph smoothness
    # ------------------------------------------

    smoothness = graph_smoothness_series(
        physiology,
        laplacian,
    )

    print(
        "\nPhysiology graph smoothness:"
    )

    print(
        f"Minimum : {smoothness.min():.8f}"
    )

    print(
        f"Maximum : {smoothness.max():.8f}"
    )

    print(
        f"Mean    : {smoothness.mean():.8f}"
    )

    print(
        f"Std     : {smoothness.std():.8f}"
    )

    # ------------------------------------------
    # Physiology GFT
    # ------------------------------------------

    (
        eigenvalues,
        eigenvectors,
    ) = graph_fourier_basis(
        laplacian
    )

    coefficients = (
        graph_fourier_transform_series(
            physiology,
            eigenvectors,
        )
    )

    spectral_features = (
        spectral_energy_features(
            coefficients,
            eigenvalues,
            low_frequency_count=1,
        )
    )

    print(
        "\nPhysiology graph frequencies:"
    )

    print(
        np.round(
            eigenvalues,
            6
        )
    )

    print(
        "\nMean physiology high-frequency ratio:"
    )

    print(
        f"{spectral_features['high_ratio'].mean():.6f}"
    )

    print(
        "\nMean physiology spectral centroid:"
    )

    print(
        f"{spectral_features['spectral_centroid'].mean():.6f}"
    )

    # ------------------------------------------
    # Validation
    # ------------------------------------------

    print(
        "\nValidation checks:"
    )

    print(
        "Physiology shape correct:",
        physiology.shape[1] == 3
    )

    print(
        "All physiology values finite:",
        np.isfinite(
            physiology
        ).all()
    )

    print(
        "Laplacian symmetric:",
        np.allclose(
            laplacian,
            laplacian.T,
        )
    )

    print(
        "Laplacian row sums zero:",
        np.allclose(
            laplacian.sum(axis=1),
            0.0,
        )
    )

    print(
        "Smoothness non-negative:",
        (
            smoothness >= -1e-12
        ).all()
    )

    print(
        "GFT values finite:",
        np.isfinite(
            coefficients
        ).all()
    )

    print(
        "Spectral features finite:",
        all(
            np.isfinite(values).all()
            for values
            in spectral_features.values()
        )
    )

    print(
        "\nPhysiology graph test completed."
    )


if __name__ == "__main__":
    main()