from pathlib import Path
import sys

import matplotlib.pyplot as plt
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

from src.graph.facial_graph import (
    get_adjacency_matrix,
    get_laplacian_matrix,
)

from src.features.gsp_features import (
    graph_fourier_basis,
    graph_fourier_transform_series,
    graph_spectral_energy,
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

OUTPUT_PLOT = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "motion_graph_spectrum.png"
)


def main():

    print(
        "\nDeepGraph-Phys — Graph Fourier Transform Test"
    )

    print("-" * 55)

    # ------------------------------------------
    # Load data
    # ------------------------------------------

    rppg_dataframe = pd.read_csv(
        RPPG_PATH
    )

    motion_dataframe = pd.read_csv(
        MOTION_PATH
    )

    dynamic_data = (
        create_dynamic_graph_data(
            rppg_dataframe,
            motion_dataframe,
        )
    )

    graph = dynamic_data[
        "graph"
    ]

    motion = dynamic_data[
        "motion"
    ]

    # ------------------------------------------
    # Graph Laplacian
    # ------------------------------------------

    adjacency = get_adjacency_matrix(
        graph
    )

    laplacian = get_laplacian_matrix(
        adjacency
    )

    # ------------------------------------------
    # Graph Fourier basis
    # ------------------------------------------

    (
        eigenvalues,
        eigenvectors,
    ) = graph_fourier_basis(
        laplacian
    )

    print(
        "Motion matrix shape:",
        motion.shape
    )

    print(
        "Laplacian shape:",
        laplacian.shape
    )

    print(
        "\nGraph frequencies "
        "(Laplacian eigenvalues):"
    )

    print(
        np.round(
            eigenvalues,
            6
        )
    )

    # ------------------------------------------
    # Apply GFT
    # ------------------------------------------

    coefficients = (
        graph_fourier_transform_series(
            motion,
            eigenvectors,
        )
    )

    energy = graph_spectral_energy(
        coefficients
    )

    print(
        "\nGFT coefficient matrix shape:",
        coefficients.shape
    )

    print(
        "Spectral energy matrix shape:",
        energy.shape
    )

    # ------------------------------------------
    # Mean energy per graph frequency
    # ------------------------------------------

    mean_energy = np.mean(
        energy,
        axis=0
    )

    print(
        "\nMean energy per graph frequency:"
    )

    for index, (
        frequency,
        frequency_energy,
    ) in enumerate(
        zip(
            eigenvalues,
            mean_energy,
        )
    ):

        print(
            f"Frequency {index}: "
            f"lambda={frequency:.6f} "
            f"energy={frequency_energy:.8f}"
        )

    # ------------------------------------------
    # Validation
    # ------------------------------------------

    print(
        "\nValidation checks:"
    )

    print(
        "Eigenvalues sorted:",
        np.all(
            np.diff(
                eigenvalues
            )
            >= -1e-12
        )
    )

    print(
        "First eigenvalue approximately zero:",
        np.isclose(
            eigenvalues[0],
            0.0,
            atol=1e-10,
        )
    )

    print(
        "All eigenvalues non-negative:",
        np.all(
            eigenvalues
            >= -1e-10
        )
    )

    identity = np.eye(
        eigenvectors.shape[1]
    )

    print(
        "Eigenvectors orthonormal:",
        np.allclose(
            eigenvectors.T
            @ eigenvectors,
            identity,
            atol=1e-10,
        )
    )

    print(
        "All GFT values finite:",
        np.isfinite(
            coefficients
        ).all()
    )

    print(
        "All spectral energies non-negative:",
        (
            energy >= 0
        ).all()
    )

    # ------------------------------------------
    # Parseval energy check
    # ------------------------------------------

    original_energy = np.sum(
        motion ** 2,
        axis=1
    )

    transformed_energy = np.sum(
        coefficients ** 2,
        axis=1
    )

    print(
        "GFT preserves signal energy:",
        np.allclose(
            original_energy,
            transformed_energy,
            atol=1e-10,
        )
    )

    # ------------------------------------------
    # Plot mean graph spectrum
    # ------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.stem(
        eigenvalues,
        mean_energy
    )

    plt.xlabel(
        "Graph frequency (Laplacian eigenvalue)"
    )

    plt.ylabel(
        "Mean spectral energy"
    )

    plt.title(
        "Mean Graph-Frequency Spectrum of Facial Motion"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    OUTPUT_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PLOT,
        dpi=150
    )

    plt.close()

    print(
        "\nGraph spectrum saved to:"
    )

    print(
        OUTPUT_PLOT
    )

    print(
        "\nGraph Fourier Transform test completed."
    )


if __name__ == "__main__":
    main()