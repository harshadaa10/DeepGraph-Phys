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

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_real_motion_spectral.csv"
)

OUTPUT_PLOT = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "motion_spectral_ratios.png"
)


def main():

    print(
        "\nDeepGraph-Phys — Spectral Feature Test"
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

    frames = dynamic_data[
        "frames"
    ]

    time = dynamic_data[
        "time"
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
    # GFT
    # ------------------------------------------

    (
        eigenvalues,
        eigenvectors,
    ) = graph_fourier_basis(
        laplacian
    )

    coefficients = (
        graph_fourier_transform_series(
            motion,
            eigenvectors,
        )
    )

    # ------------------------------------------
    # Spectral features
    # ------------------------------------------

    features = spectral_energy_features(
        coefficients,
        eigenvalues,
        low_frequency_count=3,
    )

    print(
        "Motion samples:",
        len(motion)
    )

    print(
        "Graph frequencies:",
        len(eigenvalues)
    )

    print(
        "\nFeature statistics:"
    )

    for feature_name, values in (
        features.items()
    ):

        print(
            f"{feature_name:18s} "
            f"mean={values.mean():.8f} "
            f"std={values.std():.8f}"
        )

    # ------------------------------------------
    # Validation
    # ------------------------------------------

    print(
        "\nValidation checks:"
    )

    print(
        "All total energies finite:",
        np.isfinite(
            features["total_energy"]
        ).all()
    )

    print(
        "All low energies non-negative:",
        (
            features["low_energy"]
            >= 0
        ).all()
    )

    print(
        "All high energies non-negative:",
        (
            features["high_energy"]
            >= 0
        ).all()
    )

    print(
        "Low + high equals total:",
        np.allclose(
            features["low_energy"]
            + features["high_energy"],
            features["total_energy"],
            atol=1e-10,
        )
    )

    ratio_sum = (
        features["low_ratio"]
        + features["high_ratio"]
    )

    valid_energy = (
        features["total_energy"]
        > 1e-10
    )

    print(
        "Low + high ratio approximately 1:",
        np.allclose(
            ratio_sum[
                valid_energy
            ],
            1.0,
            atol=1e-6,
        )
    )

    print(
        "Spectral centroid finite:",
        np.isfinite(
            features[
                "spectral_centroid"
            ]
        ).all()
    )

    # ------------------------------------------
    # Save feature table
    # ------------------------------------------

    output_dataframe = pd.DataFrame(
        {
            "frame": frames,
            "time_seconds": time,
            "total_energy":
                features["total_energy"],
            "low_energy":
                features["low_energy"],
            "high_energy":
                features["high_energy"],
            "low_ratio":
                features["low_ratio"],
            "high_ratio":
                features["high_ratio"],
            "spectral_centroid":
                features["spectral_centroid"],
        }
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        "\nSaved spectral feature CSV:"
    )

    print(
        OUTPUT_CSV
    )

    # ------------------------------------------
    # Plot low/high energy ratios
    # ------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time,
        features["low_ratio"],
        label="Low-frequency ratio",
    )

    plt.plot(
        time,
        features["high_ratio"],
        label="High-frequency ratio",
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Spectral energy ratio"
    )

    plt.title(
        "Low vs High Graph-Frequency Energy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    OUTPUT_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_PLOT,
        dpi=150,
    )

    plt.close()

    print(
        "\nSaved spectral ratio plot:"
    )

    print(
        OUTPUT_PLOT
    )

    print(
        "\nSpectral feature test completed."
    )


if __name__ == "__main__":
    main()