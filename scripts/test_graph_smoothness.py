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
    graph_smoothness_series,
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
    / "motion_graph_smoothness.png"
)


def main():

    print(
        "\nDeepGraph-Phys — Graph Smoothness Test"
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

    # ------------------------------------------
    # Build dynamic graph representation
    # ------------------------------------------

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

    time = dynamic_data[
        "time"
    ]

    # ------------------------------------------
    # Build Laplacian
    # ------------------------------------------

    adjacency = get_adjacency_matrix(
        graph
    )

    laplacian = get_laplacian_matrix(
        adjacency
    )

    # ------------------------------------------
    # Calculate graph smoothness
    # ------------------------------------------

    smoothness = (
        graph_smoothness_series(
            motion,
            laplacian,
        )
    )

    # ------------------------------------------
    # Statistics
    # ------------------------------------------

    print(
        "Motion matrix shape:",
        motion.shape
    )

    print(
        "Laplacian shape:",
        laplacian.shape
    )

    print(
        "Smoothness samples:",
        len(smoothness)
    )

    print(
        "\nGraph smoothness statistics:"
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
    # Mathematical checks
    # ------------------------------------------

    print(
        "\nValidation checks:"
    )

    print(
        "All values finite:",
        np.isfinite(
            smoothness
        ).all()
    )

    print(
        "All energies non-negative:",
        (
            smoothness >= -1e-12
        ).all()
    )

    print(
        "Sample count correct:",
        len(smoothness)
        == len(motion)
    )

    # ------------------------------------------
    # Find largest smoothness event
    # ------------------------------------------

    max_index = int(
        np.argmax(
            smoothness
        )
    )

    print(
        "\nLargest smoothness event:"
    )

    print(
        "Index:",
        max_index
    )

    print(
        f"Time: {time[max_index]:.4f} seconds"
    )

    print(
        f"Energy: {smoothness[max_index]:.8f}"
    )

    # ------------------------------------------
    # Visualization
    # ------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time,
        smoothness
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Graph smoothness energy"
    )

    plt.title(
        "Facial Motion Graph Smoothness"
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
        "\nSmoothness plot saved to:"
    )

    print(
        OUTPUT_PLOT
    )

    print(
        "\nGraph smoothness test completed."
    )


if __name__ == "__main__":
    main()