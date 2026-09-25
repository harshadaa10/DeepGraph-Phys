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

from src.features.gsp_features import (
    temporal_graph_variation,
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
    / "motion_temporal_variation.png"
)


def main():

    print(
        "\nDeepGraph-Phys — Temporal Graph Variation Test"
    )

    print("-" * 55)

    # ------------------------------------------
    # Load synchronized graph signals
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

    motion = dynamic_data[
        "motion"
    ]

    time = dynamic_data[
        "time"
    ]

    # ------------------------------------------
    # Temporal variation
    # ------------------------------------------

    variation = temporal_graph_variation(
        motion
    )

    print(
        "Motion matrix shape:",
        motion.shape
    )

    print(
        "Temporal variation samples:",
        len(variation)
    )

    print(
        "\nTemporal variation statistics:"
    )

    print(
        f"Minimum : {variation.min():.8f}"
    )

    print(
        f"Maximum : {variation.max():.8f}"
    )

    print(
        f"Mean    : {variation.mean():.8f}"
    )

    print(
        f"Std     : {variation.std():.8f}"
    )

    # ------------------------------------------
    # Validation
    # ------------------------------------------

    print(
        "\nValidation checks:"
    )

    print(
        "All values finite:",
        np.isfinite(
            variation
        ).all()
    )

    print(
        "All values non-negative:",
        (
            variation >= 0
        ).all()
    )

    print(
        "Sample count correct:",
        len(variation)
        == len(motion)
    )

    print(
        "First value is zero:",
        np.isclose(
            variation[0],
            0.0,
        )
    )

    # ------------------------------------------
    # Largest temporal event
    # ------------------------------------------

    max_index = int(
        np.argmax(
            variation
        )
    )

    print(
        "\nLargest temporal variation:"
    )

    print(
        "Index:",
        max_index
    )

    print(
        f"Time: {time[max_index]:.4f} seconds"
    )

    print(
        f"Variation: {variation[max_index]:.8f}"
    )

    # ------------------------------------------
    # Plot
    # ------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time,
        variation,
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Temporal graph variation"
    )

    plt.title(
        "Temporal Variation of Facial Motion Graph"
    )

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
        "\nTemporal variation plot saved to:"
    )

    print(
        OUTPUT_PLOT
    )

    print(
        "\nTemporal graph variation test completed."
    )


if __name__ == "__main__":
    main() 