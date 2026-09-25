from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Project setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.physiology.rppg_pipeline import (
    generate_rppg_dataframe,
)


# --------------------------------------------------
# File paths
# --------------------------------------------------

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_real_rgb.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_real_rppg.csv"
)

OUTPUT_PLOT = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "real_pos_rppg.png"
)


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def estimate_fps(dataframe):
    """
    Estimate FPS from the time_seconds column.
    """

    time = dataframe[
        "time_seconds"
    ].values

    if len(time) < 2:
        raise ValueError(
            "At least two time samples are required "
            "to estimate FPS."
        )

    intervals = (
        time[1:]
        - time[:-1]
    )

    mean_interval = intervals.mean()

    if mean_interval <= 0:
        raise ValueError(
            "Invalid time values."
        )

    return 1.0 / mean_interval


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print(
        "\nDeepGraph-Phys — POS rPPG Test"
    )

    print("-" * 50)

    # ----------------------------------------------
    # Load RGB signal data
    # ----------------------------------------------

    dataframe = pd.read_csv(
        INPUT_PATH
    )

    print(
        "RGB data loaded:",
        dataframe.shape
    )

    # ----------------------------------------------
    # Estimate FPS
    # ----------------------------------------------

    fps = estimate_fps(
        dataframe
    )

    print(
        f"Estimated FPS: {fps:.2f}"
    )

    # ----------------------------------------------
    # Generate POS rPPG signals
    # ----------------------------------------------

    print(
        "Generating POS rPPG signals..."
    )

    rppg_dataframe = (
        generate_rppg_dataframe(
            dataframe,
            fps
        )
    )

    print(
        "rPPG table shape:",
        rppg_dataframe.shape
    )

    # ----------------------------------------------
    # Display columns
    # ----------------------------------------------

    print("\nColumns:")

    for column in rppg_dataframe.columns:
        print(
            " -",
            column
        )

    # ----------------------------------------------
    # Signal statistics
    # ----------------------------------------------

    regions = [
        "forehead",
        "left_cheek",
        "right_cheek",
    ]

    print(
        "\nSignal statistics:"
    )

    for region in regions:

        signal = rppg_dataframe[
            f"{region}_rppg"
        ]

        print(
            f"{region:12s} "
            f"mean={signal.mean():.4f} "
            f"std={signal.std():.4f}"
        )

    # ----------------------------------------------
    # Cross-region correlation
    # ----------------------------------------------

    print(
        "\nCross-region rPPG correlation:"
    )

    signals = [
        rppg_dataframe[
            f"{region}_rppg"
        ].values
        for region in regions
    ]

    correlation_matrix = np.corrcoef(
        signals
    )

    correlation_dataframe = pd.DataFrame(
        correlation_matrix,
        index=regions,
        columns=regions
    )

    print(
        correlation_dataframe.round(3)
    )

    # ----------------------------------------------
    # Save rPPG CSV
    # ----------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rppg_dataframe.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        "\nSaved rPPG CSV:"
    )

    print(
        OUTPUT_CSV
    )

    # ----------------------------------------------
    # Plot regional rPPG signals
    # ----------------------------------------------

    time = rppg_dataframe[
        "time_seconds"
    ]

    plt.figure(
        figsize=(12, 6)
    )

    for region in regions:

        plt.plot(
            time,
            rppg_dataframe[
                f"{region}_rppg"
            ],
            label=region
        )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Normalized rPPG amplitude"
    )

    plt.title(
        "POS rPPG Signals Across Facial Regions"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    # ----------------------------------------------
    # Save plot
    # ----------------------------------------------

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
        "\nSaved rPPG plot:"
    )

    print(
        OUTPUT_PLOT
    )

    print(
        "\nPOS rPPG extraction completed successfully."
    )


if __name__ == "__main__":
    main()