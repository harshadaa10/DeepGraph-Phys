from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.physiology.signal_processing import (
    preprocess_signal,
)


CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_real_rgb.csv"
)

OUTPUT_RAW = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "raw_rgb_signals.png"
)

OUTPUT_FILTERED = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "filtered_green_signals.png"
)


# --------------------------------------------------
# FPS estimation
# --------------------------------------------------

def estimate_fps(dataframe):
    """
    Estimate video FPS using the time_seconds column.
    """

    time_values = dataframe[
        "time_seconds"
    ].values

    if len(time_values) < 2:
        raise ValueError(
            "At least two time samples are required "
            "to estimate FPS."
        )

    intervals = (
        time_values[1:]
        - time_values[:-1]
    )

    mean_interval = intervals.mean()

    if mean_interval <= 0:
        raise ValueError(
            "Invalid time values."
        )

    fps = 1.0 / mean_interval

    return fps


# --------------------------------------------------
# Raw RGB visualization
# --------------------------------------------------

def plot_raw_rgb(dataframe):
    """
    Plot raw forehead RGB signals.
    """

    time = dataframe[
        "time_seconds"
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        time,
        dataframe["forehead_R"],
        label="Forehead R"
    )

    plt.plot(
        time,
        dataframe["forehead_G"],
        label="Forehead G"
    )

    plt.plot(
        time,
        dataframe["forehead_B"],
        label="Forehead B"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Mean pixel intensity"
    )

    plt.title(
        "Raw Forehead RGB Signals"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_RAW,
        dpi=150
    )

    plt.close()


# --------------------------------------------------
# Preprocessed regional signal visualization
# --------------------------------------------------

def plot_filtered_signals(
    dataframe,
    fps
):
    """
    Preprocess and plot the green-channel signals
    from the main rPPG facial regions.
    """

    time = dataframe[
        "time_seconds"
    ]

    regions = [
        "forehead",
        "left_cheek",
        "right_cheek",
    ]

    plt.figure(
        figsize=(12, 6)
    )

    for region in regions:

        green_signal = dataframe[
            f"{region}_G"
        ].values

        processed = preprocess_signal(
            green_signal,
            fps
        )

        plt.plot(
            time,
            processed,
            label=region
        )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Normalized signal"
    )

    plt.title(
        "Preprocessed Regional Green Signals"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILTERED,
        dpi=150
    )

    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print(
        "\nDeepGraph-Phys — Signal Visualization"
    )

    print("-" * 50)

    # Load extracted RGB signals
    dataframe = pd.read_csv(
        CSV_PATH
    )

    print(
        "Loaded signal table:",
        dataframe.shape
    )

    # Automatically estimate FPS
    fps = estimate_fps(
        dataframe
    )

    print(
        f"Estimated FPS: {fps:.2f}"
    )

    # Make sure output directory exists
    OUTPUT_RAW.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Raw RGB plot
    print(
        "Generating raw RGB plot..."
    )

    plot_raw_rgb(
        dataframe
    )

    # Filtered regional signals
    print(
        "Generating filtered regional plot..."
    )

    plot_filtered_signals(
        dataframe,
        fps
    )

    print("\nSaved:")

    print(
        OUTPUT_RAW
    )

    print(
        OUTPUT_FILTERED
    )

    print(
        "\nSignal visualization completed."
    )


if __name__ == "__main__":
    main()