from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.physiology.signal_continuity import (
    RGB_COLUMNS,
    restore_regular_rgb_grid,
    continuity_statistics,
)


RGB_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_real_rgb.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_real_rgb_continuous.csv"
)

FPS = 25.0

MAX_GAP_FRAMES = 6


def main():

    print(
        "\nDeepGraph-Phys — "
        "RGB Signal Continuity Test"
    )

    print("-" * 60)

    dataframe = pd.read_csv(
        RGB_PATH
    )

    print(
        "Original detected RGB rows:",
        len(dataframe),
    )

    restored = (
        restore_regular_rgb_grid(
            dataframe,
            fps=FPS,
            max_gap_frames=MAX_GAP_FRAMES,
        )
    )

    stats = (
        continuity_statistics(
            restored
        )
    )

    print(
        "\nContinuity statistics:"
    )

    for key, value in (
        stats.items()
    ):

        if key == "valid_rate":

            print(
                f"{key}: "
                f"{value * 100:.2f}%"
            )

        else:

            print(
                f"{key}: {value}"
            )

    valid_mask = (
        restored[RGB_COLUMNS]
        .notna()
        .all(axis=1)
    )

    print(
        "\nValidation checks:"
    )

    print(
        "Frame numbers continuous:",
        (
            restored["frame"]
            .diff()
            .dropna()
            == 1
        ).all(),
    )

    print(
        "Time monotonic:",
        restored[
            "time_seconds"
        ].is_monotonic_increasing,
    )

    print(
        "No duplicate frames:",
        not restored[
            "frame"
        ].duplicated().any(),
    )

    print(
        "At least original frames retained:",
        stats["valid_frames"]
        >= stats["original_frames"],
    )

    print(
        "All usable RGB values finite:",
        (
            restored.loc[
                valid_mask,
                RGB_COLUMNS,
            ]
            .notna()
            .all()
            .all()
        ),
    )

    restored.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()