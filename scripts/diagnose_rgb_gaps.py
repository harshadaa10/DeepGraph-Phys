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


RGB_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_fake_rgb.csv"
)


def main():

    print(
        "\nDeepGraph-Phys — RGB Gap Diagnostic"
    )

    print("-" * 60)

    if not RGB_PATH.exists():
        raise FileNotFoundError(
            f"RGB CSV not found: {RGB_PATH}"
        )

    dataframe = pd.read_csv(
        RGB_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "RGB dataframe is empty."
        )

    frames = (
        dataframe["frame"]
        .astype(int)
        .sort_values()
        .to_numpy()
    )

    differences = np.diff(
        frames
    )

    missing_between = (
        differences - 1
    )

    gap_mask = (
        missing_between > 0
    )

    gap_differences = (
        differences[
            gap_mask
        ]
    )

    missing_counts = (
        missing_between[
            gap_mask
        ]
    )

    print(
        "Detected RGB rows:",
        len(dataframe)
    )

    print(
        "First detected frame:",
        frames[0]
    )

    print(
        "Last detected frame:",
        frames[-1]
    )

    print(
        "Detected-frame transitions:",
        len(differences)
    )

    print(
        "Consecutive transitions:",
        int(
            np.sum(
                differences == 1
            )
        )
    )

    print(
        "Non-consecutive transitions:",
        int(
            np.sum(
                differences > 1
            )
        )
    )

    if len(differences) > 0:

        consecutive_rate = (
            np.mean(
                differences == 1
            )
            * 100.0
        )

    else:

        consecutive_rate = 0.0

    print(
        f"Consecutive transition rate: "
        f"{consecutive_rate:.2f}%"
    )

    if len(gap_differences) == 0:

        print(
            "\nNo missing-frame gaps found."
        )

        return

    print(
        "\nLargest detected-frame difference:",
        int(
            np.max(
                gap_differences
            )
        )
    )

    print(
        "Largest actual missing run:",
        int(
            np.max(
                missing_counts
            )
        )
    )

    print(
        "Total missing frames between "
        "detections:",
        int(
            np.sum(
                missing_counts
            )
        )
    )

    # ------------------------------------------
    # Gap-size distribution
    # ------------------------------------------

    unique_sizes, counts = (
        np.unique(
            missing_counts,
            return_counts=True,
        )
    )

    print(
        "\nMissing-run distribution:"
    )

    for size, count in zip(
        unique_sizes,
        counts,
    ):

        print(
            f"{int(size):3d} missing frame(s): "
            f"{int(count)} gap(s)"
        )

    # ------------------------------------------
    # Show individual gaps
    # ------------------------------------------

    print(
        "\nIndividual gaps:"
    )

    gap_indices = np.where(
        differences > 1
    )[0]

    gap_records = []

    for gap_number, index in enumerate(
        gap_indices,
        start=1,
    ):

        previous_frame = int(
            frames[index]
        )

        next_frame = int(
            frames[index + 1]
        )

        missing_count = (
            next_frame
            - previous_frame
            - 1
        )

        record = {
            "gap_number":
                gap_number,

            "previous_detected_frame":
                previous_frame,

            "next_detected_frame":
                next_frame,

            "first_missing_frame":
                previous_frame + 1,

            "last_missing_frame":
                next_frame - 1,

            "missing_frames":
                missing_count,
        }

        gap_records.append(
            record
        )

        print(
            f"Gap {gap_number:3d}: "
            f"{previous_frame} -> "
            f"{next_frame} | "
            f"missing={missing_count}"
        )

    # ------------------------------------------
    # Save diagnostic report
    # ------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "data"
        / "features"
        / "sample_fake_rgb_gaps.csv"
    )

    gap_dataframe = pd.DataFrame(
        gap_records
    )

    gap_dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nGap report saved:"
    )

    print(
        output_path
    )

    print(
        "\nRGB gap diagnostic completed."
    )


if __name__ == "__main__":
    main()