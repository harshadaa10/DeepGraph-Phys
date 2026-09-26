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


from src.graph.facial_graph import (
    FACIAL_NODES,
)

from src.graph.dynamic_graph import (
    create_dynamic_graph_data,
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
        "\nDeepGraph-Phys — Dynamic Graph Signal Test"
    )

    print("-" * 55)

    # ------------------------------------------
    # Load signals
    # ------------------------------------------

    rppg_dataframe = pd.read_csv(
        RPPG_PATH
    )

    motion_dataframe = pd.read_csv(
        MOTION_PATH
    )

    print(
        "rPPG samples:",
        len(rppg_dataframe)
    )

    print(
        "Motion samples:",
        len(motion_dataframe)
    )

    # ------------------------------------------
    # Determine expected synchronized frames
    # ------------------------------------------

    expected_frames = np.intersect1d(
        rppg_dataframe[
            "frame"
        ].values,
        motion_dataframe[
            "frame"
        ].values,
    )

    expected_samples = len(
        expected_frames
    )

    # ------------------------------------------
    # Create dynamic graph data
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

    synchronized = dynamic_data[
        "dataframe"
    ]

    physiology = dynamic_data[
        "physiology"
    ]

    motion = dynamic_data[
        "motion"
    ]

    physiology_mask = dynamic_data[
        "physiology_mask"
    ]

    frames = dynamic_data[
        "frames"
    ]

    time = dynamic_data[
        "time"
    ]

    # ------------------------------------------
    # Basic information
    # ------------------------------------------

    print(
        "\nSynchronized samples:",
        len(frames)
    )

    if len(frames) == 0:
        raise ValueError(
            "No synchronized rPPG and motion "
            "frames were found."
        )

    print(
        "First synchronized frame:",
        frames[0]
    )

    print(
        "Last synchronized frame:",
        frames[-1]
    )

    print(
        f"First timestamp: {time[0]:.4f}"
    )

    print(
        f"Last timestamp: {time[-1]:.4f}"
    )

    print(
        "\nGraph nodes:",
        graph.number_of_nodes()
    )

    print(
        "Graph edges:",
        graph.number_of_edges()
    )

    # ------------------------------------------
    # Matrix shapes
    # ------------------------------------------

    print(
        "\nPhysiology matrix shape:",
        physiology.shape
    )

    print(
        "Motion matrix shape:",
        motion.shape
    )

    # ------------------------------------------
    # Interpolation quality information
    # ------------------------------------------

    if "interpolated" in synchronized.columns:

        interpolated_samples = int(
            synchronized[
                "interpolated"
            ].sum()
        )

        interpolation_rate = (
            interpolated_samples
            / len(synchronized)
            * 100.0
        )

        print(
            "\nSynchronized rPPG quality:"
        )

        print(
            "Interpolated synchronized samples:",
            interpolated_samples
        )

        print(
            f"Interpolation rate: "
            f"{interpolation_rate:.2f}%"
        )

    # ------------------------------------------
    # Node availability
    # ------------------------------------------

    print(
        "\nNode signal availability:"
    )

    for index, node in enumerate(
        FACIAL_NODES
    ):

        has_physiology = (
            physiology_mask[index]
        )

        print(
            f"{index}: "
            f"{node:12s} "
            f"rPPG="
            f"{'YES' if has_physiology else 'NO ':3s} "
            f"motion=YES"
        )

    # ------------------------------------------
    # First graph state
    # ------------------------------------------

    print(
        "\nFirst synchronized graph state:"
    )

    print(
        f"Frame: {frames[0]}"
    )

    print(
        f"Time : {time[0]:.4f} seconds"
    )

    for index, node in enumerate(
        FACIAL_NODES
    ):

        if physiology_mask[index]:

            physiology_value = (
                f"{physiology[0, index]:.4f}"
            )

        else:

            physiology_value = "N/A"

        motion_value = (
            motion[0, index]
        )

        print(
            f"{node:12s} "
            f"rPPG={physiology_value:>8s} "
            f"motion={motion_value:.6f}"
        )

    # ------------------------------------------
    # Validation
    # ------------------------------------------

    sample_count_correct = (
        len(frames)
        == expected_samples
    )

    synchronized_frames_correct = (
        np.array_equal(
            frames,
            expected_frames,
        )
    )

    physiology_shape_correct = (
        physiology.shape
        == (
            expected_samples,
            len(FACIAL_NODES),
        )
    )

    motion_shape_correct = (
        motion.shape
        == (
            expected_samples,
            len(FACIAL_NODES),
        )
    )

    physiology_finite = (
        np.isfinite(
            physiology[
                :,
                physiology_mask
            ]
        ).all()
    )

    motion_finite = (
        np.isfinite(
            motion
        ).all()
    )

    missing_physiology_nan = (
        np.isnan(
            physiology[
                :,
                ~physiology_mask
            ]
        ).all()
    )

    time_monotonic = (
        np.all(
            np.diff(time) > 0
        )
    )

    print(
        "\nValidation checks:"
    )

    print(
        "Expected synchronized samples:",
        expected_samples
    )

    print(
        "Sample count correct:",
        sample_count_correct
    )

    print(
        "Synchronized frame IDs correct:",
        synchronized_frames_correct
    )

    print(
        "Physiology matrix shape correct:",
        physiology_shape_correct
    )

    print(
        "Motion matrix shape correct:",
        motion_shape_correct
    )

    print(
        "Physiology finite where available:",
        physiology_finite
    )

    print(
        "Motion values finite:",
        motion_finite
    )

    print(
        "Missing physiology stored as NaN:",
        missing_physiology_nan
    )

    print(
        "Timestamps strictly increasing:",
        time_monotonic
    )

    all_checks_passed = all(
        [
            sample_count_correct,
            synchronized_frames_correct,
            physiology_shape_correct,
            motion_shape_correct,
            physiology_finite,
            motion_finite,
            missing_physiology_nan,
            time_monotonic,
        ]
    )

    print(
        "\nAll validation checks passed:",
        all_checks_passed
    )

    print(
        "\nDynamic graph signal test completed."
    )


if __name__ == "__main__":
    main()