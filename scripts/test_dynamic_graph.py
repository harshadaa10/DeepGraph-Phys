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
            f"rPPG={'YES' if has_physiology else 'NO ':3s} "
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

    expected_samples = 276

    print(
        "\nValidation checks:"
    )

    print(
        "Expected synchronized samples:",
        expected_samples
    )

    print(
        "Sample count correct:",
        len(frames) == expected_samples
    )

    print(
        "Physiology finite where available:",
        np.isfinite(
            physiology[
                :,
                physiology_mask
            ]
        ).all()
    )

    print(
        "Motion values finite:",
        np.isfinite(
            motion
        ).all()
    )

    print(
        "Missing physiology stored as NaN:",
        np.isnan(
            physiology[
                :,
                ~physiology_mask
            ]
        ).all()
    )

    print(
        "\nDynamic graph signal test completed."
    )


if __name__ == "__main__":
    main()