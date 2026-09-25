import numpy as np
import pandas as pd

from src.physiology.pos import pos_rppg


RPPG_REGIONS = [
    "forehead",
    "left_cheek",
    "right_cheek",
]


def generate_rppg_dataframe(
    rgb_dataframe,
    fps,
):
    """
    Generate POS rPPG signals for each
    physiological facial region.
    """

    output = pd.DataFrame()

    output["frame"] = (
        rgb_dataframe["frame"]
    )

    output["time_seconds"] = (
        rgb_dataframe["time_seconds"]
    )

    for region in RPPG_REGIONS:

        rgb = np.column_stack(
            [
                rgb_dataframe[
                    f"{region}_R"
                ].values,

                rgb_dataframe[
                    f"{region}_G"
                ].values,

                rgb_dataframe[
                    f"{region}_B"
                ].values,
            ]
        )

        rppg = pos_rppg(
            rgb,
            fps
        )

        output[
            f"{region}_rppg"
        ] = rppg

    return output