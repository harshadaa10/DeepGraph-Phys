import numpy as np
import pandas as pd

from src.graph.facial_graph import (
    FACIAL_NODES,
    build_facial_graph,
)


PHYSIOLOGY_REGIONS = [
    "forehead",
    "left_cheek",
    "right_cheek",
]


def synchronize_signals(
    rppg_dataframe,
    motion_dataframe,
):
    """
    Synchronize rPPG and motion data using
    their frame numbers.
    """

    synchronized = pd.merge(
        rppg_dataframe,
        motion_dataframe,
        on="frame",
        how="inner",
        suffixes=(
            "_rppg_time",
            "_motion_time",
        ),
    )

    # Use the rPPG timestamp as the common timestamp.
    synchronized["time_seconds"] = (
        synchronized[
            "time_seconds_rppg_time"
        ]
    )

    return synchronized


def build_graph_signal_matrix(
    synchronized_dataframe,
):
    """
    Convert synchronized physiological and motion
    signals into graph signal matrices.

    Returns
    -------
    physiology_matrix:
        Shape = (T, 7)

    motion_matrix:
        Shape = (T, 7)

    physiology_mask:
        Shape = (7,)
        True only for nodes where rPPG exists.
    """

    num_samples = len(
        synchronized_dataframe
    )

    num_nodes = len(
        FACIAL_NODES
    )

    physiology_matrix = np.full(
        (num_samples, num_nodes),
        np.nan,
        dtype=np.float64,
    )

    motion_matrix = np.zeros(
        (num_samples, num_nodes),
        dtype=np.float64,
    )

    physiology_mask = np.zeros(
        num_nodes,
        dtype=bool,
    )

    for node_index, region in enumerate(
        FACIAL_NODES
    ):

        # --------------------------------------
        # Motion exists for all seven nodes
        # --------------------------------------

        motion_column = (
            f"{region}_motion"
        )

        if (
            motion_column
            in synchronized_dataframe.columns
        ):

            motion_matrix[
                :,
                node_index
            ] = synchronized_dataframe[
                motion_column
            ].values

        # --------------------------------------
        # rPPG exists only for skin ROIs
        # --------------------------------------

        if region in PHYSIOLOGY_REGIONS:

            rppg_column = (
                f"{region}_rppg"
            )

            physiology_matrix[
                :,
                node_index
            ] = synchronized_dataframe[
                rppg_column
            ].values

            physiology_mask[
                node_index
            ] = True

    return (
        physiology_matrix,
        motion_matrix,
        physiology_mask,
    )


def create_dynamic_graph_data(
    rppg_dataframe,
    motion_dataframe,
):
    """
    Create the synchronized dynamic facial graph
    representation.
    """

    synchronized = synchronize_signals(
        rppg_dataframe,
        motion_dataframe,
    )

    (
        physiology_matrix,
        motion_matrix,
        physiology_mask,
    ) = build_graph_signal_matrix(
        synchronized
    )

    graph = build_facial_graph()

    return {
        "graph": graph,
        "dataframe": synchronized,
        "physiology": physiology_matrix,
        "motion": motion_matrix,
        "physiology_mask": physiology_mask,
        "frames": synchronized[
            "frame"
        ].values,
        "time": synchronized[
            "time_seconds"
        ].values,
    }