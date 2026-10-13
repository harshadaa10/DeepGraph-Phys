import numpy as np
import pandas as pd
import pytest

from src.graph.dynamic_graph import build_graph_signal_matrix
from src.graph.facial_graph import FACIAL_NODES


def make_synchronized_dataframe():
    data = {
        "forehead_rppg": [0.1, 0.2],
        "left_cheek_rppg": [0.3, 0.4],
        "right_cheek_rppg": [0.5, 0.6],
    }

    for index, region in enumerate(FACIAL_NODES):
        data[f"{region}_motion"] = [
            float(index + 1),
            float(index + 2),
        ]

    return pd.DataFrame(data)


def test_graph_signal_matrix_shapes_and_physiology_mask():
    dataframe = make_synchronized_dataframe()

    physiology, motion, mask = build_graph_signal_matrix(
        dataframe
    )

    assert physiology.shape == (2, 7)
    assert motion.shape == (2, 7)
    assert mask.shape == (7,)

    np.testing.assert_array_equal(
        mask,
        [True, True, True, False, False, False, False],
    )


def test_graph_signal_values_follow_facial_node_order():
    dataframe = make_synchronized_dataframe()

    physiology, motion, mask = build_graph_signal_matrix(
        dataframe
    )

    np.testing.assert_allclose(
        physiology[:, :3],
        [
            [0.1, 0.3, 0.5],
            [0.2, 0.4, 0.6],
        ],
    )

    np.testing.assert_allclose(
        motion[0],
        [1, 2, 3, 4, 5, 6, 7],
    )

    assert np.isnan(physiology[:, 3:]).all()


def test_missing_motion_column_is_rejected():
    dataframe = make_synchronized_dataframe()
    dataframe = dataframe.drop(columns=["nose_motion"])

    with pytest.raises(ValueError, match="nose_motion"):
        build_graph_signal_matrix(dataframe)


def test_missing_rppg_column_is_rejected():
    dataframe = make_synchronized_dataframe()
    dataframe = dataframe.drop(columns=["left_cheek_rppg"])

    with pytest.raises(ValueError, match="left_cheek_rppg"):
        build_graph_signal_matrix(dataframe)
