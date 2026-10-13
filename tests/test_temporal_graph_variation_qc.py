import numpy as np
import pytest

from src.features.gsp_features import temporal_graph_variation


def test_variation_uses_consecutive_frames_only():
    signals = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [100.0, 100.0],
        [103.0, 104.0],
    ])
    frames = np.array([0, 1, 3, 4])

    result = temporal_graph_variation(
        signals,
        frame_ids=frames,
    )

    # The transition from frame 1 to frame 3 is a gap.
    # Its large difference must be excluded.
    np.testing.assert_allclose(result, [5.0, 5.0])


def test_all_consecutive_frames_produce_t_minus_one_values():
    signals = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [6.0, 8.0],
        [9.0, 12.0],
    ])
    frames = np.array([10, 11, 12, 13])

    result = temporal_graph_variation(
        signals,
        frame_ids=frames,
    )

    assert len(result) == len(signals) - 1
    np.testing.assert_allclose(result, [5.0, 5.0, 5.0])


def test_frame_id_length_must_match_signal_rows():
    signals = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    frames = np.array([0, 1])

    with pytest.raises(ValueError, match="length"):
        temporal_graph_variation(
            signals,
            frame_ids=frames,
        )


def test_frame_ids_must_be_one_dimensional():
    signals = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    frames = np.array([[0], [1]])

    with pytest.raises(ValueError, match="one-dimensional"):
        temporal_graph_variation(
            signals,
            frame_ids=frames,
        )


def test_non_integer_frame_ids_are_rejected():
    signals = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    with pytest.raises(ValueError, match="integer"):
        temporal_graph_variation(
            signals,
            frame_ids=[0, 1.5, 2],
        )


def test_non_finite_frame_ids_are_rejected():
    signals = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    with pytest.raises(ValueError, match="finite"):
        temporal_graph_variation(
            signals,
            frame_ids=[0, np.nan, 2],
        )


@pytest.mark.parametrize(
    "frames",
    [
        [0, 1, 1],  # Duplicate frame
        [0, 2, 1],  # Decreasing frame order
    ],
)
def test_duplicate_or_unordered_frames_are_rejected(frames):
    signals = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    with pytest.raises(ValueError, match="strictly increasing"):
        temporal_graph_variation(
            signals,
            frame_ids=frames,
        )
