
import pandas as pd
import pytest

from src.graph.dynamic_graph import synchronize_signals


def make_rppg(frames):
    return pd.DataFrame({
        "frame": frames,
        "time_seconds": [f / 30.0 for f in frames],
        "interpolated": [False] * len(frames),
        "segment_id": [0] * len(frames),
        "forehead_rppg": [0.1] * len(frames),
    })


def make_motion(frames):
    return pd.DataFrame({
        "frame": frames,
        "time_seconds": [f / 30.0 for f in frames],
        "forehead_motion": [0.2] * len(frames),
    })


def test_sync_keeps_only_matching_frames():
    rppg = make_rppg([0, 1, 3])
    motion = make_motion([0, 1, 2, 3])

    result = synchronize_signals(rppg, motion)

    assert result["frame"].tolist() == [0, 1, 3]


def test_sync_preserves_rppg_metadata():
    rppg = make_rppg([0, 1, 2])
    motion = make_motion([0, 1, 2])

    rppg["interpolated"] = [False, True, False]
    rppg["segment_id"] = [0, 0, 1]

    result = synchronize_signals(rppg, motion)

    assert result["interpolated"].tolist() == [
        False, True, False
    ]
    assert result["segment_id"].tolist() == [0, 0, 1]


def test_sync_does_not_modify_input_dataframes():
    rppg = make_rppg([0, 1])
    motion = make_motion([0, 1])

    original_rppg = rppg.copy(deep=True)
    original_motion = motion.copy(deep=True)

    synchronize_signals(rppg, motion)

    pd.testing.assert_frame_equal(rppg, original_rppg)
    pd.testing.assert_frame_equal(motion, original_motion)


def test_sync_rejects_duplicate_frame_ids():
    rppg = make_rppg([0, 1, 1])
    motion = make_motion([0, 1])

    with pytest.raises(pd.errors.MergeError):
        synchronize_signals(rppg, motion)


def test_sync_rejects_missing_required_columns():
    rppg = pd.DataFrame({
        "frame": [0],
        "time_seconds": [0.0],
    })
    motion = make_motion([0])

    with pytest.raises(ValueError, match="Missing rPPG columns"):
        synchronize_signals(rppg, motion)


def test_sync_preserves_both_source_timestamps():
    rppg = make_rppg([0, 1, 2])
    motion = make_motion([0, 1, 2])

    result = synchronize_signals(rppg, motion)

    assert "time_seconds_rppg_time" in result.columns
    assert "time_seconds_motion_time" in result.columns

    assert (
        result["time_seconds"].tolist()
        == result["time_seconds_rppg_time"].tolist()
    )


def test_sync_exposes_timestamp_difference():
    rppg = make_rppg([0, 1, 2])
    motion = make_motion([0, 1, 2])

    # Introduce a deliberate 20 ms offset in motion timestamps.
    motion["time_seconds"] += 0.020

    result = synchronize_signals(rppg, motion)

    timestamp_difference = (
        result["time_seconds_rppg_time"]
        - result["time_seconds_motion_time"]
    ).abs()

    assert (timestamp_difference > 0.019).all()
    assert (timestamp_difference < 0.021).all()
