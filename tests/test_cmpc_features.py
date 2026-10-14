import numpy as np
import pandas as pd
import pytest

from src.features.cross_modal_graph import (
    CROSS_MODAL_REGIONS,
    MIN_SEGMENT_SAMPLES,
    calculate_region_cross_modal_consistency,
    extract_cross_modal_graph_features,
    extract_cross_modal_region_table,
)


def make_synthetic_data(segment_sizes=(12, 12)):
    """Build deterministic synthetic motion and rPPG signals."""
    frames = []
    offset = 0

    for segment_id, size in enumerate(segment_sizes, start=1):
        x = np.arange(size, dtype=float)

        for i in range(size):
            row = {
                "frame": offset + i,
                "segment_id": segment_id,
                "interpolated": False,
            }

            for region_index, region in enumerate(
                CROSS_MODAL_REGIONS
            ):
                motion = x[i] + region_index * 0.5
                rppg = x[i] * (1 if region_index != 1 else -1)

                row[f"{region}_motion"] = motion
                row[f"{region}_rppg"] = rppg

            frames.append(row)

        offset += size

    return pd.DataFrame(frames)


def test_valid_segments_produce_expected_correlations():
    data = make_synthetic_data()

    result = calculate_region_cross_modal_consistency(
        data, "forehead"
    )

    assert result["segment_count"] == 2
    assert result["correlation"] == pytest.approx(1.0)
    assert result["absolute_correlation"] == pytest.approx(1.0)


def test_interpolated_samples_are_excluded():
    data = make_synthetic_data(segment_sizes=(12,))

    # Add extreme values that must not affect the correlation.
    extra_rows = []

    for i in range(2):
        row = {
            "frame": len(data) + i,
            "segment_id": 1,
            "interpolated": True,
        }

        for region in CROSS_MODAL_REGIONS:
            row[f"{region}_motion"] = 1e9
            row[f"{region}_rppg"] = -1e9

        extra_rows.append(row)

    data = pd.concat(
        [data, pd.DataFrame(extra_rows)],
        ignore_index=True,
    )

    result = calculate_region_cross_modal_consistency(
        data, "forehead"
    )

    assert result["segment_count"] == 1
    assert result["correlation"] == pytest.approx(1.0)


def test_short_segments_are_ignored():
    data = make_synthetic_data(
        segment_sizes=(MIN_SEGMENT_SAMPLES - 1, 12)
    )

    result = calculate_region_cross_modal_consistency(
        data, "forehead"
    )

    assert result["segment_count"] == 1
    assert result["correlation"] == pytest.approx(1.0)


def test_constant_signals_are_handled_safely():
    data = make_synthetic_data(segment_sizes=(12,))

    for region in CROSS_MODAL_REGIONS:
        data[f"{region}_motion"] = 5.0
        data[f"{region}_rppg"] = 2.0

    result = calculate_region_cross_modal_consistency(
        data, "forehead"
    )

    assert result["correlation"] == 0.0
    assert result["absolute_correlation"] == 0.0
    assert result["motion_std"] == 0.0
    assert result["rppg_std"] == 0.0


def test_feature_vector_has_16_finite_values():
    data = make_synthetic_data()

    features = extract_cross_modal_graph_features(data)

    assert len(features) == 16
    assert all(np.isfinite(value) for value in features.values())


def test_region_table_contains_three_regions():
    data = make_synthetic_data()

    table = extract_cross_modal_region_table(data)

    assert len(table) == 3
    assert set(table["region"]) == set(CROSS_MODAL_REGIONS)
    assert "segment_count" in table.columns


def test_missing_required_column_raises_clear_error():
    data = make_synthetic_data().drop(
        columns=["forehead_motion"]
    )

    with pytest.raises(
        ValueError,
        match="Missing motion column",
    ):
        calculate_region_cross_modal_consistency(
            data, "forehead"
        )


def test_nan_samples_are_excluded_from_correlations():
    data = make_synthetic_data(segment_sizes=(12,))

    # Missing pairs should be excluded from correlation.
    data.loc[0, "forehead_motion"] = np.nan
    data.loc[1, "forehead_rppg"] = np.nan

    result = calculate_region_cross_modal_consistency(
        data, "forehead"
    )

    assert result["segment_count"] == 1
    assert np.isfinite(result["correlation"])
    assert result["correlation"] == pytest.approx(1.0)


def test_all_short_segments_return_finite_zero_features():
    data = make_synthetic_data(
        segment_sizes=(MIN_SEGMENT_SAMPLES - 1,)
    )

    features = extract_cross_modal_graph_features(data)

    assert len(features) == 16
    assert all(np.isfinite(value) for value in features.values())
    assert all(value == 0.0 for value in features.values())
