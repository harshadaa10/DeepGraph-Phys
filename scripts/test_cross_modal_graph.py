import numpy as np
import pandas as pd

from src.features.cross_modal_graph import (
    calculate_region_cross_modal_consistency,
    calculate_cross_region_consistency,
    extract_cross_modal_graph_features,
    extract_cross_modal_region_table,
)


def create_synthetic_dataframe():
    time = np.linspace(0, 10, 100)

    forehead_motion = np.sin(time)
    left_cheek_motion = np.cos(time)
    right_cheek_motion = np.sin(time * 0.5)

    dataframe = pd.DataFrame({
        "forehead_motion": forehead_motion,
        "left_cheek_motion": left_cheek_motion,
        "right_cheek_motion": right_cheek_motion,
        "forehead_rppg": forehead_motion.copy(),
        "left_cheek_rppg": -left_cheek_motion,
        "right_cheek_rppg": right_cheek_motion.copy(),
        "segment_id": [0] * 50 + [1] * 50,
        "interpolated": [False] * 100,
    })

    return dataframe


def test_positive_relationship():
    dataframe = create_synthetic_dataframe()

    result = calculate_region_cross_modal_consistency(
        dataframe, "forehead"
    )

    assert result["correlation"] > 0.99
    assert result["absolute_correlation"] > 0.99


def test_negative_relationship():
    dataframe = create_synthetic_dataframe()

    result = calculate_region_cross_modal_consistency(
        dataframe, "left_cheek"
    )

    assert result["correlation"] < -0.99
    assert result["absolute_correlation"] > 0.99


def test_cross_region_consistency():
    correlations = {
        "forehead": 0.9,
        "left_cheek": -0.8,
        "right_cheek": 0.7,
    }

    result = calculate_cross_region_consistency(correlations)

    assert np.isfinite(result["mean_correlation"])
    assert np.isfinite(result["std_correlation"])
    assert result["min_absolute_correlation"] == 0.7
    assert result["max_absolute_correlation"] == 0.9


def test_nan_values_are_handled():
    dataframe = create_synthetic_dataframe()
    dataframe.loc[0:4, "forehead_rppg"] = np.nan

    result = calculate_region_cross_modal_consistency(
        dataframe, "forehead"
    )

    assert result["segment_count"] == 2.0
    assert np.isfinite(result["correlation"])


def test_constant_signal_is_safe():
    dataframe = create_synthetic_dataframe()
    dataframe["forehead_rppg"] = 1.0

    result = calculate_region_cross_modal_consistency(
        dataframe, "forehead"
    )

    assert result["correlation"] == 0.0
    assert result["absolute_correlation"] == 0.0


def test_interpolated_samples_are_excluded():
    dataframe = create_synthetic_dataframe()

    # Mark a group of samples as interpolated.
    # Their values must not contribute to segment correlations.
    dataframe.loc[10:19, "interpolated"] = True
    dataframe.loc[10:19, "forehead_rppg"] *= -1

    result = calculate_region_cross_modal_consistency(
        dataframe, "forehead"
    )

    assert result["correlation"] > 0.99
    assert result["segment_count"] == 2.0


def test_feature_extraction_is_finite():
    dataframe = create_synthetic_dataframe()

    features = extract_cross_modal_graph_features(dataframe)

    # 9 regional + 5 cross-region + 2 signal-variability features.
    assert len(features) == 16

    # Segment counts are diagnostic metadata, not predictive features.
    assert not any(
        "segment_count" in name for name in features
    )

    for name, value in features.items():
        assert np.isfinite(value), (
            f"Feature {name} is not finite: {value}"
        )


def test_segment_count_remains_in_diagnostic_table():
    dataframe = create_synthetic_dataframe()

    region_table = extract_cross_modal_region_table(dataframe)

    assert "segment_count" in region_table.columns
    assert set(region_table["region"]) == {
        "forehead", "left_cheek", "right_cheek"
    }

def test_short_segments_do_not_contribute_to_variability():
    dataframe = create_synthetic_dataframe()

    # Keep segment 0 valid, but make segment 1 too short.
    dataframe.loc[50:, "interpolated"] = True

    result = calculate_region_cross_modal_consistency(
        dataframe, "forehead"
    )

    assert result["segment_count"] == 1.0
    assert np.isfinite(result["motion_std"])
    assert np.isfinite(result["rppg_std"])