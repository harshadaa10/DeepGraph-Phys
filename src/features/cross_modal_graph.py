"""
Segment-Aware Cross-Modal Motion–Physiology Graph Features

Experimental Phase 13 module for DeepGraph-Phys.

This module measures how consistently regional facial motion and
physiological/rPPG signals relate to each other across valid rPPG
segments.

Only the three regions where both modalities are available are used:
    - forehead
    - left_cheek
    - right_cheek

Important:
    - Interpolated rPPG samples are NOT treated as normal evidence.
    - Correlations are calculated separately for valid segments.
    - Segment correlations are aggregated using valid-sample weighting.
    - Raw sample counts are not used as predictive features.
    - This module does not modify the locked Phase 12D pipeline.
"""

import numpy as np
import pandas as pd


CROSS_MODAL_REGIONS = [
    "forehead",
    "left_cheek",
    "right_cheek",
]


MIN_SEGMENT_SAMPLES = 10


def _safe_correlation(x, y):
    """Calculate Pearson correlation safely."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return 0.0

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0

    correlation = np.corrcoef(x, y)[0, 1]

    if not np.isfinite(correlation):
        return 0.0

    return float(correlation)


def _safe_mean(values):
    """Calculate mean while ignoring non-finite values."""
    values = np.asarray(values, dtype=np.float64)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    return float(np.mean(values))


def _safe_std(values):
    """Calculate standard deviation while ignoring non-finite values."""
    values = np.asarray(values, dtype=np.float64)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    return float(np.std(values))


def _get_valid_segment_correlations(dataframe, region):
    """
    Calculate motion-rPPG correlation separately for each valid rPPG segment.

    Interpolated rPPG samples are excluded from correlation calculation.
    """

    motion_column = f"{region}_motion"
    rppg_column = f"{region}_rppg"

    if motion_column not in dataframe.columns:
        raise ValueError(f"Missing motion column: {motion_column}")

    if rppg_column not in dataframe.columns:
        raise ValueError(f"Missing rPPG column: {rppg_column}")

    if "segment_id" not in dataframe.columns:
        raise ValueError(
            "Expected 'segment_id' column for segment-aware CMPC."
        )

    motion = dataframe[motion_column].to_numpy(dtype=np.float64)
    rppg = dataframe[rppg_column].to_numpy(dtype=np.float64)

    # Interpolated samples are not used as direct physiological evidence.
    if "interpolated" in dataframe.columns:
        interpolated = dataframe["interpolated"].fillna(False).astype(bool).to_numpy()
    else:
        interpolated = np.zeros(len(dataframe), dtype=bool)

    segment_ids = dataframe["segment_id"].to_numpy()

    valid = (
        np.isfinite(motion)
        & np.isfinite(rppg)
        & (~interpolated)
        & pd.notna(segment_ids)
    )

    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:
        return []

    segment_results = []

    valid_segment_ids = segment_ids[valid_indices]

    for segment_id in pd.unique(valid_segment_ids):

        segment_mask = (
            valid
            & (segment_ids == segment_id)
        )

        segment_motion = motion[segment_mask]
        segment_rppg = rppg[segment_mask]

        sample_count = len(segment_motion)

        if sample_count < MIN_SEGMENT_SAMPLES:
            continue

        correlation = _safe_correlation(
            segment_motion,
            segment_rppg,
        )

        segment_results.append(
            {
                "segment_id": segment_id,
                "correlation": correlation,
                "absolute_correlation": abs(correlation),
                "valid_samples": int(sample_count),
            }
        )

    return segment_results


def calculate_region_cross_modal_consistency(dataframe, region):
    """
    Calculate segment-aware cross-modal consistency for one region.
    """

    segment_results = _get_valid_segment_correlations(
        dataframe,
        region,
    )

    if len(segment_results) == 0:
        return {
            "correlation": 0.0,
            "absolute_correlation": 0.0,
            "segment_std": 0.0,
            "segment_count": 0.0,
            "motion_std": 0.0,
            "rppg_std": 0.0,
        }

    correlations = np.array(
        [item["correlation"] for item in segment_results],
        dtype=np.float64,
    )

    weights = np.array(
        [item["valid_samples"] for item in segment_results],
        dtype=np.float64,
    )

    absolute_correlations = np.abs(correlations)

    total_weight = np.sum(weights)

    if total_weight > 0:
        weighted_correlation = float(
            np.average(
                correlations,
                weights=weights,
            )
        )
    else:
        weighted_correlation = 0.0

        # Calculate variability only from valid samples belonging to
    # segments that passed MIN_SEGMENT_SAMPLES.
    motion_column = f"{region}_motion"
    rppg_column = f"{region}_rppg"

    motion = dataframe[motion_column].to_numpy(dtype=np.float64)
    rppg = dataframe[rppg_column].to_numpy(dtype=np.float64)

    if "interpolated" in dataframe.columns:
        interpolated = (
            dataframe["interpolated"]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )
    else:
        interpolated = np.zeros(len(dataframe), dtype=bool)

    segment_ids = dataframe["segment_id"].to_numpy()

    accepted_segments = {
        item["segment_id"] for item in segment_results
    }

    valid_signal = (
        np.isfinite(motion)
        & np.isfinite(rppg)
        & (~interpolated)
        & pd.notna(segment_ids)
        & np.isin(segment_ids, list(accepted_segments))
    )

    motion_valid = motion[valid_signal]
    rppg_valid = rppg[valid_signal]

    return {
        "correlation": weighted_correlation,
        "absolute_correlation": abs(weighted_correlation),
        "segment_std": _safe_std(correlations),
        "segment_count": float(len(segment_results)),
        "motion_std": _safe_std(motion_valid),
        "rppg_std": _safe_std(rppg_valid),
    }


def calculate_cross_region_consistency(region_correlations):
    """
    Calculate consistency across the three shared facial regions.
    """

    correlations = np.asarray(
        list(region_correlations.values()),
        dtype=np.float64,
    )

    correlations = correlations[np.isfinite(correlations)]

    if len(correlations) == 0:
        return {
            "mean_correlation": 0.0,
            "std_correlation": 0.0,
            "mean_absolute_correlation": 0.0,
            "min_absolute_correlation": 0.0,
            "max_absolute_correlation": 0.0,
        }

    absolute_correlations = np.abs(correlations)

    return {
        "mean_correlation": float(np.mean(correlations)),
        "std_correlation": float(np.std(correlations)),
        "mean_absolute_correlation": float(
            np.mean(absolute_correlations)
        ),
        "min_absolute_correlation": float(
            np.min(absolute_correlations)
        ),
        "max_absolute_correlation": float(
            np.max(absolute_correlations)
        ),
    }

def extract_cross_modal_graph_features(dataframe):
    """
    Extract the complete segment-aware CMPC feature vector.

    Predictive features exclude segment counts, which remain
    available through the diagnostic region table.
    """
    features = {}
    region_results = {}

    # Collect results for all three shared facial regions.
    for region in CROSS_MODAL_REGIONS:
        consistency = calculate_region_cross_modal_consistency(
            dataframe, region
        )
        region_results[region] = consistency

    # Regional cross-modal relationships: 3 regions x 3 features = 9.
    for region in CROSS_MODAL_REGIONS:
        consistency = region_results[region]

        features[f"cmpc_{region}_correlation"] = (
            consistency["correlation"]
        )
        features[f"cmpc_{region}_absolute_correlation"] = (
            consistency["absolute_correlation"]
        )
        features[f"cmpc_{region}_segment_std"] = (
            consistency["segment_std"]
        )

    # Calculate consistency across regions.
    correlations = {
        region: result["correlation"]
        for region, result in region_results.items()
    }
    cross_region = calculate_cross_region_consistency(correlations)

    features["cmpc_cross_region_mean_correlation"] = (
        cross_region["mean_correlation"]
    )
    features["cmpc_cross_region_std_correlation"] = (
        cross_region["std_correlation"]
    )
    features["cmpc_cross_region_mean_abs_correlation"] = (
        cross_region["mean_absolute_correlation"]
    )
    features["cmpc_cross_region_min_abs_correlation"] = (
        cross_region["min_absolute_correlation"]
    )
    features["cmpc_cross_region_max_abs_correlation"] = (
        cross_region["max_absolute_correlation"]
    )

    # Overall signal variability: 2 features.
    features["cmpc_mean_motion_std"] = _safe_mean(
        [result["motion_std"] for result in region_results.values()]
    )
    features["cmpc_mean_rppg_std"] = _safe_mean(
        [result["rppg_std"] for result in region_results.values()]
    )

    # Replace any unexpected non-finite result safely.
    for feature_name, value in features.items():
        if not np.isfinite(value):
            features[feature_name] = 0.0

    return features


def extract_cross_modal_region_table(dataframe):
    
    """
    Return a human-readable regional CMPC table.
    """

    rows = []

    for region in CROSS_MODAL_REGIONS:

        result = calculate_region_cross_modal_consistency(
            dataframe,
            region,
        )

        rows.append(
            {
                "region": region,
                **result,
            }
        )

    return pd.DataFrame(rows)