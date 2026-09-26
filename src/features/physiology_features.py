import numpy as np


PHYSIOLOGY_COLUMNS = [
    "forehead_rppg",
    "left_cheek_rppg",
    "right_cheek_rppg",
]


def build_physiology_matrix(
    synchronized_dataframe,
):
    """
    Build rPPG matrix with columns:

    0 -> forehead
    1 -> left_cheek
    2 -> right_cheek

    Shape:
        (T, 3)
    """

    matrix = synchronized_dataframe[
        PHYSIOLOGY_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )

    return matrix


def physiology_correlation_matrix(
    physiology_matrix,
):
    """
    Calculate Pearson correlation between
    the three regional rPPG signals.

    Parameters
    ----------
    physiology_matrix : ndarray
        Shape (T, 3).

    Returns
    -------
    ndarray
        3 x 3 correlation matrix.
    """

    physiology_matrix = np.asarray(
        physiology_matrix,
        dtype=np.float64,
    )

    if (
        physiology_matrix.ndim != 2
        or physiology_matrix.shape[1] != 3
    ):
        raise ValueError(
            "physiology_matrix must have "
            "shape (T, 3)."
        )

    if len(physiology_matrix) < 2:
        raise ValueError(
            "At least two physiology samples "
            "are required for correlation."
        )

    if not np.isfinite(
        physiology_matrix
    ).all():
        raise ValueError(
            "Physiology matrix contains "
            "non-finite values."
        )

    correlation = np.corrcoef(
        physiology_matrix,
        rowvar=False,
    )

    if not np.isfinite(
        correlation
    ).all():
        raise ValueError(
            "Physiology correlation produced "
            "non-finite values."
        )

    return correlation


def segment_aware_physiology_correlation(
    synchronized_dataframe,
):
    """
    Calculate a video-level physiology correlation
    matrix while respecting POS segment boundaries.

    Pearson correlation is calculated independently
    inside each POS segment. Segment-level correlation
    matrices are then combined using sample-based
    weights.

    A segment must contain at least two synchronized
    samples to contribute.

    Returns
    -------
    correlation_matrix : ndarray
        Weighted 3 x 3 video-level correlation matrix.

    segment_details : list of dict
        QC information describing the segments that
        contributed to the result.
    """

    if synchronized_dataframe.empty:
        raise ValueError(
            "Synchronized dataframe is empty."
        )

    if (
        "segment_id"
        not in synchronized_dataframe.columns
    ):
        # Backward-compatible behavior for data
        # created before segment-aware POS.
        matrix = build_physiology_matrix(
            synchronized_dataframe
        )

        correlation = (
            physiology_correlation_matrix(
                matrix
            )
        )

        details = [
            {
                "segment_id": 1,
                "samples": len(matrix),
                "weight": max(
                    len(matrix) - 1,
                    1,
                ),
            }
        ]

        return (
            correlation,
            details,
        )

    weighted_sum = np.zeros(
        (3, 3),
        dtype=np.float64,
    )

    total_weight = 0.0

    segment_details = []

    grouped = (
        synchronized_dataframe
        .groupby(
            "segment_id",
            sort=True,
        )
    )

    for (
        segment_id,
        segment_dataframe,
    ) in grouped:

        matrix = build_physiology_matrix(
            segment_dataframe
        )

        num_samples = len(
            matrix
        )

        # Pearson correlation requires
        # at least two observations.
        if num_samples < 2:
            continue

        # Constant signals make Pearson
        # correlation undefined.
        standard_deviation = np.std(
            matrix,
            axis=0,
        )

        if np.any(
            standard_deviation < 1e-12
        ):
            continue

        correlation = (
            physiology_correlation_matrix(
                matrix
            )
        )

        # Development/MVP aggregation:
        # longer usable segments contribute
        # proportionally more evidence.
        weight = float(
            num_samples - 1
        )

        weighted_sum += (
            correlation
            * weight
        )

        total_weight += weight

        segment_details.append(
            {
                "segment_id":
                    int(segment_id),

                "samples":
                    int(num_samples),

                "weight":
                    weight,
            }
        )

    if total_weight <= 0:
        raise ValueError(
            "No valid physiology segment "
            "was available for correlation."
        )

    correlation_matrix = (
        weighted_sum
        / total_weight
    )

    # Numerical safety: the diagonal of a
    # correlation matrix represents self-
    # correlation and should be exactly 1.
    np.fill_diagonal(
        correlation_matrix,
        1.0,
    )

    if not np.isfinite(
        correlation_matrix
    ).all():
        raise ValueError(
            "Segment-aware physiology "
            "correlation contains "
            "non-finite values."
        )

    return (
        correlation_matrix,
        segment_details,
    )


def mean_pairwise_correlation(
    correlation_matrix,
):
    """
    Mean of the three unique regional
    pairwise correlations.
    """

    values = np.array(
        [
            correlation_matrix[0, 1],
            correlation_matrix[0, 2],
            correlation_matrix[1, 2],
        ],
        dtype=np.float64,
    )

    return float(
        np.mean(values)
    )


def mean_absolute_pairwise_correlation(
    correlation_matrix,
):
    """
    Mean absolute pairwise correlation.

    Useful because physiological signals can
    exhibit phase/sign differences while still
    sharing temporal structure.
    """

    values = np.array(
        [
            correlation_matrix[0, 1],
            correlation_matrix[0, 2],
            correlation_matrix[1, 2],
        ],
        dtype=np.float64,
    )

    return float(
        np.mean(
            np.abs(values)
        )
    )