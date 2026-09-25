import numpy as np


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

    columns = [
        "forehead_rppg",
        "left_cheek_rppg",
        "right_cheek_rppg",
    ]

    matrix = synchronized_dataframe[
        columns
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

    correlation = np.corrcoef(
        physiology_matrix,
        rowvar=False,
    )

    return correlation


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