import numpy as np


def graph_smoothness(
    signal,
    laplacian,
):
    """
    Calculate graph smoothness energy:

        S = x^T L x

    Parameters
    ----------
    signal : array-like
        Graph signal with shape (N,)

    laplacian : array-like
        Graph Laplacian with shape (N, N)

    Returns
    -------
    float
        Graph smoothness energy.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    laplacian = np.asarray(
        laplacian,
        dtype=np.float64,
    )

    if signal.ndim != 1:
        raise ValueError(
            "signal must be one-dimensional."
        )

    if laplacian.shape != (
        len(signal),
        len(signal),
    ):
        raise ValueError(
            "Laplacian dimensions do not match "
            "the graph signal."
        )

    energy = (
        signal.T
        @ laplacian
        @ signal
    )

    return float(
        energy
    )


def graph_smoothness_series(
    signal_matrix,
    laplacian,
):
    """
    Calculate graph smoothness for every
    temporal graph state.

    signal_matrix shape:
        (T, N)

    Returns:
        numpy array with shape (T,)
    """

    signal_matrix = np.asarray(
        signal_matrix,
        dtype=np.float64,
    )

    if signal_matrix.ndim != 2:
        raise ValueError(
            "signal_matrix must have shape (T, N)."
        )

    energies = []

    for signal in signal_matrix:

        energy = graph_smoothness(
            signal,
            laplacian,
        )

        energies.append(
            energy
        )

    return np.asarray(
        energies,
        dtype=np.float64,
    )

def graph_fourier_basis(
    laplacian,
):
    """
    Compute the Graph Fourier Transform basis
    from the eigendecomposition of the
    graph Laplacian.

    Returns
    -------
    eigenvalues : ndarray
        Graph frequencies.

    eigenvectors : ndarray
        Graph Fourier basis.
    """

    laplacian = np.asarray(
        laplacian,
        dtype=np.float64,
    )

    if (
        laplacian.ndim != 2
        or laplacian.shape[0]
        != laplacian.shape[1]
    ):
        raise ValueError(
            "Laplacian must be a square matrix."
        )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            laplacian
        )
    )

    return (
        eigenvalues,
        eigenvectors,
    )


def graph_fourier_transform(
    signal,
    eigenvectors,
):
    """
    Transform one graph signal into
    graph-frequency coefficients.

        x_hat = U^T x
    """

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    eigenvectors = np.asarray(
        eigenvectors,
        dtype=np.float64,
    )

    if signal.ndim != 1:
        raise ValueError(
            "signal must be one-dimensional."
        )

    if eigenvectors.shape[0] != len(
        signal
    ):
        raise ValueError(
            "Graph Fourier basis does not "
            "match signal size."
        )

    coefficients = (
        eigenvectors.T
        @ signal
    )

    return coefficients


def graph_fourier_transform_series(
    signal_matrix,
    eigenvectors,
):
    """
    Apply GFT to every temporal graph state.

    Input shape:
        (T, N)

    Output shape:
        (T, N)
    """

    signal_matrix = np.asarray(
        signal_matrix,
        dtype=np.float64,
    )

    if signal_matrix.ndim != 2:
        raise ValueError(
            "signal_matrix must have "
            "shape (T, N)."
        )

    coefficients = (
        signal_matrix
        @ eigenvectors
    )

    return coefficients


def graph_spectral_energy(
    coefficients,
):
    """
    Calculate energy of each graph-frequency
    coefficient.
    """

    coefficients = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    return coefficients ** 2

def spectral_energy_features(
    coefficients,
    eigenvalues,
    low_frequency_count=3,
):
    """
    Extract graph spectral-energy features
    for every temporal graph state.

    Parameters
    ----------
    coefficients : ndarray
        GFT coefficients with shape (T, N).

    eigenvalues : ndarray
        Laplacian eigenvalues with shape (N,).

    low_frequency_count : int
        Number of lowest graph frequencies
        treated as the low-frequency band.

    Returns
    -------
    dict
        Arrays containing spectral features
        for every time step.
    """

    coefficients = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    eigenvalues = np.asarray(
        eigenvalues,
        dtype=np.float64,
    )

    if coefficients.ndim != 2:
        raise ValueError(
            "coefficients must have shape (T, N)."
        )

    if coefficients.shape[1] != len(
        eigenvalues
    ):
        raise ValueError(
            "Number of GFT coefficients does not "
            "match number of eigenvalues."
        )

    if not (
        1 <= low_frequency_count
        < coefficients.shape[1]
    ):
        raise ValueError(
            "low_frequency_count must be between "
            "1 and N-1."
        )

    # Energy at every graph frequency
    energy = coefficients ** 2

    # Total graph spectral energy
    total_energy = np.sum(
        energy,
        axis=1,
    )

    # Lowest graph-frequency components
    low_energy = np.sum(
        energy[
            :,
            :low_frequency_count
        ],
        axis=1,
    )

    # Remaining higher-frequency components
    high_energy = np.sum(
        energy[
            :,
            low_frequency_count:
        ],
        axis=1,
    )

    epsilon = 1e-12

    low_ratio = (
        low_energy
        / (
            total_energy
            + epsilon
        )
    )

    high_ratio = (
        high_energy
        / (
            total_energy
            + epsilon
        )
    )

    # Weighted average graph frequency
    spectral_centroid = (
        np.sum(
            energy
            * eigenvalues[
                np.newaxis,
                :
            ],
            axis=1,
        )
        / (
            total_energy
            + epsilon
        )
    )

    return {
        "total_energy": total_energy,
        "low_energy": low_energy,
        "high_energy": high_energy,
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
        "spectral_centroid": spectral_centroid,
    }

def temporal_graph_variation(
    signal_matrix,
    frame_ids=None,
):
    """
    Measure temporal change in a graph signal.

    Variation is calculated only between samples
    that correspond to consecutive original video
    frames.

    If frame_ids are supplied and two neighboring
    rows are separated by a frame gap, no temporal
    difference is calculated across that gap.

    Parameters
    ----------
    signal_matrix : ndarray
        Shape (T, N), where T is time and N is the
        number of graph nodes.

    frame_ids : array-like or None
        Original video frame IDs corresponding to
        the rows of signal_matrix.

        When provided, temporal variation is valid
        only when:

            current_frame - previous_frame == 1

    Returns
    -------
    ndarray
        Temporal variation values for valid
        consecutive-frame transitions only.

        The first sample and samples immediately
        following temporal gaps are excluded rather
        than being assigned artificial zero values.
    """

    signal_matrix = np.asarray(
        signal_matrix,
        dtype=np.float64,
    )

    if signal_matrix.ndim != 2:
        raise ValueError(
            "signal_matrix must have shape (T, N)."
        )

    num_samples = len(
        signal_matrix
    )

    if num_samples < 2:
        return np.array(
            [],
            dtype=np.float64,
        )

    # ==========================================
    # Calculate neighboring differences
    # ==========================================

    differences = np.diff(
        signal_matrix,
        axis=0,
    )

    variation = np.linalg.norm(
        differences,
        axis=1,
    )

    # ==========================================
    # No frame information supplied
    # ==========================================

    if frame_ids is None:
        return variation

    # ==========================================
    # Validate frame IDs
    # ==========================================

    frame_ids = np.asarray(
        frame_ids,
        dtype=np.int64,
    )

    if frame_ids.ndim != 1:
        raise ValueError(
            "frame_ids must be one-dimensional."
        )

    if len(frame_ids) != num_samples:
        raise ValueError(
            "frame_ids length must match the "
            "number of signal samples."
        )

    # ==========================================
    # Keep only true consecutive transitions
    # ==========================================

    frame_differences = np.diff(
        frame_ids
    )

    consecutive_mask = (
        frame_differences == 1
    )

    return variation[
        consecutive_mask
    ]