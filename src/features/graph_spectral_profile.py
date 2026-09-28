import numpy as np

from src.graph.facial_graph import (
    get_adjacency_matrix,
    get_laplacian_matrix,
)

from src.graph.physiology_graph import (
    build_physiology_graph,
    get_physiology_adjacency,
    get_physiology_laplacian,
)

from src.features.gsp_features import (
    graph_fourier_basis,
    graph_fourier_transform_series,
)

from src.features.physiology_features import (
    build_physiology_matrix,
)


def normalized_spectral_profile(
    coefficients,
    epsilon=1e-12,
):
    """
    Convert GFT coefficients into a normalized
    graph spectral-energy profile.

    For each graph state:

        E_k = x_hat_k^2

        P_k = E_k / (sum_j E_j + epsilon)

    Parameters
    ----------
    coefficients : ndarray
        Shape (T, N).

    epsilon : float
        Numerical-stability constant.

    Returns
    -------
    ndarray
        Normalized spectral profiles with
        shape (T, N).
    """

    coefficients = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    if coefficients.ndim != 2:
        raise ValueError(
            "coefficients must have shape (T, N)."
        )

    if coefficients.shape[0] == 0:
        raise ValueError(
            "coefficients contain no samples."
        )

    if not np.isfinite(
        coefficients
    ).all():
        raise ValueError(
            "coefficients contain NaN or "
            "infinite values."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    energy = (
        coefficients ** 2
    )

    total_energy = np.sum(
        energy,
        axis=1,
        keepdims=True,
    )

    profiles = (
        energy
        / (
            total_energy
            + epsilon
        )
    )

    return profiles

def grouped_spectral_profile(
    coefficients,
    eigenvalues,
    tolerance=1e-8,
    epsilon=1e-12,
):
    """
    Group graph spectral energy by distinct
    Laplacian eigenvalues.

    This is important when the graph Laplacian
    contains repeated eigenvalues. Individual
    eigenvectors inside a repeated eigenspace are
    not unique, but the total energy contained in
    that eigenspace is invariant to orthonormal
    rotations of its basis.

    Parameters
    ----------
    coefficients : ndarray
        GFT coefficients with shape (T, N).

    eigenvalues : ndarray
        Laplacian eigenvalues with shape (N,).

    tolerance : float
        Maximum absolute eigenvalue difference
        for two frequencies to be treated as
        belonging to the same eigenspace.

    epsilon : float
        Numerical-stability constant.

    Returns
    -------
    grouped_profile : ndarray
        Normalized spectral-energy profile with
        shape (T, G), where G is the number of
        distinct eigenvalue groups.

    grouped_eigenvalues : ndarray
        Representative eigenvalue for each group.

    groups : list
        List of index arrays showing which
        original GFT components belong to each
        eigenvalue group.
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

    if eigenvalues.ndim != 1:
        raise ValueError(
            "eigenvalues must be one-dimensional."
        )

    if coefficients.shape[1] != len(
        eigenvalues
    ):
        raise ValueError(
            "Number of GFT coefficients does not "
            "match number of eigenvalues."
        )

    if coefficients.shape[0] == 0:
        raise ValueError(
            "coefficients contain no samples."
        )

    if not np.isfinite(
        coefficients
    ).all():
        raise ValueError(
            "coefficients contain NaN or "
            "infinite values."
        )

    if not np.isfinite(
        eigenvalues
    ).all():
        raise ValueError(
            "eigenvalues contain NaN or "
            "infinite values."
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    # --------------------------------------------------
    # Build groups of equal / near-equal eigenvalues
    # --------------------------------------------------

    groups = []

    grouped_eigenvalues = []

    for index, eigenvalue in enumerate(
        eigenvalues
    ):

        if len(groups) == 0:

            groups.append(
                [index]
            )

            grouped_eigenvalues.append(
                float(
                    eigenvalue
                )
            )

            continue

        previous_eigenvalue = (
            grouped_eigenvalues[-1]
        )

        if abs(
            eigenvalue
            - previous_eigenvalue
        ) <= tolerance:

            groups[-1].append(
                index
            )

        else:

            groups.append(
                [index]
            )

            grouped_eigenvalues.append(
                float(
                    eigenvalue
                )
            )

    # --------------------------------------------------
    # Energy in original GFT coordinates
    # --------------------------------------------------

    energy = (
        coefficients ** 2
    )

    # --------------------------------------------------
    # Sum energy inside each eigenspace
    # --------------------------------------------------

    grouped_energy = np.column_stack(
        [
            np.sum(
                energy[
                    :,
                    group
                ],
                axis=1,
            )
            for group in groups
        ]
    )

    total_energy = np.sum(
        grouped_energy,
        axis=1,
        keepdims=True,
    )

    grouped_profile = (
        grouped_energy
        / (
            total_energy
            + epsilon
        )
    )

    return (
        grouped_profile,
        np.asarray(
            grouped_eigenvalues,
            dtype=np.float64,
        ),
        [
            np.asarray(
                group,
                dtype=np.int64,
            )
            for group in groups
        ],
    )
    
def extract_motion_spectral_profile(
    dynamic_data,
):
    """
    Extract normalized graph spectral profiles
    from the 7-node facial-motion graph.

    Returns
    -------
    dict
        profile:
            Shape (T, 7)

        coefficients:
            Shape (T, 7)

        eigenvalues:
            Shape (7,)

        eigenvectors:
            Shape (7, 7)

        frames:
            Original synchronized frame IDs.
    """

    graph = dynamic_data[
        "graph"
    ]

    motion = np.asarray(
        dynamic_data[
            "motion"
        ],
        dtype=np.float64,
    )

    frames = np.asarray(
        dynamic_data[
            "frames"
        ],
        dtype=np.int64,
    )

    if motion.ndim != 2:
        raise ValueError(
            "Motion matrix must be two-dimensional."
        )

    if motion.shape[1] != 7:
        raise ValueError(
            "Motion matrix must contain exactly "
            "7 facial graph nodes."
        )

    if len(frames) != len(
        motion
    ):
        raise ValueError(
            "Frame count does not match "
            "motion samples."
        )

    if not np.isfinite(
        motion
    ).all():
        raise ValueError(
            "Motion matrix contains NaN or "
            "infinite values."
        )

    adjacency = (
        get_adjacency_matrix(
            graph
        )
    )

    laplacian = (
        get_laplacian_matrix(
            adjacency
        )
    )

    (
        eigenvalues,
        eigenvectors,
    ) = graph_fourier_basis(
        laplacian
    )

    coefficients = (
        graph_fourier_transform_series(
            motion,
            eigenvectors,
        )
    )

    profile = (
        normalized_spectral_profile(
            coefficients
        )
    )
    (
    grouped_profile,
    grouped_eigenvalues,
    eigenvalue_groups,
) = grouped_spectral_profile(
    coefficients,
    eigenvalues,
)

    return {
    "profile": profile,
    "grouped_profile": grouped_profile,
    "coefficients": coefficients,
    "eigenvalues": eigenvalues,
    "grouped_eigenvalues": grouped_eigenvalues,
    "eigenvalue_groups": eigenvalue_groups,
    "eigenvectors": eigenvectors,
    "laplacian": laplacian,
    "frames": frames,
}


def extract_physiology_spectral_profile(
    dynamic_data,
):
    """
    Extract normalized graph spectral profiles
    from the 3-node physiology graph.

    Only the three regions with valid rPPG are
    used:

        forehead
        left_cheek
        right_cheek

    Returns
    -------
    dict
        profile:
            Shape (T, 3)

        coefficients:
            Shape (T, 3)

        eigenvalues:
            Shape (3,)

        eigenvectors:
            Shape (3, 3)

        frames:
            Original synchronized frame IDs.
    """

    synchronized = dynamic_data[
        "dataframe"
    ]

    frames = np.asarray(
        dynamic_data[
            "frames"
        ],
        dtype=np.int64,
    )

    physiology = (
        build_physiology_matrix(
            synchronized
        )
    )

    physiology = np.asarray(
        physiology,
        dtype=np.float64,
    )

    if physiology.ndim != 2:
        raise ValueError(
            "Physiology matrix must be "
            "two-dimensional."
        )

    if physiology.shape[1] != 3:
        raise ValueError(
            "Physiology matrix must contain "
            "exactly 3 rPPG graph nodes."
        )

    if len(frames) != len(
        physiology
    ):
        raise ValueError(
            "Frame count does not match "
            "physiology samples."
        )

    if not np.isfinite(
        physiology
    ).all():
        raise ValueError(
            "Physiology matrix contains NaN "
            "or infinite values."
        )

    graph = (
        build_physiology_graph()
    )

    adjacency = (
        get_physiology_adjacency(
            graph
        )
    )

    laplacian = (
        get_physiology_laplacian(
            adjacency
        )
    )

    (
        eigenvalues,
        eigenvectors,
    ) = graph_fourier_basis(
        laplacian
    )

    coefficients = (
        graph_fourier_transform_series(
            physiology,
            eigenvectors,
        )
    )

    profile = (
        normalized_spectral_profile(
            coefficients
        )
    )
    (
    grouped_profile,
    grouped_eigenvalues,
    eigenvalue_groups,
) = grouped_spectral_profile(
    coefficients,
    eigenvalues,
)

    return {
    "profile": profile,
    "grouped_profile": grouped_profile,
    "coefficients": coefficients,
    "eigenvalues": eigenvalues,
    "grouped_eigenvalues": grouped_eigenvalues,
    "eigenvalue_groups": eigenvalue_groups,
    "eigenvectors": eigenvectors,
    "laplacian": laplacian,
    "frames": frames,
}


def extract_graph_spectral_profiles(
    dynamic_data,
):
    """
    Extract both DeepGraph-Phys graph spectral
    representations.

    Motion:
        T x 7

    Physiology:
        T x 3
    """

    motion = (
        extract_motion_spectral_profile(
            dynamic_data
        )
    )

    physiology = (
        extract_physiology_spectral_profile(
            dynamic_data
        )
    )

    return {
        "motion": motion,
        "physiology": physiology,
    }