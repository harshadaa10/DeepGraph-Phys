from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.graph.facial_graph import (
    build_facial_graph,
)

from src.features.graph_spectral_profile import (
    normalized_spectral_profile,
    grouped_spectral_profile,
    extract_graph_spectral_profiles,
)


def main():

    print(
        "\nDeepGraph-Phys — "
        "Graph Spectral Profile Test"
    )

    print("=" * 70)

    # ==================================================
    # 1. Test normalized spectral profile directly
    # ==================================================

    coefficients = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    profile = (
        normalized_spectral_profile(
            coefficients
        )
    )

    print(
        "\nDirect profile test:"
    )

    print(
        "Shape:",
        profile.shape,
    )

    print(
        "Profiles:"
    )

    print(
        profile
    )

    print(
        "Row sums:",
        np.sum(
            profile,
            axis=1,
        ),
    )

    # ==================================================
    # 2. Synthetic dynamic facial graph
    # ==================================================

    num_samples = 5

    frames = np.array(
        [10, 11, 12, 15, 16],
        dtype=np.int64,
    )

    motion = np.array(
        [
            [
                0.10,
                0.12,
                0.11,
                0.08,
                0.09,
                0.10,
                0.11,
            ],
            [
                0.11,
                0.13,
                0.12,
                0.09,
                0.10,
                0.11,
                0.12,
            ],
            [
                0.12,
                0.14,
                0.13,
                0.10,
                0.11,
                0.12,
                0.13,
            ],
            [
                0.20,
                0.08,
                0.16,
                0.07,
                0.18,
                0.09,
                0.15,
            ],
            [
                0.21,
                0.09,
                0.17,
                0.08,
                0.19,
                0.10,
                0.16,
            ],
        ],
        dtype=np.float64,
    )

    # ------------------------------------------
    # Minimal synchronized dataframe
    # needed by build_physiology_matrix()
    # ------------------------------------------

    import pandas as pd

    synchronized = pd.DataFrame(
        {
            "frame": frames,
            "forehead_rppg": [
                0.10,
                0.20,
                0.30,
                -0.10,
                -0.20,
            ],
            "left_cheek_rppg": [
                0.11,
                0.19,
                0.31,
                -0.08,
                -0.21,
            ],
            "right_cheek_rppg": [
                0.09,
                0.21,
                0.29,
                -0.12,
                -0.19,
            ],
        }
    )

    dynamic_data = {
        "graph":
            build_facial_graph(),

        "dataframe":
            synchronized,

        "motion":
            motion,

        "frames":
            frames,
    }

    profiles = (
        extract_graph_spectral_profiles(
            dynamic_data
        )
    )

    motion_result = profiles[
        "motion"
    ]

    physiology_result = profiles[
        "physiology"
    ]

    motion_profile = (
        motion_result[
            "profile"
        ]
    )

    physiology_profile = (
        physiology_result[
            "profile"
        ]
    )

    # ==================================================
    # 3. Display graph information
    # ==================================================

    print(
        "\nMotion graph:"
    )

    print(
        "Profile shape:",
        motion_profile.shape,
    )

    print(
        "Eigenvalues:",
        np.round(
            motion_result[
                "eigenvalues"
            ],
            6,
        ),
    )

    print(
        "Profile row sums:",
        np.round(
            np.sum(
                motion_profile,
                axis=1,
            ),
            10,
        ),
    )

    print(
        "\nPhysiology graph:"
    )

    print(
        "Profile shape:",
        physiology_profile.shape,
    )

    print(
        "Eigenvalues:",
        np.round(
            physiology_result[
                "eigenvalues"
            ],
            6,
        ),
    )

    print(
        "Profile row sums:",
        np.round(
            np.sum(
                physiology_profile,
                axis=1,
            ),
            10,
        ),
    )

    # ==================================================
    # 4. Mathematical validation
    # ==================================================

    motion_energy = np.sum(
        motion_result[
            "coefficients"
        ] ** 2,
        axis=1,
    )

    physiology_energy = np.sum(
        physiology_result[
            "coefficients"
        ] ** 2,
        axis=1,
    )

    motion_nonzero = (
        motion_energy > 1e-12
    )

    physiology_nonzero = (
        physiology_energy > 1e-12
    )

    motion_sums = np.sum(
        motion_profile,
        axis=1,
    )

    physiology_sums = np.sum(
        physiology_profile,
        axis=1,
    )

    print(
        "\nValidation checks:"
    )

    print(
        "Direct profile finite:",
        np.isfinite(
            profile
        ).all(),
    )

    print(
        "Direct zero-energy row remains zero:",
        np.allclose(
            profile[2],
            0.0,
        ),
    )

    print(
        "Motion shape correct:",
        motion_profile.shape
        == (5, 7),
    )

    print(
        "Physiology shape correct:",
        physiology_profile.shape
        == (5, 3),
    )

    print(
        "Motion profile finite:",
        np.isfinite(
            motion_profile
        ).all(),
    )

    print(
        "Physiology profile finite:",
        np.isfinite(
            physiology_profile
        ).all(),
    )

    print(
        "Motion non-zero rows sum to 1:",
        np.allclose(
            motion_sums[
                motion_nonzero
            ],
            1.0,
            atol=1e-8,
        ),
    )

    print(
        "Physiology non-zero rows sum to 1:",
        np.allclose(
            physiology_sums[
                physiology_nonzero
            ],
            1.0,
            atol=1e-8,
        ),
    )

    print(
        "Motion Laplacian symmetric:",
        np.allclose(
            motion_result[
                "laplacian"
            ],
            motion_result[
                "laplacian"
            ].T,
        ),
    )

    print(
        "Physiology Laplacian symmetric:",
        np.allclose(
            physiology_result[
                "laplacian"
            ],
            physiology_result[
                "laplacian"
            ].T,
        ),
    )

    print(
        "Motion eigenvalues non-negative:",
        np.all(
            motion_result[
                "eigenvalues"
            ]
            >= -1e-10
        ),
    )

    print(
        "Physiology eigenvalues non-negative:",
        np.all(
            physiology_result[
                "eigenvalues"
            ]
            >= -1e-10
        ),
    )
    motion_grouped = (
        motion_result[
            "grouped_profile"
        ]
    )

    physiology_grouped = (
        physiology_result[
            "grouped_profile"
        ]
    )

    print(
        "Motion eigenvalue groups:",
        [
            group.tolist()
            for group
            in motion_result[
                "eigenvalue_groups"
            ]
        ],
    )

    print(
        "Physiology eigenvalue groups:",
        [
            group.tolist()
            for group
            in physiology_result[
                "eigenvalue_groups"
            ]
        ],
    )

    print(
        "Motion grouped shape correct:",
        motion_grouped.shape
        == (5, 7),
    )

    print(
        "Physiology grouped shape correct:",
        physiology_grouped.shape
        == (5, 2),
    )

    print(
        "Motion grouped profile finite:",
        np.isfinite(
            motion_grouped
        ).all(),
    )

    print(
        "Physiology grouped profile finite:",
        np.isfinite(
            physiology_grouped
        ).all(),
    )

    print(
        "Motion has 7 eigenvalue groups:",
        len(
            motion_result[
                "eigenvalue_groups"
            ]
        ) == 7,
    )

    print(
        "Physiology has 2 eigenvalue groups:",
        len(
            physiology_result[
                "eigenvalue_groups"
            ]
        ) == 2,
    )

    print(
        "Physiology repeated eigenspace grouped:",
        np.array_equal(
            physiology_result[
                "eigenvalue_groups"
            ][1],
            np.array(
                [1, 2],
                dtype=np.int64,
            ),
        ),
    )

    print(
        "Motion grouped rows normalized:",
        np.allclose(
            np.sum(
                motion_grouped,
                axis=1,
            ),
            1.0,
            atol=1e-8,
        ),
    )

    print(
        "Physiology grouped rows normalized:",
        np.allclose(
            np.sum(
                physiology_grouped,
                axis=1,
            ),
            1.0,
            atol=1e-8,
        ),
    )
    
    print(
        "Frame IDs preserved:",
        np.array_equal(
            motion_result[
                "frames"
            ],
            frames,
        )
        and np.array_equal(
            physiology_result[
                "frames"
            ],
            frames,
        ),
    )


if __name__ == "__main__":
    main()