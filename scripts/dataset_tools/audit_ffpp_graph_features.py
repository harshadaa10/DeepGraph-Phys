from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT)
)


BASELINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ffpp_features.csv"
)

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ffpp_graph_features.csv"
)


IDENTITY_COLUMNS = [
    "video_name",
    "video_path",
    "label",
    "dataset",
    "manipulation",
    "source_id",
    "target_id",
    "group_id",
    "split",
]


QC_COLUMNS = [
    "fps",
    "frames_processed",
    "rgb_detected_frames",
    "face_detection_rate",
    "rppg_samples",
    "rppg_segment_count",
    "shortest_rppg_segment",
    "longest_rppg_segment",
    "interpolated_frames",
    "interpolation_rate",
    "motion_samples",
    "graph_samples",
    "consecutive_graph_transitions",
    "graph_gap_transitions",
    "largest_graph_frame_difference",
    "synchronized_interpolated_frames",
    "synchronized_interpolation_rate",
]


def main():

    print(
        "\nDeepGraph-Phys — "
        "Final FF++ Graph Dataset Audit"
    )

    print("=" * 72)

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline dataset not found: "
            f"{BASELINE_PATH}"
        )

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Graph dataset not found: "
            f"{GRAPH_PATH}"
        )

    baseline = pd.read_csv(
        BASELINE_PATH
    )

    graph = pd.read_csv(
        GRAPH_PATH
    )

    graph_feature_columns = [
        column
        for column in graph.columns
        if column.startswith(
            (
                "motion_gsp_",
                "physiology_gsp_",
            )
        )
    ]

    motion_columns = [
        column
        for column
        in graph_feature_columns
        if column.startswith(
            "motion_gsp_"
        )
    ]

    physiology_columns = [
        column
        for column
        in graph_feature_columns
        if column.startswith(
            "physiology_gsp_"
        )
    ]

    print(
        "\nDataset shapes:"
    )

    print(
        "Original 60-D dataset:",
        baseline.shape,
    )

    print(
        "Graph 45-D dataset:",
        graph.shape,
    )

    print(
        "\nGraph representation:"
    )

    print(
        "Total graph features:",
        len(
            graph_feature_columns
        ),
    )

    print(
        "Motion features:",
        len(
            motion_columns
        ),
    )

    print(
        "Physiology features:",
        len(
            physiology_columns
        ),
    )

    # ==========================================
    # Dataset distribution
    # ==========================================

    print(
        "\nLabel distribution:"
    )

    print(
        graph[
            "label"
        ].value_counts().to_string()
    )

    print(
        "\nSplit distribution:"
    )

    print(
        graph[
            "split"
        ].value_counts().to_string()
    )

    print(
        "\nLabel x split:"
    )

    print(
        pd.crosstab(
            graph["split"],
            graph["label"],
        ).to_string()
    )

    print(
        "\nUnique groups by split:"
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        groups = sorted(
            graph.loc[
                graph["split"] == split,
                "group_id",
            ]
            .unique()
            .tolist()
        )

        print(
            f"{split}:",
            groups,
        )

    # ==========================================
    # Numerical validation
    # ==========================================

    graph_values = (
        graph[
            graph_feature_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    print(
        "\nNumerical checks:"
    )

    print(
        "Duplicate graph videos:",
        int(
            graph[
                "video_name"
            ].duplicated().sum()
        ),
    )

    print(
        "NaNs:",
        int(
            np.isnan(
                graph_values
            ).sum()
        ),
    )

    print(
        "Infinities:",
        int(
            np.isinf(
                graph_values
            ).sum()
        ),
    )

    print(
        "All finite:",
        bool(
            np.isfinite(
                graph_values
            ).all()
        ),
    )

    print(
        "Global minimum:",
        float(
            np.min(
                graph_values
            )
        ),
    )

    print(
        "Global maximum:",
        float(
            np.max(
                graph_values
            )
        ),
    )

    print(
        "All bounded [0, 1]:",
        bool(
            np.all(
                graph_values
                >= -1e-12
            )
            and np.all(
                graph_values
                <= 1.0 + 1e-12
            )
        ),
    )

    print(
        "All graph feature counts = 45:",
        bool(
            (
                graph[
                    "graph_feature_count"
                ] == 45
            ).all()
        ),
    )

    # ==========================================
    # Compare provenance with old dataset
    # ==========================================

    baseline_identity = (
        baseline[
            IDENTITY_COLUMNS
        ]
        .sort_values(
            "video_name"
        )
        .reset_index(
            drop=True
        )
    )

    graph_identity = (
        graph[
            IDENTITY_COLUMNS
        ]
        .sort_values(
            "video_name"
        )
        .reset_index(
            drop=True
        )
    )

    identity_match = (
        baseline_identity.equals(
            graph_identity
        )
    )

    print(
        "\nCross-dataset consistency:"
    )

    print(
        "Same number of videos:",
        len(
            baseline
        )
        == len(
            graph
        ),
    )

    print(
        "Exact identity/provenance match:",
        identity_match,
    )

    baseline_names = set(
        baseline[
            "video_name"
        ].astype(str)
    )

    graph_names = set(
        graph[
            "video_name"
        ].astype(str)
    )

    print(
        "Exact video set match:",
        baseline_names
        == graph_names,
    )

    # ==========================================
    # QC comparison
    # ==========================================

    baseline_qc = (
        baseline[
            [
                "video_name",
                *QC_COLUMNS,
            ]
        ]
        .sort_values(
            "video_name"
        )
        .reset_index(
            drop=True
        )
    )

    graph_qc = (
        graph[
            [
                "video_name",
                *QC_COLUMNS,
            ]
        ]
        .sort_values(
            "video_name"
        )
        .reset_index(
            drop=True
        )
    )

    qc_match = True
    qc_mismatches = []

    for column in QC_COLUMNS:

        baseline_values = (
            baseline_qc[
                column
            ]
            .to_numpy()
        )

        graph_qc_values = (
            graph_qc[
                column
            ]
            .to_numpy()
        )

        if np.issubdtype(
            baseline_qc[
                column
            ].dtype,
            np.number,
        ) and np.issubdtype(
            graph_qc[
                column
            ].dtype,
            np.number,
        ):

            matches = np.allclose(
                baseline_values,
                graph_qc_values,
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            )

        else:

            matches = (
                baseline_qc[
                    column
                ]
                .fillna(
                    "<NA>"
                )
                .astype(str)
                .equals(
                    graph_qc[
                        column
                    ]
                    .fillna(
                        "<NA>"
                    )
                    .astype(str)
                )
            )

        if not matches:
            qc_match = False
            qc_mismatches.append(
                column
            )

    print(
        "QC metadata matches original run:",
        qc_match,
    )

    print(
        "QC mismatched columns:",
        qc_mismatches,
    )

    # ==========================================
    # Group leakage check
    # ==========================================

    train_groups = set(
        graph.loc[
            graph["split"] == "train",
            "group_id",
        ]
    )

    validation_groups = set(
        graph.loc[
            graph["split"] == "validation",
            "group_id",
        ]
    )

    test_groups = set(
        graph.loc[
            graph["split"] == "test",
            "group_id",
        ]
    )

    no_group_overlap = (
        train_groups.isdisjoint(
            validation_groups
        )
        and train_groups.isdisjoint(
            test_groups
        )
        and validation_groups.isdisjoint(
            test_groups
        )
    )

    print(
        "\nLeakage checks:"
    )

    print(
        "No group overlap between splits:",
        no_group_overlap,
    )

    # ==========================================
    # Final status
    # ==========================================

    required_checks = [
        graph.shape[0] == 40,
        graph.shape[1] == 72,
        len(
            graph_feature_columns
        ) == 45,
        len(
            motion_columns
        ) == 35,
        len(
            physiology_columns
        ) == 10,
        not graph[
            "video_name"
        ].duplicated().any(),
        np.isfinite(
            graph_values
        ).all(),
        np.all(
            graph_values >= -1e-12
        ),
        np.all(
            graph_values <= 1.0 + 1e-12
        ),
        (
            graph[
                "graph_feature_count"
            ] == 45
        ).all(),
        len(
            baseline
        ) == len(
            graph
        ),
        identity_match,
        baseline_names
        == graph_names,
        no_group_overlap,
    ]

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CORE AUDIT PASSED:",
        bool(
            all(
                required_checks
            )
        ),
    )

    if not qc_match:

        print(
            "\nNOTE:"
        )

        print(
            "QC differs between extraction runs."
        )

        print(
            "This does not automatically mean "
            "the graph dataset is invalid."
        )

        print(
            "We will inspect the mismatched "
            "columns before modeling."
        )

    print(
        "\nAudit completed."
    )


if __name__ == "__main__":
    main()