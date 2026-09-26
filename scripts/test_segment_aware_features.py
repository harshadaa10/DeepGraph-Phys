from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.graph.dynamic_graph import (
    create_dynamic_graph_data,
)

from src.features.gsp_features import (
    temporal_graph_variation,
)

from src.features.video_features import (
    extract_video_features,
)


RPPG_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_fake_rppg.csv"
)

MOTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sample_fake_motion.csv"
)


def main():

    print(
        "\nDeepGraph-Phys — "
        "Segment-Aware Graph Feature Test"
    )

    print("-" * 65)

    # ==========================================
    # 1. Load saved signals
    # ==========================================

    if not RPPG_PATH.exists():
        raise FileNotFoundError(
            f"rPPG file not found: {RPPG_PATH}"
        )

    if not MOTION_PATH.exists():
        raise FileNotFoundError(
            f"Motion file not found: {MOTION_PATH}"
        )

    rppg = pd.read_csv(
        RPPG_PATH
    )

    motion = pd.read_csv(
        MOTION_PATH
    )

    print(
        "rPPG rows:",
        len(rppg)
    )

    print(
        "Motion rows:",
        len(motion)
    )

    print(
        "rPPG segments:",
        rppg["segment_id"].nunique()
    )

    # ==========================================
    # 2. Create dynamic graph
    # ==========================================

    dynamic_data = (
        create_dynamic_graph_data(
            rppg,
            motion,
        )
    )

    synchronized = dynamic_data[
        "dataframe"
    ]

    motion_matrix = dynamic_data[
        "motion"
    ]

    frames = synchronized[
        "frame"
    ].to_numpy(
        dtype=int
    )

    print(
        "\nSynchronized graph samples:",
        len(synchronized)
    )

    print(
        "Motion matrix shape:",
        motion_matrix.shape
    )

    print(
        "First synchronized frame:",
        int(frames[0])
    )

    print(
        "Last synchronized frame:",
        int(frames[-1])
    )

    # ==========================================
    # 3. Inspect synchronization continuity
    # ==========================================

    frame_differences = np.diff(
        frames
    )

    consecutive_transitions = int(
        np.sum(
            frame_differences == 1
        )
    )

    gap_transitions = int(
        np.sum(
            frame_differences > 1
        )
    )

    largest_difference = (
        int(
            np.max(
                frame_differences
            )
        )
        if len(
            frame_differences
        ) > 0
        else 0
    )

    print(
        "\nSynchronized frame continuity:"
    )

    print(
        "Consecutive transitions:",
        consecutive_transitions
    )

    print(
        "Gap transitions:",
        gap_transitions
    )

    print(
        "Largest frame difference:",
        largest_difference
    )

    # ==========================================
    # 4. Test frame-aware temporal variation
    # ==========================================

    safe_temporal = (
        temporal_graph_variation(
            motion_matrix,
            frame_ids=frames,
        )
    )

    expected_temporal_samples = (
        consecutive_transitions
    )

    print(
        "\nTemporal GSP validation:"
    )

    print(
        "Temporal variation samples:",
        len(safe_temporal)
    )

    print(
        "Expected valid transitions:",
        expected_temporal_samples
    )

    print(
        "Gap transitions excluded:",
        (
            len(safe_temporal)
            == expected_temporal_samples
        )
    )

    # ==========================================
    # 5. Check POS segment boundaries
    # ==========================================

    if (
        "segment_id"
        in synchronized.columns
    ):

        segment_ids = synchronized[
            "segment_id"
        ].to_numpy()

        segment_changes = np.sum(
            np.diff(
                segment_ids
            ) != 0
        )

        print(
            "\nSynchronized POS segments:",
            synchronized[
                "segment_id"
            ].nunique()
        )

        print(
            "Segment-boundary transitions:",
            int(
                segment_changes
            )
        )

    else:

        segment_changes = None

        print(
            "\nWARNING: segment_id was not "
            "preserved during synchronization."
        )

    # ==========================================
    # 6. Extract complete feature vector
    # ==========================================

    print(
        "\nExtracting complete "
        "video feature vector..."
    )

    features = (
        extract_video_features(
            dynamic_data
        )
    )

    feature_values = np.asarray(
        list(
            features.values()
        ),
        dtype=np.float64,
    )

    print(
        "Feature count:",
        len(features)
    )

    print(
        "All features finite:",
        np.isfinite(
            feature_values
        ).all()
    )

    # ==========================================
    # 7. Show temporal feature summaries
    # ==========================================

    print(
        "\nMotion temporal features:"
    )

    for name, value in (
        features.items()
    ):

        if name.startswith(
            "motion_temporal_"
        ):
            print(
                f"{name}: "
                f"{value:.6f}"
            )

    print(
        "\nPhysiology temporal features:"
    )

    for name, value in (
        features.items()
    ):

        if name.startswith(
            "physiology_temporal_"
        ):
            print(
                f"{name}: "
                f"{value:.6f}"
            )

    # ==========================================
    # 8. Final validation
    # ==========================================

    checks = {
        "synchronized samples available":
            len(synchronized) > 0,

        "motion rows match synchronization":
            len(motion_matrix)
            == len(synchronized),

        "frame IDs unique":
            synchronized[
                "frame"
            ].is_unique,

        "frame IDs increasing":
            synchronized[
                "frame"
            ].is_monotonic_increasing,

        "segment ID preserved":
            "segment_id"
            in synchronized.columns,

        "temporal sample count correct":
            len(safe_temporal)
            == expected_temporal_samples,

        "feature count is 60":
            len(features) == 60,

        "all features finite":
            np.isfinite(
                feature_values
            ).all(),
    }

    print(
        "\nValidation checks:"
    )

    for name, result in (
        checks.items()
    ):

        print(
            f"{name}: {result}"
        )

    all_passed = all(
        checks.values()
    )

    print(
        "\nAll validation checks passed:",
        all_passed
    )

    if not all_passed:
        raise ValueError(
            "Segment-aware graph feature "
            "validation failed."
        )

    print(
        "\nSegment-aware graph/GSP "
        "integration completed successfully."
    )


if __name__ == "__main__":
    main()