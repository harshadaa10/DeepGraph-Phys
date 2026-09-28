from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.physiology.video_rgb_signals import (
    extract_video_rgb_signals,
)

from src.physiology.rppg_pipeline import (
    generate_rppg_dataframe,
)

from src.motion.video_motion import (
    extract_video_motion,
)

from src.graph.dynamic_graph import (
    create_dynamic_graph_data,
)

from src.features.graph_spectral_profile import (
    extract_graph_spectral_profiles,
)


VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "FaceForensics++"
    / "original_sequences"
    / "youtube"
    / "c23"
    / "videos"
    / "033.mp4"
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)


def summarize_profile(
    profile,
    name,
):

    print(
        f"\n{name}"
    )

    print("-" * 70)

    print(
        "Shape:",
        profile.shape,
    )

    print(
        "Finite:",
        np.isfinite(
            profile
        ).all(),
    )

    row_sums = np.sum(
        profile,
        axis=1,
    )

    print(
        "Row-sum min:",
        float(
            np.min(row_sums)
        ),
    )

    print(
        "Row-sum max:",
        float(
            np.max(row_sums)
        ),
    )

    mean_profile = np.mean(
        profile,
        axis=0,
    )

    std_profile = np.std(
        profile,
        axis=0,
    )

    print(
        "Mean spectral profile:"
    )

    print(
        np.round(
            mean_profile,
            6,
        )
    )

    print(
        "Std spectral profile:"
    )

    print(
        np.round(
            std_profile,
            6,
        )
    )

    return (
        mean_profile,
        std_profile,
        row_sums,
    )


def main():

    print(
        "\nDeepGraph-Phys — "
        "Real FF++ Graph Spectral Profile Test"
    )

    print("=" * 70)

    print(
        "\nVideo:",
        VIDEO_PATH.name,
    )

    # ==================================================
    # 1. RGB extraction
    # ==================================================

    print(
        "\n[1/5] Extracting RGB signals..."
    )

    (
        rgb_dataframe,
        rgb_metadata,
    ) = extract_video_rgb_signals(
        VIDEO_PATH,
        MODEL_PATH,
    )

    fps = float(
        rgb_metadata[
            "fps"
        ]
    )

    print(
        "FPS:",
        fps,
    )

    print(
        "RGB detected frames:",
        len(
            rgb_dataframe
        ),
    )

    # ==================================================
    # 2. rPPG
    # ==================================================

    print(
        "\n[2/5] Generating segment-aware rPPG..."
    )

    rppg_dataframe = (
        generate_rppg_dataframe(
            rgb_dataframe,
            fps,
            max_gap_frames=6,
        )
    )

    print(
        "rPPG samples:",
        len(
            rppg_dataframe
        ),
    )

    if (
        "segment_id"
        in rppg_dataframe.columns
    ):
        print(
            "rPPG segments:",
            rppg_dataframe[
                "segment_id"
            ].nunique(),
        )

    # ==================================================
    # 3. Motion
    # ==================================================

    print(
        "\n[3/5] Extracting facial motion..."
    )

    (
        motion_dataframe,
        motion_metadata,
    ) = extract_video_motion(
        VIDEO_PATH,
        MODEL_PATH,
    )

    print(
        "Motion samples:",
        len(
            motion_dataframe
        ),
    )

    # ==================================================
    # 4. Dynamic graph
    # ==================================================

    print(
        "\n[4/5] Creating synchronized "
        "dynamic graph..."
    )

    dynamic_data = (
        create_dynamic_graph_data(
            rppg_dataframe,
            motion_dataframe,
        )
    )

    frames = np.asarray(
        dynamic_data[
            "frames"
        ],
        dtype=np.int64,
    )

    print(
        "Synchronized graph samples:",
        len(
            frames
        ),
    )

    if len(frames) >= 2:

        frame_differences = (
            np.diff(
                frames
            )
        )

        print(
            "Consecutive transitions:",
            int(
                np.sum(
                    frame_differences == 1
                )
            ),
        )

        print(
            "Gap transitions:",
            int(
                np.sum(
                    frame_differences > 1
                )
            ),
        )

        print(
            "Largest frame difference:",
            int(
                np.max(
                    frame_differences
                )
            ),
        )

    # ==================================================
    # 5. Graph spectral profiles
    # ==================================================

    print(
        "\n[5/5] Extracting graph "
        "spectral profiles..."
    )

    profiles = (
        extract_graph_spectral_profiles(
            dynamic_data
        )
    )

    motion_profile = (
        profiles[
            "motion"
        ][
            "profile"
        ]
    )

    physiology_profile = (
        profiles[
            "physiology"
        ][
            "profile"
        ]
    )

    (
        motion_mean,
        motion_std,
        motion_row_sums,
    ) = summarize_profile(
        motion_profile,
        "MOTION GRAPH PROFILE",
    )

    (
        physiology_mean,
        physiology_std,
        physiology_row_sums,
    ) = summarize_profile(
        physiology_profile,
        "PHYSIOLOGY GRAPH PROFILE",
    )

    # ==================================================
    # Validation
    # ==================================================

    print(
        "\nValidation checks:"
    )

    print(
        "Motion shape correct:",
        motion_profile.shape
        == (
            len(frames),
            7,
        ),
    )

    print(
        "Physiology shape correct:",
        physiology_profile.shape
        == (
            len(frames),
            3,
        ),
    )

    print(
        "Motion finite:",
        np.isfinite(
            motion_profile
        ).all(),
    )

    print(
        "Physiology finite:",
        np.isfinite(
            physiology_profile
        ).all(),
    )

    motion_energy = np.sum(
        profiles[
            "motion"
        ][
            "coefficients"
        ] ** 2,
        axis=1,
    )

    physiology_energy = np.sum(
        profiles[
            "physiology"
        ][
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

    print(
        "Motion non-zero rows normalized:",
        np.allclose(
            motion_row_sums[
                motion_nonzero
            ],
            1.0,
            atol=1e-8,
        ),
    )

    print(
        "Physiology non-zero rows normalized:",
        np.allclose(
            physiology_row_sums[
                physiology_nonzero
            ],
            1.0,
            atol=1e-8,
        ),
    )

    print(
        "Motion mean profile normalized:",
        np.isclose(
            np.sum(
                motion_mean
            ),
            1.0,
            atol=1e-8,
        ),
    )

    print(
        "Physiology mean profile normalized:",
        np.isclose(
            np.sum(
                physiology_mean
            ),
            1.0,
            atol=1e-8,
        ),
    )

    print(
        "\nNOTE:"
    )

    print(
        "This test validates the graph-frequency "
        "representation only."
    )

    print(
        "No classifier or anomaly threshold "
        "is being evaluated here."
    )


if __name__ == "__main__":
    main()