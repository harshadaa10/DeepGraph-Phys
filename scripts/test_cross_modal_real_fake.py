
"""
Phase 13C
Test Cross-Modal Motion–Physiology Graph Features
on the real and fake demonstration videos.

Experimental only.
Does NOT modify the locked feature dataset.
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
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

from src.features.cross_modal_graph import (
    extract_cross_modal_graph_features,
    extract_cross_modal_region_table,
)


REAL_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "real"
    / "sample_real.mp4"
)

FAKE_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fake"
    / "sample_fake.mp4"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)


def analyze_video(video_path):
    print()
    print("=" * 70)
    print(f"VIDEO: {video_path.name}")
    print("=" * 70)

    print("\n1. Extracting regional RGB signals...")

    rgb_dataframe, rgb_metadata = extract_video_rgb_signals(
        video_path,
        MODEL_PATH,
    )

    print(
        f"RGB samples: {len(rgb_dataframe)}"
    )

    fps = float(
        rgb_metadata["fps"]
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print("\n2. Generating segment-aware rPPG...")

    rppg_dataframe = generate_rppg_dataframe(
        rgb_dataframe,
        fps,
        max_gap_frames=6,
    )

    print(
        f"rPPG samples: {len(rppg_dataframe)}"
    )

    print("\n3. Extracting facial motion...")

    motion_dataframe, motion_metadata = extract_video_motion(
        video_path,
        MODEL_PATH,
    )

    print(
        f"Motion samples: {len(motion_dataframe)}"
    )

    print("\n4. Synchronizing motion + physiology...")

    dynamic_data = create_dynamic_graph_data(
        rppg_dataframe,
        motion_dataframe,
    )

    dataframe = dynamic_data[
        "dataframe"
    ]

    print(
        f"Synchronized graph samples: {len(dataframe)}"
    )

    print("\nAvailable synchronized columns:")

    print(
        dataframe.columns.tolist()
    )

    print("\n5. Calculating CMPC features...")

    features = extract_cross_modal_graph_features(
        dataframe
    )

    region_table = extract_cross_modal_region_table(
        dataframe
    )

    print("\nRegional cross-modal measurements:")

    print(
        region_table.to_string(
            index=False
        )
    )

    print("\nVideo-level CMPC features:")

    for name, value in features.items():
        print(
            f"{name}: {value:.6f}"
        )

    return features


def main():

    if not REAL_VIDEO.exists():
        raise FileNotFoundError(
            f"Real video not found: {REAL_VIDEO}"
        )

    if not FAKE_VIDEO.exists():
        raise FileNotFoundError(
            f"Fake video not found: {FAKE_VIDEO}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face landmark model not found: {MODEL_PATH}"
        )

    print("=" * 70)
    print(
        "PHASE 13C — CROSS-MODAL REAL/FAKE SANITY TEST"
    )
    print("=" * 70)

    real_features = analyze_video(
        REAL_VIDEO
    )

    fake_features = analyze_video(
        FAKE_VIDEO
    )

    print()
    print("=" * 70)
    print("REAL vs FAKE COMPARISON")
    print("=" * 70)

    comparison_rows = []

    for feature_name in real_features:

        comparison_rows.append(
            {
                "feature": feature_name,
                "real": real_features[
                    feature_name
                ],
                "fake": fake_features[
                    feature_name
                ],
                "absolute_difference": abs(
                    real_features[
                        feature_name
                    ]
                    -
                    fake_features[
                        feature_name
                    ]
                ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison = comparison.sort_values(
        "absolute_difference",
        ascending=False,
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print()
    print("=" * 70)
    print("SANITY TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

