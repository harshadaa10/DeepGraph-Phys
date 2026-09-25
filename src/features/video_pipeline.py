from pathlib import Path

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

from src.features.video_features import (
    extract_video_features,
)


def process_video_to_features(
    video_path,
    model_path,
    max_frames=None,
):
    """
    Complete DeepGraph-Phys feature pipeline.

    VIDEO
        ↓
    RGB facial signals
        ↓
    POS rPPG
        +
    Facial motion
        ↓
    Dynamic graphs
        ↓
    GSP features
        ↓
    Video-level feature vector
    """

    video_path = Path(
        video_path
    )

    # ==========================================
    # 1. Extract regional RGB signals
    # ==========================================

    rgb_dataframe, rgb_metadata = (
        extract_video_rgb_signals(
            video_path,
            model_path,
            max_frames=max_frames,
        )
    )

    if len(rgb_dataframe) < 2:
        raise ValueError(
            "Not enough face-detected frames "
            "for RGB/rPPG extraction."
        )

    fps = rgb_metadata[
        "fps"
    ]

    if fps <= 0:
        raise ValueError(
            "Invalid video FPS."
        )

    # ==========================================
    # 2. POS rPPG
    # ==========================================

    rppg_dataframe = (
        generate_rppg_dataframe(
            rgb_dataframe,
            fps,
        )
    )

    # ==========================================
    # 3. Facial motion
    # ==========================================

    motion_dataframe, motion_metadata = (
        extract_video_motion(
            video_path,
            model_path,
            max_frames=max_frames,
        )
    )

    if len(motion_dataframe) == 0:
        raise ValueError(
            "No valid facial motion samples "
            "were extracted."
        )

    # ==========================================
    # 4. Synchronize into graph signals
    # ==========================================

    dynamic_data = (
        create_dynamic_graph_data(
            rppg_dataframe,
            motion_dataframe,
        )
    )

    if len(
        dynamic_data["frames"]
    ) == 0:

        raise ValueError(
            "No synchronized graph samples "
            "were produced."
        )

    # ==========================================
    # 5. Final 60-dimensional feature vector
    # ==========================================

    features = extract_video_features(
        dynamic_data
    )

    metadata = {
        "video_name":
            video_path.name,

        "fps":
            fps,

        "rgb_frames":
            len(rgb_dataframe),

        "motion_samples":
            len(motion_dataframe),

        "graph_samples":
            len(
                dynamic_data["frames"]
            ),

        "feature_count":
            len(features),
    }

    return (
        features,
        metadata,
    )