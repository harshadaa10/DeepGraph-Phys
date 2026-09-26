from pathlib import Path

import numpy as np


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
    max_gap_frames=6,
):
    """
    Complete DeepGraph-Phys feature extraction pipeline.

    VIDEO
        |
        v
    Robust face detection + landmarks
        |
        v
    Regional RGB signals
        |
        v
    Short-gap continuity restoration
        |
        v
    Segment-aware POS rPPG
        +
    Facial motion
        |
        v
    Frame-based synchronization
        |
        v
    Dynamic facial graph
        |
        v
    Gap-safe Graph Signal Processing
        |
        v
    60-dimensional video feature vector

    Notes
    -----
    Quality-control information is returned as
    metadata only. It is NOT included in the
    machine-learning feature vector.
    """

    video_path = Path(
        video_path
    )

    model_path = Path(
        model_path
    )

    # ==========================================
    # 0. Basic validation
    # ==========================================

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            "Face Landmarker model not found: "
            f"{model_path}"
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

    rgb_detected_frames = len(
        rgb_dataframe
    )

    if rgb_detected_frames < 2:
        raise ValueError(
            "Not enough face-detected frames "
            "for RGB/rPPG extraction."
        )

    fps = float(
        rgb_metadata["fps"]
    )

    if (
        not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError(
            "Invalid video FPS."
        )

    # ==========================================
    # 2. Segment-aware POS rPPG
    # ==========================================

    rppg_dataframe = (
        generate_rppg_dataframe(
            rgb_dataframe,
            fps,
            max_gap_frames=max_gap_frames,
        )
    )

    if len(rppg_dataframe) == 0:
        raise ValueError(
            "No valid rPPG samples "
            "were generated."
        )

    rppg_columns = [
        "forehead_rppg",
        "left_cheek_rppg",
        "right_cheek_rppg",
    ]

    if not np.isfinite(
        rppg_dataframe[
            rppg_columns
        ].values
    ).all():
        raise ValueError(
            "Non-finite rPPG values "
            "were generated."
        )

    rppg_samples = len(
        rppg_dataframe
    )

    # ==========================================
    # 3. rPPG continuity / segment QC
    # ==========================================

    if (
        "interpolated"
        in rppg_dataframe.columns
    ):
        interpolated_frames = int(
            rppg_dataframe[
                "interpolated"
            ].sum()
        )
    else:
        interpolated_frames = 0

    interpolation_rate = (
        interpolated_frames
        / rppg_samples
        if rppg_samples > 0
        else 0.0
    )

    if (
        "segment_id"
        in rppg_dataframe.columns
    ):

        rppg_segment_count = int(
            rppg_dataframe[
                "segment_id"
            ].nunique()
        )

        segment_sizes = (
            rppg_dataframe
            .groupby(
                "segment_id"
            )
            .size()
        )

        shortest_rppg_segment = int(
            segment_sizes.min()
        )

        longest_rppg_segment = int(
            segment_sizes.max()
        )

    else:

        rppg_segment_count = 1

        shortest_rppg_segment = (
            rppg_samples
        )

        longest_rppg_segment = (
            rppg_samples
        )

    # ==========================================
    # 4. Facial motion extraction
    # ==========================================

    motion_dataframe, motion_metadata = (
        extract_video_motion(
            video_path,
            model_path,
            max_frames=max_frames,
        )
    )

    motion_samples = len(
        motion_dataframe
    )

    if motion_samples == 0:
        raise ValueError(
            "No valid facial motion samples "
            "were extracted."
        )

    motion_columns = [
        column
        for column
        in motion_dataframe.columns
        if column.endswith(
            "_motion"
        )
    ]

    if not np.isfinite(
        motion_dataframe[
            motion_columns
        ].values
    ).all():
        raise ValueError(
            "Non-finite facial motion "
            "values were generated."
        )

    # ==========================================
    # 5. Dynamic graph synchronization
    # ==========================================

    dynamic_data = (
        create_dynamic_graph_data(
            rppg_dataframe,
            motion_dataframe,
        )
    )

    graph_samples = len(
        dynamic_data[
            "frames"
        ]
    )

    if graph_samples == 0:
        raise ValueError(
            "No synchronized graph samples "
            "were produced."
        )

    synchronized_dataframe = (
        dynamic_data[
            "dataframe"
        ]
    )

    # ==========================================
    # 6. Synchronized interpolation QC
    # ==========================================

    if (
        "interpolated"
        in synchronized_dataframe.columns
    ):

        synchronized_interpolated = int(
            synchronized_dataframe[
                "interpolated"
            ].sum()
        )

    else:

        synchronized_interpolated = 0

    synchronized_interpolation_rate = (
        synchronized_interpolated
        / graph_samples
        if graph_samples > 0
        else 0.0
    )

    # ==========================================
    # 7. Temporal graph continuity QC
    # ==========================================

    graph_frames = (
        synchronized_dataframe[
            "frame"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if len(graph_frames) >= 2:

        frame_differences = np.diff(
            graph_frames
        )

        consecutive_graph_transitions = int(
            np.sum(
                frame_differences == 1
            )
        )

        graph_gap_transitions = int(
            np.sum(
                frame_differences > 1
            )
        )

        largest_graph_frame_difference = int(
            np.max(
                frame_differences
            )
        )

    else:

        consecutive_graph_transitions = 0
        graph_gap_transitions = 0
        largest_graph_frame_difference = 0

    # ==========================================
    # 8. Final 60-dimensional feature vector
    # ==========================================

    features = extract_video_features(
        dynamic_data
    )

    feature_count = len(
        features
    )

    if feature_count != 60:
        raise ValueError(
            "Unexpected feature count. "
            f"Expected 60, received "
            f"{feature_count}."
        )

    feature_values = np.asarray(
        list(
            features.values()
        ),
        dtype=np.float64,
    )

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "Final feature vector contains "
            "NaN or infinite values."
        )

    # ==========================================
    # 9. General video QC
    # ==========================================

    frames_processed = (
        rgb_metadata.get(
            "frames_processed",
            None,
        )
    )

    if (
        frames_processed is not None
        and frames_processed > 0
    ):

        face_detection_rate = (
            rgb_detected_frames
            / frames_processed
        )

    else:

        face_detection_rate = None

    # ==========================================
    # 10. Metadata
    # ==========================================

    metadata = {
        "video_name":
            video_path.name,

        "fps":
            fps,

        "frames_processed":
            frames_processed,

        "rgb_detected_frames":
            rgb_detected_frames,

        "face_detection_rate":
            face_detection_rate,

        "rppg_samples":
            rppg_samples,

        "rppg_segment_count":
            rppg_segment_count,

        "shortest_rppg_segment":
            shortest_rppg_segment,

        "longest_rppg_segment":
            longest_rppg_segment,

        "interpolated_frames":
            interpolated_frames,

        "interpolation_rate":
            interpolation_rate,

        "motion_samples":
            motion_samples,

        "graph_samples":
            graph_samples,

        "consecutive_graph_transitions":
            consecutive_graph_transitions,

        "graph_gap_transitions":
            graph_gap_transitions,

        "largest_graph_frame_difference":
            largest_graph_frame_difference,

        "synchronized_interpolated_frames":
            synchronized_interpolated,

        "synchronized_interpolation_rate":
            synchronized_interpolation_rate,

        "feature_count":
            feature_count,
    }

    return (
        features,
        metadata,
    )