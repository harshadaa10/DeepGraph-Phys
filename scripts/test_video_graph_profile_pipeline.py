from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.features.video_pipeline import (
    process_video_to_features,
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


def main():

    print(
        "\nDeepGraph-Phys — "
        "Integrated 60-D + 45-D Pipeline Test"
    )

    print("=" * 70)

    print(
        "\nVideo:",
        VIDEO_PATH.name,
    )

    (
        baseline_features,
        graph_features,
        metadata,
    ) = process_video_to_features(
        VIDEO_PATH,
        MODEL_PATH,
        include_graph_profile=True,
    )

    baseline_values = np.asarray(
        list(
            baseline_features.values()
        ),
        dtype=np.float64,
    )

    graph_values = np.asarray(
        list(
            graph_features.values()
        ),
        dtype=np.float64,
    )

    print(
        "\nOriginal feature count:",
        len(
            baseline_features
        ),
    )

    print(
        "Graph-profile feature count:",
        len(
            graph_features
        ),
    )

    print(
        "Total representations available:",
        len(
            baseline_features
        )
        + len(
            graph_features
        ),
    )

    print(
        "\nGraph-profile first 5:"
    )

    for (
        name,
        value,
    ) in list(
        graph_features.items()
    )[:5]:

        print(
            f"{name}: "
            f"{value:.8f}"
        )

    print(
        "\nGraph-profile last 5:"
    )

    for (
        name,
        value,
    ) in list(
        graph_features.items()
    )[-5:]:

        print(
            f"{name}: "
            f"{value:.8f}"
        )

    print(
        "\nMetadata:"
    )

    print(
        "FPS:",
        metadata.get(
            "fps"
        ),
    )

    print(
        "Graph samples:",
        metadata.get(
            "graph_samples"
        ),
    )

    print(
        "rPPG segments:",
        metadata.get(
            "rppg_segments"
        ),
    )

    print(
        "\nValidation checks:"
    )

    print(
        "Original representation still 60-D:",
        len(
            baseline_features
        ) == 60,
    )

    print(
        "Graph representation exactly 45-D:",
        len(
            graph_features
        ) == 45,
    )

    print(
        "Original features finite:",
        np.isfinite(
            baseline_values
        ).all(),
    )

    print(
        "Graph features finite:",
        np.isfinite(
            graph_values
        ).all(),
    )

    print(
        "Graph features bounded [0, 1]:",
        np.all(
            graph_values >= -1e-12
        )
        and np.all(
            graph_values <= 1.0 + 1e-12
        ),
    )

    baseline_names = set(
        baseline_features.keys()
    )

    graph_names = set(
        graph_features.keys()
    )

    print(
        "No feature-name collision:",
        baseline_names.isdisjoint(
            graph_names
        ),
    )

    print(
        "Graph naming correct:",
        all(
            name.startswith(
                (
                    "motion_gsp_",
                    "physiology_gsp_",
                )
            )
            for name
            in graph_features
        ),
    )

    print(
        "\nPipeline integration successful."
    )


if __name__ == "__main__":
    main()