import numpy as np
import pandas as pd

from src.physiology.pos import (
    pos_rppg,
)

from src.physiology.signal_continuity import (
    RGB_COLUMNS,
    restore_regular_rgb_grid,
    find_valid_segments,
)


RPPG_REGIONS = [
    "forehead",
    "left_cheek",
    "right_cheek",
]


def generate_rppg_dataframe(
    rgb_dataframe,
    fps,
    max_gap_frames=6,
    window_seconds=1.6,
):
    """
    Generate regional POS-rPPG signals using
    segment-aware temporal processing.

    Short RGB gaps may be interpolated before
    segmentation. Long gaps remain invalid and
    divide the video into independent continuous
    segments.

    POS is applied separately to every segment
    that is at least one POS window long.

    Parameters
    ----------
    rgb_dataframe : pandas.DataFrame
        RGB measurements extracted from detected
        facial regions.

    fps : float
        Original video frame rate.

    max_gap_frames : int
        Maximum complete missing run that may be
        linearly interpolated.

    window_seconds : float
        POS sliding-window duration.

    Returns
    -------
    pandas.DataFrame
        rPPG values for valid continuous segments.
        Original video frame IDs are preserved.
    """

    # ==========================================
    # 1. Input validation
    # ==========================================

    if rgb_dataframe.empty:
        raise ValueError(
            "RGB dataframe is empty."
        )

    if fps <= 0:
        raise ValueError(
            "FPS must be greater than zero."
        )

    if window_seconds <= 0:
        raise ValueError(
            "window_seconds must be greater "
            "than zero."
        )

    # ==========================================
    # 2. Restore regular frame grid
    # ==========================================

    continuous = (
        restore_regular_rgb_grid(
            rgb_dataframe,
            fps=fps,
            max_gap_frames=max_gap_frames,
        )
    )

    # ==========================================
    # 3. Determine minimum POS segment length
    # ==========================================

    min_segment_frames = int(
        round(
            window_seconds
            * fps
        )
    )

    if min_segment_frames < 2:
        raise ValueError(
            "POS window is too short."
        )

    # ==========================================
    # 4. Split around unresolved long gaps
    # ==========================================

    segments = (
        find_valid_segments(
            continuous,
            min_segment_frames=(
                min_segment_frames
            ),
        )
    )

    if len(segments) == 0:
        raise ValueError(
            "No continuous RGB segment is "
            "long enough for POS processing."
        )

    # ==========================================
    # 5. Process every valid segment
    # ==========================================

    output_segments = []

    segment_id = 0

    for segment in segments:

        segment_id += 1

        # --------------------------------------
        # Safety check
        # --------------------------------------

        if (
            segment[
                RGB_COLUMNS
            ]
            .isna()
            .any()
            .any()
        ):
            raise ValueError(
                "A POS segment contains "
                "missing RGB values."
            )

        # --------------------------------------
        # Preserve frame metadata
        # --------------------------------------

        output = pd.DataFrame()

        output["frame"] = (
            segment[
                "frame"
            ]
            .astype(int)
            .values
        )

        output["time_seconds"] = (
            segment[
                "time_seconds"
            ]
            .values
        )

        output["interpolated"] = (
            segment[
                "interpolated"
            ]
            .astype(bool)
            .values
        )

        output["segment_id"] = (
            segment_id
        )

        # --------------------------------------
        # POS for each facial region
        # --------------------------------------

        for region in (
            RPPG_REGIONS
        ):

            rgb = np.column_stack(
                [
                    segment[
                        f"{region}_R"
                    ].values,

                    segment[
                        f"{region}_G"
                    ].values,

                    segment[
                        f"{region}_B"
                    ].values,
                ]
            )

            if not np.isfinite(
                rgb
            ).all():

                raise ValueError(
                    "Non-finite RGB values "
                    f"found for {region} "
                    f"in segment "
                    f"{segment_id}."
                )

            rppg = pos_rppg(
                rgb,
                fps,
                window_seconds=(
                    window_seconds
                ),
            )

            if not np.isfinite(
                rppg
            ).all():

                raise ValueError(
                    "POS produced non-finite "
                    f"values for {region} "
                    f"in segment "
                    f"{segment_id}."
                )

            output[
                f"{region}_rppg"
            ] = rppg

        output_segments.append(
            output
        )

    # ==========================================
    # 6. Combine processed segments
    # ==========================================

    result = pd.concat(
        output_segments,
        ignore_index=True,
    )

    result = (
        result
        .sort_values("frame")
        .reset_index(drop=True)
    )

    # ==========================================
    # 7. Final validation
    # ==========================================

    rppg_columns = [
        f"{region}_rppg"
        for region
        in RPPG_REGIONS
    ]

    if not np.isfinite(
        result[
            rppg_columns
        ].values
    ).all():

        raise ValueError(
            "Final rPPG dataframe contains "
            "non-finite values."
        )

    if not result[
        "frame"
    ].is_unique:

        raise ValueError(
            "Duplicate frame IDs found in "
            "rPPG dataframe."
        )

    if not result[
        "frame"
    ].is_monotonic_increasing:

        raise ValueError(
            "rPPG frame IDs are not "
            "monotonically increasing."
        )

    return result