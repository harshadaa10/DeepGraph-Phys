import numpy as np
import pandas as pd


RGB_COLUMNS = [
    "forehead_R",
    "forehead_G",
    "forehead_B",
    "left_cheek_R",
    "left_cheek_G",
    "left_cheek_B",
    "right_cheek_R",
    "right_cheek_G",
    "right_cheek_B",
]


def restore_regular_rgb_grid(
    rgb_dataframe,
    fps,
    max_gap_frames=6,
):
    """
    Restore RGB measurements to a regular video-frame grid.

    Only complete internal missing runs whose length is less
    than or equal to max_gap_frames are interpolated.

    Longer gaps remain entirely invalid. They are not partially
    interpolated.

    Parameters
    ----------
    rgb_dataframe : pandas.DataFrame
        RGB measurements from frames where a face was detected.

    fps : float
        Original video frame rate.

    max_gap_frames : int
        Maximum complete missing run that may be interpolated.

    Returns
    -------
    pandas.DataFrame
        Regular frame grid containing RGB values together with
        interpolation and original-detection flags.
    """

    if rgb_dataframe.empty:
        raise ValueError(
            "RGB dataframe is empty."
        )

    if fps <= 0:
        raise ValueError(
            "FPS must be greater than zero."
        )

    if max_gap_frames < 0:
        raise ValueError(
            "max_gap_frames cannot be negative."
        )

    dataframe = (
        rgb_dataframe
        .copy()
        .sort_values("frame")
        .reset_index(drop=True)
    )

    # ------------------------------------------
    # Build complete frame grid
    # ------------------------------------------

    first_frame = int(
        dataframe["frame"].min()
    )

    last_frame = int(
        dataframe["frame"].max()
    )

    full_frames = pd.DataFrame(
        {
            "frame": np.arange(
                first_frame,
                last_frame + 1,
                dtype=int,
            )
        }
    )

    restored = full_frames.merge(
        dataframe[
            ["frame"] + RGB_COLUMNS
        ],
        on="frame",
        how="left",
    )

    restored["time_seconds"] = (
        restored["frame"]
        / float(fps)
    )

    # ------------------------------------------
    # Record original face detections
    # ------------------------------------------

    originally_valid = (
        restored[RGB_COLUMNS]
        .notna()
        .all(axis=1)
    )

    restored[
        "original_detection"
    ] = originally_valid

    # ------------------------------------------
    # Find missing runs
    # ------------------------------------------

    missing = (
        ~originally_valid
    ).to_numpy()

    interpolated_flag = np.zeros(
        len(restored),
        dtype=bool,
    )

    index = 0

    while index < len(restored):

        if not missing[index]:
            index += 1
            continue

        run_start = index

        while (
            index < len(restored)
            and missing[index]
        ):
            index += 1

        run_end = index - 1

        run_length = (
            run_end
            - run_start
            + 1
        )

        # We interpolate only internal gaps.
        has_left_boundary = (
            run_start > 0
            and originally_valid.iloc[
                run_start - 1
            ]
        )

        has_right_boundary = (
            run_end + 1
            < len(restored)
            and originally_valid.iloc[
                run_end + 1
            ]
        )

        should_interpolate = (
            run_length
            <= max_gap_frames
            and has_left_boundary
            and has_right_boundary
        )

        if not should_interpolate:
            continue

        left_index = (
            run_start - 1
        )

        right_index = (
            run_end + 1
        )

        left_values = (
            restored.loc[
                left_index,
                RGB_COLUMNS,
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        right_values = (
            restored.loc[
                right_index,
                RGB_COLUMNS,
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        # Linear interpolation across the
        # complete short missing run.
        denominator = (
            run_length + 1
        )

        for offset in range(
            1,
            run_length + 1,
        ):

            fraction = (
                offset
                / denominator
            )

            values = (
                left_values
                + fraction
                * (
                    right_values
                    - left_values
                )
            )

            row_index = (
                run_start
                + offset
                - 1
            )

            restored.loc[
                row_index,
                RGB_COLUMNS,
            ] = values

            interpolated_flag[
                row_index
            ] = True

    restored[
        "interpolated"
    ] = interpolated_flag

    return restored


def find_valid_segments(
    restored_dataframe,
    min_segment_frames=1,
):
    """
    Find contiguous valid RGB segments.

    A segment ends whenever one or more RGB channels are
    unavailable.

    Parameters
    ----------
    restored_dataframe : pandas.DataFrame
        Output from restore_regular_rgb_grid().

    min_segment_frames : int
        Minimum number of consecutive valid frames required
        for a segment to be returned.

    Returns
    -------
    list of pandas.DataFrame
        Continuous valid RGB segments.
    """

    if restored_dataframe.empty:
        return []

    if min_segment_frames < 1:
        raise ValueError(
            "min_segment_frames must be at least 1."
        )

    valid_mask = (
        restored_dataframe[
            RGB_COLUMNS
        ]
        .notna()
        .all(axis=1)
        .to_numpy()
    )

    segments = []

    index = 0

    while index < len(
        restored_dataframe
    ):

        if not valid_mask[index]:
            index += 1
            continue

        segment_start = index

        while (
            index
            < len(restored_dataframe)
            and valid_mask[index]
        ):
            index += 1

        segment_end = index

        segment_length = (
            segment_end
            - segment_start
        )

        if (
            segment_length
            >= min_segment_frames
        ):

            segment = (
                restored_dataframe
                .iloc[
                    segment_start:
                    segment_end
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            segments.append(
                segment
            )

    return segments


def continuity_statistics(
    restored_dataframe,
):
    """
    Return RGB continuity quality-control statistics.
    """

    total = len(
        restored_dataframe
    )

    original = int(
        restored_dataframe[
            "original_detection"
        ].sum()
    )

    interpolated = int(
        restored_dataframe[
            "interpolated"
        ].sum()
    )

    valid = int(
        restored_dataframe[
            RGB_COLUMNS
        ]
        .notna()
        .all(axis=1)
        .sum()
    )

    invalid = (
        total - valid
    )

    return {
        "grid_frames":
            total,

        "original_frames":
            original,

        "interpolated_frames":
            interpolated,

        "valid_frames":
            valid,

        "invalid_frames":
            invalid,

        "valid_rate": (
            valid / total
            if total > 0
            else 0.0
        ),
    }