import numpy as np

from src.physiology.signal_processing import (
    bandpass_filter,
    normalize_signal,
)


def pos_rppg(
    rgb_signal,
    fps,
    window_seconds=1.6,
    low_frequency=0.7,
    high_frequency=4.0,
):
    """
    Extract an rPPG signal using the
    Plane-Orthogonal-to-Skin (POS) method.

    Parameters
    ----------
    rgb_signal : numpy.ndarray
        Array with shape (N, 3), ordered as R, G, B.

    fps : float
        Video frame rate.

    window_seconds : float
        POS temporal window length.

    Returns
    -------
    numpy.ndarray
        Normalized POS rPPG signal.
    """

    rgb_signal = np.asarray(
        rgb_signal,
        dtype=np.float64
    )

    if (
        rgb_signal.ndim != 2
        or rgb_signal.shape[1] != 3
    ):
        raise ValueError(
            "rgb_signal must have shape (N, 3)"
        )

    num_frames = len(rgb_signal)

    window_length = int(
        round(window_seconds * fps)
    )

    if window_length < 2:
        raise ValueError(
            "POS window is too short."
        )

    if num_frames < window_length:
        raise ValueError(
            "Video signal is shorter than "
            "the POS window."
        )

    # Output signal
    pulse = np.zeros(
        num_frames,
        dtype=np.float64
    )

    # Number of overlapping contributions
    weights = np.zeros(
        num_frames,
        dtype=np.float64
    )

    for start in range(
        0,
        num_frames - window_length + 1
    ):

        end = start + window_length

        window = rgb_signal[
            start:end
        ]

        # Mean RGB within current window
        mean_rgb = np.mean(
            window,
            axis=0
        )

        # Avoid division by zero
        mean_rgb = np.where(
            np.abs(mean_rgb) < 1e-8,
            1e-8,
            mean_rgb
        )

        # Temporal normalization
        normalized = (
            window / mean_rgb
        ).T

        # POS projection
        x_signal = (
            3.0 * normalized[0]
            - 2.0 * normalized[1]
        )

        y_signal = (
            1.5 * normalized[0]
            + normalized[1]
            - 1.5 * normalized[2]
        )

        y_std = np.std(
            y_signal
        )

        if y_std < 1e-8:
            continue

        alpha = (
            np.std(x_signal)
            / y_std
        )

        segment = (
            x_signal
            + alpha * y_signal
        )

        # Remove segment mean before
        # overlap-add reconstruction
        segment = (
            segment
            - np.mean(segment)
        )

        pulse[start:end] += segment
        weights[start:end] += 1.0

    # Average overlapping windows
    valid = weights > 0

    pulse[valid] /= weights[valid]

    # Physiological frequency filtering
    pulse = bandpass_filter(
        pulse,
        fps,
        low_frequency,
        high_frequency,
    )

    # Standardize final signal
    pulse = normalize_signal(
        pulse
    )

    return pulse