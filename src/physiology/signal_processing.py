import numpy as np

from scipy.signal import (
    butter,
    filtfilt,
    detrend,
)


def detrend_signal(signal):
    """
    Remove linear trend from a temporal signal.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    return detrend(signal)


def bandpass_filter(
    signal,
    fps,
    low_frequency=0.7,
    high_frequency=4.0,
    order=3,
):
    """
    Apply Butterworth band-pass filtering.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    nyquist = 0.5 * fps

    low = low_frequency / nyquist
    high = high_frequency / nyquist

    if not 0 < low < high < 1:
        raise ValueError(
            "Invalid band-pass frequencies "
            f"for FPS={fps:.2f}"
        )

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered


def normalize_signal(signal):
    """
    Standardize a signal to zero mean and unit variance.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-8:
        return np.zeros_like(signal)

    return (
        signal - mean
    ) / std


def preprocess_signal(
    signal,
    fps,
    low_frequency=0.7,
    high_frequency=4.0,
):
    """
    Complete temporal preprocessing pipeline.
    """

    detrended = detrend_signal(
        signal
    )

    filtered = bandpass_filter(
        detrended,
        fps,
        low_frequency,
        high_frequency,
    )

    normalized = normalize_signal(
        filtered
    )

    return normalized