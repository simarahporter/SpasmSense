import numpy as np


def segment_signal(
    signal,
    sampling_rate,
    window_seconds=0.5,
    overlap=0.5,
):
    """
    Split a continuous EMG signal into overlapping windows.

    Parameters
    ----------
    signal : array-like
        Continuous EMG signal.
    sampling_rate : float
        Sampling rate in Hz.
    window_seconds : float
        Length of each window in seconds.
    overlap : float
        Fractional overlap between windows.
        Example: 0.5 means 50% overlap.

    Returns
    -------
    np.ndarray
        Array of shape:
        (number_of_windows, samples_per_window)
    """

    signal = np.asarray(signal, dtype=float)

    if not 0 <= overlap < 1:
        raise ValueError("overlap must be between 0 and 1")

    window_size = int(window_seconds * sampling_rate)

    if window_size <= 0:
        raise ValueError("window_seconds is too small")

    step_size = int(window_size * (1 - overlap))

    if step_size <= 0:
        raise ValueError("overlap is too large")

    windows = []

    for start in range(
        0,
        len(signal) - window_size + 1,
        step_size,
    ):
        end = start + window_size
        windows.append(signal[start:end])

    return np.array(windows)


def segment_with_timestamps(
    signal,
    sampling_rate,
    window_seconds=0.5,
    overlap=0.5,
):
    """
    Segment EMG and keep the start/end time
    for every window.
    """

    signal = np.asarray(signal, dtype=float)

    window_size = int(window_seconds * sampling_rate)
    step_size = int(window_size * (1 - overlap))

    windows = []

    for start in range(
        0,
        len(signal) - window_size + 1,
        step_size,
    ):
        end = start + window_size

        windows.append(
            {
                "start_time": start / sampling_rate,
                "end_time": end / sampling_rate,
                "signal": signal[start:end],
            }
        )

    return windows