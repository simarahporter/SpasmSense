import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(
    signal,
    sampling_rate,
    low_cut=20.0,
    high_cut=450.0,
    order=4,
):
    """
    Band-pass filter for surface EMG signals.

    Parameters
    ----------
    signal : array-like
        Raw EMG signal.
    sampling_rate : float
        Sampling rate in Hz.
    low_cut : float
        Lower cutoff frequency in Hz.
    high_cut : float
        Upper cutoff frequency in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered EMG signal.
    """

    signal = np.asarray(signal, dtype=float)

    nyquist = sampling_rate / 2.0

    if high_cut >= nyquist:
        high_cut = nyquist - 1.0

    low = low_cut / nyquist
    high = high_cut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="bandpass",
    )

    return filtfilt(b, a, signal)


def notch_filter(
    signal,
    sampling_rate,
    notch_frequency=60.0,
    quality_factor=30.0,
):
    """
    Remove electrical power-line noise from an EMG signal.

    In the United States, power-line interference is commonly 60 Hz.

    Parameters
    ----------
    signal : array-like
        EMG signal.
    sampling_rate : float
        Sampling rate in Hz.
    notch_frequency : float
        Frequency to remove.
    quality_factor : float
        Controls notch width.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """

    signal = np.asarray(signal, dtype=float)

    b, a = iirnotch(
        notch_frequency,
        quality_factor,
        sampling_rate,
    )

    return filtfilt(b, a, signal)


def remove_dc_offset(signal):
    """
    Remove the mean voltage from an EMG signal.
    """

    signal = np.asarray(signal, dtype=float)
    return signal - np.mean(signal)


def rectify_signal(signal):
    """
    Full-wave rectify an EMG signal.

    This converts negative EMG amplitudes to positive amplitudes.
    """

    signal = np.asarray(signal, dtype=float)
    return np.abs(signal)


def preprocess_emg(
    signal,
    sampling_rate,
    low_cut=20.0,
    high_cut=450.0,
    notch_frequency=60.0,
):
    """
    Complete basic preprocessing pipeline for surface EMG.

    Pipeline:
        1. Remove DC offset
        2. Remove 60 Hz electrical interference
        3. Apply EMG band-pass filter

    Parameters
    ----------
    signal : array-like
        Raw EMG signal.
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Cleaned EMG signal.
    """

    cleaned = remove_dc_offset(signal)

    cleaned = notch_filter(
        cleaned,
        sampling_rate,
        notch_frequency=notch_frequency,
    )

    cleaned = bandpass_filter(
        cleaned,
        sampling_rate,
        low_cut=low_cut,
        high_cut=high_cut,
    )

    return cleaned
    