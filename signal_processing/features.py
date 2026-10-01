import numpy as np
from scipy.signal import welch


def rms(signal):
    """
    Root mean square amplitude.
    Useful for estimating overall muscle activation strength.
    """
    signal = np.asarray(signal, dtype=float)
    return np.sqrt(np.mean(signal ** 2))


def mean_absolute_value(signal):
    """
    Mean absolute value of the EMG signal.
    """
    signal = np.asarray(signal, dtype=float)
    return np.mean(np.abs(signal))


def zero_crossings(signal, threshold=0.0):
    """
    Count how many times the signal crosses zero.

    A small threshold can be used later to reduce sensitivity to noise.
    """
    signal = np.asarray(signal, dtype=float)

    crossings = 0

    for i in range(1, len(signal)):
        if (
            (signal[i - 1] < 0 and signal[i] > 0)
            or
            (signal[i - 1] > 0 and signal[i] < 0)
        ):
            if abs(signal[i] - signal[i - 1]) > threshold:
                crossings += 1

    return crossings


def waveform_length(signal):
    """
    Measure the cumulative amount of change in the EMG waveform.
    """
    signal = np.asarray(signal, dtype=float)
    return np.sum(np.abs(np.diff(signal)))


def median_frequency(signal, sampling_rate):
    """
    Compute the median frequency of the EMG power spectrum.

    This can be useful when studying muscle fatigue.
    """
    signal = np.asarray(signal, dtype=float)

    frequencies, power = welch(
        signal,
        fs=sampling_rate,
        nperseg=min(256, len(signal)),
    )

    cumulative_power = np.cumsum(power)
    half_power = cumulative_power[-1] / 2

    index = np.where(cumulative_power >= half_power)[0][0]

    return frequencies[index]


def extract_features(signal, sampling_rate):
    """
    Extract a basic feature set from one EMG window.

    Returns
    -------
    dict
        Feature names and values.
    """

    return {
        "rms": rms(signal),
        "mav": mean_absolute_value(signal),
        "zero_crossings": zero_crossings(signal),
        "waveform_length": waveform_length(signal),
        "median_frequency": median_frequency(signal, sampling_rate),
    }