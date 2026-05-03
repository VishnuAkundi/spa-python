from __future__ import annotations

import numpy as np
from scipy import signal


def db(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 20.0 * np.log10(np.maximum(np.abs(values), np.finfo(float).eps))


def detrend_constant(audio: np.ndarray) -> np.ndarray:
    return signal.detrend(np.asarray(audio, dtype=float), type="constant", axis=0)


def amplitude_envelope(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replicate initialize_spa.m preprocessing.

    MATLAB path:
    detrend constant -> abs -> 5th-order 30 Hz Butterworth/filtfilt -> normalize 0-100.
    """

    detrended = detrend_constant(audio)
    rectified = np.abs(detrended)
    cutoff_hz = 30.0
    if sample_rate <= cutoff_hz * 2:
        raise ValueError("Sample rate must be greater than 60 Hz for the 30 Hz envelope filter.")

    b, a = signal.butter(5, cutoff_hz / (sample_rate / 2.0))
    filtered = signal.filtfilt(b, a, rectified, axis=0)
    filtered = np.asarray(filtered, dtype=float)

    positive = filtered.copy()
    positive[positive <= 0] = 0.001
    peak = float(np.max(positive))
    if peak <= 0:
        normalized = np.zeros_like(positive)
    else:
        normalized = positive / peak * 100.0

    return detrended, rectified, filtered, normalized
