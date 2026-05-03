from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .models import Region


def normalize_region(region: Region, length: int) -> Region:
    start, end = int(region[0]), int(region[1])
    if end < start:
        start, end = end, start
    start = max(0, min(length, start))
    end = max(0, min(length, end))
    if end <= start:
        raise ValueError("Region must contain at least one sample.")
    return start, end


def noise_statistics(envelope: np.ndarray, noise_region: Region) -> tuple[float, float]:
    start, end = normalize_region(noise_region, len(envelope))
    section = np.asarray(envelope[start:end], dtype=float)
    ddof = 1 if section.size > 1 else 0
    return float(np.mean(section)), float(np.std(section, ddof=ddof))


def automatic_static_threshold(envelope: np.ndarray, noise_region: Region, sd_multiplier: float) -> tuple[float, float, float]:
    noise_mean, noise_std = noise_statistics(envelope, noise_region)
    return noise_mean + noise_std * float(sd_multiplier), noise_mean, noise_std


def _matlab_round(value: float) -> float:
    return float(np.floor(value + 0.5))


def adaptive_threshold_curve(
    analysis_envelope: np.ndarray,
    noise_mean: float,
    noise_std: float,
    sd_multiplier: float,
    number_blocks: int = 35,
) -> np.ndarray:
    """Replicate fine_events_altered.m.

    threshold_block = std(block)/3 + noise_mean + noise_std * SD
    then spline-interpolate block thresholds across the analysis signal.
    """

    values = np.asarray(analysis_envelope, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    if values.size < number_blocks:
        base = noise_mean + noise_std * float(sd_multiplier)
        ddof = 1 if values.size > 1 else 0
        return np.full(values.shape, base + np.std(values, ddof=ddof) / 3.0)

    blocklength = int(np.floor(values.size / number_blocks))
    usable = blocklength * number_blocks
    thresholds = np.zeros(number_blocks, dtype=float)
    centers = np.zeros(number_blocks, dtype=float)
    base = noise_mean + noise_std * float(sd_multiplier)

    for i in range(number_blocks):
        start = i * blocklength
        end = (i + 1) * blocklength
        block = values[start:end]
        ddof = 1 if block.size > 1 else 0
        thresholds[i] = np.std(block, ddof=ddof) / 3.0 + base
        centers[i] = start + _matlab_round(blocklength / 2.0)

    x = np.arange(1, usable + 1, dtype=float)
    return CubicSpline(centers, thresholds, extrapolate=True)(x)
