from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import signal

from .models import Region, SpaResult, SpaSettings, SpaSignal
from .thresholds import (
    adaptive_threshold_curve,
    automatic_static_threshold,
    noise_statistics,
    normalize_region,
)


EVENT_COLUMNS = [
    "Iteration",
    "Event Number",
    "Onset",
    "Offset",
    "Duration",
    "Minimum",
    "Maximum",
    "Mean",
    "StdDev",
]

TOTAL_COLUMNS = [
    "Iteration",
    "Threshold",
    "%Pause",
    "%Speech",
    "Pause_duration",
    "Speech_duration",
    "Total_duration",
    "Pause_events",
    "Speech_events",
    "Peak Frequency",
    "Peak Amplitude",
    "3 dB Bandwidth",
    "Speech_Threshold",
    "Pause_Threshold",
    "Type",
    "Calc_Time",
    "Mean_Pause",
    "Mean_Speech",
    "Stddev_Pause",
    "Stddev_Speech",
    "CV_Speech_Duration",
    "CV_Pause_Duration",
    "CVR",
    "Stddev_AllSignal",
    "Mean_Minimum_Speech",
    "Mean_Maximum_Speech",
    "Mean_Mean_Speech",
    "Mean_Stddev_Speech",
    "Stddev_Minimum_Speech",
    "Stddev_Maximum_Speech",
    "Stddev_Mean_Speech",
    "Stddev_Stddev_Speech",
    "CV_Minimum_Speech",
    "CV_Maximum_Speech",
    "CV_Mean_Speech",
    "CV_Stddev_Speech",
    "Mean_Minimum_Pause",
    "Mean_Maximum_Pause",
    "Mean_Mean_Pause",
    "Mean_Stddev_Pause",
    "Stddev_Minimum_Pause",
    "Stddev_Maximum_Pause",
    "Stddev_Mean_Pause",
    "Stddev_Stddev_Pause",
    "CV_Minimum_Pause",
    "CV_Maximum_Pause",
    "CV_Mean_Pause",
    "CV_Stddev_Pause",
]

VARIABLE_TYPES = {
    "ONE TIME": 1,
    "PAUSE": 2,
    "SPEECH": 3,
    "AMPLITUDE": 4,
}


def _std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    value = numerator / denominator
    return float(value) if np.isfinite(value) else 0.0


def _matlab_pause_boundaries(values: np.ndarray, speech_indices_1based: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    min_mat = values.copy()
    min_mat[speech_indices_1based - 1] = 0.0
    if min_mat.size < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    is_zero = min_mat == 0
    onset_min = np.flatnonzero(is_zero[:-1] & ~is_zero[1:]) + 2
    offset_candidates = np.flatnonzero(~is_zero[:-1] & is_zero[1:]) + 1

    if onset_min.size == 0 or offset_candidates.size == 0:
        offset_min = np.empty(0, dtype=int)
    else:
        previous_onset_index = np.searchsorted(onset_min, offset_candidates, side="right") - 1
        keep = (previous_onset_index >= 0) & (onset_min[previous_onset_index] < offset_candidates)
        offset_min = offset_candidates[keep]

    if min_mat[-2] != 0 and onset_min.size:
        onset_min = onset_min[:-1]
    return onset_min.astype(int), offset_min.astype(int)


def _same_length(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pair_count = min(len(left), len(right))
    return left[:pair_count], right[:pair_count]


def _remove_short_speech_events(
    onset_min: np.ndarray,
    offset_min: np.ndarray,
    speech_threshold_points: float,
) -> tuple[np.ndarray, np.ndarray]:
    onset_min, offset_min = _same_length(onset_min, offset_min)
    if len(onset_min) <= 1 or len(offset_min) <= 1:
        return onset_min, offset_min

    speech_event_dur = onset_min[1:] - offset_min[:-1]
    short_speech = np.flatnonzero(speech_event_dur < speech_threshold_points)
    if short_speech.size == 0:
        return onset_min, offset_min

    last_small_speech_offset = offset_min[short_speech[-1]]
    onset_min = np.delete(onset_min, short_speech + 1)
    offset_min = np.delete(offset_min, short_speech)
    if offset_min.size and last_small_speech_offset > offset_min[-1]:
        onset_min = onset_min[:-1]
        offset_min = offset_min[:-1]
    return onset_min, offset_min


def _remove_short_pauses_matlab(
    onset_min: np.ndarray,
    offset_min: np.ndarray,
    pause_threshold_points: float,
) -> tuple[np.ndarray, np.ndarray]:
    onset_min, offset_min = _same_length(onset_min, offset_min)
    if onset_min.size == 0:
        return onset_min, offset_min
    short_pause = np.flatnonzero((offset_min - onset_min) <= pause_threshold_points)
    if short_pause.size:
        onset_min = np.delete(onset_min, short_pause)
        offset_min = np.delete(offset_min, short_pause)
    return onset_min, offset_min


def _matlab_events_to_python_ranges(onset_1based: np.ndarray, offset_1based: np.ndarray) -> np.ndarray:
    onset_1based, offset_1based = _same_length(onset_1based, offset_1based)
    keep = offset_1based > onset_1based
    if not np.any(keep):
        return np.empty((0, 2), dtype=int)
    starts = onset_1based[keep] - 1
    ends = offset_1based[keep] - 1
    return np.column_stack([starts, ends]).astype(int)


def detect_events(
    analysis_envelope: np.ndarray,
    threshold_curve: np.ndarray,
    sample_rate: int,
    speech_threshold_ms: float,
    pause_threshold_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return speech and pause events using the original MATLAB boundary rules."""

    values = np.asarray(analysis_envelope, dtype=float).reshape(-1)
    curve = np.asarray(threshold_curve, dtype=float).reshape(-1)
    if values.size != curve.size:
        raise ValueError("Envelope and threshold curve must have the same length.")
    speech_indices_1based = np.flatnonzero(values > curve) + 1
    if speech_indices_1based.size == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    onset_min, offset_min = _matlab_pause_boundaries(values, speech_indices_1based)
    onset_min, offset_min = _remove_short_speech_events(
        onset_min,
        offset_min,
        (speech_threshold_ms / 1000.0) * sample_rate,
    )
    onset_min, offset_min = _remove_short_pauses_matlab(
        onset_min,
        offset_min,
        (pause_threshold_ms / 1000.0) * sample_rate,
    )

    pause = _matlab_events_to_python_ranges(onset_min, offset_min)
    speech_onsets = np.concatenate([[speech_indices_1based[0]], offset_min + 1])
    speech_offsets = np.concatenate([onset_min - 1, [values.size]])
    speech = _matlab_events_to_python_ranges(speech_onsets, speech_offsets)
    return speech, pause


def _event_dataframe(
    events: np.ndarray,
    amplitude_signal: np.ndarray,
    sample_rate: int,
    iteration: int,
) -> pd.DataFrame:
    rows = []
    for event_number, (start, end) in enumerate(np.asarray(events, dtype=int), start=1):
        start = max(0, int(start))
        end = min(len(amplitude_signal), int(end))
        if end <= start:
            continue
        section = np.asarray(amplitude_signal[start:end], dtype=float)
        rows.append(
            [
                iteration,
                event_number,
                start / sample_rate,
                end / sample_rate,
                (end - start) / sample_rate,
                float(np.min(section)),
                float(np.max(section)),
                float(np.mean(section)),
                _std(section),
            ]
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _column_stats(frame: pd.DataFrame, column: str) -> tuple[float, float, float]:
    values = frame[column].to_numpy(dtype=float) if column in frame else np.asarray([], dtype=float)
    mean_value = _safe_mean(values)
    std_value = _std(values)
    cv_value = _safe_ratio(std_value, mean_value)
    return mean_value, std_value, cv_value


def _duration_stats(frame: pd.DataFrame) -> tuple[float, float, float]:
    if frame.empty:
        return 0.0, 0.0, 0.0
    durations = frame["Duration"].to_numpy(dtype=float)
    mean_value = _safe_mean(durations)
    std_value = _std(durations)
    return mean_value, std_value, _safe_ratio(std_value, mean_value)


def spectral_summary(values: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    """Approximate fastfft.m's peak frequency, peak amplitude, and 3 dB bandwidth."""

    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size < 4:
        return 0.0, 0.0, 0.0
    detrended = signal.detrend(values, type="constant")
    if detrended.size > 300:
        decimated = signal.decimate(detrended, 100, zero_phase=True)
        fs = sample_rate / 100.0
    else:
        decimated = detrended
        fs = float(sample_rate)
    if decimated.size < 4:
        return 0.0, 0.0, 0.0
    nperseg = min(2048, decimated.size)
    freqs, power = signal.welch(decimated, fs=fs, nperseg=nperseg)
    if power.size == 0:
        return 0.0, 0.0, 0.0
    power_db = 10.0 * np.log10(np.maximum(power, np.finfo(float).eps))
    peak_index = int(np.argmax(power_db))
    peak_frequency = float(freqs[peak_index])
    peak_amplitude = float(power_db[peak_index])
    above = np.flatnonzero(power_db >= peak_amplitude - 3.0)
    if above.size >= 2:
        bandwidth = float(freqs[above[-1]] - freqs[above[0]])
    else:
        bandwidth = 0.0
    return peak_frequency, peak_amplitude, bandwidth


def _total_row(
    iteration: int,
    reported_threshold: float,
    speech_frame: pd.DataFrame,
    pause_frame: pd.DataFrame,
    amplitude_signal: np.ndarray,
    sample_rate: int,
    speech_threshold_ms: float,
    pause_threshold_ms: float,
    variable_type: int,
    calc_time: float,
) -> list[float]:
    pause_duration = float(pause_frame["Duration"].sum()) if not pause_frame.empty else 0.0
    speech_duration = float(speech_frame["Duration"].sum()) if not speech_frame.empty else 0.0
    total_duration = pause_duration + speech_duration
    if total_duration == 0:
        total_duration = len(amplitude_signal) / sample_rate if sample_rate else 0.0
    percent_pause = _safe_ratio(pause_duration * 100.0, total_duration)
    percent_speech = _safe_ratio(speech_duration * 100.0, total_duration)
    mean_pause, std_pause, cv_pause = _duration_stats(pause_frame)
    mean_speech, std_speech, cv_speech = _duration_stats(speech_frame)
    cvr = _safe_ratio(cv_pause, cv_speech)
    peak_frequency, peak_amplitude, bandwidth = spectral_summary(amplitude_signal, sample_rate)

    speech_stats = {}
    pause_stats = {}
    for column in ["Minimum", "Maximum", "Mean", "StdDev"]:
        speech_stats[column] = _column_stats(speech_frame, column)
        pause_stats[column] = _column_stats(pause_frame, column)

    return [
        iteration,
        reported_threshold,
        percent_pause,
        percent_speech,
        pause_duration,
        speech_duration,
        total_duration,
        int(len(pause_frame)),
        int(len(speech_frame)),
        peak_frequency,
        peak_amplitude,
        bandwidth,
        speech_threshold_ms,
        pause_threshold_ms,
        variable_type,
        calc_time,
        mean_pause,
        mean_speech,
        std_pause,
        std_speech,
        cv_speech,
        cv_pause,
        cvr,
        _std(amplitude_signal),
        speech_stats["Minimum"][0],
        speech_stats["Maximum"][0],
        speech_stats["Mean"][0],
        speech_stats["StdDev"][0],
        speech_stats["Minimum"][1],
        speech_stats["Maximum"][1],
        speech_stats["Mean"][1],
        speech_stats["StdDev"][1],
        speech_stats["Minimum"][2],
        speech_stats["Maximum"][2],
        speech_stats["Mean"][2],
        speech_stats["StdDev"][2],
        pause_stats["Minimum"][0],
        pause_stats["Maximum"][0],
        pause_stats["Mean"][0],
        pause_stats["StdDev"][0],
        pause_stats["Minimum"][1],
        pause_stats["Maximum"][1],
        pause_stats["Mean"][1],
        pause_stats["StdDev"][1],
        pause_stats["Minimum"][2],
        pause_stats["Maximum"][2],
        pause_stats["Mean"][2],
        pause_stats["StdDev"][2],
    ]


def _iteration_settings(settings: SpaSettings, iteration_index: int) -> tuple[float, float, float]:
    speech_threshold_ms = float(settings.speech_threshold_ms)
    pause_threshold_ms = float(settings.pause_threshold_ms)
    amplitude_offset = 0.0
    mode = settings.variable_mode.upper()
    if mode == "PAUSE":
        pause_threshold_ms += iteration_index * float(settings.time_increment_ms)
    elif mode == "SPEECH":
        speech_threshold_ms += iteration_index * float(settings.time_increment_ms)
    elif mode == "AMPLITUDE":
        amplitude_offset = iteration_index * float(settings.amplitude_increment)
    return speech_threshold_ms, pause_threshold_ms, amplitude_offset


def _iteration_count(settings: SpaSettings) -> int:
    if settings.variable_mode.upper() == "ONE TIME":
        return 1
    return max(1, int(settings.iterations))


def _base_threshold(
    signal: SpaSignal,
    settings: SpaSettings,
    noise_region: Region | None,
    manual_threshold: float | None,
) -> tuple[float, float, float]:
    if settings.threshold_mode.lower() == "manual":
        if manual_threshold is None:
            raise ValueError("Manual threshold mode requires a manual threshold.")
        return float(manual_threshold), 0.0, 0.0
    if noise_region is None:
        raise ValueError("Automatic threshold mode requires a selected noise region.")
    threshold, noise_mean, noise_std = automatic_static_threshold(
        signal.normalized_envelope,
        noise_region,
        settings.sd_multiplier,
    )
    return threshold, noise_mean, noise_std


def run_spa(
    signal: SpaSignal,
    settings: SpaSettings | None = None,
    noise_region: Region | None = None,
    analysis_region: Region | None = None,
    manual_threshold: float | None = None,
) -> SpaResult:
    """Run the single-file SPA workflow on a loaded signal."""

    settings = settings or SpaSettings()
    analysis_region = analysis_region or signal.analysis_region or (0, len(signal.raw_audio))
    analysis_region = normalize_region(analysis_region, len(signal.raw_audio))
    noise_region = normalize_region(noise_region, len(signal.raw_audio)) if noise_region is not None else signal.noise_region
    if noise_region is not None:
        noise_region = normalize_region(noise_region, len(signal.raw_audio))
    manual_threshold = manual_threshold if manual_threshold is not None else signal.manual_threshold

    signal.analysis_region = analysis_region
    signal.noise_region = noise_region
    signal.manual_threshold = manual_threshold

    analysis_start, analysis_end = analysis_region
    analysis_envelope = signal.normalized_envelope[analysis_start:analysis_end]
    amplitude_signal = signal.filtered_envelope[analysis_start:analysis_end]
    base_threshold, noise_mean, noise_std = _base_threshold(signal, settings, noise_region, manual_threshold)
    variable_type = VARIABLE_TYPES.get(settings.variable_mode.upper(), 1)
    iteration_count = _iteration_count(settings)

    speech_frames = []
    pause_frames = []
    total_rows = []
    threshold_by_iteration: dict[int, np.ndarray] = {}
    last_speech_events = np.empty((0, 2), dtype=int)
    last_pause_events = np.empty((0, 2), dtype=int)
    last_curve = np.zeros_like(analysis_envelope, dtype=float)

    for iteration_index in range(iteration_count):
        iteration = iteration_index + 1
        start_time = time.perf_counter()
        speech_threshold_ms, pause_threshold_ms, amplitude_offset = _iteration_settings(settings, iteration_index)
        reported_threshold = base_threshold + amplitude_offset

        use_adaptive = settings.adaptive and settings.threshold_mode.lower() == "automatic"
        if use_adaptive:
            curve = adaptive_threshold_curve(
                analysis_envelope,
                noise_mean,
                noise_std,
                settings.sd_multiplier,
            )
            working_envelope = analysis_envelope[: len(curve)]
            working_amplitude = amplitude_signal[: len(curve)]
        else:
            curve = np.full_like(analysis_envelope, reported_threshold, dtype=float)
            working_envelope = analysis_envelope
            working_amplitude = amplitude_signal

        speech_events, pause_events = detect_events(
            working_envelope,
            curve,
            signal.sample_rate,
            speech_threshold_ms,
            pause_threshold_ms,
        )
        calc_time = time.perf_counter() - start_time

        speech_frame = _event_dataframe(speech_events, working_amplitude, signal.sample_rate, iteration)
        pause_frame = _event_dataframe(pause_events, working_amplitude, signal.sample_rate, iteration)
        speech_frames.append(speech_frame)
        pause_frames.append(pause_frame)
        total_rows.append(
            _total_row(
                iteration,
                reported_threshold,
                speech_frame,
                pause_frame,
                working_amplitude,
                signal.sample_rate,
                speech_threshold_ms,
                pause_threshold_ms,
                variable_type,
                calc_time,
            )
        )
        threshold_by_iteration[iteration] = curve
        last_curve = curve
        last_speech_events = speech_events
        last_pause_events = pause_events

    speech_matrix = pd.concat(speech_frames, ignore_index=True) if speech_frames else pd.DataFrame(columns=EVENT_COLUMNS)
    pause_matrix = pd.concat(pause_frames, ignore_index=True) if pause_frames else pd.DataFrame(columns=EVENT_COLUMNS)
    total_matrix = pd.DataFrame(total_rows, columns=TOTAL_COLUMNS)

    return SpaResult(
        settings=replace(settings),
        signal_path=signal.path,
        sample_rate=signal.sample_rate,
        bit_depth=signal.bit_depth,
        analysis_region=analysis_region,
        noise_region=noise_region,
        threshold_curve=last_curve,
        threshold_by_iteration=threshold_by_iteration,
        speech_matrix=speech_matrix,
        pause_matrix=pause_matrix,
        total_matrix=total_matrix,
        speech_events_samples=last_speech_events,
        pause_events_samples=last_pause_events,
    )
