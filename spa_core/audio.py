from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile

from .models import SpaSignal
from .processing import amplitude_envelope


def _bit_depth_from_dtype(dtype: np.dtype) -> int:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return dtype.itemsize * 8
    if np.issubdtype(dtype, np.floating):
        return 32
    return 16


def _bit_depth_from_soundfile(path: Path) -> int | None:
    try:
        import soundfile as sf
    except Exception:
        return None
    try:
        subtype = sf.info(path).subtype.upper()
    except Exception:
        return None
    if "PCM_U8" in subtype:
        return 8
    if "PCM_16" in subtype:
        return 16
    if "PCM_24" in subtype:
        return 24
    if "PCM_32" in subtype:
        return 32
    if "FLOAT" in subtype:
        return 32
    if "DOUBLE" in subtype:
        return 64
    return None


def _to_float_audio(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(float)
    if np.issubdtype(data.dtype, np.unsignedinteger):
        midpoint = float(np.iinfo(data.dtype).max + 1) / 2.0
        return (data.astype(float) - midpoint) / midpoint
    if np.issubdtype(data.dtype, np.signedinteger):
        scale = float(max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max))
        return data.astype(float) / scale
    return data.astype(float)


def read_wav(path: str | Path) -> SpaSignal:
    path = Path(path)
    try:
        import soundfile as sf

        data, sample_rate = sf.read(path, always_2d=False)
        bit_depth = _bit_depth_from_soundfile(path) or _bit_depth_from_dtype(np.asarray(data).dtype)
        audio = np.asarray(data, dtype=float)
    except Exception:
        sample_rate, data = wavfile.read(path)
        bit_depth = _bit_depth_from_soundfile(path) or _bit_depth_from_dtype(data.dtype)
        audio = _to_float_audio(data)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=float).reshape(-1)
    detrended, rectified, filtered, normalized = amplitude_envelope(audio, sample_rate)
    return SpaSignal(
        path=path,
        raw_audio=audio,
        sample_rate=int(sample_rate),
        bit_depth=bit_depth,
        detrended_audio=detrended,
        rectified_audio=rectified,
        filtered_envelope=filtered,
        normalized_envelope=normalized,
        analysis_region=(0, len(audio)),
    )


def _from_float_audio(audio: np.ndarray, bit_depth: int) -> np.ndarray:
    audio = np.clip(np.asarray(audio, dtype=float), -1.0, 1.0)
    if bit_depth <= 8:
        return ((audio + 1.0) * 127.5).astype(np.uint8)
    if bit_depth <= 16:
        return (audio * 32767.0).astype(np.int16)
    return (audio * 2147483647.0).astype(np.int32)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int, bit_depth: int = 16) -> None:
    path = Path(path)
    try:
        import soundfile as sf
    except Exception:
        wavfile.write(path, sample_rate, _from_float_audio(audio, bit_depth))
        return
    subtype = "PCM_16"
    if bit_depth <= 8:
        subtype = "PCM_U8"
    elif bit_depth <= 16:
        subtype = "PCM_16"
    elif bit_depth <= 24:
        subtype = "PCM_24"
    elif bit_depth <= 32:
        subtype = "PCM_32"
    sf.write(path, np.asarray(audio, dtype=float), sample_rate, subtype=subtype)


def export_speech_segments(
    output_dir: str | Path,
    source_signal: SpaSignal,
    speech_events_samples: np.ndarray,
    prefix: str | None = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or source_signal.path.stem
    written: list[Path] = []
    for idx, (start, end) in enumerate(np.asarray(speech_events_samples, dtype=int), start=1):
        start = max(0, int(start) + source_signal.analysis_start)
        end = min(len(source_signal.raw_audio), int(end) + source_signal.analysis_start)
        if end <= start:
            continue
        out_path = output_dir / f"{prefix}_seg{idx}.wav"
        write_wav(out_path, source_signal.raw_audio[start:end], source_signal.sample_rate, source_signal.bit_depth)
        written.append(out_path)
    return written


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - depends on optional local audio stack
        raise RuntimeError("sounddevice is not installed; install requirements.txt to enable playback.") from exc
    sd.stop()
    sd.play(np.asarray(audio, dtype=float), sample_rate)


def stop_audio() -> None:
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - depends on optional local audio stack
        raise RuntimeError("sounddevice is not installed; install requirements.txt to enable playback.") from exc
    sd.stop()
