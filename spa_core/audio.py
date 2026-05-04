from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile

from .models import SpaSignal
from .processing import amplitude_envelope


MEDIA_FILE_FILTER = (
    "Audio/media files (*.wav *.WAV *.wave *.WAVE *.webm *.WEBM *.m4a *.M4A "
    "*.mp3 *.MP3 *.mp4 *.MP4 *.mov *.MOV *.flac *.FLAC *.ogg *.OGG *.opus *.OPUS *.aac *.AAC);;All files (*)"
)
SKIP_SCAN_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".matplotlib",
    "outputs",
    "spa_outputs",
    "metadata_json",
    "reports",
}


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


def _require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(
            f"{name} is required to read this media file. Install FFmpeg first, for example with `brew install ffmpeg`."
        )
    return executable


def _ffprobe_audio_stream(path: Path) -> dict[str, object]:
    ffprobe = _require_executable("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,codec_type,sample_rate,channels,bits_per_sample,bits_per_raw_sample",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "ffprobe could not inspect the file."
        raise RuntimeError(message)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid metadata.") from exc
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError("No audio stream was found in this file.")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise RuntimeError("ffprobe returned invalid audio stream metadata.")
    return stream


def has_audio_stream(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        _ffprobe_audio_stream(path)
    except Exception:
        return False
    return True


def iter_audio_files(directory: str | Path) -> list[Path]:
    """Return files with an audio stream, based on ffprobe rather than extension."""

    directory = Path(directory)
    audio_paths: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        if any(part in SKIP_SCAN_DIR_NAMES for part in path.relative_to(directory).parts[:-1]):
            continue
        if has_audio_stream(path):
            audio_paths.append(path)
    return audio_paths


def _bit_depth_from_ffprobe_stream(stream: dict[str, object]) -> int:
    for key in ("bits_per_raw_sample", "bits_per_sample"):
        value = stream.get(key)
        if value in (None, "", "N/A"):
            continue
        try:
            bits = int(value)
        except (TypeError, ValueError):
            continue
        if bits > 0:
            return bits
    return 32


def _read_with_ffmpeg(path: Path) -> tuple[np.ndarray, int, int]:
    stream = _ffprobe_audio_stream(path)
    try:
        sample_rate = int(stream.get("sample_rate", 0))
    except (TypeError, ValueError):
        sample_rate = 0
    try:
        channels = int(stream.get("channels", 1))
    except (TypeError, ValueError):
        channels = 1
    if sample_rate <= 0:
        raise RuntimeError("Could not determine the audio sample rate.")
    channels = max(1, channels)
    bit_depth = _bit_depth_from_ffprobe_stream(stream)

    ffmpeg = _require_executable("ffmpeg")
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-map",
        "0:a:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg could not decode the audio stream."
        raise RuntimeError(message)
    if not completed.stdout:
        raise RuntimeError("ffmpeg decoded an empty audio stream.")
    audio = np.frombuffer(completed.stdout, dtype=np.float32).astype(float)
    if channels > 1:
        usable = (audio.size // channels) * channels
        audio = audio[:usable].reshape(-1, channels).mean(axis=1)
    return audio.reshape(-1), sample_rate, bit_depth


def read_wav(path: str | Path) -> SpaSignal:
    path = Path(path)
    read_errors: list[str] = []
    try:
        import soundfile as sf

        data, sample_rate = sf.read(path, always_2d=False)
        bit_depth = _bit_depth_from_soundfile(path) or _bit_depth_from_dtype(np.asarray(data).dtype)
        audio = np.asarray(data, dtype=float)
    except Exception as exc:
        read_errors.append(f"soundfile: {exc}")
        try:
            sample_rate, data = wavfile.read(path)
            bit_depth = _bit_depth_from_soundfile(path) or _bit_depth_from_dtype(data.dtype)
            audio = _to_float_audio(data)
        except Exception as wav_exc:
            read_errors.append(f"scipy wavfile: {wav_exc}")
            try:
                audio, sample_rate, bit_depth = _read_with_ffmpeg(path)
            except Exception as ffmpeg_exc:
                read_errors.append(f"ffprobe/ffmpeg: {ffmpeg_exc}")
                details = "\n".join(f"- {error}" for error in read_errors)
                raise RuntimeError(f"Could not read audio from {path.name}.\n{details}") from ffmpeg_exc
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
