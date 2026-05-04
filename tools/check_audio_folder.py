from __future__ import annotations

import argparse
import csv
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spa_core.audio import iter_audio_files, read_wav
from spa_core.models import SpaSettings
from spa_core.segmentation import run_spa


def wav_files(audio_dir: Path) -> list[Path]:
    return iter_audio_files(audio_dir)


def quietest_region(signal, window_seconds: float = 0.5, search_seconds: float = 10.0) -> tuple[int, int]:
    envelope = np.asarray(signal.normalized_envelope, dtype=float)
    fs = signal.sample_rate
    if envelope.size == 0:
        raise ValueError("Audio file has no samples.")
    window = max(1, min(envelope.size, int(round(window_seconds * fs))))
    search_length = min(envelope.size, max(window, int(round(search_seconds * fs))))
    search = envelope[:search_length]
    kernel = np.ones(window, dtype=float) / window
    moving_mean = np.convolve(search, kernel, mode="valid")
    start = int(np.argmin(moving_mean))
    return start, start + window


def summarize_total(result) -> dict[str, float]:
    row = result.total_matrix.iloc[-1]
    return {
        "threshold": float(row["Threshold"]),
        "speech_events": int(row["Speech_events"]),
        "pause_events": int(row["Pause_events"]),
        "pct_speech": float(row["%Speech"]),
        "pct_pause": float(row["%Pause"]),
        "speech_duration": float(row["Speech_duration"]),
        "pause_duration": float(row["Pause_duration"]),
        "total_duration": float(row["Total_duration"]),
    }


def suspicious_messages(summary: dict[str, float]) -> list[str]:
    messages: list[str] = []
    if summary["speech_events"] == 0:
        messages.append("no speech events")
    if summary["pause_events"] == 0:
        messages.append("no pause events")
    if summary["pct_speech"] > 99.0:
        messages.append("speech percent > 99")
    if summary["pct_speech"] < 1.0:
        messages.append("speech percent < 1")
    if summary["total_duration"] <= 0:
        messages.append("non-positive total duration")
    for key, value in summary.items():
        if isinstance(value, float) and not np.isfinite(value):
            messages.append(f"non-finite {key}")
    return messages


def check_file(path: Path, variants: list[str], args) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    signal = read_wav(path)
    noise_region = quietest_region(signal, args.noise_window_seconds, args.noise_search_seconds)
    duration = len(signal.raw_audio) / signal.sample_rate

    for variant in variants:
        settings = SpaSettings(
            speech_threshold_ms=args.speech_threshold_ms,
            pause_threshold_ms=args.pause_threshold_ms,
            sd_multiplier=args.sd_multiplier,
            threshold_mode="automatic",
            adaptive=variant == "adaptive",
            variable_mode="ONE TIME",
            iterations=1,
            use_full_file=True,
        )
        result = run_spa(
            signal,
            settings=settings,
            noise_region=noise_region,
            analysis_region=(0, len(signal.raw_audio)),
        )
        summary = summarize_total(result)
        messages = suspicious_messages(summary)
        rows.append(
            {
                "file": str(path),
                "variant": variant,
                "status": "warning" if messages else "ok",
                "sample_rate": signal.sample_rate,
                "bit_depth": signal.bit_depth,
                "duration_seconds": duration,
                "noise_start_seconds": noise_region[0] / signal.sample_rate,
                "noise_end_seconds": noise_region[1] / signal.sample_rate,
                **summary,
                "message": "; ".join(messages),
            }
        )
    return rows


def write_report(rows: list[dict[str, object]], errors: list[dict[str, str]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"audio_check_{timestamp}.csv"
    notes_path = output_dir / f"audio_check_{timestamp}.md"

    fieldnames = [
        "file",
        "variant",
        "status",
        "sample_rate",
        "bit_depth",
        "duration_seconds",
        "noise_start_seconds",
        "noise_end_seconds",
        "threshold",
        "speech_events",
        "pause_events",
        "pct_speech",
        "pct_pause",
        "speech_duration",
        "pause_duration",
        "total_duration",
        "message",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    warning_count = sum(1 for row in rows if row["status"] == "warning")
    with notes_path.open("w") as handle:
        handle.write("# SPA Audio Folder Check\n\n")
        handle.write(f"- Rows: {len(rows)}\n")
        handle.write(f"- OK rows: {ok_count}\n")
        handle.write(f"- Warning rows: {warning_count}\n")
        handle.write(f"- Error files: {len(errors)}\n\n")
        handle.write("This is a noninteractive smoke check, not a MATLAB parity proof. ")
        handle.write("Noise regions were chosen automatically as the quietest window in the first search interval.\n\n")
        if errors:
            handle.write("## Errors\n\n")
            for error in errors:
                handle.write(f"### {error['file']}\n\n")
                handle.write("```text\n")
                handle.write(error["traceback"])
                handle.write("\n```\n\n")
        warnings = [row for row in rows if row["status"] == "warning"]
        if warnings:
            handle.write("## Warnings\n\n")
            for row in warnings[:100]:
                handle.write(
                    f"- `{Path(str(row['file'])).name}` ({row['variant']}): {row['message']} "
                    f"speech={row['speech_events']}, pause={row['pause_events']}, "
                    f"%speech={row['pct_speech']:.2f}\n"
                )
            if len(warnings) > 100:
                handle.write(f"\n... {len(warnings) - 100} more warnings in the CSV.\n")

    return csv_path, notes_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SPA Python over a folder of audio/media files and record errors.")
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--variant", choices=["adaptive", "static", "both"], default="both")
    parser.add_argument("--sd-multiplier", type=float, default=3.0)
    parser.add_argument("--speech-threshold-ms", type=float, default=25.0)
    parser.add_argument("--pause-threshold-ms", type=float, default=250.0)
    parser.add_argument("--noise-window-seconds", type=float, default=0.5)
    parser.add_argument("--noise-search-seconds", type=float, default=10.0)
    args = parser.parse_args()

    files = wav_files(args.audio_dir)
    if args.limit:
        files = files[: args.limit]
    variants = ["adaptive", "static"] if args.variant == "both" else [args.variant]

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}")
        try:
            rows.extend(check_file(path, variants, args))
        except Exception:
            errors.append({"file": str(path), "traceback": traceback.format_exc()})

    csv_path, notes_path = write_report(rows, errors, args.output_dir)
    print(f"\nCSV: {csv_path}")
    print(f"Notes: {notes_path}")
    print(f"Rows: {len(rows)}  Errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
