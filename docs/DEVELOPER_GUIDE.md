# Developer Guide

This document is for maintainers changing the Python SPA code.

## Code Organization

```text
run_spa.py
  Starts the PySide6 GUI.

spa_app/main_window.py
  Wizard interface, playback controls, saved-output resume behavior, QC audit UI.

spa_core/audio.py
  Audio reading/writing, ffprobe/ffmpeg fallback decoding, bit-depth handling,
  mono conversion, playback.

spa_core/processing.py
  MATLAB-equivalent preprocessing:
  detrend constant -> absolute value -> 30 Hz Butterworth/filtfilt envelope -> 0-100 normalization.

spa_core/thresholds.py
  Noise statistics, static threshold, adaptive 35-block threshold curve.

spa_core/segmentation.py
  MATLAB-equivalent event boundary rules, duration filtering, summary matrices.

spa_core/export.py
  MATLAB-style Excel export and RA segment-audit CSV export.

tools/
  Noninteractive folder checks and MATLAB/Python parity helpers.
```

## Core Computation Contract

Do not change these without creating a parity note and running regression checks:

- thresholding uses `SpaSignal.normalized_envelope`, not raw audio;
- static threshold is `noise_mean + noise_std * SD`;
- adaptive threshold is `noise_mean + noise_std * SD + block_std / 3`;
- adaptive mode uses 35 blocks;
- speech samples are `normalized_envelope > threshold`;
- pause samples are `normalized_envelope <= threshold`;
- event boundary behavior intentionally follows the MATLAB implementation.

## Local Setup

```bash
cd /path/to/spa_python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install FFmpeg for non-WAV media and misnamed files:

```bash
brew install ffmpeg
```

## Checks Before Pushing

```bash
source .venv/bin/activate
pytest
python -m pip check
```

Optional import smoke check:

```bash
python - <<'PY'
mods = ["numpy", "scipy", "matplotlib", "pandas", "openpyxl", "PySide6", "soundfile", "sounddevice"]
for mod in mods:
    __import__(mod)
print("imports ok")
PY
```

## Folder Smoke Check

This runs SPA over a folder noninteractively. It is useful for finding crashes
or obviously suspicious outputs. It is not a replacement for RA review because
the noise region is selected automatically.

```bash
python tools/check_audio_folder.py /path/to/audio_folder --variant both
```

Reports are written to `reports/`, which is ignored by git.

The folder checker uses `ffprobe` to decide whether a file has an audio stream.
It does not trust the extension.

## MATLAB Parity

The MATLAB parity helpers compare summary-level outputs from Python and MATLAB.
They rely on the same automatically selected quiet region used by
`tools/check_audio_folder.py`.

```bash
python tools/check_audio_folder.py /path/to/audio_folder --variant adaptive
tools/run_matlab_reference.sh reports/audio_check_<timestamp>.csv reports/matlab_reference.csv
python tools/compare_matlab_python.py reports/audio_check_<timestamp>.csv reports/matlab_reference.csv
```

When recording parity results, summarize:

- date;
- Python commit;
- MATLAB version;
- number of files compared;
- static/adaptive mode;
- pass/fail counts;
- any known mismatch reason.

## Git Hygiene

Do not commit:

- audio/media files;
- participant data;
- `spa_outputs/`;
- `outputs/`;
- `reports/`;
- `.venv/`;
- caches such as `__pycache__/`, `.pytest_cache/`, `.matplotlib/`.

The `.gitignore` is configured for these, but check `git status` before every
push.
