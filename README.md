# SPA Python

SPA Python is a desktop app for Speech Pause Analysis of WAV recordings. It was
ported from the original MATLAB SPA workflow and redesigned as a guided wizard
for research assistants.

The app lets an RA:

- choose SPA settings once per session;
- load one WAV, multiple WAVs, or a folder of WAV files;
- select a noise-only region and an analysis region;
- review detected speech and pause segments;
- QC-audit every detected speech and pause segment;
- automatically save CSV outputs and metadata beside the audio files.

## Quick Start

```bash
cd /path/to/spa_python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_spa.py
```

For day-to-day lab use, see [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## Project Layout

```text
spa_python/
  run_spa.py              # GUI entry point
  spa_app/                # PySide6 desktop wizard
  spa_core/               # SPA audio, preprocessing, thresholding, segmentation, export
  tests/                  # automated tests
  tools/                  # validation and MATLAB comparison helpers
  docs/                   # user and maintainer documentation
```

## Outputs

By default, folder processing writes to a `spa_outputs/` folder inside the
selected audio folder. Each WAV gets:

- `<file_stem>_segments.csv`
- `metadata_json/<file_stem>_segments_meta.json`

The CSV includes chronological speech and pause rows, absolute onset/offset
times, pause-position flags, and QC audit selections. See
[docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md).

## Development Checks

```bash
source .venv/bin/activate
pytest
python -m pip check
```

Optional folder smoke check:

```bash
python tools/check_audio_folder.py /path/to/audio_folder
```

Optional MATLAB parity workflow:

```bash
tools/run_matlab_reference.sh reports/audio_check_<timestamp>.csv reports/matlab_reference.csv
python tools/compare_matlab_python.py reports/audio_check_<timestamp>.csv reports/matlab_reference.csv
```

More detail is in [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md).

## Scope

Included in the Python app:

- guided single-file and folder-queue workflow;
- automatic static/adaptive SPA thresholding;
- MATLAB-style preprocessing and event boundary rules;
- speech and pause QC audit;
- automatic CSV and JSON metadata output;
- saved-output review and rerun support.

Not included from the old MATLAB GUI:

- variable/iteration GUI modes;
- Spectrum/tagging workflow;
- manual segmentation workflow;
- FFT GUI;
- recording GUI.

## Data Safety

Audio files, generated outputs, local reports, virtual environments, and caches
are ignored by `.gitignore` so protected lab data is not accidentally committed.
