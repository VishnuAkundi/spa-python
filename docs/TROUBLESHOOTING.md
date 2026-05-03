# Troubleshooting

## The App Does Not Start

Run from the project folder with the virtual environment active:

```bash
cd /path/to/spa_python
source .venv/bin/activate
python run_spa.py
```

If imports fail, reinstall dependencies:

```bash
pip install -r requirements.txt
```

## WAV Files Are Not Showing

The file picker filters for `.wav` and `.WAV`. If a file has a different audio
extension, convert it to WAV first.

## Playback Does Not Work

Playback uses `sounddevice`. Try:

1. Confirm the computer has a working audio output.
2. Connect headphones or speakers before starting the app.
3. Restart the app.
4. Run `python -m pip check` to confirm dependencies are consistent.

## The Segmentation Looks Like Everything Is Speech

Usually this means the selected noise region was not representative or the
threshold was too low.

Try:

- choose a quieter noise-only region;
- avoid including speech in the noise region;
- increase the SD multiplier in Settings;
- check whether the analysis region includes unintended content.

## Done Files Are Not Marked Done

The app checks for:

```text
<output folder>/<file_stem>_segments.csv
```

Make sure the Output Folder on the Load Audio page is the same folder that was
used previously.

## A Done File Opens But Boundaries Look Wrong

The app restores boundaries from:

```text
<output folder>/metadata_json/<file_stem>_segments_meta.json
```

If that JSON is missing, the app can still show saved segments from the CSV, but
the original noise/analysis selections may not be fully recoverable.
