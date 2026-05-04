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

## Audio Files Are Not Showing

Folder mode uses `ffprobe` to find files with an audio stream. If files are not
showing, install FFmpeg and restart the app:

```bash
brew install ffmpeg
```

The file picker also has an All files option if a file has an unusual extension.

## A File Named `.wav` Will Not Open

Some recording systems create WebM, MP4, or another container but save it with a
`.wav` name. SPA Python can open these only if FFmpeg is installed. The app uses
`ffprobe` to inspect the real file contents and `ffmpeg` to decode the first
audio stream.

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
