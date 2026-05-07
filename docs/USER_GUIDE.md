# SPA Python User Guide

This guide is for research assistants using SPA Python to review Bamboo Passage
audio files.

## 1. Install Once

Open Terminal and run:

```bash
cd /path/to/spa_python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you need to open WebM, MP4, M4A, MP3, or files that are named `.wav` but are
not actually WAV internally, install FFmpeg:

```bash
brew install ffmpeg
```

After the first setup, you only need to activate the environment and start the
app:

```bash
cd /path/to/spa_python
source .venv/bin/activate
python run_spa.py
```

## 2. Settings Page

The settings page appears once when the app starts.

Recommended defaults:

- Threshold curve: `adaptive`
- SD multiplier: `3`
- Speech minimum: `25 ms`
- Pause minimum: `250 ms`
- Analyze full file: off unless the protocol says to analyze the entire file

Use the Settings button in the top right if you need to change these later.
Changing settings for a loaded file sends you back to the boundary-selection
step so SPA can be rerun.

## 3. Load Audio

Use one of these buttons:

- Select Audio File: load one file.
- Select Multiple Audio Files: choose a manual batch.
- Select Folder: queue every file with a readable audio stream inside a folder.

The app chooses an output folder automatically:

- for a folder queue: `<selected audio folder>/spa_outputs`
- for selected files: `<common parent folder>/spa_outputs`

You can override this with Choose Output Folder.

The queue marks files as:

- `PENDING`: no output CSV exists yet.
- `DONE`: output CSV already exists in the selected output folder.

Start Next Pending File opens the first unfinished file. Open Selected File lets
you reopen any file. If the selected file is already done, the app opens the
saved segmentation review page instead of making you reselect boundaries.

## 4. Select Boundaries

The boundary page shows two aligned plots:

- top: normalized envelope used by SPA thresholding;
- bottom: original waveform.

Use the crosshair to click exact locations.

For automatic thresholding:

1. Click Select Noise Boundaries.
2. Click the start and end of a noise-only pause.
3. Adjust Noise start and Noise end with the sliders or time fields.

If Analyze full file is off by default:

1. Click Select Analysis Boundaries.
2. Click the start and end of the full section to analyze.
3. Adjust Analysis start and Analysis end with the sliders or time fields.

Use playback buttons to confirm the regions:

- Play Full Audio
- Play Noise Selection
- Play Analysis Region

The playback bar shows current time and can be dragged to move within the
current playback range. Press Space to pause or resume playback.

Click Confirm And Run SPA when the selected regions look right.

## 5. Review Segmentation

The review page shows:

- normalized signal with threshold curve;
- original waveform with detected speech and pause overlays;
- speech count, pause count, and analysis start/end times;
- Bamboo Passage text for reference.

Color guide:

- normalized signal: blue line;
- threshold: red line;
- detected speech: green overlay;
- detected pause: blue overlay.

Use Play Full Audio or Play Analysis Region if you need to listen again.
Press Space to pause or resume playback.

If the segmentation looks wrong, click Back to return to boundary selection and
adjust the noise or analysis region. If it looks acceptable, click Confirm
Segmentation.

## 6. QC Audit

The audit page walks through every detected segment in chronological order,
including both speech and pause segments.

For each segment:

1. Review the waveform and spectrogram.
2. Play the segment.
3. Select any specific QC artifacts that apply.
4. Click Next Segment.

The top plot shows the original amplitude waveform. The bottom plot is a
time-aligned spectrogram for checking frequency structure such as formants. The
red playback cursor moves across both plots.

To inspect a smaller section, click Select Zoom, then click the zoom start and
end time on either plot. Both plots will zoom to that same time window. Use Play
Zoom to play only the zoomed section, or Reset Zoom to return to the full
segment.

Use the Speed control beside Play Segment to slow down or speed up playback for
the current and later QC segments. Press Space to pause or resume segment
playback.

When you check a QC artifact, the app asks you to mark where that issue occurs.
Click the issue start and end time on either QC plot, then click Confirm
Boundary. Use Play Boundary to listen to the selected issue range before or
after confirming it. You can still play, zoom, and change speed while setting
the issue boundary. To undo the artifact, uncheck it. To adjust a saved artifact
boundary, select it in the Issue Boundary dropdown and click Edit Boundary.

The group names are labels only. Select the specific artifact checkboxes under
each group.

QC groups:

- Environmental noise: Traffic, HVAC, Pets, TV (non-speech), Beep,
  Microphone rubbing, Non-specific environmental noise
- Competing speech: TV (speech), Other human speakers
- Volume unstable: Volume too quiet, Volume too loud, Volume changes
- Clipping: Crackling on loud syllables, Grainy, sandy, buzzing voice texture,
  Whole voice sounds crushed or overloaded
- Reverberation/echo: Reverb, Echo
- Platform effects: Muffled, filtered, telephone-like; Underwater, robotic,
  warbly, codec-like
- Temporal discontinuities: Audio lagging, Audio glitching, Audio skipped/missed
- Any non-task related content: Extra or filler word, Missed word,
  Lip smacking/mouth sounds, Mouse clicking/keyboard noise, Laughing, Coughing

Use Previous Segment to go back within the current file.

## 7. Save

At the final page, click Confirm And Save. The app saves automatically. There is
no save dialog.

After saving:

- if more pending files remain, the next pending file opens automatically;
- if the queue is complete, the app returns to the Load Audio page.

You can reopen any done file from the Load Audio page to inspect saved
segmentation and QC selections.

## Supported Inputs

The app tries normal WAV reading first. If that fails, it uses `ffprobe` to find
the first audio stream and `ffmpeg` to decode that stream. This means a file can
still open even when its extension is misleading, such as a WebM recording named
`.wav`.

Folder mode also uses `ffprobe`, so it does not rely only on file extensions.
Non-audio files are skipped.

## Common Mistakes

Noise region includes speech:
The threshold can become too high or unstable. Go back and choose a quieter
pause. This can be checked by looking at the SPA output before QC.

Analysis region starts too early or ends too late:
Leading or trailing pauses may be included. This is allowed, and those pauses
are flagged in the output, but only include the intended analysis window.

Playback does not work:
Check that the computer audio output is available. If the app was launched
before headphones or speakers were connected, restart the app.

Done files are not showing as done:
Make sure the output folder selected on the Load Audio page is the same folder
that contains the previous `<file_stem>_segments.csv` files.
