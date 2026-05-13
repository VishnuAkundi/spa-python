# Output Schema

SPA Python writes one CSV per processed audio/media file:

```text
<output folder>/<file_stem>_segments.csv
```

It also writes one metadata JSON per processed file:

```text
<output folder>/metadata_json/<file_stem>_segments_meta.json
```

While a file is unfinished, the app also writes a temporary progress JSON:

```text
<output folder>/progress_json/<file_stem>_progress.json
```

## Segment CSV

Rows are chronological. SPA mode includes speech and pause segments. Fixed-window
mode includes fixed audit segments.
Each QC group column is stored as JSON text inside the CSV cell.

| Column | Meaning |
| --- | --- |
| `file_name` | Source file name only. |
| `source_path` | Full path to the source file used for review. |
| `segment_number` | Chronological segment number within the file. |
| `segment_type` | `speech`, `pause`, or `fixed`. `fixed` means the app skipped SPA and split the file into 5-second audit windows. |
| `pause_position` | `leading`, `trailing`, `leading_trailing`, or `0`. Only pause rows can be flagged. |
| `onset_seconds_absolute` | Segment onset in seconds from the beginning of the original source file. |
| `offset_seconds_absolute` | Segment offset in seconds from the beginning of the original source file. |
| `duration_seconds` | Segment duration in seconds. |
| `Environmental noise` | JSON dictionary for that QC group. Each effect is present. Values are `[]` or a list of `[issue_onset_seconds_absolute, issue_offset_seconds_absolute]` windows. |
| `Any non-task related content` | Same format. |
| `Competing speech` | Same format. |
| `Volume unstable` | Same format. |
| `Clipping` | Same format. |
| `Reverberation/echo` | Same format. |
| `Platform effects` | Same format. |
| `Temporal discontinuities` | Same format. |

Example QC columns:

```json
"Environmental noise": {
  "Traffic": [[2.55, 3.04], [6.10, 6.44]],
  "HVAC": [],
  "Pets": [],
  "TV (non-speech)": [],
  "Beep": [],
  "Microphone rubbing": [],
  "Non-specific environmental noise": []
}
```

If no effects are selected in a group, every value in that group dictionary is
an empty list:

```json
{
  "TV (speech)": [],
  "Other human speakers": []
}
```

## Pause Position

`pause_position` marks whether a pause touches the edge of the analysis region:

- `leading`: first chronological segment is a pause.
- `trailing`: last chronological segment is a pause.
- `leading_trailing`: the only segment is a pause.
- `0`: not an edge pause, or the row is speech/fixed.

## Metadata JSON

The metadata JSON is used by the app to reopen completed files without forcing
the RA to reselect boundaries.

Fields:

- `file_name`
- `source_path`
- `sample_rate`
- `noise_region_samples`
- `analysis_region_samples`
- `file_level_qc`: review-page QC selections that are projected into individual segment rows.
- `settings`

The CSV is the primary analysis output. The metadata JSON is completed-file app
state needed for review.

## Temporary Progress JSON

The progress JSON is an autosave for interrupted files. It can include:

- current wizard page;
- selected noise and analysis boundaries;
- segmentation mode and detected segment boundaries;
- file-level QC selections;
- segment-by-segment QC selections already entered;
- current audit segment index.

Progress JSON files are not final analysis outputs. They are deleted
automatically once Confirm And Save successfully writes the final CSV for that
file.
