# Output Schema

SPA Python writes one CSV per processed audio/media file:

```text
<output folder>/<file_stem>_segments.csv
```

It also writes one metadata JSON per processed file:

```text
<output folder>/metadata_json/<file_stem>_segments_meta.json
```

## Segment CSV

Rows are chronological and include both speech and pause segments.
Each QC group column is stored as JSON text inside the CSV cell.

| Column | Meaning |
| --- | --- |
| `file_name` | Source file name only. |
| `source_path` | Full path to the source file used for review. |
| `segment_number` | Chronological segment number within the file. |
| `segment_type` | `speech` or `pause`. |
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
- `0`: not an edge pause, or the row is speech.

## Metadata JSON

The metadata JSON is used by the app to reopen completed files without forcing
the RA to reselect boundaries.

Fields:

- `file_name`
- `source_path`
- `sample_rate`
- `noise_region_samples`
- `analysis_region_samples`
- `settings`

The CSV is the primary analysis output. The JSON is app state needed for resume
and review.
