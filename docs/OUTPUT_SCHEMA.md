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
| `audit_json` | JSON object containing selected QC effects. |

Example `audit_json`:

```json
{
  "selected_effects": [
    {
      "gui_name": "Environmental noise",
      "effect": "Traffic"
    },
    {
      "gui_name": "Volume unstable",
      "effect": "Volume too quiet"
    }
  ]
}
```

If no QC effects are selected, the value is:

```json
{"selected_effects": []}
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
