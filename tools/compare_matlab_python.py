from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


COUNT_COLUMNS = ["speech_events", "pause_events"]
NUMERIC_COLUMNS = [
    "threshold",
    "pct_speech",
    "pct_pause",
    "speech_duration",
    "pause_duration",
    "total_duration",
]


def compare(python_csv: Path, matlab_csv: Path, output_dir: Path, numeric_tol: float) -> tuple[Path, Path]:
    py = pd.read_csv(python_csv)
    ml = pd.read_csv(matlab_csv)
    merged = py.merge(ml, on=["file", "variant"], suffixes=("_python", "_matlab"), how="outer", indicator=True)

    rows = []
    for _, row in merged.iterrows():
        item = {
            "file": row.get("file"),
            "variant": row.get("variant"),
            "merge_status": row["_merge"],
            "pass": True,
            "message": "",
        }
        messages = []
        if row["_merge"] != "both":
            item["pass"] = False
            messages.append(f"row present in {row['_merge']}")
        else:
            py_status = row.get("status_python")
            ml_status = row.get("status_matlab")
            item["status_python"] = py_status
            item["status_matlab"] = ml_status
            if pd.notna(ml_status) and ml_status != "ok":
                item["pass"] = False
                messages.append(f"MATLAB status={ml_status}")
            if pd.notna(py_status) and py_status == "error":
                item["pass"] = False
                messages.append(f"Python status={py_status}")

            for column in COUNT_COLUMNS:
                py_value = row[f"{column}_python"]
                ml_value = row[f"{column}_matlab"]
                diff = py_value - ml_value
                item[f"{column}_python"] = py_value
                item[f"{column}_matlab"] = ml_value
                item[f"{column}_diff"] = diff
                if pd.isna(diff):
                    item["pass"] = False
                    messages.append(f"{column} missing")
                elif diff != 0:
                    item["pass"] = False
                    messages.append(f"{column} diff={diff}")

            for column in NUMERIC_COLUMNS:
                py_value = row[f"{column}_python"]
                ml_value = row[f"{column}_matlab"]
                diff = py_value - ml_value
                abs_diff = abs(diff) if pd.notna(diff) else np.nan
                item[f"{column}_python"] = py_value
                item[f"{column}_matlab"] = ml_value
                item[f"{column}_diff"] = diff
                if pd.isna(abs_diff):
                    item["pass"] = False
                    messages.append(f"{column} missing")
                elif abs_diff > numeric_tol:
                    item["pass"] = False
                    messages.append(f"{column} abs diff={abs_diff:.6g}")
        item["message"] = "; ".join(messages)
        rows.append(item)

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"matlab_python_compare_{timestamp}.csv"
    md_path = output_dir / f"matlab_python_compare_{timestamp}.md"
    result.to_csv(csv_path, index=False)

    failures = result[~result["pass"]]
    with md_path.open("w") as handle:
        handle.write("# MATLAB vs Python SPA Comparison\n\n")
        handle.write(f"- Python CSV: `{python_csv}`\n")
        handle.write(f"- MATLAB CSV: `{matlab_csv}`\n")
        handle.write(f"- Rows compared: {len(result)}\n")
        handle.write(f"- Passing rows: {int(result['pass'].sum())}\n")
        handle.write(f"- Failing rows: {len(failures)}\n")
        handle.write(f"- Numeric tolerance: {numeric_tol}\n\n")
        if failures.empty:
            handle.write("All compared rows passed.\n")
        else:
            handle.write("## Failures\n\n")
            for _, row in failures.head(100).iterrows():
                handle.write(f"- `{Path(str(row['file'])).name}` ({row['variant']}): {row['message']}\n")
            if len(failures) > 100:
                handle.write(f"\n... {len(failures) - 100} more failures in the CSV.\n")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Python and MATLAB SPA summary CSVs.")
    parser.add_argument("python_csv", type=Path)
    parser.add_argument("matlab_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports")
    parser.add_argument("--numeric-tol", type=float, default=1e-6)
    args = parser.parse_args()

    csv_path, md_path = compare(args.python_csv, args.matlab_csv, args.output_dir, args.numeric_tol)
    print(f"CSV: {csv_path}")
    print(f"Notes: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
