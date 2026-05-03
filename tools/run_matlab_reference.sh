#!/usr/bin/env bash
set -euo pipefail

MATLAB_APP="${MATLAB_APP:-/Applications/MATLAB_R2024b.app}"
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$TOOLS_DIR/.." && pwd)"

INPUT_CSV="${1:-$PROJECT_DIR/reports/audio_check_20260429_172838.csv}"
OUTPUT_CSV="${2:-$PROJECT_DIR/reports/matlab_reference_$(date +%Y%m%d_%H%M%S).csv}"
MAX_ROWS="${3:-inf}"

abs_existing_file() {
  local path="$1"
  local dir
  dir="$(cd "$(dirname "$path")" && pwd)"
  printf "%s/%s" "$dir" "$(basename "$path")"
}

abs_output_file() {
  local path="$1"
  local dir
  mkdir -p "$(dirname "$path")"
  dir="$(cd "$(dirname "$path")" && pwd)"
  printf "%s/%s" "$dir" "$(basename "$path")"
}

if [[ ! -d "$MATLAB_APP" ]]; then
  echo "MATLAB app not found: $MATLAB_APP" >&2
  exit 1
fi

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input CSV not found: $INPUT_CSV" >&2
  exit 1
fi

INPUT_CSV="$(abs_existing_file "$INPUT_CSV")"
OUTPUT_CSV="$(abs_output_file "$OUTPUT_CSV")"

MATLAB_CMD="addpath('$TOOLS_DIR'); matlab_reference_spa('$INPUT_CSV','$OUTPUT_CSV',$MAX_ROWS)"

echo "Running MATLAB reference generation..."
echo "MATLAB: $MATLAB_APP"
echo "Input:  $INPUT_CSV"
echo "Output: $OUTPUT_CSV"

cd "$MATLAB_APP"
if [[ -d "$MATLAB_APP/bin/maci64" && ! -d "$MATLAB_APP/bin/maca64" ]]; then
  arch -x86_64 ./bin/matlab -batch "$MATLAB_CMD"
else
  ./bin/matlab -batch "$MATLAB_CMD"
fi

echo "Done: $OUTPUT_CSV"
