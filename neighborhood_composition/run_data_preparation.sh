#!/usr/bin/env bash
# Run neighborhood_composition/data_preparation.py (JSON -> AnnData / h5ad).
#
# ---------------------------------------------------------------------------
# CONFIG — edit the variables below, then run with no arguments:
#   ./run_data_preparation.sh
# Or explicitly:
#   ./run_data_preparation.sh run
#
# Optional: override on the command line (same as before):
#   ./run_data_preparation.sh batch JSON_DIR [OUTPUT_DIR]
#   ./run_data_preparation.sh single JSON_PATH [OUTPUT_H5AD]
#   ./run_data_preparation.sh combine OUTPUT_H5AD JSON1.json JSON2.json ...
#   ./run_data_preparation.sh help
#
# You can also set PYTHON=/path/to/python before invoking.
# ---------------------------------------------------------------------------

set -euo pipefail

# ========== CONFIG (edit here) ==========
# Mode: batch | single | combine
MODE="batch"

# 4-class model: Others, Tumor, Lymphocyte, Fibroblast/Stroma
# (resolved relative to SCRIPT_DIR after it is set below)
TYPE_INFO_JSON=""

# Skip JSON files whose .h5ad already exists (1 = skip, 0 = overwrite)
SKIP_EXISTING=1

# --- batch: all *.json in JSON_DIR → OUTPUT_DIR (leave OUTPUT_DIR empty to write next to each JSON) ---
JSON_DIR="${JSON_DIR:-/mnt/f/data/HandE/sow1885_n201/nucsegai_pred_1003/json_filtered}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/f/data/HandE/sow1885_n201/nucsegai_pred_1003/h5ad_filtered}"

# --- single: one JSON file ---
JSON_PATH=""
OUTPUT_H5AD=""

# --- combine: one h5ad from several JSONs (list paths in the array) ---
COMBINED_H5AD="combined_tiles.h5ad"
JSON_LIST=(
  # "/path/to/tile1.json"
  # "/path/to/tile2.json"
)
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
PREP="${SCRIPT_DIR}/data_preparation.py"
TYPE_INFO_JSON="${TYPE_INFO_JSON:-${SCRIPT_DIR}/../type_info_4class.json}"

type_info_args() {
  if [[ -f "${TYPE_INFO_JSON}" ]]; then
    echo --type-info "${TYPE_INFO_JSON}"
  else
    echo "TYPE_INFO_JSON not found: ${TYPE_INFO_JSON}" >&2
    exit 1
  fi
}

skip_args() {
  # Default: skip existing .h5ad; set SKIP_EXISTING=0 to pass --overwrite
  if [[ "${SKIP_EXISTING}" == "0" ]]; then
    echo --overwrite
  fi
}

usage() {
  echo "Usage:"
  echo "  Edit CONFIG at the top of this script, then:"
  echo "    $0              # run using CONFIG"
  echo "    $0 run          # same"
  echo ""
  echo "  Or pass arguments (override CONFIG):"
  echo "    $0 batch [JSON_DIR] [OUTPUT_DIR]"
  echo "    $0 single JSON_PATH [OUTPUT_H5AD]"
  echo "    $0 combine OUTPUT_H5AD JSON1.json JSON2.json ..."
  exit "${1:-0}"
}

[[ -f "$PREP" ]] || { echo "Not found: $PREP" >&2; exit 1; }

run_from_config() {
  case "$MODE" in
    batch)
      if [[ -z "${JSON_DIR}" ]]; then
        echo "CONFIG: MODE=batch requires JSON_DIR to be set." >&2
        exit 1
      fi
      if [[ -n "${OUTPUT_DIR}" ]]; then
        exec "$PY" "$PREP" --mode batch --json-dir "$JSON_DIR" --output-dir "$OUTPUT_DIR" $(type_info_args) $(skip_args)
      else
        exec "$PY" "$PREP" --mode batch --json-dir "$JSON_DIR" $(type_info_args) $(skip_args)
      fi
      ;;
    single)
      if [[ -z "${JSON_PATH}" ]]; then
        echo "CONFIG: MODE=single requires JSON_PATH to be set." >&2
        exit 1
      fi
      if [[ -n "${OUTPUT_H5AD}" ]]; then
        exec "$PY" "$PREP" --mode single --json "$JSON_PATH" --output "$OUTPUT_H5AD" $(type_info_args) $(skip_args)
      else
        exec "$PY" "$PREP" --mode single --json "$JSON_PATH" $(type_info_args) $(skip_args)
      fi
      ;;
    combine)
      if [[ -z "${COMBINED_H5AD}" ]]; then
        echo "CONFIG: MODE=combine requires COMBINED_H5AD to be set." >&2
        exit 1
      fi
      if [[ ${#JSON_LIST[@]} -eq 0 ]]; then
        echo "CONFIG: MODE=combine requires JSON_LIST array with at least one path." >&2
        exit 1
      fi
      exec "$PY" "$PREP" --mode combine --combined-output "$COMBINED_H5AD" --json-list "${JSON_LIST[@]}" $(type_info_args) $(skip_args)
      ;;
    *)
      echo "CONFIG: MODE must be batch, single, or combine (got: ${MODE})" >&2
      exit 1
      ;;
  esac
}

# No args, or explicit "run" → use CONFIG block above
if [[ $# -eq 0 ]] || [[ "${1:-}" == "run" ]]; then
  run_from_config
fi

MODE_CLI="${1:-}"
shift || true

case "$MODE_CLI" in
  -h|--help|help)
    usage 0
    ;;
  batch)
    JSON_DIR="${1:-${JSON_DIR:-}}"
    OUT_DIR="${2:-${OUTPUT_DIR:-}}"
    if [[ -z "$JSON_DIR" ]]; then
      echo "batch: provide JSON_DIR or set JSON_DIR in CONFIG / environment." >&2
      usage 1
    fi
    if [[ -n "$OUT_DIR" ]]; then
      exec "$PY" "$PREP" --mode batch --json-dir "$JSON_DIR" --output-dir "$OUT_DIR" $(type_info_args) $(skip_args)
    else
      exec "$PY" "$PREP" --mode batch --json-dir "$JSON_DIR" $(type_info_args) $(skip_args)
    fi
    ;;
  single)
    JSON_PATH="${1:-}"
    OUT_H5AD="${2:-}"
    [[ -n "$JSON_PATH" ]] || { echo "single: require JSON_PATH" >&2; usage 1; }
    if [[ -n "$OUT_H5AD" ]]; then
      exec "$PY" "$PREP" --mode single --json "$JSON_PATH" --output "$OUT_H5AD" $(type_info_args) $(skip_args)
    else
      exec "$PY" "$PREP" --mode single --json "$JSON_PATH" $(type_info_args) $(skip_args)
    fi
    ;;
  combine)
    COMBINED="${1:-}"
    shift || true
    [[ -n "$COMBINED" && $# -ge 1 ]] || { echo "combine: require OUTPUT_H5AD and at least one JSON" >&2; usage 1; }
    exec "$PY" "$PREP" --mode combine --combined-output "$COMBINED" --json-list "$@" $(type_info_args) $(skip_args)
    ;;
  *)
    echo "Unknown command: $MODE_CLI" >&2
    echo "Use no arguments to run from CONFIG, or: batch | single | combine | help" >&2
    usage 1
    ;;
esac
