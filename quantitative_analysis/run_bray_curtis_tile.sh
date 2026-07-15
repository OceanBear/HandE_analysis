#!/usr/bin/env bash
# Run bray_curtis_tile.py: overall B–C heatmap + per-group maps (default).
# Titles always include n / Mean / Median. Tile names on axes only if SHOW_TILE_NAMES=1
# (overall: also set SHOW_GROUP_NAMES_ON_AXIS=0 if you want tile names instead of group labels).
# Edit CONFIG, then: ./run_bray_curtis_tile.sh
# Or: ./run_bray_curtis_tile.sh -- ... (pass-through to Python)

set -euo pipefail

# ========== CONFIG ==========
JSON_DIR="${JSON_DIR:-/mnt/j/HandE/results/Final/pred/json}"
# Leave empty to use script default (project .../tile_categories_88_tiles.json)
TILE_CATEGORIES_JSON=""
OUTPUT_DIR=""
TYPE_INFO_JSON="${TYPE_INFO_JSON:-/home/qxiong/projects/HandE_analysis/type_info_4class.json}"
# Axis labels: 0 = no tile names (default); 1 = --show-tile-names
SHOW_TILE_NAMES=0
# Overall heatmap: 1 = group names at cluster midpoints (default); 0 = --no-show-group-names-on-axis
SHOW_GROUP_NAMES_ON_AXIS=1
# =============================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
SCRIPT="${SCRIPT_DIR}/bray_curtis_tile.py"
DEFAULT_TYPE_INFO="${SCRIPT_DIR}/../type_info_4class.json"
TYPE_INFO_JSON="${TYPE_INFO_JSON:-${DEFAULT_TYPE_INFO}}"

[[ -f "$SCRIPT" ]] || { echo "Not found: $SCRIPT" >&2; exit 1; }

if [[ $# -gt 0 ]]; then
  exec "$PY" "$SCRIPT" "$@"
fi

args=(--json-dir "$JSON_DIR")
[[ -n "$TILE_CATEGORIES_JSON" ]] && args+=(--tile-categories-json "$TILE_CATEGORIES_JSON")
[[ -n "$OUTPUT_DIR" ]] && args+=(--output-dir "$OUTPUT_DIR")
[[ -f "$TYPE_INFO_JSON" ]] && args+=(--type-info "$TYPE_INFO_JSON")
[[ "$SHOW_TILE_NAMES" == 1 ]] && args+=(--show-tile-names)
[[ "$SHOW_GROUP_NAMES_ON_AXIS" == 0 ]] && args+=(--no-show-group-names-on-axis)

exec "$PY" "$SCRIPT" "${args[@]}"
