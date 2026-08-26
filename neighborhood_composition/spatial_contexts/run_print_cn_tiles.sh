#!/usr/bin/env bash
# Generate individual per-tile spatial CN maps (the slow, skippable step).
# Edit CONFIG below, then: ./run_print_cn_tiles.sh
set -euo pipefail

# ========== CONFIG ==========
SOURCE_H5AD_DIR="/path/to/source_h5ad_tiles"       # original tiles from data_preparation.py
CN_LABELS_DIR="/path/to/cn_unified_results/kNN_nclustersNN_seedN/cn_labels"
OUTPUT_DIR="/path/to/cn_group_results/individual_tiles"

COORD_KEY="spatial"
POINT_SIZE=10.0
PALETTE="tab20"

K=""            # leave empty to omit from titles
N_CLUSTERS=""   # leave empty to auto-infer per tile
TILE_LIST_CSV="" # optional: CSV with a 'tile' column to plot only a subset
                 # of tiles (e.g. a quick spot-check). Leave empty to plot all.

# Directory containing print_cn_tiles.py. Leave empty to default to the same
# folder as this .sh file; set explicitly if the script lives elsewhere.
SCRIPT_DIR=""
# =============================

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PYTHON:-python3}"
SCRIPT="${SCRIPT_DIR}/print_cn_tiles.py"
[[ -f "$SCRIPT" ]] || { echo "Not found: $SCRIPT" >&2; echo "Set SCRIPT_DIR in CONFIG if print_cn_tiles.py lives elsewhere." >&2; exit 1; }

# Pass-through mode: ./run_print_cn_tiles.sh -- --custom-flag value
if [[ $# -gt 0 ]]; then
  args=("$@")
else
  args=(
    --source_h5ad_dir "$SOURCE_H5AD_DIR"
    --cn_labels_dir "$CN_LABELS_DIR"
    --output_dir "$OUTPUT_DIR"
    --coord_key "$COORD_KEY"
    --point_size "$POINT_SIZE"
    --palette "$PALETTE"
  )
  [[ -n "$K" ]] && args+=(--k "$K")
  [[ -n "$N_CLUSTERS" ]] && args+=(--n_clusters "$N_CLUSTERS")
  [[ -n "$TILE_LIST_CSV" ]] && args+=(--tile_list_csv "$TILE_LIST_CSV")
fi

START_TIME=$(date +%s)
echo "Started:  $(date)"

if caffeinate -i "$PY" "$SCRIPT" "${args[@]}"; then
  STATUS=0
else
  STATUS=$?
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Finished: $(date)"
printf 'Elapsed:  %02d:%02d:%02d (hh:mm:ss)\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))

exit $STATUS
