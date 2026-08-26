#!/usr/bin/env bash
# Run group-based CN analysis (unified composition/frequency + per-group comparisons).
# Edit CONFIG below, then: ./run_vis_kmeans.sh
set -euo pipefail

# ========== CONFIG ==========
SOURCE_H5AD_DIR="/path/to/source_h5ad_tiles"       # original tiles from data_preparation.py
CN_LABELS_DIR="/path/to/cn_unified_results/kNN_nclustersNN_seedN/cn_labels"
CATEGORIES_JSON="/path/to/tile_categories.json"
OUTPUT_DIR="/path/to/cn_group_results"

CELLTYPE_KEY="cell_type"
COLOR_PALETTE="tab20"

K=20                # for titles only
N_CLUSTERS=""       # leave empty to auto-infer from data/path
GROUP=""            # leave empty to analyze all groups; set a name to run just one
TILE_LIST_CSV=""    # optional: CSV with a 'tile' column to further restrict
                    # which tiles are included, independent of categories.json
                    # or which tiles have CN labels. Leave empty to use all.
GENERATE_UNIFIED=1  # 1 = generate unified analysis (composition heatmap +
                    #     overall/per-tile frequency charts), 0 = skip it.
                    # Set to 0 if you've already run this once and just want
                    # to re-run the per-group comparisons (e.g. after editing
                    # categories.json), to save re-doing the unified step.

# Directory containing vis_kmeans.py. Leave empty to default to the same
# folder as this .sh file; set explicitly if the script lives elsewhere.
SCRIPT_DIR=""
# =============================

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PYTHON:-python3}"
SCRIPT="${SCRIPT_DIR}/vis_kmeans.py"
[[ -f "$SCRIPT" ]] || { echo "Not found: $SCRIPT" >&2; echo "Set SCRIPT_DIR in CONFIG if vis_kmeans.py lives elsewhere." >&2; exit 1; }

# Pass-through mode: ./run_vis_kmeans.sh -- --custom-flag value
if [[ $# -gt 0 ]]; then
  args=("$@")
else
  args=(
    --source_h5ad_dir "$SOURCE_H5AD_DIR"
    --cn_labels_dir "$CN_LABELS_DIR"
    --categories_json "$CATEGORIES_JSON"
    --output_dir "$OUTPUT_DIR"
    --celltype_key "$CELLTYPE_KEY"
    --color_palette "$COLOR_PALETTE"
    --k "$K"
  )
  [[ -n "$N_CLUSTERS" ]] && args+=(--n_clusters "$N_CLUSTERS")
  [[ -n "$GROUP" ]] && args+=(--group "$GROUP")
  [[ -n "$TILE_LIST_CSV" ]] && args+=(--tile_list_csv "$TILE_LIST_CSV")
  [[ "$GENERATE_UNIFIED" == 0 ]] && args+=(--no-generate_unified)
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
