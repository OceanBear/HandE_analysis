#!/usr/bin/env bash
# Run unified CN detection (k-means clustering across all tiles, CSV output only).
# Edit CONFIG below, then: ./run_cn_unified_kmeans.sh
# Since --n_clusters is required (no default), this sweeps easily by re-running
# with different N_CLUSTERS values — each gets its own output subfolder.
set -euo pipefail

# ========== CONFIG ==========
TILES_DIR="/path/to/h5ad_tiles"
OUTPUT_DIR="/path/to/cn_unified_results"   # actual results land in OUTPUT_DIR/kK_nclustersN_seedS/

K=20                # number of nearest neighbors for the spatial KNN graph
N_CLUSTERS=6        # required — number of cellular neighborhoods (try 4-7)
CELLTYPE_KEY="cell_type"
PATTERN="*.h5ad"
MAX_TILES=""        # leave empty to process all tiles; set a number for quick testing
NO_OFFSET=0         # 1 = disable spatial coordinate offsetting between tiles
RANDOM_STATE=""     # leave empty to use the script's default (0)
TILE_LIST_CSV=""    # optional: CSV with a 'tile' column to restrict which
                    # tiles are included (e.g. excluding some for QC reasons).
                    # Leave empty to process all tiles found in TILES_DIR.
SAVE_COMPOSITION=0  # 1 = also save each cell's neighbor-composition vector
                    # alongside its CN label (larger output). Only needed if
                    # you plan to run cn_subcluster.py on this run's results later.

# Directory containing cn_unified_kmeans.py. Leave empty to default to
# the same folder as this .sh file; set explicitly if the script lives elsewhere.
SCRIPT_DIR=""
# =============================

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PYTHON:-python3}"
SCRIPT="${SCRIPT_DIR}/cn_unified_kmeans.py"
[[ -f "$SCRIPT" ]] || { echo "Not found: $SCRIPT" >&2; echo "Set SCRIPT_DIR in CONFIG if cn_unified_kmeans.py lives elsewhere." >&2; exit 1; }

# Pass-through mode: ./run_cn_unified_kmeans.sh -- --n_clusters 7 --max_tiles 10
if [[ $# -gt 0 ]]; then
  args=("$@")
else
  args=(
    --tiles_dir "$TILES_DIR"
    --output_dir "$OUTPUT_DIR"
    --k "$K"
    --n_clusters "$N_CLUSTERS"
    --celltype_key "$CELLTYPE_KEY"
    --pattern "$PATTERN"
  )
  [[ -n "$MAX_TILES" ]] && args+=(--max_tiles "$MAX_TILES")
  [[ "$NO_OFFSET" == 1 ]] && args+=(--no_offset)
  [[ -n "$RANDOM_STATE" ]] && args+=(--random_state "$RANDOM_STATE")
  [[ -n "$TILE_LIST_CSV" ]] && args+=(--tile_list_csv "$TILE_LIST_CSV")
  [[ "$SAVE_COMPOSITION" == 1 ]] && args+=(--save_composition)
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
