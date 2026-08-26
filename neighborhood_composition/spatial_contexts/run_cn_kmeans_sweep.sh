#!/usr/bin/env bash
# Sweep n_clusters for cellular neighborhood detection (elbow method / silhouette
# score diagnostics), computing the expensive load/KNN-graph/aggregation steps
# only ONCE instead of once per n_clusters value.
# Edit CONFIG below, then: ./run_cn_kmeans_sweep.sh
set -euo pipefail

# ========== CONFIG ==========
TILES_DIR="/path/to/h5ad_tiles"
OUTPUT_DIR="/path/to/cn_sweep_results"   # each n_clusters gets its own kK_nclustersN_seedS/ subfolder,
                                          # plus a top-level sweep_summary.csv comparing all of them

K=20                # number of nearest neighbors for the spatial KNN graph
N_START=3           # first n_clusters value to test (inclusive)
N_END=8             # last n_clusters value to test (inclusive)
N_STEP=1            # step between n_clusters values

CELLTYPE_KEY="cell_type"
PATTERN="*.h5ad"
MAX_TILES=""        # leave empty to process all tiles; set a number for quick testing
NO_OFFSET=0         # 1 = disable spatial coordinate offsetting between tiles
RANDOM_STATE=""     # leave empty to use the script's default (0)
TILE_LIST_CSV=""    # optional: CSV with a 'tile' column to restrict which tiles are included

SILHOUETTE_SAMPLE_SIZE=20000  # max cells subsampled for the silhouette score
                               # (exact computation is O(n^2), infeasible on full datasets)

# Directory containing cn_kmeans_sweep.py (and cn_unified_kmeans.py,
# which it imports from). Leave empty to default to the same folder as this
# .sh file; set explicitly if the scripts live elsewhere.
SCRIPT_DIR=""
# =============================

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PYTHON:-python3}"
SCRIPT="${SCRIPT_DIR}/cn_kmeans_sweep.py"
[[ -f "$SCRIPT" ]] || { echo "Not found: $SCRIPT" >&2; echo "Set SCRIPT_DIR in CONFIG if cn_kmeans_sweep.py lives elsewhere." >&2; exit 1; }
[[ -f "${SCRIPT_DIR}/cn_unified_kmeans.py" ]] || { echo "Not found: ${SCRIPT_DIR}/cn_unified_kmeans.py (required import)" >&2; exit 1; }

# Pass-through mode: ./run_cn_kmeans_sweep.sh -- --n_start 4 --n_end 10
if [[ $# -gt 0 ]]; then
  args=("$@")
else
  args=(
    --tiles_dir "$TILES_DIR"
    --output_dir "$OUTPUT_DIR"
    --k "$K"
    --n_start "$N_START"
    --n_end "$N_END"
    --n_step "$N_STEP"
    --celltype_key "$CELLTYPE_KEY"
    --pattern "$PATTERN"
    --silhouette_sample_size "$SILHOUETTE_SAMPLE_SIZE"
  )
  [[ -n "$MAX_TILES" ]] && args+=(--max_tiles "$MAX_TILES")
  [[ "$NO_OFFSET" == 1 ]] && args+=(--no_offset)
  [[ -n "$RANDOM_STATE" ]] && args+=(--random_state "$RANDOM_STATE")
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
