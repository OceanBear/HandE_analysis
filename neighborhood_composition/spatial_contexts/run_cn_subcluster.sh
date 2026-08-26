#!/usr/bin/env bash
# Sub-cluster selected CNs (e.g. split CN1 and CN3 into finer sub-groups) using
# the composition vectors from a parent cn_unified_kmeans.py run.
# Requires that parent run to have been made with --save_composition.
# Edit CONFIG below, then: ./run_cn_subcluster.sh
set -euo pipefail

# ========== CONFIG ==========
SOURCE_H5AD_DIR="/path/to/source_h5ad_tiles"       # original tiles from data_preparation.py
CN_LABELS_DIR="/path/to/cn_unified_results/kNN_nclustersNN_seedN/cn_labels"  # parent run's cn_labels/
                                                    # (must have been generated with --save_composition)
OUTPUT_DIR="/path/to/cn_unified_results/kNN_nclustersNN_seedN_sub"

# Comma-separated 'CN:n_divisions' pairs, e.g. "1:2,3:3" splits CN1 into 2
# children and CN3 into 3 children. Any CN not listed here is left unchanged.
SUBCLUSTER_CONFIG="1:2,3:2"

CELLTYPE_KEY="cell_type"
RANDOM_STATE=""     # leave empty to use the script's default (0)
TILE_LIST_CSV=""    # optional: CSV with a 'tile' column to restrict which tiles are included

# Directory containing cn_subcluster.py. Leave empty to default to the same
# folder as this .sh file; set explicitly if the script lives elsewhere.
SCRIPT_DIR=""
# =============================

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PYTHON:-python3}"
SCRIPT="${SCRIPT_DIR}/cn_subcluster.py"
[[ -f "$SCRIPT" ]] || { echo "Not found: $SCRIPT" >&2; echo "Set SCRIPT_DIR in CONFIG if cn_subcluster.py lives elsewhere." >&2; exit 1; }

# Pass-through mode: ./run_cn_subcluster.sh -- --subcluster_config "2:2"
if [[ $# -gt 0 ]]; then
  args=("$@")
else
  args=(
    --source_h5ad_dir "$SOURCE_H5AD_DIR"
    --cn_labels_dir "$CN_LABELS_DIR"
    --output_dir "$OUTPUT_DIR"
    --subcluster_config "$SUBCLUSTER_CONFIG"
    --celltype_key "$CELLTYPE_KEY"
  )
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
