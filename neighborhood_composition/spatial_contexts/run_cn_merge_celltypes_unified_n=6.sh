#!/usr/bin/env bash
# Merge epithelium -> tumor in tile h5ads, then unified CN. Edit CONFIG, then: ./run_cn_merge_celltypes_unified.sh
# Or: ./run_cn_merge_celltypes_unified.sh -- ... (pass-through to Python)

set -euo pipefail

# ========== CONFIG ==========
SOURCE_TILES_DIR="/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred_03_26/h5ad"
MERGED_TILES_DIR="/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred_03_26/h5ad_tumor_merged_n=6"
OUTPUT_DIR="/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred_03_26/cn_unified_results_tumor_merged_n=6"
K=20
N_CLUSTERS=6
CELLTYPE_KEY="cell_type"
PATTERN="*.h5ad"
MAX_TILES=""
NO_OFFSET=0
RANDOM_STATE=""
SKIP_PHASE_A=0   # 1 = only re-run CN from existing MERGED_TILES_DIR
# Optional JSON overrides (leave empty)
MERGE_MAP_JSON=""
CATEGORY_ORDER_JSON=""
# =============================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
SCRIPT="${SCRIPT_DIR}/cn_merge_celltypes_unified.py"

[[ -f "$SCRIPT" ]] || { echo "Not found: $SCRIPT" >&2; exit 1; }

if [[ $# -gt 0 ]]; then
  exec "$PY" "$SCRIPT" "$@"
fi

args=(
  --source_tiles_dir "$SOURCE_TILES_DIR"
  --merged_tiles_dir "$MERGED_TILES_DIR"
  --output_dir "$OUTPUT_DIR"
  --k "$K"
  --n_clusters "$N_CLUSTERS"
  --celltype_key "$CELLTYPE_KEY"
  --pattern "$PATTERN"
)
[[ -n "$MAX_TILES" ]] && args+=(--max_tiles "$MAX_TILES")
[[ "$NO_OFFSET" == 1 ]] && args+=(--no_offset)
[[ -n "$RANDOM_STATE" ]] && args+=(--random_state "$RANDOM_STATE")
[[ "$SKIP_PHASE_A" == 1 ]] && args+=(--skip_phase_a)
[[ -n "$MERGE_MAP_JSON" ]] && args+=(--merge_map_json "$MERGE_MAP_JSON")
[[ -n "$CATEGORY_ORDER_JSON" ]] && args+=(--category_order_json "$CATEGORY_ORDER_JSON")

exec "$PY" "$SCRIPT" "${args[@]}"
