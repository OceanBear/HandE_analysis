#!/usr/bin/env bash
# Wrapper for cn_unified_kmeans.py — edit CONFIG, then: ./run_cn_unified_kmeans.sh
# Or pass flags directly: ./run_cn_unified_kmeans.sh --tiles_dir /path --n_clusters 7

set -euo pipefail

# ========== CONFIG ==========
TILES_DIR="/mnt/f/data/HandE/sow1885_n201/nucsegai_pred_1003/h5ad_filtered"
OUTPUT_DIR="cn_unified_results"
K=20
N_CLUSTERS=4
CELLTYPE_KEY="cell_type"
PATTERN="*.h5ad"
MAX_TILES=""       # e.g. 3 for testing; empty = all tiles
NO_OFFSET=0        # 1 to pass --no_offset
RANDOM_STATE=""    # empty = Python default seed
# =============================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
SCRIPT="${SCRIPT_DIR}/cn_unified_kmeans.py"

[[ -f "$SCRIPT" ]] || { echo "Not found: $SCRIPT" >&2; exit 1; }

if [[ $# -gt 0 ]]; then
  exec "$PY" "$SCRIPT" "$@"
fi

args=(--tiles_dir "$TILES_DIR" --output_dir "$OUTPUT_DIR" --k "$K" --n_clusters "$N_CLUSTERS"
      --celltype_key "$CELLTYPE_KEY" --pattern "$PATTERN")
[[ -n "$MAX_TILES" ]] && args+=(--max_tiles "$MAX_TILES")
[[ "$NO_OFFSET" == 1 ]] && args+=(--no_offset)
[[ -n "$RANDOM_STATE" ]] && args+=(--random_state "$RANDOM_STATE")

exec "$PY" "$SCRIPT" "${args[@]}"