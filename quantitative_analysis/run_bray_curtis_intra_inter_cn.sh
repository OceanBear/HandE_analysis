#!/usr/bin/env bash
# Bray-Curtis dissimilarity from CN composition (intra-case and inter-case
# pairwise comparisons), reading from an existing neighborhood_frequency_per_tile.csv
# rather than raw per-nucleus JSON.
# Edit CONFIG below, then: ./run_bray_curtis_intra_inter_cn.sh
set -euo pipefail

# ========== CONFIG ==========
# Path to neighborhood_frequency_per_tile.csv (tile x CN-proportion table)
# from cn_unified_kmeans.py or vis_kmeans.py.
FREQUENCY_CSV="/path/to/cn_unified_results/kNN_nclustersNN_seedN/unified_analysis/neighborhood_frequency_per_tile.csv"

# CSV mapping tile ID -> group (tumour/margin/bg), same format used by
# bray_curtis_intra_inter.py and vis_kmeans.py's tile categories.
GROUP_CSV="/path/to/groups.csv"

OUTPUT_DIR="/path/to/cn_bray_curtis_results"

# Directory containing bray_curtis_intra_inter_cn.py. Leave empty to default
# to the same folder as this .sh file; set explicitly if the script lives
# elsewhere.
SCRIPT_DIR=""
# =============================

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PYTHON:-python3}"
SCRIPT="${SCRIPT_DIR}/bray_curtis_intra_inter_cn.py"
[[ -f "$SCRIPT" ]] || { echo "Not found: $SCRIPT" >&2; echo "Set SCRIPT_DIR in CONFIG if bray_curtis_intra_inter_cn.py lives elsewhere." >&2; exit 1; }

# Pass-through mode: ./run_bray_curtis_intra_inter_cn.sh -- --frequency-csv path --group-csv path --output-dir path
if [[ $# -gt 0 ]]; then
  args=("$@")
else
  args=(
    --frequency-csv "$FREQUENCY_CSV"
    --group-csv "$GROUP_CSV"
    --output-dir "$OUTPUT_DIR"
  )
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
