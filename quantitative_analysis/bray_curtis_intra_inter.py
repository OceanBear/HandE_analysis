"""
Bray-Curtis Dissimilarity - Intra-case and Inter-case Pairwise Comparisons

Usage:
    python bray_curtis_intra_inter.py --json-dir DIR --output-dir DIR --group-csv FILE [--min-prob FLOAT]

Required:
    --json-dir    Directory containing per-tile nuc JSON files.
    --output-dir  Directory to save output CSV files.
    --group-csv   CSV mapping tile ID (no .json extension) to a 'group' column
                  (values: tumour, margin, bg). Tiles missing from this file,
                  or labeled 'bg', are excluded from analysis.

Optional:
    --type-info   Path to type_info JSON (default: project root type_info_4class.json).
    --min-prob    Confidence threshold; low-confidence nuclei are reclassified as Others.

Computes pairwise Bray-Curtis dissimilarity across all valid (tumour/margin)
tiles, then splits the results into two long-format CSVs:
    intratumour_BCD.csv  - pairs within the same case
    intertumour_BCD.csv  - pairs across different cases
"""

import argparse
import itertools
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from cell_type_utils import DEFAULT_TYPE_INFO_PATH, load_tile_proportions, resolve_cell_type_config

# --------------------------------------------------
# Config
# --------------------------------------------------
# --json-dir and --output-dir are now required command-line arguments
# (see main() below) instead of hardcoded paths.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CELL_TYPE_DICT: dict[int, str] = {}
CELL_TYPE_IDS: list[int] = []


def configure_cell_types(type_info_path=None):
    global CELL_TYPE_DICT, CELL_TYPE_IDS
    cell_type_dict, _, cell_type_ids, _ = resolve_cell_type_config(type_info_path)
    CELL_TYPE_DICT = cell_type_dict
    CELL_TYPE_IDS = cell_type_ids
    return cell_type_dict, cell_type_ids


configure_cell_types()


def extract_case_id(tile_id):
    """Example: JN_TS_001_tile_10009_14592 -> JN_TS_001"""
    match = re.match(r"(.+?)_tile_", tile_id)
    if match:
        return match.group(1)
    return None


def load_group_lookup(group_csv_path):
    """
    Load the tile-id -> group mapping from a CSV file.

    Expects a header row; uses the first column as the tile ID (matching the
    JSON filename minus the '.json' extension, e.g. 'JN_TS_001_tile_10009_14592')
    and a column named 'group' with values such as 'tumour', 'margin', 'bg'.

    Returns:
        dict: {tile_id: group_string}
    """
    df = pd.read_csv(group_csv_path)
    id_col = df.columns[0]
    if "group" not in df.columns:
        raise ValueError(
            f"Expected a 'group' column in {group_csv_path}, found columns: {list(df.columns)}"
        )
    tile_ids = df[id_col].astype(str).str.strip()
    groups = df["group"].astype(str).str.strip()
    return dict(zip(tile_ids, groups))


def _tile_proportions(json_path, min_prob=None):
    return load_tile_proportions(
        json_path,
        CELL_TYPE_IDS,
        CELL_TYPE_DICT,
        min_prob=min_prob,
    )


def collect_valid_tiles(json_dir, group_lookup):
    """
    Scan json_dir for tile JSON files, keeping only tiles whose group (looked
    up from group_lookup) isn't background ("bg"). Tiles missing from
    group_lookup, or with an unparseable case ID, are skipped.

    Returns:
        list of dicts: [{'tile_id':..., 'fname':..., 'case_id':..., 'group':...}, ...]
    """
    tiles = []
    skipped_bg = 0
    skipped_missing_group = 0
    skipped_unparsed_case = 0

    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue
        tile_id = fname[:-len(".json")]

        group = group_lookup.get(tile_id)
        if group is None:
            skipped_missing_group += 1
            continue
        if group.lower() == "bg":
            skipped_bg += 1
            continue

        case_id = extract_case_id(tile_id)
        if case_id is None:
            skipped_unparsed_case += 1
            continue

        tiles.append({
            "tile_id": tile_id,
            "fname": fname,
            "case_id": case_id,
            "group": group,
        })

    print(f"Found {len(tiles)} valid tumour/margin tiles.")
    if skipped_bg:
        print(f"  Skipped {skipped_bg} background (bg) tiles.")
    if skipped_missing_group:
        print(f"  Skipped {skipped_missing_group} files not found in the group CSV.")
    if skipped_unparsed_case:
        print(f"  Skipped {skipped_unparsed_case} files with unrecognized filename patterns "
              f"(couldn't parse case ID).")

    return tiles


def compute_pairwise_bc(json_dir, group_lookup, min_prob=None):
    """
    Compute Bray-Curtis dissimilarity for every pair of valid (tumour/margin)
    tiles across the whole directory in one pass.

    Returns:
        pd.DataFrame with columns:
        Case1, Tile1, Group1, Case2, Tile2, Group2, Group_Pair, BC_Distance
    """
    tiles = collect_valid_tiles(json_dir, group_lookup)
    if len(tiles) < 2:
        print("Fewer than 2 valid tiles found; no pairwise comparisons possible.")
        return pd.DataFrame(columns=[
            "Case1", "Tile1", "Group1", "Case2", "Tile2", "Group2", "Group_Pair", "BC_Distance"
        ])

    print("Computing cell-type proportion vectors for each tile...")
    vectors = []
    for t in tiles:
        path = os.path.join(json_dir, t["fname"])
        vectors.append(_tile_proportions(path, min_prob=min_prob))
    X = np.vstack(vectors)

    n = len(tiles)
    print(f"Computing pairwise Bray-Curtis distances for {n} tiles "
          f"({n * (n - 1) // 2:,} pairs)...")
    condensed = pdist(X, metric="braycurtis")

    rows = []
    for (i, j), bc in zip(itertools.combinations(range(n), 2), condensed):
        t1, t2 = tiles[i], tiles[j]
        group_pair = "-".join(sorted([t1["group"], t2["group"]]))
        rows.append({
            "Case1": t1["case_id"],
            "Tile1": t1["fname"],
            "Group1": t1["group"],
            "Case2": t2["case_id"],
            "Tile2": t2["fname"],
            "Group2": t2["group"],
            "Group_Pair": group_pair,
            "BC_Distance": bc,
        })

    return pd.DataFrame(rows)


def run(json_dir, output_dir, group_csv, *, min_prob=None):
    os.makedirs(output_dir, exist_ok=True)

    group_lookup = load_group_lookup(group_csv)
    all_pairs = compute_pairwise_bc(json_dir, group_lookup, min_prob=min_prob)
    if all_pairs.empty:
        print("No pairwise comparisons computed. Exiting.")
        return 0

    is_intra = all_pairs["Case1"] == all_pairs["Case2"]

    # --- Intra-case (within the same case) ---
    intratumour = all_pairs.loc[is_intra].copy()
    intratumour = intratumour.rename(columns={"Case1": "Case"}).drop(columns=["Case2"])
    intratumour = intratumour[["Case", "Tile1", "Group1", "Tile2", "Group2", "Group_Pair", "BC_Distance"]]
    intratumour_path = os.path.join(output_dir, "intratumour_BCD.csv")
    intratumour.to_csv(intratumour_path, index=False)
    print(f"\nIntra-case comparisons: {len(intratumour):,} rows")
    print(f"  Saved to: {intratumour_path}")

    # --- Inter-case (across different cases) ---
    intertumour = all_pairs.loc[~is_intra].copy()
    intertumour = intertumour[["Case1", "Tile1", "Group1", "Case2", "Tile2", "Group2", "Group_Pair", "BC_Distance"]]
    intertumour_path = os.path.join(output_dir, "intertumour_BCD.csv")
    intertumour.to_csv(intertumour_path, index=False)
    print(f"Inter-case comparisons: {len(intertumour):,} rows")
    print(f"  Saved to: {intertumour_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Bray-Curtis dissimilarity for tumour + margin tiles: "
                     "intra-case and inter-case pairwise comparisons."
    )
    parser.add_argument("--json-dir", required=True, help="Directory of per-tile nuc JSON files.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--group-csv",
        required=True,
        help="CSV file mapping tile ID (first column, e.g. 'JN_TS_001_tile_10009_14592', "
             "no .json extension) to a 'group' column with values like tumour/margin/bg.",
    )
    parser.add_argument(
        "--type-info",
        default=str(DEFAULT_TYPE_INFO_PATH),
        help="Path to type_info JSON (default: project root type_info_4class.json).",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=None,
        help="Optional confidence threshold; low-confidence nuclei are reclassified as Others.",
    )
    args = parser.parse_args()

    configure_cell_types(args.type_info)
    return run(
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        group_csv=args.group_csv,
        min_prob=args.min_prob,
    )


if __name__ == "__main__":
    raise SystemExit(main() or 0)
