"""
Bray-Curtis Dissimilarity from Cellular Neighborhood (CN) composition.

Computes the same intra-case / inter-case pairwise Bray-Curtis analysis as
bray_curtis_intra_inter.py, but using each tile's CN composition instead of
its cell-type composition.

Rather than re-deriving CN proportions from the lightweight cn_labels/*.json
files (a different format from the raw cell-type JSON, and CN labels are a
dynamic set rather than a fixed 4 categories), this reads directly from
neighborhood_frequency_per_tile.csv — the tile x CN-proportion table already
produced by cn_unified_kmeans.py or vis_kmeans.py. This keeps a single source
of truth for CN proportions rather than recomputing them a second way.

This is a separate script from bray_curtis_intra_inter.py (cell-type
composition) — that script is unchanged.
"""

import argparse
import itertools
import os
import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


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
    tile names used elsewhere in the pipeline, e.g. 'JN_TS_001_tile_10009_14592')
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


def load_cn_frequency_table(frequency_csv_path):
    """
    Load the tile x CN-proportion table produced by cn_unified_kmeans.py /
    vis_kmeans.py (neighborhood_frequency_per_tile.csv: one row per tile, one
    column per CN label, values are that tile's CN proportions).

    Returns:
        pd.DataFrame indexed by tile_id, one column per CN label.
    """
    df = pd.read_csv(frequency_csv_path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    return df


def collect_valid_tiles(freq_df, group_lookup):
    """
    Keep only tiles present in freq_df whose group (from group_lookup) isn't
    background ("bg"). Tiles missing from group_lookup, or with an
    unparseable case ID, are skipped.

    Returns:
        list of dicts: [{'tile_id':..., 'case_id':..., 'group':...}, ...]
    """
    tiles = []
    skipped_bg = 0
    skipped_missing_group = 0
    skipped_unparsed_case = 0

    for tile_id in freq_df.index:
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
            "case_id": case_id,
            "group": group,
        })

    print(f"Found {len(tiles)} valid tumour/margin tiles.")
    if skipped_bg:
        print(f"  Skipped {skipped_bg} background (bg) tiles.")
    if skipped_missing_group:
        print(f"  Skipped {skipped_missing_group} tiles not found in the group CSV.")
    if skipped_unparsed_case:
        print(f"  Skipped {skipped_unparsed_case} tiles with unrecognized filename patterns "
              f"(couldn't parse case ID).")

    return tiles


def compute_pairwise_bc(freq_df, group_lookup):
    """
    Compute Bray-Curtis dissimilarity for every pair of valid (tumour/margin)
    tiles, using each tile's CN composition (a row of freq_df) as the feature
    vector, across the whole table in one pass.

    Returns:
        pd.DataFrame with columns:
        Case1, Tile1, Group1, Case2, Tile2, Group2, Group_Pair, BC_Distance
    """
    tiles = collect_valid_tiles(freq_df, group_lookup)
    if len(tiles) < 2:
        print("Fewer than 2 valid tiles found; no pairwise comparisons possible.")
        return pd.DataFrame(columns=[
            "Case1", "Tile1", "Group1", "Case2", "Tile2", "Group2", "Group_Pair", "BC_Distance"
        ])

    print("Building CN composition vectors for each tile...")
    X = freq_df.loc[[t["tile_id"] for t in tiles]].to_numpy(dtype=float)

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
            "Tile1": t1["tile_id"],
            "Group1": t1["group"],
            "Case2": t2["case_id"],
            "Tile2": t2["tile_id"],
            "Group2": t2["group"],
            "Group_Pair": group_pair,
            "BC_Distance": bc,
        })

    return pd.DataFrame(rows)


def run(frequency_csv, output_dir, group_csv):
    os.makedirs(output_dir, exist_ok=True)

    group_lookup = load_group_lookup(group_csv)
    freq_df = load_cn_frequency_table(frequency_csv)
    all_pairs = compute_pairwise_bc(freq_df, group_lookup)
    if all_pairs.empty:
        print("No pairwise comparisons computed. Exiting.")
        return 0

    is_intra = all_pairs["Case1"] == all_pairs["Case2"]

    # --- Intra-case (within the same case) ---
    intratumour = all_pairs.loc[is_intra].copy()
    intratumour = intratumour.rename(columns={"Case1": "Case"}).drop(columns=["Case2"])
    intratumour = intratumour[["Case", "Tile1", "Group1", "Tile2", "Group2", "Group_Pair", "BC_Distance"]]
    intratumour_path = os.path.join(output_dir, "intratumour_CN_BCD.csv")
    intratumour.to_csv(intratumour_path, index=False)
    print(f"\nIntra-case comparisons: {len(intratumour):,} rows")
    print(f"  Saved to: {intratumour_path}")

    # --- Inter-case (across different cases) ---
    intertumour = all_pairs.loc[~is_intra].copy()
    intertumour = intertumour[["Case1", "Tile1", "Group1", "Case2", "Tile2", "Group2", "Group_Pair", "BC_Distance"]]
    intertumour_path = os.path.join(output_dir, "intertumour_CN_BCD.csv")
    intertumour.to_csv(intertumour_path, index=False)
    print(f"Inter-case comparisons: {len(intertumour):,} rows")
    print(f"  Saved to: {intertumour_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Bray-Curtis dissimilarity for tumour + margin tiles, using CN "
                     "composition instead of cell-type composition: intra-case and "
                     "inter-case pairwise comparisons."
    )
    parser.add_argument(
        "--frequency-csv",
        required=True,
        help="Path to neighborhood_frequency_per_tile.csv (tile x CN-proportion table) "
             "from cn_unified_kmeans.py or vis_kmeans.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--group-csv",
        required=True,
        help="CSV file mapping tile ID (first column, e.g. 'JN_TS_001_tile_10009_14592') "
             "to a 'group' column with values like tumour/margin/bg.",
    )
    args = parser.parse_args()

    return run(
        frequency_csv=args.frequency_csv,
        output_dir=args.output_dir,
        group_csv=args.group_csv,
    )


if __name__ == "__main__":
    raise SystemExit(main() or 0)
