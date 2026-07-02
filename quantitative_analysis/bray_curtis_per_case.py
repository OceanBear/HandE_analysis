import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import pdist, squareform

from cell_type_utils import DEFAULT_TYPE_INFO_PATH, load_tile_proportions, resolve_cell_type_config

# --------------------------------------------------
# Config
# --------------------------------------------------
JSON_DIR = r"/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred_03_26/json_reclass"

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

# Visualization parameters
SHOW_TILE_NAMES = True
SHOW_GROUP_NAMES_ONLY = False
SHOW_BC_VALUES = True


def extract_case_id(tile_id):
    """Example: JN_TS_001_tumour_inv_tile_10912_14661 -> JN_TS_001"""
    match = re.match(r"(.+?)_(tumour_inv|tumour_lep|bg|margin|tumour_scar)_tile_", tile_id)
    if match:
        return match.group(1)
    return None


def extract_group_name(tile_id):
    """Example: JN_TS_001_tumour_inv_tile_10912_14661 -> tumour_inv"""
    match = re.match(r".+?_(tumour_inv|tumour_lep|bg|margin|tumour_scar)_tile_", tile_id)
    if match:
        return match.group(1)
    return None


def simplify_tile_name(fname, group_name):
    """Example: JN_TS_010_tumour_inv_tile_11151_13664.json -> 11151_13664_tumour_inv"""
    name = fname.replace(".json", "")
    pattern = f".+?_{group_name}_tile_(\\d+_\\d+)"
    match = re.match(pattern, name)
    if match:
        numbers = match.group(1)
        return f"{numbers}_{group_name}"
    return name


def _tile_proportions(json_path, min_prob=None):
    return load_tile_proportions(
        json_path,
        CELL_TYPE_IDS,
        CELL_TYPE_DICT,
        min_prob=min_prob,
    )


def _collect_tiles_by_case(json_dir):
    tiles_by_case = defaultdict(list)
    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        tile_id = fname.replace(".json", "")
        group = extract_group_name(tile_id)
        if group not in ["tumour_inv", "tumour_lep"]:
            continue
        case_id = extract_case_id(tile_id)
        if case_id is None:
            continue
        tiles_by_case[case_id].append((tile_id, fname, group))
    return tiles_by_case


def _plot_case_heatmap(
    bc_df,
    tile_names,
    tile_groups,
    case_id,
    output_dir,
    *,
    show_tile_names,
    show_group_names_only,
    show_bc_values,
):
    simplified_tile_names = [
        simplify_tile_name(name, tile_group)
        for name, tile_group in zip(tile_names, tile_groups)
    ]

    colors = ["#0000FF", "#FFFFFF", "#FF0000"]
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)

    if show_group_names_only:
        group_boundaries = []
        current_group = None
        group_start_idx = 0
        for i, group in enumerate(tile_groups):
            if group != current_group:
                if current_group is not None:
                    group_boundaries.append((group_start_idx, i - 1, current_group))
                current_group = group
                group_start_idx = i
        if current_group is not None:
            group_boundaries.append((group_start_idx, len(tile_groups) - 1, current_group))

        x_labels = [""] * len(tile_names)
        y_labels = [""] * len(tile_names)
        for start_idx, end_idx, group_name in group_boundaries:
            middle_idx = (start_idx + end_idx) // 2
            x_labels[middle_idx] = group_name
            y_labels[middle_idx] = group_name
        x_ticklabels = x_labels
        y_ticklabels = y_labels
    elif show_tile_names:
        x_ticklabels = simplified_tile_names
        y_ticklabels = simplified_tile_names
    else:
        x_ticklabels = False
        y_ticklabels = False

    num_tiles = len(tile_names)
    upper_triangle = bc_df.values[np.triu_indices_from(bc_df.values, k=1)]
    mean_bc = float(np.mean(upper_triangle)) if upper_triangle.size else 0.0
    median_bc = float(np.median(upper_triangle)) if upper_triangle.size else 0.0

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        bc_df,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=show_bc_values,
        fmt=".3f",
        cbar_kws={"label": "Bray-Curtis Dissimilarity"},
        square=True,
        linewidths=0.5,
        linecolor="gray",
        xticklabels=x_ticklabels,
        yticklabels=y_ticklabels,
    )

    current_group = None
    group_start_idx = 0
    for i, group in enumerate(tile_groups):
        if group != current_group:
            if current_group is not None:
                ax.axhline(y=group_start_idx, color="black", linewidth=2)
                ax.axvline(x=group_start_idx, color="black", linewidth=2)
            current_group = group
            group_start_idx = i
    if tile_groups:
        ax.axhline(y=len(tile_groups), color="black", linewidth=2)
        ax.axvline(x=len(tile_groups), color="black", linewidth=2)

    title = (
        f"Bray-Curtis Dissimilarity Matrix: {case_id} (n = {num_tiles} tiles)\n"
        f"Mean: {mean_bc:.4f}; Median: {median_bc:.4f};"
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Tiles", fontsize=12)
    plt.ylabel("Tiles", fontsize=12)

    if show_group_names_only:
        plt.xticks(rotation=0, ha="center", fontsize=10, fontweight="bold")
        plt.yticks(rotation=0, fontsize=10, fontweight="bold")
    elif show_tile_names:
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)

    plt.tight_layout()
    output_filename = os.path.join(output_dir, f"bray_curtis_{case_id}_{num_tiles}_heatmap.png")
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"  Heatmap saved to: {output_filename}")
    plt.close()

    csv_filename = os.path.join(output_dir, f"bray_curtis_{case_id}_{num_tiles}_dissimilarity.csv")
    bc_df.to_csv(csv_filename)
    print(f"  CSV saved to: {csv_filename}")


def run(
    json_dir,
    output_dir,
    *,
    show_tile_names=SHOW_TILE_NAMES,
    show_group_names_only=SHOW_GROUP_NAMES_ONLY,
    show_bc_values=SHOW_BC_VALUES,
):
    os.makedirs(output_dir, exist_ok=True)
    tiles_by_case = _collect_tiles_by_case(json_dir)
    case_ids = sorted(tiles_by_case.keys())

    for case_id in case_ids:
        tiles = tiles_by_case[case_id]
        tiles.sort(key=lambda x: (0 if x[2] == "tumour_inv" else 1, x[0]))

        tile_names = []
        tile_vectors = []
        tile_groups = []

        for tile_id, fname, group in tiles:
            path = os.path.join(json_dir, fname)
            tile_names.append(fname)
            tile_vectors.append(_tile_proportions(path))
            tile_groups.append(group)

        if not tile_names:
            print(f"\nSkipping {case_id}: no tumour_inv or tumour_lep tiles found")
            continue

        num_tiles = len(tile_names)
        print(f"\nProcessing case: {case_id}")
        print(f"  Number of tiles: {num_tiles}")
        tumour_inv_count = sum(1 for g in tile_groups if g == "tumour_inv")
        tumour_lep_count = sum(1 for g in tile_groups if g == "tumour_lep")
        print(f"  tumour_inv: {tumour_inv_count}, tumour_lep: {tumour_lep_count}")

        X = np.vstack(tile_vectors)
        dist_condensed = pdist(X, metric="braycurtis")
        dist_matrix = squareform(dist_condensed)
        upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        print(f"  Mean BC: {np.mean(upper_triangle):.4f}, Median BC: {np.median(upper_triangle):.4f}")

        bc_df = pd.DataFrame(dist_matrix, index=tile_names, columns=tile_names)
        _plot_case_heatmap(
            bc_df,
            tile_names,
            tile_groups,
            case_id,
            output_dir,
            show_tile_names=show_tile_names,
            show_group_names_only=show_group_names_only,
            show_bc_values=show_bc_values,
        )

    print(f"\nProcessing complete. Processed {len(case_ids)} cases.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Bray-Curtis dissimilarity heatmaps per case (tumour_inv + tumour_lep tiles)."
    )
    parser.add_argument("--json-dir", default=JSON_DIR, help="Directory of per-tile nuc JSON files.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "quantitative_analysis", "bray_curtis_case"),
        help="Directory for PNG/CSV outputs.",
    )
    parser.add_argument(
        "--type-info",
        default=str(DEFAULT_TYPE_INFO_PATH),
        help="Path to type_info JSON (default: project root type_info_4class.json).",
    )
    parser.add_argument("--show-tile-names", action="store_true", help="Show simplified tile names on axes.")
    parser.add_argument(
        "--show-group-names-only",
        action="store_true",
        help="Show one group label per cluster instead of tile names.",
    )
    parser.add_argument(
        "--show-bc-values",
        action="store_true",
        help="Annotate each heatmap cell with Bray-Curtis values.",
    )
    args = parser.parse_args()

    configure_cell_types(args.type_info)
    return run(
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        show_tile_names=args.show_tile_names or SHOW_TILE_NAMES,
        show_group_names_only=args.show_group_names_only or SHOW_GROUP_NAMES_ONLY,
        show_bc_values=args.show_bc_values or SHOW_BC_VALUES,
    )


if __name__ == "__main__":
    raise SystemExit(main() or 0)
