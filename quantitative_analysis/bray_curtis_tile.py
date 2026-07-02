import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import LinearSegmentedColormap

from cell_type_utils import DEFAULT_TYPE_INFO_PATH, load_tile_proportions, resolve_cell_type_config

# --------------------------------------------------
# Defaults (overridden by CLI / bash)
# --------------------------------------------------
JSON_DIR = r"/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred_03_26/json_reclass"

CELL_TYPE_DICT: dict[int, str] = {}
CELL_TYPE_IDS: list[int] = []


def configure_cell_types(type_info_path=None):
    global CELL_TYPE_DICT, CELL_TYPE_IDS
    cell_type_dict, _, cell_type_ids, _ = resolve_cell_type_config(type_info_path)
    CELL_TYPE_DICT = cell_type_dict
    CELL_TYPE_IDS = cell_type_ids
    return cell_type_dict, cell_type_ids


configure_cell_types()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TILE_CATEGORIES_JSON = os.path.join(
    PROJECT_ROOT, "neighborhood_composition", "spatial_contexts", "tile_categories_88_tiles.json"
)
OUTPUT_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "quantitative_analysis", "bray_curtis")

# Annotate each heatmap cell with BC value (rarely needed)
SHOW_BC_VALUES = False

GROUP_ORDER = ["bg", "margin", "tumour_inv", "tumour_lep", "tumour_scar"]


def load_tile_categories(path):
    with open(path, "r") as f:
        data = json.load(f)

    tile_id_to_group = {}
    ordered_tile_ids = []

    for group in GROUP_ORDER:
        if group not in data or not isinstance(data[group], list):
            continue
        for tile_id in data[group]:
            tile_id_to_group[tile_id] = group
            ordered_tile_ids.append(tile_id)

    return tile_id_to_group, ordered_tile_ids


def simplify_tile_name(fname, group_name):
    name = fname.replace(".json", "")
    pattern = f"(.+?)_{group_name}_tile_(\\d+_\\d+)"
    match = re.match(pattern, name)

    if match:
        prefix = match.group(1)
        numbers = match.group(2)
        return f"{numbers}_{prefix}_{group_name}"
    return name


def _tile_proportions(json_path, min_prob=None):
    return load_tile_proportions(
        json_path,
        CELL_TYPE_IDS,
        CELL_TYPE_DICT,
        min_prob=min_prob,
    )


def _make_cmap():
    colors = ["#0000FF", "#FFFFFF", "#FF0000"]
    return LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)


def _ticklabels_overall(
    sorted_tile_names,
    sorted_tile_groups,
    *,
    show_group_names_only: bool,
    show_tile_names: bool,
):
    if show_group_names_only:
        group_boundaries = []
        current_group = None
        group_start_idx = 0

        for i, group in enumerate(sorted_tile_groups):
            if group != current_group:
                if current_group is not None:
                    group_boundaries.append((group_start_idx, i - 1, current_group))
                current_group = group
                group_start_idx = i
        if current_group is not None:
            group_boundaries.append((group_start_idx, len(sorted_tile_groups) - 1, current_group))

        x_labels = [""] * len(sorted_tile_names)
        y_labels = [""] * len(sorted_tile_names)
        for start_idx, end_idx, gname in group_boundaries:
            middle_idx = (start_idx + end_idx) // 2
            x_labels[middle_idx] = gname
            y_labels[middle_idx] = gname
        return x_labels, y_labels
    if show_tile_names:
        simplified = [
            simplify_tile_name(name, tile_group)
            for name, tile_group in zip(sorted_tile_names, sorted_tile_groups)
        ]
        return simplified, simplified
    return False, False


def _draw_group_separators(ax, group_labels):
    """Match original script: black lines between tile_groups on heatmap."""
    current_group = None
    group_start_idx = 0
    for i, group in enumerate(group_labels):
        if group != current_group:
            if current_group is not None:
                ax.axhline(y=group_start_idx, color="black", linewidth=2)
                ax.axvline(x=group_start_idx, color="black", linewidth=2)
            current_group = group
            group_start_idx = i
    if len(group_labels) > 0:
        ax.axhline(y=len(group_labels), color="black", linewidth=2)
        ax.axvline(x=len(group_labels), color="black", linewidth=2)


def _plot_heatmap(
    bc_df,
    title,
    output_path,
    x_ticklabels,
    y_ticklabels,
    cmap,
    group_labels_for_separators=None,
    *,
    show_group_names_only: bool,
    show_tile_names: bool,
):
    n = bc_df.shape[0]
    figsize = (max(8, n * 0.12), max(7, n * 0.12))
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        bc_df,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=SHOW_BC_VALUES,
        fmt=".3f",
        cbar_kws={"label": "Bray-Curtis Dissimilarity"},
        square=True,
        linewidths=0.5,
        linecolor="gray",
        xticklabels=x_ticklabels,
        yticklabels=y_ticklabels,
    )
    if group_labels_for_separators is not None:
        _draw_group_separators(ax, group_labels_for_separators)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Tiles", fontsize=12)
    plt.ylabel("Tiles", fontsize=12)
    if show_group_names_only and x_ticklabels is not False:
        plt.xticks(rotation=45, ha="right", fontsize=12, fontweight="bold")
        plt.yticks(rotation=0, fontsize=12, fontweight="bold")
    elif show_tile_names and x_ticklabels is not False:
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def run(
    json_dir: str,
    tile_categories_json: str,
    output_dir: str,
    per_group: bool,
    *,
    show_tile_names: bool = False,
    show_group_names_only: bool = True,
):
    tile_id_to_group, ordered_tile_ids = load_tile_categories(tile_categories_json)

    tile_by_id = {}
    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        tile_id = fname.replace(".json", "")
        if tile_id not in tile_id_to_group:
            continue
        path = os.path.join(json_dir, fname)
        tile_by_id[tile_id] = (fname, _tile_proportions(path))

    sorted_tile_names = []
    sorted_tile_vectors = []
    sorted_tile_groups = []

    for tile_id in ordered_tile_ids:
        if tile_id not in tile_by_id:
            continue
        fname, vec = tile_by_id[tile_id]
        sorted_tile_names.append(fname)
        sorted_tile_vectors.append(vec)
        sorted_tile_groups.append(tile_id_to_group[tile_id])

    if not sorted_tile_names:
        print("No tiles found (check JSON_DIR and tile_categories).", file=sys.stderr)
        return 1

    X_sorted = np.vstack(sorted_tile_vectors)
    num_tiles = len(sorted_tile_names)
    group_name = os.path.basename(json_dir.rstrip(os.sep))

    print("\nTile clustering by group (from tile_categories JSON):")
    current_group = None
    group_start_idx = 0
    for i, group in enumerate(sorted_tile_groups):
        if group != current_group:
            if current_group is not None:
                print(f"  {current_group}: tiles {group_start_idx} to {i-1} ({i - group_start_idx} tiles)")
            current_group = group
            group_start_idx = i
    if current_group is not None:
        n_last = len(sorted_tile_groups) - group_start_idx
        print(
            f"  {current_group}: tiles {group_start_idx} to {len(sorted_tile_groups)-1} ({n_last} tiles)"
        )
    print()

    dist_condensed = pdist(X_sorted, metric="braycurtis")
    dist_matrix = squareform(dist_condensed)
    upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    mean_bc = float(np.mean(upper_triangle))
    median_bc = float(np.median(upper_triangle))

    bc_df = pd.DataFrame(dist_matrix, index=sorted_tile_names, columns=sorted_tile_names)
    print(bc_df)
    print(f"\nMean BC: {mean_bc:.4f}, Median BC: {median_bc:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    cmap = _make_cmap()
    simplified_tile_names = [
        simplify_tile_name(name, tile_group)
        for name, tile_group in zip(sorted_tile_names, sorted_tile_groups)
    ]

    # --- Overall heatmap ---
    x_tick, y_tick = _ticklabels_overall(
        sorted_tile_names,
        sorted_tile_groups,
        show_group_names_only=show_group_names_only,
        show_tile_names=show_tile_names,
    )
    overall_png = os.path.join(output_dir, f"bray_curtis_overall_{num_tiles}_heatmap.png")
    title = (
        f"Bray-Curtis Dissimilarity Matrix: {group_name} (n = {num_tiles} tiles)\n"
        f"Mean: {mean_bc:.4f}; Median: {median_bc:.4f}"
    )
    sep = sorted_tile_groups if show_group_names_only else None
    _plot_heatmap(
        bc_df,
        title,
        overall_png,
        x_tick,
        y_tick,
        cmap,
        group_labels_for_separators=sep,
        show_group_names_only=show_group_names_only,
        show_tile_names=show_tile_names,
    )
    bc_df.to_csv(os.path.join(output_dir, f"bray_curtis_overall_{num_tiles}_dissimilarity.csv"))
    print(f"  Saved CSV: bray_curtis_overall_{num_tiles}_dissimilarity.csv")

    # --- Per-group heatmaps (submatrix of full BC matrix) ---
    if per_group:
        print("\nPer-group Bray-Curtis heatmaps:")
        for grp in GROUP_ORDER:
            idx = [i for i, g in enumerate(sorted_tile_groups) if g == grp]
            if not idx:
                print(f"  (skip {grp}: no tiles)")
                continue
            sub = bc_df.iloc[idx, idx]
            sub_mean = float(np.mean(sub.values[np.triu_indices_from(sub.values, k=1)])) if sub.shape[0] > 1 else 0.0
            sub_med = float(np.median(sub.values[np.triu_indices_from(sub.values, k=1)])) if sub.shape[0] > 1 else 0.0
            n_g = len(idx)
            xt = (
                [simplified_tile_names[i] for i in idx] if show_tile_names else False
            )
            yt = xt
            out_g = os.path.join(output_dir, f"bray_curtis_pergroup_{grp}_{n_g}_heatmap.png")
            t_g = f"Bray-Curtis within {grp} (n = {n_g} tiles)\nMean: {sub_mean:.4f}; Median: {sub_med:.4f}"
            _plot_heatmap(
                sub,
                t_g,
                out_g,
                xt,
                yt,
                cmap,
                group_labels_for_separators=None,
                show_group_names_only=False,
                show_tile_names=show_tile_names,
            )
            sub.to_csv(os.path.join(output_dir, f"bray_curtis_pergroup_{grp}_{n_g}_dissimilarity.csv"))

    return 0


def main():
    p = argparse.ArgumentParser(description="Bray-Curtis tile dissimilarity: overall + optional per-group maps.")
    p.add_argument("--json-dir", default=JSON_DIR, help="Directory of per-tile nuc JSON files.")
    p.add_argument(
        "--tile-categories-json",
        default=TILE_CATEGORIES_JSON,
        help="tile_categories_88_tiles.json path.",
    )
    p.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_DEFAULT,
        help="Directory for PNG/CSV outputs.",
    )
    p.add_argument(
        "--type-info",
        default=str(DEFAULT_TYPE_INFO_PATH),
        help="Path to type_info JSON (default: project root type_info_4class.json).",
    )
    p.add_argument(
        "--no-per-group",
        action="store_true",
        help="Only overall heatmap (no per-group PNG/CSV).",
    )
    p.add_argument(
        "--show-tile-names",
        action="store_true",
        help="Show simplified tile names on heatmap axes (default: off). Overall: only if --no-show-group-names-on-axis.",
    )
    p.add_argument(
        "--no-show-group-names-on-axis",
        action="store_true",
        help="Do not place group names at cluster midpoints on the overall heatmap (use with --show-tile-names for tile labels).",
    )
    args = p.parse_args()
    configure_cell_types(args.type_info)
    show_group_axis = not args.no_show_group_names_on_axis
    return run(
        json_dir=args.json_dir,
        tile_categories_json=args.tile_categories_json,
        output_dir=args.output_dir,
        per_group=not args.no_per_group,
        show_tile_names=args.show_tile_names,
        show_group_names_only=show_group_axis,
    )


if __name__ == "__main__":
    raise SystemExit(main() or 0)
