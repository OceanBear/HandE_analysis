"""
CN × tile-group bar charts: how CNs are distributed across tile groups.
-----------------------------------------------------------------------------
Same inputs and output directory as cn_select_optimal_k.py. For each k (and
optionally sub-clustered folders like all_n_cluster=5_sub_1), produces two
stacked bar charts in one figure:

  1. X-axis = CNs. Each bar shows the proportion of that CN's cells from each
     tile group (within each CN, how groups distribute).
  2. X-axis = tile groups. Each bar shows the proportion of that group's cells
     in each CN (within each group, how CNs distribute). CN colors match
     cn_unified_kmeans_groups (tab20).

Supports integer CN labels (1, 2, 3) and sub-clustered string labels (CN2-1, CN2-2).
Only tiles present in the categories JSON are included (no "unknown" group).

Requires: anndata or scanpy, matplotlib, seaborn, tile_categories JSON.

Example:
  python cn_by_group_barcharts.py \
    --results_root "/path/to/cn_unified_results" \
    --categories_json "tile_categories_88_tiles.json" \
    --out_dir "/path/to/cn_unified_results/k_selection" \
    --include_sub
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _import_h5ad_reader():
    """Return read_h5ad(path) -> AnnData, with helpful error if unavailable."""
    try:
        import anndata as ad  # type: ignore
        return ad.read_h5ad
    except Exception:
        try:
            import scanpy as sc  # type: ignore
            return sc.read_h5ad
        except Exception as e:
            raise ImportError(
                "Reading .h5ad requires `anndata` or `scanpy`.\n"
                "Install one of them, e.g. `pip install anndata` (or `scanpy`)."
            ) from e


def _safe_to_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _is_integer_cn_labels(labels) -> bool:
    """Check if CN labels are integers (1, 2, 3) vs strings (CN1, CN3-1)."""
    for x in labels:
        try:
            int(float(str(x).strip()))
        except (ValueError, TypeError):
            return False
    return True


def _sort_cn_labels_and_colors(labels, color_palette: str = "tab20") -> Tuple[List, List]:
    """
    Sort CN labels and return (sorted_labels, colors). Matches cn_unified_kmeans_groups.
    Supports integer labels (1, 2, 3) and string labels (CN1, CN2, CN3-1).
    """
    import seaborn as sns

    labels = list(labels)
    if not labels:
        return [], []

    if _is_integer_cn_labels(labels):
        int_labels = [int(float(str(x).strip())) for x in labels]
        sorted_labels = sorted(set(int_labels), key=lambda x: x)
        n = len(sorted_labels)
        palette = sns.color_palette(color_palette, max(sorted_labels))
        colors = [palette[x - 1] for x in sorted_labels]
        return sorted_labels, colors
    else:
        def sort_key(s):
            s = str(s).strip()
            if not s.startswith("CN"):
                return (999, 0)
            rest = s[2:]
            if "-" in rest:
                parts = rest.split("-", 1)
                return (
                    int(parts[0]) if parts[0].isdigit() else 999,
                    int(parts[1]) if parts[1].isdigit() else 0,
                )
            return (int(rest) if rest.isdigit() else 999, 0)

        sorted_labels = sorted(set(labels), key=sort_key)
        n = len(sorted_labels)
        palette = sns.color_palette(color_palette, max(n, 20))[:n]
        colors = list(palette)
        return sorted_labels, colors


def load_tile_categories(categories_json: Path) -> Tuple[Dict[str, str], List[str]]:
    """
    Load tile_categories JSON and return (tile_name -> group, ordered group list).
    Ignores the 'metadata' key. Group order follows JSON key order (excluding metadata).
    """
    with categories_json.open("r") as f:
        data = json.load(f)

    tile_to_group: Dict[str, str] = {}
    group_order: List[str] = []
    seen_groups = set()

    for key, val in data.items():
        if key == "metadata":
            continue
        if not isinstance(val, list):
            continue
        if key not in seen_groups:
            seen_groups.add(key)
            group_order.append(key)
        for t in val:
            tile_to_group[str(t)] = str(key)

    return tile_to_group, group_order


def _cn_sort_key(s) -> Tuple[int, int]:
    """Sort key for CN labels: integer (1,2,3) or string (CN1, CN2-1, CN3-2)."""
    s = str(s).strip()
    if not s.startswith("CN"):
        try:
            return (int(float(s)), 0)
        except (ValueError, TypeError):
            return (999, 0)
    rest = s[2:]
    if "-" in rest:
        parts = rest.split("-", 1)
        return (
            int(parts[0]) if parts[0].isdigit() else 999,
            int(parts[1]) if parts[1].isdigit() else 0,
        )
    return (int(rest) if rest.isdigit() else 999, 0)


def _cn_display_name(cn_label: Union[int, str]) -> str:
    """Display name for CN in plot (CN1, CN2-1, etc.)."""
    s = str(cn_label).strip()
    if s.startswith("CN"):
        return s
    try:
        return f"CN{int(float(s))}"
    except (ValueError, TypeError):
        return s


def build_group_cn_counts(
    processed_h5ad_dir: Path,
    tile_to_group: Dict[str, str],
    group_order: List[str],
    *,
    cn_key: str,
    tile_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int, List[str]]:
    """
    Load h5ad files from processed_h5ad_dir, aggregate (group × CN) counts.
    Only includes tiles present in tile_to_group (no "unknown" group).
    CN labels are discovered from data (integer or string like CN2-1).
    Returns (counts, prop_by_cn, prop_by_group, n_tiles_used, n_tiles_total, missing_tile_names).
    """
    read_h5ad = _import_h5ad_reader()

    if not processed_h5ad_dir.exists():
        raise FileNotFoundError(f"Missing directory: {processed_h5ad_dir}")

    h5ad_files = sorted(processed_h5ad_dir.glob("*.h5ad"))
    if not h5ad_files:
        raise FileNotFoundError(f"No .h5ad files found in: {processed_h5ad_dir}")

    n_tiles_total = len(h5ad_files)

    # First pass: collect unique CN labels from all files (only from tiles in tile_to_group)
    cn_labels_set: set = set()
    for f in h5ad_files:
        adata = read_h5ad(str(f))
        if cn_key not in adata.obs.columns:
            raise KeyError(f"Missing `{cn_key}` in {f.name}")
        tile_name = _get_tile_name(f, adata, tile_key, tile_to_group)
        if tile_name not in tile_to_group:
            continue
        cn_labels_set.update(adata.obs[cn_key].astype(str).str.strip().unique())
    sorted_cn_labels = sorted(cn_labels_set, key=_cn_sort_key)
    cn_to_col = {cn: j for j, cn in enumerate(sorted_cn_labels)}
    n_cns = len(sorted_cn_labels)
    if n_cns == 0:
        raise ValueError(f"No CN labels found in {processed_h5ad_dir} (or no tiles in categories)")

    # Column display names (CN1, CN2-1, etc.)
    col_names = [_cn_display_name(cn) for cn in sorted_cn_labels]

    group_idx = {g: i for i, g in enumerate(group_order)}
    counts = np.zeros((len(group_order), n_cns), dtype=np.float64)
    n_tiles_used = 0
    missing_tile_names: List[str] = []

    for f in h5ad_files:
        adata = read_h5ad(str(f))
        if cn_key not in adata.obs.columns:
            raise KeyError(f"Missing `{cn_key}` in {f.name}")

        tile_name = _get_tile_name(f, adata, tile_key, tile_to_group)
        if tile_name not in tile_to_group:
            missing_tile_names.append(tile_name)
            continue
        n_tiles_used += 1
        group = tile_to_group[tile_name]
        g_idx = group_idx[group]

        cn_series = adata.obs[cn_key].astype(str).str.strip()
        vc = cn_series.value_counts()
        for cn_label, cnt in vc.items():
            cn_key_str = str(cn_label).strip()
            if cn_key_str in cn_to_col:
                counts[g_idx, cn_to_col[cn_key_str]] += int(cnt)

    count_df = pd.DataFrame(
        counts,
        index=group_order,
        columns=col_names,
    )

    col_sums = count_df.sum(axis=0).replace(0, np.nan)
    prop_by_cn = (count_df / col_sums).fillna(0.0)
    row_sums = count_df.sum(axis=1).replace(0, np.nan)
    prop_by_group = count_df.div(row_sums, axis=0).fillna(0.0)

    return count_df, prop_by_cn, prop_by_group, n_tiles_used, n_tiles_total, missing_tile_names


def _get_tile_name(f: Path, adata, tile_key: str, tile_to_group: Optional[Dict[str, str]] = None) -> str:
    """
    Infer tile name for lookup. Prefer filename-derived name when it is in tile_to_group,
    so a typo inside adata.obs (e.g. bg_airway_tile vs bg_tile) does not cause the tile to be skipped.
    """
    name_from_file = f.stem
    if name_from_file.endswith("_adata_cns"):
        name_from_file = name_from_file[: -len("_adata_cns")]
    if tile_to_group is not None and name_from_file in tile_to_group:
        return name_from_file
    if tile_key in adata.obs.columns:
        uniq = pd.unique(adata.obs[tile_key].astype(str))
        if len(uniq) == 1:
            return str(uniq[0])
    return name_from_file


def plot_two_barcharts(
    display_label: str,
    prop_by_cn: pd.DataFrame,
    prop_by_group: pd.DataFrame,
    group_order: List[str],
    out_path: Path,
    n_tiles_used: int,
    n_tiles_total: int,
    color_palette: str = "tab20",
) -> None:
    """
    Create one figure with two stacked bar charts:
      Left: x = CNs, stacked segments = tile groups (prop_by_cn).
      Right: x = tile groups, stacked segments = CNs (prop_by_group).
    Title includes tile counts (used / total) to spot missing tiles.
    CN colors on the right chart match cn_unified_kmeans_groups (tab20).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_groups = len(prop_by_cn.index)
    n_cns = prop_by_cn.shape[1]

    # Left: group colors (tab10)
    cmap_groups = plt.get_cmap("tab10")
    colors_group = [cmap_groups(i % 10) for i in range(n_groups)]

    # Right: CN colors matching cn_unified_kmeans_groups (tab20, same sort order)
    cn_col_names = list(prop_by_group.columns)
    sorted_cn_labels, colors_cn = _sort_cn_labels_and_colors(cn_col_names, color_palette)
    # Map column name -> color (columns may be "CN1", "CN2-1"; sort returns same order as _cn_sort_key)
    cn_to_color = {}
    for lab, col in zip(sorted_cn_labels, colors_cn):
        disp = _cn_display_name(lab)
        cn_to_color[disp] = col
    colors_cn_ordered = [cn_to_color.get(c, colors_cn[0]) for c in cn_col_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: x = CNs, stacked = groups ----
    x = np.arange(n_cns)
    width = 0.65
    bottom = np.zeros(n_cns)
    for i, grp in enumerate(prop_by_cn.index):
        vals = prop_by_cn.loc[grp].values
        ax1.bar(x, vals, width, label=grp, bottom=bottom, color=colors_group[i])
        bottom += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(prop_by_cn.columns)
    ax1.set_xlabel("Cellular neighborhood (CN)")
    ax1.set_ylabel("Proportion of cells (within each CN)")
    ax1.set_title("Within each CN: proportion from each tile group")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(0, 1)

    # ---- Right: x = groups, stacked = CNs (same color pattern as cn_unified_kmeans_groups) ----
    x2 = np.arange(n_groups)
    bottom2 = np.zeros(n_groups)
    for j in range(n_cns):
        cn_label = prop_by_group.columns[j]
        vals = prop_by_group.iloc[:, j].values
        ax2.bar(x2, vals, width, label=cn_label, bottom=bottom2, color=colors_cn_ordered[j])
        bottom2 += vals
    ax2.set_xticks(x2)
    ax2.set_xticklabels(prop_by_group.index, rotation=45, ha="right")
    ax2.set_xlabel("Tile group")
    ax2.set_ylabel("Proportion of cells (within each group)")
    ax2.set_title("Within each tile group: proportion in each CN")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, 1)

    if n_tiles_used == n_tiles_total:
        tile_str = f"{n_tiles_used} tiles"
    else:
        tile_str = f"{n_tiles_used} tiles used ({n_tiles_total} total in folder)"
    fig.suptitle(f"CN × tile group distribution ({display_label}) — {tile_str}", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    script_dir = Path(__file__).parent.resolve()

    p = argparse.ArgumentParser(
        description="CN × tile group stacked bar charts (same inputs as cn_select_optimal_k)."
    )
    p.add_argument(
        "--results_root",
        type=str,
        default="/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/cn_unified_results",
        help="Path containing all_n_cluster=<k>/processed_h5ad folders.",
    )
    p.add_argument(
        "--k_min", type=int, default=4, help="Minimum k (inclusive)."
    )
    p.add_argument(
        "--k_max", type=int, default=13, help="Maximum k (inclusive)."
    )
    p.add_argument(
        "--categories_json",
        type=str,
        default="tile_categories_88_tiles.json",
        help="Tile category JSON (relative to script dir if not absolute).",
    )
    p.add_argument("--cn_key", type=str, default="cn_celltype", help="CN label column in adata.obs.")
    p.add_argument(
        "--tile_key",
        type=str,
        default="tile_name",
        help="Tile name column in adata.obs (fallback: infer from filename).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <results_root>/k_selection",
    )
    p.add_argument(
        "--save_csv",
        action="store_true",
        help="Save per-k group×CN count and proportion CSVs.",
    )
    p.add_argument(
        "--include_sub",
        action="store_true",
        help="Also process sub-clustered folders (all_n_cluster=*_sub*). Uses cn_celltype_sub.",
    )
    p.add_argument(
        "--color_palette",
        type=str,
        default="tab20",
        help="Color palette for CN segments in right chart (default: tab20, match cn_unified_kmeans_groups).",
    )

    args = p.parse_args()

    results_root = Path(args.results_root)
    categories_json = Path(args.categories_json)
    if not categories_json.is_absolute():
        categories_json = (script_dir / categories_json).resolve()

    out_dir = Path(args.out_dir) if args.out_dir else (results_root / "k_selection")
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_to_group, group_order = load_tile_categories(categories_json)
    print(f"Loaded {len(tile_to_group)} tiles in groups: {group_order}")
    print("Only tiles present in categories JSON are included (no 'unknown' group).")

    tasks: List[Tuple[Path, str, str]] = []  # (processed_h5ad_dir, display_label, cn_key)

    for k in range(args.k_min, args.k_max + 1):
        k_dir = results_root / f"all_n_cluster={k}" / "processed_h5ad"
        if k_dir.exists():
            tasks.append((k_dir, str(k), args.cn_key))

    if args.include_sub:
        # Folders like all_n_cluster=5_sub_1, all_n_cluster=9_sub
        for d in sorted(results_root.iterdir()):
            if not d.is_dir():
                continue
            if not re.match(r"all_n_cluster=.*_sub", d.name):
                continue
            sub_h5ad = d / "processed_h5ad"
            if sub_h5ad.exists():
                display = d.name.replace("all_n_cluster=", "")
                tasks.append((sub_h5ad, display, "cn_celltype_sub"))

    for processed_h5ad_dir, display_label, cn_key in tasks:
        try:
            count_df, prop_by_cn, prop_by_group, n_tiles_used, n_tiles_total, missing_tiles = build_group_cn_counts(
                processed_h5ad_dir=processed_h5ad_dir,
                tile_to_group=tile_to_group,
                group_order=group_order.copy(),
                cn_key=cn_key,
                tile_key=args.tile_key,
            )
            group_order_final = list(prop_by_cn.index)

            out_path = out_dir / f"cn_by_group_barcharts_{display_label}.png"
            plot_two_barcharts(
                display_label=display_label,
                prop_by_cn=prop_by_cn,
                prop_by_group=prop_by_group,
                group_order=group_order_final,
                out_path=out_path,
                n_tiles_used=n_tiles_used,
                n_tiles_total=n_tiles_total,
                color_palette=args.color_palette,
            )
            if n_tiles_used < n_tiles_total:
                print(f"✓ {display_label}: saved {out_path} ({n_tiles_used}/{n_tiles_total} tiles used)")
                print(f"  Missing (not in categories JSON): {', '.join(sorted(missing_tiles))}")
            else:
                print(f"✓ {display_label}: saved {out_path}")

            if args.save_csv:
                count_df.to_csv(out_dir / f"cn_by_group_counts_{display_label}.csv")
                prop_by_cn.to_csv(out_dir / f"cn_by_group_prop_within_cn_{display_label}.csv")
                prop_by_group.to_csv(out_dir / f"cn_by_group_prop_within_group_{display_label}.csv")
                print(f"  Saved CSVs for {display_label}")

        except FileNotFoundError as e:
            print(f"✗ {display_label}: {e}")
        except Exception as e:
            print(f"✗ {display_label}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
