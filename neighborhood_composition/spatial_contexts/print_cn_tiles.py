"""
Individual Tile Spatial Cellular Neighborhood Maps

Generates one spatial scatter plot per tile, showing each cell colored by its
assigned cellular neighborhood (CN). Split out from vis_kmeans.py since this
step renders one figure per tile (potentially hundreds) and is the slowest
part of the overall pipeline — keep it separate so you can skip it on faster
iteration cycles and only run it when you actually need the spatial maps.

Reads each tile's original source h5ad (from data_preparation.py; has
cell_type + spatial, untouched by clustering) and merges in that tile's CN
labels from the matching lightweight JSON file written by
cn_unified_kmeans.py. No annotated h5ad copy is ever written or required.
"""

import json
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from typing import Optional, Set
import warnings
warnings.filterwarnings('ignore')


def load_tile_selection(csv_path) -> Set[str]:
    """
    Load a set of tile names to include, from a CSV with a 'tile' column
    (e.g. one row per tile: 'JN_TS_001_tile_12883_7423'). Tile names should
    match each h5ad file's stem (filename without the .h5ad extension).

    Used to restrict plotting to a subset of tiles — e.g. a quick visual
    spot-check on a handful of tiles — without touching source_h5ad_dir or
    cn_labels_dir.
    """
    df = pd.read_csv(csv_path)
    if 'tile' not in df.columns:
        raise ValueError(
            f"Expected a 'tile' column in {csv_path}, found columns: {list(df.columns)}"
        )
    tiles = set(df['tile'].astype(str).str.strip())
    print(f"✓ Loaded tile selection: {len(tiles)} tiles from {csv_path}")
    return tiles


def _is_integer_cn_labels(labels) -> bool:
    """Check if CN labels are integers (1, 2, 3) vs strings (CN1, CN3-1)."""
    for x in labels:
        try:
            int(float(str(x).strip()))
        except (ValueError, TypeError):
            return False
    return True


def _sort_cn_labels_and_colors(labels, color_palette: str = 'tab20'):
    """
    Sort CN labels and return (sorted_labels, colors).
    Supports both integer labels (1, 2, 3) and string labels (CN1, CN2, CN3-1).
    """
    labels = list(labels)
    if not labels:
        return [], []

    if _is_integer_cn_labels(labels):
        int_labels = [int(float(str(x).strip())) for x in labels]
        sorted_labels = sorted(set(int_labels), key=lambda x: x)
        palette = sns.color_palette(color_palette, max(sorted_labels))
        colors = [palette[x - 1] for x in sorted_labels]
        return sorted_labels, colors
    else:
        def sort_key(s):
            s = str(s).strip()
            if not s.startswith('CN'):
                return (999, 0)
            rest = s[2:]
            if '-' in rest:
                parts = rest.split('-', 1)
                return (int(parts[0]) if parts[0].isdigit() else 999,
                        int(parts[1]) if parts[1].isdigit() else 0)
            return (int(rest) if rest.isdigit() else 999, 0)

        sorted_labels = sorted(set(labels), key=sort_key)
        n = len(sorted_labels)
        palette = sns.color_palette(color_palette, max(n, 20))[:n]
        colors = list(palette)
        return sorted_labels, colors


def _get_spatial_coords(adata, coord_key: str = 'spatial'):
    """Get spatial coordinates with fallback options."""
    if coord_key in adata.obsm:
        return adata.obsm[coord_key]
    elif 'spatial' in adata.obsm:
        return adata.obsm['spatial']
    return None


def generate_individual_tile_maps(
    source_h5ad_dir: str,
    cn_labels_dir: str,
    output_dir: str,
    coord_key: str = 'spatial',
    point_size: float = 10.0,
    palette: str = 'tab20',
    k: Optional[int] = None,
    n_clusters: Optional[int] = None,
    tile_selection: Optional[Set[str]] = None,
):
    """
    Generate one spatial CN scatter plot per tile.

    Reads each tile's original source h5ad (cell_type + spatial, untouched by
    clustering) one at a time, merges in that tile's CN labels from the
    matching lightweight JSON file, and plots. Nothing new is written to disk
    except the PNG — no annotated h5ad copy is created or required.

    If tile_selection is given (see load_tile_selection), only those tiles
    are plotted, even if more are available in source_h5ad_dir/cn_labels_dir.
    """
    source_h5ad_dir = Path(source_h5ad_dir)
    cn_labels_dir = Path(cn_labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5ad_files = sorted(source_h5ad_dir.glob('*.h5ad'))
    if not h5ad_files:
        raise ValueError(f"No h5ad files found in {source_h5ad_dir}")

    if tile_selection is not None:
        found_stems = {f.stem for f in h5ad_files}
        missing = tile_selection - found_stems
        if missing:
            sample = sorted(missing)[:5]
            print(f"  Warning: {len(missing)} tile(s) from the selection list were not "
                  f"found in {source_h5ad_dir}: {sample}{' ...' if len(missing) > 5 else ''}")
        h5ad_files = [f for f in h5ad_files if f.stem in tile_selection]
        print(f"  Tile selection applied: keeping {len(h5ad_files)} tiles")
        if not h5ad_files:
            raise ValueError("No tiles remain after applying tile_selection")

    cn_label_files = sorted(cn_labels_dir.glob('*_cn_labels.json'))
    suffix = '_cn_labels'
    cn_label_map = {}
    for f in cn_label_files:
        stem = f.stem
        if stem.endswith(suffix):
            cn_label_map[stem[:-len(suffix)]] = f
    if not cn_label_map:
        raise ValueError(f"No *_cn_labels.json files found in {cn_labels_dir}")

    print(f"Found {len(h5ad_files)} source h5ad files")
    print(f"Found {len(cn_label_map)} CN-label JSON files")
    print(f"Generating individual spatial CN maps for each tile...")

    n_saved = 0
    n_skipped_no_labels = 0
    for tile_idx, h5ad_file in enumerate(h5ad_files, 1):
        tile_name = h5ad_file.stem
        print(f"  [{tile_idx}/{len(h5ad_files)}] Plotting {tile_name}")

        if tile_name not in cn_label_map:
            print(f"    Warning: no CN-label JSON found for {tile_name}, skipping")
            n_skipped_no_labels += 1
            continue

        try:
            adata = ad.read_h5ad(h5ad_file)
        except Exception as e:
            print(f"    ✗ Error loading {h5ad_file}: {str(e)}")
            continue

        coords = _get_spatial_coords(adata, coord_key)
        if coords is None:
            print(f"    Warning: No spatial coordinates found for {tile_name}, skipping...")
            continue

        with open(cn_label_map[tile_name], 'r') as f:
            payload = json.load(f)
        labels = payload.get('labels', {})

        prefix = f"{tile_name}_"
        cn_values = []
        matched_mask = []
        for obs_name in adata.obs_names:
            nucleus_id = str(obs_name)[len(prefix):] if str(obs_name).startswith(prefix) else str(obs_name)
            if nucleus_id in labels:
                cn_values.append(labels[nucleus_id])
                matched_mask.append(True)
            else:
                cn_values.append(None)
                matched_mask.append(False)

        matched_mask = np.array(matched_mask)
        n_unmatched = (~matched_mask).sum()
        if n_unmatched:
            print(f"    Warning: {n_unmatched}/{adata.n_obs} cells in {tile_name} had no "
                  f"matching CN label; excluding them from the plot")

        if not matched_mask.any():
            print(f"    Warning: no cells in {tile_name} matched a CN label, skipping tile")
            continue

        coords = coords[matched_mask]
        cn_labels_arr = np.array([cn_values[j] for j in range(len(cn_values)) if matched_mask[j]])
        n_cells_plotted = matched_mask.sum()

        resolved_n_clusters = n_clusters if n_clusters is not None else len(np.unique(cn_labels_arr))
        if n_clusters is not None and len(np.unique(cn_labels_arr)) != n_clusters:
            print(f"  Warning: n_clusters={n_clusters} was requested, but data has "
                  f"{len(np.unique(cn_labels_arr))} unique CN labels; using that for titles.")
            resolved_n_clusters = len(np.unique(cn_labels_arr))

        fig, ax = plt.subplots(figsize=(10, 10))

        unique_cns = np.unique(cn_labels_arr)
        sorted_labels, colors_list = _sort_cn_labels_and_colors(unique_cns, palette)
        label_to_color = dict(zip(sorted_labels, colors_list))

        for cn_id in unique_cns:
            cn_mask = cn_labels_arr == cn_id
            color = label_to_color.get(cn_id, colors_list[0] if colors_list else 'gray')
            legend_label = str(cn_id) if str(cn_id).startswith('CN') else f'CN {cn_id}'
            ax.scatter(
                coords[cn_mask, 0],
                coords[cn_mask, 1],
                c=[color],
                s=point_size,
                alpha=0.7,
                label=legend_label
            )

        ax.set_xlabel('X coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y coordinate (pixels)', fontsize=12)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        title = f'Cellular Neighborhoods: {tile_name}'
        if k is not None:
            title += f'\n(k={k}, n_clusters={resolved_n_clusters}, {n_cells_plotted:,} cells)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.tight_layout()

        k_str = f"k{k}" if k is not None else "kNA"
        save_path = output_dir / f'{k_str}_ncluster{resolved_n_clusters}-{tile_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    ✓ Saved to: {save_path}")
        n_saved += 1

    if n_skipped_no_labels:
        print(f"\n  Note: {n_skipped_no_labels} source tile(s) had no matching CN-label file "
              f"and were skipped entirely.")

    print(f"\n✓ Generated {n_saved} spatial CN maps")
    print(f"  Results saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Generate individual per-tile spatial cellular neighborhood maps.'
    )
    parser.add_argument(
        '--source_h5ad_dir',
        required=True,
        help='Directory containing the ORIGINAL h5ad tiles from data_preparation.py '
             '(named "{tile_name}.h5ad"; already has cell_type + spatial)'
    )
    parser.add_argument(
        '--cn_labels_dir',
        required=True,
        help='Directory containing the lightweight per-tile CN-label JSON files '
             '(output_dir/cn_labels/ from cn_unified_kmeans_local.py)'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for the per-tile spatial map PNGs'
    )
    parser.add_argument(
        '--coord_key',
        default='spatial',
        help='Key in adata.obsm containing spatial coordinates (default: spatial)'
    )
    parser.add_argument(
        '--point_size',
        type=float,
        default=10.0,
        help='Marker size for each cell in the scatter plot (default: 10.0)'
    )
    parser.add_argument(
        '--palette',
        default='tab20',
        help='Color palette (default: tab20, supports up to 20 distinct colors)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=None,
        help='Number of nearest neighbors used (for titles only; optional)'
    )
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=None,
        help='Number of clusters used (for titles only). If omitted, inferred '
             'from unique CN labels in each tile.'
    )
    parser.add_argument(
        '--tile_list_csv',
        default=None,
        help="Optional CSV with a 'tile' column listing which tile names to "
             "plot (e.g. 'JN_TS_001_tile_12883_7423'), one per row. Useful "
             "for a quick spot-check on a subset without touching "
             "source_h5ad_dir or cn_labels_dir. Leave unset to plot all tiles."
    )

    args = parser.parse_args()

    tile_selection = load_tile_selection(args.tile_list_csv) if args.tile_list_csv else None

    generate_individual_tile_maps(
        source_h5ad_dir=args.source_h5ad_dir,
        cn_labels_dir=args.cn_labels_dir,
        output_dir=args.output_dir,
        coord_key=args.coord_key,
        point_size=args.point_size,
        palette=args.palette,
        k=args.k,
        n_clusters=args.n_clusters,
        tile_selection=tile_selection,
    )


if __name__ == '__main__':
    main()
