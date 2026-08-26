"""
Cellular Neighborhood Cluster-Count Sweep

Testing several n_clusters values naively means re-running the ENTIRE pipeline
from scratch for each one — re-loading every tile, rebuilding the spatial KNN
graph, and re-running neighbor aggregation — even though none of those steps
actually depend on n_clusters. Only the final k-means clustering step does.

This script runs the expensive, n_clusters-independent steps (load tiles ->
build KNN graph -> aggregate neighbor composition) exactly ONCE, then loops
over a range of n_clusters values doing only the fast clustering + diagnostics
step for each. It reuses UnifiedCellularNeighborhoodDetector from
cn_unified_kmeans_local.py directly, so the two scripts can't drift apart.

For each n_clusters value, saves the same outputs as a normal single run
(composition CSVs, frequency CSVs, CN-label JSON files, summary JSON) to its
own k{K}_nclusters{N}_seed{S}/ subfolder, plus one combined sweep_summary.csv
at the top level with two diagnostics to help choose n_clusters:

- inertia: k-means' within-cluster sum of squared distances. Always decreases
  as n_clusters increases, so look for the "elbow" — the point where adding
  more clusters stops giving a big drop and the curve flattens out.
- silhouette_score: how well-separated the clusters are (-1 to 1, higher is
  better). Computed on a random subsample of cells (see
  --silhouette_sample_size), since the exact calculation is O(n^2) and
  infeasible on a full multi-million-cell dataset.

This script produces the same CSV/JSON outputs as before, plus one summary
plot: sweep_summary.png, showing inertia (elbow method) and silhouette score
side by side across the tested n_clusters values.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cn_unified_kmeans_local import (
    UnifiedCellularNeighborhoodDetector,
    DEFAULT_RANDOM_STATE,
    load_tile_selection,
)


def plot_sweep_summary(summary_df, save_path):
    """
    Save a simple two-panel PNG: inertia vs n_clusters (elbow method) and
    silhouette score vs n_clusters, each as a line with dot markers at every
    tested value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(summary_df['n_clusters'], summary_df['inertia'], marker='o', linewidth=2)
    axes[0].set_xlabel('n_clusters')
    axes[0].set_ylabel('Inertia (within-cluster sum of squares)')
    axes[0].set_title('Elbow Method')
    axes[0].set_xticks(summary_df['n_clusters'])
    axes[0].grid(alpha=0.3, linestyle='--')

    axes[1].plot(summary_df['n_clusters'], summary_df['silhouette_score'],
                 marker='o', linewidth=2, color='darkorange')
    axes[1].set_xlabel('n_clusters')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score (higher is better)')
    axes[1].set_xticks(summary_df['n_clusters'])
    axes[1].grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_sweep(
    tiles_dir,
    output_dir,
    n_values,
    k=20,
    celltype_key='cell_type',
    pattern='*.h5ad',
    max_tiles=None,
    coord_offset=True,
    random_state=None,
    tile_selection=None,
    silhouette_sample_size=20000,
):
    """
    Run the expensive shared steps once, then sweep n_clusters.

    Returns the sweep summary as a DataFrame (also saved to
    output_dir/sweep_summary.csv).
    """
    if random_state is None:
        random_state = DEFAULT_RANDOM_STATE

    base_output_dir = Path(output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Use the first n_clusters value's own subfolder as the detector's initial
    # output_dir, so __init__'s directory setup isn't wasted on a throwaway path.
    first_subdir = base_output_dir / f"k{k}_nclusters{n_values[0]}_seed{random_state}"
    detector = UnifiedCellularNeighborhoodDetector(
        tiles_directory=tiles_dir,
        output_dir=str(first_subdir),
    )

    tile_files = detector.discover_tiles(pattern=pattern, max_tiles=max_tiles, tile_selection=tile_selection)
    if not tile_files:
        print("No tiles found! Exiting...")
        return None

    banner = "=" * 80
    print(f"\n{banner}\nSWEEP: computing shared features once (load -> KNN graph -> aggregate)\n{banner}")
    detector.load_and_combine_tiles(tile_files, celltype_key, coord_offset)
    detector.build_knn_graph(k=k)
    detector.aggregate_neighbors(celltype_key=celltype_key)
    print(f"\n✓ Shared features ready: {detector.combined_adata.n_obs:,} cells across "
          f"{len(detector.tile_list)} tiles\n")

    aggregated = detector.combined_adata.obsm['aggregated_neighbors']
    n_cells = aggregated.shape[0]
    rng = np.random.default_rng(random_state)
    if n_cells > silhouette_sample_size:
        sample_idx = rng.choice(n_cells, size=silhouette_sample_size, replace=False)
    else:
        sample_idx = np.arange(n_cells)
    agg_sample = aggregated[sample_idx]

    summary_rows = []

    for n_clusters in n_values:
        print(f"\n{banner}\nn_clusters = {n_clusters}\n{banner}")

        run_subdir = base_output_dir / f"k{k}_nclusters{n_clusters}_seed{random_state}"
        detector.output_dir = run_subdir
        detector.output_dir.mkdir(parents=True, exist_ok=True)
        (detector.output_dir / 'unified_analysis').mkdir(exist_ok=True)

        detector.detect_cellular_neighborhoods(n_clusters=n_clusters, random_state=random_state)
        inertia = detector.last_inertia_

        cn_labels_sample = detector.cn_labels[sample_idx]
        try:
            sil = silhouette_score(agg_sample, cn_labels_sample)
        except ValueError as e:
            print(f"  Warning: could not compute silhouette score: {e}")
            sil = float('nan')

        composition, composition_zscore = detector.compute_unified_cn_composition(celltype_key=celltype_key)
        composition.to_csv(detector.output_dir / 'unified_analysis' / 'unified_cn_composition.csv')
        composition_zscore.to_csv(detector.output_dir / 'unified_analysis' / 'unified_cn_composition_zscore.csv')

        freq_overall = detector.calculate_neighborhood_frequency(group_by_tile=False)
        freq_overall.to_csv(detector.output_dir / 'unified_analysis' / 'neighborhood_frequency_overall.csv', index=False)
        freq_per_tile = detector.calculate_neighborhood_frequency(group_by_tile=True)
        freq_per_tile.to_csv(detector.output_dir / 'unified_analysis' / 'neighborhood_frequency_per_tile.csv')

        detector.save_cn_labels(n_clusters=n_clusters)
        detector.save_summary_statistics(
            k=k, n_clusters=n_clusters, celltype_key=celltype_key,
            composition=composition, random_state=random_state
        )

        print(f"  Inertia: {inertia:,.2f}")
        print(f"  Silhouette score (sampled n={len(sample_idx):,}): {sil:.4f}")

        summary_rows.append({
            'n_clusters': n_clusters,
            'k': k,
            'random_state': random_state,
            'inertia': inertia,
            'silhouette_score': sil,
            'silhouette_sample_size': len(sample_idx),
            'n_cells_total': n_cells,
            'output_dir': str(detector.output_dir),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = base_output_dir / 'sweep_summary.csv'
    summary_df.to_csv(summary_path, index=False)

    plot_path = base_output_dir / 'sweep_summary.png'
    plot_sweep_summary(summary_df, plot_path)

    print(f"\n{banner}\nSWEEP COMPLETE\n{banner}")
    print(summary_df.to_string(index=False))
    print(f"\n✓ Saved sweep summary to: {summary_path}")
    print(f"✓ Saved sweep summary plot to: {plot_path}")
    print("  Pick n_clusters via the elbow method (where 'inertia' stops dropping "
          "sharply) or by maximizing 'silhouette_score'.")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Sweep n_clusters for cellular neighborhood detection, computing "
                     "the expensive load/KNN-graph/aggregation steps only once."
    )
    parser.add_argument('--tiles_dir', '-t', required=True, help='Directory containing h5ad tile files')
    parser.add_argument(
        '--output_dir', '-o', required=True,
        help='Base output directory. Each n_clusters value gets its own '
             'kK_nclustersN_seedS/ subfolder (same layout as cn_unified_kmeans_local.py), '
             'plus a top-level sweep_summary.csv comparing all of them.'
    )
    parser.add_argument('--k', type=int, default=20, help='Number of nearest neighbors (default: 20)')
    parser.add_argument('--n_start', type=int, required=True, help='First n_clusters value to test (inclusive)')
    parser.add_argument('--n_end', type=int, required=True, help='Last n_clusters value to test (inclusive)')
    parser.add_argument('--n_step', type=int, default=1, help='Step between n_clusters values (default: 1)')
    parser.add_argument('--celltype_key', '-c', default='cell_type', help='Column name for cell types (default: cell_type)')
    parser.add_argument('--max_tiles', '-m', type=int, default=None, help='Maximum number of tiles to process (for testing)')
    parser.add_argument('--pattern', '-p', default='*.h5ad', help='File pattern to match (default: *.h5ad)')
    parser.add_argument('--no_offset', action='store_true', help='Disable spatial coordinate offsetting between tiles')
    parser.add_argument(
        '--random_state', '-r', type=int, default=None,
        help=f'Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE})'
    )
    parser.add_argument(
        '--tile_list_csv', default=None,
        help="Optional CSV with a 'tile' column restricting which tiles are included"
    )
    parser.add_argument(
        '--silhouette_sample_size', type=int, default=20000,
        help='Max cells to subsample for the silhouette score, since exact computation '
             'is O(n^2) and infeasible on a full large dataset (default: 20000)'
    )

    args = parser.parse_args()

    if args.n_start > args.n_end:
        raise SystemExit(f"--n_start ({args.n_start}) must be <= --n_end ({args.n_end})")
    if args.n_step <= 0:
        raise SystemExit(f"--n_step must be positive (got {args.n_step})")

    n_values = list(range(args.n_start, args.n_end + 1, args.n_step))
    random_state = args.random_state if args.random_state is not None else DEFAULT_RANDOM_STATE
    tile_selection = load_tile_selection(args.tile_list_csv) if args.tile_list_csv else None

    print(f"Sweeping n_clusters over: {n_values}")

    run_sweep(
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
        n_values=n_values,
        k=args.k,
        celltype_key=args.celltype_key,
        pattern=args.pattern,
        max_tiles=args.max_tiles,
        coord_offset=not args.no_offset,
        random_state=random_state,
        tile_selection=tile_selection,
        silhouette_sample_size=args.silhouette_sample_size,
    )


if __name__ == '__main__':
    main()
