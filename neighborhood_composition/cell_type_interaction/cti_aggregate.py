"""
Aggregate Results from Batch Cell Type Interaction Analysis

This script reads pre-processed tile results from cti_batch.py and generates
aggregated statistics, visualizations, and summary reports.

Features:
- Aggregates z-scores across all processed tiles
- Computes mean, std, median statistics
- Generates interaction consistency analysis
- Creates summary reports and visualizations
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import warnings
import os

# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)

# Import functions from cti_tiled.py
from cti_tiled import (
    load_intermediate_results,
    aggregate_from_saved_results
)

warnings.filterwarnings('ignore')

# Legacy 7-class epithelium labels (--merge-epithelium only; not used by type_info_4class.json)
DEFAULT_EPITHELIUM_A = "Epithelium (PD-L1lo/Ki67lo)"
DEFAULT_EPITHELIUM_B = "Epithelium (PD-L1hi/Ki67hi)"


def merge_symmetric_celltype_zscore_df(
    df: pd.DataFrame,
    old_a: str = None,
    old_b: str = None,
    new_label: str = "Tumor",
) -> pd.DataFrame:
    """
    Merge two cell-type row/column labels in a square z-score matrix.

    When both labels exist, each merged entry is the nan-mean of the corresponding
    sub-block (rows × cols) from the original matrix. When only one exists, it is
    renamed to ``new_label``.
    """
    if old_a is None:
        old_a = DEFAULT_EPITHELIUM_A
    if old_b is None:
        old_b = DEFAULT_EPITHELIUM_B

    if df.shape[0] != df.shape[1] or not df.index.equals(df.columns):
        raise ValueError("merge_symmetric_celltype_zscore_df expects a square matrix with matching index/columns")

    has_a = old_a in df.index
    has_b = old_b in df.index
    if not has_a and not has_b:
        return df.copy()

    if not has_a or not has_b:
        present = old_a if has_a else old_b
        out = df.rename(index={present: new_label}, columns={present: new_label})
        return out

    new_order = []
    merged_marker = False
    for c in df.index:
        if c in (old_a, old_b):
            if not merged_marker:
                new_order.append(new_label)
                merged_marker = True
        else:
            new_order.append(c)

    def _sources(lbl):
        if lbl == new_label:
            return [x for x in (old_a, old_b) if x in df.index]
        return [lbl] if lbl in df.index else []

    n = len(new_order)
    data = np.full((n, n), np.nan, dtype=float)
    for i, ri in enumerate(new_order):
        for j, cj in enumerate(new_order):
            rs, cs = _sources(ri), _sources(cj)
            vals = [df.loc[r, c] for r in rs for c in cs if r in df.index and c in df.columns]
            if vals:
                data[i, j] = float(np.nanmean(vals))

    return pd.DataFrame(data, index=new_order, columns=new_order)


def merge_epithelium_in_interactions_df(
    df: pd.DataFrame,
    old_a: str = None,
    old_b: str = None,
    new_label: str = "Tumor",
) -> pd.DataFrame:
    """Map epithelium pair to ``new_label`` in Cell Type 1 / Cell Type 2 columns if present."""
    if old_a is None:
        old_a = DEFAULT_EPITHELIUM_A
    if old_b is None:
        old_b = DEFAULT_EPITHELIUM_B

    out = df.copy()
    if "Cell Type 1" not in out.columns or "Cell Type 2" not in out.columns:
        return out

    def _m(x):
        if pd.isna(x):
            return x
        s = str(x)
        return new_label if s in (old_a, old_b) else s

    out["Cell Type 1"] = out["Cell Type 1"].map(_m)
    out["Cell Type 2"] = out["Cell Type 2"].map(_m)
    return out


# Short names for aggregated heatmap axis labels (matches type_info_4class.json / data_preparation.py)
DEFAULT_CELL_TYPE_DISPLAY_ABBREV = {
    "Others": "Oth",
    "Tumor": "Tum",
    "Lymphocyte": "Lym",
    "Fibroblast/Stroma": "Fib/Str",
}


def cell_type_display_abbrev(label, abbrev_map=None):
    """Map one cell type string to a short plot label; unknown names pass through unchanged."""
    if abbrev_map is None:
        abbrev_map = DEFAULT_CELL_TYPE_DISPLAY_ABBREV
    s = str(label)
    if s in abbrev_map:
        return abbrev_map[s]
    lower_map = {k.lower(): v for k, v in abbrev_map.items()}
    if s.lower() in lower_map:
        return lower_map[s.lower()]
    return s


def cell_types_display_labels(cell_types, abbrev_map=None, enabled=True):
    """List of axis labels for heatmaps (full names if ``enabled`` is False)."""
    if not enabled:
        return list(cell_types)
    return [cell_type_display_abbrev(ct, abbrev_map) for ct in cell_types]


def find_processed_tiles(input_dir):
    """
    Find all processed tiles in the input directory.
    
    Parameters:
    -----------
    input_dir : str or Path
        Directory containing processed tile results
        
    Returns:
    --------
    tile_dirs : list of Path
        List of tile directory paths
    tile_names : list of str
        List of tile names
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all subdirectories that contain the required intermediate files
    tile_dirs = []
    tile_names = []
    
    for item in input_dir.iterdir():
        if item.is_dir():
            # Check if this directory has the required intermediate files
            zscore_file = item / f'{item.name}_zscore.npy'
            metadata_file = item / f'{item.name}_metadata.json'
            interactions_file = item / f'{item.name}_significant_interactions.csv'
            
            if zscore_file.exists() and metadata_file.exists() and interactions_file.exists():
                tile_dirs.append(item)
                tile_names.append(item.name)
    
    if len(tile_dirs) == 0:
        raise FileNotFoundError(
            f"No processed tiles found in {input_dir}. "
            f"Make sure tiles have been processed with cti_batch.py first."
        )
    
    print(f"Found {len(tile_dirs)} processed tiles in {input_dir}")
    for i, tile_name in enumerate(tile_names[:10]):  # Show first 10
        print(f"  {i+1}. {tile_name}")
    if len(tile_names) > 10:
        print(f"  ... and {len(tile_names) - 10} more tiles")
    
    return tile_dirs, tile_names


def aggregate_results(
    input_dir='cti_multiple_tiles', # was 'cti_multiple_tiles'
    n_perms=None,
    n_neighbors=None,
    cluster_key='cell_type',
    *,
    merge_epithelium_to_tumor=False,
    tumor_label='Tumor',
    cti_heatmap_annot_fontsize=32,
    use_short_cell_type_labels_in_plots=True,
    cell_type_abbrev_map=None,
    schapiro_sum_sigval=True,
    sigval_method='p_from_z',
    sigval_alpha=0.05,
    sigval_z_threshold=2.0,
):
    """
    Aggregate results from processed tiles.
    
    Parameters:
    -----------
    input_dir : str or Path, default='cti_multiple_tiles'
        Directory containing processed tile results
    n_perms : int, optional
        Number of permutations used (for display in plots)
    n_neighbors : int, optional
        Number of neighbors used (for display in plots)
    cluster_key : str, default='cell_type'
        Key for cell type labels
    merge_epithelium_to_tumor : bool, default=False
        Merge legacy 7-class epithelium labels into ``tumor_label`` (only needed for old h5ad outputs).
    tumor_label : str, default='Tumor'
        Target label after merging epithelium A/B.
    cti_heatmap_annot_fontsize : float, default=12
        Font size for z-score text inside aggregated mean / variability heatmap cells
        (see ``cti_tiled.aggregate_from_saved_results`` → ``sns.heatmap(..., annot_kws=...)``).
    use_short_cell_type_labels_in_plots : bool, default=True
        If True, x/y tick labels use ``DEFAULT_CELL_TYPE_DISPLAY_ABBREV`` (or ``cell_type_abbrev_map``).
        CSV outputs still use full cell type names.
    cell_type_abbrev_map : dict optional
        Override mapping full name → short label for plot ticks only.
    schapiro_sum_sigval : bool, default=True
        If True, compute summed Schapiro-style ``sigval`` across tiles (see ``cti_tiled``).
    sigval_method : str, default='p_from_z'
        Rule for per-tile significance coding before summing.
    sigval_alpha : float, default=0.05
        Alpha for ``sigval`` when using p-values / ``p_from_z``.
    sigval_z_threshold : float, default=2.0
        |z| threshold when ``sigval_method == 'z_threshold'``.

    Returns:
    --------
    results : dict
        Dictionary containing aggregated results
    """
    input_dir = Path(input_dir)
    
    print("=" * 70)
    print("AGGREGATING BATCH PROCESSED RESULTS")
    print("=" * 70)
    print(f"\nInput directory: {input_dir}")
    if merge_epithelium_to_tumor:
        print(
            f"Epithelium merge: ON -> '{DEFAULT_EPITHELIUM_A}' + '{DEFAULT_EPITHELIUM_B}' => '{tumor_label}'"
        )
    else:
        print("Epithelium merge: OFF")
    if use_short_cell_type_labels_in_plots:
        print("Heatmap axis labels: short abbreviations enabled (CSV rows unchanged).")
    else:
        print("Heatmap axis labels: full cell type names.")

    # Find processed tiles
    tile_dirs, tile_names = find_processed_tiles(input_dir)
    
    # STEP 1: Aggregate z-scores from saved results
    print("\n" + "=" * 70)
    print("STEP 1: AGGREGATING Z-SCORES")
    print("=" * 70)
    
    aggregated = aggregate_from_saved_results(
        tile_dirs=tile_dirs,
        output_dir=input_dir,
        tile_names=tile_names,
        n_perms=n_perms,
        n_neighbors=n_neighbors,
        merge_epithelium_to_tumor=merge_epithelium_to_tumor,
        tumor_label=tumor_label,
        cti_heatmap_annot_fontsize=cti_heatmap_annot_fontsize,
        use_short_cell_type_labels_in_plots=use_short_cell_type_labels_in_plots,
        cell_type_abbrev_map=cell_type_abbrev_map,
        schapiro_sum_sigval=schapiro_sum_sigval,
        sigval_method=sigval_method,
        sigval_alpha=sigval_alpha,
        sigval_z_threshold=sigval_z_threshold,
    )
    
    # STEP 2: Create summary report from saved metadata
    print("\n" + "=" * 70)
    print("STEP 2: CREATING SUMMARY REPORT")
    print("=" * 70)
    
    summary_data = []
    for tile_name, tile_dir in zip(tile_names, tile_dirs):
        try:
            # Load metadata from saved files
            metadata_result = load_intermediate_results(tile_dir, tile_name=tile_name)
            interactions_csv = tile_dir / f'{tile_name}_significant_interactions.csv'
            
            if interactions_csv.exists():
                interactions_df = pd.read_csv(interactions_csv)
                n_interactions = len(interactions_df)
            else:
                n_interactions = 0
            
            summary_data.append({
                'Tile': tile_name,
                'N_Cells': metadata_result['n_cells'],
                'N_Significant_Interactions': n_interactions,
                'Mean_Abs_Zscore': metadata_result['metadata']['mean_abs_zscore'],
                'Max_Abs_Zscore': metadata_result['metadata']['max_abs_zscore']
            })
        except Exception as e:
            print(f"  ⚠ Warning: Could not load summary for {tile_name}: {e}")
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('N_Cells', ascending=False)
    summary_df.to_csv(input_dir / 'tiles_summary.csv', index=False)
    
    print(f"\nTiles summary:")
    print(summary_df.to_string(index=False))
    print(f"\n  - Saved tiles_summary.csv")
    
    # STEP 3: Aggregate interactions from CSVs
    print("\n" + "=" * 70)
    print("STEP 3: AGGREGATING INTERACTIONS")
    print("=" * 70)
    
    print("\nAggregating interaction CSVs...")
    all_interactions = []
    for tile_name, tile_dir in zip(tile_names, tile_dirs):
        interactions_csv = tile_dir / f'{tile_name}_significant_interactions.csv'
        if interactions_csv.exists():
            try:
                tile_interactions = pd.read_csv(interactions_csv)
                tile_interactions['Tile'] = tile_name
                all_interactions.append(tile_interactions)
            except Exception as e:
                print(f"  ⚠ Warning: Could not load interactions for {tile_name}: {e}")
    
    if all_interactions:
        combined_interactions = pd.concat(all_interactions, ignore_index=True)
        if merge_epithelium_to_tumor:
            combined_interactions = merge_epithelium_in_interactions_df(
                combined_interactions, new_label=tumor_label
            )
        combined_interactions.to_csv(input_dir / 'all_tiles_interactions.csv', index=False)
        
        # Interaction consistency
        interaction_counts = combined_interactions.groupby(['Cell Type 1', 'Cell Type 2', 'Interaction']).size()
        interaction_counts = interaction_counts.reset_index(name='Count')
        interaction_counts['Frequency'] = interaction_counts['Count'] / len(tile_names)
        interaction_counts = interaction_counts.sort_values('Count', ascending=False)
        interaction_counts.to_csv(input_dir / 'interaction_consistency.csv', index=False)
        
        print(f"  - Saved all_tiles_interactions.csv ({len(combined_interactions)} total interactions)")
        print(f"  - Saved interaction_consistency.csv")
        print(f"\nMost consistent interactions (present in multiple tiles):")
        print(interaction_counts.head(10).to_string(index=False))
    else:
        print("  ⚠ Warning: No interaction files found")
    
    # Final results
    results = {
        'aggregated': aggregated,
        'summary': summary_df,
        'successful_tiles': tile_names,
        'n_tiles': len(tile_names),
        'input_dir': str(input_dir)
    }
    
    print("\n" + "=" * 70)
    print("AGGREGATION COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {input_dir}/")
    print(f"\nKey outputs:")
    print(f"  - Aggregated mean CTI: aggregated_mean_cti.png")
    print(f"  - Variability across tiles: aggregated_variability.png")
    print(f"  - Mean z-scores: aggregated_mean_zscore.csv")
    print(f"  - Std z-scores: aggregated_std_zscore.csv")
    print(f"  - Median z-scores: aggregated_median_zscore.csv")
    if schapiro_sum_sigval:
        print(f"  - Schapiro-style summed sigval: aggregated_summed_sigval.csv / .png")
    print(f"  - All interactions: all_tiles_interactions.csv")
    print(f"  - Interaction consistency: interaction_consistency.csv")
    print(f"  - Tiles summary: tiles_summary.csv")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Aggregate results from batch processed cell type interaction analysis'
    )
    parser.add_argument(
        '--input_dir',
        default='cti_multiple_tiles',
        help='Directory containing processed tile results (default: cti_multiple_tiles)'
    )
    parser.add_argument(
        '--n_perms',
        type=int,
        default=None,
        help='Number of permutations used (for display in plots)'
    )
    parser.add_argument(
        '--n_neighbors',
        type=int,
        default=None,
        help='Number of neighbors used (for display in plots)'
    )
    parser.add_argument(
        '--cluster_key',
        default='cell_type',
        help='Key for cell type labels (default: cell_type)'
    )
    parser.add_argument(
        '--merge-epithelium',
        action='store_true',
        help='Merge legacy 7-class epithelium labels into Tumor (for old h5ad outputs only).',
    )
    parser.add_argument(
        '--tumor-label',
        default='Tumor',
        help='Label used after merging epithelium types (default: Tumor).',
    )
    parser.add_argument(
        '--heatmap-annot-fontsize',
        type=float,
        default=16,
        help='Font size for z-score numbers inside aggregated heatmaps (default: 12).',
    )
    parser.add_argument(
        '--no-short-cell-type-labels',
        action='store_true',
        help='Use full cell type names on aggregated heatmap axes (default: short labels on).',
    )

    parser.add_argument(
        '--no-schapiro-sum',
        action='store_true',
        help='Disable Schapiro-style summed sigval heatmap/CSV (default: enabled).',
    )
    parser.add_argument(
        '--sigval-method',
        choices=('p_from_z', 'z_threshold'),
        default='p_from_z',
        help='How to assign per-tile sigval in {-1,0,1} before summing (default: p_from_z).',
    )
    parser.add_argument(
        '--sigval-alpha',
        type=float,
        default=0.05,
        help='Alpha for sigval when using p_from_z or Squidpy p-values (default: 0.05).',
    )
    parser.add_argument(
        '--sigval-z-threshold',
        type=float,
        default=2.0,
        help='|z| threshold when --sigval-method z_threshold (default: 2.0).',
    )

    args = parser.parse_args()

    # Run aggregation
    results = aggregate_results(
        input_dir=args.input_dir,
        n_perms=args.n_perms,
        n_neighbors=args.n_neighbors,
        cluster_key=args.cluster_key,
        merge_epithelium_to_tumor=args.merge_epithelium,
        tumor_label=args.tumor_label,
        cti_heatmap_annot_fontsize=args.heatmap_annot_fontsize,
        use_short_cell_type_labels_in_plots=not args.no_short_cell_type_labels,
        schapiro_sum_sigval=not args.no_schapiro_sum,
        sigval_method=args.sigval_method,
        sigval_alpha=args.sigval_alpha,
        sigval_z_threshold=args.sigval_z_threshold,
    )
    
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
Aggregated Analysis Results:

1. AGGREGATED MEAN CTI (CELL TYPE INTERACTION):
   - Average z-scores across all tiles
   - Shows consistent spatial patterns
   - More robust than single tile analysis

2. VARIABILITY ACROSS TILES:
   - Standard deviation of z-scores
   - High variability = interaction varies by tile/region
   - Low variability = consistent pattern across all tiles

3. INTERACTION CONSISTENCY:
   - Shows which interactions appear in multiple tiles
   - Frequency = proportion of tiles with this interaction
   - High frequency = robust, reproducible pattern

4. TILES SUMMARY:
   - Overview of all processed tiles
   - Cell counts and interaction statistics
   - Use to identify outlier tiles

5. SUMMED SIGVAL (SCHAPIRO-STYLE, if enabled):
   - Each tile: +1 interaction, -1 avoidance, 0 not significant (from aligned z-scores)
   - Heatmap/CSV: sum across tiles (range about -N to +N for N tiles)
   - Not identical to imcRtools::testInteractions unless p-value rules match

RECOMMENDATIONS:
- Focus on interactions with high consistency across tiles
- High variability suggests heterogeneous tissue regions
- Compare individual tiles to identify region-specific patterns
- Use aggregated results for overall tissue-level conclusions
""")
    
    return results


if __name__ == "__main__":
    main()
