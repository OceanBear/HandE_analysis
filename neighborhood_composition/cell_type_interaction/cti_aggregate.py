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
    cluster_key='cell_type'
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
        n_neighbors=n_neighbors
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
    print(f"  - Aggregated mean enrichment: aggregated_mean_enrichment.png")
    print(f"  - Variability across tiles: aggregated_variability.png")
    print(f"  - Mean z-scores: aggregated_mean_zscore.csv")
    print(f"  - Std z-scores: aggregated_std_zscore.csv")
    print(f"  - Median z-scores: aggregated_median_zscore.csv")
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
    
    args = parser.parse_args()
    
    # Run aggregation
    results = aggregate_results(
        input_dir=args.input_dir,
        n_perms=args.n_perms,
        n_neighbors=args.n_neighbors,
        cluster_key=args.cluster_key
    )
    
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
Aggregated Analysis Results:

1. AGGREGATED MEAN ENRICHMENT:
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

RECOMMENDATIONS:
- Focus on interactions with high consistency across tiles
- High variability suggests heterogeneous tissue regions
- Compare individual tiles to identify region-specific patterns
- Use aggregated results for overall tissue-level conclusions
""")
    
    return results


if __name__ == "__main__":
    main()
