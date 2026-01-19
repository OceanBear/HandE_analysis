"""
Neighborhood Enrichment Analysis for Multiple Tiled Images

This script processes multiple tiled images from a directory and performs
spatial neighborhood enrichment analysis on each tile individually.

Features:
- Batch processing of multiple h5ad files
- Individual analysis results for each tile
- Summary statistics across all tiles
- Consolidated results and visualizations

Author: Generated with Claude Code
Date: 2025-10-22
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent figures from popping up
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode to prevent figures from displaying
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
import warnings
import os
from pathlib import Path
# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)

# Import functions from cti_tiled.py
from cti_tiled import (
    load_and_apply_cell_type_colors,
    build_spatial_graph,
    neighborhood_enrichment_analysis,
    compute_centrality_scores,
    visualize_enrichment,
    visualize_spatial_distribution,
    summarize_interactions,
    save_intermediate_results,
    load_intermediate_results,
    aggregate_from_saved_results
)

warnings.filterwarnings('ignore')


def find_h5ad_files(directory, pattern='*.h5ad'):
    """
    Find all h5ad files in a directory.

    Parameters:
    -----------
    directory : str or Path
        Directory to search for h5ad files
    pattern : str, default='*.h5ad'
        File pattern to match

    Returns:
    --------
    h5ad_files : list
        List of Path objects for h5ad files
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    h5ad_files = sorted(directory.glob(pattern))

    if len(h5ad_files) == 0:
        raise FileNotFoundError(f"No h5ad files found in {directory}")

    print(f"Found {len(h5ad_files)} h5ad files in {directory}")
    for i, file in enumerate(h5ad_files[:10]):  # Show first 10
        print(f"  {i+1}. {file.name}")
    if len(h5ad_files) > 10:
        print(f"  ... and {len(h5ad_files) - 10} more files")

    return h5ad_files


def is_tile_processed(output_dir, tile_name):
    """
    Check if a tile has been fully processed by verifying all output files exist.

    Parameters:
    -----------
    output_dir : str or Path
        Directory where tile results should be saved
    tile_name : str
        Name of the tile (used as prefix for output files)

    Returns:
    --------
    is_complete : bool
        True if all expected output files exist, False otherwise
    """
    output_dir = Path(output_dir)

    # Check for all expected output files with tile name prefix
    expected_files = [
        f'{tile_name}_spatial_distribution.png',
        f'{tile_name}_neighborhood_enrichment.png',
        f'{tile_name}_significant_interactions.csv',
        f'{tile_name}_zscore.npy',         # Intermediate file for aggregation
        f'{tile_name}_metadata.json'       # Intermediate metadata
    ]

    for filename in expected_files:
        if not (output_dir / filename).exists():
            return False

    return True


def process_single_tile(
    adata_path,
    output_dir,
    radius=50,
    n_perms=1000,
    cluster_key='cell_type',
    save_adata=False,
    n_neighbors=20,
    skip_cooccurrence=True,
    max_cells_for_cooccurrence=50000,
    min_cells_per_type=2
):
    """
    Process a single tile using the standard spatial analysis pipeline.

    Parameters:
    -----------
    adata_path : str or Path
        Path to h5ad file
    output_dir : str or Path
        Directory to save results
    radius : float, default=50
        Radius for spatial graph
    n_perms : int, default=1000
        Number of permutations
    cluster_key : str, default='cell_type'
        Key for cell type labels
    save_adata : bool, default=False
        Whether to save processed AnnData
    skip_cooccurrence : bool, default=True
        Whether to skip co-occurrence analysis
    max_cells_for_cooccurrence : int, default=50000
        Max cells for co-occurrence
    min_cells_per_type : int, default=2
        Minimum cells per cell type required (needed for variance computation)

    Returns:
    --------
    results : dict
        Dictionary containing analysis results
    """
    adata_path = Path(adata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get tile name from file path (without .h5ad extension)
    tile_name = adata_path.stem

    # Load data
    try:
        adata = sc.read_h5ad(adata_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load h5ad file: {e}")

    # Apply cell type colors
    load_and_apply_cell_type_colors(adata, celltype_key=cluster_key)
    
    # Validate cell type distribution
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"Column '{cluster_key}' not found in adata.obs")
    
    cell_type_counts = adata.obs[cluster_key].value_counts()
    total_cell_types = len(cell_type_counts)
    
    # Check for cell types with insufficient cells (could cause division by zero)
    insufficient_types = cell_type_counts[cell_type_counts < min_cells_per_type]
    if len(insufficient_types) > 0:
        print(f"  ⚠ Warning: {len(insufficient_types)} cell type(s) have < {min_cells_per_type} cells:")
        for ct, count in insufficient_types.items():
            print(f"    - {ct}: {count} cells")
    
    # CRITICAL: Remove categories with 0 cells to avoid division by zero in analysis
    # This fixes the issue where empty categories cause errors in centrality computation
    if isinstance(adata.obs[cluster_key].dtype, pd.CategoricalDtype):
        # Get valid categories (those with at least 1 cell)
        valid_categories = cell_type_counts[cell_type_counts > 0].index.tolist()
        
        if len(valid_categories) < total_cell_types:
            removed_count = total_cell_types - len(valid_categories)
            print(f"  ⚠ Removing {removed_count} empty cell type category/categories from analysis")
            
            # Re-categorize to only include categories with cells
            adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
            adata.obs[cluster_key] = adata.obs[cluster_key].cat.set_categories(valid_categories)
            # Remove any NaN values (if any were created)
            if adata.obs[cluster_key].isna().any():
                print(f"  ⚠ Warning: Found NaN values in {cluster_key}, removing them")
                adata = adata[~adata.obs[cluster_key].isna()].copy()
            
            # Update counts after filtering
            cell_type_counts = adata.obs[cluster_key].value_counts()
            total_cell_types = len(cell_type_counts)
    
    # Check minimum cell type count (after filtering)
    if total_cell_types < 2:
        raise ValueError(
            f"Tile has only {total_cell_types} cell type(s) after filtering. "
            f"Need at least 2 cell types for interaction analysis. "
            f"Cell types present: {list(cell_type_counts.index)}"
        )
    
    print(f"  - Cell types (after filtering): {total_cell_types} ({', '.join(cell_type_counts.index.astype(str).tolist())})")
    print(f"  - Cell counts per type: {dict(cell_type_counts)}")

    # Build spatial graph
    try:
        adata = build_spatial_graph(adata, method='knn', n_neighbors=n_neighbors)
    except Exception as e:
        raise RuntimeError(f"Failed to build spatial graph: {e}")

    # Neighborhood enrichment
    try:
        adata = neighborhood_enrichment_analysis(
            adata,
            cluster_key=cluster_key,
            n_perms=n_perms
        )
    except ZeroDivisionError as e:
        raise RuntimeError(
            f"Division by zero in neighborhood enrichment. "
            f"This often occurs when:\n"
            f"  1. A cell type has insufficient cells (<{min_cells_per_type}) or no neighbors\n"
            f"  2. All cells of a type are isolated (no spatial connectivity)\n"
            f"  3. Insufficient variance in cell type distribution\n"
            f"Cell type distribution: {dict(cell_type_counts)}\n"
            f"Original error: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed neighborhood enrichment analysis: {e}")

    # Centrality scores
    try:
        adata = compute_centrality_scores(adata, cluster_key=cluster_key)
    except ZeroDivisionError as e:
        raise RuntimeError(
            f"Division by zero in centrality computation. "
            f"This can occur when cell types lack sufficient connectivity.\n"
            f"Cell type distribution: {dict(cell_type_counts)}\n"
            f"Original error: {e}"
        )
    except Exception as e:
        # Centrality is less critical, log warning but continue
        print(f"  ⚠ Warning: Centrality computation failed: {e}")
        print(f"  Continuing without centrality scores...")

    # Visualizations with tile name prefix
    visualize_spatial_distribution(
        adata,
        cluster_key=cluster_key,
        save_path=output_dir / f'{tile_name}_spatial_distribution.png'
    )

    visualize_enrichment(
        adata,
        cluster_key=cluster_key,
        n_perms=n_perms,
        n_neighbors=n_neighbors,
        save_path=output_dir / f'{tile_name}_neighborhood_enrichment.png'
    )

    # Summarize interactions
    interactions_df = summarize_interactions(adata, cluster_key=cluster_key)
    interactions_df.to_csv(output_dir / f'{tile_name}_significant_interactions.csv', index=False)

    # Save intermediate results for file-based aggregation (STEP 1)
    save_intermediate_results(
        adata=adata,
        output_dir=output_dir,
        tile_name=tile_name,
        cluster_key=cluster_key
    )

    # Save processed data if requested
    if save_adata:
        output_adata_path = output_dir / f'{tile_name}_adata_with_spatial_analysis.h5ad'
        adata.write(output_adata_path)

    # Return minimal summary (don't keep full adata in memory)
    results = {
        'tile_name': tile_name,
        'n_cells': adata.n_obs,
        'n_interactions': len(interactions_df)
    }

    return results


def run_multiple_tiles_pipeline(
    tiles_directory,
    output_dir='cti_multiple_tiles',
    radius=50,
    n_perms=1000,
    cluster_key='cell_type',
    save_adata=False,
    n_neighbors=20,
    skip_cooccurrence=True,
    max_cells_for_cooccurrence=50000,
    file_pattern='*.h5ad'
):
    """
    Run spatial analysis pipeline on multiple tiled images.

    Parameters:
    -----------
    tiles_directory : str
        Directory containing h5ad files
    output_dir : str, default='cti_multiple_tiles'
        Directory to save results
    radius : float, default=50
        Radius for spatial graph
    n_perms : int, default=1000
        Number of permutations
    cluster_key : str, default='cell_type'
        Key for cell type labels
    save_adata : bool, default=False
        Whether to save processed AnnData objects
    skip_cooccurrence : bool, default=True
        Whether to skip co-occurrence analysis
    max_cells_for_cooccurrence : int, default=50000
        Max cells for co-occurrence
    file_pattern : str, default='*.h5ad'
        Pattern to match h5ad files

    Returns:
    --------
    results : dict
        Dictionary containing all results
    """
    print("=" * 70)
    print("MULTIPLE TILES SPATIAL ANALYSIS PIPELINE")
    print("=" * 70)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nOutput directory: {output_dir}")

    # Find all h5ad files
    print(f"\nSearching for h5ad files in: {tiles_directory}")
    h5ad_files = find_h5ad_files(tiles_directory, pattern=file_pattern)
    n_tiles = len(h5ad_files)

    print(f"\n" + "=" * 70)
    print(f"PROCESSING {n_tiles} TILES")
    print("=" * 70)

    # Process each tile (memory-efficient - don't store in RAM)
    successful_tiles = []
    failed_tiles = []
    skipped_tiles = []

    for i, h5ad_path in enumerate(tqdm(h5ad_files, desc="Processing tiles")):
        tile_name = h5ad_path.stem
        tile_output_dir = output_dir / tile_name

        print(f"\n[{i+1}/{n_tiles}] Processing: {tile_name}")

        # Check if tile is already processed
        if is_tile_processed(tile_output_dir, tile_name):
            print(f"  ⊙ Skipped: Already processed (found all output files)")
            skipped_tiles.append(tile_name)
            successful_tiles.append(tile_name)  # Count as successful for aggregation
            continue

        try:
            results = process_single_tile(
                adata_path=h5ad_path,
                output_dir=tile_output_dir,
                radius=radius,
                n_perms=n_perms,
                n_neighbors=n_neighbors,
                cluster_key=cluster_key,
                save_adata=save_adata,
                skip_cooccurrence=skip_cooccurrence,
                max_cells_for_cooccurrence=max_cells_for_cooccurrence
            )
            successful_tiles.append(tile_name)
            print(f"  ✓ Success: {results['n_cells']} cells, {results['n_interactions']} significant interactions")

        except (OSError, IOError) as e:
            # I/O errors (file system issues, network mounts, corrupted files)
            error_msg = str(e)
            if "Errno 5" in error_msg or "Input/output error" in error_msg:
                error_type = "I/O Error (file system/network issue)"
            else:
                error_type = f"I/O Error: {type(e).__name__}"
            print(f"  ✗ Failed [{error_type}]: {e}")
            failed_tiles.append((tile_name, f"[{error_type}] {str(e)}"))
            continue
        except ZeroDivisionError as e:
            # Division by zero (missing cell types, insufficient variance)
            error_msg = str(e)
            print(f"  ✗ Failed [Division by Zero]: {error_msg}")
            failed_tiles.append((tile_name, f"[Division by Zero] {error_msg}"))
            continue
        except RuntimeError as e:
            # RuntimeError might wrap division by zero or other issues
            error_msg = str(e)
            if "Division by zero" in error_msg or "division by zero" in error_msg.lower():
                print(f"  ✗ Failed [Division by Zero]: {error_msg}")
                failed_tiles.append((tile_name, f"[Division by Zero] {error_msg}"))
            else:
                print(f"  ✗ Failed [RuntimeError]: {error_msg}")
                failed_tiles.append((tile_name, f"[RuntimeError] {error_msg}"))
            continue
        except ValueError as e:
            # Validation errors (missing cell types, insufficient data)
            error_msg = str(e)
            print(f"  ✗ Failed [Validation Error]: {error_msg}")
            failed_tiles.append((tile_name, f"[Validation Error] {error_msg}"))
            continue
        except Exception as e:
            # Other errors - check if it's a wrapped division by zero error
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Check if RuntimeError contains division by zero message
            if "Division by zero" in error_msg or "division by zero" in error_msg.lower():
                error_type = "Division by Zero (wrapped)"
                print(f"  ✗ Failed [{error_type}]: {error_msg}")
                failed_tiles.append((tile_name, f"[Division by Zero] {error_msg}"))
            else:
                print(f"  ✗ Failed [{error_type}]: {error_msg}")
                failed_tiles.append((tile_name, f"[{error_type}] {error_msg}"))
            continue

    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"  - Total tiles: {n_tiles}")
    print(f"  - Successfully processed: {len(successful_tiles)}")
    print(f"  - Skipped (already processed): {len(skipped_tiles)}")
    print(f"  - Failed: {len(failed_tiles)}")

    if skipped_tiles:
        print(f"\nSkipped tiles (already processed):")
        for tile_name in skipped_tiles[:10]:  # Show first 10
            print(f"  - {tile_name}")
        if len(skipped_tiles) > 10:
            print(f"  ... and {len(skipped_tiles) - 10} more")

    if failed_tiles:
        # Categorize failures
        io_errors = [t for t in failed_tiles if "[I/O Error" in t[1]]
        division_errors = [t for t in failed_tiles if "[Division by Zero]" in t[1]]
        validation_errors = [t for t in failed_tiles if "[Validation Error]" in t[1]]
        other_errors = [t for t in failed_tiles if t not in io_errors + division_errors + validation_errors]
        
        print("\nFailed tiles:")
        print(f"  Total failures: {len(failed_tiles)}")
        if io_errors:
            print(f"  - I/O Errors (file system/network): {len(io_errors)}")
        if division_errors:
            print(f"  - Division by Zero (missing/insufficient cell types): {len(division_errors)}")
        if validation_errors:
            print(f"  - Validation Errors (insufficient data): {len(validation_errors)}")
        if other_errors:
            print(f"  - Other Errors: {len(other_errors)}")
        
        print("\nDetailed failures:")
        for tile_name, error in failed_tiles:
            # Truncate long error messages
            if len(error) > 200:
                error = error[:197] + "..."
            print(f"  - {tile_name}: {error}")

    if len(successful_tiles) == 0:
        raise RuntimeError("All tiles failed to process!")

    # STEP 1 COMPLETE: All tiles have been processed and results saved
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nProcessed tiles: {len(successful_tiles)}")
    print(f"  - Individual tile results saved in: {output_dir}/<tile_name>/")
    print(f"  - Each tile directory contains:")
    print(f"    * {cluster_key}_spatial_distribution.png")
    print(f"    * {cluster_key}_neighborhood_enrichment.png")
    print(f"    * {cluster_key}_significant_interactions.csv")
    print(f"    * {cluster_key}_zscore.npy (intermediate file for aggregation)")
    print(f"    * {cluster_key}_metadata.json (intermediate metadata)")

    if failed_tiles:
        print(f"\nFailed tiles: {len(failed_tiles)}")
        print("  Note: Failed tiles will not be included in aggregation")
    
    print("\n" + "=" * 70)
    print("NEXT STEP: Run aggregation")
    print("=" * 70)
    print(f"\nTo aggregate results, run:")
    print(f"  python cti_aggregate.py --input_dir {output_dir}")

    # Return processing results (without aggregation)
    results = {
        'successful_tiles': successful_tiles,
        'failed_tiles': failed_tiles,
        'skipped_tiles': skipped_tiles,
        'output_dir': str(output_dir),
        'parameters': {
            'n_tiles': n_tiles,
            'radius': radius,
            'n_perms': n_perms,
            'cluster_key': cluster_key,
            'n_neighbors': n_neighbors
        }
    }

    return results


# Example usage
if __name__ == "__main__":
    # Configuration
    tiles_directory = '/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred/h5ad'
    output_dir = 'cti_multiple_tiles'

    # Run pipeline on multiple tiles
    results = run_multiple_tiles_pipeline(
        tiles_directory=tiles_directory,
        output_dir=output_dir,
        radius=50,                      # Adjust based on your tissue/magnification
        n_perms=1000,                   # Number of permutations
        n_neighbors=20,
        cluster_key='cell_type',        # Adjust to your cell type column
        save_adata=False,               # Set to True to save processed h5ad files
        skip_cooccurrence=True,         # Skip co-occurrence for faster processing
        max_cells_for_cooccurrence=50000,
        file_pattern='*.h5ad'           # Pattern to match files
    )

    print("\n" + "=" * 70)
    print("BATCH PROCESSING GUIDE")
    print("=" * 70)
    print("""
Batch Processing Results:

1. INDIVIDUAL TILE RESULTS:
   - Each tile processed independently
   - Results saved in separate subdirectories
   - Contains spatial distribution, enrichment heatmaps, and interactions

2. INTERMEDIATE FILES:
   - zscore.npy files for each tile (used in aggregation)
   - metadata.json files (tile statistics)
   - Allows aggregation without re-processing

3. NEXT STEPS:
   - Run cti_aggregate.py to generate aggregated results
   - Aggregation will compute mean z-scores, variability, and consistency
   - Produces summary statistics across all tiles

NOTES:
- Tiles with insufficient data are skipped or failed
- Empty cell type categories are automatically filtered
- All results are saved for later aggregation
""")