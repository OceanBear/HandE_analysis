"""
Batch Processing Script for Cellular Neighborhood Detection on Multiple Tiles

This script processes multiple tiled images using the CellularNeighborhoodDetector
from cn_kmeans_tiled.py. It handles batch processing, saves intermediate data,
and saves processed h5ad files for future aggregation.

Author: Generated with Claude Code
Date: 2025-10-24
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
import time
from typing import List, Dict, Optional
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import the cellular neighborhood detector
from cn_kmeans_tiled import CellularNeighborhoodDetector

warnings.filterwarnings('ignore')


class BatchCellularNeighborhoodProcessor:
    """
    Batch processor for cellular neighborhood detection across multiple tiles.
    """
    
    def __init__(self, tiles_directory: str, output_base_dir: str = 'cn_batch_results'):
        """
        Initialize batch processor.
        
        Parameters:
        -----------
        tiles_directory : str
            Directory containing h5ad tile files
        output_base_dir : str
            Base directory for all outputs
        """
        self.tiles_directory = Path(tiles_directory)
        self.output_base_dir = Path(output_base_dir)
        self.processed_tiles = []
        self.failed_tiles = []
        self.summary_data = []
        
        # Create output directories
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        (self.output_base_dir / 'individual_tiles').mkdir(exist_ok=True)
        (self.output_base_dir / 'intermediate_data').mkdir(exist_ok=True)
        (self.output_base_dir / 'processed_h5ad').mkdir(exist_ok=True)
        
    def discover_tiles(self, pattern: str = "*.h5ad") -> List[Path]:
        """
        Discover all h5ad files in the tiles directory.
        
        Parameters:
        -----------
        pattern : str
            File pattern to match (default: "*.h5ad")
            
        Returns:
        --------
        tile_files : List[Path]
            List of discovered tile files
        """
        print(f"Discovering tiles in: {self.tiles_directory}")
        
        tile_files = list(self.tiles_directory.glob(pattern))
        
        if not tile_files:
            print(f"Warning: No {pattern} files found in {self.tiles_directory}")
            return []
            
        print(f"Found {len(tile_files)} tile files")
        return sorted(tile_files)
    
    def is_tile_already_processed(self, tile_name: str) -> bool:
        """
        Check if a tile has already been processed by looking for output files.
        
        Parameters:
        -----------
        tile_name : str
            Name of the tile to check
            
        Returns:
        --------
        bool
            True if tile is already processed, False otherwise
        """
        # Check for processed h5ad file
        processed_h5ad_path = self.output_base_dir / 'processed_h5ad' / f'{tile_name}_adata_cns.h5ad'
        if processed_h5ad_path.exists():
            return True
            
        # Check for individual tile results
        tile_output_dir = self.output_base_dir / 'individual_tiles' / tile_name
        if tile_output_dir.exists():
            # Check for key output files
            required_files = ['spatial_cns.png', 'cn_composition_heatmap.png', 'cn_composition.csv']
            if all((tile_output_dir / file).exists() for file in required_files):
                return True
                
        return False

    def process_single_tile(self, 
                           tile_path: Path, 
                           k: int = 20,
                           n_clusters: int = 6,
                           celltype_key: str = 'cell_type',
                           img_id_key: str = 'tile_name',
                           random_state: int = 220705,
                           save_adata: bool = True,
                           skip_processed: bool = True) -> Dict:
        """
        Process a single tile for cellular neighborhood detection.
        
        Parameters:
        -----------
        tile_path : Path
            Path to the h5ad tile file
        k : int
            Number of nearest neighbors
        n_clusters : int
            Number of CNs to detect
        celltype_key : str
            Key in adata.obs containing cell type labels
        img_id_key : str
            Key in adata.obs containing image identifiers
        random_state : int
            Random seed
        save_adata : bool
            Whether to save the processed AnnData object
        skip_processed : bool
            Whether to skip already processed tiles
            
        Returns:
        --------
        result : Dict
            Dictionary containing processing results and metadata
        """
        tile_name = tile_path.stem
        print(f"\n{'='*60}")
        print(f"PROCESSING TILE: {tile_name}")
        print(f"{'='*60}")
        
        # Check if tile is already processed
        if skip_processed and self.is_tile_already_processed(tile_name):
            print(f"✓ Tile {tile_name} already processed, skipping...")
            return {
                'tile_name': tile_name,
                'tile_path': str(tile_path),
                'status': 'skipped',
                'message': 'Already processed'
            }
        
        start_time = time.time()
        
        try:
            # Load data
            print(f"Loading data from: {tile_path}")
            adata = sc.read_h5ad(tile_path)
            print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
            
            # Auto-detect column names if they don't exist
            if celltype_key not in adata.obs.columns:
                # Try common alternatives
                alternatives = ['celltype', 'cell_type', 'CellType', 'Cell_Type', 'cellType']
                for alt in alternatives:
                    if alt in adata.obs.columns:
                        celltype_key = alt
                        print(f"  - Auto-detected cell type column: {celltype_key}")
                        break
                else:
                    print(f"  - Available columns: {list(adata.obs.columns)}")
                    raise ValueError(f"Cell type column '{celltype_key}' not found. Available columns: {list(adata.obs.columns)}")
            
            if img_id_key not in adata.obs.columns:
                # Try common alternatives
                alternatives = ['tile_name', 'tile_id', 'image_id', 'sample_id', 'Tile_Name']
                for alt in alternatives:
                    if alt in adata.obs.columns:
                        img_id_key = alt
                        print(f"  - Auto-detected image ID column: {img_id_key}")
                        break
                else:
                    print(f"  - Available columns: {list(adata.obs.columns)}")
                    raise ValueError(f"Image ID column '{img_id_key}' not found. Available columns: {list(adata.obs.columns)}")
            
            # Initialize detector
            detector = CellularNeighborhoodDetector(adata)
            
            # Create tile-specific output directory
            tile_output_dir = self.output_base_dir / 'individual_tiles' / tile_name
            tile_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run full pipeline
            detector.run_full_pipeline(
                k=k,
                n_clusters=n_clusters,
                celltype_key=celltype_key,
                img_id_key=img_id_key,
                output_dir=str(tile_output_dir),
                random_state=random_state,
                save_adata=save_adata
            )
            
            # Save intermediate data for future aggregation
            self._save_intermediate_data(detector, tile_name, tile_output_dir)
            
            # Save processed h5ad file
            if save_adata:
                processed_h5ad_path = self.output_base_dir / 'processed_h5ad' / f'{tile_name}_adata_cns.h5ad'
                adata.write(processed_h5ad_path)
                print(f"Saved processed h5ad to: {processed_h5ad_path}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Collect summary information
            result = {
                'tile_name': tile_name,
                'tile_path': str(tile_path),
                'n_cells': adata.n_obs,
                'n_genes': adata.n_vars,
                'n_clusters_found': len(adata.obs['cn_celltype'].cat.categories),
                'processing_time_seconds': processing_time,
                'output_dir': str(tile_output_dir),
                'processed_h5ad_path': str(processed_h5ad_path) if save_adata else None,
                'status': 'success'
            }
            
            # Add CN composition summary
            composition = detector.compute_cn_composition(
                cn_key='cn_celltype',
                celltype_key=celltype_key
            )
            result['cn_composition_summary'] = {
                'cn_sizes': adata.obs['cn_celltype'].value_counts().to_dict(),
                'cell_type_counts': adata.obs[celltype_key].value_counts().to_dict()
            }
            
            self.processed_tiles.append(tile_name)
            print(f"✓ Successfully processed {tile_name} in {processing_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing {tile_name}: {str(e)}"
            print(f"✗ {error_msg}")
            
            result = {
                'tile_name': tile_name,
                'tile_path': str(tile_path),
                'processing_time_seconds': processing_time,
                'status': 'failed',
                'error': str(e)
            }
            
            self.failed_tiles.append(tile_name)
            return result
    
    def _save_intermediate_data(self, detector, tile_name: str, output_dir: Path):
        """
        Save intermediate data for future aggregation.
        
        Parameters:
        -----------
        detector : CellularNeighborhoodDetector
            The detector instance with processed data
        tile_name : str
            Name of the tile
        output_dir : Path
            Output directory for the tile
        """
        intermediate_dir = self.output_base_dir / 'intermediate_data' / tile_name
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Save aggregated neighbors
        if hasattr(detector, 'aggregated_neighbors') and detector.aggregated_neighbors is not None:
            aggregated_path = intermediate_dir / 'aggregated_neighbors.csv'
            detector.aggregated_neighbors.to_csv(aggregated_path)
            print(f"  - Saved aggregated neighbors to: {aggregated_path}")
        
        # Save CN labels
        if hasattr(detector, 'cn_labels') and detector.cn_labels is not None:
            cn_labels_path = intermediate_dir / 'cn_labels.csv'
            pd.Series(detector.cn_labels, name='cn_labels').to_csv(cn_labels_path, index=True)
            print(f"  - Saved CN labels to: {cn_labels_path}")
        
        # Save spatial coordinates
        if 'spatial' in detector.adata.obsm:
            coords_path = intermediate_dir / 'spatial_coordinates.csv'
            pd.DataFrame(
                detector.adata.obsm['spatial'],
                columns=['x', 'y'],
                index=detector.adata.obs_names
            ).to_csv(coords_path)
            print(f"  - Saved spatial coordinates to: {coords_path}")
        
        # Save cell metadata
        metadata_path = intermediate_dir / 'cell_metadata.csv'
        detector.adata.obs.to_csv(metadata_path)
        print(f"  - Saved cell metadata to: {metadata_path}")
        
        # Save connectivity matrix if available
        if 'spatial_connectivities_knn' in detector.adata.obsp:
            connectivity_path = intermediate_dir / 'spatial_connectivities.npz'
            connectivity = detector.adata.obsp['spatial_connectivities_knn']
            np.savez_compressed(connectivity_path, data=connectivity.data, indices=connectivity.indices, indptr=connectivity.indptr, shape=connectivity.shape)
            print(f"  - Saved connectivity matrix to: {connectivity_path}")
    
    def process_all_tiles(self,
                         k: int = 20,
                         n_clusters: int = 6,
                         celltype_key: str = 'cell_type',
                         img_id_key: str = 'tile_name',
                         random_state: int = 220705,
                         save_adata: bool = True,
                         max_tiles: Optional[int] = None,
                         skip_processed: bool = True) -> List[Dict]:
        """
        Process all discovered tiles.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors
        n_clusters : int
            Number of CNs to detect
        celltype_key : str
            Key in adata.obs containing cell type labels
        img_id_key : str
            Key in adata.obs containing image identifiers
        random_state : int
            Random seed
        save_adata : bool
            Whether to save processed AnnData objects
        max_tiles : int, optional
            Maximum number of tiles to process (for testing)
        skip_processed : bool
            Whether to skip already processed tiles
            
        Returns:
        --------
        results : List[Dict]
            List of processing results for all tiles
        """
        # Discover tiles
        tile_files = self.discover_tiles()
        
        if not tile_files:
            print("No tiles found to process.")
            return []
        
        if max_tiles:
            tile_files = tile_files[:max_tiles]
            print(f"Processing first {max_tiles} tiles for testing...")
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING {len(tile_files)} TILES")
        print(f"{'='*80}")
        
        all_results = []
        start_time = time.time()
        
        for i, tile_path in enumerate(tile_files, 1):
            print(f"\nProcessing tile {i}/{len(tile_files)}: {tile_path.name}")
            
            result = self.process_single_tile(
                tile_path=tile_path,
                k=k,
                n_clusters=n_clusters,
                celltype_key=celltype_key,
                img_id_key=img_id_key,
                random_state=random_state,
                save_adata=save_adata,
                skip_processed=skip_processed
            )
            
            all_results.append(result)
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(tile_files) - i) * avg_time
            print(f"Progress: {i}/{len(tile_files)} tiles processed")
            print(f"Estimated time remaining: {remaining/60:.1f} minutes")
        
        # Save batch summary
        self._save_batch_summary(all_results)
        
        total_time = time.time() - start_time
        successful_count = len([r for r in all_results if r.get('status') == 'success'])
        failed_count = len([r for r in all_results if r.get('status') == 'failed'])
        skipped_count = len([r for r in all_results if r.get('status') == 'skipped'])
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total processing time: {total_time/60:.1f} minutes")
        print(f"Successfully processed: {successful_count} tiles")
        print(f"Failed tiles: {failed_count} tiles")
        print(f"Skipped tiles: {skipped_count} tiles")
        
        if failed_count > 0:
            failed_tiles = [r['tile_name'] for r in all_results if r.get('status') == 'failed']
            print(f"Failed tiles: {', '.join(failed_tiles[:10])}{'...' if len(failed_tiles) > 10 else ''}")
        
        return all_results
    
    def _save_batch_summary(self, results: List[Dict]):
        """
        Save batch processing summary.
        
        Parameters:
        -----------
        results : List[Dict]
            List of processing results
        """
        summary_path = self.output_base_dir / 'batch_processing_summary.csv'
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved batch processing summary to: {summary_path}")
        
        # Count different status types
        successful_tiles = [r for r in results if r.get('status') == 'success']
        failed_tiles = [r for r in results if r.get('status') == 'failed']
        skipped_tiles = [r for r in results if r.get('status') == 'skipped']
        
        # Save detailed summary
        detailed_summary = {
            'total_tiles': len(results),
            'successful_tiles': len(successful_tiles),
            'failed_tiles': len(failed_tiles),
            'skipped_tiles': len(skipped_tiles),
            'total_processing_time_minutes': sum(r.get('processing_time_seconds', 0) for r in results) / 60,
            'average_processing_time_per_tile': np.mean([r.get('processing_time_seconds', 0) for r in results if r.get('processing_time_seconds', 0) > 0]),
            'total_cells_processed': sum(r.get('n_cells', 0) for r in results if r.get('status') == 'success'),
            'processed_tiles': [r['tile_name'] for r in successful_tiles],
            'failed_tiles': [r['tile_name'] for r in failed_tiles],
            'skipped_tiles': [r['tile_name'] for r in skipped_tiles]
        }
        
        detailed_summary_path = self.output_base_dir / 'detailed_summary.json'
        import json
        with open(detailed_summary_path, 'w') as f:
            json.dump(detailed_summary, f, indent=2)
        print(f"Saved detailed summary to: {detailed_summary_path}")


def main():
    """
    Main function to run batch processing.
    """
    # Configuration
    tiles_directory = '/mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad'
    output_base_dir = 'cn_batch_results'
    
    # Processing parameters
    k = 20
    n_clusters = 6
    celltype_key = 'cell_type'
    img_id_key = 'tile_name'
    random_state = 220705
    save_adata = True
    max_tiles = None  # Set to a number for testing, None for all tiles
    
    print("=" * 80)
    print("BATCH CELLULAR NEIGHBORHOOD DETECTION")
    print("=" * 80)
    print(f"Tiles directory: {tiles_directory}")
    print(f"Output directory: {output_base_dir}")
    print(f"Parameters: k={k}, n_clusters={n_clusters}")
    print(f"Cell type key: {celltype_key}")
    print(f"Image ID key: {img_id_key}")
    print(f"Save processed h5ad: {save_adata}")
    if max_tiles:
        print(f"Processing first {max_tiles} tiles only (testing mode)")
    print("=" * 80)
    
    # Initialize processor
    processor = BatchCellularNeighborhoodProcessor(
        tiles_directory=tiles_directory,
        output_base_dir=output_base_dir
    )
    
    # Process all tiles
    results = processor.process_all_tiles(
        k=k,
        n_clusters=n_clusters,
        celltype_key=celltype_key,
        img_id_key=img_id_key,
        random_state=random_state,
        save_adata=save_adata,
        max_tiles=max_tiles
    )
    
    print(f"\nBatch processing completed!")
    print(f"Results saved in: {output_base_dir}/")
    print(f"Individual tile results: {output_base_dir}/individual_tiles/")
    print(f"Intermediate data: {output_base_dir}/intermediate_data/")
    print(f"Processed h5ad files: {output_base_dir}/processed_h5ad/")


if __name__ == '__main__':
    main()
