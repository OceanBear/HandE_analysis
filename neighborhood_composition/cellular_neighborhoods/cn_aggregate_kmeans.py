"""
Aggregation Script for Cellular Neighborhood Batch Results

This script aggregates all processed tiles from batch processing into a single
comprehensive dataset for further analysis. It combines spatial coordinates,
cell metadata, CN labels, and composition data from all tiles.

Author: Generated with Claude Code
Date: 2025-10-24
"""

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from pathlib import Path
import time
from typing import List, Dict, Optional, Tuple
import warnings
import json
from scipy.sparse import csr_matrix
from pathlib import Path
# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)
warnings.filterwarnings('ignore')

class CNResultsAggregator:
    """
    Aggregates cellular neighborhood results from multiple tiles into a single dataset.
    """
    
    def __init__(self, batch_results_dir: str, output_dir: str = 'aggregated'):
        """
        Initialize aggregator.
        
        Parameters:
        -----------
        batch_results_dir : str
            Directory containing batch processing results
        output_dir : str
            Directory to save aggregated results
        """
        self.batch_results_dir = Path(batch_results_dir)
        self.output_dir = self.batch_results_dir / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths to different data types
        self.intermediate_data_dir = self.batch_results_dir / 'intermediate_data'
        self.processed_h5ad_dir = self.batch_results_dir / 'processed_h5ad'
        self.individual_tiles_dir = self.batch_results_dir / 'individual_tiles'
        
        # Aggregated data storage
        self.aggregated_data = {}
        self.tile_metadata = {}
        
    def discover_processed_tiles(self) -> List[str]:
        """
        Discover all processed tiles from the batch results.
        
        Returns:
        --------
        tile_names : List[str]
            List of processed tile names
        """
        print("Discovering processed tiles...")
        
        # Check processed h5ad files
        h5ad_files = list(self.processed_h5ad_dir.glob('*_adata_cns.h5ad'))
        tile_names = [f.stem.replace('_adata_cns', '') for f in h5ad_files]
        
        if not tile_names:
            print(f"Warning: No processed h5ad files found in {self.processed_h5ad_dir}")
            return []
            
        print(f"Found {len(tile_names)} processed tiles")
        return sorted(tile_names)
    
    def load_tile_data(self, tile_name: str) -> Optional[Dict]:
        """
        Load all data for a single tile.
        
        Parameters:
        -----------
        tile_name : str
            Name of the tile to load
            
        Returns:
        --------
        tile_data : Dict
            Dictionary containing all tile data
        """
        print(f"Loading data for tile: {tile_name}")
        
        try:
            # Load processed h5ad file
            h5ad_path = self.processed_h5ad_dir / f'{tile_name}_adata_cns.h5ad'
            if not h5ad_path.exists():
                print(f"  - Warning: {h5ad_path} not found, skipping tile")
                return None
                
            adata = sc.read_h5ad(h5ad_path)
            print(f"  - Loaded {adata.n_obs} cells, {adata.n_vars} genes")
            
            # Load intermediate data
            intermediate_dir = self.intermediate_data_dir / tile_name
            
            tile_data = {
                'tile_name': tile_name,
                'adata': adata,
                'n_cells': adata.n_obs,
                'n_genes': adata.n_vars
            }
            
            # Load aggregated neighbors if available
            aggregated_path = intermediate_dir / 'aggregated_neighbors.csv'
            if aggregated_path.exists():
                aggregated_neighbors = pd.read_csv(aggregated_path, index_col=0)
                tile_data['aggregated_neighbors'] = aggregated_neighbors
                print(f"  - Loaded aggregated neighbors: {aggregated_neighbors.shape}")
            
            # Load CN labels if available
            cn_labels_path = intermediate_dir / 'cn_labels.csv'
            if cn_labels_path.exists():
                cn_labels = pd.read_csv(cn_labels_path, index_col=0)
                tile_data['cn_labels'] = cn_labels
                print(f"  - Loaded CN labels: {cn_labels.shape}")
            
            # Load spatial coordinates if available
            coords_path = intermediate_dir / 'spatial_coordinates.csv'
            if coords_path.exists():
                coords = pd.read_csv(coords_path, index_col=0)
                tile_data['spatial_coordinates'] = coords
                print(f"  - Loaded spatial coordinates: {coords.shape}")
            
            # Load connectivity matrix if available
            connectivity_path = intermediate_dir / 'spatial_connectivities.npz'
            if connectivity_path.exists():
                connectivity_data = np.load(connectivity_path)
                connectivity = csr_matrix(
                    (connectivity_data['data'], 
                     connectivity_data['indices'], 
                     connectivity_data['indptr']), 
                    shape=connectivity_data['shape']
                )
                tile_data['connectivity'] = connectivity
                print(f"  - Loaded connectivity matrix: {connectivity.shape}")
            
            return tile_data
            
        except Exception as e:
            print(f"  - Error loading tile {tile_name}: {str(e)}")
            return None
    
    def aggregate_spatial_data(self, tile_data_list: List[Dict]) -> pd.DataFrame:
        """
        Aggregate spatial coordinates from all tiles.
        
        Parameters:
        -----------
        tile_data_list : List[Dict]
            List of tile data dictionaries
            
        Returns:
        --------
        aggregated_coords : pd.DataFrame
            Aggregated spatial coordinates with tile information
        """
        print("Aggregating spatial coordinates...")
        
        all_coords = []
        
        for tile_data in tile_data_list:
            if 'spatial_coordinates' in tile_data:
                coords = tile_data['spatial_coordinates'].copy()
                coords['tile_name'] = tile_data['tile_name']
                all_coords.append(coords)
        
        if not all_coords:
            print("  - No spatial coordinates found")
            return pd.DataFrame()
        
        aggregated_coords = pd.concat(all_coords, ignore_index=False)
        print(f"  - Aggregated {len(aggregated_coords)} spatial coordinates from {len(all_coords)} tiles")
        
        return aggregated_coords
    
    def aggregate_cell_metadata(self, tile_data_list: List[Dict]) -> pd.DataFrame:
        """
        Aggregate cell metadata from all tiles.
        
        Parameters:
        -----------
        tile_data_list : List[Dict]
            List of tile data dictionaries
            
        Returns:
        --------
        aggregated_metadata : pd.DataFrame
            Aggregated cell metadata with tile information
        """
        print("Aggregating cell metadata...")
        
        all_metadata = []
        
        for tile_data in tile_data_list:
            adata = tile_data['adata']
            metadata = adata.obs.copy()
            metadata['tile_name'] = tile_data['tile_name']
            all_metadata.append(metadata)
        
        if not all_metadata:
            print("  - No cell metadata found")
            return pd.DataFrame()
        
        aggregated_metadata = pd.concat(all_metadata, ignore_index=False)
        print(f"  - Aggregated {len(aggregated_metadata)} cell records from {len(all_metadata)} tiles")
        
        return aggregated_metadata
    
    def aggregate_cn_labels(self, tile_data_list: List[Dict]) -> pd.DataFrame:
        """
        Aggregate CN labels from all tiles.
        
        Parameters:
        -----------
        tile_data_list : List[Dict]
            List of tile data dictionaries
            
        Returns:
        --------
        aggregated_cn_labels : pd.DataFrame
            Aggregated CN labels with tile information
        """
        print("Aggregating CN labels...")
        
        all_cn_labels = []
        
        for tile_data in tile_data_list:
            adata = tile_data['adata']
            if 'cn_celltype' in adata.obs.columns:
                cn_labels = pd.DataFrame({
                    'cn_celltype': adata.obs['cn_celltype'],
                    'tile_name': tile_data['tile_name']
                }, index=adata.obs_names)
                all_cn_labels.append(cn_labels)
        
        if not all_cn_labels:
            print("  - No CN labels found")
            return pd.DataFrame()
        
        aggregated_cn_labels = pd.concat(all_cn_labels, ignore_index=False)
        print(f"  - Aggregated {len(aggregated_cn_labels)} CN labels from {len(all_cn_labels)} tiles")
        
        return aggregated_cn_labels
    
    def aggregate_composition_data(self, tile_data_list: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Aggregate composition data from all tiles.
        
        Parameters:
        -----------
        tile_data_list : List[Dict]
            List of tile data dictionaries
            
        Returns:
        --------
        composition_data : Dict[str, pd.DataFrame]
            Dictionary containing aggregated composition data
        """
        print("Aggregating composition data...")
        
        all_compositions = []
        all_composition_zscores = []
        
        for tile_data in tile_data_list:
            tile_name = tile_data['tile_name']
            individual_dir = self.individual_tiles_dir / tile_name
            
            # Load composition data
            comp_path = individual_dir / 'cn_composition.csv'
            comp_zscore_path = individual_dir / 'cn_composition_zscore.csv'
            
            if comp_path.exists():
                comp_data = pd.read_csv(comp_path, index_col=0)
                comp_data['tile_name'] = tile_name
                all_compositions.append(comp_data)
            
            if comp_zscore_path.exists():
                comp_zscore_data = pd.read_csv(comp_zscore_path, index_col=0)
                comp_zscore_data['tile_name'] = tile_name
                all_composition_zscores.append(comp_zscore_data)
        
        composition_data = {}
        
        if all_compositions:
            aggregated_composition = pd.concat(all_compositions, ignore_index=False)
            composition_data['composition'] = aggregated_composition
            print(f"  - Aggregated {len(aggregated_composition)} composition records")
        
        if all_composition_zscores:
            aggregated_composition_zscore = pd.concat(all_composition_zscores, ignore_index=False)
            composition_data['composition_zscore'] = aggregated_composition_zscore
            print(f"  - Aggregated {len(aggregated_composition_zscore)} z-score composition records")
        
        return composition_data
    
    def create_summary_statistics(self, tile_data_list: List[Dict]) -> Dict:
        """
        Create summary statistics for all tiles.
        
        Parameters:
        -----------
        tile_data_list : List[Dict]
            List of tile data dictionaries
            
        Returns:
        --------
        summary_stats : Dict
            Dictionary containing summary statistics
        """
        print("Creating summary statistics...")
        
        total_cells = sum(tile_data['n_cells'] for tile_data in tile_data_list)
        total_genes = tile_data_list[0]['n_genes'] if tile_data_list else 0  # Assume same genes
        
        # CN statistics
        all_cn_labels = []
        for tile_data in tile_data_list:
            adata = tile_data['adata']
            if 'cn_celltype' in adata.obs.columns:
                all_cn_labels.extend(adata.obs['cn_celltype'].values)
        
        cn_stats = {}
        if all_cn_labels:
            cn_series = pd.Series(all_cn_labels)
            cn_stats = {
                'total_cns': len(cn_series.unique()),
                'cn_distribution': cn_series.value_counts().to_dict(),
                'most_common_cn': cn_series.mode().iloc[0] if not cn_series.empty else None
            }
        
        # Cell type statistics
        all_cell_types = []
        for tile_data in tile_data_list:
            adata = tile_data['adata']
            if 'cell_type' in adata.obs.columns:
                all_cell_types.extend(adata.obs['cell_type'].values)
            elif 'celltype' in adata.obs.columns:
                all_cell_types.extend(adata.obs['celltype'].values)
        
        cell_type_stats = {}
        if all_cell_types:
            cell_type_series = pd.Series(all_cell_types)
            cell_type_stats = {
                'total_cell_types': len(cell_type_series.unique()),
                'cell_type_distribution': cell_type_series.value_counts().to_dict(),
                'most_common_cell_type': cell_type_series.mode().iloc[0] if not cell_type_series.empty else None
            }
        
        summary_stats = {
            'total_tiles': len(tile_data_list),
            'total_cells': total_cells,
            'total_genes': total_genes,
            'cn_statistics': cn_stats,
            'cell_type_statistics': cell_type_stats,
            'tile_names': [tile_data['tile_name'] for tile_data in tile_data_list]
        }
        
        print(f"  - Total tiles: {summary_stats['total_tiles']}")
        print(f"  - Total cells: {summary_stats['total_cells']}")
        print(f"  - Total genes: {summary_stats['total_genes']}")
        if cn_stats:
            print(f"  - Total CNs: {cn_stats['total_cns']}")
        if cell_type_stats:
            print(f"  - Total cell types: {cell_type_stats['total_cell_types']}")
        
        return summary_stats
    
    def save_aggregated_data(self, 
                           spatial_coords: pd.DataFrame,
                           cell_metadata: pd.DataFrame,
                           cn_labels: pd.DataFrame,
                           composition_data: Dict[str, pd.DataFrame],
                           summary_stats: Dict):
        """
        Save all aggregated data to files.
        
        Parameters:
        -----------
        spatial_coords : pd.DataFrame
            Aggregated spatial coordinates
        cell_metadata : pd.DataFrame
            Aggregated cell metadata
        cn_labels : pd.DataFrame
            Aggregated CN labels
        composition_data : Dict[str, pd.DataFrame]
            Aggregated composition data
        summary_stats : Dict
            Summary statistics
        """
        print("Saving aggregated data...")
        
        # Save spatial coordinates
        if not spatial_coords.empty:
            spatial_path = self.output_dir / 'aggregated_spatial_coordinates.csv'
            spatial_coords.to_csv(spatial_path)
            print(f"  - Saved spatial coordinates to: {spatial_path}")
        
        # Save cell metadata
        if not cell_metadata.empty:
            metadata_path = self.output_dir / 'aggregated_cell_metadata.csv'
            cell_metadata.to_csv(metadata_path)
            print(f"  - Saved cell metadata to: {metadata_path}")
        
        # Save CN labels
        if not cn_labels.empty:
            cn_labels_path = self.output_dir / 'aggregated_cn_labels.csv'
            cn_labels.to_csv(cn_labels_path)
            print(f"  - Saved CN labels to: {cn_labels_path}")
        
        # Save composition data
        for data_type, data in composition_data.items():
            if not data.empty:
                comp_path = self.output_dir / f'aggregated_{data_type}.csv'
                data.to_csv(comp_path)
                print(f"  - Saved {data_type} to: {comp_path}")
        
        # Save summary statistics
        summary_path = self.output_dir / 'aggregated_summary_statistics.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        print(f"  - Saved summary statistics to: {summary_path}")
        
        # Create a combined dataset
        self._create_combined_dataset(cell_metadata, spatial_coords, cn_labels)
    
    def _create_combined_dataset(self, 
                               cell_metadata: pd.DataFrame,
                               spatial_coords: pd.DataFrame,
                               cn_labels: pd.DataFrame):
        """
        Create a combined dataset with all information.
        
        Parameters:
        -----------
        cell_metadata : pd.DataFrame
            Cell metadata
        spatial_coords : pd.DataFrame
            Spatial coordinates
        cn_labels : pd.DataFrame
            CN labels
        """
        print("Creating combined dataset...")
        
        if cell_metadata.empty:
            print("  - No cell metadata available for combined dataset")
            return
        
        # Start with cell metadata
        combined = cell_metadata.copy()
        
        # Add spatial coordinates if available
        if not spatial_coords.empty:
            # Merge spatial coordinates
            spatial_merge = spatial_coords.set_index(combined.index)
            combined = combined.join(spatial_merge[['x', 'y']], how='left')
            print(f"  - Added spatial coordinates to {combined['x'].notna().sum()} cells")
        
        # Add CN labels if available
        if not cn_labels.empty:
            # Check if cn_celltype already exists in combined data
            if 'cn_celltype' not in combined.columns:
                # Merge CN labels only if they don't already exist
                cn_merge = cn_labels.set_index(combined.index)
                combined = combined.join(cn_merge[['cn_celltype']], how='left')
                print(f"  - Added CN labels to {combined['cn_celltype'].notna().sum()} cells")
            else:
                print(f"  - CN labels already present in cell metadata")
        
        # Save combined dataset
        combined_path = self.output_dir / 'combined_dataset.csv'
        combined.to_csv(combined_path)
        print(f"  - Saved combined dataset to: {combined_path}")
    
    def aggregate_all_results(self) -> Dict:
        """
        Aggregate all results from batch processing.
        
        Returns:
        --------
        aggregation_results : Dict
            Dictionary containing all aggregation results
        """
        print("=" * 80)
        print("CELLULAR NEIGHBORHOOD RESULTS AGGREGATION")
        print("=" * 80)
        
        start_time = time.time()
        
        # Discover processed tiles
        tile_names = self.discover_processed_tiles()
        if not tile_names:
            print("No processed tiles found!")
            return {}
        
        # Load all tile data
        print(f"\nLoading data from {len(tile_names)} tiles...")
        tile_data_list = []
        for tile_name in tile_names:
            tile_data = self.load_tile_data(tile_name)
            if tile_data:
                tile_data_list.append(tile_data)
        
        if not tile_data_list:
            print("No valid tile data found!")
            return {}
        
        print(f"\nSuccessfully loaded {len(tile_data_list)} tiles")
        
        # Aggregate different data types
        spatial_coords = self.aggregate_spatial_data(tile_data_list)
        cell_metadata = self.aggregate_cell_metadata(tile_data_list)
        cn_labels = self.aggregate_cn_labels(tile_data_list)
        composition_data = self.aggregate_composition_data(tile_data_list)
        summary_stats = self.create_summary_statistics(tile_data_list)
        
        # Save all aggregated data
        self.save_aggregated_data(
            spatial_coords, cell_metadata, cn_labels, 
            composition_data, summary_stats
        )
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"AGGREGATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total aggregation time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}/")
        
        return {
            'spatial_coordinates': spatial_coords,
            'cell_metadata': cell_metadata,
            'cn_labels': cn_labels,
            'composition_data': composition_data,
            'summary_statistics': summary_stats
        }


def main():
    """
    Main function to run aggregation.
    """
    # Configuration
    batch_results_dir = 'cn_batch_results'
    output_dir = 'aggregated'
    
    print("=" * 80)
    print("CELLULAR NEIGHBORHOOD RESULTS AGGREGATION")
    print("=" * 80)
    print(f"Batch results directory: {batch_results_dir}")
    print(f"Output directory: {batch_results_dir}/{output_dir}")
    print("=" * 80)
    
    # Initialize aggregator
    aggregator = CNResultsAggregator(
        batch_results_dir=batch_results_dir,
        output_dir=output_dir
    )
    
    # Run aggregation
    results = aggregator.aggregate_all_results()
    
    if results:
        print(f"\nAggregation completed successfully!")
        print(f"Check the results in: {batch_results_dir}/{output_dir}/")
    else:
        print(f"\nAggregation failed - no data found!")


if __name__ == '__main__':
    main()
