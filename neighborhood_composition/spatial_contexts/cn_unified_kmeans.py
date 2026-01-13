"""
Unified Cellular Neighborhood Detection Across Multiple Tiles

This script loads multiple tiles at once and performs CN detection on the combined
dataset, ensuring all tiles share the same CN composition. This is crucial for
downstream spatial context analysis.

Key Features:
- Loads 10-100 tiles into a unified dataset
- Performs k-means clustering on all cells together (default k=6 CNs)
- Generates ONE unified CN composition heatmap based on all tiles
- Generates individual spatial CN maps for EACH tile
- Saves processed h5ad files with CN annotations for spatial context analysis

Author: Generated with Claude Code
Date: 2025-10-27
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import squidpy as sq
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import os
import glob
import argparse
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans
from typing import Optional, Tuple, List, Dict
import warnings
import time
import json
from scipy.sparse import block_diag, csr_matrix

# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION: Random State
# ============================================================================
# Change this value to use a different random seed for reproducibility
# Default: 0 (matches notebook behavior)
DEFAULT_RANDOM_STATE = 0    # was 220705
# ============================================================================


class UnifiedCellularNeighborhoodDetector:
    """
    Detects cellular neighborhoods across multiple tiles using a unified approach.
    All tiles share the same CN composition, enabling cross-tile comparisons.
    """

    def __init__(self, tiles_directory: str, output_dir: str = 'cn_unified_results'):
        """
        Initialize unified CN detector.

        Parameters:
        -----------
        tiles_directory : str
            Directory containing h5ad tile files
        output_dir : str
            Base directory for all outputs
        """
        self.tiles_directory = Path(tiles_directory)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['individual_tiles', 'processed_h5ad', 'unified_analysis']:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Data storage
        self.combined_adata = None
        self.tile_list = []
        self.cn_labels = None
        self.aggregated_neighbors = None
    
    def _log_progress(self, current: int, total: int, prefix: str = ""):
        """Helper method for consistent progress logging."""
        return f"  [{current}/{total}] {prefix}"
    
    def _get_spatial_coords(self, adata, coord_key: str = 'spatial'):
        """Get spatial coordinates with fallback options."""
        if coord_key in adata.obsm:
            return adata.obsm[coord_key]
        elif 'spatial' in adata.obsm:
            return adata.obsm['spatial']
        return None

    def discover_tiles(self, pattern: str = "*.h5ad", max_tiles: Optional[int] = None) -> List[Path]:
        """Discover h5ad files in the tiles directory."""
        print(f"Discovering tiles in: {self.tiles_directory}")
        
        tile_files = sorted(self.tiles_directory.glob(pattern))[:max_tiles]
        
        if not tile_files:
            print(f"Warning: No {pattern} files found in {self.tiles_directory}")
            return []
        
        limit_msg = f" (limited to {max_tiles})" if max_tiles else ""
        print(f"Found {len(tile_files)} tile files{limit_msg}")
        
        return tile_files

    def load_and_combine_tiles(
        self,
        tile_files: List[Path],
        celltype_key: str = 'cell_type',
        coord_offset: bool = True
    ) -> ad.AnnData:
        """
        Load multiple tiles and combine them into a single AnnData object.

        Parameters:
        -----------
        tile_files : List[Path]
            List of paths to h5ad files
        celltype_key : str
            Key in adata.obs containing cell type labels
        coord_offset : bool
            Whether to offset spatial coordinates to avoid overlap between tiles.
            Note: This is only for visualization purposes. Neighbor detection uses
            original coordinates per tile to prevent cross-tile neighbors.

        Returns:
        --------
        combined_adata : AnnData
            Combined AnnData object with all tiles
        """
        print(f"\nLoading and combining {len(tile_files)} tiles...")
        
        adata_list = []
        coord_offset_x = 0
        coord_offset_y = 0
        
        for i, tile_path in enumerate(tile_files, 1):
            tile_name = tile_path.stem
            print(self._log_progress(i, len(tile_files), f"Loading: {tile_name}"))
            
            try:
                adata = sc.read_h5ad(tile_path)
                
                # Add tile identifier
                adata.obs['tile_name'] = tile_name
                adata.obs['tile_id'] = i - 1  # 0-based tile ID
                
                # Auto-detect cell type column
                if celltype_key not in adata.obs.columns:
                    alternatives = ['celltype', 'cell_type', 'CellType', 'Cell_Type']
                    celltype_key = next((alt for alt in alternatives if alt in adata.obs.columns), None)
                    
                    if not celltype_key:
                        print(f"    Warning: No cell type column found, skipping tile")
                        continue
                
                # Ensure cell types are categorical
                if not pd.api.types.is_categorical_dtype(adata.obs[celltype_key]):
                    adata.obs[celltype_key] = pd.Categorical(adata.obs[celltype_key])
                
                # Offset spatial coordinates if requested (for visualization only)
                # Neighbor detection uses original coordinates per tile
                if coord_offset and 'spatial' in adata.obsm:
                    # Store original coordinates before offset
                    adata.obsm['spatial_original'] = adata.obsm['spatial'].copy()
                    
                    # Apply offset for visualization
                    coords = adata.obsm['spatial'].copy()
                    coords[:, 0] += coord_offset_x
                    coords[:, 1] += coord_offset_y
                    adata.obsm['spatial'] = coords
                    
                    # Update offset for next tile (arrange tiles horizontally)
                    # Calculate tile dimensions to ensure proper spacing
                    tile_width = coords[:, 0].max() - coords[:, 0].min()
                    coord_offset_x = coords[:, 0].max() + max(500, tile_width * 0.1)  # 10% gap or 500px minimum
                    # Y-axis stays at 0 since we're arranging horizontally
                
                adata_list.append(adata)
                self.tile_list.append(tile_name)
                print(f"    ✓ Loaded {adata.n_obs} cells, {adata.n_vars} genes")
                
            except Exception as e:
                print(f"    ✗ Error loading {tile_path}: {str(e)}")
                continue
        
        if not adata_list:
            raise ValueError("No valid tiles could be loaded")
        
        print("\nCombining tiles into single dataset...")
        combined_adata = ad.concat(adata_list, join='outer', index_unique='-')
        
        if 'spatial' not in combined_adata.obsm:
            print("  Warning: No spatial coordinates found in combined data")
        
        print(f"✓ Combined dataset: {combined_adata.n_obs} cells, {combined_adata.n_vars} genes")
        print(f"  Tiles: {combined_adata.obs['tile_name'].nunique()}")
        print(f"  Cell types: {combined_adata.obs[celltype_key].nunique()}")
        
        self.combined_adata = combined_adata
        return combined_adata

    def build_knn_graph(
        self,
        k: int = 20,
        coord_key: str = 'spatial',
        key_added: str = 'spatial_connectivities_knn'
    ):
        """
        Build k-nearest neighbor graph separately for each tile to prevent cross-tile neighbors.
        
        This ensures that cells from different tiles (e.g., margin vs center vs adjacent_tissue)
        cannot be neighbors, even if they are spatially close in the combined coordinate space.
        """
        print(f"\nBuilding {k}-NN graph per tile (no cross-tile neighbors)...")

        # Get unique tiles
        unique_tiles = self.combined_adata.obs['tile_name'].unique()
        tile_connectivities = []
        tile_sizes = []
        
        # Build KNN graph for each tile separately
        for tile_idx, tile_name in enumerate(unique_tiles, 1):
            print(self._log_progress(tile_idx, len(unique_tiles), f"Building graph for {tile_name}"))
            
            # Extract tile data
            tile_mask = self.combined_adata.obs['tile_name'] == tile_name
            tile_adata = self.combined_adata[tile_mask].copy()
            
            # Prefer original coordinates (before offset) if available, otherwise use coord_key
            # This ensures we use actual spatial coordinates within each tile
            tile_coord_key = coord_key
            if 'spatial_original' in tile_adata.obsm:
                tile_coord_key = 'spatial_original'
                # Temporarily set as 'spatial' for squidpy compatibility
                tile_adata.obsm['spatial'] = tile_adata.obsm['spatial_original']
            
            # Get spatial coordinates for this tile
            coords = self._get_spatial_coords(tile_adata, 'spatial')
            if coords is None:
                print(f"    Warning: No spatial coordinates found for {tile_name}, skipping...")
                # Create empty connectivity matrix for this tile
                n_cells = tile_adata.n_obs
                tile_connectivities.append(csr_matrix((n_cells, n_cells)))
                tile_sizes.append(n_cells)
                continue
            
            # Build KNN graph for this tile only (using original coordinates)
            sq.gr.spatial_neighbors(
                tile_adata,
                spatial_key='spatial',  # Use 'spatial' key (set above from original if available)
                coord_type='generic',
                n_neighs=k,
                radius=None
            )
            
            # Get connectivity matrix
            if 'spatial_connectivities' in tile_adata.obsp:
                tile_conn = tile_adata.obsp['spatial_connectivities']
                tile_connectivities.append(tile_conn)
                tile_sizes.append(tile_adata.n_obs)
                avg_neighbors = tile_conn.sum(axis=1).mean()
                print(f"    ✓ {tile_adata.n_obs:,} cells, avg {avg_neighbors:.2f} neighbors")
            else:
                print(f"    Warning: Failed to build graph for {tile_name}")
                n_cells = tile_adata.n_obs
                tile_connectivities.append(csr_matrix((n_cells, n_cells)))
                tile_sizes.append(n_cells)
        
        # Combine connectivity matrices into block diagonal matrix
        # This ensures no cross-tile connections
        print(f"\n  Combining {len(tile_connectivities)} tile graphs into block diagonal matrix...")
        combined_connectivity = block_diag(tile_connectivities, format='csr')
        
        # Verify the combined matrix has correct shape
        expected_size = sum(tile_sizes)
        if combined_connectivity.shape != (expected_size, expected_size):
            raise ValueError(
                f"Connectivity matrix shape mismatch: "
                f"expected ({expected_size}, {expected_size}), "
                f"got {combined_connectivity.shape}"
            )
        
        # Store in combined adata
        self.combined_adata.obsp[key_added] = combined_connectivity
        
        # Also store as 'spatial_connectivities' for compatibility
        self.combined_adata.obsp['spatial_connectivities'] = combined_connectivity
        
        connectivity = self.combined_adata.obsp[key_added]
        avg_neighbors = connectivity.sum(axis=1).mean()
        print(f"  ✓ Combined connectivity matrix: {connectivity.shape}")
        print(f"  ✓ Average neighbors per cell: {avg_neighbors:.2f}")
        print(f"  ✓ No cross-tile neighbors (block diagonal structure)")
        
        return self

    def aggregate_neighbors(
        self,
        celltype_key: str = 'cell_type',
        connectivity_key: str = 'spatial_connectivities_knn',
        output_key: str = 'aggregated_neighbors'
    ):
        """For each cell, compute the fraction of each cell phenotype in its neighborhood."""
        print(f"\nAggregating neighbors by {celltype_key}...")

        cell_types = self.combined_adata.obs[celltype_key].values
        unique_types = self.combined_adata.obs[celltype_key].cat.categories
        connectivity = self.combined_adata.obsp[connectivity_key]
        n_cells = self.combined_adata.n_obs
        n_types = len(unique_types)
        aggregated = np.zeros((n_cells, n_types))

        print(f"  Processing {n_cells} cells...")
        
        for i in range(n_cells):
            if i % 10000 == 0 and i > 0:
                print(f"    Processed {i:,}/{n_cells:,} cells ({100*i/n_cells:.1f}%)")
            
            neighbors_mask = connectivity[i].toarray().flatten() > 0
            if neighbors_mask.sum() > 0:
                neighbor_types = cell_types[neighbors_mask]
                for j, ct in enumerate(unique_types):
                    aggregated[i, j] = (neighbor_types == ct).sum() / neighbors_mask.sum()

        self.aggregated_neighbors = pd.DataFrame(
            aggregated, columns=unique_types, index=self.combined_adata.obs_names
        )
        self.combined_adata.obsm[output_key] = aggregated
        print(f"  ✓ Aggregated neighbor fractions shape: {aggregated.shape}")
        return self

    def detect_cellular_neighborhoods(
        self,
        n_clusters: int = 7,
        random_state: int = None,
        aggregated_key: str = 'aggregated_neighbors',
        output_key: str = 'cn_celltype'
    ):
        """Cluster cells based on their neighborhood composition using MiniBatchKMeans (matching notebook)."""
        print(f"\nDetecting {n_clusters} unified cellular neighborhoods using MiniBatchKMeans...")
        
        # Use default random_state if not provided
        if random_state is None:
            random_state = DEFAULT_RANDOM_STATE

        aggregated = self.combined_adata.obsm[aggregated_key]
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
        cn_labels = kmeans.fit_predict(aggregated)

        self.cn_labels = cn_labels + 1  # 1-based indexing
        self.combined_adata.obs[output_key] = pd.Categorical(self.cn_labels)

        # Print CN sizes
        cn_counts = pd.Series(self.cn_labels).value_counts().sort_index()
        print(f"\n  ✓ Unified CN sizes (across all tiles):")
        for cn, count in cn_counts.items():
            print(f"    CN {cn}: {count:,} cells ({100 * count / len(self.cn_labels):.1f}%)")
        
        print(f"\n  CN distribution per tile:")
        for tile_name in self.combined_adata.obs['tile_name'].unique():
            tile_mask = self.combined_adata.obs['tile_name'] == tile_name
            tile_cns = self.combined_adata.obs[output_key][tile_mask]
            print(f"    {tile_name}: {tile_cns.value_counts().sort_index().to_dict()}")

        return self

    def compute_unified_cn_composition(
        self,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute unified cell phenotype fractions in each CN across ALL tiles."""
        print("\nComputing unified CN composition across all tiles...")

        composition = pd.crosstab(
            self.combined_adata.obs[cn_key],
            self.combined_adata.obs[celltype_key],
            normalize='index'
        )
        composition_zscore = composition.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        print(f"  ✓ Composition matrix shape: {composition.shape}")
        return composition, composition_zscore

    def visualize_unified_cn_composition(
        self,
        composition_zscore: pd.DataFrame,
        k: int,
        n_clusters: int,
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = 'coolwarm',
        vmin: float = -2,
        vmax: float = 2,
        save_path: Optional[str] = None,
        show_values: bool = True
    ):
        """Visualize unified CN composition as heatmap across ALL tiles."""
        print("\nVisualizing unified CN composition heatmap...")

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            composition_zscore,
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': 'Z-score'},
            linewidths=0.5,
            linecolor='white',
            ax=ax,
            annot=show_values,
            fmt='.2f' if show_values else '',
            annot_kws={'size': 12}
        )

        ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cellular Neighborhood', fontsize=12, fontweight='bold')
        
        n_tiles = self.combined_adata.obs['tile_name'].nunique()
        n_cells = self.combined_adata.n_obs
        title = (f'Unified Cell Type Composition by Cellular Neighborhood\n'
                f'(k={k}, n_clusters={n_clusters}, {n_tiles} tiles, {n_cells:,} cells)\n'
                f'Z-score scaled by column')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved unified heatmap to: {save_path}")

        return fig

    def visualize_individual_tile_cns(
        self,
        cn_key: str = 'cn_celltype',
        coord_key: str = 'spatial_original',
        point_size: float = 10.0,
        palette: str = 'Set2',
        k: Optional[int] = None,
        n_clusters: Optional[int] = None
    ):
        """Visualize cellular neighborhoods spatially for each tile."""
        print(f"\nGenerating individual spatial CN maps for each tile...")

        # Get CN labels and colors
        n_cns = len(self.combined_adata.obs[cn_key].cat.categories)
        colors_palette = sns.color_palette(palette, n_cns)

        # Process each tile
        for tile_idx, tile_name in enumerate(self.tile_list, 1):
            print(self._log_progress(tile_idx, len(self.tile_list), f"Plotting {tile_name}"))
            
            # Get cells from this tile
            tile_mask = self.combined_adata.obs['tile_name'] == tile_name
            tile_adata = self.combined_adata[tile_mask]
            
            # Get spatial coordinates
            coords = self._get_spatial_coords(tile_adata, coord_key)
            if coords is None:
                print(f"    Warning: No spatial coordinates found for {tile_name}, skipping...")
                continue
            
            cn_labels = tile_adata.obs[cn_key].values

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10))

            # Plot each CN
            for cn_id in np.unique(cn_labels):
                cn_mask = cn_labels == cn_id
                cn_idx = int(cn_id) - 1  # Convert to 0-based for color indexing

                ax.scatter(
                    coords[cn_mask, 0],
                    coords[cn_mask, 1],
                    c=[colors_palette[cn_idx]],
                    s=point_size,
                    alpha=0.7,
                    label=f'CN {cn_id}'
                )

            ax.set_xlabel('X coordinate (pixels)', fontsize=12)
            ax.set_ylabel('Y coordinate (pixels)', fontsize=12)
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add title with parameters
            title = f'Cellular Neighborhoods: {tile_name}'
            if k is not None and n_clusters is not None:
                title += f'\n(k={k}, n_clusters={n_clusters}, {tile_adata.n_obs:,} cells)'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

            # Add legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

            plt.tight_layout()

            # Save figure with tile-specific naming
            save_path = self.output_dir / 'individual_tiles' / f'spatial_cns_{tile_name}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"    ✓ Saved to: {save_path}")

        print(f"  ✓ Generated {len(self.tile_list)} spatial CN maps")

    def calculate_neighborhood_frequency(
        self,
        cn_key: str = 'cn_celltype',
        group_by_tile: bool = False
    ) -> pd.DataFrame:
        """
        Calculate the frequency of each cellular neighborhood.
        
        Parameters:
        -----------
        cn_key : str
            Key in adata.obs containing CN labels
        group_by_tile : bool
            If True, calculate frequency per tile. If False, calculate overall frequency.
            
        Returns:
        --------
        frequency_df : DataFrame
            DataFrame with CN frequencies (counts and percentages)
        """
        print(f"\nCalculating neighborhood frequency...")
        
        if group_by_tile:
            # Frequency per tile
            frequency_df = pd.crosstab(
                self.combined_adata.obs['tile_name'],
                self.combined_adata.obs[cn_key],
                normalize='index'  # Percentages per tile
            )
            print(f"  ✓ Calculated CN frequency per tile")
        else:
            # Overall frequency
            cn_counts = self.combined_adata.obs[cn_key].value_counts().sort_index()
            total_cells = len(self.combined_adata.obs)
            cn_percentages = (cn_counts / total_cells * 100).round(2)
            
            frequency_df = pd.DataFrame({
                'Count': cn_counts,
                'Percentage': cn_percentages
            })
            frequency_df.index.name = 'Cellular_Neighborhood'
            frequency_df = frequency_df.reset_index()
            print(f"  ✓ Calculated overall CN frequency")
            print(f"    Total cells: {total_cells:,}")
        
        return frequency_df

    def visualize_neighborhood_frequency(
        self,
        cn_key: str = 'cn_celltype',
        group_by_tile: bool = False,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
        color_palette: str = 'Set2'
    ):
        """
        Generate a graph showing neighborhood frequency.
        
        Parameters:
        -----------
        cn_key : str
            Key in adata.obs containing CN labels
        group_by_tile : bool
            If True, show frequency per tile. If False, show overall frequency.
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save figure
        color_palette : str
            Color palette name for the plot (default: 'Set2' to match individual tile maps)
        """
        print(f"\nGenerating neighborhood frequency graph...")
        
        frequency_df = self.calculate_neighborhood_frequency(cn_key, group_by_tile)
        
        # Get CN colors matching individual tile maps (Set2 palette)
        n_cns = len(self.combined_adata.obs[cn_key].cat.categories)
        colors_palette = sns.color_palette(color_palette, n_cns)
        
        if group_by_tile:
            # Stacked bar chart showing frequency per tile
            fig, ax = plt.subplots(figsize=figsize)
            
            # Ensure columns are sorted by CN ID and create color mapping
            # pd.crosstab returns columns that match the CN values (could be int, categorical, etc.)
            # Convert to list and sort as integers
            cn_ids = sorted([int(col) for col in frequency_df.columns])
            color_map = {cn_id: colors_palette[int(cn_id) - 1] for cn_id in cn_ids}
            
            # Reorder columns to match sorted order - use actual column names from DataFrame
            # Handle both integer and categorical column types
            column_list = list(frequency_df.columns)
            # Create mapping to get actual column names (in case they're categorical)
            col_mapping = {int(col): col for col in column_list}
            sorted_cols = [col_mapping[cn_id] for cn_id in cn_ids]
            frequency_df_sorted = frequency_df[sorted_cols]
            colors_sorted = [color_map[cn_id] for cn_id in cn_ids]
            
            frequency_df_sorted.plot(kind='bar', stacked=True, ax=ax, 
                                     color=colors_sorted, width=0.8)
            
            ax.set_xlabel('Tile', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency (Proportion)', fontsize=12, fontweight='bold')
            ax.set_title('Cellular Neighborhood Frequency by Tile', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend(title='Cellular Neighborhood', bbox_to_anchor=(1.05, 1), 
                     loc='upper left', fontsize=9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
        else:
            # SINGLE bar chart showing CN Frequency (Count) only, with percentage annotations in middle of bars
            # IMPORTANT: Only one bar chart is created - no separate percentage chart
            plt.close('all')  # Close any existing figures to avoid confusion
            fig, ax = plt.subplots(1, 1, figsize=figsize)  # Explicitly create single subplot
            
            # Sort by CN ID to ensure consistent color mapping
            frequency_df_sorted = frequency_df.sort_values('Cellular_Neighborhood')
            cn_ids = [int(cn_id) for cn_id in frequency_df_sorted['Cellular_Neighborhood']]
            colors_for_bars = [colors_palette[int(cn_id) - 1] for cn_id in cn_ids]
            
            # Create bars with count as height (y-axis = cell count) - ONLY COUNT, NOT PERCENTAGE
            bars = ax.bar(frequency_df_sorted['Cellular_Neighborhood'].astype(str), 
                         frequency_df_sorted['Count'], 
                         color=colors_for_bars)
            
            ax.set_xlabel('Cellular Neighborhood', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cell Count', fontsize=12, fontweight='bold')
            ax.set_title('CN Frequency (Count)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add count labels above bars and percentage annotations in the middle of each bar
            # Note: Percentage is only shown as annotation text INSIDE the count bars, not as a separate chart
            # Use black text with white outline for high contrast visibility
            text_outline = [path_effects.withStroke(linewidth=3, foreground='white')]
            max_count = max(frequency_df_sorted['Count'])
            for bar, count, pct in zip(bars, 
                                      frequency_df_sorted['Count'], 
                                      frequency_df_sorted['Percentage']):
                height = bar.get_height()
                # Count label above bar - black text with white outline
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count):,}',
                       ha='center', va='bottom', fontsize=14, fontweight='bold',
                       color='black', path_effects=text_outline)
                # Percentage annotation in the middle of bar - black text with white outline
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'{pct:.1f}%',
                       ha='center', va='center', fontsize=14, 
                       color='black', fontweight='bold', path_effects=text_outline)
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved frequency graph to: {save_path}")
        else:
            save_path = self.output_dir / 'unified_analysis' / 'neighborhood_frequency.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved frequency graph to: {save_path}")
        
        # Save frequency data to CSV
        csv_path = self.output_dir / 'unified_analysis' / 'neighborhood_frequency.csv'
        frequency_df.to_csv(csv_path, index=group_by_tile)
        print(f"  ✓ Saved frequency data to: {csv_path}")
        
        return fig

    def save_processed_data(self, cn_key: str = 'cn_celltype'):
        """Save processed h5ad files with CN annotations for each tile."""
        print(f"\nSaving processed h5ad files for spatial context analysis...")

        for tile_idx, tile_name in enumerate(self.tile_list, 1):
            print(self._log_progress(tile_idx, len(self.tile_list), f"Saving {tile_name}"))
            
            # Extract tile data
            tile_mask = self.combined_adata.obs['tile_name'] == tile_name
            tile_adata = self.combined_adata[tile_mask].copy()
            
            # Restore original spatial coordinates if available
            if 'spatial_original' in tile_adata.obsm:
                tile_adata.obsm['spatial'] = tile_adata.obsm['spatial_original']
                del tile_adata.obsm['spatial_original']
            
            # Save processed h5ad file
            output_path = self.output_dir / 'processed_h5ad' / f'{tile_name}_adata_cns.h5ad'
            tile_adata.write(output_path)
            print(f"    ✓ Saved to: {output_path}")

        print(f"  ✓ Saved {len(self.tile_list)} processed h5ad files")

    def save_summary_statistics(
        self,
        k: int,
        n_clusters: int,
        celltype_key: str,
        composition: pd.DataFrame,
        random_state: int = None
    ):
        """Save summary statistics for the unified CN analysis."""
        print("\nSaving summary statistics...")
        
        # Use default random_state if not provided
        if random_state is None:
            random_state = DEFAULT_RANDOM_STATE

        summary = {
            'analysis_type': 'Unified Cellular Neighborhoods',
            'n_tiles': len(self.tile_list),
            'tile_names': self.tile_list,
            'total_cells': int(self.combined_adata.n_obs),
            'total_genes': int(self.combined_adata.n_vars),
            'parameters': {
                'k_neighbors': k,
                'n_clusters': n_clusters,
                'random_state': random_state,
                'celltype_key': celltype_key
            },
            'cn_distribution': self.combined_adata.obs['cn_celltype'].value_counts().to_dict(),
            'cell_type_distribution': self.combined_adata.obs[celltype_key].value_counts().to_dict(),
            'cn_composition': composition.to_dict()
        }

        # Convert numpy types to native Python types
        def convert_to_native(obj):
            converters = {
                np.integer: int,
                np.floating: float,
                np.ndarray: lambda x: x.tolist()
            }
            for dtype, converter in converters.items():
                if isinstance(obj, dtype):
                    return converter(obj)
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        summary = convert_to_native(summary)

        # Save summary
        summary_path = self.output_dir / 'unified_analysis' / 'unified_cn_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved summary to: {summary_path}")

        # Save composition matrices
        comp_path = self.output_dir / 'unified_analysis' / 'unified_cn_composition.csv'
        composition.to_csv(comp_path)
        print(f"  ✓ Saved composition to: {comp_path}")

    def run_full_pipeline(
        self,
        tile_files: List[Path],
        k: int = 20,
        n_clusters: int = 7,
        celltype_key: str = 'cell_type',
        random_state: int = None,
        coord_offset: bool = True
    ):
        """Run the complete unified CN detection pipeline."""
        # Use default random_state if not provided
        if random_state is None:
            random_state = DEFAULT_RANDOM_STATE
            
        banner = "=" * 80
        print(f"{banner}\nUNIFIED CELLULAR NEIGHBORHOOD DETECTION PIPELINE\n{banner}")
        print(f"Processing {len(tile_files)} tiles with unified CN detection")
        print(f"Parameters: k={k}, n_clusters={n_clusters}, random_state={random_state}\n{banner}")

        start_time = time.time()

        # Step 1: Load and combine tiles
        self.load_and_combine_tiles(tile_files, celltype_key, coord_offset)

        # Step 2: Build KNN graph
        self.build_knn_graph(k=k)

        # Step 3: Aggregate neighbors
        self.aggregate_neighbors(celltype_key=celltype_key)

        # Step 4: Detect CNs
        self.detect_cellular_neighborhoods(n_clusters=n_clusters, random_state=random_state)

        # Step 5: Compute composition
        composition, composition_zscore = self.compute_unified_cn_composition(celltype_key=celltype_key)

        # Step 6: Visualize unified heatmap
        print(f"\n{banner}\nGENERATING VISUALIZATIONS\n{banner}")
        
        heatmap_path = self.output_dir / 'unified_analysis' / 'unified_cn_composition_heatmap.png'
        heatmap_fig = self.visualize_unified_cn_composition(
            composition_zscore,
            k=k,
            n_clusters=n_clusters,
            save_path=str(heatmap_path),
            show_values=True
        )
        plt.close(heatmap_fig)

        # Step 7: Visualize individual tile maps
        self.visualize_individual_tile_cns(k=k, n_clusters=n_clusters)

        # Step 8: Visualize neighborhood frequency distributions
        # Overall frequency (count-based bar chart with percentage annotations)
        overall_freq_path = self.output_dir / 'unified_analysis' / 'neighborhood_frequency_overall.png'
        overall_freq_fig = self.visualize_neighborhood_frequency(
            cn_key='cn_celltype',
            group_by_tile=False,
            figsize=(10, 6),
            save_path=str(overall_freq_path)
        )
        plt.close(overall_freq_fig)
        
        # Per-tile frequency (stacked bar chart)
        per_tile_freq_path = self.output_dir / 'unified_analysis' / 'neighborhood_frequency_per_tile.png'
        per_tile_freq_fig = self.visualize_neighborhood_frequency(
            cn_key='cn_celltype',
            group_by_tile=True,
            figsize=(14, 8),
            save_path=str(per_tile_freq_path)
        )
        plt.close(per_tile_freq_fig)

        # Step 9: Save processed data
        self.save_processed_data()

        # Step 10: Save summary statistics
        self.save_summary_statistics(k, n_clusters, celltype_key, composition, random_state)

        total_time = time.time() - start_time

        print(f"\n{banner}\nPIPELINE COMPLETE!\n{banner}")
        print(f"Total processing time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}/")
        print(f"  - Unified heatmap: {self.output_dir}/unified_analysis/")
        print(f"  - Frequency visualizations: {self.output_dir}/unified_analysis/")
        print(f"    * Overall frequency: neighborhood_frequency_overall.png")
        print(f"    * Per-tile frequency: neighborhood_frequency_per_tile.png")
        print(f"  - Individual tile maps: {self.output_dir}/individual_tiles/")
        print(f"  - Processed h5ad files: {self.output_dir}/processed_h5ad/")
        print("\nProcessed h5ad files are ready for spatial context analysis!")

        return self


def main():
    """Main function to run unified CN detection."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Unified Cellular Neighborhood Detection Across Multiple Tiles'
    )
    parser.add_argument(
        '--tiles_dir', '-t',
        default='/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/Batch_105/pred/h5ad',
        # adjacent_tissue, center, margin
        # /mnt/c/ProgramData/github_repo/image_analysis_scripts/neighborhood_composition/spatial_contexts/selected_h5ad_tiles/processed_h5ad
        # /mnt/c/ProgramData/github_repo/image_analysis_scripts/neighborhood_composition/spatial_contexts/selected_h5ad_tiles
        # default='/mnt/j/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad', # for 122 tiles
        help='Directory containing h5ad tile files'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='cn_unified_results',
        help='Output directory for results (spatial CN figures saved to cn_unified_results/individual_tiles/)'
    )
    parser.add_argument(
        '--k', type=int, default=20,
        help='Number of nearest neighbors (default: 20)'
    )
    parser.add_argument(
        '--n_clusters', '-n', type=int, default=8,  # default=7
        help='Number of cellular neighborhoods (default: 7)'
    )
    parser.add_argument(
        '--celltype_key', '-c',
        default='cell_type',
        help='Column name for cell types (default: cell_type)'
    )
    parser.add_argument(
        '--max_tiles', '-m', type=int, default=None,
        help='Maximum number of tiles to process (for testing)'
    )
    parser.add_argument(
        '--pattern', '-p',
        default='*.h5ad',
        help='File pattern to match (default: *.h5ad)'
    )
    parser.add_argument(
        '--no_offset', action='store_true',
        help='Disable spatial coordinate offsetting between tiles'
    )
    parser.add_argument(
        '--random_state', '-r', type=int, default=None,
        help=f'Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE})'
    )

    args = parser.parse_args()

    # Use default random_state if not provided
    random_state = args.random_state if args.random_state is not None else DEFAULT_RANDOM_STATE
    
    banner = "=" * 80
    print(f"{banner}\nUNIFIED CELLULAR NEIGHBORHOOD DETECTION\n{banner}")
    print(f"Tiles directory: {args.tiles_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parameters: k={args.k}, n_clusters={args.n_clusters}, random_state={random_state}")
    print(f"Cell type key: {args.celltype_key}")
    if args.max_tiles:
        print(f"Max tiles: {args.max_tiles} (testing mode)")
    print(banner)

    # Initialize detector
    detector = UnifiedCellularNeighborhoodDetector(
        tiles_directory=args.tiles_dir,
        output_dir=args.output_dir
    )

    # Discover tiles
    tile_files = detector.discover_tiles(pattern=args.pattern, max_tiles=args.max_tiles)

    if not tile_files:
        print("No tiles found! Exiting...")
        return

    # Run full pipeline
    detector.run_full_pipeline(
        tile_files=tile_files,
        k=args.k,
        n_clusters=args.n_clusters,
        celltype_key=args.celltype_key,
        random_state=random_state,
        coord_offset=not args.no_offset
    )

    print(f"\nUnified CN detection completed successfully!")
    print(f"Check the results in: {args.output_dir}/")


if __name__ == '__main__':
    main()