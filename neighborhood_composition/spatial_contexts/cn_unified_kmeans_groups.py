"""
Group-based Cellular Neighborhood Analysis and Visualization

This script reads processed h5ad files from cn_unified_kmeans.py and generates:
1. Unified analysis visualizations (unified_analysis folder):
   - Unified CN composition heatmap
   - Overall and per-tile neighborhood frequency graphs
2. Individual tile spatial CN maps (individual_tiles folder)
3. Group-specific analysis (2mm_groups folder):
   - Group-based CN composition comparisons
   - Group-specific frequency visualizations
"""

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()


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
        # Original format: 1, 2, 3, 4, 5
        int_labels = [int(float(str(x).strip())) for x in labels]
        sorted_labels = sorted(set(int_labels), key=lambda x: x)
        n = len(sorted_labels)
        palette = sns.color_palette(color_palette, max(sorted_labels))
        colors = [palette[x - 1] for x in sorted_labels]
        return sorted_labels, colors
    else:
        # Sub-clustered format: CN1, CN2, CN3-1, CN3-2, CN4-1, CN4-2, CN5
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


class GroupCNAnalyzer:
    """Analyzes cellular neighborhoods by predefined groups."""
    
    def __init__(self, processed_h5ad_dir: str, categories_json: str, output_dir: str):
        """
        Initialize group CN analyzer.
        
        Parameters:
        -----------
        processed_h5ad_dir : str
            Directory containing processed h5ad files with CN annotations
        categories_json : str
            Path to JSON file with tile categorization (relative paths resolved from script directory)
        output_dir : str
            Output directory for group-specific results (relative paths resolved from script directory)
        """
        self.processed_h5ad_dir = Path(processed_h5ad_dir)
        
        # Resolve relative paths relative to script directory
        categories_json_path = Path(categories_json)
        if not categories_json_path.is_absolute():
            self.categories_json = (SCRIPT_DIR / categories_json_path).resolve()
        else:
            self.categories_json = categories_json_path
        
        output_dir_path = Path(output_dir)
        if not output_dir_path.is_absolute():
            base_output_dir = (SCRIPT_DIR / output_dir_path).resolve()
        else:
            base_output_dir = output_dir_path
        
        # Load categorization
        with open(self.categories_json, 'r') as f:
            self.categories = json.load(f)
        
        # Store base output directory (for unified_analysis and individual_tiles)
        self.base_output_dir = base_output_dir
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unified_analysis and individual_tiles directories at base level
        (self.base_output_dir / 'unified_analysis').mkdir(exist_ok=True)
        (self.base_output_dir / 'individual_tiles').mkdir(exist_ok=True)
        
        # Extract tile size from metadata and create subfolder for group analysis
        tile_size_mm = self.categories.get('metadata', {}).get('tile_size_mm2', 2.0)
        # Convert to string like "2mm" (assuming integer tile sizes)
        tile_size_folder = f"{int(tile_size_mm)}mm_groups"
        
        # Create output directory with tile size subfolder (for group-specific analysis)
        self.output_dir = base_output_dir / tile_size_folder
        
        # Create directory - handle edge cases robustly
        if self.output_dir.exists():
            if self.output_dir.is_file():
                # If it's a file, remove it and create directory
                self.output_dir.unlink()
                self.output_dir.mkdir(parents=True, exist_ok=True)
            # If it's already a directory, nothing to do (exist_ok=True handles this)
            elif not self.output_dir.is_dir():
                # If it exists but is neither file nor dir (e.g., broken symlink), try to remove and recreate
                try:
                    self.output_dir.rmdir()
                except:
                    pass
                self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Path doesn't exist, create it
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Loaded tile categories from: {self.categories_json}")
        print(f"  Groups: {list(self.categories.keys() - {'metadata'})}")
        print(f"  Base output directory: {self.base_output_dir}")
        print(f"  Group analysis output: {self.output_dir}")
        
    def load_group_data(
        self,
        group_name: str,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type'
    ) -> ad.AnnData:
        """
        Load and combine h5ad files for a specific group.
        
        Parameters:
        -----------
        group_name : str
            Group name (e.g., 'adjacent_tissue', 'center', 'margin')
        cn_key : str
            Key in adata.obs containing CN labels
        celltype_key : str
            Key in adata.obs containing cell type labels
            
        Returns:
        --------
        combined_adata : AnnData
            Combined AnnData for the group
        """
        if group_name not in self.categories:
            raise ValueError(f"Group '{group_name}' not found in categories")
        
        tile_names = self.categories[group_name]
        print(f"\nLoading {len(tile_names)} tiles for group: {group_name}")
        
        adata_list = []
        cell_type_categories = None  # Will store the categorical order from first tile
        
        for i, tile_name in enumerate(tile_names, 1):
            h5ad_file = self.processed_h5ad_dir / f'{tile_name}_adata_cns.h5ad'
            
            if not h5ad_file.exists():
                print(f"  [{i}/{len(tile_names)}] Warning: {h5ad_file.name} not found, skipping")
                continue
            
            adata = ad.read_h5ad(h5ad_file)
            
            # Ensure cn_key exists
            if cn_key not in adata.obs.columns:
                print(f"  [{i}/{len(tile_names)}] Warning: {cn_key} not in {tile_name}, skipping")
                continue
            
            # Preserve categorical order from first tile
            if pd.api.types.is_categorical_dtype(adata.obs[celltype_key]):
                if cell_type_categories is None:
                    # Store the categorical order from the first tile
                    cell_type_categories = adata.obs[celltype_key].cat.categories.tolist()
                else:
                    # Ensure all subsequent tiles use the same categorical order
                    adata.obs[celltype_key] = adata.obs[celltype_key].astype('category')
                    adata.obs[celltype_key] = adata.obs[celltype_key].cat.set_categories(
                        cell_type_categories, ordered=True
                    )
            elif cell_type_categories is None:
                # If first tile is not categorical, convert and store order
                adata.obs[celltype_key] = pd.Categorical(adata.obs[celltype_key])
                cell_type_categories = adata.obs[celltype_key].cat.categories.tolist()
            
            print(f"  [{i}/{len(tile_names)}] Loaded {tile_name}: {adata.n_obs} cells")
            adata_list.append(adata)
        
        if not adata_list:
            raise ValueError(f"No valid h5ad files found for group: {group_name}")
        
        # Combine
        combined_adata = ad.concat(adata_list, join='outer', index_unique='-')
        
        # Ensure the combined adata preserves the categorical order
        if cell_type_categories is not None:
            if pd.api.types.is_categorical_dtype(combined_adata.obs[celltype_key]):
                # Re-set categories to ensure order is preserved
                combined_adata.obs[celltype_key] = combined_adata.obs[celltype_key].cat.set_categories(
                    cell_type_categories, ordered=True
                )
        
        print(f"✓ Combined {len(adata_list)} tiles: {combined_adata.n_obs:,} cells")
        
        return combined_adata
    
    def compute_cn_composition(
        self,
        adata: ad.AnnData,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type'
    ):
        """Compute CN composition for a group."""
        composition = pd.crosstab(
            adata.obs[cn_key],
            adata.obs[celltype_key],
            normalize='index'
        )
        
        # Ensure columns follow the categorical order from the original h5ad files
        if pd.api.types.is_categorical_dtype(adata.obs[celltype_key]):
            cell_type_order = adata.obs[celltype_key].cat.categories.tolist()
            # Reorder columns to match categorical order
            existing_cols = [col for col in cell_type_order if col in composition.columns]
            # Add any columns that exist in composition but not in categorical (shouldn't happen, but safe)
            remaining_cols = [col for col in composition.columns if col not in existing_cols]
            final_cols = existing_cols + remaining_cols
            composition = composition[final_cols]
        
        composition_zscore = composition.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        return composition, composition_zscore
    
    def load_overall_composition(self, cn_key: str = 'cn_celltype') -> Optional[pd.DataFrame]:
        """
        Load overall CN composition from unified analysis results.
        For sub-clustered data (cn_key='cn_celltype_sub'), looks for unified_cn_composition_sub.csv.
        """
        unified_dir = self.processed_h5ad_dir.parent / 'unified_analysis'
        # Sub-clustered results use unified_cn_composition_sub.csv
        if cn_key == 'cn_celltype_sub':
            overall_comp_file = unified_dir / 'unified_cn_composition_sub.csv'
        else:
            overall_comp_file = unified_dir / 'unified_cn_composition.csv'

        if overall_comp_file.exists():
            try:
                overall_composition = pd.read_csv(overall_comp_file, index_col=0)
                overall_zscore = overall_composition.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
                return overall_zscore
            except (OSError, IOError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"  Warning: Error reading overall composition file: {e}")
                return None
        else:
            print(f"  Warning: Overall composition file not found at {overall_comp_file}")
            return None
    
    def load_all_processed_tiles(
        self,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type'
    ) -> ad.AnnData:
        """
        Load all processed h5ad files and combine them into a single AnnData object.
        
        Parameters:
        -----------
        cn_key : str
            Key in adata.obs containing CN labels
        celltype_key : str
            Key in adata.obs containing cell type labels
            
        Returns:
        --------
        combined_adata : AnnData
            Combined AnnData object with all tiles
        """
        print(f"\nLoading all processed h5ad files from: {self.processed_h5ad_dir}")
        
        h5ad_files = sorted(self.processed_h5ad_dir.glob('*_adata_cns.h5ad'))
        
        if not h5ad_files:
            raise ValueError(f"No processed h5ad files found in {self.processed_h5ad_dir}")
        
        print(f"Found {len(h5ad_files)} processed h5ad files")
        
        adata_list = []
        cell_type_categories = None  # Will store the categorical order from first tile
        
        for i, h5ad_file in enumerate(h5ad_files, 1):
            tile_name = h5ad_file.stem.replace('_adata_cns', '')
            print(f"  [{i}/{len(h5ad_files)}] Loading {tile_name}...")
            
            try:
                adata = ad.read_h5ad(h5ad_file)
                
                # Ensure required keys exist
                if cn_key not in adata.obs.columns:
                    print(f"    Warning: {cn_key} not found, skipping")
                    continue
                
                if celltype_key not in adata.obs.columns:
                    print(f"    Warning: {celltype_key} not found, skipping")
                    continue
                
                # Preserve categorical order from first tile
                if pd.api.types.is_categorical_dtype(adata.obs[celltype_key]):
                    if cell_type_categories is None:
                        # Store the categorical order from the first tile
                        cell_type_categories = adata.obs[celltype_key].cat.categories.tolist()
                        print(f"    ✓ Preserving cell type order from first tile: {cell_type_categories}")
                    else:
                        # Ensure all subsequent tiles use the same categorical order
                        adata.obs[celltype_key] = adata.obs[celltype_key].astype('category')
                        adata.obs[celltype_key] = adata.obs[celltype_key].cat.set_categories(
                            cell_type_categories, ordered=True
                        )
                elif cell_type_categories is None:
                    # If first tile is not categorical, convert and store order
                    adata.obs[celltype_key] = pd.Categorical(adata.obs[celltype_key])
                    cell_type_categories = adata.obs[celltype_key].cat.categories.tolist()
                    print(f"    ✓ Cell types converted to categorical, order: {cell_type_categories}")
                
                # Add tile identifier if not present
                if 'tile_name' not in adata.obs.columns:
                    adata.obs['tile_name'] = tile_name
                
                adata_list.append(adata)
                print(f"    ✓ Loaded {adata.n_obs} cells")
                
            except Exception as e:
                print(f"    ✗ Error loading {h5ad_file}: {str(e)}")
                continue
        
        if not adata_list:
            raise ValueError("No valid h5ad files could be loaded")
        
        print(f"\nCombining {len(adata_list)} tiles...")
        combined_adata = ad.concat(adata_list, join='outer', index_unique='-')
        
        # Ensure the combined adata preserves the categorical order
        if cell_type_categories is not None:
            if pd.api.types.is_categorical_dtype(combined_adata.obs[celltype_key]):
                # Re-set categories to ensure order is preserved
                combined_adata.obs[celltype_key] = combined_adata.obs[celltype_key].cat.set_categories(
                    cell_type_categories, ordered=True
                )
        
        print(f"✓ Combined dataset: {combined_adata.n_obs:,} cells")
        print(f"  Tiles: {combined_adata.obs['tile_name'].nunique()}")
        print(f"  Cell types: {combined_adata.obs[celltype_key].nunique()}")
        
        return combined_adata
    
    def compute_unified_cn_composition(
        self,
        adata: ad.AnnData,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute unified cell phenotype fractions in each CN across ALL tiles."""
        print("\nComputing unified CN composition across all tiles...")

        composition = pd.crosstab(
            adata.obs[cn_key],
            adata.obs[celltype_key],
            normalize='index'
        )
        
        # Ensure columns follow the categorical order from the original h5ad files
        if pd.api.types.is_categorical_dtype(adata.obs[celltype_key]):
            cell_type_order = adata.obs[celltype_key].cat.categories.tolist()
            # Reorder columns to match categorical order
            existing_cols = [col for col in cell_type_order if col in composition.columns]
            # Add any columns that exist in composition but not in categorical (shouldn't happen, but safe)
            remaining_cols = [col for col in composition.columns if col not in existing_cols]
            final_cols = existing_cols + remaining_cols
            composition = composition[final_cols]
        
        composition_zscore = composition.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        print(f"  ✓ Composition matrix shape: {composition.shape}")
        return composition, composition_zscore
    
    def _get_spatial_coords(self, adata, coord_key: str = 'spatial'):
        """Get spatial coordinates with fallback options."""
        if coord_key in adata.obsm:
            return adata.obsm[coord_key]
        elif 'spatial' in adata.obsm:
            return adata.obsm['spatial']
        return None
    
    def _log_progress(self, current: int, total: int, prefix: str = ""):
        """Helper method for consistent progress logging."""
        return f"  [{current}/{total}] {prefix}"
    
    def visualize_unified_cn_composition(
        self,
        adata: ad.AnnData,
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
        """Visualize unified CN composition as heatmap across ALL tiles.
        
        Note: composition_zscore should preserve the cell type order from the original h5ad files.
        """
        print("\nVisualizing unified CN composition heatmap...")

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap (order should be preserved from original h5ad files)
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
        
        n_tiles = adata.obs['tile_name'].nunique()
        n_cells = adata.n_obs
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

        plt.close(fig)
        return fig
    
    def visualize_individual_tile_cns(
        self,
        adata: ad.AnnData,
        cn_key: str = 'cn_celltype',
        coord_key: str = 'spatial',
        point_size: float = 10.0,
        palette: str = 'tab20',
        k: Optional[int] = None,
        n_clusters: Optional[int] = None
    ):
        """Visualize cellular neighborhoods spatially for each tile."""
        print(f"\nGenerating individual spatial CN maps for each tile...")

        # Get unique tiles
        unique_tiles = adata.obs['tile_name'].unique()
        
        # Get CN labels and colors
        n_cns = len(adata.obs[cn_key].cat.categories)
        if n_cns <= 20:
            colors_palette = sns.color_palette(palette, n_cns)
        else:
            colors_palette = sns.color_palette('husl', n_cns)

        # Process each tile
        for tile_idx, tile_name in enumerate(unique_tiles, 1):
            print(self._log_progress(tile_idx, len(unique_tiles), f"Plotting {tile_name}"))
            
            # Get cells from this tile
            tile_mask = adata.obs['tile_name'] == tile_name
            tile_adata = adata[tile_mask]
            
            # Get spatial coordinates
            coords = self._get_spatial_coords(tile_adata, coord_key)
            if coords is None:
                print(f"    Warning: No spatial coordinates found for {tile_name}, skipping...")
                continue
            
            cn_labels = tile_adata.obs[cn_key].values

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10))

            # Plot each CN (support both integer and string labels e.g. CN3-1)
            unique_cns = np.unique(cn_labels)
            sorted_labels, colors_list = _sort_cn_labels_and_colors(unique_cns, palette)
            label_to_color = dict(zip(sorted_labels, colors_list))

            for cn_id in unique_cns:
                cn_mask = cn_labels == cn_id
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

            # Add title with parameters
            title = f'Cellular Neighborhoods: {tile_name}'
            if k is not None and n_clusters is not None:
                title += f'\n(k={k}, n_clusters={n_clusters}, {tile_adata.n_obs:,} cells)'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

            # Add legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

            plt.tight_layout()

            # Save figure with tile-specific naming
            save_path = self.base_output_dir / 'individual_tiles' / f'spatial_cns_{tile_name}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"    ✓ Saved to: {save_path}")

        print(f"  ✓ Generated {len(unique_tiles)} spatial CN maps")
    
    def calculate_unified_neighborhood_frequency(
        self,
        adata: ad.AnnData,
        cn_key: str = 'cn_celltype',
        group_by_tile: bool = False
    ) -> pd.DataFrame:
        """
        Calculate the frequency of each cellular neighborhood.
        
        Parameters:
        -----------
        adata : AnnData
            Combined AnnData object
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
                adata.obs['tile_name'],
                adata.obs[cn_key],
                normalize='index'  # Percentages per tile
            )
            print(f"  ✓ Calculated CN frequency per tile")
        else:
            # Overall frequency
            cn_counts = adata.obs[cn_key].value_counts().sort_index()
            total_cells = len(adata.obs)
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
    
    def visualize_unified_neighborhood_frequency(
        self,
        adata: ad.AnnData,
        cn_key: str = 'cn_celltype',
        group_by_tile: bool = False,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
        color_palette: str = 'tab20',
        show_tile_names: bool = False
    ):
        """
        Generate a graph showing neighborhood frequency.
        
        Parameters:
        -----------
        adata : AnnData
            Combined AnnData object
        cn_key : str
            Key in adata.obs containing CN labels
        group_by_tile : bool
            If True, show frequency per tile. If False, show overall frequency.
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save figure
        color_palette : str
            Color palette name for the plot (default: 'tab20' to match individual tile maps)
        show_tile_names : bool, default=False
            Whether to display tile names on x-axis when group_by_tile=True (default: False to hide names)
        """
        print(f"\nGenerating neighborhood frequency graph...")
        
        frequency_df = self.calculate_unified_neighborhood_frequency(adata, cn_key, group_by_tile)
        
        # Get CN colors matching individual tile maps (tab20 palette)
        n_cns = len(adata.obs[cn_key].cat.categories)
        if n_cns <= 20:
            colors_palette = sns.color_palette(color_palette, n_cns)
        else:
            colors_palette = sns.color_palette('husl', n_cns)
        
        if group_by_tile:
            # Stacked bar chart showing frequency per tile
            fig, ax = plt.subplots(figsize=figsize)
            
            # Ensure columns are sorted (support both integer and string CN labels)
            sorted_labels, colors_sorted = _sort_cn_labels_and_colors(frequency_df.columns, color_palette)
            sorted_cols = [c for c in sorted_labels if c in frequency_df.columns]
            frequency_df_sorted = frequency_df[sorted_cols]
            
            frequency_df_sorted.plot(kind='bar', stacked=True, ax=ax, 
                                     color=colors_sorted, width=0.8)
            
            ax.set_xlabel('Tile', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency (Proportion)', fontsize=12, fontweight='bold')
            ax.set_title('Cellular Neighborhood Frequency by Tile', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend(title='Cellular Neighborhood', bbox_to_anchor=(1.05, 1), 
                     loc='upper left', fontsize=9)
            if show_tile_names:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                # Hide tile names by default
                ax.set_xticklabels([])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
        else:
            # SINGLE bar chart showing CN Frequency (Count) only, with percentage annotations
            plt.close('all')
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
            # Sort by CN ID (support both integer and string CN labels)
            cn_labels_col = frequency_df['Cellular_Neighborhood']
            sorted_labels, colors_list = _sort_cn_labels_and_colors(cn_labels_col, color_palette)
            order = {lab: i for i, lab in enumerate(sorted_labels)}
            frequency_df_sorted = frequency_df.copy()
            frequency_df_sorted['_ord'] = frequency_df_sorted['Cellular_Neighborhood'].map(order)
            frequency_df_sorted = frequency_df_sorted.sort_values('_ord').drop(columns=['_ord'])
            colors_for_bars = [colors_list[order[lab]] for lab in frequency_df_sorted['Cellular_Neighborhood']]
            
            # Create bars with count as height
            bars = ax.bar(frequency_df_sorted['Cellular_Neighborhood'].astype(str), 
                         frequency_df_sorted['Count'], 
                         color=colors_for_bars)
            
            ax.set_xlabel('Cellular Neighborhood', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cell Count', fontsize=12, fontweight='bold')
            ax.set_title('CN Frequency (Count)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add count labels above bars and percentage annotations in the middle
            text_outline = [path_effects.withStroke(linewidth=3, foreground='white')]
            for bar, count, pct in zip(bars, 
                                      frequency_df_sorted['Count'], 
                                      frequency_df_sorted['Percentage']):
                height = bar.get_height()
                # Count label above bar
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count):,}',
                       ha='center', va='bottom', fontsize=14, fontweight='bold',
                       color='black', path_effects=text_outline)
                # Percentage annotation in the middle of bar
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
        
        # Save frequency data to CSV (only save overall frequency CSV, per-tile is in the per-tile figure)
        if not group_by_tile:
            csv_path = self.base_output_dir / 'unified_analysis' / 'neighborhood_frequency.csv'
            frequency_df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved frequency data to: {csv_path}")
        
        plt.close(fig)
        return fig
    
    def visualize_cn_composition_heatmap(
        self,
        composition_zscore: pd.DataFrame,
        group_name: str,
        n_cells: int,
        overall_zscore: Optional[pd.DataFrame] = None,
        figsize=(12, 8),
        save_path: Optional[str] = None
    ):
        """
        Visualize CN composition heatmap for a group, showing difference from overall.
        
        Parameters:
        -----------
        composition_zscore : pd.DataFrame
            Group's composition Z-scores
        group_name : str
            Name of the group
        n_cells : int
            Number of cells in the group
        overall_zscore : pd.DataFrame, optional
            Overall composition Z-scores for comparison
        """
        # Get the correct cell type order from overall_zscore if available (preserves order from original h5ad files)
        # Otherwise, use the order from composition_zscore
        if overall_zscore is not None:
            cell_type_order = overall_zscore.columns.tolist()
        else:
            cell_type_order = composition_zscore.columns.tolist()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if overall_zscore is not None:
            # Compute difference: overall - group
            # Align indices and columns first
            common_rows = composition_zscore.index.intersection(overall_zscore.index)
            common_cols = composition_zscore.columns.intersection(overall_zscore.columns)
            
            # Reorder columns according to the cell type order from overall_zscore
            # Keep only columns that exist in both dataframes and in the order list
            ordered_cols = [col for col in cell_type_order if col in common_cols]
            # Add any remaining columns that weren't in the order list (at the end)
            remaining_cols = [col for col in common_cols if col not in ordered_cols]
            final_cols = ordered_cols + remaining_cols
            
            group_aligned = composition_zscore.loc[common_rows, final_cols]
            overall_aligned = overall_zscore.loc[common_rows, final_cols]
            
            # Difference: overall - group (positive = group has less, negative = group has more)
            zscore_diff = overall_aligned - group_aligned
            
            # Ensure column order is exactly as specified
            zscore_diff = zscore_diff.reindex(columns=final_cols)
            
            # Create custom annotations: "diff(group_zscore)"
            annot_array = np.empty((len(group_aligned.index), len(final_cols)), dtype=object)
            for i, row_idx in enumerate(group_aligned.index):
                for j, col_idx in enumerate(final_cols):
                    diff_val = zscore_diff.loc[row_idx, col_idx]
                    group_val = group_aligned.loc[row_idx, col_idx]
                    annot_array[i, j] = f'{diff_val:.2f}({group_val:.2f})'
            
            # Plot difference (color scale based on difference)
            sns.heatmap(
                zscore_diff,
                cmap='RdYlGn_r',  # Red-Yellow-Green reversed (red=positive diff, green=negative diff)
                center=0,
                vmin=-3,
                vmax=3,
                cbar_kws={'label': 'Z-score Difference (Overall - Group)'},
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                annot=annot_array,
                fmt='',
                annot_kws={'size': 12}
            )
            
            title = (f'Cell Fraction Difference from Overall\n'
                    f'Group: {group_name} ({n_cells:,} cells)\n'
                    f'Format: Difference(Group Z-score)')
        else:
            # Fallback if overall not available
            # Use the order from composition_zscore (should preserve order from original h5ad files)
            composition_zscore_ordered = composition_zscore[cell_type_order] if all(col in composition_zscore.columns for col in cell_type_order) else composition_zscore
            
            sns.heatmap(
                composition_zscore_ordered,
                cmap='RdYlGn_r',
                center=0,
                vmin=-2,
                vmax=2,
                cbar_kws={'label': 'Z-score'},
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                annot=True,
                fmt='.2f',
                annot_kws={'size': 12}
            )
            
            title = (f'Cell Type Composition by Cellular Neighborhood\n'
                    f'Group: {group_name} ({n_cells:,} cells)\n'
                    f'Z-score scaled by column')
        
        ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cellular Neighborhood', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved heatmap to: {save_path}")
        
        plt.close(fig)
    
    def calculate_neighborhood_frequency(
        self,
        adata: ad.AnnData,
        cn_key: str = 'cn_celltype'
    ) -> pd.DataFrame:
        """Calculate neighborhood frequency for a group."""
        cn_counts = adata.obs[cn_key].value_counts().sort_index()
        total_cells = len(adata.obs)
        cn_percentages = (cn_counts / total_cells * 100).round(2)
        
        frequency_df = pd.DataFrame({
            'Count': cn_counts,
            'Percentage': cn_percentages
        })
        frequency_df.index.name = 'Cellular_Neighborhood'
        frequency_df = frequency_df.reset_index()
        
        return frequency_df
    
    def visualize_neighborhood_frequency(
        self,
        frequency_df: pd.DataFrame,
        group_name: str,
        figsize=(10, 6),
        save_path: Optional[str] = None,
        color_palette: str = 'tab20'
    ):
        """Visualize neighborhood frequency for a group."""
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Sort by CN ID (support both integer and string CN labels)
        cn_labels_col = frequency_df['Cellular_Neighborhood']
        sorted_labels, colors_list = _sort_cn_labels_and_colors(cn_labels_col, color_palette)
        order = {lab: i for i, lab in enumerate(sorted_labels)}
        frequency_df_sorted = frequency_df.copy()
        frequency_df_sorted['_ord'] = frequency_df_sorted['Cellular_Neighborhood'].map(order)
        frequency_df_sorted = frequency_df_sorted.sort_values('_ord').drop(columns=['_ord'])
        colors_for_bars = [colors_list[order[lab]] for lab in frequency_df_sorted['Cellular_Neighborhood']]
        
        # Create bars
        bars = ax.bar(
            frequency_df_sorted['Cellular_Neighborhood'].astype(str),
            frequency_df_sorted['Count'],
            color=colors_for_bars
        )
        
        ax.set_xlabel('Cellular Neighborhood', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cell Count', fontsize=12, fontweight='bold')
        ax.set_title(f'CN Frequency (Count) - {group_name}',
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add labels with outline
        text_outline = [path_effects.withStroke(linewidth=3, foreground='white')]
        max_count = max(frequency_df_sorted['Count'])
        
        for bar, count, pct in zip(bars,
                                   frequency_df_sorted['Count'],
                                   frequency_df_sorted['Percentage']):
            height = bar.get_height()
            # Count above bar
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count):,}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold',
                   color='black', path_effects=text_outline)
            # Percentage in middle
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{pct:.1f}%',
                   ha='center', va='center', fontsize=14,
                   color='black', fontweight='bold', path_effects=text_outline)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved frequency graph to: {save_path}")
        
        plt.close(fig)
    
    def visualize_per_tile_frequency_highlighted(
        self,
        all_tiles_h5ad_dir: Path,
        group_name: str,
        cn_key: str = 'cn_celltype',
        figsize=(14, 8),
        save_path: Optional[str] = None,
        color_palette: str = 'tab20',
        show_tile_names: bool = False
    ):
        """
        Visualize per-tile frequency with highlighted group tiles.
        
        Parameters:
        -----------
        all_tiles_h5ad_dir : Path
            Directory containing ALL processed h5ad files
        group_name : str
            Group name to highlight
        show_tile_names : bool, default=False
            Whether to display tile names on x-axis (default: False to hide names)
        """
        print(f"\nGenerating per-tile frequency with {group_name} highlighted...")
        
        # Load ALL tiles
        all_h5ad_files = sorted(all_tiles_h5ad_dir.glob('*_adata_cns.h5ad'))
        
        tile_data = {}
        for h5ad_file in all_h5ad_files:
            tile_name = h5ad_file.stem.replace('_adata_cns', '')
            adata = ad.read_h5ad(h5ad_file)
            if cn_key in adata.obs.columns:
                tile_data[tile_name] = adata
        
        # Create frequency dataframe for all tiles
        tile_names = []
        cn_frequencies = []
        
        for tile_name, adata in tile_data.items():
            tile_names.append(tile_name)
            cn_counts = adata.obs[cn_key].value_counts()
            total = len(adata.obs)
            cn_freq = cn_counts / total
            cn_frequencies.append(cn_freq)
        
        # Combine into dataframe
        frequency_df = pd.DataFrame(cn_frequencies, index=tile_names)
        frequency_df = frequency_df.fillna(0)
        
        # Sort columns by CN ID (support both integer and string CN labels)
        sorted_labels, colors_sorted = _sort_cn_labels_and_colors(frequency_df.columns, color_palette)
        sorted_cols = [c for c in sorted_labels if c in frequency_df.columns]
        frequency_df_sorted = frequency_df[sorted_cols]
        
        # Plot all tiles
        fig, ax = plt.subplots(figsize=figsize)
        
        group_tiles = self.categories[group_name]
        
        # Plot all tiles together
        frequency_df_sorted.plot(kind='bar', stacked=True, ax=ax,
                                color=colors_sorted, width=0.8, legend=False)
        
        # Extract legend colors BEFORE setting transparency
        # For stacked bars, containers are organized by CN (one container per CN)
        from matplotlib.patches import Rectangle
        
        legend_handles = []
        legend_labels = []
        cn_colors = []
        
        if ax.containers:
            # Each container represents one CN type
            # Store colors before applying transparency
            for container_idx, container in enumerate(ax.containers):
                if len(container.patches) > 0:
                    # Get color from first patch in container (before transparency is set)
                    first_patch = container.patches[0]
                    color = first_patch.get_facecolor()
                    # Normalize to RGB tuple
                    if isinstance(color, np.ndarray):
                        color = tuple(color.flatten()[:4])  # Keep RGBA
                    elif isinstance(color, tuple):
                        color = color[:4] if len(color) >= 4 else color
                    cn_colors.append(color)
                    
                    lab = sorted_labels[container_idx] if container_idx < len(sorted_labels) else f'CN{container_idx + 1}'
                    legend_labels.append(str(lab) if str(lab).startswith('CN') else f'CN {lab}')
        
        # Set transparency for non-group tiles
        # In pandas stacked bar plots, patches are organized by CN first, then by tile
        # Order: [tile0_CN1, tile1_CN1, ..., tileN_CN1, tile0_CN2, tile1_CN2, ..., tileN_CN2, ...]
        # So to find which tile: tile_idx = patch_index % n_tiles
        n_tiles = len(frequency_df_sorted.index)
        tile_names_list = list(frequency_df_sorted.index)
        
        for i, patch in enumerate(ax.patches):
            # Calculate which tile this patch belongs to
            tile_idx = i % n_tiles
            if tile_idx < len(tile_names_list):
                tile_name = tile_names_list[tile_idx]
                if tile_name not in group_tiles:
                    # Set transparency for non-group tiles (30% = alpha=0.3)
                    patch.set_alpha(0.3)
                else:
                    # Ensure group tiles are fully opaque
                    patch.set_alpha(1.0)
        
        ax.set_xlabel('Tile', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency (Proportion)', fontsize=12, fontweight='bold')
        ax.set_title(f'Cellular Neighborhood Frequency by Tile\n(Group: {group_name} highlighted)',
                    fontsize=14, fontweight='bold', pad=15)
        
        # Create legend with custom handles (full opacity, not affected by bar transparency)
        for idx, (color, label) in enumerate(zip(cn_colors, legend_labels)):
            handle = Rectangle((0, 0), 1, 1, 
                             facecolor=color,
                             edgecolor='black',
                             linewidth=0.5,
                             alpha=1.0)  # Always fully opaque for legend
            legend_handles.append(handle)
        
        ax.legend(handles=legend_handles, labels=legend_labels,
                 title='Cellular Neighborhood', bbox_to_anchor=(1.05, 1),
                 loc='upper left', fontsize=9)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Handle tile name display
        if show_tile_names:
            # Highlight group tiles with bold and red color
            x_labels = ax.get_xticklabels()
            
            for label in x_labels:
                tile_name = label.get_text()
                if tile_name in group_tiles:
                    # Highlight with bold and different color
                    label.set_weight('bold')
                    label.set_color('red')
                    label.set_fontsize(10)
            
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
        else:
            # Hide tile names by default
            ax.set_xticklabels([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved per-tile frequency (highlighted) to: {save_path}")
        
        plt.close(fig)
    
    def visualize_all_groups_per_tile_frequency(
        self,
        cn_key: str = 'cn_celltype',
        figsize=(16, 8),
        save_path: Optional[str] = None,
        color_palette: str = 'tab20'
    ):
        """
        Visualize per-tile frequency for all groups in one figure, with bars clustered by group.
        
        Parameters:
        -----------
        cn_key : str
            Key in adata.obs containing CN labels
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        color_palette : str
            Color palette name
        """
        print(f"\nGenerating combined per-tile frequency for all groups...")
        
        # Get all groups and create tile-to-group mapping
        groups = [key for key in self.categories.keys() if key != 'metadata']
        tile_to_group = {tile: group for group in groups for tile in self.categories[group]}
        
        # Load tiles and compute frequencies
        frequency_data = []
        for h5ad_file in sorted(self.processed_h5ad_dir.glob('*_adata_cns.h5ad')):
            tile_name = h5ad_file.stem.replace('_adata_cns', '')
            if tile_name in tile_to_group:
                adata = ad.read_h5ad(h5ad_file)
                if cn_key in adata.obs.columns:
                    cn_freq = adata.obs[cn_key].value_counts(normalize=True)
                    frequency_data.append((tile_name, cn_freq))
        
        # Create and sort dataframe
        frequency_df = pd.DataFrame(dict(frequency_data)).T.fillna(0)
        # Convert all column names to strings for consistency
        frequency_df.columns = [str(col) for col in frequency_df.columns]
        # Sort columns by CN ID (support both integer and string CN labels)
        sorted_labels, colors = _sort_cn_labels_and_colors(frequency_df.columns, color_palette)
        sorted_cols = [str(c) for c in sorted_labels if str(c) in frequency_df.columns]
        frequency_df = frequency_df[sorted_cols]
        
        # Sort tiles by group, then by tile name
        group_order = {group: idx for idx, group in enumerate(groups)}
        sorted_tiles = sorted(frequency_df.index, 
                            key=lambda t: (group_order.get(tile_to_group.get(t, ''), 999), t))
        frequency_df = frequency_df.reindex(sorted_tiles)
        
        fig, ax = plt.subplots(figsize=figsize)
        frequency_df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8, legend=False)
        
        # Create legend
        from matplotlib.patches import Rectangle
        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5, alpha=1.0)
            for color in colors
        ]
        legend_labels = [str(lab) if str(lab).startswith('CN') else f'CN {lab}' for lab in sorted_labels]
        
        # Calculate group boundaries and set x-axis labels
        group_boundaries = {}
        current_idx = 0
        for group_name in groups:
            group_tiles = [t for t in sorted_tiles if tile_to_group.get(t) == group_name]
            if group_tiles:
                n_tiles = len(group_tiles)
                group_boundaries[group_name] = (current_idx, current_idx + n_tiles - 1)
                current_idx += n_tiles
        
        # Set group labels at center of each group
        x_positions = [(start + end) / 2.0 for start, end in group_boundaries.values()]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(list(group_boundaries.keys()), rotation=45, ha='right', 
                          fontsize=11, fontweight='bold')
        
        # Add separator lines between groups
        for start, end in list(group_boundaries.values())[:-1]:
            ax.axvline(x=end + 0.5, color='black', linestyle='--', linewidth=2.5)
        
        # Formatting
        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency (Proportion)', fontsize=12, fontweight='bold')
        ax.set_title('Cellular Neighborhood Frequency by Tile (Grouped by Category)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(handles=legend_handles, labels=legend_labels,
                 title='Cellular Neighborhood', bbox_to_anchor=(1.05, 1),
                 loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved combined per-tile frequency to: {save_path}")
        plt.close(fig)
    
    def generate_unified_analysis(
        self,
        k: int = 20,
        n_clusters: int = 7,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type',
        color_palette: str = 'tab20'
    ):
        """
        Generate unified analysis visualizations from processed h5ad files.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors used (for title)
        n_clusters : int
            Number of clusters used (for title)
        cn_key : str
            Key in adata.obs containing CN labels
        celltype_key : str
            Key in adata.obs containing cell type labels
        color_palette : str
            Color palette name
        """
        print(f"\n{'='*80}")
        print("GENERATING UNIFIED ANALYSIS VISUALIZATIONS")
        print(f"{'='*80}")
        
        # Load all processed tiles
        combined_adata = self.load_all_processed_tiles(cn_key, celltype_key)
        
        # Compute composition (order preserved from original h5ad files)
        composition, composition_zscore = self.compute_unified_cn_composition(
            combined_adata, cn_key, celltype_key
        )
        
        # Save composition CSV (order preserved from original h5ad files)
        comp_path = self.base_output_dir / 'unified_analysis' / 'unified_cn_composition.csv'
        composition.to_csv(comp_path)
        print(f"  ✓ Saved composition to: {comp_path}")
        
        # Visualize unified heatmap (order preserved from original h5ad files)
        heatmap_path = self.base_output_dir / 'unified_analysis' / 'unified_cn_composition_heatmap.png'
        self.visualize_unified_cn_composition(
            combined_adata,
            composition_zscore,
            k=k,
            n_clusters=n_clusters,
            save_path=str(heatmap_path),
            show_values=True
        )
        
        # Visualize neighborhood frequency distributions
        # Overall frequency
        overall_freq_path = self.base_output_dir / 'unified_analysis' / 'neighborhood_frequency_overall.png'
        self.visualize_unified_neighborhood_frequency(
            combined_adata,
            cn_key=cn_key,
            group_by_tile=False,
            figsize=(10, 6),
            save_path=str(overall_freq_path),
            color_palette=color_palette
        )
        
        # Per-tile frequency
        per_tile_freq_path = self.base_output_dir / 'unified_analysis' / 'neighborhood_frequency_per_tile.png'
        self.visualize_unified_neighborhood_frequency(
            combined_adata,
            cn_key=cn_key,
            group_by_tile=True,
            figsize=(14, 8),
            save_path=str(per_tile_freq_path),
            color_palette=color_palette
        )
        
        # Save summary statistics
        print("\nSaving summary statistics...")
        tile_names = sorted(combined_adata.obs['tile_name'].unique().tolist())
        
        summary = {
            'analysis_type': 'Unified Cellular Neighborhoods',
            'n_tiles': len(tile_names),
            'tile_names': tile_names,
            'total_cells': int(combined_adata.n_obs),
            'total_genes': int(combined_adata.n_vars),
            'parameters': {
                'k_neighbors': k,
                'n_clusters': n_clusters,
                'celltype_key': celltype_key
            },
            'cn_distribution': combined_adata.obs[cn_key].value_counts().to_dict(),
            'cell_type_distribution': combined_adata.obs[celltype_key].value_counts().to_dict(),
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
        summary_path = self.base_output_dir / 'unified_analysis' / 'unified_cn_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved summary to: {summary_path}")
        
        print(f"\n✓ Unified analysis complete!")
        print(f"  Results saved to: {self.base_output_dir}/unified_analysis/")
    
    def generate_individual_tiles(
        self,
        k: int = 20,
        n_clusters: int = 7,
        cn_key: str = 'cn_celltype',
        coord_key: str = 'spatial',
        palette: str = 'tab20'
    ):
        """
        Generate individual tile spatial CN maps from processed h5ad files.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors used (for title)
        n_clusters : int
            Number of clusters used (for title)
        cn_key : str
            Key in adata.obs containing CN labels
        coord_key : str
            Key in adata.obsm containing spatial coordinates
        palette : str
            Color palette name
        """
        print(f"\n{'='*80}")
        print("GENERATING INDIVIDUAL TILE SPATIAL MAPS")
        print(f"{'='*80}")
        
        # Load all processed tiles
        combined_adata = self.load_all_processed_tiles(cn_key)
        
        # Visualize individual tile maps
        self.visualize_individual_tile_cns(
            combined_adata,
            cn_key=cn_key,
            coord_key=coord_key,
            palette=palette,
            k=k,
            n_clusters=n_clusters
        )
        
        print(f"\n✓ Individual tile maps complete!")
        print(f"  Results saved to: {self.base_output_dir}/individual_tiles/")
    
    def analyze_group(
        self,
        group_name: str,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type',
        color_palette: str = 'tab20'
    ):
        """Run complete analysis for a single group."""
        print(f"\n{'='*80}")
        print(f"ANALYZING GROUP: {group_name.upper()}")
        print(f"{'='*80}")
        
        # Load data
        adata = self.load_group_data(group_name, cn_key, celltype_key)
        
        # Load overall composition for comparison
        print("\nLoading overall CN composition for comparison...")
        overall_zscore = self.load_overall_composition(cn_key=cn_key)
        
        # Compute composition
        print("\nComputing CN composition...")
        composition, composition_zscore = self.compute_cn_composition(
            adata, cn_key, celltype_key
        )
        
        # Save composition CSV
        csv_path = self.output_dir / f'cn_cell_fraction_{group_name}.csv'
        composition.to_csv(csv_path)
        print(f"  ✓ Saved composition CSV to: {csv_path}")
        
        # Visualize heatmap with difference from overall
        print("\nGenerating cell fraction difference heatmap...")
        heatmap_path = self.output_dir / f'cell_fraction_difference_{group_name}.png'
        self.visualize_cn_composition_heatmap(
            composition_zscore,
            group_name,
            adata.n_obs,
            overall_zscore=overall_zscore,
            save_path=str(heatmap_path)
        )
        
        # Calculate frequency
        print("\nCalculating neighborhood frequency...")
        frequency_df = self.calculate_neighborhood_frequency(adata, cn_key)
        
        # Visualize frequency
        print("\nGenerating neighborhood frequency graph...")
        freq_path = self.output_dir / f'neighborhood_frequency_{group_name}.png'
        self.visualize_neighborhood_frequency(
            frequency_df,
            group_name,
            save_path=str(freq_path),
            color_palette=color_palette
        )
        
        # Note: Per-tile frequency visualization is now done for all groups together
        # in analyze_all_groups() to show bars clustered by group
        
        print(f"\n✓ Group analysis complete for: {group_name}")
        print(f"  Total cells: {adata.n_obs:,}")
        print(f"  Tiles: {len(self.categories[group_name])}")
    
    def analyze_all_groups(
        self,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type',
        color_palette: str = 'tab20',
        k: int = 20,
        n_clusters: int = 7,
        generate_unified: bool = True,
        generate_individual: bool = True
    ):
        """
        Run analysis for all groups and optionally generate unified analysis and individual tiles.
        
        Parameters:
        -----------
        cn_key : str
            Key in adata.obs containing CN labels
        celltype_key : str
            Key in adata.obs containing cell type labels
        color_palette : str
            Color palette name
        k : int
            Number of nearest neighbors (for titles)
        n_clusters : int
            Number of clusters (for titles)
        generate_unified : bool
            Whether to generate unified analysis visualizations
        generate_individual : bool
            Whether to generate individual tile spatial maps
        """
        banner = "=" * 80
        print(f"\n{banner}")
        print("GROUP-BASED CELLULAR NEIGHBORHOOD ANALYSIS")
        print(f"{banner}")
        print(f"Processed h5ad directory: {self.processed_h5ad_dir}")
        print(f"Categories JSON: {self.categories_json}")
        print(f"Base output directory: {self.base_output_dir}")
        print(f"Group analysis output: {self.output_dir}")
        print(f"{banner}\n")
        
        # Generate unified analysis and individual tiles first (if requested)
        if generate_unified:
            self.generate_unified_analysis(
                k=k,
                n_clusters=n_clusters,
                cn_key=cn_key,
                celltype_key=celltype_key,
                color_palette=color_palette
            )
        
        if generate_individual:
            self.generate_individual_tiles(
                k=k,
                n_clusters=n_clusters,
                cn_key=cn_key,
                palette=color_palette
            )
        
        # Run group-specific analysis
        groups = [key for key in self.categories.keys() if key != 'metadata']
        
        for group in groups:
            self.analyze_group(group, cn_key, celltype_key, color_palette)
        
        # Generate combined per-tile frequency figure for all groups
        print(f"\n{'='*80}")
        print("GENERATING COMBINED PER-TILE FREQUENCY FIGURE")
        print(f"{'='*80}")
        per_tile_path = self.output_dir / 'neighborhood_frequency_per_tile_all_groups.png'
        self.visualize_all_groups_per_tile_frequency(
            cn_key=cn_key,
            save_path=str(per_tile_path),
            color_palette=color_palette
        )
        
        print(f"\n{banner}")
        print("ALL ANALYSES COMPLETE!")
        print(f"{banner}")
        print(f"\nResults saved to: {self.base_output_dir}/")
        print(f"\nUnified analysis (if generated):")
        print(f"  - unified_analysis/unified_cn_composition_heatmap.png")
        print(f"  - unified_analysis/neighborhood_frequency_overall.png")
        print(f"  - unified_analysis/neighborhood_frequency_per_tile.png")
        print(f"\nIndividual tiles (if generated):")
        print(f"  - individual_tiles/spatial_cns_*.png")
        print(f"\nGroup-specific analysis:")
        print(f"  - {self.output_dir.name}/cell_fraction_difference_{{group}}.png")
        print(f"  - {self.output_dir.name}/cn_cell_fraction_{{group}}.csv")
        print(f"  - {self.output_dir.name}/neighborhood_frequency_{{group}}.png")
        print(f"  - {self.output_dir.name}/neighborhood_frequency_per_tile_all_groups.png")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Group-based Cellular Neighborhood Analysis'
    )
    parser.add_argument(
        '--processed_h5ad_dir',
        default='/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/cn_unified_results/all_n_cluster=9_sub/processed_h5ad',
        help='Directory containing processed h5ad files with CN annotations'
    )
    parser.add_argument(
        '--categories_json',
        default='tile_categories_88_tiles.json',
        help='Path to tile categories JSON file (relative to script directory)'
    )
    parser.add_argument(
        '--output_dir',
        default='cn_unified_results_groups',
        help='Output directory for group-specific results (relative to script directory)'
    )
    parser.add_argument(
        '--cn_key',
        default='cn_celltype',
        help='Column name for CN labels (default: cn_celltype). Use cn_celltype_sub for sub-clustered results.'
    )
    parser.add_argument(
        '--celltype_key',
        default='cell_type',
        help='Column name for cell types (default: cell_type)'
    )
    parser.add_argument(
        '--color_palette',
        default='tab20',
        help='Color palette (default: tab20, supports up to 20 distinct colors)'
    )
    parser.add_argument(
        '--group',
        default=None,
        help='Analyze specific group only (default: all groups)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=20,
        help='Number of nearest neighbors used (for titles, default: 20)'
    )
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=None,
        help='Number of clusters used (for titles, will try to infer from directory name if not provided)'
    )
    parser.add_argument(
        '--no-generate_unified',
        dest='generate_unified',
        action='store_false',
        default=True,
        help='Skip unified analysis visualizations (default: generate unified analysis)'
    )
    parser.add_argument(
        '--no-generate_individual',
        dest='generate_individual',
        action='store_false',
        default=True,
        help='Skip individual tile spatial maps (default: generate individual tiles)'
    )
    
    args = parser.parse_args()

    # Auto-detect cn_key for sub-clustered data: when processed_h5ad_dir contains "_sub",
    # use cn_celltype_sub so we load sub-clustered CNs and find unified_cn_composition_sub.csv
    if '_sub' in str(args.processed_h5ad_dir) and args.cn_key == 'cn_celltype':
        args.cn_key = 'cn_celltype_sub'
        print(f"Note: Detected sub-clustered data (_sub in path), using cn_key='cn_celltype_sub'")
    
    # Try to infer n_clusters from directory name if not provided
    if args.n_clusters is None:
        import re
        dir_name = Path(args.processed_h5ad_dir).parent.name
        match = re.search(r'n_cluster[=_]?(\d+)', dir_name)
        if match:
            args.n_clusters = int(match.group(1))
        else:
            args.n_clusters = 7  # Default
    
    # Initialize analyzer
    analyzer = GroupCNAnalyzer(
        processed_h5ad_dir=args.processed_h5ad_dir,
        categories_json=args.categories_json,
        output_dir=args.output_dir
    )
    
    # Run analysis
    if args.group:
        analyzer.analyze_group(
            args.group,
            cn_key=args.cn_key,
            celltype_key=args.celltype_key,
            color_palette=args.color_palette
        )
    else:
        analyzer.analyze_all_groups(
            cn_key=args.cn_key,
            celltype_key=args.celltype_key,
            color_palette=args.color_palette,
            k=args.k,
            n_clusters=args.n_clusters,
            generate_unified=args.generate_unified,
            generate_individual=args.generate_individual
        )


if __name__ == '__main__':
    main()

