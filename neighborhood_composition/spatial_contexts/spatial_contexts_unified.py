"""
Spatial Context Detection for Unified CN Results

This script processes the results from cn_unified_kmeans.py to detect spatial contexts (SCs).
It reads the processed h5ad files with CN annotations and performs spatial context analysis.

Key Features:
- Reads processed h5ad files from cn_unified_kmeans.py output
- Detects spatial contexts based on CN mixtures (k=40, threshold=0.9)
- Generates spatial context maps and interaction graphs
- Optimized for unified CN results

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
import seaborn as sns
import networkx as nx
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from scipy.sparse import csr_matrix
import warnings
import glob
import time

# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)
warnings.filterwarnings('ignore')


def discover_unified_cn_files(cn_results_dir: str) -> List[str]:
    """
    Discover processed h5ad files from cn_unified_kmeans.py results.
    
    Parameters:
    -----------
    cn_results_dir : str
        Path to cn_unified_kmeans.py results directory
        
    Returns:
    --------
    file_paths : List[str]
        List of paths to processed h5ad files
    """
    processed_h5ad_dir = Path(cn_results_dir) / 'processed_h5ad'
    
    if not processed_h5ad_dir.exists():
        print(f"Warning: Directory {processed_h5ad_dir} does not exist")
        return []
    
    # Find all h5ad files with CN annotations
    pattern = str(processed_h5ad_dir / '*_adata_cns.h5ad')
    file_paths = glob.glob(pattern)
    
    print(f"Found {len(file_paths)} processed h5ad files from unified CN detection:")
    for i, path in enumerate(file_paths, 1):
        filename = Path(path).name
        print(f"  {i}. {filename}")
    
    return sorted(file_paths)


def standardize_sc_label(sc_label: str) -> str:
    """
    Standardize spatial context label by sorting CN numbers numerically.
    
    Labels with the same numbers in different order (e.g., '6_5_3_2' and '6_2_5_3') 
    represent the same spatial context and should be standardized to the same string.
    
    Parameters:
    -----------
    sc_label : str
        SC label string (e.g., '6_5_3_2' or '2_6_3_5')
        
    Returns:
    --------
    standardized_label : str
        Standardized SC label with numbers sorted (e.g., '2_3_5_6')
        
    Examples:
    ---------
    >>> standardize_sc_label('6_5_3_2')
    '2_3_5_6'
    >>> standardize_sc_label('2_6_3_5')
    '2_3_5_6'
    >>> standardize_sc_label('1')
    '1'
    >>> standardize_sc_label('Other')
    'Other'
    """
    # Handle special cases
    if not sc_label or sc_label == 'Other':
        return sc_label
    
    try:
        # Split by underscore and convert to integers
        numbers = [int(x) for x in sc_label.split('_')]
        
        # Sort numerically
        numbers.sort()
        
        # Join back with underscores
        return '_'.join(map(str, numbers))
        
    except (ValueError, AttributeError):
        # If conversion fails, return original label
        return sc_label


def load_unified_cn_data(file_paths: List[str], coord_offset: bool = False) -> ad.AnnData:
    """
    Load and combine processed h5ad files from unified CN detection.
    
    Parameters:
    -----------
    file_paths : List[str]
        List of paths to h5ad files
    coord_offset : bool
        Whether to offset spatial coordinates to avoid overlap
        
    Returns:
    --------
    combined_adata : AnnData
        Combined AnnData object with all tiles
    """
    print(f"Loading and combining {len(file_paths)} tiles from unified CN results...")
    
    adata_list = []
    coord_offset_x = 0
    coord_offset_y = 0
    
    for i, file_path in enumerate(file_paths, 1):
        tile_name = Path(file_path).stem.replace('_adata_cns', '')
        print(f"  [{i}/{len(file_paths)}] Loading: {tile_name}")
        
        try:
            adata = sc.read_h5ad(file_path)
            
            # Add tile identifier if not present
            if 'tile_name' not in adata.obs.columns:
                adata.obs['tile_name'] = tile_name
            
            # Verify required columns
            if 'cn_celltype' not in adata.obs.columns:
                print(f"    Warning: No 'cn_celltype' column found in {file_path}")
                continue
            
            # Offset spatial coordinates if requested
            if coord_offset and 'spatial' in adata.obsm:
                coords = adata.obsm['spatial'].copy()
                coords[:, 0] += coord_offset_x
                coords[:, 1] += coord_offset_y
                adata.obsm['spatial'] = coords
                
                # Update offset for next tile
                max_x = coords[:, 0].max()
                coord_offset_x = max_x + 500  # 500 pixel gap between tiles
            
            adata_list.append(adata)
            print(f"    ✓ Loaded {adata.n_obs} cells, {adata.n_vars} genes")
            
        except Exception as e:
            print(f"    ✗ Error loading {file_path}: {str(e)}")
            continue
    
    if not adata_list:
        raise ValueError("No valid tiles could be loaded")
    
    # Combine all tiles
    print("\nCombining tiles into single dataset...")
    combined_adata = ad.concat(adata_list, join='outer', index_unique='-')
    
    print(f"✓ Combined dataset: {combined_adata.n_obs} cells, {combined_adata.n_vars} genes")
    print(f"  Tiles: {combined_adata.obs['tile_name'].nunique()}")
    print(f"  CNs: {combined_adata.obs['cn_celltype'].nunique()}")
    
    return combined_adata


class SpatialContextDetector:
    """
    Detects Spatial Contexts (SCs) based on cellular neighborhood (CN) composition
    in larger neighborhoods, similar to Figure 19 from the paper.

    Spatial Contexts represent regions where different CNs interact and are detected by:
    1. Building a larger k-NN graph (k=40) to capture broader spatial interactions
    2. Computing CN fraction composition in each cell's neighborhood
    3. Assigning minimal CN combinations that exceed a threshold (default: 0.9)
    4. Filtering rare SCs and visualizing spatial patterns

    Reference:
    Schürch et al. (2020) "Coordinated cellular neighborhoods orchestrate
    antitumoral immunity at the colorectal cancer invasive front"
    """

    def __init__(self, adata: ad.AnnData, cn_key: str = 'cn_celltype'):
        """
        Initialize SC detector.

        Parameters:
        -----------
        adata : AnnData
            AnnData object with spatial coordinates and CN annotations
        cn_key : str, default='cn_celltype'
            Key in adata.obs containing CN labels
        """
        self.adata = adata
        self.cn_key = cn_key
        self.sc_labels = None
        self.aggregated_cn_fractions = None
        self.sc_graph = None

        # Verify CN labels exist
        if cn_key not in adata.obs.columns:
            raise ValueError(f"CN key '{cn_key}' not found in adata.obs. "
                           f"Please run unified CN detection first.")

    def build_knn_graph(
        self,
        k: int = 20,
        coord_key: str = 'spatial',
        key_added: str = 'spatial_connectivities_sc'
    ):
        """
        Build k-nearest neighbor graph for spatial context detection.
        Uses a larger k than CN detection to capture broader spatial interactions.

        Parameters:
        -----------
        k : int, default=40
            Number of nearest neighbors (larger than CN detection)
        coord_key : str, default='spatial'
            Key in adata.obsm containing spatial coordinates
        key_added : str
            Key to store the connectivity matrix
        """
        print(f"Building {k}-nearest neighbor graph for SC detection...")

        # Build spatial neighbors graph
        sq.gr.spatial_neighbors(
            self.adata,
            spatial_key=coord_key,
            coord_type='generic',
            n_neighs=k,
            radius=None  # Use KNN, not radius
        )

        # Squidpy uses fixed key names, rename if needed
        actual_key = 'spatial_connectivities'
        if key_added != actual_key and actual_key in self.adata.obsp:
            self.adata.obsp[key_added] = self.adata.obsp[actual_key]

        # Print statistics
        connectivity = self.adata.obsp[key_added]
        avg_neighbors = connectivity.sum(axis=1).mean()
        print(f"  - Average neighbors per cell: {avg_neighbors:.2f}")
        print(f"  - Connectivity matrix shape: {connectivity.shape}")

        return self

    def aggregate_cn_fractions(
        self,
        connectivity_key: str = 'spatial_connectivities_sc',
        output_key: str = 'aggregated_cn_fractions'
    ):
        """
        For each cell, compute the fraction of each CN type in its neighborhood.
        This is analogous to aggregating cell types for CN detection, but now
        we aggregate CN labels instead.

        Parameters:
        -----------
        connectivity_key : str
            Key in adata.obsp containing spatial connectivity
        output_key : str
            Key to store aggregated CN fractions
        """
        print(f"Aggregating CN fractions in neighborhoods...")

        # Get CN labels
        cn_labels = self.adata.obs[self.cn_key].values
        unique_cns = sorted(self.adata.obs[self.cn_key].cat.categories)

        # Get connectivity matrix
        connectivity = self.adata.obsp[connectivity_key]

        # Initialize aggregated CN fractions matrix
        n_cells = self.adata.n_obs
        n_cns = len(unique_cns)
        aggregated = np.zeros((n_cells, n_cns))

        print(f"  Processing {n_cells} cells...")

        # For each cell, compute CN fractions in neighborhood
        for i in range(n_cells):
            if i % 10000 == 0 and i > 0:
                print(f"    Processed {i:,}/{n_cells:,} cells ({100*i/n_cells:.1f}%)")
            
            # Get neighbors (including self)
            neighbors_mask = connectivity[i].toarray().flatten() > 0

            if neighbors_mask.sum() > 0:
                # Get CN labels of neighbors
                neighbor_cns = cn_labels[neighbors_mask]

                # Compute fractions for each CN
                for j, cn in enumerate(unique_cns):
                    aggregated[i, j] = (neighbor_cns == cn).sum() / neighbors_mask.sum()

        # Store in adata
        self.aggregated_cn_fractions = pd.DataFrame(
            aggregated,
            columns=[f'CN_{cn}' for cn in unique_cns],
            index=self.adata.obs_names
        )

        self.adata.obsm[output_key] = aggregated

        print(f"  - Aggregated CN fractions shape: {aggregated.shape}")
        print(f"  - CN types: {unique_cns}")

        return self

    def detect_spatial_contexts(
        self,
        threshold: float = 0.9,
        min_fraction: float = 0.05,
        aggregated_key: str = 'aggregated_cn_fractions',
        output_key: str = 'spatial_context'
    ):
        """
        Assign spatial context labels based on minimal CN combinations that
        exceed the threshold.

        Algorithm:
        1. For each cell, sort CN fractions from high to low
        2. Add CNs cumulatively until sum >= threshold
        3. Only include CNs with fraction >= min_fraction
        4. SC label = sorted CN IDs joined by "_" (e.g., "1_2", "2_4_5")

        Parameters:
        -----------
        threshold : float, default=0.9
            Cumulative fraction threshold for SC assignment
        min_fraction : float, default=0.05
            Minimum individual CN fraction to be included (filters out noise)
        aggregated_key : str
            Key in adata.obsm containing aggregated CN fractions
        output_key : str
            Key to store SC labels in adata.obs
        """
        print(f"Detecting spatial contexts (threshold={threshold}, min_fraction={min_fraction})...")

        # Get aggregated CN fractions
        aggregated = self.adata.obsm[aggregated_key]
        unique_cns = sorted(self.adata.obs[self.cn_key].cat.categories)

        # Initialize SC labels
        sc_labels = []

        # For each cell, determine minimal CN combination
        for i in range(len(aggregated)):
            cn_fractions = aggregated[i]

            # Sort CNs by fraction (high to low)
            sorted_indices = np.argsort(cn_fractions)[::-1]
            sorted_cns = [unique_cns[idx] for idx in sorted_indices]
            sorted_fractions = cn_fractions[sorted_indices]

            # Add CNs until cumulative sum >= threshold
            cumsum = 0.0
            selected_cns = []

            for cn, frac in zip(sorted_cns, sorted_fractions):
                # Skip CNs with very small fractions (noise filtering)
                if frac < min_fraction:
                    break
                
                cumsum += frac
                selected_cns.append(str(cn))

                # Stop when we reach the threshold
                if cumsum >= threshold:
                    break

            # If no CNs meet the criteria, use the top CN
            if not selected_cns:
                selected_cns.append(str(sorted_cns[0]))

            # Create SC label by joining CN IDs (sorted by fraction)
            sc_label = '_'.join(selected_cns)
            # Standardize the label to ensure consistent ordering
            standardized_label = standardize_sc_label(sc_label)
            sc_labels.append(standardized_label)

        # Store SC labels
        self.sc_labels = pd.Categorical(sc_labels)
        self.adata.obs[output_key] = self.sc_labels

        # Print SC statistics
        sc_counts = pd.Series(sc_labels).value_counts()
        print(f"  - Detected {len(sc_counts)} unique spatial contexts")
        print(f"  - Top 10 SCs by cell count:")
        for sc, count in sc_counts.head(10).items():
            print(f"    SC '{sc}': {count} cells")

        return self

    def get_filtered_scs(
        self,
        min_cells: int = 100,
        min_groups: Optional[int] = None,
        group_key: Optional[str] = None,
        sc_key: str = 'spatial_context'
    ) -> set:
        """
        Get the set of SCs that pass the filtering criteria (for graph visualization only).
        This does NOT modify the data, just returns which SCs should be shown in the graph.

        Parameters:
        -----------
        min_cells : int, default=100
            Minimum number of cells for an SC to be retained
        min_groups : int, optional
            Minimum number of groups (e.g., tiles) an SC must appear in
        group_key : str, optional
            Key in adata.obs containing group identifiers (e.g., 'tile_name')
        sc_key : str
            Key in adata.obs containing SC labels

        Returns:
        --------
        valid_scs : set
            Set of SC labels that pass all filtering criteria
        """
        print(f"Determining SCs for graph visualization...")
        print(f"  - Minimum cells: {min_cells}")
        if min_groups is not None and group_key is not None:
            print(f"  - Minimum groups ({group_key}): {min_groups}")

        sc_labels = self.adata.obs[sc_key].values
        sc_counts = pd.Series(sc_labels).value_counts()

        # Start with all SCs
        valid_scs = set(sc_counts.index)
        print(f"  - Total SCs before filtering: {len(valid_scs)}")

        # Apply cell count filter
        if min_cells > 0:
            cell_valid = set(sc_counts[sc_counts >= min_cells].index)
            valid_scs = valid_scs.intersection(cell_valid)
            print(f"  - SCs passing cell count filter: {len(valid_scs)}")

        # Apply group occurrence filter
        if min_groups is not None and group_key is not None and min_groups > 0:
            if group_key not in self.adata.obs.columns:
                print(f"  WARNING: group_key '{group_key}' not found, skipping group filter")
            else:
                # Count groups for each SC
                sc_group_counts = self.adata.obs.groupby(sc_key)[group_key].nunique()
                group_valid = set(sc_group_counts[sc_group_counts >= min_groups].index)
                valid_scs = valid_scs.intersection(group_valid)
                print(f"  - SCs passing group filter: {len(valid_scs)}")

        print(f"  - Final SCs for graph visualization: {len(valid_scs)}")
        return valid_scs

    def filter_spatial_contexts(
        self,
        min_cells: int = 100,
        min_groups: Optional[int] = None,
        group_key: Optional[str] = None,
        sc_key: str = 'spatial_context',
        filtered_key: str = 'spatial_context_filtered'
    ):
        """
        Create a copy of SC labels for visualization purposes only.
        This does NOT filter the data - all SCs are kept in the original data.

        Parameters:
        -----------
        min_cells : int, default=100
            Minimum number of cells for an SC to be retained (for graph only)
        min_groups : int, optional
            Minimum number of groups (e.g., tiles) an SC must appear in (for graph only)
        group_key : str, optional
            Key in adata.obs containing group identifiers (e.g., 'tile_name')
        sc_key : str
            Key in adata.obs containing SC labels
        filtered_key : str
            Key to store visualization labels (same as original for now)
        """
        print(f"Preparing SC labels for visualization...")
        
        # Just copy the original labels - no filtering applied to data
        sc_labels = self.adata.obs[sc_key].values
        self.adata.obs[filtered_key] = pd.Categorical(sc_labels)
        
        print(f"  - All {len(pd.Series(sc_labels).value_counts())} SCs kept in data")
        print(f"  - Filtering will be applied only to graph visualization")

        return self

    def _should_connect_hierarchically(self, sc1: str, sc2: str) -> bool:
        """
        Determine if two SCs should be connected based on hierarchical rules.
        
        Rules:
        1. Only connect nodes in adjacent rows (differ by exactly 1 CN)
        2. The node with more CNs must contain ALL numbers from the node with fewer CNs
        3. Example: '2_4_6' can connect to '2_3_4_6' or '2_4_5_6' but not '1_2_5_6'
        
        Parameters:
        -----------
        sc1 : str
            First SC label (e.g., '2_4_6')
        sc2 : str
            Second SC label (e.g., '2_3_4_6')
            
        Returns:
        --------
        bool
            True if the SCs should be connected
        """
        try:
            # Parse SC labels into sets of numbers
            nums1 = set(int(x) for x in sc1.split('_'))
            nums2 = set(int(x) for x in sc2.split('_'))
            
            # Check if they differ by exactly 1 CN (adjacent rows)
            if abs(len(nums1) - len(nums2)) != 1:
                return False
            
            # The larger set must contain all numbers from the smaller set
            if len(nums1) < len(nums2):
                # sc1 is smaller, sc2 must contain all of sc1's numbers
                return nums1.issubset(nums2)
            else:
                # sc2 is smaller, sc1 must contain all of sc2's numbers
                return nums2.issubset(nums1)
                
        except (ValueError, AttributeError):
            # If parsing fails, don't connect
            return False

    def visualize_spatial_contexts(
        self,
        sc_key: str = 'spatial_context_filtered',
        img_id_key: str = 'tile_name',
        coord_key: str = 'spatial',
        point_size: float = 2.0,
        palette: str = 'tab20',
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        filtered_scs: Optional[set] = None
    ):
        """
        Visualize spatial contexts spatially (similar to Fig 19a).

        Parameters:
        -----------
        sc_key : str
            Key in adata.obs containing SC labels
        img_id_key : str
            Key in adata.obs containing image/sample identifiers
        coord_key : str
            Key in adata.obsm containing spatial coordinates
        point_size : float
            Size of points in scatter plot
        palette : str
            Color palette name
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save figure
        """
        print("Visualizing spatial contexts...")

        # Get unique images
        images = self.adata.obs[img_id_key].unique()
        n_images = len(images)

        # Calculate grid dimensions
        n_cols = min(4, n_images)
        n_rows = int(np.ceil(n_images / n_cols))

        if figsize is None:
            figsize = (n_cols * 4, n_rows * 4)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        if n_images == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Get coordinates
        coords = self.adata.obsm[coord_key]

        # Get SC labels
        sc_labels = self.adata.obs[sc_key].values
        
        # Filter SCs if specified
        if filtered_scs is not None:
            # Only show filtered SCs, set others to 'Other'
            filtered_sc_labels = []
            for sc in sc_labels:
                if sc in filtered_scs:
                    filtered_sc_labels.append(sc)
                else:
                    filtered_sc_labels.append('Other')
            sc_labels = np.array(filtered_sc_labels)
        
        unique_scs = sorted([sc for sc in self.adata.obs[sc_key].cat.categories if sc != 'Other'])
        
        # If filtering is applied, only show filtered SCs
        if filtered_scs is not None:
            unique_scs = [sc for sc in unique_scs if sc in filtered_scs]

        # Add 'Other' to end if it exists
        if 'Other' in self.adata.obs[sc_key].cat.categories:
            unique_scs.append('Other')

        # Get colors
        n_scs = len(unique_scs)
        if n_scs <= 20:
            colors_palette = sns.color_palette(palette, n_scs)
        else:
            colors_palette = sns.color_palette('husl', n_scs)

        color_map = {sc: colors_palette[i] for i, sc in enumerate(unique_scs)}

        # Plot each image
        for idx, img in enumerate(images):
            ax = axes[idx]

            # Get cells from this image
            mask = self.adata.obs[img_id_key] == img
            img_coords = coords[mask]
            img_scs = sc_labels[mask]

            # Plot each SC
            for sc in unique_scs:
                sc_mask = img_scs == sc
                if sc_mask.sum() > 0:
                    ax.scatter(
                        img_coords[sc_mask, 0],
                        img_coords[sc_mask, 1],
                        c=[color_map[sc]],
                        s=point_size,
                        alpha=0.7,
                        label=f'SC {sc}'
                    )

            ax.set_title(img, fontsize=10)
            ax.set_xlabel('X coordinate (pixels)')
            ax.set_ylabel('Y coordinate (pixels)')
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Remove empty subplots
        for idx in range(n_images, len(axes)):
            fig.delaxes(axes[idx])

        # Add a single legend for all subplots
        if n_scs > 0:
            # Create legend handles
            legend_handles = []
            for sc in unique_scs:
                if sc in color_map:
                    legend_handles.append(
                        plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color_map[sc], markersize=8, 
                                  label=f'SC {sc}')
                    )
            
            # Add legend to the figure
            fig.legend(handles=legend_handles, 
                      loc='center left', 
                      bbox_to_anchor=(1.02, 0.5),
                      title='Spatial Contexts',
                      fontsize=10,
                      title_fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to: {save_path}")

        plt.close()

        return fig

    def build_sc_interaction_graph(
        self,
        sc_key: str = 'spatial_context_filtered',
        connectivity_key: str = 'spatial_connectivities_sc',
        filtered_scs: Optional[set] = None
    ) -> nx.Graph:
        """
        Build interaction graph between spatial contexts.
        Nodes = SCs, edges = SCs that share neighbors in the k-NN graph.

        Parameters:
        -----------
        sc_key : str
            Key in adata.obs containing SC labels
        connectivity_key : str
            Key in adata.obsp containing spatial connectivity

        Returns:
        --------
        G : networkx.Graph
            SC interaction graph
        """
        print("Building SC interaction graph...")

        # Get SC labels and connectivity
        sc_labels = self.adata.obs[sc_key].values
        connectivity = self.adata.obsp[connectivity_key]

        # Initialize graph
        G = nx.Graph()

        # Count cells in each SC
        sc_counts = pd.Series(sc_labels).value_counts().to_dict()

        # Add nodes (SCs) with cell counts - only include filtered SCs
        for sc, count in sc_counts.items():
            if sc != 'Other' and (filtered_scs is None or sc in filtered_scs):
                G.add_node(sc, size=count)

        # Count interactions between SCs with hierarchical connection logic
        interaction_counts = {}

        for i in range(len(sc_labels)):
            sc_i = sc_labels[i]
            if sc_i == 'Other' or (filtered_scs is not None and sc_i not in filtered_scs):
                continue

            # Get neighbors
            neighbors = connectivity[i].toarray().flatten() > 0
            neighbor_scs = sc_labels[neighbors]

            # Count interactions with other SCs
            for sc_j in neighbor_scs:
                if sc_j == 'Other' or sc_j == sc_i or (filtered_scs is not None and sc_j not in filtered_scs):
                    continue

                # Apply hierarchical connection logic
                if self._should_connect_hierarchically(sc_i, sc_j):
                    # Create edge key (sorted to avoid duplicates)
                    edge = tuple(sorted([sc_i, sc_j]))

                    if edge not in interaction_counts:
                        interaction_counts[edge] = 0
                    interaction_counts[edge] += 1

        # Add edges with weights
        for (sc1, sc2), count in interaction_counts.items():
            G.add_edge(sc1, sc2, weight=count)

        self.sc_graph = G

        print(f"  - Graph nodes (SCs): {G.number_of_nodes()}")
        print(f"  - Graph edges (interactions): {G.number_of_edges()}")

        return G

    def plot_sc_graph(
        self,
        sc_key: str = 'spatial_context_filtered',
        layout: str = 'hierarchical',
        figsize: Tuple[int, int] = (16, 10),
        node_size_scale: float = 1.0,
        edge_width_scale: float = 0.002,
        save_path: Optional[str] = None
    ):
        """
        Visualize SC interaction graph with hierarchical row-based layout.
        Each row represents SCs with the same number of CNs.

        Parameters:
        -----------
        sc_key : str
            Key in adata.obs containing SC labels
        layout : str, default='hierarchical'
            Graph layout algorithm (recommended: 'hierarchical')
        figsize : tuple
            Figure size (width, height)
        node_size_scale : float
            Scaling factor for node sizes
        edge_width_scale : float
            Scaling factor for edge widths
        save_path : str, optional
            Path to save figure
        """
        print("Visualizing SC interaction graph...")

        # Build graph if not already done
        if self.sc_graph is None:
            self.build_sc_interaction_graph(sc_key=sc_key)

        G = self.sc_graph

        # Always use hierarchical layout for clean row-based organization
        # Group nodes by number of CNs
        layers = {}
        node_to_layer = {}  # Track which layer each node belongs to
        for node in G.nodes():
            n_cns = len(node.split('_'))
            if n_cns not in layers:
                layers[n_cns] = []
            layers[n_cns].append(node)
            node_to_layer[node] = n_cns
        
        # Sort nodes within each layer alphabetically
        for n_cns in layers:
            layers[n_cns] = sorted(layers[n_cns])
        
        # Create hierarchical positions with proper spacing
        pos = {}
        n_layers = len(layers)
        
        # Calculate Y positions (evenly spaced from top to bottom)
        if n_layers == 1:
            y_positions = [0.5]
        else:
            # Space rows evenly from 0.9 (top) to 0.1 (bottom)
            y_positions = [0.9 - i * (0.8 / (n_layers - 1)) for i in range(n_layers)]
        
        for layer_idx, (n_cns, nodes) in enumerate(sorted(layers.items())):
            # Y position from top to bottom
            y = y_positions[layer_idx]
            
            # X positions with even spacing
            n_nodes = len(nodes)
            if n_nodes == 1:
                x_positions = [0.5]
            else:
                # Spread nodes evenly across the width with margins
                margin = 0.08
                x_spacing = (1.0 - 2 * margin) / (n_nodes - 1) if n_nodes > 1 else 0
                x_positions = [margin + i * x_spacing for i in range(n_nodes)]
            
            for node_idx, node in enumerate(nodes):
                pos[node] = (x_positions[node_idx], y)

        # Get node sizes and normalize
        node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
        max_size = max(node_sizes) if node_sizes else 1
        min_size = min(node_sizes) if node_sizes else 1
        
        # Create size mapping for each node
        node_size_dict = {}
        for node in G.nodes():
            size = G.nodes[node]['size']
            if max_size > min_size:
                normalized = (size - min_size) / (max_size - min_size)
                scaled_size = 500 + normalized * 2000  # Range: 500-2500
            else:
                scaled_size = 1500
            node_size_dict[node] = scaled_size * node_size_scale
        
        scaled_sizes = [node_size_dict[node] for node in G.nodes()]
        
        # Create color map based on cell counts
        node_colors = []
        for node in G.nodes():
            size = G.nodes[node]['size']
            if max_size > min_size:
                norm_value = (size - min_size) / (max_size - min_size)
            else:
                norm_value = 0.5
            node_colors.append(norm_value)
        
        # Prepare edge widths based on interaction counts
        edge_weights = [G.edges[edge]['weight'] for edge in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        min_weight = min(edge_weights) if edge_weights else 1
        
        edge_widths = []
        for weight in edge_weights:
            if max_weight > min_weight:
                normalized = (weight - min_weight) / (max_weight - min_weight)
                edge_width = 1.5 + normalized * 10.5  # Range: 1.5-12 (3x larger)
            else:
                edge_width = 6.0  # 3x larger than 2.0
            edge_widths.append(edge_width)

        # Create figure with white background
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')

        # Draw all edges (hierarchical filtering already applied during graph building)
        if G.edges():
            nx.draw_networkx_edges(
                G, pos,
                width=edge_widths,
                alpha=0.3,
                edge_color='black',
                ax=ax
            )
        
        print(f"  - Edges drawn: {len(G.edges())} (hierarchical subset connections only)")

        # Draw nodes with color gradient
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=scaled_sizes,
            node_color=node_colors,
            cmap='viridis',
            vmin=0,
            vmax=1,
            alpha=0.85,
            edgecolors='black',
            linewidths=2.0,
            ax=ax
        )

        # Draw labels OUTSIDE the nodes, positioned above
        for node in G.nodes():
            x, y = pos[node]
            # Calculate offset based on node size
            node_size_pts = node_size_dict[node]
            # Convert node size to data coordinates (approximate)
            offset = 0.08  # Fixed offset above node
            
            ax.text(
                x, y + offset,
                node,
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor='black',
                    linewidth=0.8,
                    alpha=0.9
                )
            )

        # Add colorbar for node colors
        sm = plt.cm.ScalarMappable(
            cmap='viridis',
            norm=plt.Normalize(vmin=min_size, vmax=max_size)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('n_cells', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

        # Add legend for node groups (number of CNs)
        unique_n_cns = sorted(set(len(node.split('_')) for node in G.nodes()))
        legend_elements = []
        for n_cn in unique_n_cns:
            # Size for legend proportional to actual sizes
            if n_cn == 1:
                legend_size = 50
            elif n_cn == 2:
                legend_size = 80
            elif n_cn == 3:
                legend_size = 110
            else:
                legend_size = 140
            
            legend_elements.append(
                plt.scatter([], [], s=legend_size, c='gray', 
                           edgecolors='black', linewidths=1.5,
                           label=str(n_cn))
            )
        
        legend = ax.legend(
            handles=legend_elements,
            title='n_group',
            loc='upper right',
            frameon=True,
            fancybox=True,
            shadow=True,
            title_fontsize=11,
            fontsize=10
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)

        ax.axis('off')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, 1.15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  - Saved to: {save_path}")

        plt.close()

        return fig

    def compute_sc_statistics(
        self,
        sc_key: str = 'spatial_context_filtered',
        cn_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute statistics for each spatial context.

        Parameters:
        -----------
        sc_key : str
            Key in adata.obs containing SC labels
        cn_key : str, optional
            Key in adata.obs containing CN labels (for composition analysis)

        Returns:
        --------
        stats_df : DataFrame
            Statistics for each SC
        """
        print("Computing SC statistics...")

        if cn_key is None:
            cn_key = self.cn_key

        # Get SC labels
        sc_labels = self.adata.obs[sc_key]
        unique_scs = [sc for sc in sc_labels.cat.categories if sc != 'Other']

        stats = []

        for sc in unique_scs:
            sc_mask = sc_labels == sc
            n_cells = sc_mask.sum()

            # CN composition
            cn_composition = self.adata.obs[cn_key][sc_mask].value_counts()
            cn_composition_str = ', '.join([f'CN{cn}:{count}'
                                           for cn, count in cn_composition.items()])

            # Dominant CN
            dominant_cn = cn_composition.idxmax() if len(cn_composition) > 0 else None

            stats.append({
                'SC': sc,
                'n_cells': n_cells,
                'dominant_CN': dominant_cn,
                'CN_composition': cn_composition_str,
                'n_CNs': len([x for x in sc.split('_')])
            })

        stats_df = pd.DataFrame(stats).sort_values('n_cells', ascending=False)

        print(f"  - Computed statistics for {len(stats_df)} SCs")

        return stats_df

    def run_full_pipeline(
        self,
        k: int = 20,
        threshold: float = 0.9,
        min_fraction: float = 0.05,
        min_cells: int = 100,
        min_groups: Optional[int] = None,
        group_key: Optional[str] = None,
        img_id_key: str = 'tile_name',
        output_dir: str = 'spatial_contexts',
        graph_layout: str = 'spring'
    ):
        """
        Run the complete spatial context detection pipeline.

        Parameters:
        -----------
        k : int, default=40
            Number of nearest neighbors for SC detection
        threshold : float, default=0.9
            Cumulative CN fraction threshold for SC assignment
        min_fraction : float, default=0.05
            Minimum individual CN fraction to be included
        min_cells : int, default=100
            Minimum cells for SC to be retained
        min_groups : int, optional
            Minimum groups (e.g., tiles) for SC to be retained
        group_key : str, optional
            Key in adata.obs for group identifiers
        img_id_key : str
            Key in adata.obs containing image identifiers
        output_dir : str
            Directory to save results
        graph_layout : str
            Layout algorithm for SC graph ('spring', 'kamada_kawai', 'circular')
        """
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("SPATIAL CONTEXT DETECTION PIPELINE")
        print("=" * 60)

        # Step 1: Build larger KNN graph
        self.build_knn_graph(k=k)

        # Step 2: Aggregate CN fractions
        self.aggregate_cn_fractions()

        # Step 3: Detect spatial contexts
        self.detect_spatial_contexts(threshold=threshold, min_fraction=min_fraction)

        # Step 4: Prepare SC labels (no filtering applied to data)
        self.filter_spatial_contexts(
            min_cells=min_cells,
            min_groups=min_groups,
            group_key=group_key
        )

        # Step 5: Compute statistics (all SCs included)
        stats_df = self.compute_sc_statistics()
        stats_df.to_csv(f'{output_dir}/sc_statistics.csv', index=False)
        print(f"\n  - Saved statistics to: {output_dir}/sc_statistics.csv")

        # Step 6: Get filtered SCs for graph visualization only
        filtered_scs = self.get_filtered_scs(
            min_cells=min_cells,
            min_groups=min_groups,
            group_key=group_key
        )

        # Step 7: Visualizations
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        # Spatial visualization (filtered SCs only)
        self.visualize_spatial_contexts(
            img_id_key=img_id_key,
            save_path=f'{output_dir}/spatial_contexts.png',
            filtered_scs=filtered_scs
        )

        # SC interaction graph (filtered SCs only)
        self.build_sc_interaction_graph(filtered_scs=filtered_scs)
        self.plot_sc_graph(
            layout=graph_layout,
            save_path=f'{output_dir}/sc_interaction_graph.png'
        )

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Results saved to: {output_dir}/")

        return self


def main():
    """
    Main function to run spatial context detection on unified CN results.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Spatial Context Detection for Unified CN Results'
    )
    parser.add_argument(
        '--cn_results_dir', '-c',
        default='cn_unified_results',
        help='Directory containing cn_unified_kmeans.py results'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='sc_unified_results',
        help='Output directory for spatial context results'
    )
    parser.add_argument(
        '--k', type=int, default=40,
        help='Number of nearest neighbors for SC detection (default: 40)'
    )
    parser.add_argument(
        '--threshold', '-t', type=float, default=0.9,
        help='Cumulative CN fraction threshold for SC assignment (default: 0.9)'
    )
    parser.add_argument(
        '--min_fraction', '-f', type=float, default=0.1,
        help='Minimum individual CN fraction to be included (default: 0.05)'
    )
    parser.add_argument(
        '--min_cells', '-m', type=int, default=100,
        help='Minimum cells for SC to be retained (default: 100)'
    )
    parser.add_argument(
        '--min_groups', type=int, default=1,
        help='Minimum number of tiles an SC must appear in (default: 3)'
    )
    parser.add_argument(
        '--coord_offset', action='store_true',
        help='Offset spatial coordinates between tiles for visualization'
    )
    parser.add_argument(
        '--graph_layout', '-g',
        choices=['spring', 'kamada_kawai', 'circular'],
        default='spring',
        help='Layout algorithm for SC interaction graph (default: spring)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SPATIAL CONTEXT DETECTION FOR UNIFIED CN RESULTS")
    print("=" * 80)
    print(f"CN results directory: {args.cn_results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parameters: k={args.k}, threshold={args.threshold}, min_fraction={args.min_fraction}, min_cells={args.min_cells}, min_groups={args.min_groups}")
    print("=" * 80)

    try:
        # Discover processed files
        file_paths = discover_unified_cn_files(args.cn_results_dir)
        
        if not file_paths:
            print("No processed h5ad files found! Please run cn_unified_kmeans.py first.")
            return

        # Load and combine data
        combined_adata = load_unified_cn_data(file_paths, coord_offset=args.coord_offset)

        # Initialize detector
        detector = SpatialContextDetector(combined_adata, cn_key='cn_celltype')

        # Run full pipeline
        detector.run_full_pipeline(
            k=args.k,
            threshold=args.threshold,
            min_fraction=args.min_fraction,
            min_cells=args.min_cells,
            min_groups=args.min_groups,
            group_key='tile_name',
            img_id_key='tile_name',
            output_dir=args.output_dir,
            graph_layout=args.graph_layout
        )

        print(f"\nSpatial context detection completed successfully!")
        print(f"Check the results in: {args.output_dir}/")

    except Exception as e:
        print(f"Error in spatial context detection: {str(e)}")
        return


if __name__ == '__main__':
    main()
    