# cn_kmeans_tiled.py
"""
Cellular Neighborhood Detection using K-Means Clustering

This module detects cellular neighborhoods (CNs) by clustering cells based on their
local neighborhood composition, not their own cell type. The method follows the approach
from Schürch et al. (2020) and works in three steps:

1. BUILD SPATIAL GRAPH: 
   - Constructs a k-nearest neighbor (KNN) graph based on spatial coordinates
   - Each cell is connected to its k nearest spatial neighbors (default k=20)
   
2. AGGREGATE NEIGHBOR COMPOSITION:
   - For each cell, computes the fraction of each cell type in its local neighborhood
   - Creates a feature vector representing the neighborhood composition
   - Example: [0.3 T_cells, 0.5 B_cells, 0.1 Macrophages, 0.1 Epithelial, ...]
   
3. K-MEANS CLUSTERING:
   - Clusters cells based on their neighborhood composition vectors
   - Cells with similar neighborhood compositions are grouped into the same CN
   - The number of CNs is specified (default n_clusters=6)
   
KEY INSIGHT: Two cells of different types can be in the same CN if they have similar
neighborhood compositions. This captures spatial organization patterns where certain
cell type combinations co-occur locally.

Reference:
Schürch et al. (2020) "Coordinated cellular neighborhoods orchestrate antitumoral 
immunity at the colorectal cancer invasive front", Cell
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
import os
from pathlib import Path
# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class CellularNeighborhoodDetector:
    """
    Detects cellular neighborhoods (CNs) based on cell phenotype composition
    in local neighborhoods, similar to Figure 18 from the paper.

    Reference:
    Schürch et al. (2020) "Coordinated cellular neighborhoods orchestrate
    antitumoral immunity at the colorectal cancer invasive front"
    """

    def __init__(self, adata: ad.AnnData):
        """
        Initialize CN detector.

        Parameters:
        -----------
        adata : AnnData
            AnnData object with spatial coordinates and cell type annotations
        """
        self.adata = adata
        self.cn_labels = None
        self.aggregated_neighbors = None

    def build_knn_graph(
            self,
            k: int = 20,
            coord_key: str = 'spatial',
            key_added: str = 'spatial_connectivities_knn'
    ):
        """
        Build k-nearest neighbor graph based on spatial coordinates.

        Parameters:
        -----------
        k : int, default=20
            Number of nearest neighbors to consider
        coord_key : str, default='spatial'
            Key in adata.obsm containing spatial coordinates
        key_added : str
            Key to store the connectivity matrix
        """
        print(f"Building {k}-nearest neighbor graph...")

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

    def aggregate_neighbors(
            self,
            cluster_key: str = 'celltype',
            connectivity_key: str = 'spatial_connectivities_knn',
            output_key: str = 'aggregated_neighbors'
    ):
        """
        For each cell, compute the fraction of each cell phenotype in its neighborhood.

        Parameters:
        -----------
        cluster_key : str, default='celltype'
            Key in adata.obs containing cell type labels
        connectivity_key : str
            Key in adata.obsp containing spatial connectivity
        output_key : str
            Key to store aggregated neighbor fractions
        """
        print(f"Aggregating neighbors by {cluster_key}...")

        # Get cell types
        cell_types = self.adata.obs[cluster_key].values
        unique_types = self.adata.obs[cluster_key].cat.categories

        # Get connectivity matrix
        connectivity = self.adata.obsp[connectivity_key]

        # Initialize aggregated neighbors matrix
        n_cells = self.adata.n_obs
        n_types = len(unique_types)
        aggregated = np.zeros((n_cells, n_types))

        # For each cell, compute cell type fractions in neighborhood
        for i in range(n_cells):
            # Get neighbors (including self)
            neighbors_mask = connectivity[i].toarray().flatten() > 0

            if neighbors_mask.sum() > 0:
                # Get cell types of neighbors
                neighbor_types = cell_types[neighbors_mask]

                # Compute fractions
                for j, ct in enumerate(unique_types):
                    aggregated[i, j] = (neighbor_types == ct).sum() / neighbors_mask.sum()

        # Store in adata
        self.aggregated_neighbors = pd.DataFrame(
            aggregated,
            columns=unique_types,
            index=self.adata.obs_names
        )

        self.adata.obsm[output_key] = aggregated

        print(f"  - Aggregated neighbor fractions shape: {aggregated.shape}")

        return self

    def detect_cellular_neighborhoods(
            self,
            n_clusters: int = 6,    # was 6
            random_state: int = 220705,
            aggregated_key: str = 'aggregated_neighbors',
            output_key: str = 'cn_celltype'
    ):
        """
        Cluster cells based on their neighborhood composition using k-means.

        Parameters:
        -----------
        n_clusters : int, default=6
            Number of cellular neighborhoods to detect
        random_state : int
            Random seed for reproducibility
        aggregated_key : str
            Key in adata.obsm containing aggregated neighbor fractions
        output_key : str
            Key to store CN labels in adata.obs
        """
        print(f"Detecting {n_clusters} cellular neighborhoods using k-means...")

        # Get aggregated neighbor fractions
        aggregated = self.adata.obsm[aggregated_key]

        # Perform k-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

        cn_labels = kmeans.fit_predict(aggregated)

        # Store CN labels (add 1 to match R's 1-based indexing)
        self.cn_labels = cn_labels + 1
        self.adata.obs[output_key] = pd.Categorical(self.cn_labels)

        # Print CN sizes
        cn_counts = pd.Series(self.cn_labels).value_counts().sort_index()
        print(f"  - CN sizes:")
        for cn, count in cn_counts.items():
            print(f"    CN {cn}: {count} cells")

        return self

    def compute_cn_composition(
            self,
            cn_key: str = 'cn_celltype',
            celltype_key: str = 'celltype'
    ) -> pd.DataFrame:
        """
        Compute cell phenotype fractions in each CN.

        Parameters:
        -----------
        cn_key : str
            Key in adata.obs containing CN labels
        celltype_key : str
            Key in adata.obs containing cell type labels

        Returns:
        --------
        composition : DataFrame
            CN composition matrix (rows=CNs, columns=cell types)
        """
        print("Computing CN composition...")

        # Create contingency table
        composition = pd.crosstab(
            self.adata.obs[cn_key],
            self.adata.obs[celltype_key],
            normalize='index'  # Normalize by CN (rows)
        )

        return composition

    def visualize_spatial_cns(
            self,
            cn_key: str = 'cn_celltype',
            img_id_key: str = 'sample_id',
            coord_key: str = 'spatial',
            point_size: float = 0.5,
            palette: str = 'Set3',
            figsize: Optional[Tuple[int, int]] = None,
            save_path: Optional[str] = None,
            k: Optional[int] = None,
            n_clusters: Optional[int] = None,
            eps: Optional[float] = None,
            min_samples: Optional[int] = None
    ):
        """
        Visualize cellular neighborhoods spatially (similar to Fig 18a).

        Parameters:
        -----------
        cn_key : str
            Key in adata.obs containing CN labels
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
        print("Visualizing spatial CNs...")

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

        # Get CN labels
        cn_labels = self.adata.obs[cn_key].values

        # Get colors
        n_cns = len(self.adata.obs[cn_key].cat.categories)
        colors_palette = sns.color_palette(palette, n_cns)

        # Plot each image
        for idx, img in enumerate(images):
            ax = axes[idx]

            # Get cells from this image
            mask = self.adata.obs[img_id_key] == img
            img_coords = coords[mask]
            img_cns = cn_labels[mask]

            # Plot each CN
            for cn_id in np.unique(img_cns):
                cn_mask = img_cns == cn_id
                cn_idx = int(cn_id) - 1  # Convert to 0-based for color indexing

                ax.scatter(
                    img_coords[cn_mask, 0],
                    img_coords[cn_mask, 1],
                    c=[colors_palette[cn_idx]],
                    s=point_size,
                    alpha=0.7,
                    label=f'CN {cn_id}'
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

        # Add overall title with parameters if provided
        if k is not None and n_clusters is not None:
            fig.suptitle(f'Cellular Neighborhoods (k={k}, n_clusters={n_clusters})', 
                         fontsize=16, fontweight='bold', y=0.98)
        elif k is not None and eps is not None and min_samples is not None:
            fig.suptitle(f'Cellular Neighborhoods (k={k}, eps={eps}, min_samples={min_samples})', 
                         fontsize=16, fontweight='bold', y=0.98)
        elif k is not None:
            fig.suptitle(f'Cellular Neighborhoods (k={k})', 
                         fontsize=16, fontweight='bold', y=0.98)
        elif n_clusters is not None:
            fig.suptitle(f'Cellular Neighborhoods (n_clusters={n_clusters})', 
                         fontsize=16, fontweight='bold', y=0.98)
        elif eps is not None and min_samples is not None:
            fig.suptitle(f'Cellular Neighborhoods (eps={eps}, min_samples={min_samples})', 
                         fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to: {save_path}")

        plt.show()

        return fig

    def visualize_cn_composition(
            self,
            cn_key: str = 'cn_celltype',
            celltype_key: str = 'celltype',
            figsize: Tuple[int, int] = (10, 6),
            cmap: str = 'coolwarm',
            vmin: float = -2,
            vmax: float = 2,
            save_path: Optional[str] = None,
            k: Optional[int] = None,
            n_clusters: Optional[int] = None,
            eps: Optional[float] = None,
            min_samples: Optional[int] = None,
            show_values: bool = True
    ):
        """
        Visualize CN composition as heatmap (similar to Fig 18b).

        Parameters:
        -----------
        cn_key : str
            Key in adata.obs containing CN labels
        celltype_key : str
            Key in adata.obs containing cell type labels
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap name
        vmin, vmax : float
            Color scale limits for z-scores
        save_path : str, optional
            Path to save figure
        """
        print("Visualizing CN composition heatmap...")

        # Compute composition
        composition = self.compute_cn_composition(cn_key, celltype_key)

        # Z-score by column (cell type)
        composition_zscore = composition.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        # Create figure
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
            fmt='.2f' if show_values else False,
            annot_kws={'size': 8}
        )

        ax.set_xlabel('Cell Type', fontsize=12)
        ax.set_ylabel('Cellular Neighborhood', fontsize=12)
        
        # Create title with parameters if provided
        title = 'Cell Type Composition by Cellular Neighborhood\n(Z-score scaled by column)'
        if k is not None and n_clusters is not None:
            title = f'Cell Type Composition by Cellular Neighborhood (k={k}, n_clusters={n_clusters})\n(Z-score scaled by column)'
        elif k is not None and eps is not None and min_samples is not None:
            title = f'Cell Type Composition by Cellular Neighborhood (k={k}, eps={eps}, min_samples={min_samples})\n(Z-score scaled by column)'
        elif k is not None:
            title = f'Cell Type Composition by Cellular Neighborhood (k={k})\n(Z-score scaled by column)'
        elif n_clusters is not None:
            title = f'Cell Type Composition by Cellular Neighborhood (n_clusters={n_clusters})\n(Z-score scaled by column)'
        elif eps is not None and min_samples is not None:
            title = f'Cell Type Composition by Cellular Neighborhood (eps={eps}, min_samples={min_samples})\n(Z-score scaled by column)'
            
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to: {save_path}")

        plt.show()

        return fig, composition, composition_zscore

    def run_full_pipeline(
            self,
            k: int = 20,
            n_clusters: int = 6,
            celltype_key: str = 'celltype',
            img_id_key: str = 'sample_id',
            output_dir: str = 'cn_results',
            random_state: int = 220705,
            save_adata: bool = False
    ):
        """
        Run the complete CN detection pipeline.

        Parameters:
        -----------
        k : int, default=20
            Number of nearest neighbors
        n_clusters : int, default=6
            Number of CNs to detect
        celltype_key : str
            Key in adata.obs containing cell type labels
        img_id_key : str
            Key in adata.obs containing image identifiers
        output_dir : str
            Directory to save results
        random_state : int
            Random seed
        save_adata : bool, default=False
            Whether to save the processed AnnData object with CN annotations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("CELLULAR NEIGHBORHOOD DETECTION PIPELINE")
        print("=" * 60)

        # Step 1: Build KNN graph
        self.build_knn_graph(k=k)

        # Step 2: Aggregate neighbors
        self.aggregate_neighbors(cluster_key=celltype_key)

        # Step 3: Detect CNs
        self.detect_cellular_neighborhoods(
            n_clusters=n_clusters,
            random_state=random_state
        )

        # Step 4: Visualize spatial CNs
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        self.visualize_spatial_cns(
            img_id_key=img_id_key,
            save_path=f'{output_dir}/spatial_cns.png',
            k=k,
            n_clusters=n_clusters
        )

        # Step 5: Visualize CN composition
        fig, composition, composition_zscore = self.visualize_cn_composition(
            celltype_key=celltype_key,
            save_path=f'{output_dir}/cn_composition_heatmap.png',
            k=k,
            n_clusters=n_clusters,
            show_values=True
        )

        # Step 6: Save results
        composition.to_csv(f'{output_dir}/cn_composition.csv')
        composition_zscore.to_csv(f'{output_dir}/cn_composition_zscore.csv')

        # Save processed data (optional)
        if save_adata:
            input_basename = Path(output_dir).stem if hasattr(output_dir, 'stem') else 'data'
            output_filename = f'{input_basename}_adata_cns.h5ad'
            output_path = os.path.join(output_dir, output_filename)
            self.adata.write(output_path)
            print(f"\n  - Saved processed AnnData to: {output_path}")
        else:
            print(f"\n  - AnnData not saved (set save_adata=True to save)")

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Results saved to: {output_dir}/")

        return self


# Example usage function
def main():
    """
    Example usage of CellularNeighborhoodDetector.
    """
    # Input file configuration
    input_file = '../tile_39520_7904.h5ad'
    output_dir = 'cn'

    # Load data
    print("Loading data...")
    adata = sc.read_h5ad(input_file)

    # Initialize detector
    detector = CellularNeighborhoodDetector(adata)

    # Run full pipeline
    detector.run_full_pipeline(
        k=20,
        n_clusters=6,   # was 6
        celltype_key='cell_type',  # Adjust to your column name
        img_id_key='tile_name',  # Adjust to your column name
        output_dir=output_dir,
        random_state=220705,
        save_adata=False  # Set to True to save the h5ad file
    )

    print("\nYou can now explore the results:")
    print("  - Check 'cellular_neighborhoods/' folder for figures")
    print("  - Set save_adata=True if you want to save the h5ad file with CN annotations")


if __name__ == '__main__':
    main()