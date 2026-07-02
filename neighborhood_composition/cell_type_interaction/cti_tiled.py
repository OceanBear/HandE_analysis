import squidpy as sq
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)

# from dask.tests.test_config import no_read_permissions
# from networkx.algorithms.distance_measures import radius


def load_and_apply_cell_type_colors(adata, celltype_key='cell_type'):
    """Convert hex colors from h5ad to matplotlib RGB tuples."""
    # Colors already saved by data_preparation.py in hex format
    if f'{celltype_key}_colors' in adata.uns:
        colors = adata.uns[f'{celltype_key}_colors']
        # Convert hex strings to RGB tuples for matplotlib
        if isinstance(colors[0], str):
            adata.uns[f'{celltype_key}_colors'] = [
                tuple(int(c.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4))
                for c in colors
            ]
        print(f"  - Using cell type colors from h5ad file")
    else:
        print("  - Warning: Colors not found in h5ad file")


def build_spatial_graph(adata, method='radius', radius=50, n_neighbors=20, coord_type='generic'):
    """
    Build spatial neighborhood graph for cells.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial coordinates
    method : str, default='radius'
        Method to build graph: 'radius' or 'knn'
        - 'radius': All cells within radius distance
        - 'knn': K nearest neighbors
    radius : float, default=50
        Radius for neighborhood definition (in pixels)
        Typical cell diameter is 10-30 pixels, so 50 captures immediate neighbors
    n_neighbors : int, default=6
        Number of neighbors for KNN method
    coord_type : str, default='generic'
        Coordinate type for Squidpy

    Returns:
    --------
    adata : AnnData
        Modified in place with spatial graph added
    """

    print(f"Building spatial graph using {method} method...")

    if method == 'radius':
        sq.gr.spatial_neighbors(
            adata,
            spatial_key='spatial',
            coord_type=coord_type,
            radius=radius,
            n_rings=1
            # n_neighs is omitted - radius method finds all neighbors within radius
        )
        print(f"  - Using radius: {radius} pixels")

    elif method == 'knn':
        sq.gr.spatial_neighbors(
            adata,
            spatial_key='spatial',
            coord_type=coord_type,
            n_neighs=n_neighbors,
            radius=None
        )
        print(f"  - Using K-nearest neighbors: {n_neighbors}")

    # Print connectivity statistics
    connectivity = adata.obsp['spatial_connectivities']
    avg_neighbors = connectivity.sum(axis=1).mean()
    print(f"  - Average neighbors per cell: {avg_neighbors:.2f}")
    print(f"  - Connectivity matrix shape: {connectivity.shape}")

    return adata


def _extract_nhood_enrichment_pvalues(adata, cluster_key: str):
    """Return p-value matrix from Squidpy uns if present (API varies by version), else None."""
    key = f"{cluster_key}_nhood_enrichment"
    if key not in adata.uns:
        return None
    d = adata.uns[key]
    if not isinstance(d, dict):
        return None
    for pk in ("pvalues", "pvalue", "pvals", "pvalues_adj", "p_adj"):
        if pk not in d:
            continue
        p = d[pk]
        if isinstance(p, pd.DataFrame):
            return p.values.astype(float)
        return np.asarray(p, dtype=float)
    return None


def compute_sigval_matrix(
    z,
    *,
    p_values=None,
    method: str = "p_from_z",
    alpha: float = 0.05,
    z_threshold: float = 2.0,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """
    Schapiro-style signed significance in {-1, 0, +1} per cell-type pair.

    +1: interaction (enrichment vs null), -1: avoidance (depletion), 0: not significant.
    When Squidpy stores empirical p-values in ``adata.uns``, pass ``p_values``; otherwise
    ``p_from_z`` uses a two-sided normal approximation p = 2*Phi(-|z|) (not identical to
    imcRtools permutation p-values but similar use). ``z_threshold`` applies if
    ``method == 'z_threshold'``.

    Non-finite z (missing pairs after alignment) -> 0.
    """
    z = np.asarray(z, dtype=float)
    sig = np.zeros(z.shape, dtype=np.int8)
    finite = np.isfinite(z)

    if p_values is not None and np.shape(p_values) == z.shape:
        p = np.asarray(p_values, dtype=float)
        hit = finite & np.isfinite(p) & (p <= alpha)
        sig = np.where(hit & (z > 0), 1, sig)
        sig = np.where(hit & (z < 0), -1, sig)
    elif method == "z_threshold":
        hit = finite & (np.abs(z) >= z_threshold)
        sig = np.where(hit & (z > 0), 1, sig)
        sig = np.where(hit & (z < 0), -1, sig)
    elif method == "p_from_z":
        try:
            from scipy.stats import norm
        except ImportError as e:
            raise ImportError(
                "compute_sigval_matrix(method='p_from_z') requires scipy "
                "(install scipy or use method='z_threshold')."
            ) from e
        p = 2.0 * norm.sf(np.abs(z))
        hit = finite & np.isfinite(p) & (p <= alpha)
        sig = np.where(hit & (z > 0), 1, sig)
        sig = np.where(hit & (z < 0), -1, sig)
    else:
        raise ValueError(f"Unknown sigval method: {method!r}")

    if zero_diagonal and sig.shape[0] == sig.shape[1]:
        np.fill_diagonal(sig, 0)
    return sig


def cell_type_interaction_analysis(adata, cluster_key='cell_type', n_perms=1000, seed=42):
    """
    Compute cell type interaction (CTI) scores via Squidpy neighborhood enrichment.

    Wraps ``squidpy.gr.nhood_enrichment``; results are stored under the library key
    ``adata.uns['{cluster_key}_nhood_enrichment']`` (unchanged for AnnData compatibility).

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial graph
    cluster_key : str, default='cell_type'
        Key in adata.obs containing cell type labels
    n_perms : int, default=1000
        Number of permutations for statistical testing
    seed : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    adata : AnnData
        Modified in place with CTI (nhood enrichment) results in ``adata.uns``
    """

    print(f"\nComputing cell type interaction (CTI) analysis...")
    print(f"  - Cluster key: {cluster_key}")
    print(f"  - Number of permutations: {n_perms}")

    sq.gr.nhood_enrichment(
        adata,
        cluster_key=cluster_key,
        n_perms=n_perms,
        seed=seed
    )

    print(f"  - CTI analysis complete!")
    print(f"  - Results stored in adata.uns['{cluster_key}_nhood_enrichment'] (Squidpy)")

    return adata


def compute_co_occurrence(adata, cluster_key='cell_type', n_splits=20):
    """
    Compute cell type co-occurrence scores.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial coordinates
    cluster_key : str, default='cell_type'
        Key in adata.obs containing cell type labels
    n_splits : int, default=20
        Number of splits for interval computation

    Returns:
    --------
    adata : AnnData
        Modified in place with co-occurrence results
    """

    print(f"\nComputing co-occurrence analysis...")

    sq.gr.co_occurrence(
        adata,
        cluster_key=cluster_key,
        spatial_key='spatial',  # Modify to knn for large files
        n_splits=n_splits,
    )

    print(f"  - Co-occurrence analysis complete!")
    print(f"  - Results stored in adata.uns['{cluster_key}_co_occurrence']")

    return adata


def compute_centrality_scores(adata, cluster_key='cell_type'):
    """
    Compute network centrality scores for each cell type.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial graph
    cluster_key : str, default='cell_type'
        Key in adata.obs containing cell type labels

    Returns:
    --------
    adata : AnnData
        Modified in place with centrality scores
    """

    print(f"\nComputing centrality scores...")

    sq.gr.centrality_scores(
        adata,
        cluster_key=cluster_key
    )

    # Print summary of centrality scores
    centrality_cols = [col for col in adata.obs.columns if 'centrality' in col]
    print(f"  - Computed centrality metrics: {centrality_cols}")

    return adata


def visualize_cell_type_interaction(adata, cluster_key='cell_type', figsize=(10, 8), save_path=None, radius=None, n_neighbors=None, n_perms=None):
    """
    Visualize cell type interaction (CTI) as a heatmap of Squidpy permutation z-scores.

    One tile / one ``AnnData``: shows the **raw** z matrix from ``nhood_enrichment`` for that
    object (not averaged across images). Colorbar label: ``Z-score`` (permutation-null scale).

    Parameters:
    -----------
    adata : AnnData
        AnnData object with CTI (nhood enrichment) results
    cluster_key : str, default='cell_type'
        Key for cell type labels
    figsize : tuple, default=(10, 8)
        Figure size
    save_path : str, optional
        Path to save figure
    radius : float, optional
        Radius used for spatial graph (displayed in title)
    n_neighbors : int, optional
        Number of neighbors for KNN method (displayed in title)
    n_perms : int, optional
        Number of permutations used (displayed in title)
    """

    print(f"\nVisualizing cell type interaction (CTI) heatmap...")

    # Get z-scores to calculate dynamic scale
    zscore = adata.uns[f'{cluster_key}_nhood_enrichment']['zscore']
    if isinstance(zscore, pd.DataFrame):
        zscore_array = zscore.values
    else:
        zscore_array = np.array(zscore)

    # Calculate symmetric scale based on actual data range
    max_abs_z = np.abs(zscore_array).max()
    vmin, vmax = -max_abs_z, max_abs_z

    print(f"  - Z-score range: [{zscore_array.min():.2f}, {zscore_array.max():.2f}]")
    print(f"  - Color scale: [{vmin:.2f}, {vmax:.2f}]")

    fig, ax = plt.subplots(figsize=figsize)
    max_abs_value = max(abs(vmin), abs(vmax))

    # Get cell type names for labels
    cell_types = adata.obs[cluster_key].cat.categories.tolist()

    # Use seaborn directly to show annotations
    sns.heatmap(
        zscore,
        cmap='coolwarm',
        center=0,
        vmin=-np.ceil(max_abs_value),
        vmax=np.ceil(max_abs_value),
        annot=True,  # Show values in cells
        fmt='.2f',   # Format to 2 decimal places
        cbar_kws={'label': 'Z-score'},
        linewidths=0.5,
        linecolor='white',
        #xticklabels=cell_types,
        #yticklabels=cell_types,
        square=True,
        ax=ax
    )
    ax.set_xlabel('Cell Type', fontsize=20)
    ax.set_ylabel('Cell Type', fontsize=20)

    # Build title with optional radius/knn and n_perms (single-tile matrix, not averaged)
    title = 'Cell Type Interaction (CTI)\n(Permutation z, this tile)'
    if radius is not None or n_neighbors is not None or n_perms is not None:
        params = []
        if radius is not None:
            params.append(f'radius={radius}')
        if n_neighbors is not None:
            params.append(f'knn={n_neighbors}')
        if n_perms is not None:
            params.append(f'n_perms={n_perms}')
        title += f'\n({", ".join(params)})'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set tick labels
    ax.set_xticklabels(cell_types, rotation=0) # ha='right
    ax.set_yticklabels(cell_types, rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved to: {save_path}")

    plt.show()

    return fig


def visualize_spatial_distribution(adata, cluster_key='cell_type', figsize=(12, 10),
                                   size=3, save_path=None):
    """
    Visualize spatial distribution of cell types.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial coordinates
    cluster_key : str, default='cell_type'
        Key for cell type labels
    figsize : tuple, default=(12, 10)
        Figure size
    size : float, default=3
        Point size for scatter plot
    save_path : str, optional
        Path to save figure
    """

    print(f"\nVisualizing spatial distribution...")

    fig, ax = plt.subplots(figsize=figsize)

    # Get spatial coordinates
    coords = adata.obsm['spatial']

    # Get cell types and colors
    cell_types = adata.obs[cluster_key]

    # Use colors from adata.uns if available, otherwise use default palette
    if f'{cluster_key}_colors' in adata.uns:
        colors = adata.uns[f'{cluster_key}_colors']
        # Create a color map for each category
        unique_types = cell_types.cat.categories
        color_map = {ct: colors[i] for i, ct in enumerate(unique_types)}
    else:
        # Use default color palette
        unique_types = cell_types.cat.categories
        palette = sns.color_palette('tab10', n_colors=len(unique_types))
        color_map = {ct: palette[i] for i, ct in enumerate(unique_types)}

    # Plot each cell type separately for legend
    for cell_type in unique_types:
        mask = cell_types == cell_type
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color_map[cell_type]],
                   label=cell_type,
                   s=size,
                   alpha=0.7)

    ax.set_xlabel('X coordinate (pixels)', fontsize=12)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=12)
    ax.set_title('Spatial Distribution of Cell Types', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add text showing number of cells displayed
    text_str = f"Displaying {adata.n_obs:,} cells"
    ax.text(0.98, 0.02, text_str,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved to: {save_path}")

    plt.close()

    return fig


def summarize_interactions(adata, cluster_key='cell_type', threshold=2.0):
    """
    Summarize significant cell-cell interactions.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with CTI (nhood enrichment) z-scores
    cluster_key : str, default='cell_type'
        Key for cell type labels
    threshold : float, default=2.0
        Z-score threshold for significance

    Returns:
    --------
    interactions_df : DataFrame
        Summary of significant interactions
    """

    print(f"\nSummarizing cell-cell interactions (threshold: |z| > {threshold})...")

    zscore = adata.uns[f'{cluster_key}_nhood_enrichment']['zscore']

    # Get cell type names from the categorical data
    cell_types = adata.obs[cluster_key].cat.categories.tolist()

    # Convert to numpy array if it's not already
    if isinstance(zscore, pd.DataFrame):
        zscore_array = zscore.values
    else:
        zscore_array = np.array(zscore)

    # Find significant interactions
    interactions = []
    for i, ct1 in enumerate(cell_types):
        for j, ct2 in enumerate(cell_types):
            z = zscore_array[i, j]
            if abs(z) > threshold:
                interaction_type = "Attraction" if z > 0 else "Avoidance"
                interactions.append({
                    'Cell Type 1': ct1,
                    'Cell Type 2': ct2,
                    'Z-score': float(z),
                    'Interaction': interaction_type
                })

    interactions_df = pd.DataFrame(interactions).sort_values('Z-score',
                                                             key=abs,
                                                             ascending=False)

    print(f"  - Found {len(interactions_df)} significant interactions")
    print(f"\nTop interactions:")
    if len(interactions_df) > 0:
        print(interactions_df.head(10).to_string(index=False))
    else:
        print("  No significant interactions found above threshold")

    return interactions_df


def save_intermediate_results(
    adata,
    output_dir,
    tile_name=None,
    cluster_key='cell_type',
    *,
    sigval_method='p_from_z',
    sigval_alpha=0.05,
    sigval_z_threshold=2.0,
):
    """
    Save intermediate results for efficient aggregation later.

    Saves z-score matrix, Schapiro-style ``sigval`` in {-1, 0, +1}, and metadata.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with CTI (nhood enrichment) results
    output_dir : str or Path
        Directory to save intermediate results
    tile_name : str, optional
        Name of the tile (for prefixing files). If None, no prefix used.
    cluster_key : str, default='cell_type'
        Key for cell type labels
    sigval_method : str, default='p_from_z'
        ``p_from_z`` (two-sided normal p from Squidpy z) or ``z_threshold``.
    sigval_alpha : float, default=0.05
        Significance level for p-value rules.
    sigval_z_threshold : float, default=2.0
        Minimum |z| when ``sigval_method == 'z_threshold'``.

    Returns:
    --------
    saved_files : dict
        Dictionary with paths to saved files
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get prefix for files
    prefix = f'{tile_name}_' if tile_name else ''

    # Extract CTI (nhood enrichment) z-scores for aggregation
    zscore = adata.uns[f'{cluster_key}_nhood_enrichment']['zscore']
    if isinstance(zscore, pd.DataFrame):
        zscore_array = zscore.values
    else:
        zscore_array = np.array(zscore)

    p_arr = _extract_nhood_enrichment_pvalues(adata, cluster_key)
    sigval_array = compute_sigval_matrix(
        zscore_array,
        p_values=p_arr,
        method=sigval_method,
        alpha=sigval_alpha,
        z_threshold=sigval_z_threshold,
    )

    # Get cell types
    cell_types = adata.obs[cluster_key].cat.categories.tolist()

    # Save zscore as numpy binary (fast and compact)
    zscore_path = output_dir / f'{prefix}zscore.npy'
    np.save(zscore_path, zscore_array)

    sigval_path = output_dir / f'{prefix}sigval.npy'
    np.save(sigval_path, sigval_array)

    # Save metadata as JSON
    metadata = {
        'tile_name': tile_name,
        'n_cells': int(adata.n_obs),
        'cell_types': cell_types,
        'cluster_key': cluster_key,
        'zscore_shape': list(zscore_array.shape),
        'mean_abs_zscore': float(np.abs(zscore_array).mean()),
        'max_abs_zscore': float(np.abs(zscore_array).max()),
        'sigval_method': sigval_method,
        'sigval_alpha': float(sigval_alpha),
        'sigval_z_threshold': float(sigval_z_threshold),
        'used_squidpy_pvalues': p_arr is not None,
    }

    metadata_path = output_dir / f'{prefix}metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  - Saved intermediate results:")
    print(f"    • {zscore_path.name}")
    print(f"    • {sigval_path.name}")
    print(f"    • {metadata_path.name}")

    return {
        'zscore_path': zscore_path,
        'sigval_path': sigval_path,
        'metadata_path': metadata_path,
        'zscore': zscore_array,
        'sigval': sigval_array,
        'metadata': metadata
    }


def load_intermediate_results(output_dir, tile_name=None):
    """
    Load intermediate results saved by save_intermediate_results().

    Parameters:
    -----------
    output_dir : str or Path
        Directory containing saved results
    tile_name : str, optional
        Name of the tile (for prefixing files). If None, no prefix used.

    Returns:
    --------
    results : dict
        Dictionary containing zscore array, optional sigval (-1/0/1), and metadata
    """
    import json

    output_dir = Path(output_dir)
    prefix = f'{tile_name}_' if tile_name else ''

    # Load zscore
    zscore_path = output_dir / f'{prefix}zscore.npy'
    if not zscore_path.exists():
        raise FileNotFoundError(f"Zscore file not found: {zscore_path}")
    zscore = np.load(zscore_path)

    # Load metadata
    metadata_path = output_dir / f'{prefix}metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    sigval_path = output_dir / f'{prefix}sigval.npy'
    if sigval_path.exists():
        sigval = np.load(sigval_path)
    else:
        sigval = compute_sigval_matrix(
            zscore,
            p_values=None,
            method=metadata.get('sigval_method', 'p_from_z'),
            alpha=float(metadata.get('sigval_alpha', 0.05)),
            z_threshold=float(metadata.get('sigval_z_threshold', 2.0)),
        )

    return {
        'zscore': zscore,
        'sigval': sigval,
        'metadata': metadata,
        'cell_types': metadata['cell_types'],
        'n_cells': metadata['n_cells'],
        'tile_name': metadata.get('tile_name')
    }


def aggregate_from_saved_results(
    tile_dirs,
    output_dir,
    tile_names=None,
    n_perms=None,
    n_neighbors=None,
    *,
    merge_epithelium_to_tumor=False,
    tumor_label="Tumor",
    cti_heatmap_annot_fontsize=32,
    use_short_cell_type_labels_in_plots=True,
    cell_type_abbrev_map=None,
    schapiro_sum_sigval=True,
    sigval_method="p_from_z",
    sigval_alpha=0.05,
    sigval_z_threshold=2.0,
):
    """
    Aggregate results from multiple tiles using saved intermediate files.

    This function loads zscore matrices one at a time from disk and computes
    aggregated statistics without keeping all data in memory simultaneously.

    Parameters:
    -----------
    tile_dirs : list of str/Path
        List of directories containing intermediate results
    output_dir : str or Path
        Directory to save aggregated results
    tile_names : list of str, optional
        List of tile names corresponding to tile_dirs. If None, extracts from metadata.
    n_perms : int, optional
        Number of permutations used in CTI analysis (for display in plot title)
    n_neighbors : int, optional
        Number of neighbors used in spatial graph (for display in plot title)
    merge_epithelium_to_tumor : bool, default=False
        Merge legacy 7-class epithelium labels into ``tumor_label`` per tile before alignment.
    tumor_label : str, default='Tumor'
        Label for merged epithelium block.
    cti_heatmap_annot_fontsize : float, default=12
        Font size for annotated z-scores inside aggregated mean / std heatmaps.
    use_short_cell_type_labels_in_plots : bool, default=True
        Short axis labels from ``cti_aggregate.DEFAULT_CELL_TYPE_DISPLAY_ABBREV``.
    cell_type_abbrev_map : dict optional
        Override full-name → short label mapping for heatmap ticks only.
    schapiro_sum_sigval : bool, default=True
        If True, sum Schapiro-style ``sigval`` in {-1,0,1} per tile (from aligned z-scores)
        and save ``aggregated_summed_sigval.csv`` / ``aggregated_summed_sigval.png``.
    sigval_method : str, default='p_from_z'
        Passed to ``compute_sigval_matrix`` for each aligned tile (see that docstring).
    sigval_alpha : float, default=0.05
        Significance level for ``sigval`` when using p-values / ``p_from_z``.
    sigval_z_threshold : float, default=2.0
        |z| threshold when ``sigval_method == 'z_threshold'``.

    Returns:
    --------
    aggregated : dict
        Dictionary containing aggregated statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS FROM SAVED FILES")
    print("=" * 70)
    print(f"Loading from {len(tile_dirs)} tile directories...")

    # First pass: collect all zscores and cell types
    zscores_list = []
    metadata_list = []
    actual_tile_names = []
    all_cell_types_set = set()

    for i, tile_dir in enumerate(tile_dirs):
        tile_dir = Path(tile_dir)
        tile_name = tile_names[i] if tile_names and i < len(tile_names) else None

        try:
            results = load_intermediate_results(tile_dir, tile_name=tile_name)
            zscore = results["zscore"]
            metadata = dict(results["metadata"])
            tile_cell_types = metadata["cell_types"]

            if merge_epithelium_to_tumor:
                from cti_aggregate import merge_symmetric_celltype_zscore_df

                zscore_df = pd.DataFrame(zscore, index=tile_cell_types, columns=tile_cell_types)
                zscore_df = merge_symmetric_celltype_zscore_df(
                    zscore_df, new_label=tumor_label
                )
                zscore = zscore_df.values
                metadata["cell_types"] = list(zscore_df.index)
                tile_cell_types = metadata["cell_types"]

            zscores_list.append(zscore)
            metadata_list.append(metadata)
            actual_tile_names.append(results['tile_name'] or tile_dir.name)

            # Collect all unique cell types across all tiles
            all_cell_types_set.update(tile_cell_types)

            print(f"  [{i+1}/{len(tile_dirs)}] Loaded: {results['tile_name'] or tile_dir.name} "
                  f"({results['n_cells']} cells, {len(tile_cell_types)} cell types)")

        except Exception as e:
            print(f"  [!] Warning: Could not load {tile_dir}: {e}")
            continue

    if len(zscores_list) == 0:
        raise ValueError("No valid results found to aggregate!")

    # Create common cell type order: preserve order from first tile, append missing types
    # This maintains the same order as single-tile results (from h5ad categorical order)
    first_tile_cell_types = metadata_list[0]['cell_types']
    common_cell_types = list(first_tile_cell_types)  # Start with first tile's order
    
    # Append any cell types from other tiles that aren't in the first tile
    for metadata in metadata_list[1:]:
        for ct in metadata['cell_types']:
            if ct not in common_cell_types:
                common_cell_types.append(ct)
    
    n_common_types = len(common_cell_types)
    
    print(f"\nAligning z-score matrices to common cell type set...")
    print(f"  - Common cell types ({n_common_types}): {', '.join(common_cell_types)}")
    
    # Second pass: align each tile's z-score matrix to common cell type set
    aligned_zscores_list = []
    aligned_sig_list = []
    for i, (zscore, metadata) in enumerate(zip(zscores_list, metadata_list)):
        tile_cell_types = metadata['cell_types']
        tile_name = actual_tile_names[i]
        
        # Create DataFrame from z-score matrix with tile's cell types
        zscore_df = pd.DataFrame(zscore, index=tile_cell_types, columns=tile_cell_types)
        
        # Reindex to common cell types (missing will be NaN)
        aligned_zscore_df = zscore_df.reindex(index=common_cell_types, columns=common_cell_types)
        
        # Fill missing cell types with NaN (they don't exist in this tile)
        aligned_zscore_array = aligned_zscore_df.values
        
        aligned_zscores_list.append(aligned_zscore_array)

        if schapiro_sum_sigval:
            sig_tile = compute_sigval_matrix(
                aligned_zscore_array,
                p_values=None,
                method=sigval_method,
                alpha=sigval_alpha,
                z_threshold=sigval_z_threshold,
            )
            aligned_sig_list.append(sig_tile)
        
        missing_types = set(common_cell_types) - set(tile_cell_types)
        if missing_types:
            print(f"  - Tile {tile_name}: {len(missing_types)} missing cell types (filled with NaN)")

    # Stack and compute statistics
    print(f"\nComputing aggregated statistics...")
    zscores_array = np.stack(aligned_zscores_list)  # shape: (n_tiles, n_celltypes, n_celltypes)

    # Compute statistics (NaN values are handled correctly by numpy)
    mean_zscore = np.nanmean(zscores_array, axis=0)
    std_zscore = np.nanstd(zscores_array, axis=0)
    median_zscore = np.nanmedian(zscores_array, axis=0)
    min_zscore = np.nanmin(zscores_array, axis=0)
    max_zscore = np.nanmax(zscores_array, axis=0)
    
    # Count how many tiles contributed to each cell type pair (non-NaN values)
    n_valid_tiles = np.sum(~np.isnan(zscores_array), axis=0)

    # Use common cell types (all tiles aligned to this set)
    cell_types = common_cell_types
    n_tiles = len(zscores_list)

    print(f"  - Processed {n_tiles} tiles")
    print(f"  - Common cell types: {n_common_types} (from {n_tiles} tiles)")
    
    # Calculate valid (non-NaN) statistics
    valid_mean = mean_zscore[~np.isnan(mean_zscore)]
    valid_std = std_zscore[~np.isnan(std_zscore)]
    
    if len(valid_mean) > 0:
        print(f"  - Mean z-score range: [{np.nanmin(mean_zscore):.2f}, {np.nanmax(mean_zscore):.2f}]")
        print(f"  - Mean std across tiles: {np.nanmean(std_zscore):.3f}")
        print(f"  - Max std across tiles: {np.nanmax(std_zscore):.3f}")
    else:
        print(f"  - Warning: All values are NaN (no common cell types?)")

    # Save aggregated statistics
    print(f"\nSaving aggregated results to {output_dir}/...")

    mean_df = pd.DataFrame(mean_zscore, index=cell_types, columns=cell_types)
    mean_df.to_csv(output_dir / 'aggregated_mean_zscore.csv')

    std_df = pd.DataFrame(std_zscore, index=cell_types, columns=cell_types)
    std_df.to_csv(output_dir / 'aggregated_std_zscore.csv')

    median_df = pd.DataFrame(median_zscore, index=cell_types, columns=cell_types)
    median_df.to_csv(output_dir / 'aggregated_median_zscore.csv')

    sum_sigval = None
    if schapiro_sum_sigval:
        if len(aligned_sig_list) != n_tiles:
            print("  [!] Warning: schapiro_sum_sigval skipped (sig list length mismatch)")
        else:
            sig_stack = np.stack(aligned_sig_list, axis=0).astype(np.int64)
            sum_sigval = np.sum(sig_stack, axis=0)
            sum_sig_df = pd.DataFrame(sum_sigval, index=cell_types, columns=cell_types)
            sum_sig_df.to_csv(output_dir / 'aggregated_summed_sigval.csv')

    from cti_aggregate import cell_types_display_labels

    plot_tick_labels = cell_types_display_labels(
        cell_types,
        abbrev_map=cell_type_abbrev_map,
        enabled=use_short_cell_type_labels_in_plots,
    )
    heatmap_annot_kws = {"size": cti_heatmap_annot_fontsize}
    tick_label_fs = cti_heatmap_annot_fontsize

    # Visualize mean CTI (mean z-score across tiles)
    fig, ax = plt.subplots(figsize=(10, 8))
    max_abs_value = max(abs(mean_zscore.min()), abs(mean_zscore.max()))

    sns.heatmap(
        mean_zscore,
        cmap='coolwarm',
        center=0,
        vmin=-np.ceil(max_abs_value),
        vmax=np.ceil(max_abs_value),
        annot=False,
        cbar_kws={'label': 'Mean Z-score'},
        linewidths=0.5,
        linecolor='white',
        xticklabels=plot_tick_labels,
        yticklabels=plot_tick_labels,
        square=True,
        ax=ax
    )
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)
    
    # Build title with parameters if provided
    title_parts = [f'Aggregated Cell Type Interaction (CTI)']
    title_parts.append(f'(Mean Z-score across {n_tiles} tiles)')
    if n_perms is not None:
        title_parts.append(f'n_perms={n_perms}')
    if n_neighbors is not None:
        title_parts.append(f'n_neighbors={n_neighbors}')
    title = '\n'.join(title_parts)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=tick_label_fs) # ha='right
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=tick_label_fs)
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_mean_cti.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize variability
    fig, ax = plt.subplots(figsize=(10, 8))
    max_std = std_zscore.max()

    sns.heatmap(
        std_zscore,
        cmap='YlOrRd',
        vmin=0,
        vmax=np.ceil(max_std),
        annot=True,
        fmt='.2f',
        annot_kws=heatmap_annot_kws,
        cbar_kws={'label': 'Standard Deviation'},
        linewidths=0.5,
        linecolor='white',
        xticklabels=plot_tick_labels,
        yticklabels=plot_tick_labels,
        square=True,
        ax=ax
    )
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)
    
    # Build title with parameters if provided
    title_parts = [f'Variability Across Tiles']
    title_parts.append(f'(Std Dev of Z-scores, {n_tiles} tiles)')
    if n_perms is not None:
        title_parts.append(f'n_perms={n_perms}')
    if n_neighbors is not None:
        title_parts.append(f'n_neighbors={n_neighbors}')
    title = '\n'.join(title_parts)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=tick_label_fs) # ha='right
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=tick_label_fs)
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_variability.png', dpi=300, bbox_inches='tight')
    plt.close()

    if sum_sigval is not None:
        vmax = float(n_tiles)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            sum_sigval.astype(float),
            cmap='coolwarm',
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt='.0f',
            annot_kws=heatmap_annot_kws,
            cbar_kws={'label': 'Summed sigval'},
            linewidths=0.5,
            linecolor='white',
            xticklabels=plot_tick_labels,
            yticklabels=plot_tick_labels,
            square=True,
            ax=ax,
        )
        ax.set_xlabel('Cell Type', fontsize=12)
        ax.set_ylabel('Cell Type', fontsize=12)
        sig_title = (
            f'Schapiro-style summed significance (sigval)\n'
            f'+1 interaction, −1 avoidance, 0 not significant; sum over {n_tiles} tiles\n'
            f'(rule: {sigval_method}, α={sigval_alpha}'
        )
        if sigval_method == 'z_threshold':
            sig_title += f', |z|≥{sigval_z_threshold})'
        else:
            sig_title += ')'
        ax.set_title(sig_title, fontsize=12, fontweight='bold', pad=16)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_label_fs)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=tick_label_fs)
        plt.tight_layout()
        plt.savefig(output_dir / 'aggregated_summed_sigval.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  - Saved aggregated_mean_zscore.csv")
    print(f"  - Saved aggregated_std_zscore.csv")
    print(f"  - Saved aggregated_median_zscore.csv")
    print(f"  - Saved aggregated_mean_cti.png")
    print(f"  - Saved aggregated_variability.png")
    if sum_sigval is not None:
        print(f"  - Saved aggregated_summed_sigval.csv")
        print(f"  - Saved aggregated_summed_sigval.png")

    aggregated = {
        'mean_zscore': mean_zscore,
        'std_zscore': std_zscore,
        'median_zscore': median_zscore,
        'min_zscore': min_zscore,
        'max_zscore': max_zscore,
        'sum_sigval': sum_sigval,
        'cell_types': cell_types,
        'n_tiles': n_tiles,
        'tile_names': actual_tile_names,
        'metadata_list': metadata_list
    }

    print("\n" + "=" * 70)
    print("AGGREGATION COMPLETE!")
    print("=" * 70)

    return aggregated


# Main analysis pipeline
def run_spatial_analysis_pipeline(adata_path, output_dir='spatial_analysis_results',
                                  radius=50, n_neighbors=20, n_perms=1000, save_adata=False,
                                  skip_cooccurrence=False, max_cells_for_cooccurrence=50000):
    """
    Run complete spatial analysis pipeline.

    Parameters:
    -----------
    adata_path : str
        Path to h5ad file
    output_dir : str, default='spatial_analysis_results'
        Directory to save results
    radius : float, default=50
        Radius for spatial graph
    n_perms : int, default=1000
        Number of permutations for CTI (Squidpy nhood_enrichment)
    save_adata : bool, default=False
        Whether to save the processed AnnData object with analysis results
    skip_cooccurrence : bool, default=False
        Whether to skip co-occurrence analysis (useful for large datasets)
    max_cells_for_cooccurrence : int, default=50000
        Maximum number of cells for co-occurrence analysis.
        If dataset is larger, co-occurrence will be automatically skipped.

    Returns:
    --------
    adata : AnnData
        AnnData object with all analysis results
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("SPATIAL ANALYSIS PIPELINE")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  - Loaded {adata.n_obs} cells")

    # Apply cell type colors
    load_and_apply_cell_type_colors(adata)

    # Step 1: Build spatial graph (Choose radius or knn)
    #adata = build_spatial_graph(adata, method='radius', radius=radius)
    adata = build_spatial_graph(adata, method='knn',n_neighbors=n_neighbors)

    # Step 2: Cell type interaction (CTI)
    adata = cell_type_interaction_analysis(adata, n_perms=n_perms)

    # Step 3: Co-occurrence analysis (skip for large datasets to avoid memory issues)
    if not skip_cooccurrence:
        if adata.n_obs > max_cells_for_cooccurrence:
            print(f"\n⚠️  Skipping co-occurrence analysis:")
            print(f"   Dataset has {adata.n_obs} cells (> {max_cells_for_cooccurrence} threshold)")
            print(f"   Co-occurrence requires ~{(adata.n_obs**2 * 4 / 1e9):.1f} GB of RAM")
            print(f"   Set skip_cooccurrence=False and increase max_cells_for_cooccurrence to force run")
        else:
            adata = compute_co_occurrence(adata)
    else:
        print("\nSkipping co-occurrence analysis (skip_cooccurrence=True)")

    # Step 4: Centrality scores
    adata = compute_centrality_scores(adata)

    # Step 5: Visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Spatial distribution
    visualize_spatial_distribution(
        adata,
        save_path=output_dir / 'spatial_distribution.png'
    )

    # CTI heatmap
    visualize_cell_type_interaction(
        adata,
        save_path=output_dir / 'cell_type_interaction.png',
        n_neighbors=n_neighbors,
        n_perms=n_perms
    )

    # Step 6: Summarize interactions
    interactions_df = summarize_interactions(adata)
    interactions_df.to_csv(output_dir / 'significant_interactions.csv', index=False)
    print(f"\n  - Saved interactions to: {output_dir / 'significant_interactions.csv'}")

    # Save processed data (optional)
    if save_adata:
        output_adata_path = output_dir / 'adata_with_spatial_analysis.h5ad'
        adata.write(output_adata_path)
        print(f"\n  - Saved processed AnnData to: {output_adata_path}")
    else:
        print(f"\n  - AnnData not saved (set save_adata=True to save)")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)

    return adata


# Example usage
if __name__ == "__main__":
    # Run pipeline on your data
    # current_dir = os.path.dirname(__file__)
    # parent_dir = os.path.dirname(current_dir)
    # adata_path = os.path.join(parent_dir, "tile_39520_7904.h5ad")
    adata_path = '/mnt/c/ProgramData/github_repo/image_analysis_scripts/neighborhood_composition/tile_39520_7904.h5ad'
    adata = run_spatial_analysis_pipeline(
        adata_path=adata_path,
        output_dir='spatial_analysis_results',
        n_neighbors=20,  # was 6
        radius=50,  # Adjust based on your tissue/magnification
        n_perms=1000,
        save_adata=False,  # Set to True to save the h5ad file
        skip_cooccurrence=False,  # Set to True to skip co-occurrence for large datasets
        max_cells_for_cooccurrence=50000  # Auto-skip co-occurrence if more cells
    )

    print("\nYou can now explore the results:")
    print("  - Check 'spatial_analysis_results/' folder for figures")
    print("  - Set save_adata=True if you want to save 'adata_with_spatial_analysis.h5ad'")

    # ========================================================================
    # TEST: File-based aggregation functions
    # ========================================================================
    print("\n" + "=" * 70)
    print("TESTING FILE-BASED AGGREGATION FUNCTIONS")
    print("=" * 70)

    # Test 1: Save intermediate results
    print("\n[TEST 1] Saving intermediate results...")
    tile_name = Path(adata_path).stem
    saved = save_intermediate_results(
        adata=adata,
        output_dir='spatial_analysis_results',
        tile_name=tile_name
    )
    print(f"  ✓ Saved zscore shape: {saved['zscore'].shape}")
    print(f"  ✓ Metadata: {saved['metadata']['n_cells']} cells")

    # Test 2: Load intermediate results
    print("\n[TEST 2] Loading intermediate results...")
    loaded = load_intermediate_results(
        output_dir='spatial_analysis_results',
        tile_name=tile_name
    )
    print(f"  ✓ Loaded zscore shape: {loaded['zscore'].shape}")
    print(f"  ✓ Cell types: {loaded['cell_types']}")
    print(f"  ✓ N cells: {loaded['n_cells']}")

    # Test 3: Aggregate from saved results (using single tile as demo)
    print("\n[TEST 3] Testing aggregation from saved files...")
    print("  Note: Using same tile twice as a demonstration")
    aggregated = aggregate_from_saved_results(
        tile_dirs=['spatial_analysis_results', 'spatial_analysis_results'],
        output_dir='spatial_analysis_results/aggregated_test',
        tile_names=[tile_name, tile_name]  # Same tile twice for demo
    )
    print(f"  ✓ Aggregated {aggregated['n_tiles']} tiles")
    print(f"  ✓ Mean z-score shape: {aggregated['mean_zscore'].shape}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("\nThe following functions are ready to use in cti_multiple.py:")
    print("  1. save_intermediate_results() - Save zscore.npy and metadata.json")
    print("  2. load_intermediate_results() - Load from saved files")
    print("  3. aggregate_from_saved_results() - Aggregate multiple tiles from disk")
    print("\nFiles created for testing:")
    print(f"  - spatial_analysis_results/{tile_name}_zscore.npy")
    print(f"  - spatial_analysis_results/{tile_name}_metadata.json")
    print(f"  - spatial_analysis_results/aggregated_test/aggregated_*.csv")
    print(f"  - spatial_analysis_results/aggregated_test/aggregated_*.png")