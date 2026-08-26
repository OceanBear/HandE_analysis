"""
Unified Cellular Neighborhood Detection Across Multiple Tiles

This script loads multiple tiles at once and performs CN detection on the combined
dataset, ensuring all tiles share the same CN composition. This is crucial for
downstream spatial context analysis.

Key Features:
- Loads all tiles in a directory into a unified dataset
- (optional) If tile list.csv provided only those will be processed
- Performs k-means clustering on all cells together
- Saves lightweight CN-label JSON files (not full h5ad copies) for each tile
- Saves composition CSVs (raw and z-scored) for downstream heatmap generation
- Saves neighborhood frequency CSVs (overall and per-tile)

Note: This script produces data only (CSVs, JSON files) — no plots/images are
generated. Any heatmap or spatial visualization should be built downstream from
the CSV outputs.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import squidpy as sq
import os
import argparse
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from typing import Optional, Tuple, List, Set
import warnings
import time
import threading
import json
from scipy.sparse import block_diag, csr_matrix

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION: Random State
# ============================================================================
# Default random seed for reproducibility. Override with --random_state.
DEFAULT_RANDOM_STATE = 0
# ============================================================================


def load_tile_selection(csv_path) -> Set[str]:
    """
    Load a set of tile names to include, from a CSV with a 'tile' column
    (e.g. one row per tile: 'JN_TS_001_tile_12883_7423'). Tile names should
    match each h5ad file's stem (filename without the .h5ad extension).

    Used to restrict analysis to a subset of tiles — e.g. excluding tiles for
    QC reasons — without needing to move or delete the underlying h5ad files.
    """
    df = pd.read_csv(csv_path)
    if 'tile' not in df.columns:
        raise ValueError(
            f"Expected a 'tile' column in {csv_path}, found columns: {list(df.columns)}"
        )
    tiles = set(df['tile'].astype(str).str.strip())
    print(f"✓ Loaded tile selection: {len(tiles)} tiles from {csv_path}")
    return tiles


class UnifiedCellularNeighborhoodDetector:
    """
    Detects cellular neighborhoods across multiple tiles using a unified approach.
    All tiles share the same CN composition, enabling cross-tile comparisons.
    """

    def __init__(self, tiles_directory: str, output_dir: str):
        """
        Initialize unified CN detector.

        Parameters:
        -----------
        tiles_directory : str
            Directory containing h5ad tile files
        output_dir : str
            Base directory for all outputs (caller is responsible for making
            this parameter-specific, e.g. including k/n_clusters/seed in the
            path, so repeated sweeps don't overwrite each other)
        """
        self.tiles_directory = Path(tiles_directory)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for outputs
        # cn_labels/ is created by save_cn_labels() itself
        for subdir in ['unified_analysis']:
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

    def discover_tiles(
        self,
        pattern: str = "*.h5ad",
        max_tiles: Optional[int] = None,
        tile_selection: Optional[Set[str]] = None,
    ) -> List[Path]:
        """
        Discover h5ad files in the tiles directory.

        If tile_selection is given (a set of tile names, see load_tile_selection),
        only files whose stem is in that set are kept — any tile names in the
        selection that aren't found on disk are reported as a warning.
        max_tiles is applied after selection filtering, as a further cap
        (useful for quick testing on a subset of an already-filtered list).
        """
        print(f"Discovering tiles in: {self.tiles_directory}")

        all_files = sorted(self.tiles_directory.glob(pattern))

        if not all_files:
            print(f"Warning: No {pattern} files found in {self.tiles_directory}")
            return []

        if tile_selection is not None:
            found_stems = {f.stem for f in all_files}
            missing = tile_selection - found_stems
            if missing:
                sample = sorted(missing)[:5]
                print(f"  Warning: {len(missing)} tile(s) from the selection list were not "
                      f"found in {self.tiles_directory}: {sample}"
                      f"{' ...' if len(missing) > 5 else ''}")
            filtered_files = [f for f in all_files if f.stem in tile_selection]
            print(f"  Tile selection applied: keeping {len(filtered_files)}/{len(all_files)} "
                  f"tiles found on disk ({len(all_files) - len(filtered_files)} excluded)")
            all_files = filtered_files

        tile_files = all_files[:max_tiles]

        if not tile_files:
            print(f"Warning: no tiles remain after selection/limit filtering")
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
            Note: This is only for visualization purposes downstream. Neighbor
            detection uses original coordinates per tile to prevent cross-tile
            neighbors.

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

                # Offset spatial coordinates if requested (for downstream visualization only)
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
        # NOTE: no index_unique here. Each cell's obs_name is already globally
        # unique across tiles ("{tile_name}_{nucleus_id}", set in
        # data_preparation.py), so anndata's index_unique suffixing isn't
        # needed — and would otherwise append a "-{batch}" suffix to every
        # obs_name, making it harder to cleanly recover the original nucleus
        # ID later (needed for the lightweight CN-label JSON export below).
        combined_adata = ad.concat(adata_list, join='outer')

        if 'spatial' not in combined_adata.obsm:
            print("  Warning: No spatial coordinates found in combined data")

        print(f"✓ Combined dataset: {combined_adata.n_obs} cells, {combined_adata.n_vars} genes")
        print(f"  Tiles: {combined_adata.obs['tile_name'].nunique()}")
        print(f"  Cell types: {combined_adata.obs[celltype_key].nunique()}")

        self.combined_adata = combined_adata
        return combined_adata

    def build_knn_graph(
        self,
        k: int,
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
            tile_coord_key = coord_key
            if 'spatial_original' in tile_adata.obsm:
                tile_coord_key = 'spatial_original'
                # Temporarily set as 'spatial' for squidpy compatibility
                tile_adata.obsm['spatial'] = tile_adata.obsm['spatial_original']

            # Get spatial coordinates for this tile
            coords = self._get_spatial_coords(tile_adata, 'spatial')
            if coords is None:
                print(f"    Warning: No spatial coordinates found for {tile_name}, skipping...")
                n_cells = tile_adata.n_obs
                tile_connectivities.append(csr_matrix((n_cells, n_cells)))
                tile_sizes.append(n_cells)
                continue

            # Build KNN graph for this tile only (using original coordinates)
            sq.gr.spatial_neighbors(
                tile_adata,
                spatial_key='spatial',
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

        # Combine connectivity matrices into block diagonal matrix (no cross-tile connections)
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
        """
        For each cell, compute the fraction of each cell phenotype in its neighborhood.

        Vectorized via sparse matrix multiplication instead of a per-cell Python
        loop: one-hot encode cell types, multiply by the (binarized) adjacency
        matrix to get neighbor type counts per cell in one shot, then divide by
        each cell's neighbor count. This is roughly 75-100x faster than the
        equivalent per-cell loop at multi-million-cell scale, entirely on CPU —
        no GPU needed. Produces numerically identical results to the old loop
        (verified against it directly).

        The connectivity matrix is only binarized (a stray duplicate/weighted
        edge shouldn't occur in squidpy's normal output, but is checked for
        defensively) if it isn't already binary — a full copy of this matrix
        is genuinely expensive at multi-million-cell scale (potentially
        several GB), so it's only made when actually needed, not unconditionally.
        """
        print(f"\nAggregating neighbors by {celltype_key}...")

        cell_types = self.combined_adata.obs[celltype_key].values
        unique_types = self.combined_adata.obs[celltype_key].cat.categories
        connectivity = self.combined_adata.obsp[connectivity_key]
        n_cells = self.combined_adata.n_obs
        n_types = len(unique_types)

        print(f"  Processing {n_cells:,} cells (vectorized)...")

        # Binarize: treat any nonzero entry as "is a neighbor", matching the
        # original loop's semantics regardless of stored edge weight/count.
        # IMPORTANT (memory): squidpy's spatial_neighbors builds a true KNN
        # graph, so this matrix should already be binary (all values == 1) in
        # practice — a duplicate/weighted edge should never occur. Rather than
        # unconditionally copying the whole matrix just to overwrite its data
        # (which, at multi-million-cell scale, can double the memory footprint
        # of the single largest object in this pipeline — potentially several
        # GB — and risk pushing the process into swap), check first and only
        # copy in the (expected-never) case it's actually needed.
        if connectivity.data.size and not np.all(connectivity.data == 1):
            print("  Note: connectivity matrix has non-binary values; binarizing "
                  "(this allocates a second copy of the matrix)")
            connectivity_binary = connectivity.copy()
            connectivity_binary.data = np.ones_like(connectivity_binary.data)
        else:
            connectivity_binary = connectivity

        # One-hot encode cell types: (n_cells, n_types)
        type_to_idx = {ct: j for j, ct in enumerate(unique_types)}
        type_indices = np.array([type_to_idx[ct] for ct in cell_types])
        one_hot = np.zeros((n_cells, n_types), dtype=np.float32)
        one_hot[np.arange(n_cells), type_indices] = 1.0

        # Sparse matmul: for each cell, count how many of each type are among its neighbors
        neighbor_type_counts = np.asarray(connectivity_binary @ one_hot)  # (n_cells, n_types)
        neighbor_totals = np.asarray(connectivity_binary.sum(axis=1)).flatten()

        # Avoid divide-by-zero for any cell with 0 neighbors (result stays 0 for that row)
        safe_totals = np.where(neighbor_totals > 0, neighbor_totals, 1)
        aggregated = neighbor_type_counts / safe_totals[:, None]
        aggregated[neighbor_totals == 0] = 0

        self.aggregated_neighbors = pd.DataFrame(
            aggregated, columns=unique_types, index=self.combined_adata.obs_names
        )
        self.combined_adata.obsm[output_key] = aggregated
        print(f"  ✓ Aggregated neighbor fractions shape: {aggregated.shape}")
        return self

    def detect_cellular_neighborhoods(
        self,
        n_clusters: int,
        random_state: int = None,
        aggregated_key: str = 'aggregated_neighbors',
        output_key: str = 'cn_celltype',
        heartbeat_interval: float = 30.0,
    ):
        """
        Cluster cells based on their neighborhood composition using MiniBatchKMeans.

        Prints a lightweight "still running" heartbeat every heartbeat_interval
        seconds while fitting, instead of sklearn's built-in verbose=1 (which
        prints one line per mini-batch — thousands of lines at multi-million-
        cell scale, since it scales with n_cells / batch_size, not with time).
        Set heartbeat_interval=0 to disable.
        """
        print(f"\nDetecting {n_clusters} unified cellular neighborhoods using MiniBatchKMeans...")

        if random_state is None:
            random_state = DEFAULT_RANDOM_STATE

        aggregated = self.combined_adata.obsm[aggregated_key]
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)

        stop_event = threading.Event()
        heartbeat_thread = None
        if heartbeat_interval and heartbeat_interval > 0:
            def _heartbeat():
                start = time.time()
                while not stop_event.wait(heartbeat_interval):
                    elapsed = time.time() - start
                    print(f"    ... still running ({elapsed/60:.1f} min elapsed)")
            heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
            heartbeat_thread.start()

        fit_start = time.time()
        try:
            cn_labels = kmeans.fit_predict(aggregated)
        finally:
            stop_event.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join()
        fit_elapsed = time.time() - fit_start
        print(f"  ✓ Clustering completed in {fit_elapsed:.1f}s")

        # Stored for diagnostics (e.g. the elbow method) — see cn_kmeans_sweep.py
        self.last_inertia_ = kmeans.inertia_

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
            frequency_df = pd.crosstab(
                self.combined_adata.obs['tile_name'],
                self.combined_adata.obs[cn_key],
                normalize='index'
            )
            print(f"  ✓ Calculated CN frequency per tile")
        else:
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

    def save_cn_labels(
        self,
        cn_key: str = 'cn_celltype',
        n_clusters: Optional[int] = None,
        save_composition: bool = False,
        aggregated_key: str = 'aggregated_neighbors',
        celltype_key: str = 'cell_type',
    ):
        """
        Save CN labels for each tile as a small JSON file, instead of writing
        a full annotated h5ad copy per tile.

        Each file is keyed by the original HoVer-Net nucleus ID (recovered by
        stripping the "{tile_name}_" prefix from obs_names — safe since we no
        longer pass index_unique to ad.concat, so no extra suffix is present).
        This lets downstream scripts (vis_kmeans.py, print_cn_tiles.py) join
        these labels back onto the original source h5ad tiles (which already
        have cell_type + spatial) at read time, without ever needing a second,
        heavier copy of the tile data on disk.

        If save_composition=True, each cell's neighbor-composition vector (the
        same features used for clustering, from obsm[aggregated_key]) is also
        saved into the same file, under a 'composition' key (plus
        'composition_columns' recording which cell type each position
        corresponds to). This is off by default to keep the common-case
        output as small as possible — it's only needed if you plan to run
        cn_subcluster.py later, which needs these vectors to re-cluster within
        a parent CN.
        """
        print(f"\nSaving lightweight CN-label JSON files (instead of full h5ad copies)...")
        if save_composition:
            print(f"  (also saving neighbor-composition vectors, for possible subclustering later)")

        cn_labels_dir = self.output_dir / 'cn_labels'
        cn_labels_dir.mkdir(parents=True, exist_ok=True)

        composition_columns = None
        if save_composition:
            composition_columns = list(self.combined_adata.obs[celltype_key].cat.categories)

        for tile_idx, tile_name in enumerate(self.tile_list, 1):
            print(self._log_progress(tile_idx, len(self.tile_list), f"Saving {tile_name}"))

            tile_mask = self.combined_adata.obs['tile_name'] == tile_name
            tile_obs = self.combined_adata.obs.loc[tile_mask]

            prefix = f"{tile_name}_"
            labels = {}
            composition = {} if save_composition else None
            n_unexpected = 0

            if save_composition:
                aggregated = self.combined_adata.obsm[aggregated_key]
                tile_positions = np.where(tile_mask.values)[0]

            for pos_idx, (obs_name, cn_val) in enumerate(zip(tile_obs.index, tile_obs[cn_key])):
                if not str(obs_name).startswith(prefix):
                    n_unexpected += 1
                    continue
                nucleus_id = str(obs_name)[len(prefix):]
                labels[nucleus_id] = int(cn_val)
                if save_composition:
                    row = aggregated[tile_positions[pos_idx]]
                    composition[nucleus_id] = [round(float(v), 6) for v in row]

            if n_unexpected:
                print(f"    Warning: {n_unexpected} cells in {tile_name} had an obs_name "
                      f"that didn't start with the expected '{prefix}' prefix; skipped.")

            payload = {
                'tile_name': tile_name,
                'cn_key': cn_key,
                'n_clusters': n_clusters,
                'labels': labels,
            }
            if save_composition:
                payload['composition'] = composition
                payload['composition_columns'] = composition_columns

            output_path = cn_labels_dir / f'{tile_name}_cn_labels.json'
            with open(output_path, 'w') as f:
                json.dump(payload, f)

        print(f"  ✓ Saved {len(self.tile_list)} CN-label JSON files to: {cn_labels_dir}/")

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

        summary_path = self.output_dir / 'unified_analysis' / 'unified_cn_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved summary to: {summary_path}")

        comp_path = self.output_dir / 'unified_analysis' / 'unified_cn_composition.csv'
        composition.to_csv(comp_path)
        print(f"  ✓ Saved composition to: {comp_path}")

    def run_full_pipeline(
        self,
        tile_files: List[Path],
        k: int,
        n_clusters: int,
        celltype_key: str = 'cell_type',
        random_state: int = None,
        coord_offset: bool = True,
        save_composition: bool = False,
    ):
        """Run the complete unified CN detection pipeline."""
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

        # Step 5: Compute composition (raw + z-scored) and save both as CSV
        # (composition_zscore is data-only here; use it downstream to build a heatmap)
        composition, composition_zscore = self.compute_unified_cn_composition(celltype_key=celltype_key)

        comp_path = self.output_dir / 'unified_analysis' / 'unified_cn_composition.csv'
        composition.to_csv(comp_path)
        print(f"  ✓ Saved composition to: {comp_path}")

        comp_zscore_path = self.output_dir / 'unified_analysis' / 'unified_cn_composition_zscore.csv'
        composition_zscore.to_csv(comp_zscore_path)
        print(f"  ✓ Saved z-scored composition to: {comp_zscore_path}")

        # Step 6: Calculate and save neighborhood frequency (overall + per-tile), data only
        freq_overall = self.calculate_neighborhood_frequency(group_by_tile=False)
        freq_overall_path = self.output_dir / 'unified_analysis' / 'neighborhood_frequency_overall.csv'
        freq_overall.to_csv(freq_overall_path, index=False)
        print(f"  ✓ Saved overall neighborhood frequency to: {freq_overall_path}")

        freq_per_tile = self.calculate_neighborhood_frequency(group_by_tile=True)
        freq_per_tile_path = self.output_dir / 'unified_analysis' / 'neighborhood_frequency_per_tile.csv'
        freq_per_tile.to_csv(freq_per_tile_path)
        print(f"  ✓ Saved per-tile neighborhood frequency to: {freq_per_tile_path}")

        # Step 7: Save lightweight CN-label JSON files (not full h5ad copies)
        self.save_cn_labels(n_clusters=n_clusters, save_composition=save_composition, celltype_key=celltype_key)

        # Step 8: Save summary statistics
        self.save_summary_statistics(
            k=k, n_clusters=n_clusters, celltype_key=celltype_key,
            composition=composition, random_state=random_state
        )

        total_time = time.time() - start_time

        print(f"\n{banner}\nPIPELINE COMPLETE!\n{banner}")
        print(f"Total processing time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}/")
        print(f"  - CN-label JSON files (lightweight, one per tile): {self.output_dir}/cn_labels/")
        print(f"  - Composition CSV: {self.output_dir}/unified_analysis/unified_cn_composition.csv")
        print(f"  - Z-scored composition CSV (for heatmap): {self.output_dir}/unified_analysis/unified_cn_composition_zscore.csv")
        print(f"  - Neighborhood frequency CSVs: {self.output_dir}/unified_analysis/neighborhood_frequency_*.csv")

        return self


def main():
    """Main function to run unified CN detection."""
    parser = argparse.ArgumentParser(
        description='Unified Cellular Neighborhood Detection Across Multiple Tiles'
    )
    parser.add_argument(
        '--tiles_dir', '-t',
        required=True,
        help='Directory containing h5ad tile files'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='cn_unified_results',
        help='Base output directory. Results for this run are saved under a '
             'parameter-specific subfolder (e.g. k20_nclusters6_seed0/), so '
             'repeated sweeps with different parameters do not overwrite each '
             'other. Default: cn_unified_results'
    )
    parser.add_argument(
        '--k', type=int, default=20,
        help='Number of nearest neighbors for the spatial KNN graph (default: 20)'
    )
    parser.add_argument(
        '--n_clusters', '-n', type=int, required=True,
        # Recommended starting point based on prior analysis: 6 (or 4-7 range
        # depending on tissue complexity) — but this is left required rather
        # than defaulted since it's the main parameter you'll be sweeping.
        help='Number of cellular neighborhoods (required, e.g. try 4-7)'
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
        '--tile_list_csv',
        default=None,
        help="Optional CSV with a 'tile' column listing which tile names to "
             "include (e.g. 'JN_TS_001_tile_12883_7423'), one per row. Any "
             "tiles found in --tiles_dir but not in this list are excluded. "
             "Useful for dropping tiles for QC reasons without moving/deleting files."
    )
    parser.add_argument(
        '--no_offset', action='store_true',
        help='Disable spatial coordinate offsetting between tiles'
    )
    parser.add_argument(
        '--random_state', '-r', type=int, default=None,
        help=f'Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE})'
    )
    parser.add_argument(
        '--save_composition', action='store_true',
        help='Also save each cell\'s neighbor-composition vector alongside its CN '
             'label (in the same cn_labels/*.json files). Off by default to keep '
             'output minimal — only needed if you plan to run cn_subcluster.py '
             'later, which needs these vectors to re-cluster within a parent CN.'
    )

    args = parser.parse_args()

    random_state = args.random_state if args.random_state is not None else DEFAULT_RANDOM_STATE

    # Namespace the output directory by parameter combo, so sweeping k /
    # n_clusters / random_state across multiple runs never overwrites a
    # previous run's results.
    run_subdir = f"k{args.k}_nclusters{args.n_clusters}_seed{random_state}"
    output_dir = Path(args.output_dir) / run_subdir

    banner = "=" * 80
    print(f"{banner}\nUNIFIED CELLULAR NEIGHBORHOOD DETECTION\n{banner}")
    print(f"Tiles directory: {args.tiles_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: k={args.k}, n_clusters={args.n_clusters}, random_state={random_state}")
    print(f"Cell type key: {args.celltype_key}")
    if args.max_tiles:
        print(f"Max tiles: {args.max_tiles} (testing mode)")
    print(banner)

    detector = UnifiedCellularNeighborhoodDetector(
        tiles_directory=args.tiles_dir,
        output_dir=str(output_dir)
    )

    tile_selection = load_tile_selection(args.tile_list_csv) if args.tile_list_csv else None
    tile_files = detector.discover_tiles(pattern=args.pattern, max_tiles=args.max_tiles, tile_selection=tile_selection)

    if not tile_files:
        print("No tiles found! Exiting...")
        return

    detector.run_full_pipeline(
        tile_files=tile_files,
        k=args.k,
        n_clusters=args.n_clusters,
        celltype_key=args.celltype_key,
        random_state=random_state,
        coord_offset=not args.no_offset,
        save_composition=args.save_composition,
    )

    print(f"\nUnified CN detection completed successfully!")
    print(f"Check the results in: {output_dir}/")


if __name__ == '__main__':
    main()
