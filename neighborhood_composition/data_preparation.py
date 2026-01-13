import json
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from tqdm import tqdm
import os
from pathlib import Path
# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)
# Cell type mapping
CELL_TYPE_DICT = {
    0: "Undefined",
    1: "Epithelium (PD-L1lo/Ki67lo)",
    2: "Epithelium (PD-L1hi/Ki67hi)",
    3: "Macrophage",
    4: "Lymphocyte",
    5: "Vascular",
    6: "Fibroblast/Stroma"
}

# Color mapping for visualization

CELL_TYPE_COLORS = {
    0: "#000000",  # Black (RGB: 0, 0, 0) - Undefined
    1: "#387F39",  # Dark Green (RGB: 56, 127, 57) - Epithelium low
    2: "#00FF00",  # Bright Green (RGB: 0, 255, 0) - Epithelium high
    3: "#FC8D62",  # Coral/Salmon (RGB: 252, 141, 98) - Macrophage
    4: "#FFD92F",  # Yellow (RGB: 255, 217, 47) - Lymphocyte
    5: "#4535C1",  # Blue/Purple (RGB: 69, 53, 193) - VascularC
    6: "#17BECF"   # Cyan (RGB: 23, 190, 207) - Fibroblast/Stroma
}


def load_json_to_anndata(json_path, tile_name=None, image_height=None):
    """
    Convert NucSegAI JSON output to AnnData object for Squidpy analysis.

    Parameters:
    -----------
    json_path : str or Path
        Path to the JSON file
    tile_name : str, optional
        Name/identifier for this tile (useful when combining multiple tiles)
    image_height : int, optional
        Height of the image in pixels (for Y-axis inversion)
        If not provided, will be inferred from max Y coordinate

    Returns:
    --------
    adata : AnnData
        AnnData object with spatial information
    """

    # Load JSON data
    print(f"Loading JSON file: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract tile name from path if not provided
    if tile_name is None:
        tile_name = Path(json_path).stem

    # Extract nucleus data
    nuc_data = data['nuc']
    print(f"Found {len(nuc_data)} cells")

    # Lists to store cell information
    cell_ids = []
    centroids = []
    cell_types = []
    cell_type_probs = []
    bboxes = []

    # Parse each nucleus with progress bar
    print("Parsing cell data...")
    for cell_id, cell_info in tqdm(nuc_data.items(), desc="Processing cells", unit="cell"):
        cell_ids.append(f"{tile_name}_{cell_id}")

        # Centroid coordinates are stored as [x, y] in JSON
        centroid_x, centroid_y = cell_info['centroid']
        centroids.append([centroid_x, centroid_y])

        # Cell type information
        cell_type_id = cell_info['type']
        cell_types.append(CELL_TYPE_DICT[cell_type_id])
        cell_type_probs.append(cell_info['type_prob'])

        # Bounding box information
        bboxes.append(cell_info['bbox'])

    # Create observations dataframe
    print("Creating AnnData object...")
    # Ensure categorical order matches cell_type_id order
    cell_type_categories = [CELL_TYPE_DICT[i] for i in sorted(CELL_TYPE_DICT.keys())]

    obs_df = pd.DataFrame({
        'cell_id': cell_ids,
        'cell_type': pd.Categorical(cell_types, categories=cell_type_categories, ordered=True),
        'cell_type_id': [list(CELL_TYPE_DICT.keys())[list(CELL_TYPE_DICT.values()).index(ct)]
                         for ct in cell_types],
        'cell_type_prob': cell_type_probs,
        'tile_name': tile_name
    })
    obs_df.index = cell_ids

    # Convert centroids to numpy array
    spatial_coords = np.array(centroids)

    # Invert Y-axis to match image coordinate system
    # Image coordinates have Y=0 at top, matplotlib has Y=0 at bottom
    if image_height is None:
        # Infer image height from max Y coordinate
        image_height = spatial_coords[:, 1].max()

    spatial_coords[:, 1] = image_height - spatial_coords[:, 1]

    # Create placeholder expression matrix (required by AnnData)
    # We don't have expression data, so create empty matrix
    X = np.zeros((len(cell_ids), 1))

    # Create AnnData object
    adata = ad.AnnData(
        X=X,
        obs=obs_df,
        dtype=np.float32
    )

    # Add spatial coordinates
    adata.obsm['spatial'] = spatial_coords

    # Add cell type colors for visualization
    adata.uns['cell_type_colors'] = [CELL_TYPE_COLORS[ct_id]
                                     for ct_id in sorted(CELL_TYPE_DICT.keys())]

    # Store bounding box information in obsm
    # JSON format: [[y_min, x_min], [y_max, x_max]]
    # Convert to [x_min, y_min, x_max, y_max] format
    bbox_array = np.array([[bb[0][1], bb[0][0], bb[1][1], bb[1][0]]
                           for bb in bboxes])
    adata.obsm['bbox'] = bbox_array

    # Add metadata
    # Convert cell_type_distribution to use safe keys (replace / with -)
    cell_type_counts = obs_df['cell_type'].value_counts().to_dict()
    safe_cell_type_counts = {k.replace('/', '-'): v for k, v in cell_type_counts.items()}

    adata.uns['spatial_metadata'] = {
        'tile_name': tile_name,
        'coordinate_system': 'pixel',
        'n_cells': len(cell_ids),
        'cell_type_distribution': safe_cell_type_counts
    }

    print(f"Created AnnData object:")
    print(f"  - Number of cells: {adata.n_obs}")
    print(f"  - Cell types: {obs_df['cell_type'].value_counts().to_dict()}")
    print(f"  - Spatial range: X[{spatial_coords[:, 0].min():.1f}, {spatial_coords[:, 0].max():.1f}], "
          f"Y[{spatial_coords[:, 1].min():.1f}, {spatial_coords[:, 1].max():.1f}]")

    return adata


def batch_process_json_files(json_dir=None, json_paths=None, output_dir=None):
    """
    Batch process multiple JSON files and save each as a separate h5ad file.

    Parameters:
    -----------
    json_dir : str or Path, optional
        Directory containing JSON files to process
        Either json_dir or json_paths must be provided
    json_paths : list of str/Path, optional
        List of specific JSON file paths to process
        Either json_dir or json_paths must be provided
    output_dir : str or Path, optional
        Directory to save output h5ad files
        If None, saves in the same directory as each JSON file

    Returns:
    --------
    results : dict
        Dictionary with processing results:
        - 'success': list of successfully processed files
        - 'failed': list of (file, error_message) tuples for failed files
    """

    # Determine which JSON files to process
    if json_paths is None and json_dir is None:
        raise ValueError("Either json_dir or json_paths must be provided")

    if json_paths is None:
        json_dir = Path(json_dir)
        json_paths = sorted(json_dir.glob("*.json"))
        print(f"Found {len(json_paths)} JSON files in {json_dir}")
    else:
        json_paths = [Path(p) for p in json_paths]

    if len(json_paths) == 0:
        print("No JSON files found to process")
        return {'success': [], 'failed': []}

    # Setup output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Output directory: {output_dir}")

    # Track results
    results = {'success': [], 'failed': []}

    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING: {len(json_paths)} JSON FILES")
    print(f"{'='*60}\n")

    # Process each JSON file
    for json_path in tqdm(json_paths, desc="Processing JSON files", unit="file"):
        try:
            # Load and convert JSON to AnnData
            adata = load_json_to_anndata(json_path)

            # Determine output path
            if output_dir is not None:
                output_path = output_dir / f"{json_path.stem}.h5ad"
            else:
                output_path = json_path.parent / f"{json_path.stem}.h5ad"

            # Save h5ad file
            adata.write(output_path)
            results['success'].append(str(json_path))

            print(f"  ✓ Saved: {output_path.name}")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            results['failed'].append((str(json_path), error_msg))
            print(f"  ✗ Failed: {json_path.name} - {error_msg}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(results['success'])}/{len(json_paths)} files")
    if results['failed']:
        print(f"Failed: {len(results['failed'])} files")
        print("\nFailed files:")
        for filepath, error in results['failed']:
            print(f"  - {Path(filepath).name}: {error}")

    return results


def combine_multiple_tiles(json_paths, tile_positions=None):
    """
    Combine multiple tiles into a single AnnData object.

    Parameters:
    -----------
    json_paths : list of str/Path
        List of paths to JSON files
    tile_positions : dict, optional
        Dictionary mapping tile names to (x_offset, y_offset) positions
        If None, tiles will be arranged sequentially

    Returns:
    --------
    adata : AnnData
        Combined AnnData object
    """

    adatas = []

    for i, json_path in tqdm(enumerate(json_paths), total=len(json_paths), desc="Processing tiles"):
        tile_name = Path(json_path).stem
        adata_tile = load_json_to_anndata(json_path, tile_name=tile_name)

        # Adjust spatial coordinates if positions provided
        if tile_positions and tile_name in tile_positions:
            x_offset, y_offset = tile_positions[tile_name]
            adata_tile.obsm['spatial'][:, 0] += x_offset
            adata_tile.obsm['spatial'][:, 1] += y_offset

        adatas.append(adata_tile)

    # Concatenate all tiles
    adata_combined = ad.concat(adatas, join='outer', label='tile',
                               keys=[a.uns['spatial_metadata']['tile_name'] for a in adatas])

    print(f"\nCombined AnnData object:")
    print(f"  - Total cells: {adata_combined.n_obs}")
    print(f"  - Number of tiles: {len(adatas)}")

    return adata_combined


# Example usage
if __name__ == "__main__":
    # ========================================================================
    # PROCESSING MODE SELECTION
    # ========================================================================
    # Choose one of the following modes:
    # - 'single': Process a single JSON file
    # - 'batch': Process multiple JSON files (saves each as separate h5ad)
    # - 'combine': Combine multiple tiles into one AnnData object
    # ========================================================================

    # Change this to 'single', 'batch', or 'combine'

    MODE = 'batch'
    # ========================================================================
    # MODE 1: SINGLE FILE PROCESSING
    # ========================================================================
    if MODE == 'single':
        json_path = '/mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/TCGA-MN-A4N4-01Z-00-DX2.9550732D-8FB1-43D9-B094-7C0CD310E9C0.json'

        adata = load_json_to_anndata(json_path)

        # Save to h5ad format for later use
        output_path = Path(json_path).stem + '.h5ad'
        adata.write(output_path)
        print(f"\nAnnData object saved to '{output_path}'")

    # ========================================================================
    # MODE 2: BATCH PROCESSING (NEW)
    # ========================================================================
    elif MODE == 'batch':
        # Option A: Process all JSON files in a directory
        json_dir = '/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/Batch_105/pred/json'    # /mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/json
        output_dir = '/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/Batch_105/pred/h5ad'  # /mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad

        results = batch_process_json_files(
            json_dir=json_dir,
            output_dir=output_dir  # Set to None to save in same directory as JSON files
        )

        # Option B: Process specific list of JSON files (uncomment to use)
        # json_paths = [
        #     '/path/to/file1.json',
        #     '/path/to/file2.json',
        #     '/path/to/file3.json'
        # ]
        # results = batch_process_json_files(
        #     json_paths=json_paths,
        #     output_dir=None  # Optional output directory
        # )

    # ========================================================================
    # MODE 3: COMBINE MULTIPLE TILES
    # ========================================================================
    elif MODE == 'combine':
        json_paths = ['tile_1.json', 'tile_2.json', 'tile_3.json']

        # Optional: Provide tile positions for spatial alignment
        # tile_positions = {
        #     'tile_1': (0, 0),
        #     'tile_2': (1000, 0),
        #     'tile_3': (0, 1000)
        # }

        adata_combined = combine_multiple_tiles(
            json_paths,
            # tile_positions=tile_positions  # Uncomment if using positions
        )

        adata_combined.write('combined_tiles.h5ad')
        print(f"\nCombined AnnData object saved to 'combined_tiles.h5ad'")

    else:
        raise ValueError(f"Invalid MODE: '{MODE}'. Must be 'single', 'batch', or 'combine'")