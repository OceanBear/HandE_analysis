import json
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from tqdm import tqdm
import os

from cell_type_config import (
    DEFAULT_TYPE_INFO_PATH,
    cell_type_category_order,
    load_cell_type_config,
)

# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)


def load_json_to_anndata(json_path, tile_name=None, image_height=None, type_info_path=None):
    """
    Convert HoVer-Net JSON output to AnnData object for Squidpy analysis.

    Parameters:
    -----------
    json_path : str or Path
        Path to the JSON file
    tile_name : str, optional
        Name/identifier for this tile (useful when combining multiple tiles)
    image_height : int, optional
        Height of the image in pixels (for Y-axis inversion)
        If not provided, will be inferred from max Y coordinate
    type_info_path : str or Path, optional
        Path to type_info JSON (default: project root type_info_4class.json)

    Returns:
    --------
    adata : AnnData
        AnnData object with spatial information
    """
    cell_type_dict, cell_type_colors, _ = load_cell_type_config(type_info_path)

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
    cell_type_ids = []
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
        cell_type_id = int(cell_info['type'])
        if cell_type_id not in cell_type_dict:
            raise KeyError(
                f"Unknown cell type id {cell_type_id} in {json_path}. "
                f"Expected ids: {sorted(cell_type_dict.keys())}"
            )
        cell_types.append(cell_type_dict[cell_type_id])
        cell_type_ids.append(cell_type_id)
        cell_type_probs.append(cell_info['type_prob'])

        # Bounding box information
        bboxes.append(cell_info['bbox'])

    # Create observations dataframe
    print("Creating AnnData object...")
    cell_type_categories = cell_type_category_order(cell_type_dict)

    obs_df = pd.DataFrame({
        'cell_id': cell_ids,
        'cell_type': pd.Categorical(cell_types, categories=cell_type_categories, ordered=True),
        'cell_type_id': cell_type_ids,
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
    adata.uns['cell_type_colors'] = [
        cell_type_colors[ct_id] for ct_id in sorted(cell_type_dict.keys())
    ]

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


def batch_process_json_files(
    json_dir=None,
    json_paths=None,
    output_dir=None,
    type_info_path=None,
    skip_existing=True,
):
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
    type_info_path : str or Path, optional
        Path to type_info JSON (default: project root type_info_4class.json)
    skip_existing : bool, default=True
        If True, skip JSON files whose output .h5ad already exists.

    Returns:
    --------
    results : dict
        Dictionary with processing results:
        - 'success': list of successfully processed files
        - 'skipped': list of files skipped because .h5ad already exists
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
        return {'success': [], 'skipped': [], 'failed': []}

    # Setup output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Output directory: {output_dir}")

    # Track results
    results = {'success': [], 'skipped': [], 'failed': []}

    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING: {len(json_paths)} JSON FILES")
    if skip_existing:
        print("Skip existing: ON (pass --overwrite to reprocess)")
    else:
        print("Skip existing: OFF (will overwrite existing .h5ad)")
    print(f"{'='*60}\n")

    # Process each JSON file
    for json_path in tqdm(json_paths, desc="Processing JSON files", unit="file"):
        try:
            # Determine output path
            if output_dir is not None:
                output_path = output_dir / f"{json_path.stem}.h5ad"
            else:
                output_path = json_path.parent / f"{json_path.stem}.h5ad"

            if skip_existing and output_path.exists():
                results['skipped'].append(str(json_path))
                print(f"  ↷ Skipped (exists): {output_path.name}")
                continue

            # Load and convert JSON to AnnData
            adata = load_json_to_anndata(json_path, type_info_path=type_info_path)

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
    print(f"Newly processed: {len(results['success'])}/{len(json_paths)} files")
    print(f"Skipped (already exists): {len(results['skipped'])}/{len(json_paths)} files")
    if results['failed']:
        print(f"Failed: {len(results['failed'])} files")
        print("\nFailed files:")
        for filepath, error in results['failed']:
            print(f"  - {Path(filepath).name}: {error}")

    return results


def combine_multiple_tiles(json_paths, tile_positions=None, type_info_path=None):
    """
    Combine multiple tiles into a single AnnData object.

    Parameters:
    -----------
    json_paths : list of str/Path
        List of paths to JSON files
    tile_positions : dict, optional
        Dictionary mapping tile names to (x_offset, y_offset) positions
        If None, tiles will be arranged sequentially
    type_info_path : str or Path, optional
        Path to type_info JSON (default: project root type_info_4class.json)

    Returns:
    --------
    adata : AnnData
        Combined AnnData object
    """

    adatas = []

    for i, json_path in tqdm(enumerate(json_paths), total=len(json_paths), desc="Processing tiles"):
        tile_name = Path(json_path).stem
        adata_tile = load_json_to_anndata(
            json_path, tile_name=tile_name, type_info_path=type_info_path
        )

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


def _parse_main():
    import argparse

    p = argparse.ArgumentParser(
        description="Convert HoVer-Net JSON to AnnData (.h5ad): single file, batch directory, or combine tiles."
    )
    p.add_argument(
        "--mode",
        choices=("single", "batch", "combine"),
        default="batch",
        help="single: one JSON; batch: all *.json in --json-dir; combine: merge --json paths.",
    )
    p.add_argument("--json", type=str, default=None, help="Path to one JSON file (mode=single).")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .h5ad path (mode=single). Default: <json_stem>.h5ad next to JSON.",
    )
    p.add_argument(
        "--json-dir",
        type=str,
        default=None,
        help="Directory of *.json files (mode=batch).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output .h5ad files (mode=batch). Default: same as each JSON parent.",
    )
    p.add_argument(
        "--json-list",
        type=str,
        nargs="+",
        default=None,
        help="Explicit JSON paths (mode=batch or combine). For combine, order is preserved.",
    )
    p.add_argument(
        "--combined-output",
        type=str,
        default="combined_tiles.h5ad",
        help="Output path for mode=combine (default: combined_tiles.h5ad).",
    )
    p.add_argument(
        "--type-info",
        type=str,
        default=str(DEFAULT_TYPE_INFO_PATH),
        help="Path to type_info JSON (default: project root type_info_4class.json).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .h5ad files (default: skip if output already exists).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_main()
    type_info_path = args.type_info
    skip_existing = not args.overwrite

    if args.mode == "single":
        if not args.json:
            raise SystemExit("mode=single requires --json PATH")
        json_path = args.json
        out = Path(args.output) if args.output else Path(json_path).with_suffix(".h5ad")
        if skip_existing and out.exists():
            print(f"↷ Skipped (exists): {out}")
            print("Pass --overwrite to reprocess.")
        else:
            adata = load_json_to_anndata(json_path, type_info_path=type_info_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            adata.write(out)
            print(f"\nAnnData object saved to '{out}'")

    elif args.mode == "batch":
        if args.json_list:
            batch_process_json_files(
                json_paths=args.json_list,
                output_dir=args.output_dir,
                type_info_path=type_info_path,
                skip_existing=skip_existing,
            )
        elif args.json_dir:
            batch_process_json_files(
                json_dir=args.json_dir,
                output_dir=args.output_dir,
                type_info_path=type_info_path,
                skip_existing=skip_existing,
            )
        else:
            raise SystemExit("mode=batch requires --json-dir DIR or --json-list FILE1.json FILE2.json ...")

    elif args.mode == "combine":
        if not args.json_list:
            raise SystemExit("mode=combine requires --json-list FILE1.json FILE2.json ...")
        out = Path(args.combined_output)
        if skip_existing and out.exists():
            print(f"↷ Skipped (exists): {out}")
            print("Pass --overwrite to reprocess.")
        else:
            adata_combined = combine_multiple_tiles(args.json_list, type_info_path=type_info_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            adata_combined.write(out)
            print(f"\nCombined AnnData object saved to '{out}'")

    else:
        raise ValueError(f"Invalid mode: {args.mode!r}")
