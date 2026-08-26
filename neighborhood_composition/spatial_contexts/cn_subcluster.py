"""
Sub-cluster selected Cellular Neighborhoods (CNs) using composition-only approach.

This script reads the lightweight cn_labels/*.json files written by
cn_unified_kmeans_local.py (run with --save_composition, so each file has
both the CN label and the neighbor-composition vector used to compute it),
and sub-clusters specified parent CNs into child CNs using those composition
vectors.

Example:
  When n_clusters=5, sub-cluster CN3 and CN4, each into 2 child CNs:
    - CN3 -> CN3-1, CN3-2
    - CN4 -> CN4-1, CN4-2
  Result: CN1, CN2, CN3-1, CN3-2, CN4-1, CN4-2, CN5

Child CNs do not overlap: cells from a parent CN are partitioned into exactly
one child CN.

Output is written in the same lightweight cn_labels/*.json format as
cn_unified_kmeans_local.py (labels only now strings like "CN3-1"), so it plugs
directly into vis_kmeans.py / print_cn_tiles.py as --cn_labels_dir with no
changes needed there — their existing label-sorting logic already supports
this mixed "CN<n>" / "CN<n>-<m>" string format.

Usage:
  python cn_subcluster.py \\
    --source_h5ad_dir "/path/to/pred/h5ad" \\
    --cn_labels_dir "cn_unified_results/k20_nclusters5_seed0/cn_labels" \\
    --output_dir "cn_unified_results/k20_nclusters5_seed0_sub" \\
    --subcluster_config "3:2,4:2"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.cluster import MiniBatchKMeans

DEFAULT_RANDOM_STATE = 0


def _log_progress(current: int, total: int, prefix: str = "") -> str:
    return f"  [{current}/{total}] {prefix}"


def load_tile_selection(csv_path) -> Set[str]:
    """
    Load a set of tile names to include, from a CSV with a 'tile' column
    (e.g. one row per tile: 'JN_TS_001_tile_12883_7423'). Tile names should
    match each h5ad file's stem (filename without the .h5ad extension).
    """
    df = pd.read_csv(csv_path)
    if 'tile' not in df.columns:
        raise ValueError(
            f"Expected a 'tile' column in {csv_path}, found columns: {list(df.columns)}"
        )
    tiles = set(df['tile'].astype(str).str.strip())
    print(f"✓ Loaded tile selection: {len(tiles)} tiles from {csv_path}")
    return tiles


def parse_subcluster_config(config_str: str) -> Dict[int, int]:
    """
    Parse subcluster config string into {parent_cn: n_divisions}.
    Example: "3:2,4:2" -> {3: 2, 4: 2}
    """
    out = {}
    for part in config_str.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid subcluster config part: '{part}'. Use format 'CN:n_divisions' e.g. '3:2'")
        cn_str, n_str = part.split(":", 1)
        cn = int(cn_str.strip())
        n = int(n_str.strip())
        if n < 2:
            raise ValueError(f"n_divisions must be >= 2 for CN{cn}, got {n}")
        out[cn] = n
    return out


def load_cn_labels_with_composition(
    cn_labels_dir: Path,
    tile_selection: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, List[str]]:
    """
    Load every tile's cn_labels/*.json file (must include 'composition', i.e.
    the parent run was made with --save_composition).

    Returns:
        tile_names: per-cell tile name (parallel array)
        nucleus_ids: per-cell original nucleus ID (parallel array)
        cn_int: per-cell current CN label as int (parallel array)
        X: (n_cells, n_types) composition matrix
        composition_columns: cell type name for each column of X
    """
    label_files = sorted(cn_labels_dir.glob('*_cn_labels.json'))
    if not label_files:
        raise ValueError(f"No *_cn_labels.json files found in {cn_labels_dir}")

    if tile_selection is not None:
        label_files = [f for f in label_files if f.stem[:-len('_cn_labels')] in tile_selection]
        if not label_files:
            raise ValueError("No tiles remain after applying tile_selection")

    tile_names_list = []
    nucleus_ids_list = []
    cn_int_list = []
    composition_rows = []
    composition_columns = None

    for i, f in enumerate(label_files, 1):
        with open(f, 'r') as fh:
            payload = json.load(fh)

        tile_name = payload.get('tile_name', f.stem[:-len('_cn_labels')])
        labels = payload.get('labels', {})
        composition = payload.get('composition')

        if composition is None:
            raise ValueError(
                f"{f.name} has no 'composition' data. Re-run cn_unified_kmeans_local.py "
                f"with --save_composition to enable subclustering."
            )

        if composition_columns is None:
            composition_columns = payload.get('composition_columns')
        elif payload.get('composition_columns') != composition_columns:
            print(f"  Warning: {f.name}'s composition_columns differ from the first tile's; "
                  f"assuming consistent ordering anyway.")

        for nucleus_id, cn_val in labels.items():
            comp = composition.get(nucleus_id)
            if comp is None:
                continue  # shouldn't happen, but skip defensively rather than crash
            tile_names_list.append(tile_name)
            nucleus_ids_list.append(nucleus_id)
            cn_int_list.append(int(cn_val))
            composition_rows.append(comp)

        print(_log_progress(i, len(label_files), f"Loaded {tile_name}: {len(labels)} cells"))

    X = np.asarray(composition_rows, dtype=np.float64)
    cn_int = np.asarray(cn_int_list, dtype=int)
    print(f"✓ Combined {len(label_files)} tiles: {len(cn_int):,} cells")

    return tile_names_list, nucleus_ids_list, cn_int, X, composition_columns


def subcluster_cns(
    cn_int: np.ndarray,
    X: np.ndarray,
    subcluster_config: Dict[int, int],
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[np.ndarray, List[str]]:
    """
    Sub-cluster specified parent CNs. Returns (labels_array, ordered_cn_names).

    - Parent CNs in subcluster_config are split into child CNs (e.g. CN3-1, CN3-2).
    - Parent CNs not in subcluster_config keep their original label (e.g. CN1, CN2).
    - Every cell ends up with a "CN<n>"-style string label, whether subclustered or not.
    """
    n_cells = len(cn_int)
    labels_sub = np.empty(n_cells, dtype=object)
    all_cn_names: List[str] = []
    parent_cns_sorted = sorted(set(cn_int[cn_int > 0]))

    for parent_cn in parent_cns_sorted:
        mask = cn_int == parent_cn

        if parent_cn not in subcluster_config:
            labels_sub[mask] = f"CN{parent_cn}"
            all_cn_names.append(f"CN{parent_cn}")
            continue

        n_divisions = subcluster_config[parent_cn]
        n_parent = mask.sum()
        if n_parent < n_divisions:
            print(f"  Warning: CN{parent_cn} has only {n_parent} cells, cannot split into "
                  f"{n_divisions}. Keeping as CN{parent_cn}.")
            labels_sub[mask] = f"CN{parent_cn}"
            all_cn_names.append(f"CN{parent_cn}")
            continue

        X_parent = X[mask]
        kmeans = MiniBatchKMeans(n_clusters=n_divisions, random_state=random_state)
        child_idx = kmeans.fit_predict(X_parent)

        for c in range(n_divisions):
            child_mask = mask.copy()
            child_mask[mask] = child_idx == c
            labels_sub[child_mask] = f"CN{parent_cn}-{c + 1}"
            all_cn_names.append(f"CN{parent_cn}-{c + 1}")

    missing = pd.isna(labels_sub) | (labels_sub == "")
    if missing.any():
        for i in np.where(missing)[0]:
            labels_sub[i] = f"CN{cn_int[i]}" if cn_int[i] > 0 else "unknown"

    return labels_sub, all_cn_names


def compute_subcluster_composition(
    labels_sub: np.ndarray,
    tile_names: List[str],
    nucleus_ids: List[str],
    source_h5ad_dir: Path,
    celltype_key: str = "cell_type",
) -> pd.DataFrame:
    """
    Compute cell-type composition per sub-clustered CN. Reads cell_type from
    the original source h5ad tiles (not persisted in the lightweight labels).
    """
    unique_tiles = sorted(set(tile_names))
    cell_type_by_key = {}
    cell_type_categories = None

    for i, tile_name in enumerate(unique_tiles, 1):
        h5ad_path = source_h5ad_dir / f"{tile_name}.h5ad"
        if not h5ad_path.exists():
            print(f"  Warning: source h5ad not found for {tile_name}, skipping its cells "
                  f"in the composition CSV")
            continue
        adata = ad.read_h5ad(h5ad_path)
        if celltype_key not in adata.obs.columns:
            print(f"  Warning: '{celltype_key}' not in {h5ad_path.name}, skipping")
            continue

        if isinstance(adata.obs[celltype_key].dtype, pd.CategoricalDtype):
            if cell_type_categories is None:
                cell_type_categories = adata.obs[celltype_key].cat.categories.tolist()

        prefix = f"{tile_name}_"
        for obs_name, ct in zip(adata.obs_names, adata.obs[celltype_key]):
            if str(obs_name).startswith(prefix):
                nucleus_id = str(obs_name)[len(prefix):]
                cell_type_by_key[(tile_name, nucleus_id)] = ct

        print(_log_progress(i, len(unique_tiles), f"Read cell_type for {tile_name}"))

    matched_labels = []
    matched_types = []
    for label, tile_name, nucleus_id in zip(labels_sub, tile_names, nucleus_ids):
        ct = cell_type_by_key.get((tile_name, nucleus_id))
        if ct is not None:
            matched_labels.append(label)
            matched_types.append(ct)

    comp = pd.crosstab(
        pd.Series(matched_labels, name='cn_celltype_sub'),
        pd.Series(matched_types, name='cell_type'),
        normalize="index",
    )
    if cell_type_categories is not None:
        existing_cols = [c for c in cell_type_categories if c in comp.columns]
        remaining_cols = [c for c in comp.columns if c not in existing_cols]
        comp = comp[existing_cols + remaining_cols]

    return comp


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sub-cluster selected CNs using their aggregated neighbor composition."
    )
    parser.add_argument(
        "--source_h5ad_dir", required=True,
        help="Directory containing the ORIGINAL h5ad tiles from data_preparation.py "
             "(named '{tile_name}.h5ad'). Needed only to look up cell_type for the "
             "output composition CSV.",
    )
    parser.add_argument(
        "--cn_labels_dir", required=True,
        help="Directory of the parent run's lightweight cn_labels/*.json files "
             "(must have been generated with --save_composition).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory. Writes a new cn_labels/ subfolder (same lightweight "
             "format, ready to use as --cn_labels_dir in vis_kmeans.py / print_cn_tiles.py), "
             "plus unified_analysis/unified_cn_composition_sub.csv and subcluster_config.json.",
    )
    parser.add_argument(
        "--subcluster_config", required=True,
        help="Comma-separated 'CN:n_divisions' pairs. E.g. '3:2,4:2' splits CN3 and CN4 each into 2.",
    )
    parser.add_argument(
        "--celltype_key", default="cell_type",
        help="Cell type column name in the source h5ad tiles (default: cell_type)",
    )
    parser.add_argument(
        "--random_state", type=int, default=DEFAULT_RANDOM_STATE,
        help="Random seed for MiniBatchKMeans.",
    )
    parser.add_argument(
        "--tile_list_csv", default=None,
        help="Optional CSV with a 'tile' column restricting which tiles are included.",
    )

    args = parser.parse_args()

    source_h5ad_dir = Path(args.source_h5ad_dir)
    cn_labels_dir = Path(args.cn_labels_dir)
    output_dir = Path(args.output_dir)

    subcluster_config = parse_subcluster_config(args.subcluster_config)
    if not subcluster_config:
        print("Error: subcluster_config is empty. Use e.g. '3:2,4:2'.")
        return 1

    tile_selection = load_tile_selection(args.tile_list_csv) if args.tile_list_csv else None

    print("=" * 60)
    print("CN SUB-CLUSTERING")
    print("=" * 60)
    print(f"Source h5ad directory: {source_h5ad_dir}")
    print(f"CN labels directory:   {cn_labels_dir}")
    print(f"Output directory:      {output_dir}")
    print(f"Sub-cluster config:    {subcluster_config}")
    print("=" * 60)

    tile_names, nucleus_ids, cn_int, X, composition_columns = load_cn_labels_with_composition(
        cn_labels_dir, tile_selection=tile_selection
    )

    labels_sub, cn_names = subcluster_cns(
        cn_int, X, subcluster_config, random_state=args.random_state
    )

    vc = pd.Series(labels_sub).value_counts()
    print("\nSub-clustered CN sizes:")
    for name, count in vc.items():
        print(f"  {name}: {count:,} cells ({100 * count / len(labels_sub):.1f}%)")

    # --- Write new lightweight cn_labels/*.json, grouped back by tile ---
    output_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir = output_dir / 'cn_labels'
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    by_tile: Dict[str, Dict[str, dict]] = {}
    for tile_name, nucleus_id, label, comp_row in zip(tile_names, nucleus_ids, labels_sub, X):
        entry = by_tile.setdefault(tile_name, {'labels': {}, 'composition': {}})
        entry['labels'][nucleus_id] = str(label)
        entry['composition'][nucleus_id] = [round(float(v), 6) for v in comp_row]

    print(f"\nSaving sub-clustered CN-label JSON files...")
    for i, (tile_name, entry) in enumerate(sorted(by_tile.items()), 1):
        payload = {
            'tile_name': tile_name,
            'cn_key': 'cn_celltype_sub',
            'n_clusters': len(cn_names),
            'labels': entry['labels'],
            'composition': entry['composition'],
            'composition_columns': composition_columns,
        }
        out_path = out_labels_dir / f'{tile_name}_cn_labels.json'
        with open(out_path, 'w') as f:
            json.dump(payload, f)
        print(_log_progress(i, len(by_tile), f"Saved {tile_name}"))

    print(f"✓ Saved {len(by_tile)} CN-label JSON files to: {out_labels_dir}/")

    # --- Composition CSV (needs cell_type from source h5ad tiles) ---
    out_unified = output_dir / 'unified_analysis'
    out_unified.mkdir(parents=True, exist_ok=True)

    print(f"\nComputing sub-cluster composition (reading cell_type from source h5ad tiles)...")
    comp = compute_subcluster_composition(
        labels_sub, tile_names, nucleus_ids, source_h5ad_dir, celltype_key=args.celltype_key
    )
    comp_path = out_unified / 'unified_cn_composition_sub.csv'
    comp.to_csv(comp_path)
    print(f"✓ Saved composition to {comp_path}")

    # --- Config for reproducibility ---
    config_path = output_dir / 'subcluster_config.json'
    with open(config_path, 'w') as f:
        json.dump(
            {
                'source_h5ad_dir': str(source_h5ad_dir),
                'cn_labels_dir': str(cn_labels_dir),
                'subcluster_config': subcluster_config,
                'random_state': args.random_state,
                'cn_key': 'cn_celltype_sub',
                'resulting_cn_names': cn_names,
            },
            f,
            indent=2,
        )
    print(f"✓ Saved config to {config_path}")

    print(f"\nDone. Sub-clustered results in: {output_dir}")
    print(f"Use --cn_labels_dir {out_labels_dir} in vis_kmeans.py / print_cn_tiles.py to visualize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
