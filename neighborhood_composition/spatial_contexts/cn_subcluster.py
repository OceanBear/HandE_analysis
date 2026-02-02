"""
Sub-cluster selected Cellular Neighborhoods (CNs) using composition-only approach.

This script reads processed h5ad files from cn_unified_kmeans.py (e.g. all_n_cluster=5/processed_h5ad/)
and sub-clusters specified parent CNs into child CNs using the aggregated neighbor cell-type
composition (aggregated_neighbors in obsm).

Example:
  When n_clusters=5, sub-cluster CN3 and CN4, each into 2 child CNs:
    - CN3 -> CN3-1, CN3-2
    - CN4 -> CN4-1, CN4-2
  Result: CN1, CN2, CN3-1, CN3-2, CN4-1, CN4-2, CN5

Child CNs do not overlap: cells from parent CN are partitioned into exactly one child CN.

Usage:
  python cn_subcluster.py \
    --results_root "/path/to/cn_unified_results" \
    --n_clusters 5 \
    --subcluster_config "3:2,4:2"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad
except ImportError:
    try:
        import scanpy as sc
        ad = sc
    except ImportError:
        raise ImportError("Requires anndata or scanpy to read h5ad files.")

from sklearn.cluster import MiniBatchKMeans

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_RANDOM_STATE = 0


def _log_progress(current: int, total: int, prefix: str = "") -> str:
    return f"  [{current}/{total}] {prefix}"


def load_combined_adata(
    processed_h5ad_dir: Path,
    cn_key: str = "cn_celltype",
    celltype_key: str = "cell_type",
    aggregated_key: str = "aggregated_neighbors",
) -> ad.AnnData:
    """Load and combine all processed h5ad files from a processed_h5ad directory."""
    h5ad_files = sorted(processed_h5ad_dir.glob("*_adata_cns.h5ad"))
    if not h5ad_files:
        raise FileNotFoundError(f"No *_adata_cns.h5ad files in {processed_h5ad_dir}")

    adata_list = []
    cell_type_categories = None

    for i, f in enumerate(h5ad_files, 1):
        tile_name = f.stem.replace("_adata_cns", "")
        a = ad.read_h5ad(str(f))

        if cn_key not in a.obs.columns:
            raise KeyError(f"Missing '{cn_key}' in {f.name}")
        if celltype_key not in a.obs.columns:
            raise KeyError(f"Missing '{celltype_key}' in {f.name}")
        if aggregated_key not in a.obsm:
            raise KeyError(
                f"Missing '{aggregated_key}' in obsm for {f.name}. "
                "Run cn_unified_kmeans.py first to generate aggregated neighbor features."
            )

        if pd.api.types.is_categorical_dtype(a.obs[celltype_key]):
            if cell_type_categories is None:
                cell_type_categories = a.obs[celltype_key].cat.categories.tolist()
            else:
                a.obs[celltype_key] = a.obs[celltype_key].cat.set_categories(
                    cell_type_categories, ordered=True
                )
        elif cell_type_categories is None:
            a.obs[celltype_key] = pd.Categorical(a.obs[celltype_key])
            cell_type_categories = a.obs[celltype_key].cat.categories.tolist()

        if "tile_name" not in a.obs.columns:
            a.obs["tile_name"] = tile_name

        adata_list.append(a)
        print(_log_progress(i, len(h5ad_files), f"Loaded {tile_name}: {a.n_obs} cells"))

    combined = ad.concat(adata_list, join="outer", index_unique="-")
    if cell_type_categories is not None:
        combined.obs[celltype_key] = combined.obs[celltype_key].cat.set_categories(
            cell_type_categories, ordered=True
        )

    print(f"✓ Combined {len(adata_list)} tiles: {combined.n_obs:,} cells")
    return combined


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


def subcluster_cns(
    adata: ad.AnnData,
    subcluster_config: Dict[int, int],
    *,
    cn_key: str = "cn_celltype",
    aggregated_key: str = "aggregated_neighbors",
    output_key: str = "cn_celltype_sub",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[np.ndarray, List[str]]:
    """
    Sub-cluster specified parent CNs. Returns (labels_array, ordered_cn_names).

    - Parent CNs in subcluster_config are split into child CNs (e.g. CN3-1, CN3-2).
    - Parent CNs not in subcluster_config keep their original label (e.g. CN1, CN2).
    - Child CNs do not overlap: each cell belongs to exactly one child or unchanged CN.
    """
    cn_raw = adata.obs[cn_key].values
    # Normalize to int (handle categorical 1,2,3 or "1","2","3")
    cn_int = np.zeros(adata.n_obs, dtype=int)
    for i, x in enumerate(cn_raw):
        try:
            cn_int[i] = int(float(str(x).strip())) if pd.notna(x) and str(x).strip() else -1
        except (ValueError, TypeError):
            cn_int[i] = -1

    X = adata.obsm[aggregated_key]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    n_cells = adata.n_obs
    # Initialize with placeholder; we'll fill in sub-cluster labels or original
    labels_sub = np.empty(n_cells, dtype=object)

    # Collect ordered CN names for output
    all_cn_names: List[str] = []
    parent_cns_sorted = sorted(set(cn_int) & {c for c in cn_int if c > 0})

    for parent_cn in parent_cns_sorted:
        if parent_cn not in subcluster_config:
            # Keep original label
            mask = cn_int == parent_cn
            labels_sub[mask] = f"CN{parent_cn}"
            all_cn_names.append(f"CN{parent_cn}")
            continue

        n_divisions = subcluster_config[parent_cn]
        mask = cn_int == parent_cn
        n_parent = mask.sum()
        if n_parent < n_divisions:
            print(f"  Warning: CN{parent_cn} has only {n_parent} cells, cannot split into {n_divisions}. Keeping as CN{parent_cn}.")
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

    # Ensure we didn't miss any cell; fallback for unexpected CN values
    missing = pd.isna(labels_sub) | (labels_sub == "")
    if missing.any():
        for i in np.where(missing)[0]:
            labels_sub[i] = f"CN{cn_int[i]}" if cn_int[i] > 0 else "unknown"

    return labels_sub, all_cn_names


def compute_subcluster_composition(
    adata: ad.AnnData,
    labels_sub: np.ndarray,
    celltype_key: str = "cell_type",
) -> pd.DataFrame:
    """Compute cell-type composition per sub-clustered CN."""
    comp = pd.crosstab(
        labels_sub,
        adata.obs[celltype_key],
        normalize="index",
    )
    return comp


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sub-cluster selected CNs using aggregated neighbor composition."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/cn_unified_results",
        help="Path containing all_n_cluster=<k>/processed_h5ad folders (e.g. cn_unified_results).",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=9,
        help="Parent n_clusters (e.g. 5 for all_n_cluster=5).",
    )
    parser.add_argument(
        "--subcluster_config",
        type=str,
        default="5:2",
        help="Comma-separated 'CN:n_divisions' pairs. E.g. '3:2,4:2' splits CN3 and CN4 each into 2.",
    )
    parser.add_argument(
        "--cn_key",
        type=str,
        default="cn_celltype",
        help="CN label column in adata.obs.",
    )
    parser.add_argument(
        "--celltype_key",
        type=str,
        default="cell_type",
        help="Cell type column in adata.obs.",
    )
    parser.add_argument(
        "--aggregated_key",
        type=str,
        default="aggregated_neighbors",
        help="Key in adata.obsm for aggregated neighbor composition.",
    )
    parser.add_argument(
        "--output_key",
        type=str,
        default="cn_celltype_sub",
        help="New column name for sub-clustered CN labels.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for MiniBatchKMeans.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_sub",
        help="Suffix for output folder (e.g. all_n_cluster=5_sub).",
    )

    args = parser.parse_args()

    results_root = Path(args.results_root)
    processed_dir = results_root / f"all_n_cluster={args.n_clusters}" / "processed_h5ad"
    if not processed_dir.exists():
        print(f"Error: {processed_dir} not found.")
        return 1

    subcluster_config = parse_subcluster_config(args.subcluster_config)
    if not subcluster_config:
        print("Error: subcluster_config is empty. Use e.g. '3:2,4:2'.")
        return 1

    print("=" * 60)
    print("CN SUB-CLUSTERING")
    print("=" * 60)
    print(f"Results root: {results_root}")
    print(f"n_clusters: {args.n_clusters}")
    print(f"Sub-cluster config: {subcluster_config}")
    print("=" * 60)

    # Load combined adata
    combined = load_combined_adata(
        processed_dir,
        cn_key=args.cn_key,
        celltype_key=args.celltype_key,
        aggregated_key=args.aggregated_key,
    )

    # Sub-cluster
    labels_sub, cn_names = subcluster_cns(
        combined,
        subcluster_config,
        cn_key=args.cn_key,
        aggregated_key=args.aggregated_key,
        output_key=args.output_key,
        random_state=args.random_state,
    )

    combined.obs[args.output_key] = pd.Categorical(labels_sub, categories=cn_names)

    # Print sizes
    vc = pd.Series(labels_sub).value_counts()
    print("\nSub-clustered CN sizes:")
    for name, count in vc.items():
        print(f"  {name}: {count:,} cells ({100 * count / len(labels_sub):.1f}%)")

    # Output directory
    out_base = results_root / f"all_n_cluster={args.n_clusters}{args.output_suffix}"
    out_processed = out_base / "processed_h5ad"
    out_processed.mkdir(parents=True, exist_ok=True)
    out_unified = out_base / "unified_analysis"
    out_unified.mkdir(parents=True, exist_ok=True)

    # Save per-tile h5ad with sub-clustered labels
    tile_names = combined.obs["tile_name"].unique()
    for i, tile_name in enumerate(sorted(tile_names), 1):
        mask = combined.obs["tile_name"] == tile_name
        tile_adata = combined[mask].copy()
        out_path = out_processed / f"{tile_name}_adata_cns.h5ad"
        tile_adata.write(str(out_path))
        print(_log_progress(i, len(tile_names), f"Saved {tile_name}"))

    # Save composition CSV
    comp = compute_subcluster_composition(combined, labels_sub, celltype_key=args.celltype_key)
    comp_path = out_unified / "unified_cn_composition_sub.csv"
    comp.to_csv(comp_path)
    print(f"\n✓ Saved composition to {comp_path}")

    # Save sub-cluster config for reproducibility
    config_path = out_base / "subcluster_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "n_clusters": args.n_clusters,
                "subcluster_config": subcluster_config,
                "random_state": args.random_state,
                "output_key": args.output_key,
            },
            f,
            indent=2,
        )
    print(f"✓ Saved config to {config_path}")

    print(f"\nDone. Sub-clustered results in: {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
