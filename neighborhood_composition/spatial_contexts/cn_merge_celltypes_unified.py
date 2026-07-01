"""
Merge cell-type labels in per-tile h5ads, then re-run unified CN (cn_unified_kmeans pipeline).

Phase A: copy tiles from --source_tiles_dir to --merged_tiles_dir with remapped cell types
         (default: two Epithelium labels -> "Tumor" per data_preparation.py naming).
Phase B: UnifiedCellularNeighborhoodDetector on merged tiles -> --output_dir.

Does not overwrite original tiles; use a new merged_tiles_dir and output_dir.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Same directory as cn_unified_kmeans.py (it chdirs on import)
os.chdir(Path(__file__).parent)

from cn_unified_kmeans import DEFAULT_RANDOM_STATE, UnifiedCellularNeighborhoodDetector

# Matches neighborhood_composition/data_preparation.py CELL_TYPE_DICT (epithelium -> Tumor)
DEFAULT_MERGE_MAP: Dict[str, str] = {
    "Epithelium (PD-L1lo/Ki67lo)": "Tumor",
    "Epithelium (PD-L1hi/Ki67hi)": "Tumor",
}

# Stable category order after merge (6 types)
DEFAULT_CATEGORY_ORDER: List[str] = [
    "Undefined",
    "Tumor",
    "Macrophage",
    "Lymphocyte",
    "Vascular",
    "Fibroblast/Stroma",
]

OBS_KEYS_TO_DROP_BEFORE_MERGED_SAVE = ("cn_celltype", "cn_celltype_sub")


def load_merge_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return dict(DEFAULT_MERGE_MAP)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--merge_map_json must contain a JSON object of {\"old_label\": \"new_label\"}")
    return {str(k): str(v) for k, v in data.items()}


def load_category_order(path: Optional[Path]) -> List[str]:
    if path is None:
        return list(DEFAULT_CATEGORY_ORDER)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("--category_order_json must contain a JSON array of category strings")
    return [str(x) for x in data]


def merge_celltypes_in_one_h5ad(
    h5ad_path: Path,
    out_path: Path,
    celltype_key: str,
    merge_map: Dict[str, str],
    category_order: List[str],
) -> Dict[str, int]:
    """Write merged h5ad; return remap counts (old_label -> n_cells)."""
    import scanpy as sc

    adata = sc.read_h5ad(h5ad_path)
    if celltype_key not in adata.obs.columns:
        raise KeyError(f"{celltype_key} not in {h5ad_path.name}")

    for k in OBS_KEYS_TO_DROP_BEFORE_MERGED_SAVE:
        if k in adata.obs.columns:
            del adata.obs[k]

    raw = adata.obs[celltype_key].astype(str)
    remap_counts: Dict[str, int] = {}
    new_vals = []
    for v in raw:
        if v in merge_map:
            nv = merge_map[v]
            remap_counts[v] = remap_counts.get(v, 0) + 1
            new_vals.append(nv)
        else:
            new_vals.append(v)

    present = set(new_vals)
    ordered_cats = [c for c in category_order if c in present]
    ordered_cats += sorted(c for c in present if c not in ordered_cats)

    adata.obs[celltype_key] = pd.Categorical(new_vals, categories=ordered_cats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(out_path)
    return remap_counts


def phase_a_merge_tiles(
    source_dir: Path,
    merged_dir: Path,
    celltype_key: str,
    merge_map: Dict[str, str],
    category_order: List[str],
    pattern: str,
    max_tiles: Optional[int],
) -> List[Path]:
    files = sorted(source_dir.glob(pattern))
    if max_tiles is not None:
        files = files[: max_tiles]
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {source_dir}")

    merged_dir.mkdir(parents=True, exist_ok=True)
    total_remap: Dict[str, int] = {}

    print(f"\n{'='*60}\nPhase A: merge cell types -> {merged_dir}\n{'='*60}")
    for f in files:
        out = merged_dir / f.name
        counts = merge_celltypes_in_one_h5ad(f, out, celltype_key, merge_map, category_order)
        for k, v in counts.items():
            total_remap[k] = total_remap.get(k, 0) + v
        print(f"  ✓ {f.name} -> {out.name}")

    if total_remap:
        print("\n  Remapped cells (across all tiles):")
        for old, n in sorted(total_remap.items(), key=lambda x: -x[1]):
            print(f"    {old!r} -> {merge_map[old]!r}: {n:,}")
    return files


def phase_b_unified_cn(
    merged_dir: Path,
    output_dir: Path,
    celltype_key: str,
    k: int,
    n_clusters: int,
    random_state: int,
    coord_offset: bool,
    pattern: str,
    max_tiles: Optional[int],
) -> None:
    print(f"\n{'='*60}\nPhase B: unified CN -> {output_dir}\n{'='*60}")

    detector = UnifiedCellularNeighborhoodDetector(
        tiles_directory=str(merged_dir),
        output_dir=str(output_dir),
    )
    tile_files = detector.discover_tiles(pattern=pattern, max_tiles=max_tiles)
    if not tile_files:
        raise RuntimeError("No tiles found in merged directory for phase B")

    detector.run_full_pipeline(
        tile_files=tile_files,
        k=k,
        n_clusters=n_clusters,
        celltype_key=celltype_key,
        random_state=random_state,
        coord_offset=coord_offset,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Merge cell types in tile h5ads, then re-run unified CN (see DEFAULT_MERGE_MAP)."
    )
    p.add_argument(
        "--source_tiles_dir",
        type=str,
        required=True,
        help="Directory of input tile .h5ad (e.g. pred/h5ad).",
    )
    p.add_argument(
        "--merged_tiles_dir",
        type=str,
        required=True,
        help="Directory to write merged .h5ad (created if missing).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Unified CN results directory (e.g. cn_unified_results_Tumor_merged).",
    )
    p.add_argument(
        "--merge_map_json",
        type=str,
        default=None,
        help='JSON object {"old":"new",...}. Default: two Epithelium -> Tumor.',
    )
    p.add_argument(
        "--category_order_json",
        type=str,
        default=None,
        help="JSON array of category names after merge. Default: 6-type order in script.",
    )
    p.add_argument("--celltype_key", type=str, default="cell_type", help="obs column for cell types.")
    p.add_argument("--k", type=int, default=20, help="KNN k (same as cn_unified_kmeans).")
    p.add_argument("--n_clusters", "-n", type=int, default=13, help="Number of CNs.")
    p.add_argument("--pattern", type=str, default="*.h5ad", help="Glob under source dir.")
    p.add_argument("--max_tiles", type=int, default=None, help="Limit tiles (testing).")
    p.add_argument("--no_offset", action="store_true", help="Disable spatial offset between tiles.")
    p.add_argument(
        "--random_state",
        type=int,
        default=None,
        help=f"Random seed (default: {DEFAULT_RANDOM_STATE}).",
    )
    p.add_argument(
        "--skip_phase_a",
        action="store_true",
        help="Only run unified CN; merged_tiles_dir must already contain merged h5ads.",
    )

    args = p.parse_args()
    source_dir = Path(args.source_tiles_dir).resolve()
    merged_dir = Path(args.merged_tiles_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    merge_map_path = Path(args.merge_map_json).resolve() if args.merge_map_json else None
    cat_order_path = Path(args.category_order_json).resolve() if args.category_order_json else None

    merge_map = load_merge_map(merge_map_path)
    category_order = load_category_order(cat_order_path)
    rs = args.random_state if args.random_state is not None else DEFAULT_RANDOM_STATE

    if not args.skip_phase_a:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {source_dir}")
        phase_a_merge_tiles(
            source_dir,
            merged_dir,
            args.celltype_key,
            merge_map,
            category_order,
            args.pattern,
            args.max_tiles,
        )
    else:
        if not merged_dir.is_dir():
            raise FileNotFoundError(f"--skip_phase_a: merged dir missing: {merged_dir}")

    phase_b_unified_cn(
        merged_dir,
        output_dir,
        args.celltype_key,
        args.k,
        args.n_clusters,
        rs,
        not args.no_offset,
        args.pattern,
        args.max_tiles,
    )

    print("\nDone. Merged tiles:", merged_dir)
    print("Unified CN results:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
