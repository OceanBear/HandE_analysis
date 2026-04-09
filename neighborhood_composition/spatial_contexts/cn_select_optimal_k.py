"""
Select optimal CN k (n_clusters) using saved per-tile .h5ad outputs.
-----------------------------------------------------------------------------
This script is designed to be run AFTER `cn_unified_kmeans.py` has generated
folders like:

  <results_root>/all_n_cluster=7/processed_h5ad/<tile>_adata_cns.h5ad

It computes, for each k in a range (default 4..10), quantitative metrics that
support selecting an "optimal" k balancing:
  - CN redundancy (CNs with very similar cell-type composition)
  - Cluster balance / interpretability

Key ideas (from our discussion):
  1) Redundant CNs are detected via pairwise similarity of CN cell-type
     composition vectors (cosine similarity by default).
  2) If two CNs are composition-similar but used differently across tiles,
     their per-tile frequency vectors will have low correlation (Spearman).

Outputs:
  - <out_dir>/k_selection_metrics.csv  (one row per k)
  - Optional per-k matrices:
      - tile_cn_frequency_k=<k>.csv
      - cn_celltype_composition_k=<k>.csv

Notes / dependencies:
  - Requires `anndata` (or `scanpy`) to read .h5ad files.
  - Uses numpy/pandas/scipy/sklearn for metrics.

Example:
  python cn_select_optimal_k.py \
    --results_root "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/cn_unified_results" \
    --out_dir "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/cn_unified_results/k_selection"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _import_h5ad_reader():
    """
    Return a callable read_h5ad(path) -> AnnData, with a helpful error message
    if neither anndata nor scanpy is available.
    """
    try:
        import anndata as ad  # type: ignore
        return ad.read_h5ad
    except Exception:
        try:
            import scanpy as sc  # type: ignore
            return sc.read_h5ad
        except Exception as e:
            raise ImportError(
                "Reading .h5ad requires `anndata` or `scanpy`.\n"
                "Install one of them, e.g. `pip install anndata` (or `scanpy`)."
            ) from e


def _safe_to_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _normalize_rows(mat: pd.DataFrame) -> pd.DataFrame:
    denom = mat.sum(axis=1).replace(0, np.nan)
    out = mat.div(denom, axis=0).fillna(0.0)
    return out


def _pairwise_cosine_similarity(x: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity for rows of x.
    Returns an (n,n) matrix with 1s on diagonal.
    """
    # Normalize rows
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    xn = x / norms
    return xn @ xn.T


def _offdiag_values(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return mat[mask]


@dataclass
class KMetrics:
    k: int
    n_tiles: int
    n_cells: int
    n_celltypes: int
    cn_size_cv: float
    max_cn_cosine_similarity: float
    mean_cn_cosine_similarity: float
    median_cn_cosine_similarity: float
    n_redundant_pairs_cosine: int
    n_interpretable_cns_dom50: int
    n_similar_pairs_low_usage_corr: int
    n_similar_pairs_high_usage_corr: int


def analyze_k(
    k: int,
    results_root: Path,
    *,
    cn_key: str,
    celltype_key: str,
    tile_key: str,
    redundancy_cosine_threshold: float,
    usage_corr_low: float,
    usage_corr_high: float,
    save_matrices: bool,
    out_dir: Path,
) -> Tuple[KMetrics, pd.DataFrame]:
    """
    Analyze a single k folder.
    """
    read_h5ad = _import_h5ad_reader()

    k_dir = results_root / f"all_n_cluster={k}" / "processed_h5ad"
    if not k_dir.exists():
        raise FileNotFoundError(f"Missing processed_h5ad directory: {k_dir}")

    h5ad_files = sorted(k_dir.glob("*.h5ad"))
    if not h5ad_files:
        raise FileNotFoundError(f"No .h5ad files found in: {k_dir}")

    # Tile × CN counts (later normalized to frequencies)
    tile_cn_counts: Dict[str, pd.Series] = {}

    # Global CN size (cell counts per CN)
    cn_total_counts: Dict[int, int] = {i: 0 for i in range(1, k + 1)}

    # CN × celltype counts (accumulated)
    cn_celltype_counts: Optional[pd.DataFrame] = None

    total_cells = 0
    celltype_set: set = set()

    for f in h5ad_files:
        adata = read_h5ad(str(f))

        if cn_key not in adata.obs.columns:
            raise KeyError(f"Missing `{cn_key}` in {f.name}")
        if celltype_key not in adata.obs.columns:
            raise KeyError(f"Missing `{celltype_key}` in {f.name}")

        # Determine tile name
        tile_name: Optional[str] = None
        if tile_key in adata.obs.columns:
            uniq = pd.unique(adata.obs[tile_key].astype(str))
            if len(uniq) == 1:
                tile_name = str(uniq[0])
        if tile_name is None:
            # Fallback: infer from filename: "<tile>_adata_cns.h5ad"
            name = f.stem
            if name.endswith("_adata_cns"):
                name = name[: -len("_adata_cns")]
            tile_name = name

        # CN counts for this tile
        cn_series = adata.obs[cn_key]
        # CN labels might be categorical strings ("1","2",...), ints, etc.
        cn_vals = cn_series.astype(str)
        cn_int = cn_vals.map(_safe_to_int)
        if cn_int.isna().any():
            bad = cn_vals[cn_int.isna()].unique()[:5]
            raise ValueError(f"Non-integer CN labels in {f.name}: {bad}")
        cn_int = cn_int.astype(int)

        vc = cn_int.value_counts().sort_index()
        # Ensure full 1..k
        vc = vc.reindex(range(1, k + 1), fill_value=0)
        tile_cn_counts[tile_name] = vc

        # Global size accumulation
        for cn_id, cnt in vc.items():
            cn_total_counts[int(cn_id)] += int(cnt)

        # CN × celltype counts for composition
        df = pd.DataFrame(
            {
                "cn": cn_int.values,
                "celltype": adata.obs[celltype_key].astype(str).values,
            }
        )
        celltype_set.update(df["celltype"].unique().tolist())
        ct = pd.crosstab(df["cn"], df["celltype"])
        ct = ct.reindex(index=range(1, k + 1), fill_value=0)
        cn_celltype_counts = ct if cn_celltype_counts is None else cn_celltype_counts.add(ct, fill_value=0)

        total_cells += int(adata.n_obs)

    # Tile × CN frequency matrix
    tile_index = sorted(tile_cn_counts.keys())
    tile_cn = pd.DataFrame({t: tile_cn_counts[t] for t in tile_index}).T
    tile_cn.index.name = "tile_name"
    tile_cn.columns = [f"CN_{i}" for i in range(1, k + 1)]
    tile_cn_freq = _normalize_rows(tile_cn)

    # CN celltype composition matrix (rows sum to 1)
    if cn_celltype_counts is None:
        raise RuntimeError("Unexpected: no CN×celltype counts were accumulated.")
    cn_celltype_counts = cn_celltype_counts.fillna(0).astype(int)
    cn_comp = _normalize_rows(cn_celltype_counts)

    # CN redundancy via cosine similarity of CN composition vectors
    S = _pairwise_cosine_similarity(cn_comp.to_numpy(dtype=float))
    off = _offdiag_values(S)
    max_sim = float(off.max()) if off.size else float("nan")
    mean_sim = float(off.mean()) if off.size else float("nan")
    median_sim = float(np.median(off)) if off.size else float("nan")
    redundant_pairs = int(np.sum(off > redundancy_cosine_threshold))

    # Save the full similarity matrix (small: k <= ~10)
    sim_index = [f"CN_{i}" for i in range(1, k + 1)]
    sim_mat_df = pd.DataFrame(S, index=sim_index, columns=sim_index)

    # Interpretability: CNs with a dominant cell type > 50%
    dom = cn_comp.max(axis=1)
    n_interpretable = int((dom > 0.50).sum())

    # CN size balance
    sizes = np.array([cn_total_counts[i] for i in range(1, k + 1)], dtype=float)
    cn_size_cv = float(np.std(sizes) / np.mean(sizes)) if np.mean(sizes) > 0 else float("nan")

    # Similar CN pairs: usage correlation across tiles
    n_low_corr = 0
    n_high_corr = 0
    try:
        from scipy.stats import spearmanr

        # Find (i,j) pairs above threshold; work on upper triangle only
        for i in range(k):
            for j in range(i + 1, k):
                if S[i, j] <= redundancy_cosine_threshold:
                    continue
                a = tile_cn_freq.iloc[:, i].to_numpy(dtype=float)
                b = tile_cn_freq.iloc[:, j].to_numpy(dtype=float)
                r = spearmanr(a, b).correlation
                if np.isnan(r):
                    continue
                if r <= usage_corr_low:
                    n_low_corr += 1
                if r >= usage_corr_high:
                    n_high_corr += 1
    except Exception:
        n_low_corr = 0
        n_high_corr = 0

    if save_matrices:
        out_dir.mkdir(parents=True, exist_ok=True)
        tile_cn_freq.to_csv(out_dir / f"tile_cn_frequency_k={k}.csv")
        cn_comp.to_csv(out_dir / f"cn_celltype_composition_k={k}.csv")
        sim_mat_df.to_csv(out_dir / f"cn_cosine_similarity_matrix_k={k}.csv")

    return KMetrics(
        k=k,
        n_tiles=int(tile_cn_freq.shape[0]),
        n_cells=int(total_cells),
        n_celltypes=int(len(celltype_set)),
        cn_size_cv=float(cn_size_cv),
        max_cn_cosine_similarity=float(max_sim),
        mean_cn_cosine_similarity=float(mean_sim),
        median_cn_cosine_similarity=float(median_sim),
        n_redundant_pairs_cosine=int(redundant_pairs),
        n_interpretable_cns_dom50=int(n_interpretable),
        n_similar_pairs_low_usage_corr=int(n_low_corr),
        n_similar_pairs_high_usage_corr=int(n_high_corr),
    ), sim_mat_df


def main() -> int:
    script_dir = Path(__file__).parent.resolve()

    p = argparse.ArgumentParser(
        description="Select optimal CN k using saved per-tile .h5ad outputs."
    )
    p.add_argument(
        "--results_root",
        type=str,
        default="/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/cn_unified_results",
        help="Path containing all_n_cluster=<k>/processed_h5ad folders.",
    )
    p.add_argument(
        "--k_min", type=int, default=4, help="Minimum k to evaluate (inclusive)."
    )
    p.add_argument(
        "--k_max", type=int, default=13, help="Maximum k to evaluate (inclusive)."
    )
    p.add_argument("--cn_key", type=str, default="cn_celltype", help="CN label column in adata.obs.")
    p.add_argument("--celltype_key", type=str, default="cell_type", help="Cell type column in adata.obs.")
    p.add_argument(
        "--tile_key",
        type=str,
        default="tile_name",
        help="Tile name column in adata.obs (fallback: infer from filename).",
    )
    p.add_argument(
        "--redundancy_cosine_threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold to call CN pairs redundant.",
    )
    p.add_argument(
        "--usage_corr_low",
        type=float,
        default=0.50,
        help="Spearman correlation <= this means similar CNs are used differently across tiles.",
    )
    p.add_argument(
        "--usage_corr_high",
        type=float,
        default=0.70,
        help="Spearman correlation >= this means similar CNs are used similarly across tiles (more redundant).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <results_root>/k_selection",
    )
    p.add_argument(
        "--save_matrices",
        action="store_true",
        help="Also save per-k tile×CN and CN×celltype composition CSVs.",
    )
    p.add_argument(
        "--make_plots",
        action="store_true",
        help="Make a few summary plots (requires matplotlib).",
    )

    args = p.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else (results_root / "k_selection")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[KMetrics] = []
    all_pairs_long: List[pd.DataFrame] = []
    for k in range(args.k_min, args.k_max + 1):
        try:
            m, sim_mat_df = analyze_k(
                k=k,
                results_root=results_root,
                cn_key=args.cn_key,
                celltype_key=args.celltype_key,
                tile_key=args.tile_key,
                redundancy_cosine_threshold=args.redundancy_cosine_threshold,
                usage_corr_low=args.usage_corr_low,
                usage_corr_high=args.usage_corr_high,
                save_matrices=bool(args.save_matrices),
                out_dir=out_dir,
            )
            rows.append(m)
            # Save per-k CN cosine similarity matrix (X/Y axes are CN numbers)
            pairs_path = out_dir / f"cn_cosine_similarity_pairs_{k}.csv"
            sim_mat_df.to_csv(pairs_path)
            print(f"Saved: {pairs_path}")

            # Also accumulate long-form pairs for "all k" file (upper triangle only),
            # matching the previous format: k, cn_i, cn_j, cosine_similarity
            pairs_rows = []
            for i in range(1, k + 1):
                for j in range(i + 1, k + 1):
                    pairs_rows.append(
                        {
                            "k": k,
                            "cn_i": i,
                            "cn_j": j,
                            "cosine_similarity": float(sim_mat_df.iloc[i - 1, j - 1]),
                        }
                    )
            all_pairs_long.append(pd.DataFrame(pairs_rows))

            print(f"✓ k={k}: max_sim={m.max_cn_cosine_similarity:.3f}")
        except Exception as e:
            print(f"✗ k={k}: {e}")

    if not rows:
        print("No k values were successfully analyzed. Check paths / dependencies.")
        return 2

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("k")

    # Make the redundant-pairs column name encode the threshold, e.g. 0.85 -> "..._85"
    thr_pct = int(round(float(args.redundancy_cosine_threshold) * 100))
    redundant_col_old = "n_redundant_pairs_cosine"
    redundant_col_new = f"n_redundant_pairs_cosine_{thr_pct}"
    if redundant_col_old in df.columns:
        df = df.rename(columns={redundant_col_old: redundant_col_new})

    df.to_csv(out_dir / "k_selection_metrics.csv", index=False)
    print(f"\nSaved: {out_dir / 'k_selection_metrics.csv'}")

    # Save long-form CN cosine similarity pairs for all k (same format as before)
    if all_pairs_long:
        all_k_path = out_dir / "cn_cosine_similarity_pairs_all_k.csv"
        pd.concat(all_pairs_long, ignore_index=True).to_csv(all_k_path, index=False)
        print(f"Saved: {all_k_path}")

    if args.make_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Plot: max CN cosine similarity vs k
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(df["k"], df["max_cn_cosine_similarity"], marker="s", color="tab:red", label="Max CN cosine similarity")
            ax.set_xlabel("k (n_clusters)")
            ax.set_ylabel("Max CN cosine similarity (redundancy)")
            ax.grid(alpha=0.3, linestyle="--")
            ax.legend(loc="best")

            plt.tight_layout()
            plot_path = out_dir / "k_selection_summary.png"
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {plot_path}")
        except Exception as e:
            print(f"Plotting failed (non-fatal): {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
