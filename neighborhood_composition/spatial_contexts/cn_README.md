# Spatial Contexts & Cellular Neighborhoods (CN)

Scripts for **unified cellular neighborhood (CN)** detection across HandE tiles, visualization by tissue group, sub-clustering, and optional **spatial context (SC)** detection on top of CNs.

Inputs are per-tile `.h5ad` files from `neighborhood_composition/data_preparation.py` (4-class labels: Others, Tumor, Lymphocyte, Fibroblast/Stroma).

---

## Suggested pipeline (most important → supporting)

```text
h5ad tiles
  ├─► cn_unified_kmeans.py              # 1. define shared CNs (core)
  │     └─► cn_unified_kmeans_groups.py # 2. figures / tile-group analysis
  │
  ├─► cn_subcluster.py                  # 3. refine selected CNs
  ├─► cn_by_group_barcharts.py          # 4. CN × tissue-group bars
  │
  ├─► spatial_contexts_unified.py       # 5. CN mixtures → spatial contexts
  │
  └─► cn_merge_celltypes_unified.py     # optional: remap types then re-run CN
```

Supporting data: `tile_categories_88_tiles.json` (tile → group: `bg`, `margin`, `tumour_inv`, …).

---



## 1. `cn_unified_kmeans.py` — **core CN detection**

Loads many tiles together, builds neighbor cell-type composition features, and runs **k-means on all cells at once** so CN labels are shared across tiles. Writes annotated per-tile h5ads and composition tables (plots are handled by the groups script).

**Example:**

```bash
cd neighborhood_composition/spatial_contexts

python cn_unified_kmeans.py \
  --tiles_dir "/path/to/pred/h5ad" \
  --output_dir "cn_unified_results" \
  --k 20 \
  --n_clusters 5 \
  --celltype_key cell_type
```

**Useful flags:** `--max_tiles` (test run), `--pattern "*.h5ad"`, `--random_state`, `--no_offset`.

**Main outputs:** under `--output_dir` (often also `all_n_cluster=<n>/…` depending on how you organize runs):

- `processed_h5ad/<tile>_adata_cns.h5ad` — CN labels in `obs['cn_celltype']`
- composition / analysis CSVs used by downstream scripts

---



## 2. `cn_unified_kmeans_groups.py` — **main visualization**

Reads CN-annotated h5ads and produces:

1. Unified analysis — CN composition heatmap, overall / per-tile CN frequency
2. Individual tile spatial CN maps
3. Group-specific plots — compare CNs by tile groups from `tile_categories_88_tiles.json`

**Example:**

```bash
python cn_unified_kmeans_groups.py \
  --processed_h5ad_dir "/path/to/cn_unified_results_n=5/processed_h5ad" \
  --categories_json "tile_categories_88_tiles.json" \
  --output_dir "/path/to/cn_unified_results_n=5/groups" \
  --cn_key cn_celltype \
  --k 20 \
  --n_clusters 5
```

For sub-clustered results, use `--cn_key cn_celltype_sub` (auto-set if the path contains `_sub`).

**Optional:** `--group margin` (one group only), `--no-generate_unified`, `--no-generate_individual`.

---



## 3. `cn_subcluster.py` — **split selected CNs**

Partitions cells from chosen parent CNs into non-overlapping child CNs using neighbor composition only (e.g. CN3 → CN3-1, CN3-2).

**Example:**

```bash
python cn_subcluster.py \
  --results_root "/path/to/cn_unified_results" \
  --n_clusters 5 \
  --subcluster_config "3:2,4:2"
```

Writes `all_n_cluster=5_sub/` with updated `processed_h5ad/`, `unified_cn_composition_sub.csv`, and `subcluster_config.json`. Labels go in `obs['cn_celltype_sub']`.

---



## 4. `cn_by_group_barcharts.py` — **CN × tissue-group bars**

For each `k` (and optionally sub-clustered folders), builds two stacked bars:

1. Within each CN — how cells distribute across tile groups
2. Within each tile group — how cells distribute across CNs

Only tiles listed in the categories JSON are included.

**Example:**

```bash
python cn_by_group_barcharts.py \
  --results_root "/path/to/cn_unified_results" \
  --categories_json "tile_categories_88_tiles.json" \
  --out_dir "/path/to/cn_unified_results/k_selection" \
  --k_min 4 \
  --k_max 13 \
  --include_sub \
  --save_csv
```

**Outputs:** `cn_by_group_barcharts_<k>.png` (and optional CSVs).

---



## 5. `spatial_contexts_unified.py` — **spatial contexts from CNs**

Builds higher-order **spatial contexts (SCs)** from local mixtures of CNs (neighbor graph over CN-labeled cells), then filters rare / low-occupancy SCs and writes maps / interaction graphs.

Requires CN results from `cn_unified_kmeans.py` (`*_adata_cns.h5ad` under `processed_h5ad/`).

**Example:**

```bash
python spatial_contexts_unified.py \
  --cn_results_dir "cn_unified_results" \
  --output_dir "sc_unified_results" \
  --k 40 \
  --threshold 0.9 \
  --min_fraction 0.1 \
  --min_cells 100 \
  --min_groups 1
```

**Useful flags:** `--coord_offset`, `--graph_layout spring|kamada_kawai|circular`.

---



## 6. `cn_merge_celltypes_unified.py` — **optional remap + re-run CN**

Two phases:

- **A** — copy tiles to a new folder, optionally remapping cell-type labels
- **B** — run unified CN on that folder

For the **4-class** model the default merge map is empty (identity copy). Use `--merge_map_json` only if you need remapping (e.g. legacy labels).

**Example:**

```bash
python cn_merge_celltypes_unified.py \
  --source_tiles_dir "/path/to/pred/h5ad" \
  --merged_tiles_dir "/path/to/h5ad_staged_n=5" \
  --output_dir "/path/to/cn_unified_results_n=5" \
  --k 20 \
  --n_clusters 5
```

Skip Phase A if staging is already done: `--skip_phase_a`. Force remapping with `--merge_map_json map.json`.

---



## Minimal end-to-end example

```bash
cd neighborhood_composition/spatial_contexts

# 1) Detect CNs (shared labels across tiles)
python cn_unified_kmeans.py \
  --tiles_dir "/path/to/pred/h5ad" \
  --output_dir "cn_unified_results_n=5" \
  --n_clusters 5 --k 20

# 2) Visualize (+ tissue groups)
python cn_unified_kmeans_groups.py \
  --processed_h5ad_dir "cn_unified_results_n=5/processed_h5ad" \
  --categories_json "tile_categories_88_tiles.json" \
  --output_dir "cn_unified_results_n=5/groups" \
  --n_clusters 5

# 3) Optional: spatial contexts
python spatial_contexts_unified.py \
  --cn_results_dir "cn_unified_results_n=5" \
  --output_dir "sc_unified_results_n=5"
```

Optional follow-ups: `cn_subcluster.py` to refine selected CNs, or `cn_by_group_barcharts.py` to compare CNs across tissue groups (run unified CN for several `n_clusters` first if you want multiple `k` charts).

---



## Notes

- Several scripts `chdir` to this folder; relative paths are relative to `spatial_contexts/`.
- Cell-type abbreviation / order for heatmaps assumes the 4-class names from `type_info_4class.json`.

