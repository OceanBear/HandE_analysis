## CN k‑Selection Metrics – Definitions and Calculations

This document explains the metrics used in `k_selection_metrics.csv` to help choose the number of cellular neighborhoods (`k`, i.e. `n_clusters`) for the unified CN analysis.

The metrics are computed from:

- **Tile × CN frequency matrix** \(P^{(k)} \in \mathbb{R}^{N \times k}\) (here \(N = 88\) tiles), where each row sums to 1 (CN proportions per tile).
- **CN × cell-type composition matrix** \(Q \in \mathbb{R}^{k \times T}\) (here \(T = 7\) cell types), where each row sums to 1 (cell-type proportions within each CN).
- **Tile group labels** from `tile_categories_88_tiles.json`:
  - `bg`, `margin`, `tumour_inv`, `tumour_lep`, `tumour_scar`

---

### 1. `silhouette_braycurtis` (Tile group discrimination)

**What it measures**

How well CN frequency profiles separate the 5 tile groups (`bg`, `margin`, `tumour_inv`, `tumour_lep`, `tumour_scar`).

**Calculation**

1. Build the tile × CN frequency matrix:
   - \(P^{(k)} = [p_{t,c}]\) with \(t = 1,\dots,N\) (tiles), \(c = 1,\dots,k\) (CNs),
   - Each row is normalized: \(\sum_{c=1}^{k} p_{t,c} = 1\).
2. Compute pairwise **Bray–Curtis distances** between tiles using their CN frequency vectors.
3. Use tile group labels as “true” clusters and compute the **silhouette score**.

For each tile \(t\):

- \(a(t)\) = average distance from tile \(t\) to all other tiles in the **same** group.
- \(b(t)\) = minimum, over all *other* groups, of the average distance from tile \(t\) to tiles in that other group.

The silhouette value for tile \(t\) is:

\[
\text{silhouette}(t)
  = \frac{b(t) - a(t)}{\max\{a(t),\, b(t)\}}
\]

The reported metric is the mean silhouette over all tiles:

\[
\text{silhouette\_braycurtis}
  = \frac{1}{N} \sum_{t=1}^{N} \text{silhouette}(t)
\]

**Interpretation**

- Range: \(-1\) to \(+1\).
- Higher is better (better separation of tile groups in CN frequency space).
- \(\text{silhouette} > 0.3\) is often considered “good” separation in generic clustering.
- \(\text{silhouette} < 0\) means tiles are, on average, closer (in Bray–Curtis distance) to another group than to their own group.

In this project, silhouettes are slightly negative for all \(k\), indicating substantial overlap in CN profiles between the five manually defined groups. We therefore use this metric **relatively across \(k\)** rather than as an absolute pass/fail criterion.

---

### 2. `cn_size_cv` (CN size balance)

**What it measures**

How balanced the sizes of the \(k\) CNs are (in terms of total cell counts). It is the **coefficient of variation** of CN sizes.

**Calculation**

1. For each CN \(c \in \{1,\dots,k\}\), count the total number of cells assigned to that CN across all tiles:

\[
\text{size}_c = \#\{\text{cells assigned to CN } c\}
\]

2. Form the size vector:

\[
\text{sizes} = [\text{size}_1, \dots, \text{size}_k]
\]

3. Compute the coefficient of variation:

\[
\text{cn\_size\_cv}
  = \frac{\sigma(\text{sizes})}{\mu(\text{sizes})}
\]

where \(\sigma(\cdot)\) is the standard deviation and \(\mu(\cdot)\) is the mean.

**Interpretation**

- Lower is better (more balanced CN sizes).
- \(\text{cn\_size\_cv} = 0\): all CNs have exactly the same size.
- \(\text{cn\_size\_cv} > 1\): highly imbalanced clusters (some very small, some very large).
- Very small CNs (for example \< 1 % of all cells) can indicate **over-splitting** or noise.

In this dataset, `cn_size_cv` is typically around 0.2–0.3 for all \(k\), indicating reasonably balanced CNs.

---

### 3. `max_cn_cosine_similarity` (Maximum CN redundancy)

**What it measures**

The **largest** cosine similarity between the cell-type composition vectors of any two CNs. This highlights the **most redundant** pair of CN phenotypes.

**Calculation**

1. Build the CN × cell-type composition matrix:

\[
Q = [q_{c,t}] \in \mathbb{R}^{k \times T}
\]

where each row \(Q_c = [q_{c,1}, \dots, q_{c,T}]\) is the vector of cell-type fractions for CN \(c\), and

\[
\sum_{t=1}^{T} q_{c,t} = 1 \quad \text{for each } c.
\]

2. For each CN pair \((i, j)\), compute the cosine similarity:

\[
\text{cosine}(Q_i, Q_j)
  = \frac{Q_i \cdot Q_j}{\lVert Q_i \rVert \, \lVert Q_j \rVert}
\]

3. Consider only off-diagonal pairs \(i \neq j\), and take the maximum:

\[
\text{max\_cn\_cosine\_similarity}
  = \max_{i \neq j} \text{cosine}(Q_i, Q_j)
\]

**Interpretation**

- Range: 0 to 1.
- Lower is better (less redundancy between CNs).
- \(\text{cosine} > 0.85\): highly redundant CNs.
- \(\text{cosine} > 0.90\): CNs are nearly identical in composition.

---

### 4. `mean_cn_cosine_similarity` (Average CN redundancy)

**What it measures**

The **average** cosine similarity across all CN pairs, reflecting overall redundancy between CN phenotypes.

**Calculation**

Using the same composition matrix \(Q\) and cosine similarities as above:

\[
\text{mean\_cn\_cosine\_similarity}
  = \frac{1}{k(k-1)} \sum_{i \neq j} \text{cosine}(Q_i, Q_j)
\]

(i.e. the mean over all off-diagonal entries of the cosine similarity matrix.)

**Interpretation**

- Lower is better (on average, CNs are more distinct).
- Use together with `max_cn_cosine_similarity` to understand both **worst-case** and **typical** redundancy.

---

### 5. `n_redundant_pairs_cosine_<threshold>` (Count of redundant CN pairs)

**What it measures**

The number of CN pairs whose **cosine similarity of cell-type composition** exceeds a chosen threshold.  
The threshold is encoded in the column name. For example:

- `n_redundant_pairs_cosine_85` = number of CN pairs with cosine similarity \(\gt 0.85\).

**Calculation**

1. Compute the full \(k \times k\) cosine similarity matrix between CNs:

\[
S_{i,j} = \text{cosine}(Q_i, Q_j)
\]

2. Count off-diagonal pairs \((i, j)\) (with \(i < j\)) such that:

\[
S_{i,j} > \text{threshold}
\]

**Interpretation**

- Lower is better (fewer obviously redundant CN pairs).
- 0 means no CN pair exceeds the redundancy threshold.
- Higher values suggest **over-clustering** (too many CNs capturing very similar phenotypes).

---

### 6. `n_interpretable_cns_dom50` (Biological interpretability)

**What it measures**

The number of CNs that have a **single dominant cell type** contributing more than 50 % of their composition. These CNs are easier to interpret biologically (e.g. “Lymphocyte CN”, “Stromal CN”).

**Calculation**

For each CN \(c\), take its cell-type composition vector \(Q_c = [q_{c,1}, \dots, q_{c,T}]\) and compute:

\[
\text{dom}_c = \max_{t} q_{c,t}
\]

Then:

\[
\text{n\_interpretable\_cns\_dom50}
  = \#\{\, c \mid \text{dom}_c > 0.50 \,\}
\]

**Interpretation**

- Higher is better:
  - More CNs with a clear, dominant cell type.
  - Easier to annotate and reason about the CNs.
- Lower values mean more **mixed** CNs, which can be harder to interpret.

**Example**

- CN1: 73 % Epithelium (PD‑L1\(^\text{hi}\)/Ki67\(^\text{hi}\)) → **interpretable** (counted).
- CN2: 49 % Epithelium (PD‑L1\(^\text{lo}\)/Ki67\(^\text{lo}\)), 22 % Epithelium (PD‑L1\(^\text{hi}\)/Ki67\(^\text{hi}\)) → **not** counted (no single type \(\gt 50\%\)).

---

### 7. `n_similar_pairs_low_usage_corr` (Similar CNs used differently across tiles)

**What it measures**

Among **composition-similar** CN pairs (high cosine similarity), this counts how many are **used differently across tiles** (low correlation of tile-level frequencies). These are **not redundant** and may represent distinct spatial contexts.

**Calculation**

1. Identify composition-similar pairs: CN pairs \((i, j)\) with

\[
S_{i,j} = \text{cosine}(Q_i, Q_j) > \text{similarity\_threshold}
\]

   (for example, similarity\_threshold = 0.85).

2. For each such pair:
   - Let \(\mathbf{p}_i = [p_{1,i}, \dots, p_{N,i}]\) be the per-tile frequencies of CN \(i\).
   - Let \(\mathbf{p}_j = [p_{1,j}, \dots, p_{N,j}]\) be the per-tile frequencies of CN \(j\).
   - Compute **Spearman’s rank correlation** \(r_{ij}\) between \(\mathbf{p}_i\) and \(\mathbf{p}_j\).

3. Count pairs where

\[
r_{ij} \le \text{low\_threshold}
\]

(default low\_threshold = 0.50).

**Interpretation**

- Higher values mean:
  - There are CN pairs that look **similar in composition** but are **distributed differently** across tiles.
  - These CNs likely capture **distinct spatial/biological contexts**, so splitting them can be justified.

---

### 8. `n_similar_pairs_high_usage_corr` (Similar CNs used similarly across tiles)

**What it measures**

Among composition-similar CN pairs, this counts those whose per-tile usage is also **highly correlated**. These are **truly redundant** CNs (similar composition and similar distribution across tiles).

**Calculation**

Using the same \(r_{ij}\) as above, count pairs with:

\[
r_{ij} \ge \text{high\_threshold}
\]

(default high\_threshold = 0.70).

**Interpretation**

- Lower is better (fewer truly redundant CNs).
- High values indicate CNs that are almost duplicates both compositionally and spatially; these are **candidates for merging**.

---

### Summary Table for Decision-Making

| **Metric**                          | **Heuristic Range** | **What to Look For**                                      |
|-------------------------------------|----------------------|-----------------------------------------------------------|
| `silhouette_braycurtis`            | \(> 0.3\) (generic)  | Higher = better tile-group separation (relative use here) |
| `cn_size_cv`                       | \< 1.0               | Lower = more balanced CN sizes                            |
| `max_cn_cosine_similarity`         | \< 0.85              | Lower = fewer highly redundant CN pairs                   |
| `mean_cn_cosine_similarity`        | Lower                | Lower = overall less redundancy                           |
| `n_redundant_pairs_cosine_*`       | 0 ideally            | Lower = fewer redundant CN pairs                          |
| `n_interpretable_cns_dom50`        | Higher               | Higher = more biologically interpretable CNs              |
| `n_similar_pairs_low_usage_corr`   | Higher (if many similar pairs) | Justifies keeping composition-similar CNs        |
| `n_similar_pairs_high_usage_corr`  | 0 ideally            | Lower = fewer truly redundant CNs                         |

In this project, because all `silhouette_braycurtis` values are slightly negative, we primarily rely on:

- CN redundancy metrics (`max_cn_cosine_similarity`, `n_redundant_pairs_cosine_*`),
- CN size balance (`cn_size_cv`),
- Biological interpretability (`n_interpretable_cns_dom50`),

to select an appropriate \(k\), while using `silhouette_braycurtis` as a **secondary, relative** indicator rather than as an absolute threshold.

