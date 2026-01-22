import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# --------------------------------------------------
# Config
# --------------------------------------------------
JSON_DIR = r"/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred/json/tumour_inv"
CELL_TYPES = list(range(7))  # 0–6 as defined by NucSegAI

# --------------------------------------------------
# Helper: extract cell-type proportions from one JSON
# --------------------------------------------------
def load_tile_proportions(json_path, min_prob=None):
    """
    Returns a vector of cell-type proportions for one tile.
    Optionally filter by type_prob >= min_prob.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    counts = {t: 0 for t in CELL_TYPES}

    # data["nuc"] is a dictionary, not a list - iterate over values
    for nuc in data["nuc"].values():
        if min_prob is not None and nuc.get("type_prob", 1.0) < min_prob:
            continue
        counts[nuc["type"]] += 1

    total = sum(counts.values())
    if total == 0:
        return np.zeros(len(CELL_TYPES))

    return np.array([counts[t] / total for t in CELL_TYPES])


# --------------------------------------------------
# Load all tiles
# --------------------------------------------------
tile_names = []
tile_vectors = []

for fname in sorted(os.listdir(JSON_DIR)):
    if fname.endswith(".json"):
        path = os.path.join(JSON_DIR, fname)
        tile_names.append(fname)
        tile_vectors.append(load_tile_proportions(path))

X = np.vstack(tile_vectors)

# --------------------------------------------------
# Bray–Curtis pairwise dissimilarity
# --------------------------------------------------
dist_condensed = pdist(X, metric="braycurtis")
dist_matrix = squareform(dist_condensed)

# --------------------------------------------------
# Output as DataFrame
# --------------------------------------------------
bc_df = pd.DataFrame(
    dist_matrix,
    index=tile_names,
    columns=tile_names
)

print(bc_df)

# Optional: save to CSV
bc_df.to_csv("quantitative_analysis/bray_curtis_dissimilarity.csv")