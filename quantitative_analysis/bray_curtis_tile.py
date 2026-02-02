import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend: no figure window pops up
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------
# Config
# --------------------------------------------------
JSON_DIR = r"/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred/overall"
CELL_TYPES = list(range(7))  # 0–6 as defined by NucSegAI

# Path to tile categories JSON (defines the 5 groups: bg, margin, tumour_inv, tumour_lep, tumour_scar)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TILE_CATEGORIES_JSON = os.path.join(
    PROJECT_ROOT, "neighborhood_composition", "spatial_contexts", "tile_categories_88_tiles.json"
)

# Visualization parameters
SHOW_TILE_NAMES = False  # Set to True to display tile names on axes
SHOW_GROUP_NAMES_ONLY = True  # Set to True to show only group names (one per cluster) instead of individual tile names
SHOW_BC_VALUES = False  # Set to True to display Bray-Curtis dissimilarity values in each cell of the heatmap

# --------------------------------------------------
# Helper: load tile categories and build ordered tile list
# --------------------------------------------------
GROUP_ORDER = ['bg', 'margin', 'tumour_inv', 'tumour_lep', 'tumour_scar']


def load_tile_categories(path):
    """
    Load tile_categories_88_tiles.json and return:
      - tile_id_to_group: dict mapping tile_id (no .json) -> group name
      - ordered_tile_ids: list of tile IDs in display order (5 clusters, one per group)
    """
    with open(path, "r") as f:
        data = json.load(f)

    tile_id_to_group = {}
    ordered_tile_ids = []

    for group in GROUP_ORDER:
        if group not in data or not isinstance(data[group], list):
            continue
        for tile_id in data[group]:
            tile_id_to_group[tile_id] = group
            ordered_tile_ids.append(tile_id)

    return tile_id_to_group, ordered_tile_ids


# --------------------------------------------------
# Helper: simplify tile name for display
# --------------------------------------------------
def simplify_tile_name(fname, group_name):
    """
    Simplify tile filename for display.
    Example: "JN_TS_005_tumour_inv_tile_13587_10688.json" -> "13587_10688_JN_TS_005"
    """
    # Remove .json extension
    name = fname.replace(".json", "")
    
    # Pattern: {prefix}_{group_name}_tile_{num1}_{num2}
    # Extract: prefix (e.g., JN_TS_005) and numbers (e.g., 13587_10688)
    pattern = f"(.+?)_{group_name}_tile_(\\d+_\\d+)"
    match = re.match(pattern, name)
    
    if match:
        prefix = match.group(1)  # e.g., "JN_TS_005"
        numbers = match.group(2)  # e.g., "13587_10688"
        return f"{numbers}_{prefix}_{group_name}"
    else:
        # Fallback: if pattern doesn't match, return original without .json
        return name

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
# Load tile categories (5 groups)
# --------------------------------------------------
tile_id_to_group, ordered_tile_ids = load_tile_categories(TILE_CATEGORIES_JSON)

# --------------------------------------------------
# Load all tiles from JSON_DIR
# --------------------------------------------------
tile_by_id = {}  # tile_id -> (fname, vector)

for fname in os.listdir(JSON_DIR):
    if not fname.endswith(".json"):
        continue
    tile_id = fname.replace(".json", "")
    if tile_id not in tile_id_to_group:
        continue
    path = os.path.join(JSON_DIR, fname)
    tile_by_id[tile_id] = (fname, load_tile_proportions(path))

# Keep only tiles that are in both categories and JSON_DIR, in categories order
sorted_tile_names = []
sorted_tile_vectors = []
sorted_tile_groups = []

for tile_id in ordered_tile_ids:
    if tile_id not in tile_by_id:
        continue
    fname, vec = tile_by_id[tile_id]
    sorted_tile_names.append(fname)
    sorted_tile_vectors.append(vec)
    sorted_tile_groups.append(tile_id_to_group[tile_id])

X_sorted = np.vstack(sorted_tile_vectors)
num_tiles = len(sorted_tile_names)
group_name = os.path.basename(JSON_DIR)

# Print group clustering info
print("\nTile clustering by group (from tile_categories_88_tiles.json):")
current_group = None
group_start_idx = 0
for i, group in enumerate(sorted_tile_groups):
    if group != current_group:
        if current_group is not None:
            print(f"  {current_group}: tiles {group_start_idx} to {i-1} ({i - group_start_idx} tiles)")
        current_group = group
        group_start_idx = i
if current_group is not None:
    n_last = len(sorted_tile_groups) - group_start_idx
    print(f"  {current_group}: tiles {group_start_idx} to {len(sorted_tile_groups)-1} ({n_last} tiles)")
print()

# --------------------------------------------------
# Bray–Curtis pairwise dissimilarity (using sorted data)
# --------------------------------------------------
dist_condensed = pdist(X_sorted, metric="braycurtis")
dist_matrix = squareform(dist_condensed)

# Calculate mean and median (excluding diagonal)
# Get upper triangle (excluding diagonal)
upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
mean_bc = np.mean(upper_triangle)
median_bc = np.median(upper_triangle)

# --------------------------------------------------
# Output as DataFrame
# --------------------------------------------------
bc_df = pd.DataFrame(
    dist_matrix,
    index=sorted_tile_names,
    columns=sorted_tile_names
)

print(bc_df)
print(f"\nMean BC: {mean_bc:.4f}, Median BC: {median_bc:.4f}")

# --------------------------------------------------
# Visualize as colored table (heatmap)
# --------------------------------------------------
# Create simplified tile names for display (using each tile's own group name)
simplified_tile_names = [simplify_tile_name(name, tile_group) 
                         for name, tile_group in zip(sorted_tile_names, sorted_tile_groups)]

# Create custom colormap: blue (0) -> white (0.5) -> red (1)
colors = ['#0000FF', '#FFFFFF', '#FF0000']  # Blue -> White -> Red
n_bins = 256
cmap = LinearSegmentedColormap.from_list('blue_white_red', colors, N=n_bins)

# Prepare tick labels based on display mode
if SHOW_GROUP_NAMES_ONLY:
    # Calculate group boundaries and middle positions
    group_boundaries = []
    current_group = None
    group_start_idx = 0
    
    for i, group in enumerate(sorted_tile_groups):
        if group != current_group:
            if current_group is not None:
                group_boundaries.append((group_start_idx, i - 1, current_group))
            current_group = group
            group_start_idx = i
    # Add final group
    if current_group is not None:
        group_boundaries.append((group_start_idx, len(sorted_tile_groups) - 1, current_group))
    
    # Create labels: group name at middle of cluster, empty elsewhere
    x_labels = [''] * len(sorted_tile_names)
    y_labels = [''] * len(sorted_tile_names)
    
    for start_idx, end_idx, group_name in group_boundaries:
        middle_idx = (start_idx + end_idx) // 2
        x_labels[middle_idx] = group_name
        y_labels[middle_idx] = group_name
    
    x_ticklabels = x_labels
    y_ticklabels = y_labels
elif SHOW_TILE_NAMES:
    x_ticklabels = simplified_tile_names
    y_ticklabels = simplified_tile_names
else:
    x_ticklabels = False
    y_ticklabels = False

# Set up the plot
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    bc_df,
    cmap=cmap,
    vmin=0,
    vmax=1,
    annot=SHOW_BC_VALUES,  # Show BC values in each cell if enabled
    fmt='.3f',
    cbar_kws={'label': 'Bray-Curtis Dissimilarity'},
    square=True,
    linewidths=0.5,
    linecolor='gray',
    xticklabels=x_ticklabels,
    yticklabels=y_ticklabels
)

# Add visual separators between groups
current_group = None
group_start_idx = 0
for i, group in enumerate(sorted_tile_groups):
    if group != current_group:
        if current_group is not None:
            # Draw separator lines between groups
            ax.axhline(y=group_start_idx, color='black', linewidth=2)
            ax.axvline(x=group_start_idx, color='black', linewidth=2)
        current_group = group
        group_start_idx = i
# Draw final separator
if len(sorted_tile_groups) > 0:
    ax.axhline(y=len(sorted_tile_groups), color='black', linewidth=2)
    ax.axvline(x=len(sorted_tile_groups), color='black', linewidth=2)

title = f'Bray-Curtis Dissimilarity Matrix: {group_name} (n = {num_tiles} tiles)\nMean: {mean_bc:.4f}; Median: {median_bc:.4f};'
plt.title(title, fontsize=14, fontweight='bold')
plt.xlabel('Tiles', fontsize=12)
plt.ylabel('Tiles', fontsize=12)

# Set label rotation and font size based on display mode
if SHOW_GROUP_NAMES_ONLY:
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
elif SHOW_TILE_NAMES:
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

plt.tight_layout()

# Create output directory if it doesn't exist
output_dir = os.path.join(PROJECT_ROOT, "quantitative_analysis", "bray_curtis")
os.makedirs(output_dir, exist_ok=True)

# Save the figure
if SHOW_GROUP_NAMES_ONLY:
    output_filename = os.path.join(output_dir, f"bray_curtis_overall_{num_tiles}_heatmap.png")
else:
    output_filename = os.path.join(output_dir, f"bray_curtis_{group_name}_{num_tiles}_heatmap.png")
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nHeatmap saved to: {output_filename}")

# plt.show()  # Omitted when using Agg backend to avoid figure popup

# Optional: save to CSV
if SHOW_GROUP_NAMES_ONLY:
    csv_filename = os.path.join(output_dir, f"bray_curtis_overall_{num_tiles}_dissimilarity.csv")
else:
    csv_filename = os.path.join(output_dir, f"bray_curtis_{group_name}_{num_tiles}_dissimilarity.csv")
bc_df.to_csv(csv_filename)