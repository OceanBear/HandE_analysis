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
from collections import defaultdict

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
SHOW_TILE_NAMES = True  # Set to True to display tile names on axes
SHOW_GROUP_NAMES_ONLY = False  # Set to True to show only group names (one per cluster) instead of individual tile names
SHOW_BC_VALUES = True  # Set to True to display Bray-Curtis dissimilarity values in each cell of the heatmap

# --------------------------------------------------
# Helper: extract case ID from tile filename
# --------------------------------------------------
def extract_case_id(tile_id):
    """
    Extract case ID from tile ID.
    Example: "JN_TS_001_tumour_inv_tile_10912_14661" -> "JN_TS_001"
    """
    # Pattern: {case_id}_{group}_tile_{num1}_{num2}
    match = re.match(r"(.+?)_(tumour_inv|tumour_lep|bg|margin|tumour_scar)_tile_", tile_id)
    if match:
        return match.group(1)
    return None


# --------------------------------------------------
# Helper: extract group name from tile ID
# --------------------------------------------------
def extract_group_name(tile_id):
    """
    Extract group name from tile ID.
    Example: "JN_TS_001_tumour_inv_tile_10912_14661" -> "tumour_inv"
    """
    match = re.match(r".+?_(tumour_inv|tumour_lep|bg|margin|tumour_scar)_tile_", tile_id)
    if match:
        return match.group(1)
    return None


# --------------------------------------------------
# Helper: simplify tile name for display
# --------------------------------------------------
def simplify_tile_name(fname, group_name):
    """
    Simplify tile filename for display.
    Example: "JN_TS_010_tumour_inv_tile_11151_13664.json" -> "11151_13664_tumour_inv"
    """
    # Remove .json extension
    name = fname.replace(".json", "")
    
    # Pattern: {prefix}_{group_name}_tile_{num1}_{num2}
    # Extract: numbers (e.g., 11151_13664) and group_name (e.g., tumour_inv)
    pattern = f".+?_{group_name}_tile_(\\d+_\\d+)"
    match = re.match(pattern, name)
    
    if match:
        numbers = match.group(1)  # e.g., "11151_13664"
        return f"{numbers}_{group_name}"
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
# Load all tiles from JSON_DIR and group by case
# --------------------------------------------------
tiles_by_case = defaultdict(list)  # case_id -> list of (tile_id, fname, group)

for fname in os.listdir(JSON_DIR):
    if not fname.endswith(".json"):
        continue
    tile_id = fname.replace(".json", "")
    group = extract_group_name(tile_id)
    
    # Only consider tumour_inv and tumour_lep
    if group not in ['tumour_inv', 'tumour_lep']:
        continue
    
    case_id = extract_case_id(tile_id)
    if case_id is None:
        continue
    
    tiles_by_case[case_id].append((tile_id, fname, group))

# Sort cases
case_ids = sorted(tiles_by_case.keys())

# --------------------------------------------------
# Process each case
# --------------------------------------------------
output_dir = os.path.join(PROJECT_ROOT, "quantitative_analysis", "bray_curtis_case")
os.makedirs(output_dir, exist_ok=True)

for case_id in case_ids:
    tiles = tiles_by_case[case_id]
    
    # Sort tiles: tumour_inv first, then tumour_lep
    tiles.sort(key=lambda x: (0 if x[2] == 'tumour_inv' else 1, x[0]))
    
    # Load tile vectors
    tile_names = []
    tile_vectors = []
    tile_groups = []
    
    for tile_id, fname, group in tiles:
        path = os.path.join(JSON_DIR, fname)
        tile_names.append(fname)
        tile_vectors.append(load_tile_proportions(path))
        tile_groups.append(group)
    
    if len(tile_names) == 0:
        print(f"\nSkipping {case_id}: no tumour_inv or tumour_lep tiles found")
        continue
    
    X = np.vstack(tile_vectors)
    num_tiles = len(tile_names)
    
    print(f"\nProcessing case: {case_id}")
    print(f"  Number of tiles: {num_tiles}")
    
    # Count tiles per group
    tumour_inv_count = sum(1 for g in tile_groups if g == 'tumour_inv')
    tumour_lep_count = sum(1 for g in tile_groups if g == 'tumour_lep')
    print(f"  tumour_inv: {tumour_inv_count}, tumour_lep: {tumour_lep_count}")
    
    # --------------------------------------------------
    # Bray–Curtis pairwise dissimilarity
    # --------------------------------------------------
    dist_condensed = pdist(X, metric="braycurtis")
    dist_matrix = squareform(dist_condensed)
    
    # Calculate mean and median (excluding diagonal)
    # Get upper triangle (excluding diagonal)
    upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    mean_bc = np.mean(upper_triangle)
    median_bc = np.median(upper_triangle)
    
    print(f"  Mean BC: {mean_bc:.4f}, Median BC: {median_bc:.4f}")
    
    # --------------------------------------------------
    # Output as DataFrame
    # --------------------------------------------------
    bc_df = pd.DataFrame(
        dist_matrix,
        index=tile_names,
        columns=tile_names
    )
    
    # --------------------------------------------------
    # Visualize as colored table (heatmap)
    # --------------------------------------------------
    # Create simplified tile names for display
    simplified_tile_names = [simplify_tile_name(name, tile_group) 
                             for name, tile_group in zip(tile_names, tile_groups)]
    
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
        
        for i, group in enumerate(tile_groups):
            if group != current_group:
                if current_group is not None:
                    group_boundaries.append((group_start_idx, i - 1, current_group))
                current_group = group
                group_start_idx = i
        # Add final group
        if current_group is not None:
            group_boundaries.append((group_start_idx, len(tile_groups) - 1, current_group))
        
        # Create labels: group name at middle of cluster, empty elsewhere
        x_labels = [''] * len(tile_names)
        y_labels = [''] * len(tile_names)
        
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
    for i, group in enumerate(tile_groups):
        if group != current_group:
            if current_group is not None:
                # Draw separator lines between groups
                ax.axhline(y=group_start_idx, color='black', linewidth=2)
                ax.axvline(x=group_start_idx, color='black', linewidth=2)
            current_group = group
            group_start_idx = i
    # Draw final separator
    if len(tile_groups) > 0:
        ax.axhline(y=len(tile_groups), color='black', linewidth=2)
        ax.axvline(x=len(tile_groups), color='black', linewidth=2)
    
    # Title with mean and median
    title = f'Bray-Curtis Dissimilarity Matrix: {case_id} (n = {num_tiles} tiles)\nMean: {mean_bc:.4f}; Median: {median_bc:.4f};'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Tiles', fontsize=12)
    plt.ylabel('Tiles', fontsize=12)
    
    # Set label rotation and font size based on display mode
    if SHOW_GROUP_NAMES_ONLY:
        plt.xticks(rotation=0, ha='center', fontsize=10, fontweight='bold')
        plt.yticks(rotation=0, fontsize=10, fontweight='bold')
    elif SHOW_TILE_NAMES:
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
    
    plt.tight_layout()
    
    # Save the figure
    output_filename = os.path.join(output_dir, f"bray_curtis_{case_id}_{num_tiles}_heatmap.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"  Heatmap saved to: {output_filename}")
    plt.close()  # Close figure to free memory
    
    # Save to CSV
    csv_filename = os.path.join(output_dir, f"bray_curtis_{case_id}_{num_tiles}_dissimilarity.csv")
    bc_df.to_csv(csv_filename)
    print(f"  CSV saved to: {csv_filename}")

print(f"\nProcessing complete. Processed {len(case_ids)} cases.")
