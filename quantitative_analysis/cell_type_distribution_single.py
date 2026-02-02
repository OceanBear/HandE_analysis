"""
NucSegAI Result Analysis - Cell Type Quantification (Single Tile)

This script analyzes a single NucSegAI JSON output file to extract:
- Cell counts by type
- Cell type proportions
- Cell density by type (cells/tile and cells/mm²)
- Type probability statistics

Output: CSV files and figures saved to quantitative_analysis/ctd_single/
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
# ⚠️ Update this path to your specific input JSON file
INPUT_FILE = "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/Batch_105/pred/json/JN_TS_006_tile_3175_8574.json"

# Output directory
OUTPUT_DIR = Path("quantitative_analysis/ctd_single")

# Threshold for confidence filtering
CONFIDENCE_THRESHOLD = 0.5

# Tile physical size configuration (mm²)
TILE_AREA_MM2 = 4.0

# Cell type mapping
CELL_TYPE_DICT = {
    0: "Undefined",
    1: "Epithelium (PD-L1lo/Ki67lo)",
    2: "Epithelium (PD-L1hi/Ki67hi)",
    3: "Macrophage",
    4: "Lymphocyte",
    5: "Vascular",
    6: "Fibroblast/Stroma"
}

# Color mapping for visualization (converted from RGB to hex)
CELL_TYPE_COLORS = {
    0: "#000000",  # Black - Undefined
    1: "#387F39",  # Dark Green - Epithelium low
    2: "#00FF00",  # Bright Green - Epithelium high
    3: "#FC8D62",  # Coral/Salmon - Macrophage
    4: "#FFD92F",  # Yellow - Lymphocyte
    5: "#4535C1",  # Blue/Purple - Vascular
    6: "#17BECF"   # Cyan - Fibroblast/Stroma
}

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def analyze_single_json(json_path, tile_area_mm2=TILE_AREA_MM2):
    """
    Analyze a single NucSegAI JSON output file.

    Args:
        json_path (str or Path): Path to JSON file
        tile_area_mm2 (float): Area of each tile in square millimeters

    Returns:
        dict: Analysis results including cell counts, proportions, density, type probability stats
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    nuclei = data.get('nuc', {})

    if not nuclei:
        return {
            'filename': Path(json_path).name,
            'total_cells': 0,
            'num_tiles': 1,
            'tile_area_mm2': tile_area_mm2,
            'cell_counts': {},
            'cell_proportions': {},
            'cell_density_per_tile': {},
            'cell_density_per_mm2': {},
            'type_prob_stats_overall': {},
            'type_prob_stats_by_type': {}
        }

    cell_types = []
    type_probs = []
    type_probs_by_type = defaultdict(list)

    for nucleus_id, nucleus_data in nuclei.items():
        cell_type = nucleus_data.get('type', 0)
        type_prob = nucleus_data.get('type_prob', 0)
        cell_types.append(cell_type)
        type_probs.append(type_prob)
        type_probs_by_type[cell_type].append(type_prob)

    cell_counts = Counter(cell_types)
    total_cells = len(cell_types)
    num_tiles = 1

    cell_proportions = {
        ct: count / total_cells if total_cells > 0 else 0
        for ct, count in cell_counts.items()
    }
    cell_density_per_tile = {ct: count / num_tiles for ct, count in cell_counts.items()}
    cell_density_per_mm2 = {
        ct: count / tile_area_mm2 if tile_area_mm2 > 0 else 0
        for ct, count in cell_counts.items()
    }

    type_prob_stats_overall = {
        'min': np.min(type_probs) if type_probs else 0,
        'median': np.median(type_probs) if type_probs else 0,
        'mean': np.mean(type_probs) if type_probs else 0,
        'max': np.max(type_probs) if type_probs else 0,
        'std': np.std(type_probs) if type_probs else 0
    }

    type_prob_stats_by_type = {
        ct: {
            'min': np.min(probs),
            'median': np.median(probs),
            'mean': np.mean(probs),
            'max': np.max(probs),
            'std': np.std(probs)
        }
        for ct, probs in type_probs_by_type.items()
    }

    return {
        'filename': Path(json_path).name,
        'total_cells': total_cells,
        'num_tiles': num_tiles,
        'tile_area_mm2': tile_area_mm2,
        'cell_counts': dict(cell_counts),
        'cell_proportions': cell_proportions,
        'cell_density_per_tile': cell_density_per_tile,
        'cell_density_per_mm2': cell_density_per_mm2,
        'type_prob_stats_overall': type_prob_stats_overall,
        'type_prob_stats_by_type': type_prob_stats_by_type
    }


def apply_confidence_filter(json_path, threshold=0.5, tile_area_mm2=TILE_AREA_MM2):
    """
    Apply confidence threshold filter to reclassify low-confidence predictions as Undefined.

    Args:
        json_path (str or Path): Path to JSON file
        threshold (float): Minimum type probability threshold
        tile_area_mm2 (float): Area of each tile in square millimeters

    Returns:
        dict: Filtered analysis results
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    nuclei = data.get('nuc', {})

    if not nuclei:
        return {
            'filename': Path(json_path).name,
            'threshold': threshold,
            'total_cells': 0,
            'num_tiles': 1,
            'tile_area_mm2': tile_area_mm2,
            'reclassified_count': 0,
            'cell_counts': {},
            'cell_proportions': {},
            'cell_density_per_tile': {},
            'cell_density_per_mm2': {},
            'type_prob_stats_overall': {},
            'type_prob_stats_by_type': {},
            'original_cell_counts': {}
        }

    original_types = []
    filtered_types = []
    type_probs = []
    type_probs_by_type = defaultdict(list)
    reclassified_count = 0

    for nucleus_id, nucleus_data in nuclei.items():
        original_type = nucleus_data.get('type', 0)
        type_prob = nucleus_data.get('type_prob', 0)
        original_types.append(original_type)

        if type_prob < threshold and original_type != 0:
            filtered_type = 0
            reclassified_count += 1
        else:
            filtered_type = original_type

        filtered_types.append(filtered_type)
        type_probs.append(type_prob)
        type_probs_by_type[filtered_type].append(type_prob)

    cell_counts = Counter(filtered_types)
    total_cells = len(filtered_types)
    num_tiles = 1

    cell_proportions = {
        ct: count / total_cells if total_cells > 0 else 0
        for ct, count in cell_counts.items()
    }
    cell_density_per_tile = {ct: count / num_tiles for ct, count in cell_counts.items()}
    cell_density_per_mm2 = {
        ct: count / tile_area_mm2 if tile_area_mm2 > 0 else 0
        for ct, count in cell_counts.items()
    }

    type_prob_stats_overall = {
        'min': np.min(type_probs) if type_probs else 0,
        'median': np.median(type_probs) if type_probs else 0,
        'mean': np.mean(type_probs) if type_probs else 0,
        'max': np.max(type_probs) if type_probs else 0,
        'std': np.std(type_probs) if type_probs else 0
    }

    type_prob_stats_by_type = {
        ct: {
            'min': np.min(probs),
            'median': np.median(probs),
            'mean': np.mean(probs),
            'max': np.max(probs),
            'std': np.std(probs)
        }
        for ct, probs in type_probs_by_type.items()
    }

    tile_name = Path(json_path).stem
    return {
        'filename': f"{tile_name}_threshold_{int(threshold*100)}.json",
        'threshold': threshold,
        'total_cells': total_cells,
        'num_tiles': num_tiles,
        'tile_area_mm2': tile_area_mm2,
        'reclassified_count': reclassified_count,
        'cell_counts': dict(cell_counts),
        'cell_proportions': cell_proportions,
        'cell_density_per_tile': cell_density_per_tile,
        'cell_density_per_mm2': cell_density_per_mm2,
        'type_prob_stats_overall': type_prob_stats_overall,
        'type_prob_stats_by_type': type_prob_stats_by_type,
        'original_cell_counts': dict(Counter(original_types))
    }


def display_results(results):
    """Display analysis results in a formatted way."""
    print("=" * 80)
    print(f"Analysis Results for: {results['filename']}")
    print("=" * 80)
    print(f"\n📊 Total Cells Detected: {results['total_cells']:,}")
    if 'num_tiles' in results:
        print(f"📊 Number of Tiles: {results['num_tiles']:,}")
    if 'tile_area_mm2' in results:
        print(f"📊 Tile Area: {results['tile_area_mm2']} mm²")
    print()

    print("─" * 80)
    print("Cell Type Distribution with Density")
    print("─" * 80)
    print(f"{'Cell Type':<35} {'Count':>12} {'Proportion':>12} {'Percentage':>10} {'Cells/Tile':>12} {'Cells/mm²':>12}")
    print("─" * 80)

    for cell_type in sorted(results['cell_counts'].keys()):
        count = results['cell_counts'][cell_type]
        proportion = results['cell_proportions'][cell_type]
        percentage = proportion * 100
        type_name = CELL_TYPE_DICT.get(cell_type, f"Unknown ({cell_type})")
        density_per_tile = results.get('cell_density_per_tile', {}).get(cell_type, 0)
        density_per_mm2 = results.get('cell_density_per_mm2', {}).get(cell_type, 0)
        print(f"{type_name:<35} {count:>12,} {proportion:>12.4f} {percentage:>9.2f}% {density_per_tile:>12.2f} {density_per_mm2:>12.2f}")

    if results.get('type_prob_stats_overall'):
        print("\n" + "─" * 80)
        print("Overall Type Probability Statistics")
        print("─" * 80)
        stats = results['type_prob_stats_overall']
        print(f"  Minimum:  {stats['min']:.10f}")
        print(f"  Median:   {stats['median']:.10f}")
        print(f"  Mean:     {stats['mean']:.10f}")
        print(f"  Maximum:  {stats['max']:.10f}")
        print(f"  Std Dev:  {stats['std']:.10f}")

    if results.get('type_prob_stats_by_type'):
        print("\n" + "─" * 80)
        print("Type Probability Statistics by Cell Type")
        print("─" * 80)
        for cell_type in sorted(results['type_prob_stats_by_type'].keys()):
            type_name = CELL_TYPE_DICT.get(cell_type, f"Unknown ({cell_type})")
            stats = results['type_prob_stats_by_type'][cell_type]
            print(f"\n{type_name}:")
            print(f"  Min: {stats['min']:.6f} | Med: {stats['median']:.6f} | Mean: {stats['mean']:.6f} | Max: {stats['max']:.6f} | Std: {stats['std']:.6f}")

    print("\n" + "=" * 80)


def plot_cell_type_distribution(results, output_path=None):
    """
    Create visualizations for cell type distribution (bar chart + pie chart).
    Saves to output_path if provided, otherwise does nothing (no display).
    """
    if not results['cell_counts']:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cell_types = sorted(results['cell_counts'].keys())
    cell_names = [CELL_TYPE_DICT.get(ct, f"Type {ct}") for ct in cell_types]
    counts = [results['cell_counts'][ct] for ct in cell_types]
    colors = [CELL_TYPE_COLORS.get(ct, "#808080") for ct in cell_types]

    # Bar plot
    axes[0].bar(range(len(cell_types)), counts, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Cell Type', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Cell Counts by Type\n{results["filename"]}', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(cell_types)))
    axes[0].set_xticklabels(cell_names, rotation=45, ha='right', rotation_mode='anchor')
    axes[0].grid(axis='y', alpha=0.3)
    for i, count in enumerate(counts):
        axes[0].text(i, count, f'{count:,}', ha='center', va='bottom', fontweight='bold')

    # Pie chart
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct > 3 else ''

    wedges, texts, autotexts = axes[1].pie(
        counts, labels=None, colors=colors, autopct=autopct_format,
        startangle=90, textprops={'fontsize': 10, 'weight': 'bold', 'color': 'black'},
        pctdistance=0.85
    )
    for t in autotexts:
        t.set_color('black')
        t.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='white')])

    legend_labels = [f'{name}: {count:,} ({count/sum(counts)*100:.1f}%)' for name, count in zip(cell_names, counts)]
    axes[1].legend(wedges, legend_labels, title="Cell Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    axes[1].set_title(f'Cell Type Proportions\n{results["filename"]}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_type_probability_distribution(results, output_path=None):
    """
    Create bar plot for type probabilities by cell type.
    Saves to output_path if provided.
    """
    if not results.get('type_prob_stats_by_type'):
        return
    fig, ax = plt.subplots(figsize=(14, 6))

    cell_types = sorted(results['type_prob_stats_by_type'].keys())
    cell_names = [CELL_TYPE_DICT.get(ct, f"Type {ct}") for ct in cell_types]
    means = [results['type_prob_stats_by_type'][ct]['mean'] for ct in cell_types]
    stds = [results['type_prob_stats_by_type'][ct]['std'] for ct in cell_types]
    colors_list = [CELL_TYPE_COLORS.get(ct, "#808080") for ct in cell_types]

    ax.bar(range(len(cell_types)), means, yerr=stds, color=colors_list,
           edgecolor='black', linewidth=1.5, capsize=5, alpha=0.8)
    ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Type Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'Type Probability by Cell Type (Mean ± Std Dev)\n{results["filename"]}',
                 fontsize=14, fontweight='bold', pad=30)
    ax.set_xticks(range(len(cell_types)))
    ax.set_xticklabels(cell_names, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def export_results_to_csv(results, output_path, is_filtered=False):
    """Export analysis results to CSV file."""
    data_rows = []
    for cell_type in sorted(results['cell_counts'].keys()):
        type_name = CELL_TYPE_DICT.get(cell_type, f"Type {cell_type}")
        count = results['cell_counts'][cell_type]
        proportion = results['cell_proportions'][cell_type]
        stats = results['type_prob_stats_by_type'][cell_type]
        density_per_tile = results.get('cell_density_per_tile', {}).get(cell_type, 0)
        density_per_mm2 = results.get('cell_density_per_mm2', {}).get(cell_type, 0)

        row_data = {
            'Filename': results['filename'],
            'Cell_Type_ID': cell_type,
            'Cell_Type_Name': type_name,
            'Count': count,
            'Proportion': proportion,
            'Percentage': proportion * 100,
            'Cells_Per_Tile': density_per_tile,
            'Cells_Per_mm2': density_per_mm2,
            'TypeProb_Min': stats['min'],
            'TypeProb_Median': stats['median'],
            'TypeProb_Mean': stats['mean'],
            'TypeProb_Max': stats['max'],
            'TypeProb_Std': stats['std']
        }
        if is_filtered and 'original_cell_counts' in results:
            row_data['Original_Count'] = results['original_cell_counts'].get(cell_type, 0)
            row_data['Count_Change'] = count - results['original_cell_counts'].get(cell_type, 0)
            row_data['Threshold'] = results.get('threshold', 'N/A')
        data_rows.append(row_data)

    df = pd.DataFrame(data_rows)
    if is_filtered and 'original_cell_counts' in results:
        column_order = ['Filename', 'Cell_Type_ID', 'Cell_Type_Name', 'Original_Count', 'Count', 'Count_Change',
                        'Proportion', 'Percentage', 'Cells_Per_Tile', 'Cells_Per_mm2',
                        'TypeProb_Min', 'TypeProb_Median', 'TypeProb_Mean', 'TypeProb_Max', 'TypeProb_Std', 'Threshold']
    else:
        column_order = ['Filename', 'Cell_Type_ID', 'Cell_Type_Name', 'Count', 'Proportion', 'Percentage',
                        'Cells_Per_Tile', 'Cells_Per_mm2',
                        'TypeProb_Min', 'TypeProb_Median', 'TypeProb_Mean', 'TypeProb_Max', 'TypeProb_Std']
    df = df[[c for c in column_order if c in df.columns]]
    df.to_csv(output_path, index=False)
    return df


def export_density_summary_to_csv(results, output_path, is_filtered=False):
    """Export cell density metrics (cells/tile and cells/mm²) by cell type to CSV."""
    data_rows = []
    for cell_type in sorted(results['cell_counts'].keys()):
        type_name = CELL_TYPE_DICT.get(cell_type, f"Type {cell_type}")
        count = results['cell_counts'][cell_type]
        density_per_tile = results.get('cell_density_per_tile', {}).get(cell_type, 0)
        density_per_mm2 = results.get('cell_density_per_mm2', {}).get(cell_type, 0)
        row_data = {
            'Filename': results['filename'],
            'Cell_Type_ID': cell_type,
            'Cell_Type_Name': type_name,
            'Count': count,
            'Cells_Per_Tile': density_per_tile,
            'Cells_Per_mm2': density_per_mm2,
            'Tile_Area_mm2': results.get('tile_area_mm2', TILE_AREA_MM2)
        }
        if 'num_tiles' in results:
            row_data['Num_Tiles'] = results['num_tiles']
        if is_filtered and 'threshold' in results:
            row_data['Threshold'] = results.get('threshold', 'N/A')
        data_rows.append(row_data)

    df = pd.DataFrame(data_rows)
    base_columns = ['Filename', 'Cell_Type_ID', 'Cell_Type_Name', 'Count', 'Cells_Per_Tile', 'Cells_Per_mm2', 'Tile_Area_mm2']
    if 'num_tiles' in results:
        base_columns.insert(-1, 'Num_Tiles')
    if is_filtered and 'threshold' in results:
        base_columns.append('Threshold')
    df = df[[c for c in base_columns if c in df.columns]]
    df.to_csv(output_path, index=False)
    return df


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = Path(INPUT_FILE)
    if not json_path.exists():
        print(f"❌ Error: File not found at {INPUT_FILE}")
        print("Please update the INPUT_FILE variable with the correct path.")
        return

    print(f"✅ File found: {INPUT_FILE}")
    print(f"📁 Output directory: {output_dir}\n")

    # Step 1: Analyze unfiltered
    results = analyze_single_json(json_path)
    print("📊 Unfiltered analysis complete")

    # Step 2: Apply confidence filter
    filtered_results = apply_confidence_filter(json_path, threshold=CONFIDENCE_THRESHOLD)
    print(f"📊 Filtered analysis complete (threshold={CONFIDENCE_THRESHOLD})")
    print(f"   Reclassified {filtered_results['reclassified_count']:,} cells as Undefined\n")

    # Step 3: Display results
    display_results(filtered_results)
    display_results(results)

    # Step 4: Save figures (no display)
    print("\n📊 Saving figures...")
    plot_cell_type_distribution(results, output_dir / "cell_type_distribution_unfiltered.png")
    plot_cell_type_distribution(filtered_results, output_dir / "cell_type_distribution_filtered.png")
    plot_type_probability_distribution(results, output_dir / "type_probability_by_cell_type_unfiltered.png")
    plot_type_probability_distribution(filtered_results, output_dir / "type_probability_by_cell_type_filtered.png")
    print("   ✅ cell_type_distribution_unfiltered.png")
    print("   ✅ cell_type_distribution_filtered.png")
    print("   ✅ type_probability_by_cell_type_unfiltered.png")
    print("   ✅ type_probability_by_cell_type_filtered.png")

    # Step 5: Export CSV files
    threshold_str = str(int(CONFIDENCE_THRESHOLD * 100))
    print("\n📁 Exporting CSV files...")
    export_results_to_csv(results, output_dir / "cell_analysis_unfiltered.csv", is_filtered=False)
    export_results_to_csv(filtered_results, output_dir / f"cell_analysis_filtered_{threshold_str}.csv", is_filtered=True)
    export_density_summary_to_csv(results, output_dir / "cell_density_by_type_unfiltered.csv", is_filtered=False)
    export_density_summary_to_csv(filtered_results, output_dir / f"cell_density_by_type_filtered_{threshold_str}.csv", is_filtered=True)
    print("   ✅ cell_analysis_unfiltered.csv")
    print(f"   ✅ cell_analysis_filtered_{threshold_str}.csv")
    print("   ✅ cell_density_by_type_unfiltered.csv")
    print(f"   ✅ cell_density_by_type_filtered_{threshold_str}.csv")

    print("\n" + "=" * 80)
    print("✅ All results saved to", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
