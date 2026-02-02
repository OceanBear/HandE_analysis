"""
NucSegAI Result Analysis - Cell Type Quantification (Batch)

This script processes all JSON files in a directory, aggregates results,
and produces the same output as the notebook's batch analysis.

Output: CSV files and figures saved to quantitative_analysis/ctd_batch/
"""

import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# Reuse functions and constants from single-tile module
from cell_type_distribution_single import (
    analyze_single_json,
    apply_confidence_filter,
    display_results,
    plot_cell_type_distribution,
    plot_type_probability_distribution,
    export_results_to_csv,
    export_density_summary_to_csv,
    TILE_AREA_MM2,
)

# --- Configuration ---
# ⚠️ Update this path to your directory containing JSON files
INPUT_DIR = "/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/Batch_105/pred/json"

# Output directory
OUTPUT_DIR = Path("quantitative_analysis/ctd_batch")

# Threshold for confidence filtering
CONFIDENCE_THRESHOLD = 0.5


def find_json_files(directory_path):
    """
    Find all JSON files in a directory.

    Parameters:
    -----------
    directory_path : str or Path
        Path to directory containing JSON files (or a single JSON file; parent dir will be used)

    Returns:
    --------
    list : List of Path objects for JSON files
    """
    path = Path(directory_path)
    if path.is_file():
        path = path.parent
    directory = path

    if not directory.exists():
        print(f"❌ Error: Directory not found at {directory_path}")
        return []

    json_files = sorted(directory.glob("*.json"))

    if not json_files:
        print(f"⚠️ No JSON files found in {directory}")
        return []

    print(f"✅ Found {len(json_files)} JSON files in {directory}")
    return json_files


def analyze_multiple_json_files(directory_path, apply_filter=False, threshold=0.5, tile_area_mm2=TILE_AREA_MM2):
    """
    Analyze all JSON files in a directory.

    Args:
        directory_path (str or Path): Path to directory containing JSON files
        apply_filter (bool): Whether to apply confidence threshold filter
        threshold (float): Confidence threshold (only used when apply_filter=True)
        tile_area_mm2 (float): Area of each tile in square millimeters

    Returns:
        list: List of result dictionaries, one per file
    """
    json_files = find_json_files(directory_path)

    if not json_files:
        return []

    all_results = []

    print(f"\n{'='*80}")
    print(f"Processing {len(json_files)} JSON files...")
    if apply_filter:
        print(f"Applying confidence threshold: {threshold}")
    print(f"{'='*80}\n")

    for i, json_file in enumerate(json_files, 1):
        try:
            if i % 10 == 0 or i == 1 or i == len(json_files):
                print(f"Processing file {i}/{len(json_files)}: {json_file.name}")

            if apply_filter:
                result = apply_confidence_filter(json_file, threshold=threshold, tile_area_mm2=tile_area_mm2)
            else:
                result = analyze_single_json(json_file, tile_area_mm2=tile_area_mm2)

            all_results.append(result)

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {str(e)}")
            continue

    print(f"\n✅ Successfully processed {len(all_results)} files")
    return all_results


def aggregate_results(results_list, is_filtered=False):
    """
    Aggregate results from multiple JSON files.

    Args:
        results_list (list): List of result dictionaries from analyze_multiple_json_files()
        is_filtered (bool): Whether results are filtered

    Returns:
        dict: Aggregated results across all files
    """
    if not results_list:
        return {
            'filename': 'Aggregated_0_files',
            'num_files': 0,
            'total_cells': 0,
            'num_tiles': 0,
            'tile_area_mm2': TILE_AREA_MM2,
            'cell_counts': {},
            'cell_proportions': {},
            'cell_density_per_tile': {},
            'cell_density_per_mm2': {},
            'type_prob_stats_overall': {},
            'type_prob_stats_by_type': {}
        }

    total_cells = 0
    total_reclassified = 0
    num_tiles = 0
    tile_area_mm2 = results_list[0].get('tile_area_mm2', TILE_AREA_MM2)
    cell_counts = Counter()
    original_cell_counts = Counter()
    weighted_stats = defaultdict(lambda: {'count': 0, 'sum_mean': 0, 'sum_sq': 0, 'min': float('inf'), 'max': 0})

    for result in results_list:
        total_cells += result['total_cells']
        num_tiles += result.get('num_tiles', 1)

        for cell_type, count in result['cell_counts'].items():
            cell_counts[cell_type] += count
            stats = result['type_prob_stats_by_type'][cell_type]
            ws = weighted_stats[cell_type]
            ws['count'] += count
            ws['sum_mean'] += stats['mean'] * count
            ws['sum_sq'] += (stats['std']**2 + stats['mean']**2) * count
            ws['min'] = min(ws['min'], stats['min'])
            ws['max'] = max(ws['max'], stats['max'])

        if is_filtered:
            if 'original_cell_counts' in result:
                for cell_type, count in result['original_cell_counts'].items():
                    original_cell_counts[cell_type] += count
            if 'reclassified_count' in result:
                total_reclassified += result['reclassified_count']

    cell_proportions = {ct: count / total_cells if total_cells > 0 else 0 for ct, count in cell_counts.items()}
    cell_density_per_tile = {ct: count / num_tiles if num_tiles > 0 else 0 for ct, count in cell_counts.items()}
    total_area_mm2 = num_tiles * tile_area_mm2
    cell_density_per_mm2 = {
        ct: count / total_area_mm2 if total_area_mm2 > 0 else 0
        for ct, count in cell_counts.items()
    }

    type_prob_stats_by_type = {}
    for cell_type, ws in weighted_stats.items():
        mean = ws['sum_mean'] / ws['count'] if ws['count'] > 0 else 0
        variance = (ws['sum_sq'] / ws['count']) - mean**2 if ws['count'] > 0 else 0
        std = np.sqrt(max(0, variance))
        type_prob_stats_by_type[cell_type] = {
            'min': ws['min'],
            'median': mean,
            'mean': mean,
            'max': ws['max'],
            'std': std
        }

    total_sum_mean = sum(ws['sum_mean'] for ws in weighted_stats.values())
    total_sum_sq = sum(ws['sum_sq'] for ws in weighted_stats.values())
    overall_mean = total_sum_mean / total_cells if total_cells > 0 else 0
    overall_variance = (total_sum_sq / total_cells) - overall_mean**2 if total_cells > 0 else 0
    type_prob_stats_overall = {
        'min': min(ws['min'] for ws in weighted_stats.values()) if weighted_stats else 0,
        'median': overall_mean,
        'mean': overall_mean,
        'max': max(ws['max'] for ws in weighted_stats.values()) if weighted_stats else 0,
        'std': np.sqrt(max(0, overall_variance))
    }

    num_files = len(results_list)
    threshold_str = f"_threshold_{int(results_list[0].get('threshold', 0.5) * 100)}" if is_filtered else ""
    filename = f'Aggregated_{num_files}_files{threshold_str}'

    aggregated = {
        'filename': filename,
        'num_files': num_files,
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

    if is_filtered:
        aggregated['reclassified_count'] = total_reclassified
        aggregated['threshold'] = results_list[0].get('threshold', 'N/A')
        aggregated['original_cell_counts'] = dict(original_cell_counts)

    return aggregated


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(INPUT_DIR)
    if input_path.is_file():
        input_path = input_path.parent
    if not input_path.exists():
        print(f"❌ Error: Directory not found at {INPUT_DIR}")
        return

    print(f"📁 Input directory: {input_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Confidence threshold: {CONFIDENCE_THRESHOLD}\n")

    # Analyze unfiltered
    all_results = analyze_multiple_json_files(input_path, apply_filter=False)
    aggregated_unfiltered = aggregate_results(all_results, is_filtered=False)

    # Analyze filtered
    filtered_results = analyze_multiple_json_files(
        input_path, apply_filter=True, threshold=CONFIDENCE_THRESHOLD
    )
    aggregated_filtered = aggregate_results(filtered_results, is_filtered=True)

    if not all_results and not filtered_results:
        print("⚠️ No results to process. Exiting.")
        return

    # Display results
    print("\n📊 Generating comparison visualizations...\n")
    display_results(aggregated_filtered)
    display_results(aggregated_unfiltered)

    # Save figures (no display)
    print("\n📊 Saving figures...")
    plot_cell_type_distribution(aggregated_filtered, output_dir / "cell_type_distribution_filtered.png")
    plot_cell_type_distribution(aggregated_unfiltered, output_dir / "cell_type_distribution_unfiltered.png")
    plot_type_probability_distribution(aggregated_filtered, output_dir / "type_probability_by_cell_type_filtered.png")
    plot_type_probability_distribution(aggregated_unfiltered, output_dir / "type_probability_by_cell_type_unfiltered.png")
    print("   ✅ cell_type_distribution_filtered.png")
    print("   ✅ cell_type_distribution_unfiltered.png")
    print("   ✅ type_probability_by_cell_type_filtered.png")
    print("   ✅ type_probability_by_cell_type_unfiltered.png")

    # Export CSV files
    threshold_str = str(int(CONFIDENCE_THRESHOLD * 100))
    print("\n📁 Exporting CSV files...")
    export_results_to_csv(
        aggregated_unfiltered,
        output_dir / "batch_aggregated_unfiltered.csv",
        is_filtered=False
    )
    export_results_to_csv(
        aggregated_filtered,
        output_dir / f"batch_aggregated_filtered_{threshold_str}.csv",
        is_filtered=True
    )
    export_density_summary_to_csv(
        aggregated_unfiltered,
        output_dir / "cell_density_by_type_unfiltered.csv",
        is_filtered=False
    )
    export_density_summary_to_csv(
        aggregated_filtered,
        output_dir / f"cell_density_by_type_filtered_{threshold_str}.csv",
        is_filtered=True
    )
    print("   ✅ batch_aggregated_unfiltered.csv")
    print(f"   ✅ batch_aggregated_filtered_{threshold_str}.csv")
    print("   ✅ cell_density_by_type_unfiltered.csv")
    print(f"   ✅ cell_density_by_type_filtered_{threshold_str}.csv")

    # Summary
    num_files = aggregated_filtered.get('num_files', 0)
    num_tiles = aggregated_filtered.get('num_tiles', num_files)
    total_cells_filtered = aggregated_filtered.get('total_cells', 0)
    tile_area_mm2 = aggregated_filtered.get('tile_area_mm2', TILE_AREA_MM2)
    total_area_mm2 = num_tiles * tile_area_mm2
    cells_per_tile = total_cells_filtered / num_tiles if num_tiles > 0 else 0
    avg_cell_density = total_cells_filtered / total_area_mm2 if total_area_mm2 > 0 else 0

    print("\n" + "=" * 80)
    print("Export Summary:")
    print("=" * 80)
    print(f"✅ Total files processed: {num_files}")
    print(f"✅ Total tiles: {num_tiles}")
    print(f"✅ Tile area: {tile_area_mm2} mm² per tile")
    print(f"✅ Total area: {total_area_mm2:.2f} mm²")
    print(f"✅ Total cells (unfiltered): {aggregated_unfiltered.get('total_cells', 0):,}")
    print(f"✅ Total cells (filtered): {total_cells_filtered:,}")
    print(f"✅ Total cells reclassified: {aggregated_filtered.get('reclassified_count', 0):,}")
    print(f"✅ Average cells per tile: {cells_per_tile:,.2f}")
    print(f"✅ Average cell density: {avg_cell_density:,.1f} cells/mm²")
    print("\n" + "=" * 80)
    print("✅ All results saved to", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
