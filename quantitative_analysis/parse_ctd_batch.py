"""
HoVer-Net Result Analysis - Cell Type Quantification (Batch)

This script processes all JSON files in a directory, aggregates results,
and produces three types of outputs, all per image as well as summarized,
and all with and without filtering based on input confidence value (e.g. 50):
- Batch aggregated: Cell counts, proportion, percentage, type probability
- Cell density by type: cells/tile and cells/mm²
- Confidence distribution by cell type: count of cells per confidence bin (10% increments)

Usage:
    python parse_ctd_batch.py --input-dir DIR --output-dir DIR [--limit N]

Required:
    --input-dir   Directory containing HoVer-Net JSON tile files.
    --output-dir  Directory to save output CSV files.

Optional:
    --limit       Only process the first N JSON files found (for quick testing).
"""

import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

# Reuse functions and constants from single-tile module
from cell_type_distribution_single import (
    analyze_single_json,
    apply_confidence_filter,
    configure_cell_types,
    display_results,
    export_results_to_csv,
    export_density_summary_to_csv,
    extract_confidence_by_type,
    export_confidence_distribution_to_csv,
    TILE_AREA_MM2,
)
from cell_type_utils import DEFAULT_TYPE_INFO_PATH

# --- Configuration ---
# Input directory and output directory are now required command-line arguments
# (see --input-dir and --output-dir below) instead of hardcoded paths.

# Threshold for confidence filtering
CONFIDENCE_THRESHOLD = 0.5


def extract_case_id(filename):
    """
    Derive the case ID from a tile filename by taking everything before
    '_tile_'. E.g. 'JN_TS_154_tile_5425_11991.json' -> 'JN_TS_154'.

    Falls back to the filename stem (extension stripped) if '_tile_' isn't
    present, so unexpected filename formats don't crash the run.
    """
    name = Path(filename).name
    match = re.match(r'^(.*?)_tile_', name)
    if match:
        return match.group(1)
    return Path(filename).stem


def find_json_files(directory_path, limit=None):
    """
    Find all JSON files in a directory.

    Parameters:
    -----------
    directory_path : str or Path
        Path to directory containing JSON files (or a single JSON file; parent dir will be used)
    limit : int, optional
        If set, stop after finding this many files. Skips the full directory
        sort, which avoids scanning every file on slow/network drives — useful
        for quick tests on large remote directories.

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

    if limit:
        # Take the first `limit` files as returned by the OS, without a full
        # directory listing + sort (much faster on network shares).
        json_files = []
        for f in directory.glob("*.json"):
            json_files.append(f)
            if len(json_files) >= limit:
                break
        json_files = sorted(json_files)
    else:
        json_files = sorted(directory.glob("*.json"))

    if not json_files:
        print(f"⚠️ No JSON files found in {directory}")
        return []

    print(f"✅ Found {len(json_files)} JSON files in {directory}" + (" (limited scan)" if limit else ""))
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


def aggregate_confidence_by_type(json_files, threshold=None):
    """
    Aggregate confidence (type probability) values across all JSON files.

    Args:
        json_files: List of Path objects to JSON files
        threshold (float, optional): If set, apply confidence filter per file

    Returns:
        tuple: (filename for display, dict mapping cell_type -> list of confidence values)
    """
    probs_by_type = defaultdict(list)
    for json_path in json_files:
        _, tile_probs = extract_confidence_by_type(json_path, threshold=threshold)
        for cell_type, probs in tile_probs.items():
            probs_by_type[cell_type].extend(probs)
    num_files = len(json_files)
    threshold_str = f"_threshold_{int(threshold * 100)}" if threshold is not None else ""
    filename = f"Aggregated_{num_files}_files{threshold_str}"
    return filename, dict(probs_by_type)


def build_confidence_detail_csv(json_files, threshold, output_dir, output_filename):
    """
    Build a per-file confidence-distribution CSV: one file's binned confidence
    values per row-group, tagged with Filename and Case, stacked across all
    json_files.

    Uses the existing (unmodified) export_confidence_distribution_to_csv from
    the single-tile module via a temporary CSV per file, since that function
    writes directly to a path rather than returning a DataFrame. Temp files
    are cleaned up after reading.
    """
    tmp_dir = output_dir / "_tmp_confidence_csv"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for json_file in json_files:
        try:
            fname, probs = extract_confidence_by_type(json_file, threshold=threshold)
        except Exception as e:
            print(f"❌ Error extracting confidence for {json_file.name}: {str(e)}")
            continue
        if not probs:
            continue

        tmp_path = tmp_dir / f"{json_file.stem}__conf_tmp.csv"
        export_confidence_distribution_to_csv(probs, tmp_path, fname)
        df = pd.read_csv(tmp_path)
        df.insert(1, 'Case', extract_case_id(json_file.name))
        frames.append(df)
        tmp_path.unlink()

    try:
        tmp_dir.rmdir()
    except OSError:
        pass  # not empty or already gone; harmless

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(output_dir / output_filename, index=False)
    return combined


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Batch cell-type distribution analysis for HoVer-Net JSON tiles."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of JSON files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--type-info",
        type=str,
        default=str(DEFAULT_TYPE_INFO_PATH),
        help="Path to type_info JSON (default: project root type_info_4class.json).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Reclassify low-confidence cells as Others (type 0).",
    )
    parser.add_argument(
        "--tile-area-mm2",
        type=float,
        default=TILE_AREA_MM2,
        help="Tile area in mm² for density calculations.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N JSON files found (for quick testing).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    configure_cell_types(args.type_info)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_dir)
    confidence_threshold = args.confidence_threshold
    tile_area_mm2 = args.tile_area_mm2
    if input_path.is_file():
        input_path = input_path.parent
    if not input_path.exists():
        print(f"❌ Error: Directory not found at {args.input_dir}")
        return

    print(f"📁 Input directory: {input_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Confidence threshold: {confidence_threshold}")
    if args.limit:
        print(f"🔎 Test mode: limiting to first {args.limit} files")
    print()

    json_files_all = find_json_files(input_path, limit=args.limit)

    # Analyze unfiltered
    # Keep (json_file, result) pairs together so we can derive Case/Filename
    # from the actual source file later, even if the result dict renames it
    # (e.g. filtered results append a "_threshold_NN" suffix).
    unfiltered_pairs = []
    for i, json_file in enumerate(json_files_all, 1):
        print(f"   Processing (unfiltered) {i}/{len(json_files_all)}: {json_file.name}")
        try:
            result = analyze_single_json(json_file, tile_area_mm2=tile_area_mm2)
            unfiltered_pairs.append((json_file, result))
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {str(e)}")
    all_results = [r for _, r in unfiltered_pairs]
    aggregated_unfiltered = aggregate_results(all_results, is_filtered=False)

    # Analyze filtered
    filtered_pairs = []
    for i, json_file in enumerate(json_files_all, 1):
        print(f"   Processing (filtered) {i}/{len(json_files_all)}: {json_file.name}")
        try:
            result = apply_confidence_filter(json_file, threshold=confidence_threshold, tile_area_mm2=tile_area_mm2)
            filtered_pairs.append((json_file, result))
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {str(e)}")
    filtered_results = [r for _, r in filtered_pairs]
    aggregated_filtered = aggregate_results(filtered_results, is_filtered=True)

    if not all_results and not filtered_results:
        print("⚠️ No results to process. Exiting.")
        return

    # Display results
    display_results(aggregated_filtered)
    display_results(aggregated_unfiltered)

    # Export CSV files
    threshold_str = str(int(confidence_threshold * 100))
    print("\n📁 Exporting CSV files...")

    # --- Per-file detail: cell counts/proportions/density/type-prob stats, one row per (file, cell type) ---
    def _detail_frame(pairs, export_fn, is_filtered):
        frames = []
        for json_file, result in pairs:
            df = export_fn(result, output_path=None, is_filtered=is_filtered)
            df.insert(1, 'Case', extract_case_id(json_file.name))
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    detail_counts_unfiltered = _detail_frame(unfiltered_pairs, export_results_to_csv, is_filtered=False)
    detail_counts_unfiltered.to_csv(output_dir / "batch_aggregated_unfiltered.csv", index=False)

    detail_counts_filtered = _detail_frame(filtered_pairs, export_results_to_csv, is_filtered=True)
    detail_counts_filtered.to_csv(output_dir / f"batch_aggregated_filtered_{threshold_str}.csv", index=False)

    # --- Per-file detail: cell density metrics, one row per (file, cell type) ---
    detail_density_unfiltered = _detail_frame(unfiltered_pairs, export_density_summary_to_csv, is_filtered=False)
    detail_density_unfiltered.to_csv(output_dir / "cell_density_by_type_unfiltered.csv", index=False)

    detail_density_filtered = _detail_frame(filtered_pairs, export_density_summary_to_csv, is_filtered=True)
    detail_density_filtered.to_csv(output_dir / f"cell_density_by_type_filtered_{threshold_str}.csv", index=False)

    # --- Per-file detail: confidence distribution (0-1 binned), one row per (file, cell type, bin) ---
    detail_confidence_unfiltered = build_confidence_detail_csv(
        json_files_all, threshold=None, output_dir=output_dir,
        output_filename="confidence_distribution_by_cell_type_unfiltered.csv"
    )
    detail_confidence_filtered = build_confidence_detail_csv(
        json_files_all, threshold=confidence_threshold, output_dir=output_dir,
        output_filename=f"confidence_distribution_by_cell_type_filtered_{threshold_str}.csv"
    )

    # --- Aggregated summary across all files (same as before, now with _summary suffix) ---
    export_results_to_csv(
        aggregated_unfiltered,
        output_dir / "batch_aggregated_unfiltered_summary.csv",
        is_filtered=False
    )
    export_results_to_csv(
        aggregated_filtered,
        output_dir / f"batch_aggregated_filtered_{threshold_str}_summary.csv",
        is_filtered=True
    )
    export_density_summary_to_csv(
        aggregated_unfiltered,
        output_dir / "cell_density_by_type_unfiltered_summary.csv",
        is_filtered=False
    )
    export_density_summary_to_csv(
        aggregated_filtered,
        output_dir / f"cell_density_by_type_filtered_{threshold_str}_summary.csv",
        is_filtered=True
    )
    if json_files_all:
        filename_unf, probs_unf = aggregate_confidence_by_type(json_files_all)
        if probs_unf:
            export_confidence_distribution_to_csv(
                probs_unf, output_dir / "confidence_distribution_by_cell_type_unfiltered_summary.csv", filename_unf
            )
        filename_filt, probs_filt = aggregate_confidence_by_type(json_files_all, threshold=confidence_threshold)
        if probs_filt:
            export_confidence_distribution_to_csv(
                probs_filt,
                output_dir / f"confidence_distribution_by_cell_type_filtered_{threshold_str}_summary.csv",
                filename_filt,
            )

    print("   ✅ batch_aggregated_unfiltered.csv (per file)")
    print(f"   ✅ batch_aggregated_filtered_{threshold_str}.csv (per file)")
    print("   ✅ cell_density_by_type_unfiltered.csv (per file)")
    print(f"   ✅ cell_density_by_type_filtered_{threshold_str}.csv (per file)")
    print("   ✅ confidence_distribution_by_cell_type_unfiltered.csv (per file)")
    print(f"   ✅ confidence_distribution_by_cell_type_filtered_{threshold_str}.csv (per file)")
    print("   ✅ batch_aggregated_unfiltered_summary.csv (all files combined)")
    print(f"   ✅ batch_aggregated_filtered_{threshold_str}_summary.csv (all files combined)")
    print("   ✅ cell_density_by_type_unfiltered_summary.csv (all files combined)")
    print(f"   ✅ cell_density_by_type_filtered_{threshold_str}_summary.csv (all files combined)")
    print("   ✅ confidence_distribution_by_cell_type_unfiltered_summary.csv (all files combined)")
    print(f"   ✅ confidence_distribution_by_cell_type_filtered_{threshold_str}_summary.csv (all files combined)")

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
