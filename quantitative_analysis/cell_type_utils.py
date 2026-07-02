"""
Shared cell-type helpers for quantitative_analysis scripts.

Defaults to project-root type_info_4class.json via neighborhood_composition/cell_type_config.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_NEIGHBORHOOD = Path(__file__).resolve().parent.parent / "neighborhood_composition"
if str(_NEIGHBORHOOD) not in sys.path:
    sys.path.insert(0, str(_NEIGHBORHOOD))

from cell_type_config import (  # noqa: E402
    DEFAULT_TYPE_INFO_PATH,
    cell_type_category_order,
    load_cell_type_config,
)


def resolve_cell_type_config(
    type_info_path: Path | str | None = None,
) -> Tuple[Dict[int, str], Dict[int, str], List[int], Path]:
    """Return (id->name, id->hex_color, sorted_ids, resolved_path)."""
    cell_type_dict, cell_type_colors, resolved = load_cell_type_config(type_info_path)
    ordered_names = cell_type_category_order(cell_type_dict)
    name_to_id = {name: tid for tid, name in cell_type_dict.items()}
    ordered_ids = [name_to_id[name] for name in ordered_names]
    return cell_type_dict, cell_type_colors, ordered_ids, resolved


def parse_cell_type_id(
    raw,
    cell_type_dict: Dict[int, str],
    *,
    context: str = "",
) -> int:
    """Cast JSON ``type`` to int and validate against the loaded config."""
    try:
        type_id = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid cell type id {raw!r}{context}") from exc
    if type_id not in cell_type_dict:
        expected = sorted(cell_type_dict.keys())
        raise KeyError(
            f"Unknown cell type id {type_id}{context}. Expected ids: {expected}"
        )
    return type_id


def empty_type_counts(cell_type_ids: List[int]) -> Dict[int, int]:
    return {type_id: 0 for type_id in cell_type_ids}


def load_tile_proportions(
    json_path: Path | str,
    cell_type_ids: List[int],
    cell_type_dict: Dict[int, str],
    min_prob: Optional[float] = None,
) -> np.ndarray:
    """Return a proportion vector (one entry per cell_type_ids order) for one tile JSON."""
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    counts = empty_type_counts(cell_type_ids)
    for nuc in data.get("nuc", {}).values():
        if min_prob is not None and nuc.get("type_prob", 1.0) < min_prob:
            continue
        type_id = parse_cell_type_id(
            nuc.get("type", 0),
            cell_type_dict,
            context=f" in {json_path}",
        )
        counts[type_id] += 1

    total = sum(counts.values())
    if total == 0:
        return np.zeros(len(cell_type_ids))

    return np.array([counts[type_id] / total for type_id in cell_type_ids])
