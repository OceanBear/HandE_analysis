"""
Load HoVer-Net cell-type names and colors from type_info_4class.json.

JSON format (project root):
    {
        "0": ["Others",            [128, 128, 128]],
        "1": ["Tumor",             [0,   255,   0]],
        ...
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_TYPE_INFO_PATH = Path(__file__).resolve().parent.parent / "type_info_4class.json"


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(x) for x in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def load_cell_type_config(
    path: Path | str | None = None,
) -> Tuple[Dict[int, str], Dict[int, str], Path]:
    """
    Returns (cell_type_dict, cell_type_colors_hex, resolved_path).
    """
    resolved = Path(path) if path is not None else DEFAULT_TYPE_INFO_PATH
    with resolved.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    cell_type_dict: Dict[int, str] = {}
    cell_type_colors: Dict[int, str] = {}
    for key, value in raw.items():
        type_id = int(key)
        name, rgb = value
        cell_type_dict[type_id] = name
        cell_type_colors[type_id] = rgb_to_hex(rgb)

    return cell_type_dict, cell_type_colors, resolved


def cell_type_category_order(cell_type_dict: Dict[int, str]) -> list[str]:
    return [cell_type_dict[i] for i in sorted(cell_type_dict.keys())]
