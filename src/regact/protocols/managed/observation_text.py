"""An observation as printed to an agent without the experience buffer.

JSON exactly as the controller receives it, one grid row per line. With ``frame_cells = "hex"``,
a grid of integers 0-15 is printed first in hex, and a placeholder in the JSON shows where it
sits in ``frame``. Only the last grid of a list of grids (an animation) is printed; everything
else decodes back exactly.
"""

from __future__ import annotations

import json
from typing import Any

HEX = "0123456789abcdef"


def _is_cell(value: Any, cells: str) -> bool:
    if cells == "list":
        return isinstance(value, list) and not any(isinstance(x, (list, dict)) for x in value)
    return not isinstance(value, (list, dict))


def _is_grid(value: Any, cells: str) -> bool:
    """A non-empty list of equal-length, non-empty rows of cells."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) and row and len(row) == len(value[0]) for row in value)
        and all(_is_cell(cell, cells) for row in value for cell in row)
    )


def _is_hex_grid(value: Any) -> bool:
    return _is_grid(value, "scalar") and all(
        type(cell) is int and 0 <= cell < 16 for row in value for cell in row
    )


def _depth(value: Any) -> int:
    return 1 + max((_depth(x) for x in value), default=0) if isinstance(value, list) else 0


def dump(value: Any, row_depth: int, indent: int = 0) -> str:
    """JSON with one dictionary key per line, and lists broken until their items are rows."""
    pad = "  " * indent
    if isinstance(value, dict) and value:
        items = [
            f"{pad}  {json.dumps(k)}: {dump(v, row_depth, indent + 1)}" for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list) and value and _depth(value) > row_depth:
        items = [pad + "  " + dump(v, row_depth, indent + 1) for v in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(value, separators=(",", ":"))


_SLOT = "\x00frame\x00"  # stands for frame in the JSON until the placeholder replaces it


def _hex_frame(frame: Any) -> tuple[str, list[list[int]]] | None:
    """The placeholder and the grid to print in hex, or None when the frame is not integer
    grids."""
    if _is_hex_grid(frame):
        return "<the grid above>", frame
    if (
        isinstance(frame, list)
        and frame
        and all(_is_hex_grid(g) and len(g) == len(frame[0]) for g in frame)
        and len({len(g[0]) for g in frame}) == 1
    ):
        hidden = len(frame) - 1
        before = f"<{hidden} grid{'s' * (hidden > 1)} not shown>, " if hidden else ""
        return f"[{before}<the grid above>]", frame[-1]
    return None


def observation_text(obs: dict[str, Any], cells: str) -> str:
    """``cells`` is the problem's ``frame_cells``: "hex", "scalar" or "list"."""
    row_depth = 2 if cells == "list" else 1
    hexed = _hex_frame(obs.get("frame")) if cells == "hex" else None
    if hexed is None:
        return dump(obs, row_depth)
    placeholder, grid = hexed
    rows = ["".join(HEX[cell] for cell in row) for row in grid]
    text = dump({**obs, "frame": _SLOT}, row_depth).replace(json.dumps(_SLOT), placeholder)
    return "\n".join([*rows, text])
