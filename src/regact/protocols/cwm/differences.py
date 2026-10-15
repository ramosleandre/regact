"""Predicted-versus-observed differences as the agent reads them (docs/reading_differences.md).

Every top-level field that differs is listed with its own budget. Any 2-D grid inside a field
is summarized as regions of touching differing cells; other values are listed by path.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from typing import Any

from regact.protocols.cwm import limits as budgets
from regact.protocols.cwm.config import FeedbackConfig

# A grid cell may be a short value list (RGB, RGBA). Shorter than both grid sides, so that a
# small stack of layers is not read as one image of channels.
_MAX_CELL_CHANNELS = 4
_MAX_VALUE_CHARS = 120
_MAX_EXACT_RECTANGLES = 3  # a region covered by this many rectangles is written exactly


def summarize(predicted: Any, observed: Any, config: FeedbackConfig) -> dict[str, Any]:
    """``{field: [difference, ...]}`` for every top-level field that differs."""
    if not (isinstance(predicted, dict) and isinstance(observed, dict)):
        return {"observation": _Field(config).visit(predicted, observed, "observation")}
    out = {}
    for key in sorted(set(predicted) | set(observed)):
        p, o = predicted.get(key, "<missing>"), observed.get(key, "<missing>")
        if _same(p, o):
            continue
        out[key] = _Field(config).visit(p, o, key)
    return out


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _scalar(x: Any) -> bool:
    return x is None or isinstance(x, (bool, int, float, str))


def _grid_shape(g: Any) -> tuple[int, int] | None:
    """(rows, columns) of equal-length rows whose cells are all scalars or all short lists."""
    if not isinstance(g, list) or not g or not all(isinstance(r, list) and r for r in g):
        return None
    if len({len(r) for r in g}) != 1:
        return None
    rows, cols = len(g), len(g[0])
    cells = [c for r in g for c in r]
    if not all(map(_scalar, cells)) and not all(
        isinstance(c, list)
        and 0 < len(c) <= _MAX_CELL_CHANNELS
        and len(c) < min(rows, cols)
        and all(map(_scalar, c))
        for c in cells
    ):
        return None
    return rows, cols


def _shape_text(g: list) -> str:
    if g and all(isinstance(r, list) for r in g):
        lengths = sorted({len(r) for r in g})
        if len(lengths) == 1:
            return f"{len(g)} rows of {lengths[0]}"
        return f"{len(g)} rows of {lengths[0]} to {lengths[-1]}"
    return f"list of {len(g)}"


def _text(x: Any) -> str:
    return json.dumps(x, separators=(", ", ": ")) if isinstance(x, list) else str(x)


def _value(x: Any) -> Any:
    text = json.dumps(x)
    return x if len(text) <= _MAX_VALUE_CHARS else text[: _MAX_VALUE_CHARS - 3] + "..."


def _span(a: int, b: int) -> str:
    return str(a) if b == a + 1 else f"{a}:{b}"


class _Field:
    """One top-level field's differences, under its own budgets."""

    def __init__(self, config: FeedbackConfig) -> None:
        self.config = config
        self.grids: list[dict[str, Any]] = []
        self.values: list[dict[str, Any]] = []
        self.omitted = 0

    def visit(self, a: Any, b: Any, path: str) -> list[dict[str, Any]]:
        self._visit(a, b, path)
        if self.omitted:
            self.values.append({"other differences": self.omitted})
        return self.grids + self.values

    def _visit(self, a: Any, b: Any, path: str) -> None:
        if _same(a, b):
            return
        sa, sb = _grid_shape(a), _grid_shape(b)
        if sa and sb:
            if sa == sb:
                self.grids.append(_grid(path, a, b, self.config))
            else:
                self.grids.append(
                    {"where": path, "shape": {"predicted": list(sa), "observed": list(sb)}}
                )
        elif (sa or sb) and isinstance(a, list) and isinstance(b, list):
            self.grids.append(
                {"where": path, "shape": {"predicted": _shape_text(a), "observed": _shape_text(b)}}
            )
        elif isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                self._visit(a.get(key, "<missing>"), b.get(key, "<missing>"), f"{path}.{key}")
        elif isinstance(a, list) and isinstance(b, list) and not all(map(_scalar, a + b)):
            if len(a) != len(b):
                self._value(f"len({path})", len(a), len(b))
            for i, (x, y) in enumerate(zip(a, b, strict=False)):
                self._visit(x, y, f"{path}[{i}]")
        else:
            self._value(path, a, b)

    def _value(self, path: str, a: Any, b: Any) -> None:
        if budgets.reached(len(self.values), self.config.max_diff_items):
            self.omitted += 1
        else:
            self.values.append({"where": path, "predicted": _value(a), "observed": _value(b)})


def _grid(path: str, p: list, o: list, config: FeedbackConfig) -> dict[str, Any]:
    cells = {(y, x) for y in range(len(p)) for x in range(len(p[0])) if p[y][x] != o[y][x]}
    regions = sorted(_components(cells), key=lambda c: (-len(c), min(c)))
    shown = regions
    if config.max_diff_regions is not None:
        shown = regions[: config.max_diff_regions]
    out: dict[str, Any] = {
        "where": path,
        "cells_differing": len(cells),
        "cells_total": len(p) * len(p[0]),
        "regions": [_region(path, region, p, o, config) for region in shown],
    }
    if len(regions) > len(shown):
        rest = regions[len(shown) :]
        out["other_regions"] = f"{len(rest)} more regions, {sum(map(len, rest))} cells"
    return out


def _components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Groups of cells touching by a side or a corner."""
    left, groups = set(cells), []
    while left:
        stack = [left.pop()]
        group = set(stack)
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (y + dy, x + dx) in left:
                        left.remove((y + dy, x + dx))
                        group.add((y + dy, x + dx))
                        stack.append((y + dy, x + dx))
        groups.append(group)
    return groups


def _rectangles(cells: set[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """Greedy row-major cover by rectangles ``(row0, row1, col0, col1)``, ends exclusive."""
    left, rects = set(cells), []
    while left:
        r, c = min(left)
        c2 = c
        while (r, c2 + 1) in left:
            c2 += 1
        r2 = r
        while all((r2 + 1, x) in left for x in range(c, c2 + 1)):
            r2 += 1
        left -= {(y, x) for y in range(r, r2 + 1) for x in range(c, c2 + 1)}
        rects.append((r, r2 + 1, c, c2 + 1))
    return rects


def _region(
    path: str, cells: set[tuple[int, int]], p: list, o: list, config: FeedbackConfig
) -> dict[str, Any]:
    rects = _rectangles(cells)
    r0, r1 = min(y for y, _ in cells), max(y for y, _ in cells) + 1
    c0, c1 = min(x for _, x in cells), max(x for _, x in cells) + 1
    box = f"{path}[{_span(r0, r1)}, {_span(c0, c1)}]"
    exact = len(rects) <= _MAX_EXACT_RECTANGLES
    out: dict[str, Any] = {
        "where": (
            " + ".join(f"{path}[{_span(a, b)}, {_span(c, d)}]" for a, b, c, d in rects)
            if exact
            else f"within {box}"
        ),
        "cells": len(cells),
    }
    changes = Counter(f"{_text(p[y][x])} -> {_text(o[y][x])}" for y, x in sorted(cells))
    small = not budgets.reached((r1 - r0) * (c1 - c0) - 1, config.max_crop_cells)
    if (len(changes) > 1 or not exact) and small:
        if len(rects) > 1:
            out["crop"] = box
        width = max(
            len(_text(g[y][x])) for g in (p, o) for y in range(r0, r1) for x in range(c0, c1)
        )

        def row(g: list, y: int) -> str:
            return " ".join(_text(g[y][x]).rjust(width) for x in range(c0, c1))

        out["predicted | observed"] = [f"{row(p, y)} | {row(o, y)}" for y in range(r0, r1)]
        return out
    top = changes.most_common(config.max_diff_changes)
    out["changes"] = dict(top)
    if len(changes) > len(top):
        out["changes"]["other changes"] = len(cells) - sum(n for _, n in top)
    return out


_STATE_CHARS = 2000  # describe_diagnostic prints each State up to this many characters
_BULKY = ("predicted", "observed", "state_before", "state", "first_output", "second_output")
_OVERRIDES = ("max_diff_items", "max_diff_regions", "max_diff_changes", "max_crop_cells")


def describe(
    record: dict[str, Any], diagnostic_id: int, config: FeedbackConfig, options: dict[str, Any]
) -> dict[str, Any]:
    """A diagnostic as the agent reads it: bounded differences and States, no full frames."""
    unknown = set(options) - {*_OVERRIDES, "max_state_chars"}
    if unknown:
        raise ValueError(
            f"unknown options {sorted(unknown)}; use {[*_OVERRIDES, 'max_state_chars']}"
        )
    for key, value in options.items():
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be a positive integer or None")
    budgets_ = dataclasses.replace(config, **{k: options[k] for k in _OVERRIDES if k in options})
    out = {k: v for k, v in record.items() if k not in (*_BULKY, "differences")}
    p, o = record.get("predicted"), record.get("observed")
    if isinstance(p, dict) and isinstance(o, dict):
        out["differences"] = summarize(p, o, budgets_)
    limit = options.get("max_state_chars", _STATE_CHARS)
    for key in _BULKY[2:]:
        if key in record:
            out[key] = _bounded(_readable(record[key]), limit, diagnostic_id)
    return out


def _readable(value: Any) -> Any:
    """A State as its fields: instances become {"ClassName": fields}, tuples lists."""
    if isinstance(value, list):
        return [_readable(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "@tuple" in value:
        return [_readable(v) for v in value["@tuple"]]
    if "@class" in value:
        return {value["@class"].rsplit(":", 1)[-1]: _readable(value["fields"])}
    return {k: _readable(v) for k, v in value.items()}


def _bounded(value: Any, limit: int | None, diagnostic_id: int) -> Any:
    text = json.dumps(value, separators=(", ", ": "))
    if limit is None or len(text) <= limit:
        return value
    return (
        text[:limit] + f"... ({len(text)} characters; "
        f"data_api.load_diagnostic({diagnostic_id}) returns it whole)"
    )
