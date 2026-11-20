"""Unknown cells (``protocol.allow_unknown_cells``): ``render`` may leave parts of the frame
unpredicted. An unknown part is skipped by every comparison with the real observation; the rest
of the frame, and every other observation field, must still match exactly.

The CWM decides what is unknown from its State alone: ``render`` never sees the observation. The
State is still carried by ``step``, so an unknown part that later shows in a known one fails there
as an ordinary mismatch."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

from regact.protocols.cwm.store import canonical

UNKNOWN = "UNKNOWN"  # in place of any value of the frame: that value is not predicted
# As the first element of the frame, when the frame is a list of frames (an animation): any
# number of leading frames, their count included, is not predicted.
UNKNOWN_FRAMES = "UNKNOWN_FRAMES"
# Share of the cells that ever change in the recordings that a CWM may leave unknown.
MAX_UNKNOWN_SHARE = 0.9

Unknown = list[tuple[str, Any]]  # (path in the observation, the observed value left unpredicted)


def resolve(rendered: Any, obs: dict[str, Any], allowed: bool) -> tuple[Any, Unknown]:
    """The prediction with its unknown parts filled from the observation, and those parts. What
    comes back is compared, and shown in differences, like a prediction without unknown parts."""
    if not allowed or not isinstance(rendered, dict) or "frame" not in rendered:
        return rendered, []
    unknown: Unknown = []
    frame = _fill(rendered["frame"], obs["frame"], "frame", unknown)
    return {**rendered, "frame": frame}, unknown


def _fill(predicted: Any, observed: Any, path: str, unknown: Unknown) -> Any:
    if isinstance(predicted, str) and predicted == UNKNOWN:
        unknown.append((path, observed))
        return observed
    if isinstance(predicted, dict) and isinstance(observed, dict):
        return {
            key: _fill(value, observed[key], f"{path}.{key}", unknown) if key in observed else value
            for key, value in predicted.items()
        }
    if isinstance(predicted, list) and isinstance(observed, list):
        skipped = 0
        if path == "frame" and predicted[:1] == [UNKNOWN_FRAMES]:
            skipped = len(observed) - len(predicted) + 1
            if skipped < 0:
                return predicted  # more known frames than were observed: a mismatch
            predicted = [*observed[:skipped], *predicted[1:]]
            unknown.extend((f"{path}[{i}]", observed[i]) for i in range(skipped))
        return [
            _fill(value, observed[i], f"{path}[{i}]", unknown)
            if skipped <= i < len(observed)
            else value
            for i, value in enumerate(predicted)
        ]
    return predicted


def unknown_bytes(unknown: Unknown) -> int:
    """Serialized bytes of the observation the CWM did not predict: they earn no State size."""
    return sum(len(canonical(value).encode()) for _, value in unknown)


def leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from leaves(value[key], f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from leaves(item, f"{path}[{i}]")
    else:
        yield path, value


class UnknownShare:
    """How much of what changes in the recordings a CWM left unknown. A cell is a leaf of the
    frame; only the cells whose value changes at least once in the checked observations count,
    on both sides: static background costs nothing to leave unknown and earns nothing known."""

    def __init__(self) -> None:
        self.cells: Counter[str] = Counter()

    def add(self, unknown: Unknown) -> None:
        for path, value in unknown:
            self.cells.update(leaf for leaf, _ in leaves(value, path))

    def report(self, frames: Iterable[Any]) -> dict[str, Any]:
        """``frames``: the frame of every checked observation."""
        first: dict[str, Any] = {}
        seen: Counter[str] = Counter()
        changing: set[str] = set()
        for frame in frames:
            for path, value in leaves(frame, "frame"):
                seen[path] += 1
                if first.setdefault(path, value) != value:
                    changing.add(path)
        total = sum(seen[path] for path in changing)
        unknown = sum(self.cells[path] for path in changing)
        return {
            "cells": sum(self.cells.values()),
            "changing_cells": unknown,
            "share_of_changing_cells": round(unknown / total, 4) if total else 0.0,
            "allowed_share": MAX_UNKNOWN_SHARE,
        }
