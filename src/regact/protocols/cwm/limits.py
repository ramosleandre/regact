"""Small helpers for optional CWM budgets. None always means no configured cap."""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterable


def reached(value: int | float, cap: int | float | None) -> bool:
    return cap is not None and value >= cap


def deadline(seconds: float | None, *, start: float | None = None) -> float:
    return (
        float("inf")
        if seconds is None
        else (time.monotonic() if start is None else start) + seconds
    )


def action_indices(cap: int | None) -> Iterable[int]:
    return itertools.count() if cap is None else range(cap)


def describe(cap: int | float | None) -> str:
    return "unlimited" if cap is None else f"{cap:g}"


def truncate_error(text: str, cap: int | None) -> str:
    """Preserve the error's context and final cause, within the character budget."""
    if cap is None or len(text) <= cap:
        return text
    marker = " ... [truncated] ... " if cap >= 24 else "…"
    remaining = cap - len(marker)
    head = (remaining + 1) // 2
    tail = remaining // 2
    return text[:head] + marker + (text[-tail:] if tail else "")
