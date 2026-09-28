"""Compact, inclusive ID ranges shared by feedback and dataset queries."""

import re
from collections.abc import Iterable


def format_id_ranges(ids: Iterable[int]) -> str:
    """Format a set of IDs as '[1:4, 8, 10:12]'; both range endpoints are included."""
    values = list(ids)
    if any(type(i) is not int or i < 1 for i in values):
        raise ValueError("IDs must be positive integers")
    values = sorted(set(values))
    if not values:
        return "[]"
    parts: list[str] = []
    start = end = values[0]
    for identifier in values[1:]:
        if identifier == end + 1:
            end = identifier
        else:
            parts.append(str(start) if start == end else f"{start}:{end}")
            start = end = identifier
    parts.append(str(start) if start == end else f"{start}:{end}")
    return "[" + ", ".join(parts) + "]"


def expand_id_ranges(value: str, *, max_items: int | None) -> list[int]:
    """Parse inclusive ranges without allocating more than the query allowance."""
    if not isinstance(value, str) or (max_items is not None and len(value) > max_items * 50 + 2):
        raise ValueError("ID range string is too long")
    text = value.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError("Use ID notation such as [1:4, 8]; range endpoints are inclusive")
    if not text[1:-1].strip():
        return []
    result: list[int] = []
    for part in text[1:-1].split(","):
        match = re.fullmatch(r"\s*([1-9][0-9]*)(?:\s*:\s*([1-9][0-9]*))?\s*", part)
        if not match:
            raise ValueError("Invalid ID range; use [1:4, 8] with positive inclusive endpoints")
        start = int(match[1])
        end = int(match[2]) if match[2] else start
        if end < start:
            raise ValueError("ID ranges must be ascending")
        if max_items is not None and end - start + 1 > max_items - len(result):
            raise ValueError(f"query supports at most {max_items} items; use smaller ID ranges")
        result.extend(range(start, end + 1))
    return result
