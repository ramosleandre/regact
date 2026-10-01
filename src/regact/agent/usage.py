"""Token usage of a CLI agent run, harvested from the CLI's own session logs.

Claude Code and Codex run in a fresh per-task home that is deleted at teardown, and
with it the only complete record of what the run consumed. These readers summarize
that record before deletion, so every run keeps its exact token counts (per model)
and, for Codex, the account rate-limit readings the CLI saw at the run's start and end.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

_CLAUDE_FIELDS = {
    "input_tokens": "input",
    "output_tokens": "output",
    "cache_creation_input_tokens": "cache_write",
    "cache_read_input_tokens": "cache_read",
}
# Codex's input_tokens INCLUDES its cached_input_tokens (Claude reports the two disjointly).
_CODEX_FIELDS = {
    "input_tokens": "input",
    "cached_input_tokens": "cache_read",
    "output_tokens": "output",
    "reasoning_output_tokens": "reasoning_output",
}


def _records(paths: list[str]) -> Iterator[dict[str, Any]]:
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def claude_usage(config_dir: str) -> dict[str, Any] | None:
    """Per-model token totals of every API response logged under ``config_dir``."""
    paths = sorted(glob.glob(os.path.join(config_dir, "projects", "**", "*.jsonl"), recursive=True))
    per_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen: set[tuple[Any, Any]] = set()
    responses = 0
    for record in _records(paths):
        message = record.get("message")
        usage = message.get("usage") if isinstance(message, dict) else None
        if not isinstance(usage, dict):
            continue
        # One response is logged once per content block, each line repeating its usage.
        key = (message.get("id"), record.get("requestId"))
        if key != (None, None) and key in seen:
            continue
        seen.add(key)
        model = str(message.get("model") or "unknown")
        if model == "<synthetic>":
            continue
        responses += 1
        for source, name in _CLAUDE_FIELDS.items():
            per_model[model][name] += int(usage.get(source) or 0)
    if not responses:
        return None
    return {
        "source": "claude_code_session_logs",
        "responses": responses,
        "models": {model: dict(counts) for model, counts in per_model.items()},
    }


def _limit_reading(limits: dict[str, Any], timestamp: Any) -> dict[str, Any]:
    windows = {}
    for slot in ("primary", "secondary"):
        window = limits.get(slot)
        if isinstance(window, dict) and window.get("used_percent") is not None:
            windows[slot] = {
                "used_percent": window.get("used_percent"),
                "window_minutes": window.get("window_minutes"),
                "resets_at": window.get("resets_at"),
            }
    return {"timestamp": timestamp, **windows}


def codex_usage(codex_home: str) -> dict[str, Any] | None:
    """Token totals and first/last rate-limit readings from the rollouts under ``codex_home``."""
    pattern = os.path.join(codex_home, "sessions", "**", "rollout-*.jsonl")
    totals: dict[str, int] = defaultdict(int)
    readings: list[dict[str, Any]] = []
    model = None
    for path in sorted(glob.glob(pattern, recursive=True)):
        last_total: dict[str, Any] | None = None
        for record in _records([path]):
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            if record.get("type") == "turn_context" and payload.get("model"):
                model = payload.get("model")
            if payload.get("type") != "token_count":
                continue
            info = payload.get("info")
            if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                last_total = info["total_token_usage"]  # cumulative within one rollout
            limits = payload.get("rate_limits")
            if isinstance(limits, dict):
                readings.append(_limit_reading(limits, record.get("timestamp")))
        for source, name in _CODEX_FIELDS.items():
            totals[name] += int((last_total or {}).get(source) or 0)
    if not any(totals.values()) and not readings:
        return None
    usage: dict[str, Any] = {
        "source": "codex_rollouts",
        "models": {str(model or "unknown"): dict(totals)},
    }
    if readings:
        usage["rate_limits_start"] = readings[0]
        usage["rate_limits_end"] = readings[-1]
    return usage
