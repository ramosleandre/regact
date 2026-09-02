"""Analytics over a game's turns + submissions (the quantitative proxies).

Language/reasoning quality is not auto-scorable cheaply; these are the proxies a
human analyst reads alongside the conversation: cost (tokens), effort (turns,
tool histogram), progress (score per submission), reasoning volume (thinking).
"""

from __future__ import annotations

import json
from typing import Any

from regact.viz.reader import GameView


def game_metrics(game: GameView) -> dict[str, Any]:
    """A flat dict of proxies for one game (used by the dashboard + overview).

    Game-specific scores (ARC's levels, MiniGrid's reward, …) are NOT named here:
    the whole final aggregate is passed through opaquely as ``final_aggregate`` and
    the per-submission trajectory carries every numeric key each game reports, so a
    new game needs no viz change (see :func:`_submission_trajectory`).
    """
    tokens = _token_totals(game)
    tools = _tool_histogram(game)
    submissions = _submission_trajectory(game)
    flagged = _flagged_calls(game)  # re-derived from the transcript with the CURRENT policy
    return {
        "n_turns": len(game.turns),
        "n_tool_calls": sum(tools.values()),
        "tool_histogram": tools,
        "n_submissions": sum(1 for s in game.submissions if s.name.isdigit()),
        "tokens": tokens,
        "thinking_chars": sum(len(t) for turn in game.turns for t in turn.thinkings),
        "text_chars": sum(len(t) for turn in game.turns for t in turn.texts),
        "submission_trajectory": submissions,
        # The final submission's whole metric dict, opaque — each game owns its keys.
        "final_aggregate": _final_aggregate(game),
        # The same submission's un-replayed (direct) score, when shadow-replay also ran; empty
        # otherwise. A gap vs final_aggregate flags cheating OR a replay/scoring bug.
        "final_aggregate_unverified": _final_aggregate_unverified(game),
        # The reported submission's per-feature metrics, opaque + game/feature-agnostic (each
        # feature owns its keys), e.g. {"cwm": {"n_conflicting_transitions": 0, ...}}.
        "feature_metrics": _final_features(game),
        # The reported submission's problem-derived metrics (ARC rhae/lrhae) - secondary to the
        # game score, shown under "Other" in the viz.
        "derived_metrics": _final_derived(game),
        "duration_s": game.state.get("duration_s", 0),
        # Cumulative real env.step calls this task made (exploration + eval rollouts). None for runs
        # recorded before it was tracked, so the viz shows "-" rather than a wrong 0.
        "env_moves": game.state.get("env_moves"),
        "success_rate": _final_metric(game, "success_rate"),
        "last_error_category": game.state.get("last_error_category"),
        "exit_reason": game.state.get("exit_reason"),  # None while running
        "exit_requested": game.state.get("exit_requested"),
        # Re-derived count (current policy), so the KPI matches the conversation/panel — not the
        # live state count, which may be stale (an older, noisier policy version). A keyword
        # match is a lower bound on attempts, never a cheat rate.
        "flagged_tool_calls": len(flagged),
        "flagged_calls": flagged,
    }


def _flagged_calls(game: GameView) -> list[dict[str, Any]]:
    """Each flagged tool call: the tool, a short arg preview, and why it matched."""
    out: list[dict[str, Any]] = []
    for i, turn in enumerate(game.turns):
        for call in turn.tools:
            if call.tag == "cheat":
                out.append(
                    {
                        "turn": i,
                        "tool": call.name,
                        "args": json.dumps(call.input)[:200],
                        "flags": call.flags,
                    }
                )
    return out


def _token_totals(game: GameView) -> dict[str, int]:
    out = {"output": 0, "input": 0, "cache_read": 0}
    for turn in game.turns:
        usage = turn.usage or {}
        out["output"] += int(usage.get("output_tokens", 0) or 0)
        out["input"] += int(usage.get("input_tokens", 0) or 0)
        out["cache_read"] += int(usage.get("cache_read_input_tokens", 0) or 0)
    return out


def _tool_histogram(game: GameView) -> dict[str, int]:
    hist: dict[str, int] = {}
    for turn in game.turns:
        for call in turn.tools:
            hist[call.name] = hist.get(call.name, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: kv[1], reverse=True))


def _submission_trajectory(game: GameView) -> list[dict[str, Any]]:
    """Per numbered submission, in order — to see whether the agent improved.

    Game-agnostic: each row carries the submission's whole numeric aggregate under
    ``metrics`` (every key the game reports), so the dashboard can plot whichever
    keys a given game emits without this layer knowing their names.
    """
    out: list[dict[str, Any]] = []
    for sub in game.submissions:
        if not sub.name.isdigit():
            continue
        metrics = {
            k: v
            for k, v in sub.aggregate.items()
            if isinstance(v, (int, float)) and k != "n_episodes"
        }
        out.append(
            {
                "submission": int(sub.name),
                "metrics": metrics,
                "error": sub.error,
            }
        )
    return out


def _scored_submission(game: GameView) -> Any:
    """The submission whose score we report: 'final' if it actually scored, else the last
    numbered submission that has a non-empty aggregate. So an errored/empty 'final' (e.g. a
    teardown ReadTimeout) falls back to the last real score instead of rendering a blank."""
    final = next((s for s in game.submissions if s.name == "final"), None)
    if final is not None and final.aggregate:
        return final
    numbered = [s for s in game.submissions if s.name.isdigit() and s.aggregate]
    return numbered[-1] if numbered else final


def _final_aggregate(game: GameView) -> dict[str, Any]:
    """The reported submission's whole metric dict (verified score on a shadow run), opaque."""
    sub = _scored_submission(game)
    return dict(sub.aggregate) if sub else {}


def _final_aggregate_unverified(game: GameView) -> dict[str, Any]:
    """The reported submission's un-replayed (direct) score, when shadow-replay also ran."""
    sub = _scored_submission(game)
    return dict(sub.aggregate_unverified) if sub and sub.aggregate_unverified else {}


def _final_features(game: GameView) -> dict[str, Any]:
    """The reported submission's per-feature metrics (e.g. CWM data-integrity), opaque."""
    sub = _scored_submission(game)
    return dict(sub.features) if sub and sub.features else {}


def _final_derived(game: GameView) -> dict[str, Any]:
    """The reported submission's problem-derived metrics (ARC rhae/lrhae), opaque."""
    sub = _scored_submission(game)
    return dict(sub.derived) if sub and sub.derived else {}


def _final_metric(game: GameView, key: str) -> Any:
    """One metric from the reported submission (verified score on a shadow run)."""
    sub = _scored_submission(game)
    return sub.aggregate.get(key) if sub else None
