"""Analytics over a game's turns + submissions (the quantitative proxies).

Language/reasoning quality is not auto-scorable cheaply; these are the proxies a
human analyst reads alongside the conversation: cost (tokens), effort (turns,
tool histogram), progress (score per submission), reasoning volume (thinking).
"""

from __future__ import annotations

from typing import Any

from regact.viz.reader import GameView


def game_metrics(game: GameView) -> dict[str, Any]:
    """A flat dict of proxies for one game (used by the dashboard + overview)."""
    tokens = _token_totals(game)
    tools = _tool_histogram(game)
    submissions = _submission_trajectory(game)
    best, final = _levels(game)
    return {
        "n_turns": len(game.turns),
        "n_tool_calls": sum(tools.values()),
        "tool_histogram": tools,
        "n_submissions": sum(1 for s in game.submissions if s.name.isdigit()),
        "tokens": tokens,
        "thinking_chars": sum(len(t) for turn in game.turns for t in turn.thinkings),
        "text_chars": sum(len(t) for turn in game.turns for t in turn.texts),
        "submission_trajectory": submissions,
        "best_levels": best,
        "final_levels": final,
        "total_levels": game.state.get("win_levels"),  # from the first obs; shown from the start
        "duration_s": game.state.get("duration_s", 0),
        "success_rate": _final_metric(game, "success_rate"),
        "last_error_category": game.state.get("last_error_category"),
        "exit_requested": game.state.get("exit_requested"),
        "cheat_attempts": game.state.get("cheat_attempts", 0),
    }


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
    """Score per numbered submission, in order — to see whether the agent improved."""
    out: list[dict[str, Any]] = []
    for sub in game.submissions:
        if not sub.name.isdigit():
            continue
        agg = sub.aggregate
        out.append(
            {
                "submission": int(sub.name),
                "success_rate": agg.get("success_rate"),
                "levels": agg.get("mean_levels_completed"),
                "mean_steps": agg.get("mean_steps"),
                "error": sub.error,
            }
        )
    return out


def _levels(game: GameView) -> tuple[float | None, float | None]:
    """Best mean_levels_completed across submissions, and the 'final' one."""
    values: list[float] = [
        float(s.aggregate["mean_levels_completed"])
        for s in game.submissions
        if s.aggregate.get("mean_levels_completed") is not None
    ]
    best = max(values) if values else None
    final = next(
        (s.aggregate.get("mean_levels_completed") for s in game.submissions if s.name == "final"),
        None,
    )
    return best, final


def _final_metric(game: GameView, key: str) -> Any:
    """The metric from the 'final' submission, else the last numbered one."""
    final = next((s for s in game.submissions if s.name == "final"), None)
    if final is not None:
        return final.aggregate.get(key)
    numbered = [s for s in game.submissions if s.name.isdigit()]
    return numbered[-1].aggregate.get(key) if numbered else None
