"""Standalone, sandboxable eval runner.

Run as ``python -m regact.controllers.eval_runner`` from the agent's workdir: it
connects to the env via the workdir's ``framework.make_env`` (URL/game id baked in),
drives the submitted controller for each episode, and writes the **raw** per-episode
outcomes to ``--output``. It computes no score — the trusted orchestrator does that
(:func:`regact.controllers.executor.score_episodes`), so a controller cannot fake its
own metrics. The caller wraps this argv with the OS sandbox to confine untrusted code.
"""

from __future__ import annotations

import argparse
import json

from regact.config.schema import Lifecycle
from regact.controllers.executor import run_episodes_raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regact.controllers.eval_runner")
    parser.add_argument("--solution", required=True)
    parser.add_argument("--lifecycle", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-moves", type=int, default=400)
    parser.add_argument("--output", required=True)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args(argv)

    # The workdir is the cwd: its framework.make_env carries the baked env URL + game id.
    from framework.make_env import make_env  # type: ignore[import-not-found]

    try:
        episodes = run_episodes_raw(
            make_env(),
            args.solution,
            lifecycle=Lifecycle(args.lifecycle),
            n_episodes=args.episodes,
            max_moves=args.max_moves,
            record_video=args.record_video,
        )
        payload: dict[str, object] = {"episodes": episodes}
    except Exception as exc:  # could not load solution.py — report it for scoring
        payload = {"load_error": f"{type(exc).__name__}: {exc}"}

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
