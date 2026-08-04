"""Competition entry point (Kaggle): argparse + a YAML profile -> RunConfig.

No Hydra here — a plain profile YAML (single_instance + offline ARC by default)
loaded into the typed config, with a few CLI flags for the notebook cell. Builds
the same :class:`RunConfig` as ``run_exp`` and runs the same experiment.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from regact.config.loader import run_config_from_mapping
from regact.config.schema import AgentName, RunConfig
from regact.orchestration.experiment import resolve_run_dir, run_experiment

_DEFAULT_PROFILE = Path(__file__).parent / "conf" / "experiment" / "competition.yaml"


def build_run_config_from_profile(profile_path: str) -> RunConfig:
    """Load a plain-YAML profile into a typed :class:`RunConfig`."""
    raw = OmegaConf.to_container(OmegaConf.load(profile_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"profile {profile_path!r} must be a mapping")
    return run_config_from_mapping(cast("dict[str, Any]", raw))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="regact.run_kaggle")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_PROFILE),
        help="Path to a profile YAML (default: the packaged competition profile).",
    )
    parser.add_argument("--games", nargs="*", default=None, help="Override the task list.")
    parser.add_argument("--parallel", type=int, default=None, help="Worker count (>1 => parallel).")
    parser.add_argument("--output-root", default=None, help="Where to write experiment outputs.")
    parser.add_argument(
        "--agent",
        default=None,
        help=(
            "Override the profile's agent (e.g. 'scripted' for a no-LLM plumbing smoke, "
            "or 'alan'). Keeps the profile's model/base_url/args."
        ),
    )
    return parser.parse_args(argv)


def run_kaggle(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = build_run_config_from_profile(args.config)
    if args.games is not None:
        config.problem.tasks = args.games
    if args.parallel is not None:
        config.parallel_workers = args.parallel
    if args.output_root is not None:
        config.output_root = args.output_root
    if args.agent is not None:
        config.agent.name = AgentName(args.agent)

    run_dir = resolve_run_dir(config)
    reasons = asyncio.run(run_experiment(config, output_root=run_dir))
    for task, reason in reasons.items():
        print(f"{task}: {reason}")

    _print_arc_summary(config, list(reasons), run_dir)
    return 0


def _print_arc_summary(config: RunConfig, tasks: list[str], run_dir: str) -> None:
    """Print the ARC-AGI-3 RHAE recap when the problem is arc_agi (no-op otherwise).

    Kept behind the problem-name check so this generic entrypoint never imports an
    ARC-specific module for a non-ARC run.
    """
    if config.problem.name != "arc_agi" or not tasks:
        return
    from regact.problems.arc_agi.scoring import summarize_run
    from regact.problems.arc_agi.tasks import discover_tasks

    env_dir = str(config.problem.kwargs.get("environments_dir") or "environnement")
    catalog = discover_tasks(env_dir)  # offline metadata carries the human baselines
    baselines = {t: (catalog[t].baseline_actions if t in catalog else None) for t in tasks}
    print("\n" + summarize_run(run_dir, tasks, baselines))


if __name__ == "__main__":
    raise SystemExit(run_kaggle())
